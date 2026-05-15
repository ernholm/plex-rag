from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import sqlite3
import json
import os
from pathlib import Path
from typing import List, Optional

# Initialize FastAPI app
app = FastAPI(title="Plex RAG Search")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Database setup
DB_PATH = "./plex_embeddings.db"

def init_db():
    """Initialize SQLite database for embeddings"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS embeddings (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            type TEXT NOT NULL,
            description TEXT,
            plex_key TEXT,
            poster_url TEXT,
            embedding BLOB NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    import math
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    mag1 = math.sqrt(sum(a * a for a in vec1))
    mag2 = math.sqrt(sum(b * b for b in vec2))
    if mag1 == 0 or mag2 == 0:
        return 0
    return dot_product / (mag1 * mag2)

def get_all_metadata(conn):
    """Get all unique actors and genres from database"""
    c = conn.cursor()

    # Get all actors
    c.execute('SELECT DISTINCT actors FROM embeddings WHERE actors IS NOT NULL')
    all_actors = set()
    for row in c.fetchall():
        try:
            actors = json.loads(row[0])
            all_actors.update(actors)
        except:
            pass

    # Get all genres
    c.execute('SELECT DISTINCT genres FROM embeddings WHERE genres IS NOT NULL')
    all_genres = set()
    for row in c.fetchall():
        try:
            genres = json.loads(row[0])
            all_genres.update(genres)
        except:
            pass

    return all_actors, all_genres

def extract_metadata_filters(query, all_actors, all_genres):
    """Extract actor names and genres from query"""
    query_lower = query.lower()
    matched_actors = []
    matched_genres = []

    # Check for actor names (case-insensitive, exact and partial matches)
    for actor in all_actors:
        actor_lower = actor.lower()
        # Exact match or word boundary match
        if actor_lower in query_lower:
            # Check it's not a substring of a larger word
            query_parts = query_lower.split()
            actor_parts = actor_lower.split()
            # If single word actor, check if any query word starts with it
            if len(actor_parts) == 1:
                if any(part.startswith(actor_lower) or actor_lower in part for part in query_parts):
                    matched_actors.append(actor)
            else:
                # Multi-word actor name
                if actor_lower in query_lower:
                    matched_actors.append(actor)

    # Check for genre names (case-insensitive)
    for genre in all_genres:
        if genre.lower() in query_lower:
            matched_genres.append(genre)

    # Log what we found for debugging
    if matched_actors or matched_genres:
        import logging
        logging.getLogger(__name__).debug(f"Query '{query}' -> Actors: {matched_actors}, Genres: {matched_genres}")

    return matched_actors, matched_genres

init_db()

# Models
class SearchQuery(BaseModel):
    query: str
    limit: int = 12
    min_relevance: float = 0.30
    strict_filter: bool = False

class SearchResult(BaseModel):
    title: str
    type: str
    description: str
    relevance: float
    plex_key: str
    poster_url: str
    rating: Optional[float] = None
    actors: List[str] = []
    year: Optional[int] = None
    duration: Optional[int] = None
    director: Optional[str] = None
    genres: List[str] = []
    resolution: Optional[str] = None

class IndexStatusResponse(BaseModel):
    total_items: int
    status: str

# Routes
@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/search", response_model=list[SearchResult])
async def search(query_data: SearchQuery):
    """Search the Plex library using semantic search"""
    try:
        # Embed the query
        query_embedding = embedder.encode(query_data.query)

        # Get all embeddings from database
        conn = sqlite3.connect(DB_PATH)

        # Extract metadata filters from query
        all_actors, all_genres = get_all_metadata(conn)
        matched_actors, matched_genres = extract_metadata_filters(query_data.query, all_actors, all_genres)

        c = conn.cursor()
        c.execute('SELECT id, title, type, description, plex_key, poster_url, rating, actors, year, duration, director, genres, resolution, embedding FROM embeddings')
        rows = c.fetchall()
        conn.close()

        if not rows:
            return []

        # Calculate similarity for all rows
        results = []
        for row in rows:
            try:
                doc_id, title, doc_type, description, plex_key, poster_url, rating, actors_json, year, duration, director, genres_json, resolution, embedding_blob = row

                # Deserialize embedding, actors, and genres
                embedding = json.loads(embedding_blob)
                actors = json.loads(actors_json) if actors_json else []
                genres = json.loads(genres_json) if genres_json else []

                # Calculate similarity
                similarity = cosine_similarity(query_embedding, embedding)

                # Boost score if query terms appear in title (case-insensitive)
                query_lower = query_data.query.lower()
                title_lower = title.lower()
                if len(query_lower) > 2:  # Only for meaningful queries
                    # Check if significant portions of query appear in title
                    query_words = query_lower.split()
                    matched_words = sum(1 for word in query_words if len(word) > 2 and word in title_lower)
                    if matched_words > 0:
                        # Boost based on match ratio
                        match_ratio = matched_words / len([w for w in query_words if len(w) > 2])
                        similarity = min(0.99, similarity + (0.3 * match_ratio))  # Add up to 30% boost

                # Boost score for metadata matches
                metadata_boost = 0

                # Boost if result has matching actor (significantly higher boost)
                if matched_actors and actors:
                    matching_actors = [a for a in actors if any(a.lower() == ma.lower() for ma in matched_actors)]
                    if matching_actors:
                        # +0.25 per matching actor (up from +0.15)
                        metadata_boost += min(0.30, 0.25 * len(matching_actors))

                # Boost if result has matching genre
                if matched_genres and genres:
                    matching_genres = [g for g in genres if any(g.lower() == mg.lower() for mg in matched_genres)]
                    if matching_genres:
                        metadata_boost += min(0.20, 0.10 * len(matching_genres))

                similarity = min(0.99, similarity + metadata_boost)

                results.append({
                    'id': doc_id,
                    'title': title,
                    'type': doc_type,
                    'description': description,
                    'plex_key': plex_key,
                    'poster_url': poster_url,
                    'rating': rating,
                    'actors': actors,
                    'year': year,
                    'duration': duration,
                    'director': director,
                    'genres': genres,
                    'resolution': resolution,
                    'similarity': similarity
                })
            except Exception as e:
                print(f"Error processing row: {e}")
                continue

        # Deduplicate by plex_key, keeping the highest similarity score
        seen_keys = {}
        deduped_results = []
        for result in results:
            key = result['plex_key']
            if key not in seen_keys:
                seen_keys[key] = result
                deduped_results.append(result)
            else:
                # Keep the result with higher similarity
                if result['similarity'] > seen_keys[key]['similarity']:
                    # Replace in list
                    idx = next(i for i, r in enumerate(deduped_results) if r['plex_key'] == key)
                    deduped_results[idx] = result
                    seen_keys[key] = result

        # Sort by similarity and take top N
        deduped_results.sort(key=lambda x: x['similarity'], reverse=True)

        # Apply strict filtering if enabled: only return results with matching actors/genres
        if query_data.strict_filter and (matched_actors or matched_genres):
            strict_filtered = []
            for result in deduped_results:
                has_matching_actor = False
                has_matching_genre = False

                # Check for matching actors
                if matched_actors and result['actors']:
                    result_actors_lower = [a.lower() for a in result['actors']]
                    has_matching_actor = any(
                        ma.lower() in result_actors_lower
                        for ma in matched_actors
                    )

                # Check for matching genres
                if matched_genres and result['genres']:
                    result_genres_lower = [g.lower() for g in result['genres']]
                    has_matching_genre = any(
                        mg.lower() in result_genres_lower
                        for mg in matched_genres
                    )

                # Include result if it matches required filters
                # If both actors and genres mentioned, require at least one of each
                # If only actors mentioned, require at least one actor
                # If only genres mentioned, require at least one genre
                if matched_actors and matched_genres:
                    if has_matching_actor and has_matching_genre:
                        strict_filtered.append(result)
                elif matched_actors:
                    if has_matching_actor:
                        strict_filtered.append(result)
                elif matched_genres:
                    if has_matching_genre:
                        strict_filtered.append(result)

            deduped_results = strict_filtered

        # Filter by minimum relevance threshold and limit results
        filtered_results = [r for r in deduped_results if r['similarity'] >= query_data.min_relevance]
        top_results = filtered_results[:query_data.limit]

        return [
            SearchResult(
                title=r['title'],
                type=r['type'],
                description=r['description'],
                relevance=round(r['similarity'], 3),
                plex_key=r['plex_key'],
                poster_url=r['poster_url'],
                rating=r['rating'],
                actors=r['actors'],
                year=r['year'],
                duration=r['duration'],
                director=r['director'],
                genres=r['genres'],
                resolution=r['resolution']
            )
            for r in top_results
        ]

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

@app.get("/index-status", response_model=IndexStatusResponse)
async def index_status():
    """Get indexing status"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('SELECT COUNT(*) FROM embeddings')
        total_chunks = c.fetchone()[0]
        c.execute('SELECT COUNT(DISTINCT plex_key) FROM embeddings')
        total_items = c.fetchone()[0]
        conn.close()

        return IndexStatusResponse(
            total_items=total_items,
            status=f"indexed - {total_chunks} chunks from {total_items} items" if total_chunks > 0 else "empty"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Status error: {str(e)}")

@app.post("/rebuild-index")
async def rebuild_index():
    """Trigger a rebuild of the index from Plex"""
    try:
        from indexer import index_plex_library
        count = index_plex_library(embedder)
        return {
            "status": "indexed",
            "items_indexed": count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing error: {str(e)}")

# Serve static frontend
static_dir = Path("./static")
if static_dir.exists():
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
