from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import sqlite3
import json
import os
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Version tracking
VERSION = "1.5.12"

# Indexing state — updated by background thread, read by /index-progress
indexing_state = {
    "running": False,
    "items_done": 0,
    "items_total": 0,
    "error": None,
}
_index_executor = ThreadPoolExecutor(max_workers=1)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Plex RAG Search v{VERSION} starting up")
    yield

# Initialize FastAPI app
app = FastAPI(title="Plex RAG Search", lifespan=lifespan)

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
DB_PATH = "/app/data/plex_embeddings.db"

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

# Maps demonyms and country adjectives to Plex country names
NATIONALITY_TO_COUNTRY = {
    'korean': 'South Korea',
    'japanese': 'Japan',
    'chinese': 'China',
    'french': 'France',
    'german': 'Germany',
    'italian': 'Italy',
    'spanish': 'Spain',
    'swedish': 'Sweden',
    'danish': 'Denmark',
    'norwegian': 'Norway',
    'finnish': 'Finland',
    'russian': 'Russia',
    'indian': 'India',
    'iranian': 'Iran',
    'thai': 'Thailand',
    'mexican': 'Mexico',
    'brazilian': 'Brazil',
    'argentinian': 'Argentina',
    'australian': 'Australia',
    'british': 'United Kingdom',
    'english': 'United Kingdom',
    'american': 'United States of America',
    'canadian': 'Canada',
    'hong kong': 'Hong Kong',
    'taiwanese': 'Taiwan',
    'turkish': 'Turkey',
    'polish': 'Poland',
    'romanian': 'Romania',
    'greek': 'Greece',
    'portuguese': 'Portugal',
    'dutch': 'Netherlands',
    'belgian': 'Belgium',
    'austrian': 'Austria',
    'swiss': 'Switzerland',
    'israeli': 'Israel',
}

def extract_metadata_filters(query, all_actors, all_genres):
    """Extract actor names, genres and countries from query"""
    query_lower = query.lower()
    matched_actors = []
    matched_genres = []
    matched_countries = []

    # Common multi-word phrases that shouldn't be matched as genres
    # These are semantic search concepts, not genre filters
    excluded_phrases = [
        'time travel', 'space opera', 'time machine', 'space travel',
        'action packed', 'family friendly'
    ]

    # Remove excluded phrases from consideration for genre matching
    query_for_genres = query_lower
    for phrase in excluded_phrases:
        query_for_genres = query_for_genres.replace(phrase, '')

    # Check for actor names (case-insensitive)
    for actor in all_actors:
        actor_lower = actor.lower()
        # Exact substring match - must match complete actor name
        if actor_lower in query_lower:
            # Verify it's a complete word match (not part of another word)
            # Check word boundaries
            actor_words = actor_lower.split()
            if len(actor_words) >= 2:
                # For multi-word names like "Bruce Willis", require full name match
                matched_actors.append(actor)
            elif len(actor_lower) > 4:
                # For single names, require word boundary
                query_words = query_lower.split()
                if actor_lower in query_words:
                    matched_actors.append(actor)

    # Check for genre names (case-insensitive)
    # Handle both single-word genres (e.g., "Comedy") and multi-word genres (e.g., "Science Fiction")
    query_words_for_genres = query_for_genres.split()
    query_words = query_lower.split()

    for genre in all_genres:
        genre_lower = genre.lower()
        genre_words = genre_lower.split()

        if len(genre_words) == 1:
            # Single-word genre: match as complete word only
            # Use the cleaned query (without excluded phrases)
            if genre_lower in query_words_for_genres:
                matched_genres.append(genre)
        else:
            # Multi-word genre: check if words appear consecutively in query
            # e.g., "Science Fiction" should match in "bruce willis science fiction movies"
            found = False
            for i in range(len(query_words_for_genres) - len(genre_words) + 1):
                # Check if consecutive words in query match all words of genre
                if query_words_for_genres[i:i+len(genre_words)] == genre_words:
                    found = True
                    break
            if found:
                matched_genres.append(genre)

    # Check for nationality/country words
    query_words = query_lower.split()
    for word in query_words:
        if word in NATIONALITY_TO_COUNTRY:
            country = NATIONALITY_TO_COUNTRY[word]
            if country not in matched_countries:
                matched_countries.append(country)
    # Also check two-word phrases (e.g. "hong kong")
    for i in range(len(query_words) - 1):
        phrase = f"{query_words[i]} {query_words[i+1]}"
        if phrase in NATIONALITY_TO_COUNTRY:
            country = NATIONALITY_TO_COUNTRY[phrase]
            if country not in matched_countries:
                matched_countries.append(country)

    # Log what we found for debugging
    print(f"DEBUG: extract_metadata_filters - Query '{query}' -> Actors: {matched_actors}, Genres: {matched_genres}, Countries: {matched_countries}")
    print(f"DEBUG: Query words (original): {query_words}")
    print(f"DEBUG: Query words (after phrase removal): {query_words_for_genres}")
    print(f"DEBUG: Available genres in database: {sorted(list(all_genres))[:30]}")

    return matched_actors, matched_genres, matched_countries

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
    countries: List[str] = []

class IndexStatusResponse(BaseModel):
    total_items: int
    status: str

# Routes
@app.get("/health")
async def health():
    return {"status": "ok", "version": VERSION}

@app.post("/search", response_model=list[SearchResult])
async def search(query_data: SearchQuery):
    """Search the Plex library using semantic search"""
    try:
        # Get all embeddings from database
        conn = sqlite3.connect(DB_PATH)

        # Extract metadata filters from query
        all_actors, all_genres = get_all_metadata(conn)
        matched_actors, matched_genres, matched_countries = extract_metadata_filters(query_data.query, all_actors, all_genres)

        # Strip matched actor names and nationality words from query before
        # embedding so semantic search focuses on concepts
        semantic_query = query_data.query
        for actor in matched_actors:
            semantic_query = semantic_query.lower().replace(actor.lower(), "").strip()
        for word, _ in NATIONALITY_TO_COUNTRY.items():
            semantic_query = semantic_query.lower().replace(word, "").strip()
        semantic_query = semantic_query.strip() or query_data.query

        query_embedding = embedder.encode(semantic_query)

        c = conn.cursor()
        c.execute('SELECT id, title, type, description, plex_key, poster_url, rating, actors, year, duration, director, genres, resolution, countries, embedding FROM embeddings')
        rows = c.fetchall()
        conn.close()

        if not rows:
            return []

        # Calculate similarity for all rows
        results = []
        for row in rows:
            try:
                doc_id, title, doc_type, description, plex_key, poster_url, rating, actors_json, year, duration, director, genres_json, resolution, countries_json, embedding_blob = row

                embedding = json.loads(embedding_blob)
                actors = json.loads(actors_json) if actors_json else []
                genres = json.loads(genres_json) if genres_json else []
                countries = json.loads(countries_json) if countries_json else []

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

                # Boost if result has matching actor
                if matched_actors and actors:
                    matching_actors = [a for a in actors if any(a.lower() == ma.lower() for ma in matched_actors)]
                    if matching_actors:
                        metadata_boost += min(0.10, 0.08 * len(matching_actors))

                # Boost if result has matching genre
                if matched_genres and genres:
                    matching_genres = [g for g in genres if any(g.lower() == mg.lower() for mg in matched_genres)]
                    if matching_genres:
                        metadata_boost += min(0.20, 0.10 * len(matching_genres))

                # Boost if result has matching country
                if matched_countries and countries:
                    if any(c.lower() == mc.lower() for c in countries for mc in matched_countries):
                        metadata_boost += 0.20

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
                    'countries': countries,
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

        # Apply strict filtering if enabled
        if query_data.strict_filter and (matched_actors or matched_genres or matched_countries):
            print(f"DEBUG: Strict filter enabled. Actors: {matched_actors}, Genres: {matched_genres}, Countries: {matched_countries}")
            strict_filtered = []
            for result in deduped_results:
                has_matching_actor = False
                has_matching_genre = False
                has_matching_country = False

                if matched_actors and result['actors']:
                    result_actors_lower = [a.lower() for a in result['actors']]
                    has_matching_actor = any(
                        ma.lower() == actor_name
                        for ma in matched_actors
                        for actor_name in result_actors_lower
                    )

                if matched_genres and result['genres']:
                    result_genres_lower = [g.lower() for g in result['genres']]
                    has_matching_genre = any(
                        mg.lower() == genre_name
                        for mg in matched_genres
                        for genre_name in result_genres_lower
                    )

                if matched_countries and result['countries']:
                    result_countries_lower = [c.lower() for c in result['countries']]
                    has_matching_country = any(
                        mc.lower() == c
                        for mc in matched_countries
                        for c in result_countries_lower
                    )

                # Build list of which filters were specified and which matched
                required = []
                if matched_actors:
                    required.append(has_matching_actor)
                if matched_genres:
                    required.append(has_matching_genre)
                if matched_countries:
                    required.append(has_matching_country)

                if all(required):
                    strict_filtered.append(result)
                    print(f"DEBUG: Included {result['title']}")

            print(f"DEBUG: Strict filter results: {len(strict_filtered)} items")
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
                resolution=r['resolution'],
                countries=r['countries']
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

@app.get("/debug/genres")
async def debug_genres():
    """Debug endpoint to see all genres in the database"""
    try:
        conn = sqlite3.connect(DB_PATH)
        all_actors, all_genres = get_all_metadata(conn)
        conn.close()

        return {
            "total_genres": len(all_genres),
            "genres": sorted(list(all_genres)),
            "total_actors": len(all_actors),
            "sample_actors": sorted(list(all_actors))[:20],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Debug error: {str(e)}")

@app.get("/debug/actors")
async def debug_actors():
    """Return a sample of movies with their stored actor lists"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('SELECT DISTINCT title, actors FROM embeddings WHERE actors IS NOT NULL LIMIT 20')
        rows = c.fetchall()
        conn.close()
        return [
            {"title": row[0], "actors": json.loads(row[1])}
            for row in rows
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Debug error: {str(e)}")

@app.get("/index-progress")
async def index_progress():
    return indexing_state

@app.post("/rebuild-index")
async def rebuild_index():
    """Start a background index rebuild — returns immediately"""
    if indexing_state["running"]:
        return {"status": "already_running"}

    def _progress(done: int, total: int):
        indexing_state["items_done"] = done
        indexing_state["items_total"] = total

    def _run():
        indexing_state.update({"running": True, "items_done": 0, "items_total": 0, "error": None})
        try:
            from indexer import index_plex_library
            index_plex_library(embedder, progress_callback=_progress)
        except Exception as e:
            indexing_state["error"] = str(e)
            logger.error(f"Indexing error: {e}")
        finally:
            indexing_state["running"] = False

    loop = asyncio.get_event_loop()
    loop.run_in_executor(_index_executor, _run)
    return {"status": "started"}

# Serve static frontend
static_dir = Path("./static")
if static_dir.exists():
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
