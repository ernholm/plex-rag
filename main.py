from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import sqlite3
import json
import os
from pathlib import Path
from typing import List

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

init_db()

# Models
class SearchQuery(BaseModel):
    query: str
    limit: int = 10

class SearchResult(BaseModel):
    title: str
    type: str
    description: str
    relevance: float
    plex_key: str
    poster_url: str
    rating: float = None
    actors: List[str] = []

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
        c = conn.cursor()
        c.execute('SELECT id, title, type, description, plex_key, poster_url, rating, actors, embedding FROM embeddings')
        rows = c.fetchall()
        conn.close()

        if not rows:
            return []

        # Calculate similarity for all rows
        results = []
        for row in rows:
            try:
                doc_id, title, doc_type, description, plex_key, poster_url, rating, actors_json, embedding_blob = row

                # Deserialize embedding and actors
                embedding = json.loads(embedding_blob)
                actors = json.loads(actors_json) if actors_json else []

                # Calculate similarity
                similarity = cosine_similarity(query_embedding, embedding)

                results.append({
                    'id': doc_id,
                    'title': title,
                    'type': doc_type,
                    'description': description,
                    'plex_key': plex_key,
                    'poster_url': poster_url,
                    'rating': rating,
                    'actors': actors,
                    'similarity': similarity
                })
            except Exception as e:
                print(f"Error processing row: {e}")
                continue

        # Sort by similarity and take top N
        results.sort(key=lambda x: x['similarity'], reverse=True)
        top_results = results[:query_data.limit]

        return [
            SearchResult(
                title=r['title'],
                type=r['type'],
                description=r['description'],
                relevance=round(r['similarity'], 3),
                plex_key=r['plex_key'],
                poster_url=r['poster_url'],
                rating=r['rating'],
                actors=r['actors']
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
