from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import os
from pathlib import Path

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

# Initialize Chroma client
db_dir = Path("./chroma_db")
db_dir.mkdir(exist_ok=True)
client = chromadb.PersistentClient(path=str(db_dir))

# Get or create collection
try:
    collection = client.get_collection(name="plex_library")
except:
    collection = client.create_collection(
        name="plex_library",
        metadata={"hnsw:space": "cosine"}
    )

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
        query_embedding = embedder.encode(query_data.query).tolist()

        # Search in Chroma
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=query_data.limit,
            include=["documents", "metadatas", "distances"]
        )

        if not results["ids"] or not results["ids"][0]:
            return []

        # Transform results
        search_results = []
        for i, doc_id in enumerate(results["ids"][0]):
            metadata = results["metadatas"][0][i]
            distance = results["distances"][0][i]
            # Convert distance to relevance score (cosine distance -> similarity)
            relevance = 1 - distance

            result = SearchResult(
                title=metadata.get("title", "Unknown"),
                type=metadata.get("type", "unknown"),
                description=metadata.get("description", ""),
                relevance=round(relevance, 3),
                plex_key=metadata.get("plex_key", ""),
                poster_url=metadata.get("poster_url", "")
            )
            search_results.append(result)

        return search_results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

@app.get("/index-status", response_model=IndexStatusResponse)
async def index_status():
    """Get indexing status"""
    try:
        count = collection.count()
        return IndexStatusResponse(
            total_items=count,
            status="indexed" if count > 0 else "empty"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Status error: {str(e)}")

@app.post("/rebuild-index")
async def rebuild_index():
    """Trigger a rebuild of the index from Plex"""
    try:
        # Import here to avoid circular dependency
        from indexer import index_plex_library

        count = index_plex_library(collection, embedder)
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
