"""
Indexer for Plex library
Fetches metadata from Plex and creates embeddings for ChromaDB
"""
import os
from plexapi.server import PlexServer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_plex_connection():
    """Connect to Plex server"""
    plex_url = os.getenv("PLEX_URL", "http://localhost:32400")
    plex_token = os.getenv("PLEX_TOKEN")

    if not plex_token:
        raise ValueError(
            "PLEX_TOKEN environment variable not set. "
            "Get it from: Settings > Remote Access > Get Token"
        )

    try:
        server = PlexServer(plex_url, plex_token)
        logger.info(f"Connected to Plex server: {server.friendlyName}")
        return server
    except Exception as e:
        raise ConnectionError(f"Failed to connect to Plex: {str(e)}")

def chunk_description(text, chunk_size=300, overlap=50):
    """
    Split description into overlapping chunks for better embedding coverage
    """
    if not text:
        return []

    chunks = []
    words = text.split()

    current_chunk = []
    current_length = 0

    for word in words:
        current_chunk.append(word)
        current_length += len(word) + 1

        if current_length >= chunk_size:
            chunks.append(" ".join(current_chunk))
            # Keep last few words for overlap
            current_chunk = current_chunk[-int(overlap / 5):]
            current_length = sum(len(w) for w in current_chunk) + len(current_chunk)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks or [text]

def index_plex_library(collection, embedder):
    """
    Index all Plex library content
    Returns count of indexed items
    """
    try:
        plex = get_plex_connection()
    except Exception as e:
        logger.error(f"Plex connection error: {e}")
        return 0

    # Clear existing collection
    collection.delete(where={})
    logger.info("Cleared existing index")

    indexed_count = 0
    item_id = 0

    # Index movies
    try:
        movie_section = plex.library.section("Movies")
        for movie in movie_section.all():
            try:
                description = movie.summary or f"Movie: {movie.title}"
                chunks = chunk_description(description)

                for chunk_idx, chunk in enumerate(chunks):
                    embedding = embedder.encode(chunk).tolist()

                    collection.add(
                        ids=[f"movie_{movie.key}_{chunk_idx}"],
                        embeddings=[embedding],
                        documents=[chunk],
                        metadatas=[{
                            "title": movie.title,
                            "type": "movie",
                            "description": movie.summary or "",
                            "plex_key": str(movie.key),
                            "poster_url": movie.thumb or "",
                            "year": movie.year or 0,
                        }]
                    )
                    indexed_count += 1

                logger.info(f"Indexed movie: {movie.title}")
            except Exception as e:
                logger.warning(f"Failed to index movie {movie.title}: {e}")
                continue

    except Exception as e:
        logger.warning(f"Could not access Movies section: {e}")

    # Index TV shows
    try:
        tv_section = plex.library.section("TV Shows")
        for show in tv_section.all():
            try:
                description = show.summary or f"TV Show: {show.title}"
                chunks = chunk_description(description)

                for chunk_idx, chunk in enumerate(chunks):
                    embedding = embedder.encode(chunk).tolist()

                    collection.add(
                        ids=[f"show_{show.key}_{chunk_idx}"],
                        embeddings=[embedding],
                        documents=[chunk],
                        metadatas=[{
                            "title": show.title,
                            "type": "tv_show",
                            "description": show.summary or "",
                            "plex_key": str(show.key),
                            "poster_url": show.thumb or "",
                            "year": show.year or 0,
                        }]
                    )
                    indexed_count += 1

                logger.info(f"Indexed TV show: {show.title}")
            except Exception as e:
                logger.warning(f"Failed to index show {show.title}: {e}")
                continue

    except Exception as e:
        logger.warning(f"Could not access TV Shows section: {e}")

    logger.info(f"Indexing complete. Total chunks indexed: {indexed_count}")
    return indexed_count

if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer
    import chromadb
    from pathlib import Path

    # Setup
    db_dir = Path("./chroma_db")
    db_dir.mkdir(exist_ok=True)
    client = chromadb.PersistentClient(path=str(db_dir))

    try:
        collection = client.get_collection(name="plex_library")
    except:
        collection = client.create_collection(
            name="plex_library",
            metadata={"hnsw:space": "cosine"}
        )

    embedder = SentenceTransformer('all-MiniLM-L6-v2')

    # Run indexing
    count = index_plex_library(collection, embedder)
    print(f"\n✓ Successfully indexed {count} chunks")
