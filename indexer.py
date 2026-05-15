"""
Indexer for Plex library
Fetches metadata from Plex and creates embeddings, storing in SQLite
"""
import os
from plexapi.server import PlexServer
import logging
import sqlite3
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_PATH = "./plex_embeddings.db"

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

def init_db():
    """Initialize SQLite database"""
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
            rating REAL,
            actors TEXT,
            embedding BLOB NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def index_plex_library(embedder):
    """
    Index all Plex library content
    Returns count of indexed items
    """
    init_db()

    try:
        plex = get_plex_connection()
    except Exception as e:
        logger.error(f"Plex connection error: {e}")
        return 0

    # Get full Plex URL for poster images
    plex_url = os.getenv("PLEX_URL", "http://localhost:32400")
    plex_token = os.getenv("PLEX_TOKEN")

    # Clear existing database
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('DELETE FROM embeddings')
    conn.commit()
    conn.close()
    logger.info("Cleared existing index")

    indexed_count = 0

    # Index movies
    try:
        movie_section = plex.library.section("Movies")
        for movie in movie_section.all():
            try:
                description = movie.summary or f"Movie: {movie.title}"
                chunks = chunk_description(description)

                for chunk_idx, chunk in enumerate(chunks):
                    embedding = embedder.encode(chunk).tolist()

                    # Build full poster URL
                    poster_url = ""
                    if movie.thumb:
                        poster_url = f"{plex_url}{movie.thumb}?X-Plex-Token={plex_token}"

                    # Get actors (up to 5)
                    actors = []
                    try:
                        if hasattr(movie, 'roles') and movie.roles:
                            actors = [actor.tag for actor in movie.roles[:5]]
                    except:
                        pass

                    # Get rating
                    rating = getattr(movie, 'rating', None)

                    conn = sqlite3.connect(DB_PATH)
                    c = conn.cursor()
                    c.execute('''
                        INSERT INTO embeddings (id, title, type, description, plex_key, poster_url, rating, actors, embedding)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        f"movie_{movie.key}_{chunk_idx}",
                        movie.title,
                        "movie",
                        movie.summary or "",
                        str(movie.key),
                        poster_url,
                        rating,
                        json.dumps(actors),
                        json.dumps(embedding)
                    ))
                    conn.commit()
                    conn.close()
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

                    # Build full poster URL
                    poster_url = ""
                    if show.thumb:
                        poster_url = f"{plex_url}{show.thumb}?X-Plex-Token={plex_token}"

                    # Get actors (up to 5)
                    actors = []
                    try:
                        if hasattr(show, 'roles') and show.roles:
                            actors = [actor.tag for actor in show.roles[:5]]
                    except:
                        pass

                    # Get rating
                    rating = getattr(show, 'rating', None)

                    conn = sqlite3.connect(DB_PATH)
                    c = conn.cursor()
                    c.execute('''
                        INSERT INTO embeddings (id, title, type, description, plex_key, poster_url, rating, actors, embedding)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        f"show_{show.key}_{chunk_idx}",
                        show.title,
                        "tv_show",
                        show.summary or "",
                        str(show.key),
                        poster_url,
                        rating,
                        json.dumps(actors),
                        json.dumps(embedding)
                    ))
                    conn.commit()
                    conn.close()
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

    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    count = index_plex_library(embedder)
    print(f"\n✓ Successfully indexed {count} chunks")
