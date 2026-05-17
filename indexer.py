"""
Indexer for Plex library
Fetches metadata from Plex and creates embeddings, storing in SQLite
"""
import os
from plexapi.server import PlexServer
import logging
import sqlite3
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_PATH = "/app/data/plex_embeddings.db"
MAX_WORKERS = 4  # Number of concurrent threads for fetching metadata

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
        # Increase timeout to 60 seconds for slow connections
        server = PlexServer(plex_url, plex_token, timeout=60)
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
    # Drop old table if it exists to ensure schema is up to date
    c.execute('DROP TABLE IF EXISTS embeddings')
    c.execute('''
        CREATE TABLE embeddings (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            type TEXT NOT NULL,
            description TEXT,
            plex_key TEXT,
            poster_url TEXT,
            rating REAL,
            actors TEXT,
            year INTEGER,
            duration INTEGER,
            director TEXT,
            genres TEXT,
            resolution TEXT,
            embedding BLOB NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def index_item(item, item_type, plex_url, plex_token, embedder):
    """Index a single movie or show item"""
    indexed_count = 0

    try:
        # Use basic title/summary if fetching full metadata times out
        try:
            description = item.summary or f"{item_type.replace('_', ' ').title()}: {item.title}"
        except Exception as e:
            logger.warning(f"Could not fetch full summary for {item.title}, using title only: {e}")
            description = f"{item_type.replace('_', ' ').title()}: {item.title}"

        chunks = chunk_description(description)

        # Build full poster URL
        poster_url = ""
        if item.thumb:
            poster_url = f"{plex_url}{item.thumb}?X-Plex-Token={plex_token}"

        # Get actors (up to 10)
        actors = []
        try:
            if hasattr(item, 'roles') and item.roles:
                actors = [actor.tag for actor in item.roles[:10]]
        except:
            pass

        # Get rating
        rating = getattr(item, 'rating', None)

        # Get year
        year = getattr(item, 'year', None)

        # Get duration (in minutes)
        duration = getattr(item, 'duration', None)
        if duration:
            duration = duration // 60000  # Convert from milliseconds to minutes

        # Get director (usually in the directors list)
        director = ""
        try:
            if hasattr(item, 'directors') and item.directors:
                director = item.directors[0].tag
        except:
            pass

        # Get genres (up to 3)
        genres = []
        try:
            if hasattr(item, 'genres') and item.genres:
                genres = [genre.tag for genre in item.genres[:3]]
        except:
            pass

        # Get resolution
        resolution = ""
        try:
            if hasattr(item, 'media') and item.media and len(item.media) > 0:
                video_res = getattr(item.media[0], 'videoResolution', None)
                if video_res:
                    # Normalize resolution format
                    res_lower = str(video_res).lower()
                    if '4k' in res_lower or '2160' in res_lower:
                        resolution = "4K"
                    elif '1080' in res_lower:
                        resolution = "1080p"
                    elif '720' in res_lower:
                        resolution = "720p"
                    elif '480' in res_lower or '540' in res_lower:
                        resolution = "SD"
                    else:
                        resolution = video_res
        except:
            pass

        for chunk_idx, chunk in enumerate(chunks):
            embedding = embedder.encode(chunk).tolist()

            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute('''
                INSERT INTO embeddings (id, title, type, description, plex_key, poster_url, rating, actors, year, duration, director, genres, resolution, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                f"{item_type}_{item.key}_{chunk_idx}",
                item.title,
                item_type,
                item.summary or "",
                str(item.key),
                poster_url,
                rating,
                json.dumps(actors),
                year,
                duration,
                director,
                json.dumps(genres),
                resolution,
                json.dumps(embedding)
            ))
            conn.commit()
            conn.close()
            indexed_count += 1

        logger.info(f"Indexed {item_type}: {item.title}")
    except Exception as e:
        logger.warning(f"Failed to index {item_type} {item.title}: {e}")

    return indexed_count

def index_plex_library(embedder):
    """
    Index all configured Plex library sections
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

    # Get configured sections to index
    sections_config = os.getenv("PLEX_SECTIONS", "Movies,TV Shows")
    sections_to_index = [s.strip() for s in sections_config.split(",") if s.strip()]

    logger.info(f"Sections to index: {sections_to_index}")

    # Get all available sections
    available_sections = {section.title: section for section in plex.library.sections()}
    logger.info(f"Available sections: {list(available_sections.keys())}")

    indexed_count = 0

    # Index each configured section
    for section_name in sections_to_index:
        if section_name not in available_sections:
            logger.warning(f"Section '{section_name}' not found. Available: {list(available_sections.keys())}")
            continue

        section = available_sections[section_name]
        section_type = section.type

        logger.info(f"Indexing section: {section_name} (type: {section_type})")

        try:
            items = list(section.all())
            item_type = "movie" if section_type == "movie" else "tv_show"
            logger.info(f"Found {len(items)} items in {section_name}, processing with {MAX_WORKERS} workers...")

            # Process items in parallel
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = [
                    executor.submit(index_item, item, item_type, plex_url, plex_token, embedder)
                    for item in items
                ]

                completed = 0
                for future in as_completed(futures):
                    try:
                        indexed_count += future.result()
                        completed += 1
                        if completed % 10 == 0:
                            logger.info(f"Progress: {completed}/{len(items)} items processed")
                    except Exception as e:
                        logger.error(f"Error in parallel task: {e}")

        except Exception as e:
            logger.error(f"Error indexing section {section_name}: {e}")

    logger.info(f"Indexing complete. Total chunks indexed: {indexed_count}")
    return indexed_count

if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer

    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    count = index_plex_library(embedder)
    print(f"\n✓ Successfully indexed {count} chunks")
