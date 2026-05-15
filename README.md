# Plex RAG Search

A semantic search service for your Plex library using RAG (Retrieval-Augmented Generation) and AI embeddings. Ask natural questions about your media and get smart results.

## Features

- **Semantic Search**: Ask questions like "movies with supernatural monsters" and find content matching the intent (e.g., werewolves) even if not in the title
- **Metadata Filtering**: Search by actor name or genre and get boosted results - "Bruce Willis sci-fi" will prioritize films with Bruce Willis in sci-fi genre
- **Rich Metadata Display**: View year, duration, director, genres, resolution (4K/1080p/720p/SD), rating, and cast
- **Title Match Boosting**: Direct title matches get ranked higher in results
- **Configurable Libraries**: Index specific Plex sections via `PLEX_SECTIONS` env var
- **Smart Deduplication**: Same movie only appears once per search with highest relevance score
- **Adjustable Results**: Choose 12, 25, 50, or 100 results per search
- **Modal Details View**: Click any result to see full details including poster, description, and cast
- **Proper Aspect Ratios**: Movie posters display in correct portrait/landscape proportions
- **Parallel Indexing**: 4x faster indexing with concurrent processing
- **Web UI**: Clean, modern interface for searching your library
- **Docker Ready**: Easy deployment in your homelab

## Quick Start

### 1. Get Your Plex Token

1. Go to Plex Settings → Remote Access → Get Token
2. Copy your token (looks like `xxxxxxxxxxxxxxxxxxxx`)

### 2. Setup

```bash
# Clone/download the project
cd plex-rag

# Create .env file
cp .env.example .env

# Edit .env with your Plex details
nano .env
```

Set these values in `.env`:
```
PLEX_URL=http://your-plex-ip:32400
PLEX_TOKEN=your_token_here
PLEX_SECTIONS=Movies,TV Shows
```

### 3. Run with Docker

```bash
docker-compose up --build
```

The service will be available at `http://localhost:8000`

### 4. Index Your Library

Open the web UI - you'll see a status and a purple "⟳ Rebuild" button. Click it to start indexing your library. Progress will display in the status.

Or use the API:
```bash
curl -X POST http://localhost:8000/rebuild-index
```

## Usage

### Web UI

1. Open `http://localhost:8000`
2. Select number of results (12, 25, 50, or 100)
3. Type a natural language query:
   - "horror movies with zombies"
   - "Bruce Willis time travel sci-fi"
   - "comedies from the 90s"
   - "supernatural monsters like werewolves"
   - "4K action movies"
4. Results show matching content with relevance scores, resolution badges, and type labels
5. Click any result card to see full details in a modal view including:
   - Poster image
   - Title, type, and match percentage
   - Rating and resolution
   - Year, duration, and director
   - Genres
   - Cast (top 5 actors)
   - Full description

### API

**Search**
```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "movies with supernatural monsters", "limit": 25}'
```

**Index Status**
```bash
curl http://localhost:8000/index-status
```

Returns: `"indexed - 3300 chunks from 2500 items"`

**Rebuild Index**
```bash
curl -X POST http://localhost:8000/rebuild-index
```

## Configuration

### Environment Variables

```env
# Plex Server
PLEX_URL=http://192.168.1.100:32400
PLEX_TOKEN=your_plex_token

# Which Plex libraries to index (comma-separated)
# Leave empty to use default: Movies,TV Shows
PLEX_SECTIONS=Movies,TV Shows,4K Movies
```

### Library Selection Examples

```env
# Default (Movies and TV Shows)
PLEX_SECTIONS=Movies,TV Shows

# Multiple users
PLEX_SECTIONS=Movies,TV Shows,John's Movies,John's TV Shows

# 4K content only
PLEX_SECTIONS=4K Movies,4K TV Shows

# Single library
PLEX_SECTIONS=Movies
```

## Architecture

- **FastAPI Backend**: RESTful API for search and indexing
- **SQLite Database**: Stores embeddings and metadata locally
- **Sentence Transformers**: AI embeddings (all-MiniLM-L6-v2 model)
- **PlexAPI**: Plex library metadata integration
- **Frontend**: Vanilla JavaScript web UI with responsive design
- **Docker**: Containerized deployment

## How It Works

### Indexing

1. **Fetch**: Retrieves all items from configured Plex libraries
2. **Chunk**: Splits descriptions into overlapping chunks (300 words with 50-word overlap)
3. **Extract Metadata**: Pulls year, duration, director, genres, resolution, rating, cast
4. **Embed**: Converts each chunk to a semantic vector using Sentence Transformers
5. **Store**: Saves embeddings and metadata to SQLite database
6. **Deduplicate**: Same item only stored once with best metadata match

### Searching

1. **Query Embedding**: Your query is converted to a semantic vector
2. **Metadata Extraction**: Query is parsed for actor names and genres
3. **Vector Search**: Finds most similar chunks in the database
4. **Boosting**: Results boosted by:
   - Title matches (+30%)
   - Actor mentions (+15% per match)
   - Genre matches (+10% per genre, max +20%)
5. **Deduplication**: Same movie appears only once with highest score
6. **Ranking**: Results sorted by final relevance score (0-1)
7. **Return**: Top N results with all metadata

## Performance

- **Indexing**: ~2500 movies in 8-10 minutes with 4 concurrent workers
- **Searches**: Typically <100ms response time
- **Storage**: ~100MB for 2500 movies with embeddings
- **Data Persistence**: Embeddings persist in named Docker volume

## Troubleshooting

**"Cannot connect to Plex"**
- Verify `PLEX_URL` points to your Plex server
- Confirm `PLEX_TOKEN` is valid (get fresh token from Plex settings)
- Ensure Plex is running and accessible from the container
- On Linux, use actual IP instead of localhost: `PLEX_URL=http://192.168.x.x:32400`

**"Index is empty"**
- Click the purple "⟳ Rebuild" button in the web UI
- Or run: `docker exec plex-rag-search python indexer.py`
- Check logs: `docker logs plex-rag-search`

**"Indexing is slow or timing out"**
- Increase timeout or reduce concurrent workers if Plex server is slow
- Edit `indexer.py` to adjust `MAX_WORKERS` (default: 4)
- Check Plex server performance with `docker logs plex-rag-search`

**"Searches return wrong results"**
- Rebuild index to ensure fresh embeddings: click "⟳ Rebuild"
- Try more specific queries with actor names or genres
- Increase result limit to 50 or 100 to see more options

**"Posters not loading"**
- Ensure Plex token is valid
- Check that `PLEX_URL` is accessible and includes proper protocol
- Verify poster URLs in database: `sqlite3 plex_embeddings.db "SELECT poster_url FROM embeddings LIMIT 1;"`

## Development

To run locally without Docker:

```bash
# Install dependencies
pip install -r requirements.txt

# Configure .env
cp .env.example .env
# Edit .env with your Plex details

# Index your library
python indexer.py

# Start the server
python main.py
```

Server will run on `http://localhost:8000`

## Future Enhancements

- Advanced filtering (by year, rating range, duration)
- Watch history integration
- Personalized recommendations based on watch history
- Real-time library updates (incremental indexing)
- Custom embedding models for better accuracy
- Analytics dashboard (top genres, most-watched actors, etc.)
- Multi-user support with per-user search history
