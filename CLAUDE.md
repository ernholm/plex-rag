# Plex RAG Search - Project Documentation

**For Future Claude Sessions**

## Project Overview
A semantic search service for ernholm's homelab Plex library. Users ask natural language questions and get results matching the intent, not just keywords. Example: "movies with supernatural monsters" → finds werewolves even if not in title.

## GitHub Repository
- **URL**: https://github.com/ernholm/plex-rag
- **Owner**: ernholm
- **Visibility**: Public

## Architecture

### Backend (`main.py`)
- FastAPI server on port 8000
- Endpoints:
  - `POST /search` - Semantic search query
  - `GET /index-status` - Check if library is indexed
  - `POST /rebuild-index` - Trigger re-indexing from Plex
  - `GET /health` - Health check

### Indexer (`indexer.py`)
- Connects to Plex via PlexAPI
- Fetches Movies and TV Shows sections
- Chunks descriptions for better coverage
- Embeds chunks using Sentence Transformers (`all-MiniLM-L6-v2`)
- Stores in ChromaDB with metadata (title, type, description, plex_key, poster_url)

### Frontend (`static/index.html`)
- Vanilla JavaScript web UI
- Search input with live results
- Result cards showing title, type, relevance %, description
- Poster images from Plex
- Responsive grid layout

### Database
- **ChromaDB**: Vector database for embeddings
- **Storage**: Named Docker volume `chroma_data`
- **Distance Metric**: Cosine similarity
- **Embedding Model**: Sentence Transformers all-MiniLM-L6-v2

## Configuration

### Environment Variables
```
PLEX_URL=http://your-plex-ip:32400
PLEX_TOKEN=your_plex_token_here
```

Get token from: Plex Settings → Remote Access → Get Token

### Docker Deployment
```bash
docker-compose up --build
```

- Uses named volume for data persistence
- Exposes port 8000
- Auto-restarts on failure
- Built-in health checks

## How Search Works

1. **Query Embedding**: User's query is embedded to vector
2. **Vector Search**: ChromaDB finds most similar chunks via cosine distance
3. **Ranking**: Results ranked by similarity score (0-1)
4. **Metadata Enrichment**: Returns title, description, poster URL, type

Example flow:
- Query: "movies with supernatural monsters"
- Embedded to semantic vector
- Matched against movie descriptions in DB
- Returns: Werewolf movies, vampire films, etc.

## Initial Setup (from GitHub)

```bash
# Clone
git clone https://github.com/ernholm/plex-rag.git
cd plex-rag

# Configure
cp .env.example .env
# Edit .env with your Plex credentials

# Deploy
docker-compose up --build

# Index library (first time or when library changes)
docker exec plex-rag-search python indexer.py
```

## Development Notes

### Dependencies
- `fastapi` - Web framework
- `chromadb` - Vector database
- `sentence-transformers` - Embeddings
- `plexapi` - Plex library access
- `uvicorn` - ASGI server

### Key Design Decisions
- **Overlapping chunks**: Descriptions split with overlap for better semantic coverage
- **Named volumes**: Docker manages permissions, works cross-platform
- **ChromaDB**: Lightweight, persistent, no external DB needed
- **Small embedding model**: all-MiniLM-L6-v2 is fast and accurate for this use case

### Performance
- Indexing: ~few minutes for typical library (depends on size)
- Search: ~100ms per query
- Memory: ~500MB with typical library

## Future Enhancements
- Advanced filters (year, rating, genre, duration)
- Watch history integration
- Personalized recommendations
- Multi-library support
- Switch embedding models (larger ones for more accuracy)
- Scheduled re-indexing
- Query logging/analytics

## Troubleshooting

**"Cannot connect to Plex"**
- Verify PLEX_URL is correct and Plex is running
- Check PLEX_TOKEN is valid
- On Linux, use actual IP instead of localhost

**"Index is empty"**
- Run: `docker exec plex-rag-search python indexer.py`
- Check logs: `docker logs plex-rag-search`

**Permission errors with volume**
- Use named volumes (already configured)
- Don't use bind mounts if possible

## Deployment Notes
- Container: `plex-rag-search`
- Port: 8000
- Volume: `chroma_data`
- Network: `plex-network` (bridge driver)
- Restart policy: unless-stopped
