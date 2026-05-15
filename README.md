# Plex RAG Search

A semantic search service for your Plex library using RAG (Retrieval-Augmented Generation) and AI embeddings. Ask natural questions about your media and get smart results.

## Features

- **Semantic Search**: Ask questions like "movies with supernatural monsters" and find content matching the intent (e.g., werewolves) even if not in the title
- **Full Metadata Indexing**: Searches descriptions, titles, and other metadata
- **Web UI**: Clean, modern interface for searching your library
- **Docker Ready**: Easy deployment in your homelab
- **Fast**: Uses ChromaDB with vector embeddings for quick searches

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
```

### 3. Run with Docker

```bash
docker-compose up --build
```

The service will be available at `http://localhost:8000`

### 4. Index Your Library

Open the web UI and it will automatically check the index status. If empty, you'll need to trigger indexing:

```bash
# Inside the container
docker exec plex-rag-search python indexer.py
```

Or hit the API:
```bash
curl -X POST http://localhost:8000/rebuild-index
```

## Usage

### Web UI

1. Open `http://localhost:8000`
2. Type a natural language query:
   - "horror movies with zombies"
   - "comedies from the 90s"
   - "sci-fi with time travel"
   - "supernatural monsters like werewolves"
3. Results show matching content with relevance scores

### API

**Search**
```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "movies with supernatural monsters", "limit": 10}'
```

**Index Status**
```bash
curl http://localhost:8000/index-status
```

**Rebuild Index**
```bash
curl -X POST http://localhost:8000/rebuild-index
```

## Architecture

- **FastAPI Backend**: RESTful API for search
- **ChromaDB**: Vector database for embeddings
- **Sentence Transformers**: AI embeddings (all-MiniLM-L6-v2)
- **PlexAPI**: Plex library integration
- **Frontend**: Vanilla JavaScript web UI

## How It Works

1. **Indexing**: Your Plex library is fetched and descriptions are split into chunks
2. **Embeddings**: Each chunk is converted to a semantic embedding vector
3. **Storage**: Embeddings are stored in ChromaDB with metadata
4. **Search**: Your query is embedded and matched against the library using semantic similarity
5. **Results**: Ranked by relevance score (0-1)

## Performance Notes

- First index takes a few minutes depending on library size
- Searches are typically <100ms
- Embeddings are cached locally
- Index persists in `./chroma_db/` volume

## Troubleshooting

**"Cannot connect to Plex"**
- Check `PLEX_URL` is correct
- Verify `PLEX_TOKEN` is valid
- Ensure Plex is running and accessible from the container

**"Index is empty"**
- Run `docker exec plex-rag-search python indexer.py`
- Check logs: `docker logs plex-rag-search`

**"Docker can't find host.docker.internal"**
- On Linux, use your actual Plex IP instead: `PLEX_URL=http://192.168.x.x:32400`

## Development

To run locally without Docker:

```bash
pip install -r requirements.txt
python indexer.py  # Index your library
python main.py     # Start the server
```

## Future Enhancements

- Advanced filters (year, rating, genre)
- Watch history integration
- Recommendations based on searches
- Multi-library support
- Custom embedding models
