# Plex RAG Search

A semantic search service for your Plex library. Ask natural questions about your media and get smart results based on meaning, not just keywords.

## Features

- **Semantic Search**: "movies about grief and loss" finds relevant films even if those words aren't in the title
- **Actor & Genre Filtering**: "Bruce Willis sci-fi" boosts films with Bruce Willis in sci-fi genre; enable Strict Filter to require exact matches
- **Country/Nationality Filtering**: "Korean thrillers" or "Japanese horror" boosts and filters by country of origin — works for 35+ nationalities
- **Year, Rating & Resolution Filters**: "feel-good movies in 4K" or "rating of 8 or higher before 2000" apply hard filters — only matching results are returned. Supported resolutions: `4K` (or `UHD`), `1080p` (or `HD`, `Full HD`), `720p`, `SD`
- **Rating Sort**: "best/great/good/highest/top rated" sorts by rating descending; "worst/lowest rated" sorts ascending — generic queries like "best tv shows" return all items of that type sorted by rating
- **Year Sort**: "first/oldest" sorts by year ascending (oldest first); "most recent/latest/newest/recent" sorts by year descending (newest first) — e.g. "the first Bond movies" or "most recent horror films"
- **Added Sort**: "newly added", "recently added", "just added", or "new additions" sorts by Plex library add date descending — e.g. "newly added tv shows" or "recently added horror movies"
- **Exclusions**: "spy movies without James Bond", "thrillers excluding Tom Cruise", "horror not starring Sigourney Weaver" — excluded term is matched against title, description, and cast
- **Type Filtering**: Queries mentioning "movies"/"films" return only movies; "shows"/"series" returns only TV shows; ambiguous queries return both
- **Smart Query Parsing**: Actor names and nationality words are stripped before embedding so the semantic search focuses on the concept, not the name
- **Rich Metadata**: Year, duration, director, genres, resolution (4K/1080p/720p/SD), audience rating, and cast (up to 10 actors)
- **Infinite Scroll**: Starts with 12 results, loads more as you scroll — fetches up to 50 total per query
- **Strict Filter**: Checkbox to require all mentioned actors, genres, and countries to be present
- **Mobile-Friendly**: Responsive layout, modal slides up from bottom on iPhone/Android
- **Parallel Indexing**: 4 concurrent workers for fast library indexing
- **Docker Ready**: Easy homelab deployment with persistent storage

## Quick Start

### 1. Get Your Plex Token

Go to Plex Settings → Remote Access → Get Token and copy your token.

### 2. Setup

```bash
git clone https://github.com/ernholm/plex-rag.git
cd plex-rag

cp .env.example .env
# Edit .env with your Plex details
```

```env
PLEX_URL=http://your-plex-ip:32400
PLEX_TOKEN=your_token_here
PLEX_SECTIONS=Movies,TV Shows
```

### 3. Run with Docker

```bash
docker-compose up --build
```

Service available at `http://localhost:8000`

### 4. Index Your Library

Click **↻ Rebuild index** in the web UI. Indexing runs in the background — a progress bar shows how many items have been processed. The service remains searchable while indexing runs.

Or trigger via API (returns immediately, runs in background):

```bash
curl -X POST http://localhost:8000/rebuild-index

# Check progress
curl http://localhost:8000/index-progress
```

Indexing takes a few minutes — it reloads full metadata per item from Plex to get the complete cast, country, and rating data.

## Usage

### Web UI

1. Open `http://localhost:8000`
2. Type a natural language query and press **Enter** or click **Search**:
   - `horror movies with zombies`
   - `Bruce Willis time travel` — strips "Bruce Willis" before semantic search, boosts his films
   - `Korean thrillers` — strips "Korean" before semantic search, boosts South Korean films
   - `funny 80s comedies`
   - `movies about friendship and loss`
3. Results load 12 at a time; scroll down to load more
4. Click any result for full details: poster, rating, duration, director, genres, cast

**Strict Filter** checkbox: when enabled, only results matching *all* mentioned actors, genres, and countries are shown. Useful for "Bruce Willis science fiction" to exclude non-sci-fi Bruce Willis films.

**Min quality** dropdown: filters out low-confidence matches (default 30%).

### API

**Search**
```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "Korean crime thriller", "limit": 50, "min_relevance": 0.3, "strict_filter": false}'
```

**Index Status**
```bash
curl http://localhost:8000/index-status
```

**Rebuild Index**
```bash
curl -X POST http://localhost:8000/rebuild-index
```

**Health / Version**
```bash
curl http://localhost:8000/health
```

## Configuration

### Environment Variables

```env
PLEX_URL=http://192.168.1.100:32400
PLEX_TOKEN=your_plex_token

# Comma-separated Plex library section names to index
PLEX_SECTIONS=Movies,TV Shows
```

## Architecture

| Component | Technology |
|---|---|
| Backend | FastAPI (Python) |
| Embeddings | Sentence Transformers `all-MiniLM-L6-v2` |
| Database | SQLite (embeddings + metadata) |
| Plex integration | PlexAPI |
| Frontend | Vanilla JS, no frameworks |
| Deployment | Docker + named volume |

## How It Works

### Indexing

1. Fetch all items from configured Plex libraries
2. Call `item.reload()` per item to get full metadata (cast, countries, rating)
3. Chunk descriptions into overlapping 300-word segments
4. Embed each chunk with Sentence Transformers
5. Normalize non-English genre names to English equivalents (e.g. Dokumentär → Documentary)
6. Store embeddings + metadata (title, year, duration, director, genres, resolution, countries, rating, cast, added date) in SQLite

### Searching

1. Parse query for actor names, genre names, nationality words, year constraints, and rating constraints
2. Strip actor names and nationality words from the query before embedding — semantic search focuses on the concept
3. Embed the cleaned query
4. Score all chunks by cosine similarity
5. Apply metadata boosts:
   - Title match: up to +30%
   - Actor match: +40% per actor (max +50%)
   - Genre match: +10% per genre (max +20%)
   - Country match: +20%
6. Deduplicate by Plex key (keep highest-scoring chunk per item)
7. Apply Strict Filter if enabled (require all matched actors/genres/countries)
8. Apply year/rating hard filters when explicitly stated (e.g. "last couple of years", "rating of 8 or higher")
9. Return top results sorted by score

### Supported Nationalities

Korean, Japanese, Chinese, French, German, Italian, Spanish, Swedish, Danish, Norwegian, Finnish, Russian, Indian, Iranian, Thai, Mexican, Brazilian, Australian, British, American, Canadian, Hong Kong, Taiwanese, Turkish, Polish, Romanian, Greek, Portuguese, Dutch, Belgian, Austrian, Swiss, Israeli, and more.

## Performance

- **Indexing**: ~2500 movies in 8–10 minutes with 4 concurrent workers
- **Search**: <100ms per query
- **Storage**: ~100MB for 2500 movies with embeddings

## Troubleshooting

**"Cannot connect to Plex"**
- Verify `PLEX_URL` is correct and Plex is running
- Check `PLEX_TOKEN` is valid
- On Linux, use actual IP instead of `localhost`

**"Index is empty"**
- Click **↻ Rebuild index** in the web UI
- Check logs: `docker logs plex-rag-search`

**Ratings show as blank**
- Rebuild the index — ratings use `audienceRating` from Plex which requires full metadata

**Only 3 actors showing**
- Rebuild the index — the current indexer calls `item.reload()` to fetch the full cast (up to 10 actors)

**Strict filter returns no results**
- The mentioned actor, genre, or country must be present in the indexed metadata
- Try without strict filter first to verify the data is there
