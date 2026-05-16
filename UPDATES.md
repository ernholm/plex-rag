# Recent Updates to Plex RAG Search

## Version 1.5.0 - Debug Improvements & Genre Analysis

### Changes Made

#### 1. Version Tracking (Complete ✓)
- **Version number**: Updated to `1.5.0` in `main.py`
- **Health endpoint**: Returns `{"status": "ok", "version": "1.5.0"}`
- **UI display**: Version shown in header under "Plex RAG Search" subtitle
- **Frontend**: `loadVersion()` function fetches and displays version on page load
- **Benefit**: Users can now immediately see what code version is running without guessing

#### 2. Debug Improvements
- **Debug logging**: Enhanced genre/actor extraction debugging with more detailed logs
- **New endpoint**: `/debug/genres` (GET)
  - Returns all unique genres in the database
  - Returns sample of actors in the database
  - Useful for diagnosing why certain queries match/don't match

#### 3. Genre Matching Logic - FIXED ✓
- **Bug Found**: Multi-word genres like "Science Fiction" weren't being matched
- **Root Cause**: Code only checked if genres were complete words in query list, but "science fiction" (2-word genre) would never equal "science" or "fiction" (1-word entries)
- **Fix Applied**: Now checks if genre words appear consecutively in query
  - Single-word genres (e.g., "Comedy"): matched as complete words only
  - Multi-word genres (e.g., "Science Fiction"): matched when words appear consecutively
- **Example Fix**: Query "bruce willis science fiction" now correctly extracts:
  - Actors: ["Bruce Willis"]
  - Genres: ["Science Fiction"]

### How to Debug the Genre Matching Issue

1. **Rebuild your index** (if not already done):
   ```bash
   curl -X POST http://localhost:8000/rebuild-index
   ```

2. **Check what genres exist in your database**:
   ```bash
   curl http://localhost:8000/debug/genres
   ```
   This will show you exactly what genres are indexed. Look for "Travel" or similar.

3. **Search with logging**:
   - Open the search page at `http://localhost:8000`
   - Notice the version number displayed in the header (should be 1.5.0)
   - Try a search like "bruce willis travels back in time"
   - Check Docker logs: `docker logs plex-rag-search`
   - Look for "DEBUG: extract_metadata_filters" lines showing what was matched

### Expected Behavior with Strict Filtering

When you search for "bruce willis time travel" with strict filtering:
- If "Travel" is NOT a genre in your database, strict filtering should work correctly
- If "Travel" IS a genre (from your Plex library), it will be extracted as a matched genre
- Strict filtering with actors requires actor match; genres are used for ranking boost only

### Next Steps for Investigation

1. **Verify database content**:
   - Run `/debug/genres` endpoint to see actual genres
   - Check if "Travel" is really a genre or if something similar exists

2. **Test strict filter logic**:
   - Search "bruce willis time travel" WITHOUT strict filter (baseline)
   - Search "bruce willis time travel" WITH strict filter
   - Compare results to understand what's being matched

3. **Monitor logs**:
   - Look at the DEBUG output to see exactly what actors/genres were extracted from your query
   - This will show if the matching logic is working as expected

### Files Modified

- `main.py`:
  - Added `/debug/genres` endpoint
  - Improved debug logging in `extract_metadata_filters()`
  - VERSION constant = "1.5.0"

- `static/index.html`:
  - Already had version display and `loadVersion()` function

### Testing the New Features

```bash
# Check service version
curl http://localhost:8000/health
# Response: {"status":"ok","version":"1.5.0"}

# Debug genres in database
curl http://localhost:8000/debug/genres
# Response: {"total_genres": N, "genres": [...], "total_actors": M, "sample_actors": [...]}

# Search with full query
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "bruce willis time travel", "limit": 12, "min_relevance": 0.30, "strict_filter": true}'
```

### For Your Next Rebuild

When you rebuild the Docker container:
```bash
# Rebuild without deleting the volume
docker-compose down
docker-compose up --build

# DO NOT use -v flag (it deletes the volume)
# If you already have an index, it will be preserved
```

The service should now show version "1.5.0" in the web UI header immediately after starting.
