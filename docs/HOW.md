# How pixmap works

This document walks through the internals — how images get indexed, how search works, and why the stack is what it is.

## The pieces

Pixmap has a pretty simple architecture. There are three core components and a thin engine that ties them together:

```
PixmapEngine (src/engine.ts)
├── ImageEmbedder  (src/embedder.ts)   — image → 512-d vector
├── VectorStore    (src/vectorStore.ts) — HNSW index (add/search/save)
└── MetadataDb     (src/metadataDb.ts)  — SQLite (paths, ids, timestamps)
```

On top of these, `indexer.ts` and `searcher.ts` are small workflow modules that coordinate the add and search operations respectively. The engine itself just wires everything together and exposes the public API.

## Indexing an image

When you call `engine.add(imagePath)`, here's what actually happens:

**1. Dedup check.** The path is upserted into SQLite. If it already exists with `indexed = 1`, we bail early — no work to do. If it exists with `indexed = 0` (maybe a previous run crashed halfway through), we re-process it.

**2. Preprocessing.** Sharp resizes the image to 224x224 pixels using a cover fit (maintains aspect ratio, crops the excess). Alpha channels get stripped. The output is a raw PNG buffer — this is what the CLIP model expects.

**3. Embedding.** The preprocessed image is fed through `clip-vit-base-patch32` running in ONNX via `@xenova/transformers`. This produces a 512-dimensional float vector. The vector gets L2-normalized so it sits on the unit hypersphere, which is needed for cosine similarity to work correctly with HNSW.

**4. Indexing.** The normalized vector is inserted into the HNSW index with the SQLite row ID as its key. The index is saved to disk immediately.

**5. Marking done.** The SQLite row gets `indexed = 1`. This is the commit point — if the process dies before this, the next run will re-process the image.

## Searching

Search follows the same embedding path, then diverges:

1. The query image goes through the same Sharp → CLIP → normalize pipeline.
2. The resulting vector is handed to HNSW, which does an approximate nearest-neighbor search and returns the top-K vector IDs along with their cosine distances.
3. Distances are converted to similarity scores (`1 - distance`, so 1.0 = identical).
4. Each ID is looked up in SQLite to get the file path and timestamp.
5. Results come back as an array of `{ id, path, score, created, indexed }`.

The search itself (step 2) takes under 2ms even with 100K vectors in the index. The bottleneck is always the CLIP embedding step, which runs ~200-500ms on CPU.

## Embedding: CLIP in detail

### Why CLIP

CLIP understands images semantically. Two photos of a dog — one close-up, one from across a park — will have similar embeddings even though their pixels are completely different. This is what makes it useful for "find images like this one" rather than just pixel-matching.

We use `clip-vit-base-patch32`, a ViT-B/32 variant. It's about 150MB, runs fine on CPU, and gives 512-d vectors. The model downloads automatically on first run and gets cached in `~/.cache/huggingface/`.

### The preprocessing pipeline

CLIP needs exactly 224x224x3 input. Here's how we get there:

1. **Resize** to 224x224 with cover fit — aspect ratio is preserved, overflow is cropped.
2. **Strip alpha** — drop down to 3 channels (RGB).
3. **Export as PNG** — lossless buffer, no compression artifacts.
4. **Build RawImage** — 150,528 pixel values as a Uint8Array, ready for the model.

### Normalization

After inference, we normalize the vector to unit length:

```
normalized[i] = v[i] / sqrt(v[0]² + v[1]² + ... + v[511]²)
```

This matters because HNSW's cosine distance mode assumes unit vectors. Without normalization, similarity scores would be inconsistent — vectors with larger magnitudes would distort the distance calculations.

## Vector index: HNSW

### How HNSW works

HNSW (Hierarchical Navigable Small World) is a graph-based approximate nearest neighbor algorithm. Think of it like a skip list but in vector space:

- The bottom layer has every vector, connected to its closest neighbors.
- Each layer above has fewer and fewer vectors, with longer-range connections.
- To search, you start at the top (where there are few nodes but long jumps) and work down, getting more precise at each layer.

```
Layer 2:  A ──────────── D              (express lanes)
          │              │
Layer 1:  A ── B ── C ── D ── E         (mid-range)
          │   │   │   │   │
Layer 0:  A  B  C  D  E  F  G  H  I    (all vectors, local connections)
```

This gives O(log n) search instead of O(n) brute force. At 100K images, that's the difference between ~2ms and ~500ms per query.

### Configuration

The index is set up with:

- **512 dimensions** (matching CLIP output)
- **Cosine distance** (since vectors are normalized)
- **100K initial capacity** (auto-resizes to 1.5x when full)

The resize creates a new index, copies everything over, and swaps it in. Not the fastest operation, but it's a one-time cost and it keeps things simple.

### Persistence

After every `add()` call, the index is written to `index.hnsw`. On startup, if this file exists, it's loaded back. The format is hnswlib's native binary — it includes the vectors, graph edges, and config in a single file.

## Metadata: SQLite

### Schema

```sql
CREATE TABLE IF NOT EXISTS images (
  id       INTEGER PRIMARY KEY AUTOINCREMENT,
  path     TEXT    NOT NULL UNIQUE,
  indexed  INTEGER NOT NULL DEFAULT 0,
  created  INTEGER NOT NULL
);
```

The `id` column does double duty — it's both the primary key in SQLite and the vector ID in HNSW. When HNSW returns vector ID 42, we look up `images WHERE id = 42` to get the file path. Simple 1:1 mapping, no translation layer needed.

The `indexed` column (0 or 1) tracks whether the embedding pipeline completed. The `UNIQUE` constraint on `path` prevents the same file from being indexed twice.

### Why SQLite instead of just a JSON file

A JSON file would work for small collections, but it falls apart quickly:

- No atomic writes — a crash mid-save corrupts the whole thing.
- No indexed lookups — finding a record by path means scanning the entire file.
- No concurrent access — reading while writing is a recipe for data loss.

SQLite handles all of this out of the box. `better-sqlite3` gives us synchronous calls (no callback juggling in the indexer), and WAL mode means reads don't block writes.

## Crash recovery

The indexing pipeline is designed so that a crash at any point leaves things in a recoverable state:

- **Crash before embedding completes:** SQLite has the row with `indexed = 0`. Next time you add the same path, it picks up where it left off.
- **Crash after embedding but before HNSW save:** The vector is lost (wasn't persisted). SQLite still shows `indexed = 0`, so the next add re-embeds and re-inserts.
- **Crash after HNSW save but before marking indexed:** On next run, the image gets re-embedded and the vector is re-inserted into HNSW. HNSW handles duplicate IDs by overwriting, so no corruption.
- **Crash after everything:** Both stores are consistent. `indexed = 1` prevents reprocessing.

There's no transaction spanning both stores — instead, the two-phase design (insert metadata first, mark complete last) naturally handles partial failures.

## Initialization

When `engine.init()` is called, three things happen:

1. The CLIP model loads. First run downloads ~150MB from Hugging Face; subsequent runs use the cache.
2. The HNSW index loads from `index.hnsw` if it exists, otherwise a fresh empty index is created.
3. SQLite opens `metadata.db`, running the CREATE TABLE statement if needed (it's idempotent).

All three dependencies are loaded via dynamic `import()`. This means you can `import { PixmapEngine } from "pixmap"` without triggering any model downloads or native module loading — that only happens when you call `init()`.

There's no explicit shutdown or cleanup. HNSW is saved to disk after each add, SQLite handles its own connection lifecycle, and the ONNX session gets garbage-collected normally.

## Performance

Some rough numbers to set expectations:

**Indexing (per image):**

| Step | Time |
|------|------|
| Sharp resize | ~20ms |
| CLIP embedding | ~200-500ms |
| HNSW insert | <1ms |
| SQLite upsert | <1ms |
| HNSW disk save | ~5-50ms |

The CLIP step dominates. On a modern laptop, expect around 2-4 images per second.

**Search:**

The embedding step is the same ~200-500ms. The actual HNSW lookup is under 2ms even at 100K scale. So search latency is essentially just the time to embed the query.

**Storage:**

Each indexed image costs about 2KB (512 floats for the vector, plus graph edges, plus the SQLite row). At 100K images, you're looking at roughly 220MB for the index and 20MB for the database. The original images aren't copied or stored — pixmap only keeps the vectors and paths.
