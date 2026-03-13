# pixmap

Local image similarity search using CLIP embeddings, HNSW indexing, and SQLite. Everything runs on your machine — no cloud, no API keys, your images stay private.

## Install

```bash
npm install pixmap
```

Requires Node.js 18+. Native modules (`sharp`, `better-sqlite3`, `hnswlib-node`) compile during install.

## Usage

### Command Line

```bash
pixmap add ./photo.jpg            # index a single image
pixmap add ./photos/              # index a directory (recursive)
pixmap search ./query.jpg         # find similar images
pixmap search ./query.jpg -k 10   # return more results
pixmap list                       # show indexed images
pixmap status                     # index stats
```
### Terminal View
<img src="https://github.com/user-attachments/assets/0cfc7f0f-86b5-4a80-baaa-10feb28f811e" width="580" alt="Image"/>


### As a library

```typescript
import { PixmapEngine } from "pixmap";

const engine = new PixmapEngine({ dataDir: "./data" });
await engine.init();

await engine.add("./photos/dog.jpg");
await engine.add("./photos/cat.png");

const results = await engine.search("./query.jpg", 5);

for (const r of results) {
  console.log(`${r.path} — ${(r.score * 100).toFixed(1)}%`);
}
```

Options:

- `-d, --data-dir <path>` — where to store index and db (default: `./data`)
- `-k, --top-k <n>` — number of results (default: `5`)

## API

### `new PixmapEngine(options)`

```typescript
{
  dataDir: string;          // where to store index.hnsw and metadata.db
  indexFileName?: string;   // default: "index.hnsw"
  dbFileName?: string;      // default: "metadata.db"
}
```

### Methods

- `**init()**` — loads the CLIP model, opens or creates the HNSW index and SQLite db. Must be called before anything else.
- `**add(imagePath)**` — indexes an image. Returns `{ id, skipped, record }`. If the image was already indexed, `skipped` is `true`.
- `**search(imagePath, topK?)**` — embeds the query image and returns the top-K most similar indexed images. Default K is 5.
- `**listImages(limit?)**` — returns indexed images, newest first.
- `**getImage(id)**` — look up a single image by its id.
- `**getIndexedCount()**` — total number of indexed images.

## How it works

Each image is resized to 224x224 with Sharp, run through a CLIP vision model (ViT-B/32, via ONNX), producing a 512-dimensional embedding vector. That vector goes into an HNSW index for fast nearest-neighbor lookup. File paths, IDs, and timestamps are tracked in a SQLite database.

Searching works the same way: embed the query image, ask HNSW for the closest vectors, look up the metadata.

See [docs/HOW.md](docs/HOW.md) for the full technical breakdown.

## What gets stored

Everything lives in your `dataDir`:

- `index.hnsw` — the vector index (binary, hnswlib format)
- `metadata.db` — SQLite database with image paths and metadata

About ~2KB per indexed image. A 100K image index is roughly 220MB.

## Supported formats

JPEG, PNG, WebP, BMP, GIF, AVIF, TIFF — anything Sharp can read.

## Dependencies

- `[@xenova/transformers](https://github.com/xenova/transformers.js)` — runs the CLIP model locally via ONNX
- `[sharp](https://github.com/lovell/sharp)` — image resize and preprocessing
- `[hnswlib-node](https://github.com/yoshoku/hnswlib-node)` — approximate nearest neighbor index
- `[better-sqlite3](https://github.com/WiseLibs/better-sqlite3)` — metadata storage

## License

MIT
