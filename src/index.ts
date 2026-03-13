#!/usr/bin/env node

import fs from 'node:fs';
import path from 'node:path';
import { ImageEmbedder } from './embedder.js';
import { addImage } from './indexer.js';
import { MetadataDb } from './metadataDb.js';
import { renderImage, renderImageRow } from './preview.js';
import { findSimilar } from './searcher.js';
import { DEFAULT_TOP_K } from './types.js';
import { VectorStore } from './vectorStore.js';

type CliOptions = {
  dataDir: string;
  topK: number;
};

const IMAGE_EXTENSIONS = new Set(['.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif', '.avif', '.tiff']);

async function main(): Promise<void> {
  const args = process.argv.slice(2);

  if (args.length === 0 || args[0] === '--help' || args[0] === '-h') {
    printUsage();
    return;
  }

  const command = args[0];
  const rest = args.slice(1);
  const options = parseOptions(rest);

  const dataDir = path.resolve(options.dataDir);
  fs.mkdirSync(dataDir, { recursive: true });

  const dbPath = path.join(dataDir, 'metadata.db');
  const indexPath = path.join(dataDir, 'index.hnsw');

  if (command === 'list') {
    const db = new MetadataDb(dbPath);
    const images = db.list(200);

    if (images.length === 0) {
      console.log('No indexed images.');
      return;
    }

    console.log(`\n  Indexed images (${images.length}):\n`);
    for (const img of images) {
      const date = new Date(img.created * 1000).toISOString().slice(0, 19);
      const name = path.basename(img.path);
      console.log(`  [${img.id}]  ${date}  ${name}`);
    }
    console.log();
    return;
  }

  if (command === 'show') {
    const target = rest.find((x) => !x.startsWith('--'));
    if (!target) {
      console.error('Error: Provide an image path or id.\n');
      process.exitCode = 1;
      return;
    }

    let imagePath: string;
    const asId = Number(target);
    if (Number.isFinite(asId) && asId > 0) {
      const db = new MetadataDb(dbPath);
      const record = db.get(asId);
      if (!record) {
        console.error(`Error: No image with id ${asId}`);
        process.exitCode = 1;
        return;
      }
      imagePath = record.path;
    } else {
      imagePath = path.resolve(target);
    }

    assertImageExists(imagePath);
    console.log();
    const preview = await renderImage(imagePath);
    console.log(preview);
    console.log(`    ${path.basename(imagePath)}\n`);
    return;
  }

  if (command === 'status') {
    const db = new MetadataDb(dbPath);
    const store = new VectorStore();
    store.initOrLoad(indexPath);
    const images = db.list(999999);

    console.log(`\n  pixmap status`);
    console.log(`  data dir:   ${dataDir}`);
    console.log(`  vectors:    ${store.getCount()}`);
    console.log(`  images:     ${images.length}`);
    console.log();
    return;
  }

  // Commands below require the embedder
  console.log('Loading CLIP model...');
  const embedder = new ImageEmbedder();
  await embedder.init();

  const store = new VectorStore();
  store.initOrLoad(indexPath);

  const db = new MetadataDb(dbPath);

  if (command === 'add') {
    const targets = rest.filter((x) => !x.startsWith('--'));
    if (targets.length === 0) {
      console.error('Error: No image path provided.\n');
      printUsage();
      process.exitCode = 1;
      return;
    }

    let added = 0;
    let skipped = 0;

    for (const target of targets) {
      const absolutePath = path.resolve(target);
      const stat = safeStat(absolutePath);

      if (!stat) {
        console.error(`  skip: not found — ${absolutePath}`);
        continue;
      }

      const files = stat.isDirectory() ? walkImages(absolutePath) : [absolutePath];

      for (const file of files) {
        const result = await addImage(file, embedder, store, db);
        if (result.skipped) {
          skipped += 1;
          console.log(`  skip: already indexed — ${path.basename(file)}`);
        } else {
          added += 1;
          console.log(`  added: id=${result.id} — ${path.basename(file)}`);
        }
      }
    }

    store.save(indexPath);
    console.log(`\nDone. added=${added} skipped=${skipped}`);
    return;
  }

  if (command === 'search' || command === 'similar') {
    const imagePath = rest.find((x) => !x.startsWith('--'));
    if (!imagePath) {
      console.error('Error: No query image provided.\n');
      printUsage();
      process.exitCode = 1;
      return;
    }

    const absolutePath = path.resolve(imagePath);
    assertImageExists(absolutePath);

    if (store.getCount() === 0) {
      console.log('Index is empty. Add images first with: pixmap add <path>');
      return;
    }

    // Show query image preview
    console.log(`\n  Query:\n`);
    const queryPreview = await renderImage(absolutePath, undefined, 20);
    console.log(queryPreview);
    console.log(`    ${path.basename(absolutePath)}\n`);

    const results = await findSimilar(absolutePath, embedder, store, db, options.topK);

    if (results.length === 0) {
      console.log('  No similar images found.\n');
      return;
    }

    console.log(`  ── Results (${results.length}) ──\n`);

    const PER_ROW = 3;

    for (let i = 0; i < results.length; i += PER_ROW) {
      const batch = results.slice(i, i + PER_ROW);
      const panels = batch
        .map((r, j) => {
          const pct = (r.score * 100).toFixed(1);
          const name = path.basename(r.path);
          return fs.existsSync(r.path)
            ? { label: `#${i + j + 1}  ${pct}%  ${name}`, imagePath: r.path }
            : null;
        })
        .filter((p): p is NonNullable<typeof p> => p !== null);

      if (panels.length > 0) {
        const row = await renderImageRow(panels);
        console.log(row);
        console.log();
      }
    }
    return;
  }

  console.error(`Unknown command: ${command}\n`);
  printUsage();
  process.exitCode = 1;
}

function parseOptions(args: string[]): CliOptions {
  let dataDir = 'data';
  let topK = DEFAULT_TOP_K;

  for (let i = 0; i < args.length; i += 1) {
    const arg = args[i];

    if (arg === '--data-dir' || arg === '-d') {
      dataDir = args[i + 1] ?? dataDir;
      i += 1;
      continue;
    }

    if (arg === '--top-k' || arg === '-k') {
      const parsed = Number(args[i + 1]);
      if (!Number.isNaN(parsed) && parsed > 0) {
        topK = Math.floor(parsed);
      }
      i += 1;
      continue;
    }
  }

  return { dataDir, topK };
}

function walkImages(rootDir: string): string[] {
  const result: string[] = [];
  const stack = [rootDir];

  while (stack.length > 0) {
    const current = stack.pop();
    if (!current) continue;

    const entries = fs.readdirSync(current, { withFileTypes: true });
    for (const entry of entries) {
      const fullPath = path.join(current, entry.name);
      if (entry.isDirectory()) {
        stack.push(fullPath);
        continue;
      }

      const ext = path.extname(entry.name).toLowerCase();
      if (IMAGE_EXTENSIONS.has(ext)) {
        result.push(fullPath);
      }
    }
  }

  return result.sort();
}

function safeStat(p: string): fs.Stats | null {
  try {
    return fs.statSync(p);
  } catch {
    return null;
  }
}

function assertImageExists(imagePath: string): void {
  if (!fs.existsSync(imagePath)) {
    throw new Error(`File not found: ${imagePath}`);
  }
  if (!fs.statSync(imagePath).isFile()) {
    throw new Error(`Not a file: ${imagePath}`);
  }
}

function printUsage(): void {
  console.log(`
  pixmap — local image similarity search

  Usage:
    pixmap add <image|dir> [...]     Add image(s) or directory to the index
    pixmap search <image> [-k N]     Find similar indexed images
    pixmap show <image|id>           Preview an image in the terminal
    pixmap list                      Show all indexed images
    pixmap status                    Show index stats

  Options:
    -d, --data-dir <path>   Data directory (default: ./data)
    -k, --top-k <number>    Number of results (default: ${DEFAULT_TOP_K})
    -h, --help              Show this help

  Examples:
    pixmap add photo.jpg
    pixmap add ./photos/
    pixmap search query.png -k 10
    pixmap show 3
    pixmap list
`);
}

main().catch((error: unknown) => {
  const message = error instanceof Error ? error.message : String(error);
  console.error(`Error: ${message}`);
  process.exitCode = 1;
});
