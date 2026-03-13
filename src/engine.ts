import fs from 'node:fs';
import path from 'node:path';
import { DEFAULT_TOP_K, ImageRecord, SearchResult } from './types.js';

type EmbedderInstance = {
  init: () => Promise<void>;
  embed: (imagePath: string) => Promise<Float32Array>;
};

type StoreInstance = {
  initOrLoad: (indexPath: string) => void;
  save: (indexPath: string) => void;
  getCount: () => number;
};

type DbInstance = {
  get: (id: number) => ImageRecord | undefined;
  list: (limit?: number) => ImageRecord[];
};

export type PixmapEngineOptions = {
  dataDir: string;
  indexFileName?: string;
  dbFileName?: string;
};

export class PixmapEngine {
  private readonly dataDir: string;
  private readonly indexPath: string;
  private readonly dbPath: string;

  private embedder: EmbedderInstance | null = null;
  private store: StoreInstance | null = null;
  private db: DbInstance | null = null;

  constructor(options: PixmapEngineOptions) {
    this.dataDir = path.resolve(options.dataDir);
    this.indexPath = path.join(this.dataDir, options.indexFileName ?? 'index.hnsw');
    this.dbPath = path.join(this.dataDir, options.dbFileName ?? 'metadata.db');
  }

  async init(): Promise<void> {
    fs.mkdirSync(this.dataDir, { recursive: true });

    const [{ ImageEmbedder }, { VectorStore }, { MetadataDb }] = await Promise.all([
      import('./embedder.js'),
      import('./vectorStore.js'),
      import('./metadataDb.js'),
    ]);

    this.embedder = new ImageEmbedder();
    await this.embedder.init();

    this.store = new VectorStore();
    this.store.initOrLoad(this.indexPath);

    this.db = new MetadataDb(this.dbPath);
  }

  async add(imagePath: string): Promise<{ id: number; skipped: boolean; record?: ImageRecord }> {
    this.assertInitialized();

    const absolute = path.resolve(imagePath);
    const { addImage } = await import('./indexer.js');
    const result = await (addImage as any)(absolute, this.embedder!, this.store!, this.db!);
    this.store!.save(this.indexPath);

    return {
      ...result,
      record: this.db!.get(result.id),
    };
  }

  async search(queryImagePath: string, topK = DEFAULT_TOP_K): Promise<SearchResult[]> {
    this.assertInitialized();
    const absolute = path.resolve(queryImagePath);

    if (this.store!.getCount() === 0) {
      return [];
    }

    const { findSimilar } = await import('./searcher.js');
    return (findSimilar as any)(absolute, this.embedder!, this.store!, this.db!, topK);
  }

  listImages(limit = 200): ImageRecord[] {
    this.assertInitialized();
    return this.db!.list(limit);
  }

  getImage(id: number): ImageRecord | undefined {
    this.assertInitialized();
    return this.db!.get(id);
  }

  getIndexedCount(): number {
    this.assertInitialized();
    return this.store!.getCount();
  }

  private assertInitialized(): void {
    if (!this.embedder || !this.store || !this.db) {
      throw new Error('PixmapEngine not initialized. Call init() first.');
    }
  }
}
