import { ImageEmbedder } from './embedder.js';
import { MetadataDb } from './metadataDb.js';
import { VectorStore } from './vectorStore.js';

export async function addImage(
  imagePath: string,
  embedder: ImageEmbedder,
  store: VectorStore,
  db: MetadataDb
): Promise<{ id: number; skipped: boolean }> {
  const record = db.upsert(imagePath);
  if (record.indexed) {
    return { id: record.id, skipped: true };
  }

  const vector = await embedder.embed(imagePath);
  store.add(record.id, vector);
  db.markIndexed(record.id);

  return { id: record.id, skipped: false };
}
