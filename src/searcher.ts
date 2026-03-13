import { ImageEmbedder } from './embedder.js';
import { MetadataDb } from './metadataDb.js';
import { DEFAULT_TOP_K, SearchResult } from './types.js';
import { VectorStore } from './vectorStore.js';

export async function findSimilar(
  queryImagePath: string,
  embedder: ImageEmbedder,
  store: VectorStore,
  db: MetadataDb,
  topK = DEFAULT_TOP_K
): Promise<SearchResult[]> {
  const queryVec = await embedder.embed(queryImagePath);
  const hits = store.search(queryVec, topK);

  const results: SearchResult[] = [];
  for (const hit of hits) {
    const metadata = db.get(hit.id);
    if (!metadata) {
      continue;
    }

    results.push({
      ...metadata,
      score: hit.score,
    });
  }

  return results;
}
