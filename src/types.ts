export const VECTOR_DIMENSIONS = 512;
export const DEFAULT_TOP_K = 5;
export const DEFAULT_MAX_ELEMENTS = 100_000;

export type SimilarityHit = {
  id: number;
  score: number;
};

export type ImageRecord = {
  id: number;
  path: string;
  created: number;
  indexed: number;
};

export type SearchResult = ImageRecord & {
  score: number;
};
