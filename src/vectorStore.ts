import fs from 'node:fs';
import hnswlib from 'hnswlib-node';
import { DEFAULT_MAX_ELEMENTS, SimilarityHit, VECTOR_DIMENSIONS } from './types.js';

const { HierarchicalNSW } = hnswlib;

export class VectorStore {
  private index: any;
  private readonly dim: number;
  private maxElements: number;

  constructor(dim = VECTOR_DIMENSIONS, maxElements = DEFAULT_MAX_ELEMENTS) {
    this.dim = dim;
    this.maxElements = maxElements;
    this.index = new HierarchicalNSW('cosine', dim);
  }

  initOrLoad(indexPath: string): void {
    if (fs.existsSync(indexPath)) {
      this.index.readIndexSync(indexPath);
      return;
    }

    this.index.initIndex(this.maxElements);
  }

  add(id: number, vector: Float32Array): void {
    this.ensureCapacity();
    this.index.addPoint(Array.from(vector), id);
  }

  search(queryVector: Float32Array, topK: number): SimilarityHit[] {
    const result = this.index.searchKnn(Array.from(queryVector), topK);

    return result.neighbors.map((id: number, i: number) => ({
      id,
      score: 1 - result.distances[i],
    }));
  }

  save(path: string): void {
    this.index.writeIndex(path);
  }

  getCount(): number {
    return this.index.getCurrentCount();
  }

  private ensureCapacity(): void {
    const count = this.index.getCurrentCount();
    if (count < this.maxElements) {
      return;
    }

    const next = Math.ceil(this.maxElements * 1.5);
    this.index.resizeIndex(next);
    this.maxElements = next;
  }
}
