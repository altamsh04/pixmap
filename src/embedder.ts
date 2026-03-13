import sharp from 'sharp';
import { RawImage, pipeline } from '@xenova/transformers';
import { VECTOR_DIMENSIONS } from './types.js';

type Extractor = (input: unknown) => Promise<unknown>;

export class ImageEmbedder {
  private extractor: Extractor | null = null;

  async init(): Promise<void> {
    this.extractor = (await pipeline(
      'image-feature-extraction',
      'Xenova/clip-vit-base-patch32'
    )) as Extractor;
  }

  async embed(imagePath: string): Promise<Float32Array> {
    if (!this.extractor) {
      throw new Error('ImageEmbedder is not initialized. Call init() first.');
    }

    const preprocessed = await sharp(imagePath)
      .resize(224, 224, { fit: 'cover' })
      .removeAlpha()
      .toFormat('png')
      .toBuffer();

    const image = await this.bufferToRawImage(preprocessed);
    const output = await this.extractor(image);
    const vector = this.extractVector(output);

    if (vector.length !== VECTOR_DIMENSIONS) {
      throw new Error(
        `Unexpected embedding dimensions: ${vector.length}. Expected ${VECTOR_DIMENSIONS}.`
      );
    }

    return this.l2Normalize(vector);
  }

  private async bufferToRawImage(buffer: Buffer): Promise<unknown> {
    const raw = RawImage as unknown as {
      fromBlob?: (blob: Blob) => Promise<unknown>;
      fromBuffer?: (b: Buffer) => Promise<unknown>;
      read?: (b: Buffer) => Promise<unknown>;
    };

    if (typeof raw.fromBlob === 'function') {
      const view = Uint8Array.from(buffer);
      return raw.fromBlob(new Blob([view], { type: 'image/png' }));
    }
    if (typeof raw.fromBuffer === 'function') {
      return raw.fromBuffer(buffer);
    }
    if (typeof raw.read === 'function') {
      return raw.read(buffer);
    }

    throw new Error('No compatible RawImage constructor found in @xenova/transformers.');
  }

  private extractVector(output: unknown): Float32Array {
    let data: number[] | Float32Array | undefined;

    if (output instanceof Float32Array) {
      data = output;
    } else if (Array.isArray(output) && output.every((x) => typeof x === 'number')) {
      data = output as number[];
    } else if (typeof output === 'object' && output !== null) {
      const maybeObject = output as { data?: number[] | Float32Array };
      if (maybeObject.data) {
        data = maybeObject.data;
      } else if (Array.isArray(output)) {
        const first = output[0] as { data?: number[] | Float32Array } | undefined;
        data = first?.data;
      }
    }

    if (!data) {
      throw new Error('Failed to extract vector data from model output.');
    }

    return data instanceof Float32Array ? data : new Float32Array(data);
  }

  private l2Normalize(vector: Float32Array): Float32Array {
    let sumSquares = 0;
    for (let i = 0; i < vector.length; i += 1) {
      sumSquares += vector[i] * vector[i];
    }

    const norm = Math.sqrt(sumSquares) || 1;
    const normalized = new Float32Array(vector.length);
    for (let i = 0; i < vector.length; i += 1) {
      normalized[i] = vector[i] / norm;
    }

    return normalized;
  }
}
