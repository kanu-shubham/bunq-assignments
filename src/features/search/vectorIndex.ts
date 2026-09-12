/**
 * Dense retrieval: exhaustive cosine similarity over stored embeddings.
 *
 * Vectors are L2-normalised on insert, which turns cosine similarity into a
 * plain dot product. Search is O(N · d) — fine up to the tens of thousands of
 * documents this widget-scale corpus implies, and the point where you would
 * reach for an ANN index (HNSW/IVF) rather than hand-roll one.
 */

import { l2Normalize } from './embedder';
import { sortByScore, type ScoredId } from './keywordIndex';

/** Cosine similarity of two equal-length vectors, in [-1, 1]. */
export function cosineSimilarity(a: readonly number[], b: readonly number[]): number {
  if (a.length !== b.length) {
    throw new RangeError(`Vector length mismatch: ${a.length} vs ${b.length}`);
  }
  let dot = 0;
  let aSquared = 0;
  let bSquared = 0;
  for (let i = 0; i < a.length; i += 1) {
    dot += a[i] * b[i];
    aSquared += a[i] * a[i];
    bSquared += b[i] * b[i];
  }
  const denominator = Math.sqrt(aSquared) * Math.sqrt(bSquared);
  return denominator === 0 ? 0 : dot / denominator;
}

export interface VectorSearchOptions {
  limit?: number;
  /** Drop results below this cosine similarity. Default -Infinity. */
  minScore?: number;
}

export class VectorIndex {
  private readonly vectors = new Map<string, number[]>();
  private dimensions: number | null = null;

  get size(): number {
    return this.vectors.size;
  }

  /** Dimensionality fixed by the first insert; mismatches are a bug, not a warning. */
  get dimension(): number | null {
    return this.dimensions;
  }

  add(id: string, vector: readonly number[]): void {
    if (!id) throw new TypeError('vector id is required');
    if (vector.length === 0) throw new RangeError('vector must not be empty');
    if (this.dimensions === null) {
      this.dimensions = vector.length;
    } else if (vector.length !== this.dimensions) {
      throw new RangeError(
        `Vector dimension mismatch: index holds ${this.dimensions}, got ${vector.length}`,
      );
    }
    this.vectors.set(id, l2Normalize(vector));
  }

  addAll(entries: readonly { id: string; vector: readonly number[] }[]): void {
    for (const { id, vector } of entries) this.add(id, vector);
  }

  remove(id: string): boolean {
    return this.vectors.delete(id);
  }

  get(id: string): readonly number[] | undefined {
    return this.vectors.get(id);
  }

  search(
    query: readonly number[],
    { limit = 10, minScore = -Infinity }: VectorSearchOptions = {},
  ): ScoredId[] {
    if (this.vectors.size === 0) return [];
    if (this.dimensions !== null && query.length !== this.dimensions) {
      throw new RangeError(
        `Query dimension mismatch: index holds ${this.dimensions}, got ${query.length}`,
      );
    }
    const normalizedQuery = l2Normalize(query);

    const results: ScoredId[] = [];
    for (const [id, vector] of this.vectors) {
      let dot = 0;
      for (let i = 0; i < vector.length; i += 1) dot += vector[i] * normalizedQuery[i];
      if (dot >= minScore) results.push({ id, score: dot });
    }
    return sortByScore(results).slice(0, limit);
  }
}
