/**
 * The facade: one corpus, two retrievers, one ranking.
 *
 * Query path — run BM25 and cosine search independently over the same corpus,
 * each returning a candidate pool deeper than the requested page, then fuse the
 * two rankings and cut to `limit`. Over-fetching matters: a document ranked 12th
 * by both retrievers deserves to surface in a top-5 hybrid list, and it only can
 * if both pools reach that far down.
 */

import type { Embedder } from './embedder';
import {
  KeywordIndex,
  type KeywordDocument,
  type KeywordIndexOptions,
  type ScoredId,
} from './keywordIndex';
import { VectorIndex } from './vectorIndex';
import {
  reciprocalRankFusion,
  weightedFusion,
  type FusedResult,
  type NamedRanking,
  type NormalizationStrategy,
} from './hybrid';

export const RETRIEVER = { KEYWORD: 'keyword', SEMANTIC: 'semantic' } as const;
export type Retriever = (typeof RETRIEVER)[keyof typeof RETRIEVER];

export type SearchMode = 'hybrid' | 'keyword' | 'semantic';
export type FusionStrategy = 'weighted' | 'rrf';

export interface HybridSearchOptions {
  embed: Embedder;
  keyword?: KeywordIndexOptions;
  /** Defaults applied to every `search()` call; overridable per query. */
  defaults?: Omit<QueryOptions, 'limit'>;
}

export interface QueryOptions {
  limit?: number;
  /** `hybrid` (default) runs both retrievers; the others are useful for A/B. */
  mode?: SearchMode;
  /** How to combine the two rankings. Default `rrf`. */
  fusion?: FusionStrategy;
  /** Relative influence per retriever. Default 1 each. */
  weights?: Partial<Record<Retriever, number>>;
  /** Score normalisation for `weighted` fusion. Default `minmax`. */
  normalize?: NormalizationStrategy;
  /** Rank damping for `rrf` fusion. Default 60. */
  rrfK?: number;
  /** Candidates pulled from each retriever before fusion. Default max(5·limit, 50). */
  candidates?: number;
}

export interface HybridResult<T extends KeywordDocument> extends FusedResult {
  document: T;
}

const DEFAULT_QUERY: Required<Omit<QueryOptions, 'candidates' | 'weights'>> & {
  weights: Partial<Record<Retriever, number>>;
} = {
  limit: 10,
  mode: 'hybrid',
  fusion: 'rrf',
  normalize: 'minmax',
  rrfK: 60,
  weights: {},
};

export class HybridSearchEngine<T extends KeywordDocument = KeywordDocument> {
  readonly keywordIndex: KeywordIndex;
  readonly vectorIndex = new VectorIndex();

  private readonly embed: Embedder;
  private readonly documents = new Map<string, T>();
  private readonly defaults: Omit<QueryOptions, 'limit'>;

  constructor({ embed, keyword, defaults = {} }: HybridSearchOptions) {
    if (typeof embed !== 'function') throw new TypeError('embed must be a function');
    this.embed = embed;
    this.keywordIndex = new KeywordIndex(keyword);
    this.defaults = defaults;
  }

  get size(): number {
    return this.documents.size;
  }

  get(id: string): T | undefined {
    return this.documents.get(id);
  }

  /**
   * Index (or re-index) documents. Embeddings are requested in a single call so
   * the {@link Embedder} can batch them; keyword indexing is synchronous.
   */
  async index(documents: readonly T[]): Promise<void> {
    if (documents.length === 0) return;

    const vectors = await this.embed(documents.map(({ text }) => text ?? ''));
    if (vectors.length !== documents.length) {
      throw new Error(
        `Embedder returned ${vectors.length} vectors for ${documents.length} documents`,
      );
    }

    documents.forEach((document, i) => {
      this.keywordIndex.add(document);
      this.vectorIndex.add(document.id, vectors[i]);
      this.documents.set(document.id, document);
    });
  }

  remove(id: string): boolean {
    this.vectorIndex.remove(id);
    this.keywordIndex.remove(id);
    return this.documents.delete(id);
  }

  async search(query: string, options: QueryOptions = {}): Promise<HybridResult<T>[]> {
    const {
      limit,
      mode,
      fusion,
      normalize,
      rrfK,
      weights,
      candidates = Math.max(limit * 5, 50),
    } = { ...DEFAULT_QUERY, ...this.defaults, ...options };

    if (!query?.trim() || this.documents.size === 0) return [];

    const wantsKeyword = mode !== 'semantic';
    const wantsSemantic = mode !== 'keyword';

    const keywordResults = wantsKeyword
      ? this.keywordIndex.search(query, { limit: candidates })
      : [];
    const semanticResults = wantsSemantic ? await this.searchSemantic(query, candidates) : [];

    if (!wantsSemantic) return this.hydrate(single(RETRIEVER.KEYWORD, keywordResults, limit));
    if (!wantsKeyword) return this.hydrate(single(RETRIEVER.SEMANTIC, semanticResults, limit));

    const rankings: NamedRanking[] = [
      {
        name: RETRIEVER.KEYWORD,
        results: keywordResults,
        weight: weights[RETRIEVER.KEYWORD] ?? 1,
      },
      {
        name: RETRIEVER.SEMANTIC,
        results: semanticResults,
        weight: weights[RETRIEVER.SEMANTIC] ?? 1,
      },
    ];

    const fused =
      fusion === 'rrf'
        ? reciprocalRankFusion(rankings, { k: rrfK, limit })
        : weightedFusion(rankings, { normalize, limit });

    return this.hydrate(fused);
  }

  private async searchSemantic(query: string, limit: number): Promise<ScoredId[]> {
    if (this.vectorIndex.size === 0) return [];
    const [vector] = await this.embed([query]);
    if (!vector) throw new Error('Embedder returned no vector for the query');
    return this.vectorIndex.search(vector, { limit });
  }

  /** Attach the source documents, skipping any id removed mid-flight. */
  private hydrate(results: readonly FusedResult[]): HybridResult<T>[] {
    const hydrated: HybridResult<T>[] = [];
    for (const result of results) {
      const document = this.documents.get(result.id);
      if (document) hydrated.push({ ...result, document });
    }
    return hydrated;
  }
}

/** Single-retriever mode keeps raw scores — there is nothing to make comparable. */
function single(name: Retriever, results: readonly ScoredId[], limit: number): FusedResult[] {
  return results.slice(0, limit).map((result, index) => ({
    id: result.id,
    score: result.score,
    components: {
      [name]: { rawScore: result.score, fusedScore: result.score, rank: index + 1 },
    },
  }));
}

/** Convenience constructor, mirroring the module's other factory functions. */
export function createHybridSearchEngine<T extends KeywordDocument = KeywordDocument>(
  options: HybridSearchOptions,
): HybridSearchEngine<T> {
  return new HybridSearchEngine<T>(options);
}
