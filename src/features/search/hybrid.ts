/**
 * Fusing keyword and embedding results.
 *
 * The hard part is not running both retrievers, it is that their scores are not
 * comparable. BM25 is unbounded and corpus-dependent (a 14 here, a 3 there);
 * cosine similarity lives in [-1, 1] and, for most models, clusters in a narrow
 * band near the top. Adding them raw lets BM25 silently dominate.
 *
 * Two answers, both here:
 *
 *   • {@link weightedFusion} — normalise each list to a common range per query,
 *     then take a weighted sum. Keeps score *margins* (a runaway top hit still
 *     looks like a runaway top hit), but is sensitive to outliers, and the
 *     normalisation is per-query so scores are not comparable across queries.
 *
 *   • {@link reciprocalRankFusion} — throw the scores away and combine ranks as
 *     Σ w / (k + rank). Scale-free, needs no tuning, and is hard to beat as a
 *     default; the cost is that it cannot tell a landslide from a photo finish.
 *
 * A document found by only one retriever is scored as absent from the other
 * (0 contribution), so agreement between retrievers is rewarded — which is the
 * behaviour you want from a hybrid.
 */

import { sortByScore, type ScoredId } from './keywordIndex';

export type NormalizationStrategy = 'minmax' | 'zscore' | 'none';

/**
 * Map scores onto [0, 1] by min/max within the list.
 *
 * When every score is identical the range is undefined; they are all equally
 * good within this list, so they all get 1.
 */
export function minMaxNormalize(results: readonly ScoredId[]): ScoredId[] {
  if (results.length === 0) return [];
  let min = Infinity;
  let max = -Infinity;
  for (const { score } of results) {
    if (score < min) min = score;
    if (score > max) max = score;
  }
  const range = max - min;
  return results.map(({ id, score }) => ({
    id,
    score: range === 0 ? 1 : (score - min) / range,
  }));
}

/** Centre on the mean and scale by the standard deviation. Outlier-tolerant. */
export function zScoreNormalize(results: readonly ScoredId[]): ScoredId[] {
  if (results.length === 0) return [];
  const mean = results.reduce((sum, { score }) => sum + score, 0) / results.length;
  const variance =
    results.reduce((sum, { score }) => sum + (score - mean) ** 2, 0) / results.length;
  const deviation = Math.sqrt(variance);
  return results.map(({ id, score }) => ({
    id,
    score: deviation === 0 ? 0 : (score - mean) / deviation,
  }));
}

export function normalizeScores(
  results: readonly ScoredId[],
  strategy: NormalizationStrategy,
): ScoredId[] {
  switch (strategy) {
    case 'minmax':
      return minMaxNormalize(results);
    case 'zscore':
      return zScoreNormalize(results);
    case 'none':
      return results.map(({ id, score }) => ({ id, score }));
    default: {
      const exhaustive: never = strategy;
      throw new TypeError(`Unknown normalization strategy: ${String(exhaustive)}`);
    }
  }
}

/** One retriever's contribution to a fused result. */
export interface Contribution {
  /** Score as the retriever reported it, before normalisation. */
  rawScore: number;
  /** Score as it entered the sum (normalised, or the RRF reciprocal). */
  fusedScore: number;
  /** 1-based position in that retriever's own ranking. */
  rank: number;
}

export interface FusedResult extends ScoredId {
  /** Per-retriever breakdown, keyed by the name given in the input. */
  components: Record<string, Contribution>;
}

/** A named ranking to fuse, already sorted best-first. */
export interface NamedRanking {
  name: string;
  results: readonly ScoredId[];
  /** Relative influence. Default 1. */
  weight?: number;
}

export interface WeightedFusionOptions {
  normalize?: NormalizationStrategy;
  limit?: number;
}

/** Weighted sum of per-list normalised scores. */
export function weightedFusion(
  rankings: readonly NamedRanking[],
  { normalize = 'minmax', limit = 10 }: WeightedFusionOptions = {},
): FusedResult[] {
  return fuse(rankings, limit, (results) => {
    const normalized = normalizeScores(results, normalize);
    return (index: number) => normalized[index].score;
  });
}

export interface RrfOptions {
  /**
   * Rank damping. Small k sharpens the advantage of the top few positions;
   * the conventional 60 comes from the original RRF paper and is a sane default.
   */
  k?: number;
  limit?: number;
}

/** Reciprocal rank fusion: Σ weight / (k + rank). */
export function reciprocalRankFusion(
  rankings: readonly NamedRanking[],
  { k = 60, limit = 10 }: RrfOptions = {},
): FusedResult[] {
  if (k <= 0) throw new RangeError('k must be > 0');
  return fuse(rankings, limit, () => (index: number) => 1 / (k + index + 1));
}

/** Shared plumbing: per-list contribution function → weighted sum → sorted. */
function fuse(
  rankings: readonly NamedRanking[],
  limit: number,
  contributionFor: (results: readonly ScoredId[]) => (index: number) => number,
): FusedResult[] {
  const merged = new Map<string, FusedResult>();

  for (const { name, results, weight = 1 } of rankings) {
    const contribution = contributionFor(results);

    results.forEach((result, index) => {
      const fusedScore = weight * contribution(index);
      let entry = merged.get(result.id);
      if (!entry) {
        entry = { id: result.id, score: 0, components: {} };
        merged.set(result.id, entry);
      }
      entry.score += fusedScore;
      entry.components[name] = { rawScore: result.score, fusedScore, rank: index + 1 };
    });
  }

  return sortByScore([...merged.values()]).slice(0, limit);
}
