/**
 * Keyword retrieval from scratch: an in-memory inverted index scored with
 * either BM25 (default) or classic TF-IDF cosine.
 *
 * Both scorers share the same three intuitions:
 *   1. a term that occurs in few documents is more informative  (IDF)
 *   2. more occurrences mean more relevance, with diminishing returns (TF)
 *   3. a long document should not win just by being long   (length norm)
 *
 * They differ in how 2 and 3 are shaped. TF-IDF damps term frequency with a
 * logarithm and divides by the document's vector norm — so every term's weight
 * depends on every other term in the document, and neither knob is tunable.
 * BM25 saturates term frequency towards an explicit ceiling of `k1 + 1` and
 * interpolates length normalisation with `b`, which is why it is the better
 * default: the two behaviours you actually want to tune are two parameters
 * rather than an emergent property of the vector norm.
 *
 * Search cost is driven by the postings actually touched, not by corpus size:
 * only documents containing at least one query term are ever scored.
 */

import { tokenize as defaultTokenize, type Tokenizer } from './tokenize';

export interface KeywordDocument {
  id: string;
  text: string;
}

export interface ScoredId {
  id: string;
  score: number;
}

export type Scorer = 'bm25' | 'tfidf';

export interface KeywordIndexOptions {
  /**
   * Term-frequency saturation. 0 collapses TF to a presence bit; larger values
   * keep rewarding repeats for longer. Lucene's default, 1.2, is a good prior.
   */
  k1?: number;
  /**
   * Length normalisation, in [0, 1]. 0 ignores document length entirely,
   * 1 divides fully by the length ratio. 0.75 is the usual compromise.
   */
  b?: number;
  tokenize?: Tokenizer;
}

export interface SearchOptions {
  limit?: number;
  scorer?: Scorer;
  /** Drop results at or below this score. Default 0 (BM25 can score 0). */
  minScore?: number;
}

interface IndexedDocument {
  /** term → frequency in this document */
  terms: Map<string, number>;
  length: number;
  /** ‖(1 + ln tf)‖₂ — the `lnc` document norm used by the TF-IDF scorer. */
  tfidfNorm: number;
}

const DEFAULTS = { k1: 1.2, b: 0.75 } as const;

export class KeywordIndex {
  private readonly k1: number;
  private readonly b: number;
  private readonly tokenize: Tokenizer;

  /** term → (docId → term frequency). The inverted index. */
  private readonly postings = new Map<string, Map<string, number>>();
  private readonly documents = new Map<string, IndexedDocument>();
  private totalLength = 0;

  /** IDF depends on N and df, so it is memoised per corpus generation. */
  private idfCache = new Map<string, number>();

  constructor({ k1 = DEFAULTS.k1, b = DEFAULTS.b, tokenize = defaultTokenize }: KeywordIndexOptions = {}) {
    if (k1 < 0) throw new RangeError('k1 must be >= 0');
    if (b < 0 || b > 1) throw new RangeError('b must be within [0, 1]');
    this.k1 = k1;
    this.b = b;
    this.tokenize = tokenize;
  }

  get size(): number {
    return this.documents.size;
  }

  get averageDocumentLength(): number {
    return this.documents.size === 0 ? 0 : this.totalLength / this.documents.size;
  }

  get vocabularySize(): number {
    return this.postings.size;
  }

  /** Adding an existing id replaces it, so re-indexing an edited document is safe. */
  add({ id, text }: KeywordDocument): void {
    if (!id) throw new TypeError('document id is required');
    this.remove(id);

    const tokens = this.tokenize(text ?? '');
    const terms = new Map<string, number>();
    for (const token of tokens) terms.set(token, (terms.get(token) ?? 0) + 1);

    let squared = 0;
    for (const [term, tf] of terms) {
      let posting = this.postings.get(term);
      if (!posting) {
        posting = new Map();
        this.postings.set(term, posting);
      }
      posting.set(id, tf);
      const weight = 1 + Math.log(tf);
      squared += weight * weight;
    }

    this.documents.set(id, {
      terms,
      length: tokens.length,
      tfidfNorm: Math.sqrt(squared),
    });
    this.totalLength += tokens.length;
    this.idfCache.clear();
  }

  addAll(documents: readonly KeywordDocument[]): void {
    for (const document of documents) this.add(document);
  }

  remove(id: string): boolean {
    const existing = this.documents.get(id);
    if (!existing) return false;

    for (const term of existing.terms.keys()) {
      const posting = this.postings.get(term);
      if (!posting) continue;
      posting.delete(id);
      if (posting.size === 0) this.postings.delete(term); // keep df honest
    }
    this.documents.delete(id);
    this.totalLength -= existing.length;
    this.idfCache.clear();
    return true;
  }

  /** Number of documents containing `term` (the term must already be analyzed). */
  documentFrequency(term: string): number {
    return this.postings.get(term)?.size ?? 0;
  }

  /**
   * Lucene-flavoured probabilistic IDF: ln(1 + (N − df + 0.5) / (df + 0.5)).
   *
   * The outer `1 +` is what keeps it non-negative — the textbook form goes
   * negative for terms in more than half the corpus, which lets a common term
   * subtract score a document earned elsewhere.
   */
  idf(term: string): number {
    const cached = this.idfCache.get(term);
    if (cached !== undefined) return cached;

    const df = this.documentFrequency(term);
    const value = df === 0 ? 0 : Math.log(1 + (this.documents.size - df + 0.5) / (df + 0.5));
    this.idfCache.set(term, value);
    return value;
  }

  /** Smoothed IDF for the TF-IDF scorer: ln(1 + N / df). */
  private smoothIdf(term: string): number {
    const df = this.documentFrequency(term);
    return df === 0 ? 0 : Math.log(1 + this.documents.size / df);
  }

  search(query: string, { limit = 10, scorer = 'bm25', minScore = 0 }: SearchOptions = {}): ScoredId[] {
    const queryTerms = new Map<string, number>();
    for (const token of this.tokenize(query ?? '')) {
      queryTerms.set(token, (queryTerms.get(token) ?? 0) + 1);
    }
    if (queryTerms.size === 0 || this.documents.size === 0) return [];

    const scores =
      scorer === 'bm25' ? this.scoreBm25(queryTerms) : this.scoreTfIdf(queryTerms);

    const results: ScoredId[] = [];
    for (const [id, score] of scores) {
      if (score > minScore) results.push({ id, score });
    }
    return sortByScore(results).slice(0, limit);
  }

  /** Explain a single document's score — useful when tuning k1/b. */
  explain(query: string, id: string, scorer: Scorer = 'bm25'): Array<{ term: string; contribution: number }> {
    const document = this.documents.get(id);
    if (!document) return [];

    const seen = new Map<string, number>();
    for (const token of this.tokenize(query ?? '')) seen.set(token, (seen.get(token) ?? 0) + 1);

    const queryNorm = scorer === 'tfidf' ? this.tfidfQueryNorm(seen) : 1;

    const parts: Array<{ term: string; contribution: number }> = [];
    for (const [term, qtf] of seen) {
      const tf = document.terms.get(term) ?? 0;
      if (tf === 0) continue;
      const contribution =
        scorer === 'bm25'
          ? qtf * this.idf(term) * this.saturate(tf, document.length)
          : this.tfidfContribution(term, qtf, tf, document.tfidfNorm, queryNorm);
      parts.push({ term, contribution });
    }
    return parts.sort((a, b) => b.contribution - a.contribution);
  }

  /** tf · (k1 + 1) / (tf + k1 · (1 − b + b · dl/avgdl)) */
  private saturate(tf: number, documentLength: number): number {
    const avgdl = this.averageDocumentLength || 1;
    const norm = 1 - this.b + this.b * (documentLength / avgdl);
    return (tf * (this.k1 + 1)) / (tf + this.k1 * norm);
  }

  private scoreBm25(queryTerms: Map<string, number>): Map<string, number> {
    const scores = new Map<string, number>();

    for (const [term, qtf] of queryTerms) {
      const posting = this.postings.get(term);
      if (!posting) continue;
      const idf = this.idf(term);
      if (idf === 0) continue;

      for (const [id, tf] of posting) {
        const document = this.documents.get(id);
        if (!document) continue;
        const contribution = qtf * idf * this.saturate(tf, document.length);
        scores.set(id, (scores.get(id) ?? 0) + contribution);
      }
    }
    return scores;
  }

  private tfidfQueryNorm(queryTerms: Map<string, number>): number {
    let squared = 0;
    for (const [term, qtf] of queryTerms) {
      const weight = (1 + Math.log(qtf)) * this.smoothIdf(term);
      squared += weight * weight;
    }
    return Math.sqrt(squared) || 1;
  }

  private tfidfContribution(
    term: string,
    qtf: number,
    tf: number,
    documentNorm: number,
    queryNorm: number,
  ): number {
    if (documentNorm === 0) return 0;
    const queryWeight = ((1 + Math.log(qtf)) * this.smoothIdf(term)) / queryNorm; // ltc
    const documentWeight = (1 + Math.log(tf)) / documentNorm;                      // lnc
    return queryWeight * documentWeight;
  }

  /**
   * `ltc.lnc` cosine: log-damped TF with IDF on the query side, log-damped TF
   * with cosine normalisation on the document side.
   */
  private scoreTfIdf(queryTerms: Map<string, number>): Map<string, number> {
    const scores = new Map<string, number>();

    const queryWeights = new Map<string, number>();
    for (const [term, qtf] of queryTerms) {
      const weight = (1 + Math.log(qtf)) * this.smoothIdf(term);
      if (weight === 0) continue;
      queryWeights.set(term, weight);
    }
    const queryNorm = this.tfidfQueryNorm(queryTerms);

    for (const [term, queryWeight] of queryWeights) {
      const posting = this.postings.get(term);
      if (!posting) continue;

      for (const [id, tf] of posting) {
        const document = this.documents.get(id);
        if (!document || document.tfidfNorm === 0) continue;
        const contribution =
          (queryWeight / queryNorm) * ((1 + Math.log(tf)) / document.tfidfNorm);
        scores.set(id, (scores.get(id) ?? 0) + contribution);
      }
    }
    return scores;
  }
}

/** Descending by score, ties broken by id so results are deterministic. */
export function sortByScore<T extends ScoredId>(results: T[]): T[] {
  return results.sort((a, b) => (b.score - a.score) || a.id.localeCompare(b.id));
}
