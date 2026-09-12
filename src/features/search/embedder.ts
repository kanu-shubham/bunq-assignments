/**
 * The embedding seam.
 *
 * Everything downstream (vector index, hybrid fusion) only knows the
 * {@link Embedder} type, so a real model is a one-line swap and tests stay
 * offline and deterministic — the same dependency-injection shape the feedback
 * widget uses for `submitFeedback`.
 */

import { tokenize as defaultTokenize, fold, type Tokenizer } from './tokenize';

/** Maps texts to dense vectors. Order of the output matches the input. */
export type Embedder = (texts: readonly string[]) => Promise<number[][]>;

export function l2Normalize(vector: readonly number[]): number[] {
  let squared = 0;
  for (const value of vector) squared += value * value;
  const norm = Math.sqrt(squared);
  if (norm === 0) return vector.slice();
  return vector.map((value) => value / norm);
}

/** FNV-1a, 32-bit. Cheap, well-spread, and no dependency. */
function fnv1a(text: string, seed: number): number {
  let hash = 0x811c9dc5 ^ seed;
  for (let i = 0; i < text.length; i += 1) {
    hash ^= text.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193);
  }
  return hash >>> 0;
}

export interface HashingEmbedderOptions {
  /** Vector width. Larger means fewer hash collisions. Default 256. */
  dimensions?: number;
  seed?: number;
  /**
   * Character n-gram range mixed in alongside whole tokens, which is what buys
   * robustness to typos and unseen morphology. `false` disables them.
   */
  charNgrams?: { min: number; max: number } | false;
  /** Weight of character n-grams relative to whole tokens. Default 0.5. */
  charNgramWeight?: number;
  tokenize?: Tokenizer;
}

/**
 * A deterministic, offline stand-in for a real embedding model: the hashing
 * trick (signed feature hashing) over tokens and character n-grams.
 *
 * Be clear about what this is. It captures *lexical* similarity — shared
 * subwords, typos, morphology — not meaning: "car" and "automobile" stay far
 * apart. It exists so the pipeline is runnable and the tests are deterministic
 * with no network. For real semantic recall, pass an API-backed
 * {@link Embedder} instead; nothing else in the module changes.
 */
export function createHashingEmbedder({
  dimensions = 256,
  seed = 0,
  charNgrams = { min: 3, max: 4 },
  charNgramWeight = 0.5,
  tokenize = defaultTokenize,
}: HashingEmbedderOptions = {}): Embedder {
  if (dimensions <= 0) throw new RangeError('dimensions must be > 0');

  const embedOne = (text: string): number[] => {
    const vector = new Array<number>(dimensions).fill(0);

    const addFeature = (feature: string, weight: number): void => {
      const bucket = fnv1a(feature, seed) % dimensions;
      // A second, independent hash picks the sign: collisions then cancel out
      // in expectation instead of always inflating the bucket.
      const sign = fnv1a(feature, seed + 0x9e3779b9) & 1 ? 1 : -1;
      vector[bucket] += sign * weight;
    };

    const counts = new Map<string, number>();
    for (const token of tokenize(text)) counts.set(token, (counts.get(token) ?? 0) + 1);

    for (const [token, count] of counts) {
      const weight = 1 + Math.log(count); // sublinear TF, as in the keyword index
      addFeature(`w:${token}`, weight);

      if (!charNgrams) continue;
      const padded = `^${token}$`;
      for (let n = charNgrams.min; n <= charNgrams.max; n += 1) {
        for (let i = 0; i + n <= padded.length; i += 1) {
          addFeature(`c:${padded.slice(i, i + n)}`, weight * charNgramWeight);
        }
      }
    }

    return l2Normalize(vector);
  };

  return async (texts) => texts.map((text) => embedOne(fold(text)));
}

export interface RemoteEmbedderOptions {
  endpoint: string;
  model?: string;
  apiKey?: string;
  fetchImpl?: typeof fetch;
  /** Texts per HTTP request. Default 64. */
  batchSize?: number;
  /** Pull vectors out of a provider-specific response body. */
  parse?: (body: unknown) => number[][];
}

function defaultParse(body: unknown): number[][] {
  // The de-facto shape: { data: [{ embedding: number[] }, ...] }.
  const data = (body as { data?: Array<{ embedding?: number[] }> })?.data;
  if (!Array.isArray(data)) throw new Error('Embedding response missing "data" array');
  return data.map(({ embedding }) => {
    if (!Array.isArray(embedding)) throw new Error('Embedding response entry missing "embedding"');
    return embedding;
  });
}

/**
 * Adapter for a hosted embedding API, batched so one `index()` call does not
 * become one request per document.
 */
export function createRemoteEmbedder({
  endpoint,
  model,
  apiKey,
  fetchImpl = fetch,
  batchSize = 64,
  parse = defaultParse,
}: RemoteEmbedderOptions): Embedder {
  if (batchSize <= 0) throw new RangeError('batchSize must be > 0');

  return async (texts) => {
    const vectors: number[][] = [];

    for (let offset = 0; offset < texts.length; offset += batchSize) {
      const batch = texts.slice(offset, offset + batchSize);
      const response = await fetchImpl(endpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(apiKey ? { Authorization: `Bearer ${apiKey}` } : {}),
        },
        body: JSON.stringify(model ? { model, input: batch } : { input: batch }),
      });

      if (!response.ok) throw new Error(`Embedding request failed: ${response.status}`);

      const parsed = parse(await response.json());
      if (parsed.length !== batch.length) {
        throw new Error(`Embedding response size mismatch: expected ${batch.length}, got ${parsed.length}`);
      }
      vectors.push(...parsed);
    }

    return vectors;
  };
}
