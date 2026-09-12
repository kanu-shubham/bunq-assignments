# Hybrid search

Keyword retrieval (BM25 / TF-IDF) and dense retrieval (embeddings + cosine),
implemented from scratch with no search dependency, then fused into one ranking.

```
                 ┌──────────────┐  BM25 over an inverted index
  query ────────▶│ KeywordIndex │──────────────┐
     │           └──────────────┘              │  top-N candidates
     │                                         ▼
     │           ┌──────────────┐        ┌──────────┐
     └──embed───▶│ VectorIndex  │───────▶│  fusion  │────▶ ranked results
                 └──────────────┘        └──────────┘      + per-retriever
                  cosine similarity       RRF | weighted      breakdown
```

| File | What it holds |
| --- | --- |
| `tokenize.ts` | fold → stopwords → light stemmer; the analyzer both sides share |
| `keywordIndex.ts` | inverted index, BM25 and TF-IDF scorers, `explain()` |
| `embedder.ts` | the `Embedder` seam, an offline hashing embedder, a remote-API adapter |
| `vectorIndex.ts` | cosine search over L2-normalised vectors |
| `hybrid.ts` | score normalisation, weighted fusion, reciprocal rank fusion |
| `hybridSearch.ts` | the facade that owns both indexes and runs a query end to end |

## Usage

```ts
import { createHybridSearchEngine, createHashingEmbedder } from './features/search';

const engine = createHybridSearchEngine({
  embed: createHashingEmbedder(),        // swap in a real model, see below
  keyword: { k1: 1.2, b: 0.75 },
});

await engine.index([
  { id: 'a', text: 'A guide to cycling: pedal technique and saddle height' },
  { id: 'b', text: 'The bike lane on the bridge is closed for repairs' },
]);

const results = await engine.search('bicycle repairs', { limit: 5 });
// [{ id, score, document, components: { keyword: {...}, semantic: {...} } }, ...]
```

Every result carries the per-retriever breakdown — raw score, fused score and
rank — so a ranking can be explained rather than guessed at. For the keyword
side alone, `KeywordIndex.explain(query, id)` decomposes a score term by term.

Useful knobs on `search()`: `mode` (`hybrid` | `keyword` | `semantic`, handy for
A/B'ing the two retrievers), `fusion`, `weights`, `normalize`, `rrfK`,
`candidates`.

## Why BM25

Both scorers are built on the same three intuitions — rare terms are
informative (IDF), repetition means relevance but with diminishing returns (TF),
and length should not buy rank. They differ in the shape of the last two:

```
BM25(q, d) = Σ  idf(t) · ─────────tf · (k1 + 1)──────────
            t∈q                tf + k1 · (1 − b + b · dl/avgdl)

idf(t) = ln(1 + (N − df + 0.5) / (df + 0.5))
```

* **`k1` — saturation.** A term's contribution approaches `idf · (k1 + 1)` and
  stops. The 30th occurrence of "bike" cannot buy what the 3rd already did;
  `k1 = 0` collapses TF to a presence bit.
* **`b` — length normalisation.** Interpolates between ignoring length (`b = 0`)
  and dividing fully by `dl/avgdl` (`b = 1`).
* **That `1 +` inside the IDF.** The textbook form goes negative for terms
  appearing in more than half the corpus, which lets a common term *subtract*
  score a document earned elsewhere. Lucene's variant keeps it non-negative.

The TF-IDF scorer (`scorer: 'tfidf'`, an `ltc.lnc` cosine) is included for
comparison. It damps TF logarithmically and divides by the document's vector
norm, so each term's weight depends on every other term in the document, and
neither behaviour is tunable. BM25 turns both into parameters — which is the
reason it, not TF-IDF, is the default.

Search touches only the postings of terms that actually occur in the query, so
cost tracks the query's selectivity rather than corpus size.

## Why hybrid, and how the scores are combined

The two retrievers fail in opposite directions. BM25 cannot match "bicycle"
against a document that only ever says "cycling"; embeddings routinely miss an
exact identifier, product code or rare proper noun that appears verbatim. Run
both and you cover both failure modes — but the scores are not comparable:
BM25 is unbounded and corpus-dependent, cosine sits in `[-1, 1]` and, for most
models, in a narrow band near the top of it. Summing them raw lets BM25 decide
everything.

Two ways out, both implemented:

**Reciprocal rank fusion** (default) — `Σ weight / (k + rank)`. Throws the scores
away and combines positions, so nothing needs calibrating and no retriever can
swamp the other. The cost is that it cannot tell a landslide from a photo
finish: rank 1 by a mile and rank 1 by a hair score identically.

**Weighted fusion** — normalise each list (`minmax` or `zscore`), then take a
weighted sum. Keeps the margins RRF discards, at the price of sensitivity to
outliers and per-query normalisation (fused scores are not comparable across
queries). `zscore` is the sturdier choice when one runaway hit would otherwise
compress everything below it.

A document found by only one retriever contributes nothing from the other, so
agreement between the two is rewarded — which is the behaviour you want from a
hybrid. Both retrievers are asked for a deeper candidate pool than the page
being returned (`candidates`, default `max(5 · limit, 50)`): a document ranked
12th by both deserves to surface in a top-5 hybrid list, and it only can if both
pools reach that far down.

## Plugging in a real embedding model

Everything downstream depends only on the `Embedder` type
(`(texts: readonly string[]) => Promise<number[][]>`) — the same
dependency-injection seam the feedback widget uses for `submitFeedback`.

`createHashingEmbedder()` is the offline default: signed feature hashing over
tokens and character n-grams. Be clear about what it is — it captures *lexical*
similarity (shared subwords, typos, morphology), not meaning. "car" and
"automobile" stay far apart. It exists so the pipeline runs and the tests stay
deterministic with no network.

For real semantic recall, pass an API-backed embedder instead; nothing else
changes:

```ts
const engine = createHybridSearchEngine({
  embed: createRemoteEmbedder({
    endpoint: 'https://api.example.com/v1/embeddings',
    model: 'embed-v1',
    apiKey: process.env.EMBEDDING_API_KEY,
    batchSize: 64,
  }),
});
```

## Limits, and what would come next

* **Brute-force vector search** is `O(N · d)` per query. Fine for the corpus
  sizes a widget deals with; past ~10⁵ documents this wants an ANN index
  (HNSW/IVF) rather than a hand-rolled one.
* **Everything is in memory** and rebuilt on start: no persistence, no sharding.
* **Analysis is English-leaning.** The stemmer is a Porter-lite, the stopword
  list is English, and CJK is indexed as unigrams rather than segmented — hence
  the injectable `tokenize`.
* **Fusion weights are static.** Tuning them (or `k1`/`b`) needs labelled
  queries and an offline metric — nDCG@10, MRR — which is the natural next step
  and the point at which `explain()` and the per-retriever breakdown earn their
  keep.
* **No re-ranking stage.** The usual third step is a cross-encoder over the
  fused top-k, which is where most of the remaining quality lives.
