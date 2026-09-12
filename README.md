# bunq assignments

Two self-contained assignments under `src/features/`, each with its own tests
and README.

## 1. Feedback Widget — `src/features/feedback`

Feature-rating popup.

**Flow** — a finite-state machine in [`src/features/feedback/state/feedbackMachine.ts`](./src/features/feedback/state/feedbackMachine.ts):
`CLOSED → RATING → { NEGATIVE_FORM → SUBMITTING → THANK_YOU | THANK_YOU } → { CLOSED | TRUSTPILOT }`.
`Action` is a discriminated union; the reducer's `default` holds `const _: never = action`, so a new variant without a case is a compile error.

**DI seam** — the widget accepts a `submitFeedback` prop (typed `(p: FeedbackPayload) => Promise<unknown>`). Production injects nothing and uses [`src/features/feedback/services/feedbackService.ts`](./src/features/feedback/services/feedbackService.ts); tests inject a mock.

**Accessibility** — portal-mounted modal with focus trap + restoration, ESC, backdrop dismiss, `aria-modal` + labelled title; thank-you toast is `role="status" aria-live="polite"`; the NEGATIVE form has `role="group"`.

**Run** — `npm install && npm start` opens the demo (click the button). `npm test` runs the whole suite (FSM transitions, service contract, integration flow incl. ESC / failure / STELLAR → Trustpilot).

## 2. Hybrid search — `src/features/search`

BM25 keyword search built from scratch over an inverted index, dense retrieval
over embeddings, and the fusion that combines them into one ranking. Details,
formulas and trade-offs in [its own README](./src/features/search/README.md).

**Keyword** — [`keywordIndex.ts`](./src/features/search/keywordIndex.ts) holds the
inverted index and two scorers: BM25 (default; `k1` saturates term frequency,
`b` interpolates length normalisation, non-negative IDF) and an `ltc.lnc` TF-IDF
cosine for comparison. `explain()` decomposes any score term by term.

**Semantic** — [`vectorIndex.ts`](./src/features/search/vectorIndex.ts) does cosine
search over L2-normalised vectors. Embeddings come through the `Embedder` seam
(the same DI shape as `submitFeedback`), so tests and the demo run offline on a
hashing embedder and a hosted model is a one-line swap.

**Fusion** — [`hybrid.ts`](./src/features/search/hybrid.ts) implements reciprocal
rank fusion (default, scale-free) and weighted fusion over min-max or z-score
normalised scores, because BM25's unbounded scale and cosine's `[-1, 1]` band
cannot be summed raw. Every result carries a per-retriever breakdown — raw
score, fused score, rank.

```ts
const engine = createHybridSearchEngine({ embed: createHashingEmbedder() });
await engine.index(documents);
const results = await engine.search('bicycle repairs', { limit: 5 });
```

## Repository layout

```
src/
├── App.tsx, App.css      ← minimal launcher
├── index.tsx
├── setupTests.ts
└── features/
    ├── feedback/         ← assignment 1 (see its own README for details)
    │   ├── FeedbackWidget.tsx + test
    │   ├── components/   (Modal, RatingPrompt, NegativeFeedbackForm,
    │   │                  ThankYouToast, TrustpilotPrompt + CSS)
    │   ├── hooks/        (useFocusTrap, useEscapeKey,
    │   │                  useAutoDismiss, useStableId)
    │   ├── services/     (feedbackService + test)
    │   ├── state/        (feedbackMachine + test)
    │   └── index.ts
    └── search/           ← assignment 2 (see its own README for details)
        ├── tokenize.ts       (fold → stopwords → light stemmer)
        ├── keywordIndex.ts   (inverted index, BM25 + TF-IDF)
        ├── embedder.ts       (Embedder seam, hashing + remote embedders)
        ├── vectorIndex.ts    (cosine search)
        ├── hybrid.ts         (normalisation, RRF, weighted fusion)
        ├── hybridSearch.ts   (the engine tying both retrievers together)
        └── index.ts
```

Every module has a colocated `*.test.ts`.

