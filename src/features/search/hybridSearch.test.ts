import { createHashingEmbedder, type Embedder } from './embedder';
import { HybridSearchEngine, createHybridSearchEngine } from './hybridSearch';
import type { KeywordDocument } from './keywordIndex';

/**
 * A hand-written "concept" embedder: four axes — cycling, trains, payments,
 * infrastructure — and a lexicon mapping words onto them, so the vectors carry
 * meaning the keyword index cannot see. "bicycle" and "cycling" share an axis
 * while sharing no term; that gap is exactly what hybrid search exists to
 * close, and faking it keeps the test deterministic and offline.
 */
type Concept = [number, number, number, number];

const CONCEPTS: Record<string, Concept> = {
  bike: [1, 0, 0, 0],
  bikes: [1, 0, 0, 0],
  bicycle: [1, 0, 0, 0],
  cycling: [1, 0, 0, 0],
  pedal: [0.9, 0, 0, 0],
  saddle: [0.9, 0, 0, 0],
  train: [0, 1, 0, 0],
  rail: [0, 1, 0, 0],
  timetable: [0, 1, 0, 0],
  platform: [0, 0.4, 0, 0], // polysemous, so a deliberately weak signal
  payment: [0, 0, 1, 0],
  invoice: [0, 0, 1, 0],
  iban: [0, 0, 1, 0],
  refund: [0, 0, 0.9, 0],
  lane: [0, 0, 0, 1],
  bridge: [0, 0, 0, 1],
  closed: [0, 0, 0, 1],
  repair: [0, 0, 0, 1],
  repairs: [0, 0, 0, 1],
};

const conceptEmbedder: Embedder = async (texts) =>
  texts.map((text) => {
    const vector: Concept = [0, 0, 0, 0];
    for (const word of text.toLowerCase().split(/[^a-z]+/)) {
      const concept = CONCEPTS[word];
      if (!concept) continue;
      for (let i = 0; i < vector.length; i += 1) vector[i] += concept[i];
    }
    return vector;
  });

const DOCUMENTS: KeywordDocument[] = [
  { id: 'cycling-guide', text: 'A guide to cycling: pedal technique and saddle height' },
  { id: 'bike-lane', text: 'The bike lane on the bridge is closed for repairs' },
  { id: 'rail-times', text: 'Train timetable changes, platform updates for the summer' },
  { id: 'iban-help', text: 'How to find your IBAN before making a payment' },
  { id: 'refund-policy', text: 'Refund and invoice policy for a failed payment' },
];

const build = async (documents = DOCUMENTS): Promise<HybridSearchEngine> => {
  const engine = createHybridSearchEngine({ embed: conceptEmbedder });
  await engine.index(documents);
  return engine;
};

const ids = (results: { id: string }[]): string[] => results.map(({ id }) => id);

describe('indexing', () => {
  test('populates both indexes from a single embedder call', async () => {
    const embed = jest.fn(conceptEmbedder);
    const engine = new HybridSearchEngine({ embed });
    await engine.index(DOCUMENTS);

    expect(embed).toHaveBeenCalledTimes(1);
    expect(engine.size).toBe(5);
    expect(engine.keywordIndex.size).toBe(5);
    expect(engine.vectorIndex.size).toBe(5);
  });

  test('indexing nothing does not call the embedder', async () => {
    const embed = jest.fn(conceptEmbedder);
    await new HybridSearchEngine({ embed }).index([]);
    expect(embed).not.toHaveBeenCalled();
  });

  test('rejects an embedder that returns the wrong number of vectors', async () => {
    const engine = new HybridSearchEngine({ embed: async () => [[1, 0, 0, 0]] });
    await expect(engine.index(DOCUMENTS)).rejects.toThrow(/2 vectors|1 vectors/);
  });

  test('remove clears the document from both indexes', async () => {
    const engine = await build();
    expect(engine.remove('bike-lane')).toBe(true);
    expect(engine.remove('bike-lane')).toBe(false);
    expect(engine.size).toBe(4);
    expect(engine.keywordIndex.size).toBe(4);
    expect(engine.vectorIndex.size).toBe(4);
    expect(ids(await engine.search('bike'))).not.toContain('bike-lane');
  });

  test('requires an embedder', () => {
    expect(() => new HybridSearchEngine({ embed: undefined as unknown as Embedder })).toThrow(
      TypeError,
    );
  });
});

describe('single-retriever modes', () => {
  test('keyword mode matches terms exactly and misses the synonym', async () => {
    const engine = await build();
    const results = await engine.search('bicycle', { mode: 'keyword' });
    expect(results).toEqual([]);
  });

  test('semantic mode finds the synonym the keyword index cannot', async () => {
    const engine = await build();
    const results = await engine.search('bicycle', { mode: 'semantic', limit: 2 });
    expect(ids(results)).toContain('cycling-guide');
    expect(results[0].components.semantic.rank).toBe(1);
  });

  test('single-retriever results keep their raw scores', async () => {
    const engine = await build();
    const [top] = await engine.search('bike lane', { mode: 'keyword' });
    expect(top.score).toBe(top.components.keyword.rawScore);
    expect(top.components.semantic).toBeUndefined();
  });

  test('semantic search is skipped when nothing is indexed', async () => {
    const embed = jest.fn(conceptEmbedder);
    const engine = new HybridSearchEngine({ embed });
    expect(await engine.search('bike')).toEqual([]);
    expect(embed).not.toHaveBeenCalled();
  });
});

describe('hybrid search', () => {
  test('recovers the semantic match and keeps the lexical one', async () => {
    const engine = await build();
    const hybrid = ids(await engine.search('bicycle repairs', { limit: 5 }));

    // "bike-lane" shares the word "repairs"; "cycling-guide" shares only meaning.
    expect(hybrid).toContain('bike-lane');
    expect(hybrid).toContain('cycling-guide');
  });

  test('a document both retrievers rank beats one either found alone', async () => {
    const engine = await build();
    const [top] = await engine.search('bike pedal saddle', { limit: 3 });
    expect(top.id).toBe('cycling-guide');
    expect(top.components.keyword).toBeDefined();
    expect(top.components.semantic).toBeDefined();
  });

  test('attaches the source document to every result', async () => {
    const engine = await build();
    const [top] = await engine.search('iban payment');
    expect(top.document).toEqual(DOCUMENTS.find(({ id }) => id === top.id));
  });

  test('weights decide which retriever wins a disagreement', async () => {
    const engine = await build();
    // "platform" is the only literal match in the corpus (rail-times), while
    // "bicycle" points at cycling-guide by meaning alone. The two retrievers
    // therefore disagree about the answer, and the weights settle it.
    const query = 'bicycle platform';
    const keywordHeavy = await engine.search(query, {
      fusion: 'weighted',
      weights: { keyword: 4, semantic: 1 },
      limit: 5,
    });
    const semanticHeavy = await engine.search(query, {
      fusion: 'weighted',
      weights: { keyword: 1, semantic: 4 },
      limit: 5,
    });

    expect(ids(keywordHeavy)[0]).toBe('rail-times');
    expect(ids(semanticHeavy)[0]).toBe('cycling-guide');
  });

  test('rrf answers the same disagreement by rank alone, ignoring the margins', async () => {
    const engine = await build();
    // cycling-guide wins the semantic list, but rail-times is a close second
    // *and* wins the keyword list. Weighted fusion can see that the semantic
    // margin is wide; RRF only sees rank 1 vs rank 2, so the double hit wins.
    const results = await engine.search('bicycle platform', {
      fusion: 'rrf',
      weights: { keyword: 1, semantic: 4 },
      limit: 5,
    });
    expect(ids(results)[0]).toBe('rail-times');
    expect(results[0].components.keyword.rank).toBe(1);
    expect(results[0].components.semantic.rank).toBe(2);
  });

  test('a retriever weighted to zero contributes nothing to the score', async () => {
    const engine = await build();
    const results = await engine.search('bicycle refund', {
      weights: { semantic: 0 },
      limit: 5,
    });
    const semanticOnlyHit = results.find(({ id }) => id === 'cycling-guide');

    expect(ids(results)[0]).toBe('refund-policy');
    expect(semanticOnlyHit?.score).toBe(0);
    expect(semanticOnlyHit?.components.keyword).toBeUndefined();
  });

  test('both fusion strategies rank the obvious answer first', async () => {
    const engine = await build();
    for (const fusion of ['rrf', 'weighted'] as const) {
      const [top] = await engine.search('iban payment', { fusion, limit: 3 });
      expect(top.id).toBe('iban-help');
    }
  });

  test('weighted fusion normalises before summing, so BM25 cannot swamp cosine', async () => {
    const engine = await build();
    const results = await engine.search('train timetable', {
      fusion: 'weighted',
      limit: 5,
    });
    for (const { components } of results) {
      for (const contribution of Object.values(components)) {
        expect(contribution.fusedScore).toBeGreaterThanOrEqual(0);
        expect(contribution.fusedScore).toBeLessThanOrEqual(1);
      }
    }
  });

  test('over-fetches candidates so a mid-ranked agreement can still surface', async () => {
    const engine = await build();
    const embedSpy = jest.spyOn(engine.keywordIndex, 'search');
    await engine.search('bike', { limit: 2 });
    expect(embedSpy).toHaveBeenCalledWith('bike', { limit: 50 });

    await engine.search('bike', { limit: 2, candidates: 3 });
    expect(embedSpy).toHaveBeenLastCalledWith('bike', { limit: 3 });
  });

  test('respects the limit and returns nothing for an empty query', async () => {
    const engine = await build();
    expect(await engine.search('bike', { limit: 1 })).toHaveLength(1);
    expect(await engine.search('   ')).toEqual([]);
  });

  test('constructor defaults apply unless the query overrides them', async () => {
    const engine = new HybridSearchEngine({
      embed: conceptEmbedder,
      defaults: { mode: 'semantic' },
    });
    await engine.index(DOCUMENTS);

    expect(ids(await engine.search('bicycle'))).toContain('cycling-guide');
    expect(await engine.search('bicycle', { mode: 'keyword' })).toEqual([]);
  });
});

describe('with the offline hashing embedder', () => {
  test('the whole pipeline runs end to end without a model', async () => {
    const engine = createHybridSearchEngine({
      embed: createHashingEmbedder({ dimensions: 256 }),
    });
    await engine.index(DOCUMENTS);

    const [top] = await engine.search('train timetable', { limit: 3 });
    expect(top.id).toBe('rail-times');
    expect(top.components.keyword).toBeDefined();
    expect(top.components.semantic).toBeDefined();
  });

  test('subword similarity rescues a misspelled query that BM25 drops', async () => {
    const engine = createHybridSearchEngine({
      embed: createHashingEmbedder({ dimensions: 512 }),
    });
    await engine.index(DOCUMENTS);

    expect(await engine.search('timetabel', { mode: 'keyword' })).toEqual([]);
    expect(ids(await engine.search('timetabel', { limit: 3 }))).toContain('rail-times');
  });
});
