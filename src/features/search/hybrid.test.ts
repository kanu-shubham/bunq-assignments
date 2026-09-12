import {
  minMaxNormalize,
  normalizeScores,
  reciprocalRankFusion,
  weightedFusion,
  zScoreNormalize,
  type NamedRanking,
} from './hybrid';

const ids = (results: { id: string }[]): string[] => results.map(({ id }) => id);

describe('minMaxNormalize', () => {
  test('maps the list onto [0, 1]', () => {
    const normalized = minMaxNormalize([
      { id: 'a', score: 14 },
      { id: 'b', score: 9 },
      { id: 'c', score: 4 },
    ]);
    expect(normalized.map(({ score }) => score)).toEqual([1, 0.5, 0]);
  });

  test('gives every entry 1 when all scores are equal', () => {
    const normalized = minMaxNormalize([
      { id: 'a', score: 3 },
      { id: 'b', score: 3 },
    ]);
    expect(normalized.map(({ score }) => score)).toEqual([1, 1]);
  });

  test('handles the empty and single-result cases', () => {
    expect(minMaxNormalize([])).toEqual([]);
    expect(minMaxNormalize([{ id: 'a', score: 7 }])).toEqual([{ id: 'a', score: 1 }]);
  });
});

describe('zScoreNormalize', () => {
  test('centres on the mean and scales by the deviation', () => {
    const normalized = zScoreNormalize([
      { id: 'a', score: 10 },
      { id: 'b', score: 20 },
      { id: 'c', score: 30 },
    ]);
    expect(normalized[1].score).toBeCloseTo(0, 10);
    expect(normalized[0].score).toBeCloseTo(-normalized[2].score, 10);
  });

  test('is less distorted by a single outlier than min-max', () => {
    const results = [
      { id: 'outlier', score: 1000 },
      { id: 'a', score: 10 },
      { id: 'b', score: 8 },
    ];
    const [, minMaxA, minMaxB] = minMaxNormalize(results);
    const [, zA, zB] = zScoreNormalize(results);
    expect(minMaxA.score - minMaxB.score).toBeLessThan(zA.score - zB.score);
  });

  test('returns zeros when there is no spread', () => {
    expect(zScoreNormalize([{ id: 'a', score: 5 }]).map(({ score }) => score)).toEqual([0]);
  });
});

describe('normalizeScores', () => {
  test('none passes scores through untouched', () => {
    expect(normalizeScores([{ id: 'a', score: 42 }], 'none')).toEqual([{ id: 'a', score: 42 }]);
  });

  test('rejects an unknown strategy', () => {
    expect(() =>
      normalizeScores([{ id: 'a', score: 1 }], 'bogus' as unknown as 'minmax'),
    ).toThrow(TypeError);
  });
});

describe('weightedFusion', () => {
  // BM25 is unbounded, cosine sits in [-1, 1]: the raw scales are incomparable,
  // which is the whole reason normalisation happens before the sum.
  const rankings: NamedRanking[] = [
    {
      name: 'keyword',
      results: [
        { id: 'doc1', score: 14.0 },
        { id: 'doc2', score: 9.2 },
        { id: 'doc3', score: 2.0 },
      ],
    },
    {
      name: 'semantic',
      results: [
        { id: 'doc3', score: 0.91 },
        { id: 'doc2', score: 0.79 },
        { id: 'doc4', score: 0.61 },
      ],
    },
  ];

  test('rewards the document both retrievers like', () => {
    expect(ids(weightedFusion(rankings))[0]).toBe('doc2');
  });

  test('keeps the raw score, the fused score and the rank per retriever', () => {
    const [top] = weightedFusion(rankings);
    expect(top.components.keyword.rawScore).toBe(9.2);
    expect(top.components.keyword.fusedScore).toBeCloseTo(0.6, 10);
    expect(top.components.keyword.rank).toBe(2);
    expect(top.components.semantic.rank).toBe(2);
    expect(top.score).toBeCloseTo(
      top.components.keyword.fusedScore + top.components.semantic.fusedScore,
      10,
    );
  });

  test('a document only one retriever found still ranks, scored 0 by the other', () => {
    const fused = weightedFusion(rankings);
    const only = fused.find(({ id }) => id === 'doc4');
    expect(only?.components.keyword).toBeUndefined();
    expect(only?.components.semantic).toBeDefined();
  });

  test('weights shift which retriever decides the ranking', () => {
    const keywordHeavy = weightedFusion([
      { ...rankings[0], weight: 4 },
      { ...rankings[1], weight: 1 },
    ]);
    const semanticHeavy = weightedFusion([
      { ...rankings[0], weight: 1 },
      { ...rankings[1], weight: 4 },
    ]);
    expect(ids(keywordHeavy)[0]).toBe('doc1');
    expect(ids(semanticHeavy)[0]).toBe('doc3');
  });

  test('a zero weight silences a retriever without dropping its breakdown', () => {
    const fused = weightedFusion([{ ...rankings[0], weight: 0 }, rankings[1]]);
    expect(ids(fused)[0]).toBe('doc3');
    expect(fused[0].components.keyword.fusedScore).toBe(0);
  });

  test('without normalisation the larger scale dominates', () => {
    expect(ids(weightedFusion(rankings, { normalize: 'none' }))[0]).toBe('doc1');
  });

  test('respects the limit', () => {
    expect(weightedFusion(rankings, { limit: 2 })).toHaveLength(2);
  });

  test('fusing nothing yields nothing', () => {
    expect(weightedFusion([])).toEqual([]);
    expect(weightedFusion([{ name: 'keyword', results: [] }])).toEqual([]);
  });
});

describe('reciprocalRankFusion', () => {
  const rankings: NamedRanking[] = [
    {
      name: 'keyword',
      results: [
        { id: 'doc1', score: 999 },
        { id: 'doc2', score: 9.1 },
      ],
    },
    {
      name: 'semantic',
      results: [
        { id: 'doc2', score: 0.91 },
        { id: 'doc3', score: 0.9 },
      ],
    },
  ];

  test('scores by rank alone: 1 / (k + rank)', () => {
    const [top] = reciprocalRankFusion(rankings, { k: 60 });
    expect(top.id).toBe('doc2');
    expect(top.score).toBeCloseTo(1 / 62 + 1 / 61, 10);
  });

  test('ignores score magnitude, so no retriever needs calibrating', () => {
    const inflated: NamedRanking[] = [
      { name: 'keyword', results: rankings[0].results.map((r) => ({ ...r, score: r.score * 1e6 })) },
      rankings[1],
    ];
    const before = reciprocalRankFusion(rankings);
    const after = reciprocalRankFusion(inflated);
    expect(ids(after)).toEqual(ids(before));
    expect(after.map(({ score }) => score)).toEqual(before.map(({ score }) => score));
  });

  test('a small k sharpens the advantage of the top rank', () => {
    const single: NamedRanking[] = [
      { name: 'keyword', results: [{ id: 'first', score: 5 }, { id: 'second', score: 4 }] },
    ];
    const sharp = reciprocalRankFusion(single, { k: 1 });
    const flat = reciprocalRankFusion(single, { k: 1000 });
    expect(sharp[0].score / sharp[1].score).toBeGreaterThan(flat[0].score / flat[1].score);
  });

  test('weights scale a retriever\'s reciprocal contribution', () => {
    const fused = reciprocalRankFusion([
      { ...rankings[0], weight: 2 },
      { ...rankings[1], weight: 0 },
    ]);
    expect(ids(fused)[0]).toBe('doc1');
    expect(fused[0].score).toBeCloseTo(2 / 61, 10);
  });

  test('rejects a non-positive k', () => {
    expect(() => reciprocalRankFusion(rankings, { k: 0 })).toThrow(RangeError);
  });
});
