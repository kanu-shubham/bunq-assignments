import { KeywordIndex, type KeywordDocument } from './keywordIndex';

const CORPUS: KeywordDocument[] = [
  { id: 'a', text: 'The bike lane on the bridge is closed for repairs' },
  { id: 'b', text: 'Bike repair shop opening hours and prices' },
  { id: 'c', text: 'A guide to repairing a bike tyre at home, step by step' },
  { id: 'd', text: 'Train timetable changes for the summer' },
];

const build = (documents = CORPUS, options = {}): KeywordIndex => {
  const index = new KeywordIndex(options);
  index.addAll(documents);
  return index;
};

const ids = (results: { id: string }[]): string[] => results.map(({ id }) => id);

describe('indexing', () => {
  test('tracks corpus statistics', () => {
    const index = build();
    expect(index.size).toBe(4);
    expect(index.averageDocumentLength).toBeGreaterThan(0);
    expect(index.vocabularySize).toBeGreaterThan(0);
  });

  test('an empty index answers with no results', () => {
    expect(new KeywordIndex().search('bike')).toEqual([]);
  });

  test('re-adding an id replaces the document rather than duplicating it', () => {
    const index = build();
    index.add({ id: 'd', text: 'bike bike bike' });

    expect(index.size).toBe(4);
    const results = index.search('bike');
    expect(ids(results).filter((id) => id === 'd')).toHaveLength(1);
    expect(index.search('timetable')).toEqual([]);
  });

  test('remove drops the document and its postings', () => {
    const index = build();
    expect(index.remove('b')).toBe(true);
    expect(index.remove('b')).toBe(false);
    expect(index.size).toBe(3);
    expect(ids(index.search('bike'))).not.toContain('b');
  });

  test('document frequency reflects the analyzed term', () => {
    const index = build();
    expect(index.documentFrequency('bike')).toBe(3);
    expect(index.documentFrequency('nonexistent')).toBe(0);
  });
});

describe('idf', () => {
  test('rarer terms score higher', () => {
    const index = build();
    expect(index.idf('timetable')).toBeGreaterThan(index.idf('bike'));
  });

  test('stays non-negative for terms in most of the corpus', () => {
    const index = build([
      { id: 'a', text: 'common word here' },
      { id: 'b', text: 'common word there' },
      { id: 'c', text: 'common word everywhere' },
    ]);
    expect(index.idf('common')).toBeGreaterThanOrEqual(0);
  });

  test('unknown terms contribute nothing', () => {
    expect(build().idf('quantum')).toBe(0);
  });
});

describe('bm25 search', () => {
  test('ranks documents matching more query terms first', () => {
    const index = build();
    // "b" is the only document carrying all three terms.
    expect(ids(index.search('bike repair shop'))[0]).toBe('b');
  });

  test('a rarer term outweighs a common one', () => {
    const index = build();
    // "bike" is in three of four documents, "timetable" in one.
    expect(ids(index.search('bike timetable'))[0]).toBe('d');
  });

  test('only returns documents containing a query term', () => {
    expect(ids(build().search('bike'))).toEqual(expect.not.arrayContaining(['d']));
  });

  test('respects the limit', () => {
    expect(build().search('bike', { limit: 2 })).toHaveLength(2);
  });

  test('a query of only stopwords or unknown terms returns nothing', () => {
    const index = build();
    expect(index.search('the and of')).toEqual([]);
    expect(index.search('quantum chromodynamics')).toEqual([]);
    expect(index.search('')).toEqual([]);
  });

  test('matches across inflections because both sides share the analyzer', () => {
    const index = build();
    expect(ids(index.search('repairing bikes'))).toContain('b');
  });

  test('ties are broken by id, so results are deterministic', () => {
    const index = build([
      { id: 'z', text: 'kayak' },
      { id: 'y', text: 'kayak' },
    ]);
    expect(ids(index.search('kayak'))).toEqual(['y', 'z']);
  });
});

describe('bm25 parameters', () => {
  const repeated = [
    { id: 'once', text: 'bike lane closed' },
    { id: 'many', text: 'bike bike bike bike bike bike lane closed' },
  ];

  test('term frequency saturates: 6 occurrences are worth well under 6x one', () => {
    const index = build(repeated);
    const [top, second] = index.search('bike');
    expect(top.id).toBe('many');
    expect(top.score).toBeGreaterThan(second.score);
    expect(top.score).toBeLessThan(second.score * 6);
  });

  test('k1 = 0 collapses term frequency to a presence bit', () => {
    const index = build(repeated, { k1: 0 });
    const [first, second] = index.search('bike');
    expect(first.score).toBeCloseTo(second.score, 10);
  });

  test('b = 0 disables length normalisation, b = 1 applies it fully', () => {
    const documents = [
      { id: 'short', text: 'bike' },
      { id: 'long', text: `bike ${'filler word padding sentence '.repeat(10)}` },
    ];
    const noNorm = build(documents, { b: 0 }).search('bike');
    const fullNorm = build(documents, { b: 1 }).search('bike');

    expect(noNorm[0].score).toBeCloseTo(noNorm[1].score, 10);
    expect(ids(fullNorm)[0]).toBe('short');
    expect(fullNorm[0].score).toBeGreaterThan(fullNorm[1].score);
  });

  test('rejects out-of-range parameters', () => {
    expect(() => new KeywordIndex({ k1: -1 })).toThrow(RangeError);
    expect(() => new KeywordIndex({ b: 1.5 })).toThrow(RangeError);
  });
});

describe('tfidf scorer', () => {
  test('ranks the same obvious winner as bm25 here', () => {
    const index = build();
    expect(ids(index.search('bike repair shop', { scorer: 'tfidf' }))[0]).toBe('b');
  });

  test('cosine scores stay within [0, 1]', () => {
    for (const { score } of build().search('bike repair', { scorer: 'tfidf' })) {
      expect(score).toBeGreaterThan(0);
      expect(score).toBeLessThanOrEqual(1);
    }
  });

  test('normalises by the document vector, so a repeated term cannot run away', () => {
    const documents = [
      { id: 'once', text: 'bike lane' },
      { id: 'many', text: `${'bike '.repeat(20)}lane` },
    ];
    const index = build(documents, { b: 0 });
    expect(ratio(index.search('bike', { scorer: 'tfidf' }))).toBeLessThan(1.5);
  });
});

describe('bm25 saturation ceiling', () => {
  test('a term contributes at most idf · (k1 + 1), however often it repeats', () => {
    // The property BM25 buys over raw TF-IDF: an explicit, tunable asymptote.
    const k1 = 1.2;
    const index = build(
      [
        { id: 'once', text: 'bike lane' },
        { id: 'many', text: `${'bike '.repeat(1000)}lane` },
      ],
      { k1, b: 0 },
    );
    const [top] = index.search('bike');
    const ceiling = index.idf('bike') * (k1 + 1);

    expect(top.score).toBeLessThan(ceiling);
    expect(top.score).toBeGreaterThan(ceiling * 0.99); // 1000 repeats ≈ the asymptote
  });
});

describe('explain', () => {
  test('breaks a score down per term, largest contribution first', () => {
    const index = build();
    const parts = index.explain('bike repair', 'b');
    expect(parts.map((part) => part.term)).toHaveLength(2);
    expect(parts[0].contribution).toBeGreaterThanOrEqual(parts[1].contribution);
    const total = parts.reduce((sum, part) => sum + part.contribution, 0);
    const [match] = index.search('bike repair').filter(({ id }) => id === 'b');
    expect(total).toBeCloseTo(match.score, 10);
  });

  test('returns nothing for an unknown document', () => {
    expect(build().explain('bike', 'missing')).toEqual([]);
  });
});

function ratio(results: { id: string; score: number }[]): number {
  const many = results.find(({ id }) => id === 'many')?.score ?? 0;
  const once = results.find(({ id }) => id === 'once')?.score ?? 1;
  return many / once;
}
