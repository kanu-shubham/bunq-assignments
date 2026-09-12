import { VectorIndex, cosineSimilarity } from './vectorIndex';

describe('cosineSimilarity', () => {
  test('is 1 for identical directions and 0 for orthogonal ones', () => {
    expect(cosineSimilarity([1, 0], [2, 0])).toBeCloseTo(1, 10);
    expect(cosineSimilarity([1, 0], [0, 3])).toBeCloseTo(0, 10);
    expect(cosineSimilarity([1, 0], [-1, 0])).toBeCloseTo(-1, 10);
  });

  test('is magnitude-independent', () => {
    expect(cosineSimilarity([1, 2, 3], [10, 20, 30])).toBeCloseTo(1, 10);
  });

  test('treats the zero vector as unrelated instead of dividing by zero', () => {
    expect(cosineSimilarity([0, 0], [1, 1])).toBe(0);
  });

  test('refuses mismatched lengths', () => {
    expect(() => cosineSimilarity([1, 0], [1, 0, 0])).toThrow(RangeError);
  });
});

describe('VectorIndex', () => {
  const build = (): VectorIndex => {
    const index = new VectorIndex();
    index.addAll([
      { id: 'east', vector: [1, 0] },
      { id: 'north', vector: [0, 1] },
      { id: 'northeast', vector: [1, 1] },
    ]);
    return index;
  };

  test('ranks by cosine similarity to the query', () => {
    const results = build().search([1, 0.1]);
    expect(results.map(({ id }) => id)).toEqual(['east', 'northeast', 'north']);
    expect(results[0].score).toBeGreaterThan(results[1].score);
  });

  test('normalises on insert, so magnitude does not buy rank', () => {
    const index = new VectorIndex();
    index.add('small', [1, 0]);
    index.add('huge', [0, 1000]);
    const [top] = index.search([1, 0]);
    expect(top.id).toBe('small');
  });

  test('respects limit and minScore', () => {
    expect(build().search([1, 0], { limit: 2 })).toHaveLength(2);
    expect(build().search([1, 0], { minScore: 0.9 }).map(({ id }) => id)).toEqual(['east']);
  });

  test('replaces a vector when the id is reused', () => {
    const index = build();
    index.add('north', [1, 0]);
    expect(index.size).toBe(3);
    expect(index.search([1, 0])[0].score).toBeCloseTo(1, 10);
  });

  test('remove takes the vector out of the results', () => {
    const index = build();
    expect(index.remove('east')).toBe(true);
    expect(index.remove('east')).toBe(false);
    expect(index.search([1, 0]).map(({ id }) => id)).not.toContain('east');
  });

  test('an empty index answers with no results', () => {
    expect(new VectorIndex().search([1, 0])).toEqual([]);
  });

  test('guards the dimension invariant on both insert and query', () => {
    const index = build();
    expect(() => index.add('bad', [1, 2, 3])).toThrow(/dimension mismatch/);
    expect(() => index.search([1, 2, 3])).toThrow(/dimension mismatch/);
    expect(() => index.add('empty', [])).toThrow(RangeError);
  });
});
