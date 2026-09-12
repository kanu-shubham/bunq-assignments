import { createHashingEmbedder, createRemoteEmbedder, l2Normalize } from './embedder';
import { cosineSimilarity } from './vectorIndex';

describe('l2Normalize', () => {
  test('scales a vector to unit length', () => {
    const [x, y] = l2Normalize([3, 4]);
    expect(x).toBeCloseTo(0.6, 10);
    expect(y).toBeCloseTo(0.8, 10);
  });

  test('leaves the zero vector alone instead of dividing by zero', () => {
    expect(l2Normalize([0, 0])).toEqual([0, 0]);
  });
});

describe('createHashingEmbedder', () => {
  const embed = createHashingEmbedder({ dimensions: 128 });

  test('is deterministic and unit length', async () => {
    const [first] = await embed(['bike repair shop']);
    const [again] = await embed(['bike repair shop']);
    expect(first).toEqual(again);
    expect(first).toHaveLength(128);
    expect(cosineSimilarity(first, first)).toBeCloseTo(1, 10);
  });

  test('embeds each input once, in order', async () => {
    const vectors = await embed(['alpha', 'beta', 'gamma']);
    expect(vectors).toHaveLength(3);
    expect(vectors[0]).not.toEqual(vectors[1]);
  });

  test('related text is closer than unrelated text', async () => {
    const [query, related, unrelated] = await embed([
      'bike repair shop',
      'shop for bike repairs',
      'summer train timetable',
    ]);
    expect(cosineSimilarity(query, related)).toBeGreaterThan(
      cosineSimilarity(query, unrelated),
    );
  });

  test('character n-grams give it tolerance for typos, which BM25 has none of', () => {
    const withNgrams = createHashingEmbedder({ dimensions: 256 });
    const withoutNgrams = createHashingEmbedder({ dimensions: 256, charNgrams: false });

    return Promise.all([
      withNgrams(['bicycle', 'bicyle']),
      withoutNgrams(['bicycle', 'bicyle']),
    ]).then(([[a, b], [c, d]]) => {
      // Two different tokens share no whole-token feature at all: without
      // n-grams the vectors are orthogonal, with them they stay neighbours.
      expect(cosineSimilarity(c, d)).toBeCloseTo(0, 10);
      expect(cosineSimilarity(a, b)).toBeGreaterThan(0.3);
    });
  });

  test('the zero vector is the honest answer for text with no terms', async () => {
    const [vector] = await embed(['the and of']);
    expect(vector.every((value) => value === 0)).toBe(true);
  });

  test('a different seed produces a different projection', async () => {
    const [a] = await createHashingEmbedder({ dimensions: 64, seed: 1 })(['bike']);
    const [b] = await createHashingEmbedder({ dimensions: 64, seed: 2 })(['bike']);
    expect(a).not.toEqual(b);
  });

  test('rejects a non-positive dimension', () => {
    expect(() => createHashingEmbedder({ dimensions: 0 })).toThrow(RangeError);
  });
});

describe('createRemoteEmbedder', () => {
  const respondWith = (vectors: number[][]): Response =>
    ({
      ok: true,
      status: 200,
      json: async () => ({ data: vectors.map((embedding) => ({ embedding })) }),
    }) as Response;

  test('posts the batch and returns the vectors in order', async () => {
    const fetchImpl = jest.fn(async () => respondWith([[1, 0], [0, 1]]));
    const embed = createRemoteEmbedder({
      endpoint: '/api/embeddings',
      model: 'embed-v1',
      apiKey: 'secret',
      fetchImpl: fetchImpl as unknown as typeof fetch,
    });

    expect(await embed(['a', 'b'])).toEqual([[1, 0], [0, 1]]);

    const [endpoint, init] = fetchImpl.mock.calls[0] as unknown as [string, RequestInit];
    expect(endpoint).toBe('/api/embeddings');
    expect((init.headers as Record<string, string>).Authorization).toBe('Bearer secret');
    expect(JSON.parse(init.body as string)).toEqual({ model: 'embed-v1', input: ['a', 'b'] });
  });

  test('splits large inputs into batches', async () => {
    const fetchImpl = jest.fn(async (_url: string, init: RequestInit) => {
      const { input } = JSON.parse(init.body as string) as { input: string[] };
      return respondWith(input.map(() => [1, 0]));
    });
    const embed = createRemoteEmbedder({
      endpoint: '/api/embeddings',
      batchSize: 2,
      fetchImpl: fetchImpl as unknown as typeof fetch,
    });

    expect(await embed(['a', 'b', 'c', 'd', 'e'])).toHaveLength(5);
    expect(fetchImpl).toHaveBeenCalledTimes(3);
  });

  test('surfaces HTTP failures', async () => {
    const fetchImpl = jest.fn(async () => ({ ok: false, status: 429 }) as Response);
    const embed = createRemoteEmbedder({
      endpoint: '/api/embeddings',
      fetchImpl: fetchImpl as unknown as typeof fetch,
    });
    await expect(embed(['a'])).rejects.toThrow('429');
  });

  test('rejects a truncated response rather than misaligning documents', async () => {
    const fetchImpl = jest.fn(async () => respondWith([[1, 0]]));
    const embed = createRemoteEmbedder({
      endpoint: '/api/embeddings',
      fetchImpl: fetchImpl as unknown as typeof fetch,
    });
    await expect(embed(['a', 'b'])).rejects.toThrow('size mismatch');
  });

  test('a custom parser adapts a provider-specific response shape', async () => {
    const fetchImpl = jest.fn(async () => ({
      ok: true,
      status: 200,
      json: async () => ({ embeddings: [[0, 1]] }),
    }) as unknown as Response);
    const embed = createRemoteEmbedder({
      endpoint: '/api/embeddings',
      fetchImpl: fetchImpl as unknown as typeof fetch,
      parse: (body) => (body as { embeddings: number[][] }).embeddings,
    });
    expect(await embed(['a'])).toEqual([[0, 1]]);
  });
});
