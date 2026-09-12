import {
  DEFAULT_STOPWORDS,
  createTokenizer,
  fold,
  lightStem,
  split,
  tokenize,
} from './tokenize';

describe('fold', () => {
  test('lowercases and strips diacritics', () => {
    expect(fold('Café RÉSUMÉ')).toBe('cafe resume');
  });
});

describe('split', () => {
  test('splits on punctuation and whitespace', () => {
    expect(split('Hello, world — search/engine!')).toEqual([
      'hello',
      'world',
      'search',
      'engine',
    ]);
  });

  test('keeps digits and intra-word apostrophes', () => {
    expect(split("don't ship 3 bugs")).toEqual(['don\'t', 'ship', '3', 'bugs']);
  });

  test('trims apostrophes at token edges', () => {
    expect(split("'quoted' word")).toEqual(['quoted', 'word']);
  });

  test('treats CJK characters as unigrams', () => {
    expect(split('検索エンジン')).toEqual(['検', '索', 'エ', 'ン', 'ジ', 'ン']);
  });
});

describe('lightStem', () => {
  test.each([
    ['bikes', 'bike'],
    ['biking', 'bike'],
    ['boxes', 'box'],
    ['queries', 'query'],
    ['classes', 'class'],
    ['running', 'run'],
    ['shipped', 'ship'],
    ['quickly', 'quick'],
  ])('%s → %s', (input, expected) => {
    expect(lightStem(input)).toBe(expected);
  });

  test('leaves short tokens alone', () => {
    expect(lightStem('as')).toBe('as');
    expect(lightStem('gas')).toBe('gas');
  });

  test('does not strip the s from -ss words', () => {
    expect(lightStem('grass')).toBe('grass');
  });
});

describe('tokenize', () => {
  test('drops stopwords', () => {
    expect(tokenize('the state of the art')).toEqual(['state', 'art']);
  });

  test('folds inflections of a word onto one term', () => {
    const [bike] = tokenize('bike');
    expect(tokenize('bikes')).toEqual([bike]);
  });

  test('drops single characters but keeps CJK unigrams', () => {
    expect(tokenize('a b search')).toEqual(['search']);
    expect(tokenize('検索')).toEqual(['検', '索']);
  });

  test('returns nothing for empty or stopword-only input', () => {
    expect(tokenize('')).toEqual([]);
    expect(tokenize('the and of')).toEqual([]);
  });
});

describe('createTokenizer', () => {
  test('stopwords and stemming are injectable', () => {
    const raw = createTokenizer({ stopwords: new Set(), stem: (t) => t, minLength: 1 });
    expect(raw('the bikes')).toEqual(['the', 'bikes']);
  });

  test('custom stopword list replaces the default', () => {
    const custom = createTokenizer({ stopwords: new Set(['bunq']) });
    expect(custom('bunq the bank')).toEqual(['the', 'bank']);
    expect(DEFAULT_STOPWORDS.has('the')).toBe(true);
  });
});
