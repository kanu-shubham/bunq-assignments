/**
 * Text analysis for the keyword index.
 *
 * Indexing and querying MUST run through the same analyzer: BM25 matches on
 * exact token equality, so "Bikes" in a document and "biking" in a query only
 * meet if both sides are folded to the same string.
 *
 * Scope: whitespace/punctuation-delimited scripts, plus CJK handled as
 * unigrams. Anything else (proper morphology, Thai, compound splitting) wants
 * a real analyzer — hence `TokenizerOptions.tokenize` on every public API.
 */

/** A closed-class word carries almost no discriminative signal. */
export const DEFAULT_STOPWORDS: ReadonlySet<string> = new Set([
  'a', 'an', 'and', 'are', 'as', 'at', 'be', 'but', 'by', 'for', 'from', 'has',
  'have', 'he', 'her', 'his', 'i', 'in', 'is', 'it', 'its', 'of', 'on', 'or',
  'she', 'that', 'the', 'their', 'them', 'then', 'there', 'they', 'this', 'to',
  'was', 'were', 'will', 'with', 'you', 'your',
]);

export interface TokenizerOptions {
  /** Dropped after folding, before stemming. Pass an empty set to keep everything. */
  stopwords?: ReadonlySet<string>;
  /** Suffix folding. Defaults to {@link lightStem}; pass `(t) => t` to disable. */
  stem?: (token: string) => string;
  /** Tokens shorter than this are dropped (CJK unigrams are exempt). Default 2. */
  minLength?: number;
}

const COMBINING_MARKS = /[\u0300-\u036f]/g;

/**
 * Lowercase + strip Latin diacritics so "café" and "cafe" collide.
 *
 * Only the Latin combining block is removed, and the result is recomposed:
 * Japanese dakuten are combining marks too, and dropping them would turn
 * "ジ" into "シ".
 */
export function fold(text: string): string {
  return text.normalize('NFD').replace(COMBINING_MARKS, '').normalize('NFC').toLowerCase();
}

function isDigit(ch: string): boolean {
  return ch >= '0' && ch <= '9';
}

/**
 * Letter test without unicode property escapes (which would pin the compile
 * target to ES2018+): a letter is a character that cases differently.
 */
function isLetter(ch: string): boolean {
  return ch.toLowerCase() !== ch.toUpperCase();
}

/** Scripts written without spaces; each character indexes as its own token. */
function isUnigramScript(code: number): boolean {
  return (
    (code >= 0x3040 && code <= 0x30ff) || // kana
    (code >= 0x3400 && code <= 0x4dbf) || // CJK ext. A
    (code >= 0x4e00 && code <= 0x9fff) || // CJK unified
    (code >= 0xf900 && code <= 0xfaff)    // CJK compatibility
  );
}

const INTRA_WORD = new Set(["'", '’']); // don't split "user's"

/** Split folded text into raw tokens. Exported for tests and custom analyzers. */
export function split(text: string): string[] {
  const tokens: string[] = [];
  let current = '';

  const flush = (): void => {
    if (current) tokens.push(current);
    current = '';
  };

  for (const ch of fold(text)) {
    const code = ch.codePointAt(0) ?? 0;

    if (isUnigramScript(code)) {
      flush();
      tokens.push(ch);
      continue;
    }
    if (isDigit(ch) || isLetter(ch)) {
      current += ch;
      continue;
    }
    // An apostrophe only stays inside a word ("don't"), never at an edge.
    if (INTRA_WORD.has(ch) && current) {
      current += ch;
      continue;
    }
    flush();
  }
  flush();

  return tokens.map((t) => trimEdges(t));
}

function trimEdges(token: string): string {
  let start = 0;
  let end = token.length;
  while (start < end && INTRA_WORD.has(token[start])) start += 1;
  while (end > start && INTRA_WORD.has(token[end - 1])) end -= 1;
  return token.slice(start, end);
}

/**
 * A Porter-lite suffix stripper: plural/participle folding only.
 *
 * Linguistic perfection is not the goal — applying the *same* fold to documents
 * and queries is, because that is what makes "bike", "bikes" and "biking" one
 * index term. The `-es` and `-ing`/`-ed` rules carry the two repairs that
 * matter in practice: only strip a full "es" after a sibilant ("boxes" → "box",
 * but "bikes" → "bike"), and restore the silent "e" a stripped participle left
 * behind ("biking" → "bik" → "bike").
 */
export function lightStem(token: string): string {
  if (token.length <= 3) return token;

  if (token.endsWith('ies') && token.length > 4) return `${token.slice(0, -3)}y`;
  if (token.endsWith('ss')) return token;

  if (token.endsWith('es') && token.length > 3) {
    const stem = token.slice(0, -2);
    // "boxes"/"classes"/"wishes" lose the whole "es"; "bikes" only the "s".
    return endsWithSibilant(stem) ? stem : token.slice(0, -1);
  }
  if (token.endsWith('s')) return token.slice(0, -1);

  if (token.endsWith('ing') && token.length > 5) return restore(token.slice(0, -3));
  if (token.endsWith('ed') && token.length > 4) return restore(token.slice(0, -2));
  if (token.endsWith('ly') && token.length > 4) return token.slice(0, -2);

  return token;
}

function endsWithSibilant(stem: string): boolean {
  return /(?:s|x|z|ch|sh)$/.test(stem);
}

const VOWELS = 'aeiou';

const isVowel = (ch: string): boolean => VOWELS.includes(ch);

/** Porter's step-1b repairs, trimmed to the two that earn their keep. */
function restore(stem: string): string {
  const n = stem.length;
  if (n < 2) return stem;

  // "runn" → "run": a doubled final consonant came from the suffix.
  if (stem[n - 1] === stem[n - 2] && !isVowel(stem[n - 1])) return stem.slice(0, -1);

  // "bik" → "bike": a short consonant-vowel-consonant stem lost a silent "e".
  if (
    n === 3 &&
    !isVowel(stem[0]) &&
    isVowel(stem[1]) &&
    !isVowel(stem[2]) &&
    !'wxy'.includes(stem[2])
  ) {
    return `${stem}e`;
  }
  return stem;
}

export type Tokenizer = (text: string) => string[];

/** Build a reusable analyzer. */
export function createTokenizer({
  stopwords = DEFAULT_STOPWORDS,
  stem = lightStem,
  minLength = 2,
}: TokenizerOptions = {}): Tokenizer {
  return (text: string): string[] => {
    const out: string[] = [];
    for (const raw of split(text)) {
      if (!raw) continue;
      const exempt = isUnigramScript(raw.codePointAt(0) ?? 0);
      if (!exempt && raw.length < minLength) continue;
      if (stopwords.has(raw)) continue;
      const stemmed = exempt ? raw : stem(raw);
      if (stemmed) out.push(stemmed);
    }
    return out;
  };
}

/** The default analyzer: fold → drop stopwords → light stem. */
export const tokenize: Tokenizer = createTokenizer();
