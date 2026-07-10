/**
 * Shared BPE (Byte Pair Encoding) tokenizer — used by the BPE demo, Skip-gram
 * embeddings, and transformer training.
 *
 * BPE starts with individual characters and iteratively merges the most frequent
 * adjacent pair into a new token. After enough merges, characters become subwords,
 * then whole words. GPT-2, GPT-4, and LLaMA all use this algorithm.
 *
 * @see https://arxiv.org/abs/1508.07909 — Sennrich, Haddow & Birch (2016) "Neural Machine Translation of Rare Words with Subword Units"
 */

const MAX_MERGES = 1000;

/**
 * Regex that splits text into words, spaces, and punctuation — just like real
 * tokenizers do before BPE runs.
 *
 * Real tokenizers (GPT-2, GPT-4) use a pre-tokenization regex to split text
 * into coarse chunks first. BPE then operates *within* each chunk, never across
 * them. This is what prevents cross-word merges like "cat sat" becoming one token.
 *
 * Our pattern splits into: word characters, individual spaces, or individual punctuation.
 *
 * @example
 * "the cat sat!".match(PRE_TOKEN_RE)
 * // => ["the", " ", "cat", " ", "sat", "!"]
 * // BPE will learn merges inside "the", "cat", "sat" independently.
 * // It can never merge "cat" + " " because they're in separate chunks.
 */
export const PRE_TOKEN_RE = /\w+|\s|[^\w\s]/g;

export type Merge = { pair: [string, string]; merged: string; frequency: number };

/**
 * Splits raw text into pre-tokens (words, spaces, punctuation) and counts how
 * often each one appears.
 *
 * This is the key optimization that makes BPE scale to trillions of tokens.
 * Instead of running BPE on the full text, we count unique word types and their
 * frequencies. If "the" appears 50 billion times in the corpus, we only process
 * its character sequence ["t","h","e"] once — but weight its pair counts by
 * 50 billion.
 *
 * @example
 * countWords("the cat and the cat")
 * // => Map { "the" => 2, " " => 4, "cat" => 2, "and" => 1 }
 * // "the" appears twice, so pairs inside it ("t"+"h", "h"+"e") each get weight 2.
 * // We only process 4 unique words instead of 9 tokens.
 */
export function countWords(text: string, regex: RegExp = PRE_TOKEN_RE): Map<string, number> {
  const counts = new Map<string, number>();
  const matches = text.match(regex);
  if (!matches)
    return counts;
  for (const word of matches) {
    counts.set(word, (counts.get(word) || 0) + 1);
  }
  return counts;
}

/**
 * Replaces every occurrence of an adjacent pair with a single merged token.
 *
 * Once we've identified the most frequent pair, this function walks through a
 * token sequence and joins each occurrence into one token. This is what actually
 * "builds up" larger tokens from smaller ones — after enough merges, individual
 * characters become subwords, then whole words.
 *
 * @example
 * mergeTokens(["c", "a", "t", "s"], ["c", "a"], "ca")
 * // => ["ca", "t", "s"]
 * // The "c"+"a" pair is merged into "ca". Next round might merge "ca"+"t" => "cat".
 */
export function mergeTokens(tokens: string[], pair: [string, string], merged: string): string[] {
  const result: string[] = [];
  let i = 0;
  while (i < tokens.length) {
    if (i < tokens.length - 1 && tokens[i] === pair[0] && tokens[i + 1] === pair[1]) {
      result.push(merged);
      i += 2;
    }
    else {
      result.push(tokens[i]);
      i++;
    }
  }
  return result;
}

/**
 * Trains a BPE tokenizer using word frequencies — the same approach as real tokenizers.
 *
 * Instead of operating on the full text as one giant sequence, this works on a
 * vocabulary of unique words, each split into characters. Pair frequencies are
 * weighted by how often each word appears in the corpus.
 *
 * For example, if "cat" appears 100 times, the pair "c"+"a" inside it contributes
 * 100 to the pair frequency — without us needing to store 100 copies of "cat".
 *
 * Each merge is applied within every word independently, so tokens never cross
 * word boundaries. This is how GPT-2, GPT-4, and LLaMA tokenizers all work.
 *
 * Stops when maxMerges is reached or when no pairs remain (every word has been
 * fully merged into a single token). Real tokenizers stop at their target vocab
 * size (e.g., 100k for GPT-4) — same idea, just a bigger number.
 *
 * @example
 * trainBpe(Map { "cat" => 3, "car" => 2 })
 * // Word splits: "cat" => ["c","a","t"], "car" => ["c","a","r"]
 * // Pair "c"+"a" has freq 3+2 = 5 (appears in both words, weighted by count)
 * // Pair "a"+"t" has freq 3, "a"+"r" has freq 2
 * // Step 1: merge "c"+"a" => "ca" in all words
 * //   "cat" => ["ca","t"], "car" => ["ca","r"]
 * // Step 2: merge "ca"+"t" => "cat" (freq 3)
 * //   "cat" => ["cat"], "car" => ["ca","r"]
 * // Step 3: merge "ca"+"r" => "car" (freq 2)
 * //   "cat" => ["cat"], "car" => ["car"]
 * // No pairs left — every word is a single token. Training complete.
 */
export function trainBpe(wordFreqs: Map<string, number>, maxMerges = MAX_MERGES) {
  const wordSplits = new Map<string, string[]>();
  for (const word of wordFreqs.keys()) {
    wordSplits.set(word, [...word]);
  }

  const merges: Merge[] = [];

  for (let step = 0; step < maxMerges; step++) {
    const pairFreq = new Map<string, number>();
    for (const [word, tokens] of wordSplits) {
      const weight = wordFreqs.get(word)!;
      for (let i = 0; i < tokens.length - 1; i++) {
        const key = `${tokens[i]}\0${tokens[i + 1]}`;
        pairFreq.set(key, (pairFreq.get(key) || 0) + weight);
      }
    }

    if (pairFreq.size === 0)
      break;

    let bestKey = "";
    let bestCount = 0;
    for (const [key, count] of pairFreq) {
      if (count > bestCount) {
        bestCount = count;
        bestKey = key;
      }
    }

    const [a, b] = bestKey.split("\0");
    const merged = a + b;

    for (const [word, tokens] of wordSplits) {
      wordSplits.set(word, mergeTokens(tokens, [a, b], merged));
    }

    merges.push({ pair: [a, b], merged, frequency: bestCount });
  }

  return { merges, wordSplits };
}

/**
 * Tokenizes new text by pre-tokenizing then replaying the learned merge rules.
 *
 * This mirrors how a trained tokenizer is actually used at inference time:
 * 1. Pre-tokenize the text into words/spaces/punctuation (same regex as training)
 * 2. Split each pre-token into characters
 * 3. Replay each merge rule in order within each pre-token
 *
 * The order matters! Early merges (like "c"+"a" => "ca") must happen before
 * later ones (like "ca"+"t" => "cat") because later merges depend on tokens
 * that earlier merges created.
 *
 * Because merges are applied within pre-tokens, tokens never cross word boundaries —
 * "cat" and " " are always separate, no matter how many merges we do.
 *
 * @example
 * const merges = [
 *   { pair: ["c", "a"], merged: "ca" },
 *   { pair: ["ca", "t"], merged: "cat" },
 * ];
 * applyMerges("a cat", merges)
 * // Pre-tokenize: ["a", " ", "cat"]
 * // "a" => ["a"] (no merges apply)
 * // " " => [" "] (no merges apply)
 * // "cat": ["c","a","t"] => ["ca","t"] => ["cat"]
 * // => ["a", " ", "cat"]
 */
export function applyMerges(text: string, merges: Merge[], regex: RegExp = PRE_TOKEN_RE): string[] {
  const preTokens = text.match(regex) || [];
  const result: string[] = [];
  for (const preToken of preTokens) {
    let tokens = [...preToken];
    for (const { pair, merged } of merges) {
      tokens = mergeTokens(tokens, pair, merged);
    }
    result.push(...tokens);
  }
  return result;
}

/** Convenience: train BPE directly on a text string. */
export function trainBpeOnText(text: string, maxMerges?: number, regex?: RegExp) {
  return trainBpe(countWords(text, regex), maxMerges);
}
