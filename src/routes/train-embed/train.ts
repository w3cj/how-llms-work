/**
 * Word2Vec Skip-gram with negative sampling — trains word embeddings from scratch.
 *
 * Mikolov et al. (2013) showed that a shallow neural network trained to predict
 * context words from a target word learns vectors where geometry encodes meaning:
 * words used in similar contexts cluster together, and vector arithmetic works
 * (king - man + woman ≈ queen).
 *
 * @see https://arxiv.org/abs/1301.3781 — "Efficient Estimation of Word Representations in Vector Space"
 * @see https://arxiv.org/abs/1310.4546 — "Distributed Representations of Words and Phrases" (negative sampling)
 *
 * The model has two weight matrices (W_in, W_out), each vocab_size × embed_dim.
 * For each (target, context) pair from the corpus, we push their vectors closer.
 * For K random "negative" words, we push their vectors apart. After training,
 * W_in rows are the learned embeddings.
 *
 * Yields InitResult, then EpochResults during training, then a final TrainEmbedResult.
 */
import { writeFile } from "node:fs/promises";
import { applyMerges } from "../../server/lib/bpe.js";
import { cosineSimilarity } from "../../server/lib/math.js";
import { BPE_MERGES, buildVocab, CORPUS, tokenize } from "./corpus.js";

const tick = () => new Promise(resolve => setTimeout(resolve, 0));

// Mulberry32 — fast, seedable 32-bit PRNG
function mulberry32(seed: number): () => number {
  return () => {
    seed |= 0;
    seed = (seed + 0x6D2B79F5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function sigmoid(x: number): number {
  if (x > 6)
    return 1;
  if (x < -6)
    return 0;
  return 1 / (1 + Math.exp(-x));
}

export type InitResult = {
  vocabSize: number;
  sentenceCount: number;
  embeddingDim: number;
  windowSize: number;
  totalPairs: number;
};

export type EpochResult = { epoch: number; loss: number };

export type WordEmbedding = { word: string; vector: number[] };
export type Neighbor = { word: string; nearest: { word: string; score: number }[] };
export type SimilarityPair = { a: string; b: string; score: number };
export type Analogy = { query: string; result: string; score: number };

export type TrainEmbedResult = {
  embeddings: WordEmbedding[];
  neighbors: Neighbor[];
  similarities: SimilarityPair[];
  analogies: Analogy[];
  warnings: string[];
};

type TrainOpts = {
  words: string[];
  epochs: number;
  dimensions: number;
  windowSize: number;
  negativeSamples: number;
};

export async function* trainSkipGram(opts: TrainOpts): AsyncGenerator<InitResult | EpochResult | TrainEmbedResult> {
  const { words, epochs, dimensions: dim, windowSize, negativeSamples } = opts;
  const { wordToIndex, indexToWord, freq } = buildVocab(CORPUS);
  const vocabSize = indexToWord.length;
  const rand = mulberry32(42);

  // Build training pairs: (target, context) indices
  const pairs: [number, number][] = [];
  for (const sentence of CORPUS) {
    const tokens = tokenize(sentence);
    const indices = tokens
      .map(w => wordToIndex.get(w)!)
      .filter(i => i !== undefined) as number[];
    for (let i = 0; i < indices.length; i++) {
      for (let j = Math.max(0, i - windowSize); j <= Math.min(indices.length - 1, i + windowSize); j++) {
        if (j !== i)
          pairs.push([indices[i], indices[j]]);
      }
    }
  }

  yield {
    vocabSize,
    sentenceCount: CORPUS.length,
    embeddingDim: dim,
    windowSize,
    totalPairs: pairs.length,
  };

  // Unigram distribution raised to 0.75 power for negative sampling (Mikolov's trick)
  const unigramPower = new Float64Array(vocabSize);
  let unigramSum = 0;
  for (let i = 0; i < vocabSize; i++) {
    const count = freq.get(indexToWord[i]) ?? 1;
    unigramPower[i] = count ** 0.75;
    unigramSum += unigramPower[i];
  }
  for (let i = 0; i < vocabSize; i++) {
    unigramPower[i] /= unigramSum;
  }

  // Build alias table for O(1) negative sampling
  const cumulative = new Float64Array(vocabSize);
  cumulative[0] = unigramPower[0];
  for (let i = 1; i < vocabSize; i++) {
    cumulative[i] = cumulative[i - 1] + unigramPower[i];
  }

  function sampleNegative(): number {
    const r = rand();
    let lo = 0;
    let hi = vocabSize - 1;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (cumulative[mid] < r)
        lo = mid + 1;
      else
        hi = mid;
    }
    return lo;
  }

  // Initialize weight matrices with small random values
  const scale = 0.5 / dim;
  const wIn = new Float64Array(vocabSize * dim);
  const wOut = new Float64Array(vocabSize * dim);
  for (let i = 0; i < wIn.length; i++) {
    wIn[i] = (rand() - 0.5) * scale;
    wOut[i] = (rand() - 0.5) * scale;
  }

  const step = Math.max(1, Math.floor(epochs / 50));
  const lrStart = 0.025;
  const lrEnd = 0.001;

  // Fisher-Yates shuffle
  function shuffle(arr: [number, number][]) {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = Math.floor(rand() * (i + 1));
      [arr[i], arr[j]] = [arr[j], arr[i]];
    }
  }

  for (let epoch = 0; epoch <= epochs; epoch++) {
    const lr = lrStart - (lrStart - lrEnd) * (epoch / epochs);
    let totalLoss = 0;
    shuffle(pairs);

    for (const [target, context] of pairs) {
      const tOff = target * dim;
      const cOff = context * dim;

      // Positive sample: push target and context closer
      let dot = 0;
      for (let d = 0; d < dim; d++) dot += wIn[tOff + d] * wOut[cOff + d];
      const score = sigmoid(dot);
      const grad = lr * (1 - score);
      for (let d = 0; d < dim; d++) {
        const wInD = wIn[tOff + d];
        wIn[tOff + d] += grad * wOut[cOff + d];
        wOut[cOff + d] += grad * wInD;
      }
      totalLoss += -Math.log(score + 1e-10);

      // Negative samples: push target and random words apart
      for (let n = 0; n < negativeSamples; n++) {
        const neg = sampleNegative();
        if (neg === context)
          continue;
        const nOff = neg * dim;

        let nDot = 0;
        for (let d = 0; d < dim; d++) nDot += wIn[tOff + d] * wOut[nOff + d];
        const nScore = sigmoid(nDot);
        const nGrad = lr * nScore;
        for (let d = 0; d < dim; d++) {
          const wInD = wIn[tOff + d];
          wIn[tOff + d] -= nGrad * wOut[nOff + d];
          wOut[nOff + d] -= nGrad * wInD;
        }
        totalLoss += -Math.log(1 - nScore + 1e-10);
      }
    }

    const loss = totalLoss / pairs.length;

    if (epoch % step === 0 || epoch === epochs) {
      yield { epoch, loss: Math.round(loss * 1000000) / 1000000 };
      await tick();
    }
  }

  // Extract embeddings as regular arrays
  function getVector(wordIdx: number): number[] {
    const vec: number[] = [];
    const off = wordIdx * dim;
    for (let d = 0; d < dim; d++) {
      vec.push(Math.round(wIn[off + d] * 1000000) / 1000000);
    }
    return vec;
  }

  // Build results for query words
  const warnings: string[] = [];
  const queryIndices: number[] = [];

  for (const w of words) {
    const idx = wordToIndex.get(w.toLowerCase());
    if (idx === undefined) {
      const bpeTokens = applyMerges(w.toLowerCase(), BPE_MERGES);
      warnings.push(`"${w}" is not a single BPE token — it splits into [${bpeTokens.join(", ")}]`);
    }
    else {
      queryIndices.push(idx);
    }
  }

  const embeddings: WordEmbedding[] = queryIndices.map(i => ({
    word: indexToWord[i],
    vector: getVector(i),
  }));

  // Nearest neighbors for each query word
  const neighbors: Neighbor[] = queryIndices.map((qi) => {
    const qVec = getVector(qi);
    const scores: { word: string; score: number }[] = [];
    for (let i = 0; i < vocabSize; i++) {
      if (i === qi)
        continue;
      scores.push({
        word: indexToWord[i],
        score: Math.round(cosineSimilarity(qVec, getVector(i)) * 100) / 100,
      });
    }
    scores.sort((a, b) => b.score - a.score);
    return { word: indexToWord[qi], nearest: scores.slice(0, 5) };
  });

  // Pairwise similarity between query words
  const similarities: SimilarityPair[] = [];
  for (let i = 0; i < queryIndices.length; i++) {
    for (let j = i + 1; j < queryIndices.length; j++) {
      similarities.push({
        a: indexToWord[queryIndices[i]],
        b: indexToWord[queryIndices[j]],
        score: Math.round(cosineSimilarity(getVector(queryIndices[i]), getVector(queryIndices[j])) * 100) / 100,
      });
    }
  }

  // Vector analogies: a - b + c ≈ ?
  const ANALOGIES: [string, string, string][] = [
    ["king", "man", "woman"],
    ["queen", "woman", "man"],
    ["prince", "boy", "girl"],
    ["kitten", "cat", "dog"],
    ["puppy", "dog", "cat"],
    ["he", "man", "woman"],
    ["his", "man", "woman"],
  ];

  const analogies: Analogy[] = [];
  for (const [a, b, c] of ANALOGIES) {
    const aIdx = wordToIndex.get(a);
    const bIdx = wordToIndex.get(b);
    const cIdx = wordToIndex.get(c);
    if (aIdx === undefined || bIdx === undefined || cIdx === undefined)
      continue;

    const resultVec: number[] = [];
    for (let d = 0; d < dim; d++) {
      resultVec.push(wIn[aIdx * dim + d] - wIn[bIdx * dim + d] + wIn[cIdx * dim + d]);
    }

    let bestIdx = -1;
    let bestScore = -Infinity;
    const exclude = new Set([aIdx, bIdx, cIdx]);
    for (let i = 0; i < vocabSize; i++) {
      if (exclude.has(i))
        continue;
      const score = cosineSimilarity(resultVec, getVector(i));
      if (score > bestScore) {
        bestScore = score;
        bestIdx = i;
      }
    }

    if (bestIdx >= 0) {
      analogies.push({
        query: `${a} - ${b} + ${c}`,
        result: indexToWord[bestIdx],
        score: Math.round(bestScore * 100) / 100,
      });
    }
  }

  const savedModel = {
    type: "word2vec-skipgram" as const,
    dimensions: dim,
    vocab: indexToWord,
    merges: BPE_MERGES.map(m => ({ pair: m.pair, merged: m.merged })),
    embeddings: Object.fromEntries(
      indexToWord.map((word, i) => [word, getVector(i)]),
    ),
  };
  const weightsPath = new URL("../../../.data/embedding-weights.json", import.meta.url);
  await writeFile(weightsPath, `${JSON.stringify(savedModel, null, 2)}\n`);

  yield { embeddings, neighbors, similarities, analogies, warnings };
}
