/**
 * Training orchestrator for a tiny GPT — tokenizes a corpus, trains with data parallelism, streams results.
 *
 * Radford et al. (2018) showed that a decoder-only transformer pre-trained on raw text
 * learns surprisingly general language representations. This file trains the same
 * architecture from scratch on a small story corpus, demonstrating every step:
 * BPE tokenization, sequence construction, multi-worker gradient accumulation, and
 * autoregressive text generation.
 *
 * @see https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf — Radford et al. (2018) "Improving Language Understanding by Generative Pre-Training" (GPT-1)
 * @see https://arxiv.org/abs/1508.07909 — Sennrich, Haddow & Birch (2016) "Neural Machine Translation of Rare Words with Subword Units" (BPE)
 *
 * Training uses data parallelism via SharedArrayBuffer: each worker thread runs
 * forward+backward on its slice of sequences and accumulates gradients into shared
 * memory. The main thread sums worker gradients and runs one Adam step per epoch.
 *
 * Yields InitResult, then EpochResults during training, then a final TransformerTrainResult.
 */
import type { BlockWeights, TransformerConfig, TransformerWeights } from "./transformer.js";
import { readFile, writeFile } from "node:fs/promises";
import { cpus } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { Worker } from "node:worker_threads";

import { buildSync } from "esbuild";

import { applyMerges, trainBpeOnText } from "../../server/lib/bpe.js";
import { STORIES } from "../train-embed/corpus.js";
import { resetRand } from "./matrix.js";
import { adamUpdate, BLOCK_KEYS, countParams, crossEntropyLoss, forward, generateText, initAdam, initWeights, zeroGrads } from "./transformer.js";
import { computeLayout, copyWeightsFromSab, createWeightViews, packWeightsToSab, sumGradsFromWorkers } from "./weight-layout.js";

/** Yields to the event loop so SSE events can flush during long training loops. */
const tick = () => new Promise(resolve => setTimeout(resolve, 0));
const round = (arr: Float32Array) => Array.from(arr, v => Math.round(v * 1000000) / 1000000);

/** Serialize the full model (weights + config + vocab + BPE merges) to a JSON-cacheable object. */
function serializeModel(
  w: TransformerWeights,
  cfg: TransformerConfig,
  vocab: string[],
  merges: { pair: [string, string]; merged: string }[],
) {
  return {
    type: "decoder-transformer" as const,
    config: cfg,
    vocab,
    merges,
    weights: {
      tokEmb: round(w.tokEmb),
      posEmb: round(w.posEmb),
      blocks: w.blocks.map((bw) => {
        const obj: Record<string, number[]> = {};
        for (const key of BLOCK_KEYS) obj[key] = round(bw[key]);
        return obj;
      }),
      lnFGamma: round(w.lnFGamma),
      lnFBeta: round(w.lnFBeta),
      headW: round(w.headW),
      headB: round(w.headB),
    },
  };
}

/** Try to load pre-trained weights from a cached JSON file, returning null if not found. */
async function tryLoadWeights(path: URL): Promise<TransformerWeights | null> {
  try {
    const json = JSON.parse(await readFile(path, "utf-8"));
    const sw = json.weights;
    return {
      tokEmb: new Float32Array(sw.tokEmb),
      posEmb: new Float32Array(sw.posEmb),
      blocks: sw.blocks.map((b: Record<string, number[]>) => {
        const bw = {} as BlockWeights;
        for (const [k, v] of Object.entries(b)) {
          (bw as any)[k] = new Float32Array(v);
        }
        return bw;
      }),
      lnFGamma: new Float32Array(sw.lnFGamma),
      lnFBeta: new Float32Array(sw.lnFBeta),
      headW: new Float32Array(sw.headW),
      headB: new Float32Array(sw.headB),
    };
  }
  catch {
    return null;
  }
}

export type InitResult = {
  vocabSize: number;
  contextLen: number;
  embeddingDim: number;
  numHeads: number;
  ffDim: number;
  numLayers: number;
  totalParams: number;
  temperature: number;
  topP: number;
  corpusSentences: number;
  trainingSequences: number;
};

export type EpochResult = {
  epoch: number;
  loss: number;
  sample?: string;
};

export type TransformerTrainResult = {
  architecture: string;
  finalLoss: number;
  samples: { epoch: number; text: string }[];
};

type TrainOpts = {
  epochs: number;
  temperature: number;
  topP: number;
  numLayers: number;
  maxTokens: number;
};

/** Bundle the train-worker.ts entry point with esbuild so it can be loaded as a Worker thread. */
function buildWorkerBundle(): string {
  const workerEntry = fileURLToPath(new URL("./train-worker.ts", import.meta.url));
  const outfile = join(fileURLToPath(new URL(".", import.meta.url)), ".train-worker.mjs");
  buildSync({
    entryPoints: [workerEntry],
    bundle: true,
    format: "esm",
    platform: "node",
    outfile,
    write: true,
  });
  return outfile;
}

/**
 * Train a decoder-only transformer from scratch — the main training pipeline.
 *
 * Steps:
 * 1. Build a BPE vocabulary from the story corpus
 * 2. Tokenize all sentences and construct overlapping (input, target) sequences
 * 3. If cached weights exist for this config, load them and skip training
 * 4. Otherwise, spawn N worker threads for data-parallel gradient computation
 * 5. Each epoch: workers compute forward+backward → main thread sums gradients → Adam step
 * 6. Periodically generate sample text to show learning progress
 *
 * Yields InitResult (architecture info), then EpochResult (loss + samples), then TransformerTrainResult.
 */
export async function* trainTransformer(opts: TrainOpts): AsyncGenerator<InitResult | EpochResult | TransformerTrainResult> {
  const { epochs, temperature, topP, numLayers, maxTokens } = opts;
  resetRand();

  const PRE_TOKEN_RE = /\s\w+|[^\w\s]/g;
  const corpusText = STORIES.map(s => ` ${s}`).join("").toLowerCase();
  const { merges: bpeMerges } = trainBpeOnText(corpusText, 1000, PRE_TOKEN_RE);

  function tokenize(text: string): string[] {
    return applyMerges(` ${text}`.toLowerCase(), bpeMerges, PRE_TOKEN_RE);
  }

  const freq = new Map<string, number>();
  for (const sentence of STORIES) {
    for (const tok of tokenize(sentence)) {
      freq.set(tok, (freq.get(tok) ?? 0) + 1);
    }
  }
  const indexToWord: string[] = [];
  const wordToIndex = new Map<string, number>();
  for (const [word] of [...freq.entries()].sort((a, b) => b[1] - a[1])) {
    wordToIndex.set(word, indexToWord.length);
    indexToWord.push(word);
  }
  const vocabSize = indexToWord.length;

  const cfg: TransformerConfig = {
    vocabSize,
    contextLen: 32,
    embDim: 32,
    numHeads: 2,
    ffDim: 128,
    numLayers,
  };

  const allTokenIds: number[] = [];
  for (const sentence of STORIES) {
    for (const tok of tokenize(sentence)) {
      const idx = wordToIndex.get(tok);
      if (idx !== undefined)
        allTokenIds.push(idx);
    }
  }

  const seqLen = Math.min(cfg.contextLen, 16);
  const sequences: { input: number[]; target: number[] }[] = [];
  for (let i = 0; i <= allTokenIds.length - seqLen - 1; i++) {
    sequences.push({
      input: allTokenIds.slice(i, i + seqLen),
      target: allTokenIds.slice(i + 1, i + seqLen + 1),
    });
  }

  const weightsFilename = `transformer-weights-e${epochs}-l${numLayers}-d${cfg.embDim}-h${cfg.numHeads}-ff${cfg.ffDim}-ctx${cfg.contextLen}.json`;
  const weightsPath = new URL(`../../../.data/${weightsFilename}`, import.meta.url);

  const cached = await tryLoadWeights(weightsPath);
  if (cached) {
    const w = cached;
    const seedIds = allTokenIds.slice(0, 3);

    yield {
      vocabSize,
      contextLen: cfg.contextLen,
      embeddingDim: cfg.embDim,
      numHeads: cfg.numHeads,
      ffDim: cfg.ffDim,
      numLayers: cfg.numLayers,
      totalParams: countParams(w),
      temperature,
      topP,
      corpusSentences: STORIES.length,
      trainingSequences: sequences.length,
    };

    let totalLoss = 0;
    for (const seq of sequences) {
      const cache = forward(seq.input, w, cfg);
      totalLoss += crossEntropyLoss(cache.probs, seq.target, seqLen, vocabSize);
    }
    const finalLoss = totalLoss / sequences.length;

    const sample = generateText(w, cfg, seedIds, maxTokens, indexToWord, temperature, topP, seqLen);

    yield {
      architecture: `Decoder-Only Transformer (${cfg.numLayers} layers, ${cfg.embDim}d, ${cfg.numHeads}h, ${cfg.ffDim}ff) — cached`,
      finalLoss: Math.round(finalLoss * 1000000) / 1000000,
      samples: [{ epoch: epochs, text: sample }],
    };
    return;
  }

  const w = initWeights(cfg);
  const adamBuf = initAdam(w);

  yield {
    vocabSize,
    contextLen: cfg.contextLen,
    embeddingDim: cfg.embDim,
    numHeads: cfg.numHeads,
    ffDim: cfg.ffDim,
    numLayers: cfg.numLayers,
    totalParams: countParams(w),
    temperature,
    topP,
    corpusSentences: STORIES.length,
    trainingSequences: sequences.length,
  };

  const step = 10;
  const sampleEvery = 10;
  const lr = 0.001;
  const serializedMerges = bpeMerges.map(m => ({ pair: m.pair, merged: m.merged }));
  const collectedSamples: { epoch: number; text: string }[] = [];
  const seedIds = allTokenIds.slice(0, 3);

  const { layout, totalFloats } = computeLayout(cfg);
  const weightSab = packWeightsToSab(w, layout, totalFloats);

  const workerPath = buildWorkerBundle();

  const numWorkers = Math.max(1, cpus().length - 2);
  const seqsPerWorker = Math.ceil(sequences.length / numWorkers);

  const workerGradSabs: SharedArrayBuffer[] = [];
  const workers: Worker[] = [];

  for (let i = 0; i < numWorkers; i++) {
    const workerSeqs = sequences.slice(i * seqsPerWorker, (i + 1) * seqsPerWorker);
    const gradSab = new SharedArrayBuffer(totalFloats * 4);
    workerGradSabs.push(gradSab);

    const worker = new Worker(workerPath, {
      workerData: {
        weightSab,
        gradSab,
        layout,
        cfg,
        sequences: workerSeqs,
        seqLen,
      },
    });
    workers.push(worker);
  }

  await Promise.all(workers.map(worker =>
    new Promise<void>((resolve) => {
      worker.once("message", (msg: { type: string }) => {
        if (msg.type === "ready")
          resolve();
      });
    }),
  ));

  const grads = zeroGrads(cfg);

  const wShared = createWeightViews(weightSab, layout, cfg.numLayers);

  for (let epoch = 0; epoch <= epochs; epoch++) {
    const results = await Promise.all(workers.map(worker =>
      new Promise<{ loss: number }>((resolve) => {
        worker.once("message", (msg: { type: string; loss: number }) => {
          if (msg.type === "done")
            resolve({ loss: msg.loss });
        });
        worker.postMessage({ type: "compute" });
      }),
    ));

    let totalLoss = 0;
    for (const r of results) totalLoss += r.loss;

    sumGradsFromWorkers(grads, workerGradSabs, layout);
    adamUpdate(wShared, grads, adamBuf, lr, epoch + 1);

    const loss = totalLoss / sequences.length;

    if (epoch % step === 0 || epoch === epochs) {
      let sample: string | undefined;
      if (epoch % sampleEvery === 0 || epoch === epochs) {
        copyWeightsFromSab(weightSab, layout, w);
        sample = generateText(w, cfg, seedIds, maxTokens, indexToWord, temperature, topP, seqLen);
        collectedSamples.push({ epoch, text: sample });
      }
      yield { epoch, loss: Math.round(loss * 1000000) / 1000000, sample };
      await tick();
    }

    if (epoch > 0 && epoch % 100 === 0) {
      copyWeightsFromSab(weightSab, layout, w);
      const checkpointPath = new URL(`../../../.data/transformer-weights-e${epoch}-l${numLayers}-d${cfg.embDim}-h${cfg.numHeads}-ff${cfg.ffDim}-ctx${cfg.contextLen}.json`, import.meta.url);
      await writeFile(checkpointPath, `${JSON.stringify(serializeModel(w, cfg, indexToWord, serializedMerges), null, 2)}\n`);
    }
  }

  await Promise.all(workers.map(worker => worker.terminate()));

  copyWeightsFromSab(weightSab, layout, w);

  const finalLoss = (() => {
    let total = 0;
    for (const seq of sequences) {
      const cache = forward(seq.input, w, cfg);
      total += crossEntropyLoss(cache.probs, seq.target, seqLen, vocabSize);
    }
    return total / sequences.length;
  })();

  await writeFile(weightsPath, `${JSON.stringify(serializeModel(w, cfg, indexToWord, serializedMerges), null, 2)}\n`);

  yield {
    architecture: `Decoder-Only Transformer (${cfg.numLayers} layers, ${cfg.embDim}d, ${cfg.numHeads}h, ${cfg.ffDim}ff)`,
    finalLoss: Math.round(finalLoss * 1000000) / 1000000,
    samples: collectedSamples,
  };
}
