/**
 * Worker thread for data-parallel transformer training.
 *
 * Each worker owns a slice of the training sequences. On a "compute" message, it
 * zeros its gradient buffer, runs forward+backward on every sequence in its slice,
 * and posts the total loss back. Gradients accumulate directly into a SharedArrayBuffer
 * that the main thread reads — no serialization or copying needed.
 */
import type { TransformerConfig } from "./transformer.js";
import type { WeightLayout } from "./weight-layout.js";
import { parentPort, workerData } from "node:worker_threads";
import { backward, crossEntropyLoss, forward } from "./transformer.js";
import { createGradViews, createWeightViews } from "./weight-layout.js";

const {
  weightSab,
  gradSab,
  layout,
  cfg,
  sequences,
  seqLen,
} = workerData as {
  weightSab: SharedArrayBuffer;
  gradSab: SharedArrayBuffer;
  layout: WeightLayout;
  cfg: TransformerConfig;
  sequences: { input: number[]; target: number[] }[];
  seqLen: number;
};

const w = createWeightViews(weightSab, layout, cfg.numLayers);
const grads = createGradViews(gradSab, layout, cfg.numLayers);

parentPort!.on("message", (msg: { type: string }) => {
  if (msg.type === "compute") {
    new Float32Array(gradSab).fill(0);
    let totalLoss = 0;

    for (const seq of sequences) {
      const cache = forward(seq.input, w, cfg);
      totalLoss += crossEntropyLoss(cache.probs, seq.target, seqLen, cfg.vocabSize);
      backward(cache, seq.target, w, cfg, grads);
    }

    parentPort!.postMessage({ type: "done", loss: totalLoss });
  }
});

parentPort!.postMessage({ type: "ready" });
