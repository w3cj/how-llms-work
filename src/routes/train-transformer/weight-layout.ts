/**
 * SharedArrayBuffer memory layout for cross-thread weight and gradient sharing.
 *
 * Training uses data parallelism: multiple worker threads each run forward+backward
 * on different sequences, accumulating gradients into shared memory. This module
 * defines the flat memory layout so all threads can read weights and write gradients
 * without copying — just different Float32Array views into the same SharedArrayBuffer.
 *
 * Layout order: [shared embeddings] [block 0 weights] [block 1 weights] ... [head weights].
 */
import type { BlockGrads, BlockWeights, TransformerConfig, TransformerGrads, TransformerWeights } from "./transformer.js";
import { BLOCK_KEYS } from "./transformer.js";

export type LayoutEntry = {
  key: string;
  blockIdx: number;
  offset: number;
  length: number;
};

export type WeightLayout = LayoutEntry[];

const SHARED_KEYS = ["tokEmb", "posEmb"] as const;
const HEAD_KEYS = ["lnFGamma", "lnFBeta", "headW", "headB"] as const;

/** Map a weight key name to its size in floats, given the model config. */
function weightSize(key: string, cfg: TransformerConfig): number {
  const { vocabSize: V, contextLen: C, embDim: D, ffDim: F } = cfg;
  const sizes: Record<string, number> = {
    tokEmb: V * D,
    posEmb: C * D,
    ln1Gamma: D,
    ln1Beta: D,
    wQ: D * D,
    bQ: D,
    wK: D * D,
    bK: D,
    wV: D * D,
    bV: D,
    wO: D * D,
    bO: D,
    ln2Gamma: D,
    ln2Beta: D,
    ff1W: D * F,
    ff1B: F,
    ff2W: F * D,
    ff2B: D,
    lnFGamma: D,
    lnFBeta: D,
    headW: D * V,
    headB: V,
  };
  return sizes[key];
}

/** Compute the offset map for packing all model weights into a single flat buffer. */
export function computeLayout(cfg: TransformerConfig): { layout: WeightLayout; totalFloats: number } {
  let offset = 0;
  const layout: WeightLayout = [];

  for (const key of SHARED_KEYS) {
    const length = weightSize(key, cfg);
    layout.push({ key, blockIdx: -1, offset, length });
    offset += length;
  }

  for (let b = 0; b < cfg.numLayers; b++) {
    for (const key of BLOCK_KEYS) {
      const length = weightSize(key, cfg);
      layout.push({ key, blockIdx: b, offset, length });
      offset += length;
    }
  }

  for (const key of HEAD_KEYS) {
    const length = weightSize(key, cfg);
    layout.push({ key, blockIdx: -1, offset, length });
    offset += length;
  }

  return { layout, totalFloats: offset };
}

/** Copy model weights into a new SharedArrayBuffer so worker threads can read them. */
export function packWeightsToSab(w: TransformerWeights, layout: WeightLayout, totalFloats: number): SharedArrayBuffer {
  const sab = new SharedArrayBuffer(totalFloats * 4);
  const buf = new Float32Array(sab);
  for (const { key, blockIdx, offset } of layout) {
    const src = blockIdx === -1
      ? (w as any)[key] as Float32Array
      : (w.blocks[blockIdx] as any)[key] as Float32Array;
    buf.set(src, offset);
  }
  return sab;
}

/** Copy weights back from a SharedArrayBuffer into the structured TransformerWeights object. */
export function copyWeightsFromSab(sab: SharedArrayBuffer, layout: WeightLayout, w: TransformerWeights) {
  const buf = new Float32Array(sab);
  for (const { key, blockIdx, offset, length } of layout) {
    const dst = blockIdx === -1
      ? (w as any)[key] as Float32Array
      : (w.blocks[blockIdx] as any)[key] as Float32Array;
    dst.set(buf.subarray(offset, offset + length));
  }
}

/** Create zero-copy Float32Array views into the shared weight buffer — writes to views update the buffer directly. */
export function createWeightViews(sab: SharedArrayBuffer, layout: WeightLayout, numLayers: number): TransformerWeights {
  const w = { blocks: [] as BlockWeights[] } as TransformerWeights;
  for (let i = 0; i < numLayers; i++) w.blocks.push({} as BlockWeights);

  for (const { key, blockIdx, offset, length } of layout) {
    const view = new Float32Array(sab, offset * 4, length);
    if (blockIdx === -1) {
      (w as any)[key] = view;
    }
    else {
      (w.blocks[blockIdx] as any)[key] = view;
    }
  }
  return w;
}

/** Create zero-copy Float32Array views into a shared gradient buffer — same idea as createWeightViews. */
export function createGradViews(sab: SharedArrayBuffer, layout: WeightLayout, numLayers: number): TransformerGrads {
  const g = { blocks: [] as BlockGrads[] } as TransformerGrads;
  for (let i = 0; i < numLayers; i++) g.blocks.push({} as BlockGrads);

  for (const { key, blockIdx, offset, length } of layout) {
    const view = new Float32Array(sab, offset * 4, length);
    if (blockIdx === -1) {
      (g as any)[key] = view;
    }
    else {
      (g.blocks[blockIdx] as any)[key] = view;
    }
  }
  return g;
}

/** Sum gradients from all worker threads into a single gradient struct — the reduce step of data-parallel training. */
export function sumGradsFromWorkers(
  target: TransformerGrads,
  workerGradSabs: SharedArrayBuffer[],
  layout: WeightLayout,
) {
  const workerBufs = workerGradSabs.map(sab => new Float32Array(sab));
  for (const { key, blockIdx, offset, length } of layout) {
    const dst = blockIdx === -1
      ? (target as any)[key] as Float32Array
      : (target.blocks[blockIdx] as any)[key] as Float32Array;
    for (let i = 0; i < length; i++) {
      let sum = 0;
      for (const buf of workerBufs) sum += buf[offset + i];
      dst[i] = sum;
    }
  }
}
