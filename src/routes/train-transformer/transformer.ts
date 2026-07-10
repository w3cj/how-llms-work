/**
 * A decoder-only transformer implemented entirely from scratch — no ML libraries.
 *
 * Vaswani et al. (2017) introduced the transformer in "Attention Is All You Need."
 * This file implements every operation by hand: forward pass, backpropagation through
 * attention, layer norm, feed-forward layers, and embedding lookups. The code is
 * intentionally verbose so you can point at any line and explain the math.
 *
 * @see https://arxiv.org/abs/1706.03762 — "Attention Is All You Need"
 *
 * Architecture (decoder-only, like GPT):
 *   token embeddings + positional embeddings
 *   → N transformer blocks (layer norm → multi-head causal self-attention → residual
 *                           → layer norm → feed-forward → residual)
 *   → layer norm → linear head → logits
 *
 * Every weight is a flat Float32Array. Every gradient is computed analytically.
 */
import { addBias, addInPlace, matmul, matmulTransA, matmulTransB, ones, rand, sumCols, xavierInit, zeros } from "./matrix.js";

// ─── Types ───────────────────────────────────────────────────────────────────

export type TransformerConfig = {
  vocabSize: number;
  contextLen: number;
  embDim: number;
  numHeads: number;
  ffDim: number;
  numLayers: number;
};

export type BlockWeights = {
  ln1Gamma: Float32Array;
  ln1Beta: Float32Array;
  wQ: Float32Array;
  bQ: Float32Array;
  wK: Float32Array;
  bK: Float32Array;
  wV: Float32Array;
  bV: Float32Array;
  wO: Float32Array;
  bO: Float32Array;
  ln2Gamma: Float32Array;
  ln2Beta: Float32Array;
  ff1W: Float32Array;
  ff1B: Float32Array;
  ff2W: Float32Array;
  ff2B: Float32Array;
};

export type TransformerWeights = {
  tokEmb: Float32Array;
  posEmb: Float32Array;
  blocks: BlockWeights[];
  lnFGamma: Float32Array;
  lnFBeta: Float32Array;
  headW: Float32Array;
  headB: Float32Array;
};

export type BlockGrads = { [K in keyof BlockWeights]: Float32Array };
export type TransformerGrads = {
  tokEmb: Float32Array;
  posEmb: Float32Array;
  blocks: BlockGrads[];
  lnFGamma: Float32Array;
  lnFBeta: Float32Array;
  headW: Float32Array;
  headB: Float32Array;
};

type BlockAdamBuf = { [K in keyof BlockWeights]: { m: Float32Array; v: Float32Array } };
type AdamBuf = {
  tokEmb: { m: Float32Array; v: Float32Array };
  posEmb: { m: Float32Array; v: Float32Array };
  blocks: BlockAdamBuf[];
  lnFGamma: { m: Float32Array; v: Float32Array };
  lnFBeta: { m: Float32Array; v: Float32Array };
  headW: { m: Float32Array; v: Float32Array };
  headB: { m: Float32Array; v: Float32Array };
};

export const BLOCK_KEYS: (keyof BlockWeights)[] = [
  "ln1Gamma",
  "ln1Beta",
  "wQ",
  "bQ",
  "wK",
  "bK",
  "wV",
  "bV",
  "wO",
  "bO",
  "ln2Gamma",
  "ln2Beta",
  "ff1W",
  "ff1B",
  "ff2W",
  "ff2B",
];

type BlockCache = {
  input: Float32Array;
  ln1Out: Float32Array;
  ln1Mean: Float32Array;
  ln1Var: Float32Array;
  ln1Hat: Float32Array;
  Q: Float32Array;
  K: Float32Array;
  V: Float32Array;
  attnWeights: Float32Array;
  attnOut: Float32Array;
  x1: Float32Array;
  ln2Out: Float32Array;
  ln2Mean: Float32Array;
  ln2Var: Float32Array;
  ln2Hat: Float32Array;
  ff1Out: Float32Array;
  reluOut: Float32Array;
  ff2Out: Float32Array;
  output: Float32Array;
};

export type ForwardCache = {
  tokenIds: number[];
  x0: Float32Array;
  blockCaches: BlockCache[];
  lnFOut: Float32Array;
  lnFMean: Float32Array;
  lnFVar: Float32Array;
  lnFHat: Float32Array;
  logits: Float32Array;
  probs: Float32Array;
};

// ─── Initialization ──────────────────────────────────────────────────────────

/** Create one transformer block's weight matrices, initialized with Xavier/Glorot. */
function initBlockWeights(D: number, F: number): BlockWeights {
  return {
    ln1Gamma: ones(D),
    ln1Beta: zeros(D),
    wQ: xavierInit(D * D, D, D),
    bQ: zeros(D),
    wK: xavierInit(D * D, D, D),
    bK: zeros(D),
    wV: xavierInit(D * D, D, D),
    bV: zeros(D),
    wO: xavierInit(D * D, D, D),
    bO: zeros(D),
    ln2Gamma: ones(D),
    ln2Beta: zeros(D),
    ff1W: xavierInit(D * F, D, F),
    ff1B: zeros(F),
    ff2W: xavierInit(F * D, F, D),
    ff2B: zeros(D),
  };
}

/** Allocate zeroed gradient buffers matching one block's weight shapes. */
function zeroBlockGrads(D: number, F: number): BlockGrads {
  return {
    ln1Gamma: zeros(D),
    ln1Beta: zeros(D),
    wQ: zeros(D * D),
    bQ: zeros(D),
    wK: zeros(D * D),
    bK: zeros(D),
    wV: zeros(D * D),
    bV: zeros(D),
    wO: zeros(D * D),
    bO: zeros(D),
    ln2Gamma: zeros(D),
    ln2Beta: zeros(D),
    ff1W: zeros(D * F),
    ff1B: zeros(F),
    ff2W: zeros(F * D),
    ff2B: zeros(D),
  };
}

/**
 * Initialize all transformer weights — token/positional embeddings, N blocks, final layer norm, and output head.
 *
 * Every weight matrix uses Xavier/Glorot uniform initialization: values drawn from
 * [-sqrt(6/(fanIn+fanOut)), +sqrt(6/(fanIn+fanOut))]. This keeps activations and
 * gradients at a reasonable scale so training doesn't explode or vanish in the first steps.
 *
 * @see https://proceedings.mlr.press/v9/glorot10a.html — Glorot & Bengio (2010) "Understanding the difficulty of training deep feedforward neural networks"
 */
export function initWeights(cfg: TransformerConfig): TransformerWeights {
  const { vocabSize: V, contextLen: C, embDim: D, ffDim: F, numLayers } = cfg;
  const blocks: BlockWeights[] = [];
  for (let i = 0; i < numLayers; i++) {
    blocks.push(initBlockWeights(D, F));
  }
  return {
    tokEmb: xavierInit(V * D, V, D),
    posEmb: xavierInit(C * D, C, D),
    blocks,
    lnFGamma: ones(D),
    lnFBeta: zeros(D),
    headW: xavierInit(D * V, D, V),
    headB: zeros(V),
  };
}

/** Allocate zeroed gradient buffers for every parameter in the model. */
export function zeroGrads(cfg: TransformerConfig): TransformerGrads {
  const { vocabSize: V, contextLen: C, embDim: D, ffDim: F, numLayers } = cfg;
  const blocks: BlockGrads[] = [];
  for (let i = 0; i < numLayers; i++) {
    blocks.push(zeroBlockGrads(D, F));
  }
  return {
    tokEmb: zeros(V * D),
    posEmb: zeros(C * D),
    blocks,
    lnFGamma: zeros(D),
    lnFBeta: zeros(D),
    headW: zeros(D * V),
    headB: zeros(V),
  };
}

/**
 * Create Adam optimizer state — first-moment (m) and second-moment (v) buffers for every parameter.
 *
 * Adam tracks an exponential moving average of each gradient (m) and each squared gradient (v).
 * These running averages let Adam adapt the learning rate per-parameter: parameters with
 * consistently large gradients get smaller steps, noisy ones get smoothed out.
 *
 * @see https://arxiv.org/abs/1412.6980 — Kingma & Ba (2014) "Adam: A Method for Stochastic Optimization"
 */
export function initAdam(w: TransformerWeights): AdamBuf {
  function adamPair(arr: Float32Array) {
    return { m: zeros(arr.length), v: zeros(arr.length) };
  }
  const blocks: BlockAdamBuf[] = w.blocks.map((bw) => {
    const buf = {} as BlockAdamBuf;
    for (const key of BLOCK_KEYS) {
      buf[key] = adamPair(bw[key]);
    }
    return buf;
  });
  return {
    tokEmb: adamPair(w.tokEmb),
    posEmb: adamPair(w.posEmb),
    blocks,
    lnFGamma: adamPair(w.lnFGamma),
    lnFBeta: adamPair(w.lnFBeta),
    headW: adamPair(w.headW),
    headB: adamPair(w.headB),
  };
}

/** Count the total number of trainable parameters (floats) in the model. */
export function countParams(w: TransformerWeights): number {
  let total = w.tokEmb.length + w.posEmb.length + w.lnFGamma.length + w.lnFBeta.length + w.headW.length + w.headB.length;
  for (const bw of w.blocks) {
    for (const key of BLOCK_KEYS) {
      total += bw[key].length;
    }
  }
  return total;
}

// ─── Layer Norm ──────────────────────────────────────────────────────────────

/**
 * Layer normalization forward pass — normalize each position's activations to zero mean, unit variance.
 *
 * Unlike batch norm (which normalizes across the batch), layer norm normalizes across
 * the embedding dimension for each token independently. This makes it stable for
 * autoregressive models where sequence lengths vary.
 *
 * @see https://arxiv.org/abs/1607.06450 — Ba, Kiros & Hinton (2016) "Layer Normalization"
 */
function layerNormForward(
  x: Float32Array,
  gamma: Float32Array,
  beta: Float32Array,
  seqLen: number,
  D: number,
  mean: Float32Array,
  variance: Float32Array,
  xHat: Float32Array,
): Float32Array {
  const out = new Float32Array(seqLen * D);
  const eps = 1e-5;

  for (let i = 0; i < seqLen; i++) {
    let m = 0;
    for (let d = 0; d < D; d++) m += x[i * D + d];
    m /= D;
    mean[i] = m;

    let v = 0;
    for (let d = 0; d < D; d++) {
      const diff = x[i * D + d] - m;
      v += diff * diff;
    }
    v /= D;
    variance[i] = v;

    const invStd = 1 / Math.sqrt(v + eps);
    for (let d = 0; d < D; d++) {
      const hat = (x[i * D + d] - m) * invStd;
      xHat[i * D + d] = hat;
      out[i * D + d] = gamma[d] * hat + beta[d];
    }
  }
  return out;
}

/**
 * Layer normalization backward pass — computes gradients for input, gamma, and beta.
 *
 * The derivative is more complex than a simple elementwise function because the mean
 * and variance depend on all elements in the row, creating cross-element dependencies.
 */
function layerNormBackward(
  dOut: Float32Array,
  xHat: Float32Array,
  gamma: Float32Array,
  mean: Float32Array,
  variance: Float32Array,
  x: Float32Array,
  seqLen: number,
  D: number,
  dGamma: Float32Array,
  dBeta: Float32Array,
): Float32Array {
  const dX = new Float32Array(seqLen * D);
  const eps = 1e-5;

  for (let i = 0; i < seqLen; i++) {
    const invStd = 1 / Math.sqrt(variance[i] + eps);

    let sumDxHat = 0;
    let sumDxHatXHat = 0;
    for (let d = 0; d < D; d++) {
      const dxh = dOut[i * D + d] * gamma[d];
      sumDxHat += dxh;
      sumDxHatXHat += dxh * xHat[i * D + d];
    }

    for (let d = 0; d < D; d++) {
      const dxh = dOut[i * D + d] * gamma[d];
      dX[i * D + d] = invStd * (dxh - sumDxHat / D - xHat[i * D + d] * sumDxHatXHat / D);
    }
  }

  for (let i = 0; i < seqLen; i++) {
    for (let d = 0; d < D; d++) {
      dGamma[d] += dOut[i * D + d] * xHat[i * D + d];
      dBeta[d] += dOut[i * D + d];
    }
  }

  return dX;
}

// ─── Softmax (row-wise) ─────────────────────────────────────────────────────

/** Numerically stable row-wise softmax — subtracts the row max before exponentiating to prevent overflow. */
function softmaxRows(x: Float32Array, rows: number, cols: number): Float32Array {
  const out = new Float32Array(rows * cols);
  for (let i = 0; i < rows; i++) {
    let max = -Infinity;
    for (let j = 0; j < cols; j++) {
      if (x[i * cols + j] > max)
        max = x[i * cols + j];
    }
    let sum = 0;
    for (let j = 0; j < cols; j++) {
      out[i * cols + j] = Math.exp(x[i * cols + j] - max);
      sum += out[i * cols + j];
    }
    for (let j = 0; j < cols; j++) {
      out[i * cols + j] /= sum;
    }
  }
  return out;
}

// ─── Block Forward / Backward ───────────────────────────────────────────────

/**
 * Forward pass through one transformer block — the repeating unit of the architecture.
 *
 * Each block applies:
 *   1. Layer norm → multi-head causal self-attention → add residual
 *   2. Layer norm → feed-forward network (expand → ReLU → project) → add residual
 *
 * The causal mask ensures position i can only attend to positions ≤ i, which is what
 * makes this a decoder (autoregressive) model — it can't peek at future tokens.
 */
function blockForward(input: Float32Array, bw: BlockWeights, cfg: TransformerConfig): BlockCache {
  const { embDim: D, numHeads: H, ffDim: F } = cfg;
  const S = input.length / D;
  const headDim = D / H;

  const ln1Mean = new Float32Array(S);
  const ln1Var = new Float32Array(S);
  const ln1Hat = new Float32Array(S * D);
  const ln1Out = layerNormForward(input, bw.ln1Gamma, bw.ln1Beta, S, D, ln1Mean, ln1Var, ln1Hat);

  const Q = matmul(ln1Out, bw.wQ, S, D, D);
  addBias(Q, bw.bQ, S, D);
  const K = matmul(ln1Out, bw.wK, S, D, D);
  addBias(K, bw.bK, S, D);
  const Va = matmul(ln1Out, bw.wV, S, D, D);
  addBias(Va, bw.bV, S, D);

  const attnWeights = new Float32Array(H * S * S);
  const attnVals = new Float32Array(S * D);
  const scale = 1 / Math.sqrt(headDim);

  for (let h = 0; h < H; h++) {
    const scores = new Float32Array(S * S);
    for (let i = 0; i < S; i++) {
      for (let j = 0; j < S; j++) {
        if (j > i) {
          scores[i * S + j] = -Infinity;
        }
        else {
          let dot = 0;
          for (let d = 0; d < headDim; d++) {
            dot += Q[i * D + h * headDim + d] * K[j * D + h * headDim + d];
          }
          scores[i * S + j] = dot * scale;
        }
      }
    }

    const weights = softmaxRows(scores, S, S);
    for (let idx = 0; idx < S * S; idx++) {
      attnWeights[h * S * S + idx] = weights[idx];
    }

    for (let i = 0; i < S; i++) {
      for (let d = 0; d < headDim; d++) {
        let sum = 0;
        for (let j = 0; j <= i; j++) {
          sum += weights[i * S + j] * Va[j * D + h * headDim + d];
        }
        attnVals[i * D + h * headDim + d] = sum;
      }
    }
  }

  const attnOut = matmul(attnVals, bw.wO, S, D, D);
  addBias(attnOut, bw.bO, S, D);

  const x1 = new Float32Array(S * D);
  for (let i = 0; i < S * D; i++) x1[i] = input[i] + attnOut[i];

  const ln2Mean = new Float32Array(S);
  const ln2Var = new Float32Array(S);
  const ln2Hat = new Float32Array(S * D);
  const ln2Out = layerNormForward(x1, bw.ln2Gamma, bw.ln2Beta, S, D, ln2Mean, ln2Var, ln2Hat);

  const ff1Out = matmul(ln2Out, bw.ff1W, S, D, F);
  addBias(ff1Out, bw.ff1B, S, F);

  const reluOut = new Float32Array(S * F);
  for (let i = 0; i < S * F; i++) reluOut[i] = ff1Out[i] > 0 ? ff1Out[i] : 0;

  const ff2Out = matmul(reluOut, bw.ff2W, S, F, D);
  addBias(ff2Out, bw.ff2B, S, D);

  const output = new Float32Array(S * D);
  for (let i = 0; i < S * D; i++) output[i] = x1[i] + ff2Out[i];

  return {
    input,
    ln1Out,
    ln1Mean,
    ln1Var,
    ln1Hat,
    Q,
    K,
    V: Va,
    attnWeights,
    attnOut,
    x1,
    ln2Out,
    ln2Mean,
    ln2Var,
    ln2Hat,
    ff1Out,
    reluOut,
    ff2Out,
    output,
  };
}

/**
 * Backward pass through one transformer block — computes gradients for all block parameters.
 *
 * Mirrors blockForward in reverse: backprop through feed-forward, then attention,
 * accumulating gradients into bg. Returns dInput so the previous block (or embeddings)
 * can continue the chain.
 */
function blockBackward(
  dOutput: Float32Array,
  bc: BlockCache,
  bw: BlockWeights,
  cfg: TransformerConfig,
  bg: BlockGrads,
): Float32Array {
  const { embDim: D, numHeads: H, ffDim: F } = cfg;
  const S = bc.input.length / D;
  const headDim = D / H;

  const dX1 = new Float32Array(dOutput);
  const dFF2Out = dOutput;

  addInPlace(bg.ff2B, sumCols(dFF2Out, S, D));
  addInPlace(bg.ff2W, matmulTransA(bc.reluOut, dFF2Out, S, F, D));
  const dRelu = matmulTransB(dFF2Out, bw.ff2W, S, D, F);

  const dFF1Out = new Float32Array(S * F);
  for (let i = 0; i < S * F; i++) {
    dFF1Out[i] = bc.ff1Out[i] > 0 ? dRelu[i] : 0;
  }

  addInPlace(bg.ff1B, sumCols(dFF1Out, S, F));
  addInPlace(bg.ff1W, matmulTransA(bc.ln2Out, dFF1Out, S, D, F));
  const dLn2Out = matmulTransB(dFF1Out, bw.ff1W, S, F, D);

  const dLn2In = layerNormBackward(dLn2Out, bc.ln2Hat, bw.ln2Gamma, bc.ln2Mean, bc.ln2Var, bc.x1, S, D, bg.ln2Gamma, bg.ln2Beta);

  const dInput = new Float32Array(S * D);
  for (let i = 0; i < S * D; i++) {
    dInput[i] = dX1[i] + dLn2In[i];
  }
  const dAttnOut = new Float32Array(S * D);
  for (let i = 0; i < S * D; i++) {
    dAttnOut[i] = dX1[i] + dLn2In[i];
  }

  addInPlace(bg.bO, sumCols(dAttnOut, S, D));

  // Reconstruct attnVals from attention weights and V for the wO gradient
  const attnVals = new Float32Array(S * D);
  for (let h = 0; h < H; h++) {
    for (let i = 0; i < S; i++) {
      for (let d = 0; d < headDim; d++) {
        let sum = 0;
        for (let j = 0; j <= i; j++) {
          sum += bc.attnWeights[h * S * S + i * S + j] * bc.V[j * D + h * headDim + d];
        }
        attnVals[i * D + h * headDim + d] = sum;
      }
    }
  }

  addInPlace(bg.wO, matmulTransA(attnVals, dAttnOut, S, D, D));
  const dAttnVals = matmulTransB(dAttnOut, bw.wO, S, D, D);

  const dQ = new Float32Array(S * D);
  const dK = new Float32Array(S * D);
  const dV = new Float32Array(S * D);
  const scale = 1 / Math.sqrt(headDim);

  for (let h = 0; h < H; h++) {
    const dWeights = new Float32Array(S * S);

    for (let i = 0; i < S; i++) {
      for (let j = 0; j <= i; j++) {
        let dot = 0;
        for (let d = 0; d < headDim; d++) {
          dot += dAttnVals[i * D + h * headDim + d] * bc.V[j * D + h * headDim + d];
        }
        dWeights[i * S + j] = dot;
      }

      for (let d = 0; d < headDim; d++) {
        for (let j = 0; j <= i; j++) {
          dV[j * D + h * headDim + d] += bc.attnWeights[h * S * S + i * S + j] * dAttnVals[i * D + h * headDim + d];
        }
      }
    }

    const dScores = new Float32Array(S * S);
    for (let i = 0; i < S; i++) {
      let dotSum = 0;
      for (let j = 0; j <= i; j++) {
        dotSum += dWeights[i * S + j] * bc.attnWeights[h * S * S + i * S + j];
      }
      for (let j = 0; j <= i; j++) {
        dScores[i * S + j] = bc.attnWeights[h * S * S + i * S + j] * (dWeights[i * S + j] - dotSum) * scale;
      }
    }

    for (let i = 0; i < S; i++) {
      for (let j = 0; j <= i; j++) {
        for (let d = 0; d < headDim; d++) {
          dQ[i * D + h * headDim + d] += dScores[i * S + j] * bc.K[j * D + h * headDim + d];
          dK[j * D + h * headDim + d] += dScores[i * S + j] * bc.Q[i * D + h * headDim + d];
        }
      }
    }
  }

  addInPlace(bg.bQ, sumCols(dQ, S, D));
  addInPlace(bg.wQ, matmulTransA(bc.ln1Out, dQ, S, D, D));
  addInPlace(bg.bK, sumCols(dK, S, D));
  addInPlace(bg.wK, matmulTransA(bc.ln1Out, dK, S, D, D));
  addInPlace(bg.bV, sumCols(dV, S, D));
  addInPlace(bg.wV, matmulTransA(bc.ln1Out, dV, S, D, D));

  const dLn1Out = new Float32Array(S * D);
  addInPlace(dLn1Out, matmulTransB(dQ, bw.wQ, S, D, D));
  addInPlace(dLn1Out, matmulTransB(dK, bw.wK, S, D, D));
  addInPlace(dLn1Out, matmulTransB(dV, bw.wV, S, D, D));

  const dLn1In = layerNormBackward(dLn1Out, bc.ln1Hat, bw.ln1Gamma, bc.ln1Mean, bc.ln1Var, bc.input, S, D, bg.ln1Gamma, bg.ln1Beta);

  for (let i = 0; i < S * D; i++) {
    dInput[i] += dLn1In[i];
  }

  return dInput;
}

// ─── Forward Pass ────────────────────────────────────────────────────────────

/**
 * Full forward pass — token IDs in, probability distribution over vocabulary out.
 *
 * Pipeline: look up token + positional embeddings → pass through N transformer blocks
 * → final layer norm → linear projection to vocab size → softmax → probabilities.
 *
 * Returns a ForwardCache containing every intermediate activation, which backward()
 * needs to compute gradients. This is the memory cost of backpropagation — you have
 * to remember everything from the forward pass.
 */
export function forward(tokenIds: number[], w: TransformerWeights, cfg: TransformerConfig): ForwardCache {
  const { vocabSize: V, embDim: D } = cfg;
  const S = tokenIds.length;

  const x0 = new Float32Array(S * D);
  for (let i = 0; i < S; i++) {
    for (let d = 0; d < D; d++) {
      x0[i * D + d] = w.tokEmb[tokenIds[i] * D + d] + w.posEmb[i * D + d];
    }
  }

  const blockCaches: BlockCache[] = [];
  let x: Float32Array<ArrayBufferLike> = x0;
  for (let i = 0; i < w.blocks.length; i++) {
    const bc = blockForward(x, w.blocks[i], cfg);
    blockCaches.push(bc);
    x = bc.output;
  }

  const lnFMean = new Float32Array(S);
  const lnFVar = new Float32Array(S);
  const lnFHat = new Float32Array(S * D);
  const lnFOut = layerNormForward(x, w.lnFGamma, w.lnFBeta, S, D, lnFMean, lnFVar, lnFHat);

  const logits = matmul(lnFOut, w.headW, S, D, V);
  addBias(logits, w.headB, S, V);

  const probs = softmaxRows(logits, S, V);

  return {
    tokenIds,
    x0,
    blockCaches,
    lnFOut,
    lnFMean,
    lnFVar,
    lnFHat,
    logits,
    probs,
  };
}

// ─── Loss ────────────────────────────────────────────────────────────────────

/**
 * Cross-entropy loss — measures how far the model's predictions are from the true next tokens.
 *
 * For each position, we look up the probability the model assigned to the correct token
 * and take -log of it. A perfect prediction (probability 1.0) gives loss 0; a terrible
 * prediction (near 0) gives a huge loss. The average across all positions is the loss
 * value that training tries to minimize.
 */
export function crossEntropyLoss(probs: Float32Array, targets: number[], seqLen: number, V: number): number {
  let loss = 0;
  for (let i = 0; i < seqLen; i++) {
    loss -= Math.log(probs[i * V + targets[i]] + 1e-10);
  }
  return loss / seqLen;
}

// ─── Backward Pass ───────────────────────────────────────────────────────────

/**
 * Full backward pass — compute gradients for every parameter in the model.
 *
 * Starting from the cross-entropy loss, gradients flow backward through the output head,
 * final layer norm, each transformer block (in reverse order), and finally into the
 * token and positional embeddings. This is backpropagation — the same algorithm
 * Rumelhart, Hinton & Williams described in 1986, applied to a transformer.
 *
 * Gradients are accumulated (+=) into the grads struct, not overwritten. This allows
 * multiple sequences to contribute gradients before a single optimizer step.
 */
export function backward(
  cache: ForwardCache,
  targets: number[],
  w: TransformerWeights,
  cfg: TransformerConfig,
  grads: TransformerGrads,
) {
  const { vocabSize: V, embDim: D } = cfg;
  const S = cache.tokenIds.length;

  const dLogits = new Float32Array(S * V);
  for (let i = 0; i < S; i++) {
    for (let j = 0; j < V; j++) {
      dLogits[i * V + j] = cache.probs[i * V + j] / S;
    }
    dLogits[i * V + targets[i]] -= 1 / S;
  }

  addInPlace(grads.headB, sumCols(dLogits, S, V));
  addInPlace(grads.headW, matmulTransA(cache.lnFOut, dLogits, S, D, V));
  let dX = matmulTransB(dLogits, w.headW, S, V, D);

  const lastBlockOutput = cache.blockCaches.length > 0
    ? cache.blockCaches[cache.blockCaches.length - 1].output
    : cache.x0;
  dX = layerNormBackward(dX, cache.lnFHat, w.lnFGamma, cache.lnFMean, cache.lnFVar, lastBlockOutput, S, D, grads.lnFGamma, grads.lnFBeta);

  for (let i = cache.blockCaches.length - 1; i >= 0; i--) {
    dX = blockBackward(dX, cache.blockCaches[i], w.blocks[i], cfg, grads.blocks[i]);
  }

  for (let i = 0; i < S; i++) {
    const tokIdx = cache.tokenIds[i];
    for (let d = 0; d < D; d++) {
      grads.tokEmb[tokIdx * D + d] += dX[i * D + d];
      grads.posEmb[i * D + d] += dX[i * D + d];
    }
  }
}

// ─── Adam Optimizer ──────────────────────────────────────────────────────────

/** Update a single parameter array using Adam — bias-corrected first and second moment estimates. */
function adamUpdateParam(
  param: Float32Array,
  grad: Float32Array,
  buf: { m: Float32Array; v: Float32Array },
  lr: number,
  bc1: number,
  bc2: number,
  beta1: number,
  beta2: number,
  eps: number,
) {
  const { m, v } = buf;
  for (let i = 0; i < param.length; i++) {
    m[i] = beta1 * m[i] + (1 - beta1) * grad[i];
    v[i] = beta2 * v[i] + (1 - beta2) * grad[i] * grad[i];
    const mHat = m[i] / bc1;
    const vHat = v[i] / bc2;
    param[i] -= lr * mHat / (Math.sqrt(vHat) + eps);
  }
}

/**
 * Adam optimizer step — update all model weights using adaptive learning rates.
 *
 * Adam (Adaptive Moment Estimation) maintains a per-parameter running average of
 * the gradient (first moment, m) and squared gradient (second moment, v). At each step,
 * it computes bias-corrected estimates and updates: param -= lr * m̂ / (√v̂ + ε).
 * This gives each parameter its own effective learning rate, making training more stable
 * than plain SGD.
 *
 * @see https://arxiv.org/abs/1412.6980 — Kingma & Ba (2014) "Adam: A Method for Stochastic Optimization"
 */
export function adamUpdate(
  w: TransformerWeights,
  grads: TransformerGrads,
  buf: AdamBuf,
  lr: number,
  t: number,
  beta1 = 0.9,
  beta2 = 0.999,
  eps = 1e-8,
) {
  const bc1 = 1 - beta1 ** t;
  const bc2 = 1 - beta2 ** t;
  const args = [lr, bc1, bc2, beta1, beta2, eps] as const;

  adamUpdateParam(w.tokEmb, grads.tokEmb, buf.tokEmb, ...args);
  adamUpdateParam(w.posEmb, grads.posEmb, buf.posEmb, ...args);

  for (let i = 0; i < w.blocks.length; i++) {
    for (const key of BLOCK_KEYS) {
      adamUpdateParam(w.blocks[i][key], grads.blocks[i][key], buf.blocks[i][key], ...args);
    }
  }

  adamUpdateParam(w.lnFGamma, grads.lnFGamma, buf.lnFGamma, ...args);
  adamUpdateParam(w.lnFBeta, grads.lnFBeta, buf.lnFBeta, ...args);
  adamUpdateParam(w.headW, grads.headW, buf.headW, ...args);
  adamUpdateParam(w.headB, grads.headB, buf.headB, ...args);
}

// ─── Text Generation ─────────────────────────────────────────────────────────

/**
 * Autoregressive text generation — feed tokens in, sample the next one, repeat.
 *
 * At each step, the model runs a forward pass on the current token sequence,
 * looks at the logits for the last position, applies temperature scaling (higher =
 * more random), then uses nucleus (top-p) sampling: sort tokens by probability,
 * keep the smallest set whose cumulative probability exceeds topP, and sample from
 * that set. This avoids both the dullness of greedy decoding and the incoherence
 * of sampling from the full distribution.
 *
 * @see https://arxiv.org/abs/1904.09751 — Holtzman et al. (2019) "The Curious Case of Neural Text Degeneration"
 */
export function generateText(
  w: TransformerWeights,
  cfg: TransformerConfig,
  seedIds: number[],
  maxLen: number,
  idxToToken: string[],
  temperature = 0.8,
  topP = 0.9,
  seqLen?: number,
): string {
  const windowSize = seqLen ?? cfg.contextLen;
  const ids = [...seedIds];

  for (let step = 0; step < maxLen; step++) {
    const contextIds = ids.slice(-windowSize);
    const cache = forward(contextIds, w, cfg);
    const lastPos = contextIds.length - 1;

    const logits = new Float32Array(cfg.vocabSize);
    for (let v = 0; v < cfg.vocabSize; v++) {
      logits[v] = cache.logits[lastPos * cfg.vocabSize + v];
    }

    for (let v = 0; v < cfg.vocabSize; v++) {
      logits[v] /= temperature;
    }

    let max = -Infinity;
    for (let v = 0; v < cfg.vocabSize; v++) {
      if (logits[v] > max)
        max = logits[v];
    }
    const probs = new Float32Array(cfg.vocabSize);
    let sum = 0;
    for (let v = 0; v < cfg.vocabSize; v++) {
      probs[v] = Math.exp(logits[v] - max);
      sum += probs[v];
    }
    for (let v = 0; v < cfg.vocabSize; v++) {
      probs[v] /= sum;
    }

    const sorted = Array.from(probs)
      .map((p, i) => ({ prob: p, idx: i }))
      .sort((a, b) => b.prob - a.prob);

    let cumProb = 0;
    const nucleus: { prob: number; idx: number }[] = [];
    for (const entry of sorted) {
      nucleus.push(entry);
      cumProb += entry.prob;
      if (cumProb >= topP)
        break;
    }

    let nucleusSum = 0;
    for (const entry of nucleus) nucleusSum += entry.prob;

    const r = rand() * nucleusSum;
    let cumulative = 0;
    let sampled = nucleus[0].idx;
    for (const entry of nucleus) {
      cumulative += entry.prob;
      if (r < cumulative) {
        sampled = entry.idx;
        break;
      }
    }

    ids.push(sampled);
  }

  return ids.map(id => idxToToken[id]).join("");
}
