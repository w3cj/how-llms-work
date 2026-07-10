// src/routes/train-transformer/train-worker.ts
import { parentPort, workerData } from "node:worker_threads";

// src/routes/train-transformer/matrix.ts
function mulberry32(seed) {
  return () => {
    seed |= 0;
    seed = seed + 1831565813 | 0;
    let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}
const rand = mulberry32(42);
function matmul(a, b, M, K, N) {
  const out = new Float32Array(M * N);
  for (let i = 0; i < M; i++) {
    for (let k = 0; k < K; k++) {
      const aik = a[i * K + k];
      for (let j = 0; j < N; j++) {
        out[i * N + j] += aik * b[k * N + j];
      }
    }
  }
  return out;
}
function matmulTransB(a, b, M, K, N) {
  const out = new Float32Array(M * N);
  for (let i = 0; i < M; i++) {
    for (let j = 0; j < N; j++) {
      let sum = 0;
      for (let k = 0; k < K; k++) {
        sum += a[i * K + k] * b[j * K + k];
      }
      out[i * N + j] = sum;
    }
  }
  return out;
}
function matmulTransA(a, b, K, M, N) {
  const out = new Float32Array(M * N);
  for (let k = 0; k < K; k++) {
    for (let i = 0; i < M; i++) {
      const aki = a[k * M + i];
      for (let j = 0; j < N; j++) {
        out[i * N + j] += aki * b[k * N + j];
      }
    }
  }
  return out;
}
function addBias(out, bias, rows, cols) {
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      out[i * cols + j] += bias[j];
    }
  }
}
function addInPlace(target, source) {
  for (let i = 0; i < target.length; i++) {
    target[i] += source[i];
  }
}
function sumCols(mat, rows, cols) {
  const out = new Float32Array(cols);
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      out[j] += mat[i * cols + j];
    }
  }
  return out;
}

// src/routes/train-transformer/transformer.ts
function layerNormForward(x, gamma, beta, seqLen2, D, mean, variance, xHat) {
  const out = new Float32Array(seqLen2 * D);
  const eps = 1e-5;
  for (let i = 0; i < seqLen2; i++) {
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
function layerNormBackward(dOut, xHat, gamma, mean, variance, x, seqLen2, D, dGamma, dBeta) {
  const dX = new Float32Array(seqLen2 * D);
  const eps = 1e-5;
  for (let i = 0; i < seqLen2; i++) {
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
  for (let i = 0; i < seqLen2; i++) {
    for (let d = 0; d < D; d++) {
      dGamma[d] += dOut[i * D + d] * xHat[i * D + d];
      dBeta[d] += dOut[i * D + d];
    }
  }
  return dX;
}
function softmaxRows(x, rows, cols) {
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
function blockForward(input, bw, cfg2) {
  const { embDim: D, numHeads: H, ffDim: F } = cfg2;
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
function blockBackward(dOutput, bc, bw, cfg2, bg) {
  const { embDim: D, numHeads: H, ffDim: F } = cfg2;
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
function forward(tokenIds, w2, cfg2) {
  const { vocabSize: V, embDim: D } = cfg2;
  const S = tokenIds.length;
  const x0 = new Float32Array(S * D);
  for (let i = 0; i < S; i++) {
    for (let d = 0; d < D; d++) {
      x0[i * D + d] = w2.tokEmb[tokenIds[i] * D + d] + w2.posEmb[i * D + d];
    }
  }
  const blockCaches = [];
  let x = x0;
  for (let i = 0; i < w2.blocks.length; i++) {
    const bc = blockForward(x, w2.blocks[i], cfg2);
    blockCaches.push(bc);
    x = bc.output;
  }
  const lnFMean = new Float32Array(S);
  const lnFVar = new Float32Array(S);
  const lnFHat = new Float32Array(S * D);
  const lnFOut = layerNormForward(x, w2.lnFGamma, w2.lnFBeta, S, D, lnFMean, lnFVar, lnFHat);
  const logits = matmul(lnFOut, w2.headW, S, D, V);
  addBias(logits, w2.headB, S, V);
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
function crossEntropyLoss(probs, targets, seqLen2, V) {
  let loss = 0;
  for (let i = 0; i < seqLen2; i++) {
    loss -= Math.log(probs[i * V + targets[i]] + 1e-10);
  }
  return loss / seqLen2;
}
function backward(cache, targets, w2, cfg2, grads2) {
  const { vocabSize: V, embDim: D } = cfg2;
  const S = cache.tokenIds.length;
  const dLogits = new Float32Array(S * V);
  for (let i = 0; i < S; i++) {
    for (let j = 0; j < V; j++) {
      dLogits[i * V + j] = cache.probs[i * V + j] / S;
    }
    dLogits[i * V + targets[i]] -= 1 / S;
  }
  addInPlace(grads2.headB, sumCols(dLogits, S, V));
  addInPlace(grads2.headW, matmulTransA(cache.lnFOut, dLogits, S, D, V));
  let dX = matmulTransB(dLogits, w2.headW, S, V, D);
  const lastBlockOutput = cache.blockCaches.length > 0 ? cache.blockCaches[cache.blockCaches.length - 1].output : cache.x0;
  dX = layerNormBackward(dX, cache.lnFHat, w2.lnFGamma, cache.lnFMean, cache.lnFVar, lastBlockOutput, S, D, grads2.lnFGamma, grads2.lnFBeta);
  for (let i = cache.blockCaches.length - 1; i >= 0; i--) {
    dX = blockBackward(dX, cache.blockCaches[i], w2.blocks[i], cfg2, grads2.blocks[i]);
  }
  for (let i = 0; i < S; i++) {
    const tokIdx = cache.tokenIds[i];
    for (let d = 0; d < D; d++) {
      grads2.tokEmb[tokIdx * D + d] += dX[i * D + d];
      grads2.posEmb[i * D + d] += dX[i * D + d];
    }
  }
}

// src/routes/train-transformer/weight-layout.ts
function createWeightViews(sab, layout2, numLayers) {
  const w2 = { blocks: [] };
  for (let i = 0; i < numLayers; i++) w2.blocks.push({});
  for (const { key, blockIdx, offset, length } of layout2) {
    const view = new Float32Array(sab, offset * 4, length);
    if (blockIdx === -1) {
      w2[key] = view;
    }
    else {
      w2.blocks[blockIdx][key] = view;
    }
  }
  return w2;
}
function createGradViews(sab, layout2, numLayers) {
  const g = { blocks: [] };
  for (let i = 0; i < numLayers; i++) g.blocks.push({});
  for (const { key, blockIdx, offset, length } of layout2) {
    const view = new Float32Array(sab, offset * 4, length);
    if (blockIdx === -1) {
      g[key] = view;
    }
    else {
      g.blocks[blockIdx][key] = view;
    }
  }
  return g;
}

// src/routes/train-transformer/train-worker.ts
const {
  weightSab,
  gradSab,
  layout,
  cfg,
  sequences,
  seqLen,
} = workerData;
const w = createWeightViews(weightSab, layout, cfg.numLayers);
const grads = createGradViews(gradSab, layout, cfg.numLayers);
parentPort.on("message", (msg) => {
  if (msg.type === "compute") {
    new Float32Array(gradSab).fill(0);
    let totalLoss = 0;
    for (const seq of sequences) {
      const cache = forward(seq.input, w, cfg);
      totalLoss += crossEntropyLoss(cache.probs, seq.target, seqLen, cfg.vocabSize);
      backward(cache, seq.target, w, cfg, grads);
    }
    parentPort.postMessage({ type: "done", loss: totalLoss });
  }
});
parentPort.postMessage({ type: "ready" });
