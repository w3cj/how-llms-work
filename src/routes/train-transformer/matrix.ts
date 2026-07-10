/**
 * Matrix operations on flat Float32Arrays — the linear algebra beneath a transformer.
 *
 * Every tensor in the model is stored as a flat Float32Array with explicit dimensions.
 * This is the same representation PyTorch and NumPy use under the hood — just without
 * the abstraction layer. When you see `matmul(A, B, M, K, N)`, that's the exact same
 * operation as `torch.mm(A, B)` or `np.dot(A, B)`.
 *
 * Matrix layout: row-major. Element (i, j) of an (M, N) matrix is at index `i * N + j`.
 */

function mulberry32(seed: number): () => number {
  return () => {
    seed |= 0;
    seed = (seed + 0x6D2B79F5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Current PRNG instance — deterministic (seeded), so training is reproducible. */
// eslint-disable-next-line import/no-mutable-exports
export let rand = mulberry32(42);

/** Reset the PRNG to a given seed, ensuring reproducible weight initialization and sampling. */
export function resetRand(seed = 42) {
  rand = mulberry32(seed);
}

/** (M,K) × (K,N) → (M,N). The fundamental operation of neural networks. */
export function matmul(a: Float32Array, b: Float32Array, M: number, K: number, N: number): Float32Array {
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

/** (M,K) × (N,K)ᵀ → (M,N). Equivalent to A @ B.T — avoids transposing B first. */
export function matmulTransB(a: Float32Array, b: Float32Array, M: number, K: number, N: number): Float32Array {
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

/** (K,M)ᵀ × (K,N) → (M,N). Equivalent to A.T @ B — avoids transposing A first. */
export function matmulTransA(a: Float32Array, b: Float32Array, K: number, M: number, N: number): Float32Array {
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

/** Add bias vector to each row: out[i][j] += bias[j]. */
export function addBias(out: Float32Array, bias: Float32Array, rows: number, cols: number) {
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      out[i * cols + j] += bias[j];
    }
  }
}

/** Element-wise: target[i] += source[i]. */
export function addInPlace(target: Float32Array, source: Float32Array) {
  for (let i = 0; i < target.length; i++) {
    target[i] += source[i];
  }
}

/** Zero-filled array of given length. */
export function zeros(len: number): Float32Array {
  return new Float32Array(len);
}

/** Ones-filled array of given length (used for layer norm gamma init). */
export function ones(len: number): Float32Array {
  const out = new Float32Array(len);
  out.fill(1);
  return out;
}

/** Xavier/Glorot uniform initialization: uniform in [-limit, limit] where limit = sqrt(6 / (fanIn + fanOut)). */
export function xavierInit(len: number, fanIn: number, fanOut: number): Float32Array {
  const limit = Math.sqrt(6 / (fanIn + fanOut));
  const out = new Float32Array(len);
  for (let i = 0; i < len; i++) {
    out[i] = (rand() * 2 - 1) * limit;
  }
  return out;
}

/** Sum columns: for (M,N) matrix, returns (N,) vector where out[j] = sum_i(mat[i][j]). */
export function sumCols(mat: Float32Array, rows: number, cols: number): Float32Array {
  const out = new Float32Array(cols);
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      out[j] += mat[i * cols + j];
    }
  }
  return out;
}
