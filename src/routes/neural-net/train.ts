/**
 * Neural network training for the XOR problem — single-layer vs. multi-layer.
 *
 * This demonstrates the foundational story of neural networks. In 1958, Frank
 * Rosenblatt built the Perceptron — a single-layer network that could learn simple
 * classifications. In 1969, Minsky and Papert proved it couldn't solve XOR (exclusive
 * or), triggering the first "AI winter."
 *
 * The breakthrough came in 1986 when Rumelhart, Hinton, and Williams described
 * backpropagation — a method for training multi-layer networks. Make a prediction,
 * measure error, propagate it backward through every layer, adjusting each weight.
 * Every neural network today (including ChatGPT) is trained with some variant of this.
 *
 * @see https://www.nature.com/articles/323533a0 — Rumelhart et al. (1986) "Learning representations by back-propagating errors"
 *
 * This file provides two async generators:
 * 1. `trainSingleLayer` — a perceptron that FAILS on XOR (proves Minsky/Papert right)
 * 2. `trainMultiLayer` — a 2→4→1 network with backprop that SUCCEEDS
 *
 * Both yield `EpochResult` during training and a final `TrainResult` with predictions.
 */
import type { MultiLayerWeights, SavedNetwork, SingleLayerWeights } from "./serialize.js";

const XOR_INPUTS = [[0, 0], [0, 1], [1, 0], [1, 1]];
const XOR_TARGETS = [0, 1, 1, 0];

/**
 * Sigmoid activation function — squashes any number into the range (0, 1).
 * Used as the neuron's "firing" function: values near 1 = active, near 0 = inactive.
 *
 * @example
 * sigmoid(0)   // => 0.5 (neutral)
 * sigmoid(10)  // => 0.9999 (strongly active)
 * sigmoid(-10) // => 0.0001 (strongly inactive)
 */
function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

/**
 * Derivative of sigmoid, used during backpropagation.
 * Takes the sigmoid OUTPUT (not the raw input) — this shortcut works because
 * the derivative of sigmoid(x) equals sigmoid(x) * (1 - sigmoid(x)).
 */
function sigmoidDeriv(s: number): number {
  return s * (1 - s);
}

/** Random weight in [-1, 1]. Networks are initialized with random weights before training. */
function randWeight(): number {
  return Math.random() * 2 - 1;
}

/** Yields to the event loop so SSE events can flush during long training loops. */
const tick = () => new Promise(resolve => setTimeout(resolve, 0));

export type EpochResult = { epoch: number; loss: number };

export type Prediction = { input: number[]; expected: number; actual: number };

export type TrainResult = {
  architecture: string;
  predictions: Prediction[];
  verdict: string;
  weights: SavedNetwork;
};

/**
 * Trains a single-layer perceptron on XOR. This will FAIL.
 *
 * A single-layer perceptron can only learn linearly separable problems — it draws
 * a straight line to divide inputs into categories. XOR isn't linearly separable
 * (you can't draw one line to separate [0,1],[1,0] from [0,0],[1,1]), so the loss
 * gets stuck around 0.25 and predictions hover near 0.5 (random guessing).
 *
 * Architecture: 2 inputs → 1 output (sigmoid), gradient descent with MSE loss.
 *
 * Yields `EpochResult` at ~50 intervals during training, then a final `TrainResult`.
 */
export async function* trainSingleLayer(epochs: number): AsyncGenerator<EpochResult | TrainResult> {
  let w1 = randWeight();
  let w2 = randWeight();
  let bias = 0;
  const lr = 1.0;

  const step = Math.max(1, Math.floor(epochs / 50));

  for (let epoch = 0; epoch <= epochs; epoch++) {
    let totalLoss = 0;

    for (let i = 0; i < 4; i++) {
      const [x1, x2] = XOR_INPUTS[i];
      const target = XOR_TARGETS[i];

      const output = sigmoid(x1 * w1 + x2 * w2 + bias);
      const error = output - target;
      totalLoss += error * error;

      const delta = error * sigmoidDeriv(output);
      w1 -= lr * delta * x1;
      w2 -= lr * delta * x2;
      bias -= lr * delta;
    }

    const loss = totalLoss / 4;

    if (epoch % step === 0 || epoch === epochs) {
      yield { epoch, loss: Math.round(loss * 1000000) / 1000000 };
      await tick();
    }
  }

  const weights: SingleLayerWeights = { type: "single-layer", w1, w2, bias };

  const predictions: Prediction[] = XOR_INPUTS.map((input, i) => ({
    input,
    expected: XOR_TARGETS[i],
    actual: Math.round(sigmoid(input[0] * w1 + input[1] * w2 + bias) * 100) / 100,
  }));

  const success = predictions.every(p => Math.abs(p.actual - p.expected) < 0.1);

  yield {
    architecture: "Single-Layer Perceptron (2 → 1)",
    predictions,
    verdict: success
      ? "SUCCESS — network learned XOR"
      : "FAILED — loss stuck, predictions are random guesses",
    weights,
  };
}

/**
 * Trains a multi-layer network on XOR using backpropagation. This SUCCEEDS.
 *
 * By adding a hidden layer of 4 neurons between input and output, the network
 * can learn non-linear decision boundaries. The hidden layer transforms the inputs
 * into a space where XOR becomes linearly separable, then the output layer draws
 * the line.
 *
 * Architecture: 2 inputs → 4 hidden (sigmoid) → 1 output (sigmoid).
 *
 * Training loop per epoch:
 * 1. Forward pass: compute hidden activations, then output
 * 2. Backward pass: compute output error, propagate to hidden layer
 * 3. Update: adjust all weights by learning rate × gradient
 *
 * Yields `EpochResult` at ~50 intervals, then a final `TrainResult` with predictions.
 */
export async function* trainMultiLayer(epochs: number): AsyncGenerator<EpochResult | TrainResult> {
  const HIDDEN = 4;
  const lr = 1.0;

  // Weights: input (2) -> hidden (4)
  const w1: number[][] = Array.from({ length: 2 }, () =>
    Array.from({ length: HIDDEN }, () => randWeight()));
  const b1: number[] = Array.from<number>({ length: HIDDEN }).fill(0);

  // Weights: hidden (4) -> output (1)
  const w2: number[] = Array.from({ length: HIDDEN }, () => randWeight());
  let b2 = 0;

  const step = Math.max(1, Math.floor(epochs / 50));

  for (let epoch = 0; epoch <= epochs; epoch++) {
    let totalLoss = 0;

    for (let i = 0; i < 4; i++) {
      const [x1, x2] = XOR_INPUTS[i];
      const target = XOR_TARGETS[i];

      // Forward: input -> hidden
      const hidden: number[] = [];
      for (let j = 0; j < HIDDEN; j++) {
        hidden[j] = sigmoid(x1 * w1[0][j] + x2 * w1[1][j] + b1[j]);
      }

      // Forward: hidden -> output
      let sum = b2;
      for (let j = 0; j < HIDDEN; j++) {
        sum += hidden[j] * w2[j];
      }
      const output = sigmoid(sum);

      const error = output - target;
      totalLoss += error * error;

      // Backward: output layer
      const outputDelta = error * sigmoidDeriv(output);

      // Backward: hidden layer
      const hiddenDelta: number[] = [];
      for (let j = 0; j < HIDDEN; j++) {
        hiddenDelta[j] = outputDelta * w2[j] * sigmoidDeriv(hidden[j]);
      }

      // Update: hidden -> output weights
      for (let j = 0; j < HIDDEN; j++) {
        w2[j] -= lr * outputDelta * hidden[j];
      }
      b2 -= lr * outputDelta;

      // Update: input -> hidden weights
      const inputs = [x1, x2];
      for (let j = 0; j < HIDDEN; j++) {
        for (let k = 0; k < 2; k++) {
          w1[k][j] -= lr * hiddenDelta[j] * inputs[k];
        }
        b1[j] -= lr * hiddenDelta[j];
      }
    }

    const loss = totalLoss / 4;

    if (epoch % step === 0 || epoch === epochs) {
      yield { epoch, loss: Math.round(loss * 1000000) / 1000000 };
      await tick();
    }
  }

  const weights: MultiLayerWeights = { type: "multi-layer", w1, b1, w2, b2 };

  // Final predictions
  const predictions: Prediction[] = XOR_INPUTS.map((input, i) => {
    const [x1, x2] = input;
    const hidden: number[] = [];
    for (let j = 0; j < HIDDEN; j++) {
      hidden[j] = sigmoid(x1 * w1[0][j] + x2 * w1[1][j] + b1[j]);
    }
    let sum = b2;
    for (let j = 0; j < HIDDEN; j++) {
      sum += hidden[j] * w2[j];
    }
    return {
      input,
      expected: XOR_TARGETS[i],
      actual: Math.round(sigmoid(sum) * 100) / 100,
    };
  });

  const success = predictions.every(p => Math.abs(p.actual - p.expected) < 0.1);

  yield {
    architecture: "Multi-Layer Network (2 → 4 → 1)",
    predictions,
    verdict: success
      ? "SUCCESS — network learned XOR via backpropagation"
      : "FAILED — network did not converge, try more epochs",
    weights,
  };
}
