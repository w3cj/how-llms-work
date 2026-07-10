/**
 * Serialization for neural network weights.
 *
 * After training, the network's weights (the learned parameters) are saved to disk
 * as JSON. When a model is described as having "175 billion parameters" (GPT-3),
 * those parameters are exactly these kinds of weight values — just far more of them.
 *
 * Two weight formats:
 * - `SingleLayerWeights` — 2 weights + 1 bias (the perceptron)
 * - `MultiLayerWeights` — weight matrices + bias vectors for each layer
 */
import { writeFile } from "node:fs/promises";

export type SingleLayerWeights = {
  type: "single-layer";
  w1: number;
  w2: number;
  bias: number;
};

export type MultiLayerWeights = {
  type: "multi-layer";
  w1: number[][];
  b1: number[];
  w2: number[];
  b2: number;
};

export type SavedNetwork = SingleLayerWeights | MultiLayerWeights;

/** Writes the trained weights to a JSON file for later reuse. */
export async function saveNetwork(network: SavedNetwork, path: string): Promise<void> {
  await writeFile(path, `${JSON.stringify(network, null, 2)}\n`);
}
