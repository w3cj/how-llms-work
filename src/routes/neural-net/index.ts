/**
 * Neural network route — trains a tiny network on XOR and streams the results.
 *
 * Takes a mode ("single-layer" or "multi-layer") and epoch count. Delegates to
 * the training generators in `train.ts`, streaming each epoch's loss as it goes.
 * On completion, persists the trained weights to disk and emits the final result
 * with predictions and a pass/fail verdict.
 *
 * SSE event flow:
 * 1. "epoch" × ~50 — loss values during training (throttled to ~50 updates)
 * 2. "done" — architecture, predictions for all 4 XOR inputs, and verdict
 */
import type { TrainResult } from "./train.js";
import { zValidator } from "@hono/zod-validator";
import { Hono } from "hono";
import { streamSSE } from "hono/streaming";
import NeuralNetRequest from "../../schemas/neural-net-request.js";
import { createEmitter } from "../../server/lib/sse.js";
import { saveNetwork } from "./serialize.js";
import { trainMultiLayer, trainSingleLayer } from "./train.js";

/** Type guard to distinguish the final TrainResult from intermediate EpochResults. */
function isTrainResult(value: unknown): value is TrainResult {
  return typeof value === "object" && value !== null && "predictions" in value;
}

export default new Hono().post(
  "/neural-net",
  zValidator("json", NeuralNetRequest),
  (c) => {
    const { mode, epochs } = c.req.valid("json");
    const trainer = mode === "single-layer"
      ? trainSingleLayer(epochs)
      : trainMultiLayer(epochs);

    return streamSSE(c, async (stream) => {
      const { emit } = createEmitter(stream);
      for await (const result of trainer) {
        if (isTrainResult(result)) {
          await saveNetwork(result.weights, `.data/${mode}-weights.json`);
          const { weights: _, ...payload } = result;
          await emit(payload, "done");
        }
        else {
          await emit(result, "epoch", 20);
        }
      }
    });
  },
);
