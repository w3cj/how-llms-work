/**
 * Transformer training route — trains a tiny GPT from scratch and streams results.
 *
 * Takes an epoch count, trains on the shared BPE-tokenized corpus, and streams
 * epoch losses and text generation samples as training progresses.
 *
 * SSE event flow:
 * 1. "init" — model architecture and corpus stats
 * 2. "epoch" × ~50 — loss values, with periodic generated text samples
 * 3. "done" — final loss and collection of generated samples across training
 */
import type { TransformerTrainResult } from "./train.js";
import { zValidator } from "@hono/zod-validator";
import { Hono } from "hono";
import { streamSSE } from "hono/streaming";
import TrainTransformerRequest from "../../schemas/train-transformer-request.js";
import { createEmitter } from "../../server/lib/sse.js";
import { trainTransformer } from "./train.js";

function isTrainResult(value: unknown): value is TransformerTrainResult {
  return typeof value === "object" && value !== null && "architecture" in value;
}

function isInitResult(value: unknown): value is { vocabSize: number } {
  return typeof value === "object" && value !== null && "vocabSize" in value;
}

export default new Hono().post(
  "/train-transformer",
  zValidator("json", TrainTransformerRequest),
  (c) => {
    const body = c.req.valid("json");
    const trainer = trainTransformer(body);

    return streamSSE(c, async (stream) => {
      const { emit } = createEmitter(stream);
      for await (const result of trainer) {
        if (isTrainResult(result)) {
          await emit(result, "done");
        }
        else if (isInitResult(result)) {
          await emit(result, "init");
        }
        else {
          await emit(result, "epoch", 20);
        }
      }
    });
  },
);
