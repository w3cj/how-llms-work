/**
 * Train embeddings route — trains a Word2Vec Skip-gram model from scratch and streams results.
 *
 * Takes query words and optional hyperparameters. Trains on a curated corpus,
 * streaming epoch losses as training progresses. On completion, emits the learned
 * embeddings, nearest neighbors, pairwise similarity, and vector analogies.
 *
 * SSE event flow:
 * 1. "init" — corpus and model stats (vocab size, dimensions, training pairs)
 * 2. "epoch" × ~50 — loss values during training
 * 3. "done" — embeddings, neighbors, similarities, analogies for the query words
 */
import type { TrainEmbedResult } from "./train.js";
import { zValidator } from "@hono/zod-validator";
import { Hono } from "hono";
import { streamSSE } from "hono/streaming";
import TrainEmbedRequest from "../../schemas/train-embed-request.js";
import { createEmitter } from "../../server/lib/sse.js";
import { trainSkipGram } from "./train.js";

function isTrainEmbedResult(value: unknown): value is TrainEmbedResult {
  return typeof value === "object" && value !== null && "embeddings" in value;
}

function isInitResult(value: unknown): value is { vocabSize: number } {
  return typeof value === "object" && value !== null && "vocabSize" in value;
}

export default new Hono().post(
  "/train-embed",
  zValidator("json", TrainEmbedRequest),
  (c) => {
    const body = c.req.valid("json");
    const trainer = trainSkipGram(body);

    return streamSSE(c, async (stream) => {
      const { emit } = createEmitter(stream);
      for await (const result of trainer) {
        if (isTrainEmbedResult(result)) {
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
