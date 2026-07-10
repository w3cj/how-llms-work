/**
 * BPE tokenizer route — trains Byte Pair Encoding on user input and streams every merge step.
 *
 * This is how real tokenizers like tiktoken and sentencepiece build their vocabularies:
 * start with individual characters, find the most frequent pair, merge it into a new token,
 * repeat. The result is a compact vocabulary that balances between whole words and characters.
 *
 * @see https://arxiv.org/abs/1508.07909 — Sennrich, Haddow & Birch (2016) "Neural Machine Translation of Rare Words with Subword Units"
 *
 * SSE event flow:
 * 1. "init" — corpus stats (character count, word count)
 * 2. "merge" × N — one event per BPE merge (pair, frequency, new token, vocab size)
 * 3. "result" — final tokenization of the input with compression ratio
 */
import { zValidator } from "@hono/zod-validator";
import { Hono } from "hono";
import { streamSSE } from "hono/streaming";
import ChatRequest from "../schemas/chat-request.js";
import { applyMerges, countWords, trainBpe } from "../server/lib/bpe.js";
import { createEmitter } from "../server/lib/sse.js";

export default new Hono().post(
  "/bpe-tokenize",
  zValidator("json", ChatRequest),
  (c) => {
    const { message } = c.req.valid("json");
    const wordFreqs = countWords(message);
    const characters = [...message];

    const { merges } = trainBpe(wordFreqs);

    return streamSSE(c, async (stream) => {
      const { emit } = createEmitter(stream);

      await emit({
        corpus: message,
        characters: characters.length > 200 ? characters.slice(0, 200) : characters,
        charCount: characters.length,
        wordCount: wordFreqs.size,
      }, "init", 800);

      const vocab = new Set(characters);
      let totalTokenCount = characters.length;

      for (let i = 0; i < merges.length; i++) {
        const { pair, merged, frequency } = merges[i];
        vocab.add(merged);
        totalTokenCount -= frequency;

        await emit({
          step: i + 1,
          pair,
          frequency,
          newToken: merged,
          vocabSize: vocab.size,
          tokenCount: totalTokenCount,
        }, "merge");
      }

      const inputTokens = applyMerges(message, merges);

      await emit({
        inputTokens,
        tokenCount: inputTokens.length,
        originalCharCount: message.length,
        compressionRatio: message.length > 0
          ? `${(message.length / inputTokens.length).toFixed(1)}x`
          : "N/A",
      }, "result");
    });
  },
);
