/**
 * Simple pattern-matching chatbot — the "simplest possible thing inside the black box."
 *
 * Demonstrates that the streaming plumbing (SSE) works identically whether the
 * "intelligence" is a trillion-parameter AI model or twenty lines of if-statements.
 * The words stream in word by word, creating the same "typing effect" as ChatGPT.
 *
 * Based on ELIZA, created in 1966 by MIT professor Joseph Weizenbaum. It used pure
 * string manipulation — "I feel X" becomes "Why do you feel X?" — yet people
 * formed emotional attachments to it. Weizenbaum's own secretary asked him to
 * leave the room so she could talk to ELIZA privately.
 *
 * @see https://dl.acm.org/doi/10.1145/365153.365168 — Weizenbaum (1966) "ELIZA"
 *
 * SSE event flow:
 * 1. "start" — signals the UI to show a loading state
 * 2. "word" × N — one event per word, streamed with 200ms delays
 * 3. "done" — signals completion
 */
import { zValidator } from "@hono/zod-validator";
import { Hono } from "hono";
import { streamSSE } from "hono/streaming";
import ChatRequest from "../schemas/chat-request.js";
import { createEmitter } from "../server/lib/sse.js";

const greetings = [
  "hello",
  "hi",
  "hey",
  "morning",
  "afternoon",
  "evening",
];

/** Returns a random fallback response when no pattern matches — ELIZA's "tell me more." */
function getRandomContinuation(): string {
  const continuations = [
    "Please go on.",
    "Can you tell me more about that?",
    "Tell me more.",
    "How does that make you feel?",
  ];
  return continuations[Math.floor(Math.random() * continuations.length)];
}

/**
 * Pattern-matches the user's message and returns a canned response.
 *
 * This is the entire "AI" — a chain of string checks. It handles:
 * - Greetings ("hello", "hi", "hey")
 * - "I feel X" → "Why do you feel X?" (ELIZA's signature move)
 * - "my X" → "Tell me more about your X"
 * - "worried" → a follow-up question
 * - Everything else → random continuation
 *
 * @example
 * getSimpleChatResponse("I feel anxious")  // => "Why do you feel anxious?"
 * getSimpleChatResponse("hello there")     // => "Hello! How can I help you today?"
 * getSimpleChatResponse("my dog is sick")  // => "Tell me more about your dog."
 */
function getSimpleChatResponse(message: string): string {
  const messageLower = message.toLowerCase();
  const words = messageLower
    .split(" ")
    .map(word => word.replaceAll(/[!?.;,:"']/g, ""))
    .filter(word => word.trim() !== "");

  if (greetings.some(greet => words.includes(greet))) {
    return "Hello! How can I help you today?";
  }

  const iFeel = "i feel";
  if (messageLower.startsWith(iFeel)) {
    const feeling = messageLower.slice(iFeel.length + 1).replace(" i ", " you ");
    return `Why do you feel ${feeling}?`;
  }

  const subjectIndex = words.indexOf("my") + 1;
  if (subjectIndex > 0 && subjectIndex < words.length) {
    const subject = words[subjectIndex];
    return `Tell me more about your ${subject}.`;
  }

  if (messageLower.includes("worried")) {
    return "How long have you been worried about this?";
  }

  return getRandomContinuation();
}

export default new Hono()
  .post(
    "/simple-chat",
    zValidator("json", ChatRequest),
    (c) => {
      const { message } = c.req.valid("json");
      const response = getSimpleChatResponse(message);
      const words = response.split(" ");

      return streamSSE(c, async (stream) => {
        const { emit } = createEmitter(stream);
        await emit({}, "start", 1000);
        for (const word of words) {
          await emit({ word }, "word", 200);
        }
        await emit({}, "done");
      });
    },
  );
