/**
 * Client-side SSE (Server-Sent Events) reader — the browser half of the streaming pipeline.
 *
 * Every feature in this app streams results from the server via SSE. This module handles
 * the client side: sends a POST request, reads the streaming response chunk by chunk,
 * parses SSE events, and calls `onEvent` for each one.
 *
 * Two parsing modes:
 * - "json" (default) — each `data:` line is a standalone JSON object (used by most routes)
 * - "multiline" — events can span multiple `data:` lines (used by LLM chat for raw text streaming)
 *
 * Flow: `readSSE(options)` → POST to endpoint → stream chunks → parse events → `onEvent(parsed)`
 *
 * @see {@link file://src/server/lib/sse.ts} for the server-side emitter
 */
import { parseError } from "./parse-error.js";

export type SSEOptions<TEvent = Record<string, unknown>> = {
  endpoint: string;
  body: unknown;
  onEvent: (parsed: TEvent) => void;
  onOpen?: () => void;
  mode?: "json" | "multiline";
};

export type SSEResult = { ok: true } | { ok: false; error: string };

/**
 * Sends a POST request and reads the SSE response stream, invoking `onEvent` for each parsed event.
 * Returns `{ ok: true }` on success, or `{ ok: false, error: string }` if the request fails.
 */
export async function readSSE<TEvent = Record<string, unknown>>(options: SSEOptions<TEvent>): Promise<SSEResult> {
  const { endpoint, body, onEvent, onOpen, mode = "json" } = options;

  const response = await fetch(endpoint, {
    body: JSON.stringify(body),
    headers: { "Content-Type": "application/json" },
    method: "POST",
  });

  if (!response.ok) {
    const error = await parseError(response);
    return { ok: false, error };
  }

  onOpen?.();

  const reader = response.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done)
      break;

    buffer += decoder.decode(value, { stream: true });

    if (mode === "multiline") {
      const messages = buffer.split("\n\n");
      buffer = messages.pop()!;

      for (const msg of messages) {
        const lines = msg.split("\n");
        let event = "";
        const dataLines: string[] = [];

        for (const line of lines) {
          if (line.startsWith("event: ")) {
            event = line.slice(7).trim();
          }
          else if (line.startsWith("data: ")) {
            dataLines.push(line.slice(6));
          }
          else if (line === "data:") {
            dataLines.push("");
          }
        }

        if (dataLines.length === 0)
          continue;
        const data = dataLines.join("\n");
        if (data === "")
          continue;

        onEvent({ event, data } as TEvent);
      }
    }
    else {
      const lines = buffer.split("\n");
      buffer = lines.pop()!;

      for (const line of lines) {
        if (line.startsWith("event:"))
          continue;
        if (line.startsWith("data: ")) {
          const parsed = JSON.parse(line.slice(6)) as TEvent;
          onEvent(parsed);
        }
      }
    }
  }

  return { ok: true };
}
