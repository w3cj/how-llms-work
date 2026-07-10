/**
 * Server-Sent Events (SSE) emitter for streaming JSON data to the browser.
 *
 * SSE is the protocol that makes ChatGPT's "typing effect" work. Instead of
 * waiting for the entire response, the server pushes data piece by piece over
 * a long-lived HTTP connection. Each piece arrives the moment it's ready.
 *
 * Every route in this app uses SSE to stream results — tokens appearing one
 * at a time, training epochs ticking by, merge steps animating in. The flow is:
 *
 * 1. Route handler calls `streamSSE(c, async (stream) => { ... })`
 * 2. Inside that callback, call `createEmitter(stream)` to get `emit` / `emitError`
 * 3. Call `emit(data, eventName, delay?)` for each piece of data
 * 4. The client's `readSSE()` parses each event and triggers a UI update
 *
 * @see {@link file://src/client/lib/sse.ts} for the client-side reader
 */
import type { streamSSE } from "hono/streaming";

type SSEStream = Parameters<Parameters<typeof streamSSE>[1]>[0];

const sleep = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

/**
 * Creates an emitter that writes JSON events to an SSE stream.
 *
 * @example
 * return streamSSE(c, async (stream) => {
 *   const { emit, emitError } = createEmitter(stream);
 *   await emit({ count: 7 }, "start", 1000);  // send "start" event, wait 1s
 *   await emit({ token: "hello" }, "token", 50); // send "token" event, wait 50ms
 *   await emit({ done: true }, "done");           // send "done" event, no delay
 * });
 */
export function createEmitter(stream: SSEStream) {
  return {
    /**
     * Serializes `data` as JSON and writes it as an SSE event.
     * Optionally waits `delay` ms afterward — used to pace streaming animations
     * so users can watch tokens/steps appear one at a time.
     */
    async emit<T>(data: T, event: string, delay?: number) {
      await stream.writeSSE({ data: JSON.stringify(data), event });
      if (delay)
        await sleep(delay);
    },
    /** Writes an error event. The client's `readSSE()` surfaces this as `result.error`. */
    async emitError(err: unknown) {
      await stream.writeSSE({
        data: JSON.stringify({ error: err instanceof Error ? err.message : String(err) }),
        event: "error",
      });
    },
  };
}
