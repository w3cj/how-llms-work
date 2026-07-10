/**
 * Shared message type used by both client and server.
 * `content` is a JSX `Child` — it can be a string, a number, or a full React component tree.
 * This is what allows each feature to render rich visualizations (token grids, attention maps,
 * training progress) inside chat bubbles rather than just plain text.
 */
import type { Child } from "hono/jsx";

export type Message = {
  content: Child;
  id: string;
  role: "assistant" | "user";
};
