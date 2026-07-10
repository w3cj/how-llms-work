/** Wraps children with `ChatContext`, making chat state available to all descendants. */
import type { Child } from "hono/jsx";
import type { ChatState } from "./chat-context.js";

import { ChatContext } from "./chat-context.js";

export function ChatProvider({ children, value }: { children: Child; value: ChatState }) {
  return <ChatContext value={value}>{children}</ChatContext>;
}
