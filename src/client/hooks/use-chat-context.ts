/** Convenience hook to access `ChatContext`. Throws if used outside a `ChatProvider`. */
import { useContext } from "hono/jsx";

import { ChatContext } from "../context/chat-context.js";

export function useChatContext() {
  // eslint-disable-next-line react/no-use-context
  const ctx = useContext(ChatContext);
  if (!ctx) {
    throw new Error("useChatContext must be used within a ChatProvider");
  }
  return ctx;
}
