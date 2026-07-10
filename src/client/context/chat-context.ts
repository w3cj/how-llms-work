/**
 * Chat context — shared state that flows from each page's hook down to all UI components.
 *
 * Each page (tokenize, embed, neural net, etc.) creates a `ChatState` via its hook,
 * which the `App` component provides via `ChatProvider`. Child components like `Header`,
 * `ChatInput`, and `MessageList` consume it via `useChatContext()`.
 */
import type { Message } from "../../shared/types/message.js";

import { createContext } from "hono/jsx";

export type ChatState = {
  input: string;
  loading: boolean;
  messages: Message[];
  sendMessage: () => void;
  setInput: (value: string) => void;
  tagline: string;
  title: string;
};

export const ChatContext = createContext<ChatState | null>(null);
