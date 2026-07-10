/** Hook for the ELIZA-style pattern-matching chatbot. Streams words one at a time via SSE. */
import { useSSEChat } from "./use-sse-chat.js";

export function useSimpleChat() {
  return useSSEChat<{ words: string[] }, { word?: string }>({
    endpoint: "/simple-chat",
    title: "Simple Chat Bot",
    tagline: "a simple pattern matching chat bot",
    initState: () => ({ words: [] }),
    onEvent: (parsed, state) => {
      if (parsed.word) {
        state.words.push(parsed.word);
        return state.words.join(" ");
      }
    },
  });
}
