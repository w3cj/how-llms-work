/** Chat input textarea with send button. Enter sends, Shift+Enter adds a newline. */
import { useEffect, useRef } from "hono/jsx";

import { useChatContext } from "../../hooks/use-chat-context.js";
import styles from "./styles.module.css";

type ChatInputProps = {
  onSend: () => void;
};

export function ChatInput({ onSend }: ChatInputProps) {
  const { input, loading, setInput } = useChatContext();
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  return (
    <div class={styles.wrapper}>
      <textarea
        ref={inputRef}
        rows={4}
        value={input}
        onInput={event => setInput((event.target as HTMLTextAreaElement).value)}
        onKeyDown={(event) => {
          if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault();
            onSend();
          }
        }}
        placeholder="Type your message..."
      />
      <button
        onClick={onSend}
        disabled={loading}
        data-variant="primary"
      >
        Send
      </button>
    </div>
  );
}
