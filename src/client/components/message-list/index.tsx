/** Scrollable list of chat bubbles with auto-scroll and empty state. */
import type { RefObject } from "hono/jsx";

import { useChatContext } from "../../hooks/use-chat-context.js";
import { ChatBubble } from "../chat-bubble/index.js";
import { EmptyState } from "../empty-state/index.js";
import styles from "./styles.module.css";

type MessageListProps = {
  onScroll: () => void;
  scrollRef: RefObject<HTMLDivElement>;
};

export function MessageList({ onScroll, scrollRef }: MessageListProps) {
  const { messages } = useChatContext();

  return (
    <div
      ref={scrollRef}
      onScroll={onScroll}
      class={styles.list}
    >
      {messages.length === 0 && <EmptyState />}
      {messages.map(message => (
        <ChatBubble key={message.id} message={message} />
      ))}
    </div>
  );
}
