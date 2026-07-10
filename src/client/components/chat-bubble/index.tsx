/** Individual chat message bubble — renders user messages as text and assistant messages as rich JSX content (components, visualizations, etc.). Shows a typing indicator while loading. */
import type { Message } from "../../../shared/types/message.js";

import clsx from "clsx";
import { BouncingDots } from "../bouncing-dots/index.js";
import styles from "./styles.module.css";

export function ChatBubble({ message }: { message: Message }) {
  const showTyping = message.role === "assistant" && message.content === "";
  const isUser = message.role === "user";

  return (
    <div class={clsx(styles.row, isUser ? styles.rowUser : styles.rowAssistant)}>
      {isUser
        ? (
            <article class={clsx("card", styles.bubbleUser)}>
              {message.content}
            </article>
          )
        : (
            <div class={styles.assistant}>
              {showTyping ? <BouncingDots /> : message.content}
            </div>
          )}
    </div>
  );
}
