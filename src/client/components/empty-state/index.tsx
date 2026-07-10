/** Placeholder shown when no messages exist yet. */
import styles from "./styles.module.css";

export function EmptyState() {
  return (
    <div class={styles.emptyState}>
      <p>Send a message to start chatting.</p>
    </div>
  );
}
