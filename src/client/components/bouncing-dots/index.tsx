/** Animated three-dot typing indicator shown while the assistant is "thinking." */
import styles from "./styles.module.css";

export function BouncingDots() {
  return (
    <div class={styles.dots}>
      {[0, 1, 2].map(i => (
        <span
          key={i}
          class={styles.dot}
          style={{ animation: `bounce-dot 1.4s ease-in-out ${i * 0.2}s infinite` }}
        />
      ))}
    </div>
  );
}
