/**
 * Auto-scroll hook for the message list.
 *
 * Keeps the chat scrolled to the bottom as new content streams in (tokens appearing,
 * epochs ticking, etc.). If the user manually scrolls up to review earlier content,
 * auto-scroll pauses so they aren't yanked back to the bottom.
 *
 * Returns a ref to attach to the scrollable container, plus `handleScroll` and `scrollToBottom`.
 */
import { useEffect, useRef } from "hono/jsx";

export function useAutoScroll(deps: unknown[]) {
  const ref = useRef<HTMLDivElement>(null);
  const autoScrollRef = useRef(true);
  const isScrollingRef = useRef(false);

  useEffect(() => {
    if (!autoScrollRef.current || !ref.current)
      return;
    isScrollingRef.current = true;
    ref.current.scrollTop = ref.current.scrollHeight;
    requestAnimationFrame(() => {
      isScrollingRef.current = false;
    });
  // eslint-disable-next-line react/exhaustive-deps -- deps are passed dynamically by the caller
  }, deps);

  const handleScroll = () => {
    if (isScrollingRef.current)
      return;
    const el = ref.current;
    if (!el)
      return;
    const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 40;
    autoScrollRef.current = atBottom;
  };

  const scrollToBottom = () => {
    autoScrollRef.current = true;
  };

  return { ref, handleScroll, scrollToBottom };
}
