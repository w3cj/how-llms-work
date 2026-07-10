/** Page header with navigation dropdown (selects the demo page), title, and tagline. */
import type { RoutePath } from "@w3cj/ruta";
import { Link, useLocation } from "@w3cj/ruta";
import { useChatContext } from "../../hooks/use-chat-context.js";
import styles from "./styles.module.css";

const options: { path: RoutePath; label: string }[] = [
  { path: "/", label: "Simple Chat" },
  { path: "/neural-net-xor", label: "XOR Neural Net" },
  { path: "/bpe-token", label: "Basic Tokenizer" },
  { path: "/train-embed", label: "Train Embeddings" },
  { path: "/train-transformer", label: "Train Transformer" },
];

export function Header() {
  const { title, tagline } = useChatContext();
  const { location } = useLocation();

  return (
    <div class={styles.header}>
      <select aria-label="Select an page" class={styles.select}>
        {options.map(route => (
          <Link
            key={route.path}
            to={route.path}
            asChild
          >
            <option selected={location === route.path} value={route.path}>{route.label}</option>
          </Link>
        ))}
      </select>
      <h1 class={styles.title}>{title}</h1>
      <p class={styles.tagline}>{tagline}</p>
    </div>
  );
}
