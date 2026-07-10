/** Root component — sets up client-side routing. Each route maps to a different LLM concept demo. */
import { Router } from "@w3cj/ruta";
import { routes } from "./routes.js";

export function Root() {
  return <Router routes={routes} />;
}
