/**
 * Route definitions — maps URL paths to chat page components.
 *
 * Each route pairs a path with a hook that manages that page's state and SSE streaming.
 */
import { defineRoutes } from "@w3cj/ruta";
import { App } from "./components/app/index.js";
import { useBpeTokenizeChat } from "./hooks/use-bpe-tokenize-chat.js";
import { useNeuralNetChat } from "./hooks/use-neural-net-chat.js";
import { useSimpleChat } from "./hooks/use-simple-chat.js";
import { useTrainEmbedChat } from "./hooks/use-train-embed-chat.js";
import { useTrainTransformerChat } from "./hooks/use-train-transformer-chat.js";

export const routes = defineRoutes(route => [
  route("/", () => <App chat={useSimpleChat()} />),
  route("/bpe-token", () => <App chat={useBpeTokenizeChat()} />),
  route("/neural-net-xor", () => <App chat={useNeuralNetChat()} />),
  route("/train-embed", () => <App chat={useTrainEmbedChat()} />),
  route("/train-transformer", () => <App chat={useTrainTransformerChat()} />),
]);

declare module "@w3cj/ruta" {
  // eslint-disable-next-line ts/consistent-type-definitions
  interface Register {
    routes: typeof routes;
  }
}
