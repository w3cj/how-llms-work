/** Server entry point — serves the SPA shell, static assets, and all demo API routes. */
import { serve } from "@hono/node-server";
import { serveStatic } from "@hono/node-server/serve-static";
import { Hono } from "hono";
import { jsxRenderer } from "hono/jsx-renderer";

import bpeTokenize from "./routes/bpe-tokenize.js";
import neuralNet from "./routes/neural-net/index.js";
import simpleChat from "./routes/simple-chat.js";
import trainEmbed from "./routes/train-embed/index.js";
import trainTransformer from "./routes/train-transformer/index.js";

const app = new Hono();

app.use("/static/*", serveStatic({ root: "./dist" }));

app.get(
  "*",
  jsxRenderer(({ children }) => (
    <html lang="en">
      <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <title>How LLMs Work</title>
      </head>
      <body>{children}</body>
    </html>
  )),
);

app.get("/", c =>
  c.render(
    <>
      <div id="root" />
      {import.meta.env.PROD && <link rel="stylesheet" href="/static/index.css" />}
      {import.meta.env.PROD && <script type="module" src="/static/client.js" />}
      {!import.meta.env.PROD && <script type="module" src="/src/client/index.tsx" />}
    </>,
  ));

app.notFound(c => c.render(
  <>
    <div id="root" />
    {import.meta.env.PROD && <link rel="stylesheet" href="/static/index.css" />}
    {import.meta.env.PROD && <script type="module" src="/static/client.js" />}
    {!import.meta.env.PROD && <script type="module" src="/src/client/index.tsx" />}
  </>,
));

const _routes = app.route("/", simpleChat).route("/", bpeTokenize).route("/", trainEmbed).route("/", trainTransformer).route("/", neuralNet);

export type App = typeof _routes;
export default app;

if (import.meta.env.PROD) {
  const port = Number(process.env.PORT || 3000);
  serve({ fetch: app.fetch, port }, () => {
    console.log(`Server running at http://localhost:${port}`);
  });
}
