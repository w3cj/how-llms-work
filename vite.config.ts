import devServer from "@hono/vite-dev-server";
import { defineConfig } from "vite";

export default defineConfig(({ mode }) => {
  if (mode === "client") {
    return {
      build: {
        rollupOptions: {
          input: "./src/client/index.tsx",
          output: {
            entryFileNames: "static/client.js",
            assetFileNames: "static/[name][extname]",
          },
        },
      },
      esbuild: {
        jsxImportSource: "hono/jsx/dom",
      },
    };
  }

  return {
    build: {
      ssr: "src/index.tsx",
      emptyOutDir: false,
    },
    esbuild: {
      jsxImportSource: "hono/jsx",
    },
    plugins: [
      devServer({
        entry: "src/index.tsx",
      }),
    ],
  };
});
