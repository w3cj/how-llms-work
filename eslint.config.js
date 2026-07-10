import antfu from "@antfu/eslint-config";

export default antfu({
  ignores: ["src/routes/train-transformer/.train-worker.mjs"],
  type: "app",
  react: true,
  typescript: true,
  formatters: true,
  stylistic: {
    indent: 2,
    semi: true,
    quotes: "double",
    overrides: {
      "style/no-tabs": "off",
    },
  },
}, {
  rules: {
    "ts/no-redeclare": "off",
    "ts/consistent-type-definitions": ["error", "type"],
    "no-console": ["warn"],
    "antfu/no-top-level-await": ["off"],
    "node/prefer-global/process": ["off"],
    "node/no-process-env": ["error"],
    "react/no-array-index-key": ["off"],
    "perfectionist/sort-imports": [
      "error",
      {
        groups: [
          "type-import",
          ["type-parent", "type-sibling", "type-index", "type-internal"],
          "value-builtin",
          "value-external",
          "value-internal",
          ["value-parent", "value-sibling", "value-index"],
          "side-effect",
          "ts-equals-import",
          "unknown",
        ],
        newlinesBetween: "ignore",
        newlinesInside: "ignore",
        order: "asc",
        type: "natural",
      },
    ],
    "unicorn/filename-case": ["error", {
      case: "kebabCase",
      ignore: ["README.md"],
    }],
  },
}, {
  files: ["**/*.md"],
  rules: {
    "perfectionist/sort-imports": "off",
  },
});
