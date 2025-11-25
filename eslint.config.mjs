import eslint from "@eslint/js";
import importPlugin from "eslint-plugin-import";
import commentLengthPlugin from "eslint-plugin-comment-length";
import tseslint from "typescript-eslint";

export default tseslint.config(
  {
    ignores: [
      "docs/**",
      "**/node_modules/",
      "**/pnpm-lock.yaml",
      "__tests__/**",
      "dist/**",
      ".wireit/**",
      "scripts/**",
      "worker-configuration.d.ts",
      "eslint.config.mjs",
      "vitest.config.ts",
      "prettier.config.js",
    ],
  },
  eslint.configs.recommended,
  ...tseslint.configs.recommended,
  importPlugin.flatConfigs.recommended,
  {
    languageOptions: {
      parserOptions: {
        projectService: true,
        tsconfigRootDir: import.meta.dirname,
      },
    },
    plugins: {
      "comment-length": commentLengthPlugin,
    },
    rules: {
      "@typescript-eslint/consistent-type-imports": [
        "error",
        { fixStyle: "inline-type-imports" },
      ],
      "@typescript-eslint/consistent-type-exports": [
        "error",
        { fixMixedExportsWithInlineTypeSpecifier: true },
      ],
      "import/no-unresolved": "off",
      "import/order": [
        "warn",
        {
          "newlines-between": "always",
          groups: ["builtin", "external", "internal", "parent", "sibling", "index"],
          alphabetize: {
            order: "asc",
            caseInsensitive: true,
          },
        },
      ],
      "comment-length/limit-single-line-comments": ["warn", { maxLength: 100 }],
      "comment-length/limit-multi-line-comments": ["warn", { maxLength: 100 }],
    },
  }
);
