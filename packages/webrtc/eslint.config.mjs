import globals from "globals";
import tseslint from "typescript-eslint";

import shared from "../../eslint.config.mjs";

export default tseslint.config(...shared, {
  languageOptions: {
    globals: globals.browser,
  },
  rules: {
    "@typescript-eslint/no-explicit-any": "off",
    "import/named": "off",
  },
});
