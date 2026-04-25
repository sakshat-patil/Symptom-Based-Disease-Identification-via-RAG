import type { Config } from "tailwindcss";

// The design system uses CSS custom properties (defined in app/globals.css)
// rather than a Tailwind palette, so this config stays minimal. Tailwind
// is still in the build for utility classes (flex, gap, padding, etc.)
// used inside components.
const config: Config = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ["Inter", "ui-sans-serif", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "ui-monospace", "Menlo", "monospace"],
      },
    },
  },
  plugins: [],
};
export default config;
