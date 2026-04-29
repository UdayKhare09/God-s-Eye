import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  build: {
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (id.includes("node_modules/@react-three/fiber")) return "r3f";
          if (id.includes("node_modules/@react-three/drei")) return "drei";
          if (id.includes("node_modules/three/examples/jsm")) {
            return "three-examples";
          }
          if (id.includes("node_modules/three/src")) return "three-src";
          if (id.includes("node_modules/three")) return "three-core";
          if (id.includes("node_modules/postprocessing")) {
            return "postprocessing";
          }
          if (id.includes("node_modules/@react-three/postprocessing")) {
            return "r3f-postprocessing";
          }
          if (id.includes("node_modules/framer-motion")) return "framer-motion";
          if (id.includes("node_modules/gsap")) return "gsap";
          if (
            id.includes("node_modules/react") ||
            id.includes("node_modules/react-dom")
          ) {
            return "react-vendor";
          }
        },
      },
    },
  },
});
