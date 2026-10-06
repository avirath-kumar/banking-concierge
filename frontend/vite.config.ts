import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const apiTarget = process.env.CONCIERGE_API_URL || "http://localhost:2024";

// We serve the built bundle from the LangGraph agent server at `/concierge/`.
// Use a relative base so all asset URLs are sub-path safe.
export default defineConfig({
  plugins: [react()],
  base: "./",
  server: {
    port: 5173,
    proxy: {
      // In `npm run dev`, forward agent-server endpoints to `langgraph dev`
      // on :2024 so the frontend can talk to the local agent.
      "/threads": apiTarget,
      "/runs": apiTarget,
      "/assistants": apiTarget,
      "/info": apiTarget,
      "/concierge-api": apiTarget,
    },
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
});
