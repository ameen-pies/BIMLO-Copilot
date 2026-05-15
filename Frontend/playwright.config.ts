import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./src/test/e2e",
  timeout: 30000,
  retries: 1,
  use: {
    baseURL: "http://localhost:5173",
    headless: true,
  },
  webServer: {
    command: "npm run dev",
    port: 5173,
    timeout: 30000,
    reuseExistingServer: true,
  },
});
