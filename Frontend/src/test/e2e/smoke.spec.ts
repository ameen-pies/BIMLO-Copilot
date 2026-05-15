import { test, expect } from "@playwright/test";

test.describe("Smoke tests", () => {
  test("homepage loads and shows nav", async ({ page }) => {
    await page.goto("/");
    await expect(page.locator("nav")).toBeVisible();
    await expect(page.getByText("Bimlo Copilot")).toBeVisible();
  });

  test("chat page loads with input", async ({ page }) => {
    await page.goto("/chat");
    await expect(page.locator("textarea")).toBeVisible();
  });

  test("login modal opens from nav", async ({ page }) => {
    await page.goto("/");
    await page.getByText("Log in").first().click();
    await expect(page.getByText("Welcome back")).toBeVisible();
  });
});
