import { expect, test } from "@playwright/test";

test("runs a review and stores a human decision", async ({ page }) => {
  await page.goto("/");

  await expect(page.getByRole("heading", { name: /Review your code/ })).toBeVisible();
  await page.getByRole("button", { name: /Run analysis/ }).click();

  await expect(page.getByRole("heading", { name: "Analysis results" })).toBeVisible({
    timeout: 60_000,
  });

  const defer = page.getByRole("button", { name: "Defer" });
  if (await defer.isVisible()) {
    await defer.click();
    await expect(page.getByText(/Decision recorded: defer/i)).toBeVisible();
    await page.getByRole("button", { name: "Refresh review" }).click();
    await expect(page.getByText(/Decision recorded: defer/i)).toBeVisible();
  } else {
    await expect(page.getByText(/No review pending|Decision recorded/i)).toBeVisible();
  }

  await expect(page.getByText("Analysis results")).toBeVisible();
});
