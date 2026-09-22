// Opt-in local running-Compose browser check. Credentials never leave this process/browser.
import { chromium, expect } from "@playwright/test";
import { execFileSync } from "node:child_process";
import fs from "node:fs";

const token = JSON.parse(
  execFileSync(
    "docker",
    [
      "compose",
      "-f",
      "docker/compose.yml",
      "exec",
      "-T",
      "modelport",
      "modelport",
      "token",
    ],
    { cwd: "..", encoding: "utf8" },
  ),
).token;
const baseURL = `http://127.0.0.1:${process.env.MODELPORT_PORT || 8765}`;
const response = await fetch(`${baseURL}/api/v1/models`, {
  headers: { Authorization: `Bearer ${token}` },
});
expect(response.status).toBe(200);
const models = await response.json();
expect(Array.isArray(models)).toBe(true);
expect(models.length).toBeGreaterThanOrEqual(4);
const evidenceDirectory = "../.tools/docker-evidence/browser";
fs.mkdirSync(evidenceDirectory, { recursive: true });
const browser = await chromium.launch();
try {
  const page = await browser.newPage({
    viewport: { width: 1440, height: 1100 },
  });
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto(baseURL);
  await page.getByLabel("Operator token").fill(token);
  await page.getByRole("button", { name: "Open workspace" }).click();
  await expect(page.locator(".library-panel tbody tr")).toHaveCount(
    models.length,
  );
  await page.screenshot({
    path: `${evidenceDirectory}/docker.png`,
    fullPage: true,
  });
  await page.locator(".table-action").first().click();
  await expect(
    page.getByRole("heading", { name: "Tensor signatures" }),
  ).toBeVisible();
  await page.setViewportSize({ width: 390, height: 844 });
  await page.screenshot({
    path: `${evidenceDirectory}/mobile.png`,
    fullPage: true,
  });
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth,
    ),
  ).toBe(true);
  await page.setViewportSize({ width: 1440, height: 1100 });
  await page.getByRole("button", { name: "Environment", exact: true }).click();
  await expect(
    page.getByRole("heading", { name: "Executable capabilities" }),
  ).toBeVisible();
  await expect(page.getByText("5 adapters passed real probes")).toBeVisible();
  expect(errors).toEqual([]);
  console.log(
    "Docker dashboard: authenticated library, five CPU probes and no JavaScript errors.",
  );
} finally {
  await browser.close();
}
