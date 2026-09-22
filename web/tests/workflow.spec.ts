import { expect, test } from "@playwright/test";
import fs from "node:fs";
import path from "node:path";
import AxeBuilder from "@axe-core/playwright";

const token = "browser-test-only-operator-credential";

test("real upload → conversion → validation → benchmark → artifact download", async ({
  page,
  request,
}) => {
  const browserErrors: string[] = [];
  page.on("pageerror", (error) => browserErrors.push(error.message));
  await expect
    .poll(
      async () =>
        (
          await request.get("/api/v1/capabilities", {
            headers: { Authorization: `Bearer ${token}` },
          })
        ).status(),
      { timeout: 90_000 },
    )
    .toBe(200);
  const info = JSON.parse(
    fs.readFileSync("../.tools/e2e-info.json", "utf8"),
  ) as { fixture: string; calibration: string; validation: string };
  await page.goto("/");
  await page.getByLabel("Operator token").fill(token);
  await page.getByRole("button", { name: "Open workspace" }).click();
  await expect(
    page.getByRole("heading", { name: "Model library", exact: true }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Import model", exact: true }).click();
  await page
    .getByLabel("Model files", { exact: true })
    .setInputFiles([
      path.join(info.fixture, "bundle.json"),
      path.join(info.fixture, "weights.safetensors"),
    ]);
  await page.getByRole("button", { name: "Import & inspect" }).click();
  await expect(
    page.getByRole("heading", { name: "Tensor signatures" }),
  ).toBeVisible();
  await page.getByRole("tab", { name: "Convert & optimize" }).click();
  await page.getByRole("button", { name: "Preview route" }).click();
  await expect(page.getByText("torch-onnx", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Convert & validate" }).click();
  await expect(
    page.getByRole("heading", { name: "Numerical validation", exact: true }),
  ).toBeVisible({ timeout: 90_000 });
  await expect(page.locator(".validation-card .badge").first()).toHaveText(
    "validated",
  );
  await expect(
    page.getByText("fp32-default / v1", { exact: true }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Benchmarks", exact: true }).click();
  await page
    .getByRole("button", { name: "Run benchmark", exact: true })
    .click();
  await expect(page.locator(".chart-row")).toHaveCount(1, { timeout: 60_000 });
  await expect(
    page.getByRole("columnheader", { name: "Items / s" }),
  ).toBeVisible();
  const reportDownload = page.waitForEvent("download");
  await page.getByRole("button", { name: "HTML" }).click();
  expect((await reportDownload).suggestedFilename()).toBe("benchmark.html");
  await page
    .getByRole("button", { name: "Model library", exact: true })
    .click();
  const artifactDownload = page.waitForEvent("download");
  await page.getByRole("button", { name: "Download artifact" }).click();
  expect((await artifactDownload).suggestedFilename()).toMatch(
    /modelport-.*\.zip/,
  );
  await page.getByRole("tab", { name: "inspection", exact: true }).click();
  const accessibility = await new AxeBuilder({ page })
    .withTags(["wcag2a", "wcag2aa"])
    .analyze();
  expect(accessibility.violations).toEqual([]);
  fs.mkdirSync("../docs/screenshots", { recursive: true });
  await page.screenshot({
    path: "../docs/screenshots/library.png",
    fullPage: true,
  });
  await page.getByRole("button", { name: "Job activity", exact: true }).click();
  await expect(
    page.getByRole("heading", { name: "Execution history" }),
  ).toBeVisible();
  await page.reload();
  await expect(page.getByLabel("Operator token")).toBeVisible(); // no persistent credential storage
  await page.getByLabel("Operator token").fill(token);
  await page.getByRole("button", { name: "Open workspace" }).click();
  await expect(page.locator(".library-panel tbody tr")).toHaveCount(2);
  // Real dataset inspection and calibrated conversion through the dashboard.
  await page
    .locator(".library-panel tbody tr")
    .filter({ hasText: "ONNX" })
    .getByRole("button")
    .first()
    .click();
  await page.getByRole("tab", { name: "Convert & optimize" }).click();
  await page.locator("#precision").selectOption("static-int8");
  await page
    .getByLabel("Upload calibration NPZ")
    .setInputFiles(info.calibration);
  await expect(page.getByLabel("Calibration dataset")).not.toHaveValue("");
  await page.getByLabel("Upload validation NPZ").setInputFiles(info.validation);
  await expect(page.getByLabel("Validation dataset")).not.toHaveValue("");
  await page.getByRole("button", { name: "Preview route" }).click();
  await expect(
    page.getByText("onnx-static-int8", { exact: true }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Convert & validate" }).click();
  await expect(
    page.getByText("int8-calibrated-v1 / v1", { exact: true }),
  ).toBeVisible({ timeout: 90_000 });
  await expect(page.locator(".validation-card .badge").first()).toHaveText(
    "validated",
  );
  await expect(page.locator(".library-panel tbody tr")).toHaveCount(3);
  await page.getByRole("button", { name: "Import model", exact: true }).click();
  await page.getByLabel("Source", { exact: true }).selectOption("https");
  await page.getByLabel("Import manifest (JSON)").fill(
    JSON.stringify({
      files: [
        {
          url: "https://127.0.0.1/model.onnx",
          path: "model.onnx",
          sha256: "a".repeat(64),
        },
      ],
    }),
  );
  await page.getByRole("button", { name: "Import & inspect" }).click();
  await expect(page.getByRole("dialog").getByRole("alert")).toContainText(
    "REMOTE_ADDRESS_BLOCKED",
  );
  await page.getByRole("button", { name: "Close import" }).click();
  await page.getByRole("button", { name: "Dismiss error" }).click();
  await page.setViewportSize({ width: 390, height: 844 });
  await page.screenshot({
    path: "../docs/screenshots/mobile.png",
    fullPage: true,
  });
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth,
    ),
  ).toBe(true);
  expect(browserErrors).toEqual([]);
});
