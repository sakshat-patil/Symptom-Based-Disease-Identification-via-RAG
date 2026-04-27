// Capture README screenshots of the running app via headless Chromium.
//
// Usage:
//   (terminal #1) uvicorn service.api:app  --host 127.0.0.1 --port 8001
//   (terminal #2) cd web && npm run dev
//   (terminal #3) node scripts/capture_screenshots.mjs
//
// Drops PNGs into docs/screenshots/ named after each scene.
//
// Why a script and not just hand-screenshots? The README needs the same shot
// to be reproducible after every UI change. Re-run this whenever the layout
// shifts and the README will pick up the new images.

// playwright is installed under web/node_modules (the only place we have a
// node_modules tree); import from the absolute path so this script can live
// in scripts/ without a separate package.json.
import { chromium } from "../web/node_modules/playwright/index.mjs";
import { mkdir } from "fs/promises";
import { dirname, resolve } from "path";
import { fileURLToPath } from "url";

const TARGET_URL = process.env.UI_URL || "http://localhost:3010";
const HERE = dirname(fileURLToPath(import.meta.url));
const OUT = resolve(HERE, "../docs/screenshots") + "/";

await mkdir(OUT, { recursive: true });

const browser = await chromium.launch();
const ctx = await browser.newContext({
  viewport: { width: 1440, height: 1100 },
  deviceScaleFactor: 2, // retina-quality PNGs
});
const page = await ctx.newPage();

async function shot(name) {
  const path = `${OUT}${name}.png`;
  await page.screenshot({ path, fullPage: true });
  console.log(`wrote ${path}`);
}

console.log(`opening ${TARGET_URL}…`);
await page.goto(TARGET_URL, { waitUntil: "networkidle" });

// Scene 1: empty state. Wait for the rail to render.
await page.waitForSelector(".rail", { timeout: 10_000 });
await page.waitForTimeout(400);
await shot("01-empty-state");

// Scene 2: load the Cardiac event preset and run a diagnosis.
await page.locator("button", { hasText: "Cardiac event" }).first().click();
await page.waitForTimeout(200);
// Click the rail's primary CTA, not the empty-state inline button.
await page.locator(".rail .btn--primary").click();

// Wait for the diagnosis cards to actually render. The PipelineTimeline
// shows immediately, so wait for the actual disease cards instead.
await page.waitForSelector(".dx", { timeout: 60_000 });
await page.waitForTimeout(800); // let entrance animations settle
await shot("02-results-cardiac");

// Scene 3: chip → AI explain modal. Click a chip on the rail.
await page.locator(".chip--clickable").first().click();
await page.waitForSelector(".modal", { timeout: 5_000 });
await page.waitForTimeout(400);
await shot("03-explain-modal");
// Scene 4: dark mode. Reload (clears the modal), re-run the diagnosis, flip theme.
await page.reload({ waitUntil: "networkidle" });
await page.waitForSelector(".rail", { timeout: 10_000 });
await page.locator("button", { hasText: "Cardiac event" }).first().click();
await page.waitForTimeout(150);
await page.locator(".rail .btn--primary").click();
await page.waitForSelector(".dx", { timeout: 60_000 });
await page.waitForTimeout(700);
// Flip theme: the topbar's last icon-btn is the moon/sun toggle.
await page.locator(".topbar .icon-btn").last().click();
await page.waitForTimeout(400);
await shot("04-dark-mode");

await browser.close();
console.log("done.");
