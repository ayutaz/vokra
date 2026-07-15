#!/usr/bin/env node
// serve.mjs — dependency-free static server with the COOP/COEP headers the
// Vokra WebGPU backend needs (M4-01-T21). SharedArrayBuffer (the worker ↔
// GPU-proxy bridge, ADR M4-01 §3) is only enabled on cross-origin-isolated
// pages, so every response carries:
//
//   Cross-Origin-Opener-Policy:   same-origin
//   Cross-Origin-Embedder-Policy: require-corp
//
// The CPU backend does not need these headers — any static host works.
//
// Usage (from the repo root, after `scripts/build-wasm.sh pkg`):
//   node web/demo/serve.mjs [port]     # default 8788, serves the repo root
// then open http://localhost:8788/web/demo/
//
// (`web/demo/index.html` imports `./pkg/index.js` — the `web/demo/pkg`
// symlink-free path is served by mapping /web/demo/pkg/* → /web/pkg/*.)

import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { extname, join, normalize } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = join(fileURLToPath(new URL(".", import.meta.url)), "..", "..");
const PORT = Number(process.argv[2] ?? 8788);

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".wasm": "application/wasm",
  ".json": "application/json",
  ".wav": "audio/wav",
  ".gguf": "application/octet-stream",
  ".css": "text/css",
  ".ts": "text/plain",
  ".md": "text/plain; charset=utf-8",
};

const server = createServer(async (req, res) => {
  try {
    let path = decodeURIComponent(new URL(req.url, "http://x").pathname);
    if (path.endsWith("/")) path += "index.html";
    // The demo page imports ./pkg/* relative to /web/demo/.
    if (path.startsWith("/web/demo/pkg/")) {
      path = path.replace("/web/demo/pkg/", "/web/pkg/");
    }
    const safe = normalize(path).replace(/^(\.\.[/\\])+/, "");
    const file = join(ROOT, safe);
    if (!file.startsWith(ROOT)) throw Object.assign(new Error("forbidden"), { code: "EACCES" });
    const body = await readFile(file);
    res.writeHead(200, {
      "Content-Type": MIME[extname(file)] ?? "application/octet-stream",
      // Cross-origin isolation (SharedArrayBuffer / Atomics.wait bridge):
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
      // Same-origin subresources under COEP:
      "Cross-Origin-Resource-Policy": "same-origin",
      "Cache-Control": "no-cache",
    });
    res.end(body);
  } catch (e) {
    res.writeHead(e.code === "ENOENT" ? 404 : 500, { "Content-Type": "text/plain" });
    res.end(`${e.code ?? ""} ${req.url}\n`);
  }
});

server.listen(PORT, () => {
  console.log(`Vokra demo server (COOP/COEP enabled): http://localhost:${PORT}/web/demo/`);
  console.log("cpu backend works on any host; webgpu needs this server's headers.");
});
