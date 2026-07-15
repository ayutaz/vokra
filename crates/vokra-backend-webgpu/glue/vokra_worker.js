// vokra_worker.js — the dedicated compute worker for the Vokra Web runtime
// (M4-01-T10/T20; ADR M4-01-webgpu-wasm §3).
//
// The inference wasm runs HERE (workers may block on Atomics.wait — the main
// thread may not), and the WebGPU calls are forwarded to the main-thread GPU
// proxy over the SharedArrayBuffer command channel. Zero npm dependencies.
//
// init message (from web/pkg/index.js):
//   { type: "init", wasmBytes: ArrayBuffer, gpu: { available, error },
//     ctl?: SharedArrayBuffer, data?: SharedArrayBuffer }
// requests:
//   { type: "createSession", id, gguf: ArrayBuffer (transferred), backend: 0|1 }
//   { type: "transcribe", id, handle, pcm: ArrayBuffer (transferred f32 LE) }
//   { type: "destroySession", id, handle }
// replies:
//   { type: "reply", id, ok, value?, error?, timings? }

import { createProxyImports, createUnavailableImports } from "./vokra_webgpu.js";

let exportsRef = null;
let memoryRef = null;

function utf8Out(lenFn, readFn) {
  const len = lenFn();
  if (len === 0) return "";
  const ptr = exportsRef.vokra_wasm_alloc(len);
  const n = readFn(ptr, len);
  const bytes = new Uint8Array(memoryRef.buffer, ptr, n).slice();
  exportsRef.vokra_wasm_free(ptr, len);
  return new TextDecoder().decode(bytes);
}

function lastError() {
  return utf8Out(
    () => exportsRef.vokra_wasm_last_error_len(),
    (p, c) => exportsRef.vokra_wasm_last_error_read(p, c),
  );
}

function lastText() {
  return utf8Out(
    () => exportsRef.vokra_wasm_text_len(),
    (p, c) => exportsRef.vokra_wasm_text_read(p, c),
  );
}

/** Copies an ArrayBuffer into a registered wasm allocation (byte-wise — no
 * alignment assumption; the Rust side decodes LE f32 where needed). */
function intoWasm(bytes) {
  const src = new Uint8Array(bytes);
  const ptr = exportsRef.vokra_wasm_alloc(src.length);
  if (ptr === 0) throw new Error("vokra_wasm_alloc failed");
  new Uint8Array(memoryRef.buffer, ptr, src.length).set(src);
  return { ptr, len: src.length };
}

self.addEventListener("message", async (ev) => {
  const msg = ev.data;
  if (!msg || typeof msg !== "object") return;

  if (msg.type === "init") {
    try {
      let gpuImports;
      if (msg.gpu && msg.gpu.available && msg.ctl && msg.data) {
        const ctl = new Int32Array(msg.ctl);
        const data = new Uint8Array(msg.data);
        gpuImports = createProxyImports({
          ctl,
          data,
          kick: () => self.postMessage({ vokraKick: true }),
          getMemory: () => memoryRef,
        });
      } else {
        gpuImports = createUnavailableImports(
          (msg.gpu && msg.gpu.error) ||
            "no WebGPU adapter (or the SharedArrayBuffer bridge is not deployed — COOP/COEP \
headers required; see docs/tutorials/web.md)",
          () => memoryRef,
        );
      }
      const { instance } = await WebAssembly.instantiate(msg.wasmBytes, {
        vokra_webgpu: gpuImports,
      });
      exportsRef = instance.exports;
      memoryRef = instance.exports.memory;
      self.postMessage({
        type: "ready",
        simd128: exportsRef.vokra_wasm_simd128_active() === 1,
      });
    } catch (e) {
      self.postMessage({ type: "initError", error: `${e}` });
    }
    return;
  }

  if (!exportsRef) {
    if (msg.id !== undefined) {
      self.postMessage({ type: "reply", id: msg.id, ok: false, error: "worker not initialised" });
    }
    return;
  }

  try {
    if (msg.type === "createSession") {
      const { ptr, len } = intoWasm(msg.gguf);
      // session_create takes ownership of (ptr, len) — no free here.
      const handle = exportsRef.vokra_wasm_session_create(ptr, len, msg.backend >>> 0);
      if (handle === 0) {
        self.postMessage({ type: "reply", id: msg.id, ok: false, error: lastError() });
      } else {
        self.postMessage({ type: "reply", id: msg.id, ok: true, value: handle });
      }
    } else if (msg.type === "transcribe") {
      const nSamples = (msg.pcm.byteLength / 4) >>> 0;
      const { ptr, len } = intoWasm(msg.pcm);
      const t0 = performance.now();
      const rc = exportsRef.vokra_wasm_transcribe(msg.handle >>> 0, ptr, nSamples);
      const wallMs = performance.now() - t0;
      exportsRef.vokra_wasm_free(ptr, len);
      if (rc !== 0) {
        self.postMessage({ type: "reply", id: msg.id, ok: false, error: lastError() });
      } else {
        const audioMs = (nSamples / 16000) * 1000;
        self.postMessage({
          type: "reply",
          id: msg.id,
          ok: true,
          value: lastText(),
          timings: { wallMs, audioMs, rtf: wallMs / audioMs },
        });
      }
    } else if (msg.type === "destroySession") {
      exportsRef.vokra_wasm_session_destroy(msg.handle >>> 0);
      self.postMessage({ type: "reply", id: msg.id, ok: true });
    }
  } catch (e) {
    if (msg.id !== undefined) {
      self.postMessage({ type: "reply", id: msg.id, ok: false, error: `${e}` });
    }
  }
});
