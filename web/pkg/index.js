// @vokra/web — public loader + session API (M4-01-T20).
//
// Hand-written, zero npm dependencies (ADR M4-01-webgpu-wasm §5). The API is
// concept-symmetric with the C ABI (session / handle / explicit backend —
// include/vokra.h): createSession(...) → session.transcribe(...) →
// session.close().
//
// Artifact selection (ADR M4-01 §4): WASM has no runtime CPU feature
// detection — SIMD acceptance is a module-validation decision — so the
// package ships TWO artifacts (vokra_wasm_simd128.wasm / vokra_wasm_base.wasm)
// and this loader picks one with a WebAssembly.validate probe.
//
// Backend policy (FR-EX-08, no silent fallback): backend "webgpu" requires
// (a) cross-origin isolation (COOP/COEP — the SharedArrayBuffer bridge) and
// (b) a WebGPU adapter. Missing either is an EXPLICIT error that names the
// fix; running on the CPU instead is YOUR explicit `backend: "cpu"` choice.

import { initGpuProxy, attachVokraGpuProxy, CTL_WORDS, DATA_CAPACITY, HEADER_BYTES } from "./vokra_webgpu.js";

// (module (func (result v128) v128.const i32x4 0 0 0 0)) — minimal SIMD
// module for the validate() feature probe.
const SIMD_PROBE = new Uint8Array([
  0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
  0x01, 0x05, 0x01, 0x60, 0x00, 0x01, 0x7b,
  0x03, 0x02, 0x01, 0x00,
  0x0a, 0x16, 0x01, 0x14, 0x00,
  0xfd, 0x0c,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x0b,
]);

/** Which artifact this host will load ("simd128" | "base"). */
export function selectArtifact() {
  return WebAssembly.validate(SIMD_PROBE) ? "simd128" : "base";
}

let nextRequestId = 1;

class VokraSession {
  /** @internal */
  constructor(worker, handle, meta) {
    this._worker = worker;
    this._handle = handle;
    this._meta = meta;
    this._closed = false;
  }

  /** Loader metadata: { artifact: "simd128"|"base", backend, simd128: bool }. */
  get meta() {
    return this._meta;
  }

  /**
   * Transcribes 16 kHz mono audio. Accepts a Float32Array of PCM samples or
   * a WAV (RIFF PCM16 16 kHz mono) ArrayBuffer/Uint8Array. Resolves to
   * { text, wallMs, audioMs, rtf }.
   */
  async transcribe(audio) {
    if (this._closed) throw new Error("session is closed");
    const pcm = audio instanceof Float32Array ? audio : wavToF32(toU8(audio));
    // Transfer a copy (the caller keeps their buffer).
    const buf = pcm.buffer.slice(pcm.byteOffset, pcm.byteOffset + pcm.byteLength);
    const reply = await request(this._worker, { type: "transcribe", handle: this._handle, pcm: buf }, [buf]);
    return { text: reply.value, ...reply.timings };
  }

  /** Releases the model. The worker stays alive for other sessions. */
  async close() {
    if (this._closed) return;
    this._closed = true;
    await request(this._worker, { type: "destroySession", handle: this._handle });
  }
}

function toU8(x) {
  if (x instanceof Uint8Array) return x;
  if (x instanceof ArrayBuffer) return new Uint8Array(x);
  throw new Error("expected Float32Array PCM, Uint8Array or ArrayBuffer WAV");
}

/** Minimal RIFF PCM16 (16 kHz mono) → Float32Array decoder. */
export function wavToF32(bytes) {
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (dv.getUint32(0, false) !== 0x52494646 || dv.getUint32(8, false) !== 0x57415645) {
    throw new Error("not a RIFF/WAVE file (pass raw Float32Array PCM instead)");
  }
  let off = 12;
  let fmt = null;
  let data = null;
  while (off + 8 <= dv.byteLength) {
    const id = dv.getUint32(off, false);
    const size = dv.getUint32(off + 4, true);
    if (id === 0x666d7420) {
      fmt = {
        audioFormat: dv.getUint16(off + 8, true),
        channels: dv.getUint16(off + 10, true),
        sampleRate: dv.getUint32(off + 12, true),
        bitsPerSample: dv.getUint16(off + 22, true),
      };
    } else if (id === 0x64617461) {
      data = new Uint8Array(bytes.buffer, bytes.byteOffset + off + 8, size);
    }
    off += 8 + size + (size % 2);
  }
  if (!fmt || !data) throw new Error("missing fmt/data chunk");
  if (fmt.audioFormat !== 1 || fmt.channels !== 1 || fmt.bitsPerSample !== 16 || fmt.sampleRate !== 16000) {
    throw new Error(
      `WAV must be 16 kHz mono PCM16 (got ${JSON.stringify(fmt)}); resample offline or pass Float32Array PCM`,
    );
  }
  const n = Math.floor(data.byteLength / 2);
  const ddv = new DataView(data.buffer, data.byteOffset, data.byteLength);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = ddv.getInt16(2 * i, true) / 32768;
  return out;
}

function request(worker, msg, transfer = []) {
  return new Promise((resolve, reject) => {
    const id = nextRequestId++;
    const onMessage = (ev) => {
      const m = ev.data;
      if (!m || m.type !== "reply" || m.id !== id) return;
      worker.removeEventListener("message", onMessage);
      if (m.ok) resolve(m);
      else reject(new Error(m.error));
    };
    worker.addEventListener("message", onMessage);
    worker.postMessage({ ...msg, id }, transfer);
  });
}

/**
 * Creates a Vokra ASR session.
 *
 * @param {ArrayBuffer|Uint8Array} modelBytes — a Vokra whisper-base .gguf
 *   (fetch it yourself; models are never bundled in the npm package).
 * @param {{ backend?: "cpu" | "webgpu", baseUrl?: string | URL }} options —
 *   backend defaults to "cpu" (the explicit-choice contract, FR-EX-08);
 *   baseUrl overrides where the .wasm/worker assets are resolved from
 *   (defaults to this module's own URL).
 * @returns {Promise<VokraSession>}
 */
export async function createSession(modelBytes, options = {}) {
  const backend = options.backend ?? "cpu";
  if (backend !== "cpu" && backend !== "webgpu") {
    throw new Error(`unknown backend "${backend}" (use "cpu" or "webgpu" — no silent fallback)`);
  }
  const base = options.baseUrl ?? import.meta.url;
  const artifact = selectArtifact();
  const wasmUrl = new URL(`./vokra_wasm_${artifact}.wasm`, base);
  const workerUrl = new URL("./vokra_worker.js", base);

  // --- WebGPU preconditions are EXPLICIT errors, never silent degradation.
  let gpu = { available: false, error: "backend cpu selected" };
  let ctl = null;
  let data = null;
  let proxyInfo = null;
  if (backend === "webgpu") {
    if (typeof crossOriginIsolated !== "undefined" && !crossOriginIsolated) {
      throw new Error(
        "backend webgpu needs SharedArrayBuffer, which needs cross-origin isolation: serve the " +
          "page with `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: " +
          "require-corp` (see docs/tutorials/web.md, web/demo/serve.mjs). Vokra does not fall " +
          "back to the CPU silently — pass { backend: \"cpu\" } to choose the CPU explicitly.",
      );
    }
    const init = await initGpuProxy();
    if (!init.ok) {
      throw new Error(
        `backend webgpu unavailable: ${init.error}. Vokra does not fall back to the CPU ` +
          `silently — pass { backend: "cpu" } to choose the CPU explicitly.`,
      );
    }
    ctl = new SharedArrayBuffer(CTL_WORDS * 4);
    data = new SharedArrayBuffer(HEADER_BYTES + DATA_CAPACITY);
    gpu = { available: true };
    proxyInfo = { device: init.device };
  }

  const wasmBytes = await (await fetch(wasmUrl)).arrayBuffer();
  const worker = new Worker(workerUrl, { type: "module" });
  if (proxyInfo) {
    attachVokraGpuProxy(worker, {
      ctl: new Int32Array(ctl),
      data: new Uint8Array(data),
      device: proxyInfo.device,
    });
  }

  const ready = new Promise((resolve, reject) => {
    const onMessage = (ev) => {
      const m = ev.data;
      if (!m) return;
      if (m.type === "ready") {
        worker.removeEventListener("message", onMessage);
        resolve(m);
      } else if (m.type === "initError") {
        worker.removeEventListener("message", onMessage);
        reject(new Error(m.error));
      }
    };
    worker.addEventListener("message", onMessage);
  });
  worker.postMessage({ type: "init", wasmBytes, gpu, ctl, data }, [wasmBytes]);
  const readyMsg = await ready;

  const model = modelBytes instanceof Uint8Array
    ? modelBytes.buffer.slice(modelBytes.byteOffset, modelBytes.byteOffset + modelBytes.byteLength)
    : modelBytes.slice(0);
  const reply = await request(
    worker,
    { type: "createSession", gguf: model, backend: backend === "webgpu" ? 1 : 0 },
    [model],
  );
  return new VokraSession(worker, reply.value, {
    artifact,
    backend,
    simd128: readyMsg.simd128 === true,
  });
}
