#!/usr/bin/env node
// run-whisper-wasm.mjs — Whisper base WASM CPU e2e (M4-01-T19).
//
// Hand-written Node runner, ZERO npm dependencies. Exercises the production
// `vokra_wasm_*` session API of BOTH artifacts end-to-end on the JFK fixture:
//
//   GGUF bytes → vokra_wasm_alloc → vokra_wasm_session_create(backend=CPU)
//   → WAV decode (RIFF parse here, PCM16 → f32) → vokra_wasm_transcribe
//   → transcript compared against the parity-whisper-real expectation.
//
// This is the in-memory GGUF path (no mmap on wasm — vokra-mmap is
// native-only) with the caller's EXPLICIT BackendKind::Cpu selection
// (FR-EX-08: the runner also asserts that backend=webgpu fails with the
// explicit BackendUnavailable message in this GPU-less host — the
// no-silent-fallback proof).
//
// Model gating (fabricated-pass prohibition): the GGUF is NOT committed. The
// runner looks at $VOKRA_WHISPER_GGUF, then models/whisper-base.gguf; when
// absent it prints an explicit SKIP with the fetch instructions and exits 0
// (the same sidecar-gating posture as parity-whisper-real.yml). CI wires the
// model in via the web-wasm.yml `run_whisper_e2e` opt-in input.
//
// Usage:
//   scripts/build-wasm.sh harness
//   VOKRA_WHISPER_GGUF=/path/to/whisper-base.gguf node tools/wasm/run-whisper-wasm.mjs

import { readFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { createUnavailableImports } from "../../crates/vokra-backend-webgpu/glue/vokra_webgpu.js";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..", "..");

// The parity-whisper-real dumper's greedy_text for tests/fixtures/audio/
// jfk-30s.wav on whisper-base (docs/m2-owner-verification-checklist.md,
// run 28954469020).
const EXPECTED =
  " And so my fellow Americans, ask not what your country can do for you, ask what you can do" +
  " for your country.";

const ggufPath =
  process.env.VOKRA_WHISPER_GGUF && existsSync(process.env.VOKRA_WHISPER_GGUF)
    ? process.env.VOKRA_WHISPER_GGUF
    : join(ROOT, "models", "whisper-base.gguf");

if (!existsSync(ggufPath)) {
  console.log(
    "SKIP: no whisper-base GGUF found (set VOKRA_WHISPER_GGUF or place models/whisper-base.gguf" +
      " — scripts/fetch-demo-models.sh has the recipe). The e2e transcription leg is" +
      " model-gated like parity-whisper-real.yml; this is an explicit skip, not a pass.",
  );
  process.exit(0);
}

// --- WAV (RIFF PCM16) → f32 ----------------------------------------------------

function wavToF32(bytes) {
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (dv.getUint32(0, false) !== 0x52494646 /* RIFF */) throw new Error("not a RIFF file");
  if (dv.getUint32(8, false) !== 0x57415645 /* WAVE */) throw new Error("not a WAVE file");
  let off = 12;
  let fmt = null;
  let data = null;
  while (off + 8 <= dv.byteLength) {
    const id = dv.getUint32(off, false);
    const size = dv.getUint32(off + 4, true);
    if (id === 0x666d7420 /* fmt  */) {
      fmt = {
        audioFormat: dv.getUint16(off + 8, true),
        channels: dv.getUint16(off + 10, true),
        sampleRate: dv.getUint32(off + 12, true),
        bitsPerSample: dv.getUint16(off + 22, true),
      };
    } else if (id === 0x64617461 /* data */) {
      data = bytes.subarray(off + 8, off + 8 + size);
    }
    off += 8 + size + (size % 2);
  }
  if (!fmt || !data) throw new Error("missing fmt/data chunk");
  if (fmt.audioFormat !== 1 || fmt.channels !== 1 || fmt.bitsPerSample !== 16 || fmt.sampleRate !== 16000) {
    throw new Error(`fixture must be 16 kHz mono PCM16; got ${JSON.stringify(fmt)}`);
  }
  const n = data.byteLength / 2;
  const ddv = new DataView(data.buffer, data.byteOffset, data.byteLength);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = ddv.getInt16(2 * i, true) / 32768;
  return out;
}

// --- session-API driver ----------------------------------------------------------

function utf8Out(exp, lenFn, readFn) {
  const len = lenFn();
  if (len === 0) return "";
  const ptr = exp.vokra_wasm_alloc(len);
  const n = readFn(ptr, len);
  const s = new TextDecoder().decode(new Uint8Array(exp.memory.buffer, ptr, n).slice());
  exp.vokra_wasm_free(ptr, len);
  return s;
}

async function runArtifact(rel, ggufBytes, pcm) {
  const bytes = await readFile(join(ROOT, rel));
  let memoryRef = null;
  const { instance } = await WebAssembly.instantiate(bytes, {
    vokra_webgpu: createUnavailableImports(
      "Node e2e host: no WebGPU adapter (requestAdapter unavailable outside a browser)",
      () => memoryRef,
    ),
  });
  const exp = instance.exports;
  memoryRef = exp.memory;
  const lastError = () =>
    utf8Out(exp, () => exp.vokra_wasm_last_error_len(), (p, c) => exp.vokra_wasm_last_error_read(p, c));

  console.log(`\n== ${rel} (simd128_active=${exp.vokra_wasm_simd128_active()})`);

  // FR-EX-08 negative leg first: an EXPLICIT webgpu selection must fail with
  // the BackendUnavailable message on this adapterless host — never fall
  // back to CPU silently. (Uses a copy of the GGUF: session_create consumes
  // its buffer.)
  {
    const p = exp.vokra_wasm_alloc(ggufBytes.length);
    new Uint8Array(exp.memory.buffer, p, ggufBytes.length).set(ggufBytes);
    const h = exp.vokra_wasm_session_create(p, ggufBytes.length, 1 /* webgpu */);
    if (h === 0) {
      // Session creation may defer the adapter probe to transcribe time —
      // either failure point is acceptable as long as it is explicit.
      console.log(`  ok  webgpu session refused at create: ${lastError().slice(0, 120)}…`);
    } else {
      const pcmPtr = exp.vokra_wasm_alloc(pcm.byteLength);
      new Uint8Array(exp.memory.buffer, pcmPtr, pcm.byteLength).set(
        new Uint8Array(pcm.buffer, pcm.byteOffset, pcm.byteLength),
      );
      const rc = exp.vokra_wasm_transcribe(h, pcmPtr, pcm.length);
      exp.vokra_wasm_free(pcmPtr, pcm.byteLength);
      exp.vokra_wasm_session_destroy(h);
      if (rc === 0) {
        throw new Error(
          "FR-EX-08 violation: webgpu-backend transcribe SUCCEEDED on a host with no adapter " +
            "(silent CPU fallback?)",
        );
      }
      const msg = lastError();
      if (!/BackendUnavailable|WebGPU|webgpu/i.test(msg)) {
        throw new Error(`webgpu failure message does not name the backend: ${msg}`);
      }
      console.log(`  ok  webgpu transcribe refused explicitly: ${msg.slice(0, 120)}…`);
    }
  }

  // Positive leg: explicit CPU selection.
  const p = exp.vokra_wasm_alloc(ggufBytes.length);
  new Uint8Array(exp.memory.buffer, p, ggufBytes.length).set(ggufBytes);
  const handle = exp.vokra_wasm_session_create(p, ggufBytes.length, 0 /* cpu */);
  if (handle === 0) throw new Error(`session_create failed: ${lastError()}`);

  const pcmPtr = exp.vokra_wasm_alloc(pcm.byteLength);
  new Uint8Array(exp.memory.buffer, pcmPtr, pcm.byteLength).set(
    new Uint8Array(pcm.buffer, pcm.byteOffset, pcm.byteLength),
  );
  const t0 = performance.now();
  const rc = exp.vokra_wasm_transcribe(handle, pcmPtr, pcm.length);
  const wallMs = performance.now() - t0;
  exp.vokra_wasm_free(pcmPtr, pcm.byteLength);
  if (rc !== 0) throw new Error(`transcribe failed: ${lastError()}`);
  const text = utf8Out(exp, () => exp.vokra_wasm_text_len(), (p2, c) => exp.vokra_wasm_text_read(p2, c));
  exp.vokra_wasm_session_destroy(handle);

  const audioS = pcm.length / 16000;
  const rtf = wallMs / 1000 / audioS;
  console.log(`  transcript: ${JSON.stringify(text)}`);
  console.log(`  wall ${(wallMs / 1000).toFixed(2)} s / audio ${audioS.toFixed(2)} s → RTF ${rtf.toFixed(3)} (Node, not browser)`);
  if (text !== EXPECTED) {
    throw new Error(`transcript mismatch:\n  got      ${JSON.stringify(text)}\n  expected ${JSON.stringify(EXPECTED)}`);
  }
  console.log("  ok  transcript matches the parity-whisper-real expectation");
  return { rtf, simd: exp.vokra_wasm_simd128_active() === 1 };
}

const gguf = new Uint8Array(await readFile(ggufPath));
const wav = new Uint8Array(await readFile(join(ROOT, "tests", "fixtures", "audio", "jfk-30s.wav")));
const pcm = wavToF32(wav);
console.log(`model: ${ggufPath} (${(gguf.length / 1e6).toFixed(1)} MB), audio: ${pcm.length} samples`);

const rSimd = await runArtifact("web/dist/vokra_wasm_simd128.wasm", gguf, pcm);
const rBase = await runArtifact("web/dist/vokra_wasm_base.wasm", gguf, pcm);
console.log(
  `\nboth artifacts transcribed identically. RTF simd128=${rSimd.rtf.toFixed(3)} base=${rBase.rtf.toFixed(3)} (Node ${process.version})`,
);
