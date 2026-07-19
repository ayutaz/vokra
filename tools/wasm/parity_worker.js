// parity_worker.js — browser per-kernel parity worker (M4-01-T18).
//
// Runs the test-entry artifact (web/dist/vokra_wasm_simd128.wasm — built
// WITH `test-entries`) inside a dedicated worker: the WebGPU calls forward
// to the main-thread GPU proxy over the SAB bridge, and each kernel's
// output is diffed against the CPU oracle (`vokra_test_*` in the same
// instance) right here. Zero npm dependencies.
//
// messages in : { type: "init", wasmBytes, gpu, ctl, data }
//               { type: "runKernels", id }
//               { type: "transcribeBoth", id, gguf: ArrayBuffer, pcm: ArrayBuffer }
// messages out: { type: "ready" } / { type: "initError", error }
//               { type: "reply", id, ok, value?, error? }

import { createProxyImports, createUnavailableImports } from "../../crates/vokra-backend-webgpu/glue/vokra_webgpu.js";

let exp = null;
let mem = null;

function f32sIn(values) {
  const ptr = exp.vokra_wasm_alloc(values.length * 4);
  new Float32Array(mem.buffer, ptr, values.length).set(values);
  return ptr;
}
function f32sOut(ptr, n) {
  return Array.from(new Float32Array(mem.buffer, ptr, n));
}
function lastError() {
  const len = exp.vokra_wasm_last_error_len();
  if (len === 0) return "";
  const p = exp.vokra_wasm_alloc(len);
  const n = exp.vokra_wasm_last_error_read(p, len);
  const s = new TextDecoder().decode(new Uint8Array(mem.buffer, p, n).slice());
  exp.vokra_wasm_free(p, len);
  return s;
}

function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const randVec = (n, seed) => {
  const r = rng(seed);
  return Float32Array.from({ length: n }, () => Math.fround(r() * 2 - 1));
};
const maxAbsDiff = (x, y) => {
  let m = 0;
  for (let i = 0; i < x.length; i++) m = Math.max(m, Math.abs(x[i] - y[i]));
  return m;
};

// CPU references via the forced-scalar CPU test entries in the same
// instance (the M0-08 differential-oracle posture).
function cpuGemm(m, n, k, a, b, bias) {
  const pa = f32sIn(a), pb = f32sIn(b), pbias = bias ? f32sIn(bias) : 0;
  const po = exp.vokra_wasm_alloc(m * n * 4);
  if (exp.vokra_test_gemm(m, n, k, pa, pb, pbias, po, 0) !== 0) throw new Error("cpu gemm failed");
  return f32sOut(po, m * n);
}
function cpuGemv(m, k, a, x, bias) {
  const pa = f32sIn(a), px = f32sIn(x), pb = bias ? f32sIn(bias) : 0;
  const po = exp.vokra_wasm_alloc(m * 4);
  if (exp.vokra_test_gemv(m, k, pa, px, pb, po, 0) !== 0) throw new Error("cpu gemv failed");
  return f32sOut(po, m);
}

// JS-side references for the ops with no CPU test entry (softmax /
// layer_norm / gelu / conv1d / activation) — straight transcriptions of the
// scalar kernels in crates/vokra-backend-cpu/src/kernels/scalar.rs.
function refSoftmax(rows, cols, x, causal, offset) {
  const out = new Float32Array(rows * cols);
  for (let r = 0; r < rows; r++) {
    let max = -Infinity;
    for (let c = 0; c < cols; c++) {
      const v = causal && c > r + offset ? -Infinity : x[r * cols + c];
      max = Math.max(max, v);
    }
    let sum = 0;
    for (let c = 0; c < cols; c++) {
      const v = causal && c > r + offset ? -Infinity : x[r * cols + c];
      const e = Math.exp(Math.fround(v - max));
      out[r * cols + c] = e;
      sum += e;
    }
    for (let c = 0; c < cols; c++) out[r * cols + c] = Math.fround(out[r * cols + c] / sum);
  }
  return out;
}
function refLayerNorm(rows, cols, eps, x, gamma, beta) {
  const out = new Float32Array(rows * cols);
  for (let r = 0; r < rows; r++) {
    let mean = 0;
    for (let c = 0; c < cols; c++) mean += x[r * cols + c];
    mean /= cols;
    let v = 0;
    for (let c = 0; c < cols; c++) v += (x[r * cols + c] - mean) ** 2;
    v /= cols;
    const inv = 1 / Math.sqrt(v + eps);
    for (let c = 0; c < cols; c++) {
      out[r * cols + c] = Math.fround((x[r * cols + c] - mean) * inv * gamma[c] + beta[c]);
    }
  }
  return out;
}
function erfAS(v) {
  // A&S 7.1.26 — identical coefficients to the CPU + WGSL kernels.
  const s = v < 0 ? -1 : 1;
  const ax = Math.abs(v);
  const t = 1 / (1 + 0.3275911 * ax);
  const poly = ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t;
  return s * (1 - poly * Math.exp(-ax * ax));
}
const refGelu = (x) => Float32Array.from(x, (v) => Math.fround(0.5 * v * (1 + erfAS(v * Math.SQRT1_2))));
function refConv1d(inCh, inLen, outCh, kernel, stride, padding, x, w, bias) {
  const outLen = Math.floor((inLen + 2 * padding - kernel) / stride) + 1;
  const out = new Float32Array(outCh * outLen);
  for (let oc = 0; oc < outCh; oc++) {
    for (let t = 0; t < outLen; t++) {
      let acc = bias ? bias[oc] : 0;
      for (let ic = 0; ic < inCh; ic++) {
        for (let kk = 0; kk < kernel; kk++) {
          const pos = t * stride + kk - padding;
          if (pos >= 0 && pos < inLen) acc += w[(oc * inCh + ic) * kernel + kk] * x[ic * inLen + pos];
        }
      }
      out[oc * outLen + t] = Math.fround(acc);
    }
  }
  return out;
}
const refActivation = (kind, x) =>
  Float32Array.from(x, (v) =>
    kind === 0 ? Math.max(v, 0) : kind === 1 ? Math.fround(1 / (1 + Math.exp(-v))) : Math.tanh(v),
  );

const ATOL = 0.01; // NFR-QL-01 FP32 parity bound.

function runKernelSuite() {
  const results = [];
  const push = (name, gpuOut, ref) => {
    const d = maxAbsDiff(gpuOut, ref);
    results.push({ name, maxAbsDiff: d, pass: d <= ATOL });
  };

  // copy — bit-exact expectation.
  {
    const x = randVec(1023, 1);
    const px = f32sIn(x), po = exp.vokra_wasm_alloc(1023 * 4);
    if (exp.vokra_test_webgpu_copy(1023, px, po) !== 0) throw new Error(lastError());
    push("copy_f32", f32sOut(po, 1023), x);
  }
  // elementwise add / mul.
  for (const [op, name] of [[0, "elementwise(add)"], [1, "elementwise(mul)"]]) {
    const a = randVec(1023, 2 + op), b = randVec(1023, 4 + op);
    const pa = f32sIn(a), pb = f32sIn(b), po = exp.vokra_wasm_alloc(1023 * 4);
    if (exp.vokra_test_webgpu_elementwise(op, 1023, pa, pb, po) !== 0) throw new Error(lastError());
    const ref = Float32Array.from(a, (v, i) => (op === 1 ? Math.fround(v * b[i]) : Math.fround(v + b[i])));
    push(name, f32sOut(po, 1023), ref);
  }
  // gemm vs CPU oracle (bias on).
  {
    const m = 33, n = 47, k = 65;
    const a = randVec(m * k, 6), b = randVec(k * n, 7), bias = randVec(n, 8);
    const pa = f32sIn(a), pb = f32sIn(b), pbias = f32sIn(bias);
    const po = exp.vokra_wasm_alloc(m * n * 4);
    if (exp.vokra_test_webgpu_gemm(m, n, k, pa, pb, pbias, po) !== 0) throw new Error(lastError());
    push("gemm_f32", f32sOut(po, m * n), cpuGemm(m, n, k, a, b, bias));
  }
  // gemv vs CPU oracle.
  {
    const m = 129, k = 257;
    const a = randVec(m * k, 9), x = randVec(k, 10), bias = randVec(m, 11);
    const pa = f32sIn(a), px = f32sIn(x), pbias = f32sIn(bias);
    const po = exp.vokra_wasm_alloc(m * 4);
    if (exp.vokra_test_webgpu_gemv(m, k, pa, px, pbias, po) !== 0) throw new Error(lastError());
    push("gemv_f32", f32sOut(po, m), cpuGemv(m, k, a, x, bias));
  }
  // softmax + properties (row sums 1) + causal.
  {
    const rows = 17, cols = 129;
    const x = randVec(rows * cols, 12);
    const px = f32sIn(x), po = exp.vokra_wasm_alloc(rows * cols * 4);
    if (exp.vokra_test_webgpu_softmax(rows, cols, 0, 0, px, po) !== 0) throw new Error(lastError());
    const got = f32sOut(po, rows * cols);
    push("softmax", got, refSoftmax(rows, cols, x, false, 0));
    let sumErr = 0;
    for (let r = 0; r < rows; r++) {
      let s = 0;
      for (let c = 0; c < cols; c++) s += got[r * cols + c];
      sumErr = Math.max(sumErr, Math.abs(s - 1));
    }
    results.push({ name: "softmax rows sum to 1", maxAbsDiff: sumErr, pass: sumErr <= 1e-4 });

    const offset = cols - rows; // decoder-step style
    const pc = exp.vokra_wasm_alloc(rows * cols * 4);
    if (exp.vokra_test_webgpu_softmax(rows, cols, 1, offset, px, pc) !== 0) throw new Error(lastError());
    const gotC = f32sOut(pc, rows * cols);
    push("softmax_causal", gotC, refSoftmax(rows, cols, x, true, offset));
    let maskErr = 0;
    for (let r = 0; r < rows; r++)
      for (let c = r + offset + 1; c < cols; c++) maskErr = Math.max(maskErr, Math.abs(gotC[r * cols + c]));
    results.push({ name: "softmax_causal masked cols are 0", maxAbsDiff: maskErr, pass: maskErr === 0 });
  }
  // layer_norm (eps = the model-config value the CPU path uses).
  {
    const rows = 9, cols = 130, eps = 1e-5;
    const x = randVec(rows * cols, 13), gamma = randVec(cols, 14), beta = randVec(cols, 15);
    const px = f32sIn(x), pg = f32sIn(gamma), pbeta = f32sIn(beta);
    const po = exp.vokra_wasm_alloc(rows * cols * 4);
    if (exp.vokra_test_webgpu_layer_norm(rows, cols, eps, px, pg, pbeta, po) !== 0) throw new Error(lastError());
    push("layer_norm", f32sOut(po, rows * cols), refLayerNorm(rows, cols, eps, x, gamma, beta));
  }
  // gelu (A&S 7.1.26 both sides — records the driver-exp residual).
  {
    const x = randVec(4096, 16);
    const px = f32sIn(x), po = exp.vokra_wasm_alloc(4096 * 4);
    if (exp.vokra_test_webgpu_gelu(4096, px, po) !== 0) throw new Error(lastError());
    push("gelu", f32sOut(po, 4096), refGelu(x));
  }
  // conv1d — the two Whisper stem envelopes (stride 1 and 2).
  for (const stride of [1, 2]) {
    const inCh = 8, inLen = 64, outCh = 6, kernel = 3, padding = 1;
    const outLen = Math.floor((inLen + 2 * padding - kernel) / stride) + 1;
    const x = randVec(inCh * inLen, 17 + stride), w = randVec(outCh * inCh * kernel, 19), bias = randVec(outCh, 20);
    const px = f32sIn(x), pw = f32sIn(w), pb = f32sIn(bias);
    const po = exp.vokra_wasm_alloc(outCh * outLen * 4);
    if (exp.vokra_test_webgpu_conv1d(inCh, inLen, outCh, kernel, stride, padding, px, pw, pb, po, outLen) !== 0) {
      throw new Error(lastError());
    }
    push(`conv1d(stride=${stride})`, f32sOut(po, outCh * outLen), refConv1d(inCh, inLen, outCh, kernel, stride, padding, x, w, bias));
  }
  // activation relu/sigmoid/tanh.
  for (const [kind, name] of [[0, "relu"], [1, "sigmoid"], [2, "tanh"]]) {
    const x = randVec(2048, 21 + kind);
    const px = f32sIn(x), po = exp.vokra_wasm_alloc(2048 * 4);
    if (exp.vokra_test_webgpu_activation(kind, 2048, px, po) !== 0) throw new Error(lastError());
    push(`activation(${name})`, f32sOut(po, 2048), refActivation(kind, x));
  }
  return results;
}

function transcribe(ggufBytes, pcmF32, backend) {
  const g = new Uint8Array(ggufBytes);
  const p = exp.vokra_wasm_alloc(g.length);
  new Uint8Array(mem.buffer, p, g.length).set(g);
  const h = exp.vokra_wasm_session_create(p, g.length, backend);
  if (h === 0) throw new Error(lastError());
  const pcmBytes = new Uint8Array(pcmF32.buffer, pcmF32.byteOffset, pcmF32.byteLength);
  const pp = exp.vokra_wasm_alloc(pcmBytes.length);
  new Uint8Array(mem.buffer, pp, pcmBytes.length).set(pcmBytes);
  const t0 = performance.now();
  const rc = exp.vokra_wasm_transcribe(h, pp, pcmF32.length);
  const wallMs = performance.now() - t0;
  exp.vokra_wasm_free(pp, pcmBytes.length);
  if (rc !== 0) {
    const err = lastError();
    exp.vokra_wasm_session_destroy(h);
    throw new Error(err);
  }
  const len = exp.vokra_wasm_text_len();
  const tp = exp.vokra_wasm_alloc(len);
  const n = exp.vokra_wasm_text_read(tp, len);
  const text = new TextDecoder().decode(new Uint8Array(mem.buffer, tp, n).slice());
  exp.vokra_wasm_free(tp, len);
  exp.vokra_wasm_session_destroy(h);
  return { text, wallMs, rtf: wallMs / ((pcmF32.length / 16000) * 1000) };
}

self.addEventListener("message", async (ev) => {
  const msg = ev.data;
  if (!msg) return;
  if (msg.type === "init") {
    try {
      const gpuImports =
        msg.gpu && msg.gpu.available && msg.ctl && msg.data
          ? createProxyImports({
              ctl: new Int32Array(msg.ctl),
              data: new Uint8Array(msg.data),
              kick: () => self.postMessage({ vokraKick: true }),
              getMemory: () => mem,
            })
          : createUnavailableImports(msg.gpu?.error ?? "no WebGPU", () => mem);
      const { instance } = await WebAssembly.instantiate(msg.wasmBytes, { vokra_webgpu: gpuImports });
      exp = instance.exports;
      mem = instance.exports.memory;
      self.postMessage({ type: "ready" });
    } catch (e) {
      self.postMessage({ type: "initError", error: `${e}` });
    }
    return;
  }
  try {
    if (msg.type === "runKernels") {
      self.postMessage({ type: "reply", id: msg.id, ok: true, value: runKernelSuite() });
    } else if (msg.type === "transcribeBoth") {
      const pcm = new Float32Array(msg.pcm);
      const cpu = transcribe(msg.gguf, pcm, 0);
      const gpu = transcribe(msg.gguf, pcm, 1);
      self.postMessage({ type: "reply", id: msg.id, ok: true, value: { cpu, gpu, match: cpu.text === gpu.text } });
    }
  } catch (e) {
    self.postMessage({ type: "reply", id: msg.id, ok: false, error: `${e}` });
  }
});
