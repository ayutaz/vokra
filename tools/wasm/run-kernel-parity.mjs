#!/usr/bin/env node
// run-kernel-parity.mjs — Node WASM kernel-parity harness (M4-01-T06).
//
// Hand-written runner, ZERO npm dependencies (NFR-DS-02 / ADR M4-01 §5;
// wasm-bindgen-test is an external crate and is not used). Drives the
// `vokra_test_*` entries of the two build artifacts:
//
//   web/dist/vokra_wasm_simd128.wasm  (RUSTFLAGS=-C target-feature=+simd128)
//   web/dist/vokra_wasm_base.wasm     (no SIMD → scalar dispatch)
//
// Differentials (expectations follow the kernel rustdoc,
// crates/vokra-backend-cpu/src/kernels/wasm_simd128.rs, and the native
// differential.rs tolerances):
//
//   gemm / add / mul : SIMD128 dispatched vs forced-scalar — BIT-EXACT
//                      (the wasm gemm keeps the scalar bias-seeded
//                      ascending-k mul+add chain; no fma in baseline SIMD).
//   gemv             : 4-lane partial sums (NEON idiom) — tolerance-bounded
//                      (GEMV_ATOL=1e-4 / RTOL=1e-4, the native differential
//                      bounds); the measured max |Δ| is printed (honest
//                      record, not a fabricated exact match).
//   cross-artifact   : base artifact must dispatch scalar and agree with the
//                      simd artifact's forced-scalar output bit-exactly.
//
// The runner HARD-FAILS (exit 1) if this Node lacks SIMD support — an
// explicit failure, never a skip (fabricated-pass prohibition). Run with:
//
//   scripts/build-wasm.sh harness && node tools/wasm/run-kernel-parity.mjs

import { readFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { createUnavailableImports } from "../../crates/vokra-backend-webgpu/glue/vokra_webgpu.js";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..", "..");

// --- SIMD feature probe (WebAssembly.validate over a minimal simd module) ----
// Module: (module (func (result v128) v128.const i32x4 0 0 0 0)) — encoded by
// hand; validate() returns false on engines without the simd proposal.
const SIMD_PROBE = new Uint8Array([
  0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, // magic + version
  0x01, 0x05, 0x01, 0x60, 0x00, 0x01, 0x7b,       // type: () -> v128
  0x03, 0x02, 0x01, 0x00,                          // func section
  0x0a, 0x16, 0x01, 0x14, 0x00,                    // code section, body size 20 (locals + v128.const + end)
  0xfd, 0x0c,                                      // v128.const (simd prefix 0xfd, opcode 12)
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // 16-byte zero immediate
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x0b,                                            // end
]);

if (!WebAssembly.validate(SIMD_PROBE)) {
  console.error(
    "FAIL: this Node build does not validate WASM SIMD128 — the harness cannot run its " +
      "differentials here (explicit failure, not a skip; use Node 16.4+ / any current LTS).",
  );
  process.exit(1);
}
console.log("SIMD128 probe: supported by this Node");

// --- instantiate helpers ------------------------------------------------------

async function load(rel) {
  const bytes = await readFile(join(ROOT, rel));
  let memoryRef = null;
  const imports = {
    vokra_webgpu: createUnavailableImports(
      "Node harness: WebGPU is not available (kernel parity is a CPU-path test)",
      () => memoryRef,
    ),
  };
  const { instance } = await WebAssembly.instantiate(bytes, imports);
  memoryRef = instance.exports.memory;
  return instance.exports;
}

function f32sIn(exp, values) {
  const ptr = exp.vokra_wasm_alloc(values.length * 4);
  new Float32Array(exp.memory.buffer, ptr, values.length).set(values);
  return ptr;
}

function f32sOut(exp, ptr, n) {
  return Array.from(new Float32Array(exp.memory.buffer, ptr, n));
}

// Deterministic PRNG (mulberry32) so both artifacts see identical inputs.
function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function randVec(n, seed) {
  const r = rng(seed);
  return Float32Array.from({ length: n }, () => Math.fround(r() * 2 - 1));
}

let failures = 0;
function check(label, ok, detail = "") {
  if (ok) {
    console.log(`  ok  ${label}${detail ? ` (${detail})` : ""}`);
  } else {
    failures += 1;
    console.error(`  FAIL ${label}${detail ? ` (${detail})` : ""}`);
  }
}

function maxAbsDiff(x, y) {
  let m = 0;
  for (let i = 0; i < x.length; i++) m = Math.max(m, Math.abs(x[i] - y[i]));
  return m;
}

function withinTol(x, y, atol, rtol) {
  for (let i = 0; i < x.length; i++) {
    const d = Math.abs(x[i] - y[i]);
    if (d > atol + rtol * Math.abs(y[i])) return false;
  }
  return true;
}

// Native differential.rs bounds (crates/vokra-backend-cpu/tests/differential.rs).
const GEMV_ATOL = 1e-4;
const GEMV_RTOL = 1e-4;
// Activation (sigmoid / tanh / gelu) and reduction (softmax / layer_norm)
// SIMD-vs-scalar ceilings — the same ACTIVATION_ATOL / REDUCTION_ATOL the
// native differential.rs uses (both well under the NFR-QL-01 atol=0.01 bound).
const ACTIVATION_ATOL = 1e-4;
const ACTIVATION_RTOL = 1e-4;
const REDUCTION_ATOL = 1e-4;
const REDUCTION_RTOL = 1e-4;

// --- run one kernel set on one artifact ---------------------------------------

function runKernels(exp, forcedScalar) {
  const m = 17, n = 33, k = 129; // deliberately lane-ragged shapes
  const a = randVec(m * k, 11);
  const b = randVec(k * n, 22);
  const bias = randVec(n, 33);
  const x = randVec(k, 44);
  const gemvBias = randVec(m, 55);
  const va = randVec(1023, 66);
  const vb = randVec(1023, 77);

  const pa = f32sIn(exp, a), pb = f32sIn(exp, b), pbias = f32sIn(exp, bias);
  const pout = exp.vokra_wasm_alloc(m * n * 4);
  if (exp.vokra_test_gemm(m, n, k, pa, pb, pbias, pout, forcedScalar) !== 0) {
    throw new Error("vokra_test_gemm returned an error");
  }
  const gemm = f32sOut(exp, pout, m * n);

  const px = f32sIn(exp, x), pgb = f32sIn(exp, gemvBias);
  const pgo = exp.vokra_wasm_alloc(m * 4);
  if (exp.vokra_test_gemv(m, k, pa, px, pgb, pgo, forcedScalar) !== 0) {
    throw new Error("vokra_test_gemv returned an error");
  }
  const gemv = f32sOut(exp, pgo, m);

  const pva = f32sIn(exp, va), pvb = f32sIn(exp, vb);
  const pao = exp.vokra_wasm_alloc(1023 * 4);
  if (exp.vokra_test_add(1023, pva, pvb, pao, forcedScalar) !== 0) {
    throw new Error("vokra_test_add returned an error");
  }
  const add = f32sOut(exp, pao, 1023);
  const pmo = exp.vokra_wasm_alloc(1023 * 4);
  if (exp.vokra_test_mul(1023, pva, pvb, pmo, forcedScalar) !== 0) {
    throw new Error("vokra_test_mul returned an error");
  }
  const mul = f32sOut(exp, pmo, 1023);

  // Activation kernels (M4-01 SIMD128 completion): a lane-ragged length
  // (1023 = 4*255 + 3) exercises both the vector body and the scalar tail.
  const act = randVec(1023, 88);
  const pact = f32sIn(exp, act);
  const pre = exp.vokra_wasm_alloc(1023 * 4);
  if (exp.vokra_test_relu(1023, pact, pre, forcedScalar) !== 0) {
    throw new Error("vokra_test_relu returned an error");
  }
  const relu = f32sOut(exp, pre, 1023);
  const psi = exp.vokra_wasm_alloc(1023 * 4);
  if (exp.vokra_test_sigmoid(1023, pact, psi, forcedScalar) !== 0) {
    throw new Error("vokra_test_sigmoid returned an error");
  }
  const sigmoid = f32sOut(exp, psi, 1023);
  const pth = exp.vokra_wasm_alloc(1023 * 4);
  if (exp.vokra_test_tanh(1023, pact, pth, forcedScalar) !== 0) {
    throw new Error("vokra_test_tanh returned an error");
  }
  const tanh = f32sOut(exp, pth, 1023);
  const pge = exp.vokra_wasm_alloc(1023 * 4);
  if (exp.vokra_test_gelu(1023, pact, pge, forcedScalar) !== 0) {
    throw new Error("vokra_test_gelu returned an error");
  }
  const gelu = f32sOut(exp, pge, 1023);

  // Reduction kernels: cols=130 is lane-ragged (130 = 4*32 + 2).
  const rows = 7, cols = 130;
  const smIn = randVec(rows * cols, 99);
  const psm = f32sIn(exp, smIn);
  const psmo = exp.vokra_wasm_alloc(rows * cols * 4);
  if (exp.vokra_test_softmax(rows, cols, psm, psmo, forcedScalar) !== 0) {
    throw new Error("vokra_test_softmax returned an error");
  }
  const softmax = f32sOut(exp, psmo, rows * cols);

  const lnIn = randVec(rows * cols, 111);
  const gamma = randVec(cols, 122);
  const beta = randVec(cols, 133);
  const pln = f32sIn(exp, lnIn), pg = f32sIn(exp, gamma), pbeta = f32sIn(exp, beta);
  const plno = exp.vokra_wasm_alloc(rows * cols * 4);
  if (exp.vokra_test_layer_norm(rows, cols, 1e-5, pln, pg, pbeta, plno, forcedScalar) !== 0) {
    throw new Error("vokra_test_layer_norm returned an error");
  }
  const layerNorm = f32sOut(exp, plno, rows * cols);

  return { gemm, gemv, add, mul, relu, sigmoid, tanh, gelu, softmax, layerNorm };
}

// --- main ----------------------------------------------------------------------

const simd = await load("web/dist/vokra_wasm_simd128.wasm");
const base = await load("web/dist/vokra_wasm_base.wasm");

console.log("\nartifact identity:");
check("simd artifact reports simd128 active", simd.vokra_wasm_simd128_active() === 1);
check("base artifact reports simd128 inactive", base.vokra_wasm_simd128_active() === 0);
check(
  "simd artifact dispatches IsaPath::WasmSimd128",
  simd.vokra_test_active_isa_code() === 4,
  `code=${simd.vokra_test_active_isa_code()}`,
);
check(
  "base artifact dispatches IsaPath::Scalar",
  base.vokra_test_active_isa_code() === 0,
  `code=${base.vokra_test_active_isa_code()}`,
);

console.log("\nin-artifact differential (simd128 dispatched vs forced scalar):");
const sd = runKernels(simd, 0); // dispatched = wasm_simd128 kernels
const ss = runKernels(simd, 1); // forced scalar in the same artifact
check("gemm bit-exact vs scalar", maxAbsDiff(sd.gemm, ss.gemm) === 0, `max|Δ|=${maxAbsDiff(sd.gemm, ss.gemm)}`);
check("add bit-exact vs scalar", maxAbsDiff(sd.add, ss.add) === 0);
check("mul bit-exact vs scalar", maxAbsDiff(sd.mul, ss.mul) === 0);
const gemvDiff = maxAbsDiff(sd.gemv, ss.gemv);
check(
  "gemv within native differential bounds (partial-sum reorder)",
  withinTol(sd.gemv, ss.gemv, GEMV_ATOL, GEMV_RTOL),
  `measured max|Δ|=${gemvDiff.toExponential(3)} vs atol=${GEMV_ATOL}/rtol=${GEMV_RTOL}`,
);

// relu is a lane-wise f32x4_max — bit-identical to scalar for the finite
// harness inputs (same class as add / mul).
check("relu bit-exact vs scalar", maxAbsDiff(sd.relu, ss.relu) === 0);

// sigmoid / tanh / gelu run the vectorized poly `exp`, so they must (a) stay
// within ACTIVATION_ATOL of the scalar `std::exp` reference AND (b) differ by
// a strictly-positive amount — a zero delta would mean the dispatched path is
// still the scalar passthrough (this lower bound is exactly what detects an
// un-vectorized kernel, i.e. the M4-01-T05 scalar-delegation gap).
for (const name of ["sigmoid", "tanh", "gelu"]) {
  const d = maxAbsDiff(sd[name], ss[name]);
  check(
    `${name} SIMD poly-exp active & within bound`,
    d > 0 && withinTol(sd[name], ss[name], ACTIVATION_ATOL, ACTIVATION_RTOL),
    `max|Δ|=${d.toExponential(3)} (0 < Δ ≤ atol=${ACTIVATION_ATOL})`,
  );
}

// softmax / layer_norm reorder their row reductions (4-lane partial sums) and
// softmax also uses the poly `exp`, so they are tolerance-bounded, not exact.
const softmaxDiff = maxAbsDiff(sd.softmax, ss.softmax);
check(
  "softmax within reduction bound (reorder + poly-exp)",
  withinTol(sd.softmax, ss.softmax, REDUCTION_ATOL, REDUCTION_RTOL),
  `max|Δ|=${softmaxDiff.toExponential(3)} vs atol=${REDUCTION_ATOL}`,
);
const lnDiff = maxAbsDiff(sd.layerNorm, ss.layerNorm);
check(
  "layer_norm within reduction bound (mean/var reorder)",
  withinTol(sd.layerNorm, ss.layerNorm, REDUCTION_ATOL, REDUCTION_RTOL),
  `max|Δ|=${lnDiff.toExponential(3)} vs atol=${REDUCTION_ATOL}`,
);

console.log("\ncross-artifact (base dispatched == scalar semantics):");
const bd = runKernels(base, 0); // base artifact dispatches scalar
check("gemm base == simd-artifact scalar", maxAbsDiff(bd.gemm, ss.gemm) === 0);
check("gemv base == simd-artifact scalar", maxAbsDiff(bd.gemv, ss.gemv) === 0);
check("add base == simd-artifact scalar", maxAbsDiff(bd.add, ss.add) === 0);
check("mul base == simd-artifact scalar", maxAbsDiff(bd.mul, ss.mul) === 0);
// The base (no-SIMD) artifact dispatches scalar for every kernel, so it must
// agree bit-exactly with the simd artifact's forced-scalar output — including
// the transcendentals (both run the scalar `std::exp` reference here).
check("relu base == simd-artifact scalar", maxAbsDiff(bd.relu, ss.relu) === 0);
check("sigmoid base == simd-artifact scalar", maxAbsDiff(bd.sigmoid, ss.sigmoid) === 0);
check("tanh base == simd-artifact scalar", maxAbsDiff(bd.tanh, ss.tanh) === 0);
check("gelu base == simd-artifact scalar", maxAbsDiff(bd.gelu, ss.gelu) === 0);
check("softmax base == simd-artifact scalar", maxAbsDiff(bd.softmax, ss.softmax) === 0);
check("layer_norm base == simd-artifact scalar", maxAbsDiff(bd.layerNorm, ss.layerNorm) === 0);

if (failures > 0) {
  console.error(`\n${failures} kernel-parity check(s) FAILED`);
  process.exit(1);
}
console.log("\nall kernel-parity checks passed");
