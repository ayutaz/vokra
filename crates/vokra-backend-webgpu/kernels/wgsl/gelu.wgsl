// gelu — element-wise exact (erf-based) GELU (M4-01-T14):
//   out[i] = 0.5 * x[i] * (1 + erf(x[i] / sqrt(2)))
//
// This is OpenAI Whisper's nn.GELU() (default approximate='none' — the
// exact/erf form, NOT the tanh approximation). WGSL has no erf builtin, so
// the kernel uses the **Abramowitz & Stegun 7.1.26** rational approximation
// with the IDENTICAL coefficients the CPU kernel uses
// (crates/vokra-backend-cpu/src/kernels/scalar.rs ERF_P / ERF_A1..A5 — max
// abs error 1.5e-7 per A&S, far inside the FP32 parity ceiling atol = 0.01,
// NFR-QL-01). The only numeric difference vs the CPU path is the GPU
// driver's exp() rounding (measured in the browser harness; the formula
// transcription itself is pinned by a native test in src/plan.rs that
// re-evaluates this exact expression in Rust against kernels::gelu_f32).
// FP32 fixed.
//
// Source: Abramowitz & Stegun, "Handbook of Mathematical Functions",
// formula 7.1.26 (p = 0.3275911, a1 = 0.254829592, a2 = -0.284496736,
// a3 = 1.421413741, a4 = -1.453152027, a5 = 1.061405429).
//
// Bind contract:
//   binding 0: x      (storage, read)
//   binding 1: out    (storage, read_write)
//   binding 2: params (uniform)
//
// Dispatch: ceil(n / 256) workgroups on x (plan_gelu).

struct Params {
  n: u32,
}

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> out_buf: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

const ERF_P: f32 = 0.3275911;
const ERF_A1: f32 = 0.254829592;
const ERF_A2: f32 = -0.284496736;
const ERF_A3: f32 = 1.421413741;
const ERF_A4: f32 = -1.453152027;
const ERF_A5: f32 = 1.061405429;
const FRAC_1_SQRT_2: f32 = 0.70710678118654752440;

fn erf_approx(v: f32) -> f32 {
  let s = sign(v);
  let ax = abs(v);
  let t = 1.0 / (1.0 + ERF_P * ax);
  let poly = ((((ERF_A5 * t + ERF_A4) * t + ERF_A3) * t + ERF_A2) * t + ERF_A1) * t;
  let y = 1.0 - poly * exp(-ax * ax);
  return s * y;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i < params.n) {
    let v = x[i];
    out_buf[i] = 0.5 * v * (1.0 + erf_approx(v * FRAC_1_SQRT_2));
  }
}
