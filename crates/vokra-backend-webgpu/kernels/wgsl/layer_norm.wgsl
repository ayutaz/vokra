// layer_norm — affine layer normalisation over the innermost axis
// (M4-01-T14): out[r, c] = (x[r, c] - mean_r) / sqrt(var_r + eps) * gamma[c] + beta[c]
//
// One workgroup per row: 256 threads take strided partials for the sum and
// the squared-deviation sum, tree-reducing through shared memory. `var_r` is
// the BIASED variance (divide by cols) — identical to the CPU kernel
// (crates/vokra-backend-cpu/src/kernels/scalar.rs::layer_norm) and PyTorch
// nn.LayerNorm. **eps arrives via the uniform from the model config — the
// same value the CPU path uses (M0-08 LAYER_NORM_DEFAULT_EPS = PyTorch 1e-5
// unless the checkpoint overrides); never invented here** (M4-01 spec T14).
// FP32 fixed (NFR-QL-01); reduction association differs from the scalar
// left-to-right fold → tolerance-bounded parity.
//
// Bind contract:
//   binding 0: x      (storage, read)        rows x cols
//   binding 1: gamma  (storage, read)        cols
//   binding 2: beta   (storage, read)        cols
//   binding 3: out    (storage, read_write)  rows x cols
//   binding 4: params (uniform)
//
// Dispatch: (rows, 1, 1) workgroups (plan_layer_norm).

struct Params {
  rows: u32,
  cols: u32,
  eps: f32,
}

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> gamma: array<f32>;
@group(0) @binding(2) var<storage, read> beta: array<f32>;
@group(0) @binding(3) var<storage, read_write> out_buf: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

const WG: u32 = 256u;

var<workgroup> scratch: array<f32, 256>;

fn reduce_sum(lid_x: u32) -> f32 {
  workgroupBarrier();
  var stride = WG / 2u;
  while (stride > 0u) {
    if (lid_x < stride) {
      scratch[lid_x] = scratch[lid_x] + scratch[lid_x + stride];
    }
    workgroupBarrier();
    stride = stride / 2u;
  }
  let total = scratch[0];
  workgroupBarrier();
  return total;
}

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wgid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let row = wgid.x;
  let base = row * params.cols;
  let inv_cols = 1.0 / f32(params.cols);

  // Pass 1: mean.
  var s = 0.0;
  var c = lid.x;
  while (c < params.cols) {
    s = s + x[base + c];
    c = c + WG;
  }
  scratch[lid.x] = s;
  let mean = reduce_sum(lid.x) * inv_cols;

  // Pass 2: biased variance.
  var q = 0.0;
  c = lid.x;
  while (c < params.cols) {
    let d = x[base + c] - mean;
    q = q + d * d;
    c = c + WG;
  }
  scratch[lid.x] = q;
  let variance = reduce_sum(lid.x) * inv_cols;
  let inv_std = 1.0 / sqrt(variance + params.eps);

  // Pass 3: normalize + affine.
  c = lid.x;
  while (c < params.cols) {
    out_buf[base + c] = (x[base + c] - mean) * inv_std * gamma[c] + beta[c];
    c = c + WG;
  }
}
