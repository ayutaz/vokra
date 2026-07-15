// softmax — numerically-stable row softmax (M4-01-T13).
//
//   out[r, c] = exp(x[r, c] - max_r) / sum_c exp(x[r, c] - max_r)
//
// One workgroup per row: 256 threads take strided partials for the max and
// the exp-sum, tree-reducing through shared memory (max-shift + exp + sum +
// divide — the same stabilization as the CPU scalar kernel). FP32 fixed
// (NFR-QL-01). Reduction association differs from the scalar left-to-right
// fold, so parity vs the CPU oracle is tolerance-bounded; the softmax
// properties (sum == 1, shift invariance) are asserted by the browser
// harness (M4-01 spec T13).
//
// Bind contract:
//   binding 0: x      (storage, read)        rows x cols
//   binding 1: out    (storage, read_write)  rows x cols
//   binding 2: params (uniform)
//
// Dispatch: (rows, 1, 1) workgroups (plan_softmax).

struct Params {
  rows: u32,
  cols: u32,
}

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> out_buf: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

const WG: u32 = 256u;
const NEG_INF: f32 = -3.402823466e+38; // f32::MIN as the -inf stand-in for empty lanes

var<workgroup> scratch: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wgid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let row = wgid.x;
  let base = row * params.cols;

  // Pass 1: row max (strided partials + tree reduce).
  var m = NEG_INF;
  var c = lid.x;
  while (c < params.cols) {
    m = max(m, x[base + c]);
    c = c + WG;
  }
  scratch[lid.x] = m;
  workgroupBarrier();
  var stride = WG / 2u;
  while (stride > 0u) {
    if (lid.x < stride) {
      scratch[lid.x] = max(scratch[lid.x], scratch[lid.x + stride]);
    }
    workgroupBarrier();
    stride = stride / 2u;
  }
  let row_max = scratch[0];
  workgroupBarrier();

  // Pass 2: exp(x - max) into out, accumulating the sum.
  var s = 0.0;
  c = lid.x;
  while (c < params.cols) {
    let e = exp(x[base + c] - row_max);
    out_buf[base + c] = e;
    s = s + e;
    c = c + WG;
  }
  scratch[lid.x] = s;
  workgroupBarrier();
  stride = WG / 2u;
  while (stride > 0u) {
    if (lid.x < stride) {
      scratch[lid.x] = scratch[lid.x] + scratch[lid.x + stride];
    }
    workgroupBarrier();
    stride = stride / 2u;
  }
  let inv = 1.0 / scratch[0];
  workgroupBarrier();

  // Pass 3: normalize.
  c = lid.x;
  while (c < params.cols) {
    out_buf[base + c] = out_buf[base + c] * inv;
    c = c + WG;
  }
}
