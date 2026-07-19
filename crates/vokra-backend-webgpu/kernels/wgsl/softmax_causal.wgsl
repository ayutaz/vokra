// softmax_causal — causal-masked row softmax (M4-01-T13).
//
// Row r may attend to columns c <= r + offset (offset = t_k - t_q for a
// decoder step over a KV cache; offset = 0 for square self-attention).
// Masked columns take the -inf path so exp(-inf) = 0 — the same host-mask +
// softmax equivalence the Metal (M2-01 Phase 1) / CUDA (M2-03) causal
// kernels pin. FP32 fixed (NFR-QL-01).
//
// Bind contract:
//   binding 0: x      (storage, read)        rows x cols
//   binding 1: out    (storage, read_write)  rows x cols
//   binding 2: params (uniform)
//
// Dispatch: (rows, 1, 1) workgroups (plan_softmax_causal).

struct Params {
  rows: u32,
  cols: u32,
  offset: u32, // column budget of row r is c <= r + offset
}

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> out_buf: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

const WG: u32 = 256u;
const NEG_INF: f32 = -3.402823466e+38;

var<workgroup> scratch: array<f32, 256>;

fn masked_load(base: u32, row: u32, c: u32) -> f32 {
  if (c > row + params.offset) {
    return NEG_INF;
  }
  return x[base + c];
}

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wgid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let row = wgid.x;
  let base = row * params.cols;

  // Pass 1: masked row max.
  var m = NEG_INF;
  var c = lid.x;
  while (c < params.cols) {
    m = max(m, masked_load(base, row, c));
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

  // Pass 2: exp of masked value (masked lanes write exactly 0.0).
  var s = 0.0;
  c = lid.x;
  while (c < params.cols) {
    var e = 0.0;
    if (c <= row + params.offset) {
      e = exp(x[base + c] - row_max);
    }
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

  // Pass 3: normalize (masked lanes stay exactly 0.0: 0 * inv = 0).
  c = lid.x;
  while (c < params.cols) {
    out_buf[base + c] = out_buf[base + c] * inv;
    c = c + WG;
  }
}
