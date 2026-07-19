// gemv_f32 — row-major matrix-vector product with optional per-row bias
// (M4-01-T12): out[i] = bias[i] + sum_l a[i, l] * x[l]. The Whisper
// tied-logits head (HotOp::Gemv).
//
// One workgroup per output row: 64 threads take strided partial sums over k
// then tree-reduce through workgroup shared memory. FP32 accumulator fixed
// (NFR-QL-01). The partial-sum association differs from the CPU scalar
// left-to-right chain (same posture as the NEON/AVX2 gemv kernels), so
// parity vs the CPU oracle is tolerance-bounded (atol = 0.01 model-level).
//
// Bind contract:
//   binding 0: a      (storage, read)        m x k
//   binding 1: x      (storage, read)        k
//   binding 2: bias   (storage, read)        m (dummy 1-elem buffer when use_bias = 0)
//   binding 3: out    (storage, read_write)  m
//   binding 4: params (uniform)
//
// Dispatch: (m, 1, 1) workgroups (plan_gemv).

struct Params {
  m: u32,
  k: u32,
  use_bias: u32,
}

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> out_buf: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

var<workgroup> partial: array<f32, 64>;

@compute @workgroup_size(64)
fn main(
  @builtin(workgroup_id) wgid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let row = wgid.x;
  var s = 0.0;
  var i = lid.x;
  while (i < params.k) {
    s = s + a[row * params.k + i] * x[i];
    i = i + 64u;
  }
  partial[lid.x] = s;
  workgroupBarrier();
  var stride = 32u;
  while (stride > 0u) {
    if (lid.x < stride) {
      partial[lid.x] = partial[lid.x] + partial[lid.x + stride];
    }
    workgroupBarrier();
    stride = stride / 2u;
  }
  if (lid.x == 0u && row < params.m) {
    var r = partial[0];
    if (params.use_bias == 1u) {
      r = bias[row] + r;
    }
    out_buf[row] = r;
  }
}
