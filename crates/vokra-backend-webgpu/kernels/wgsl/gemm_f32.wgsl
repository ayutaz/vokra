// gemm_f32 — row-major GEMM with optional per-column bias (M4-01-T12).
//
//   out[i, j] = bias[j] + sum_l a[i, l] * b[l, j]      (a: m x k, b: k x n)
//
// Standard 16x16 workgroup-shared-memory tiled GEMM. **FP32 storage and
// accumulator, fixed** (NFR-QL-01 / CLAUDE.md BF16-mantissa rule; no f16).
// NO fused-attention shape here — the FA v3 red line keeps attention as
// standard GEMM + softmax (M4-07 is the only FA v3 WP).
//
// Accumulation order: the accumulator is seeded with bias[j] BEFORE the
// k-loop and tiles advance in ascending k, so every output element runs the
// same bias-seeded ascending-k mul+add chain as the CPU scalar kernel
// (crates/vokra-backend-cpu/src/kernels/scalar.rs::gemm). Zero-padded tile
// lanes contribute exact `+ 0.0` terms (value-preserving except the -0.0
// edge). GPU drivers may contract mul+add to fma, so parity vs the CPU
// oracle is judged at atol = 0.01 (browser harness), not bit-exactness.
//
// Bind contract:
//   binding 0: a      (storage, read)            m x k
//   binding 1: b      (storage, read)            k x n
//   binding 2: bias   (storage, read)            n (dummy 1-elem buffer when use_bias = 0)
//   binding 3: out    (storage, read_write)      m x n
//   binding 4: params (uniform)
//
// Dispatch: (ceil(n / 16), ceil(m / 16), 1) workgroups (plan_gemm).

struct Params {
  m: u32,
  n: u32,
  k: u32,
  use_bias: u32,
}

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> out_buf: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

const TILE: u32 = 16u;

var<workgroup> tile_a: array<f32, 256>; // TILE x TILE
var<workgroup> tile_b: array<f32, 256>; // TILE x TILE

@compute @workgroup_size(16, 16)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let col = gid.x; // output column j (n axis)
  let row = gid.y; // output row i (m axis)

  var acc = 0.0;
  if (params.use_bias == 1u && col < params.n) {
    acc = bias[col];
  }

  let tiles = (params.k + TILE - 1u) / TILE;
  for (var t = 0u; t < tiles; t = t + 1u) {
    let ak = t * TILE + lid.x;
    if (row < params.m && ak < params.k) {
      tile_a[lid.y * TILE + lid.x] = a[row * params.k + ak];
    } else {
      tile_a[lid.y * TILE + lid.x] = 0.0;
    }
    let bk = t * TILE + lid.y;
    if (bk < params.k && col < params.n) {
      tile_b[lid.y * TILE + lid.x] = b[bk * params.n + col];
    } else {
      tile_b[lid.y * TILE + lid.x] = 0.0;
    }
    workgroupBarrier();
    for (var i = 0u; i < TILE; i = i + 1u) {
      acc = acc + tile_a[lid.y * TILE + i] * tile_b[i * TILE + lid.x];
    }
    workgroupBarrier();
  }

  if (row < params.m && col < params.n) {
    out_buf[row * params.n + col] = acc;
  }
}
