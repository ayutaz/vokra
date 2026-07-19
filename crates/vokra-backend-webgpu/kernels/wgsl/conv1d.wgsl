// conv1d — 1-D convolution, Whisper encoder stem envelope (M4-01-T15):
//   out[oc, t] = bias[oc] + sum_ic sum_kk w[oc, ic, kk] * x[ic, t*stride + kk - padding]
//
// Direct convolution (no im2col staging buffer on the GPU). The accumulation
// order is ic-major then kk — identical to the CPU path's im2col + GEMM
// ascending-l order (l = ic * kernel + kk), with the accumulator seeded by
// bias like the CPU GEMM. Out-of-range input positions (the zero padding)
// contribute exact `+ 0.0` terms. stride / padding cover the M0-06 Whisper
// stem values (kernel 3, stride 1 and 2, padding 1); the plan layer
// validates shapes host-side, the kernel just guards. FP32 fixed
// (NFR-QL-01).
//
// Bind contract:
//   binding 0: x      (storage, read)        in_ch x in_len
//   binding 1: w      (storage, read)        out_ch x in_ch x kernel
//   binding 2: bias   (storage, read)        out_ch (dummy 1-elem buffer when use_bias = 0)
//   binding 3: out    (storage, read_write)  out_ch x out_len
//   binding 4: params (uniform)
//
// Dispatch: (ceil(out_len / 64), out_ch, 1) workgroups (plan_conv1d).

struct Params {
  in_ch: u32,
  in_len: u32,
  out_ch: u32,
  kernel: u32,
  stride: u32,
  padding: u32,
  out_len: u32,
  use_bias: u32,
}

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> out_buf: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let t = gid.x;  // output position
  let oc = gid.y; // output channel
  if (t >= params.out_len || oc >= params.out_ch) {
    return;
  }

  var acc = 0.0;
  if (params.use_bias == 1u) {
    acc = bias[oc];
  }
  // Signed origin: t*stride - padding can be negative at the left edge.
  let origin = i32(t * params.stride) - i32(params.padding);
  for (var ic = 0u; ic < params.in_ch; ic = ic + 1u) {
    let w_base = (oc * params.in_ch + ic) * params.kernel;
    let x_base = ic * params.in_len;
    for (var kk = 0u; kk < params.kernel; kk = kk + 1u) {
      let pos = origin + i32(kk);
      if (pos >= 0 && pos < i32(params.in_len)) {
        acc = acc + w[w_base + kk] * x[x_base + u32(pos)];
      }
    }
  }
  out_buf[oc * params.out_len + t] = acc;
}
