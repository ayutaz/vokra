// activation — element-wise activation switch: relu / sigmoid / tanh
// (M4-01-T15). Forward asset for the Silero VAD / piper-plus Web paths; the
// Whisper route mainly uses gelu.wgsl. The kind is a uniform flag (same
// single-pipeline rationale as elementwise.wgsl). FP32 fixed (NFR-QL-01).
//
//   kind 0: relu(v)    = max(v, 0)
//   kind 1: sigmoid(v) = 1 / (1 + exp(-v))
//   kind 2: tanh(v)    (WGSL builtin)
//
// Bind contract:
//   binding 0: x      (storage, read)
//   binding 1: out    (storage, read_write)
//   binding 2: params (uniform)
//
// Dispatch: ceil(n / 256) workgroups on x (plan_activation).

struct Params {
  n: u32,
  kind: u32,
}

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> out_buf: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i < params.n) {
    let v = x[i];
    var r = 0.0;
    if (params.kind == 0u) {
      r = max(v, 0.0);
    } else if (params.kind == 1u) {
      r = 1.0 / (1.0 + exp(-v));
    } else {
      r = tanh(v);
    }
    out_buf[i] = r;
  }
}
