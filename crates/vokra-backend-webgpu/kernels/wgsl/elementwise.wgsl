// elementwise — binary element-wise op switch: add / mul (M4-01-T11).
//
// The `OpKind::Mul` graph arm and the generic element-wise seam entry.
// The op is a uniform flag (0 = add, 1 = mul) rather than a WGSL `override`
// specialization constant so the glue's pipeline cache stays keyed by shader
// name alone (one pipeline, two ops — the uniform read costs nothing next to
// the memory traffic). FP32 storage (NFR-QL-01).
//
// Bind contract:
//   binding 0: a      (storage, read)
//   binding 1: b      (storage, read)
//   binding 2: out    (storage, read_write)
//   binding 3: params (uniform)
//
// Dispatch: ceil(n / 256) workgroups on x (plan_elementwise).

struct Params {
  n: u32,
  op: u32, // 0 = add, 1 = mul
}

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out_buf: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i < params.n) {
    if (params.op == 1u) {
      out_buf[i] = a[i] * b[i];
    } else {
      out_buf[i] = a[i] + b[i];
    }
  }
}
