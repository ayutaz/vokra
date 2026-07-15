// add_f32 — element-wise sum out[i] = a[i] + b[i] (M4-01-T11).
//
// Dedicated kernel (not the elementwise op-switch) so the graph-executor
// `OpKind::Add` arm mirrors the Vulkan arm's hand-crafted add_f32 exactly.
// FP32 storage (NFR-QL-01). Lane order identical to the CPU scalar kernel,
// so the result is bit-identical up to GPU driver rounding of `+` (which is
// IEEE-754 exact for f32 add — expected bit-exact; measured in the browser
// harness, honest atol if a driver deviates).
//
// Bind contract:
//   binding 0: a      (storage, read)
//   binding 1: b      (storage, read)
//   binding 2: out    (storage, read_write)
//   binding 3: params (uniform)
//
// Dispatch: ceil(n / 256) workgroups on x (plan_add).

struct Params {
  n: u32,
}

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out_buf: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i < params.n) {
    out_buf[i] = a[i] + b[i];
  }
}
