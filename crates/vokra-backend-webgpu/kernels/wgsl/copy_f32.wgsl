// copy_f32 — identity element-wise copy (M4-01-T11).
//
// The smoke/proof kernel for the whole dispatch chain (mirrors the Vulkan
// hand-crafted copy_f32.spv role). FP32 storage (NFR-QL-01).
//
// Bind contract (glue: storage buffers 0..n-1, uniform at n — see
// crates/vokra-backend-webgpu/src/plan.rs):
//   binding 0: src   (storage, read)
//   binding 1: dst   (storage, read_write)
//   binding 2: params (uniform)
//
// Dispatch: ceil(n / 256) workgroups on x (plan_copy).

struct Params {
  n: u32,
}

@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i < params.n) {
    dst[i] = src[i];
  }
}
