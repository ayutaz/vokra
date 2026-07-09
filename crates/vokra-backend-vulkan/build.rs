//! Build script — M3-02-T13. **Skeleton** for the SPIR-V precompile
//! invariant.
//!
//! (See below for design notes.)

// build.rs — M3-02-T13. **Skeleton** build script for the SPIR-V precompile
// invariant.
//
// Design: we do NOT invoke `glslc` from build.rs (that would pull an
// external toolchain into the workspace's build graph — a hidden runtime
// dependency in spirit even if not in `Cargo.lock`). Instead:
//
//   1. Developers run `glslc` locally per shader and commit the `.spv` blob
//      under `kernels/precompiled/`.
//   2. This build script verifies the precompiled directory exists (`cargo
//      warning:` if not) — a soft check that catches "someone deleted the
//      folder" rather than a hard build gate (the include_bytes! call sites
//      themselves would fail hard once the shaders are wired in T14).
//
// This keeps `cargo build -p vokra-backend-vulkan` dependency-free on any
// host: no shader toolchain in the workspace's build graph, zero `Cargo.lock`
// churn.

use std::path::PathBuf;

fn main() {
    // Cargo will re-run this script only when any of these files change.
    println!("cargo:rerun-if-changed=kernels/glsl");
    println!("cargo:rerun-if-changed=kernels/precompiled");
    println!("cargo:rerun-if-changed=kernels/README.md");

    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let precompiled = manifest_dir.join("kernels").join("precompiled");
    if !precompiled.is_dir() {
        println!(
            "cargo:warning=Vulkan precompiled/ dir missing at {} — SPIR-V include_bytes! call sites in T14+ will fail. See kernels/README.md.",
            precompiled.display()
        );
    }
}
