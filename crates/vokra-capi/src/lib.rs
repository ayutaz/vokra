//! # vokra-capi
//!
//! C ABI surface for the Vokra runtime (SRS §1.3: "C ABI"), the single
//! binding layer for Unity / Godot / other host runtimes (FR-API-01).
//!
//! M0-02 ships only the crate skeleton with
//! `crate-type = ["cdylib", "staticlib"]`. The `extern "C"` API, the
//! cbindgen-generated `vokra.h`, and the binding conventions (static
//! callbacks + `userdata` for IL2CPP, `DllImport("__Internal")` for iOS
//! static linking) are implemented in **M0-09**. Note that the C ABI has no
//! stability commitment before v1.0 (IF-01; milestones.md §4.2 表注).
//!
//! # Unsafe policy (NFR-RL-07, SRS §5-(1))
//!
//! A C ABI inevitably requires `unsafe` (raw pointers across the FFI
//! boundary), so this crate opts out of the workspace-wide
//! `unsafe_code = "deny"` below. Safety must be guaranteed *at the API
//! boundary* (argument validation, no panics across FFI), and every
//! `unsafe` block requires a `// SAFETY:` comment (enforced by
//! `clippy::undocumented_unsafe_blocks` at the workspace level).

// Local opt-out from the workspace `unsafe_code = "deny"` lint — see the
// crate-level "Unsafe policy" docs above (M0-02-T03).
#![allow(unsafe_code)]

#[cfg(test)]
mod tests {
    #[test]
    fn links_against_vokra_core() {
        // Smoke test for the crate wiring (M0-02-T02): the C ABI wraps
        // vokra-core types (wrapping itself is M0-09).
        let err = vokra_core::VokraError::NotImplemented("C ABI lands in M0-09");
        assert!(err.to_string().contains("M0-09"));
    }
}
