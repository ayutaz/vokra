//! M3-02-T14 / M4-13-T10 selector integration surface: host-independent
//! tests that pin `select_gemm_pipeline_variant` to the ADR-declared
//! behaviour and prove the pipeline variant → shader name mapping is stable
//! across `VulkanCapabilities` shapes.
//!
//! The M4-13-T10 red line distinguishes two coop-matrix outcomes, both
//! asserted below:
//!
//! - **(a) capability-driven fallback** (`PreferCoopMatrix` on a device
//!   without the preconditions → `Subgroup`): graceful degradation, the op
//!   still runs on the GPU with a shader the hardware supports. This is NOT
//!   the FR-EX-08 "silent CPU fallback" — nothing leaves the GPU.
//! - **(b) explicit error** (`RequireCoopMatrix` on a device without the
//!   preconditions → `VokraError::BackendUnavailable`): the caller demanded
//!   the coop-matrix path, so refusing loudly is the only honest answer —
//!   never a quiet switch to a different pipeline.
//!
//! These do not require a Vulkan device — they exercise the *decision*
//! surface, which is what downstream backend selection code depends on.

use vokra_backend_vulkan::{
    GemmPipelinePreference, GemmPipelineVariant, VulkanCapabilities, select_gemm_pipeline_variant,
    spirv::SHADERS,
};
use vokra_core::VokraError;

/// Build a synthetic `VulkanCapabilities` — used only in tests.
fn caps_with(coop_matrix: bool, subgroup: bool) -> VulkanCapabilities {
    VulkanCapabilities {
        api_version: 0x0040_3000,
        api_version_major: 1,
        api_version_minor: if coop_matrix { 3 } else { 1 },
        device_count: 1,
        device_name: "synthetic".to_owned(),
        vendor_id: 0x1002,
        device_type: 2,
        subgroup_ready: subgroup,
        coop_matrix_precondition_met: coop_matrix,
        has_khr_cooperative_matrix: coop_matrix,
        has_nv_cooperative_matrix: false,
        compute_queue_family_index: Some(0),
    }
}

/// Prefer-coop-matrix on a coop-matrix-capable device → coop-matrix.
#[test]
fn coop_matrix_preferred_and_available_picks_coop_matrix() {
    let caps = caps_with(true, true);
    assert_eq!(
        select_gemm_pipeline_variant(&caps, GemmPipelinePreference::PreferCoopMatrix)
            .expect("PreferCoopMatrix never errors"),
        GemmPipelineVariant::CoopMatrix,
    );
}

/// (a) Prefer-coop-matrix on a device that cannot → subgroup fallback
/// (capability-driven graceful degradation — the GEMM still runs on the
/// GPU; NOT the FR-EX-08 silent *CPU* fallback).
#[test]
fn coop_matrix_preferred_but_unavailable_falls_back_to_subgroup() {
    let caps = caps_with(false, true);
    assert_eq!(
        select_gemm_pipeline_variant(&caps, GemmPipelinePreference::PreferCoopMatrix)
            .expect("PreferCoopMatrix never errors"),
        GemmPipelineVariant::Subgroup,
    );
}

/// Force-subgroup path (T33 parity harness): always subgroup, even on
/// coop-matrix-capable hardware.
#[test]
fn force_subgroup_ignores_hardware_capability() {
    for caps in [caps_with(true, true), caps_with(false, true)] {
        assert_eq!(
            select_gemm_pipeline_variant(&caps, GemmPipelinePreference::ForceSubgroup)
                .expect("ForceSubgroup never errors"),
            GemmPipelineVariant::Subgroup,
        );
    }
}

/// (b) M4-13-T10 red line: `RequireCoopMatrix` on a device WITHOUT the
/// preconditions is an explicit `BackendUnavailable` — never a quiet
/// downgrade to `Subgroup`, never a CPU substitute. The diagnostic names
/// the missing capability so triage needs no debugger.
#[test]
fn require_coop_matrix_without_preconditions_is_explicit_error() {
    let caps = caps_with(false, true);
    let err = select_gemm_pipeline_variant(&caps, GemmPipelinePreference::RequireCoopMatrix)
        .expect_err("RequireCoopMatrix must refuse on a non-coop-matrix device");
    assert!(
        matches!(err, VokraError::BackendUnavailable(_)),
        "expected BackendUnavailable, got {err:?}"
    );
    let msg = format!("{err}");
    assert!(
        msg.contains("cooperative-matrix") || msg.contains("coop-matrix"),
        "diagnostic must name the missing capability: {msg}"
    );
}

/// (b, positive leg) `RequireCoopMatrix` on capable hardware selects the
/// coop-matrix pipeline — the requirement is about refusing degradation,
/// not disabling the fast path.
#[test]
fn require_coop_matrix_with_preconditions_picks_coop_matrix() {
    let caps = caps_with(true, true);
    assert_eq!(
        select_gemm_pipeline_variant(&caps, GemmPipelinePreference::RequireCoopMatrix)
            .expect("preconditions met; RequireCoopMatrix must succeed"),
        GemmPipelineVariant::CoopMatrix,
    );
}

/// The `shader_name()` value round-trips through the SPIR-V manifest — the
/// selector's output can be handed straight to `spirv::load_spv`.
#[test]
fn selector_output_names_are_known_manifest_entries() {
    let names: Vec<&str> = SHADERS.iter().map(|s| s.name).collect();
    assert!(names.contains(&GemmPipelineVariant::CoopMatrix.shader_name()));
    assert!(names.contains(&GemmPipelineVariant::Subgroup.shader_name()));
}
