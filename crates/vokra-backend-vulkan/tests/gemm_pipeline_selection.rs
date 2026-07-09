//! M3-02-T14 selector integration surface: host-independent tests that pin
//! `select_gemm_pipeline_variant` to the ADR-declared behaviour ("prefer
//! cooperative-matrix, fall back to subgroup on hardware that lacks the
//! preconditions") and prove the pipeline variant → shader name mapping is
//! stable across `VulkanCapabilities` shapes.
//!
//! These do not require a Vulkan device — they exercise the *decision*
//! surface, which is what downstream backend selection code depends on and
//! is a follow-up hazard when T14 lands its actual pipeline creation.

use vokra_backend_vulkan::{
    GemmPipelinePreference, GemmPipelineVariant, VulkanCapabilities, select_gemm_pipeline_variant,
    spirv::SHADERS,
};

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
        compute_queue_family_index: Some(0),
    }
}

/// Prefer-coop-matrix on a coop-matrix-capable device → coop-matrix.
#[test]
fn coop_matrix_preferred_and_available_picks_coop_matrix() {
    let caps = caps_with(true, true);
    assert_eq!(
        select_gemm_pipeline_variant(&caps, GemmPipelinePreference::PreferCoopMatrix),
        GemmPipelineVariant::CoopMatrix,
    );
}

/// Prefer-coop-matrix on a device that cannot → subgroup fallback
/// (capability-driven — NOT silent op fallback).
#[test]
fn coop_matrix_preferred_but_unavailable_falls_back_to_subgroup() {
    let caps = caps_with(false, true);
    assert_eq!(
        select_gemm_pipeline_variant(&caps, GemmPipelinePreference::PreferCoopMatrix),
        GemmPipelineVariant::Subgroup,
    );
}

/// Force-subgroup path (T33 parity harness): always subgroup, even on
/// coop-matrix-capable hardware.
#[test]
fn force_subgroup_ignores_hardware_capability() {
    for caps in [caps_with(true, true), caps_with(false, true)] {
        assert_eq!(
            select_gemm_pipeline_variant(&caps, GemmPipelinePreference::ForceSubgroup),
            GemmPipelineVariant::Subgroup,
        );
    }
}

/// The `shader_name()` value round-trips through the SPIR-V manifest — the
/// selector's output can be handed straight to `spirv::load_spv`.
#[test]
fn selector_output_names_are_known_manifest_entries() {
    let names: Vec<&str> = SHADERS.iter().map(|s| s.name).collect();
    assert!(names.contains(&GemmPipelineVariant::CoopMatrix.shader_name()));
    assert!(names.contains(&GemmPipelineVariant::Subgroup.shader_name()));
}
