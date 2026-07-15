//! [`VulkanBackend`] — the `vokra-core` [`Backend`] implementation (M3-02-T23 /
//! T24 / T35).
//!
//! Symmetric with `MetalBackend` (M2-01) and `CudaBackend` (M2-03): two entry
//! points.
//!
//! 1. **Direct kernels** — future SPIR-V compute pipelines dispatched via
//!    `VulkanContext` (M3-02-T14 onwards). None ship in this foundation slice.
//! 2. **Graph execution** — [`Backend::eval_op`] evaluates one op on resolved
//!    [`Tensor`](vokra_core::Tensor) inputs by dispatching to the GPU, and
//!    [`vokra_core::run_graph`] drives it node-by-node. Every uncovered op is
//!    an explicit [`VokraError::UnsupportedOp`], never a silent CPU fallback
//!    (FR-EX-08). [`Backend::execute`] stays a coverage-only check.
//!
//! In this foundation slice, [`VulkanBackend::new`] runs the probe and
//! returns [`VokraError::BackendUnavailable`] if no Vulkan is present.
//! `supports()` returns `false` for every [`OpKind`] until the SPIR-V kernels
//! land (T14〜T22), so any graph execution attempt today surfaces
//! `UnsupportedOp` — the honest state.

use vokra_core::{AudioGraph, Backend, OpKind, Result, Tensor, VokraError};

use crate::spirv::ShaderVariant;

/// Which GEMM SPIR-V pipeline the runtime binds for a given device
/// (M3-02-T14 selection surface).
///
/// The probe (`vokra_vulkan_probe`) reports whether the device meets the
/// cooperative-matrix preconditions (Vulkan 1.3+, `VK_KHR_cooperative_matrix`);
/// [`VulkanBackend::select_gemm_pipeline_variant`] combines that with the
/// caller's *preference* (see [`GemmPipelinePreference`]) to pick either the
/// fast cooperative-matrix pipeline (Ampere+ / RDNA3+ / Adreno 750+) or the
/// subgroup-only fallback (broad Android — Adreno 6xx+ / Mali G7x+).
///
/// This is **capability-driven pipeline selection**, not a silent-fallback op
/// behaviour: the GEMM op still runs, it just picks the shader the hardware
/// actually supports. When *no* SPIR-V blob has been produced yet (the
/// foundation slice), [`VulkanBackend::select_gemm_pipeline_variant`] still
/// returns the *would-be* variant so callers can log the decision — the
/// actual pipeline create call sites (T14+) surface `UnsupportedOp` when
/// [`crate::spirv::load_spv`] returns `None`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GemmPipelineVariant {
    /// Cooperative-matrix + subgroup pipeline
    /// (`gemm_coopmat.spv`). Selected on Ampere+ / RDNA3+ / Adreno 750+.
    CoopMatrix,
    /// Subgroup-only pipeline (`gemm_subgroup.spv`). The Android baseline
    /// (Adreno 6xx+ / Mali G7x+ / Immortalis).
    Subgroup,
}

impl GemmPipelineVariant {
    /// Basename of the corresponding [`crate::spirv::SpirvShader`]
    /// (e.g. `"gemm_coopmat"`).
    #[must_use]
    pub fn shader_name(self) -> &'static str {
        match self {
            GemmPipelineVariant::CoopMatrix => "gemm_coopmat",
            GemmPipelineVariant::Subgroup => "gemm_subgroup",
        }
    }
}

impl From<GemmPipelineVariant> for ShaderVariant {
    fn from(v: GemmPipelineVariant) -> Self {
        match v {
            GemmPipelineVariant::CoopMatrix => ShaderVariant::CoopMatrix,
            GemmPipelineVariant::Subgroup => ShaderVariant::Subgroup,
        }
    }
}

/// Caller preference for [`VulkanBackend::select_gemm_pipeline_variant`].
/// For the two M3-02 preferences the final decision is
/// `min(preference, hardware_capability)` — a caller that *prefers*
/// cooperative-matrix still falls back to subgroup on hardware that lacks
/// the preconditions (capability-driven graceful degradation: the GEMM
/// still runs on the GPU, just with the shader the hardware supports;
/// NOT the FR-EX-08 silent *CPU* fallback).
///
/// [`GemmPipelinePreference::RequireCoopMatrix`] (M4-13-T10) is the
/// explicit-error escape hatch: a caller that *demands* the
/// cooperative-matrix pipeline gets
/// [`VokraError::BackendUnavailable`](vokra_core::VokraError::BackendUnavailable)
/// on hardware without the preconditions — never a quiet downgrade
/// (milestones §8 M4-13 "coop-matrix 非対応 device は明示エラー").
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GemmPipelinePreference {
    /// Prefer the cooperative-matrix pipeline; fall back to subgroup on
    /// devices that lack the preconditions (Vokra's default).
    #[default]
    PreferCoopMatrix,
    /// Force the subgroup pipeline (useful for CI parity gating against the
    /// broad Android baseline; T33 parity harness uses this).
    ForceSubgroup,
    /// Demand the cooperative-matrix pipeline: selection FAILS with an
    /// explicit `BackendUnavailable` when the device lacks the
    /// preconditions, instead of degrading to subgroup (M4-13-T10 red
    /// line — the caller opted out of graceful degradation).
    RequireCoopMatrix,
}

/// Vulkan backend handle.
///
/// On Vulkan-capable targets it holds a [`crate::context::VulkanInstance`]
/// (loader + `VkInstance`), created — and device-probed — in
/// [`VulkanBackend::new`]. On every other target the type still exists (so
/// downstream code can name it) but [`VulkanBackend::new`] fails explicitly:
/// the Vulkan backend is compiled out (NFR-PT-01), never a silent CPU
/// substitute (FR-EX-08).
#[cfg(all(
    feature = "vulkan",
    any(target_os = "linux", target_os = "android", target_os = "windows")
))]
pub struct VulkanBackend {
    // Owns the whole VulkanDevice → VulkanInstance chain (M3-02-T08). The
    // instance is nested inside the device so `vkDestroyDevice` runs before
    // `vkDestroyInstance` on shutdown — Vulkan spec ordering constraint.
    _device: crate::context::VulkanDevice,
    caps: crate::probe::VulkanCapabilities,
}

/// Vulkan backend handle (stub for other targets / feature-off builds — see
/// the Vulkan-target docs above).
#[cfg(not(all(
    feature = "vulkan",
    any(target_os = "linux", target_os = "android", target_os = "windows")
)))]
pub struct VulkanBackend {
    _private: (),
}

// A manual `Debug` impl is used because `VulkanInstance` deliberately does not
// derive `Debug` (raw handles + function pointers are not useful to format).
impl core::fmt::Debug for VulkanBackend {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        #[cfg(all(
            feature = "vulkan",
            any(target_os = "linux", target_os = "android", target_os = "windows")
        ))]
        {
            f.debug_struct("VulkanBackend")
                .field("device_name", &self.caps.device_name)
                .field(
                    "api_version",
                    &format_args!(
                        "{}.{}",
                        self.caps.api_version_major, self.caps.api_version_minor
                    ),
                )
                .finish()
        }
        #[cfg(not(all(
            feature = "vulkan",
            any(target_os = "linux", target_os = "android", target_os = "windows")
        )))]
        {
            f.write_str("VulkanBackend(unavailable)")
        }
    }
}

#[cfg(all(
    feature = "vulkan",
    any(target_os = "linux", target_os = "android", target_os = "windows")
))]
impl VulkanBackend {
    /// Creates a Vulkan backend, running the probe up-front so a missing
    /// loader / device becomes an explicit [`VokraError::BackendUnavailable`]
    /// at construction time (NFR-RL-06 — no silent CPU fall back).
    ///
    /// # Errors
    ///
    /// [`VokraError::BackendUnavailable`] if the Vulkan loader is not
    /// present, the loader is pre-1.1, or no physical device is enumerated
    /// (see [`crate::vokra_vulkan_probe`]).
    pub fn new() -> Result<VulkanBackend> {
        let caps = crate::probe::vokra_vulkan_probe()?;
        if !caps.subgroup_ready {
            return Err(VokraError::BackendUnavailable(format!(
                "Vulkan device present but subgroup precondition not met (Vokra requires \
                 Vulkan 1.1+ and a non-OTHER device type): {}",
                caps.summary()
            )));
        }
        let instance = crate::context::VulkanInstance::new()?;
        // Upgrade to a full VulkanDevice — M3-02-T08 wired all the way through
        // so backend construction actually opens a compute queue on the GPU.
        let device = crate::context::VulkanDevice::new(instance)?;
        // Smoke-test the T08〜T12 runtime object stack against the live device
        // so a broken driver surfaces at construction time rather than
        // silently getting a `NotImplemented` mid-dispatch (FR-EX-08).
        crate::context::smoke_test_runtime_object_stack(&device)?;
        Ok(VulkanBackend {
            _device: device,
            caps,
        })
    }

    /// Access the owned [`crate::context::VulkanDevice`] — used by T14+
    /// dispatch code to allocate buffers / descriptor sets / pipelines.
    #[must_use]
    #[allow(dead_code)] // T14+ dispatch code lands the consumer
    pub(crate) fn device(&self) -> &crate::context::VulkanDevice {
        &self._device
    }

    /// Access the discovered [`crate::probe::VulkanCapabilities`].
    #[must_use]
    pub fn capabilities(&self) -> &crate::probe::VulkanCapabilities {
        &self.caps
    }

    /// Selects the GEMM pipeline variant this device should bind
    /// (M3-02-T14 dispatcher entry, M4-13-T10 explicit-error path).
    ///
    /// For `PreferCoopMatrix` / `ForceSubgroup` the decision is clamped to
    /// what the hardware supports and never errors — capability-driven
    /// pipeline selection, not silent op fallback (the GEMM op still runs,
    /// it just picks the shader the hardware can execute). For
    /// `RequireCoopMatrix` a device without the preconditions is an explicit
    /// [`VokraError::BackendUnavailable`] instead of a downgrade.
    ///
    /// # Errors
    ///
    /// [`VokraError::BackendUnavailable`] only for
    /// [`GemmPipelinePreference::RequireCoopMatrix`] on a device whose probe
    /// reports `coop_matrix_precondition_met == false`.
    pub fn select_gemm_pipeline_variant(
        &self,
        preference: GemmPipelinePreference,
    ) -> Result<GemmPipelineVariant> {
        select_gemm_pipeline_variant(&self.caps, preference)
    }
}

#[cfg(not(all(
    feature = "vulkan",
    any(target_os = "linux", target_os = "android", target_os = "windows")
)))]
impl VulkanBackend {
    /// Non-Vulkan stub: always fails — the Vulkan backend is not compiled
    /// for this target / feature set (NFR-PT-01), and per FR-EX-08 that is
    /// an explicit error rather than a silent CPU substitute.
    ///
    /// # Errors
    ///
    /// Always [`VokraError::BackendUnavailable`].
    pub fn new() -> Result<VulkanBackend> {
        Err(VokraError::BackendUnavailable(
            "Vulkan backend not compiled for this target / feature set (needs --features vulkan \
             on Linux / Android / Windows)."
                .to_owned(),
        ))
    }

    /// Off-target stub of the Vulkan-target method of the same name, so
    /// host-portable integration tests compile everywhere. Unreachable in
    /// practice ([`VulkanBackend::new`] always fails off-target); returns
    /// the conservative Android-baseline variant (and honours the
    /// M4-13-T10 `RequireCoopMatrix` explicit-error contract).
    ///
    /// # Errors
    ///
    /// [`VokraError::BackendUnavailable`] for
    /// [`GemmPipelinePreference::RequireCoopMatrix`] (no Vulkan here at
    /// all, so the preconditions are trivially unmet).
    pub fn select_gemm_pipeline_variant(
        &self,
        preference: GemmPipelinePreference,
    ) -> Result<GemmPipelineVariant> {
        match preference {
            GemmPipelinePreference::RequireCoopMatrix => Err(VokraError::BackendUnavailable(
                "GEMM cooperative-matrix pipeline was explicitly required (RequireCoopMatrix) \
                 but the Vulkan backend is not compiled for this target / feature set \
                 (M4-13-T10, FR-EX-08)."
                    .to_owned(),
            )),
            _ => Ok(GemmPipelineVariant::Subgroup),
        }
    }
}

/// Pure-function form of [`VulkanBackend::select_gemm_pipeline_variant`] —
/// takes a [`crate::VulkanCapabilities`] directly so this decision surface
/// can be exercised host-independently (no Vulkan device required for
/// tests, and downstream crates on non-Vulkan targets can still reason
/// about which pipeline *would* be selected).
///
/// Callers on a Vulkan host normally go through
/// [`VulkanBackend::select_gemm_pipeline_variant`], which forwards here.
///
/// # Errors
///
/// [`VokraError::BackendUnavailable`] only for
/// [`GemmPipelinePreference::RequireCoopMatrix`] on a device whose probe
/// reports `coop_matrix_precondition_met == false` (M4-13-T10 — the caller
/// demanded the coop-matrix path, so refusing loudly is the only honest
/// answer; `PreferCoopMatrix` / `ForceSubgroup` never error).
pub fn select_gemm_pipeline_variant(
    caps: &crate::probe::VulkanCapabilities,
    preference: GemmPipelinePreference,
) -> Result<GemmPipelineVariant> {
    match preference {
        GemmPipelinePreference::ForceSubgroup => Ok(GemmPipelineVariant::Subgroup),
        GemmPipelinePreference::PreferCoopMatrix => {
            if caps.coop_matrix_precondition_met {
                Ok(GemmPipelineVariant::CoopMatrix)
            } else {
                // Capability-driven graceful degradation: the GEMM still
                // runs on the GPU via the subgroup shader — NOT the
                // FR-EX-08 silent *CPU* fallback.
                Ok(GemmPipelineVariant::Subgroup)
            }
        }
        GemmPipelinePreference::RequireCoopMatrix => {
            if caps.coop_matrix_precondition_met {
                Ok(GemmPipelineVariant::CoopMatrix)
            } else {
                Err(VokraError::BackendUnavailable(format!(
                    "GEMM cooperative-matrix pipeline was explicitly required \
                     (RequireCoopMatrix) but this device does not meet the preconditions \
                     (Vulkan 1.3+ AND VK_KHR_cooperative_matrix): {}. Refusing to degrade to \
                     the subgroup pipeline — drop the requirement (PreferCoopMatrix) to opt \
                     back into capability-driven selection (M4-13-T10, FR-EX-08).",
                    caps.summary()
                )))
            }
        }
    }
}

/// Maps a graph [`OpKind`] to the SPIR-V shader that backs its Vulkan
/// graph-executor arm (M4-13-T09) — the single source of truth both
/// [`VulkanBackend::supports`] and the `eval_vulkan_op` dispatcher derive
/// from, so the M3-02-T35 lock-step invariant holds **by construction**.
///
/// `None` means the op has no Vulkan graph arm — either it is a front-end
/// signal op executed by `vokra-ops` (`Stft` / `MelFilterbank` / … — the
/// CUDA arm covers none of these either; `Stft` is the honest gap the
/// M4-13-T14 coverage table records), or its kernel exists only as a
/// model-level primitive with no `OpKind` variant (`gemv` / `layer_norm` /
/// `gelu` / `conv1d` / `softmax_causal` / `transpose` / `gather` —
/// surface 2 of the M4-13-T01 two-surface distinction).
///
/// `gemm_variant` threads the probe's GEMM pipeline selection through:
/// `MatMul`'s backing shader is variant-dependent (`gemm_subgroup` /
/// `gemm_coopmat`), every other covered op has a fixed shader.
#[must_use]
pub fn graph_op_backing_shader(
    op: &OpKind,
    gemm_variant: GemmPipelineVariant,
) -> Option<&'static str> {
    match op {
        // Hand-crafted smoke kernels (always available).
        OpKind::Copy => Some("copy_f32"),
        OpKind::Add => Some("add_f32"),
        // glslc kernels (available once the owner commits their .spv —
        // M4-13-T16; `spirv::has_blob` gates the actual coverage claim).
        OpKind::MatMul => Some(gemm_variant.shader_name()),
        OpKind::Mul => Some("elementwise"),
        OpKind::Softmax => Some("softmax"),
        _ => None,
    }
}

impl Backend for VulkanBackend {
    fn name(&self) -> &str {
        "vulkan"
    }

    fn supports(&self, op: &OpKind) -> bool {
        // M3-02-T35 lock-step gate, blob-driven since M4-13-T09 (ADR
        // M3-02-spirv-generation §7 addendum (b)): `supports()` returns
        // `true` only when (1) the op has a Vulkan graph arm
        // (`graph_op_backing_shader`) AND (2) the arm's SPIR-V blob is
        // actually loadable today (`spirv::has_blob`). While the owner has
        // not committed the glslc blobs (M4-13-T16), MatMul / Mul / Softmax
        // therefore stay `false` — conservative honesty: advertising an op
        // that dispatch would immediately fail with `UnsupportedOp` would
        // make `run_graph`'s coverage precheck lie. Once a blob lands, the
        // op lights up here automatically (no code change).
        //
        // The eval dispatcher derives from the same decision function, so
        // supports() == true ⟺ eval_op reaches a dispatchable kernel — the
        // `supports_and_eval_op_are_lock_step` test pins the pair, and the
        // catch-all arm in `eval_vulkan_op` keeps the FR-EX-08 contract
        // honest for direct callers.
        #[cfg(all(
            feature = "vulkan",
            any(target_os = "linux", target_os = "android", target_os = "windows")
        ))]
        {
            let variant =
                select_gemm_pipeline_variant(&self.caps, GemmPipelinePreference::default())
                    .expect("the default preference (PreferCoopMatrix) never errors");
            graph_op_backing_shader(op, variant).is_some_and(crate::spirv::has_blob)
        }
        #[cfg(not(all(
            feature = "vulkan",
            any(target_os = "linux", target_os = "android", target_os = "windows")
        )))]
        {
            // Off-target the backend cannot even be constructed
            // (`new()` fails), so it honestly supports nothing.
            let _ = op;
            false
        }
    }

    fn execute(&self, graph: &AudioGraph) -> Result<()> {
        for node in graph.nodes() {
            if !self.supports(node.op()) {
                return Err(VokraError::UnsupportedOp(format!(
                    "vulkan backend has no kernel for {:?} (no silent CPU fallback, FR-EX-08)",
                    node.op()
                )));
            }
        }
        // Coverage is satisfied. `execute` stays a coverage-only check; the
        // data-carrying path is `vokra_core::run_graph`, which drives
        // `eval_op` (symmetric with CpuBackend / MetalBackend / CudaBackend).
        Err(VokraError::NotImplemented(
            "vulkan graph-level execution is vokra_core::run_graph (drives eval_op); execute is \
             coverage-only",
        ))
    }

    fn eval_op(&self, op: &OpKind, inputs: &[&Tensor]) -> Result<Vec<Tensor>> {
        // Delegated to the eval module (T24). In the foundation slice the
        // dispatcher rejects every op with an explicit `UnsupportedOp` — no
        // silent CPU fall back. Kept in a separate module so T24〜T29 can
        // extend it op-by-op without touching this file.
        #[cfg(all(
            feature = "vulkan",
            any(target_os = "linux", target_os = "android", target_os = "windows")
        ))]
        {
            crate::eval::eval_vulkan_op(self, op, inputs)
        }
        #[cfg(not(all(
            feature = "vulkan",
            any(target_os = "linux", target_os = "android", target_os = "windows")
        )))]
        {
            let _ = inputs;
            Err(VokraError::UnsupportedOp(format!(
                "vulkan backend is not compiled for this target / feature set; no kernel for \
                 {op:?}"
            )))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::probe::VulkanCapabilities;

    /// Build a synthetic `VulkanCapabilities` — used only in tests so
    /// pipeline-selection is exercisable without a Vulkan host.
    fn caps_with(coop_matrix: bool, subgroup: bool) -> VulkanCapabilities {
        VulkanCapabilities {
            api_version: 0x0040_3000, // encoded 1.3.0 (unused by selector)
            api_version_major: 1,
            api_version_minor: if coop_matrix { 3 } else { 1 },
            device_count: 1,
            device_name: "synthetic".to_owned(),
            vendor_id: 0x1002, // AMD (arbitrary, unused by selector)
            device_type: 2,    // discrete
            subgroup_ready: subgroup,
            coop_matrix_precondition_met: coop_matrix,
            has_khr_cooperative_matrix: coop_matrix,
            has_nv_cooperative_matrix: false,
            compute_queue_family_index: Some(0),
        }
    }

    /// M3-02-T14 selection surface: cooperative-matrix preferred and available
    /// → coop-matrix; cooperative-matrix preferred but unavailable → subgroup
    /// fallback; forced subgroup → subgroup regardless. M4-13-T10:
    /// required-but-unavailable → explicit `BackendUnavailable`.
    #[test]
    fn select_gemm_pipeline_variant_is_capability_driven() {
        // Prefer coop-matrix on a device that supports it — pick coop-matrix.
        assert_eq!(
            select_gemm_pipeline_variant(
                &caps_with(true, true),
                GemmPipelinePreference::PreferCoopMatrix,
            )
            .unwrap(),
            GemmPipelineVariant::CoopMatrix,
        );

        // Prefer coop-matrix on a device that cannot — fall back to subgroup
        // (capability-driven, NOT silent op fallback).
        assert_eq!(
            select_gemm_pipeline_variant(
                &caps_with(false, true),
                GemmPipelinePreference::PreferCoopMatrix,
            )
            .unwrap(),
            GemmPipelineVariant::Subgroup,
        );

        // Force subgroup — always subgroup even when coop-matrix is available
        // (T33 parity harness path — hard baseline for Android).
        assert_eq!(
            select_gemm_pipeline_variant(
                &caps_with(true, true),
                GemmPipelinePreference::ForceSubgroup,
            )
            .unwrap(),
            GemmPipelineVariant::Subgroup,
        );

        // Require coop-matrix on a device that cannot — explicit error,
        // never a quiet downgrade (M4-13-T10).
        assert!(matches!(
            select_gemm_pipeline_variant(
                &caps_with(false, true),
                GemmPipelinePreference::RequireCoopMatrix,
            ),
            Err(VokraError::BackendUnavailable(_)),
        ));
        // …and on capable hardware it selects the fast path.
        assert_eq!(
            select_gemm_pipeline_variant(
                &caps_with(true, true),
                GemmPipelinePreference::RequireCoopMatrix,
            )
            .unwrap(),
            GemmPipelineVariant::CoopMatrix,
        );

        // Preference::default() is PreferCoopMatrix.
        assert_eq!(
            GemmPipelinePreference::default(),
            GemmPipelinePreference::PreferCoopMatrix,
        );
    }

    /// The variant → shader-name mapping is stable and matches the manifest.
    #[test]
    fn gemm_variant_shader_names_match_manifest() {
        use crate::spirv::{SHADERS, ShaderVariant};
        for v in [
            GemmPipelineVariant::CoopMatrix,
            GemmPipelineVariant::Subgroup,
        ] {
            let name = v.shader_name();
            let matches: Vec<_> = SHADERS.iter().filter(|s| s.name == name).collect();
            assert_eq!(
                matches.len(),
                1,
                "shader `{name}` from GemmPipelineVariant not found in SHADERS",
            );
            let expected_variant: ShaderVariant = v.into();
            assert_eq!(
                matches[0].variant, expected_variant,
                "manifest ShaderVariant for `{name}` does not match GemmPipelineVariant",
            );
        }
    }

    #[test]
    fn backend_name_is_vulkan() {
        // `name()` is target-independent and needs no device. On a
        // non-Vulkan host `new()` errors — assert both branches.
        #[cfg(all(
            feature = "vulkan",
            any(target_os = "linux", target_os = "android", target_os = "windows")
        ))]
        {
            if let Ok(backend) = VulkanBackend::new() {
                assert_eq!(backend.name(), "vulkan");
                // Blob-driven coverage (M4-13-T09): MatMul is supported
                // exactly when the probe-selected GEMM variant's blob is
                // loadable — false in the foundation slice, true after the
                // owner's T16 commit, with no test edit either way.
                let variant = backend
                    .select_gemm_pipeline_variant(GemmPipelinePreference::default())
                    .expect("default preference never errors");
                assert_eq!(
                    backend.supports(&OpKind::MatMul),
                    crate::spirv::has_blob(variant.shader_name()),
                );
            }
        }
        #[cfg(not(all(
            feature = "vulkan",
            any(target_os = "linux", target_os = "android", target_os = "windows")
        )))]
        {
            assert!(matches!(
                VulkanBackend::new(),
                Err(VokraError::BackendUnavailable(_))
            ));
        }
    }

    /// M3-02-T35: silent CPU fall-back is explicitly forbidden. A graph that
    /// carries any op the backend does not cover MUST surface as
    /// `UnsupportedOp` from `execute`, never quietly succeed.
    #[test]
    fn execute_rejects_uncovered_ops_explicitly() {
        // On a non-Vulkan host we can't build a real VulkanBackend; the type
        // still exists so we exercise the trait via the target-agnostic error
        // wiring (backend.rs's `#[cfg(not(..))]` arm). On a Vulkan host we go
        // through the real path.
        #[cfg(all(
            feature = "vulkan",
            any(target_os = "linux", target_os = "android", target_os = "windows")
        ))]
        {
            let Ok(backend) = VulkanBackend::new() else {
                eprintln!("no Vulkan; skipping execute() coverage test");
                return;
            };
            use vokra_core::{DType, GraphBuilder, TensorDesc};
            // A PERMANENTLY uncovered graph op (front-end signal op with no
            // Vulkan graph arm — `graph_op_backing_shader` returns None):
            // stays an explicit UnsupportedOp even after the owner's blob
            // commit widens the covered set (M4-13-T09 blob-driven
            // coverage made MatMul time-dependent, so it no longer serves
            // as the permanent negative here).
            let mut mb = GraphBuilder::new();
            let x = mb.add_tensor(TensorDesc::new("x", DType::F32, [64]));
            let y = mb.add_tensor(TensorDesc::new("y", DType::F32, [64]));
            mb.add_node(OpKind::DcOffsetRemove, &[x], &[y]);
            mb.mark_input(x);
            mb.mark_output(y);
            let g = mb.finish().expect("valid graph");
            assert!(matches!(
                backend.execute(&g),
                Err(VokraError::UnsupportedOp(_))
            ));
        }
    }
}
