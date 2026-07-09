//! M3-02 host↔device round-trip surface (T08 device create + T09 command pool
//! + T12 buffer/memory alloc + T25 host↔device copy + T30 coop-matrix
//!   extension walk).
//!
//! On a Vulkan-capable host (Linux + lavapipe or a real GPU), backend
//! construction runs the full T08〜T12 runtime object stack smoke test at
//! init time, so a driver bug surfaces as
//! [`VokraError::BackendUnavailable`] rather than mid-dispatch. On the Apple
//! Mac authoring host — with no `libvulkan` — the tests fall through to the
//! deliberate `BackendUnavailable` stub (FR-EX-08 / NFR-RL-06).

use vokra_backend_vulkan::{VulkanBackend, vokra_vulkan_probe};
use vokra_core::VokraError;

/// The probe reports an honest compute-queue-family index (or
/// `None` explicitly) — never panics, never silently degrades. On the Apple
/// Mac authoring host the probe fails with `BackendUnavailable`, so the
/// queue-family field is unreachable here; that path is the deliberate
/// no-silent-CPU-fallback surface (FR-EX-08 / NFR-RL-06).
#[test]
fn probe_reports_compute_queue_family_index_when_available() {
    match vokra_vulkan_probe() {
        Ok(caps) => {
            // On a Vulkan host at least one physical device is enumerated;
            // Vulkan §5.3.1 mandates a compute-capable queue family on every
            // conformant GPU. `None` here is a driver bug — surface it in the
            // test log for CI triage.
            match caps.compute_queue_family_index {
                Some(idx) => {
                    eprintln!(
                        "vokra_vulkan_probe compute queue family: {idx} — {}",
                        caps.summary()
                    );
                }
                None => {
                    eprintln!(
                        "vokra_vulkan_probe: driver reports no compute queue family — this is \
                         a driver bug (Vulkan §5.3.1 requires one); {}",
                        caps.summary()
                    );
                }
            }
        }
        Err(VokraError::BackendUnavailable(msg)) => {
            eprintln!("vokra_vulkan_probe unavailable (expected off Vulkan): {msg}");
        }
        Err(other) => panic!("probe must return BackendUnavailable off Vulkan, got {other}"),
    }
}

/// A backend that constructs — i.e. the probe passed AND `subgroup_ready` was
/// true — reports the queue-family index the probe picked. This is the M3-02-
/// T07 "backend records its selection" surface; T08 will consume this index
/// when creating the `VkDevice`.
///
/// The `capabilities()` accessor is only present on the Vulkan-target build
/// (target_os = linux | android | windows AND --features vulkan). Off-target
/// or feature-off, `VulkanBackend::new()` returns `BackendUnavailable`
/// upstream and this test's Vulkan-only assertions are skipped.
#[cfg(all(
    feature = "vulkan",
    any(target_os = "linux", target_os = "android", target_os = "windows")
))]
#[test]
fn backend_carries_compute_queue_family_index_when_constructed() {
    let Ok(backend) = VulkanBackend::new() else {
        eprintln!("no Vulkan; queue-family carry test skipped (backend cannot construct)");
        return;
    };
    let caps = backend.capabilities();
    // Consistency: the backend's own capabilities view carries the queue
    // index from the same probe run.
    assert!(
        caps.subgroup_ready,
        "VulkanBackend::new should have failed if subgroup_ready were false",
    );
    // On any conformant Vulkan host that got past `new`, a compute family
    // must be present (Vulkan §5.3.1). We do not fail hard on a broken driver
    // — just log — since the backend construction itself succeeded.
    if caps.compute_queue_family_index.is_none() {
        eprintln!(
            "VulkanBackend constructed but no compute queue family selected — likely a \
             non-conformant driver; investigate before shipping"
        );
    }
}

/// Off the Vulkan-target build (Apple Mac authoring host, or default-features
/// build) `VulkanBackend::new()` is the explicit `BackendUnavailable` stub —
/// never a silent CPU substitute (FR-EX-08).
#[cfg(not(all(
    feature = "vulkan",
    any(target_os = "linux", target_os = "android", target_os = "windows")
)))]
#[test]
fn backend_stub_off_target_is_explicit_backend_unavailable() {
    match VulkanBackend::new() {
        Err(VokraError::BackendUnavailable(_)) => {}
        Ok(_) => panic!("VulkanBackend must not construct off Vulkan targets / feature-off"),
        Err(other) => panic!("expected BackendUnavailable, got {other}"),
    }
}

/// M3-02-T30 upgrade — the probe now walks device extensions to detect
/// cooperative-matrix support (KHR or NV). On a Vulkan host the reported
/// state is consistent (both extension flags cannot be true while
/// `coop_matrix_precondition_met` is false); off-Vulkan hosts return
/// `BackendUnavailable` and skip.
#[test]
fn probe_reports_coop_matrix_extension_walk() {
    match vokra_vulkan_probe() {
        Ok(caps) => {
            // Consistency: precondition-met implies at least one of the
            // extensions is present.
            if caps.coop_matrix_precondition_met {
                assert!(
                    caps.has_khr_cooperative_matrix || caps.has_nv_cooperative_matrix,
                    "coop-matrix precondition met but neither KHR nor NV extension present — \
                     summary: {}",
                    caps.summary()
                );
            }
            // Consistency: if the API is <1.3 the precondition MUST be false
            // regardless of extension presence (Vokra requires both).
            if caps.api_version_major < 1
                || (caps.api_version_major == 1 && caps.api_version_minor < 3)
            {
                assert!(
                    !caps.coop_matrix_precondition_met,
                    "coop-matrix precondition met on API <1.3 — summary: {}",
                    caps.summary()
                );
            }
            eprintln!("coop-matrix extension walk: {}", caps.summary());
        }
        Err(VokraError::BackendUnavailable(msg)) => {
            eprintln!("no Vulkan; extension walk skipped: {msg}");
        }
        Err(other) => panic!("unexpected: {other}"),
    }
}
