//! Integration test: [`vokra_vulkan_probe`] behaves honestly on any host.
//!
//! Foundation-slice test — a real Vulkan device is NOT required. On a
//! non-Vulkan host (Apple Mac / feature-off build) the probe returns
//! [`VokraError::BackendUnavailable`], which is the deliberate no-silent-CPU-
//! fallback surface (FR-EX-08 / NFR-RL-06). On a Vulkan-capable Linux + lava-
//! pipe host (CI T36 target) it returns a populated capability struct.
//!
//! This test is the M3-02 analogue of `probe_is_honest_and_never_panics` in
//! `vokra-backend-cuda` and the Metal-side backend name assertion in
//! `vokra-backend-metal`.

use vokra_backend_vulkan::{VulkanBackend, vokra_vulkan_probe};
use vokra_core::{Backend, VokraError};

#[test]
fn probe_never_panics_and_is_honest_off_vulkan() {
    match vokra_vulkan_probe() {
        Ok(caps) => {
            assert!(caps.device_count >= 1);
            assert!(!caps.device_name.is_empty());
            eprintln!("vokra_vulkan_probe: {}", caps.summary());
        }
        Err(VokraError::BackendUnavailable(msg)) => {
            eprintln!("vokra_vulkan_probe unavailable: {msg}");
        }
        Err(other) => panic!("probe must return BackendUnavailable off Vulkan, got {other}"),
    }
}

#[test]
fn backend_new_gives_backend_unavailable_off_vulkan() {
    // The Vulkan backend cannot be constructed off a Vulkan-capable host.
    // The stub must be explicit — never a silent CPU substitute.
    match VulkanBackend::new() {
        Ok(_backend) => {
            // If we're on a Vulkan host, the backend must at least report a
            // stable name.
            eprintln!("Vulkan backend constructed — running on a Vulkan-capable host");
        }
        Err(VokraError::BackendUnavailable(msg)) => {
            eprintln!("Vulkan backend unavailable (expected off Vulkan): {msg}");
        }
        Err(other) => panic!("expected BackendUnavailable off Vulkan, got {other}"),
    }
}

#[test]
fn constructed_backend_name_is_vulkan() {
    let Ok(backend) = VulkanBackend::new() else {
        eprintln!("no Vulkan; skipping backend-name test");
        return;
    };
    assert_eq!(backend.name(), "vulkan");
}
