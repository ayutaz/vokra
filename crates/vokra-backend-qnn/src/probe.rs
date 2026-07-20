//! QNN runtime probe (M5-02-T05; FR-EX-08 / NFR-RL-06).
//!
//! [`vokra_qnn_probe`] is the QNN analogue of `vokra_cuda_probe`: it dlopens the
//! QNN HTP (Hexagon) runtime and reports whether it — plus a representative
//! interface entry symbol — is reachable. It embodies the Qualcomm EULA "install
//! model" (the runtime is *detected* at runtime, nothing bundled —
//! `third_party/QUALCOMM-QNN-NOTES.md`), so it compiles on any host in this cfg
//! and returns `BackendUnavailable` where no QNN runtime exists.
//!
//! **A missing library / missing symbol / non-QNN target is an explicit
//! [`VokraError::BackendUnavailable`] — never a silent fall back to the CPU
//! backend** (FR-EX-08 permanent constraint, NFR-RL-06). Whether to run on the
//! CPU instead is the *caller's* explicit backend choice.
//!
//! # Scaffold scope
//!
//! This slice only checks *reachability* — library load + symbol resolve. It
//! deliberately does **not** query the QNN device / backend identity (HTP vs
//! CPU/GPU backend), enumerate providers, or read any struct: those require a
//! QNN entry-point call with a verified signature and struct layout that do not
//! exist without the SDK header (owner T11). The execution-device readout (ADR
//! case D — proving work ran on the Hexagon NPU, not a CPU/GPU fallback) is part
//! of the graph-construction re-issue wave.

use vokra_core::{Result, VokraError};

/// What [`vokra_qnn_probe`] discovered about the host's QNN runtime.
///
/// Minimal by design: without the SDK the probe can only confirm that the
/// runtime library loaded and a representative interface symbol resolved. Device
/// identity, provider list and Hexagon-NPU generation are answered by the
/// graph-construction re-issue wave once the SDK is present (owner T11).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QnnCapabilities {
    /// The QNN library name (a candidate, or the `VOKRA_QNN_LIB` override) that
    /// actually loaded — e.g. `"libQnnHtp.so"`.
    pub library_name: String,
}

impl QnnCapabilities {
    /// Human-readable one-line summary, e.g.
    /// `"QNN: libQnnHtp.so loaded, interface entry resolved"`.
    #[must_use]
    pub fn summary(&self) -> String {
        format!(
            "QNN: {} loaded, interface entry resolved (scaffold reachability check only)",
            self.library_name
        )
    }
}

/// Detects the QNN HTP (Hexagon) runtime on Android / Linux / Windows.
///
/// # Errors
///
/// Returns [`VokraError::BackendUnavailable`] when the QNN runtime library is
/// not present, or when it loads but the representative interface entry symbol
/// is absent (an incompatible / too-old runtime). This is the deliberate
/// *explicit error* of FR-EX-08 / NFR-RL-06: the QNN backend never silently
/// degrades to CPU.
#[cfg(all(
    feature = "qnn",
    any(target_os = "android", target_os = "linux", target_os = "windows")
))]
pub fn vokra_qnn_probe() -> Result<QnnCapabilities> {
    use crate::sys;

    let lib = sys::load_qnn_library()?;

    // Reachability check: the interface entry symbol must resolve. A library
    // that loads but lacks it is an incompatible/too-old runtime — an explicit
    // BackendUnavailable, never a silent CPU fall back (NFR-RL-06). The symbol
    // name is UNVERIFIED (owner T11); on a real QNN host where it differs, this
    // is where the corrected name is exercised.
    if !lib.has_symbol(sys::QNN_INTERFACE_ENTRY_SYMBOL) {
        return Err(VokraError::BackendUnavailable(format!(
            "QNN library `{}` loaded but the interface entry symbol did not resolve — an \
             incompatible or too-old Qualcomm AI Engine Direct runtime. Vokra does not silently \
             fall back to the CPU (FR-EX-08).",
            lib.name()
        )));
    }

    Ok(QnnCapabilities {
        library_name: lib.name().to_owned(),
    })
}

/// Off-target / feature-off stub: the QNN backend is compiled out, so probing
/// always fails explicitly (FR-EX-08 / NFR-RL-06 — never a silent CPU fall
/// back). QNN is **not** an Apple backend, so macOS / iOS always take this path.
///
/// # Errors
///
/// Always returns [`VokraError::BackendUnavailable`].
#[cfg(not(all(
    feature = "qnn",
    any(target_os = "android", target_os = "linux", target_os = "windows")
)))]
pub fn vokra_qnn_probe() -> Result<QnnCapabilities> {
    Err(VokraError::BackendUnavailable(
        "QNN backend is not compiled for this target/feature combination (needs the `qnn` feature \
         on Android / Linux / Windows — QNN is the Qualcomm Hexagon NPU delegate, not an Apple \
         backend, and is NOT NNAPI)"
            .to_owned(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(all(
        feature = "qnn",
        any(target_os = "android", target_os = "linux", target_os = "windows")
    ))]
    #[test]
    fn probe_is_ok_or_backend_unavailable_never_a_fabricated_pass() {
        // Runs on a QNN-capable host (owner Snapdragon device / a runner with
        // the SDK installed). A host with no QNN runtime (every CI runner today,
        // the authoring Mac) is a legitimate BackendUnavailable — a probe-gated
        // skip, never a fabricated pass (FR-EX-08).
        match vokra_qnn_probe() {
            Ok(caps) => {
                assert!(
                    !caps.library_name.is_empty(),
                    "Ok(_) must carry the loaded library name"
                );
                eprintln!("vokra_qnn_probe: {}", caps.summary());
            }
            Err(VokraError::BackendUnavailable(msg)) => {
                eprintln!("no QNN runtime on this host; skipping ({msg})");
            }
            Err(other) => panic!("probe must be Ok or BackendUnavailable, got {other:?}"),
        }
    }

    #[cfg(not(all(
        feature = "qnn",
        any(target_os = "android", target_os = "linux", target_os = "windows")
    )))]
    #[test]
    fn probe_is_explicit_error_off_target() {
        assert!(matches!(
            vokra_qnn_probe(),
            Err(VokraError::BackendUnavailable(_))
        ));
    }
}
