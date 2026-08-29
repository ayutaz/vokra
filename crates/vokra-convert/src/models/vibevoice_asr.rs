#![allow(clippy::doc_lazy_continuation)]
//! **VibeVoice-ASR** (`microsoft/VibeVoice-ASR`, **MIT**):
//! safetensors → GGUF conversion (Wave 6 residual, 2026-08-01).
//!
//! Microsoft VibeVoice sibling with **ASR head** (VibeVoiceForASRTraining
//! per config.json). Distinct arch tag `vibevoice_asr` vs sibling TTS
//! `vibevoice` (already published as vokra/vibevoice / vokra/vibevoice-
//! realtime-0.5b) — silently sharing arch would misroute runtime
//! dispatch to a wrong-head forward (TTS head expects encoder ID stream,
//! ASR head expects raw audio → text tokens).
//!
//! # Scale — vast.ai handoff (~17.3 GB, 8-shard safetensors)
//!
//! Full VibeVoice-ASR 9B + ASR head. Above M1 iMac safe threshold per
//! memory `[[feedback-large-models-on-vast-ai]]`. The eight shards remain
//! separate; a VAST inspector records their manifests without merging them.
//! This converter is intentionally disabled until that review lands.

use std::path::Path;

use crate::ConvertError;

#[allow(dead_code)] // Retained as inspection-only dispatch metadata until binding is authenticated.
pub const ARCH: &str = "vibevoice_asr";
#[allow(dead_code)] // Retained as inspection-only model metadata until binding is authenticated.
pub const NAME: &str = "vibevoice-asr";
#[allow(dead_code)] // Retained as inspection-only model metadata until binding is authenticated.
pub const CATEGORY: &str = "asr";
pub const UPSTREAM_HF: &str = "microsoft/VibeVoice-ASR";
pub const UPSTREAM_HF_REVISION: &str = "d0c9efdb8d614685062c04425d91e01b6f37d944";
pub const OFFICIAL_SOURCE_REPOSITORY: &str = "https://github.com/microsoft/VibeVoice";
pub const OFFICIAL_SOURCE_REVISION: &str = "94da20d98b2fa7688e9cbfaf7692ddb4954f7600";
#[allow(dead_code)] // Retained as inspection-only license metadata until binding is authenticated.
pub const DEFAULT_LICENSE_SPDX: &str = "mit";

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct VibeVoiceAsrReport {
    pub read: usize,
    pub written: usize,
    pub skipped_non_float: usize,
    pub bf16_passthrough: usize,
}

pub fn convert_vibevoice_asr_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<VibeVoiceAsrReport, ConvertError> {
    let _ = (input, output, license);
    Err(ConvertError::Usage(format!(
        "VibeVoice-ASR conversion is INSPECTION_ONLY until all 8 shards, processor/tokenizer companions, config, and official source revision are reviewed (HF {UPSTREAM_HF}@{UPSTREAM_HF_REVISION}; source {OFFICIAL_SOURCE_REPOSITORY}@{OFFICIAL_SOURCE_REVISION})"
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    fn tmp_path(tag: &str) -> PathBuf {
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let n = SEQ.fetch_add(1, Ordering::Relaxed);
        let mut p = std::env::temp_dir();
        p.push(format!(
            "vokra-convert-vibevoice-asr-{tag}-{}-{n}",
            std::process::id()
        ));
        p
    }

    #[test]
    fn public_conversion_is_explicitly_inspection_only() {
        let inp = tmp_path("f32-in");
        let outp = tmp_path("f32-out");
        let error = convert_vibevoice_asr_file(&inp, &outp, Some(DEFAULT_LICENSE_SPDX))
            .expect_err("unreviewed VibeVoice-ASR must refuse conversion");
        assert!(error.to_string().contains("INSPECTION_ONLY"));
        assert!(!outp.exists());
        let _ = std::fs::remove_file(&inp);
        let _ = std::fs::remove_file(&outp);
    }
}
