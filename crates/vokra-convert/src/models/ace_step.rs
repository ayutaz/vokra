#![allow(clippy::doc_lazy_continuation)]
//! **ACE-Step 1.5** (`ACE-Step/Ace-Step1.5`, license pending audit):
//! inspection-only boundary for the ACE-Step 1.5 composite bundle.
//!
//! ACE-Step 1.5 flagship music generation = **multi-component bundle**:
//! - `Qwen3-Embedding-0.6B/model.safetensors` (text embedding)
//! - `acestep-5Hz-lm-1.7B/model.safetensors` (music-token AR LM)
//! - `acestep-v15-turbo/model.safetensors` (turbo diffusion head)
//! - `vae/diffusion_pytorch_model.safetensors` (VAE decoder)
//! - `silence_latent.pt` (silence latent bootstrap)
//!
//! The public converter stays disabled until VAST evidence authenticates every
//! component, companion, source dependency, and license independently.
//!
//! License identity is not stamped from this module. The HF bundle, source,
//! and dependency licenses must be recorded from their primary files by the
//! VAST inspector before any publication decision.
//!
//! # Scale — vast.ai handoff (~9.6 GB bundle)
//!
//! Above M1 iMac safe threshold per memory
//! `[[feedback-large-models-on-vast-ai]]`. The VAST wave inventories the
//! composite bundle and deliberately performs no conversion.

use std::path::Path;

use crate::ConvertError;

#[allow(dead_code)] // Retained as inspection-only dispatch metadata until binding is authenticated.
pub const ARCH: &str = "ace_step";
#[allow(dead_code)] // Retained as inspection-only model metadata until binding is authenticated.
pub const NAME: &str = "ace-step-1.5";
#[allow(dead_code)] // Retained as inspection-only model metadata until binding is authenticated.
pub const CATEGORY: &str = "music";
pub const UPSTREAM_HF: &str = "ACE-Step/Ace-Step1.5";
pub const UPSTREAM_HF_REVISION: &str = "19671f406d603126926c1b7e2adc169acbcade22";
pub const OFFICIAL_SOURCE_REPOSITORY: &str = "https://github.com/ace-step/ACE-Step-1.5";
pub const OFFICIAL_SOURCE_REVISION: &str = "7202bc354d7fc31d1c0e5a90b0b49fb610e52362";

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct AceStepReport {
    pub read: usize,
    pub written: usize,
    pub skipped_non_float: usize,
    pub bf16_passthrough: usize,
}

pub fn convert_ace_step_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<AceStepReport, ConvertError> {
    let _ = (input, output, license);
    Err(ConvertError::Usage(format!(
        "ACE-Step 1.5 conversion is INSPECTION_ONLY until the composite HF bundle, all component tensors, official source, dependency lock, and license evidence are reviewed (HF {UPSTREAM_HF}@{UPSTREAM_HF_REVISION}; source {OFFICIAL_SOURCE_REPOSITORY}@{OFFICIAL_SOURCE_REVISION})"
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
            "vokra-convert-ace-step-{tag}-{}-{n}",
            std::process::id()
        ));
        p
    }

    #[test]
    fn public_conversion_is_explicitly_inspection_only() {
        let inp = tmp_path("f32-in");
        let outp = tmp_path("f32-out");
        let error = convert_ace_step_file(&inp, &outp, Some("mit"))
            .expect_err("unreviewed ACE-Step bundle must refuse conversion");
        assert!(error.to_string().contains("INSPECTION_ONLY"));
        assert!(!outp.exists());
        let _ = std::fs::remove_file(&inp);
        let _ = std::fs::remove_file(&outp);
    }
}
