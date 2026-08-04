#![allow(clippy::doc_lazy_continuation)]
//! **w2v-BERT 2.0** (`facebook/w2v-bert-2.0`, **MIT**): safetensors →
//! GGUF conversion (2026-08-04 hf-audio-gap-comprehensive-2026-07-30
//! §3 SSL foundational encoder residual).
//!
//! w2v-BERT 2.0 (Chung et al., 2021 arXiv:2108.06209 "w2v-BERT: Combining
//! Contrastive Learning and Masked Language Modeling for Self-Supervised
//! Speech Pre-Training", released as part of the Seamless-M4T v2 stack
//! per Barrault et al. 2023 arXiv:2312.05187) = ~580M-parameter
//! self-supervised speech encoder combining a Conformer encoder body
//! with (a) a wav2vec 2.0-style contrastive-quantiser branch and (b) a
//! BERT-style masked-language-modeling branch over the shared
//! representation. **Distinct from siblings
//! [`crate::models::hubert_large_ls960`] and
//! [`crate::models::wav2vec2_ctc`]**: HuBERT uses BERT-style
//! masked-feature-prediction over k-means-clustered features, wav2vec 2.0
//! uses a contrastive convnet objective with Gumbel-softmax quantised
//! negatives, and w2v-BERT combines both branches over a Conformer
//! (not vanilla Transformer) encoder body. The three share the general
//! feature-extractor + Transformer-body + SSL-head shape but the
//! pretraining objectives and encoder topology differ enough that
//! silently sharing an arch tag would mis-route the runtime dispatch —
//! FR-EX-08 (no silent op-shape misroute) requires the distinct
//! `w2v-bert-2` arch tag.
//!
//! # Standalone vs internal-subgraph identity
//!
//! Prior to this converter, w2v-BERT 2.0 tensors were present in the
//! Vokra converter tree only as an INTERNAL subgraph inside two
//! composite models: (a) `vieneu-tts` (VieNeu TTS uses w2v-BERT as its
//! speaker encoder), (b) `seamless-m4t-v2-large` (Seamless uses
//! w2v-BERT as its speech encoder). Neither exposes w2v-BERT for
//! standalone use — a downstream fine-tune workflow (SSL feature
//! extraction for a new ASR / speaker / VAD head across 143+
//! languages) requires a standalone `ModelKind::W2vBert2` binder that
//! packs the encoder alone, without composite scaffolding. This lands
//! precisely that standalone path: a downstream who trains a
//! per-language ASR head on top of w2v-BERT features can now bind the
//! shared encoder from a single GGUF, without having to strip it out
//! of the Seamless-M4T composite.
//!
//! # License posture — MIT (**Permissive**)
//!
//! `facebook/w2v-bert-2.0` HF `cardData.license = mit` (primary
//! source: HF API `https://huggingface.co/api/models/facebook/w2v-bert-2.0`
//! → `{"license":"mit","cardData":{"license":"mit"}}`, CC-verified
//! 2026-08-04). Same permissive posture as sibling Whisper family +
//! piper-plus + Silero + CAM++ + Moonshine + HuBERT-Large-LS960 + the
//! wav2vec 2.0 family (all apache-2.0 or MIT). No NC / SA gate; publish
//! workflow uses the default T1 (Commercial) path with
//! [`vokra_core::LicenseClass::Permissive`] default.
//!
//! # Scale — vast.ai handoff (~2.16 GB single-file safetensors)
//!
//! w2v-BERT 2.0 ships one `model.safetensors` file at **2,322,063,736
//! bytes** (~2.16 GB — HF `/api/models/facebook/w2v-bert-2.0/tree/main`
//! primary source, CC-verified 2026-08-04). This **exceeds the 2 GB
//! local-convert owner threshold** for the CC publish workflow (per
//! per-item owner directive "ローカルで対応するのは2GB以下 — 上回るモデル
//! はこの workflow では扱わない") = **vast.ai handoff required** per
//! memory `[[feedback-large-models-on-vast-ai]]` and
//! `docs/handoff/vast-ai-large-model-publish.md`. This converter +
//! §3.1 audit row + `signoff_match.py` entry land today so the future
//! vast.ai owner publish is one command away
//! (`bash scripts/publish/publish-one.sh w2v-bert-2-0 --push`), but
//! the actual convert + upload happens on vast.ai (not M1 iMac).
//!
//! # BF16 pass-through skeleton
//!
//! Every F32 / F16 / BF16 tensor passes through verbatim under its
//! upstream safetensors name (the standard skeleton pattern — mirror
//! of sibling `hubert_large_ls960.rs` / `moonshine_base.rs` /
//! `musicgen_small.rs` / `openwakeword.rs`). No convert-time widening;
//! runtime widens BF16 → f32 losslessly via the single choke point
//! `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16` (BF16 is the
//! top 16 bits of an f32 — `bits << 16` is exact). Upstream ships F32
//! (per HF `safetensors.parameters.F32 = 580_493_120`), so the BF16
//! path is exercised only when a downstream re-quantises the
//! checkpoint before conversion.
//!
//! # Runtime binder — deferred to owner sign-off
//!
//! Runtime forward (Conformer encoder body + SSL feature extraction)
//! is deferred to a follow-up (`docs/license-audit.md` §3.1 sign-off).
//! Consumers needing a foundational SSL encoder today should use one
//! of the already-bound siblings (wav2vec2_ctc / hubert_large_ls960 /
//! data2vec-audio-base) under the same Permissive posture; the
//! w2v-BERT 2.0 forward is a Conformer variant (not a vanilla
//! Transformer encoder like wav2vec2 / HuBERT / data2vec-audio), so
//! the runtime binder cannot silently share their loaders — the
//! future native forward will reuse the shared `vokra_ops::conformer`
//! primitive (SoTA plan Phase 2 landed op, no per-model op
//! duplication).

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

pub const ARCH: &str = "w2v-bert-2";
pub const NAME: &str = "w2v-bert-2.0";
pub const CATEGORY: &str = "asr";
pub const UPSTREAM_HF: &str = "facebook/w2v-bert-2.0";
pub const DEFAULT_LICENSE_SPDX: &str = "mit";

const UPSTREAM_SOURCE: &str =
    "facebook/w2v-bert-2.0 (Meta w2v-BERT 2.0 ~580M self-supervised speech encoder, MIT)";

const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct W2vBert2Report {
    pub read: usize,
    pub written: usize,
    pub skipped_non_float: usize,
    pub bf16_passthrough: usize,
}

pub fn convert_w2v_bert_2_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<W2vBert2Report, ConvertError> {
    let bytes = std::fs::read(input)?;
    let st = SafetensorsFile::parse(bytes)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    b.add_string(KEY_MODEL_CATEGORY, CATEGORY);

    let (spdx, class) = match license {
        Some(s) if !s.is_empty() => (s.to_owned(), LicenseClass::from_license_str(s)),
        _ => (DEFAULT_LICENSE_SPDX.to_owned(), LicenseClass::Permissive),
    };
    vokra_core::stamp_provenance(&mut b, class, &spdx, Some(NAME), Some(UPSTREAM_SOURCE));
    b.add_string(KEY_PROVENANCE_UPSTREAM_HF, UPSTREAM_HF);

    let mut report = W2vBert2Report::default();
    for t in st.tensors() {
        report.read += 1;
        match t.dtype {
            GgmlType::F32 | GgmlType::F16 | GgmlType::BF16 => {
                b.add_tensor(
                    &t.name,
                    t.dtype,
                    t.shape.clone(),
                    st.tensor_bytes(t).to_vec(),
                )
                .map_err(|e| ConvertError::Gguf(e.to_string()))?;
                report.written += 1;
                if t.dtype == GgmlType::BF16 {
                    report.bf16_passthrough += 1;
                }
            }
            _ => {
                report.skipped_non_float += 1;
            }
        }
    }

    let out_bytes = b
        .to_bytes()
        .map_err(|e| ConvertError::Gguf(e.to_string()))?;
    std::fs::write(output, out_bytes)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};
    use vokra_core::gguf::GgufFile;

    fn tmp_path(tag: &str) -> PathBuf {
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let n = SEQ.fetch_add(1, Ordering::Relaxed);
        let mut p = std::env::temp_dir();
        p.push(format!(
            "vokra-convert-w2v-bert-2-{tag}-{}-{n}",
            std::process::id()
        ));
        p
    }

    fn safetensors_one(name: &str, dtype: &str, shape: &[u64], payload: &[u8]) -> Vec<u8> {
        let shape_str = shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let header = format!(
            r#"{{"{name}":{{"dtype":"{dtype}","shape":[{shape_str}],"data_offsets":[0,{}]}}}}"#,
            payload.len()
        );
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(payload);
        out
    }

    #[test]
    fn f32_tensor_passes_through_and_default_license_is_permissive() {
        let inp = tmp_path("f32-in");
        let outp = tmp_path("f32-out");
        let payload: Vec<u8> = [1.0_f32, 2.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let st = safetensors_one("encoder.layer0.attn.q_proj", "F32", &[1, 2], &payload);
        std::fs::write(&inp, &st).unwrap();
        let r = convert_w2v_bert_2_file(&inp, &outp, None).unwrap();
        assert_eq!(r.read, 1);
        assert_eq!(r.written, 1);
        assert_eq!(r.bf16_passthrough, 0);

        let g = GgufFile::open(&outp).unwrap();
        let read_str = |key: &str| -> String {
            g.get(key)
                .and_then(|v| v.as_str())
                .unwrap_or_else(|| panic!("{key}: missing"))
                .to_owned()
        };
        assert_eq!(read_str(chunks::KEY_MODEL_ARCH), ARCH);
        assert_eq!(read_str(chunks::KEY_MODEL_NAME), NAME);
        assert_eq!(read_str(KEY_MODEL_CATEGORY), CATEGORY);
        assert_eq!(read_str(KEY_PROVENANCE_UPSTREAM_HF), UPSTREAM_HF);
        // Default license is mit (Permissive) — sibling to HuBERT
        // (apache-2.0) / Moonshine (MIT) / wav2vec2 (apache-2.0) with
        // the same T1 Permissive tier.
        assert_eq!(read_str("vokra.provenance.license"), DEFAULT_LICENSE_SPDX);
        let _ = std::fs::remove_file(&inp);
        let _ = std::fs::remove_file(&outp);
    }

    #[test]
    fn bf16_tensor_passes_through_verbatim() {
        let inp = tmp_path("bf16-in");
        let outp = tmp_path("bf16-out");
        let payload: Vec<u8> = [1.0_f32, 2.0]
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        let st = safetensors_one("encoder.layer0.attn.k_proj", "BF16", &[1, 2], &payload);
        std::fs::write(&inp, &st).unwrap();
        let r = convert_w2v_bert_2_file(&inp, &outp, None).unwrap();
        assert_eq!(r.bf16_passthrough, 1);
        assert_eq!(r.written, 1);
        let _ = std::fs::remove_file(&inp);
        let _ = std::fs::remove_file(&outp);
    }

    #[test]
    fn license_override_swaps_stamp() {
        let inp = tmp_path("lic-in");
        let outp = tmp_path("lic-out");
        let payload: Vec<u8> = [1.0_f32].iter().flat_map(|v| v.to_le_bytes()).collect();
        let st = safetensors_one("x", "F32", &[1], &payload);
        std::fs::write(&inp, &st).unwrap();
        convert_w2v_bert_2_file(&inp, &outp, Some("apache-2.0")).unwrap();
        let g = GgufFile::open(&outp).unwrap();
        assert_eq!(
            g.get("vokra.provenance.license").and_then(|v| v.as_str()),
            Some("apache-2.0")
        );
        let _ = std::fs::remove_file(&inp);
        let _ = std::fs::remove_file(&outp);
    }
}
