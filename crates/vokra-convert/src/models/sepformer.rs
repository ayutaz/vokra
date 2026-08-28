//! **SepFormer** (SpeechBrain sepformer family, apache-2.0):
//! safetensors → GGUF conversion for the Transformer-based dual-path
//! source-separation / enhancement family.
//!
//! # Family (7 variants share this converter)
//!
//! - **`speechbrain/sepformer-wsj02mix`** — 2-speaker source
//!   separation trained on WSJ0-2mix.
//!   [`SepformerVariant::Wsj02mix`]. `vokra.model.category = "separation"`.
//!   `vokra.sepformer.n_out = 2`.
//! - **`speechbrain/sepformer-libri2mix`** — 2-speaker source
//!   separation trained on LibriMix (LibriSpeech-derived, CC-BY-4.0
//!   corpus). Same 2-speaker head as the WSJ0-2mix sibling — the two
//!   differ only in the corpus the model was fit to.
//!   [`SepformerVariant::Libri2Mix`]. `vokra.model.category = "separation"`.
//!   `vokra.sepformer.n_out = 2`.
//! - **`speechbrain/sepformer-libri3mix`** — **3-speaker** source
//!   separation trained on LibriMix (Libri3Mix, cocktail-party setup).
//!   Same SepFormer topology as the sibling `libri2mix` variant with
//!   the same LibriSpeech-derived training corpus — only the masker
//!   output head branches into **3 parallel speaker streams instead of
//!   2**. [`SepformerVariant::Libri3Mix`].
//!   `vokra.model.category = "separation"`. `vokra.sepformer.n_out = 3`.
//! - **`speechbrain/sepformer-wham16k-enhancement`** — single-speaker
//!   speech enhancement (WHAM! 16 kHz).
//!   [`SepformerVariant::Wham16kEnhancement`]. `vokra.model.category = "enhancement"`.
//!   `vokra.sepformer.n_out = 1`.
//! - **`speechbrain/sepformer-whamr16k`** — 2-speaker source
//!   separation with noise and reverberation (WHAMR! 16 kHz).
//!   [`SepformerVariant::Whamr16k`]. `vokra.model.category = "separation"`.
//!   `vokra.sepformer.n_out = 2`.
//! - **`speechbrain/sepformer-whamr`** — 2-speaker source separation
//!   with noise and reverberation
//!   (WHAMR! **8 kHz** — the base-sample-rate sibling of the 16 kHz
//!   variant above; same reverberant conditioning + masker head, only
//!   the sample rate differs). [`SepformerVariant::Whamr8k`].
//!   `vokra.model.category = "separation"`. `vokra.sepformer.n_out = 2`.
//! - **`speechbrain/sepformer-dns4-16k-enhancement`** — single-speaker
//!   speech enhancement (Microsoft DNS-4 challenge corpus, 16 kHz).
//!   Distinct training corpus from the WHAM! enhancement and WHAMR!
//!   separation siblings (Microsoft Deep Noise Suppression Challenge 4 rather
//!   than WSJ0-derived WHAM! / WHAMR!) with the same SepFormer
//!   topology and single-output masker head.
//!   [`SepformerVariant::Dns4Enhancement`].
//!   `vokra.model.category = "enhancement"`. `vokra.sepformer.n_out = 1`.
//!
//! All seven variants share the same SepFormer architecture (encoder +
//! dual-path Transformer masker + decoder — Subakan et al. 2021 /
//! Chen et al. 2022 WHAMR extension); only the training data + sample
//! rate + head count differ. One `sepformer.rs` converter therefore
//! covers all seven — the caller passes a [`SepformerVariant`] and the
//! emitted GGUF's `vokra.model.name`, `vokra.provenance.upstream_hf`,
//! `vokra.model.category`, `vokra.sepformer.variant`, and
//! `vokra.sepformer.n_out` stamps reflect the specific release, while
//! `vokra.model.arch = "sepformer"` is shared across all seven (silently
//! sharing would misroute a downstream loader if the family ever
//! diverged in tensor topology — today they do not, but the shared
//! arch tag + explicit variant tag + explicit output-count tag is the
//! honest posture).
//!
//! # Provenance
//!
//! - **License (SPDX)**: `apache-2.0` for all seven variants per HF
//!   model-card `cardData.license` (SpeechBrain ships Apache-2.0
//!   end-to-end — `github.com/speechbrain/speechbrain/blob/develop/LICENSE`).
//!   Class = `Permissive`.
//! - **Attribution**: none required by license.
//!
//! # BF16 pass-through (mirror of qwen3_tts / vibevoice / voxcpm2 / wespeaker)
//!
//! F32 / F16 / BF16 float tensors ride the verbatim pass-through arm.
//! BF16 stays GGUF type 30 (`GgmlType::BF16`); runtime widens
//! BF16 → f32 losslessly at load via
//! `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16`.
//!
//! # Tensor naming contract
//!
//! GGUF tensor names are the **upstream safetensors names verbatim**.
//! The native runtime binds this exact upstream naming scheme and executes
//! the complete dual-path Transformer. Independent official parity fixtures
//! cover the encoder and final enhanced waveform.
//!
//! # No ONNX (permanent)
//!
//! SpeechBrain ships PyTorch checkpoints (safetensors); this converter
//! **never** touches ONNX (FR-LD-05).

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` for every SepFormer variant — the topology is
/// shared across `wsj02mix` / `wham16k-enhancement` / `whamr16k`.
pub const ARCH: &str = "sepformer";

/// Default upstream weight license (SPDX) — apache-2.0 for every
/// SepFormer variant per HF model-card `cardData.license`.
pub const DEFAULT_LICENSE_SPDX: &str = "apache-2.0";

// Raw string keys not covered by `crate::gguf::chunks` — kept as
// converter-side constants.
const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";
/// `vokra.sepformer.variant`: `"wsj02mix"` / `"libri2mix"` /
/// `"libri3mix"` / `"wham16k-enhancement"` / `"whamr16k"` /
/// `"whamr8k"` / `"dns4-16k-enhancement"`. Consumers pick a specific
/// separation / enhancement head without parsing free-text
/// `vokra.model.name`.
pub const KEY_SEPFORMER_VARIANT: &str = "vokra.sepformer.variant";

/// `vokra.sepformer.n_out`: the **number of parallel output streams** the
/// masker head emits — the shape-load axis a downstream binder needs to
/// pre-allocate the output tensor bank.
///
/// - `1` for the single-stream enhancement variants
///   (`wham16k-enhancement`, `dns4-16k-enhancement`).
/// - `2` for the standard 2-speaker separation task
///   (`wsj02mix`, `libri2mix`, `whamr16k`, `whamr8k`).
/// - `3` for the LibriMix 3-speaker cocktail-party separation head
///   (`libri3mix`).
///
/// Before this key was added, `n_out` was implicit — a downstream loader
/// had to hard-code the mapping from `vokra.sepformer.variant` (or worse,
/// from a free-text `vokra.model.name`) to an output-stream count. Every
/// GGUF the converter now emits stamps this explicitly so a `sepformer-*`
/// variant added later (e.g. hypothetical `sepformer-libri5mix`) does not
/// silently inherit a `libri3mix` binder's `n_out = 3` when the correct
/// axis is different.
pub const KEY_SEPFORMER_N_OUT: &str = "vokra.sepformer.n_out";

/// Which SepFormer release the caller is converting.
///
/// All seven variants share the [`ARCH`] tag `sepformer`; the
/// category, upstream HF slug, sample rate, variant tag, and
/// output-stream count (`n_out`) differ.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SepformerVariant {
    /// `speechbrain/sepformer-wsj02mix`: 2-speaker source separation
    /// on WSJ0-2mix. Category = `separation`,
    /// `vokra.sepformer.variant = "wsj02mix"`, `n_out = 2`.
    Wsj02mix,
    /// `speechbrain/sepformer-libri2mix`: 2-speaker source separation
    /// on LibriMix (LibriSpeech-derived, CC-BY-4.0 corpus). Same
    /// 2-speaker head as [`Self::Wsj02mix`] — the two differ only in
    /// the corpus. Category = `separation`,
    /// `vokra.sepformer.variant = "libri2mix"`, `n_out = 2`.
    Libri2Mix,
    /// `speechbrain/sepformer-libri3mix`: **3-speaker** source
    /// separation on LibriMix (Libri3Mix cocktail-party mixture set).
    /// Same SepFormer topology (encoder + dual-path Transformer masker
    /// + decoder — Subakan et al. 2021) as the sibling [`Self::Libri2Mix`]
    /// with the same LibriSpeech-derived training corpus; the sole
    /// difference is the masker output head branches into **3 parallel
    /// speaker streams instead of 2**. Category = `separation`,
    /// `vokra.sepformer.variant = "libri3mix"`, `n_out = 3`. The
    /// distinct enum arm ensures the artifact does NOT silently
    /// inherit the 2-speaker sibling's `vokra.provenance.upstream_hf`
    /// = wrong CDN attribution + `vokra.sepformer.n_out = 2` = wrong
    /// binder output-stream axis.
    Libri3Mix,
    /// `speechbrain/sepformer-wham16k-enhancement`: single-speaker
    /// speech enhancement (WHAM! 16 kHz). Category = `enhancement`,
    /// `vokra.sepformer.variant = "wham16k-enhancement"`, `n_out = 1`.
    Wham16kEnhancement,
    /// `speechbrain/sepformer-whamr16k`: 2-speaker source separation
    /// with noise and reverberation (WHAMR! 16 kHz). Category =
    /// `separation`, `vokra.sepformer.variant = "whamr16k"`, `n_out = 2`.
    Whamr16k,
    /// `speechbrain/sepformer-whamr`: 2-speaker source separation with
    /// noise and reverberation
    /// (WHAMR! **8 kHz** — the base-sample-rate sibling of
    /// [`Self::Whamr16k`]; same reverberant conditioning + 2-speaker masker
    /// head, only the sample rate differs). Category = `separation`,
    /// `vokra.sepformer.variant = "whamr8k"`, `n_out = 2`. The distinct
    /// enum arm ensures the artifact does NOT silently inherit the
    /// 16 kHz sibling's `vokra.provenance.upstream_hf` = wrong CDN
    /// attribution.
    Whamr8k,
    /// `speechbrain/sepformer-dns4-16k-enhancement`: single-speaker
    /// speech enhancement trained on the **Microsoft DNS-4** (Deep
    /// Noise Suppression Challenge 4) corpus at 16 kHz. Same
    /// SepFormer topology (encoder + dual-path Transformer masker +
    /// decoder — Subakan et al. 2021) and same single-output masker
    /// head as the WHAM! enhancement sibling; the sole difference is the training corpus
    /// (Microsoft DNS-4 vs WSJ0-derived WHAM! / WHAMR!). Category =
    /// `enhancement`, `vokra.sepformer.variant = "dns4-16k-enhancement"`,
    /// `n_out = 1`. The distinct enum arm ensures the artifact does
    /// NOT silently inherit any sibling variant's
    /// `vokra.provenance.upstream_hf` = wrong CDN attribution (the
    /// single-stream enhancement siblings share `n_out = 1`, so silent misrouting
    /// would not fail loudly at the binder — the distinct provenance
    /// stamp is the honest posture).
    Dns4Enhancement,
}

impl SepformerVariant {
    /// The `vokra.model.name` string for this release.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Wsj02mix => "sepformer-wsj02mix",
            Self::Libri2Mix => "sepformer-libri2mix",
            Self::Libri3Mix => "sepformer-libri3mix",
            Self::Wham16kEnhancement => "sepformer-wham16k-enhancement",
            Self::Whamr16k => "sepformer-whamr16k",
            Self::Whamr8k => "sepformer-whamr",
            Self::Dns4Enhancement => "sepformer-dns4-16k-enhancement",
        }
    }

    /// The `vokra.provenance.upstream_hf` slug (`org/name`).
    pub const fn upstream_hf(self) -> &'static str {
        match self {
            Self::Wsj02mix => "speechbrain/sepformer-wsj02mix",
            Self::Libri2Mix => "speechbrain/sepformer-libri2mix",
            Self::Libri3Mix => "speechbrain/sepformer-libri3mix",
            Self::Wham16kEnhancement => "speechbrain/sepformer-wham16k-enhancement",
            Self::Whamr16k => "speechbrain/sepformer-whamr16k",
            Self::Whamr8k => "speechbrain/sepformer-whamr",
            Self::Dns4Enhancement => "speechbrain/sepformer-dns4-16k-enhancement",
        }
    }

    /// The `vokra.sepformer.variant` tag.
    pub const fn tag(self) -> &'static str {
        match self {
            Self::Wsj02mix => "wsj02mix",
            Self::Libri2Mix => "libri2mix",
            Self::Libri3Mix => "libri3mix",
            Self::Wham16kEnhancement => "wham16k-enhancement",
            Self::Whamr16k => "whamr16k",
            Self::Whamr8k => "whamr8k",
            Self::Dns4Enhancement => "dns4-16k-enhancement",
        }
    }

    /// The `vokra.model.category` value. Wsj02mix, LibriMix, and both
    /// WHAMR variants are **source-separation** tasks (N speakers out
    /// of 1 mixture — Libri2Mix / Wsj02mix differ only in training
    /// corpus; Libri3Mix uses the same LibriMix corpus family but a
    /// 3-stream output head instead of 2); WHAM! and Microsoft DNS-4
    /// are single-output **enhancement** tasks, so a downstream that speaks either API
    /// can pick the correct load path from this alone.
    pub const fn category(self) -> &'static str {
        match self {
            Self::Wsj02mix | Self::Libri2Mix | Self::Libri3Mix | Self::Whamr16k | Self::Whamr8k => {
                "separation"
            }
            Self::Wham16kEnhancement | Self::Dns4Enhancement => "enhancement",
        }
    }

    /// The number of parallel output streams the masker head emits —
    /// the shape-load axis a downstream binder needs to pre-allocate
    /// its output tensor bank. Stamped into
    /// [`KEY_SEPFORMER_N_OUT`] on the artifact.
    ///
    /// - `1` for the WHAM! and DNS4 single-stream enhancement variants.
    /// - `2` for the standard 2-speaker separation task
    ///   ([`Self::Wsj02mix`] / [`Self::Libri2Mix`] / both WHAMR variants).
    /// - `3` for the LibriMix 3-speaker cocktail-party head
    ///   ([`Self::Libri3Mix`]).
    pub const fn n_out(self) -> u32 {
        match self {
            Self::Wham16kEnhancement | Self::Dns4Enhancement => 1,
            Self::Wsj02mix | Self::Libri2Mix | Self::Whamr16k | Self::Whamr8k => 2,
            Self::Libri3Mix => 3,
        }
    }

    /// One-line free-text description used for the
    /// `vokra.provenance.source` stamp.
    pub const fn source_description(self) -> &'static str {
        match self {
            Self::Wsj02mix => {
                "speechbrain/sepformer-wsj02mix (SepFormer 2-speaker separation, WSJ0-2mix, apache-2.0)"
            }
            Self::Libri2Mix => {
                "speechbrain/sepformer-libri2mix (SepFormer 2-speaker separation, LibriMix corpus CC-BY-4.0, apache-2.0)"
            }
            Self::Libri3Mix => {
                "speechbrain/sepformer-libri3mix (SepFormer 3-speaker cocktail-party separation, LibriMix corpus CC-BY-4.0, apache-2.0)"
            }
            Self::Wham16kEnhancement => {
                "speechbrain/sepformer-wham16k-enhancement (SepFormer speech enhancement, WHAM! 16 kHz, apache-2.0)"
            }
            Self::Whamr16k => {
                "speechbrain/sepformer-whamr16k (SepFormer 2-speaker separation with noise and reverberation, WHAMR! 16 kHz, apache-2.0)"
            }
            Self::Whamr8k => {
                "speechbrain/sepformer-whamr (SepFormer 2-speaker separation with noise and reverberation, WHAMR! 8 kHz, apache-2.0)"
            }
            Self::Dns4Enhancement => {
                "speechbrain/sepformer-dns4-16k-enhancement (SepFormer speech enhancement, Microsoft DNS-4 16 kHz, apache-2.0)"
            }
        }
    }
}

/// Outcome of a SepFormer conversion.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct SepformerReport {
    /// Total tensors surfaced by the safetensors reader.
    pub read: usize,
    /// Float tensors written verbatim (F32 / F16 / BF16).
    pub written: usize,
    /// Non-float tensors skipped (defensive counter).
    pub skipped_non_float: usize,
    /// BF16 tensors that landed on the pass-through arm.
    pub bf16_passthrough: usize,
    /// Which SepFormer variant was written.
    pub variant: Option<SepformerVariant>,
}

/// File-based SepFormer converter.
///
/// Reads `input` (upstream `speechbrain/sepformer-*` safetensors),
/// writes a Vokra GGUF to `output` carrying every F32 / F16 / BF16
/// tensor verbatim under its upstream name + the `vokra.model.*` +
/// `vokra.provenance.*` metadata chunks + the
/// `vokra.sepformer.variant` tag.
///
/// `variant` selects which SepFormer release the input came from —
/// the GGUF's `vokra.model.name`, `vokra.provenance.upstream_hf`,
/// `vokra.model.category`, and `vokra.sepformer.variant` stamps
/// reflect it.
///
/// `license` optionally overrides the default `apache-2.0` provenance
/// stamp (same override pattern as
/// `wespeaker::convert_wespeaker_file`).
///
/// # Errors
///
/// [`ConvertError::Io`] on read / write failure;
/// [`ConvertError::Parse`] on a malformed safetensors input;
/// [`ConvertError::Gguf`] if the GGUF serialization fails.
pub fn convert_sepformer_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
    variant: SepformerVariant,
) -> Result<SepformerReport, ConvertError> {
    let bytes = std::fs::read(input)?;
    let st = SafetensorsFile::parse(bytes)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, variant.name());
    b.add_string(KEY_MODEL_CATEGORY, variant.category());
    b.add_string(KEY_PROVENANCE_UPSTREAM_HF, variant.upstream_hf());
    b.add_string(KEY_SEPFORMER_VARIANT, variant.tag());
    // The number of parallel output streams the masker head emits —
    // stamped explicitly so a downstream binder does not have to
    // hard-code the variant → output-count mapping (a
    // hypothetical future `sepformer-libri5mix` would otherwise
    // silently inherit whichever mapping the loader guessed).
    b.add_u32(KEY_SEPFORMER_N_OUT, variant.n_out());

    let (spdx, class) = match license {
        Some(s) if !s.is_empty() => (s.to_owned(), LicenseClass::from_license_str(s)),
        _ => (DEFAULT_LICENSE_SPDX.to_owned(), LicenseClass::Permissive),
    };
    vokra_core::stamp_provenance(
        &mut b,
        class,
        &spdx,
        Some(variant.name()),
        Some(variant.source_description()),
    );

    let mut report = SepformerReport {
        variant: Some(variant),
        ..SepformerReport::default()
    };
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
    use vokra_core::gguf::GgufFile;

    fn safetensors_one_bf16(name: &str, shape: &[u64], bf16_bytes: &[u8]) -> Vec<u8> {
        let elems: u64 = shape.iter().product();
        let expected = elems as usize * 2;
        assert_eq!(bf16_bytes.len(), expected, "shape × 2 BF16");
        let shape_str = shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let header = format!(
            r#"{{"{name}":{{"dtype":"BF16","shape":[{shape_str}],"data_offsets":[0,{}]}}}}"#,
            bf16_bytes.len()
        );
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(bf16_bytes);
        out
    }

    fn write_temp(kind: &str, bytes: &[u8]) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "vokra-sepformer-{kind}-{}-{}.bin",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.subsec_nanos())
                .unwrap_or(0)
        ));
        std::fs::write(&p, bytes).expect("write temp file");
        p
    }

    /// BF16 round-trip + Wsj02mix-specific stamps (category =
    /// `separation`, not the sibling variants' `enhancement`).
    #[test]
    fn wsj02mix_bf16_round_trip_and_separation_category() {
        let values: [f32; 6] = [1.0, -2.5, 0.15625, 3.5, -0.5, 42.0];
        let bf16: Vec<u8> = values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        let input_bytes = safetensors_one_bf16("masker.dpt.0.weight", &[2, 3], &bf16);
        let input = write_temp("wsj02mix-in", &input_bytes);
        let output = write_temp("wsj02mix-out", &[]);

        let report = convert_sepformer_file(&input, &output, None, SepformerVariant::Wsj02mix)
            .expect("convert sepformer-wsj02mix");
        assert_eq!(report.written, 1);
        assert_eq!(report.bf16_passthrough, 1);
        assert_eq!(report.variant, Some(SepformerVariant::Wsj02mix));

        let out_bytes = std::fs::read(&output).expect("read output");
        let file = GgufFile::parse(out_bytes).expect("parse GGUF");
        let info = file
            .tensor_info("masker.dpt.0.weight")
            .expect("BF16 tensor present");
        assert_eq!(info.dtype, GgmlType::BF16);
        assert_eq!(file.tensor_bytes(info), bf16.as_slice());

        // Wsj02mix's category is `separation` — distinct from the
        // enhancement variants.
        assert_eq!(
            file.get(KEY_MODEL_CATEGORY).and_then(|v| v.as_str()),
            Some("separation")
        );
        assert_eq!(
            file.get(KEY_SEPFORMER_VARIANT).and_then(|v| v.as_str()),
            Some("wsj02mix")
        );
        assert_eq!(
            file.get(chunks::KEY_MODEL_NAME).and_then(|v| v.as_str()),
            Some("sepformer-wsj02mix")
        );
        assert_eq!(
            file.get(KEY_PROVENANCE_UPSTREAM_HF)
                .and_then(|v| v.as_str()),
            Some("speechbrain/sepformer-wsj02mix")
        );
        // Arch is shared across every variant.
        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
            Some(ARCH)
        );

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }

    /// Libri2Mix variant carries `separation` category (same
    /// 2-speaker head as Wsj02mix, differs only in training corpus)
    /// and its own model.name / upstream_hf / variant tag stamps.
    /// The distinct ModelKind + row exist so the artifact does NOT
    /// silently inherit the Wsj02mix sibling's provenance.
    #[test]
    fn libri2mix_carries_separation_category_and_distinct_stamps() {
        let values: [f32; 4] = [0.25, -0.125, 1.5, -3.0];
        let bf16: Vec<u8> = values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        let input_bytes = safetensors_one_bf16("masker.dpt.0.weight", &[2, 2], &bf16);
        let input = write_temp("libri2mix-in", &input_bytes);
        let output = write_temp("libri2mix-out", &[]);

        let report = convert_sepformer_file(&input, &output, None, SepformerVariant::Libri2Mix)
            .expect("convert sepformer-libri2mix");
        assert_eq!(report.written, 1);
        assert_eq!(report.bf16_passthrough, 1);
        assert_eq!(report.variant, Some(SepformerVariant::Libri2Mix));

        let out_bytes = std::fs::read(&output).expect("read output");
        let file = GgufFile::parse(out_bytes).expect("parse GGUF");
        // Libri2Mix is a separation task (2-speaker head), NOT an
        // enhancement task — a downstream that dispatches on
        // vokra.model.category picks the correct load path.
        assert_eq!(
            file.get(KEY_MODEL_CATEGORY).and_then(|v| v.as_str()),
            Some("separation"),
            "Libri2Mix must emit `separation`, not `enhancement`"
        );
        assert_eq!(
            file.get(KEY_SEPFORMER_VARIANT).and_then(|v| v.as_str()),
            Some("libri2mix"),
            "variant tag must NOT be inherited from the Wsj02mix sibling"
        );
        assert_eq!(
            file.get(chunks::KEY_MODEL_NAME).and_then(|v| v.as_str()),
            Some("sepformer-libri2mix")
        );
        assert_eq!(
            file.get(KEY_PROVENANCE_UPSTREAM_HF)
                .and_then(|v| v.as_str()),
            Some("speechbrain/sepformer-libri2mix")
        );
        // Arch is shared with every sibling — Libri2Mix does not
        // introduce a new arch tag.
        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
            Some(ARCH)
        );

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }

    /// Libri3Mix variant carries `separation` category with `n_out = 3`
    /// (3-speaker cocktail-party head — same LibriMix corpus family as
    /// [`SepformerVariant::Libri2Mix`], but the masker output branches
    /// into 3 parallel speaker streams instead of 2). Verifies the
    /// artifact carries distinct model.name / upstream_hf / variant /
    /// n_out stamps (must NOT silently inherit the 2-speaker sibling's
    /// provenance = wrong CDN attribution + wrong binder output-stream
    /// axis).
    #[test]
    fn libri3mix_carries_separation_category_and_n_out_3() {
        let values: [f32; 6] = [0.375, -0.75, 1.125, -1.5, 0.125, -0.25];
        let bf16: Vec<u8> = values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        let input_bytes = safetensors_one_bf16("masker.dpt.0.weight", &[3, 2], &bf16);
        let input = write_temp("libri3mix-in", &input_bytes);
        let output = write_temp("libri3mix-out", &[]);

        let report = convert_sepformer_file(&input, &output, None, SepformerVariant::Libri3Mix)
            .expect("convert sepformer-libri3mix");
        assert_eq!(report.written, 1);
        assert_eq!(report.bf16_passthrough, 1);
        assert_eq!(report.variant, Some(SepformerVariant::Libri3Mix));

        let out_bytes = std::fs::read(&output).expect("read output");
        let file = GgufFile::parse(out_bytes).expect("parse GGUF");
        // Libri3Mix is a separation task (3-speaker head), NOT an
        // enhancement task.
        assert_eq!(
            file.get(KEY_MODEL_CATEGORY).and_then(|v| v.as_str()),
            Some("separation"),
            "Libri3Mix must emit `separation`, not `enhancement`"
        );
        assert_eq!(
            file.get(KEY_SEPFORMER_VARIANT).and_then(|v| v.as_str()),
            Some("libri3mix"),
            "variant tag must NOT be inherited from the Libri2Mix sibling"
        );
        assert_eq!(
            file.get(chunks::KEY_MODEL_NAME).and_then(|v| v.as_str()),
            Some("sepformer-libri3mix")
        );
        assert_eq!(
            file.get(KEY_PROVENANCE_UPSTREAM_HF)
                .and_then(|v| v.as_str()),
            Some("speechbrain/sepformer-libri3mix"),
            "upstream_hf must point at the 3-speaker repo, NOT the 2-speaker sibling"
        );
        // The whole point of stamping n_out explicitly: a downstream
        // binder can pre-allocate 3 output streams without inspecting
        // free-text variant tags.
        assert_eq!(
            file.get(KEY_SEPFORMER_N_OUT).and_then(|v| v.as_u64()),
            Some(3),
            "Libri3Mix must emit n_out = 3, NOT the 2-speaker sibling's implicit 2"
        );
        // Arch is shared with every sibling — Libri3Mix does not
        // introduce a new arch tag.
        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
            Some(ARCH)
        );

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }

    /// Regression pin: every SepFormer variant must emit
    /// `vokra.sepformer.n_out` with the correct axis, so a downstream
    /// binder can pre-allocate output streams without ever inferring
    /// them from the variant tag string. Adding a variant means adding
    /// a row here.
    #[test]
    fn every_variant_stamps_the_expected_n_out_axis() {
        let cases: [(SepformerVariant, u32); 7] = [
            (SepformerVariant::Wsj02mix, 2),
            (SepformerVariant::Libri2Mix, 2),
            (SepformerVariant::Libri3Mix, 3),
            (SepformerVariant::Wham16kEnhancement, 1),
            (SepformerVariant::Whamr16k, 2),
            (SepformerVariant::Whamr8k, 2),
            (SepformerVariant::Dns4Enhancement, 1),
        ];
        // Reused 2-value BF16 tensor keeps the fixture cheap.
        let values: [f32; 2] = [1.0, -1.0];
        let bf16: Vec<u8> = values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        let input_bytes = safetensors_one_bf16("masker.w", &[1, 2], &bf16);
        let input = write_temp("nout-in", &input_bytes);
        for (variant, expected) in cases {
            let output = write_temp("nout-out", &[]);
            convert_sepformer_file(&input, &output, None, variant)
                .expect("convert sepformer variant");
            let out_bytes = std::fs::read(&output).expect("read output");
            let file = GgufFile::parse(out_bytes).expect("parse GGUF");
            let got = file
                .get(KEY_SEPFORMER_N_OUT)
                .and_then(|v| v.as_u64())
                .map(|v| v as u32);
            assert_eq!(
                got,
                Some(expected),
                "variant {variant:?} must stamp n_out = {expected}"
            );
            // Cross-check: the const method and the metadata agree.
            assert_eq!(variant.n_out(), expected);
            std::fs::remove_file(&output).ok();
        }
        std::fs::remove_file(&input).ok();
    }

    /// Wham16k-enhancement variant carries the `enhancement` category
    /// and its own model.name / upstream_hf.
    #[test]
    fn wham16k_enhancement_carries_enhancement_category() {
        let values: [f32; 2] = [0.5, -0.5];
        let bf16: Vec<u8> = values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        let input_bytes = safetensors_one_bf16("encoder.conv.weight", &[1, 2], &bf16);
        let input = write_temp("wham16k-in", &input_bytes);
        let output = write_temp("wham16k-out", &[]);

        convert_sepformer_file(&input, &output, None, SepformerVariant::Wham16kEnhancement)
            .expect("convert sepformer-wham16k-enhancement");

        let out_bytes = std::fs::read(&output).expect("read output");
        let file = GgufFile::parse(out_bytes).expect("parse GGUF");
        assert_eq!(
            file.get(KEY_MODEL_CATEGORY).and_then(|v| v.as_str()),
            Some("enhancement"),
            "Wham16kEnhancement must NOT emit `separation`"
        );
        assert_eq!(
            file.get(KEY_SEPFORMER_VARIANT).and_then(|v| v.as_str()),
            Some("wham16k-enhancement")
        );
        assert_eq!(
            file.get(chunks::KEY_MODEL_NAME).and_then(|v| v.as_str()),
            Some("sepformer-wham16k-enhancement")
        );
        assert_eq!(
            file.get(KEY_PROVENANCE_UPSTREAM_HF)
                .and_then(|v| v.as_str()),
            Some("speechbrain/sepformer-wham16k-enhancement")
        );

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }

    /// Whamr16k carries `separation` and two outputs, matching the official
    /// `num_spks: 2` topology and model card.
    #[test]
    fn whamr16k_carries_separation_category_and_two_outputs() {
        let values: [f32; 2] = [1.0, -1.0];
        let bf16: Vec<u8> = values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        let input_bytes = safetensors_one_bf16("decoder.conv.weight", &[1, 2], &bf16);
        let input = write_temp("whamr16k-in", &input_bytes);
        let output = write_temp("whamr16k-out", &[]);

        convert_sepformer_file(&input, &output, None, SepformerVariant::Whamr16k)
            .expect("convert sepformer-whamr16k");

        let out_bytes = std::fs::read(&output).expect("read output");
        let file = GgufFile::parse(out_bytes).expect("parse GGUF");
        assert_eq!(
            file.get(KEY_MODEL_CATEGORY).and_then(|v| v.as_str()),
            Some("separation")
        );
        assert_eq!(
            file.get(KEY_SEPFORMER_N_OUT).and_then(|v| v.as_u64()),
            Some(2)
        );
        assert_eq!(
            file.get(KEY_SEPFORMER_VARIANT).and_then(|v| v.as_str()),
            Some("whamr16k")
        );
        assert_eq!(
            file.get(chunks::KEY_MODEL_NAME).and_then(|v| v.as_str()),
            Some("sepformer-whamr16k")
        );

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }

    /// Whamr8k variant (8 kHz WHAMR! sibling of Whamr16k) carries
    /// `separation` category, two outputs, its own model.name / upstream_hf, and
    /// the distinct `whamr8k` tag — the artifact must NOT silently
    /// inherit the 16 kHz sibling's provenance stamps.
    #[test]
    fn whamr8k_carries_distinct_provenance_from_16k_sibling() {
        let values: [f32; 2] = [0.75, -0.75];
        let bf16: Vec<u8> = values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        let input_bytes = safetensors_one_bf16("masker.dpt.0.weight", &[1, 2], &bf16);
        let input = write_temp("whamr8k-in", &input_bytes);
        let output = write_temp("whamr8k-out", &[]);

        let report = convert_sepformer_file(&input, &output, None, SepformerVariant::Whamr8k)
            .expect("convert sepformer-whamr (8 kHz)");
        assert_eq!(report.written, 1);
        assert_eq!(report.bf16_passthrough, 1);
        assert_eq!(report.variant, Some(SepformerVariant::Whamr8k));

        let out_bytes = std::fs::read(&output).expect("read output");
        let file = GgufFile::parse(out_bytes).expect("parse GGUF");
        assert_eq!(
            file.get(KEY_MODEL_CATEGORY).and_then(|v| v.as_str()),
            Some("separation"),
            "Whamr8k must emit source separation, matching the official model card"
        );
        assert_eq!(
            file.get(KEY_SEPFORMER_N_OUT).and_then(|v| v.as_u64()),
            Some(2),
            "Whamr8k must emit the official two speaker streams"
        );
        assert_eq!(
            file.get(KEY_SEPFORMER_VARIANT).and_then(|v| v.as_str()),
            Some("whamr8k"),
            "variant tag must NOT be inherited from the Whamr16k sibling"
        );
        assert_eq!(
            file.get(chunks::KEY_MODEL_NAME).and_then(|v| v.as_str()),
            Some("sepformer-whamr"),
            "model.name must be the base 8 kHz upstream HF slug, NOT the 16k sibling's suffix"
        );
        assert_eq!(
            file.get(KEY_PROVENANCE_UPSTREAM_HF)
                .and_then(|v| v.as_str()),
            Some("speechbrain/sepformer-whamr"),
            "upstream_hf must point at the 8 kHz repo, NOT sepformer-whamr16k"
        );
        // Arch is shared with every sibling — Whamr8k does not
        // introduce a new arch tag (same encoder + dual-path
        // Transformer masker + decoder topology).
        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
            Some(ARCH)
        );

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }

    /// Dns4Enhancement variant (Microsoft DNS-4 16 kHz sibling of the
    /// WHAM! enhancement model) carries `enhancement`
    /// category, its own model.name / upstream_hf, and the distinct
    /// `dns4-16k-enhancement` tag — the artifact must NOT silently
    /// inherit any WHAM / WHAMR sibling's provenance stamps. The two
    /// single-stream enhancement models share `n_out = 1`, so provenance is the
    /// signal that would surface a routing mistake at load time.
    #[test]
    fn dns4_enhancement_carries_distinct_provenance_from_wham_family() {
        let values: [f32; 2] = [0.375, -0.375];
        let bf16: Vec<u8> = values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        let input_bytes = safetensors_one_bf16("masker.dpt.0.weight", &[1, 2], &bf16);
        let input = write_temp("dns4-in", &input_bytes);
        let output = write_temp("dns4-out", &[]);

        let report =
            convert_sepformer_file(&input, &output, None, SepformerVariant::Dns4Enhancement)
                .expect("convert sepformer-dns4-16k-enhancement");
        assert_eq!(report.written, 1);
        assert_eq!(report.bf16_passthrough, 1);
        assert_eq!(report.variant, Some(SepformerVariant::Dns4Enhancement));

        let out_bytes = std::fs::read(&output).expect("read output");
        let file = GgufFile::parse(out_bytes).expect("parse GGUF");
        assert_eq!(
            file.get(KEY_MODEL_CATEGORY).and_then(|v| v.as_str()),
            Some("enhancement"),
            "Dns4Enhancement must emit `enhancement`"
        );
        assert_eq!(
            file.get(KEY_SEPFORMER_VARIANT).and_then(|v| v.as_str()),
            Some("dns4-16k-enhancement"),
            "variant tag must NOT be inherited from any WHAM / WHAMR sibling"
        );
        assert_eq!(
            file.get(chunks::KEY_MODEL_NAME).and_then(|v| v.as_str()),
            Some("sepformer-dns4-16k-enhancement")
        );
        assert_eq!(
            file.get(KEY_PROVENANCE_UPSTREAM_HF)
                .and_then(|v| v.as_str()),
            Some("speechbrain/sepformer-dns4-16k-enhancement"),
            "upstream_hf must point at the DNS-4 repo, NOT any WHAM / WHAMR sibling"
        );
        // n_out is 1 (single-output enhancement head), unlike WHAMR's
        // two-speaker separation head.
        assert_eq!(
            file.get(KEY_SEPFORMER_N_OUT).and_then(|v| v.as_u64()),
            Some(1),
            "Dns4Enhancement must emit n_out = 1 (single-output enhancement head)"
        );
        // Arch is shared with every sibling — Dns4Enhancement does not
        // introduce a new arch tag (same encoder + dual-path Transformer
        // masker + decoder topology).
        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
            Some(ARCH)
        );

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }

    /// F32 + F16 mixed pass-through with BF16 counter at 0.
    #[test]
    fn f32_and_f16_pass_through() {
        let f32_vals: [f32; 2] = [7.0, -8.25];
        let f32_bytes: Vec<u8> = f32_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let f16_words: [u16; 4] = [0x3C00, 0xC000, 0xB800, 0x4200];
        let f16_bytes: Vec<u8> = f16_words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let f32_len = f32_bytes.len();
        let total = f32_len + f16_bytes.len();
        let header = format!(
            r#"{{"a.w":{{"dtype":"F32","shape":[1,2],"data_offsets":[0,{f32_len}]}},"b.w":{{"dtype":"F16","shape":[2,2],"data_offsets":[{f32_len},{total}]}}}}"#
        );
        let mut input_bytes = Vec::new();
        input_bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        input_bytes.extend_from_slice(header.as_bytes());
        input_bytes.extend_from_slice(&f32_bytes);
        input_bytes.extend_from_slice(&f16_bytes);
        let input = write_temp("mixed-in", &input_bytes);
        let output = write_temp("mixed-out", &[]);

        let report =
            convert_sepformer_file(&input, &output, None, SepformerVariant::Wham16kEnhancement)
                .expect("convert mixed");
        assert_eq!(report.read, 2);
        assert_eq!(report.written, 2);
        assert_eq!(report.bf16_passthrough, 0);
        assert_eq!(report.skipped_non_float, 0);

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }

    /// License override lands on the artifact — apache-2.0 default,
    /// override to mit stays Permissive.
    #[test]
    fn license_override_reaches_the_artifact() {
        let values: [f32; 2] = [0.5, -0.5];
        let bf16: Vec<u8> = values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        let input_bytes = safetensors_one_bf16("masker.w", &[1, 2], &bf16);
        let input = write_temp("license-in", &input_bytes);
        let output = write_temp("license-out", &[]);

        convert_sepformer_file(&input, &output, Some("mit"), SepformerVariant::Wsj02mix)
            .expect("convert with override");

        let out_bytes = std::fs::read(&output).expect("read output");
        let file = GgufFile::parse(out_bytes).expect("parse GGUF");
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some("mit")
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::Permissive.as_str())
        );

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }
}
