#![allow(clippy::doc_lazy_continuation)]
//! **ConvTasNet Libri1Mix Enhancement** (Asteroid,
//! `JorisCos/ConvTasNet_Libri1Mix_enhsingle_16k`, **license unresolved**):
//! safetensors → GGUF conversion.
//!
//! ConvTasNet (Luo & Mesgarani 2019, arXiv:1809.07454) — fully
//! convolutional TasNet variant, encoder + temporal-convolutional
//! network (stacked dilated 1-D convs) mask estimator + decoder.
//! This checkpoint is the Asteroid recipe fine-tuned on **Libri1Mix
//! `enhsingle`** (single-speaker enhancement, 16 kHz — one clean
//! speaker + additive noise, one output stream).
//!
//! # Family posture — distinct from SepFormer / Demucs
//!
//! Distinct arch tag `conv_tasnet` from sibling separator families
//! (`sepformer` = dual-path Transformer masker, `demucs` = hybrid
//! U-Net + spectrogram + cross-domain attention). ConvTasNet's stacked
//! dilated TCN is topologically distinct from both — FR-EX-08 forbids
//! silent shape misroute across separator families. `category =
//! "enhancement"` (single-output enhancement head, mirrors the
//! SepFormer WHAM / DNS-4 enhancement and WHAMR separation variants; the
//! multi-speaker separation ConvTasNet variants would carry
//! `category = "separation"` when they land in a follow-up).
//!
//! # License posture — conflicting declarations (**fail closed**)
//!
//! At upstream revision `bb8a876bc157b5cf3c405994accb798c49146016`, HF
//! cardData says CC-BY-SA-4.0, the model-card weight notice says
//! CC-BY-SA-3.0, and the card identifies WHAM-derived training material as
//! CC-BY-NC-4.0 (Research only).  The converter therefore defaults to
//! [`LicenseClass::Unknown`] and SPDX `unknown`; publication stays blocked
//! until an owner/legal review resolves the conflict.  `--license` remains
//! available only for callers who independently hold this exact checkpoint
//! under terms they can substantiate.
//!
//! # Upstream format — pytorch_model.bin (owner prep)
//!
//! Asteroid ships a single ~20 MB `pytorch_model.bin` checkpoint whose
//! seven-field wrapper is not a bare state dict. Owners run the dedicated
//! `tools/parity/conv_tasnet_prepare_checkpoint.py` safe extraction before
//! pointing this converter at the resulting `.safetensors`. This converter
//! deliberately never reads
//! `pytorch_model.bin` directly — pickle deserialization inside the
//! Rust runtime would violate the FR-LD-05 "no arbitrary code
//! execution at load" rule.
//!
//! # BF16 pass-through (mirror of musicgen_small / sepformer)
//!
//! F32 / F16 / BF16 float tensors ride the verbatim pass-through arm.
//! BF16 stays GGUF type 30 (`GgmlType::BF16`); runtime widens
//! BF16 → f32 losslessly at load via
//! `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16`. The
//! ConvTasNet checkpoint is F32 in the wild (Asteroid recipe emits
//! F32), so BF16 pass-through is a defensive skeleton for a future
//! quantised sibling — not the primary path.
//!
//! # Tensor naming contract
//!
//! GGUF tensor names are the **upstream state-dict keys verbatim**. The exact
//! 345-tensor manifest and every shape are checked before any output is
//! written; a sibling or incomplete checkpoint fails closed.
//!
//! # No ONNX (permanent)
//!
//! Asteroid ships PyTorch checkpoints only; this converter **never**
//! touches ONNX (FR-LD-05).

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` = `conv_tasnet` — distinct from sibling
/// separator arch tags (`sepformer` / `demucs` / `tiger_separator` /
/// `bs_roformer` / `mp_senet`). FR-EX-08 forbids silent shape
/// misroute across separator families.
pub const ARCH: &str = "conv_tasnet";

/// `vokra.model.name` — the Libri1Mix `enhsingle` 16 kHz enhancement
/// checkpoint slug. Sibling ConvTasNet variants (multi-speaker
/// separation, other enhancement corpora) would each carry their own
/// [`ModelKind`](crate::ModelKind) + distinct name stamp per the
/// SepFormer variant precedent — silently sharing a single "conv-tasnet"
/// slug would misroute a downstream binder that dispatches on category
/// or output-stream count.
pub const NAME: &str = "conv-tasnet-libri1mix";

/// `vokra.model.category` = `enhancement` — single-speaker output
/// head (Libri1Mix `enhsingle` = one clean speaker + additive noise,
/// one output stream). Mirrors the SepFormer WHAM / WHAMR / DNS-4
/// enhancement sibling posture. Future multi-speaker ConvTasNet
/// variants would carry `category = "separation"` under a distinct
/// `ModelKind` arm.
pub const CATEGORY: &str = "enhancement";

/// Upstream HuggingFace slug (`org/name`) — used for
/// `vokra.provenance.upstream_hf`.
pub const UPSTREAM_HF: &str = "JorisCos/ConvTasNet_Libri1Mix_enhsingle_16k";

/// Fail-closed default because the upstream model card conflicts with itself.
pub const DEFAULT_LICENSE_SPDX: &str = "unknown";

const UPSTREAM_SOURCE: &str = "JorisCos/ConvTasNet_Libri1Mix_enhsingle_16k@bb8a876bc157b5cf3c405994accb798c49146016 (license conflict: cardData CC-BY-SA-4.0; card notice CC-BY-SA-3.0; WHAM-derived data CC-BY-NC-4.0 Research only)";

const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";
const KEY_PROVENANCE_UPSTREAM_REVISION: &str = "vokra.provenance.upstream_revision";
const KEY_N_FILTERS: &str = "vokra.conv_tasnet.n_filters";
const KEY_N_KERNEL: &str = "vokra.conv_tasnet.n_kernel";
const KEY_STRIDE: &str = "vokra.conv_tasnet.stride";
const KEY_N_BLOCKS: &str = "vokra.conv_tasnet.n_blocks";
const KEY_N_REPEATS: &str = "vokra.conv_tasnet.n_repeats";
const KEY_BN_CHAN: &str = "vokra.conv_tasnet.bn_chan";
const KEY_HID_CHAN: &str = "vokra.conv_tasnet.hid_chan";
const KEY_SKIP_CHAN: &str = "vokra.conv_tasnet.skip_chan";
const KEY_CONV_KERNEL_SIZE: &str = "vokra.conv_tasnet.conv_kernel_size";
const KEY_SAMPLE_RATE: &str = "vokra.conv_tasnet.sample_rate";
const KEY_N_SRC: &str = "vokra.conv_tasnet.n_src";
const KEY_CAUSAL: &str = "vokra.conv_tasnet.causal";

const N_FILTERS: u32 = 512;
const N_KERNEL: u32 = 32;
const STRIDE: u32 = 16;
const N_BLOCKS: u32 = 8;
const N_REPEATS: u32 = 3;
const BN_CHAN: u32 = 128;
const HID_CHAN: u32 = 512;
const SKIP_CHAN: u32 = 128;
const CONV_KERNEL_SIZE: u32 = 3;
const SAMPLE_RATE: u32 = 16_000;
const N_SRC: u32 = 1;
const CAUSAL: u32 = 0;
const UPSTREAM_REVISION: &str = "bb8a876bc157b5cf3c405994accb798c49146016";
const TENSOR_COUNT: usize = 345;

fn expected_shapes() -> Vec<(String, Vec<u64>)> {
    let mut shapes = vec![
        ("encoder.filterbank._filters".to_owned(), vec![512, 1, 32]),
        ("masker.bottleneck.0.gamma".to_owned(), vec![512]),
        ("masker.bottleneck.0.beta".to_owned(), vec![512]),
        ("masker.bottleneck.1.weight".to_owned(), vec![128, 512, 1]),
        ("masker.bottleneck.1.bias".to_owned(), vec![128]),
    ];
    for block in 0..24 {
        let prefix = format!("masker.TCN.{block}");
        for (suffix, shape) in [
            ("shared_block.0.weight", vec![512, 128, 1]),
            ("shared_block.0.bias", vec![512]),
            ("shared_block.1.weight", vec![1]),
            ("shared_block.2.gamma", vec![512]),
            ("shared_block.2.beta", vec![512]),
            ("shared_block.3.weight", vec![512, 1, 3]),
            ("shared_block.3.bias", vec![512]),
            ("shared_block.4.weight", vec![1]),
            ("shared_block.5.gamma", vec![512]),
            ("shared_block.5.beta", vec![512]),
            ("res_conv.weight", vec![128, 512, 1]),
            ("res_conv.bias", vec![128]),
            ("skip_conv.weight", vec![128, 512, 1]),
            ("skip_conv.bias", vec![128]),
        ] {
            shapes.push((format!("{prefix}.{suffix}"), shape));
        }
    }
    shapes.extend([
        ("masker.mask_net.0.weight".to_owned(), vec![1]),
        ("masker.mask_net.1.weight".to_owned(), vec![512, 128, 1]),
        ("masker.mask_net.1.bias".to_owned(), vec![512]),
        ("decoder.filterbank._filters".to_owned(), vec![512, 1, 32]),
    ]);
    debug_assert_eq!(shapes.len(), TENSOR_COUNT);
    shapes
}

fn validate_manifest(st: &SafetensorsFile) -> Result<(), ConvertError> {
    if st.tensors().len() != TENSOR_COUNT {
        return Err(ConvertError::Parse(format!(
            "conv_tasnet: checkpoint has {} tensors, expected exactly {TENSOR_COUNT}",
            st.tensors().len()
        )));
    }
    for (name, shape) in expected_shapes() {
        let tensor = st
            .tensors()
            .iter()
            .find(|tensor| tensor.name == name)
            .ok_or_else(|| ConvertError::Parse(format!("conv_tasnet: missing tensor `{name}`")))?;
        if tensor.shape != shape {
            return Err(ConvertError::Parse(format!(
                "conv_tasnet: tensor `{name}` has shape {:?}, expected {shape:?}",
                tensor.shape
            )));
        }
        if !matches!(tensor.dtype, GgmlType::F32 | GgmlType::F16 | GgmlType::BF16) {
            return Err(ConvertError::Parse(format!(
                "conv_tasnet: tensor `{name}` has non-floating dtype {:?}",
                tensor.dtype
            )));
        }
    }
    Ok(())
}

fn stamp_topology(builder: &mut GgufBuilder) {
    for (key, value) in [
        (KEY_N_FILTERS, N_FILTERS),
        (KEY_N_KERNEL, N_KERNEL),
        (KEY_STRIDE, STRIDE),
        (KEY_N_BLOCKS, N_BLOCKS),
        (KEY_N_REPEATS, N_REPEATS),
        (KEY_BN_CHAN, BN_CHAN),
        (KEY_HID_CHAN, HID_CHAN),
        (KEY_SKIP_CHAN, SKIP_CHAN),
        (KEY_CONV_KERNEL_SIZE, CONV_KERNEL_SIZE),
        (KEY_SAMPLE_RATE, SAMPLE_RATE),
        (KEY_N_SRC, N_SRC),
        (KEY_CAUSAL, CAUSAL),
    ] {
        builder.add_u32(key, value);
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct ConvTasnetLibri1mixReport {
    pub read: usize,
    pub written: usize,
    pub skipped_non_float: usize,
    pub bf16_passthrough: usize,
}

pub fn convert_conv_tasnet_libri1mix_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<ConvTasnetLibri1mixReport, ConvertError> {
    let bytes = std::fs::read(input)?;
    let st = SafetensorsFile::parse(bytes)?;
    validate_manifest(&st)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    b.add_string(KEY_MODEL_CATEGORY, CATEGORY);
    stamp_topology(&mut b);
    b.add_string(KEY_PROVENANCE_UPSTREAM_REVISION, UPSTREAM_REVISION);

    let (spdx, class) = match license {
        Some(s) if !s.is_empty() => (s.to_owned(), LicenseClass::from_license_str(s)),
        // The upstream declarations conflict.  Never guess the least
        // restrictive term: `Unknown` makes both load/publication policy fail
        // closed until the caller supplies a separately established licence.
        _ => (DEFAULT_LICENSE_SPDX.to_owned(), LicenseClass::Unknown),
    };
    vokra_core::stamp_provenance(&mut b, class, &spdx, Some(NAME), Some(UPSTREAM_SOURCE));
    b.add_string(KEY_PROVENANCE_UPSTREAM_HF, UPSTREAM_HF);

    let mut report = ConvTasnetLibri1mixReport::default();
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
            _ => unreachable!("validate_manifest rejects non-floating Conv-TasNet tensors"),
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
            "vokra-convert-conv-tasnet-libri1mix-{tag}-{}-{n}",
            std::process::id()
        ));
        p
    }

    fn safetensors_checkpoint(override_name: &str, override_dtype: &str) -> Vec<u8> {
        let mut offset = 0usize;
        let mut payload = Vec::new();
        let mut entries = Vec::new();
        for (name, shape) in expected_shapes() {
            let dtype = if name == override_name {
                override_dtype
            } else {
                "F16"
            };
            let element_bytes = match dtype {
                "F32" => 4,
                "F16" | "BF16" => 2,
                other => panic!("unsupported test dtype {other}"),
            };
            let bytes = shape.iter().product::<u64>() as usize * element_bytes;
            let end = offset + bytes;
            let shape = shape
                .iter()
                .map(u64::to_string)
                .collect::<Vec<_>>()
                .join(",");
            entries.push(format!(
                r#""{name}":{{"dtype":"{dtype}","shape":[{shape}],"data_offsets":[{offset},{end}]}}"#
            ));
            payload.resize(payload.len() + bytes, 0);
            offset = end;
        }
        let header = format!("{{{}}}", entries.join(","));
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(&payload);
        out
    }

    /// F32 pass-through + provenance stamps + fail-closed default.
    #[test]
    fn f32_pass_through_and_default_license_is_unknown() {
        let inp = tmp_path("f32-in");
        let outp = tmp_path("f32-out");
        // Use an upstream state-dict-like name so the naming contract
        // is exercised (verbatim key pass-through).
        let st = safetensors_checkpoint("masker.TCN.0.shared_block.0.weight", "F32");
        std::fs::write(&inp, &st).unwrap();
        let r = convert_conv_tasnet_libri1mix_file(&inp, &outp, None).unwrap();
        assert_eq!(r.read, TENSOR_COUNT);
        assert_eq!(r.written, TENSOR_COUNT);
        assert_eq!(r.bf16_passthrough, 0);
        assert_eq!(r.skipped_non_float, 0);

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
        for (key, expected) in [
            (KEY_N_FILTERS, N_FILTERS),
            (KEY_N_KERNEL, N_KERNEL),
            (KEY_STRIDE, STRIDE),
            (KEY_N_BLOCKS, N_BLOCKS),
            (KEY_N_REPEATS, N_REPEATS),
            (KEY_BN_CHAN, BN_CHAN),
            (KEY_HID_CHAN, HID_CHAN),
            (KEY_SKIP_CHAN, SKIP_CHAN),
            (KEY_CONV_KERNEL_SIZE, CONV_KERNEL_SIZE),
            (KEY_SAMPLE_RATE, SAMPLE_RATE),
            (KEY_N_SRC, N_SRC),
            (KEY_CAUSAL, CAUSAL),
        ] {
            assert_eq!(
                g.get(key),
                Some(&vokra_core::gguf::GgufMetadataValue::U32(expected)),
                "{key}"
            );
        }
        assert_eq!(
            read_str(chunks::KEY_PROVENANCE_LICENSE),
            DEFAULT_LICENSE_SPDX
        );
        assert_eq!(
            read_str(chunks::KEY_PROVENANCE_WEIGHT_LICENSE),
            LicenseClass::Unknown.as_str(),
            "conflicting upstream declarations must default fail-closed"
        );
        // Cross-check: the upstream tensor name is preserved
        // verbatim in the GGUF (naming contract).
        assert!(
            g.tensor_info("masker.TCN.0.shared_block.0.weight")
                .is_some()
        );
        let _ = std::fs::remove_file(&inp);
        let _ = std::fs::remove_file(&outp);
    }

    /// BF16 pass-through arm — the ConvTasNet Asteroid checkpoint is
    /// F32 in the wild, but the defensive BF16 path is exercised for
    /// parity with the sibling BF16 pass-through skeletons.
    #[test]
    fn bf16_tensor_passes_through_verbatim() {
        let inp = tmp_path("bf16-in");
        let outp = tmp_path("bf16-out");
        let st = safetensors_checkpoint("encoder.filterbank._filters", "BF16");
        std::fs::write(&inp, &st).unwrap();
        let r = convert_conv_tasnet_libri1mix_file(&inp, &outp, None).unwrap();
        assert_eq!(r.bf16_passthrough, 1);

        let g = GgufFile::open(&outp).unwrap();
        let info = g
            .tensor_info("encoder.filterbank._filters")
            .expect("BF16 tensor present");
        assert_eq!(info.dtype, GgmlType::BF16);
        assert!(g.tensor_bytes(info).iter().all(|&byte| byte == 0));
        let _ = std::fs::remove_file(&inp);
        let _ = std::fs::remove_file(&outp);
    }

    /// A caller supplying `--license mit` overrides the fail-closed default.
    /// Same override escape hatch
    /// used by Whisper / kokoro / vits-ja / xcodec2 / musicgen for
    /// callers who legitimately hold the weight under a different
    /// SPDX id (e.g. an Asteroid-compatible retraining released
    /// under a permissive licence).
    #[test]
    fn license_override_swaps_stamp_off_unknown() {
        let inp = tmp_path("lic-in");
        let outp = tmp_path("lic-out");
        let st = safetensors_checkpoint("encoder.filterbank._filters", "F32");
        std::fs::write(&inp, &st).unwrap();
        convert_conv_tasnet_libri1mix_file(&inp, &outp, Some("mit")).unwrap();
        let g = GgufFile::open(&outp).unwrap();
        assert_eq!(
            g.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some("mit")
        );
        assert_eq!(
            g.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::Permissive.as_str())
        );
        let _ = std::fs::remove_file(&inp);
        let _ = std::fs::remove_file(&outp);
    }
}
