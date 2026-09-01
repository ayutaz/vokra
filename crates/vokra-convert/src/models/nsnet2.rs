//! **NSNet2** (Microsoft DNS Challenge NR baseline): safetensors checkpoint
//! → GGUF conversion (Coverage-audit 2026-08-03 Wave A ticket).
//!
//! Input: the upstream Microsoft DNS-Challenge NR baseline —
//! `NSNet2-baseline/nsnet2-20ms-baseline.onnx` (~10.8 MB). Because the upstream
//! release is ONNX-only and Vokra's runtime never links ONNX / protobuf
//! (FR-LD-05, NFR-DS-02), the offline sidecar
//! `tools/parity/nsnet2_prepare_checkpoint.py` first bridges ONNX → safetensors;
//! this converter then consumes that safetensors input and stamps the
//! `vokra.model.*` / `vokra.provenance.*` chunk groups a future native
//! `vokra-models::nsnet2::*` implementation will read.
//!
//! # Model class
//!
//! NSNet2 is a 20 ms-frame single-channel noise-suppression baseline (ICASSP
//! 2020, `arXiv:2005.07551`): a 2-layer GRU + 3-Linear mask predictor operating
//! over the 161-bin log-power spectrum of the 16 kHz input (STFT `n_fft=320`,
//! hop 10 ms, 20 ms square-root Hann window). Its role in the Vokra catalogue is the
//! quantization-CI / industry-baseline reference for the `denoise` op family;
//! it is deliberately **weaker** than DeepFilterNet3 (M4-20 T17) but
//! architecturally distinct enough that silently sharing the `denoise` arch tag
//! would misroute the runtime dispatch.
//!
//! # License
//!
//! Both code and weights ship **MIT** end-to-end
//! (`github.com/microsoft/DNS-Challenge/blob/master/LICENSE`, fetched
//! 2026-08-03 — CLAUDE.md「ハルシネーション厳禁」). MIT is a `Permissive`
//! license class — same commercial verdict as apache-2.0 (no runtime-side
//! attribution obligation).
//!
//! # Dtype posture
//!
//! The pinned official ONNX stores all 14 initializers as F32. This converter
//! accepts that exact dtype and manifest only. A half-precision or otherwise
//! altered checkpoint is a different artifact and is rejected until it has its
//! own independently verified conversion contract.
//!
//! # Tensor naming contract
//!
//! The official ONNX graph uses numeric initializer ids for every matrix and
//! PyTorch module names only for biases. This converter accepts the exact
//! 14-initializer manifest, renames it to the runtime's stable semantic names,
//! removes ONNX's singleton direction axes, and transposes MatMul weights from
//! `[in, out]` to runtime `[out, in]`. Missing, extra, retyped or reshaped
//! tensors are hard errors.
//!
//! # No ONNX (permanent)
//!
//! NSNet2 is distributed as ONNX; this converter **never** touches ONNX
//! directly (FR-LD-05). The offline
//! `tools/parity/nsnet2_prepare_checkpoint.py` sidecar performs the ONNX →
//! safetensors bridge with `onnx` + `numpy` + `safetensors` in a Python venv
//! that is not part of the runtime shipping surface (mirror of
//! `bin_to_safetensors.py`'s posture for pytorch `.bin` inputs). The pipeline
//! will be re-implemented natively when a `crates/vokra-models/src/nsnet2/`
//! lands (whisper.cpp 型 self re-implementation, CLAUDE.md 設計判断 4).

#![allow(dead_code)]

use std::io::Write;
use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` for NSNet2 GGUFs. Distinct from every sibling arch tag
/// (in particular from `denoise` = DeepFilterNet3, which is a completely
/// different topology — DFN3 uses an ERB analysis / synthesis pair around a
/// convolutional recurrent network, whereas NSNet2 is a 2-layer GRU + 3-Linear
/// mask over 161-bin STFT log-power). Silently sharing an arch tag would
/// misroute the runtime dispatch.
pub const ARCH: &str = "nsnet2";

/// `vokra.model.name` value written for the canonical NSNet2 20 ms baseline
/// GGUF. Matches the upstream ONNX filename stem (dashes preserved) so a
/// downstream reader can reconstruct the source from the artifact alone.
pub const NAME: &str = "nsnet2-20ms-baseline";

/// `vokra.model.category` value — `enhancement` (the noise-suppression /
/// speech-enhancement family). Consumed by the model-card generator + zoo
/// manifest tier gate so a NR baseline is not accidentally advertised as an
/// ASR / TTS release.
pub const CATEGORY: &str = "enhancement";

/// `vokra.provenance.upstream_url` value — the GitHub tree the release ships
/// from. NSNet2 is not hosted on HuggingFace (the upstream is Microsoft's
/// public DNS Challenge repository), so this uses `upstream_url` rather than
/// `upstream_hf`; the model-card generator picks up either.
pub const UPSTREAM_URL: &str = "github.com/microsoft/DNS-Challenge/tree/8b87a33b2892f147b5c7ad39ea978453730db269/NSNet2-baseline";

/// Canonical released-model license SPDX (`cc-by-4.0`). Microsoft's fixed
/// DNS-Challenge revision puts source code under `LICENSE-CODE` (MIT), while
/// its root `LICENSE` and README Legal Notices put documentation and other
/// released content under CC-BY-4.0. The ONNX is released model content, so the
/// weight artifact is classified attribution-required unless a checkpoint
/// owner supplies stronger, checkpoint-specific terms.
pub const DEFAULT_LICENSE: &str = "cc-by-4.0";

/// Ad-hoc metadata key for the model category. Kept as a converter-side
/// constant (not a `chunks::KEY_*` alias) matching the sibling
/// `emotion2vec` / `ecapa_tdnn` posture until a first-class `category`
/// consumer lands in `vokra-core`.
const KEY_MODEL_CATEGORY: &str = "vokra.model.category";

/// Ad-hoc metadata key for the upstream URL (used for non-HF sources such as
/// GitHub / Zenodo / ModelScope). Sibling to
/// `emotion2vec::KEY_PROVENANCE_UPSTREAM_HF` — kept as a converter-side
/// constant to avoid premature promotion until a second non-HF converter
/// lands.
const KEY_PROVENANCE_UPSTREAM_URL: &str = "vokra.provenance.upstream_url";

// ---- `vokra.nsnet2.*` hparam chunk group ---------------------------------
//
// Mirror of `fsmn_vad::KEY_*` posture: every runtime hparam the future
// `vokra-models::nsnet2::Nsnet2V1::from_gguf` needs is stamped here so a
// downstream reader is fully self-describing (no external config side-car
// needed). Values are FunASR-style `u32` chunks; a `0`-sentinel on any of
// them makes the runtime binder refuse to load (FR-EX-08 — no silent
// default).

/// GGUF metadata key: STFT bin count (u32; upstream = 161 = `n_fft/2 + 1`).
pub const KEY_N_BINS: &str = "vokra.nsnet2.n_bins";
/// GGUF metadata key: GRU / fc_in hidden width (u32; upstream = 400).
pub const KEY_HIDDEN_DIM: &str = "vokra.nsnet2.hidden_dim";
/// GGUF metadata key: `fc_1` output width (u32; upstream = 600).
pub const KEY_FC1_DIM: &str = "vokra.nsnet2.fc1_dim";
/// GGUF metadata key: `fc_2` output width (u32; upstream = 600).
pub const KEY_FC2_DIM: &str = "vokra.nsnet2.fc2_dim";
/// GGUF metadata key: STFT FFT length (u32; upstream = 320).
pub const KEY_N_FFT: &str = "vokra.nsnet2.n_fft";
/// GGUF metadata key: STFT hop (u32 samples; upstream = 160 = 10 ms @ 16 kHz).
pub const KEY_HOP: &str = "vokra.nsnet2.hop";
/// GGUF metadata key: STFT window length (u32 samples; upstream = 320 = 20 ms
/// @ 16 kHz). A window shorter than `n_fft` is centred and zero-padded to
/// `n_fft` by the analysis op.
pub const KEY_WIN_LENGTH: &str = "vokra.nsnet2.win_length";
/// GGUF metadata key: PCM sample rate (u32 Hz; upstream = 16 000).
pub const KEY_SAMPLE_RATE: &str = "vokra.nsnet2.sample_rate";

/// Upstream STFT bin count (`n_fft/2 + 1` for `n_fft = 320`).
pub const DEFAULT_N_BINS: u32 = 161;
/// Upstream GRU / fc_in hidden width.
pub const DEFAULT_HIDDEN_DIM: u32 = 400;
/// Upstream `fc_1` output width.
pub const DEFAULT_FC1_DIM: u32 = 600;
/// Upstream `fc_2` output width.
pub const DEFAULT_FC2_DIM: u32 = 600;
/// Upstream FFT length (samples).
pub const DEFAULT_N_FFT: u32 = 320;
/// Upstream STFT hop (samples).
pub const DEFAULT_HOP: u32 = 160;
/// Upstream STFT window length (samples).
pub const DEFAULT_WIN_LENGTH: u32 = 320;
/// Upstream PCM sample rate (Hz).
pub const DEFAULT_SAMPLE_RATE: u32 = 16_000;

#[derive(Debug, Clone, Copy)]
struct TensorMap {
    source: &'static str,
    target: &'static str,
    source_shape: &'static [u64],
    target_shape: &'static [u64],
    transpose_2d: bool,
}

/// Exact initializer walk of Microsoft's pinned
/// `nsnet2-20ms-baseline.onnx` (commit `8b87a33b…`). Numeric names are ONNX
/// graph ids; semantic target names are the stable native runtime contract.
const TENSOR_MAP: &[TensorMap] = &[
    TensorMap {
        source: "172",
        target: "fc_in.weight",
        source_shape: &[161, 400],
        target_shape: &[400, 161],
        transpose_2d: true,
    },
    TensorMap {
        source: "fc_in.0.bias",
        target: "fc_in.bias",
        source_shape: &[400],
        target_shape: &[400],
        transpose_2d: false,
    },
    TensorMap {
        source: "192",
        target: "gru_1.W",
        source_shape: &[1, 1200, 400],
        target_shape: &[1200, 400],
        transpose_2d: false,
    },
    TensorMap {
        source: "193",
        target: "gru_1.R",
        source_shape: &[1, 1200, 400],
        target_shape: &[1200, 400],
        transpose_2d: false,
    },
    TensorMap {
        source: "194",
        target: "gru_1.B",
        source_shape: &[1, 2400],
        target_shape: &[2400],
        transpose_2d: false,
    },
    TensorMap {
        source: "212",
        target: "gru_2.W",
        source_shape: &[1, 1200, 400],
        target_shape: &[1200, 400],
        transpose_2d: false,
    },
    TensorMap {
        source: "213",
        target: "gru_2.R",
        source_shape: &[1, 1200, 400],
        target_shape: &[1200, 400],
        transpose_2d: false,
    },
    TensorMap {
        source: "214",
        target: "gru_2.B",
        source_shape: &[1, 2400],
        target_shape: &[2400],
        transpose_2d: false,
    },
    TensorMap {
        source: "215",
        target: "fc_1.weight",
        source_shape: &[400, 600],
        target_shape: &[600, 400],
        transpose_2d: true,
    },
    TensorMap {
        source: "fc_out.0.bias",
        target: "fc_1.bias",
        source_shape: &[600],
        target_shape: &[600],
        transpose_2d: false,
    },
    TensorMap {
        source: "216",
        target: "fc_2.weight",
        source_shape: &[600, 600],
        target_shape: &[600, 600],
        transpose_2d: true,
    },
    TensorMap {
        source: "fc_out.2.bias",
        target: "fc_2.bias",
        source_shape: &[600],
        target_shape: &[600],
        transpose_2d: false,
    },
    TensorMap {
        source: "217",
        target: "mask.weight",
        source_shape: &[600, 161],
        target_shape: &[161, 600],
        transpose_2d: true,
    },
    TensorMap {
        source: "fc_out.4.bias",
        target: "mask.bias",
        source_shape: &[161],
        target_shape: &[161],
        transpose_2d: false,
    },
];

/// Outcome of an NSNet2 conversion.
///
/// The official checkpoint contains exactly 14 F32 initializers. The legacy
/// counters remain in the shared conversion-report shape, but strict manifest
/// validation means a successful conversion always has `read == written == 14`
/// and both skip counters are zero.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Nsnet2Report {
    /// Total tensors surfaced by the safetensors reader (the sum of
    /// `written + skipped_non_float`). Pins the budget so a truncated header
    /// cannot silently drop tensors without the caller noticing.
    pub read: usize,
    /// Official F32 tensors renamed and written to the runtime schema.
    pub written: usize,
    /// Always zero after a successful strict-manifest conversion.
    pub skipped_non_float: usize,
    /// Always zero: the official manifest is F32-only.
    pub bf16_passthrough: usize,
}

/// Reads a safetensors checkpoint at `input` and writes an NSNet2 GGUF to
/// `output`.
///
/// The exact official 14-initializer manifest is renamed, singleton GRU
/// direction axes are removed, and MatMul weights are transposed into the
/// native row-major runtime layout. The `vokra.model.*` (arch / name / category) and
/// `vokra.provenance.*` (weight_license / license / model_id / source /
/// upstream_url) chunk groups are stamped for the runtime compliance gate
/// (FR-CP-03). `vokra.schema.*` is written unconditionally by the GGUF
/// writer.
///
/// `license` overrides `DEFAULT_LICENSE` (`"cc-by-4.0"`) — the same mechanism
/// `lib.rs::convert_file_licensed` uses when the implementation is
/// clean-room but the redistributed checkpoint carries a different SPDX
/// grant.
///
/// # Errors
///
/// [`ConvertError::Io`] on read / write failure; [`ConvertError::Parse`]
/// on a malformed safetensors input.
pub fn convert_nsnet2_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<Nsnet2Report, ConvertError> {
    // Whole-file read: NSNet2 ships as a ~10.8 MB ONNX which the prep script
    // flattens into a similarly small safetensors — no need for the
    // streaming path the Moshi 15 GB / Voxtral 8.7 GB converters run.
    let bytes = std::fs::read(input).map_err(ConvertError::Io)?;
    let st = SafetensorsFile::parse(bytes)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    b.add_string(KEY_MODEL_CATEGORY, CATEGORY);

    // Self-describing redistribution: the artifact carries its own licence.
    // At fixed commit 8b87a33b…, LICENSE-CODE grants MIT for code, but the
    // root LICENSE and README Legal Notices assign documentation and other
    // released content to CC-BY-4.0. Treat the released ONNX weights as that
    // attribution-required content; do not broaden the code licence to them.
    // The override remains available only for a checkpoint-specific grant.
    let effective_license = license.unwrap_or(DEFAULT_LICENSE);
    let effective_class = LicenseClass::from_license_str(effective_license);
    vokra_core::stamp_provenance(
        &mut b,
        effective_class,
        effective_license,
        Some(NAME),
        Some(
            "Microsoft DNS-Challenge NSNet2-baseline commit 8b87a33b2892f147b5c7ad39ea978453730db269 (code MIT; released model content CC-BY-4.0)",
        ),
    );
    b.add_string(KEY_PROVENANCE_UPSTREAM_URL, UPSTREAM_URL);

    // NSNet2 has one canonical topology — the 20 ms baseline
    // (`nsnet2-20ms-baseline.onnx`) — and every hparam is fixed at that
    // release. Stamping them here (mirror of `fsmn_vad::stamp_hparams`
    // posture) makes the artifact self-describing so the future
    // `vokra-models::nsnet2::Nsnet2V1::from_gguf` binder can validate
    // against these values loudly (FR-EX-08 — a checkpoint that came from a
    // different topology cannot silently misload).
    b.add_u32(KEY_N_BINS, DEFAULT_N_BINS);
    b.add_u32(KEY_HIDDEN_DIM, DEFAULT_HIDDEN_DIM);
    b.add_u32(KEY_FC1_DIM, DEFAULT_FC1_DIM);
    b.add_u32(KEY_FC2_DIM, DEFAULT_FC2_DIM);
    b.add_u32(KEY_N_FFT, DEFAULT_N_FFT);
    b.add_u32(KEY_HOP, DEFAULT_HOP);
    b.add_u32(KEY_WIN_LENGTH, DEFAULT_WIN_LENGTH);
    b.add_u32(KEY_SAMPLE_RATE, DEFAULT_SAMPLE_RATE);

    if st.tensors().len() != TENSOR_MAP.len() {
        return Err(ConvertError::Parse(format!(
            "nsnet2: official initializer manifest has {} tensors, input has {}",
            TENSOR_MAP.len(),
            st.tensors().len(),
        )));
    }
    for tensor in st.tensors() {
        if !TENSOR_MAP
            .iter()
            .any(|mapping| mapping.source == tensor.name)
        {
            return Err(ConvertError::Parse(format!(
                "nsnet2: unexpected initializer `{}`; refusing a different topology",
                tensor.name,
            )));
        }
    }

    let mut report = Nsnet2Report {
        read: st.tensors().len(),
        ..Nsnet2Report::default()
    };
    for mapping in TENSOR_MAP {
        let tensor = st.tensor_info(mapping.source).ok_or_else(|| {
            ConvertError::Parse(format!(
                "nsnet2: missing official initializer `{}`",
                mapping.source,
            ))
        })?;
        if tensor.dtype != GgmlType::F32 {
            return Err(ConvertError::Parse(format!(
                "nsnet2: initializer `{}` has dtype {:?}, expected F32",
                mapping.source, tensor.dtype,
            )));
        }
        if tensor.shape.as_slice() != mapping.source_shape {
            return Err(ConvertError::Parse(format!(
                "nsnet2: initializer `{}` has shape {:?}, expected {:?}",
                mapping.source, tensor.shape, mapping.source_shape,
            )));
        }
        let source = st.tensor_bytes(tensor);
        let payload = if mapping.transpose_2d {
            transpose_f32_bytes(
                source,
                mapping.source_shape[0] as usize,
                mapping.source_shape[1] as usize,
            )?
        } else {
            source.to_vec()
        };
        b.add_tensor(
            mapping.target,
            GgmlType::F32,
            mapping.target_shape.to_vec(),
            payload,
        )?;
        report.written += 1;
    }

    let out_bytes = b
        .to_bytes()
        .map_err(|e| ConvertError::Parse(e.to_string()))?;
    // A replacement dry-run must never clobber an existing public artifact or
    // a concurrent writer's result. The VAST worker supplies an absent work
    // directory, while this helper also protects direct converter callers.
    write_no_replace(output, &out_bytes).map_err(ConvertError::Io)?;
    Ok(report)
}

/// Publish bytes to an absent path without replacement, using a temporary
/// sibling and a same-filesystem hard link as the portable std-only
/// no-replace primitive. The link operation is atomic and fails if another
/// writer has claimed `output`; cleanup runs for both conversion and publish
/// errors.
fn write_no_replace(output: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let parent = output.parent().unwrap_or_else(|| Path::new("."));
    let name = output.file_name().ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::InvalidInput, "output has no file name")
    })?;
    let mut temporary = None;
    for attempt in 0..100u32 {
        let candidate = parent.join(format!(
            ".{}.vokra-nsnet2-{}-{}",
            name.to_string_lossy(),
            std::process::id(),
            attempt
        ));
        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&candidate)
        {
            Ok(file) => {
                temporary = Some((candidate, file));
                break;
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error),
        }
    }
    let Some((temporary_path, mut temporary_file)) = temporary else {
        return Err(std::io::Error::new(
            std::io::ErrorKind::AlreadyExists,
            "could not allocate a unique NSNet2 temporary output",
        ));
    };

    let operation = (|| {
        temporary_file.write_all(bytes)?;
        temporary_file.flush()?;
        temporary_file.sync_all()?;
        drop(temporary_file);
        std::fs::hard_link(&temporary_path, output)
    })();
    let cleanup = std::fs::remove_file(&temporary_path);
    match operation {
        Err(error) => {
            let _ = cleanup;
            Err(error)
        }
        Ok(()) => cleanup,
    }
}

fn transpose_f32_bytes(bytes: &[u8], rows: usize, cols: usize) -> Result<Vec<u8>, ConvertError> {
    let expected = rows
        .checked_mul(cols)
        .and_then(|elements| elements.checked_mul(4))
        .ok_or_else(|| ConvertError::Parse("nsnet2: matrix byte length overflow".to_owned()))?;
    if bytes.len() != expected {
        return Err(ConvertError::Parse(format!(
            "nsnet2: matrix payload has {} bytes, expected {expected}",
            bytes.len(),
        )));
    }
    let mut output = vec![0u8; bytes.len()];
    for row in 0..rows {
        for col in 0..cols {
            let source = (row * cols + col) * 4;
            let target = (col * rows + row) * 4;
            output[target..target + 4].copy_from_slice(&bytes[source..source + 4]);
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use vokra_core::gguf::GgufFile;

    /// Per-test unique scratch path (PID + nanosecond timestamp — the
    /// emotion2vec / ecapa_tdnn test pattern; no external `tempfile` dep,
    /// preserving zero-dep NFR-DS-02).
    fn scratch_path(tag: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "vokra-nsnet2-{}-{}-{}.bin",
            tag,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or_default(),
        ));
        p
    }

    /// Builds the exact official 14-initializer manifest with synthetic F32
    /// payloads. Only `172` (`fc_in.weight`) carries non-zero sentinels; all
    /// other tensors are zero-filled.
    fn synthetic_f32_safetensors() -> (Vec<u8>, Vec<u8>) {
        let mut entries = Vec::with_capacity(TENSOR_MAP.len());
        let mut payload = Vec::new();
        let mut expected_fc_in = Vec::new();
        for mapping in TENSOR_MAP {
            let start = payload.len();
            let elements = mapping
                .source_shape
                .iter()
                .try_fold(1usize, |acc, &dim| acc.checked_mul(dim as usize))
                .expect("synthetic tensor element count");
            let mut tensor = vec![0u8; elements * 4];
            if mapping.source == "172" {
                for (row, col, value) in [
                    (0usize, 0usize, 1.0f32),
                    (0, 399, -2.5),
                    (7, 23, 0.15625),
                    (160, 0, 3.5),
                    (160, 399, 42.0),
                ] {
                    let offset = (row * 400 + col) * 4;
                    tensor[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
                }
                expected_fc_in = transpose_f32_bytes(&tensor, 161, 400).unwrap();
            }
            payload.extend_from_slice(&tensor);
            let shape = mapping
                .source_shape
                .iter()
                .map(u64::to_string)
                .collect::<Vec<_>>()
                .join(",");
            entries.push(format!(
                "\"{}\":{{\"dtype\":\"F32\",\"shape\":[{}],\"data_offsets\":[{},{}]}}",
                mapping.source,
                shape,
                start,
                payload.len(),
            ));
        }
        let header = format!("{{{}}}", entries.join(","));
        let mut buf = Vec::new();
        buf.extend_from_slice(&(header.len() as u64).to_le_bytes());
        buf.extend_from_slice(header.as_bytes());
        buf.extend_from_slice(&payload);
        (buf, expected_fc_in)
    }

    /// Builds an intentionally invalid one-tensor BF16 checkpoint. The strict
    /// official-manifest converter must reject it before emitting a GGUF.
    fn synthetic_bf16_safetensors() -> (Vec<u8>, Vec<u8>) {
        let values: [f32; 6] = [1.0, -2.5, 0.15625, 3.5, -0.5, 42.0];
        let bf16: Vec<u8> = values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        assert_eq!(bf16.len(), 12, "6 elements x 2 bytes BF16 payload");
        let header = r#"{"gru_2.W":{"dtype":"BF16","shape":[2,3],"data_offsets":[0,12]}}"#;
        let mut buf = Vec::new();
        buf.extend_from_slice(&(header.len() as u64).to_le_bytes());
        buf.extend_from_slice(header.as_bytes());
        buf.extend_from_slice(&bf16);
        (buf, bf16)
    }

    #[test]
    fn output_publish_is_atomic_and_no_replace() {
        let directory = scratch_path("atomic-output-dir");
        std::fs::create_dir(&directory).expect("create atomic output test directory");
        let output = directory.join("model.gguf");
        std::fs::write(&output, b"original").expect("write existing output");

        let error = write_no_replace(&output, b"replacement")
            .expect_err("existing output must reject replacement");
        assert_eq!(error.kind(), std::io::ErrorKind::AlreadyExists);
        assert_eq!(
            std::fs::read(&output).expect("read existing output"),
            b"original"
        );

        let fresh = directory.join("fresh.gguf");
        write_no_replace(&fresh, b"complete payload").expect("publish fresh output");
        assert_eq!(
            std::fs::read(&fresh).expect("read fresh output"),
            b"complete payload"
        );
        let entries = std::fs::read_dir(&directory)
            .expect("read test directory")
            .collect::<std::io::Result<Vec<_>>>()
            .expect("collect test directory entries");
        assert_eq!(entries.len(), 2, "temporary sibling must be cleaned up");

        std::fs::remove_dir_all(&directory).expect("remove atomic output test directory");
    }

    /// Exact-manifest pin: official numeric initializer names are converted
    /// to semantic runtime names, MatMul layout is transposed, and provenance
    /// / topology chunks land on the artifact.
    #[test]
    fn official_manifest_converts_to_runtime_schema() {
        let (input_bytes, payload) = synthetic_f32_safetensors();
        let input = scratch_path("f32-in");
        let output = scratch_path("f32-out");
        std::fs::write(&input, &input_bytes).expect("write safetensors input");

        let report = convert_nsnet2_file(&input, &output, None).expect("convert");

        assert_eq!(report.read, 14, "official initializer count");
        assert_eq!(report.written, 14, "every official tensor is written");
        assert_eq!(
            report.skipped_non_float, 0,
            "F32 must not land in the skipped counter"
        );
        assert_eq!(
            report.bf16_passthrough, 0,
            "F32-only input must leave the BF16 subset counter at Default 0"
        );

        let out_bytes = std::fs::read(&output).expect("read gguf output");
        let file = GgufFile::parse(out_bytes).expect("parse gguf");
        let info = file
            .tensor_info("fc_in.weight")
            .expect("numeric initializer 172 renamed to fc_in.weight");
        assert_eq!(info.dtype, GgmlType::F32, "F32 stays F32");
        assert_eq!(info.dimensions, vec![400, 161]);
        assert_eq!(
            file.tensor_bytes(info),
            payload.as_slice(),
            "[161,400] MatMul matrix must be transposed to [400,161]"
        );

        // Provenance + category chunks pinned on the artifact itself.
        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
            Some(ARCH)
        );
        assert_eq!(
            file.get(chunks::KEY_MODEL_NAME).and_then(|v| v.as_str()),
            Some(NAME)
        );
        assert_eq!(
            file.get(KEY_MODEL_CATEGORY).and_then(|v| v.as_str()),
            Some(CATEGORY),
            "category chunk pins NSNet2 as `enhancement`"
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some(DEFAULT_LICENSE)
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::AttributionRequired.as_str()),
            "CC-BY-4.0 weights normalise to LicenseClass::AttributionRequired"
        );
        assert_eq!(
            file.get(KEY_PROVENANCE_UPSTREAM_URL)
                .and_then(|v| v.as_str()),
            Some(UPSTREAM_URL),
            "upstream_url chunk pins the GitHub tree the release ships from"
        );
        // Every `vokra.nsnet2.*` hparam must be stamped verbatim so a
        // downstream `Nsnet2V1::from_gguf` binder validates the topology.
        for (k, want) in [
            (KEY_N_BINS, DEFAULT_N_BINS),
            (KEY_HIDDEN_DIM, DEFAULT_HIDDEN_DIM),
            (KEY_FC1_DIM, DEFAULT_FC1_DIM),
            (KEY_FC2_DIM, DEFAULT_FC2_DIM),
            (KEY_N_FFT, DEFAULT_N_FFT),
            (KEY_HOP, DEFAULT_HOP),
            (KEY_WIN_LENGTH, DEFAULT_WIN_LENGTH),
            (KEY_SAMPLE_RATE, DEFAULT_SAMPLE_RATE),
        ] {
            let got = file.get(k).and_then(|v| v.as_u64());
            assert_eq!(
                got,
                Some(u64::from(want)),
                "hparam `{k}` must be stamped as {want}"
            );
        }
        // Schema stamp is written unconditionally by the GGUF writer.
        assert!(
            file.get(chunks::KEY_SCHEMA_VERSION).is_some(),
            "vokra.schema.version must be stamped"
        );
        assert!(
            file.get(chunks::KEY_SCHEMA_PRODUCER).is_some(),
            "vokra.schema.producer must be stamped"
        );

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }

    /// A one-tensor or retyped checkpoint is not the official topology and
    /// must be rejected before an unusable GGUF is emitted.
    #[test]
    fn non_official_manifest_is_rejected() {
        let (input_bytes, _bf16_payload) = synthetic_bf16_safetensors();
        let input = scratch_path("bf16-in");
        let output = scratch_path("bf16-out");
        std::fs::write(&input, &input_bytes).expect("write safetensors input");

        let err = convert_nsnet2_file(&input, &output, None).unwrap_err();
        assert!(
            format!("{err}").contains("official initializer manifest"),
            "strict manifest error must explain the rejection: {err}"
        );

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }

    /// Licence override pin: passing `Some("mit")` re-derives the
    /// class through `LicenseClass::from_license_str` and stamps the new
    /// SPDX + class on the artifact. Guards against a hard-coded
    /// class instead of retaining the attribution-required default. This is
    /// an API-mechanics test; callers still need a checkpoint-specific grant.
    #[test]
    fn license_override_re_derives_class() {
        let (input_bytes, _payload) = synthetic_f32_safetensors();
        let input = scratch_path("override-in");
        let output = scratch_path("override-out");
        std::fs::write(&input, &input_bytes).expect("write safetensors input");

        let _report =
            convert_nsnet2_file(&input, &output, Some("mit")).expect("convert with override");

        let out_bytes = std::fs::read(&output).expect("read gguf output");
        let file = GgufFile::parse(out_bytes).expect("parse gguf");
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some("mit"),
            "override SPDX lands verbatim"
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::Permissive.as_str()),
            "MIT override normalises to LicenseClass::Permissive"
        );

        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
    }
}
