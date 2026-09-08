//! **CLAP** (LAION contrastive language-audio pretraining):
//! safetensors → GGUF conversion (TIER 1 F wave, 2026-07-30).
//!
//! Input: the upstream `laion/clap-htsat-fused` release — a
//! Contrastive Language-Audio Pretraining model with an HTSAT audio
//! encoder + text encoder trained contrastively (Wu et al. 2023 —
//! `arXiv:2211.06687`). One of the highest-download HF audio releases
//! (8.1M+). Output: a GGUF carrying every F32 / F16 / BF16 tensor
//! verbatim plus the `vokra.provenance.*` / `vokra.model.*` metadata
//! chunks a future `vokra-models::clap::*` loader will read.
//!
//! # Provenance
//!
//! - **HF path**: `laion/clap-htsat-fused` (fetched 2026-07-30 —
//!   CLAUDE.md「ハルシネーション厳禁」).
//! - **SPDX**: `apache-2.0` (`LicenseClass::Permissive`).
//! - **Category**: `classification` (audio-text embedding — the model
//!   surface is "return an embedding compatible with the paired text
//!   encoder"; downstream users project into an N-way classification by
//!   picking a text prompt vocabulary).
//!
//! # BF16 pass-through
//!
//! Mirror of `wespeaker` / `neucodec` / `ecapa_tdnn`.
//!
//! # Tensor naming contract
//!
//! GGUF tensor names are the **upstream safetensors names verbatim**.
//! CLAP has a two-tower topology (audio encoder + text encoder + fused
//! projection); the pass-through preserves both towers so a future
//! `ClapWeights::from_gguf` can walk each side. Real-weight parity is
//! deferred to owner sign-off (loud-partial precedent — internal
//! HTSAT + text encoder forward will land behind
//! `VokraError::UnsupportedOp`).

use std::io::Write;
use std::path::{Component, Path};
use std::sync::atomic::{AtomicU64, Ordering};

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` — first `"clap"` in the converter tree.
/// Distinct from every sibling arch tag because CLAP's two-tower
/// contrastive topology (HTSAT audio encoder + text encoder + shared
/// embedding space) is unrelated to speaker-encoder / VAD / ASR / TTS
/// families.
pub const ARCH: &str = "clap";

pub const NAME: &str = "clap-htsat-fused";
pub const CATEGORY: &str = "classification";
pub const UPSTREAM_HF: &str = "laion/clap-htsat-fused";
/// Immutable Hugging Face revision used by the reference-side inspection
/// contract.  A model name alone is not sufficient provenance for a
/// two-tower checkpoint: the upstream files and state-dict topology can
/// change while the repository slug remains stable.
pub const UPSTREAM_REVISION: &str = "365dea6ef167def6676140ed93bbc43f84dabb28";
pub const DEFAULT_LICENSE_SPDX: &str = "apache-2.0";
static OUTPUT_SEQUENCE: AtomicU64 = AtomicU64::new(0);

const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";
const KEY_PROVENANCE_UPSTREAM_REVISION: &str = "vokra.provenance.upstream_revision";

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct ClapReport {
    pub read: usize,
    pub written: usize,
    pub skipped_non_float: usize,
    pub bf16_passthrough: usize,
}

pub fn convert_clap_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<ClapReport, ConvertError> {
    ensure_file_identity_support()?;
    validate_conversion_paths(input, output)?;
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
    vokra_core::stamp_provenance(
        &mut b,
        class,
        &spdx,
        Some(NAME),
        Some(
            "laion/clap-htsat-fused (contrastive language-audio pretraining, \
             HTSAT audio encoder + text encoder, apache-2.0)",
        ),
    );
    b.add_string(KEY_PROVENANCE_UPSTREAM_HF, UPSTREAM_HF);
    b.add_string(KEY_PROVENANCE_UPSTREAM_REVISION, UPSTREAM_REVISION);

    let mut report = ClapReport::default();
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
    write_no_clobber(output, &out_bytes)?;
    Ok(report)
}

fn validate_conversion_paths(input: &Path, output: &Path) -> Result<(), ConvertError> {
    for (path, label) in [(input, "input"), (output, "output")] {
        reject_unsafe_path(path, label)?;
    }
    if input.is_symlink() || !input.is_file() {
        return Err(ConvertError::Usage(format!(
            "CLAP input must be a regular non-symlink file: {}",
            input.display()
        )));
    }
    if output.exists() || output.is_symlink() {
        return Err(ConvertError::Usage(format!(
            "CLAP output already exists or is symlinked: {}",
            output.display()
        )));
    }
    let parent = output.parent().ok_or_else(|| {
        ConvertError::Usage("CLAP output must have a parent directory".to_owned())
    })?;
    if !parent.is_dir() || parent.is_symlink() {
        return Err(ConvertError::Usage(format!(
            "CLAP output parent must be an existing regular non-symlink directory: {}",
            parent.display()
        )));
    }
    Ok(())
}

fn reject_unsafe_path(path: &Path, label: &str) -> Result<(), ConvertError> {
    let raw = path.to_string_lossy();
    #[cfg(windows)]
    let has_lexical_dot = raw
        .split(['/', '\\'])
        .any(|part| matches!(part, "." | ".."));
    #[cfg(not(windows))]
    let has_lexical_dot = raw.split('/').any(|part| matches!(part, "." | ".."));
    if has_lexical_dot
        || path
            .components()
            .any(|component| matches!(component, Component::CurDir | Component::ParentDir))
    {
        return Err(ConvertError::Usage(format!(
            "CLAP {label} must not contain lexical dot components"
        )));
    }
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .map_err(ConvertError::Io)?
            .join(path)
    };
    let mut current = absolute.as_path();
    loop {
        if current.is_symlink() && current != Path::new("/var") {
            return Err(ConvertError::Usage(format!(
                "CLAP {label} has symlink ancestry: {}",
                current.display()
            )));
        }
        let Some(parent) = current.parent() else {
            break;
        };
        if parent == current {
            break;
        }
        current = parent;
    }
    Ok(())
}

fn write_no_clobber(destination: &Path, payload: &[u8]) -> Result<(), ConvertError> {
    validate_publish_path(destination)?;
    ensure_file_identity_support()?;
    let parent = destination.parent().expect("validated output parent");
    let name = destination
        .file_name()
        .ok_or_else(|| ConvertError::Usage("CLAP output must name a file".to_owned()))?;
    for _ in 0..32 {
        let sequence = OUTPUT_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let temporary = parent.join(format!(
            ".{}.tmp-{}-{}",
            name.to_string_lossy(),
            std::process::id(),
            sequence
        ));
        let mut file = match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
        {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(ConvertError::Io(error)),
        };
        if let Err(error) = file.write_all(payload).and_then(|_| file.sync_all()) {
            remove_owned_temp(&file, &temporary);
            return Err(ConvertError::Io(error));
        }
        if let Err(error) = validate_publish_path(destination) {
            remove_owned_temp(&file, &temporary);
            return Err(error);
        }
        if !same_file_identity(&file, &temporary) {
            remove_owned_temp(&file, &temporary);
            return Err(ConvertError::Usage(
                "CLAP temporary output was replaced before publication".to_owned(),
            ));
        }
        if let Err(error) = std::fs::hard_link(&temporary, destination) {
            remove_owned_temp(&file, &temporary);
            return Err(ConvertError::Io(error));
        }
        if !same_file_identity(&file, destination) {
            // Do not remove the destination: it may be an unowned file
            // linked by a replacement race in the verify→link window.
            remove_owned_temp(&file, &temporary);
            return Err(ConvertError::Usage(
                "CLAP published output failed regular-file identity verification".to_owned(),
            ));
        }
        // The final link is already published.  Cleanup is best-effort and
        // identity-guarded so a replacement temp is never unlinked and a
        // cleanup failure cannot turn a successful publication into Err.
        remove_owned_temp(&file, &temporary);
        return Ok(());
    }
    Err(ConvertError::Usage(
        "CLAP exhausted temporary output candidates".to_owned(),
    ))
}

fn ensure_file_identity_support() -> Result<(), ConvertError> {
    #[cfg(unix)]
    {
        Ok(())
    }
    #[cfg(not(unix))]
    {
        Err(ConvertError::Usage(
            "CLAP atomic publication requires a supported file identity API on this platform"
                .to_owned(),
        ))
    }
}

fn same_file_identity(file: &std::fs::File, path: &Path) -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;

        let Ok(expected) = file.metadata() else {
            return false;
        };
        let Ok(actual) = std::fs::symlink_metadata(path) else {
            return false;
        };
        actual.file_type().is_file()
            && expected.dev() == actual.dev()
            && expected.ino() == actual.ino()
    }
    #[cfg(not(unix))]
    {
        let _ = (file, path);
        false
    }
}

fn remove_owned_temp(file: &std::fs::File, path: &Path) {
    if same_file_identity(file, path) {
        let _ = std::fs::remove_file(path);
    }
}

fn validate_publish_path(destination: &Path) -> Result<(), ConvertError> {
    reject_unsafe_path(destination, "output")?;
    let parent = destination.parent().ok_or_else(|| {
        ConvertError::Usage("CLAP output must have a parent directory".to_owned())
    })?;
    if !parent.is_dir() || parent.is_symlink() {
        return Err(ConvertError::Usage(format!(
            "CLAP output parent must be an existing regular non-symlink directory: {}",
            parent.display()
        )));
    }
    if destination.is_symlink() {
        return Err(ConvertError::Usage(format!(
            "CLAP output must not be a symlink: {}",
            destination.display()
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(unix)]
    use vokra_core::gguf::GgufFile;

    #[cfg(unix)]
    fn scratch_path(tag: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "vokra-clap-{tag}-{}-{}.bin",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.subsec_nanos())
                .unwrap_or(0)
        ));
        p
    }

    #[cfg(unix)]
    fn safetensors_two_towers(
        audio_name: &str,
        audio_bf16: &[u8],
        text_name: &str,
        text_f32: &[u8],
    ) -> Vec<u8> {
        // CLAP is two-tower: pin both an audio-encoder and a text-encoder
        // tensor in the same fixture so the round-trip proves the
        // pass-through doesn't collapse one tower.
        let audio_len = audio_bf16.len();
        let total = audio_len + text_f32.len();
        let header = format!(
            r#"{{"{audio_name}":{{"dtype":"BF16","shape":[2,3],"data_offsets":[0,{audio_len}]}},"{text_name}":{{"dtype":"F32","shape":[2,3],"data_offsets":[{audio_len},{total}]}}}}"#
        );
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(audio_bf16);
        out.extend_from_slice(text_f32);
        out
    }

    #[test]
    fn lexical_dot_paths_are_rejected_on_every_platform() {
        let root = std::env::temp_dir().join(format!(
            "vokra-clap-dot-paths-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .subsec_nanos()
        ));
        std::fs::create_dir_all(&root).expect("root");
        assert!(reject_unsafe_path(&root.join(".").join("artifact.gguf"), "output").is_err());
        assert!(reject_unsafe_path(&root.join("..").join("artifact.gguf"), "output").is_err());
        std::fs::remove_dir_all(root).expect("cleanup");
    }

    #[cfg(unix)]
    #[test]
    fn both_towers_pass_through_verbatim() {
        // Audio tower (HTSAT) — BF16.
        let audio_vals: [f32; 6] = [1.0, -2.5, 0.15625, 3.5, -0.5, 42.0];
        let audio_bf16: Vec<u8> = audio_vals
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        // Text tower — F32.
        let text_vals: [f32; 6] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let text_bytes: Vec<u8> = text_vals.iter().flat_map(|v| v.to_le_bytes()).collect();

        let input_bytes = safetensors_two_towers(
            "audio_encoder.htsat.blocks.0.attn.qkv.weight",
            &audio_bf16,
            "text_encoder.blocks.0.attn.qkv.weight",
            &text_bytes,
        );
        let input = scratch_path("two-tower-in");
        let output = scratch_path("two-tower-out");
        std::fs::write(&input, &input_bytes).expect("write");

        let report = convert_clap_file(&input, &output, None).expect("convert");
        assert_eq!(report.read, 2, "two tensors — one per tower");
        assert_eq!(report.written, 2, "both towers survive pass-through");
        assert_eq!(report.bf16_passthrough, 1, "only audio tower is BF16");

        let out = std::fs::read(&output).expect("read");
        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();
        let file = GgufFile::parse(out).expect("parse");

        let audio_info = file
            .tensor_info("audio_encoder.htsat.blocks.0.attn.qkv.weight")
            .expect("audio tower tensor present");
        assert_eq!(audio_info.dtype, GgmlType::BF16, "BF16 audio tower");
        assert_eq!(file.tensor_bytes(audio_info), audio_bf16.as_slice());

        let text_info = file
            .tensor_info("text_encoder.blocks.0.attn.qkv.weight")
            .expect("text tower tensor present");
        assert_eq!(text_info.dtype, GgmlType::F32, "F32 text tower");
        assert_eq!(file.tensor_bytes(text_info), text_bytes.as_slice());

        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
            Some(ARCH)
        );
        assert_eq!(
            file.get(KEY_MODEL_CATEGORY).and_then(|v| v.as_str()),
            Some(CATEGORY)
        );
        assert_eq!(
            file.get(KEY_PROVENANCE_UPSTREAM_HF)
                .and_then(|v| v.as_str()),
            Some(UPSTREAM_HF)
        );
        assert_eq!(
            file.get(KEY_PROVENANCE_UPSTREAM_REVISION)
                .and_then(|v| v.as_str()),
            Some(UPSTREAM_REVISION)
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some(DEFAULT_LICENSE_SPDX)
        );
    }

    #[cfg(unix)]
    #[test]
    fn output_publication_is_no_clobber_and_path_bounded() {
        let root = std::env::temp_dir().join(format!(
            "vokra-clap-output-boundary-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .subsec_nanos()
        ));
        std::fs::create_dir_all(&root).expect("root");
        let output = root.join("artifact.gguf");
        write_no_clobber(&output, b"first").expect("first publication");
        let second = write_no_clobber(&output, b"second").expect_err("clobber accepted");
        assert!(matches!(
            second,
            ConvertError::Io(ref error)
                if error.kind() == std::io::ErrorKind::AlreadyExists
        ));
        assert_eq!(
            std::fs::read(&output).expect("published artifact"),
            b"first"
        );
        #[cfg(unix)]
        {
            use std::os::unix::fs::symlink;
            let real = root.join("real");
            let linked = root.join("linked");
            std::fs::create_dir(&real).expect("real");
            symlink(&real, &linked).expect("symlink");
            assert!(validate_publish_path(&linked.join("artifact.gguf")).is_err());
        }
        std::fs::remove_dir_all(root).expect("cleanup");
    }

    #[cfg(not(unix))]
    #[test]
    fn non_unix_publication_fails_closed_without_touching_output() {
        let root = std::env::temp_dir().join(format!(
            "vokra-clap-unsupported-publication-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .subsec_nanos()
        ));
        std::fs::create_dir_all(&root).expect("root");
        let output = root.join("artifact.gguf");

        let input = root.join("input.safetensors");
        let error = convert_clap_file(&input, &output, None).expect_err("unsupported conversion");
        assert!(matches!(error, ConvertError::Usage(_)));
        assert!(!output.exists(), "unsupported publication created output");

        std::fs::write(&input, b"not a safetensors file").expect("input");
        std::fs::write(&output, b"existing").expect("existing output");
        let error = convert_clap_file(&input, &output, None).expect_err("unsupported conversion");
        assert!(matches!(error, ConvertError::Usage(_)));
        assert_eq!(
            std::fs::read(&output).expect("read preserved output"),
            b"existing"
        );
        std::fs::remove_dir_all(root).expect("cleanup");
    }

    #[cfg(unix)]
    #[test]
    fn replacement_temp_is_not_removed_by_owned_cleanup() {
        let root = std::env::temp_dir().join(format!(
            "vokra-clap-temp-race-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .subsec_nanos()
        ));
        std::fs::create_dir_all(&root).expect("root");
        let temporary = root.join("artifact.tmp");
        std::fs::write(&temporary, b"owned").expect("temporary");
        let file = std::fs::OpenOptions::new()
            .read(true)
            .open(&temporary)
            .expect("open owned temp");
        let destination = root.join("destination");
        std::fs::hard_link(&temporary, &destination).expect("publish owned temp");
        assert!(same_file_identity(&file, &destination));
        std::fs::remove_file(&destination).expect("remove destination");
        std::fs::write(&destination, b"destination replacement").expect("destination replacement");
        assert!(!same_file_identity(&file, &destination));
        std::fs::remove_file(&temporary).expect("remove owned path");
        std::fs::write(&temporary, b"replacement").expect("replacement");
        assert!(!same_file_identity(&file, &temporary));
        remove_owned_temp(&file, &temporary);
        assert_eq!(
            std::fs::read(&temporary).expect("replacement remains"),
            b"replacement"
        );
        std::fs::remove_dir_all(root).expect("cleanup");
    }
}
