//! Real-checkpoint parity for aiola/whisper-medusa-v1 module 0.
//!
//! `tools/parity/whisper_medusa/dump_reference.py` imports the pinned official
//! source and emits the committed fixture directory.  The real GGUF remains
//! opt-in; an explicit reference override is useful for regeneration audits.

use std::path::{Path, PathBuf};

use vokra_models::whisper_medusa::WhisperMedusa;

const GGUF_ENV: &str = "VOKRA_WHISPER_MEDUSA_GGUF";
const REFERENCE_ENV: &str = "VOKRA_WHISPER_MEDUSA_REFERENCE";
const PREFIX: [u32; 4] = [50258, 50259, 50359, 50363];
const LOGITS_ATOL: f32 = 5.0e-4;

fn read_f32(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|error| panic!("read {path:?}: {error}"));
    assert_eq!(bytes.len() % 4, 0, "{path:?}: partial f32");
    bytes
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
        .collect()
}

fn read_u32(path: &Path) -> Vec<u32> {
    let bytes = std::fs::read(path).unwrap_or_else(|error| panic!("read {path:?}: {error}"));
    assert_eq!(bytes.len() % 4, 0, "{path:?}: partial u32");
    bytes
        .chunks_exact(4)
        .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()))
        .collect()
}

fn paths() -> Option<(PathBuf, PathBuf)> {
    match (std::env::var_os(GGUF_ENV), std::env::var_os(REFERENCE_ENV)) {
        (None, None) => {
            eprintln!("skip: set {GGUF_ENV} for official Whisper-Medusa parity");
            None
        }
        (None, Some(_)) => panic!("{REFERENCE_ENV} cannot opt in without {GGUF_ENV}"),
        (Some(gguf), reference) => Some((
            gguf.into(),
            reference.map_or_else(
                || PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/whisper_medusa"),
                PathBuf::from,
            ),
        )),
    }
}

#[test]
fn official_module_zero_logits_and_greedy_tokens() {
    let Some((gguf_path, reference)) = paths() else {
        return;
    };
    let file = vokra_mmap::open_gguf(&gguf_path)
        .unwrap_or_else(|error| panic!("mmap {gguf_path:?}: {error}"));
    let model = WhisperMedusa::from_gguf(&file).expect("strict official bind");
    assert_eq!(model.num_heads(), 10);
    assert_eq!(model.module_count(), 11);

    let pcm = read_f32(&reference.join("pcm.f32"));
    let expected_logits = read_f32(&reference.join("prefix_logits.f32"));
    let actual_logits = model
        .prefix_logits(&pcm, &PREFIX)
        .expect("official module-0 prefix logits");
    assert_eq!(actual_logits.len(), expected_logits.len());
    let (max_index, max_abs) = actual_logits
        .iter()
        .zip(&expected_logits)
        .enumerate()
        .map(|(index, (&actual, &expected))| (index, (actual - expected).abs()))
        .max_by(|left, right| left.1.total_cmp(&right.1))
        .expect("non-empty logits");
    eprintln!("Whisper-Medusa module-0 logits max_abs={max_abs:.9e}@{max_index}");
    assert!(
        max_abs <= LOGITS_ATOL,
        "module-0 logits max_abs={max_abs:.9e}@{max_index} exceeds {LOGITS_ATOL:.9e}"
    );

    #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
    {
        use vokra_core::{BackendKind, VokraError};

        let metal_model = WhisperMedusa::from_gguf(&file)
            .expect("strict official Metal bind")
            .with_backend(BackendKind::Metal);
        match metal_model.prefix_logits(&pcm, &PREFIX) {
            Ok(metal_logits) => {
                assert_eq!(metal_logits.len(), actual_logits.len());
                let (metal_index, metal_max_abs) = metal_logits
                    .iter()
                    .zip(&actual_logits)
                    .enumerate()
                    .map(|(index, (&metal, &cpu))| (index, (metal - cpu).abs()))
                    .max_by(|left, right| left.1.total_cmp(&right.1))
                    .expect("non-empty Metal logits");
                eprintln!(
                    "Whisper-Medusa CPU/Metal module-0 logits max_abs={metal_max_abs:.9e}@{metal_index}"
                );
                assert!(
                    metal_max_abs <= 0.01,
                    "CPU/Metal module-0 max_abs={metal_max_abs:.9e}@{metal_index} exceeds 1e-2"
                );
                let cpu_argmax = actual_logits
                    .iter()
                    .enumerate()
                    .max_by(|left, right| left.1.total_cmp(right.1))
                    .map(|(index, _)| index);
                let metal_argmax = metal_logits
                    .iter()
                    .enumerate()
                    .max_by(|left, right| left.1.total_cmp(right.1))
                    .map(|(index, _)| index);
                assert_eq!(metal_argmax, cpu_argmax, "CPU/Metal argmax mismatch");
            }
            Err(VokraError::BackendUnavailable(error)) => {
                eprintln!("skip Whisper-Medusa Metal parity: {error}");
            }
            Err(error) => panic!("Whisper-Medusa Metal execution failed: {error}"),
        }
    }

    let expected_tokens = read_u32(&reference.join("greedy_tokens.u32"));
    assert!(
        !expected_tokens.is_empty(),
        "official greedy fixture must contain at least one token"
    );
    let mut decoder_ids = PREFIX.to_vec();
    let mut actual_tokens = Vec::with_capacity(expected_tokens.len());
    let first_token = actual_logits
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .map(|(index, _)| index as u32)
        .expect("non-empty vocabulary logits");
    actual_tokens.push(first_token);
    decoder_ids.push(first_token);
    for _ in 1..expected_tokens.len() {
        let logits = model
            .prefix_logits(&pcm, &decoder_ids)
            .expect("official module-0 bounded greedy step");
        let token = logits
            .iter()
            .enumerate()
            .max_by(|left, right| left.1.total_cmp(right.1))
            .map(|(index, _)| index as u32)
            .expect("non-empty vocabulary logits");
        actual_tokens.push(token);
        decoder_ids.push(token);
    }
    assert_eq!(actual_tokens, expected_tokens, "exact greedy token parity");
}
