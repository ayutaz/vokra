//! VAST-only real-weight VibeVoice-1.5B validation.
//!
//! This test is ignored by default and intentionally has no local fixture.
//! The VAST worker supplies an authenticated GGUF, caller-owned packet, and
//! independent Microsoft reference bundle.  Discrete output decisions are
//! exact-gated; unregistered PCM metrics are reported as MEASURED_NOT_GATED.

use std::path::{Path, PathBuf};

use vokra_core::backend::BackendKind;
use vokra_core::gguf::GgufFile;
use vokra_models::vibevoice::{VibeVoiceComposite, VibeVoiceGenerationPacket};

const SAMPLE_RATE: u32 = 24_000;
const LATENT_WIDTH: usize = 64;

fn required_path(name: &str) -> PathBuf {
    std::env::var_os(name)
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("set {name} in the VAST environment"))
}

fn bytes(path: &Path) -> Vec<u8> {
    std::fs::read(path).unwrap_or_else(|error| panic!("read {}: {error}", path.display()))
}

fn u32_rows(path: &Path) -> Vec<u32> {
    let raw = bytes(path);
    assert_eq!(raw.len() % 4, 0, "{} is not u32le", path.display());
    raw.chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

fn f32_rows(path: &Path) -> Vec<f32> {
    let raw = bytes(path);
    assert_eq!(raw.len() % 4, 0, "{} is not f32le", path.display());
    raw.chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

fn scalar_f32(path: &Path) -> f32 {
    let text = String::from_utf8(bytes(path)).expect("scalar UTF-8");
    text.trim().parse().expect("finite scalar f32")
}

fn scalar_usize(path: &Path) -> usize {
    let text = String::from_utf8(bytes(path)).expect("scalar UTF-8");
    text.trim().parse().expect("bounded scalar usize")
}

fn manifest_top_level_string(manifest: &str, key: &str) -> String {
    let prefix = format!("  \"{key}\": ");
    let rows: Vec<&str> = manifest
        .lines()
        .filter(|line| line.starts_with(&prefix))
        .collect();
    assert_eq!(
        rows.len(),
        1,
        "manifest key {key} must occur exactly once at top level"
    );
    let value = rows[0]
        .strip_prefix(&prefix)
        .unwrap()
        .trim_end_matches(',')
        .trim();
    let value = value
        .strip_prefix('"')
        .and_then(|value| value.strip_suffix('"'))
        .unwrap_or_else(|| panic!("manifest key {key} is not a JSON string"));
    value.to_owned()
}

fn manifest_contains_key(manifest: &str, key: &str) {
    assert!(
        manifest
            .lines()
            .any(|line| line.contains(&format!("\"{key}\""))),
        "reference missing key {key}"
    );
}

fn require_reference_files(reference: &Path) {
    const REQUIRED: &[&str] = &[
        "manifest.json",
        "packet.json",
        "token_ids.u32le",
        "prompt_pcm.f32le",
        "prompt_latent.f32le",
        "diffusion_initial.f32le",
        "diffusion_initial_native.f32le",
        "speech_input_mask.u8",
        "speech_masks.u8",
        "speech_replacement_positions.u32le",
        "generated_tokens.u32le",
        "guidance-scale.txt",
        "max-generated-tokens.txt",
        "official_pcm.f32le",
        "official_diffusion_latents.f32le",
    ];
    for name in REQUIRED {
        let path = reference.join(name);
        assert!(path.is_file(), "missing reference artifact {name}");
        assert!(
            std::fs::metadata(&path).unwrap().len() > 0,
            "empty reference artifact {name}"
        );
    }
    for entry in std::fs::read_dir(reference).expect("read reference directory") {
        let entry = entry.expect("reference directory entry");
        let file_type = entry.file_type().expect("reference artifact type");
        if file_type.is_file() {
            let name = entry.file_name();
            let name = name.to_str().expect("UTF-8 reference artifact name");
            assert!(
                REQUIRED.contains(&name),
                "unexpected/orphan reference artifact {name}"
            );
        } else {
            panic!("non-file reference artifact {}", entry.path().display());
        }
    }
}

#[test]
#[ignore = "requires fixed VAST GGUF/reference bundle"]
fn vibevoice_1_5b_real_cpu_matches_official_reference() {
    let gguf_path = required_path("VOKRA_VIBEVOICE_GGUF");
    let reference = required_path("VOKRA_VIBEVOICE_REFERENCE_DIR");
    let manifest = String::from_utf8(bytes(&reference.join("manifest.json")))
        .expect("reference manifest UTF-8");
    assert_eq!(manifest_top_level_string(&manifest, "status"), "BLOCKED");
    assert_eq!(
        manifest_top_level_string(&manifest, "evidence_stage"),
        "INSPECTION_ONLY"
    );
    assert_eq!(
        manifest_top_level_string(&manifest, "runtime_status"),
        "NOT_IMPLEMENTED_FAIL_CLOSED"
    );
    assert_eq!(
        manifest_top_level_string(&manifest, "cpu_status"),
        "UNSUPPORTED"
    );
    assert_eq!(
        manifest_top_level_string(&manifest, "metal_status"),
        "BLOCKED_BY_CPU"
    );
    assert_eq!(
        manifest_top_level_string(&manifest, "parity_status"),
        "NOT_RUN"
    );
    assert_eq!(
        manifest_top_level_string(&manifest, "reference_status"),
        "REFERENCE_EVIDENCE_COMPLETE"
    );
    assert_eq!(
        manifest_top_level_string(&manifest, "publication"),
        "NO_UPLOAD"
    );
    manifest_contains_key(&manifest, "random_draws_consumed");
    manifest_contains_key(&manifest, "callsite");
    manifest_contains_key(&manifest, "public_artifact");
    require_reference_files(&reference);

    let token_ids = u32_rows(&reference.join("token_ids.u32le"));
    assert!(!token_ids.is_empty());
    assert!(token_ids.iter().all(|token| *token < 151_936));
    let prompt_pcm = f32_rows(&reference.join("prompt_pcm.f32le"));
    assert!(!prompt_pcm.is_empty() && prompt_pcm.len() % 3_200 == 0);
    assert!(prompt_pcm.iter().all(|sample| sample.is_finite()));
    let prompt_latent_draws = f32_rows(&reference.join("prompt_latent.f32le"));
    assert_eq!(prompt_latent_draws.len() % LATENT_WIDTH, 0);
    let diffusion_flat = f32_rows(&reference.join("diffusion_initial.f32le"));
    assert_eq!(diffusion_flat.len() % LATENT_WIDTH, 0);
    let diffusion_initial_draws = f32_rows(&reference.join("diffusion_initial_native.f32le"))
        .chunks_exact(LATENT_WIDTH)
        .map(<[f32; LATENT_WIDTH]>::try_from)
        .map(|row| row.expect("width-64 diffusion draw").to_vec())
        .collect();
    let replacements = u32_rows(&reference.join("speech_replacement_positions.u32le"));
    let input_mask = bytes(&reference.join("speech_input_mask.u8"));
    assert_eq!(input_mask.len(), token_ids.len());
    let mask_positions: Vec<u32> = input_mask
        .iter()
        .enumerate()
        .filter_map(|(index, value)| (*value != 0).then_some(index as u32))
        .collect();
    assert_eq!(mask_positions, replacements);
    let speech_masks = bytes(&reference.join("speech_masks.u8"));
    assert_eq!(
        speech_masks.iter().filter(|value| **value != 0).count(),
        replacements.len()
    );
    let generated_reference = u32_rows(&reference.join("generated_tokens.u32le"));
    assert!(!generated_reference.is_empty());
    assert!(generated_reference.iter().all(|token| *token < 151_936));
    let guidance_scale = scalar_f32(&reference.join("guidance-scale.txt"));
    let max_generated_tokens = scalar_usize(&reference.join("max-generated-tokens.txt"));
    assert!(guidance_scale.is_finite());
    assert!(generated_reference.len() <= max_generated_tokens);
    manifest_contains_key(&manifest, "official_pcm");

    let packet = VibeVoiceGenerationPacket {
        token_ids,
        speech_replacement_positions: replacements
            .into_iter()
            .map(|value| value as usize)
            .collect(),
        prompt_pcm: Some(prompt_pcm),
        prompt_sample_rate_hz: SAMPLE_RATE,
        prompt_latent_draws,
        diffusion_initial_draws,
        guidance_scale,
        max_generated_tokens,
    };
    let backend = match std::env::var("VOKRA_VIBEVOICE_BACKEND").as_deref() {
        Ok("cpu") => BackendKind::Cpu,
        Ok("metal") => BackendKind::Metal,
        Ok(value) => panic!("unsupported VOKRA_VIBEVOICE_BACKEND={value}"),
        Err(_) => BackendKind::Cpu,
    };
    let file = GgufFile::open(&gguf_path).expect("open authenticated VibeVoice GGUF");
    let model = VibeVoiceComposite::from_gguf_with_backend(&file, backend)
        .expect("bind all authenticated VibeVoice components");
    let result = model
        .generate(&packet)
        .expect("native VibeVoice generation");
    assert_eq!(
        result.generated_tokens, generated_reference,
        "CPU discrete decisions differ"
    );
    let official_pcm = f32_rows(&reference.join("official_pcm.f32le"));
    assert_eq!(
        result.pcm.len(),
        official_pcm.len(),
        "native/official PCM lengths differ"
    );
    assert!(!result.pcm.is_empty() && result.pcm.iter().all(|sample| sample.is_finite()));
    assert!(official_pcm.iter().all(|sample| sample.is_finite()));
    let official_diffusion_latents = f32_rows(&reference.join("official_diffusion_latents.f32le"));
    assert!(!official_diffusion_latents.is_empty());
    assert_eq!(official_diffusion_latents.len() % LATENT_WIDTH, 0);
    assert!(
        official_diffusion_latents
            .iter()
            .all(|value| value.is_finite())
    );
    let max_abs = result
        .pcm
        .iter()
        .zip(&official_pcm)
        .map(|(native, official)| (native - official).abs())
        .fold(0.0_f32, f32::max);
    let (mut dot, mut native_norm, mut official_norm) = (0.0_f32, 0.0_f32, 0.0_f32);
    for (native, official) in result.pcm.iter().zip(&official_pcm) {
        dot += native * official;
        native_norm += native * native;
        official_norm += official * official;
    }
    let cosine = dot / (native_norm.sqrt() * official_norm.sqrt()).max(f32::MIN_POSITIVE);
    let backend_name = if backend == BackendKind::Cpu {
        "CPU"
    } else {
        "METAL"
    };
    println!("VIBEVOICE_{backend_name}_TOKENS_MEASURED exact=true");
    println!(
        "VIBEVOICE_{backend_name}_PCM_MEASURED samples={} max_abs={max_abs:.9e} cosine={cosine:.9e} status=MEASURED_NOT_GATED",
        result.pcm.len(),
    );
    println!(
        "VIBEVOICE_{backend_name}_OFFICIAL_DIFFUSION_LATENTS_CAPTURED samples={} status=MEASURED_NOT_GATED",
        official_diffusion_latents.len(),
    );
}
