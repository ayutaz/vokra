//! Guarded VibeVoice Realtime CLI route.
//!
//! Every artifact and owner/reference contract is authenticated before the
//! native runtime is bound. This route deliberately keeps the raw Microsoft
//! checkpoint digest separate from the derived GGUF image digest.
//!
//! The supplied Vokra HEAD/tree values are owner-packet bindings, not a claim
//! that this binary can self-attest its build provenance. An external packet
//! must therefore bind them to the reviewed source/runtime identity.

use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use vokra_core::Session;
use vokra_models::vibevoice_streaming::{
    VIBEVOICE_REALTIME_INFERENCE_STEPS, VibeVoiceRealtimePresetCache, VibeVoiceRealtimeRuntime,
    VibeVoiceRealtimeSynthesisConfig, VibeVoiceRealtimeSynthesisStep, VibeVoiceRealtimeTokenizer,
};

use crate::runtime_contracts::{
    VibeVoiceRealtimeNoiseTape, VibeVoiceRealtimeOwnerContract, VibeVoiceRealtimeOwnerInputs,
    load_vibevoice_realtime_noise_tape, sha256,
};
use crate::wav;

use super::RunArgs;

const MAX_TEXT_BYTES: u64 = 16 * 1024;
const MAX_PRESET_MANIFEST_BYTES: u64 = 256 * 1024;
const MAX_PRESET_CACHE_BYTES: u64 = 16 * 1024 * 1024;
const TOKENIZER_FILES: &[(&str, u64, &str)] = &[
    (
        "vocab.json",
        2_776_833,
        "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
    ),
    (
        "merges.txt",
        1_671_839,
        "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3",
    ),
    (
        "tokenizer_config.json",
        7_228,
        "c91efca15ceff6e9ee9424db58a6f59cd41294e550a86cbd07e3c1fb500b34f9",
    ),
    (
        "tokenizer.json",
        7_031_645,
        "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
    ),
];

/// Runs the fully authenticated Realtime path. The generic `--tokenizer`
/// option remains rejected by `run::main`; this route consumes only the
/// dedicated fixed tokenizer directory and owner-bound text file.
pub(super) fn run(session: &Session, args: &RunArgs) -> Result<(), String> {
    if args.tokenizer.is_some() {
        return Err(
            "run (VibeVoice-Realtime): generic --tokenizer is unsupported; use the dedicated fixed tokenizer directory"
                .to_owned(),
        );
    }
    if args.text.is_some() {
        return Err(
            "run (VibeVoice-Realtime): generic --text is unsupported; use --realtime-text-file"
                .to_owned(),
        );
    }
    let (validated, mut runtime) =
        preflight_and_bind(preflight(session.gguf().file_bytes(), args), |_validated| {
            VibeVoiceRealtimeRuntime::from_gguf(session.gguf(), args.backend)
                .map_err(|error| format!("Realtime native bind: {error}"))
        })?;
    let config = VibeVoiceRealtimeSynthesisConfig {
        max_new_tokens: validated.owner.execution.max_new_tokens,
        max_speech_steps: validated.owner.execution.max_new_tokens,
        guidance_scale: validated.owner.execution.cfg_scale,
    };
    let mut synthesis = runtime
        .start_session(
            &validated.preset,
            &validated.tokenizer,
            &validated.text,
            config,
        )
        .map_err(|error| format!("Realtime session start: {error}"))?;

    let mut consumed_draws = 0usize;
    let mut steps = Vec::new();
    for draw in &validated.noise.draws {
        consumed_draws += 1;
        let step = synthesis
            .step(draw, false)
            .map_err(|error| format!("Realtime generation: {error}"))?;
        let step_finished = matches!(&step, VibeVoiceRealtimeSynthesisStep::Finished { .. });
        steps.push(step);
        if synthesis.is_finished() && !step_finished {
            // The runtime can return the terminal audio/drain event while
            // marking itself finished. The empty call is explicitly allowed
            // only in this already-finished state and consumes no draw.
            let terminal = synthesis
                .step(&[], false)
                .map_err(|error| format!("Realtime terminal state: {error}"))?;
            if !matches!(&terminal, VibeVoiceRealtimeSynthesisStep::Finished { .. }) {
                return Err("Realtime runtime did not produce Finished".to_owned());
            }
            steps.push(terminal);
        }
        if synthesis.is_finished() {
            break;
        }
    }
    let pcm = consume_step_sequence(steps, consumed_draws, validated.noise.len())?;

    wav::write_wav_create_new(&validated.output, &pcm, 24_000)
        .map_err(|error| format!("Realtime --output {}: {error}", validated.output.display()))?;
    println!(
        "vibevoice-realtime: wrote {} samples @ 24000 Hz -> {}",
        pcm.len(),
        validated.output.display()
    );
    Ok(())
}

struct ValidatedInputs {
    text: String,
    output: PathBuf,
    owner: VibeVoiceRealtimeOwnerContract,
    preset: VibeVoiceRealtimePresetCache,
    tokenizer: VibeVoiceRealtimeTokenizer,
    noise: VibeVoiceRealtimeNoiseTape,
}

/// Authenticates every external activation input before the native binder is
/// callable. The parsed GGUF bytes are passed directly from the mapped session;
/// this function never reopens or executes the model.
fn preflight(file_bytes: &[u8], args: &RunArgs) -> Result<ValidatedInputs, String> {
    let gguf_sha256 = required(&args.realtime_gguf_sha256, "--realtime-gguf-sha256")?;
    let reference_dir = required_path(&args.realtime_reference_dir, "--realtime-reference-dir")?;
    let reference_sha256 = required(
        &args.realtime_reference_sha256,
        "--realtime-reference-sha256",
    )?;
    let owner_scope = required_path(&args.realtime_owner_scope, "--realtime-owner-scope")?;
    let owner_scope_sha256 = required(
        &args.realtime_owner_scope_sha256,
        "--realtime-owner-scope-sha256",
    )?;
    let owner_canonical =
        required_path(&args.realtime_owner_canonical, "--realtime-owner-canonical")?;
    let owner_canonical_sha256 = required(
        &args.realtime_owner_canonical_sha256,
        "--realtime-owner-canonical-sha256",
    )?;
    let preset_safetensors = required_path(
        &args.realtime_preset_safetensors,
        "--realtime-preset-safetensors",
    )?;
    let preset_manifest =
        required_path(&args.realtime_preset_manifest, "--realtime-preset-manifest")?;
    let preset_manifest_sha256 = required(
        &args.realtime_preset_manifest_sha256,
        "--realtime-preset-manifest-sha256",
    )?;
    let tokenizer_dir = required_path(&args.realtime_tokenizer_dir, "--realtime-tokenizer-dir")?;
    let text_path = required_path(&args.realtime_text_file, "--realtime-text-file")?;
    let expected_head = required(&args.realtime_vokra_head, "--realtime-vokra-head")?;
    let expected_tree = required(&args.realtime_vokra_tree, "--realtime-vokra-tree")?;
    let reference_script_sha256 = required(
        &args.realtime_reference_script_sha256,
        "--realtime-reference-script-sha256",
    )?;
    let uv_lock_sha256 = required(&args.realtime_uv_lock_sha256, "--realtime-uv-lock-sha256")?;
    let trusted_runner_sha256 = required(
        &args.realtime_trusted_runner_sha256,
        "--realtime-trusted-runner-sha256",
    )?;
    let output = required_path(&args.output, "--output")?;

    let text_bytes = read_bounded(&text_path, MAX_TEXT_BYTES, "Realtime text")?;
    let text = String::from_utf8(text_bytes.clone())
        .map_err(|error| format!("Realtime text is not UTF-8: {error}"))?;
    if text.trim().is_empty() {
        return Err("Realtime text must contain a non-whitespace prompt".to_owned());
    }
    let text_sha256 = hex_digest(&sha256(&text_bytes));

    // Hash the exact parsed/mapped image. Do not reopen the path, and do not
    // compare this derived digest with the raw Microsoft checkpoint digest.
    verify_gguf_digest(file_bytes, gguf_sha256)?;

    validate_output_destination(&output)?;
    let reference_json = reference_dir.join("reference.json");
    let owner = VibeVoiceRealtimeOwnerContract::load(VibeVoiceRealtimeOwnerInputs {
        owner_scope: &owner_scope,
        canonical_payload: &owner_canonical,
        expected_owner_scope_sha256: &owner_scope_sha256,
        expected_canonical_sha256: &owner_canonical_sha256,
        reference_json: &reference_json,
        expected_reference_sha256: &reference_sha256,
        expected_vokra_head: &expected_head,
        expected_vokra_tree_sha1: &expected_tree,
        measured_text_sha256: &text_sha256,
        expected_reference_script_sha256: &reference_script_sha256,
        expected_uv_lock_sha256: &uv_lock_sha256,
        expected_trusted_runner_sha256: &trusted_runner_sha256,
    })?;

    let preset_bytes = read_bounded(
        &preset_safetensors,
        MAX_PRESET_CACHE_BYTES,
        "Realtime preset safetensors",
    )?;
    let manifest_bytes = read_bounded(
        &preset_manifest,
        MAX_PRESET_MANIFEST_BYTES,
        "Realtime preset manifest",
    )?;
    verify_manifest_digest(&manifest_bytes, &preset_manifest_sha256)?;
    let preset = VibeVoiceRealtimePresetCache::from_bytes(
        &preset_bytes,
        &manifest_bytes,
        &preset_manifest_sha256,
    )
    .map_err(|error| format!("Realtime preset authentication: {error}"))?;

    let tokenizer = load_tokenizer(&tokenizer_dir)?;
    let noise = load_vibevoice_realtime_noise_tape(&reference_dir, &reference_sha256)?;
    if noise.is_empty() {
        return Err("Realtime authenticated noise tape is empty".to_owned());
    }
    if noise.len() > owner.execution.max_new_tokens {
        return Err(format!(
            "Realtime noise tape has {} draws, exceeding owner max_new_tokens {}",
            noise.len(),
            owner.execution.max_new_tokens
        ));
    }
    if owner.execution.ddpm_steps != VIBEVOICE_REALTIME_INFERENCE_STEPS {
        return Err(format!(
            "Realtime owner ddpm_steps {} differs from native {}",
            owner.execution.ddpm_steps, VIBEVOICE_REALTIME_INFERENCE_STEPS
        ));
    }

    Ok(ValidatedInputs {
        text,
        output,
        owner,
        preset,
        tokenizer,
        noise,
    })
}

fn consume_step(step: VibeVoiceRealtimeSynthesisStep, pcm: &mut Vec<f32>) -> Result<bool, String> {
    match step {
        VibeVoiceRealtimeSynthesisStep::Audio(chunk) => {
            if chunk.pcm.iter().any(|value| !value.is_finite()) {
                return Err("Realtime generation returned non-finite PCM".to_owned());
            }
            pcm.extend_from_slice(&chunk.pcm);
            Ok(false)
        }
        VibeVoiceRealtimeSynthesisStep::Draining { .. } => Ok(false),
        VibeVoiceRealtimeSynthesisStep::Finished { .. } => Ok(true),
    }
}

fn consume_step_sequence(
    steps: impl IntoIterator<Item = VibeVoiceRealtimeSynthesisStep>,
    consumed_draws: usize,
    tape_draws: usize,
) -> Result<Vec<f32>, String> {
    let mut pcm = Vec::new();
    let mut finished = false;
    for step in steps {
        if finished {
            return Err("Realtime noise tape contains draws after Finished".to_owned());
        }
        finished = consume_step(step, &mut pcm)?;
    }
    if !finished {
        return Err(format!(
            "Realtime noise tape ended after {consumed_draws} draws before Finished"
        ));
    }
    if consumed_draws != tape_draws {
        let difference = tape_draws.abs_diff(consumed_draws);
        return Err(format!(
            "Realtime noise tape draw count differs from consumed draws by {difference}"
        ));
    }
    if pcm.is_empty() || pcm.iter().any(|value| !value.is_finite()) {
        return Err("Realtime generation produced no finite PCM".to_owned());
    }
    Ok(pcm)
}

fn verify_gguf_digest(file_bytes: &[u8], expected: &str) -> Result<(), String> {
    let actual = hex_digest(&sha256(file_bytes));
    if actual != expected {
        return Err(format!(
            "Realtime derived GGUF SHA-256 mismatch: expected {expected}, got {actual}"
        ));
    }
    Ok(())
}

fn preflight_and_bind<T>(
    preflight: Result<ValidatedInputs, String>,
    binder: impl FnOnce(&ValidatedInputs) -> Result<T, String>,
) -> Result<(ValidatedInputs, T), String> {
    let validated = preflight?;
    let bound = binder(&validated)?;
    Ok((validated, bound))
}

fn load_tokenizer(root: &Path) -> Result<VibeVoiceRealtimeTokenizer, String> {
    let mut files = Vec::with_capacity(TOKENIZER_FILES.len());
    for &(name, expected_bytes, expected_sha256) in TOKENIZER_FILES {
        let bytes = read_bounded(
            &root.join(name),
            expected_bytes,
            &format!("Realtime tokenizer {name}"),
        )?;
        if bytes.len() as u64 != expected_bytes {
            return Err(format!(
                "Realtime tokenizer {name} has {} bytes; expected {expected_bytes}",
                bytes.len()
            ));
        }
        let actual = hex_digest(&sha256(&bytes));
        if actual != expected_sha256 {
            return Err(format!(
                "Realtime tokenizer {name} SHA-256 mismatch: expected {expected_sha256}, got {actual}"
            ));
        }
        files.push(bytes);
    }
    VibeVoiceRealtimeTokenizer::from_parts(&files[0], &files[1], &files[2], &files[3])
        .map_err(|error| format!("Realtime tokenizer authentication: {error}"))
}

fn verify_manifest_digest(manifest_bytes: &[u8], expected: &str) -> Result<(), String> {
    let actual = hex_digest(&sha256(manifest_bytes));
    if actual != expected {
        return Err(format!(
            "Realtime preset manifest SHA-256 mismatch: expected {expected}, got {actual}"
        ));
    }
    Ok(())
}

fn required<'a>(value: &'a Option<String>, flag: &str) -> Result<&'a str, String> {
    value
        .as_deref()
        .filter(|value| !value.is_empty())
        .ok_or_else(|| {
            format!("run (VibeVoice-Realtime): {flag} is required for authenticated activation")
        })
}

fn required_path(value: &Option<String>, flag: &str) -> Result<PathBuf, String> {
    Ok(PathBuf::from(required(value, flag)?))
}

fn read_bounded(path: &Path, max_bytes: u64, label: &str) -> Result<Vec<u8>, String> {
    reject_symlink_ancestors(path, label)?;
    let path_metadata = fs::symlink_metadata(path)
        .map_err(|error| format!("{label} `{}`: {error}", path.display()))?;
    if path_metadata.file_type().is_symlink() || !path_metadata.is_file() {
        return Err(format!(
            "{label} `{}` is not a regular non-symlink file",
            path.display()
        ));
    }
    let mut file =
        fs::File::open(path).map_err(|error| format!("{label} `{}`: {error}", path.display()))?;
    let metadata = file
        .metadata()
        .map_err(|error| format!("{label} metadata `{}`: {error}", path.display()))?;
    if !metadata.is_file() {
        return Err(format!(
            "{label} `{}` is not a regular non-symlink file",
            path.display()
        ));
    }
    if metadata.len() > max_bytes {
        return Err(format!(
            "{label} `{}` is {} bytes; bounded maximum is {max_bytes}",
            path.display(),
            metadata.len()
        ));
    }
    let expected_len = metadata.len();
    let read_limit = max_bytes
        .checked_add(1)
        .ok_or_else(|| format!("{label} bounded read limit overflows"))?;
    let capacity = usize::try_from(expected_len).map_err(|_| {
        format!(
            "{label} `{}` is too large for this platform",
            path.display()
        )
    })?;
    let mut bytes = Vec::with_capacity(capacity);
    (&mut file)
        .take(read_limit)
        .read_to_end(&mut bytes)
        .map_err(|error| format!("{label} `{}`: {error}", path.display()))?;
    if bytes.len() as u64 > max_bytes || bytes.len() as u64 != expected_len {
        return Err(format!(
            "{label} `{}` changed while being read",
            path.display()
        ));
    }
    let final_len = file
        .metadata()
        .map_err(|error| format!("{label} `{}`: {error}", path.display()))?
        .len();
    if final_len != expected_len {
        return Err(format!(
            "{label} `{}` changed while being read",
            path.display()
        ));
    }
    Ok(bytes)
}

fn reject_symlink_ancestors(path: &Path, label: &str) -> Result<(), String> {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .map_err(|error| format!("{label}: unable to resolve current directory: {error}"))?
            .join(path)
    };
    let mut current = absolute.as_path();
    loop {
        match fs::symlink_metadata(current) {
            Ok(metadata) if metadata.file_type().is_symlink() => {
                return Err(format!(
                    "{label} path or ancestor is a symlink: {}",
                    current.display()
                ));
            }
            Ok(_) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(format!("{label} `{}`: {error}", current.display())),
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

fn validate_output_destination(path: &Path) -> Result<(), String> {
    reject_symlink_ancestors(path, "Realtime output")?;
    if path.exists() || path.is_symlink() {
        return Err(format!(
            "Realtime output `{}` already exists; no-clobber activation refuses it",
            path.display()
        ));
    }
    let parent = output_parent(path);
    let metadata = fs::metadata(parent).map_err(|error| {
        format!(
            "Realtime output parent `{}` is unavailable: {error}",
            parent.display()
        )
    })?;
    if !metadata.is_dir() {
        return Err(format!(
            "Realtime output parent `{}` is not a directory",
            parent.display()
        ));
    }
    Ok(())
}

fn output_parent(path: &Path) -> &Path {
    path.parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
}

fn hex_digest(bytes: &[u8; 32]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(64);
    for &byte in bytes {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_models::vibevoice_streaming::VibeVoiceRealtimeAudioChunk;

    fn unique_fixture_dir(tag: &str) -> PathBuf {
        let base = std::fs::canonicalize(std::env::temp_dir()).expect("canonical temp directory");
        for attempt in 0..1024u32 {
            let path = base.join(format!(
                "vokra-realtime-route-{tag}-{}-{attempt}",
                std::process::id()
            ));
            match fs::create_dir(&path) {
                Ok(()) => return path,
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => panic!("fixture directory {path:?}: {error}"),
            }
        }
        panic!("unable to allocate unique fixture directory");
    }

    fn parsed_realtime_args(root: &Path, gguf_sha256: &str) -> super::super::RunArgs {
        let sha = "0".repeat(64);
        let head = "1".repeat(40);
        let tree = "2".repeat(40);
        let text = root.join("prompt.txt");
        fs::write(&text, b"model-free preflight prompt").expect("fixture prompt");
        let parts = vec![
            "--model".to_owned(),
            root.join("model.gguf").display().to_string(),
            "--realtime-text-file".to_owned(),
            text.display().to_string(),
            "--realtime-gguf-sha256".to_owned(),
            gguf_sha256.to_owned(),
            "--realtime-reference-dir".to_owned(),
            root.join("reference").display().to_string(),
            "--realtime-reference-sha256".to_owned(),
            sha.clone(),
            "--realtime-owner-scope".to_owned(),
            root.join("owner-scope.json").display().to_string(),
            "--realtime-owner-scope-sha256".to_owned(),
            sha.clone(),
            "--realtime-owner-canonical".to_owned(),
            root.join("owner-canonical.json").display().to_string(),
            "--realtime-owner-canonical-sha256".to_owned(),
            sha.clone(),
            "--realtime-preset-safetensors".to_owned(),
            root.join("cache.safetensors").display().to_string(),
            "--realtime-preset-manifest".to_owned(),
            root.join("cache.manifest.json").display().to_string(),
            "--realtime-preset-manifest-sha256".to_owned(),
            sha.clone(),
            "--realtime-tokenizer-dir".to_owned(),
            root.join("tokenizer").display().to_string(),
            "--realtime-vokra-head".to_owned(),
            head,
            "--realtime-vokra-tree".to_owned(),
            tree,
            "--realtime-reference-script-sha256".to_owned(),
            sha.clone(),
            "--realtime-uv-lock-sha256".to_owned(),
            sha.clone(),
            "--realtime-trusted-runner-sha256".to_owned(),
            sha,
            "--output".to_owned(),
            root.join("output.wav").display().to_string(),
        ];
        super::super::parse_args(&parts).expect("complete dedicated Realtime args parse")
    }

    #[test]
    fn raw_carter_source_and_derived_cache_are_distinct_identities() {
        let raw_pt_identity = sha256(b"fixture raw Carter .pt source bytes");
        let derived_cache_identity = sha256(b"fixture exported F32 safetensors bytes");
        assert_ne!(raw_pt_identity, derived_cache_identity);
        assert_eq!(hex_digest(&raw_pt_identity).len(), 64);
        assert_eq!(hex_digest(&derived_cache_identity).len(), 64);
    }

    #[test]
    fn externally_bound_manifest_digest_accepts_exact_bytes_and_rejects_tampering() {
        let manifest = b"fixture manifest bytes";
        let expected = hex_digest(&sha256(manifest));
        verify_manifest_digest(manifest, &expected).expect("exact external manifest binding");
        assert!(verify_manifest_digest(b"tampered manifest bytes", &expected).is_err());
    }

    #[test]
    fn preflight_rejection_never_invokes_native_binder_probe() {
        let root = unique_fixture_dir("preflight");
        let file_bytes = b"model-free GGUF fixture";
        let correct_sha = hex_digest(&sha256(file_bytes));
        let wrong_sha = "0".repeat(64);

        let wrong_args = parsed_realtime_args(&root, &wrong_sha);
        let mut wrong_bound = false;
        let wrong = preflight_and_bind(preflight(file_bytes, &wrong_args), |_validated| {
            wrong_bound = true;
            Ok::<(), String>(())
        });
        assert!(wrong.is_err());
        assert!(matches!(
            wrong.as_ref(),
            Err(error) if error.contains("derived GGUF SHA-256 mismatch")
        ));
        assert!(!wrong_bound);

        let mut bad_owner_hash_args = parsed_realtime_args(&root, &correct_sha);
        bad_owner_hash_args.realtime_owner_scope_sha256 = Some("not-a-sha".to_owned());
        let mut bad_hash_bound = false;
        let bad_hash =
            preflight_and_bind(preflight(file_bytes, &bad_owner_hash_args), |_validated| {
                bad_hash_bound = true;
                Ok::<(), String>(())
            });
        assert!(bad_hash.is_err());
        assert!(matches!(
            bad_hash.as_ref(),
            Err(error) if error.contains("owner scope file SHA-256")
        ));
        assert!(!bad_hash_bound);

        let mut bad_canonical_hash_args = parsed_realtime_args(&root, &correct_sha);
        bad_canonical_hash_args.realtime_owner_canonical_sha256 = Some("not-a-sha".to_owned());
        let mut bad_canonical_hash_bound = false;
        let bad_canonical_hash = preflight_and_bind(
            preflight(file_bytes, &bad_canonical_hash_args),
            |_validated| {
                bad_canonical_hash_bound = true;
                Ok::<(), String>(())
            },
        );
        assert!(bad_canonical_hash.is_err());
        assert!(matches!(
            bad_canonical_hash.as_ref(),
            Err(error) if error.contains("owner canonical SHA-256")
        ));
        assert!(!bad_canonical_hash_bound);

        let mut bad_reference_hash_args = parsed_realtime_args(&root, &correct_sha);
        bad_reference_hash_args.realtime_reference_sha256 = Some("not-a-sha".to_owned());
        let mut bad_reference_hash_bound = false;
        let bad_reference_hash = preflight_and_bind(
            preflight(file_bytes, &bad_reference_hash_args),
            |_validated| {
                bad_reference_hash_bound = true;
                Ok::<(), String>(())
            },
        );
        assert!(bad_reference_hash.is_err());
        assert!(matches!(
            bad_reference_hash.as_ref(),
            Err(error) if error.contains("reference.json SHA-256")
        ));
        assert!(!bad_reference_hash_bound);

        let owner_path_args = parsed_realtime_args(&root, &correct_sha);
        let mut owner_bound = false;
        let owner = preflight_and_bind(preflight(file_bytes, &owner_path_args), |_validated| {
            owner_bound = true;
            Ok::<(), String>(())
        });
        assert!(owner.is_err());
        assert!(matches!(
            owner.as_ref(),
            Err(error) if error.contains("owner scope")
        ));
        assert!(!owner_bound);
        fs::remove_dir_all(&root).expect("remove owned fixture directory");

        let canonical_root = unique_fixture_dir("canonical-path");
        let mut canonical_args = parsed_realtime_args(&canonical_root, &correct_sha);
        fs::write(canonical_root.join("owner-scope.json"), b"{}").expect("fixture owner scope");
        canonical_args.realtime_owner_scope_sha256 = Some(hex_digest(&sha256(b"{}")));
        let mut canonical_bound = false;
        let canonical = preflight_and_bind(preflight(file_bytes, &canonical_args), |_validated| {
            canonical_bound = true;
            Ok::<(), String>(())
        });
        assert!(canonical.is_err());
        assert!(matches!(
            canonical.as_ref(),
            Err(error) if error.contains("owner canonical payload")
        ));
        assert!(!canonical_bound);
        fs::remove_dir_all(&canonical_root).expect("remove owned fixture directory");

        let reference_root = unique_fixture_dir("reference-path");
        let mut reference_args = parsed_realtime_args(&reference_root, &correct_sha);
        fs::write(reference_root.join("owner-scope.json"), b"{}").expect("fixture scope");
        fs::write(reference_root.join("owner-canonical.json"), b"{}")
            .expect("fixture canonical payload");
        let empty_sha = hex_digest(&sha256(b"{}"));
        reference_args.realtime_owner_scope_sha256 = Some(empty_sha.clone());
        reference_args.realtime_owner_canonical_sha256 = Some(empty_sha);
        fs::create_dir(reference_root.join("reference")).expect("fixture reference directory");
        let mut reference_bound = false;
        let reference = preflight_and_bind(preflight(file_bytes, &reference_args), |_validated| {
            reference_bound = true;
            Ok::<(), String>(())
        });
        assert!(reference.is_err());
        assert!(matches!(
            reference.as_ref(),
            Err(error) if error.contains("reference.json")
        ));
        assert!(!reference_bound);
        fs::remove_dir_all(&reference_root).expect("remove owned fixture directory");
    }

    #[test]
    fn bounded_reads_reject_oversized_and_symlink_inputs() {
        let root = unique_fixture_dir("bounded");
        let file = root.join("text.txt");
        fs::write(&file, b"hello").expect("fixture file");
        assert!(read_bounded(&file, 4, "text").is_err());
        assert_eq!(
            read_bounded(&file, 16, "text").expect("bounded regular read"),
            b"hello"
        );
        let directory = root.join("not-a-file");
        fs::create_dir(&directory).expect("fixture directory");
        assert!(read_bounded(&directory, 16, "text").is_err());
        #[cfg(unix)]
        {
            let link = root.join("link.txt");
            std::os::unix::fs::symlink(&file, &link).expect("fixture symlink");
            assert!(read_bounded(&link, 16, "text").is_err());
        }
        fs::remove_dir_all(&root).expect("remove owned fixture directory");
    }

    #[test]
    fn output_preflight_rejects_existing_destination_without_mutation() {
        let root = unique_fixture_dir("output");
        let output = root.join("out.wav");
        fs::write(&output, b"sentinel").expect("fixture output");
        assert!(validate_output_destination(&output).is_err());
        assert_eq!(fs::read(&output).expect("sentinel output"), b"sentinel");
        assert_eq!(output_parent(Path::new("speech.wav")), Path::new("."));
        fs::remove_dir_all(&root).expect("remove owned fixture directory");
    }

    #[test]
    fn state_machine_collects_audio_suppresses_draining_and_requires_finished() {
        let mut pcm = Vec::new();
        let audio = VibeVoiceRealtimeAudioChunk {
            pcm: vec![0.25, -0.5],
            text_window_index: Some(0),
            speech_step: 0,
            generated_positions: 1,
            terminal_after: None,
        };
        assert!(
            !consume_step(VibeVoiceRealtimeSynthesisStep::Audio(audio), &mut pcm)
                .expect("finite audio")
        );
        assert!(
            !consume_step(
                VibeVoiceRealtimeSynthesisStep::Draining {
                    text_window_index: Some(0),
                    speech_step: 1,
                },
                &mut pcm
            )
            .expect("draining step")
        );
        assert_eq!(pcm, [0.25, -0.5]);
        assert!(consume_step(
            VibeVoiceRealtimeSynthesisStep::Finished {
                reason: vokra_models::vibevoice_streaming::VibeVoiceRealtimeGenerationStopReason::EndOfSpeech,
            },
            &mut pcm
        )
        .expect("terminal step"));
    }

    #[test]
    fn state_machine_rejects_nonfinite_audio_without_mutation() {
        let mut pcm = vec![0.5];
        let error = consume_step(
            VibeVoiceRealtimeSynthesisStep::Audio(VibeVoiceRealtimeAudioChunk {
                pcm: vec![f32::NAN],
                text_window_index: None,
                speech_step: 0,
                generated_positions: 0,
                terminal_after: None,
            }),
            &mut pcm,
        )
        .expect_err("nonfinite audio must fail closed");
        assert!(error.contains("non-finite PCM"));
        assert_eq!(pcm, [0.5]);
    }

    #[test]
    fn state_machine_rejects_missing_and_extra_terminal_events() {
        let audio = VibeVoiceRealtimeSynthesisStep::Audio(VibeVoiceRealtimeAudioChunk {
            pcm: vec![0.25],
            text_window_index: None,
            speech_step: 0,
            generated_positions: 1,
            terminal_after: None,
        });
        assert!(
            consume_step_sequence(vec![audio], 1, 1)
                .expect_err("missing Finished must fail closed")
                .contains("before Finished")
        );

        let finished = VibeVoiceRealtimeSynthesisStep::Finished {
            reason: vokra_models::vibevoice_streaming::VibeVoiceRealtimeGenerationStopReason::EndOfSpeech,
        };
        let extra = VibeVoiceRealtimeSynthesisStep::Finished {
            reason: vokra_models::vibevoice_streaming::VibeVoiceRealtimeGenerationStopReason::EndOfSpeech,
        };
        assert!(
            consume_step_sequence(vec![finished, extra], 1, 1)
                .expect_err("events after Finished must fail closed")
                .contains("after Finished")
        );
    }
}
