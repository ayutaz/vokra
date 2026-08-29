//! Real-weight Qwen3-TTS parity for the four released main checkpoints.
//!
//! The reference packet is generated only by the official Qwen3-TTS Python
//! package. This test compares the same prompt ids and complete sixteen-row
//! generated code matrix exactly, then reports finite/nonzero PCM metrics as
//! measurement-only until a reviewed numeric bound exists.

use std::path::{Path, PathBuf};

use vokra_core::backend::BackendKind;
use vokra_models::qwen3_tts::{Qwen3TtsGenerationOptions, Qwen3TtsMain, Qwen3TtsSynthesis};

const TEXT: &str = "The Vokra parity packet is short and deterministic.";
const LANGUAGE: &str = "English";
const CODEBOOKS: usize = 16;

#[derive(Clone, Copy)]
struct Case {
    slug: &'static str,
    env: &'static str,
    model_name: &'static str,
    base: bool,
    speaker_dim: usize,
}

const CASES: [Case; 4] = [
    Case {
        slug: "0.6b-base",
        env: "VOKRA_QWEN3_TTS_0_6B_BASE",
        model_name: "qwen3-tts-12hz-0.6b-base",
        base: true,
        speaker_dim: 1024,
    },
    Case {
        slug: "0.6b-customvoice",
        env: "VOKRA_QWEN3_TTS_0_6B_CUSTOMVOICE",
        model_name: "qwen3-tts-12hz-0.6b-customvoice",
        base: false,
        speaker_dim: 0,
    },
    Case {
        slug: "1.7b-base",
        env: "VOKRA_QWEN3_TTS_1_7B_BASE",
        model_name: "qwen3-tts-12hz-1.7b-base",
        base: true,
        speaker_dim: 2048,
    },
    Case {
        slug: "1.7b-customvoice",
        env: "VOKRA_QWEN3_TTS_1_7B_CUSTOMVOICE",
        model_name: "qwen3-tts-12hz-1.7b-customvoice",
        base: false,
        speaker_dim: 0,
    },
];

fn required_path(env: &str, label: &str) -> PathBuf {
    std::env::var_os(env)
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("{label}: {env} is required for the real-weight parity leg"))
}

fn read_u32(path: &Path) -> Vec<u32> {
    let bytes =
        std::fs::read(path).unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
    assert_eq!(
        bytes.len() % 4,
        0,
        "{} is not a u32le packet",
        path.display()
    );
    bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

fn read_f32(path: &Path) -> Vec<f32> {
    let bytes =
        std::fs::read(path).unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
    assert_eq!(
        bytes.len() % 4,
        0,
        "{} is not an f32le packet",
        path.display()
    );
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

fn manifest_hash(manifest: &str, file: &str) -> String {
    let key = format!("\"sha256_{}\":", file.replace('.', "_"));
    manifest
        .lines()
        .find_map(|line| {
            line.strip_prefix("  ")
                .and_then(|line| line.strip_prefix(&key))
                .and_then(|value| value.trim().trim_end_matches(',').strip_prefix('"'))
                .and_then(|value| value.strip_suffix('"'))
                .map(ToOwned::to_owned)
        })
        .unwrap_or_else(|| panic!("manifest lacks hash for {file}"))
}

fn verify_manifest_hashes(reference: &Path, manifest: &str, case: Case) {
    let mut files = vec![
        "prompt_ids.u32le",
        "codes.u32le",
        "pcm.f32le",
        "environment.json",
    ];
    if case.base {
        files.push("speaker_embedding.f32le");
    }
    for file in files {
        let bytes = std::fs::read(reference.join(file))
            .unwrap_or_else(|error| panic!("{} {file}: {error}", case.slug));
        assert_eq!(
            manifest_hash(manifest, file),
            sha256_hex(&bytes),
            "{} {file} differs from manifest",
            case.slug
        );
    }
}

// Zero-dependency FIPS 180-4 SHA-256 for the staged reference packet.
fn sha256_hex(data: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut h = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    let mut msg = data.to_vec();
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&((data.len() as u64) * 8).to_be_bytes());
    for block in msg.chunks_exact(64) {
        let mut w = [0u32; 64];
        for (i, chunk) in block.chunks_exact(4).enumerate() {
            w[i] = u32::from_be_bytes(chunk.try_into().unwrap());
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let (mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut j) =
            (h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]);
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let t1 = j
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            j = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        for (slot, value) in h.iter_mut().zip([a, b, c, d, e, f, g, j]) {
            *slot = slot.wrapping_add(value);
        }
    }
    h.iter().map(|value| format!("{value:08x}")).collect()
}

#[test]
fn sha256_manifest_verifier_known_vector() {
    assert_eq!(
        sha256_hex(b"abc"),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
}

fn assert_reference_manifest(reference: &Path, case: Case) -> String {
    let manifest = std::fs::read_to_string(reference.join("manifest.json"))
        .unwrap_or_else(|error| panic!("{} manifest: {error}", case.slug));
    let (repo, revision, speaker) = match case.slug {
        "0.6b-base" => (
            "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
            "5d83992436eae1d760afd27aff78a71d676296fc",
            "official_x_vector_only",
        ),
        "0.6b-customvoice" => (
            "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
            "85e237c12c027371202489a0ec509ded67b5e4b5",
            "Serena",
        ),
        "1.7b-base" => (
            "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            "fd4b254389122332181a7c3db7f27e918eec64e3",
            "official_x_vector_only",
        ),
        "1.7b-customvoice" => (
            "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
            "0c0e3051f131929182e2c023b9537f8b1c68adfe",
            "Serena",
        ),
        _ => unreachable!(),
    };
    for marker in [
        "\"schema\": \"vokra-qwen3-tts-reference-v1\"",
        &format!("\"upstream_repo\": \"{repo}\""),
        &format!("\"upstream_revision\": \"{revision}\""),
        &format!("\"variant\": \"{}\"", case.slug),
        &format!("\"model_name\": \"{}\"", case.model_name),
        "\"official_source_repo\": \"QwenLM/Qwen3-TTS\"",
        "\"official_source_revision\": \"022e286b98fbec7e1e916cb940cdf532cd9f488e\"",
        "\"decoder_repo\": \"Qwen/Qwen3-TTS-Tokenizer-12Hz\"",
        "\"decoder_revision\": \"a87c50897bb00837eb857d0538b29d117541d7f6\"",
        "\"decoder_checkpoint_sha256\": \"836b7b357f5ea43e889936a3709af68dfe3751881acefe4ecf0dbd30ba571258\"",
        "\"nested_decoder_sha256\": \"836b7b357f5ea43e889936a3709af68dfe3751881acefe4ecf0dbd30ba571258\"",
        "\"text\": \"The Vokra parity packet is short and deterministic.\"",
        "\"language\": \"English\"",
        &format!("\"speaker\": \"{speaker}\""),
        "\"qwen_tts_version\": \"0.1.1\"",
        "\"max_new_tokens\": 8",
        "\"min_new_tokens\": 2",
        "\"sample_rate\": 24000",
        "\"sampling\": \"greedy\"",
        "\"codebooks\": 16",
    ] {
        assert!(
            manifest.contains(marker),
            "{} reference manifest lost {marker}",
            case.slug
        );
    }
    manifest
}

fn measure_pcm(case: Case, actual: &[f32], expected: &[f32], backend: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{} {backend} PCM length",
        case.slug
    );
    assert!(!actual.is_empty(), "{} {backend} PCM is empty", case.slug);
    assert!(
        actual.iter().chain(expected).all(|value| value.is_finite()),
        "{} {backend} PCM is non-finite",
        case.slug
    );
    let mut max_abs = 0.0f64;
    let mut sum_sq = 0.0f64;
    let mut actual_sq = 0.0f64;
    let mut expected_sq = 0.0f64;
    for (&left, &right) in actual.iter().zip(expected) {
        let left = f64::from(left);
        let right = f64::from(right);
        max_abs = max_abs.max((left - right).abs());
        sum_sq += (left - right) * (left - right);
        actual_sq += left * left;
        expected_sq += right * right;
    }
    let rmse = (sum_sq / actual.len() as f64).sqrt();
    assert!(
        actual_sq.is_finite() && actual_sq > 0.0,
        "{} {backend} PCM actual norm invalid",
        case.slug
    );
    assert!(
        expected_sq.is_finite() && expected_sq > 0.0,
        "{} {backend} PCM reference norm invalid",
        case.slug
    );
    assert!(
        max_abs.is_finite() && rmse.is_finite(),
        "{} {backend} PCM metric is non-finite",
        case.slug
    );
    eprintln!(
        "QWEN3_TTS_MEASUREMENT variant={} backend={} pcm_max_abs={max_abs:.9e} pcm_rmse={rmse:.9e} numeric_bound=UNSET verdict=MEASURED_NOT_GATED",
        case.slug, backend
    );
}

fn run_case(case: Case, backend: BackendKind, backend_name: &str) -> Qwen3TtsSynthesis {
    let gguf = required_path(&format!("{}_GGUF", case.env), case.slug);
    let decoder = required_path(&format!("{}_DECODER_GGUF", case.env), case.slug);
    let reference = required_path(&format!("{}_REFERENCE_DIR", case.env), case.slug);
    let manifest = assert_reference_manifest(&reference, case);
    verify_manifest_hashes(&reference, &manifest, case);
    let prompt_ids = read_u32(&reference.join("prompt_ids.u32le"));
    let expected_codes = read_u32(&reference.join("codes.u32le"));
    assert!(
        !expected_codes.is_empty() && expected_codes.len().is_multiple_of(CODEBOOKS),
        "{} official code matrix is empty or not [frames,16]",
        case.slug
    );
    let expected_pcm = read_f32(&reference.join("pcm.f32le"));

    let model = Qwen3TtsMain::open_mapped(&gguf, backend)
        .unwrap_or_else(|error| panic!("{} {backend_name} main bind: {error}", case.slug));
    assert_eq!(model.checkpoint().model_name(), case.model_name);
    let actual_prompt = model
        .tokenizer()
        .assistant_ids(TEXT)
        .expect("native prompt tokenize");
    assert_eq!(
        actual_prompt, prompt_ids,
        "{} prompt token ids differ",
        case.slug
    );
    let mut options = Qwen3TtsGenerationOptions::greedy(8);
    options.language = LANGUAGE.to_owned();
    if case.base {
        let embedding = read_f32(&reference.join("speaker_embedding.f32le"));
        assert_eq!(
            embedding.len(),
            case.speaker_dim,
            "{} speaker embedding width",
            case.slug
        );
        assert!(
            embedding.iter().all(|value| value.is_finite()),
            "{} speaker embedding non-finite",
            case.slug
        );
        options.speaker_embedding = Some(embedding);
    } else {
        options.speaker = Some("Serena".to_owned());
    }
    let decoder_model =
        vokra_models::qwen3_tts::Qwen3TtsTokenizer12HzDecoder::open_mapped_with_backend(
            &decoder, backend,
        )
        .unwrap_or_else(|error| panic!("{} {backend_name} decoder bind: {error}", case.slug));
    let synthesis = model
        .synthesize_with_decoder(&decoder_model, TEXT, &options)
        .unwrap_or_else(|error| panic!("{} {backend_name} synthesis: {error}", case.slug));
    assert_eq!(synthesis.sample_rate, 24_000);
    assert_eq!(
        synthesis.generation.as_frame_major(),
        expected_codes,
        "{} {} generated code matrix differs (first codebook + 15 predictor rows are covered exactly)",
        case.slug,
        backend_name
    );
    assert!(
        synthesis.generation.frames() > 0,
        "{} generated no frames",
        case.slug
    );
    measure_pcm(case, &synthesis.pcm, &expected_pcm, backend_name);
    eprintln!(
        "QWEN3_TTS_PARITY variant={} backend={} prompt_ids=exact codes_exact=PASS pcm=MEASURED_NOT_GATED",
        case.slug, backend_name
    );
    synthesis
}

fn compare_cpu_metal(case: Case, cpu: &Qwen3TtsSynthesis, metal: &Qwen3TtsSynthesis) {
    assert_eq!(
        cpu.generation.as_frame_major(),
        metal.generation.as_frame_major(),
        "{} CPU and Metal generated code matrices differ",
        case.slug
    );
    assert_eq!(
        cpu.sample_rate, metal.sample_rate,
        "{} CPU/Metal sample rate",
        case.slug
    );
    measure_pcm(case, &metal.pcm, &cpu.pcm, "metal_vs_cpu");
    eprintln!(
        "QWEN3_TTS_METAL_CPU variant={} codes_exact=PASS pcm=MEASURED_NOT_GATED",
        case.slug
    );
}

#[test]
#[ignore = "requires four VAST-produced corrected main/decoder GGUF pairs and official references"]
fn qwen3_tts_real_cpu_matches_official_reference() {
    for case in CASES {
        run_case(case, BackendKind::Cpu, "cpu");
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
#[ignore = "requires disposable Apple Silicon and four VAST-produced GGUF/reference pairs"]
fn qwen3_tts_real_metal_matches_cpu_and_official_reference() {
    vokra_backend_metal::vokra_metal_probe()
        .expect("Qwen3-TTS Metal parity requires a real Metal device");
    for case in CASES {
        let cpu = run_case(case, BackendKind::Cpu, "cpu");
        let metal = run_case(case, BackendKind::Metal, "metal");
        compare_cpu_metal(case, &cpu, &metal);
    }
}
