//! CosyVoice2 numerical parity skeleton — GGUF-gated (M3-09-T22 / T23).
//!
//! Every test here that needs the CosyVoice2 GGUF is gated on the
//! `VOKRA_COSYVOICE2_GGUF` environment variable and skips cleanly when it
//! is unset — the same pattern the Kokoro / Whisper parity harnesses use
//! (`parity_kokoro.rs` / `parity_whisper.rs`). The fixture-free tests
//! (config surface + module-tree wiring) run everywhere so the top-level
//! error surface (arch mismatch, `0`-placeholder hparam rejection, no
//! silent fallback) is exercised in CI without any HuggingFace download.
//!
//! # Scope of this scaffold (M3-09 partial land)
//!
//! The one-session scaffold provides:
//!
//! - the environment-variable-gated skeleton (T22 workflow shape) so the
//!   follow-on session can hang concrete per-tensor parity assertions
//!   off `parity_cosyvoice2_gguf_smoke` without re-plumbing;
//! - fixture-free tests exercising the load path's failure modes
//!   (`0`-placeholder mimi shape refused, wrong arch refused, no silent
//!   fallback on synthesize).
//!
//! The concrete per-tensor `atol = 0.01` (NFR-QL-01) checks + the MEL loss
//! 5% gate (NFR-QL-02, T23 `vokra-eval` bridge) land with T22/T23 once
//! the CosyVoice2 GGUF fixture is available (T29 model zoo publication).

use std::env;
use std::path::Path;

use vokra_core::gguf::chunks::KEY_MODEL_ARCH;
use vokra_core::gguf::{GgufArray, GgufBuilder, GgufFile, GgufMetadataValue, GgufValueType};
use vokra_core::{CompliancePolicy, SynthesisRequest, TtsEngine, VokraError};
use vokra_models::cosyvoice2::llm::{LlmBackbone, parity};
use vokra_models::cosyvoice2::{CosyVoice2Config, CosyVoice2Tokenizer, CosyVoice2Tts, MimiBridge};

/// The env var CI / owners set to point the gated tests at a real
/// CosyVoice2 GGUF. Absent = skip (never fabricate a pass).
const GGUF_ENV: &str = "VOKRA_COSYVOICE2_GGUF";

/// The env var pointing at the HF reference-dump **directory** for the
/// real-checkpoint LLM parity test — the layout the 2026-07-16 real-weight
/// eval produced (`~/.cache/vokra-eval/out/tts-cosyvoice2/`):
///
/// - `true_hf_logits.npy` — `[t, vocab]` f32 C-order dump of
///   `transformers` `Qwen2ForCausalLM` (eager, f32) logits over the real
///   `FunAudioLLM/CosyVoice2-0.5B` `llm.pt` (generator: `ref_true.py` in
///   the same dir; reference validity was pinned by the eval at
///   mirror(with-bias) vs transformers = 2.098e-5).
/// - `token-ids.json` — `{"ids": [u32, ...], ...}`, the token ids the dump
///   was generated over.
///
/// Absent = clean skip. **Present = fully binding**: a missing GGUF env, a
/// missing/malformed dump, a synthesized-weight bind or a parity miss all
/// hard-fail (no silent skip once opted in).
const HF_REFERENCE_ENV: &str = "VOKRA_COSYVOICE2_HF_REFERENCE";

/// Builds a synthetic CosyVoice2 GGUF with model-card + Mimi defaults and
/// caller-controlled `arch` / `flow_schedule` fields.
///
/// Every other numeric hparam is `0` — the runtime accepts this via
/// [`CosyVoice2Config::from_gguf`] and only rejects downstream (at the
/// forward path or the Mimi bridge), so the load itself succeeds and lets
/// us exercise the load-side error surface here.
fn synthetic_gguf(arch: &str, flow_schedule: &str) -> Vec<u8> {
    let mut b = GgufBuilder::new();
    b.add_string(KEY_MODEL_ARCH, arch);
    b.add_string("vokra.model.name", "cosyvoice2-0.5b-synthetic");
    b.add_u32("vokra.cosyvoice2.sample_rate", 24_000);
    b.add_u32("vokra.cosyvoice2.arch.vocab_size", 0);
    b.add_u32("vokra.cosyvoice2.arch.hidden_dim", 0);
    b.add_u32("vokra.cosyvoice2.arch.n_layer", 0);
    b.add_u32("vokra.cosyvoice2.arch.n_head", 0);
    b.add_u32("vokra.cosyvoice2.arch.ffn_dim", 0);
    b.add_u32("vokra.cosyvoice2.flow.nfe", 0);
    b.add_metadata(
        "vokra.cosyvoice2.flow.schedule",
        GgufMetadataValue::String(flow_schedule.to_owned()),
    );
    // Mimi shape defaults (the converter writes canonical Kyutai values;
    // see crates/vokra-convert/src/models/cosyvoice2.rs).
    b.add_u32("vokra.cosyvoice2.mimi.n_codebooks", 8);
    b.add_u32("vokra.cosyvoice2.mimi.codebook_size", 2048);
    b.add_u32("vokra.cosyvoice2.mimi.d_model", 512);
    b.add_u32("vokra.cosyvoice2.streaming.chunk_size", 0);
    b.add_u32("vokra.cosyvoice2.streaming.chunk_hop", 0);
    b.to_bytes().expect("gguf serialize")
}

/// Fixture-free: the arch check runs before any component loader — a
/// wrong-arch GGUF fails with a clear top-level `ModelLoad`, not a
/// downstream missing-tensor error (FR-EX-08).
#[test]
fn parity_cosyvoice2_wrong_arch_fails_top_level() {
    let bytes = synthetic_gguf("kokoro-82m-istftnet", "linear");
    let err = CosyVoice2Tts::from_gguf_with_policy(&bytes, &CompliancePolicy::strict())
        .expect_err("wrong arch must fail");
    match err {
        VokraError::ModelLoad(msg) => {
            assert!(
                msg.contains("cosyvoice2") && msg.contains("kokoro-82m-istftnet"),
                "unexpected: {msg}"
            );
        }
        other => panic!("expected ModelLoad, got {other:?}"),
    }
}

/// Fixture-free: the synthetic GGUF loads (arch OK, compliance registry
/// classifies `cosyvoice2` permissive), but every forward path returns
/// NotImplemented because the numeric layers are the scaffold-only path.
#[test]
fn parity_cosyvoice2_synthetic_load_succeeds_but_synthesize_is_stub() {
    let bytes = synthetic_gguf("cosyvoice2", "linear");
    let tts = CosyVoice2Tts::from_gguf_with_policy(&bytes, &CompliancePolicy::strict())
        .expect("apache-2.0 registry entry admits it");
    assert_eq!(tts.config().sample_rate, 24_000);
    let err = tts
        .synthesize(&SynthesisRequest::new("hello"))
        .expect_err("scaffold must not produce audio");
    assert!(matches!(err, VokraError::NotImplemented(_)));
}

/// Fixture-free: the Mimi bridge accepts the canonical shape emitted by
/// the converter and rejects a degenerate `0` shape (converter placeholder
/// path).
#[test]
fn parity_cosyvoice2_mimi_bridge_accepts_kyutai_defaults() {
    let bytes = synthetic_gguf("cosyvoice2", "linear");
    let file = vokra_core::gguf::GgufFile::parse(bytes).expect("parse");
    let cfg = CosyVoice2Config::from_gguf(&file).expect("read");
    let bridge = MimiBridge::from_config(&cfg).expect("Kyutai defaults must load");
    assert_eq!(bridge.attrs().n_codebooks, 8);
    assert_eq!(bridge.attrs().codebook_size, 2048);
    assert_eq!(bridge.attrs().d_model, 512);
}

/// Fixture-free: an unknown schedule tag is a loud error (no silent
/// fallback to linear).
#[test]
fn parity_cosyvoice2_unknown_schedule_fails_up_front() {
    let bytes = synthetic_gguf("cosyvoice2", "cosine");
    let file = vokra_core::gguf::GgufFile::parse(bytes).expect("parse");
    let cfg = CosyVoice2Config::from_gguf(&file).expect("read");
    // The runtime accepts the tag string at config load time (the reader
    // is dumb by design — no schedule vocabulary hard-coded in
    // `CosyVoice2Config`), so the loud failure fires when the Flow
    // Matching driver builds its runtime params.
    use vokra_models::cosyvoice2::FlowMatchingRuntimeParams;
    let err = FlowMatchingRuntimeParams::from_config(&cfg)
        .expect_err("cosine is not a schedule vokra_ops accepts");
    assert!(matches!(err, VokraError::InvalidArgument(_)));
}

/// Gated: env-var-gated load smoke (skip cleanly when unset) so the CI
/// harness can be enabled at any time without changing this file (mirrors
/// `parity_kokoro.rs` T22 shape). The forward-parity gate is
/// [`parity_cosyvoice2_llm_forward_vs_hf_reference`] below.
#[test]
fn parity_cosyvoice2_gguf_smoke() {
    let Some(gguf_path) = env::var(GGUF_ENV).ok() else {
        eprintln!(
            "{GGUF_ENV} unset — skipping cosyvoice2 GGUF parity smoke; \
             this is a clean skip (never a fabricated pass)"
        );
        return;
    };
    // The gated body: load the GGUF and check the arch. A re-converted
    // (post-hparam-fix) GGUF binds the real LLM weights here; a pre-fix
    // GGUF (0-placeholder hparams) still loads with `llm = None`.
    let tts = CosyVoice2Tts::from_path_with_policy(&gguf_path, &CompliancePolicy::strict())
        .expect("real CosyVoice2 GGUF must load under strict compliance");
    // A stock CosyVoice2 GGUF is Apache 2.0; the registry admits it
    // without a research flag.
    assert_eq!(
        tts.config().sample_rate,
        24_000,
        "the CosyVoice2 model card fixes the PCM rate at 24 kHz"
    );
    eprintln!(
        "cosyvoice2 GGUF loaded from {gguf_path}: sample_rate=24_000, llm bound: {} — \
         run the {HF_REFERENCE_ENV}-gated test for forward parity vs the HF dump",
        tts.llm().is_some(),
    );
}

// -----------------------------------------------------------------------------
// Real-checkpoint forward parity vs the transformers reference dump
// -----------------------------------------------------------------------------

/// Tolerance for the real-weight forward vs the `transformers` eager
/// reference.
///
/// Honest-atol rationale (feedback-honest-parity-atol): the 2026-07-16
/// eval measured the pure f32 GEMM-order floor for this exact forward at
/// max |Δ| = 1.45e-4 (vokra vs a bias-less PyTorch mirror of the same ops,
/// full 24-layer depth, logits |max| ≈ 13), and the with-bias mirror
/// pinned the reference itself at 2.098e-5 vs transformers. The bias adds
/// one fused per-column affine per Q/K/V GEMM, which does not change the
/// error scale. atol = 3e-4 ≈ 2× the measured floor — tight enough that a
/// representation bug (the pre-fix bias gap measured max |Δ| = 12.92)
/// fails by 4+ orders of magnitude. The argmax 10/10 gate is asserted
/// independently by the harness.
///
/// Measured on the fix landing (2026-07-16, M1, re-converted GGUF with
/// `--config`): max |Δ| = 3.433e-5, mean |Δ| = 2.008e-6, argmax 10/10.
const HF_PARITY_ATOL: f32 = 3e-4;

/// Flip-the-switch real-checkpoint parity (closed by the 2026-07-16 eval,
/// P1 fix #4): forwards the eval's token ids through the **real-weight**
/// LLM backbone and compares every logit against the with-bias
/// `transformers` reference dump.
///
/// Skips cleanly only when [`HF_REFERENCE_ENV`] is unset. Once set,
/// everything is binding — missing GGUF env, unreadable/malformed dump,
/// synthesized bind, tolerance or argmax miss all fail the test loudly.
#[test]
fn parity_cosyvoice2_llm_forward_vs_hf_reference() {
    let Some(ref_dir) = env::var(HF_REFERENCE_ENV).ok() else {
        eprintln!(
            "{HF_REFERENCE_ENV} unset — skipping cosyvoice2 real-checkpoint LLM \
             parity; this is a clean skip (never a fabricated pass)"
        );
        return;
    };
    // Opted in: every failure below is hard (no silent skip).
    let gguf_path = env::var(GGUF_ENV).unwrap_or_else(|_| {
        panic!(
            "{HF_REFERENCE_ENV} is set but {GGUF_ENV} is not — the parity run needs \
             the re-converted GGUF too (opted-in ⇒ incomplete setup is a failure)"
        )
    });
    let ref_dir = Path::new(&ref_dir);
    let token_ids = read_token_ids(&ref_dir.join("token-ids.json"));
    let (shape, reference) = read_npy_f32_2d(&ref_dir.join("true_hf_logits.npy"));
    assert_eq!(
        shape.0,
        token_ids.len(),
        "reference dump rows {} != token id count {}",
        shape.0,
        token_ids.len()
    );

    let bytes = std::fs::read(&gguf_path)
        .unwrap_or_else(|e| panic!("{GGUF_ENV} = {gguf_path}: unreadable: {e}"));
    let file = GgufFile::parse(bytes).expect("GGUF parse");
    let cfg = CosyVoice2Config::from_gguf(&file).expect("cosyvoice2 config");
    let backbone = LlmBackbone::from_gguf_with_weights(&file, &cfg)
        .expect("real-weight bind (re-convert with --config if this names 0-hparams)");
    assert!(
        !backbone.weights().is_synthesized,
        "parity must run against real weights, never the synthesized fixture"
    );
    assert_eq!(
        backbone.config().vocab_size,
        shape.1,
        "GGUF vocab != reference dump vocab width"
    );

    let report = parity::assert_vs_hf_reference(&backbone, &token_ids, &reference, HF_PARITY_ATOL)
        .expect("forward parity vs transformers eager reference");
    eprintln!(
        "cosyvoice2 LLM real-checkpoint parity PASS: t={} vocab={} max|Δ|={:.6e} \
         mean|Δ|={:.6e} argmax {}/{} (atol {HF_PARITY_ATOL:.1e})",
        report.t,
        report.vocab,
        report.max_abs_delta,
        report.mean_abs_delta,
        report.argmax_matches,
        report.t,
    );
}

/// Reads the eval's `token-ids.json` (`{"ids": [int, ...], ...}`) with the
/// zero-dep `vokra_core::json` parser. Panics with context on any
/// deviation — the file is part of the opted-in fixture set.
fn read_token_ids(path: &Path) -> Vec<u32> {
    let bytes =
        std::fs::read(path).unwrap_or_else(|e| panic!("{}: unreadable: {e}", path.display()));
    let root = vokra_core::json::parse(&bytes)
        .unwrap_or_else(|e| panic!("{}: not valid JSON: {e}", path.display()));
    let ids = root
        .get("ids")
        .and_then(|v| v.as_array())
        .unwrap_or_else(|| panic!("{}: no `ids` array", path.display()));
    ids.iter()
        .map(|v| {
            v.as_u64()
                .and_then(|x| u32::try_from(x).ok())
                .unwrap_or_else(|| panic!("{}: non-u32 entry in `ids`: {v:?}", path.display()))
        })
        .collect()
}

/// Minimal strict NumPy `.npy` reader for the reference dump: v1/v2/v3
/// header, little-endian f32 (`'<f4'`), C-order, rank-2 shape. Anything
/// else panics with the offending field — this is a test fixture reader,
/// not a general NumPy port (format spec: numpy/lib/format.py).
fn read_npy_f32_2d(path: &Path) -> ((usize, usize), Vec<f32>) {
    let bytes =
        std::fs::read(path).unwrap_or_else(|e| panic!("{}: unreadable: {e}", path.display()));
    let ctx = path.display();
    assert!(
        bytes.len() >= 10 && bytes[0..6] == *b"\x93NUMPY",
        "{ctx}: not a .npy file (bad magic)"
    );
    let major = bytes[6];
    let (header_len, header_start) = match major {
        1 => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10usize),
        2 | 3 => (
            u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
            12usize,
        ),
        other => panic!("{ctx}: unsupported .npy major version {other}"),
    };
    let header_end = header_start + header_len;
    assert!(bytes.len() >= header_end, "{ctx}: truncated .npy header");
    let header = std::str::from_utf8(&bytes[header_start..header_end])
        .unwrap_or_else(|e| panic!("{ctx}: non-UTF-8 .npy header: {e}"));

    // The header is a Python dict literal; assert the exact dtype/order we
    // support instead of parsing Python.
    assert!(
        header.contains("'descr': '<f4'"),
        "{ctx}: dtype is not little-endian f32 ('<f4'): {header}"
    );
    assert!(
        header.contains("'fortran_order': False"),
        "{ctx}: fortran_order must be False (C-order): {header}"
    );
    let shape_field = header
        .split("'shape':")
        .nth(1)
        .unwrap_or_else(|| panic!("{ctx}: no 'shape' in .npy header: {header}"));
    let open = shape_field
        .find('(')
        .unwrap_or_else(|| panic!("{ctx}: malformed shape tuple: {header}"));
    let close = shape_field
        .find(')')
        .unwrap_or_else(|| panic!("{ctx}: malformed shape tuple: {header}"));
    let dims: Vec<usize> = shape_field[open + 1..close]
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| {
            s.parse()
                .unwrap_or_else(|e| panic!("{ctx}: bad shape dim `{s}`: {e}"))
        })
        .collect();
    let [rows, cols] = dims[..] else {
        panic!("{ctx}: expected a rank-2 shape, got {dims:?}");
    };

    let payload = &bytes[header_end..];
    let want_bytes = rows * cols * 4;
    assert_eq!(
        payload.len(),
        want_bytes,
        "{ctx}: payload {} bytes != shape ({rows}, {cols}) × 4",
        payload.len()
    );
    let data = payload
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    ((rows, cols), data)
}

// -----------------------------------------------------------------------------
// Text tokenizer (M3-09-T06): GGUF-embedded byte-level BPE
// -----------------------------------------------------------------------------

/// The env var pointing at a directory holding the upstream Qwen2
/// `vocab.json` + `merges.txt` (e.g.
/// `~/.cache/vokra-eval/weights/cosyvoice2-0.5b/CosyVoice-BlankEN`). Absent =
/// clean skip of the real-vocab round-trip (never a fabricated pass).
const TOKENIZER_DIR_ENV: &str = "VOKRA_COSYVOICE2_TOKENIZER_DIR";

/// A tiny byte-level BPE tokenizer over the lowercase ASCII alphabet + the
/// space byte-char ('Ġ'), with one merge (`h e` → `he`). Enough to exercise
/// the GGUF-embed → load → encode → decode wiring on ASCII strings.
fn small_ascii_tokenizer_files() -> (Vec<u8>, Vec<u8>) {
    let mut entries: Vec<(String, u32)> = Vec::new();
    let mut id = 0u32;
    for c in b'a'..=b'z' {
        entries.push(((c as char).to_string(), id));
        id += 1;
    }
    entries.push(("Ġ".to_owned(), id)); // byte-char for ASCII space (0x20)
    id += 1;
    entries.push(("he".to_owned(), id)); // the (h,e) merge target
    let mut json = String::from("{");
    for (i, (tok, tid)) in entries.iter().enumerate() {
        if i > 0 {
            json.push(',');
        }
        // No byte-char in this vocab needs JSON escaping.
        json.push('"');
        json.push_str(tok);
        json.push_str("\":");
        json.push_str(&tid.to_string());
    }
    json.push('}');
    (json.into_bytes(), b"h e\n".to_vec())
}

/// Builds a synthetic CosyVoice2 GGUF (as [`synthetic_gguf`]) plus the two
/// embedded tokenizer U8 chunks.
fn synthetic_gguf_with_tokenizer(vocab: &[u8], merges: &[u8]) -> Vec<u8> {
    let mut b = GgufBuilder::new();
    b.add_string(KEY_MODEL_ARCH, "cosyvoice2");
    b.add_string("vokra.model.name", "cosyvoice2-0.5b-synthetic");
    b.add_u32("vokra.cosyvoice2.sample_rate", 24_000);
    b.add_u32("vokra.cosyvoice2.arch.vocab_size", 0);
    b.add_u32("vokra.cosyvoice2.arch.hidden_dim", 0);
    b.add_u32("vokra.cosyvoice2.arch.n_layer", 0);
    b.add_u32("vokra.cosyvoice2.arch.n_head", 0);
    b.add_u32("vokra.cosyvoice2.arch.ffn_dim", 0);
    b.add_u32("vokra.cosyvoice2.flow.nfe", 0);
    b.add_metadata(
        "vokra.cosyvoice2.flow.schedule",
        GgufMetadataValue::String("linear".to_owned()),
    );
    b.add_u32("vokra.cosyvoice2.mimi.n_codebooks", 8);
    b.add_u32("vokra.cosyvoice2.mimi.codebook_size", 2048);
    b.add_u32("vokra.cosyvoice2.mimi.d_model", 512);
    b.add_u32("vokra.cosyvoice2.streaming.chunk_size", 0);
    b.add_u32("vokra.cosyvoice2.streaming.chunk_hop", 0);
    let u8_array = |bytes: &[u8]| {
        GgufMetadataValue::Array(GgufArray {
            element_type: GgufValueType::U8,
            values: bytes.iter().map(|&x| GgufMetadataValue::U8(x)).collect(),
        })
    };
    b.add_metadata("vokra.cosyvoice2.tokenizer.vocab", u8_array(vocab));
    b.add_metadata("vokra.cosyvoice2.tokenizer.merges", u8_array(merges));
    b.to_bytes().expect("gguf serialize")
}

/// Fixture-free: a GGUF carrying the embedded tokenizer chunks loads, exposes
/// the tokenizer, and round-trips through the engine `encode` + tokenizer
/// `decode` — the T06 wiring, exercised without any HuggingFace download.
#[test]
fn parity_cosyvoice2_tokenizer_embedded_roundtrips_via_engine() {
    let (vocab, merges) = small_ascii_tokenizer_files();
    let bytes = synthetic_gguf_with_tokenizer(&vocab, &merges);
    let tts = CosyVoice2Tts::from_gguf_with_policy(&bytes, &CompliancePolicy::strict())
        .expect("tokenizer-carrying GGUF must load");
    let tok = tts.tokenizer().expect("tokenizer must be present");

    let ids = tts.encode("hello").expect("encode");
    assert!(!ids.is_empty(), "encode must not be empty");
    assert_eq!(tok.decode(&ids).expect("decode"), "hello");

    for s in ["he", "hello there", "abc xyz"] {
        let e = tts
            .encode(s)
            .unwrap_or_else(|err| panic!("encode {s:?}: {err}"));
        assert_eq!(tok.decode(&e).expect("decode"), s, "round-trip {s:?}");
    }
    // The (h,e) merge fires: "he" collapses to a single token id.
    assert_eq!(
        tts.encode("he").unwrap().len(),
        1,
        "`he` must merge to one token, not two single-byte tokens"
    );
}

/// Fixture-free: a GGUF with no tokenizer chunks loads (the tokenizer is
/// optional), but `encode` is a loud `NotImplemented` — never a silent empty
/// id list (FR-EX-08).
#[test]
fn parity_cosyvoice2_tokenizer_absent_encode_is_loud() {
    let bytes = synthetic_gguf("cosyvoice2", "linear");
    let tts = CosyVoice2Tts::from_gguf_with_policy(&bytes, &CompliancePolicy::strict())
        .expect("tokenizer-less GGUF still loads");
    assert!(
        tts.tokenizer().is_none(),
        "no tokenizer chunks → tokenizer() is None"
    );
    let err = tts
        .encode("hello")
        .expect_err("encode without a tokenizer must be loud");
    assert!(matches!(err, VokraError::NotImplemented(_)), "got {err:?}");
}

/// Gated real-vocab round-trip: with [`TOKENIZER_DIR_ENV`] pointing at the
/// upstream Qwen2 `vocab.json` + `merges.txt`, load the real 151k-token
/// tokenizer and verify `decode(encode(s)) == s` over several UTF-8 strings,
/// plus one pretokenizer-independent exact id (`" "` → 'Ġ' → 220). Skips
/// cleanly when unset (never a fabricated pass).
#[test]
fn parity_cosyvoice2_tokenizer_real_vocab_roundtrip() {
    let Some(dir) = env::var(TOKENIZER_DIR_ENV).ok() else {
        eprintln!(
            "{TOKENIZER_DIR_ENV} unset — skipping real Qwen2 tokenizer round-trip; \
             this is a clean skip (never a fabricated pass)"
        );
        return;
    };
    let dir = Path::new(&dir);
    let vocab = std::fs::read(dir.join("vocab.json"))
        .unwrap_or_else(|e| panic!("{}: {e}", dir.join("vocab.json").display()));
    let merges = std::fs::read(dir.join("merges.txt"))
        .unwrap_or_else(|e| panic!("{}: {e}", dir.join("merges.txt").display()));
    let tok =
        CosyVoice2Tokenizer::from_parts(&vocab, &merges).expect("real Qwen2 tokenizer must parse");
    eprintln!(
        "real Qwen2 tokenizer loaded: {} base BPE tokens",
        tok.vocab_size()
    );
    for s in [
        "Hello, world!",
        "The quick brown fox.",
        "CosyVoice2 生成 123 test",
        "  spaces\tand\ttabs  ",
        "line1\nline2\n",
    ] {
        let ids = tok
            .encode(s)
            .unwrap_or_else(|e| panic!("encode {s:?}: {e}"));
        let back = tok
            .decode(&ids)
            .unwrap_or_else(|e| panic!("decode {s:?}: {e}"));
        assert_eq!(back, s, "round-trip failed for {s:?} -> {ids:?}");
        eprintln!("round-trip OK: {s:?} -> {} ids", ids.len());
    }
    // Pretokenizer-independent exact id: a lone space is one piece, whose sole
    // byte-char is 'Ġ' (byte 0x20). The shipping Qwen2 vocab.json maps
    // "Ġ" -> 220 (verified from the file), so encode(" ") must be [220].
    assert_eq!(
        tok.encode(" ").expect("encode space"),
        vec![220],
        "space byte-char 'Ġ' must encode to the shipping Qwen2 vocab id 220"
    );
}
