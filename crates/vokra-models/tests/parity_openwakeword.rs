//! openWakeWord numerical parity harness — env-gated (KWS tier, 2026-08-05).
//!
//! Sibling of `parity_rmvpe.rs` / `parity_cosyvoice2.rs` / `parity_kokoro.rs`:
//! every test that needs a real openWakeWord GGUF is gated on the
//! [`GGUF_ENV`] environment variable and skips cleanly when unset
//! (never a fabricated pass, memory `[[project-real-weight-eval]]`
//! pattern). Once opted in, every failure is hard: a missing / malformed
//! / wrong-shaped fixture is a loud panic (FR-EX-08).
//!
//! # Fixture recipe (owner-side)
//!
//! The upstream `dscripka/openWakeWord` release ships two ONNX models:
//! `embedding_model.onnx` (the frozen Google `speech_embedding` extractor)
//! and one `<wakeword>.onnx` per wake-word (the per-wake-word classifier
//! MLP). Bridge them offline to a merged safetensors + reference JSON:
//!
//! ```text
//! # 1. Fetch upstream ONNX from github.com/dscripka/openWakeWord releases
//! # 2. Merge to Vokra safetensors + config side-car + reference probs (uv):
//! uv run python tools/parity/openwakeword_prepare_checkpoint.py \
//!     --embedding     ~/openwakeword/embedding_model.onnx \
//!     --wakeword      alexa=~/openwakeword/alexa_v0.1.onnx \
//!     --wakeword      hey_jarvis=~/openwakeword/hey_jarvis_v0.1.onnx \
//!     --input-wav     ~/test-speech.wav \
//!     --output-st     ~/openwakeword.safetensors \
//!     --output-config ~/openwakeword_config.json \
//!     --output-ref    ~/openwakeword_reference.json \
//!     --output-wav    ~/openwakeword-16k.wav
//! # 3. Convert safetensors → GGUF (--config is REQUIRED, see below):
//! vokra-cli convert --model openwakeword-op \
//!     --input  ~/openwakeword.safetensors \
//!     --config ~/openwakeword_config.json \
//!     --output ~/openwakeword.gguf
//! # 4. Point the parity harness at all three artefacts:
//! export VOKRA_OPENWAKEWORD_REAL_GGUF=~/openwakeword.gguf
//! export VOKRA_OPENWAKEWORD_REAL_WAV=~/openwakeword-16k.wav
//! export VOKRA_OPENWAKEWORD_REFERENCE_JSON=~/openwakeword_reference.json
//! cargo test -p vokra-models --test parity_openwakeword -- --nocapture
//! ```
//!
//! # Why step 3 takes a `--config` (2026-08-15)
//!
//! This recipe used to omit it, and as written it could not work: the
//! converter stamped none of the seven `vokra.openwakeword.*` keys
//! [`OpenwakewordSession::from_gguf`] requires, so step 4 died at the
//! first load and the harness never ran. The converter now emits the
//! whole chunk group, deriving `n_wakewords` and `embedding_dim` from
//! the classifier tensors and taking the per-wake-word labels from the
//! side-car — those labels exist nowhere in the safetensors (tensors are
//! indexed positionally) and are not invented.
//!
//! Because that gap was invisible to CI — the unit tests below
//! hand-build their GGUF, and everything needing a real one is gated on
//! [`GGUF_ENV`] and skips — two fixture-free tests now run the real
//! converter into the real binder on every CI run:
//! `crates/vokra-convert/tests/openwakeword_op_roundtrip.rs` and
//! `crates/vokra-models/tests/openwakeword_convert_bind.rs`. This
//! harness stays the place real-weight NUMERICS are checked; those two
//! keep the two halves able to talk at all.
//!
//! # Numeric parity contract
//!
//! openWakeWord's per-wake-word head emits one sigmoid probability per
//! rolling window step. The parity check is a per-hop `max |Δ|` bound
//! ([`PROB_ATOL`]) against the upstream `openwakeword` + `onnxruntime`
//! Python reference. At the reference release's default 96-d embedding
//! plus tiny classifier MLP, a bound of `1e-4` corresponds to typical
//! ONNX-RT-vs-Rust float rounding on this class (mirror of
//! `parity_kokoro.rs`'s per-tensor atol scaffold — see
//! `[[feedback-honest-parity-atol]]`).
//!
//! **When the Google `speech_embedding` real-weight bundle wave lands**
//! (deferred per `docs/license-audit.md` §3.1 owner sign-off + real-
//! checkpoint parity), the `parity_openwakeword_gated_hop_probs` test
//! automatically takes its already-wired real branch and performs a
//! hop-by-hop probability match. The runtime wave only needs to bind the
//! extractor and flip
//! `OpenwakewordSession`'s `EmbeddingExtractor::has_real_embedding_weights`
//! flag inside `from_gguf`.

use std::collections::BTreeMap;
use std::env;
use std::path::Path;

use vokra_core::VokraError;
use vokra_core::engines::KwsEngine;
use vokra_models::kws::openwakeword::{
    BoundClassifier, EmbeddingExtractor, OpenwakewordSession, classify_embedding,
};
use vokra_ops::OpenwakewordClassifierWeights;

/// Env var the owner sets to point the gated harness at a real
/// openWakeWord GGUF. Absent = skip cleanly (never a fabricated pass).
const GGUF_ENV: &str = "VOKRA_OPENWAKEWORD_REAL_GGUF";

/// Env var pointing at a 16 kHz mono WAV the reference dumper also
/// consumed (see `tools/parity/openwakeword_prepare_checkpoint.py`).
const WAV_ENV: &str = "VOKRA_OPENWAKEWORD_REAL_WAV";

/// Env var pointing at the reference JSON side-car the prep script
/// emits: `{"hop_probs": {"<wakeword>": [p0, p1, ...]}}`.
const REFERENCE_JSON_ENV: &str = "VOKRA_OPENWAKEWORD_REFERENCE_JSON";

/// Per-hop probability |Δ| tolerance (consumed once the embedding
/// extractor lights up — see the module docs).
const PROB_ATOL: f32 = 1e-4;

#[derive(Debug)]
struct OpenwakewordReference {
    sample_rate: u32,
    predict_chunk_samples: usize,
    wakeword_names: Vec<String>,
    hop_probs: BTreeMap<String, Vec<f32>>,
}

fn reference_number(value: &vokra_core::json::JsonValue, context: &str) -> f32 {
    let number = match value {
        vokra_core::json::JsonValue::Float(value) => *value,
        vokra_core::json::JsonValue::Int(value) => *value as f64,
        other => panic!("{context}: expected JSON number, got {other:?}"),
    };
    assert!(number.is_finite(), "{context}: probability must be finite");
    number as f32
}

fn parse_openwakeword_reference(path: &Path) -> OpenwakewordReference {
    let bytes = std::fs::read(path)
        .unwrap_or_else(|error| panic!("read reference JSON {}: {error}", path.display()));
    let root = vokra_core::json::parse(&bytes)
        .unwrap_or_else(|error| panic!("parse reference JSON {}: {error}", path.display()));

    let required_u32 = |key: &str| -> u32 {
        let raw = root
            .get(key)
            .unwrap_or_else(|| panic!("reference JSON {} missing `{key}`", path.display()))
            .as_u64()
            .unwrap_or_else(|| panic!("reference JSON `{key}` must be a non-negative integer"));
        u32::try_from(raw).unwrap_or_else(|_| panic!("reference JSON `{key}` overflows u32"))
    };

    let wakeword_names = root
        .get("wakeword_names")
        .and_then(vokra_core::json::JsonValue::as_array)
        .unwrap_or_else(|| panic!("reference JSON `wakeword_names` must be an array"))
        .iter()
        .enumerate()
        .map(|(index, value)| {
            value
                .as_str()
                .unwrap_or_else(|| panic!("wakeword_names[{index}] must be a string"))
                .to_owned()
        })
        .collect::<Vec<_>>();
    assert!(
        !wakeword_names.is_empty(),
        "reference JSON wakeword_names is empty"
    );

    let probs_object = root
        .get("hop_probs")
        .and_then(vokra_core::json::JsonValue::as_object)
        .unwrap_or_else(|| panic!("reference JSON `hop_probs` must be an object"));
    let mut hop_probs = BTreeMap::new();
    for (name, value) in probs_object {
        assert!(
            wakeword_names.iter().any(|candidate| candidate == name),
            "reference JSON hop_probs has undeclared wake-word `{name}`"
        );
        let values = value
            .as_array()
            .unwrap_or_else(|| panic!("hop_probs[{name:?}] must be an array"))
            .iter()
            .enumerate()
            .map(|(index, value)| {
                let probability = reference_number(value, &format!("hop_probs[{name:?}][{index}]"));
                assert!(
                    (0.0..=1.0).contains(&probability),
                    "hop_probs[{name:?}][{index}]={probability} is outside [0,1]"
                );
                probability
            })
            .collect::<Vec<_>>();
        assert!(
            !values.is_empty(),
            "hop_probs[{name:?}] is empty; regenerate with the upstream openwakeword package installed"
        );
        assert!(
            hop_probs.insert(name.clone(), values).is_none(),
            "duplicate hop_probs key `{name}`"
        );
    }
    for name in &wakeword_names {
        assert!(
            hop_probs.contains_key(name),
            "reference JSON has no hop_probs entry for `{name}`"
        );
    }

    OpenwakewordReference {
        sample_rate: required_u32("sample_rate"),
        predict_chunk_samples: required_u32("predict_chunk_samples") as usize,
        wakeword_names,
        hop_probs,
    }
}

/// FIXTURE-FREE: the per-wake-word MLP classifier is a real primitive
/// unit-testable independent of a real GGUF. Pins the classifier
/// output shape and sigmoid range for a hand-constructed embedding so
/// the classifier half never silently regresses regardless of the
/// (owner-blocked) Google `speech_embedding` extractor status.
#[test]
fn openwakeword_classify_embedding_sigmoid_range_and_shape() {
    // Two toy wake-words with tiny MLPs (embedding_dim=4, hidden_dim=3).
    // The weights are non-zero so a silent all-zero regression would
    // land at sigmoid(bias) rather than sigmoid(0) — makes the range
    // check catch a silent-zero forward.
    let make_bc = |name: &str, bias0: f32, bias1: f32| BoundClassifier {
        name: name.to_owned(),
        weights: OpenwakewordClassifierWeights {
            embedding_dim: 4,
            hidden_dim: 3,
            linear1_weight: vec![
                0.1, 0.2, -0.1, 0.05, // row 0
                -0.05, 0.1, 0.2, -0.1, // row 1
                0.15, -0.1, 0.1, 0.05, // row 2
            ],
            linear1_bias: vec![bias0, -0.02, 0.03],
            linear2_weight: vec![0.5, -0.3, 0.4],
            linear2_bias: vec![bias1],
        },
    };
    let classifiers = [
        make_bc("alexa", 0.1, -0.05),
        make_bc("hey_jarvis", -0.05, 0.02),
    ];
    let embedding = vec![0.3f32, -0.2, 0.5, -0.1];
    let out = classify_embedding(&classifiers, &embedding).expect("classify");
    assert_eq!(out.len(), 2, "two wake-words → two output tuples");
    for (i, (name, prob)) in out.iter().enumerate() {
        assert_eq!(name, &classifiers[i].name);
        assert!(
            (0.0..=1.0).contains(prob) && prob.is_finite(),
            "wake-word `{name}` prob {prob} must be a sigmoid output in [0,1]"
        );
    }
}

/// FIXTURE-FREE: the loud-partial `EmbeddingExtractor::forward`
/// contract must surface a `VokraError::UnsupportedOp` naming both the
/// env-gate and the parity-harness recipe so an owner reading the
/// error knows exactly where to flip the switch. This pins the
/// FR-EX-08 honest-pending posture even without a real GGUF fixture.
#[test]
fn openwakeword_embedding_extractor_is_loud_partial() {
    let ext = EmbeddingExtractor {
        has_real_embedding_weights: false,
        embedding_dim: 96,
    };
    // The melspec window shape doesn't matter — the loud-partial gate
    // fires before any tensor arithmetic runs.
    let mel = vec![0.0f32; 76 * 32];
    let err = ext
        .forward(&mel)
        .expect_err("loud-partial extractor must not return Ok");
    let msg = match err {
        VokraError::UnsupportedOp(m) => m,
        other => panic!("expected UnsupportedOp, got {other:?}"),
    };
    assert!(
        msg.contains(GGUF_ENV),
        "loud-partial message must name the env-gate `{GGUF_ENV}`: {msg}"
    );
    assert!(
        msg.contains("parity_openwakeword"),
        "loud-partial message must name the parity harness: {msg}"
    );
}

/// GATED: opens a real openWakeWord GGUF and verifies the load path is
/// a genuine bind (real config parse, real per-wake-word classifier
/// tensor bind). This exercises everything up to — and including —
/// the honest-pending gate on the Google `speech_embedding` extractor.
///
/// Skips cleanly when [`GGUF_ENV`] is unset. Once set, all failures
/// are hard: a missing / malformed / wrong-arch fixture fails loudly.
#[test]
fn parity_openwakeword_gguf_smoke() {
    let Some(gguf_path) = env::var(GGUF_ENV).ok() else {
        eprintln!(
            "{GGUF_ENV} unset — skipping openwakeword GGUF parity smoke; \
             this is a clean skip (never a fabricated pass). See the \
             module docs for the fixture recipe."
        );
        return;
    };

    let path = Path::new(&gguf_path);
    let session = OpenwakewordSession::open(path).unwrap_or_else(|e| {
        panic!(
            "openwakeword GGUF at {gguf_path} failed to load: {e:?} \
             (opted-in ⇒ any error is a hard failure — FR-EX-08)"
        )
    });

    let cfg = session.config();
    assert!(
        cfg.n_wakewords >= 1,
        "real openwakeword GGUF must carry at least one wake-word classifier; \
         got {}",
        cfg.n_wakewords
    );
    assert_eq!(
        cfg.sample_rate, 16_000,
        "openwakeword is trained at 16 kHz PCM in; a differently-rated GGUF \
         is either misconfigured or a non-canonical fork (loud-fail)"
    );
    assert_eq!(
        cfg.embedding_dim, 96,
        "reference openwakeword release uses a 96-d shared embedding; a \
         different width breaks the classifier MLP shape (loud-fail)"
    );
    assert_eq!(
        cfg.wakeword_names.len(),
        cfg.n_wakewords,
        "wakeword_names count must match n_wakewords (already checked at \
         load, re-pinned here)"
    );
    assert_eq!(
        session.classifiers().len(),
        cfg.n_wakewords,
        "one classifier per wake-word"
    );
    for bc in session.classifiers() {
        assert_eq!(bc.weights.embedding_dim, cfg.embedding_dim);
        assert!(
            bc.weights.hidden_dim > 0,
            "classifier `{}` must have hidden_dim > 0",
            bc.name
        );
    }
    eprintln!(
        "openwakeword GGUF loaded from {gguf_path}: {} wake-words \
         ({:?}), embedding_dim={}, window_frames={}, sample_rate={} Hz",
        cfg.n_wakewords, cfg.wakeword_names, cfg.embedding_dim, cfg.window_frames, cfg.sample_rate
    );
}

/// GATED: pushes real 16 kHz PCM through the session and compares the
/// emitted per-hop wake-word probabilities against the owner-provisioned
/// reference JSON. While the extractor is pending the test enforces the
/// loud [`VokraError::UnsupportedOp`] posture. Once it lights up, the same
/// test validates the JSON schema, label ordering, chunk size, and every
/// emitted probability within [`PROB_ATOL`] — no fixture-presence panic.
#[test]
fn parity_openwakeword_gated_hop_probs() {
    let Some(gguf_path) = env::var(GGUF_ENV).ok() else {
        eprintln!(
            "{GGUF_ENV} unset — skipping openwakeword hop-probability parity; \
             clean skip. See module docs for the fixture recipe."
        );
        return;
    };
    let Some(wav_path) = env::var(WAV_ENV).ok() else {
        eprintln!(
            "{WAV_ENV} unset — skipping openwakeword hop-probability parity; \
             clean skip. See module docs for the fixture recipe."
        );
        return;
    };
    // REFERENCE_JSON_ENV is consumed once the embedding extractor lights
    // up; mention it early so an owner staging artifacts sees the missing
    // side-car before the runtime transition.
    if env::var(REFERENCE_JSON_ENV).is_err() {
        eprintln!(
            "note: {REFERENCE_JSON_ENV} unset — the loud-partial gate \
             still fires without it; the sidecar becomes required once \
             the embedding extractor lights up."
        );
    }

    let mut session = OpenwakewordSession::open(&gguf_path)
        .unwrap_or_else(|e| panic!("openwakeword GGUF at {gguf_path} failed to load: {e:?}"));

    // Read the WAV as 16 kHz mono f32 through the shared `silero_vad::wav`
    // helper (already used by silero-vad's own parity harnesses).
    let wav = vokra_models::silero_vad::wav::read_wav_f32(&wav_path)
        .unwrap_or_else(|e| panic!("wav read {wav_path} failed: {e:?}"));
    assert_eq!(
        wav.sample_rate, 16_000,
        "openwakeword parity fixture must be 16 kHz mono (got {} Hz)",
        wav.sample_rate
    );
    assert!(
        !wav.samples.is_empty(),
        "openwakeword parity WAV must not be empty"
    );

    // Push a nontrivial slice — at least one hop's worth (`hop_samples`
    // frames = 160 samples at the reference release). The current
    // landing hits the loud-partial gate here.
    let hop = session.config().hop_samples;
    let take = (hop * 8).min(wav.samples.len()); // 8 hops = ~80 ms
    let result = session.push_pcm16k(&wav.samples[..take]);

    match result {
        Err(VokraError::UnsupportedOp(msg)) => {
            assert!(
                msg.contains("VOKRA_OPENWAKEWORD_REAL_GGUF"),
                "loud-partial message must name the env-gate for owner flip: {msg}"
            );
            eprintln!(
                "openwakeword hop-probability parity: loud-partial gate fired \
                 as expected — the embedding extractor is deferred to \
                 owner-provisioned Google speech_embedding bundle. \
                 When that lands, this test flips to hop_probs comparison \
                 (see the module docs)."
            );
        }
        Ok(hops) => {
            let ref_json_path = env::var(REFERENCE_JSON_ENV).unwrap_or_else(|_| {
                panic!(
                    "the embedding extractor is now real but {REFERENCE_JSON_ENV} \
                     is unset — provide the reference JSON side-car per \
                     tools/parity/openwakeword_prepare_checkpoint.py"
                )
            });
            let reference = parse_openwakeword_reference(Path::new(&ref_json_path));
            assert_eq!(reference.sample_rate, 16_000, "reference sample rate");
            assert_eq!(
                reference.predict_chunk_samples, take,
                "the test pushes exactly one upstream predict chunk"
            );
            assert_eq!(
                reference.wakeword_names,
                session.config().wakeword_names,
                "reference/GGUF wake-word label ordering drifted"
            );
            assert!(
                !hops.is_empty(),
                "real embedding path must emit at least one hop for the \
                 pushed PCM slice"
            );
            let mut seen = BTreeMap::<String, usize>::new();
            let mut max_abs = 0.0_f32;
            for (name, probability) in &hops {
                let index = seen.entry(name.clone()).or_default();
                let expected = reference
                    .hop_probs
                    .get(name)
                    .unwrap_or_else(|| panic!("runtime emitted unknown wake-word `{name}`"))
                    .get(*index)
                    .copied()
                    .unwrap_or_else(|| {
                        panic!("runtime emitted more `{name}` hops than the reference")
                    });
                let abs = (probability - expected).abs();
                max_abs = max_abs.max(abs);
                assert!(
                    abs <= PROB_ATOL,
                    "{name} hop {index}: runtime={probability:.9e}, reference={expected:.9e}, abs={abs:.9e} > {PROB_ATOL:.1e}"
                );
                *index += 1;
            }
            for name in &reference.wakeword_names {
                assert_eq!(
                    seen.get(name).copied(),
                    Some(1),
                    "one predict chunk must emit exactly one probability for `{name}`"
                );
            }
            eprintln!(
                "openwakeword real hop parity: {} labels, max_abs={max_abs:.9e}",
                reference.wakeword_names.len()
            );
        }
        Err(other) => {
            panic!("openwakeword push_pcm16k returned unexpected error: {other:?}")
        }
    }
}
