//! M4-06 T15 — staged Moshi parity (backbone hidden / text logits / text
//! tokens / undelayed frame codes) against the `tools/parity/moshi_dump.py`
//! fixtures.
//!
//! # Gating (fabricated pass 禁止)
//!
//! - The **committed synthetic fixture** (`tests/parity/moshi/self-test/`,
//!   written by `moshi_dump.py self-test`) pins the file/manifest format:
//!   the reader + sha256 verification run in every CI build. It carries no
//!   reference semantics.
//! - The **real reference leg** is env-gated on `VOKRA_MOSHI_PARITY_DIR`
//!   (owner sets it after the T29 weight sourcing + `moshi_dump.py real`
//!   run + `vokra-cli convert --model moshi`): absent → clean skip with a
//!   printed reason. Present → the comparison **actually fires**: the
//!   directory must also carry the converted `model.gguf`, whose weights
//!   the runtime binds through the real `from_gguf` path (unlike CSM,
//!   the Moshi binding landed with the T02 manifest — this leg is a true
//!   flip-the-switch, not a stub).
//!
//! # Judgement (ADR M4-06 §D7)
//!
//! Token streams (text + frame codes): **bit-exact** under the greedy
//! (argmax) dump contract. Float stages: FP32 `atol = 0.01` (NFR-QL-01)
//! starting point; per-tensor relaxation only with an architectural-bound
//! rationale recorded in rustdoc + ADR + CI (Kokoro PROSODY_F0_ATOL
//! precedent). No relaxation exists today (no real comparison has run).

use std::path::{Path, PathBuf};

use vokra_models::moshi::{
    MoshiBackboneState, MoshiConfig, MoshiFrameOut, MoshiGenerationState, MoshiModel,
    MoshiSamplerPair,
};

/// FP32 default tolerance (NFR-QL-01).
const ATOL: f32 = 0.01;

fn self_test_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/parity/moshi/self-test")
}

fn read_f32(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    assert_eq!(bytes.len() % 4, 0, "{}: not f32-aligned", path.display());
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn read_u32(path: &Path) -> Vec<u32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    assert_eq!(bytes.len() % 4, 0, "{}: not u32-aligned", path.display());
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Recomputes every `sha256 <name> <hex>` manifest line (bit-rot guard —
/// the csm/kokoro fixture discipline).
fn verify_manifest(dir: &Path) {
    let manifest = std::fs::read_to_string(dir.join("manifest.txt"))
        .unwrap_or_else(|e| panic!("{}: manifest.txt: {e}", dir.display()));
    let mut checked = 0usize;
    for line in manifest.lines() {
        let Some(rest) = line.strip_prefix("sha256 ") else {
            continue;
        };
        let mut parts = rest.split_whitespace();
        let (name, want) = (
            parts.next().expect("manifest name"),
            parts.next().expect("manifest hex"),
        );
        let bytes = std::fs::read(dir.join(name))
            .unwrap_or_else(|e| panic!("{}: {name}: {e}", dir.display()));
        assert_eq!(
            sha256_hex(&bytes),
            want,
            "{name}: fixture bytes drifted from the manifest"
        );
        checked += 1;
    }
    assert!(checked >= 6, "manifest lists too few files ({checked})");
}

/// Minimal SHA-256 (zero-dep; FIPS 180-4 compression loop — test-local,
/// the parity_csm.rs precedent).
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
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    let mut msg = data.to_vec();
    let bit_len = (data.len() as u64).wrapping_mul(8);
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bit_len.to_be_bytes());
    for chunk in msg.chunks_exact(64) {
        let mut w = [0u32; 64];
        for (i, word) in chunk.chunks_exact(4).enumerate() {
            w[i] = u32::from_be_bytes([word[0], word[1], word[2], word[3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let (mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh) =
            (h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]);
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let t1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }
    h.iter().map(|v| format!("{v:08x}")).collect()
}

fn json_usize(v: &vokra_core::json::JsonValue, key: &str) -> usize {
    match v {
        vokra_core::json::JsonValue::Object(map) => map
            .iter()
            .find(|(k, _)| k == key)
            .and_then(|(_, v)| match v {
                vokra_core::json::JsonValue::Int(n) => Some(*n as usize),
                _ => None,
            })
            .unwrap_or_else(|| panic!("context.json: missing numeric `{key}`")),
        _ => panic!("context.json: not an object"),
    }
}

// ---------------------------------------------------------------------------
// Always-on leg: committed synthetic fixture (format pin — no reference
// semantics)
// ---------------------------------------------------------------------------

#[test]
fn synthetic_fixture_manifest_roundtrip() {
    let dir = self_test_dir();
    assert!(
        dir.join("manifest.txt").exists(),
        "committed self-test fixture missing at {} — regenerate with \
         `python3 tools/parity/moshi_dump.py self-test --out tests/parity/moshi/self-test`",
        dir.display()
    );
    verify_manifest(&dir);
    let ctx = vokra_core::json::parse(&std::fs::read(dir.join("context.json")).unwrap())
        .expect("context.json parses");
    let (n_steps, n_user, n_channels) = (
        json_usize(&ctx, "n_steps"),
        json_usize(&ctx, "n_user"),
        json_usize(&ctx, "n_channels"),
    );
    let (d, text_card, dep_q, n_emitted) = (
        json_usize(&ctx, "d_model"),
        json_usize(&ctx, "text_card"),
        json_usize(&ctx, "dep_q"),
        json_usize(&ctx, "n_emitted"),
    );
    assert_eq!(
        read_u32(&dir.join("user_codes.u32")).len(),
        n_steps * n_user
    );
    assert_eq!(
        read_u32(&dir.join("input_tokens.u32")).len(),
        n_steps * n_channels
    );
    let hidden = read_f32(&dir.join("backbone_hidden.f32"));
    assert_eq!(hidden.len(), n_steps * d);
    assert!(hidden.iter().all(|v| v.is_finite()));
    assert_eq!(
        read_f32(&dir.join("text_logits.f32")).len(),
        n_steps * text_card
    );
    assert_eq!(read_u32(&dir.join("text_tokens.u32")).len(), n_steps);
    assert_eq!(
        read_u32(&dir.join("frame_codes.u32")).len(),
        n_emitted * dep_q
    );
}

// ---------------------------------------------------------------------------
// Env-gated leg: real reference fixtures + converted GGUF (owner, post-T29)
// ---------------------------------------------------------------------------

#[test]
fn staged_reference_parity_is_env_gated_and_fires_when_supplied() {
    let Some(dir) = std::env::var_os("VOKRA_MOSHI_PARITY_DIR") else {
        println!(
            "skip: VOKRA_MOSHI_PARITY_DIR not set — the real staged reference is \
             the T29 owner dump (tools/parity/moshi_dump.py real) plus the \
             converted model.gguf; this is a clean gated skip, not a pass"
        );
        return;
    };
    let dir = PathBuf::from(dir);
    verify_manifest(&dir);

    // Unlike CSM (whose real binding was a T29 stub), the Moshi
    // `from_gguf` path landed with the T02 manifest — so with fixtures
    // AND the converted GGUF present, the comparison genuinely runs.
    let gguf_path = dir.join("model.gguf");
    assert!(
        gguf_path.exists(),
        "VOKRA_MOSHI_PARITY_DIR fixtures verify but {} is missing — convert the \
         T29 checkpoint (`vokra-cli convert --model moshi ...`) into the fixture \
         dir. Refusing to report a pass that did not run (fabricated pass 禁止).",
        gguf_path.display()
    );
    let bytes = std::fs::read(&gguf_path).expect("read model.gguf");
    let file = vokra_core::gguf::GgufFile::parse(bytes).expect("parse model.gguf");
    let cfg = MoshiConfig::from_gguf(&file).expect("vokra.moshi.* chunk group");
    cfg.validate_for_forward().expect("hparams populated");
    let bb =
        vokra_models::moshi::MoshiBackboneWeights::from_gguf(&file, &cfg).expect("backbone bind");
    let dp = vokra_models::moshi::MoshiDepthWeights::from_gguf(&file, &cfg).expect("depth bind");
    assert!(!bb.is_synthesized && !dp.is_synthesized);
    let model = MoshiModel::new(cfg.clone(), bb, dp).expect("model");

    let ctx = vokra_core::json::parse(&std::fs::read(dir.join("context.json")).unwrap())
        .expect("context.json");
    let n_steps = json_usize(&ctx, "n_steps");
    let n_user = json_usize(&ctx, "n_user");
    let n_channels = json_usize(&ctx, "n_channels");
    let d = json_usize(&ctx, "d_model");
    assert_eq!(n_user, cfg.n_user_streams(), "fixture/config user streams");
    assert_eq!(n_channels, cfg.n_channels(), "fixture/config channels");
    assert_eq!(d, cfg.temporal.d_model, "fixture/config d_model");

    let user = read_u32(&dir.join("user_codes.u32"));
    let inputs = read_u32(&dir.join("input_tokens.u32"));
    let ref_hidden = read_f32(&dir.join("backbone_hidden.f32"));
    let ref_tlogits = read_f32(&dir.join("text_logits.f32"));
    let ref_ttoks = read_u32(&dir.join("text_tokens.u32"));
    let ref_codes = read_u32(&dir.join("frame_codes.u32"));

    // ---- Stage A: backbone per-step (independent of the ring plumbing).
    let mut bb_state = MoshiBackboneState::new(&cfg).unwrap();
    let mut hidden = vec![0.0f32; cfg.temporal.d_model];
    let mut tlogits = vec![0.0f32; cfg.text_card];
    let mut worst_hidden = 0.0f32;
    let mut worst_logits = 0.0f32;
    for s in 0..n_steps {
        let row = &inputs[s * n_channels..(s + 1) * n_channels];
        model
            .backbone()
            .step_into(&mut bb_state, row, &mut hidden)
            .expect("backbone step");
        for (a, b) in hidden.iter().zip(&ref_hidden[s * d..(s + 1) * d]) {
            worst_hidden = worst_hidden.max((a - b).abs());
        }
        model
            .backbone()
            .text_logits_into(&hidden, &mut tlogits)
            .expect("text head");
        for (a, b) in tlogits
            .iter()
            .zip(&ref_tlogits[s * cfg.text_card..(s + 1) * cfg.text_card])
        {
            worst_logits = worst_logits.max((a - b).abs());
        }
    }
    assert!(
        worst_hidden <= ATOL,
        "backbone hidden max |Δ| = {worst_hidden} > atol {ATOL}"
    );
    assert!(
        worst_logits <= ATOL,
        "text logits max |Δ| = {worst_logits} > atol {ATOL}"
    );

    // ---- Stage B: the full greedy loop — undelayed streams bit-exact.
    let mut state = MoshiGenerationState::new(&cfg).unwrap();
    let mut samplers = MoshiSamplerPair::greedy();
    let mut out = MoshiFrameOut::new(&cfg);
    let mut got_text = Vec::new();
    let mut got_codes = Vec::new();
    for s in 0..n_steps {
        let row = &user[s * n_user..(s + 1) * n_user];
        if model
            .step_into(&mut state, row, &mut samplers, &mut out)
            .expect("full step")
        {
            got_text.push(out.text);
            got_codes.extend_from_slice(&out.audio);
        }
    }
    assert_eq!(
        got_text, ref_ttoks,
        "undelayed text stream must be bit-exact"
    );
    assert_eq!(
        got_codes, ref_codes,
        "undelayed frame codes must be bit-exact"
    );

    // ---- Stage C (M4 cc-06): the mmap + mapped-lazy load reproduces the
    // resident token streams BIT-exactly against the same reference. The
    // model is rebuilt through the from_path assembly (real mmap of the
    // same model.gguf, head eager + per-layer block materialization) and
    // driven through the identical greedy loop.
    let mapped_model = {
        use std::sync::Arc;
        let file = Arc::new(vokra_mmap::open_gguf(&gguf_path).expect("mmap model.gguf"));
        let head = vokra_models::moshi::MoshiBackboneWeights::head_from_gguf(&file, &cfg)
            .expect("mapped head bind");
        let mapped = vokra_models::moshi::MappedTemporalBlocks::bind(Arc::clone(&file), &cfg)
            .expect("mapped block bind");
        let backbone = vokra_models::moshi::MoshiBackbone::new_mapped(cfg.clone(), head, mapped)
            .expect("mapped backbone");
        let depth_w = vokra_models::moshi::MoshiDepthWeights::from_gguf(&file, &cfg)
            .expect("depth bind (mapped leg)");
        let depth = vokra_models::moshi::MoshiDepthTransformer::new(cfg.clone(), depth_w)
            .expect("depformer");
        MoshiModel::from_parts(backbone, depth).expect("mapped model")
    };
    let mut state = MoshiGenerationState::new(&cfg).unwrap();
    let mut samplers = MoshiSamplerPair::greedy();
    let mut out = MoshiFrameOut::new(&cfg);
    let mut mapped_text = Vec::new();
    let mut mapped_codes = Vec::new();
    for s in 0..n_steps {
        let row = &user[s * n_user..(s + 1) * n_user];
        if mapped_model
            .step_into(&mut state, row, &mut samplers, &mut out)
            .expect("mapped full step")
        {
            mapped_text.push(out.text);
            mapped_codes.extend_from_slice(&out.audio);
        }
    }
    assert_eq!(
        mapped_text, ref_ttoks,
        "mapped-lazy text stream must be bit-exact vs the reference"
    );
    assert_eq!(
        mapped_codes, ref_codes,
        "mapped-lazy frame codes must be bit-exact vs the reference"
    );

    println!(
        "moshi staged parity: hidden max |Δ| = {worst_hidden:.3e}, text logits \
         max |Δ| = {worst_logits:.3e}, {} frames bit-exact (resident) + {} \
         frames bit-exact (mmap mapped-lazy)",
        got_text.len(),
        mapped_text.len()
    );
}
