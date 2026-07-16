//! M4-05 T24 — staged CSM parity (backbone hidden / c0 logits / depth
//! logits / frame codes / decode PCM) against the `tools/parity/csm_dump.py`
//! fixtures.
//!
//! # Gating (fabricated pass 禁止)
//!
//! - The **committed synthetic fixture** (`tests/parity/csm/self-test/`,
//!   written by `csm_dump.py self-test`) pins the file/manifest format:
//!   the reader + sha256 verification run in every CI build. It carries no
//!   reference semantics.
//! - The **real reference legs** are env-gated on `VOKRA_CSM_PARITY_DIR`
//!   (owner sets it after the T29 dump): absent → clean skip with a
//!   printed reason. Present → the loader now *binds* real weights
//!   (`CsmBackboneWeights::from_gguf` reads the documented torchtune names),
//!   but those names are **not header-confirmed** (gated repo), so a real
//!   comparison must not auto-run: the leg **panics loudly** naming the two
//!   owner steps (confirm names vs. the real header; wire the staged
//!   comparison) — never a fabricated pass.
//!
//! # Judgement (ADR M4-05 §D7 / tools/parity/README-csm.md)
//!
//! Frame codes: bit-exact. Float stages: FP32 `atol = 0.01` (NFR-QL-01)
//! starting point; per-tensor relaxation only with an architectural-bound
//! rationale recorded in rustdoc + ADR + CI (Kokoro PROSODY_F0_ATOL
//! precedent). No relaxation exists today (no real comparison has run).

use std::path::{Path, PathBuf};

/// FP32 default tolerance (NFR-QL-01). No per-tensor override exists yet —
/// the first real comparison (post-T29) decides honestly whether one is
/// architecturally required.
#[allow(dead_code)] // consumed by the T29 flip-the-switch leg below
const ATOL: f32 = 0.01;

fn self_test_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/parity/csm/self-test")
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

/// Recomputes every `sha256 <name> <hex>` line of a fixture manifest —
/// the same verification the dumper self-test performs, repeated on the
/// Rust side so a bit-rotted fixture fails loudly in CI.
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
        let got = sha256_hex(&bytes);
        assert_eq!(got, want, "{name}: fixture bytes drifted from the manifest");
        checked += 1;
    }
    assert!(checked >= 5, "manifest lists too few files ({checked})");
}

/// Minimal SHA-256 (zero-dep — vokra ships no hashing crate; this is the
/// FIPS 180-4 compression loop over the fixture bytes, small enough to
/// keep in a test).
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
    let bit_len = (data.len() as u64) * 8;
    let mut msg = data.to_vec();
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bit_len.to_be_bytes());
    for block in msg.chunks_exact(64) {
        let mut w = [0u32; 64];
        for (i, c) in block.chunks_exact(4).enumerate() {
            w[i] = u32::from_be_bytes([c[0], c[1], c[2], c[3]]);
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
         `python3 tools/parity/csm_dump.py self-test --out tests/parity/csm/self-test`",
        dir.display()
    );
    verify_manifest(&dir);
    // Shape sanity from context.json (stdlib parse — the values are small).
    let ctx = std::fs::read_to_string(dir.join("context.json")).expect("context.json");
    assert!(ctx.contains("\"n_codebooks\": 4"));
    let codes = read_u32(&dir.join("frame_codes.u32"));
    assert_eq!(codes.len(), 3 * 4, "[n_frames, n_codebooks]");
    assert!(
        (0..3).all(|t| codes[t * 4..(t + 1) * 4].iter().any(|&c| c != 0)),
        "the synthetic fixture must not contain an EOS (all-zero) frame"
    );
    let hidden = read_f32(&dir.join("backbone_hidden.f32"));
    assert_eq!(hidden.len(), 3 * 8);
    assert!(hidden.iter().all(|v| v.is_finite()));
    let pcm = read_f32(&dir.join("decode_pcm.f32"));
    assert_eq!(pcm.len(), 3 * 8);
}

// ---------------------------------------------------------------------------
// Env-gated legs: real reference fixtures (owner, post-T29)
// ---------------------------------------------------------------------------

#[test]
fn staged_reference_parity_is_env_gated() {
    let Some(dir) = std::env::var_os("VOKRA_CSM_PARITY_DIR") else {
        println!(
            "skip: VOKRA_CSM_PARITY_DIR not set — the real staged reference is the \
             T29 owner dump (tools/parity/README-csm.md); this is a clean gated \
             skip, not a pass"
        );
        return;
    };
    let dir = PathBuf::from(dir);
    verify_manifest(&dir);
    // Fixtures exist and verify. `CsmBackboneWeights::from_gguf` now binds
    // the documented torchtune names, but they are NOT header-confirmed
    // (sesame/csm-1b is gated), so auto-running a comparison would risk
    // reporting a pass off unverified naming. This leg *fails loudly* until
    // the owner (1) confirms the tensor names against the real checkpoint
    // header and (2) wires the staged comparison in place of this panic.
    panic!(
        "VOKRA_CSM_PARITY_DIR is set and the fixture manifest verifies. The \
         runtime now binds real weights (CsmBackboneWeights::from_gguf reads \
         the documented torchtune names), but those names are not \
         header-confirmed (gated repo). Owner: confirm the names against the \
         real checkpoint header, then replace this panic with the staged \
         comparison (frame codes bit-exact; float stages atol = {ATOL}). \
         Refusing to report a pass that did not run (fabricated pass 禁止)."
    );
}
