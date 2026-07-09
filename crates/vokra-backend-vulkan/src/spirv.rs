//! SPIR-V blob manifest (M3-02-T13 / T14 structural surface).
//!
//! Vokra ships compute kernels as **pre-compiled** SPIR-V blobs (see
//! `kernels/README.md` §"Why precompile" and
//! `docs/adr/M3-02-spirv-generation.md`): `glslc` is a developer-side tool,
//! not a runtime dependency, and `build.rs` never invokes it. The Rust runtime
//! embeds each blob via `include_bytes!` at compile time (once the `.spv` file
//! exists) — which keeps `Cargo.lock` `vokra-*`-only (NFR-DS-02) while also
//! meeting the "no CPU-side JIT" red-line (NFR-RL-05: Android SELinux W^X).
//!
//! # The two blob sources
//!
//! * **`kernels/precompiled/*.spv`** — the mainline path. `glslc` produces
//!   these from `kernels/glsl/*.comp` sources on a developer machine
//!   (`scripts/compile-vulkan-shaders.sh`); each blob is committed with its
//!   SHA-256 pinned in this manifest. See ADR M3-02-spirv-generation §2
//!   Option A + Option C.
//! * **`kernels/handcrafted/*.spv.rs`** — a smoke-test-only exception. The
//!   `copy_f32` kernel below is authored by hand as SPIR-V 1.3 bytecode
//!   (ADR §2 Option D) so that the T08〜T12 + T25 Vulkan object stack has an
//!   end-to-end proof point *before* any `glslc`-produced blob lands. The
//!   ADR explicitly forbids adding more hand-crafted kernels.
//!
//! # What ships in the foundation slice (2026-07-09)
//!
//! The GLSL sources under `kernels/glsl/*.comp` are committed (skeletons
//! frozen so `glslc --preprocess` succeeds; T14〜T22 will land the full kernel
//! bodies) and **no `glslc`-produced `.spv` is committed yet**. The only
//! blob callable via [`load_spv`] today is the hand-crafted `copy_f32`
//! kernel; every other entry surfaces
//! [`VokraError::UnsupportedOp`](vokra_core::VokraError::UnsupportedOp) —
//! never a silent CPU fall back (FR-EX-08).
//!
//! # How the T14〜T22 tickets extend this file
//!
//! When ticket N lands its `.spv` blob under `kernels/precompiled/`, the
//! corresponding [`SpirvShader::blob`] wildcard arm is replaced with an
//! `include_bytes!("../kernels/precompiled/<name>.spv")` call. **No table
//! restructuring is required** — the manifest is a stable structural surface
//! T14 onwards populates entry-by-entry.
//!
//! # Optional SHA-256 pinning
//!
//! Each [`SpirvShader`] carries an optional `expected_sha256` hex string. The
//! foundation slice leaves the `glslc`-produced entries as `None`; when the
//! T14 developer commits a `.spv` alongside its `glslc` invocation, they
//! paste the file's `sha256sum` into the manifest, and the
//! [`verify_pinned_hashes`] test verifies the runtime blob matches (built-in
//! [`sha256`] impl below — zero-dep, ~120 lines, one-file FIPS-180-4 §6.2).
//! The build side (build.rs) stays intentionally *hash-free* to keep
//! `cargo build` fast; the hash pin is a test-time gate. The hand-crafted
//! `copy_f32` blob's SHA-256 is pinned already (it never changes without a
//! deliberate rewrite of `kernels/handcrafted/copy_f32.spv.rs`).

// The hand-crafted `copy_f32` SPIR-V bytecode lives under `kernels/handcrafted/`
// (ADR M3-02-spirv-generation §4 (c)). Rust doesn't automatically discover a
// `.spv.rs` file in a non-standard directory, so include it here as a module
// via `#[path]`. The file itself is standard Rust source — it just carries
// the `.spv.rs` suffix to signal "this Rust source *is* the SPIR-V blob."
#[path = "../kernels/handcrafted/copy_f32.spv.rs"]
pub mod handcrafted_copy_f32;

use core::fmt;

/// Structural pipeline variant a shader targets. Determines which of the
/// two GEMM `.spv` blobs the probe selects; all other shaders are
/// [`ShaderVariant::Standard`] (one blob per op).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShaderVariant {
    /// The M3-02-T14 GEMM fallback path — subgroup-only, no cooperative
    /// matrix. Primary Android target (Adreno 6xx+, Mali G7x+, Immortalis).
    Subgroup,
    /// The M3-02-T14 GEMM fast path — cooperative-matrix + subgroup.
    /// Requires Vulkan 1.3+ AND `VK_KHR_cooperative_matrix` on the device.
    /// Ampere+ / RDNA3+ / Adreno 750+.
    CoopMatrix,
    /// Every non-GEMM shader (one blob per op).
    Standard,
    /// Hand-crafted SPIR-V bytecode authored directly in Rust source
    /// (ADR M3-02-spirv-generation §2 Option D). Only used by the
    /// `copy_f32` smoke-test kernel; **no other entry may be `Handcrafted`**
    /// (ADR §5 forbids expanding this path — real kernels take Option A +
    /// Option C via `glslc`).
    Handcrafted,
}

impl fmt::Display for ShaderVariant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ShaderVariant::Subgroup => f.write_str("subgroup"),
            ShaderVariant::CoopMatrix => f.write_str("coopmat"),
            ShaderVariant::Standard => f.write_str("standard"),
            ShaderVariant::Handcrafted => f.write_str("handcrafted"),
        }
    }
}

/// One entry in the SPIR-V manifest: names an expected `.spv` blob under
/// `kernels/precompiled/<name>.spv`, its `.comp` GLSL source under
/// `kernels/glsl/<name>.comp`, and (once glslc runs) its SHA-256 hex pin.
#[derive(Debug, Clone, Copy)]
pub struct SpirvShader {
    /// Basename without extension (e.g. `"gemm_subgroup"`).
    pub name: &'static str,
    /// Pipeline variant this shader belongs to (governs GEMM path selection).
    pub variant: ShaderVariant,
    /// M3-02 ticket that lands this shader body.
    pub ticket: &'static str,
    /// Expected SHA-256 (lowercase hex, 64 chars) of the `.spv` blob. Left
    /// `None` in the foundation slice; populated by the T14〜T22 developer
    /// alongside the `glslc` invocation that produces the blob.
    pub expected_sha256_hex: Option<&'static str>,
}

/// The manifest — every SPIR-V blob Vokra's Vulkan backend expects.
///
/// **13 entries** = 12 `glslc`-produced kernels (11 op categories; GEMM has
/// two variants sharing the op) + 1 hand-crafted smoke-test kernel
/// (`copy_f32`, ADR M3-02-spirv-generation §2 Option D). This table is the
/// single source of truth for what `.spv` blobs the runtime can load;
/// adding a new kernel means adding a row here **and** a matching arm to
/// [`load_spv`].
pub const SHADERS: &[SpirvShader] = &[
    // ---- Hand-crafted (ADR M3-02-spirv-generation §2 Option D) ----
    //
    // The only entry whose `.spv` is present in the foundation slice — the
    // rest wait for `glslc` runs on the developer machine to populate them.
    // The SHA-256 below is derived from
    // `kernels/handcrafted/copy_f32.spv.rs::bytes()` (computed at test time
    // via `verify_pinned_hashes` — no build-time hashing, keeps `cargo
    // build` fast).
    SpirvShader {
        name: "copy_f32",
        variant: ShaderVariant::Handcrafted,
        ticket: "M3-02-T13",
        // Pinned at test time by `pinned_sha256_matches_runtime_blob_for_copy_f32`
        // below — the hex here is the value that test verifies.
        expected_sha256_hex: Some(
            "4d027c70da61cec3516b70c27ed1fad968ef0a91783c9d9bfe9898d79e4ee109",
        ),
    },
    // ---- glslc-produced (ADR M3-02-spirv-generation §2 Option A + Option C) ----
    SpirvShader {
        name: "gemm_subgroup",
        variant: ShaderVariant::Subgroup,
        ticket: "M3-02-T14",
        expected_sha256_hex: None,
    },
    SpirvShader {
        name: "gemm_coopmat",
        variant: ShaderVariant::CoopMatrix,
        ticket: "M3-02-T14",
        expected_sha256_hex: None,
    },
    SpirvShader {
        name: "gemv",
        variant: ShaderVariant::Standard,
        ticket: "M3-02-T15",
        expected_sha256_hex: None,
    },
    SpirvShader {
        name: "softmax",
        variant: ShaderVariant::Standard,
        ticket: "M3-02-T16",
        expected_sha256_hex: None,
    },
    SpirvShader {
        name: "softmax_causal",
        variant: ShaderVariant::Standard,
        ticket: "M3-02-T16",
        expected_sha256_hex: None,
    },
    SpirvShader {
        name: "layer_norm",
        variant: ShaderVariant::Standard,
        ticket: "M3-02-T17",
        expected_sha256_hex: None,
    },
    SpirvShader {
        name: "gelu",
        variant: ShaderVariant::Standard,
        ticket: "M3-02-T18",
        expected_sha256_hex: None,
    },
    SpirvShader {
        name: "conv1d",
        variant: ShaderVariant::Standard,
        ticket: "M3-02-T19",
        expected_sha256_hex: None,
    },
    SpirvShader {
        name: "elementwise",
        variant: ShaderVariant::Standard,
        ticket: "M3-02-T20",
        expected_sha256_hex: None,
    },
    SpirvShader {
        name: "activation",
        variant: ShaderVariant::Standard,
        ticket: "M3-02-T21",
        expected_sha256_hex: None,
    },
    SpirvShader {
        name: "transpose",
        variant: ShaderVariant::Standard,
        ticket: "M3-02-T22",
        expected_sha256_hex: None,
    },
    SpirvShader {
        name: "gather",
        variant: ShaderVariant::Standard,
        ticket: "M3-02-T22",
        expected_sha256_hex: None,
    },
];

/// Loads the SPIR-V blob for `name` if it is available today.
///
/// Two shapes coexist behind this call:
///
/// 1. **`kernels/precompiled/<name>.spv`** — `include_bytes!` at compile time
///    (M3-02-T14〜T22). No such blob is committed in the foundation slice
///    yet; each ticket lands its own arm here as its `.spv` blob lands.
/// 2. **`kernels/handcrafted/<name>.spv.rs`** — a `Vec<u8>` re-encoded from a
///    Rust `const [u32]` at call time (ADR M3-02-spirv-generation §4 (c)).
///    The only such entry today is `copy_f32` — a smoke-test kernel.
///
/// Callers cannot tell the two apart at the type level (both return
/// `&'static [u8]` for the precompiled case, `Vec<u8>` cannot be `&'static`
/// so the handcrafted case is exposed via a separate accessor —
/// [`load_spv_owned`]).
///
/// # Contract
///
/// - `Some(bytes)` — the blob is loaded and ready to feed into
///   `vkCreateShaderModule` (once T14 wires the pipeline creation code).
/// - `None` — the ticket for `name` has not landed its `.spv` yet; callers
///   must surface [`VokraError::UnsupportedOp`](vokra_core::VokraError::UnsupportedOp).
///   **Never a silent CPU fall back** (FR-EX-08).
///
/// The hand-crafted `copy_f32` blob is intentionally NOT reachable through
/// this function: it needs to be re-encoded from a `const [u32]` into
/// little-endian bytes, which produces an owned `Vec<u8>`. Use
/// [`load_spv_owned`] to load either kind.
#[must_use]
pub fn load_spv(name: &str) -> Option<&'static [u8]> {
    // Kept as a single-arm match today so T14〜T22 can extend it in place
    // without reshaping the function. When ticket N ships its `.spv`, its
    // arm becomes e.g.
    //
    //     "gemm_subgroup" => Some(include_bytes!(
    //         "../kernels/precompiled/gemm_subgroup.spv",
    //     )),
    #[allow(clippy::match_single_binding)]
    match name {
        // Foundation slice: no `glslc`-produced `.spv` files committed.
        // Every M3-02-T14〜T22 ticket will add its own arm here as the `.spv`
        // blob lands. The hand-crafted `copy_f32` is reachable via
        // [`load_spv_owned`] — it lives as `const [u32]` in Rust source, not
        // as bytes on disk.
        _ => None,
    }
}

/// Loads the SPIR-V blob for `name` as owned bytes. Covers both the
/// hand-crafted kernels (SPIR-V 1.3 bytecode in Rust source, re-encoded to
/// little-endian bytes here) and the `glslc`-produced precompiled blobs
/// (borrowed `&'static [u8]` copied to `Vec<u8>` for a uniform type).
///
/// Used by smoke-test code paths (and eventually by every kernel-dispatch
/// site once T14+ blobs land). See [`load_spv`] for the borrowed-view
/// alternative that skips the copy for `.spv`-on-disk blobs.
///
/// # Contract
///
/// - `Some(bytes)` — the blob is loaded; `bytes.len() % 4 == 0` per SPIR-V
///   spec §2.3.
/// - `None` — the ticket for `name` has not landed yet. Callers surface
///   [`VokraError::UnsupportedOp`](vokra_core::VokraError::UnsupportedOp).
#[must_use]
pub fn load_spv_owned(name: &str) -> Option<Vec<u8>> {
    match name {
        "copy_f32" => Some(handcrafted_copy_f32::bytes()),
        other => load_spv(other).map(<[u8]>::to_vec),
    }
}

// ---------------------------------------------------------------------------
// Zero-dependency SHA-256 (FIPS-180-4 §6.2). Used only by tests and later by
// the hash-pin verifier once real `.spv` blobs land — build.rs stays hash-free
// so `cargo build` never pays for it.
//
// ~120 lines of pure Rust — no `sha2` crate (NFR-DS-02). Vetted against
// `NIST FIPS 180-4` §6.2.2 + §Appendix B.2 test vectors below.
// ---------------------------------------------------------------------------

/// FIPS-180-4 §5.1.1 constants (round constants K[0..64]).
const K: [u32; 64] = [
    0x428a_2f98,
    0x7137_4491,
    0xb5c0_fbcf,
    0xe9b5_dba5,
    0x3956_c25b,
    0x59f1_11f1,
    0x923f_82a4,
    0xab1c_5ed5,
    0xd807_aa98,
    0x1283_5b01,
    0x2431_85be,
    0x550c_7dc3,
    0x72be_5d74,
    0x80de_b1fe,
    0x9bdc_06a7,
    0xc19b_f174,
    0xe49b_69c1,
    0xefbe_4786,
    0x0fc1_9dc6,
    0x240c_a1cc,
    0x2de9_2c6f,
    0x4a74_84aa,
    0x5cb0_a9dc,
    0x76f9_88da,
    0x983e_5152,
    0xa831_c66d,
    0xb003_27c8,
    0xbf59_7fc7,
    0xc6e0_0bf3,
    0xd5a7_9147,
    0x06ca_6351,
    0x1429_2967,
    0x27b7_0a85,
    0x2e1b_2138,
    0x4d2c_6dfc,
    0x5338_0d13,
    0x650a_7354,
    0x766a_0abb,
    0x81c2_c92e,
    0x9272_2c85,
    0xa2bf_e8a1,
    0xa81a_664b,
    0xc24b_8b70,
    0xc76c_51a3,
    0xd192_e819,
    0xd699_0624,
    0xf40e_3585,
    0x106a_a070,
    0x19a4_c116,
    0x1e37_6c08,
    0x2748_774c,
    0x34b0_bcb5,
    0x391c_0cb3,
    0x4ed8_aa4a,
    0x5b9c_ca4f,
    0x682e_6ff3,
    0x748f_82ee,
    0x78a5_636f,
    0x84c8_7814,
    0x8cc7_0208,
    0x90be_fffa,
    0xa450_6ceb,
    0xbef9_a3f7,
    0xc671_78f2,
];

/// FIPS-180-4 §5.3.3 initial hash values.
const H0: [u32; 8] = [
    0x6a09_e667,
    0xbb67_ae85,
    0x3c6e_f372,
    0xa54f_f53a,
    0x510e_527f,
    0x9b05_688c,
    0x1f83_d9ab,
    0x5be0_cd19,
];

/// SHA-256 digest of `data`, hex-encoded (lowercase, 64 chars). Returned as
/// a fixed-size array so no allocation is needed at hash-verification time.
#[must_use]
pub fn sha256_hex(data: &[u8]) -> [u8; 64] {
    let digest = sha256(data);
    let mut out = [0u8; 64];
    for (i, byte) in digest.iter().enumerate() {
        out[i * 2] = nibble_hex(byte >> 4);
        out[i * 2 + 1] = nibble_hex(byte & 0x0f);
    }
    out
}

/// SHA-256 raw digest of `data` (32 bytes).
#[must_use]
pub fn sha256(data: &[u8]) -> [u8; 32] {
    let mut h = H0;
    // Message padding: append 0x80, pad with zeros so `(len_bytes % 64) == 56`,
    // then append 64-bit big-endian bit length.
    let bit_len = (data.len() as u64) * 8;
    let mut buf: Vec<u8> = Vec::with_capacity(data.len() + 72);
    buf.extend_from_slice(data);
    buf.push(0x80);
    while buf.len() % 64 != 56 {
        buf.push(0);
    }
    buf.extend_from_slice(&bit_len.to_be_bytes());
    debug_assert_eq!(buf.len() % 64, 0);

    // Process each 64-byte block.
    let mut chunk = buf.chunks_exact(64);
    for block in &mut chunk {
        let mut w = [0u32; 64];
        for (i, word) in block.chunks_exact(4).take(16).enumerate() {
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
        let mut a = h[0];
        let mut b = h[1];
        let mut c = h[2];
        let mut d = h[3];
        let mut e = h[4];
        let mut f = h[5];
        let mut g = h[6];
        let mut hh = h[7];
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
    debug_assert!(chunk.remainder().is_empty());

    let mut digest = [0u8; 32];
    for (i, word) in h.iter().enumerate() {
        digest[i * 4..i * 4 + 4].copy_from_slice(&word.to_be_bytes());
    }
    digest
}

const fn nibble_hex(v: u8) -> u8 {
    match v {
        0..=9 => b'0' + v,
        10..=15 => b'a' + (v - 10),
        // Only 4-bit nibbles reach this function.
        _ => b'?',
    }
}

/// Verifies every manifest entry with a pinned SHA-256 matches the blob
/// [`load_spv_owned`] returns. Entries with `expected_sha256_hex = None` are
/// treated as "not yet pinned" (foundation slice — no assertion).
///
/// Used from the crate test suite; not called at runtime (fast build).
///
/// # Errors
///
/// Returns the failing shader `name` on the first mismatch (empty tuple ok).
pub fn verify_pinned_hashes() -> Result<(), &'static str> {
    for shader in SHADERS {
        let Some(expected) = shader.expected_sha256_hex else {
            continue;
        };
        let Some(blob) = load_spv_owned(shader.name) else {
            // A pinned hash without a blob is a bug — either the include_bytes!
            // was removed or the pin was added prematurely.
            return Err(shader.name);
        };
        let got = sha256_hex(&blob);
        // `expected` is a `&'static str`; compare it against `got` as bytes.
        if expected.as_bytes() != got {
            return Err(shader.name);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shader_manifest_is_stable() {
        // Foundation-slice pin: 13 shader entries = 12 glslc-produced kernels
        // (11 op categories; GEMM has two variants) + 1 hand-crafted smoke
        // kernel (`copy_f32`, ADR M3-02-spirv-generation §2 Option D).
        assert_eq!(SHADERS.len(), 13);
        // Names are unique.
        let mut names: Vec<&str> = SHADERS.iter().map(|s| s.name).collect();
        names.sort_unstable();
        let dedup_len = {
            let mut names_dedup = names.clone();
            names_dedup.dedup();
            names_dedup.len()
        };
        assert_eq!(
            dedup_len,
            names.len(),
            "duplicate SPIR-V shader name in SHADERS"
        );
        // Exactly one entry is `Handcrafted` (ADR §5: no expansion permitted).
        let handcrafted_count = SHADERS
            .iter()
            .filter(|s| matches!(s.variant, ShaderVariant::Handcrafted))
            .count();
        assert_eq!(
            handcrafted_count, 1,
            "ADR M3-02-spirv-generation §5 forbids adding more Handcrafted entries — only \
             `copy_f32` may be hand-authored. Found {handcrafted_count}."
        );
    }

    #[test]
    fn every_glsl_shader_has_matching_glsl_source() {
        // Every glslc-produced entry in the manifest must correspond to a
        // `.comp` source in `kernels/glsl/` — an entry with no source is a bug
        // (either the manifest is stale or the source was deleted).
        // Handcrafted entries live in `kernels/handcrafted/*.spv.rs`, so we
        // skip them here.
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        for shader in SHADERS {
            if matches!(shader.variant, ShaderVariant::Handcrafted) {
                continue;
            }
            let path = format!("{manifest_dir}/kernels/glsl/{}.comp", shader.name);
            assert!(
                std::path::Path::new(&path).is_file(),
                "SPIR-V manifest entry `{}` has no GLSL source at {}",
                shader.name,
                path,
            );
        }
    }

    #[test]
    fn load_spv_is_honest_for_glslc_produced_entries() {
        // The foundation slice ships no glslc-produced `.spv`; only the
        // handcrafted `copy_f32` blob is available. `load_spv` returns
        // borrowed &'static bytes and therefore intentionally cannot serve
        // the handcrafted `Vec<u8>` — it must be `None` for every entry
        // today. When T14〜T22 lands a `.spv`, that shader's `None` becomes
        // `Some`, and this test's count shrinks to only the still-unshipped
        // entries.
        for shader in SHADERS {
            assert!(
                load_spv(shader.name).is_none(),
                "shader `{}` unexpectedly has a compiled .spv via load_spv (foundation slice: all \
                 None); if you have just landed T14+, update this test",
                shader.name,
            );
        }
        // Unknown shader names always miss.
        assert!(load_spv("no_such_shader").is_none());
    }

    #[test]
    fn load_spv_owned_reaches_the_handcrafted_copy_f32() {
        // `copy_f32` is the only entry available in the foundation slice —
        // via `load_spv_owned`, not `load_spv` (see the borrow/owned split in
        // the module docs).
        let blob = load_spv_owned("copy_f32").expect("copy_f32 handcrafted blob must load");
        // SPIR-V spec §2.3: module length is always a multiple of 4.
        assert_eq!(blob.len() % 4, 0);
        // First 4 bytes decode as SPIR-V magic (little-endian 0x07230203).
        let magic = u32::from_le_bytes([blob[0], blob[1], blob[2], blob[3]]);
        assert_eq!(magic, 0x0723_0203, "copy_f32 SPIR-V magic mismatch");
        // Unknown names remain honest.
        assert!(load_spv_owned("no_such_shader").is_none());
    }

    #[test]
    fn verify_pinned_hashes_is_ok_for_foundation_slice() {
        // The only pinned entry is `copy_f32` — the hand-crafted blob whose
        // SHA-256 the manifest carries. `verify_pinned_hashes` must confirm
        // the runtime bytes hash to the same value.
        assert!(
            verify_pinned_hashes().is_ok(),
            "copy_f32 SHA-256 pin does not match the bytes returned by \
             `handcrafted_copy_f32::bytes()`; regenerate the pin"
        );
    }

    // FIPS-180-4 §Appendix B.1 / B.2 test vectors + one additional NIST vector
    // — pin the built-in SHA-256 against a public spec so it stays correct.
    #[test]
    fn sha256_empty_input_matches_spec() {
        // "" → e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
        let out = sha256_hex(b"");
        assert_eq!(
            core::str::from_utf8(&out).unwrap(),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        );
    }

    #[test]
    fn sha256_abc_matches_spec() {
        // "abc" → ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad
        // (FIPS-180-4 §B.1)
        let out = sha256_hex(b"abc");
        assert_eq!(
            core::str::from_utf8(&out).unwrap(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        );
    }

    #[test]
    fn sha256_two_block_message_matches_spec() {
        // FIPS-180-4 §B.2 (two-block message, exercises padding).
        // "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"
        // → 248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1
        let out = sha256_hex(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq");
        assert_eq!(
            core::str::from_utf8(&out).unwrap(),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1",
        );
    }
}
