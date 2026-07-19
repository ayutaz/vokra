//! WGSL compute-kernel manifest + SHA-256 source pins (M4-01-T11).
//!
//! The WebGPU analogue of `vokra-backend-vulkan/src/spirv.rs`, with one
//! structural difference: **WGSL is text and text is the shipped artifact**.
//! The Web standard has no binary shader format (there is no SPIR-V
//! precompile path in WebGPU — W3C WebGPU takes WGSL source in
//! `createShaderModule`), so the manifest embeds each kernel's source with
//! `include_str!` and the drift gate is a **source-text SHA-256 pin**
//! (`verify_pinned_hashes`, run natively by the crate tests) rather than a
//! recompile-and-diff.
//!
//! # NFR-RL-05 (no CPU-side JIT) — the T01-(f) ADR record
//!
//! Embedding WGSL text and handing it to `createShaderModule` is NOT
//! host-side JIT: the WGSL → GPU ISA compilation happens in the browser /
//! GPU driver's address space, which is the driver's responsibility — the
//! same separation M4-13 records for the driver-side SPIR-V → GPU ISA
//! translation. Vokra never *generates* shader code at runtime; the strings
//! below are frozen at build time and hash-pinned (ADR M4-01-webgpu-wasm
//! §7).
//!
//! # Every entry is pinned (stricter than the Vulkan manifest)
//!
//! The Vulkan manifest allows `expected_sha256_hex = None` while the owner
//! has not yet compiled a `.spv`; here the source IS the artifact and is
//! always embedded, so an unpinned entry would just be a hole in the drift
//! gate. `verify_pinned_hashes` therefore fails on any missing pin — no
//! fabricated pass, nothing to defer.

/// One WGSL compute kernel: name, entry point, embedded source, pinned hash.
#[derive(Debug, Clone, Copy)]
pub struct WgslShader {
    /// Manifest key (also the JS-glue pipeline-cache key).
    pub name: &'static str,
    /// The `@compute` entry function name inside `source`.
    pub entry_point: &'static str,
    /// Number of storage buffers the kernel binds (bindings `0..n`); the
    /// uniform params buffer, when present, binds at index `n` — the
    /// glue-side bind contract (`plan.rs` packs the uniform bytes).
    pub n_storage_buffers: u32,
    /// The WGSL source text (embedded at build time — NFR-RL-05).
    pub source: &'static str,
    /// Pinned SHA-256 of `source` (lowercase hex). Every entry MUST be
    /// pinned — see the module docs.
    pub expected_sha256_hex: &'static str,
}

/// The kernel manifest (M4-01-T11〜T15). Names are stable API for the JS
/// glue's pipeline cache and the graph arm's backing-shader map
/// (`crate::backend::graph_op_backing_shader`).
pub const SHADERS: &[WgslShader] = &[
    WgslShader {
        name: "copy_f32",
        entry_point: "main",
        n_storage_buffers: 2,
        source: include_str!("../kernels/wgsl/copy_f32.wgsl"),
        expected_sha256_hex: "fb1ec08a9721874fb8035ca38e70b71d649882fb4a259082ae2bb08e42941425",
    },
    WgslShader {
        name: "add_f32",
        entry_point: "main",
        n_storage_buffers: 3,
        source: include_str!("../kernels/wgsl/add_f32.wgsl"),
        expected_sha256_hex: "da54e332c4cd2a5f7552f93c035f1be30e0ab45382667d55d53ec3e4bac1958c",
    },
    WgslShader {
        name: "elementwise",
        entry_point: "main",
        n_storage_buffers: 3,
        source: include_str!("../kernels/wgsl/elementwise.wgsl"),
        expected_sha256_hex: "27be229e6b05d13eab81931e7d180a542f77b0f69c6cf585168e30bf4e30785b",
    },
    WgslShader {
        name: "gemm_f32",
        entry_point: "main",
        n_storage_buffers: 4,
        source: include_str!("../kernels/wgsl/gemm_f32.wgsl"),
        expected_sha256_hex: "c3c2d5e4759edf5ac607cd4ef81311865b267fc5bbe4fef3401e77f80401d503",
    },
    WgslShader {
        name: "gemv_f32",
        entry_point: "main",
        n_storage_buffers: 4,
        source: include_str!("../kernels/wgsl/gemv_f32.wgsl"),
        expected_sha256_hex: "3c734b044f3b5ae7455ec7b5628b90dacdb1963ef1ef366bfa778f8642739353",
    },
    WgslShader {
        name: "softmax",
        entry_point: "main",
        n_storage_buffers: 2,
        source: include_str!("../kernels/wgsl/softmax.wgsl"),
        expected_sha256_hex: "92003883dcb524c19f4b15369fc7345012ba88b7806078a1b81ec36de8e95cd5",
    },
    WgslShader {
        name: "softmax_causal",
        entry_point: "main",
        n_storage_buffers: 2,
        source: include_str!("../kernels/wgsl/softmax_causal.wgsl"),
        expected_sha256_hex: "e24ff343060004ad9468ab5533dcafe3342fdb54213b12faa96efb2b34fd3c99",
    },
    WgslShader {
        name: "layer_norm",
        entry_point: "main",
        n_storage_buffers: 4,
        source: include_str!("../kernels/wgsl/layer_norm.wgsl"),
        expected_sha256_hex: "92351173f72243221efda208726917e17a1c49592b098f0fc6c59b6cdd318dc3",
    },
    WgslShader {
        name: "gelu",
        entry_point: "main",
        n_storage_buffers: 2,
        source: include_str!("../kernels/wgsl/gelu.wgsl"),
        expected_sha256_hex: "9e9334763818c6184dedc835d88dede190cba224beaca904ce2e4e23e7785190",
    },
    WgslShader {
        name: "conv1d",
        entry_point: "main",
        n_storage_buffers: 4,
        source: include_str!("../kernels/wgsl/conv1d.wgsl"),
        expected_sha256_hex: "8559efaf13a706921d0e23a68bb7f2fa6dada01374dcfcd6b52f7f79c80c919c",
    },
    WgslShader {
        name: "activation",
        entry_point: "main",
        n_storage_buffers: 2,
        source: include_str!("../kernels/wgsl/activation.wgsl"),
        expected_sha256_hex: "5655a3e64ae1bcf7efd4b19fe6ef3855684f7339fbb4f5afc24d8500a26f5a4d",
    },
];

/// Looks up a manifest entry by name.
#[must_use]
pub fn get(name: &str) -> Option<&'static WgslShader> {
    SHADERS.iter().find(|s| s.name == name)
}

/// Whether `name` is a manifest kernel. Unlike the Vulkan `has_blob` this is
/// not blob-gated — WGSL sources are always embedded — so coverage claims
/// derived from it (`WebGpuBackend::supports`) are static.
#[must_use]
pub fn has_shader(name: &str) -> bool {
    get(name).is_some()
}

/// Verifies every manifest entry's embedded source matches its pinned
/// SHA-256 (the drift gate — M4-01-T11). Returns the first failing shader
/// name. Run by the crate tests on every target; not called at runtime.
///
/// # Errors
///
/// The failing shader's `name`, on the first hash mismatch or missing pin.
pub fn verify_pinned_hashes() -> Result<(), &'static str> {
    for shader in SHADERS {
        let digest = sha256_hex(shader.source.as_bytes());
        let digest_str = core::str::from_utf8(&digest).expect("hex digest is ASCII");
        if digest_str != shader.expected_sha256_hex {
            return Err(shader.name);
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Zero-dependency SHA-256 (FIPS-180-4 §6.2) — the same one-file
// implementation the Vulkan backend vets against the NIST Appendix B test
// vectors (crates/vokra-backend-vulkan/src/spirv.rs); duplicated rather than
// imported so the WebGPU crate does not grow a cross-backend dependency for
// ~120 lines of pure Rust (NFR-DS-02).
// ---------------------------------------------------------------------------

/// FIPS-180-4 §5.1.1 round constants K[0..64].
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

/// SHA-256 digest of `data`, hex-encoded (lowercase, 64 chars).
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
    let bit_len = (data.len() as u64) * 8;
    let mut buf: Vec<u8> = Vec::with_capacity(data.len() + 72);
    buf.extend_from_slice(data);
    buf.push(0x80);
    while buf.len() % 64 != 56 {
        buf.push(0);
    }
    buf.extend_from_slice(&bit_len.to_be_bytes());
    debug_assert_eq!(buf.len() % 64, 0);

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

#[cfg(test)]
mod tests {
    use super::*;

    /// NIST FIPS 180-4 Appendix B.1 test vector: SHA-256("abc").
    #[test]
    fn sha256_matches_nist_appendix_b1() {
        let hex = sha256_hex(b"abc");
        assert_eq!(
            core::str::from_utf8(&hex).unwrap(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    /// NIST FIPS 180-4 Appendix B (empty message).
    #[test]
    fn sha256_matches_nist_empty_message() {
        let hex = sha256_hex(b"");
        assert_eq!(
            core::str::from_utf8(&hex).unwrap(),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    /// The T11 drift gate: every embedded WGSL source must match its pin.
    /// On failure the message prints the offender's actual hash so the pin
    /// can be updated deliberately (drift is a reviewed change, never
    /// silent).
    #[test]
    fn wgsl_sources_match_pinned_hashes() {
        if let Err(name) = verify_pinned_hashes() {
            let shader = get(name).expect("failing name is a manifest entry");
            let actual = sha256_hex(shader.source.as_bytes());
            panic!(
                "WGSL source drift for `{name}`: expected pin {}, actual {}. If the source \
                 change is intentional, update expected_sha256_hex in src/wgsl.rs (drift is a \
                 reviewed change).",
                shader.expected_sha256_hex,
                core::str::from_utf8(&actual).unwrap()
            );
        }
    }

    /// Structural sanity for every kernel: a @compute entry with the declared
    /// entry-point name, a workgroup_size, balanced braces/parens, and the
    /// FP32-only rule (no f16 in any source — NFR-QL-01 / CLAUDE.md).
    #[test]
    fn wgsl_sources_are_structurally_sane() {
        for shader in SHADERS {
            let src = shader.source;
            assert!(
                src.contains("@compute"),
                "{}: missing @compute stage attribute",
                shader.name
            );
            assert!(
                src.contains("@workgroup_size("),
                "{}: missing workgroup_size",
                shader.name
            );
            assert!(
                src.contains(&format!("fn {}(", shader.entry_point)),
                "{}: entry point `{}` not found",
                shader.name,
                shader.entry_point
            );
            let opens = src.matches('{').count();
            let closes = src.matches('}').count();
            assert_eq!(opens, closes, "{}: unbalanced braces", shader.name);
            let po = src.matches('(').count();
            let pc = src.matches(')').count();
            assert_eq!(po, pc, "{}: unbalanced parens", shader.name);
            // FP32-only rule: no f16 in CODE (comment lines stripped — the
            // rule is allowed to be *documented* in a comment).
            let code_only: String = src
                .lines()
                .filter(|l| !l.trim_start().starts_with("//"))
                .collect::<Vec<_>>()
                .join("\n");
            assert!(
                !code_only.contains("f16"),
                "{}: f16 found in code — FP32 storage/accumulator is fixed (NFR-QL-01)",
                shader.name
            );
            // The uniform params buffer (when the struct exists) must bind at
            // index n_storage_buffers per the glue bind contract.
            if src.contains("var<uniform>") {
                let expected = format!("@binding({}) var<uniform>", shader.n_storage_buffers);
                assert!(
                    src.contains(&expected),
                    "{}: uniform must bind at index {} (glue bind contract)",
                    shader.name,
                    shader.n_storage_buffers
                );
            }
            // Every storage binding index 0..n must be declared.
            for i in 0..shader.n_storage_buffers {
                let b = format!("@binding({i}) var<storage");
                assert!(
                    src.contains(&b),
                    "{}: storage binding {} not declared",
                    shader.name,
                    i
                );
            }
        }
    }

    /// Manifest keys are unique and get()/has_shader() agree.
    #[test]
    fn manifest_names_are_unique_and_lookup_agrees() {
        for (i, s) in SHADERS.iter().enumerate() {
            assert!(
                !SHADERS[..i].iter().any(|p| p.name == s.name),
                "duplicate manifest name {}",
                s.name
            );
            assert!(has_shader(s.name));
            assert_eq!(get(s.name).unwrap().name, s.name);
        }
        assert!(!has_shader("no_such_kernel"));
        assert!(get("no_such_kernel").is_none());
    }
}
