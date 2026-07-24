//! `tts-dac` family flip-the-switch real-checkpoint parity CI
//! (SoTA plan Phase 1-4/1-5, 2026-07-24 → 2026-07-25).
//!
//! # Family
//!
//! Two TTS models whose terminal codec is Descript **DAC** (MIT):
//!
//! | arch  | HF repo                         | license    | codec         | native module                  |
//! | ----- | ------------------------------- | ---------- | ------------- | ------------------------------ |
//! | dia   | `nari-labs/Dia-1.6B`            | Apache-2.0 | DAC 44.1 kHz  | [`vokra_models::dia`]          |
//! | zonos | `Zyphra/Zonos-v0.1-transformer` | Apache-2.0 | DAC 44.1 kHz  | [`vokra_models::zonos`]        |
//!
//! Both native modules are **scaffold-only** at this branch: [`DiaTts`] /
//! [`ZonosTts`] validate config + weight shapes and today return
//! [`VokraError::NotImplemented`] from `synthesize` when built off
//! [`DiaWeights::synthesized`] / [`ZonosWeights::synthesized`] (FR-EX-08 —
//! no fabricated audio from synthesized weights). The real forward + `from_gguf`
//! path is a follow-up wave gated on the upstream tensor-name manifest fetch
//! (the CSM T29 / CosyVoice2 T02 precedent).
//!
//! Because there is no real forward yet, this harness tests **what IS
//! available** on a real converted GGUF:
//!
//!   * **Loading** — [`GgufFile::open`] must succeed on a real
//!     `vokra-cli convert --model <arch>` output.
//!   * **Config extraction** — every `vokra.<arch>.*` metadata chunk value
//!     must match the primary-source [`DiaConfig::dia_1_6b`] /
//!     [`ZonosConfig::zonos_v0_1_transformer`] constants transcribed from
//!     the upstream HuggingFace `config.json`. A drift in the converter or
//!     in the transcribed constants fails loudly.
//!   * **Weight-shape validation** — every tensor advertised by the GGUF
//!     header is inspected (dtype, ndim, non-zero element count, finite
//!     payload for FP32 tensors). Failure names the first offending tensor.
//!
//! # Gating (FR-EX-08 — fabricated pass 禁止)
//!
//!   * `VOKRA_<ARCH>_GGUF` unset → **clean skip** with a printed reason
//!     naming the env var and the primary-source `vokra-cli convert` recipe.
//!     No numbers claim to have been produced.
//!   * `VOKRA_<ARCH>_GGUF` set → **the tests actually fire**. Loading,
//!     metadata, and shape gates run on the file the env var points at.
//!     Every mismatch is a hard fail (never `println!("skipped")`).
//!   * `VOKRA_<ARCH>_REFDIR` **also** set → **flip-the-switch step**. The
//!     leg intentionally fails loudly today, naming the two owner steps that
//!     unblock a real stage-tap comparison: (1) landing `<Arch>Tts::from_gguf`
//!     against the upstream tensor-name manifest, (2) wiring a stage tap
//!     accessor for the reference tap the dumper writes. Refuses to report
//!     a pass that did not run (the CSM `staged_reference_parity_is_env_gated`
//!     precedent).
//!
//! # Judgement
//!
//! - **Config chunks**: exact-equality (integers) / `PartialEq` (bool / float).
//!   A single mismatch is a hard fail (FR-EX-08 — no rounding, no defaulting).
//! - **Weight tensors**: shape well-formed + FP32 payloads finite. The
//!   `vokra-cli convert` path is expected to passthrough F32/F16 bytes; a
//!   converter emitting NaN/Inf for a real safetensors input is a bug.
//! - **Stage-tap parity (VOKRA_<ARCH>_REFDIR)**: reserved for the real
//!   forward; today the leg refuses to auto-run.

#![allow(clippy::items_after_statements)]

use std::env;
use std::path::{Path, PathBuf};

use vokra_core::gguf::chunks::KEY_MODEL_ARCH;
use vokra_core::gguf::{GgmlType, GgufBuilder, GgufFile};

use vokra_models::dia::{self, DiaConfig};
use vokra_models::zonos::{self, ZonosConditionerKind, ZonosConfig};

// -----------------------------------------------------------------------------
// Shared helpers
// -----------------------------------------------------------------------------

/// The two env vars this harness reads, per architecture.
///
///   * `.0` — `VOKRA_<ARCH>_GGUF`, the pre-converted Vokra GGUF path (owner
///     runs `vokra-cli convert --model <arch> --input <upstream.safetensors>`
///     against a pinned HF snapshot; the workflow ships that command).
///   * `.1` — `VOKRA_<ARCH>_REFDIR`, an optional directory of upstream
///     stage-tap reference dumps (`.f32` / `.i64` little-endian, one file per
///     tensor). Present = "flip-the-switch fires" — the harness tries to
///     compare against the dumps and hard-fails when the real forward is
///     not wired yet (fabricated pass 禁止).
///
/// A returned `Some` end-to-end guarantees the referenced path exists; a
/// missing file is treated exactly like the env var being unset so a stale
/// pointer never turns into a fabricated pass.
fn env_paths_for(arch: &str) -> (Option<PathBuf>, Option<PathBuf>) {
    let arch_upper = arch.to_ascii_uppercase();
    let gguf_key = format!("VOKRA_{arch_upper}_GGUF");
    let refdir_key = format!("VOKRA_{arch_upper}_REFDIR");

    let gguf = env::var_os(&gguf_key)
        .map(PathBuf::from)
        .filter(|p| p.is_file());
    let refdir = env::var_os(&refdir_key)
        .map(PathBuf::from)
        .filter(|p| p.is_dir());
    (gguf, refdir)
}

/// A human-readable skip annotation for the "env var not set" path.
///
/// The message names both env vars and the exact `vokra-cli convert`
/// invocation the owner runs so the CI log line is self-contained (a common
/// audit finding — "why is this skipping" needs no cross-reference).
fn skip_reason(arch: &str, hf_repo: &str, revision: &str, license: &str) -> String {
    let arch_upper = arch.to_ascii_uppercase();
    format!(
        "SKIP [parity_tts_dac::{arch}]: env var VOKRA_{arch_upper}_GGUF is unset (or points \
         at a missing file). Set it to a Vokra GGUF produced by \
         `vokra-cli convert --model {arch} --input <upstream_safetensors> --output \
         <path>.gguf` against `{hf_repo}` @ revision `{revision}` ({license}). \
         VOKRA_{arch_upper}_REFDIR (optional) points at upstream stage-tap dumps for \
         the flip-the-switch step. This is a clean gated skip — no parity numbers \
         were produced (fabricated pass 禁止, FR-EX-08)."
    )
}

/// Hard-fail message when the flip-the-switch leg fires but the real forward
/// isn't wired yet. Names the two concrete owner steps that unblock it so a
/// future reader has an obvious "what next".
fn flip_the_switch_deferred(arch: &str, refdir: &Path) -> String {
    format!(
        "VOKRA_{arch_upper}_REFDIR = {refdir} is set and the GGUF loaded, but the \
         `{arch}` native module is a scaffold: `{ty}Tts::synthesize` returns \
         NotImplemented until the real forward + `from_gguf` weight bind lands \
         (the CSM T29 / CosyVoice2 T02 precedent). Owner steps that unblock this \
         leg: (1) implement `{ty}Tts::from_gguf` and cross-check the tensor names \
         against the pinned upstream safetensors header, (2) add stage-tap \
         accessors matching what the dumper writes into VOKRA_{arch_upper}_REFDIR, \
         then replace this panic with the staged comparison (FP32 atol = 0.01, \
         NFR-QL-01). Refusing to report a pass that did not run (FR-EX-08 — \
         fabricated pass 禁止).",
        arch_upper = arch.to_ascii_uppercase(),
        refdir = refdir.display(),
        arch = arch,
        ty = capitalize_first(arch),
    )
}

fn capitalize_first(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        None => String::new(),
        Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
    }
}

/// Common GGUF-side gates: header well-formed, arch matches, every tensor
/// shape non-degenerate, every FP32 payload finite. Applied to whichever
/// GGUF the per-arch tests load.
///
/// A converter regression (writing arch=`whisper` into a Dia GGUF, or
/// emitting a shape=[0, 0] tensor, or leaking NaN into a passthrough F32) is
/// caught here uniformly rather than being replicated per arch.
fn assert_common_gguf_invariants(gguf: &GgufFile, expected_arch: &str) {
    // (1) arch metadata matches the runtime's EXPECTED_ARCH.
    let arch = gguf
        .get(KEY_MODEL_ARCH)
        .and_then(|v| v.as_str())
        .unwrap_or_else(|| panic!("GGUF missing `{KEY_MODEL_ARCH}` metadata"));
    assert_eq!(
        arch, expected_arch,
        "GGUF `{KEY_MODEL_ARCH}` = {arch:?}, expected {expected_arch:?} \
         (converter or Rust EXPECTED_ARCH drift)"
    );

    // (2) at least one tensor advertised — a zero-tensor GGUF is a truncated
    // convert (write_tensors ran on an empty iterator).
    let tensors = gguf.tensors();
    assert!(
        !tensors.is_empty(),
        "GGUF for {expected_arch} advertises zero tensors — converter truncated?"
    );

    // (3) per-tensor shape + FP32 payload sanity. F16 / K-quant tensors are
    // skipped for the payload check (this harness has no local dequantizer);
    // shape checks apply to every tensor.
    let mut checked_f32 = 0usize;
    for info in tensors {
        assert!(
            !info.name.is_empty(),
            "GGUF tensor has empty name (converter bug)"
        );
        assert!(
            !info.dimensions.is_empty(),
            "tensor {:?}: dimensions has rank 0",
            info.name
        );
        assert!(
            info.dimensions.iter().all(|d| *d > 0),
            "tensor {:?}: dimensions {:?} contain a zero-sized axis",
            info.name,
            info.dimensions,
        );
        let elems = info
            .element_count()
            .unwrap_or_else(|e| panic!("tensor {:?}: element_count overflow: {e}", info.name));
        assert!(
            elems > 0,
            "tensor {:?}: element product 0 (dimensions {:?})",
            info.name,
            info.dimensions,
        );

        if info.dtype == GgmlType::F32 {
            let payload = gguf
                .tensor_f32(&info.name)
                .unwrap_or_else(|e| panic!("tensor {:?}: tensor_f32 failed: {e}", info.name));
            assert_eq!(
                payload.len() as u64,
                elems,
                "tensor {:?}: f32 payload len {} != element_count {}",
                info.name,
                payload.len(),
                elems,
            );
            let non_finite = payload.iter().position(|v| !v.is_finite());
            assert!(
                non_finite.is_none(),
                "tensor {:?}: non-finite value at index {} (F32 passthrough should \
                 preserve safetensors bytes)",
                info.name,
                non_finite.unwrap(),
            );
            checked_f32 += 1;
        }
    }

    eprintln!(
        "[parity_tts_dac::{expected_arch}] GGUF invariants OK: {} tensors advertised, \
         {} F32 payloads validated finite",
        tensors.len(),
        checked_f32,
    );
}

/// Metadata reader: unsigned integer keys widen to `u64`, so the caller
/// asks for the concrete width they expect. `panic!` naming the key on the
/// first miss (a metadata drift is a bug we want called out, not silently
/// downcast to zero).
fn read_u32(gguf: &GgufFile, key: &str) -> u32 {
    let v = gguf
        .get(key)
        .unwrap_or_else(|| panic!("GGUF missing metadata key {key:?}"));
    let widened = v
        .as_u64()
        .unwrap_or_else(|| panic!("GGUF metadata {key:?} is not an unsigned integer"));
    u32::try_from(widened)
        .unwrap_or_else(|_| panic!("GGUF metadata {key:?} = {widened} does not fit in u32"))
}

fn read_usize(gguf: &GgufFile, key: &str) -> usize {
    read_u32(gguf, key) as usize
}

fn read_bool(gguf: &GgufFile, key: &str) -> bool {
    let v = gguf
        .get(key)
        .unwrap_or_else(|| panic!("GGUF missing metadata key {key:?}"));
    v.as_bool()
        .unwrap_or_else(|| panic!("GGUF metadata {key:?} is not a bool"))
}

fn read_f32(gguf: &GgufFile, key: &str) -> f32 {
    let v = gguf
        .get(key)
        .unwrap_or_else(|| panic!("GGUF missing metadata key {key:?}"));
    let widened = v
        .as_f64()
        .unwrap_or_else(|| panic!("GGUF metadata {key:?} is not a float"));
    widened as f32
}

// -----------------------------------------------------------------------------
// dia
// -----------------------------------------------------------------------------

/// nari-labs/Dia-1.6B parity gate — Apache 2.0, DAC 44.1 kHz codec.
///
/// # What runs
///
///   * `VOKRA_DIA_GGUF` unset → clean skip with the recipe reproduced in
///     the log.
///   * set → open the file, apply [`assert_common_gguf_invariants`], and
///     cross-check every `vokra.dia.*` chunk value against the primary
///     source ([`DiaConfig::dia_1_6b`], transcribed from
///     `huggingface.co/nari-labs/Dia-1.6B/config.json`).
///   * additionally `VOKRA_DIA_REFDIR` set → refuse to report a pass:
///     stage-tap comparison is deferred until the scaffold gains a real
///     forward + `from_gguf` (fabricated pass 禁止).
#[test]
fn parity_tts_dac_dia() {
    let (Some(gguf_path), refdir) = env_paths_for("dia") else {
        eprintln!(
            "{}",
            skip_reason(
                "dia",
                "nari-labs/Dia-1.6B",
                // Snapshot SHA fetched 2026-07-25 via huggingface.co/api/models/nari-labs/Dia-1.6B
                "257bc72f9b78182ccc6fa07675a9ae4c1a44e2cd",
                "Apache-2.0",
            )
        );
        return;
    };

    let gguf = GgufFile::open(&gguf_path)
        .unwrap_or_else(|e| panic!("open VOKRA_DIA_GGUF = {gguf_path:?}: {e}"));
    assert_common_gguf_invariants(&gguf, dia::EXPECTED_ARCH);

    // Cross-check `vokra.dia.*` metadata against the primary-source config.
    // Values below are the Rust-side transcription of the upstream
    // `config.json` fields (`DiaConfig::dia_1_6b`); a drift here means
    // either the converter dropped a chunk or the Rust constants moved.
    let cfg = DiaConfig::dia_1_6b();

    assert_eq!(
        read_u32(&gguf, "vokra.dia.sample_rate"),
        cfg.sample_rate,
        "vokra.dia.sample_rate must equal DIA_SAMPLE_RATE ({})",
        cfg.sample_rate,
    );

    // Encoder hparams.
    assert_eq!(
        read_usize(&gguf, "vokra.dia.arch.encoder.n_layer"),
        cfg.encoder.n_layer,
    );
    assert_eq!(
        read_usize(&gguf, "vokra.dia.arch.encoder.n_embd"),
        cfg.encoder.n_embd,
    );
    assert_eq!(
        read_usize(&gguf, "vokra.dia.arch.encoder.n_head"),
        cfg.encoder.n_head,
    );
    assert_eq!(
        read_usize(&gguf, "vokra.dia.arch.encoder.head_dim"),
        cfg.encoder.head_dim,
    );
    assert_eq!(
        read_usize(&gguf, "vokra.dia.arch.encoder.n_hidden"),
        cfg.encoder.n_hidden,
    );

    // Decoder hparams — GQA + cross-attn.
    assert_eq!(
        read_usize(&gguf, "vokra.dia.arch.decoder.n_layer"),
        cfg.decoder.n_layer,
    );
    assert_eq!(
        read_usize(&gguf, "vokra.dia.arch.decoder.n_embd"),
        cfg.decoder.n_embd,
    );
    assert_eq!(
        read_usize(&gguf, "vokra.dia.arch.decoder.gqa_query_heads"),
        cfg.decoder.gqa_query_heads,
    );
    assert_eq!(
        read_usize(&gguf, "vokra.dia.arch.decoder.kv_heads"),
        cfg.decoder.kv_heads,
    );
    assert_eq!(
        read_usize(&gguf, "vokra.dia.arch.decoder.gqa_head_dim"),
        cfg.decoder.gqa_head_dim,
    );
    assert_eq!(
        read_usize(&gguf, "vokra.dia.arch.decoder.cross_query_heads"),
        cfg.decoder.cross_query_heads,
    );
    assert_eq!(
        read_usize(&gguf, "vokra.dia.arch.decoder.cross_head_dim"),
        cfg.decoder.cross_head_dim,
    );
    assert_eq!(
        read_usize(&gguf, "vokra.dia.arch.decoder.n_hidden"),
        cfg.decoder.n_hidden,
    );

    // Vocab / data.
    assert_eq!(
        read_usize(&gguf, "vokra.dia.src_vocab_size"),
        cfg.src_vocab_size,
    );
    assert_eq!(
        read_usize(&gguf, "vokra.dia.tgt_vocab_size"),
        cfg.tgt_vocab_size,
    );
    assert_eq!(read_usize(&gguf, "vokra.dia.channels"), cfg.channels);
    assert_eq!(read_usize(&gguf, "vokra.dia.text_length"), cfg.text_length);
    assert_eq!(
        read_usize(&gguf, "vokra.dia.audio_length"),
        cfg.audio_length
    );
    assert_eq!(
        read_u32(&gguf, "vokra.dia.text_pad_value"),
        cfg.text_pad_value,
    );
    assert_eq!(
        read_u32(&gguf, "vokra.dia.audio_bos_value"),
        cfg.audio_bos_value,
    );
    assert_eq!(
        read_u32(&gguf, "vokra.dia.audio_eos_value"),
        cfg.audio_eos_value,
    );
    assert_eq!(
        read_u32(&gguf, "vokra.dia.audio_pad_value"),
        cfg.audio_pad_value,
    );

    // Delay pattern — count + every element.
    let delay_count = read_usize(&gguf, "vokra.dia.delay_pattern_count");
    assert_eq!(
        delay_count,
        cfg.delay_pattern.len(),
        "vokra.dia.delay_pattern_count = {delay_count}, expected {} \
         (channels / delay_pattern must stay in lockstep)",
        cfg.delay_pattern.len(),
    );
    for (i, expected) in cfg.delay_pattern.iter().enumerate() {
        let key = format!("vokra.dia.delay_pattern.{i}");
        assert_eq!(
            read_usize(&gguf, &key),
            *expected,
            "vokra.dia.delay_pattern.{i} drift",
        );
    }

    // Norm / RoPE scalars.
    let norm_eps = read_f32(&gguf, "vokra.dia.norm_eps");
    assert!(
        (norm_eps - cfg.norm_eps).abs() <= f32::EPSILON * 8.0,
        "vokra.dia.norm_eps = {norm_eps}, expected {}",
        cfg.norm_eps,
    );
    let rope_max = read_f32(&gguf, "vokra.dia.rope_max_timescale");
    assert!(
        (rope_max - cfg.rope_max_timescale).abs() <= 1.0,
        "vokra.dia.rope_max_timescale = {rope_max}, expected {}",
        cfg.rope_max_timescale,
    );

    eprintln!(
        "[parity_tts_dac::dia] {} tensors, config extracted OK against DiaConfig::dia_1_6b()",
        gguf.tensors().len(),
    );

    // Flip-the-switch gate: real forward not wired yet → refuse to report a pass
    // that did not run. See `flip_the_switch_deferred` for the two owner steps.
    if let Some(refdir) = refdir {
        panic!("{}", flip_the_switch_deferred("dia", &refdir));
    }
}

// -----------------------------------------------------------------------------
// zonos
// -----------------------------------------------------------------------------

/// Zyphra/Zonos-v0.1-transformer parity gate — Apache 2.0, DAC 44.1 kHz codec.
///
/// # What runs
///
///   * `VOKRA_ZONOS_GGUF` unset → clean skip with the recipe reproduced in
///     the log.
///   * set → open the file, apply [`assert_common_gguf_invariants`], and
///     cross-check every `vokra.zonos.*` chunk value against the primary
///     source ([`ZonosConfig::zonos_v0_1_transformer`], transcribed from
///     `huggingface.co/Zyphra/Zonos-v0.1-transformer/config.json`).
///   * additionally `VOKRA_ZONOS_REFDIR` set → refuse to report a pass:
///     stage-tap comparison is deferred until the scaffold gains a real
///     forward + `from_gguf` (fabricated pass 禁止).
#[test]
fn parity_tts_dac_zonos() {
    let (Some(gguf_path), refdir) = env_paths_for("zonos") else {
        eprintln!(
            "{}",
            skip_reason(
                "zonos",
                "Zyphra/Zonos-v0.1-transformer",
                // Snapshot SHA fetched 2026-07-25 via
                // huggingface.co/api/models/Zyphra/Zonos-v0.1-transformer
                "9d8331fc49cb5ba8aad2bb56cafd809c66598f4e",
                "Apache-2.0",
            )
        );
        return;
    };

    let gguf = GgufFile::open(&gguf_path)
        .unwrap_or_else(|e| panic!("open VOKRA_ZONOS_GGUF = {gguf_path:?}: {e}"));
    assert_common_gguf_invariants(&gguf, zonos::EXPECTED_ARCH);

    // Cross-check `vokra.zonos.*` metadata against the primary-source config.
    let cfg = ZonosConfig::zonos_v0_1_transformer();
    let bb = &cfg.backbone;

    assert_eq!(
        read_u32(&gguf, "vokra.zonos.sample_rate"),
        cfg.sample_rate,
        "vokra.zonos.sample_rate must equal ZONOS_SAMPLE_RATE ({})",
        cfg.sample_rate,
    );

    // Backbone hparams.
    assert_eq!(
        read_usize(&gguf, "vokra.zonos.arch.backbone.n_layer"),
        bb.n_layer,
    );
    assert_eq!(
        read_usize(&gguf, "vokra.zonos.arch.backbone.d_model"),
        bb.d_model,
    );
    assert_eq!(
        read_usize(&gguf, "vokra.zonos.arch.backbone.d_intermediate"),
        bb.d_intermediate,
    );
    assert_eq!(
        read_usize(&gguf, "vokra.zonos.arch.backbone.num_heads"),
        bb.num_heads,
    );
    assert_eq!(
        read_usize(&gguf, "vokra.zonos.arch.backbone.num_heads_kv"),
        bb.num_heads_kv,
    );
    assert_eq!(
        read_usize(&gguf, "vokra.zonos.arch.backbone.rotary_emb_dim"),
        bb.rotary_emb_dim,
    );
    assert_eq!(
        read_bool(&gguf, "vokra.zonos.arch.backbone.rotary_emb_interleaved"),
        bb.rotary_emb_interleaved,
    );
    assert_eq!(
        read_bool(&gguf, "vokra.zonos.arch.backbone.causal"),
        bb.causal,
    );
    assert_eq!(
        read_bool(&gguf, "vokra.zonos.arch.backbone.qkv_proj_bias"),
        bb.qkv_proj_bias,
    );
    assert_eq!(
        read_bool(&gguf, "vokra.zonos.arch.backbone.out_proj_bias"),
        bb.out_proj_bias,
    );
    assert_eq!(
        read_bool(&gguf, "vokra.zonos.arch.backbone.rms_norm"),
        bb.rms_norm,
        "vokra.zonos.arch.backbone.rms_norm must be false — Zonos uses \
         LayerNorm(weight+bias), NOT RMSNorm (upstream config toggle)",
    );
    let norm_eps = read_f32(&gguf, "vokra.zonos.arch.backbone.norm_epsilon");
    assert!(
        (norm_eps - bb.norm_epsilon).abs() <= f32::EPSILON * 8.0,
        "vokra.zonos.arch.backbone.norm_epsilon = {norm_eps}, expected {}",
        bb.norm_epsilon,
    );

    // Codebook I/O + delay pattern.
    assert_eq!(
        read_usize(&gguf, "vokra.zonos.num_codebooks"),
        cfg.num_codebooks,
    );
    assert_eq!(
        read_usize(&gguf, "vokra.zonos.codebook_vocab"),
        cfg.codebook_vocab,
    );
    assert_eq!(read_usize(&gguf, "vokra.zonos.head_vocab"), cfg.head_vocab,);
    assert_eq!(
        read_u32(&gguf, "vokra.zonos.eos_token_id"),
        cfg.eos_token_id,
    );
    assert_eq!(
        read_u32(&gguf, "vokra.zonos.masked_token_id"),
        cfg.masked_token_id,
    );

    let delay_count = read_usize(&gguf, "vokra.zonos.delay_pattern_count");
    assert_eq!(
        delay_count,
        cfg.delay_pattern.len(),
        "vokra.zonos.delay_pattern_count = {delay_count}, expected {} \
         (num_codebooks / delay_pattern must stay in lockstep)",
        cfg.delay_pattern.len(),
    );
    for (i, expected) in cfg.delay_pattern.iter().enumerate() {
        let key = format!("vokra.zonos.delay_pattern.{i}");
        assert_eq!(
            read_usize(&gguf, &key),
            *expected,
            "vokra.zonos.delay_pattern.{i} drift",
        );
    }

    // Prefix conditioner descriptor — count matches, and every conditioner's
    // primary-source name appears in metadata. The projection weights
    // themselves live in the tensor payloads (already covered by the finite
    // sweep above); this pins the descriptor ordering / naming.
    let cond_count = read_usize(&gguf, "vokra.zonos.prefix_conditioner.count");
    assert_eq!(
        cond_count,
        cfg.conditioners.len(),
        "prefix_conditioner.count = {cond_count}, expected {} — Zonos-v0.1's \
         7-conditioner descriptor is a hard shape contract",
        cfg.conditioners.len(),
    );
    for (i, cond) in cfg.conditioners.iter().enumerate() {
        let name_key = format!("vokra.zonos.prefix_conditioner.{i}.name");
        let name = gguf
            .get(&name_key)
            .and_then(|v| v.as_str())
            .unwrap_or_else(|| panic!("GGUF missing {name_key}"));
        assert_eq!(name, cond.name, "{name_key} drift");
    }

    // Sanity anchor: the language_id descriptor's range is the only one that
    // legally goes negative (-1 = unset), so pin it as an early canary if
    // the converter ever loses the sign.
    if let Some(lang) = cfg
        .conditioners
        .iter()
        .find(|c| c.name == "language_id")
        .map(|c| c.kind.clone())
    {
        if let ZonosConditionerKind::Integer { min_val, .. } = lang {
            assert_eq!(
                min_val, -1,
                "language_id.min_val must be -1 (upstream 'unset' sentinel)",
            );
        } else {
            panic!("language_id must be an Integer conditioner");
        }
    }

    eprintln!(
        "[parity_tts_dac::zonos] {} tensors, config extracted OK against \
         ZonosConfig::zonos_v0_1_transformer()",
        gguf.tensors().len(),
    );

    if let Some(refdir) = refdir {
        panic!("{}", flip_the_switch_deferred("zonos", &refdir));
    }
}

// -----------------------------------------------------------------------------
// Unit-only helper coverage (fires on every `cargo test`)
// -----------------------------------------------------------------------------

/// `env_paths_for` must return `(None, None)` for an arch whose env vars are
/// unset (the CI baseline). This pins the "unset → clean skip" contract at
/// the helper level so a future refactor that accidentally returns `Some`
/// on an empty env fails loudly here rather than in a live parity job.
///
/// The workspace forbids `unsafe` (`-D unsafe-code`), and `std::env::set_var`
/// is unsafe in Rust 2024, so this test cannot exercise the "env set but
/// file missing" branch directly without breaking the lint. The
/// `!p.is_file()` / `!p.is_dir()` filter inside `env_paths_for` is
/// documented on the function and is trivially inspectable there; a stronger
/// test would need `#[allow(unsafe_code)]` at the test level, which the
/// project posture does not admit for a mere assertion.
#[test]
fn env_paths_for_returns_none_when_unset() {
    // An arch string namespaced to this helper so no CI env var of the
    // corresponding `VOKRA_*_GGUF` / `_REFDIR` name can plausibly be set.
    let arch = "tts_dac_harness_helper_only";
    let (gguf, refdir) = env_paths_for(arch);
    assert!(
        gguf.is_none(),
        "expected VOKRA_{}_GGUF unset in the test env, got Some({:?}) — env leakage?",
        arch.to_ascii_uppercase(),
        gguf,
    );
    assert!(
        refdir.is_none(),
        "expected VOKRA_{}_REFDIR unset in the test env, got Some({:?})",
        arch.to_ascii_uppercase(),
        refdir,
    );
}

/// `skip_reason` must contain the arch name, the HF repo, the pinned SHA,
/// and the `vokra-cli convert` recipe. Guards against a future edit that
/// silently drops the recipe (leaving a bare "skipped" annotation).
#[test]
fn skip_reason_contains_reproduction_recipe() {
    let msg = skip_reason("dia", "nari-labs/Dia-1.6B", "abc123", "Apache-2.0");
    for token in &[
        "VOKRA_DIA_GGUF",
        "nari-labs/Dia-1.6B",
        "abc123",
        "Apache-2.0",
        "vokra-cli convert",
        "--model dia",
        "fabricated pass",
    ] {
        assert!(
            msg.contains(token),
            "skip_reason must contain {token:?}; got: {msg}"
        );
    }
}

/// `flip_the_switch_deferred` must name both owner steps AND the target
/// module type, so the failure is self-guiding.
#[test]
fn flip_the_switch_deferred_names_both_owner_steps() {
    let msg = flip_the_switch_deferred("dia", &PathBuf::from("/tmp/refdir"));
    for token in &[
        "VOKRA_DIA_REFDIR",
        "/tmp/refdir",
        "from_gguf",
        "stage-tap",
        "fabricated pass",
        "DiaTts",
    ] {
        assert!(
            msg.contains(token),
            "flip_the_switch_deferred must contain {token:?}; got: {msg}"
        );
    }
}

// =============================================================================
// Additional helper-level coverage (Phase 1 SoTA plan audit gap-fill,
// 2026-07-25). Each block's rationale is inlined so a future auditor sees
// why the test exists without cross-reference. Six gaps are addressed:
//
//   (1) `capitalize_first` had zero tests — a pure helper whose output the
//       owner-facing panic in `flip_the_switch_deferred` transitively relies
//       on. A regression that returned `""` or dropped the multi-byte tail
//       would produce panic text like `TtsTts` or bare `Tts` and no other
//       test would notice.
//   (2) `skip_reason` / `flip_the_switch_deferred` were dia-only — the zonos
//       leg's messages were never asserted, so a hard-coded `"dia"` or
//       `"DIA"` leak in either format string would ship undetected.
//   (3) `skip_reason` interpolation was tested only with a real arch — a
//       hard-coded `"DIA"` in the format string would still pass the dia
//       test because the tokens coincide with the hardcoded value.
//   (4) `env_paths_for`'s file-existence filter had no test (documented as
//       "unsafe env::set_var forbids" — the subprocess workaround preserves
//       `-D unsafe-code` at the workspace level).
//   (5) `assert_common_gguf_invariants` panic arms had no negative tests —
//       untested assertions are dead code from a verification standpoint.
//   (6) `env_paths_for` determinism across calls not pinned — locks out an
//       accidental global mutation / thread-local cache regression.
// =============================================================================

// -----------------------------------------------------------------------------
// (Gap 1) `capitalize_first` — pure helper the panic messages rely on.
// -----------------------------------------------------------------------------

#[test]
fn capitalize_first_empty_returns_empty() {
    // Pins the `chars().next()` early-return arm (`None => String::new()`).
    // A refactor that made this arm allocate a non-empty string (or panic)
    // would fail here before it surfaced downstream in `flip_the_switch_deferred`.
    assert_eq!(capitalize_first(""), "");
}

#[test]
fn capitalize_first_uppercases_ascii_first_char() {
    assert_eq!(capitalize_first("a"), "A");
    // Idempotent on an already-capital first char: `'D'.to_uppercase() == 'D'`.
    assert_eq!(capitalize_first("Dia"), "Dia");
}

#[test]
fn capitalize_first_pins_the_two_ty_strings_flip_switch_depends_on() {
    // The `flip_the_switch_deferred` panic embeds `"{ty}Tts"` where `ty` is
    // this function's output. Changing either transformation here would
    // silently change the owner-facing failure text (`DiaTts` / `ZonosTts`),
    // and only `flip_the_switch_deferred_names_both_owner_steps` +
    // `_for_zonos` catch it transitively — pin it directly.
    assert_eq!(capitalize_first("dia"), "Dia");
    assert_eq!(capitalize_first("zonos"), "Zonos");
}

#[test]
fn capitalize_first_preserves_multibyte_tail() {
    // The `chars().next()` + `to_uppercase().collect::<String>() + c.as_str()`
    // composition must not silently drop the tail when the first char is
    // multi-byte. `'é'.to_uppercase()` yields the single char `'É'`, so a
    // byte-oriented substitute (e.g. `s[..1].to_uppercase() + &s[1..]`)
    // would corrupt the tail here.
    assert_eq!(capitalize_first("éclair"), "Éclair");
}

// -----------------------------------------------------------------------------
// (Gap 2) Per-model zonos mirror of the dia skip/flip tests.
// -----------------------------------------------------------------------------

/// `skip_reason` — zonos arch. Mirrors `skip_reason_contains_reproduction_recipe`
/// so a hardcoded "dia" / "DIA" leak in the format string surfaces here.
#[test]
fn skip_reason_contains_zonos_reproduction_recipe() {
    let msg = skip_reason(
        "zonos",
        "Zyphra/Zonos-v0.1-transformer",
        "9d8331fc49cb5ba8aad2bb56cafd809c66598f4e",
        "Apache-2.0",
    );
    for token in &[
        "VOKRA_ZONOS_GGUF",
        "Zyphra/Zonos-v0.1-transformer",
        "9d8331fc49cb5ba8aad2bb56cafd809c66598f4e",
        "Apache-2.0",
        "vokra-cli convert",
        "--model zonos",
        "fabricated pass",
    ] {
        assert!(
            msg.contains(token),
            "skip_reason(zonos) must contain {token:?}; got: {msg}"
        );
    }
}

/// `flip_the_switch_deferred` — zonos arch. Also exercises `capitalize_first("zonos")`
/// end-to-end via the embedded `{ty}Tts` (`ZonosTts`) token.
#[test]
fn flip_the_switch_deferred_names_both_owner_steps_for_zonos() {
    let msg = flip_the_switch_deferred("zonos", &PathBuf::from("/tmp/zonos_refdir"));
    for token in &[
        "VOKRA_ZONOS_REFDIR",
        "/tmp/zonos_refdir",
        "from_gguf",
        "stage-tap",
        "fabricated pass",
        "ZonosTts", // pins capitalize_first("zonos") end-to-end
    ] {
        assert!(
            msg.contains(token),
            "flip_the_switch_deferred(zonos) must contain {token:?}; got: {msg}"
        );
    }
}

// -----------------------------------------------------------------------------
// (Gap 3) `skip_reason` interpolation with an arbitrary arch name.
// -----------------------------------------------------------------------------

/// Uses a *synthetic* arch name so a regression that hardcoded any real arch
/// literal (`"dia"` / `"DIA"` / `"zonos"`) into the format string would still
/// pass the dia + zonos tests (their tokens coincide with the hardcoded value)
/// but would fail here because none of those literals appears in the expected
/// output. This is the interpolation-vs-coincidence guard.
#[test]
fn skip_reason_interpolates_arch_verbatim() {
    let msg = skip_reason("foobar", "acme/foobar-1b", "deadbeef", "MIT");
    for token in &[
        "VOKRA_FOOBAR_GGUF",      // pins `.to_ascii_uppercase()` on the arch
        "VOKRA_FOOBAR_REFDIR",    // pins the second interpolation site
        "--model foobar",         // pins the arch flows into the CLI recipe
        "parity_tts_dac::foobar", // pins the log-prefix
        "acme/foobar-1b",         // pins hf_repo substitution
        "deadbeef",               // pins revision substitution
        "MIT",                    // pins license substitution
    ] {
        assert!(
            msg.contains(token),
            "skip_reason must interpolate {token:?} verbatim; got: {msg}"
        );
    }
}

// -----------------------------------------------------------------------------
// (Gap 4) `env_paths_for` file-existence filter via subprocess.
//
// `std::env::set_var` is unsafe in Rust 2024 and the workspace commits to
// `-D unsafe-code`, so the `.is_file()` / `.is_dir()` filter — the seam
// that keeps a stale env pointer from turning into a fabricated pass —
// cannot be exercised in-process. Spawn `current_exe()` with the target
// env vars set to paths that do NOT exist and `--exact` filtered to this
// test; the inner branch (guarded by a helper env var) then asserts
// `env_paths_for` drops both to `None`. Fires uniformly on Linux/macOS/Windows.
// -----------------------------------------------------------------------------

#[test]
fn env_paths_for_drops_missing_paths_via_subprocess() {
    const HELPER_ENV: &str = "VOKRA_TTS_DAC_FILTER_SUBPROC";
    const HELPER_ARCH: &str = "tts_dac_filter_probe";
    const OUTER_TEST_NAME: &str = "env_paths_for_drops_missing_paths_via_subprocess";
    const CANARY: &str = "VOKRA_TTS_DAC_FILTER_SUBPROC_CANARY_OK";

    // Inner (subprocess) branch: HELPER_ENV is set, and the two probe env
    // vars point at paths that DO NOT exist. `env_paths_for` must drop both
    // to `None` via its `.is_file()` / `.is_dir()` filter. A canary marker
    // is printed to stderr on success so the parent can distinguish "test
    // ran + passed" from "test did not match the --exact filter" (which
    // would otherwise be a silent false-pass).
    if std::env::var_os(HELPER_ENV).is_some() {
        let (gguf, refdir) = env_paths_for(HELPER_ARCH);
        assert!(
            gguf.is_none(),
            "filter regression: env_paths_for kept a non-existent GGUF path; \
             got Some({gguf:?}) — a stale VOKRA_*_GGUF pointer would now \
             yield a fabricated pass at GgufFile::open (FR-EX-08)"
        );
        assert!(
            refdir.is_none(),
            "filter regression: env_paths_for kept a non-existent REFDIR path; \
             got Some({refdir:?}) — a stale VOKRA_*_REFDIR pointer would now \
             flip the switch on reference dumps that never arrived"
        );
        eprintln!("{CANARY}");
        return;
    }

    // Outer branch: construct two ghost paths (built but never created) and
    // spawn the same test binary with them wired into the target env vars.
    let ghost_stem = format!(
        "vokra-tts-dac-filter-ghost-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0),
    );
    let ghost_dir = std::env::temp_dir().join(&ghost_stem);
    let ghost_file = ghost_dir.join("no-such.gguf");
    // NB: neither `ghost_dir` nor `ghost_file` is created — `is_dir()` /
    // `is_file()` must return false on the subprocess side.

    let exe = std::env::current_exe().expect("current_exe under cargo test");
    let out = std::process::Command::new(&exe)
        .args(["--exact", OUTER_TEST_NAME, "--nocapture"])
        .env(HELPER_ENV, "1")
        .env(
            format!("VOKRA_{}_GGUF", HELPER_ARCH.to_ascii_uppercase()),
            &ghost_file,
        )
        .env(
            format!("VOKRA_{}_REFDIR", HELPER_ARCH.to_ascii_uppercase()),
            &ghost_dir,
        )
        .output()
        .expect("spawn env_paths_for filter subprocess");

    let stderr = String::from_utf8_lossy(&out.stderr);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "env_paths_for filter subprocess failed:\n  status: {:?}\n  stdout:\n{stdout}\n  stderr:\n{stderr}",
        out.status,
    );
    assert!(
        stderr.contains(CANARY),
        "subprocess did not run the inner branch — `--exact {OUTER_TEST_NAME}` \
         matched zero tests (typo? test moved?); stderr:\n{stderr}\nstdout:\n{stdout}"
    );
}

// -----------------------------------------------------------------------------
// (Gap 5) `assert_common_gguf_invariants` panic arms — three focused
// negative tests pin the six failure arms via representative failure modes.
// Uses `GgufBuilder` (a shared writer primitive in vokra-core) to build
// synthetic GGUF bytes; `catch_unwind` isolates the panic and lets us
// inspect the message. No real weight file is required — the whole point
// of these tests is that a converter regression would surface here loud
// and early, rather than waiting for a bad real GGUF to arrive.
// -----------------------------------------------------------------------------

/// Extract the `&str` panic message from a `catch_unwind` payload. `panic!`
/// with a formatted `String` yields a `String` payload; the plain `&str`
/// variant is also fielded so the helper is drop-in for both.
fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&'static str>() {
        (*s).to_owned()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "<non-string panic payload>".to_owned()
    }
}

/// Build a GGUF byte image with the given arch tag and one well-formed F32
/// tensor. A common base the negative tests then mutate one field at a time.
/// (Not used by the rank-0 / NaN tests, which need a specific bad tensor.)
fn build_minimal_gguf(arch: &str) -> Vec<u8> {
    let mut b = GgufBuilder::new();
    b.add_string(KEY_MODEL_ARCH, arch);
    // One well-formed tensor so the "zero tensors" arm doesn't fire ahead
    // of the arch-mismatch arm.
    b.add_tensor("t.f32", GgmlType::F32, vec![1], vec![0u8; 4])
        .expect("writer accepts a well-formed 1-elem F32 tensor");
    b.to_bytes().expect("serialize synthetic GGUF")
}

#[test]
fn assert_common_gguf_invariants_rejects_wrong_arch() {
    // Arch mismatch is the first arm (fires before the tensor loop). Pins
    // that a converter emitting the wrong `vokra.model.arch` is caught with
    // BOTH names in the message so drift is instantly diagnosable.
    let gguf = GgufFile::parse(build_minimal_gguf("whisper")).expect("parse synthetic");
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        assert_common_gguf_invariants(&gguf, "dia");
    }))
    .expect_err("arch mismatch must panic (converter or EXPECTED_ARCH drift)");
    let msg = panic_message(&*panic);
    assert!(
        msg.contains("whisper") && msg.contains("dia"),
        "arch-mismatch panic must name BOTH arch strings so drift is \
         instantly diagnosable; got: {msg}"
    );
}

#[test]
fn assert_common_gguf_invariants_rejects_rank_zero_tensor() {
    // Rank-0 tensor is what the "dimensions has rank 0" arm catches. The
    // writer accepts it (empty-dim element count = 1), the reader accepts
    // it (`n_dims = 0` is representable), the invariant guard rejects it
    // — this test pins that middle arm and names the offending tensor.
    let mut b = GgufBuilder::new();
    b.add_string(KEY_MODEL_ARCH, "dia");
    b.add_tensor("t.rank0", GgmlType::F32, vec![], vec![0u8; 4])
        .expect("writer accepts rank-0 (element count = 1)");
    let gguf = GgufFile::parse(b.to_bytes().expect("serialize")).expect("parse");
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        assert_common_gguf_invariants(&gguf, "dia");
    }))
    .expect_err("rank-0 tensor must panic");
    let msg = panic_message(&*panic);
    assert!(
        msg.contains("t.rank0") && msg.contains("rank 0"),
        "rank-0 panic must name the offending tensor and the 'rank 0' \
         arm; got: {msg}"
    );
}

#[test]
fn assert_common_gguf_invariants_rejects_nan_in_f32_payload() {
    // A non-finite F32 payload is what the "F32 passthrough" NaN/Inf arm
    // catches — the arm that keeps a converter regression from silently
    // leaking NaN into an "otherwise valid" GGUF (the F32 passthrough
    // path is expected to preserve safetensors bytes verbatim).
    let mut b = GgufBuilder::new();
    b.add_string(KEY_MODEL_ARCH, "dia");
    let mut payload = Vec::with_capacity(8);
    payload.extend_from_slice(&1.0f32.to_le_bytes()); // finite, index 0
    payload.extend_from_slice(&f32::NAN.to_le_bytes()); // non-finite, index 1
    b.add_tensor("t.nan", GgmlType::F32, vec![2], payload)
        .expect("payload length matches 2 * 4 bytes");
    let gguf = GgufFile::parse(b.to_bytes().expect("serialize")).expect("parse");
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        assert_common_gguf_invariants(&gguf, "dia");
    }))
    .expect_err("NaN payload must panic");
    let msg = panic_message(&*panic);
    assert!(
        msg.contains("t.nan") && msg.contains("non-finite"),
        "NaN panic must name the offending tensor and mention 'non-finite'; \
         got: {msg}"
    );
    // The NaN we injected was at index 1; the check names the first offender
    // so triage points straight at the bad element.
    assert!(
        msg.contains("index 1"),
        "NaN panic must name the offending index (1) so triage is direct; \
         got: {msg}"
    );
}

// -----------------------------------------------------------------------------
// (Gap 6) `env_paths_for` determinism — no observable state across calls.
// -----------------------------------------------------------------------------

/// Pins that `env_paths_for` is pure: repeated calls with the same arch
/// return the same tuple. Guards against an accidental global mutation /
/// thread-local cache regression that would silently change observed
/// behaviour on the 2nd call. Trivial cost, complements the single-call
/// `env_paths_for_returns_none_when_unset`.
#[test]
fn env_paths_for_is_deterministic_across_calls() {
    // Namespace differs from the other helper tests so there is no
    // cross-test env-var collision even under parallel `cargo test`.
    let arch = "tts_dac_harness_determinism_probe";
    let a = env_paths_for(arch);
    let b = env_paths_for(arch);
    assert_eq!(
        a, b,
        "env_paths_for must be pure — repeated calls with the same arch \
         must return identical tuples"
    );
    assert_eq!(
        a,
        (None, None),
        "with no env set, both entries must be None"
    );
}
