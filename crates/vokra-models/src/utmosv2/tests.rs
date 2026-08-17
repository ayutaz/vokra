//! UTMOSv2 runtime-binder tests (contract pins / metadata round-trip /
//! FR-EX-08 loud errors / loud-partial gate).
//!
//! Mirror of `crate::dnsmos_p808_p835::tests` and `crate::emotion2vec`'s
//! inline suite (the RMVPE loud-partial precedent).
//!
//! # What "round-trip" means here
//!
//! On a real checkpoint the end-to-end round-trip would be
//! `predict_mos(pcm) -> MOS ∈ [1, 5]`, but the multi-modal forward is a
//! loud-partial (see the module doc + [`Utmosv2::predict_mos`]).
//! Fabricating a score to make a test green would be exactly the
//! fake-complete CLAUDE.md 教訓 (a) warns about. The round-trips we *can*
//! honestly assert:
//!
//! 1. **Contract-constant pin** — every mirrored `pub const` matches the
//!    converter's value, so a converter-side drift lands here in the same
//!    commit or fails.
//! 2. **Metadata round-trip** — a synthetic GGUF built with exactly the keys
//!    `convert_utmosv2_file` writes binds, and every config field reads back.
//! 3. **Tensor-manifest round-trip** — every emitted tensor is bound with its
//!    real shape and dtype, and a BF16 payload dequantises bit-exactly.
//! 4. **Negative-space round-trip** — every documented gate fires at its
//!    documented surface, in its documented error variant, naming the
//!    offending key / tensor.
//! 5. **Loud-partial pin** — `predict_mos` returns `UnsupportedOp` naming
//!    every deferred stage, the metadata keys to stamp (verbatim from the
//!    `KEY_UTMOSV2_*` constants), the absent sidecar, the re-conversion
//!    command and both primary sources.

use super::*;
use vokra_core::gguf::GgufBuilder;

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// A builder pre-stamped with exactly the metadata
/// `vokra-convert::models::utmosv2::convert_utmosv2_file` writes.
fn utmosv2_metadata() -> GgufBuilder {
    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    b.add_string(KEY_MODEL_CATEGORY, CATEGORY);
    b.add_string(KEY_PROVENANCE_UPSTREAM_HF, UPSTREAM_HF);
    b.add_string(chunks::KEY_PROVENANCE_LICENSE, DEFAULT_LICENSE_SPDX);
    b.add_string(
        chunks::KEY_PROVENANCE_WEIGHT_LICENSE,
        LicenseClass::Permissive.as_str(),
    );
    b.add_string(chunks::KEY_PROVENANCE_MODEL_ID, NAME);
    b.add_string(
        chunks::KEY_PROVENANCE_SOURCE,
        "sarulab-speech/UTMOSv2 (synthetic test fixture)",
    );
    b
}

/// Little-endian F32 payload.
fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// Little-endian BF16 payload (top 16 bits of the f32 pattern — the exact
/// encoding `vokra_core::gguf::quant::decode_bf16` reverses). Callers pass
/// values whose low 16 mantissa bits are zero so the round-trip is exact.
fn bf16_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
        .collect()
}

/// Adds a dense F32 tensor.
fn add_f32(b: &mut GgufBuilder, name: &str, dims: &[u64], values: &[f32]) {
    b.add_tensor(name, GgmlType::F32, dims.to_vec(), f32_bytes(values))
        .expect("add F32 tensor");
}

/// The four BF16 values used by the pass-through fixture. Each is exactly
/// representable in bfloat16 (low 16 mantissa bits are zero), so the
/// `load_f32` round-trip is bit-exact rather than merely close.
const BF16_FIXTURE_VALUES: [f32; 4] = [1.0, -2.5, 0.5, 4.0];

/// A complete, legitimate UTMOSv2 GGUF.
///
/// Tensor names mirror the converter's own test fixtures (which the
/// converter docstring records as the upstream `state_dict` key convention
/// preserved verbatim through the offline flatten). They are fixture names
/// here, not a required manifest — see [`KNOWN_MODULE_PREFIXES`].
fn valid_gguf() -> GgufFile {
    let mut b = utmosv2_metadata();
    add_f32(
        &mut b,
        "ssl_encoder.encoder.layers.0.norm.weight",
        &[4],
        &[0.25, 0.5, 0.75, 1.0],
    );
    add_f32(
        &mut b,
        "listener_head.embedding.bias",
        &[3],
        &[-1.0, 0.0, 1.0],
    );
    // Rank-2 Linear weight — satisfies the "at least one weight matrix"
    // structural gate.
    add_f32(
        &mut b,
        "mos_head.linear.weight",
        &[3, 2],
        &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    );
    // BF16 pass-through arm (the converter emits GGUF type 30 verbatim).
    b.add_tensor(
        "ssl_encoder.encoder.layers.0.attn.q_proj.weight",
        GgmlType::BF16,
        vec![2, 2],
        bf16_bytes(&BF16_FIXTURE_VALUES),
    )
    .expect("add BF16 tensor");
    GgufFile::parse(b.to_bytes().expect("serialize")).expect("parse")
}

/// Unwraps a [`VokraError::ModelLoad`] payload, panicking on any other
/// variant so a gate that fires with the wrong variant is caught.
fn model_load_msg(err: VokraError) -> String {
    match err {
        VokraError::ModelLoad(m) => m,
        other => panic!("expected VokraError::ModelLoad, got {other:?}"),
    }
}

/// Unwraps a [`VokraError::UnsupportedOp`] payload.
fn unsupported_op_msg(err: VokraError) -> String {
    match err {
        VokraError::UnsupportedOp(m) => m,
        other => panic!("expected VokraError::UnsupportedOp, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// 1 — Contract-constant pins (cross-crate consistency with the converter)
// ---------------------------------------------------------------------------

#[test]
fn contract_constants_mirror_the_converter() {
    assert_eq!(ARCH, "utmosv2", "vokra.model.arch pin");
    assert_eq!(NAME, "utmosv2", "vokra.model.name pin");
    assert_eq!(CATEGORY, "eval", "vokra.model.category pin (eval family)");
    assert_eq!(
        UPSTREAM_HF, "sarulab-speech/UTMOSv2",
        "vokra.provenance.upstream_hf pin"
    );
    assert_eq!(
        DEFAULT_LICENSE_SPDX, "mit",
        "the converter records the upstream LICENSE as standard MIT"
    );
    assert_eq!(KEY_MODEL_CATEGORY, "vokra.model.category");
    assert_eq!(KEY_PROVENANCE_UPSTREAM_HF, "vokra.provenance.upstream_hf");
    assert_eq!(
        LicenseClass::from_license_str(DEFAULT_LICENSE_SPDX),
        LicenseClass::Permissive,
        "mit must classify Permissive (T1 tier) — the class the converter stamps"
    );
    // The dtype allow-list is transcribed from the converter's pass-through
    // match arm; a converter-side widening must land here too.
    assert_eq!(
        ACCEPTED_DTYPES,
        [GgmlType::F32, GgmlType::F16, GgmlType::BF16]
    );
    assert!((MOS_MIN - 1.0).abs() < f32::EPSILON);
    assert!((MOS_MAX - 5.0).abs() < f32::EPSILON);
    assert_eq!(Utmosv2::mos_range(), (MOS_MIN, MOS_MAX));
}

#[test]
fn arch_tag_is_distinct_from_every_sibling_eval_arch() {
    assert!(
        !SIBLING_EVAL_ARCH_TAGS.contains(&ARCH),
        "`{ARCH}` must not collide with a sibling eval-family arch tag"
    );
    assert_eq!(
        SIBLING_EVAL_ARCH_TAGS,
        ["utmos", "dnsmos", "nisqa_v2_weight", "torchaudio_squim"],
        "sibling enumeration pin — the wrong-arch diagnostic lists these verbatim"
    );
    // `utmos` (UTMOS22-strong, wav2vec2-BASE) is the closest neighbour and the
    // one a reader is most likely to confuse with UTMOSv2.
    assert!(SIBLING_EVAL_ARCH_TAGS.contains(&"utmos"));
}

#[test]
fn known_module_prefixes_are_hints_not_a_manifest() {
    assert_eq!(
        KNOWN_MODULE_PREFIXES,
        ["ssl_encoder.", "listener_head.", "mos_head."]
    );
    // A GGUF whose tensors match NONE of the hint prefixes must still bind —
    // the converter renames nothing, so a fork may legitimately use other
    // top-level module names and rejecting it would be a fabricated manifest.
    let mut b = utmosv2_metadata();
    add_f32(
        &mut b,
        "some.fork.linear.weight",
        &[2, 2],
        &[1.0, 2.0, 3.0, 4.0],
    );
    let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
    let m = Utmosv2::from_gguf(&file).expect("prefix hints must not gate the load");
    assert_eq!(m.tensor_count(), 1);
    for (_, count) in m.weights().module_prefix_inventory() {
        assert_eq!(count, 0, "no fixture tensor matches a hint prefix");
    }
}

// ---------------------------------------------------------------------------
// 2 — Metadata + tensor-manifest round-trip
// ---------------------------------------------------------------------------

#[test]
fn from_gguf_round_trips_a_synthetic_checkpoint() {
    let file = valid_gguf();
    let m = Utmosv2::from_gguf(&file).expect("a legitimate utmosv2 GGUF must bind");

    let cfg = m.config();
    assert_eq!(cfg.name, NAME);
    assert_eq!(cfg.category, CATEGORY);
    assert_eq!(cfg.upstream_hf, UPSTREAM_HF);
    assert!(cfg.is_canonical_upstream());
    assert_eq!(cfg.license_spdx.as_deref(), Some(DEFAULT_LICENSE_SPDX));
    assert_eq!(cfg.model_id.as_deref(), Some(NAME));
    assert!(
        cfg.source.is_some(),
        "free-text provenance source round-trips"
    );
    assert!(
        cfg.topology_keys_present.is_empty(),
        "the current converter stamps no `{KEY_UTMOSV2_PREFIX}` axes"
    );
    assert_eq!(m.weight_license(), LicenseClass::Permissive);

    assert_eq!(m.tensor_count(), 4, "every emitted tensor is bound");
    let w = m.weights();
    let head = w
        .require("mos_head.linear.weight")
        .expect("regressor head weight must be present");
    assert_eq!(head.dims, vec![3, 2]);
    assert_eq!(head.rank(), 2);
    assert_eq!(head.element_count(), 6);
    assert_eq!(head.dtype, GgmlType::F32);

    let bf16 = w
        .require("ssl_encoder.encoder.layers.0.attn.q_proj.weight")
        .expect("BF16 tensor must be present");
    assert_eq!(
        bf16.dtype,
        GgmlType::BF16,
        "BF16 rides the pass-through arm"
    );
    assert_eq!(bf16.dims, vec![2, 2]);
}

#[test]
fn module_prefix_inventory_counts_known_prefixes() {
    let file = valid_gguf();
    let m = Utmosv2::from_gguf(&file).unwrap();
    let inv = m.weights().module_prefix_inventory();
    assert_eq!(inv.len(), KNOWN_MODULE_PREFIXES.len());
    assert_eq!(inv[0], ("ssl_encoder.", 2));
    assert_eq!(inv[1], ("listener_head.", 1));
    assert_eq!(inv[2], ("mos_head.", 1));
    assert_eq!(m.weights().count_with_prefix("mos_head."), 1);
    assert_eq!(m.weights().count_with_prefix("nothing."), 0);
}

#[test]
fn load_f32_dequantises_bf16_bit_exactly() {
    let file = valid_gguf();
    let m = Utmosv2::from_gguf(&file).unwrap();
    let got = m
        .weights()
        .load_f32(&file, "ssl_encoder.encoder.layers.0.attn.q_proj.weight")
        .expect("BF16 payload must dequantise");
    assert_eq!(
        got, BF16_FIXTURE_VALUES,
        "bfloat16 is the top 16 bits of the f32 pattern — the round-trip is exact"
    );

    let f32_got = m
        .weights()
        .load_f32(&file, "listener_head.embedding.bias")
        .expect("F32 payload must decode");
    assert_eq!(f32_got, vec![-1.0, 0.0, 1.0]);
}

#[test]
fn load_f32_on_a_missing_tensor_is_loud_and_names_it() {
    let file = valid_gguf();
    let m = Utmosv2::from_gguf(&file).unwrap();
    let Err(err) = m.weights().load_f32(&file, "mos_head.linear.bias") else {
        panic!("expected a loud error for a tensor absent from the manifest");
    };
    let msg = model_load_msg(err);
    assert!(
        msg.contains("mos_head.linear.bias"),
        "must name the tensor: {msg}"
    );
}

// ---------------------------------------------------------------------------
// 3 — Arch-tag gates (strict verification, sibling mis-route refused)
// ---------------------------------------------------------------------------

#[test]
fn from_gguf_rejects_missing_arch() {
    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    add_f32(
        &mut b,
        "mos_head.linear.weight",
        &[2, 2],
        &[1.0, 2.0, 3.0, 4.0],
    );
    let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();

    let Err(err) = Utmosv2::from_gguf(&file) else {
        panic!("expected ModelLoad when `vokra.model.arch` is absent");
    };
    let msg = model_load_msg(err);
    assert!(
        msg.contains("vokra.model.arch"),
        "must name the missing key: {msg}"
    );
    assert!(
        msg.contains(CONVERT_COMMAND),
        "must name the repro command: {msg}"
    );
}

#[test]
fn from_gguf_rejects_wrong_arch_naming_expected_and_actual() {
    // `utmos` = UTMOS22-strong, the closest sibling and the likeliest
    // mis-route: same family, same output width, different SSL backbone.
    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, "utmos");
    b.add_string(chunks::KEY_MODEL_NAME, "utmos22-strong");
    add_f32(&mut b, "probe", &[2, 2], &[0.0, 0.0, 0.0, 0.0]);
    let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();

    let Err(err) = Utmosv2::from_gguf(&file) else {
        panic!("expected ModelLoad on a wrong arch tag");
    };
    let msg = model_load_msg(err);
    assert!(
        msg.contains("`utmos`"),
        "must name the arch actually seen: {msg}"
    );
    assert!(
        msg.contains("`utmosv2`"),
        "must name the arch expected: {msg}"
    );
    for sibling in SIBLING_EVAL_ARCH_TAGS {
        assert!(
            msg.contains(sibling),
            "must enumerate sibling `{sibling}`: {msg}"
        );
    }
    assert!(
        msg.contains("FR-EX-08"),
        "must cite the no-silent-load clause: {msg}"
    );
}

#[test]
fn from_gguf_rejects_non_string_arch() {
    let mut b = GgufBuilder::new();
    b.add_u32(chunks::KEY_MODEL_ARCH, 7);
    add_f32(&mut b, "probe", &[2, 2], &[0.0, 0.0, 0.0, 0.0]);
    let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();

    let Err(err) = Utmosv2::from_gguf(&file) else {
        panic!("expected ModelLoad when the arch chunk is not a String");
    };
    let msg = model_load_msg(err);
    assert!(
        msg.contains("expected String"),
        "must name the type gap: {msg}"
    );
}

// ---------------------------------------------------------------------------
// 4 — Config gates
// ---------------------------------------------------------------------------

#[test]
fn from_gguf_rejects_missing_name() {
    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(KEY_MODEL_CATEGORY, CATEGORY);
    b.add_string(KEY_PROVENANCE_UPSTREAM_HF, UPSTREAM_HF);
    add_f32(
        &mut b,
        "mos_head.linear.weight",
        &[2, 2],
        &[1.0, 2.0, 3.0, 4.0],
    );
    let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();

    let Err(err) = Utmosv2::from_gguf(&file) else {
        panic!("expected ModelLoad when `vokra.model.name` is absent");
    };
    let msg = model_load_msg(err);
    assert!(
        msg.contains(chunks::KEY_MODEL_NAME),
        "must name the key: {msg}"
    );
}

#[test]
fn from_gguf_rejects_missing_category() {
    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    b.add_string(KEY_PROVENANCE_UPSTREAM_HF, UPSTREAM_HF);
    add_f32(
        &mut b,
        "mos_head.linear.weight",
        &[2, 2],
        &[1.0, 2.0, 3.0, 4.0],
    );
    let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();

    let Err(err) = Utmosv2::from_gguf(&file) else {
        panic!("expected ModelLoad when `vokra.model.category` is absent");
    };
    let msg = model_load_msg(err);
    assert!(msg.contains(KEY_MODEL_CATEGORY), "must name the key: {msg}");
}

#[test]
fn from_gguf_rejects_wrong_category() {
    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    // A plausible mis-stamp: the ASR family tag.
    b.add_string(KEY_MODEL_CATEGORY, "asr");
    b.add_string(KEY_PROVENANCE_UPSTREAM_HF, UPSTREAM_HF);
    add_f32(
        &mut b,
        "mos_head.linear.weight",
        &[2, 2],
        &[1.0, 2.0, 3.0, 4.0],
    );
    let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();

    let Err(err) = Utmosv2::from_gguf(&file) else {
        panic!("expected ModelLoad on a wrong category");
    };
    let msg = model_load_msg(err);
    assert!(msg.contains("`asr`"), "must name the category seen: {msg}");
    assert!(
        msg.contains("`eval`"),
        "must name the category expected: {msg}"
    );
}

#[test]
fn from_gguf_rejects_missing_upstream_hf() {
    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    b.add_string(KEY_MODEL_CATEGORY, CATEGORY);
    add_f32(
        &mut b,
        "mos_head.linear.weight",
        &[2, 2],
        &[1.0, 2.0, 3.0, 4.0],
    );
    let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();

    let Err(err) = Utmosv2::from_gguf(&file) else {
        panic!("expected ModelLoad when the upstream slug is absent");
    };
    let msg = model_load_msg(err);
    assert!(
        msg.contains(KEY_PROVENANCE_UPSTREAM_HF),
        "must name the key: {msg}"
    );
}

#[test]
fn missing_license_stamp_fails_closed_to_unknown() {
    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    b.add_string(KEY_MODEL_CATEGORY, CATEGORY);
    b.add_string(KEY_PROVENANCE_UPSTREAM_HF, UPSTREAM_HF);
    // No `vokra.provenance.weight_license` stamp at all.
    add_f32(
        &mut b,
        "mos_head.linear.weight",
        &[2, 2],
        &[1.0, 2.0, 3.0, 4.0],
    );
    let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();

    let m = Utmosv2::from_gguf(&file).expect("licence is a compliance surface, not a load gate");
    assert_eq!(
        m.weight_license(),
        LicenseClass::Unknown,
        "an absent stamp must fail closed to Unknown, never to a permissive default"
    );
    assert!(m.config().license_spdx.is_none());
}

#[test]
fn non_canonical_upstream_slug_binds_but_is_reported() {
    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    b.add_string(KEY_MODEL_CATEGORY, CATEGORY);
    b.add_string(KEY_PROVENANCE_UPSTREAM_HF, "some-fork/UTMOSv2");
    add_f32(
        &mut b,
        "mos_head.linear.weight",
        &[2, 2],
        &[1.0, 2.0, 3.0, 4.0],
    );
    let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();

    let m = Utmosv2::from_gguf(&file).expect("a mirror / fork slug is not an error");
    assert!(!m.config().is_canonical_upstream());
    assert_eq!(m.config().upstream_hf, "some-fork/UTMOSv2");
}

// ---------------------------------------------------------------------------
// 5 — Tensor-manifest gates
// ---------------------------------------------------------------------------

#[test]
fn from_gguf_rejects_empty_tensor_manifest() {
    let b = utmosv2_metadata();
    let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();

    let Err(err) = Utmosv2::from_gguf(&file) else {
        panic!("expected ModelLoad on a zero-tensor GGUF");
    };
    let msg = model_load_msg(err);
    assert!(msg.contains("zero tensors"), "must name the offence: {msg}");
    assert!(msg.contains("FR-EX-08"), "must cite the clause: {msg}");
    assert!(msg.contains(CONVERT_COMMAND), "must give the repro: {msg}");
    assert!(msg.contains(SIDECAR_PATH), "must name the sidecar: {msg}");
}

#[test]
fn from_gguf_rejects_a_quantized_tensor() {
    // A Q6_K super-block is 256 elements / 210 bytes. The converter can never
    // emit this — its pass-through arm matches F32 / F16 / BF16 only — so its
    // presence means the file was re-quantised after conversion.
    let mut b = utmosv2_metadata();
    add_f32(
        &mut b,
        "mos_head.linear.weight",
        &[2, 2],
        &[1.0, 2.0, 3.0, 4.0],
    );
    b.add_tensor(
        "ssl_encoder.encoder.layers.0.fc1.weight",
        GgmlType::Q6K,
        vec![256],
        vec![0u8; 210],
    )
    .expect("add Q6_K tensor");
    let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();

    let Err(err) = Utmosv2::from_gguf(&file) else {
        panic!("expected ModelLoad on a quantised tensor");
    };
    let msg = model_load_msg(err);
    assert!(
        msg.contains("ssl_encoder.encoder.layers.0.fc1.weight"),
        "must name the offending tensor: {msg}"
    );
    assert!(msg.contains("Q6K"), "must name the offending dtype: {msg}");
    assert!(
        msg.contains("F32 / F16 / BF16"),
        "must state the allowed dtypes: {msg}"
    );
    assert!(
        msg.contains("NFR-QL-02"),
        "must explain why a re-quantised instrument is refused: {msg}"
    );
}

#[test]
fn from_gguf_rejects_a_rank0_tensor() {
    let mut b = utmosv2_metadata();
    add_f32(
        &mut b,
        "mos_head.linear.weight",
        &[2, 2],
        &[1.0, 2.0, 3.0, 4.0],
    );
    // Rank-0 scalar: no dimensions, one element.
    b.add_tensor("mos_head.scale", GgmlType::F32, vec![], f32_bytes(&[1.0]))
        .expect("add rank-0 tensor");
    let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();

    let Err(err) = Utmosv2::from_gguf(&file) else {
        panic!("expected ModelLoad on a rank-0 tensor");
    };
    let msg = model_load_msg(err);
    assert!(
        msg.contains("mos_head.scale"),
        "must name the tensor: {msg}"
    );
    assert!(msg.contains("rank-0"), "must name the offence: {msg}");
}

#[test]
fn from_gguf_rejects_a_zero_extent_dimension() {
    let mut b = utmosv2_metadata();
    // dims [2, 0] -> zero elements -> an empty payload that would read
    // downstream as "all weights are zero". Declared FIRST so its payload
    // sits at tensor-data offset 0 (a trailing zero-length tensor would land
    // exactly at the end of the data region, which is legal but a needlessly
    // fiddly fixture).
    b.add_tensor(
        "listener_head.embedding.weight",
        GgmlType::F32,
        vec![2, 0],
        Vec::new(),
    )
    .expect("add zero-extent tensor");
    add_f32(
        &mut b,
        "mos_head.linear.weight",
        &[2, 2],
        &[1.0, 2.0, 3.0, 4.0],
    );
    let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();

    let Err(err) = Utmosv2::from_gguf(&file) else {
        panic!("expected ModelLoad on a zero-extent dimension");
    };
    let msg = model_load_msg(err);
    assert!(
        msg.contains("listener_head.embedding.weight"),
        "must name the tensor: {msg}"
    );
    assert!(msg.contains("zero-extent"), "must name the offence: {msg}");
    assert!(
        msg.contains("axis 1"),
        "must name the offending axis: {msg}"
    );
}

#[test]
fn from_gguf_rejects_a_manifest_with_no_weight_matrix() {
    // Only 1-D biases / norms: a Regressor head cannot exist without a
    // single Linear weight matrix, so this is always a truncated flatten.
    let mut b = utmosv2_metadata();
    add_f32(
        &mut b,
        "listener_head.embedding.bias",
        &[3],
        &[0.0, 1.0, 2.0],
    );
    add_f32(&mut b, "mos_head.linear.bias", &[1], &[0.5]);
    let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();

    let Err(err) = Utmosv2::from_gguf(&file) else {
        panic!("expected ModelLoad when no rank>=2 tensor exists");
    };
    let msg = model_load_msg(err);
    assert!(
        msg.contains("rank >= 2"),
        "must state the structural gate: {msg}"
    );
    assert!(msg.contains(SIDECAR_PATH), "must name the sidecar: {msg}");
}

#[test]
fn duplicate_tensor_names_are_refused() {
    // `GgufBuilder::add_tensor` already refuses duplicates, so this defence
    // is exercised directly against the helper (a non-Vokra writer is the
    // only way a duplicate reaches the binder).
    let dup = |name: &str| Utmosv2Tensor {
        name: name.to_owned(),
        dims: vec![2, 2],
        dtype: GgmlType::F32,
    };
    let ok = [dup("a.weight"), dup("b.weight")];
    check_duplicate_names(&ok).expect("distinct names must pass");

    let bad = [dup("a.weight"), dup("b.weight"), dup("a.weight")];
    let Err(err) = check_duplicate_names(&bad) else {
        panic!("expected ModelLoad on a duplicated tensor name");
    };
    let msg = model_load_msg(err);
    assert!(msg.contains("a.weight"), "must name the duplicate: {msg}");
}

// ---------------------------------------------------------------------------
// 6 — Named-tensor accessors (what the follow-up forward wave binds against)
// ---------------------------------------------------------------------------

#[test]
fn require_missing_tensor_names_it_and_lists_nearby_names() {
    let file = valid_gguf();
    let m = Utmosv2::from_gguf(&file).unwrap();

    let Err(err) = m.weights().require("mos_head.linear.bias") else {
        panic!("expected ModelLoad for a tensor absent from the manifest");
    };
    let msg = model_load_msg(err);
    assert!(
        msg.contains("mos_head.linear.bias"),
        "must name the tensor asked for: {msg}"
    );
    assert!(
        msg.contains("mos_head.linear.weight"),
        "must list the nearby name that IS present: {msg}"
    );
    assert!(msg.contains(SIDECAR_PATH), "must name the sidecar: {msg}");
    assert!(msg.contains("FR-EX-08"), "must refuse to zero-fill: {msg}");
}

#[test]
fn require_falls_back_to_manifest_head_when_nothing_is_near() {
    let file = valid_gguf();
    let m = Utmosv2::from_gguf(&file).unwrap();

    let Err(err) = m.weights().require("totally.unrelated.key") else {
        panic!("expected ModelLoad for an unrelated tensor name");
    };
    let msg = model_load_msg(err);
    assert!(
        msg.contains("ssl_encoder.encoder.layers.0.norm.weight"),
        "with no prefix match the diagnostic falls back to real manifest names: {msg}"
    );
}

#[test]
fn require_shape_accepts_the_declared_shape_and_rejects_a_mismatch() {
    let file = valid_gguf();
    let m = Utmosv2::from_gguf(&file).unwrap();

    let t = m
        .weights()
        .require_shape("mos_head.linear.weight", &[3, 2])
        .expect("declared shape must match");
    assert_eq!(t.dims, vec![3, 2]);

    let Err(err) = m.weights().require_shape("mos_head.linear.weight", &[2, 3]) else {
        panic!("expected ModelLoad on a shape mismatch");
    };
    let msg = model_load_msg(err);
    assert!(
        msg.contains("mos_head.linear.weight"),
        "must name the tensor: {msg}"
    );
    assert!(msg.contains("[3, 2]"), "must state the actual shape: {msg}");
    assert!(
        msg.contains("[2, 3]"),
        "must state the expected shape: {msg}"
    );
}

// ---------------------------------------------------------------------------
// 7 — Loud-partial gate (no fabricated MOS, ever)
// ---------------------------------------------------------------------------

#[test]
fn predict_mos_is_a_loud_partial_naming_every_missing_piece() {
    let file = valid_gguf();
    let m = Utmosv2::from_gguf(&file).unwrap();
    // 1 s of silence at 16 kHz — a perfectly well-formed clip. The gate must
    // still fire: the forward is missing, not the input.
    let pcm = vec![0.0_f32; 16_000];

    let Err(err) = m.predict_mos(&pcm) else {
        panic!("predict_mos must loud-partial — a fabricated MOS is forbidden");
    };
    let msg = unsupported_op_msg(err);

    assert!(
        msg.contains("utmosv2 predict_mos"),
        "names the surface: {msg}"
    );
    assert!(msg.contains("loud-partial"), "names the posture: {msg}");

    // Every deferred stage.
    assert!(
        msg.contains("spectrogram"),
        "names the spectrogram branch: {msg}"
    );
    assert!(
        msg.contains("wav2vec2-large"),
        "names the SSL encoder: {msg}"
    );
    assert!(
        msg.contains("listener / domain conditioning"),
        "names the conditioning: {msg}"
    );
    assert!(
        msg.contains("Regressor head"),
        "names the head fusion: {msg}"
    );

    // Why it cannot be best-guessed.
    assert!(
        msg.contains("verbatim float pass-through"),
        "explains that the conversion contract stamps no topology: {msg}"
    );
    assert!(msg.contains("silent-wrong"), "states the hazard: {msg}");

    // The flip-the-switch recipe.
    assert!(
        msg.contains(SIDECAR_PATH),
        "names the absent sidecar: {msg}"
    );
    assert!(
        msg.contains(CONVERT_COMMAND),
        "names the re-conversion command: {msg}"
    );

    // Primary sources.
    assert!(msg.contains(PRIMARY_SOURCE_CODE), "cites the code: {msg}");
    assert!(msg.contains(PRIMARY_SOURCE_PAPER), "cites the paper: {msg}");

    // The no-fabrication clause and the gate it protects.
    assert!(msg.contains("FR-EX-08"), "cites the clause: {msg}");
    assert!(
        msg.contains("NFR-QL-02"),
        "names the quality gate a fake score would corrupt: {msg}"
    );
}

#[test]
fn loud_partial_message_references_topology_key_consts_verbatim() {
    // A rename of any `KEY_UTMOSV2_*` constant must land in the same commit
    // as the message, or the owner-facing recipe silently drifts (the
    // `dnsmos_p808_p835` precedent).
    let msg = unsupported_op_msg(predict_mos_loud_partial(&[]));
    for key in [
        KEY_UTMOSV2_PREFIX,
        KEY_UTMOSV2_SAMPLE_RATE,
        KEY_UTMOSV2_SSL_N_LAYER,
        KEY_UTMOSV2_SSL_HIDDEN_DIM,
        KEY_UTMOSV2_SPEC_N_MELS,
        KEY_UTMOSV2_HEAD_DIMS,
    ] {
        assert!(
            msg.contains(key),
            "recipe must name `{key}` verbatim: {msg}"
        );
    }
    assert!(
        msg.contains("neither half of the flip has landed"),
        "a GGUF with no topology axes must say so: {msg}"
    );
}

#[test]
fn loud_partial_reports_when_the_converter_half_already_landed() {
    // A GGUF that already advertises topology axes: the runtime forward is
    // then the ONLY remaining step, and the diagnostic must say so rather
    // than repeating the full recipe as if nothing had landed.
    let mut b = utmosv2_metadata();
    b.add_u32(KEY_UTMOSV2_SAMPLE_RATE, 16_000);
    b.add_u32(KEY_UTMOSV2_SSL_N_LAYER, 24);
    add_f32(
        &mut b,
        "mos_head.linear.weight",
        &[2, 2],
        &[1.0, 2.0, 3.0, 4.0],
    );
    let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();

    let m = Utmosv2::from_gguf(&file).expect("advertised axes must not block the load");
    assert_eq!(
        m.config().topology_keys_present,
        vec![
            KEY_UTMOSV2_SAMPLE_RATE.to_owned(),
            KEY_UTMOSV2_SSL_N_LAYER.to_owned(),
        ],
        "topology keys are surfaced sorted"
    );

    let Err(err) = m.predict_mos(&[0.0; 16]) else {
        panic!("advertised axes do not flip the forward on by themselves");
    };
    let msg = unsupported_op_msg(err);
    assert!(
        msg.contains("the converter half of the flip has landed"),
        "must distinguish a half-landed flip: {msg}"
    );
    assert!(
        msg.contains(KEY_UTMOSV2_SAMPLE_RATE),
        "echoes the keys: {msg}"
    );
}

#[test]
fn predict_mos_gate_fires_before_any_input_inspection() {
    // An empty clip is a perfectly good reason to reject a call, but the
    // caller must not be able to confuse "clip too short" with "not
    // implemented": the loud-partial fires first, unconditionally.
    let file = valid_gguf();
    let m = Utmosv2::from_gguf(&file).unwrap();
    let Err(err) = m.predict_mos(&[]) else {
        panic!("predict_mos must loud-partial even on an empty clip");
    };
    assert!(matches!(err, VokraError::UnsupportedOp(_)));
}

// ---------------------------------------------------------------------------
// 8 — The one real numeric stage: the terminal ACR clamp
// ---------------------------------------------------------------------------

#[test]
fn clamp_to_mos_range_clamps_onto_the_acr_scale() {
    let mid = clamp_to_mos_range(3.25).expect("in-range value passes through");
    assert!((mid - 3.25).abs() < 1e-6);

    let low = clamp_to_mos_range(-2.0).expect("below-range value clamps");
    assert!((low - MOS_MIN).abs() < 1e-6);

    let high = clamp_to_mos_range(11.5).expect("above-range value clamps");
    assert!((high - MOS_MAX).abs() < 1e-6);

    // The bounds themselves are fixed points.
    assert!((clamp_to_mos_range(MOS_MIN).unwrap() - MOS_MIN).abs() < 1e-6);
    assert!((clamp_to_mos_range(MOS_MAX).unwrap() - MOS_MAX).abs() < 1e-6);
}

#[test]
fn clamp_to_mos_range_refuses_non_finite_input() {
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let Err(err) = clamp_to_mos_range(bad) else {
            panic!("a non-finite regressor output must not be silently clamped");
        };
        match err {
            VokraError::InvalidArgument(m) => {
                assert!(m.contains("non-finite"), "must name the offence: {m}");
                assert!(m.contains("FR-EX-08"), "must cite the clause: {m}");
            }
            other => panic!("expected InvalidArgument, got {other:?}"),
        }
    }
}
