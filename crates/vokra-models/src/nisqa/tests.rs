//! NISQA v2 runtime-binder tests — contract-constant pins, metadata
//! round-trip, manifest-derived variant discrimination, the load-bearing
//! head-order pin, and negative-space round-trips on every loud gate.
//!
//! # What "round-trip" can honestly mean here
//!
//! On a real waveform this would be `score(...)` returning five MOS
//! values, but the forward needs `F.adaptive_max_pool2d`, which does not
//! exist in `vokra-ops` (see the module doc). Fabricating a score would
//! violate CLAUDE.md 教訓 (a) ("loud-partial は fake-complete より
//! honest"). What is honestly testable:
//!
//! 1. **Contract-constant pin** — `ARCH` / `NAME` / `CATEGORY` /
//!    `UPSTREAM_URL` / `DEFAULT_LICENSE_SPDX` match the converter
//!    (`crates/vokra-convert/src/models/nisqa_v2_weight.rs`) exactly, so
//!    a converter-side rename lands here in the same commit or goes red.
//! 2. **Licence-tier pin** — the default SPDX really does resolve to the
//!    research-only T4 class, so the "never publish without
//!    `--allow-noncommercial`" claim in the module doc is enforced, not
//!    just asserted in prose.
//! 3. **Head-order pin** — the tensor layout (`mos, noi, dis, col,
//!    loud`) is pinned against the paper's prose order, which swaps
//!    coloration and discontinuity.
//! 4. **Result-struct population** — [`NisqaScore::from_heads`] fills all
//!    five fields from a head vector and round-trips through
//!    [`NisqaScore::to_heads`]. This is the "five-dimension result struct
//!    is populated" half of the task contract; the forward is the
//!    loud-partial half.
//! 5. **Loud-error negative space** — every documented blocker (missing
//!    arch / wrong arch / empty manifest / no pooling head / a missing
//!    cloned head / no CNN stage / a half-stamped `vokra.nisqa.*` group /
//!    an even `seg_length`) fires at its documented surface, in the
//!    documented error variant, naming the thing that is wrong.

use super::*;
use vokra_core::gguf::{GgmlType, GgufBuilder};

// ---------------------------------------------------------------------------
// Fixture builders
// ---------------------------------------------------------------------------

/// Adds a zero-filled F32 tensor of the given dims (mirror of the
/// `dnsmos_p808_p835::tests` helper).
fn add_zero(b: &mut GgufBuilder, name: &str, dims: &[u64]) {
    let n: u64 = dims.iter().product();
    b.add_tensor(
        name,
        GgmlType::F32,
        dims.to_vec(),
        vec![0u8; (n * 4) as usize],
    )
    .expect("add tensor");
}

/// A builder carrying the four metadata chunks the converter always
/// stamps, plus the weight-license class when the caller wants one.
fn base_builder(weight_license: Option<LicenseClass>) -> GgufBuilder {
    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    b.add_string(KEY_MODEL_CATEGORY, CATEGORY);
    b.add_string(KEY_PROVENANCE_UPSTREAM_URL, UPSTREAM_URL);
    if let Some(cls) = weight_license {
        b.add_string(chunks::KEY_PROVENANCE_WEIGHT_LICENSE, cls.as_str());
    }
    b
}

/// Adds the framewise CNN + time-dependency tensors every released
/// checkpoint carries. Names follow the upstream `state_dict` keys the
/// converter writes verbatim (`NISQA_DIM.cnn` is a `Framewise` wrapper
/// whose inner module is `.model`).
fn add_backbone(b: &mut GgufBuilder) {
    add_zero(b, "cnn.model.conv1.weight", &[16, 1, 3, 3]);
    add_zero(b, "cnn.model.bn1.weight", &[16]);
    add_zero(b, "time_dependency.model.linear.weight", &[64, 384]);
    add_zero(b, "time_dependency.model.norm1.weight", &[64]);
}

/// Adds the multidimensional variant's five cloned attention-pooling
/// heads, optionally omitting one so a test can prove the missing-clone
/// gate fires.
fn add_multidim_heads(b: &mut GgufBuilder, omit: Option<usize>) {
    for head in 0..N_HEADS {
        if omit == Some(head) {
            continue;
        }
        add_zero(
            b,
            &format!("{TENSOR_PREFIX_POOL_LAYERS}{head}.model.linear1.weight"),
            &[128, 64],
        );
        add_zero(
            b,
            &format!("{TENSOR_PREFIX_POOL_LAYERS}{head}.model.linear3.weight"),
            &[1, 64],
        );
    }
}

/// Adds the single-output variant's one attention-pooling head
/// (`NISQA.pool`).
fn add_single_head(b: &mut GgufBuilder) {
    add_zero(b, "pool.model.linear1.weight", &[128, 64]);
    add_zero(b, "pool.model.linear3.weight", &[1, 64]);
}

/// Stamps a complete, valid `vokra.nisqa.*` mel front-end group. The
/// values are the upstream standard-settings config
/// (`config/train_nisqa_cnn_sa_ap.yaml`); `sample_rate = 0` is the
/// sentinel for that file's `ms_sr: null`.
fn add_front_end_group(b: &mut GgufBuilder) {
    b.add_u32(KEY_NISQA_SAMPLE_RATE, 0);
    b.add_u32(KEY_NISQA_N_FFT, 4096);
    b.add_f32(KEY_NISQA_HOP_LENGTH_SEC, 0.01);
    b.add_f32(KEY_NISQA_WIN_LENGTH_SEC, 0.02);
    b.add_u32(KEY_NISQA_N_MELS, 48);
    b.add_f32(KEY_NISQA_FMAX, 20_000.0);
    b.add_u32(KEY_NISQA_SEG_LENGTH, 15);
    b.add_u32(KEY_NISQA_SEG_HOP_LENGTH, 4);
    b.add_u32(KEY_NISQA_MAX_SEGMENTS, 1300);
}

/// Stamps a complete, valid `vokra.nisqa.*` topology group (upstream
/// standard settings: `cnn_pool_1: [24,7]`, `cnn_pool_2: [12,5]`,
/// `cnn_pool_3: [6,3]`, `td_sa_nhead: 1`).
fn add_topology_group(b: &mut GgufBuilder) {
    b.add_u32(KEY_NISQA_CNN_POOL_1_H, 24);
    b.add_u32(KEY_NISQA_CNN_POOL_1_W, 7);
    b.add_u32(KEY_NISQA_CNN_POOL_2_H, 12);
    b.add_u32(KEY_NISQA_CNN_POOL_2_W, 5);
    b.add_u32(KEY_NISQA_CNN_POOL_3_H, 6);
    b.add_u32(KEY_NISQA_CNN_POOL_3_W, 3);
    b.add_u32(KEY_NISQA_TD_SA_NHEAD, 1);
}

/// Serialises a builder into a parsed [`GgufFile`].
fn finish(b: &GgufBuilder) -> GgufFile {
    GgufFile::parse(b.to_bytes().expect("serialise NISQA fixture")).expect("parse NISQA fixture")
}

/// The canonical fixture: a multidimensional checkpoint stamped exactly
/// the way the current converter stamps one (no `vokra.nisqa.*` group).
fn multidim_gguf(weight_license: Option<LicenseClass>) -> GgufFile {
    let mut b = base_builder(weight_license);
    add_backbone(&mut b);
    add_multidim_heads(&mut b, None);
    finish(&b)
}

/// Convenience: unwraps a [`VokraError::ModelLoad`] message or panics
/// with the variant actually seen.
fn expect_model_load(err: VokraError) -> String {
    match err {
        VokraError::ModelLoad(m) => m,
        other => panic!("expected VokraError::ModelLoad, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// 1 — Contract-constant pins (cross-crate consistency with the converter)
// ---------------------------------------------------------------------------

#[test]
fn contract_constants_mirror_the_converter() {
    assert_eq!(ARCH, "nisqa_v2_weight", "arch tag pin");
    assert_eq!(NAME, "nisqa_v2_weight", "model name pin");
    assert_eq!(
        CATEGORY, "eval",
        "NISQA is an eval-tier MOS predictor, sibling of dnsmos / utmos / \
         utmosv2 / torchaudio_squim"
    );
    assert_eq!(UPSTREAM_URL, "github.com/gabrielmittag/NISQA");
    assert_eq!(DEFAULT_LICENSE_SPDX, "cc-by-nc-sa-4.0");
    assert_eq!(KEY_MODEL_CATEGORY, "vokra.model.category");
    assert_eq!(KEY_PROVENANCE_UPSTREAM_URL, "vokra.provenance.upstream_url");
    // NISQA is GitHub-only — there is no HF mirror, which is why
    // provenance rides the `upstream_url` key rather than `upstream_hf`.
    assert!(
        !UPSTREAM_URL.contains("huggingface"),
        "NISQA has no HF mirror; provenance must ride the GitHub URL key"
    );
}

// ---------------------------------------------------------------------------
// 2 — Licence-tier pin (T4 / research-only is enforced, not just claimed)
// ---------------------------------------------------------------------------

#[test]
fn default_license_is_the_research_only_t4_class() {
    // Upstream README: code MIT, but "The model weights (nisqa.tar,
    // nisqa_mos_only.tar, nisqa_tts.tar) are provided under a Creative
    // Commons Attribution-NonCommercial-ShareAlike 4.0 International
    // (CC BY-NC-SA 4.0) License". The converter stamps that SPDX; this
    // pins what the class system makes of it.
    let cls = LicenseClass::from_license_str(DEFAULT_LICENSE_SPDX);
    assert_eq!(
        cls,
        LicenseClass::NonCommercialShareAlike,
        "cc-by-nc-sa-4.0 must resolve to NonCommercialShareAlike — the same \
         assertion the converter's own test makes"
    );
    assert!(
        cls.requires_research_flag(),
        "NISQA weights are non-commercial: loading must be research-flagged"
    );
    assert!(
        !cls.commercial_ok(),
        "NISQA weights must never be advertised as commercial-OK"
    );
    assert!(
        cls.requires_license_preserved(),
        "share-alike cascades: a derived GGUF stays CC-BY-NC-SA-4.0"
    );
}

// ---------------------------------------------------------------------------
// 3 — Head-order pin (tensor layout, NOT the paper's prose order)
// ---------------------------------------------------------------------------

#[test]
fn head_order_pins_the_tensor_layout_not_the_paper_prose() {
    // Verbatim from the `y_hat[:, i]` assignments in NISQA_lib.py.
    assert_eq!(HEAD_ORDER, ["mos", "noi", "dis", "col", "loud"]);
    assert_eq!(HEAD_ORDER.len(), N_HEADS);
    assert_eq!(N_HEADS, 5, "NISQA_DIM clones its pooling module 5 times");

    // The trap this pin exists for: the paper's prose lists "Noisiness,
    // Coloration, Discontinuity, Loudness" — coloration before
    // discontinuity — while the tensor layout is the other way round.
    let dis = HEAD_ORDER
        .iter()
        .position(|h| *h == "dis")
        .expect("dis head present");
    let col = HEAD_ORDER
        .iter()
        .position(|h| *h == "col")
        .expect("col head present");
    assert!(
        dis < col,
        "discontinuity precedes coloration in the tensor layout (index {dis} vs \
         {col}); following the paper's prose order here would silently swap two \
         plausible-looking scores"
    );

    assert_eq!(NisqaVariant::MultiDim.n_heads(), N_HEADS);
    assert_eq!(NisqaVariant::SingleOutput.n_heads(), 1);
    assert_eq!(NisqaVariant::MultiDim.upstream_class(), "NISQA_DIM");
    assert_eq!(NisqaVariant::SingleOutput.upstream_class(), "NISQA");
}

// ---------------------------------------------------------------------------
// 4 — The five-dimension result struct is really populated
// ---------------------------------------------------------------------------

#[test]
fn nisqa_score_from_heads_populates_all_five_dimensions() {
    // Deterministic synthetic head vector — distinct values so a
    // field-order mix-up cannot pass.
    let heads = [4.25_f32, 3.5, 2.75, 1.5, 4.75];
    let score = NisqaScore::from_heads(&heads).expect("5 heads must build a score");

    assert!((score.mos - 4.25).abs() < 1e-9, "head 0 -> mos");
    assert!((score.noisiness - 3.5).abs() < 1e-9, "head 1 -> noisiness");
    assert!(
        (score.discontinuity - 2.75).abs() < 1e-9,
        "head 2 -> discontinuity (NOT coloration)"
    );
    assert!(
        (score.coloration - 1.5).abs() < 1e-9,
        "head 3 -> coloration (NOT discontinuity)"
    );
    assert!((score.loudness - 4.75).abs() < 1e-9, "head 4 -> loudness");

    // Round-trip through the head vector.
    assert_eq!(score.to_heads(), heads);

    // Name-keyed lookup follows the same map; an unknown name yields
    // None rather than a default (FR-EX-08).
    for (i, name) in HEAD_ORDER.iter().enumerate() {
        let got = score.get(name).expect("every HEAD_ORDER name resolves");
        assert!(
            (got - heads[i]).abs() < 1e-9,
            "get(\"{name}\") must return head index {i}"
        );
    }
    assert!(
        score.get("naturalness").is_none(),
        "an unknown dimension name must be None, never a fabricated default"
    );
}

#[test]
fn nisqa_score_from_heads_rejects_a_wrong_width_vector() {
    for bad in [vec![1.0_f32], vec![1.0_f32; 4], vec![1.0_f32; 6]] {
        let n = bad.len();
        let Err(err) = NisqaScore::from_heads(&bad) else {
            panic!("expected InvalidArgument for a {n}-wide head vector");
        };
        match err {
            VokraError::InvalidArgument(m) => {
                assert!(
                    m.contains("FR-EX-08"),
                    "must cite the no-fabrication clause, got `{m}`"
                );
                assert!(
                    m.contains("fabricate"),
                    "must say padding would fabricate sub-scores, got `{m}`"
                );
            }
            other => panic!("expected VokraError::InvalidArgument, got {other:?}"),
        }
    }
}

// ---------------------------------------------------------------------------
// 5 — Synthetic GGUF binds (both variants)
// ---------------------------------------------------------------------------

#[test]
fn from_gguf_binds_a_multidim_checkpoint() {
    let file = multidim_gguf(Some(LicenseClass::NonCommercialShareAlike));
    let m = Nisqa::from_gguf(&file).expect("a well-formed multidim GGUF must bind");

    assert_eq!(m.variant(), NisqaVariant::MultiDim);
    assert_eq!(m.config().variant, NisqaVariant::MultiDim);
    assert_eq!(
        m.weight_license(),
        LicenseClass::NonCommercialShareAlike,
        "the converter's default stamp must round-trip"
    );
    assert!(
        m.is_research_only(),
        "CC-BY-NC-SA-4.0 weights are research-only (T4)"
    );
    // 4 backbone tensors + 2 per head x 5 heads.
    assert_eq!(m.tensor_count(), 4 + 2 * N_HEADS);

    // The current converter stamps no `vokra.nisqa.*` group, so both
    // spec groups are absent — that absence is the documented gap, not
    // an error.
    assert!(m.config().front_end.is_none());
    assert!(m.config().topology.is_none());
}

#[test]
fn from_gguf_binds_a_single_output_checkpoint() {
    let mut b = base_builder(Some(LicenseClass::NonCommercialShareAlike));
    add_backbone(&mut b);
    add_single_head(&mut b);
    let file = finish(&b);

    let m = Nisqa::from_gguf(&file).expect("a well-formed single-output GGUF must bind");
    assert_eq!(
        m.variant(),
        NisqaVariant::SingleOutput,
        "a `pool.` prefix (and no `pool_layers.`) means upstream `class NISQA`"
    );
    assert_eq!(m.variant().short(), "single-output");
}

#[test]
fn missing_license_stamp_fail_closes_to_unknown() {
    let file = multidim_gguf(None);
    let m = Nisqa::from_gguf(&file).expect("an unstamped GGUF still binds");
    assert_eq!(
        m.weight_license(),
        LicenseClass::Unknown,
        "no stamp must fail-closed to Unknown, never to Permissive"
    );
    assert!(
        m.is_research_only(),
        "Unknown is research-flagged, so an unstamped GGUF is gated too"
    );
}

// ---------------------------------------------------------------------------
// 6 — Arch verification (strict; refuses foreign GGUFs loudly)
// ---------------------------------------------------------------------------

#[test]
fn from_gguf_rejects_a_missing_arch_tag() {
    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_NAME, "some-other-name");
    add_backbone(&mut b);
    add_multidim_heads(&mut b, None);
    let file = finish(&b);

    let Err(err) = Nisqa::from_gguf(&file) else {
        panic!("expected ModelLoad when `vokra.model.arch` is absent");
    };
    let m = expect_model_load(err);
    assert!(
        m.contains("`vokra.model.arch`"),
        "must name the missing key, got `{m}`"
    );
    assert!(
        m.contains("not a Vokra-native nisqa GGUF"),
        "must name the missing-arch surface, got `{m}`"
    );
    assert!(
        m.contains(ARCH),
        "must say which arch value the converter stamps, got `{m}`"
    );
}

#[test]
fn from_gguf_rejects_a_foreign_arch_and_enumerates_the_eval_siblings() {
    // A DNSMOS GGUF handed to the NISQA binder by mistake. Both are
    // non-intrusive MOS predictors in `category = "eval"`, so the
    // mis-route is realistic — and DNSMOS's 1 + 3 output layout would
    // silently under-fill a five-dimension score.
    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, "dnsmos");
    b.add_string(chunks::KEY_MODEL_NAME, "dnsmos-p808-p835");
    add_zero(&mut b, "p808.conv1/kernel", &[3, 3, 1, 16]);
    let file = finish(&b);

    let Err(err) = Nisqa::from_gguf(&file) else {
        panic!("expected ModelLoad on a foreign arch tag");
    };
    let m = expect_model_load(err);
    assert!(
        m.contains("`dnsmos`") && m.contains("`nisqa_v2_weight`"),
        "must name both the got and the expected arch, got `{m}`"
    );
    for sibling in ["dnsmos", "utmos", "utmosv2", "torchaudio_squim"] {
        assert!(
            m.contains(sibling),
            "must enumerate eval-family sibling `{sibling}` so the reader can \
             tell which converter to re-run, got `{m}`"
        );
    }
    assert!(
        m.contains("FR-EX-08"),
        "must cite the no-silent-misroute clause, got `{m}`"
    );
}

// ---------------------------------------------------------------------------
// 7 — Missing-tensor gates (each names the thing that is missing)
// ---------------------------------------------------------------------------

#[test]
fn from_gguf_rejects_an_empty_tensor_manifest() {
    let b = base_builder(Some(LicenseClass::NonCommercialShareAlike));
    let file = finish(&b);

    let Err(err) = Nisqa::from_gguf(&file) else {
        panic!("expected ModelLoad on a zero-tensor GGUF");
    };
    let m = expect_model_load(err);
    assert!(m.contains("zero tensors"), "must name the gap, got `{m}`");
    assert!(m.contains("FR-EX-08"), "must cite the clause, got `{m}`");
    assert!(
        m.contains("vokra-cli convert --model nisqa-v2-weight"),
        "must include the repro command, got `{m}`"
    );
    assert!(
        m.contains(SIDECAR_PATH),
        "must name the sidecar that flattens the upstream pickle, got `{m}`"
    );
}

#[test]
fn from_gguf_rejects_a_checkpoint_with_no_pooling_head() {
    // Backbone only — no `pool_layers.` and no `pool.`.
    let mut b = base_builder(Some(LicenseClass::NonCommercialShareAlike));
    add_backbone(&mut b);
    let file = finish(&b);

    let Err(err) = Nisqa::from_gguf(&file) else {
        panic!("expected ModelLoad when neither pooling prefix is present");
    };
    let m = expect_model_load(err);
    assert!(
        m.contains(TENSOR_PREFIX_POOL_LAYERS),
        "must name the multidim pooling prefix, got `{m}`"
    );
    assert!(
        m.contains(TENSOR_PREFIX_POOL),
        "must name the single-output pooling prefix, got `{m}`"
    );
    assert!(
        m.contains("NISQA_DIM") && m.contains("NISQA"),
        "must name the upstream classes so the reader knows which is which, got `{m}`"
    );
}

#[test]
fn from_gguf_rejects_a_multidim_checkpoint_missing_one_cloned_head() {
    // Omit head 3 (`col`, coloration) — the manifest still looks
    // multidimensional, so without this gate the shortened head vector
    // would only surface after the forward had already run.
    let mut b = base_builder(Some(LicenseClass::NonCommercialShareAlike));
    add_backbone(&mut b);
    add_multidim_heads(&mut b, Some(3));
    let file = finish(&b);

    let Err(err) = Nisqa::from_gguf(&file) else {
        panic!("expected ModelLoad when a cloned pooling head is missing");
    };
    let m = expect_model_load(err);
    assert!(
        m.contains("pool_layers.3."),
        "must name the exact missing tensor prefix, got `{m}`"
    );
    assert!(
        m.contains("col"),
        "must name which dimension the missing clone predicts, got `{m}`"
    );
    assert!(m.contains("FR-EX-08"), "must cite the clause, got `{m}`");
}

#[test]
fn from_gguf_rejects_a_checkpoint_with_no_framewise_cnn_stage() {
    let mut b = base_builder(Some(LicenseClass::NonCommercialShareAlike));
    add_zero(&mut b, "time_dependency.model.linear.weight", &[64, 384]);
    add_multidim_heads(&mut b, None);
    let file = finish(&b);

    let Err(err) = Nisqa::from_gguf(&file) else {
        panic!("expected ModelLoad when the framewise CNN stage is absent");
    };
    let m = expect_model_load(err);
    assert!(
        m.contains(TENSOR_PREFIX_CNN),
        "must name the missing `cnn.` prefix, got `{m}`"
    );
    assert!(
        m.contains("AdaptCNN"),
        "must name the upstream framewise class, got `{m}`"
    );
}

#[test]
fn from_gguf_rejects_a_gguf_carrying_both_pooling_layouts() {
    // Two checkpoints merged into one artefact — upstream's two
    // top-level classes are alternatives, never both.
    let mut b = base_builder(Some(LicenseClass::NonCommercialShareAlike));
    add_backbone(&mut b);
    add_multidim_heads(&mut b, None);
    add_single_head(&mut b);
    let file = finish(&b);

    let Err(err) = Nisqa::from_gguf(&file) else {
        panic!("expected ModelLoad when both pooling layouts are present");
    };
    let m = expect_model_load(err);
    assert!(
        m.contains("BOTH"),
        "must say both layouts were seen, got `{m}`"
    );
    assert!(
        m.contains("FR-EX-08"),
        "must cite the refuse-rather-than-pick clause, got `{m}`"
    );
}

// ---------------------------------------------------------------------------
// 8 — The loud-partial forward
// ---------------------------------------------------------------------------

#[test]
fn score_loud_partials_and_names_the_missing_primitive() {
    let file = multidim_gguf(Some(LicenseClass::NonCommercialShareAlike));
    let m = Nisqa::from_gguf(&file).expect("fixture must bind");

    // 1 s of silence — a legitimate PCM shape; the gate must fire
    // before any front-end work, not because the input is degenerate.
    let pcm = vec![0.0_f32; 48_000];
    let Err(err) = m.score(&pcm) else {
        panic!("score must loud-partial — it cannot have produced a real MOS");
    };
    match err {
        VokraError::UnsupportedOp(msg) => {
            assert!(msg.contains("nisqa score"), "names the surface: {msg}");
            assert!(msg.contains("loud-partial"), "labels the posture: {msg}");

            // (1) the missing primitive, by exact identifier.
            assert!(
                msg.contains("adaptive_max_pool2d"),
                "must name the missing primitive by identifier: {msg}"
            );
            assert!(
                msg.contains("vokra-ops"),
                "must say which crate lacks it: {msg}"
            );

            // (2) the missing metadata group.
            assert!(
                msg.contains("vokra.nisqa."),
                "must name the metadata group to stamp: {msg}"
            );
            assert!(
                msg.contains(KEY_NISQA_TD_SA_NHEAD),
                "must name the head-count key that no tensor shape reveals: {msg}"
            );

            // (3) the missing sidecar.
            assert!(
                msg.contains(SIDECAR_PATH),
                "must name the sidecar to write: {msg}"
            );

            // Primary sources, so a follow-up wave has anchors to walk.
            for anchor in [
                PRIMARY_SOURCE_CODE,
                PRIMARY_SOURCE_PAPER,
                PRIMARY_SOURCE_MODEL_DEF,
                PRIMARY_SOURCE_CONFIG,
            ] {
                assert!(msg.contains(anchor), "must cite `{anchor}`: {msg}");
            }

            // The output contract the follow-up wave targets.
            for head in HEAD_ORDER {
                assert!(msg.contains(head), "must echo head `{head}`: {msg}");
            }
            assert!(
                msg.contains("FR-EX-08"),
                "must cite the no-fabrication clause: {msg}"
            );
        }
        other => panic!("expected VokraError::UnsupportedOp, got {other:?}"),
    }
}

#[test]
fn score_on_a_single_output_checkpoint_refuses_rather_than_fabricating() {
    let mut b = base_builder(Some(LicenseClass::NonCommercialShareAlike));
    add_backbone(&mut b);
    add_single_head(&mut b);
    let file = finish(&b);
    let m = Nisqa::from_gguf(&file).expect("single-output fixture must bind");

    let Err(err) = m.score(&[0.0_f32; 1024]) else {
        panic!("a single-output checkpoint cannot yield five dimensions");
    };
    match err {
        VokraError::InvalidArgument(msg) => {
            assert!(
                msg.contains("single-output"),
                "must name the bound variant: {msg}"
            );
            assert!(
                msg.contains("score_overall()"),
                "must point at the call that does work here: {msg}"
            );
            assert!(
                msg.contains("nisqa.tar"),
                "must name the release that carries all five heads: {msg}"
            );
            assert!(
                msg.contains("FR-EX-08"),
                "must cite the no-fabrication clause: {msg}"
            );
        }
        other => panic!("expected VokraError::InvalidArgument, got {other:?}"),
    }
}

#[test]
fn score_overall_loud_partials_on_both_variants() {
    // Multidimensional.
    let file = multidim_gguf(Some(LicenseClass::NonCommercialShareAlike));
    let m = Nisqa::from_gguf(&file).expect("multidim fixture must bind");
    let Err(err) = m.score_overall(&[0.0_f32; 1024]) else {
        panic!("score_overall must loud-partial on the multidim variant");
    };
    assert!(matches!(err, VokraError::UnsupportedOp(_)));

    // Single-output: still a loud-partial, NOT the InvalidArgument that
    // `score` raises — the overall MOS head does exist here, it is only
    // the forward that is missing.
    let mut b = base_builder(Some(LicenseClass::NonCommercialShareAlike));
    add_backbone(&mut b);
    add_single_head(&mut b);
    let single = finish(&b);
    let s = Nisqa::from_gguf(&single).expect("single-output fixture must bind");
    let Err(err) = s.score_overall(&[0.0_f32; 1024]) else {
        panic!("score_overall must loud-partial on the single-output variant");
    };
    match err {
        VokraError::UnsupportedOp(msg) => {
            assert!(
                msg.contains("NISQA"),
                "must name the upstream class it would have run: {msg}"
            );
        }
        other => panic!("expected VokraError::UnsupportedOp, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// 9 — The optional `vokra.nisqa.*` groups
// ---------------------------------------------------------------------------

#[test]
fn front_end_and_topology_groups_round_trip_when_stamped() {
    let mut b = base_builder(Some(LicenseClass::NonCommercialShareAlike));
    add_backbone(&mut b);
    add_multidim_heads(&mut b, None);
    add_front_end_group(&mut b);
    add_topology_group(&mut b);
    let file = finish(&b);

    let m = Nisqa::from_gguf(&file).expect("a fully stamped GGUF must bind");
    let fe = m
        .config()
        .front_end
        .expect("the stamped front-end group must parse");
    assert_eq!(fe.sample_rate, 0, "0 is the `ms_sr: null` sentinel");
    assert_eq!(fe.n_fft, 4096);
    assert!((fe.hop_length_sec - 0.01).abs() < 1e-9, "hop is in SECONDS");
    assert!(
        (fe.win_length_sec - 0.02).abs() < 1e-9,
        "window is in SECONDS"
    );
    assert_eq!(fe.n_mels, 48);
    assert!((fe.fmax - 20_000.0).abs() < 1e-3);
    assert_eq!(fe.seg_length, 15, "odd, as upstream requires");
    assert_eq!(fe.seg_hop_length, 4);
    assert_eq!(fe.max_segments, 1300);

    let topo = m
        .config()
        .topology
        .expect("the stamped topology group must parse");
    assert_eq!(topo.cnn_pool, [[24, 7], [12, 5], [6, 3]]);
    assert_eq!(topo.td_sa_nhead, 1);
}

#[test]
fn front_end_group_rejects_an_even_segment_length() {
    // Upstream `segment_specs` raises `ValueError('seg_length must be
    // odd!')` — an even width has no centre frame. Every other key of
    // the group is stamped correctly so the failure is unambiguously the
    // parity check and not the all-or-nothing rule.
    let mut b = base_builder(Some(LicenseClass::NonCommercialShareAlike));
    add_backbone(&mut b);
    add_multidim_heads(&mut b, None);
    b.add_u32(KEY_NISQA_SAMPLE_RATE, 0);
    b.add_u32(KEY_NISQA_N_FFT, 4096);
    b.add_f32(KEY_NISQA_HOP_LENGTH_SEC, 0.01);
    b.add_f32(KEY_NISQA_WIN_LENGTH_SEC, 0.02);
    b.add_u32(KEY_NISQA_N_MELS, 48);
    b.add_f32(KEY_NISQA_FMAX, 20_000.0);
    b.add_u32(KEY_NISQA_SEG_LENGTH, 14);
    b.add_u32(KEY_NISQA_SEG_HOP_LENGTH, 4);
    b.add_u32(KEY_NISQA_MAX_SEGMENTS, 1300);
    let file = finish(&b);

    let Err(err) = Nisqa::from_gguf(&file) else {
        panic!("expected ModelLoad on an even seg_length");
    };
    let m = expect_model_load(err);
    assert!(
        m.contains(KEY_NISQA_SEG_LENGTH),
        "must name the offending key, got `{m}`"
    );
    assert!(
        m.contains("odd"),
        "must explain the odd-width requirement, got `{m}`"
    );
    assert!(m.contains("FR-EX-08"), "must cite the clause, got `{m}`");
}

#[test]
fn a_half_stamped_front_end_group_fails_loud() {
    // Only two of the nine keys — the all-or-nothing rule must fire
    // rather than defaulting the missing seven.
    let mut b = base_builder(Some(LicenseClass::NonCommercialShareAlike));
    add_backbone(&mut b);
    add_multidim_heads(&mut b, None);
    b.add_u32(KEY_NISQA_N_MELS, 48);
    b.add_u32(KEY_NISQA_SEG_LENGTH, 15);
    let file = finish(&b);

    let Err(err) = Nisqa::from_gguf(&file) else {
        panic!("expected ModelLoad on a half-stamped front-end group");
    };
    let m = expect_model_load(err);
    assert!(
        m.contains("all-or-nothing"),
        "must explain the group rule, got `{m}`"
    );
    assert!(
        m.contains(SIDECAR_PATH),
        "must point at the sidecar that produced the partial group, got `{m}`"
    );
}

#[test]
fn a_wrongly_typed_front_end_key_fails_loud() {
    // `n_mels` stamped as a String — coercing it would be a silent
    // shape change (FR-EX-08).
    let mut b = base_builder(Some(LicenseClass::NonCommercialShareAlike));
    add_backbone(&mut b);
    add_multidim_heads(&mut b, None);
    b.add_u32(KEY_NISQA_SAMPLE_RATE, 0);
    b.add_u32(KEY_NISQA_N_FFT, 4096);
    b.add_f32(KEY_NISQA_HOP_LENGTH_SEC, 0.01);
    b.add_f32(KEY_NISQA_WIN_LENGTH_SEC, 0.02);
    b.add_string(KEY_NISQA_N_MELS, "48");
    b.add_f32(KEY_NISQA_FMAX, 20_000.0);
    b.add_u32(KEY_NISQA_SEG_LENGTH, 15);
    b.add_u32(KEY_NISQA_SEG_HOP_LENGTH, 4);
    b.add_u32(KEY_NISQA_MAX_SEGMENTS, 1300);
    let file = finish(&b);

    let Err(err) = Nisqa::from_gguf(&file) else {
        panic!("expected ModelLoad on a wrongly typed metadata value");
    };
    let m = expect_model_load(err);
    assert!(
        m.contains(KEY_NISQA_N_MELS),
        "must name the offending key, got `{m}`"
    );
    assert!(
        m.contains("unsigned integer"),
        "must say what type was expected, got `{m}`"
    );
}

#[test]
fn topology_group_rejects_a_zero_pool_extent() {
    let mut b = base_builder(Some(LicenseClass::NonCommercialShareAlike));
    add_backbone(&mut b);
    add_multidim_heads(&mut b, None);
    b.add_u32(KEY_NISQA_CNN_POOL_1_H, 24);
    b.add_u32(KEY_NISQA_CNN_POOL_1_W, 7);
    b.add_u32(KEY_NISQA_CNN_POOL_2_H, 12);
    b.add_u32(KEY_NISQA_CNN_POOL_2_W, 5);
    b.add_u32(KEY_NISQA_CNN_POOL_3_H, 0);
    b.add_u32(KEY_NISQA_CNN_POOL_3_W, 3);
    b.add_u32(KEY_NISQA_TD_SA_NHEAD, 1);
    let file = finish(&b);

    let Err(err) = Nisqa::from_gguf(&file) else {
        panic!("expected ModelLoad on a zero adaptive-pool extent");
    };
    let m = expect_model_load(err);
    assert!(
        m.contains(KEY_NISQA_CNN_POOL_3_H),
        "must name the offending key, got `{m}`"
    );
    assert!(
        m.contains("adaptive-max-pool"),
        "must explain what the extent feeds, got `{m}`"
    );
}

// ---------------------------------------------------------------------------
// 10 — Fixed mel constants (hard-coded upstream, so knowable without
//      metadata — pinning them guards against a silent front-end drift)
// ---------------------------------------------------------------------------

#[test]
fn fixed_mel_constants_match_upstream_get_librosa_melspec() {
    // Verbatim from `get_librosa_melspec` in NISQA_lib.py:
    //   power=1.0, fmin=0.0, htk=False, norm='slaney', and
    //   amplitude_to_db(S, ref=1.0, amin=1e-4, top_db=80.0)
    assert!(
        (MEL_POWER - 1.0).abs() < 1e-9,
        "power=1.0 -> an AMPLITUDE mel-spectrogram, not the usual power one"
    );
    assert!((MEL_FMIN - 0.0).abs() < 1e-9);
    assert!((MEL_DB_REF - 1.0).abs() < 1e-9);
    assert!((MEL_DB_AMIN - 1e-4).abs() < 1e-12);
    assert!((MEL_DB_TOP_DB - 80.0).abs() < 1e-9);
    // Bound through a local so this is a runtime assertion rather than a
    // constant the compiler folds away before it can fail.
    let htk = MEL_HTK;
    assert!(
        !htk,
        "htk=False upstream -> the Slaney mel scale with norm='slaney', not the \
         HTK formula"
    );
}
