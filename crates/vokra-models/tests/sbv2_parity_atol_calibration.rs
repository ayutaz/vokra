//! Wave-4 PER-TENSOR-ATOL-CALIB (audit rank 18): pins the
//! [`AtolCalibration`] status of every [`PER_TENSOR_ATOL`] entry so
//! nobody can silently loosen a bound without touching the ADR
//! (memory `feedback-honest-parity-atol` redundant-recording rule).
//!
//! Every entry starts at [`AtolCalibration::EstimatedPreFixture`] and
//! flips to [`AtolCalibration::Measured`] only after the owner runs
//! the parity CI workflow_dispatch on a real SBV2 v2 fine-tune
//! checkpoint and records the measured max|Δ| in
//! `docs/adr/sbv2-parity-atol.md`. See [`PER_TENSOR_ATOL`]'s doc
//! "Wave-4 PER-TENSOR-ATOL-CALIB (2026-08-09) — status per entry"
//! section for the exact owner-side calibration steps.

use std::path::{Path, PathBuf};

use vokra_core::json::{self, JsonValue};
use vokra_models::sbv2::{
    ATOL_DEFAULT, AtolCalibration, MEL_LOSS_ATOL, PER_TENSOR_ATOL, atol_calibration_for,
    tolerance_for,
};

/// Every [`PER_TENSOR_ATOL`] key MUST have a match arm in
/// [`atol_calibration_for`] — a silent add without corresponding
/// status entry is disallowed. Failing here means someone added a new
/// override without deciding whether it's a pre-fixture estimate or
/// a measured bound (the audit's "no silent loosening" contract).
#[test]
fn every_atol_entry_has_a_calibration_status() {
    for (name, _atol) in PER_TENSOR_ATOL {
        assert!(
            atol_calibration_for(name).is_some(),
            "PER_TENSOR_ATOL entry `{name}` has no `atol_calibration_for` \
             match arm — add one so nobody can silently loosen the bound \
             without touching docs/adr/sbv2-parity-atol.md"
        );
    }
}

/// Repo-root-relative fixture directory (mirror of
/// [`parity_sbv2_real::fixtures_dir`]) — `CARGO_MANIFEST_DIR` is
/// `<repo>/crates/vokra-models`, and `cargo test`'s per-test working
/// directory is the crate root, not the invocation directory.
fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("tests")
        .join("fixtures")
        .join("sbv2")
}

/// Enumerates every tensor name listed in `reference_dump.manifest.json`'s
/// `tensors[]` array. Unlike the manifest reader in `parity_sbv2_real.rs`,
/// this helper does NOT need the reference `.bin` files themselves — the
/// manifest JSON is committed as a schema anchor even when the real
/// fixtures are absent, so the calibration-coverage assertion this file
/// makes can fire on every `cargo test` run (not gated by `--ignored`).
fn manifest_tensor_names() -> Vec<String> {
    let manifest_path = fixtures_dir().join("reference_dump.manifest.json");
    let bytes = std::fs::read(&manifest_path).unwrap_or_else(|e| {
        panic!(
            "{}: cannot read committed manifest schema anchor: {e}",
            manifest_path.display()
        )
    });
    let manifest = json::parse(&bytes)
        .unwrap_or_else(|e| panic!("{}: JSON parse error: {e}", manifest_path.display()));
    let tensors = manifest
        .get("tensors")
        .and_then(JsonValue::as_array)
        .unwrap_or_else(|| {
            panic!(
                "{}: `tensors` key missing or not an array — the manifest schema is broken",
                manifest_path.display(),
            )
        });
    tensors
        .iter()
        .map(|t| {
            t.get("name")
                .and_then(JsonValue::as_str)
                .unwrap_or_else(|| {
                    panic!(
                        "{}: tensors[] entry missing `name` string field",
                        manifest_path.display(),
                    )
                })
                .to_string()
        })
        .collect()
}

/// WP-01 CALIBRATION-COVERAGE (2026-08-09): every tensor named in the
/// committed `reference_dump.manifest.json` MUST have a match arm in
/// [`atol_calibration_for`] — even tensors that fall through to
/// [`ATOL_DEFAULT`] must be explicitly acknowledged via
/// [`AtolCalibration::UnmeasuredDefault`], so a future maintainer cannot
/// add a new manifest tensor and silently rely on the fall-through
/// without touching this file or the ADR.
///
/// Fail-closed rationale (memory `feedback-honest-parity-atol`): the
/// pre-WP-01 hole let 6 manifest tensors (`phoneme_embed`, `text_hidden`,
/// `bert_bridge_out`, `speaker_embed`, `style_projected`, `mel_hidden`)
/// pass `parity_sbv2_real`'s per-tensor gate at `ATOL_DEFAULT = 0.01`
/// with no derivation on record — indistinguishable at the CI log level
/// from "we derived a tight bound and hit it". Requiring
/// `atol_calibration_for` to return `Some(_)` — even
/// `UnmeasuredDefault` — forces the tensor into the pinning-status table
/// below, whose diff a future PR reviewer sees.
#[test]
fn every_manifest_tensor_has_a_calibration_status() {
    for name in manifest_tensor_names() {
        assert!(
            atol_calibration_for(&name).is_some(),
            "manifest tensor `{name}` has no `atol_calibration_for` match arm \
             — add one (either the applicable measured/estimated variant when \
             the bound has been derived, or `AtolCalibration::UnmeasuredDefault` \
             to acknowledge the fall-through to `ATOL_DEFAULT`). See \
             `docs/adr/sbv2-parity-atol.md` §5 for the fail-closed rationale."
        );
    }
}

/// The **snapshot** of every manifest tensor's current calibration
/// status. If the owner runs the parity CI + updates the ADR + flips
/// an arm to `Measured` (or derives a new bound and promotes an
/// `UnmeasuredDefault` to `EstimatedPreFixture`), this test forces THEM
/// to update this snapshot too — no drift between the code's status
/// and the test's expectation.
///
/// Kokoro-precedent: this same style of pin caught silent-loosening
/// attempts on the `PROSODY_F0_ATOL = 0.05` entry.
///
/// WP-01 CALIBRATION-COVERAGE (2026-08-09): the snapshot now covers
/// EVERY manifest tensor (not just PER_TENSOR_ATOL entries) — a
/// manifest addition that only touches `atol_calibration_for` without
/// also updating this snapshot trips the count-mismatch assertion
/// below.
#[test]
fn atol_calibration_status_is_pinned() {
    // Every manifest tensor's current calibration status. See
    // `PER_TENSOR_ATOL`'s doc "Wave-4 PER-TENSOR-ATOL-CALIB" section
    // for how to flip an `EstimatedPreFixture` to `Measured`; see
    // `docs/adr/sbv2-parity-atol.md` §5 for how to promote an
    // `UnmeasuredDefault` to `EstimatedPreFixture` (derive the bound
    // + add a `PER_TENSOR_ATOL` override) and then to `Measured`.
    let expected: &[(&str, AtolCalibration)] = &[
        // ---- PER_TENSOR_ATOL overrides ----
        ("bert_hidden_ja", AtolCalibration::EstimatedPreFixture),
        ("bert_hidden_en", AtolCalibration::EstimatedPreFixture),
        ("sdp_sample", AtolCalibration::EstimatedPreFixture),
        ("z_latent", AtolCalibration::EstimatedPreFixture),
        // Wave-9 (2026-08-09): `waveform` = 1.5 is `Measured` from CI
        // run 31303426623 max |Δ| = 0.9248 × ~1.6× margin. See
        // `PER_TENSOR_ATOL`'s `"waveform"` block-doc for derivation +
        // `docs/adr/sbv2-libm-strategy.md` §2.2 for why the bit-exact
        // libm follow-up is a documented deferral (not a fabricated pass).
        ("waveform", AtolCalibration::Measured),
        // ---- WP-01 pinned pass-throughs (manifest but no override) ----
        // Each of these currently falls back to ATOL_DEFAULT (0.01) in
        // `tolerance_for`. The `UnmeasuredDefault` status makes that
        // pass-through diff-visible; a future owner deriving a tighter
        // bound must promote the status here AND add a corresponding
        // entry to `PER_TENSOR_ATOL`, per `docs/adr/sbv2-parity-atol.md`
        // §5.
        ("phoneme_embed", AtolCalibration::UnmeasuredDefault),
        ("text_hidden", AtolCalibration::UnmeasuredDefault),
        ("bert_bridge_out", AtolCalibration::UnmeasuredDefault),
        ("speaker_embed", AtolCalibration::UnmeasuredDefault),
        ("style_projected", AtolCalibration::UnmeasuredDefault),
        ("mel_hidden", AtolCalibration::UnmeasuredDefault),
    ];
    // Count parity: WP-01 requires EVERY manifest tensor to be pinned
    // here. The manifest is the ground truth; a drift in either
    // direction (PR forgot to add a snapshot entry, or removed a
    // manifest tensor without shrinking this list) fires here.
    let manifest_len = manifest_tensor_names().len();
    assert_eq!(
        expected.len(),
        manifest_len,
        "atol_calibration_status_is_pinned snapshot has {} entries but manifest has {} \
         tensors — update this snapshot table to match the manifest, and add a \
         matching arm to `atol_calibration_for` (docs/adr/sbv2-parity-atol.md §6)",
        expected.len(),
        manifest_len,
    );
    for (name, expected_status) in expected {
        let actual = atol_calibration_for(name).unwrap_or_else(|| {
            panic!(
                "manifest tensor `{name}` lost its calibration status arm in \
                 `atol_calibration_for`"
            )
        });
        assert_eq!(
            actual, *expected_status,
            "manifest tensor `{name}` calibration status changed to {actual:?} \
             — if this is intentional, update BOTH this snapshot AND \
             `docs/adr/sbv2-parity-atol.md` (memory \
             `feedback-honest-parity-atol` redundant-recording rule)"
        );
    }
}

/// Every atol value is a POSITIVE float — the CI's per-tensor pass/fail
/// check would silently succeed on a zero or negative tolerance
/// (`|delta| <= 0` is only satisfied by delta==0, which never happens
/// on non-bit-exact reference forward passes). A pre-Wave-4 audit
/// found no ordering enforced here; this test locks it in.
#[test]
fn every_atol_value_is_positive() {
    for (name, atol) in PER_TENSOR_ATOL {
        assert!(
            *atol > 0.0 && atol.is_finite(),
            "PER_TENSOR_ATOL entry `{name}` = {atol} is non-positive or non-finite — \
             an atol of 0 makes the per-tensor pass gate silently unmeetable"
        );
    }
}

/// `tolerance_for` on any manifest tensor name returns a finite
/// positive f32. Consumers rely on this to divide by the atol or feed
/// it into the "abs(delta) <= atol" pass gate — a NaN or negative
/// would be a silent-wrong outcome the audit called out.
#[test]
fn tolerance_for_returns_finite_positive_on_every_manifest_tensor() {
    // Every tensor the Python dumper emits — see the design doc §10
    // "The 11 dumped tensors" table (dumper's TENSOR_SCHEMA).
    let manifest_tensors: &[&str] = &[
        "phoneme_embed",
        "text_hidden",
        "bert_hidden_ja",
        "bert_hidden_en",
        "bert_bridge_out",
        "speaker_embed",
        "style_projected",
        "sdp_sample",
        "mel_hidden",
        "z_latent",
        "waveform",
    ];
    for &name in manifest_tensors {
        let atol = tolerance_for(name);
        assert!(
            atol > 0.0 && atol.is_finite(),
            "tolerance_for({name}) = {atol} is non-positive or non-finite"
        );
    }
}

/// WP-04 (2026-08-09): `MEL_LOSS_ATOL` is a derived-aggregate atol (not
/// keyed under any dumped tensor name) that
/// `crates/vokra-models/tests/parity_sbv2_real.rs`'s mel-loss aggregator
/// uses. Pin its value here as a redundant-recording drift detector
/// (memory `feedback-honest-parity-atol` = the constant's own rustdoc
/// in `crates/vokra-models/src/sbv2/parity.rs` derives the value; this
/// pin fails if the const drifts without touching the derivation
/// docstring). Status is `EstimatedPreFixture`-equivalent — no
/// per-tensor `AtolCalibration` entry because MEL_LOSS_ATOL is not
/// a tolerance_for lookup target; instead we assert the raw const
/// stays at the scaffolded pre-fixture 0.05 until a real CI
/// measurement flips it (WP-04 follow-up = same owner-side workflow as
/// PER_TENSOR_ATOL `Measured` promotion in `docs/adr/sbv2-parity-atol.md`
/// §5-§6).
#[test]
fn mel_loss_atol_is_pinned_at_wp04_scaffold_value() {
    assert!(
        (MEL_LOSS_ATOL - 0.05).abs() < f32::EPSILON,
        "MEL_LOSS_ATOL drifted from the WP-04 scaffold 0.05. To flip \
         this to `Measured`, run parity-sbv2-real workflow_dispatch, \
         capture max mel-loss, update the derivation docstring in \
         `crates/vokra-models/src/sbv2/parity.rs::MEL_LOSS_ATOL` \
         (append-never-delete, per Kokoro PROSODY_F0_ATOL precedent), \
         and update THIS pin in the same commit"
    );
}

/// WP-04 (2026-08-09): `ATOL_DEFAULT` is the fall-through returned by
/// `tolerance_for` for any tensor NOT in `PER_TENSOR_ATOL`. WP-01
/// closed the atol_calibration_for hole by pinning every manifest
/// tensor's calibration status, but the fall-through value itself
/// (0.01, NFR-QL-01) can still silently drift. This pin fires as a
/// last-line-of-defense drift detector.
#[test]
fn atol_default_is_pinned_at_nfr_ql_01_scaffold() {
    assert!(
        (ATOL_DEFAULT - 0.01).abs() < f32::EPSILON,
        "ATOL_DEFAULT drifted from 0.01 (NFR-QL-01). Any change here \
         cascades to every UnmeasuredDefault tensor's effective bound \
         (WP-01 landed 6 such tensors) — update the derivation ADR \
         + this pin together, honest-atol discipline (memory \
         feedback-honest-parity-atol)."
    );
}
