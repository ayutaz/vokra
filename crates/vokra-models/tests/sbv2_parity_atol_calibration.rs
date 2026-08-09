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
        //
        // Status update 2026-08-09: the SBV2 hot-path libm swap (WP-05,
        // ~40h CC budget) has been ACCEPTED as a scoped exception; see
        // `docs/adr/sbv2-libm-strategy.md` §3.2.1. The "documented
        // deferral" wording above is preserved as pre-decision history
        // (append-never-delete). Workspace-wide vendoring of
        // `rust-lang/libm` / RLIBM / SLEEF stays rejected (§3.1/§3.2);
        // the in-tree hot-path swap is now the tightening path. When
        // WP-06 → WP-07 → WP-08 lands and CI measures a tighter bound,
        // flip this row's value in `PER_TENSOR_ATOL` per
        // `docs/adr/sbv2-parity-atol.md` §5 Revert 手続き (5-point
        // simultaneous update; do NOT loosen). The status here stays
        // `Measured` — the WP-05 land tightens the value, not the tier.
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

/// WP-02 (2026-08-09): drift detector — every `PER_TENSOR_ATOL` value
/// must stay within ±50% of the pinned baseline recorded in
/// `tests/fixtures/sbv2/atol-measurements.json`. A "silent" change to
/// any of the five atol numbers (or to any future entry) — that is,
/// tightening because the CI happened to pass, or loosening to unstick
/// a red run without capturing a fresh measurement — trips this test
/// even when the honesty-audit tests above still pass, because the
/// baseline file is what changes when the owner runs the empirical
/// measurement cycle (owner CI dispatch -> `sbv2_atol_updater.py`
/// proposes a new baseline -> owner reviews and commits).
///
/// # The 50% threshold: intentional, not arbitrary
///
/// The Kokoro precedent (`feedback-honest-parity-atol`, T17-fixup
/// #5/#6 REVERT) shows the risk pattern: an owner or an automated fix
/// nudges an atol from `0.02` to `0.05` "because CI is red" without
/// recording a measured max|Δ|. `0.05 / 0.02 = 2.5×` is caught here
/// (>50% drift). Conversely, an owner running the empirical cycle and
/// legitimately tightening `bert_hidden_ja` from an `EstimatedPreFixture`
/// `0.02` upper bound to a `Measured` `0.001` (say, `measured_max_diff
/// = 6e-4 × 1.6× margin`) is also caught here (drift ≫ 50%) — and that
/// is CORRECT: promoting `EstimatedPreFixture` → `Measured` with a
/// large tightening is exactly the moment the owner should be forced
/// to also update this baseline and the ADR, per the redundant-
/// recording rule. A small owner-driven tightening (say, `0.02 -> 0.015`,
/// 25% drift) passes here silently, which is fine — the ADR and the
/// snapshot table above ALSO gate that path.
///
/// # Why this file and not the ADR
///
/// `docs/adr/sbv2-parity-atol.md` is gitignore-local by the standing
/// `docs/adr/` convention (see this repo's `.gitignore` §"Second batch
/// 2026-07-04"); a test cannot depend on a file that a fresh clone
/// does not carry. `tests/fixtures/sbv2/atol-measurements.json` is
/// tracked so `cargo test` in a fresh clone can find it — the ADR
/// records the *human* narrative for a measurement, this file records
/// the *machine-checkable* baseline the drift detector reads.
///
/// # RED->GREEN->REFACTOR
///
/// A missing or malformed baseline file is a loud panic (FR-EX-08), not
/// a soft-skip: this is the "prep infra for the empirical cycle" landed
/// by WP-02, so the file MUST be committed alongside the test that
/// consumes it. A baseline entry with a non-finite / non-positive
/// number is also a loud panic — the drift ratio calculation would
/// silently divide by zero or overflow otherwise.
#[test]
fn atol_values_are_pinned_against_baseline_drift() {
    /// The maximum allowed ratio between an atol value and its pinned
    /// baseline. 50% drift in either direction is the "loud change"
    /// threshold — see the test's rustdoc for the Kokoro precedent
    /// this ratio was chosen against.
    const MAX_DRIFT_FRACTION: f32 = 0.50;

    let baseline_path = fixtures_dir().join("atol-measurements.json");
    let bytes = std::fs::read(&baseline_path).unwrap_or_else(|e| {
        panic!(
            "{}: cannot read the WP-02 pinned atol baseline ({e}). The file MUST be \
             committed alongside this test — a fresh clone must be able to run \
             `cargo test -p vokra-models --test sbv2_parity_atol_calibration` \
             without external CI artifacts. See the test's rustdoc for the \
             owner-side procedure that updates this baseline.",
            baseline_path.display(),
        )
    });
    let baseline = json::parse(&bytes)
        .unwrap_or_else(|e| panic!("{}: JSON parse error: {e}", baseline_path.display()));
    let per_tensor = baseline
        .get("per_tensor_atol")
        .and_then(JsonValue::as_object)
        .unwrap_or_else(|| {
            panic!(
                "{}: `per_tensor_atol` key missing or not an object — the WP-02 \
                 baseline schema is `{{\"per_tensor_atol\": {{<name>: <value>, ...}}}}`",
                baseline_path.display(),
            )
        });

    // Cross-check: every current `PER_TENSOR_ATOL` entry MUST be in the
    // baseline. A new entry added to the code without a matching baseline
    // update trips this — FR-EX-08 loud rather than a silent
    // "unpinned, so no drift possible" fall-through.
    for (name, atol) in PER_TENSOR_ATOL {
        let baseline_value = per_tensor
            .iter()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v)
            .unwrap_or_else(|| {
                panic!(
                    "{}: PER_TENSOR_ATOL entry `{name}` = {atol} has no baseline \
                     entry — add \"{name}\": {atol} to `per_tensor_atol` in \
                     the baseline file so future silent drift can be detected. \
                     Do NOT leave it out of the baseline just to make this test \
                     pass; the whole point is that new entries also get pinned.",
                    baseline_path.display(),
                )
            });
        let baseline_atol = match baseline_value {
            JsonValue::Float(f) => *f as f32,
            JsonValue::Int(i) => *i as f32,
            other => panic!(
                "{}: `per_tensor_atol.{name}` is not a JSON number: {other:?}",
                baseline_path.display(),
            ),
        };
        assert!(
            baseline_atol > 0.0 && baseline_atol.is_finite(),
            "{}: `per_tensor_atol.{name}` = {baseline_atol} is non-positive or \
             non-finite — baselines must be finite positive floats or the drift \
             ratio calculation is nonsense.",
            baseline_path.display(),
        );

        // Symmetric ratio: `abs(current - baseline) / baseline`. Because
        // `baseline_atol > 0` above, this cannot NaN or divide by zero.
        let drift = ((*atol - baseline_atol).abs()) / baseline_atol;
        assert!(
            drift <= MAX_DRIFT_FRACTION,
            "PER_TENSOR_ATOL entry `{name}` drifted {:.1}% from the pinned \
             baseline ({baseline_atol} -> {atol}); the WP-02 drift detector \
             refuses any change larger than {:.0}% without a matching baseline \
             update. Rerun the owner cycle (workflow_dispatch on \
             `parity-sbv2-real.yml`, then feed the atol-summary artifact into \
             `tools/parity/sbv2_atol_updater.py --apply`), review the proposal, \
             and commit the updated baseline + code change TOGETHER — never one \
             without the other. See `docs/adr/sbv2-parity-atol.md` §5.",
            drift * 100.0,
            MAX_DRIFT_FRACTION * 100.0,
        );
    }

    // Reverse cross-check: every baseline entry MUST have a matching
    // `PER_TENSOR_ATOL` entry. A stale baseline row (someone removed
    // an override from `PER_TENSOR_ATOL` but forgot to prune the
    // baseline) is a silent "why is this here" and eventually starts
    // masking bugs — fail LOUDLY on the first PR that drops the code
    // side without cleaning up the baseline.
    for (name, _) in per_tensor {
        let present_in_code = PER_TENSOR_ATOL.iter().any(|(n, _)| *n == name);
        assert!(
            present_in_code,
            "{}: baseline has an entry for `{name}` but `PER_TENSOR_ATOL` \
             does not — either the code lost an override (add it back or \
             promote `atol_calibration_for` to `UnmeasuredDefault` for a \
             pass-through) or the baseline is stale (remove the row and \
             record the removal in `docs/adr/sbv2-parity-atol.md`).",
            baseline_path.display(),
        );
    }
}
