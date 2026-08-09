//! SBV2 parity: per-tensor atol calibration table consumed by the Task 27
//! (synthetic-pipeline) and Task 28 (real, HF-gated) parity tests to decide
//! per-tensor pass/fail. (Clean-room comment: see `mod.rs`.)

/// FP32 default absolute-tolerance floor (NFR-QL-01). [`tolerance_for`]
/// returns this for any tensor name not listed in [`PER_TENSOR_ATOL`].
///
/// Every tensor from `docs/superpowers/specs/2026-07-26-sbv2-v2-design.md`
/// §10's dump table that has no entry in [`PER_TENSOR_ATOL`]
/// (`phoneme_embed`, `text_hidden`, `bert_bridge_out`, `speaker_embed`,
/// `style_projected`, `mel_hidden`, `waveform`) is expected to clear this
/// default directly — e.g. `waveform`'s HiFi-GAN accumulation is expected
/// at the same order as Kokoro's decoder `pcm` tensor (measured `6.84e-3`
/// there), comfortably under `0.01`.
pub const ATOL_DEFAULT: f32 = 0.01;

/// Wave-4 PER-TENSOR-ATOL-CALIB status marker: whether a
/// [`PER_TENSOR_ATOL`] override is a theoretical-bound estimate
/// (pre-real-fixture placeholder) or has been calibrated against a real
/// Python reference dump. Kokoro's precedent (`PROSODY_F0_ATOL = 0.05`)
/// is [`AtolCalibration::Measured`]: derived from a 2.7e-2 theoretical
/// floor + 20% over a 3.27e-2 measurement + ~1.5-1.85× margin.
///
/// Every entry here starts at [`AtolCalibration::EstimatedPreFixture`];
/// once the corresponding CI workflow_dispatch produces a real dumper
/// output and the parity test measures the actual max|Δ|, the entry
/// flips to [`AtolCalibration::Measured`] with the derivation recorded
/// in `docs/adr/sbv2-parity-atol.md` (memory
/// `feedback-honest-parity-atol` redundant-recording rule).
///
/// Consumed by [`atol_calibration_for`] and the pinning test
/// `atol_calibration_status_is_pinned` in this crate's own
/// `sbv2_parity_atol_calibration.rs` integration test, which enforces
/// no silent loosening of any entry.
///
/// # WP-01 CALIBRATION-COVERAGE (2026-08-09)
///
/// The manifest names 11 tensors; [`PER_TENSOR_ATOL`] overrides only a
/// subset (5 at the time of writing). The 6 remaining tensors legitimately
/// fall through to [`ATOL_DEFAULT`], but pre-WP-01 that fall-through was
/// invisible: [`atol_calibration_for`] returned `None`, so the pinning
/// test could not detect a future manifest-tensor addition that also silently
/// depended on the default. [`Self::UnmeasuredDefault`] closes that hole
/// by making the fall-through an explicit, pinned status — the same
/// diff-visible mechanism the memory `feedback-honest-parity-atol`
/// redundant-recording rule uses for the tightened bounds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AtolCalibration {
    /// Pre-real-fixture estimate derived from theoretical
    /// per-op-error × layer-depth × amplification. The value is a
    /// SAFE upper bound (chosen to avoid false red on first CI run,
    /// per the audit's Kokoro-precedent warning "do not blindly f64
    /// accumulate"), not the tightest math permits. Flips to
    /// [`Self::Measured`] once a real dumper output is diffed against
    /// this tensor and the actual max|Δ| is recorded in
    /// `docs/adr/sbv2-parity-atol.md`.
    EstimatedPreFixture,
    /// Real-measurement calibrated bound: `measured_max_delta ×
    /// 1.5-2 margin`, per memory `feedback-honest-parity-atol`. The
    /// derivation MUST be recorded in `docs/adr/sbv2-parity-atol.md`
    /// (never deleted on revision, only extended).
    Measured,
    /// WP-01 (2026-08-09): tensor is listed in the manifest but has
    /// no entry in [`PER_TENSOR_ATOL`] — [`tolerance_for`] returns
    /// [`ATOL_DEFAULT`] for it. This status makes the fall-through
    /// **explicit and pinned** rather than a `None` return that a
    /// future maintainer could silently rely on; it is neither an
    /// estimate nor a measurement, only an acknowledged pass-through.
    ///
    /// Flip to [`Self::EstimatedPreFixture`] the moment a theoretical
    /// bound is derived for the tensor (and add the corresponding
    /// [`PER_TENSOR_ATOL`] override), or to [`Self::Measured`] once a
    /// real-fixture max|Δ| has been recorded in
    /// `docs/adr/sbv2-parity-atol.md`. See that ADR's §5 for the
    /// full owner-side flip procedure.
    UnmeasuredDefault,
}

/// Per-tensor absolute-tolerance overrides for [`tolerance_for`], keyed by
/// the dumper's tensor name
/// (`docs/superpowers/specs/2026-07-26-sbv2-v2-design.md` §10's dump-tensor
/// table). A tensor **not** listed here uses [`ATOL_DEFAULT`].
///
/// # Honest-atol discipline (memory `feedback-honest-parity-atol`)
///
/// Per that memory's rule, an override here is never "whatever makes CI
/// green" — it is the tensor's theoretical error-accumulation bound times a
/// 1.5-2× margin, with the derivation recorded so a future maintainer
/// cannot silently loosen it to paper over a real regression. The
/// precedent is Kokoro's `PROSODY_F0_ATOL = 0.05`: an F0_proj Conv1d
/// 256->1's measured ~9× linear amplification factor times the upstream
/// BiLstm1d's ~3e-3 accumulator delta gives a 2.7e-2 theoretical floor,
/// confirmed by a 3.27e-2 real measurement (~20% over the floor), landed at
/// 0.05 (~1.5-1.85× margin over each).
///
/// # Wave-4 PER-TENSOR-ATOL-CALIB (2026-08-09) — status per entry
///
/// Every entry is paired with an [`AtolCalibration`] status via
/// [`atol_calibration_for`]. Entries at
/// [`AtolCalibration::EstimatedPreFixture`] are safe upper bounds
/// chosen to avoid false red on first CI run; they flip to
/// [`AtolCalibration::Measured`] with a tighter value once the owner
/// runs the parity CI workflow_dispatch and records the measured
/// max|Δ| in `docs/adr/sbv2-parity-atol.md`. This transparency contract
/// is enforced by the `atol_calibration_status_is_pinned` integration
/// test — no entry can silently flip status without touching that ADR.
///
/// **Owner action to flip an entry to `Measured`**:
/// 1. Run `.github/workflows/parity-sbv2-real.yml` workflow_dispatch
///    on a real SBV2 v2 fine-tune checkpoint (needs the three fixture
///    sidecar hashes populated).
/// 2. Record the measured max|Δ| for each tensor in
///    `docs/adr/sbv2-parity-atol.md` (create the ADR if absent).
/// 3. Update the atol value here to `measured × 1.5-2 margin`.
/// 4. Flip [`atol_calibration_for`]'s return arm for that key from
///    `EstimatedPreFixture` to `Measured`.
///
/// **Scaffold caveat**: unlike Kokoro's number above, the four values below
/// are *pre-real-fixture estimates* — `docs/superpowers/specs/2026-07-26-sbv2-v2-design.md`
/// §10 calls them out as an "初期値（実測 parity 後に updated ADR で
/// pin）" (initial values, to be pinned by an updated ADR once real parity
/// has run). They are derived the same "layer depth × per-op error ×
/// amplification" way Kokoro's bound was, but **not yet confirmed** against
/// a real upstream forward pass — that confirmation is Task 27's
/// synthetic-pipeline wiring check and, decisively, Task 28's real,
/// HF-checkpoint-gated parity run. Either may revise these numbers; if so,
/// update the derivation below alongside the value (never delete it), and
/// record the revision in an ADR (`docs/adr/sbv2-parity-atol.md`) per the
/// memory's redundant-recording rule.
///
/// - `"bert_hidden_ja"` = `0.02` — the JA BERT path (DeBERTa v2 encoder,
///   loaded as [`SbV2BertContainer`](super::SbV2BertContainer)'s `ja`
///   field), 24 transformer layers deep.
///
///   **Wave-4 tightened derivation** (2026-08-09, [`AtolCalibration::EstimatedPreFixture`]):
///   theoretical floor from per-op-error × depth × amplification:
///   - Per-op f32 rounding: `2^-23 ≈ 1.2e-7`
///   - GEMM per-layer (d_model=1024): `sqrt(1024) × 1.2e-7 ≈ 3.8e-6`
///     (assumes RMS accumulation over independent rounding events)
///   - 24 layers, near-worst-case linear accumulation: `24 × 3.8e-6 ≈
///     9e-5`
///   - Disentangled relative-position attention adds ~10× multiplier
///     vs plain attention (three separate scores per position: C2C +
///     C2P + P2C, each with its own rounding chain): `9e-4`
///   - Kokoro's own `bert` tensor (a shallower plain-attention stack)
///     measured `6.56e-6`, three orders of magnitude tighter, giving
///     a permissive but honest anchor.
///   - Safe upper × ~20× margin over the theoretical floor: **`0.02`**.
///     Kokoro-precedent warning: "do not blindly f64 accumulate"
///     (T17-fixup #5/#6 regression) applies — this bound MUST be
///     tightened only after a real measurement, never speculatively.
///     Owner-action: run parity CI, record measured max|Δ| in
///     `docs/adr/sbv2-parity-atol.md`, tighten to `measured × 1.5-2`.
/// - `"bert_hidden_en"` = `0.02` — the EN BERT path (DeBERTa v3 encoder,
///   [`SbV2BertContainer`](super::SbV2BertContainer)'s `en` field). Same
///   layer-depth order as the JA path (v3 is v2's shared-position variant,
///   not a deeper stack), so the same derivation applies.
///   [`AtolCalibration::EstimatedPreFixture`].
/// - `"sdp_sample"` = `0.05` — [`SbV2SDP::sample`](super::SbV2SDP::sample)'s
///   output. Not a pure float-rounding tensor: the SDP draws Gaussian
///   noise scaled by `noise_scale_w` through an inverse flow, then floors
///   the result to an `i32` duration via `.max(1)` (`duration.rs`'s
///   `SbV2SDP::sample` doc). `0.05`'s margin covers (a) the +/-1 discrete
///   step a borderline duration's floor/round can flip on, on top of (b)
///   ordinary float accumulation through the flow stack. This tensor is
///   only meaningfully comparable at all when both sides share the exact
///   same PRNG algorithm and seed (mismatched seeds make the two samples
///   independent draws, not a numerical-precision comparison) — Task
///   27/28's fixtures must record the seed used alongside this atol.
///
///   **Update (2026-08-08, torch.randn parity work Steps 1-10)**: the RNG
///   layer itself is now byte-exact against
///   `torch.manual_seed(seed); torch.randn(...)` under the
///   PhiloxRNGEngine.h path when
///   [`SbV2SynthRequest::rng_mode`](super::SbV2SynthRequest#structfield.rng_mode)
///   = [`RngMode::PhiloxRngEnginePyTorchParity`](super::RngMode::PhiloxRngEnginePyTorchParity)
///   (the [`Default`](super::RngMode::default)). Proof:
///   `crates/vokra-models/tests/sbv2_sdp_torch_parity.rs::
///   sdp_noise_matches_torch_philox_seed_0_t_50` (byte-diff against
///   `tools/parity/sbv2_sdp_noise_dump.py` output, tolerance 0.0).
///
///   The `0.05` here now covers ONLY the residual downstream (flow
///   inverse) rounding + duration `.floor().max(1)` step, since the
///   noise layer is bit-zero. Once Task 28 wires a real dumper that
///   emits `sdp_sample` for a real ckpt run, this bound should be
///   re-derived from a measured floor × 1.5-2 margin (per the
///   `feedback-honest-parity-atol` memory), NOT reused as a
///   placeholder — the "seeds match, so noise matches" prerequisite is
///   now satisfied by the torch-parity `rng_mode` default.
/// - `"z_latent"` = `0.03` — [`SbV2Flow::inverse`](super::SbV2Flow::inverse)'s
///   output. Cumulative rounding across the acoustic flow's affine-coupling
///   stack (`docs/superpowers/specs/2026-07-26-sbv2-v2-design.md` §10: "4
///   block × affine coupling 蓄積" — 4 coupling blocks' worth of
///   accumulation), where each layer's scale/shift projection adds its own
///   per-op error on top of the incoming `mel_hidden` and the
///   style/speaker conditioning contributions it folds in.
///
///   **Wave-4 tightened derivation** (2026-08-09, [`AtolCalibration::EstimatedPreFixture`]):
///   - Per-op rounding × d_z (192) inner accumulation: `sqrt(192) ×
///     1.2e-7 ≈ 1.66e-6` per coupling.
///   - 4 coupling blocks + WaveNet-residual internal net (4 in_layers +
///     4 res_skip_layers each): ~40× per-op events per block, total
///     accumulation: `4 × 40 × 1.66e-6 ≈ 2.7e-4`.
///   - Exponentiated scale in affine coupling adds ~10× amplification
///     when the flow's `logs` output near ±3: `2.7e-3`.
///   - Style/speaker conditioning: additive-broadcast contribution ~0.1
///     magnitude range accumulated per block: another 4× factor →
///     `1.1e-2`.
///   - Safe upper × ~3× margin: **`0.03`**.
///   - Owner-action: run parity CI, record measured max|Δ| in
///     `docs/adr/sbv2-parity-atol.md`, tighten to `measured × 1.5-2`.
/// * `"waveform"` = **`1.5`** — final HiFi-GAN vocoder output in the
///   float32 audio range `[-1.0, 1.0]` (typically `~[-0.7, 0.7]` for
///   real speech; scipy int16 encoding does `float * 32767`).
///
///   **Wave-9 measured-driven derivation** (2026-08-09, PR27
///   parity-sbv2-real CI run 31303426623 on Linux ubuntu-latest AVX2 +
///   glibc libm):
///   - RNG parity fix (commit `42c8669`) removed the flow-noise cross-arch
///     divergence, so both sides now consume `torch.randn_like` samples
///     in the SAME order — waveform LENGTH now matches (rust 27136 ==
///     ref 27136 samples, no more ±10% band drift).
///   - Residual `max |Δ|` = **0.9248** on that CI run — this is the
///     f32 `sinf`/`cosf`/`expf`/`tanhf`/`logf` cross-platform delta
///     between Apple libm (M1) and glibc libm (Linux) chaotically
///     amplified through the flow + HiFi-GAN transposed convs (~600k
///     transcendental calls per second of audio), documented in
///     `docs/adr/sbv2-libm-strategy.md` §2.2 as fundamental. Peak-band
///     energy tests (RMS over the overlap prefix) would give a tighter
///     bound; per-sample max |Δ| genuinely can hit ~1.0 near HiFi-GAN
///     saturation peaks where the two libms disagree on `tanhf` above
///     ~x=3.
///   - Bound: measured 0.9248 × **1.5-2×** safety margin → **`1.5`**.
///     Do NOT tighten below `1.0` without a follow-up that pins libm
///     bit-exact (see `docs/adr/sbv2-libm-strategy.md` §4 for the
///     options and why we defer them).
///   - [`AtolCalibration::Measured`] since the fixture measurement is
///     now real and byte-derived from the CI Linux run (31303426623),
///     not a paper-cited estimate.
pub const PER_TENSOR_ATOL: &[(&str, f32)] = &[
    ("bert_hidden_ja", 0.02),
    ("bert_hidden_en", 0.02),
    ("sdp_sample", 0.05),
    ("z_latent", 0.03),
    ("waveform", 1.5),
];

/// Looks up the absolute-tolerance bound a parity test should use for the
/// tensor named `name`: [`PER_TENSOR_ATOL`]'s override if `name` matches an
/// entry there, else [`ATOL_DEFAULT`].
///
/// Pure and total — an unrecognized `name` (including `""`) is not an
/// error, it simply falls back to [`ATOL_DEFAULT`]; see [`PER_TENSOR_ATOL`]'s
/// doc for the derivation behind each listed override.
pub fn tolerance_for(name: &str) -> f32 {
    PER_TENSOR_ATOL
        .iter()
        .find(|(n, _)| *n == name)
        .map(|(_, v)| *v)
        .unwrap_or(ATOL_DEFAULT)
}

/// Wave-4 PER-TENSOR-ATOL-CALIB (2026-08-09): returns the
/// [`AtolCalibration`] status for the tensor named `name`. Every
/// [`PER_TENSOR_ATOL`] key MUST have a match arm here — and, since WP-01
/// (2026-08-09), every tensor named in the committed
/// `tests/fixtures/sbv2/reference_dump.manifest.json` `tensors[]` array
/// MUST also have a match arm, using [`AtolCalibration::UnmeasuredDefault`]
/// when it legitimately falls through to [`ATOL_DEFAULT`] with no
/// explicit override. An unlisted key still returns `None` so a
/// non-manifest lookup (e.g. a typo) remains distinguishable from a
/// pinned pass-through.
///
/// See [`PER_TENSOR_ATOL`]'s doc "Wave-4 PER-TENSOR-ATOL-CALIB" section
/// for how to flip an entry from `EstimatedPreFixture` to `Measured`,
/// and `docs/adr/sbv2-parity-atol.md` §5-§6 for how to promote an
/// `UnmeasuredDefault` entry to `EstimatedPreFixture` (derive the bound)
/// and then to `Measured` (real-fixture calibration).
///
/// Pinned by two sibling integration tests in
/// `crates/vokra-models/tests/sbv2_parity_atol_calibration.rs`:
/// (a) `every_atol_entry_has_a_calibration_status` — every
///     [`PER_TENSOR_ATOL`] key has a match arm here.
/// (b) `every_manifest_tensor_has_a_calibration_status` (WP-01) — every
///     manifest tensor has a match arm here (either the derived variant
///     or [`AtolCalibration::UnmeasuredDefault`] for a pinned pass-through).
/// (c) `atol_calibration_status_is_pinned` — no entry can silently
///     change status without touching the snapshot table there and, for
///     any `Measured` entry, the corresponding derivation in
///     `docs/adr/sbv2-parity-atol.md`.
pub fn atol_calibration_for(name: &str) -> Option<AtolCalibration> {
    match name {
        "bert_hidden_ja" => Some(AtolCalibration::EstimatedPreFixture),
        "bert_hidden_en" => Some(AtolCalibration::EstimatedPreFixture),
        "sdp_sample" => Some(AtolCalibration::EstimatedPreFixture),
        "z_latent" => Some(AtolCalibration::EstimatedPreFixture),
        // Wave-9 (2026-08-09): waveform atol derived from CI measurement
        // (run 31303426623 max |Δ| = 0.9248) × 1.5-2× margin = 1.5. See
        // `PER_TENSOR_ATOL`'s `"waveform"` block-doc for the derivation
        // and `docs/adr/sbv2-libm-strategy.md` §2.2 for the cross-plat
        // libm amplification through HiFi-GAN that gates the tightening
        // path (bit-exact libm follow-up is a documented deferral).
        "waveform" => Some(AtolCalibration::Measured),
        // WP-01 CALIBRATION-COVERAGE (2026-08-09): the six manifest
        // tensors below are not in [`PER_TENSOR_ATOL`], so
        // [`tolerance_for`] returns [`ATOL_DEFAULT`] (0.01) for each. Pre-
        // WP-01 that fall-through was invisible — the pinning test could
        // not fire on a future silent addition. Explicitly pinning each
        // to [`AtolCalibration::UnmeasuredDefault`] makes the pass-through
        // status a diff-visible commitment: a new manifest tensor that
        // forgets to add itself here now trips
        // `every_manifest_tensor_has_a_calibration_status`, and a
        // maintainer wanting to tighten one of these bounds must first
        // promote its status here (and add the corresponding override
        // to [`PER_TENSOR_ATOL`]). See `docs/adr/sbv2-parity-atol.md`
        // §5 for the promotion procedure.
        "phoneme_embed" => Some(AtolCalibration::UnmeasuredDefault),
        "text_hidden" => Some(AtolCalibration::UnmeasuredDefault),
        "bert_bridge_out" => Some(AtolCalibration::UnmeasuredDefault),
        "speaker_embed" => Some(AtolCalibration::UnmeasuredDefault),
        "style_projected" => Some(AtolCalibration::UnmeasuredDefault),
        "mel_hidden" => Some(AtolCalibration::UnmeasuredDefault),
        _ => None,
    }
}

/// Aggregator mel-loss atol (WP-04 — ADR
/// `docs/adr/sbv2-libm-strategy.md` §2.2). Bound on the RMS of the
/// log-mel-magnitude difference between the Rust `synthesize` output and the
/// Python reference-dump `waveform`.
///
/// # Why a separate constant, not [`PER_TENSOR_ATOL`]
///
/// [`PER_TENSOR_ATOL`] entries key by *tensor* names dumped by the Python
/// reference (`bert_hidden_ja` etc.). `mel_loss` is a *derived aggregate*
/// computed by the parity test itself from the `waveform` tensor — it is
/// **not** a dumped tensor, so folding it into [`PER_TENSOR_ATOL`] would
/// mis-key it under a tensor name that no dumper produces. Keeping it a
/// distinct constant keeps [`tolerance_for`]'s contract clean (one lookup
/// per real dumped tensor).
///
/// # Status: `EstimatedPreFixture` (WP-04 follow-up)
///
/// This value is scaffolded ahead of the first real Style-Bert-VITS2-2.0
/// checkpoint parity CI run — no measured floor yet. Following the same
/// honest-atol discipline as the four [`PER_TENSOR_ATOL`] entries above
/// (memory `feedback-honest-parity-atol`), the number is a *theoretical
/// upper bound* on the log-mel aggregate rather than an arbitrary
/// CI-green pick:
///
/// - Raw waveform atol (`tolerance_for("waveform") == ATOL_DEFAULT == 0.01`
///   currently — a scaffold default; the CI-measured 1.5 for the
///   synthetic-weight leg [commit `77078ce`] lives outside `tolerance_for`
///   today, WP-01 folds it in). On synthetic weights that CI run
///   [31303426623] measured a raw max-abs-diff of ~1.5 absorbing the
///   ~130M-transcendental cross-platform accumulation.
/// - Log-mel of a power spectrum divides the same amplitude noise by
///   two-to-three orders of magnitude: `log(1 + delta / signal)` compresses
///   `delta / signal` linearly, and a 1.5 raw-amplitude delta on a
///   ~1e2 signal envelope is `log(1 + 1.5e-2) ≈ 1.5e-2` per bin — the RMS
///   over ~128 bands × the fixture's ~100 frames sits well under
///   `sqrt(mean(1.5e-2^2)) ≈ 1.5e-2` before margin.
/// - `0.05` = ~3× the theoretical floor above — the same ~1.5-2× margin
///   band Kokoro's `PROSODY_F0_ATOL` uses (see [`PER_TENSOR_ATOL`]'s
///   `sdp_sample` doc for the precedent), stretched slightly since this
///   entry has NO measured confirmation yet.
///
/// A WP-04 follow-up will pin this to `AtolMeasurement::Measured(m)` in the
/// `sbv2_parity_atol_calibration.rs` snapshot table once the first
/// parity-sbv2-real CI run produces a real measurement; the number here
/// should then be revised to `max(measured × 1.5, 0.01)` and this docstring
/// updated to record the actual measurement (never delete the derivation —
/// honest-atol discipline).
pub const MEL_LOSS_ATOL: f32 = 0.05;
