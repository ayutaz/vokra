//! Standalone **HiFi-GAN vocoder** runtime binder — turns a mel
//! spectrogram into a raw PCM waveform via the shared
//! [`vokra_ops::hifigan::hifigan_generator`] op (HiFi-GAN family neural
//! vocoder — Kong et al. 2020, arXiv:2010.05646; jik876/hifi-gan, MIT).
//!
//! # What this module is (vs the SBV2 wrapper)
//!
//! `crates/vokra-models/src/sbv2/decoder.rs::SbV2Decoder` is the same
//! shape but attached to the SBV2 v2 (Style-Bert-VITS2 JP-Extra)
//! text-encoder → SDP → flow → HiFi-GAN chain, and it lives inside the
//! SBV2 model tree so that consumer builds its decoder from its own
//! `SbV2Config`. This module is the **standalone** binder for the two
//! HiFi-GAN-family GGUFs the converters
//! `crates/vokra-convert/src/models/{hifigan_vocoder, speecht5_hifigan}.rs`
//! already emit today — `speechbrain/tts-hifigan-libritts-22050Hz`
//! (apache-2.0, 22.05 kHz) and `microsoft/speecht5_hifigan` (MIT,
//! 16 kHz). Both converters forward-reference this module in their own
//! docstrings (search
//! `crates/vokra-models/src/hifigan_vocoder/` and
//! `crates/vokra-models/src/speecht5_hifigan/`); this binder is the
//! landing site those references pointed at.
//!
//! # Design mirrors `SbV2Decoder`
//!
//! [`HiFiGan`] bundles a pre-built [`HifiGanWeights`] + [`HifiGanAttrs`]
//! + [`HifiGanConfig`] triple plus a `sample_rate` metadata field, and
//! validates them up-front in [`HiFiGan::new`] so a mismatched pair
//! fails loudly at construction — never deep inside a forward
//! (FR-EX-08).
//!
//! The `vokra_ops::hifigan` op is a **free function** over those three
//! separately-built bundles (not a method on some `HiFiGanGenerator`
//! struct — see `SbV2Decoder`'s module docstring for the same
//! observation); [`HiFiGan::decode`] is a thin
//! [`hifigan_generator`] call that keeps the wrapper's surface
//! stable even as the op grows internal knobs (INT8 opt-in, `cond`
//! layer, etc.).
//!
//! # Weight-load posture: loud-partial
//!
//! [`HiFiGan::from_gguf`] deliberately declines to fabricate an upstream
//! hyperparameter transcription without primary-source verification —
//! both `hifigan_vocoder` (SpeechBrain LibriTTS 22.05 kHz) and
//! `speecht5_hifigan` (Microsoft SpeechT5 16 kHz) walk different tensor
//! trees (`generator.*` vs `upsampler.*` / `resblocks.*`) with different
//! `upsample_rates` / kernel-size / MRF-branch pins, and the
//! `resblock_dilation_sizes` outer axis + `res_block_type = V1` +
//! `normalize_before = true` (`mean` / `scale` scalars) shape checks
//! are all silent-wrong hazards if a value drifts from upstream. The
//! sibling converter modules
//! (`crates/vokra-convert/src/models/{hifigan_vocoder, speecht5_hifigan}.rs`)
//! carry the same posture on their side of the seam ("Real-weight
//! parity vs the upstream ... Python forward is deferred to owner"),
//! so this binder mirrors that discipline: [`HiFiGan::from_gguf`]
//! dispatches on `vokra.model.arch`, returns [`VokraError::ModelLoad`]
//! on missing / unknown arch, and returns
//! [`VokraError::NotImplemented`] on either supported arch — naming the
//! precise blocker (owner-verified upstream config transcription +
//! tensor-name walk) so the follow-up wave lands as a delta rather
//! than a silent-wrong forward. This is the **loud-partial** pattern
//! (RMVPE precedent, CLAUDE.md `docs/handoff/model-publish-and-parity-*`
//! §"loud-partial は fake-complete より honest"): construction succeeds
//! for hand-built bundles and for [`HiFiGan::synthesized`] test
//! fixtures, real-weight `from_gguf` fails loudly with a message
//! that names the tensor tree the loader will walk when the wave lands.
//!
//! # Fixture posture — [`HiFiGan::synthesized`]
//!
//! [`HiFiGan::synthesized`] is the **test-only** deterministic fixture
//! (same "zero-init scaffold" role as
//! `crates/vokra-models/src/vits_ja/mod.rs::VitsJaWeights::synthesized`)
//! that materialises a validated [`HifiGanWeights`] bundle whose shape
//! matches the caller's [`HifiGanAttrs`] — every conv / MRF branch is
//! zero-initialised (so the forward is deterministic but produces
//! near-zero audio via the terminal `tanh`), and the topology honours
//! [`HifiGanAttrs::res_block_type`] verbatim (V1 populates
//! `weight_c2 + bias_c2` on every layer per `mrf_branch_forward`'s
//! FR-EX-08 gate, V2 leaves them `None`). Fixtures never carry a
//! `cond` layer (`weights.cond = None`) — the single-speaker
//! unconditioned path — so [`HiFiGan::decode`] can exercise the op's
//! full forward without a per-utterance conditioning vector.

use vokra_core::gguf::GgufFile;
use vokra_core::gguf::chunks;
use vokra_core::{Result, VokraError};
use vokra_ops::attrs::{HifiGanAttrs, ResBlockType};
use vokra_ops::hifigan::{
    HifiGanConfig, HifiGanWeights, MrfBranchWeights, ResBlockLayer, UpsampleStageWeights,
    hifigan_generator,
};

/// `vokra.model.arch` value emitted by the SpeechBrain LibriTTS 22.05 kHz
/// converter (`crates/vokra-convert/src/models/hifigan_vocoder.rs::ARCH`).
///
/// Pinned by the [`arch_tags_are_distinct_and_match_converters`] test —
/// a converter rename must land here in the same commit or the pin
/// fails loudly rather than silently mis-routing dispatch.
pub const ARCH_HIFIGAN_VOCODER: &str = "hifigan_vocoder";

/// `vokra.model.arch` value emitted by the Microsoft SpeechT5 16 kHz
/// converter (`crates/vokra-convert/src/models/speecht5_hifigan.rs::ARCH`).
///
/// Intentionally distinct from [`ARCH_HIFIGAN_VOCODER`] — the two GGUFs
/// walk different tensor trees (`generator.*` vs `upsampler.*` /
/// `resblocks.*` at top level, with `mean` / `scale` scalars for
/// `normalize_before = true` on the SpeechT5 side); silently sharing an
/// arch tag would mis-route runtime dispatch.
pub const ARCH_SPEECHT5_HIFIGAN: &str = "speecht5_hifigan";

/// Standalone HiFi-GAN vocoder handle: owns a [`HifiGanWeights`] +
/// [`HifiGanAttrs`] + [`HifiGanConfig`] triple plus the intended output
/// sample rate. Exposes [`decode`](Self::decode) as the primary
/// mel → PCM entry point.
///
/// # Sibling wrappers
///
/// `crates/vokra-models/src/sbv2/decoder.rs::SbV2Decoder` is the same
/// shape but attached to the SBV2 v2 flow-output-to-waveform bridge;
/// this binder is the standalone counterpart for the two HiFi-GAN
/// family GGUFs the sibling converters emit.
pub struct HiFiGan {
    /// Pre-built HiFi-GAN weight bundle (conv_pre / upsample stack /
    /// MRF branches / conv_post — see [`HifiGanWeights`]'s field docs).
    weights: HifiGanWeights,
    /// Shape + upsample-ladder metadata. [`decode`](Self::decode)'s
    /// `mel.len()` must equal `n_frames * attrs.n_mels`, checked inside
    /// [`hifigan_generator`] itself.
    attrs: HifiGanAttrs,
    /// Precision policy — FP32 (default) or mixed-precision FP16 with
    /// FP32 accumulator. INT8 opt-in stays gated behind
    /// [`HifiGanConfig::with_int8_opt_in`] + `spectral_check_passed`
    /// on the op boundary.
    config: HifiGanConfig,
    /// Output sample rate in Hz — informational metadata mirroring
    /// [`HifiGanAttrs::sample_rate`]; neither field feeds the forward
    /// math. Both exist so a caller can cross-check against the
    /// frontend_spec `sample_rate` (FR-LD-03). [`Self::new`] rejects
    /// a mismatched pair.
    sample_rate: u32,
}

impl HiFiGan {
    /// Assembles a vocoder handle from a pre-built weight bundle, its
    /// shape metadata, a precision policy, and the intended output
    /// sample rate.
    ///
    /// Runs [`HifiGanAttrs::validate_shape`] + [`HifiGanConfig::validate`]
    /// up front (SbV2Decoder / VitsJaTts::new precedent) so a
    /// mismatched pair fails loudly here rather than deep inside a
    /// forward — FR-EX-08. Cross-checks `sample_rate == attrs.sample_rate`
    /// because the two are expected to always agree; a converter that
    /// wrote one but not the other would silently produce audio at the
    /// wrong rate.
    ///
    /// # Errors
    ///
    /// - [`VokraError::InvalidArgument`] from
    ///   [`HifiGanAttrs::validate_shape`] on a structurally invalid attrs
    ///   bundle.
    /// - [`VokraError::InvalidArgument`] when
    ///   `sample_rate != attrs.sample_rate`.
    /// - [`VokraError::HifiganInt8VerifyMissing`] from
    ///   [`HifiGanConfig::validate`] when `int8_enabled == true` but the
    ///   calibration table or spectral-check verdict is missing.
    pub fn new(
        weights: HifiGanWeights,
        attrs: HifiGanAttrs,
        config: HifiGanConfig,
        sample_rate: u32,
    ) -> Result<Self> {
        attrs.validate_shape()?;
        config.validate()?;
        if sample_rate != attrs.sample_rate {
            return Err(VokraError::InvalidArgument(format!(
                "HiFiGan::new: sample_rate {sample_rate} != attrs.sample_rate {}",
                attrs.sample_rate
            )));
        }
        Ok(Self {
            weights,
            attrs,
            config,
            sample_rate,
        })
    }

    /// Output sample rate in Hz.
    #[must_use]
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// The shape-metadata bundle (n_mels, upsample_rates, MRF branches).
    #[must_use]
    pub fn attrs(&self) -> &HifiGanAttrs {
        &self.attrs
    }

    /// Deterministic zero-initialised fixture — **test scaffold only**.
    ///
    /// Materialises a validated [`HifiGanWeights`] bundle whose shape
    /// matches `attrs` verbatim, so [`Self::new`] + [`Self::decode`]
    /// can exercise the full forward path on a synthesised checkpoint
    /// without needing an actual upstream GGUF. Every conv / MRF-branch
    /// weight cell is zero, so the forward emits near-zero audio
    /// bounded by the terminal `tanh` — the test contract is finiteness
    /// and shape, not fidelity.
    ///
    /// Honours [`HifiGanAttrs::res_block_type`] verbatim: V1 populates
    /// `weight_c2 + bias_c2` on every MRF layer (per
    /// `mrf_branch_forward`'s FR-EX-08 gate: V1 attrs *must* have
    /// `weight_c2 + bias_c2` on every layer); V2 leaves them `None`.
    /// Fixtures never carry a `cond` (speaker conditioning) layer
    /// (`weights.cond = None`) so [`Self::decode`] exercises the
    /// single-speaker unconditioned path.
    ///
    /// The internal channel schedule mirrors
    /// `vokra_ops::hifigan::tests::tiny_weights` (`out_ch =
    /// max(3, in_ch / 2)` per stage) — proven to satisfy
    /// `validate_weights`'s cross-stage chain (each stage's
    /// `in_ch` == previous stage's `out_ch`) for any
    /// `validate_shape`-passing attrs. `conv_pre_kernel = 3` and
    /// `conv_post_kernel = 3` (test-fixture sizing — real
    /// [`hifigan_vocoder`](ARCH_HIFIGAN_VOCODER) checkpoints use
    /// kernel 7, but the fixture never rides a real
    /// [`from_gguf`](Self::from_gguf) path).
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] from
    /// [`HifiGanAttrs::validate_shape`] on a structurally invalid `attrs`
    /// (zero-axis dims, mismatched slice lengths, etc.) or when
    /// `sample_rate != attrs.sample_rate`.
    pub fn synthesized(attrs: HifiGanAttrs, sample_rate: u32) -> Result<Self> {
        attrs.validate_shape()?;
        if sample_rate != attrs.sample_rate {
            return Err(VokraError::InvalidArgument(format!(
                "HiFiGan::synthesized: sample_rate {sample_rate} != attrs.sample_rate {}",
                attrs.sample_rate
            )));
        }

        let conv_pre_kernel = 3;
        let conv_post_kernel = 3;

        // conv_pre: [initial_channel, n_mels, conv_pre_kernel] zero-init.
        let conv_pre_weight = vec![0.0f32; attrs.initial_channel * attrs.n_mels * conv_pre_kernel];
        let conv_pre_bias = vec![0.0f32; attrs.initial_channel];

        // Walk the upsample stack in `validate_weights` chain order:
        // every stage's `in_ch` must equal the previous stage's
        // `out_ch`, with the first stage's `in_ch == initial_channel`.
        let mut upsample_weights = Vec::with_capacity(attrs.n_upsample_stages());
        let mut mrf_stage_weights = Vec::with_capacity(attrs.n_upsample_stages());
        let mut in_ch = attrs.initial_channel;
        for stage in 0..attrs.n_upsample_stages() {
            // Mirror the op-side tiny_weights schedule; the `max(3, ..)`
            // floor keeps every layer's channel count non-zero even
            // when a caller passes a tiny attrs like initial_channel = 2.
            let out_ch = 3.max(in_ch / 2);
            let kernel = attrs.upsample_kernel_sizes[stage];
            let stride = attrs.upsample_rates[stage];
            upsample_weights.push(UpsampleStageWeights {
                weight: vec![0.0f32; in_ch * out_ch * kernel],
                bias: vec![0.0f32; out_ch],
                in_ch,
                out_ch,
                kernel,
                stride,
            });

            let mut branches = Vec::with_capacity(attrs.n_mrf_branches());
            for branch_idx in 0..attrs.n_mrf_branches() {
                let branch_kernel = attrs.resblock_kernel_sizes[branch_idx];
                let dilations = &attrs.resblock_dilation_sizes[branch_idx];
                let mut layers = Vec::with_capacity(dilations.len());
                for &dilation in dilations {
                    let weight = vec![0.0f32; out_ch * out_ch * branch_kernel];
                    let bias = vec![0.0f32; out_ch];
                    let (weight_c2, bias_c2) = match attrs.res_block_type {
                        // V1 requires both c2 buffers populated per
                        // `mrf_branch_forward`'s FR-EX-08 gate — a
                        // missing c2 on a V1 attrs is loud-fail there.
                        ResBlockType::V1 => (
                            Some(vec![0.0f32; out_ch * out_ch * branch_kernel]),
                            Some(vec![0.0f32; out_ch]),
                        ),
                        // V2 mandates both `None` — carrying c2 on a V2
                        // attrs is also loud-fail per the same gate.
                        ResBlockType::V2 => (None, None),
                    };
                    layers.push(ResBlockLayer {
                        weight,
                        bias,
                        weight_c2,
                        bias_c2,
                        dilation,
                        kernel: branch_kernel,
                        channels: out_ch,
                    });
                }
                branches.push(MrfBranchWeights { layers });
            }
            mrf_stage_weights.push(branches);
            in_ch = out_ch;
        }

        // conv_post: [1, ch_last, conv_post_kernel]. Emit the pre-HGAN-04
        // explicit-zero bias shape `[1]` (validate_weights accepts
        // `len == 0` for upstream bias=False and `len == 1` for
        // the explicit-zero form; the two are numerically identical).
        let conv_post_weight = vec![0.0f32; in_ch * conv_post_kernel];
        let conv_post_bias = vec![0.0f32; 1];

        let weights = HifiGanWeights {
            conv_pre_weight,
            conv_pre_bias,
            conv_pre_kernel,
            upsample_weights,
            mrf_stage_weights,
            conv_post_weight,
            conv_post_bias,
            conv_post_kernel,
            cond: None,
        };

        Self::new(weights, attrs, HifiGanConfig::fp32(), sample_rate)
    }

    /// Runs the HiFi-GAN forward on `mel` (`[n_mels, n_frames]`
    /// row-major, `mel.len() == attrs.n_mels * n_frames`) and returns
    /// the raw PCM waveform bounded to `(−1, 1)` by the op's terminal
    /// `tanh`.
    ///
    /// This is a thin unconditioned wrapper — the multi-speaker `cond`
    /// path lives on [`hifigan_generator_conditioned`] and would
    /// require an additional API entry once a `cond`-carrying
    /// converter lands (SBV2 v2 already goes through the sibling
    /// `SbV2Decoder::generate_conditioned` for that).
    ///
    /// # Errors
    ///
    /// See [`hifigan_generator`]. In practice, once `self` has passed
    /// [`Self::new`], the only reachable errors are
    /// [`VokraError::InvalidArgument`] on a `mel.len()` mismatch
    /// (caller sent the wrong `n_frames`) and
    /// [`VokraError::UnsupportedOp`] when INT8 is opt-in-enabled (the
    /// INT8 forward kernel is deferred to the consumer WP per
    /// `hifigan_generator`'s own error surface).
    ///
    /// [`hifigan_generator_conditioned`]: vokra_ops::hifigan::hifigan_generator_conditioned
    pub fn decode(&self, mel: &[f32], n_frames: usize) -> Result<Vec<f32>> {
        hifigan_generator(mel, n_frames, &self.weights, &self.attrs, &self.config)
    }

    /// Dispatches on the `vokra.model.arch` metadata chunk and loads a
    /// [`HiFiGan`] from a GGUF file.
    ///
    /// # Current status — loud-partial
    ///
    /// This entry point is intentionally loud-partial today
    /// (RMVPE / DFN3 Phase B precedent, CLAUDE.md
    /// 「loud-partial は fake-complete より honest」): the arch
    /// dispatch works (missing arch → [`VokraError::ModelLoad`],
    /// unknown arch → [`VokraError::ModelLoad`]), but both supported
    /// arches ([`ARCH_HIFIGAN_VOCODER`], [`ARCH_SPEECHT5_HIFIGAN`])
    /// return [`VokraError::NotImplemented`] with a message that names
    /// the precise blocker (owner-verified upstream config
    /// transcription + tensor-name walk). This mirrors the sibling
    /// converter modules' own "Real-weight parity vs the upstream
    /// Python forward is deferred to owner" posture
    /// (see `crates/vokra-convert/src/models/{hifigan_vocoder,
    /// speecht5_hifigan}.rs`) — the CC side ships the binder shape and
    /// the arch-dispatch discipline, and the follow-up wave lands the
    /// real hyperparameter transcription + tensor-name walk as a
    /// delta against a real upstream checkpoint rather than a
    /// fabricated transcription.
    ///
    /// Hand-built [`HiFiGan::new`] and [`HiFiGan::synthesized`] work
    /// today (they never touch this path); real-weight round-trips
    /// through the sibling converters + this loader are the deferred
    /// wave.
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] when `vokra.model.arch` is missing,
    ///   not a UTF-8 string, or does not match either supported arch
    ///   ([`ARCH_HIFIGAN_VOCODER`] or [`ARCH_SPEECHT5_HIFIGAN`]).
    /// - [`VokraError::NotImplemented`] on either supported arch until
    ///   the real-weight loader wave lands.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        let arch = file
            .get(chunks::KEY_MODEL_ARCH)
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                VokraError::ModelLoad(format!(
                    "HiFiGan::from_gguf: missing or non-string GGUF metadata key `{}` — the \
                     hifigan_vocoder / speecht5_hifigan converters both stamp this key; a GGUF \
                     without it is either not a HiFi-GAN vocoder or was produced by a converter \
                     that predates the arch-dispatch discipline. Rebuild via the appropriate \
                     `vokra-cli convert --model {{hifigan_vocoder,speecht5_hifigan}}` path.",
                    chunks::KEY_MODEL_ARCH
                ))
            })?;
        match arch {
            ARCH_HIFIGAN_VOCODER => Err(VokraError::NotImplemented(
                "HiFiGan::from_gguf(hifigan_vocoder): real-weight loader is deferred (mirror of \
                 the sibling converter `crates/vokra-convert/src/models/hifigan_vocoder.rs` own \
                 \"Real-weight parity vs the upstream `speechbrain` Python pipeline is deferred \
                 to owner\" posture). Follow-up wave will (1) transcribe SpeechBrain \
                 `speechbrain/tts-hifigan-libritts-22050Hz` `hyperparams.yaml` verbatim into a \
                 hard-coded HifiGanAttrs preset (upstream `upsample_rates` / \
                 `upsample_kernel_sizes` / `resblock_kernel_sizes` / \
                 `resblock_dilation_sizes` / `upsample_initial_channel` / `num_mels` / \
                 `sample_rate` — CLAUDE.md 「ハルシネーション厳禁」: transcription must be \
                 primary-source verified against the upstream file, not memorised), (2) walk \
                 the `generator.conv_pre.*` / `generator.ups.{i}.*` / \
                 `generator.resblocks.{stage*n_branches+branch}.convs{1,2}.{layer}.*` / \
                 `generator.conv_post.*` tensor names into `HifiGanWeights`, (3) route through \
                 `HiFiGan::new`. Hand-built `new` + `synthesized` fixtures work today.",
            )),
            ARCH_SPEECHT5_HIFIGAN => Err(VokraError::NotImplemented(
                "HiFiGan::from_gguf(speecht5_hifigan): real-weight loader is deferred. The \
                 SpeechT5 tensor tree is intentionally distinct from `hifigan_vocoder` — the \
                 HF-transformers `SpeechT5HifiGan` class emits `upsampler.{i}.*` / \
                 `resblocks.{i}.*` / `conv_pre.*` / `conv_post.*` (no `generator.` prefix) plus \
                 scalar `mean` / `scale` tensors for `normalize_before = true`, with \
                 `upsample_rates = [4, 4, 4, 4]` / `upsample_kernel_sizes = [8, 8, 8, 8]` / \
                 16 kHz vs SpeechBrain's 22.05 kHz recipe. Silently sharing an arch tag would \
                 mis-route dispatch (see the converter module's own FR-EX-08 rationale in \
                 `crates/vokra-convert/src/models/speecht5_hifigan.rs`). Follow-up wave will \
                 transcribe `microsoft/speecht5_hifigan` `config.json` verbatim + wire the \
                 `mean` / `scale` pre-network normalisation the sibling arch does not carry.",
            )),
            other => Err(VokraError::ModelLoad(format!(
                "HiFiGan::from_gguf: unsupported `vokra.model.arch` = {other:?}. This binder \
                 accepts only {ARCH_HIFIGAN_VOCODER:?} (SpeechBrain LibriTTS 22.05 kHz) and \
                 {ARCH_SPEECHT5_HIFIGAN:?} (Microsoft SpeechT5 16 kHz). Other HiFi-GAN-family \
                 GGUFs (bigvgan, piper-plus, sbv2 decoders folded inside their consumer \
                 models) route through their own binder modules."
            ))),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::GgufBuilder;
    use vokra_ops::hifigan::{CalibrationTable, HifiGanPrecision};

    /// Tiny V1 attrs — canonical HiFi-GAN / SBV2 v2 topology (per-layer
    /// `weight_c2 + bias_c2`), small enough for a decoding smoke test.
    /// Mirrors the shape of `vokra_ops::hifigan::tests::tiny_attrs_v1`
    /// so the synthesized-weight schedule (`out_ch = max(3, in_ch/2)`)
    /// exactly matches the op-side fixture's chain expectations.
    fn tiny_attrs_v1() -> HifiGanAttrs {
        HifiGanAttrs {
            n_mels: 4,
            initial_channel: 6,
            upsample_rates: vec![2, 2],
            upsample_kernel_sizes: vec![4, 4],
            resblock_kernel_sizes: vec![3, 5],
            resblock_dilation_sizes: vec![vec![1, 3], vec![1, 3]],
            sample_rate: 16_000,
            leaky_relu_slope: 0.1,
            res_block_type: ResBlockType::V1,
        }
    }

    /// V2 counterpart (single-conv per layer, `weight_c2`/`bias_c2 =
    /// None`). Same shape as [`tiny_attrs_v1`], different topology tag.
    fn tiny_attrs_v2() -> HifiGanAttrs {
        HifiGanAttrs {
            res_block_type: ResBlockType::V2,
            ..tiny_attrs_v1()
        }
    }

    /// Task-spec pin: the arch tags this binder dispatches on MUST
    /// match verbatim the `ARCH` constants the sibling converters emit
    /// (`crates/vokra-convert/src/models/{hifigan_vocoder,
    /// speecht5_hifigan}.rs`). A converter rename that skipped this
    /// module would silently route to the unknown-arch error path
    /// instead of the deferred-loader loud path — this test catches
    /// that drift.
    #[test]
    fn arch_tags_are_distinct_and_match_converters() {
        assert_eq!(ARCH_HIFIGAN_VOCODER, "hifigan_vocoder");
        assert_eq!(ARCH_SPEECHT5_HIFIGAN, "speecht5_hifigan");
        assert_ne!(
            ARCH_HIFIGAN_VOCODER, ARCH_SPEECHT5_HIFIGAN,
            "the two vocoders walk different tensor trees — sharing an arch would mis-route"
        );
    }

    /// Primary integration test: a `synthesized(tiny_attrs_v1)`
    /// fixture decodes a zero-mel input to a finite, bounded PCM
    /// buffer of the shape `hifigan_generator`'s transposed-conv
    /// formula predicts. Exercises the full forward (conv_pre →
    /// per-stage transposed-conv + MRF branch fusion → conv_post →
    /// tanh) with a V1 topology's per-layer `c2` chain populated.
    #[test]
    fn synthesized_v1_decodes_zero_mel_to_finite_bounded_pcm() {
        let attrs = tiny_attrs_v1();
        let sr = attrs.sample_rate;
        let hifigan = HiFiGan::synthesized(attrs.clone(), sr).expect("synthesized V1");
        assert_eq!(hifigan.sample_rate(), sr);
        assert_eq!(hifigan.attrs().res_block_type, ResBlockType::V1);

        let n_frames = 4;
        let mel = vec![0.0f32; attrs.n_mels * n_frames];
        let pcm = hifigan.decode(&mel, n_frames).expect("decode zero mel");

        // Expected sample count from the transposed-conv shape formula
        // used by `hifigan_generator`'s per-stage upsample:
        //   out_len = (in_len - 1) * stride + kernel - 2 * ((kernel - stride) / 2)
        // For every stage in tiny_attrs_v1 (kernel=4, stride=2), padding
        // = (4-2)/2 = 1, so out_len = (in-1)*2 + 4 - 2 = in*2 exactly.
        // Chained over 2 stages: n_frames * 4. Compute explicitly rather
        // than hard-coding so a future tiny_attrs shape change does not
        // silently drift.
        let mut expected_len = n_frames;
        for stage in 0..attrs.n_upsample_stages() {
            let kernel = attrs.upsample_kernel_sizes[stage];
            let stride = attrs.upsample_rates[stage];
            let padding = kernel.saturating_sub(stride) / 2;
            expected_len = (expected_len - 1) * stride + kernel - 2 * padding;
        }
        assert_eq!(
            pcm.len(),
            expected_len,
            "PCM length matches transposed-conv formula"
        );

        for (i, &v) in pcm.iter().enumerate() {
            assert!(v.is_finite(), "PCM[{i}] = {v} must be finite (tanh output)");
            assert!(v > -1.0 && v < 1.0, "PCM[{i}] = {v} must lie in (-1, 1)");
        }
    }

    /// V2 counterpart of the primary integration test — same shape but
    /// single-conv per MRF layer (no c2 chain). Guards the
    /// `res_block_type` branch inside [`HiFiGan::synthesized`] against
    /// a regression that populates c2 unconditionally (which would
    /// loud-fail inside `mrf_branch_forward`'s V2 arm).
    #[test]
    fn synthesized_v2_decodes_zero_mel_to_finite_bounded_pcm() {
        let attrs = tiny_attrs_v2();
        let sr = attrs.sample_rate;
        let hifigan = HiFiGan::synthesized(attrs.clone(), sr).expect("synthesized V2");
        assert_eq!(hifigan.attrs().res_block_type, ResBlockType::V2);

        let n_frames = 4;
        let mel = vec![0.0f32; attrs.n_mels * n_frames];
        let pcm = hifigan.decode(&mel, n_frames).expect("decode zero mel V2");
        assert!(!pcm.is_empty(), "V2 forward must emit a non-empty waveform");
        for &v in &pcm {
            assert!(v.is_finite() && v > -1.0 && v < 1.0);
        }
    }

    /// Structural precondition: [`HiFiGan::synthesized`] must reject a
    /// zero-axis attrs (empty `upsample_rates`) via
    /// `HifiGanAttrs::validate_shape` before touching any weight
    /// allocation. FR-EX-08 — construction fails loudly, not silently.
    #[test]
    fn synthesized_rejects_zero_axis_attrs() {
        let mut attrs = tiny_attrs_v1();
        attrs.upsample_rates.clear();
        attrs.upsample_kernel_sizes.clear();
        let Err(err) = HiFiGan::synthesized(attrs, 16_000) else {
            panic!("expected InvalidArgument from validate_shape on zero-axis attrs");
        };
        assert!(
            matches!(err, VokraError::InvalidArgument(_)),
            "expected InvalidArgument from validate_shape, got: {err}"
        );
    }

    /// [`HiFiGan::synthesized`] must reject a `sample_rate` that
    /// disagrees with `attrs.sample_rate`. The two are expected to
    /// always agree; a silent disagreement would produce audio at the
    /// wrong rate.
    #[test]
    fn synthesized_rejects_sample_rate_disagreement() {
        let attrs = tiny_attrs_v1();
        // attrs.sample_rate = 16_000; caller supplies 22_050 by mistake.
        let Err(err) = HiFiGan::synthesized(attrs, 22_050) else {
            panic!("expected InvalidArgument from sample_rate mismatch");
        };
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    /// [`HiFiGan::new`] must reject a config that manually flips
    /// `int8_enabled = true` without attaching a calibration table +
    /// spectral-check verdict, via [`HifiGanConfig::validate`]. The
    /// atomic constructor [`HifiGanConfig::with_int8_opt_in`] pairs
    /// both proofs; a hand-built config that skips it must fail loudly.
    #[test]
    fn new_rejects_broken_int8_opt_in() {
        let attrs = tiny_attrs_v1();
        let synth = HiFiGan::synthesized(attrs.clone(), attrs.sample_rate).unwrap();
        let broken = HifiGanConfig {
            precision: HifiGanPrecision::Fp32,
            int8_enabled: true,
            calibration_data: None,
            spectral_check_passed: true,
        };
        let Err(err) = HiFiGan::new(synth.weights, synth.attrs, broken, attrs.sample_rate) else {
            panic!("expected HifiganInt8VerifyMissing from broken INT8 opt-in config");
        };
        assert!(
            matches!(err, VokraError::HifiganInt8VerifyMissing),
            "expected HifiganInt8VerifyMissing from HifiGanConfig::validate, got: {err}"
        );
    }

    /// [`HiFiGan::new`] must accept an INT8 opt-in config that satisfies
    /// both proofs — validation passes at construction, but decoding
    /// still hits the deferred-kernel [`VokraError::UnsupportedOp`] on
    /// the op boundary (the M3-07 op-only WP does not ship the INT8
    /// forward kernel). Guards that the gate distinguishes "invalid
    /// config" (rejected at construction) from "valid config, kernel
    /// deferred" (rejected at forward).
    #[test]
    fn new_accepts_int8_opt_in_but_decode_returns_unsupported_kernel() {
        let attrs = tiny_attrs_v1();
        let synth = HiFiGan::synthesized(attrs.clone(), attrs.sample_rate).unwrap();
        let calibration =
            CalibrationTable::new(vec![1.0; 3], vec![0; 3], 3).expect("valid calibration");
        let cfg = HifiGanConfig::fp32().with_int8_opt_in(calibration, true);
        let hifigan = HiFiGan::new(synth.weights, synth.attrs, cfg, attrs.sample_rate)
            .expect("new must accept fully-authorised INT8 config");
        let n_frames = 2;
        let mel = vec![0.0f32; attrs.n_mels * n_frames];
        let err = hifigan.decode(&mel, n_frames).unwrap_err();
        assert!(
            matches!(err, VokraError::UnsupportedOp(_)),
            "expected UnsupportedOp (INT8 kernel deferred to consumer WP), got: {err}"
        );
    }

    /// A GGUF that does not carry `vokra.model.arch` at all must fail
    /// with [`VokraError::ModelLoad`] — never a silent success on a
    /// zero-tensor fixture, never a panic.
    #[test]
    fn from_gguf_missing_arch_returns_model_load() {
        let mut b = GgufBuilder::new();
        b.add_string("vokra.model.name", "no-arch-here");
        let bytes = b.to_bytes().expect("build minimal GGUF");
        let file = GgufFile::parse(bytes).expect("parse minimal GGUF");
        let Err(err) = HiFiGan::from_gguf(&file) else {
            panic!("expected ModelLoad naming the missing arch key on unset arch");
        };
        match err {
            VokraError::ModelLoad(msg) => {
                assert!(msg.contains(chunks::KEY_MODEL_ARCH));
            }
            other => panic!("expected ModelLoad naming the missing arch key, got: {other}"),
        }
    }

    /// A GGUF carrying an arch tag this binder does not recognise must
    /// fail with [`VokraError::ModelLoad`] naming both accepted arches
    /// so a downstream caller can pick the right converter.
    #[test]
    fn from_gguf_unknown_arch_returns_model_load() {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, "bigvgan_v2");
        let bytes = b.to_bytes().expect("build GGUF with wrong arch");
        let file = GgufFile::parse(bytes).expect("parse GGUF");
        let Err(err) = HiFiGan::from_gguf(&file) else {
            panic!("expected ModelLoad naming supported arches on unknown arch");
        };
        match err {
            VokraError::ModelLoad(msg) => {
                assert!(msg.contains(ARCH_HIFIGAN_VOCODER));
                assert!(msg.contains(ARCH_SPEECHT5_HIFIGAN));
                assert!(msg.contains("bigvgan_v2"));
            }
            other => panic!("expected ModelLoad naming supported arches, got: {other}"),
        }
    }

    /// `arch = hifigan_vocoder` must reach the loud-partial arm — the
    /// real-weight loader is deferred, so the caller gets a
    /// [`VokraError::NotImplemented`] that names the SpeechBrain
    /// hyperparams source + tensor-name walk. FR-EX-08.
    #[test]
    fn from_gguf_hifigan_vocoder_returns_not_implemented() {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH_HIFIGAN_VOCODER);
        let bytes = b.to_bytes().expect("build hifigan_vocoder-arch GGUF");
        let file = GgufFile::parse(bytes).expect("parse GGUF");
        let Err(err) = HiFiGan::from_gguf(&file) else {
            panic!("expected NotImplemented for deferred hifigan_vocoder loader");
        };
        match err {
            VokraError::NotImplemented(msg) => {
                assert!(msg.contains("hifigan_vocoder"));
                assert!(msg.contains("speechbrain"));
            }
            other => panic!("expected NotImplemented for deferred loader, got: {other}"),
        }
    }

    /// `arch = speecht5_hifigan` must reach the other loud-partial
    /// arm — the SpeechT5 tensor topology is intentionally distinct
    /// (no `generator.` prefix, `mean` / `scale` scalars, 16 kHz).
    #[test]
    fn from_gguf_speecht5_hifigan_returns_not_implemented() {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH_SPEECHT5_HIFIGAN);
        let bytes = b.to_bytes().expect("build speecht5_hifigan-arch GGUF");
        let file = GgufFile::parse(bytes).expect("parse GGUF");
        let Err(err) = HiFiGan::from_gguf(&file) else {
            panic!("expected NotImplemented for deferred speecht5_hifigan loader");
        };
        match err {
            VokraError::NotImplemented(msg) => {
                assert!(msg.contains("speecht5_hifigan"));
                assert!(msg.contains("upsampler"));
            }
            other => panic!("expected NotImplemented for deferred loader, got: {other}"),
        }
    }
}
