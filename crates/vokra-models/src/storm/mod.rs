//! **StoRM** (`sp-uhh/storm`, MIT) — Stochastic Regeneration Model for
//! Speech Enhancement and Dereverberation runtime binder for the
//! `storm` converter arch (Wave 7 2026-08-14 audit follow-up RETRY of
//! a Wave 6 lost item — workflow silently swallowed the previous
//! result; see WAVE 6 LESSON in the directive).
//!
//! # Primary source
//!
//! - Paper: Lay et al. 2023 arXiv:2312.09386
//!   *"StoRM: A Diffusion-based Stochastic Regeneration Model for
//!   Speech Enhancement and Dereverberation"*.
//! - Reference implementation:
//!   <https://github.com/sp-uhh/storm>
//! - Weight license: **MIT** per upstream repo LICENSE
//!   (`github.com/sp-uhh/storm/blob/main/LICENSE`, per task scout
//!   input 2026-08-14 — owner must primary-source confirm at sign-off
//!   time).
//!
//! # Runtime layout (loud-partial, sepformer / conv_tasnet / demucs /
//! gtcrn separation-fleet posture per CLAUDE.md 教訓 (a))
//!
//! ```text
//! Mixture PCM (mono f32, 16 kHz per [`StormConfig::sample_rate`],
//!   typically noisy + reverberant speech)
//!   -> STFT (n_fft=510, hop=128, full complex spectrum)
//!                                                 [already covered by
//!                                                  `vokra_ops::stft`]
//!   -> initial deterministic predictive estimator ← **loud-partial**
//!        (an NCSN++ v2 U-Net variant per arXiv:2312.09386 §III trained
//!         under an MSE objective rather than the score-matching
//!         objective — same topology family as the second-stage score
//!         network but a *distinct forward*: the predictor's forward
//!         pass produces a point estimate of the clean STFT via a
//!         standard U-Net regression, no sigma conditioning.)
//!   -> NCSN++ v2 U-Net score-network              ← **loud-partial**
//!        (Noise Conditional Score Network++ v2 backbone — U-Net with
//!         attention blocks + feature-wise linear modulation (FiLM)
//!         over noise-conditioning σ, per Song et al. arXiv:2011.13456
//!         §3.3 as extended in StoRM §III. NOT covered by existing
//!         `vokra_ops` primitives — no U-Net with sigma-conditional
//!         FiLM primitive exists in the catalogue.)
//!   -> OUVE-SDE predictor-corrector sampler       ← **loud-partial**
//!        (Ornstein-Uhlenbeck Variance-Exploding stochastic
//!         differential equation predictor-corrector Langevin dynamics
//!         iterative refinement over the sigma schedule per
//!         arXiv:2312.09386 §III + Welker et al. 2022 SGMSE+
//!         Interspeech precedent. `vokra_ops::flow_sampler` covers
//!         ODE-style flow matching but NOT the SDE-style predictor-
//!         corrector Langevin dynamics StoRM requires — the two are
//!         different sampler families and cannot be silently aliased.)
//!   -> per-frame refined complex STFT
//!   -> iSTFT                                      [already covered by
//!                                                  `vokra_ops::istft_streaming`]
//!   -> denoised + dereverberated PCM
//! ```
//!
//! # Loud-partial classification (design § — CLAUDE.md 教訓 (a))
//!
//! - **Real (this WP)**: [`Storm::from_gguf`] with strict
//!   `vokra.model.arch == "storm"` validation + strict `vokra.storm.*`
//!   chunk-group presence enforcement (every axis required — no
//!   primary-source constant fallback because the converter
//!   transcribes the axes from arXiv:2312.09386 §III and stamps them,
//!   and this binder mirrors those stamps rather than silently
//!   defaulting to a fabricated axis), [`StormWeights::from_gguf`]
//!   with a floor of non-empty tensor count enforced loud (a GGUF that
//!   carries zero tensors is refused rather than silently running an
//!   all-zero forward — FR-EX-08), and weight-license class surfacing
//!   (defaults to [`LicenseClass::Unknown`] on a stamp-free fixture,
//!   fail-closed at the M2-13 compliance gate — the converter stamps
//!   [`LicenseClass::Permissive`] in production per the MIT default).
//! - **Loud-partial (this WP)**: [`Storm::enhance`] returns
//!   [`VokraError::UnsupportedOp`] naming **all three** deferred
//!   primitives + the two-stage compose:
//!   (i) **initial deterministic predictive estimator** (StoRM's
//!   first-stage regression sub-network, an NCSN++ v2 U-Net variant
//!   trained under an MSE objective — same topology family as the
//!   score network but a distinct forward);
//!   (ii) **NCSN++ v2 U-Net score-network** (Noise Conditional Score
//!   Network++ v2 backbone — U-Net with attention blocks + FiLM over
//!   sigma conditioning, NOT covered by existing `vokra_ops`);
//!   (iii) **OUVE-SDE predictor-corrector sampler** (Ornstein-Uhlenbeck
//!   Variance-Exploding SDE predictor-corrector Langevin dynamics —
//!   `vokra_ops::flow_sampler` covers ODE-style flow matching but NOT
//!   the SDE-style predictor-corrector StoRM requires); and
//!   (iv) the two-stage compose (predictor output → score refinement
//!   loop over sigma schedule).
//!   The error cites both primary sources (upstream GitHub repo README
//!   + arXiv:2312.09386 paper) and echoes every config axis so a
//!     reader diagnosing this gap has exactly two anchors to walk and
//!     knows the topology the follow-up wave targets.
//!
//! Rationale (RMVPE / pyannote / hifigan / vocos / bigvgan / snac /
//! beat_this / mt3 / redimnet / sortformer / sepformer / conv_tasnet /
//! demucs / gtcrn Wave 1-6 loud-partial precedent, CLAUDE.md 教訓 (a) —
//! "loud-partial は fake-complete より honest"): the surrounding
//! scaffold + `from_gguf` chunk-group validation + FR-EX-08 loud-fails
//! land today so a follow-up wave can flip the switch by (i) landing
//! the tensor-name walk against a real StoRM state_dict (the release
//! ships PyTorch state dicts via Google Drive that
//! `tools/parity/nemo_pt_to_safetensors.py` uv-managed Python 3.12
//! sidecar bridges to safetensors), (ii) landing the three missing
//! primitives (predictor + NCSN++ v2 score-network + OUVE-SDE
//! predictor-corrector sampler) in `vokra_ops`, and (iii) composing
//! the two-stage forward against the stamped `vokra.storm.*` axes.
//!
//! # `vokra.storm.*` chunk group (read here)
//!
//! Written by `vokra-convert::models::storm::convert_storm_file`:
//!
//! - `vokra.model.arch` (`String`): must equal [`ARCH`] (`"storm"`).
//!   Distinct from every sibling denoise / separator arch
//!   (`denoise` (DFN3), `rnnoise`, `nsnet2`, `dnsmos`,
//!   `metricgan_plus`, `mp_senet_dns`, `sepformer`, `conv_tasnet`,
//!   `demucs`, `gtcrn`, `frcrn`, `mossformer2_ss_16k`,
//!   `facebook_denoiser`) — silently sharing would misroute runtime
//!   dispatch (FR-EX-08).
//! - `vokra.model.name` (`String`): `"storm"` — auxiliary check.
//! - `vokra.model.category` (`String`): `"enhancement"` (single-mask
//!   enhancement + dereverberation head, mirror of sibling DFN3 /
//!   NSNet2 / GTCRN posture).
//! - `vokra.storm.{sample_rate, n_fft, hop, d_model, n_stages,
//!   score_channels}` (`u32` each): the 6-axis topology from
//!   arXiv:2312.09386 §III + SGMSE+ Interspeech 2022 precedent. Read
//!   strict — a partially-stamped GGUF is caught here rather than
//!   silently defaulting to a fabricated axis.
//! - `vokra.provenance.*`: license class + raw license string, so the
//!   runtime compliance gate (FR-CP-03) can classify the artifact
//!   without re-inspecting the safetensors provenance. Defaults to
//!   `Permissive` in production per the MIT stamp; missing provenance
//!   falls back to `Unknown` (fail-closed at the M2-13 gate).
//!
//! # Cross-crate constant duplication
//!
//! Mirror of the converter's [`ARCH`] / [`KEY_STORM_*`] — same rule
//! the sibling BF16 pass-through binders (`pyannote` / `snac` /
//! `hifigan` / `beat_this` / `mt3` / `redimnet` /
//! `sortformer_diar_4spk_v1` / `sepformer` / `conv_tasnet` /
//! `demucs` / `gtcrn`) use so `vokra-models` does not gain a
//! dependency edge onto `vokra-convert`, preserving the layered
//! convention `vokra-ops → nothing GGUF-aware`, `vokra-core → GGUF
//! reader`, `vokra-models → GGUF binder`, `vokra-convert → GGUF
//! writer`. A `[test]` at the bottom of this module pins the mirror
//! so a converter-side rename lands here in the same commit or fails
//! the pin.
//!
//! # Family posture — distinct from every sibling enhancement / separator arch
//!
//! [`ARCH`] = `"storm"` is **deliberately distinct** from every sibling
//! enhancement / separator arch tag; a downstream binder that silently
//! aliases would attempt to walk a StoRM checkpoint through a
//! wrong-topology loader:
//!
//! - `denoise` — DeepFilterNet3 (ERB analysis / synthesis + CRN — a
//!   completely different topology axis from StoRM's diffusion score-
//!   model refinement);
//! - `rnnoise` — Xiph RNNoise (GRU + BSD BFCC / Bark features);
//! - `nsnet2` — Microsoft DNS baseline (2-layer GRU + 3-Linear mask
//!   over 257-bin STFT log-magnitude — same STFT frontend family but
//!   a fundamentally different mask predictor topology, and StoRM is
//!   NOT even a mask predictor — it is a generative score-based
//!   refinement);
//! - `dnsmos` — Microsoft P.808 / P.835 DNSMOS objective quality
//!   estimator (a metric, not a denoiser);
//! - `gtcrn` — GTCRN (grouped Conv2D + SB-TF-LSTM + ERB grouping —
//!   ~23K parameter mask predictor, different topology axis from
//!   StoRM's diffusion score-model refinement);
//! - `metricgan_plus`, `mp_senet_dns`, `frcrn`, `facebook_denoiser`,
//!   `mossformer2_ss_16k` — other enhancement variants with distinct
//!   topologies;
//! - `sepformer`, `conv_tasnet`, `demucs`, `tiger_separator`,
//!   `bs_roformer`, `mp_senet` — separator families with fundamentally
//!   different masker topologies.
//!
//! Silently sharing arch would let runtime dispatch mis-route a StoRM
//! checkpoint onto a wrong-topology loader — FR-EX-08 forbids the
//! silent shape misroute across enhancement / separation families.
//! **StoRM is the FIRST diffusion-based entry on the enhancement arm**
//! from the Wave 7 audit follow-up retry.
//!
//! # No ONNX / no pickle (permanent)
//!
//! StoRM ships as PyTorch state dict upstream (distributed via Google
//! Drive per sp-uhh/storm README convention); this runtime **never**
//! touches ONNX or pickle (FR-LD-05 / NFR-DS-02). The `.pt` →
//! safetensors bridge lives offline through
//! `tools/parity/nemo_pt_to_safetensors.py` (uv-managed Python 3.12
//! sidecar per memory `[[feedback-python-uses-uv]]` +
//! `[[feedback-python-3-12]]`), not part of the runtime — pickle
//! deserialization inside the Rust runtime would violate the FR-LD-05
//! "no arbitrary code execution at load" rule.

use vokra_core::gguf::{GgufFile, chunks};
use vokra_core::{LicenseClass, Result, VokraError};

// ---------------------------------------------------------------------------
// Arch / metadata-key constants — mirror of
// `crates/vokra-convert/src/models/storm.rs`. See module docstring for
// the cross-crate duplication rationale.
// ---------------------------------------------------------------------------

/// Expected `vokra.model.arch` value written by
/// `vokra-cli convert --model storm`.
///
/// Distinct from every sibling denoise / separator arch tag —
/// `denoise` (DFN3), `rnnoise`, `nsnet2`, `dnsmos`, `metricgan_plus`,
/// `mp_senet_dns`, `sepformer`, `conv_tasnet`, `demucs`, `gtcrn`,
/// `frcrn`, `mossformer2_ss_16k`, `facebook_denoiser`. Silently
/// sharing an arch would misroute runtime dispatch (FR-EX-08).
/// Version-neutral (StoRM ships a single 16 kHz release; sibling
/// variants would keep the tag and pick up distinct [`NAME`] stamps).
pub const ARCH: &str = "storm";

/// Expected `vokra.model.name` value — matches the `vokra/storm`
/// publish slug (when it lands under an owner ADR decision — publish
/// is currently gated on the choice between T4 Research-only precedent
/// vs new T1 Permissive GitHub-source precedent since upstream has no
/// HF mirror, Google Drive distribution only).
pub const NAME: &str = "storm";

/// Expected `vokra.model.category` value — single-mask enhancement
/// head. Mirror of sibling `denoise` (DFN3) / `nsnet2` / `rnnoise` /
/// `gtcrn` enhancement family posture. Distinct from separator
/// families (`sepformer` / `conv_tasnet` / `demucs`) which carry
/// `category = "separation"` for multi-source outputs.
pub const CATEGORY: &str = "enhancement";

/// `vokra.storm.sample_rate` — PCM sample rate Hz (typical StoRM =
/// 16000 per arXiv:2312.09386 §III).
pub const KEY_STORM_SAMPLE_RATE: &str = "vokra.storm.sample_rate";
/// `vokra.storm.n_fft` — STFT window size (typical StoRM = 510 per
/// NCSN++ speech-enhancement standard config).
pub const KEY_STORM_N_FFT: &str = "vokra.storm.n_fft";
/// `vokra.storm.hop` — STFT hop in samples (typical StoRM = 128 for
/// 16 kHz diffusion speech enhancement).
pub const KEY_STORM_HOP: &str = "vokra.storm.hop";
/// `vokra.storm.d_model` — NCSN++ v2 U-Net base channel width
/// (typical StoRM = 128).
pub const KEY_STORM_D_MODEL: &str = "vokra.storm.d_model";
/// `vokra.storm.n_stages` — NCSN++ U-Net down/up-sampling stage count
/// (typical StoRM = 4).
pub const KEY_STORM_N_STAGES: &str = "vokra.storm.n_stages";
/// `vokra.storm.score_channels` — score network base output width
/// (typical StoRM = 128, mirrors `d_model` by construction in
/// NCSN++ v2 but stamped separately so a downstream reader can
/// validate the shape).
pub const KEY_STORM_SCORE_CHANNELS: &str = "vokra.storm.score_channels";

/// Primary-source anchor: upstream GitHub repository. Cited in the
/// loud-partial error so a reader diagnosing the gap knows the
/// definitive reference implementation source.
const PRIMARY_SOURCE_REPO: &str = "github.com/sp-uhh/storm";
/// Primary-source anchor: Lay et al. 2023 arXiv paper. Cited
/// alongside the repo anchor so a reader has the theoretical context
/// as well.
const PRIMARY_SOURCE_PAPER: &str = "arxiv.org/abs/2312.09386";

// ---------------------------------------------------------------------------
// StormConfig — the topology axes read from the `vokra.storm.*` chunk
// group. STRICT: every axis is required (FR-EX-08 — no primary-source
// constant fallback since a partial stamp would fabricate axes without
// primary-source backing; the converter always stamps every axis so a
// proper conversion carries the full group).
// ---------------------------------------------------------------------------

/// StoRM topology hyperparameters as they ride the `vokra.storm.*`
/// chunk group.
///
/// [`from_gguf`](Self::from_gguf) is a **strict** loader: every axis
/// is required (FR-EX-08 — never a silent primary-source constant
/// fallback because the fallback would fabricate axes the runtime
/// then binds against). A GGUF missing any `vokra.storm.*` chunk is
/// rejected loudly with a [`VokraError::ModelLoad`] naming the absent
/// key.
///
/// **StoRM-specific note on structural invariants**: unlike sibling
/// GTCRN (n_bands = n_fft/2 + 1 real-input FFT), StoRM's score network
/// operates on the **full complex STFT** (real + imaginary channels),
/// not the half-real spectrum. So n_fft = 510 is even but the config
/// does not decompose to a 256-bin band-count invariant — there is no
/// n_bands axis in [`StormConfig`] by design.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StormConfig {
    /// PCM sample rate in Hz (typical StoRM = 16000).
    pub sample_rate: u32,
    /// STFT window size (typical StoRM = 510 per NCSN++ speech-
    /// enhancement standard config).
    pub n_fft: u32,
    /// STFT hop in samples (typical StoRM = 128 for 16 kHz diffusion
    /// speech enhancement).
    pub hop: u32,
    /// NCSN++ v2 U-Net base channel width (typical StoRM = 128).
    pub d_model: u32,
    /// NCSN++ U-Net down/up-sampling stage count (typical StoRM = 4).
    pub n_stages: u32,
    /// Score network base output width (typical StoRM = 128, mirrors
    /// `d_model` by construction in NCSN++ v2 but held separately so a
    /// downstream reader can validate the shape).
    pub score_channels: u32,
    /// Model category as a `'static` slice — the converter stamps
    /// [`CATEGORY`] verbatim (`"enhancement"`).
    pub category: &'static str,
}

impl StormConfig {
    /// The typical StoRM axes transcribed from arXiv:2312.09386 §III
    /// + SGMSE+ Interspeech 2022 precedent (implementer MUST
    ///   re-confirm against `github.com/sp-uhh/storm/configs/*.yaml` at
    ///   land time rather than trusting the transcribed constants alone
    ///   — CLAUDE.md「ハルシネーション厳禁」).
    ///
    /// Used by the unit tests and as a diagnostic reference. The
    /// runtime loader does NOT default to these; it reads the stamped
    /// values via [`Self::from_gguf`] and fails loud on any missing
    /// chunk.
    #[must_use]
    pub const fn typical_default() -> Self {
        Self::for_stamped_axes(16_000, 510, 128, 128, 4, 128)
    }

    /// Builds a config from caller-supplied axes (used both by the
    /// binder's [`Self::from_gguf`] and by the unit tests). All axes
    /// are `u32`; the category is hard-set to [`CATEGORY`].
    #[must_use]
    pub const fn for_stamped_axes(
        sample_rate: u32,
        n_fft: u32,
        hop: u32,
        d_model: u32,
        n_stages: u32,
        score_channels: u32,
    ) -> Self {
        Self {
            sample_rate,
            n_fft,
            hop,
            d_model,
            n_stages,
            score_channels,
            category: CATEGORY,
        }
    }

    /// Reads every `vokra.storm.*` chunk from `gguf`. Missing axis =
    /// loud [`VokraError::ModelLoad`] naming the absent key (FR-EX-08
    /// — no primary-source constant fallback).
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] when any of the 6 mandatory
    ///   `vokra.storm.*` u32 chunks is absent.
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        fn req_u32(gguf: &GgufFile, key: &str) -> Result<u32> {
            gguf.get(key)
                .and_then(|v| v.as_u64())
                .map(|v| v as u32)
                .ok_or_else(|| {
                    VokraError::ModelLoad(format!(
                        "storm: GGUF is missing required u32 chunk `{key}` — the \
                         upstream `sp-uhh/storm` release ships a single canonical \
                         config and the converter transcribes every axis from \
                         arXiv:2312.09386 §III and stamps them, so a proper conversion \
                         carries the full `vokra.storm.*` chunk group. This binder \
                         refuses to fabricate topology axes from primary-source \
                         constants (FR-EX-08). Re-run `vokra-cli convert --model storm` \
                         against a safetensors checkpoint flattened via \
                         `tools/parity/nemo_pt_to_safetensors.py` (uv-managed Python \
                         3.12 sidecar per memory `[[feedback-python-uses-uv]]`)."
                    ))
                })
        }
        Ok(Self::for_stamped_axes(
            req_u32(gguf, KEY_STORM_SAMPLE_RATE)?,
            req_u32(gguf, KEY_STORM_N_FFT)?,
            req_u32(gguf, KEY_STORM_HOP)?,
            req_u32(gguf, KEY_STORM_D_MODEL)?,
            req_u32(gguf, KEY_STORM_N_STAGES)?,
            req_u32(gguf, KEY_STORM_SCORE_CHANNELS)?,
        ))
    }
}

// ---------------------------------------------------------------------------
// StormWeights — bound the tensor manifest with a non-emptiness gate.
// Under the loud-partial WP the weights are counted but the two-stage
// forward is deferred. Mirror of `GtcrnWeights` / `ConvTasnetWeights` /
// `SepformerWeights` / `ReDimNetWeights`.
// ---------------------------------------------------------------------------

/// Weight tensors bound from a StoRM GGUF.
///
/// **Contract**: [`from_gguf`](Self::from_gguf) is a *loud*
/// verification step. A GGUF that carries zero tensors is rejected
/// with [`VokraError::ModelLoad`] (FR-EX-08 — an empty GGUF is never
/// a valid StoRM checkpoint).
///
/// Under the current landing this struct stores the tensor names +
/// GGUF-side dims discovered on disk. The follow-up wave that lands
/// the two-stage forward sizes its dequant per its kernel needs —
/// today only the count + names are consumed so a future
/// `StormWeights::bind_two_stage_weights` tensor walk can find its
/// inputs without re-parsing the GGUF.
#[derive(Debug)]
pub struct StormWeights {
    /// Tensors discovered on disk, indexed by upstream `state_dict`
    /// name with their GGUF-side dims. Used by the load-time
    /// non-emptiness gate and by the future follow-up two-stage-
    /// forward wave.
    tensors: Vec<(String, Vec<usize>)>,
}

impl StormWeights {
    /// Scans `gguf` for the StoRM state_dict tensors. Refuses to
    /// bind if the GGUF carries zero tensors (FR-EX-08 — an empty
    /// GGUF is never a valid StoRM checkpoint).
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] when the GGUF carries zero tensors.
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let mut tensors: Vec<(String, Vec<usize>)> = Vec::new();
        for info in gguf.tensors() {
            let dims: Vec<usize> = info.dimensions.iter().map(|&d| d as usize).collect();
            tensors.push((info.name.clone(), dims));
        }

        if tensors.is_empty() {
            return Err(VokraError::ModelLoad(
                "storm: GGUF carries zero tensors — refusing to bind an all-zero \
                 forward (FR-EX-08). Re-run `vokra-cli convert --model storm` against \
                 an upstream safetensors checkpoint (upstream ships PyTorch state \
                 dicts via Google Drive which the sibling `tools/parity/\
                 nemo_pt_to_safetensors.py` bridge flattens to safetensors — pickle \
                 deserialization inside the Rust runtime would violate FR-LD-05)."
                    .to_owned(),
            ));
        }
        Ok(Self { tensors })
    }

    /// Number of tensors bound from the GGUF. Purely a diagnostic
    /// accessor — the two-stage-forward wave uses it to size its
    /// expectations.
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }
}

// ---------------------------------------------------------------------------
// Storm — the runtime binder handle
// ---------------------------------------------------------------------------

/// StoRM runtime binder (`sp-uhh/storm`, MIT).
///
/// Bind with [`from_gguf`](Self::from_gguf), then call
/// [`enhance`](Self::enhance) on a mixed mono PCM buffer to obtain the
/// enhanced + dereverberated PCM. See the module doc for the current
/// implementation-status matrix and the FR-EX-08 loud-error contract
/// on the two-stage forward composition.
#[derive(Debug)]
pub struct Storm {
    config: StormConfig,
    // The bound weights are held (real, counted) but the two-stage
    // forward composition is a follow-up wave; the field is
    // deliberately `#[allow(dead_code)]` until the composition lands
    // so a reader is not misled by an unused field. Same posture as
    // RMVPE / pyannote / mt3 / beat_this / sortformer / sepformer /
    // conv_tasnet / demucs / redimnet / gtcrn.
    #[allow(dead_code)]
    weights: StormWeights,
    weight_license: LicenseClass,
}

impl Storm {
    /// Binds a StoRM GGUF: validates arch, reads the strict topology
    /// chunk group, discovers tensors, and surfaces the stamped
    /// weight-license class for compliance gate cross-checks.
    ///
    /// This binder is a *loud* validation step. Every failure is a
    /// distinct [`VokraError::ModelLoad`] naming the missing / wrong
    /// key so a reader diagnosing a mis-produced GGUF has exactly one
    /// place to walk (FR-EX-08 — never a silent partial bind).
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] when `vokra.model.arch` is absent
    ///   or not `"storm"` (a sibling denoise / separator GGUF handed
    ///   to us by mistake fails with a clear message naming every
    ///   sibling arch rather than a downstream "vokra.storm.n_fft
    ///   missing" — same pattern as `Gtcrn::from_gguf` /
    ///   `Mt3::from_gguf` / `ConvTasnet::from_gguf` /
    ///   `SepFormer::from_gguf`).
    /// - [`VokraError::ModelLoad`] when any `vokra.storm.*` chunk is
    ///   absent ([`StormConfig::from_gguf`] is strict — no
    ///   primary-source constant fallback).
    /// - [`VokraError::ModelLoad`] when the GGUF carries zero tensors
    ///   ([`StormWeights::from_gguf`] refuses to bind an all-zero
    ///   forward).
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        // 1. Arch check — always first so a mis-typed model handed
        //    here fails with a specific message instead of a
        //    downstream "vokra.storm.sample_rate missing" error.
        match file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()) {
            Some(a) if a == ARCH => {}
            Some(other) => {
                return Err(VokraError::ModelLoad(format!(
                    "storm: GGUF arch is `{other}`, expected `{ARCH}` (was this GGUF \
                     produced by `vokra-cli convert --model storm`? Note that sibling \
                     denoise / separator arches — `denoise` (DeepFilterNet3, ERB \
                     analysis/synthesis + CRN), `rnnoise` (Xiph GRU + BFCC), `nsnet2` \
                     (Microsoft DNS baseline, 2-layer GRU + 3-Linear mask), `dnsmos` \
                     (P.808/P.835 metric only), `gtcrn` (grouped Conv2D + SB-TF-LSTM + \
                     ERB grouping), `metricgan_plus`, `mp_senet_dns`, `sepformer` \
                     (SpeechBrain dual-path Transformer), `conv_tasnet` (Asteroid \
                     dilated TCN), `demucs` (Meta hybrid U-Net + cross-domain \
                     attention) — all have completely different topologies from StoRM's \
                     NCSN++ v2 score-network + OUVE-SDE predictor-corrector \
                     diffusion-based two-stage stack. StoRM is the FIRST diffusion-based \
                     entry on the enhancement arm — no near-neighbor exists in the \
                     catalogue. Silently aliasing arch would misroute the runtime \
                     dispatch, FR-EX-08.)"
                )));
            }
            None => {
                return Err(VokraError::ModelLoad(
                    "storm: GGUF is missing `vokra.model.arch` (converter did not \
                     stamp it — this is not a Vokra-native storm GGUF)"
                        .to_owned(),
                ));
            }
        }

        // 2. Strict topology axes from the `vokra.storm.*` chunk group.
        let config = StormConfig::from_gguf(file)?;

        // 3. Load the tensor manifest with the non-emptiness gate.
        let weights = StormWeights::from_gguf(file)?;

        // 4. Provenance surfacing — read the stamped weight-license
        //    class for compliance gate cross-checks. The StoRM
        //    converter defaults to `Permissive` per the upstream repo
        //    LICENSE `mit`. Missing provenance falls back to `Unknown`
        //    which is fail-closed at the M2-13 compliance gate — same
        //    posture as GTCRN / MT3 / Sortformer / ConvTasnet /
        //    SepFormer.
        let weight_license = file
            .get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
            .and_then(|v| v.as_str())
            .and_then(LicenseClass::from_class_str)
            .unwrap_or(LicenseClass::Unknown);

        Ok(Self {
            config,
            weights,
            weight_license,
        })
    }

    /// Constructs a test-only [`Storm`] with a placeholder tensor and
    /// the typical [`StormConfig`]. Used by structural tests in this
    /// module — production callers reach [`Self::from_gguf`] instead.
    #[cfg(test)]
    #[must_use]
    pub fn synthesized() -> Self {
        Self {
            config: StormConfig::typical_default(),
            weights: StormWeights {
                tensors: vec![("placeholder.weight".to_owned(), vec![1])],
            },
            weight_license: LicenseClass::Unknown,
        }
    }

    /// The bound topology axes (read from the `vokra.storm.*` chunk
    /// group).
    #[inline]
    #[must_use]
    pub const fn config(&self) -> &StormConfig {
        &self.config
    }

    /// PCM sample rate in Hz (from the stamped
    /// `vokra.storm.sample_rate` chunk).
    #[inline]
    #[must_use]
    pub const fn sample_rate(&self) -> u32 {
        self.config.sample_rate
    }

    /// STFT window size (from the stamped `vokra.storm.n_fft` chunk).
    #[inline]
    #[must_use]
    pub const fn n_fft(&self) -> u32 {
        self.config.n_fft
    }

    /// STFT hop in samples (from the stamped `vokra.storm.hop` chunk).
    #[inline]
    #[must_use]
    pub const fn hop(&self) -> u32 {
        self.config.hop
    }

    /// NCSN++ v2 U-Net base channel width (from the stamped
    /// `vokra.storm.d_model` chunk).
    #[inline]
    #[must_use]
    pub const fn d_model(&self) -> u32 {
        self.config.d_model
    }

    /// NCSN++ U-Net down/up-sampling stage count (from the stamped
    /// `vokra.storm.n_stages` chunk).
    #[inline]
    #[must_use]
    pub const fn n_stages(&self) -> u32 {
        self.config.n_stages
    }

    /// Score network base output width (from the stamped
    /// `vokra.storm.score_channels` chunk).
    #[inline]
    #[must_use]
    pub const fn score_channels(&self) -> u32 {
        self.config.score_channels
    }

    /// Model category — `"enhancement"` for the single-mask
    /// enhancement + dereverberation head.
    #[inline]
    #[must_use]
    pub const fn category(&self) -> &'static str {
        self.config.category
    }

    /// The stamped weight-license class surfaced from the GGUF's
    /// `vokra.provenance.weight_license` chunk. The StoRM converter
    /// stamps `Permissive` by default per the upstream repo LICENSE
    /// `mit` (T1 tier — publish redistributable pending owner ADR on
    /// GitHub-source publish path, no runtime attribution obligation).
    /// A GGUF missing the stamp reads back as
    /// [`LicenseClass::Unknown`] which is also fail-closed at the
    /// M2-13 compliance gate.
    #[inline]
    #[must_use]
    pub const fn weight_license(&self) -> LicenseClass {
        self.weight_license
    }

    /// Number of tensors bound from the GGUF. Purely a diagnostic
    /// accessor — the two-stage-forward wave uses it to size its
    /// expectations.
    #[inline]
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.weights.tensor_count()
    }

    /// Enhances a mixed mono PCM buffer (16 kHz per
    /// [`StormConfig::sample_rate`], typically noisy + reverberant
    /// speech) into an enhanced + dereverberated PCM buffer.
    ///
    /// # Loud-partial (this WP)
    ///
    /// Returns [`VokraError::UnsupportedOp`] — StoRM's inference path
    /// requires **three** deferred primitives + the two-stage compose:
    ///
    /// 1. **initial deterministic predictive estimator** — StoRM's
    ///    first-stage regression sub-network, an NCSN++ v2 U-Net
    ///    variant per arXiv:2312.09386 §III trained under an MSE
    ///    objective. Same topology family as the second-stage score
    ///    network but a distinct forward (no sigma conditioning).
    /// 2. **NCSN++ v2 U-Net score-network** — Noise Conditional Score
    ///    Network++ v2 backbone with attention blocks + feature-wise
    ///    linear modulation (FiLM) over noise-conditioning σ, per
    ///    Song et al. arXiv:2011.13456 §3.3 as extended in StoRM §III.
    ///    NOT covered by existing `vokra_ops` primitives — no U-Net
    ///    with sigma-conditional FiLM primitive exists.
    /// 3. **OUVE-SDE predictor-corrector sampler** — Ornstein-Uhlenbeck
    ///    Variance-Exploding stochastic differential equation
    ///    predictor-corrector Langevin dynamics iterative refinement
    ///    per arXiv:2312.09386 §III + Welker et al. 2022 SGMSE+
    ///    Interspeech precedent. `vokra_ops::flow_sampler` covers
    ///    ODE-style flow matching but NOT the SDE-style
    ///    predictor-corrector Langevin dynamics StoRM requires — the
    ///    two are different sampler families and cannot be silently
    ///    aliased.
    /// 4. The **two-stage compose** — predictor output → score
    ///    refinement loop over sigma schedule + the tensor-name walk
    ///    from upstream `sp-uhh/storm` state_dict prefixes to
    ///    primitive inputs (pending manifest fetch — same posture as
    ///    pyannote/Charsiu real-weight bind).
    ///
    /// The error names all three primitives + the compose + both
    /// primary-source anchors (upstream repo + arXiv paper) so a
    /// reader diagnosing this gap has exactly two places to walk.
    /// Every config axis is echoed so the reader can cross-check what
    /// topology the follow-up wave targets. **No fabricated denoised
    /// waveform is ever emitted** (FR-EX-08).
    ///
    /// # Errors
    ///
    /// - [`VokraError::UnsupportedOp`] — the loud-partial gate for
    ///   the deferred two-stage forward composition.
    pub fn enhance(&self, mixed_pcm: &[f32]) -> Result<Vec<f32>> {
        // Bind unused arg so a `#[warn(unused_variables)]` change does
        // not silently mask the loud-partial fire path; the future
        // real implementation will consume it.
        let _ = mixed_pcm;
        Err(enhance_forward_loud_partial(&self.config))
    }
}

/// Constructs the loud-partial [`VokraError::UnsupportedOp`] returned
/// by [`Storm::enhance`] until the tensor-name walk + two-stage
/// forward composition + three missing primitives land.
///
/// Names **all three** deferred primitives (predictor + NCSN++ v2
/// score-network + OUVE-SDE predictor-corrector) + the two-stage
/// compose so a reader diagnosing the gap knows exactly which
/// `vokra_ops` extensions are required. Cites both primary source
/// URLs (upstream repo README + arXiv paper) so the reader has both
/// the implementation and theoretical anchors. Mirrors the
/// Sortformer / MT3 / beat_this / RMVPE / pyannote / snac / hifigan /
/// vocos / bigvgan / sepformer / conv_tasnet / demucs / gtcrn Wave
/// 3-6 loud-partial-message precedent — CLAUDE.md 教訓 (a).
///
/// Echoes every [`StormConfig`] axis so the reader can cross-check
/// what topology the follow-up wave targets.
///
/// Note: uses [`VokraError::UnsupportedOp`] (not `NotImplemented`)
/// because the message is dynamic-formatted via [`format!`] — the
/// `NotImplemented` variant takes only a `&'static str` and would
/// fail to compile with a `format!` result (Wave 5 canary_qwen
/// E0308 lesson).
fn enhance_forward_loud_partial(cfg: &StormConfig) -> VokraError {
    VokraError::UnsupportedOp(format!(
        "storm enhance: initial deterministic predictive estimator + NCSN++ v2 U-Net \
         score-network + OUVE-SDE (Ornstein-Uhlenbeck Variance-Exploding SDE) \
         predictor-corrector sampler + two-stage compose pending. StoRM's reference \
         implementation decomposes as (a) an initial deterministic predictive \
         estimator (StoRM's first-stage regression sub-network, an NCSN++ v2 U-Net \
         variant per arXiv:2312.09386 §III trained under an MSE objective rather than \
         the score-matching objective — same topology family as the score network but \
         a distinct forward: the predictor's forward pass produces a point estimate of \
         the clean STFT via a standard U-Net regression, no sigma conditioning), (b) \
         an NCSN++ v2 U-Net score-network (Noise Conditional Score Network++ v2 \
         backbone — U-Net with attention blocks + feature-wise linear modulation \
         (FiLM) over noise-conditioning σ, per Song et al. arXiv:2011.13456 §3.3 as \
         extended in StoRM §III — NOT covered by existing `vokra_ops` primitives, no \
         U-Net with sigma-conditional FiLM primitive exists in the catalogue), (c) an \
         OUVE-SDE predictor-corrector sampler (Ornstein-Uhlenbeck Variance-Exploding \
         stochastic differential equation predictor-corrector Langevin dynamics \
         iterative refinement over the sigma schedule per arXiv:2312.09386 §III + \
         Welker et al. 2022 SGMSE+ Interspeech precedent — `vokra_ops::flow_sampler` \
         covers ODE-style flow matching but NOT the SDE-style predictor-corrector \
         Langevin dynamics StoRM requires, the two are different sampler families and \
         cannot be silently aliased), and (d) the two-stage compose (predictor output \
         → score refinement loop over the sigma schedule). Every piece needs (i) the \
         tensor-name walk from the upstream `sp-uhh/storm` state_dict prefixes to the \
         appropriate `vokra_ops` primitives' inputs (pending the manifest fetch — same \
         posture as pyannote / Charsiu real-weight bind), (ii) the three missing \
         primitives themselves landing in `vokra_ops` (predictor U-Net + NCSN++ v2 \
         score-network with sigma FiLM + OUVE-SDE predictor-corrector sampler), and \
         (iii) the two-stage predictor-then-refine composition itself. StoRM is the \
         FIRST diffusion-based entry on the enhancement arm — no near-neighbor exists \
         in the catalogue (`vokra_ops::flow_sampler` is the closest but covers a \
         different sampler family). Config: sample_rate={sample_rate}, n_fft={n_fft}, \
         hop={hop}, d_model={d_model}, n_stages={n_stages}, \
         score_channels={score_channels}, category={category}. Primary sources: {repo} \
         + {paper}. Loud pending (CLAUDE.md 教訓 (a) — 'loud-partial は fake-complete \
         より honest') — no silent fabricated denoised waveform ever emitted \
         (FR-EX-08).",
        sample_rate = cfg.sample_rate,
        n_fft = cfg.n_fft,
        hop = cfg.hop,
        d_model = cfg.d_model,
        n_stages = cfg.n_stages,
        score_channels = cfg.score_channels,
        category = cfg.category,
        repo = PRIMARY_SOURCE_REPO,
        paper = PRIMARY_SOURCE_PAPER,
    ))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    //! Tests for the StoRM runtime binder — cross-crate constant mirror
    //! + config default pin + full topology round-trip on the strict
    //!   chunk group + negative-space round-trip on the loud-partial
    //!   gates + arch-tag distinctness pin.
    //!
    //! # What "round-trip" means here
    //!
    //! The task spec asks for 5+ unit tests. On real PCM this would
    //! be `enhance(...)` returning enhanced + dereverberated audio,
    //! but the two-stage forward + tensor-name walk + three missing
    //! primitives (predictor + NCSN++ v2 score-network + OUVE-SDE
    //! predictor-corrector sampler) are all deferred (see the module
    //! doc + [`Storm::enhance`] rustdoc). Fabricating a real-PCM
    //! output would violate CLAUDE.md 教訓 (a) ("loud-partial は
    //! fake-complete より honest").
    //!
    //! The round-trip semantics we *can* honestly test:
    //!
    //! 1. **Cross-crate constant mirror pin**: [`ARCH`] +
    //!    [`KEY_STORM_*`] (6 axes) + [`CATEGORY`] mirror the converter
    //!    verbatim.
    //! 2. **Config default pin**: [`StormConfig::typical_default`]
    //!    matches the primary-source-transcribed axes.
    //! 3. **Synthesized round-trip**: [`Storm::synthesized`] yields
    //!    the expected accessor values.
    //! 4. **Metadata round-trip**: `from_gguf` binds a legitimate
    //!    GGUF (arch + name + category + full 6-axis chunk group +
    //!    provenance license + one representative tensor), reads back
    //!    every axis + license class + tensor count.
    //! 5. **Loud-error negative-space round-trip**: every stated
    //!    blocker (missing arch / wrong arch / missing chunk /
    //!    unsupported forward surface) fires at its documented
    //!    surface point, in the documented error variant.
    //! 6. **Arch-tag distinctness pin**: [`ARCH`] is deliberately
    //!    distinct from every sibling denoise / separator arch
    //!    (including `gtcrn` — StoRM ≠ GTCRN).

    use super::*;
    use vokra_core::gguf::{GgmlType, GgufBuilder};

    /// Builds a minimal StoRM GGUF carrying the arch tag + name +
    /// category + full `vokra.storm.*` chunk group + one
    /// representative tensor. Optional `weight_license_class` is
    /// written under `vokra.provenance.weight_license` (or omitted
    /// if `None`).
    fn storm_gguf(cfg: StormConfig, weight_license_class: Option<LicenseClass>) -> GgufFile {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_string(chunks::KEY_MODEL_NAME, NAME);
        b.add_string("vokra.model.category", CATEGORY);
        b.add_u32(KEY_STORM_SAMPLE_RATE, cfg.sample_rate);
        b.add_u32(KEY_STORM_N_FFT, cfg.n_fft);
        b.add_u32(KEY_STORM_HOP, cfg.hop);
        b.add_u32(KEY_STORM_D_MODEL, cfg.d_model);
        b.add_u32(KEY_STORM_N_STAGES, cfg.n_stages);
        b.add_u32(KEY_STORM_SCORE_CHANNELS, cfg.score_channels);
        if let Some(cls) = weight_license_class {
            b.add_string(chunks::KEY_PROVENANCE_WEIGHT_LICENSE, cls.as_str());
        }
        // One representative tensor so the non-emptiness gate passes.
        // Uses a plausible upstream state_dict-like name (NCSN++ v2
        // U-Net down-sampling stage 0 initial conv) so the naming
        // contract (verbatim key pass-through by the converter) is
        // exercised alongside.
        b.add_tensor(
            "score_network.down.0.conv.weight",
            GgmlType::F32,
            vec![128, 2, 3, 3],
            vec![0u8; 128 * 2 * 3 * 3 * 4],
        )
        .expect("add_tensor");
        GgufFile::parse(b.to_bytes().expect("serialize")).expect("parse")
    }

    // -----------------------------------------------------------------
    // Test 1 — Cross-crate constant mirror pin
    // -----------------------------------------------------------------

    /// Pin the [`ARCH`] + 6 [`KEY_STORM_*`] + [`CATEGORY`] constants
    /// to the exact strings the converter stamps. A rename in either
    /// crate must land in the same commit or fail this pin.
    #[test]
    fn cross_crate_constant_mirror_pin() {
        // Match the converter's stamps byte-for-byte (see
        // `crates/vokra-convert/src/models/storm.rs`).
        assert_eq!(ARCH, "storm");
        assert_eq!(NAME, "storm");
        assert_eq!(CATEGORY, "enhancement");
        assert_eq!(KEY_STORM_SAMPLE_RATE, "vokra.storm.sample_rate");
        assert_eq!(KEY_STORM_N_FFT, "vokra.storm.n_fft");
        assert_eq!(KEY_STORM_HOP, "vokra.storm.hop");
        assert_eq!(KEY_STORM_D_MODEL, "vokra.storm.d_model");
        assert_eq!(KEY_STORM_N_STAGES, "vokra.storm.n_stages");
        assert_eq!(KEY_STORM_SCORE_CHANNELS, "vokra.storm.score_channels");
    }

    // -----------------------------------------------------------------
    // Test 2 — Arch-tag distinctness pin
    // -----------------------------------------------------------------

    /// Pin `ARCH = "storm"` and assert distinctness against every
    /// sibling denoise / separator arch string. A future rename of
    /// any sibling would land here in the same commit or fail this
    /// test. `gtcrn` included: StoRM ≠ GTCRN (different topology).
    #[test]
    fn arch_tag_distinct_from_sibling_denoise_separator_arches() {
        assert_eq!(ARCH, "storm");
        for sibling in [
            "denoise",            // DeepFilterNet3
            "rnnoise",            // Xiph RNNoise (BSD)
            "nsnet2",             // Microsoft DNS baseline
            "dnsmos",             // Microsoft DNSMOS metric
            "metricgan_plus",     // MetricGAN+
            "mp_senet_dns",       // MP-SENet DNS variant
            "sepformer",          // SpeechBrain SepFormer
            "conv_tasnet",        // Asteroid ConvTasNet
            "demucs",             // Facebook Demucs / HT-Demucs
            "frcrn",              // FRCRN
            "mossformer2_ss_16k", // MossFormer2
            "facebook_denoiser",  // Meta Denoiser
            "gtcrn",              // GTCRN (Wave 6 sibling — StoRM ≠ GTCRN)
        ] {
            assert_ne!(
                ARCH, sibling,
                "storm (NCSN++ v2 score-network + OUVE-SDE predictor-corrector) and \
                 `{sibling}` are distinct enhancement / separator arches — sharing arch \
                 tag would misroute the runtime dispatch (FR-EX-08)"
            );
        }
    }

    // -----------------------------------------------------------------
    // Test 3 — StormConfig default matches typical StoRM axes
    // -----------------------------------------------------------------

    /// Pin [`StormConfig::typical_default`] to the arXiv:2312.09386
    /// §III + SGMSE+ Interspeech 2022 precedent typical config. A
    /// rename or axis-value change would land here in the same commit
    /// or fail this test. Implementer MUST re-confirm against
    /// `github.com/sp-uhh/storm/configs/*.yaml` at land time.
    ///
    /// **StoRM-specific note**: unlike sibling GTCRN, we do NOT
    /// assert `n_bands == n_fft/2 + 1` because StoRM's score network
    /// operates on the full complex STFT (there is no n_bands axis
    /// in StormConfig).
    #[test]
    fn config_typical_default_matches_transcribed_axes() {
        let cfg = StormConfig::typical_default();
        assert_eq!(cfg.sample_rate, 16_000, "sample_rate typical pin");
        assert_eq!(cfg.n_fft, 510, "n_fft typical pin");
        assert_eq!(cfg.hop, 128, "hop typical pin");
        assert_eq!(cfg.d_model, 128, "d_model typical pin");
        assert_eq!(cfg.n_stages, 4, "n_stages typical pin");
        assert_eq!(cfg.score_channels, 128, "score_channels typical pin");
        assert_eq!(
            cfg.category, "enhancement",
            "StoRM is a single-mask enhancement + dereverberation head (not a separator)"
        );
        // `for_stamped_axes` builds the same value.
        assert_eq!(
            cfg,
            StormConfig::for_stamped_axes(16_000, 510, 128, 128, 4, 128),
            "for_stamped_axes must yield the same value as typical_default"
        );
    }

    // -----------------------------------------------------------------
    // Test 4 — Synthesized round-trip
    // -----------------------------------------------------------------

    /// Pin the [`Storm::synthesized`] accessors so a later refactor of
    /// the accessor surface cannot silently change what the test
    /// fixture exposes.
    #[test]
    fn synthesized_round_trip() {
        let s = Storm::synthesized();
        assert_eq!(s.sample_rate(), 16_000);
        assert_eq!(s.n_fft(), 510);
        assert_eq!(s.hop(), 128);
        assert_eq!(s.d_model(), 128);
        assert_eq!(s.n_stages(), 4);
        assert_eq!(s.score_channels(), 128);
        assert_eq!(s.category(), "enhancement");
        assert_eq!(s.tensor_count(), 1);
        assert_eq!(
            s.weight_license(),
            LicenseClass::Unknown,
            "synthesized fixture uses Unknown (fail-closed at M2-13)"
        );
        assert_eq!(*s.config(), StormConfig::typical_default());
    }

    // -----------------------------------------------------------------
    // Test 5 — from_gguf full chunk-group round-trip
    // -----------------------------------------------------------------

    /// Build a legitimate GGUF (arch + name + category + full 6-axis
    /// chunk group + provenance license class + one representative
    /// tensor). The binder must bind, hold the primary-source axes,
    /// surface the Permissive license class, and report at least one
    /// tensor bound.
    #[test]
    fn from_gguf_metadata_round_trip() {
        let cfg = StormConfig::typical_default();
        let file = storm_gguf(cfg, Some(LicenseClass::Permissive));
        let s = Storm::from_gguf(&file).expect("valid GGUF must bind");
        // Config round-trip — every axis stamped by the converter
        // reads back into the same StormConfig value.
        assert_eq!(*s.config(), cfg);
        // Accessor round-trip: every accessor surfaces the stamped
        // axis unchanged.
        assert_eq!(s.sample_rate(), cfg.sample_rate);
        assert_eq!(s.n_fft(), cfg.n_fft);
        assert_eq!(s.hop(), cfg.hop);
        assert_eq!(s.d_model(), cfg.d_model);
        assert_eq!(s.n_stages(), cfg.n_stages);
        assert_eq!(s.score_channels(), cfg.score_channels);
        assert_eq!(s.category(), CATEGORY);
        // License-class surface: the StoRM converter defaults to
        // Permissive per the MIT stamp; missing provenance falls back
        // to Unknown (fail-closed at M2-13).
        assert_eq!(
            s.weight_license(),
            LicenseClass::Permissive,
            "storm converter defaults to Permissive per MIT stamp"
        );
        assert!(
            s.tensor_count() >= 1,
            "at least one tensor must be bound from the legitimate GGUF fixture"
        );
    }

    // -----------------------------------------------------------------
    // Test 6 — from_gguf rejects missing arch chunk
    // -----------------------------------------------------------------

    /// A GGUF that carries no `vokra.model.arch` at all (e.g. a
    /// hand-assembled fixture from an unrelated pipeline) must fail
    /// loud rather than mis-bind (FR-EX-08).
    #[test]
    fn from_gguf_rejects_missing_arch_chunk() {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_NAME, "not-storm");
        // No `vokra.model.arch`.
        b.add_tensor(
            "some.tensor.weight",
            GgmlType::F32,
            vec![4, 4],
            vec![0u8; 64],
        )
        .expect("add_tensor");
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let Err(err) = Storm::from_gguf(&file) else {
            panic!("expected ModelLoad on missing arch");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains("missing `vokra.model.arch`"),
                    "message must call out the missing arch key, got `{m}`"
                );
                assert!(
                    m.contains("storm"),
                    "message must name the storm binder so a reader knows which \
                     loader complained, got `{m}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------
    // Test 7 — from_gguf rejects wrong arch (never silently mis-routes)
    // -----------------------------------------------------------------

    /// A `gtcrn` GGUF handed to the StoRM binder by mistake must fail
    /// loud with a specific message naming both `gtcrn` and `storm`
    /// as well as sibling denoise / separator arches, plus the
    /// NCSN++/OUVE-SDE topology callout + FR-EX-08 clause.
    /// StoRM's diffusion score-based two-stage stack and GTCRN's
    /// grouped Conv2D + SB-TF-LSTM + ERB grouping stack are
    /// FUNDAMENTALLY different topology axes.
    #[test]
    fn from_gguf_rejects_wrong_arch() {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, "gtcrn");
        b.add_tensor("some.tensor", GgmlType::F32, vec![4, 4], vec![0u8; 64])
            .expect("add_tensor");
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let Err(err) = Storm::from_gguf(&file) else {
            panic!("expected ModelLoad on wrong arch");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains("`gtcrn`") && m.contains("`storm`"),
                    "message must name both the got and expected arch tags, got `{m}`"
                );
                // The error message names sibling denoise / separator
                // arches so a reader knows which sibling should not be
                // aliased.
                for sibling in [
                    "denoise",
                    "rnnoise",
                    "nsnet2",
                    "dnsmos",
                    "sepformer",
                    "conv_tasnet",
                    "demucs",
                    "gtcrn",
                ] {
                    assert!(
                        m.contains(sibling),
                        "message must name sibling `{sibling}`, got `{m}`"
                    );
                }
                assert!(
                    m.contains("NCSN++") && m.contains("OUVE-SDE"),
                    "message should call out StoRM's characteristic primitives \
                     (NCSN++ + OUVE-SDE) so the reader knows why the arches are \
                     distinct, got `{m}`"
                );
                assert!(
                    m.contains("FR-EX-08"),
                    "message must cite the FR-EX-08 clause, got `{m}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------
    // Test 8 — Missing topology chunk fails loud (parametrized over
    //          each of the 6 axes)
    // -----------------------------------------------------------------

    /// For each of the 6 mandatory `vokra.storm.*` axes, omit exactly
    /// that one and assert the binder loud-fails with the missing
    /// key named in the error. A partially-stamped GGUF must be
    /// caught here, not silently defaulted to a fabricated axis
    /// (FR-EX-08 — the converter always stamps every axis, so a
    /// missing chunk always signals a partial / mis-produced GGUF).
    #[test]
    fn from_gguf_rejects_missing_topology_axis_each_of_six() {
        // Owned iteration over a fixed-size array so `k` / `skip_key`
        // bind as owned `&'static str` (not `&&str`) — avoids
        // auto-deref ambiguity when passing to `add_u32(key: &str,
        // ...)`.
        let axes: [(&str, u32); 6] = [
            (KEY_STORM_SAMPLE_RATE, 16_000),
            (KEY_STORM_N_FFT, 510),
            (KEY_STORM_HOP, 128),
            (KEY_STORM_D_MODEL, 128),
            (KEY_STORM_N_STAGES, 4),
            (KEY_STORM_SCORE_CHANNELS, 128),
        ];
        for skip_idx in 0..axes.len() {
            let skip_key = axes[skip_idx].0;
            let mut b = GgufBuilder::new();
            b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
            for (i, (k, v)) in axes.iter().enumerate() {
                if i == skip_idx {
                    continue;
                }
                b.add_u32(k, *v);
            }
            b.add_tensor(
                "score_network.down.0.conv.weight",
                GgmlType::F32,
                vec![4, 4],
                vec![0u8; 64],
            )
            .expect("add_tensor");
            let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
            let Err(err) = Storm::from_gguf(&file) else {
                panic!("expected ModelLoad on missing axis `{skip_key}`");
            };
            match err {
                VokraError::ModelLoad(m) => {
                    assert!(
                        m.contains(skip_key),
                        "message must name the missing axis key `{skip_key}`, got `{m}`"
                    );
                    assert!(
                        m.contains("arXiv:2312.09386"),
                        "message should cite the arXiv anchor so the reader knows the \
                         primary source of the typical values, got `{m}`"
                    );
                }
                other => panic!("expected VokraError::ModelLoad, got {other:?}"),
            }
        }
    }

    // -----------------------------------------------------------------
    // Test 9 — enhance returns UnsupportedOp naming all three missing
    //          primitives + the two-stage compose + both primary
    //          source URLs + every config axis
    // -----------------------------------------------------------------

    /// [`Storm::enhance`] must loud-partial with
    /// [`VokraError::UnsupportedOp`] naming all three deferred
    /// primitives (predictor + NCSN++ v2 score-network + OUVE-SDE
    /// predictor-corrector) + the two-stage compose + both primary
    /// source URLs (upstream repo + arXiv paper) + every config axis
    /// + the FR-EX-08 clause + the CLAUDE.md 教訓 (a) reference.
    #[test]
    fn enhance_returns_unsupported_op_with_all_primitives_named() {
        let s = Storm::synthesized();
        // 1 s of 16 kHz mono silence — legitimate input shape, so the
        // loud-partial gate fires (not some pre-enhance validation).
        let pcm = vec![0.0f32; 16_000];
        let Err(err) = s.enhance(&pcm) else {
            panic!("enhance must loud-partial");
        };
        match err {
            VokraError::UnsupportedOp(m) => {
                assert!(
                    m.contains("storm enhance"),
                    "message must call out the storm enhance surface, got `{m}`"
                );
                // All three missing primitives + the compose must be
                // named by exact identifier so the follow-up wave knows
                // what to walk.
                assert!(
                    m.contains("NCSN++"),
                    "message must name the NCSN++ v2 U-Net score-network gap, got `{m}`"
                );
                assert!(
                    m.contains("OUVE-SDE"),
                    "message must name the OUVE-SDE predictor-corrector sampler gap, \
                     got `{m}`"
                );
                assert!(
                    m.contains("predictor-corrector"),
                    "message must name the predictor-corrector sampler family, got `{m}`"
                );
                assert!(
                    m.contains("score"),
                    "message must reference the score-network, got `{m}`"
                );
                // Both primary source URLs must be cited.
                assert!(
                    m.contains("github.com/sp-uhh/storm"),
                    "message must contain the upstream repo URL, got `{m}`"
                );
                assert!(
                    m.contains("arxiv.org/abs/2312.09386"),
                    "message must contain the arXiv paper URL, got `{m}`"
                );
                // Every config axis must be echoed so the reader can
                // cross-check what topology the follow-up wave targets.
                assert!(
                    m.contains("sample_rate=16000"),
                    "sample_rate axis missing: {m}"
                );
                assert!(m.contains("n_fft=510"), "n_fft axis missing: {m}");
                assert!(m.contains("hop=128"), "hop axis missing: {m}");
                assert!(m.contains("d_model=128"), "d_model axis missing: {m}");
                assert!(m.contains("n_stages=4"), "n_stages axis missing: {m}");
                assert!(
                    m.contains("score_channels=128"),
                    "score_channels axis missing: {m}"
                );
                assert!(
                    m.contains("category=enhancement"),
                    "category axis missing: {m}"
                );
                // FR-EX-08 clause citation.
                assert!(
                    m.contains("FR-EX-08"),
                    "message must cite the FR-EX-08 clause, got `{m}`"
                );
                // CLAUDE.md 教訓 (a) reference.
                assert!(
                    m.contains("教訓 (a)") || m.contains("loud-partial は fake-complete"),
                    "message must cite CLAUDE.md 教訓 (a), got `{m}`"
                );
                // "storm" identifier in the message (for grep-ability).
                assert!(
                    m.contains("storm"),
                    "message must contain the storm identifier for greppability, got `{m}`"
                );
            }
            other => panic!("expected VokraError::UnsupportedOp, got {other:?}"),
        }
    }
}
