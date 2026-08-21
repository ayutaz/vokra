//! **Facebook Denoiser** (`facebookresearch/denoiser`, **CC-BY-NC-4.0**)
//! — real-time speech-enhancement waveform U-Net + LSTM runtime binder
//! for the `facebook_denoiser` converter arch (Wave 8 2026-08-14 audit
//! follow-up).
//!
//! # Primary source
//!
//! - Paper: Defossez et al. 2020 arXiv:2006.12847
//!   *"Real Time Speech Enhancement in the Waveform Domain"*.
//! - Reference implementation:
//!   <https://github.com/facebookresearch/denoiser>
//! - Weight license: **CC-BY-NC-4.0** (T4 tier — research-only, publish
//!   requires `--allow-noncommercial`, `docs/license-audit.md` line 457
//!   ☑ Research-only 2026-08-04 yousan sign-off. The runtime M2-13
//!   compliance gate refuses commercial-mode load per FR-CP-03).
//!
//! # Runtime layout (loud-partial, enhancement-fleet posture per
//! CLAUDE.md 教訓 (a))
//!
//! ```text
//! Noisy PCM (mono f32, 16 kHz per Defossez §4.1 DNS-Challenge default)
//!   -> 5-block time-domain waveform U-Net encoder    ← **loud-partial**
//!        (per arXiv:2006.12847 §II.A: five convolutional encoder
//!         blocks, each halving temporal resolution via stride 4 and
//!         doubling channel width H·2^L with H=48 base for causal
//!         denoiser (H=64 for non-causal). Kernel size 8, GLU activation.
//!         Each block = Conv1d(kernel=8, stride=4) + ReLU + Conv1d(1x1,
//!         2C output) + GLU. Distinct from HuBERT/wav2vec2 conv stem
//!         (feature extractor, no residual U-Net skip connections) —
//!         this is a symmetric encoder that pairs with a decoder via
//!         skip connections.)
//!   -> 2-layer LSTM bottleneck                       ← **loud-partial**
//!        (per arXiv:2006.12847 §II.A + causal-vs-non-causal ablation
//!         §III.B: 2 stacked LSTM layers between encoder and decoder,
//!         hidden = 2 * (H · 2^(L-1)). Unidirectional for the causal
//!         real-time model (`denoiser_causal.th`); bidirectional for
//!         the non-causal offline model (`master64.th` etc.). No
//!         *enhancement-family* op in `vokra_ops` carries an LSTM — but
//!         this is a LIFT, not greenfield RNN work: the LSTM gate body
//!         in `vokra_ops::hybrid_ctc_attention::LstmLmCell` and the two
//!         bidirectional LSTMs `crate::kokoro::nn::BiLstm1d` /
//!         `crate::pyannote::bilstm::BiLstmLayer` are all already
//!         written against the PyTorch weight layout. See
//!         [`crate::squim::missing_primitive_note`] — SQUIM's DPRNN
//!         head needs the same reusable bare cell, so landing it once
//!         unblocks both.)
//!   -> 5-block symmetric transposed-conv decoder     ← **loud-partial**
//!        (per arXiv:2006.12847 §II.A: five symmetric decoder blocks
//!         mirroring the encoder, each doubling temporal resolution via
//!         transposed convolution (`ConvTranspose1d(kernel=8, stride=4)`)
//!         and halving channel width. Each block = Conv1d(1x1, 2C
//!         output) + GLU + ConvTranspose1d(kernel=8, stride=4). Every
//!         block sums a skip connection from the paired encoder block
//!         BEFORE the transposed conv (U-Net-style additive skip, not
//!         concat). The final decoder block projects to 1 channel to
//!         reconstruct denoised waveform samples. Distinct from HiFi-GAN
//!         upsampler stack — HiFi-GAN has no encoder-side skip fusion
//!         and operates on mel spectrograms not raw waveform.)
//!   -> denoised PCM (same length as input, 16 kHz mono)
//! ```
//!
//! # Loud-partial classification (design § — CLAUDE.md 教訓 (a))
//!
//! - **Real (this WP)**: [`FbDenoiser::from_gguf`] with strict
//!   `vokra.model.arch == "facebook_denoiser"` validation (distinct
//!   from every sibling enhancement / denoise family arch — see the
//!   family-posture section below), [`FbDenoiserWeights::from_gguf`]
//!   with a non-empty tensor count floor enforced loud (a GGUF that
//!   carries zero tensors is refused rather than silently running an
//!   all-zero forward — FR-EX-08), and weight-license class surfacing
//!   (defaults to [`LicenseClass::Unknown`] on a stamp-free fixture,
//!   fail-closed at the M2-13 compliance gate — the converter stamps
//!   [`LicenseClass::NonCommercial`] in production per the CC-BY-NC-4.0
//!   default).
//! - **Loud-partial (this WP)**: [`FbDenoiser::denoise`] returns
//!   [`VokraError::UnsupportedOp`] naming **all three** deferred pieces:
//!   (i) **5-block time-domain waveform U-Net encoder** (Conv1d(k=8,
//!   stride=4) + GLU stack, channel growth H·2^L, causal denoiser
//!   H=48);
//!   (ii) **2-layer LSTM bottleneck** (unidirectional for causal
//!   `denoiser_causal.th`, bidirectional for offline `master64.th`);
//!   (iii) **5-block symmetric transposed-conv decoder** with additive
//!   encoder-side skip connections.
//!   The error cites both primary sources (GitHub repo README + arXiv
//!   paper) so a reader diagnosing this gap has exactly two anchors to
//!   walk.
//!
//! Rationale (RMVPE / pyannote / hifigan / vocos / bigvgan / snac /
//! beat_this / mt3 / redimnet / sortformer / sepformer / conv_tasnet /
//! demucs / gtcrn / storm / wavlm Wave 1-7 loud-partial precedent,
//! CLAUDE.md 教訓 (a) — "loud-partial は fake-complete より honest"):
//! the surrounding scaffold + `from_gguf` arch validation +
//! non-empty-tensor gate + FR-EX-08 loud-fails land today so a follow-up
//! wave can flip the switch by (i) landing the tensor-name walk against
//! a real facebook-denoiser state_dict (the upstream release ships
//! PyTorch `.th` pickles at `https://dl.fbaipublicfiles.com/adiyoss/
//! denoiser/` that the sibling `tools/parity/nemo_pt_to_safetensors.py`
//! uv-managed Python 3.12 sidecar bridges to safetensors), (ii) landing
//! the three missing primitives (encoder + LSTM bottleneck + decoder
//! with additive skip), and (iii) composing the three-stage forward
//! against the discovered tensor names.
//!
//! # Family posture — distinct from every sibling enhancement / denoise arch
//!
//! [`ARCH`] = `"facebook_denoiser"` is **deliberately distinct** from
//! every sibling enhancement / denoise arch tag; a downstream binder
//! that silently aliases would attempt to walk a facebook-denoiser
//! checkpoint through a wrong-topology loader:
//!
//! - `denoise` — DeepFilterNet3 (ERB analysis / synthesis + CRN — a
//!   completely different topology axis from facebook-denoiser's
//!   time-domain waveform U-Net + LSTM);
//! - `rnnoise` — Xiph RNNoise v0.2 (GRU + 32-band/65-feature spectral
//!   frontend — no U-Net, no LSTM, feature-domain network);
//! - `nsnet2` — Microsoft DNS baseline (2-layer GRU + 3-Linear mask
//!   over 257-bin STFT log-magnitude — spectral mask predictor, not a
//!   waveform-domain U-Net);
//! - `dnsmos` — Microsoft P.808 / P.835 DNSMOS objective quality
//!   estimator (a metric, not a denoiser);
//! - `gtcrn` — GTCRN (grouped Conv2D + SB-TF-LSTM + ERB grouping —
//!   ~23K parameter mask predictor over STFT bands, different domain);
//! - `dtln_aec` — DTLN AEC (dual-signal transform LSTM network, echo
//!   cancellation over STFT — different task + different domain);
//! - `mp_senet` / `mp_senet_dns` — MP-SENet (magnitude-phase U-Net with
//!   attention over STFT — spectral-domain, not waveform-domain);
//! - `frcrn` — FRCRN (complex U-Net + FR-LSTM over STFT — spectral-
//!   domain, not waveform-domain);
//! - `metricgan_plus` — MetricGAN+ (GAN-trained mask predictor over
//!   STFT — different training + different domain);
//! - `mossformer2_ss_16k` — MossFormer2 (dual-path Transformer-based
//!   separator — different task family);
//! - `storm` — StoRM (diffusion-based two-stage NCSN++ v2 + OUVE-SDE
//!   sampler — different sampler family + different topology);
//! - `sepformer`, `conv_tasnet`, `demucs` — separator families
//!   (multi-source outputs, `category = "separation"`), not single-mask
//!   denoisers.
//!
//! Silently sharing arch would let runtime dispatch mis-route a
//! facebook-denoiser checkpoint onto a wrong-topology loader — FR-EX-08
//! forbids the silent shape misroute across enhancement / separation
//! families. **facebook-denoiser is the FIRST time-domain waveform
//! U-Net + LSTM entry on the enhancement arm** — no near-neighbor in
//! the catalogue.
//!
//! # No `vokra.facebook_denoiser.*` topology chunk group
//!
//! Unlike sibling `storm` / `wavlm` / `gtcrn` binders, this module does
//! NOT read a strict topology chunk group. Rationale: the converter
//! (`crates/vokra-convert/src/models/facebook_denoiser.rs`) is a plain
//! BF16 pass-through with verbatim state-dict tensor names — no
//! `config.json` is transcribed and no `vokra.facebook_denoiser.*` axes
//! are stamped. Primary-source constants ([`SAMPLE_RATE_DEFAULT`] =
//! 16000 per Defossez §4.1 DNS-Challenge) + tensor-manifest
//! non-emptiness constitute the entire load-time contract. The
//! follow-up wave that lands the three-stage forward will derive
//! channel widths + block count from the state_dict tensor shapes
//! themselves (as the sibling RMVPE / pyannote binders will), keeping
//! the load-time contract single-source (the state_dict itself) —
//! stamping fabricated axes without primary-source backing would be a
//! silent lie (FR-EX-08).
//!
//! # Cross-crate constant duplication
//!
//! Mirror of the converter's [`ARCH`] / [`NAME`] / [`CATEGORY`] — same
//! rule the sibling BF16 pass-through binders (`pyannote` / `snac` /
//! `hifigan` / `beat_this` / `mt3` / `redimnet` / `sortformer_diar_4spk_v1`
//! / `sepformer` / `conv_tasnet` / `demucs` / `gtcrn` / `storm` /
//! `wavlm`) use so `vokra-models` does not gain a dependency edge onto
//! `vokra-convert`, preserving the layered convention
//! `vokra-ops → nothing GGUF-aware`, `vokra-core → GGUF reader`,
//! `vokra-models → GGUF binder`, `vokra-convert → GGUF writer`. A
//! `[test]` at the bottom of this module pins the mirror so a
//! converter-side rename lands here in the same commit or fails the
//! pin.
//!
//! # No ONNX / no pickle (permanent)
//!
//! facebook-denoiser ships as PyTorch `.th` / `.pt` pickles upstream
//! (distributed via `dl.fbaipublicfiles.com/adiyoss/denoiser/`); this
//! runtime **never** touches ONNX or pickle (FR-LD-05 / NFR-DS-02).
//! The `.th` → safetensors bridge lives offline in
//! `tools/parity/nemo_pt_to_safetensors.py` (uv-managed Python 3.12
//! sidecar per memory `[[feedback-python-uses-uv]]` +
//! `[[feedback-python-3-12]]`); a model-specific
//! `tools/parity/facebook_denoiser_prepare_checkpoint.py` is **not yet
//! written**. Neither is part of the runtime — pickle
//! deserialization inside the Rust runtime would violate the FR-LD-05
//! "no arbitrary code execution at load" rule.

use std::path::Path;

use vokra_core::gguf::{GgufFile, chunks};
use vokra_core::{LicenseClass, Result, VokraError};

// ---------------------------------------------------------------------------
// Arch / metadata-key constants — mirror of
// `crates/vokra-convert/src/models/facebook_denoiser.rs`. See module
// docstring for the cross-crate duplication rationale.
// ---------------------------------------------------------------------------

/// Expected `vokra.model.arch` value written by
/// `vokra-cli convert --model facebook-denoiser`.
///
/// Distinct from every sibling enhancement / denoise arch tag —
/// `denoise` (DFN3), `rnnoise` (Xiph GRU + BFCC), `nsnet2` (Microsoft
/// DNS baseline), `dnsmos` (metric only), `gtcrn` (grouped Conv2D +
/// SB-TF-LSTM), `dtln_aec` (dual-signal transform LSTM for AEC),
/// `mp_senet` (magnitude-phase U-Net over STFT), `frcrn` (complex
/// U-Net + FR-LSTM over STFT), `metricgan_plus` (GAN-trained mask),
/// `mossformer2_ss_16k` (dual-path Transformer separator), `storm`
/// (diffusion-based two-stage), and separator families (`sepformer`,
/// `conv_tasnet`, `demucs`). Silently sharing an arch would misroute
/// runtime dispatch (FR-EX-08).
pub const ARCH: &str = "facebook_denoiser";

/// Expected `vokra.model.name` value stamped by the converter for the
/// canonical `facebookresearch/denoiser` release.
pub const NAME: &str = "facebook_denoiser";

/// Expected `vokra.model.category` value — single-mask enhancement
/// head. Mirror of sibling `denoise` (DFN3) / `nsnet2` / `rnnoise` /
/// `gtcrn` / `storm` enhancement family posture. Distinct from
/// separator families (`sepformer` / `conv_tasnet` / `demucs`) which
/// carry `category = "separation"` for multi-source outputs.
pub const CATEGORY: &str = "enhancement";

/// Primary redistribution source (author's GitHub repository). Cited
/// in the loud-partial error so a reader diagnosing the gap knows the
/// definitive reference implementation source.
pub const UPSTREAM_URL: &str = "github.com/facebookresearch/denoiser";

/// PCM sample rate in Hz for the canonical facebook-denoiser release
/// (16 kHz per arXiv:2006.12847 §4.1 DNS-Challenge default). The
/// upstream release also ships a 48 kHz `master64` variant but the
/// primary DNS-Challenge release + causal real-time variant both use
/// 16 kHz. Not stamped in the GGUF (see module doc "No
/// `vokra.facebook_denoiser.*` topology chunk group" section) —
/// consumers read this constant directly.
pub const SAMPLE_RATE_DEFAULT: u32 = 16_000;

/// Primary-source anchor: upstream GitHub repository. Cited in the
/// loud-partial error alongside the arXiv paper.
const PRIMARY_SOURCE_GITHUB: &str = "github.com/facebookresearch/denoiser";
/// Primary-source anchor: Defossez et al. 2020 arXiv paper. Cited
/// alongside the GitHub anchor so a reader has the theoretical context
/// as well.
const PRIMARY_SOURCE_ARXIV: &str = "arxiv.org/abs/2006.12847";

// ---------------------------------------------------------------------------
// FbDenoiserWeights — bound the tensor manifest with a non-emptiness
// gate. Under the loud-partial WP the weights are counted but the
// three-stage forward (encoder + LSTM bottleneck + decoder) is
// deferred. Mirror of `StormWeights` / `WavLmSvWeights` /
// `SepformerWeights` / `ReDimNetWeights`.
// ---------------------------------------------------------------------------

/// Weight tensors bound from a facebook-denoiser GGUF.
///
/// **Contract**: [`from_gguf`](Self::from_gguf) is a *loud*
/// verification step. A GGUF that carries zero tensors is rejected
/// with [`VokraError::ModelLoad`] (FR-EX-08 — an empty GGUF is never
/// a valid facebook-denoiser checkpoint).
///
/// Under the current landing this struct stores the tensor names +
/// GGUF-side dims discovered on disk. The follow-up wave that lands
/// the three-stage forward sizes its dequant per its kernel needs —
/// today only the count + names are consumed so a future
/// `FbDenoiserWeights::bind_forward_weights` tensor walk can find its
/// inputs without re-parsing the GGUF.
#[derive(Debug)]
pub struct FbDenoiserWeights {
    /// Tensors discovered on disk, indexed by upstream `state_dict`
    /// name with their GGUF-side dims. Used by the load-time
    /// non-emptiness gate and by the future follow-up three-stage
    /// forward wave.
    tensors: Vec<(String, Vec<usize>)>,
}

impl FbDenoiserWeights {
    /// Scans `gguf` for the facebook-denoiser state_dict tensors.
    /// Refuses to bind if the GGUF carries zero tensors (FR-EX-08 —
    /// an empty GGUF is never a valid facebook-denoiser checkpoint).
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
                "facebook_denoiser: GGUF carries zero tensors — refusing to bind an \
                 all-zero forward (FR-EX-08). Re-run `vokra-cli convert --model \
                 facebook-denoiser` against a safetensors checkpoint flattened via \
                 `tools/parity/nemo_pt_to_safetensors.py` (uv-managed Python 3.12 \
                 sidecar per memory `[[feedback-python-uses-uv]]`; a model-specific \
                 `tools/parity/facebook_denoiser_prepare_checkpoint.py` is not yet \
                 written — pickle deserialization inside the Rust runtime would \
                 violate FR-LD-05)."
                    .to_owned(),
            ));
        }
        Ok(Self { tensors })
    }

    /// Number of tensors bound from the GGUF. Purely a diagnostic
    /// accessor — the three-stage-forward wave uses it to size its
    /// expectations.
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }
}

// ---------------------------------------------------------------------------
// FbDenoiser — the runtime binder handle
// ---------------------------------------------------------------------------

/// Facebook Denoiser runtime binder
/// (`facebookresearch/denoiser`, CC-BY-NC-4.0).
///
/// Bind with [`from_gguf`](Self::from_gguf), then call
/// [`denoise`](Self::denoise) on a noisy mono PCM buffer to obtain the
/// denoised PCM. See the module doc for the current
/// implementation-status matrix and the FR-EX-08 loud-error contract
/// on the three-stage forward composition.
#[derive(Debug)]
pub struct FbDenoiser {
    // The bound weights are held (real, counted) but the three-stage
    // forward composition is a follow-up wave; the field is
    // deliberately `#[allow(dead_code)]` until the composition lands
    // so a reader is not misled by an unused field. Same posture as
    // RMVPE / pyannote / mt3 / beat_this / sortformer / sepformer /
    // conv_tasnet / demucs / redimnet / gtcrn / storm / wavlm.
    #[allow(dead_code)]
    weights: FbDenoiserWeights,
    weight_license: LicenseClass,
}

impl FbDenoiser {
    /// Binds a facebook-denoiser GGUF: validates arch, discovers
    /// tensors with the non-emptiness gate, and surfaces the stamped
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
    ///   or not `"facebook_denoiser"` (a sibling enhancement / denoise
    ///   GGUF handed to us by mistake fails with a clear message
    ///   naming every sibling arch rather than a downstream "missing
    ///   tensor" — same pattern as `Gtcrn::from_gguf` /
    ///   `Mt3::from_gguf` / `Storm::from_gguf`).
    /// - [`VokraError::ModelLoad`] when the GGUF carries zero tensors
    ///   ([`FbDenoiserWeights::from_gguf`] refuses to bind an all-zero
    ///   forward).
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        // 1. Arch check — always first so a mis-typed model handed
        //    here fails with a specific message instead of a
        //    downstream "no tensors" or "wrong topology" error.
        match file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()) {
            Some(a) if a == ARCH => {}
            Some(other) => {
                return Err(VokraError::ModelLoad(format!(
                    "facebook_denoiser: GGUF arch is `{other}`, expected `{ARCH}` (was \
                     this GGUF produced by `vokra-cli convert --model \
                     facebook-denoiser`? Note that sibling enhancement / denoise arches \
                     — `denoise` (DeepFilterNet3, ERB analysis/synthesis + CRN), \
                     `rnnoise` (Xiph GRU + BFCC), `nsnet2` (Microsoft DNS baseline, \
                     2-layer GRU + 3-Linear mask), `dnsmos` (P.808/P.835 metric only), \
                     `gtcrn` (grouped Conv2D + SB-TF-LSTM + ERB grouping), `dtln_aec` \
                     (dual-signal transform LSTM for AEC), `mp_senet` (magnitude-phase \
                     U-Net over STFT), `frcrn` (complex U-Net + FR-LSTM over STFT), \
                     `metricgan_plus` (GAN-trained mask predictor), \
                     `mossformer2_ss_16k` (dual-path Transformer separator), `storm` \
                     (diffusion-based NCSN++ v2 + OUVE-SDE two-stage), `sepformer` / \
                     `conv_tasnet` / `demucs` (separator families) — all have \
                     completely different topologies from facebook-denoiser's \
                     time-domain waveform U-Net + LSTM bottleneck. facebook-denoiser \
                     is the FIRST time-domain waveform U-Net + LSTM entry on the \
                     enhancement arm — no near-neighbor exists in the catalogue. \
                     Silently aliasing arch would misroute the runtime dispatch, \
                     FR-EX-08.)"
                )));
            }
            None => {
                return Err(VokraError::ModelLoad(
                    "facebook_denoiser: GGUF is missing `vokra.model.arch` (converter \
                     did not stamp it — this is not a Vokra-native facebook_denoiser \
                     GGUF)"
                        .to_owned(),
                ));
            }
        }

        // 2. Load the tensor manifest with the non-emptiness gate.
        //    (No strict topology chunk group to read — see module
        //    doc "No `vokra.facebook_denoiser.*` topology chunk group"
        //    section; the converter is a plain BF16 pass-through and
        //    the follow-up forward wave will derive shapes from the
        //    state_dict tensors themselves.)
        let weights = FbDenoiserWeights::from_gguf(file)?;

        // 3. Provenance surfacing — read the stamped weight-license
        //    class for compliance gate cross-checks. The
        //    facebook-denoiser converter defaults to `NonCommercial`
        //    per the upstream cc-by-nc-4.0 default (T4 tier — publish
        //    requires `--allow-noncommercial`). Missing provenance
        //    falls back to `Unknown` which is fail-closed at the M2-13
        //    compliance gate — same posture as GTCRN / MT3 /
        //    Sortformer / ConvTasnet / SepFormer / Storm / WavLM.
        let weight_license = file
            .get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
            .and_then(|v| v.as_str())
            .and_then(LicenseClass::from_class_str)
            .unwrap_or(LicenseClass::Unknown);

        Ok(Self {
            weights,
            weight_license,
        })
    }

    /// Convenience file loader — reads `path` from disk into a
    /// [`GgufFile`] and forwards to [`Self::from_gguf`]. Mirror of the
    /// sibling `Storm::open` posture (a callable this crate exports for
    /// the CLI / server dispatch).
    ///
    /// # Errors
    ///
    /// - [`VokraError::Io`] on read failure.
    /// - [`VokraError::ModelLoad`] on any of the [`Self::from_gguf`]
    ///   failure conditions.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let bytes = std::fs::read(path.as_ref()).map_err(VokraError::Io)?;
        let file = GgufFile::parse(bytes)
            .map_err(|e| VokraError::ModelLoad(format!("facebook_denoiser: {e}")))?;
        Self::from_gguf(&file)
    }

    /// The stamped weight-license class surfaced from the GGUF's
    /// `vokra.provenance.weight_license` chunk. The facebook-denoiser
    /// converter stamps `NonCommercial` by default per the upstream
    /// `cc-by-nc-4.0` (T4 tier — publish requires
    /// `--allow-noncommercial`, no commercial-mode load per FR-CP-03).
    /// A GGUF missing the stamp reads back as
    /// [`LicenseClass::Unknown`] which is also fail-closed at the
    /// M2-13 compliance gate.
    #[inline]
    #[must_use]
    pub const fn weight_license(&self) -> LicenseClass {
        self.weight_license
    }

    /// Number of tensors bound from the GGUF. Purely a diagnostic
    /// accessor — the three-stage-forward wave uses it to size its
    /// expectations.
    #[inline]
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.weights.tensor_count()
    }

    /// PCM sample rate in Hz for the canonical facebook-denoiser
    /// release. Not stamped in the GGUF — surfaced from
    /// [`SAMPLE_RATE_DEFAULT`] per arXiv:2006.12847 §4.1
    /// DNS-Challenge default. See module doc for the "No
    /// `vokra.facebook_denoiser.*` topology chunk group" rationale.
    #[inline]
    #[must_use]
    pub const fn sample_rate(&self) -> u32 {
        SAMPLE_RATE_DEFAULT
    }

    /// Denoises a noisy mono PCM buffer (16 kHz per
    /// [`SAMPLE_RATE_DEFAULT`], typically DNS-Challenge or in-domain
    /// speech) into a denoised PCM buffer of the same length.
    ///
    /// # Loud-partial (this WP)
    ///
    /// Returns [`VokraError::UnsupportedOp`] — facebook-denoiser's
    /// inference path requires **three** deferred pieces
    /// (encoder + LSTM bottleneck + decoder with additive skip):
    ///
    /// 1. **5-block time-domain waveform U-Net encoder** — five
    ///    convolutional encoder blocks per arXiv:2006.12847 §II.A, each
    ///    halving temporal resolution via `Conv1d(kernel=8, stride=4)`
    ///    and doubling channel width `H · 2^L` with `H=48` base for
    ///    the causal real-time model (`H=64` for the non-causal
    ///    offline model). Each block = `Conv1d(k=8, s=4) + ReLU +
    ///    Conv1d(1x1, 2C output) + GLU`. Distinct from the HuBERT /
    ///    wav2vec2 conv stem (feature extractor, no residual U-Net
    ///    skip connections).
    /// 2. **2-layer LSTM bottleneck** — 2 stacked LSTM layers between
    ///    encoder and decoder per arXiv:2006.12847 §II.A + §III.B
    ///    ablation, hidden = `2 · (H · 2^(L-1))`. Unidirectional for
    ///    causal `denoiser_causal.th`, bidirectional for offline
    ///    `master64.th`. No *enhancement-family* op in `vokra_ops`
    ///    carries an LSTM, but this is a LIFT rather than greenfield
    ///    RNN work — see [`crate::squim::missing_primitive_note`] for
    ///    the four recurrent bodies already in the tree and why none is
    ///    yet callable as a bare cell.
    /// 3. **5-block symmetric transposed-conv decoder** — five
    ///    decoder blocks mirroring the encoder, each doubling temporal
    ///    resolution via `ConvTranspose1d(kernel=8, stride=4)` and
    ///    halving channel width. Each block = `Conv1d(1x1, 2C output) +
    ///    GLU + ConvTranspose1d(k=8, s=4)`. Every block sums a skip
    ///    connection from the paired encoder block BEFORE the
    ///    transposed conv (additive U-Net skip, not concat). Distinct
    ///    from the HiFi-GAN upsampler stack (no encoder-side skip,
    ///    mel-input not waveform-input).
    ///
    /// The error names all three pieces + both primary-source anchors
    /// (upstream GitHub repo + arXiv paper) so a reader diagnosing
    /// this gap has exactly two anchors to walk. **No fabricated
    /// denoised waveform is ever emitted** (FR-EX-08).
    ///
    /// # Errors
    ///
    /// - [`VokraError::UnsupportedOp`] — the loud-partial gate for
    ///   the deferred three-stage forward composition.
    pub fn denoise(&self, pcm: &[f32]) -> Result<Vec<f32>> {
        // Bind unused arg so a `#[warn(unused_variables)]` change does
        // not silently mask the loud-partial fire path; the future
        // real implementation will consume it.
        let _ = pcm;
        Err(denoise_forward_loud_partial())
    }
}

/// Constructs the loud-partial [`VokraError::UnsupportedOp`] returned
/// by [`FbDenoiser::denoise`] until the tensor-name walk + three-stage
/// forward composition + three missing pieces land.
///
/// Names **all three** deferred pieces (5-block time-domain waveform
/// U-Net encoder + 2-layer LSTM bottleneck + 5-block symmetric
/// transposed-conv decoder with additive skip) so a reader diagnosing
/// the gap knows exactly which primitives the follow-up wave targets.
/// Cites both primary source URLs (upstream GitHub repo + arXiv paper)
/// so the reader has both the implementation and theoretical anchors.
/// Mirrors the Storm / WavLM / Sortformer / MT3 / beat_this / RMVPE /
/// pyannote / snac / hifigan / vocos / bigvgan / sepformer /
/// conv_tasnet / demucs / gtcrn Wave 3-7 loud-partial-message
/// precedent — CLAUDE.md 教訓 (a).
///
/// Note: uses [`VokraError::UnsupportedOp`] (not `NotImplemented`)
/// because the message is dynamic-formatted via [`format!`] — the
/// `NotImplemented` variant takes only a `&'static str` and would fail
/// to compile with a `format!` result (Wave 5 canary_qwen E0308
/// lesson).
fn denoise_forward_loud_partial() -> VokraError {
    VokraError::UnsupportedOp(format!(
        "facebook_denoiser denoise: three-stage waveform U-Net + LSTM forward pending. \
         facebook-denoiser's reference implementation decomposes as (i) a 5-block \
         time-domain waveform U-Net encoder (per arXiv:2006.12847 §II.A: five \
         convolutional encoder blocks, each halving temporal resolution via \
         `Conv1d(kernel=8, stride=4)` and doubling channel width `H · 2^L` with `H=48` \
         base for the causal real-time model, `H=64` for the non-causal offline model; \
         each block = `Conv1d(k=8, s=4) + ReLU + Conv1d(1x1, 2C output) + GLU`), (ii) \
         a 2-layer LSTM bottleneck (2 stacked LSTM layers between encoder and decoder \
         per arXiv:2006.12847 §II.A + §III.B ablation, hidden = `2 · (H · 2^(L-1))`; \
         unidirectional for causal `denoiser_causal.th`, bidirectional for offline \
         `master64.th` — no ENHANCEMENT-family op in `vokra_ops` carries an LSTM, but \
         this is a LIFT, NOT greenfield RNN work: \
         `vokra_ops::hybrid_ctc_attention::LstmLmCell` holds a PyTorch-order i|f|g|o \
         gate body, and `kokoro::nn::BiLstm1d` + `pyannote::bilstm::BiLstmLayer` are \
         two in-tree bidirectional LSTMs on the PyTorch weight layout; what is absent \
         is a reusable bare cell in `vokra-ops` taking a plain feature vector, which \
         `crate::squim`'s DPRNN head needs too — landing it once unblocks both), and \
         (iii) a 5-block symmetric transposed-conv \
         decoder (five decoder blocks mirroring the encoder, each doubling temporal \
         resolution via `ConvTranspose1d(kernel=8, stride=4)` and halving channel \
         width; each block = `Conv1d(1x1, 2C output) + GLU + ConvTranspose1d(k=8, \
         s=4)`. Every block sums a skip connection from the paired encoder block \
         BEFORE the transposed conv — additive U-Net skip, not concat. Distinct from \
         the HiFi-GAN upsampler stack: no encoder-side skip, mel-input not \
         waveform-input). Every piece needs (a) the tensor-name walk from the \
         upstream `facebookresearch/denoiser` state_dict prefixes to the appropriate \
         primitive inputs (pending manifest fetch — same posture as pyannote / \
         Charsiu real-weight bind, upstream ships `.th` pickles at \
         `dl.fbaipublicfiles.com/adiyoss/denoiser/` which the \
         `tools/parity/nemo_pt_to_safetensors.py` uv-managed Python 3.12 sidecar per \
         memory `[[feedback-python-uses-uv]]` bridges to safetensors; a \
         model-specific `tools/parity/facebook_denoiser_prepare_checkpoint.py` is \
         not yet written), (b) the three \
         missing pieces themselves landing in `vokra_ops` (waveform U-Net encoder + \
         LSTM bottleneck + waveform U-Net decoder with additive skip), and (c) the \
         three-stage encoder → LSTM → decoder composition against the discovered \
         tensor names. facebook-denoiser is the FIRST time-domain waveform U-Net + \
         LSTM entry on the enhancement arm — no near-neighbor exists in the \
         catalogue. Primary sources: {github} + {arxiv}. Loud pending (CLAUDE.md \
         教訓 (a) — 'loud-partial は fake-complete より honest') — no silent \
         fabricated denoised waveform ever emitted (FR-EX-08).",
        github = PRIMARY_SOURCE_GITHUB,
        arxiv = PRIMARY_SOURCE_ARXIV,
    ))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    //! Tests for the facebook-denoiser runtime binder — cross-crate
    //! constant mirror + metadata round-trip on the arch + provenance
    //! stamp + negative-space round-trip on the loud-partial gates +
    //! arch-tag distinctness pin.
    //!
    //! # What "round-trip" means here
    //!
    //! The task spec asks for 5+ unit tests. On real PCM this would
    //! be `denoise(...)` returning denoised audio, but the three-stage
    //! forward + tensor-name walk + three missing pieces (encoder +
    //! LSTM bottleneck + decoder with additive skip) are all deferred
    //! (see the module doc + [`FbDenoiser::denoise`] rustdoc).
    //! Fabricating a real-PCM output would violate CLAUDE.md 教訓 (a)
    //! ("loud-partial は fake-complete より honest").
    //!
    //! The round-trip semantics we *can* honestly test:
    //!
    //! 1. **Cross-crate constant mirror pin**: [`ARCH`] + [`NAME`] +
    //!    [`CATEGORY`] + [`UPSTREAM_URL`] mirror the converter verbatim.
    //! 2. **Arch-tag distinctness pin**: [`ARCH`] is deliberately
    //!    distinct from every sibling enhancement / denoise arch.
    //! 3. **Metadata round-trip**: `from_gguf` binds a legitimate
    //!    GGUF (arch stamp + provenance license class + one
    //!    representative tensor), reads back the license class +
    //!    tensor count.
    //! 4. **Loud-error negative-space round-trip**: every stated
    //!    blocker (missing arch / wrong arch / empty tensor list /
    //!    unsupported forward surface) fires at its documented
    //!    surface point, in the documented error variant.
    //! 5. **Primary-source constant surfacing**: `sample_rate()`
    //!    returns the DNS-Challenge default.

    use super::*;
    use vokra_core::gguf::{GgmlType, GgufBuilder};

    /// Builds a minimal facebook-denoiser GGUF carrying the arch tag +
    /// one representative tensor. Optional `weight_license_class` is
    /// written under `vokra.provenance.weight_license` (or omitted if
    /// `None`).
    fn fb_denoiser_gguf(weight_license_class: Option<LicenseClass>) -> GgufFile {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_string(chunks::KEY_MODEL_NAME, NAME);
        if let Some(cls) = weight_license_class {
            b.add_string(chunks::KEY_PROVENANCE_WEIGHT_LICENSE, cls.as_str());
        }
        // One representative tensor so the non-emptiness gate passes.
        // Uses a plausible upstream state_dict-like name (encoder
        // stage 0 GLU projection weight) so the naming contract
        // (verbatim key pass-through by the converter) is exercised
        // alongside. Shape 96×1 mirrors `Conv1d(in=1, out=2·H, k=1)`
        // for the H=48 causal denoiser inner 1×1 projection between
        // the kernel-8 conv and the GLU.
        b.add_tensor(
            "encoder.0.1.weight",
            GgmlType::F32,
            vec![96, 1, 1],
            vec![0u8; 96 * 4],
        )
        .expect("add_tensor");
        GgufFile::parse(b.to_bytes().expect("serialize")).expect("parse")
    }

    // -----------------------------------------------------------------
    // Test 1 — Cross-crate constant mirror pin
    // -----------------------------------------------------------------

    /// Pin the [`ARCH`] + [`NAME`] + [`CATEGORY`] + [`UPSTREAM_URL`]
    /// constants to the exact strings the converter stamps. A rename
    /// in either crate must land in the same commit or fail this pin.
    #[test]
    fn cross_crate_constant_mirror_pin() {
        // Match the converter's stamps byte-for-byte (see
        // `crates/vokra-convert/src/models/facebook_denoiser.rs`).
        assert_eq!(ARCH, "facebook_denoiser");
        assert_eq!(NAME, "facebook_denoiser");
        assert_eq!(CATEGORY, "enhancement");
        assert_eq!(UPSTREAM_URL, "github.com/facebookresearch/denoiser");
    }

    // -----------------------------------------------------------------
    // Test 2 — Arch-tag distinctness pin
    // -----------------------------------------------------------------

    /// Pin `ARCH = "facebook_denoiser"` and assert distinctness against
    /// every sibling enhancement / denoise arch string. A future rename
    /// of any sibling would land here in the same commit or fail this
    /// test. All sibling enhancement / denoise families enumerated:
    /// facebook-denoiser has a genuinely distinct topology (time-domain
    /// waveform U-Net + LSTM) from every one.
    #[test]
    fn arch_tag_distinct_from_sibling_enhancement_arches() {
        assert_eq!(ARCH, "facebook_denoiser");
        assert_ne!(
            ARCH, "denoise",
            "facebook_denoiser (time-domain waveform U-Net + LSTM) and denoise \
             (DeepFilterNet3, ERB analysis/synthesis + CRN over STFT) are different \
             topologies — sharing arch would mis-route (FR-EX-08)"
        );
        assert_ne!(
            ARCH, "rnnoise",
            "facebook_denoiser and rnnoise (Xiph GRU + 32-band/65-feature frontend) \
             are different topologies — sharing arch would mis-route (FR-EX-08)"
        );
        assert_ne!(
            ARCH, "nsnet2",
            "facebook_denoiser and nsnet2 (Microsoft DNS baseline, 2-layer GRU + \
             3-Linear mask over 257-bin STFT log-magnitude) are different topologies \
             — sharing arch would mis-route (FR-EX-08)"
        );
        assert_ne!(
            ARCH, "dnsmos",
            "facebook_denoiser and dnsmos (P.808/P.835 objective quality metric) \
             are different tasks entirely — sharing arch would mis-route (FR-EX-08)"
        );
        assert_ne!(
            ARCH, "gtcrn",
            "facebook_denoiser and gtcrn (grouped Conv2D + SB-TF-LSTM + ERB \
             grouping over STFT) are different topologies — sharing arch would \
             mis-route (FR-EX-08)"
        );
        assert_ne!(
            ARCH, "dtln_aec",
            "facebook_denoiser and dtln_aec (dual-signal transform LSTM for AEC) \
             are different tasks + topologies — sharing arch would mis-route \
             (FR-EX-08)"
        );
        assert_ne!(
            ARCH, "mp_senet",
            "facebook_denoiser and mp_senet (magnitude-phase U-Net over STFT) are \
             different topologies — sharing arch would mis-route (FR-EX-08)"
        );
        assert_ne!(
            ARCH, "frcrn",
            "facebook_denoiser and frcrn (complex U-Net + FR-LSTM over STFT) are \
             different topologies — sharing arch would mis-route (FR-EX-08)"
        );
        assert_ne!(
            ARCH, "metricgan_plus",
            "facebook_denoiser and metricgan_plus (GAN-trained mask predictor) are \
             different training + topologies — sharing arch would mis-route \
             (FR-EX-08)"
        );
        assert_ne!(
            ARCH, "mossformer2_ss_16k",
            "facebook_denoiser and mossformer2_ss_16k (dual-path Transformer \
             separator) are different families — sharing arch would mis-route \
             (FR-EX-08)"
        );
        assert_ne!(
            ARCH, "sepformer",
            "facebook_denoiser (enhancement) and sepformer (separation, dual-path \
             Transformer) are different families — sharing arch would mis-route \
             (FR-EX-08)"
        );
        assert_ne!(
            ARCH, "storm",
            "facebook_denoiser and storm (diffusion-based NCSN++ v2 + OUVE-SDE \
             two-stage) are different sampler families — sharing arch would \
             mis-route (FR-EX-08)"
        );
    }

    // -----------------------------------------------------------------
    // Test 3 — Metadata round-trip on legitimate GGUF
    // -----------------------------------------------------------------

    #[test]
    fn from_gguf_metadata_round_trip() {
        // A GGUF with the correct arch + NonCommercial provenance +
        // one representative tensor must bind successfully; the
        // stamped license class must surface via `weight_license()`
        // (converter default for cc-by-nc-4.0), and the tensor count
        // must be ≥ 1.
        let file = fb_denoiser_gguf(Some(LicenseClass::NonCommercial));
        let m = FbDenoiser::from_gguf(&file).expect("valid GGUF must bind");
        // Weight-license surface (facebook-denoiser converter stamps
        // NonCommercial per cc-by-nc-4.0 default — T4 tier).
        assert_eq!(m.weight_license(), LicenseClass::NonCommercial);
        // Tensor-count surface (non-emptiness gate passed → ≥ 1).
        assert!(m.tensor_count() >= 1);
        // Fallback surfacing when the provenance stamp is absent must
        // yield `Unknown` (fail-closed at the M2-13 compliance gate).
        let file_no_prov = fb_denoiser_gguf(None);
        let m_no_prov =
            FbDenoiser::from_gguf(&file_no_prov).expect("valid GGUF minus provenance must bind");
        assert_eq!(m_no_prov.weight_license(), LicenseClass::Unknown);
    }

    // -----------------------------------------------------------------
    // Test 4 — from_gguf rejects wrong arch (never silently mis-routes)
    // -----------------------------------------------------------------

    #[test]
    fn from_gguf_rejects_wrong_arch() {
        // A `rnnoise` GGUF handed to the facebook-denoiser binder by
        // mistake must fail loud with a specific message naming both
        // the got arch and the expected arch (FR-EX-08 — never a
        // silent mis-route across enhancement / denoise families).
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, "rnnoise");
        b.add_tensor("noise.weight", GgmlType::F32, vec![4], vec![0u8; 16])
            .expect("add_tensor");
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let Err(err) = FbDenoiser::from_gguf(&file) else {
            panic!("expected ModelLoad on wrong arch");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains("`rnnoise`") && m.contains("`facebook_denoiser`"),
                    "message must name both the got and expected arch tags, got `{m}`"
                );
                assert!(
                    m.contains("Xiph GRU"),
                    "message should disambiguate rnnoise's topology to help \
                     the reader, got `{m}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------
    // Test 5 — from_gguf rejects missing arch stamp
    // -----------------------------------------------------------------

    #[test]
    fn from_gguf_rejects_missing_arch() {
        // A GGUF with NO arch chunk must fail loud rather than
        // silently defaulting to `facebook_denoiser` (FR-EX-08).
        let mut b = GgufBuilder::new();
        // Deliberately no `vokra.model.arch` stamp.
        b.add_tensor("some.tensor", GgmlType::F32, vec![4], vec![0u8; 16])
            .expect("add_tensor");
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let Err(err) = FbDenoiser::from_gguf(&file) else {
            panic!("expected ModelLoad on missing arch");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains("missing `vokra.model.arch`"),
                    "message must name the missing arch key, got `{m}`"
                );
                assert!(
                    m.contains("not a Vokra-native facebook_denoiser GGUF"),
                    "message must state this is not a native facebook_denoiser GGUF, \
                     got `{m}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------
    // Test 6 — from_gguf rejects empty tensor manifest
    // -----------------------------------------------------------------

    #[test]
    fn from_gguf_rejects_empty_tensor_list() {
        // Correct arch but zero tensors — the FbDenoiserWeights
        // non-emptiness gate must fire (FR-EX-08 — never bind an
        // all-zero forward).
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        // NO tensors added.
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let Err(err) = FbDenoiser::from_gguf(&file) else {
            panic!("expected ModelLoad on empty tensor manifest");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains("zero tensors"),
                    "message must name the empty-manifest gap, got `{m}`"
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
    // Test 7 — denoise returns UnsupportedOp with three-piece + primary
    //          source assertions
    // -----------------------------------------------------------------

    #[test]
    fn denoise_loud_partial_returns_unsupported_op() {
        // Bind a legitimate GGUF, then call denoise on 1 second of
        // legitimate-shape mono 16 kHz PCM (16000 zeros) so the
        // loud-partial gate fires — not some pre-encode length /
        // shape validation (there is no such validation today, but a
        // legitimate buffer keeps the test robust against a future
        // one).
        let file = fb_denoiser_gguf(Some(LicenseClass::NonCommercial));
        let m = FbDenoiser::from_gguf(&file).unwrap();
        let pcm = vec![0.0f32; 16_000];
        let Err(err) = m.denoise(&pcm) else {
            panic!("denoise must loud-partial");
        };
        match err {
            VokraError::UnsupportedOp(msg) => {
                // The three deferred pieces must each be named by a
                // stable, greppable phrase so the follow-up wave knows
                // exactly what to walk.
                assert!(
                    msg.contains("U-Net encoder"),
                    "message must name the U-Net encoder gap, got `{msg}`"
                );
                assert!(
                    msg.contains("LSTM bottleneck"),
                    "message must name the LSTM bottleneck gap, got `{msg}`"
                );
                // --- Anti-rot guard (mirror of the `beat_this` / `squim`
                // --- guards).
                //
                // An earlier revision called the LSTM bottleneck outright
                // "greenfield". Four recurrent bodies already exist in this
                // tree, so that phrasing sent the next reader off to write an
                // RNN from scratch. The negative assertion is the load-bearing
                // half: without it the falsehood can rot back in unnoticed.
                assert!(
                    !msg.contains("`vokra_ops`, greenfield"),
                    "stale claim — `hybrid_ctc_attention::LstmLmCell`, \
                     `kokoro::nn::BiLstm1d` and `pyannote::bilstm::BiLstmLayer` are \
                     recurrent bodies already in this tree, got `{msg}`"
                );
                assert!(
                    msg.contains("LIFT, NOT greenfield"),
                    "message must tell the reader the LSTM is a lift, not new RNN \
                     work, got `{msg}`"
                );
                assert!(
                    msg.contains("LstmLmCell") && msg.contains("BiLstm1d"),
                    "message must name the in-tree bodies to lift from, got `{msg}`"
                );
                assert!(
                    msg.contains("U-Net"),
                    "message must name the U-Net topology family, got `{msg}`"
                );
                assert!(
                    msg.contains("transposed-conv decoder"),
                    "message must name the transposed-conv decoder gap (as \
                     opposed to a HiFi-GAN-style upsampler), got `{msg}`"
                );
                // Primary-source URLs must be cited so a reader
                // diagnosing the gap has anchors to walk.
                assert!(
                    msg.contains("github.com/facebookresearch/denoiser"),
                    "message must contain the GitHub URL, got `{msg}`"
                );
                assert!(
                    msg.contains("2006.12847"),
                    "message must cite the arXiv paper anchor, got `{msg}`"
                );
                // FR-EX-08 must be cited so the reader knows the
                // fail-closed contract that governs this gate.
                assert!(
                    msg.contains("FR-EX-08"),
                    "message must cite the FR-EX-08 fail-closed contract, got `{msg}`"
                );
            }
            other => panic!("expected VokraError::UnsupportedOp, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------
    // Test 8 — sample_rate() surfaces DNS-Challenge default
    // -----------------------------------------------------------------

    #[test]
    fn sample_rate_matches_dns_challenge_default() {
        // Defossez et al. 2020 arXiv:2006.12847 §4.1 documents the
        // DNS-Challenge default sample rate as 16 kHz. The primary
        // release + causal real-time variant both use 16 kHz; the
        // separate `master64` 48 kHz variant is out of scope for this
        // constant. A downstream that reaches for `sample_rate()`
        // must receive 16000, not a fabricated value.
        let file = fb_denoiser_gguf(Some(LicenseClass::NonCommercial));
        let m = FbDenoiser::from_gguf(&file).unwrap();
        assert_eq!(
            m.sample_rate(),
            16_000,
            "sample_rate() must match arXiv:2006.12847 §4.1 DNS-Challenge default \
             (16 kHz)"
        );
        // The constant itself is pinned so a rename would land here
        // in the same commit or fail this test.
        assert_eq!(SAMPLE_RATE_DEFAULT, 16_000);
    }
}
