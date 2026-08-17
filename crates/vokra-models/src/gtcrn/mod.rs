//! **GTCRN** (`Xiaobin-Rong/gtcrn`, MIT) — Groupwise Temporal
//! Convolutional Recurrent Network speech enhancement runtime binder
//! for the `gtcrn` converter arch (Wave 6 2026-08-14 audit follow-up,
//! denoise alternative sibling to DFN3 / NSNet2 / RNNoise).
//!
//! # Primary source
//!
//! - Paper: Rong et al. arXiv:2211.02063
//!   *"GTCRN: A Speech Enhancement Model Requiring Ultralow Computational
//!   Resources"*.
//! - Reference implementation:
//!   <https://github.com/Xiaobin-Rong/gtcrn>
//! - Weight license: **MIT** per upstream repo LICENSE
//!   (`github.com/Xiaobin-Rong/gtcrn/blob/main/LICENSE`, per task scout
//!   input 2026-08-14 — owner must primary-source confirm at sign-off
//!   time).
//!
//! # Runtime layout (loud-partial, sepformer / conv_tasnet / demucs
//! separation-fleet posture per CLAUDE.md 教訓 (a))
//!
//! ```text
//! Mixture PCM (mono f32, 16 kHz per [`GtcrnConfig::sample_rate`])
//!   -> STFT (n_fft=512, hop=256, center)          [already covered by
//!                                                  `vokra_ops::stft`]
//!   -> grouped 2D Conv encoder                    ← **loud-partial**
//!        (channel-grouped 2D convolutions over the log-magnitude
//!         STFT. `vokra_ops` exposes NO public 2-D convolution at
//!         all — grouped or un-grouped. Every convolution helper in
//!         the crate is private to the module that owns it:
//!         `denoise.rs` has a DFN3-internal `struct Conv2d`, and
//!         `hifigan` / `hiftnet` / `bigvgan_generator` ship private
//!         1-D helpers. The gap is therefore a 2-D convolution op
//!         *with* grouping as an axis — not a grouping extension to
//!         some existing un-grouped one.)
//!   -> PReLU activations                          ← **loud-partial**
//!        (parameterized ReLU with a learned per-channel scalar
//!         slope. `vokra_ops` has no PReLU — and no public ReLU,
//!         GELU or SiLU cell either: the only public pointwise activation
//!         functions in the crate are `snake_activation_f32` and
//!         `snake_beta_f32`, and GELU exists solely as private
//!         per-module helpers. This is the cheapest of the four
//!         gaps — a two-branch scalar map — and is listed for
//!         completeness, not because it is hard.)
//!   -> ERB (equivalent rectangular bandwidth) grouping ← **loud-partial**
//!        (perceptual band aggregation over the linear STFT bins.
//!         An ERB *partition* helper does exist and is public —
//!         `vokra_ops::DeepFilterNetConfig::erb_widths` — but it is
//!         parameterised by DFN3's own band config and DFN3 uses ERB
//!         as a real analysis / synthesis pair, whereas GTCRN uses it
//!         as an internal efficiency-preserving feature pooler. Reusing
//!         DFN3's partition under GTCRN's axes would be shape-valid and
//!         numerically wrong, i.e. silent — the failure FR-EX-08 exists
//!         to forbid. What is missing is GTCRN's own aggregation matrix,
//!         not the notion of ERB.)
//!   -> SB-TF-LSTM (sub-band time-frequency LSTM) bottleneck
//!                                                 ← **loud-partial**
//!        (dual-axis LSTM composition. The one public LSTM in
//!         `vokra_ops` is `hybrid_ctc_attention::LstmLmCell`, and it
//!         is LM-shaped rather than generic: `step()` consumes a
//!         token id plus a candidate id and returns one
//!         log-probability, and the struct bundles a token embedding
//!         table and a vocab output projection. Running a feature
//!         sequence through it is not possible without rewriting it,
//!         so it is the wrong function here rather than a missing
//!         one. Silero's LSTM is not reusable either: it lives in the
//!         separate `vokra-vad-micro` crate as a `pub(crate)`
//!         `lstm_forward` hard-wired to `HIDDEN = 128`.)
//!   -> grouped 2D Conv decoder + PReLU            ← **loud-partial**
//!        (mirror of the encoder — same grouped Conv2D gap.)
//!   -> per-bin gain mask G ∈ [0, 1]
//!   -> Y = G * X (phase preserved verbatim)
//!   -> iSTFT (streaming, tail buffer)             [already covered by
//!                                                  `vokra_ops::istft_streaming`]
//!   -> denoised PCM
//! ```
//!
//! # Loud-partial classification (design § — CLAUDE.md 教訓 (a))
//!
//! - **Real (this WP)**: [`Gtcrn::from_gguf`] with strict
//!   `vokra.model.arch == "gtcrn"` validation + strict
//!   `vokra.gtcrn.*` chunk-group presence enforcement (every axis
//!   required — no primary-source constant fallback because the
//!   converter transcribes the axes from arXiv:2211.02063 §III and
//!   stamps them, and this binder mirrors those stamps rather than
//!   silently defaulting to a fabricated axis),
//!   [`GtcrnWeights::from_gguf`] with a floor of non-empty tensor
//!   count enforced loud (a GGUF that carries zero tensors is refused
//!   rather than silently running an all-zero forward — FR-EX-08),
//!   and weight-license class surfacing (defaults to
//!   [`LicenseClass::Unknown`] on a stamp-free fixture, fail-closed at
//!   the M2-13 compliance gate — the converter stamps
//!   [`LicenseClass::Permissive`] in production per the MIT default).
//! - **Loud-partial (this WP)**: [`Gtcrn::denoise`] returns
//!   [`VokraError::UnsupportedOp`] naming **all four** deferred
//!   primitives from the encoder-bottleneck-decoder decomposition:
//!   (i) **2-D convolution with grouping** (`vokra_ops` has no public
//!   2-D convolution at all, grouped or un-grouped — every conv
//!   helper in the crate is private to its owning module),
//!   (ii) **PReLU** (no PReLU cell — and no public ReLU / GELU / SiLU
//!   cell either; the only public pointwise activation functions are
//!   `snake_activation_f32` / `snake_beta_f32`),
//!   (iii) **SB-TF-LSTM** (the one public LSTM,
//!   `vokra_ops::hybrid_ctc_attention::LstmLmCell`, is LM-shaped —
//!   token id in, one log-probability out, embedding + vocab
//!   projection bundled in — so it is the wrong function here, and
//!   Silero's is a `pub(crate)` fixed-width cell in the separate
//!   `vokra-vad-micro` crate), and
//!   (iv) **ERB grouping** (the public partition helper
//!   `vokra_ops::DeepFilterNetConfig::erb_widths` carries DFN3's band
//!   config and DFN3's analysis / synthesis semantics; GTCRN's
//!   internal-pooler aggregation matrix is what is absent).
//!   The error cites both primary sources (upstream GitHub repo
//!   README + arXiv:2211.02063) and echoes every config axis so a
//!   reader diagnosing this gap has exactly two anchors to walk and
//!   knows the topology the follow-up wave targets.
//!
//! Rationale (RMVPE / pyannote / hifigan / vocos / bigvgan / snac /
//! beat_this / mt3 / redimnet / sortformer / sepformer / conv_tasnet /
//! demucs Wave 1-5 loud-partial precedent, CLAUDE.md 教訓 (a) —
//! "loud-partial は fake-complete より honest"): the surrounding
//! scaffold + `from_gguf` chunk-group validation + FR-EX-08 loud-fails
//! land today so a follow-up wave can flip the switch by (i) landing
//! the tensor-name walk against a real GTCRN state_dict (the release
//! ships PyTorch state dict on GitHub that
//! `tools/parity/nemo_pt_to_safetensors.py` uv-managed Python 3.12
//! sidecar bridges to safetensors), (ii) landing the four missing
//! primitives in `vokra_ops` (a 2-D convolution op carrying grouping
//! as an axis + PReLU + a generic sequence LSTM + the GTCRN ERB
//! aggregation matrix), and (iii) composing the
//! encoder-bottleneck-decoder forward against the stamped
//! `vokra.gtcrn.*` axes.
//!
//! # `vokra.gtcrn.*` chunk group (read here)
//!
//! Written by `vokra-convert::models::gtcrn::convert_gtcrn_file`:
//!
//! - `vokra.model.arch` (`String`): must equal [`ARCH`] (`"gtcrn"`).
//!   Distinct from every sibling denoise / separator arch
//!   (`denoise` (DFN3), `rnnoise`, `nsnet2`, `dnsmos`,
//!   `metricgan_plus`, `mp_senet_dns`, `sepformer`, `conv_tasnet`,
//!   `demucs`, `frcrn`, `mossformer2_ss_16k`, `facebook_denoiser`) —
//!   silently sharing would misroute runtime dispatch (FR-EX-08).
//! - `vokra.model.name` (`String`): `"gtcrn"` — auxiliary check.
//! - `vokra.model.category` (`String`): `"enhancement"` (single-mask
//!   denoise head, mirror of sibling DFN3 / NSNet2 posture).
//! - `vokra.gtcrn.{sample_rate, n_fft, hop, n_bands, gru_hidden}`
//!   (`u32` each): the 5-axis topology from arXiv:2211.02063 §III.
//!   Read strict — a partially-stamped GGUF is caught here rather
//!   than silently defaulting to a fabricated axis.
//! - `vokra.provenance.*`: license class + raw license string, so the
//!   runtime compliance gate (FR-CP-03) can classify the artifact
//!   without re-inspecting the safetensors provenance. Defaults to
//!   `Permissive` in production per the MIT stamp; missing provenance
//!   falls back to `Unknown` (fail-closed at the M2-13 gate).
//!
//! # Cross-crate constant duplication
//!
//! Mirror of the converter's [`ARCH`] / [`KEY_GTCRN_*`] — same rule
//! the sibling BF16 pass-through binders (`pyannote` / `snac` /
//! `hifigan` / `beat_this` / `mt3` / `redimnet` /
//! `sortformer_diar_4spk_v1` / `sepformer` / `conv_tasnet` /
//! `demucs`) use so `vokra-models` does not gain a dependency edge
//! onto `vokra-convert`, preserving the layered convention
//! `vokra-ops → nothing GGUF-aware`, `vokra-core → GGUF reader`,
//! `vokra-models → GGUF binder`, `vokra-convert → GGUF writer`. A
//! `[test]` at the bottom of this module pins the mirror so a
//! converter-side rename lands here in the same commit or fails the
//! pin.
//!
//! # Family posture — distinct from every sibling enhancement / separator arch
//!
//! [`ARCH`] = `"gtcrn"` is **deliberately distinct** from every sibling
//! enhancement / separator arch tag; a downstream binder that silently
//! aliases would attempt to walk a GTCRN checkpoint through a
//! wrong-topology loader:
//!
//! - `denoise` — DeepFilterNet3 (ERB analysis / synthesis + CRN — a
//!   different ERB posture: DFN3 uses a real ERB analysis / synthesis
//!   pair around a CRN, while GTCRN uses ERB grouping only for
//!   feature aggregation over a grouped Conv2D backbone; the two ERB
//!   usages are NOT compatible);
//! - `rnnoise` — Xiph RNNoise (GRU + BSD BFCC / Bark features);
//! - `nsnet2` — Microsoft DNS baseline (2-layer GRU + 3-Linear mask
//!   over 257-bin STFT log-magnitude — same STFT frontend but a
//!   fundamentally different mask predictor topology);
//! - `dnsmos` — Microsoft P.808 / P.835 DNSMOS objective quality
//!   estimator (a metric, not a denoiser);
//! - `metricgan_plus`, `mp_senet_dns`, `frcrn`, `facebook_denoiser`,
//!   `mossformer2_ss_16k` — other enhancement variants with distinct
//!   topologies;
//! - `sepformer`, `conv_tasnet`, `demucs`, `tiger_separator`,
//!   `bs_roformer`, `mp_senet` — separator families with fundamentally
//!   different masker topologies.
//!
//! Silently sharing arch would let runtime dispatch mis-route a GTCRN
//! checkpoint onto a wrong-topology loader — FR-EX-08 forbids the
//! silent shape misroute across enhancement / separation families.
//!
//! # No ONNX / no pickle (permanent)
//!
//! GTCRN ships as PyTorch state dict upstream; this runtime **never**
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
// `crates/vokra-convert/src/models/gtcrn.rs`. See module docstring for
// the cross-crate duplication rationale.
// ---------------------------------------------------------------------------

/// Expected `vokra.model.arch` value written by
/// `vokra-cli convert --model gtcrn`.
///
/// Distinct from every sibling denoise / separator arch tag —
/// `denoise` (DFN3), `rnnoise`, `nsnet2`, `dnsmos`, `metricgan_plus`,
/// `mp_senet_dns`, `sepformer`, `conv_tasnet`, `demucs`, `frcrn`,
/// `mossformer2_ss_16k`, `facebook_denoiser`. Silently sharing an
/// arch would misroute runtime dispatch (FR-EX-08). Version-neutral
/// (GTCRN ships a single 16 kHz release; sibling variants would keep
/// the tag and pick up distinct [`NAME`] stamps).
pub const ARCH: &str = "gtcrn";

/// Expected `vokra.model.name` value — matches the `vokra/gtcrn`
/// publish slug (when it lands under the T1 Permissive distribution
/// gate after §3.1 sign-off).
pub const NAME: &str = "gtcrn";

/// Expected `vokra.model.category` value — single-mask enhancement
/// head. Mirror of sibling `denoise` (DFN3) / `nsnet2` / `rnnoise`
/// enhancement family posture. Distinct from separator families
/// (`sepformer` / `conv_tasnet` / `demucs`) which carry
/// `category = "separation"` for multi-source outputs.
pub const CATEGORY: &str = "enhancement";

/// `vokra.gtcrn.sample_rate` — PCM sample rate Hz (typical GTCRN =
/// 16000 per arXiv:2211.02063 §III).
pub const KEY_GTCRN_SAMPLE_RATE: &str = "vokra.gtcrn.sample_rate";
/// `vokra.gtcrn.n_fft` — STFT window size (typical GTCRN = 512 per
/// arXiv:2211.02063 §III).
pub const KEY_GTCRN_N_FFT: &str = "vokra.gtcrn.n_fft";
/// `vokra.gtcrn.hop` — STFT hop in samples (typical GTCRN = 256 =
/// 16 ms at 16 kHz per arXiv:2211.02063 §III — a longer hop than
/// NSNet2's 10 ms per GTCRN's low-latency streaming budget).
pub const KEY_GTCRN_HOP: &str = "vokra.gtcrn.hop";
/// `vokra.gtcrn.n_bands` — STFT bin count (= `n_fft/2 + 1`, typical
/// GTCRN = 257 per arXiv:2211.02063 §III).
pub const KEY_GTCRN_N_BANDS: &str = "vokra.gtcrn.n_bands";
/// `vokra.gtcrn.gru_hidden` — sub-band recurrent hidden width
/// (typical GTCRN = 64 per arXiv:2211.02063 §III). The metadata key
/// uses the generic `gru_hidden` label — the upstream sub-band branch
/// is actually **LSTM** per arXiv:2211.02063 §III, not GRU, but the
/// on-disk metadata surface uses the neutral label to avoid promoting
/// an implementation-detail RNN cell kind into the schema (mirror of
/// nsnet2's `KEY_HIDDEN_DIM` posture).
pub const KEY_GTCRN_GRU_HIDDEN: &str = "vokra.gtcrn.gru_hidden";

/// Primary-source anchor: upstream GitHub repository. Cited in the
/// loud-partial error so a reader diagnosing the gap knows the
/// definitive reference implementation source.
const PRIMARY_SOURCE_REPO: &str = "github.com/Xiaobin-Rong/gtcrn";
/// Primary-source anchor: Rong et al. 2022 arXiv paper. Cited
/// alongside the repo anchor so a reader has the theoretical context
/// as well.
const PRIMARY_SOURCE_PAPER: &str = "arxiv.org/abs/2211.02063";

// ---------------------------------------------------------------------------
// GtcrnConfig — the topology axes read from the `vokra.gtcrn.*` chunk
// group. STRICT: every axis is required (FR-EX-08 — no primary-source
// constant fallback since a partial stamp would fabricate axes without
// primary-source backing; the converter always stamps every axis so a
// proper conversion carries the full group).
// ---------------------------------------------------------------------------

/// GTCRN topology hyperparameters as they ride the `vokra.gtcrn.*`
/// chunk group.
///
/// [`from_gguf`](Self::from_gguf) is a **strict** loader: every axis
/// is required (FR-EX-08 — never a silent primary-source constant
/// fallback because the fallback would fabricate axes the runtime
/// then binds against). A GGUF missing any `vokra.gtcrn.*` chunk is
/// rejected loudly with a [`VokraError::ModelLoad`] naming the absent
/// key.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GtcrnConfig {
    /// PCM sample rate in Hz (typical GTCRN = 16000).
    pub sample_rate: u32,
    /// STFT window size (typical GTCRN = 512).
    pub n_fft: u32,
    /// STFT hop in samples (typical GTCRN = 256 = 16 ms at 16 kHz).
    pub hop: u32,
    /// STFT bin count (typical GTCRN = 257 = `n_fft/2 + 1`).
    pub n_bands: u32,
    /// Sub-band recurrent hidden width (typical GTCRN = 64). The
    /// upstream sub-band branch uses **LSTM** cells per
    /// arXiv:2211.02063 §III; the on-disk key label `gru_hidden` is
    /// generic-RNN-neutral (mirror of nsnet2's `KEY_HIDDEN_DIM`
    /// posture).
    pub gru_hidden: u32,
    /// Model category as a `'static` slice — the converter stamps
    /// [`CATEGORY`] verbatim (`"enhancement"`).
    pub category: &'static str,
}

impl GtcrnConfig {
    /// The typical GTCRN axes transcribed from arXiv:2211.02063 §III
    /// (implementer MUST re-confirm against `github.com/Xiaobin-Rong/
    /// gtcrn` at land time rather than trusting the transcribed
    /// constants alone — CLAUDE.md「ハルシネーション厳禁」).
    ///
    /// Used by the unit tests and as a diagnostic reference. The
    /// runtime loader does NOT default to these; it reads the stamped
    /// values via [`Self::from_gguf`] and fails loud on any missing
    /// chunk.
    #[must_use]
    pub const fn typical_default() -> Self {
        Self::for_stamped_axes(16_000, 512, 256, 257, 64)
    }

    /// Builds a config from caller-supplied axes (used both by the
    /// binder's [`Self::from_gguf`] and by the unit tests). All axes
    /// are `u32`; the category is hard-set to [`CATEGORY`].
    #[must_use]
    pub const fn for_stamped_axes(
        sample_rate: u32,
        n_fft: u32,
        hop: u32,
        n_bands: u32,
        gru_hidden: u32,
    ) -> Self {
        Self {
            sample_rate,
            n_fft,
            hop,
            n_bands,
            gru_hidden,
            category: CATEGORY,
        }
    }

    /// Reads every `vokra.gtcrn.*` chunk from `gguf`. Missing axis =
    /// loud [`VokraError::ModelLoad`] naming the absent key (FR-EX-08
    /// — no primary-source constant fallback).
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] when any of the 5 mandatory
    ///   `vokra.gtcrn.*` u32 chunks is absent.
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        fn req_u32(gguf: &GgufFile, key: &str) -> Result<u32> {
            gguf.get(key)
                .and_then(|v| v.as_u64())
                .map(|v| v as u32)
                .ok_or_else(|| {
                    VokraError::ModelLoad(format!(
                        "gtcrn: GGUF is missing required u32 chunk `{key}` — the \
                         upstream `Xiaobin-Rong/gtcrn` release ships a single canonical \
                         config and the converter transcribes every axis from \
                         arXiv:2211.02063 §III and stamps them, so a proper conversion \
                         carries the full `vokra.gtcrn.*` chunk group. This binder \
                         refuses to fabricate topology axes from primary-source \
                         constants (FR-EX-08). Re-run `vokra-cli convert --model gtcrn` \
                         against a safetensors checkpoint flattened via \
                         `tools/parity/nemo_pt_to_safetensors.py`."
                    ))
                })
        }
        Ok(Self::for_stamped_axes(
            req_u32(gguf, KEY_GTCRN_SAMPLE_RATE)?,
            req_u32(gguf, KEY_GTCRN_N_FFT)?,
            req_u32(gguf, KEY_GTCRN_HOP)?,
            req_u32(gguf, KEY_GTCRN_N_BANDS)?,
            req_u32(gguf, KEY_GTCRN_GRU_HIDDEN)?,
        ))
    }
}

// ---------------------------------------------------------------------------
// GtcrnWeights — bound the tensor manifest with a non-emptiness gate.
// Under the loud-partial WP the weights are counted but the encoder-
// bottleneck-decoder forward is deferred. Mirror of
// `ConvTasnetWeights` / `SepformerWeights` / `ReDimNetWeights`.
// ---------------------------------------------------------------------------

/// Weight tensors bound from a GTCRN GGUF.
///
/// **Contract**: [`from_gguf`](Self::from_gguf) is a *loud*
/// verification step. A GGUF that carries zero tensors is rejected
/// with [`VokraError::ModelLoad`] (FR-EX-08 — an empty GGUF is never
/// a valid GTCRN checkpoint).
///
/// Under the current landing this struct stores the tensor names +
/// GGUF-side dims discovered on disk. The follow-up wave that lands
/// the encoder-bottleneck-decoder forward sizes its dequant per its
/// kernel needs — today only the count + names are consumed so a
/// future `GtcrnWeights::bind_encoder_bottleneck_decoder_weights`
/// tensor walk can find its inputs without re-parsing the GGUF.
#[derive(Debug)]
pub struct GtcrnWeights {
    /// Tensors discovered on disk, indexed by upstream `state_dict`
    /// name with their GGUF-side dims. Used by the load-time
    /// non-emptiness gate and by the future follow-up
    /// encoder-bottleneck-decoder-forward wave.
    tensors: Vec<(String, Vec<usize>)>,
}

impl GtcrnWeights {
    /// Scans `gguf` for the GTCRN state_dict tensors. Refuses to
    /// bind if the GGUF carries zero tensors (FR-EX-08 — an empty
    /// GGUF is never a valid GTCRN checkpoint).
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
                "gtcrn: GGUF carries zero tensors — refusing to bind an all-zero \
                 forward (FR-EX-08). Re-run `vokra-cli convert --model gtcrn` against \
                 an upstream safetensors checkpoint (upstream ships a PyTorch state \
                 dict which the sibling `tools/parity/nemo_pt_to_safetensors.py` \
                 bridge flattens to safetensors — pickle deserialization inside the \
                 Rust runtime would violate FR-LD-05)."
                    .to_owned(),
            ));
        }
        Ok(Self { tensors })
    }

    /// Number of tensors bound from the GGUF. Purely a diagnostic
    /// accessor — the encoder-bottleneck-decoder-forward wave uses it
    /// to size its expectations.
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }
}

// ---------------------------------------------------------------------------
// Gtcrn — the runtime binder handle
// ---------------------------------------------------------------------------

/// GTCRN runtime binder (`Xiaobin-Rong/gtcrn`, MIT).
///
/// Bind with [`from_gguf`](Self::from_gguf), then call
/// [`denoise`](Self::denoise) on a mixed mono PCM buffer to obtain the
/// enhanced PCM. See the module doc for the current implementation-
/// status matrix and the FR-EX-08 loud-error contract on the encoder-
/// bottleneck-decoder composition.
#[derive(Debug)]
pub struct Gtcrn {
    config: GtcrnConfig,
    // The bound weights are held (real, counted) but the encoder-
    // bottleneck-decoder composition is a follow-up wave; the field
    // is deliberately `#[allow(dead_code)]` until the composition
    // lands so a reader is not misled by an unused field. Same
    // posture as RMVPE / pyannote / mt3 / beat_this / sortformer /
    // sepformer / conv_tasnet / demucs / redimnet.
    #[allow(dead_code)]
    weights: GtcrnWeights,
    weight_license: LicenseClass,
}

impl Gtcrn {
    /// Binds a GTCRN GGUF: validates arch, reads the strict topology
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
    ///   or not `"gtcrn"` (a sibling denoise / separator GGUF handed
    ///   to us by mistake fails with a clear message naming every
    ///   sibling arch rather than a downstream "vokra.gtcrn.n_fft
    ///   missing" — same pattern as `Mt3::from_gguf` /
    ///   `ConvTasnet::from_gguf` / `SepFormer::from_gguf`).
    /// - [`VokraError::ModelLoad`] when any `vokra.gtcrn.*` chunk is
    ///   absent ([`GtcrnConfig::from_gguf`] is strict — no
    ///   primary-source constant fallback).
    /// - [`VokraError::ModelLoad`] when the GGUF carries zero tensors
    ///   ([`GtcrnWeights::from_gguf`] refuses to bind an all-zero
    ///   forward).
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        // 1. Arch check — always first so a mis-typed model handed
        //    here fails with a specific message instead of a
        //    downstream "vokra.gtcrn.sample_rate missing" error.
        match file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()) {
            Some(a) if a == ARCH => {}
            Some(other) => {
                return Err(VokraError::ModelLoad(format!(
                    "gtcrn: GGUF arch is `{other}`, expected `{ARCH}` (was this GGUF \
                     produced by `vokra-cli convert --model gtcrn`? Note that sibling \
                     denoise / separator arches — `denoise` (DeepFilterNet3, ERB \
                     analysis/synthesis + CRN), `rnnoise` (Xiph GRU + BFCC), `nsnet2` \
                     (Microsoft DNS baseline, 2-layer GRU + 3-Linear mask), `dnsmos` \
                     (P.808/P.835 metric only), `metricgan_plus`, `mp_senet_dns`, \
                     `sepformer` (SpeechBrain dual-path Transformer), `conv_tasnet` \
                     (Asteroid dilated TCN), `demucs` (Meta hybrid U-Net + \
                     cross-domain attention) — all have completely different \
                     topologies from GTCRN's grouped Conv2D + SB-TF-LSTM + ERB \
                     grouping stack. Silently aliasing arch would misroute the runtime \
                     dispatch, FR-EX-08.)"
                )));
            }
            None => {
                return Err(VokraError::ModelLoad(
                    "gtcrn: GGUF is missing `vokra.model.arch` (converter did not \
                     stamp it — this is not a Vokra-native gtcrn GGUF)"
                        .to_owned(),
                ));
            }
        }

        // 2. Strict topology axes from the `vokra.gtcrn.*` chunk group.
        let config = GtcrnConfig::from_gguf(file)?;

        // 3. Load the tensor manifest with the non-emptiness gate.
        let weights = GtcrnWeights::from_gguf(file)?;

        // 4. Provenance surfacing — read the stamped weight-license
        //    class for compliance gate cross-checks. The GTCRN
        //    converter defaults to `Permissive` per the upstream repo
        //    LICENSE `mit`. Missing provenance falls back to `Unknown`
        //    which is fail-closed at the M2-13 compliance gate — same
        //    posture as MT3 / Sortformer / ConvTasnet / SepFormer.
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

    /// Constructs a test-only [`Gtcrn`] with a placeholder tensor
    /// and the typical [`GtcrnConfig`]. Used by structural tests in
    /// this module — production callers reach [`Self::from_gguf`]
    /// instead.
    #[cfg(test)]
    #[must_use]
    pub fn synthesized() -> Self {
        Self {
            config: GtcrnConfig::typical_default(),
            weights: GtcrnWeights {
                tensors: vec![("placeholder.weight".to_owned(), vec![1])],
            },
            weight_license: LicenseClass::Unknown,
        }
    }

    /// The bound topology axes (read from the `vokra.gtcrn.*` chunk
    /// group).
    #[inline]
    #[must_use]
    pub const fn config(&self) -> &GtcrnConfig {
        &self.config
    }

    /// PCM sample rate in Hz (from the stamped
    /// `vokra.gtcrn.sample_rate` chunk).
    #[inline]
    #[must_use]
    pub const fn sample_rate(&self) -> u32 {
        self.config.sample_rate
    }

    /// STFT window size (from the stamped `vokra.gtcrn.n_fft` chunk).
    #[inline]
    #[must_use]
    pub const fn n_fft(&self) -> u32 {
        self.config.n_fft
    }

    /// STFT hop in samples (from the stamped `vokra.gtcrn.hop` chunk).
    #[inline]
    #[must_use]
    pub const fn hop(&self) -> u32 {
        self.config.hop
    }

    /// STFT bin count (from the stamped `vokra.gtcrn.n_bands` chunk).
    #[inline]
    #[must_use]
    pub const fn n_bands(&self) -> u32 {
        self.config.n_bands
    }

    /// Sub-band recurrent hidden width (from the stamped
    /// `vokra.gtcrn.gru_hidden` chunk).
    #[inline]
    #[must_use]
    pub const fn gru_hidden(&self) -> u32 {
        self.config.gru_hidden
    }

    /// Model category — `"enhancement"` for the single-mask denoise
    /// head.
    #[inline]
    #[must_use]
    pub const fn category(&self) -> &'static str {
        self.config.category
    }

    /// The stamped weight-license class surfaced from the GGUF's
    /// `vokra.provenance.weight_license` chunk. The GTCRN converter
    /// stamps `Permissive` by default per the upstream repo LICENSE
    /// `mit` (T1 tier — publish redistributable, no runtime
    /// attribution obligation). A GGUF missing the stamp reads back
    /// as [`LicenseClass::Unknown`] which is also fail-closed at the
    /// M2-13 compliance gate.
    #[inline]
    #[must_use]
    pub const fn weight_license(&self) -> LicenseClass {
        self.weight_license
    }

    /// Number of tensors bound from the GGUF. Purely a diagnostic
    /// accessor — the encoder-bottleneck-decoder-forward wave uses it
    /// to size its expectations.
    #[inline]
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.weights.tensor_count()
    }

    /// Denoises a mixed mono PCM buffer (16 kHz per
    /// [`GtcrnConfig::sample_rate`]) into an enhanced PCM buffer.
    ///
    /// # Loud-partial (this WP)
    ///
    /// Returns [`VokraError::UnsupportedOp`] — GTCRN's inference path
    /// requires **four** deferred primitives from the encoder-
    /// bottleneck-decoder decomposition:
    ///
    /// 1. **2-D convolution with grouping** (channel-grouped
    ///    depthwise-style convolutions over log-magnitude STFT).
    ///    `vokra_ops` exposes no public 2-D convolution whatsoever —
    ///    grouped or un-grouped. Every convolution helper in the
    ///    crate is private to the module that owns it (`denoise.rs`
    ///    holds a DFN3-internal `struct Conv2d`; `hifigan` /
    ///    `hiftnet` / `bigvgan_generator` hold private 1-D helpers),
    ///    so this is a new op rather than an extension of an
    ///    existing one.
    /// 2. **PReLU** (parameterized ReLU with a learned per-channel
    ///    scalar slope). `vokra_ops` has no PReLU cell, and no
    ///    public ReLU / GELU / SiLU cell either — the only public
    ///    activation functions are `snake_activation_f32` and
    ///    `snake_beta_f32`, with GELU present only as private
    ///    per-module helpers. This is the cheapest of the four gaps.
    /// 3. **SB-TF-LSTM** (sub-band time-frequency LSTM bottleneck).
    ///    The one public LSTM in `vokra_ops` is
    ///    `hybrid_ctc_attention::LstmLmCell`, which is LM-shaped:
    ///    `step()` takes a token id plus a candidate id and returns a
    ///    single log-probability, and the struct bundles an embedding
    ///    table and a vocab projection. It is the wrong function for
    ///    a feature-sequence bottleneck rather than a missing one.
    ///    Silero's LSTM is a `pub(crate)` `lstm_forward` hard-wired to
    ///    `HIDDEN = 128` in the separate `vokra-vad-micro` crate.
    /// 4. **ERB (equivalent rectangular bandwidth) frequency-band
    ///    grouping**. The public partition helper
    ///    `vokra_ops::DeepFilterNetConfig::erb_widths` exists, but it
    ///    carries DFN3's band config and DFN3 uses ERB as an
    ///    analysis / synthesis pair while GTCRN uses it as an internal
    ///    feature pooler. Aliasing the two would be shape-valid and
    ///    numerically wrong, i.e. silent (FR-EX-08). GTCRN's own
    ///    aggregation matrix is what is absent.
    ///
    /// The error names all four primitives + both primary-source
    /// anchors (upstream repo + arXiv paper) so a reader diagnosing
    /// this gap has exactly two places to walk. Every config axis is
    /// echoed so the reader can cross-check what topology the
    /// follow-up wave targets. **No fabricated denoised waveform is
    /// ever emitted** (FR-EX-08).
    ///
    /// # Errors
    ///
    /// - [`VokraError::UnsupportedOp`] — the loud-partial gate for
    ///   the deferred encoder-bottleneck-decoder composition.
    pub fn denoise(&self, mixed_pcm: &[f32]) -> Result<Vec<f32>> {
        // Bind unused arg so a `#[warn(unused_variables)]` change does
        // not silently mask the loud-partial fire path; the future
        // real implementation will consume it.
        let _ = mixed_pcm;
        Err(denoise_forward_loud_partial(&self.config))
    }
}

/// Constructs the loud-partial [`VokraError::UnsupportedOp`] returned
/// by [`Gtcrn::denoise`] until the tensor-name walk + encoder-
/// bottleneck-decoder composition + four missing primitives land.
///
/// Names **all four** deferred primitives (2-D convolution with
/// grouping + PReLU + SB-TF-LSTM + ERB grouping) so a reader
/// diagnosing the gap knows exactly which `vokra_ops` extensions are
/// required. Cites both primary source URLs (upstream repo README +
/// arXiv paper) so the reader has both the implementation and
/// theoretical anchors. Mirrors the Sortformer / MT3 / beat_this /
/// RMVPE / pyannote / snac / hifigan / vocos / bigvgan / sepformer /
/// conv_tasnet / demucs Wave 3-5 loud-partial-message precedent —
/// CLAUDE.md 教訓 (a).
///
/// Every clause here is checked against `crates/vokra-ops/src` rather
/// than restated from a neighbouring comment. An earlier revision
/// claimed the grouped conv was "NOT covered by `vokra_ops::conv2d`
/// which handles only the un-grouped case" and that "`vokra_ops` has
/// plain ReLU / GELU / SiLU but no PReLU cell". Neither held: there is
/// no `conv2d` module and no public 2-D convolution of any kind, and
/// the only public pointwise activation functions are `snake_activation_f32` /
/// `snake_beta_f32`. A message that names a primitive which does not
/// exist is worse than no message — it sends the next reader to a
/// module they cannot find, and the second claim reached the right
/// conclusion (no PReLU) from a false premise, which is the kind of
/// accident that stops being right the moment anything nearby moves.
/// The test `denoise_loud_partial_names_only_real_primitives` asserts
/// the stale phrasing is ABSENT as well as asserting the live blockers
/// are present, so it cannot rot back in (mirror of the `beat_this`
/// guard).
///
/// Echoes every [`GtcrnConfig`] axis so the reader can cross-check
/// what topology the follow-up wave targets.
///
/// Note: uses [`VokraError::UnsupportedOp`] (not `NotImplemented`)
/// because the message is dynamic-formatted via [`format!`] — the
/// `NotImplemented` variant takes only a `&'static str` and would
/// fail to compile with a `format!` result (Wave 5 canary_qwen
/// E0308 lesson).
fn denoise_forward_loud_partial(cfg: &GtcrnConfig) -> VokraError {
    VokraError::UnsupportedOp(format!(
        "gtcrn denoise: grouped Conv2D encoder + PReLU + SB-TF-LSTM (sub-band \
         time-frequency LSTM) bottleneck + ERB (equivalent rectangular bandwidth) \
         frequency-band grouping + grouped Conv2D decoder composition pending. \
         ALREADY RESOLVED, do not re-report — the STFT / iSTFT framing this model \
         needs is supplied by `vokra_ops::stft` and `vokra_ops::istft_streaming`, and \
         the `vokra.gtcrn.*` group is read strictly. GTCRN's reference implementation \
         then decomposes as (a) a grouped 2D Conv encoder (channel-grouped \
         depthwise-style convolutions over the log-magnitude STFT). `vokra_ops` \
         exposes NO public 2-D convolution AT ALL — grouped or un-grouped: every \
         convolution helper in the crate is private to the module that owns it \
         (`denoise.rs` holds a DFN3-internal `struct Conv2d`; `hifigan` / `hiftnet` / \
         `bigvgan_generator` hold private 1-D helpers), so this is a NEW op, not a \
         grouping extension to an existing un-grouped one. (b) PReLU activations \
         (parameterized ReLU with a learned per-channel scalar slope). There is no \
         PReLU cell, and no public ReLU / GELU / SiLU cell either — the only public \
         POINTWISE activation functions in `vokra_ops` are `snake_activation_f32` and \
         `snake_beta_f32`, GELU existing solely as private per-module helpers. This \
         is the CHEAPEST of the four gaps (a two-branch scalar map), listed for \
         completeness rather than difficulty. (c) an SB-TF-LSTM bottleneck (sub-band \
         time-frequency LSTM composition). The one public LSTM in `vokra_ops` is \
         `vokra_ops::hybrid_ctc_attention::LstmLmCell`, which is LM-shaped: `step()` \
         consumes a token id plus a candidate id and returns a single \
         log-probability, and the struct bundles a token embedding table and a vocab \
         output projection — it is the WRONG FUNCTION for a feature-sequence \
         bottleneck rather than a missing one, and substituting it is not even \
         shape-valid. Silero's LSTM is not reusable either: it is a `pub(crate)` \
         `lstm_forward` hard-wired to `HIDDEN = 128` living in the separate \
         `vokra-vad-micro` crate (NOT in `silero_vad`, which is only a std veneer \
         over it). (d) ERB frequency-band grouping. The public ERB partition helper \
         `vokra_ops::DeepFilterNetConfig::erb_widths` DOES exist, but it carries \
         DFN3's own band config and DFN3 uses ERB as a real analysis / synthesis \
         stage whereas GTCRN uses it as an internal feature aggregator; reusing it \
         under GTCRN's axes would be shape-valid and numerically wrong, i.e. silent. \
         What is absent is GTCRN's own aggregation matrix, not the notion of ERB. \
         (e) a grouped 2D Conv decoder + PReLU mirror of the encoder. Every piece \
         needs (i) the tensor-name walk from the upstream `Xiaobin-Rong/gtcrn` \
         state_dict prefixes to the appropriate primitives' inputs (pending the \
         manifest fetch — same posture as pyannote / Charsiu real-weight bind), \
         (ii) the missing primitives themselves landing in `vokra_ops` (2-D \
         convolution with grouping + PReLU + a generic sequence LSTM + the GTCRN ERB \
         aggregation matrix), and (iii) the encoder-bottleneck-decoder block \
         composition itself. Config: \
         sample_rate={sample_rate}, n_fft={n_fft}, hop={hop}, n_bands={n_bands}, \
         gru_hidden={gru_hidden}, category={category}. Primary sources: {repo} + \
         {paper}. Loud pending (CLAUDE.md 教訓 (a) — 'loud-partial は fake-complete \
         より honest') — no silent fabricated denoised waveform ever emitted \
         (FR-EX-08).",
        sample_rate = cfg.sample_rate,
        n_fft = cfg.n_fft,
        hop = cfg.hop,
        n_bands = cfg.n_bands,
        gru_hidden = cfg.gru_hidden,
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
    //! Tests for the GTCRN runtime binder — cross-crate constant mirror
    //! + config default pin + full topology round-trip on the strict
    //!   chunk group + negative-space round-trip on the loud-partial
    //!   gates + arch-tag distinctness pin.
    //!
    //! # What "round-trip" means here
    //!
    //! The task spec asks for 5+ unit tests. On real PCM this would
    //! be `denoise(...)` returning enhanced audio, but the encoder-
    //! bottleneck-decoder forward + tensor-name walk + four missing
    //! primitives (grouped Conv2D + PReLU + SB-TF-LSTM + ERB grouping)
    //! are all deferred (see the module doc + [`Gtcrn::denoise`]
    //! rustdoc). Fabricating a real-PCM output would violate
    //! CLAUDE.md 教訓 (a) ("loud-partial は fake-complete より
    //! honest").
    //!
    //! The round-trip semantics we *can* honestly test:
    //!
    //! 1. **Cross-crate constant mirror pin**: [`ARCH`] +
    //!    [`KEY_GTCRN_*`] (5 axes) + [`CATEGORY`] mirror the converter
    //!    verbatim.
    //! 2. **Config default pin**: [`GtcrnConfig::typical_default`]
    //!    matches the primary-source-transcribed axes.
    //! 3. **Synthesized round-trip**: [`Gtcrn::synthesized`] yields
    //!    the expected accessor values.
    //! 4. **Metadata round-trip**: `from_gguf` binds a legitimate
    //!    GGUF (arch + name + category + full 5-axis chunk group +
    //!    provenance license + one representative tensor), reads back
    //!    every axis + license class + tensor count.
    //! 5. **Loud-error negative-space round-trip**: every stated
    //!    blocker (missing arch / wrong arch / missing chunk / empty
    //!    tensor list / unsupported forward surface) fires at its
    //!    documented surface point, in the documented error variant.
    //! 6. **Arch-tag distinctness pin**: [`ARCH`] is deliberately
    //!    distinct from every sibling denoise / separator arch.

    use super::*;
    use vokra_core::gguf::{GgmlType, GgufBuilder};

    /// Builds a minimal GTCRN GGUF carrying the arch tag + name +
    /// category + full `vokra.gtcrn.*` chunk group + one
    /// representative tensor. Optional `weight_license_class` is
    /// written under `vokra.provenance.weight_license` (or omitted
    /// if `None`).
    fn gtcrn_gguf(cfg: GtcrnConfig, weight_license_class: Option<LicenseClass>) -> GgufFile {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_string(chunks::KEY_MODEL_NAME, NAME);
        b.add_string("vokra.model.category", CATEGORY);
        b.add_u32(KEY_GTCRN_SAMPLE_RATE, cfg.sample_rate);
        b.add_u32(KEY_GTCRN_N_FFT, cfg.n_fft);
        b.add_u32(KEY_GTCRN_HOP, cfg.hop);
        b.add_u32(KEY_GTCRN_N_BANDS, cfg.n_bands);
        b.add_u32(KEY_GTCRN_GRU_HIDDEN, cfg.gru_hidden);
        if let Some(cls) = weight_license_class {
            b.add_string(chunks::KEY_PROVENANCE_WEIGHT_LICENSE, cls.as_str());
        }
        // One representative tensor so the non-emptiness gate passes.
        // Uses a plausible upstream state_dict-like name (encoder-side
        // grouped Conv2D block) so the naming contract (verbatim key
        // pass-through by the converter) is exercised alongside.
        b.add_tensor(
            "en_conv.0.conv.weight",
            GgmlType::F32,
            vec![16, 1, 3, 3],
            vec![0u8; 16 * 3 * 3 * 4],
        )
        .expect("add_tensor");
        GgufFile::parse(b.to_bytes().expect("serialize")).expect("parse")
    }

    // -----------------------------------------------------------------
    // Test 1 — Cross-crate constant mirror pin
    // -----------------------------------------------------------------

    /// Pin the [`ARCH`] + 5 [`KEY_GTCRN_*`] + [`CATEGORY`] constants
    /// to the exact strings the converter stamps. A rename in either
    /// crate must land in the same commit or fail this pin.
    #[test]
    fn cross_crate_constant_mirror_pin() {
        // Match the converter's stamps byte-for-byte (see
        // `crates/vokra-convert/src/models/gtcrn.rs`).
        assert_eq!(ARCH, "gtcrn");
        assert_eq!(NAME, "gtcrn");
        assert_eq!(CATEGORY, "enhancement");
        assert_eq!(KEY_GTCRN_SAMPLE_RATE, "vokra.gtcrn.sample_rate");
        assert_eq!(KEY_GTCRN_N_FFT, "vokra.gtcrn.n_fft");
        assert_eq!(KEY_GTCRN_HOP, "vokra.gtcrn.hop");
        assert_eq!(KEY_GTCRN_N_BANDS, "vokra.gtcrn.n_bands");
        assert_eq!(KEY_GTCRN_GRU_HIDDEN, "vokra.gtcrn.gru_hidden");
    }

    // -----------------------------------------------------------------
    // Test 2 — Arch-tag distinctness pin
    // -----------------------------------------------------------------

    /// Pin `ARCH = "gtcrn"` and assert distinctness against every
    /// sibling denoise / separator arch string. A future rename of
    /// any sibling would land here in the same commit or fail this
    /// test.
    #[test]
    fn arch_tag_distinct_from_sibling_denoise_and_separator_arches() {
        assert_eq!(ARCH, "gtcrn");
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
        ] {
            assert_ne!(
                ARCH, sibling,
                "gtcrn (grouped Conv2D + SB-TF-LSTM + ERB grouping) and `{sibling}` \
                 are distinct enhancement / separator arches — sharing arch tag would \
                 misroute the runtime dispatch (FR-EX-08)"
            );
        }
    }

    // -----------------------------------------------------------------
    // Test 3 — GtcrnConfig default matches typical GTCRN axes
    // -----------------------------------------------------------------

    /// Pin [`GtcrnConfig::typical_default`] to the arXiv:2211.02063
    /// §III typical config. A rename or axis-value change would land
    /// here in the same commit or fail this test. Implementer MUST
    /// re-confirm against `github.com/Xiaobin-Rong/gtcrn` at land
    /// time.
    #[test]
    fn config_typical_default_matches_arxiv_axes() {
        let cfg = GtcrnConfig::typical_default();
        assert_eq!(cfg.sample_rate, 16_000, "sample_rate typical pin");
        assert_eq!(cfg.n_fft, 512, "n_fft typical pin");
        assert_eq!(cfg.hop, 256, "hop typical pin (16 ms at 16 kHz)");
        assert_eq!(cfg.n_bands, 257, "n_bands typical pin (= n_fft/2 + 1)");
        assert_eq!(cfg.gru_hidden, 64, "gru_hidden typical pin");
        assert_eq!(
            cfg.category, "enhancement",
            "GTCRN is a single-mask enhancement head (not a separator)"
        );
        // Structural invariant: n_bands = n_fft/2 + 1 (real-input
        // FFT).
        assert_eq!(
            cfg.n_bands,
            cfg.n_fft / 2 + 1,
            "structural invariant: n_bands must equal n_fft/2 + 1"
        );
        // `for_stamped_axes` builds the same value.
        assert_eq!(
            cfg,
            GtcrnConfig::for_stamped_axes(16_000, 512, 256, 257, 64),
            "for_stamped_axes must yield the same value as typical_default"
        );
    }

    // -----------------------------------------------------------------
    // Test 4 — Synthesized round-trip
    // -----------------------------------------------------------------

    /// Pin the [`Gtcrn::synthesized`] accessors so a later refactor of
    /// the accessor surface cannot silently change what the test
    /// fixture exposes.
    #[test]
    fn synthesized_round_trip() {
        let g = Gtcrn::synthesized();
        assert_eq!(g.sample_rate(), 16_000);
        assert_eq!(g.n_fft(), 512);
        assert_eq!(g.hop(), 256);
        assert_eq!(g.n_bands(), 257);
        assert_eq!(g.gru_hidden(), 64);
        assert_eq!(g.category(), "enhancement");
        assert_eq!(g.tensor_count(), 1);
        assert_eq!(
            g.weight_license(),
            LicenseClass::Unknown,
            "synthesized fixture uses Unknown (fail-closed at M2-13)"
        );
        assert_eq!(*g.config(), GtcrnConfig::typical_default());
    }

    // -----------------------------------------------------------------
    // Test 5 — from_gguf full chunk-group round-trip
    // -----------------------------------------------------------------

    /// Build a legitimate GGUF (arch + name + category + full 5-axis
    /// chunk group + provenance license class + one representative
    /// tensor). The binder must bind, hold the primary-source axes,
    /// surface the Permissive license class, and report at least one
    /// tensor bound.
    #[test]
    fn from_gguf_metadata_round_trip() {
        let cfg = GtcrnConfig::typical_default();
        let file = gtcrn_gguf(cfg, Some(LicenseClass::Permissive));
        let g = Gtcrn::from_gguf(&file).expect("valid GGUF must bind");
        // Config round-trip — every axis stamped by the converter
        // reads back into the same GtcrnConfig value.
        assert_eq!(*g.config(), cfg);
        // Accessor round-trip: every accessor surfaces the stamped
        // axis unchanged.
        assert_eq!(g.sample_rate(), cfg.sample_rate);
        assert_eq!(g.n_fft(), cfg.n_fft);
        assert_eq!(g.hop(), cfg.hop);
        assert_eq!(g.n_bands(), cfg.n_bands);
        assert_eq!(g.gru_hidden(), cfg.gru_hidden);
        assert_eq!(g.category(), CATEGORY);
        // License-class surface: the GTCRN converter defaults to
        // Permissive per the MIT stamp; missing provenance falls back
        // to Unknown (fail-closed at M2-13).
        assert_eq!(
            g.weight_license(),
            LicenseClass::Permissive,
            "gtcrn converter defaults to Permissive per MIT stamp"
        );
        assert!(
            g.tensor_count() >= 1,
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
        b.add_string(chunks::KEY_MODEL_NAME, "not-gtcrn");
        // No `vokra.model.arch`.
        b.add_tensor(
            "some.tensor.weight",
            GgmlType::F32,
            vec![4, 4],
            vec![0u8; 64],
        )
        .expect("add_tensor");
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let Err(err) = Gtcrn::from_gguf(&file) else {
            panic!("expected ModelLoad on missing arch");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains("missing `vokra.model.arch`"),
                    "message must call out the missing arch key, got `{m}`"
                );
                assert!(
                    m.contains("gtcrn"),
                    "message must name the gtcrn binder so a reader knows which \
                     loader complained, got `{m}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------
    // Test 7 — from_gguf rejects wrong arch (never silently mis-routes)
    // -----------------------------------------------------------------

    /// A `denoise` (DFN3) GGUF handed to the GTCRN binder by mistake
    /// must fail loud with a specific message naming every sibling
    /// denoise / separator arch rather than a downstream missing-key
    /// error (FR-EX-08). GTCRN's grouped Conv2D + SB-TF-LSTM + ERB
    /// grouping stack and DFN3's ERB analysis / synthesis + CRN are
    /// completely different topologies.
    #[test]
    fn from_gguf_rejects_wrong_arch() {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, "denoise");
        b.add_tensor("some.tensor", GgmlType::F32, vec![4, 4], vec![0u8; 64])
            .expect("add_tensor");
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let Err(err) = Gtcrn::from_gguf(&file) else {
            panic!("expected ModelLoad on wrong arch");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains("`denoise`") && m.contains("`gtcrn`"),
                    "message must name both the got and expected arch tags, got `{m}`"
                );
                // The error message names every sibling denoise /
                // separator arch so a reader knows which sibling
                // should not be aliased.
                for sibling in [
                    "denoise",
                    "rnnoise",
                    "nsnet2",
                    "dnsmos",
                    "sepformer",
                    "conv_tasnet",
                    "demucs",
                ] {
                    assert!(
                        m.contains(sibling),
                        "message must name sibling `{sibling}`, got `{m}`"
                    );
                }
                assert!(
                    m.contains("grouped Conv2D") && m.contains("SB-TF-LSTM"),
                    "message should call out GTCRN's characteristic primitives so the \
                     reader knows why the arches are distinct, got `{m}`"
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
    //          each of the 5 axes)
    // -----------------------------------------------------------------

    /// For each of the 5 mandatory `vokra.gtcrn.*` axes, omit exactly
    /// that one and assert the binder loud-fails with the missing
    /// key named in the error. A partially-stamped GGUF must be
    /// caught here, not silently defaulted to a fabricated axis
    /// (FR-EX-08 — the converter always stamps every axis, so a
    /// missing chunk always signals a partial / mis-produced GGUF).
    #[test]
    fn from_gguf_rejects_missing_topology_axis_each_of_five() {
        // Owned iteration over a fixed-size array so `k` / `skip_key`
        // bind as owned `&'static str` (not `&&str`) — avoids
        // auto-deref ambiguity when passing to `add_u32(key: &str,
        // ...)`.
        let axes: [(&str, u32); 5] = [
            (KEY_GTCRN_SAMPLE_RATE, 16_000),
            (KEY_GTCRN_N_FFT, 512),
            (KEY_GTCRN_HOP, 256),
            (KEY_GTCRN_N_BANDS, 257),
            (KEY_GTCRN_GRU_HIDDEN, 64),
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
                "en_conv.0.conv.weight",
                GgmlType::F32,
                vec![4, 4],
                vec![0u8; 64],
            )
            .expect("add_tensor");
            let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
            let Err(err) = Gtcrn::from_gguf(&file) else {
                panic!("expected ModelLoad on missing axis `{skip_key}`");
            };
            match err {
                VokraError::ModelLoad(m) => {
                    assert!(
                        m.contains(skip_key),
                        "message must name the missing axis key `{skip_key}`, got `{m}`"
                    );
                    assert!(
                        m.contains("arXiv:2211.02063"),
                        "message should cite the arXiv anchor so the reader knows the \
                         primary source of the typical values, got `{m}`"
                    );
                }
                other => panic!("expected VokraError::ModelLoad, got {other:?}"),
            }
        }
    }

    // -----------------------------------------------------------------
    // Test 9 — Empty tensor manifest fails loud (never binds all-zero
    //          forward)
    // -----------------------------------------------------------------

    /// Correct arch + full chunk group but zero tensors — the
    /// [`GtcrnWeights`] non-emptiness gate must fire with an FR-EX-08
    /// clause.
    #[test]
    fn from_gguf_rejects_empty_tensor_list() {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_u32(KEY_GTCRN_SAMPLE_RATE, 16_000);
        b.add_u32(KEY_GTCRN_N_FFT, 512);
        b.add_u32(KEY_GTCRN_HOP, 256);
        b.add_u32(KEY_GTCRN_N_BANDS, 257);
        b.add_u32(KEY_GTCRN_GRU_HIDDEN, 64);
        // NO tensors added.
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let Err(err) = Gtcrn::from_gguf(&file) else {
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
                assert!(
                    m.contains("gtcrn"),
                    "message must name the gtcrn binder / converter --model slug so \
                     the reader knows how to re-produce the GGUF, got `{m}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------
    // Test 10 — Weight-license round-trip
    // -----------------------------------------------------------------

    /// A GGUF that carries the arch tag + full chunk group + one
    /// tensor but NO `vokra.provenance.weight_license` chunk must
    /// fall back to [`LicenseClass::Unknown`] — the same fail-closed
    /// posture the MT3 / Sortformer / ConvTasnet / SepFormer binders
    /// take. `Unknown` is refused at the M2-13 compliance gate so a
    /// mis-stamped artifact cannot slip past commercial-mode dispatch.
    /// Stamping [`LicenseClass::Permissive`] flips the surfaced class
    /// (round-trip pin).
    #[test]
    fn weight_license_round_trip_unknown_default_and_permissive_stamp() {
        // Missing provenance -> Unknown (fail-closed).
        let file_no_prov = gtcrn_gguf(GtcrnConfig::typical_default(), None);
        let g = Gtcrn::from_gguf(&file_no_prov).expect("bind without provenance");
        assert_eq!(
            g.weight_license(),
            LicenseClass::Unknown,
            "missing provenance stamp must fall back to Unknown (fail-closed at M2-13)"
        );
        // With Permissive stamped -> Permissive surfaces (round-trip).
        let file_permissive = gtcrn_gguf(
            GtcrnConfig::typical_default(),
            Some(LicenseClass::Permissive),
        );
        let g2 = Gtcrn::from_gguf(&file_permissive).expect("bind with Permissive stamp");
        assert_eq!(
            g2.weight_license(),
            LicenseClass::Permissive,
            "Permissive stamp must round-trip verbatim"
        );
    }

    // -----------------------------------------------------------------
    // Test 11 — denoise returns UnsupportedOp naming all four missing
    //           primitives + both primary source URLs + every config
    //           axis
    // -----------------------------------------------------------------

    /// [`Gtcrn::denoise`] must loud-partial with [`VokraError::UnsupportedOp`]
    /// naming all four deferred primitives (grouped Conv2D + PReLU +
    /// SB-TF-LSTM + ERB grouping) + both primary source URLs (upstream
    /// repo + arXiv paper) + every config axis + the FR-EX-08 clause
    /// + the CLAUDE.md 教訓 (a) reference.
    #[test]
    fn denoise_loud_partial_returns_unsupported_op() {
        let file = gtcrn_gguf(
            GtcrnConfig::typical_default(),
            Some(LicenseClass::Permissive),
        );
        let g = Gtcrn::from_gguf(&file).unwrap();
        // 1 s of 16 kHz mono silence — legitimate input shape, so the
        // loud-partial gate fires (not some pre-denoise validation).
        let pcm = vec![0.0f32; 16_000];
        let Err(err) = g.denoise(&pcm) else {
            panic!("denoise must loud-partial");
        };
        match err {
            VokraError::UnsupportedOp(m) => {
                assert!(
                    m.contains("gtcrn denoise"),
                    "message must call out the gtcrn denoise surface, got `{m}`"
                );
                // All four missing primitives must be named by exact
                // identifier so the follow-up wave knows what to walk.
                assert!(
                    m.contains("grouped Conv2D"),
                    "message must name the grouped Conv2D primitive gap, got `{m}`"
                );
                assert!(
                    m.contains("PReLU"),
                    "message must name the PReLU activation gap, got `{m}`"
                );
                assert!(
                    m.contains("SB-TF-LSTM"),
                    "message must name the SB-TF-LSTM bottleneck gap, got `{m}`"
                );
                assert!(
                    m.contains("ERB"),
                    "message must name the ERB grouping gap, got `{m}`"
                );
                // Both primary source URLs must be cited.
                assert!(
                    m.contains("Xiaobin-Rong/gtcrn"),
                    "message must contain the upstream repo URL, got `{m}`"
                );
                assert!(
                    m.contains("2211.02063"),
                    "message must contain the arXiv paper id, got `{m}`"
                );
                // Every config axis must be echoed so the reader can
                // cross-check what topology the follow-up wave targets.
                assert!(
                    m.contains("sample_rate=16000"),
                    "sample_rate axis missing: {m}"
                );
                assert!(m.contains("n_fft=512"), "n_fft axis missing: {m}");
                assert!(m.contains("hop=256"), "hop axis missing: {m}");
                assert!(m.contains("n_bands=257"), "n_bands axis missing: {m}");
                assert!(m.contains("gru_hidden=64"), "gru_hidden axis missing: {m}");
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
                // DFN3 ERB distinction is called out so a reader
                // knows why aliasing is forbidden.
                assert!(
                    m.contains("DeepFilterNet3") || m.contains("DFN3"),
                    "message must call out the DFN3 ERB distinction so the reader \
                     knows why silent aliasing is forbidden, got `{m}`"
                );
            }
            other => panic!("expected VokraError::UnsupportedOp, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------
    // Test 12 — the loud-partial message may only name primitives that
    //           actually exist in `vokra-ops`, and may only claim a
    //           primitive is missing when it really is
    // -----------------------------------------------------------------

    /// [`Gtcrn::denoise`]'s message must not resurrect any of the four
    /// phantom claims an earlier revision carried.
    ///
    /// It cited `vokra_ops::conv2d` as an existing un-grouped
    /// convolution the grouped case merely fell outside of — there is
    /// no `conv2d` module and no public 2-D convolution of any kind in
    /// the crate, so that sentence sent a reader to a module they could
    /// never find. It justified the PReLU gap by contrast with "plain
    /// ReLU / GELU / SiLU" — the only public pointwise activation functions are
    /// `snake_activation_f32` and `snake_beta_f32`, so the conclusion
    /// (no PReLU) was right by accident off a false premise. It named
    /// `vokra_ops::lstm`, which was never landed. And it placed the
    /// Silero LSTM in `silero_vad::model`, a module that does not
    /// exist — the cell lives in the separate `vokra-vad-micro` crate.
    ///
    /// The negative assertions are the load-bearing half: without them
    /// the stale phrasing can rot back with nothing to catch it. The
    /// positive assertions pin what is actually true, so the row also
    /// fails if someone deletes the real reason instead of fixing it.
    #[test]
    fn denoise_loud_partial_names_only_real_primitives() {
        let file = gtcrn_gguf(
            GtcrnConfig::typical_default(),
            Some(LicenseClass::Permissive),
        );
        let g = Gtcrn::from_gguf(&file).unwrap();
        let pcm = vec![0.0f32; 16_000];
        let Err(err) = g.denoise(&pcm) else {
            panic!("denoise must loud-partial");
        };
        let VokraError::UnsupportedOp(m) = err else {
            panic!("expected VokraError::UnsupportedOp");
        };

        // --- Phantom primitive #1: `vokra_ops::conv2d` does not exist.
        //
        // Verified by grep over `crates/vokra-ops/src`: no `pub mod
        // conv2d`, no `pub fn conv2d`, and `pub fn conv` matches
        // nothing crate-wide. The only Conv2d is a private `struct
        // Conv2d` inside `denoise.rs`.
        assert!(
            !m.contains("vokra_ops::conv2d"),
            "`vokra_ops::conv2d` does not exist — naming it sends the reader to a \
             module that is not there: {m}"
        );

        // --- Phantom claim #2: the ReLU / GELU / SiLU contrast.
        //
        // `vokra_ops` exposes exactly two public activation functions,
        // `snake_activation_f32` and `snake_beta_f32`. Asserting a
        // family of plain activations is present as the foil for PReLU
        // is false, even though "no PReLU" is itself correct.
        assert!(
            !m.contains("has plain ReLU"),
            "`vokra_ops` has no public ReLU / GELU / SiLU cell — the only public \
             pointwise activation fns are snake_activation_f32 / snake_beta_f32, so \
             this contrast is false: {m}"
        );

        // --- Phantom primitive #3: `vokra_ops::lstm` does not exist,
        // --- and the Silero cell is not where this message said it was.
        //
        // The Silero LSTM lives in the separate `vokra-vad-micro` crate
        // as a `pub(crate) fn lstm_forward`; there is no
        // `silero_vad::model` module at all (the directory holds
        // mod.rs / parity.rs / stream.rs / wav.rs).
        assert!(
            !m.contains("vokra_ops::lstm"),
            "`vokra_ops::lstm` does not exist — the module was never landed: {m}"
        );
        assert!(
            !m.contains("silero_vad::model"),
            "there is no `silero_vad::model` module; the Silero LSTM lives in the \
             separate `vokra-vad-micro` crate: {m}"
        );

        // --- and the message must say the true things POSITIVELY, so a
        // --- later edit cannot satisfy this test by going silent.
        assert!(
            m.contains("NO public 2-D convolution AT ALL"),
            "must state the real conv gap: there is no public 2-D convolution in \
             `vokra_ops`, grouped or un-grouped: {m}"
        );
        assert!(
            m.contains("snake_activation_f32") && m.contains("snake_beta_f32"),
            "must name the activations that DO exist, so the PReLU gap rests on a \
             checked premise rather than an invented contrast: {m}"
        );
        assert!(
            m.contains("LstmLmCell") && m.contains("WRONG FUNCTION"),
            "must name the public LSTM that exists and say why it is the wrong \
             function here, rather than claiming no LSTM exists: {m}"
        );
        assert!(
            m.contains("vokra-vad-micro"),
            "must point at the crate the Silero LSTM actually lives in: {m}"
        );
        assert!(
            m.contains("erb_widths"),
            "must name the public ERB partition helper that DOES exist, so the ERB \
             gap is stated as GTCRN's aggregation matrix rather than as ERB being \
             absent: {m}"
        );

        // --- The already-resolved half, so the reader is told what NOT
        // --- to re-report (beat_this precedent).
        assert!(
            m.contains("ALREADY RESOLVED"),
            "the message must tell the reader which pieces are done: {m}"
        );
        assert!(
            m.contains("vokra_ops::stft") && m.contains("vokra_ops::istft_streaming"),
            "must name the framing primitives that already exist so nobody rewrites \
             STFT/iSTFT for this model: {m}"
        );
    }
}
