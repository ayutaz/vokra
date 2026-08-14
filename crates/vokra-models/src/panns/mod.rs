//! PANNs Cnn14 (Pre-trained Audio Neural Networks, Cnn14 backbone) — 527-class
//! AudioSet audio-tagging runtime binder (Wave 7 2026-08-14 audit follow-up,
//! loud-partial per audioldm2 / redimnet / musicgen / sortformer precedent).
//!
//! # Primary source
//!
//! - HF mirror: <https://huggingface.co/nicofarr/panns_Cnn14> (LICENSE
//!   un-verified — fail-closed `Unknown`)
//! - Upstream reference (MIT): <https://github.com/qiuqiangkong/audioset_tagging_cnn>
//!   (`pytorch/models.py class Cnn14` — the topology walk anchor)
//! - Inference wrapper: <https://github.com/qiuqiangkong/panns_inference>
//! - Paper: Kong et al. 2020, *"PANNs: Large-Scale Pretrained Audio Neural
//!   Networks for Audio Pattern Recognition"*, IEEE/ACM TASLP
//!   (<https://arxiv.org/abs/1912.10211>)
//! - AudioSet ontology (527-class output width anchor):
//!   <https://research.google.com/audioset/ontology/>
//!
//! # Architecture (transcribed from primary sources — Kong et al. 2020 §III-A + `pytorch/models.py class Cnn14`)
//!
//! ```text
//! PCM (mono f32, 32 kHz)
//!   -> log-mel front-end                             ← **loud-partial**
//!        (n_fft=1024, hop=320, win=1024, n_mels=64, f_min=50,
//!         f_max=14000; upstream uses `torchlibrosa.STFT` +
//!         `LogmelFilterBank` with ref=1.0, amin=1e-10, top_db=None. Vokra
//!         has a native `vokra_ops::mel_filterbank` primitive but binding
//!         it to PANNs' exact torchlibrosa reference (Slaney vs HTK mel
//!         normalization) needs a walk against
//!         `pytorch/pytorch_utils.py`.)
//!   -> CNN14 conv backbone                           ← **loud-partial**
//!        (6 stages × 2 conv blocks/stage; each conv block =
//!         Conv2D(3×3, stride=1, padding=1) + BatchNorm2D + ReLU;
//!         each stage-end = AvgPool2D(2×2). Channel plan
//!         64 → 64 → 128 → 128 → 256 → 256 → 512 → 512 → 1024 → 1024 →
//!         2048 → 2048 per `class Cnn14`. The residual-VGG topology is
//!         distinct from sibling YAMNet's MobileNetV1 depthwise-separable
//!         backbone — silent aliasing would misroute the runtime
//!         dispatch, FR-EX-08.)
//!   -> attention pooling head + projection           ← **loud-partial**
//!        (global attention pooling: mean over time + max over time
//!         + concat → 2048-d; fc1 Linear(2048, 2048) + ReLU;
//!         fc_audioset Linear(2048, 527) + sigmoid — multi-label output
//!         per Kong et al. §III-A.)
//!   -> 527-way probability vector
//! ```
//!
//! # Loud-partial classification (design § — CLAUDE.md 教訓 (a))
//!
//! - **Real (this WP)**:
//!   - [`Panns::from_gguf`] with strict `vokra.model.arch == "panns"`
//!     validation. The sibling audio-tagging / audio-embedding arch tags
//!     (`yamnet` / `ast` / `clap` / `mert` / `muq` / `dasheng` / `beats`)
//!     fail with a specific sibling-mis-route [`VokraError::ModelLoad`]
//!     enumerating the whole audio-tagging fleet — silent aliasing would
//!     misroute the runtime dispatch to a family with a wrong-topology
//!     loader (FR-EX-08).
//!   - [`PannsConfig::from_gguf`] with primary-source constant fallback
//!     (the PANNs converter does NOT currently stamp the `vokra.panns.*`
//!     chunk group — only arch / name / category / upstream_hf /
//!     provenance — so a *strict* reader would refuse the already-produced
//!     PANNs GGUF; the primary-source axes are transcribed from Kong et
//!     al. 2020 §III-A + `pytorch/config.py`, so fallback does not
//!     fabricate values. Mirror of the audioldm2 / musicgen / sortformer /
//!     conv_tasnet fallback precedent). A future converter sub-wave that
//!     starts stamping the chunk group upgrades this to real-stamped reads
//!     per-key with no runtime code change.
//!   - [`PannsWeights::from_gguf`] with a floor of non-empty tensor count
//!     enforced loud (a GGUF that carries zero tensors is refused rather
//!     than silently running an all-zero forward — FR-EX-08).
//!   - Weight-license class surfacing (defaults to
//!     [`LicenseClass::Unknown`] per the PANNs converter's stamped
//!     `unknown` — HF mirror LICENSE un-verified, fail-closed at the
//!     runtime compliance gate M2-13).
//!
//! - **Loud-partial (this WP)**: [`Panns::classify`] returns
//!   [`VokraError::UnsupportedOp`] naming **three** deferred pieces:
//!   1. the **log-mel front-end** (n_fft=1024, hop=320, 64-mel, fmin=50,
//!      fmax=14000) bound against upstream `torchlibrosa.STFT` +
//!      `LogmelFilterBank` (see `pytorch/pytorch_utils.py`);
//!      `vokra_ops::mel_filterbank` covers the primitive but PANNs uses
//!      specific torchlibrosa params (`ref=1.0, amin=1e-10, top_db=None`)
//!      that must be confirmed against a real checkpoint dump;
//!   2. the **CNN14 conv backbone** — 6 stages × 2 conv blocks/stage
//!      (`Conv2D(3×3, stride=1, padding=1) + BatchNorm2D + ReLU`), each
//!      stage-end `AvgPool2D(2×2)`; channel plan `64 → 64 → 128 → 128 →
//!      256 → 256 → 512 → 512 → 1024 → 1024 → 2048 → 2048` per
//!      `pytorch/models.py class Cnn14`;
//!   3. the **attention pooling head + projection** — global attention
//!      pooling (mean over time + max over time + concat → 2048-d),
//!      `fc1` Linear(2048, 2048) + ReLU, `fc_audioset` Linear(2048, 527)
//!      + sigmoid (multi-label output per Kong et al. §III-A).
//!
//! The error names **all four** primary source URLs (paper + code + inference
//! wrapper + AudioSet ontology) so a reader diagnosing this gap has exactly
//! four places to walk. **No fabricated probability array is ever emitted**
//! (FR-EX-08).
//!
//! # Sibling family distinctness (audio-tagging / audio-embedding neighbourhood)
//!
//! [`ARCH`] = `"panns"` is **deliberately distinct** from every sibling
//! audio-tagging / audio-embedding arch tag — the residual-VGG Cnn14
//! backbone has no analog in the neighbourhood:
//!
//! - `yamnet` — Google YAMNet (MobileNetV1 depthwise-separable, 521-class
//!   AudioSet subset — distinct topology + smaller class count);
//! - `ast` — MIT Audio Spectrogram Transformer (patch-embed Transformer);
//! - `clap` — LAION CLAP (contrastive text-audio dual-encoder);
//! - `mert` — HuBERT-derived masked-prediction music encoder;
//! - `muq` — Mel-RVQ + BEATs teacher self-supervised encoder;
//! - `dasheng` — MAE ViT / ConvNeXt universal audio encoder;
//! - `beats` — iterative acoustic-tokenizer bidirectional Transformer.
//!
//! Silently sharing arch would let runtime dispatch mis-route a PANNs
//! checkpoint onto a depthwise-conv / patch-embed / contrastive / masked-
//! prediction loader — the tensor-name walks would fail with a downstream
//! missing-tensor error instead of a specific arch-mismatch message.
//! FR-EX-08 forbids the silent shape misroute across audio-tagging arches.
//!
//! # Cross-crate constant duplication
//!
//! Mirror of the converter's [`ARCH`] / [`NAME`] / [`CATEGORY`] /
//! [`N_CLASSES_AUDIOSET`] — same rule the sibling BF16 pass-through binders
//! (`hifigan` / `snac` / `pyannote` / `beat_this` / `mt3` / `musicgen` /
//! `conv_tasnet` / `sepformer` / `redimnet` / `sortformer_diar_4spk_v1` /
//! `audioldm2` / `audiogen` / `jasco`) use so `vokra-models` does not gain a
//! dependency edge onto `vokra-convert`, preserving the layered convention
//! `vokra-ops → nothing GGUF-aware`, `vokra-core → GGUF reader`,
//! `vokra-models → GGUF binder`, `vokra-convert → GGUF writer`.
//!
//! # No ONNX / no pickle (permanent)
//!
//! PANNs ships upstream as a PyTorch `.pth` pickle
//! (`Cnn14_mAP=0.431.pth`, ~350 MB); this runtime **never** touches ONNX
//! or pickle (FR-LD-05 / NFR-DS-02). Callers pre-flatten offline via a
//! future `tools/parity/panns_prepare_checkpoint.py` sidecar (an
//! uv-managed Python 3.12 wrapper per memory `[[feedback-python-uses-uv]]`
//! + `[[feedback-python-3-12]]` — not part of the runtime), mirroring the
//! sibling audio-tagging / MIR bridge pattern.

use vokra_core::gguf::{GgufFile, chunks};
use vokra_core::{LicenseClass, Result, VokraError};

// ---------------------------------------------------------------------------
// Contract constants — mirror of `crates/vokra-convert/src/models/panns.rs`.
// See module docstring for the cross-crate duplication rationale.
// ---------------------------------------------------------------------------

/// Expected `vokra.model.arch` value written by
/// `vokra-cli convert --model panns`.
///
/// Distinct from every sibling audio-tagging / audio-embedding arch tag —
/// `yamnet` (MobileNetV1 depthwise-separable), `ast` (patch-embed
/// Transformer), `clap` (contrastive text-audio), `mert` (HuBERT-derived
/// masked prediction), `muq` (Mel-RVQ + BEATs teacher), `dasheng` (MAE
/// ViT/ConvNeXt), `beats` (iterative acoustic tokenizer). Silent aliasing
/// would misroute runtime dispatch to a wrong-topology loader (FR-EX-08
/// boundary — see the module docstring "Sibling family distinctness"
/// section).
pub const ARCH: &str = "panns";

/// Expected `vokra.model.name` value written by the converter — canonical
/// `nicofarr/panns_Cnn14` mirror slug (Cnn14 variant of the PANNs family).
pub const NAME: &str = "panns-cnn14";

/// Expected `vokra.model.category` value — audio tagging (527-class
/// AudioSet ontology).
pub const CATEGORY: &str = "audio-tagging";

/// Number of AudioSet ontology classes (Kong et al. 2020, upstream
/// README + `metadata/class_labels_indices.csv` in
/// `qiuqiangkong/audioset_tagging_cnn`). Primary source:
/// <https://research.google.com/audioset/ontology/>.
pub const N_CLASSES_AUDIOSET: u32 = 527;

// GGUF metadata-key constants for the topology chunk group. Reader-side
// only — the PANNs converter (as of 2026-08-13) does NOT stamp these,
// so `PannsConfig::from_gguf` falls back to the primary-source constants
// per-key. A future converter sub-wave that adds the stamps upgrades
// this reader to real-stamped reads seamlessly.

/// `vokra.panns.sample_rate` — audio sample rate (Hz). Primary source:
/// Kong et al. 2020 §III-A (Cnn14 32 kHz).
pub const GGUF_KEY_SAMPLE_RATE: &str = "vokra.panns.sample_rate";
/// `vokra.panns.n_fft` — STFT window size. Primary source: 1024.
pub const GGUF_KEY_N_FFT: &str = "vokra.panns.n_fft";
/// `vokra.panns.hop_length` — STFT hop length (samples). Primary source:
/// 320 (10 ms @ 32 kHz).
pub const GGUF_KEY_HOP_LENGTH: &str = "vokra.panns.hop_length";
/// `vokra.panns.win_length` — STFT window length. Primary source: 1024.
pub const GGUF_KEY_WIN_LENGTH: &str = "vokra.panns.win_length";
/// `vokra.panns.n_mels` — log-mel filterbank count. Primary source: 64.
pub const GGUF_KEY_N_MELS: &str = "vokra.panns.n_mels";
/// `vokra.panns.f_min` — log-mel lower cutoff (Hz). Primary source: 50.
pub const GGUF_KEY_F_MIN: &str = "vokra.panns.f_min";
/// `vokra.panns.f_max` — log-mel upper cutoff (Hz). Primary source: 14000.
pub const GGUF_KEY_F_MAX: &str = "vokra.panns.f_max";
/// `vokra.panns.n_classes` — output width (`fc_audioset` Linear out_features).
/// Primary source: 527 = [`N_CLASSES_AUDIOSET`].
pub const GGUF_KEY_N_CLASSES: &str = "vokra.panns.n_classes";
/// `vokra.panns.embed_dim` — clip-level embedding dim (`fc1` Linear
/// in/out_features). Primary source: 2048 (Kong et al. §III-B).
pub const GGUF_KEY_EMBED_DIM: &str = "vokra.panns.embed_dim";

// Primary-source URL constants — cited in the loud-partial error so a
// reader diagnosing the gap has fully specified anchors.

/// Primary-source anchor for the Cnn14 topology walk
/// (`pytorch/models.py class Cnn14`).
pub const PRIMARY_SOURCE_BASE: &str =
    "github.com/qiuqiangkong/audioset_tagging_cnn/blob/master/pytorch/models.py";
/// Primary-source anchor for the inference wrapper reference implementation.
pub const PRIMARY_SOURCE_INFERENCE: &str = "github.com/qiuqiangkong/panns_inference";
/// Primary-source anchor for the paper (Kong et al. 2020 IEEE/ACM TASLP).
pub const PRIMARY_SOURCE_PAPER: &str = "arxiv.org/abs/1912.10211";
/// Primary-source anchor for the AudioSet ontology (527-class output
/// width anchor).
pub const PRIMARY_SOURCE_ONTOLOGY: &str = "research.google.com/audioset/ontology/";

// ---------------------------------------------------------------------------
// PannsConfig — CNN14 topology axes.
// ---------------------------------------------------------------------------

/// Front-end + backbone + head axes for the PANNs `Cnn14` variant.
///
/// Values are transcribed from Kong et al. 2020 §III-A + upstream
/// `pytorch/config.py` for the primary 32 kHz Cnn14 checkpoint
/// (`Cnn14_mAP=0.431.pth`). Every numeric axis is `u32` in the GGUF
/// (mirror of audioldm2 / musicgen / conv_tasnet).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PannsConfig {
    /// Input sample rate (Hz). Primary source: 32000 (Cnn14 32 kHz).
    pub sample_rate: u32,
    /// STFT window size (samples). Primary source: 1024.
    pub n_fft: u32,
    /// STFT hop length (samples). Primary source: 320 (10 ms @ 32 kHz).
    pub hop_length: u32,
    /// STFT window length (samples). Primary source: 1024.
    pub win_length: u32,
    /// Log-mel filterbank count. Primary source: 64.
    pub n_mels: u32,
    /// Log-mel lower cutoff (Hz). Primary source: 50.
    pub f_min: u32,
    /// Log-mel upper cutoff (Hz). Primary source: 14000.
    pub f_max: u32,
    /// Number of AudioSet ontology output classes (`fc_audioset`
    /// out_features). Primary source: 527 = [`N_CLASSES_AUDIOSET`].
    pub n_classes: u32,
    /// Clip-level embedding dimension (`fc1` in/out_features). Primary
    /// source: 2048 (Kong et al. §III-B).
    pub embed_dim: u32,
}

impl Default for PannsConfig {
    /// The primary-source-transcribed Cnn14 32 kHz defaults. Used by
    /// [`PannsConfig::from_gguf`] as the fallback path when the topology
    /// chunk group is absent.
    fn default() -> Self {
        Self::cnn14_default()
    }
}

impl PannsConfig {
    /// The Cnn14 32 kHz primary-source axes as a `const` — transcribed
    /// from Kong et al. 2020 §III-A + upstream `pytorch/config.py` for
    /// the primary `Cnn14_mAP=0.431.pth` checkpoint.
    #[must_use]
    pub const fn cnn14_default() -> Self {
        Self {
            sample_rate: 32_000,
            n_fft: 1024,
            hop_length: 320,
            win_length: 1024,
            n_mels: 64,
            f_min: 50,
            f_max: 14_000,
            n_classes: N_CLASSES_AUDIOSET,
            embed_dim: 2048,
        }
    }

    /// Reads every `vokra.panns.*` chunk from `gguf`, falling back to
    /// the primary-source Cnn14 defaults per absent key.
    ///
    /// **AudioLDM 2 fallback posture** (not strict): the PANNs converter
    /// does not currently stamp this chunk group (only arch / name /
    /// category / upstream_hf / provenance), so on an already-produced
    /// PANNs GGUF every axis falls through to its primary-source default.
    /// A stamped `0` is treated as absent (defensive — a legitimately-
    /// stamped Cnn14 GGUF cannot have any zero axis).
    ///
    /// A future converter sub-wave that adds the stamps upgrades this
    /// reader to real-stamped reads per-key with no runtime code change.
    ///
    /// Mirror of [`crate::audioldm2::AudioLdm2Config::from_gguf`] +
    /// [`crate::musicgen::MusicGenConfig::from_gguf`] fallback pattern.
    #[must_use]
    pub fn from_gguf(gguf: &GgufFile) -> Self {
        let default = Self::cnn14_default();
        let read_u32 = |key: &str, fallback: u32| -> u32 {
            match gguf.get(key).and_then(|v| v.as_u64()) {
                Some(v) if v != 0 => v as u32,
                _ => fallback,
            }
        };
        Self {
            sample_rate: read_u32(GGUF_KEY_SAMPLE_RATE, default.sample_rate),
            n_fft: read_u32(GGUF_KEY_N_FFT, default.n_fft),
            hop_length: read_u32(GGUF_KEY_HOP_LENGTH, default.hop_length),
            win_length: read_u32(GGUF_KEY_WIN_LENGTH, default.win_length),
            n_mels: read_u32(GGUF_KEY_N_MELS, default.n_mels),
            f_min: read_u32(GGUF_KEY_F_MIN, default.f_min),
            f_max: read_u32(GGUF_KEY_F_MAX, default.f_max),
            n_classes: read_u32(GGUF_KEY_N_CLASSES, default.n_classes),
            embed_dim: read_u32(GGUF_KEY_EMBED_DIM, default.embed_dim),
        }
    }
}

// ---------------------------------------------------------------------------
// PannsWeights — non-empty tensor gate.
// ---------------------------------------------------------------------------

/// Weight tensors bound from a PANNs GGUF.
///
/// **Contract**: [`from_gguf`](Self::from_gguf) is a *loud* verification
/// step. A GGUF that carries zero tensors is rejected with
/// [`VokraError::ModelLoad`] (FR-EX-08 — an empty GGUF is never a valid
/// PANNs checkpoint; the CNN14 backbone alone has hundreds of Conv2D +
/// BN2D parameters, so an empty manifest always signals a mis-produced
/// GGUF).
///
/// Under the current landing this struct stores the tensor names + GGUF-
/// side dims discovered on disk. The follow-up wave sizes its dequant per
/// its kernel needs — today only the count + names are consumed so a
/// future `PannsWeights::bind_backbone_weights` /
/// `bind_head_weights` tensor walk can find its inputs without re-parsing
/// the GGUF.
#[derive(Debug)]
pub struct PannsWeights {
    /// Tensors discovered on disk, indexed by upstream `state_dict` name
    /// with their GGUF-side dims. Used by the load-time non-emptiness
    /// gate and by the future follow-up CNN14 + attention-pooling +
    /// projection forward wave.
    tensors: Vec<(String, Vec<usize>)>,
}

impl PannsWeights {
    /// Scans `gguf` for the PANNs state_dict tensors. Refuses to bind if
    /// the GGUF carries zero tensors (FR-EX-08 — an empty GGUF is never
    /// a valid PANNs Cnn14 checkpoint).
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
            return Err(VokraError::ModelLoad(format!(
                "panns: GGUF carries zero tensors — refusing to bind an all-zero \
                 forward (FR-EX-08). A legitimate PANNs Cnn14 checkpoint carries \
                 hundreds of Conv2D + BatchNorm2D parameters (arch={ARCH}, name={NAME}); \
                 zero tensors always signals a mis-produced GGUF. Re-run \
                 `vokra-cli convert --model panns` against an upstream \
                 `nicofarr/panns_Cnn14` safetensors checkpoint (`.pth` pickle \
                 must be pre-flattened offline via a future \
                 `tools/parity/panns_prepare_checkpoint.py`)."
            )));
        }
        Ok(Self { tensors })
    }

    /// Number of tensors bound from the GGUF. Purely a diagnostic
    /// accessor — the follow-up CNN14 + attention-pooling + projection
    /// forward wave uses it to size its expectations.
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }
}

// ---------------------------------------------------------------------------
// Panns — the runtime binder handle.
// ---------------------------------------------------------------------------

/// PANNs Cnn14 (Pre-trained Audio Neural Networks, 527-class AudioSet
/// audio tagger) runtime binder (`nicofarr/panns_Cnn14`, `Unknown`
/// fail-closed).
///
/// Bind with [`from_gguf`](Self::from_gguf), then call
/// [`classify`](Self::classify) on a mono f32 PCM waveform to obtain a
/// 527-way multi-label probability vector. See the module doc for the
/// current implementation-status matrix and the FR-EX-08 loud-error
/// contract on the log-mel + CNN14 + attention-pooling + projection
/// composition.
#[derive(Debug)]
pub struct Panns {
    config: PannsConfig,
    // The bound weights are held (real, counted) but the CNN14 + attention-
    // pooling + projection composition is a follow-up wave; the field is
    // deliberately `#[allow(dead_code)]` until the composition lands so a
    // reader is not misled by an unused field. Same posture as audioldm2 /
    // musicgen / redimnet / sortformer / pyannote / RMVPE / mt3 /
    // beat_this.
    #[allow(dead_code)]
    weights: PannsWeights,
    weight_license: LicenseClass,
}

impl Panns {
    /// Binds a PANNs Cnn14 GGUF: validates arch, reads the topology chunk
    /// group (with primary-source constant fallback per key), discovers
    /// tensors, and surfaces the stamped weight-license class for the
    /// compliance-gate cross-checks.
    ///
    /// This binder is a *loud* validation step. Every failure is a distinct
    /// [`VokraError::ModelLoad`] naming the missing / wrong key so a
    /// reader diagnosing a mis-produced GGUF has exactly one place to walk
    /// (FR-EX-08 — never a silent partial bind).
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] when `vokra.model.arch` is absent or
    ///   not `"panns"` (a sibling audio-tagging / audio-embedding GGUF
    ///   handed here by mistake — `yamnet` / `ast` / `clap` / `mert` /
    ///   `muq` / `dasheng` / `beats` — fails with a clear message
    ///   instead of a downstream missing-tensor error).
    /// - [`VokraError::ModelLoad`] when the GGUF carries zero tensors
    ///   ([`PannsWeights::from_gguf`] refuses to bind an all-zero
    ///   forward).
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        // 1. Arch check — always first so a mis-typed model handed here
        //    fails with a specific message instead of a downstream
        //    missing-tensor error.
        match file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()) {
            Some(a) if a == ARCH => {}
            Some(other) => {
                return Err(VokraError::ModelLoad(format!(
                    "panns: GGUF arch is `{other}`, expected `{ARCH}` (was this GGUF \
                     produced by `vokra-cli convert --model panns`? Note that sibling \
                     audio-tagging / audio-embedding arch tags — `yamnet` (Google \
                     MobileNetV1 depthwise-separable, 521-class AudioSet subset), \
                     `ast` (MIT Audio Spectrogram Transformer, patch-embed), `clap` \
                     (LAION contrastive text-audio dual-encoder), `mert` (HuBERT-\
                     derived masked-prediction music encoder), `muq` (Mel-RVQ + BEATs \
                     teacher self-supervised encoder), `dasheng` (MAE ViT/ConvNeXt \
                     universal audio encoder), `beats` (iterative acoustic-tokenizer \
                     bidirectional Transformer) — all live in the same audio-tagging \
                     / audio-embedding neighbourhood but have completely different \
                     backbone topologies. PANNs' residual-VGG Cnn14 (6-stage 2D-CNN \
                     with 3×3 Conv2D + BN2D + ReLU + AvgPool2D 2×2) has no analog in \
                     any sibling — silently aliasing arch would misroute the runtime \
                     dispatch (FR-EX-08 — no silent partial load)."
                )));
            }
            None => {
                return Err(VokraError::ModelLoad(
                    "panns: GGUF is missing `vokra.model.arch` (converter did not \
                     stamp it — this is not a Vokra-native panns GGUF)"
                        .to_owned(),
                ));
            }
        }

        // 2. Topology axes from the `vokra.panns.*` chunk group (fallback-
        //    friendly — see PannsConfig::from_gguf docstring for the
        //    fallback rationale).
        let config = PannsConfig::from_gguf(file);

        // 3. Load the tensor manifest with the non-emptiness gate.
        let weights = PannsWeights::from_gguf(file)?;

        // 4. Provenance surfacing — read the stamped weight-license class
        //    for the compliance-gate cross-checks. The PANNs converter
        //    defaults to `Unknown` per the HF mirror's un-verified LICENSE
        //    (fail-closed at M2-13). A GGUF missing the stamp also reads
        //    back as `Unknown` — same posture as musicgen / MT3 /
        //    sortformer / conv_tasnet.
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

    /// The bound topology axes (from `vokra.panns.*` chunk group with
    /// primary-source constant fallback — see
    /// [`PannsConfig::from_gguf`] docstring for the fallback rationale).
    #[inline]
    #[must_use]
    pub const fn config(&self) -> &PannsConfig {
        &self.config
    }

    /// The stamped weight-license class surfaced from the GGUF's
    /// `vokra.provenance.weight_license` chunk. The PANNs converter
    /// stamps `Unknown` by default per the HF mirror's un-verified
    /// LICENSE (fail-closed at the M2-13 compliance gate — publish
    /// refused until owner primary-source-confirms the mirror LICENSE
    /// and re-converts with `--license <spdx>`).
    #[inline]
    #[must_use]
    pub const fn weight_license(&self) -> LicenseClass {
        self.weight_license
    }

    /// Number of tensors bound from the GGUF. Purely a diagnostic
    /// accessor — the follow-up CNN14 + attention-pooling + projection
    /// forward wave uses it to size its expectations.
    #[inline]
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.weights.tensor_count()
    }

    /// Multi-label 527-class classification of a PCM waveform.
    ///
    /// # Loud-partial (this WP)
    ///
    /// Returns [`VokraError::UnsupportedOp`] — the full PANNs Cnn14 forward
    /// requires **three** deferred pieces:
    ///
    /// 1. the **log-mel front-end** (n_fft=1024, hop=320, 64-mel, fmin=50,
    ///    fmax=14000) bound against upstream `torchlibrosa.STFT` +
    ///    `LogmelFilterBank` (see `pytorch/pytorch_utils.py`);
    ///    `vokra_ops::mel_filterbank` covers the primitive but PANNs uses
    ///    specific torchlibrosa params (`ref=1.0, amin=1e-10,
    ///    top_db=None`) that must be confirmed against a real checkpoint
    ///    dump;
    /// 2. the **CNN14 conv backbone** — 6 stages × 2 conv blocks/stage
    ///    (`Conv2D(3×3, stride=1, padding=1) + BatchNorm2D + ReLU`), each
    ///    stage-end `AvgPool2D(2×2)`; channel plan `64 → 64 → 128 → 128 →
    ///    256 → 256 → 512 → 512 → 1024 → 1024 → 2048 → 2048` per
    ///    `pytorch/models.py class Cnn14`;
    /// 3. the **attention pooling head + projection** — global attention
    ///    pooling (`torch.mean` over time + `torch.max` over time + concat
    ///    → 2048-d), `fc1` Linear(2048, 2048) + ReLU, `fc_audioset`
    ///    Linear(2048, 527) + sigmoid (multi-label output per Kong et al.
    ///    §III-A).
    ///
    /// The error names **all four** primary source URLs (paper +
    /// AudioCraft-style code walk + inference wrapper + AudioSet ontology)
    /// so a reader diagnosing this gap has exactly four places to walk.
    /// Every config axis is echoed so the reader can cross-check what
    /// topology the follow-up wave targets. **No fabricated probability
    /// array is ever emitted** (FR-EX-08 — no silent partial output).
    ///
    /// The `_pcm` argument is treated as the raw waveform at
    /// `config.sample_rate` (mono, f32 in `[-1, 1]`); shape mismatch will
    /// be a loud error rather than a resample surprise when the real
    /// forward lands.
    ///
    /// # Errors
    ///
    /// - [`VokraError::UnsupportedOp`] — the loud-partial gate for the
    ///   deferred log-mel + CNN14 + attention-pooling composition.
    pub fn classify(&self, _pcm: &[f32]) -> Result<Vec<f32>> {
        Err(classify_forward_loud_partial(&self.config))
    }
}

/// Construct the loud-partial [`VokraError::UnsupportedOp`] returned by
/// [`Panns::classify`] until the log-mel + CNN14 + attention-pooling +
/// projection composition lands.
///
/// Names **four** primary source URLs (paper + code + inference wrapper +
/// AudioSet ontology) so a reader diagnosing the gap has exactly four
/// places to walk. Every config axis is echoed. Mirror of the audioldm2 /
/// musicgen / conv_tasnet / redimnet / sortformer / RMVPE / pyannote
/// loud-partial-message precedent (CLAUDE.md 教訓 (a)).
fn classify_forward_loud_partial(cfg: &PannsConfig) -> VokraError {
    VokraError::UnsupportedOp(format!(
        "panns classify (loud-partial): the full forward is deferred; \
         three missing primitives must land before real logits can be emitted: \
         (1) log-mel front-end \
         (n_fft={n_fft}, hop_length={hop}, win_length={win}, n_mels={n_mels}, \
         f_min={fmin}, f_max={fmax}, sample_rate={sr}) bound against upstream \
         `torchlibrosa.STFT` + `LogmelFilterBank` (see {base}, torchlibrosa \
         params ref=1.0, amin=1e-10, top_db=None require confirmation against \
         a real checkpoint dump); \
         (2) CNN14 conv backbone: 6 stages x 2 conv blocks/stage \
         (Conv2D(3x3, stride=1, padding=1) + BatchNorm2D + ReLU), each \
         stage-end AvgPool2D(2x2), channel plan \
         64 -> 64 -> 128 -> 128 -> 256 -> 256 -> 512 -> 512 -> 1024 -> 1024 \
         -> {embed} -> {embed} per pytorch/models.py class Cnn14 (see {base}); \
         (3) attention pooling head + projection: global attention pooling \
         (mean over time + max over time + concat -> {embed}-d), \
         fc1 Linear({embed}, {embed}) + ReLU, \
         fc_audioset Linear({embed}, {n_classes}) + sigmoid \
         (multi-label output per Kong et al. arXiv:1912.10211 section III-A). \
         Primary sources: paper {paper}, code {base}, inference wrapper {inf}, \
         AudioSet ontology {onto}. Runtime cannot fabricate a probability array \
         (FR-EX-08 no silent partial output).",
        n_fft = cfg.n_fft,
        hop = cfg.hop_length,
        win = cfg.win_length,
        n_mels = cfg.n_mels,
        fmin = cfg.f_min,
        fmax = cfg.f_max,
        sr = cfg.sample_rate,
        embed = cfg.embed_dim,
        n_classes = cfg.n_classes,
        paper = PRIMARY_SOURCE_PAPER,
        base = PRIMARY_SOURCE_BASE,
        inf = PRIMARY_SOURCE_INFERENCE,
        onto = PRIMARY_SOURCE_ONTOLOGY,
    ))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    //! Tests for the PANNs Cnn14 runtime binder — config default pin +
    //! metadata round-trip + negative-space round-trip on the loud-partial
    //! gates + arch-tag distinctness pin.
    //!
    //! # What "round-trip" means here
    //!
    //! The task spec asks for 5+ unit tests. On a real 32 kHz PCM waveform
    //! this would be `classify(...)` returning a 527-way probability
    //! vector, but the log-mel + CNN14 + attention-pooling + projection
    //! composition is deferred (see the module doc + [`Panns::classify`]
    //! rustdoc). Fabricating a real classification output would violate
    //! CLAUDE.md 教訓 (a) ("loud-partial は fake-complete より honest").
    //!
    //! The round-trip semantics we *can* honestly test:
    //!
    //! 1. **Config default pin**: the Cnn14 32 kHz axes match Kong et al.
    //!    2020 §III-A + upstream `pytorch/config.py`.
    //! 2. **Config round-trip**: `from_gguf` reads every axis stamped by
    //!    a future converter sub-wave (via the fallback path today; the
    //!    stamped-axes path is future-compat).
    //! 3. **Loud-error negative-space round-trip**: every stated blocker
    //!    (missing arch / wrong arch / empty tensor list / unsupported
    //!    forward surface) fires at its documented surface point, in the
    //!    documented error variant.
    //! 4. **Arch-tag distinctness pin**: the arch string is stable and
    //!    distinct from every sibling audio-tagging / audio-embedding arch.
    //! 5. **AudioSet ontology pin**: the 527-class output width is a
    //!    load-bearing constant.

    use super::*;
    use vokra_core::gguf::{GgmlType, GgufBuilder};

    /// Helper: builds a legitimate PANNs GGUF (arch + name + category +
    /// license class + one representative Cnn14 tensor). Optionally
    /// stamps the topology chunk group.
    fn panns_gguf(
        cfg: Option<PannsConfig>,
        weight_license_class: Option<LicenseClass>,
    ) -> GgufFile {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_string(chunks::KEY_MODEL_NAME, NAME);
        b.add_string("vokra.model.category", CATEGORY);
        if let Some(cfg) = cfg {
            b.add_u32(GGUF_KEY_SAMPLE_RATE, cfg.sample_rate);
            b.add_u32(GGUF_KEY_N_FFT, cfg.n_fft);
            b.add_u32(GGUF_KEY_HOP_LENGTH, cfg.hop_length);
            b.add_u32(GGUF_KEY_WIN_LENGTH, cfg.win_length);
            b.add_u32(GGUF_KEY_N_MELS, cfg.n_mels);
            b.add_u32(GGUF_KEY_F_MIN, cfg.f_min);
            b.add_u32(GGUF_KEY_F_MAX, cfg.f_max);
            b.add_u32(GGUF_KEY_N_CLASSES, cfg.n_classes);
            b.add_u32(GGUF_KEY_EMBED_DIM, cfg.embed_dim);
        }
        if let Some(cls) = weight_license_class {
            b.add_string(chunks::KEY_PROVENANCE_WEIGHT_LICENSE, cls.as_str());
        }
        // One representative Cnn14 tensor so the non-emptiness gate
        // passes. Uses the upstream `conv_block1.conv1.weight` name
        // (first Conv2D in the first stage-1 block per
        // `pytorch/models.py class Cnn14`, matching the converter test's
        // chosen sample tensor for consistency).
        b.add_tensor(
            "conv_block1.conv1.weight",
            GgmlType::F32,
            vec![64, 1, 3, 3],
            vec![0u8; 64 * 1 * 3 * 3 * 4],
        )
        .expect("add_tensor");
        GgufFile::parse(b.to_bytes().expect("serialize")).expect("parse")
    }

    // -----------------------------------------------------------------------
    // Test 1 — Config default pin (Cnn14 32 kHz axes match Kong et al.
    //          2020 §III-A + upstream pytorch/config.py)
    // -----------------------------------------------------------------------

    #[test]
    fn config_default_matches_kong_et_al_config_py() {
        // Pin the Cnn14 32 kHz axes. A rename or axis-value drift would
        // land here in the same commit or fail this test.
        let cfg = PannsConfig::cnn14_default();
        assert_eq!(cfg.sample_rate, 32_000, "Cnn14 primary sample_rate");
        assert_eq!(cfg.n_fft, 1024, "Cnn14 primary n_fft");
        assert_eq!(
            cfg.hop_length, 320,
            "Cnn14 primary hop_length (10 ms @ 32 kHz)"
        );
        assert_eq!(cfg.win_length, 1024, "Cnn14 primary win_length");
        assert_eq!(cfg.n_mels, 64, "Cnn14 primary n_mels");
        assert_eq!(cfg.f_min, 50, "Cnn14 primary f_min");
        assert_eq!(cfg.f_max, 14_000, "Cnn14 primary f_max");
        assert_eq!(cfg.n_classes, 527, "AudioSet ontology output width");
        assert_eq!(cfg.embed_dim, 2048, "Cnn14 embedding + fc1 width");
        // Structural invariant: Default matches cnn14_default (both must
        // be primary-source-transcribed constants; no silent divergence).
        assert_eq!(PannsConfig::default(), cfg);
    }

    // -----------------------------------------------------------------------
    // Test 2 — from_gguf metadata round-trip (fallback config, provenance
    //          stamp present, non-empty tensor gate passed)
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_metadata_round_trip() {
        // Build a legitimate GGUF with arch + license + one tensor but
        // NO topology chunks. The binder must bind, hold the primary-
        // source Cnn14 defaults via fallback, surface the Unknown
        // license class, and report at least one tensor bound.
        let file = panns_gguf(None, Some(LicenseClass::Unknown));
        let p = Panns::from_gguf(&file).expect("valid GGUF must bind");
        // Config fallback: absent topology chunks fall through to Cnn14
        // primary-source defaults.
        assert_eq!(*p.config(), PannsConfig::cnn14_default());
        // License-class surface: Unknown per the HF mirror's un-verified
        // LICENSE (fail-closed at M2-13).
        assert_eq!(p.weight_license(), LicenseClass::Unknown);
        assert!(
            p.tensor_count() >= 1,
            "at least one tensor must be bound from the legitimate GGUF fixture"
        );
    }

    // -----------------------------------------------------------------------
    // Test 3 — from_gguf rejects wrong arch (never silently mis-routes)
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_rejects_wrong_arch() {
        // A `yamnet` (MobileNetV1 depthwise-separable) GGUF handed to
        // the PANNs binder by mistake must fail loud with a specific
        // message rather than silently mis-binding (FR-EX-08). PANNs'
        // residual-VGG Cnn14 and YAMNet's MobileNetV1 depthwise-separable
        // are completely different backbone topologies, so silent aliasing
        // would misroute the runtime dispatch.
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, "yamnet");
        b.add_string(chunks::KEY_MODEL_NAME, "yamnet");
        b.add_tensor("yamnet.probe", GgmlType::F32, vec![4, 4], vec![0u8; 64])
            .expect("add_tensor");
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let Err(err) = Panns::from_gguf(&file) else {
            panic!("expected ModelLoad on wrong arch");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains("`yamnet`") && m.contains("`panns`"),
                    "message must name both the got and expected arch tags, got `{m}`"
                );
                // The message must enumerate the whole audio-tagging /
                // audio-embedding sibling fleet so the reader has fully
                // specified anchors.
                for sibling in ["yamnet", "ast", "clap", "mert", "muq", "dasheng", "beats"] {
                    assert!(
                        m.contains(sibling),
                        "expected sibling '{sibling}' disambiguation in error: {m}"
                    );
                }
                // The message must call out the residual-VGG vs
                // MobileNetV1 depthwise-separable topology divergence.
                assert!(
                    m.contains("depthwise-separable") || m.contains("residual-VGG"),
                    "message should call out the residual-VGG vs depthwise-separable \
                     topology divergence, got `{m}`"
                );
                assert!(
                    m.contains("FR-EX-08"),
                    "message must cite the FR-EX-08 clause, got `{m}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Test 4 — classify returns UnsupportedOp with primary-source anchors
    //          + every config axis echoed + FR-EX-08 rationale
    // -----------------------------------------------------------------------

    #[test]
    fn classify_loud_partial_returns_unsupported_op() {
        let file = panns_gguf(None, Some(LicenseClass::Unknown));
        let p = Panns::from_gguf(&file).unwrap();

        // Legitimate PCM shape: 1 s of silence at 32 kHz (mono).
        let pcm = vec![0.0_f32; p.config().sample_rate as usize];
        let Err(err) = p.classify(&pcm) else {
            panic!("classify must loud-partial");
        };
        match err {
            VokraError::UnsupportedOp(msg) => {
                // Names the surface + posture.
                assert!(
                    msg.contains("panns classify"),
                    "surface must be called out: {msg}"
                );
                assert!(msg.contains("loud-partial"), "posture label: {msg}");

                // Names the three missing pieces by exact identifier.
                assert!(
                    msg.contains("log-mel front-end"),
                    "message must name the log-mel front-end gap, got `{msg}`"
                );
                assert!(
                    msg.contains("CNN14 conv backbone"),
                    "message must name the CNN14 conv backbone gap, got `{msg}`"
                );
                assert!(
                    msg.contains("attention pooling head"),
                    "message must name the attention pooling head gap, got `{msg}`"
                );

                // Cites all four primary source URLs so a reader
                // diagnosing the gap has anchors to walk.
                for url in [
                    PRIMARY_SOURCE_PAPER,
                    PRIMARY_SOURCE_BASE,
                    PRIMARY_SOURCE_INFERENCE,
                    PRIMARY_SOURCE_ONTOLOGY,
                ] {
                    assert!(
                        msg.contains(url),
                        "expected primary source URL '{url}' cited: {msg}"
                    );
                }

                // Every config axis is echoed so the reader can
                // cross-check what topology the follow-up wave targets.
                assert!(msg.contains("n_fft=1024"), "n_fft axis missing: {msg}");
                assert!(
                    msg.contains("hop_length=320"),
                    "hop_length axis missing: {msg}"
                );
                assert!(
                    msg.contains("win_length=1024"),
                    "win_length axis missing: {msg}"
                );
                assert!(msg.contains("n_mels=64"), "n_mels axis missing: {msg}");
                assert!(msg.contains("f_min=50"), "f_min axis missing: {msg}");
                assert!(msg.contains("f_max=14000"), "f_max axis missing: {msg}");
                assert!(
                    msg.contains("sample_rate=32000"),
                    "sample_rate axis missing: {msg}"
                );

                // FR-EX-08 rationale cited.
                assert!(
                    msg.contains("FR-EX-08"),
                    "expected FR-EX-08 rationale for no fake logits: {msg}"
                );
            }
            other => panic!("expected VokraError::UnsupportedOp, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Test 5 — Arch tag pin + sibling distinctness (silent aliasing would
    //          misroute the runtime dispatch across audio-tagging arches)
    // -----------------------------------------------------------------------

    #[test]
    fn arch_tag_distinct_from_sibling_audio_tagging() {
        // Pin the arch string so a rename would land here in the same
        // commit or fail this test. The sibling audio-tagging /
        // audio-embedding arches MUST NOT collide with ours.
        assert_eq!(ARCH, "panns", "PANNs arch tag pin");
        for sibling in ["yamnet", "ast", "clap", "mert", "muq", "dasheng", "beats"] {
            assert_ne!(
                ARCH, sibling,
                "PANNs arch tag must differ from '{sibling}' — silent alias would \
                 misroute GGUF into a wrong-topology loader (FR-EX-08 boundary)"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Test 6 — Empty tensor manifest fails loud (never binds all-zero
    //          forward — FR-EX-08)
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_rejects_empty_tensor_list() {
        // Correct arch + name + license class but zero tensors — the
        // PannsWeights non-emptiness gate must fire.
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_string(chunks::KEY_MODEL_NAME, NAME);
        b.add_string(chunks::KEY_PROVENANCE_WEIGHT_LICENSE, "unknown");
        // NO tensors added.
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let Err(err) = Panns::from_gguf(&file) else {
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
                    m.contains("vokra-cli convert --model panns"),
                    "message must include the repro command, got `{m}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Test 7 — AudioSet ontology pin (527-class output width is a
    //          load-bearing constant, a rename or drift must be caught)
    // -----------------------------------------------------------------------

    #[test]
    fn n_classes_pin_matches_audioset_ontology() {
        assert_eq!(N_CLASSES_AUDIOSET, 527, "AudioSet ontology has 527 classes");
        assert_eq!(
            PannsConfig::cnn14_default().n_classes,
            N_CLASSES_AUDIOSET,
            "PannsConfig::cnn14_default must lock to N_CLASSES_AUDIOSET"
        );
    }

    // -----------------------------------------------------------------------
    // Test 8 — from_gguf reads stamped topology axes (round-trip future
    //          compat when a converter sub-wave adds the topology stamps)
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_reads_stamped_topology_axes() {
        // Stamp non-default axis values (a hypothetical 16 kHz Cnn14
        // variant) so we can verify the read picks them up vs falling
        // through to defaults. This test pins the future-proofing
        // invariant: when a converter sub-wave adds the topology stamps,
        // this reader picks them up per-key.
        let cfg = PannsConfig {
            sample_rate: 16_000,
            n_mels: 128,
            ..PannsConfig::cnn14_default()
        };
        let file = panns_gguf(Some(cfg), Some(LicenseClass::Unknown));
        let p = Panns::from_gguf(&file).expect("valid stamped-axes GGUF must bind");
        // Stamped topology axes must round-trip exactly (no silent
        // fallback to primary-source defaults when the chunk group is
        // present).
        assert_eq!(
            p.config().sample_rate,
            16_000,
            "stamped sample_rate honored"
        );
        assert_eq!(p.config().n_mels, 128, "stamped n_mels honored");
        // Unstamped axes remain at Cnn14 primary defaults.
        assert_eq!(p.config().n_fft, 1024, "unstamped n_fft = default");
        assert_eq!(
            p.config().n_classes,
            N_CLASSES_AUDIOSET,
            "unstamped n_classes = AudioSet 527"
        );
    }
}
