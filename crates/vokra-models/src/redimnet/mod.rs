//! **ReDimNet** (`Wespeaker/wespeaker-voxceleb-redimnet2-B6-LM`,
//! apache-2.0) — Reshape Dimensions Network speaker-embedding
//! encoder (Yakovlev et al. arXiv:2402.01049 "Reshape Dimensions
//! Network for Speaker Recognition") — runtime binder for the
//! `redimnet` converter arch.
//!
//! # Runtime layout (loud-partial, wespeaker / titanet / ecapa_tdnn
//! speaker-fleet posture per CLAUDE.md 教訓 (a))
//!
//! ```text
//! log-mel fbank (mono f32, [T × N_MELS=72] flat row-major)
//!   -> 2D dim-reduction stem (basic_resnet)   ← **loud-partial**
//!        (Vokra's shared Conv2D / BatchNorm2D / ReLU primitives
//!         cover the tensor math; the wire-up onto ReDimNet2's exact
//!         basic_resnet block sequence lands with the WeSpeaker
//!         Python source transcription wave.)
//!   -> 1D conv+att transformer-lite blocks    ← **loud-partial**
//!        (Conformer-lite variant distinct from `vokra_ops::conformer`
//!         full Conformer — the `conv+att` block sequence needs a
//!         topology walk against
//!         `github.com/wenet-e2e/wespeaker/blob/master/wespeaker/models/redimnet.py`
//!         + `redimnet2.py` that no sibling in the tree supplies.)
//!   -> ASTP (Attentive Statistics Pooling)    ← **loud-partial**
//!        (Attentive statistics pooling to a fixed-length embedding —
//!         the pooling head kernel needs a walk against
//!         `wespeaker/pooling_layers.py` `AttentiveStatisticsPool2d`.)
//!   -> 192-d embedding (`EMBED_DIM`)
//! ```
//!
//! # Loud-partial classification (design § — CLAUDE.md 教訓 (a))
//!
//! - **Real (this WP)**: [`ReDimNet::from_gguf`] with strict
//!   `vokra.model.arch == "redimnet"` validation + strict
//!   `vokra.redimnet.*` chunk-group presence enforcement (every axis
//!   required — no primary-source constant fallback because the
//!   converter transcribes the axes from the upstream `config.yaml`
//!   and stamps them, and this binder mirrors those stamps rather
//!   than silently defaulting to a fabricated axis),
//!   [`ReDimNetWeights::from_gguf`] with a floor of non-empty tensor
//!   count enforced loud (a GGUF that carries no ReDimNet-typical
//!   tensors is refused rather than silently running an all-zero
//!   forward), license-class surfacing.
//! - **Loud-partial (this WP)**: [`ReDimNet::encode`] returns
//!   [`VokraError::UnsupportedOp`] naming the three exact missing
//!   pieces:
//!   (i) 2D `basic_resnet` block sequence walk,
//!   (ii) 1D `conv+att` block sequence walk (Conformer-lite variant),
//!   (iii) ASTP (Attentive Statistics Pooling) head.
//!   Every message echoes every config axis so the reader can
//!   cross-check what topology the follow-up wave targets.
//!
//! Rationale (RMVPE / pyannote / hifigan / vocos / bigvgan / snac /
//! beat_this / mt3 Wave 1-3 precedent, CLAUDE.md 教訓 (a)): the
//! surrounding scaffold + `from_gguf` chunk-group validation +
//! FR-EX-08 loud-fails land today so a follow-up wave can flip the
//! switch by transcribing the WeSpeaker Python
//! `wespeaker/models/redimnet2.py` topology + writing the encode
//! forward against those axes. The [`VokraError::UnsupportedOp`]
//! message cites both the redimnet.py + redimnet2.py primary sources
//! + the arXiv paper so a reader diagnosing this gap has exactly
//! three anchors to walk.
//!
//! # `vokra.redimnet.*` chunk group (read here)
//!
//! Written by `vokra-convert::models::redimnet::convert_redimnet_file`:
//!
//! - `vokra.model.arch` (`String`): must equal [`ARCH`] (`"redimnet"`).
//!   Distinct from every sibling speaker-fleet arch (`wespeaker`,
//!   `ecapa_tdnn`, `titanet`, `speaker_3d`, `campplus`) — silently
//!   sharing would misroute runtime dispatch (FR-EX-08).
//! - `vokra.model.name` (`String`):
//!   `"wespeaker-voxceleb-redimnet2-b6-lm"` — auxiliary check.
//! - `vokra.redimnet.{embed_dim, out_channels, c, f, n_mels, n_fft,
//!   hop_length, win_length, sample_rate, f_min, f_max, do_preemph}`
//!   (`u32` each): the ReDimNet2 B6-LM topology axes + mel-spec
//!   front-end axes.
//! - `vokra.provenance.*`: license class + raw license string, so the
//!   runtime compliance gate (FR-CP-03) can classify the artifact
//!   without re-inspecting the safetensors provenance. Defaults to
//!   `Permissive` in production per apache-2.0 stamp.
//!
//! # Cross-crate constant duplication
//!
//! Mirror of the converter's [`ARCH`] / [`GGUF_KEY_*`] — same rule
//! the sibling BF16 pass-through binders (`pyannote` / `snac` /
//! `hifigan` / `beat_this` / `mt3`) use so `vokra-models` does not
//! gain a dependency edge onto `vokra-convert`, preserving the
//! layered convention `vokra-ops → nothing GGUF-aware`,
//! `vokra-core → GGUF reader`, `vokra-models → GGUF binder`,
//! `vokra-convert → GGUF writer`.
//!
//! # No ONNX / no pickle (permanent)
//!
//! ReDimNet ships upstream as a PyTorch `.pt` pickle
//! (`avg_model.pt`, ~55.5 MB averaged across the LM fine-tune
//! ensemble); this runtime **never** touches ONNX or pickle
//! (FR-LD-05 / NFR-DS-02). The `.pt` → safetensors bridge lives
//! offline through the sibling
//! `tools/parity/nemo_pt_to_safetensors.py` sidecar (uv-managed
//! Python 3.12 per memory `[[feedback-python-uses-uv]]` +
//! `[[feedback-python-3-12]]`).

use vokra_core::gguf::{GgufFile, chunks};
use vokra_core::{LicenseClass, Result, VokraError};

// ---------------------------------------------------------------------------
// Arch / metadata-key constants — mirror of
// `crates/vokra-convert/src/models/redimnet.rs` (see module docstring
// for the cross-crate duplication rationale).
// ---------------------------------------------------------------------------

/// Expected `vokra.model.arch` value written by
/// `vokra-cli convert --model redimnet`.
///
/// Distinct from every sibling speaker-fleet arch — never
/// `wespeaker` (ResNet-34), never `ecapa_tdnn` (TDNN stack), never
/// `titanet` (depth-wise separable Conv1D), never `speaker_3d`
/// (ERes2Net), never `campplus` (CAM++). Silently sharing an arch
/// would misroute runtime dispatch (FR-EX-08).
pub const ARCH: &str = "redimnet";

/// `vokra.redimnet.embed_dim` — speaker embedding dimension (192).
pub const GGUF_KEY_EMBED_DIM: &str = "vokra.redimnet.embed_dim";
/// `vokra.redimnet.out_channels` — last 1D block channel count (224).
pub const GGUF_KEY_OUT_CHANNELS: &str = "vokra.redimnet.out_channels";
/// `vokra.redimnet.c` — ReDimNet2 channel expansion base (64).
pub const GGUF_KEY_C: &str = "vokra.redimnet.c";
/// `vokra.redimnet.f` — mel-frequency dim after the 2D stem (72).
pub const GGUF_KEY_F: &str = "vokra.redimnet.f";
/// `vokra.redimnet.n_mels` — log-mel filterbank count (72).
pub const GGUF_KEY_N_MELS: &str = "vokra.redimnet.n_mels";
/// `vokra.redimnet.n_fft` — STFT window size (512).
pub const GGUF_KEY_N_FFT: &str = "vokra.redimnet.n_fft";
/// `vokra.redimnet.hop_length` — STFT hop (160 = 10 ms at 16 kHz).
pub const GGUF_KEY_HOP_LENGTH: &str = "vokra.redimnet.hop_length";
/// `vokra.redimnet.win_length` — STFT window length (400 = 25 ms).
pub const GGUF_KEY_WIN_LENGTH: &str = "vokra.redimnet.win_length";
/// `vokra.redimnet.sample_rate` — audio sample rate (16 kHz mono).
pub const GGUF_KEY_SAMPLE_RATE: &str = "vokra.redimnet.sample_rate";
/// `vokra.redimnet.f_min` — log-mel lower frequency Hz (20).
pub const GGUF_KEY_F_MIN: &str = "vokra.redimnet.f_min";
/// `vokra.redimnet.f_max` — log-mel upper frequency Hz (7600).
pub const GGUF_KEY_F_MAX: &str = "vokra.redimnet.f_max";
/// `vokra.redimnet.do_preemph` — pre-emphasis flag (1 = on).
pub const GGUF_KEY_DO_PREEMPH: &str = "vokra.redimnet.do_preemph";

/// Primary-source anchor: base ReDimNet reference implementation.
/// Cited in the loud-partial error so a reader diagnosing this gap
/// knows the base topology to walk.
const PRIMARY_SOURCE_BASE: &str =
    "github.com/wenet-e2e/wespeaker/blob/master/wespeaker/models/redimnet.py";
/// Primary-source anchor: ReDimNet2 (the "B6-LM" release) topology
/// wrap. Cited in the loud-partial error so a reader knows the
/// upstream ReDimNet2Wrap sequence to walk.
const PRIMARY_SOURCE_REDIMNET2: &str =
    "github.com/wenet-e2e/wespeaker/blob/master/wespeaker/models/redimnet2.py";
/// Paper anchor (Yakovlev et al. 2024) — cited alongside the source
/// URLs so a reader has the theoretical context as well.
const PRIMARY_SOURCE_PAPER: &str = "arxiv.org/abs/2402.01049";

// ---------------------------------------------------------------------------
// ReDimNetConfig — the topology axes read from the `vokra.redimnet.*`
// chunk group. STRICT: every axis is required (FR-EX-08 — no
// primary-source constant fallback since a partial stamp would
// fabricate axes without primary-source backing; the converter always
// stamps every axis so a proper conversion carries the full group).
// ---------------------------------------------------------------------------

/// ReDimNet2 "B6-LM" hyperparameters as they ride the
/// `vokra.redimnet.*` chunk group.
///
/// [`from_gguf`](Self::from_gguf) is a **strict** loader: every axis
/// is required (FR-EX-08 — never a silent primary-source constant
/// fallback because the fallback would fabricate axes the runtime
/// then binds against). A GGUF missing any `vokra.redimnet.*` chunk
/// is rejected loudly with a [`VokraError::ModelLoad`] naming the
/// absent key.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReDimNetConfig {
    /// Speaker embedding dimension (`embed_dim`), typically 192.
    pub embed_dim: u32,
    /// Output channel count of the last 1D `conv+att` block
    /// (`out_channels`), typically 224.
    pub out_channels: u32,
    /// ReDimNet2 channel expansion base (`C`), typically 64.
    pub c: u32,
    /// Mel-frequency dim after the 2D stem (`F`), typically 72.
    /// Matches [`Self::n_mels`] (the 2D dim-reduction stem preserves
    /// the input mel-spec frequency axis).
    pub f: u32,
    /// Log-mel filterbank count (`n_mels`), typically 72.
    pub n_mels: u32,
    /// STFT window size (`n_fft`), typically 512.
    pub n_fft: u32,
    /// STFT hop size (`hop_length`), typically 160 (10 ms @ 16 kHz).
    pub hop_length: u32,
    /// STFT window length (`win_length`), typically 400 (25 ms @ 16 kHz).
    pub win_length: u32,
    /// Audio sample rate, typically 16000 (16 kHz mono).
    pub sample_rate: u32,
    /// Log-mel lower frequency Hz (`f_min`), typically 20.
    pub f_min: u32,
    /// Log-mel upper frequency Hz (`f_max`), typically 7600.
    pub f_max: u32,
    /// Pre-emphasis flag (`do_preemph`), typically 1 (0.97
    /// pre-emphasis on the raw waveform).
    pub do_preemph: u32,
}

impl ReDimNetConfig {
    /// The ReDimNet2 "B6-LM" defaults transcribed from the upstream
    /// `Wespeaker/wespeaker-voxceleb-redimnet2-B6-LM/config.yaml`.
    /// Used by the unit tests and as a diagnostic reference — the
    /// runtime loader does NOT default to these; it reads the
    /// stamped values and fails loud on any missing chunk (see
    /// [`Self::from_gguf`]).
    #[must_use]
    pub const fn b6_lm_default() -> Self {
        Self {
            embed_dim: 192,
            out_channels: 224,
            c: 64,
            f: 72,
            n_mels: 72,
            n_fft: 512,
            hop_length: 160,
            win_length: 400,
            sample_rate: 16000,
            f_min: 20,
            f_max: 7600,
            do_preemph: 1,
        }
    }

    /// Reads every `vokra.redimnet.*` chunk from `gguf`. Missing axis
    /// = loud [`VokraError::ModelLoad`] naming the absent key
    /// (FR-EX-08 — no primary-source constant fallback).
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] when any of the 12 mandatory
    ///   `vokra.redimnet.*` u32 chunks is absent.
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        fn req_u32(gguf: &GgufFile, key: &str) -> Result<u32> {
            gguf.get(key)
                .and_then(|v| v.as_u64())
                .map(|v| v as u32)
                .ok_or_else(|| {
                    VokraError::ModelLoad(format!(
                        "redimnet: GGUF is missing required u32 chunk `{key}` — the \
                         upstream `Wespeaker/wespeaker-voxceleb-redimnet2-B6-LM` release \
                         carries a first-class `config.yaml`, and the converter transcribes \
                         every axis from it and stamps them, so a proper conversion carries \
                         the full `vokra.redimnet.*` chunk group. This binder refuses to \
                         fabricate topology axes from primary-source constants (FR-EX-08). \
                         Re-run `vokra-cli convert --model redimnet` against a safetensors \
                         checkpoint flattened via `tools/parity/nemo_pt_to_safetensors.py`."
                    ))
                })
        }
        Ok(Self {
            embed_dim: req_u32(gguf, GGUF_KEY_EMBED_DIM)?,
            out_channels: req_u32(gguf, GGUF_KEY_OUT_CHANNELS)?,
            c: req_u32(gguf, GGUF_KEY_C)?,
            f: req_u32(gguf, GGUF_KEY_F)?,
            n_mels: req_u32(gguf, GGUF_KEY_N_MELS)?,
            n_fft: req_u32(gguf, GGUF_KEY_N_FFT)?,
            hop_length: req_u32(gguf, GGUF_KEY_HOP_LENGTH)?,
            win_length: req_u32(gguf, GGUF_KEY_WIN_LENGTH)?,
            sample_rate: req_u32(gguf, GGUF_KEY_SAMPLE_RATE)?,
            f_min: req_u32(gguf, GGUF_KEY_F_MIN)?,
            f_max: req_u32(gguf, GGUF_KEY_F_MAX)?,
            do_preemph: req_u32(gguf, GGUF_KEY_DO_PREEMPH)?,
        })
    }
}

// ---------------------------------------------------------------------------
// ReDimNetWeights — bound the tensor manifest with a non-emptiness
// gate. Under the loud-partial WP the weights are counted but the
// `basic_resnet` + `conv+att` + `ASTP` forward is deferred (the
// three-block encode pipeline would consume them). Mirrors the
// `BeatThisWeights` / `Mt3Weights` posture.
// ---------------------------------------------------------------------------

/// Weight tensors bound from a ReDimNet GGUF.
///
/// **Contract**: [`from_gguf`](Self::from_gguf) is a *loud*
/// verification step. A GGUF that carries zero tensors is rejected
/// with [`VokraError::ModelLoad`] (FR-EX-08 — an empty GGUF is never
/// a valid ReDimNet checkpoint).
///
/// Under the current landing this struct stores the tensor names +
/// dims discovered on disk. The 2D `basic_resnet` + 1D `conv+att` +
/// ASTP forward is deferred (see [`ReDimNet::encode`] loud-partial),
/// so the payload is not yet dequantised — the follow-up wave sizes
/// the dequant per its kernel needs.
#[derive(Debug)]
pub struct ReDimNetWeights {
    /// Tensors discovered on disk, indexed by upstream `state_dict`
    /// name with their GGUF-side dims. Used by the load-time
    /// non-emptiness gate and by the future follow-up encode forward
    /// wave.
    tensors: Vec<(String, Vec<usize>)>,
}

impl ReDimNetWeights {
    /// Scans `gguf` for the ReDimNet state_dict tensors. Refuses to
    /// bind if the GGUF carries zero tensors (FR-EX-08 — an empty
    /// GGUF is never a valid ReDimNet checkpoint).
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
                "redimnet: GGUF carries zero tensors — refusing to bind an all-zero \
                 forward (FR-EX-08). Re-run `vokra-cli convert --model redimnet` \
                 against a safetensors checkpoint flattened via \
                 `tools/parity/nemo_pt_to_safetensors.py`."
                    .to_owned(),
            ));
        }
        Ok(Self { tensors })
    }

    /// Number of tensors bound from the GGUF. Purely a diagnostic
    /// accessor — the encode forward wave uses it to size its
    /// expectations.
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }
}

// ---------------------------------------------------------------------------
// ReDimNet — the runtime binder handle
// ---------------------------------------------------------------------------

/// ReDimNet2 "B6-LM" speaker-embedding encoder (WeSpeaker/VoxCeleb,
/// apache-2.0).
///
/// Bind with [`from_gguf`](Self::from_gguf), then call
/// [`encode`](Self::encode) on a log-mel fbank buffer to obtain a
/// 192-d speaker embedding. See the module doc for the current
/// implementation-status matrix and the FR-EX-08 loud-error contract
/// on the encode forward.
#[derive(Debug)]
pub struct ReDimNet {
    config: ReDimNetConfig,
    // The bound weights are held (real, counted) but the encode
    // forward (basic_resnet 2D stem + conv+att 1D blocks + ASTP
    // pooling) is a follow-up wave; the field is deliberately
    // `#[allow(dead_code)]` until the kernel lands so a reader is not
    // misled by an unused field. Same posture as RMVPE / pyannote /
    // Charsiu / beat_this / mt3.
    #[allow(dead_code)]
    weights: ReDimNetWeights,
    weight_license: LicenseClass,
}

impl ReDimNet {
    /// Binds a ReDimNet GGUF: validates arch, reads the strict
    /// topology chunk group, discovers tensors, and surfaces the
    /// stamped weight-license class for compliance gate cross-checks.
    ///
    /// This binder is a *loud* validation step. Every failure is a
    /// distinct [`VokraError::ModelLoad`] naming the missing / wrong
    /// key so a reader diagnosing a mis-produced GGUF has exactly one
    /// place to walk (FR-EX-08 — never a silent partial bind).
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] when `vokra.model.arch` is absent
    ///   or not `"redimnet"` (a `wespeaker` / `ecapa_tdnn` /
    ///   `titanet` / `speaker_3d` / `campplus` GGUF handed to us by
    ///   mistake fails with a clear message instead of a downstream
    ///   "missing tensor" — same pattern as `Mt3::from_gguf`).
    /// - [`VokraError::ModelLoad`] when any `vokra.redimnet.*` chunk
    ///   is absent ([`ReDimNetConfig::from_gguf`] is strict — no
    ///   primary-source constant fallback).
    /// - [`VokraError::ModelLoad`] when the GGUF carries zero
    ///   tensors ([`ReDimNetWeights::from_gguf`] refuses to bind an
    ///   all-zero forward).
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        // 1. Arch check — always first so a mis-typed model handed
        //    here fails with a specific message instead of a
        //    downstream "vokra.redimnet.embed_dim missing".
        match file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()) {
            Some(a) if a == ARCH => {}
            Some(other) => {
                return Err(VokraError::ModelLoad(format!(
                    "redimnet: GGUF arch is `{other}`, expected `{ARCH}` (was this GGUF \
                     produced by `vokra-cli convert --model redimnet`? Note that sibling \
                     speaker-fleet arches — `wespeaker` (ResNet-34 backbone), \
                     `ecapa_tdnn` (TDNN stack), `titanet` (depth-wise separable Conv1D \
                     backbone), `speaker_3d` (ERes2Net backbone), `campplus` (CAM++) \
                     — are all distinct topologies)"
                )));
            }
            None => {
                return Err(VokraError::ModelLoad(
                    "redimnet: GGUF is missing `vokra.model.arch` (converter did not \
                     stamp it — this is not a Vokra-native redimnet GGUF)"
                        .to_owned(),
                ));
            }
        }

        // 2. Strict topology axes from the `vokra.redimnet.*` chunk
        //    group.
        let config = ReDimNetConfig::from_gguf(file)?;

        // 3. Load the tensor manifest with the non-emptiness gate.
        let weights = ReDimNetWeights::from_gguf(file)?;

        // 4. Provenance surfacing — read the stamped weight-license
        //    class for compliance gate cross-checks (defaults to
        //    `Unknown` if absent, which is fail-closed at the gate).
        //    The ReDimNet converter stamps `Permissive` in production
        //    per the apache-2.0 default. Not raising a `ModelLoad`
        //    on missing provenance keeps the binder able to load
        //    hand-assembled GGUFs the test harness uses without
        //    forcing every fixture to stamp the full provenance chunk.
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

    /// The bound topology axes (read from the `vokra.redimnet.*`
    /// chunk group).
    #[inline]
    #[must_use]
    pub const fn config(&self) -> &ReDimNetConfig {
        &self.config
    }

    /// The stamped weight-license class surfaced from the GGUF's
    /// `vokra.provenance.weight_license` chunk. The ReDimNet
    /// converter stamps `Permissive` in production per the
    /// apache-2.0 default; a GGUF missing the stamp reads back as
    /// [`LicenseClass::Unknown`] (fail-closed).
    #[inline]
    #[must_use]
    pub const fn weight_license(&self) -> LicenseClass {
        self.weight_license
    }

    /// Number of tensors bound from the GGUF. Purely a diagnostic
    /// accessor — the encode forward wave uses it to size its
    /// expectations.
    #[inline]
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.weights.tensor_count()
    }

    /// Encodes a log-mel fbank buffer to a 192-d speaker embedding.
    ///
    /// # Loud-partial (this WP)
    ///
    /// Returns [`VokraError::UnsupportedOp`] — the ReDimNet2 "B6-LM"
    /// **encode forward** requires transcribing three distinct
    /// pieces from the WeSpeaker Python source:
    ///
    /// 1. The **2D `basic_resnet` block sequence** (dim-reduction
    ///    stem) — needs a walk against
    ///    `github.com/wenet-e2e/wespeaker/blob/master/wespeaker/models/redimnet2.py`
    ///    `ReDimNet2Wrap` for the block ordering and channel plan.
    /// 2. The **1D `conv+att` block sequence** (Conformer-lite variant
    ///    distinct from `vokra_ops::conformer` full Conformer — the
    ///    ReDimNet2 body uses a lighter-weight `conv + attention`
    ///    combo that no sibling supplies) — needs a walk against
    ///    `github.com/wenet-e2e/wespeaker/blob/master/wespeaker/models/redimnet.py`
    ///    base ReDimNet.
    /// 3. The **ASTP (Attentive Statistics Pooling) head** — needs a
    ///    walk against `wespeaker/pooling_layers.py`
    ///    `AttentiveStatisticsPool2d` for the pooling kernel plus a
    ///    linear projection to `embed_dim`.
    ///
    /// The error message names all three pieces + primary-source URLs
    /// so a reader diagnosing this gap has exactly three anchors to
    /// walk. Every config axis is echoed so the reader can
    /// cross-check what topology the follow-up wave targets.
    ///
    /// # Errors
    ///
    /// - [`VokraError::UnsupportedOp`] — the loud-partial gate for
    ///   the deferred encode forward.
    pub fn encode(&self, fbank: &[f32]) -> Result<Vec<f32>> {
        // Bind unused arg so a `#[warn(unused_variables)]` change
        // does not silently mask the loud-partial fire path; the
        // future real implementation will consume it.
        let _ = fbank;
        Err(encode_forward_loud_partial(&self.config))
    }
}

/// Constructs the loud-partial [`VokraError::UnsupportedOp`]
/// returned by [`ReDimNet::encode`] until the WeSpeaker Python
/// source transcription wave lands (2D basic_resnet stem + 1D
/// conv+att blocks + ASTP head).
///
/// Names **three** primary source anchors (base ReDimNet reference
/// + ReDimNet2 wrap reference + arXiv paper) so a reader diagnosing
/// the gap has exactly three places to walk (RMVPE / pyannote / snac
/// / hifigan / beat_this / mt3 Wave 1-3 loud-partial-message
/// precedent — CLAUDE.md 教訓 (a)).
fn encode_forward_loud_partial(cfg: &ReDimNetConfig) -> VokraError {
    VokraError::UnsupportedOp(format!(
        "redimnet encode: ReDimNet2 encode forward pending — the WeSpeaker Python source \
         transcription wave has not landed. Three pieces are missing: \
         (i) the 2D `basic_resnet` block sequence (dim-reduction stem), \
         (ii) the 1D `conv+att` block sequence (Conformer-lite variant — distinct from \
         `vokra_ops::conformer` full Conformer; ReDimNet2 uses a lighter-weight `conv + \
         attention` combo that no sibling in the tree supplies), and \
         (iii) the ASTP (Attentive Statistics Pooling) head + linear projection to \
         embed_dim. Config: embed_dim={embed}, out_channels={out}, c={c}, f={f}, \
         n_mels={n_mels}, n_fft={n_fft}, hop_length={hop}, win_length={win}, \
         sample_rate={sr}, f_min={fmin}, f_max={fmax}, do_preemph={preemph}. \
         Primary sources: {base} + {redimnet2} + {paper}. Loud pending (CLAUDE.md 教訓 \
         (a) — 'loud-partial は fake-complete より honest') — no silent fabricated \
         speaker embedding ever emitted (FR-EX-08).",
        embed = cfg.embed_dim,
        out = cfg.out_channels,
        c = cfg.c,
        f = cfg.f,
        n_mels = cfg.n_mels,
        n_fft = cfg.n_fft,
        hop = cfg.hop_length,
        win = cfg.win_length,
        sr = cfg.sample_rate,
        fmin = cfg.f_min,
        fmax = cfg.f_max,
        preemph = cfg.do_preemph,
        base = PRIMARY_SOURCE_BASE,
        redimnet2 = PRIMARY_SOURCE_REDIMNET2,
        paper = PRIMARY_SOURCE_PAPER,
    ))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    //! Tests for the ReDimNet runtime binder — round-trip on the
    //! topology chunk group + negative-space round-trip on the
    //! loud-partial gates + arch-tag distinctness pin.
    //!
    //! # What "round-trip" means here
    //!
    //! The task spec asks for 5+ unit tests. On real fbank this would
    //! be `encode(...)` returning a real 192-d speaker embedding, but
    //! the ReDimNet2 encode forward (basic_resnet + conv+att + ASTP)
    //! has not been transcribed from the WeSpeaker Python source
    //! (see the module doc + [`ReDimNet::encode`] rustdoc).
    //! Fabricating a real-fbank output would violate CLAUDE.md 教訓
    //! (a) ("loud-partial は fake-complete より honest").
    //!
    //! The round-trip semantics we *can* honestly test:
    //!
    //! 1. **Config default pin**: the ReDimNet2 B6-LM axes match the
    //!    upstream `config.yaml` transcription.
    //! 2. **Config round-trip**: `from_gguf` reads every axis stamped
    //!    by the converter (full 12-axis metadata round-trip).
    //! 3. **Loud-error negative-space round-trip**: every stated
    //!    blocker (missing arch / wrong arch / missing chunk / empty
    //!    tensor list / unsupported forward surface) fires at its
    //!    documented surface point, in the documented error variant.
    //! 4. **Arch-tag distinctness pin**: the arch string is stable
    //!    and distinct from every sibling speaker-fleet arch.
    use super::*;
    use vokra_core::gguf::{GgmlType, GgufBuilder, GgufFile};

    /// Builds a minimal ReDimNet GGUF carrying the arch tag + full
    /// `vokra.redimnet.*` chunk group + one representative tensor
    /// whose outer dim matches the given `embed_dim`.
    /// `weight_license_class` is written under
    /// `vokra.provenance.weight_license` (or omitted if `None`).
    fn redimnet_gguf(cfg: ReDimNetConfig, weight_license_class: Option<LicenseClass>) -> GgufFile {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_string(chunks::KEY_MODEL_NAME, "wespeaker-voxceleb-redimnet2-b6-lm");
        b.add_u32(GGUF_KEY_EMBED_DIM, cfg.embed_dim);
        b.add_u32(GGUF_KEY_OUT_CHANNELS, cfg.out_channels);
        b.add_u32(GGUF_KEY_C, cfg.c);
        b.add_u32(GGUF_KEY_F, cfg.f);
        b.add_u32(GGUF_KEY_N_MELS, cfg.n_mels);
        b.add_u32(GGUF_KEY_N_FFT, cfg.n_fft);
        b.add_u32(GGUF_KEY_HOP_LENGTH, cfg.hop_length);
        b.add_u32(GGUF_KEY_WIN_LENGTH, cfg.win_length);
        b.add_u32(GGUF_KEY_SAMPLE_RATE, cfg.sample_rate);
        b.add_u32(GGUF_KEY_F_MIN, cfg.f_min);
        b.add_u32(GGUF_KEY_F_MAX, cfg.f_max);
        b.add_u32(GGUF_KEY_DO_PREEMPH, cfg.do_preemph);
        if let Some(cls) = weight_license_class {
            b.add_string(chunks::KEY_PROVENANCE_WEIGHT_LICENSE, cls.as_str());
        }
        // One representative tensor so the non-emptiness gate passes.
        // The `embed_dim` dim is deliberately at axis 0 so a future
        // shape-consistency check has something to walk.
        let d = u64::from(cfg.embed_dim);
        b.add_tensor(
            "speaker.head.projection.weight",
            GgmlType::F32,
            vec![d, d],
            vec![0u8; (d * d * 4) as usize],
        )
        .expect("add_tensor");
        GgufFile::parse(b.to_bytes().expect("serialize")).expect("parse")
    }

    // -----------------------------------------------------------------------
    // 1. ReDimNetConfig default matches config.yaml transcription
    // -----------------------------------------------------------------------

    #[test]
    fn config_default_matches_config_yaml_axes() {
        // Pin the B6-LM hparams transcribed from
        // Wespeaker/wespeaker-voxceleb-redimnet2-B6-LM/config.yaml.
        // A rename or axis-value change would land here in the same
        // commit or fail this test.
        let cfg = ReDimNetConfig::b6_lm_default();
        assert_eq!(cfg.embed_dim, 192);
        assert_eq!(cfg.out_channels, 224);
        assert_eq!(cfg.c, 64);
        assert_eq!(cfg.f, 72);
        assert_eq!(cfg.n_mels, 72);
        assert_eq!(cfg.n_fft, 512);
        assert_eq!(cfg.hop_length, 160);
        assert_eq!(cfg.win_length, 400);
        assert_eq!(cfg.sample_rate, 16000);
        assert_eq!(cfg.f_min, 20);
        assert_eq!(cfg.f_max, 7600);
        assert_eq!(cfg.do_preemph, 1);
        // Structural invariant: the 2D stem preserves the input
        // mel-frequency axis, so `f == n_mels` at the config level.
        assert_eq!(
            cfg.f, cfg.n_mels,
            "ReDimNet2 dim-reduction stem preserves the frequency axis: F must equal n_mels"
        );
    }

    // -----------------------------------------------------------------------
    // 2. from_gguf full topology chunk-group round-trip
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_metadata_round_trip() {
        let cfg = ReDimNetConfig::b6_lm_default();
        let file = redimnet_gguf(cfg, Some(LicenseClass::Permissive));
        let m = ReDimNet::from_gguf(&file).expect("valid GGUF must bind");
        // Config round-trip — every axis stamped by the converter
        // reads back into the same ReDimNetConfig value.
        assert_eq!(*m.config(), cfg);
        // Weight-license surface (ReDimNet converter stamps
        // Permissive per apache-2.0 default).
        assert_eq!(m.weight_license(), LicenseClass::Permissive);
        assert!(m.tensor_count() >= 1);
    }

    // -----------------------------------------------------------------------
    // 3. from_gguf rejects wrong arch (never silently mis-routes)
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_rejects_wrong_arch() {
        // A `wespeaker` (ResNet-34) GGUF handed to the ReDimNet binder
        // by mistake must fail loud with a specific message rather
        // than silently mis-binding (FR-EX-08).
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, "wespeaker");
        b.add_u32(GGUF_KEY_EMBED_DIM, 192);
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let Err(err) = ReDimNet::from_gguf(&file) else {
            panic!("expected ModelLoad on wrong arch");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains("`wespeaker`") && m.contains("`redimnet`"),
                    "message must name both the got and expected arch tags, got `{m}`"
                );
                assert!(
                    m.contains("ResNet-34"),
                    "message should disambiguate wespeaker's topology to help \
                     the reader, got `{m}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // 4. encode returns UnsupportedOp with primary-source anchors +
    //    every config axis echoed
    // -----------------------------------------------------------------------

    #[test]
    fn encode_loud_partial_returns_unsupported_op() {
        let cfg = ReDimNetConfig::b6_lm_default();
        let file = redimnet_gguf(cfg, Some(LicenseClass::Permissive));
        let m = ReDimNet::from_gguf(&file).unwrap();
        // 1 second of legitimate-shape fbank (72 mels × 100 frames)
        // so the loud-partial gate fires (not some pre-encode
        // validation).
        let fbank = vec![0.0f32; 72 * 100];
        let Err(err) = m.encode(&fbank) else {
            panic!("encode must loud-partial");
        };
        match err {
            VokraError::UnsupportedOp(msg) => {
                assert!(
                    msg.contains("redimnet encode"),
                    "message must call out the redimnet encode surface, got `{msg}`"
                );
                // All three missing pieces must be named by exact
                // identifier so the follow-up wave knows what to walk.
                assert!(
                    msg.contains("basic_resnet"),
                    "message must name the 2D basic_resnet stem gap, got `{msg}`"
                );
                assert!(
                    msg.contains("conv+att"),
                    "message must name the 1D conv+att block gap, got `{msg}`"
                );
                assert!(
                    msg.contains("ASTP"),
                    "message must name the ASTP pooling head gap, got `{msg}`"
                );
                // Primary-source URLs must be cited so a reader
                // diagnosing the gap has anchors to walk.
                assert!(
                    msg.contains("wenet-e2e/wespeaker"),
                    "message must contain the primary source URL substring \
                     (wenet-e2e/wespeaker), got `{msg}`"
                );
                assert!(
                    msg.contains("redimnet2.py") && msg.contains("redimnet.py"),
                    "message must cite BOTH the redimnet.py (base) + redimnet2.py \
                     (wrap) references, got `{msg}`"
                );
                assert!(
                    msg.contains("2402.01049"),
                    "message must cite the arXiv paper anchor, got `{msg}`"
                );
                // Every config axis must be echoed so the reader can
                // cross-check what topology the follow-up wave targets.
                assert!(
                    msg.contains("embed_dim=192"),
                    "embed_dim axis missing: {msg}"
                );
                assert!(
                    msg.contains("out_channels=224"),
                    "out_channels axis missing: {msg}"
                );
                assert!(msg.contains("c=64"), "c axis missing: {msg}");
                assert!(msg.contains("f=72"), "f axis missing: {msg}");
                assert!(msg.contains("n_mels=72"), "n_mels axis missing: {msg}");
                assert!(msg.contains("n_fft=512"), "n_fft axis missing: {msg}");
                assert!(
                    msg.contains("hop_length=160"),
                    "hop_length axis missing: {msg}"
                );
                assert!(
                    msg.contains("win_length=400"),
                    "win_length axis missing: {msg}"
                );
                assert!(
                    msg.contains("sample_rate=16000"),
                    "sample_rate axis missing: {msg}"
                );
                assert!(msg.contains("f_min=20"), "f_min axis missing: {msg}");
                assert!(msg.contains("f_max=7600"), "f_max axis missing: {msg}");
                assert!(
                    msg.contains("do_preemph=1"),
                    "do_preemph axis missing: {msg}"
                );
            }
            other => panic!("expected VokraError::UnsupportedOp, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // 5. Structural pin — arch tag is stable and distinct from every
    //    sibling speaker-fleet arch
    // -----------------------------------------------------------------------

    #[test]
    fn arch_tag_distinct_from_sibling_speaker_arches() {
        // Pin the arch string so a rename would land here in the same
        // commit or fail this test. The sibling speaker-fleet arches
        // MUST NOT collide with ours.
        assert_eq!(ARCH, "redimnet");
        assert_ne!(
            ARCH, "wespeaker",
            "redimnet (ReDimNet2 2D-stem+conv+att backbone) and wespeaker \
             (ResNet-34 backbone) are different topologies — sharing arch \
             would mis-route runtime dispatch (FR-EX-08)"
        );
        assert_ne!(
            ARCH, "ecapa_tdnn",
            "redimnet and ecapa_tdnn (TDNN stack backbone) are different \
             topologies — sharing arch would mis-route (FR-EX-08)"
        );
        assert_ne!(
            ARCH, "titanet",
            "redimnet and titanet (depth-wise separable Conv1D backbone) \
             are different topologies — sharing arch would mis-route (FR-EX-08)"
        );
        assert_ne!(
            ARCH, "speaker_3d",
            "redimnet and speaker_3d (ERes2Net backbone) are different \
             topologies — sharing arch would mis-route (FR-EX-08)"
        );
        assert_ne!(
            ARCH, "campplus",
            "redimnet and campplus (CAM++ backbone) are different topologies \
             — sharing arch would mis-route (FR-EX-08)"
        );
    }

    // -----------------------------------------------------------------------
    // 6. Missing topology chunk fails loud (no primary-source fallback)
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_rejects_missing_topology_chunk() {
        // Correct arch but missing one of the mandatory
        // `vokra.redimnet.*` chunks — a partially-stamped GGUF must
        // be caught here, not silently defaulted to a fabricated axis
        // (FR-EX-08 — the converter always stamps every axis, so a
        // missing chunk always signals a partial / mis-produced GGUF).
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_u32(GGUF_KEY_EMBED_DIM, 192);
        b.add_u32(GGUF_KEY_OUT_CHANNELS, 224);
        b.add_u32(GGUF_KEY_C, 64);
        b.add_u32(GGUF_KEY_F, 72);
        // deliberately omit n_mels
        b.add_u32(GGUF_KEY_N_FFT, 512);
        b.add_u32(GGUF_KEY_HOP_LENGTH, 160);
        b.add_u32(GGUF_KEY_WIN_LENGTH, 400);
        b.add_u32(GGUF_KEY_SAMPLE_RATE, 16000);
        b.add_u32(GGUF_KEY_F_MIN, 20);
        b.add_u32(GGUF_KEY_F_MAX, 7600);
        b.add_u32(GGUF_KEY_DO_PREEMPH, 1);
        b.add_tensor(
            "speaker.head.projection.weight",
            GgmlType::F32,
            vec![4, 4],
            vec![0u8; 16 * 4],
        )
        .expect("add_tensor");
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let Err(err) = ReDimNet::from_gguf(&file) else {
            panic!("expected ModelLoad on missing n_mels chunk");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains(GGUF_KEY_N_MELS),
                    "message must name the missing n_mels key, got `{m}`"
                );
                assert!(
                    m.contains("config.yaml"),
                    "message should mention the upstream config.yaml transcription \
                     path so the reader knows the fallback rationale, got `{m}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // 7. Empty tensor manifest fails loud (never binds all-zero forward)
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_rejects_empty_tensor_list() {
        // Correct arch + full chunk group but zero tensors — the
        // ReDimNetWeights non-emptiness gate must fire.
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_u32(GGUF_KEY_EMBED_DIM, 192);
        b.add_u32(GGUF_KEY_OUT_CHANNELS, 224);
        b.add_u32(GGUF_KEY_C, 64);
        b.add_u32(GGUF_KEY_F, 72);
        b.add_u32(GGUF_KEY_N_MELS, 72);
        b.add_u32(GGUF_KEY_N_FFT, 512);
        b.add_u32(GGUF_KEY_HOP_LENGTH, 160);
        b.add_u32(GGUF_KEY_WIN_LENGTH, 400);
        b.add_u32(GGUF_KEY_SAMPLE_RATE, 16000);
        b.add_u32(GGUF_KEY_F_MIN, 20);
        b.add_u32(GGUF_KEY_F_MAX, 7600);
        b.add_u32(GGUF_KEY_DO_PREEMPH, 1);
        // NO tensors added.
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let Err(err) = ReDimNet::from_gguf(&file) else {
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
}
