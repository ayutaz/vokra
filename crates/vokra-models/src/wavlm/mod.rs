//! **WavLM Base+ SV** (`microsoft/wavlm-base-plus-sv`, CC-BY-SA-3.0)
//! — WavLM speaker-verification encoder (Chen et al. arXiv:2110.13900
//! "WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack
//! Speech Processing") — runtime binder for the `wavlm_sv` converter
//! arch.
//!
//! # Runtime layout (loud-partial, speaker-fleet posture per
//! CLAUDE.md 教訓 (a))
//!
//! ```text
//! raw waveform (mono f32, [T] 16 kHz)
//!   -> 7-layer 1D conv feature extractor stem            ← **loud-partial**
//!        (HuBERT/wav2vec2 lineage: `conv_dim` × `conv_stride` ×
//!         `conv_kernel` per axis-array chunk group,
//!         `feat_extract_norm = group` for Base+. The strided conv
//!         stem downsamples 16 kHz raw audio to a ~50 Hz feature grid.)
//!   -> WavLM Transformer encoder (12 layers, 768 hidden, 12 heads) ← **loud-partial**
//!        (Distinct from vanilla Transformer AND from HuBERT/wav2vec2:
//!         WavLM adds a **gated relative position bias** + a
//!         **convolutional position-bias fusion** on top of the
//!         attention softmax — the primitive that neither wav2vec2
//!         nor HuBERT expose, and that no sibling in the tree
//!         supplies today. Requires a walk against
//!         `github.com/microsoft/UniSpeech/tree/main/WavLM/WavLM.py`
//!         `TransformerSentenceEncoderLayer::forward` + the gated
//!         relative-position-bias primitive.)
//!   -> XVector head (5-block TDNN + statistics pooling)   ← **loud-partial**
//!        (5-block TDNN with kernels `[5,3,3,1,1]` + dilations
//!         `[1,2,3,1,1]` per axis-array chunk group → statistics
//!         pooling → 512-d embedding through Additive Margin Softmax.
//!         Requires a walk against
//!         `github.com/microsoft/UniSpeech/tree/main/downstreams/speaker_verification`
//!         `XVectorHead` + the AM-Softmax loss head.)
//!   -> 512-d speaker embedding (`xvector_output_dim`)
//! ```
//!
//! # Loud-partial classification (design § — CLAUDE.md 教訓 (a))
//!
//! - **Real (this WP)**: [`WavLmSv::from_gguf`] with strict
//!   `vokra.model.arch == "wavlm_sv"` validation + strict
//!   `vokra.wavlm.*` scalar + axis-array chunk-group presence
//!   enforcement (every axis required — no primary-source constant
//!   fallback because the converter transcribes the axes from the
//!   upstream `config.json` and stamps them, and this binder mirrors
//!   those stamps rather than silently defaulting to a fabricated
//!   axis), [`WavLmSvWeights::from_gguf`] with a floor of non-empty
//!   tensor count enforced loud (a GGUF that carries no
//!   WavLM-typical tensors is refused rather than silently running
//!   an all-zero forward), license-class surfacing.
//! - **Loud-partial (this WP)**: [`WavLmSv::encode`] returns
//!   [`VokraError::UnsupportedOp`] naming the three exact missing
//!   pieces:
//!   (i) 7-layer 1D conv feature-extractor stem walk,
//!   (ii) WavLM Transformer encoder walk (gated relative position bias
//!        + convolutional position-bias fusion — the WavLM-specific
//!        primitive that neither wav2vec2 nor HuBERT expose),
//!   (iii) XVector head + Additive Margin Softmax walk.
//!   Every message echoes every config axis so the reader can
//!   cross-check what topology the follow-up wave targets.
//!
//! Rationale (RMVPE / pyannote / hifigan / vocos / bigvgan / snac /
//! beat_this / mt3 / redimnet Wave 1-4 precedent, CLAUDE.md 教訓 (a)):
//! the surrounding scaffold + `from_gguf` chunk-group validation +
//! FR-EX-08 loud-fails land today so a follow-up wave can flip the
//! switch by transcribing the WavLM Python `WavLM/WavLM.py` topology
//! + the UniSpeech `downstreams/speaker_verification` XVector head
//! + writing the encode forward against those axes. The
//! [`VokraError::UnsupportedOp`] message cites the HF card + the
//! UniSpeech GitHub tree + the arXiv paper so a reader diagnosing
//! this gap has exactly three anchors to walk.
//!
//! # `vokra.wavlm.*` chunk group (read here)
//!
//! Written by `vokra-convert::models::wavlm_sv::convert_wavlm_sv_file`:
//!
//! - `vokra.model.arch` (`String`): must equal [`ARCH`] (`"wavlm_sv"`).
//!   Distinct from every sibling speaker-fleet arch (`campplus`,
//!   `wespeaker`, `ecapa_tdnn`, `titanet`, `speaker_3d`, `redimnet`)
//!   — silently sharing would misroute runtime dispatch (FR-EX-08).
//! - `vokra.model.name` (`String`): `"wavlm-base-plus-sv"` — auxiliary
//!   check.
//! - Scalar topology: `vokra.wavlm.{hidden_size, num_hidden_layers,
//!   num_attention_heads, intermediate_size, num_feat_extract_layers,
//!   xvector_output_dim, num_ctc_classes, num_conv_pos_embeddings,
//!   num_conv_pos_embedding_groups, sample_rate,
//!   layer_norm_eps_scaled_1e9, feat_extract_norm_group,
//!   hidden_dropout_scaled_1e3}` (`u32` each).
//! - Axis arrays (`u32` each, indexed `_0` .. `_6` or `_0` .. `_4`):
//!   `vokra.wavlm.{conv_dim, conv_stride, conv_kernel}_{0..6}` (7 each);
//!   `vokra.wavlm.{tdnn_dim, tdnn_kernel, tdnn_dilation}_{0..4}`
//!   (5 each).
//! - `vokra.provenance.*`: license class + raw license string so the
//!   runtime compliance gate (FR-CP-03) can classify the artifact
//!   without re-inspecting the safetensors provenance. Defaults to
//!   `Copyleft` in production per cc-by-sa-3.0 stamp.
//!
//! # Cross-crate constant duplication
//!
//! Mirror of the converter's [`ARCH`] / `GGUF_KEY_*` — same rule
//! the sibling BF16 pass-through binders (`pyannote` / `snac` /
//! `hifigan` / `beat_this` / `mt3` / `redimnet`) use so
//! `vokra-models` does not gain a dependency edge onto
//! `vokra-convert`, preserving the layered convention
//! `vokra-ops → nothing GGUF-aware`, `vokra-core → GGUF reader`,
//! `vokra-models → GGUF binder`, `vokra-convert → GGUF writer`.
//!
//! # No ONNX / no pickle (permanent)
//!
//! WavLM ships upstream as PyTorch `pytorch_model.bin` (~377 MB);
//! this runtime **never** touches ONNX or pickle (FR-LD-05 /
//! NFR-DS-02). The `.bin` → safetensors bridge lives offline
//! through the sibling `tools/parity/nemo_pt_to_safetensors.py`
//! sidecar (uv-managed Python 3.12 per memory
//! `[[feedback-python-uses-uv]]` + `[[feedback-python-3-12]]`).

use vokra_core::gguf::{GgufFile, chunks};
use vokra_core::{LicenseClass, Result, VokraError};

// ---------------------------------------------------------------------------
// Arch / metadata-key constants — mirror of
// `crates/vokra-convert/src/models/wavlm_sv.rs` (see module docstring
// for the cross-crate duplication rationale).
// ---------------------------------------------------------------------------

/// Expected `vokra.model.arch` value written by
/// `vokra-cli convert --model wavlm-base-plus-sv`.
///
/// Distinct from every sibling speaker-fleet arch — never `campplus`
/// (CAM++), never `wespeaker` (ResNet-34), never `ecapa_tdnn` (TDNN
/// stack), never `titanet` (depth-wise separable Conv1D), never
/// `speaker_3d` (ERes2Net), never `redimnet` (2D dim-reduction + 1D
/// conv+att). Silently sharing an arch would misroute runtime
/// dispatch (FR-EX-08).
pub const ARCH: &str = "wavlm_sv";

// Scalar topology chunk keys.
pub const GGUF_KEY_HIDDEN_SIZE: &str = "vokra.wavlm.hidden_size";
pub const GGUF_KEY_NUM_HIDDEN_LAYERS: &str = "vokra.wavlm.num_hidden_layers";
pub const GGUF_KEY_NUM_ATTENTION_HEADS: &str = "vokra.wavlm.num_attention_heads";
pub const GGUF_KEY_INTERMEDIATE_SIZE: &str = "vokra.wavlm.intermediate_size";
pub const GGUF_KEY_NUM_FEAT_EXTRACT_LAYERS: &str = "vokra.wavlm.num_feat_extract_layers";
pub const GGUF_KEY_XVECTOR_OUTPUT_DIM: &str = "vokra.wavlm.xvector_output_dim";
pub const GGUF_KEY_NUM_CTC_CLASSES: &str = "vokra.wavlm.num_ctc_classes";
pub const GGUF_KEY_NUM_CONV_POS_EMBEDDINGS: &str = "vokra.wavlm.num_conv_pos_embeddings";
pub const GGUF_KEY_NUM_CONV_POS_EMBEDDING_GROUPS: &str =
    "vokra.wavlm.num_conv_pos_embedding_groups";
pub const GGUF_KEY_SAMPLE_RATE: &str = "vokra.wavlm.sample_rate";
pub const GGUF_KEY_LAYER_NORM_EPS_SCALED_1E9: &str = "vokra.wavlm.layer_norm_eps_scaled_1e9";
pub const GGUF_KEY_FEAT_EXTRACT_NORM_GROUP: &str = "vokra.wavlm.feat_extract_norm_group";
pub const GGUF_KEY_HIDDEN_DROPOUT_SCALED_1E3: &str = "vokra.wavlm.hidden_dropout_scaled_1e3";

// Axis-array chunk-key prefixes (indexed `_0` .. `_N`).
pub const GGUF_KEY_CONV_DIM_PREFIX: &str = "vokra.wavlm.conv_dim";
pub const GGUF_KEY_CONV_STRIDE_PREFIX: &str = "vokra.wavlm.conv_stride";
pub const GGUF_KEY_CONV_KERNEL_PREFIX: &str = "vokra.wavlm.conv_kernel";
pub const GGUF_KEY_TDNN_DIM_PREFIX: &str = "vokra.wavlm.tdnn_dim";
pub const GGUF_KEY_TDNN_KERNEL_PREFIX: &str = "vokra.wavlm.tdnn_kernel";
pub const GGUF_KEY_TDNN_DILATION_PREFIX: &str = "vokra.wavlm.tdnn_dilation";

/// Conv feature-extractor axis-array length — 7 (WavLM Base+
/// `num_feat_extract_layers`).
pub const CONV_AXIS_LEN: usize = 7;
/// XVector TDNN axis-array length — 5.
pub const TDNN_AXIS_LEN: usize = 5;

/// Primary-source anchor: WavLM HF card.
const PRIMARY_SOURCE_HF: &str = "huggingface.co/microsoft/wavlm-base-plus-sv";
/// Primary-source anchor: UniSpeech GitHub tree (WavLM + XVector head).
const PRIMARY_SOURCE_UNISPEECH: &str = "github.com/microsoft/UniSpeech";
/// Paper anchor (Chen et al. 2022) — cited alongside the source URLs
/// so a reader has the theoretical context as well.
const PRIMARY_SOURCE_PAPER: &str = "arxiv.org/abs/2110.13900";

// ---------------------------------------------------------------------------
// WavLmSvConfig — the topology axes read from the `vokra.wavlm.*`
// chunk group. STRICT: every axis is required (FR-EX-08 — no
// primary-source constant fallback since a partial stamp would
// fabricate axes without primary-source backing; the converter always
// stamps every axis so a proper conversion carries the full group).
// ---------------------------------------------------------------------------

/// WavLM Base+ SV hyperparameters as they ride the `vokra.wavlm.*`
/// chunk group.
///
/// [`from_gguf`](Self::from_gguf) is a **strict** loader: every axis
/// is required (FR-EX-08 — never a silent primary-source constant
/// fallback because the fallback would fabricate axes the runtime
/// then binds against). A GGUF missing any `vokra.wavlm.*` chunk is
/// rejected loudly with a [`VokraError::ModelLoad`] naming the
/// absent key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WavlmSvConfig {
    /// Transformer hidden size (typically 768 for Base+).
    pub hidden_size: u32,
    /// Transformer layer count (typically 12).
    pub num_hidden_layers: u32,
    /// Transformer attention head count (typically 12).
    pub num_attention_heads: u32,
    /// Transformer FFN intermediate size (typically 3072).
    pub intermediate_size: u32,
    /// Number of 1D conv feature-extractor layers (typically 7).
    pub num_feat_extract_layers: u32,
    /// X-Vector head output dim (speaker embedding dim, typically 512
    /// — distinct from the 192-d CAM++/ReDimNet fleet).
    pub xvector_output_dim: u32,
    /// CTC vocab size (unused for SV, stamped for audit).
    pub num_ctc_classes: u32,
    /// Convolutional positional-embedding kernel size (typically 128).
    pub num_conv_pos_embeddings: u32,
    /// Convolutional positional-embedding group count (typically 16).
    pub num_conv_pos_embedding_groups: u32,
    /// Audio sample rate (typically 16 kHz mono).
    pub sample_rate: u32,
    /// LayerNorm epsilon scaled by 1e9 (u32-encoded — typical 10_000
    /// = 1e-5 scaled).
    pub layer_norm_eps_scaled_1e9: u32,
    /// `feat_extract_norm` flag: 1 = group, 0 = layer.
    pub feat_extract_norm_group: u32,
    /// Hidden dropout scaled by 1e3 (u32-encoded — typical 100 = 0.1
    /// scaled).
    pub hidden_dropout_scaled_1e3: u32,
    /// Conv feature-extractor output channel counts (`conv_dim`)
    /// — length [`CONV_AXIS_LEN`].
    pub conv_dim: Vec<u32>,
    /// Conv feature-extractor strides (`conv_stride`) — length
    /// [`CONV_AXIS_LEN`].
    pub conv_stride: Vec<u32>,
    /// Conv feature-extractor kernels (`conv_kernel`) — length
    /// [`CONV_AXIS_LEN`].
    pub conv_kernel: Vec<u32>,
    /// XVector TDNN output channel counts (`tdnn_dim`) — length
    /// [`TDNN_AXIS_LEN`].
    pub tdnn_dim: Vec<u32>,
    /// XVector TDNN kernel sizes (`tdnn_kernel`) — length
    /// [`TDNN_AXIS_LEN`].
    pub tdnn_kernel: Vec<u32>,
    /// XVector TDNN dilations (`tdnn_dilation`) — length
    /// [`TDNN_AXIS_LEN`].
    pub tdnn_dilation: Vec<u32>,
}

impl WavlmSvConfig {
    /// The WavLM Base+ SV defaults transcribed from the upstream
    /// `microsoft/wavlm-base-plus-sv/config.json`. Used by the unit
    /// tests and as a diagnostic reference — the runtime loader does
    /// NOT default to these; it reads the stamped values and fails
    /// loud on any missing chunk (see [`Self::from_gguf`]).
    #[must_use]
    pub fn base_plus_default() -> Self {
        Self {
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            intermediate_size: 3072,
            num_feat_extract_layers: 7,
            xvector_output_dim: 512,
            num_ctc_classes: 80,
            num_conv_pos_embeddings: 128,
            num_conv_pos_embedding_groups: 16,
            sample_rate: 16000,
            layer_norm_eps_scaled_1e9: 10_000,
            feat_extract_norm_group: 1,
            hidden_dropout_scaled_1e3: 100,
            conv_dim: vec![512, 512, 512, 512, 512, 512, 512],
            conv_stride: vec![5, 2, 2, 2, 2, 2, 2],
            conv_kernel: vec![10, 3, 3, 3, 3, 2, 2],
            tdnn_dim: vec![512, 512, 512, 512, 1500],
            tdnn_kernel: vec![5, 3, 3, 1, 1],
            tdnn_dilation: vec![1, 2, 3, 1, 1],
        }
    }

    /// Reads every `vokra.wavlm.*` chunk from `gguf`. Missing axis =
    /// loud [`VokraError::ModelLoad`] naming the absent key
    /// (FR-EX-08 — no primary-source constant fallback).
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] when any of the mandatory
    ///   `vokra.wavlm.*` u32 chunks is absent.
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        fn req_u32(gguf: &GgufFile, key: &str) -> Result<u32> {
            gguf.get(key)
                .and_then(|v| v.as_u64())
                .map(|v| v as u32)
                .ok_or_else(|| {
                    VokraError::ModelLoad(format!(
                        "wavlm_sv: GGUF is missing required u32 chunk `{key}` — the \
                         upstream `microsoft/wavlm-base-plus-sv` release carries a \
                         first-class `config.json`, and the converter transcribes every \
                         axis from it and stamps them, so a proper conversion carries the \
                         full `vokra.wavlm.*` chunk group. This binder refuses to \
                         fabricate topology axes from primary-source constants (FR-EX-08). \
                         Re-run `vokra-cli convert --model wavlm-base-plus-sv` against a \
                         safetensors checkpoint flattened via \
                         `tools/parity/nemo_pt_to_safetensors.py`."
                    ))
                })
        }
        fn req_u32_array(gguf: &GgufFile, prefix: &str, len: usize) -> Result<Vec<u32>> {
            let mut out = Vec::with_capacity(len);
            for i in 0..len {
                let key = format!("{prefix}_{i}");
                out.push(req_u32(gguf, &key)?);
            }
            Ok(out)
        }
        Ok(Self {
            hidden_size: req_u32(gguf, GGUF_KEY_HIDDEN_SIZE)?,
            num_hidden_layers: req_u32(gguf, GGUF_KEY_NUM_HIDDEN_LAYERS)?,
            num_attention_heads: req_u32(gguf, GGUF_KEY_NUM_ATTENTION_HEADS)?,
            intermediate_size: req_u32(gguf, GGUF_KEY_INTERMEDIATE_SIZE)?,
            num_feat_extract_layers: req_u32(gguf, GGUF_KEY_NUM_FEAT_EXTRACT_LAYERS)?,
            xvector_output_dim: req_u32(gguf, GGUF_KEY_XVECTOR_OUTPUT_DIM)?,
            num_ctc_classes: req_u32(gguf, GGUF_KEY_NUM_CTC_CLASSES)?,
            num_conv_pos_embeddings: req_u32(gguf, GGUF_KEY_NUM_CONV_POS_EMBEDDINGS)?,
            num_conv_pos_embedding_groups: req_u32(gguf, GGUF_KEY_NUM_CONV_POS_EMBEDDING_GROUPS)?,
            sample_rate: req_u32(gguf, GGUF_KEY_SAMPLE_RATE)?,
            layer_norm_eps_scaled_1e9: req_u32(gguf, GGUF_KEY_LAYER_NORM_EPS_SCALED_1E9)?,
            feat_extract_norm_group: req_u32(gguf, GGUF_KEY_FEAT_EXTRACT_NORM_GROUP)?,
            hidden_dropout_scaled_1e3: req_u32(gguf, GGUF_KEY_HIDDEN_DROPOUT_SCALED_1E3)?,
            conv_dim: req_u32_array(gguf, GGUF_KEY_CONV_DIM_PREFIX, CONV_AXIS_LEN)?,
            conv_stride: req_u32_array(gguf, GGUF_KEY_CONV_STRIDE_PREFIX, CONV_AXIS_LEN)?,
            conv_kernel: req_u32_array(gguf, GGUF_KEY_CONV_KERNEL_PREFIX, CONV_AXIS_LEN)?,
            tdnn_dim: req_u32_array(gguf, GGUF_KEY_TDNN_DIM_PREFIX, TDNN_AXIS_LEN)?,
            tdnn_kernel: req_u32_array(gguf, GGUF_KEY_TDNN_KERNEL_PREFIX, TDNN_AXIS_LEN)?,
            tdnn_dilation: req_u32_array(gguf, GGUF_KEY_TDNN_DILATION_PREFIX, TDNN_AXIS_LEN)?,
        })
    }
}

// ---------------------------------------------------------------------------
// WavLmSvWeights — bound the tensor manifest with a non-emptiness
// gate. Under the loud-partial WP the weights are counted but the
// 7-layer conv stem + Transformer encoder + XVector head forward is
// deferred (the three-block encode pipeline would consume them).
// Mirrors the `BeatThisWeights` / `Mt3Weights` / `ReDimNetWeights`
// posture.
// ---------------------------------------------------------------------------

/// Weight tensors bound from a WavLM-SV GGUF.
///
/// **Contract**: [`from_gguf`](Self::from_gguf) is a *loud*
/// verification step. A GGUF that carries zero tensors is rejected
/// with [`VokraError::ModelLoad`] (FR-EX-08 — an empty GGUF is never
/// a valid WavLM-SV checkpoint).
///
/// Under the current landing this struct stores the tensor names +
/// dims discovered on disk. The 7-layer conv stem + Transformer
/// encoder + XVector head forward is deferred (see
/// [`WavLmSv::encode`] loud-partial), so the payload is not yet
/// dequantised — the follow-up wave sizes the dequant per its kernel
/// needs.
#[derive(Debug)]
pub struct WavLmSvWeights {
    /// Tensors discovered on disk, indexed by upstream `state_dict`
    /// name with their GGUF-side dims. Used by the load-time
    /// non-emptiness gate and by the future follow-up encode forward
    /// wave.
    tensors: Vec<(String, Vec<usize>)>,
}

impl WavLmSvWeights {
    /// Scans `gguf` for the WavLM-SV state_dict tensors. Refuses to
    /// bind if the GGUF carries zero tensors (FR-EX-08 — an empty
    /// GGUF is never a valid WavLM-SV checkpoint).
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
                "wavlm_sv: GGUF carries zero tensors — refusing to bind an all-zero \
                 forward (FR-EX-08). Re-run `vokra-cli convert --model \
                 wavlm-base-plus-sv` against a safetensors checkpoint flattened via \
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
// WavLmSv — the runtime binder handle
// ---------------------------------------------------------------------------

/// WavLM Base+ SV speaker-verification encoder (Microsoft, cc-by-sa-3.0).
///
/// Bind with [`from_gguf`](Self::from_gguf), then call
/// [`encode`](Self::encode) on a raw waveform buffer to obtain a
/// 512-d speaker embedding. See the module doc for the current
/// implementation-status matrix and the FR-EX-08 loud-error contract
/// on the encode forward.
#[derive(Debug)]
pub struct WavLmSv {
    config: WavlmSvConfig,
    // The bound weights are held (real, counted) but the encode
    // forward (7-layer conv stem + Transformer encoder + XVector
    // head) is a follow-up wave; the field is deliberately
    // `#[allow(dead_code)]` until the kernel lands so a reader is
    // not misled by an unused field. Same posture as RMVPE / pyannote
    // / Charsiu / beat_this / mt3 / redimnet.
    #[allow(dead_code)]
    weights: WavLmSvWeights,
    weight_license: LicenseClass,
}

impl WavLmSv {
    /// Binds a WavLM-SV GGUF: validates arch, reads the strict
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
    ///   or not `"wavlm_sv"` (a `campplus` / `wespeaker` /
    ///   `ecapa_tdnn` / `titanet` / `speaker_3d` / `redimnet` GGUF
    ///   handed to us by mistake fails with a clear message instead
    ///   of a downstream "missing tensor" — same pattern as
    ///   `Mt3::from_gguf`).
    /// - [`VokraError::ModelLoad`] when any `vokra.wavlm.*` chunk is
    ///   absent ([`WavlmSvConfig::from_gguf`] is strict — no
    ///   primary-source constant fallback).
    /// - [`VokraError::ModelLoad`] when the GGUF carries zero tensors
    ///   ([`WavLmSvWeights::from_gguf`] refuses to bind an all-zero
    ///   forward).
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        // 1. Arch check — always first so a mis-typed model handed
        //    here fails with a specific message instead of a
        //    downstream "vokra.wavlm.hidden_size missing".
        match file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()) {
            Some(a) if a == ARCH => {}
            Some(other) => {
                return Err(VokraError::ModelLoad(format!(
                    "wavlm_sv: GGUF arch is `{other}`, expected `{ARCH}` (was this GGUF \
                     produced by `vokra-cli convert --model wavlm-base-plus-sv`? Note that \
                     sibling speaker-fleet arches — `campplus` (CAM++ D-TDNN backbone), \
                     `wespeaker` (ResNet-34 backbone), `ecapa_tdnn` (TDNN stack), `titanet` \
                     (depth-wise separable Conv1D backbone), `speaker_3d` (ERes2Net \
                     backbone), `redimnet` (2D dim-reduction + 1D conv+att + ASTP) — are \
                     all distinct topologies)"
                )));
            }
            None => {
                return Err(VokraError::ModelLoad(
                    "wavlm_sv: GGUF is missing `vokra.model.arch` (converter did not \
                     stamp it — this is not a Vokra-native wavlm_sv GGUF)"
                        .to_owned(),
                ));
            }
        }

        // 2. Strict topology axes from the `vokra.wavlm.*` chunk group.
        let config = WavlmSvConfig::from_gguf(file)?;

        // 3. Load the tensor manifest with the non-emptiness gate.
        let weights = WavLmSvWeights::from_gguf(file)?;

        // 4. Provenance surfacing — read the stamped weight-license
        //    class for compliance gate cross-checks (defaults to
        //    `Unknown` if absent, which is fail-closed at the gate).
        //    The WavLM-SV converter stamps `Copyleft` in production
        //    per the cc-by-sa-3.0 default. Not raising a `ModelLoad`
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

    /// The bound topology axes (read from the `vokra.wavlm.*` chunk
    /// group).
    #[inline]
    #[must_use]
    pub const fn config(&self) -> &WavlmSvConfig {
        &self.config
    }

    /// The stamped weight-license class surfaced from the GGUF's
    /// `vokra.provenance.weight_license` chunk. The WavLM-SV
    /// converter stamps `Copyleft` in production per the
    /// cc-by-sa-3.0 default; a GGUF missing the stamp reads back as
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

    /// Encodes a raw waveform buffer to a 512-d speaker embedding.
    ///
    /// # Loud-partial (this WP)
    ///
    /// Returns [`VokraError::UnsupportedOp`] — the WavLM Base+ SV
    /// **encode forward** requires transcribing three distinct
    /// pieces from the WavLM / UniSpeech Python source:
    ///
    /// 1. The **7-layer 1D conv feature-extractor stem** (HuBERT /
    ///    wav2vec2 lineage — `conv_dim` × `conv_stride` ×
    ///    `conv_kernel` per axis-array chunk group,
    ///    `feat_extract_norm = group` for Base+, ~50 Hz output
    ///    frame rate) — needs a walk against
    ///    `github.com/microsoft/UniSpeech/tree/main/WavLM/WavLM.py`
    ///    `ConvFeatureExtractionModel::forward`.
    /// 2. The **WavLM Transformer encoder** (12 layers, 768 hidden,
    ///    12 heads) with **gated relative position bias +
    ///    convolutional position-bias fusion** — this is the WavLM-
    ///    specific primitive that neither wav2vec2 nor HuBERT expose
    ///    and that no sibling in the tree supplies today. Needs a
    ///    walk against
    ///    `github.com/microsoft/UniSpeech/tree/main/WavLM/WavLM.py`
    ///    `TransformerSentenceEncoderLayer::forward` + the gated
    ///    relative-position-bias primitive.
    /// 3. The **XVector head with TDNN backbone + Additive Margin
    ///    Softmax** (5-block TDNN with kernels `[5,3,3,1,1]` +
    ///    dilations `[1,2,3,1,1]` → statistics pooling → 512-d
    ///    embedding through AM-Softmax loss layer) — needs a walk
    ///    against
    ///    `github.com/microsoft/UniSpeech/tree/main/downstreams/speaker_verification`
    ///    `XVectorHead` + `AMSoftmax` loss.
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
    pub fn encode(&self, pcm: &[f32]) -> Result<Vec<f32>> {
        // Bind unused arg so a `#[warn(unused_variables)]` change
        // does not silently mask the loud-partial fire path; the
        // future real implementation will consume it.
        let _ = pcm;
        Err(encode_forward_loud_partial(&self.config))
    }
}

/// Constructs the loud-partial [`VokraError::UnsupportedOp`]
/// returned by [`WavLmSv::encode`] until the WavLM Python source
/// transcription wave lands (7-layer conv stem + Transformer encoder
/// with gated relative position bias + XVector head with AM-Softmax).
///
/// Names **three** primary source anchors (HF card + UniSpeech
/// GitHub tree + arXiv paper) so a reader diagnosing the gap has
/// exactly three places to walk (RMVPE / pyannote / snac / hifigan /
/// beat_this / mt3 / redimnet Wave 1-4 loud-partial-message
/// precedent — CLAUDE.md 教訓 (a)).
fn encode_forward_loud_partial(cfg: &WavlmSvConfig) -> VokraError {
    VokraError::UnsupportedOp(format!(
        "wavlm_sv encode: WavLM Base+ SV encode forward pending — the WavLM Python source \
         transcription wave has not landed. Three pieces are missing: \
         (i) the 7-layer 1D conv feature-extractor stem (HuBERT/wav2vec2 lineage — \
         `conv_dim` x `conv_stride` x `conv_kernel` per axis-array chunk group, \
         `feat_extract_norm = group` for Base+), \
         (ii) the WavLM Transformer encoder (12 layers, 768 hidden, 12 heads) with \
         **gated relative position bias + convolutional position-bias fusion** — distinct \
         from vanilla Transformer AND from HuBERT/wav2vec2; WavLM introduces this fused \
         positional bias combo that no sibling in the tree supplies today, and \
         (iii) the XVector head with TDNN backbone + Additive Margin Softmax (5-block \
         TDNN with kernels [5,3,3,1,1] + dilations [1,2,3,1,1] -> statistics pooling -> \
         512-d embedding). Config: hidden_size={hs}, num_hidden_layers={nl}, \
         num_attention_heads={nh}, intermediate_size={is_}, num_feat_extract_layers={nfe}, \
         xvector_output_dim={xod}, num_ctc_classes={nc}, num_conv_pos_embeddings={ncp}, \
         num_conv_pos_embedding_groups={ncpg}, sample_rate={sr}, \
         layer_norm_eps_scaled_1e9={lne}, feat_extract_norm_group={fen}, \
         hidden_dropout_scaled_1e3={hd}, conv_dim={cd:?}, conv_stride={cs:?}, \
         conv_kernel={ck:?}, tdnn_dim={td:?}, tdnn_kernel={tk:?}, tdnn_dilation={tdi:?}. \
         Primary sources: {hf} + {uni} + {paper}. Loud pending (CLAUDE.md 教訓 (a) — \
         'loud-partial は fake-complete より honest') — no silent fabricated speaker \
         embedding ever emitted (FR-EX-08).",
        hs = cfg.hidden_size,
        nl = cfg.num_hidden_layers,
        nh = cfg.num_attention_heads,
        is_ = cfg.intermediate_size,
        nfe = cfg.num_feat_extract_layers,
        xod = cfg.xvector_output_dim,
        nc = cfg.num_ctc_classes,
        ncp = cfg.num_conv_pos_embeddings,
        ncpg = cfg.num_conv_pos_embedding_groups,
        sr = cfg.sample_rate,
        lne = cfg.layer_norm_eps_scaled_1e9,
        fen = cfg.feat_extract_norm_group,
        hd = cfg.hidden_dropout_scaled_1e3,
        cd = cfg.conv_dim,
        cs = cfg.conv_stride,
        ck = cfg.conv_kernel,
        td = cfg.tdnn_dim,
        tk = cfg.tdnn_kernel,
        tdi = cfg.tdnn_dilation,
        hf = PRIMARY_SOURCE_HF,
        uni = PRIMARY_SOURCE_UNISPEECH,
        paper = PRIMARY_SOURCE_PAPER,
    ))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    //! Tests for the WavLM-SV runtime binder — round-trip on the
    //! topology chunk group + negative-space round-trip on the
    //! loud-partial gates + arch-tag distinctness pin.
    //!
    //! # What "round-trip" means here
    //!
    //! The task spec asks for 5+ unit tests. On real PCM this would
    //! be `encode(...)` returning a real 512-d speaker embedding, but
    //! the WavLM-SV encode forward (7-layer conv stem + Transformer
    //! encoder + XVector head) has not been transcribed from the
    //! WavLM Python source (see the module doc + [`WavLmSv::encode`]
    //! rustdoc). Fabricating a real-PCM output would violate
    //! CLAUDE.md 教訓 (a) ("loud-partial は fake-complete より
    //! honest").
    //!
    //! The round-trip semantics we *can* honestly test:
    //!
    //! 1. **Config default pin**: the WavLM Base+ SV axes match the
    //!    upstream `config.json` transcription (`base_plus_default`).
    //! 2. **Config round-trip**: `from_gguf` reads every axis stamped
    //!    by the converter (full metadata + axis-array round-trip).
    //! 3. **Loud-error negative-space round-trip**: every stated
    //!    blocker (missing arch / wrong arch / missing chunk / empty
    //!    tensor list / unsupported forward surface) fires at its
    //!    documented surface point, in the documented error variant.
    //! 4. **Arch-tag distinctness pin**: the arch string is stable
    //!    and distinct from every sibling speaker-fleet arch.
    use super::*;
    use vokra_core::gguf::{GgmlType, GgufBuilder, GgufFile};

    /// Builds a minimal WavLM-SV GGUF carrying the arch tag + full
    /// `vokra.wavlm.*` chunk group + one representative tensor whose
    /// outer dim matches the given `xvector_output_dim`.
    /// `weight_license_class` is written under
    /// `vokra.provenance.weight_license` (or omitted if `None`).
    fn wavlm_sv_gguf(cfg: &WavlmSvConfig, weight_license_class: Option<LicenseClass>) -> GgufFile {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_string(chunks::KEY_MODEL_NAME, "wavlm-base-plus-sv");
        b.add_u32(GGUF_KEY_HIDDEN_SIZE, cfg.hidden_size);
        b.add_u32(GGUF_KEY_NUM_HIDDEN_LAYERS, cfg.num_hidden_layers);
        b.add_u32(GGUF_KEY_NUM_ATTENTION_HEADS, cfg.num_attention_heads);
        b.add_u32(GGUF_KEY_INTERMEDIATE_SIZE, cfg.intermediate_size);
        b.add_u32(
            GGUF_KEY_NUM_FEAT_EXTRACT_LAYERS,
            cfg.num_feat_extract_layers,
        );
        b.add_u32(GGUF_KEY_XVECTOR_OUTPUT_DIM, cfg.xvector_output_dim);
        b.add_u32(GGUF_KEY_NUM_CTC_CLASSES, cfg.num_ctc_classes);
        b.add_u32(
            GGUF_KEY_NUM_CONV_POS_EMBEDDINGS,
            cfg.num_conv_pos_embeddings,
        );
        b.add_u32(
            GGUF_KEY_NUM_CONV_POS_EMBEDDING_GROUPS,
            cfg.num_conv_pos_embedding_groups,
        );
        b.add_u32(GGUF_KEY_SAMPLE_RATE, cfg.sample_rate);
        b.add_u32(
            GGUF_KEY_LAYER_NORM_EPS_SCALED_1E9,
            cfg.layer_norm_eps_scaled_1e9,
        );
        b.add_u32(
            GGUF_KEY_FEAT_EXTRACT_NORM_GROUP,
            cfg.feat_extract_norm_group,
        );
        b.add_u32(
            GGUF_KEY_HIDDEN_DROPOUT_SCALED_1E3,
            cfg.hidden_dropout_scaled_1e3,
        );
        for (i, &v) in cfg.conv_dim.iter().enumerate() {
            b.add_u32(&format!("{GGUF_KEY_CONV_DIM_PREFIX}_{i}"), v);
        }
        for (i, &v) in cfg.conv_stride.iter().enumerate() {
            b.add_u32(&format!("{GGUF_KEY_CONV_STRIDE_PREFIX}_{i}"), v);
        }
        for (i, &v) in cfg.conv_kernel.iter().enumerate() {
            b.add_u32(&format!("{GGUF_KEY_CONV_KERNEL_PREFIX}_{i}"), v);
        }
        for (i, &v) in cfg.tdnn_dim.iter().enumerate() {
            b.add_u32(&format!("{GGUF_KEY_TDNN_DIM_PREFIX}_{i}"), v);
        }
        for (i, &v) in cfg.tdnn_kernel.iter().enumerate() {
            b.add_u32(&format!("{GGUF_KEY_TDNN_KERNEL_PREFIX}_{i}"), v);
        }
        for (i, &v) in cfg.tdnn_dilation.iter().enumerate() {
            b.add_u32(&format!("{GGUF_KEY_TDNN_DILATION_PREFIX}_{i}"), v);
        }
        if let Some(cls) = weight_license_class {
            b.add_string(chunks::KEY_PROVENANCE_WEIGHT_LICENSE, cls.as_str());
        }
        // One representative tensor so the non-emptiness gate passes.
        // The `xvector_output_dim` dim is deliberately at axis 0 so a
        // future shape-consistency check has something to walk.
        let d = u64::from(cfg.xvector_output_dim);
        b.add_tensor(
            "objective.projection.weight",
            GgmlType::F32,
            vec![d, d],
            vec![0u8; (d * d * 4) as usize],
        )
        .expect("add_tensor");
        GgufFile::parse(b.to_bytes().expect("serialize")).expect("parse")
    }

    // -----------------------------------------------------------------------
    // 1. WavlmSvConfig default matches config.json transcription
    // -----------------------------------------------------------------------

    #[test]
    fn config_default_matches_config_json_axes() {
        // Pin the Base+ SV hparams transcribed from
        // microsoft/wavlm-base-plus-sv/config.json (2026-08-14 scout WebFetch).
        // A rename or axis-value change would land here in the same
        // commit or fail this test.
        let cfg = WavlmSvConfig::base_plus_default();
        assert_eq!(cfg.hidden_size, 768);
        assert_eq!(cfg.num_hidden_layers, 12);
        assert_eq!(cfg.num_attention_heads, 12);
        assert_eq!(cfg.intermediate_size, 3072);
        assert_eq!(cfg.num_feat_extract_layers, 7);
        assert_eq!(cfg.xvector_output_dim, 512);
        assert_eq!(cfg.num_ctc_classes, 80);
        assert_eq!(cfg.num_conv_pos_embeddings, 128);
        assert_eq!(cfg.num_conv_pos_embedding_groups, 16);
        assert_eq!(cfg.sample_rate, 16000);
        assert_eq!(cfg.layer_norm_eps_scaled_1e9, 10_000);
        assert_eq!(cfg.feat_extract_norm_group, 1);
        assert_eq!(cfg.hidden_dropout_scaled_1e3, 100);
        assert_eq!(cfg.conv_dim, vec![512, 512, 512, 512, 512, 512, 512]);
        assert_eq!(cfg.conv_stride, vec![5, 2, 2, 2, 2, 2, 2]);
        assert_eq!(cfg.conv_kernel, vec![10, 3, 3, 3, 3, 2, 2]);
        assert_eq!(cfg.tdnn_dim, vec![512, 512, 512, 512, 1500]);
        assert_eq!(cfg.tdnn_kernel, vec![5, 3, 3, 1, 1]);
        assert_eq!(cfg.tdnn_dilation, vec![1, 2, 3, 1, 1]);
        // Structural invariant: the number of feat-extract layers
        // equals the length of each conv axis array.
        assert_eq!(cfg.conv_dim.len(), cfg.num_feat_extract_layers as usize);
        assert_eq!(cfg.conv_stride.len(), cfg.num_feat_extract_layers as usize);
        assert_eq!(cfg.conv_kernel.len(), cfg.num_feat_extract_layers as usize);
        // Structural invariant: TDNN axis arrays are all length 5.
        assert_eq!(cfg.tdnn_dim.len(), TDNN_AXIS_LEN);
        assert_eq!(cfg.tdnn_kernel.len(), TDNN_AXIS_LEN);
        assert_eq!(cfg.tdnn_dilation.len(), TDNN_AXIS_LEN);
    }

    // -----------------------------------------------------------------------
    // 2. from_gguf full topology chunk-group round-trip
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_metadata_round_trip() {
        let cfg = WavlmSvConfig::base_plus_default();
        let file = wavlm_sv_gguf(&cfg, Some(LicenseClass::Copyleft));
        let m = WavLmSv::from_gguf(&file).expect("valid GGUF must bind");
        // Config round-trip — every axis stamped by the converter
        // reads back into the same WavlmSvConfig value.
        assert_eq!(*m.config(), cfg);
        // Weight-license surface (WavLM-SV converter stamps Copyleft
        // per cc-by-sa-3.0 default).
        assert_eq!(m.weight_license(), LicenseClass::Copyleft);
        assert!(m.tensor_count() >= 1);
    }

    // -----------------------------------------------------------------------
    // 3. from_gguf rejects wrong arch (never silently mis-routes)
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_rejects_wrong_arch() {
        // A `redimnet` GGUF handed to the WavLM-SV binder by mistake
        // must fail loud with a specific message rather than
        // silently mis-binding (FR-EX-08).
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, "redimnet");
        b.add_u32(GGUF_KEY_HIDDEN_SIZE, 768);
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let Err(err) = WavLmSv::from_gguf(&file) else {
            panic!("expected ModelLoad on wrong arch");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains("`redimnet`") && m.contains("`wavlm_sv`"),
                    "message must name both the got and expected arch tags, got `{m}`"
                );
                assert!(
                    m.contains("2D dim-reduction"),
                    "message should disambiguate redimnet's topology to help \
                     the reader, got `{m}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // 4. encode returns UnsupportedOp with three-piece + primary-source
    //    + every-axis assertions
    // -----------------------------------------------------------------------

    #[test]
    fn encode_loud_partial_returns_unsupported_op() {
        let cfg = WavlmSvConfig::base_plus_default();
        let file = wavlm_sv_gguf(&cfg, Some(LicenseClass::Copyleft));
        let m = WavLmSv::from_gguf(&file).unwrap();
        // 1 second of legitimate-shape raw waveform (16000 samples)
        // so the loud-partial gate fires (not some pre-encode
        // validation).
        let pcm = vec![0.0f32; 16000];
        let Err(err) = m.encode(&pcm) else {
            panic!("encode must loud-partial");
        };
        match err {
            VokraError::UnsupportedOp(msg) => {
                assert!(
                    msg.contains("wavlm_sv encode"),
                    "message must call out the wavlm_sv encode surface, got `{msg}`"
                );
                // All three missing pieces must be named by exact
                // identifier so the follow-up wave knows what to walk.
                assert!(
                    msg.contains("conv feature-extractor stem"),
                    "message must name the 7-layer conv stem gap, got `{msg}`"
                );
                assert!(
                    msg.contains("Transformer encoder"),
                    "message must name the WavLM Transformer encoder gap, got `{msg}`"
                );
                assert!(
                    msg.contains("gated relative position bias"),
                    "message must name the WavLM-specific gated relative position bias \
                     primitive, got `{msg}`"
                );
                assert!(
                    msg.contains("convolutional position-bias fusion"),
                    "message must name the WavLM-specific convolutional position-bias \
                     fusion, got `{msg}`"
                );
                assert!(
                    msg.contains("XVector head") && msg.contains("Additive Margin Softmax"),
                    "message must name the XVector head + AM-Softmax gap, got `{msg}`"
                );
                // Primary-source URLs must be cited so a reader
                // diagnosing the gap has anchors to walk.
                assert!(
                    msg.contains("huggingface.co/microsoft/wavlm-base-plus-sv"),
                    "message must contain the HF card URL, got `{msg}`"
                );
                assert!(
                    msg.contains("github.com/microsoft/UniSpeech"),
                    "message must contain the UniSpeech GitHub URL, got `{msg}`"
                );
                assert!(
                    msg.contains("2110.13900"),
                    "message must cite the arXiv paper anchor, got `{msg}`"
                );
                // Every config axis must be echoed so the reader can
                // cross-check what topology the follow-up wave targets.
                assert!(
                    msg.contains("hidden_size=768"),
                    "hidden_size axis missing: {msg}"
                );
                assert!(
                    msg.contains("num_hidden_layers=12"),
                    "num_hidden_layers axis missing: {msg}"
                );
                assert!(
                    msg.contains("num_attention_heads=12"),
                    "num_attention_heads axis missing: {msg}"
                );
                assert!(
                    msg.contains("intermediate_size=3072"),
                    "intermediate_size axis missing: {msg}"
                );
                assert!(
                    msg.contains("num_feat_extract_layers=7"),
                    "num_feat_extract_layers axis missing: {msg}"
                );
                assert!(
                    msg.contains("xvector_output_dim=512"),
                    "xvector_output_dim axis missing: {msg}"
                );
                assert!(
                    msg.contains("num_ctc_classes=80"),
                    "num_ctc_classes axis missing: {msg}"
                );
                assert!(
                    msg.contains("num_conv_pos_embeddings=128"),
                    "num_conv_pos_embeddings axis missing: {msg}"
                );
                assert!(
                    msg.contains("num_conv_pos_embedding_groups=16"),
                    "num_conv_pos_embedding_groups axis missing: {msg}"
                );
                assert!(
                    msg.contains("sample_rate=16000"),
                    "sample_rate axis missing: {msg}"
                );
                // Array axes echoed as Debug repr — must contain at
                // least one representative value from each array.
                assert!(
                    msg.contains("conv_dim=[512"),
                    "conv_dim array axis missing: {msg}"
                );
                assert!(
                    msg.contains("conv_stride=[5"),
                    "conv_stride array axis missing: {msg}"
                );
                assert!(
                    msg.contains("conv_kernel=[10"),
                    "conv_kernel array axis missing: {msg}"
                );
                assert!(
                    msg.contains("tdnn_dim=[512"),
                    "tdnn_dim array axis missing: {msg}"
                );
                assert!(
                    msg.contains("tdnn_kernel=[5"),
                    "tdnn_kernel array axis missing: {msg}"
                );
                assert!(
                    msg.contains("tdnn_dilation=[1"),
                    "tdnn_dilation array axis missing: {msg}"
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
        assert_eq!(ARCH, "wavlm_sv");
        assert_ne!(
            ARCH, "campplus",
            "wavlm_sv (Transformer + XVector) and campplus (CAM++ D-TDNN) are \
             different topologies — sharing arch would mis-route runtime dispatch \
             (FR-EX-08)"
        );
        assert_ne!(
            ARCH, "wespeaker",
            "wavlm_sv and wespeaker (ResNet-34 backbone) are different topologies \
             — sharing arch would mis-route (FR-EX-08)"
        );
        assert_ne!(
            ARCH, "ecapa_tdnn",
            "wavlm_sv and ecapa_tdnn (TDNN stack backbone) are different topologies \
             — sharing arch would mis-route (FR-EX-08)"
        );
        assert_ne!(
            ARCH, "titanet",
            "wavlm_sv and titanet (depth-wise separable Conv1D backbone) are \
             different topologies — sharing arch would mis-route (FR-EX-08)"
        );
        assert_ne!(
            ARCH, "speaker_3d",
            "wavlm_sv and speaker_3d (ERes2Net backbone) are different topologies \
             — sharing arch would mis-route (FR-EX-08)"
        );
        assert_ne!(
            ARCH, "redimnet",
            "wavlm_sv (Transformer + XVector) and redimnet (2D dim-reduction + \
             1D conv+att + ASTP) are different topologies — sharing arch would \
             mis-route (FR-EX-08)"
        );
    }

    // -----------------------------------------------------------------------
    // 6. Missing topology chunk fails loud (no primary-source fallback)
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_rejects_missing_topology_chunk() {
        // Correct arch but missing one of the mandatory
        // `vokra.wavlm.*` chunks — a partially-stamped GGUF must
        // be caught here, not silently defaulted to a fabricated axis
        // (FR-EX-08 — the converter always stamps every axis, so a
        // missing chunk always signals a partial / mis-produced GGUF).
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_u32(GGUF_KEY_HIDDEN_SIZE, 768);
        b.add_u32(GGUF_KEY_NUM_HIDDEN_LAYERS, 12);
        b.add_u32(GGUF_KEY_NUM_ATTENTION_HEADS, 12);
        // deliberately omit intermediate_size
        b.add_u32(GGUF_KEY_NUM_FEAT_EXTRACT_LAYERS, 7);
        b.add_u32(GGUF_KEY_XVECTOR_OUTPUT_DIM, 512);
        b.add_u32(GGUF_KEY_NUM_CTC_CLASSES, 80);
        b.add_u32(GGUF_KEY_NUM_CONV_POS_EMBEDDINGS, 128);
        b.add_u32(GGUF_KEY_NUM_CONV_POS_EMBEDDING_GROUPS, 16);
        b.add_u32(GGUF_KEY_SAMPLE_RATE, 16000);
        b.add_u32(GGUF_KEY_LAYER_NORM_EPS_SCALED_1E9, 10_000);
        b.add_u32(GGUF_KEY_FEAT_EXTRACT_NORM_GROUP, 1);
        b.add_u32(GGUF_KEY_HIDDEN_DROPOUT_SCALED_1E3, 100);
        b.add_tensor(
            "objective.projection.weight",
            GgmlType::F32,
            vec![4, 4],
            vec![0u8; 16 * 4],
        )
        .expect("add_tensor");
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let Err(err) = WavLmSv::from_gguf(&file) else {
            panic!("expected ModelLoad on missing intermediate_size chunk");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains(GGUF_KEY_INTERMEDIATE_SIZE),
                    "message must name the missing intermediate_size key, got `{m}`"
                );
                assert!(
                    m.contains("config.json"),
                    "message should mention the upstream config.json transcription \
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
        // WavLmSvWeights non-emptiness gate must fire.
        let cfg = WavlmSvConfig::base_plus_default();
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_u32(GGUF_KEY_HIDDEN_SIZE, cfg.hidden_size);
        b.add_u32(GGUF_KEY_NUM_HIDDEN_LAYERS, cfg.num_hidden_layers);
        b.add_u32(GGUF_KEY_NUM_ATTENTION_HEADS, cfg.num_attention_heads);
        b.add_u32(GGUF_KEY_INTERMEDIATE_SIZE, cfg.intermediate_size);
        b.add_u32(
            GGUF_KEY_NUM_FEAT_EXTRACT_LAYERS,
            cfg.num_feat_extract_layers,
        );
        b.add_u32(GGUF_KEY_XVECTOR_OUTPUT_DIM, cfg.xvector_output_dim);
        b.add_u32(GGUF_KEY_NUM_CTC_CLASSES, cfg.num_ctc_classes);
        b.add_u32(
            GGUF_KEY_NUM_CONV_POS_EMBEDDINGS,
            cfg.num_conv_pos_embeddings,
        );
        b.add_u32(
            GGUF_KEY_NUM_CONV_POS_EMBEDDING_GROUPS,
            cfg.num_conv_pos_embedding_groups,
        );
        b.add_u32(GGUF_KEY_SAMPLE_RATE, cfg.sample_rate);
        b.add_u32(
            GGUF_KEY_LAYER_NORM_EPS_SCALED_1E9,
            cfg.layer_norm_eps_scaled_1e9,
        );
        b.add_u32(
            GGUF_KEY_FEAT_EXTRACT_NORM_GROUP,
            cfg.feat_extract_norm_group,
        );
        b.add_u32(
            GGUF_KEY_HIDDEN_DROPOUT_SCALED_1E3,
            cfg.hidden_dropout_scaled_1e3,
        );
        for (i, &v) in cfg.conv_dim.iter().enumerate() {
            b.add_u32(&format!("{GGUF_KEY_CONV_DIM_PREFIX}_{i}"), v);
        }
        for (i, &v) in cfg.conv_stride.iter().enumerate() {
            b.add_u32(&format!("{GGUF_KEY_CONV_STRIDE_PREFIX}_{i}"), v);
        }
        for (i, &v) in cfg.conv_kernel.iter().enumerate() {
            b.add_u32(&format!("{GGUF_KEY_CONV_KERNEL_PREFIX}_{i}"), v);
        }
        for (i, &v) in cfg.tdnn_dim.iter().enumerate() {
            b.add_u32(&format!("{GGUF_KEY_TDNN_DIM_PREFIX}_{i}"), v);
        }
        for (i, &v) in cfg.tdnn_kernel.iter().enumerate() {
            b.add_u32(&format!("{GGUF_KEY_TDNN_KERNEL_PREFIX}_{i}"), v);
        }
        for (i, &v) in cfg.tdnn_dilation.iter().enumerate() {
            b.add_u32(&format!("{GGUF_KEY_TDNN_DILATION_PREFIX}_{i}"), v);
        }
        // NO tensors added.
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let Err(err) = WavLmSv::from_gguf(&file) else {
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
