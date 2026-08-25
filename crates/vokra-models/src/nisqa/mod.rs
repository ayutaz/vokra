//! **NISQA v2** (`gabrielmittag/NISQA`) — runtime binder for the
//! `nisqa_v2_weight` converter arch (Wave A, 2026-08-15).
//!
//! Non-intrusive (reference-free) multidimensional speech-quality
//! predictor. Unlike DNSMOS (`crate::dnsmos_p808_p835`, a P.808 scalar +
//! a P.835 triple) NISQA v2 emits **five** scores in one forward: an
//! overall MOS plus the four degradation dimensions **Noisiness /
//! Discontinuity / Coloration / Loudness**. Collapsing that to a single
//! scalar would throw away the entire reason to run NISQA instead of
//! DNSMOS, so [`NisqaScore`] carries all five.
//!
//! # Gap this module closes
//!
//! `crates/vokra-convert/src/models/nisqa_v2_weight.rs` has been able to
//! emit a `nisqa_v2_weight` GGUF since the coverage-audit-2026-08-03
//! Wave D T4 landing, but **nothing in the workspace read the arch
//! string back** — converted weights were unloadable. This module is the
//! read side of that contract.
//!
//! # Primary sources
//!
//! - Reference code + weight release:
//!   <https://github.com/gabrielmittag/NISQA> (code **MIT**, weights
//!   **CC BY-NC-SA 4.0** — see "Licensing" below)
//! - Paper: Mittag, Naderi, Chehadi, Möller 2021, *"NISQA: A Deep
//!   CNN-Self-Attention Model for Multidimensional Speech Quality
//!   Prediction with Crowdsourced Datasets"*,
//!   <https://arxiv.org/abs/2104.09494>
//! - Standard-settings config (the hyper-parameter names this module's
//!   `vokra.nisqa.*` group mirrors):
//!   `config/train_nisqa_cnn_sa_ap.yaml`
//! - Model definition: `nisqa/NISQA_lib.py` (`class NISQA_DIM`,
//!   `class NISQA`, `class AdaptCNN`, `class SelfAttention`,
//!   `class PoolAttFF`, `def get_librosa_melspec`,
//!   `def segment_specs`)
//!
//! Every number quoted below was read out of one of those files — none
//! is estimated (CLAUDE.md「ハルシネーション厳禁」).
//!
//! # Licensing (T4 / research-only) — verified against the converter
//!
//! The upstream `README.md` states, verbatim: *"The NISQA code is
//! licensed under MIT License"* and *"The model weights (nisqa.tar,
//! nisqa_mos_only.tar, nisqa_tts.tar) are provided under a Creative
//! Commons Attribution-NonCommercial-ShareAlike 4.0 International
//! (CC BY-NC-SA 4.0) License"*. **Code MIT, weights CC-BY-NC-SA-4.0** —
//! which is exactly what the 2026-08-03 coverage audit (Wave D) recorded.
//!
//! Cross-checking the converter: `nisqa_v2_weight.rs` sets
//! `DEFAULT_LICENSE_SPDX = "cc-by-nc-sa-4.0"`, feeds it through
//! `LicenseClass::from_license_str`, and its own test asserts the stamp
//! resolves to [`LicenseClass::NonCommercialShareAlike`]. **The
//! converter's license handling is correct and matches the upstream
//! primary source** — no discrepancy found. Consequences the binder
//! surfaces rather than re-derives:
//!
//! - [`LicenseClass::requires_research_flag`] is `true` and
//!   [`LicenseClass::commercial_ok`] is `false` → **T4 / research-only
//!   tier**. It must never enter the official model zoo without
//!   `publish-one.sh --allow-noncommercial`.
//! - Share-alike cascades: a GGUF derived from these weights is itself
//!   CC-BY-NC-SA-4.0 and cannot be relabelled.
//! - The strict converter refuses a conflicting license override, so the
//!   canonical weights cannot be relabelled as permissive. The binder also
//!   requires the non-commercial-share-alike provenance class.
//! - The `docs/license-audit.md` §3.1 sign-off column stays **blank** —
//!   owner-only, CC does not sign (memory
//!   `[[feedback-license-signoff-primary-source]]`).
//!
//! # Runtime layout (transcribed from `nisqa/NISQA_lib.py`)
//!
//! ```text
//! PCM (mono f32)
//!   -> librosa-compatible mel-spectrogram            ← native host DSP
//!        `get_librosa_melspec`: power=1.0 (amplitude,
//!        NOT power), window='hann', center=True,
//!        pad_mode='reflect', fmin=0.0, htk=False,
//!        norm='slaney', then
//!        amplitude_to_db(ref=1.0, amin=1e-4, top_db=80.0).
//!        NOTE the upstream trap: `ms_hop_length` /
//!        `ms_win_length` are in **seconds** and are
//!        multiplied by `sr` inside the function —
//!        librosa's own arguments are in samples.
//!   -> `segment_specs`: slide a `ms_seg_length`-wide       ← native host glue
//!        window over the time axis (must be ODD; upstream
//!        raises `ValueError` otherwise), stride
//!        `ms_seg_hop_length`, pad to `ms_max_segments`.
//!        [H x W] -> [W-(seg_length-1) x 1 x H x seg_length]
//!   -> `AdaptCNN` framewise stage                    ← Compute GEMM + host glue
//!        6x (Conv2d -> BatchNorm2d -> ReLU) with
//!        `F.adaptive_max_pool2d` after conv1 / conv2 /
//!        conv4 to `cnn_pool_1` / `_2` / `_3`.
//!        adaptive max pooling uses exact PyTorch floor/ceil bins.
//!   -> `SelfAttention` time-dependency stage         ← Compute backend
//!        Linear(fan_out, d_model) -> LayerNorm ->
//!        `td_sa_num_layers` x SelfAttentionLayer
//!        (MultiheadAttention + FFN), positional
//!        encoding off in the standard config.
//!   -> `Pooling` = `PoolAttFF` attention-pooling      ← Compute backend
//!        att = Linear(h,1)(dropout(relu(Linear(d,h)(x))));
//!        masked softmax over valid windows; bmm; Linear.
//!   -> NISQA_DIM: **5 cloned pooling heads**, concatenated
//!      NISQA:     **1 pooling head** (overall MOS only)
//! ```
//!
//! # Head order is load-bearing
//!
//! `NISQA_lib.py` assigns the concatenated head vector like this:
//!
//! ```text
//! ds.df['mos_pred']  = y_hat[:,0]
//! ds.df['noi_pred']  = y_hat[:,1]
//! ds.df['dis_pred']  = y_hat[:,2]
//! ds.df['col_pred']  = y_hat[:,3]
//! ds.df['loud_pred'] = y_hat[:,4]
//! ```
//!
//! So the order is **mos, noi, dis, col, loud** — *discontinuity before
//! coloration*. The paper's prose lists the dimensions as "Noisiness,
//! Coloration, Discontinuity, Loudness", i.e. **col and dis swapped
//! relative to the tensor layout**. Reading the prose order into the
//! tensor layout silently swaps two scores that both look plausible, so
//! [`HEAD_ORDER`] pins the tensor order and a test asserts it.
//!
//! # Variant discrimination is real (from the tensor manifest)
//!
//! The converter writes upstream `state_dict` keys verbatim, and the two
//! upstream top-level classes differ in exactly one attribute:
//!
//! - `class NISQA_DIM` → `self.pool_layers = self._get_clones(pool, 5)`
//!   → keys under `pool_layers.0.` … `pool_layers.4.`
//! - `class NISQA` → `self.pool` → keys under `pool.`
//!
//! [`NisqaVariant`] is therefore derived from the GGUF tensor names, not
//! guessed — see [`NisqaConfig::from_gguf`].
//!
//! # `vokra.nisqa.*` chunk group (the flip-the-switch contract)
//!
//! The strict converter stamps the checkpoint-derived `vokra.nisqa.*` group.
//! The historical public GGUF predates that group; it is accepted only after
//! the exact complete 94-tensor manifest and generic provenance match, then
//! receives the same audited checkpoint values in memory.
//!
//! Which hyper-parameters actually have to be stamped is a real
//! question, and the answer is narrower than "all of them":
//!
//! - **Recoverable from tensor shapes** (do *not* need metadata):
//!   `cnn_c_out_1/2/3` (conv weight out-channels), `td_sa_d_model` /
//!   `td_sa_h` (Linear shapes), `pool_att_h` (PoolAttFF Linear shapes),
//!   `td_sa_num_layers` (count of `layers.N.` prefixes).
//! - **NOT recoverable** (must be stamped):
//!   - `cnn_pool_1/2/3` — adaptive-max-pool *output* sizes appear in no
//!     weight tensor at all ([`NisqaTopologySpec::cnn_pool`]);
//!   - `td_sa_nhead` — `MultiheadAttention` packs all heads into one
//!     `in_proj_weight`, so the head count is invisible in the shapes
//!     ([`NisqaTopologySpec::td_sa_nhead`]);
//!   - the whole mel front-end ([`NisqaFrontEndSpec`]) — pure
//!     hyper-parameters, no tensors involved.
//!
//! Keys, mirroring the upstream `ms_*` / `cnn_*` / `td_*` arg names:
//! [`KEY_NISQA_SAMPLE_RATE`], [`KEY_NISQA_N_FFT`],
//! [`KEY_NISQA_HOP_LENGTH_SEC`], [`KEY_NISQA_WIN_LENGTH_SEC`],
//! [`KEY_NISQA_N_MELS`], [`KEY_NISQA_FMAX`], [`KEY_NISQA_SEG_LENGTH`],
//! [`KEY_NISQA_SEG_HOP_LENGTH`], [`KEY_NISQA_MAX_SEGMENTS`],
//! [`KEY_NISQA_CNN_POOL_1_H`] … [`KEY_NISQA_CNN_POOL_3_W`],
//! [`KEY_NISQA_TD_SA_NHEAD`].
//!
//! Each group is **all-or-nothing**: a half-stamped group is a sidecar
//! bug and fails the load rather than silently defaulting (FR-EX-08).
//!
//! # Why this module does not implement `MosScorerEngine`
//!
//! `vokra_core::engines::MosScore` has exactly four slots (`p808`,
//! `sig`, `bak`, `ovrl`) shaped for DNSMOS. NISQA's coloration,
//! discontinuity and loudness have no slot, and mapping NISQA's overall
//! MOS onto `p808` would advertise a P.808 predictor this is not.
//! Implementing the trait would therefore silently drop three of five
//! dimensions — precisely the collapse this module exists to avoid. A
//! follow-up wave that wants a shared seam should widen `MosScore` (or
//! add a multidimensional sibling trait) in `vokra-core` first.
//!
//! # Cross-crate constant duplication
//!
//! [`ARCH`] / [`NAME`] / [`CATEGORY`] / [`UPSTREAM_URL`] /
//! [`DEFAULT_LICENSE_SPDX`] are **mirrors of the converter's `pub const`
//! surface** — the same duplication convention every sibling binder
//! (`dnsmos_p808_p835` / `emotion2vec` / `fsmn_vad` / `openwakeword`)
//! uses so `vokra-models` gains no dependency edge onto `vokra-convert`,
//! preserving the layered convention `vokra-ops → nothing GGUF-aware`,
//! `vokra-core → GGUF reader`, `vokra-models → GGUF binder`,
//! `vokra-convert → GGUF writer`.
//!
//! # No ONNX / no pickle (permanent)
//!
//! Upstream ships torch `.tar` pickles only. They are flattened to
//! safetensors offline by the pinned sidecar; neither pickle nor ONNX
//! ever enters the runtime (FR-LD-05 / NFR-DS-02).

mod nn;
#[cfg(test)]
mod tests;
mod weights;

use std::sync::Arc;

use vokra_core::backend::BackendKind;
use vokra_core::gguf::{GgufFile, GgufMetadataValue, chunks};
use vokra_core::{LicenseClass, Result, VokraError};

use crate::compute::{Compute, HotOp};
use crate::strict_checkpoint::{StrictCheckpoint, StrictCheckpointSpec};

pub use self::weights::NisqaWeights;

// ---------------------------------------------------------------------------
// Contract constants — mirror of
// `crates/vokra-convert/src/models/nisqa_v2_weight.rs`.
// See the module docstring for the cross-crate duplication rationale.
// ---------------------------------------------------------------------------

/// Expected `vokra.model.arch` value written by
/// `vokra-cli convert --model nisqa-v2-weight`.
///
/// Deliberately distinct from every sibling MOS-predictor arch tag —
/// `dnsmos` (Microsoft DNS-Challenge CNN, P.808 + P.835 heads), `utmos`
/// (SaruLab wav2vec2 regression), `utmosv2` (UTMOS v2), and
/// `torchaudio_squim` (Meta SQUIM, objective + subjective metrics).
/// They all sit in `category = "eval"` but have completely different
/// topologies and output widths; silently sharing an arch tag would
/// mis-route runtime dispatch (FR-EX-08).
pub const ARCH: &str = "nisqa_v2_weight";

/// Expected `vokra.model.name` value written for the canonical
/// `gabrielmittag/NISQA` release.
pub const NAME: &str = "nisqa_v2_weight";

/// Expected `vokra.model.category` value — `"eval"`, shared with the
/// sibling non-intrusive MOS predictors (`dnsmos`, `utmos`, `utmosv2`,
/// `torchaudio_squim`).
pub const CATEGORY: &str = "eval";

/// Primary redistribution source. NISQA is GitHub-only (there is no HF
/// mirror), so provenance rides `vokra.provenance.upstream_url` rather
/// than `vokra.provenance.upstream_hf` — the NKF-AEC / RNNoise / NSNet2
/// / DNSMOS precedent.
pub const UPSTREAM_URL: &str = "github.com/gabrielmittag/NISQA";

/// Default upstream weight licence (SPDX), mirrored from the converter.
/// Resolves to [`LicenseClass::NonCommercialShareAlike`] — **T4 /
/// research-only**, publish requires `--allow-noncommercial` and the
/// share-alike obligation cascades to derived artefacts.
pub const DEFAULT_LICENSE_SPDX: &str = "cc-by-nc-sa-4.0";

/// Learned tensors in the exact public multidimensional checkpoint.
pub const TENSOR_COUNT: usize = 94;

/// Backend-dispatched learned reductions required by the native forward.
pub const NISQA_HOT_OPS: &[HotOp] = &[HotOp::Gemm, HotOp::Softmax, HotOp::LayerNorm];

const SOURCE_REVISION: &str = "fe84f0f252abec382b24367d5b22498a7ce34dbb";
const SOURCE_MODEL_DEF_SHA256: &str =
    "f3ace1c00e21ae06e5d0fed9710f4e988c13685b2316a3b3ded46607fb25b71e";
const SOURCE_CONFIG_SHA256: &str =
    "afa752835c45f5d052787c024b10eab26eba980e0bde85632e674dbe557ec764";
const SOURCE_WEIGHT_LICENSE_SHA256: &str =
    "5b8e7938e1b5e0a675869ffe429cc8e7cc187d76a7c6ea1e0546c412782a43da";
const SOURCE_CHECKPOINT_SHA256: &str =
    "7ec4cf937514dd3f8860b21e66fabd8ca87a168572675ef8d979c4c4ad2e805c";
const PUBLIC_HF: &str = "vokra/nisqa-v2-weight";
const PUBLIC_REVISION: &str = "89718b026e17d3d048aa394ef8c8ddd14fee9cd8";
const PUBLIC_GGUF_SHA256: &str = "a2cacbe6f81ea2e8255eb0e2137d70d245823758e1cc4bb180c6b7cccc131e07";
const MANIFEST_SHA256: &str = "4845124c35587de7417acecac877e0f7bb131183d4aace79e47f361b7dc673f4";

const KEY_SOURCE_REVISION: &str = "vokra.nisqa.source_revision";
const KEY_SOURCE_MODEL_DEF_SHA256: &str = "vokra.nisqa.source_model_def_sha256";
const KEY_SOURCE_CONFIG_SHA256: &str = "vokra.nisqa.source_config_sha256";
const KEY_SOURCE_WEIGHT_LICENSE_SHA256: &str = "vokra.nisqa.source_weight_license_sha256";
const KEY_SOURCE_CHECKPOINT_SHA256: &str = "vokra.nisqa.source_checkpoint_sha256";
const KEY_PUBLIC_HF: &str = "vokra.nisqa.public_hf";
const KEY_PUBLIC_REVISION: &str = "vokra.nisqa.public_revision";
const KEY_PUBLIC_GGUF_SHA256: &str = "vokra.nisqa.public_gguf_sha256";
const KEY_MANIFEST_SHA256: &str = "vokra.nisqa.manifest_sha256";

pub(super) const SPEC: StrictCheckpointSpec = StrictCheckpointSpec {
    label: "nisqa",
    arch: ARCH,
    model_name: NAME,
    model_name_alias: None,
    tensor_count: TENSOR_COUNT,
    manifest_sha256: [
        0x48, 0x45, 0x12, 0x4c, 0x35, 0x58, 0x7d, 0xe7, 0x41, 0x7a, 0xce, 0xca, 0xc8, 0x77, 0xe0,
        0xf7, 0xbb, 0x13, 0x11, 0x83, 0xd4, 0xaa, 0xce, 0x79, 0xe4, 0x7f, 0x36, 0x1b, 0x7d, 0xc6,
        0x73, 0xf4,
    ],
};

/// GGUF metadata key: model category tag (mirror of the converter's
/// local const).
pub const KEY_MODEL_CATEGORY: &str = "vokra.model.category";

/// GGUF metadata key: primary redistribution source URL for GitHub-only
/// releases (mirror of the converter's local const).
pub const KEY_PROVENANCE_UPSTREAM_URL: &str = "vokra.provenance.upstream_url";

// ---------------------------------------------------------------------------
// Primary-source anchors used by the audited converter/parity chain.
// ---------------------------------------------------------------------------

/// Primary-source anchor: reference implementation + weight release.
pub const PRIMARY_SOURCE_CODE: &str = "github.com/gabrielmittag/NISQA";

/// Primary-source anchor: the paper (Mittag et al. 2021).
pub const PRIMARY_SOURCE_PAPER: &str = "arxiv.org/abs/2104.09494";

/// Primary-source anchor: the model definition file whose classes this
/// binder transcribes (`NISQA_DIM` / `AdaptCNN` / `SelfAttention` /
/// `PoolAttFF` / `get_librosa_melspec` / `segment_specs`).
pub const PRIMARY_SOURCE_MODEL_DEF: &str = "gabrielmittag/NISQA/nisqa/NISQA_lib.py";

/// Primary-source anchor: the standard-settings config whose `ms_*` /
/// `cnn_*` / `td_*` keys the [`KEY_NISQA_SAMPLE_RATE`] group mirrors.
pub const PRIMARY_SOURCE_CONFIG: &str = "gabrielmittag/NISQA/config/train_nisqa_cnn_sa_ap.yaml";

/// Pinned offline sidecar that flattens the upstream `.tar` pickle to
/// safetensors after source/checkpoint validation.
pub const SIDECAR_PATH: &str = "tools/parity/nisqa_v2_weight_prepare_checkpoint.py";

// ---------------------------------------------------------------------------
// Fixed mel front-end constants — hard-coded in upstream
// `get_librosa_melspec` (NOT driven by the checkpoint's `args` dict, so
// unlike the `ms_*` group these are knowable without metadata).
// ---------------------------------------------------------------------------

/// `power=1.0` in the upstream `librosa.feature.melspectrogram` call —
/// an **amplitude** mel-spectrogram, not the usual power spectrogram.
pub const MEL_POWER: f32 = 1.0;

/// `fmin=0.0` in the upstream mel filterbank construction.
pub const MEL_FMIN: f32 = 0.0;

/// `ref=1.0` in the upstream `librosa.core.amplitude_to_db` call.
pub const MEL_DB_REF: f32 = 1.0;

/// `amin=1e-4` in the upstream `librosa.core.amplitude_to_db` call.
pub const MEL_DB_AMIN: f32 = 1e-4;

/// `top_db=80.0` in the upstream `librosa.core.amplitude_to_db` call.
pub const MEL_DB_TOP_DB: f32 = 80.0;

/// `htk=False` upstream → Slaney-style mel scale with `norm='slaney'`.
pub const MEL_HTK: bool = false;

// ---------------------------------------------------------------------------
// Head-order contract.
// ---------------------------------------------------------------------------

/// Number of pooling heads on the multidimensional variant
/// (`NISQA_DIM._get_clones(pool, 5)`).
pub const N_HEADS: usize = 5;

/// Concatenated head order, verbatim from the `y_hat[:, i]` assignments
/// in `NISQA_lib.py`: index 0 `mos`, 1 `noi`, 2 `dis`, 3 `col`,
/// 4 `loud`.
///
/// **Load-bearing.** The paper's prose lists the dimensions as
/// "Noisiness, Coloration, Discontinuity, Loudness" — `col` and `dis`
/// are swapped relative to this tensor layout. A silent reorder would
/// exchange two plausible-looking scores with no crash, so this array is
/// pinned by a test.
pub const HEAD_ORDER: [&str; N_HEADS] = ["mos", "noi", "dis", "col", "loud"];

// ---------------------------------------------------------------------------
// Tensor-name prefixes (upstream `state_dict` keys, written verbatim by
// the converter).
// ---------------------------------------------------------------------------

/// Prefix of the framewise CNN stage's parameters (`NISQA_DIM.cnn`,
/// a `Framewise` wrapper whose inner module is `.model`).
pub const TENSOR_PREFIX_CNN: &str = "cnn.";

/// Prefix of the first time-dependency stage's parameters
/// (`NISQA_DIM.time_dependency`).
pub const TENSOR_PREFIX_TIME_DEPENDENCY: &str = "time_dependency.";

/// Prefix of the multidimensional variant's cloned pooling heads
/// (`NISQA_DIM.pool_layers`, a 5-element `ModuleList`).
pub const TENSOR_PREFIX_POOL_LAYERS: &str = "pool_layers.";

/// Prefix of the single-output variant's pooling head (`NISQA.pool`).
pub const TENSOR_PREFIX_POOL: &str = "pool.";

// ---------------------------------------------------------------------------
// `vokra.nisqa.*` metadata keys — mel front-end group (mirrors the
// upstream `ms_*` args).
// ---------------------------------------------------------------------------

/// `ms_sr` — resample target in Hz. **`0` is the sentinel for upstream's
/// `ms_sr: null`** ("keep the file's native rate"); the upstream comment
/// notes the window length is adjusted automatically for different
/// sample frequencies.
pub const KEY_NISQA_SAMPLE_RATE: &str = "vokra.nisqa.sample_rate";
/// `ms_n_fft` — padded FFT window length in bins.
pub const KEY_NISQA_N_FFT: &str = "vokra.nisqa.n_fft";
/// `ms_hop_length` — hop length in **seconds** (multiplied by `sr`
/// inside `get_librosa_melspec`, unlike librosa's own sample-valued
/// argument).
pub const KEY_NISQA_HOP_LENGTH_SEC: &str = "vokra.nisqa.hop_length_sec";
/// `ms_win_length` — FFT window length in **seconds**, zero-padded up to
/// `ms_n_fft`.
pub const KEY_NISQA_WIN_LENGTH_SEC: &str = "vokra.nisqa.win_length_sec";
/// `ms_n_mels` — number of mel bands.
pub const KEY_NISQA_N_MELS: &str = "vokra.nisqa.n_mels";
/// `ms_fmax` — maximum considered mel-band frequency in Hz.
pub const KEY_NISQA_FMAX: &str = "vokra.nisqa.fmax";
/// `ms_seg_length` — width of an extracted mel-spec segment in bins.
/// Upstream `segment_specs` raises `ValueError` unless this is **odd**.
pub const KEY_NISQA_SEG_LENGTH: &str = "vokra.nisqa.seg_length";
/// `ms_seg_hop_length` — hop length between segments in bins.
pub const KEY_NISQA_SEG_HOP_LENGTH: &str = "vokra.nisqa.seg_hop_length";
/// `ms_max_segments` — padded maximum segment count.
pub const KEY_NISQA_MAX_SEGMENTS: &str = "vokra.nisqa.max_segments";

/// The mel front-end group in canonical order — all-or-nothing.
const FRONT_END_KEYS: [&str; 9] = [
    KEY_NISQA_SAMPLE_RATE,
    KEY_NISQA_N_FFT,
    KEY_NISQA_HOP_LENGTH_SEC,
    KEY_NISQA_WIN_LENGTH_SEC,
    KEY_NISQA_N_MELS,
    KEY_NISQA_FMAX,
    KEY_NISQA_SEG_LENGTH,
    KEY_NISQA_SEG_HOP_LENGTH,
    KEY_NISQA_MAX_SEGMENTS,
];

// ---------------------------------------------------------------------------
// `vokra.nisqa.*` metadata keys — topology group. Deliberately minimal:
// only the values that are NOT recoverable from the tensor shapes (see
// the module docstring's "flip-the-switch contract" section).
// ---------------------------------------------------------------------------

/// `cnn_pool_1[0]` — height of the first adaptive-max-pool output.
pub const KEY_NISQA_CNN_POOL_1_H: &str = "vokra.nisqa.cnn_pool_1_h";
/// `cnn_pool_1[1]` — width of the first adaptive-max-pool output.
pub const KEY_NISQA_CNN_POOL_1_W: &str = "vokra.nisqa.cnn_pool_1_w";
/// `cnn_pool_2[0]` — height of the second adaptive-max-pool output.
pub const KEY_NISQA_CNN_POOL_2_H: &str = "vokra.nisqa.cnn_pool_2_h";
/// `cnn_pool_2[1]` — width of the second adaptive-max-pool output.
pub const KEY_NISQA_CNN_POOL_2_W: &str = "vokra.nisqa.cnn_pool_2_w";
/// `cnn_pool_3[0]` — height of the third adaptive-max-pool output.
/// Also the row count the CNN output is flattened over
/// (`x.view(-1, conv6.out_channels * pool_3[0])`).
pub const KEY_NISQA_CNN_POOL_3_H: &str = "vokra.nisqa.cnn_pool_3_h";
/// `cnn_pool_3[1]` — width of the third adaptive-max-pool output. Also
/// the kernel width of the last conv layer
/// (`kernel_size_last = (kernel_size[0], pool_3[1])`).
pub const KEY_NISQA_CNN_POOL_3_W: &str = "vokra.nisqa.cnn_pool_3_w";
/// `td_sa_nhead` — self-attention head count. Invisible in the weight
/// shapes because `nn.MultiheadAttention` packs every head into one
/// `in_proj_weight`.
pub const KEY_NISQA_TD_SA_NHEAD: &str = "vokra.nisqa.td_sa_nhead";

/// The topology group in canonical order — all-or-nothing.
const TOPOLOGY_KEYS: [&str; 7] = [
    KEY_NISQA_CNN_POOL_1_H,
    KEY_NISQA_CNN_POOL_1_W,
    KEY_NISQA_CNN_POOL_2_H,
    KEY_NISQA_CNN_POOL_2_W,
    KEY_NISQA_CNN_POOL_3_H,
    KEY_NISQA_CNN_POOL_3_W,
    KEY_NISQA_TD_SA_NHEAD,
];

// ---------------------------------------------------------------------------
// NisqaScore — the five-dimension result.
// ---------------------------------------------------------------------------

/// The five NISQA v2 predictions from a single forward.
///
/// Field order matches [`HEAD_ORDER`] — the tensor layout, **not** the
/// paper's prose order (which swaps coloration and discontinuity).
///
/// All five are plain `f32`, never `Option`: the multidimensional
/// checkpoint always emits all five, and the single-output checkpoint is
/// refused by [`Nisqa::score`] with a loud error rather than being
/// padded with fabricated sub-scores (FR-EX-08).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NisqaScore {
    /// Overall quality MOS (head index 0, upstream `mos_pred`).
    pub mos: f32,
    /// Noisiness dimension (head index 1, upstream `noi_pred`).
    pub noisiness: f32,
    /// Discontinuity dimension (head index 2, upstream `dis_pred`).
    ///
    /// Note the index: discontinuity comes **before** coloration in the
    /// tensor layout even though the paper's prose lists coloration
    /// first.
    pub discontinuity: f32,
    /// Coloration dimension (head index 3, upstream `col_pred`).
    pub coloration: f32,
    /// Loudness dimension (head index 4, upstream `loud_pred`).
    pub loudness: f32,
}

impl NisqaScore {
    /// Builds a score from the concatenated pooling-head output in the
    /// upstream tensor order (see [`HEAD_ORDER`]).
    ///
    /// This is the only place the head-index → field mapping is written,
    /// so the native forward cannot re-derive (and re-misorder) it.
    ///
    /// # Errors
    ///
    /// - [`VokraError::InvalidArgument`] when `heads.len() != 5` — a
    ///   short or long head vector means the bound checkpoint is not the
    ///   multidimensional variant, and padding it would fabricate
    ///   sub-scores (FR-EX-08).
    pub fn from_heads(heads: &[f32]) -> Result<Self> {
        if heads.len() != N_HEADS {
            return Err(VokraError::InvalidArgument(format!(
                "nisqa: expected {N_HEADS} pooling-head outputs in the order \
                 {HEAD_ORDER:?} (upstream `NISQA_DIM._get_clones(pool, 5)`), got \
                 {}. A head vector of the wrong width means this is not the \
                 multidimensional checkpoint — padding it would fabricate \
                 sub-scores (FR-EX-08).",
                heads.len()
            )));
        }
        Ok(Self {
            mos: heads[0],
            noisiness: heads[1],
            discontinuity: heads[2],
            coloration: heads[3],
            loudness: heads[4],
        })
    }

    /// Reconstructs the concatenated head vector in [`HEAD_ORDER`].
    /// Round-trips with [`Self::from_heads`].
    #[must_use]
    pub const fn to_heads(self) -> [f32; N_HEADS] {
        [
            self.mos,
            self.noisiness,
            self.discontinuity,
            self.coloration,
            self.loudness,
        ]
    }

    /// Looks a dimension up by its upstream short name (`"mos"`,
    /// `"noi"`, `"dis"`, `"col"`, `"loud"`), returning `None` for an
    /// unknown name rather than a default (FR-EX-08).
    #[must_use]
    pub fn get(self, head_name: &str) -> Option<f32> {
        match head_name {
            "mos" => Some(self.mos),
            "noi" => Some(self.noisiness),
            "dis" => Some(self.discontinuity),
            "col" => Some(self.coloration),
            "loud" => Some(self.loudness),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// NisqaVariant — derived from the tensor manifest.
// ---------------------------------------------------------------------------

/// Which upstream top-level class the bound checkpoint implements.
///
/// Derived from the GGUF tensor names, not from metadata: the two
/// classes differ in exactly one attribute (`pool_layers` vs `pool`), so
/// the discriminator is primary-source-derived and cannot drift.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NisqaVariant {
    /// `class NISQA_DIM` — five cloned pooling heads, all five
    /// dimensions. This is upstream's `weights/nisqa.tar`.
    MultiDim,
    /// `class NISQA` — one pooling head, overall MOS (or, for the TTS
    /// checkpoint, naturalness) only. This is upstream's
    /// `weights/nisqa_mos_only.tar` / `weights/nisqa_tts.tar`.
    SingleOutput,
}

impl NisqaVariant {
    /// Canonical short name for logs and diagnostics.
    #[must_use]
    pub const fn short(self) -> &'static str {
        match self {
            Self::MultiDim => "multidim",
            Self::SingleOutput => "single-output",
        }
    }

    /// The upstream class name this variant corresponds to.
    #[must_use]
    pub const fn upstream_class(self) -> &'static str {
        match self {
            Self::MultiDim => "NISQA_DIM",
            Self::SingleOutput => "NISQA",
        }
    }

    /// How many pooling heads this variant carries (5 vs 1).
    #[must_use]
    pub const fn n_heads(self) -> usize {
        match self {
            Self::MultiDim => N_HEADS,
            Self::SingleOutput => 1,
        }
    }
}

// ---------------------------------------------------------------------------
// NisqaFrontEndSpec — optional `vokra.nisqa.*` mel front-end group.
// ---------------------------------------------------------------------------

/// The mel front-end hyper-parameters, read from the optional
/// `vokra.nisqa.*` group (upstream `ms_*` args).
///
/// Stamped by the strict converter; reconstructed only for the exact
/// historical public manifest that predates the group.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NisqaFrontEndSpec {
    /// Resample target in Hz; `0` means "keep the native rate"
    /// (upstream `ms_sr: null`).
    pub sample_rate: u32,
    /// Padded FFT window length in bins (`ms_n_fft`).
    pub n_fft: u32,
    /// Hop length in **seconds** (`ms_hop_length`).
    pub hop_length_sec: f32,
    /// FFT window length in **seconds** (`ms_win_length`).
    pub win_length_sec: f32,
    /// Number of mel bands (`ms_n_mels`).
    pub n_mels: u32,
    /// Maximum considered mel-band frequency in Hz (`ms_fmax`).
    pub fmax: f32,
    /// Segment width in bins (`ms_seg_length`) — must be odd.
    pub seg_length: u32,
    /// Segment hop in bins (`ms_seg_hop_length`).
    pub seg_hop_length: u32,
    /// Padded maximum segment count (`ms_max_segments`).
    pub max_segments: u32,
}

impl NisqaFrontEndSpec {
    /// Validates the front-end spec loudly (FR-EX-08).
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] when any field is out of range, in
    ///   particular an **even** `seg_length` (upstream `segment_specs`
    ///   raises `ValueError: seg_length must be odd!`).
    pub fn validate(&self) -> Result<()> {
        if self.n_fft == 0 {
            return Err(VokraError::ModelLoad(format!(
                "nisqa: `{KEY_NISQA_N_FFT}` = 0 — the padded FFT window length must \
                 be positive (upstream standard config uses 4096)"
            )));
        }
        if self.n_mels == 0 {
            return Err(VokraError::ModelLoad(format!(
                "nisqa: `{KEY_NISQA_N_MELS}` = 0 — the mel-band count must be positive"
            )));
        }
        if self.seg_length == 0 {
            return Err(VokraError::ModelLoad(format!(
                "nisqa: `{KEY_NISQA_SEG_LENGTH}` = 0 — the segment width must be positive"
            )));
        }
        if self.seg_length % 2 == 0 {
            return Err(VokraError::ModelLoad(format!(
                "nisqa: `{KEY_NISQA_SEG_LENGTH}` = {} is even — upstream \
                 `segment_specs` in `{PRIMARY_SOURCE_MODEL_DEF}` raises \
                 `ValueError('seg_length must be odd!')` because the segment is \
                 centred on the current frame with (seg_length-1)/2 neighbours on \
                 each side. An even width has no centre frame, so accepting it \
                 would silently shift every segment by half a frame (FR-EX-08).",
                self.seg_length
            )));
        }
        if self.seg_hop_length == 0 {
            return Err(VokraError::ModelLoad(format!(
                "nisqa: `{KEY_NISQA_SEG_HOP_LENGTH}` = 0 — a zero segment hop would \
                 never advance the window"
            )));
        }
        if self.max_segments == 0 {
            return Err(VokraError::ModelLoad(format!(
                "nisqa: `{KEY_NISQA_MAX_SEGMENTS}` = 0 — the padded segment count \
                 must be positive"
            )));
        }
        // `x <= 0.0` is false for NaN, so the `!is_finite()` arm is what
        // rejects NaN and both infinities. Written in this order (rather
        // than as one negated conjunction) so the condition is already in
        // its minimal boolean form.
        if self.hop_length_sec <= 0.0 || !self.hop_length_sec.is_finite() {
            return Err(VokraError::ModelLoad(format!(
                "nisqa: `{KEY_NISQA_HOP_LENGTH_SEC}` = {} — the hop length is in \
                 SECONDS upstream (multiplied by `sr` inside `get_librosa_melspec`) \
                 and must be finite and positive",
                self.hop_length_sec
            )));
        }
        if self.win_length_sec <= 0.0 || !self.win_length_sec.is_finite() {
            return Err(VokraError::ModelLoad(format!(
                "nisqa: `{KEY_NISQA_WIN_LENGTH_SEC}` = {} — the window length is in \
                 SECONDS upstream and must be finite and positive",
                self.win_length_sec
            )));
        }
        if self.fmax <= 0.0 || !self.fmax.is_finite() {
            return Err(VokraError::ModelLoad(format!(
                "nisqa: `{KEY_NISQA_FMAX}` = {} — the maximum mel-band frequency \
                 must be finite and positive",
                self.fmax
            )));
        }
        Ok(())
    }

    /// Reads the group from a GGUF. Returns `Ok(None)` when **no** key of
    /// the group is present (the state of every GGUF the current
    /// converter produces); returns a loud [`VokraError::ModelLoad`] when
    /// the group is only partially stamped.
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] on a partially stamped group, a wrong
    ///   value type, or a failed [`Self::validate`].
    pub fn from_gguf(gguf: &GgufFile) -> Result<Option<Self>> {
        if !group_present(gguf, &FRONT_END_KEYS) {
            return Ok(None);
        }
        let spec = Self {
            sample_rate: read_u32_key(gguf, KEY_NISQA_SAMPLE_RATE)?,
            n_fft: read_u32_key(gguf, KEY_NISQA_N_FFT)?,
            hop_length_sec: read_f32_key(gguf, KEY_NISQA_HOP_LENGTH_SEC)?,
            win_length_sec: read_f32_key(gguf, KEY_NISQA_WIN_LENGTH_SEC)?,
            n_mels: read_u32_key(gguf, KEY_NISQA_N_MELS)?,
            fmax: read_f32_key(gguf, KEY_NISQA_FMAX)?,
            seg_length: read_u32_key(gguf, KEY_NISQA_SEG_LENGTH)?,
            seg_hop_length: read_u32_key(gguf, KEY_NISQA_SEG_HOP_LENGTH)?,
            max_segments: read_u32_key(gguf, KEY_NISQA_MAX_SEGMENTS)?,
        };
        spec.validate()?;
        Ok(Some(spec))
    }
}

// ---------------------------------------------------------------------------
// NisqaTopologySpec — optional `vokra.nisqa.*` topology group.
// ---------------------------------------------------------------------------

/// The topology hyper-parameters that are **not** recoverable from the
/// weight tensor shapes, read from the optional `vokra.nisqa.*` group.
///
/// Everything else (`cnn_c_out_*`, `td_sa_d_model`, `td_sa_h`,
/// `pool_att_h`, `td_sa_num_layers`) is derivable from the tensor
/// manifest, so it deliberately has no key here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NisqaTopologySpec {
    /// `cnn_pool_1` / `_2` / `_3` as `[height, width]` pairs — the three
    /// `F.adaptive_max_pool2d` output sizes. These appear in no weight
    /// tensor.
    pub cnn_pool: [[u32; 2]; 3],
    /// `td_sa_nhead` — invisible in the weight shapes because
    /// `nn.MultiheadAttention` packs all heads into one `in_proj_weight`.
    pub td_sa_nhead: u32,
}

impl NisqaTopologySpec {
    /// Validates the topology spec loudly (FR-EX-08).
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] when any adaptive-pool extent or the
    ///   head count is zero.
    pub fn validate(&self) -> Result<()> {
        let key_names: [[&str; 2]; 3] = [
            [KEY_NISQA_CNN_POOL_1_H, KEY_NISQA_CNN_POOL_1_W],
            [KEY_NISQA_CNN_POOL_2_H, KEY_NISQA_CNN_POOL_2_W],
            [KEY_NISQA_CNN_POOL_3_H, KEY_NISQA_CNN_POOL_3_W],
        ];
        for (pair, keys) in self.cnn_pool.iter().zip(key_names.iter()) {
            for (extent, key) in pair.iter().zip(keys.iter()) {
                if *extent == 0 {
                    return Err(VokraError::ModelLoad(format!(
                        "nisqa: `{key}` = 0 — an adaptive-max-pool output extent must \
                         be positive (`F.adaptive_max_pool2d` cannot produce a \
                         zero-sized axis)"
                    )));
                }
            }
        }
        if self.td_sa_nhead == 0 {
            return Err(VokraError::ModelLoad(format!(
                "nisqa: `{KEY_NISQA_TD_SA_NHEAD}` = 0 — the self-attention head \
                 count must be positive"
            )));
        }
        Ok(())
    }

    /// Reads the group from a GGUF. Returns `Ok(None)` when no key of the
    /// group is present; loud on a partially stamped group.
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] on a partially stamped group, a wrong
    ///   value type, or a failed [`Self::validate`].
    pub fn from_gguf(gguf: &GgufFile) -> Result<Option<Self>> {
        if !group_present(gguf, &TOPOLOGY_KEYS) {
            return Ok(None);
        }
        let spec = Self {
            cnn_pool: [
                [
                    read_u32_key(gguf, KEY_NISQA_CNN_POOL_1_H)?,
                    read_u32_key(gguf, KEY_NISQA_CNN_POOL_1_W)?,
                ],
                [
                    read_u32_key(gguf, KEY_NISQA_CNN_POOL_2_H)?,
                    read_u32_key(gguf, KEY_NISQA_CNN_POOL_2_W)?,
                ],
                [
                    read_u32_key(gguf, KEY_NISQA_CNN_POOL_3_H)?,
                    read_u32_key(gguf, KEY_NISQA_CNN_POOL_3_W)?,
                ],
            ],
            td_sa_nhead: read_u32_key(gguf, KEY_NISQA_TD_SA_NHEAD)?,
        };
        spec.validate()?;
        Ok(Some(spec))
    }
}

/// Exact front-end arguments embedded in upstream `weights/nisqa.tar`.
pub const CANONICAL_FRONT_END: NisqaFrontEndSpec = NisqaFrontEndSpec {
    sample_rate: 0,
    n_fft: 4096,
    hop_length_sec: 0.01,
    win_length_sec: 0.02,
    n_mels: 48,
    fmax: 20_000.0,
    seg_length: 15,
    seg_hop_length: 4,
    max_segments: 1300,
};

/// Exact non-shape topology arguments embedded in upstream `weights/nisqa.tar`.
pub const CANONICAL_TOPOLOGY: NisqaTopologySpec = NisqaTopologySpec {
    cnn_pool: [[24, 7], [12, 5], [6, 3]],
    td_sa_nhead: 1,
};

// ---------------------------------------------------------------------------
// Metadata read helpers.
// ---------------------------------------------------------------------------

/// `true` when **any** key of an all-or-nothing group is present.
fn group_present(gguf: &GgufFile, keys: &[&str]) -> bool {
    keys.iter().any(|k| gguf.get(k).is_some())
}

/// Reads a required unsigned-integer key, refusing a wrong value type
/// rather than coercing (FR-EX-08).
fn read_u32_key(gguf: &GgufFile, key: &str) -> Result<u32> {
    let raw = gguf.get(key).and_then(|v| v.as_u64()).ok_or_else(|| {
        VokraError::ModelLoad(format!(
            "nisqa: GGUF metadata `{key}` is missing or is not an unsigned integer. \
             The `vokra.nisqa.*` groups are all-or-nothing — a partially stamped \
             group is a bug in `{SIDECAR_PATH}`, and silently defaulting the \
             missing half would produce a wrong-shaped front-end with no crash \
             (FR-EX-08)."
        ))
    })?;
    u32::try_from(raw).map_err(|_| {
        VokraError::ModelLoad(format!(
            "nisqa: GGUF metadata `{key}` = {raw} does not fit in u32"
        ))
    })
}

/// Reads a required float key, refusing a wrong value type rather than
/// coercing (FR-EX-08).
fn read_f32_key(gguf: &GgufFile, key: &str) -> Result<f32> {
    let raw = gguf.get(key).and_then(|v| v.as_f64()).ok_or_else(|| {
        VokraError::ModelLoad(format!(
            "nisqa: GGUF metadata `{key}` is missing or is not a float. The \
             `vokra.nisqa.*` groups are all-or-nothing — a partially stamped group \
             is a bug in `{SIDECAR_PATH}` (FR-EX-08)."
        ))
    })?;
    Ok(raw as f32)
}

// ---------------------------------------------------------------------------
// NisqaConfig.
// ---------------------------------------------------------------------------

/// NISQA runtime config.
///
/// [`Self::variant`] is always real (derived from the tensor manifest);
/// the two spec groups are `None` until the sidecar stamps them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NisqaConfig {
    /// Which upstream class the checkpoint implements.
    pub variant: NisqaVariant,
    /// Mel front-end hyper-parameters, when stamped.
    pub front_end: Option<NisqaFrontEndSpec>,
    /// Non-shape-recoverable topology hyper-parameters, when stamped.
    pub topology: Option<NisqaTopologySpec>,
}

impl NisqaConfig {
    /// Derives the config from a parsed GGUF.
    ///
    /// The variant is discriminated by the pooling-head tensor prefix
    /// (`pool_layers.` → [`NisqaVariant::MultiDim`], `pool.` →
    /// [`NisqaVariant::SingleOutput`]); a GGUF carrying neither is
    /// refused rather than defaulted.
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] when neither pooling prefix is
    ///   present, when both are, when a `pool_layers.N.` index is
    ///   missing, or when a stamped `vokra.nisqa.*` group is malformed.
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let names: Vec<&str> = gguf.tensors().iter().map(|t| t.name.as_str()).collect();

        let has_multi = names
            .iter()
            .any(|n| n.starts_with(TENSOR_PREFIX_POOL_LAYERS));
        // `TENSOR_PREFIX_POOL` carries a trailing dot, so `pool_layers.0.`
        // (underscore) does not match it today. The explicit exclusion is
        // defensive: if either prefix const is ever edited to drop its
        // separator, the two probes must not start overlapping silently.
        let has_single = names.iter().any(|n| {
            n.starts_with(TENSOR_PREFIX_POOL) && !n.starts_with(TENSOR_PREFIX_POOL_LAYERS)
        });

        let variant = match (has_multi, has_single) {
            (true, false) => NisqaVariant::MultiDim,
            (false, true) => NisqaVariant::SingleOutput,
            (true, true) => {
                return Err(VokraError::ModelLoad(format!(
                    "nisqa: GGUF carries BOTH `{TENSOR_PREFIX_POOL_LAYERS}` (upstream \
                     `NISQA_DIM.pool_layers`) and `{TENSOR_PREFIX_POOL}` (upstream \
                     `NISQA.pool`) tensors. Those are alternative top-level classes \
                     in `{PRIMARY_SOURCE_MODEL_DEF}`, never both — this GGUF merges \
                     two checkpoints. Refusing rather than picking one (FR-EX-08)."
                )));
            }
            (false, false) => {
                return Err(VokraError::ModelLoad(format!(
                    "nisqa: GGUF carries no pooling-head tensors — expected either \
                     `{TENSOR_PREFIX_POOL_LAYERS}0.` … \
                     `{TENSOR_PREFIX_POOL_LAYERS}{last}.` (upstream `NISQA_DIM`, \
                     five cloned attention-pooling heads) or `{TENSOR_PREFIX_POOL}` \
                     (upstream `NISQA`, one head). Without a pooling head there is \
                     nothing to read a score out of. Was this GGUF produced by \
                     `vokra-cli convert --model nisqa-v2-weight` from a \
                     `{UPSTREAM_URL}` checkpoint?",
                    last = N_HEADS - 1
                )));
            }
        };

        // For the multidimensional variant every one of the five cloned
        // heads must be present — a gap would silently shorten the head
        // vector, and `NisqaScore::from_heads` would then reject a
        // forward that had already run.
        if variant == NisqaVariant::MultiDim {
            for (head, dimension) in HEAD_ORDER.iter().enumerate() {
                let prefix = format!("{TENSOR_PREFIX_POOL_LAYERS}{head}.");
                if !names.iter().any(|n| n.starts_with(prefix.as_str())) {
                    return Err(VokraError::ModelLoad(format!(
                        "nisqa: multidimensional checkpoint is missing every tensor \
                         under `{prefix}` (the `{dimension}` head, upstream \
                         `{dimension}_pred`). `NISQA_DIM` clones its pooling module \
                         {N_HEADS} times ({HEAD_ORDER:?}); a missing clone would \
                         silently shorten the score vector (FR-EX-08)."
                    )));
                }
            }
        }

        // The framewise CNN stage is present in every upstream variant
        // that is not configured with `cnn_model: skip`, and no released
        // checkpoint uses `skip`.
        if !names.iter().any(|n| n.starts_with(TENSOR_PREFIX_CNN)) {
            return Err(VokraError::ModelLoad(format!(
                "nisqa: GGUF carries no tensors under `{TENSOR_PREFIX_CNN}` — the \
                 framewise stage (upstream `NISQA_DIM.cnn`, an `AdaptCNN` with six \
                 Conv2d + BatchNorm2d layers) is missing. Every released \
                 `{UPSTREAM_URL}` checkpoint carries it; a GGUF without it is \
                 mis-produced (FR-EX-08)."
            )));
        }

        Ok(Self {
            variant,
            front_end: NisqaFrontEndSpec::from_gguf(gguf)?,
            topology: NisqaTopologySpec::from_gguf(gguf)?,
        })
    }
}

// ---------------------------------------------------------------------------
// Nisqa — the runtime binder handle.
// ---------------------------------------------------------------------------

/// Strict native NISQA v2 multidimensional scorer.
///
/// The learned CNN, attention and pooling reductions run through one selected
/// [`Compute`] backend. Front-end DSP, activations, layout changes, inference
/// BatchNorm and adaptive pooling are deterministic host glue, not a fallback.
#[derive(Debug, Clone)]
pub struct Nisqa {
    cfg: NisqaConfig,
    weights: Arc<NisqaWeights>,
    weight_license: LicenseClass,
    backend: BackendKind,
}

impl Nisqa {
    /// Strictly binds the exact public 94-tensor multidimensional release.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        let checkpoint = StrictCheckpoint::bind(file, SPEC)?;
        require_string(file, chunks::KEY_PROVENANCE_MODEL_ID, NAME)?;
        require_string(file, KEY_MODEL_CATEGORY, CATEGORY)?;
        require_string(file, KEY_PROVENANCE_UPSTREAM_URL, UPSTREAM_URL)?;
        require_string(file, chunks::KEY_PROVENANCE_LICENSE, DEFAULT_LICENSE_SPDX)?;
        require_string(
            file,
            chunks::KEY_PROVENANCE_WEIGHT_LICENSE,
            LicenseClass::NonCommercialShareAlike.as_str(),
        )?;
        validate_additive_contract(file)?;

        let mut cfg = NisqaConfig::from_gguf(file)?;
        if cfg.variant != NisqaVariant::MultiDim {
            return Err(VokraError::ModelLoad(
                "nisqa: the pinned public release must carry five `pool_layers.*` heads".to_owned(),
            ));
        }
        match cfg.front_end {
            Some(actual) if actual != CANONICAL_FRONT_END => {
                return Err(VokraError::ModelLoad(format!(
                    "nisqa: stamped front-end {actual:?} differs from the audited checkpoint args {CANONICAL_FRONT_END:?}"
                )));
            }
            None => cfg.front_end = Some(CANONICAL_FRONT_END),
            Some(_) => {}
        }
        match cfg.topology {
            Some(actual) if actual != CANONICAL_TOPOLOGY => {
                return Err(VokraError::ModelLoad(format!(
                    "nisqa: stamped topology {actual:?} differs from the audited checkpoint args {CANONICAL_TOPOLOGY:?}"
                )));
            }
            None => cfg.topology = Some(CANONICAL_TOPOLOGY),
            Some(_) => {}
        }
        let weights = Arc::new(NisqaWeights::bind(file)?);

        Ok(Self {
            cfg,
            weights,
            weight_license: checkpoint.weight_license(),
            backend: BackendKind::Cpu,
        })
    }

    /// Opens and binds the model from a GGUF file on disk.
    ///
    /// # Errors
    ///
    /// - Whatever [`GgufFile::open`] returns, plus every error of
    ///   [`Self::from_gguf`].
    pub fn from_path(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let gguf = GgufFile::open(path)?;
        Self::from_gguf(&gguf)
    }

    /// Strictly binds a GGUF and preflights one complete backend route.
    pub fn from_gguf_with_backend(file: &GgufFile, backend: BackendKind) -> Result<Self> {
        Compute::for_backend(backend, NISQA_HOT_OPS)?;
        Ok(Self::from_gguf(file)?.with_backend(backend))
    }

    /// Opens a GGUF and preflights one complete backend route.
    pub fn from_path_with_backend(
        path: impl AsRef<std::path::Path>,
        backend: BackendKind,
    ) -> Result<Self> {
        Self::from_gguf_with_backend(&GgufFile::open(path)?, backend)
    }

    /// Selects a backend; availability is checked again before scoring.
    #[must_use]
    pub fn with_backend(mut self, backend: BackendKind) -> Self {
        self.backend = backend;
        self
    }

    /// Selected backend.
    #[must_use]
    pub const fn backend(&self) -> BackendKind {
        self.backend
    }

    /// The manifest-derived config.
    #[inline]
    #[must_use]
    pub const fn config(&self) -> &NisqaConfig {
        &self.cfg
    }

    /// Which upstream class the bound checkpoint implements.
    #[inline]
    #[must_use]
    pub const fn variant(&self) -> NisqaVariant {
        self.cfg.variant
    }

    /// Number of tensors bound from the GGUF.
    #[inline]
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.weights.tensor_count()
    }

    /// The stamped weight-license class.
    ///
    /// The converter stamps `cc-by-nc-sa-4.0` →
    /// [`LicenseClass::NonCommercialShareAlike`] by default; a GGUF
    /// without the stamp reads back as [`LicenseClass::Unknown`]
    /// (fail-closed at the M2-13 compliance gate).
    #[inline]
    #[must_use]
    pub const fn weight_license(&self) -> LicenseClass {
        self.weight_license
    }

    /// `true` when the bound weights may only be used behind a research
    /// flag — the canonical `gabrielmittag/NISQA` weights
    /// (CC-BY-NC-SA-4.0) and any unstamped GGUF both answer `true`.
    ///
    /// Use this to gate a pipeline; it does **not** replace the
    /// publish-side `--allow-noncommercial` gate.
    #[inline]
    #[must_use]
    pub fn is_research_only(&self) -> bool {
        self.weight_license.requires_research_flag()
    }

    /// Legacy rate-less entry point.
    ///
    /// The released checkpoint keeps the file's native sample rate, so the
    /// caller must use [`Self::score_at_sample_rate`]. Guessing 48 kHz here
    /// would silently change STFT window and hop sizes for 16/44.1 kHz input.
    pub fn score(&self, _pcm: &[f32]) -> Result<NisqaScore> {
        Err(VokraError::InvalidArgument(
            "nisqa: native input sample rate is required; call `score_at_sample_rate(pcm, sample_rate)`"
                .to_owned(),
        ))
    }

    /// Predicts all five NISQA v2 dimensions for mono PCM at `sample_rate`.
    pub fn score_at_sample_rate(&self, pcm: &[f32], sample_rate: u32) -> Result<NisqaScore> {
        let compute = Compute::for_backend(self.backend, NISQA_HOT_OPS)?;
        let front_end = self.cfg.front_end.as_ref().ok_or_else(|| {
            VokraError::ModelLoad("nisqa: audited front-end is unavailable".to_owned())
        })?;
        let topology = self.cfg.topology.as_ref().ok_or_else(|| {
            VokraError::ModelLoad("nisqa: audited topology is unavailable".to_owned())
        })?;
        nn::score(
            &compute,
            &self.weights,
            front_end,
            topology,
            pcm,
            sample_rate,
        )
    }

    /// Legacy rate-less overall-MOS entry point; see [`Self::score`].
    pub fn score_overall(&self, _pcm: &[f32]) -> Result<f32> {
        Err(VokraError::InvalidArgument(
            "nisqa: native input sample rate is required; call `score_overall_at_sample_rate(pcm, sample_rate)`"
                .to_owned(),
        ))
    }

    /// Predicts the overall MOS (head zero) for mono PCM at `sample_rate`.
    pub fn score_overall_at_sample_rate(&self, pcm: &[f32], sample_rate: u32) -> Result<f32> {
        Ok(self.score_at_sample_rate(pcm, sample_rate)?.mos)
    }
}

fn validate_additive_contract(file: &GgufFile) -> Result<()> {
    let keys = [
        KEY_SOURCE_REVISION,
        KEY_SOURCE_MODEL_DEF_SHA256,
        KEY_SOURCE_CONFIG_SHA256,
        KEY_SOURCE_WEIGHT_LICENSE_SHA256,
        KEY_SOURCE_CHECKPOINT_SHA256,
        KEY_PUBLIC_HF,
        KEY_PUBLIC_REVISION,
        KEY_PUBLIC_GGUF_SHA256,
        KEY_MANIFEST_SHA256,
    ];
    // The historical public artifact predates the richer pins. Its exact
    // complete manifest plus generic provenance is the compatibility proof.
    if !keys.iter().any(|key| file.get(key).is_some()) {
        return Ok(());
    }
    for (key, expected) in [
        (KEY_SOURCE_REVISION, SOURCE_REVISION),
        (KEY_SOURCE_MODEL_DEF_SHA256, SOURCE_MODEL_DEF_SHA256),
        (KEY_SOURCE_CONFIG_SHA256, SOURCE_CONFIG_SHA256),
        (
            KEY_SOURCE_WEIGHT_LICENSE_SHA256,
            SOURCE_WEIGHT_LICENSE_SHA256,
        ),
        (KEY_SOURCE_CHECKPOINT_SHA256, SOURCE_CHECKPOINT_SHA256),
        (KEY_PUBLIC_HF, PUBLIC_HF),
        (KEY_PUBLIC_REVISION, PUBLIC_REVISION),
        (KEY_PUBLIC_GGUF_SHA256, PUBLIC_GGUF_SHA256),
        (KEY_MANIFEST_SHA256, MANIFEST_SHA256),
    ] {
        require_string(file, key, expected)?;
    }
    Ok(())
}

fn require_string(file: &GgufFile, key: &str, expected: &str) -> Result<()> {
    let actual = file.get(key).and_then(GgufMetadataValue::as_str);
    if actual != Some(expected) {
        return Err(VokraError::ModelLoad(format!(
            "nisqa: metadata `{key}`={actual:?}, expected {expected:?}"
        )));
    }
    Ok(())
}
