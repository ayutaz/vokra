#![allow(clippy::doc_lazy_continuation)]
//! **ATST** (`Audio-WestlakeU/audiossl/tree/main/audiossl/methods/atst`,
//! **cc-by-4.0 weight** / mit code): safetensors → GGUF conversion
//! (SSL audio-encoder wave, 2026-08-13).
//!
//! Input: the upstream `Audio-WestlakeU/audiossl` release — ATST
//! ("Audio Teacher-Student Transformer") is a self-supervised
//! audio encoder trained via a **BYOL-style EMA teacher +
//! student patchout** objective over log-mel spectrogram
//! (Li et al. 2022, INTERSPEECH, arXiv:2204.12076; frame-level
//! extension "atstframe" Li et al. 2023, TASLP, arXiv:2306.04186).
//! Positioned as an efficient audio-embedding backbone for
//! downstream sound-event detection / audio-tagging / speaker
//! tasks. ~86M parameter class base variant (~200 MB checkpoint).
//!
//! # Vokra scope — SSL audio encoder (2026-07-30 scope expansion)
//!
//! Sibling of `beats` (iterative-tokenizer SSL), `eat`
//! (utterance-level Transformer + inverse block masking),
//! `dasheng` (universal MAE), `m2d` (masked-modeling-duo).
//! Distinct arch tag `atst` because the BYOL-style teacher-
//! student patchout topology is a distinct axis from every sibling
//! SSL encoder (contrastive / masked / dual-branch objectives all
//! differ) — silently sharing would misroute the runtime
//! dispatch and try to bind e.g. a MAE decoder over a
//! teacher-student checkpoint (FR-EX-08). Category
//! `audio-embedding`.
//!
//! # License posture — **cc-by-4.0 (weight) / mit (code)** — split
//!
//! **The upstream README explicitly separates code and weight
//! licenses**:
//!
//! > "The pretrained checkpoints hyper-linked in this repo are
//! > licensed under CC BY 4.0. To view a copy of this license,
//! > visit http://creativecommons.org/licenses/by/4.0/
//! >
//! > audiossl is licenced under MIT Licence."
//!
//! (`raw.githubusercontent.com/Audio-WestlakeU/audiossl/main/LICENSE`,
//! primary source task input 2026-08-13). GitHub API
//! `/repos/Audio-WestlakeU/audiossl/license` returns
//! `spdx_id: NOASSERTION` — GitHub's classifier does not know how
//! to combine "code MIT / weight CC-BY-4.0" into a single SPDX,
//! so the primary source is the LICENSE file text itself.
//!
//! **Vokra records the WEIGHT license (`cc-by-4.0`,
//! `AttributionRequired`) since `vokra.provenance.weight_license`
//! is a weight-tracking stamp**, not a code-tracking stamp. The
//! code SPDX (`mit`) applies to the ATST training code but is
//! not what a weight-provenance stamp records. Downstream
//! distributors of the weight must comply with CC BY 4.0
//! attribution requirements. §3.1 sign-off stays blank fail-closed
//! until owner completes primary-source confirmation.
//!
//! # Scale — local convert OK (~0.2 GB)
//!
//! Well below the M1 iMac 16 GB local-convert threshold (memory
//! [[feedback-large-models-on-vast-ai]]: <2 GB safe). No vast.ai
//! handoff required.
//!
//! # No ONNX / no pickle (permanent)
//!
//! ATST ships as PyTorch `.ckpt` pickle from the upstream repo
//! release; this converter **never** touches ONNX or pickle
//! (FR-LD-05 / NFR-DS-02). Callers pre-flatten via a future
//! `tools/parity/atst_prepare_checkpoint.py` uv-managed Python
//! 3.12 sidecar (memory `[[feedback-python-uses-uv]]` +
//! `[[feedback-python-3-12]]`) mirroring the DAC / Kokoro /
//! UTMOSv2 bridge pattern.
//!
//! # BF16 pass-through
//!
//! F32 / F16 / BF16 all ride the verbatim pass-through arm. BF16
//! is emitted as GGUF type 30 ([`GgmlType::BF16`]); the runtime
//! widens BF16 → f32 losslessly at load via the single choke
//! point `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16`.
//!
//! # Topology — the `vokra.atst.*` chunk group
//!
//! Until 2026-08-15 this converter stamped only `vokra.model.*` +
//! `vokra.provenance.*`, so a converted artifact said nothing about
//! its own shape and the runtime binder
//! (`crates/vokra-models/src/atst/mod.rs`) had to list "no
//! `vokra.atst.*` axis chunk group" as blocker (i) of its
//! loud-partial forward. This module now stamps that group:
//! Transformer width / depth / head count, the ViT patch grid and
//! position-table length, and the full log-mel front-end.
//!
//! **Every value is transcribed from the upstream source tree, not
//! from a config file — ATST publishes none.** The released
//! `base.ckpt` comes from `train_base.sh`, which passes `--arch base`
//! and no topology flag, and the flag never reaches the encoder
//! (`ATSTLightningModule` constructs `ATST(arch=arch)` and `ATST`
//! forwards an empty `**kwargs` to `AST_base()`), so the
//! source-level defaults are binding rather than merely indicative.
//! The constants block below records the exact file and line for
//! each axis, plus the axes deliberately **omitted** — the tensor
//! naming, the teacher/student inference branch, the parameter
//! count and the projection-head dims are all still unsourced, and
//! a guessed value there would bind shape-valid garbage (FR-EX-08).
//! Binder blockers (iii) and (iv) therefore stand; this change
//! closes only (i).

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` for ATST GGUFs. Distinct from sibling SSL
/// audio-encoder arch tags (`beats` / `eat` / `dasheng` / `m2d` /
/// `mert` / `muq`) — ATST's BYOL-style teacher-student patchout
/// training target is a distinct topology axis from every sibling.
pub const ARCH: &str = "atst";

/// `vokra.model.name` — canonical `atst-base` size point (the
/// INTERSPEECH 2022 release; sibling `atst-frame` is the
/// frame-level TASLP 2023 extension published as its own future
/// `NAME` following the snac_24khz / snac_44khz pattern).
pub const NAME: &str = "atst-base";

/// `vokra.model.category` — general audio-embedding (sibling of
/// `dasheng` / `beats` / `eat` / `m2d`; downstream sound-event
/// detection / audio-tagging / speaker heads feed from the
/// encoder's hidden states).
pub const CATEGORY: &str = "audio-embedding";

/// `vokra.provenance.upstream_url` value — the GitHub tree the
/// release ships from. ATST is not hosted on HuggingFace, so this
/// uses `upstream_url` rather than `upstream_hf`; the model-card
/// generator picks up either. Sibling of `beats::UPSTREAM_URL` /
/// `eat::UPSTREAM_URL` / `nsnet2::UPSTREAM_URL` posture.
pub const UPSTREAM_URL: &str =
    "github.com/Audio-WestlakeU/audiossl/tree/main/audiossl/methods/atst";

/// Default SPDX. **Weight license** = `cc-by-4.0` per upstream
/// LICENSE file text primary source (task input 2026-08-13):
///
/// > "The pretrained checkpoints hyper-linked in this repo are
/// > licensed under CC BY 4.0."
///
/// The code SPDX would be `mit` but `vokra.provenance.weight_license`
/// tracks the weight tier, not the code tier — CC-BY-4.0 is the
/// enforceable posture for weight redistribution
/// (`AttributionRequired` — downstream distributors must credit
/// the ATST authors per CC BY 4.0). A caller with a different
/// attestation may override at the outer boundary
/// (`--license <spdx>`).
pub const DEFAULT_LICENSE_SPDX: &str = "cc-by-4.0";

// ---------------------------------------------------------------------------
// ATST-base topology axes — the `vokra.atst.*` chunk group
//
// **Where these numbers come from.** ATST publishes no `config.json`: the
// released `base.ckpt` is produced by a shell script whose every topology axis
// is a *source-level default*, so the primary source is the upstream code
// itself. Each file below was fetched raw from
// `raw.githubusercontent.com/Audio-WestlakeU/audiossl/main/...` on 2026-08-15
// and the defaults are transcribed verbatim. The call chain that pins them:
//
//   1. `audiossl/methods/atst/train_base.sh` — the script that produced the
//      published `base.ckpt`:
//        `python train.py --arch base --data_path $1 --save_path $2 --nproc 6
//         --batch_size_per_gpu 256 --warmup_steps 15860 --max_steps 318000
//         --ema 0.9995 --subset 3000000 --learning_rate 2e-4`
//      → `--arch base`, and **no** topology flag is passed.
//   2. `audiossl/methods/atst/model.py`:
//        `parser.add_argument("--arch",type=str,default="small")`
//        `self.model = ATST(arch=arch)`
//      → only `arch` reaches `ATST`; `add_model_specific_args` declares no
//        `--patch_h` / `--patch_w` / `--spec_h` / `--spec_w` (its other
//        arguments are `--learning_rate` / `--ema` / `--warmup_steps` /
//        `--max_steps`, all training-schedule only).
//   3. `audiossl/models/atst/atst.py`:
//        `def __init__(self,arch="small",ncrops=2,**kwargs):`
//        `elif arch == "base": encoder_fn = AST_base` with `embed_dim = 768`
//        `self.student=MultiCropWrapper(encoder_fn(**kwargs),embed_dim,predictor=True)`
//        `self.teacher=MultiCropWrapper(encoder_fn(**kwargs),embed_dim,predictor=False)`
//      → because step 2 forwards nothing, `kwargs` is **empty** and
//        `AST_base()` runs with every argument at its declared default. That
//        is what makes the source-level defaults binding rather than merely
//        indicative.
//   4. `audiossl/models/atst/audio_transformer.py`:
//        `def AST_base(patch_h=64,patch_w=4,**kwargs): return AST(patch_h=patch_h,
//         patch_w=patch_w,embed_dim=768,depth=12,num_heads=12,qkv_bias=False,
//         norm_layer=partial(nn.LayerNorm, eps=1e-6),**kwargs)`
//        `AST(... spec_h=64,spec_w=1001 ... mlp_ratio=4. ... in_chans=1,
//         num_classes=0 ... use_cls=True ... drop_rate=0., attn_drop_rate=0.,
//         drop_path_rate=0.1 ... pos_type="cut")`
//        `num_patches = get_num_patches(spec_h,spec_w,patch_h,patch_w)` where
//        `get_num_patches` returns `(height // patch_height) * (width // patch_width)`
//        `self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))`
//        `self.patch_embed = PatchEmbed_v2(patch_h,patch_w,embed_dim)` whose
//        body is `Rearrange('b c (h p1) (w p2) -> b (w h) (p1 p2 c)', p1 =
//        patch_height, p2 = patch_width)` + `nn.Linear(patch_height*patch_width,embed_dim)`.
//   5. `audiossl/methods/atst/transform.py` — the log-mel front-end.
//   6. `audiossl/modules/transformer.py` — `Block` / `Mlp` / `Attention`:
//        `act_layer=nn.GELU`, `nn.Linear(dim, dim * 3, bias=qkv_bias)`, and an
//        FFN hidden width of `int(dim * mlp_ratio)`.
//
// **Derived values are computed here from the transcribed primitives** rather
// than written as literals, so the upstream formula (quoted above) stays
// visible and the arithmetic cannot drift: see [`NUM_PATCHES`],
// [`POS_EMBED_LEN`], [`MLP_HIDDEN_DIM`], [`PATCH_EMBED_IN_FEATURES`],
// [`PATCH_GRID`].
//
// **Deliberate omissions.** A missing key is honest; a guessed one binds
// shape-valid garbage (FR-EX-08). Not stamped, and why:
//
//   * **Tensor-name manifest / branch prefix.** The on-disk `state_dict` key
//     chain runs `ATSTLightningModule.model` → `ATST.student` / `.teacher` →
//     `MultiCropWrapper` → the `AST` encoder, so the real prefix is at least
//     `model.student.` and not the bare `student.` this module's own test
//     fixtures use. No checkpoint key listing was read, so no prefix is
//     stamped and blocker (iv) of `vokra-models/src/atst/mod.rs` stands.
//   * **Which branch inference uses.** `atst.py` proves both branches exist
//     but names no inference entry point, so blocker (iii) stands. Nothing
//     here resolves it.
//   * **Parameter count.** The "~86M params" figure in this module's own
//     docstring is not stated by any upstream file read here, so it is not
//     promoted to a stamped axis.
//   * **Projection-head dims** (`MultiCropWrapper` internals) — not read, and
//     SSL-pretraining-only in any case.
//   * **Training crop length.** `transform.py` carries anchor/positive crop
//     *ranges*, not a single inference-time length; [`SPEC_W`] is the axis
//     that actually sizes the position table, so the crop range is omitted.
// ---------------------------------------------------------------------------

/// Transformer width (`embed_dim`) — **768** (`AST_base`).
pub const EMBED_DIM: u32 = 768;

/// Transformer block count (`depth`) — **12** (`AST_base`).
pub const DEPTH: u32 = 12;

/// Attention head count (`num_heads`) — **12** (`AST_base`), i.e. a head dim
/// of `EMBED_DIM / NUM_HEADS` = 64.
pub const NUM_HEADS: u32 = 12;

/// FFN expansion ratio (`mlp_ratio=4.`) scaled by 1000 so it round-trips as an
/// integer chunk without floating-point serialization ambiguity — **4000**.
/// The sibling `vokra.wavlm.*` group uses the same scaled-integer dance.
pub const MLP_RATIO_SCALED_1E3: u32 = 4_000;

/// FFN hidden width — derived as `int(dim * mlp_ratio)` per the `Block` body in
/// `audiossl/modules/transformer.py`, evaluated in exact integer arithmetic
/// from [`EMBED_DIM`] and [`MLP_RATIO_SCALED_1E3`] (768 × 4.0 = **3072**).
pub const MLP_HIDDEN_DIM: u32 = EMBED_DIM * MLP_RATIO_SCALED_1E3 / 1_000;

/// Patch height over the **mel** axis (`patch_h`) — **64** (`AST_base`).
/// Equal to [`N_MELS`], so one patch spans the full mel range and the grid is
/// one row tall.
pub const PATCH_H: u32 = 64;

/// Patch width over the **frame** axis (`patch_w`) — **4** (`AST_base`).
pub const PATCH_W: u32 = 4;

/// Mel-plane height the position table is built for (`spec_h`) — **64**.
/// Cross-check: identical to [`N_MELS`] from the front-end, as it must be.
pub const SPEC_H: u32 = 64;

/// Mel-plane width the position table is built for (`spec_w`) — **1001**.
///
/// This is a **maximum**, not a fixed input length: `pos_type="cut"` means the
/// position table is allocated for `spec_w` frames and sliced down to the
/// actual input length at forward time. It is therefore a genuine weight-shape
/// axis (it sizes `pos_embed`), which is why it is stamped.
pub const SPEC_W: u32 = 1001;

/// Patch-grid rows — `spec_h // patch_h` = **1**.
pub const PATCH_GRID_H: u32 = SPEC_H / PATCH_H;

/// Patch-grid columns — `spec_w // patch_w` = **250**.
pub const PATCH_GRID_W: u32 = SPEC_W / PATCH_W;

/// Patch grid as a 2-element axis array `[rows, cols]` = `[1, 250]`, stamped
/// under indexed keys (`vokra.atst.patch_grid_0` / `_1`) exactly like the
/// sibling `vokra.wavlm.conv_*_{i}` arrays so a reader reconstructs order
/// without array parsing.
pub const PATCH_GRID: [u32; 2] = [PATCH_GRID_H, PATCH_GRID_W];

/// Patch count — `get_num_patches(spec_h,spec_w,patch_h,patch_w)`, i.e.
/// `(spec_h // patch_h) * (spec_w // patch_w)` = **250**.
pub const NUM_PATCHES: u32 = PATCH_GRID_H * PATCH_GRID_W;

/// Length of the position table — `num_patches + 1` = **251**, matching
/// `nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))`. The `+ 1` is the
/// CLS slot (see [`USE_CLS`]).
pub const POS_EMBED_LEN: u32 = NUM_PATCHES + 1;

/// Input width of the patch-embedding `nn.Linear` — `patch_h * patch_w` =
/// **256**, so the projection weight is `[EMBED_DIM, 256]`.
pub const PATCH_EMBED_IN_FEATURES: u32 = PATCH_H * PATCH_W;

/// How patches are embedded — `"linear"`.
///
/// Load-bearing distinction: `PatchEmbed_v2` is a `Rearrange` followed by
/// `nn.Linear(patch_h*patch_w, embed_dim)`, **not** the `Conv2d` stem that
/// most ViT ports assume. A Conv2d reading would look for a 4-D weight and
/// fail, or worse, bind a reshaped one.
pub const PATCH_EMBED_KIND: &str = "linear";

/// Patch flattening order — `"wh"`, transcribed from the `Rearrange` pattern
/// `'b c (h p1) (w p2) -> b (w h) (p1 p2 c)'`: the sequence axis is built
/// **width-major, then height**. Stamped because a `hw` reading yields a
/// silently transposed sequence rather than an error. Degenerate at
/// [`PATCH_GRID_H`] = 1 but recorded so a future non-square variant cannot
/// inherit the wrong assumption.
pub const PATCH_ORDER: &str = "wh";

/// `qkv_bias=False` — the fused QKV projection is
/// `nn.Linear(dim, dim * 3, bias=False)` and carries **no** bias tensor.
pub const QKV_BIAS: bool = false;

/// `use_cls=True` — a CLS token is prepended, which is what makes
/// [`POS_EMBED_LEN`] one longer than [`NUM_PATCHES`].
pub const USE_CLS: bool = true;

/// LayerNorm epsilon — **1e-6**, from `partial(nn.LayerNorm, eps=1e-6)` in the
/// `AST_base` factory. Stamped as an f32 chunk (the GGUF reader widens F32 →
/// f64 exactly), unlike the sibling `vokra.wavlm.layer_norm_eps_scaled_1e9`
/// integer encoding.
pub const LAYER_NORM_EPS: f32 = 1e-6;

/// Position-embedding interpolation policy — `"cut"`, i.e. allocate for
/// [`SPEC_W`] frames and slice, never interpolate. See [`SPEC_W`].
pub const POS_TYPE: &str = "cut";

/// FFN activation — `"gelu"`, from `act_layer=nn.GELU` on both `Block` and
/// `Mlp` in `audiossl/modules/transformer.py`.
pub const ACT_LAYER: &str = "gelu";

/// Input channel count (`in_chans`) — **1** (a single mono mel plane).
pub const IN_CHANS: u32 = 1;

/// Classifier width (`num_classes`) — **0**.
///
/// Zero is the load-bearing value, not a placeholder: the release ships a bare
/// feature extractor with **no task head**, so a consumer must never look for
/// a classifier or invent a label list.
pub const NUM_CLASSES: u32 = 0;

/// Stochastic-depth rate (`drop_path_rate=0.1`) scaled by 1000 — **100**.
/// Training-time regularization only (inactive at inference); stamped for
/// audit completeness, mirroring the sibling
/// `vokra.wavlm.hidden_dropout_scaled_1e3`.
pub const DROP_PATH_RATE_SCALED_1E3: u32 = 100;

/// Front-end sample rate (`sr`) — **16000** Hz, mono.
pub const SAMPLE_RATE: u32 = 16_000;

/// Front-end FFT size (`n_fft`) — **1024**.
pub const N_FFT: u32 = 1024;

/// Front-end hop (`hop_length`) — **160** samples = 10 ms at 16 kHz.
pub const HOP_LENGTH: u32 = 160;

/// Front-end window length (`win_length`) — **1024**, equal to [`N_FFT`].
pub const WIN_LENGTH: u32 = 1024;

/// Mel band count (`n_mels`) — **64**. Cross-check: equals [`SPEC_H`].
pub const N_MELS: u32 = 64;

/// Mel low edge (`f_min`) — **60** Hz.
pub const F_MIN: u32 = 60;

/// Mel high edge (`f_max`) — **7800** Hz.
pub const F_MAX: u32 = 7800;

/// `AmplitudeToDB` dynamic-range clamp (`top_db`) — **80** dB.
pub const AMP_TO_DB_TOP_DB: u32 = 80;

/// `AmplitudeToDB` input scale (`stype`) — `"power"` (not `"magnitude"`; the
/// two differ by a factor of two in the dB conversion).
pub const AMP_TO_DB_STYPE: &str = "power";

/// Min-max normalization floor — **-79.6482** dB, from
/// `MinMax(min=-79.6482,max=50.6842)` in `transform.py`.
///
/// A dataset-fitted constant, not a derivable one: it must be reproduced
/// exactly or every embedding shifts.
pub const NORM_MIN: f32 = -79.6482;

/// Min-max normalization ceiling — **50.6842** dB. See [`NORM_MIN`].
pub const NORM_MAX: f32 = 50.6842;

const UPSTREAM_SOURCE: &str = "Audio-WestlakeU/audiossl/methods/atst (Audio Teacher-Student Transformer, BYOL-style \
     EMA teacher + student patchout SSL audio encoder, ~86M params base, Li et al. \
     arXiv:2204.12076 INTERSPEECH 2022 + arXiv:2306.04186 TASLP 2023, code mit / weight cc-by-4.0)";

const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
const KEY_PROVENANCE_UPSTREAM_URL: &str = "vokra.provenance.upstream_url";

// ---------------------------------------------------------------------------
// `vokra.atst.*` chunk keys.
//
// **Naming provenance.** As of this landing the runtime binder
// (`crates/vokra-models/src/atst/mod.rs`) defines **no** `vokra.atst.*` key
// constants — it only documents their absence as blocker (i) of its
// loud-partial `encode` / `embed`. The spellings below therefore mirror the
// `vokra.wavlm.*` convention (`crates/vokra-convert/src/models/wavlm_sv.rs`):
// bare snake_case axis names under a `vokra.<arch>.` prefix, integer-scaled
// suffixes (`_scaled_1e3`) where a float is encoded as an integer, and
// indexed `_{i}` suffixes for axis arrays. The binder must mirror these
// spellings when it lands its `GGUF_KEY_ATST_*` constants.
// ---------------------------------------------------------------------------

// Transformer topology.
const KEY_ATST_EMBED_DIM: &str = "vokra.atst.embed_dim";
const KEY_ATST_DEPTH: &str = "vokra.atst.depth";
const KEY_ATST_NUM_HEADS: &str = "vokra.atst.num_heads";
const KEY_ATST_MLP_RATIO_SCALED_1E3: &str = "vokra.atst.mlp_ratio_scaled_1e3";
const KEY_ATST_MLP_HIDDEN_DIM: &str = "vokra.atst.mlp_hidden_dim";
const KEY_ATST_LAYER_NORM_EPS: &str = "vokra.atst.layer_norm_eps";
const KEY_ATST_QKV_BIAS: &str = "vokra.atst.qkv_bias";
const KEY_ATST_USE_CLS: &str = "vokra.atst.use_cls";
const KEY_ATST_ACT_LAYER: &str = "vokra.atst.act_layer";
const KEY_ATST_IN_CHANS: &str = "vokra.atst.in_chans";
const KEY_ATST_NUM_CLASSES: &str = "vokra.atst.num_classes";
const KEY_ATST_DROP_PATH_RATE_SCALED_1E3: &str = "vokra.atst.drop_path_rate_scaled_1e3";

// Patch grid / position table.
const KEY_ATST_PATCH_H: &str = "vokra.atst.patch_h";
const KEY_ATST_PATCH_W: &str = "vokra.atst.patch_w";
const KEY_ATST_SPEC_H: &str = "vokra.atst.spec_h";
const KEY_ATST_SPEC_W: &str = "vokra.atst.spec_w";
const KEY_ATST_NUM_PATCHES: &str = "vokra.atst.num_patches";
const KEY_ATST_POS_EMBED_LEN: &str = "vokra.atst.pos_embed_len";
const KEY_ATST_POS_TYPE: &str = "vokra.atst.pos_type";
const KEY_ATST_PATCH_EMBED_IN_FEATURES: &str = "vokra.atst.patch_embed_in_features";
const KEY_ATST_PATCH_EMBED_KIND: &str = "vokra.atst.patch_embed_kind";
const KEY_ATST_PATCH_ORDER: &str = "vokra.atst.patch_order";

/// Axis-array prefix — stamped as `vokra.atst.patch_grid_0` / `_1`.
const KEY_ATST_PATCH_GRID_PREFIX: &str = "vokra.atst.patch_grid";

// Log-mel front-end.
const KEY_ATST_SAMPLE_RATE: &str = "vokra.atst.sample_rate";
const KEY_ATST_N_FFT: &str = "vokra.atst.n_fft";
const KEY_ATST_HOP_LENGTH: &str = "vokra.atst.hop_length";
const KEY_ATST_WIN_LENGTH: &str = "vokra.atst.win_length";
const KEY_ATST_N_MELS: &str = "vokra.atst.n_mels";
const KEY_ATST_F_MIN: &str = "vokra.atst.f_min";
const KEY_ATST_F_MAX: &str = "vokra.atst.f_max";
const KEY_ATST_AMP_TO_DB_TOP_DB: &str = "vokra.atst.amp_to_db_top_db";
const KEY_ATST_AMP_TO_DB_STYPE: &str = "vokra.atst.amp_to_db_stype";
const KEY_ATST_NORM_MIN: &str = "vokra.atst.norm_min";
const KEY_ATST_NORM_MAX: &str = "vokra.atst.norm_max";

/// Outcome of an ATST conversion. Mirrors the counter shape of
/// the sibling BF16 pass-through converters (`beats` / `eat` /
/// `dasheng` / `mert` / `muq` / `yamnet`) — the invariant
/// `read == written + skipped_non_float` is auditable at the
/// report level.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct AtstReport {
    /// Total tensor entries observed on the safetensors input side.
    pub read: usize,
    /// Float tensors written verbatim (F32 / F16 / BF16).
    pub written: usize,
    /// Non-float tensors skipped (defensive counter — the safetensors
    /// reader accepts only F32 / F16 / BF16, so any tensor reaching
    /// this counter would signal a reader change upstream).
    pub skipped_non_float: usize,
    /// Of the tensors in [`Self::written`], how many were BF16 (subset
    /// counter). Emits GGUF type 30 verbatim; the runtime widens BF16
    /// → f32 losslessly via the single choke point
    /// `vokra_core::gguf::quant::decode_bf16`.
    pub bf16_passthrough: usize,
}

/// Converts an ATST safetensors checkpoint at `input` into a
/// Vokra-native GGUF at `output`, returning an [`AtstReport`].
///
/// Every F32 / F16 / BF16 tensor passes through under its upstream
/// state-dict name; the `vokra.model.*` / `vokra.provenance.*` chunks
/// are stamped for the runtime compliance gate (FR-CP-03).
///
/// `license` optionally overrides the stamped weight license (raw SPDX
/// string). The default is `DEFAULT_LICENSE_SPDX` (`"cc-by-4.0"`,
/// `AttributionRequired`) since the weight is the tracked artifact.
///
/// # Errors
///
/// - [`ConvertError::Io`] on read/write failure.
/// - [`ConvertError::Parse`] on malformed safetensors input.
/// - [`ConvertError::Gguf`] on GGUF assembly failure.
pub fn convert_atst_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<AtstReport, ConvertError> {
    let bytes = std::fs::read(input)?;
    let st = SafetensorsFile::parse(bytes)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    b.add_string(KEY_MODEL_CATEGORY, CATEGORY);
    b.add_string(KEY_PROVENANCE_UPSTREAM_URL, UPSTREAM_URL);

    // ATST-base topology axes. Transcribed from the upstream source tree —
    // see the constants block above for the exact file / line provenance of
    // every value and for the list of axes deliberately NOT stamped. This
    // group is purely additive: no pre-existing chunk changes value, and no
    // tensor payload is touched.
    b.add_u32(KEY_ATST_EMBED_DIM, EMBED_DIM);
    b.add_u32(KEY_ATST_DEPTH, DEPTH);
    b.add_u32(KEY_ATST_NUM_HEADS, NUM_HEADS);
    b.add_u32(KEY_ATST_MLP_RATIO_SCALED_1E3, MLP_RATIO_SCALED_1E3);
    b.add_u32(KEY_ATST_MLP_HIDDEN_DIM, MLP_HIDDEN_DIM);
    b.add_f32(KEY_ATST_LAYER_NORM_EPS, LAYER_NORM_EPS);
    b.add_bool(KEY_ATST_QKV_BIAS, QKV_BIAS);
    b.add_bool(KEY_ATST_USE_CLS, USE_CLS);
    b.add_string(KEY_ATST_ACT_LAYER, ACT_LAYER);
    b.add_u32(KEY_ATST_IN_CHANS, IN_CHANS);
    b.add_u32(KEY_ATST_NUM_CLASSES, NUM_CLASSES);
    b.add_u32(
        KEY_ATST_DROP_PATH_RATE_SCALED_1E3,
        DROP_PATH_RATE_SCALED_1E3,
    );

    b.add_u32(KEY_ATST_PATCH_H, PATCH_H);
    b.add_u32(KEY_ATST_PATCH_W, PATCH_W);
    b.add_u32(KEY_ATST_SPEC_H, SPEC_H);
    b.add_u32(KEY_ATST_SPEC_W, SPEC_W);
    b.add_u32(KEY_ATST_NUM_PATCHES, NUM_PATCHES);
    b.add_u32(KEY_ATST_POS_EMBED_LEN, POS_EMBED_LEN);
    b.add_string(KEY_ATST_POS_TYPE, POS_TYPE);
    b.add_u32(KEY_ATST_PATCH_EMBED_IN_FEATURES, PATCH_EMBED_IN_FEATURES);
    b.add_string(KEY_ATST_PATCH_EMBED_KIND, PATCH_EMBED_KIND);
    b.add_string(KEY_ATST_PATCH_ORDER, PATCH_ORDER);

    // Axis array: indexed keys (`vokra.atst.patch_grid_0` / `_1`) so the
    // reader reconstructs `[rows, cols]` order deterministically without
    // needing array parsing — the sibling `vokra.wavlm.conv_*_{i}` pattern.
    for (i, &v) in PATCH_GRID.iter().enumerate() {
        b.add_u32(&format!("{KEY_ATST_PATCH_GRID_PREFIX}_{i}"), v);
    }

    b.add_u32(KEY_ATST_SAMPLE_RATE, SAMPLE_RATE);
    b.add_u32(KEY_ATST_N_FFT, N_FFT);
    b.add_u32(KEY_ATST_HOP_LENGTH, HOP_LENGTH);
    b.add_u32(KEY_ATST_WIN_LENGTH, WIN_LENGTH);
    b.add_u32(KEY_ATST_N_MELS, N_MELS);
    b.add_u32(KEY_ATST_F_MIN, F_MIN);
    b.add_u32(KEY_ATST_F_MAX, F_MAX);
    b.add_u32(KEY_ATST_AMP_TO_DB_TOP_DB, AMP_TO_DB_TOP_DB);
    b.add_string(KEY_ATST_AMP_TO_DB_STYPE, AMP_TO_DB_STYPE);
    b.add_f32(KEY_ATST_NORM_MIN, NORM_MIN);
    b.add_f32(KEY_ATST_NORM_MAX, NORM_MAX);

    let spdx = license.unwrap_or(DEFAULT_LICENSE_SPDX);
    let class = LicenseClass::from_license_str(spdx);
    vokra_core::stamp_provenance(&mut b, class, spdx, Some(NAME), Some(UPSTREAM_SOURCE));

    let mut report = AtstReport::default();
    for t in st.tensors() {
        report.read += 1;
        match t.dtype {
            GgmlType::F32 | GgmlType::F16 | GgmlType::BF16 => {
                b.add_tensor(
                    &t.name,
                    t.dtype,
                    t.shape.clone(),
                    st.tensor_bytes(t).to_vec(),
                )?;
                report.written += 1;
                if t.dtype == GgmlType::BF16 {
                    report.bf16_passthrough += 1;
                }
            }
            _ => {
                report.skipped_non_float += 1;
            }
        }
    }

    let out_bytes = b.to_bytes()?;
    std::fs::write(output, &out_bytes)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};
    use vokra_core::gguf::GgufFile;

    fn tmp_path(tag: &str) -> PathBuf {
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let n = SEQ.fetch_add(1, Ordering::Relaxed);
        let mut p = std::env::temp_dir();
        p.push(format!(
            "vokra-convert-atst-{tag}-{}-{n}",
            std::process::id()
        ));
        p
    }

    fn safetensors_one(name: &str, dtype: &str, shape: &[u64], payload: &[u8]) -> Vec<u8> {
        let shape_str = shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let header = format!(
            r#"{{"{name}":{{"dtype":"{dtype}","shape":[{shape_str}],"data_offsets":[0,{}]}}}}"#,
            payload.len()
        );
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(payload);
        out
    }

    #[test]
    fn f32_tensor_passes_through_and_default_license_is_attribution_required() {
        let inp = tmp_path("f32-in");
        let outp = tmp_path("f32-out");
        let payload: Vec<u8> = [1.0_f32, 2.0, 3.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        // ATST uses a teacher/student duo — realistic upstream state-dict
        // name from the BYOL-style objective.
        let st = safetensors_one(
            "student.encoder.blocks.0.norm1.weight",
            "F32",
            &[3],
            &payload,
        );
        std::fs::write(&inp, &st).unwrap();

        let r = convert_atst_file(&inp, &outp, None).expect("convert F32");
        assert_eq!(r.read, 1);
        assert_eq!(r.written, 1);

        let g = GgufFile::open(&outp).unwrap();
        let read_str = |k: &str| -> String {
            g.get(k)
                .and_then(|v| v.as_str())
                .unwrap_or_else(|| panic!("{k}: missing"))
                .to_owned()
        };
        assert_eq!(read_str(chunks::KEY_MODEL_ARCH), ARCH);
        assert_eq!(read_str(chunks::KEY_MODEL_NAME), NAME);
        assert_eq!(read_str(KEY_MODEL_CATEGORY), CATEGORY);
        assert_eq!(read_str(KEY_PROVENANCE_UPSTREAM_URL), UPSTREAM_URL);
        assert_eq!(
            read_str(chunks::KEY_PROVENANCE_LICENSE),
            DEFAULT_LICENSE_SPDX
        );
        assert_eq!(
            read_str(chunks::KEY_PROVENANCE_WEIGHT_LICENSE),
            LicenseClass::AttributionRequired.as_str(),
            "cc-by-4.0 must resolve to AttributionRequired"
        );

        let _ = std::fs::remove_file(&inp);
        let _ = std::fs::remove_file(&outp);
    }

    #[test]
    fn bf16_tensor_passes_through_verbatim() {
        let inp = tmp_path("bf16-in");
        let outp = tmp_path("bf16-out");
        let values: [f32; 4] = [1.0, -0.5, 0.25, 8.0];
        let payload: Vec<u8> = values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        let st = safetensors_one(
            "teacher.encoder.blocks.0.attn.qkv.weight",
            "BF16",
            &[2, 2],
            &payload,
        );
        std::fs::write(&inp, &st).unwrap();

        let r = convert_atst_file(&inp, &outp, None).expect("convert BF16");
        assert_eq!(r.bf16_passthrough, 1);

        let out_bytes = std::fs::read(&outp).unwrap();
        let file = GgufFile::parse(out_bytes).expect("parse GGUF");
        let info = file
            .tensor_info("teacher.encoder.blocks.0.attn.qkv.weight")
            .expect("BF16 tensor present");
        assert_eq!(info.dtype, GgmlType::BF16);
        assert_eq!(file.tensor_bytes(info), payload.as_slice());

        let _ = std::fs::remove_file(&inp);
        let _ = std::fs::remove_file(&outp);
    }

    #[test]
    fn license_override_to_permissive_flips_class() {
        let inp = tmp_path("lic-in");
        let outp = tmp_path("lic-out");
        let payload: Vec<u8> = [1.0_f32].iter().flat_map(|v| v.to_le_bytes()).collect();
        let st = safetensors_one("x", "F32", &[1], &payload);
        std::fs::write(&inp, &st).unwrap();

        // A caller with a different license attestation escapes the
        // AttributionRequired default — mirror of the audiogen_medium /
        // musicgen_medium escape hatch.
        convert_atst_file(&inp, &outp, Some("mit")).expect("convert with override");
        let g = GgufFile::open(&outp).unwrap();
        assert_eq!(
            g.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some("mit"),
        );
        assert_eq!(
            g.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::Permissive.as_str()),
        );

        let _ = std::fs::remove_file(&inp);
        let _ = std::fs::remove_file(&outp);
    }

    // -----------------------------------------------------------------------
    // `vokra.atst.*` topology chunk group
    // -----------------------------------------------------------------------

    /// Converts a one-tensor fixture and hands back the parsed GGUF, so the
    /// topology assertions below do not each re-implement the setup.
    fn convert_minimal(tag: &str) -> GgufFile {
        let inp = tmp_path(&format!("{tag}-in"));
        let outp = tmp_path(&format!("{tag}-out"));
        let payload: Vec<u8> = [0.0_f32].iter().flat_map(|v| v.to_le_bytes()).collect();
        let st = safetensors_one("dummy", "F32", &[1], &payload);
        std::fs::write(&inp, &st).unwrap();

        convert_atst_file(&inp, &outp, None).expect("convert minimal fixture");
        let g = GgufFile::open(&outp).expect("parse output GGUF");

        let _ = std::fs::remove_file(&inp);
        let _ = std::fs::remove_file(&outp);
        g
    }

    #[test]
    fn topology_scalar_axes_round_trip() {
        let g = convert_minimal("topology-scalars");
        let u = |k: &str| -> Option<u64> { g.get(k).and_then(|v| v.as_u64()) };

        // Transformer topology (AST_base defaults).
        assert_eq!(u(KEY_ATST_EMBED_DIM), Some(u64::from(EMBED_DIM)));
        assert_eq!(u(KEY_ATST_DEPTH), Some(u64::from(DEPTH)));
        assert_eq!(u(KEY_ATST_NUM_HEADS), Some(u64::from(NUM_HEADS)));
        assert_eq!(
            u(KEY_ATST_MLP_RATIO_SCALED_1E3),
            Some(u64::from(MLP_RATIO_SCALED_1E3))
        );
        assert_eq!(u(KEY_ATST_MLP_HIDDEN_DIM), Some(u64::from(MLP_HIDDEN_DIM)));
        assert_eq!(u(KEY_ATST_IN_CHANS), Some(u64::from(IN_CHANS)));
        assert_eq!(u(KEY_ATST_NUM_CLASSES), Some(u64::from(NUM_CLASSES)));
        assert_eq!(
            u(KEY_ATST_DROP_PATH_RATE_SCALED_1E3),
            Some(u64::from(DROP_PATH_RATE_SCALED_1E3))
        );

        // Patch grid / position table.
        assert_eq!(u(KEY_ATST_PATCH_H), Some(u64::from(PATCH_H)));
        assert_eq!(u(KEY_ATST_PATCH_W), Some(u64::from(PATCH_W)));
        assert_eq!(u(KEY_ATST_SPEC_H), Some(u64::from(SPEC_H)));
        assert_eq!(u(KEY_ATST_SPEC_W), Some(u64::from(SPEC_W)));
        assert_eq!(u(KEY_ATST_NUM_PATCHES), Some(u64::from(NUM_PATCHES)));
        assert_eq!(u(KEY_ATST_POS_EMBED_LEN), Some(u64::from(POS_EMBED_LEN)));
        assert_eq!(
            u(KEY_ATST_PATCH_EMBED_IN_FEATURES),
            Some(u64::from(PATCH_EMBED_IN_FEATURES))
        );

        // Log-mel front-end.
        assert_eq!(u(KEY_ATST_SAMPLE_RATE), Some(u64::from(SAMPLE_RATE)));
        assert_eq!(u(KEY_ATST_N_FFT), Some(u64::from(N_FFT)));
        assert_eq!(u(KEY_ATST_HOP_LENGTH), Some(u64::from(HOP_LENGTH)));
        assert_eq!(u(KEY_ATST_WIN_LENGTH), Some(u64::from(WIN_LENGTH)));
        assert_eq!(u(KEY_ATST_N_MELS), Some(u64::from(N_MELS)));
        assert_eq!(u(KEY_ATST_F_MIN), Some(u64::from(F_MIN)));
        assert_eq!(u(KEY_ATST_F_MAX), Some(u64::from(F_MAX)));
        assert_eq!(
            u(KEY_ATST_AMP_TO_DB_TOP_DB),
            Some(u64::from(AMP_TO_DB_TOP_DB))
        );
    }

    #[test]
    fn topology_float_bool_and_string_axes_round_trip() {
        let g = convert_minimal("topology-nonint");

        // f32-typed axes widen to f64 exactly on read (`f64::from(f32)` is
        // lossless), so an exact compare is correct here — no tolerance.
        assert_eq!(
            g.get(KEY_ATST_LAYER_NORM_EPS).and_then(|v| v.as_f64()),
            Some(f64::from(LAYER_NORM_EPS))
        );
        assert_eq!(
            g.get(KEY_ATST_NORM_MIN).and_then(|v| v.as_f64()),
            Some(f64::from(NORM_MIN))
        );
        assert_eq!(
            g.get(KEY_ATST_NORM_MAX).and_then(|v| v.as_f64()),
            Some(f64::from(NORM_MAX))
        );

        // Bool-typed axes must stay Bool — an int encoding would read back
        // `None` here and silently look "absent" to the binder.
        assert_eq!(
            g.get(KEY_ATST_QKV_BIAS).and_then(|v| v.as_bool()),
            Some(QKV_BIAS),
            "qkv_bias=False decides whether a bias tensor exists at all"
        );
        assert_eq!(
            g.get(KEY_ATST_USE_CLS).and_then(|v| v.as_bool()),
            Some(USE_CLS)
        );

        // String-typed axes.
        assert_eq!(
            g.get(KEY_ATST_POS_TYPE).and_then(|v| v.as_str()),
            Some(POS_TYPE)
        );
        assert_eq!(
            g.get(KEY_ATST_ACT_LAYER).and_then(|v| v.as_str()),
            Some(ACT_LAYER)
        );
        assert_eq!(
            g.get(KEY_ATST_PATCH_EMBED_KIND).and_then(|v| v.as_str()),
            Some(PATCH_EMBED_KIND),
            "a Conv2d reading of the patch stem would look for a 4-D weight"
        );
        assert_eq!(
            g.get(KEY_ATST_PATCH_ORDER).and_then(|v| v.as_str()),
            Some(PATCH_ORDER),
            "an `hw` reading yields a silently transposed sequence"
        );
    }

    #[test]
    fn patch_grid_axis_array_round_trips_in_order() {
        let g = convert_minimal("topology-array");
        for (i, &expected) in PATCH_GRID.iter().enumerate() {
            let k = format!("{KEY_ATST_PATCH_GRID_PREFIX}_{i}");
            assert_eq!(
                g.get(&k).and_then(|v| v.as_u64()),
                Some(u64::from(expected)),
                "patch_grid[{i}] mismatch"
            );
        }
        // The array must be exactly as long as it claims — an unstamped index
        // 2 proves nothing spilled past `[rows, cols]`.
        let overrun = format!("{KEY_ATST_PATCH_GRID_PREFIX}_{}", PATCH_GRID.len());
        assert!(
            g.get(&overrun).is_none(),
            "patch_grid must stamp exactly {} entries",
            PATCH_GRID.len()
        );
    }

    /// The stamped values must equal the constants transcribed from upstream —
    /// pinned as literals so an accidental edit to a constant fails here
    /// instead of silently re-shaping every future artifact.
    ///
    /// Sources (fetched raw from
    /// `raw.githubusercontent.com/Audio-WestlakeU/audiossl/main/...`,
    /// 2026-08-15) are named per group in the constants block above.
    #[test]
    #[allow(clippy::assertions_on_constants)] // Compile-time drift guards are intentional.
    fn topology_constants_match_transcribed_upstream_values() {
        // `AST_base(patch_h=64,patch_w=4,...)` → embed_dim / depth / num_heads.
        assert_eq!(EMBED_DIM, 768);
        assert_eq!(DEPTH, 12);
        assert_eq!(NUM_HEADS, 12);
        assert_eq!(PATCH_H, 64);
        assert_eq!(PATCH_W, 4);
        const { assert!(!QKV_BIAS, "AST_base passes qkv_bias=False") };

        // `AST.__init__` defaults reached because `**kwargs` is empty.
        assert_eq!(SPEC_H, 64);
        assert_eq!(SPEC_W, 1001);
        assert_eq!(MLP_RATIO_SCALED_1E3, 4_000, "mlp_ratio=4.");
        assert_eq!(IN_CHANS, 1);
        assert_eq!(NUM_CLASSES, 0, "no task head ships with the checkpoint");
        const { assert!(USE_CLS, "use_cls=True") };
        assert_eq!(DROP_PATH_RATE_SCALED_1E3, 100, "drop_path_rate=0.1");
        assert_eq!(POS_TYPE, "cut");
        assert_eq!(ACT_LAYER, "gelu");
        assert_eq!(PATCH_EMBED_KIND, "linear");
        assert_eq!(PATCH_ORDER, "wh");
        assert_eq!(LAYER_NORM_EPS, 1e-6, "partial(nn.LayerNorm, eps=1e-6)");

        // `transform.py` front-end.
        assert_eq!(SAMPLE_RATE, 16_000);
        assert_eq!(N_FFT, 1024);
        assert_eq!(HOP_LENGTH, 160);
        assert_eq!(WIN_LENGTH, 1024);
        assert_eq!(N_MELS, 64);
        assert_eq!(F_MIN, 60);
        assert_eq!(F_MAX, 7800);
        assert_eq!(AMP_TO_DB_TOP_DB, 80);
        assert_eq!(AMP_TO_DB_STYPE, "power");
        assert_eq!(NORM_MIN, -79.6482_f32);
        assert_eq!(NORM_MAX, 50.6842_f32);

        // Derived axes must equal the upstream formulas applied to the above:
        //   get_num_patches -> (spec_h // patch_h) * (spec_w // patch_w)
        //   pos_embed       -> torch.zeros(1, num_patches + 1, embed_dim)
        //   Mlp hidden      -> int(dim * mlp_ratio)
        //   PatchEmbed_v2   -> nn.Linear(patch_h * patch_w, embed_dim)
        assert_eq!(PATCH_GRID_H, 1, "64 // 64");
        assert_eq!(PATCH_GRID_W, 250, "1001 // 4 truncates");
        assert_eq!(PATCH_GRID, [1, 250]);
        assert_eq!(NUM_PATCHES, 250);
        assert_eq!(POS_EMBED_LEN, 251, "num_patches + 1 CLS slot");
        assert_eq!(MLP_HIDDEN_DIM, 3072, "768 * 4.0");
        assert_eq!(PATCH_EMBED_IN_FEATURES, 256, "64 * 4");

        // Cross-check across the two independent source files: the mel-plane
        // height the encoder is built for must equal the band count the
        // front-end produces, or the patch stem cannot tile the input.
        assert_eq!(
            SPEC_H, N_MELS,
            "audio_transformer.py spec_h and transform.py n_mels must agree"
        );
        assert_eq!(
            PATCH_H, N_MELS,
            "one patch spans the full mel range, so the grid is one row tall"
        );
        assert_eq!(WIN_LENGTH, N_FFT, "transform.py sets both to 1024");
    }

    /// Adding the topology group must be **purely additive**: the arch, name,
    /// category, provenance and licence stamps a pre-existing consumer already
    /// reads must be untouched, and the tensor payload must stay byte-exact.
    #[test]
    fn topology_group_is_additive_and_leaves_prior_stamps_intact() {
        let inp = tmp_path("additive-in");
        let outp = tmp_path("additive-out");
        let values: [f32; 4] = [1.0, -0.5, 0.25, 8.0];
        let payload: Vec<u8> = values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        let st = safetensors_one(
            "teacher.encoder.blocks.0.attn.qkv.weight",
            "BF16",
            &[2, 2],
            &payload,
        );
        std::fs::write(&inp, &st).unwrap();

        convert_atst_file(&inp, &outp, None).expect("convert");
        let g = GgufFile::open(&outp).unwrap();
        let s = |k: &str| -> Option<&str> { g.get(k).and_then(|v| v.as_str()) };

        // Pre-existing stamps, unchanged.
        assert_eq!(s(chunks::KEY_MODEL_ARCH), Some(ARCH));
        assert_eq!(s(chunks::KEY_MODEL_NAME), Some(NAME));
        assert_eq!(s(KEY_MODEL_CATEGORY), Some(CATEGORY));
        assert_eq!(s(KEY_PROVENANCE_UPSTREAM_URL), Some(UPSTREAM_URL));
        assert_eq!(
            s(chunks::KEY_PROVENANCE_LICENSE),
            Some(DEFAULT_LICENSE_SPDX)
        );
        assert_eq!(
            s(chunks::KEY_PROVENANCE_WEIGHT_LICENSE),
            Some(LicenseClass::AttributionRequired.as_str())
        );

        // The topology group must not have leaked into the `vokra.model.*` or
        // `vokra.provenance.*` namespaces.
        assert!(
            KEY_ATST_EMBED_DIM.starts_with("vokra.atst."),
            "topology keys live under their own arch prefix"
        );

        // Tensor payload untouched by the metadata change.
        let info = g
            .tensor_info("teacher.encoder.blocks.0.attn.qkv.weight")
            .expect("BF16 tensor present");
        assert_eq!(info.dtype, GgmlType::BF16);
        assert_eq!(g.tensor_bytes(info), payload.as_slice());

        let _ = std::fs::remove_file(&inp);
        let _ = std::fs::remove_file(&outp);
    }

    /// The licence override path must still stamp the full topology group —
    /// a caller re-licensing an artifact must not silently lose its shape.
    #[test]
    fn topology_group_survives_license_override() {
        let inp = tmp_path("override-topology-in");
        let outp = tmp_path("override-topology-out");
        let payload: Vec<u8> = [1.0_f32].iter().flat_map(|v| v.to_le_bytes()).collect();
        let st = safetensors_one("x", "F32", &[1], &payload);
        std::fs::write(&inp, &st).unwrap();

        convert_atst_file(&inp, &outp, Some("mit")).expect("convert with override");
        let g = GgufFile::open(&outp).unwrap();

        assert_eq!(
            g.get(KEY_ATST_EMBED_DIM).and_then(|v| v.as_u64()),
            Some(u64::from(EMBED_DIM))
        );
        assert_eq!(
            g.get(KEY_ATST_N_MELS).and_then(|v| v.as_u64()),
            Some(u64::from(N_MELS))
        );
        assert_eq!(
            g.get(KEY_ATST_PATCH_ORDER).and_then(|v| v.as_str()),
            Some(PATCH_ORDER)
        );

        let _ = std::fs::remove_file(&inp);
        let _ = std::fs::remove_file(&outp);
    }
}
