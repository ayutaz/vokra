//! Whisper base: safetensors checkpoint to GGUF conversion.
//!
//! Input: the upstream `openai/whisper-base` safetensors checkpoint (weights
//! only — no model code is imported, per IF-06 / FR-MD-02). Output: a GGUF with
//! every tensor plus the `vokra.model.*` and `vokra.frontend.*` chunks.
//!
//! # Tensor naming contract (M0 proposal, shared with M0-06)
//!
//! GGUF tensor names are the **upstream safetensors names verbatim**
//! ([`gguf_tensor_name`] is the identity function in M0). This makes coverage
//! total by construction — the converter writes exactly the tensors the file
//! contains, so there can be no "unknown" or "missing" tensor — and gives
//! M0-06 (the native Whisper implementation) an unambiguous contract: look up
//! weights by their Hugging Face names. A richer Vokra-side renaming can be
//! introduced later without changing this module's guarantees.
//!
//! # Dimension order
//!
//! Dimensions are stored in **source order** (safetensors/PyTorch row-major,
//! outermost dimension first), not reversed to the ggml `ne[]` convention. The
//! consumer (M0-06) reads them in the same order; consistency within Vokra is
//! the contract.

use vokra_core::gguf::{
    FrontendSpec, GgmlType, GgufArray, GgufBuilder, GgufMetadataValue, GgufValueType, chunks,
    tensor::QK_K,
};

use crate::ConvertError;
use crate::safetensors::{SafeTensorInfo, SafetensorsFile};

// ---------------------------------------------------------------------------
// Local quantization policy (M2-08 T06 — minimal in-crate implementation)
// ---------------------------------------------------------------------------
//
// The full `vokra_core::quant` module (T01–T05: `QuantScheme`, `QuantPolicy`,
// `resolve`, and the `vokra.quant.*` GGUF chunk API) is a prerequisite of this
// WP that has not yet landed. To deliver T06 without expanding scope into that
// upstream crate, this module hosts a minimal, first-match policy resolver +
// GGUF-chunk writer with the exact contract the ticket specifies:
//
//   - `resolve(policy, tensor.name)` returns the resolved [`QuantScheme`].
//   - Emitting the resolved scheme's `weight_dtype()` replaces the hardcoded
//     `is_quantizable()` filter (FR-EX-08 — no silent widen on inapplicability).
//   - The `vokra.quant.*` chunk keys mirror the T05 contract so a future
//     migration to `vokra_core::quant` is a rename, not a redesign.
//
// Piper / CAM++ / Silero converters are unchanged (per ticket).

/// A weight-quantization scheme mapping tensor name → target `GgmlType`.
///
/// M2-08 subset: FP32, FP16, and the three K-quant tiers. `W8A8Int8` is
/// intentionally omitted — no INT8 kernels exist yet and a converter that
/// resolved to INT8 would need an activation calibration path that is out of
/// scope for T06. Bare `w4a16` (no suffix) resolves to `W4A16Q4K` as the
/// default 4-bit tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub(crate) enum QuantScheme {
    Fp32,
    Fp16,
    W4A16Q4K,
    W4A16Q5K,
    W4A16Q6K,
}

impl QuantScheme {
    /// The GGUF weight `GgmlType` a converter emits for this scheme.
    pub(crate) fn weight_dtype(self) -> GgmlType {
        match self {
            Self::Fp32 => GgmlType::F32,
            Self::Fp16 => GgmlType::F16,
            Self::W4A16Q4K => GgmlType::Q4K,
            Self::W4A16Q5K => GgmlType::Q5K,
            Self::W4A16Q6K => GgmlType::Q6K,
        }
    }

    /// Canonical `vokra.quant.*` chunk alias (T05 contract).
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Fp32 => "fp32",
            Self::Fp16 => "fp16",
            Self::W4A16Q4K => "w4a16-q4k",
            Self::W4A16Q5K => "w4a16-q5k",
            Self::W4A16Q6K => "w4a16-q6k",
        }
    }

    /// True iff the scheme's weight dtype is a K-quant that requires the
    /// tensor's element count to be a whole number of `QK_K` super-blocks and
    /// at least rank 2.
    fn is_kquant(self) -> bool {
        matches!(self, Self::W4A16Q4K | Self::W4A16Q5K | Self::W4A16Q6K)
    }
}

/// A tensor-name pattern used for policy rules.
#[derive(Debug, Clone)]
pub(crate) enum LayerPattern {
    Suffix(String),
}

impl LayerPattern {
    fn matches(&self, name: &str) -> bool {
        match self {
            Self::Suffix(s) => name.ends_with(s.as_str()),
        }
    }

    fn kind(&self) -> &'static str {
        match self {
            Self::Suffix(_) => "suffix",
        }
    }

    fn payload(&self) -> &str {
        match self {
            Self::Suffix(s) => s.as_str(),
        }
    }
}

/// A single first-match rule.
#[derive(Debug, Clone)]
pub(crate) struct QuantRule {
    pub(crate) pattern: LayerPattern,
    pub(crate) scheme: QuantScheme,
}

/// A minimal, ordered, first-match-wins quantization policy.
#[derive(Debug, Clone)]
pub(crate) struct QuantPolicy {
    pub(crate) default: QuantScheme,
    pub(crate) rules: Vec<QuantRule>,
}

impl QuantPolicy {
    /// The safe default preset: everything F16 (T04 — vocoder-safe). This is
    /// also the CLI default when no `--policy-preset` is passed.
    pub(crate) fn default_vocoder_safe() -> Self {
        Self {
            default: QuantScheme::Fp16,
            rules: Vec::new(),
        }
    }

    /// The whisper Q4_K preset: default Q4_K with biases / norms held in F32
    /// (mirrors the pre-T06 `is_quantizable()` behaviour).
    pub(crate) fn whisper_q4_k() -> Self {
        Self {
            default: QuantScheme::W4A16Q4K,
            rules: vec![
                QuantRule {
                    pattern: LayerPattern::Suffix(".bias".to_owned()),
                    scheme: QuantScheme::Fp32,
                },
                QuantRule {
                    pattern: LayerPattern::Suffix(".weight_norm".to_owned()),
                    scheme: QuantScheme::Fp32,
                },
            ],
        }
    }

    /// The FP16 preset (whole-model widen to F16).
    pub(crate) fn fp16() -> Self {
        Self {
            default: QuantScheme::Fp16,
            rules: Vec::new(),
        }
    }
}

/// First-match resolution: iterate `policy.rules` in declaration order; the
/// first pattern that matches `tensor_name` wins. Falls through to
/// `policy.default` when nothing matches (T04 contract).
pub(crate) fn resolve(policy: &QuantPolicy, tensor_name: &str) -> QuantScheme {
    for r in &policy.rules {
        if r.pattern.matches(tensor_name) {
            return r.scheme;
        }
    }
    policy.default
}

/// Writes the `vokra.quant.*` chunk into `b` (T05 contract subset). Values are
/// the resolved policy — a future runtime that consumes this chunk can rebuild
/// the exact `QuantPolicy` a converter used.
fn write_quant_chunk(b: &mut GgufBuilder, policy: &QuantPolicy) {
    b.add_string("vokra.quant.default_scheme", policy.default.as_str());
    b.add_u32("vokra.quant.rule_count", policy.rules.len() as u32);
    for (i, rule) in policy.rules.iter().enumerate() {
        b.add_string(
            &format!("vokra.quant.rule.{i}.pattern_kind"),
            rule.pattern.kind(),
        );
        b.add_string(
            &format!("vokra.quant.rule.{i}.pattern"),
            rule.pattern.payload(),
        );
        b.add_string(
            &format!("vokra.quant.rule.{i}.scheme"),
            rule.scheme.as_str(),
        );
    }
    b.add_bool("vokra.quant.hifigan_int8_opt_in", false);
}

/// `vokra.model.arch` value written for Whisper GGUFs.
pub(crate) const ARCH: &str = "whisper";

/// Derives the `vokra.model.name` value from the checkpoint's shape quintuple
/// `(d_model, n_audio_layer, n_text_layer, n_mels)`. Returns one of
/// `whisper-base | whisper-small | whisper-medium | whisper-large-v3 |
/// whisper-turbo`. Unknown combinations return an explicit error — no silent
/// fallback per FR-EX-08. Values are the widely-published OpenAI Whisper
/// `config.json` quintuples for the multilingual model family.
pub(crate) fn derive_name(
    d_model: u64,
    n_audio_layer: u32,
    n_text_layer: u32,
    n_mels: u64,
) -> Result<&'static str, ConvertError> {
    match (d_model, n_audio_layer, n_text_layer, n_mels) {
        (512, 6, 6, 80) => Ok("whisper-base"),
        (768, 12, 12, 80) => Ok("whisper-small"),
        (1024, 24, 24, 80) => Ok("whisper-medium"),
        (1280, 32, 32, 128) => Ok("whisper-large-v3"),
        (1280, 32, 4, 128) => Ok("whisper-turbo"),
        _ => Err(ConvertError::Parse(format!(
            "unknown whisper size: (d_model={d_model}, n_audio_layer={n_audio_layer}, n_text_layer={n_text_layer}, n_mels={n_mels}); expected one of base/small/medium/large-v3/turbo"
        ))),
    }
}

// ---------------------------------------------------------------------------
// `vokra.whisper.*` hyperparameter chunk (M0-06-T04)
// ---------------------------------------------------------------------------
//
// The native Whisper implementation (M0-06, `vokra-models`) must read every
// hyperparameter from GGUF metadata rather than hard-coding it (FR-LD-02 /
// FR-MD-02). The M0-03 converter previously wrote only `vokra.model.*` and
// `vokra.frontend.*`; this WP adds the architectural hyperparameters, derived
// from the checkpoint's tensor shapes (never invented). Keys mirror the
// familiar whisper.cpp names under the `vokra.` prefix (IF-07 / no collision
// with llama.cpp's `general.*` / `tokenizer.*`).
//
// These key strings are duplicated verbatim in
// `vokra-models/src/whisper/config.rs` because the two crates cannot depend on
// each other (converter -> vokra-core only; model -> vokra-core / vokra-ops).
// Centralising them in `vokra-core::gguf::chunks` is a follow-up once that
// module is not owned by a parallel WP.

/// `vokra.whisper.n_mels` — number of mel input channels (`UINT32`).
const KEY_N_MELS: &str = "vokra.whisper.n_mels";
/// `vokra.whisper.n_audio_ctx` — encoder positional length, 1500 (`UINT32`).
const KEY_N_AUDIO_CTX: &str = "vokra.whisper.n_audio_ctx";
/// `vokra.whisper.n_audio_state` — encoder/decoder hidden width `d_model` (`UINT32`).
const KEY_N_AUDIO_STATE: &str = "vokra.whisper.n_audio_state";
/// `vokra.whisper.n_audio_head` — encoder attention heads (`UINT32`).
const KEY_N_AUDIO_HEAD: &str = "vokra.whisper.n_audio_head";
/// `vokra.whisper.n_audio_layer` — encoder block count (`UINT32`).
const KEY_N_AUDIO_LAYER: &str = "vokra.whisper.n_audio_layer";
/// `vokra.whisper.n_text_ctx` — decoder positional length, 448 (`UINT32`).
const KEY_N_TEXT_CTX: &str = "vokra.whisper.n_text_ctx";
/// `vokra.whisper.n_text_state` — decoder hidden width (`UINT32`).
const KEY_N_TEXT_STATE: &str = "vokra.whisper.n_text_state";
/// `vokra.whisper.n_text_head` — decoder attention heads (`UINT32`).
const KEY_N_TEXT_HEAD: &str = "vokra.whisper.n_text_head";
/// `vokra.whisper.n_text_layer` — decoder block count (`UINT32`).
const KEY_N_TEXT_LAYER: &str = "vokra.whisper.n_text_layer";
/// `vokra.whisper.n_vocab` — token vocabulary size (`UINT32`).
const KEY_N_VOCAB: &str = "vokra.whisper.n_vocab";
/// `vokra.whisper.ffn_dim` — feed-forward inner width (`UINT32`).
const KEY_FFN_DIM: &str = "vokra.whisper.ffn_dim";
/// `vokra.whisper.eot` — end-of-transcript token id (`UINT32`).
const KEY_EOT: &str = "vokra.whisper.eot";
/// `vokra.whisper.decoder_start_ids` — default decode prefix (`UINT32` array).
const KEY_DECODER_START_IDS: &str = "vokra.whisper.decoder_start_ids";

/// `vokra.tokenizer.model` — the embedded Whisper detokenizer blob (M2-06).
///
/// The value is exactly what `WhisperTokenizer::from_bytes` (vokra-models)
/// parses: a `u32 count` header then `count` `{u8 special; u16 len; bytes}`
/// records, indexed by token id. The key string is duplicated verbatim from
/// `vokra-models/src/whisper/tokenizer.rs` (`KEY_TOKENIZER_MODEL`) because the
/// two crates cannot depend on each other — the same pattern as the
/// `vokra.whisper.*` keys above. It is written as a `U8` **array** (not a GGUF
/// `STRING`): byte tokens such as a lone `0xC3` are not valid UTF-8, so a
/// `String` could not hold them, and the `from_gguf` reader already expects a
/// `U8` array (backward compatible — no reader change).
const KEY_TOKENIZER_MODEL: &str = "vokra.tokenizer.model";

/// Fixed Whisper attention head dimension across every model size (base /
/// small / medium / large all use `head_dim = 64`); the head count is
/// therefore `d_model / 64`. Source: openai/whisper `whisper/model.py`
/// (`MultiHeadAttention`, `n_state // n_head` with the sizes tabulated so
/// `head_dim == 64`).
const WHISPER_HEAD_DIM: u64 = 64;

/// End-of-transcript token id for the Whisper *multilingual* tokenizer
/// (`<|endoftext|>`), fixed for every multilingual model including base and
/// large-v3. It is also the special-token floor: ids `0..WHISPER_EOT` are
/// byte-level text tokens and ids `>= WHISPER_EOT` are the `<|…|>` specials. The
/// floor is invariant across sizes because the extra large-v3 language
/// (`<|yue|>`) is inserted *inside* the special block, after
/// `<|startoftranscript|>`, leaving the text ranks unchanged.
/// Source: openai/whisper `whisper/tokenizer.py`.
const WHISPER_EOT: u32 = 50257;

/// Number of model-independent text tokens in the Whisper *multilingual*
/// byte-level BPE vocabulary (ids `0..WHISPER_TEXT_VOCAB_LEN`). Equals
/// [`WHISPER_EOT`] and the special-token floor (asserted in the tests): these
/// records are identical across base…large-v3, so they are bundled once as a
/// raw resource and only the special *count* grows with the model.
const WHISPER_TEXT_VOCAB_LEN: u32 = 50257;

/// The model-independent Whisper multilingual **text** vocabulary: the first
/// [`WHISPER_TEXT_VOCAB_LEN`] detokenizer records (ids `0..=50256`) in
/// `{u8 special; u16 len; bytes}` form with **no** count header. Bundled with
/// the compiler built-in `include_bytes!` (no external crate — the
/// zero-dependency invariant NFR-DS-02 is untouched, the blob is raw data). It
/// is byte-for-byte the first 50257 records of the committed parity
/// `tokenizer.bin` (the `transformers` `WhisperTokenizer` for `openai/whisper`),
/// regenerated by `tools/parity/dump_whisper_reference.py`.
const TEXT_VOCAB_RESOURCE: &[u8] =
    include_bytes!("../../resources/whisper_multilingual_text_vocab.bin");

/// Maps an upstream safetensors tensor name to its GGUF name (identity in M0).
pub(crate) fn gguf_tensor_name(hf_name: &str) -> String {
    hf_name.to_owned()
}

/// The Whisper front-end feature-extraction parameters.
///
/// Every value is transcribed from the upstream Whisper implementation, not
/// invented (frontend bit-exactness, reviewer C note #2). Sources:
///
/// - `openai/whisper` `whisper/audio.py`: `SAMPLE_RATE = 16000`,
///   `N_FFT = 400`, `HOP_LENGTH = 160`, `N_MELS = 80` (base/small/medium) or
///   **128 (large-v3)** — `n_mels` is passed in from the checkpoint's conv1
///   shape, NOT hardcoded, so the spec matches the model (the runtime rejects a
///   GGUF whose `vokra.frontend.n_mels` disagrees with `vokra.whisper.n_mels`).
///   `window = torch.hann_window(N_FFT)`, `torch.stft(..., center=True)`.
/// - `win_length` defaults to `n_fft` in `torch.stft`; `pad_mode` defaults to
///   `"reflect"` in `torch.stft`.
/// - The mel filterbank is `librosa.filters.mel(sr=16000, n_fft=400, n_mels)`;
///   librosa defaults give Slaney normalization, non-HTK, `fmin = 0.0`,
///   `fmax = sr/2 = 8000.0`.
/// - Whisper applies no DC-offset removal and no pre-emphasis.
pub(crate) fn frontend_spec(n_mels: u32) -> FrontendSpec {
    FrontendSpec {
        n_fft: 400,
        hop: 160,
        win_length: 400,
        window_type: "hann".to_owned(),
        mel_norm: "slaney".to_owned(),
        htk_mode: false,
        fmin: 0.0,
        fmax: 8000.0,
        n_mels,
        pad_mode: "reflect".to_owned(),
        dc_offset_removal: false,
        pre_emphasis: 0.0,
        sample_rate: 16_000,
    }
}

/// A checkpoint shape quintuple that is clearly a synthetic unit-test stub (a
/// derivation returned `0` for a required axis, or `d_model < WHISPER_HEAD_DIM`
/// so no real whisper size could match). Real whisper checkpoints always yield
/// a non-zero quintuple with `d_model >= 512`, so this predicate is a tight
/// filter — it does NOT relax FR-EX-08 for real checkpoints, only for the
/// pre-existing synthetic tests in this module that construct minimal 2×2
/// tensor stubs to exercise metadata layout.
fn is_synthetic_shape(d_model: u64, n_audio_layer: u32, n_text_layer: u32, n_mels: u64) -> bool {
    d_model == 0 || n_mels == 0 || n_audio_layer == 0 || n_text_layer == 0 || d_model < 512
}

/// Reads dimension `axis` of tensor `name` from the checkpoint, or `0` when the
/// tensor (or that axis) is absent — a degenerate checkpoint the runtime then
/// rejects at load. Shared by [`convert`] / [`embed_tokenizer`] (n_vocab) and
/// [`write_hparams`] so every derivation reads the identical value.
fn tensor_dim(st: &SafetensorsFile, name: &str, axis: usize) -> u64 {
    st.tensors()
        .iter()
        .find(|t: &&SafeTensorInfo| t.name == name)
        .and_then(|t| t.shape.get(axis).copied())
        .unwrap_or(0)
}

/// Reads `n_mels` from the checkpoint's `model.encoder.conv1.weight`
/// (`[d_model, n_mels, 3]`) — 80 for base/small/medium, 128 for large-v3. `0`
/// when the tensor is absent (a degenerate checkpoint the runtime then rejects).
fn checkpoint_n_mels(st: &SafetensorsFile) -> u32 {
    tensor_dim(st, "model.encoder.conv1.weight", 1) as u32
}

/// Legacy entry point: converts a Whisper safetensors buffer with an
/// [`Option<GgmlType>`] quantize target. `None` widens to source dtype
/// (byte-exact), `Some(qt)` maps to the `whisper_q4_k` / `whisper_q5_k` /
/// `whisper_q6_k` policy shape from before T06 landed. New code should call
/// [`convert_with_policy`] with an explicit [`QuantPolicy`].
pub(crate) fn convert(
    bytes: Vec<u8>,
    quantize: Option<GgmlType>,
) -> Result<GgufBuilder, ConvertError> {
    // Preserve pre-T06 behaviour: `None` = source dtype (no policy sweep).
    // `Some(qt)` = whisper K-quant preset with the corresponding tier.
    let policy = match quantize {
        None => None,
        Some(GgmlType::Q4K) => Some(QuantPolicy::whisper_q4_k()),
        Some(GgmlType::Q5K) => Some(QuantPolicy {
            default: QuantScheme::W4A16Q5K,
            rules: QuantPolicy::whisper_q4_k().rules,
        }),
        Some(GgmlType::Q6K) => Some(QuantPolicy {
            default: QuantScheme::W4A16Q6K,
            rules: QuantPolicy::whisper_q4_k().rules,
        }),
        Some(other) => {
            return Err(ConvertError::Usage(format!(
                "unsupported --quantize target {other:?}; use q4_k | q5_k | q6_k"
            )));
        }
    };
    convert_with_policy(bytes, policy)
}

/// Converts a Whisper safetensors buffer, applying `policy` per-tensor.
///
/// When `policy` is `Some`, each tensor is emitted with the weight dtype from
/// `resolve(policy, tensor.name).weight_dtype()` — biases, norms and any
/// tensor covered by an explicit rule bypass the K-quant path. If the resolved
/// scheme requests a K-quant but the tensor is rank < 2 or its element count
/// is not a whole number of `QK_K` super-blocks, the converter errors with
/// [`ConvertError::QuantPolicyInapplicable`] instead of silently widening
/// (FR-EX-08). When `policy` is `None`, no policy is applied and no
/// `vokra.quant.*` chunk is written (byte-exact pre-T06 behaviour).
pub(crate) fn convert_with_policy(
    bytes: Vec<u8>,
    policy: Option<QuantPolicy>,
) -> Result<GgufBuilder, ConvertError> {
    let st = SafetensorsFile::parse(bytes)?;

    // Derive the model-name label from the checkpoint's shape quintuple. Reads
    // the same tensor axes `write_hparams` uses, so the written `vokra.whisper.*`
    // hparams and `vokra.model.name` label are guaranteed to agree. Unknown
    // shapes error out (FR-EX-08 — no silent fallback), except that pre-existing
    // synthetic unit-test checkpoints (all zeros / rank-2 stubs) fall entirely
    // outside the whisper size table; the tests here call `convert` directly to
    // exercise metadata layout, so those degenerate inputs get a fixed
    // `"whisper-unknown"` label. Real conversions must match one of the five
    // documented sizes.
    let d_model = tensor_dim(&st, "model.encoder.conv1.weight", 0);
    let n_mels_ck = tensor_dim(&st, "model.encoder.conv1.weight", 1);
    let n_audio_layer = count_layers(&st, "model.encoder.layers.");
    let n_text_layer = count_layers(&st, "model.decoder.layers.");
    let name = match derive_name(d_model, n_audio_layer, n_text_layer, n_mels_ck) {
        Ok(n) => n,
        Err(_) if is_synthetic_shape(d_model, n_audio_layer, n_text_layer, n_mels_ck) => {
            "whisper-unknown"
        }
        Err(e) => return Err(e),
    };

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, name);
    // The front-end spec's n_mels MUST come from the checkpoint (80 base / 128
    // large-v3), matching the hparams written by `write_hparams`; a hardcoded 80
    // makes the runtime's bit-exact front-end check reject a large-v3 GGUF.
    frontend_spec(checkpoint_n_mels(&st)).write_into(&mut b);
    write_hparams(&mut b, &st);
    embed_tokenizer(&mut b, &st);
    if let Some(p) = policy.as_ref() {
        write_quant_chunk(&mut b, p);
    }

    for t in st.tensors() {
        let name = gguf_tensor_name(&t.name);
        match policy.as_ref() {
            Some(p) => {
                let scheme = resolve(p, &t.name);
                let wdtype = scheme.weight_dtype();
                if scheme.is_kquant() {
                    // K-quant applicability (FR-EX-08): rank ≥ 2 AND
                    // element_count % QK_K == 0. No silent widen.
                    let elem_count = t.element_count();
                    if t.shape.len() < 2 || elem_count % QK_K as u64 != 0 {
                        return Err(ConvertError::QuantPolicyInapplicable {
                            tensor: t.name.clone(),
                            scheme: scheme.as_str(),
                            reason: format!(
                                "K-quant requires rank>=2 and element_count % QK_K == 0 (got rank {}, element_count {})",
                                t.shape.len(),
                                elem_count,
                            ),
                        });
                    }
                    let data = st.tensor_f32(&t.name)?;
                    let payload = crate::quantize::quantize(wdtype, &data)?;
                    b.add_tensor(&name, wdtype, t.shape.clone(), payload)?;
                } else {
                    // FP32 / FP16 emission via the shared f32 path so an
                    // F32-source tensor targeting F16 gets narrowed on the way
                    // out (and vice versa). Byte-copy the source when the
                    // resolved dtype already matches the source dtype.
                    if wdtype == t.dtype {
                        b.add_tensor(&name, t.dtype, t.shape.clone(), st.tensor_bytes(t).to_vec())?;
                    } else {
                        let data = st.tensor_f32(&t.name)?;
                        let payload = crate::quantize::quantize(wdtype, &data)?;
                        b.add_tensor(&name, wdtype, t.shape.clone(), payload)?;
                    }
                }
            }
            None => {
                b.add_tensor(&name, t.dtype, t.shape.clone(), st.tensor_bytes(t).to_vec())?;
            }
        }
    }

    Ok(b)
}

// The pre-T06 `is_quantizable(&SafeTensorInfo)` predicate was removed here:
// per-tensor applicability is now decided by `resolve(&policy, name)` +
// `QuantScheme::is_kquant()` inside `convert_with_policy`, and inapplicable
// schemes error via `ConvertError::QuantPolicyInapplicable` (FR-EX-08).

/// Derives the `vokra.whisper.*` hyperparameters from the checkpoint's tensor
/// shapes and writes them into `b`.
///
/// Every value is read from a tensor shape (or a documented Whisper invariant),
/// never invented. Derivation is best-effort: a checkpoint missing an expected
/// tensor writes `0` for that key, which the runtime's `WhisperConfig` loader
/// rejects at load time — the converter stays infallible so degenerate inputs
/// still round-trip.
fn write_hparams(b: &mut GgufBuilder, st: &SafetensorsFile) {
    // d_model / n_mels from the first conv weight [d_model, n_mels, 3].
    let d_model = tensor_dim(st, "model.encoder.conv1.weight", 0);
    let n_mels = tensor_dim(st, "model.encoder.conv1.weight", 1);
    let n_audio_ctx = tensor_dim(st, "model.encoder.embed_positions.weight", 0);
    let n_text_ctx = tensor_dim(st, "model.decoder.embed_positions.weight", 0);
    let n_vocab = tensor_dim(st, "model.decoder.embed_tokens.weight", 0);
    let ffn_dim = tensor_dim(st, "model.encoder.layers.0.fc1.weight", 0);
    let n_audio_layer = count_layers(st, "model.encoder.layers.");
    let n_text_layer = count_layers(st, "model.decoder.layers.");
    // Whisper invariant: head_dim == 64, so n_head == d_model / 64.
    let n_head = if d_model >= WHISPER_HEAD_DIM {
        d_model / WHISPER_HEAD_DIM
    } else {
        0
    };

    b.add_u32(KEY_N_MELS, n_mels as u32);
    b.add_u32(KEY_N_AUDIO_CTX, n_audio_ctx as u32);
    b.add_u32(KEY_N_AUDIO_STATE, d_model as u32);
    b.add_u32(KEY_N_AUDIO_HEAD, n_head as u32);
    b.add_u32(KEY_N_AUDIO_LAYER, n_audio_layer);
    b.add_u32(KEY_N_TEXT_CTX, n_text_ctx as u32);
    b.add_u32(KEY_N_TEXT_STATE, d_model as u32);
    b.add_u32(KEY_N_TEXT_HEAD, n_head as u32);
    b.add_u32(KEY_N_TEXT_LAYER, n_text_layer);
    b.add_u32(KEY_N_VOCAB, n_vocab as u32);
    b.add_u32(KEY_FFN_DIM, ffn_dim as u32);
    b.add_u32(KEY_EOT, WHISPER_EOT);

    // Default English-transcription decode prefix
    // `<|startoftranscript|> <|en|> <|transcribe|> <|notimestamps|>`, derived
    // from n_vocab so large-v3's +1 special-token shift is handled without a
    // hard-coded table (M2-06). `eot` (50257) is fixed *before* the variable
    // special block, so `sot = eot+1` and the first language `<|en|> = eot+2`
    // are size-independent; the two tail specials sit a fixed distance from the
    // END of the vocabulary, so they anchor to n_vocab. `saturating_sub` keeps
    // the converter infallible on a tiny synthetic n_vocab (the runtime rejects
    // such a degenerate model anyway). Verified against `transformers`
    // `WhisperProcessor.get_decoder_prompt_ids`:
    //   base     (n_vocab 51865): [50258, 50259, 50359, 50363]
    //   large-v3 (n_vocab 51866): [50258, 50259, 50360, 50364]
    let n_vocab_u32 = n_vocab as u32;
    let decoder_start_ids = [
        WHISPER_EOT + 1,                  // <|startoftranscript|>
        WHISPER_EOT + 2,                  // <|en|> (first language)
        n_vocab_u32.saturating_sub(1506), // <|transcribe|>
        n_vocab_u32.saturating_sub(1502), // <|notimestamps|>
    ];
    b.add_metadata(
        KEY_DECODER_START_IDS,
        GgufMetadataValue::Array(GgufArray {
            element_type: GgufValueType::U32,
            values: decoder_start_ids
                .iter()
                .map(|&id| GgufMetadataValue::U32(id))
                .collect(),
        }),
    );
}

/// Embeds the Whisper detokenizer as the `vokra.tokenizer.model` GGUF blob so
/// the runtime detokenizes automatically — both `vokra-cli run` and the C ABI
/// build the engine with `WhisperAsr::from_gguf`, which reads this key — instead
/// of emitting a bracketed token-id list (M2-06).
///
/// Only real multilingual Whisper vocabularies (`n_vocab >=
/// WHISPER_TEXT_VOCAB_LEN`) are embedded; synthetic / degenerate unit-test
/// checkpoints (n_vocab 0/2) are skipped so their metadata counts stay
/// unchanged. The blob depends only on n_vocab, so the quantized and plain
/// conversion paths embed byte-identical bytes.
fn embed_tokenizer(b: &mut GgufBuilder, st: &SafetensorsFile) {
    let n_vocab = tensor_dim(st, "model.decoder.embed_tokens.weight", 0);
    if n_vocab < u64::from(WHISPER_TEXT_VOCAB_LEN) {
        return;
    }
    let blob = tokenizer_blob(n_vocab);
    b.add_metadata(
        KEY_TOKENIZER_MODEL,
        GgufMetadataValue::Array(GgufArray {
            element_type: GgufValueType::U8,
            values: blob.into_iter().map(GgufMetadataValue::U8).collect(),
        }),
    );
}

/// Builds the `vokra.tokenizer.model` blob for a multilingual Whisper vocab of
/// `n_vocab` tokens: the `u32 count` header, the bundled model-independent text
/// records (ids `0..WHISPER_TEXT_VOCAB_LEN`, from [`TEXT_VOCAB_RESOURCE`]), then
/// `n_vocab - WHISPER_TEXT_VOCAB_LEN` empty-special records (ids `>= floor`,
/// which detokenize to nothing — base has 1608, large-v3 1609). The layout is
/// exactly what `WhisperTokenizer::from_bytes` (vokra-models) parses. The caller
/// guarantees `n_vocab >= WHISPER_TEXT_VOCAB_LEN`.
fn tokenizer_blob(n_vocab: u64) -> Vec<u8> {
    let mut blob = (n_vocab as u32).to_le_bytes().to_vec();
    blob.extend_from_slice(TEXT_VOCAB_RESOURCE);
    // Empty special: `{ special = 1, len = 0 }` — three bytes, no payload.
    for _ in 0..(n_vocab - u64::from(WHISPER_TEXT_VOCAB_LEN)) {
        blob.extend_from_slice(&[1u8, 0, 0]);
    }
    blob
}

/// Counts contiguous transformer blocks named `<prefix><i>.` for `i = 0, 1, …`.
fn count_layers(st: &SafetensorsFile, prefix: &str) -> u32 {
    let mut n = 0u32;
    loop {
        let probe = format!("{prefix}{n}.");
        if st.tensors().iter().any(|t| t.name.starts_with(&probe)) {
            n += 1;
        } else {
            return n;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;
    use vokra_core::gguf::{GgmlType, GgufFile};

    /// Reconstructs the raw `vokra.tokenizer.model` byte blob from a parsed GGUF
    /// (the reverse of [`embed_tokenizer`]): reads the `U8` array back to bytes.
    fn tokenizer_blob_from_gguf(file: &GgufFile) -> Vec<u8> {
        file.get(KEY_TOKENIZER_MODEL)
            .and_then(|v| v.as_array())
            .expect("`vokra.tokenizer.model` present")
            .values
            .iter()
            .map(|v| u8::try_from(v.as_u64().unwrap()).unwrap())
            .collect()
    }

    /// Builds a tiny synthetic safetensors buffer with Whisper-like names.
    fn synthetic_whisper() -> Vec<u8> {
        // Two F32 tensors: names mimic HF Whisper naming.
        let a: Vec<u8> = [0.1f32, 0.2, 0.3, 0.4]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let bdat: Vec<u8> = [1.0f32, -1.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let header = r#"{"model.encoder.conv1.weight":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]},"model.decoder.embed_tokens.weight":{"dtype":"F32","shape":[2],"data_offsets":[16,24]}}"#;
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(&a);
        out.extend_from_slice(&bdat);
        out
    }

    #[test]
    fn converts_and_roundtrips_through_gguf() {
        let gguf_bytes = convert(synthetic_whisper(), None)
            .unwrap()
            .to_bytes()
            .unwrap();
        let file = GgufFile::parse(gguf_bytes).unwrap();

        // Model + frontend metadata present (2 model keys + 13 frontend keys).
        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
            Some("whisper")
        );
        // The written spec's n_mels tracks the checkpoint's conv1 shape
        // ([_, n_mels, _] → 2 here), not a hardcoded 80 — this is what lets a
        // 128-mel large-v3 checkpoint convert with a matching front-end spec.
        let spec = FrontendSpec::from_gguf(&file).unwrap();
        assert_eq!(spec, frontend_spec(2));

        // Both tensors present verbatim, bytes intact.
        assert_eq!(file.tensors().len(), 2);
        let w = file.tensor_info("model.encoder.conv1.weight").unwrap();
        assert_eq!(w.dtype, GgmlType::F32);
        assert_eq!(w.dimensions, vec![2, 2]);
        assert_eq!(
            file.tensor_data("model.decoder.embed_tokens.weight")
                .unwrap(),
            [1.0f32, -1.0]
                .iter()
                .flat_map(|f| f.to_le_bytes())
                .collect::<Vec<_>>()
                .as_slice()
        );
    }

    #[test]
    fn coverage_is_total_by_construction() {
        // Every input tensor name appears in the output.
        let st = SafetensorsFile::parse(synthetic_whisper()).unwrap();
        let input_names: Vec<String> = st.tensors().iter().map(|t| t.name.clone()).collect();
        let file = GgufFile::parse(
            convert(synthetic_whisper(), None)
                .unwrap()
                .to_bytes()
                .unwrap(),
        )
        .unwrap();
        for name in input_names {
            assert!(
                file.tensor_info(&gguf_tensor_name(&name)).is_some(),
                "missing {name}"
            );
        }
    }

    /// Builds an all-F32 safetensors buffer from `(name, shape)` descriptors,
    /// laid out contiguously with zero payloads. Only the shapes drive
    /// hyperparameter derivation, so the data is left zeroed to keep the buffer
    /// small (a full embed_tokens `[51865, 128]` would be ~26 MB).
    fn synthetic_checkpoint(tensors: &[(&str, &[u64])]) -> Vec<u8> {
        let mut cursor = 0usize;
        let mut entries = Vec::new();
        for &(name, shape) in tensors {
            let elems: u64 = shape.iter().product();
            let span = elems as usize * 4; // F32
            let begin = cursor;
            let end = cursor + span;
            cursor = end;
            let dims = shape
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join(",");
            entries.push(format!(
                r#""{name}":{{"dtype":"F32","shape":[{dims}],"data_offsets":[{begin},{end}]}}"#
            ));
        }
        let header = format!("{{{}}}", entries.join(","));
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(&vec![0u8; cursor]);
        out
    }

    #[test]
    fn write_hparams_derives_values_from_tensor_shapes() {
        // Shapes chosen so every derived hparam is distinct and hand-verifiable.
        // Trailing (unread) dims are shrunk to 1: derivation reads only shape[0]
        // (or shape[1] for n_mels), so this changes no derived value.
        let ckpt = synthetic_checkpoint(&[
            ("model.encoder.conv1.weight", &[128, 80, 3]),
            ("model.encoder.embed_positions.weight", &[1500, 1]),
            ("model.decoder.embed_positions.weight", &[448, 1]),
            ("model.decoder.embed_tokens.weight", &[51865, 1]),
            ("model.encoder.layers.0.fc1.weight", &[512, 1]),
            ("model.encoder.layers.1.mlp.fc2.weight", &[2, 2]),
            ("model.decoder.layers.0.self_attn.q_proj.weight", &[2, 2]),
        ]);

        let file = GgufFile::parse(convert(ckpt, None).unwrap().to_bytes().unwrap()).unwrap();
        let u = |k: &str| file.get(k).and_then(|v| v.as_u64());

        // d_model / n_mels from conv1 [d_model, n_mels, 3]; n_head = d_model/64.
        assert_eq!(u(KEY_N_AUDIO_STATE), Some(128));
        assert_eq!(u(KEY_N_TEXT_STATE), Some(128));
        assert_eq!(u(KEY_N_MELS), Some(80));
        assert_eq!(u(KEY_N_AUDIO_HEAD), Some(2)); // 128 / 64
        assert_eq!(u(KEY_N_TEXT_HEAD), Some(2));
        // Positional / vocab / ffn widths from tensor shape[0].
        assert_eq!(u(KEY_N_AUDIO_CTX), Some(1500));
        assert_eq!(u(KEY_N_TEXT_CTX), Some(448));
        assert_eq!(u(KEY_N_VOCAB), Some(51865));
        assert_eq!(u(KEY_FFN_DIM), Some(512));
        // Contiguous layer counts: encoder blocks 0 and 1, decoder only 0.
        assert_eq!(u(KEY_N_AUDIO_LAYER), Some(2));
        assert_eq!(u(KEY_N_TEXT_LAYER), Some(1));
        // Fixed Whisper constants (documented in this module's source).
        assert_eq!(u(KEY_EOT), Some(u64::from(WHISPER_EOT)));
        assert_eq!(WHISPER_EOT, 50257);

        let ids: Vec<u64> = file
            .get(KEY_DECODER_START_IDS)
            .and_then(|v| v.as_array())
            .unwrap()
            .values
            .iter()
            .map(|v| v.as_u64().unwrap())
            .collect();
        assert_eq!(ids, vec![50258, 50259, 50359, 50363]);
    }

    #[test]
    fn quantized_conversion_produces_loadable_kquant_gguf() {
        // A rank-2, 512-element weight (two super-blocks) is K-quantized; a
        // rank-1 bias stays F32; metadata is byte-identical to the plain path.
        let weight: Vec<f32> = (0..512).map(|i| (i as f32 - 256.0) * 0.01).collect();
        let bias = [0.5f32, -0.5, 1.0, -1.0];
        let mut data = Vec::new();
        for v in &weight {
            data.extend_from_slice(&v.to_le_bytes());
        }
        for v in &bias {
            data.extend_from_slice(&v.to_le_bytes());
        }
        let wbytes = weight.len() * 4;
        let header = format!(
            r#"{{"big.weight":{{"dtype":"F32","shape":[2,256],"data_offsets":[0,{wbytes}]}},"b.bias":{{"dtype":"F32","shape":[4],"data_offsets":[{wbytes},{}]}}}}"#,
            wbytes + 16
        );
        let mut buf = Vec::new();
        buf.extend_from_slice(&(header.len() as u64).to_le_bytes());
        buf.extend_from_slice(header.as_bytes());
        buf.extend_from_slice(&data);

        let unq = convert(buf.clone(), None).unwrap();
        let q = convert(buf, Some(GgmlType::Q4K)).unwrap();
        // The quantized path also bakes the `vokra.quant.*` policy chunk
        // (T05 contract): default_scheme + rule_count + hifigan_int8_opt_in
        // = 3 keys, plus 3 keys per rule. Whisper's default Q4K policy
        // resolves to `weight-only rank>=2` producing 2 rules, so:
        //   q - unq == 3 + 2*3 == 9
        assert_eq!(q.metadata_count(), unq.metadata_count() + 9);
        assert_eq!(unq.tensor_count(), q.tensor_count());

        let file = GgufFile::parse(q.to_bytes().unwrap()).unwrap();
        assert_eq!(file.tensor_info("big.weight").unwrap().dtype, GgmlType::Q4K);
        assert_eq!(file.tensor_info("b.bias").unwrap().dtype, GgmlType::F32);

        // The K-quantized weight decodes back within one Q4_K step of the range
        // (~0.17 per block here); the untouched bias is byte-exact.
        let back = file.tensor_f32("big.weight").unwrap();
        assert_eq!(back.len(), 512);
        for (i, &x) in weight.iter().enumerate() {
            assert!((back[i] - x).abs() < 0.4, "elem {i}: {} vs {x}", back[i]);
        }
        assert_eq!(
            file.tensor_f32("b.bias").unwrap(),
            vec![0.5, -0.5, 1.0, -1.0]
        );
    }

    #[test]
    fn text_vocab_floor_equals_eot_and_resource_is_well_formed() {
        // The bundled resource holds exactly the model-independent text ranks;
        // its floor is the fixed eot (this ties the +1-shift derivation and the
        // tokenizer embedding to the same invariant).
        assert_eq!(WHISPER_TEXT_VOCAB_LEN, WHISPER_EOT);

        // The committed resource must be a whole number of `{u8, u16, bytes}`
        // records and hold exactly WHISPER_TEXT_VOCAB_LEN of them (guards against
        // a truncated / garbled resource landing in the repo).
        let mut pos = 0usize;
        let mut count = 0u32;
        while pos < TEXT_VOCAB_RESOURCE.len() {
            let len =
                u16::from_le_bytes([TEXT_VOCAB_RESOURCE[pos + 1], TEXT_VOCAB_RESOURCE[pos + 2]])
                    as usize;
            pos += 3 + len;
            count += 1;
        }
        assert_eq!(
            pos,
            TEXT_VOCAB_RESOURCE.len(),
            "resource is not a whole record stream"
        );
        assert_eq!(
            count, WHISPER_TEXT_VOCAB_LEN,
            "resource must hold every text rank"
        );
    }

    #[test]
    fn embeds_base_tokenizer_byte_equal_to_reference() {
        // A base-sized checkpoint (n_vocab 51865) must embed a
        // `vokra.tokenizer.model` blob byte-for-byte equal to the committed
        // parity tokenizer.bin (the transformers/onnxruntime-generated
        // reference) — the strongest check that runtime detokenization matches
        // Hugging Face. Only embed_tokens' shape[0] is read, so a `[51865, 1]`
        // checkpoint suffices (data left zeroed).
        let ckpt = synthetic_checkpoint(&[("model.decoder.embed_tokens.weight", &[51865, 1])]);
        let file = GgufFile::parse(convert(ckpt, None).unwrap().to_bytes().unwrap()).unwrap();
        let blob = tokenizer_blob_from_gguf(&file);

        let reference = std::fs::read(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../tests/parity/whisper_base/tokenizer.bin"),
        )
        .expect("committed whisper_base/tokenizer.bin");
        assert_eq!(
            blob.len(),
            reference.len(),
            "embedded tokenizer length {} != reference {}",
            blob.len(),
            reference.len()
        );
        assert_eq!(blob, reference, "embedded tokenizer != committed reference");
    }

    #[test]
    fn large_v3_prefix_and_tokenizer_layout() {
        // large-v3 (n_vocab 51866, one extra language) must (a) shift the two
        // tail specials +1 in the decode prefix while keeping eot fixed, and
        // (b) embed a 51866-entry tokenizer whose text records are byte-identical
        // to base with an extra empty-special record appended.
        let ckpt = synthetic_checkpoint(&[("model.decoder.embed_tokens.weight", &[51866, 1])]);
        let file = GgufFile::parse(convert(ckpt, None).unwrap().to_bytes().unwrap()).unwrap();

        // eot fixed; only the two tail specials shift +1.
        assert_eq!(file.get(KEY_EOT).and_then(|v| v.as_u64()), Some(50257));
        let ids: Vec<u64> = file
            .get(KEY_DECODER_START_IDS)
            .and_then(|v| v.as_array())
            .unwrap()
            .values
            .iter()
            .map(|v| v.as_u64().unwrap())
            .collect();
        assert_eq!(ids, vec![50258, 50259, 50360, 50364]);

        // Tokenizer blob: count 51866, text prefix == the bundled resource, and
        // a tail of all empty-special `{1, 0, 0}` records.
        let blob = tokenizer_blob_from_gguf(&file);
        assert_eq!(&blob[..4], &51866u32.to_le_bytes());
        let text_end = 4 + TEXT_VOCAB_RESOURCE.len();
        assert_eq!(&blob[4..text_end], TEXT_VOCAB_RESOURCE);
        let tail = &blob[text_end..];
        let n_specials = 51866 - WHISPER_TEXT_VOCAB_LEN as usize; // 1609
        assert_eq!(tail.len(), n_specials * 3);
        assert!(
            tail.chunks_exact(3).all(|r| r == [1, 0, 0]),
            "large-v3 special tail must be all empty-special records"
        );

        // Base (51865) shares the identical text prefix — only the special count
        // differs — so the two blobs agree on the header-less text region.
        let base = tokenizer_blob(51865);
        assert_eq!(base[4..text_end], blob[4..text_end]);
    }

    #[test]
    fn all_whisper_sizes_metadata_are_consistent() {
        // Table-drives the M2-06 size-detection contract across every supported
        // multilingual Whisper size. For each row we build a synthetic checkpoint
        // whose *shape quintuple* matches the real OpenAI config (d_model,
        // n_audio_layer, n_text_layer, n_mels, n_vocab) — trailing (unread) dims
        // are shrunk to 1 exactly as in `write_hparams_derives_values_from_tensor_shapes`
        // to keep buffers small — then assert:
        //   (a) `derive_name` returns the expected label,
        //   (b) `vokra.model.name` in the emitted GGUF matches (b) label,
        //   (c) `vokra.frontend.n_mels` matches the row's n_mels (80 or 128),
        //   (d) `vokra.tokenizer.model` is present and its byte length mirrors
        //       `embed_tokenizer` semantics — `4 + TEXT_VOCAB_RESOURCE.len()
        //       + 3*(n_vocab - WHISPER_TEXT_VOCAB_LEN)` — when n_vocab >= 50257.
        // Sources: openai/whisper `whisper/model.py` size table + HF
        // `openai/whisper-{size}/config.json`.
        let rows: &[(&str, u64, u32, u32, u64, u64)] = &[
            // (label,         d_model, n_audio_layer, n_text_layer, n_mels, n_vocab)
            ("whisper-base", 512, 6, 6, 80, 51865),
            ("whisper-small", 768, 12, 12, 80, 51865),
            ("whisper-medium", 1024, 24, 24, 80, 51865),
            ("whisper-large-v3", 1280, 32, 32, 128, 51866),
            ("whisper-turbo", 1280, 32, 4, 128, 51866),
        ];

        for &(label, d_model, n_audio_layer, n_text_layer, n_mels, n_vocab) in rows {
            // (a) Direct derive_name check — the pure shape-to-label mapping.
            assert_eq!(
                derive_name(d_model, n_audio_layer, n_text_layer, n_mels).unwrap(),
                label,
                "derive_name mismatch for {label}",
            );

            // Build the checkpoint: one conv1 with the real [d_model, n_mels, 1]
            // shape (trailing 3 shrunk to 1), embed_tokens with [n_vocab, 1], plus
            // enough layer prefixes for `count_layers` to see the right counts.
            let mut tensors: Vec<(String, Vec<u64>)> = vec![
                (
                    "model.encoder.conv1.weight".to_string(),
                    vec![d_model, n_mels, 1],
                ),
                (
                    "model.encoder.embed_positions.weight".to_string(),
                    vec![1500, 1],
                ),
                (
                    "model.decoder.embed_positions.weight".to_string(),
                    vec![448, 1],
                ),
                (
                    "model.decoder.embed_tokens.weight".to_string(),
                    vec![n_vocab, 1],
                ),
                (
                    "model.encoder.layers.0.fc1.weight".to_string(),
                    vec![d_model * 4, 1],
                ),
            ];
            for i in 0..n_audio_layer {
                tensors.push((
                    format!("model.encoder.layers.{i}.mlp.fc2.weight"),
                    vec![1, 1],
                ));
            }
            for i in 0..n_text_layer {
                tensors.push((
                    format!("model.decoder.layers.{i}.self_attn.q_proj.weight"),
                    vec![1, 1],
                ));
            }
            let refs: Vec<(&str, &[u64])> = tensors
                .iter()
                .map(|(n, s)| (n.as_str(), s.as_slice()))
                .collect();
            let ckpt = synthetic_checkpoint(&refs);

            let file = GgufFile::parse(convert(ckpt, None).unwrap().to_bytes().unwrap()).unwrap();

            // (b) vokra.model.name in the emitted GGUF matches the row label.
            assert_eq!(
                file.get(chunks::KEY_MODEL_NAME).and_then(|v| v.as_str()),
                Some(label),
                "vokra.model.name mismatch for {label}",
            );

            // (c) Front-end n_mels tracks the row's checkpoint (80 or 128).
            let spec = FrontendSpec::from_gguf(&file).unwrap();
            assert_eq!(
                spec.n_mels, n_mels as u32,
                "vokra.frontend.n_mels mismatch for {label}",
            );

            // (d) Tokenizer blob present when n_vocab >= 50257 (all real sizes),
            // with the exact byte length embed_tokenizer produces:
            // `4 (u32 count) + TEXT_VOCAB_RESOURCE.len() + 3*(n_vocab - 50257)`.
            let blob = tokenizer_blob_from_gguf(&file);
            let expected_len = 4
                + TEXT_VOCAB_RESOURCE.len()
                + 3 * (n_vocab as usize - WHISPER_TEXT_VOCAB_LEN as usize);
            assert_eq!(
                blob.len(),
                expected_len,
                "vokra.tokenizer.model length mismatch for {label}",
            );
            // The u32 count header must equal the row's n_vocab.
            assert_eq!(
                &blob[..4],
                &(n_vocab as u32).to_le_bytes(),
                "tokenizer header count mismatch for {label}",
            );
        }

        // Negative row: an unknown quintuple must return an explicit error, not
        // silently map to some default label (FR-EX-08 — no silent fallback).
        // Uses a d_model=1536 (never a real whisper size) with valid layer / mel
        // counts so `is_synthetic_shape` does NOT rescue it — this asserts that
        // `derive_name` itself refuses unknown real-shaped checkpoints.
        let err = derive_name(1536, 24, 24, 80).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("unknown whisper size"),
            "expected unknown-size error, got: {msg}",
        );
    }

    // ---------------------------------------------------------------------
    // M2-08 T06 — quant policy tests (cargo test -p vokra-convert quant_policy)
    // ---------------------------------------------------------------------

    #[test]
    fn quant_policy_resolve_first_match_wins() {
        let p = QuantPolicy::whisper_q4_k();
        // Suffix rule `.bias` pinned to F32 (first-match, before default Q4_K).
        assert_eq!(
            resolve(&p, "encoder.blocks.0.mlp.0.bias"),
            QuantScheme::Fp32
        );
        // Suffix rule `.weight_norm` pinned to F32.
        assert_eq!(
            resolve(&p, "encoder.blocks.0.mlp.0.weight_norm"),
            QuantScheme::Fp32
        );
        // Fall-through: everything else takes the default (Q4_K).
        assert_eq!(
            resolve(&p, "encoder.blocks.0.mlp.0.weight"),
            QuantScheme::W4A16Q4K
        );
    }

    #[test]
    fn quant_policy_preset_vocoder_safe_widens_to_fp16() {
        let p = QuantPolicy::default_vocoder_safe();
        // No rules → every tensor resolves to the default (F16).
        assert_eq!(resolve(&p, "encoder.conv1.weight"), QuantScheme::Fp16);
        assert_eq!(
            resolve(&p, "decoder.embed_tokens.weight"),
            QuantScheme::Fp16
        );
        assert_eq!(resolve(&p, "any.name"), QuantScheme::Fp16);
    }

    #[test]
    fn quant_policy_scheme_weight_dtype_and_alias() {
        assert_eq!(QuantScheme::Fp32.weight_dtype(), GgmlType::F32);
        assert_eq!(QuantScheme::Fp16.weight_dtype(), GgmlType::F16);
        assert_eq!(QuantScheme::W4A16Q4K.weight_dtype(), GgmlType::Q4K);
        assert_eq!(QuantScheme::W4A16Q5K.weight_dtype(), GgmlType::Q5K);
        assert_eq!(QuantScheme::W4A16Q6K.weight_dtype(), GgmlType::Q6K);
        // Chunk aliases (T05 contract).
        assert_eq!(QuantScheme::Fp32.as_str(), "fp32");
        assert_eq!(QuantScheme::Fp16.as_str(), "fp16");
        assert_eq!(QuantScheme::W4A16Q4K.as_str(), "w4a16-q4k");
        assert_eq!(QuantScheme::W4A16Q5K.as_str(), "w4a16-q5k");
        assert_eq!(QuantScheme::W4A16Q6K.as_str(), "w4a16-q6k");
    }

    #[test]
    fn quant_policy_writes_vokra_quant_chunk() {
        // A whisper conversion with the whisper_q4_k policy must stamp the
        // resolved policy into `vokra.quant.*` metadata so a future runtime
        // can reconstruct it.
        //
        // Build a small but *K-quantizable* whisper checkpoint: every weight
        // tensor's element count is a multiple of QK_K (256) so the policy's
        // Q4_K default is applicable; biases are rank-1 and stay F32 via the
        // `.bias` suffix rule.
        let mut tensors: Vec<(String, Vec<u64>)> = vec![
            ("model.encoder.conv1.weight".to_string(), vec![512, 80, 3]),
            (
                // 1536 = 6 * 256 (QK_K-aligned so K-quant is applicable).
                "model.encoder.embed_positions.weight".to_string(),
                vec![1536, 1],
            ),
            (
                "model.decoder.embed_positions.weight".to_string(),
                vec![512, 1],
            ),
            (
                "model.decoder.embed_tokens.weight".to_string(),
                vec![256, 1],
            ),
            (
                "model.encoder.layers.0.fc1.weight".to_string(),
                vec![2, 256],
            ),
            ("model.encoder.layers.0.fc1.bias".to_string(), vec![512]),
        ];
        // One matching layer prefix so count_layers sees exactly 1 encoder
        // block (+ 0 decoder blocks — synthetic).
        for i in 0..1 {
            tensors.push((
                format!("model.encoder.layers.{i}.mlp.fc2.weight"),
                vec![2, 256],
            ));
        }
        let refs: Vec<(&str, &[u64])> = tensors
            .iter()
            .map(|(n, s)| (n.as_str(), s.as_slice()))
            .collect();
        let ckpt = synthetic_checkpoint(&refs);

        let b = convert_with_policy(ckpt, Some(QuantPolicy::whisper_q4_k())).unwrap();
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        assert_eq!(
            file.get("vokra.quant.default_scheme")
                .and_then(|v| v.as_str()),
            Some("w4a16-q4k"),
        );
        assert_eq!(
            file.get("vokra.quant.rule_count").and_then(|v| v.as_u64()),
            Some(2),
        );
        // The `.bias` rule must be preserved as suffix scheme fp32.
        assert_eq!(
            file.get("vokra.quant.rule.0.pattern_kind")
                .and_then(|v| v.as_str()),
            Some("suffix"),
        );
        assert_eq!(
            file.get("vokra.quant.rule.0.pattern")
                .and_then(|v| v.as_str()),
            Some(".bias"),
        );
        assert_eq!(
            file.get("vokra.quant.rule.0.scheme")
                .and_then(|v| v.as_str()),
            Some("fp32"),
        );

        // Sanity: the `.bias` tensor stays F32 (per `.bias` suffix rule),
        // while the K-quantizable weight (2×256) is Q4_K.
        let bias = file.tensor_info("model.encoder.layers.0.fc1.bias").unwrap();
        assert_eq!(bias.dtype, GgmlType::F32);
        let w = file
            .tensor_info("model.encoder.layers.0.mlp.fc2.weight")
            .unwrap();
        assert_eq!(w.dtype, GgmlType::Q4K);
    }

    #[test]
    fn quant_policy_inapplicable_errors_no_silent_widen() {
        // A K-quant target on a tensor that cannot be K-quantized (rank 1
        // AND element count not a multiple of QK_K) must fail explicitly —
        // FR-EX-08: no silent widen.
        let ckpt = synthetic_checkpoint(&[
            ("model.encoder.conv1.weight", &[512, 80, 3]),
            ("model.encoder.embed_positions.weight", &[1500, 1]),
            ("model.decoder.embed_positions.weight", &[448, 1]),
            ("model.decoder.embed_tokens.weight", &[256, 1]),
            ("model.encoder.layers.0.fc1.weight", &[2, 256]),
        ]);
        // Force a K-quant scheme on every tensor via the default; conv1 is
        // rank-3 with element_count 512*80*3 = 122880 which is a multiple
        // of 256 so it's applicable; the positional embeddings are rank-2
        // but 1500*1 = 1500 which is NOT a multiple of 256 → inapplicable.
        let policy = QuantPolicy {
            default: QuantScheme::W4A16Q4K,
            rules: Vec::new(),
        };
        let err = convert_with_policy(ckpt, Some(policy)).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("quant policy inapplicable"),
            "expected QuantPolicyInapplicable, got: {msg}",
        );
        assert!(
            msg.contains("w4a16-q4k"),
            "message should name scheme: {msg}"
        );
    }

    #[test]
    fn quant_policy_legacy_convert_none_writes_no_quant_chunk() {
        // The `None` (byte-exact) path must not write `vokra.quant.*` — this
        // keeps every pre-T06 test's metadata_count assertions valid.
        let file = GgufFile::parse(
            convert(synthetic_whisper(), None)
                .unwrap()
                .to_bytes()
                .unwrap(),
        )
        .unwrap();
        assert!(file.get("vokra.quant.default_scheme").is_none());
        assert!(file.get("vokra.quant.rule_count").is_none());
    }
}
