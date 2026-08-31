//! **FireRedTeam/FireRedASR-AED-L** — runtime binder for the
//! `firered_asr_aed_l` converter arch (Wave C1 2026-08-15 audit
//! follow-up; loud-partial per the `firered_vad` / `emotion2vec` /
//! `smart_turn` / RMVPE / NISQA / panns precedent — CLAUDE.md 教訓 (a):
//! 「loud-partial は fake-complete より honest」).
//!
//! # The gap this closes
//!
//! `crates/vokra-convert/src/models/firered_asr_aed_l.rs`
//! (coverage-audit-2026-08-03 wave-b) writes a GGUF stamped
//! `vokra.model.arch = "firered_asr_aed_l"`, `vokra.model.name =
//! "firered-asr-aed-l"`, `vokra.model.category = "asr"` and
//! `vokra.provenance.upstream_hf = "FireRedTeam/FireRedASR-AED-L"` — but
//! a workspace-wide grep proved that **nothing anywhere read that arch
//! string back**. Every converted FireRedASR-AED-L checkpoint was
//! therefore unloadable: the bytes sat on disk in the right container
//! with the right provenance stamps, and no code path could turn them
//! into a model handle. This module is that missing consumer.
//!
//! # Primary sources
//!
//! - Upstream release: <https://huggingface.co/FireRedTeam/FireRedASR-AED-L>
//!   (Apache-2.0 both code and weights per the converter docstring;
//!   `docs/license-audit.md` §3.1 already carries an owner ☑ Commercial
//!   row for this release — this module neither reads nor writes that
//!   sign-off).
//! - Family reference code: <https://github.com/FireRedTeam/FireRedASR>
//!   (the FireRedTeam speech family: FireRedASR-AED-L / FireRedASR-LLM-L
//!   / FireRedVAD / FireRedTTS).
//! - In-repo contract: [`CONVERTER_PATH`] — the GGUF writer whose `ARCH`
//!   / `NAME` / `CATEGORY` / `UPSTREAM_HF` constants this module mirrors
//!   verbatim.
//! - In-repo audit ticket: [`AUDIT_TICKET_PATH`] — records the model as
//!   ~1.1 B params / ~2.2 GB BF16 safetensors, category `asr`, and,
//!   load-bearingly for this module, records that the release is **not**
//!   shape-compatible with Whisper (see "Why the forward is a
//!   loud-partial" below).
//!
//! Note the provenance key: the converter stamps
//! `vokra.provenance.upstream_hf` (a HuggingFace slug), **not**
//! `vokra.provenance.upstream_url`.
//!
//! # Model class
//!
//! FireRedASR-AED-L is an **AED** — an Attention-Encoder-Decoder: an
//! acoustic encoder over a log-mel front-end, a Transformer decoder, and
//! cross-attention from the decoder into the encoder output, decoded with
//! a beam search over a vocabulary head. It is trained for Mandarin
//! Chinese ASR. That places it in the same *shape family* as Whisper /
//! Canary rather than the CTC (`parakeet-ctc`, `omniasr-ctc`) or
//! LLM-decoder (`voxtral`, `canary-qwen`, `firered_asr_llm_l`) families.
//!
//! # Why the forward is a loud-partial (CLAUDE.md 教訓 (a))
//!
//! Being in Whisper's *shape family* is not the same as having Whisper's
//! *shape*, and the in-repo audit ticket says so explicitly: its
//! §Converter section records
//!
//! > 既存 ModelKind reuse? no (FireRedTeam AED は Whisper と shape
//! > 互換ではない、独自 hparam)
//!
//! i.e. **the release has its own hyper-parameters and is not
//! shape-compatible with Whisper**. The remaining execution contract has
//! four concrete gaps, and none of them is a kernel:
//!
//! 1. **The native frontend is not fully wired.** [`native`] now contains
//!    source-faithful CMVN, positional encoding, and Conv2d subsampling
//!    helpers. The complete fbank/CMVN input path, tensor binding, masks, and
//!    encoder graph still remain fail-closed until the VAST tensor evidence is
//!    consumed by a native model handle.
//! 2. **No native tensor-name consumer.** The independent upstream dumper
//!    authenticates all 940 names against `named_parameters()` /
//!    `named_buffers()` and records their roles. The converter/binder still
//!    has no Rust field mapping that consumes that sidecar, so a best-guess
//!    walk could bind wrong weights with valid shapes.
//! 3. **No tokenizer blob binding.** The source-authenticated
//!    SentencePiece/TokenDict contract and 7832-entry dictionary are known,
//!    but the converter stamps no [`KEY_TOKENIZER_MODEL`] blob. This binder
//!    cannot render decoder ids as Mandarin text until a native mapping is
//!    added. [`FireredAsrAed::has_tokenizer`] reports blob presence.
//! 4. **Native graph gap.** The pinned Conformer source uses its Conv2d
//!    subsampling stem, relative-position attention, and source-faithful
//!    inference-only Conformer block; [`native`] exposes those exact
//!    CPU/Metal-dispatched building blocks. The fbank input and strict
//!    940-field model graph are not yet wired, so this binder remains
//!    fail-closed.
//!
//! The upstream config is additionally awkward to reach: the handoff for
//! the sibling LLM release
//! (`docs/handoff/vast-ai-publish-firered-asr-llm-l.md` §0.3) records
//! that the HF-side `config.yaml` for that release is **0 bytes**, so
//! the real configuration has to be read out of
//! `github.com/FireRedTeam/FireRedASR` instead. Whether the AED release
//! shares that posture is **not** verified anywhere in this repository,
//! and this module does not assert that it does.
//!
//! So: the remaining blockers are the full native frontend/field consumer,
//! tokenizer blob, and complete 940-field Conformer/AED graph. The reusable
//! native blocks are present, but their equivalence is not claimed until an
//! independent VAST parity run consumes the authenticated checkpoint.
//!
//! # Loud-partial classification
//!
//! - **Real (this WP)**:
//!   - [`FireredAsrAed::from_gguf`] with **strict** `vokra.model.arch ==
//!     "firered_asr_aed_l"` verification. A foreign GGUF — including
//!     every sibling `category = "asr"` arch tag, and most dangerously
//!     the FireRedTeam LLM sibling `firered_asr_llm_l` — is refused
//!     loudly with the whole ASR fleet enumerated (see "Sibling family
//!     distinctness").
//!   - [`FireredAsrAedWeights::from_gguf`] with a non-empty
//!     tensor-manifest gate: a GGUF carrying zero tensors is refused
//!     rather than bound into an all-zero forward (FR-EX-08).
//!   - [`FireredAsrAedWeights::dims`] — a by-name lookup that **names
//!     the absent tensor** instead of returning `None` for a caller to
//!     swallow.
//!   - The optional [`KEY_REQUIRED_TENSORS`] manifest gate — when a
//!     producer declares the tensor names it wrote, a truncated or
//!     mis-merged GGUF fails at **load** time naming the first missing
//!     tensor rather than surprising a forward halfway through.
//!   - The optional all-or-nothing [`FireredAsrAedConfig`] group
//!     (`vokra.firered_asr_aed_l.*`) — now stamped by the VAST converter for
//!     the authenticated release geometry. Absent → [`FireredAsrAed::config`]
//!     is `None` and a minimal inspection fixture still binds. Partially
//!     stamped → loud
//!     [`VokraError::ModelLoad`] naming the missing key. A `0` sentinel
//!     or an indivisible `d_model % n_head` → loud.
//!   - Sample-rate guarding: [`FireredAsrAed::transcribe_tokens`]
//!     refuses a mismatched rate with [`VokraError::InvalidArgument`]
//!     once the group is stamped — Vokra never silently resamples
//!     (FR-EX-08).
//!   - Weight-license surfacing, fail-closed to
//!     [`LicenseClass::Unknown`] when the stamp is absent.
//!
//! - **Loud-partial (this WP)**: [`FireredAsrAed::transcribe_tokens`]
//!   and the [`AsrEngine`] trait path return
//!   [`VokraError::UnsupportedOp`] naming the four remaining gaps above plus
//!   the primary sources, so a reader diagnosing the gap has fully
//!   specified places to walk. **No fabricated token ids or text are
//!   ever emitted** (FR-EX-08 — no silent partial output).
//!
//! # Sibling family distinctness (`category = "asr"` neighbourhood)
//!
//! [`ARCH`] = `"firered_asr_aed_l"` is **deliberately distinct** from
//! every sibling ASR arch tag in the converter tree. They share a
//! category and, in several cases, a topology *family*, but never a
//! tensor manifest:
//!
//! - `firered_asr_llm_l` — **the same team's other release**: Conformer
//!   encoder + linear/MLP audio-text adapter + a Qwen2 LM decoder
//!   (~16.6 GB). Same upstream org, same `asr` category, completely
//!   different decoder half. This is the single most likely
//!   mis-dispatch, which is why the arch check runs first;
//! - `whisper` / `distil-whisper` / `kotoba-whisper` — OpenAI Whisper
//!   and its distilled / Japanese-tuned derivatives: also AEDs, but with
//!   Whisper's own hparams and BPE vocabulary, which the audit ticket
//!   explicitly records FireRedASR-AED-L does **not** share;
//! - `canary` / `canary-1b-flash` / `canary-qwen` — NVIDIA NeMo
//!   FastConformer encoders with, respectively, a Transformer AED
//!   decoder, a faster AED variant, and a Qwen LM decoder;
//! - `parakeet-tdt` / `parakeet-ctc` / `omniasr-ctc` — CTC / RNN-T-TDT
//!   heads with no attention decoder at all;
//! - `kyutai-stt` / `voxtral` / `nemotron_asr_streaming` — streaming and
//!   LLM-decoder ASR with their own state and prompt contracts;
//! - `moonshine` — a variable-length-input encoder-decoder with no
//!   fixed 30 s window.
//!
//! Silently aliasing any of these would mis-route runtime dispatch: the
//! tensor-name walk would fail with a downstream missing-tensor error
//! instead of a specific arch-mismatch message (FR-EX-08).
//!
//! # Structuring for the LLM sibling
//!
//! [`FireredAsrAedEncoderConfig`] and [`FireredAsrAedDecoderConfig`] are
//! **separate public types** rather than flattened fields, so that if a
//! real-checkpoint transcription later shows the two FireRedTeam
//! releases share an acoustic encoder, the encoder half can be lifted
//! into a shared type by a *move*, not a rewrite. This module does **not
//! assert** that they share one: the in-repo descriptions differ
//! (`crates/vokra-convert/src/models/firered_asr_llm_l.rs` calls the AED
//! release's encoder a "Transformer encoder" and its own a "Conformer
//! encoder"), and no primary source in this repository settles it. See
//! the follow-ups.
//!
//! # Cross-crate constant duplication
//!
//! [`ARCH`] / [`NAME`] / [`CATEGORY`] / [`UPSTREAM_HF`] mirror the
//! converter's constants — the same rule every sibling binder
//! (`firered_vad` / `emotion2vec` / `smart_turn` / `panns` / `snac` / …)
//! follows so `vokra-models` does not gain a dependency edge onto
//! `vokra-convert`. The layered convention holds: `vokra-ops → nothing
//! GGUF-aware`, `vokra-core → GGUF reader`, `vokra-models → GGUF
//! binder`, `vokra-convert → GGUF writer`. The handshake is a plain
//! string, pinned on both sides by tests. Note the arch uses `_` while
//! the name uses `-`; both spellings are load-bearing on the wire and
//! are pinned separately.
//!
//! # Licensing
//!
//! The converter stamps `apache-2.0` → [`LicenseClass::Permissive`] by
//! default. This binder only **surfaces** whatever class a GGUF carries
//! and fail-closes to [`LicenseClass::Unknown`] when nothing is stamped.
//! `docs/license-audit.md` §3.1 is owner-only per memory
//! `[[feedback-license-signoff-primary-source]]`; this module does not
//! read it, write it, or treat the converter's default as a sign-off.
//!
//! # No ONNX / no pickle (permanent)
//!
//! The upstream release ships a PyTorch pickle bridged offline by
//! [`SIDECAR_PATH`] (uv-managed Python 3.12 per memory
//! `[[feedback-python-uses-uv]]` + `[[feedback-python-3-12]]`); the
//! runtime never touches ONNX or a pickle (FR-LD-05 / NFR-DS-02).

use vokra_core::engines::AsrEngine;
use vokra_core::gguf::{GgufFile, GgufMetadataValue, GgufValueType, chunks};
use vokra_core::tasks::Transcription;
use vokra_core::{BackendKind, LicenseClass, Result, VokraError};

mod native;

pub use native::{
    FIRERED_ASR_AED_HOT_OPS, FireRedCmvn, FireRedConformerBlock, FireRedConformerBlockWeights,
    FireRedConformerConvolution, FireRedConformerFeedForward, FireRedConv2dSubsampling,
    FireRedRelativeAttention, relative_positional_encoding,
};

// ---------------------------------------------------------------------------
// Contract constants — mirror of
// `crates/vokra-convert/src/models/firered_asr_aed_l.rs`.
// See the module docstring for the cross-crate duplication rationale.
// ---------------------------------------------------------------------------

/// Expected `vokra.model.arch` value written by
/// `vokra-cli convert --model firered-asr-aed-l`.
///
/// Deliberately distinct from every sibling `category = "asr"` arch tag
/// — above all from the same team's `firered_asr_llm_l` release — see
/// the module docstring's "Sibling family distinctness" section for why
/// aliasing would be an FR-EX-08 violation.
pub const ARCH: &str = "firered_asr_aed_l";

/// Expected `vokra.model.name` value written by the converter.
///
/// Note the hyphens: the *name* is `firered-asr-aed-l` while the *arch*
/// is `firered_asr_aed_l` (underscores). Both spellings are load-bearing
/// on the wire, so they are pinned separately by a test.
pub const NAME: &str = "firered-asr-aed-l";

/// Expected `vokra.model.category` value — `"asr"`, shared with the
/// whole speech-recognition fleet (`whisper`, `distil-whisper`,
/// `kotoba-whisper`, `canary`, `canary-qwen`, `parakeet-ctc`,
/// `omniasr-ctc`, `kyutai-stt`, `voxtral`, `firered_asr_llm_l`, …). The
/// category alone can therefore never disambiguate a checkpoint — only
/// [`ARCH`] can.
pub const CATEGORY: &str = "asr";

/// Upstream HuggingFace slug recorded under
/// [`KEY_PROVENANCE_UPSTREAM_HF`].
pub const UPSTREAM_HF: &str = "FireRedTeam/FireRedASR-AED-L";

/// Default upstream weight licence (SPDX), mirrored from the converter.
/// Resolves to [`LicenseClass::Permissive`].
///
/// See the module docstring's "Licensing" section: this is what the
/// *converter* stamps, not a sign-off. `docs/license-audit.md` §3.1 is
/// owner-only.
pub const DEFAULT_LICENSE_SPDX: &str = "apache-2.0";

/// GGUF metadata key: model category tag (mirror of the converter's
/// module-private const).
pub const KEY_MODEL_CATEGORY: &str = "vokra.model.category";

/// GGUF metadata key: upstream HuggingFace slug (mirror of the
/// converter's module-private const). FireRedASR-AED-L ships on HF, so
/// provenance rides `upstream_hf` rather than `upstream_url`.
pub const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";

/// GGUF metadata key carrying an embedded tokenizer blob.
///
/// The same wire key the `whisper` binder reads. Today's FireRedASR-AED-L
/// converter never writes it, which is loud-partial blocker (3): an AED
/// decoder emits token ids, and without the upstream Mandarin vocabulary
/// there is nothing to render them with.
/// [`FireredAsrAed::has_tokenizer`] reports its presence per GGUF.
pub const KEY_TOKENIZER_MODEL: &str = "vokra.tokenizer.model";

// ---------------------------------------------------------------------------
// Primary-source anchors — cited verbatim in the loud-partial message so a
// reader diagnosing the gap has fully specified places to walk.
// ---------------------------------------------------------------------------

/// Primary-source anchor: the upstream HuggingFace release.
pub const PRIMARY_SOURCE_HF: &str = "huggingface.co/FireRedTeam/FireRedASR-AED-L";

/// Primary-source anchor: the FireRedTeam speech-family reference code
/// (FireRedASR-AED-L / FireRedASR-LLM-L / FireRedVAD / FireRedTTS).
pub const PRIMARY_SOURCE_FAMILY_CODE: &str = "github.com/FireRedTeam/FireRedASR";

/// In-repo contract anchor: the converter this binder mirrors.
pub const CONVERTER_PATH: &str = "crates/vokra-convert/src/models/firered_asr_aed_l.rs";

/// In-repo anchor: the coverage-audit ticket that records the model's
/// scale, category, and the "not shape-compatible with Whisper" finding
/// this module's loud-partial rests on.
pub const AUDIT_TICKET_PATH: &str =
    "docs/tickets/coverage-audit-2026-08-03/wave-b/firered-asr-aed-l.md";

/// The offline sidecar that bridges the upstream PyTorch pickle to the
/// safetensors the converter consumes. It is also the natural place for
/// a real checkpoint's topology to be transcribed and the
/// [`FIREREDASRAED_SPEC_KEYS`] group emitted. Never shipped inside the
/// `vokra-*` runtime (FR-LD-05 / NFR-DS-02).
pub const SIDECAR_PATH: &str = "tools/parity/firered_asr_aed_l_prepare_checkpoint.py";

// ---------------------------------------------------------------------------
// `vokra.firered_asr_aed_l.*` — the optional, all-or-nothing hyper-parameter
// group.
//
// The VAST converter stamps this group from the authenticated release
// inspection. Declaring it here lets `from_gguf` verify the complete group
// instead of silently defaulting one half of it (FR-EX-08).
// ---------------------------------------------------------------------------

/// Sample rate the checkpoint expects, in Hz. Load-bearing: the binder
/// refuses PCM offered at any other rate rather than resampling
/// silently.
pub const KEY_SAMPLE_RATE: &str = "vokra.firered_asr_aed_l.sample_rate";

/// Log-mel band count of the acoustic front-end.
pub const KEY_N_MELS: &str = "vokra.firered_asr_aed_l.n_mels";

/// Vocabulary size of the decoder output head — the width of the logits
/// a beam search selects over, and the id space of the tokens
/// [`FireredAsrAed::transcribe_tokens`] will return once the forward
/// lands.
pub const KEY_VOCAB_SIZE: &str = "vokra.firered_asr_aed_l.vocab_size";

/// Acoustic encoder depth (block count).
pub const KEY_ENC_N_LAYER: &str = "vokra.firered_asr_aed_l.encoder.n_layer";

/// Acoustic encoder model width.
pub const KEY_ENC_D_MODEL: &str = "vokra.firered_asr_aed_l.encoder.d_model";

/// Acoustic encoder attention-head count. Invisible in the weight shapes
/// whenever QKV is packed into one projection, which is why it needs a
/// metadata key of its own.
pub const KEY_ENC_N_HEAD: &str = "vokra.firered_asr_aed_l.encoder.n_head";

/// Acoustic encoder feed-forward inner width.
pub const KEY_ENC_FFN_DIM: &str = "vokra.firered_asr_aed_l.encoder.ffn_dim";

/// Transformer decoder depth (block count).
pub const KEY_DEC_N_LAYER: &str = "vokra.firered_asr_aed_l.decoder.n_layer";

/// Transformer decoder model width. Need **not** equal the encoder width
/// — in an AED the cross-attention K/V projections map the encoder width
/// into the decoder width, so the two are independent axes. This module
/// deliberately does not assert their equality (see
/// [`FireredAsrAedConfig::validate`]).
pub const KEY_DEC_D_MODEL: &str = "vokra.firered_asr_aed_l.decoder.d_model";

/// Transformer decoder attention-head count (shared by the self- and
/// cross-attention stacks in the standard AED layout).
pub const KEY_DEC_N_HEAD: &str = "vokra.firered_asr_aed_l.decoder.n_head";

/// Transformer decoder feed-forward inner width.
pub const KEY_DEC_FFN_DIM: &str = "vokra.firered_asr_aed_l.decoder.ffn_dim";

/// Conformer depthwise-convolution kernel width.
pub const KEY_ENC_KERNEL_SIZE: &str = "vokra.firered_asr_aed_l.encoder.kernel_size";

/// Authenticated decoder/special-token ids from the checkpoint args.
pub const KEY_BLANK_ID: &str = "vokra.firered_asr_aed_l.blank_id";
pub const KEY_SOS_ID: &str = "vokra.firered_asr_aed_l.sos_id";
pub const KEY_EOS_ID: &str = "vokra.firered_asr_aed_l.eos_id";
pub const KEY_PAD_ID: &str = "vokra.firered_asr_aed_l.pad_id";

/// The hyper-parameter group in canonical read order — **all-or-nothing**.
///
/// [`FireredAsrAedConfig::from_gguf`] returns `Ok(None)` when *no* key is
/// present and a loud [`VokraError::ModelLoad`] when only *some* are:
/// silently defaulting the missing half would build a wrong-shaped
/// encoder or decoder that still runs (FR-EX-08).
pub const FIREREDASRAED_SPEC_KEYS: [&str; 16] = [
    KEY_SAMPLE_RATE,
    KEY_N_MELS,
    KEY_VOCAB_SIZE,
    KEY_ENC_N_LAYER,
    KEY_ENC_D_MODEL,
    KEY_ENC_N_HEAD,
    KEY_ENC_FFN_DIM,
    KEY_DEC_N_LAYER,
    KEY_DEC_D_MODEL,
    KEY_DEC_N_HEAD,
    KEY_DEC_FFN_DIM,
    KEY_ENC_KERNEL_SIZE,
    KEY_BLANK_ID,
    KEY_SOS_ID,
    KEY_EOS_ID,
    KEY_PAD_ID,
];

/// Optional `Array<String>` metadata key: the exact tensor names the
/// producer wrote.
///
/// When present, [`FireredAsrAed::from_gguf`] verifies every listed name
/// is in the manifest and fails loud naming the first absent one. This
/// turns a truncated / mis-merged / partially-uploaded GGUF into a
/// **load-time** failure instead of a surprise halfway through a
/// forward.
///
/// Absent → skipped for minimal inspection fixtures. The VAST converter
/// supplies this array from the audited prepared artifact; the binder still
/// treats the names as an input manifest and does not guess field mappings.
pub const KEY_REQUIRED_TENSORS: &str = "vokra.firered_asr_aed_l.required_tensors";

/// Optional Array<String> declaration carrying the exact prepared tensor
/// contract (`name|dtype-tag|dim,dim,...`).  The converter emits this beside
/// [`KEY_REQUIRED_TENSORS`]; the binder compares every row with the GGUF
/// tensor descriptor and rejects missing, extra, shape, or dtype drift.
pub const KEY_TENSOR_MANIFEST: &str = "vokra.firered_asr_aed_l.tensor_manifest";

// ---------------------------------------------------------------------------
// Metadata read helpers.
// ---------------------------------------------------------------------------

/// `true` when **any** key of an all-or-nothing group is present.
fn group_present(gguf: &GgufFile, keys: &[&str]) -> bool {
    keys.iter().any(|k| gguf.get(k).is_some())
}

/// Reads a required unsigned-integer key, refusing a wrong value type
/// rather than coercing it (FR-EX-08).
fn read_u32_key(gguf: &GgufFile, key: &str) -> Result<u32> {
    let raw = gguf.get(key).and_then(|v| v.as_u64()).ok_or_else(|| {
        VokraError::ModelLoad(format!(
            "firered-asr-aed-l: GGUF metadata `{key}` is missing or is not an \
             unsigned integer. The `vokra.firered_asr_aed_l.*` group is \
             all-or-nothing — a partially stamped group is a bug in \
             `{CONVERTER_PATH}` (or in `{SIDECAR_PATH}`), and silently defaulting \
             the missing half would build a wrong-shaped encoder / decoder that \
             still runs (FR-EX-08)."
        ))
    })?;
    u32::try_from(raw).map_err(|_| {
        VokraError::ModelLoad(format!(
            "firered-asr-aed-l: GGUF metadata `{key}` = {raw} does not fit in u32"
        ))
    })
}

/// Reads the optional [`KEY_REQUIRED_TENSORS`] declaration.
///
/// Returns `Ok(None)` when the key is absent. Refuses a wrong container
/// type, a wrong element type, a non-string element, or an empty list —
/// an empty declaration asserts nothing and is always a producer bug.
fn read_required_tensors(gguf: &GgufFile) -> Result<Option<Vec<String>>> {
    let Some(value) = gguf.get(KEY_REQUIRED_TENSORS) else {
        return Ok(None);
    };
    let arr = value.as_array().ok_or_else(|| {
        VokraError::ModelLoad(format!(
            "firered-asr-aed-l: GGUF metadata `{KEY_REQUIRED_TENSORS}` is not an \
             array (expected Array<String> naming the tensors the producer wrote), \
             got {:?}",
            value.value_type()
        ))
    })?;
    if arr.element_type != GgufValueType::String {
        return Err(VokraError::ModelLoad(format!(
            "firered-asr-aed-l: GGUF metadata `{KEY_REQUIRED_TENSORS}` has \
             element_type {:?}, expected String",
            arr.element_type
        )));
    }
    let mut out = Vec::with_capacity(arr.values.len());
    let mut seen = std::collections::BTreeSet::new();
    for (i, v) in arr.values.iter().enumerate() {
        match v {
            GgufMetadataValue::String(s) if !s.is_empty() && seen.insert(s.clone()) => {
                out.push(s.clone())
            }
            other => {
                return Err(VokraError::ModelLoad(format!(
                    "firered-asr-aed-l: GGUF metadata `{KEY_REQUIRED_TENSORS}[{i}]` \
                     is not a unique non-empty string (got {:?})",
                    other.value_type()
                )));
            }
        }
    }
    if out.is_empty() {
        return Err(VokraError::ModelLoad(format!(
            "firered-asr-aed-l: GGUF metadata `{KEY_REQUIRED_TENSORS}` is an empty \
             list — an empty required-tensor declaration asserts nothing, so \
             stamping it is always a producer bug. Omit the key entirely, or list \
             the tensor names `{CONVERTER_PATH}` actually wrote (FR-EX-08)."
        )));
    }
    Ok(Some(out))
}

/// Validates the converter's strict name/dtype/shape sidecar against the
/// actual GGUF tensor table.  The sidecar is intentionally a flat string
/// array so it remains a zero-dependency GGUF metadata value; it is not a
/// substitute for checking the descriptors themselves.
fn validate_tensor_manifest(gguf: &GgufFile, required: Option<&[String]>) -> Result<()> {
    let Some(value) = gguf.get(KEY_TENSOR_MANIFEST) else {
        if required.is_some_and(|names| names.len() == 940) {
            return Err(VokraError::ModelLoad(format!(
                "firered-asr-aed-l: `{KEY_TENSOR_MANIFEST}` is required when the authenticated 940-tensor declaration is present"
            )));
        }
        return Ok(());
    };
    let array = value.as_array().ok_or_else(|| {
        VokraError::ModelLoad(format!(
            "firered-asr-aed-l: `{KEY_TENSOR_MANIFEST}` must be Array<String>, got {:?}",
            value.value_type()
        ))
    })?;
    if array.element_type != GgufValueType::String || array.values.is_empty() {
        return Err(VokraError::ModelLoad(format!(
            "firered-asr-aed-l: `{KEY_TENSOR_MANIFEST}` must be a non-empty Array<String>"
        )));
    }
    if array.values.len() != gguf.tensors().len() {
        return Err(VokraError::ModelLoad(format!(
            "firered-asr-aed-l: tensor manifest count {} does not match GGUF tensor count {}",
            array.values.len(),
            gguf.tensors().len()
        )));
    }
    let mut seen = std::collections::BTreeSet::new();
    for (index, value) in array.values.iter().enumerate() {
        let GgufMetadataValue::String(encoded) = value else {
            return Err(VokraError::ModelLoad(format!(
                "firered-asr-aed-l: `{KEY_TENSOR_MANIFEST}[{index}]` is not a string"
            )));
        };
        let mut fields = encoded.splitn(3, '|');
        let Some(name) = fields.next().filter(|name| !name.is_empty()) else {
            return Err(VokraError::ModelLoad(format!(
                "firered-asr-aed-l: `{KEY_TENSOR_MANIFEST}[{index}]` has an empty name"
            )));
        };
        let dtype = fields.next().and_then(|tag| tag.parse::<u32>().ok());
        let dims = fields.next();
        let Some(dtype) = dtype else {
            return Err(VokraError::ModelLoad(format!(
                "firered-asr-aed-l: `{KEY_TENSOR_MANIFEST}[{index}]` has an invalid dtype tag"
            )));
        };
        if required.is_some_and(|names| names.len() == 940) && dtype != 0 {
            return Err(VokraError::ModelLoad(format!(
                "firered-asr-aed-l: authenticated FireRedASR-AED-L tensor `{name}` has dtype tag {dtype}; prepared release requires F32"
            )));
        }
        let Some(dims) = dims else {
            return Err(VokraError::ModelLoad(format!(
                "firered-asr-aed-l: `{KEY_TENSOR_MANIFEST}[{index}]` has no shape"
            )));
        };
        let dims: Result<Vec<u64>> = if dims.is_empty() {
            Ok(Vec::new())
        } else {
            dims.split(',')
                .map(|dim| {
                    dim.parse::<u64>().map_err(|_| {
                        VokraError::ModelLoad(format!(
                            "firered-asr-aed-l: `{KEY_TENSOR_MANIFEST}[{index}]` has invalid shape"
                        ))
                    })
                })
                .collect()
        };
        let dims = dims?;
        if !seen.insert(name.to_owned()) {
            return Err(VokraError::ModelLoad(format!(
                "firered-asr-aed-l: duplicate tensor `{name}` in `{KEY_TENSOR_MANIFEST}`"
            )));
        }
        let actual = gguf.tensor_info(name).ok_or_else(|| {
            VokraError::ModelLoad(format!(
                "firered-asr-aed-l: tensor `{name}` in `{KEY_TENSOR_MANIFEST}` is extra or missing from GGUF"
            ))
        })?;
        if actual.dtype.tag() != dtype || actual.dimensions != dims {
            return Err(VokraError::ModelLoad(format!(
                "firered-asr-aed-l: tensor `{name}` shape/dtype mismatch: manifest tag {dtype}, dims {dims:?}; GGUF tag {}, dims {:?}",
                actual.dtype.tag(),
                actual.dimensions
            )));
        }
    }
    for actual in gguf.tensors() {
        if !seen.contains(&actual.name) {
            return Err(VokraError::ModelLoad(format!(
                "firered-asr-aed-l: GGUF tensor `{}` is extra and absent from `{KEY_TENSOR_MANIFEST}`",
                actual.name
            )));
        }
    }
    if let Some(required) = required {
        if required.len() != seen.len() || required.iter().any(|name| !seen.contains(name)) {
            return Err(VokraError::ModelLoad(format!(
                "firered-asr-aed-l: `{KEY_REQUIRED_TENSORS}` and `{KEY_TENSOR_MANIFEST}` disagree"
            )));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Config — the optional `vokra.firered_asr_aed_l.*` hyper-parameter group.
// ---------------------------------------------------------------------------

/// Acoustic-encoder hyper-parameters of the AED.
///
/// A **separate public type** on purpose: if a real-checkpoint
/// transcription later shows the FireRedTeam AED and LLM releases share
/// an acoustic encoder, this half can be lifted into a shared type by a
/// move rather than a rewrite. This module does not assert that they
/// share one — see the module docstring's "Structuring for the LLM
/// sibling" section.
///
/// No field carries a default: Vokra has no primary-source transcription
/// of FireRedASR-AED-L's real values, and inventing one would be the
/// exact silent-wrong failure this module refuses (CLAUDE.md
/// ハルシネーション厳禁).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FireredAsrAedEncoderConfig {
    /// Encoder depth in blocks ([`KEY_ENC_N_LAYER`]).
    pub n_layer: u32,
    /// Encoder residual width ([`KEY_ENC_D_MODEL`]).
    pub d_model: u32,
    /// Encoder attention-head count ([`KEY_ENC_N_HEAD`]).
    pub n_head: u32,
    /// Encoder feed-forward inner width ([`KEY_ENC_FFN_DIM`]).
    pub ffn_dim: u32,
}

impl FireredAsrAedEncoderConfig {
    /// Per-head attention width (`d_model / n_head`).
    ///
    /// Every field is public, so a caller can hand-build a config that
    /// never went through [`FireredAsrAedConfig::validate`]. The
    /// `n_head == 0` arm returns `0` rather than dividing — a diagnostic
    /// accessor must never be the thing that panics, least of all inside
    /// [`forward_loud_partial`], whose whole job is to report a problem.
    /// A validated config can never take that arm.
    #[inline]
    #[must_use]
    pub const fn head_dim(&self) -> u32 {
        // `match` rather than `unwrap_or`, which is not const-callable.
        match self.d_model.checked_div(self.n_head) {
            Some(v) => v,
            None => 0,
        }
    }
}

/// Transformer-decoder hyper-parameters of the AED.
///
/// The decoder is the half that genuinely differs from the FireRedTeam
/// LLM sibling, which replaces it with a Qwen2 language model plus an
/// audio-to-text adapter. Kept separate from
/// [`FireredAsrAedEncoderConfig`] for the same reason.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FireredAsrAedDecoderConfig {
    /// Decoder depth in blocks ([`KEY_DEC_N_LAYER`]).
    pub n_layer: u32,
    /// Decoder residual width ([`KEY_DEC_D_MODEL`]). Need not equal the
    /// encoder width — see [`KEY_DEC_D_MODEL`].
    pub d_model: u32,
    /// Decoder attention-head count ([`KEY_DEC_N_HEAD`]).
    pub n_head: u32,
    /// Decoder feed-forward inner width ([`KEY_DEC_FFN_DIM`]).
    pub ffn_dim: u32,
}

impl FireredAsrAedDecoderConfig {
    /// Per-head attention width (`d_model / n_head`).
    ///
    /// Returns `0` when `n_head == 0` for the same
    /// never-panic-in-a-diagnostic reason as
    /// [`FireredAsrAedEncoderConfig::head_dim`].
    #[inline]
    #[must_use]
    pub const fn head_dim(&self) -> u32 {
        match self.d_model.checked_div(self.n_head) {
            Some(v) => v,
            None => 0,
        }
    }
}

/// FireRedASR-AED-L hyper-parameters, read from the optional
/// all-or-nothing `vokra.firered_asr_aed_l.*` group.
///
/// The VAST converter stamps these sixteen geometry and special-id keys for the authenticated
/// release. A hand-built inspection fixture may omit the group, in which case
/// [`FireredAsrAed::config`] remains `None` and the sample-rate guard cannot
/// invent an expected rate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FireredAsrAedConfig {
    /// Sample rate the checkpoint expects, in Hz ([`KEY_SAMPLE_RATE`]).
    pub sample_rate: u32,
    /// Log-mel band count of the front-end ([`KEY_N_MELS`]).
    pub n_mels: u32,
    /// Decoder output-head vocabulary size ([`KEY_VOCAB_SIZE`]).
    pub vocab_size: u32,
    /// Decoder special-token ids, copied from authenticated checkpoint args.
    pub blank_id: u32,
    pub sos_id: u32,
    pub eos_id: u32,
    pub pad_id: u32,
    /// Acoustic encoder geometry.
    pub encoder: FireredAsrAedEncoderConfig,
    /// Transformer decoder geometry.
    pub decoder: FireredAsrAedDecoderConfig,
    /// Conformer depthwise-convolution kernel width.
    pub kernel_size: u32,
}

impl FireredAsrAedConfig {
    /// Validates the group loudly (FR-EX-08).
    ///
    /// Two classes of check, both universal to any attention
    /// encoder-decoder and therefore safe to assert **without** a
    /// FireRedASR-specific transcription:
    ///
    /// 1. every geometry field must be `> 0` — a `0` is the classic
    ///    half-populated-metadata sentinel, and a zero width / depth /
    ///    vocabulary collapses the whole pipeline. Special-token ids are
    ///    allowed to be zero (the authenticated blank id) but must be inside
    ///    the vocabulary range;
    /// 2. `d_model % n_head == 0` on **both** stacks — multi-head
    ///    attention splits the model width across heads, so an
    ///    indivisible pair can only come from a mis-stamp.
    ///
    /// Deliberately **not** asserted: `encoder.d_model ==
    /// decoder.d_model`. In an AED the cross-attention K/V projections
    /// map the encoder width into the decoder width, so the two are
    /// independent axes; requiring equality would be an invented
    /// constraint that could reject a legitimate checkpoint.
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] naming the offending key.
    pub fn validate(&self) -> Result<()> {
        for (key, value) in [
            (KEY_SAMPLE_RATE, self.sample_rate),
            (KEY_N_MELS, self.n_mels),
            (KEY_VOCAB_SIZE, self.vocab_size),
            (KEY_ENC_N_LAYER, self.encoder.n_layer),
            (KEY_ENC_D_MODEL, self.encoder.d_model),
            (KEY_ENC_N_HEAD, self.encoder.n_head),
            (KEY_ENC_FFN_DIM, self.encoder.ffn_dim),
            (KEY_DEC_N_LAYER, self.decoder.n_layer),
            (KEY_DEC_D_MODEL, self.decoder.d_model),
            (KEY_DEC_N_HEAD, self.decoder.n_head),
            (KEY_DEC_FFN_DIM, self.decoder.ffn_dim),
            (KEY_ENC_KERNEL_SIZE, self.kernel_size),
        ] {
            if value == 0 {
                return Err(VokraError::ModelLoad(format!(
                    "firered-asr-aed-l: `{key}` = 0 — every \
                     `vokra.firered_asr_aed_l.*` hyper-parameter must be positive. \
                     A zero is the classic half-populated-metadata sentinel; \
                     accepting it would build a collapsed encoder / decoder that \
                     still runs (FR-EX-08)."
                )));
            }
        }
        for (key, value) in [
            (KEY_BLANK_ID, self.blank_id),
            (KEY_SOS_ID, self.sos_id),
            (KEY_EOS_ID, self.eos_id),
            (KEY_PAD_ID, self.pad_id),
        ] {
            if value >= self.vocab_size {
                return Err(VokraError::ModelLoad(format!(
                    "firered-asr-aed-l: `{key}` = {value} is outside vocabulary size `{KEY_VOCAB_SIZE}` = {}",
                    self.vocab_size
                )));
            }
        }
        if self.kernel_size % 2 == 0 {
            return Err(VokraError::ModelLoad(format!(
                "firered-asr-aed-l: `{KEY_ENC_KERNEL_SIZE}` = {} must be odd for symmetric Conformer padding",
                self.kernel_size
            )));
        }
        for (stack, d_key, d, h_key, h) in [
            (
                "encoder",
                KEY_ENC_D_MODEL,
                self.encoder.d_model,
                KEY_ENC_N_HEAD,
                self.encoder.n_head,
            ),
            (
                "decoder",
                KEY_DEC_D_MODEL,
                self.decoder.d_model,
                KEY_DEC_N_HEAD,
                self.decoder.n_head,
            ),
        ] {
            if d % h != 0 {
                return Err(VokraError::ModelLoad(format!(
                    "firered-asr-aed-l: `{d_key}` = {d} is not divisible by \
                     `{h_key}` = {h} (the {stack} stack) — multi-head attention \
                     splits the model width evenly across heads, so an indivisible \
                     pair can only come from a mis-stamped group. Refusing rather \
                     than truncating the per-head width (FR-EX-08)."
                )));
            }
        }
        Ok(())
    }

    /// Reads the group from a parsed GGUF.
    ///
    /// Returns `Ok(None)` when **no** key of the group is present — the
    /// state of minimal inspection fixtures. Returns a loud
    /// [`VokraError::ModelLoad`] when the group is only partially
    /// stamped, when a value has the wrong type, or when
    /// [`Self::validate`] fails.
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] on a partial group, a wrong value
    ///   type, or a failed validation.
    pub fn from_gguf(gguf: &GgufFile) -> Result<Option<Self>> {
        if !group_present(gguf, &FIREREDASRAED_SPEC_KEYS) {
            return Ok(None);
        }
        let cfg = Self {
            sample_rate: read_u32_key(gguf, KEY_SAMPLE_RATE)?,
            n_mels: read_u32_key(gguf, KEY_N_MELS)?,
            vocab_size: read_u32_key(gguf, KEY_VOCAB_SIZE)?,
            blank_id: read_u32_key(gguf, KEY_BLANK_ID)?,
            sos_id: read_u32_key(gguf, KEY_SOS_ID)?,
            eos_id: read_u32_key(gguf, KEY_EOS_ID)?,
            pad_id: read_u32_key(gguf, KEY_PAD_ID)?,
            encoder: FireredAsrAedEncoderConfig {
                n_layer: read_u32_key(gguf, KEY_ENC_N_LAYER)?,
                d_model: read_u32_key(gguf, KEY_ENC_D_MODEL)?,
                n_head: read_u32_key(gguf, KEY_ENC_N_HEAD)?,
                ffn_dim: read_u32_key(gguf, KEY_ENC_FFN_DIM)?,
            },
            decoder: FireredAsrAedDecoderConfig {
                n_layer: read_u32_key(gguf, KEY_DEC_N_LAYER)?,
                d_model: read_u32_key(gguf, KEY_DEC_D_MODEL)?,
                n_head: read_u32_key(gguf, KEY_DEC_N_HEAD)?,
                ffn_dim: read_u32_key(gguf, KEY_DEC_FFN_DIM)?,
            },
            kernel_size: read_u32_key(gguf, KEY_ENC_KERNEL_SIZE)?,
        };
        cfg.validate()?;
        Ok(Some(cfg))
    }
}

// ---------------------------------------------------------------------------
// Weights — tensor manifest with a non-empty gate.
// ---------------------------------------------------------------------------

/// Weight tensors bound from a FireRedASR-AED-L GGUF.
///
/// Names + GGUF-side dims only. The forward is a loud-partial, so
/// preloading every tensor into RAM would buy nothing; the follow-up
/// wave that lands the real forward picks its own caching shape. What
/// this struct *does* buy today are the loud gates the module docstring
/// promises: the non-empty manifest check, the by-name lookup that names
/// an absent tensor, and the optional producer-declared required-tensor
/// check.
#[derive(Debug)]
pub struct FireredAsrAedWeights {
    tensors: Vec<(String, Vec<usize>)>,
}

impl FireredAsrAedWeights {
    /// Scans `gguf` for the FireRedASR-AED-L `state_dict` tensors,
    /// refusing an empty manifest (FR-EX-08).
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
                "firered-asr-aed-l: GGUF carries zero tensors — refusing to bind an \
                 all-zero forward (FR-EX-08). A legitimate FireRedASR-AED-L \
                 checkpoint is ~1.1 B parameters and carries the whole acoustic \
                 encoder stack, the Transformer decoder stack with its \
                 cross-attention projections, and the vocabulary head \
                 (arch={ARCH}, name={NAME}); zero tensors always signals a \
                 mis-produced GGUF. Re-run `vokra-cli convert --model \
                 firered-asr-aed-l` against an upstream `{UPSTREAM_HF}` checkpoint \
                 bridged through `{SIDECAR_PATH}`."
            )));
        }
        Ok(Self { tensors })
    }

    /// Number of tensors bound from the GGUF.
    #[inline]
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    /// The bound tensor names with their GGUF-side dims — a diagnostic
    /// accessor for the follow-up forward wave.
    #[inline]
    #[must_use]
    pub fn tensors(&self) -> &[(String, Vec<usize>)] {
        &self.tensors
    }

    /// `true` when `name` is present in the bound manifest.
    #[must_use]
    pub fn has(&self, name: &str) -> bool {
        self.tensors.iter().any(|(n, _)| n.as_str() == name)
    }

    /// Looks a tensor's GGUF-side dims up by name.
    ///
    /// Returns a loud [`VokraError::ModelLoad`] **naming the absent
    /// tensor** rather than `None`, so a caller cannot swallow the miss
    /// with `unwrap_or_default()` and silently proceed on an implicit
    /// zero shape (FR-EX-08).
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] when `name` is not in the manifest.
    pub fn dims(&self, name: &str) -> Result<&[usize]> {
        self.tensors
            .iter()
            .find(|(n, _)| n.as_str() == name)
            .map(|(_, d)| d.as_slice())
            .ok_or_else(|| {
                VokraError::ModelLoad(format!(
                    "firered-asr-aed-l: tensor `{name}` is absent from the GGUF \
                     manifest ({count} tensors present). FireRedASR-AED-L GGUFs \
                     carry the upstream safetensors names verbatim (see \
                     `{CONVERTER_PATH}`), so a miss means either a mis-produced \
                     GGUF or a stale name in the caller. The converter preserves \
                     audited upstream names verbatim, but this runtime has not \
                     mapped every name to a native field yet (FR-EX-08 — no \
                     silent zero-shape fallback).",
                    count = self.tensors.len()
                ))
            })
    }

    /// Verifies every name in a [`KEY_REQUIRED_TENSORS`] declaration is
    /// present, failing loud on the **first** absent one.
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] naming the first missing tensor.
    pub fn require_all(&self, names: &[String]) -> Result<()> {
        for name in names {
            if !self.has(name) {
                return Err(VokraError::ModelLoad(format!(
                    "firered-asr-aed-l: required tensor `{name}` is declared in \
                     `{KEY_REQUIRED_TENSORS}` but absent from the GGUF manifest \
                     ({count} tensors present, {declared} declared). The producer \
                     asserted it wrote this tensor, so the GGUF is truncated, \
                     mis-merged or partially uploaded — refusing at load time \
                     rather than surprising a forward halfway through (FR-EX-08).",
                    count = self.tensors.len(),
                    declared = names.len(),
                )));
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// FireredAsrAed — the runtime binder handle.
// ---------------------------------------------------------------------------

/// FireRedASR-AED-L (`FireRedTeam/FireRedASR-AED-L`, Apache-2.0) runtime
/// binder — the consumer of the `firered_asr_aed_l` converter arch.
///
/// Bind with [`from_gguf`](Self::from_gguf) / [`from_path`](Self::from_path),
/// then transcribe either through the inherent
/// [`transcribe_tokens`](Self::transcribe_tokens) (which carries an
/// explicit sample rate and therefore a rate guard) or through the
/// generic [`AsrEngine`] trait, so `vokra-core`'s session glue sees no
/// FireRedASR-vs-Whisper-vs-Canary asymmetry.
///
/// Both entry points are loud-partials today; see the module doc for
/// exactly which four pieces are missing and why guessing them would be
/// silent-wrong.
#[derive(Debug)]
pub struct FireredAsrAed {
    /// The `vokra.firered_asr_aed_l.*` group when stamped. The VAST converter
    /// stamps it for the authenticated release; minimal inspection fixtures
    /// may omit it — see [`FireredAsrAedConfig`].
    cfg: Option<FireredAsrAedConfig>,
    weights: FireredAsrAedWeights,
    weight_license: LicenseClass,
    has_tokenizer: bool,
}

impl FireredAsrAed {
    /// Binds a FireRedASR-AED-L GGUF: verifies the arch tag strictly,
    /// binds the tensor manifest, honours an optional required-tensor
    /// declaration, reads the optional `vokra.firered_asr_aed_l.*`
    /// group, and surfaces the stamped weight-license class plus whether
    /// a tokenizer blob rides along.
    ///
    /// Every failure is a distinct [`VokraError::ModelLoad`] naming the
    /// missing / wrong key or tensor, so a reader diagnosing a
    /// mis-produced GGUF has exactly one place to walk (FR-EX-08 — never
    /// a silent partial bind).
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] when `vokra.model.arch` is absent or
    ///   is not [`ARCH`] (a sibling `category = "asr"` GGUF handed here
    ///   by mistake — above all the FireRedTeam LLM release — fails with
    ///   a specific message rather than a downstream missing-tensor
    ///   error);
    /// - [`VokraError::ModelLoad`] when the GGUF carries zero tensors;
    /// - [`VokraError::ModelLoad`] when a [`KEY_REQUIRED_TENSORS`]
    ///   declaration names a tensor that is not in the manifest;
    /// - [`VokraError::ModelLoad`] when the `vokra.firered_asr_aed_l.*`
    ///   group is partially stamped or fails
    ///   [`FireredAsrAedConfig::validate`].
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        // 1. Arch check first, always — a `firered_asr_llm_l` / `whisper`
        //    / `canary` / `parakeet-ctc` GGUF handed here by mistake must
        //    fail with a clear message, not a downstream shape error.
        match file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()) {
            Some(a) if a == ARCH => {}
            Some(other) => {
                return Err(VokraError::ModelLoad(format!(
                    "firered-asr-aed-l: GGUF arch is `{other}`, expected `{ARCH}` \
                     (was this GGUF produced by `vokra-cli convert --model \
                     firered-asr-aed-l`? The sibling `category = \"asr\"` arch tags \
                     all share this category and several share the \
                     attention-encoder-decoder family, but never a tensor manifest: \
                     `firered_asr_llm_l` (THE SAME TEAM's other release — Conformer \
                     encoder + linear/MLP audio-text adapter + a Qwen2 LM decoder, \
                     ~16.6 GB; the single most likely mis-dispatch), `whisper` / \
                     `distil-whisper` / `kotoba-whisper` (OpenAI Whisper and its \
                     distilled / Japanese-tuned derivatives — also AEDs, but with \
                     Whisper's own hparams and BPE vocabulary, which the audit \
                     ticket `{AUDIT_TICKET_PATH}` explicitly records \
                     FireRedASR-AED-L does NOT share), `canary` / `canary-1b-flash` \
                     / `canary-qwen` (NVIDIA NeMo FastConformer encoders with an \
                     AED, a faster AED, and a Qwen LM decoder respectively), \
                     `parakeet-tdt` / `parakeet-ctc` / `omniasr-ctc` (CTC / \
                     RNN-T-TDT heads with no attention decoder at all), \
                     `kyutai-stt` / `voxtral` / `nemotron_asr_streaming` (streaming \
                     and LLM-decoder ASR with their own state and prompt \
                     contracts), `moonshine` (variable-length-input \
                     encoder-decoder). The category tag alone can never \
                     disambiguate them — only the arch tag can; silently aliasing \
                     it would mis-route runtime dispatch (FR-EX-08 — no silent \
                     partial load)."
                )));
            }
            None => {
                return Err(VokraError::ModelLoad(format!(
                    "firered-asr-aed-l: GGUF is missing `vokra.model.arch` — this \
                     is not a Vokra-native firered_asr_aed_l GGUF (was it produced \
                     by `vokra-cli convert --model firered-asr-aed-l`? that \
                     converter, `{CONVERTER_PATH}`, stamps `vokra.model.arch = \
                     {ARCH}`)."
                )));
            }
        }

        // 2. Tensor manifest with the non-emptiness gate, then the
        //    optional producer-declared required-tensor check.
        let weights = FireredAsrAedWeights::from_gguf(file)?;
        let required = read_required_tensors(file)?;
        if let Some(required) = required.as_ref() {
            weights.require_all(required)?;
        }
        validate_tensor_manifest(file, required.as_deref())?;

        // 3. The optional all-or-nothing hyper-parameter group. `None` is
        //    accepted for minimal inspection fixtures; the VAST converter
        //    stamps the authenticated release geometry.
        let cfg = FireredAsrAedConfig::from_gguf(file)?;

        // 4. Provenance surfacing. The converter stamps `apache-2.0` →
        //    Permissive by default; a GGUF with no stamp fail-closes to
        //    Unknown (memory `[[feedback-license-signoff-primary-source]]`).
        let weight_license = file
            .get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
            .and_then(|v| v.as_str())
            .and_then(LicenseClass::from_class_str)
            .unwrap_or(LicenseClass::Unknown);

        // 5. Tokenizer presence — surfaced, never required. The current
        //    converter does not embed a tokenizer blob; that is loud-partial
        //    blocker (3) and is reported in the forward's error.
        let has_tokenizer = file.get(KEY_TOKENIZER_MODEL).is_some();

        Ok(Self {
            cfg,
            weights,
            weight_license,
            has_tokenizer,
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

    /// The `vokra.firered_asr_aed_l.*` hyper-parameter group, when
    /// stamped.
    ///
    /// `None` for minimal inspection fixtures; converted release artifacts
    /// carry the authenticated geometry group.
    // Deliberately not `const fn`: `Option::as_ref` in a const context is
    // newer than this workspace's MSRV floor is worth betting on, and no
    // caller needs a const config accessor.
    #[inline]
    #[must_use]
    pub fn config(&self) -> Option<&FireredAsrAedConfig> {
        self.cfg.as_ref()
    }

    /// The bound tensor manifest.
    #[inline]
    #[must_use]
    pub const fn weights(&self) -> &FireredAsrAedWeights {
        &self.weights
    }

    /// Number of tensors bound from the GGUF.
    #[inline]
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.weights.tensor_count()
    }

    /// The stamped weight-license class.
    ///
    /// The converter stamps [`DEFAULT_LICENSE_SPDX`] (`apache-2.0`) →
    /// [`LicenseClass::Permissive`] by default; a GGUF without the stamp
    /// reads back as [`LicenseClass::Unknown`] (fail-closed at the M2-13
    /// compliance gate). This is a *surface*, not a sign-off — see the
    /// module docstring's "Licensing" section.
    #[inline]
    #[must_use]
    pub const fn weight_license(&self) -> LicenseClass {
        self.weight_license
    }

    /// `true` when the bound weights may only be used behind a research
    /// flag. An unstamped GGUF answers `true` (fail-closed).
    #[inline]
    #[must_use]
    pub fn is_research_only(&self) -> bool {
        self.weight_license.requires_research_flag()
    }

    /// `true` when the GGUF carries a [`KEY_TOKENIZER_MODEL`] blob.
    ///
    /// Today's converter never writes one, so this is `false` for every
    /// GGUF it produces. It is surfaced rather than required because a
    /// missing tokenizer is a *rendering* blocker, not a *binding* one:
    /// the weights are still legitimately bound and inspectable.
    #[inline]
    #[must_use]
    pub const fn has_tokenizer(&self) -> bool {
        self.has_tokenizer
    }

    /// Transcribes mono `f32` PCM at `sample_rate` Hz to decoder token
    /// ids in the `vokra.firered_asr_aed_l.vocab_size` id space.
    ///
    /// Token ids rather than text: rendering Mandarin text needs the
    /// upstream vocabulary, which today's GGUFs do not carry (see
    /// [`has_tokenizer`](Self::has_tokenizer)).
    ///
    /// # Loud-partial (this WP)
    ///
    /// Returns [`VokraError::UnsupportedOp`]. FireRedASR-AED-L's
    /// exact native frontend/weight mapping and vocabulary are not yet
    /// independently authenticated —
    /// see [`forward_loud_partial`] for the full message and the
    /// flip-the-switch recipe. **No fabricated token ids are ever
    /// emitted** (FR-EX-08).
    ///
    /// # Errors
    ///
    /// - [`VokraError::InvalidArgument`] when `pcm` is empty;
    /// - [`VokraError::InvalidArgument`] when the
    ///   `vokra.firered_asr_aed_l.*` group is stamped and `sample_rate`
    ///   differs from its [`FireredAsrAedConfig::sample_rate`] — Vokra
    ///   never silently resamples (FR-EX-08);
    /// - [`VokraError::UnsupportedOp`] — the loud-partial gate.
    pub fn transcribe_tokens(&self, pcm: &[f32], sample_rate: u32) -> Result<Vec<u32>> {
        self.transcribe_inner(pcm, Some(sample_rate))
    }

    /// Shared body of the inherent and trait transcription paths.
    ///
    /// `sample_rate` is `None` on the [`AsrEngine`] path, whose signature
    /// carries no rate: rather than invent one (CLAUDE.md
    /// ハルシネーション厳禁) the guard is simply skipped there, and the
    /// rustdoc on the trait impl says so.
    fn transcribe_inner(&self, pcm: &[f32], sample_rate: Option<u32>) -> Result<Vec<u32>> {
        if pcm.is_empty() {
            return Err(VokraError::InvalidArgument(
                "firered-asr-aed-l: empty PCM slice — a transcription needs at least \
                 one sample. Returning an empty token sequence would be \
                 indistinguishable from 'the model heard silence' (FR-EX-08)."
                    .to_owned(),
            ));
        }
        if let Some(rate) = sample_rate {
            check_sample_rate(self.cfg.as_ref(), rate)?;
        }
        // The gate fires BEFORE any front-end work so a caller can never
        // observe a partial computation that looks like a real forward.
        Err(forward_loud_partial(self.cfg.as_ref(), self.has_tokenizer))
    }
}

/// The generic ASR surface, so a FireRedASR-AED-L handle is reachable
/// through `vokra-core`'s session glue exactly like Whisper / Voxtral /
/// distil-Whisper.
///
/// The trait signature carries **no sample rate**, so the rate guard of
/// [`FireredAsrAed::transcribe_tokens`] cannot fire on this path — a
/// caller that needs it must use the inherent method. Everything else is
/// identical, and both paths reach the same loud-partial.
impl AsrEngine for FireredAsrAed {
    fn transcribe(&self, pcm: &[f32]) -> Result<Transcription> {
        // `transcribe_inner` is named distinctly from this trait method so
        // that the call cannot resolve back into itself.
        let ids = self.transcribe_inner(pcm, None)?;
        // Not reached while the forward above is a loud-partial. Kept as
        // the honest shape of the finished composition (token ids ->
        // vocabulary -> text) rather than an `unreachable!()`, so the
        // follow-up wave replaces one expression instead of restructuring
        // the impl — and so the remaining gap (rendering, blocker 3) is
        // stated at the exact place it will be closed.
        Err(VokraError::UnsupportedOp(format!(
            "firered-asr-aed-l transcribe: the decoder produced {n} token ids but \
             they cannot be rendered as text — no `{KEY_TOKENIZER_MODEL}` blob \
             rides on this GGUF (see `{CONVERTER_PATH}`). Use \
             `FireredAsrAed::transcribe_tokens` for the raw ids (FR-EX-08 — no \
             fabricated transcript).",
            n = ids.len()
        )))
    }

    /// Reports `Cpu`. This engine has no backend selector at all — every
    /// transcription entry point is a loud partial (three named blockers),
    /// so no forward executes on any device and the answer cannot
    /// contradict where inference happened. When the forward lands, this
    /// must start reporting the real backend rather than staying a
    /// constant: that is the lie the trait's missing default exists to
    /// prevent.
    fn backend(&self) -> BackendKind {
        BackendKind::Cpu
    }
}

/// Rejects PCM offered at a rate the stamped config does not expect.
///
/// A `None` config (as in a minimal inspection fixture) cannot decide
/// the question, so the guard passes — the forward's loud-partial fires
/// immediately afterwards either way, and inventing an expected rate
/// would be exactly the fabrication this module refuses.
///
/// # Errors
///
/// - [`VokraError::InvalidArgument`] on a mismatch.
fn check_sample_rate(cfg: Option<&FireredAsrAedConfig>, sample_rate: u32) -> Result<()> {
    let Some(cfg) = cfg else {
        return Ok(());
    };
    if cfg.sample_rate != sample_rate {
        return Err(VokraError::InvalidArgument(format!(
            "firered-asr-aed-l: PCM sample rate {sample_rate} Hz does not match the \
             checkpoint's `{KEY_SAMPLE_RATE}` = {expected} Hz. Vokra never silently \
             resamples — resample upstream of this call, or bind a checkpoint that \
             expects {sample_rate} Hz (FR-EX-08).",
            expected = cfg.sample_rate
        )));
    }
    Ok(())
}

/// Constructs the loud-partial [`VokraError::UnsupportedOp`] returned by
/// [`FireredAsrAed::transcribe_tokens`] and the [`AsrEngine`] path until
/// the FireRedASR-AED-L forward lands.
///
/// Names all remaining blockers, reports which of them the GGUF at hand has
/// already cleared, and cites every primary source, so a reader
/// diagnosing the gap has fully specified places to walk. Mirror of the
/// `firered_vad` / `emotion2vec` / `panns` / RMVPE loud-partial-message
/// precedent (CLAUDE.md 教訓 (a)).
#[must_use]
pub fn forward_loud_partial(cfg: Option<&FireredAsrAedConfig>, has_tokenizer: bool) -> VokraError {
    let spec_status = match cfg {
        Some(c) => format!(
            "the `vokra.firered_asr_aed_l.*` group IS stamped on this GGUF \
             (sample_rate={sr} Hz, n_mels={mels}, vocab_size={vocab}, encoder \
             n_layer={el} d_model={ed} n_head={eh} -> head_dim={ehd} \
             ffn_dim={eff}, decoder n_layer={dl} d_model={dd} n_head={dh} -> \
             head_dim={dhd} ffn_dim={dff}), so the source geometry is authenticated \
             but the native frontend remains unimplemented as a complete graph; \
             reusable helpers exist — blockers (2)-(4) are \
             reported below",
            sr = c.sample_rate,
            mels = c.n_mels,
            vocab = c.vocab_size,
            el = c.encoder.n_layer,
            ed = c.encoder.d_model,
            eh = c.encoder.n_head,
            ehd = c.encoder.head_dim(),
            eff = c.encoder.ffn_dim,
            dl = c.decoder.n_layer,
            dd = c.decoder.d_model,
            dh = c.decoder.n_head,
            dhd = c.decoder.head_dim(),
            dff = c.decoder.ffn_dim,
        ),
        None => format!(
            "the `vokra.firered_asr_aed_l.*` group is NOT stamped on this GGUF \
             (this is a minimal inspection fixture; the VAST converter stamps \
             the authenticated release geometry), so gap (1) applies in full"
        ),
    };
    let tokenizer_status = if has_tokenizer {
        format!(
            "a `{KEY_TOKENIZER_MODEL}` blob IS present on this GGUF, so blocker (3) \
             is already cleared for it"
        )
    } else {
        format!(
            "no `{KEY_TOKENIZER_MODEL}` blob is present on this GGUF (the normal \
             state today), so blocker (3) applies in full"
        )
    };
    VokraError::UnsupportedOp(format!(
        "firered-asr-aed-l transcribe (loud-partial): the FireRedASR-AED-L forward \
         is deferred; four pieces must land before real token ids can be emitted. \
         (1) FRONTEND CONTRACT: the all-or-nothing `vokra.firered_asr_aed_l.*` \
         group ({keys:?}) — {spec_status}. The pinned source and VAST evidence \
             authenticate the 80-bin fbank/CMVN rules, and reusable native \
             frontend helpers now exist, but the complete native frontend tap \
             remains outside this runtime contract. \
         (2) MISSING NATIVE TENSOR MAPPING: `{CONVERTER_PATH}` copies every float \
         tensor under its verbatim prepared safetensors name and stamps the full \
         required manifest. The independent upstream dumper authenticates each \
         name's parameter/buffer role, but no native Rust field mapping consumes \
         that evidence yet. \
         (3) MISSING TOKENIZER: an AED decoder emits token ids in a \
         `{KEY_VOCAB_SIZE}`-wide id space. The source-authenticated \
         SentencePiece/TokenDict and dictionary still need a native GGUF \
         binding — {tokenizer_status}. \
             (4) NATIVE OPERATOR GAP: the pinned Conformer uses a Conv2d \
             subsampling stem, relative-position attention, and a \
             source-faithful inference-only Conformer block; CPU/Metal helper \
             routes now exist, but the complete Conformer graph and its 940-field \
             binding remain unimplemented and must be parity-\
             tested before this model can run. \
         Output once real: decoder token ids per utterance, rendered to text only \
         once a tokenizer blob rides along. \
         Primary sources: HF release {hf}, family reference code {code}, in-repo \
         converter contract {CONVERTER_PATH}, in-repo audit ticket \
         {AUDIT_TICKET_PATH}; the offline bridge is {SIDECAR_PATH} and no Python \
         ever enters the runtime (FR-LD-05 / NFR-DS-02). Runtime cannot fabricate \
         a transcription (FR-EX-08 — no silent partial output).",
        keys = FIREREDASRAED_SPEC_KEYS,
        hf = PRIMARY_SOURCE_HF,
        code = PRIMARY_SOURCE_FAMILY_CODE,
    ))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    //! Tests for the FireRedASR-AED-L runtime binder — contract-constant
    //! pins (the cross-crate string handshake with the converter),
    //! metadata round-trip, and negative-space round-trip on every loud
    //! gate.
    //!
    //! # What "round-trip" means here
    //!
    //! On a real checkpoint this would be `transcribe_tokens(...)`
    //! returning decoder token ids. The VAST evidence pins the release
    //! geometry and tensor identity, but the native frontend/decoder and
    //! tokenizer are still deliberately fail-closed; fabricating a token
    //! sequence would violate CLAUDE.md 教訓 (a)「loud-partial は
    //! fake-complete より honest」.
    //!
    //! The round-trip semantics we *can* honestly test:
    //!
    //! 1. **Contract-constant pin** — arch / name / category / upstream
    //!    slug / default SPDX match the converter exactly, and the arch
    //!    is distinct from every sibling ASR tag.
    //! 2. **Metadata round-trip** — `from_gguf` reads arch, tensor
    //!    manifest, licence stamp, tokenizer presence and the optional
    //!    hyper-parameter group with the documented semantics.
    //! 3. **Negative-space round-trip** — every stated blocker (missing
    //!    arch / foreign arch / empty manifest / absent declared tensor /
    //!    partial group / zero sentinel / indivisible heads / mismatched
    //!    rate / empty PCM / deferred forward) fires at its documented
    //!    surface point, in the documented error variant.
    //!
    //! Every numeric value in the fixtures below is **synthetic** — it is
    //! *not* a claim about FireRedASR-AED-L's real geometry, which no
    //! in-repo source states.

    use super::*;
    use vokra_core::gguf::{GgmlType, GgufArray, GgufBuilder};

    /// A synthetic hyper-parameter group. **NOT** FireRedASR-AED-L's real
    /// values — see the module-test doc. `d_model` / `n_head` pairs are
    /// chosen divisible so the happy path passes `validate`.
    const FIXTURE_SPEC: [(&str, u32); 16] = [
        (KEY_SAMPLE_RATE, 16_000),
        (KEY_N_MELS, 80),
        (KEY_VOCAB_SIZE, 7_000),
        (KEY_ENC_N_LAYER, 4),
        (KEY_ENC_D_MODEL, 256),
        (KEY_ENC_N_HEAD, 4),
        (KEY_ENC_FFN_DIM, 1_024),
        (KEY_DEC_N_LAYER, 2),
        (KEY_DEC_D_MODEL, 128),
        (KEY_DEC_N_HEAD, 2),
        (KEY_DEC_FFN_DIM, 512),
        (KEY_ENC_KERNEL_SIZE, 33),
        (KEY_BLANK_ID, 0),
        (KEY_SOS_ID, 3),
        (KEY_EOS_ID, 4),
        (KEY_PAD_ID, 2),
    ];

    /// A representative tensor name. FireRedASR-AED-L GGUFs carry the
    /// upstream safetensors names verbatim; this mirrors the placeholder
    /// the converter's own test module uses so the two files stay legible
    /// together. It is a fixture, not a transcription.
    const FIXTURE_TENSOR: &str = "encoder.blocks.0.attn.qkv_proj.weight";

    /// Builds a base FireRedASR-AED-L GGUF: arch + name + category +
    /// upstream slug, an optional weight-licence stamp, and one
    /// representative tensor so the non-emptiness gate passes.
    fn base_builder(weight_license_class: Option<LicenseClass>) -> GgufBuilder {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_string(chunks::KEY_MODEL_NAME, NAME);
        b.add_string(KEY_MODEL_CATEGORY, CATEGORY);
        b.add_string(KEY_PROVENANCE_UPSTREAM_HF, UPSTREAM_HF);
        if let Some(cls) = weight_license_class {
            b.add_string(chunks::KEY_PROVENANCE_WEIGHT_LICENSE, cls.as_str());
        }
        b.add_tensor(
            FIXTURE_TENSOR,
            GgmlType::F32,
            vec![2, 3],
            vec![0u8; 2 * 3 * 4],
        )
        .expect("add_tensor");
        b
    }

    fn finish(b: &GgufBuilder) -> GgufFile {
        GgufFile::parse(b.to_bytes().expect("serialize")).expect("parse")
    }

    /// A minimal inspection GGUF: arch + provenance + tensors, with no
    /// optional release contract or tokenizer blob.
    fn converter_shaped_gguf() -> GgufFile {
        finish(&base_builder(Some(LicenseClass::Permissive)))
    }

    fn manifest_gguf(entry: &str) -> GgufFile {
        let mut b = base_builder(Some(LicenseClass::Permissive));
        b.add_metadata(
            KEY_TENSOR_MANIFEST,
            GgufMetadataValue::Array(GgufArray {
                element_type: GgufValueType::String,
                values: vec![GgufMetadataValue::String(entry.to_owned())],
            }),
        );
        finish(&b)
    }

    /// A GGUF with the full synthetic hyper-parameter group stamped.
    fn spec_stamped_gguf() -> GgufFile {
        let mut b = base_builder(Some(LicenseClass::Permissive));
        for (key, value) in FIXTURE_SPEC {
            b.add_u32(key, value);
        }
        finish(&b)
    }

    #[test]
    fn tensor_manifest_binds_exact_shape_and_dtype() {
        let good = manifest_gguf("encoder.blocks.0.attn.qkv_proj.weight|0|2,3");
        FireredAsrAed::from_gguf(&good).expect("exact tensor manifest must bind");

        for bad in [
            "encoder.blocks.0.attn.qkv_proj.weight|0|2,4",
            "encoder.blocks.0.attn.qkv_proj.weight|1|2,3",
            "encoder.blocks.0.attn.qkv_proj.bias|0|2,3",
        ] {
            let error = FireredAsrAed::from_gguf(&manifest_gguf(bad))
                .expect_err("shape, dtype, and name drift must fail closed");
            assert!(
                matches!(error, VokraError::ModelLoad(_)),
                "strict manifest error must be ModelLoad: {error:?}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Test 1 — Contract-constant pin (cross-crate handshake with the
    //          converter) + sibling arch-tag distinctness.
    // -----------------------------------------------------------------------

    #[test]
    fn contract_constants_pin_matches_converter() {
        // Mirrors of `crates/vokra-convert/src/models/firered_asr_aed_l.rs`.
        // A converter-side drift without a binder-side follow-through
        // lands here in the same commit or fails this test.
        assert_eq!(ARCH, "firered_asr_aed_l", "arch tag pin (underscores)");
        assert_eq!(NAME, "firered-asr-aed-l", "model name pin (hyphens)");
        assert_eq!(CATEGORY, "asr", "category pin");
        assert_eq!(
            UPSTREAM_HF, "FireRedTeam/FireRedASR-AED-L",
            "upstream HF slug pin"
        );
        assert_eq!(DEFAULT_LICENSE_SPDX, "apache-2.0", "default SPDX pin");

        // The arch / name spellings genuinely differ — a "helpful"
        // normalisation of one into the other would break the wire
        // handshake.
        assert_ne!(ARCH, NAME, "arch uses underscores, name uses hyphens");

        // Distinct from every sibling `category = "asr"` arch tag. The
        // FireRedTeam LLM sibling heads the list: same org, same
        // category, completely different decoder half.
        for sibling in [
            "firered_asr_llm_l",
            "whisper",
            "distil-whisper",
            "kotoba-whisper",
            "canary",
            "canary-1b-flash",
            "canary-qwen",
            "parakeet-tdt",
            "parakeet-ctc",
            "omniasr-ctc",
            "kyutai-stt",
            "voxtral",
            "nemotron_asr_streaming",
            "moonshine",
            // and the same team's VAD release, which shares the org but
            // not the category.
            "firered_vad",
        ] {
            assert_ne!(
                ARCH, sibling,
                "arch tag must stay distinct from sibling `{sibling}`"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Test 2 — Hyper-parameter key group pin: the wire contract a future
    //          converter extension must satisfy.
    // -----------------------------------------------------------------------

    #[test]
    fn spec_key_group_is_well_formed() {
        assert_eq!(
            FIREREDASRAED_SPEC_KEYS.len(),
            16,
            "the all-or-nothing group includes geometry and special-token ids"
        );
        for key in FIREREDASRAED_SPEC_KEYS {
            assert!(
                key.starts_with("vokra.firered_asr_aed_l."),
                "every group key must live under the model's own namespace, got `{key}`"
            );
        }
        // No duplicates — a duplicate would make `group_present` and the
        // read order silently disagree about how many keys exist.
        let mut sorted: Vec<&str> = FIREREDASRAED_SPEC_KEYS.to_vec();
        sorted.sort_unstable();
        let before = sorted.len();
        sorted.dedup();
        assert_eq!(before, sorted.len(), "group keys must be unique");

        // The required-tensor + tokenizer keys are deliberately NOT part
        // of the all-or-nothing group: both are independently optional.
        assert!(!FIREREDASRAED_SPEC_KEYS.contains(&KEY_REQUIRED_TENSORS));
        assert!(!FIREREDASRAED_SPEC_KEYS.contains(&KEY_TOKENIZER_MODEL));
        assert_eq!(KEY_TOKENIZER_MODEL, "vokra.tokenizer.model");
    }

    // -----------------------------------------------------------------------
    // Test 3 — A missing `vokra.model.arch` is loud.
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_rejects_missing_arch() {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_NAME, "some-other-name");
        b.add_tensor("some.tensor", GgmlType::F32, vec![2, 2], vec![0u8; 16])
            .expect("add_tensor");
        let file = finish(&b);

        let Err(err) = FireredAsrAed::from_gguf(&file) else {
            panic!("expected ModelLoad when `vokra.model.arch` is absent");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains("`vokra.model.arch`"),
                    "message must name the missing key, got `{m}`"
                );
                assert!(
                    m.contains("not a Vokra-native firered_asr_aed_l GGUF"),
                    "message must name the missing-arch surface, got `{m}`"
                );
                assert!(
                    m.contains(CONVERTER_PATH),
                    "message must point at the converter, got `{m}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Test 4 — A foreign arch is loud and names BOTH the expected and the
    //          actual tag. Covers the same-team LLM sibling, which is the
    //          most likely mis-dispatch.
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_rejects_foreign_arch() {
        for foreign in ["firered_asr_llm_l", "whisper", "canary", "parakeet-ctc"] {
            let mut b = GgufBuilder::new();
            b.add_string(chunks::KEY_MODEL_ARCH, foreign);
            b.add_string(KEY_MODEL_CATEGORY, "asr");
            b.add_tensor("probe.weight", GgmlType::F32, vec![4, 4], vec![0u8; 64])
                .expect("add_tensor");
            let file = finish(&b);

            let Err(err) = FireredAsrAed::from_gguf(&file) else {
                panic!("expected ModelLoad on foreign arch `{foreign}`");
            };
            match err {
                VokraError::ModelLoad(m) => {
                    assert!(
                        m.contains(foreign),
                        "message must name the ACTUAL arch `{foreign}`, got `{m}`"
                    );
                    assert!(
                        m.contains(ARCH),
                        "message must name the EXPECTED arch `{ARCH}`, got `{m}`"
                    );
                    assert!(
                        m.contains("FR-EX-08"),
                        "message must cite the FR-EX-08 clause, got `{m}`"
                    );
                    // The whole sibling fleet is enumerated so the reader
                    // can tell which loader they actually wanted.
                    for sibling in ["firered_asr_llm_l", "whisper", "canary", "parakeet-ctc"] {
                        assert!(
                            m.contains(sibling),
                            "expected sibling `{sibling}` disambiguation in error: {m}"
                        );
                    }
                    // And the AED-vs-Whisper shape claim is anchored on
                    // the in-repo audit ticket rather than asserted bare.
                    assert!(
                        m.contains(AUDIT_TICKET_PATH),
                        "message must anchor the not-Whisper-compatible claim: {m}"
                    );
                }
                other => panic!("expected VokraError::ModelLoad, got {other:?}"),
            }
        }
    }

    // -----------------------------------------------------------------------
    // Test 5 — A converter-shaped GGUF binds, with the documented
    //          surfaces (metadata round-trip).
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_binds_converter_shaped_gguf() {
        let file = converter_shaped_gguf();
        let m = FireredAsrAed::from_gguf(&file).expect("a converter-shaped GGUF must bind");

        assert_eq!(
            m.weight_license(),
            LicenseClass::Permissive,
            "the converter's apache-2.0 stamp must round-trip as Permissive"
        );
        assert!(!m.is_research_only(), "Permissive is not research-only");
        assert_eq!(
            m.tensor_count(),
            1,
            "the fixture carries exactly one tensor"
        );
        assert!(
            m.config().is_none(),
            "a minimal inspection fixture may omit the optional contract group, \
             and an absent group must NOT be a load failure — refusing it would \
             re-open the inspection/binding gap this module closes"
        );
        assert!(
            !m.has_tokenizer(),
            "today's converter writes no tokenizer blob"
        );
        // The by-name lookup finds the fixture tensor and reports its dims.
        assert_eq!(
            m.weights().dims(FIXTURE_TENSOR).expect("fixture tensor"),
            &[2usize, 3usize],
            "dims must come from the GGUF header, not a guess"
        );
        assert!(m.weights().has(FIXTURE_TENSOR));
        assert_eq!(m.weights().tensors().len(), 1);
    }

    // -----------------------------------------------------------------------
    // Test 6 — An absent licence stamp fail-closes to Unknown.
    // -----------------------------------------------------------------------

    #[test]
    fn absent_license_stamp_fails_closed_to_unknown() {
        let file = finish(&base_builder(None));
        let m = FireredAsrAed::from_gguf(&file).expect("a stampless GGUF still binds");
        assert_eq!(
            m.weight_license(),
            LicenseClass::Unknown,
            "a missing weight-licence stamp must fail closed to Unknown"
        );
        assert!(
            m.is_research_only(),
            "Unknown must be treated as research-only (fail-closed)"
        );
    }

    // -----------------------------------------------------------------------
    // Test 7 — An empty tensor manifest is loud (never an all-zero bind).
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_rejects_empty_tensor_manifest() {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_string(chunks::KEY_MODEL_NAME, NAME);
        b.add_string(chunks::KEY_PROVENANCE_WEIGHT_LICENSE, "permissive");
        // NO tensors added.
        let file = finish(&b);

        let Err(err) = FireredAsrAed::from_gguf(&file) else {
            panic!("expected ModelLoad on an empty tensor manifest");
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
                    m.contains("vokra-cli convert --model firered-asr-aed-l"),
                    "message must include the repro command, got `{m}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Test 8 — The by-name lookup NAMES an absent tensor rather than
    //          returning `None` for a caller to swallow.
    // -----------------------------------------------------------------------

    #[test]
    fn dims_names_the_absent_tensor() {
        const ABSENT: &str = "decoder.blocks.7.cross_attn.k_proj.weight";
        let file = converter_shaped_gguf();
        let m = FireredAsrAed::from_gguf(&file).expect("bind");

        assert!(!m.weights().has(ABSENT));
        let Err(err) = m.weights().dims(ABSENT) else {
            panic!("expected ModelLoad for an absent tensor");
        };
        match err {
            VokraError::ModelLoad(msg) => {
                assert!(
                    msg.contains(ABSENT),
                    "message must name the absent tensor `{ABSENT}`, got `{msg}`"
                );
                assert!(
                    msg.contains("FR-EX-08"),
                    "message must cite the FR-EX-08 clause, got `{msg}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Test 9 — The optional required-tensor declaration: satisfied binds,
    //          absent is loud AT LOAD TIME naming the tensor, empty is
    //          itself loud.
    // -----------------------------------------------------------------------

    #[test]
    fn required_tensor_declaration_edge_cases() {
        const MISSING: &str = "decoder.blocks.3.mlp.fc2.weight";

        // (a) A satisfied declaration binds.
        let mut ok = base_builder(Some(LicenseClass::Permissive));
        ok.add_metadata(
            KEY_REQUIRED_TENSORS,
            GgufMetadataValue::Array(GgufArray {
                element_type: GgufValueType::String,
                values: vec![GgufMetadataValue::String(FIXTURE_TENSOR.to_owned())],
            }),
        );
        FireredAsrAed::from_gguf(&finish(&ok)).expect("a satisfied declaration must bind");

        // (b) A declared-but-absent tensor fails at LOAD time, naming it.
        let mut absent = base_builder(Some(LicenseClass::Permissive));
        absent.add_metadata(
            KEY_REQUIRED_TENSORS,
            GgufMetadataValue::Array(GgufArray {
                element_type: GgufValueType::String,
                values: vec![
                    GgufMetadataValue::String(FIXTURE_TENSOR.to_owned()),
                    GgufMetadataValue::String(MISSING.to_owned()),
                ],
            }),
        );
        let Err(err) = FireredAsrAed::from_gguf(&finish(&absent)) else {
            panic!("expected ModelLoad when a declared tensor is absent");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains(MISSING),
                    "message must name the absent tensor `{MISSING}`, got `{m}`"
                );
                assert!(
                    m.contains(KEY_REQUIRED_TENSORS),
                    "message must name the declaration key, got `{m}`"
                );
                assert!(
                    m.contains("FR-EX-08"),
                    "message must cite the FR-EX-08 clause, got `{m}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }

        // (c) An empty declaration asserts nothing → always a producer bug.
        let mut empty = base_builder(Some(LicenseClass::Permissive));
        empty.add_metadata(
            KEY_REQUIRED_TENSORS,
            GgufMetadataValue::Array(GgufArray {
                element_type: GgufValueType::String,
                values: Vec::new(),
            }),
        );
        let Err(err) = FireredAsrAed::from_gguf(&finish(&empty)) else {
            panic!("expected ModelLoad on an empty required-tensor declaration");
        };
        match err {
            VokraError::ModelLoad(m) => assert!(
                m.contains("empty list") || m.contains("asserts nothing"),
                "message must explain why an empty declaration is a bug, got `{m}`"
            ),
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Test 10 — The hyper-parameter group is all-or-nothing, validated,
    //           and round-trips when complete.
    // -----------------------------------------------------------------------

    #[test]
    fn config_group_is_all_or_nothing_and_validated() {
        // (a) Complete + valid → Some, with both stacks' head_dim derived.
        let m = FireredAsrAed::from_gguf(&spec_stamped_gguf()).expect("bind");
        let cfg = *m.config().expect("a fully stamped group must be read back");
        assert_eq!(cfg.sample_rate, 16_000);
        assert_eq!(cfg.n_mels, 80);
        assert_eq!(cfg.vocab_size, 7_000);
        assert_eq!(cfg.kernel_size, 33);
        assert_eq!(
            (cfg.blank_id, cfg.sos_id, cfg.eos_id, cfg.pad_id),
            (0, 3, 4, 2)
        );
        assert_eq!(cfg.encoder.head_dim(), 64, "256 / 4");
        assert_eq!(cfg.decoder.head_dim(), 64, "128 / 2");
        cfg.validate().expect("the fixture group is valid");
        // Encoder and decoder widths deliberately DIFFER in the fixture:
        // the validator must not have invented an equality constraint.
        assert_ne!(cfg.encoder.d_model, cfg.decoder.d_model);

        // (b) Partial group → loud, naming the first missing key.
        let mut partial = base_builder(Some(LicenseClass::Permissive));
        partial.add_u32(KEY_SAMPLE_RATE, 16_000);
        // ... and nothing else.
        let Err(err) = FireredAsrAed::from_gguf(&finish(&partial)) else {
            panic!("expected ModelLoad on a partially stamped group");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains(KEY_N_MELS),
                    "message must name the first missing key, got `{m}`"
                );
                assert!(
                    m.contains("all-or-nothing"),
                    "message must explain the group contract, got `{m}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }

        // (c) A zero sentinel anywhere in the group is loud, naming the key.
        let mut zeroed = base_builder(Some(LicenseClass::Permissive));
        for (key, value) in FIXTURE_SPEC {
            zeroed.add_u32(key, if key == KEY_DEC_FFN_DIM { 0 } else { value });
        }
        let Err(err) = FireredAsrAed::from_gguf(&finish(&zeroed)) else {
            panic!("expected ModelLoad on a zero sentinel");
        };
        match err {
            VokraError::ModelLoad(m) => assert!(
                m.contains(KEY_DEC_FFN_DIM) && m.contains("must be positive"),
                "message must name the zeroed key, got `{m}`"
            ),
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }

        // (d) An indivisible d_model / n_head pair is loud, naming the stack.
        let mut skewed = base_builder(Some(LicenseClass::Permissive));
        for (key, value) in FIXTURE_SPEC {
            skewed.add_u32(key, if key == KEY_ENC_N_HEAD { 5 } else { value });
        }
        let Err(err) = FireredAsrAed::from_gguf(&finish(&skewed)) else {
            panic!("expected ModelLoad on an indivisible d_model / n_head pair");
        };
        match err {
            VokraError::ModelLoad(m) => assert!(
                m.contains(KEY_ENC_D_MODEL) && m.contains(KEY_ENC_N_HEAD) && m.contains("encoder"),
                "message must name both keys and the offending stack, got `{m}`"
            ),
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Test 11 — The forward is a LOUD-PARTIAL naming every missing piece.
    // -----------------------------------------------------------------------

    #[test]
    fn transcribe_is_a_loud_partial_naming_the_missing_pieces() {
        let file = converter_shaped_gguf();
        let m = FireredAsrAed::from_gguf(&file).expect("bind");

        // One second of silence at 16 kHz — a legitimate PCM shape.
        let pcm = vec![0.0_f32; 16_000];
        let Err(err) = m.transcribe_tokens(&pcm, 16_000) else {
            panic!("transcribe_tokens must loud-partial, never fabricate token ids");
        };
        match err {
            VokraError::UnsupportedOp(msg) => {
                assert!(
                    msg.contains("firered-asr-aed-l transcribe"),
                    "the surface must be named: {msg}"
                );
                assert!(msg.contains("loud-partial"), "posture label: {msg}");

                // Gap (1): the frontend contract, named as a group and
                // reported as unstamped for this minimal fixture.
                assert!(
                    msg.contains("FRONTEND CONTRACT"),
                    "gap (1) must be named: {msg}"
                );
                assert!(
                    msg.contains("is NOT stamped on this GGUF"),
                    "gap (1) must report this GGUF's actual state: {msg}"
                );
                assert!(
                    msg.contains(KEY_ENC_N_HEAD) && msg.contains(KEY_DEC_N_HEAD),
                    "the head-count keys must be named — they are unrecoverable \
                     from the tensor shapes: {msg}"
                );

                // Blocker (2): the tensor-name manifest.
                assert!(
                    msg.contains("MISSING NATIVE TENSOR MAPPING"),
                    "gap (2) must be named: {msg}"
                );

                // Blocker (3): the tokenizer, reported as absent here.
                assert!(
                    msg.contains("MISSING TOKENIZER") && msg.contains(KEY_TOKENIZER_MODEL),
                    "blocker (3) must be named: {msg}"
                );
                assert!(
                    msg.contains("applies in full"),
                    "blocker (3) must report this GGUF's actual state: {msg}"
                );

                // The honest diagnosis: the authenticated topology still has
                // an explicit native operator gap.
                assert!(
                    msg.contains("NATIVE OPERATOR GAP")
                        && msg.contains("Conv2d")
                        && msg.contains("relative-position attention"),
                    "the message must name the exact missing operator class: {msg}"
                );

                // Every primary source is cited so the reader has anchors.
                for anchor in [
                    PRIMARY_SOURCE_HF,
                    PRIMARY_SOURCE_FAMILY_CODE,
                    CONVERTER_PATH,
                    AUDIT_TICKET_PATH,
                    SIDECAR_PATH,
                ] {
                    assert!(
                        msg.contains(anchor),
                        "expected primary-source anchor `{anchor}` cited: {msg}"
                    );
                }

                assert!(
                    msg.contains("FR-EX-08"),
                    "expected the FR-EX-08 no-fabrication rationale: {msg}"
                );
            }
            other => panic!("expected VokraError::UnsupportedOp, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Test 12 — A stamped group reports authenticated geometry but does not
    //           flip the native frontend/operator gate.
    // -----------------------------------------------------------------------

    #[test]
    fn stamped_group_reports_native_gaps_but_still_defers() {
        let m = FireredAsrAed::from_gguf(&spec_stamped_gguf()).expect("bind");
        let pcm = vec![0.0_f32; 1_600];
        let Err(err) = m.transcribe_tokens(&pcm, 16_000) else {
            panic!("a stamped group must NOT be mistaken for a landed forward");
        };
        match err {
            VokraError::UnsupportedOp(msg) => {
                assert!(
                    msg.contains("IS stamped on this GGUF"),
                    "the message must report the stamped group: {msg}"
                );
                assert!(
                    msg.contains("source geometry is authenticated")
                        && msg.contains("native frontend remains unimplemented"),
                    "the message must distinguish source evidence from native work: {msg}"
                );
                assert!(
                    msg.contains("blockers (2)-(4) are reported below"),
                    "the message must keep the remaining blockers explicit: {msg}"
                );
                // This fixture stamps no tokenizer, so blocker (3) must
                // still be reported in full.
                assert!(
                    msg.contains("blocker (3) applies in full"),
                    "the missing tokenizer must remain explicit: {msg}"
                );
                // The stamped geometry is echoed so a reader can sanity-check it.
                assert!(
                    msg.contains("head_dim=64"),
                    "derived head_dim echoed: {msg}"
                );
            }
            other => panic!("expected VokraError::UnsupportedOp, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Test 13 — Argument guards fire BEFORE the loud-partial: empty PCM and
    //           a mismatched sample rate are InvalidArgument, not
    //           UnsupportedOp.
    // -----------------------------------------------------------------------

    #[test]
    fn argument_guards_fire_before_the_loud_partial() {
        let m = FireredAsrAed::from_gguf(&spec_stamped_gguf()).expect("bind");

        // (a) Empty PCM.
        let Err(err) = m.transcribe_tokens(&[], 16_000) else {
            panic!("expected InvalidArgument on empty PCM");
        };
        match err {
            VokraError::InvalidArgument(msg) => assert!(
                msg.contains("empty PCM"),
                "message must name the empty-input gap, got `{msg}`"
            ),
            other => panic!("expected VokraError::InvalidArgument, got {other:?}"),
        }

        // (b) A rate the stamped config does not expect — never a silent
        //     resample.
        let pcm = vec![0.0_f32; 800];
        let Err(err) = m.transcribe_tokens(&pcm, 8_000) else {
            panic!("expected InvalidArgument on a mismatched sample rate");
        };
        match err {
            VokraError::InvalidArgument(msg) => {
                assert!(
                    msg.contains("8000") && msg.contains("16000"),
                    "message must name both the offered and expected rates, got `{msg}`"
                );
                assert!(
                    msg.contains("never silently resamples"),
                    "message must state the no-resample rule, got `{msg}`"
                );
            }
            other => panic!("expected VokraError::InvalidArgument, got {other:?}"),
        }

        // (c) The matching rate reaches the loud-partial, not a guard.
        let Err(err) = m.transcribe_tokens(&pcm, 16_000) else {
            panic!("expected the loud-partial at the matching rate");
        };
        assert!(matches!(err, VokraError::UnsupportedOp(_)));
    }

    // -----------------------------------------------------------------------
    // Test 14 — The generic `AsrEngine` path reaches the same loud-partial
    //           (so a session-glue caller never gets a fabricated result).
    // -----------------------------------------------------------------------

    #[test]
    fn asr_engine_trait_path_also_loud_partials() {
        let file = converter_shaped_gguf();
        let m = FireredAsrAed::from_gguf(&file).expect("bind");
        let engine: &dyn AsrEngine = &m;

        let pcm = vec![0.0_f32; 16_000];
        let Err(err) = engine.transcribe(&pcm) else {
            panic!("the AsrEngine path must loud-partial, never fabricate text");
        };
        assert!(
            matches!(err, VokraError::UnsupportedOp(_)),
            "expected UnsupportedOp from the trait path, got {err:?}"
        );

        // The empty-PCM guard is shared by both paths.
        let Err(err) = engine.transcribe(&[]) else {
            panic!("expected InvalidArgument on empty PCM through the trait");
        };
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    // -----------------------------------------------------------------------
    // Test 15 — A tokenizer blob flips blocker (3) in the message.
    // -----------------------------------------------------------------------

    #[test]
    fn tokenizer_presence_is_surfaced_not_required() {
        let mut b = base_builder(Some(LicenseClass::Permissive));
        b.add_string(KEY_TOKENIZER_MODEL, "synthetic-vocab-blob");
        let m = FireredAsrAed::from_gguf(&finish(&b)).expect("bind");
        assert!(m.has_tokenizer(), "the blob must be surfaced");

        let pcm = vec![0.0_f32; 320];
        let Err(err) = m.transcribe_tokens(&pcm, 16_000) else {
            panic!("a tokenizer alone must NOT be mistaken for a landed forward");
        };
        match err {
            VokraError::UnsupportedOp(msg) => {
                assert!(
                    msg.contains("blob IS present on this GGUF"),
                    "the message must credit the present tokenizer: {msg}"
                );
                assert!(
                    msg.contains("blocker (3) is already cleared"),
                    "the message must say which blocker the tokenizer clears: {msg}"
                );
                // Blockers (1) and (2) are untouched by a tokenizer, so the
                // gate itself must not move.
                assert!(
                    msg.contains("FRONTEND CONTRACT")
                        && msg.contains("MISSING NATIVE TENSOR MAPPING"),
                    "the remaining blockers must stay explicit: {msg}"
                );
            }
            other => panic!("expected VokraError::UnsupportedOp, got {other:?}"),
        }
    }
}
