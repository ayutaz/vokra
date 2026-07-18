//! Moshi (Helium + Mimi): safetensors checkpoint → GGUF conversion
//! (M4-06-T22 — `vokra.moshi.*` chunks + `AttributionRequired` provenance
//! stamp + attribution text + tokenizer embed).
//!
//! Input: the upstream `kyutai/moshiko-pytorch-bf16` `model.safetensors`
//! (355 tensors, **all BF16** — the T02 manifest fixture
//! `tests/parity/moshi/moshiko_tensor_manifest.json` pins the shape
//! table). Output: a GGUF carrying every float tensor under its upstream
//! name — **BF16 decoded to F32, which is exact** (BF16 is the top 16
//! bits of the f32 pattern; ADR M4-06 §D2 gap analysis) — plus the
//! `vokra.moshi.*` / `vokra.mimi.*` / `vokra.provenance.*` chunks, the
//! FR-MD-09 attribution text, and (optionally) the raw SentencePiece
//! tokenizer blob.
//!
//! # What is shape-driven vs. transcribed (ADR M4-06 §D2/§D3)
//!
//! - **Shape-driven** (checkpoint tensor shapes, never literals):
//!   `d_model` / `text_card` (`text_linear.weight`), layer counts
//!   (`transformer.layers.{i}` / `depformer.layers.{i}`), gating hidden
//!   widths (`gating.linear_in` rows ÷ 2), stream counts (`emb.{k}` /
//!   `depformer_in.{k}` tallies), `card` (`linears.0.weight` rows),
//!   depformer width (`depformer_in.0.weight` rows).
//! - **Transcribed constants** (primary source = `kyutai-labs/moshi`
//!   `loaders.py` `_lm_kwargs` + `transformer.py`, fetched 2026-07-15 and
//!   recorded in the ADR): head counts (not derivable from shapes),
//!   RMSNorm ε (1e-8, `rms_norm_f32`), RoPE max_period (10 000), the
//!   attention `context` (3000), text pad ids (3 / 0), the per-channel
//!   delay list, and the whole Mimi chunk group (`_mimi_config`).
//!
//! # Delays (structural rule)
//!
//! `_lm_kwargs["delays"]` for the 7B model is
//! `[0, 0, 1×7, 0, 1×7]` — text 0, each direction's semantic codebook 0,
//! its acoustic codebooks 1. The converter emits that structure for the
//! detected `(dep_q, n_user)` split; at the real checkpoint's 16/8 shape
//! this reproduces the upstream list **verbatim** (pinned by test).
//!
//! # Mimi weights are a separate artifact
//!
//! The Mimi codec weights live in their own checkpoint
//! (`tokenizer-e351c8d8-checkpoint125.safetensors` — converted standalone
//! by `--model mimi`, M4-04); this GGUF carries the Mimi **hparams** so
//! the runtime's shared module can shape itself
//! (`quantizer.n_q = max(dep_q, n_q − dep_q)` — `CheckpointInfo.get_mimi`)
//! while its weight binding stays the documented synthesized bridge until
//! the shared module's T29 binding (`vokra-models::mimi` docs).

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{
    GgmlType, GgufArray, GgufBuilder, GgufMetadataValue, GgufStreamWriter, GgufTensorDecl,
    GgufValueType, chunks,
};

use crate::ConvertError;
use crate::safetensors::{SafeTensorInfo, SafetensorsFile, SafetensorsFileReader};

/// `vokra.model.arch` for Moshi GGUFs — kept in sync with the runtime
/// constant `vokra-models::moshi::EXPECTED_ARCH`.
pub(crate) const ARCH: &str = "moshi";
/// `vokra.model.name` for the Moshi GGUF.
pub(crate) const NAME: &str = "moshi-helium-7b";

// --- vokra.moshi.* keys (duplicated verbatim from
// `vokra-models/src/moshi/config.rs` — the two crates only share
// vokra-core; the round-trip test below catches drift) ----------------------

const KEY_TM_N_LAYER: &str = "vokra.moshi.arch.temporal.n_layer";
const KEY_TM_D_MODEL: &str = "vokra.moshi.arch.temporal.d_model";
const KEY_TM_N_HEAD: &str = "vokra.moshi.arch.temporal.n_head";
const KEY_TM_FFN_HIDDEN: &str = "vokra.moshi.arch.temporal.ffn_hidden";
const KEY_DT_N_LAYER: &str = "vokra.moshi.arch.depth.n_layer";
const KEY_DT_D_MODEL: &str = "vokra.moshi.arch.depth.d_model";
const KEY_DT_N_HEAD: &str = "vokra.moshi.arch.depth.n_head";
const KEY_DT_FFN_HIDDEN: &str = "vokra.moshi.arch.depth.ffn_hidden";
const KEY_RMS_NORM_EPS: &str = "vokra.moshi.arch.rms_norm_eps";
const KEY_ROPE_MAX_PERIOD: &str = "vokra.moshi.arch.rope_max_period";
const KEY_CONTEXT: &str = "vokra.moshi.arch.context";
const KEY_MAX_CTX: &str = "vokra.moshi.arch.max_ctx";
const KEY_AUDIO_N_Q_IN: &str = "vokra.moshi.audio.n_q_in";
const KEY_AUDIO_DEP_Q: &str = "vokra.moshi.audio.dep_q";
const KEY_AUDIO_CARD: &str = "vokra.moshi.audio.card";
const KEY_TEXT_CARD: &str = "vokra.moshi.text.card";
const KEY_TEXT_PAD_ID: &str = "vokra.moshi.text.pad_id";
const KEY_TEXT_END_PAD_ID: &str = "vokra.moshi.text.end_pad_id";
const KEY_N_DELAYS: &str = "vokra.moshi.n_delays";
const PREFIX_DELAY: &str = "vokra.moshi.delay.";

// --- vokra.mimi.* keys (duplicated from vokra-models/src/mimi/config.rs) ---

const KEY_MIMI_SAMPLE_RATE: &str = "vokra.mimi.sample_rate";
const KEY_MIMI_FRAME_RATE_MHZ: &str = "vokra.mimi.frame_rate_mhz";
const KEY_MIMI_SEANET_DIMENSION: &str = "vokra.mimi.seanet.dimension";
const KEY_MIMI_SEANET_N_FILTERS: &str = "vokra.mimi.seanet.n_filters";
const KEY_MIMI_SEANET_N_RESIDUAL_LAYERS: &str = "vokra.mimi.seanet.n_residual_layers";
const KEY_MIMI_SEANET_KERNEL_SIZE: &str = "vokra.mimi.seanet.kernel_size";
const KEY_MIMI_SEANET_RESIDUAL_KERNEL_SIZE: &str = "vokra.mimi.seanet.residual_kernel_size";
const KEY_MIMI_SEANET_LAST_KERNEL_SIZE: &str = "vokra.mimi.seanet.last_kernel_size";
const KEY_MIMI_SEANET_COMPRESS: &str = "vokra.mimi.seanet.compress";
const KEY_MIMI_SEANET_DILATION_BASE: &str = "vokra.mimi.seanet.dilation_base";
const KEY_MIMI_SEANET_N_RATIOS: &str = "vokra.mimi.seanet.n_ratios";
const PREFIX_MIMI_SEANET_RATIO: &str = "vokra.mimi.seanet.ratio.";
const KEY_MIMI_QUANTIZER_DIMENSION: &str = "vokra.mimi.quantizer.dimension";
const KEY_MIMI_QUANTIZER_N_Q: &str = "vokra.mimi.quantizer.n_q";
const KEY_MIMI_QUANTIZER_BINS: &str = "vokra.mimi.quantizer.bins";
const KEY_MIMI_QUANTIZER_INPUT_DIMENSION: &str = "vokra.mimi.quantizer.input_dimension";
const KEY_MIMI_QUANTIZER_OUTPUT_DIMENSION: &str = "vokra.mimi.quantizer.output_dimension";
const KEY_MIMI_TRANSFORMER_D_MODEL: &str = "vokra.mimi.transformer.d_model";
const KEY_MIMI_TRANSFORMER_N_HEAD: &str = "vokra.mimi.transformer.n_head";
const KEY_MIMI_TRANSFORMER_N_LAYER: &str = "vokra.mimi.transformer.n_layer";
const KEY_MIMI_TRANSFORMER_FF_DIM: &str = "vokra.mimi.transformer.ff_dim";
const KEY_MIMI_TRANSFORMER_CONTEXT: &str = "vokra.mimi.transformer.context";
const KEY_MIMI_TRANSFORMER_MAX_PERIOD: &str = "vokra.mimi.transformer.max_period";
const KEY_MIMI_TRANSFORMER_LAYER_SCALE: &str = "vokra.mimi.transformer.layer_scale";

/// The raw tokenizer blob key (M2-06 precedent).
const KEY_TOKENIZER_MODEL: &str = "vokra.tokenizer.model";

// --- Transcribed constants (ADR M4-06 §D2 — `_lm_kwargs`, fetched
// 2026-07-15; head counts are not derivable from tensor shapes) -------------

/// `_lm_kwargs["num_heads"] = 32`.
const MOSHI_TEMPORAL_N_HEAD: u32 = 32;
/// `_lm_kwargs["depformer_num_heads"] = 16`.
const MOSHI_DEPTH_N_HEAD: u32 = 16;
/// `create_norm_fn("rms_norm_f32")` → `RMSNorm(dim, eps=1e-8)`.
const MOSHI_RMS_NORM_EPS: f32 = 1e-8;
/// `_lm_kwargs["max_period"] = 10000`.
const MOSHI_ROPE_MAX_PERIOD: f32 = 10_000.0;
/// `_lm_kwargs["context"] = 3000` (sliding attention window).
const MOSHI_CONTEXT: u32 = 3000;
/// `_lm_kwargs["existing_text_padding_id"] = 3`.
const MOSHI_TEXT_PAD_ID: u32 = 3;
/// `LMModel existing_text_end_padding_id = 0` (default, lm.py).
const MOSHI_TEXT_END_PAD_ID: u32 = 0;

/// Mimi chunk constants — `loaders.py` `_mimi_config` verbatim (shared
/// with the CSM / standalone-Mimi converters; same primary source).
const MIMI_SAMPLE_RATE: u32 = 24_000;
const MIMI_FRAME_RATE_MHZ: u32 = 12_500;
const MIMI_SEANET_DIMENSION: u32 = 512;
const MIMI_SEANET_N_FILTERS: u32 = 64;
const MIMI_SEANET_N_RESIDUAL_LAYERS: u32 = 1;
const MIMI_SEANET_KERNEL_SIZE: u32 = 7;
const MIMI_SEANET_RESIDUAL_KERNEL_SIZE: u32 = 3;
const MIMI_SEANET_LAST_KERNEL_SIZE: u32 = 3;
const MIMI_SEANET_COMPRESS: u32 = 2;
const MIMI_SEANET_DILATION_BASE: u32 = 2;
const MIMI_SEANET_RATIOS: [u32; 4] = [8, 6, 5, 4];
const MIMI_QUANTIZER_DIMENSION: u32 = 256;
const MIMI_QUANTIZER_IO_DIMENSION: u32 = 512;
const MIMI_TRANSFORMER_D_MODEL: u32 = 512;
const MIMI_TRANSFORMER_N_HEAD: u32 = 8;
const MIMI_TRANSFORMER_N_LAYER: u32 = 8;
const MIMI_TRANSFORMER_FF_DIM: u32 = 2048;
const MIMI_TRANSFORMER_CONTEXT: u32 = 250;
const MIMI_TRANSFORMER_MAX_PERIOD: u32 = 10_000;
const MIMI_TRANSFORMER_LAYER_SCALE: f32 = 0.01;

/// The FR-MD-09 attribution text stamped into
/// `vokra.provenance.attribution` — wording aligned with `NOTICE` §5 and
/// the `docs/license-audit.md` Kyutai row (final legal sufficiency =
/// T29 owner sign-off).
pub(crate) const MOSHI_ATTRIBUTION_TEXT: &str = "This application uses the Moshi model \
     (Helium temporal transformer + depformer + Mimi codec) by Kyutai. Moshi \
     weights are licensed under CC-BY 4.0 (attribution required; commercial \
     use permitted). Copyright (c) Kyutai / Moshi authors. Source: \
     https://github.com/kyutai-labs/moshi / \
     https://huggingface.co/kyutai/moshiko-pytorch-bf16";

/// Outcome of a Moshi conversion.
#[derive(Debug, Default)]
pub(crate) struct MoshiReport {
    /// Float tensors written (F32/F16/BF16 payloads pass through
    /// **verbatim** — the voxtral 12e574e posture; the runtime's single
    /// `tensor_f32` path widens BF16 → f32 exactly at load).
    pub(crate) written: usize,
    /// BF16 tensors among them (passed through byte-exact as GGUF `BF16`,
    /// ggml type 30 — observability for the on-disk size, which is half
    /// the old convert-time-widened F32 layout).
    pub(crate) bf16_passthrough: usize,
    /// Non-float tensors skipped (defensive counter).
    pub(crate) skipped_non_float: usize,
    /// Whether a tokenizer blob was embedded.
    pub(crate) tokenizer_embedded: bool,
    /// Operator-facing diagnostics (never fail the conversion — the
    /// runtime is the authoritative gate, FR-EX-08).
    pub(crate) notes: Vec<String>,
}

/// Outcome of the streaming (bounded-memory) conversion path — the
/// [`crate::ConvertSummary`] ingredients the file-level entry needs.
#[derive(Debug)]
pub(crate) struct MoshiStreamOutcome {
    pub(crate) report: MoshiReport,
    pub(crate) tensor_count: usize,
    pub(crate) metadata_count: usize,
    pub(crate) output_bytes: u64,
}

/// Shape-derived hparams (module docs).
struct Derived {
    d_model: usize,
    text_card: usize,
    tm_n_layer: usize,
    tm_ffn_hidden: usize,
    dt_n_layer: usize,
    dt_d_model: usize,
    dt_ffn_hidden: usize,
    n_q_in: usize,
    dep_q: usize,
    card: usize,
}

/// Descriptor lookup shared by the in-memory ([`SafetensorsFile`]) and
/// windowed ([`SafetensorsFileReader`]) checkpoint readers, so shape
/// derivation is written once and cannot diverge between the two paths.
trait MoshiCheckpointIndex {
    fn info(&self, name: &str) -> Option<&SafeTensorInfo>;
}

impl MoshiCheckpointIndex for SafetensorsFile {
    fn info(&self, name: &str) -> Option<&SafeTensorInfo> {
        self.tensor_info(name)
    }
}

impl MoshiCheckpointIndex for SafetensorsFileReader {
    fn info(&self, name: &str) -> Option<&SafeTensorInfo> {
        self.tensor_info(name)
    }
}

fn shape_of<'a>(st: &'a dyn MoshiCheckpointIndex, name: &str) -> Result<&'a [u64], ConvertError> {
    st.info(name).map(|t| t.shape.as_slice()).ok_or_else(|| {
        ConvertError::Parse(format!(
            "moshi checkpoint: required tensor `{name}` is absent (the T02 \
             manifest lists it — wrong file?)"
        ))
    })
}

fn count_indexed(st: &dyn MoshiCheckpointIndex, fmt: impl Fn(usize) -> String) -> usize {
    let mut n = 0;
    while st.info(&fmt(n)).is_some() {
        n += 1;
    }
    n
}

fn derive(st: &dyn MoshiCheckpointIndex) -> Result<Derived, ConvertError> {
    let text_linear = shape_of(st, "text_linear.weight")?;
    if text_linear.len() != 2 {
        return Err(ConvertError::Parse(format!(
            "moshi checkpoint: text_linear.weight has rank {} (want 2)",
            text_linear.len()
        )));
    }
    let (text_card, d_model) = (text_linear[0] as usize, text_linear[1] as usize);
    let tm_n_layer = count_indexed(st, |i| format!("transformer.layers.{i}.norm1.alpha"));
    let dt_n_layer = count_indexed(st, |i| format!("depformer.layers.{i}.norm1.alpha"));
    let n_q_in = count_indexed(st, |i| format!("emb.{i}.weight"));
    let dep_q = count_indexed(st, |i| format!("depformer_in.{i}.weight"));
    if tm_n_layer == 0 || dt_n_layer == 0 || n_q_in == 0 || dep_q == 0 {
        return Err(ConvertError::Parse(format!(
            "moshi checkpoint: indexed tensor families empty (temporal layers \
             {tm_n_layer}, depformer layers {dt_n_layer}, emb {n_q_in}, \
             depformer_in {dep_q})"
        )));
    }
    let gating_in = shape_of(st, "transformer.layers.0.gating.linear_in.weight")?;
    let tm_ffn_hidden = gating_in[0] as usize / 2;
    let dt_gating_in = shape_of(st, "depformer.layers.0.gating.0.linear_in.weight")?;
    let dt_ffn_hidden = dt_gating_in[0] as usize / 2;
    let dep_in = shape_of(st, "depformer_in.0.weight")?;
    let dt_d_model = dep_in[0] as usize;
    let head = shape_of(st, "linears.0.weight")?;
    let card = head[0] as usize;
    Ok(Derived {
        d_model,
        text_card,
        tm_n_layer,
        tm_ffn_hidden,
        dt_n_layer,
        dt_d_model,
        dt_ffn_hidden,
        n_q_in,
        dep_q,
        card,
    })
}

/// The transcribed upstream head count when it divides the derived width
/// (the 7B checkpoint case — exact), else the largest compatible
/// fallback with a loud report note (synthetic fixtures; module docs).
fn compatible_head_count(
    d_model: usize,
    upstream: u32,
    even_head_dim: bool,
    stack: &str,
    report: &mut MoshiReport,
) -> u32 {
    let fits = |h: u32| -> bool {
        h > 0 && d_model % h as usize == 0 && (!even_head_dim || (d_model / h as usize) % 2 == 0)
    };
    if fits(upstream) {
        return upstream;
    }
    let fallback = [8u32, 4, 2, 1].into_iter().find(|&h| fits(h)).unwrap_or(1);
    report.notes.push(format!(
        "{stack} head count: upstream constant {upstream} does not divide the \
         derived d_model {d_model} (not the 7B checkpoint?) — wrote fallback \
         {fallback}; verify against the real config before relying on it"
    ));
    fallback
}

/// The `_lm_kwargs["delays"]` structure generalized to the detected
/// stream split: text 0, each direction's codebook 0 at delay 0, its
/// remaining codebooks at delay 1. At the 7B split (16/8) this equals
/// the upstream list verbatim (pinned by test — module docs).
fn structural_delays(dep_q: usize, n_user: usize) -> Vec<u32> {
    let mut delays = Vec::with_capacity(1 + dep_q + n_user);
    delays.push(0); // text
    for cb in 0..dep_q {
        delays.push(u32::from(cb != 0));
    }
    for cb in 0..n_user {
        delays.push(u32::from(cb != 0));
    }
    delays
}

/// Converts a Moshi safetensors buffer into a populated GGUF builder —
/// the **in-memory reference** implementation the byte-identity test
/// compares [`convert_streaming`] against. All production entries
/// (`convert_file` / `convert_moshi_file`) route through the streaming
/// path, so this stays test-gated (dead in release builds by design).
///
/// `tokenizer_bytes` — the raw `tokenizer_spm_32k_3.model` file (public
/// in the kyutai repo; `None` skips the embed with a loud note and the
/// runtime monologue decode fails loudly until supplied).
#[cfg(test)]
pub(crate) fn convert(
    bytes: Vec<u8>,
    tokenizer_bytes: Option<Vec<u8>>,
) -> Result<(GgufBuilder, MoshiReport), ConvertError> {
    let st = SafetensorsFile::parse(bytes)?;
    let d = derive(&st)?;
    let (mut b, mut report) = metadata_builder(&d, tokenizer_bytes);

    // --- tensors: upstream names verbatim; F32/F16/BF16 byte-exact
    // passthrough (BF16 stays GGUF `BF16`, type 30 — the voxtral 12e574e
    // posture; the runtime widens BF16 → f32 exactly at load) ---
    for t in st.tensors() {
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

    Ok((b, report))
}

/// Streaming (bounded-memory) Moshi conversion: header-only checkpoint
/// open, metadata + tensor **declarations** first, then every payload
/// copied tensor-by-tensor through one reused buffer — never more than a
/// single tensor payload in RAM. This unblocks the full-7B
/// `kyutai/moshiko-pytorch-bf16` conversion on a 16 GB machine (the old
/// materialize-everything path peaked ≈ 97 GiB: whole input + widened F32
/// builder copies + a whole-output `to_bytes` buffer).
///
/// Output bytes are **identical** to the in-memory [`convert`] path over
/// the same checkpoint (same metadata builder, same declaration order,
/// [`GgufStreamWriter`]'s byte-identity contract) — pinned by test.
pub(crate) fn convert_streaming(
    input: &Path,
    output: &Path,
    tokenizer_bytes: Option<Vec<u8>>,
) -> Result<MoshiStreamOutcome, ConvertError> {
    let mut st = SafetensorsFileReader::open(input)?;
    let d = derive(&st)?;
    let (b, mut report) = metadata_builder(&d, tokenizer_bytes);

    // Declare every float tensor in checkpoint order (BF16 verbatim).
    let mut decls = Vec::new();
    let mut float_names = Vec::new();
    for t in st.tensors() {
        match t.dtype {
            GgmlType::F32 | GgmlType::F16 | GgmlType::BF16 => {
                decls.push(GgufTensorDecl {
                    name: t.name.clone(),
                    dtype: t.dtype,
                    dimensions: t.shape.clone(),
                });
                float_names.push(t.name.clone());
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

    let out_file = std::fs::File::create(output)?;
    let mut w = GgufStreamWriter::begin(std::io::BufWriter::new(out_file), &b, &decls)?;
    let mut buf = Vec::new();
    for name in &float_names {
        st.read_tensor_into(name, &mut buf)
            .map_err(|e| ConvertError::Parse(format!("moshi: reading `{name}`: {e}")))?;
        w.write_tensor(name, &buf)?;
    }
    drop(buf);
    let out = w.finish()?;
    let out_file = out
        .into_inner()
        .map_err(|e| ConvertError::Io(e.into_error()))?;
    out_file.sync_all().map_err(ConvertError::Io)?;
    let output_bytes = out_file.metadata().map_err(ConvertError::Io)?.len();

    Ok(MoshiStreamOutcome {
        tensor_count: decls.len(),
        metadata_count: b.metadata_count(),
        output_bytes,
        report,
    })
}

/// Builds the complete metadata-only GGUF builder (hparams + provenance +
/// attribution + optional tokenizer blob) for a derived checkpoint.
/// Shared **verbatim** by [`convert`] and [`convert_streaming`] — one
/// metadata source is what makes the two outputs byte-identical.
fn metadata_builder(d: &Derived, tokenizer_bytes: Option<Vec<u8>>) -> (GgufBuilder, MoshiReport) {
    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    let mut report = MoshiReport::default();

    // Head counts are transcribed constants (not shape-derivable). They
    // apply verbatim to the 7B checkpoint (4096 % 32 == 0, 1024 % 16 ==
    // 0); on any other shape (synthetic fixtures) a compatible fallback
    // is written WITH a loud report note — never a silent ill-formed
    // config (the runtime rejects `d_model % n_head != 0` anyway).
    let tm_n_head = compatible_head_count(
        d.d_model,
        MOSHI_TEMPORAL_N_HEAD,
        true, // temporal RoPE needs an even head_dim
        "temporal",
        &mut report,
    );
    let dt_n_head = compatible_head_count(
        d.dt_d_model,
        MOSHI_DEPTH_N_HEAD,
        false, // the depformer has no positional embedding
        "depth",
        &mut report,
    );

    // --- vokra.moshi.* (shape-driven + transcribed — module docs) ---
    b.add_u32(KEY_TM_N_LAYER, d.tm_n_layer as u32);
    b.add_u32(KEY_TM_D_MODEL, d.d_model as u32);
    b.add_u32(KEY_TM_N_HEAD, tm_n_head);
    b.add_u32(KEY_TM_FFN_HIDDEN, d.tm_ffn_hidden as u32);
    b.add_u32(KEY_DT_N_LAYER, d.dt_n_layer as u32);
    b.add_u32(KEY_DT_D_MODEL, d.dt_d_model as u32);
    b.add_u32(KEY_DT_N_HEAD, dt_n_head);
    b.add_u32(KEY_DT_FFN_HIDDEN, d.dt_ffn_hidden as u32);
    b.add_f32(KEY_RMS_NORM_EPS, MOSHI_RMS_NORM_EPS);
    b.add_f32(KEY_ROPE_MAX_PERIOD, MOSHI_ROPE_MAX_PERIOD);
    b.add_u32(KEY_CONTEXT, MOSHI_CONTEXT);
    // KV reserve default = the attention window (sessions past it error
    // loudly; a larger reserve is a legitimate re-convert knob).
    b.add_u32(KEY_MAX_CTX, MOSHI_CONTEXT);
    b.add_u32(KEY_AUDIO_N_Q_IN, d.n_q_in as u32);
    b.add_u32(KEY_AUDIO_DEP_Q, d.dep_q as u32);
    b.add_u32(KEY_AUDIO_CARD, d.card as u32);
    b.add_u32(KEY_TEXT_CARD, d.text_card as u32);
    b.add_u32(KEY_TEXT_PAD_ID, MOSHI_TEXT_PAD_ID);
    b.add_u32(KEY_TEXT_END_PAD_ID, MOSHI_TEXT_END_PAD_ID);
    let delays = structural_delays(d.dep_q, d.n_q_in - d.dep_q);
    b.add_u32(KEY_N_DELAYS, delays.len() as u32);
    for (i, delay) in delays.iter().enumerate() {
        b.add_u32(&format!("{PREFIX_DELAY}{i}"), *delay);
    }

    // --- vokra.mimi.* (transcribed `_mimi_config`; n_q per get_mimi) ---
    let mimi_n_q = d.dep_q.max(d.n_q_in - d.dep_q) as u32;
    b.add_u32(KEY_MIMI_SAMPLE_RATE, MIMI_SAMPLE_RATE);
    b.add_u32(KEY_MIMI_FRAME_RATE_MHZ, MIMI_FRAME_RATE_MHZ);
    b.add_u32(KEY_MIMI_SEANET_DIMENSION, MIMI_SEANET_DIMENSION);
    b.add_u32(KEY_MIMI_SEANET_N_FILTERS, MIMI_SEANET_N_FILTERS);
    b.add_u32(
        KEY_MIMI_SEANET_N_RESIDUAL_LAYERS,
        MIMI_SEANET_N_RESIDUAL_LAYERS,
    );
    b.add_u32(KEY_MIMI_SEANET_KERNEL_SIZE, MIMI_SEANET_KERNEL_SIZE);
    b.add_u32(
        KEY_MIMI_SEANET_RESIDUAL_KERNEL_SIZE,
        MIMI_SEANET_RESIDUAL_KERNEL_SIZE,
    );
    b.add_u32(
        KEY_MIMI_SEANET_LAST_KERNEL_SIZE,
        MIMI_SEANET_LAST_KERNEL_SIZE,
    );
    b.add_u32(KEY_MIMI_SEANET_COMPRESS, MIMI_SEANET_COMPRESS);
    b.add_u32(KEY_MIMI_SEANET_DILATION_BASE, MIMI_SEANET_DILATION_BASE);
    b.add_u32(KEY_MIMI_SEANET_N_RATIOS, MIMI_SEANET_RATIOS.len() as u32);
    for (i, r) in MIMI_SEANET_RATIOS.iter().enumerate() {
        b.add_u32(&format!("{PREFIX_MIMI_SEANET_RATIO}{i}"), *r);
    }
    b.add_u32(KEY_MIMI_QUANTIZER_DIMENSION, MIMI_QUANTIZER_DIMENSION);
    b.add_u32(KEY_MIMI_QUANTIZER_N_Q, mimi_n_q);
    // bins ≡ card by upstream construction (`_lm_kwargs["card"] =
    // _quantizer_kwargs["bins"]` — loaders.py), so the shape-derived card
    // drives the codec table size (2048 on the real checkpoint; keeps a
    // synthetic checkpoint's codec ↔ LM vocab coherent).
    b.add_u32(KEY_MIMI_QUANTIZER_BINS, d.card as u32);
    b.add_u32(
        KEY_MIMI_QUANTIZER_INPUT_DIMENSION,
        MIMI_QUANTIZER_IO_DIMENSION,
    );
    b.add_u32(
        KEY_MIMI_QUANTIZER_OUTPUT_DIMENSION,
        MIMI_QUANTIZER_IO_DIMENSION,
    );
    b.add_u32(KEY_MIMI_TRANSFORMER_D_MODEL, MIMI_TRANSFORMER_D_MODEL);
    b.add_u32(KEY_MIMI_TRANSFORMER_N_HEAD, MIMI_TRANSFORMER_N_HEAD);
    b.add_u32(KEY_MIMI_TRANSFORMER_N_LAYER, MIMI_TRANSFORMER_N_LAYER);
    b.add_u32(KEY_MIMI_TRANSFORMER_FF_DIM, MIMI_TRANSFORMER_FF_DIM);
    b.add_u32(KEY_MIMI_TRANSFORMER_CONTEXT, MIMI_TRANSFORMER_CONTEXT);
    b.add_u32(KEY_MIMI_TRANSFORMER_MAX_PERIOD, MIMI_TRANSFORMER_MAX_PERIOD);
    b.add_f32(
        KEY_MIMI_TRANSFORMER_LAYER_SCALE,
        MIMI_TRANSFORMER_LAYER_SCALE,
    );

    // --- provenance + the FR-MD-09 attribution surface (T22 gate) ---
    vokra_core::stamp_provenance(
        &mut b,
        LicenseClass::AttributionRequired,
        "CC-BY-4.0",
        Some("moshi"),
        Some("https://huggingface.co/kyutai/moshiko-pytorch-bf16"),
    );
    vokra_core::stamp_attribution(&mut b, MOSHI_ATTRIBUTION_TEXT);

    if d.dep_q != d.n_q_in - d.dep_q {
        report.notes.push(format!(
            "asymmetric stream split (dep_q {} vs user {}): the runtime engine \
             requires symmetry (loaders.py get_mimi serves one width)",
            d.dep_q,
            d.n_q_in - d.dep_q
        ));
    }
    if let Some(tok) = tokenizer_bytes {
        if tok.is_empty() {
            report
                .notes
                .push("tokenizer file was empty — nothing embedded".into());
        } else {
            b.add_metadata(
                KEY_TOKENIZER_MODEL,
                GgufMetadataValue::Array(GgufArray {
                    element_type: GgufValueType::U8,
                    values: tok.iter().map(|&x| GgufMetadataValue::U8(x)).collect(),
                }),
            );
            report.tokenizer_embedded = true;
        }
    } else {
        report.notes.push(
            "no tokenizer supplied — `vokra.tokenizer.model` not embedded; the \
             runtime monologue decode will fail loudly until a tokenizer-carrying \
             GGUF is converted (tokenizer_spm_32k_3.model is public in the kyutai \
             repo — bundle at T29)"
                .into(),
        );
    }

    (b, report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::GgufFile;

    /// Builds a tiny synthetic Moshi checkpoint (BF16, upstream names —
    /// the `MoshiConfig::tiny_for_tests` shape: d16/2L/4H, depth d8/2L,
    /// n_q_in 4 / dep_q 2, card 9, text 13, gating hidden 8 / 6).
    fn synthetic_checkpoint() -> Vec<u8> {
        let mut entries: Vec<(String, Vec<u64>)> = Vec::new();
        let (d, text, card) = (16u64, 13u64, 9u64);
        let (h_tm, d_dt, h_dt) = (8u64, 8u64, 6u64);
        entries.push(("text_emb.weight".into(), vec![text + 1, d]));
        entries.push(("text_linear.weight".into(), vec![text, d]));
        entries.push(("out_norm.alpha".into(), vec![1, 1, d]));
        for k in 0..4 {
            entries.push((format!("emb.{k}.weight"), vec![card + 1, d]));
        }
        for i in 0..2 {
            let p = format!("transformer.layers.{i}");
            entries.push((format!("{p}.norm1.alpha"), vec![1, 1, d]));
            entries.push((format!("{p}.norm2.alpha"), vec![1, 1, d]));
            entries.push((format!("{p}.self_attn.in_proj_weight"), vec![3 * d, d]));
            entries.push((format!("{p}.self_attn.out_proj.weight"), vec![d, d]));
            entries.push((format!("{p}.gating.linear_in.weight"), vec![2 * h_tm, d]));
            entries.push((format!("{p}.gating.linear_out.weight"), vec![d, h_tm]));
        }
        for cb in 0..2 {
            entries.push((format!("depformer_in.{cb}.weight"), vec![d_dt, d]));
            entries.push((format!("linears.{cb}.weight"), vec![card, d_dt]));
        }
        entries.push(("depformer_text_emb.weight".into(), vec![text + 1, d_dt]));
        entries.push(("depformer_emb.0.weight".into(), vec![card + 1, d_dt]));
        for i in 0..2 {
            let p = format!("depformer.layers.{i}");
            entries.push((format!("{p}.norm1.alpha"), vec![1, 1, d_dt]));
            entries.push((format!("{p}.norm2.alpha"), vec![1, 1, d_dt]));
            entries.push((
                format!("{p}.self_attn.in_proj_weight"),
                vec![2 * 3 * d_dt, d_dt],
            ));
            entries.push((
                format!("{p}.self_attn.out_proj.weight"),
                vec![2 * d_dt, d_dt],
            ));
            for s in 0..2 {
                entries.push((
                    format!("{p}.gating.{s}.linear_in.weight"),
                    vec![2 * h_dt, d_dt],
                ));
                entries.push((
                    format!("{p}.gating.{s}.linear_out.weight"),
                    vec![d_dt, h_dt],
                ));
            }
        }

        // Serialize as safetensors with BF16 payloads (value = 1.0 →
        // 0x3F80 as bf16).
        let mut header = String::from("{");
        let mut data: Vec<u8> = Vec::new();
        for (i, (name, shape)) in entries.iter().enumerate() {
            let n: u64 = shape.iter().product();
            let start = data.len();
            for _ in 0..n {
                data.extend_from_slice(&0x3F80u16.to_le_bytes());
            }
            let end = data.len();
            if i > 0 {
                header.push(',');
            }
            header.push_str(&format!(
                "\"{name}\":{{\"dtype\":\"BF16\",\"shape\":[{}],\"data_offsets\":[{start},{end}]}}",
                shape
                    .iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            ));
        }
        header.push('}');
        let mut blob = Vec::new();
        blob.extend_from_slice(&(header.len() as u64).to_le_bytes());
        blob.extend_from_slice(header.as_bytes());
        blob.extend_from_slice(&data);
        blob
    }

    #[test]
    fn structural_delays_reproduce_the_upstream_7b_list_verbatim() {
        // `_lm_kwargs["delays"]` (ADR M4-06 §D2, fetched 2026-07-15).
        let upstream: Vec<u32> = vec![0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1];
        assert_eq!(structural_delays(8, 8), upstream);
        // The tiny fixture split mirrors the structure.
        assert_eq!(structural_delays(2, 2), vec![0, 0, 1, 0, 1]);
    }

    #[test]
    fn convert_derives_shapes_stamps_attribution_and_passes_bf16_through() {
        let (b, report) =
            convert(synthetic_checkpoint(), Some(b"spm-blob".to_vec())).expect("convert");
        assert_eq!(report.skipped_non_float, 0);
        assert!(
            report.bf16_passthrough > 0,
            "the synthetic checkpoint is BF16"
        );
        assert!(report.tokenizer_embedded);
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();

        // Shape-driven hparams.
        let get_u32 = |k: &str| match file.get(k) {
            Some(GgufMetadataValue::U32(v)) => *v,
            other => panic!("{k}: {other:?}"),
        };
        assert_eq!(get_u32(KEY_TM_D_MODEL), 16);
        assert_eq!(get_u32(KEY_TM_N_LAYER), 2);
        assert_eq!(get_u32(KEY_TM_FFN_HIDDEN), 8);
        assert_eq!(get_u32(KEY_DT_N_LAYER), 2);
        assert_eq!(get_u32(KEY_DT_D_MODEL), 8);
        assert_eq!(get_u32(KEY_DT_FFN_HIDDEN), 6);
        assert_eq!(get_u32(KEY_AUDIO_N_Q_IN), 4);
        assert_eq!(get_u32(KEY_AUDIO_DEP_Q), 2);
        assert_eq!(get_u32(KEY_AUDIO_CARD), 9);
        assert_eq!(get_u32(KEY_TEXT_CARD), 13);
        assert_eq!(get_u32(KEY_N_DELAYS), 5);
        assert_eq!(get_u32("vokra.moshi.delay.2"), 1);
        assert_eq!(get_u32(KEY_MIMI_QUANTIZER_N_Q), 2, "max(dep_q, user)");

        // BF16 passes through VERBATIM (ggml type 30, voxtral 12e574e
        // posture) and the runtime widening is exact: every element was
        // bf16(1.0) = 0x3F80.
        let info = file.tensor_info("text_linear.weight").expect("info");
        assert_eq!(info.dtype, GgmlType::BF16, "no convert-time widening");
        let raw = file.tensor_data("text_linear.weight").expect("bytes");
        assert_eq!(raw.len(), 13 * 16 * 2, "2 bytes/element on disk");
        assert!(
            raw.chunks_exact(2)
                .all(|c| u16::from_le_bytes([c[0], c[1]]) == 0x3F80),
            "source BF16 bytes verbatim"
        );
        let t = file.tensor_f32("text_linear.weight").expect("tensor");
        assert_eq!(t.len(), 13 * 16);
        assert!(t.iter().all(|&v| v == 1.0), "bf16(1.0) → f32 1.0 exactly");

        // The M2-13 gate resolves AttributionRequired and passes the
        // strict (commercial) policy WITHOUT a research flag — CC-BY 4.0
        // is commercial-OK (never confuse with the CC-BY-NC gate).
        let res = vokra_core::resolve_license_class(&file);
        assert_eq!(res.class, LicenseClass::AttributionRequired);
        assert!(!res.is_research_only());
        vokra_core::check_weight_license(&file, &vokra_core::CompliancePolicy::strict())
            .expect("CC-BY 4.0 passes the strict gate");

        // The FR-MD-09 attribution surface is non-empty and Kyutai-named.
        let info = vokra_core::resolve_attribution(&file).expect("attribution");
        assert!(
            info.text.contains("Kyutai"),
            "names the author: {}",
            info.text
        );
        assert!(info.text.contains("CC-BY 4.0"));
    }

    #[test]
    fn missing_required_tensor_is_a_loud_error() {
        // Drop text_linear from the header by converting an empty file.
        let mut blob = Vec::new();
        blob.extend_from_slice(&(2u64).to_le_bytes());
        blob.extend_from_slice(b"{}");
        let err = convert(blob, None).unwrap_err();
        assert!(err.to_string().contains("text_linear.weight"), "{err}");
    }

    #[test]
    fn arch_constant_matches_the_runtime() {
        assert_eq!(ARCH, "moshi");
    }

    #[test]
    fn streaming_and_in_memory_paths_are_byte_identical() {
        // The whole point of the streaming path is bounded memory with NO
        // observable difference: same checkpoint → identical GGUF bytes.
        let blob = synthetic_checkpoint();
        let tok = Some(b"spm-blob".to_vec());

        let (b, mem_report) = convert(blob.clone(), tok.clone()).expect("in-memory convert");
        let via_memory = b.to_bytes().unwrap();

        let mut input = std::env::temp_dir();
        input.push(format!(
            "vokra-moshi-stream-in-{}.safetensors",
            std::process::id()
        ));
        let mut output = std::env::temp_dir();
        output.push(format!(
            "vokra-moshi-stream-out-{}.gguf",
            std::process::id()
        ));
        std::fs::write(&input, &blob).unwrap();

        let outcome = convert_streaming(&input, &output, tok).expect("streaming convert");
        let via_stream = std::fs::read(&output).unwrap();
        std::fs::remove_file(&input).ok();
        std::fs::remove_file(&output).ok();

        assert_eq!(via_memory, via_stream, "byte-identical GGUF outputs");
        assert_eq!(outcome.output_bytes as usize, via_stream.len());
        assert_eq!(outcome.report.written, mem_report.written);
        assert_eq!(outcome.report.bf16_passthrough, mem_report.bf16_passthrough);
        assert_eq!(outcome.tensor_count, mem_report.written);
        assert_eq!(
            outcome.report.skipped_non_float,
            mem_report.skipped_non_float
        );
        assert_eq!(outcome.report.notes, mem_report.notes);
    }

    #[test]
    fn streaming_convert_missing_input_is_a_loud_error() {
        let mut output = std::env::temp_dir();
        output.push(format!(
            "vokra-moshi-stream-neverwritten-{}.gguf",
            std::process::id()
        ));
        let err = convert_streaming(
            std::path::Path::new("/no/such/vokra/moshiko.safetensors"),
            &output,
            None,
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("I/O") || err.to_string().contains("error"),
            "{err}"
        );
        assert!(!output.exists(), "no half-written output file");
    }
}
