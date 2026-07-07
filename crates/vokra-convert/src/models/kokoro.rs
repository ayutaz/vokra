//! Kokoro-82M (StyleTTS 2 派生 iSTFTNet): safetensors checkpoint to GGUF
//! conversion (M2-07-T06/T07 foundation).
//!
//! Input: the upstream `hexgrad/Kokoro-82M` safetensors checkpoint (weights
//! only — no model code is imported, per IF-06 / FR-MD-02). Output: a GGUF
//! carrying every float tensor plus the `vokra.model.*` and `vokra.kokoro.*`
//! metadata chunks the native Kokoro implementation (a later WP) loads
//! against.
//!
//! # Tensor naming contract (M2-07 foundation)
//!
//! GGUF tensor names are the **upstream safetensors names verbatim** (same
//! contract Whisper uses). Rich Vokra-side renaming can arrive later without
//! changing the guarantees of this module.
//!
//! # No `vokra.frontend.*` chunk
//!
//! Kokoro is a TTS decoder — it has no audio front-end (mel/STFT feature
//! extractor) that the runtime controls. Its **output-side** iSTFT is stored
//! under `vokra.kokoro.istft.*` (mirroring piper's `vokra.piper.istft.*`),
//! **not** under the `vokra.frontend.*` input-side chunk.
//!
//! # iSTFTNet head, not vocos
//!
//! Kokoro's vocoder is StyleTTS 2 派生 iSTFTNet (レビュアー A 修正, CLAUDE.md
//! モデル表). The runtime decoder will lower magnitude+phase to complex re/im
//! inline and call `vokra_ops::istft` (FR-OP-01), not `vocos_head` (FR-OP-12).
//! This converter never emits a `vocos_*` metadata key.
//!
//! # Scope
//!
//! Foundation WP only: verbatim safetensors → GGUF, shape-driven hparams where
//! possible with `0` placeholders (mirroring Whisper's degenerate-shape
//! pattern) for values that need T02 upstream inspection. Voicepack layout,
//! phoneme table, and voice name list default to synthesized placeholders — a
//! caller who has the real misaki phoneme table + voicepack index passes them
//! via [`convert_with_config`] and its CLI surface `--config config.json`
//! (same shape as piper-plus).
//!
//! # `config.json` schema (accepted by [`convert_with_config`])
//!
//! The parser is lenient about field names to accommodate the varied upstream
//! forks. All keys are optional individually, but **at least one field from
//! each family** (symbols + voices) must be present; missing both raises
//! [`ConvertError::Parse`]. First-match wins per family:
//!
//! Symbol family (in precedence order):
//!
//! 1. `vocab: { "<symbol>": <id>, … }` — Kokoro / misaki-style symbol→id map.
//!    Table length = `max(id)+1`. Missing slots stay as `""`.
//! 2. `phoneme_symbols: [<str>, …]` — id-indexed symbol array.
//! 3. `symbols: [<str>, …]` — alias of `phoneme_symbols`.
//!
//! Voice family (in precedence order):
//!
//! 1. `voices: [<str>, …]` — voice-name list (canonical release ships these as
//!    separate `voices/*.pt` files, so this is authoritative for `num_voices`
//!    when present).
//! 2. `voice_names: [<str>, …]` — alias of `voices`.
//!
//! When a config is passed, `num_voices` is overridden from
//! `voice_names.len()` and a note is emitted on any tensor-vs-metadata
//! disagreement (converter stays infallible; runtime rejects at load per
//! FR-EX-08).

use vokra_core::gguf::{
    GgmlType, GgufArray, GgufBuilder, GgufMetadataValue, GgufValueType, chunks,
};

use crate::ConvertError;
use crate::json::{self, JsonValue};
use crate::safetensors::{SafeTensorInfo, SafetensorsFile};

/// `vokra.model.arch` value written for Kokoro-82M GGUFs.
pub(crate) const ARCH: &str = "kokoro-82m-istftnet";
/// `vokra.model.name` value written for the Kokoro-82M GGUF.
pub(crate) const NAME: &str = "kokoro-82m";

// --- vokra.kokoro.* metadata keys (M2-07-T06 chunk design) ------------------

/// `vokra.kokoro.sample_rate` — output PCM sample rate, Hz (`UINT32`).
const KEY_SAMPLE_RATE: &str = "vokra.kokoro.sample_rate";
/// `vokra.kokoro.style_dim` — per-voice style vector dimension (`UINT32`).
const KEY_STYLE_DIM: &str = "vokra.kokoro.style_dim";
/// `vokra.kokoro.num_voices` — voicepack voice count (`UINT32`).
const KEY_NUM_VOICES: &str = "vokra.kokoro.num_voices";
/// `vokra.kokoro.n_text_layers` — text encoder block count (`UINT32`).
const KEY_N_TEXT_LAYERS: &str = "vokra.kokoro.n_text_layers";
/// `vokra.kokoro.n_decoder_layers` — iSTFTNet decoder upsample stage count
/// (`UINT32`).
const KEY_N_DECODER_LAYERS: &str = "vokra.kokoro.n_decoder_layers";
/// `vokra.kokoro.hidden_dim` — text encoder hidden width (`UINT32`).
const KEY_HIDDEN_DIM: &str = "vokra.kokoro.hidden_dim";
/// `vokra.kokoro.istft.n_fft` — decoder iSTFT FFT size (`UINT32`).
const KEY_ISTFT_N_FFT: &str = "vokra.kokoro.istft.n_fft";
/// `vokra.kokoro.istft.hop` — decoder iSTFT hop length (`UINT32`).
const KEY_ISTFT_HOP: &str = "vokra.kokoro.istft.hop";
/// `vokra.kokoro.istft.win_length` — decoder iSTFT window length (`UINT32`).
const KEY_ISTFT_WIN_LENGTH: &str = "vokra.kokoro.istft.win_length";
/// `vokra.kokoro.phoneme_symbols` — phoneme string per id (`ARRAY<STRING>`).
const KEY_PHONEME_SYMBOLS: &str = "vokra.kokoro.phoneme_symbols";
/// `vokra.kokoro.voice_names` — voicepack entry name per row (`ARRAY<STRING>`).
const KEY_VOICE_NAMES: &str = "vokra.kokoro.voice_names";

/// Kokoro-82M output sample rate (Hz). Sourced from the hexgrad/Kokoro-82M
/// Hugging Face model card — publicly documented and not invented (constraint
/// note: hparam numbers that come from official model cards are permitted;
/// `0`-placeholder values on this module's other keys are reserved for the
/// truly TBD `istft.*` triple).
const KOKORO_SAMPLE_RATE: u32 = 24_000;

// Kokoro-82M iSTFT hyper-parameters — derived from the decoder manifest at
// `crates/vokra-models/src/kokoro/data/upstream_tensors_v1_0.tsv`:
//
//   `decoder.module.generator.conv_post.weight_v` = (22, 128, 7)
//     → conv_post.out_ch = 22 = 2·n_half ⇒ n_half = 11 ⇒ n_fft = 20.
//
// `hop_length = 5` is the StyleTTS 2 iSTFTNet convention (ups_stride0 ·
// ups_stride1 · n_fft/2 = 10·6·… → the head sits at n_fft/4 hop rate per
// upstream reference; verified by conv_post output rate matching
// `t_frames · 60` after both `ups.0 (stride=10)` and `ups.1 (stride=6)`).
// `win_length = n_fft` (symmetric Hann, the piper-plus decoder convention).
//
// These three values are structural to Kokoro's iSTFTNet head, so we can
// write them at convert-time rather than leave `0`-placeholders — the M2-07-T15
// decoder rewrite (upstream 375-tensor manifest binding) confirmed the axes.
const KOKORO_ISTFT_N_FFT: u32 = 20;
const KOKORO_ISTFT_HOP: u32 = 5;
const KOKORO_ISTFT_WIN_LENGTH: u32 = 20;

/// Outcome of a Kokoro conversion.
#[derive(Debug, Default)]
pub(crate) struct KokoroReport {
    /// Number of float weight tensors written to the GGUF.
    pub(crate) written: usize,
    /// Tensors whose dtype falls outside the F32/F16 range and were skipped.
    ///
    /// The upstream safetensors reader (`vokra_core::safetensors`) already
    /// rejects unknown dtypes at parse time (`SafetensorsError::UnsupportedDtype`),
    /// so a validly parsed buffer that reaches this converter only ever holds
    /// F32/F16 tensors. This counter is defensive/forward-compat — if the
    /// reader is later extended to admit non-float dtypes (e.g. INT8 quant),
    /// the skip path already exists and the report already reports.
    pub(crate) skipped_non_float: usize,
    /// Voice names in voicepack order (populated by [`convert_with_config`]
    /// when a `--config config.json` is passed; empty on the placeholder path).
    pub(crate) voices: Vec<String>,
    /// Per-voice style vector dimension (derived from `voicepack` shape[1]
    /// when the tensor is present, else `0`).
    pub(crate) style_dim: usize,
    /// Number of phoneme symbols in the emitted `vokra.kokoro.phoneme_symbols`
    /// array. Matches either the placeholder count (n_vocab from
    /// `text_encoder.embedding.weight[0]`) or the config-supplied count.
    pub(crate) phoneme_symbol_count: usize,
    /// Diagnostic notes surfaced to the CLI operator (e.g. tensor vs. config
    /// mismatch on `phoneme_symbols` count, or `voicepack` rows vs.
    /// `voice_names` length). The converter never fails on a mismatch — the
    /// runtime is the authoritative gate (FR-EX-08) — but a loud warning is
    /// printed so the operator does not learn about it only at load time.
    pub(crate) notes: Vec<String>,
}

/// Reads dimension `axis` of tensor `name` from the checkpoint, or `0` when
/// the tensor (or that axis) is absent — a degenerate checkpoint the runtime
/// then rejects at load (FR-EX-08). Shared by [`convert`] and
/// [`write_hparams`] so every derivation reads the identical value.
fn tensor_dim(st: &SafetensorsFile, name: &str, axis: usize) -> u64 {
    st.tensors()
        .iter()
        .find(|t: &&SafeTensorInfo| t.name == name)
        .and_then(|t| t.shape.get(axis).copied())
        .unwrap_or(0)
}

/// Counts contiguous transformer blocks named `<prefix><i>.` for
/// `i = 0, 1, …`.
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

/// Parsed `config.json` payload used by [`convert_with_config`].
///
/// The parser (see [`KokoroJsonConfig::parse`]) recognizes multiple upstream
/// spellings of each field; see the module docstring for the accepted schema.
#[derive(Debug, Default)]
pub(crate) struct KokoroJsonConfig {
    /// Phoneme symbol per id; index = id. Length is `max(id)+1` for the
    /// `vocab: {…}` shape or the array length for the `phoneme_symbols` /
    /// `symbols` shapes.
    pub(crate) phoneme_symbols: Vec<String>,
    /// Voice name per id; index = id (`voice_names` == `voices`).
    pub(crate) voice_names: Vec<String>,
}

impl KokoroJsonConfig {
    /// Parses a Kokoro `config.json` payload. See the module docstring for
    /// the accepted schema (first-match wins per field family; at least one
    /// of `{vocab, phoneme_symbols, symbols}` **and** one of `{voices,
    /// voice_names}` must be present).
    pub(crate) fn parse(bytes: &[u8]) -> Result<Self, ConvertError> {
        let root = json::parse(bytes).map_err(|e| ConvertError::Parse(e.to_string()))?;

        // Symbol family: vocab (map) > phoneme_symbols (array) > symbols (array).
        let phoneme_symbols = if let Some(vocab) = root.get("vocab").and_then(JsonValue::as_object)
        {
            // symbol → id map. Table length = max(id)+1; missing slots left "".
            let mut table: Vec<String> = Vec::new();
            for (symbol, id) in vocab {
                if let Some(id) = id.as_u64() {
                    let id = id as usize;
                    if id >= table.len() {
                        table.resize(id + 1, String::new());
                    }
                    table[id] = symbol.clone();
                }
            }
            table
        } else if let Some(arr) = root.get("phoneme_symbols").and_then(JsonValue::as_array) {
            arr.iter()
                .filter_map(|v| v.as_str().map(str::to_owned))
                .collect()
        } else if let Some(arr) = root.get("symbols").and_then(JsonValue::as_array) {
            arr.iter()
                .filter_map(|v| v.as_str().map(str::to_owned))
                .collect()
        } else {
            return Err(ConvertError::Parse(
                "kokoro config: no phoneme symbols found (expected `vocab`, \
                 `phoneme_symbols`, or `symbols`)"
                    .to_owned(),
            ));
        };

        // Voice family: voices > voice_names.
        let voice_names = if let Some(arr) = root.get("voices").and_then(JsonValue::as_array) {
            arr.iter()
                .filter_map(|v| v.as_str().map(str::to_owned))
                .collect()
        } else if let Some(arr) = root.get("voice_names").and_then(JsonValue::as_array) {
            arr.iter()
                .filter_map(|v| v.as_str().map(str::to_owned))
                .collect()
        } else {
            return Err(ConvertError::Parse(
                "kokoro config: no voice list found (expected `voices` or `voice_names`)"
                    .to_owned(),
            ));
        };

        Ok(Self {
            phoneme_symbols,
            voice_names,
        })
    }
}

/// Result of writing the `vokra.kokoro.*` hparam chunk. Aggregates the value
/// derivations the [`convert_with_config`] caller needs for its [`KokoroReport`].
struct HparamOutcome {
    style_dim: u64,
    phoneme_symbol_count: usize,
    voice_names: Vec<String>,
    notes: Vec<String>,
}

/// Derives the `vokra.kokoro.*` hparams from tensor shapes (and optional
/// config overrides) and writes them into `b`.
///
/// Every numeric value is read from a tensor shape (or a well-documented
/// model-card invariant like `sample_rate = 24_000`). Missing tensors write
/// `0` — the converter stays infallible so degenerate synthetic inputs still
/// round-trip, but a `0` on a required hparam is rejected by the runtime
/// loader at load time (FR-EX-08 — no silent fallback in the runtime).
///
/// When `config` is `Some`, the `phoneme_symbols` and `voice_names` arrays
/// (plus `num_voices`) are taken from the config verbatim. Otherwise the
/// placeholder path is used: `p0..p_{n_vocab-1}` symbols and an empty voice
/// list — the same values the module has emitted since M2-07 T06 (kept for
/// backward compatibility with the roundtrip test that does not pass a config).
fn write_hparams(
    b: &mut GgufBuilder,
    st: &SafetensorsFile,
    config: Option<&KokoroJsonConfig>,
) -> HparamOutcome {
    // Shape-driven derivations. Real Kokoro-82M ships with the
    // ``nn.DataParallel`` ``.module.`` prefix baked into every tensor name
    // (canonical `kokoro-v1_0.pth`); the "plain" alternates below stay as
    // fallbacks for downstream forks that strip the prefix and for the
    // synthetic-tensor tests in this file (which do not carry the prefix).
    //
    // - voicepack[num_voices, style_dim]           — style-vector table
    //   (present only on forks that stack voices inline; upstream stores
    //   them as separate ``voices/*.pt`` files → derive style_dim from the
    //   AdaLN fc.weight ``[2·d_model, style_dim]`` axis 1 instead).
    // - text_encoder.module.embedding.weight[n_sym, hidden] — text embedding
    // - predictor.module.F0.0.norm1.fc.weight[2·d, style_dim] — StyleTTS 2
    //   AdaLN, gives style_dim when the voicepack tensor is absent.
    // - text_encoder.module.cnn.<i>.                — encoder CNN blocks
    //   (the ``lstm`` at the tail is a fixed final BiLSTM, not a counted
    //   layer, matching ``text_encoder::TextEncoder``).
    // - decoder.module.generator.ups.<i>.           — iSTFTNet upsample stages.
    let num_voices = tensor_dim(st, "voicepack", 0);
    // Prefer the ``voicepack`` axis 1 (a downstream fork's stacked style
    // table); fall back to the upstream ``predictor.module.F0.0.norm1.fc.weight``
    // whose axis 1 is exactly ``style_dim`` (StyleTTS 2 AdaLN structure).
    let style_dim = {
        let v = tensor_dim(st, "voicepack", 1);
        if v > 0 {
            v
        } else {
            tensor_dim(st, "predictor.module.F0.0.norm1.fc.weight", 1)
        }
    };
    let hidden_dim = {
        let v = tensor_dim(st, "text_encoder.embedding.weight", 1);
        if v > 0 {
            v
        } else {
            tensor_dim(st, "text_encoder.module.embedding.weight", 1)
        }
    };
    let n_text_layers = {
        let v = count_layers(st, "text_encoder.layers.");
        if v > 0 {
            v
        } else {
            count_layers(st, "text_encoder.module.cnn.")
        }
    };
    let n_decoder_layers = {
        let v = count_layers(st, "decoder.generator.upsamples.");
        if v > 0 {
            v
        } else {
            count_layers(st, "decoder.module.generator.ups.")
        }
    };

    // Config overrides (when passed): `voice_names.len()` becomes authoritative
    // for `num_voices` — Kokoro's canonical release ships voice styles as
    // separate ``voices/*.pt`` files (per the reference dumper's
    // ``open_checkpoint`` doc at ``tools/parity/dump_kokoro_reference.py``), so
    // the in-checkpoint ``voicepack`` tensor is often absent and the config is
    // the true source of truth. When both are present and disagree, we emit a
    // note rather than silently masking the mismatch.
    let mut notes: Vec<String> = Vec::new();
    let (num_voices_written, voice_names_out) = if let Some(cfg) = config {
        let cfg_n = cfg.voice_names.len();
        if num_voices > 0 && (num_voices as usize) != cfg_n {
            notes.push(format!(
                "kokoro config: voicepack rows ({}) != voice_names length ({}); \
                 using config-authoritative num_voices = {}",
                num_voices, cfg_n, cfg_n,
            ));
        }
        (cfg_n as u32, cfg.voice_names.clone())
    } else {
        (num_voices as u32, Vec::new())
    };

    b.add_u32(KEY_SAMPLE_RATE, KOKORO_SAMPLE_RATE);
    b.add_u32(KEY_STYLE_DIM, style_dim as u32);
    b.add_u32(KEY_NUM_VOICES, num_voices_written);
    b.add_u32(KEY_N_TEXT_LAYERS, n_text_layers);
    b.add_u32(KEY_N_DECODER_LAYERS, n_decoder_layers);
    b.add_u32(KEY_HIDDEN_DIM, hidden_dim as u32);
    // iSTFT hyper-parameters — pinned to Kokoro-82M manifest values (see
    // module-level constants).
    b.add_u32(KEY_ISTFT_N_FFT, KOKORO_ISTFT_N_FFT);
    b.add_u32(KEY_ISTFT_HOP, KOKORO_ISTFT_HOP);
    b.add_u32(KEY_ISTFT_WIN_LENGTH, KOKORO_ISTFT_WIN_LENGTH);
    // Phoneme symbols — when a caller passes ``--config config.json`` the real
    // misaki phoneme table lands in the GGUF verbatim. Otherwise synthesise a
    // placeholder table of the right size (n_vocab derived from the text
    // embedding axis 0). The runtime rejects an empty table (`kokoro text
    // encoder: config.phoneme_symbols is empty`), so a zero fallback would
    // block loading altogether — the placeholder path exists so a caller
    // without the misaki table can still exercise the numeric path
    // (T14/T15/T17 parity) with phoneme *ids* directly.
    let n_vocab_tensor = {
        let v = tensor_dim(st, "text_encoder.embedding.weight", 0);
        if v > 0 {
            v
        } else {
            tensor_dim(st, "text_encoder.module.embedding.weight", 0)
        }
    } as usize;
    let phoneme_symbols_values: Vec<GgufMetadataValue> = if let Some(cfg) = config {
        if n_vocab_tensor > 0 && cfg.phoneme_symbols.len() != n_vocab_tensor {
            notes.push(format!(
                "kokoro config: phoneme_symbols length ({}) != \
                 text_encoder.embedding.weight rows ({}); runtime will reject at load",
                cfg.phoneme_symbols.len(),
                n_vocab_tensor,
            ));
        }
        cfg.phoneme_symbols
            .iter()
            .map(|s| GgufMetadataValue::String(s.clone()))
            .collect()
    } else {
        (0..n_vocab_tensor)
            .map(|i| GgufMetadataValue::String(format!("p{i}")))
            .collect()
    };
    let phoneme_symbol_count = phoneme_symbols_values.len();
    b.add_metadata(
        KEY_PHONEME_SYMBOLS,
        GgufMetadataValue::Array(GgufArray {
            element_type: GgufValueType::String,
            values: phoneme_symbols_values,
        }),
    );
    let voice_name_values: Vec<GgufMetadataValue> = voice_names_out
        .iter()
        .map(|s| GgufMetadataValue::String(s.clone()))
        .collect();
    b.add_metadata(
        KEY_VOICE_NAMES,
        GgufMetadataValue::Array(GgufArray {
            element_type: GgufValueType::String,
            values: voice_name_values,
        }),
    );

    HparamOutcome {
        style_dim,
        phoneme_symbol_count,
        voice_names: voice_names_out,
        notes,
    }
}

/// Converts a Kokoro-82M safetensors buffer into a populated GGUF builder
/// plus a report of what was written vs. skipped.
///
/// Thin delegate to [`convert_with_config`] with `None` — kept as a stable
/// entry point for the `convert_file(ModelKind::Kokoro, …)` placeholder path
/// (backward compat: caller without a `--config config.json` still gets the
/// `p0..p_{n_vocab-1}` phoneme placeholders and an empty `voice_names` array).
pub(crate) fn convert(bytes: Vec<u8>) -> Result<(GgufBuilder, KokoroReport), ConvertError> {
    convert_with_config(bytes, None)
}

/// Converts a Kokoro-82M safetensors buffer (plus an optional Kokoro
/// `config.json` payload) into a populated GGUF builder and a report.
///
/// Every tensor is written verbatim (bytes, dtype and shape preserved); no
/// FP16 → FP32 widening (M2-07 keeps the source dtype so the follow-up
/// quantization policy can act on the same bytes the checkpoint shipped).
///
/// When `config_bytes` is `Some`, the parsed [`KokoroJsonConfig`] populates the
/// `vokra.kokoro.phoneme_symbols` / `.voice_names` arrays verbatim and
/// overrides `vokra.kokoro.num_voices` from `voice_names.len()`. Otherwise the
/// placeholder path is used (see [`convert`]).
pub(crate) fn convert_with_config(
    bytes: Vec<u8>,
    config_bytes: Option<&[u8]>,
) -> Result<(GgufBuilder, KokoroReport), ConvertError> {
    let st = SafetensorsFile::parse(bytes)?;
    let config = match config_bytes {
        Some(bytes) => Some(KokoroJsonConfig::parse(bytes)?),
        None => None,
    };

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    let outcome = write_hparams(&mut b, &st, config.as_ref());

    let mut report = KokoroReport {
        style_dim: outcome.style_dim as usize,
        phoneme_symbol_count: outcome.phoneme_symbol_count,
        voices: outcome.voice_names,
        notes: outcome.notes,
        ..Default::default()
    };

    for t in st.tensors() {
        match t.dtype {
            GgmlType::F32 | GgmlType::F16 => {
                b.add_tensor(
                    &t.name,
                    t.dtype,
                    t.shape.clone(),
                    st.tensor_bytes(t).to_vec(),
                )?;
                report.written += 1;
            }
            // Defensive: the upstream safetensors reader rejects non-F32/F16
            // at parse time, so this arm is currently unreachable through a
            // validly parsed buffer. Kept so a future reader extension does
            // not silently write an unsupported dtype (FR-EX-08).
            _ => {
                report.skipped_non_float += 1;
            }
        }
    }

    Ok((b, report))
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::GgufFile;

    /// Builds a synthetic Kokoro-like safetensors buffer with a small set of
    /// F32 tensors laid out contiguously.
    ///
    /// The names track the foundation shape-driver in [`write_hparams`] so
    /// every `vokra.kokoro.*` numeric hparam derives a non-zero value from
    /// this buffer; the payloads are minimal (all-zero) since only shapes
    /// drive the assertions.
    fn synthetic_kokoro_safetensors() -> Vec<u8> {
        // (name, shape) — element count = product; F32 payload = 4 * elems.
        let entries: &[(&str, &[u64])] = &[
            // voicepack [num_voices=2, style_dim=4] → 32 bytes.
            ("voicepack", &[2, 4]),
            // text_encoder.embedding.weight [n_sym=3, hidden=8] → 96 bytes.
            ("text_encoder.embedding.weight", &[3, 8]),
            // Two encoder blocks → contiguous prefix "text_encoder.layers.<i>."
            ("text_encoder.layers.0.attn.q_proj.weight", &[1, 1]),
            ("text_encoder.layers.1.attn.q_proj.weight", &[1, 1]),
            // One decoder upsample stage.
            ("decoder.generator.upsamples.0.weight", &[1, 1]),
            // A synthesis-side tensor.
            ("decoder.generator.conv_pre.weight", &[1, 1]),
            // A prosody predictor tensor.
            ("predictor.duration.weight", &[1, 1]),
        ];

        let mut cursor = 0usize;
        let mut header_entries = Vec::new();
        for &(name, shape) in entries {
            let elems: u64 = shape.iter().product();
            let span = elems as usize * 4;
            let begin = cursor;
            let end = cursor + span;
            cursor = end;
            let dims = shape
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join(",");
            header_entries.push(format!(
                r#""{name}":{{"dtype":"F32","shape":[{dims}],"data_offsets":[{begin},{end}]}}"#
            ));
        }
        let header = format!("{{{}}}", header_entries.join(","));
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(&vec![0u8; cursor]);
        out
    }

    #[test]
    fn converts_and_writes_kokoro_metadata_keys() {
        let (builder, report) = convert(synthetic_kokoro_safetensors()).expect("convert");
        let file = GgufFile::parse(builder.to_bytes().unwrap()).unwrap();

        // Model chunk present.
        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
            Some(ARCH)
        );
        assert_eq!(
            file.get(chunks::KEY_MODEL_NAME).and_then(|v| v.as_str()),
            Some(NAME)
        );

        // Every `vokra.kokoro.*` key from the T06 chunk design is present.
        let u = |k: &str| file.get(k).and_then(|v| v.as_u64());
        assert_eq!(u(KEY_SAMPLE_RATE), Some(u64::from(KOKORO_SAMPLE_RATE)));
        // Shape-driven derivations: `voicepack` [2, 4], embedding [3, 8], two
        // text_encoder.layers.<i>., one decoder.generator.upsamples.<i>.
        assert_eq!(u(KEY_NUM_VOICES), Some(2));
        assert_eq!(u(KEY_STYLE_DIM), Some(4));
        assert_eq!(u(KEY_HIDDEN_DIM), Some(8));
        assert_eq!(u(KEY_N_TEXT_LAYERS), Some(2));
        assert_eq!(u(KEY_N_DECODER_LAYERS), Some(1));
        // iSTFT triple: pinned to Kokoro-82M manifest constants.
        assert_eq!(u(KEY_ISTFT_N_FFT), Some(u64::from(KOKORO_ISTFT_N_FFT)));
        assert_eq!(u(KEY_ISTFT_HOP), Some(u64::from(KOKORO_ISTFT_HOP)));
        assert_eq!(
            u(KEY_ISTFT_WIN_LENGTH),
            Some(u64::from(KOKORO_ISTFT_WIN_LENGTH))
        );
        // String-array keys present. `phoneme_symbols` carries an
        // `n_vocab`-sized placeholder table (``p0..p_{n_vocab-1}``, sized from
        // the text embedding's axis-0) so a runtime load doesn't trip the
        // "empty vocab" check before the misaki phoneme table is wired via
        // ``--config config.json`` (follow-up ticket). ``voice_names`` stays
        // empty because voice styles ship as separate ``voices/*.pt`` files
        // in the canonical release.
        let syms = file
            .get(KEY_PHONEME_SYMBOLS)
            .and_then(|v| v.as_array())
            .expect("phoneme_symbols present");
        assert_eq!(syms.element_type, GgufValueType::String);
        // n_vocab = 3 in this synthetic buffer (embedding [3, 8]).
        assert_eq!(syms.values.len(), 3);
        let voices = file
            .get(KEY_VOICE_NAMES)
            .and_then(|v| v.as_array())
            .expect("voice_names present");
        assert_eq!(voices.element_type, GgufValueType::String);
        assert!(voices.values.is_empty());

        // No `vokra.frontend.*` chunk (Kokoro is TTS-only, no input front-end).
        assert!(file.get(chunks::KEY_FRONTEND_N_FFT).is_none());

        // Every input tensor round-tripped verbatim.
        assert_eq!(report.written, 7);
        assert_eq!(report.skipped_non_float, 0);
        assert_eq!(report.style_dim, 4);
        // Bytes preserved for at least one representative tensor.
        let info = file.tensor_info("voicepack").expect("voicepack in gguf");
        assert_eq!(info.dtype, GgmlType::F32);
        assert_eq!(info.dimensions, vec![2, 4]);
    }

    #[test]
    fn skips_non_float_and_reports() {
        // The upstream safetensors reader admits only F32 and F16
        // (`SafetensorsFile::parse` returns `UnsupportedDtype` on anything
        // else), so a validly parsed buffer that reaches `convert()` cannot
        // hold a truly non-float tensor. Two complementary assertions still
        // verify the reporter contract:
        //
        // (1) The `skipped_non_float` counter is present on the report and
        //     correctly reports `0` for an all-F32 buffer (the counter is
        //     defensive/forward-compat — if the safetensors reader is later
        //     extended, the skip path already exists).
        // (2) A safetensors buffer whose *declared* dtype is non-float (I64
        //     here) is rejected at parse time with a `ConvertError::Parse`
        //     wrapping the reader's `UnsupportedDtype` — non-float bytes never
        //     silently reach `add_tensor`.
        let (_, report) = convert(synthetic_kokoro_safetensors()).expect("convert");
        assert_eq!(
            report.skipped_non_float, 0,
            "all-F32 buffer must report zero skipped tensors"
        );
        assert!(
            report.written > 0,
            "all-F32 buffer must report written tensors"
        );

        // Non-float buffer: an I64 tensor in a safetensors header. Parse-side
        // rejection is the runtime's non-float gate for this converter today.
        let header = r#"{"i.const":{"dtype":"I64","shape":[1],"data_offsets":[0,8]}}"#;
        let mut buf = Vec::new();
        buf.extend_from_slice(&(header.len() as u64).to_le_bytes());
        buf.extend_from_slice(header.as_bytes());
        buf.extend_from_slice(&[0u8; 8]);
        let err = convert(buf).expect_err("I64 must be rejected at parse time");
        let msg = format!("{err}");
        assert!(
            msg.contains("I64") || msg.contains("dtype"),
            "expected parse-side non-float rejection, got: {msg}"
        );
    }

    // -----------------------------------------------------------------------
    // KokoroJsonConfig parser + convert_with_config
    // -----------------------------------------------------------------------

    #[test]
    fn config_parses_vocab_map_and_voices_array() {
        // `vocab: {symbol: id}` (Kokoro / misaki spelling), `voices: [str]`
        // (canonical release spelling). Table length = max(id)+1; missing
        // slots stay as "".
        let raw = br#"{"vocab":{"_":0,"a":1,"b":3},"voices":["af","am_michael","bf_emma"]}"#;
        let cfg = KokoroJsonConfig::parse(raw).expect("parse vocab+voices");
        assert_eq!(cfg.phoneme_symbols.len(), 4);
        assert_eq!(cfg.phoneme_symbols[0], "_");
        assert_eq!(cfg.phoneme_symbols[1], "a");
        assert_eq!(cfg.phoneme_symbols[2], ""); // gap left blank
        assert_eq!(cfg.phoneme_symbols[3], "b");
        assert_eq!(cfg.voice_names, vec!["af", "am_michael", "bf_emma"]);
    }

    #[test]
    fn config_parses_phoneme_symbols_and_voice_names_aliases() {
        // Fallback shape: `phoneme_symbols: [str]` array + `voice_names: [str]`
        // alias. First-match wins so this only fires when `vocab` / `voices`
        // are absent.
        let raw = br#"{"phoneme_symbols":["_","a","b"],"voice_names":["v0","v1"]}"#;
        let cfg = KokoroJsonConfig::parse(raw).expect("parse array shapes");
        assert_eq!(cfg.phoneme_symbols, vec!["_", "a", "b"]);
        assert_eq!(cfg.voice_names, vec!["v0", "v1"]);
    }

    #[test]
    fn config_parses_symbols_alias() {
        // Third accepted shape for the symbol family: `symbols: [str]`.
        let raw = br#"{"symbols":["_","a"],"voices":["v0"]}"#;
        let cfg = KokoroJsonConfig::parse(raw).expect("parse symbols alias");
        assert_eq!(cfg.phoneme_symbols, vec!["_", "a"]);
        assert_eq!(cfg.voice_names, vec!["v0"]);
    }

    #[test]
    fn config_rejects_missing_symbol_family() {
        let raw = br#"{"voices":["v0"]}"#;
        let err = KokoroJsonConfig::parse(raw).expect_err("missing symbols");
        assert!(
            format!("{err}").contains("no phoneme symbols found"),
            "expected missing-symbols error, got: {err}"
        );
    }

    #[test]
    fn config_rejects_missing_voice_family() {
        let raw = br#"{"symbols":["_","a"]}"#;
        let err = KokoroJsonConfig::parse(raw).expect_err("missing voices");
        assert!(
            format!("{err}").contains("no voice list found"),
            "expected missing-voices error, got: {err}"
        );
    }

    #[test]
    fn convert_with_config_overrides_symbols_and_voices() {
        // Synthetic checkpoint: `text_encoder.embedding.weight [3, 8]` ⇒
        // n_vocab = 3, and `voicepack [2, 4]` ⇒ tensor num_voices = 2. Pass a
        // config with 3 symbols and 3 voices — the config voice count wins
        // (voicepack-vs-config mismatch surfaces as a note, not a fail).
        let ckpt = synthetic_kokoro_safetensors();
        let cfg = br#"{"phoneme_symbols":["_","a","b"],"voices":["af","am_michael","bf_emma"]}"#;
        let (builder, report) = convert_with_config(ckpt, Some(cfg)).expect("convert_with_config");
        let file = GgufFile::parse(builder.to_bytes().unwrap()).unwrap();

        let syms = file
            .get(KEY_PHONEME_SYMBOLS)
            .and_then(|v| v.as_array())
            .expect("phoneme_symbols present");
        assert_eq!(syms.values.len(), 3);
        // First symbol is the actual name, not the `p0` placeholder.
        assert_eq!(
            syms.values[0],
            GgufMetadataValue::String("_".to_owned()),
            "config symbols override the p{{i}} placeholder"
        );

        let voices = file
            .get(KEY_VOICE_NAMES)
            .and_then(|v| v.as_array())
            .expect("voice_names present");
        assert_eq!(voices.values.len(), 3);
        assert_eq!(
            voices.values[1],
            GgufMetadataValue::String("am_michael".to_owned())
        );

        // `num_voices` is config-authoritative (=3), not the voicepack tensor
        // count (=2). Disagreement surfaces as a report note.
        assert_eq!(
            file.get(KEY_NUM_VOICES).and_then(|v| v.as_u64()),
            Some(3),
            "config voice_names.len() overrides voicepack rows"
        );
        assert_eq!(report.voices.len(), 3);
        assert_eq!(report.phoneme_symbol_count, 3);
        assert!(
            report.notes.iter().any(|n| n.contains("voicepack rows")),
            "expected voicepack-vs-voice_names mismatch note, got: {:?}",
            report.notes
        );
    }

    #[test]
    fn convert_with_config_notes_phoneme_count_mismatch() {
        // n_vocab tensor axis-0 = 3; config supplies only 2 symbols. Runtime
        // will reject at load — the converter surfaces a note but does not
        // fail (FR-EX-08: runtime is the authoritative gate).
        let ckpt = synthetic_kokoro_safetensors();
        let cfg = br#"{"phoneme_symbols":["_","a"],"voices":["v0"]}"#;
        let (_, report) = convert_with_config(ckpt, Some(cfg)).expect("convert_with_config");
        assert!(
            report
                .notes
                .iter()
                .any(|n| n.contains("phoneme_symbols length")),
            "expected phoneme-symbol-count mismatch note, got: {:?}",
            report.notes
        );
    }
}
