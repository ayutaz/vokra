//! **MT3** (`magenta/mt3`, apache-2.0 code / weight license UNCLEAR)
//! — Multi-Task Multitrack Music Transcription (Gardner et al. ICLR
//! 2022, arXiv:2111.03017) runtime binder for the `mt3` converter arch.
//!
//! # Runtime layout (loud-partial, RMVPE / pyannote / hifigan Wave 1
//! precedent — CLAUDE.md 教訓 (a))
//!
//! ```text
//! PCM (mono f32, 16 kHz per MT3 upstream spec)
//!   -> log-mel front-end                             ← **loud-partial**
//!        (Vokra's shared STFT + Mel-filterbank primitives cover the
//!         front-end math; the wire-up onto MT3's exact mel spec
//!         lands with the T5 encoder-decoder forward wave.)
//!   -> T5-small encoder (12 layers × MHA + FFN)      ← **loud-partial**
//!        (T5 relative-attention-bias multi-head attention needs a
//!         `t5_relative_attention_bias` primitive that does NOT exist
//!         in `vokra-ops` today — every Transformer model in the
//!         tree (whisper / canary / voxtral) re-implements attention
//!         from `softmax` + `GEMM` + `LayerNorm`, but T5's
//!         *relative* attention-bias bucketing (Raffel et al. 2020
//!         §2.1, distinct from DeBERTa's `make_log_bucket_position`
//!         used by `vokra-bert`) is a T5-specific primitive that no
//!         sibling supplies. First landing of this primitive is
//!         out-of-scope for this WP.)
//!   -> T5-small decoder (12 layers, autoregressive)  ← **loud-partial**
//!        (Same T5 relative-attention-bias gap on the decoder side +
//!         cross-attention to encoder output.)
//!   -> MIDI event token stream                       ← **loud-partial**
//!        (Post-processing via a Rust port of `mt3/event_codec.py`
//!         maps decoder tokens → `MidiEvent` variants covering
//!         note-on / note-off / program-change / velocity across
//!         multiple simultaneous instruments. The port itself is
//!         out-of-scope for this WP.)
//!   -> Vec<MidiEvent>
//! ```
//!
//! # Loud-partial classification (design §)
//!
//! - **Real (this WP)**: [`Mt3::from_gguf`] with strict
//!   `vokra.model.arch == "mt3"` validation + strict `vokra.mt3.*`
//!   chunk-group presence enforcement (every T5 topology axis
//!   required — no primary-source constant fallback because the
//!   upstream MT3 checkpoint on `gs://mt3/checkpoints/` does NOT
//!   ship a first-class `config.json`; the converter transcribes
//!   the T5-small axes from `magenta/mt3/mt3/network.py` and stamps
//!   them, and this binder mirrors those stamps rather than
//!   silently defaulting to a fabricated axis), [`Mt3Weights::from_gguf`]
//!   with a floor of non-empty tensor count enforced loud (a GGUF
//!   that carries no MT3-typical tensors is refused rather than
//!   silently running an all-zero forward), [`MidiEvent`] enum
//!   defined per task hint with the four variants covering the
//!   MIDI event token stream MT3 emits, license-class surfacing.
//! - **Loud-partial (this WP)**: [`Mt3::transcribe`] returns
//!   [`VokraError::UnsupportedOp`] naming the two exact missing
//!   pieces (T5 `t5_relative_attention_bias` primitive absent from
//!   `vokra-ops` + MIDI event codec Rust port not yet written) and
//!   citing the primary source URLs so a reader diagnosing this gap
//!   has exactly two places to walk.
//!
//! Rationale (RMVPE / pyannote / hifigan / vocos / bigvgan / snac /
//! beat_this Wave 1 precedent, CLAUDE.md 教訓 (a)): the surrounding
//! scaffold + `from_gguf` chunk-group validation + `MidiEvent` enum
//! surface + FR-EX-08 loud-fails land today so a follow-up wave can
//! flip the switch by (i) landing the T5 relative-attention-bias
//! primitive in `vokra-ops` (distinct from `vokra-bert`'s DeBERTa
//! log-bucket bucketing per Raffel et al. 2020 §2.1) plus (ii)
//! porting `mt3/event_codec.py` to Rust and writing the T5X
//! checkpoint flattener — a future
//! `tools/parity/mt3_prepare_checkpoint.py` (not yet written;
//! uv-managed Python 3.12 sidecar per memory
//! `[[feedback-python-uses-uv]]` + `[[feedback-python-3-12]]`).
//! The [`VokraError::UnsupportedOp`] message cites both primary
//! sources so the follow-up wave has exactly the two anchors it
//! needs to walk.
//!
//! # `vokra.mt3.*` chunk group (read here)
//!
//! Written by `vokra-convert::models::mt3::convert_mt3_file`:
//!
//! - `vokra.model.arch` (`String`): must equal [`ARCH`] (`"mt3"`).
//!   Distinct from every sibling music-tree arch (`basic-pitch`
//!   Spotify polyphonic-CNN, `beat-this` Transformer beat-tracker,
//!   `musicgen` text-to-music AR LM) and every T5 speech-tree arch
//!   — silently sharing would misroute runtime dispatch (FR-EX-08).
//! - `vokra.model.name` (`String`): `"mt3-multitrack"` — auxiliary check.
//! - `vokra.mt3.{d_model, d_ff, n_heads, d_kv, num_enc_layers,
//!   num_dec_layers, music_vocab_size, rel_attn_num_buckets,
//!   rel_attn_max_distance}` (`u32` each): the T5-small topology
//!   axes + MT3-specific music-vocab / relative-attention params.
//! - `vokra.provenance.*`: license class + raw license string, so the
//!   runtime compliance gate (FR-CP-03) can classify the artifact
//!   without re-inspecting the safetensors provenance. **The MT3
//!   converter always stamps `LicenseClass::Unknown` here regardless
//!   of the raw SPDX** — fail-closed policy because the MT3 weight
//!   bucket has no per-bucket LICENSE and no HF mirror as of
//!   2026-08-14.
//!
//! # Cross-crate constant duplication
//!
//! Mirror of the converter's [`ARCH`] / [`GGUF_KEY_*`] — same rule
//! the sibling BF16 pass-through binders (`pyannote` / `snac` /
//! `hifigan` / `beat_this`) use so `vokra-models` does not gain a
//! dependency edge onto `vokra-convert`, preserving the layered
//! convention `vokra-ops → nothing GGUF-aware`, `vokra-core → GGUF
//! reader`, `vokra-models → GGUF binder`, `vokra-convert → GGUF
//! writer`.
//!
//! # No ONNX / no JAX / no pickle (permanent)
//!
//! MT3 ships upstream as a **T5X / JAX checkpoint** on
//! `gs://mt3/checkpoints/` (no HF mirror). This runtime **never**
//! touches ONNX, JAX, or pickle (FR-LD-05 / NFR-DS-02). The T5X →
//! safetensors bridge is a future
//! `tools/parity/mt3_prepare_checkpoint.py` (**not yet written** — an
//! offline uv-managed Python 3.12 sidecar, never part of the runtime),
//! mirroring the DAC / Kokoro / UTMOSv2 / beats bridge pattern.

use vokra_core::gguf::{GgufFile, chunks};
use vokra_core::{LicenseClass, Result, VokraError};

// ---------------------------------------------------------------------------
// Arch / metadata-key constants — mirror of
// `crates/vokra-convert/src/models/mt3.rs` (see module docstring for
// the cross-crate duplication rationale).
// ---------------------------------------------------------------------------

/// Expected `vokra.model.arch` value written by
/// `vokra-cli convert --model mt3`.
///
/// Distinct from every sibling music-tree arch (`basic-pitch` /
/// `beat-this` / `musicgen`) and every T5 speech-tree arch. Sharing
/// arch would let runtime dispatch bind an MT3 T5 encoder-decoder
/// binder over a Basic-Pitch CNN checkpoint (or vice versa), a
/// silent-wrong shape mismatch FR-EX-08 forbids.
pub const ARCH: &str = "mt3";

/// `vokra.mt3.d_model` — T5-small hidden dimension.
pub const GGUF_KEY_D_MODEL: &str = "vokra.mt3.d_model";
/// `vokra.mt3.d_ff` — T5-small FFN inner dimension.
pub const GGUF_KEY_D_FF: &str = "vokra.mt3.d_ff";
/// `vokra.mt3.n_heads` — T5-small attention head count.
pub const GGUF_KEY_N_HEADS: &str = "vokra.mt3.n_heads";
/// `vokra.mt3.d_kv` — T5-small per-head dimension (distinct from
/// `d_model / n_heads` in T5, Raffel et al. 2020).
pub const GGUF_KEY_D_KV: &str = "vokra.mt3.d_kv";
/// `vokra.mt3.num_enc_layers` — MT3 encoder stack depth.
pub const GGUF_KEY_NUM_ENC_LAYERS: &str = "vokra.mt3.num_enc_layers";
/// `vokra.mt3.num_dec_layers` — MT3 decoder stack depth.
pub const GGUF_KEY_NUM_DEC_LAYERS: &str = "vokra.mt3.num_dec_layers";
/// `vokra.mt3.music_vocab_size` — MT3 music-event vocabulary size.
pub const GGUF_KEY_MUSIC_VOCAB_SIZE: &str = "vokra.mt3.music_vocab_size";
/// `vokra.mt3.rel_attn_num_buckets` — T5 relative-attention bucket count.
pub const GGUF_KEY_REL_ATTN_NUM_BUCKETS: &str = "vokra.mt3.rel_attn_num_buckets";
/// `vokra.mt3.rel_attn_max_distance` — T5 relative-attention max distance.
pub const GGUF_KEY_REL_ATTN_MAX_DISTANCE: &str = "vokra.mt3.rel_attn_max_distance";

/// Primary-source anchor for the T5 relative-attention-bias primitive
/// gap. Cited in the loud-partial error so a reader diagnosing this
/// gap knows the T5 reference implementation to walk.
const PRIMARY_SOURCE_T5_NETWORK: &str = "github.com/magenta/mt3/blob/main/mt3/network.py";
/// Primary-source anchor for the MIDI event codec gap. Cited in the
/// loud-partial error so a reader diagnosing this gap knows the event
/// codec reference implementation to port.
const PRIMARY_SOURCE_EVENT_CODEC: &str = "github.com/magenta/mt3/blob/main/mt3/event_codec.py";
/// Paper anchor (Gardner et al. ICLR 2022) — cited alongside the two
/// source URLs so a reader has the theoretical context as well.
const PRIMARY_SOURCE_PAPER: &str = "arxiv.org/abs/2111.03017";

// ---------------------------------------------------------------------------
// Mt3Config — the T5 topology axes read from the `vokra.mt3.*` chunk
// group. STRICT: every axis is required (FR-EX-08 — no primary-source
// constant fallback since the upstream MT3 checkpoint does NOT ship a
// first-class `config.json` on `gs://mt3/checkpoints/`; the converter
// transcribes the axes from `network.py` and stamps them, and this
// binder mirrors those stamps rather than silently defaulting).
// ---------------------------------------------------------------------------

/// MT3 T5-small hyperparameters as they ride the `vokra.mt3.*` chunk
/// group.
///
/// [`from_gguf`](Self::from_gguf) is a **strict** loader: every axis
/// is required (FR-EX-08 — never a silent primary-source constant
/// fallback because the upstream MT3 checkpoint does not carry a
/// first-class `config.json`, so any fallback here would fabricate
/// axes the runtime then binds against). A GGUF missing any
/// `vokra.mt3.*` chunk is rejected loudly with a
/// [`VokraError::ModelLoad`] naming the absent key.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Mt3Config {
    /// T5-small hidden dimension (`d_model`), typically 512.
    pub d_model: u32,
    /// T5-small FFN inner dimension (`d_ff`), typically 1024.
    pub d_ff: u32,
    /// T5-small attention head count (`n_heads`), typically 6.
    pub n_heads: u32,
    /// T5-small per-head dimension (`d_kv`), typically 64 —
    /// intentionally not `d_model / n_heads` per T5 (Raffel et al.
    /// 2020).
    pub d_kv: u32,
    /// MT3 encoder stack depth (`num_encoder_layers`), typically 12.
    pub num_enc_layers: u32,
    /// MT3 decoder stack depth (`num_decoder_layers`), typically 12.
    pub num_dec_layers: u32,
    /// MT3 music-event vocabulary size (approximate `1200` from
    /// `event_codec.py`).
    pub music_vocab_size: u32,
    /// T5 relative-attention bucket count (`num_buckets`, T5 default 32).
    pub rel_attn_num_buckets: u32,
    /// T5 relative-attention max distance (`max_distance`, T5 default 128).
    pub rel_attn_max_distance: u32,
}

impl Mt3Config {
    /// The T5-small defaults transcribed from
    /// `github.com/magenta/mt3/blob/main/mt3/network.py`. Used by the
    /// unit tests and as a diagnostic reference — the runtime loader
    /// does NOT default to these; it reads the stamped values and
    /// fails loud on any missing chunk (see [`Self::from_gguf`]).
    #[must_use]
    pub const fn t5_small_default() -> Self {
        Self {
            d_model: 512,
            d_ff: 1024,
            n_heads: 6,
            d_kv: 64,
            num_enc_layers: 12,
            num_dec_layers: 12,
            music_vocab_size: 1200,
            rel_attn_num_buckets: 32,
            rel_attn_max_distance: 128,
        }
    }

    /// Reads every `vokra.mt3.*` chunk from `gguf`. Missing axis =
    /// loud [`VokraError::ModelLoad`] naming the absent key (FR-EX-08
    /// — no primary-source constant fallback since the upstream MT3
    /// checkpoint does not carry a first-class `config.json`; any
    /// fallback here would fabricate axes without primary-source
    /// backing).
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] when any of the nine mandatory
    ///   `vokra.mt3.*` u32 chunks is absent.
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        fn req_u32(gguf: &GgufFile, key: &str) -> Result<u32> {
            gguf.get(key)
                .and_then(|v| v.as_u64())
                .map(|v| v as u32)
                .ok_or_else(|| {
                    VokraError::ModelLoad(format!(
                        "mt3: GGUF is missing required u32 chunk `{key}` — the \
                         upstream `magenta/mt3` checkpoint on `gs://mt3/checkpoints/` \
                         does not carry a first-class `config.json`, so this runtime \
                         binder refuses to fabricate topology axes from primary-source \
                         constants (FR-EX-08). Re-run `vokra-cli convert --model mt3` \
                         with a T5X checkpoint flattened to safetensors offline (a \
                         future `tools/parity/mt3_prepare_checkpoint.py` is not yet \
                         written, so that step is manual today) — the converter \
                         transcribes the T5-small axes from \
                         `github.com/magenta/mt3/blob/main/mt3/network.py` and stamps \
                         them, so a proper conversion carries every mandatory chunk)."
                    ))
                })
        }
        Ok(Self {
            d_model: req_u32(gguf, GGUF_KEY_D_MODEL)?,
            d_ff: req_u32(gguf, GGUF_KEY_D_FF)?,
            n_heads: req_u32(gguf, GGUF_KEY_N_HEADS)?,
            d_kv: req_u32(gguf, GGUF_KEY_D_KV)?,
            num_enc_layers: req_u32(gguf, GGUF_KEY_NUM_ENC_LAYERS)?,
            num_dec_layers: req_u32(gguf, GGUF_KEY_NUM_DEC_LAYERS)?,
            music_vocab_size: req_u32(gguf, GGUF_KEY_MUSIC_VOCAB_SIZE)?,
            rel_attn_num_buckets: req_u32(gguf, GGUF_KEY_REL_ATTN_NUM_BUCKETS)?,
            rel_attn_max_distance: req_u32(gguf, GGUF_KEY_REL_ATTN_MAX_DISTANCE)?,
        })
    }
}

// ---------------------------------------------------------------------------
// Mt3Weights — bound the tensor manifest with a non-emptiness gate.
// Under the loud-partial WP the weights are counted but the T5
// encoder-decoder forward is deferred (the encoder + decoder + event
// codec post-processing would consume them). Mirrors the
// `BeatThisWeights` posture from `crates/vokra-models/src/beat_this/mod.rs`.
// ---------------------------------------------------------------------------

/// Weight tensors bound from an MT3 GGUF.
///
/// **Contract**: [`from_gguf`](Self::from_gguf) is a *loud*
/// verification step. A GGUF that carries zero tensors is rejected
/// with [`VokraError::ModelLoad`] (FR-EX-08 — an empty GGUF is never
/// a valid MT3 checkpoint).
///
/// Under the current landing this struct stores the tensor names +
/// dims discovered on disk. The T5 encoder + decoder + event-codec
/// forward is deferred (see [`Mt3::transcribe`] loud-partial), so
/// the payload is not yet dequantised — the follow-up wave sizes
/// the dequant per its kernel needs.
#[derive(Debug)]
pub struct Mt3Weights {
    /// Tensors discovered on disk, indexed by upstream `state_dict`
    /// name with their GGUF-side dims. Used by the load-time
    /// non-emptiness gate and by the future follow-up
    /// encoder-decoder-forward wave.
    tensors: Vec<(String, Vec<usize>)>,
}

impl Mt3Weights {
    /// Scans `gguf` for the MT3 state_dict tensors. Refuses to bind
    /// if the GGUF carries zero tensors (FR-EX-08 — an empty GGUF is
    /// never a valid MT3 checkpoint).
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
                "mt3: GGUF carries zero tensors — refusing to bind an all-zero \
                 forward (FR-EX-08). Re-run `vokra-cli convert --model mt3` \
                 against a T5X checkpoint flattened to safetensors offline (a future \
                 `tools/parity/mt3_prepare_checkpoint.py` is not yet written)."
                    .to_owned(),
            ));
        }
        Ok(Self { tensors })
    }

    /// Number of tensors bound from the GGUF. Purely a diagnostic
    /// accessor — the encoder-decoder forward wave uses it to size
    /// its expectations.
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    /// Load-time shape gate — validates that at least one bound
    /// tensor has an axis matching `config.d_model`. Under the
    /// current landing this is a **soft** gate (mismatch is silently
    /// ignored) because the T5 encoder-decoder wave has not yet
    /// pinned the exact tensor-name convention the T5X flatten step
    /// will use — a hard shape assertion today would fail against
    /// every legitimate future flatten shape.
    ///
    /// The follow-up wave will replace this soft accessor with a
    /// hard pin against the primary-source-verified tensor-name walk
    /// (mirror of `pyannote::PyanNetWeights::verify_core_shapes`).
    ///
    /// Kept as a `#[must_use]` accessor so the read is deliberate.
    #[must_use]
    pub fn matches_config(&self, config: &Mt3Config) -> bool {
        let d = config.d_model as usize;
        self.tensors.iter().any(|(_, dims)| dims.contains(&d))
    }
}

// ---------------------------------------------------------------------------
// MidiEvent — the public output surface for `Mt3::transcribe` once
// the T5 relative-attention-bias primitive + MIDI event codec Rust
// port land. Defined here per the task hint ("If MidiEvent type does
// not exist, define a minimal enum {note_on, note_off, program_change,
// velocity} in this module") — pinned as the surface a future
// forward wave binds against.
// ---------------------------------------------------------------------------

/// A MIDI event emitted by MT3's decoder token stream after
/// post-processing via the future Rust port of
/// `github.com/magenta/mt3/blob/main/mt3/event_codec.py`.
///
/// The four variants match the four task-listed MIDI event kinds MT3
/// produces (`note_on` / `note_off` / `program_change` / `velocity`).
/// Every variant carries a `tick` field (u64) locating it on MT3's
/// output timeline — the exact tick resolution (typically 100 Hz or
/// per-frame at the encoder's mel resolution) is a follow-up wave
/// decision that depends on the event codec Rust port's chosen
/// convention.
///
/// This enum is a **surface pin**: the future forward wave lands
/// consumers that construct these variants, and this pin ensures the
/// variant shape is stable at type-check time from the day this WP
/// lands. A rename or field-shape change would need to land here in
/// the same commit or fail the surface pin test at the bottom of
/// this module.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MidiEvent {
    /// A note-on event — a specific pitch (MIDI note number 0-127)
    /// begins sounding on a specific instrument program with a
    /// specific velocity, at a specific tick on the output timeline.
    NoteOn {
        /// MIDI pitch (0-127), the standard General MIDI note range.
        pitch: u8,
        /// Attack velocity (0-127). Zero is per MIDI convention
        /// treated as a note-off by many synthesizers; MT3 emits
        /// distinct [`Self::NoteOff`] events instead.
        velocity: u8,
        /// General MIDI program number (0-127) identifying the
        /// instrument voice.
        program: u8,
        /// Timeline tick — position on MT3's decoder output timeline.
        tick: u64,
    },
    /// A note-off event — a previously-sounding pitch on a specific
    /// program stops sounding.
    NoteOff {
        /// MIDI pitch (0-127) being released.
        pitch: u8,
        /// General MIDI program number (0-127) whose voice is
        /// releasing this pitch.
        program: u8,
        /// Timeline tick — position on MT3's decoder output timeline.
        tick: u64,
    },
    /// A program-change event — subsequent note events use a new
    /// General MIDI program number.
    ProgramChange {
        /// New General MIDI program number (0-127).
        program: u8,
        /// Timeline tick — position on MT3's decoder output timeline.
        tick: u64,
    },
    /// A velocity-quantum event — sets the default velocity for
    /// subsequent note events until the next velocity token. MT3's
    /// event codec quantizes velocity into discrete tokens rather
    /// than embedding it on every note event.
    Velocity {
        /// Velocity quantum (0-127).
        velocity: u8,
        /// Timeline tick — position on MT3's decoder output timeline.
        tick: u64,
    },
}

// ---------------------------------------------------------------------------
// Mt3 — the runtime binder handle
// ---------------------------------------------------------------------------

/// MT3 T5-small multi-track music transcription (Google Magenta,
/// apache-2.0 code / weight license UNCLEAR).
///
/// Bind with [`from_gguf`](Self::from_gguf), then call
/// [`transcribe`](Self::transcribe) on a PCM buffer to obtain a
/// `Vec<MidiEvent>`. See the module doc for the current
/// implementation-status matrix and the FR-EX-08 loud-error contract
/// on the T5 encoder-decoder forward + event codec post-processing.
#[derive(Debug)]
pub struct Mt3 {
    config: Mt3Config,
    // The bound weights are held (real, counted) but the T5
    // encoder-decoder forward + event codec post-processing is a
    // follow-up wave; the field is deliberately `#[allow(dead_code)]`
    // until the kernel lands so a reader is not misled by an unused
    // field. Same posture as RMVPE / pyannote / Charsiu / beat_this.
    #[allow(dead_code)]
    weights: Mt3Weights,
    weight_license: LicenseClass,
}

impl Mt3 {
    /// Binds an MT3 GGUF: validates arch, reads the strict topology
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
    ///   or not `"mt3"` (a `basic-pitch` / `beat-this` / `musicgen`
    ///   / any T5 speech-tree GGUF handed to us by mistake fails
    ///   with a clear message instead of a downstream "missing
    ///   tensor" — same pattern as `BeatThis::from_gguf`).
    /// - [`VokraError::ModelLoad`] when any `vokra.mt3.*` chunk is
    ///   absent ([`Mt3Config::from_gguf`] is strict — no
    ///   primary-source constant fallback since the upstream
    ///   checkpoint does not carry a first-class `config.json`).
    /// - [`VokraError::ModelLoad`] when the GGUF carries zero
    ///   tensors ([`Mt3Weights::from_gguf`] refuses to bind an
    ///   all-zero forward).
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        // 1. Arch check — always first so a mis-typed model handed
        //    here fails with a specific message instead of a
        //    downstream "vokra.mt3.d_model missing".
        match file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()) {
            Some(a) if a == ARCH => {}
            Some(other) => {
                return Err(VokraError::ModelLoad(format!(
                    "mt3: GGUF arch is `{other}`, expected `{ARCH}` (was this GGUF \
                     produced by `vokra-cli convert --model mt3`? Note that sibling \
                     music-tree arches — `basic-pitch` (Spotify polyphonic-CNN \
                     posteriorgram), `beat-this` (Transformer beat + downbeat \
                     tracker), `musicgen` (text-to-music AR LM) — are completely \
                     different topologies)"
                )));
            }
            None => {
                return Err(VokraError::ModelLoad(
                    "mt3: GGUF is missing `vokra.model.arch` (converter did not \
                     stamp it — this is not a Vokra-native mt3 GGUF)"
                        .to_owned(),
                ));
            }
        }

        // 2. Strict topology axes from the `vokra.mt3.*` chunk group.
        let config = Mt3Config::from_gguf(file)?;

        // 3. Load the tensor manifest with the non-emptiness gate.
        let weights = Mt3Weights::from_gguf(file)?;

        // 4. Provenance surfacing — read the stamped weight-license
        //    class for compliance gate cross-checks (defaults to
        //    `Unknown` if absent, which is fail-closed at the gate).
        //    The MT3 converter always stamps `Unknown` here
        //    regardless of the raw SPDX (see converter fn rustdoc),
        //    so this read is expected to return `Unknown` in
        //    production. Not raising a `ModelLoad` on missing
        //    provenance keeps the binder able to load hand-assembled
        //    GGUFs the test harness uses without forcing every
        //    fixture to stamp the full provenance chunk.
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

    /// The bound topology axes (read from the `vokra.mt3.*` chunk
    /// group).
    #[inline]
    #[must_use]
    pub const fn config(&self) -> &Mt3Config {
        &self.config
    }

    /// The stamped weight-license class surfaced from the GGUF's
    /// `vokra.provenance.weight_license` chunk. The MT3 converter
    /// always stamps `Unknown` here (fail-closed policy — MT3 weight
    /// bucket has no per-bucket LICENSE and no HF mirror as of
    /// 2026-08-14), so this accessor typically returns
    /// [`LicenseClass::Unknown`]. A GGUF missing the stamp also
    /// reads back as `Unknown`.
    #[inline]
    #[must_use]
    pub const fn weight_license(&self) -> LicenseClass {
        self.weight_license
    }

    /// Number of tensors bound from the GGUF. Purely a diagnostic
    /// accessor — the encoder-decoder forward wave uses it to size
    /// its expectations.
    #[inline]
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.weights.tensor_count()
    }

    /// Transcribes a mono PCM buffer to a MIDI event stream.
    ///
    /// # Loud-partial (this WP)
    ///
    /// Returns [`VokraError::UnsupportedOp`] — the MT3 T5-small
    /// **encoder-decoder forward** requires a
    /// `t5_relative_attention_bias` primitive that does NOT exist in
    /// `vokra-ops` today (every Transformer model in the tree —
    /// `whisper` / `canary` / `voxtral` — re-implements attention
    /// from `softmax` + `GEMM` + `LayerNorm`, but T5's *relative*
    /// attention-bias bucketing (Raffel et al. 2020 §2.1, distinct
    /// from DeBERTa's `make_log_bucket_position` used by
    /// `vokra-bert`) is a T5-specific primitive that no sibling
    /// supplies). In addition the **MIDI event codec Rust port** of
    /// `github.com/magenta/mt3/blob/main/mt3/event_codec.py` has
    /// not been written — the decoder token stream cannot become a
    /// `Vec<MidiEvent>` without it. The error message names both
    /// primary source URLs so a reader diagnosing this gap has
    /// exactly two places to walk.
    ///
    /// # Errors
    ///
    /// - [`VokraError::UnsupportedOp`] — the loud-partial gate for
    ///   the deferred T5 encoder-decoder forward + event codec
    ///   Rust port.
    pub fn transcribe(&self, pcm: &[f32]) -> Result<Vec<MidiEvent>> {
        // Bind unused arg so a `#[warn(unused_variables)]` change
        // does not silently mask the loud-partial fire path; the
        // future real implementation will consume it.
        let _ = pcm;
        Err(transcribe_forward_loud_partial(&self.config))
    }
}

/// Constructs the loud-partial [`VokraError::UnsupportedOp`]
/// returned by [`Mt3::transcribe`] until the T5
/// relative-attention-bias primitive lands + the MIDI event codec
/// Rust port lands.
///
/// Names **both** primary source URLs (T5 network reference +
/// event codec reference) so a reader diagnosing the gap has
/// exactly two places to walk (RMVPE / pyannote / snac / hifigan /
/// beat_this Wave 1 loud-partial-message precedent — CLAUDE.md 教訓
/// (a)).
fn transcribe_forward_loud_partial(cfg: &Mt3Config) -> VokraError {
    VokraError::UnsupportedOp(format!(
        "mt3 transcribe: T5 encoder-decoder forward + MIDI event codec Rust port \
         pending — no `t5_relative_attention_bias` primitive in vokra-ops (T5's \
         relative attention-bias bucketing per Raffel et al. 2020 §2.1 is distinct \
         from DeBERTa's `make_log_bucket_position` used by vokra-bert, and every \
         Transformer model in the tree — whisper / canary / voxtral — re-implements \
         attention from softmax + GEMM + LayerNorm without a shared T5 primitive), \
         and `mt3/event_codec.py` has not been ported to Rust so the decoder token \
         stream cannot become MidiEvent variants. Config: num_enc_layers={enc}, \
         num_dec_layers={dec}, d_model={d_model}, d_ff={d_ff}, n_heads={n_heads}, \
         d_kv={d_kv}, music_vocab_size={vocab}, rel_attn_num_buckets={buckets}, \
         rel_attn_max_distance={max_dist}. Primary sources: {t5_network} + \
         {event_codec} + {paper}. Loud pending (CLAUDE.md 教訓 (a) — 'loud-partial \
         は fake-complete より honest') — no silent fabricated MidiEvent stream \
         ever emitted (FR-EX-08).",
        enc = cfg.num_enc_layers,
        dec = cfg.num_dec_layers,
        d_model = cfg.d_model,
        d_ff = cfg.d_ff,
        n_heads = cfg.n_heads,
        d_kv = cfg.d_kv,
        vocab = cfg.music_vocab_size,
        buckets = cfg.rel_attn_num_buckets,
        max_dist = cfg.rel_attn_max_distance,
        t5_network = PRIMARY_SOURCE_T5_NETWORK,
        event_codec = PRIMARY_SOURCE_EVENT_CODEC,
        paper = PRIMARY_SOURCE_PAPER,
    ))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    //! Tests for the MT3 runtime binder — round-trip on the topology
    //! chunk group + negative-space round-trip on the loud-partial
    //! gates + MidiEvent surface pin.
    //!
    //! # What "round-trip" means here
    //!
    //! The task spec asks for 5+ unit tests. On real PCM this would
    //! be `transcribe(...)` returning real MIDI event streams, but
    //! the T5 relative-attention-bias primitive does not exist in
    //! `vokra-ops` today and the MIDI event codec Rust port has not
    //! been written (see the module doc + [`Mt3::transcribe`]
    //! rustdoc). Fabricating a real-PCM output would violate
    //! CLAUDE.md 教訓 (a) ("loud-partial は fake-complete より
    //! honest").
    //!
    //! The round-trip semantics we *can* honestly test:
    //!
    //! 1. **Config round-trip**: `from_gguf` reads every axis
    //!    stamped by the converter.
    //! 2. **Loud-error negative-space round-trip**: every stated
    //!    blocker (missing arch / wrong arch / missing chunk / empty
    //!    tensor list / unsupported forward surface) fires at its
    //!    documented surface point, in the documented error variant.
    //! 3. **MidiEvent surface pin**: the four variants match the
    //!    task-listed MIDI event kinds MT3 produces.
    use super::*;
    use vokra_core::gguf::{GgmlType, GgufBuilder};

    /// Builds a minimal MT3 GGUF carrying the arch tag + full
    /// `vokra.mt3.*` chunk group + one representative T5 encoder
    /// tensor whose outer dim matches the given `d_model`.
    /// `weight_license_class` is written under
    /// `vokra.provenance.weight_license` (or omitted if `None`).
    fn mt3_gguf(cfg: Mt3Config, weight_license_class: Option<LicenseClass>) -> GgufFile {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_string(chunks::KEY_MODEL_NAME, "mt3-multitrack");
        b.add_u32(GGUF_KEY_D_MODEL, cfg.d_model);
        b.add_u32(GGUF_KEY_D_FF, cfg.d_ff);
        b.add_u32(GGUF_KEY_N_HEADS, cfg.n_heads);
        b.add_u32(GGUF_KEY_D_KV, cfg.d_kv);
        b.add_u32(GGUF_KEY_NUM_ENC_LAYERS, cfg.num_enc_layers);
        b.add_u32(GGUF_KEY_NUM_DEC_LAYERS, cfg.num_dec_layers);
        b.add_u32(GGUF_KEY_MUSIC_VOCAB_SIZE, cfg.music_vocab_size);
        b.add_u32(GGUF_KEY_REL_ATTN_NUM_BUCKETS, cfg.rel_attn_num_buckets);
        b.add_u32(GGUF_KEY_REL_ATTN_MAX_DISTANCE, cfg.rel_attn_max_distance);
        if let Some(cls) = weight_license_class {
            b.add_string(chunks::KEY_PROVENANCE_WEIGHT_LICENSE, cls.as_str());
        }
        // One representative encoder tensor so the non-emptiness
        // gate passes and the shape-consistency accessor has
        // something to walk. The `d_model` dim is deliberately at
        // axis 0 so `matches_config` returns true.
        let d = cfg.d_model as u64;
        b.add_tensor(
            "encoder.layers.0.self_attn.q_proj.weight",
            GgmlType::F32,
            vec![d, d],
            vec![0u8; (d * d * 4) as usize],
        )
        .expect("add_tensor");
        GgufFile::parse(b.to_bytes().expect("serialize")).expect("parse")
    }

    // -----------------------------------------------------------------------
    // 1. Mt3Config default matches T5-small
    // -----------------------------------------------------------------------

    #[test]
    fn mt3_config_default_matches_t5_small() {
        // Pin the T5-small hparams transcribed from magenta/mt3
        // network.py. A rename or axis-value change would land
        // here in the same commit or fail this test.
        let cfg = Mt3Config::t5_small_default();
        assert_eq!(cfg.d_model, 512);
        assert_eq!(cfg.d_ff, 1024);
        assert_eq!(cfg.n_heads, 6);
        assert_eq!(cfg.d_kv, 64);
        assert_eq!(cfg.num_enc_layers, 12);
        assert_eq!(cfg.num_dec_layers, 12);
        assert_eq!(cfg.music_vocab_size, 1200);
        assert_eq!(cfg.rel_attn_num_buckets, 32);
        assert_eq!(cfg.rel_attn_max_distance, 128);
        // T5 quirk: d_kv is NOT d_model / n_heads (Raffel et al.
        // 2020 §2.1 — attention head dim is a first-class hparam,
        // not a derived quantity).
        assert_ne!(
            cfg.d_kv,
            cfg.d_model / cfg.n_heads,
            "d_kv=64 must be independent of d_model/n_heads per T5 \
             (Raffel et al. 2020 §2.1) — a check that catches a \
             future 'simplification' that would silently derive d_kv"
        );
    }

    // -----------------------------------------------------------------------
    // 2. from_gguf full topology chunk-group round-trip
    // -----------------------------------------------------------------------

    #[test]
    fn mt3_from_gguf_round_trip_metadata() {
        let cfg = Mt3Config::t5_small_default();
        let file = mt3_gguf(cfg, Some(LicenseClass::Unknown));
        let mt3 = Mt3::from_gguf(&file).expect("valid GGUF must bind");
        // Config round-trip — every axis stamped by the converter
        // reads back into the same Mt3Config value.
        assert_eq!(*mt3.config(), cfg);
        // Weight-license surface (MT3 converter always stamps
        // Unknown per fail-closed policy — the fixture stamps
        // Unknown to match production).
        assert_eq!(mt3.weight_license(), LicenseClass::Unknown);
        assert!(mt3.tensor_count() >= 1);
    }

    // -----------------------------------------------------------------------
    // 3. from_gguf rejects wrong arch (never silently mis-routes)
    // -----------------------------------------------------------------------

    #[test]
    fn mt3_from_gguf_rejects_wrong_arch() {
        // A `basic-pitch` GGUF handed to the MT3 binder by mistake
        // must fail loud with a specific message rather than
        // silently mis-binding (FR-EX-08).
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, "basic-pitch");
        b.add_u32(GGUF_KEY_D_MODEL, 512);
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let Err(err) = Mt3::from_gguf(&file) else {
            panic!("expected ModelLoad on wrong arch");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains("`basic-pitch`") && m.contains("`mt3`"),
                    "message must name both the got and expected arch tags, got `{m}`"
                );
                assert!(
                    m.contains("polyphonic-CNN"),
                    "message should disambiguate basic-pitch's topology to help \
                     the reader, got `{m}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // 4. transcribe returns UnsupportedOp with primary-source anchors
    // -----------------------------------------------------------------------

    #[test]
    fn mt3_transcribe_loud_partial_returns_unsupported_op() {
        let cfg = Mt3Config::t5_small_default();
        let file = mt3_gguf(cfg, Some(LicenseClass::Unknown));
        let mt3 = Mt3::from_gguf(&file).unwrap();
        // 1 second of 16 kHz mono silence — legitimate input shape,
        // so the loud-partial gate fires (not some pre-transcribe
        // validation).
        let pcm = vec![0.0f32; 16_000];
        let Err(err) = mt3.transcribe(&pcm) else {
            panic!("transcribe must loud-partial");
        };
        match err {
            VokraError::UnsupportedOp(m) => {
                assert!(
                    m.contains("mt3 transcribe"),
                    "message must call out the mt3 transcribe surface, got `{m}`"
                );
                assert!(
                    m.contains("t5_relative_attention_bias"),
                    "message must name the missing T5 primitive by exact identifier \
                     so the follow-up wave knows what to add to vokra-ops, got `{m}`"
                );
                assert!(
                    m.contains("event_codec.py"),
                    "message must name the missing event codec port so the \
                     follow-up wave knows the reference to port, got `{m}`"
                );
                // Primary-source URLs must be cited — the task's
                // hint requires the message contain the primary
                // source URL substring.
                assert!(
                    m.contains("github.com/magenta/mt3"),
                    "message must contain the primary source URL substring \
                     (github.com/magenta/mt3), got `{m}`"
                );
                // Every config axis must be echoed so the reader
                // can cross-check what topology the follow-up wave
                // targets.
                assert!(
                    m.contains("num_enc_layers=12"),
                    "num_enc_layers axis missing: {m}"
                );
                assert!(m.contains("d_model=512"), "d_model axis missing: {m}");
                assert!(m.contains("n_heads=6"), "n_heads axis missing: {m}");
                assert!(m.contains("d_kv=64"), "d_kv axis missing: {m}");
                assert!(
                    m.contains("music_vocab_size=1200"),
                    "music_vocab_size axis missing: {m}"
                );
            }
            other => panic!("expected VokraError::UnsupportedOp, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // 5. MidiEvent surface pin — all four variants construct + pattern match
    // -----------------------------------------------------------------------

    #[test]
    fn midi_event_note_on_note_off_smoke() {
        // Surface pin: the four variants named in the task hint
        // (note_on / note_off / program_change / velocity) must all
        // construct and pattern-match at type-check time. A rename
        // or shape change would land here or fail this test.
        let e1 = MidiEvent::NoteOn {
            pitch: 60,
            velocity: 100,
            program: 0,
            tick: 0,
        };
        let e2 = MidiEvent::NoteOff {
            pitch: 60,
            program: 0,
            tick: 100,
        };
        let e3 = MidiEvent::ProgramChange {
            program: 41, // Violin
            tick: 200,
        };
        let e4 = MidiEvent::Velocity {
            velocity: 80,
            tick: 300,
        };

        // Pattern-match every variant — exhaustive match forces a
        // compile-time failure if a variant is renamed or removed.
        for e in [e1, e2, e3, e4] {
            let tick = match e {
                MidiEvent::NoteOn {
                    pitch,
                    velocity,
                    program,
                    tick,
                } => {
                    assert!(pitch <= 127, "MIDI pitch must be in 0-127");
                    assert!(velocity <= 127, "MIDI velocity must be in 0-127");
                    assert!(program <= 127, "MIDI program must be in 0-127");
                    tick
                }
                MidiEvent::NoteOff {
                    pitch,
                    program,
                    tick,
                } => {
                    assert!(pitch <= 127);
                    assert!(program <= 127);
                    tick
                }
                MidiEvent::ProgramChange { program, tick } => {
                    assert!(program <= 127);
                    tick
                }
                MidiEvent::Velocity { velocity, tick } => {
                    assert!(velocity <= 127);
                    tick
                }
            };
            let _ = tick;
        }
    }

    // -----------------------------------------------------------------------
    // 6. Non-default music vocab size round-trips through the chunk group
    // -----------------------------------------------------------------------

    #[test]
    fn mt3_music_vocab_size_stamp_round_trip() {
        // Verify the music vocab size chunk is a first-class stamp
        // (not silently defaulted to the const). A fixture with a
        // deliberately non-default value must round-trip through
        // Mt3Config::from_gguf.
        let mut cfg = Mt3Config::t5_small_default();
        cfg.music_vocab_size = 1536; // arbitrary non-default
        let file = mt3_gguf(cfg, Some(LicenseClass::Unknown));
        let mt3 = Mt3::from_gguf(&file).expect("valid GGUF must bind");
        assert_eq!(mt3.config().music_vocab_size, 1536);
    }

    // -----------------------------------------------------------------------
    // 7. Missing topology chunk fails loud (no primary-source fallback)
    // -----------------------------------------------------------------------

    #[test]
    fn mt3_from_gguf_rejects_missing_topology_chunk() {
        // Correct arch but missing one of the mandatory
        // `vokra.mt3.*` chunks — a partially-stamped GGUF must be
        // caught here, not silently defaulted to a fabricated axis
        // (FR-EX-08 — the upstream checkpoint carries no first-class
        // config.json, so fallback would fabricate).
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_u32(GGUF_KEY_D_MODEL, 512);
        b.add_u32(GGUF_KEY_D_FF, 1024);
        b.add_u32(GGUF_KEY_N_HEADS, 6);
        // deliberately omit d_kv
        b.add_u32(GGUF_KEY_NUM_ENC_LAYERS, 12);
        b.add_u32(GGUF_KEY_NUM_DEC_LAYERS, 12);
        b.add_u32(GGUF_KEY_MUSIC_VOCAB_SIZE, 1200);
        b.add_u32(GGUF_KEY_REL_ATTN_NUM_BUCKETS, 32);
        b.add_u32(GGUF_KEY_REL_ATTN_MAX_DISTANCE, 128);
        b.add_tensor(
            "encoder.layers.0.self_attn.q_proj.weight",
            GgmlType::F32,
            vec![4, 4],
            vec![0u8; 16 * 4],
        )
        .expect("add_tensor");
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let Err(err) = Mt3::from_gguf(&file) else {
            panic!("expected ModelLoad on missing d_kv chunk");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains(GGUF_KEY_D_KV),
                    "message must name the missing d_kv key, got `{m}`"
                );
                assert!(
                    m.contains("config.json"),
                    "message should explain why fallback is refused (no upstream \
                     config.json), got `{m}`"
                );
            }
            other => panic!("expected VokraError::ModelLoad, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // 8. Empty tensor manifest fails loud (never binds all-zero forward)
    // -----------------------------------------------------------------------

    #[test]
    fn mt3_from_gguf_rejects_empty_tensor_list() {
        // Correct arch + full chunk group but zero tensors — the
        // Mt3Weights non-emptiness gate must fire.
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_u32(GGUF_KEY_D_MODEL, 512);
        b.add_u32(GGUF_KEY_D_FF, 1024);
        b.add_u32(GGUF_KEY_N_HEADS, 6);
        b.add_u32(GGUF_KEY_D_KV, 64);
        b.add_u32(GGUF_KEY_NUM_ENC_LAYERS, 12);
        b.add_u32(GGUF_KEY_NUM_DEC_LAYERS, 12);
        b.add_u32(GGUF_KEY_MUSIC_VOCAB_SIZE, 1200);
        b.add_u32(GGUF_KEY_REL_ATTN_NUM_BUCKETS, 32);
        b.add_u32(GGUF_KEY_REL_ATTN_MAX_DISTANCE, 128);
        // NO tensors added.
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let Err(err) = Mt3::from_gguf(&file) else {
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

    // -----------------------------------------------------------------------
    // 9. Structural pin — arch tag is stable and distinct from siblings
    // -----------------------------------------------------------------------

    #[test]
    fn arch_tag_is_stable_and_distinct_from_sibling_music_arches() {
        // Pin the arch string so a rename would land here in the
        // same commit or fail this test. The sibling music-tree
        // arches MUST NOT collide with ours.
        assert_eq!(ARCH, "mt3");
        assert_ne!(
            ARCH, "basic-pitch",
            "mt3 (T5 encoder-decoder multi-track transcription) and basic-pitch \
             (Spotify polyphonic-CNN posteriorgram) are different topologies — \
             sharing arch would mis-route runtime dispatch (FR-EX-08)"
        );
        assert_ne!(
            ARCH, "beat-this",
            "mt3 and beat-this are different tasks (transcription vs \
             beat-tracking) — sharing arch would mis-route (FR-EX-08)"
        );
        assert_ne!(
            ARCH, "musicgen",
            "mt3 (transcription) and musicgen (generation) are opposite \
             directions — sharing arch would mis-route (FR-EX-08)"
        );
    }
}
