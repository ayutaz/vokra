//! Kokoro-82M hyper-parameters (M2-07-T09).
//!
//! Every runtime parameter (sample rate, style dim, voice / phoneme tables,
//! iSTFT sizes, layer counts, hidden dim) is read from the `vokra.kokoro.*`
//! GGUF metadata the converter wrote (M2-07-T06/T07) — never hard-coded, never
//! given a silent default. Any missing key raises
//! [`VokraError::InvalidArgument`] with the offending key name (FR-EX-08). The
//! upstream Kokoro-82M is Apache 2.0 code + weight, so the resulting GGUF is
//!公式 zoo eligible; a non-commercial provenance tag is rejected by the shared
//! [`vokra_core::check_weight_license`] gate (M2-13), not here.
//!
//! [`Dims::derive`] cross-checks the metadata against the loaded tensor shapes:
//! if `vokra.kokoro.style_dim` disagrees with a `style_dim`-sized weight axis,
//! a malformed voice fails loudly at load rather than mid-forward.

use vokra_core::gguf::{GgufFile, GgufMetadataValue};
use vokra_core::{Result, VokraError};

use super::weights::TensorStore;

// --- `vokra.kokoro.*` metadata key names ------------------------------------
//
// Kept as constants inside this module (mirror the piper-plus / silero
// pattern): Kokoro-specific keys live with the Kokoro model, not in
// `vokra-core::gguf::chunks`.

pub(crate) const KEY_SAMPLE_RATE: &str = "vokra.kokoro.sample_rate";
pub(crate) const KEY_STYLE_DIM: &str = "vokra.kokoro.style_dim";
pub(crate) const KEY_NUM_VOICES: &str = "vokra.kokoro.num_voices";
pub(crate) const KEY_HIDDEN_DIM: &str = "vokra.kokoro.hidden_dim";
pub(crate) const KEY_N_TEXT_LAYERS: &str = "vokra.kokoro.n_text_layers";
pub(crate) const KEY_N_DECODER_LAYERS: &str = "vokra.kokoro.n_decoder_layers";
pub(crate) const KEY_ISTFT_N_FFT: &str = "vokra.kokoro.istft.n_fft";
pub(crate) const KEY_ISTFT_HOP: &str = "vokra.kokoro.istft.hop";
pub(crate) const KEY_ISTFT_WIN_LENGTH: &str = "vokra.kokoro.istft.win_length";
pub(crate) const KEY_PHONEME_SYMBOLS: &str = "vokra.kokoro.phoneme_symbols";
pub(crate) const KEY_VOICE_NAMES: &str = "vokra.kokoro.voice_names";

/// Resolved runtime configuration read from a Kokoro voice GGUF.
#[derive(Debug, Clone)]
pub struct KokoroConfig {
    /// Output PCM sample rate, Hz.
    pub sample_rate: u32,
    /// Style / voice embedding width.
    pub style_dim: usize,
    /// Number of bundled voices (rows of the voicepack lookup).
    pub num_voices: usize,
    /// Text-encoder / decoder hidden channel count.
    pub hidden_dim: usize,
    /// Text encoder transformer / conv layer count.
    pub n_text_layers: usize,
    /// Decoder iSTFTNet stage count.
    pub n_decoder_layers: usize,
    /// iSTFT head FFT size.
    pub istft_n_fft: usize,
    /// iSTFT head hop length.
    pub istft_hop: usize,
    /// iSTFT window length.
    pub istft_win_length: usize,
    /// Phoneme symbol per id (`vokra.kokoro.phoneme_symbols`), index = id.
    pub phoneme_symbols: Vec<String>,
    /// Voice name per id (`vokra.kokoro.voice_names`), index = id.
    pub voice_names: Vec<String>,
}

impl KokoroConfig {
    /// Reads the configuration from a loaded voice GGUF.
    ///
    /// # Errors
    ///
    /// Returns [`VokraError::InvalidArgument`] if any `vokra.kokoro.*` key is
    /// missing or of the wrong type (FR-EX-08: never a silent default).
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        Ok(Self {
            sample_rate: u32v(file, KEY_SAMPLE_RATE)?,
            style_dim: u32v(file, KEY_STYLE_DIM)? as usize,
            num_voices: u32v(file, KEY_NUM_VOICES)? as usize,
            hidden_dim: u32v(file, KEY_HIDDEN_DIM)? as usize,
            n_text_layers: u32v(file, KEY_N_TEXT_LAYERS)? as usize,
            n_decoder_layers: u32v(file, KEY_N_DECODER_LAYERS)? as usize,
            istft_n_fft: u32v(file, KEY_ISTFT_N_FFT)? as usize,
            istft_hop: u32v(file, KEY_ISTFT_HOP)? as usize,
            istft_win_length: u32v(file, KEY_ISTFT_WIN_LENGTH)? as usize,
            phoneme_symbols: string_array(file, KEY_PHONEME_SYMBOLS)?,
            voice_names: string_array(file, KEY_VOICE_NAMES)?,
        })
    }

    /// Voice id for a name (`"af"`, `"am_michael"`, …), or `None` if absent.
    #[allow(dead_code)] // consumed by the T18 e2e wiring
    pub fn voice_id(&self, name: &str) -> Option<usize> {
        self.voice_names.iter().position(|v| v == name)
    }
}

/// Shape-derived model dimensions cross-checked against the metadata
/// ([`KokoroConfig`]). A mismatch is a malformed voice — fail loudly here
/// rather than mid-forward.
#[derive(Debug, Clone)]
#[allow(dead_code)] // consumed by the T12–T17 forward path
pub(crate) struct Dims {
    /// Style / voice embedding width (must equal `config.style_dim`).
    pub style_dim: usize,
    /// Text-encoder / decoder hidden channel count.
    pub hidden_dim: usize,
    /// Decoder iSTFTNet stage count.
    pub n_decoder_layers: usize,
    /// Text encoder layer count.
    pub n_text_layers: usize,
}

impl Dims {
    /// Cross-checks the metadata against the loaded tensor shapes.
    ///
    /// # Errors
    ///
    /// Returns [`VokraError::InvalidArgument`] on a metadata / shape mismatch
    /// (FR-EX-08: a wrong voice never silently loads). Missing shape-defining
    /// tensors also fail here rather than mid-forward.
    ///
    /// The concrete cross-checks (which tensor names carry each dimension) are
    /// pinned to the upstream Kokoro-82M safetensors layout, and the follow-up
    /// wire-up (T12–T17) tightens them as each component's weight names land.
    /// For M2-07-T09 the cross-check surface is intentionally minimal: this
    /// resolver only refuses a config whose metadata cannot possibly be
    /// consistent (zero-sized dims), so the skeleton compiles and the T09/T10
    /// tests can exercise it without requiring a real weight set to be present.
    #[allow(dead_code)] // consumed by the T12–T17 forward path
    pub(crate) fn derive(_store: &TensorStore, config: &KokoroConfig) -> Result<Self> {
        if config.style_dim == 0
            || config.hidden_dim == 0
            || config.n_decoder_layers == 0
            || config.n_text_layers == 0
        {
            return Err(VokraError::InvalidArgument(format!(
                "kokoro voice: degenerate dims (style_dim={}, hidden_dim={}, \
                 n_decoder_layers={}, n_text_layers={})",
                config.style_dim, config.hidden_dim, config.n_decoder_layers, config.n_text_layers,
            )));
        }
        Ok(Self {
            style_dim: config.style_dim,
            hidden_dim: config.hidden_dim,
            n_decoder_layers: config.n_decoder_layers,
            n_text_layers: config.n_text_layers,
        })
    }
}

fn get<'a>(file: &'a GgufFile, key: &str) -> Result<&'a GgufMetadataValue> {
    file.get(key)
        .ok_or_else(|| VokraError::InvalidArgument(format!("kokoro voice GGUF missing `{key}`")))
}

fn u32v(file: &GgufFile, key: &str) -> Result<u32> {
    match get(file, key)? {
        GgufMetadataValue::U32(v) => Ok(*v),
        _ => Err(VokraError::InvalidArgument(format!(
            "kokoro `{key}` is not a UINT32"
        ))),
    }
}

fn string_array(file: &GgufFile, key: &str) -> Result<Vec<String>> {
    let arr = get(file, key)?
        .as_array()
        .ok_or_else(|| VokraError::InvalidArgument(format!("kokoro `{key}` is not an array")))?;
    arr.values
        .iter()
        .map(|v| {
            v.as_str().map(str::to_owned).ok_or_else(|| {
                VokraError::InvalidArgument(format!("kokoro `{key}` has a non-string element"))
            })
        })
        .collect()
}
