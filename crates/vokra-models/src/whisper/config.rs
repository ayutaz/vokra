//! Whisper hyperparameters, read from GGUF `vokra.whisper.*` metadata.
//!
//! Nothing here is hard-coded (FR-LD-02 / FR-MD-02): every field comes from a
//! metadata key the M0-03 converter wrote (derived, in turn, from the upstream
//! checkpoint's tensor shapes — see `vokra-convert/src/models/whisper.rs`). The
//! loader validates the values so a malformed or truncated GGUF fails with an
//! explicit [`VokraError`] rather than mis-shaping a later matmul.
//!
//! # Key contract (M0-06-T04)
//!
//! The `vokra.whisper.*` key strings below are duplicated verbatim in the
//! converter because the two crates cannot depend on each other. Centralising
//! them in `vokra_core::gguf::chunks` is a follow-up (that module is owned by a
//! parallel WP in M0).

use vokra_core::gguf::{GgufFile, GgufMetadataValue};
use vokra_core::{Result, VokraError};

/// `vokra.whisper.n_mels` — number of mel input channels.
pub(crate) const KEY_N_MELS: &str = "vokra.whisper.n_mels";
/// `vokra.whisper.n_audio_ctx` — encoder positional length (1500 for base).
pub(crate) const KEY_N_AUDIO_CTX: &str = "vokra.whisper.n_audio_ctx";
/// `vokra.whisper.n_audio_state` — hidden width `d_model`.
pub(crate) const KEY_N_AUDIO_STATE: &str = "vokra.whisper.n_audio_state";
/// `vokra.whisper.n_audio_head` — encoder attention heads.
pub(crate) const KEY_N_AUDIO_HEAD: &str = "vokra.whisper.n_audio_head";
/// `vokra.whisper.n_audio_layer` — encoder block count.
pub(crate) const KEY_N_AUDIO_LAYER: &str = "vokra.whisper.n_audio_layer";
/// `vokra.whisper.n_text_ctx` — decoder positional length (448 for base).
pub(crate) const KEY_N_TEXT_CTX: &str = "vokra.whisper.n_text_ctx";
/// `vokra.whisper.n_text_state` — decoder hidden width (equals `d_model`).
pub(crate) const KEY_N_TEXT_STATE: &str = "vokra.whisper.n_text_state";
/// `vokra.whisper.n_text_head` — decoder attention heads.
pub(crate) const KEY_N_TEXT_HEAD: &str = "vokra.whisper.n_text_head";
/// `vokra.whisper.n_text_layer` — decoder block count.
pub(crate) const KEY_N_TEXT_LAYER: &str = "vokra.whisper.n_text_layer";
/// `vokra.whisper.n_vocab` — token vocabulary size.
pub(crate) const KEY_N_VOCAB: &str = "vokra.whisper.n_vocab";
/// `vokra.whisper.ffn_dim` — feed-forward inner width.
pub(crate) const KEY_FFN_DIM: &str = "vokra.whisper.ffn_dim";
/// `vokra.whisper.eot` — end-of-transcript token id.
pub(crate) const KEY_EOT: &str = "vokra.whisper.eot";
/// `vokra.whisper.decoder_start_ids` — default decode prefix (`UINT32` array).
pub(crate) const KEY_DECODER_START_IDS: &str = "vokra.whisper.decoder_start_ids";

/// Whisper architectural hyperparameters (all read from GGUF metadata).
///
/// The encoder and decoder share `d_model` in Whisper; the two `n_*_state`
/// metadata keys are still read independently and cross-checked so a
/// mis-converted model is rejected rather than silently mis-shaped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WhisperConfig {
    /// Mel input channels (encoder conv1 in-channels), 80 for base.
    pub n_mels: usize,
    /// Hidden width `d_model` shared by encoder and decoder.
    pub d_model: usize,
    /// Encoder positional length (`n_audio_ctx`), 1500 for base.
    pub n_audio_ctx: usize,
    /// Encoder attention heads.
    pub n_audio_head: usize,
    /// Encoder block count.
    pub n_audio_layer: usize,
    /// Decoder positional length (`n_text_ctx`), 448 for base.
    pub n_text_ctx: usize,
    /// Decoder attention heads.
    pub n_text_head: usize,
    /// Decoder block count.
    pub n_text_layer: usize,
    /// Token vocabulary size.
    pub n_vocab: usize,
    /// Feed-forward inner width.
    pub ffn_dim: usize,
    /// End-of-transcript token id (decode stop condition).
    pub eot: u32,
    /// Default decode prefix (`<|startoftranscript|> <|en|> <|transcribe|>
    /// <|notimestamps|>` for English transcription).
    pub decoder_start_ids: Vec<u32>,
}

impl WhisperConfig {
    /// Per-head width. Whisper fixes this at 64 across model sizes; here it is
    /// simply `d_model / n_audio_head` (validated non-zero and exact at load).
    pub fn head_dim(&self) -> usize {
        self.d_model / self.n_audio_head
    }

    /// Reads and validates the config from a parsed GGUF file.
    ///
    /// # Errors
    ///
    /// [`VokraError::ModelLoad`] if a key is missing, has the wrong type, holds
    /// a zero/degenerate value, or the head count does not divide `d_model`
    /// (which would make attention head-splitting ill-defined).
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        let n_mels = req_u32(file, KEY_N_MELS)?;
        let d_model = req_u32(file, KEY_N_AUDIO_STATE)?;
        let n_text_state = req_u32(file, KEY_N_TEXT_STATE)?;
        let n_audio_ctx = req_u32(file, KEY_N_AUDIO_CTX)?;
        let n_audio_head = req_u32(file, KEY_N_AUDIO_HEAD)?;
        let n_audio_layer = req_u32(file, KEY_N_AUDIO_LAYER)?;
        let n_text_ctx = req_u32(file, KEY_N_TEXT_CTX)?;
        let n_text_head = req_u32(file, KEY_N_TEXT_HEAD)?;
        let n_text_layer = req_u32(file, KEY_N_TEXT_LAYER)?;
        let n_vocab = req_u32(file, KEY_N_VOCAB)?;
        let ffn_dim = req_u32(file, KEY_FFN_DIM)?;
        let eot = req_u32(file, KEY_EOT)?;
        let decoder_start_ids = req_u32_array(file, KEY_DECODER_START_IDS)?;

        if n_text_state != d_model {
            return Err(bad(format!(
                "n_text_state ({n_text_state}) != n_audio_state ({d_model}); \
                 Whisper shares d_model across encoder and decoder"
            )));
        }
        if n_audio_head == 0 || d_model % n_audio_head != 0 {
            return Err(bad(format!(
                "n_audio_head ({n_audio_head}) must divide d_model ({d_model})"
            )));
        }
        if n_text_head == 0 || d_model % n_text_head != 0 {
            return Err(bad(format!(
                "n_text_head ({n_text_head}) must divide d_model ({d_model})"
            )));
        }
        if n_mels == 0
            || n_audio_ctx == 0
            || n_audio_layer == 0
            || n_text_ctx == 0
            || n_text_layer == 0
            || n_vocab == 0
            || ffn_dim == 0
        {
            return Err(bad("a required Whisper hyperparameter was zero".to_owned()));
        }
        if decoder_start_ids.is_empty() {
            return Err(bad("decoder_start_ids must not be empty".to_owned()));
        }

        Ok(Self {
            n_mels: n_mels as usize,
            d_model: d_model as usize,
            n_audio_ctx: n_audio_ctx as usize,
            n_audio_head: n_audio_head as usize,
            n_audio_layer: n_audio_layer as usize,
            n_text_ctx: n_text_ctx as usize,
            n_text_head: n_text_head as usize,
            n_text_layer: n_text_layer as usize,
            n_vocab: n_vocab as usize,
            ffn_dim: ffn_dim as usize,
            eot,
            decoder_start_ids,
        })
    }
}

fn bad(msg: String) -> VokraError {
    VokraError::ModelLoad(format!("whisper config: {msg}"))
}

fn req_u32(file: &GgufFile, key: &str) -> Result<u32> {
    match file.get(key) {
        Some(v) => v
            .as_u64()
            .and_then(|n| u32::try_from(n).ok())
            .ok_or_else(|| bad(format!("metadata key `{key}` is not a u32-range integer"))),
        None => Err(bad(format!("missing metadata key `{key}`"))),
    }
}

fn req_u32_array(file: &GgufFile, key: &str) -> Result<Vec<u32>> {
    let arr = match file.get(key) {
        Some(GgufMetadataValue::Array(a)) => a,
        Some(_) => return Err(bad(format!("metadata key `{key}` is not an array"))),
        None => return Err(bad(format!("missing metadata key `{key}`"))),
    };
    arr.values
        .iter()
        .map(|v| {
            v.as_u64()
                .and_then(|n| u32::try_from(n).ok())
                .ok_or_else(|| bad(format!("`{key}` element is not a u32-range integer")))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::{GgufArray, GgufBuilder, GgufFile, GgufMetadataValue, GgufValueType};

    /// Builds a GGUF carrying a full, valid `vokra.whisper.*` chunk.
    fn valid_builder() -> GgufBuilder {
        let mut b = GgufBuilder::new();
        b.add_u32(KEY_N_MELS, 80);
        b.add_u32(KEY_N_AUDIO_CTX, 1500);
        b.add_u32(KEY_N_AUDIO_STATE, 512);
        b.add_u32(KEY_N_AUDIO_HEAD, 8);
        b.add_u32(KEY_N_AUDIO_LAYER, 6);
        b.add_u32(KEY_N_TEXT_CTX, 448);
        b.add_u32(KEY_N_TEXT_STATE, 512);
        b.add_u32(KEY_N_TEXT_HEAD, 8);
        b.add_u32(KEY_N_TEXT_LAYER, 6);
        b.add_u32(KEY_N_VOCAB, 51865);
        b.add_u32(KEY_FFN_DIM, 2048);
        b.add_u32(KEY_EOT, 50257);
        b.add_metadata(
            KEY_DECODER_START_IDS,
            GgufMetadataValue::Array(GgufArray {
                element_type: GgufValueType::U32,
                values: [50258u32, 50259, 50359, 50363]
                    .iter()
                    .map(|&id| GgufMetadataValue::U32(id))
                    .collect(),
            }),
        );
        b
    }

    fn parse(b: GgufBuilder) -> GgufFile {
        GgufFile::parse(b.to_bytes().unwrap()).unwrap()
    }

    #[test]
    fn reads_whisper_base_hparams() {
        let cfg = WhisperConfig::from_gguf(&parse(valid_builder())).unwrap();
        assert_eq!(cfg.n_mels, 80);
        assert_eq!(cfg.d_model, 512);
        assert_eq!(cfg.n_audio_ctx, 1500);
        assert_eq!(cfg.n_audio_layer, 6);
        assert_eq!(cfg.n_text_layer, 6);
        assert_eq!(cfg.n_audio_head, 8);
        assert_eq!(cfg.head_dim(), 64);
        assert_eq!(cfg.n_vocab, 51865);
        assert_eq!(cfg.ffn_dim, 2048);
        assert_eq!(cfg.eot, 50257);
        assert_eq!(cfg.decoder_start_ids, vec![50258, 50259, 50359, 50363]);
    }

    #[test]
    fn missing_key_is_model_load_error() {
        let mut b = valid_builder();
        // Rebuild without n_vocab by starting fresh minus that key.
        b = {
            let mut nb = GgufBuilder::new();
            for (k, v) in parse(b).metadata() {
                if k != KEY_N_VOCAB {
                    nb.add_metadata(k, v.clone());
                }
            }
            nb
        };
        assert!(matches!(
            WhisperConfig::from_gguf(&parse(b)),
            Err(VokraError::ModelLoad(_))
        ));
    }

    #[test]
    fn head_not_dividing_d_model_is_rejected() {
        let mut b = valid_builder();
        b.add_u32(KEY_N_AUDIO_HEAD, 7); // 512 % 7 != 0
        assert!(matches!(
            WhisperConfig::from_gguf(&parse(b)),
            Err(VokraError::ModelLoad(_))
        ));
    }

    #[test]
    fn zero_hparam_is_rejected() {
        let mut b = valid_builder();
        b.add_u32(KEY_N_AUDIO_LAYER, 0);
        assert!(matches!(
            WhisperConfig::from_gguf(&parse(b)),
            Err(VokraError::ModelLoad(_))
        ));
    }
}
