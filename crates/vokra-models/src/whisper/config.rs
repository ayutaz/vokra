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
/// Optional (M4-20): flat `[layer0, head0, layer1, head1, ...]` u32 pairs of the
/// **alignment heads** used for cross-attention DTW word timestamps
/// (openai-whisper `model.alignment_heads`, ADR M4-20 §D-3). Model-specific
/// data (must not be fabricated); absent → no word-timestamp support (an
/// explicit FR-EX-08 error at request time, never a silent no-op).
pub(crate) const KEY_ALIGNMENT_HEADS: &str = "vokra.whisper.alignment_heads";

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
    /// Cross-attention **alignment heads** `(layer, head)` for word-level
    /// timestamps (M4-20, optional metadata `vokra.whisper.alignment_heads`).
    /// Empty when the model carries no alignment-head blob — in which case
    /// word-timestamp requests fail explicitly (FR-EX-08), never silently
    /// (ADR M4-20 §D-3).
    pub alignment_heads: Vec<(usize, usize)>,
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
        // Optional: alignment heads for word timestamps (M4-20). A flat u32
        // array of (layer, head) pairs; absent → empty (word timestamps then
        // fail explicitly at request time, FR-EX-08). Each layer index must be
        // in range; an odd-length or out-of-range blob is a load error (not a
        // silent truncation).
        let alignment_heads = opt_pair_array(file, KEY_ALIGNMENT_HEADS, n_text_layer as usize)?;

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
            alignment_heads,
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

/// Reads an OPTIONAL flat u32 array as `(layer, head)` pairs (M4-20 alignment
/// heads). Absent key → empty vec. An odd length or an out-of-range layer index
/// (`>= n_layer`) is a load error — a malformed alignment-head blob is rejected
/// loudly rather than silently truncated (FR-EX-08).
fn opt_pair_array(file: &GgufFile, key: &str, n_layer: usize) -> Result<Vec<(usize, usize)>> {
    let arr = match file.get(key) {
        Some(GgufMetadataValue::Array(a)) => a,
        Some(_) => return Err(bad(format!("metadata key `{key}` is not an array"))),
        None => return Ok(Vec::new()),
    };
    if arr.values.len() % 2 != 0 {
        return Err(bad(format!(
            "`{key}` must be an even-length (layer, head) pair array, got {}",
            arr.values.len()
        )));
    }
    let flat: Vec<usize> = arr
        .values
        .iter()
        .map(|v| {
            v.as_u64()
                .and_then(|n| usize::try_from(n).ok())
                .ok_or_else(|| bad(format!("`{key}` element is not a usize-range integer")))
        })
        .collect::<Result<_>>()?;
    let mut pairs = Vec::with_capacity(flat.len() / 2);
    for chunk in flat.chunks_exact(2) {
        let (layer, head) = (chunk[0], chunk[1]);
        if layer >= n_layer {
            return Err(bad(format!(
                "`{key}` layer index {layer} >= n_text_layer {n_layer}"
            )));
        }
        pairs.push((layer, head));
    }
    Ok(pairs)
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

    fn with_alignment_heads(mut b: GgufBuilder, flat: &[u32]) -> GgufBuilder {
        b.add_metadata(
            KEY_ALIGNMENT_HEADS,
            GgufMetadataValue::Array(GgufArray {
                element_type: GgufValueType::U32,
                values: flat.iter().map(|&v| GgufMetadataValue::U32(v)).collect(),
            }),
        );
        b
    }

    #[test]
    fn alignment_heads_absent_is_empty() {
        // M4-20: the alignment-head blob is optional; absent → empty (word
        // timestamps then fail explicitly at request time, not here).
        let cfg = WhisperConfig::from_gguf(&parse(valid_builder())).unwrap();
        assert!(cfg.alignment_heads.is_empty());
    }

    #[test]
    fn alignment_heads_parse_as_layer_head_pairs() {
        // Flat [3,1, 4,2] → [(3,1), (4,2)]. n_text_layer = 6, both in range.
        let cfg =
            WhisperConfig::from_gguf(&parse(with_alignment_heads(valid_builder(), &[3, 1, 4, 2])))
                .unwrap();
        assert_eq!(cfg.alignment_heads, vec![(3, 1), (4, 2)]);
    }

    #[test]
    fn alignment_heads_odd_length_is_rejected() {
        assert!(matches!(
            WhisperConfig::from_gguf(&parse(with_alignment_heads(valid_builder(), &[3, 1, 4]))),
            Err(VokraError::ModelLoad(_))
        ));
    }

    #[test]
    fn alignment_heads_out_of_range_layer_is_rejected() {
        // n_text_layer = 6 → layer 6 is out of range (valid 0..=5).
        assert!(matches!(
            WhisperConfig::from_gguf(&parse(with_alignment_heads(valid_builder(), &[6, 0]))),
            Err(VokraError::ModelLoad(_))
        ));
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

    /// M2-06-T03: verifies `WhisperConfig::from_gguf` faithfully round-trips
    /// the hparam quintuple for all 5 supported whisper sizes. The runtime
    /// stays fully data-driven — these values are the expected outputs for
    /// synthetic GGUF stubs, sourced from OpenAI's published whisper
    /// `config.json` files (n_mels, n_vocab, n_audio_layer, n_text_layer,
    /// d_model, n_audio_head, n_text_head, ffn_dim). No runtime hardcoding.
    #[test]
    fn reads_all_whisper_size_hparams() {
        // n_head = d_model / 64 (WHISPER_HEAD_DIM invariant across sizes);
        // ffn_dim = 4 * d_model per Whisper architecture.
        struct Row {
            name: &'static str,
            n_audio_layer: u32,
            n_text_layer: u32,
            n_mels: u32,
            n_vocab: u32,
            d_model: u32,
            n_head: u32,
            ffn_dim: u32,
            n_audio_ctx: u32,
            n_text_ctx: u32,
        }
        let rows = [
            Row {
                name: "base",
                n_audio_layer: 6,
                n_text_layer: 6,
                n_mels: 80,
                n_vocab: 51865,
                d_model: 512,
                n_head: 8,
                ffn_dim: 2048,
                n_audio_ctx: 1500,
                n_text_ctx: 448,
            },
            Row {
                name: "small",
                n_audio_layer: 12,
                n_text_layer: 12,
                n_mels: 80,
                n_vocab: 51865,
                d_model: 768,
                n_head: 12,
                ffn_dim: 3072,
                n_audio_ctx: 1500,
                n_text_ctx: 448,
            },
            Row {
                name: "medium",
                n_audio_layer: 24,
                n_text_layer: 24,
                n_mels: 80,
                n_vocab: 51865,
                d_model: 1024,
                n_head: 16,
                ffn_dim: 4096,
                n_audio_ctx: 1500,
                n_text_ctx: 448,
            },
            Row {
                name: "large-v3",
                n_audio_layer: 32,
                n_text_layer: 32,
                n_mels: 128,
                n_vocab: 51866,
                d_model: 1280,
                n_head: 20,
                ffn_dim: 5120,
                n_audio_ctx: 1500,
                n_text_ctx: 448,
            },
            Row {
                name: "turbo",
                n_audio_layer: 32,
                n_text_layer: 4,
                n_mels: 128,
                n_vocab: 51866,
                d_model: 1280,
                n_head: 20,
                ffn_dim: 5120,
                n_audio_ctx: 1500,
                n_text_ctx: 448,
            },
        ];

        for Row {
            name,
            n_audio_layer,
            n_text_layer,
            n_mels,
            n_vocab,
            d_model,
            n_head,
            ffn_dim,
            n_audio_ctx,
            n_text_ctx,
        } in rows
        {
            let mut b = GgufBuilder::new();
            b.add_u32(KEY_N_MELS, n_mels);
            b.add_u32(KEY_N_AUDIO_CTX, n_audio_ctx);
            b.add_u32(KEY_N_AUDIO_STATE, d_model);
            b.add_u32(KEY_N_AUDIO_HEAD, n_head);
            b.add_u32(KEY_N_AUDIO_LAYER, n_audio_layer);
            b.add_u32(KEY_N_TEXT_CTX, n_text_ctx);
            b.add_u32(KEY_N_TEXT_STATE, d_model);
            b.add_u32(KEY_N_TEXT_HEAD, n_head);
            b.add_u32(KEY_N_TEXT_LAYER, n_text_layer);
            b.add_u32(KEY_N_VOCAB, n_vocab);
            b.add_u32(KEY_FFN_DIM, ffn_dim);
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

            let cfg = WhisperConfig::from_gguf(&parse(b))
                .unwrap_or_else(|e| panic!("{name}: from_gguf failed: {e:?}"));

            let expected = WhisperConfig {
                n_mels: n_mels as usize,
                d_model: d_model as usize,
                n_audio_ctx: n_audio_ctx as usize,
                n_audio_head: n_head as usize,
                n_audio_layer: n_audio_layer as usize,
                n_text_ctx: n_text_ctx as usize,
                n_text_head: n_head as usize,
                n_text_layer: n_text_layer as usize,
                n_vocab: n_vocab as usize,
                ffn_dim: ffn_dim as usize,
                eot: 50257,
                decoder_start_ids: vec![50258, 50259, 50359, 50363],
                alignment_heads: Vec::new(),
            };
            assert_eq!(cfg, expected, "{name}: WhisperConfig mismatch");
            assert_eq!(cfg.head_dim(), 64, "{name}: head_dim must equal 64");
        }
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

    #[test]
    fn n_text_state_not_equal_d_model_is_rejected() {
        // The encoder/decoder d_model cross-check (config.rs line 109): a
        // mis-converted model whose text state disagrees with the audio state.
        let mut b = valid_builder();
        b.add_u32(KEY_N_TEXT_STATE, 256); // != n_audio_state (512)
        assert!(matches!(
            WhisperConfig::from_gguf(&parse(b)),
            Err(VokraError::ModelLoad(_))
        ));
    }

    #[test]
    fn empty_decoder_start_ids_is_rejected() {
        let mut b = valid_builder();
        b.add_metadata(
            KEY_DECODER_START_IDS,
            GgufMetadataValue::Array(GgufArray {
                element_type: GgufValueType::U32,
                values: Vec::new(),
            }),
        );
        assert!(matches!(
            WhisperConfig::from_gguf(&parse(b)),
            Err(VokraError::ModelLoad(_))
        ));
    }

    #[test]
    fn non_array_decoder_start_ids_is_rejected() {
        // A scalar overwrites the array: req_u32_array's not-an-array branch.
        let mut b = valid_builder();
        b.add_u32(KEY_DECODER_START_IDS, 50258);
        assert!(matches!(
            WhisperConfig::from_gguf(&parse(b)),
            Err(VokraError::ModelLoad(_))
        ));
    }

    #[test]
    fn out_of_u32_range_decoder_start_element_is_rejected() {
        // A u64 element beyond u32::MAX exercises req_u32_array's bad-element
        // path (config.rs line 181).
        let mut b = valid_builder();
        b.add_metadata(
            KEY_DECODER_START_IDS,
            GgufMetadataValue::Array(GgufArray {
                element_type: GgufValueType::U64,
                values: vec![GgufMetadataValue::U64(u64::from(u32::MAX) + 1)],
            }),
        );
        assert!(matches!(
            WhisperConfig::from_gguf(&parse(b)),
            Err(VokraError::ModelLoad(_))
        ));
    }
}
