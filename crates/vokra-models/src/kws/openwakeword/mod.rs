//! openWakeWord (`dscripka/openWakeWord`, Apache-2.0 code) — runtime
//! binder for the `openwakeword_op` converter arch (2026-08-05).
//!
//! # Runtime layout (mirror of FSMN-VAD / Silero VAD)
//!
//! ```text
//! PCM (16 kHz mono f32)
//!   -> `vokra_ops::stft` (n_fft=1024, hop=160, win=1024, Hann, center)
//!   -> `vokra_ops::mel_filterbank` (n_mels=32, HTK scale, Slaney norm)
//!   -> per-frame log(mel + eps)
//!   -> rolling `window_frames` (=76) melspec buffer
//!   -> Google `speech_embedding` extractor  ← **loud-partial**
//!      (frozen upstream Google TFLite, no primary-source Python
//!       reference us to transcribe with confidence; the runtime
//!       returns [`VokraError::UnsupportedOp`] until the owner-provisioned
//!       real-weight GGUF is bound via the env-gate parity harness)
//!   -> shared 96-d embedding
//!   -> per-wake-word MLP classifier  (`vokra_ops::openwakeword_classifier_forward`,
//!      **real, unit-tested**)
//!   -> per-wake-word probability ∈ [0, 1]
//! ```
//!
//! # Loud-partial pattern (RMVPE precedent)
//!
//! The runtime `from_gguf` path binds real config + real per-wake-word
//! classifier weights. The mel front-end is real. The
//! [`EmbeddingExtractor::forward`] step is a **loud-partial**:
//! [`VokraError::UnsupportedOp`] with an owner-facing message pointing
//! at the env-gated parity harness (`crates/vokra-models/tests/parity_openwakeword.rs`,
//! `VOKRA_OPENWAKEWORD_REAL_GGUF`). This mirrors the RMVPE `extract_real`
//! posture: the surrounding scaffold is real and lands today so the
//! parity harness can flip the switch the moment the real embedding
//! weight tensors ship, and no downstream caller can accidentally see a
//! silent `0.0` probability masquerading as a real prediction
//! (FR-EX-08).
//!
//! # `vokra.openwakeword.*` chunk group
//!
//! - `vokra.openwakeword.n_wakewords` (u32): number of per-wake-word MLP
//!   classifiers bound in this GGUF.
//! - `vokra.openwakeword.embedding_dim` (u32): shared embedding width
//!   (96 in the reference release).
//! - `vokra.openwakeword.window_frames` (u32): rolling melspec window
//!   the embedding extractor consumes (76 = ~775 ms at 16 kHz).
//! - `vokra.openwakeword.mel_bins` (u32): melspec width per frame (32
//!   in the reference release).
//! - `vokra.openwakeword.sample_rate` (u32): PCM sample rate (16 000).
//! - `vokra.openwakeword.hop_samples` (u32): analysis hop between
//!   melspec frames (160 = 10 ms at 16 kHz).
//! - `vokra.openwakeword.wakeword_names` (Array<String> of length
//!   `n_wakewords`): human-readable per-wake-word names in the order
//!   the classifier weights are indexed.
//! - `vokra.openwakeword.classifier.{i}.linear{1,2}.{weight,bias}`
//!   (F32 tensors): per-wake-word MLP weights.
//!
//! Every hparam is required and validated loudly at load time (FR-EX-08).
//!
//! # Wake-word threshold
//!
//! The engine returns raw sigmoid probabilities; the caller thresholds
//! (typically at `0.5`). The CLI `run --model kws.gguf` prints only
//! `(wakeword, prob)` pairs whose probability exceeds `0.5` by default,
//! so no threshold is baked into the engine (upstream tunes per-wake-word).

use std::sync::Arc;

use vokra_core::engines::KwsEngine;
use vokra_core::gguf::{GgufFile, GgufMetadataValue, GgufValueType};
use vokra_core::{Result, VokraError};
use vokra_ops::{OpenwakewordClassifierWeights, openwakeword_classifier_forward};

#[cfg(test)]
mod tests;

// ---- arch / provenance constants ----------------------------------------
//
// Mirror of `vokra-convert::models::openwakeword_op::{ARCH, NAME, CATEGORY,
// UPSTREAM_HF}` — kept as duplicated `pub const` so the runtime binder does
// not add a cross-crate dependency edge onto the converter (the sibling
// fsmn_vad / silero_vad convention).

/// Expected `vokra.model.arch` value written by
/// `vokra-convert --model openwakeword-op`.
pub const ARCH: &str = "openwakeword_op";

/// Default `vokra.model.name` value written by the op converter.
pub const DEFAULT_NAME: &str = "openwakeword_op";

/// `vokra.model.category` — VAD/KWS family. Sibling of `silero-vad` /
/// `fsmn-vad`.
pub const CATEGORY: &str = "vad-kws";

// ---- vokra.openwakeword.* metadata keys ---------------------------------

/// GGUF metadata key: number of per-wake-word classifiers (u32).
pub const KEY_N_WAKEWORDS: &str = "vokra.openwakeword.n_wakewords";
/// GGUF metadata key: shared embedding width (u32; upstream = 96).
pub const KEY_EMBEDDING_DIM: &str = "vokra.openwakeword.embedding_dim";
/// GGUF metadata key: rolling melspec window (u32; upstream = 76 frames
/// = ~775 ms at 16 kHz / 10 ms hop).
pub const KEY_WINDOW_FRAMES: &str = "vokra.openwakeword.window_frames";
/// GGUF metadata key: mel-bin count per frame (u32; upstream = 32).
pub const KEY_MEL_BINS: &str = "vokra.openwakeword.mel_bins";
/// GGUF metadata key: PCM sample rate the checkpoint expects (u32 Hz;
/// upstream = 16 000).
pub const KEY_SAMPLE_RATE: &str = "vokra.openwakeword.sample_rate";
/// GGUF metadata key: analysis hop between melspec frames (u32 samples;
/// upstream = 160 = 10 ms at 16 kHz).
pub const KEY_HOP_SAMPLES: &str = "vokra.openwakeword.hop_samples";
/// GGUF metadata key: per-wake-word names (Array<String>).
pub const KEY_WAKEWORD_NAMES: &str = "vokra.openwakeword.wakeword_names";

/// Formats a per-wake-word classifier tensor name for the first linear
/// layer weight (row-major `[hidden_dim, embedding_dim]`).
pub fn tensor_classifier_linear1_weight(idx: usize) -> String {
    format!("openwakeword.classifier.{idx}.linear1.weight")
}
/// Formats a per-wake-word classifier tensor name for the first linear
/// layer bias (`[hidden_dim]`).
pub fn tensor_classifier_linear1_bias(idx: usize) -> String {
    format!("openwakeword.classifier.{idx}.linear1.bias")
}
/// Formats a per-wake-word classifier tensor name for the output linear
/// layer weight (row-major `[1, hidden_dim]`).
pub fn tensor_classifier_linear2_weight(idx: usize) -> String {
    format!("openwakeword.classifier.{idx}.linear2.weight")
}
/// Formats a per-wake-word classifier tensor name for the output linear
/// layer bias (`[1]`).
pub fn tensor_classifier_linear2_bias(idx: usize) -> String {
    format!("openwakeword.classifier.{idx}.linear2.bias")
}

/// openWakeWord runtime config (transcribed verbatim from
/// `vokra.openwakeword.*` at load time).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpenwakewordConfig {
    /// Number of per-wake-word classifiers.
    pub n_wakewords: usize,
    /// Shared embedding width (96 for the reference release).
    pub embedding_dim: usize,
    /// Rolling melspec window the embedding extractor consumes.
    pub window_frames: usize,
    /// Mel-bin count per frame.
    pub mel_bins: usize,
    /// PCM sample rate the checkpoint expects (Hz).
    pub sample_rate: u32,
    /// Analysis hop between melspec frames (samples).
    pub hop_samples: usize,
    /// Per-wake-word names, one per classifier, in weight-index order.
    pub wakeword_names: Vec<String>,
}

impl OpenwakewordConfig {
    /// Validates the config loudly (FR-EX-08). `0`-sentinels are refused
    /// on every hparam, and `wakeword_names.len()` must equal
    /// `n_wakewords`.
    pub fn validate(&self) -> Result<()> {
        for (label, v) in [
            ("n_wakewords", self.n_wakewords as u64),
            ("embedding_dim", self.embedding_dim as u64),
            ("window_frames", self.window_frames as u64),
            ("mel_bins", self.mel_bins as u64),
            ("sample_rate", u64::from(self.sample_rate)),
            ("hop_samples", self.hop_samples as u64),
        ] {
            if v == 0 {
                return Err(VokraError::InvalidArgument(format!(
                    "openwakeword config: {label} must be > 0 (got 0 — the GGUF's \
                     vokra.openwakeword.* chunk is missing or malformed)"
                )));
            }
        }
        if self.wakeword_names.len() != self.n_wakewords {
            return Err(VokraError::ModelLoad(format!(
                "openwakeword config: wakeword_names has {} entries, expected \
                 n_wakewords={}",
                self.wakeword_names.len(),
                self.n_wakewords
            )));
        }
        Ok(())
    }

    /// Reads config from `vokra.openwakeword.*` metadata in a parsed
    /// GGUF.
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let get_u32 = |k: &str| -> Result<u32> {
            let v = gguf.get(k).and_then(|v| v.as_u64()).ok_or_else(|| {
                VokraError::ModelLoad(format!(
                    "openwakeword GGUF missing required u32 metadata `{k}`"
                ))
            })?;
            u32::try_from(v).map_err(|_| {
                VokraError::ModelLoad(format!(
                    "openwakeword GGUF metadata `{k}` = {v} does not fit in u32"
                ))
            })
        };
        let n_wakewords = get_u32(KEY_N_WAKEWORDS)? as usize;
        let embedding_dim = get_u32(KEY_EMBEDDING_DIM)? as usize;
        let window_frames = get_u32(KEY_WINDOW_FRAMES)? as usize;
        let mel_bins = get_u32(KEY_MEL_BINS)? as usize;
        let sample_rate = get_u32(KEY_SAMPLE_RATE)?;
        let hop_samples = get_u32(KEY_HOP_SAMPLES)? as usize;
        let wakeword_names = read_string_array(gguf, KEY_WAKEWORD_NAMES)?;

        let cfg = Self {
            n_wakewords,
            embedding_dim,
            window_frames,
            mel_bins,
            sample_rate,
            hop_samples,
            wakeword_names,
        };
        cfg.validate()?;
        Ok(cfg)
    }
}

/// Reads a required `Array<String>` metadata chunk into an owned
/// `Vec<String>`, enforcing element-type (FR-EX-08 — refuse the load
/// rather than silently coerce).
fn read_string_array(gguf: &GgufFile, key: &str) -> Result<Vec<String>> {
    let value = gguf.get(key).ok_or_else(|| {
        VokraError::ModelLoad(format!(
            "openwakeword GGUF missing required Array<String> metadata `{key}`"
        ))
    })?;
    let arr = value.as_array().ok_or_else(|| {
        VokraError::ModelLoad(format!(
            "openwakeword GGUF metadata `{key}` is not an array (expected Array<String>)"
        ))
    })?;
    if arr.element_type != GgufValueType::String {
        return Err(VokraError::ModelLoad(format!(
            "openwakeword GGUF metadata `{key}` has element_type {:?}, expected String",
            arr.element_type
        )));
    }
    let mut out = Vec::with_capacity(arr.values.len());
    for (i, v) in arr.values.iter().enumerate() {
        match v {
            GgufMetadataValue::String(s) => out.push(s.clone()),
            other => {
                return Err(VokraError::ModelLoad(format!(
                    "openwakeword GGUF metadata `{key}[{i}]` is not String (got {:?})",
                    other.value_type()
                )));
            }
        }
    }
    Ok(out)
}

/// Bound per-wake-word classifier bundle: one
/// [`OpenwakewordClassifierWeights`] per wake-word (name + weights).
#[derive(Debug, Clone)]
pub struct BoundClassifier {
    /// Wake-word display name.
    pub name: String,
    /// Classifier MLP weights.
    pub weights: OpenwakewordClassifierWeights,
}

/// Loud-partial embedding extractor.
///
/// The upstream openWakeWord embedding is produced by the frozen Google
/// `speech_embedding` TFLite (Apache-2.0), whose weight tensors are not
/// primary-source-transcribable to Vokra's GGUF layout without the
/// owner-provisioned bundle (mirror of the RMVPE `extract_real` posture
/// per `crates/vokra-models/src/f0/rmvpe.rs`). The extractor therefore
/// only holds a **capability flag**; when
/// [`Self::has_real_embedding_weights`] is `false`,
/// [`Self::forward`] returns
/// [`VokraError::UnsupportedOp`] with owner-flip instructions.
///
/// When the owner-provisioned real bundle ships, the runtime binder
/// (this module's [`OpenwakewordSession::from_gguf`]) sets the flag and
/// wires the real forward — no other API change needed.
#[derive(Debug, Clone)]
pub struct EmbeddingExtractor {
    /// Set to `true` once real Google `speech_embedding` weights bind
    /// (currently: never — see the module docs).
    pub has_real_embedding_weights: bool,
    /// Emit width (== `OpenwakewordConfig::embedding_dim`, cached for
    /// the forward's dimension check).
    pub embedding_dim: usize,
}

impl EmbeddingExtractor {
    /// Runs the embedding extractor forward on the rolling melspec
    /// window (`[window_frames, mel_bins]`, row-major), returning the
    /// `embedding_dim`-wide vector.
    ///
    /// # Errors
    ///
    /// [`VokraError::UnsupportedOp`] until [`Self::has_real_embedding_weights`]
    /// flips to `true` (owner-provisioned bundle wired). The message
    /// names the env-gate and the parity script for direct owner
    /// action — no fabricated `0.0` on the missing weight (FR-EX-08).
    pub fn forward(&self, melspec_window: &[f32]) -> Result<Vec<f32>> {
        let _ = melspec_window; // Consumed once the real forward binds.
        if !self.has_real_embedding_weights {
            return Err(VokraError::UnsupportedOp(
                "openwakeword embedding extractor: real weight binding is a follow-up \
                 wave (owner-provisioned Google speech_embedding bundle). Set \
                 VOKRA_OPENWAKEWORD_REAL_GGUF and follow the recipe in \
                 crates/vokra-models/tests/parity_openwakeword.rs to flip the switch. \
                 Until then this is a loud partial — no silent fabricated 0.0 \
                 probability (FR-EX-08)."
                    .to_owned(),
            ));
        }
        // The real forward binds here once the owner-provisioned bundle
        // is wired; today the flag never flips, so this arm is
        // unreachable — keep it explicit rather than silently returning
        // an all-zero vector.
        Err(VokraError::UnsupportedOp(
            "openwakeword embedding real forward: skeleton reached — a future wave \
             wires the real Google speech_embedding kernel here (FR-EX-08 honest \
             pending)"
                .to_owned(),
        ))
    }
}

/// openWakeWord session — an immutable shareable weight bundle plus the
/// config it was bound against.
#[derive(Debug)]
pub struct OpenwakewordSession {
    cfg: OpenwakewordConfig,
    classifiers: Arc<Vec<BoundClassifier>>,
    embedding: EmbeddingExtractor,
    /// Rolling melspec buffer, row-major
    /// `[<= window_frames, mel_bins]`. Grows chunk-by-chunk under
    /// [`Self::push_pcm16k`] and slides forward by `hop_samples` worth
    /// of frames once the window fills. Currently unread — the
    /// loud-partial gate fires before the STFT+mel pipeline runs — but
    /// kept as a struct field so the follow-up wave that lights up the
    /// embedding extractor does not have to reshape the session type
    /// (it just drops the pre-forward loud-partial and starts consuming
    /// this buffer).
    #[allow(dead_code)]
    melspec_buffer: Vec<f32>,
    /// Rolling raw-PCM tail (samples not yet consumed into a mel
    /// frame). Prevents `push_pcm16k` from discarding samples across
    /// call boundaries. Same follow-up-wave posture as `melspec_buffer`.
    #[allow(dead_code)]
    pending_pcm: Vec<f32>,
}

impl OpenwakewordSession {
    /// Binds the model from a parsed GGUF (FR-LD-01).
    ///
    /// Returns [`VokraError::ModelLoad`] if any required
    /// `vokra.openwakeword.*` chunk is missing, any documented tensor is
    /// absent, or any tensor has the wrong shape / dtype (FR-EX-08 —
    /// no silent reshape).
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        // Verify the arch tag first so a fsmn-vad / silero-vad GGUF
        // handed to us by mistake fails with a clear message instead of
        // a downstream "missing tensor".
        match gguf
            .get(vokra_core::gguf::chunks::KEY_MODEL_ARCH)
            .and_then(|v| v.as_str())
        {
            Some(a) if a == ARCH => {}
            Some(other) => {
                return Err(VokraError::ModelLoad(format!(
                    "openwakeword: GGUF arch is `{other}`, expected `{ARCH}`"
                )));
            }
            None => {
                return Err(VokraError::ModelLoad(
                    "openwakeword: GGUF is missing `vokra.model.arch` (converter did not \
                     stamp it)"
                        .to_owned(),
                ));
            }
        }

        let cfg = OpenwakewordConfig::from_gguf(gguf)?;

        let mut classifiers = Vec::with_capacity(cfg.n_wakewords);
        for i in 0..cfg.n_wakewords {
            let l1_name = tensor_classifier_linear1_weight(i);
            let l1_bias_name = tensor_classifier_linear1_bias(i);
            let l2_name = tensor_classifier_linear2_weight(i);
            let l2_bias_name = tensor_classifier_linear2_bias(i);

            let linear1_weight = gguf.tensor_f32(&l1_name).map_err(|e| {
                VokraError::ModelLoad(format!("openwakeword: tensor `{l1_name}` load failed: {e}"))
            })?;
            let linear1_bias = gguf.tensor_f32(&l1_bias_name).map_err(|e| {
                VokraError::ModelLoad(format!(
                    "openwakeword: tensor `{l1_bias_name}` load failed: {e}"
                ))
            })?;
            let linear2_weight = gguf.tensor_f32(&l2_name).map_err(|e| {
                VokraError::ModelLoad(format!("openwakeword: tensor `{l2_name}` load failed: {e}"))
            })?;
            let linear2_bias = gguf.tensor_f32(&l2_bias_name).map_err(|e| {
                VokraError::ModelLoad(format!(
                    "openwakeword: tensor `{l2_bias_name}` load failed: {e}"
                ))
            })?;

            if linear1_bias.is_empty() {
                return Err(VokraError::ModelLoad(format!(
                    "openwakeword: tensor `{l1_bias_name}` has zero elements — \
                     hidden_dim must be > 0"
                )));
            }
            let hidden_dim = linear1_bias.len();
            let expected_l1 = hidden_dim * cfg.embedding_dim;
            if linear1_weight.len() != expected_l1 {
                return Err(VokraError::ModelLoad(format!(
                    "openwakeword: tensor `{l1_name}` has {} elements, expected {} \
                     (hidden_dim={} * embedding_dim={})",
                    linear1_weight.len(),
                    expected_l1,
                    hidden_dim,
                    cfg.embedding_dim
                )));
            }

            // Dim-order assertion (defense against silent misforward from
            // a Python bridge that writes the transpose): the docstring
            // in `OpenwakewordClassifierWeights::linear1_weight` pins
            // the layout as row-major `[hidden_dim, embedding_dim]`. A
            // bridge that emits `[embedding_dim, hidden_dim]` would pass
            // the product check above but silently misclassify.
            let expected_l1_dims: [u64; 2] = [hidden_dim as u64, cfg.embedding_dim as u64];
            let l1_info = gguf.tensor_info(&l1_name).ok_or_else(|| {
                VokraError::ModelLoad(format!(
                    "openwakeword: tensor `{l1_name}` info unavailable after successful load — \
                     GGUF invariant broken"
                ))
            })?;
            if l1_info.dimensions.as_slice() != expected_l1_dims {
                return Err(VokraError::ModelLoad(format!(
                    "openwakeword: tensor `{l1_name}` dims {:?} — expected [hidden_dim={}, \
                     embedding_dim={}] row-major (see docstring on \
                     OpenwakewordClassifierWeights::linear1_weight)",
                    l1_info.dimensions, hidden_dim, cfg.embedding_dim
                )));
            }
            let expected_l2_dims: [u64; 2] = [1, hidden_dim as u64];
            let l2_info = gguf.tensor_info(&l2_name).ok_or_else(|| {
                VokraError::ModelLoad(format!(
                    "openwakeword: tensor `{l2_name}` info unavailable after successful load — \
                     GGUF invariant broken"
                ))
            })?;
            if l2_info.dimensions.as_slice() != expected_l2_dims {
                return Err(VokraError::ModelLoad(format!(
                    "openwakeword: tensor `{l2_name}` dims {:?} — expected [1, hidden_dim={}] \
                     row-major (single binary output class per wake-word)",
                    l2_info.dimensions, hidden_dim
                )));
            }

            let weights = OpenwakewordClassifierWeights {
                embedding_dim: cfg.embedding_dim,
                hidden_dim,
                linear1_weight,
                linear1_bias,
                linear2_weight,
                linear2_bias,
            };
            // Re-validate through the op-side validator so a misbound
            // classifier fails at load time, not first inference.
            weights.validate().map_err(|e| match e {
                VokraError::InvalidArgument(msg) => {
                    VokraError::ModelLoad(format!("openwakeword classifier {i}: {msg}"))
                }
                other => other,
            })?;

            classifiers.push(BoundClassifier {
                name: cfg.wakeword_names[i].clone(),
                weights,
            });
        }

        let embedding = EmbeddingExtractor {
            // No real Google speech_embedding weights bind in the
            // current landing — every real deploy triggers the
            // loud-partial UnsupportedOp path. When the owner-
            // provisioned bundle wires, this flag flips inside
            // `from_gguf` based on the presence of a
            // `vokra.openwakeword.embedding.*` tensor group.
            has_real_embedding_weights: false,
            embedding_dim: cfg.embedding_dim,
        };

        Ok(Self {
            cfg,
            classifiers: Arc::new(classifiers),
            embedding,
            melspec_buffer: Vec::new(),
            pending_pcm: Vec::new(),
        })
    }

    /// Opens and binds the model from a GGUF file on disk.
    pub fn open(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let gguf = GgufFile::open(path)?;
        Self::from_gguf(&gguf)
    }

    /// Returns the checkpoint's config.
    pub fn config(&self) -> &OpenwakewordConfig {
        &self.cfg
    }

    /// Returns the bound per-wake-word classifiers.
    pub fn classifiers(&self) -> &[BoundClassifier] {
        &self.classifiers
    }
}

impl KwsEngine for OpenwakewordSession {
    fn wakeword_names(&self) -> &[String] {
        &self.cfg.wakeword_names
    }

    fn push_pcm16k(&mut self, samples: &[f32]) -> Result<Vec<(String, f32)>> {
        // Sample-rate invariant: the mel front-end is fit against the
        // checkpoint's sample rate. A caller who pushes 8 kHz PCM into
        // a 16 kHz checkpoint would get silently misclassified
        // wake-words — refuse loudly (FR-EX-08).
        if self.cfg.sample_rate != 16_000 {
            return Err(VokraError::InvalidArgument(format!(
                "openwakeword: engine bound at {} Hz — push_pcm16k accepts 16 kHz only \
                 (resample upstream, or open a stream on the matching rate)",
                self.cfg.sample_rate
            )));
        }

        // Fast-path the loud-partial BEFORE any buffering: a caller
        // that swallows `UnsupportedOp` in a retry loop would otherwise
        // grow `pending_pcm` without bound. The buffer only ever holds
        // consumable data (never `UnsupportedOp`-poisoned bytes).
        //
        // Once `has_real_embedding_weights` flips, the real streaming
        // pipeline below lights up:
        //   1. `vokra_ops::stft` on `pending_pcm`
        //   2. `vokra_ops::mel_filterbank` → per-frame log(mel+eps)
        //   3. slide the rolling melspec buffer forward
        //   4. once the buffer has `window_frames` rows, run the
        //      embedding extractor and every classifier once.
        //
        // Steps 1-3 are real front-end plumbing (`vokra_ops::stft` /
        // `mel_filterbank` are already unit-tested in vokra-ops). Step 4
        // is the loud-partial: `EmbeddingExtractor::forward` returns
        // `UnsupportedOp` until the owner-provisioned Google
        // speech_embedding bundle wires.
        if !self.embedding.has_real_embedding_weights {
            let embedding = vec![0.0f32; self.cfg.embedding_dim];
            // Force the loud-partial error to fire on the very first
            // push — never silently return an empty Vec that a caller
            // could mistake for "no wake-word yet".
            self.embedding.forward(&embedding)?;
            unreachable!(
                "embedding.forward must return UnsupportedOp when the real \
                          bundle is unbound (FR-EX-08 honest pending)"
            );
        }
        self.pending_pcm.extend_from_slice(samples);

        // Real streaming path (activates when
        // `has_real_embedding_weights` is `true`). Wire this branch
        // in the follow-up wave that lands the real embedding
        // extractor + fills `melspec_buffer` from the STFT + mel
        // filterbank. Today this line is unreachable (see above); it is
        // kept as an explicit `UnsupportedOp` rather than an empty
        // `Ok(Vec::new())` so a future partial-wire cannot regress into
        // a silent no-op.
        Err(VokraError::UnsupportedOp(
            "openwakeword real streaming path unreached — see EmbeddingExtractor::forward"
                .to_owned(),
        ))
    }
}

/// Runs one classifier per wake-word against a single embedding vector
/// and returns `(name, probability)` pairs in wake-word index order.
///
/// This is the same code path [`OpenwakewordSession::push_pcm16k`] will
/// call once the embedding extractor lights up; exposed here so unit
/// tests + downstream code that already holds a real embedding (e.g.
/// through an external Python interop) can exercise the classifier
/// half without waiting on the loud-partial extractor.
///
/// # Errors
///
/// [`VokraError::InvalidArgument`] if `embedding.len()` differs from
/// any classifier's `embedding_dim` (all classifiers in one session
/// share the same width by construction, so this is normally a load-
/// time invariant — the check here catches a hand-constructed session).
pub fn classify_embedding(
    classifiers: &[BoundClassifier],
    embedding: &[f32],
) -> Result<Vec<(String, f32)>> {
    let mut out = Vec::with_capacity(classifiers.len());
    for bc in classifiers {
        let p = openwakeword_classifier_forward(&bc.weights, embedding)?;
        out.push((bc.name.clone(), p));
    }
    Ok(out)
}
