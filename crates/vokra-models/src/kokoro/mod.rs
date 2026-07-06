//! Kokoro-82M native TTS (M2-07) — module skeleton.
//!
//! Native re-implementation of the Kokoro-82M inference core (StyleTTS 2 派生
//! iSTFTNet) in the whisper.cpp style: the model definition lives in Rust,
//! only the upstream **checkpoint** (Apache 2.0, converted offline to GGUF by
//! `vokra-convert`) is consumed at runtime. No ONNX runs at runtime
//! (FR-LD-05). G2P (misaki) is out of scope for M2-07; the runtime consumes
//! phoneme ids only (see [`docs/adr/0007-kokoro-native.md`]).
//!
//! # Layout (M2-07)
//!
//! - [`config`] — `vokra.kokoro.*` metadata + shape-cross-checked dims
//!   (T09);
//! - [`weights`] — F32-only [`GgufFile`](vokra_core::gguf::GgufFile) tensor
//!   store, rejecting a non-F32 tensor as a converter bug (T10);
//! - [`nn`] — 1-D dilated / grouped / transposed convolutions, activations
//!   plus a private [`nn::adain`] helper (StyleTTS 2 AdaIN as a composition
//!   of instance-norm + affine, **not** a new first-class op — FR-EX-08
//!   permits composition);
//! - [`text_encoder`] / [`prosody`] / [`decoder`] — component skeletons; the
//!   concrete forward paths land at T12–T17. The iSTFT head uses FR-OP-01
//!   `istft`, **not** the FR-OP-12 `vocos_head` — Kokoro is iSTFTNet 系.
//!
//! # Hot ops (M2-08 alignment)
//!
//! Kokoro dispatches **GEMM only** through the [`Compute`](crate::compute::Compute)
//! seam (every conv routes through [`nn::conv1d`]'s im2col + GEMM); the
//! LeakyReLU / GELU / sigmoid / AdaIN / iSTFT / voicepack lookup glue is
//! model-internal scalar work. Kokoro is **not** a FR-OP-12 `vocos_head`
//! consumer, so it does not opt in to any `vocos_head` FP16-forbidden
//! registry entry in M2-08 (`docs/adr/0007-kokoro-native.md` §Op gap).

mod config;
mod decoder;
mod nn;
mod prosody;
mod text_encoder;
mod weights;

use std::path::Path;

use vokra_core::gguf::GgufFile;
use vokra_core::{
    BackendKind, CompliancePolicy, Result, SynthesisRequest, SynthesizedAudio, TtsEngine,
    VokraError, check_weight_license,
};

use crate::compute::HotOp;

pub use config::KokoroConfig;

use config::Dims;
use decoder::Decoder;
use prosody::ProsodyPredictor;
use text_encoder::TextEncoder;
use weights::TensorStore;

/// The backend hot ops the Kokoro-82M native TTS dispatches: **GEMM only**
/// (same rationale as [`crate::piper_plus`]).
#[allow(dead_code)] // consumed by the T18 e2e wire-up (`Compute::for_backend`)
pub(crate) const KOKORO_HOT_OPS: &[HotOp] = &[HotOp::Gemm];

/// `vokra.model.arch` a Kokoro-82M voice GGUF must carry. Written by
/// `vokra-convert::models::kokoro::ARCH`; kept in sync by that converter.
const EXPECTED_ARCH: &str = "kokoro-82m-istftnet";

/// A loaded Kokoro-82M voice.
///
/// Built from a voice GGUF (produced offline by `vokra-convert`, T07); no ONNX
/// is touched at runtime (FR-LD-05). The iSTFTNet inference core is assembled
/// here from the loaded weights (T09/T10 skeleton, T12–T17 wire-up).
pub struct KokoroTts {
    config: KokoroConfig,
    #[allow(dead_code)] // consumed by the T12–T17 forward path
    dims: Dims,
    #[allow(dead_code)] // consumed by the T12–T17 forward path
    text_encoder: TextEncoder,
    #[allow(dead_code)] // consumed by the T13/T15 wire-up
    prosody: ProsodyPredictor,
    #[allow(dead_code)] // consumed by the T16/T17 wire-up
    decoder: Decoder,
    /// Backend selector (`Copy`; never a live `!Send` backend, same rationale
    /// as [`crate::piper_plus::PiperPlusTts`]).
    #[allow(dead_code)] // consumed by the T18 e2e wire-up
    backend_kind: BackendKind,
}

impl KokoroTts {
    /// Loads a voice from a GGUF file on disk.
    ///
    /// # Errors
    ///
    /// Propagates GGUF parse errors and any metadata / shape mismatch.
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        Self::from_path_with_policy(path, &CompliancePolicy::strict())
    }

    /// Loads a voice from disk under an explicit compliance `policy`.
    pub fn from_path_with_policy(
        path: impl AsRef<Path>,
        policy: &CompliancePolicy,
    ) -> Result<Self> {
        let bytes = std::fs::read(path.as_ref()).map_err(VokraError::Io)?;
        Self::from_gguf_with_policy(&bytes, policy)
    }

    /// Loads a voice from raw GGUF bytes under an explicit compliance
    /// `policy`.
    ///
    /// The `vokra.model.arch` is checked first, so a non-Kokoro (or wrong
    /// architecture) GGUF fails with a clear [`VokraError::ModelLoad`] rather
    /// than a confusing missing-tensor error deep in a component loader.
    /// Then the shared **weight-license gate**
    /// ([`check_weight_license`], FR-CP-03) runs on the container *before*
    /// any weight tensor is bound — a non-commercial / unknown weight license
    /// (`vokra.provenance.*`) without a research flag is refused with
    /// [`VokraError::ResearchLicenseRequired`], not a silent load.
    ///
    /// Kokoro-82M is Apache 2.0 code + weight, so a stock (unlabelled)
    /// Kokoro voice classifies permissive (built-in registry, arch
    /// `kokoro-82m-istftnet`) and passes.
    pub fn from_gguf_with_policy(bytes: &[u8], policy: &CompliancePolicy) -> Result<Self> {
        let file = GgufFile::parse(bytes.to_vec())
            .map_err(|e| VokraError::ModelLoad(format!("kokoro voice GGUF: {e}")))?;
        let store = TensorStore::new(file);
        let arch = store
            .file()
            .get(vokra_core::gguf::chunks::KEY_MODEL_ARCH)
            .and_then(|v| v.as_str());
        if arch != Some(EXPECTED_ARCH) {
            return Err(VokraError::ModelLoad(format!(
                "not a Kokoro voice GGUF: vokra.model.arch = {arch:?}, expected `{EXPECTED_ARCH}`"
            )));
        }
        // Weight-license research-flag gate (FR-CP-03).
        check_weight_license(store.file(), policy)?;
        let config = KokoroConfig::from_gguf(store.file())?;
        let dims = Dims::derive(&store, &config)?;
        let text_encoder = TextEncoder::load(&store, &config)?;
        let prosody = ProsodyPredictor::load(&store, &config)?;
        let decoder = Decoder::load(&store, &config)?;
        // `store` (and its GGUF backing bytes) drops here.
        Ok(Self {
            config,
            dims,
            text_encoder,
            prosody,
            decoder,
            backend_kind: BackendKind::Cpu,
        })
    }

    /// The resolved voice configuration (sample rate, tables, iSTFT sizes, …).
    pub fn config(&self) -> &KokoroConfig {
        &self.config
    }

    /// Selects the backend the synthesis hot path runs on (default
    /// [`BackendKind::Cpu`]; wired at T18).
    #[must_use]
    pub fn with_backend(mut self, backend: BackendKind) -> Self {
        self.backend_kind = backend;
        self
    }

    /// Synthesizes PCM from a phoneme id sequence — the M2-07-T18 low-level
    /// native path, mirroring [`crate::piper_plus::PiperPlusTts::synthesize_phonemes`].
    ///
    /// The pipeline is
    /// `text_encoder → prosody → length_regulate → decoder → PCM`, with the
    /// text-encoder output transposed from `[t, hidden]` row-major to
    /// `[hidden, t]` channel-major (the layout every downstream stage
    /// consumes; the layout mismatch is pinned at the module boundary here,
    /// not silently inside a component).
    ///
    /// # Style resolution
    ///
    /// Exactly one of the two style sources must be present; both absent is a
    /// loud [`VokraError::InvalidArgument`] rather than a silent zero-style
    /// default (FR-EX-08):
    ///
    /// - `style_override = Some(vec)` — the caller-supplied style vector wins;
    ///   `vec.len()` must equal `config.style_dim`.
    /// - `voice = Some(name)` — the name is looked up in the voice table
    ///   ([`KokoroConfig::voice_names`]); an unknown name is a loud
    ///   [`VokraError::InvalidArgument`]. Because the voicepack tensor schema
    ///   is TBD until M2-07-T02 upstream inspection
    ///   (`docs/adr/0007-kokoro-native.md` §Voicepack), a **known** name
    ///   returns [`VokraError::NotImplemented`] — never a silent zero-style
    ///   fallback.
    ///
    /// `style_override` takes precedence over `voice` when both are set.
    ///
    /// # Scales
    ///
    /// - `noise_scale` is reserved for the Kokoro stochastic prosody path
    ///   (M2-07-T13 follow-on); the current deterministic scaffold ignores it
    ///   rather than pretending to apply a noise it has not yet wired.
    /// - `length_scale` scales each per-phoneme duration before frame
    ///   expansion (`w = max(1, ceil(exp(log_dur) · length_scale))`), matching
    ///   the piper convention. `1.0` disables scaling.
    ///
    /// # Errors
    ///
    /// Any component error propagates verbatim (all typed): out-of-range
    /// phoneme id from the text encoder, shape mismatch inside prosody /
    /// decoder, or the two style-resolution errors above.
    pub fn synthesize_phonemes(
        &self,
        phoneme_ids: &[i64],
        voice: Option<&str>,
        style_override: Option<&[f32]>,
        noise_scale: f32,
        length_scale: f32,
    ) -> Result<SynthesizedAudio> {
        // Reserved for the stochastic prosody path (M2-07-T13 follow-on).
        // Consumed here so the parameter is not silently dropped.
        let _ = noise_scale;

        // 1) Resolve the style vector (FR-EX-08: never a silent zero default).
        let style: Vec<f32> = if let Some(s) = style_override {
            if s.len() != self.config.style_dim {
                return Err(VokraError::InvalidArgument(format!(
                    "kokoro TTS: style_override len {} != style_dim ({})",
                    s.len(),
                    self.config.style_dim,
                )));
            }
            s.to_vec()
        } else if let Some(name) = voice {
            let _voice_id = self.config.voice_id(name).ok_or_else(|| {
                VokraError::InvalidArgument(format!(
                    "kokoro TTS: unknown voice `{name}` (voice_names = {:?})",
                    self.config.voice_names,
                ))
            })?;
            return Err(VokraError::NotImplemented(
                "kokoro voicepack style lookup TBD at M2-07-T02 (pass style_override in the meantime)",
            ));
        } else {
            return Err(VokraError::InvalidArgument(
                "kokoro TTS: no style — pass style_override or a voice name".to_owned(),
            ));
        };

        // 2) Text encoder → [t, hidden_dim] row-major, then transpose to
        //    [hidden_dim, T] channel-major (the layout prosody / length
        //    regulation / decoder consume — piper's convention).
        let enc_arr = self.text_encoder.forward(phoneme_ids)?;
        let t_in = enc_arr.rows;
        let hidden = enc_arr.cols;
        if hidden != self.config.hidden_dim {
            return Err(VokraError::InvalidArgument(format!(
                "kokoro TTS: text encoder produced cols {} != config.hidden_dim ({})",
                hidden, self.config.hidden_dim,
            )));
        }
        let mut encoded_ch = vec![0.0f32; hidden * t_in];
        for ti in 0..t_in {
            for c in 0..hidden {
                encoded_ch[c * t_in + ti] = enc_arr.data[ti * hidden + c];
            }
        }

        // 3) Prosody predictor → (log_dur, f0, energy) each [T]. `deterministic
        //    = true`: the stochastic path is deferred and returns
        //    NotImplemented rather than being silently skipped.
        let (log_dur, _f0, _energy) =
            self.prosody
                .forward(&encoded_ch, &style, t_in, /*deterministic=*/ true)?;

        // 4) Length regulation: `w = max(1, ceil(exp(log_dur) · length_scale))`
        //    (piper convention). Values are clamped to `[1, 1024]` per phoneme
        //    to keep a degenerate scaffold-time `log_dur` from allocating an
        //    unbounded frame buffer via a `+inf as usize` saturation.
        let durations: Vec<usize> = log_dur
            .iter()
            .map(|&l| {
                let v = (l.exp() * length_scale).ceil();
                if v.is_finite() && v >= 1.0 {
                    (v as usize).min(1024)
                } else {
                    1
                }
            })
            .collect();
        let (z, t_frames) = nn::length_regulate(&encoded_ch, hidden, t_in, &durations);

        // 5) Decoder → PCM at `config.sample_rate`. The decoder scaffold
        //    checks its own shapes and produces `t_frames · istft_hop` samples
        //    of bounded, finite audio.
        let pcm = self.decoder.forward(&z, t_frames, &style)?;

        Ok(SynthesizedAudio::new(pcm, self.config.sample_rate))
    }
}

impl TtsEngine for KokoroTts {
    /// Text → PCM adapter around [`KokoroTts::synthesize_phonemes`].
    ///
    /// Style resolution (documented / wired here for the future G2P bridge):
    ///
    /// - `request.speaker_embedding = Some(vec)` (matching `style_dim`) is
    ///   used verbatim as the style vector — the parity of the low-level
    ///   `synthesize_phonemes(style_override = …)` path.
    /// - Otherwise the voicepack table is queried (TBD at M2-07-T02 follow-on;
    ///   returns [`VokraError::NotImplemented`]).
    /// - Both absent — the voice has no named voicepack **and** no embedding is
    ///   supplied — is a loud [`VokraError::InvalidArgument`], never a silent
    ///   zero-style default (FR-EX-08).
    ///
    /// The text → phoneme_ids step requires G2P (misaki; eSpeak-NG fallback
    /// GPL-3.0 excluded), which is out of scope for M2-07 (see
    /// `docs/adr/0007-kokoro-native.md` §Design). Until a G2P bridge lands,
    /// callers exercise the native pipeline via
    /// [`KokoroTts::synthesize_phonemes`] directly with phoneme ids from a
    /// separate G2P integration.
    fn synthesize(&self, request: &SynthesisRequest) -> Result<SynthesizedAudio> {
        let has_style = request.speaker_embedding.is_some() || !self.config.voice_names.is_empty();
        if !has_style {
            return Err(VokraError::InvalidArgument(
                "kokoro TTS: no style — pass request.speaker_embedding or use a voice GGUF \
                 with a named voicepack"
                    .to_owned(),
            ));
        }
        // The text is not silently dropped — its consumer is the future G2P
        // bridge. Reference it so the intent is documented in-source.
        let _ = request.text.as_str();
        Err(VokraError::NotImplemented(
            "kokoro TtsEngine::synthesize needs a G2P bridge (out of scope M2-07); \
             use KokoroTts::synthesize_phonemes with phoneme ids from a separate G2P",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::config::{
        KEY_HIDDEN_DIM, KEY_ISTFT_HOP, KEY_ISTFT_N_FFT, KEY_ISTFT_WIN_LENGTH, KEY_N_DECODER_LAYERS,
        KEY_N_TEXT_LAYERS, KEY_NUM_VOICES, KEY_PHONEME_SYMBOLS, KEY_SAMPLE_RATE, KEY_STYLE_DIM,
        KEY_VOICE_NAMES,
    };
    use super::*;
    use vokra_core::gguf::{
        GgmlType, GgufArray, GgufBuilder, GgufFile, GgufMetadataValue, GgufValueType,
    };

    fn str_array(items: &[&str]) -> GgufMetadataValue {
        GgufMetadataValue::Array(GgufArray {
            element_type: GgufValueType::String,
            values: items
                .iter()
                .map(|s| GgufMetadataValue::String((*s).to_owned()))
                .collect(),
        })
    }

    /// A builder carrying all 11 `vokra.kokoro.*` keys with distinct values so
    /// a field-swap regression is caught.
    fn valid_kokoro_builder() -> GgufBuilder {
        let mut b = GgufBuilder::new();
        b.add_u32(KEY_SAMPLE_RATE, 24_000);
        b.add_u32(KEY_STYLE_DIM, 256);
        b.add_u32(KEY_NUM_VOICES, 3);
        b.add_u32(KEY_HIDDEN_DIM, 512);
        b.add_u32(KEY_N_TEXT_LAYERS, 4);
        b.add_u32(KEY_N_DECODER_LAYERS, 6);
        b.add_u32(KEY_ISTFT_N_FFT, 20);
        b.add_u32(KEY_ISTFT_HOP, 5);
        b.add_u32(KEY_ISTFT_WIN_LENGTH, 20);
        b.add_metadata(KEY_PHONEME_SYMBOLS, str_array(&["_", "^", "$", "a"]));
        b.add_metadata(KEY_VOICE_NAMES, str_array(&["af", "am", "bf"]));
        b
    }

    #[test]
    fn config_from_gguf_reads_all_keys() {
        let file =
            GgufFile::parse(valid_kokoro_builder().to_bytes().expect("serialize")).expect("parse");
        let cfg = KokoroConfig::from_gguf(&file).expect("valid config");
        // Every field is verified against the distinct value written above so a
        // field-swap regression is caught.
        assert_eq!(cfg.sample_rate, 24_000);
        assert_eq!(cfg.style_dim, 256);
        assert_eq!(cfg.num_voices, 3);
        assert_eq!(cfg.hidden_dim, 512);
        assert_eq!(cfg.n_text_layers, 4);
        assert_eq!(cfg.n_decoder_layers, 6);
        assert_eq!(cfg.istft_n_fft, 20);
        assert_eq!(cfg.istft_hop, 5);
        assert_eq!(cfg.istft_win_length, 20);
        assert_eq!(cfg.phoneme_symbols, ["_", "^", "$", "a"]);
        assert_eq!(cfg.voice_names, ["af", "am", "bf"]);
        // Voice-id lookup: index in the name table; absent name = None.
        assert_eq!(cfg.voice_id("af"), Some(0));
        assert_eq!(cfg.voice_id("bf"), Some(2));
        assert_eq!(cfg.voice_id("zz"), None);
    }

    #[test]
    fn config_from_gguf_rejects_missing_style_dim() {
        // Every key except `style_dim` — the loader must refuse rather than
        // silently defaulting (FR-EX-08).
        let mut b = valid_kokoro_builder();
        // GgufBuilder does not expose a delete API, so rebuild without the
        // key: reconstruct the builder skipping `style_dim`.
        let mut without_style_dim = GgufBuilder::new();
        without_style_dim.add_u32(KEY_SAMPLE_RATE, 24_000);
        // `style_dim` deliberately omitted.
        without_style_dim.add_u32(KEY_NUM_VOICES, 3);
        without_style_dim.add_u32(KEY_HIDDEN_DIM, 512);
        without_style_dim.add_u32(KEY_N_TEXT_LAYERS, 4);
        without_style_dim.add_u32(KEY_N_DECODER_LAYERS, 6);
        without_style_dim.add_u32(KEY_ISTFT_N_FFT, 20);
        without_style_dim.add_u32(KEY_ISTFT_HOP, 5);
        without_style_dim.add_u32(KEY_ISTFT_WIN_LENGTH, 20);
        without_style_dim.add_metadata(KEY_PHONEME_SYMBOLS, str_array(&["_", "^", "$", "a"]));
        without_style_dim.add_metadata(KEY_VOICE_NAMES, str_array(&["af", "am", "bf"]));
        // `b` is left mutated but unused — silence dead_code.
        let _ = &mut b;

        let file =
            GgufFile::parse(without_style_dim.to_bytes().expect("serialize")).expect("parse");
        match KokoroConfig::from_gguf(&file) {
            Err(VokraError::InvalidArgument(msg)) => {
                assert!(
                    msg.contains(KEY_STYLE_DIM),
                    "error should name the missing key; got: {msg}"
                );
            }
            other => panic!("expected InvalidArgument, got {other:?}"),
        }
    }

    /// The T18 e2e smoke test: an in-memory synthetic GGUF loads through the
    /// weight-license gate, [`KokoroTts::synthesize_phonemes`] orchestrates
    /// `text_encoder → prosody → length_regulate → decoder`, and the returned
    /// PCM satisfies the shape / finiteness / sample-rate assertions the
    /// M2-07-T18 spec calls out. Small synthetic dims (kokoro's converter is
    /// shape-driven so dimensions are not baked into the runtime); all-zero
    /// projections + LN affine (`gamma = 1`, `beta = 0`) keep the numeric path
    /// bounded so the "all finite" assertion holds by construction — the
    /// non-trivial variations of the decoder / prosody scaffolds are covered
    /// in their own unit tests.
    #[test]
    fn synthesize_smoke_produces_expected_shape() {
        let n_vocab: usize = 6;
        let hidden: usize = 16;
        let style_dim: usize = 8;
        let sample_rate: u32 = 24_000;

        // F32 helpers — length-`n` payloads laid out as GGUF-ready LE bytes.
        let zeros = |n: usize| -> Vec<u8> { vec![0u8; n * 4] };
        let ones = |n: usize| -> Vec<u8> { (0..n).flat_map(|_| 1.0f32.to_le_bytes()).collect() };

        let mut b = GgufBuilder::new();
        b.add_string(vokra_core::gguf::chunks::KEY_MODEL_ARCH, EXPECTED_ARCH);
        b.add_u32(KEY_SAMPLE_RATE, sample_rate);
        b.add_u32(KEY_STYLE_DIM, style_dim as u32);
        b.add_u32(KEY_NUM_VOICES, 2);
        b.add_u32(KEY_HIDDEN_DIM, hidden as u32);
        b.add_u32(KEY_N_TEXT_LAYERS, 2);
        b.add_u32(KEY_N_DECODER_LAYERS, 2);
        b.add_u32(KEY_ISTFT_N_FFT, 20);
        b.add_u32(KEY_ISTFT_HOP, 5);
        b.add_u32(KEY_ISTFT_WIN_LENGTH, 20);
        // num_symbols ≥ 5 so `phoneme_ids = [0,1,2,3,4]` stays in range
        // (FR-EX-08: text encoder rejects an id ≥ num_symbols).
        b.add_metadata(
            KEY_PHONEME_SYMBOLS,
            str_array(&["_", "^", "$", "a", "b", "c"]),
        );
        b.add_metadata(KEY_VOICE_NAMES, str_array(&["af", "am"]));

        // Text-encoder tensors (see kokoro/text_encoder.rs::new).
        //  - `norm.weight = 1` / `norm.bias = 0`: non-degenerate LayerNorm
        //    (a `gamma = 0` collapse would still succeed but skip the affine
        //    branch, hiding a scale-path regression);
        //  - all other weights zero so the numeric output is bounded and
        //    finite by construction — plenty for a smoke assertion.
        b.add_tensor(
            "text_encoder.embedding.weight",
            GgmlType::F32,
            vec![n_vocab as u64, hidden as u64],
            zeros(n_vocab * hidden),
        )
        .expect("emb");
        b.add_tensor(
            "text_encoder.norm.weight",
            GgmlType::F32,
            vec![hidden as u64],
            ones(hidden),
        )
        .expect("ln_g");
        b.add_tensor(
            "text_encoder.norm.bias",
            GgmlType::F32,
            vec![hidden as u64],
            zeros(hidden),
        )
        .expect("ln_b");
        b.add_tensor(
            "text_encoder.proj.weight",
            GgmlType::F32,
            vec![hidden as u64, hidden as u64],
            zeros(hidden * hidden),
        )
        .expect("proj_w");
        b.add_tensor(
            "text_encoder.proj.bias",
            GgmlType::F32,
            vec![hidden as u64],
            zeros(hidden),
        )
        .expect("proj_b");

        let bytes = b.to_bytes().expect("serialize");
        let tts =
            KokoroTts::from_gguf_with_policy(&bytes, &CompliancePolicy::strict()).expect("load ok");

        // Explicit `style_override` — the voicepack lookup path is TBD until
        // M2-07-T02 (`docs/adr/0007-kokoro-native.md` §Voicepack), so a
        // `voice` name would return `NotImplemented`. `noise_scale = 0` /
        // `length_scale = 1` = the parity / deterministic setting.
        let style = vec![0.0f32; style_dim];
        let audio = tts
            .synthesize_phonemes(&[0, 1, 2, 3, 4], None, Some(&style), 0.0, 1.0)
            .expect("synth ok");
        assert!(
            !audio.samples.is_empty(),
            "T18 orchestration must produce non-empty PCM"
        );
        assert!(
            audio.samples.iter().all(|s| s.is_finite()),
            "PCM must be all-finite (FR-EX-08)"
        );
        assert_eq!(audio.sample_rate, sample_rate);
    }

    #[test]
    fn tensor_store_rejects_wrong_shape() {
        // A tensor `w` shaped [3]; asking for [2] must fail loudly rather than
        // truncate silently (FR-EX-08).
        let mut b = GgufBuilder::new();
        b.add_string(vokra_core::gguf::chunks::KEY_MODEL_ARCH, EXPECTED_ARCH);
        let bytes: Vec<u8> = [1.0f32, 2.0, 3.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        b.add_tensor("w", GgmlType::F32, vec![3], bytes)
            .expect("add F32 tensor");
        let file = GgufFile::parse(b.to_bytes().expect("serialize")).expect("parse");
        let store = TensorStore::new(file);
        // Correct shape roundtrips.
        assert_eq!(
            store.tensor_shaped("w", &[3]).expect("shape ok"),
            vec![1.0, 2.0, 3.0]
        );
        // Wrong shape is rejected.
        assert!(matches!(
            store.tensor_shaped("w", &[2]),
            Err(VokraError::InvalidArgument(_))
        ));
    }
}
