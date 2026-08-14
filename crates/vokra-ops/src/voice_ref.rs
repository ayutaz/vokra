//! Unified voice-reference type (Wave 4 2026-08-14 audit follow-up).
//!
//! Vokra's TTS engines each carry their own way to identify which voice a
//! caller wants:
//!
//! - **Kokoro-82M** (`vokra-models::kokoro`) selects a voice by looking up a
//!   name in `KokoroConfig::voice_id(&str) -> Option<usize>` and then
//!   pulling the per-voice reference embedding row from
//!   `VoicePack::ref_s(voice_id, phoneme_count)`. The identifier reaching the
//!   engine is an integer row index.
//! - **piper-plus native** (`vokra-models::piper_plus`) binds a single fixed
//!   voice at GGUF-load time — the voice name is embedded in the GGUF
//!   metadata (`vokra.piper.voice_name`) and no per-request override is
//!   accepted.
//! - **CosyVoice2** (`vokra-models::cosyvoice2`) is a zero-shot voice-clone
//!   TTS: the caller supplies **reference audio** (a waveform + sample rate)
//!   from which the model derives the voice at synthesis time.
//!
//! [`VoiceRef`] is the union of those three shapes, so a downstream API
//! (an HTTP request, a CLI flag, or the future `SynthesisRequest::voice_ref`
//! field) can express "which voice" once without leaking model-specific
//! plumbing. Adapters translate their model's native representation into a
//! [`VoiceRef`] via the [`VoiceRefSource`] trait.
//!
//! # Scope
//!
//! This module ships **API surface only**: the enum, constructors,
//! validation, and the [`VoiceRefSource`] trait. Wiring into the per-engine
//! `TtsEngine` implementations (Kokoro → [`VoiceRef::VoiceId`], piper-plus
//! → [`VoiceRef::FixedVoice`], CosyVoice2 → [`VoiceRef::ReferenceAudio`])
//! and adding an ABI-additive `voice_ref: Option<VoiceRef>` field on the
//! `#[non_exhaustive]` `vokra_core::engines::SynthesisRequest` is a
//! follow-up wave. Landing the type independently unblocks HTTP-level API
//! design (`vokra-server` OpenAI `/v1/audio/speech` extension) in parallel
//! with the per-engine adapter work.
//!
//! # No silent CPU fallback (FR-EX-08)
//!
//! Invalid inputs — an empty PCM buffer, a zero sample rate — raise
//! [`VokraError::InvalidArgument`] at construction time rather than being
//! silently clamped, defaulted, or accepted as a "best-effort" placeholder.
//! The [`VoiceRef::validate`] method re-runs the same checks on a value
//! constructed by variant syntax (`VoiceRef::ReferenceAudio { ... }`)
//! rather than via the [`VoiceRef::reference_audio`] constructor, so a
//! caller that skipped the constructor still has a fail-loud path.
//!
//! # Zero-dependency posture (NFR-DS-02)
//!
//! Only `vokra_core::{Result, VokraError}` and the `std` `Vec` / `String`
//! prelude — no `serde`, no external crate. `Debug` / `Clone` / `PartialEq`
//! are stock `derive` (equivalent to hand-writing them, but auto-checked by
//! the compiler against every variant); no serialisation format is
//! prescribed at this layer. Downstream JSON goes through the existing
//! `vokra_core::json` helpers, which do not require a `serde` derive here.
//! The root `Cargo.lock` continues to list only `vokra-*` packages.

use vokra_core::{Result, VokraError};

/// Unified voice reference for TTS engines.
///
/// See the [module docs](self) for the three shapes this type unifies
/// (Kokoro voice-id / piper-plus fixed / CosyVoice2 reference audio) and the
/// deliberately-deferred wiring into per-engine adapters and
/// `SynthesisRequest`.
///
/// The enum is `#[non_exhaustive]` so a v2.0+ shape (e.g. a URL, a hash-
/// addressed cached embedding, a `Vec<u8>` binary blob) can be added
/// without breaking downstream `match` sites.
///
/// # Example
///
/// ```
/// use vokra_ops::VoiceRef;
///
/// // Kokoro-style: an integer row index.
/// let v = VoiceRef::voice_id(0);
/// assert!(matches!(v, VoiceRef::VoiceId(0)));
///
/// // piper-plus-style: a name string bound at GGUF load time.
/// let v = VoiceRef::fixed_voice("en_US-lessac-medium");
/// assert!(matches!(v, VoiceRef::FixedVoice(_)));
///
/// // CosyVoice2-style: a reference-audio waveform + sample rate.
/// let v = VoiceRef::reference_audio(vec![0.0_f32; 16_000], 16_000)
///     .expect("valid PCM + sample rate");
/// assert!(v.validate().is_ok());
/// ```
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum VoiceRef {
    /// Integer row index into a model-specific voice table. Kokoro-82M's
    /// `VoicePack::ref_s(voice_id, phoneme_count)` is the primary consumer;
    /// `u64` is a superset of Kokoro's `usize` (M1 iMac aarch64: 64 bits,
    /// 32-bit hosts also fit), so a caller building a [`VoiceRef`] on a
    /// 32-bit target can pass any `usize` losslessly.
    VoiceId(u64),
    /// A voice **name** string. The primary consumer is piper-plus native,
    /// which binds a single voice at GGUF-load time (the name is
    /// self-describing metadata; the engine ignores the field for
    /// synthesis routing but downstream logs / attribution / audit trails
    /// use it).
    FixedVoice(String),
    /// A reference-audio waveform used by zero-shot voice-clone TTS
    /// (CosyVoice2). `pcm` is a mono f32 waveform in [-1, 1]; `sample_rate`
    /// is in Hz.
    ///
    /// Construct via [`VoiceRef::reference_audio`] to get up-front
    /// validation of the two fields; the enum variant is left publicly
    /// destructurable for callers that need pattern matching, but such
    /// callers should call [`VoiceRef::validate`] before handing the value
    /// to an engine (FR-EX-08).
    ReferenceAudio {
        /// Mono f32 waveform, expected range `[-1, 1]`. Must be non-empty.
        pcm: Vec<f32>,
        /// Sample rate in Hz. Must be strictly positive.
        sample_rate: u32,
    },
}

impl VoiceRef {
    /// Construct a [`VoiceRef::VoiceId`]. Infallible — every `u64` is a
    /// valid identifier at this layer (the engine decides whether the
    /// specific index exists in its voice table).
    pub fn voice_id(id: u64) -> Self {
        Self::VoiceId(id)
    }

    /// Construct a [`VoiceRef::FixedVoice`]. Infallible — the empty string
    /// is allowed here because piper-plus's fixed-voice binding is opaque
    /// at this layer (the GGUF loader has already resolved the actual
    /// voice; the string is metadata, not a lookup key). Adapters that
    /// need a lookup semantic (i.e. name → row) should reject an empty
    /// name at the adapter boundary.
    pub fn fixed_voice(name: impl Into<String>) -> Self {
        Self::FixedVoice(name.into())
    }

    /// Construct a [`VoiceRef::ReferenceAudio`] with up-front validation.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on any of:
    /// - `pcm` is empty (CosyVoice2 needs at least one sample to derive a
    ///   voice embedding);
    /// - `sample_rate` is zero (a zero sample rate would divide-by-zero in
    ///   downstream resampling / STFT hop conversion).
    ///
    /// Non-finite (NaN / ±∞) samples inside `pcm` are **not** rejected
    /// here — a NaN in a caller-supplied reference waveform is a data-quality
    /// issue the caller must catch, and scanning every sample would make
    /// this constructor O(n). Downstream STFT / resample paths already
    /// have their own numerical guards.
    pub fn reference_audio(pcm: Vec<f32>, sample_rate: u32) -> Result<Self> {
        if pcm.is_empty() {
            return Err(VokraError::InvalidArgument(
                "voice_ref::reference_audio: pcm must not be empty".to_owned(),
            ));
        }
        if sample_rate == 0 {
            return Err(VokraError::InvalidArgument(
                "voice_ref::reference_audio: sample_rate must be > 0 (got 0)".to_owned(),
            ));
        }
        Ok(Self::ReferenceAudio { pcm, sample_rate })
    }

    /// Re-run the constructor invariants on an existing value.
    ///
    /// A caller that built a [`VoiceRef::ReferenceAudio`] via variant
    /// syntax (`VoiceRef::ReferenceAudio { pcm, sample_rate }`) rather
    /// than [`VoiceRef::reference_audio`] can use this to re-validate before
    /// handing the value to an engine. The `VoiceId` and `FixedVoice`
    /// variants are structurally always valid (every `u64` is a valid id;
    /// every `String` including the empty string is an accepted metadata
    /// string per [`VoiceRef::fixed_voice`]), so `validate` is a no-op for
    /// them.
    ///
    /// # Errors
    ///
    /// Same as [`VoiceRef::reference_audio`] when the variant is
    /// [`VoiceRef::ReferenceAudio`].
    pub fn validate(&self) -> Result<()> {
        match self {
            Self::VoiceId(_) | Self::FixedVoice(_) => Ok(()),
            Self::ReferenceAudio { pcm, sample_rate } => {
                if pcm.is_empty() {
                    return Err(VokraError::InvalidArgument(
                        "voice_ref::reference_audio: pcm must not be empty".to_owned(),
                    ));
                }
                if *sample_rate == 0 {
                    return Err(VokraError::InvalidArgument(
                        "voice_ref::reference_audio: sample_rate must be > 0 (got 0)".to_owned(),
                    ));
                }
                Ok(())
            }
        }
    }
}

/// Trait implemented by TTS engines / adapters to expose a caller-visible
/// [`VoiceRef`] derived from their internal representation.
///
/// A typical Kokoro adapter returns `Ok(VoiceRef::voice_id(id as u64))`
/// where `id` is the row it resolved via `KokoroConfig::voice_id(&str)`. A
/// piper-plus adapter returns `Ok(VoiceRef::fixed_voice(name))` reading
/// `vokra.piper.voice_name` from the GGUF metadata. A CosyVoice2 adapter
/// returns `Ok(VoiceRef::reference_audio(pcm, sr)?)` from the reference
/// audio it was configured with.
///
/// # Contract
///
/// - **Total function.** Implementations return `Ok(_)` whenever the
///   adapter is in a synthesis-ready state (e.g. GGUF has finished loading,
///   the reference audio has been supplied). Adapters not yet ready return
///   [`VokraError::InvalidArgument`] with a descriptive message rather than
///   silently returning a placeholder identifier (FR-EX-08).
/// - **Validation.** Implementations should return a value that satisfies
///   [`VoiceRef::validate`] — either construct via the fallible
///   [`VoiceRef::reference_audio`] and propagate the error, or call
///   `.validate()` before returning if the value was built by variant
///   syntax.
pub trait VoiceRefSource {
    /// Return the caller-visible [`VoiceRef`] for this adapter.
    ///
    /// See the trait-level contract for behavioural requirements.
    fn voice_ref(&self) -> Result<VoiceRef>;
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Variant construction ------------------------------------------

    #[test]
    fn voice_id_variant_construction() {
        // Round-trip: the constructor stores the exact u64 the caller
        // passed; no silent widening / narrowing.
        let v = VoiceRef::voice_id(42);
        assert!(matches!(v, VoiceRef::VoiceId(42)));
        assert!(v.validate().is_ok());
    }

    #[test]
    fn fixed_voice_variant_construction() {
        // The Into<String> constructor accepts &str; the value is stored
        // verbatim. PartialEq gives us the round-trip check.
        let v = VoiceRef::fixed_voice("af_heart");
        assert_eq!(v, VoiceRef::FixedVoice("af_heart".to_owned()));
        assert!(v.validate().is_ok());
    }

    #[test]
    fn reference_audio_valid_constructs() {
        // 1 second of silence at 16 kHz is a plausible minimal reference.
        let pcm = vec![0.1_f32; 16_000];
        let sr = 16_000_u32;
        let v = VoiceRef::reference_audio(pcm.clone(), sr).expect("valid");
        match &v {
            VoiceRef::ReferenceAudio {
                pcm: stored_pcm,
                sample_rate: stored_sr,
            } => {
                assert_eq!(stored_pcm.len(), 16_000);
                assert_eq!(*stored_sr, 16_000);
                // Verify the samples were not mutated by the constructor.
                assert_eq!(stored_pcm, &pcm);
            }
            other => panic!("expected ReferenceAudio, got {other:?}"),
        }
        assert!(v.validate().is_ok());
    }

    // ---- Validation (constructor path) ---------------------------------

    #[test]
    fn reference_audio_empty_pcm_rejects() {
        // An empty PCM buffer must be rejected up-front (FR-EX-08).
        let err = VoiceRef::reference_audio(Vec::<f32>::new(), 16_000).unwrap_err();
        match err {
            VokraError::InvalidArgument(msg) => {
                assert!(
                    msg.contains("pcm must not be empty"),
                    "unexpected message: {msg}"
                );
            }
            other => panic!("expected InvalidArgument, got {other:?}"),
        }
    }

    #[test]
    fn reference_audio_zero_sample_rate_rejects() {
        // A zero sample rate would divide-by-zero downstream (STFT hop
        // conversion / resample). Reject up-front.
        let err = VoiceRef::reference_audio(vec![0.0_f32; 100], 0).unwrap_err();
        match err {
            VokraError::InvalidArgument(msg) => {
                assert!(
                    msg.contains("sample_rate must be > 0"),
                    "unexpected message: {msg}"
                );
            }
            other => panic!("expected InvalidArgument, got {other:?}"),
        }
    }

    // ---- Validation (variant-syntax path) ------------------------------

    #[test]
    fn validate_catches_variant_syntax_empty_pcm() {
        // A caller that bypassed the constructor and built the variant
        // directly must still be caught by validate() (defence in depth).
        let v = VoiceRef::ReferenceAudio {
            pcm: Vec::new(),
            sample_rate: 16_000,
        };
        assert!(matches!(v.validate(), Err(VokraError::InvalidArgument(_))));
    }

    #[test]
    fn validate_catches_variant_syntax_zero_sample_rate() {
        let v = VoiceRef::ReferenceAudio {
            pcm: vec![0.5_f32; 10],
            sample_rate: 0,
        };
        assert!(matches!(v.validate(), Err(VokraError::InvalidArgument(_))));
    }

    #[test]
    fn validate_is_noop_for_voice_id_and_fixed_voice() {
        // The two other variants are structurally always valid: every
        // u64 is a legal id; every String (including "") is legal
        // metadata per the fixed_voice constructor rustdoc.
        assert!(VoiceRef::VoiceId(u64::MAX).validate().is_ok());
        assert!(VoiceRef::VoiceId(0).validate().is_ok());
        assert!(VoiceRef::FixedVoice(String::new()).validate().is_ok());
        assert!(
            VoiceRef::FixedVoice("en_US-lessac-medium".to_owned())
                .validate()
                .is_ok()
        );
    }

    // ---- Derive plumbing (Debug / Clone / PartialEq) -------------------

    #[test]
    fn clone_and_partial_eq_round_trip() {
        // Derived Clone + PartialEq: a cloned VoiceRef equals its source
        // across all three variants. Guards against a future variant being
        // added without extending the derive coverage (would be a
        // compile-time error with #[non_exhaustive] + a new field lacking
        // Clone/PartialEq, so this test doubles as a regression sentinel).
        let cases = [
            VoiceRef::voice_id(7),
            VoiceRef::fixed_voice("test"),
            VoiceRef::reference_audio(vec![0.25_f32, -0.5, 0.75], 24_000).expect("valid"),
        ];
        for v in &cases {
            let cloned = v.clone();
            assert_eq!(&cloned, v, "clone must equal source for {v:?}");
        }
    }

    // ---- VoiceRefSource trait plumbing ---------------------------------

    /// Mock adapter mimicking what the future Kokoro adapter will do:
    /// resolve a name → row lookup into a `VoiceId`. Used to prove the
    /// trait wiring works today without needing the per-engine adapters
    /// to have landed.
    struct MockKokoroAdapter {
        voice_id: u64,
    }

    impl VoiceRefSource for MockKokoroAdapter {
        fn voice_ref(&self) -> Result<VoiceRef> {
            Ok(VoiceRef::voice_id(self.voice_id))
        }
    }

    #[test]
    fn voice_ref_source_trait_impl_kokoro_style() {
        let adapter = MockKokoroAdapter { voice_id: 3 };
        let v = adapter.voice_ref().expect("adapter is ready");
        assert!(matches!(v, VoiceRef::VoiceId(3)));
        assert!(v.validate().is_ok());
    }

    /// Mock adapter mimicking a piper-plus binding: return the fixed voice
    /// name read from GGUF metadata.
    struct MockPiperAdapter {
        voice_name: String,
    }

    impl VoiceRefSource for MockPiperAdapter {
        fn voice_ref(&self) -> Result<VoiceRef> {
            Ok(VoiceRef::fixed_voice(self.voice_name.clone()))
        }
    }

    #[test]
    fn voice_ref_source_trait_impl_piper_style() {
        let adapter = MockPiperAdapter {
            voice_name: "en_US-lessac-medium".to_owned(),
        };
        let v = adapter.voice_ref().expect("adapter is ready");
        assert_eq!(v, VoiceRef::FixedVoice("en_US-lessac-medium".to_owned()));
    }

    /// Mock adapter mimicking a CosyVoice2 binding: return the caller-
    /// supplied reference audio, propagating validation errors.
    struct MockCosyAdapter {
        pcm: Vec<f32>,
        sample_rate: u32,
    }

    impl VoiceRefSource for MockCosyAdapter {
        fn voice_ref(&self) -> Result<VoiceRef> {
            VoiceRef::reference_audio(self.pcm.clone(), self.sample_rate)
        }
    }

    #[test]
    fn voice_ref_source_trait_impl_cosy_style_and_error_propagation() {
        // Happy path.
        let good = MockCosyAdapter {
            pcm: vec![0.0_f32; 8_000],
            sample_rate: 16_000,
        };
        let v = good.voice_ref().expect("valid reference");
        assert!(v.validate().is_ok());

        // Error path: an unready adapter (empty PCM) propagates the
        // constructor's error rather than swallowing it and returning a
        // placeholder (FR-EX-08).
        let bad = MockCosyAdapter {
            pcm: Vec::new(),
            sample_rate: 16_000,
        };
        let err = bad.voice_ref().unwrap_err();
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }
}
