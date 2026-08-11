//! Keyword-spotting / wake-word engines (audio-dialect KWS family, FR-OP `kws`).
//!
//! The KWS family is the wake-word / hotword tier of the streaming input
//! path — a shallow classifier over a short (~775 ms) rolling embedding
//! window that fires an event when a trained wake-word is detected. This
//! is intentionally distinct from the VAD family
//! ([`crate::silero_vad`] / [`crate::fsmn_vad`]): VAD emits binary
//! "voice / non-voice", KWS emits per-wake-word `[0, 1]` probabilities
//! and is expected to run **downstream** of a VAD (VAD gates the mic;
//! KWS decides which wake-word fired inside the voiced region).
//!
//! Every member implements
//! [`vokra_core::engines::KwsEngine`]
//! so it can be injected into a `Session` without `vokra-core` knowing
//! any model specifics — mirror of the VAD / ASR / TTS families.
//!
//! # Members
//!
//! - [`openwakeword`] — `dscripka/openWakeWord` (Apache-2.0 code): the
//!   canonical open-source wake-word family. Runtime binder for the
//!   `openwakeword_op` converter arch (2026-08-04). Ships with the mel
//!   front-end + classifier MLP wired for real, and the shared 96-d
//!   embedding extractor as a **loud-partial** follow-up
//!   ([`vokra_core::VokraError::UnsupportedOp`] with owner-flip
//!   instructions per the RMVPE precedent).

pub mod openwakeword;
