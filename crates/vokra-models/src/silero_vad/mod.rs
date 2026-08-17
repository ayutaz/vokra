//! Silero VAD v5 as a 1:1-preserved dedicated subgraph (M0-05).
//!
//! # Design red lines (permanent)
//!
//! - **1:1 preservation (FR-LD-06 / FR-OP-50)**: Silero VAD is kept as a
//!   dedicated subgraph, *not* lowered to generic audio-dialect ops, and it is
//!   *not* an audio-dialect op itself. Its internal recurrent state (LSTM
//!   `h`/`c`) and the learned pseudo-STFT are hidden behind the stream handle
//!   (`VadStream`, via [`vokra_core::engines::VadEngine`]).
//! - **No librosa/FFT STFT approximation (NFR-QL-05)**: the pseudo-STFT is a
//!   *learned* `Conv1d(1, 2*bins, k)`, reproduced op-for-op and never lowered to
//!   a standard `stft` op (FR-OP-01) — see [`vokra_vad_micro::pseudo_stft`].
//!
//! # Topology (M5-03, ADR M5-03-iot-tier3-nostd §(a))
//!
//! The numeric **forward core** — GGUF weight binding, the learned pseudo-STFT,
//! the encoder conv stack, the LSTM cell + head — lives in the `#![no_std]`
//! crate [`vokra_vad_micro`], so it cross-compiles for bare-metal Cortex-M55
//! (IoT Tier 3 / NFR-PT-03) without pulling in the std-heavy `vokra-ops` /
//! `vokra-backend-cpu` that `vokra-models` depends on. This module is the **std
//! veneer** over it: the [`VadEngine`] implementation, the streaming handle
//! (`stream`), the WAV reader ([`wav`]) and the file-path [`open`](SileroVadV5::open)
//! constructor. There is therefore ONE forward — the std and no_std builds are
//! bit-identical by construction (M5-03 T08/T11).
//!
//! # Architecture (source: `docs/_research/03-speech-specialized-runtimes.md`
//! §3.1, cross-checked against the upstream `silero_vad.onnx`)
//!
//! ```text
//! input [ctx + frame = 576 @16k / 288 @8k]  (official; raw graph interface = bare 512 / 256)
//!  -> reflect-pad right by n_fft/4
//!  -> Conv1d(1, 2*bins, k=n_fft, stride=n_fft/2)      (learned pseudo-STFT)
//!  -> magnitude = sqrt(real^2 + imag^2)   [bins, 4]   (bins = 129 @16k / 65 @8k; 3 on raw input)
//!  -> encoder: Conv1d(+ReLU) x4, strides 1,2,2,1      [128, 1]
//!  -> LSTM(128,128)  (h/c carried across frames)      [128]
//!  -> ReLU -> Conv1d(128,1,k=1) -> Sigmoid            -> probability
//! ```
//!
//! The graph's time axis is dynamic; **official usage** (the upstream python
//! wrapper, and this module's default stream) prepends a rolling
//! [`SampleRate::context_len`] audio context — the previous frame's tail,
//! zeros at start — to every fixed frame. Feeding bare frames is numerically
//! valid but collapses on real speech (2026-07-16 real-weight eval P1).
//!
//! Silero v5 is really **two** independently-trained networks with this same
//! topology, one per sample rate, chosen in the ONNX by `If(sr == 16000)`. The
//! per-layer GGUF tensor map, the two-branch weight gap, the exact pad/gate
//! findings and the parity methodology are recorded in
//! `crates/vokra-models/src/silero_vad/SPEC.md`.

mod stream;

pub mod wav;

#[cfg(test)]
mod parity;

use std::sync::Arc;

use vokra_core::engines::{VadEngine, VadStreamHandle};
use vokra_core::gguf::{GgufFile, chunks};
use vokra_core::{Result, VokraError};

use vokra_vad_micro::SileroWeights;

/// The `vokra.model.arch` value a Silero VAD GGUF must carry.
///
/// Mirror of `crates/vokra-convert/src/models/silero.rs::ARCH` — the
/// converter owns the writer contract, this module owns the reader
/// contract (the same deliberate two-copies convention
/// [`crate::pyannote`] documents; a compile-time check would need
/// `vokra-convert` in `vokra-models`'s dependency graph, which the
/// workspace pins forbid).
///
/// **One tag covers both upstream releases.** v5 and v6.2.1 are
/// architecturally identical per upstream (same `Conv1d(1, 258, k=256,
/// stride=128)` pseudo-STFT + 4-conv encoder + `LSTMCell(128, 128)` +
/// `Conv1d(128, 1, 1)` head — see `crates/vokra-models/tests/parity_silero_v6.rs`
/// module docs for the primary-source walk), so the release is recorded in
/// the separate `vokra.silero.version` key that [`SileroVadV5::variant`]
/// reads, **not** in the arch tag. Sibling VAD models that are *not* this
/// subgraph (`fsmn-vad`, `firered-vad`, `pyannote-segmentation`) carry
/// their own distinct arch tags and are rejected here.
pub const EXPECTED_ARCH: &str = "silero-vad";

/// The sample rate a Silero v5 model handles, re-exported from the no_std
/// forward core so `vokra_models::silero_vad::SampleRate` keeps resolving for
/// existing consumers (`vokra-cli`, `vokra-capi`, the `vad_demo` example).
pub use vokra_vad_micro::SampleRate;

/// The upstream Silero VAD release tag carried by a loaded GGUF (v5 vs
/// v6.2.1). Re-exported from the no_std forward core so consumers can key
/// telemetry, license attribution, and parity-fixture selection on the
/// artifact's provenance without depending on `vokra-core::gguf::silero`
/// directly.
pub use vokra_vad_micro::SileroVariant;

/// Silero VAD v5 — a 1:1-preserved subgraph model (M0-05).
///
/// Load with [`from_gguf`](Self::from_gguf) / [`open`](Self::open), then obtain
/// a stateful stream through the [`VadEngine`] trait ([`open_stream`]). The
/// model itself is immutable and shareable; all mutable recurrent state lives in
/// the stream handle (FR-LD-06). The numeric weights + forward live in
/// [`vokra_vad_micro::SileroWeights`] (the no_std core).
///
/// [`open_stream`]: VadEngine::open_stream
pub struct SileroVadV5 {
    weights: Arc<SileroWeights>,
}

impl SileroVadV5 {
    /// Binds the model from a parsed GGUF (FR-LD-01).
    ///
    /// Accepts the corrected both-rate GGUF (`sr8k.*` / `sr16k.*`) or the legacy
    /// single-rate one. Returns [`vokra_core::VokraError::ModelLoad`] if no Silero weights
    /// are present or a tensor has the wrong shape/dtype.
    ///
    /// # Arch verification (FR-EX-08)
    ///
    /// `vokra.model.arch` is checked against [`EXPECTED_ARCH`] **before**
    /// any tensor is bound. The Silero binder matches on bare tensor names
    /// (`encoder.*` / `decoder.rnn.*`), which are generic enough that a
    /// foreign checkpoint could bind a partial, meaningless model — this
    /// gate is what keeps that from happening silently.
    ///
    /// # Errors
    ///
    /// - [`vokra_core::VokraError::ModelLoad`] when `vokra.model.arch` is
    ///   absent or is not [`EXPECTED_ARCH`].
    /// - [`vokra_core::VokraError::ModelLoad`] from
    ///   [`SileroWeights::from_gguf`] for a missing / mis-shaped tensor.
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        verify_arch(gguf)?;
        Ok(Self {
            weights: Arc::new(SileroWeights::from_gguf(gguf)?),
        })
    }

    /// Opens and binds the model from a GGUF file on disk.
    ///
    /// (There is no `open` in the no_std core — the Cortex-M55 build has no
    /// filesystem; it loads GGUF from a flash-mapped `&[u8]` via
    /// `GgufFile::from_external` + [`SileroWeights::from_gguf`].)
    pub fn open(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let gguf = GgufFile::open(path)?;
        Self::from_gguf(&gguf)
    }

    /// Returns whether the loaded GGUF carries weights for `rate`.
    pub fn supports(&self, rate: SampleRate) -> bool {
        self.weights.rate(rate).is_some()
    }

    /// The upstream Silero VAD release tag the loaded GGUF was stamped
    /// with. Defaults to [`SileroVariant::V5`] when the GGUF omits the
    /// `vokra.silero.version` metadata key (backward compatibility with
    /// pre-tagging artifacts, including the committed parity fixture).
    /// Both variants share the same forward today — v5 and v6.2.1 have
    /// architecturally identical topology per upstream — so this is a
    /// provenance signal, not a shape divergence knob.
    pub fn variant(&self) -> SileroVariant {
        self.weights.variant()
    }

    /// Runs a single fixed-size frame from a **fresh zero state** and returns its
    /// speech probability (the T07 single-chunk entry point).
    ///
    /// Follows the official interface: a zero rolling context of
    /// [`SampleRate::context_len`] samples is prepended (exactly the first
    /// frame of a fresh [`VadEngine::open_stream`] stream, and of the upstream
    /// python wrapper after `reset_states`). Delegates to
    /// [`SileroWeights::forward_chunk`] — the shared no_std forward.
    ///
    /// `frame` must be exactly [`SampleRate::frame_len`] samples. Errors if the
    /// model lacks weights for `rate` or the frame length is wrong.
    pub fn forward_chunk(&self, rate: SampleRate, frame: &[f32]) -> Result<f32> {
        self.weights.forward_chunk(rate, frame)
    }

    /// Opens a stream over the **raw** 1:1 ONNX frame interface: bare
    /// [`SampleRate::frame_len`] frames, no rolling audio context (only the
    /// LSTM `h`/`c` crosses frames). This is the interface the bare-frame
    /// parity fixtures (`probs_16k.txt` / `probs_8k.txt`) are generated on;
    /// it is **not** how the model is used upstream and it cannot detect real
    /// speech (2026-07-16 eval P1) — test-gated, for parity only, so no
    /// production path can reach the collapsed semantics.
    #[cfg(test)]
    pub(crate) fn open_raw_stream(&self) -> Box<dyn VadStreamHandle + Send> {
        Box::new(stream::VadStream::new_raw(Arc::clone(&self.weights)))
    }
}

impl VadEngine for SileroVadV5 {
    fn open_stream(&self) -> Box<dyn VadStreamHandle + Send> {
        Box::new(stream::VadStream::new(Arc::clone(&self.weights)))
    }
}

/// Rejects a GGUF whose `vokra.model.arch` is absent or is not
/// [`EXPECTED_ARCH`].
///
/// A *loud* validation step (FR-EX-08) — see [`SileroVadV5::from_gguf`].
fn verify_arch(gguf: &GgufFile) -> Result<()> {
    match gguf.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()) {
        Some(a) if a == EXPECTED_ARCH => Ok(()),
        Some(other) => Err(VokraError::ModelLoad(format!(
            "silero-vad: GGUF arch is `{other}`, expected `{EXPECTED_ARCH}` (was this GGUF \
             produced by `vokra-cli convert --model silero-vad`? Sibling VAD arch tags — \
             `fsmn-vad` (FunASR FSMN-VAD, an FSMN memory-block stack, not this 1:1-preserved \
             subgraph), `firered-vad`, `pyannote-segmentation` (PyanNet SincNet + BiLSTM \
             powerset segmentation) — are all voice-activity models but have completely \
             different tensor manifests. Note the v5 / v6.2.1 *release* is NOT carried in \
             this key: both stamp `{EXPECTED_ARCH}` and differ only in \
             `vokra.silero.version` (see SileroVadV5::variant). Binding a foreign \
             checkpoint on bare tensor-name matches would produce a partial, meaningless \
             model (FR-EX-08 — no silent partial load)."
        ))),
        None => Err(VokraError::ModelLoad(format!(
            "silero-vad: GGUF is missing `{}` — this is not a Vokra-native Silero VAD GGUF \
             (was it produced by `vokra-cli convert --model silero-vad`?)",
            chunks::KEY_MODEL_ARCH,
        ))),
    }
}

/// Absolute path to the committed parity fixture GGUF (both rates), for tests.
#[cfg(test)]
pub(crate) fn test_gguf_path() -> std::path::PathBuf {
    parity_dir().join("silero-vad-v5.gguf")
}

/// Absolute path to the `tests/parity/silero_vad` fixture directory.
#[cfg(test)]
pub(crate) fn parity_dir() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/parity/silero_vad")
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::VokraError;
    use vokra_core::gguf::{GgmlType, GgufBuilder};

    #[test]
    fn loads_both_rates_from_fixture() {
        let m = SileroVadV5::open(test_gguf_path()).expect("load fixture gguf");
        assert!(m.supports(SampleRate::Hz8000));
        assert!(m.supports(SampleRate::Hz16000));
    }

    /// The committed fixture predates the `vokra.silero.version` tag, so it
    /// must load and default to [`SileroVariant::V5`] — the backward-compat
    /// contract at the model-wrapper level (mirrors the binder-level test
    /// `from_gguf_defaults_to_v5_when_release_tag_absent` in
    /// `crates/vokra-vad-micro/src/weights.rs` — that `weights` module is
    /// private, so the file is the only referent).
    #[test]
    fn fixture_gguf_reports_v5_variant() {
        let m = SileroVadV5::open(test_gguf_path()).unwrap();
        assert_eq!(m.variant(), SileroVariant::V5);
    }

    #[test]
    fn from_gguf_reports_missing_tensor() {
        // A correctly-stamped but weightless GGUF has no Silero weights ->
        // explicit ModelLoad error. The arch stamp is required so this test
        // keeps exercising the *tensor* gate rather than short-circuiting on
        // the arch gate added above.
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, EXPECTED_ARCH);
        let gguf = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let Err(VokraError::ModelLoad(msg)) = SileroVadV5::from_gguf(&gguf) else {
            panic!("a weightless silero GGUF must fail with ModelLoad");
        };
        assert!(
            !msg.contains(chunks::KEY_MODEL_ARCH),
            "must reach the tensor gate, not stop at the arch gate: {msg}",
        );
    }

    #[test]
    fn from_gguf_rejects_gguf_without_arch_stamp() {
        // FR-EX-08: an unstamped GGUF is not a Vokra-native Silero artifact.
        // Silero binds on bare tensor names (`encoder.*` / `decoder.rnn.*`),
        // so without this gate a foreign checkpoint could bind a partial
        // model and emit confident, meaningless speech probabilities.
        let bytes = GgufBuilder::new().to_bytes().unwrap();
        let gguf = GgufFile::parse(bytes).unwrap();
        let Err(VokraError::ModelLoad(msg)) = SileroVadV5::from_gguf(&gguf) else {
            panic!("an unstamped GGUF must fail with ModelLoad");
        };
        assert!(
            msg.contains(chunks::KEY_MODEL_ARCH),
            "message must name the missing key: {msg}",
        );
        assert!(
            msg.contains(EXPECTED_ARCH),
            "message must name the expected arch: {msg}",
        );
    }

    #[test]
    fn from_gguf_rejects_foreign_arch_naming_expected_and_actual() {
        // `fsmn-vad` is the most plausible mis-route (a sibling first-class
        // VAD alternative). The refusal must name BOTH the expectation and
        // what was actually handed over.
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, "fsmn-vad");
        let gguf = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let Err(VokraError::ModelLoad(msg)) = SileroVadV5::from_gguf(&gguf) else {
            panic!("a foreign-arch GGUF must fail with ModelLoad");
        };
        assert!(
            msg.contains("fsmn-vad"),
            "message must name the actual arch: {msg}",
        );
        assert!(
            msg.contains(EXPECTED_ARCH),
            "message must name the expected arch: {msg}",
        );
    }

    #[test]
    fn forward_chunk_rejects_wrong_frame_len() {
        let m = SileroVadV5::open(test_gguf_path()).unwrap();
        assert!(m.forward_chunk(SampleRate::Hz16000, &[0.0; 400]).is_err());
    }

    /// The single-chunk entry point follows the official semantics: it must be
    /// bit-identical to the first frame of a fresh official stream.
    #[test]
    fn forward_chunk_matches_official_stream_first_frame() {
        use vokra_core::engines::VadEngine;

        let model = SileroVadV5::open(test_gguf_path()).unwrap();
        let wav = wav::read_wav_f32(parity_dir().join("test_16k.wav")).unwrap();
        let frame = &wav.samples[..SampleRate::Hz16000.frame_len()];

        let single = model.forward_chunk(SampleRate::Hz16000, frame).unwrap();
        let mut stream = model.open_stream();
        let first = stream.push_pcm(frame, 16_000).unwrap();
        assert_eq!(first, vec![single]);
    }

    /// A legacy single-rate model exercised through the std `SileroVadV5`
    /// wrapper + `VadEngine` stream (the model-level half of the coverage the
    /// binder-level checks moved to `crates/vokra-vad-micro/src/weights.rs`,
    /// a private module with no importable path): a stream over a
    /// rate the model lacks is an explicit error, and the present rate works.
    #[test]
    fn legacy_single_rate_stream_rejects_absent_rate() {
        // Minimal legacy (bare-name) 8 kHz GGUF: kernel 128 -> 8 kHz only.
        // The arch stamp is required by `from_gguf` (FR-EX-08); "legacy"
        // here refers to the bare tensor-name layout, not to an unstamped
        // artifact.
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, EXPECTED_ARCH);
        let add = |b: &mut GgufBuilder, name: &str, dims: &[u64]| {
            let n: u64 = dims.iter().product();
            b.add_tensor(
                name,
                GgmlType::F32,
                dims.to_vec(),
                vec![0u8; n as usize * 4],
            )
            .expect("add tensor");
        };
        add(&mut b, "stft.forward_basis_buffer", &[130, 1, 128]);
        for (name, dims) in [
            ("encoder.0.reparam_conv.weight", &[128, 65, 3][..]),
            ("encoder.0.reparam_conv.bias", &[128][..]),
            ("encoder.1.reparam_conv.weight", &[64, 128, 3][..]),
            ("encoder.1.reparam_conv.bias", &[64][..]),
            ("encoder.2.reparam_conv.weight", &[64, 64, 3][..]),
            ("encoder.2.reparam_conv.bias", &[64][..]),
            ("encoder.3.reparam_conv.weight", &[128, 64, 3][..]),
            ("encoder.3.reparam_conv.bias", &[128][..]),
            ("decoder.rnn.weight_ih", &[512, 128][..]),
            ("decoder.rnn.weight_hh", &[512, 128][..]),
            ("decoder.rnn.bias_ih", &[512][..]),
            ("decoder.rnn.bias_hh", &[512][..]),
            ("decoder.decoder.2.weight", &[1, 128, 1][..]),
            ("decoder.decoder.2.bias", &[1][..]),
        ] {
            add(&mut b, name, dims);
        }
        let gguf = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let m = SileroVadV5::from_gguf(&gguf).expect("legacy 8 kHz model loads");

        assert!(m.supports(SampleRate::Hz8000));
        assert!(!m.supports(SampleRate::Hz16000));

        // A stream over the absent 16 kHz rate is an explicit error (FR-EX-08).
        let mut s16 = m.open_stream();
        assert!(s16.push_pcm(&[0.0; 512], 16000).is_err());

        // The present 8 kHz rate yields one probability per 256-sample frame.
        let mut s8 = m.open_stream();
        assert_eq!(s8.push_pcm(&[0.0; 256], 8000).unwrap().len(), 1);
    }
}
