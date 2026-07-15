//! CSM dialog-input AEC front (M4-05-T16, gate G1 consumer — FR-OP-60).
//!
//! **AEC 無しの Moshi/CSM は自己エコーで即崩壊** (レビュアー C 指摘 #3):
//! a dialog model that hears its own playback re-encodes its own voice as
//! the user's turn and spirals. The M4-03 `aec` op (SpeexDSP MDF/AUMDF
//! Rust port) with the runtime-level, sample-clock-tagged far-end queue
//! (`vokra_core::stream::aec_ref_queue`) sits in front of every mic
//! sample the CSM context ingests:
//!
//! ```text
//! CSM output PCM ──(playback clock tag)──> AecRefWriter ─┐
//! mic PCM ──> Aec::process(mic, mic_pos, reader) ────────┴─> cleaned PCM
//!          ──> Mimi encode ──> backbone audio frames
//! ```
//!
//! # Echo path is an explicit choice (FR-EX-08)
//!
//! [`EchoPath::AecRequired`] is the **default** for interactive dialog.
//! [`EchoPath::BypassRecordedInput`] exists **only** for recorded-file
//! input where no acoustic echo path can physically exist (e.g. the T20
//! CLI demo fed a WAV); it is opt-in by name and never inferred — a
//! silent no-op AEC would reintroduce the collapse the hard gate G1
//! exists to prevent. Real-acoustics acceptance is the T30 owner check
//! (合成 echo テストでは代替不能).

use vokra_core::stream::{AecRefReader, AecRefWriter, aec_ref_queue};
use vokra_core::{Result, VokraError};
use vokra_ops::aec::{Aec, AecAttrs, AecStatus};

/// How dialog input audio reaches the CSM context (module docs).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EchoPath {
    /// Interactive default: every mic frame passes the M4-03 canceller
    /// against the time-tagged CSM-output reference queue.
    AecRequired,
    /// **Explicit opt-in** for recorded-file input with no acoustic echo
    /// path (CLI file demo). Never inferred; the constructor of the
    /// consuming engine takes it by name.
    BypassRecordedInput,
}

/// The assembled mic front: canceller + far-end reader. The paired
/// [`AecRefWriter`] is handed to the playback side (the CSM streaming
/// loop pushes every emitted PCM chunk with its playback clock).
pub struct AecFront {
    aec: Aec,
    reader: AecRefReader,
}

impl std::fmt::Debug for AecFront {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AecFront")
            .field("frame_size", &self.aec.frame_size())
            .finish()
    }
}

impl AecFront {
    /// Builds the canceller + far-end queue pair. Returns the front and
    /// the **writer half** the playback side owns (SPSC discipline — the
    /// M4-03 queue contract).
    ///
    /// `queue_capacity_samples` bounds how much played audio the far-end
    /// queue retains (older samples are overwritten — the queue's
    /// documented drop rule; a reader that falls behind sees
    /// `Partial`/`Empty` coverage, never stale data).
    ///
    /// # Errors
    ///
    /// Propagates [`Aec::new`] / [`aec_ref_queue`] validation errors.
    pub fn new(attrs: &AecAttrs, queue_capacity_samples: usize) -> Result<(Self, AecRefWriter)> {
        let aec = Aec::new(attrs)?;
        let (writer, reader) = aec_ref_queue(queue_capacity_samples, attrs.sample_rate)?;
        Ok((Self { aec, reader }, writer))
    }

    /// Samples per [`Self::process_mic`] frame.
    #[must_use]
    pub fn frame_size(&self) -> usize {
        self.aec.frame_size()
    }

    /// Cancels one mic frame (`mic.len() == frame_size`) captured at
    /// absolute sample position `mic_pos` (same clock the playback side
    /// stamps on [`AecRefWriter::push`]), writing the cleaned frame into
    /// `out`. The returned [`AecStatus`] makes every degraded mode
    /// visible (FR-EX-08 — `PartialReference` / `PassThrough` / `Reset`
    /// are never silent).
    ///
    /// # Errors
    ///
    /// Propagates [`Aec::process`] frame/rate validation errors.
    pub fn process_mic(&mut self, mic: &[f32], mic_pos: u64, out: &mut [f32]) -> Result<AecStatus> {
        self.aec.process(mic, mic_pos, &mut self.reader, out)
    }

    /// Resets the canceller's adaptive state (turn boundary / device
    /// change). The far-end queue is untouched — its clock keeps running
    /// with playback.
    pub fn reset(&mut self) {
        self.aec.reset();
    }
}

/// Validates a caller-selected echo path against the input kind — the
/// loud guard the engine calls before ingesting audio (FR-EX-08).
///
/// # Errors
///
/// [`VokraError::InvalidArgument`] when `path` is
/// [`EchoPath::AecRequired`] but no [`AecFront`] was wired.
pub fn require_echo_path_wiring(path: EchoPath, front_present: bool) -> Result<()> {
    match path {
        EchoPath::AecRequired if !front_present => Err(VokraError::InvalidArgument(
            "csm dialog: EchoPath::AecRequired but no AecFront is wired — either \
             construct the engine with an AEC front (interactive default) or opt \
             in to EchoPath::BypassRecordedInput for recorded-file input \
             (FR-OP-60: AEC 無しの CSM は自己エコーで即崩壊)"
                .into(),
        )),
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::stream::AecRefWindowStatus;

    fn attrs() -> AecAttrs {
        AecAttrs {
            sample_rate: 16_000,
            frame_size: 128,
            filter_length: 512,
        }
    }

    /// Synthetic far-end signal (what CSM "played").
    fn far_signal(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let t = i as f32;
                (t * 0.09).sin() * 0.6 + (t * 0.023).sin() * 0.25
            })
            .collect()
    }

    #[test]
    fn synthetic_echo_is_attenuated_in_the_erle_direction() {
        // Echo-only mic (known delay + attenuation of the played signal):
        // after adaptation the residual energy must drop vs the raw mic —
        // the ERLE improvement *direction* (the reference-level assert;
        // real-acoustics acceptance = T30 owner).
        let a = attrs();
        let (mut front, mut writer) = AecFront::new(&a, 16_000).unwrap();
        let n = a.frame_size * 60;
        let delay = 32usize;
        let atten = 0.5f32;
        let far = far_signal(n + delay);
        writer.push(&far, 0).unwrap();

        let mut echo_in = 0.0f64;
        let mut echo_out = 0.0f64;
        let mut out = vec![0.0f32; a.frame_size];
        for f in 0..60 {
            let base = f * a.frame_size;
            let mic: Vec<f32> = (0..a.frame_size)
                .map(|i| {
                    let t = base + i;
                    if t >= delay {
                        far[t - delay] * atten
                    } else {
                        0.0
                    }
                })
                .collect();
            front.process_mic(&mic, base as u64, &mut out).unwrap();
            // Measure over the adapted tail only (last 20 frames).
            if f >= 40 {
                echo_in += mic
                    .iter()
                    .map(|v| f64::from(*v) * f64::from(*v))
                    .sum::<f64>();
                echo_out += out
                    .iter()
                    .map(|v| f64::from(*v) * f64::from(*v))
                    .sum::<f64>();
            }
        }
        assert!(
            echo_out < echo_in * 0.5,
            "canceller must attenuate the pure echo (in {echo_in:.4}, out {echo_out:.4})"
        );
    }

    #[test]
    fn time_tag_queue_order_and_drop_rules_are_visible() {
        // Windows ahead of / behind the pushed range must report their
        // coverage honestly (Partial with the missing count / Empty), and
        // pushes must be monotone in the playback clock.
        let (mut writer, mut reader) = aec_ref_queue(1024, 16_000).unwrap();
        writer.push(&[0.5f32; 256], 0).unwrap();
        let mut win = vec![0.0f32; 128];
        assert_eq!(reader.window(64, &mut win), AecRefWindowStatus::Complete);
        // A window past the pushed end is partially covered — the gap is
        // counted, never silently zero-filled without saying so.
        match reader.window(200, &mut win) {
            AecRefWindowStatus::Partial { missing } => assert_eq!(missing, 200 + 128 - 256),
            other => panic!("expected Partial, got {other:?}"),
        }
        // Far future: nothing covers it (and everything older is consumed
        // — the queue's fixed-memory release rule).
        assert_eq!(reader.window(5_000, &mut win), AecRefWindowStatus::Empty);
        // The writer clock is append-only: a later push at position 256
        // provides 256..384. A backward window (200..328) sees the newly
        // pushed 256..328 = 72 samples; the released 200..256 region reads
        // as missing (56) — re-reading a consumed region is *reported*,
        // never silently zero-filled without saying so (FR-EX-08).
        writer.push(&[0.25f32; 128], 256).unwrap();
        match reader.window(200, &mut win) {
            AecRefWindowStatus::Partial { missing } => assert_eq!(missing, 56),
            other => panic!("expected Partial over the released region, got {other:?}"),
        }
        // The writer rejects a non-monotone (overlapping) playback tag.
        assert!(writer.push(&[0.0f32; 8], 100).is_err());
    }

    #[test]
    fn bypass_is_explicit_and_missing_wiring_is_loud() {
        // The guard the engine calls: AecRequired without a front is an
        // error; the bypass is only reachable by naming it.
        assert!(require_echo_path_wiring(EchoPath::AecRequired, true).is_ok());
        assert!(require_echo_path_wiring(EchoPath::BypassRecordedInput, false).is_ok());
        let err = require_echo_path_wiring(EchoPath::AecRequired, false).unwrap_err();
        assert!(err.to_string().contains("BypassRecordedInput"));
    }

    #[test]
    fn frame_size_contract_is_enforced() {
        let a = attrs();
        let (mut front, _writer) = AecFront::new(&a, 4096).unwrap();
        let mut out = vec![0.0f32; a.frame_size];
        assert!(front.process_mic(&[0.0; 64], 0, &mut out).is_err());
    }
}
