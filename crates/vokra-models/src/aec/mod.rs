//! Acoustic Echo Cancellation (AEC) family — neural side of the audio
//! dialect §"Speech Enhancement / AGC / AEC" (CLAUDE.md 音声特化
//! オペレータ).
//!
//! # Scope
//!
//! Neural AEC removes the loudspeaker signal (`farend`) leaking into a
//! microphone capture (`mic = near-end + echo`), leaving only the
//! near-end speech. It is orthogonal to the algorithmic
//! [`vokra_ops::aec`] path (M4-03 SpeexDSP / WebRTC AEC3 Rust port,
//! surfaced through the `vokra_aec_*` C ABI) — that path uses
//! adaptive-filter analytic updates and consumes no neural weights.
//! Both live side-by-side; a duplex engine (Moshi / CSM) can choose
//! either through the [`vokra_core::engines::AecEngine`] trait
//! introduced by this family module.
//!
//! # Members
//!
//! - [`nkf_aec`] — **NKF-AEC**
//!   (`fjiang9/NKF-AEC`, MIT repo LICENSE + BSD-3-Clause source file —
//!   Yang et al. "Low-complexity Acoustic Echo Cancellation with
//!   Neural Kalman Filtering", ICASSP 2023, arXiv:2207.11388).
//!   Per-bin adaptive Kalman filter with a shared neural Kalman-gain
//!   network (small `ComplexGRU` + `ComplexDense`); 5.3 KB checkpoint,
//!   real-time causal by construction. Runtime binder for the
//!   `nkf_aec` converter arch (2026-08-05).
//! - [`dtln_aec`] — **DTLN-AEC**
//!   (`breizhn/DTLN-aec`, MIT — Westhausen & Meyer,
//!   "Acoustic Echo Cancellation with the Dual-Signal Transformation
//!   LSTM Network", INTERSPEECH 2021, arXiv:2010.15754). Dual-signal
//!   (STFT-domain LSTM mask over |mic|⊕|farend| + time-domain LSTM
//!   residual) neural AEC; 128 / 256 / 512-unit variants; 16 kHz.
//!   Loud-partial scaffold pending the generic LSTM primitive in
//!   `vokra_ops` (2026-08-14).
//!
//! Members implement [`vokra_core::engines::AecEngine`] and hand out
//! [`vokra_core::engines::AecStreamHandle`] instances (mirror of the
//! `VadEngine` / `VadStreamHandle` shape — AEC is inherently paired
//! streaming with mic + far-end frames aligned sample-for-sample).

pub mod nkf_aec;
// Wave 6 (2026-08-14 post-audit-cc-gap): DTLN-AEC dual-signal LSTM
// neural AEC (`breizhn/DTLN-aec`, MIT — Westhausen & Meyer INTERSPEECH
// 2021 arXiv:2010.15754). Loud-partial pending the generic LSTM
// primitive in `vokra_ops` (LIB.RS RULE — appended at end of members
// list with the Wave 6 comment marker).
pub mod dtln_aec;
