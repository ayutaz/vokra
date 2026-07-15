//! `aec` — acoustic echo cancellation (M4-03, FR-OP-60): a from-scratch Rust
//! port of the SpeexDSP echo canceller (`libspeexdsp/mdf.c`, BSD-3-Clause,
//! Copyright (C) 2003-2008 Jean-Marc Valin — attribution in `NOTICE` and
//! `THIRD_PARTY_LICENSES/speexdsp-LICENSE.txt`).
//!
//! # Algorithm (upstream citations)
//!
//! The canceller is the **MDF** (multidelay block frequency domain) adaptive
//! filter — J. S. Soo, K. K. Pang, IEEE Trans. ASSP-38(2), 1990 — in the
//! **AUMDF** alternating-update variant (mdf.c L834-836), with the
//! double-talk-robust variable learning rate of Valin, *On Adjusting the
//! Learning Rate in Frequency Domain Echo Cancellation With Double-Talk*,
//! IEEE TASLP 15(3), 2007 (leak estimate = `Pey/Pyy` regression; no explicit
//! double-talk detector), and the **TWO_PATH** foreground/background filter
//! pair (mdf.c `#define TWO_PATH`): the adaptive background filter is copied
//! to the output-facing foreground filter only when statistically better,
//! and backtracked when significantly worse.
//!
//! The port is the **float build** (`FLOATING_POINT`) of mdf.c at upstream
//! commit `7a158783df74efe7c2d1c6ee8363c1e695c71226` (github.com/xiph/speexdsp,
//! 2025-07-05), mono (`C = K = 1` — the M4-05/M4-06 full-duplex consumers are
//! mono×mono; multichannel is a follow-up). All arithmetic — constants,
//! accumulation grouping, `f64` promotions where C promotes to `double` — is
//! transcribed from the upstream source, not invented (per-function citations
//! inline). See `docs/adr/M4-03-aec-op.md` §D-(a).
//!
//! # Sample domain: int16-scale internals, `[-1, 1]` API
//!
//! The upstream float build feeds int16-valued samples through `float` math;
//! its absolute constants (`power` floor `+1`, `N*100` / `N*1000` /
//! `N*10000` / `N*1e9` sanity levels, the ±32000 saturation test) are
//! calibrated to that scale. The public API takes `[-1, 1]` f32 PCM and the
//! implementation multiplies by `32768` on entry and by `1/32768` on exit —
//! both exact powers of two, so the boundary scaling is lossless and the
//! internal arithmetic stays verbatim-comparable to upstream (ADR §D-(a)).
//! The output is **not** rounded to the int16 grid (upstream's `WORD2INT`
//! rounds only the emitted integer; the recursive state keeps the unrounded
//! value, so skipping the rounding loses nothing and keeps sub-LSB detail).
//!
//! # Runtime object, not an `OpKind` variant
//!
//! [`Aec`] carries live state across frames (adaptive weights, frequency-
//! domain far-end history, convergence statistics, the far-end queue
//! window) — the same runtime-function rationale as `mimi_rvq` (that module,
//! L61-77) and `flow_sampler` (FR-EX-10 spirit): the `OpKind` dispatch
//! surface has no place for a borrowed state handle, and the M4-05/M4-06
//! full-duplex loops want the tight [`Aec::process`] API, not a graph-node
//! round trip. "第一級 op" means first-class in the public API / C ABI /
//! docs, not an `OpKind` variant (ADR M4-03 §D-(b)). `dispatch.rs` has no
//! AEC arm — graph-side calls fall into the existing `UnsupportedOp` default
//! (FR-EX-08).
//!
//! # No GPU seam (CPU only)
//!
//! The AEC is a low-dimensional (window = 2·frame ≤ a few thousand samples)
//! per-frame streaming filter; device transfer would dominate any GPU win,
//! so there is **no** Compute-seam `HotOp` variant and no Metal / CUDA /
//! Vulkan path (ADR M4-03 §D-(i)). If a future consumer shows a measured
//! need, the seam to extend is `vokra-models/src/compute.rs` (one kernel per
//! (backend, op) — M2 pattern); nothing here silently falls back.
//!
//! # Scope boundary (M4-20 / NLP)
//!
//! This module is FR-OP-60 `aec` only. `denoise` / `agc` / `hpf` /
//! `loudness_norm` (FR-OP-61/62/63) belong to M4-20; the SpeexDSP
//! *preprocessor* NLP (residual echo suppression, a separate upstream
//! module) is deliberately not ported — the linear canceller is the required
//! core, and the NLP need is judged empirically in the M4-05/06 full-duplex
//! demo acceptance (ADR M4-03 §D-(f)).

use vokra_core::stream::{AecRefReader, AecRefWindowStatus};
use vokra_core::{Complex32, Result, VokraError};

use crate::fft::FftPlan;

/// Boundary scale between the `[-1, 1]` public PCM domain and the int16-value
/// domain the upstream float build calibrates its constants to (exact power
/// of two — lossless both ways).
const INT16_SCALE: f32 = 32768.0;

// Float-build constants of mdf.c (upstream L106-117; FLOATING_POINT branch).
const MIN_LEAK: f32 = 0.005;
const VAR1_SMOOTH: f32 = 0.36;
const VAR2_SMOOTH: f32 = 0.7225;
const VAR1_UPDATE: f32 = 0.5;
const VAR2_UPDATE: f32 = 0.25;
const VAR_BACKTRACK: f32 = 4.0;

/// Construction attributes of an [`Aec`] instance.
///
/// The upstream guidance (speex_echo.h L73-74): `frame_size` "should
/// correspond to 10-20 ms" and `filter_length` (the echo tail, in samples)
/// "should generally correspond to 100-500 ms". A constant playback/capture
/// offset is absorbed as long as the echo stays inside the tail
/// (ADR M4-03 §D-(d)).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AecAttrs {
    /// Sample rate of both mic and far-end PCM (a mismatch with the far-end
    /// queue is an explicit error — FR-EX-08).
    pub sample_rate: u32,
    /// Samples per [`Aec::process`] call. Must be even (upstream's
    /// `mdf_inner_prod` consumes sample pairs; an odd frame would silently
    /// drop the last sample — Vokra rejects it explicitly instead) and the
    /// window `2·frame_size` must stay on the FFT's alloc-free Direct path
    /// (largest prime factor ≤ 61; every practical frame size — 128 / 160 /
    /// 256 / 320 / 512 … — qualifies).
    pub frame_size: usize,
    /// Echo tail length in samples; internally rounded up to
    /// `M = ceil(filter_length / frame_size)` blocks (mdf.c L421).
    pub filter_length: usize,
}

impl Default for AecAttrs {
    /// The upstream example configuration (`speexdsp/libspeexdsp/testecho.c`:
    /// `NN = 128`, `TAIL = 1024`, `sampleRate = 8000`).
    fn default() -> Self {
        Self {
            sample_rate: 8000,
            frame_size: 128,
            filter_length: 1024,
        }
    }
}

/// Per-frame outcome of [`Aec::process`] (FR-EX-08: every degraded mode is
/// visible in the return value, never silent).
///
/// Precedence: [`Reset`](AecStatus::Reset) (the divergence guard fired this
/// frame) wins over the reference-coverage statuses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AecStatus {
    /// The far-end window was fully covered and the canceller ran normally.
    Cancelled,
    /// No far-end data covered this frame **and** the filter's far-end
    /// history is entirely zero, so no echo can exist ("再生していなければ
    /// エコーは存在しない" — documented semantics, ADR M4-03 §D-(e)): the mic
    /// frame was copied to the output bit-exactly and the filter state was
    /// left untouched (frozen, not equivalent to processing zeros — the
    /// adaptation statistics do not decay while frozen).
    PassThrough,
    /// `missing` samples of the far-end window had no data and were
    /// zero-filled before processing (`missing == frame_size` means the
    /// window was empty but the filter still holds a live echo tail from
    /// recently played audio, which must keep being cancelled).
    PartialReference {
        /// Number of far-end window samples that had no pushed data.
        missing: usize,
    },
    /// The divergence guard (upstream `screwed_up` machinery, mdf.c
    /// L1013-1037) detected numerical breakage (NaN / negative energy /
    /// blow-up) and reset the canceller; the output of this frame is what
    /// upstream emits on that path (zeros on the hard-failure branch).
    Reset,
}

/// Outcome of one internal canceller frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FrameOutcome {
    Ran,
    ResetByGuard,
}

/// SpeexDSP MDF/AUMDF echo canceller state (mono), the Rust image of
/// `SpeexEchoState_` (mdf.c L126-185) minus the fixed-point-only and
/// `play_buf` members (the crude built-in 2-frame playback queue is replaced
/// by the sample-clock [`vokra_core::stream::aec_ref_queue`] — ADR §D-(c)/(d)).
pub struct Aec {
    attrs: AecAttrs,
    /// `frame_size` (upstream `st->frame_size`).
    n: usize,
    /// Window size `N = 2·frame_size`.
    win: usize,
    /// Number of filter blocks `M = ceil(filter_length / frame_size)`.
    m: usize,

    cancel_count: u64,
    adapted: bool,
    saturated: u32,
    screwed_up: u32,

    spec_average: f32,
    beta0: f32,
    beta_max: f32,
    sum_adapt: f32,
    leak_estimate: f32,

    /// Error/scratch time buffer (`st->e`, len `N`).
    e: Vec<f32>,
    /// Far-end sliding time window (`st->x`, len `N`).
    x: Vec<f32>,
    /// Notched + pre-emphasized mic frame (`st->input`, len `n`).
    input: Vec<f32>,
    /// Filter-response time buffer (`st->y`, len `N`).
    y: Vec<f32>,
    /// Dead-state echo memory for the (unported) NLP hook (`st->last_y`).
    last_y: Vec<f32>,
    /// Far-end spectra history (`st->X`, `(M+1)·N` packed).
    x_freq: Vec<f32>,
    /// Error spectrum (`st->E`, len `N` packed).
    e_freq: Vec<f32>,
    /// Filter-response spectrum scratch (`st->Y`, len `N` packed).
    y_freq: Vec<f32>,
    /// Background filter weights (`st->W`, `M·N` packed).
    w: Vec<f32>,
    /// Foreground filter weights (`st->foreground`, `M·N` packed).
    foreground: Vec<f32>,
    /// Gradient scratch (`st->PHI`, len `N`).
    phi: Vec<f32>,

    davg1: f32,
    davg2: f32,
    dvar1: f32,
    dvar2: f32,

    /// Smoothed far-end power (`st->power`, len `n+1`).
    power: Vec<f32>,
    /// Per-bin learning rate (`st->power_1`, len `n+1`).
    power_1: Vec<f32>,
    /// Weight-constraint scratch (`st->wtmp`, len `N`).
    wtmp: Vec<f32>,
    /// Error power spectrum (`st->Rf`, len `n+1`).
    rf: Vec<f32>,
    /// Response power spectrum (`st->Yf`, len `n+1`).
    yf: Vec<f32>,
    /// Far-end power spectrum (`st->Xf`, len `n+1`).
    xf: Vec<f32>,
    /// Smoothed error spectrum (`st->Eh`, len `n+1`).
    eh: Vec<f32>,
    /// Smoothed response spectrum (`st->Yh`, len `n+1`).
    yh: Vec<f32>,
    pey: f32,
    pyy: f32,

    /// Hann window (`st->window`, len `N`; float build L472-473).
    window: Vec<f32>,
    /// Per-block adaptation proportions (`st->prop`, len `M`).
    prop: Vec<f32>,

    mem_x: f32,
    mem_d: f32,
    mem_e: f32,
    preemph: f32,
    notch_radius: f32,
    notch_mem: [f32; 2],

    /// FFT plan of length `N` (Direct path — checked at construction) and its
    /// pre-allocated complex buffers (`process` is allocation-free:
    /// FR-EX-05 / NFR-RL-08).
    plan: FftPlan,
    cplx_a: Vec<Complex32>,
    cplx_b: Vec<Complex32>,
    cplx_scratch: Vec<Complex32>,

    /// Far-end window scratch (`[-1,1]` domain) for [`Aec::process`].
    far_window: Vec<f32>,
    /// int16-domain scratch frames for the boundary scaling.
    mic_scaled: Vec<f32>,
    far_scaled: Vec<f32>,
    out_scaled: Vec<f32>,

    /// Consecutive all-zero far-end frames processed (value-based). Once it
    /// reaches `m + 2` the whole far-end history (`x` window + `X` blocks,
    /// including the pre-emphasis flush impulse of the first silent frame)
    /// is exactly zero, so an Empty window can pass through bit-exactly.
    /// Initialized to `m + 2`: a fresh (or reset) filter has an all-zero
    /// history by construction.
    zero_far_streak: u64,
}

impl Aec {
    /// Builds a canceller for `attrs` (mdf.c `speex_echo_state_init` +
    /// `SPEEX_ECHO_SET_SAMPLING_RATE`, float build, mono).
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] when `sample_rate == 0`,
    /// `frame_size == 0`, `frame_size` is odd, `filter_length < frame_size`,
    /// or the window `2·frame_size` cannot run on the FFT's alloc-free
    /// Direct path (largest prime factor > 61) — each an explicit error, no
    /// silent adjustment (FR-EX-08).
    pub fn new(attrs: &AecAttrs) -> Result<Self> {
        if attrs.sample_rate == 0 {
            return Err(VokraError::InvalidArgument(
                "aec: sample_rate must be non-zero".into(),
            ));
        }
        if attrs.frame_size == 0 {
            return Err(VokraError::InvalidArgument(
                "aec: frame_size must be non-zero".into(),
            ));
        }
        if attrs.frame_size % 2 != 0 {
            return Err(VokraError::InvalidArgument(format!(
                "aec: frame_size {} must be even (upstream mdf_inner_prod consumes sample \
                 pairs and would silently drop the last sample of an odd frame)",
                attrs.frame_size
            )));
        }
        if attrs.filter_length < attrs.frame_size {
            return Err(VokraError::InvalidArgument(format!(
                "aec: filter_length {} must be >= frame_size {} (at least one filter block)",
                attrs.filter_length, attrs.frame_size
            )));
        }
        let n = attrs.frame_size;
        let win = 2 * n;
        let plan = FftPlan::new(win);
        if !plan.is_direct() {
            return Err(VokraError::InvalidArgument(format!(
                "aec: window size {win} (2*frame_size) has a prime factor above the FFT \
                 Direct-path threshold; pick a smooth frame_size (e.g. 128/160/256/320/512) \
                 so the hot path stays allocation-free"
            )));
        }
        // M = (filter_length + frame_size - 1) / frame_size (mdf.c L421).
        let m = attrs.filter_length.div_ceil(n);
        let rate = attrs.sample_rate;

        // Rate-derived scalars (mdf.c L428-434 float branch / ctl L1233-1246).
        let spec_average = n as f32 / rate as f32;
        let beta0 = (2.0f32 * n as f32) / rate as f32;
        let beta_max = (0.5f32 * n as f32) / rate as f32;
        let notch_radius = notch_radius_for(rate);

        // Hann window (mdf.c L472-473: .5-.5*cos(2πi/N), double math → f32).
        let window = (0..win)
            .map(|i| {
                (0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / win as f64).cos()) as f32
            })
            .collect();

        let mut aec = Self {
            attrs: *attrs,
            n,
            win,
            m,
            cancel_count: 0,
            adapted: false,
            saturated: 0,
            screwed_up: 0,
            spec_average,
            beta0,
            beta_max,
            sum_adapt: 0.0,
            leak_estimate: 0.0,
            e: vec![0.0; win],
            x: vec![0.0; win],
            input: vec![0.0; n],
            y: vec![0.0; win],
            last_y: vec![0.0; win],
            x_freq: vec![0.0; (m + 1) * win],
            e_freq: vec![0.0; win],
            y_freq: vec![0.0; win],
            w: vec![0.0; m * win],
            foreground: vec![0.0; m * win],
            phi: vec![0.0; win],
            davg1: 0.0,
            davg2: 0.0,
            dvar1: 0.0,
            dvar2: 0.0,
            power: vec![0.0; n + 1],
            power_1: vec![1.0; n + 1],
            wtmp: vec![0.0; win],
            rf: vec![0.0; n + 1],
            yf: vec![0.0; n + 1],
            xf: vec![0.0; n + 1],
            eh: vec![0.0; n + 1],
            yh: vec![0.0; n + 1],
            pey: 1.0,
            pyy: 1.0,
            window,
            prop: vec![0.0; m],
            mem_x: 0.0,
            mem_d: 0.0,
            mem_e: 0.0,
            preemph: 0.9,
            notch_radius,
            notch_mem: [0.0; 2],
            plan,
            cplx_a: vec![Complex32::ZERO; win],
            cplx_b: vec![Complex32::ZERO; win],
            cplx_scratch: vec![Complex32::ZERO; win],
            far_window: vec![0.0; n],
            mic_scaled: vec![0.0; n],
            far_scaled: vec![0.0; n],
            out_scaled: vec![0.0; n],
            zero_far_streak: (m + 2) as u64,
        };
        aec.init_prop();
        Ok(aec)
    }

    /// The initial per-block adaptation-rate ladder (mdf.c L479-494 float
    /// build): ratio of ~10 between the first and last block.
    fn init_prop(&mut self) {
        let m = self.m;
        // decay = exp(-2.4/M): C computes 2.4/M in f32, exp in double
        // (math_approx.h float branch: spx_exp = exp), stores f32; the
        // float-build SHR32(x, 1) is the identity.
        let decay = f64::exp(f64::from(-(2.4f32 / m as f32))) as f32;
        self.prop[0] = 0.7;
        let mut sum = self.prop[0];
        for i in 1..m {
            self.prop[i] = self.prop[i - 1] * decay;
            sum += self.prop[i];
        }
        for i in (0..m).rev() {
            self.prop[i] = (0.8f32 * self.prop[i]) / sum;
        }
    }

    /// Frame size (samples per [`process`](Self::process) call).
    #[must_use]
    pub fn frame_size(&self) -> usize {
        self.n
    }

    /// Number of MDF filter blocks `M`.
    #[must_use]
    pub fn filter_blocks(&self) -> usize {
        self.m
    }

    /// Construction attributes.
    #[must_use]
    pub fn attrs(&self) -> &AecAttrs {
        &self.attrs
    }

    /// Current leak estimate (Valin 2007 `Pey/Pyy` regression) — a
    /// diagnostic convergence signal in `[MIN_LEAK, 1]` once adaptation has
    /// engaged; `0.0` before the first adapted frame.
    #[must_use]
    pub fn leak_estimate(&self) -> f32 {
        self.leak_estimate
    }

    /// Whether the filter considers itself past minimal adaptation
    /// (mdf.c L1124-1127).
    #[must_use]
    pub fn is_adapted(&self) -> bool {
        self.adapted
    }

    /// Resets the canceller to its as-new state: the next output sequence is
    /// bit-exact with a fresh [`Aec::new`] of the same attrs (pinned by the
    /// `reset_reproduces_a_fresh_instance` test).
    ///
    /// Upstream note: `speex_echo_state_reset` deliberately keeps the
    /// adjusted `prop` ladder and half of `last_y`; this port re-initializes
    /// them too — the Vokra contract is *as-new* (the same thought as the
    /// M3-14 barge-in reset, which this method pairs with: call `reset`
    /// after `Stream::interrupt`).
    pub fn reset(&mut self) {
        self.cancel_count = 0;
        self.adapted = false;
        self.saturated = 0;
        self.screwed_up = 0;
        self.sum_adapt = 0.0;
        self.leak_estimate = 0.0;
        for v in self
            .e
            .iter_mut()
            .chain(self.x.iter_mut())
            .chain(self.input.iter_mut())
            .chain(self.y.iter_mut())
            .chain(self.last_y.iter_mut())
            .chain(self.x_freq.iter_mut())
            .chain(self.e_freq.iter_mut())
            .chain(self.y_freq.iter_mut())
            .chain(self.w.iter_mut())
            .chain(self.foreground.iter_mut())
            .chain(self.phi.iter_mut())
            .chain(self.wtmp.iter_mut())
            .chain(self.rf.iter_mut())
            .chain(self.yf.iter_mut())
            .chain(self.xf.iter_mut())
            .chain(self.eh.iter_mut())
            .chain(self.yh.iter_mut())
            .chain(self.power.iter_mut())
        {
            *v = 0.0;
        }
        self.power_1.fill(1.0);
        self.davg1 = 0.0;
        self.davg2 = 0.0;
        self.dvar1 = 0.0;
        self.dvar2 = 0.0;
        self.pey = 1.0;
        self.pyy = 1.0;
        self.mem_x = 0.0;
        self.mem_d = 0.0;
        self.mem_e = 0.0;
        self.notch_mem = [0.0; 2];
        self.init_prop();
        self.zero_far_streak = (self.m + 2) as u64;
    }

    /// Cancels the echo of one mic frame against the time-aligned far-end
    /// window from `reader` (the [`vokra_core::stream::aec_ref_queue`]
    /// reader half). `mic_pos` is the mic frame's absolute sample position
    /// on the same sample clock the playback side uses for
    /// `AecRefWriter::push`.
    ///
    /// Statuses (ADR M4-03 §D-(e)): [`AecStatus::Cancelled`] on a fully
    /// covered window; [`AecStatus::PartialReference`] when part (or all) of
    /// the window was zero-filled but the filter still holds a live echo
    /// tail; [`AecStatus::PassThrough`] when nothing is playing *and* the
    /// far-end history is silent (mic copied bit-exactly, state frozen);
    /// [`AecStatus::Reset`] when the divergence guard fired.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on `mic.len() != frame_size`,
    /// `out.len() != frame_size`, or a queue/attrs sample-rate mismatch
    /// (FR-EX-08 — no silent resampling or reframing).
    // ZERO-ALLOC-BEGIN (M4-03-T11, FR-EX-05 / NFR-RL-08: the whole
    // process()/process_with_far() call tree below runs from the audio /
    // inference thread and must not allocate — scratch is pre-allocated in
    // `new`; guarded by scripts/check-hot-path-allocs.sh plus the counting-
    // allocator proof in tests/aec_hot_path_alloc.rs. Error paths may build
    // a `format!` message: errors are rare and off the hot path.)
    pub fn process(
        &mut self,
        mic: &[f32],
        mic_pos: u64,
        reader: &mut AecRefReader,
        out: &mut [f32],
    ) -> Result<AecStatus> {
        self.validate_frame(mic, out)?;
        if reader.sample_rate() != self.attrs.sample_rate {
            return Err(VokraError::InvalidArgument(format!(
                "aec: far-end queue sample rate {} != AecAttrs.sample_rate {}",
                reader.sample_rate(),
                self.attrs.sample_rate
            )));
        }
        // Split-borrow the scratch out of `self` so the closure over the
        // reader does not fight the `&mut self` of the canceller call.
        let mut far_window = std::mem::take(&mut self.far_window);
        let win_status = reader.window(mic_pos, &mut far_window);

        // True pass-through: nothing plays now and the whole far-end history
        // is zero, so no echo can exist (documented semantics — not a silent
        // fallback: the status says so). Bit-exact copy, state frozen.
        if win_status == AecRefWindowStatus::Empty && self.zero_far_streak >= (self.m + 2) as u64 {
            out.copy_from_slice(mic);
            self.far_window = far_window;
            return Ok(AecStatus::PassThrough);
        }

        let outcome = self.cancel_frame_unit(mic, &far_window, out);

        // Value-based silence streak (covers Empty windows, gap zero-fill,
        // and pushed zero-valued audio alike — they are the same signal).
        if far_window.iter().all(|&v| v == 0.0) {
            self.zero_far_streak = self.zero_far_streak.saturating_add(1);
        } else {
            self.zero_far_streak = 0;
        }
        self.far_window = far_window;

        Ok(match outcome {
            FrameOutcome::ResetByGuard => AecStatus::Reset,
            FrameOutcome::Ran => match win_status {
                AecRefWindowStatus::Complete => AecStatus::Cancelled,
                AecRefWindowStatus::Partial { missing } => AecStatus::PartialReference { missing },
                AecRefWindowStatus::Empty => AecStatus::PartialReference { missing: self.n },
            },
        })
    }

    /// Queue-less variant: cancels one mic frame against a caller-aligned
    /// far-end frame of the same length (the surface the parity harness and
    /// the M4-05/06 tests drive directly). Always runs the canceller — the
    /// pass-through / partial semantics live in [`Aec::process`], which owns
    /// the far-end coverage information.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on any length mismatch.
    pub fn process_with_far(
        &mut self,
        mic: &[f32],
        far: &[f32],
        out: &mut [f32],
    ) -> Result<AecStatus> {
        self.validate_frame(mic, out)?;
        if far.len() != self.n {
            return Err(VokraError::InvalidArgument(format!(
                "aec: far frame length {} != frame_size {}",
                far.len(),
                self.n
            )));
        }
        // The queue-less surface still maintains the silence streak so a
        // later `process` call sees a consistent far-end history.
        let all_zero = far.iter().all(|&v| v == 0.0);
        let outcome = self.cancel_frame_unit(mic, far, out);
        if all_zero {
            self.zero_far_streak = self.zero_far_streak.saturating_add(1);
        } else {
            self.zero_far_streak = 0;
        }
        Ok(match outcome {
            FrameOutcome::ResetByGuard => AecStatus::Reset,
            FrameOutcome::Ran => AecStatus::Cancelled,
        })
    }

    fn validate_frame(&self, mic: &[f32], out: &mut [f32]) -> Result<()> {
        if mic.len() != self.n {
            return Err(VokraError::InvalidArgument(format!(
                "aec: mic frame length {} != frame_size {}",
                mic.len(),
                self.n
            )));
        }
        if out.len() != self.n {
            return Err(VokraError::InvalidArgument(format!(
                "aec: out frame length {} != frame_size {}",
                out.len(),
                self.n
            )));
        }
        Ok(())
    }

    /// `[-1,1]`-domain wrapper: scales into the int16 domain, runs the
    /// upstream-verbatim frame, scales back (both scalings are exact
    /// powers of two).
    fn cancel_frame_unit(&mut self, mic: &[f32], far: &[f32], out: &mut [f32]) -> FrameOutcome {
        for i in 0..self.n {
            self.mic_scaled[i] = mic[i] * INT16_SCALE;
            self.far_scaled[i] = far[i] * INT16_SCALE;
        }
        let mut mic_scaled = std::mem::take(&mut self.mic_scaled);
        let mut far_scaled = std::mem::take(&mut self.far_scaled);
        let mut out_scaled = std::mem::take(&mut self.out_scaled);
        let outcome = self.cancel_frame_scaled(&mic_scaled, &far_scaled, &mut out_scaled);
        for i in 0..self.n {
            out[i] = out_scaled[i] * (1.0 / INT16_SCALE);
        }
        // Return the scratch buffers (no allocation on any path).
        mic_scaled.fill(0.0);
        far_scaled.fill(0.0);
        self.mic_scaled = mic_scaled;
        self.far_scaled = far_scaled;
        self.out_scaled = out_scaled;
        outcome
    }

    /// The upstream frame: mdf.c `speex_echo_cancellation` (float build,
    /// `C = K = 1`), int16-domain samples. Structured as the same sequence
    /// of steps as the C function; line references are to the upstream
    /// commit pinned in the module doc.
    fn cancel_frame_scaled(&mut self, mic: &[f32], far: &[f32], out: &mut [f32]) -> FrameOutcome {
        let n = self.n;
        let win = self.win;
        let m = self.m;

        self.cancel_count += 1;
        // ss = .35/M, ss_1 = 1-ss (L712-713).
        let ss = 0.35f32 / m as f32;
        let ss_1 = 1.0 - ss;

        // --- DC notch on the mic frame (L719, filter_dc_notch16). ---------
        self.filter_dc_notch(mic);

        // --- Mic pre-emphasis (L722-743, float branch has no saturation). --
        for i in 0..n {
            let tmp = self.input[i] - self.preemph * self.mem_d;
            self.mem_d = self.input[i];
            self.input[i] = tmp;
        }

        // --- Far-end window slide + pre-emphasis (L748-768). ---------------
        for (i, &far_i) in far.iter().enumerate() {
            self.x[i] = self.x[i + n];
            let tmp = far_i - self.preemph * self.mem_x;
            self.x[i + n] = tmp;
            self.mem_x = far_i;
        }

        // --- Shift the far-end spectra history and transform the new
        //     window (L771-781). ---------------------------------------------
        for j in (0..m).rev() {
            let (src, dst) = self.x_freq.split_at_mut((j + 1) * win);
            dst[..win].copy_from_slice(&src[j * win..(j + 1) * win]);
        }
        self.fft_pack_x0();

        // --- Sxx part 1 (L783-788). The upstream also accumulates Xf here;
        //     that accumulation is dead (Xf is zeroed at L868-869 before its
        //     only read at L1051) and is skipped; the *double* Sxx
        //     accumulation (here + L1042-1046) is live upstream behaviour
        //     and preserved verbatim. -----------------------------------------
        let mut sxx = inner_prod(&self.x[n..], &self.x[n..]);

        // --- Foreground filter response + error (L790-801). ---------------
        spectral_mul_accum(&self.x_freq, &self.foreground, &mut self.y_freq, win, m);
        self.ifft_unpack(YFreq, EBuf);
        for i in 0..n {
            self.e[i] = self.input[i] - self.e[i + n];
        }
        let sff = inner_prod(&self.e[..n], &self.e[..n]);

        // --- Adjust proportional adaptation rates (L803-806). --------------
        if self.adapted {
            adjust_prop(&self.w, win, &mut self.prop);
        }

        // --- Gradient update of the background filter (L807-824); uses the
        //     PREVIOUS frame's error spectrum E against the shifted far-end
        //     history — that one-frame delay is the upstream structure. ------
        if self.saturated == 0 {
            for j in (0..m).rev() {
                weighted_spectral_mul_conj(
                    &self.power_1,
                    self.prop[j],
                    &self.x_freq[(j + 1) * win..(j + 2) * win],
                    &self.e_freq,
                    &mut self.phi,
                );
                for i in 0..win {
                    self.w[j * win + i] += self.phi[i];
                }
            }
        } else {
            self.saturated -= 1;
        }

        // --- AUMDF circular-convolution constraint (L826-865, float
        //     branch): block 0 every frame, plus one rotating block. ---------
        for j in 0..m {
            if j == 0 || (m > 1 && self.cancel_count % (m as u64 - 1) == j as u64 - 1) {
                self.constrain_block(j);
            }
        }

        // --- Zero the power accumulators (L867-869). -----------------------
        for i in 0..=n {
            self.rf[i] = 0.0;
            self.yf[i] = 0.0;
            self.xf[i] = 0.0;
        }

        // --- Background filter response, Dbf, See (L871-886, TWO_PATH). ----
        spectral_mul_accum(&self.x_freq, &self.w, &mut self.y_freq, win, m);
        self.ifft_unpack(YFreq, YBuf);
        for i in 0..n {
            self.e[i] = self.e[i + n] - self.y[i + n];
        }
        let dbf = 10.0f32 + inner_prod(&self.e[..n], &self.e[..n]);
        for i in 0..n {
            self.e[i] = self.input[i] - self.y[i + n];
        }
        let mut see = inner_prod(&self.e[..n], &self.e[..n]);

        // --- TWO_PATH statistics + foreground update / backtrack
        //     (L892-956; float constants L108-115, the equivalent-float
        //     comment L901-906 documents the grouping). ----------------------
        self.davg1 = 0.6f32 * self.davg1 + 0.4f32 * (sff - see);
        self.davg2 = 0.85f32 * self.davg2 + 0.15f32 * (sff - see);
        self.dvar1 = VAR1_SMOOTH * self.dvar1 + (0.4f32 * sff) * (0.4f32 * dbf);
        self.dvar2 = VAR2_SMOOTH * self.dvar2 + (0.15f32 * sff) * (0.15f32 * dbf);

        let update_foreground = ((sff - see) * (sff - see).abs() > sff * dbf)
            || (self.davg1 * self.davg1.abs() > VAR1_UPDATE * self.dvar1)
            || (self.davg2 * self.davg2.abs() > VAR2_UPDATE * self.dvar2);

        if update_foreground {
            self.davg1 = 0.0;
            self.davg2 = 0.0;
            self.dvar1 = 0.0;
            self.dvar2 = 0.0;
            self.foreground.copy_from_slice(&self.w);
            // Smooth transition to avoid blocking artifacts (L926-929).
            for i in 0..n {
                self.e[i + n] = self.window[i + n] * self.e[i + n] + self.window[i] * self.y[i + n];
            }
        } else {
            let reset_background = ((-(sff - see)) * (sff - see).abs()
                > VAR_BACKTRACK * (sff * dbf))
                || ((-self.davg1) * self.davg1.abs() > VAR_BACKTRACK * self.dvar1)
                || ((-self.davg2) * self.davg2.abs() > VAR_BACKTRACK * self.dvar2);
            if reset_background {
                self.w.copy_from_slice(&self.foreground);
                for i in 0..n {
                    self.y[i + n] = self.e[i + n];
                }
                for i in 0..n {
                    self.e[i] = self.input[i] - self.y[i + n];
                }
                see = sff;
                self.davg1 = 0.0;
                self.davg2 = 0.0;
                self.dvar1 = 0.0;
                self.dvar2 = 0.0;
            }
        }

        // --- Output with de-emphasis + mic saturation flag (L959-980). -----
        for i in 0..n {
            let mut tmp_out = self.input[i] - self.e[i + n];
            tmp_out += self.preemph * self.mem_e;
            // Arbitrary saturation test on the raw mic signal (L973-977);
            // in the [-1,1] API domain this is |mic| >= 32000/32768.
            if (mic[i] <= -32000.0 || mic[i] >= 32000.0) && self.saturated == 0 {
                self.saturated = 1;
            }
            out[i] = tmp_out;
            self.mem_e = tmp_out;
        }

        // --- Error signal for the next frame's filter update (L986-991). ---
        for i in 0..n {
            self.e[i + n] = self.e[i];
            self.e[i] = 0.0;
        }

        // --- Correlations (L993-997). --------------------------------------
        let sey = inner_prod(&self.e[n..], &self.y[n..]);
        let syy = inner_prod(&self.y[n..], &self.y[n..]);
        let sdd = inner_prod(&self.input, &self.input);

        // --- Error / response spectra + their powers (L999-1008). ----------
        self.fft_pack(EBuf, EFreq);
        for i in 0..n {
            self.y[i] = 0.0;
        }
        self.fft_pack(YBuf, YFreq);
        power_spectrum_accum(&self.e_freq, &mut self.rf);
        power_spectrum_accum(&self.y_freq, &mut self.yf);

        // --- Sanity check / divergence guard (L1013-1037). -----------------
        // The float build's SHR32 is the identity, so the "adds echo"
        // condition is Sff > Sdd + N*10000 (not the fixed-point shifts);
        // the N*1e9 bound is a double comparison in C (1e9 is a double
        // literal). NaN fails BOTH `>= 0` and `< bound`, so the negated
        // conjunction (kept as a named bool — do NOT De-Morgan the
        // comparisons into `< 0.0` / `>= bound`, which would let NaN pass)
        // lands NaN in the hard branch exactly like upstream.
        let energies_sane = syy >= 0.0
            && sxx >= 0.0
            && see >= 0.0
            && f64::from(sff) < win as f64 * 1e9
            && f64::from(syy) < win as f64 * 1e9
            && f64::from(sxx) < win as f64 * 1e9;
        let really_bad = !energies_sane;
        if really_bad {
            self.screwed_up += 50;
            out[..n].fill(0.0);
        } else if sff > sdd + (win as f32) * 10000.0 {
            self.screwed_up += 1;
        } else {
            self.screwed_up = 0;
        }
        if self.screwed_up >= 50 {
            // "The echo canceller started acting funny and got slapped
            // (reset)" (L1034). The reset is surfaced to the caller as
            // AecStatus::Reset — never a silent recovery (FR-EX-08).
            self.reset();
            return FrameOutcome::ResetByGuard;
        }

        // --- Far-end energy floor + the second (live) Sxx/Xf accumulation
        //     (L1039-1046; the float-build floor is N*100). ------------------
        see = see.max(win as f32 * 100.0);
        sxx += inner_prod(&self.x[n..], &self.x[n..]);
        power_spectrum_accum(&self.x_freq[..win], &mut self.xf);

        // --- Smooth the far-end power estimate (L1049-1051). ---------------
        for j in 0..=n {
            self.power[j] = (ss_1 * self.power[j] + 1.0) + ss * self.xf[j];
        }

        // --- Filtered spectra + correlations (L1053-1068; the descending
        //     loop order is the upstream accumulation order). ----------------
        let mut pey_cur = 1.0f32;
        let mut pyy_cur = 1.0f32;
        for j in (0..=n).rev() {
            let eh_diff = self.rf[j] - self.eh[j];
            let yh_diff = self.yf[j] - self.yh[j];
            pey_cur += eh_diff * yh_diff;
            pyy_cur += yh_diff * yh_diff;
            self.eh[j] = (1.0 - self.spec_average) * self.eh[j] + self.spec_average * self.rf[j];
            self.yh[j] = (1.0 - self.spec_average) * self.yh[j] + self.spec_average * self.yf[j];
        }
        // sqrt in double (spx_sqrt = sqrt), stored f32 (L1070-1071).
        pyy_cur = f64::sqrt(f64::from(pyy_cur)) as f32;
        pey_cur /= pyy_cur;

        // --- Leak estimate update (L1073-1095). -----------------------------
        let mut tmp32 = self.beta0 * syy;
        if tmp32 > self.beta_max * see {
            tmp32 = self.beta_max * see;
        }
        let alpha = tmp32 / see;
        let alpha_1 = 1.0 - alpha;
        self.pey = alpha_1 * self.pey + alpha * pey_cur;
        self.pyy = alpha_1 * self.pyy + alpha * pyy_cur;
        if self.pyy < 1.0 {
            self.pyy = 1.0;
        }
        if self.pey < MIN_LEAK * self.pyy {
            self.pey = MIN_LEAK * self.pyy;
        }
        if self.pey > self.pyy {
            self.pey = self.pyy;
        }
        // Float build: leak = Pey/Pyy directly (the Q14→Q15 doubling at
        // L1092-1095 is dead in float — leak ≤ 1 can never exceed 16383).
        self.leak_estimate = self.pey / self.pyy;

        // --- Residual-to-error ratio (L1114-1121, float branch): .0001 and
        //     3. are double literals in C, so the main expression runs in
        //     f64 before the f32 store; the lower bound runs in f32. ---------
        let mut rer = ((0.0001f64 * f64::from(sxx)) + 3.0f64 * f64::from(self.leak_estimate * syy))
            / f64::from(see);
        let rer_bound = f64::from(sey * sey / (1.0 + see * syy));
        if rer < rer_bound {
            rer = rer_bound;
        }
        if rer > 0.5 {
            rer = 0.5;
        }
        let rer = rer as f32;

        // --- Minimal-adaptation test (L1123-1127). --------------------------
        if !self.adapted && self.sum_adapt > m as f32 && self.leak_estimate * syy > 0.03f32 * syy {
            self.adapted = true;
        }

        // --- Per-bin learning rate (L1129-1171). ----------------------------
        if self.adapted {
            for i in 0..=n {
                let mut r = self.leak_estimate * self.yf[i];
                let e_ = self.rf[i] + 1.0;
                if f64::from(r) > 0.5 * f64::from(e_) {
                    r = 0.5 * e_;
                }
                // C: r = .7*r + .3*(RER*e) with .7/.3 as double literals and
                // the (RER*e) product rounded to f32 by the explicit cast.
                r = ((0.7f64 * f64::from(r)) + 0.3f64 * f64::from(rer * e_)) as f32;
                self.power_1[i] = r / (e_ * (self.power[i] + 10.0));
            }
        } else {
            let mut adapt_rate = 0.0f32;
            if sxx > win as f32 * 1000.0 {
                let mut tmp = 0.25f32 * sxx;
                if f64::from(tmp) > 0.25 * f64::from(see) {
                    tmp = 0.25 * see;
                }
                adapt_rate = tmp / see;
            }
            for i in 0..=n {
                self.power_1[i] = adapt_rate / (self.power[i] + 10.0);
            }
            self.sum_adapt += adapt_rate;
        }

        // --- last_y bookkeeping (L1173-1185): dead state kept for the
        //     future NLP hook (`speex_echo_get_residual` is not ported —
        //     ADR §D-(f)). Upstream uses the WORD2INT-rounded output here;
        //     this port keeps the unrounded value (dead-state-only
        //     difference, documented). ----------------------------------------
        for i in 0..n {
            self.last_y[i] = self.last_y[n + i];
        }
        if self.adapted {
            for i in 0..n {
                self.last_y[n + i] = mic[i] - out[i];
            }
        }

        FrameOutcome::Ran
    }

    /// mdf.c `filter_dc_notch16` (L187-209, float branch), mono/stride 1:
    /// reads the raw mic frame, writes `self.input`.
    fn filter_dc_notch(&mut self, mic: &[f32]) {
        let radius = self.notch_radius;
        // C: den2 = radius*radius + .7*(1-radius)*(1-radius) — the first
        // product rounds in f32, the rest promotes to double (the .7
        // literal), and the sum stores to f32.
        let r2 = radius * radius;
        let one_minus = 1.0f32 - radius;
        let den2 = (f64::from(r2) + 0.7f64 * f64::from(one_minus) * f64::from(one_minus)) as f32;
        for (&vin, inp) in mic.iter().zip(self.input.iter_mut()) {
            let vout = self.notch_mem[0] + vin;
            self.notch_mem[0] = self.notch_mem[1] + 2.0 * (-vin + radius * vout);
            self.notch_mem[1] = vin - den2 * vout;
            *inp = radius * vout;
        }
    }

    /// The AUMDF constraint on weight block `j` (L855-860, float branch):
    /// take the block to the time domain, zero the non-causal half, and
    /// return to the frequency domain.
    fn constrain_block(&mut self, j: usize) {
        let win = self.win;
        let n = self.n;
        // wtmp = ifft(W[j]); wtmp[n..] = 0; W[j] = fft(wtmp).
        unpack_to_complex(&self.w[j * win..(j + 1) * win], &mut self.cplx_a);
        self.plan
            .inverse_raw_into(&self.cplx_a, &mut self.cplx_b, &mut self.cplx_scratch);
        for i in 0..n {
            self.wtmp[i] = self.cplx_b[i].re;
        }
        for i in n..win {
            self.wtmp[i] = 0.0;
        }
        // Forward with the spx_fft 1/N pre-scale.
        let scale = 1.0f32 / win as f32;
        for i in 0..win {
            self.cplx_a[i] = Complex32::from_real(self.wtmp[i] * scale);
        }
        self.plan
            .forward_raw_into(&self.cplx_a, &mut self.cplx_b, &mut self.cplx_scratch);
        pack_from_complex(&self.cplx_b, &mut self.w[j * win..(j + 1) * win]);
    }

    /// spx_fft of the freshly slid far-end window into `X[0]` (history block
    /// zero) — L779-781.
    fn fft_pack_x0(&mut self) {
        let win = self.win;
        let scale = 1.0f32 / win as f32;
        for i in 0..win {
            self.cplx_a[i] = Complex32::from_real(self.x[i] * scale);
        }
        self.plan
            .forward_raw_into(&self.cplx_a, &mut self.cplx_b, &mut self.cplx_scratch);
        pack_from_complex(&self.cplx_b, &mut self.x_freq[..win]);
    }

    /// spx_fft: `src` time buffer → `dst` packed spectrum (with the 1/N
    /// pre-scale of speexdsp's `fftwrap.c` float path).
    fn fft_pack(&mut self, src: TimeBuf, dst: FreqBuf) {
        let win = self.win;
        let scale = 1.0f32 / win as f32;
        {
            let s = match src {
                EBuf => &self.e,
                YBuf => &self.y,
            };
            for (c, &sv) in self.cplx_a.iter_mut().zip(s.iter()) {
                *c = Complex32::from_real(sv * scale);
            }
        }
        self.plan
            .forward_raw_into(&self.cplx_a, &mut self.cplx_b, &mut self.cplx_scratch);
        let d = match dst {
            EFreq => &mut self.e_freq,
            YFreq => &mut self.y_freq,
        };
        pack_from_complex(&self.cplx_b, d);
    }

    /// spx_ifft: `src` packed spectrum → `dst` time buffer (unnormalized
    /// backward, matching speexdsp's `fftwrap.c` float path — the round trip
    /// `ifft(fft(x)) == x` carries the whole normalization).
    fn ifft_unpack(&mut self, src: FreqBuf, dst: TimeBuf) {
        {
            let s = match src {
                EFreq => &self.e_freq,
                YFreq => &self.y_freq,
            };
            unpack_to_complex(s, &mut self.cplx_a);
        }
        self.plan
            .inverse_raw_into(&self.cplx_a, &mut self.cplx_b, &mut self.cplx_scratch);
        let d = match dst {
            EBuf => &mut self.e,
            YBuf => &mut self.y,
        };
        for (dv, c) in d.iter_mut().zip(self.cplx_b.iter()) {
            *dv = c.re;
        }
    }
    // ZERO-ALLOC-END
}

/// Named internal time buffers (borrow-splitting helper for the FFT calls).
#[derive(Clone, Copy, PartialEq, Eq)]
enum TimeBuf {
    EBuf,
    YBuf,
}
use TimeBuf::{EBuf, YBuf};

/// Named internal packed-spectrum buffers.
#[derive(Clone, Copy, PartialEq, Eq)]
enum FreqBuf {
    EFreq,
    YFreq,
}
use FreqBuf::{EFreq, YFreq};

/// mdf.c notch radius by rate (L500-505 / ctl L1241-1246).
fn notch_radius_for(rate: u32) -> f32 {
    if rate < 12_000 {
        0.9
    } else if rate < 24_000 {
        0.982
    } else {
        0.992
    }
}

// ZERO-ALLOC-BEGIN (M4-03-T11: the free-function DSP kernels below are all
// called per frame from the marked region above; slice-only, no allocation.)
/// mdf.c `mdf_inner_prod` (L212-225, float): a pairwise dot product — two
/// products summed into `part`, `part` folded into `sum` — preserving the
/// upstream f32 accumulation grouping. Length must be even (guaranteed by
/// the even-`frame_size` gate).
fn inner_prod(x: &[f32], y: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for (cx, cy) in x.chunks_exact(2).zip(y.chunks_exact(2)) {
        let part = cx[0] * cy[0] + cx[1] * cy[1];
        sum += part;
    }
    sum
}

/// mdf.c `power_spectrum_accum` (L240-249) over the packed layout
/// `[r0, r1, i1, r2, i2, …, r_{N/2}]` (smallft / FFTPACK halfcomplex order —
/// the layout `power_spectrum` L228-237 pins).
fn power_spectrum_accum(x: &[f32], ps: &mut [f32]) {
    let n = x.len();
    ps[0] += x[0] * x[0];
    let mut i = 1;
    let mut j = 1;
    while i < n - 1 {
        ps[j] += x[i] * x[i] + x[i + 1] * x[i + 1];
        i += 2;
        j += 1;
    }
    ps[j] += x[i] * x[i];
}

/// mdf.c float `spectral_mul_accum` (L309-326): per-block complex
/// multiply-accumulate over the packed layout, preserving the upstream
/// accumulation order (outer block loop, inner bin loop).
fn spectral_mul_accum(x: &[f32], w: &[f32], acc: &mut [f32], win: usize, m: usize) {
    for a in acc.iter_mut() {
        *a = 0.0;
    }
    for j in 0..m {
        let xb = &x[j * win..(j + 1) * win];
        let wb = &w[j * win..(j + 1) * win];
        acc[0] += xb[0] * wb[0];
        let mut i = 1;
        while i < win - 1 {
            acc[i] += xb[i] * wb[i] - xb[i + 1] * wb[i + 1];
            acc[i + 1] += xb[i + 1] * wb[i] + xb[i] * wb[i + 1];
            i += 2;
        }
        acc[win - 1] += xb[win - 1] * wb[win - 1];
    }
}

/// mdf.c `weighted_spectral_mul_conj` (L331-345): the gradient correlation
/// `prod = (p·w[bin]) · conj(X) · E` over the packed layout.
fn weighted_spectral_mul_conj(w: &[f32], p: f32, x: &[f32], y: &[f32], prod: &mut [f32]) {
    let n = x.len();
    let mut wt = p * w[0];
    prod[0] = wt * (x[0] * y[0]);
    let mut i = 1;
    let mut j = 1;
    while i < n - 1 {
        wt = p * w[j];
        prod[i] = wt * (x[i] * y[i] + x[i + 1] * y[i + 1]);
        prod[i + 1] = wt * ((-x[i + 1]) * y[i] + x[i] * y[i + 1]);
        i += 2;
        j += 1;
    }
    wt = p * w[j];
    prod[i] = wt * (x[i] * y[i]);
}

/// mdf.c `mdf_adjust_prop` (L347-377, float): per-block proportional rates
/// from the block weight norms (sqrt in double, stored f32). The block count
/// is `prop.len()` (`w.len() == prop.len() * win`).
fn adjust_prop(w: &[f32], win: usize, prop: &mut [f32]) {
    let mut max_sum = 1.0f32;
    for (i, p) in prop.iter_mut().enumerate() {
        let mut tmp = 1.0f32;
        for j in 0..win {
            let v = w[i * win + j];
            tmp += v * v;
        }
        *p = f64::sqrt(f64::from(tmp)) as f32;
        if *p > max_sum {
            max_sum = *p;
        }
    }
    let mut prop_sum = 1.0f32;
    for p in prop.iter_mut() {
        *p += 0.1f32 * max_sum;
        prop_sum += *p;
    }
    for p in prop.iter_mut() {
        *p = (0.99f32 * *p) / prop_sum;
    }
}

/// Packs a full complex spectrum (Hermitian, from a real input) into the
/// smallft/FFTPACK halfcomplex order `[r0, r1, i1, …, r_{N/2}]` that every
/// mdf.c kernel above indexes.
fn pack_from_complex(spec: &[Complex32], out: &mut [f32]) {
    let n = out.len();
    let half = n / 2;
    out[0] = spec[0].re;
    for k in 1..half {
        out[2 * k - 1] = spec[k].re;
        out[2 * k] = spec[k].im;
    }
    out[n - 1] = spec[half].re;
}

/// Inverse of [`pack_from_complex`]: rebuilds the full Hermitian complex
/// spectrum from the packed halfcomplex layout.
fn unpack_to_complex(packed: &[f32], out: &mut [Complex32]) {
    let n = packed.len();
    let half = n / 2;
    out[0] = Complex32::from_real(packed[0]);
    for k in 1..half {
        out[k] = Complex32::new(packed[2 * k - 1], packed[2 * k]);
    }
    out[half] = Complex32::from_real(packed[n - 1]);
    for k in half + 1..n {
        out[k] = out[n - k].conj();
    }
}
// ZERO-ALLOC-END

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::SplitMix64;
    use vokra_core::stream::aec_ref_queue;

    fn attrs_16k() -> AecAttrs {
        AecAttrs {
            sample_rate: 16_000,
            frame_size: 64,
            filter_length: 256, // M = 4
        }
    }

    /// int16-domain white-ish far-end noise in [-amp, amp].
    fn noise(len: usize, amp: f32, seed: u64) -> Vec<f32> {
        let mut rng = SplitMix64::new(seed);
        (0..len)
            .map(|_| (rng.next_unit_f32() * 2.0 - 1.0) * amp)
            .collect()
    }

    /// A fixed echo path: a short decaying FIR inside the filter tail.
    fn echo_path() -> Vec<f32> {
        // Support 40 taps < filter_length; a direct spike + a few
        // reflections, in the [-1,1] convolution-gain domain.
        let mut h = vec![0.0f32; 40];
        h[2] = 0.5;
        h[11] = -0.3;
        h[25] = 0.15;
        h[39] = -0.05;
        h
    }

    /// f64 linear convolution (near-end = echo of far-end).
    fn convolve(far: &[f32], h: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0f32; far.len()];
        for i in 0..far.len() {
            let mut acc = 0.0f64;
            for (k, &hk) in h.iter().enumerate() {
                if i >= k {
                    acc += f64::from(hk) * f64::from(far[i - k]);
                }
            }
            out[i] = acc as f32;
        }
        out
    }

    fn energy(x: &[f32]) -> f64 {
        x.iter().map(|&v| f64::from(v) * f64::from(v)).sum()
    }

    // ---- T04: attrs validation --------------------------------------------

    #[test]
    fn default_attrs_match_upstream_testecho() {
        let d = AecAttrs::default();
        assert_eq!(d.sample_rate, 8000);
        assert_eq!(d.frame_size, 128);
        assert_eq!(d.filter_length, 1024);
        assert!(Aec::new(&d).is_ok());
    }

    #[test]
    fn attrs_negatives_are_explicit_errors() {
        let base = attrs_16k();
        for bad in [
            AecAttrs {
                sample_rate: 0,
                ..base
            },
            AecAttrs {
                frame_size: 0,
                ..base
            },
            AecAttrs {
                frame_size: 63, // odd
                ..base
            },
            AecAttrs {
                filter_length: 32, // < frame_size
                ..base
            },
        ] {
            assert!(
                matches!(Aec::new(&bad), Err(VokraError::InvalidArgument(_))),
                "{bad:?} must be rejected"
            );
        }
        // Window with a large prime factor (2*118 = 236 = 4*59 is fine;
        // 2*122 = 244 = 4*61 fine; 2*134 = 268 = 4*67 → prime 67 > 61).
        let bluestein = AecAttrs {
            frame_size: 134,
            filter_length: 536,
            sample_rate: 16_000,
        };
        assert!(matches!(
            Aec::new(&bluestein),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn block_count_matches_upstream_formula() {
        let aec = Aec::new(&AecAttrs {
            sample_rate: 16_000,
            frame_size: 256,
            filter_length: 1000,
        })
        .unwrap();
        // (1000 + 255) / 256 = 4 (mdf.c L421).
        assert_eq!(aec.filter_blocks(), 4);
        assert_eq!(aec.frame_size(), 256);
    }

    // ---- FFT packing --------------------------------------------------------

    /// spx_fft/spx_ifft round trip: ifft(fft(x)) == x within FP32 noise.
    #[test]
    fn packed_fft_round_trip_is_identity() {
        let mut aec = Aec::new(&attrs_16k()).unwrap();
        let win = aec.win;
        let src: Vec<f32> = (0..win)
            .map(|i| ((i * 37 + 11) % 101) as f32 - 50.0)
            .collect();
        aec.e.copy_from_slice(&src);
        aec.fft_pack(EBuf, EFreq);
        // Move the spectrum over to Y so the unpack writes into y.
        let spec = aec.e_freq.clone();
        aec.y_freq.copy_from_slice(&spec);
        aec.ifft_unpack(YFreq, YBuf);
        for (i, (&got, &want)) in aec.y.iter().zip(src.iter()).enumerate() {
            assert!(
                (got - want).abs() <= 1e-3,
                "roundtrip sample {i}: {got} vs {want}"
            );
        }
    }

    /// The packed forward transform of a one-sample delay: X[k] = e^{-2πik/N}
    /// (packed layout r/i pairs), fixing both the layout and the sign
    /// convention against the analytic DFT.
    #[test]
    fn packed_fft_matches_analytic_dft_of_a_delay() {
        let mut aec = Aec::new(&attrs_16k()).unwrap();
        let win = aec.win;
        aec.e.fill(0.0);
        aec.e[1] = win as f32; // spx_fft pre-scales by 1/N → spectrum of δ[n-1]
        aec.fft_pack(EBuf, EFreq);
        let half = win / 2;
        for k in 0..=half {
            let angle = -2.0 * std::f64::consts::PI * (k as f64) / (win as f64);
            let (re, im) = (angle.cos() as f32, angle.sin() as f32);
            let (got_re, got_im) = if k == 0 {
                (aec.e_freq[0], 0.0)
            } else if k == half {
                (aec.e_freq[win - 1], 0.0)
            } else {
                (aec.e_freq[2 * k - 1], aec.e_freq[2 * k])
            };
            assert!(
                (got_re - re).abs() < 1e-4 && (got_im - im).abs() < 1e-4,
                "bin {k}: got ({got_re}, {got_im}) want ({re}, {im})"
            );
        }
    }

    // ---- T06: known-coefficient filter application vs direct oracle --------

    /// With the adaptive machinery untouched, a hand-planted weight block
    /// equal to the spectrum of a known FIR must reproduce time-domain
    /// convolution on the second (valid) half of the window — the
    /// overlap-save identity the MDF filter is built on.
    #[test]
    fn planted_filter_block_reproduces_time_domain_convolution() {
        let attrs = attrs_16k();
        let mut aec = Aec::new(&attrs).unwrap();
        let n = aec.n;
        let win = aec.win;

        // FIR with support in the first half of the window.
        let mut h = vec![0.0f32; win];
        h[0] = 0.8;
        h[3] = -0.4;
        h[10] = 0.25;
        h[n - 1] = 0.1;

        // Plant W[0] = FFT(h) (packed, with the spx_fft 1/N scale folded the
        // same way the constraint step re-transforms weights).
        let scale = 1.0 / win as f32;
        for (c, &hv) in aec.cplx_a.iter_mut().zip(h.iter()) {
            *c = Complex32::from_real(hv * scale);
        }
        aec.plan
            .forward_raw_into(&aec.cplx_a.clone(), &mut aec.cplx_b, &mut aec.cplx_scratch);
        let mut w0 = vec![0.0f32; win];
        pack_from_complex(&aec.cplx_b, &mut w0);
        aec.w[..win].copy_from_slice(&w0);

        // Far-end window: two consecutive frames of deterministic noise.
        let far = noise(win, 1000.0, 7);
        aec.x.copy_from_slice(&far);
        aec.fft_pack_x0();

        // Apply the filter: y = ifft(sum_j X[j]·W[j]) — only block 0 is
        // non-zero here.
        spectral_mul_accum(
            &aec.x_freq.clone(),
            &aec.w.clone(),
            &mut aec.y_freq,
            win,
            aec.m,
        );
        aec.ifft_unpack(YFreq, YBuf);

        // Direct oracle: the second half of the circular convolution of
        // (far ⊛ h) equals the linear convolution there because h's support
        // (< n) never wraps into it — the ifft output is scaled by N times
        // the 1/N of the two forward transforms, i.e. y = (1/N)·conv.
        for i in n..win {
            let mut acc = 0.0f64;
            for (k, &hk) in h.iter().enumerate() {
                if hk != 0.0 && i >= k {
                    acc += f64::from(hk) * f64::from(far[i - k]);
                }
            }
            let want = (acc / f64::from(win as u32)) as f32;
            assert!(
                (aec.y[i] - want).abs() <= 2e-2_f32.max(want.abs() * 1e-3),
                "sample {i}: {} vs {}",
                aec.y[i],
                want
            );
        }
    }

    // ---- T07: adaptation, reset, divergence guard --------------------------

    /// Far-end-only convergence: with a fixed echo path and no near-end
    /// speech, the residual energy of late frames must drop well below the
    /// early frames (the relative comparison the ticket asks for; absolute
    /// dB thresholds belong to the ERLE e2e test, T09).
    #[test]
    fn far_end_only_residual_energy_decreases() {
        let attrs = attrs_16k();
        let mut aec = Aec::new(&attrs).unwrap();
        let n = attrs.frame_size;
        let frames = 240;
        let far = noise(frames * n, 8000.0, 42);
        let near = convolve(&far, &echo_path());

        let mut out = vec![0.0f32; n];
        let mut early = 0.0f64;
        let mut late = 0.0f64;
        for f in 0..frames {
            let mic: Vec<f32> = near[f * n..(f + 1) * n]
                .iter()
                .map(|v| v / INT16_SCALE)
                .collect();
            let farf: Vec<f32> = far[f * n..(f + 1) * n]
                .iter()
                .map(|v| v / INT16_SCALE)
                .collect();
            let status = aec.process_with_far(&mic, &farf, &mut out).unwrap();
            assert_ne!(
                status,
                AecStatus::Reset,
                "guard must not fire on clean data"
            );
            let e = energy(&out);
            if (20..60).contains(&f) {
                early += e;
            }
            if (200..240).contains(&f) {
                late += e;
            }
        }
        assert!(
            late < early,
            "adaptive filter must reduce residual energy: early {early:e} late {late:e}"
        );
        assert!(
            late < 0.25 * early,
            "expected a clear reduction (got early {early:e} → late {late:e})"
        );
    }

    /// reset() must reproduce a fresh instance bit-for-bit.
    #[test]
    fn reset_reproduces_a_fresh_instance() {
        let attrs = attrs_16k();
        let n = attrs.frame_size;
        let frames = 30;
        let far = noise(frames * n, 6000.0, 9);
        let near = convolve(&far, &echo_path());
        let run = |aec: &mut Aec| -> Vec<u32> {
            let mut bits = Vec::new();
            let mut out = vec![0.0f32; n];
            for f in 0..frames {
                let mic: Vec<f32> = near[f * n..(f + 1) * n]
                    .iter()
                    .map(|v| v / INT16_SCALE)
                    .collect();
                let farf: Vec<f32> = far[f * n..(f + 1) * n]
                    .iter()
                    .map(|v| v / INT16_SCALE)
                    .collect();
                aec.process_with_far(&mic, &farf, &mut out).unwrap();
                bits.extend(out.iter().map(|v| v.to_bits()));
            }
            bits
        };

        let mut fresh = Aec::new(&attrs).unwrap();
        let want = run(&mut fresh);

        let mut reused = Aec::new(&attrs).unwrap();
        let _ = run(&mut reused); // dirty the state
        reused.reset();
        let got = run(&mut reused);
        assert_eq!(got, want, "post-reset run must be bit-exact vs fresh");
    }

    /// NaN input trips the sanity check: zeroed output, Reset status, and
    /// the next clean frame behaves like a fresh instance (the guard reset
    /// is the full as-new reset).
    #[test]
    fn divergence_guard_resets_and_reports() {
        let attrs = attrs_16k();
        let n = attrs.frame_size;
        let mut aec = Aec::new(&attrs).unwrap();
        let mut out = vec![0.0f32; n];

        let mut mic = vec![0.01f32; n];
        mic[3] = f32::NAN;
        let far = vec![0.02f32; n];
        let status = aec.process_with_far(&mic, &far, &mut out).unwrap();
        assert_eq!(status, AecStatus::Reset, "guard fires on NaN");
        assert!(
            out.iter().all(|&v| v == 0.0),
            "hard-failure output is zeroed"
        );

        // Next clean frame == first frame of a fresh instance, bit-exact.
        let clean_mic = vec![0.01f32; n];
        let clean_far = vec![0.02f32; n];
        let mut out_after = vec![0.0f32; n];
        aec.process_with_far(&clean_mic, &clean_far, &mut out_after)
            .unwrap();
        let mut fresh = Aec::new(&attrs).unwrap();
        let mut out_fresh = vec![0.0f32; n];
        fresh
            .process_with_far(&clean_mic, &clean_far, &mut out_fresh)
            .unwrap();
        let got: Vec<u32> = out_after.iter().map(|v| v.to_bits()).collect();
        let want: Vec<u32> = out_fresh.iter().map(|v| v.to_bits()).collect();
        assert_eq!(got, want);
    }

    // ---- T08: process() + queue integration --------------------------------

    #[test]
    fn process_cancels_with_full_reference_coverage() {
        let attrs = attrs_16k();
        let n = attrs.frame_size;
        let mut aec = Aec::new(&attrs).unwrap();
        let (mut tx, mut rx) = aec_ref_queue(4096, attrs.sample_rate).unwrap();

        let frames = 200;
        let far_i16 = noise(frames * n, 8000.0, 21);
        let near_i16 = convolve(&far_i16, &echo_path());
        let far_unit: Vec<f32> = far_i16.iter().map(|v| v / INT16_SCALE).collect();
        let near_unit: Vec<f32> = near_i16.iter().map(|v| v / INT16_SCALE).collect();

        let mut out = vec![0.0f32; n];
        let mut early = 0.0f64;
        let mut late = 0.0f64;
        for f in 0..frames {
            let pos = (f * n) as u64;
            assert_eq!(tx.push(&far_unit[f * n..(f + 1) * n], pos).unwrap(), n);
            let status = aec
                .process(&near_unit[f * n..(f + 1) * n], pos, &mut rx, &mut out)
                .unwrap();
            assert_eq!(status, AecStatus::Cancelled, "frame {f}");
            let e = energy(&out);
            if (20..60).contains(&f) {
                early += e;
            }
            if (160..200).contains(&f) {
                late += e;
            }
        }
        assert!(late < 0.25 * early, "echo reduced via the queue path too");
    }

    #[test]
    fn process_passes_through_when_nothing_was_ever_played() {
        let attrs = attrs_16k();
        let n = attrs.frame_size;
        let mut aec = Aec::new(&attrs).unwrap();
        let (_tx, mut rx) = aec_ref_queue(1024, attrs.sample_rate).unwrap();
        let mic: Vec<f32> = (0..n).map(|i| (i as f32 - 31.5) / 64.0).collect();
        let mut out = vec![0.0f32; n];
        let status = aec.process(&mic, 0, &mut rx, &mut out).unwrap();
        assert_eq!(status, AecStatus::PassThrough);
        let got: Vec<u32> = out.iter().map(|v| v.to_bits()).collect();
        let want: Vec<u32> = mic.iter().map(|v| v.to_bits()).collect();
        assert_eq!(got, want, "pass-through is a bit-exact copy");
    }

    #[test]
    fn process_reports_partial_reference_and_keeps_cancelling_the_tail() {
        let attrs = attrs_16k();
        let n = attrs.frame_size;
        let mut aec = Aec::new(&attrs).unwrap();
        let (mut tx, mut rx) = aec_ref_queue(4096, attrs.sample_rate).unwrap();

        // Play one frame of audio, then stop pushing.
        let far: Vec<f32> = noise(n, 6000.0, 5)
            .iter()
            .map(|v| v / INT16_SCALE)
            .collect();
        assert_eq!(tx.push(&far, 0).unwrap(), n);
        let mic = vec![0.001f32; n];
        let mut out = vec![0.0f32; n];
        assert_eq!(
            aec.process(&mic, 0, &mut rx, &mut out).unwrap(),
            AecStatus::Cancelled
        );

        // The next window is empty, but the filter history is non-zero →
        // the echo tail is still live: PartialReference{n}, not PassThrough.
        assert_eq!(
            aec.process(&mic, n as u64, &mut rx, &mut out).unwrap(),
            AecStatus::PartialReference { missing: n }
        );

        // After m+2 zero frames the history is fully flushed → PassThrough.
        let mut frame = 2u64;
        loop {
            let status = aec
                .process(&mic, frame * n as u64, &mut rx, &mut out)
                .unwrap();
            if status == AecStatus::PassThrough {
                break;
            }
            assert_eq!(status, AecStatus::PartialReference { missing: n });
            frame += 1;
            assert!(
                frame < 16,
                "pass-through must engage once the tail is flushed"
            );
        }
    }

    #[test]
    fn process_half_covered_window_is_partial_with_count() {
        let attrs = attrs_16k();
        let n = attrs.frame_size;
        let mut aec = Aec::new(&attrs).unwrap();
        let (mut tx, mut rx) = aec_ref_queue(4096, attrs.sample_rate).unwrap();
        // Cover only the first half of the first frame.
        let half: Vec<f32> = noise(n / 2, 4000.0, 3)
            .iter()
            .map(|v| v / INT16_SCALE)
            .collect();
        assert_eq!(tx.push(&half, 0).unwrap(), n / 2);
        let mic = vec![0.001f32; n];
        let mut out = vec![0.0f32; n];
        assert_eq!(
            aec.process(&mic, 0, &mut rx, &mut out).unwrap(),
            AecStatus::PartialReference { missing: n / 2 }
        );
    }

    #[test]
    fn process_validates_lengths_and_rate() {
        let attrs = attrs_16k();
        let n = attrs.frame_size;
        let mut aec = Aec::new(&attrs).unwrap();

        // Rate mismatch between queue and attrs.
        let (_tx, mut rx_bad) = aec_ref_queue(256, 8000).unwrap();
        let mic = vec![0.0f32; n];
        let mut out = vec![0.0f32; n];
        assert!(matches!(
            aec.process(&mic, 0, &mut rx_bad, &mut out),
            Err(VokraError::InvalidArgument(_))
        ));

        let (_tx, mut rx) = aec_ref_queue(256, attrs.sample_rate).unwrap();
        // Wrong mic length.
        assert!(matches!(
            aec.process(&mic[..n - 2], 0, &mut rx, &mut out),
            Err(VokraError::InvalidArgument(_))
        ));
        // Wrong out length.
        let mut short_out = vec![0.0f32; n - 2];
        assert!(matches!(
            aec.process(&mic, 0, &mut rx, &mut short_out),
            Err(VokraError::InvalidArgument(_))
        ));
        // Wrong far length on the queue-less surface.
        assert!(matches!(
            aec.process_with_far(&mic, &mic[..n - 2], &mut out),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    /// The scratch buffers must not grow across frames (the capacity-
    /// stability oracle the repo's hot-path discipline pairs with the
    /// ZERO-ALLOC marker scan; the allocation-count proof lives in
    /// tests/aec_hot_path_alloc.rs).
    #[test]
    fn buffer_capacities_are_stable_across_frames() {
        let attrs = attrs_16k();
        let n = attrs.frame_size;
        let mut aec = Aec::new(&attrs).unwrap();
        let (mut tx, mut rx) = aec_ref_queue(4096, attrs.sample_rate).unwrap();
        let far = noise(n, 5000.0, 2);
        let far_unit: Vec<f32> = far.iter().map(|v| v / INT16_SCALE).collect();
        let mic = vec![0.01f32; n];
        let mut out = vec![0.0f32; n];

        let caps = |a: &Aec| {
            (
                a.e.capacity(),
                a.x_freq.capacity(),
                a.w.capacity(),
                a.cplx_a.capacity(),
                a.far_window.capacity(),
                a.mic_scaled.capacity(),
            )
        };
        let before = caps(&aec);
        for f in 0..50u64 {
            tx.push(&far_unit, f * n as u64).unwrap();
            aec.process(&mic, f * n as u64, &mut rx, &mut out).unwrap();
        }
        assert_eq!(caps(&aec), before, "no scratch reallocation across frames");
    }
}
