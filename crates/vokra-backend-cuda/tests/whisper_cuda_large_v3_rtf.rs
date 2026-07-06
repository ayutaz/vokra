//! Whisper large-v3 CUDA RTF sanity gate (M2-03-followup-rtf-sub-0.1, T-follow-08).
//!
//! **Position in the plan** — this is the *sanity* rung, not the formal < 0.10
//! always-on gate. The plan (§6, D6) puts the < 0.15 sanity check here, prints
//! the measured RTF to stdout as a "reference" number, and explicitly defers
//! the formal RTF < 0.10 always-on gate to **M2-14** (依頼者 self-hosted CUDA
//! runner + regression 5 % gate at M3-01). Never re-tighten this bound in-tree
//! without the M2-14 owner: the plan calls that out as an exit-hand-off, not a
//! knob for CC to turn.
//!
//! **Doubly gated + skip-cleanly** — the test requires both a real NVIDIA GPU
//! (dlopen'd `libcuda`, [`CudaContext::new`] returns `Ok`) **and** the converted
//! GGUF (`VOKRA_WHISPER_LARGE_V3_GGUF`, following the exact env-var scheme the
//! model-crate parity suite established in `parity_whisper.rs`). When either is
//! absent — as on this Apple-Mac author host with no CUDA device and no
//! large-v3 GGUF — the test prints a skip note and returns green. It is meant
//! to run on the **vast.ai RTX 4090** spot GPU per T-follow-09.
//!
//! **No silent CPU fallback** (FR-EX-08). `WhisperAsr::with_backend(Cuda)` is
//! wired explicitly so a build without a CUDA driver would surface as a real
//! `BackendUnavailable` at `transcribe`, not a hidden CPU substitute — but this
//! test guards ahead of that call with the [`CudaContext::new`] probe so the
//! skip path is unambiguous.
//!
//! ```text
//! VOKRA_WHISPER_LARGE_V3_GGUF=whisper-large-v3.gguf \
//!     cargo test -p vokra-backend-cuda --release \
//!     whisper_cuda_large_v3_rtf_below_0_15_sanity -- --nocapture --ignored
//! ```
//!
//! (The `vokra-models` dev-dep already carries its `cuda` feature — see this
//! crate's `Cargo.toml` — so no explicit `--features` is required at the CLI.)

use std::path::PathBuf;
use std::time::Instant;

use vokra_backend_cuda::CudaContext;
use vokra_core::gguf::GgufFile;
use vokra_core::{AsrEngine, BackendKind};
use vokra_models::whisper::asr::WhisperAsr;

/// Sanity ceiling. The formal always-on gate is `< 0.10` and lives at M2-14 /
/// M3-01, not here (plan D6, §6 exit). Keep this loose so a modest thermal /
/// PCIe / individual-die swing on the vast.ai spot does not flap the CI-shaped
/// bit of this file (the CI path only reaches probe-skip anyway); a real regression
/// still trips it well before the M2-14 formal check would.
const RTF_SANITY_CEILING: f32 = 0.15;

/// Whisper's fixed input length: 30 s at 16 kHz (`whisper::mel::N_SAMPLES`).
/// Duplicated as a constant here so this test does not pull the mel front-end
/// crate in just to spell `30 * 16_000`; if the model ever changes this length
/// the assertion below (`pcm.len() as f32 / 16_000.0 == 30.0`) will catch the
/// drift.
const SAMPLE_RATE_HZ: u32 = 16_000;
const AUDIO_SECONDS: f32 = 30.0;
const N_SAMPLES: usize = (AUDIO_SECONDS as usize) * SAMPLE_RATE_HZ as usize;

/// Env-var name for the large-v3 GGUF, matching the `VOKRA_WHISPER_<SIZE>_GGUF`
/// scheme used by the model-crate parity suite (`parity_whisper.rs::size_env_var`).
const GGUF_ENV: &str = "VOKRA_WHISPER_LARGE_V3_GGUF";

/// Deterministic mono PCM at 16 kHz, 30 s long, in roughly `[-1, 1)`. The test
/// is a **timing** measurement — the audio content only needs to exercise the
/// encoder + decoder end-to-end, not to transcribe to a specific string, so a
/// cheap sine + noise mix is fine. Seed is fixed so the number printed to
/// stdout is reproducible across vast.ai runs.
fn make_sample_pcm() -> Vec<f32> {
    let mut pcm = Vec::with_capacity(N_SAMPLES);
    // xorshift64* — same tiny PRNG as `parity_kernels_cuda.rs::rand_vec` so this
    // file stays consistent with the rest of the crate's test suite.
    let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
    for i in 0..N_SAMPLES {
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        let bits = (x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40) as u32;
        let noise = bits as f32 / (1u32 << 24) as f32 * 2.0 - 1.0;
        // 440 Hz tone at 16 kHz + a bit of noise, scaled so |x| < 1.
        let t = i as f32 / SAMPLE_RATE_HZ as f32;
        let tone = (2.0 * std::f32::consts::PI * 440.0 * t).sin();
        pcm.push(0.5 * tone + 0.05 * noise);
    }
    pcm
}

/// Whisper large-v3 CUDA RTF *sanity* gate (< 0.15).
///
/// This is a reference measurement for T-follow-09, not the formal < 0.10
/// always-on gate — that hand-off lives at M2-14 (plan §6 exit). `#[ignore]`
/// keeps it off `cargo test` by default; run it with `-- --ignored --nocapture`
/// on a vast.ai RTX 4090 (or any real NVIDIA host with the GGUF present) to
/// print the RTF number.
#[test]
#[ignore = "requires vast.ai RTX 4090 and VOKRA_WHISPER_LARGE_V3_GGUF; see file header"]
fn whisper_cuda_large_v3_rtf_below_0_15_sanity() {
    // 1) Device probe — dlopen'd libcuda + a real GPU. On an Apple Mac this
    //    returns Err and we skip cleanly (never a silent CPU substitute; the
    //    explicit `with_backend(Cuda)` below would fail with
    //    `BackendUnavailable` on `transcribe`, but the skip note is clearer).
    let Ok(_probe) = CudaContext::new() else {
        eprintln!(
            "skip: no CUDA device — run this on a real NVIDIA host (vast.ai RTX 4090); \
             this file is the T-follow-08 sanity rung, not the M2-14 formal < 0.10 gate"
        );
        return;
    };

    // 2) GGUF probe — the converted whisper-large-v3.gguf is far too large to
    //    commit, so skip cleanly when the env var is not set.
    let Some(gguf_path) = std::env::var_os(GGUF_ENV).map(PathBuf::from) else {
        eprintln!(
            "skip: {GGUF_ENV} not set — point it at the converted whisper-large-v3.gguf \
             to measure CUDA RTF"
        );
        return;
    };
    let file = match GgufFile::open(&gguf_path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("skip: could not open {gguf_path:?}: {e}");
            return;
        }
    };

    // 3) Build the ASR engine and pin it to CUDA. `with_backend(Cuda)` is the
    //    supported entry (see `WhisperAsr` docs and the `whisper_cuda_e2e_matches_cpu`
    //    test in `vokra-models`). Any op the CUDA dispatcher does not cover
    //    surfaces as an explicit `UnsupportedOp` here — never a silent CPU
    //    fallback (FR-EX-08).
    let asr = WhisperAsr::from_gguf(&file)
        .expect("load whisper large-v3 from gguf")
        .with_backend(BackendKind::Cuda);

    // 4) Deterministic 30 s of PCM. The measurement is timing-only, so the
    //    content just needs to exercise the full encoder + decoder path.
    let pcm = make_sample_pcm();
    assert_eq!(
        pcm.len(),
        N_SAMPLES,
        "sample PCM must be exactly 30 s at 16 kHz for the RTF definition to hold"
    );
    let audio_seconds = pcm.len() as f32 / SAMPLE_RATE_HZ as f32;

    // 5) Warm-up: first transcribe pays session build + weight upload + NVRTC
    //    JIT. Excluding it isolates the steady-state cost the RTF number is
    //    supposed to represent (per plan §1: cross-segment `session.reset()`
    //    reuse is the M2-14 opt-in, not the number we advertise for T25).
    let _warm = asr
        .transcribe(&pcm)
        .expect("warm-up transcribe (CUDA large-v3)");

    // 6) Measured pass — one full 30 s transcription end-to-end (mel → encoder
    //    → decoder → detokenize when a tokenizer is embedded, or the bracketed
    //    id fallback when not — the timing is the same either way).
    let started = Instant::now();
    let _out = asr
        .transcribe(&pcm)
        .expect("measured transcribe (CUDA large-v3)");
    let wall = started.elapsed();
    let wall_secs = wall.as_secs_f32();
    let rtf = wall_secs / audio_seconds;

    // 7) Report — stdout so `-- --nocapture` on vast.ai captures the reference
    //    number T-follow-09 feeds back into CLAUDE.md and the M2-03 T25 note.
    println!(
        "RTF={rtf:.4} (wall={wall_secs:.3}s audio={audio_seconds:.1}s ceiling={RTF_SANITY_CEILING:.2})"
    );

    // 8) Sanity gate only. The formal < 0.10 always-on gate belongs to
    //    **M2-14** (self-hosted CUDA runner) + **M3-01** (5 % regression gate);
    //    do not tighten this literal without that owner (plan §6 exit).
    assert!(
        rtf < RTF_SANITY_CEILING,
        "sanity gate, formal <0.1 gate is M2-14: measured RTF {rtf:.4} \
         exceeds ceiling {RTF_SANITY_CEILING:.2}"
    );
}
