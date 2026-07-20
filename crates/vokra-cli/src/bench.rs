//! `vokra-cli bench` — measure RTF / TTFA / jitter / p50-p95-p99 (M1-10a) and
//! the relative perf-regression gate (M1-10b, NFR-PF-13).
//!
//! ```text
//! vokra-cli bench --model vad.gguf --input speech.wav --iters 20 --warmup 3
//! vokra-cli bench --model voice.gguf --text "hello" --format json --baseline base.json
//! vokra-cli bench --model kokoro.gguf --text "<phonemes>" --style ref_s.f32  # X-06-T24
//! ```
//!
//! Timing uses `std::time::Instant` only (no external bench crate — NFR-DS-02),
//! mirroring the `vokra-ops` / `vokra-models` `harness = false` benches. With
//! `--baseline <report.json>` the measured RTF is compared against the baseline
//! and a **>5% relative** regression sets a non-zero exit code (NFR-PF-13).
//!
//! The **absolute** NFR-PF-01 (Whisper base RTF < 1.0) / NFR-PF-02 (piper-plus
//! RTF < 0.5) thresholds are intentionally NOT asserted here: they require the
//! real full models and a stable measurement lab (blocked), so only the
//! relative-regression scaffold ships in this WP.

use std::process::ExitCode;
use std::time::Instant;

use vokra_core::BackendKind;
use vokra_core::quant::{
    DegradationReport, QuantPolicy, verify_hifigan_int8 as core_verify_hifigan_int8,
};

use crate::engine::{self, ModelTask, TaskHint};
use crate::report::{self, BenchReport, RegressionCheck};
use crate::wav;

/// Relative regression tolerance for the M1-10b gate (NFR-PF-13: 5%).
const REGRESSION_THRESHOLD: f64 = 0.05;

/// Default utterance for TTS bench when `--text` is not supplied.
const DEFAULT_BENCH_TEXT: &str = "the quick brown fox jumps over the lazy dog";

/// Output serialization format for the bench report.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Format {
    /// Flat `key=value` line.
    Kv,
    /// Compact JSON object.
    Json,
}

/// Parsed `bench` arguments.
struct BenchArgs {
    /// GGUF path. Required for every task except `mel-frontend`, which reads
    /// its `n_mels` from `vokra.whisper.n_mels` if a GGUF is given and
    /// otherwise defaults to `n_mels = 80` (Whisper base / small / medium /
    /// large-v3 all use 80). Keeping the CLI self-contained lets the CI
    /// `bench-regression` job run without shipping a Whisper GGUF fixture.
    model: Option<String>,
    input: Option<String>,
    text: Option<String>,
    /// `--style <path>` — kokoro only (X-06-T24): a raw little-endian f32 style
    /// vector, `style_dim` or `2·style_dim` floats. Mirrors `run --style`. When
    /// absent the Kokoro arch keeps its explicit reject (benching with a
    /// fabricated style would measure a synthesis the model was never asked to
    /// perform — FR-EX-08). `None` for every other arch.
    style: Option<String>,
    iters: usize,
    warmup: usize,
    format: Format,
    baseline: Option<String>,
    /// Backend the model's hot ops run on (ASR only today). Default CPU; `metal`
    /// / `cuda` require the CLI built with the matching feature, else inference
    /// fails with an explicit unsupported-backend error (no silent CPU fallback).
    backend: BackendKind,
    /// Optional task override — only `mel-frontend` (Whisper log-mel only,
    /// M2-04-T11) is recognized today. Absent → default arch → task mapping.
    task_hint: Option<TaskHint>,

    // ---- M3-15-T11 HTTP-boundary bench routing --------------------------
    //
    // Recognising these flags in `parse_args` is the first half of the
    // M3-15-T11 gap-fill: `docs/m3-15-server-latency-handover.md` § 4
    // documents that `vokra-cli bench` did not accept `--server` /
    // `--endpoint` / `--concurrent` before this WP. The second half — the
    // actual HTTP-boundary measurement — happens in
    // `integrations/vokra-cli-bench-server/` (excluded workspace,
    // pure-`std::net::TcpStream`, zero third-party deps). This CLI does NOT
    // route requests over the wire itself; when `--server` is supplied it
    // emits an explicit FR-EX-08 error that points the operator at the
    // dedicated binary. Rationale: putting a TCP client inside the
    // root-workspace CLI would either (a) require a subprocess spawn per
    // bench run (adds ~1 ms + doubles CLI parsing, easy to get wrong), or
    // (b) duplicate the HTTP/1.1 client that lives — auditable — in the
    // excluded workspace crate. See the redirect message emitted by
    // `execute_http_bench_redirect` below for the operator's exact invocation.
    /// `--server URL` — HTTP-boundary bench base URL. Trigger for the
    /// FR-EX-08 redirect to `vokra-cli-bench-server`. Default None.
    server: Option<String>,
    /// `--endpoint PATH` — URL path appended to `--server` for the HTTP
    /// bench. Default `/api/tts` (piper-plus HTTP). Recognised at parse
    /// time even when `--server` is absent so a caller building an
    /// argv list once for both paths does not need to strip the flag.
    endpoint: String,
    /// `--concurrent N` — concurrent worker count for the HTTP bench.
    /// Default 1. Recognised at parse time (see `endpoint`).
    concurrent: usize,
    /// `--voice NAME` — Piper voice tag included in the TTS body.
    /// Default `en_US-libritts-high` (matches the handover runbook).
    voice: String,
    /// `--budget-ms N` — latency budget echoed into the artifact by
    /// the HTTP bench. Default 75 (NFR-PF-05 v0.9 value).
    budget_ms: u64,
    /// `--timeout-secs N` — per-request HTTP timeout used by the HTTP
    /// bench. Default 30.
    timeout_secs: u64,
}

/// Parses a `--backend` value. The variants always exist (core enum); whether
/// they are *usable* depends on the CLI's compiled features — an unavailable
/// backend fails loudly at inference, never silently on CPU.
pub(crate) fn parse_backend(v: &str) -> Result<BackendKind, String> {
    match v {
        "cpu" => Ok(BackendKind::Cpu),
        "metal" => Ok(BackendKind::Metal),
        "cuda" => Ok(BackendKind::Cuda),
        // Vulkan (M3-02) — recognised at parse time so the CLI accepts the name;
        // the foundation slice has no SPIR-V kernel wired yet, so any actual run
        // surfaces an explicit `UnsupportedOp` from `Compute::for_backend` (no
        // silent CPU fall back, FR-EX-08).
        "vulkan" => Ok(BackendKind::Vulkan),
        // CoreML delegate (M5-01) — recognised at parse time so the CLI accepts
        // the name; the scaffold slice has no wired execution path, so any
        // actual run surfaces an explicit `UnsupportedOp` (ANE present) or
        // `BackendUnavailable` (no ANE / `coreml` feature off) from
        // `Compute::for_backend` (no silent CPU fall back, FR-EX-08).
        "coreml" => Ok(BackendKind::CoreMl),
        // QNN delegate (Qualcomm Hexagon NPU, M5-02) — recognised at parse time
        // so the CLI accepts the name; the scaffold slice has no wired execution
        // path, so any actual run surfaces an explicit `UnsupportedOp` (QNN
        // runtime present) or `BackendUnavailable` (no runtime / `qnn` feature
        // off) from `Compute::for_backend` (no silent CPU fall back, FR-EX-08).
        // NOT NNAPI (FR-BE-07) — QNN is the Hexagon NPU delegate.
        "qnn" => Ok(BackendKind::Qnn),
        other => Err(format!(
            "unknown --backend `{other}` (cpu | metal | cuda | vulkan | coreml | qnn)"
        )),
    }
}

/// Parses a `--task` value into an optional [`TaskHint`]. Unknown values are
/// rejected loudly (no silent fallback to the arch default — FR-EX-08).
fn parse_task_hint(v: &str) -> Result<TaskHint, String> {
    match v {
        "mel-frontend" => Ok(TaskHint::MelFrontend),
        "cosyvoice2-synthetic" => Ok(TaskHint::Cosyvoice2Synthetic),
        other => Err(format!(
            "unknown --task `{other}` (mel-frontend | cosyvoice2-synthetic)"
        )),
    }
}

/// Returns `true` if the loaded model exercises the HiFi-GAN op.
///
/// M2-08-T12 gate: only piper-plus (MB-iSTFT-VITS2's HiFi-GAN ResBlock2
/// stack, `vokra-models::piper_plus::decoder`) uses HiFi-GAN today. Whisper,
/// Silero VAD, and the mel-frontend task do not, so they short-circuit the
/// verify gate. Kokoro (M2-07, future) uses an iSTFTNet head that is
/// deliberately *not* registered as HiFi-GAN (CLAUDE.md レビュアー A 修正) —
/// the registry marker (`quant::HIFIGAN_GENERATOR_OP`) matches piper-plus's
/// TTS decoder only.
fn hifigan_in_use(task: ModelTask) -> bool {
    matches!(task, ModelTask::Tts)
}

/// Runs the HiFi-GAN INT8 opt-in verification gate at bench construction
/// (M2-08-T12).
///
/// Three branches (`vokra_core::quant::verify_hifigan_int8`):
/// - opt-in + eval pass → `Ok(())`
/// - opt-in + eval not run → `VokraError::HifiganInt8VerifyMissing`
/// - opt-in + eval fail → `VokraError::HifiganInt8DegradationExceeded`
///
/// `policy` is `None` until the runtime chunk reader lands (T05 landing pad
/// in `vokra_core::quant::chunk`); until then bench uses the safe default
/// (no opt-in) and the gate is a no-op. The signature accepts an optional
/// [`DegradationReport`] so a follow-up ticket can wire
/// `vokra-cli bench --check-degradation …` into this seam without touching
/// the branching logic.
fn verify_hifigan_int8(
    task: ModelTask,
    policy: Option<&QuantPolicy>,
    report: Option<&DegradationReport>,
) -> Result<(), String> {
    let Some(policy) = policy else {
        // No policy attached (chunk reader not yet wired): the safe default
        // is `no opt-in` per M2-08 (see `resolve::default_vocoder_safe`), so
        // the gate is a no-op.
        return Ok(());
    };
    core_verify_hifigan_int8(policy, hifigan_in_use(task), report).map_err(|e| e.to_string())
}

/// A finished bench run: the report plus the optional regression verdict.
struct BenchOutcome {
    report: BenchReport,
    regression: Option<RegressionCheck>,
}

pub(crate) const USAGE: &str = "\
vokra-cli bench — measure RTF / TTFA / jitter / p50-p95-p99

USAGE:
    In-process GGUF bench (default):
        vokra-cli bench --model <model.gguf> [--input <in.wav>] [--text <string>]
                        [--iters <n>] [--warmup <n>] [--format kv|json] [--baseline <report.json>]

    HTTP-boundary bench (M3-15-T11): recognises the flags below and points at
    the excluded-workspace binary `vokra-cli-bench-server` (pure-std, zero-dep).
    See docs/m3-15-server-latency-handover.md § 4 Option C.

IN-PROCESS OPTIONS:
    --model <path>       GGUF model file (arch selects VAD / ASR / TTS)
    --input <path>       mono WAV input (required for VAD and ASR)
    --text <string>      text to synthesize (TTS; defaults to a fixed phrase).
                         For kokoro this is misaki IPA phonemes (or a raw-id
                         sequence), not English text — there is no G2P bridge.
    --style <path>       kokoro only (X-06-T24): raw little-endian f32 style
                         vector (style_dim or 2*style_dim floats). Required to
                         bench kokoro (a forward conditions on a style row);
                         absent, the kokoro arch is an explicit reject, never a
                         fabricated-input measurement (FR-EX-08).
    --iters <n>          timed iterations           [default 10]
    --warmup <n>         untimed warm-up iterations [default 3]
    --format kv|json     report format              [default kv]
    --baseline <path>    a previous JSON report; a >5% RTF regression exits non-zero
    --backend <name>     cpu | metal | cuda — ASR hot ops backend [default cpu]
                         (metal/cuda need the CLI built with that feature)
    --task <name>        override the arch default task. Recognized values:
                         - `mel-frontend` (Whisper only): benches the log-mel
                           front-end alone (M2-04-T11), so the fused vs unfused
                           RTF isn't polluted by encoder / decoder time;
                         - `cosyvoice2-synthetic` (M3-09-T24 scaffold): runs the
                           CosyVoice2 chunk-aware pipeline with injected
                           deterministic closures + identity Mimi decoder over a
                           fixed 1 s target — no --model required.

HTTP-BOUNDARY OPTIONS (M3-15-T11):
    --server <URL>       server base URL — TRIGGER for the HTTP-boundary bench.
                         When present, vokra-cli emits an explicit FR-EX-08
                         redirect: the actual TCP/HTTP client is the excluded-
                         workspace binary `vokra-cli-bench-server` (pure-std,
                         Cargo.lock contains only itself). Rationale: putting
                         a TCP client inside the root workspace CLI would
                         either add a subprocess spawn (~1 ms + doubled arg
                         parsing) or duplicate ~500 lines of auditable HTTP/1.1
                         code that lives — once — in the excluded workspace.
    --endpoint <PATH>    URL path appended to --server [default /api/tts]
    --concurrent <n>     concurrent worker threads for HTTP bench [default 1]
    --voice <name>       Piper voice tag [default en_US-libritts-high]
    --budget-ms <n>      latency budget echoed by HTTP bench [default 75]
    --timeout-secs <n>   per-request HTTP timeout [default 30]

    -h, --help           print this help
";

fn parse_args(args: &[String]) -> Result<BenchArgs, String> {
    let mut model: Option<String> = None;
    let mut input: Option<String> = None;
    let mut text: Option<String> = None;
    let mut style: Option<String> = None;
    let mut iters: usize = 10;
    let mut warmup: usize = 3;
    let mut format = Format::Kv;
    let mut baseline: Option<String> = None;
    let mut backend = BackendKind::Cpu;
    let mut task_hint: Option<TaskHint> = None;
    // ---- M3-15-T11 HTTP-boundary bench routing --------------------------
    let mut server: Option<String> = None;
    let mut endpoint: String = "/api/tts".to_owned();
    let mut concurrent: usize = 1;
    let mut voice: String = "en_US-libritts-high".to_owned();
    let mut budget_ms: u64 = 75;
    let mut timeout_secs: u64 = 30;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                model = Some(args.get(i + 1).ok_or("--model requires a value")?.clone());
                i += 2;
            }
            "--input" => {
                input = Some(args.get(i + 1).ok_or("--input requires a value")?.clone());
                i += 2;
            }
            "--text" => {
                text = Some(args.get(i + 1).ok_or("--text requires a value")?.clone());
                i += 2;
            }
            "--style" => {
                // X-06-T24: kokoro RTF measurement input. Recognised at parse
                // time; only the `ModelTask::TtsKokoro` arm consumes it. Mirrors
                // `run --style` — a raw f32 style dump.
                style = Some(args.get(i + 1).ok_or("--style requires a path")?.clone());
                i += 2;
            }
            "--iters" => {
                let v = args.get(i + 1).ok_or("--iters requires a value")?;
                iters = v.parse().map_err(|_| format!("invalid --iters `{v}`"))?;
                i += 2;
            }
            "--warmup" => {
                let v = args.get(i + 1).ok_or("--warmup requires a value")?;
                warmup = v.parse().map_err(|_| format!("invalid --warmup `{v}`"))?;
                i += 2;
            }
            "--format" => {
                let v = args.get(i + 1).ok_or("--format requires a value")?;
                format = match v.as_str() {
                    "kv" => Format::Kv,
                    "json" => Format::Json,
                    other => return Err(format!("unknown --format `{other}` (kv | json)")),
                };
                i += 2;
            }
            "--baseline" => {
                baseline = Some(
                    args.get(i + 1)
                        .ok_or("--baseline requires a value")?
                        .clone(),
                );
                i += 2;
            }
            "--backend" => {
                let v = args.get(i + 1).ok_or("--backend requires a value")?;
                backend = parse_backend(v)?;
                i += 2;
            }
            "--task" => {
                let v = args.get(i + 1).ok_or("--task requires a value")?;
                task_hint = Some(parse_task_hint(v)?);
                i += 2;
            }

            // ---- M3-15-T11 HTTP-boundary bench routing ------------------
            //
            // These four flags are RECOGNISED (they parse cleanly and are
            // stored on `BenchArgs`) but only `--server` triggers the
            // redirect emitted from `main`. The other three flags land as
            // defaults if not supplied — no interaction with the in-process
            // GGUF path.
            "--server" => {
                server = Some(args.get(i + 1).ok_or("--server requires a value")?.clone());
                i += 2;
            }
            "--endpoint" => {
                endpoint = args
                    .get(i + 1)
                    .ok_or("--endpoint requires a value")?
                    .clone();
                i += 2;
            }
            "--concurrent" => {
                let v = args.get(i + 1).ok_or("--concurrent requires a value")?;
                concurrent = v
                    .parse()
                    .map_err(|_| format!("invalid --concurrent `{v}`"))?;
                i += 2;
            }
            "--voice" => {
                voice = args.get(i + 1).ok_or("--voice requires a value")?.clone();
                i += 2;
            }
            "--budget-ms" => {
                let v = args.get(i + 1).ok_or("--budget-ms requires a value")?;
                budget_ms = v
                    .parse()
                    .map_err(|_| format!("invalid --budget-ms `{v}`"))?;
                i += 2;
            }
            "--timeout-secs" => {
                let v = args.get(i + 1).ok_or("--timeout-secs requires a value")?;
                timeout_secs = v
                    .parse()
                    .map_err(|_| format!("invalid --timeout-secs `{v}`"))?;
                i += 2;
            }

            other => return Err(format!("unexpected argument `{other}`")),
        }
    }

    if iters == 0 {
        return Err("--iters must be > 0".to_owned());
    }
    if concurrent == 0 {
        return Err("--concurrent must be > 0".to_owned());
    }

    // M3-15-T11: `--server` triggers the HTTP-boundary redirect path and
    // is EXCLUSIVE with the in-process GGUF flags. Reject the combination
    // at parse time so a caller doesn't quietly get "GGUF run, HTTP flags
    // silently ignored" (FR-EX-08). `--model` is the canonical trigger for
    // the in-process path; if both are supplied it is genuinely ambiguous.
    if server.is_some() && model.is_some() {
        return Err(
            "--server (HTTP-boundary bench) and --model (in-process GGUF bench) are \
             mutually exclusive; supply only one. See `vokra-cli bench --help`."
                .to_owned(),
        );
    }
    if server.is_some() && baseline.is_some() {
        return Err(
            "--baseline is only meaningful for the in-process GGUF bench; the \
             HTTP-boundary bench emits P50/P95/P99 latency artifacts instead. \
             See docs/m3-15-server-latency-handover.md § 4."
                .to_owned(),
        );
    }
    if server.is_some() && task_hint.is_some() {
        return Err(
            "--task is only meaningful for the in-process bench; --server routes to \
             the HTTP-boundary bench binary. Drop --task if you intended --server."
                .to_owned(),
        );
    }

    // `--model` is required for every task except the self-contained bench
    // tasks (`mel-frontend`, `cosyvoice2-synthetic`) OR when `--server`
    // takes over. Enforce that asymmetry at parse time so the CLI keeps
    // its FR-EX-08 "loud errors, no silent fallback" posture.
    let no_model_ok = matches!(
        task_hint,
        Some(TaskHint::MelFrontend) | Some(TaskHint::Cosyvoice2Synthetic)
    ) || server.is_some();
    if model.is_none() && !no_model_ok {
        return Err("--model is required".to_owned());
    }
    Ok(BenchArgs {
        model,
        input,
        text,
        style,
        iters,
        warmup,
        format,
        baseline,
        backend,
        task_hint,
        server,
        endpoint,
        concurrent,
        voice,
        budget_ms,
        timeout_secs,
    })
}

/// Runs `f` for `warmup` untimed iterations, then `iters` timed iterations,
/// returning the per-iteration wall-clock latency in seconds.
///
/// Propagates the first error from `f` (a failed inference aborts the bench).
fn time_iters<F>(warmup: usize, iters: usize, mut f: F) -> Result<Vec<f64>, String>
where
    F: FnMut() -> Result<(), String>,
{
    for _ in 0..warmup {
        f()?;
    }
    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let start = Instant::now();
        f()?;
        samples.push(start.elapsed().as_secs_f64());
    }
    Ok(samples)
}

/// GGUF metadata key holding the Whisper mel-band count. Kept in sync with the
/// converter (see `vokra-models::whisper::mod.rs`).
const KEY_WHISPER_N_MELS: &str = "vokra.whisper.n_mels";

/// Loads the model, times the task and (optionally) checks the baseline.
fn execute(args: &BenchArgs) -> Result<BenchOutcome, String> {
    // Special-case: `--task mel-frontend` without `--model`. Skip the GGUF
    // load entirely and run `whisper::mel::log_mel` on either the caller's
    // `--input` WAV or a self-contained 30 s deterministic PCM. `n_mels` is
    // fixed at 80 (the value all shipping Whisper sizes carry in their
    // `vokra.whisper.n_mels` metadata). Wired to feed the CI `bench-regression`
    // job (`docs/bench-baselines/mel_frontend_baseline.json`) without shipping
    // a Whisper GGUF fixture.
    if args.model.is_none() {
        debug_assert!(matches!(
            args.task_hint,
            Some(TaskHint::MelFrontend) | Some(TaskHint::Cosyvoice2Synthetic)
        ));
        return match args.task_hint {
            Some(TaskHint::MelFrontend) => execute_mel_frontend_standalone(args),
            Some(TaskHint::Cosyvoice2Synthetic) => execute_cosyvoice2_synthetic_standalone(args),
            Some(TaskHint::CsmFixtureTokenizer) => Err(
                "bench: --task does not select the CSM fixture tokenizer (it is a \
                 `vokra-cli run --fixture-tokenizer` flag; bench has no CSM task yet)"
                    .to_owned(),
            ),
            None => Err("unreachable: parse_args guarantees a task hint".to_owned()),
        };
    }
    let model_path = args.model.as_deref().expect("model is Some here");
    let (session, task) =
        engine::load_session_with_backend(model_path, args.backend, args.task_hint)?;

    // M2-08-T12: HiFi-GAN INT8 opt-in verify gate. `None` policy short-circuits
    // (the `vokra.quant.*` chunk reader is a T05 landing pad — see
    // `vokra_core::quant::chunk`). When T05 lands this reads
    // `QuantPolicy::from_gguf(session.gguf())?` and threads a caller-supplied
    // `DegradationReport` (`--check-degradation`) into the same seam.
    verify_hifigan_int8(task, None, None)?;

    let (task_name, audio_seconds, samples) = match task {
        // S2S (CSM, M4-05) has no bench harness yet — RTF/TTFA reference
        // numbers ride the streaming test + owner track (T19/T30). Reject
        // rather than fabricate a measurement (FR-EX-08).
        ModelTask::S2s => {
            return Err(
                "bench: arch `csm` (S2S) has no bench task yet — TTFA/RTF reference \
                 numbers come from the M4-05 streaming reference measurement; real-model \
                 numbers are the owner track"
                    .to_owned(),
            );
        }
        // Voxtral (M3-10 / P2 cc-10) is routed through `vokra-cli run` but
        // has no bench task: a real-checkpoint RTF number needs the
        // multi-GB weights + a stable measurement lab (the M2-14 defer),
        // and the synthetic path would measure nothing meaningful. Reject
        // rather than fabricate (FR-EX-08).
        ModelTask::AsrVoxtral => {
            return Err(
                "bench: arch `voxtral` has no bench task yet — real-checkpoint RTF is the \
                 owner track (needs the multi-GB weights on a stable measurement host); \
                 use `vokra-cli run --model <voxtral.gguf> --input <in.wav>` for a \
                 functional run"
                    .to_owned(),
            );
        }
        // Same posture for the Moshi duplex (M4-06): per-frame latency
        // reference numbers ride the duplex demo + owner track (T26/T30).
        ModelTask::S2sDuplex => {
            return Err(
                "bench: arch `moshi` (full-duplex S2S) has no bench task yet — \
                 real-model per-frame latency is the owner track (M4-06 T30); \
                 refusing to fabricate a measurement (FR-EX-08)"
                    .to_owned(),
            );
        }
        // Speaker embedding (CAM++): no bench task is defined — the run-side
        // Speaker arm prints the embedding L2-norm / cosine, but a timing
        // harness needs a settled fbank+embed window definition first.
        // Reject rather than fabricate a measurement (FR-EX-08).
        ModelTask::Speaker => {
            return Err(
                "bench: arch `campplus` (speaker embedding) has no bench task yet — \
                 use `vokra-cli run --model <campplus.gguf> --input <a.wav>` for the \
                 embedding itself (FR-EX-08: refusing to fabricate a measurement)"
                    .to_owned(),
            );
        }
        ModelTask::Vad => {
            let path = args
                .input
                .as_deref()
                .ok_or("bench (VAD): --input <in.wav> is required")?;
            let clip = wav::read_wav(path)?;
            let sr = clip.sample_rate;
            let pcm = clip.samples;
            let audio_seconds = pcm.len() as f64 / f64::from(sr);
            let samples = time_iters(args.warmup, args.iters, || {
                let mut handle = session.open_vad_stream().map_err(|e| e.to_string())?;
                handle.push_pcm(&pcm, sr).map_err(|e| e.to_string())?;
                Ok(())
            })?;
            ("vad", audio_seconds, samples)
        }
        ModelTask::Asr => {
            let path = args
                .input
                .as_deref()
                .ok_or("bench (ASR): --input <in.wav> is required")?;
            let clip = wav::read_wav(path)?;
            let audio_seconds = clip.samples.len() as f64 / f64::from(clip.sample_rate);
            let pcm = clip.samples;
            let samples = time_iters(args.warmup, args.iters, || {
                session.asr().transcribe(&pcm).map_err(|e| e.to_string())?;
                Ok(())
            })?;
            ("asr", audio_seconds, samples)
        }
        ModelTask::Tts => {
            let text = args.text.as_deref().unwrap_or(DEFAULT_BENCH_TEXT);
            // One synth up front to learn the output length (RTF denominator).
            let first = session.tts().synthesize(text).map_err(|e| e.to_string())?;
            let audio_seconds = first.samples.len() as f64 / f64::from(first.sample_rate);
            let samples = time_iters(args.warmup, args.iters, || {
                session.tts().synthesize(text).map_err(|e| e.to_string())?;
                Ok(())
            })?;
            ("tts", audio_seconds, samples)
        }
        ModelTask::MelFrontend => {
            // M2-04-T11: bench the Whisper log-mel front-end alone. Running
            // `whisper::mel::log_mel` directly against the input WAV keeps the
            // measurement isolated to the fused / unfused STFT → power → mel →
            // log10 → transpose path (M2-04-T08 toggle) — no encoder / decoder
            // time leaks into the RTF. `n_mels` is read from the GGUF exactly
            // the way `WhisperConfig` reads it, so the bench and the full ASR
            // path exercise the same front-end shape.
            let path = args
                .input
                .as_deref()
                .ok_or("bench (mel-frontend): --input <in.wav> is required")?;
            let clip = wav::read_wav(path)?;
            let audio_seconds = clip.samples.len() as f64 / f64::from(clip.sample_rate);
            let pcm = clip.samples;
            let n_mels = session
                .gguf()
                .get(KEY_WHISPER_N_MELS)
                .and_then(|v| v.as_u64())
                .ok_or_else(|| format!("GGUF is missing the `{KEY_WHISPER_N_MELS}` metadata key"))?
                as usize;
            let samples = time_iters(args.warmup, args.iters, || {
                // `black_box` prevents LLVM from noticing the result is
                // unused and dead-code-eliminating the whole call. Same trick
                // used by the ops-crate benches (see `fft_bench.rs`).
                let out = vokra_models::whisper::mel::log_mel(&pcm, n_mels);
                std::hint::black_box(out);
                Ok(())
            })?;
            ("mel-frontend", audio_seconds, samples)
        }
        // Kokoro (M2-07 / cc-24) needs an explicit style vector to synthesize:
        // a Kokoro forward conditions its decoder + prosody predictor on a
        // style row, and benching with a fabricated (e.g. zero) style would
        // measure a synthesis the model was never asked to perform. X-06-T24
        // adds `bench --style` so the NFR-PF-09 "Kokoro real-time" row can be
        // measured; ABSENT a --style the arch keeps its explicit reject
        // (FR-EX-08 — no fabricated input). The synthesis path is the exact
        // one `run --style` exercises (`KokoroTts::synthesize_phonemes`), so
        // the RTF numbers are comparable and the helper functions are shared.
        ModelTask::TtsKokoro => {
            let Some(style_path) = args.style.as_deref() else {
                return Err(
                    "bench: arch `kokoro-82m-istftnet` requires --style <s.f32> — a Kokoro \
                     forward conditions its decoder + prosody predictor on a style row, and \
                     benching without one would measure a synthesis the model was never asked \
                     to perform (FR-EX-08). Pass --style <raw f32 dump> (style_dim or \
                     2*style_dim floats) with --text <phonemes>, as in \
                     `vokra-cli run --model <kokoro.gguf> --text <phonemes> --style <s.f32>`"
                        .to_owned(),
                );
            };
            // Kokoro takes misaki IPA phonemes (or a raw-id sequence), NOT the
            // English DEFAULT_BENCH_TEXT: feeding graphemes would fail the
            // phoneme lookup. Require --text explicitly rather than defaulting.
            let text = args.text.as_deref().ok_or(
                "bench (kokoro): --text <phonemes> is required (Kokoro takes misaki IPA \
                 phonemes or a raw-id sequence, not the default English text)",
            )?;
            // Rebuild the concrete engine from the model path — Kokoro's
            // `TtsEngine::synthesize` is a hard NotImplemented (no G2P bridge),
            // so `session.tts()` cannot serve it; this mirrors `run_kokoro`.
            let tts = vokra_models::kokoro::KokoroTts::from_path(model_path)
                .map_err(|e| e.to_string())?
                .with_backend(args.backend);
            let config = tts.config();
            let ids = crate::run::kokoro_phoneme_ids(text, &config.phoneme_symbols)?;
            let style = crate::run::read_style_vector(style_path, config.style_dim)?;
            // One synth up front to learn the output length (RTF denominator).
            let first = tts
                .synthesize_phonemes(&ids, None, Some(&style), 0.0, 1.0)
                .map_err(|e| e.to_string())?;
            let audio_seconds = first.samples.len() as f64 / f64::from(first.sample_rate);
            let samples = time_iters(args.warmup, args.iters, || {
                tts.synthesize_phonemes(&ids, None, Some(&style), 0.0, 1.0)
                    .map_err(|e| e.to_string())?;
                Ok(())
            })?;
            ("kokoro", audio_seconds, samples)
        }
        ModelTask::Cosyvoice2Synthetic => {
            // The engine's load_session does NOT route the cosyvoice2 arch
            // today (T07/T08 real forward path deferred), so this arm is
            // strictly unreachable from a GGUF-driven bench: the standalone
            // path (`execute_cosyvoice2_synthetic_standalone`) handles all
            // Cosyvoice2Synthetic runs. Kept exhaustive so a future
            // engine.rs change that DOES route cosyvoice2 arches surfaces
            // an explicit unimplemented signal (FR-EX-08) instead of
            // silently falling back to the standalone path.
            return Err(
                "bench (cosyvoice2-synthetic): --model is not accepted with this task \
                 hint today; run without --model to exercise the standalone synthetic \
                 bench (M3-09-T24). The GGUF-driven CosyVoice2 bench lands once the T07/\
                 T08 LLM forward wires up."
                    .to_owned(),
            );
        }
    };

    let stats = report::summarize(&samples).ok_or("no timing samples (iters must be > 0)")?;
    let rtf = if audio_seconds > 0.0 {
        stats.mean / audio_seconds
    } else {
        0.0
    };
    let report = BenchReport {
        task: task_name.to_owned(),
        iters: args.iters,
        warmup: args.warmup,
        audio_seconds,
        rtf,
        // Non-streaming: the first audio chunk is the whole clip (mean latency).
        ttfa_ms: stats.mean * 1e3,
        latency: stats,
    };

    let regression = match &args.baseline {
        Some(path) => {
            let bytes = std::fs::read(path).map_err(|e| format!("reading baseline {path}: {e}"))?;
            let baseline_rtf = report::parse_baseline_rtf(&bytes)?;
            Some(report::compare(
                baseline_rtf,
                report.rtf,
                REGRESSION_THRESHOLD,
            ))
        }
        None => None,
    };

    Ok(BenchOutcome { report, regression })
}

/// Default mel-band count used by the standalone mel-frontend bench when no
/// GGUF is provided. Matches `vokra.whisper.n_mels` for every shipping
/// Whisper size (base / small / medium / large-v3 / turbo all carry 80).
const MEL_FRONTEND_STANDALONE_N_MELS: usize = 80;

/// Whisper's fixed audio window in samples (30 s at 16 kHz). Matches
/// `whisper::mel::N_SAMPLES` (kept as a local const so this file compiles
/// even when the model crate is stripped).
const MEL_FRONTEND_STANDALONE_N_SAMPLES: usize = 30 * 16000;

/// Runs the `--task mel-frontend` bench without touching a GGUF. Called from
/// [`execute`] when `args.model` is `None`. Uses the caller's `--input` WAV
/// if given, otherwise a deterministic 30 s PCM (three sine tones + light
/// pseudo-noise, byte-reproducible on every runner) so the CI
/// `bench-regression` job runs against a fixed baseline JSON without
/// shipping any external WAV / GGUF fixture.
fn execute_mel_frontend_standalone(args: &BenchArgs) -> Result<BenchOutcome, String> {
    let (pcm, audio_seconds) = if let Some(path) = args.input.as_deref() {
        let clip = wav::read_wav(path)?;
        let audio_seconds = clip.samples.len() as f64 / f64::from(clip.sample_rate);
        (clip.samples, audio_seconds)
    } else {
        let mut pcm = Vec::with_capacity(MEL_FRONTEND_STANDALONE_N_SAMPLES);
        // Three-tone chord (220 / 440 / 660 Hz) + phase-shifted small noise.
        // Deterministic (no RNG, no time), amplitude bounded well inside f32
        // range so `log_mel` exercises the mel-band accumulator and the log10
        // approximation on non-trivial values.
        for k in 0..MEL_FRONTEND_STANDALONE_N_SAMPLES {
            let t = k as f32 / 16_000.0;
            let phi = t * std::f32::consts::TAU;
            let s = 0.3 * (phi * 220.0).sin()
                + 0.2 * (phi * 440.0).sin()
                + 0.1 * (phi * 660.0).sin()
                + 0.02 * (phi * 12_345.6).sin();
            pcm.push(s);
        }
        (pcm, 30.0)
    };

    let n_mels = MEL_FRONTEND_STANDALONE_N_MELS;
    let samples = time_iters(args.warmup, args.iters, || {
        let out = vokra_models::whisper::mel::log_mel(&pcm, n_mels);
        std::hint::black_box(out);
        Ok(())
    })?;
    let stats = report::summarize(&samples).ok_or("no timing samples (iters must be > 0)")?;
    let rtf = if audio_seconds > 0.0 {
        stats.mean / audio_seconds
    } else {
        0.0
    };
    let report = BenchReport {
        task: "mel-frontend".to_owned(),
        iters: args.iters,
        warmup: args.warmup,
        audio_seconds,
        rtf,
        ttfa_ms: stats.mean * 1e3,
        latency: stats,
    };
    let regression = match &args.baseline {
        Some(path) => {
            let bytes = std::fs::read(path).map_err(|e| format!("reading baseline {path}: {e}"))?;
            let baseline_rtf = report::parse_baseline_rtf(&bytes)?;
            Some(report::compare(
                baseline_rtf,
                report.rtf,
                REGRESSION_THRESHOLD,
            ))
        }
        None => None,
    };
    Ok(BenchOutcome { report, regression })
}

// ---- CosyVoice2 synthetic bench (M3-09-T24 scaffold) ---------------------

/// Runtime CosyVoice2 audio sample rate (Hz). Matches the Mimi codec native
/// rate + the CosyVoice2 model card constant (24 kHz).
const COSYVOICE2_SYNTHETIC_SAMPLE_RATE: u32 = 24_000;

/// Target audio duration for the standalone synthetic bench. Fixed at 1 s
/// so the RTF measurement path exercises multiple chunk boundaries (mimi
/// native rate 12.5–50 Hz → 12–50 chunks per second in typical use).
const COSYVOICE2_SYNTHETIC_TARGET_SECONDS: f64 = 1.0;

/// Chunk size (frames per chunk boundary) for the synthetic bench. Chosen
/// to be small enough that a 1 s target frame count yields several chunks
/// (documents the chunk-aware streaming path, FR-EX-05 hot-path
/// scheduling) without hard-coding an upstream-derived value.
const COSYVOICE2_SYNTHETIC_CHUNK_SIZE: u32 = 4;

/// Runs the `--task cosyvoice2-synthetic` bench without touching a GGUF or
/// safetensors checkpoint. Called from [`execute`] when `args.model` is
/// `None` and the task hint is [`TaskHint::Cosyvoice2Synthetic`].
///
/// Builds a synthetic CosyVoice2 GGUF in memory (arch, Mimi shape defaults,
/// streaming chunk_size / hop), loads a [`CosyVoice2Tts`] from it, and runs
/// the chunk-aware streaming pipeline with injected deterministic closures
/// (zero velocity, constant-ones code closure) over a 1 s target-frame
/// budget. This exercises the T24 RTF measurement API path without a real
/// safetensors checkpoint — the identity Mimi decoder (M3-06 fixture)
/// produces a deterministic feature buffer so the measurement is byte-
/// reproducible on every runner.
///
/// The RTF reported here is **NOT** the real-model RTF (the LLM velocity
/// path is stubbed — see [`TaskHint::Cosyvoice2Synthetic`] doc): it
/// measures the pipeline's overhead (length_conditioning +
/// flow_sample step scheduling + identity Mimi decode + code closure).
/// The T24 real-checkpoint RTF < 1.0 hard-assert lands with the T19
/// CUDA seam + a self-hosted CUDA runner (mirrors the M2-14 defer).
fn execute_cosyvoice2_synthetic_standalone(args: &BenchArgs) -> Result<BenchOutcome, String> {
    use vokra_core::gguf::GgufBuilder;
    use vokra_core::gguf::chunks::KEY_MODEL_ARCH;
    use vokra_core::{CompliancePolicy, ir::graph::LengthConditioningAttrs};
    use vokra_models::cosyvoice2::CosyVoice2Tts;
    use vokra_ops::FlowSamplerState;

    // Non-degenerate synthetic GGUF (mirrors the internal-oracle fixture
    // in vokra-models::cosyvoice2::tests::nondegenerate_gguf_bytes so a
    // caller compiling against this bench sees the same pipeline path
    // the crate's own unit tests exercise).
    let mut b = GgufBuilder::new();
    b.add_string(KEY_MODEL_ARCH, "cosyvoice2");
    b.add_string("vokra.model.name", "cosyvoice2-synthetic-bench");
    b.add_u32(
        "vokra.cosyvoice2.sample_rate",
        COSYVOICE2_SYNTHETIC_SAMPLE_RATE,
    );
    b.add_u32("vokra.cosyvoice2.arch.vocab_size", 32);
    b.add_u32("vokra.cosyvoice2.arch.hidden_dim", 16);
    b.add_u32("vokra.cosyvoice2.arch.n_layer", 2);
    b.add_u32("vokra.cosyvoice2.arch.n_head", 2);
    b.add_u32("vokra.cosyvoice2.arch.ffn_dim", 32);
    b.add_u32("vokra.cosyvoice2.flow.nfe", 2);
    b.add_string("vokra.cosyvoice2.flow.schedule", "linear");
    b.add_u32("vokra.cosyvoice2.mimi.n_codebooks", 2);
    b.add_u32("vokra.cosyvoice2.mimi.codebook_size", 8);
    b.add_u32("vokra.cosyvoice2.mimi.d_model", 4);
    b.add_u32(
        "vokra.cosyvoice2.streaming.chunk_size",
        COSYVOICE2_SYNTHETIC_CHUNK_SIZE,
    );
    b.add_u32(
        "vokra.cosyvoice2.streaming.chunk_hop",
        COSYVOICE2_SYNTHETIC_CHUNK_SIZE,
    );
    let bytes = b
        .to_bytes()
        .map_err(|e| format!("synthetic CosyVoice2 GGUF: {e}"))?;

    // Compliance strict — the registry classifies `cosyvoice2` permissive,
    // so a synthetic (unlabelled) GGUF passes. This exercises the same
    // load path the real-checkpoint bench will use once T07/T08 lands.
    let tts = CosyVoice2Tts::from_gguf_with_policy(&bytes, &CompliancePolicy::strict())
        .map_err(|e| format!("synthetic CosyVoice2 load: {e}"))?;
    let backend_kind = args.backend;
    let tts = tts.with_backend(backend_kind);

    // Target frame count: 1 s of 24 kHz PCM. The pipeline treats target as
    // "chunk-aware frames" (not raw samples), so the number here maps to
    // the chunk_size boundary — pick a target that's a multiple of
    // chunk_size to keep the last-chunk-shorter branch out of the RTF
    // measurement (that branch is exercised in the crate's unit tests).
    // For 1 s of audio at 24 kHz with chunk_size=4, we use 24 chunks =
    // 96 frames — the pipeline generates 24 chunks in a run.
    let target_frames = 96.0f32;
    let audio_seconds = COSYVOICE2_SYNTHETIC_TARGET_SECONDS;

    // Fixed initial Flow Matching state; the identity Mimi decoder does
    // not care about the actual values (shape only).
    let x0 = FlowSamplerState::new(vec![1], vec![0.0]).map_err(|e| e.to_string())?;

    let samples = time_iters(args.warmup, args.iters, || {
        let length_input = LengthConditioningAttrs::user_specified_frames(target_frames);
        let out = tts
            .synthesize_with_pipeline(
                length_input,
                &x0,
                // Zero velocity: each chunk's terminal is the chunk's initial
                // state. Deterministic; documents the "identity" oracle path.
                |s, _t, _p, _c| {
                    Ok(FlowSamplerState {
                        shape: s.shape.clone(),
                        data: vec![0.0; s.data.len()],
                    })
                },
                // Constant-ones codes: identity Mimi decoder produces the same
                // feature every step (col 1 = n_codebooks, else 0), so the
                // measurement isolates pipeline overhead.
                |_s, chunk_frames, n_cb| Ok(vec![1u32; chunk_frames * n_cb]),
            )
            .map_err(|e| e.to_string())?;
        // Prevent LLVM DCE — same trick used by execute_mel_frontend_standalone.
        std::hint::black_box(out);
        Ok(())
    })?;

    let stats = report::summarize(&samples).ok_or("no timing samples (iters must be > 0)")?;
    let rtf = if audio_seconds > 0.0 {
        stats.mean / audio_seconds
    } else {
        0.0
    };
    let report = BenchReport {
        task: "cosyvoice2-synthetic".to_owned(),
        iters: args.iters,
        warmup: args.warmup,
        audio_seconds,
        rtf,
        ttfa_ms: stats.mean * 1e3,
        latency: stats,
    };
    let regression = match &args.baseline {
        Some(path) => {
            let bytes = std::fs::read(path).map_err(|e| format!("reading baseline {path}: {e}"))?;
            let baseline_rtf = report::parse_baseline_rtf(&bytes)?;
            Some(report::compare(
                baseline_rtf,
                report.rtf,
                REGRESSION_THRESHOLD,
            ))
        }
        None => None,
    };
    Ok(BenchOutcome { report, regression })
}

/// Entry point for `vokra-cli bench`.
/// FR-EX-08 redirect: `--server` triggers the HTTP-boundary bench which lives
/// in the excluded workspace crate `integrations/vokra-cli-bench-server/`
/// (pure-`std::net::TcpStream`, zero third-party deps). vokra-cli itself does
/// NOT open a socket. This function prints a byte-stable diagnostic that names
/// the destination binary and reprints the operator's arguments so an
/// automation pipeline can pipe the message into a `run.sh` re-invocation.
///
/// Returns exit code 4 (dedicated "wrong binary for this task" code, distinct
/// from 2 = bad args and 3 = regression). Silent fallback to the in-process
/// bench is deliberately NOT offered — that would silently ignore the
/// operator's intent (FR-EX-08).
fn execute_http_bench_redirect(a: &BenchArgs) -> Result<ExitCode, String> {
    // Safe: the caller only reaches here after parse_args verified server.is_some().
    let server = a.server.as_deref().expect("server checked by caller");
    // The redirect is written to STDERR so a pipeline consuming stdout for a
    // KV / JSON artifact does not accidentally slurp the diagnostic. The
    // process exit code is what CI reads (see docstring above).
    eprintln!(
        "vokra-cli bench: --server routes HTTP-boundary latency to the excluded-\n\
         workspace binary `vokra-cli-bench-server` (pure-std, zero third-party \n\
         deps). vokra-cli itself does not open a socket — putting a TCP client \n\
         inside the root workspace CLI would either require a subprocess spawn \n\
         (~1 ms + doubled CLI parsing) or duplicate the HTTP/1.1 client that \n\
         lives — once, auditable — in the excluded workspace crate.\n\
         \n\
         Build + run:\n\
         \n\
             cargo build --release \\\n\
                 --manifest-path integrations/vokra-cli-bench-server/Cargo.toml\n\
             integrations/vokra-cli-bench-server/target/release/vokra-cli-bench-server \\\n\
                 --server {server} \\\n\
                 --endpoint {endpoint} \\\n\
                 --text \"{text}\" \\\n\
                 --voice {voice} \\\n\
                 --iters {iters} --warmup {warmup} --concurrent {concurrent} \\\n\
                 --budget-ms {budget_ms} --timeout-secs {timeout_secs} \\\n\
                 --format {format}\n\
         \n\
         See docs/m3-15-server-latency-handover.md § 4 Option C for the JSON \n\
         schema, the exit-code contract, and the multi-session / graceful \n\
         degradation semantics both binaries honour.",
        server = server,
        endpoint = a.endpoint,
        // The utterance passes through as-is; if it contains a `"` the operator
        // will notice and re-quote. We do not attempt shell-escape here because
        // the exact escape rules differ across shells (bash / zsh / fish / cmd),
        // and getting it silently wrong would be worse than a visible copy-paste
        // fixup.
        text = a.text.as_deref().unwrap_or("Hello world"),
        voice = a.voice,
        iters = a.iters,
        warmup = a.warmup,
        concurrent = a.concurrent,
        budget_ms = a.budget_ms,
        timeout_secs = a.timeout_secs,
        format = match a.format {
            Format::Kv => "kv",
            Format::Json => "json",
        },
    );
    // Exit 4 = "not in this binary — use the redirect target". Distinct from
    // 2 (bad args, we could still parse them) and 3 (regression gate failure,
    // which requires a real run). CI can add `|| test $? -eq 4` if it wants
    // to auto-re-invoke the excluded workspace binary.
    Ok(ExitCode::from(4))
}

pub(crate) fn main(args: &[String]) -> Result<ExitCode, String> {
    if args.iter().any(|a| a == "-h" || a == "--help") {
        print!("{USAGE}");
        return Ok(ExitCode::SUCCESS);
    }
    let a = parse_args(args)?;

    // M3-15-T11: `--server` short-circuits into the HTTP-boundary redirect.
    // Never reach `execute` (which would try to load a GGUF and fail with a
    // confusing "--model required" error). The redirect is FR-EX-08 clean.
    if a.server.is_some() {
        return execute_http_bench_redirect(&a);
    }

    let outcome = execute(&a)?;

    match a.format {
        Format::Kv => println!("{}", outcome.report.to_kv()),
        Format::Json => println!("{}", outcome.report.to_json()),
    }

    if let Some(reg) = &outcome.regression {
        println!(
            "regression: metric=rtf baseline={:.6} current={:.6} ratio={:.4} \
             threshold={:.2} regressed={}",
            reg.baseline, reg.current, reg.ratio, reg.threshold, reg.regressed
        );
        if reg.regressed {
            eprintln!(
                "bench: RTF regressed by more than {:.0}% vs baseline (NFR-PF-13)",
                reg.threshold * 100.0
            );
            return Ok(ExitCode::from(3));
        }
    }
    Ok(ExitCode::SUCCESS)
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::quant::{CalibrationRef, DEGRADATION_THRESHOLD, QuantPolicy, QuantScheme};

    fn args(parts: &[&str]) -> Vec<String> {
        parts.iter().map(|s| (*s).to_owned()).collect()
    }

    #[test]
    fn parse_backend_accepts_every_selector_and_rejects_unknown() {
        // Each name maps to its enum variant; whether the backend is *usable*
        // depends on the compiled features, but the name must always parse so
        // an unavailable backend fails loudly at inference, never silently on
        // CPU (FR-EX-08). `coreml` (M5-01) / `qnn` (M5-02) are accepted here even
        // though their execution paths are scaffold-only.
        assert_eq!(parse_backend("cpu"), Ok(BackendKind::Cpu));
        assert_eq!(parse_backend("metal"), Ok(BackendKind::Metal));
        assert_eq!(parse_backend("cuda"), Ok(BackendKind::Cuda));
        assert_eq!(parse_backend("vulkan"), Ok(BackendKind::Vulkan));
        assert_eq!(parse_backend("coreml"), Ok(BackendKind::CoreMl));
        assert_eq!(parse_backend("qnn"), Ok(BackendKind::Qnn));
        // An unknown name is a loud parse error naming the valid set. `tpu` is
        // deliberately not a Vokra backend (NNAPI is likewise not a selector —
        // FR-BE-07, permanently unsupported — so it must never parse either).
        let err = parse_backend("tpu").expect_err("tpu is not a selectable backend");
        assert!(err.contains("coreml"), "error must list coreml: {err}");
        assert!(err.contains("qnn"), "error must list qnn: {err}");
        assert!(
            parse_backend("nnapi").is_err(),
            "nnapi must never parse (FR-BE-07)"
        );
    }

    // ----- M2-08-T12: HiFi-GAN INT8 opt-in verify gate --------------------

    fn opt_in_policy() -> QuantPolicy {
        QuantPolicy::new(QuantScheme::Fp16)
            .with_hifigan_int8_opt_in(CalibrationRef::new("hifigan-int8-cal-v1"))
    }

    #[test]
    fn hifigan_int8_verify_no_policy_short_circuits() {
        // No `vokra.quant.*` chunk (today's steady state — T05 landing pad):
        // the gate is a no-op even on a TTS session (piper-plus HiFi-GAN).
        verify_hifigan_int8(ModelTask::Tts, None, None).unwrap();
        verify_hifigan_int8(ModelTask::Asr, None, None).unwrap();
    }

    #[test]
    fn hifigan_int8_verify_opt_in_but_no_tts_short_circuits() {
        // A Whisper session with an opt-in policy (e.g. shared policy across
        // a batch of models) does not exercise the HiFi-GAN op — the gate
        // short-circuits without demanding a report.
        let policy = opt_in_policy();
        verify_hifigan_int8(ModelTask::Asr, Some(&policy), None).unwrap();
        verify_hifigan_int8(ModelTask::Vad, Some(&policy), None).unwrap();
        verify_hifigan_int8(ModelTask::MelFrontend, Some(&policy), None).unwrap();
    }

    #[test]
    fn hifigan_int8_verify_opt_in_plus_tts_without_report_errors_verify_missing() {
        // Branch (b): opt-in + piper-plus TTS + no attached report → hard
        // error mentioning verify-missing.
        let policy = opt_in_policy();
        let err = verify_hifigan_int8(ModelTask::Tts, Some(&policy), None).unwrap_err();
        assert!(
            err.contains("hifigan int8 verify missing"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn hifigan_int8_verify_opt_in_plus_tts_with_passing_report_ok() {
        // Branch (a): opt-in + piper-plus TTS + passing report → allowed.
        let policy = opt_in_policy();
        let report = DegradationReport::from_mel_loss(1.0, 1.02, DEGRADATION_THRESHOLD);
        assert!(report.passes_5pct_gate);
        verify_hifigan_int8(ModelTask::Tts, Some(&policy), Some(&report)).unwrap();
    }

    #[test]
    fn hifigan_int8_verify_opt_in_plus_tts_with_failing_report_errors_degradation_exceeded() {
        // Branch (c): opt-in + piper-plus TTS + failing report → hard error
        // mentioning degradation-exceeded and carrying the delta.
        let policy = opt_in_policy();
        let report = DegradationReport::from_mel_loss(1.0, 1.20, DEGRADATION_THRESHOLD);
        assert!(!report.passes_5pct_gate);
        let err = verify_hifigan_int8(ModelTask::Tts, Some(&policy), Some(&report)).unwrap_err();
        assert!(
            err.contains("hifigan int8 degradation exceeded"),
            "unexpected error: {err}"
        );
        assert!(err.contains("threshold 0.0500"), "unexpected error: {err}");
    }

    #[test]
    fn hifigan_int8_verify_no_opt_in_never_errors() {
        // Policy present but opt-in is off → gate is a no-op regardless of
        // task or report.
        let policy = QuantPolicy::new(QuantScheme::Fp16);
        verify_hifigan_int8(ModelTask::Tts, Some(&policy), None).unwrap();
        verify_hifigan_int8(ModelTask::Asr, Some(&policy), None).unwrap();
    }

    fn silero_fixture() -> String {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/parity/silero_vad/silero-vad-v5.gguf")
            .to_string_lossy()
            .into_owned()
    }

    #[test]
    fn parses_defaults_and_overrides() {
        let a = parse_args(&args(&["--model", "m.gguf"])).expect("valid");
        assert_eq!(a.iters, 10);
        assert_eq!(a.warmup, 3);
        assert_eq!(a.format, Format::Kv);

        let a = parse_args(&args(&[
            "--model", "m.gguf", "--iters", "50", "--warmup", "5", "--format", "json",
        ]))
        .expect("valid");
        assert_eq!(a.iters, 50);
        assert_eq!(a.warmup, 5);
        assert_eq!(a.format, Format::Json);
    }

    #[test]
    fn rejects_bad_args() {
        assert_eq!(
            parse_args(&args(&["--iters", "10"])).err().unwrap(),
            "--model is required"
        );
        assert!(
            parse_args(&args(&["--model", "m", "--iters", "xyz"]))
                .err()
                .unwrap()
                .contains("invalid --iters")
        );
        assert!(
            parse_args(&args(&["--model", "m", "--format", "yaml"]))
                .err()
                .unwrap()
                .contains("unknown --format")
        );
        assert_eq!(
            parse_args(&args(&["--model", "m", "--iters", "0"]))
                .err()
                .unwrap(),
            "--iters must be > 0"
        );
        assert!(
            parse_args(&args(&["--model", "m", "--stray"]))
                .err()
                .unwrap()
                .contains("unexpected argument")
        );
        assert!(
            parse_args(&args(&["--model", "m", "--task", "asr"]))
                .err()
                .unwrap()
                .contains("unknown --task")
        );
    }

    /// X-06-T24: `--style <path>` parses into `BenchArgs.style` for the Kokoro
    /// RTF measurement. The arch-level consumption (a real synth) needs a
    /// Kokoro GGUF and is exercised by the owner track / `vokra-cli run`
    /// parity; parse-level is the CC-verifiable surface.
    #[test]
    fn parses_style_flag_for_kokoro_rtf() {
        let a = parse_args(&args(&["--model", "kokoro.gguf", "--style", "ref_s.f32"]))
            .expect("valid --style");
        assert_eq!(a.style.as_deref(), Some("ref_s.f32"));
        // Default: no style (every non-Kokoro arch, and the reject path).
        let a = parse_args(&args(&["--model", "m.gguf"])).expect("valid");
        assert_eq!(a.style, None);
    }

    /// `--style` with no following value is a loud error, not a silent drop.
    #[test]
    fn rejects_style_flag_without_value() {
        let err = parse_args(&args(&["--model", "kokoro.gguf", "--style"]))
            .err()
            .unwrap();
        assert!(err.contains("--style requires a path"), "got: {err}");
    }

    #[test]
    fn parses_task_mel_frontend_hint() {
        let a = parse_args(&args(&["--model", "m.gguf", "--task", "mel-frontend"]))
            .expect("valid --task");
        assert_eq!(a.task_hint, Some(TaskHint::MelFrontend));
        // Default: no hint.
        let a = parse_args(&args(&["--model", "m.gguf"])).expect("valid");
        assert_eq!(a.task_hint, None);
    }

    /// `--task mel-frontend` without `--model` parses cleanly — the standalone
    /// mel-frontend path (`execute_mel_frontend_standalone`) uses default
    /// `n_mels = 80` and synthesizes a 30 s PCM. Wired to feed the CI
    /// `bench-regression` job without a GGUF fixture.
    #[test]
    fn parses_mel_frontend_without_model() {
        let a = parse_args(&args(&["--task", "mel-frontend"])).expect("valid");
        assert_eq!(a.model, None);
        assert_eq!(a.task_hint, Some(TaskHint::MelFrontend));
    }

    /// The `--model` requirement asymmetry: every non-mel-frontend task still
    /// rejects a missing `--model`.
    #[test]
    fn rejects_missing_model_for_non_mel_frontend() {
        assert_eq!(parse_args(&args(&[])).err().unwrap(), "--model is required");
    }

    // ---- M3-15-T11 --server / --endpoint / --concurrent flag parsing -----

    /// `--server URL` is a valid trigger for the HTTP-boundary bench and
    /// waives the `--model` requirement (the redirect target reads the
    /// server URL directly, no GGUF is loaded here).
    #[test]
    fn parses_server_flag_and_waives_model_requirement() {
        let a = parse_args(&args(&["--server", "http://127.0.0.1:8080"])).expect("valid");
        assert_eq!(a.server.as_deref(), Some("http://127.0.0.1:8080"));
        assert_eq!(a.endpoint, "/api/tts");
        assert_eq!(a.concurrent, 1);
        assert_eq!(a.voice, "en_US-libritts-high");
        assert_eq!(a.budget_ms, 75);
        assert_eq!(a.timeout_secs, 30);
    }

    /// All HTTP-boundary flags parse and set the expected fields.
    #[test]
    fn parses_all_http_boundary_flags() {
        let a = parse_args(&args(&[
            "--server",
            "http://api.example:9000",
            "--endpoint",
            "/v1/audio/transcriptions",
            "--concurrent",
            "8",
            "--voice",
            "ja_JP-my-voice",
            "--budget-ms",
            "90",
            "--timeout-secs",
            "60",
        ]))
        .expect("valid");
        assert_eq!(a.server.as_deref(), Some("http://api.example:9000"));
        assert_eq!(a.endpoint, "/v1/audio/transcriptions");
        assert_eq!(a.concurrent, 8);
        assert_eq!(a.voice, "ja_JP-my-voice");
        assert_eq!(a.budget_ms, 90);
        assert_eq!(a.timeout_secs, 60);
    }

    /// `--concurrent 0` is rejected loudly (FR-EX-08: a zero-worker bench
    /// silently does nothing, which is worse than a hard error).
    #[test]
    fn rejects_concurrent_zero() {
        assert_eq!(
            parse_args(&args(&["--server", "http://x:9", "--concurrent", "0"]))
                .err()
                .unwrap(),
            "--concurrent must be > 0"
        );
    }

    /// `--concurrent NON_INT` is rejected loudly.
    #[test]
    fn rejects_concurrent_non_integer() {
        assert!(
            parse_args(&args(&["--server", "http://x:9", "--concurrent", "many"]))
                .err()
                .unwrap()
                .contains("invalid --concurrent")
        );
    }

    /// `--budget-ms NON_INT` is rejected loudly.
    #[test]
    fn rejects_budget_ms_non_integer() {
        assert!(
            parse_args(&args(&["--server", "http://x:9", "--budget-ms", "fast"]))
                .err()
                .unwrap()
                .contains("invalid --budget-ms")
        );
    }

    /// `--server` and `--model` are mutually exclusive (FR-EX-08: without
    /// this gate, a caller could get either "in-process bench with server
    /// silently ignored" or vice versa — both are silent-fallback bugs).
    #[test]
    fn rejects_server_plus_model_combination() {
        let err = parse_args(&args(&["--server", "http://x:9", "--model", "m.gguf"]))
            .err()
            .unwrap();
        assert!(err.contains("mutually exclusive"), "got: {err}");
    }

    /// `--server` and `--baseline` are mutually exclusive — a baseline JSON
    /// is only meaningful for the in-process RTF regression gate, not for
    /// HTTP-boundary latency.
    #[test]
    fn rejects_server_plus_baseline_combination() {
        let err = parse_args(&args(&[
            "--server",
            "http://x:9",
            "--baseline",
            "baseline.json",
        ]))
        .err()
        .unwrap();
        assert!(err.contains("--baseline"), "got: {err}");
        assert!(err.contains("HTTP-boundary"), "got: {err}");
    }

    /// `--server` and `--task` are mutually exclusive — the redirect target
    /// does not understand `--task`, so accepting it here would silently
    /// discard the operator's intent.
    #[test]
    fn rejects_server_plus_task_combination() {
        let err = parse_args(&args(&["--server", "http://x:9", "--task", "mel-frontend"]))
            .err()
            .unwrap();
        assert!(err.contains("--task"), "got: {err}");
        assert!(err.contains("HTTP-boundary"), "got: {err}");
    }

    /// The redirect message emitted from `execute_http_bench_redirect`
    /// names the excluded workspace binary explicitly. Guard-rail: an
    /// operator reading the message can copy-paste the invocation
    /// verbatim, so the binary path and every documented flag MUST appear.
    #[test]
    fn http_bench_redirect_exit_code_is_4() {
        let a = parse_args(&args(&[
            "--server",
            "http://127.0.0.1:8080",
            "--endpoint",
            "/api/tts",
            "--concurrent",
            "4",
            "--iters",
            "50",
            "--warmup",
            "5",
        ]))
        .expect("valid");
        let code = execute_http_bench_redirect(&a).expect("redirect emits code");
        // ExitCode does not derive PartialEq; format-print to compare (this
        // is the same trick std uses internally). Testing the redirect exits
        // 4 (not 0/2/3) makes it visible from CI without stdout parsing.
        assert_eq!(format!("{code:?}"), format!("{:?}", ExitCode::from(4)));
    }

    /// The USAGE string documents every M3-15-T11 flag AND names the
    /// redirect binary. Regression fence for the copy-paste UX.
    #[test]
    fn usage_documents_m3_15_t11_flags_and_redirect_binary() {
        for flag in [
            "--server",
            "--endpoint",
            "--concurrent",
            "--voice",
            "--budget-ms",
            "--timeout-secs",
        ] {
            assert!(USAGE.contains(flag), "USAGE missing {flag}");
        }
        // The excluded-workspace binary MUST be named so an operator sees
        // it in --help and doesn't have to grep the source for it.
        assert!(
            USAGE.contains("vokra-cli-bench-server"),
            "USAGE missing redirect binary name",
        );
        // Cross-reference to the handover doc so the two never drift.
        assert!(
            USAGE.contains("m3-15-server-latency-handover.md"),
            "USAGE missing handover doc reference",
        );
    }

    #[test]
    fn parses_task_cosyvoice2_synthetic_hint() {
        // `--task cosyvoice2-synthetic` parses cleanly with or without
        // `--model` (M3-09-T24 scaffold; standalone RTF bench).
        let a = parse_args(&args(&["--task", "cosyvoice2-synthetic"])).expect("valid");
        assert_eq!(a.model, None);
        assert_eq!(a.task_hint, Some(TaskHint::Cosyvoice2Synthetic));
    }

    #[test]
    fn bench_cosyvoice2_synthetic_measures_rtf_without_a_gguf_fixture() {
        // The T24 scaffold: no --model, no --input, deterministic
        // synthetic path. The measurement must run to completion and
        // report well-formed RTF / latency stats.
        let a = BenchArgs {
            model: None,
            input: None,
            text: None,
            style: None,
            iters: 2,
            warmup: 1,
            format: Format::Kv,
            baseline: None,
            backend: BackendKind::Cpu,
            task_hint: Some(TaskHint::Cosyvoice2Synthetic),
            server: None,
            endpoint: "/api/tts".to_owned(),
            concurrent: 1,
            voice: "en_US-libritts-high".to_owned(),
            budget_ms: 75,
            timeout_secs: 30,
        };
        let outcome = execute(&a).expect("cosyvoice2-synthetic bench runs");
        assert_eq!(outcome.report.task, "cosyvoice2-synthetic");
        assert_eq!(outcome.report.iters, 2);
        assert_eq!(outcome.report.latency.count, 2);
        // Audio duration is the fixed 1 s target-frame budget.
        assert!(
            (outcome.report.audio_seconds - 1.0).abs() < 1e-9,
            "audio_seconds should be exactly 1.0, got {}",
            outcome.report.audio_seconds
        );
        // RTF must be a finite non-negative — the identity Mimi decoder
        // path is fast, so we expect RTF << 1.0 on any modern CPU, but
        // we do NOT assert a hard upper bound here (that's the T24
        // deferred always-on gate against a self-hosted CUDA runner —
        // mirrors M2-14 defer, `docs/m2-cuda-rtf-variance-2026-07-08.md`).
        assert!(outcome.report.rtf.is_finite() && outcome.report.rtf >= 0.0);
        assert!(outcome.regression.is_none());
        // The report round-trips through the baseline parser (same shape
        // any future baseline comparison consumes).
        let rtf = report::parse_baseline_rtf(outcome.report.to_json().as_bytes()).unwrap();
        assert!(rtf.is_finite());
    }

    #[test]
    fn bench_cosyvoice2_synthetic_ignores_input_flag_gracefully() {
        // Passing --input for the synthetic path is a no-op (the standalone
        // path does not read the WAV) — the current implementation simply
        // does not consult args.input. Verify the outcome is unchanged.
        let mut wav_path = std::env::temp_dir();
        wav_path.push(format!(
            "vokra-cli-bench-cosyv2-noise-{}.wav",
            std::process::id()
        ));
        wav::write_wav(&wav_path, &vec![0.5f32; 24_000], 24_000).expect("write wav");
        let a = BenchArgs {
            model: None,
            input: Some(wav_path.to_string_lossy().into_owned()),
            text: None,
            style: None,
            iters: 1,
            warmup: 0,
            format: Format::Kv,
            baseline: None,
            backend: BackendKind::Cpu,
            task_hint: Some(TaskHint::Cosyvoice2Synthetic),
            server: None,
            endpoint: "/api/tts".to_owned(),
            concurrent: 1,
            voice: "en_US-libritts-high".to_owned(),
            budget_ms: 75,
            timeout_secs: 30,
        };
        let outcome = execute(&a).expect("cosyvoice2-synthetic bench runs");
        let _ = std::fs::remove_file(&wav_path);
        // Audio duration is still the fixed 1 s target — the input WAV is
        // not consulted for the synthetic path.
        assert!((outcome.report.audio_seconds - 1.0).abs() < 1e-9);
    }

    #[test]
    fn bench_cosyvoice2_synthetic_reports_deterministic_target_seconds() {
        // Two runs with identical BenchArgs report identical target
        // seconds (the deterministic-fixture invariant). Latencies will
        // differ (CPU scheduling), but the audio window is fixed.
        let mk = || BenchArgs {
            model: None,
            input: None,
            text: None,
            style: None,
            iters: 1,
            warmup: 0,
            format: Format::Kv,
            baseline: None,
            backend: BackendKind::Cpu,
            task_hint: Some(TaskHint::Cosyvoice2Synthetic),
            server: None,
            endpoint: "/api/tts".to_owned(),
            concurrent: 1,
            voice: "en_US-libritts-high".to_owned(),
            budget_ms: 75,
            timeout_secs: 30,
        };
        let a1 = execute(&mk()).expect("run 1");
        let a2 = execute(&mk()).expect("run 2");
        assert_eq!(a1.report.audio_seconds, a2.report.audio_seconds);
    }

    #[test]
    fn time_iters_warms_up_then_times_each_iteration() {
        let mut calls = 0u32;
        let samples = time_iters(2, 5, || {
            calls += 1;
            Ok(())
        })
        .expect("no error");
        assert_eq!(calls, 7, "2 warm-up + 5 timed calls");
        assert_eq!(samples.len(), 5);
        assert!(samples.iter().all(|s| s.is_finite() && *s >= 0.0));
    }

    #[test]
    fn time_iters_propagates_errors() {
        let r = time_iters(0, 3, || Err("boom".to_owned()));
        assert_eq!(r.err().unwrap(), "boom");
    }

    /// Builds a bare Whisper GGUF sufficient for the `mel-frontend` task (arch
    /// key + `n_mels` metadata). The bench arm skips weight loading and only
    /// needs `vokra.whisper.n_mels`, so no encoder / decoder tensors are
    /// required — this keeps the test asset synthetic and in-repo (no external
    /// fixture needed for M2-04-T11's routing test).
    fn write_bare_whisper_gguf() -> std::path::PathBuf {
        let mut b = vokra_core::gguf::GgufBuilder::new();
        b.add_string("vokra.model.arch", "whisper");
        b.add_u32("vokra.whisper.n_mels", 80);
        let bytes = b.to_bytes().expect("serialize gguf");
        let mut path = std::env::temp_dir();
        path.push(format!(
            "vokra-cli-bench-melfront-{}.gguf",
            std::process::id()
        ));
        std::fs::write(&path, bytes).unwrap();
        path
    }

    #[test]
    fn bench_mel_frontend_runs_log_mel_and_reports_rtf() {
        // 1 s of silence @ 16 kHz — `log_mel` pads to 30 s internally.
        let mut wav_path = std::env::temp_dir();
        wav_path.push(format!("vokra-cli-bench-mel-{}.wav", std::process::id()));
        wav::write_wav(&wav_path, &vec![0.0f32; 16_000], 16_000).expect("write wav");
        let gguf_path = write_bare_whisper_gguf();

        let a = BenchArgs {
            model: Some(gguf_path.to_string_lossy().into_owned()),
            input: Some(wav_path.to_string_lossy().into_owned()),
            text: None,
            style: None,
            iters: 1,
            warmup: 0,
            format: Format::Kv,
            baseline: None,
            backend: BackendKind::Cpu,
            task_hint: Some(TaskHint::MelFrontend),
            server: None,
            endpoint: "/api/tts".to_owned(),
            concurrent: 1,
            voice: "en_US-libritts-high".to_owned(),
            budget_ms: 75,
            timeout_secs: 30,
        };
        let outcome = execute(&a).expect("bench runs");
        let _ = std::fs::remove_file(&wav_path);
        let _ = std::fs::remove_file(&gguf_path);

        assert_eq!(outcome.report.task, "mel-frontend");
        assert_eq!(outcome.report.iters, 1);
        assert_eq!(outcome.report.latency.count, 1);
        assert!(outcome.report.audio_seconds > 0.0);
        assert!(outcome.report.rtf.is_finite() && outcome.report.rtf >= 0.0);
        assert!(outcome.regression.is_none());
    }

    #[test]
    fn bench_mel_frontend_rejects_non_whisper_arch() {
        // The hint is Whisper-only: pointing it at Silero must fail loudly
        // (FR-EX-08: no silent fallback to VAD).
        let a = BenchArgs {
            model: Some(silero_fixture()),
            input: None,
            text: None,
            style: None,
            iters: 1,
            warmup: 0,
            format: Format::Kv,
            baseline: None,
            backend: BackendKind::Cpu,
            task_hint: Some(TaskHint::MelFrontend),
            server: None,
            endpoint: "/api/tts".to_owned(),
            concurrent: 1,
            voice: "en_US-libritts-high".to_owned(),
            budget_ms: 75,
            timeout_secs: 30,
        };
        let err = match execute(&a) {
            Err(e) => e,
            Ok(_) => panic!("hint on non-whisper arch must be rejected"),
        };
        assert!(
            err.contains("only supported on arch `whisper`"),
            "got: {err}"
        );
    }

    #[test]
    fn bench_vad_over_committed_fixture_reports_well_formed_stats() {
        // Internal-oracle only: committed Silero GGUF + a generated silence WAV.
        let mut wav_path = std::env::temp_dir();
        wav_path.push(format!("vokra-cli-bench-{}.wav", std::process::id()));
        wav::write_wav(&wav_path, &vec![0.0f32; 16_000], 16_000).expect("write wav");

        let a = BenchArgs {
            model: Some(silero_fixture()),
            input: Some(wav_path.to_string_lossy().into_owned()),
            text: None,
            style: None,
            iters: 2,
            warmup: 1,
            format: Format::Kv,
            baseline: None,
            backend: BackendKind::Cpu,
            task_hint: None,
            server: None,
            endpoint: "/api/tts".to_owned(),
            concurrent: 1,
            voice: "en_US-libritts-high".to_owned(),
            budget_ms: 75,
            timeout_secs: 30,
        };
        let outcome = execute(&a).expect("bench runs");
        let _ = std::fs::remove_file(&wav_path);

        assert_eq!(outcome.report.task, "vad");
        assert_eq!(outcome.report.iters, 2);
        assert_eq!(outcome.report.latency.count, 2);
        assert!(outcome.report.audio_seconds > 0.0);
        assert!(outcome.report.rtf.is_finite() && outcome.report.rtf >= 0.0);
        assert!(outcome.regression.is_none());
        // The report serializes to parseable JSON with a readable-back rtf.
        let rtf = report::parse_baseline_rtf(outcome.report.to_json().as_bytes()).unwrap();
        assert!(rtf.is_finite());
    }
}
