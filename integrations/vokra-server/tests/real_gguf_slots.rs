//! cc-40 — real-GGUF slot verification (2026-07-19 M4-residual audit).
//!
//! # Why this suite exists
//!
//! The 2026-07-17 campaign-2 report §(f) recorded honestly that the
//! `voxtral` / `silero-vad` / `kokoro` registry slots were *wired* (commit
//! `fc8feec`) but never *exercised with real weights*. A slot that compiles
//! and a slot that loads a 9 GB checkpoint and answers HTTP are different
//! claims; only the second is worth making. cc-39's new whisper size slots
//! are in the same position on arrival, so they are verified here too.
//!
//! Every leg boots the FULL production startup path
//! ([`spawn_server_for_test_wired`] → `build_service` → `InferenceService::build`)
//! against files under `~/.cache/vokra-eval/gguf/` and asserts at the HTTP
//! layer — not against a mock, and not against the loader in isolation.
//!
//! # Running it
//!
//! Every leg is env-gated and SKIPS (printing what to set) when its GGUF is
//! not supplied, so the default `cargo test` stays hermetic. The required
//! minimum for any wired boot is whisper-base + a piper voice:
//!
//! ```text
//! G=~/.cache/vokra-eval/gguf
//! VOKRA_WHISPER_GGUF=$G/whisper-base.gguf \
//! VOKRA_PIPER_GGUF=$G/piper-plus-css10-ja-6lang-neutralspk.gguf \
//! VOKRA_KOKORO_GGUF=$G/kokoro-82m.gguf \
//! VOKRA_SILERO_GGUF=$G/silero-vad-v5-master.gguf \
//! VOKRA_WHISPER_SMALL_GGUF=$G/whisper-small.gguf \
//!   cargo test --release --test real_gguf_slots -- --nocapture
//! ```
//!
//! The env var names match the existing suites (`piper_http_compat.rs`,
//! `tts_g2p_injection.rs`) so one exported block drives them all.
//!
//! # Honest scope, stated up front
//!
//! * **kokoro is advertise + explicit 501 only.** `KokoroTts::synthesize` is
//!   genuinely unimplemented (M2-07 vocoder bridge deferred). This suite
//!   verifies exactly that contract — the GGUF loads, the id appears in the
//!   catalogue, and BOTH TTS routes answer 501. It does not synthesize, and
//!   a passing run must never be read as "kokoro TTS works".
//! * **silero is load + boot only, because nothing consumes it.**
//!   `InferenceService.vad` is populated and then never read by any request
//!   path — verified by grep across `src/` on 2026-07-19: the field's
//!   `#[allow(dead_code)] // consumed by T15 (Wyoming ASR chunk framing)`
//!   comment does not match the code, as `api/wyoming.rs` contains no `vad`
//!   reference at all. So the honest claim this suite can make is "a real
//!   Silero v5 GGUF passes the loader and the compliance gate, and the server
//!   boots and serves with it configured" — which is a genuine gate (a
//!   corrupt file fails startup hard) but is NOT end-to-end VAD behaviour.
//!   Asserting anything about chunk boundaries here would be fabricated.
//! * **voxtral needs ~9 GB of RAM to load.** `GgufFile::open` is
//!   `std::fs::read` — a full in-memory read, no mmap (`vokra-core`
//!   `gguf/reader.rs:112`) — so the 8.72 GiB `voxtral-mini-3b-bf16-fs.gguf`
//!   needs that much resident before the engine is even constructed. The leg
//!   below measures free memory first and skips with the measurement rather
//!   than driving the machine into swap.

mod support;

use std::path::PathBuf;
use std::time::Duration;

use vokra_server::config::Config;
use vokra_server::server::spawn_server_for_test_wired;

// ---------------------------------------------------------------------------
// Env gates
// ---------------------------------------------------------------------------

fn gguf(var: &str) -> Option<PathBuf> {
    let p = std::env::var_os(var).map(PathBuf::from)?;
    assert!(
        p.exists(),
        "real_gguf_slots: {var} points at a missing file: {}",
        p.display()
    );
    Some(p)
}

/// The two GGUFs every wired boot requires (`build_service`'s minimum).
fn base_pair() -> Option<(PathBuf, PathBuf)> {
    Some((gguf("VOKRA_WHISPER_GGUF")?, gguf("VOKRA_PIPER_GGUF")?))
}

fn skip(what: &str, vars: &str) {
    eprintln!("real_gguf_slots: SKIPPING {what} — set {vars} to run it.");
}

fn base_config(whisper: PathBuf, piper: PathBuf) -> Config {
    Config {
        http_bind: "127.0.0.1:0".parse().unwrap(),
        wyoming_bind: "127.0.0.1:0".parse().unwrap(),
        config_file: None,
        whisper_base_gguf: Some(whisper),
        piper_plus_gguf: Some(piper),
        ..Config::default()
    }
}

/// Bytes of physical memory currently available (free + inactive +
/// speculative), via `vm_stat`. Used to decide whether the ~9 GB voxtral load
/// can be attempted at all. Returns `None` if the platform tool is absent —
/// in which case the caller skips rather than guesses.
#[cfg(target_os = "macos")]
fn available_memory_bytes() -> Option<u64> {
    let out = std::process::Command::new("vm_stat").output().ok()?;
    let text = String::from_utf8(out.stdout).ok()?;
    let mut page_size = 4096u64;
    let mut pages = 0u64;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("Mach Virtual Memory Statistics:") {
            // "(page size of 16384 bytes)"
            if let Some(n) = rest.split_whitespace().nth(4).and_then(|s| s.parse().ok()) {
                page_size = n;
            }
        }
        for key in ["Pages free:", "Pages inactive:", "Pages speculative:"] {
            if let Some(rest) = line.strip_prefix(key) {
                if let Ok(n) = rest.trim().trim_end_matches('.').parse::<u64>() {
                    pages += n;
                }
            }
        }
    }
    Some(pages * page_size)
}

#[cfg(not(target_os = "macos"))]
fn available_memory_bytes() -> Option<u64> {
    None
}

// ---------------------------------------------------------------------------
// kokoro — advertise + explicit 501 (NOT synthesis)
// ---------------------------------------------------------------------------

/// The kokoro slot loads a real 327 MB GGUF, the id is advertised in
/// `GET /v1/models`, and BOTH TTS routes answer an explicit 501.
///
/// This is the complete kokoro contract as of M2-07: advertised, never
/// synthesizing, never substituting piper. A 200 here would be a bug.
#[tokio::test]
async fn kokoro_slot_advertises_and_501s_on_both_tts_routes() {
    let (Some((whisper, piper)), Some(kokoro)) = (base_pair(), gguf("VOKRA_KOKORO_GGUF")) else {
        skip(
            "kokoro slot",
            "VOKRA_WHISPER_GGUF + VOKRA_PIPER_GGUF + VOKRA_KOKORO_GGUF",
        );
        return;
    };

    let cfg = Config {
        kokoro_gguf: Some(kokoro),
        ..base_config(whisper, piper)
    };
    let (handles, trigger) = spawn_server_for_test_wired(cfg)
        .await
        .expect("kokoro GGUF must load and the server must boot");

    // 1) Advertised in the catalogue.
    let (status, body) = support::http_get(handles.http_actual, "/v1/models")
        .await
        .expect("GET /v1/models");
    assert_eq!(status, 200);
    let catalogue: serde_json::Value = serde_json::from_slice(&body).expect("models JSON");
    let ids: Vec<String> = catalogue["data"]
        .as_array()
        .expect("data array")
        .iter()
        .map(|m| m["id"].as_str().unwrap_or_default().to_owned())
        .collect();
    assert!(
        ids.iter().any(|id| id == "kokoro"),
        "a configured kokoro must be advertised; got {ids:?}"
    );
    eprintln!("cc40 kokoro: GET /v1/models ids = {ids:?}");

    // 2) piper-plus HTTP route → 501.
    let body =
        serde_json::json!({ "text": "1 30 2", "voice": "default", "model": "kokoro" }).to_string();
    let (status, _h, resp) =
        support::http_post_json_with_head(handles.http_actual, "/api/tts", body.as_bytes())
            .await
            .expect("POST /api/tts (kokoro)");
    assert_eq!(
        status,
        501,
        "kokoro synthesis is deferred (M2-07); body: {}",
        String::from_utf8_lossy(&resp)
    );
    eprintln!(
        "cc40 kokoro: POST /api/tts -> 501 {}",
        String::from_utf8_lossy(&resp)
    );

    // 3) OpenAI speech route (cc-38) → the SAME 501, via the shared mapper.
    let body =
        serde_json::json!({ "model": "kokoro", "input": "1 30 2", "voice": "default" }).to_string();
    let (status, _h, resp) =
        support::http_post_json_with_head(handles.http_actual, "/v1/audio/speech", body.as_bytes())
            .await
            .expect("POST /v1/audio/speech (kokoro)");
    assert_eq!(
        status,
        501,
        "kokoro must 501 on the OpenAI surface too; body: {}",
        String::from_utf8_lossy(&resp)
    );
    let err: serde_json::Value = serde_json::from_slice(&resp).expect("error envelope");
    assert_eq!(err["error"]["type"], "not_implemented");
    eprintln!(
        "cc40 kokoro: POST /v1/audio/speech -> 501 {}",
        String::from_utf8_lossy(&resp)
    );

    trigger.trigger();
    tokio::time::sleep(Duration::from_millis(20)).await;
}

// ---------------------------------------------------------------------------
// silero — load + boot (the honest limit; see module docs)
// ---------------------------------------------------------------------------

/// A real Silero VAD v5 GGUF passes the loader + compliance gate and the
/// server boots and serves with it configured.
///
/// Deliberately asserts nothing about VAD behaviour: `InferenceService.vad`
/// is never read by any request path (module docs). What this DOES prove is
/// that the slot is a real gate — the same boot with a corrupt file fails,
/// which the negative leg below demonstrates.
#[tokio::test]
async fn silero_slot_loads_and_server_serves() {
    let (Some((whisper, piper)), Some(silero)) = (base_pair(), gguf("VOKRA_SILERO_GGUF")) else {
        skip(
            "silero slot",
            "VOKRA_WHISPER_GGUF + VOKRA_PIPER_GGUF + VOKRA_SILERO_GGUF",
        );
        return;
    };

    let cfg = Config {
        silero_vad_gguf: Some(silero),
        ..base_config(whisper, piper)
    };
    let (handles, trigger) = spawn_server_for_test_wired(cfg)
        .await
        .expect("silero GGUF must load and the server must boot");

    let (status, _b) = support::http_get(handles.http_actual, "/health")
        .await
        .expect("GET /health");
    assert_eq!(status, 200, "server must serve with a VAD configured");

    // The VAD is not advertised as a model (it is not an ASR/TTS engine), so
    // the catalogue must NOT grow an entry for it — an id clients cannot use
    // would be a fabricated capability.
    let (status, body) = support::http_get(handles.http_actual, "/v1/models")
        .await
        .expect("GET /v1/models");
    assert_eq!(status, 200);
    let catalogue: serde_json::Value = serde_json::from_slice(&body).expect("models JSON");
    let ids: Vec<String> = catalogue["data"]
        .as_array()
        .expect("data array")
        .iter()
        .map(|m| m["id"].as_str().unwrap_or_default().to_owned())
        .collect();
    assert!(
        !ids.iter()
            .any(|id| id.contains("silero") || id.contains("vad")),
        "the VAD is not a callable model and must not be advertised; got {ids:?}"
    );
    eprintln!("cc40 silero: loaded + boot OK; catalogue = {ids:?} (no VAD entry, as intended)");

    trigger.trigger();
    tokio::time::sleep(Duration::from_millis(20)).await;
}

/// The counterpart that gives the leg above its meaning: a garbage file in
/// the silero slot must fail the boot HARD, never be skipped (FR-EX-08). Runs
/// without any real GGUF beyond the required base pair.
#[tokio::test]
async fn silero_slot_rejects_a_corrupt_file_at_startup() {
    let Some((whisper, piper)) = base_pair() else {
        skip(
            "silero corrupt-file leg",
            "VOKRA_WHISPER_GGUF + VOKRA_PIPER_GGUF",
        );
        return;
    };

    let bogus =
        std::env::temp_dir().join(format!("vokra-cc40-not-a-gguf-{}.bin", std::process::id()));
    std::fs::write(&bogus, b"definitely not a GGUF header").expect("write bogus file");

    let cfg = Config {
        silero_vad_gguf: Some(bogus.clone()),
        ..base_config(whisper, piper)
    };
    let err = spawn_server_for_test_wired(cfg)
        .await
        .err()
        .expect("a corrupt VAD GGUF must fail the boot, not be silently skipped");
    let msg = err.to_string();
    assert!(
        msg.contains("silero-vad"),
        "the startup error must name the failing slot; got: {msg}"
    );
    eprintln!("cc40 silero: corrupt file -> hard startup error: {msg}");

    let _ = std::fs::remove_file(&bogus);
}

// ---------------------------------------------------------------------------
// cc-39 whisper sizes — routed, advertised, never substituted
// ---------------------------------------------------------------------------

/// Each configured size is advertised and transcribes; every size NOT
/// configured on that same boot is a 404.
///
/// One boot per size (rather than all four at once) keeps peak RSS to
/// base + piper + the size under test — `GgufFile::open` reads whole files
/// into memory, so loading small+medium+turbo together would cost ~6 GB for
/// no extra coverage.
///
/// The second half is the cc-39 red line: an unconfigured size must NOT fall
/// through to base. A 200 for a size this boot never loaded would prove a
/// silent substitution, which is exactly the failure the slot exists to
/// prevent.
#[tokio::test]
async fn whisper_size_slots_route_and_unconfigured_sizes_404() {
    let Some((whisper, piper)) = base_pair() else {
        skip(
            "whisper size slots",
            "VOKRA_WHISPER_GGUF + VOKRA_PIPER_GGUF",
        );
        return;
    };
    let Some(wav) = std::env::var_os("VOKRA_TEST_WAV").map(PathBuf::from) else {
        skip("whisper size slots", "VOKRA_TEST_WAV=<16k mono wav>");
        return;
    };
    let wav_bytes = std::fs::read(&wav).expect("read test wav");

    // (env var, model id, Config field setter) for each cc-39 size.
    #[allow(clippy::type_complexity)]
    let sizes: [(&str, &str, fn(&mut Config, PathBuf)); 3] = [
        ("VOKRA_WHISPER_SMALL_GGUF", "whisper-small", |c, p| {
            c.whisper_small_gguf = Some(p)
        }),
        ("VOKRA_WHISPER_MEDIUM_GGUF", "whisper-medium", |c, p| {
            c.whisper_medium_gguf = Some(p)
        }),
        ("VOKRA_WHISPER_TURBO_GGUF", "whisper-turbo", |c, p| {
            c.whisper_turbo_gguf = Some(p)
        }),
    ];

    let mut ran = 0usize;
    for (var, model_id, set) in sizes {
        let Some(path) = gguf(var) else {
            skip(&format!("whisper size `{model_id}`"), var);
            continue;
        };
        ran += 1;

        let mut cfg = base_config(whisper.clone(), piper.clone());
        set(&mut cfg, path);
        let (handles, trigger) = spawn_server_for_test_wired(cfg)
            .await
            .unwrap_or_else(|e| panic!("{model_id} GGUF must load and the server must boot: {e}"));

        // Advertised — and ONLY this size is.
        let ids = catalogue_ids(handles.http_actual).await;
        assert!(
            ids.iter().any(|id| id == model_id),
            "a configured `{model_id}` must be advertised; got {ids:?}"
        );
        eprintln!("cc40/cc39 [{model_id}]: catalogue = {ids:?}");

        // Transcribes.
        let boundary = "----vokraCc40Boundary";
        let body = build_multipart(boundary, &wav_bytes, model_id);
        let (status, resp) = support::http_post_multipart_timeout(
            handles.http_actual,
            "/v1/audio/transcriptions",
            boundary,
            &body,
            ASR_READ_TIMEOUT,
        )
        .await
        .unwrap_or_else(|e| panic!("POST /v1/audio/transcriptions ({model_id}): {e}"));
        assert_eq!(
            status,
            200,
            "{model_id} must transcribe; body: {}",
            String::from_utf8_lossy(&resp)
        );
        let out: serde_json::Value = serde_json::from_slice(&resp).expect("transcription JSON");
        let text = out["text"].as_str().unwrap_or_default().to_owned();
        assert!(
            !text.trim().is_empty(),
            "{model_id} transcription must not be empty"
        );
        eprintln!("cc40/cc39 [{model_id}] transcript: {text:?}");

        // RED LINE: every OTHER size is unconfigured on this boot → 404.
        for (_, absent, _) in sizes.iter().filter(|(_, id, _)| *id != model_id) {
            assert!(
                !ids.iter().any(|id| id == absent),
                "unconfigured `{absent}` must not be advertised; got {ids:?}"
            );
            let body = build_multipart(boundary, &wav_bytes, absent);
            let (status, resp) = support::http_post_multipart_timeout(
                handles.http_actual,
                "/v1/audio/transcriptions",
                boundary,
                &body,
                ASR_READ_TIMEOUT,
            )
            .await
            .unwrap_or_else(|e| panic!("POST for {absent}: {e}"));
            assert_eq!(
                status,
                404,
                "unconfigured `{absent}` must 404, NEVER be served by `{model_id}` or base; \
                 body: {}",
                String::from_utf8_lossy(&resp)
            );
        }
        eprintln!("cc40/cc39 [{model_id}]: other sizes 404 as required (no silent substitution)");

        trigger.trigger();
        tokio::time::sleep(Duration::from_millis(20)).await;
    }

    if ran == 0 {
        skip(
            "all whisper size slots",
            "at least one of VOKRA_WHISPER_{SMALL,MEDIUM,TURBO}_GGUF",
        );
    }
}

/// `GET /v1/models` → the list of advertised ids.
async fn catalogue_ids(addr: std::net::SocketAddr) -> Vec<String> {
    let (status, body) = support::http_get(addr, "/v1/models")
        .await
        .expect("GET /v1/models");
    assert_eq!(status, 200);
    let catalogue: serde_json::Value = serde_json::from_slice(&body).expect("models JSON");
    catalogue["data"]
        .as_array()
        .expect("data array")
        .iter()
        .map(|m| m["id"].as_str().unwrap_or_default().to_owned())
        .collect()
}

// ---------------------------------------------------------------------------
// cc-38 — OpenAI /v1/audio/speech against the real piper voice + real G2P
// ---------------------------------------------------------------------------

/// The cc-38 route synthesizes real audio from plain text on a `--piper-g2p`
/// boot, and holds its four explicit-error contracts on the same server.
///
/// This is the leg that makes cc-38 more than a schema: the bytes come from
/// the real MB-iSTFT-VITS2 voice through the real 8-language G2P.
#[tokio::test]
async fn openai_speech_route_synthesizes_with_real_voice_and_g2p() {
    let Some((whisper, piper)) = base_pair() else {
        skip(
            "OpenAI /v1/audio/speech e2e",
            "VOKRA_WHISPER_GGUF + VOKRA_PIPER_GGUF",
        );
        return;
    };

    let cfg = Config {
        piper_g2p: Some(true),
        ..base_config(whisper, piper)
    };
    let (handles, trigger) = spawn_server_for_test_wired(cfg)
        .await
        .expect("wired boot with --piper-g2p");

    // 1) Plain text → real WAV.
    let body = serde_json::json!({
        "model": "tts-1",
        "input": "Hello from Vokra.",
        "voice": "default",
    })
    .to_string();
    let (status, head, wav) = support::http_post_json_with_head_timeout(
        handles.http_actual,
        "/v1/audio/speech",
        body.as_bytes(),
        TTS_READ_TIMEOUT,
    )
    .await
    .expect("POST /v1/audio/speech");
    assert_eq!(
        status,
        200,
        "plain text must synthesize under --piper-g2p; body: {}",
        String::from_utf8_lossy(&wav[..wav.len().min(256)])
    );
    assert!(
        head.to_ascii_lowercase()
            .contains("content-type: audio/wav"),
        "the response must declare audio/wav; headers:\n{head}"
    );
    assert_eq!(&wav[0..4], b"RIFF", "body must be a RIFF container");
    let sr = u32::from_le_bytes([wav[24], wav[25], wav[26], wav[27]]);
    let n_samples = (wav.len() - 44) / 2;
    let dur = n_samples as f64 / f64::from(sr);
    let peak = wav[44..]
        .chunks_exact(2)
        .map(|c| i32::from(i16::from_le_bytes([c[0], c[1]])).abs())
        .max()
        .unwrap_or(0);
    assert!(
        (0.2..=20.0).contains(&dur),
        "duration implausible: {dur:.3}s ({n_samples} samples @ {sr} Hz)"
    );
    assert!(peak > 500, "audio must be non-silent (peak |i16| = {peak})");
    eprintln!("cc40/cc38 speech: 200 audio/wav, {dur:.3}s @ {sr} Hz, peak |i16| = {peak}");

    // 2) The four explicit-error contracts, on this same live server.
    for (label, payload, want) in [
        (
            "compressed format",
            serde_json::json!({"model":"tts-1","input":"hi","voice":"default","response_format":"mp3"}),
            501u16,
        ),
        (
            "non-unit speed",
            serde_json::json!({"model":"tts-1","input":"hi","voice":"default","speed":1.5}),
            501,
        ),
        (
            "OpenAI stock voice",
            serde_json::json!({"model":"tts-1","input":"hi","voice":"alloy"}),
            404,
        ),
        (
            "unknown field",
            serde_json::json!({"model":"tts-1","input":"hi","voice":"default","instructions":"cheerful"}),
            400,
        ),
    ] {
        let (status, _h, resp) = support::http_post_json_with_head_timeout(
            handles.http_actual,
            "/v1/audio/speech",
            payload.to_string().as_bytes(),
            TTS_READ_TIMEOUT,
        )
        .await
        .unwrap_or_else(|e| panic!("POST /v1/audio/speech ({label}): {e}"));
        assert_eq!(
            status,
            want,
            "{label} must be {want}; got {status}, body: {}",
            String::from_utf8_lossy(&resp)
        );
        eprintln!(
            "cc40/cc38 speech [{label}] -> {status} {}",
            String::from_utf8_lossy(&resp[..resp.len().min(200)])
        );
    }

    trigger.trigger();
    tokio::time::sleep(Duration::from_millis(20)).await;
}

// ---------------------------------------------------------------------------
// voxtral — memory-gated
// ---------------------------------------------------------------------------

/// Minimum free memory before attempting the voxtral load, in bytes.
///
/// **Was 10 GiB**, on the premise that `GgufFile::open` reads the whole file
/// into a `Vec<u8>` and `VoxtralAsr::from_gguf` then widens the entire
/// `language_model` group to owned f32 — 8.71 GiB resident plus 14.95 GiB
/// widened at the real 3B shape. On this 16 GiB machine that floor was never
/// met, so the leg had never actually run.
///
/// The registry now maps the file and binds the decoder blocks lazily
/// (`map_gguf` + `VoxtralAsr::from_gguf_mapped`), which changes the arithmetic
/// entirely. Measured end-to-end on the real 8.71 GiB checkpoint (M1 iMac,
/// `vokra-cli run --model voxtral-mini-3b-bf16-fs.gguf`, full 30 s transcribe):
///
/// ```text
///   max RSS         5,258,625,024 bytes  (4.90 GiB)
///   peak footprint  7,043,400,768 bytes  (6.56 GiB)
///   swaps           0
/// ```
///
/// The floor is set to **8 GiB**: above the measured 6.56 GiB peak with room
/// for a concurrently-served whisper + piper pair (this harness boots all three
/// slots), and low enough that an ordinary idle 16 GiB host clears it. It is
/// deliberately *not* tightened to the measured value — a floor that only just
/// fits the best observed case turns a memory-pressure flake into a test
/// failure.
///
/// If the mapped bind is ever unavailable (a quantized GGUF), the registry
/// falls back to the resident loader with a printed note, and this floor is
/// then too low — that path prints its own warning rather than pretending.
const VOXTRAL_MIN_FREE_BYTES: u64 = 8 * 1024 * 1024 * 1024;

/// Read deadline for a real-weight transcription. The shared default (5 s)
/// is tuned for the hermetic suites; a real Whisper size on ~11 s of audio,
/// on a CPU shared with sibling build jobs, legitimately exceeds it.
const ASR_READ_TIMEOUT: Duration = Duration::from_secs(300);

/// Read deadline for a real-weight synthesis (same rationale as
/// [`ASR_READ_TIMEOUT`]; TTS is the faster of the two but shares the host).
const TTS_READ_TIMEOUT: Duration = Duration::from_secs(120);

/// Boot with the real Voxtral checkpoint and verify the aliases are
/// advertised and routable.
///
/// Skips — with the measurement that caused the skip — when the machine does
/// not have the memory. Attempting it anyway would swap-thrash a shared
/// 16 GB host, and a test that OOM-kills its runner reports nothing useful.
#[tokio::test]
async fn voxtral_slot_advertises_and_routes() {
    let (Some((whisper, piper)), Some(voxtral)) = (base_pair(), gguf("VOKRA_VOXTRAL_GGUF")) else {
        skip(
            "voxtral slot",
            "VOKRA_WHISPER_GGUF + VOKRA_PIPER_GGUF + VOKRA_VOXTRAL_GGUF",
        );
        return;
    };

    let size = std::fs::metadata(&voxtral).map(|m| m.len()).unwrap_or(0);
    match available_memory_bytes() {
        Some(avail) if avail >= VOXTRAL_MIN_FREE_BYTES => {
            eprintln!(
                "voxtral: proceeding — {:.2} GiB available >= {:.2} GiB floor (checkpoint \
                 {:.2} GiB, bound through the mmap + mapped-block path)",
                avail as f64 / 1073741824.0,
                VOXTRAL_MIN_FREE_BYTES as f64 / 1073741824.0,
                size as f64 / 1073741824.0,
            );
        }
        Some(avail) => {
            eprintln!(
                "real_gguf_slots: SKIPPING voxtral slot — only {:.2} GiB available, need >= \
                 {:.2} GiB for the {:.2} GiB checkpoint (mapped bind: measured peak 6.56 GiB \
                 end-to-end, floor set above it with headroom for the co-resident whisper + \
                 piper slots). Re-run on an idle host.",
                avail as f64 / 1073741824.0,
                VOXTRAL_MIN_FREE_BYTES as f64 / 1073741824.0,
                size as f64 / 1073741824.0,
            );
            return;
        }
        None => {
            eprintln!(
                "real_gguf_slots: SKIPPING voxtral slot — cannot measure available memory on \
                 this platform, and guessing is how a shared host ends up in swap."
            );
            return;
        }
    }

    let cfg = Config {
        voxtral_gguf: Some(voxtral),
        ..base_config(whisper, piper)
    };
    let (handles, trigger) = spawn_server_for_test_wired(cfg)
        .await
        .expect("voxtral GGUF must load and the server must boot");

    let (status, body) = support::http_get(handles.http_actual, "/v1/models")
        .await
        .expect("GET /v1/models");
    assert_eq!(status, 200);
    let catalogue: serde_json::Value = serde_json::from_slice(&body).expect("models JSON");
    let ids: Vec<String> = catalogue["data"]
        .as_array()
        .expect("data array")
        .iter()
        .map(|m| m["id"].as_str().unwrap_or_default().to_owned())
        .collect();
    for alias in ["voxtral", "voxtral-mini-3b", "voxtral-small-24b"] {
        assert!(
            ids.iter().any(|id| id == alias),
            "configured voxtral must advertise `{alias}`; got {ids:?}"
        );
    }
    eprintln!("cc40 voxtral: catalogue = {ids:?}");

    trigger.trigger();
    tokio::time::sleep(Duration::from_millis(20)).await;
}

// ---------------------------------------------------------------------------
// Shared multipart builder (mirrors openai_compat.rs — kept local so the two
// suites stay independent).
// ---------------------------------------------------------------------------

fn build_multipart(boundary: &str, wav: &[u8], model: &str) -> Vec<u8> {
    let mut b = Vec::new();
    b.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
    b.extend_from_slice(
        b"Content-Disposition: form-data; name=\"file\"; filename=\"audio.wav\"\r\n",
    );
    b.extend_from_slice(b"Content-Type: audio/wav\r\n\r\n");
    b.extend_from_slice(wav);
    b.extend_from_slice(b"\r\n");
    b.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
    b.extend_from_slice(b"Content-Disposition: form-data; name=\"model\"\r\n\r\n");
    b.extend_from_slice(model.as_bytes());
    b.extend_from_slice(b"\r\n");
    b.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
    b
}

// ---------------------------------------------------------------------------
// cc-30 — the backend override, actually taking effect
// ---------------------------------------------------------------------------

/// A `--backend metal` boot really runs Whisper on the GPU and produces the
/// same transcript as the CPU boot.
///
/// This is the leg that makes cc-30 more than a config surface: it proves the
/// selected backend reaches the engines. Compiled only with `--features
/// metal` (the passthrough feature added for cc-30) and skipped elsewhere —
/// on a CPU-only build the interesting assertion is the startup REJECTION,
/// which `service::size_slots_and_backends::cc30_uncompiled_backend_error_names_its_cargo_feature`
/// covers.
///
/// Equality with the CPU transcript is asserted, not RTF: this suite is about
/// correct wiring, and the M1-Metal-vs-CPU byte-identity of Whisper is
/// already established (campaign-2 `c68038f`).
#[cfg(feature = "metal")]
#[tokio::test]
async fn cc30_metal_backend_override_reaches_the_engine() {
    use vokra_core::BackendKind;

    let Some((whisper, piper)) = base_pair() else {
        skip(
            "cc-30 metal backend leg",
            "VOKRA_WHISPER_GGUF + VOKRA_PIPER_GGUF",
        );
        return;
    };
    let Some(wav) = std::env::var_os("VOKRA_TEST_WAV").map(PathBuf::from) else {
        skip("cc-30 metal backend leg", "VOKRA_TEST_WAV=<16k mono wav>");
        return;
    };
    let wav_bytes = std::fs::read(&wav).expect("read test wav");
    let boundary = "----vokraCc30Boundary";

    // Transcribe the same audio on each backend and compare.
    let mut transcripts = Vec::new();
    for backend in [BackendKind::Cpu, BackendKind::Metal] {
        let cfg = Config {
            backend: Some(backend),
            ..base_config(whisper.clone(), piper.clone())
        };
        let (handles, trigger) = spawn_server_for_test_wired(cfg)
            .await
            .unwrap_or_else(|e| panic!("--backend {backend:?} boot must succeed: {e}"));

        let body = build_multipart(boundary, &wav_bytes, "whisper-base");
        let (status, resp) = support::http_post_multipart_timeout(
            handles.http_actual,
            "/v1/audio/transcriptions",
            boundary,
            &body,
            ASR_READ_TIMEOUT,
        )
        .await
        .unwrap_or_else(|e| panic!("POST on {backend:?}: {e}"));
        assert_eq!(
            status,
            200,
            "{backend:?} must transcribe; body: {}",
            String::from_utf8_lossy(&resp)
        );
        let out: serde_json::Value = serde_json::from_slice(&resp).expect("transcription JSON");
        let text = out["text"].as_str().unwrap_or_default().to_owned();
        assert!(!text.trim().is_empty(), "{backend:?} transcript was empty");
        eprintln!("cc40/cc30 [{backend:?}] transcript: {text:?}");
        transcripts.push((backend, text));

        trigger.trigger();
        tokio::time::sleep(Duration::from_millis(20)).await;
    }

    assert_eq!(
        transcripts[0].1, transcripts[1].1,
        "Metal and CPU must agree on the transcript (cpu={:?}, metal={:?})",
        transcripts[0].1, transcripts[1].1
    );
    eprintln!("cc40/cc30: --backend metal reached the engine; transcript matches CPU");
}
