//! Campaign-2 P1 #3 acceptance — real 8-language G2P injection + `/api/tts`.
//!
//! Two layers, both env-gated on real weights (clean, non-silent skip when
//! absent — never a fabricated pass):
//!
//! 1. **Direct G2P** (`VOKRA_PIPER_GGUF` only): builds
//!    [`vokra_piper_g2p::PiperPlusG2p`] from the voice's own GGUF metadata and
//!    checks JA + EN text → phoneme-id conversion (framing, language ids,
//!    prosody, determinism). When the voice is the campaign css10-ja-6lang
//!    checkpoint, the EN ids are additionally byte-compared against the
//!    campaign-1 real-G2P dump (`out/tts-piper/en-g2p-dump.txt`, reproduced as
//!    a constant below) — the same fixture the 2026-07-17 server-real leg used
//!    for its raw-id probe.
//! 2. **Wired server e2e** (`VOKRA_PIPER_GGUF` + `VOKRA_WHISPER_GGUF`): boots
//!    the FULL production startup path (`spawn_server_for_test_wired`) twice —
//!    once with `--piper-g2p` (plain JA text → 200 `audio/wav`, non-silent,
//!    deterministic) and once with the default passthrough (raw phoneme ids →
//!    200 unchanged; plain text → explicit 400 naming the passthrough,
//!    FR-EX-08).
//!
//! Suggested invocation on the eval cache:
//!
//! ```sh
//! VOKRA_PIPER_GGUF=~/.cache/vokra-eval/gguf/piper-plus-css10-ja-6lang-neutralspk.gguf \
//! VOKRA_WHISPER_GGUF=~/.cache/vokra-eval/gguf/whisper-base.gguf \
//! cargo test --release --manifest-path integrations/vokra-server/Cargo.toml \
//!   --test tts_g2p_injection
//! ```

use std::net::SocketAddr;
use std::path::PathBuf;
use std::time::Duration;

use vokra_models::piper_plus::PiperPlusTts;
use vokra_piper_g2p::PiperPlusG2p;
use vokra_piper_plus::Phonemizer as _;
use vokra_server::{Config, spawn_server_for_test_wired};

// ---------------------------------------------------------------------------
// Gating
// ---------------------------------------------------------------------------

fn piper_gguf() -> Option<PathBuf> {
    std::env::var_os("VOKRA_PIPER_GGUF").map(PathBuf::from)
}

fn whisper_gguf() -> Option<PathBuf> {
    std::env::var_os("VOKRA_WHISPER_GGUF").map(PathBuf::from)
}

// ---------------------------------------------------------------------------
// Campaign-1 golden (css10-ja-6lang voice): EN sentence → 96 framed ids,
// lid = 1 ("en" is index 1 in that voice's language_codes), zero prosody.
// Source: ~/.cache/vokra-eval/out/tts-piper/en-g2p-dump.txt (verified live by
// the campaign raw-id Wyoming probe: 26368 samples @ 22050 Hz, peak 7419).
// ---------------------------------------------------------------------------

const CAMPAIGN_EN_SENTENCE: &str = "This is a speech synthesis test of the Vokra runtime.";

const CAMPAIGN_EN_IDS: &[i64] = &[
    1, 0, 0, 54, 0, 11, 0, 48, 0, 0, 11, 0, 48, 0, 0, 10, 0, 0, 48, 0, 42, 0, 13, 0, 13, 0, 0, 54,
    0, 0, 48, 0, 64, 0, 57, 0, 38, 0, 54, 0, 13, 0, 48, 0, 11, 0, 48, 0, 0, 38, 0, 13, 0, 48, 0,
    38, 0, 0, 14, 0, 53, 0, 0, 38, 0, 54, 0, 13, 0, 0, 0, 14, 0, 32, 0, 61, 0, 10, 0, 0, 61, 0, 12,
    0, 57, 0, 38, 0, 11, 0, 59, 0, 13, 0, 0, 2,
];

/// The e2e greeting (task: "a short JA greeting").
const JA_GREETING: &str = "こんにちは。";

// ---------------------------------------------------------------------------
// Layer 1 — direct G2P: text → ids on the real crate, JA + EN.
// ---------------------------------------------------------------------------

#[test]
fn g2p_real_ja_and_en_text_to_ids() {
    let Some(gguf) = piper_gguf() else {
        eprintln!(
            "tts_g2p_injection: skipping — set VOKRA_PIPER_GGUF=<voice.gguf> to run the \
             real-G2P text→ids checks."
        );
        return;
    };
    let voice = PiperPlusTts::from_path(&gguf).expect("load piper voice");
    let cfg = voice.config().clone();
    let g2p = PiperPlusG2p::from_voice(&voice).expect("build real G2P from voice");

    let id_of = |sym: &str| -> i64 {
        cfg.phoneme_symbols
            .iter()
            .position(|s| s == sym)
            .unwrap_or_else(|| panic!("voice phoneme table lacks framing symbol {sym:?}"))
            as i64
    };
    let (bos, eos, pad) = (id_of("^"), id_of("$"), id_of("_"));

    // ---- JA ----
    let ja = g2p.phonemize_full(JA_GREETING).expect("JA phonemize");
    assert!(
        ja.ids.len() > 4,
        "JA greeting must produce a non-trivial id sequence, got {:?}",
        ja.ids
    );
    assert_eq!(ja.ids.first(), Some(&bos), "JA ids must start with BOS `^`");
    assert_eq!(ja.ids.last(), Some(&eos), "JA ids must end with EOS `$`");
    assert!(
        ja.ids.contains(&pad),
        "JA ids must carry interleaved `_` pads (piper framing)"
    );
    if let Some(want_lid) = cfg.language_id("ja") {
        assert_eq!(ja.lid, want_lid, "JA text must detect the voice's ja lid");
    } else {
        eprintln!("tts_g2p_injection: voice has no `ja` — skipping JA lid assert");
    }
    assert_eq!(
        ja.prosody.len(),
        ja.ids.len(),
        "prosody triples must align 1:1 with ids"
    );
    assert!(
        ja.prosody.iter().any(|p| *p != [0, 0, 0]),
        "JA must carry non-zero accent prosody (A1,A2,A3) triples"
    );
    // Determinism: the G2P is a pure function of the text.
    let ja2 = g2p.phonemize_full(JA_GREETING).expect("JA phonemize again");
    assert_eq!(ja.ids, ja2.ids, "JA ids must be deterministic");
    assert_eq!(ja.prosody, ja2.prosody, "JA prosody must be deterministic");

    // ---- EN ----
    let en = g2p
        .phonemize_full(CAMPAIGN_EN_SENTENCE)
        .expect("EN phonemize");
    assert_eq!(en.ids.first(), Some(&bos), "EN ids must start with BOS `^`");
    assert_eq!(en.ids.last(), Some(&eos), "EN ids must end with EOS `$`");
    if let Some(want_lid) = cfg.language_id("en") {
        assert_eq!(en.lid, want_lid, "EN text must detect the voice's en lid");
    } else {
        eprintln!("tts_g2p_injection: voice has no `en` — skipping EN lid assert");
    }
    assert!(
        en.prosody.iter().all(|p| *p == [0, 0, 0]),
        "EN carries no JA accent prosody — all triples must be zero"
    );
    let en2 = g2p
        .phonemize_full(CAMPAIGN_EN_SENTENCE)
        .expect("EN phonemize again");
    assert_eq!(en.ids, en2.ids, "EN ids must be deterministic");

    // JA and EN must not collapse onto the same language id (multilingual
    // dispatch is real, not a single-language fallback) — when the voice has
    // both languages.
    if cfg.language_id("ja").is_some() && cfg.language_id("en").is_some() {
        assert_ne!(ja.lid, en.lid, "JA and EN must detect different lids");
    }

    // ---- Campaign golden (css10-ja-6lang signature only) ----
    let is_campaign_voice = pad == 0
        && bos == 1
        && eos == 2
        && cfg.language_codes.first().map(String::as_str) == Some("ja")
        && cfg.language_codes.get(1).map(String::as_str) == Some("en");
    if is_campaign_voice {
        assert_eq!(
            en.ids, CAMPAIGN_EN_IDS,
            "EN ids must byte-match the campaign-1 real-G2P dump for this voice"
        );
        assert_eq!(en.lid, 1, "campaign dump recorded lid=1 for en");
    } else {
        eprintln!(
            "tts_g2p_injection: voice is not the campaign css10-ja-6lang checkpoint — \
             structural asserts only (golden id comparison skipped)."
        );
    }
}

// ---------------------------------------------------------------------------
// Raw HTTP JSON client (mirrors tests/piper_http_compat.rs — raw bytes so the
// binary WAV body is validated without charset re-decoding).
// ---------------------------------------------------------------------------

async fn http_post_json(
    addr: SocketAddr,
    path: &str,
    body: &[u8],
) -> std::io::Result<(u16, String, Vec<u8>)> {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let mut sock = tokio::net::TcpStream::connect(addr).await?;
    let head = format!(
        "POST {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n\
         Content-Length: {len}\r\nConnection: close\r\n\r\n",
        len = body.len(),
    );
    sock.write_all(head.as_bytes()).await?;
    sock.write_all(body).await?;
    sock.flush().await?;

    let mut buf = Vec::new();
    tokio::time::timeout(Duration::from_secs(30), sock.read_to_end(&mut buf))
        .await
        .map_err(|_| std::io::Error::other("http read timeout"))??;

    let sep = buf
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .ok_or_else(|| std::io::Error::other("no header terminator"))?;
    let head_str = std::str::from_utf8(&buf[..sep])
        .map_err(|e| std::io::Error::other(format!("utf8: {e}")))?
        .to_string();
    let status: u16 = head_str
        .lines()
        .next()
        .unwrap_or("")
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| std::io::Error::other("bad status line"))?;
    Ok((status, head_str, buf[sep + 4..].to_vec()))
}

/// Decode the canonical 44-byte-header PCM16 mono WAV our handler emits.
/// Returns `(sample_rate, samples)`.
fn decode_wav_pcm16(body: &[u8]) -> (u32, Vec<i16>) {
    assert!(body.len() >= 44, "WAV too small: {} bytes", body.len());
    assert_eq!(&body[0..4], b"RIFF");
    assert_eq!(&body[8..12], b"WAVE");
    assert_eq!(&body[36..40], b"data", "canonical 44-byte header expected");
    let sr = u32::from_le_bytes([body[24], body[25], body[26], body[27]]);
    let data_size = u32::from_le_bytes([body[40], body[41], body[42], body[43]]) as usize;
    assert_eq!(
        44 + data_size,
        body.len(),
        "data chunk size must match body length"
    );
    let samples = body[44..]
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect();
    (sr, samples)
}

// ---------------------------------------------------------------------------
// Layer 2 — wired-server e2e: production startup path, both phonemizer modes.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn wired_server_g2p_plain_text_and_passthrough_unchanged() {
    let (Some(piper), Some(whisper)) = (piper_gguf(), whisper_gguf()) else {
        eprintln!(
            "tts_g2p_injection: skipping — set VOKRA_PIPER_GGUF + VOKRA_WHISPER_GGUF to \
             run the wired-server e2e."
        );
        return;
    };

    let base_cfg = Config {
        http_bind: "127.0.0.1:0".parse().unwrap(),
        wyoming_bind: "127.0.0.1:0".parse().unwrap(),
        config_file: None,
        whisper_base_gguf: Some(whisper),
        piper_plus_gguf: Some(piper.clone()),
        ..Config::default()
    };

    // ---- Boot A: --piper-g2p → plain JA text really synthesizes. ----
    let cfg = Config {
        piper_g2p: Some(true),
        ..base_cfg.clone()
    };
    let (handles, trigger) = spawn_server_for_test_wired(cfg)
        .await
        .expect("spawn G2P server");

    let body = serde_json::json!({ "text": JA_GREETING, "voice": "default" }).to_string();
    let (status, headers, wav1) = http_post_json(handles.http_actual, "/api/tts", body.as_bytes())
        .await
        .expect("POST /api/tts (G2P)");
    assert_eq!(
        status,
        200,
        "plain JA text must synthesize under --piper-g2p; body: {:?}",
        String::from_utf8_lossy(&wav1[..wav1.len().min(256)])
    );
    assert!(
        headers
            .to_ascii_lowercase()
            .contains("content-type: audio/wav"),
        "response must be audio/wav; headers:\n{headers}"
    );
    let (sr, samples) = decode_wav_pcm16(&wav1);
    let dur_s = samples.len() as f64 / f64::from(sr);
    let peak = samples
        .iter()
        .map(|s| i32::from(*s).abs())
        .max()
        .unwrap_or(0);
    assert!(
        (0.2..=20.0).contains(&dur_s),
        "JA greeting duration implausible: {dur_s:.3}s ({} samples @ {sr} Hz)",
        samples.len()
    );
    assert!(
        peak > 500,
        "JA greeting must be non-silent (peak |i16| = {peak})"
    );

    // Determinism: same request twice → byte-identical WAV (the VITS noise
    // is drawn from a fixed-seed RNG per synthesize call).
    let (status2, _h2, wav2) = http_post_json(handles.http_actual, "/api/tts", body.as_bytes())
        .await
        .expect("POST /api/tts (G2P, run 2)");
    assert_eq!(status2, 200);
    assert_eq!(wav1, wav2, "same text must produce byte-identical WAV");

    // Unknown voice stays an honest 404 on the same boot.
    let bad = serde_json::json!({ "text": JA_GREETING, "voice": "nope" }).to_string();
    let (status_bad, _h, _b) = http_post_json(handles.http_actual, "/api/tts", bad.as_bytes())
        .await
        .expect("POST /api/tts (bad voice)");
    assert_eq!(status_bad, 404, "unknown voice must 404, never reroute");

    trigger.trigger();
    tokio::time::sleep(Duration::from_millis(20)).await;

    eprintln!(
        "tts_g2p_injection e2e (G2P boot): {} samples @ {sr} Hz = {dur_s:.3}s, peak |i16| {peak}, \
         deterministic across 2 runs",
        samples.len()
    );

    // ---- Boot B: default (no --piper-g2p) → behaviour unchanged. ----
    let (handles, trigger) = spawn_server_for_test_wired(base_cfg)
        .await
        .expect("spawn passthrough server");

    // Plain text = explicit 400 naming the passthrough (FR-EX-08), exactly
    // the campaign-observed contract.
    let (status_txt, _h, body_txt) =
        http_post_json(handles.http_actual, "/api/tts", body.as_bytes())
            .await
            .expect("POST /api/tts (passthrough, text)");
    assert_eq!(
        status_txt, 400,
        "plain text without G2P must stay an explicit 400"
    );
    let err_json: serde_json::Value = serde_json::from_slice(&body_txt).expect("error envelope");
    assert!(
        err_json["error"]["message"]
            .as_str()
            .unwrap_or_default()
            .contains("PassthroughPhonemizer"),
        "explicit error must name the phonemizer, got {err_json}"
    );

    // Raw phoneme ids still synthesize unchanged. Derive real content ids for
    // the same greeting from the G2P (strip framing/pads — the passthrough
    // re-frames), mirroring the campaign raw-id probe recipe.
    let voice = PiperPlusTts::from_path(&piper).expect("load voice");
    let g2p = PiperPlusG2p::from_voice(&voice).expect("g2p");
    let cfgv = voice.config().clone();
    let pad = cfgv
        .phoneme_symbols
        .iter()
        .position(|s| s == "_")
        .expect("pad symbol") as i64;
    let utt = g2p.phonemize_full(JA_GREETING).expect("phonemize");
    let content: Vec<String> = utt.ids[1..utt.ids.len() - 1]
        .iter()
        .filter(|&&i| i != pad)
        .map(|i| i.to_string())
        .collect();
    let raw_body = serde_json::json!({ "text": content.join(" "), "voice": "default" }).to_string();
    let (status_ids, headers_ids, wav_ids) =
        http_post_json(handles.http_actual, "/api/tts", raw_body.as_bytes())
            .await
            .expect("POST /api/tts (passthrough, raw ids)");
    assert_eq!(
        status_ids,
        200,
        "raw phoneme-id requests must keep working on the default boot; body: {:?}",
        String::from_utf8_lossy(&wav_ids[..wav_ids.len().min(256)])
    );
    assert!(
        headers_ids
            .to_ascii_lowercase()
            .contains("content-type: audio/wav")
    );
    let (_sr_ids, samples_ids) = decode_wav_pcm16(&wav_ids);
    let peak_ids = samples_ids
        .iter()
        .map(|s| i32::from(*s).abs())
        .max()
        .unwrap_or(0);
    assert!(
        peak_ids > 500,
        "raw-id synthesis must be non-silent (peak |i16| = {peak_ids})"
    );

    trigger.trigger();
    tokio::time::sleep(Duration::from_millis(20)).await;
}
