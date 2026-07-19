//! Runtime transcription-prompt layout (P2 cc-05/07 follow-up):
//! bit-check vs the offline `mistral_common` dump + real-checkpoint e2e.
//!
//! # What this pins
//!
//! The 1956ca6 land proved the real Voxtral pipeline transcribes JFK exactly
//! — but only via a prompt whose token ids were dumped OFFLINE with
//! `mistral_common`. This file pins the runtime replacement:
//! [`VoxtralTokenizer::transcription_prompt`] builds the same ids from the
//! GGUF-embedded compact vocab alone, and [`VoxtralAsr::transcribe`] (default
//! layout) produces the exact JFK sentence with no offline input.
//!
//! # Gate posture
//!
//! Every test here is env-gated — unset → clean skip with a diagnostic,
//! never a fabricated pass:
//!
//! - `VOKRA_VOXTRAL_TEKKEN_VOCAB` — the tekken **compact-vocab** blob (the
//!   same bytes the converter embeds as `vokra.tokenizer.model`). Cheaper
//!   than opening the 9.4 GB GGUF when only the vocab is needed; when unset
//!   the vocab is read from `VOKRA_VOXTRAL_GGUF` instead.
//! - `VOKRA_VOXTRAL_REF_DIR` — the offline reference dir carrying
//!   `prompt.json` (produced by `dump_transcription_prompt.py`, which drives
//!   `mistral_common`'s `TranscriptionRequest` +
//!   `encode_transcription` — the exact upstream path
//!   `VoxtralProcessor.apply_transcription_request` uses).
//! - `VOKRA_VOXTRAL_GGUF` — a converted real Voxtral GGUF (e2e leg).
//! - `VOKRA_VOXTRAL_WAV` — 16 kHz mono WAV for the e2e leg; defaults to the
//!   committed `tests/fixtures/audio/jfk-30s.wav`.
//!
//! # Measured (2026-07-19, M1 iMac, `voxtral-mini-3b-bf16-fs.gguf`)
//!
//! - prompt bit-check: runtime `pre_audio = [1, 3, 25]` /
//!   `post_audio = [4, 9909, 1058, 1262, 34]` == the `mistral_common` dump,
//!   element-for-element;
//! - e2e: the exact JFK sentence, EOS-terminated (see the e2e test's
//!   `EXPECT`).

use std::path::PathBuf;

use vokra_models::voxtral::{VoxtralAsr, VoxtralTokenizer};

/// Mistral's shipping EOS (`</s>` = 2) — only used to construct the
/// tokenizer; the prompt build never consults it.
const MISTRAL_EOS: u32 = 2;

fn env_path(key: &str) -> Option<PathBuf> {
    std::env::var_os(key).map(PathBuf::from)
}

/// Loads the tokenizer from the compact-vocab blob when available, else from
/// the real GGUF's `vokra.tokenizer.model` chunk.
fn load_tokenizer() -> Option<VoxtralTokenizer> {
    if let Some(p) = env_path("VOKRA_VOXTRAL_TEKKEN_VOCAB") {
        let bytes = std::fs::read(&p).expect("read the tekken compact-vocab blob");
        let tok = VoxtralTokenizer::from_bytes(bytes, MISTRAL_EOS)
            .expect("parse the tekken compact-vocab blob");
        assert!(
            tok.has_compact_vocab(),
            "VOKRA_VOXTRAL_TEKKEN_VOCAB={} is not a compact-vocab dump (the runtime prompt \
             builder needs the per-id table)",
            p.display()
        );
        return Some(tok);
    }
    let gguf = env_path("VOKRA_VOXTRAL_GGUF")?;
    let file = vokra_mmap::open_gguf(&gguf).expect("mmap-parse the Voxtral GGUF");
    Some(VoxtralTokenizer::from_gguf(&file, MISTRAL_EOS).expect("read the embedded tokenizer"))
}

/// Extracts a flat `[u32]` JSON array field from the dump (the file is a
/// single-level object of int arrays / ints, so a bracket scan is exact —
/// no JSON dep is pulled into the workspace, NFR-DS-02).
fn json_u32_array(src: &str, field: &str) -> Vec<u32> {
    let key = format!("\"{field}\":");
    let start = src
        .find(&key)
        .unwrap_or_else(|| panic!("prompt.json has no `{field}` field"))
        + key.len();
    let rest = &src[start..];
    let open = rest.find('[').expect("array open bracket");
    let close = rest.find(']').expect("array close bracket");
    rest[open + 1..close]
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.parse().expect("array element is a u32"))
        .collect()
}

/// The runtime-built prompt must equal the offline `mistral_common` dump
/// **element-for-element** — the whole point of the P2 change (no offline
/// prompt input at inference time).
#[test]
fn runtime_prompt_ids_match_offline_mistral_common_dump() {
    let Some(ref_dir) = env_path("VOKRA_VOXTRAL_REF_DIR") else {
        eprintln!(
            "[voxtral_transcription_prompt] SKIP: set VOKRA_VOXTRAL_REF_DIR to the offline \
             reference dir (carrying prompt.json from dump_transcription_prompt.py)."
        );
        return;
    };
    let Some(tok) = load_tokenizer() else {
        eprintln!(
            "[voxtral_transcription_prompt] SKIP: set VOKRA_VOXTRAL_TEKKEN_VOCAB (compact-vocab \
             blob) or VOKRA_VOXTRAL_GGUF to supply the tekken vocab."
        );
        return;
    };
    let dump = std::fs::read_to_string(ref_dir.join("prompt.json")).expect("read prompt.json");
    let want_pre = json_u32_array(&dump, "pre_audio");
    let want_post = json_u32_array(&dump, "post_audio");

    // The dump was produced with language="en" (see dump_transcription_prompt.py).
    let got = tok
        .transcription_prompt(Some("en"))
        .expect("build the transcription prompt at runtime");

    eprintln!(
        "[voxtral_transcription_prompt] runtime pre={:?} post={:?} | dump pre={:?} post={:?}",
        got.pre_audio, got.post_audio, want_pre, want_post
    );
    assert_eq!(
        got.pre_audio, want_pre,
        "runtime pre-audio segment must equal the mistral_common dump"
    );
    assert_eq!(
        got.post_audio, want_post,
        "runtime post-audio segment must equal the mistral_common dump"
    );

    // Structural cross-checks against the shipping tekken table, so a vocab
    // swap that silently renumbers the specials is caught here too.
    assert_eq!(tok.token_id_of_special("<s>").unwrap(), got.pre_audio[0]);
    assert_eq!(tok.token_id_of_special("[INST]").unwrap(), got.pre_audio[1]);
    assert_eq!(
        tok.token_id_of_special("[BEGIN_AUDIO]").unwrap(),
        got.pre_audio[2]
    );
    assert_eq!(
        tok.token_id_of_special("[/INST]").unwrap(),
        got.post_audio[0]
    );
    assert_eq!(
        tok.token_id_of_special("[TRANSCRIBE]").unwrap(),
        *got.post_audio.last().unwrap()
    );

    // `language = None` drops exactly the "lang:xx" run (upstream:
    // `if request.language is not None`).
    let no_lang = tok.transcription_prompt(None).unwrap();
    assert_eq!(no_lang.pre_audio, got.pre_audio);
    assert_eq!(
        no_lang.post_audio,
        vec![got.post_audio[0], *got.post_audio.last().unwrap()],
        "language=None must leave only [/INST] + [TRANSCRIBE]"
    );
}

/// Minimal RIFF/WAVE PCM16 mono reader (test-only; keeps the workspace
/// zero-dep). Returns `(samples, sample_rate)`.
fn read_wav_pcm16(path: &PathBuf) -> (Vec<f32>, u32) {
    let b = std::fs::read(path).expect("read WAV");
    assert!(b.len() > 44, "WAV too short: {}", path.display());
    assert_eq!(&b[0..4], b"RIFF", "not a RIFF file");
    assert_eq!(&b[8..12], b"WAVE", "not a WAVE file");
    let mut pos = 12usize;
    let (mut sample_rate, mut channels, mut bits) = (0u32, 0u16, 0u16);
    let mut data: Option<(usize, usize)> = None;
    while pos + 8 <= b.len() {
        let id = &b[pos..pos + 4];
        let size = u32::from_le_bytes([b[pos + 4], b[pos + 5], b[pos + 6], b[pos + 7]]) as usize;
        let body = pos + 8;
        if id == b"fmt " {
            channels = u16::from_le_bytes([b[body + 2], b[body + 3]]);
            sample_rate = u32::from_le_bytes([b[body + 4], b[body + 5], b[body + 6], b[body + 7]]);
            bits = u16::from_le_bytes([b[body + 14], b[body + 15]]);
        } else if id == b"data" {
            data = Some((body, size.min(b.len() - body)));
        }
        pos = body + size + (size & 1);
    }
    let (off, len) = data.expect("WAV has no data chunk");
    assert_eq!(channels, 1, "expected mono WAV");
    assert_eq!(bits, 16, "expected PCM16 WAV");
    let samples = b[off..off + len]
        .chunks_exact(2)
        .map(|c| f32::from(i16::from_le_bytes([c[0], c[1]])) / 32768.0)
        .collect();
    (samples, sample_rate)
}

/// The P2 acceptance: the real 9.4 GB checkpoint transcribes JFK exactly
/// through `VoxtralAsr::transcribe` with a **runtime-constructed** prompt
/// (default [`AsrPromptLayout::Transcription`]) — no env/offline prompt
/// input anywhere on the path.
#[test]
fn real_gguf_transcribes_jfk_with_runtime_prompt() {
    /// The upstream-verified transcript (1956ca6: offline tekken decode ==
    /// runtime decode on this clip).
    const EXPECT: &str = "And so, my fellow Americans, ask not what your country can do for \
                          you, ask what you can do for your country.";

    let Some(gguf_path) = env_path("VOKRA_VOXTRAL_GGUF") else {
        eprintln!(
            "[voxtral_transcription_prompt] SKIP e2e: set VOKRA_VOXTRAL_GGUF to a converted \
             real Voxtral GGUF (adapter chunk + embedded tokenizer)."
        );
        return;
    };
    let wav_path = env_path("VOKRA_VOXTRAL_WAV").unwrap_or_else(|| {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/audio/jfk-30s.wav")
    });
    if !wav_path.is_file() {
        eprintln!(
            "[voxtral_transcription_prompt] SKIP e2e: no WAV at {} (set VOKRA_VOXTRAL_WAV).",
            wav_path.display()
        );
        return;
    }
    let (pcm, sr) = read_wav_pcm16(&wav_path);
    assert_eq!(sr, 16_000, "the Voxtral front-end is fixed at 16 kHz");

    let t0 = std::time::Instant::now();
    let file = vokra_mmap::open_gguf(&gguf_path).expect("mmap-parse the Voxtral GGUF");
    let asr = VoxtralAsr::from_gguf(&file).expect("load the Voxtral model + tokenizer");
    eprintln!(
        "[voxtral_transcription_prompt] loaded in {:.1} s (layout={:?}, lang={:?})",
        t0.elapsed().as_secs_f64(),
        asr.prompt_layout(),
        asr.language(),
    );

    // ONE inference: `AsrEngine::transcribe` is literally
    // `decode(transcribe_ids(pcm))`, so driving the id path once and
    // detokenizing with the same engine tokenizer exercises the identical
    // code with half the wall time (the 9.4 GB checkpoint is slow enough
    // that a second forward would double an already-long gated run).
    let t1 = std::time::Instant::now();
    let ids = asr.transcribe_ids(&pcm).expect("greedy transcribe");
    let decode_secs = t1.elapsed().as_secs_f64();
    let text = asr
        .tokenizer()
        .expect("the real GGUF embeds its tokenizer")
        .decode(&ids)
        .expect("detokenize");
    eprintln!(
        "[voxtral_transcription_prompt] {} tokens in {decode_secs:.1} s -> {text:?}",
        ids.len(),
    );

    assert_eq!(
        text.trim(),
        EXPECT,
        "the trained transcription-prompt layout must reproduce the upstream transcript"
    );
    assert_eq!(
        ids.last().copied(),
        Some(asr.eos_id()),
        "the decode must terminate on EOS, not the max-new-token cap"
    );
}
