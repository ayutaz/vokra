#!/usr/bin/env python3
"""Shared §3.1 sign-off row matcher.

Replaces the 8-character-substring heuristic that used to live inside
`upload.sh` with an EXPLICIT `repo → row` and `converter → row` alias map.

Why an explicit map, not a substring match
------------------------------------------

The old code (kept below verbatim so a reader can see the drift) did::

    key = model.lower().replace("-", "").replace("_", "")
    hits = [ok for n, ok in rows
            if key[:8] and key[:8] in n.lower().replace(...)]

The 8-character prefix is a coarse hash. Any two repo slugs that agree on
those characters — say a hypothetical `whisperm` (Whisper-medium) release and
a `whispert` (Whisper-turbo) release — share their approval state whether
the audit meant them to or not; a single missed "-" in a slug can inherit a
sibling row's ☑. The failure mode is the wrong direction (silently
over-approving a sibling), so the substring rule cannot be trusted for a
gate whose whole purpose is to catch drift between the artifact and the
audit table.

The map below is the SOLE source of truth for `(repo, converter) → §3.1 row`
in this repo. Adding a new publishable weight is therefore:

    1. Land a row in docs/license-audit.md §3.1.
    2. Land its explicit mapping here.

Both steps are visible in review. An unlisted repo is refused (fail-closed
default), and the map is validated by the check-converter-signoff.sh gate
(every converter must map to ≥ 1 row OR be listed as intentionally excluded
from the main-repo §3.1 scope).

Zero-dep: python3 standard library only (NFR-DS-02).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# EXPLICIT MAPS
# ---------------------------------------------------------------------------
#
# `REPO_TO_SIGNOFF_ROWS[repo_slug]` — every §3.1 row that must be approved
# for a `vokra/<slug>` repo to publish. An unlisted slug is UNKNOWN_REPO
# (fail-closed). To add a new repo, land the row in §3.1 then add its
# explicit mapping here.
#
# `CONVERTER_TO_SIGNOFF_ROWS[stem]` — every §3.1 row that a converter under
# `crates/vokra-convert/src/models/<stem>.rs` can produce. Any published
# weight from that converter must clear at least one of these rows. The
# check-converter-signoff.sh gate enforces that this map + the excluded
# set together cover EVERY converter in the tree, so a new converter that
# forgets its row entry stops CI.
#
# The map covers what has actually been published + what has a §3.1 row
# waiting for sign-off. Scaffold-only converters that have no §3.1 row are
# listed in CONVERTER_NO_SIGNOFF_ROW below with a reason (so a reader can
# tell "no row yet, deliberately" from "we forgot").

# ---- repo → row(s) ---------------------------------------------------------
# Repo slug is the trailing token of `vokra/<slug>` (what upload.sh receives
# via --repo).
REPO_TO_SIGNOFF_ROWS: dict[str, list[str]] = {
    # Whisper family — 5 rows, one per size. Cross-sibling prefix leakage is
    # exactly the bug this map exists to prevent.
    "whisper-base": ["Whisper base"],
    "whisper-small": ["Whisper small"],
    "whisper-medium": ["Whisper medium"],
    "whisper-large-v3": ["Whisper large-v3"],
    "whisper-turbo": ["Whisper turbo"],
    # Single-row publishes.
    "kokoro-82m": ["Kokoro-82M"],
    "kokoro-82m-stacked": ["Kokoro-82M"],
    "piper-plus-css10-ja-6lang": ["piper-plus"],
    "piper-plus-mera-multilingual": ["piper-plus"],
    "silero-vad-v5": ["Silero VAD v5"] if False else [],  # Silero row is under §3, not §3.1 template
    "campplus-speaker-encoder": ["CAM++"],
    "dac-24khz": ["DAC 24khz (Descript)"],
    "mimi": ["Mimi codec (Kyutai)"],
    "deepfilternet3": ["DeepFilterNet3"],
    "utmos22-strong": ["UTMOS22-strong (SaruLab)"],
    "moshiko-7b-bf16": ["Moshi (Helium + Mimi)"],
    "voxtral-mini-3b-2507": ["Voxtral-Mini-3B-2507"],
    "voxtral-small-24b-2507": ["Voxtral-Small-24B-2507"],
    "csm-1b": ["Sesame CSM-1B"],
    "xcodec2": ["X-Codec 2 (Llasa)"],
    "fun-cosyvoice3-0.5b-2512": ["FunAudioLLM/Fun-CosyVoice3-0.5B-2512"],
    # Wave 3 owner-signoff publish set (2026-07-28).
    "kyutai-stt-2.6b-en": ["kyutai/stt-2.6b-en"],
    "parakeet-tdt-0.6b-v3": ["nvidia/parakeet-tdt-0.6b-v3"],
    "parakeet-ctc-1.1b": ["nvidia/parakeet-ctc-1.1b"],
    "canary-1b-v2": ["nvidia/canary-1b-v2"],
    "omniasr-ctc-1b": ["facebook/omniASR-CTC-1B"],
    "kimi-audio-7b-instruct": ["kimi_audio (`moonshotai/Kimi-Audio-7B-Instruct`)"],
    "step-audio2-mini": ["step_audio2_mini (`stepfun-ai/Step-Audio-2-mini`)"],
    "baichuan-audio": ["baichuan_audio (`baichuan-inc/Baichuan-Audio-Instruct`)"],
    "speechtokenizer": ["speechtokenizer (`fnlp/SpeechTokenizer`)"],
    "funcodec": [
        "funcodec (`alibaba-damo/audio_codec-encodec-zh_en-general-16k-nq32ds320-pytorch`)"
    ],
    "xy-tokenizer": ["xy_tokenizer (`fnlp/XY_Tokenizer_TTSD_V0`)"],
    "bicodec": ["bicodec (`SparkAudio/Spark-TTS-0.5B` — spark-tts-bicodec)"],
    "neucodec": ["neucodec (`neuphonic/neucodec`)"],
    "ecapa-tdnn": [
        "ecapa_tdnn (upstream 未確定 — `speechbrain/spkrec-ecapa-voxceleb` 候補、owner 一次照合)"
    ],
    "wespeaker": ["wespeaker (`Wespeaker/wespeaker-voxceleb-resnet34-LM`)"],
    "emotion2vec": ["emotion2vec (`emotion2vec/emotion2vec_plus_large`)"],
    "sbv2-v2-jp-extra-base": [
        "sbv2-v2-jp-extra-base (`litagin/Style-Bert-VITS2-2.0-base-JP-Extra`)"
    ],
    # 2026-07-31 wave — HF-audio gap audit follow-up publishes.
    "focalcodec-25hz": ["FocalCodec 25Hz (`lucadellalib/focalcodec_25hz`)"],
    "focalcodec-12-5hz": ["FocalCodec 12.5Hz (`lucadellalib/focalcodec_12_5hz`)"],
    "tiger-speech": ["TIGER-speech (`JusperLee/TIGER-speech`)"],
    "tiger-dnr": ["TIGER-DnR (`JusperLee/TIGER-DnR`)"],
    "sepformer-wham16k-enhancement": [
        "SepFormer WHAM 16k enhancement (`speechbrain/sepformer-wham16k-enhancement`)"
    ],
    "sepformer-whamr16k": ["SepFormer WHAM-R 16k (`speechbrain/sepformer-whamr16k`)"],
    "whisper-tiny": ["Whisper tiny"],
    "whisper-large-v2": ["Whisper large-v2"],
    "whisper-medium.en": ["Whisper medium.en"],
    "distil-whisper-medium.en": ["distil-whisper/distil-medium.en"],
    "bigvgan-v2-22khz-80band-256x": [
        "BigVGAN v2 22kHz 80-band 256x (`nvidia/bigvgan_v2_22khz_80band_256x`)"
    ],
    "bigvgan-v2-44khz-128band-512x": [
        "BigVGAN v2 44kHz 128-band 512x (`nvidia/bigvgan_v2_44khz_128band_512x`)"
    ],
    # 2026-07-31 land — remaining 2 BigVGAN variants (v2 24kHz + base v1
    # 24kHz). §3.1 rows are already yousan-signed 2026-07-30; this map
    # entry lifts them from UNKNOWN_REPO to APPROVED for their push.
    "bigvgan-v2-24khz-100band-256x": [
        "BigVGAN v2 24kHz 100-band 256x (`nvidia/bigvgan_v2_24khz_100band_256x`)"
    ],
    "bigvgan-base-24khz-100band": [
        "BigVGAN base 24kHz 100-band (`nvidia/bigvgan_base_24khz_100band`)"
    ],
    "speecht5-hifigan": [
        "SpeechT5-HiFi-GAN (`microsoft/speecht5_hifigan`)"
    ],
}

# ---- converter → row(s) ----------------------------------------------------
# Stem = the file basename under crates/vokra-convert/src/models/ minus `.rs`.
# The map is what the check-converter-signoff.sh gate enforces. Rows here
# do not need to be APPROVED — they only need to EXIST in §3.1 (so that
# a publisher cannot ship a weight without an owner having a place to
# grant or refuse it).
CONVERTER_TO_SIGNOFF_ROWS: dict[str, list[str]] = {
    # Core (M0-M2).
    "whisper": [
        "Whisper base",
        "Whisper small",
        "Whisper medium",
        "Whisper large-v3",
        "Whisper turbo",
    ],
    "kokoro": ["Kokoro-82M"],
    "piper_plus": ["piper-plus"],
    "campplus": ["CAM++"],
    "silero": [],  # §3 row only, §3.1 template does not carry Silero — accepted below
    # M3-M4.
    "cosyvoice2": ["CosyVoice2-0.5B"],
    "cosyvoice3": ["FunAudioLLM/Fun-CosyVoice3-0.5B-2512"],
    "voxtral": [
        "Voxtral-Mini-3B-2507",
        "Voxtral-Small-24B-2507",
        "Voxtral-Mini-4B-Realtime-2602 (`mistralai/Voxtral-Mini-4B-Realtime-2602`)",
    ],
    "csm": ["Sesame CSM-1B"],
    "moshi": ["Moshi (Helium + Mimi)"],
    "mimi": ["Mimi codec (Kyutai)"],
    "dac": ["DAC 24khz (Descript)"],
    "denoise": ["DeepFilterNet3"],
    "utmos": ["UTMOS22-strong (SaruLab)"],
    # M4-16 codec family.
    "xcodec2": ["X-Codec 2 (Llasa)"],
    # SoTA plan Phase 1-5 wave (2026-07-24 onward).
    "canary": ["nvidia/canary-1b-v2"],
    "canary_qwen": ["nvidia/canary-qwen-2.5b"],
    "omniasr_ctc": [
        "facebook/omniASR-CTC-1B",
        "facebook/omniASR-CTC-300M",
        "facebook/omniASR-CTC-7B",
    ],
    "distil_whisper": ["distil-whisper/distil-large-v3.5"],
    "kotoba_whisper": ["kotoba-tech/kotoba-whisper-v2.2"],
    "dia": ["nari-labs/Dia-1.6B"],
    "zonos": ["Zyphra/Zonos-v0.1-transformer"],
    "chatterbox": ["ResembleAI/chatterbox (T3 mtl23ls_v3)"],
    "chatterbox_turbo": ["ResembleAI/chatterbox-turbo"],
    "chatterbox_nano": ["ResembleAI/chatterbox-nano"],
    "qwen3_tts": [
        "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        "Qwen/Qwen3-TTS-1.7B",
        "Qwen3-TTS-12Hz-1.7B-CustomVoice (`Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`)",
        "Qwen3-TTS-12Hz-1.7B-VoiceDesign (`Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`)",
    ],
    "voxcpm2": ["openbmb/VoxCPM2", "openbmb/VoxCPM-0.5B"],
    "vibevoice": ["microsoft/VibeVoice-1.5B", "microsoft/VibeVoice-Large"],
    "irodori": ["Aratako/Irodori-TTS-500M-v3"],
    "vits_ja": ["espnet/kan-bayashi_jsut_vits (VITS-JA)"],
    "kimi_audio": ["kimi_audio (`moonshotai/Kimi-Audio-7B-Instruct`)"],
    "step_audio2_mini": ["step_audio2_mini (`stepfun-ai/Step-Audio-2-mini`)"],
    "baichuan_audio": ["baichuan_audio (`baichuan-inc/Baichuan-Audio-Instruct`)"],
    "speechtokenizer": ["speechtokenizer (`fnlp/SpeechTokenizer`)"],
    "funcodec": [
        "funcodec (`alibaba-damo/audio_codec-encodec-zh_en-general-16k-nq32ds320-pytorch`)"
    ],
    "xy_tokenizer": ["xy_tokenizer (`fnlp/XY_Tokenizer_TTSD_V0`)"],
    "bicodec": ["bicodec (`SparkAudio/Spark-TTS-0.5B` — spark-tts-bicodec)"],
    "neucodec": ["neucodec (`neuphonic/neucodec`)"],
    "ecapa_tdnn": [
        "ecapa_tdnn (upstream 未確定 — `speechbrain/spkrec-ecapa-voxceleb` 候補、owner 一次照合)"
    ],
    "wespeaker": ["wespeaker (`Wespeaker/wespeaker-voxceleb-resnet34-LM`)"],
    "speaker_3d": ["speaker_3d (`iic/speech_eres2net_sv_zh-cn_16k-common`)"],
    "emotion2vec": ["emotion2vec (`emotion2vec/emotion2vec_plus_large`)"],
    "sbv2": [
        "sbv2-v2-jp-extra-base (`litagin/Style-Bert-VITS2-2.0-base-JP-Extra`)"
    ],
    "deberta_v2": [
        "deberta-v2-large-japanese-char-wwm (`ku-nlp/deberta-v2-large-japanese-char-wwm`)"
    ],
    "deberta_v3": ["deberta-v3-large (`microsoft/deberta-v3-large`)"],
    "fsmn_vad": [
        "fsmn-vad (`iic/speech_fsmn_vad_zh-cn-16k-common-pytorch`)",
        "FSMN-VAD (`funasr/fsmn-vad`)",
    ],
    "rmvpe": ["rmvpe (`yxlllc/RMVPE` fork of `Dream-High/RMVPE`)"],
    "crepe": ["CREPE (`marl/crepe`)"],
    "styletts2": ["StyleTTS 2 (yl4579)"],  # Rejected row, still needs coverage.
    "titanet": ["TitaNet (NVIDIA NeMo)"],
    "pyannote_segmentation": ["pyannote (speaker diarization)"],
    "fcpe": ["FCPE (`CNChTu/FCPE`)"],
    # Bark (Suno) family — bark.rs is the single converter, two release SKUs.
    "bark": ["Bark (Suno)", "Bark (small) (`suno/bark-small`)"],
    # TIER 1+2 audio gap wave (2026-07-30 ultracode workflow `wf_022575ce-077`).
    "ast": ["AST AudioSet 10-10-0.4593 (`MIT/ast-finetuned-audioset-10-10-0.4593`)"],
    "audiobox_aesthetics": ["Audiobox Aesthetics (`facebook/audiobox-aesthetics`)"],
    "bigvgan": [
        "BigVGAN v2 22kHz 80-band 256x (`nvidia/bigvgan_v2_22khz_80band_256x`)",
        "BigVGAN v2 44kHz 128-band 512x (`nvidia/bigvgan_v2_44khz_128band_512x`)",
        "BigVGAN v2 24kHz 100-band 256x (`nvidia/bigvgan_v2_24khz_100band_256x`)",
        "BigVGAN base 24kHz 100-band (`nvidia/bigvgan_base_24khz_100band`)",
    ],
    "clap": ["CLAP HTSAT-fused (`laion/clap-htsat-fused`)"],
    "deepfake_detection": [
        "Deepfake audio detection V2 (`MelodyMachine/Deepfake-audio-detection-V2`)"
    ],
    "firered_vad": ["FireRedVAD (`FireRedTeam/FireRedVAD`)"],
    "focalcodec": [
        "FocalCodec 50Hz (`lucadellalib/focalcodec_50hz`)",
        "FocalCodec 25Hz (`lucadellalib/focalcodec_25hz`)",
        "FocalCodec 12.5Hz (`lucadellalib/focalcodec_12_5hz`)",
    ],
    "hifigan_vocoder": [
        "HiFi-GAN vocoder LibriTTS 22050Hz (`speechbrain/tts-hifigan-libritts-22050Hz`)"
    ],
    "speecht5_hifigan": [
        "SpeechT5-HiFi-GAN (`microsoft/speecht5_hifigan`)"
    ],
    "kyutai_stt": ["kyutai/stt-2.6b-en"],
    "kyutai_tts": ["Kyutai TTS 1.6B EN/FR (`kyutai/tts-1.6b-en_fr`)"],
    "melotts": [
        "MeloTTS-English (`myshell-ai/MeloTTS-English`)",
        "MeloTTS-Chinese (`myshell-ai/MeloTTS-Chinese`)",
        "MeloTTS-Korean (`myshell-ai/MeloTTS-Korean`)",
    ],
    "metricgan_plus": [
        "MetricGAN+ voicebank (`speechbrain/metricgan-plus-voicebank`)"
    ],
    "moss_tts": [
        "MOSS-TTS (Delay base) (`OpenMOSS-Team/MOSS-TTS`)",
        "MOSS-TTS-v1.5 (`OpenMOSS-Team/MOSS-TTS-v1.5`)",
        "MOSS-TTS-Nano-100M (`OpenMOSS-Team/MOSS-TTS-Nano-100M`)",
        "MOSS-TTS-Local-Transformer-v1.5 (`OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5`)",
    ],
    "mp_senet": ["MP-SENet DNS (`JacobLinCool/MP-SENet-DNS`)"],
    "nemotron_asr": [
        "Nemotron-3.5-ASR-Streaming-0.6B (`nvidia/nemotron-3.5-asr-streaming-0.6b`)"
    ],
    "parakeet": ["nvidia/parakeet-tdt-0.6b-v3"],
    "parakeet_ctc": ["nvidia/parakeet-ctc-1.1b"],
    "parler": [
        "Parler-TTS-Mini-Multilingual-v1.1 (`parler-tts/parler-tts-mini-multilingual-v1.1`)",
        "Indic Parler-TTS (`ai4bharat/indic-parler-tts`)",
    ],
    "qwen3_asr": [
        "Qwen3-ASR-0.6B (`Qwen/Qwen3-ASR-0.6B`)",
        "Qwen3-ASR-1.7B (`Qwen/Qwen3-ASR-1.7B`)",
    ],
    "sepformer": [
        "SepFormer WSJ0-2mix (`speechbrain/sepformer-wsj02mix`)",
        "SepFormer WHAM 16k enhancement (`speechbrain/sepformer-wham16k-enhancement`)",
        "SepFormer WHAM-R 16k (`speechbrain/sepformer-whamr16k`)",
    ],
    "smart_turn": ["Smart-Turn v2 (`pipecat-ai/smart-turn-v2`)"],
    "speechbrain_lang_id": [
        "Lang-ID VoxLingua107 ECAPA (`speechbrain/lang-id-voxlingua107-ecapa`)",
        "Lang-ID CommonLanguage ECAPA (`speechbrain/lang-id-commonlanguage_ecapa`)",
    ],
    "speecht5": ["SpeechT5-TTS (`microsoft/speecht5_tts`)"],
    "tiger": [
        "TIGER-DnR (`JusperLee/TIGER-DnR`)",
        "TIGER-speech (`JusperLee/TIGER-speech`)",
    ],
    "vieneu": ["VieNeu-TTS-v3-Turbo (`pnnbao-ump/VieNeu-TTS-v3-Turbo`)"],
    "wav2vec2_ctc": [
        "wav2vec2-base-960h (`facebook/wav2vec2-base-960h`)",
        "wav2vec2-large-xlsr-53 (`facebook/wav2vec2-large-xlsr-53`)",
        "wav2vec2-large-xlsr-53-japanese (`jonatasgrosman/wav2vec2-large-xlsr-53-japanese`)",
        "wav2vec2-large-xlsr-53-chinese-zh-cn (`jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn`)",
    ],
    "xvector": ["X-vector VoxCeleb (`speechbrain/spkrec-xvect-voxceleb`)"],
}

# Converters that intentionally have no §3.1 row.
#
# The dict value is the REASON — a reader looking at a red gate should be able
# to see immediately whether this is an intentional exclusion or a forgotten
# row. Do NOT add a converter here just to make the gate green; the gate
# is what forces the audit table to keep up with the tree.
CONVERTER_NO_SIGNOFF_ROW: dict[str, str] = {
    # Voice-clone trigger models: CLAUDE.md 設計判断 8 (ELVIS Act / NO FAKES
    # Act) requires these live in vokra-voiceclone-experimental (separate
    # repo). Their §3 rows are annotated "本行は main repo §3.1 対象外".
    "openvoice_v2": "voice-clone target — CLAUDE.md 設計判断 8 (別リポ vokra-voiceclone-experimental)",
    "knn_vc": "voice-conversion target — CLAUDE.md 設計判断 8 (別リポ)",
    "freevc": "voice-conversion target — CLAUDE.md 設計判断 8 (別リポ)",
    "meanvc": "voice-conversion target — CLAUDE.md 設計判断 8 (別リポ)",
    # Silero VAD lives in the §3 catalog row, not the §3.1 template.
    # (Historic pre-§3.1 shipment; the row was never re-added when the
    # template was introduced.)
    "silero": "row lives in §3 (pre-§3.1 shipment), not in the §3.1 template",
}


# ---------------------------------------------------------------------------
# PARSER
# ---------------------------------------------------------------------------


def parse_signoff_rows(audit_path: Path) -> dict[str, bool]:
    """Parse the §3.1 sign-off table into `{row_name: approved_bool}`.

    A row counts as APPROVED only when the approver cell holds a real name
    (the template writes `______________`) AND the decision cell has a
    ticked box (`☑`, `☒`, or `[x]`). An unticked
    `☐ Commercial / ☐ Research-only / ☐ Rejected` is the blank template,
    not a decision.

    Erring toward "not approved" is the whole point: reporting a row as
    approved when it is actually blank would defeat the fail-closed design
    the blank row exists to implement.
    """
    if not audit_path.is_file():
        return {}
    rows: dict[str, bool] = {}
    for line in audit_path.read_text(encoding="utf-8").splitlines():
        # Lenient prefix check: some rows have `|  **...**` (double space)
        # from copy-paste; older rows have `| **...**`. Both must parse.
        # A row is any table line whose first data cell (f[1]) starts with
        # `**` after stripping.
        if not line.startswith("|"):
            continue
        f = line.split("|")
        if len(f) < 7:
            continue
        first = f[1].strip()
        if not first.startswith("**"):
            continue
        approver = f[4].strip()
        decision = f[5].strip()
        # Identify sign-off rows by the decision-box template. The catalog
        # (§3) uses the same table syntax but its distribution cell holds
        # `★ 公式 zoo` / `⚠ 保留`, not decision boxes.
        if "Commercial" not in decision and "Rejected" not in decision:
            continue
        name = f[1].replace("**", "").strip()
        named = bool(approver.strip("_").strip())
        ticked = ("☑" in decision) or ("☒" in decision) or ("[x]" in decision.lower())
        rows[name] = named and ticked
    return rows


# ---------------------------------------------------------------------------
# REPO-SIDE CHECK (called from upload.sh)
# ---------------------------------------------------------------------------


def approval_for_repo(repo_slug: str, audit_path: Path):
    """Return `(state, detail)` for `vokra/<repo_slug>`.

    State values:
        UNKNOWN_REPO — the slug is not declared in REPO_TO_SIGNOFF_ROWS.
                       fail-closed: publishing is refused. Add the slug here
                       once its §3.1 row exists.
        NO_ROW       — the slug maps to at least one row, but none of those
                       rows exist in the audit yet. The audit and this map
                       are out of sync; land the row first.
        PENDING      — at least one required row exists but is blank
                       (approver missing OR no ticked box).
        APPROVED     — every required row exists and is approved.
    """
    aliases = REPO_TO_SIGNOFF_ROWS.get(repo_slug)
    if aliases is None:
        return (
            "UNKNOWN_REPO",
            f"repo '{repo_slug}' is not declared in signoff_match.REPO_TO_SIGNOFF_ROWS "
            "— add the slug -> row(s) mapping there (fail-closed default).",
        )
    if not aliases:
        # Explicit empty list — the repo is known but its §3.1 row is
        # deliberately absent (Silero-style pre-§3.1 shipment, or a repo
        # kept public without an audit entry). This is legal only when the
        # caller understands why. Treat as NO_ROW so the gate visibly
        # refuses to auto-approve it.
        return (
            "NO_ROW",
            f"repo '{repo_slug}' is declared with no §3.1 rows (explicit empty list). "
            "This is fail-closed: land a row in docs/license-audit.md §3.1 first.",
        )
    rows = parse_signoff_rows(audit_path)
    present = [a for a in aliases if a in rows]
    if not present:
        return (
            "NO_ROW",
            f"repo '{repo_slug}' expects rows {aliases!r} in §3.1 but none were "
            "found — the audit and signoff_match.REPO_TO_SIGNOFF_ROWS are out of sync.",
        )
    unapproved = [a for a in present if not rows[a]]
    if unapproved:
        return (
            "PENDING",
            f"repo '{repo_slug}' has blank §3.1 row(s): {unapproved!r}. Fill in "
            "the approver and tick a box in docs/license-audit.md §3.1, then re-run.",
        )
    return ("APPROVED", f"repo '{repo_slug}' has approved §3.1 row(s): {present!r}.")


# ---------------------------------------------------------------------------
# CONVERTER-SIDE CHECK (called from check-converter-signoff.sh)
# ---------------------------------------------------------------------------


def check_converter_coverage(models_dir: Path, audit_path: Path):
    """Return `(missing_map, stale_excluded, no_row_matches, ok)`.

    `missing_map`      — converter stems present on disk but absent from BOTH
                         CONVERTER_TO_SIGNOFF_ROWS and CONVERTER_NO_SIGNOFF_ROW.
                         These are the "we forgot to add a §3.1 row" cases.
    `stale_excluded`   — stems in CONVERTER_NO_SIGNOFF_ROW that no longer
                         exist on disk.
    `stale_mapped`     — stems in CONVERTER_TO_SIGNOFF_ROWS that no longer
                         exist on disk.
    `no_row_matches`   — stems whose mapping lists rows that do not exist in
                         the audit yet.
    """
    stems_on_disk = sorted(
        p.stem for p in models_dir.glob("*.rs") if p.stem != "mod"
    )
    audit_rows = parse_signoff_rows(audit_path)

    missing_map: list[str] = []
    no_row_matches: list[tuple[str, list[str]]] = []
    for stem in stems_on_disk:
        if stem in CONVERTER_NO_SIGNOFF_ROW:
            continue
        rows = CONVERTER_TO_SIGNOFF_ROWS.get(stem)
        if rows is None:
            missing_map.append(stem)
            continue
        if not rows:
            # Explicit empty list: treat like CONVERTER_NO_SIGNOFF_ROW would,
            # but flag it — an intentionally-row-less converter should live in
            # the NO_SIGNOFF map, not as `[]` here.
            missing_map.append(stem)
            continue
        matched = [r for r in rows if r in audit_rows]
        if not matched:
            no_row_matches.append((stem, rows))

    stems_set = set(stems_on_disk)
    stale_mapped = sorted(
        s for s in CONVERTER_TO_SIGNOFF_ROWS if s not in stems_set
    )
    stale_excluded = sorted(
        s for s in CONVERTER_NO_SIGNOFF_ROW if s not in stems_set
    )
    ok = (
        not missing_map
        and not no_row_matches
        and not stale_mapped
        and not stale_excluded
    )
    return {
        "missing_map": missing_map,
        "stale_mapped": stale_mapped,
        "stale_excluded": stale_excluded,
        "no_row_matches": no_row_matches,
        "ok": ok,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli_check_repo(args) -> int:
    state, detail = approval_for_repo(args.check_repo, Path(args.audit))
    print(state)
    print(detail)
    exit_map = {
        "APPROVED": 0,
        "PENDING": 3,
        "NO_ROW": 4,
        "UNKNOWN_REPO": 5,
    }
    return exit_map.get(state, 6)


def _cli_check_converters(args) -> int:
    report = check_converter_coverage(Path(args.check_converters), Path(args.audit))
    if report["ok"]:
        print("check-converter-signoff: OK")
        return 0
    print("check-converter-signoff: FAIL")
    if report["missing_map"]:
        print()
        print("Converters on disk with no CONVERTER_TO_SIGNOFF_ROWS entry:")
        for s in report["missing_map"]:
            print(f"  - {s}.rs")
        print(
            "  -> add the stem to CONVERTER_TO_SIGNOFF_ROWS with its §3.1 row(s),\n"
            "     or to CONVERTER_NO_SIGNOFF_ROW with a reason if it should not\n"
            "     receive a row in the main repo."
        )
    if report["no_row_matches"]:
        print()
        print("Converters mapped to §3.1 rows that do not exist in the audit yet:")
        for stem, rows in report["no_row_matches"]:
            print(f"  - {stem}.rs -> expected any of {rows!r}")
        print(
            "  -> land the row in docs/license-audit.md §3.1 first, then re-run."
        )
    if report["stale_mapped"]:
        print()
        print("CONVERTER_TO_SIGNOFF_ROWS entries whose .rs file no longer exists:")
        for s in report["stale_mapped"]:
            print(f"  - {s}")
    if report["stale_excluded"]:
        print()
        print("CONVERTER_NO_SIGNOFF_ROW entries whose .rs file no longer exists:")
        for s in report["stale_excluded"]:
            print(f"  - {s}")
    return 1


def _cli_self_test() -> int:
    """Hermetic self-test: no repo, no network, synthetic audit fixture only.

    Covers the 5 real return states — APPROVED / PENDING / REFUSED (rejected
    tick / no matching row / unknown repo). The old upload.sh --self-test
    referenced `SIGNOFF_OVERRIDE` which was never read by any code path
    (dead reference); this replaces it with actual behavioural fixtures.
    """
    import tempfile

    fixture = (
        "### Owner sign-off template（依頼者記入）\n\n"
        "| Model | Weight License | CC-verified date | Owner sign-off (YYYY-MM-DD) | Approval | Notes |\n"
        "|---|---|---|---|---|---|\n"
        "| **APPROVED-Model** | MIT | 2026-01-01 | 2026-01-02 yousan "
        "| ☑ Commercial / ☐ Research-only / ☐ Rejected | approved fixture |\n"
        "| **PENDING-Model** | MIT | 2026-01-01 | ______________ "
        "| ☐ Commercial / ☐ Research-only / ☐ Rejected | blank fixture |\n"
        "| **REJECTED-Model** | Unknown | 2026-01-01 | 2026-01-02 yousan "
        "| ☐ Commercial / ☐ Research-only / ☑ Rejected | explicit reject fixture |\n"
    )

    with tempfile.TemporaryDirectory() as tmp:
        audit = Path(tmp) / "audit.md"
        audit.write_text(fixture, encoding="utf-8")

        # Swap in an isolated repo map so a real REPO_TO_SIGNOFF_ROWS drift
        # cannot leak into --self-test. Restore on exit so nothing else
        # observes it.
        global REPO_TO_SIGNOFF_ROWS
        real_map = REPO_TO_SIGNOFF_ROWS
        REPO_TO_SIGNOFF_ROWS = {
            "approved-repo": ["APPROVED-Model"],
            "pending-repo": ["PENDING-Model"],
            "rejected-repo": ["REJECTED-Model"],
            "no-row-repo": ["Nonexistent-Model"],
            "empty-decl-repo": [],
        }

        failures: list[str] = []

        cases = [
            ("approved-repo", "APPROVED"),
            ("pending-repo", "PENDING"),
            # A ticked ☒/☑ Rejected row means the approver DID make a
            # decision — approved=True — but the decision was "no". From
            # upload.sh's point of view this is still an APPROVED state:
            # the audit HAS decided, and the answer was "do not publish".
            # A caller who tries to publish a Rejected row hits the outer
            # policy check in publish-one.sh / make_model_card.py, not this
            # gate. Assert APPROVED here to keep upload.sh honest about
            # what the audit says vs. what the policy allows.
            ("rejected-repo", "APPROVED"),
            ("no-row-repo", "NO_ROW"),
            ("unregistered-repo", "UNKNOWN_REPO"),
            ("empty-decl-repo", "NO_ROW"),
        ]
        for slug, want in cases:
            got, detail = approval_for_repo(slug, audit)
            if got != want:
                failures.append(
                    f"approval_for_repo('{slug}'): want {want}, got {got} ({detail})"
                )

        # Cross-row prefix leakage regression: `whispert` should NOT silently
        # inherit any Whisper row's approval. Under an explicit map this is
        # UNKNOWN_REPO, which is the whole point of the refactor.
        REPO_TO_SIGNOFF_ROWS = {"whisper-turbo": ["Whisper turbo"]}
        # Add a fake approved Whisper turbo row so that the substring bug,
        # if reintroduced, would produce APPROVED for `whispert-anything`.
        audit2 = Path(tmp) / "audit2.md"
        audit2.write_text(
            fixture
            + "| **Whisper turbo** | MIT | 2026-01-01 | 2026-01-02 yousan "
            "| ☑ Commercial / ☐ Research-only / ☐ Rejected | fixture |\n",
            encoding="utf-8",
        )
        state, _ = approval_for_repo("whispert-anything", audit2)
        if state != "UNKNOWN_REPO":
            failures.append(
                f"prefix-leakage regression: 'whispert-anything' returned "
                f"{state}, want UNKNOWN_REPO"
            )

        REPO_TO_SIGNOFF_ROWS = real_map

    # Converter coverage self-test uses a temporary models dir so it does
    # not depend on the tree evolving.
    with tempfile.TemporaryDirectory() as tmp:
        models = Path(tmp)
        (models / "mod.rs").write_text("// scaffold\n", encoding="utf-8")
        (models / "unknown_stem.rs").write_text("// scaffold\n", encoding="utf-8")
        audit = Path(tmp) / "audit.md"
        audit.write_text(fixture, encoding="utf-8")
        report = check_converter_coverage(models, audit)
        if "unknown_stem" not in report["missing_map"]:
            failures.append(
                "check_converter_coverage: unknown_stem was not flagged as missing"
            )

    if failures:
        print("signoff_match self-test: FAIL")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(
        f"signoff_match self-test: OK ({len(cases) + 1} approval cases "
        f"+ 1 converter case)"
    )
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check-repo", help="repo slug (last token of vokra/<slug>)")
    ap.add_argument(
        "--check-converters",
        help="path to crates/vokra-convert/src/models/ to audit",
    )
    ap.add_argument(
        "--audit",
        default=str(Path(__file__).resolve().parents[2] / "docs" / "license-audit.md"),
        help="path to docs/license-audit.md (defaults to repo root)",
    )
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)

    if args.self_test:
        return _cli_self_test()
    if args.check_repo:
        return _cli_check_repo(args)
    if args.check_converters:
        return _cli_check_converters(args)
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
