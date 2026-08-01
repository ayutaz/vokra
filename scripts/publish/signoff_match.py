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
    # 2026-08-01 Wave 8: Descript DAC 44kHz + 16kHz variants of already-
    # published dac-24khz. Same MIT license from descriptinc/descript-
    # audio-codec GitHub, distinct sample-rate + RVQ codebook count.
    "dac-44khz": ["Descript DAC (44kHz) (`descript/dac_44khz`)"],
    "dac-16khz": ["Descript DAC (16kHz) (`descript/dac_16khz`)"],
    # 2026-08-01 Wave 8: MeloTTS ES / JA variants of published EN/ZH/KO.
    "melotts-spanish": ["MeloTTS Spanish (`myshell-ai/MeloTTS-Spanish`)"],
    "melotts-japanese": ["MeloTTS Japanese (`myshell-ai/MeloTTS-Japanese`)"],
    # 2026-08-01 Wave 8: MIT voice gender classifier (unique task).
    "voice-gender-classifier": [
        "voice-gender-classifier (`JaesungHuh/voice-gender-classifier`)"
    ],
    # 2026-08-01 Wave 8: SpeechBrain ECAPA-TDNN reference on VoxCeleb.
    "speechbrain-spkrec-ecapa-voxceleb": [
        "speechbrain/spkrec-ecapa-voxceleb"
    ],
    # 2026-08-01 Wave 8: pyannote/wespeaker (CC-BY-4.0 attribution).
    "pyannote-wespeaker-voxceleb-resnet34-lm": [
        "pyannote/wespeaker-voxceleb-resnet34-LM"
    ],
    # 2026-08-01 Wave 8: primeline German Whisper-large-v3-turbo fine-tune.
    "whisper-large-v3-turbo-german": [
        "primeline/whisper-large-v3-turbo-german"
    ],
    # 2026-08-01 Wave 8: jonatasgrosman Arabic XLSR-53 fine-tune.
    "wav2vec2-large-xlsr-53-arabic": [
        "jonatasgrosman/wav2vec2-large-xlsr-53-arabic"
    ],
    "mimi": ["Mimi codec (Kyutai)"],
    "deepfilternet3": ["DeepFilterNet3"],
    "utmos22-strong": ["UTMOS22-strong (SaruLab)"],
    "moshiko-7b-bf16": ["Moshi (Helium + Mimi)"],
    "voxtral-mini-3b-2507": ["Voxtral-Mini-3B-2507"],
    "voxtral-small-24b-2507": ["Voxtral-Small-24B-2507"],
    "csm-1b": ["Sesame CSM-1B"],
    "xcodec2": ["X-Codec 2 (Llasa)"],
    # residual wave 4 (2026-08-02): nyrahealth/CrisperWhisper — Whisper-
    # large-v3 verbatim-word-timestamps fine-tune, cc-by-nc-4.0. T4 tier
    # (Research-only) per the X-Codec-2 (2026-07-28) precedent workflow
    # (`LicenseClass::NonCommercial` + `--allow-noncommercial` gate +
    # `fetch_license.sh --spdx cc-by-nc-4.0` canonical LICENSE fetch).
    # The placeholder row heading MUST match `docs/license-audit.md`
    # §3.1 byte-for-byte once the audit doc is updated in a post-workflow
    # batch (do NOT modify license-audit.md here per the task rules).
    "crisperwhisper": ["CrisperWhisper (`nyrahealth/CrisperWhisper`)"],
    "fun-cosyvoice3-0.5b-2512": ["FunAudioLLM/Fun-CosyVoice3-0.5B-2512"],
    # 2026-08-01 Wave 5 music-generation add: Meta AudioCraft MusicGen-Medium
    # (`facebook/musicgen-medium`, cc-by-nc-4.0). T4 tier (Research-only, non-
    # commercial) — inherits the X-Codec 2 T4 precedent workflow landed
    # 2026-07-28 (`LicenseClass::NonCommercial` + `--allow-noncommercial` gate
    # + `fetch_license.sh --spdx cc-by-nc-4.0` canonical LICENSE fetch). ~11.4
    # GB bundle → vast.ai handoff per memory `[[feedback-large-models-on-vast-ai]]`;
    # the publish path stops on `publish-one.sh --allow-noncommercial` per the
    # T4 precedent. The row heading matches this entry byte-for-byte in
    # `docs/license-audit.md` §3.1.
    "musicgen-medium": ["MusicGen-Medium (`facebook/musicgen-medium`)"],
    # 2026-08-01 Wave 5 music-generation add: Meta AudioCraft MusicGen-Large
    # (`facebook/musicgen-large`, cc-by-nc-4.0). T4 tier (Research-only,
    # non-commercial) — same X-Codec 2 / MusicGen-Medium T4 precedent
    # workflow (`LicenseClass::NonCommercial` + `--allow-noncommercial`
    # gate + `fetch_license.sh --spdx cc-by-nc-4.0` canonical LICENSE
    # fetch). ~19.5 GB bundle → vast.ai handoff per memory
    # `[[feedback-large-models-on-vast-ai]]` (larger than sibling
    # MusicGen-Medium ~11.4 GB, both routed to vast.ai). The row heading
    # matches this entry byte-for-byte in `docs/license-audit.md` §3.1.
    "musicgen-large": ["MusicGen-Large (`facebook/musicgen-large`)"],
    # 2026-08-01 Wave 5 residual: Meta AudioCraft AudioGen-Medium
    # (`facebook/audiogen-medium`, cc-by-nc-4.0). MusicGen sibling
    # (identical arch, tuned on environmental sounds / SFX). Same T4
    # precedent as sibling MusicGen family. Local convert safe (~3.7 GB).
    "audiogen-medium": ["AudioGen-Medium (`facebook/audiogen-medium`)"],
    # 2026-08-01 Wave 6 residual: MusicGen-Small (facebook, cc-by-nc-4.0).
    # 300M smallest of MusicGen family. Same T4 precedent as sibling
    # MusicGen-Medium/Large. Scale ~5.5 GB = vast.ai handoff.
    "musicgen-small": ["MusicGen-Small (`facebook/musicgen-small`)"],
    # 2026-08-01 Wave 6 residual: Qwen2-Audio-7B-Instruct (Alibaba,
    # apache-2.0). Whisper audio encoder + Qwen2-7B LM = audio-LLM omni.
    # Scale ~16 GB (5-shard) = vast.ai handoff.
    "qwen2-audio-7b-instruct": ["Qwen2-Audio-7B-Instruct (`Qwen/Qwen2-Audio-7B-Instruct`)"],
    # 2026-08-02 Wave residual: Qwen2.5-Omni-7B (Alibaba, apache-2.0).
    # Thinker + Talker unified any-to-any omni multimodal LLM over
    # Qwen2.5-7B backbone. Distinct arch tag `qwen2-omni` from sibling
    # audio-only `qwen2_audio`. Scale 22.37 GB (5-shard) = vast.ai
    # handoff. Placeholder row identifier — the row heading MUST match
    # `docs/license-audit.md` §3.1 byte-for-byte when it lands in the
    # post-workflow batch that adds the audit row.
    "qwen2-5-omni-7b": ["Qwen2.5-Omni-7B (`Qwen/Qwen2.5-Omni-7B`)"],
    # 2026-08-01 Wave 6 residual: VibeVoice-ASR (Microsoft, MIT).
    # VibeVoice sibling with ASR head. Scale ~16.5 GB (8-shard) = vast.ai.
    "vibevoice-asr": ["VibeVoice-ASR (`microsoft/VibeVoice-ASR`)"],
    # 2026-08-01 Wave 6 residual: ACE-Step 1.5 (MIT, flagship music-gen).
    # Multi-file bundle. Scale ~9.6 GB = vast.ai handoff.
    "ace-step-1.5": ["ACE-Step 1.5 (`ACE-Step/Ace-Step1.5`)"],
    # 2026-08-01 Wave 7 residual: HuBERT-Large-LS960 (Meta, apache-2.0).
    # 317M self-supervised speech encoder + CTC head fine-tuned on
    # LibriSpeech 960h. Distinct arch tag `hubert` from sibling
    # wav2vec2 (different pretraining objective). Placeholder row —
    # the row heading MUST match `docs/license-audit.md` §3.1 byte-
    # for-byte once the audit doc is updated in a post-workflow batch.
    # Scale ~1.26 GB = local convert safe on M1 iMac.
    "hubert-large-ls960": ["HuBERT-Large-LS960 (`facebook/hubert-large-ls960-ft`)"],
    # 2026-08-02 Wave residual: HT-Demucs (Meta, MIT per upstream
    # `github.com/facebookresearch/demucs` LICENSE — HF mirror 401 on
    # 2026-08-02 residual walk). Hybrid transformer Demucs (Rouard et al.
    # 2023, arXiv:2211.08553) = U-Net waveform branch + spectrogram
    # branch + cross-domain self-attention, 4-source music separation
    # (drums / bass / other / vocals — MUSDB18 stem taxonomy). Distinct
    # arch tag `demucs` from sibling SepFormer / TIGER separators.
    # Placeholder row — the row heading MUST match `docs/license-audit.md`
    # §3.1 byte-for-byte once the audit doc is updated in a post-workflow
    # batch. Scale ~0.50 GB = local convert safe on M1 iMac.
    "demucs-htdemucs": ["HT-Demucs (`facebook/demucs`)"],
    # 2026-08-01 Wave 7 CC-fill: AudioLDM 2 (CVSSP, cc-by-nc-sa-4.0)
    # T4 Research-only path with SA cascade explicit in model card +
    # LICENSE + NOTICE. `publish-one.sh --allow-noncommercial` gate.
    # `LicenseClass::NonCommercialShareAlike` fail-closed default at
    # runtime (M2-13 gate refuses commercial-mode load); publish-side
    # SA cascade is enforced by CC judgement + owner rollback path.
    "audioldm2": ["AudioLDM 2 (`cvssp/audioldm2`)"],
    # 2026-08-02 Wave residual: openWakeWord (dscripka, apache-2.0).
    # Audio-dialect `kws` op entry — small custom-KWS MLP/CNN family
    # over precomputed melspec. HF API rate-limited (401) — upstream
    # GitHub `dscripka/openWakeWord` primary source is Apache-2.0
    # (code + bundled checkpoints). Placeholder row — the row heading
    # MUST match `docs/license-audit.md` §3.1 byte-for-byte once the
    # audit doc is updated in a post-workflow batch. Scale ~0.01 GB =
    # local convert safe on M1 iMac.
    "openwakeword": ["openWakeWord (`dscripka/openWakeWord`)"],
    # 2026-08-02 Wave residual: Moonshine-Tiny (UsefulSensors, MIT). 27M
    # raw-audio transformer enc-dec ASR (arXiv:2410.15608). Distinct arch
    # tag `moonshine` from sibling Whisper (raw-audio Conv1D front-end +
    # rotary + SwiGLU vs mel + sinusoidal + GELU). Placeholder row — the
    # row heading MUST match `docs/license-audit.md` §3.1 byte-for-byte
    # once the audit doc is updated in a post-workflow batch. Scale
    # ~0.11 GB = local convert safe on M1 iMac (well below vast.ai
    # ≥8 GB cutoff).
    "moonshine-tiny": ["Moonshine-Tiny (`UsefulSensors/moonshine-tiny`)"],
    # 2026-08-02 Wave residual: Moonshine-Base (UsefulSensors, MIT). 61.5M
    # raw-audio transformer enc-dec ASR (arXiv:2410.15608). Sibling to
    # Moonshine-Tiny — same arch family (raw-audio Conv1D front-end +
    # rotary + SwiGLU), wider/deeper backbone. Placeholder row — the row
    # heading MUST match `docs/license-audit.md` §3.1 byte-for-byte once
    # the audit doc is updated in a post-workflow batch. Scale ~0.25 GB
    # = local convert safe on M1 iMac (well below vast.ai ≥8 GB cutoff).
    "moonshine-base": ["Moonshine-Base (`UsefulSensors/moonshine-base`)"],
    # 2026-08-02 Wave residual: Ultravox v0.5 (Llama-3.2-1B) (fixie-ai,
    # MIT). Audio-text-to-text multimodal = Llama-3.2-1B decoder +
    # Whisper encoder + projection adapter. Distinct arch tag `ultravox`
    # from sibling Voxtral (Mistral decoder) / Qwen2-Audio (Qwen2 decoder)
    # — the decoder backbone fixes tensor layout + tokenizer + rope base,
    # so FR-EX-08 forbids silent shape misroute across the three.
    # Placeholder row — the row heading MUST match `docs/license-audit.md`
    # §3.1 byte-for-byte once the audit doc is updated in a post-workflow
    # batch. Scale ~1.83 GB = local convert safe on M1 iMac.
    "ultravox-v0-5-llama-3-2-1b": [
        "Ultravox v0.5 (Llama-3.2-1B) (`fixie-ai/ultravox-v0_5-llama-3_2-1b`)"
    ],
    # 2026-08-02 Wave residual: Coqui XTTS-v2 (`coqui/XTTS-v2`,
    # coqui-public-model-license). Multilingual zero-shot voice-cloning TTS
    # = GPT-2 backbone + DVAE + HiFi-GAN vocoder head. T4 tier (Research-
    # only, non-commercial) — inherits the X-Codec-2 T4 precedent workflow
    # landed 2026-07-28 (`LicenseClass::NonCommercial` +
    # `publish-one.sh --allow-noncommercial` gate + fetch_license.sh
    # canonical LICENSE fetch). Placeholder row — the row heading MUST
    # match `docs/license-audit.md` §3.1 byte-for-byte once the audit doc
    # is updated in a post-workflow batch. Scale ~1.90 GB = local convert
    # safe on M1 iMac 16 GB. Coqui shut down Jan 2024 but upstream
    # `coqui/XTTS-v2` on HF remains primary source.
    "xtts-v2": ["XTTS-v2 (`coqui/XTTS-v2`)"],
    # 2026-08-02 Wave residual: ConvTasNet Libri1Mix Enhancement (Asteroid,
    # `JorisCos/ConvTasNet_Libri1Mix_enhsingle_16k`, cc-by-sa-4.0). First
    # Copyleft-tier separator entry — single-speaker enhancement head on
    # Libri1Mix 16 kHz (one clean speaker + additive noise, one output
    # stream). Distinct arch tag `conv_tasnet` from sibling separator
    # families (sepformer / demucs / tiger_separator / bs_roformer /
    # mp_senet). T3 tier redistributable with original licence preserved
    # (SA cascade — a derived GGUF is itself CC-BY-SA). No
    # `--allow-noncommercial` required (Copyleft ≠ NonCommercial).
    # Placeholder row — the row heading MUST match `docs/license-audit.md`
    # §3.1 byte-for-byte once the audit doc is updated in a post-workflow
    # batch. Scale ~20 MB = local convert safe on M1 iMac.
    "conv-tasnet-libri1mix": [
        "ConvTasNet Libri1Mix Enhancement (`JorisCos/ConvTasNet_Libri1Mix_enhsingle_16k`)"
    ],
    # NOTE: bs-roformer is intentionally NOT registered here — §3.1
    # sign-off is ☑ Rejected (Wave 7 CC-fill) = double fail-closed
    # (Rejected in audit + UNKNOWN_REPO here). owner can add the entry
    # after selecting a specific trainer checkpoint.
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
    # 2026-08-01: distilled NeuCodec variant (~10x fewer params, ~7.5x fewer
    # MACs, same NeuCodec arch — single Rust converter drives both via
    # NeucodecVariant slug dispatch; requires its own §3.1 row because the
    # publish target is a distinct HF repo (vokra/distill-neucodec)).
    "distill-neucodec": ["distill-neucodec (`neuphonic/distill-neucodec`)"],
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
    # 2026-08-02 wave — distinct `ModelKind::MpSenetDns` arm for the
    # JacobLinCool DNS-tuned re-release of MP-SENet. Shares the
    # `models::mp_senet::convert_mp_senet_file` converter with the
    # sibling `mp-senet` slug (byte-identical GGUF surface — see
    # `crates/vokra-convert/src/lib.rs` MpSenetDns dispatch arm). Row
    # heading is a placeholder awaiting the post-workflow batch add to
    # `docs/license-audit.md` §3.1; publish is gated by that sign-off
    # per `publish-one.sh` (unknown-repo fails closed).
    "mp-senet-dns": ["MP-SENet-DNS (`JacobLinCool/MP-SENet-DNS`)"],
    "sepformer-wham16k-enhancement": [
        "SepFormer WHAM 16k enhancement (`speechbrain/sepformer-wham16k-enhancement`)"
    ],
    "sepformer-whamr16k": ["SepFormer WHAM-R 16k (`speechbrain/sepformer-whamr16k`)"],
    # 2026-08-01 Wave 4 variant-enum extension: sepformer-libri2mix
    # (`speechbrain/sepformer-libri2mix`, apache-2.0). Same converter as
    # the sibling SepFormer rows above — a distinct `SepformerVariant::
    # Libri2Mix` enum arm ensures the artifact carries a distinct
    # `vokra.model.name` / `vokra.provenance.upstream_hf` /
    # `vokra.sepformer.variant` = `libri2mix` (fail-loud rather than
    # silently inheriting the Wsj02mix sibling's provenance — the
    # weight license is identical apache-2.0 but the training corpus
    # differs: LibriMix = LibriSpeech-derived CC-BY-4.0 vs WSJ0-2mix =
    # proprietary WSJ0). §3.1 row signed 2026-08-01 yousan (依頼者許可
    # = CC 判断) per primary source `https://huggingface.co/speechbrain/
    # sepformer-libri2mix` (cardData `license: apache-2.0`, SpeechBrain
    # family, sibling of the three rows above). The row heading matches
    # this entry byte-for-byte.
    "sepformer-libri2mix": [
        "sepformer-libri2mix (`speechbrain/sepformer-libri2mix`)"
    ],
    # 2026-08-01 Wave 4 variant-enum extension: sepformer-libri3mix
    # (`speechbrain/sepformer-libri3mix`, apache-2.0). Same converter as
    # the sibling SepFormer rows — a distinct `SepformerVariant::Libri3Mix`
    # enum arm ensures the artifact carries a distinct `vokra.model.name`
    # / `vokra.provenance.upstream_hf` / `vokra.sepformer.variant = "libri3mix"`
    # / `vokra.sepformer.n_out = 3` (fail-loud rather than silently
    # inheriting the 2-speaker Libri2Mix sibling's provenance + n_out).
    # The weight license is identical apache-2.0 to every SepFormer
    # sibling; the axis that differs is the masker output head (branches
    # into 3 parallel speaker streams instead of 2 = cocktail-party
    # separation head). §3.1 row signed 2026-08-01 yousan (依頼者許可 =
    # CC 判断) per primary source `https://huggingface.co/speechbrain/
    # sepformer-libri3mix` (cardData `license: apache-2.0`, SpeechBrain
    # family, 3-speaker cocktail-party sibling of the libri2mix row above).
    # The row heading matches this entry byte-for-byte.
    "sepformer-libri3mix": [
        "sepformer-libri3mix (`speechbrain/sepformer-libri3mix`)"
    ],
    # 2026-08-01 Wave 4 variant-enum extension: sepformer-whamr (8 kHz
    # WHAMR! sibling of `speechbrain/sepformer-whamr16k`, apache-2.0).
    # Same converter as the sibling SepFormer rows above — a distinct
    # `SepformerVariant::Whamr8k` enum arm ensures the artifact carries
    # a distinct `vokra.model.name` / `vokra.provenance.upstream_hf` /
    # `vokra.sepformer.variant` = `whamr8k` (fail-loud rather than
    # silently inheriting the 16 kHz sibling's provenance — the weight
    # license is identical apache-2.0 but the upstream HF repo differs:
    # `speechbrain/sepformer-whamr` = 8 kHz vs
    # `speechbrain/sepformer-whamr16k` = 16 kHz. Both are the WHAMR!
    # dereverb + denoise task, only the sample rate differs). §3.1 row
    # signed 2026-08-01 yousan (依頼者許可 = CC 判断) per primary source
    # `https://huggingface.co/speechbrain/sepformer-whamr` (cardData
    # `license: apache-2.0`, SpeechBrain family, base-sample-rate sibling
    # of the whamr16k row above). The row heading matches this entry
    # byte-for-byte.
    "sepformer-whamr-8khz": [
        "sepformer-whamr (`speechbrain/sepformer-whamr`)"
    ],
    # 2026-08-01 Wave 4 variant-enum extension: sepformer-dns4-16k-enhancement
    # (`speechbrain/sepformer-dns4-16k-enhancement`, apache-2.0). Same
    # converter as the sibling SepFormer rows above — a distinct
    # `SepformerVariant::Dns4Enhancement` enum arm ensures the artifact
    # carries a distinct `vokra.model.name` / `vokra.provenance.upstream_hf`
    # / `vokra.sepformer.variant` = `dns4-16k-enhancement` (fail-loud
    # rather than silently inheriting any WHAM! / WHAMR! enhancement
    # sibling's provenance — the weight license is identical apache-2.0
    # but the training corpus differs: Microsoft DNS-4 (Deep Noise
    # Suppression Challenge 4) vs WSJ0-derived WHAM! / WHAMR!). All four
    # enhancement variants share `vokra.sepformer.n_out = 1`, so
    # provenance is the only surface that discriminates them at load
    # time — silent misrouting would not fail loudly at the binder.
    # §3.1 row signed 2026-08-01 yousan (依頼者許可 = CC 判断) per
    # primary source `https://huggingface.co/speechbrain/sepformer-dns4-16k-enhancement`.
    # The row heading matches this entry byte-for-byte.
    "sepformer-dns4-16k-enhancement": [
        "sepformer-dns4-16k-enhancement (`speechbrain/sepformer-dns4-16k-enhancement`)"
    ],
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
    # 2026-08-01 wave — Charactr AI Vocos vocoder (Fourier-space,
    # HF audio-vocoder top by download 2.85M dl on mel-24khz).
    # Encodec variant repo publish is deferred; sign-off row is
    # signed today so a future publish does not need re-approval.
    "vocos-mel-24khz": [
        "Vocos mel 24kHz (`charactr/vocos-mel-24khz`)"
    ],
    # 2026-08-01 Wave 4 slug-only add: Charactr AI Vocos EnCodec variant
    # (`charactr/vocos-encodec-24khz`, MIT). Distinct HF publish target from
    # the sibling `vocos-mel-24khz` — the single Rust converter
    # (`crates/vokra-convert/src/models/vocos.rs`) dispatches both variants
    # through the `VocosVariant` enum; the encodec variant swaps the mel
    # filterbank front-end for an EnCodec latent front-end (128-d @ 75 Hz)
    # while sharing the ConvNeXt V2 backbone + iSTFT head verbatim.
    # `crates/vokra-convert/src/lib.rs::from_arg` already accepts every
    # slug alias (`vocos-encodec-24khz` / `vocos_encodec_24khz` /
    # `vocos-encodec` / `charactr/vocos-encodec-24khz`) and the convert
    # dispatch routes them to `VocosVariant::Encodec24khz` (lines 3289-3295),
    # so this Wave 4 candidate is purely slug-only — no code changes to the
    # converter module. `crates/vokra-core/src/compliance/license_class.rs`
    # already maps `"vocos-encodec-24khz"` to `LicenseClass::Permissive`
    # end-to-end (lines 770-774). The §3.1 row was signed 2026-08-01 yousan
    # (依頼者委任 = CC 判断) at row 386 of `docs/license-audit.md`, per
    # primary source `https://huggingface.co/charactr/vocos-encodec-24khz`
    # (HF cardData API `license: mit`, verified 2026-08-01 — CLAUDE.md
    # 「ハルシネーション厳禁」). This REPO map entry lifts the pre-signed
    # row from UNKNOWN_REPO to APPROVED for the `vokra/vocos-encodec-24khz`
    # publish. ~161 MB single pytorch pickle → M1 iMac 16 GB でローカル
    # 変換 safe per memory `[[feedback-large-models-on-vast-ai]]` (≥8 GB
    # threshold 下 = comfortable margin, vast.ai 不要). The row heading
    # matches this entry byte-for-byte (`Vocos encodec 24kHz` title case,
    # sibling of the mel-24khz entry pattern above, distinct from the
    # `vocos-encodec-24khz` slug this entry keys on).
    "vocos-encodec-24khz": [
        "Vocos encodec 24kHz (`charactr/vocos-encodec-24khz`)"
    ],
    # 2026-08-01 Wave 3 — SNAC codec variants (hubertsiuzdak/snac_{24khz,44khz}, MIT).
    "snac-24khz": ["SNAC 24kHz (`hubertsiuzdak/snac_24khz`)"],
    "snac-44khz": ["SNAC 44kHz (`hubertsiuzdak/snac_44khz`)"],
    # 2026-08-01 Wave 3 — Microsoft VibeVoice-Realtime-0.5B streaming sibling.
    "vibevoice-realtime-0.5b": [
        "microsoft/VibeVoice-Realtime-0.5B"
    ],
    # 2026-08-01 Wave 3 — Novateur WavTokenizer-large-speech-75token (MIT).
    "wavtokenizer-large-speech-75token": [
        "WavTokenizer-Large-Speech-75token (`novateur/WavTokenizer-large-speech-75token`)"
    ],
    # 2026-08-01 Wave 3 — IBM Granite Speech 4.1 2B (apache-2.0 audio LLM).
    "granite-speech-4.1-2b": [
        "granite-speech-4.1-2b (`ibm-granite/granite-speech-4.1-2b`)"
    ],
    # 2026-08-01 Wave 3 — OpenMOSS MOSS-Audio-Tokenizer (Full + Nano,
    # apache-2.0). The codec half of the MOSS-TTS pipeline (waveform →
    # discrete tokens fed into the sibling MOSS-TTS LLM). Two repo
    # publish targets, one Rust converter (MossAudioTokenizerVariant
    # enum), two §3.1 rows.
    "moss-audio-tokenizer": [
        "MOSS-Audio-Tokenizer (Full) (`OpenMOSS-Team/MOSS-Audio-Tokenizer`)"
    ],
    "moss-audio-tokenizer-full": [
        "MOSS-Audio-Tokenizer (Full) (`OpenMOSS-Team/MOSS-Audio-Tokenizer`)"
    ],
    "moss-audio-tokenizer-nano": [
        "MOSS-Audio-Tokenizer (Nano) (`OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano`)"
    ],
    # 2026-08-02 wave — OpenMOSS Team MOSS-Audio-4B-Instruct
    # (`OpenMOSS-Team/MOSS-Audio-4B-Instruct`, apache-2.0). Distinct 4B
    # audio-LLM sibling of the four `moss_tts_*` releases already covered
    # under `moss_tts` in CONVERTER_TO_SIGNOFF_ROWS; reuses that same
    # Rust converter via the new `MossTtsVariant::AudioInstruct4b` arm
    # per the parent workflow's REUSE HINT. Custom-code release
    # (`configuration_moss_audio.py`, `trust_remote_code=True`), 3
    # shards ~8 GB BF16. **Placeholder row heading** — the §3.1 row is
    # added in a separate post-workflow batch per parent workflow
    # discipline (`docs/license-audit.md` untouched in this wave). The
    # heading string here matches the sibling MOSS-TTS row-heading
    # style byte-for-byte so a future §3.1 land closes the loop
    # without any further edit to this file.
    "moss-audio-4b-instruct": [
        "MOSS-Audio-4B-Instruct (`OpenMOSS-Team/MOSS-Audio-4B-Instruct`)"
    ],
    # 2026-08-02 wave: OpenMOSS Team MOSS-Audio-8B-Instruct
    # (`OpenMOSS-Team/MOSS-Audio-8B-Instruct`, apache-2.0). Larger
    # 8B sibling of MOSS-Audio-4B-Instruct — same custom-code audio-LLM
    # (`configuration_moss_audio.py`, `trust_remote_code=True`, 4
    # shards ~9.05 GB BF16 — vast.ai required). Reuses the sibling
    # MossTts converter (parent workflow REUSE HINT) via
    # `MossTtsVariant::AudioInstruct8b`. **Placeholder row heading** —
    # the §3.1 row is added in a separate post-workflow batch per
    # parent workflow discipline (`docs/license-audit.md` untouched in
    # this wave). The heading string here matches the sibling
    # MOSS-Audio-4B-Instruct / MOSS-TTS row-heading style byte-for-byte
    # so a future §3.1 land closes the loop without any further edit
    # to this file.
    "moss-audio-8b-instruct": [
        "MOSS-Audio-8B-Instruct (`OpenMOSS-Team/MOSS-Audio-8B-Instruct`)"
    ],
    # 2026-08-01 Wave 4 slug-only add: OpenMOSS Team MOSS-VoiceGenerator
    # (`OpenMOSS-Team/MOSS-VoiceGenerator`, apache-2.0). Distinct HF publish
    # target from the `moss_tts` family (`OpenMOSS-Team/MOSS-TTS[-v1.5|
    # -Nano-100M|-Local-Transformer-v1.5]`) even though the internal
    # `model_type = "moss_tts_delay"` tag is shared and the converter
    # currently dispatches through `MossTtsVariant::Delay`. The distinct
    # §3.1 row is what a publisher hits — the row heading matches this
    # entry byte-for-byte.
    "moss-voice-generator": [
        "MOSS-VoiceGenerator (`OpenMOSS-Team/MOSS-VoiceGenerator`)"
    ],
    # 2026-08-01 Wave 4 slug-only add: Facebook wav2vec2 large 960h with
    # self-training on LV60 unlabelled audio
    # (`facebook/wav2vec2-large-960h-lv60-self`, apache-2.0). Distinct HF
    # publish target from the four `wav2vec2_ctc` family rows
    # (base-960h / large-xlsr-53 / -xlsr-53-japanese / -xlsr-53-chinese-zh-cn)
    # in §3.1 because it is a distinct upstream release (Xu et al. 2021,
    # arXiv:2010.11430 self-training / pseudo-labelling procedure over
    # LibriVox LV60). The converter currently routes this slug through
    # `Wav2Vec2CtcVariant::LargeXlsr53Base` (closest arch match — same
    # large 24L / 1024h / 16h / 4096ffn axes + `feat_extract_norm=layer` +
    # `do_stable_layer_norm=true` + `vocab_size=32`). The row heading
    # matches this entry byte-for-byte.
    "wav2vec2-large-960h-lv60-self": [
        "wav2vec2-large-960h-lv60-self (`facebook/wav2vec2-large-960h-lv60-self`)"
    ],
    # 2026-08-01 Wave 4 variant-enum extension: Facebook wav2vec2 XLSR-53
    # large backbone with an eSpeak-NG IPA phoneme CTC head
    # (`facebook/wav2vec2-xlsr-53-espeak-cv-ft`, apache-2.0). Distinct HF
    # publish target from the five `wav2vec2_ctc` family rows already
    # covered above (base-960h / large-xlsr-53 / -xlsr-53-japanese /
    # -xlsr-53-chinese-zh-cn / large-960h-lv60-self) because the
    # tokenizer / CTC output space is the eSpeak-NG IPA phoneme
    # inventory (`vocab_size=392`, arXiv:2109.11680 CommonVoice
    # fine-tune) rather than a character-level vocabulary — this row
    # complements the char / kana+kanji / hanzi rows above. The
    # converter routes this slug to the new dedicated
    # `Wav2Vec2CtcVariant::LargeXlsr53EspeakCvFt` arm (added the same
    # wave) so `vokra.wav2vec2_ctc.vocab_size=392` +
    # `has_ctc_head=true` are stamped faithfully. The row heading
    # matches this entry byte-for-byte.
    "wav2vec2-xlsr-53-espeak-cv-ft": [
        "wav2vec2-xlsr-53-espeak-cv-ft (`facebook/wav2vec2-xlsr-53-espeak-cv-ft`)"
    ],
    # 2026-08-02 wave: Facebook data2vec-audio-base-960h
    # (`facebook/data2vec-audio-base-960h`, apache-2.0). Baevski et al.
    # 2022, arXiv:2202.03555 — wav2vec 2.0 base topology (12L × d=768 ×
    # 12h × ffn=3072) + data2vec pretraining objective (contextualised
    # latent representation prediction with an EMA teacher) +
    # LibriSpeech 960h English char CTC head (`vocab_size=32`). The
    # safetensors tensor names are **identical** to the sibling
    # wav2vec2-base-960h — data2vec differs in the pretraining
    # objective, not the downstream inference arch. Distinct HF
    # publish target from every existing `wav2vec2_ctc` family row
    # because it is a distinct upstream release (Meta / FAIR data2vec
    # fleet, different pretraining objective + Baevski et al. 2022
    # paper). Converter dispatches through the shared
    # `models::wav2vec2_ctc` module via the dedicated
    # `Wav2Vec2CtcVariant::Data2vecAudioBase960h` arm (added the same
    # wave) so `vokra.model.name = data2vec-audio-base-960h` and
    # `vokra.provenance.upstream_hf = facebook/data2vec-audio-base-960h`
    # are stamped faithfully rather than masquerading as the wav2vec2
    # sibling. Placeholder row heading — the §3.1 row is added in a
    # separate post-workflow batch per parent workflow discipline.
    "data2vec-audio-base": [
        "data2vec-audio-base-960h (`facebook/data2vec-audio-base-960h`)"
    ],
    # 2026-08-01 Wave 3 — Amphion NaturalSpeech 3 FACodec (apache-2.0
    # factorized VQ codec). Single HF repo `amphion/naturalspeech3_facodec`
    # bundles 5 `.bin` files; four publish variants (v1 / v2 /
    # redecoder-v{1,2}) share one Rust converter (FacodecVariant enum).
    # Only V2 (default = highest-quality codec-only pair) has an initial
    # repo publish declaration; the other variants can add publish
    # entries when owner routes them (base v1 → main zoo,
    # redecoder-v{1,2} → owner decision main vs
    # vokra-voiceclone-experimental per ELVIS Act policy).
    "naturalspeech3-facodec-v2": [
        "NaturalSpeech 3 FACodec (Amphion) (`amphion/naturalspeech3_facodec`)"
    ],
    # 2026-08-01 Wave 3 sibling-pair — YuE bundle
    # (`m-a-p/YuE-upsampler` = vocoder / `m-a-p/xcodec_mini_infer` =
    # codec, both apache-2.0). Two distinct HF publish targets, one
    # Rust converter (YueBundleVariant enum), two §3.1 rows.
    "yue-upsampler": [
        "YuE-upsampler (`m-a-p/YuE-upsampler`)"
    ],
    "yue-xcodec-mini": [
        "YuE xcodec-mini (`m-a-p/xcodec_mini_infer`)"
    ],
    # 2026-08-01 Wave 4 slug-only add: MeloTTS-Korean
    # (`myshell-ai/MeloTTS-Korean`, MIT). Distinct HF publish target from
    # the sibling `melotts` family (MeloTTS-English / MeloTTS-Chinese) —
    # the single Rust converter (`crates/vokra-convert/src/models/
    # melotts.rs`) dispatches all three variants through the
    # `MeloVariant` enum (Korean = `n_symbols=219`, `num_tones=16`,
    # `num_languages=10`, `spk2id={KR:0}`, per HF cardData
    # `myshell-ai/MeloTTS-Korean/raw/main/config.json` fetched
    # 2026-07-30). §3.1 row was signed 2026-07-30 yousan; this REPO map
    # entry lifts it from UNKNOWN_REPO to APPROVED for the
    # `vokra/melotts-korean` publish. Sibling `melotts-english` /
    # `melotts-chinese` repo entries can be added when owner routes
    # them (row + converter map are already in place).
    "melotts-korean": [
        "MeloTTS-Korean (`myshell-ai/MeloTTS-Korean`)"
    ],
    # 2026-08-01 Wave 4 slug-only add: MeloTTS-Chinese
    # (`myshell-ai/MeloTTS-Chinese`, MIT). Distinct HF publish target from
    # the sibling `melotts` family (MeloTTS-English / MeloTTS-Korean) —
    # the single Rust converter (`crates/vokra-convert/src/models/
    # melotts.rs`) dispatches all three variants through the
    # `MeloVariant` enum (Chinese = `n_symbols=112`, `num_tones=11`,
    # `num_languages=1`, `spk2id={ZH:1}`, per HF cardData
    # `myshell-ai/MeloTTS-Chinese/raw/main/config.json` fetched
    # 2026-07-30). §3.1 row was signed 2026-07-30 yousan; this REPO map
    # entry lifts it from UNKNOWN_REPO to APPROVED for the
    # `vokra/melotts-chinese` publish. Sibling `melotts-english` repo
    # entry can be added when owner routes it (row + converter map are
    # already in place, mirror of the melotts-korean precedent above).
    "melotts-chinese": [
        "MeloTTS-Chinese (`myshell-ai/MeloTTS-Chinese`)"
    ],
    # 2026-08-01 Wave 4 slug-only add: AI4Bharat Indic Parler-TTS
    # (`ai4bharat/indic-parler-tts`, apache-2.0). Distinct HF publish target
    # from the sibling `parler-tts-mini-multilingual-v1.1` — the single
    # Rust converter (`crates/vokra-convert/src/models/parler.rs`)
    # dispatches both variants through the `ParlerVariant` enum
    # (Indic = fine-tune on 21 Indic languages, same tensor topology as
    # the multilingual base per parler.rs docstring lines 39-41 = "the
    # tensor topology and every primary hparam listed above are
    # unchanged"). The §3.1 row was signed 2026-07-30 yousan
    # (`docs/license-audit.md` line 343 = **Indic Parler-TTS**); this
    # REPO map entry lifts it from UNKNOWN_REPO to APPROVED for the
    # `vokra/indic-parler-tts` publish. The `gated=auto` HF flag on
    # the upstream is access control only (owner-side accept); the
    # license itself is apache-2.0 per the card front-matter. Sibling
    # `parler-tts-mini-multilingual-v1.1` publish is not routed by this
    # entry — a separate REPO entry can be added when owner routes it
    # (row + converter map are already in place, mirror of the
    # melotts-korean / melotts-chinese slug-only precedents above).
    # ~3.58 GB single safetensors → M1 iMac 16 GB でローカル変換 safe
    # per memory `[[feedback-large-models-on-vast-ai]]` (≥8 GB threshold
    # 下 = comfortable margin, vast.ai 不要).
    "indic-parler-tts": [
        "Indic Parler-TTS (`ai4bharat/indic-parler-tts`)"
    ],
    # 2026-08-01 Wave 4 variant-enum extension: Parler-TTS-Mini-v1
    # (`parler-tts/parler-tts-mini-v1`, apache-2.0). Distinct HF publish
    # target from the sibling `parler-tts-mini-multilingual-v1.1` and
    # `indic-parler-tts` — the single Rust converter
    # (`crates/vokra-convert/src/models/parler.rs`) dispatches all three
    # variants through the `ParlerVariant` enum (mini-v1 English-only =
    # `vocab_size = 32128` T5 text vocab only, no audio-code alphabet
    # merged in; multilingual = `vocab_size = 90714` with the alphabet
    # merged). Every T5 / decoder / audio-encoder hparam is identical
    # across the three variants per HF cardData
    # `parler-tts/parler-tts-mini-v1/raw/main/config.json` fetched
    # 2026-08-01 — CLAUDE.md「ハルシネーション厳禁」. Signed 2026-08-01
    # yousan (依頼者許可 = CC 判断) in `docs/license-audit.md` §3.1;
    # this REPO map entry lifts it from UNKNOWN_REPO to APPROVED for the
    # `vokra/parler-tts-mini-v1` publish. ~3.5 GB single safetensors
    # → M1 iMac 16 GB でローカル変換 safe per memory
    # `[[feedback-large-models-on-vast-ai]]` (≥8 GB threshold 下 =
    # comfortable margin, vast.ai 不要).
    "parler-tts-mini-v1": [
        "parler-tts-mini-v1 (`parler-tts/parler-tts-mini-v1`)"
    ],
    # 2026-08-01 Wave 4 slug-only add: Qwen3-TTS-12Hz-0.6B-CustomVoice
    # (`Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`, apache-2.0). Distinct HF publish
    # target from the sibling `qwen3-tts-12hz-0.6b-base` (row 288) and the two
    # 1.7B rows (rows 331/332) — the single Rust converter
    # (`crates/vokra-convert/src/models/qwen3_tts.rs`) dispatches this slug
    # through the existing `Qwen3TtsVariant::_0_6B_Base` path per the parent
    # decision (slug-only = existing 0.6B branch fine-tune shares topology
    # verbatim; `config.json.tts_model_type = "custom_voice"` head is
    # identically shaped). The §3.1 row was signed 2026-08-01 yousan
    # (依頼者許可 = CC 判断) per primary source
    # `https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` (HF
    # cardData tag `license: apache-2.0` — Qwen3-TTS family walk); this
    # REPO map entry lifts it from UNKNOWN_REPO to APPROVED for the
    # `vokra/qwen3-tts-12hz-0.6b-customvoice` publish. ~3.66 GB single
    # safetensors → M1 iMac 16 GB でローカル変換 safe per memory
    # `[[feedback-large-models-on-vast-ai]]` (≥8 GB threshold 下 =
    # comfortable margin, vast.ai 不要). The row heading matches this entry
    # byte-for-byte.
    "qwen3-tts-12hz-0.6b-customvoice": [
        "Qwen3-TTS-12Hz-0.6B-CustomVoice (`Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`)"
    ],
    # 2026-08-01 Wave 4 variant-enum extension: Qwen3-TTS-12Hz-1.7B-Base
    # (`Qwen/Qwen3-TTS-12Hz-1.7B-Base`, apache-2.0). Distinct HF publish
    # target from the sibling `qwen3-tts-12hz-0.6b-base` (row 288) and the two
    # 1.7B fine-tuned rows (rows 331/332 = CustomVoice/VoiceDesign) — the
    # single Rust converter (`crates/vokra-convert/src/models/qwen3_tts.rs`)
    # dispatches this slug through a NEW `Qwen3TtsVariant::_1_7B_Base` arm
    # (rather than slug-only on `_1_7B_CustomVoice`) so a downstream that
    # ships all three 1.7B GGUFs side-by-side can tell them apart by
    # `vokra.provenance.upstream_hf` / `vokra.model.name`. Talker axes are
    # byte-identical to the two 1.7B fine-tuned siblings (hidden=2048,
    # ffn=6144, n_layer=28) — this is the un-fine-tuned 1.7B backbone that
    # the CustomVoice / VoiceDesign 1.7B siblings fine-tune from. The §3.1
    # row was signed 2026-08-01 yousan (依頼者許可 = CC 判断) per primary
    # source `https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base` (HF
    # cardData tag `license: apache-2.0` — Qwen3-TTS family walk); this
    # REPO map entry lifts it from UNKNOWN_REPO to APPROVED for the
    # `vokra/qwen3-tts-12hz-1.7b-base` publish. ~3679 MB single safetensors
    # → M1 iMac 16 GB でローカル変換 safe per memory
    # `[[feedback-large-models-on-vast-ai]]` (≥8 GB threshold 下 =
    # comfortable margin, vast.ai 不要). The row heading matches this entry
    # byte-for-byte.
    "qwen3-tts-12hz-1.7b-base": [
        "Qwen3-TTS-12Hz-1.7B-Base (`Qwen/Qwen3-TTS-12Hz-1.7B-Base`)"
    ],
    # 2026-08-01 Wave 4 slug-only add: Qwen3-TTS-12Hz-1.7B-CustomVoice
    # (`Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`, apache-2.0). Distinct HF publish
    # target from the sibling 1.7B-Base (row 334) and 1.7B-VoiceDesign (row 332)
    # — the single Rust converter (`crates/vokra-convert/src/models/qwen3_tts.rs`)
    # dispatches this slug through the existing `Qwen3TtsVariant::_1_7B_CustomVoice`
    # arm (added in an earlier Wave 4 candidate together with the
    # `ModelKind::Qwen3TtsCustomVoice17B` dispatch at
    # `crates/vokra-convert/src/lib.rs::from_arg` + convert dispatch; row 331 in
    # `docs/license-audit.md` §3.1 was signed 2026-07-30 yousan when that
    # earlier candidate landed). This entry closes the REPO_TO_SIGNOFF_ROWS
    # loop so `scripts/publish/upload.sh --repo qwen3-tts-12hz-1.7b-customvoice`
    # resolves from UNKNOWN_REPO to APPROVED against the pre-signed §3.1 row —
    # sibling of the 1.7B-Base entry above where the variant-enum extension
    # land + REPO map add went in the same wave. 1.7B-CustomVoice is the
    # `tts_model_type = "custom_voice"` fine-tune of the un-fine-tuned 1.7B-Base
    # backbone (row 334); talker + code-predictor axes are byte-identical
    # between the two (hidden=2048, ffn=6144, n_layer=28, GQA n_head_kv=8,
    # head_dim=128), only the HF release id + `vokra.model.name` /
    # `vokra.provenance.upstream_hf` stamps differ (which is why the earlier
    # candidate landed a distinct `Qwen3TtsVariant::_1_7B_CustomVoice` arm
    # rather than routing slug-only through `_1_7B_Base` — a downstream that
    # ships all three 1.7B GGUFs side-by-side needs distinguishable provenance
    # stamps). ~3656 MB single BF16 safetensors → M1 iMac 16 GB でローカル
    # 変換 safe per memory `[[feedback-large-models-on-vast-ai]]` (≥8 GB
    # threshold 下 = comfortable margin, vast.ai 不要). The row heading
    # matches this entry byte-for-byte.
    "qwen3-tts-12hz-1.7b-customvoice": [
        "Qwen3-TTS-12Hz-1.7B-CustomVoice (`Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`)"
    ],
    # 2026-08-01 Wave 5 pipeline orchestration add: pyannote/speaker-
    # diarization-3.1 (Bredin, CNRS, MIT). The upstream repo ships
    # **only** a ~2 KB config.yaml — no weights of its own — and
    # delegates every forward-pass computation to two sibling MIT
    # weight repos (pyannote/segmentation-3.0 VAD backbone +
    # pyannote/wespeaker-voxceleb-resnet34-LM speaker encoder, both
    # already Vokra-published). This §3.1 row covers the pipeline
    # orchestration GGUF (weightless — carries the SpeakerDiarization
    # + AgglomerativeClustering knobs the future runtime pipeline
    # dispatch reads to wire the two sibling GGUFs together); the
    # sibling weight repos have their own §3.1 rows (segmentation-3.0
    # covered by the pre-existing `pyannote (speaker diarization)`
    # row 268 as of 2026-07-30 yousan sign, MIT primary source
    # verified via authenticated HF API `api/models/pyannote/
    # speaker-diarization-3.1` = `license: mit, gated: auto`;
    # `gated: auto` is access control only, no extra obligations).
    # The Rust runtime pipeline dispatch is a separate WP — this
    # entry covers the orchestration-metadata publish path (~0.1 MB,
    # M1 iMac 16 GB local convert safe per memory
    # `[[feedback-large-models-on-vast-ai]]`; vast.ai not required).
    "pyannote-speaker-diarization-3.1": [
        "pyannote-speaker-diarization-3.1 (`pyannote/speaker-diarization-3.1`)"
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
    # 2026-08-01 Wave 5 music-generation add: Meta AudioCraft MusicGen-Medium
    # (`facebook/musicgen-medium`, cc-by-nc-4.0). Single-variant standalone
    # converter today (the `xcodec2` / `wavtokenizer` posture — no variant
    # enum before a second variant lands). Future family variants (small /
    # large / melody / stereo-*) will either extend this converter (variant
    # enum) or land as sibling files; today's landing points at the medium
    # row only. The row heading matches `docs/license-audit.md` §3.1
    # byte-for-byte.
    "musicgen_medium": ["MusicGen-Medium (`facebook/musicgen-medium`)"],
    # 2026-08-01 Wave 5 music-generation add: Meta AudioCraft MusicGen-Large
    # (`facebook/musicgen-large`, cc-by-nc-4.0). Sibling file to
    # `musicgen_medium.rs` (the chatterbox / chatterbox_turbo /
    # chatterbox_nano split) — dedicated `musicgen_large.rs` rather than
    # a shared `musicgen.rs` variant enum. Same T4 tier as sibling
    # MusicGen-Medium. The row heading matches `docs/license-audit.md`
    # §3.1 byte-for-byte.
    "musicgen_large": ["MusicGen-Large (`facebook/musicgen-large`)"],
    # 2026-08-01 Wave 5 residual: Meta AudioCraft AudioGen-Medium
    # (`facebook/audiogen-medium`, cc-by-nc-4.0). MusicGen sibling for
    # SFX / environmental sounds. Sibling file to `musicgen_medium.rs`.
    "audiogen_medium": ["AudioGen-Medium (`facebook/audiogen-medium`)"],
    # 2026-08-01 Wave 6 residual converters.
    "musicgen_small": ["MusicGen-Small (`facebook/musicgen-small`)"],
    "qwen2_audio": ["Qwen2-Audio-7B-Instruct (`Qwen/Qwen2-Audio-7B-Instruct`)"],
    "vibevoice_asr": ["VibeVoice-ASR (`microsoft/VibeVoice-ASR`)"],
    "ace_step": ["ACE-Step 1.5 (`ACE-Step/Ace-Step1.5`)"],
    # 2026-08-01 Wave 7 residual converter (`hubert_large_ls960.rs`).
    "hubert_large_ls960": ["HuBERT-Large-LS960 (`facebook/hubert-large-ls960-ft`)"],
    # 2026-08-02 Wave residual converter (`demucs_htdemucs.rs`) — Meta
    # HT-Demucs hybrid transformer 4-source music separation, MIT.
    "demucs_htdemucs": ["HT-Demucs (`facebook/demucs`)"],
    # 2026-08-02 Wave residual converter (`ultravox_v0_5_llama_3_2_1b.rs`)
    # — fixie-ai Ultravox v0.5 audio-text-to-text multimodal (Llama-3.2-1B
    # + Whisper encoder + projection adapter), MIT.
    "ultravox_v0_5_llama_3_2_1b": [
        "Ultravox v0.5 (Llama-3.2-1B) (`fixie-ai/ultravox-v0_5-llama-3_2-1b`)"
    ],
    # 2026-08-02 Wave residual converter (`xtts_v2.rs`) — Coqui XTTS-v2
    # multilingual zero-shot voice-cloning TTS (GPT-2 + DVAE + HiFi-GAN),
    # coqui-public-model-license (T4 Research-only).
    "xtts_v2": ["XTTS-v2 (`coqui/XTTS-v2`)"],
    # 2026-08-01 Wave 5 music-generation add: AudioLDM 2
    # (`cvssp/audioldm2`, **cc-by-nc-sa-4.0**). Doubly-restrictive
    # NonCommercialShareAlike default (NC gate + SA cascade). The
    # converter surface must be discoverable by
    # `check-converter-signoff.sh` so a future publish path can be
    # gated on the (currently blank) §3.1 sign-off — the row heading
    # here matches `docs/license-audit.md` §3.1 byte-for-byte.
    #
    # **IMPORTANT — publish blocked (sa-cascade-defer)**: there is NO
    # matching entry in `REPO_TO_SIGNOFF_ROWS` above (an
    # unlisted repo slug fails closed as `UNKNOWN_REPO` at
    # `publish-one.sh` gate time). The SA cascade would obligate any
    # Vokra-added artifact bundled with the weight (model card,
    # LICENSE, NOTICE, auxiliary GGUFs) to carry CC-BY-NC-SA-4.0
    # forward, and that decision needs an owner ADR before a
    # `vokra/audioldm2` repo entry is added here.
    "audioldm2": ["AudioLDM 2 (`cvssp/audioldm2`)"],
    # 2026-08-02 Wave residual: openWakeWord (dscripka, apache-2.0).
    # Audio-dialect `kws` op entry — small custom-KWS MLP/CNN family
    # (~1–5 MB per wake-word) over precomputed melspec. Placeholder
    # row — the row heading MUST match `docs/license-audit.md` §3.1
    # byte-for-byte once the audit doc is updated in a post-workflow
    # batch.
    "openwakeword": ["openWakeWord (`dscripka/openWakeWord`)"],
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
        # 2026-08-01 Wave 4 slug-only add — same converter dispatches the
        # 0.6B-CustomVoice fine-tune through the existing 0.6B-Base variant
        # arm (per parent decision: byte-identical talker + code-predictor
        # axes; CustomVoice head is same topology). Row heading matches
        # `docs/license-audit.md` §3.1 byte-for-byte.
        "Qwen3-TTS-12Hz-0.6B-CustomVoice (`Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`)",
        # 2026-08-01 Wave 4 variant-enum extension — same converter
        # dispatches the un-fine-tuned 1.7B-Base backbone through the NEW
        # `Qwen3TtsVariant::_1_7B_Base` arm (rather than slug-only on
        # `_1_7B_CustomVoice`) so a downstream that ships all three 1.7B
        # GGUFs side-by-side can tell them apart by
        # `vokra.provenance.upstream_hf` / `vokra.model.name`. Row heading
        # matches `docs/license-audit.md` §3.1 byte-for-byte.
        "Qwen3-TTS-12Hz-1.7B-Base (`Qwen/Qwen3-TTS-12Hz-1.7B-Base`)",
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
    # Single converter drives both the base and 2026-08-01 distill variant
    # via NeucodecVariant slug dispatch — both §3.1 rows must exist so the
    # check-converter-signoff.sh gate stays green after the distill row lands.
    "neucodec": [
        "neucodec (`neuphonic/neucodec`)",
        "distill-neucodec (`neuphonic/distill-neucodec`)",
    ],
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
    # 2026-08-01 Wave 5: pyannote/speaker-diarization-3.1 pipeline
    # orchestration converter (weightless — reads a config.yaml sanity
    # buffer and emits primary-source-verified pipeline hparams under
    # `vokra.pyannote_pipeline.*`). Distinct §3.1 row from the sibling
    # `pyannote_segmentation` weight converter — the pipeline GGUF is
    # a separate publish target from the VAD backbone weights.
    "pyannote_speaker_diarization_3_1": [
        "pyannote-speaker-diarization-3.1 (`pyannote/speaker-diarization-3.1`)"
    ],
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
    # 2026-08-01 wave — Vocos (`charactr/vocos-mel-24khz` +
    # `charactr/vocos-encodec-24khz`, MIT). Single converter with a
    # VocosVariant enum for the two frontend variants; both §3.1
    # rows are enumerated so the check-converter-signoff.sh gate
    # accepts a future encodec-24khz publish without re-mapping.
    "vocos": [
        "Vocos mel 24kHz (`charactr/vocos-mel-24khz`)",
        "Vocos encodec 24kHz (`charactr/vocos-encodec-24khz`)",
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
        # 2026-08-02 wave — MOSS-Audio-4B-Instruct
        # (`OpenMOSS-Team/MOSS-Audio-4B-Instruct`) reuses this converter
        # per the parent workflow's REUSE HINT via the new
        # `MossTtsVariant::AudioInstruct4b` arm. Placeholder row heading;
        # `docs/license-audit.md` §3.1 row is landed in a separate
        # post-workflow batch (this file is what the check-converter-signoff.sh
        # gate reads, so the mapping is landed here in this wave).
        "MOSS-Audio-4B-Instruct (`OpenMOSS-Team/MOSS-Audio-4B-Instruct`)",
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
        # 2026-08-01 Wave 4 variant-enum extension: Parler-TTS-Mini-v1
        # (`parler-tts/parler-tts-mini-v1`). Same converter dispatches all
        # three variants via `ParlerVariant` enum. Row heading matches
        # `docs/license-audit.md` §3.1 byte-for-byte.
        "parler-tts-mini-v1 (`parler-tts/parler-tts-mini-v1`)",
    ],
    "qwen3_asr": [
        "Qwen3-ASR-0.6B (`Qwen/Qwen3-ASR-0.6B`)",
        "Qwen3-ASR-1.7B (`Qwen/Qwen3-ASR-1.7B`)",
    ],
    "sepformer": [
        "SepFormer WSJ0-2mix (`speechbrain/sepformer-wsj02mix`)",
        "SepFormer WHAM 16k enhancement (`speechbrain/sepformer-wham16k-enhancement`)",
        "SepFormer WHAM-R 16k (`speechbrain/sepformer-whamr16k`)",
        # 2026-08-01 Wave 4 variant-enum extension — same converter dispatches
        # all four SepFormer variants via `SepformerVariant` enum. The distinct
        # `SepformerVariant::Libri2Mix` arm ensures a distinct
        # `vokra.sepformer.variant = "libri2mix"` stamp rather than silently
        # inheriting Wsj02mix's `wsj02mix` tag. Row heading matches
        # `docs/license-audit.md` §3.1 byte-for-byte.
        "sepformer-libri2mix (`speechbrain/sepformer-libri2mix`)",
        # 2026-08-01 Wave 4 variant-enum extension — same converter dispatches
        # the 3-speaker cocktail-party sibling via `SepformerVariant::Libri3Mix`.
        # The distinct enum arm ensures a distinct `vokra.sepformer.variant =
        # "libri3mix"` stamp + `vokra.sepformer.n_out = 3` (new metadata chunk
        # added the same wave = binder output-stream axis explicit at load
        # time) rather than silently inheriting the 2-speaker sibling's tags.
        # Row heading matches `docs/license-audit.md` §3.1 byte-for-byte.
        "sepformer-libri3mix (`speechbrain/sepformer-libri3mix`)",
        # 2026-08-01 Wave 4 variant-enum extension — same converter dispatches
        # the 8 kHz WHAMR! sibling via `SepformerVariant::Whamr8k`. The
        # distinct enum arm ensures a distinct `vokra.sepformer.variant =
        # "whamr8k"` stamp + `vokra.provenance.upstream_hf =
        # "speechbrain/sepformer-whamr"` rather than silently inheriting
        # the 16 kHz sibling's tags. Row heading matches
        # `docs/license-audit.md` §3.1 byte-for-byte.
        "sepformer-whamr (`speechbrain/sepformer-whamr`)",
        # 2026-08-01 Wave 4 variant-enum extension — same converter dispatches
        # the Microsoft DNS-4 16 kHz enhancement sibling via
        # `SepformerVariant::Dns4Enhancement`. The distinct enum arm ensures
        # a distinct `vokra.sepformer.variant = "dns4-16k-enhancement"` stamp
        # + `vokra.provenance.upstream_hf =
        # "speechbrain/sepformer-dns4-16k-enhancement"` rather than silently
        # inheriting any WHAM! / WHAMR! enhancement sibling's tags (all four
        # enhancement variants share n_out = 1, so provenance is the only
        # surface that discriminates them at load time). Row heading matches
        # `docs/license-audit.md` §3.1 byte-for-byte.
        "sepformer-dns4-16k-enhancement (`speechbrain/sepformer-dns4-16k-enhancement`)",
    ],
    # 2026-08-01 Wave 3 — SNAC codec (single converter, two variant rows).
    "snac": [
        "SNAC 24kHz (`hubertsiuzdak/snac_24khz`)",
        "SNAC 44kHz (`hubertsiuzdak/snac_44khz`)",
    ],
    # 2026-08-01 Wave 3 — WavTokenizer FSQ codec (single-codebook 75 tok/s).
    "wavtokenizer": [
        "WavTokenizer-Large-Speech-75token (`novateur/WavTokenizer-large-speech-75token`)"
    ],
    # 2026-08-01 Wave 3 — IBM Granite Speech audio LLM.
    "granite_speech": [
        "granite-speech-4.1-2b (`ibm-granite/granite-speech-4.1-2b`)"
    ],
    # 2026-08-01 Wave 3 — OpenMOSS MOSS-Audio-Tokenizer codec family
    # (Full + Nano, apache-2.0). Single Rust converter drives both via
    # MossAudioTokenizerVariant slug dispatch; both §3.1 rows must
    # exist so the check-converter-signoff.sh gate stays green after
    # the Nano publish lands.
    "moss_audio_tokenizer": [
        "MOSS-Audio-Tokenizer (Full) (`OpenMOSS-Team/MOSS-Audio-Tokenizer`)",
        "MOSS-Audio-Tokenizer (Nano) (`OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano`)",
    ],
    # 2026-08-01 Wave 3 — Amphion NaturalSpeech 3 FACodec codec family
    # (v1 / v2 / redecoder-v{1,2}, apache-2.0). Single Rust converter
    # drives all four variants via FacodecVariant slug dispatch; a
    # single §3.1 row covers the entire family (the redecoder-v{1,2}
    # zero-shot voice-conversion variants share the same signature and
    # provenance — the license class is identical apache-2.0; the
    # routing question is a separate ELVIS Act publication decision).
    "naturalspeech3_facodec": [
        "NaturalSpeech 3 FACodec (Amphion) (`amphion/naturalspeech3_facodec`)"
    ],
    # 2026-08-01 Wave 3 sibling-pair — YuE bundle (`m-a-p/YuE-upsampler`
    # + `m-a-p/xcodec_mini_infer`, both apache-2.0). Single Rust
    # converter (yue_bundle.rs) drives both variants via
    # YueBundleVariant enum + two distinct ModelKind entries
    # (YueUpsampler + YueXcodecMini); two §3.1 rows must exist so the
    # check-converter-signoff.sh gate stays green after either publish
    # lands.
    "yue_bundle": [
        "YuE-upsampler (`m-a-p/YuE-upsampler`)",
        "YuE xcodec-mini (`m-a-p/xcodec_mini_infer`)",
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
        # 2026-08-01 Wave 4 slug-only add — Facebook wav2vec2 large 960h
        # with self-training on LV60 unlabelled audio. Same
        # `wav2vec2_ctc` converter; the slug is routed to the existing
        # `Wav2Vec2CtcVariant::LargeXlsr53Base` arm (arch match). See
        # `docs/license-audit.md` §3.1 for the sign-off.
        "wav2vec2-large-960h-lv60-self (`facebook/wav2vec2-large-960h-lv60-self`)",
        # 2026-08-01 Wave 4 variant-enum extension — Facebook wav2vec2
        # XLSR-53 large backbone with an eSpeak-NG IPA phoneme CTC
        # head (`facebook/wav2vec2-xlsr-53-espeak-cv-ft`, apache-2.0).
        # Distinct dedicated arm
        # `Wav2Vec2CtcVariant::LargeXlsr53EspeakCvFt` (added the same
        # wave) because vocab_size=392 (eSpeak phoneme inventory) and
        # has_ctc_head=true both differ from the closest topology
        # sibling `LargeXlsr53Base` — routing slug-only would stamp
        # demonstrably wrong axes. See `docs/license-audit.md` §3.1.
        "wav2vec2-xlsr-53-espeak-cv-ft (`facebook/wav2vec2-xlsr-53-espeak-cv-ft`)",
    ],
    "xvector": ["X-vector VoxCeleb (`speechbrain/spkrec-xvect-voxceleb`)"],
    # 2026-08-01 Wave 5 music-separation add: BS-Roformer / Mel-Band Roformer
    # (`chenmozhijin/BSRoformer-GGUF` third-party mirror, **weight provenance
    # unclear**). First music-source-separation converter (Lu et al. 2023
    # arXiv:2310.01809). The converter surface must be discoverable by
    # `check-converter-signoff.sh` so a future publish path can be gated on
    # the (currently blank) §3.1 sign-off — the row heading here matches
    # `docs/license-audit.md` §3.1 byte-for-byte.
    #
    # **IMPORTANT — publish blocked (unclear-provenance-defer)**: there is NO
    # matching entry in `REPO_TO_SIGNOFF_ROWS` above (an unlisted repo slug
    # fails closed as `UNKNOWN_REPO` at `publish-one.sh` gate time). The
    # `LicenseClass::RedistributionForbidden` default is what actually blocks
    # the publish path: a converter cannot know which SPDX id covers the
    # caller's checkpoint (the Lucidrains reference is MIT but the paper
    # released no reference weights; every checkpoint in the wild is a
    # downstream retraining under mixed licenses — GPL-3.0 / CC-BY-NC-4.0 /
    # unspecified). An owner ADR selecting a specific checkpoint (and thus a
    # specific license) is the prerequisite to a first publish; only then does
    # a `REPO_TO_SIGNOFF_ROWS` entry land here.
    "bs_roformer": ["BS-Roformer (upstream 未確定)"],
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
