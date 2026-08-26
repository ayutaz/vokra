#!/usr/bin/env python3
"""Audit Vokra's public Hugging Face GGUFs against Mac runtime coverage.

This is a read-only inventory tool. It deliberately distinguishes source-tree
reachability from real-artifact verification: an architecture can have a full
CPU/Metal code path while an old public GGUF still needs replacement metadata,
tokenizer data, or a fresh parity run.

Run through the repository's Python policy:

    uv run --no-project --python 3.12 python tools/audit/hf_mac_coverage.py
    uv run --no-project --python 3.12 python tools/audit/hf_mac_coverage.py --format tsv
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import json
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Iterable


API_ROOT = "https://huggingface.co/api/models"
HF_ROOT = "https://huggingface.co"
USER_AGENT = "vokra-hf-mac-coverage/1.0"

# These arches have an ARCH_* route but not a public-artifact-complete CPU
# runtime. MAGNeT/MelodyFlow are loud partial diagnostics; CSM still binds a
# synthesized model bridge instead of released weights; SBV2 has a current
# strict converter/runtime, but its live artifact and production G2P remain
# incomplete as recorded below.
# RMVPE now has an exact code route, but its live artifact remains blocked
# below by invalid provenance. Pyannote is complete: its exact historical
# 54-F32-tensor artifact receives a narrowly scoped metadata repair and then
# executes the default-on CPU/Metal route.
# Subtracting only BOUND_ARCHES would therefore overstate CPU coverage.
ROUTED_PARTIAL_ARCHES = {
    "csm",
    "magnet_small_10secs",
    "magnet_medium_30secs",
    "melodyflow_t24_30secs",
    "sbv2",
}

# Artifact-specific failures that cannot be inferred from the repo-card arch.
# Keep these fail-closed even when sibling checkpoints sharing the arch have a
# complete runtime. The value is the actionable public-file verdict.
PUBLIC_ARTIFACT_CPU_BLOCKERS = {
    "vokra/qwen3-asr-0.6b": (
        "partial",
        "the live dense GGUF predates the executable Qwen3-ASR conversion "
        "contract: it lacks the three exact audio-execution metadata keys and "
        "all five byte-authenticated tokenizer/chat/generation sidecars. The "
        "native source route is complete on CPU/Metal, but deliberately rejects "
        "this historical artifact pending an authorized gated replacement",
    ),
    "vokra/qwen3-asr-1.7b": (
        "partial",
        "the live dense GGUF predates the executable Qwen3-ASR conversion "
        "contract: it lacks the three exact audio-execution metadata keys and "
        "all five byte-authenticated tokenizer/chat/generation sidecars. The "
        "native source route is complete on CPU/Metal, but deliberately rejects "
        "this historical artifact pending an authorized gated replacement",
    ),
    "vokra/speecht5-tts": (
        "no-runtime-binder",
        "the live 393-F32-tensor GGUF predates the strict pinned SpeechT5 "
        "conversion contract and contains neither the exact 79-piece "
        "spm_char.model plus two fixed added tokens nor the fixed "
        "source/manifest/runtime "
        "metadata. It cannot execute text TTS and must remain a loud public "
        "artifact boundary pending the native runtime and an authorized "
        "gated replacement",
    ),
    "vokra/sbv2-v2-jp-extra-base": (
        "partial",
        "the live 1,264-tensor GGUF predates the strict SBV2 converter and keeps "
        "raw dec/dp/enc_p/enc_q/flow/sdp upstream names without the required "
        "runtime metadata. Native CPU parity exists for a corrected conversion, "
        "but the public artifact and production Japanese G2P remain explicit "
        "boundaries",
    ),
    "vokra/bicodec": (
        "no-runtime-binder",
        "the live 840-tensor GGUF has no runtime binder and stamps Apache-2.0/"
        "permissive provenance, while the pinned SparkAudio weight audit classifies "
        "the released weights as CC-BY-NC-SA-4.0 research-only. Refuse the public "
        "artifact pending a gated provenance-correct replacement",
    ),
    "vokra/xy-tokenizer": (
        "no-runtime-binder",
        "the live GGUF has an empty tensor manifest and contains metadata only, so "
        "there is no checkpoint payload for a CPU or Metal runtime to bind",
    ),
    "vokra/canary-1b-v2": (
        "partial",
        "the two live GGUF filenames are identical 688-tensor copies of the "
        "24-layer timestamp auxiliary CTC checkpoint, while their metadata claims "
        "the 32-layer/8-layer AED model; they contain no Transformer decoder or "
        "tokenizer. The pinned 6.36 GB .nemo carries the correct 3.85 GB "
        "./model_weights.ckpt separately, and the runtime must reject the live "
        "artifact pending a corrected complete conversion",
    ),
    "vokra/canary-1b-flash": (
        "partial",
        "the live 1,292-tensor GGUF contains only the FastConformer encoder; "
        "the strict complete CPU/Metal runtime requires the released four-layer AED "
        "decoder, 5,248-piece aggregate tokenizer and authenticated 1,374-tensor "
        "manifest prepared from the pinned .nemo checkpoint",
    ),
    "vokra/reazonspeech-nemo-v2": (
        "partial",
        "the live 965-tensor GGUF has the authenticated complete F32 encoder/RNN-T "
        "manifest, but predates the embedded 3,000-piece tokenizer vocabulary and "
        "the pinned runtime-axis metadata. Native CPU/Metal token inference is bound; "
        "CLI/C ASR text output rejects the legacy artifact before weight decode until "
        "an authorized gated replacement is converted from the fixed NeMo revision",
    ),
    "vokra/moss-audio-tokenizer-nano": (
        "partial",
        "the exact 374-tensor Nano checkpoint has a native CPU/Metal decode route, but "
        "the live GGUF incorrectly stamps Full name, variant and upstream provenance; "
        "the manifest-scoped legacy repair remains visible pending an authorized "
        "correctly stamped replacement",
    ),
    "vokra/moss-audio-4b-instruct": (
        "no-runtime-binder",
        "the public file shares the broad moss_tts arch tag but is not one of the "
        "strictly authenticated Nano, Base, v1.5, Local or VoiceGenerator releases; "
        "the CLI refuses family-shared topology inference",
    ),
    "vokra/moss-audio-8b-instruct": (
        "no-runtime-binder",
        "the public file shares the broad moss_tts arch tag but is not one of the "
        "strictly authenticated Nano, Base, v1.5, Local or VoiceGenerator releases; "
        "the CLI refuses family-shared topology inference",
    ),
    "vokra/moss-tts-local-transformer-v1.5": (
        "partial",
        "the fixed-revision Local binder and CPU/Metal generation source are present, "
        "but the CLI intentionally has no Local task until the required 48 kHz stereo "
        "MOSS Audio Tokenizer v2 companion is strictly bound and real-weight verified",
    ),
    "vokra/audiogen-medium": (
        "partial",
        "the authenticated legacy LM-only artifact has a native CPU/Metal raw-code route, "
        "but contains neither its T5-large conditioner nor the official 16 kHz EnCodec "
        "decoder; both remain explicit companion boundaries",
    ),
    "vokra/musicgen-medium": (
        "partial",
        "the public AudioCraft-layout artifact has native CPU/Metal LM generation but "
        "contains neither T5-base nor EnCodec; explicit authenticated companions are "
        "required before token-id-to-waveform execution",
    ),
    "vokra/musicgen-large": (
        "partial",
        "the public AudioCraft-layout artifact has native CPU/Metal LM generation but "
        "contains neither T5-base nor EnCodec; explicit authenticated companions are "
        "required before token-id-to-waveform execution",
    ),
    "vokra/nsnet2": (
        "partial",
        "the exact public tensor header is runtime-compatible, but the live GGUF "
        "mis-stamps the released model as MIT/permissive; Microsoft's fixed source "
        "revision assigns released non-code content to CC-BY-4.0, so corrected "
        "provenance and attribution require an authorized gated replacement",
    ),
    "vokra/conv-tasnet-libri1mix": (
        "partial",
        "the public GGUF stamps conflicting CC-BY-SA-3.0 provenance and the old "
        "kernel=16/stride=8 topology; the pinned official checkpoint is "
        "kernel=32/stride=16 and upstream license declarations remain unresolved",
    ),
    "vokra/mms-1b-all-base": (
        "partial",
        "the public 8.9 MB file is an adapter only, not the verified 1B "
        "backbone, and its topology/license stamp is incompatible",
    ),
    "vokra/voice-gender-classifier": (
        "partial",
        "the public file is mis-stamped as canonical SpeechBrain ECAPA but carries a distinct "
        "202-tensor conv1/layer1-3/attention/fc6/fc7 gender-classifier topology; the strict "
        "200-tensor speaker binder refuses it instead of misrouting",
    ),
    "vokra/speechbrain-spkrec-ecapa-voxceleb": (
        "partial",
        "the 83,239,904-byte public restamped GGUF has tensor data out of bounds at "
        "mfa.conv.conv.weight; the native 200-tensor runtime and strict replacement "
        "converter pass parity, but the live artifact itself must be replaced",
    ),
    "vokra/wespeaker": (
        "partial",
        "the public 219-tensor file contains the supported official ResNet34-LM topology, "
        "but stamps its CC-BY-4.0 weights as apache-2.0/permissive and omits required "
        "attribution; the strict native binder refuses the provenance mismatch pending "
        "an authorized gated replacement",
    ),
    "vokra/lang-id-voxlingua107": (
        "partial",
        "the public GGUF contains only 200 ECAPA embedding tensors: it omits the official "
        "XVector classifier, ordered 107-label encoder and the variant-specific 60-mel/256-d "
        "contract; the strict converter/runtime now require and execute the official prepared "
        "checkpoint, but the live artifact needs an authorized gated replacement",
    ),
    "vokra/rmvpe": (
        "partial",
        "the public 181,010,688-byte GGUF at revision "
        "3eb5fa8946f1074ba3959074c5cde95ec22b8c91 carries the supported fixed "
        "E2E0 tensor manifest, but mis-stamps yxlllc/RMVPE as MIT/permissive even "
        "though that exact repository declares no license; the strict CPU/Metal "
        "loader rejects it pending an authorized provenance-corrected replacement",
    ),
}

# Conservative code-path inventory. Every entry must also be a full routed
# CPU arch; the unit test and live main path enforce that. Artifact-specific
# real-weight parity remains a separate ledger column outside this script.
METAL_CODE_ARCHES = {
    "audiobox-aesthetics",
    "audioseal_real_weight",
    "ast",
    "bert_base",
    "bigvgan",
    "campplus",
    "canary",
    "canary-1b-flash",
    "conv_tasnet",
    "crisper-whisper",
    "dac",
    "data2vec_audio",
    "deberta_v2",
    "deberta_v3",
    "deepfake_detection",
    "denoise",
    "distil-whisper",
    "dnsmos",
    "ecapa_tdnn",
    "emotion2vec",
    "facebook_denoiser",
    "facodec",
    "fcpe",
    "firered_vad",
    "focalcodec",
    "frcrn",
    "funcodec",
    "speechtokenizer",
    "fsmn-vad",
    "hifigan_vocoder",
    "hubert",
    "kokoro-82m-istftnet",
    "kotoba-whisper",
    "lang_id_ecapa",
    "melotts",
    "metricgan_plus",
    "mimi",
    "miocodec",
    "mossformer2_ss_16k",
    "moss_tts",
    "moss_audio_tokenizer",
    "musicgen",
    "mp_senet",
    "moshi",
    "moonshine",
    "nkf_aec",
    "neucodec",
    "nemotron_asr_streaming",
    "nisqa_v2_weight",
    "nsnet2",
    "parakeet-ctc",
    "parakeet-tdt",
    "parakeet-tdt-1_1b",
    "piper-plus-mb-istft-vits2",
    "pyannote-segmentation",
    "pyannote-speaker-diarization",
    "qwen3_asr",
    "reazonspeech_nemo_v2",
    "rnnoise",
    "rmvpe",
    "silero-vad",
    "snac",
    "sepformer",
    "smart_turn",
    "speecht5_hifigan",
    "ten_vad",
    "tiger_separator",
    "titanet-large",
    "utmos",
    "vocos",
    "voxtral",
    "wav2vec2_ctc",
    "wavtokenizer",
    "whisper",
    "whisper-medusa-v1",
    "wespeaker",
    "xcodec2",
    "xvector",
    "yue_upsampler",
}

ARCH_ROW = re.compile(
    r"^\|\s*Architecture\s*\|\s*(?:`(?P<quoted>[^`]+)`|(?P<plain>[^|]+?))\s*\|\s*$",
    re.MULTILINE,
)
ARCH_CONST = re.compile(r'^const ARCH_[A-Z0-9_]+: &str = "([^"]+)";', re.MULTILINE)
BOUND_ROW = re.compile(r'\barch:\s*"([^"]+)"')


@dataclasses.dataclass(frozen=True)
class RepoRecord:
    repo: str
    revision: str
    gguf_files: tuple[str, ...]
    architecture: str | None


@dataclasses.dataclass(frozen=True)
class Coverage:
    cpu_code: str
    metal_code: str
    reason: str


def _request_text(url: str, retries: int = 4) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.read().decode("utf-8")
        except urllib.error.HTTPError as error:
            if error.code not in {429, 500, 502, 503, 504} or attempt + 1 == retries:
                raise
        except urllib.error.URLError:
            if attempt + 1 == retries:
                raise
        time.sleep(0.5 * (2**attempt))
    raise AssertionError("retry loop must return or raise")


def parse_readme_architecture(readme: str) -> str | None:
    match = ARCH_ROW.search(readme)
    if match is None:
        return None
    return (match.group("quoted") or match.group("plain")).strip()


def parse_engine_arches(source: str) -> tuple[set[str], set[str]]:
    routed = set(ARCH_CONST.findall(source))
    marker = "const BOUND_ARCHES"
    start = source.find(marker)
    if start < 0:
        raise ValueError("engine source has no const BOUND_ARCHES registry")
    end = source.find("\n];", start)
    if end < 0:
        raise ValueError("engine source has an unterminated const BOUND_ARCHES registry")
    bound = set(BOUND_ROW.findall(source[start:end]))
    if not routed:
        raise ValueError("engine source yielded zero routed ARCH_* constants")
    if not bound:
        raise ValueError("engine source yielded zero BOUND_ARCHES rows")
    return routed, bound


def classify(
    record: RepoRecord, routed: set[str], bound: set[str]
) -> Coverage:
    if not record.gguf_files:
        return Coverage("not-artifact", "not-artifact", "public repo has no GGUF")
    architecture = record.architecture
    if architecture is None:
        return Coverage(
            "unknown",
            "unknown",
            "GGUF repo card has no machine-readable Architecture row",
        )
    public_blocker = PUBLIC_ARTIFACT_CPU_BLOCKERS.get(record.repo)
    if public_blocker is not None:
        cpu_code, reason = public_blocker
        return Coverage(cpu_code, "blocked-by-cpu", reason)
    if architecture in ROUTED_PARTIAL_ARCHES or architecture in bound:
        return Coverage(
            "partial",
            "blocked-by-cpu",
            "route/binder exists, but the released-artifact CPU forward is incomplete",
        )
    if architecture not in routed:
        return Coverage(
            "no-runtime-binder",
            "blocked-by-cpu",
            "converter/public artifact has no complete CLI runtime binder",
        )
    if architecture in METAL_CODE_ARCHES:
        return Coverage(
            "full",
            "full",
            "complete CPU route and declared Metal code path; artifact parity is separate",
        )
    return Coverage(
        "full",
        "cpu-only",
        "complete CPU route; non-CPU selection is an explicit unsupported error",
    )


def fetch_records(org: str, workers: int) -> list[RepoRecord]:
    payload = json.loads(_request_text(f"{API_ROOT}?author={org}&limit=1000&full=true"))
    if not isinstance(payload, list):
        raise ValueError("Hugging Face organization query did not return a list")

    summaries: list[tuple[str, str, tuple[str, ...]]] = []
    for item in payload:
        repo = item["id"]
        revision = item.get("sha") or ""
        gguf_files = tuple(
            sibling["rfilename"]
            for sibling in item.get("siblings", [])
            if sibling["rfilename"].endswith(".gguf")
        )
        summaries.append((repo, revision, gguf_files))

    def resolve(summary: tuple[str, str, tuple[str, ...]]) -> RepoRecord:
        repo, revision, gguf_files = summary
        if not gguf_files:
            return RepoRecord(repo, revision, gguf_files, None)
        readme = _request_text(f"{HF_ROOT}/{repo}/raw/{revision}/README.md")
        return RepoRecord(repo, revision, gguf_files, parse_readme_architecture(readme))

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        records = list(executor.map(resolve, summaries))
    return sorted(records, key=lambda record: record.repo)


def _summary_lines(
    records: Iterable[RepoRecord], routed: set[str], bound: set[str]
) -> list[str]:
    records = list(records)
    coverage = [classify(record, routed, bound) for record in records]
    cpu = Counter(item.cpu_code for item in coverage)
    metal = Counter(item.metal_code for item in coverage)
    return [
        f"public_repos={len(records)}",
        f"gguf_repos={sum(bool(record.gguf_files) for record in records)}",
        f"gguf_files={sum(len(record.gguf_files) for record in records)}",
        "cpu_code=" + ",".join(f"{key}:{cpu[key]}" for key in sorted(cpu)),
        "metal_code=" + ",".join(f"{key}:{metal[key]}" for key in sorted(metal)),
        "note=live-artifact reachability; full code routes can remain partial on "
        "artifact-specific blockers, and real-weight parity is a separate gate",
    ]


def render_tsv(
    records: Iterable[RepoRecord], routed: set[str], bound: set[str]
) -> str:
    rows = ["repo\trevision\tgguf_files\tarchitecture\tcpu_code\tmetal_code\treason"]
    for record in records:
        coverage = classify(record, routed, bound)
        rows.append(
            "\t".join(
                [
                    record.repo,
                    record.revision,
                    str(len(record.gguf_files)),
                    record.architecture or "",
                    coverage.cpu_code,
                    coverage.metal_code,
                    coverage.reason,
                ]
            )
        )
    return "\n".join(rows)


def cpu_only_repositories(
    records: Iterable[RepoRecord], routed: set[str], bound: set[str]
) -> list[str]:
    """Return live artifacts that claim complete CPU but lack Metal code."""
    return sorted(
        record.repo
        for record in records
        if classify(record, routed, bound).metal_code == "cpu-only"
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--org", default="vokra")
    parser.add_argument(
        "--engine",
        type=Path,
        default=root / "crates/vokra-cli/src/engine.rs",
    )
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--format", choices=("summary", "tsv"), default="summary")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.workers < 1 or args.workers > 16:
        raise ValueError("--workers must be in 1..16")
    routed, bound = parse_engine_arches(args.engine.read_text(encoding="utf-8"))
    invalid_metal = METAL_CODE_ARCHES - (routed - ROUTED_PARTIAL_ARCHES)
    if invalid_metal:
        raise ValueError(f"Metal audit registry names non-runnable arches: {sorted(invalid_metal)}")
    records = fetch_records(args.org, args.workers)
    missing_arch = [
        record.repo
        for record in records
        if record.gguf_files and not record.architecture
    ]
    if missing_arch:
        raise ValueError(f"GGUF repos without an Architecture card row: {missing_arch}")
    cpu_only = cpu_only_repositories(records, routed, bound)
    if cpu_only:
        raise ValueError(
            "CPU-complete public repos without complete Metal code paths: "
            f"{cpu_only}"
        )
    if args.format == "tsv":
        print(render_tsv(records, routed, bound))
    else:
        print("\n".join(_summary_lines(records, routed, bound)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
