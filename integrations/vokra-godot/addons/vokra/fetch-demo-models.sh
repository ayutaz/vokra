#!/usr/bin/env bash
# fetch-demo-models.sh — downloads MIT-only demo weights into the Godot
# demo project's res://models/ directory (M3-11 / ADR-0011 §D5).
#
# =============================================================================
#  Godot mirror of the Unity Samples~ fetch script
#  (bindings/unity/com.vokra.unity/Samples~/VadAsrTts/Scripts/fetch-demo-models.sh)
#
#  The two scripts are structurally identical (curl/wget probe, env override,
#  license manifest, --help / --license flags) and share the same MIT-only
#  red-line policy. Divergence points, all documented inline below:
#    (a) destination path — Godot places models at res://models/ (project
#        root + "models/"), the Unity mirror places them at
#        StreamingAssets/models/;
#    (b) filename set — the Godot demo scaffolds reference
#        `whisper-base.gguf` (asr_demo) and `piper-en-amy.gguf` (tts_demo);
#        Silero VAD is intentionally NOT downloaded here because neither
#        Godot demo references it (M3-11-T14 + T15 scope). If a future
#        VAD demo lands, add a fetch_one line — this file is the seam;
#    (c) the piper-plus voice targeted here is en_US-amy (voice ID
#        `en_us_amy_medium`) rather than the Unity mirror's en_US-lessac,
#        because the M3-11-T15 tts_demo main.gd pins that voice ID.
# =============================================================================
#
# MIT-only policy: this addon deliberately downloads ONLY MIT-licensed
# artifacts (Whisper base, piper-plus en_US-amy voice). CC-BY-NC weights
# (F5-TTS, Fish-Speech, EnCodec) and voice-clone weights (RVC / GPT-SoVITS)
# are EXPLICITLY EXCLUDED — those are handled by the M2-13 compliance gate
# via a research flag and are shipped from a separate, non-official model
# index. Bundling them here would (a) invert the license of the demo,
# (b) route CC-BY-NC weights through the "official" AssetLib distribution,
# and (c) break the BR-10 / M2-13 provenance contract.
#
# Models are NEVER committed to git — see `integrations/vokra-godot/demos/`
# README.md and NFR-DS-04. This script is the reproducible way to
# reconstitute the models directory before opening either demo project
# in the Godot 4.x Editor.
#
# Usage (from a Godot project root that has installed the vokra AssetLib):
#   bash addons/vokra/fetch-demo-models.sh                 # download all
#   VOKRA_MODELS_DIR=/tmp/vokra bash addons/vokra/fetch-demo-models.sh
#
# Environment overrides (all optional):
#   VOKRA_MODELS_DIR   Destination directory
#                      (default: <project_root>/models = res://models/)
#   VOKRA_WHISPER_URL  Whisper base GGUF URL (must resolve to MIT weights)
#   VOKRA_PIPER_URL    piper-plus en_US-amy voice GGUF URL
#                      (must resolve to MIT weights)
#
# Zero external dependency: uses curl (or wget as fallback) which are POSIX
# base tooling; no npm/pip/cargo. Matches the Unity mirror's zero-dep
# posture and the Vokra runtime's NFR-DS-02 spirit.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Godot project layout: this script sits at
#   <project_root>/addons/vokra/fetch-demo-models.sh
# so <project_root> is two `..` up from SCRIPT_DIR. The demo GDScripts
# expect models under `res://models/`, which is `<project_root>/models`.
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEST_DIR="${VOKRA_MODELS_DIR:-${PROJECT_ROOT}/models}"

# Default download endpoints. The maintainer publishes these as GitHub
# Release assets on ayutaz/vokra so their URLs are stable and their license
# provenance is committed to docs/license-audit.md. Override any of them
# via the env vars above when testing an unreleased build.
#
# Piper voice pin rationale: the M3-11-T15 tts_demo main.gd hard-codes
# VOICE_ID = "en_us_amy_medium", so the sensible default here is the
# en_US-amy medium checkpoint. Owners packaging a different voice should
# rename the destination file AND the const in tts_demo/main.gd in lock-step.
: "${VOKRA_WHISPER_URL:=https://github.com/ayutaz/vokra/releases/latest/download/whisper-base.gguf}"
: "${VOKRA_PIPER_URL:=https://github.com/ayutaz/vokra/releases/latest/download/piper-plus-en-us-amy.gguf}"

# License attribution surfaced by --license and echoed at the end of a run
# so the operator has to acknowledge each attribution. Do NOT extend this
# list with CC-BY-NC / research-flag models — those go through a separate
# index (see M2-13 compliance gate).
license_manifest() {
    cat <<'EOF'
License attribution (all MIT):
  * Whisper base            — MIT, openai/whisper
  * piper-plus en_US-amy    — MIT, ayutaz/piper-plus

CC-BY-NC / non-commercial weights (F5-TTS, Fish-Speech, EnCodec) and
voice-clone weights (RVC, GPT-SoVITS) are intentionally EXCLUDED from this
demo per M2-13 compliance. Enable them via the vokra-cli research flag
against the separate research model index.
EOF
}

case "${1:-}" in
    -h|--help)
        cat <<EOF
Usage: $0 [--license]

Downloads MIT-only demo weights into ${DEST_DIR}.
Set VOKRA_MODELS_DIR to write elsewhere. Set VOKRA_{WHISPER,PIPER}_URL
to override individual sources.

Godot layout: this script assumes it lives at
  <project_root>/addons/vokra/fetch-demo-models.sh
and writes to <project_root>/models/ so the demo GDScripts can load
  res://models/whisper-base.gguf   (asr_demo)
  res://models/piper-en-amy.gguf   (tts_demo)
EOF
        license_manifest
        exit 0
        ;;
    --license)
        license_manifest
        exit 0
        ;;
esac

# Pick a downloader. curl is preferred (universally available on macOS/Linux
# CI runners), wget is the fallback for minimal container images. Matches
# the Unity mirror's downloader probe verbatim so behavior is consistent
# across the two demo channels.
if command -v curl >/dev/null 2>&1; then
    DOWNLOAD_CMD="curl -fL --retry 3 --connect-timeout 15 -o"
elif command -v wget >/dev/null 2>&1; then
    DOWNLOAD_CMD="wget -O"
else
    echo "error: neither curl nor wget is installed; cannot download models" >&2
    exit 1
fi

mkdir -p "${DEST_DIR}"

fetch_one() {
    local url="$1"
    local dest="$2"
    local label="$3"

    if [ -f "${dest}" ] && [ -s "${dest}" ]; then
        echo "skip: ${label} already present at ${dest}"
        return 0
    fi

    echo "fetch: ${label} <- ${url}"
    # shellcheck disable=SC2086
    # DOWNLOAD_CMD is an intentional multi-word command (curl -fL --retry ...
    # or wget -O); word-splitting is the desired behavior here. Matches the
    # Unity mirror's identical use.
    ${DOWNLOAD_CMD} "${dest}" "${url}"

    if [ ! -s "${dest}" ]; then
        echo "error: downloaded ${label} is empty; check the URL" >&2
        rm -f "${dest}"
        exit 1
    fi
}

fetch_one "${VOKRA_WHISPER_URL}" "${DEST_DIR}/whisper-base.gguf"  "Whisper base (MIT)"
fetch_one "${VOKRA_PIPER_URL}"   "${DEST_DIR}/piper-en-amy.gguf"  "piper-plus en_US-amy (MIT)"

echo
echo "OK: MIT-only demo weights in ${DEST_DIR}"
echo
license_manifest
