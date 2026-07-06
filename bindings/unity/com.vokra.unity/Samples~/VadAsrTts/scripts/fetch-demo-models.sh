#!/usr/bin/env bash
# fetch-demo-models.sh — downloads MIT-only demo weights into the sample's
# StreamingAssets/models/ directory (M2-11-T11 / plan §D9).
#
# MIT-only policy: this sample deliberately downloads ONLY MIT-licensed
# artifacts (Silero VAD v5, Whisper base, piper-plus voice). CC-BY-NC weights
# (F5-TTS, Fish-Speech) and voice-clone weights (RVC / GPT-SoVITS) are
# EXPLICITLY EXCLUDED — those are handled by the M2-13 compliance gate via a
# research flag and are shipped from a separate, non-official model index.
# Bundling them here would (a) invert the license of the sample, (b) route
# CC-BY-NC weights through the "official" com.vokra.unity distribution, and
# (c) break the BR-10 / M2-13 provenance contract.
#
# Models are NEVER committed to git — see the Samples~/VadAsrTts/.gitignore
# entry for *.gguf and NFR-DS-04. This script is the reproducible way to
# reconstitute the models directory before opening the sample scene.
#
# Usage:
#   bash fetch-demo-models.sh                      # download all defaults
#   VOKRA_MODELS_DIR=/tmp/vokra bash fetch-demo-models.sh
#
# Environment overrides (all optional):
#   VOKRA_MODELS_DIR   Destination directory (default: <script>/../StreamingAssets/models)
#   VOKRA_SILERO_URL   Silero VAD v5 GGUF URL (must resolve to MIT weights)
#   VOKRA_WHISPER_URL  Whisper base GGUF URL (must resolve to MIT weights)
#   VOKRA_PIPER_URL    piper-plus voice GGUF URL (must resolve to MIT weights)
#
# Zero external dependency: uses curl (or wget as fallback) which are POSIX
# base tooling; no npm/pip/cargo.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SAMPLE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEST_DIR="${VOKRA_MODELS_DIR:-${SAMPLE_DIR}/StreamingAssets/models}"

# Default download endpoints. The maintainer publishes these as GitHub Release
# assets on ayutaz/vokra so their URLs are stable and their license provenance
# is committed to docs/license-audit.md. Override any of them via the env vars
# above when testing an unreleased build.
: "${VOKRA_SILERO_URL:=https://github.com/ayutaz/vokra/releases/latest/download/silero-vad-v5.gguf}"
: "${VOKRA_WHISPER_URL:=https://github.com/ayutaz/vokra/releases/latest/download/whisper-base.gguf}"
: "${VOKRA_PIPER_URL:=https://github.com/ayutaz/vokra/releases/latest/download/piper-plus-en-us-lessac.gguf}"

# License attribution surfaced by --license and echoed at the end of a run so
# the operator has to acknowledge each attribution. Do NOT extend this list
# with CC-BY-NC / research-flag models — those go through a separate index.
license_manifest() {
    cat <<'EOF'
License attribution (all MIT):
  * Silero VAD v5           — MIT, snakers4/silero-vad
  * Whisper base            — MIT, openai/whisper
  * piper-plus voice        — MIT, ayutaz/piper-plus

CC-BY-NC / non-commercial weights (F5-TTS, Fish-Speech, EnCodec) and
voice-clone weights (RVC, GPT-SoVITS) are intentionally EXCLUDED from this
sample per M2-13 compliance. Enable them via the vokra-cli research flag
against the separate research model index.
EOF
}

case "${1:-}" in
    -h|--help)
        cat <<EOF
Usage: $0 [--license]

Downloads MIT-only demo weights into ${DEST_DIR}.
Set VOKRA_MODELS_DIR to write elsewhere. Set VOKRA_{SILERO,WHISPER,PIPER}_URL
to override individual sources.
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
# CI runners), wget is the fallback for minimal container images.
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
    ${DOWNLOAD_CMD} "${dest}" "${url}"

    if [ ! -s "${dest}" ]; then
        echo "error: downloaded ${label} is empty; check the URL" >&2
        rm -f "${dest}"
        exit 1
    fi
}

fetch_one "${VOKRA_SILERO_URL}"  "${DEST_DIR}/silero-vad-v5.gguf" "Silero VAD v5 (MIT)"
fetch_one "${VOKRA_WHISPER_URL}" "${DEST_DIR}/whisper-base.gguf"  "Whisper base (MIT)"
fetch_one "${VOKRA_PIPER_URL}"   "${DEST_DIR}/voice.gguf"         "piper-plus voice (MIT)"

echo
echo "OK: MIT-only demo weights in ${DEST_DIR}"
echo
license_manifest
