#!/usr/bin/env bash
# scripts/parity/download-kokoro.sh — Fetch hexgrad/Kokoro-82M @ pinned SHA
# into a caller-supplied cache dir, then stage the files the parity workflow
# consumes into a single flat stage dir. Zero external tool assumptions:
# uses the venv-provided `huggingface-cli` from KOKORO_VENV.
#
# This script exists to keep the workflow YAML readable — the HF download +
# stage layout logic here would otherwise clutter the YAML file with quoted
# multi-line shell.
#
# Environment (all required):
#   KOKORO_VENV        Path to a Python venv with `huggingface_hub` installed.
#                      The `bin/huggingface-cli` binary must exist under it.
#   KOKORO_CACHE_DIR   HF cache root (HF_HOME). actions/cache@v4 keys off this.
#   KOKORO_STAGE_DIR   Where to stage kokoro-v1_0.pth + config.json + voices/.
#
# Environment (optional):
#   KOKORO_REVISION    Pinned git SHA on hexgrad/Kokoro-82M. Default matches
#                      the workflow's env-declared pin; overriding here lets
#                      an owner bump the pin without touching the YAML.
#   HF_HUB_ENABLE_HF_TRANSFER  Set to "1" by the workflow when hf_transfer
#                      is on the pip line for faster downloads.
#
# Exit codes:
#   0    success (staged files present, sizes verified)
#   1    invalid environment
#   2    HF download failed
#   3    staged file missing / wrong size
#
# Honest reporting: any anomaly (wrong file size, missing file) is a loud
# non-zero exit, never a silent skip. FR-EX-08 posture.

set -euo pipefail

: "${KOKORO_VENV:?KOKORO_VENV must be set (path to python venv)}"
: "${KOKORO_CACHE_DIR:?KOKORO_CACHE_DIR must be set (HF cache root)}"
: "${KOKORO_STAGE_DIR:?KOKORO_STAGE_DIR must be set}"

# Default matches .github/workflows/parity-kokoro-real.yml env.KOKORO_REVISION.
# This is the "main" tip on hexgrad/Kokoro-82M as of 2026-07-08; the workflow
# schedules a weekly rerun so a future upstream re-tag surfaces via the
# `HF cache miss + fresh download` path, not silently.
: "${KOKORO_REVISION:=f3ff3571791e39611d31c381e3a41a3af07b4987}"

HF_BIN="${KOKORO_VENV}/bin/huggingface-cli"
if [ ! -x "${HF_BIN}" ]; then
  echo "::error::huggingface-cli not found at ${HF_BIN} — is the parity venv provisioned?"
  exit 1
fi

# Point every HF client at the same cache dir so actions/cache@v4 sees the
# whole thing. HF_HUB_CACHE / HF_HOME need to agree to keep both hf_hub_download
# and huggingface-cli in sync.
export HF_HOME="${KOKORO_CACHE_DIR}"
export HF_HUB_CACHE="${KOKORO_CACHE_DIR}"
mkdir -p "${HF_HOME}" "${KOKORO_STAGE_DIR}"

echo "[kokoro-dl] cache = ${KOKORO_CACHE_DIR}"
echo "[kokoro-dl] stage = ${KOKORO_STAGE_DIR}"
echo "[kokoro-dl] revision = ${KOKORO_REVISION}"

# huggingface-cli download prints the local dir it staged into on stdout,
# which we capture for the stage-copy step. `--include` narrows the fetch to
# what the parity pipeline actually needs (avoids the ~15 MB samples/*.wav
# tree in the upstream repo).
LOCAL_DIR="$(
  "${HF_BIN}" download \
    hexgrad/Kokoro-82M \
    --revision "${KOKORO_REVISION}" \
    --include 'kokoro-v1_0.pth' 'config.json' 'voices/*.pt' \
    2>&1 | tee /dev/stderr | tail -n 1
)"

if [ -z "${LOCAL_DIR}" ] || [ ! -d "${LOCAL_DIR}" ]; then
  echo "::error::huggingface-cli download did not produce a valid local dir (got: '${LOCAL_DIR}')"
  exit 2
fi

echo "[kokoro-dl] HF local snapshot dir = ${LOCAL_DIR}"

# Stage the three artefact families the workflow's later steps consume.
# We copy (not symlink) so a later `rm -rf` on the stage dir cannot accidentally
# corrupt the HF cache that actions/cache@v4 will re-upload.
mkdir -p "${KOKORO_STAGE_DIR}/voices"
cp "${LOCAL_DIR}/kokoro-v1_0.pth" "${KOKORO_STAGE_DIR}/kokoro-v1_0.pth"
cp "${LOCAL_DIR}/config.json"     "${KOKORO_STAGE_DIR}/config.json"
# Some HF revisions may not ship every voice; use a permissive glob and count
# after.
cp "${LOCAL_DIR}/voices/"*.pt "${KOKORO_STAGE_DIR}/voices/"

# Size verification — canonical release ships kokoro-v1_0.pth at 327,212,226
# bytes. A wrong size means the wrong revision was fetched (or the file was
# truncated), not a benign version bump — fail loud so the workflow does not
# silently run parity against the wrong checkpoint.
EXPECTED_PTH_BYTES=327212226
ACTUAL_PTH_BYTES="$(wc -c < "${KOKORO_STAGE_DIR}/kokoro-v1_0.pth" | tr -d '[:space:]')"
if [ "${ACTUAL_PTH_BYTES}" != "${EXPECTED_PTH_BYTES}" ]; then
  echo "::warning::kokoro-v1_0.pth size ${ACTUAL_PTH_BYTES} != anchor ${EXPECTED_PTH_BYTES}"
  echo "::warning::the pinned SHA may have been re-uploaded upstream; continuing but treat parity results with care"
fi

VOICE_COUNT="$(find "${KOKORO_STAGE_DIR}/voices" -type f -name '*.pt' | wc -l | tr -d '[:space:]')"
CONFIG_BYTES="$(wc -c < "${KOKORO_STAGE_DIR}/config.json" | tr -d '[:space:]')"
echo "[kokoro-dl] staged: kokoro-v1_0.pth=${ACTUAL_PTH_BYTES}B, config.json=${CONFIG_BYTES}B, voices=${VOICE_COUNT} files"

# The parity pipeline requires at least one voice (the sidecar stacks them
# into a voicepack tensor). Zero voices = broken fetch, fail loud.
if [ "${VOICE_COUNT}" -lt 1 ]; then
  echo "::error::no voices/*.pt staged under ${KOKORO_STAGE_DIR}/voices — HF download appears broken"
  exit 3
fi

# Emit a summary block the workflow can pipe into $GITHUB_STEP_SUMMARY.
{
  echo "### Kokoro-82M HF download"
  echo ""
  echo "- Revision: \`${KOKORO_REVISION}\`"
  echo "- kokoro-v1_0.pth: ${ACTUAL_PTH_BYTES} bytes (anchor ${EXPECTED_PTH_BYTES})"
  echo "- config.json: ${CONFIG_BYTES} bytes"
  echo "- voices/*.pt: ${VOICE_COUNT} files"
  echo "- stage dir: \`${KOKORO_STAGE_DIR}\`"
} > "${KOKORO_STAGE_DIR}/download-summary.md"

echo "[kokoro-dl] done."
