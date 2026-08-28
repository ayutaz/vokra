#!/usr/bin/env bash
# fetch_rmvpe_pt.sh — Owner helper: download the upstream RMVPE .pt
# checkpoint from the pinned GitHub release ZIP and verify sha256.
#
# CC does NOT run this script — per 依頼者ルール #3 the owner fetches
# upstream weights on their own machine (the runtime is Rust and never
# imports the .pt at inference time). The dumper (dump_reference.py)
# is the only consumer; the produced hidden.f32 / argmax.u32 are what
# actually cross into the Vokra parity harness.
#
# Usage:
#   bash fetch_rmvpe_pt.sh --output ~/rmvpe-fixtures/rmvpe.pt
#
# Overrides:
#   --url <url>       ZIP URL (default = pinned yxlllc/RMVPE release)
#   --output <path>   local destination (required)
#   --sha256 <hex>    expected extracted model.pt SHA-256 (recommended)
#   --archive-sha256 <hex> expected release ZIP SHA-256 (optional)
#
# License: Dream-High/RMVPE code is Apache-2.0. The yxlllc/RMVPE release
# repository does not declare checkpoint terms; generated artifacts must
# remain fail-closed (`unknown`) unless the owner verifies an exact grant.
set -euo pipefail

# GitHub release `230917` exposes `rmvpe.zip`, not a direct model.pt asset.
# Header-only central-directory audit on 2026-08-26 confirms `model.pt` is the
# archive member consumed by src.inference.RMVPE.
DEFAULT_URL="https://github.com/yxlllc/RMVPE/releases/download/230917/rmvpe.zip"

URL="$DEFAULT_URL"
OUTPUT=""
EXPECTED_SHA256=""
EXPECTED_ARCHIVE_SHA256=""

while [ $# -gt 0 ]; do
    case "$1" in
        --url)     URL="$2";           shift 2 ;;
        --output)  OUTPUT="$2";        shift 2 ;;
        --sha256)  EXPECTED_SHA256="$2"; shift 2 ;;
        --archive-sha256) EXPECTED_ARCHIVE_SHA256="$2"; shift 2 ;;
        -h|--help)
            grep -E '^#( |$)' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "unknown arg: $1" >&2
            exit 2
            ;;
    esac
done

if [ -z "$OUTPUT" ]; then
    echo "--output <path> is required" >&2
    exit 2
fi

mkdir -p "$(dirname "$OUTPUT")"
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT
ARCHIVE="$WORK_DIR/rmvpe.zip"

echo "[fetch] URL:    $URL"
echo "[fetch] OUTPUT: $OUTPUT (archive member model.pt)"

# --location follows redirects (GitHub Releases 302s to a CDN);
# --fail turns non-2xx into non-zero exit. The archive lives in a fresh
# temporary directory and is deleted after model.pt extraction.
curl --location --fail --output "$ARCHIVE" "$URL"

sha256_of() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    else
        shasum -a 256 "$1" | awk '{print $1}'
    fi
}

if [ -n "$EXPECTED_ARCHIVE_SHA256" ]; then
    GOT_ARCHIVE_SHA256="$(sha256_of "$ARCHIVE")"
    if [ "$GOT_ARCHIVE_SHA256" != "$EXPECTED_ARCHIVE_SHA256" ]; then
        echo "[fetch] ARCHIVE SHA256 MISMATCH" >&2
        echo "[fetch]   expected: $EXPECTED_ARCHIVE_SHA256" >&2
        echo "[fetch]   got:      $GOT_ARCHIVE_SHA256" >&2
        exit 3
    fi
    echo "[fetch] archive sha256 verified: $GOT_ARCHIVE_SHA256"
fi

if ! command -v unzip >/dev/null 2>&1; then
    echo "[fetch] unzip is required to extract model.pt" >&2
    exit 2
fi
unzip -p "$ARCHIVE" model.pt >"$OUTPUT"
if [ ! -s "$OUTPUT" ]; then
    echo "[fetch] extracted model.pt is empty" >&2
    exit 3
fi

if [ -n "$EXPECTED_SHA256" ]; then
    GOT_SHA="$(sha256_of "$OUTPUT")"
    if [ "$GOT_SHA" != "$EXPECTED_SHA256" ]; then
        echo "[fetch] SHA256 MISMATCH" >&2
        echo "[fetch]   expected: $EXPECTED_SHA256" >&2
        echo "[fetch]   got:      $GOT_SHA" >&2
        exit 3
    fi
    echo "[fetch] sha256 verified: $GOT_SHA"
else
    echo "[fetch] (no --sha256 supplied; consider recording the digest for provenance)"
fi

echo "[fetch] done: $OUTPUT"
