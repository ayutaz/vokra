#!/usr/bin/env bash
# fetch_rmvpe_pt.sh — Owner helper: download the upstream RMVPE .pt
# checkpoint from GitHub Releases and verify sha256.
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
#   --url <url>       download URL (default = pinned yxlllc/RMVPE release)
#   --output <path>   local destination (required)
#   --sha256 <hex>    if set, verify integrity after download
#
# License: upstream is MIT (yxlllc/RMVPE and Dream-High/RMVPE both).
set -euo pipefail

# Pinned default: the current upstream release. Verify the tag on
# https://github.com/yxlllc/RMVPE/releases if a newer one lands.
DEFAULT_URL="https://github.com/yxlllc/RMVPE/releases/download/230917/model.pt"

URL="$DEFAULT_URL"
OUTPUT=""
EXPECTED_SHA256=""

while [ $# -gt 0 ]; do
    case "$1" in
        --url)     URL="$2";           shift 2 ;;
        --output)  OUTPUT="$2";        shift 2 ;;
        --sha256)  EXPECTED_SHA256="$2"; shift 2 ;;
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

echo "[fetch] URL:    $URL"
echo "[fetch] OUTPUT: $OUTPUT"

# --location follows redirects (GitHub Releases 302s to a CDN);
# --fail turns non-2xx into non-zero exit; -C - resumes partial downloads
# so a re-run is idempotent.
curl --location --fail --continue-at - --output "$OUTPUT" "$URL"

if [ -n "$EXPECTED_SHA256" ]; then
    if command -v sha256sum >/dev/null 2>&1; then
        GOT_SHA=$(sha256sum "$OUTPUT" | awk '{print $1}')
    else
        # macOS ships shasum, not sha256sum.
        GOT_SHA=$(shasum -a 256 "$OUTPUT" | awk '{print $1}')
    fi
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
