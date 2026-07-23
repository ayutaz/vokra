#!/usr/bin/env bash
# publish-one.sh — take one model from a re-converted GGUF to a live HF repo.
#
# Chains the pieces that were proven by hand on piper-plus, so the repetitive
# 11-model run does not re-type (and mis-type) them:
#   1. stage via upload.sh (card from artifact + §3.1 sign-off gate + NOTICE/SOURCE)
#   2. fetch the correct LICENSE text (fetch_license.sh)
#   3. push via upload.sh --push
#
# The GGUF must ALREADY be re-converted with provenance stamped (upload.sh
# refuses an unstamped artifact). This script does not convert — conversion is
# memory-bound and model-specific, so it stays in the caller's hands.
#
# DRY-RUN by default; --push publishes. Publishing is irreversible.
#
# Usage:
#   publish-one.sh --gguf <file> --repo vokra/<name> \
#     ( --license-url <raw-url> | --license-spdx <spdx> ) [--push] [--allow-noncommercial]
#
# HF token: HF_TOKEN or HF in the environment.

set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

gguf=""; repo=""; lurl=""; lspdx=""; push=0; nc=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gguf) gguf="$2"; shift 2 ;;
    --repo) repo="$2"; shift 2 ;;
    --license-url) lurl="$2"; shift 2 ;;
    --license-spdx) lspdx="$2"; shift 2 ;;
    --push) push=1; shift ;;
    --allow-noncommercial) nc=1; shift ;;
    *) echo "publish-one: unexpected arg $1" >&2; exit 2 ;;
  esac
done
[[ -f "$gguf" ]] || { echo "publish-one: --gguf must be an existing file" >&2; exit 2; }
[[ -n "$repo" ]] || { echo "publish-one: --repo is required" >&2; exit 2; }
[[ -n "$lurl" || -n "$lspdx" ]] || { echo "publish-one: one of --license-url / --license-spdx is required" >&2; exit 2; }

model_name="${repo##*/}"
outdir="$(cd "$(git -C "$here" rev-parse --show-toplevel)" && pwd)/target/publish/$model_name"

nc_flag=(); [[ $nc -eq 1 ]] && nc_flag=(--allow-noncommercial)

echo "############ $repo ############"
echo "== stage (dry-run) =="
"$here/upload.sh" "$gguf" --repo "$repo" --out "$outdir" ${nc_flag[@]+"${nc_flag[@]}"}

echo "== LICENSE =="
if [[ -n "$lurl" ]]; then
  "$here/fetch_license.sh" --url "$lurl" "$outdir/LICENSE"
else
  "$here/fetch_license.sh" --spdx "$lspdx" "$outdir/LICENSE"
fi

if [[ $push -eq 0 ]]; then
  echo "== DRY RUN complete — staged in $outdir. Re-run with --push to publish. =="
  exit 0
fi

echo "== push =="
"$here/upload.sh" "$gguf" --repo "$repo" --out "$outdir" --push ${nc_flag[@]+"${nc_flag[@]}"}
echo "== done: https://huggingface.co/$repo =="
