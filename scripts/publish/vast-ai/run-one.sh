#!/usr/bin/env bash
# run-one.sh — driver for one full publish on a provisioned vast.ai instance.
#
# Chains: HF snapshot_download → vokra-cli convert → publish-one.sh.
#
# 2026-07-28 policy (memory feedback-large-models-on-vast-ai): convert +
# upload for HF weights defaults to vast.ai. This script is the per-model
# entry point. Assumes provision.sh has already run on this box (Rust +
# uv + hf-transfer + repo + vokra-cli built, VOKRA_PUBLISH_ON_VAST=1
# marker in shell so publish-one.sh gate 7 auto-bypasses).
#
# DRY-RUN by default. Publishing (--push) is irreversible.
#
# Usage:
#   run-one.sh --hf-repo <slug> --vokra-slug <name> --model-kind <kind> \
#              --license-spdx <spdx> [--push] [--allow-noncommercial] \
#              [--acknowledge-copyleft] [--allow-large] \
#              [--include <glob>]... [--input-name <basename>] \
#              [--config-name <basename>]
#   run-one.sh --self-test
#
# Example:
#   export HF_TOKEN='hf_xxxxxx'
#   run-one.sh --hf-repo mistralai/Voxtral-Small-24B-2507 \
#              --vokra-slug voxtral-small-24b-2507 \
#              --model-kind voxtral --license-spdx apache-2.0 --push

set -euo pipefail

VOKRA_ROOT="${VOKRA_ROOT:-$HOME/vokra}"
VOKRA_SCRATCH="${VOKRA_SCRATCH:-$HOME/scratchpad}"

log()  { printf '[run-one] %s\n' "$*" >&2; }
step() { printf '\n\033[1;36m[run-one] ==== %s ====\033[0m\n' "$*" >&2; }

usage() {
  cat <<'EOF' >&2
usage: run-one.sh --hf-repo <slug> --vokra-slug <name> --model-kind <kind> \
                  --license-spdx <spdx> \
                  [--push] [--allow-noncommercial] [--acknowledge-copyleft] [--allow-large] \
                  [--include <glob>]... [--input-name <basename>] [--config-name <basename>]
       run-one.sh --self-test

Required:
  --hf-repo      upstream HF slug, e.g. mistralai/Voxtral-Small-24B-2507
  --vokra-slug   name under huggingface.co/vokra/<slug>
  --model-kind   passed to `vokra-cli convert --model <kind>` (see `vokra-cli convert --help`)
  --license-spdx SPDX id passed to publish-one.sh (canonical LICENSE will be fetched)

Optional:
  --push                    perform the upload (default: dry-run stage only)
  --allow-noncommercial     required for T4 (cc-by-nc-*) weights
  --acknowledge-copyleft    required for T3 (agpl / cc-by-sa) weights
  --allow-large             pass through to publish-one.sh gate 7 (usually unnecessary on vast.ai
                            because VOKRA_PUBLISH_ON_VAST=1 marker auto-bypasses)
  --include <glob>          allow_patterns for HF snapshot_download; repeatable.
                            Defaults to: "*.safetensors" "model.safetensors.index.json" "*.json"
                            "merges.txt" "vocab.json" "tokenizer.model" "params.json"
  --input-name <basename>   which file inside the snapshot to hand to convert
                            (default: auto-detect model.safetensors.index.json ▸ model.safetensors)
  --config-name <basename>  arch config to hand to convert as --config
                            (default: config.json if present, else omit)

HF token: HF_TOKEN or HF must be set in env (fail-closed).
EOF
}

# --- HF DL ----------------------------------------------------------------
# Uses huggingface_hub.snapshot_download via uv-managed Python. hf-transfer
# is opt-in via HF_HUB_ENABLE_HF_TRANSFER=1 (Rust-backed helper, ~40x faster
# for large files — memory project-huggingface-vokra-publication).
hf_download() {
  local repo="$1" cache_dir="$2"
  shift 2
  local patterns=("$@")
  local pattern_json
  pattern_json="$(uv run --no-project python -c 'import json,sys; print(json.dumps(sys.argv[1:]))' "${patterns[@]}")"

  # Run inside tools/parity's uv env so hf-transfer is resolvable (provision.sh
  # `uv add hf-transfer huggingface_hub` there). Falls back to --with if that
  # env is not populated.
  local uv_cwd="$VOKRA_ROOT/tools/parity"
  local uv_cmd=(uv run --with huggingface_hub --with hf-transfer python)
  if [[ ! -f "$uv_cwd/pyproject.toml" ]]; then
    log "note: $uv_cwd/pyproject.toml missing — running with --with fallback"
    uv_cwd="$VOKRA_ROOT"
  fi

  # Prefix-form assignment on a compound command is not valid bash — export
  # into the subshell instead so `set -u` also stays honest.
  (
    export HF_HUB_ENABLE_HF_TRANSFER=1
    cd "$uv_cwd"
    "${uv_cmd[@]}" - "$repo" "$cache_dir" "$pattern_json" <<'PY'
import json, os, sys
from huggingface_hub import snapshot_download

repo, cache_dir, pattern_json = sys.argv[1], sys.argv[2], sys.argv[3]
patterns = json.loads(pattern_json)
token = os.environ.get("HF_TOKEN") or os.environ.get("HF")

os.makedirs(cache_dir, exist_ok=True)
path = snapshot_download(
    repo_id=repo,
    cache_dir=cache_dir,
    allow_patterns=patterns,
    token=token,
)
print(path)
PY
  )
}

# Auto-detect input file inside the downloaded snapshot. Prefer multi-shard
# index.json (mistralai / kimi / step-audio / baichuan / etc), fall back to
# single-file model.safetensors. Fail with a helpful listing if neither.
autodetect_input() {
  local snap="$1"
  if [[ -f "$snap/model.safetensors.index.json" ]]; then
    printf '%s\n' "model.safetensors.index.json"
    return 0
  fi
  if [[ -f "$snap/model.safetensors" ]]; then
    printf '%s\n' "model.safetensors"
    return 0
  fi
  # Last resort: pick the first *.safetensors alphabetically. Warn — this may
  # be wrong for exotic layouts and the caller should probably pass --input-name.
  local first
  first="$(cd "$snap" && ls -1 *.safetensors 2>/dev/null | head -1 || true)"
  if [[ -n "$first" ]]; then
    log "WARN: no model.safetensors[.index.json] — using first match: $first"
    log "      pass --input-name <basename> if this is not the right file"
    printf '%s\n' "$first"
    return 0
  fi
  log "ERROR: no .safetensors found in $snap"
  log "       directory contents:"
  ( cd "$snap" && ls -la ) | head -20 >&2
  return 2
}

# --- self-test -----------------------------------------------------------
# Pure: exercises autodetect_input against a hand-built directory. No HF
# fetch, no CLI invocation.
run_self_test() {
  local tmp cases=0 fail=0 out
  tmp="$(mktemp -d)"
  # shellcheck disable=SC2064
  trap "rm -rf '$tmp'" EXIT

  # Case: index.json wins over single-file.
  mkdir -p "$tmp/both"
  : > "$tmp/both/model.safetensors.index.json"
  : > "$tmp/both/model.safetensors"
  cases=$((cases + 1))
  out="$(autodetect_input "$tmp/both")"
  [[ "$out" == "model.safetensors.index.json" ]] \
    || { echo "self-test FAIL: autodetect_input preferred wrong file (got $out)" >&2; fail=1; }

  # Case: single-file only.
  mkdir -p "$tmp/single"
  : > "$tmp/single/model.safetensors"
  cases=$((cases + 1))
  out="$(autodetect_input "$tmp/single")"
  [[ "$out" == "model.safetensors" ]] \
    || { echo "self-test FAIL: single-file autodetect (got $out)" >&2; fail=1; }

  # Case: exotic name — pick first alphabetically, warn.
  mkdir -p "$tmp/exotic"
  : > "$tmp/exotic/weights_bf16.safetensors"
  : > "$tmp/exotic/weights_fp16.safetensors"
  cases=$((cases + 1))
  out="$(autodetect_input "$tmp/exotic" 2>/dev/null)"
  [[ "$out" == "weights_bf16.safetensors" ]] \
    || { echo "self-test FAIL: exotic autodetect (got $out)" >&2; fail=1; }

  # Case: no safetensors at all — must fail with rc 2.
  mkdir -p "$tmp/empty"
  : > "$tmp/empty/config.json"
  cases=$((cases + 1))
  if out="$(autodetect_input "$tmp/empty" 2>&1)"; then
    echo "self-test FAIL: autodetect returned $out on empty dir instead of failing" >&2; fail=1
  fi

  # Argument sanity: usage() must mention every required flag verbatim.
  # A trivial regression guard against silently dropping flags.
  local u; u="$(usage 2>&1)"
  for flag in --hf-repo --vokra-slug --model-kind --license-spdx --push \
              --allow-noncommercial --acknowledge-copyleft --allow-large \
              --include --input-name --config-name; do
    cases=$((cases + 1))
    if ! grep -Fq -- "$flag" <<<"$u"; then
      echo "self-test FAIL: usage() dropped flag $flag" >&2; fail=1
    fi
  done

  if [[ $fail -eq 0 ]]; then
    echo "run-one.sh self-test: OK ($cases cases)"
    return 0
  fi
  return 1
}

# --- main ----------------------------------------------------------------

main() {
  local hf_repo="" vokra_slug="" model_kind="" license_spdx=""
  local push=0 nc=0 ack=0 allow_large=0 self_test=0
  local input_name="" config_name=""
  local include=()

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --hf-repo)               hf_repo="$2"; shift 2 ;;
      --vokra-slug)            vokra_slug="$2"; shift 2 ;;
      --model-kind)            model_kind="$2"; shift 2 ;;
      --license-spdx)          license_spdx="$2"; shift 2 ;;
      --push)                  push=1; shift ;;
      --allow-noncommercial)   nc=1; shift ;;
      --acknowledge-copyleft)  ack=1; shift ;;
      --allow-large)           allow_large=1; shift ;;
      --include)               include+=("$2"); shift 2 ;;
      --input-name)            input_name="$2"; shift 2 ;;
      --config-name)           config_name="$2"; shift 2 ;;
      --self-test)             self_test=1; shift ;;
      -h|--help)               usage; exit 0 ;;
      *)                       echo "run-one: unknown flag '$1'" >&2; usage; exit 2 ;;
    esac
  done

  if [[ $self_test -eq 1 ]]; then
    run_self_test
    exit $?
  fi

  # Fail-closed on required args.
  [[ -n "$hf_repo"      ]] || { echo "run-one: --hf-repo required" >&2; exit 2; }
  [[ -n "$vokra_slug"   ]] || { echo "run-one: --vokra-slug required" >&2; exit 2; }
  [[ -n "$model_kind"   ]] || { echo "run-one: --model-kind required" >&2; exit 2; }
  [[ -n "$license_spdx" ]] || { echo "run-one: --license-spdx required" >&2; exit 2; }
  [[ -n "${HF_TOKEN:-${HF:-}}" ]] \
    || { echo "run-one: HF_TOKEN (or HF) must be set in env — provision.sh does not persist tokens" >&2; exit 2; }
  [[ -x "$VOKRA_ROOT/target/release/vokra-cli" ]] \
    || { echo "run-one: $VOKRA_ROOT/target/release/vokra-cli missing — run scripts/publish/vast-ai/provision.sh first" >&2; exit 2; }

  # Default include patterns — covers the common cases (mistralai multi-shard,
  # openbmb single-file, sentencepiece / bpe tokenizers). Owner can override
  # by passing --include multiple times.
  if [[ ${#include[@]} -eq 0 ]]; then
    include=(
      "*.safetensors"
      "model.safetensors.index.json"
      "*.json"
      "merges.txt"
      "vocab.json"
      "tokenizer.model"
      "params.json"
    )
  fi

  step "$hf_repo -> vokra/$vokra_slug ($model_kind, $license_spdx)"

  local staging="$VOKRA_SCRATCH/staging/$vokra_slug"
  local cache="$VOKRA_SCRATCH/hf-cache"
  mkdir -p "$staging" "$cache"
  log "staging : $staging"
  log "cache   : $cache"

  # --- DL ---------------------------------------------------------------
  step "HF snapshot_download (hf-transfer, allow_patterns=${include[*]})"
  local snap
  snap="$(hf_download "$hf_repo" "$cache" "${include[@]}")"
  log "snapshot: $snap"

  # --- input auto-detect ------------------------------------------------
  if [[ -z "$input_name" ]]; then
    input_name="$(autodetect_input "$snap")" || exit 2
  fi
  [[ -f "$snap/$input_name" ]] || { echo "run-one: --input-name '$input_name' not found in snapshot" >&2; exit 2; }
  log "input   : $input_name"

  # --- config auto-detect (optional) ------------------------------------
  local config_args=()
  if [[ -n "$config_name" ]]; then
    [[ -f "$snap/$config_name" ]] || { echo "run-one: --config-name '$config_name' not found in snapshot" >&2; exit 2; }
    config_args=(--config "$snap/$config_name")
    log "config  : $config_name"
  elif [[ -f "$snap/config.json" ]]; then
    config_args=(--config "$snap/config.json")
    log "config  : config.json (auto-detected)"
  else
    log "config  : (none — omitting --config)"
  fi

  # --- convert ----------------------------------------------------------
  step "vokra-cli convert --model $model_kind"
  local gguf="$staging/model.gguf"
  "$VOKRA_ROOT/target/release/vokra-cli" convert \
    --model "$model_kind" \
    --input "$snap/$input_name" \
    ${config_args[@]+"${config_args[@]}"} \
    --output "$gguf"
  log "GGUF written: $gguf ($(du -h "$gguf" | cut -f1))"

  # --- publish (dry-run first) ------------------------------------------
  local pub_flags=()
  [[ $nc          -eq 1 ]] && pub_flags+=(--allow-noncommercial)
  [[ $ack         -eq 1 ]] && pub_flags+=(--acknowledge-copyleft)
  [[ $allow_large -eq 1 ]] && pub_flags+=(--allow-large)

  step "publish-one.sh (dry-run)"
  "$VOKRA_ROOT/scripts/publish/publish-one.sh" \
    --gguf "$gguf" \
    --repo "vokra/$vokra_slug" \
    --license-spdx "$license_spdx" \
    ${pub_flags[@]+"${pub_flags[@]}"}

  if [[ $push -eq 0 ]]; then
    step "DRY RUN complete — re-run with --push to publish"
    exit 0
  fi

  step "publish-one.sh --push (irreversible)"
  "$VOKRA_ROOT/scripts/publish/publish-one.sh" \
    --gguf "$gguf" \
    --repo "vokra/$vokra_slug" \
    --license-spdx "$license_spdx" \
    --push \
    ${pub_flags[@]+"${pub_flags[@]}"}

  step "live check"
  local url="https://huggingface.co/vokra/$vokra_slug"
  local code
  code="$(curl -sI "$url" | head -1 || true)"
  log "$code -> $url"
  log ""
  log "next model? re-invoke run-one.sh with a different --hf-repo / --vokra-slug"
  log "done?       vastai destroy <instance-id>  # or destroy from the vast.ai UI"
}

main "$@"
