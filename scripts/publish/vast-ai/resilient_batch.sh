#!/usr/bin/env bash
# resilient_batch.sh — Wave 9 residual retry driver.
#
# 2026-08-02 — Wave 9 vast.ai publish left 7 models stuck because the
# inline snapshot_download in run-one.sh cannot resume mid-chunk on flaky
# vast.ai egress:
#
#   1. HF_HUB_ENABLE_HF_TRANSFER=1 streams over HTTP range requests without
#      hf-hub-level resume — mid-chunk drops kill the shard.
#   2. Xet routing was silently rerouting large shards.
#   3. No per-file retry, no header validation on .safetensors.
#   4. snapshot_download re-resolves the revision on each call — if the
#      upstream repo pushes a commit mid-run the index/shard set diverges.
#
# This script fixes all four for the 7 residual models: it invokes
# scripts/publish/vast-ai/resilient_download.py (5-attempt exponential
# backoff, safetensors header validation, corrupt-blob eviction, pinned
# revision) instead of the inline snapshot_download in run-one.sh.
# The convert + publish chain is unchanged (still uses vokra-cli convert
# and scripts/publish/publish-one.sh, same layout as run-one.sh).
#
# We deliberately do NOT modify run-one.sh — that path stays as the
# well-tested single-model workflow for the common case.
#
# Usage:
#   resilient_batch.sh                    # DRY RUN — download + convert + stage, no upload
#   resilient_batch.sh --push             # actually upload
#   resilient_batch.sh --only <slug>[,<slug>...]  # subset of the 7 targets
#   resilient_batch.sh --self-test        # dry-run + wiring smoke, no HF I/O
#
# HF_TOKEN or HF must be set in env (fail-closed, per run-one.sh contract).

set -euo pipefail
set -o errtrace

VOKRA_ROOT="${VOKRA_ROOT:-$HOME/vokra}"
VOKRA_SCRATCH="${VOKRA_SCRATCH:-$HOME/scratchpad}"

# --- HF env forced for vast.ai flaky-egress resilience -------------------
# (see resilient_download.py module doc for rationale)
export HF_HUB_ENABLE_HF_TRANSFER=0            # cannot resume mid-chunk
export HF_HUB_DISABLE_XET=1                   # skip xet routing (pinned hh < 0.30)
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-1800}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-30}"

# vast.ai boxes have Ubuntu's system CA bundle at this canonical location.
# Only set if unset AND the file exists — do not clobber owner overrides,
# and do not point at nothing on non-Debian hosts.
if [[ -z "${SSL_CERT_FILE:-}" && -f /etc/ssl/certs/ca-certificates.crt ]]; then
  export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
fi
if [[ -z "${REQUESTS_CA_BUNDLE:-}" && -f /etc/ssl/certs/ca-certificates.crt ]]; then
  export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
fi

log()  { printf '[resilient-batch] %s\n' "$*" >&2; }
step() { printf '\n\033[1;36m[resilient-batch] ==== %s ====\033[0m\n' "$*" >&2; }

usage() {
  cat <<'EOF' >&2
usage: resilient_batch.sh [--push] [--only <slug>[,<slug>...]] [--self-test]

Wave 9 residual retry driver. Downloads + converts + publishes 7 models
that failed with the inline snapshot_download codepath in run-one.sh.

Targets (--only accepts these slugs):
  musicgen-melody           facebook/musicgen-melody         (cc-by-nc-4.0)
  mms-1b-all-base           facebook/mms-1b-all              (cc-by-nc-4.0, BASE only)
  moss-audio-4b-instruct    OpenMOSS-Team/MOSS-Audio-4B-Instruct (apache-2.0)
  moss-audio-8b-instruct    OpenMOSS-Team/MOSS-Audio-8B-Instruct (apache-2.0)
  audioldm2-large           cvssp/audioldm2-large            (cc-by-nc-sa-4.0)
  qwen2-5-omni-7b           Qwen/Qwen2.5-Omni-7B             (apache-2.0)
  seamless-m4t-v2-large     facebook/seamless-m4t-v2-large   (cc-by-nc-4.0)

Options:
  --push          upload to huggingface.co/vokra/<slug> (default: dry-run only)
  --only LIST     comma-separated slug list; unrecognized slugs fail loudly
  --self-test     no HF I/O; wiring smoke + resilient_download self-test
  -h,--help       this

HF_TOKEN or HF must be set in env before invocation.
EOF
}

# --- Target table --------------------------------------------------------
# Each row: slug|hf_repo|model_kind|license|extra_publish_flag|comment
# Fields are '|' separated; empty extra_publish_flag = no extra flag.
# order here matters: this is the resume order for a fresh vast.ai box.
TARGETS=(
  "musicgen-melody|facebook/musicgen-melody|musicgen-melody|cc-by-nc-4.0|--allow-noncommercial|musicgen family"
  "mms-1b-all-base|facebook/mms-1b-all|mms-1b-all|cc-by-nc-4.0|--allow-noncommercial|BASE only, adapters skipped"
  "moss-audio-4b-instruct|OpenMOSS-Team/MOSS-Audio-4B-Instruct|moss-audio|apache-2.0||"
  "moss-audio-8b-instruct|OpenMOSS-Team/MOSS-Audio-8B-Instruct|moss-audio|apache-2.0||"
  "audioldm2-large|cvssp/audioldm2-large|audioldm2-large|cc-by-nc-sa-4.0|--allow-noncommercial|8-submodule composite"
  "qwen2-5-omni-7b|Qwen/Qwen2.5-Omni-7B|qwen2-5-omni|apache-2.0||sharded"
  "seamless-m4t-v2-large|facebook/seamless-m4t-v2-large|seamless-m4t-v2|cc-by-nc-4.0|--allow-noncommercial|prefer .safetensors, drop .pt duplicates"
)

# Per-slug fetch patterns. Baked in (not a separate JSON registry) so the
# script is self-contained on a fresh vast.ai box. Each function prints
# include-globs then a sentinel '--' then exclude-globs, one per line.
# Callers pipe through mapfile.
patterns_for_slug() {
  local slug="$1"
  case "$slug" in
    musicgen-melody)
      # Sharded checkpoint + tokenizer + audio-encoder config.
      cat <<'EOP'
*.safetensors
model.safetensors.index.json
*.json
merges.txt
vocab.json
tokenizer.model
--
*.bin
*.msgpack
*.pt
EOP
      ;;
    mms-1b-all-base)
      # BASE model only: keep root-level files. mms-1b-all ships 2000+ LoRA
      # adapter shards under adapter_*/ — SKIP those and skip the flat
      # adapter_*.safetensors mirror too. The base model is model.safetensors
      # at the root plus config.json + tokenizer + preprocessor_config.
      cat <<'EOP'
config.json
preprocessor_config.json
tokenizer*
vocab.json
special_tokens_map.json
*.safetensors
--
adapter_*/*
adapter_*.safetensors
*.bin
*.msgpack
EOP
      ;;
    moss-audio-4b-instruct|moss-audio-8b-instruct)
      # Standard sharded HF LLM pattern.
      cat <<'EOP'
*.safetensors
model.safetensors.index.json
*.json
merges.txt
vocab.json
tokenizer.model
tokenizer*
special_tokens*
generation_config.json
--
*.bin
*.msgpack
*.pt
EOP
      ;;
    audioldm2-large)
      # Diffusers composite: 8 submodules
      # (text_encoder, text_encoder_2, projection_model, tokenizer,
      #  tokenizer_2, unet, vae, vocoder) + scheduler + feature_extractor.
      # Drop fp16 / non-ema duplicates to avoid same-weight-twice trap.
      cat <<'EOP'
model_index.json
*/config.json
*/preprocessor_config.json
*/tokenizer*
*/vocab.json
*/merges.txt
*/*.safetensors
scheduler/scheduler_config.json
--
*.bin
*.msgpack
*.ckpt
*.onnx
*/*.fp16.safetensors
*/*.non_ema.safetensors
EOP
      ;;
    qwen2-5-omni-7b)
      # Sharded, multimodal — keep audio + text configs + all shards.
      cat <<'EOP'
*.safetensors
model.safetensors.index.json
*.json
merges.txt
vocab.json
tokenizer*
special_tokens*
generation_config.json
preprocessor_config.json
--
*.bin
*.msgpack
*.pt
EOP
      ;;
    seamless-m4t-v2-large)
      # Prefer .safetensors, drop .pt duplicates (upstream ships both).
      cat <<'EOP'
*.safetensors
model.safetensors.index.json
config.json
tokenizer*
vocab*
special_tokens*
generation_config.json
preprocessor_config.json
sentencepiece*
--
*.pt
*.bin
*.msgpack
*_original.*
EOP
      ;;
    *)
      log "ERROR: patterns_for_slug: unknown slug '$slug'"
      return 2
      ;;
  esac
}

# --- Auto-detect input file inside snapshot (copy of run-one.sh logic) ---
autodetect_input() {
  local snap="$1"
  if [[ -f "$snap/model.safetensors.index.json" ]]; then
    printf '%s\n' "model.safetensors.index.json"; return 0
  fi
  if [[ -f "$snap/model_index.json" ]]; then
    # audioldm2 composite — convert reads the composite manifest.
    printf '%s\n' "model_index.json"; return 0
  fi
  if [[ -f "$snap/model.safetensors" ]]; then
    printf '%s\n' "model.safetensors"; return 0
  fi
  local first
  first="$(cd "$snap" && ls -1 *.safetensors 2>/dev/null | head -1 || true)"
  if [[ -n "$first" ]]; then
    log "WARN: no model.safetensors[.index.json] — using first match: $first"
    printf '%s\n' "$first"; return 0
  fi
  log "ERROR: no .safetensors found in $snap"
  ( cd "$snap" && ls -la ) | head -20 >&2
  return 2
}

# --- Fetch driver: call resilient_download.py under the tools/parity uv env ---
fetch_one() {
  local slug="$1" repo="$2" local_dir="$3"

  # Read the pattern block for this slug, split into include/exclude arrays
  # at the '--' sentinel.
  local raw
  raw="$(patterns_for_slug "$slug")"
  local -a include=() exclude=()
  local in_exclude=0
  local line
  while IFS= read -r line; do
    if [[ "$line" == "--" ]]; then in_exclude=1; continue; fi
    if [[ -z "$line" ]]; then continue; fi
    if [[ $in_exclude -eq 1 ]]; then
      exclude+=("$line")
    else
      include+=("$line")
    fi
  done <<<"$raw"

  # Build --include / --exclude args (repeated flags).
  local -a inc_args=() exc_args=()
  local g
  for g in "${include[@]}"; do inc_args+=(--include "$g"); done
  for g in "${exclude[@]}"; do exc_args+=(--exclude "$g"); done

  # Run under tools/parity's uv env so safetensors + huggingface_hub are
  # resolvable (provision.sh already `uv add`'d them). Fall back to --with
  # for a fresh box that has not run provision yet.
  local uv_cwd="$VOKRA_ROOT/tools/parity"
  local -a uv_cmd
  if [[ -f "$uv_cwd/pyproject.toml" ]]; then
    uv_cmd=(uv run python)
  else
    log "note: $uv_cwd/pyproject.toml missing — using --with fallback"
    uv_cwd="$VOKRA_ROOT"
    uv_cmd=(uv run --with huggingface_hub --with safetensors --with requests python)
  fi

  # The script itself lives beside this bash file. Resolve to abs path so a
  # cd into $uv_cwd does not break the invocation.
  local here
  here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  local driver="$here/resilient_download.py"
  [[ -f "$driver" ]] || { log "ERROR: driver missing: $driver"; return 2; }

  step "resilient_download $repo -> $local_dir"
  log "include: ${include[*]}"
  log "exclude: ${exclude[*]}"

  # HF_TOKEN is read from env by the driver — never on the command line.
  (
    cd "$uv_cwd"
    "${uv_cmd[@]}" "$driver" \
      --repo "$repo" \
      --local-dir "$local_dir" \
      ${inc_args[@]+"${inc_args[@]}"} \
      ${exc_args[@]+"${exc_args[@]}"}
  )
}

# --- Convert + publish for one target -----------------------------------
process_one() {
  local slug="$1" repo="$2" model_kind="$3" license="$4" extra_flag="$5"
  local push="$6"

  step "$slug ($repo, kind=$model_kind, license=$license)"

  local staging="$VOKRA_SCRATCH/staging/$slug"
  local snap="$VOKRA_SCRATCH/hf-cache/$slug"
  mkdir -p "$staging" "$snap"

  # 1. Download with the resilient driver.
  fetch_one "$slug" "$repo" "$snap"

  # 2. Detect input file inside the snapshot.
  local input_name
  input_name="$(autodetect_input "$snap")" || return 2
  [[ -f "$snap/$input_name" ]] || { log "ERROR: input '$input_name' missing after download"; return 2; }
  log "input: $input_name"

  # 3. Config auto-detect (root-level config.json — audioldm2 uses model_index.json
  # as the input itself, so no separate --config in that case).
  local -a config_args=()
  if [[ "$input_name" == "model_index.json" ]]; then
    log "config: (composite manifest is input — omitting --config)"
  elif [[ -f "$snap/config.json" ]]; then
    config_args=(--config "$snap/config.json")
    log "config: config.json"
  else
    log "config: (none)"
  fi

  # 3.5. Shard-merge prep (Wave 11 fix): vokra-cli's shard-index converters
  # refuse .index.json directly ("parse error: safetensors buffer truncated").
  # For known-sharded slugs, invoke the matching tools/parity/*_prepare_checkpoint.py
  # to pre-merge shards into a single model.merged.safetensors, then override
  # input_name so the vokra-cli convert step picks up the merged file.
  local prep_script=""
  case "$slug" in
    musicgen-melody)         prep_script="musicgen_melody_prepare_checkpoint.py" ;;
    moss-audio-4b-instruct)  prep_script="moss_audio_4b_instruct_prepare_checkpoint.py" ;;
    moss-audio-8b-instruct)  prep_script="moss_audio_8b_instruct_prepare_checkpoint.py" ;;
    audioldm2-large)         prep_script="audioldm2_large_prepare_checkpoint.py" ;;
    qwen2-5-omni-7b)         prep_script="qwen2_5_omni_7b_prepare_checkpoint.py" ;;
    seamless-m4t-v2-large)   prep_script="seamless_m4t_v2_large_prepare_checkpoint.py" ;;
  esac
  if [[ -n "$prep_script" ]] && [[ -f "$VOKRA_ROOT/tools/parity/$prep_script" ]]; then
    step "prep: tools/parity/$prep_script (shard-merge)"
    local merged="$snap/model.merged.safetensors"
    # Re-derive uv command here (uv_cmd inside fetch_one is out of scope).
    local prep_uv_cwd="$VOKRA_ROOT/tools/parity"
    local -a prep_uv_cmd
    if [[ -f "$prep_uv_cwd/pyproject.toml" ]]; then
      prep_uv_cmd=(uv run python)
    else
      prep_uv_cwd="$VOKRA_ROOT"
      prep_uv_cmd=(uv run --with huggingface_hub --with safetensors --with torch --with numpy python)
    fi
    if [[ ! -f "$merged" ]]; then
      (
        cd "$prep_uv_cwd"
        "${prep_uv_cmd[@]}" "$VOKRA_ROOT/tools/parity/$prep_script" \
          --input-dir "$snap" --output "$merged" \
          || "${prep_uv_cmd[@]}" "$VOKRA_ROOT/tools/parity/$prep_script" \
            --local-dir "$snap" --output "$merged"
      )
    fi
    if [[ -f "$merged" ]]; then
      input_name="model.merged.safetensors"
      log "prep: merged shards → $merged ($(du -h "$merged" | cut -f1))"
    else
      log "WARN: prep script did not produce $merged — falling back to $input_name"
    fi
  fi

  # 4. Convert to GGUF.
  step "vokra-cli convert --model $model_kind"
  local gguf="$staging/model.gguf"
  "$VOKRA_ROOT/target/release/vokra-cli" convert \
    --model "$model_kind" \
    --input "$snap/$input_name" \
    ${config_args[@]+"${config_args[@]}"} \
    --output "$gguf"
  log "GGUF written: $gguf ($(du -h "$gguf" | cut -f1))"

  # 5. Publish (dry-run first).
  local -a pub_flags=()
  [[ -n "$extra_flag" ]] && pub_flags+=("$extra_flag")

  step "publish-one.sh (dry-run)"
  "$VOKRA_ROOT/scripts/publish/publish-one.sh" \
    --gguf "$gguf" \
    --repo "vokra/$slug" \
    --license-spdx "$license" \
    ${pub_flags[@]+"${pub_flags[@]}"}

  if [[ $push -eq 0 ]]; then
    log "DRY RUN complete for $slug — re-run with --push to publish"
    return 0
  fi

  step "publish-one.sh --push $slug (irreversible)"
  "$VOKRA_ROOT/scripts/publish/publish-one.sh" \
    --gguf "$gguf" \
    --repo "vokra/$slug" \
    --license-spdx "$license" \
    --push \
    ${pub_flags[@]+"${pub_flags[@]}"}

  local url="https://huggingface.co/vokra/$slug"
  local code
  code="$(curl -sI "$url" | head -1 || true)"
  log "$code -> $url"
}

# --- self-test -----------------------------------------------------------
run_self_test() {
  local cases=0 fail=0
  echo "resilient_batch.sh self-test — no HF I/O, no publish"

  # (1) target table parses cleanly, 7 rows, unique slugs.
  cases=$((cases + 1))
  if [[ ${#TARGETS[@]} -ne 7 ]]; then
    echo "  FAIL: expected 7 targets, got ${#TARGETS[@]}"; fail=1
  fi
  local -a slugs=()
  local row
  for row in "${TARGETS[@]}"; do
    local IFS='|'
    read -r -a parts <<<"$row"
    if [[ ${#parts[@]} -lt 4 ]]; then
      echo "  FAIL: target row has <4 fields: $row"; fail=1; continue
    fi
    slugs+=("${parts[0]}")
  done
  # dedup check
  cases=$((cases + 1))
  local unique_slugs
  unique_slugs="$(printf '%s\n' "${slugs[@]}" | sort -u | wc -l | tr -d ' ')"
  if [[ "$unique_slugs" != "${#slugs[@]}" ]]; then
    echo "  FAIL: duplicate slugs in TARGETS: ${slugs[*]}"; fail=1
  fi

  # (2) every slug has a patterns_for_slug case + a '--' sentinel.
  local slug
  for slug in "${slugs[@]}"; do
    cases=$((cases + 1))
    local body
    if ! body="$(patterns_for_slug "$slug" 2>/dev/null)"; then
      echo "  FAIL: patterns_for_slug '$slug' returned non-zero"; fail=1; continue
    fi
    if ! grep -q '^--$' <<<"$body"; then
      echo "  FAIL: patterns_for_slug '$slug' missing '--' separator"; fail=1
    fi
  done

  # (3) resilient_download.py exists and its own self-test passes.
  cases=$((cases + 1))
  local here
  here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  local driver="$here/resilient_download.py"
  if [[ ! -f "$driver" ]]; then
    echo "  FAIL: resilient_download.py missing at $driver"; fail=1
  else
    # Run its self-test if python3 is on PATH. Skip cleanly if not.
    if command -v python3 >/dev/null 2>&1; then
      cases=$((cases + 1))
      if ! python3 "$driver" --self-test >/dev/null 2>&1; then
        echo "  FAIL: resilient_download.py --self-test returned non-zero"; fail=1
      fi
    else
      echo "  [skip] python3 not on PATH — cannot self-test resilient_download.py"
    fi
  fi

  # (4) usage() names every flag we handle.
  cases=$((cases + 1))
  local u
  u="$(usage 2>&1)"
  local flag
  for flag in --push --only --self-test; do
    if ! grep -Fq -- "$flag" <<<"$u"; then
      echo "  FAIL: usage() dropped flag $flag"; fail=1
    fi
  done

  # (5) HF env vars set to the expected values.
  cases=$((cases + 1))
  [[ "${HF_HUB_ENABLE_HF_TRANSFER}" == "0" ]] || { echo "  FAIL: HF_HUB_ENABLE_HF_TRANSFER != 0"; fail=1; }
  [[ "${HF_HUB_DISABLE_XET}" == "1" ]] || { echo "  FAIL: HF_HUB_DISABLE_XET != 1"; fail=1; }
  [[ "${HF_ENDPOINT}" == "https://huggingface.co" ]] || { echo "  FAIL: HF_ENDPOINT wrong: $HF_ENDPOINT"; fail=1; }

  if [[ $fail -eq 0 ]]; then
    echo "resilient_batch.sh self-test: OK ($cases cases)"
    return 0
  fi
  echo "resilient_batch.sh self-test: FAILED"
  return 1
}

# --- main ----------------------------------------------------------------
main() {
  local push=0 self_test=0 only=""
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --push)       push=1; shift ;;
      --only)       only="$2"; shift 2 ;;
      --self-test)  self_test=1; shift ;;
      -h|--help)    usage; exit 0 ;;
      *)            echo "resilient_batch: unknown flag '$1'" >&2; usage; exit 2 ;;
    esac
  done

  if [[ $self_test -eq 1 ]]; then
    run_self_test
    exit $?
  fi

  [[ -n "${HF_TOKEN:-${HF:-}}" ]] \
    || { echo "resilient_batch: HF_TOKEN or HF must be set in env (never on the CLI)" >&2; exit 2; }
  [[ -x "$VOKRA_ROOT/target/release/vokra-cli" ]] \
    || { echo "resilient_batch: $VOKRA_ROOT/target/release/vokra-cli missing — run provision.sh first" >&2; exit 2; }

  # Optional subset selector.
  local -a only_slugs=()
  if [[ -n "$only" ]]; then
    IFS=',' read -r -a only_slugs <<<"$only"
  fi

  step "Wave 9 residual retry: 7 models (push=$push, only='${only:-*}')"
  log "HF_HUB_ENABLE_HF_TRANSFER=$HF_HUB_ENABLE_HF_TRANSFER"
  log "HF_HUB_DISABLE_XET=$HF_HUB_DISABLE_XET"
  log "HF_ENDPOINT=$HF_ENDPOINT"
  log "HF_HUB_DOWNLOAD_TIMEOUT=$HF_HUB_DOWNLOAD_TIMEOUT"

  local -a failed=()
  local row
  for row in "${TARGETS[@]}"; do
    local IFS='|'
    read -r -a parts <<<"$row"
    IFS=$' \t\n'
    local slug="${parts[0]}"
    local repo="${parts[1]}"
    local kind="${parts[2]}"
    local license="${parts[3]}"
    local extra="${parts[4]:-}"

    if [[ ${#only_slugs[@]} -gt 0 ]]; then
      local match=0 s
      for s in "${only_slugs[@]}"; do
        [[ "$s" == "$slug" ]] && match=1 && break
      done
      if [[ $match -eq 0 ]]; then
        log "skip: $slug (not in --only list)"
        continue
      fi
    fi

    if ! process_one "$slug" "$repo" "$kind" "$license" "$extra" "$push"; then
      log "FAIL: $slug — continuing to next model"
      failed+=("$slug")
    fi
  done

  step "batch complete"
  if [[ ${#failed[@]} -gt 0 ]]; then
    log "FAILED slugs: ${failed[*]}"
    log "re-run with --only ${failed[*]// /,} once the network stabilizes"
    exit 1
  fi
  log "all targets OK"
}

main "$@"
