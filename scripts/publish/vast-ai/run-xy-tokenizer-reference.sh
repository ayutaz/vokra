#!/usr/bin/env bash
# VAST-only XY-Tokenizer official-reference worker.
#
# The worker is report-only and never uploads weights.  It intentionally stops
# at the dependency/license gate until a dedicated exact uv.lock and
# version-keyed primary-source license review are supplied.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
REFERENCE="$ROOT/tools/parity/xy_tokenizer_dump_reference.py"
DEPENDENCY_PROJECT="$ROOT/tools/parity/xy_tokenizer_reference"

die() { echo "xy-tokenizer-reference: ERROR: $*" >&2; exit 2; }

usage() {
  cat >&2 <<'EOF'
usage: run-xy-tokenizer-reference.sh --source-dir DIR --checkpoint FILE \
  --config FILE --output EMPTY_TMPFS_DIR
       run-xy-tokenizer-reference.sh --self-test

The real route is Linux/x86_64 VAST only and requires VOKRA_PUBLISH_ON_VAST=1.
It calls the exact official XY_Tokenizer methods and emits same-execution f32
taps only after dependency/license evidence is affirmatively reviewed.  It
does not download, publish, or upload any model.  An existing empty output is
accepted; an absent output is created only after the model-free dependency
gate succeeds.
EOF
}

self_test() {
  command -v uv >/dev/null 2>&1 || die "uv is required"
  [[ -f "$REFERENCE" && ! -L "$REFERENCE" ]] || die "reference dumper is missing or symlinked"
  uv run --no-project --python 3.12 python "$REFERENCE" --self-test
  local token
  for token in \
    'SOURCE_REVISION = "5df5609c5883e555bd39a2d0b1005ca8f1a8f12e"' \
    'UPSTREAM_REVISION = "c83433728e698ed0698e88cb5096bc221fb8f8c5"' \
    'CHECKPOINT_SHA256' 'CONFIG_SHA256' 'DEPENDENCY_CLOSURE_LICENSE_UNVERIFIED_BLOCKER' \
    'inference_tokenize' 'inference_detokenize' 'feature_extractor' 'semantic_encoder' \
    'acoustic_encoder' 'quantizer' 'acoustic_decoder' 'vocos' 'input_waveform.f32' \
    'dependency_audit.json' 'uv.lock' '--dependency-audit' 'NO_UPLOAD' 'must be empty'; do
    grep -Fq -- "$token" "$REFERENCE" || die "reference contract lost token: $token"
  done
  grep -Fq -- 'after every model-free gate' "${BASH_SOURCE[0]}" \
    || die "worker output-creation gate comment is missing"
  grep -Fq -- "mkdir -p \"\$output\"" "${BASH_SOURCE[0]}" \
    || die "worker output creation is missing"
  if grep -Eq '^[[:space:]]*(python3?|pip)([[:space:]]|$)' "$REFERENCE"; then
    die "direct Python/pip invocation found in reference"
  fi
  echo "run-xy-tokenizer-reference.sh self-test: OK (model-free contract checks)"
}

source_dir=""
checkpoint=""
config=""
output=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --source-dir) [[ $# -ge 2 ]] || die "--source-dir requires a path"; source_dir="$2"; shift 2 ;;
    --checkpoint) [[ $# -ge 2 ]] || die "--checkpoint requires a path"; checkpoint="$2"; shift 2 ;;
    --config) [[ $# -ge 2 ]] || die "--config requires a path"; config="$2"; shift 2 ;;
    --output) [[ $# -ge 2 ]] || die "--output requires a path"; output="$2"; shift 2 ;;
    --self-test) [[ $# == 1 ]] || die "--self-test accepts no arguments"; self_test; exit 0 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ -n "$source_dir" && -n "$checkpoint" && -n "$config" && -n "$output" ]] \
  || { usage; exit 2; }
for path in "$source_dir" "$checkpoint" "$config" "$output"; do
  [[ "$path" = /* ]] || die "all source/model/output paths must be absolute"
  [[ ! -L "$path" ]] || die "source/model/output paths must not be symlinks"
done
[[ ! -L "$DEPENDENCY_PROJECT" ]] || die "DEPENDENCY_PROJECT must not be a symlink"
[[ -d "$DEPENDENCY_PROJECT" ]] || die "dedicated dependency project is absent"
for path in "$DEPENDENCY_PROJECT/pyproject.toml" "$DEPENDENCY_PROJECT/uv.lock" \
  "$DEPENDENCY_PROJECT/dependency_audit.json"; do
  [[ ! -L "$path" ]] || die "dependency project files must not be symlinks: $path"
  [[ -f "$path" ]] || die "dependency project file is absent: $path"
done
[[ -d "$(dirname "$output")" && ! -L "$(dirname "$output")" ]] \
  || die "output parent must be an existing non-symlink directory"
[[ "$(uname -s)" == "Linux" ]] || die "official model execution is Linux/VAST-only"
[[ "$(uname -m)" == "x86_64" ]] || die "official model execution requires x86_64"
[[ "${VOKRA_PUBLISH_ON_VAST:-0}" == "1" ]] || die "VOKRA_PUBLISH_ON_VAST=1 is absent"
[[ -d "$ROOT/.git" ]] || die "not a Vokra checkout"
[[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] \
  || die "VAST checkout must be clean"
command -v findmnt >/dev/null 2>&1 || die "findmnt is required"
[[ "$(findmnt -T "$(dirname "$output")" -no FSTYPE 2>/dev/null || true)" == "tmpfs" ]] \
  || die "output parent must be tmpfs/RAM disk"
[[ ! -e "$output" || -z "$(find "$output" -mindepth 1 -print -quit 2>/dev/null)" ]] \
  || die "output must be absent or empty"

[[ -e "$source_dir" && -e "$checkpoint" && -e "$config" ]] \
  || die "source/checkpoint/config must exist for canonical disjointness check"
output_parent_real="$(realpath -e "$(dirname "$output")")" \
  || die "output parent cannot be canonicalized"
output_real="$output_parent_real/$(basename "$output")"
for path in "$source_dir" "$checkpoint" "$config" "$DEPENDENCY_PROJECT"; do
  path_real="$(realpath -e "$path")" || die "path cannot be canonicalized: $path"
  if [[ "$output_real" == "$path_real" || "$output_real" == "$path_real"/* \
    || "$path_real" == "$output_real"/* ]]; then
    die "output must be canonical and disjoint from source/checkpoint/config/dependency project"
  fi
done

# This is a deliberate fail-closed gate.  The source requirements are
# unpinned, and the broad parity lock has no XY-specific, version-keyed
# primary-source license review for the torchaudio/librosa/scipy closure.
[[ -f "$DEPENDENCY_PROJECT/pyproject.toml" && -f "$DEPENDENCY_PROJECT/uv.lock" \
  && -f "$DEPENDENCY_PROJECT/dependency_audit.json" ]] \
  || die "DEPENDENCY_CLOSURE_LICENSE_UNVERIFIED_BLOCKER: no exact reviewed closure; no source/model execution"

# Validate the complete audit before creating an absent output directory.
UV_CACHE_DIR="${XY_UV_CACHE_DIR:-/private/tmp/vokra-xy-uv-cache}" \
  uv run --no-project --python 3.12 python "$REFERENCE" \
  --dependency-audit --dependency-project "$DEPENDENCY_PROJECT" \
  || die "DEPENDENCY_CLOSURE_LICENSE_UNVERIFIED_BLOCKER: dependency audit failed"
# This mkdir is deliberately after every model-free gate; the dumper accepts
# this now-empty directory and never receives a pre-existing payload.
mkdir -p "$output"
uv run --frozen --project "$DEPENDENCY_PROJECT" --python 3.12 python "$REFERENCE" \
  --source-dir "$source_dir" --checkpoint "$checkpoint" --config "$config" --output "$output" \
  --dependency-project "$DEPENDENCY_PROJECT"
[[ -s "$output/manifest.json" ]] || die "reference manifest was not produced"
echo "XY official reference evidence: $output (NO_UPLOAD)"
