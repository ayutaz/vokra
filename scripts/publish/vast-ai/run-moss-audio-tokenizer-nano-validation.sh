#!/usr/bin/env bash
# VAST-only conversion and real-weight measurement for MOSS Audio Tokenizer Nano.
# This worker never publishes or uploads an artifact.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VOKRA_ROOT="${VOKRA_ROOT:-$DEFAULT_ROOT}"
VOKRA_SCRATCH="${VOKRA_SCRATCH:-$HOME/scratchpad}"
PARITY_PROJECT="$VOKRA_ROOT/tools/parity"
NANO_PROJECT="$PARITY_PROJECT/moss_audio_tokenizer_nano"
LICENSE_GATE="$NANO_PROJECT/license_gate.py"
LICENSE_MANIFEST="$NANO_PROJECT/license_gate_manifest.json"
PREPARER="$PARITY_PROJECT/moss_audio_tokenizer_prepare_checkpoint.py"
REFERENCE_DUMPER="$PARITY_PROJECT/moss_audio_tokenizer_dump_reference.py"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-1}"

UPSTREAM_REPO="OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano"
UPSTREAM_REVISION="6aa02b01e445cc585582cf0ba480bc3ea6c8dd68"
CORRECTED_MODEL_NAME="moss-audio-tokenizer-nano"
CORRECTED_VARIANT="nano"
LEGACY_PUBLIC_REPO="vokra/moss-audio-tokenizer-nano"
LEGACY_NOTE="historical public Nano GGUF is manifest-authenticated but mis-stamped; never canonical"
# The official custom-code module paths and the Transformers route have not
# been authenticated for this revision.  These code-bound sentinels are
# deliberate: a reference cannot pass until the owner replaces them together
# with reviewed identities and the gate manifest.
EXPECTED_MODEL_SOURCE_PATH="UNRESOLVED"
EXPECTED_CONFIG_SOURCE_PATH="UNRESOLVED"
EXPECTED_MODEL_SOURCE_SHA256="UNRESOLVED"
EXPECTED_CONFIG_SOURCE_SHA256="UNRESOLVED"
EXPECTED_TORCH_VERSION="UNRESOLVED"
EXPECTED_TRANSFORMERS_VERSION="UNRESOLVED"
EXPECTED_QUANTIZER_SHAPE="UNRESOLVED"
EXPECTED_DECODER_TAP_COUNT="UNRESOLVED"
EXPECTED_DECODER_TAP_SHAPES="UNRESOLVED"
EXPECTED_CODES="17,520,1023,502,1005,484,987,466,969,448,951,430,933,412,915,394,274,777,256,759,238,741,220,723,202,705,184,687,166,669,148,651"

MIN_VAST_MEM_KIB=30_000_000
MIN_FREE_DISK_KIB=5_000_000

log() { printf '[moss-tokenizer-nano-vast] %s\n' "$*" >&2; }
step() { printf '\n[moss-tokenizer-nano-vast] ==== %s ====\n' "$*" >&2; }
die() { log "ERROR: $*"; return 2; }

usage() {
  cat <<'EOF' >&2
usage: run-moss-audio-tokenizer-nano-validation.sh --approval-evidence <file> [--work-dir <absent-dir>]
       run-moss-audio-tokenizer-nano-validation.sh --self-test

VAST-only corrected replacement validation for the immutable official Nano
release. It snapshots the pinned upstream revision, prepares and converts a
correctly stamped GGUF on VAST, generates the independent official reference,
and runs the named real-weight CPU test. The Nano numeric bound is not yet
reviewed, so the verdict remains MEASURED_NOT_GATED. The historical public
vokra/moss-audio-tokenizer-nano file is never treated as canonical.

Actual runs require Linux, VOKRA_PUBLISH_ON_VAST=1, at least 30,000,000 KiB
RAM, and 5,000,000 KiB free disk. No publication or upload operation exists
in this worker; pull evidence and destroy the VAST instance afterward.
EOF
}

sha256_file() {
  local path="$1"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$path" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$path" | awk '{print $1}'
  else
    die "neither sha256sum nor shasum is available"
  fi
}

license_preflight() {
  local approval="$1"
  command -v uv >/dev/null 2>&1 || { die "required tool missing: uv"; return 2; }
  [[ -f "$LICENSE_GATE" && -f "$LICENSE_MANIFEST" ]] || { die "Nano approval gate/manifest is missing"; return 2; }
  [[ -f "$approval" && ! -L "$approval" ]] || { die "--approval-evidence must be a regular non-symlink file"; return 2; }
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python "$LICENSE_GATE" \
    --lock "$NANO_PROJECT/uv.lock" --project "$NANO_PROJECT/pyproject.toml" \
    --manifest "$LICENSE_MANIFEST" --approval-evidence "$approval"
}

canonical_candidate() {
  local value="$1" suffix='' parent
  [[ "$value" = /* ]] || value="$PWD/$value"
  value="${value%/}"
  [[ -n "$value" ]] || { die "path is empty"; return 2; }
  parent="$value"
  while [[ "$parent" != / ]]; do
    [[ ! -L "$parent" ]] || { die "path contains a symlink ancestor: $parent"; return 2; }
    parent="$(dirname "$parent")"
  done
  while [[ ! -e "$value" && ! -L "$value" ]]; do
    parent="$(dirname "$value")"
    suffix="/$(basename "$value")$suffix"
    [[ "$parent" != "$value" ]] || { die "path has no canonical parent"; return 2; }
    value="$parent"
  done
  [[ -d "$value" && ! -L "$value" ]] || { die "path parent is not a real directory"; return 2; }
  (cd -P "$value" && printf '%s%s\n' "$PWD" "$suffix")
}

canonical_existing_path() {
  local value="$1" parent
  [[ "$value" = /* ]] || value="$PWD/$value"
  value="${value%/}"
  [[ -e "$value" && ! -L "$value" ]] || return 1
  parent="$value"
  while [[ "$parent" != / ]]; do
    [[ ! -L "$parent" ]] || return 1
    parent="$(dirname "$parent")"
  done
  if [[ -d "$value" ]]; then
    (cd -P "$value" && printf '%s\n' "$PWD")
  else
    parent="$(dirname "$value")"
    (cd -P "$parent" && printf '%s/%s\n' "$PWD" "$(basename "$value")")
  fi
}

paths_overlap() { local left="${1%/}" right="${2%/}"; [[ "$left" == "$right" || "$left" == "$right"/* || "$right" == "$left"/* ]]; }

validate_work_dir() {
  local work="$1" approval="$2" canonical_work canonical_root canonical_project approval_real
  [[ "$work" = /* ]] || { die "--work-dir must be an absolute path"; return 2; }
  [[ ! -e "$work" && ! -L "$work" ]] || { die "--work-dir must be absent/nonexistent"; return 2; }
  canonical_work="$(canonical_candidate "$work")" || return 2
  canonical_root="$(canonical_candidate "$VOKRA_ROOT")" || return 2
  canonical_project="$(canonical_candidate "$NANO_PROJECT")" || return 2
  [[ -f "$approval" && ! -L "$approval" ]] || { die "approval evidence must be a regular non-symlink file"; return 2; }
  approval_real="$(canonical_existing_path "$approval")" || { die "approval evidence path contains a symlink ancestor"; return 2; }
  paths_overlap "$canonical_work" "$canonical_root" && { die "--work-dir overlaps checkout"; return 2; }
  paths_overlap "$canonical_work" "$canonical_project" && { die "--work-dir overlaps project"; return 2; }
  paths_overlap "$canonical_work" "$approval_real" && { die "--work-dir overlaps approval"; return 2; }
}

require_vast_host() {
  local mem_kib free_kib disk_path parent
  [[ "${VOKRA_PUBLISH_ON_VAST:-0}" == "1" ]] \
    || die "VOKRA_PUBLISH_ON_VAST=1 is absent; run provision.sh first"
  [[ "$(uname -s)" == "Linux" ]] \
    || die "Nano conversion/parity is VAST-only; refusing host $(uname -s)"
  [[ -r /proc/meminfo ]] || die "/proc/meminfo is unavailable"
  mem_kib="$(awk '$1 == "MemTotal:" {print $2; exit}' /proc/meminfo)"
  [[ "$mem_kib" =~ ^[0-9]+$ ]] || die "could not read MemTotal"
  (( mem_kib >= MIN_VAST_MEM_KIB )) \
    || die "MemTotal=${mem_kib} KiB is below the Nano VAST guard"
  disk_path="$VOKRA_SCRATCH"
  while [[ ! -e "$disk_path" ]]; do
    parent="$(dirname "$disk_path")"
    [[ "$parent" != "$disk_path" ]] || die "scratch parent cannot be resolved"
    disk_path="$parent"
  done
  [[ -d "$disk_path" && ! -L "$disk_path" ]] || die "scratch filesystem path is not a real directory"
  free_kib="$(df -Pk "$disk_path" | awk 'NR == 2 {print $4}')"
  [[ "$free_kib" =~ ^[0-9]+$ ]] || die "could not read free disk"
  (( free_kib >= MIN_FREE_DISK_KIB )) \
    || die "free disk=${free_kib} KiB is below the Nano VAST guard"
}

require_tooling() {
  local tool
  for tool in uv cargo rustc git awk grep find tee wc tr sort; do
    command -v "$tool" >/dev/null 2>&1 || die "required tool missing: $tool"
  done
  [[ -d "$VOKRA_ROOT/.git" ]] || die "$VOKRA_ROOT is not a git checkout"
  [[ -f "$NANO_PROJECT/uv.lock" && -f "$NANO_PROJECT/pyproject.toml" ]] || die "dedicated Nano uv project is missing"
  [[ -f "$PREPARER" ]] || die "Nano checkpoint preparer is missing"
  [[ -f "$REFERENCE_DUMPER" ]] || die "official reference dumper is missing"
  [[ -f "$VOKRA_ROOT/Cargo.toml" ]] || die "root Cargo.toml is missing"
  if [[ -n "$(git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all)" ]]; then
    die "VAST checkout must be clean so evidence names an exact commit"
  fi
}

require_cpu_test_evidence() {
  local path="$1" named result result_lines test_lines cpu cpu_lines
  named="$(grep -Ec '^test parity_moss_audio_tokenizer_nano_real::official_nano_decode_matches_cpu_and_optional_metal \.\.\. ok$' "$path" || true)"
  result="$(grep -Ec '^test result: ok\. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out(; finished in [0-9]+\.[0-9]+s)?$' "$path" || true)"
  result_lines="$(grep -Ec '^test result:' "$path" || true)"
  test_lines="$(awk '/^test / && $0 !~ /^test result:/ {count++} END {print count + 0}' "$path")"
  cpu="$(grep -Ec '^MOSS_AUDIO_TOKENIZER_NANO_MEASUREMENT_ONLY backend=cpu numeric_bounds=UNSET verdict=MEASURED_NOT_GATED max_abs=[0-9.eE+-]+ rms=[0-9.eE+-]+ index=[0-9]+ actual=[0-9.eE+-]+ reference=[0-9.eE+-]+$' "$path" || true)"
  cpu_lines="$(grep -Ec '^MOSS_AUDIO_TOKENIZER_NANO_MEASUREMENT_ONLY backend=cpu ' "$path" || true)"
  if [[ "$named" != 1 || "$result" != 1 || "$result_lines" != 1 || "$test_lines" != 1 || "$cpu" != 1 || "$cpu_lines" != 1 ]]; then
    die 'Nano CPU evidence requires exactly one named pass/result/sentinel'; return 2
  fi
}

require_reference() {
  local path="$1" count runtime model_source config_source quantizer_shape decoder_count decoder_shapes
  awk -F, -v source_row="source,nano,$UPSTREAM_REPO,$UPSTREAM_REVISION" -v codes_row="codes,$EXPECTED_CODES" '
    $0 == source_row ||
    $0 ~ /^runtime,torch-[^,]+,transformers-[^,]+$/ ||
    $0 ~ /^environment,cpu,[^,]+,machine-[^,]+,logical-[0-9]+,torch-capability-[^,]+$/ ||
    $0 == "environment,device,cpu" ||
    ($1 == "source_file" && ($2 == "model" || $2 == "config") && NF == 4 &&
      $3 ~ /^transformers_modules\/[^,]+$/ && length($4) == 64 && $4 !~ /[^0-9a-f]/) ||
    $0 == "contract,2,16,1024,48000,2,3840" ||
    $0 == codes_row ||
    $0 ~ /^tensor,(quantizer|decoder_[0-9]+),[0-9]+(x[0-9]+)+,[-+0-9.eE]+(,[-+0-9.eE]+)*$/ ||
    $0 ~ /^tensor,audio,1x2x7680,[-+0-9.eE]+(,[-+0-9.eE]+)*$/ { next }
    { exit 1 }
  ' "$path" || { die 'reference contains an unknown or malformed row'; return 2; }
  count="$(awk -F, -v wanted="source,nano,$UPSTREAM_REPO,$UPSTREAM_REVISION" '$0 == wanted {count++} END {print count + 0}' "$path")"
  [[ "$count" == 1 ]] || { die 'reference must contain exactly one pinned official source row'; return 2; }
  for role in model config; do
    count="$(awk -F, -v role="$role" '$1 == "source_file" && $2 == role {count++} END {print count + 0}' "$path")"
    [[ "$count" == 1 ]] || { die "reference must contain exactly one $role source row"; return 2; }
    awk -F, -v role="$role" '$1 == "source_file" && $2 == role {if (NF != 4 || $3 !~ /^transformers_modules\/[^,]+$/ || length($4) != 64 || $4 ~ /[^0-9a-f]/) exit 1; found=1} END {exit(found ? 0 : 1)}' "$path" \
      || { die "reference $role source row is not authenticated"; return 2; }
  done
  runtime="$(awk -F, '$1 == "runtime" {print; count++} END {if (count != 1) exit 1}' "$path")" || { die 'reference must contain exactly one runtime row'; return 2; }
  [[ "$runtime" == "runtime,torch-${EXPECTED_TORCH_VERSION},transformers-${EXPECTED_TRANSFORMERS_VERSION}" ]] || { die 'reference Transformers route is not the reviewed exact route'; return 2; }
  [[ "$EXPECTED_TORCH_VERSION" != UNRESOLVED && "$EXPECTED_TRANSFORMERS_VERSION" != UNRESOLVED ]] || { die 'reference Transformers route is unresolved; owner review is required'; return 2; }
  model_source="$(awk -F, '$1 == "source_file" && $2 == "model" {print $3 "," $4}' "$path")"
  config_source="$(awk -F, '$1 == "source_file" && $2 == "config" {print $3 "," $4}' "$path")"
  [[ "$model_source" == "$EXPECTED_MODEL_SOURCE_PATH,$EXPECTED_MODEL_SOURCE_SHA256" && "$config_source" == "$EXPECTED_CONFIG_SOURCE_PATH,$EXPECTED_CONFIG_SOURCE_SHA256" ]] || { die 'reference source identities differ from reviewed fixed identities'; return 2; }
  [[ "$EXPECTED_MODEL_SOURCE_PATH" != UNRESOLVED && "$EXPECTED_CONFIG_SOURCE_PATH" != UNRESOLVED && "$EXPECTED_MODEL_SOURCE_SHA256" != UNRESOLVED && "$EXPECTED_CONFIG_SOURCE_SHA256" != UNRESOLVED ]] || { die 'reference source identities are unresolved; owner review is required'; return 2; }
  count="$(awk -F, '$1 == "environment" && $2 == "cpu" {count++} END {print count + 0}' "$path")"; [[ "$count" == 1 ]] || { die 'reference must contain exactly one CPU row'; return 2; }
  count="$(awk -F, '$0 == "environment,device,cpu" {count++} END {print count + 0}' "$path")"; [[ "$count" == 1 ]] || { die 'reference must contain exactly one CPU device row'; return 2; }
  count="$(awk -F, '$0 == "contract,2,16,1024,48000,2,3840" {count++} END {print count + 0}' "$path")"; [[ "$count" == 1 ]] || { die 'reference contract is not exact'; return 2; }
  count="$(awk -F, -v wanted="codes,$EXPECTED_CODES" '$0 == wanted {count++} END {print count + 0}' "$path")"; [[ "$count" == 1 ]] || { die 'reference codes are not the exact deterministic 32-code packet'; return 2; }
  count="$(awk -F, '$1 == "tensor" && $2 == "quantizer" {count++} END {print count + 0}' "$path")"; [[ "$count" == 1 ]] || { die 'reference quantizer tap missing or duplicated'; return 2; }
  quantizer_shape="$(awk -F, '$1 == "tensor" && $2 == "quantizer" {print $3}' "$path")"
  [[ "$quantizer_shape" == "$EXPECTED_QUANTIZER_SHAPE" ]] || { die 'reference quantizer shape differs from reviewed contract'; return 2; }
  [[ "$EXPECTED_QUANTIZER_SHAPE" != UNRESOLVED ]] || { die 'reference quantizer shape is unresolved; owner review is required'; return 2; }
  count="$(awk -F, '$1 == "tensor" && $2 == "audio" {count++} END {print count + 0}' "$path")"; [[ "$count" == 1 ]] || { die 'reference audio tap missing or duplicated'; return 2; }
  awk -F, '$1 == "tensor" && $2 ~ /^decoder_[0-9]+$/ {idx=$2; sub(/^decoder_/, "", idx); if (seen[idx]++) exit 1; count++} END {for (idx=0; idx<count; idx++) if (!(idx in seen)) exit 1}' "$path" || { die 'reference decoder tensor rows are not a unique contiguous sequence'; return 2; }
  decoder_count="$(awk -F, '$1 == "tensor" && $2 ~ /^decoder_[0-9]+$/ {count++} END {print count + 0}' "$path")"
  (( decoder_count > 0 )) || { die 'reference must contain a nonzero decoder tap sequence'; return 2; }
  [[ "$decoder_count" == "$EXPECTED_DECODER_TAP_COUNT" ]] || { die 'reference decoder tap count differs from reviewed contract'; return 2; }
  decoder_shapes="$(awk -F, '$1 == "tensor" && $2 ~ /^decoder_[0-9]+$/ {idx=$2; sub(/^decoder_/, "", idx); shape_by_idx[idx]=$3; count++} END {for (idx=0; idx<count; idx++) if (!(idx in shape_by_idx)) exit 1; for (idx=0; idx<count; idx++) {if (idx) printf ","; printf "%s", shape_by_idx[idx]}}' "$path")" || { die 'reference decoder shape sequence is malformed'; return 2; }
  [[ "$decoder_shapes" == "$EXPECTED_DECODER_TAP_SHAPES" ]] || { die 'reference decoder shape sequence differs from reviewed contract'; return 2; }
  [[ "$EXPECTED_DECODER_TAP_COUNT" != UNRESOLVED && "$EXPECTED_DECODER_TAP_SHAPES" != UNRESOLVED ]] || { die 'reference decoder tap contract is unresolved; owner review is required'; return 2; }
}

download_snapshot() {
  local output="$1"
  mkdir -p "$output"
  uv run --project "$NANO_PROJECT" --frozen --python 3.12 python -c \
    'import sys; from huggingface_hub import snapshot_download
snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2], local_dir=sys.argv[3], allow_patterns=["LICENSE", "README.md", "config.json", "configuration_moss_audio_tokenizer.py", "modeling_moss_audio_tokenizer.py", "model.safetensors.index.json", "model-00001-of-00001.safetensors"])' \
    "$UPSTREAM_REPO" "$UPSTREAM_REVISION" "$output"
}

verify_snapshot_contract() {
  local snapshot="$1" entry name
  for entry in "$snapshot"/* "$snapshot"/.[!.]*; do
    [[ -e "$entry" || -L "$entry" ]] || continue
    name="${entry##*/}"
    if [[ "$name" == .cache ]]; then
      [[ -d "$entry" && ! -L "$entry" ]] || die 'Nano transport .cache must be a real directory'
      continue
    fi
    case "$name" in
      LICENSE|README.md|config.json|configuration_moss_audio_tokenizer.py|modeling_moss_audio_tokenizer.py|model.safetensors.index.json|model-00001-of-00001.safetensors) ;;
      *) die "Nano snapshot contains an unexpected entry: $name" ;;
    esac
    [[ -f "$entry" && ! -L "$entry" ]] || die "Nano snapshot payload is not a regular file: $name"
  done
  for entry in LICENSE README.md config.json configuration_moss_audio_tokenizer.py \
    modeling_moss_audio_tokenizer.py model.safetensors.index.json \
    model-00001-of-00001.safetensors; do
    [[ -f "$snapshot/$entry" && ! -L "$snapshot/$entry" ]] || die "pinned Nano snapshot is missing or non-regular: $entry"
  done
  uv run --project "$NANO_PROJECT" --frozen --python 3.12 python -c \
    'import json,pathlib,sys
def reject(pairs):
 d={}
 for k,v in pairs:
  if k in d: raise ValueError(f"duplicate JSON key: {k}")
  d[k]=v
 return d
root=pathlib.Path(sys.argv[1]); config=json.loads((root/"config.json").read_text(), object_pairs_hook=reject)
assert config.get("sampling_rate") == 48000
assert config.get("downsample_rate") == 3840
assert config.get("number_channels") == 2
q=config.get("quantizer_kwargs")
assert isinstance(q,dict)
assert q.get("num_quantizers") == 16
assert q.get("codebook_size") == 1024
assert q.get("codebook_dim") == 8
assert q.get("rvq_dim") == 512
assert q.get("output_dim") == 768
index=json.loads((root/"model.safetensors.index.json").read_text(), object_pairs_hook=reject)
wm=index.get("weight_map")
assert isinstance(wm,dict) and wm
assert set(wm.values()) == {"model-00001-of-00001.safetensors"}
print("authenticated Nano source contract: 48kHz stereo, 16 quantizers, 3840 hop")' \
    "$snapshot" \
    || { die 'Nano snapshot JSON validation failed'; return 2; }
}

record_environment() {
  local output="$1"
  {
    echo "utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "git_commit=$(git -C "$VOKRA_ROOT" rev-parse HEAD)"
    echo "uname=$(uname -a)"
    awk '$1 == "MemTotal:" {print "mem_total_kib=" $2; exit}' /proc/meminfo
    rustc --version --verbose
    cargo --version
    uv --version
    uv run --project "$NANO_PROJECT" --frozen --python 3.12 python -c \
      'import platform,torch,transformers; print(f"python={platform.python_version()}"); print(f"torch={torch.__version__}"); print(f"transformers={transformers.__version__}"); print(f"cuda={torch.version.cuda}"); print(f"cuda_available={torch.cuda.is_available()}")'
  } | tee "$output"
}

write_apple_args() {
  local output="$1" gguf_sha="$2" reference_sha="$3"
  {
    printf '#!/usr/bin/env bash\nset -eu\n'
    printf '%s ' 'scripts/verify/apple-silicon-moss-audio-tokenizer-nano.sh'
    printf '%s ' --gguf "'<APPLE_MOSS_AUDIO_TOKENIZER_NANO_GGUF_PATH>'" --reference "'<APPLE_MOSS_AUDIO_TOKENIZER_NANO_REFERENCE_PATH>'"
    printf '%q ' --gguf-sha256 "$gguf_sha" --reference-sha256 "$reference_sha"
    printf '%s ' --approval-evidence "'<APPLE_MOSS_AUDIO_TOKENIZER_NANO_APPROVAL_EVIDENCE>'"
    printf '%s\n' --evidence-dir "'<APPLE_MOSS_AUDIO_TOKENIZER_NANO_EVIDENCE_DIR>'"
  } > "$output"
  chmod +x "$output"
}

run_self_test_work_paths() {
  local probe status=0
  probe="$(cd -P "$(mktemp -d)" && pwd -P)"
  mkdir -p "$probe/real/existing"
  ln -s "$probe/real" "$probe/link"
  if validate_work_dir "$probe/link/existing/nested/new" "$probe/approval.json" >/dev/null 2>&1; then
    die 'existing descendant under symlink ancestor accepted'
    status=1
  fi
  rm -rf "$probe"
  return "$status"
}

run_self_test() {
  local script_path="${BASH_SOURCE[0]}" fail=0 cases=0 required tmp fake_root fake_home fake_log rc
  run_self_test_work_paths
  tmp="$(mktemp -d)"
  trap 'rm -rf "$tmp"' EXIT
  expect_exit_2_no_path() {
    local label="$1" path="$2" status
    shift 2
    if "$@" >/dev/null 2>&1; then
      status=0
    else
      status=$?
    fi
    if [[ $status -ne 2 || -e "$path" || -L "$path" ]]; then
      log "self-test FAIL: $label was not a controlled reject without output"
      return 1
    fi
  }
  for required in \
    "$UPSTREAM_REPO" "$UPSTREAM_REVISION" "$CORRECTED_MODEL_NAME" \
    "$CORRECTED_VARIANT" "$LEGACY_PUBLIC_REPO" \
    "moss_audio_tokenizer_prepare_checkpoint.py" \
    "moss_audio_tokenizer_dump_reference.py" "--variant nano" "--num-quantizers 16" \
    "--model moss-audio-tokenizer-nano" "--frozen --python 3.12" \
    "license_preflight" "--no-project --offline" "license_gate.py" "moss_audio_tokenizer_nano" \
    "parity_moss_audio_tokenizer_nano_real" \
    "official_nano_decode_matches_cpu_and_optional_metal" \
    "numeric_bounds=UNSET" "MEASURED_NOT_GATED" "object_pairs_hook=reject"; do
    cases=$((cases + 1))
    if ! grep -Fq -- "$required" "$script_path"; then
      log "self-test FAIL: worker contract lost token: $required"
      fail=1
    fi
  done
  if ! grep -Fq 'object_pairs_hook=reject_duplicate_json_keys' "$PREPARER"; then
    log 'self-test FAIL: checkpoint index duplicate-key rejection is missing'
    fail=1
  fi
  cases=$((cases + 1))
  if ! grep -Fq "REVISION = \"$UPSTREAM_REVISION\"" "$NANO_PROJECT/license_gate.py" \
    || ! grep -Fq "\"revision\": \"$UPSTREAM_REVISION\"" "$REFERENCE_DUMPER" \
    || ! grep -Fq '(frame * 257 + quantizer * 503 + 17) % CODEBOOK_SIZE' "$REFERENCE_DUMPER"; then
    log 'self-test FAIL: worker/gate/dumper upstream revision contract diverged'; fail=1
  fi
  cases=$((cases + 1))
  if ! grep -Fq 'EXPECTED_MODEL_SOURCE_PATH="UNRESOLVED"' "$script_path" \
    || ! grep -Fq 'EXPECTED_TRANSFORMERS_VERSION="UNRESOLVED"' "$script_path" \
    || ! grep -Fq '"status": "UNRESOLVED"' "$NANO_PROJECT/license_gate.py"; then
    log 'self-test FAIL: unresolved source blocker was weakened'; fail=1
  fi
  cases=$((cases + 1))
  EXPECTED_MODEL_SOURCE_PATH='transformers_modules/OpenMOSS-Team/Nano/model.py'
  EXPECTED_CONFIG_SOURCE_PATH='transformers_modules/OpenMOSS-Team/Nano/config.py'
  EXPECTED_MODEL_SOURCE_SHA256='aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'
  EXPECTED_CONFIG_SOURCE_SHA256='bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb'
  EXPECTED_TORCH_VERSION='2.13.0'
  EXPECTED_TRANSFORMERS_VERSION='5.15.0'
  EXPECTED_QUANTIZER_SHAPE='1x1'
  EXPECTED_DECODER_TAP_COUNT='2'
  EXPECTED_DECODER_TAP_SHAPES='1x1,1x2'
  printf '%s\n' \
    "source,nano,$UPSTREAM_REPO,$UPSTREAM_REVISION" \
    'runtime,torch-2.13.0,transformers-5.15.0' \
    'environment,cpu,test,machine-x,logical-1,torch-capability-unknown' \
    'environment,device,cpu' \
    'source_file,model,transformers_modules/OpenMOSS-Team/Nano/model.py,aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa' \
    'source_file,config,transformers_modules/OpenMOSS-Team/Nano/config.py,bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb' \
    'contract,2,16,1024,48000,2,3840' \
    "codes,$EXPECTED_CODES" \
    'tensor,quantizer,1x1,0' 'tensor,decoder_0,1x1,0' \
    'tensor,decoder_1,1x2,0' 'tensor,audio,1x2x7680,0' > "$tmp/reference.csv"
  require_reference "$tmp/reference.csv" || { log 'self-test FAIL: valid decoder shape contract rejected'; fail=1; }
  for tamper in shape order extra missing hash; do
    cp "$tmp/reference.csv" "$tmp/reference-$tamper.csv"
    case "$tamper" in
      shape) sed 's/tensor,decoder_1,1x2/tensor,decoder_1,9x9/' "$tmp/reference-$tamper.csv" > "$tmp/reference-$tamper.tmp" ;;
      order) sed -e 's/tensor,decoder_0,1x1/tensor,decoder_0,1x2/' -e 's/tensor,decoder_1,1x2/tensor,decoder_1,1x1/' "$tmp/reference-$tamper.csv" > "$tmp/reference-$tamper.tmp" ;;
      extra) printf 'tensor,decoder_2,1x3,0\n' >> "$tmp/reference-$tamper.csv"; cp "$tmp/reference-$tamper.csv" "$tmp/reference-$tamper.tmp" ;;
      missing) sed '/tensor,decoder_1,/d' "$tmp/reference-$tamper.csv" > "$tmp/reference-$tamper.tmp" ;;
      hash) sed 's/^\(source_file,model,[^,]*,\)./\1A/' "$tmp/reference-$tamper.csv" > "$tmp/reference-$tamper.tmp" ;;
    esac
    mv "$tmp/reference-$tamper.tmp" "$tmp/reference-$tamper.csv"
    if require_reference "$tmp/reference-$tamper.csv"; then log "self-test FAIL: decoder $tamper tamper accepted"; fail=1; fi
  done
  write_apple_args "$tmp/apple-args.sh" \
    'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa' \
    'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb'
  bash -n "$tmp/apple-args.sh"
  grep -Fq "scripts/verify/apple-silicon-moss-audio-tokenizer-nano.sh --gguf '<APPLE_MOSS_AUDIO_TOKENIZER_NANO_GGUF_PATH>' --reference '<APPLE_MOSS_AUDIO_TOKENIZER_NANO_REFERENCE_PATH>'" "$tmp/apple-args.sh" \
    || { log 'self-test FAIL: Apple args are not portable placeholders'; fail=1; }
  grep -Fq -- "--approval-evidence '<APPLE_MOSS_AUDIO_TOKENIZER_NANO_APPROVAL_EVIDENCE>'" "$tmp/apple-args.sh" \
    || { log 'self-test FAIL: Apple approval placeholder is missing'; fail=1; }
  if grep -Eq '(/stage/|/reference/|VOKRA_ROOT=|moss-tokenizer-nano-validation/)' "$tmp/apple-args.sh"; then
    log 'self-test FAIL: Apple args embed VAST paths'; fail=1
  fi
  cases=$((cases + 1))
  if grep -En '^[[:space:]]*(python3|python|pip)([[:space:]]|$)' "$script_path" >/dev/null; then
    log "self-test FAIL: direct Python/pip command found"
    fail=1
  fi
  cases=$((cases + 1))
  if grep -En '(publish-one\.sh|upload\.sh|--push([[:space:]]|$)|huggingface-cli[[:space:]])' "$script_path" >/dev/null; then
    log "self-test FAIL: publication/upload operation found"
    fail=1
  fi
  printf approval > "$tmp/approval-target"
  ln -s "$tmp/approval-target" "$tmp/approval-link"
  if license_preflight "$tmp/approval-link" >/dev/null 2>&1; then
    log 'self-test FAIL: symlink approval was accepted'; fail=1
  fi
  cases=$((cases + 1))
  expect_exit_2_no_path 'duplicate --self-test' "$tmp/duplicate-self-test-output" \
    "$script_path" --self-test --self-test || fail=1
  cases=$((cases + 1))
  expect_exit_2_no_path 'bare --work-dir' "$tmp/bare-work-output" \
    "$script_path" --work-dir --self-test || fail=1
  cases=$((cases + 1))
  expect_exit_2_no_path 'negative --work-dir' "$tmp/negative-work-output" \
    "$script_path" --work-dir -x || fail=1
  cases=$((cases + 1))
  expect_exit_2_no_path 'empty --work-dir' "$tmp/empty-work-output" \
    "$script_path" --work-dir "" || fail=1
  cases=$((cases + 1))
  expect_exit_2_no_path 'trailing argument' "$tmp/trailing-output" \
    "$script_path" --self-test trailing || fail=1
  mkdir -p "$tmp/real/existing"
  ln -s "$tmp/real" "$tmp/link"
  printf '{}\n' > "$tmp/approval.json"
  cases=$((cases + 1))
  expect_exit_2_no_path 'relative work path' 'relative-nano-work' \
    validate_work_dir 'relative-nano-work' "$tmp/approval.json" || fail=1
  cases=$((cases + 1))
  expect_exit_2_no_path 'work path under symlink ancestor' "$tmp/link/existing/nested/new" \
    validate_work_dir "$tmp/link/existing/nested/new" "$tmp/approval.json" || fail=1
  mkdir -p "$tmp/approval-real"
  printf '{}\n' > "$tmp/approval-real/evidence.json"
  ln -s "$tmp/approval-real" "$tmp/approval-parent-link"
  cases=$((cases + 1))
  expect_exit_2_no_path 'approval path under symlink ancestor' "$tmp/approval-work" \
    validate_work_dir "$tmp/approval-work" "$tmp/approval-parent-link/evidence.json" || fail=1
  cases=$((cases + 1))
  expect_exit_2_no_path 'lexical checkout overlap' "$NANO_PROJECT/../nano-lexical-work" \
    validate_work_dir "$NANO_PROJECT/../nano-lexical-work" "$tmp/approval.json" || fail=1
  mkdir "$tmp/empty-work"
  cases=$((cases + 1))
  if validate_work_dir "$tmp/empty-work" "$tmp/approval.json" >/dev/null 2>&1; then
    log 'self-test FAIL: pre-existing empty Nano work directory accepted'; fail=1
  fi
  cases=$((cases + 1))
  expect_exit_2_no_path 'checkout-overlapping work path' "$VOKRA_ROOT/nano-self-test-work" \
    validate_work_dir "$VOKRA_ROOT/nano-self-test-work" "$tmp/approval.json" || fail=1
  cases=$((cases + 1))
  expect_exit_2_no_path 'approval-overlapping work path' "$tmp/approval.json/child" \
    validate_work_dir "$tmp/approval.json/child" "$tmp/approval.json" || fail=1
  cases=$((cases + 1))
  printf 'test parity_moss_audio_tokenizer_nano_real::official_nano_decode_matches_cpu_and_optional_metal ... ok\ntest result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out\nMOSS_AUDIO_TOKENIZER_NANO_MEASUREMENT_ONLY backend=cpu numeric_bounds=UNSET verdict=MEASURED_NOT_GATED max_abs=1.0e-9 rms=1.0e-9 index=0 actual=1.0e-9 reference=1.0e-9\n' > "$tmp/cpu.log"
  require_cpu_test_evidence "$tmp/cpu.log" || { log 'self-test FAIL: valid CPU evidence rejected'; fail=1; }
  cases=$((cases + 1))
  cp "$tmp/cpu.log" "$tmp/duplicate-result.log"
  printf 'test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out\n' >> "$tmp/duplicate-result.log"
  if require_cpu_test_evidence "$tmp/duplicate-result.log"; then log 'self-test FAIL: duplicate result accepted'; fail=1; fi
  cases=$((cases + 1))
  awk 'NR == 2 { print "test result: ok. 1 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out"; next } { print }' "$tmp/cpu.log" > "$tmp/malformed-result.log"
  if require_cpu_test_evidence "$tmp/malformed-result.log"; then log 'self-test FAIL: malformed result accepted'; fail=1; fi
  cases=$((cases + 1))
  printf 'test parity_moss_audio_tokenizer_nano_real::official_nano_decode_matches_cpu_and_optional_metal ... ok\ntest extra_case ... ok\ntest result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out\nMOSS_AUDIO_TOKENIZER_NANO_MEASUREMENT_ONLY backend=cpu numeric_bounds=UNSET verdict=MEASURED_NOT_GATED max_abs=1.0e-9 rms=1.0e-9 index=0 actual=1.0e-9 reference=1.0e-9\n' > "$tmp/extra-test.log"
  if require_cpu_test_evidence "$tmp/extra-test.log"; then log 'self-test FAIL: extra test accepted'; fail=1; fi
  cases=$((cases + 1))
  sed 's/\.\.\. ok$/.\.\.\. FAILED/' "$tmp/cpu.log" > "$tmp/failed-test.log"
  if require_cpu_test_evidence "$tmp/failed-test.log"; then log 'self-test FAIL: failed named test accepted'; fail=1; fi
  cases=$((cases + 1))
  sed 's/filtered out$/filtered out; finished in nope/' "$tmp/cpu.log" > "$tmp/bad-timing.log"
  if require_cpu_test_evidence "$tmp/bad-timing.log"; then log 'self-test FAIL: malformed timing accepted'; fail=1; fi
  cases=$((cases + 1))
  for bad in duplicate prefix suffix fail; do
    cp "$tmp/cpu.log" "$tmp/$bad-sentinel.log"
    case "$bad" in
      duplicate) cat "$tmp/cpu.log" >> "$tmp/$bad-sentinel.log" ;;
      prefix) sed 's/^MOSS_AUDIO_TOKENIZER_NANO_MEASUREMENT_ONLY /xMOSS_AUDIO_TOKENIZER_NANO_MEASUREMENT_ONLY /' "$tmp/$bad-sentinel.log" > "$tmp/$bad-sentinel.tmp"; mv "$tmp/$bad-sentinel.tmp" "$tmp/$bad-sentinel.log" ;;
      suffix) sed 's/$/ trailing/' "$tmp/$bad-sentinel.log" > "$tmp/$bad-sentinel.tmp"; mv "$tmp/$bad-sentinel.tmp" "$tmp/$bad-sentinel.log" ;;
      fail) sed 's/verdict=MEASURED_NOT_GATED/verdict=FAIL/' "$tmp/$bad-sentinel.log" > "$tmp/$bad-sentinel.tmp"; mv "$tmp/$bad-sentinel.tmp" "$tmp/$bad-sentinel.log" ;;
    esac
    if require_cpu_test_evidence "$tmp/$bad-sentinel.log"; then log "self-test FAIL: $bad CPU sentinel accepted"; fail=1; fi
  done

  fake_root="$tmp/root"; fake_home="$tmp/home"; fake_log="$tmp/fake-uv.log"
  mkdir -p "$fake_root/tools/parity/moss_audio_tokenizer_nano" "$fake_home/.local/bin"
  cp "$NANO_PROJECT/license_gate.py" "$NANO_PROJECT/license_gate_manifest.json" "$NANO_PROJECT/uv.lock" "$NANO_PROJECT/pyproject.toml" "$fake_root/tools/parity/moss_audio_tokenizer_nano/"
  cat > "$fake_home/.local/bin/uv" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' "$*" >> "${MOSS_NANO_SELF_TEST_UV_LOG:?}"
exit 2
EOF
  chmod +x "$fake_home/.local/bin/uv"
  printf '{}' > "$tmp/approval.json"
  set +e
  HOME="$fake_home" PATH="$fake_home/.local/bin:$PATH" MOSS_NANO_SELF_TEST_UV_LOG="$fake_log" \
    VOKRA_ROOT="$fake_root" VOKRA_SCRATCH="$tmp/scratch" "$script_path" \
      --approval-evidence "$tmp/approval.json" --work-dir "$tmp/blocked-work" >"$tmp/worker.log" 2>&1
  rc=$?
  set -e
  if [[ $rc -ne 2 || ! -s "$fake_log" || -e "$tmp/scratch" ]]; then
    log 'self-test FAIL: Nano gate did not block before host/scratch'; fail=1
  fi
  rm -rf "$tmp"
  trap - EXIT
  if [[ $fail -eq 0 ]]; then
    echo "run-moss-audio-tokenizer-nano-validation.sh self-test: OK ($cases cases)"
    return 0
  fi
  return 1
}

on_exit() {
  local rc=$?
  if [[ -n "${summary_file:-}" && ! -f "$summary_file" ]]; then
    printf 'execution_status=FAIL\nexit_code=%s\n' "$rc" > "$summary_file"
  fi
  exit "$rc"
}

main() {
  local self_test=0 requested_work_dir="" approval_evidence="" run_stamp work_dir snapshot stage logs reference
  local seen_work_dir=0 seen_self_test=0 seen_approval=0
  local merged gguf reference_csv reference_sha256 gguf_sha256 run_log env_log summary_file prep_manifest cpu_log
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --approval-evidence)
        (( ! seen_approval++ )) || { die "duplicate --approval-evidence"; return 2; }
        [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || { die "--approval-evidence requires a file"; return 2; }
        approval_evidence="$2"; shift 2 ;;
      --work-dir)
        (( ! seen_work_dir++ )) || { die "duplicate --work-dir"; return 2; }
        [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || { die "--work-dir requires a directory"; return 2; }
        requested_work_dir="$2"
        shift 2
        ;;
      --self-test)
        (( ! seen_self_test++ )) || { die "duplicate --self-test"; return 2; }
        self_test=1; shift ;;
      -h|--help) usage; return 0 ;;
      *) die "unknown argument: $1"; usage; return 2 ;;
    esac
  done
  if [[ $self_test -eq 1 ]]; then
    [[ -z "$approval_evidence$requested_work_dir" ]] || { die "--self-test accepts no other arguments"; return 2; }
    run_self_test
    return $?
  fi

  # The dependency/license gate is the first substantive production action.
  # It runs before host probing, scratch/cache creation, sync, download,
  # conversion, Cargo, or CUDA work.
  [[ -n "$approval_evidence" ]] || { die "--approval-evidence is required"; usage; return 2; }
  [[ -f "$approval_evidence" && ! -L "$approval_evidence" ]] || { die "--approval-evidence must be a regular non-symlink file"; return 2; }
  license_preflight "$approval_evidence"
  require_tooling
  run_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  work_dir="${requested_work_dir:-$VOKRA_SCRATCH/moss-tokenizer-nano-validation/$run_stamp}"
  validate_work_dir "$work_dir" "$approval_evidence"
  require_vast_host
  snapshot="$work_dir/upstream"
  stage="$work_dir/stage"
  logs="$work_dir/logs"
  reference="$work_dir/reference"
  merged="$stage/moss-audio-tokenizer-nano.safetensors"
  gguf="$stage/moss-audio-tokenizer-nano.gguf"
  prep_manifest="$merged.stripped-manifest.json"
  reference_csv="$reference/moss-audio-tokenizer-nano-reference.csv"
  mkdir -p "$snapshot" "$stage" "$logs" "$reference"
  export UV_CACHE_DIR="$VOKRA_SCRATCH/uv-cache-moss-tokenizer-nano"
  export HF_HOME="$VOKRA_SCRATCH/hf-home-moss-tokenizer-nano"
  run_log="$logs/run.log"
  env_log="$logs/environment.txt"
  summary_file="$logs/summary.txt"
  cpu_log="$logs/cpu-measurement.log"
  exec > >(tee -a "$run_log") 2>&1
  trap on_exit EXIT

  step "Sync locked Python 3.12 parity environment"
  uv sync --project "$NANO_PROJECT" --frozen --python 3.12

  step "Download immutable official Nano snapshot"
  download_snapshot "$snapshot"
  uv run --project "$NANO_PROJECT" --frozen --python 3.12 python "$LICENSE_GATE" \
    --verify-snapshot --snapshot "$snapshot" --manifest "$LICENSE_MANIFEST"
  verify_snapshot_contract "$snapshot"

  step "Hash pinned source inputs"
  (cd "$snapshot" && find . -maxdepth 1 -type f -print | sort | xargs sha256sum | tee "$logs/source-SHA256SUMS")

  step "Merge the exact Nano shard into one converter input"
  uv run --project "$NANO_PROJECT" --frozen --python 3.12 python "$PREPARER" \
    --hf-repo "$UPSTREAM_REPO" --revision "$UPSTREAM_REVISION" \
    --local-dir "$snapshot" --output "$merged"
  [[ -f "$prep_manifest" ]] || die "preparer did not emit stripped manifest"
  uv run --project "$NANO_PROJECT" --frozen --python 3.12 python -c \
    'import json,pathlib,sys
def reject(pairs):
 d={}
 for k,v in pairs:
  if k in d: raise ValueError(f"duplicate JSON key: {k}")
  d[k]=v
 return d
data=json.loads(pathlib.Path(sys.argv[1]).read_text(), object_pairs_hook=reject)
assert data["hf_repo"] == sys.argv[2]
assert data["kept_count"] > 0
assert data["unknown_stripped"] == []
print(f"prepared Nano tensors={data[\"kept_count\"]} sha256={data[\"sha256\"]}")' \
    "$prep_manifest" "$UPSTREAM_REPO" \
    || { die 'Nano preparation manifest JSON validation failed'; return 2; }

  step "Build vokra-cli on VAST"
  cargo build --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release -p vokra-cli

  step "Convert only the corrected Nano replacement"
  "$VOKRA_ROOT/target/release/vokra-cli" convert \
    --model moss-audio-tokenizer-nano --input "$merged" --output "$gguf"
  [[ -s "$gguf" ]] || die "corrected Nano GGUF is empty"

  step "Record environment before independent reference"
  record_environment "$env_log"

  step "Generate independent official Nano CPU reference"
  uv run --project "$NANO_PROJECT" --frozen --python 3.12 python \
    "$REFERENCE_DUMPER" --variant nano --frames 2 --num-quantizers 16 \
    --device cpu --output "$reference_csv"
  require_reference "$reference_csv"

  step "Run named nonzero native CPU validation"
  cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release \
    -p vokra-models --test parity_moss_audio_tokenizer_nano_real \
    official_nano_decode_matches_cpu_and_optional_metal \
    -- --ignored --exact --nocapture --test-threads=1 2>&1 | tee "$cpu_log"
  require_cpu_test_evidence "$cpu_log"

  step "Write evidence summary and checksums"
  reference_sha256="$(sha256_file "$reference_csv")"
  gguf_sha256="$(sha256_file "$gguf")"
  write_apple_args "$logs/apple-silicon-moss-audio-tokenizer-nano-args.sh" \
    "$gguf_sha256" "$reference_sha256"
  {
    echo "execution_status=PASS"
    echo "scope=CORRECTED_NANO_ARTIFACT_AND_INDEPENDENT_REFERENCE"
    echo "numeric_verdict=MEASURED_NOT_GATED"
    echo "numeric_bounds=UNSET"
    echo "git_commit=$(git -C "$VOKRA_ROOT" rev-parse HEAD)"
    echo "upstream_repo=$UPSTREAM_REPO"
    echo "upstream_revision=$UPSTREAM_REVISION"
    echo "corrected_model_name=$CORRECTED_MODEL_NAME"
    echo "corrected_variant=$CORRECTED_VARIANT"
    echo "legacy_public_repo=$LEGACY_PUBLIC_REPO"
    echo "legacy_public_note=$LEGACY_NOTE"
    echo "gguf_sha256=$gguf_sha256"
    echo "reference_sha256=$reference_sha256"
    grep -F "MOSS_AUDIO_TOKENIZER_NANO_MEASUREMENT_ONLY backend=cpu" "$cpu_log"
  } | tee "$summary_file"
  (
    cd "$work_dir"
    find upstream stage logs reference -type f ! -name SHA256SUMS -print0 \
      | sort -z | xargs -0 sha256sum > logs/SHA256SUMS
  )
  trap - EXIT
  log "PASS: pull $logs and $reference, then destroy the VAST instance"
}

main "$@"
