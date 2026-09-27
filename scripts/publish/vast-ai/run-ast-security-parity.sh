#!/usr/bin/env bash
# Measure the AST dependency-update candidate against the official HF oracle.
# This worker is Linux/VAST-only and never publishes or uploads anything.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VOKRA_ROOT="${VOKRA_ROOT:-$DEFAULT_ROOT}"
VOKRA_SCRATCH="${VOKRA_SCRATCH:-$HOME/scratchpad}"
PARITY_PROJECT="$VOKRA_ROOT/tools/parity/ast"
REFERENCE_SCRIPT="$PARITY_PROJECT/dump_reference.py"
FIXTURE_DIR="$VOKRA_ROOT/tests/fixtures/ast"
INPUT_WAV="$VOKRA_ROOT/tests/fixtures/audio/jfk-30s.wav"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

PUBLIC_REPO="vokra/ast-finetuned-audioset"
PUBLIC_REVISION="b23eb8b8fdc5514b911afd18077fe00618932b13"
PUBLIC_FILE="ast.gguf"
PUBLIC_SHA256="f06bf05078d4267193554ec76e143f8541bd3130c3a81ae2a3d6b5424c8b1ac2"

UPSTREAM_REPO="MIT/ast-finetuned-audioset-10-10-0.4593"
UPSTREAM_REVISION="f826b80d28226b62986cc218e5cec390b1096902"
UPSTREAM_FILE="model.safetensors"
UPSTREAM_SHA256="ae0c1e2ad4e1381d851fa9bf298ba13ebc9c5a914cdee2dbe427a6583869924d"
INPUT_SHA256="58adb4ea501d955fcd40bfbb69128f8f40428b81d8716b9ed337949773be253f"

TORCH_VERSION="2.13.0"
TORCHAUDIO_VERSION="2.11.0"
TRANSFORMERS_VERSION="5.10.4"
TRANSFORMERS_WHEEL_SHA256="8c5b99b141b53619435a76629b0284f04d27ff46d788b463fc0ecb23b8ff130e"
FEATURE_SOURCE_SHA256="ab4957749b5113067413dcd662dc212952b9a610d297e8b4515e2cab1ff1fce4"
MODELING_SOURCE_SHA256="5ef9fe1c7847400453095c158c76191913226788eaa1f4ba6afbb378b9e70547"

log() { printf '[ast-security-vast] %s\n' "$*" >&2; }
step() { printf '\n[ast-security-vast] ==== %s ====\n' "$*" >&2; }
die() { log "ERROR: $*"; return 2; }

usage() {
  cat <<'EOF' >&2
usage: run-ast-security-parity.sh [--work-dir <empty-dir>]
       run-ast-security-parity.sh --self-test

VAST-only AST dependency-update candidate worker. It verifies the exact public
GGUF, immutable upstream AST checkpoint, input WAV, clean git HEAD, and locked
Python versions before generating a candidate official reference. The candidate
fixture is installed only temporarily in the disposable VAST checkout while
the ignored real-GGUF CPU parity test runs; the original committed fixture is
restored before the worker exits. No upload or publish operation exists.

Actual runs require Linux, VOKRA_PUBLISH_ON_VAST=1 from provision.sh, uv, and
Cargo. Run --self-test on a maintainer machine for the model-free contract
check; never run the actual worker locally.
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

verify_sha256() {
  local path="$1" expected="$2" actual
  [[ -f "$path" ]] || die "missing pinned input: $path"
  actual="$(sha256_file "$path")"
  if [[ "$actual" != "$expected" ]]; then
    die "SHA-256 mismatch for $path: got $actual, expected $expected"
    return 2
  fi
  log "identity OK: $path sha256=$actual"
}

download_hf_file() {
  local repository="$1" revision="$2" filename="$3" output="$4"
  mkdir -p "$output"
  uv run --project "$PARITY_PROJECT" --frozen --python 3.12 python -c \
    'import sys; from huggingface_hub import hf_hub_download; print(hf_hub_download(repo_id=sys.argv[1], revision=sys.argv[2], filename=sys.argv[3], local_dir=sys.argv[4]))' \
    "$repository" "$revision" "$filename" "$output"
  [[ -f "$output/$filename" ]] || die "Hugging Face download did not produce $output/$filename"
}

require_vast_host() {
  [[ "${VOKRA_PUBLISH_ON_VAST:-0}" == "1" ]] || die "VOKRA_PUBLISH_ON_VAST=1 is absent; run provision.sh first"
  [[ "$(uname -s)" == "Linux" ]] || die "model work is Linux/VAST-only; refusing host $(uname -s)"
  [[ "$(uname -m)" == "x86_64" ]] || die "VAST Linux x86_64 host required; got $(uname -m)"
  mkdir -p "$VOKRA_SCRATCH"
}

require_tooling() {
  local tool
  for tool in uv cargo rustc git awk grep find tee wc tr cp cmp date nproc; do
    command -v "$tool" >/dev/null 2>&1 || die "required tool missing: $tool"
  done
  [[ -d "$VOKRA_ROOT/.git" ]] || die "$VOKRA_ROOT is not a git checkout"
  [[ -f "$PARITY_PROJECT/uv.lock" ]] || die "AST parity uv.lock is missing"
  [[ -f "$REFERENCE_SCRIPT" ]] || die "AST reference dumper is missing"
  [[ -f "$INPUT_WAV" ]] || die "pinned AST input WAV is missing"
  [[ -f "$FIXTURE_DIR/manifest.json" ]] || die "committed AST manifest is missing"
  [[ -z "$(git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all)" ]] || die "VAST checkout must be clean so evidence names an exact commit"
}

record_environment() {
  local output="$1" versions
  versions="$(uv run --project "$PARITY_PROJECT" --frozen --python 3.12 python -c \
    'import importlib.metadata, json; print(json.dumps({name: importlib.metadata.version(name) for name in ("torch", "torchaudio", "transformers")}, sort_keys=True))')"
  check_exact_versions "$versions" >/dev/null
  {
    echo "utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "git_commit=$(git -C "$VOKRA_ROOT" rev-parse HEAD)"
    echo "uname=$(uname -a)"
    echo "cpu_model=$(awk -F ':' '$1 ~ /model name/ {sub(/^[[:space:]]+/, "", $2); print $2; exit}' /proc/cpuinfo)"
    echo "cpu_flags=$(awk -F ':' '$1 ~ /flags/ {sub(/^[[:space:]]+/, "", $2); print $2; exit}' /proc/cpuinfo)"
    echo "nproc=$(nproc)"
    awk '$1 == "MemTotal:" {print "mem_total_kib=" $2; exit}' /proc/meminfo
    rustc --version --verbose
    cargo --version
    uv --version
    uv run --project "$PARITY_PROJECT" --frozen --python 3.12 python -c \
      'import platform; print(f"python={platform.python_version()}")'
    echo "versions=$versions"
  } | tee "$output"
}

check_exact_versions() {
  local actual_json="$1"
  uv run --no-project --python 3.12 python -c \
    'import json, sys; expected={"torch":"2.13.0","torchaudio":"2.11.0","transformers":"5.10.4"}; actual=json.loads(sys.argv[1]); sys.exit(f"dependency versions mismatch: {actual} != {expected}") if actual != expected else None; print(f"versions={actual}")' \
    "$actual_json"
}

run_self_test() {
  local tmp payload actual script_path cases=0 fail=0
  tmp="$(mktemp -d)"
  trap 'rm -rf "$tmp"' EXIT
  payload="$tmp/payload"
  printf 'vokra-ast-security-self-test\n' > "$payload"
  actual="$(sha256_file "$payload")"

  cases=$((cases + 1))
  verify_sha256 "$payload" "$actual" >/dev/null 2>&1 || { log "self-test FAIL: valid identity rejected"; fail=1; }
  cases=$((cases + 1))
  if verify_sha256 "$payload" "$(printf '%064d' 0)" >/dev/null 2>&1; then
    log "self-test FAIL: invalid SHA-256 accepted"
    fail=1
  fi

  cases=$((cases + 1))
  if ! check_exact_versions '{"torch":"2.13.0","torchaudio":"2.11.0","transformers":"5.10.4"}' >/dev/null 2>&1; then
    log "self-test FAIL: valid dependency versions rejected"
    fail=1
  fi
  cases=$((cases + 1))
  if check_exact_versions '{"torch":"2.13.0","torchaudio":"2.11.0","transformers":"5.10.3"}' >/dev/null 2>&1; then
    log "self-test FAIL: mismatched dependency versions accepted"
    fail=1
  fi

  script_path="${BASH_SOURCE[0]}"
  for required in "$PUBLIC_REVISION" "$PUBLIC_SHA256" "$UPSTREAM_REVISION" \
    "$UPSTREAM_SHA256" "$INPUT_SHA256" "$TRANSFORMERS_WHEEL_SHA256" \
    "$FEATURE_SOURCE_SHA256" "$MODELING_SOURCE_SHA256" \
    "torch==2.13.0" "torchaudio==2.11.0" "transformers==5.10.4" \
    "dump_reference.py" "real_ast_frontend_and_logits_match_official" \
    "--frozen --python 3.12" "VOKRA_PUBLISH_ON_VAST=1" \
    "numeric_verdict=PASS_PREEXISTING_BOUNDS_CANDIDATE"; do
    cases=$((cases + 1))
    if ! grep -Fq -- "$required" "$script_path"; then
      log "self-test FAIL: worker contract lost token: $required"
      fail=1
    fi
  done

  cases=$((cases + 1))
  if grep -En '^[[:space:]]*(python3|python|pip)([[:space:]]|$)' "$script_path" >/dev/null; then
    log "self-test FAIL: direct Python/pip command found"
    fail=1
  fi

  cases=$((cases + 1))
  UV_CACHE_DIR="${UV_CACHE_DIR:-$tmp/uv-cache}" \
    uv run --no-project --python 3.12 python -c \
      'import ast, pathlib, sys; ast.parse(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))' \
      "$REFERENCE_SCRIPT" >/dev/null 2>&1 \
    || { log "self-test FAIL: AST dumper syntax parse failed"; fail=1; }

  rm -rf "$tmp"
  trap - EXIT
  if [[ $fail -eq 0 ]]; then
    echo "run-ast-security-parity.sh self-test: OK ($cases cases)"
    return 0
  fi
  return 1
}

main() {
  local self_test=0 requested_work_dir="" run_stamp work_dir inputs_dir logs_dir
  local public_dir upstream_dir gguf checkpoint candidate original_dir
  local run_log env_log cpu_log summary_file
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --work-dir)
        [[ $# -ge 2 && -n "$2" ]] || { die "--work-dir requires a directory"; return 2; }
        requested_work_dir="$2"
        shift 2
        ;;
      --self-test) self_test=1; shift ;;
      -h|--help) usage; return 0 ;;
      *) die "unknown argument: $1"; usage; return 2 ;;
    esac
  done
  if [[ $self_test -eq 1 ]]; then
    run_self_test
    return $?
  fi

  require_vast_host
  require_tooling
  run_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  work_dir="${requested_work_dir:-$VOKRA_SCRATCH/ast-security-parity/$run_stamp}"
  if [[ -e "$work_dir" ]] && [[ -n "$(find "$work_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    die "--work-dir must be absent or empty: $work_dir"
  fi
  inputs_dir="$work_dir/inputs"
  logs_dir="$work_dir/logs"
  public_dir="$inputs_dir/public"
  upstream_dir="$inputs_dir/upstream"
  candidate="$work_dir/reference"
  original_dir="$work_dir/original-fixture"
  mkdir -p "$logs_dir" "$public_dir" "$upstream_dir" "$original_dir"
  export UV_CACHE_DIR="$VOKRA_SCRATCH/uv-cache-ast-security"
  run_log="$logs_dir/run.log"
  env_log="$logs_dir/environment.txt"
  cpu_log="$logs_dir/cpu.log"
  summary_file="$logs_dir/summary.txt"
  exec > >(tee -a "$run_log") 2>&1
  trap 'rc=$?; if [[ -n "${summary_file:-}" && ! -f "$summary_file" ]]; then printf "execution_status=FAIL\nexit_code=%s\n" "$rc" > "$summary_file"; fi; exit "$rc"' EXIT

  step "Sync exact locked Python 3.12 oracle"
  uv sync --project "$PARITY_PROJECT" --frozen --python 3.12

  step "Download and verify exact public and upstream inputs"
  download_hf_file "$PUBLIC_REPO" "$PUBLIC_REVISION" "$PUBLIC_FILE" "$public_dir"
  download_hf_file "$UPSTREAM_REPO" "$UPSTREAM_REVISION" "$UPSTREAM_FILE" "$upstream_dir"
  gguf="$public_dir/$PUBLIC_FILE"
  checkpoint="$upstream_dir/$UPSTREAM_FILE"
  verify_sha256 "$gguf" "$PUBLIC_SHA256"
  verify_sha256 "$checkpoint" "$UPSTREAM_SHA256"
  verify_sha256 "$INPUT_WAV" "$INPUT_SHA256"

  step "Record VAST identity and exact dependency versions before model execution"
  record_environment "$env_log"

  step "Generate candidate official Transformers reference"
  uv run --project "$PARITY_PROJECT" --frozen --python 3.12 python \
    "$REFERENCE_SCRIPT" --audio "$INPUT_WAV" --output "$candidate"
  cp "$candidate/manifest.json" "$logs_dir/reference-manifest.json"

  step "Temporarily install candidate fixture in disposable checkout"
  cp "$FIXTURE_DIR/input_values.f32le" "$original_dir/input_values.f32le"
  cp "$FIXTURE_DIR/logits.f32le" "$original_dir/logits.f32le"
  cp "$FIXTURE_DIR/manifest.json" "$original_dir/manifest.json"
  restore_fixture() {
    cp "$original_dir/input_values.f32le" "$FIXTURE_DIR/input_values.f32le"
    cp "$original_dir/logits.f32le" "$FIXTURE_DIR/logits.f32le"
    cp "$original_dir/manifest.json" "$FIXTURE_DIR/manifest.json"
  }
  trap 'rc=$?; if [[ -d "${original_dir:-}" && -f "${original_dir:-}/manifest.json" ]]; then restore_fixture; fi; if [[ -n "${summary_file:-}" && ! -f "$summary_file" ]]; then printf "execution_status=FAIL\nexit_code=%s\n" "$rc" > "$summary_file"; fi; exit "$rc"' EXIT
  cp "$candidate/input_values.f32le" "$FIXTURE_DIR/input_values.f32le"
  cp "$candidate/logits.f32le" "$FIXTURE_DIR/logits.f32le"
  cp "$candidate/manifest.json" "$FIXTURE_DIR/manifest.json"

  step "Run candidate official CPU parity on VAST"
  VOKRA_AST_GGUF="$gguf" VOKRA_AST_BACKEND=cpu \
    cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release \
      -p vokra-models --test parity_ast_real \
      real_ast_frontend_and_logits_match_official -- --exact --nocapture 2>&1 | tee "$cpu_log"
  grep -F "[parity_ast_real] Cpu:" "$cpu_log" >/dev/null || die "AST CPU parity evidence sentinel missing"

  restore_fixture
  [[ -z "$(git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all)" ]] || die "fixture restoration left VAST checkout dirty"
  {
    echo "execution_status=PASS"
    echo "numeric_verdict=PASS_PREEXISTING_BOUNDS_CANDIDATE"
    echo "cpu_parity_status=PASS"
    echo "git_commit=$(git -C "$VOKRA_ROOT" rev-parse HEAD)"
    echo "public_revision=$PUBLIC_REVISION"
    echo "public_sha256=$PUBLIC_SHA256"
    echo "upstream_revision=$UPSTREAM_REVISION"
    echo "upstream_sha256=$UPSTREAM_SHA256"
    echo "input_sha256=$INPUT_SHA256"
    echo "torch=$TORCH_VERSION"
    echo "torchaudio=$TORCHAUDIO_VERSION"
    echo "transformers=$TRANSFORMERS_VERSION"
    echo "transformers_wheel_sha256=$TRANSFORMERS_WHEEL_SHA256"
    echo "feature_source_sha256=$FEATURE_SOURCE_SHA256"
    echo "modeling_source_sha256=$MODELING_SOURCE_SHA256"
    echo "reference_manifest_sha256=$(sha256_file "$logs_dir/reference-manifest.json")"
    grep -F "[parity_ast_real] Cpu:" "$cpu_log"
  } | tee "$summary_file"
  trap - EXIT
  log "PASS: pull $logs_dir, then destroy the VAST instance"
}

main "$@"
