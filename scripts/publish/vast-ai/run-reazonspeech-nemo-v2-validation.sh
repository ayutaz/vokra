#!/usr/bin/env bash
# VAST-only validation worker for ReazonSpeech NeMo v2.
# It never uploads, publishes, pushes Git refs, stops, or destroys an instance.
# Pull the small evidence/reference files, then destroy the instance externally.

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run-reazonspeech-nemo-v2-validation.sh --nemo <reazonspeech-nemo-v2.nemo> \
    --approval-evidence <owner-approval.json> [--work-dir /workspace/vokra-reazonspeech-validation]
  run-reazonspeech-nemo-v2-validation.sh --self-test

Requires Linux and VOKRA_PUBLISH_ON_VAST=1 from provision.sh, plus the
rustfmt/clippy components and cargo-deny/cargo-audit executables. Produces a
complete local GGUF, an official NeMo encoder/token reference, exact CPU parity
evidence, and Rust verification logs. It performs no Hugging Face upload.
EOF
}

die() {
  echo "run-reazonspeech-nemo-v2-validation: $*" >&2
  return 2
}

UPSTREAM_REPO="reazon-research/reazonspeech-nemo-v2"
UPSTREAM_REVISION="33693408be76b7cba9fd4a7546a0a8772430211b"
MODEL_KIND="reazonspeech-nemo-v2"
PARITY_TEST="released_cpu_encoder_and_alsd_tokens_text_match_official_nemo"
GGUF_ENV="VOKRA_REAZONSPEECH_NEMO_V2_GGUF"
REFERENCE_DIR_ENV="VOKRA_REAZONSPEECH_NEMO_V2_REFERENCE_DIR"
PROJECT_FILE="tools/parity/pyproject.toml"
LOCK_FILE="tools/parity/uv.lock"
SIGNOFF_REPO="reazonspeech-nemo-v2"

license_preflight() {
  local approval="$1" project_sha lock_sha
  [[ -f "$PROJECT_FILE" && ! -L "$PROJECT_FILE" && -f "$LOCK_FILE" && ! -L "$LOCK_FILE" ]] \
    || die "locked parity project is missing or symlinked"
  [[ -f "$approval" && ! -L "$approval" && -s "$approval" ]] \
    || die "--approval-evidence must be a nonempty regular non-symlink file"
  project_sha="$(sha256sum "$PROJECT_FILE" | awk '{print $1}')"
  lock_sha="$(sha256sum "$LOCK_FILE" | awk '{print $1}')"
  # Dependency-free, duplicate-key rejecting approval contract.  The digest
  # binds the exact committed lock/project bytes, fixed model identity, the
  # separate license decision, and the no-upload policy.
  if UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python - "$approval" "$project_sha" "$lock_sha" <<'PY'
import hashlib, json, pathlib, sys

path, project_sha, lock_sha = sys.argv[1:]
def reject(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError("duplicate JSON key: " + key)
        out[key] = value
    return out
try:
    data = json.loads(pathlib.Path(path).read_text(encoding="utf-8"), object_pairs_hook=reject)
    expected = {"schema", "model", "upstream_repo", "upstream_revision", "license_spdx", "project_sha256", "lock_sha256", "no_upload", "decision", "signer", "scope_sha256"}
    if set(data) != expected:
        raise ValueError("approval schema is not exact")
    if data["schema"] != "vokra-validation-approval-v1" or data["model"] != "reazonspeech-nemo-v2": raise ValueError("approval identity mismatch")
    if data["upstream_repo"] != "reazon-research/reazonspeech-nemo-v2" or data["upstream_revision"] != "33693408be76b7cba9fd4a7546a0a8772430211b": raise ValueError("upstream identity mismatch")
    if data["license_spdx"] != "apache-2.0" or data["project_sha256"] != project_sha or data["lock_sha256"] != lock_sha or data["no_upload"] is not True or data["decision"] != "APPROVED": raise ValueError("approval facts mismatch")
    if not isinstance(data["signer"], str) or not data["signer"].strip() or data["signer"].strip().upper() in {"TODO", "UNRESOLVED", "OWNER_SIGNOFF_REQUIRED"}: raise ValueError("approval signer unresolved")
    scope = {"license_spdx": data["license_spdx"], "lock_sha256": lock_sha, "model": data["model"], "no_upload": True, "project_sha256": project_sha, "upstream_repo": data["upstream_repo"], "upstream_revision": data["upstream_revision"]}
    digest = hashlib.sha256(json.dumps(scope, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if data["scope_sha256"] != digest: raise ValueError("approval scope digest mismatch")
except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
    raise SystemExit(f"approval gate BLOCKED: {exc}")
PY
  then :; else die 'approval evidence is invalid or offline Python is unavailable'; fi
  if UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python scripts/publish/signoff_match.py --check-repo "$SIGNOFF_REPO" \
    --audit docs/license-audit.md
  then :; else die 'repository signoff is unresolved'; fi
}

canonical_absent_path() {
  local path="$1" suffix='' rest component scan name parent
  [[ "$path" == /* ]] || path="$PWD/$path"
  rest="${path#/}"; scan=''
  while [[ -n "$rest" ]]; do
    component="${rest%%/*}"; rest="${rest#*/}"
    [[ "$component" == "$rest" ]] && rest=''
    [[ -n "$component" && "$component" != . && "$component" != .. ]] || continue
    scan="$scan/$component"
    [[ ! -L "$scan" || "$scan" == "/var" ]] || return 1
  done
  while [[ ! -d "$path" || -L "$path" ]]; do
    name="${path##*/}"; [[ -n "$name" ]] && suffix="/$name$suffix"
    parent="${path%/*}"; [[ "$parent" == "$path" ]] && parent=/
    path="$parent"
  done
  (cd -P "$path" && printf '%s%s\n' "$PWD" "$suffix")
}

paths_overlap() { [[ "$1" == "$2" || "$1" == "$2"/* || "$2" == "$1"/* ]]; }

require_absent_work_dir() {
  local target="$1" approval="$2" candidate protected other
  [[ ! -e "$target" && ! -L "$target" ]] || { die "--work-dir must be absent and non-symlink"; return 2; }
  candidate="$(canonical_absent_path "$target")" || { die "--work-dir has a symlinked ancestor"; return 2; }
  for protected in "$PWD" "$PWD/$PROJECT_FILE" "$PWD/$LOCK_FILE" "$approval"; do
    [[ -e "$protected" || -L "$protected" ]] || continue
    [[ ! -L "$protected" ]] || { die "protected input is symlinked"; return 2; }
    other="$(canonical_absent_path "$protected")" || { die "protected path cannot be canonicalized"; return 2; }
    paths_overlap "$candidate" "$other" && { die "--work-dir overlaps protected input"; return 2; }
  done
  return 0
}

run_self_test() {
  local script_path="${BASH_SOURCE[0]}" tmp fail=0 cases=0 required
  tmp="$(mktemp -d)"
  # shellcheck disable=SC2064
  trap "rm -rf '$tmp'" EXIT

  cases=$((cases + 1))
  for required in \
    "$UPSTREAM_REPO" "$UPSTREAM_REVISION" "$MODEL_KIND" "$PARITY_TEST" \
    "$GGUF_ENV" "$REFERENCE_DIR_ENV" \
    "tools/parity/reazonspeech_nemo_v2_prepare_checkpoint.py" \
    "tools/parity/reazonspeech_nemo_v2_dump_reference.py" \
    "--frozen --project tools/parity --python 3.12 python" \
    "run_logged env \"\$GGUF_ENV=" "\"\$REFERENCE_DIR_ENV=" \
    "cargo test --locked --release -p vokra-models" \
    "env -u \"\$GGUF_ENV\" -u \"\$REFERENCE_DIR_ENV\"" \
    "--test parity_reazonspeech_nemo_v2"; do
    if ! grep -Fq -- "$required" "$script_path"; then
      echo "run-reazonspeech-nemo-v2-validation: self-test FAIL: contract lost token: $required" >&2
      fail=1
    fi
  done

  cases=$((cases + 1))
  if grep -En '^[[:space:]]*(python3|python|pip)([[:space:]]|$)' \
    "$script_path" >/dev/null; then
    echo "run-reazonspeech-nemo-v2-validation: self-test FAIL: direct Python/pip command found" >&2
    fail=1
  fi
  if grep -En -- '^[[:space:]]*(git[[:space:]]+push|.*upload\.sh|.*publish-one\.sh)([[:space:]]|$)' \
    "$script_path" >/dev/null; then
    echo "run-reazonspeech-nemo-v2-validation: self-test FAIL: publication command found" >&2
    fail=1
  fi
  if grep -Fq "export \"\$GGUF_ENV=" "$script_path" || grep -Fq "export \"\$REFERENCE_DIR_ENV=" "$script_path"; then
    echo "run-reazonspeech-nemo-v2-validation: self-test FAIL: model environment globally exported" >&2
    fail=1
  fi

  cases=$((cases + 1))
  for required in 'uname -s' 'VOKRA_PUBLISH_ON_VAST' 'git status --porcelain --untracked-files=all' \
    'cargo fmt --all -- --check' 'cargo test --locked --workspace' \
    'cargo clippy --locked --workspace --all-targets -- -D warnings'; do
    if ! grep -Fq -- "$required" "$script_path"; then
      echo "run-reazonspeech-nemo-v2-validation: self-test FAIL: fail-closed guard lost token: $required" >&2
      fail=1
    fi
  done

  cases=$((cases + 1))
  if "$script_path" --self-test --work-dir "$tmp/nonempty" >/dev/null 2>&1; then
    echo "run-reazonspeech-nemo-v2-validation: self-test FAIL: extra self-test argument accepted" >&2
    fail=1
  fi
  if "$script_path" --nemo >/dev/null 2>&1; then
    echo "run-reazonspeech-nemo-v2-validation: self-test FAIL: missing --nemo value accepted" >&2
    fail=1
  fi
  if "$script_path" --nemo -bad >/dev/null 2>&1 || "$script_path" --nemo a --nemo b >/dev/null 2>&1 || "$script_path" --approval-evidence >/dev/null 2>&1 || "$script_path" --self-test --self-test >/dev/null 2>&1 || "$script_path" --self-test --approval-evidence x >/dev/null 2>&1; then
    echo "run-reazonspeech-nemo-v2-validation: self-test FAIL: malformed or duplicate options accepted" >&2
    fail=1
  fi
  printf '{}\n' > "$tmp/approval.json"
  require_absent_work_dir "$tmp/new/nested/work" "$tmp/approval.json" || { echo 'self-test FAIL: nested absent work path rejected' >&2; fail=1; }
  mkdir "$tmp/empty-work"
  if require_absent_work_dir "$tmp/empty-work" "$tmp/approval.json" >/dev/null 2>&1; then echo 'self-test FAIL: existing empty work accepted' >&2; fail=1; fi
  ln -s "$tmp/missing" "$tmp/dangling-work"
  if require_absent_work_dir "$tmp/dangling-work" "$tmp/approval.json" >/dev/null 2>&1; then echo 'self-test FAIL: dangling work symlink accepted' >&2; fail=1; fi
  if "$script_path" --unknown-self-test-flag >/dev/null 2>&1; then
    echo "run-reazonspeech-nemo-v2-validation: self-test FAIL: unknown argument accepted" >&2
    fail=1
  fi

  rm -rf "$tmp"
  trap - EXIT
  if [[ $fail -eq 0 ]]; then
    echo "run-reazonspeech-nemo-v2-validation.sh self-test: OK ($cases cases)"
    return 0
  fi
  return 1
}

nemo_path=""
work_dir="/workspace/vokra-reazonspeech-validation"
approval_evidence=""
self_test=0
seen_self_test=0
seen_nemo=0
seen_work=0
seen_approval=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --self-test)
      seen_self_test=$((seen_self_test + 1))
      self_test=1
      shift
      ;;
    --nemo)
      (( seen_nemo == 0 )) || die "duplicate --nemo"
      [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die "--nemo requires a nonempty path"
      seen_nemo=1
      nemo_path="$2"
      shift 2
      ;;
    --work-dir)
      (( seen_work == 0 )) || die "duplicate --work-dir"
      [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die "--work-dir requires a nonempty path"
      seen_work=1
      work_dir="$2"
      shift 2
      ;;
    --approval-evidence)
      (( seen_approval == 0 )) || die "duplicate --approval-evidence"
      [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die "--approval-evidence requires a nonempty path"
      seen_approval=1
      approval_evidence="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

if [[ $self_test -eq 1 ]]; then
  [[ $seen_self_test -eq 1 && -z "$nemo_path$approval_evidence" && "$work_dir" == "/workspace/vokra-reazonspeech-validation" ]] \
    || die "--self-test accepts no other arguments"
  run_self_test
  exit $?
fi

[[ $seen_approval -eq 1 ]] || die "--approval-evidence is required"
license_preflight "$approval_evidence"
require_absent_work_dir "$work_dir" "$approval_evidence"

[[ "$(uname -s)" == "Linux" ]] || die "actual validation is Linux/VAST-only"
[[ "${VOKRA_PUBLISH_ON_VAST:-0}" == "1" ]] \
  || die "VOKRA_PUBLISH_ON_VAST=1 is absent; run provision.sh first"
[[ -n "$nemo_path" ]] || die "--nemo is required"
[[ -f "$nemo_path" && ! -L "$nemo_path" ]] || die "checkpoint is not a regular non-symlink file: $nemo_path"
[[ -f Cargo.toml && -d crates/vokra-models ]] \
  || die "run from the Vokra repository root"

for command in rustfmt cargo-deny cargo-audit; do
  command -v "$command" >/dev/null 2>&1 \
    || die "required VAST verification tool is missing: $command"
done
cargo clippy --version >/dev/null 2>&1 \
  || die "the clippy component is missing; install rustfmt/clippy on the VAST host"
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] \
  || die "worktree changes or untracked files are present; validate a clean committed git-bundle checkpoint"

mkdir -p "$work_dir"
work_dir="$(cd "$work_dir" && pwd)"
nemo_path="$(cd "$(dirname "$nemo_path")" && pwd)/$(basename "$nemo_path")"
log_path="$work_dir/validation.log"
evidence_dir="$work_dir/evidence"
prepared_dir="$work_dir/prepared"
reference_dir="$evidence_dir/reference"
mkdir -p "$evidence_dir" "$prepared_dir" "$reference_dir"

run_logged() {
  echo "+ $*" | tee -a "$log_path"
  "$@" 2>&1 | tee -a "$log_path"
}

require_cargo_result() {
  local file="$1" test_name="$2" named tests results
  named="$(grep -Ec "^test $test_name \.\.\. ok$" "$file" || true)"
  tests="$(grep -Ec '^test [^ ]+ \.\.\.' "$file" || true)"
  results="$(grep -Ec '^test result:' "$file" || true)"
  [[ "$named" == 1 && "$tests" == 1 && "$results" == 1 ]] || die 'Cargo evidence has duplicate/missing test or result lines'
  grep -Eq '^test result: ok\. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out(; finished in [0-9]+\.[0-9]+s)?$' "$file" || die 'Cargo result is not the exact one-pass result'
}

export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-4}"
export RUST_BACKTRACE=1

run_logged cargo fmt --all -- --check
run_logged bash scripts/check-forbidden-symbols.sh
run_logged bash scripts/check-zero-deps.sh
run_logged bash scripts/check-bound-arch-coverage.sh

run_logged uv run --frozen --project tools/parity --python 3.12 python \
  tools/parity/reazonspeech_nemo_v2_prepare_checkpoint.py \
  --input "$nemo_path" --output-dir "$prepared_dir"

run_logged cargo build --locked --release -p vokra-convert -p vokra-cli
run_logged target/release/vokra-cli convert \
  --model "$MODEL_KIND" \
  --input "$prepared_dir/reazonspeech-nemo-v2.prepared.safetensors" \
  --tokenizer "$prepared_dir/tokenizer.vocab" \
  --output "$work_dir/reazonspeech-nemo-v2.gguf"

run_logged uv run --frozen --project tools/parity --extra titanet --python 3.12 python \
  tools/parity/reazonspeech_nemo_v2_dump_reference.py \
  --nemo "$nemo_path" --output-dir "$reference_dir"

run_logged env "$GGUF_ENV=$work_dir/reazonspeech-nemo-v2.gguf" \
  "$REFERENCE_DIR_ENV=$reference_dir" \
  cargo test --locked --release -p vokra-models \
  --test parity_reazonspeech_nemo_v2 \
  "$PARITY_TEST" -- --nocapture
require_cargo_result "$log_path" "$PARITY_TEST"

run_logged target/release/vokra-cli run \
  --model "$work_dir/reazonspeech-nemo-v2.gguf" \
  --input tests/fixtures/audio/jfk-30s.wav --backend cpu

run_logged env -u "$GGUF_ENV" -u "$REFERENCE_DIR_ENV" \
  cargo test --locked --workspace
run_logged cargo clippy --locked --workspace --all-targets -- -D warnings
run_logged cargo deny check licenses advisories bans
run_logged cargo audit

{
  echo "commit=$(git rev-parse HEAD)"
  echo "branch=$(git branch --show-current)"
  echo "rustc=$(rustc --version)"
  echo "cargo=$(cargo --version)"
  echo "kernel=$(uname -srmo)"
  echo "cpu=$(awk -F ': ' '/^model name/{print $2; exit}' /proc/cpuinfo)"
  echo "nemo_sha256=$(sha256sum "$nemo_path" | awk '{print $1}')"
  echo "gguf_sha256=$(sha256sum "$work_dir/reazonspeech-nemo-v2.gguf" | awk '{print $1}')"
  echo "reference_sha256=$(sha256sum "$reference_dir/reference.json" | awk '{print $1}')"
  echo "reference_encoder_sha256=$(sha256sum "$reference_dir/encoder.f32" | awk '{print $1}')"
  echo "reference_tokens_sha256=$(sha256sum "$reference_dir/tokens.u32" | awk '{print $1}')"
  echo "verdict=PASS"
} > "$evidence_dir/validation-summary.txt"

cp "$prepared_dir/prepare-audit.json" "$evidence_dir/prepare-audit.json"
echo "run-reazonspeech-nemo-v2-validation: PASS"
echo "Pull before destroy: $evidence_dir and $log_path"
echo "Do not pull the multi-GB .nemo/.safetensors/.gguf artifacts to the maintainer Mac."
