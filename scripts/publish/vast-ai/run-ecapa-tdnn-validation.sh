#!/usr/bin/env bash
# VAST-only validation worker for the pinned SpeechBrain ECAPA-TDNN model.
# The worker stages the corrupt public artifact only to authenticate the
# replacement target; parity and CLI checks use a fresh strict conversion.
# There is deliberately no publish, upload, or Git-push operation here.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VOKRA_ROOT="${VOKRA_ROOT:-$DEFAULT_ROOT}"
VOKRA_SCRATCH="${VOKRA_SCRATCH:-$HOME/scratchpad}"
PARITY_PROJECT="$VOKRA_ROOT/tools/parity"
PROJECT_FILE="$PARITY_PROJECT/pyproject.toml"
LOCK_FILE="$PARITY_PROJECT/uv.lock"
PREPARER="$PARITY_PROJECT/ecapa_tdnn_prepare_checkpoint.py"
PARITY_DUMPER="$PARITY_PROJECT/ecapa_tdnn_dump_reference.py"
JFK_WAV="$VOKRA_ROOT/tests/fixtures/audio/jfk-30s.wav"

# Every identity value below is recorded in repository handoff/parity
# evidence. Missing or mismatching bytes are fatal; no observed replacement
# digest may be substituted here.
UPSTREAM_REPO="speechbrain/spkrec-ecapa-voxceleb"
UPSTREAM_REVISION="0f99f2d0ebe89ac095bcc5903c4dd8f72b367286"
UPSTREAM_CHECKPOINT="embedding_model.ckpt"
UPSTREAM_CHECKPOINT_SHA256="0575cb64845e6b9a10db9bcb74d5ac32b326b8dc90352671d345e2ee3d0126a2"
MODEL_KIND="ecapa-tdnn"
LICENSE_SPDX="apache-2.0"

# The public artifact being replaced is recorded as malformed. Authenticate it
# so the run cannot silently validate a different replacement target.
CORRUPT_REPO="vokra/speechbrain-spkrec-ecapa-voxceleb"
CORRUPT_REVISION="3dc7704b2dcb80b8ea8eb2d3db7280f682ac3657"
CORRUPT_FILE="spkrec-ecapa-voxceleb.restamped.gguf"
CORRUPT_BYTES=83239904
CORRUPT_SHA256="75e74d4e41d16bf2af5a0176c189fc1c7f7597fe66aae47cacef17343cbb4c01"

PARITY_TEST="public_artifact_matches_speechbrain"
GGUF_ENV="VOKRA_ECAPA_GGUF"
JFK_BYTES=352078
JFK_SHA256="58adb4ea501d955fcd40bfbb69128f8f40428b81d8716b9ed337949773be253f"

# The Rust parity test consumes these committed oracle files. The source WAV
# used to create them is not checked in, so the worker reconstructs its exact
# PCM16 representation from pcm.f32.bin and authenticates the recorded WAV
# digest before asking the independent SpeechBrain dumper to run.
FIXTURE_DIR="$VOKRA_ROOT/crates/vokra-models/tests/fixtures/ecapa_tdnn"
FIXTURE_PCM="$FIXTURE_DIR/pcm.f32.bin"
FIXTURE_FEATURES="$FIXTURE_DIR/features.f32.bin"
FIXTURE_EMBEDDING="$FIXTURE_DIR/embedding.f32.bin"
FIXTURE_PCM_SHA256="48aedc3a10b14b49ebe8da2efd1dd91cbe7dbbaf58278732e7fdb04f6d6cc1e9"
FIXTURE_FEATURES_SHA256="6ea88148da19e1179e9c8bc27fa9b76c742d5b82376e9c4f153bf1da3cd6a191"
FIXTURE_EMBEDDING_SHA256="f6b297f3c9e8746d0a2ceaded702b1ce5e741fd3957ce1879b07608f7bd082e4"
FIXTURE_WAV_BYTES=104390
FIXTURE_WAV_SHA256="bf2dde5cb516939ff619d62fc07d4f4bec5b5d521aee3d07ae51828c9d93be0b"

MIN_VAST_MEM_KIB=67108864
MIN_FREE_DISK_KIB=150000000

license_preflight() {
  local approval="$1" project_sha lock_sha
  [[ -f "$PROJECT_FILE" && ! -L "$PROJECT_FILE" && -f "$LOCK_FILE" && ! -L "$LOCK_FILE" ]] || die 'locked parity project is missing or symlinked'
  [[ -f "$approval" && ! -L "$approval" && -s "$approval" ]] || die '--approval-evidence must be a nonempty regular non-symlink file'
  project_sha="$(sha256_file "$PROJECT_FILE")"; lock_sha="$(sha256_file "$LOCK_FILE")"
  if UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python - "$approval" "$project_sha" "$lock_sha" <<'PY'
import hashlib, json, pathlib, sys
def hook(pairs):
    out = {}
    for key, value in pairs:
        if key in out: raise ValueError('duplicate JSON key: ' + key)
        out[key] = value
    return out
try:
    d = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding='utf-8'), object_pairs_hook=hook)
    keys = {'schema','model','upstream_repo','upstream_revision','license_spdx','project_sha256','lock_sha256','no_upload','decision','signer','scope_sha256'}
    if set(d) != keys: raise ValueError('approval schema is not exact')
    if (d['schema'],d['model'],d['upstream_repo'],d['upstream_revision'],d['license_spdx']) != ('vokra-validation-approval-v1','ecapa-tdnn','speechbrain/spkrec-ecapa-voxceleb','0f99f2d0ebe89ac095bcc5903c4dd8f72b367286','apache-2.0'): raise ValueError('approval identity mismatch')
    if d['project_sha256'] != sys.argv[2] or d['lock_sha256'] != sys.argv[3] or d['no_upload'] is not True or d['decision'] != 'APPROVED': raise ValueError('approval facts mismatch')
    if not isinstance(d['signer'],str) or not d['signer'].strip() or d['signer'].strip().upper() in {'TODO','UNRESOLVED','OWNER_SIGNOFF_REQUIRED'}: raise ValueError('approval signer unresolved')
    scope={'license_spdx':d['license_spdx'],'lock_sha256':sys.argv[3],'model':d['model'],'no_upload':True,'project_sha256':sys.argv[2],'upstream_repo':d['upstream_repo'],'upstream_revision':d['upstream_revision']}
    if d['scope_sha256'] != hashlib.sha256(json.dumps(scope,sort_keys=True,separators=(',',':')).encode()).hexdigest(): raise ValueError('approval scope digest mismatch')
except (OSError,TypeError,ValueError,json.JSONDecodeError) as exc: raise SystemExit('approval gate BLOCKED: '+str(exc))
PY
  then :; else die 'approval evidence is invalid or offline Python is unavailable'; fi
  if UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python scripts/publish/signoff_match.py --check-repo speechbrain-spkrec-ecapa-voxceleb --audit docs/license-audit.md
  then :; else die 'repository signoff is unresolved'; fi
}

canonical_absent_path() {
  local path="$1" suffix='' rest component scan name parent
  [[ "$path" == /* ]] || path="$PWD/$path"; rest="${path#/}"; scan=''
  while [[ -n "$rest" ]]; do component="${rest%%/*}"; rest="${rest#*/}"; [[ "$component" == "$rest" ]] && rest=''; [[ -n "$component" && "$component" != . && "$component" != .. ]] || continue; scan="$scan/$component"; [[ ! -L "$scan" || "$scan" == "/var" ]] || return 1; done
  while [[ ! -d "$path" || -L "$path" ]]; do name="${path##*/}"; [[ -n "$name" ]] && suffix="/$name$suffix"; parent="${path%/*}"; [[ "$parent" == "$path" ]] && parent=/; path="$parent"; done
  (cd -P "$path" && printf '%s%s\n' "$PWD" "$suffix")
}
paths_overlap() { [[ "$1" == "$2" || "$1" == "$2"/* || "$2" == "$1"/* ]]; }
require_absent_work_dir() {
  local target="$1" approval="$2" candidate protected other
  [[ ! -e "$target" && ! -L "$target" ]] || { die '--work-dir must be absent and non-symlink'; return 2; }
  candidate="$(canonical_absent_path "$target")" || { die '--work-dir has a symlinked ancestor'; return 2; }
  for protected in "$VOKRA_ROOT" "$PARITY_PROJECT" "$PROJECT_FILE" "$LOCK_FILE" "$approval" "$JFK_WAV"; do
    [[ -e "$protected" || -L "$protected" ]] || continue; [[ ! -L "$protected" ]] || { die 'protected input is symlinked'; return 2; }; other="$(canonical_absent_path "$protected")" || { die 'protected path cannot be canonicalized'; return 2; }; paths_overlap "$candidate" "$other" && { die '--work-dir overlaps protected input'; return 2; }
  done
  return 0
}

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

log() { printf '[ecapa-tdnn-vast] %s\n' "$*" >&2; }
step() { printf '\n[ecapa-tdnn-vast] ==== %s ====\n' "$*" >&2; }
die() { log "ERROR: $*"; return 2; }

usage() {
  cat <<'EOF' >&2
usage: run-ecapa-tdnn-validation.sh --approval-evidence <owner-approval.json> [--work-dir <absent-dir>]
       run-ecapa-tdnn-validation.sh --self-test

VAST-only, no-publish ECAPA-TDNN validation worker. It downloads the exact
SpeechBrain checkpoint and the recorded malformed public artifact, verifies
both identities, uses the safe weights_only checkpoint bridge and strict
ecapa-tdnn converter, generates an independent SpeechBrain 1.0.3 oracle,
runs the real CPU parity test and CLI speaker embedding e2e, then runs the
workspace gates. The malformed artifact is never used as parity input.

Normal runs require Linux x86_64, VOKRA_PUBLISH_ON_VAST=1, at least 64 GiB
RAM, and 150 GB free disk. --self-test is pure offline and performs no
network, model download, Python, Cargo, credentials, or publication action.
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

verify_file() {
  local path="$1" expected_hash="$2" expected_bytes="${3:-}"
  local actual_hash actual_bytes=""
  [[ -f "$path" && ! -L "$path" ]] || { die "missing, symlinked, or non-regular pinned input: $path"; return 2; }
  if [[ -n "$expected_bytes" ]]; then
    actual_bytes="$(wc -c < "$path" | tr -d '[:space:]')"
    [[ "$actual_bytes" == "$expected_bytes" ]] || {
      die "byte-size mismatch for $path: got $actual_bytes, expected $expected_bytes"
      return 2
    }
  fi
  actual_hash="$(sha256_file "$path")"
  [[ "$actual_hash" == "$expected_hash" ]] || {
    die "SHA-256 mismatch for $path: got $actual_hash, expected $expected_hash"
    return 2
  }
  log "identity OK: $path sha256=$actual_hash${actual_bytes:+ bytes=$actual_bytes}"
}

download_hf_file() {
  local repository="$1" revision="$2" filename="$3" output_dir="$4"
  mkdir -p "$output_dir"
  uv run --project "$PARITY_PROJECT" --frozen --python 3.12 python -c \
    'import sys; from huggingface_hub import hf_hub_download; print(hf_hub_download(repo_id=sys.argv[1], revision=sys.argv[2], filename=sys.argv[3], local_dir=sys.argv[4], local_dir_use_symlinks=False))' \
    "$repository" "$revision" "$filename" "$output_dir"
  [[ -f "$output_dir/$filename" ]] || die "download did not produce $output_dir/$filename"
}

require_vast_host() {
  local mem_kib free_kib
  [[ "${VOKRA_PUBLISH_ON_VAST:-0}" == "1" ]] \
    || die "VOKRA_PUBLISH_ON_VAST=1 is absent; run provision.sh first"
  [[ "$(uname -s)" == "Linux" ]] \
    || die "ECAPA checkpoint work is VAST/Linux-only; refusing $(uname -s)"
  [[ "$(uname -m)" == "x86_64" ]] \
    || die "locked SpeechBrain environment targets Linux x86_64, got $(uname -m)"
  [[ -r /proc/meminfo ]] || die "/proc/meminfo is unavailable"
  mem_kib="$(awk '$1 == "MemTotal:" {print $2; exit}' /proc/meminfo)"
  [[ "$mem_kib" =~ ^[0-9]+$ ]] || die "could not read MemTotal"
  (( mem_kib >= MIN_VAST_MEM_KIB )) \
    || die "MemTotal=${mem_kib} KiB is below the 64-GiB guard (67108864 KiB)"
  mkdir -p "$VOKRA_SCRATCH"
  free_kib="$(df -Pk "$VOKRA_SCRATCH" | awk 'NR == 2 {print $4}')"
  [[ "$free_kib" =~ ^[0-9]+$ ]] || die "could not read free disk"
  (( free_kib >= MIN_FREE_DISK_KIB )) \
    || die "free disk=${free_kib} KiB is below the 150-GB guard (150000000 KiB)"
}

require_tooling() {
  local tool path
  for tool in uv cargo rustc git awk grep find tee wc tr df nproc rustfmt cargo-deny cargo-audit; do
    command -v "$tool" >/dev/null 2>&1 || die "required VAST tool missing: $tool"
  done
  cargo clippy --version >/dev/null 2>&1 \
    || die "the clippy component is missing on the VAST host"
  [[ -d "$VOKRA_ROOT/.git" && -f "$VOKRA_ROOT/Cargo.toml" ]] \
    || die "VOKRA_ROOT is not the repository checkout: $VOKRA_ROOT"
  [[ -f "$PARITY_PROJECT/pyproject.toml" && -f "$PARITY_PROJECT/uv.lock" ]] \
    || die "tools/parity locked Python project is missing"
  for path in "$PREPARER" "$PARITY_DUMPER" "$JFK_WAV" \
    "$FIXTURE_PCM" "$FIXTURE_FEATURES" "$FIXTURE_EMBEDDING"; do
    [[ -f "$path" ]] || die "required ECAPA validation input is missing: $path"
  done
  grep -Fq 'torch.load(args.input, map_location="cpu", weights_only=True)' "$PREPARER" \
    || die "ECAPA preparer is not pinned to torch.load(weights_only=True)"
  [[ -z "$(git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all)" ]] \
    || die "VAST checkout must be clean so evidence names an exact commit"
}

materialize_fixture_wav() {
  local output="$1"
  mkdir -p "$(dirname "$output")"
  uv run --project "$PARITY_PROJECT" --frozen --python 3.12 python - \
    "$FIXTURE_PCM" "$output" <<'PY'
import sys
import wave
from pathlib import Path

import numpy as np

source = Path(sys.argv[1])
output = Path(sys.argv[2])
samples = np.fromfile(source, dtype="<f4")
if samples.size != 52_173 or not np.isfinite(samples).all():
    raise SystemExit("unexpected committed ECAPA PCM fixture")
scaled = np.rint(samples.astype(np.float64) * 32_768.0)
if np.any(scaled < -32_768) or np.any(scaled > 32_767):
    raise SystemExit("committed ECAPA PCM is outside signed PCM16")
pcm = scaled.astype("<i2")
if not np.array_equal(samples, pcm.astype(np.float32) / 32_768.0):
    raise SystemExit("committed ECAPA PCM is not an exact PCM16 representation")
with wave.open(str(output), "wb") as stream:
    stream.setnchannels(1)
    stream.setsampwidth(2)
    stream.setframerate(16_000)
    stream.writeframes(pcm.tobytes())
PY
}

verify_committed_oracle() {
  local generated_dir="$1" generated fixture expected path
  for path in "$FIXTURE_PCM" "$FIXTURE_FEATURES" "$FIXTURE_EMBEDDING"; do
    [[ -f "$path" ]] || { die "missing committed ECAPA oracle file: $path"; return 2; }
  done
  verify_file "$FIXTURE_PCM" "$FIXTURE_PCM_SHA256" 208692
  verify_file "$FIXTURE_FEATURES" "$FIXTURE_FEATURES_SHA256" 104640
  verify_file "$FIXTURE_EMBEDDING" "$FIXTURE_EMBEDDING_SHA256" 768
  for path in pcm.f32.bin features.f32.bin embedding.f32.bin; do
    generated="$generated_dir/$path"
    fixture="$FIXTURE_DIR/$path"
    [[ -f "$generated" ]] || { die "oracle did not emit $generated"; return 2; }
    cmp -s "$generated" "$fixture" || {
      die "fresh oracle differs byte-for-byte from committed fixture: $path"
      return 2
    }
    expected="$(sha256_file "$fixture")"
    [[ "$(sha256_file "$generated")" == "$expected" ]] || {
      die "fresh oracle SHA-256 differs from committed fixture: $path"
      return 2
    }
    log "oracle fixture exact match: $path sha256=$expected"
  done
}

require_cargo_result() {
  local file="$1" test_name="$2" named tests results
  named="$(grep -Ec "^test $test_name \.\.\. ok$" "$file" || true)"
  tests="$(grep -Ec '^test [^ ]+ \.\.\.' "$file" || true)"
  results="$(grep -Ec '^test result:' "$file" || true)"
  [[ "$named" == 1 && "$tests" == 1 && "$results" == 1 ]] || { die 'Cargo evidence has duplicate/missing test or result lines'; return 2; }
  grep -Eq '^test result: ok\. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out(; finished in [0-9]+\.[0-9]+s)?$' "$file" || { die 'Cargo result is not the exact one-pass result'; return 2; }
}

record_environment() {
  local output="$1" cpu_model cpu_flags
  cpu_model="$(awk -F ':' '$1 ~ /model name/ {sub(/^[[:space:]]+/, "", $2); print $2; exit}' /proc/cpuinfo)"
  cpu_flags="$(awk -F ':' '$1 ~ /flags/ {sub(/^[[:space:]]+/, "", $2); print $2; exit}' /proc/cpuinfo)"
  [[ -n "$cpu_model" && -n "$cpu_flags" ]] || die "CPU provenance unavailable"
  {
    echo "utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "git_commit=$(git -C "$VOKRA_ROOT" rev-parse HEAD)"
    echo "git_branch=$(git -C "$VOKRA_ROOT" branch --show-current)"
    echo "upstream_repo=$UPSTREAM_REPO"
    echo "upstream_revision=$UPSTREAM_REVISION"
    echo "upstream_checkpoint=$UPSTREAM_CHECKPOINT"
    echo "upstream_checkpoint_sha256=$UPSTREAM_CHECKPOINT_SHA256"
    echo "corrupt_repo=$CORRUPT_REPO"
    echo "corrupt_revision=$CORRUPT_REVISION"
    echo "corrupt_file=$CORRUPT_FILE"
    echo "corrupt_sha256=$CORRUPT_SHA256"
    echo "uname=$(uname -a)"
    echo "cpu_model=$cpu_model"
    echo "cpu_flags=$cpu_flags"
    echo "nproc=$(nproc)"
    awk '$1 == "MemTotal:" {print "mem_total_kib=" $2; exit}' /proc/meminfo
    rustc --version --verbose
    cargo --version
    uv --version
    uv run --project "$PARITY_PROJECT" --frozen --python 3.12 python -c \
      'import platform, speechbrain, torch, torchaudio; print(f"python={platform.python_version()}"); print(f"speechbrain={speechbrain.__version__}"); print(f"torch={torch.__version__}"); print(f"torchaudio={torchaudio.__version__}")'
  } | tee "$output"
}

# shellcheck disable=SC2016
run_self_test() {
  local script_path="${BASH_SOURCE[0]}" tmp fail=0 cases=0 required
  tmp="$(mktemp -d)"
  trap 'rm -rf -- "$tmp"' EXIT

  cases=$((cases + 1))
  for required in \
    "$UPSTREAM_REPO" "$UPSTREAM_REVISION" "$UPSTREAM_CHECKPOINT" \
    "$UPSTREAM_CHECKPOINT_SHA256" "$MODEL_KIND" "$LICENSE_SPDX" \
    "$CORRUPT_REPO" "$CORRUPT_REVISION" "$CORRUPT_FILE" "$CORRUPT_BYTES" \
    "$CORRUPT_SHA256" "$PARITY_TEST" "$GGUF_ENV" "$JFK_SHA256" \
    "$FIXTURE_PCM_SHA256" "$FIXTURE_FEATURES_SHA256" "$FIXTURE_EMBEDDING_SHA256" \
    "$FIXTURE_WAV_SHA256" "$FIXTURE_WAV_BYTES" \
    "tools/parity/ecapa_tdnn_prepare_checkpoint.py" \
    "tools/parity/ecapa_tdnn_dump_reference.py" \
    'uv run --project "\$PARITY_PROJECT" --frozen --python 3.12 python' \
    'target/release/vokra-cli convert' '  --model "\$MODEL_KIND"' \
    '  --license "\$LICENSE_SPDX"'; do
    if ! grep -Fq -- "$required" "$script_path"; then
      log "self-test FAIL: worker contract lost token: $required"
      fail=1
    fi
  done

  cases=$((cases + 1))
  for required in \
    'uname -s' 'uname -m' 'VOKRA_PUBLISH_ON_VAST' 'MIN_VAST_MEM_KIB=67108864' \
    'MIN_FREE_DISK_KIB=150000000' 'MemTotal=' 'df -Pk' \
    'git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all' \
    'weights_only=True' 'torch.load(args.input, map_location="cpu", weights_only=True)' \
    'cmp -s "\$generated" "\$fixture"' 'verify_committed_oracle' \
    'cargo fmt --all -- --check' \
    'cargo test --locked --workspace' \
    'cargo clippy --locked --workspace --all-targets -- -D warnings'; do
    if ! grep -Fq -- "$required" "$script_path"; then
      log "self-test FAIL: fail-closed contract lost token: $required"
      fail=1
    fi
  done

  cases=$((cases + 1))
  if grep -En '^[[:space:]]*(python3|python|pip)([[:space:]]|$)' "$script_path" >/dev/null; then
    log "self-test FAIL: direct Python/pip command found"
    fail=1
  fi
  if grep -En -- '^[[:space:]]*(git[[:space:]]+push|.*upload\.sh|.*publish-one\.sh)([[:space:]]|$)' \
    "$script_path" >/dev/null; then
    log "self-test FAIL: publication command found"
    fail=1
  fi
  if grep -En -- '(^|[[:space:]])(--push|--upload|--publish)([[:space:]]|$)' "$script_path" >/dev/null; then
    log "self-test FAIL: publication option found"
    fail=1
  fi

  cases=$((cases + 1))
  if "$script_path" --self-test --work-dir "$tmp/other" >/dev/null 2>&1; then
    log "self-test FAIL: extra self-test argument accepted"
    fail=1
  fi
  if "$script_path" --work-dir >/dev/null 2>&1; then
    log "self-test FAIL: missing --work-dir value accepted"
    fail=1
  fi
  if "$script_path" --unknown-self-test-flag >/dev/null 2>&1; then
    log "self-test FAIL: unknown argument accepted"
    fail=1
  fi
  if "$script_path" --work-dir -bad >/dev/null 2>&1 || "$script_path" --work-dir a --work-dir b >/dev/null 2>&1 || "$script_path" --approval-evidence >/dev/null 2>&1 || "$script_path" --self-test --approval-evidence x >/dev/null 2>&1; then
    log "self-test FAIL: malformed or duplicate options accepted"
    fail=1
  fi
  printf '{}\n' > "$tmp/approval.json"
  require_absent_work_dir "$tmp/new/nested/work" "$tmp/approval.json" || { log 'self-test FAIL: nested absent work path rejected'; fail=1; }
  mkdir "$tmp/empty-work"
  if require_absent_work_dir "$tmp/empty-work" "$tmp/approval.json" >/dev/null 2>&1; then log 'self-test FAIL: existing empty work accepted'; fail=1; fi
  ln -s "$tmp/missing" "$tmp/dangling-work"
  if require_absent_work_dir "$tmp/dangling-work" "$tmp/approval.json" >/dev/null 2>&1; then log 'self-test FAIL: dangling work symlink accepted'; fail=1; fi

  rm -rf -- "$tmp"
  trap - EXIT
  if [[ $fail -eq 0 ]]; then
    echo "run-ecapa-tdnn-validation.sh self-test: OK ($cases cases)"
    return 0
  fi
  return 1
}

main() {
  local self_test=0 requested_work_dir="" approval_evidence="" run_stamp work_dir
  local input_dir evidence_dir oracle_dir logs_dir
  local checkpoint corrupt_gguf prepared_path gguf_path cli_embedding
  local reference_wav run_log env_log parity_log oracle_log cli_log workspace_log clippy_log summary_file
  local seen_self=0 seen_work=0 seen_approval=0

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --work-dir)
        (( seen_work == 0 )) || { die 'duplicate --work-dir'; return 2; }
        [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || { die "--work-dir requires a nonempty directory"; return 2; }
        seen_work=1
        requested_work_dir="$2"
        shift 2
        ;;
      --self-test)
        (( seen_self == 0 )) || { die 'duplicate --self-test'; return 2; }
        seen_self=1
        self_test=1
        shift
        ;;
      --approval-evidence)
        (( seen_approval == 0 )) || { die 'duplicate --approval-evidence'; return 2; }
        [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || { die '--approval-evidence requires a nonempty file'; return 2; }
        seen_approval=1
        approval_evidence="$2"
        shift 2
        ;;
      -h|--help)
        usage
        return 0
        ;;
      *)
        die "unknown argument: $1"
        usage
        return 2
        ;;
    esac
  done

  if [[ $self_test -eq 1 ]]; then
    [[ -z "$requested_work_dir$approval_evidence" ]] || { die "--self-test accepts no other arguments"; return 2; }
    run_self_test
    return $?
  fi

  [[ $seen_approval -eq 1 ]] || { die '--approval-evidence is required'; return 2; }
  license_preflight "$approval_evidence"
  run_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  work_dir="${requested_work_dir:-$VOKRA_SCRATCH/ecapa-tdnn-validation/$run_stamp}"
  require_absent_work_dir "$work_dir" "$approval_evidence"
  require_vast_host
  require_tooling
  cd "$VOKRA_ROOT"

  input_dir="$work_dir/input"
  evidence_dir="$work_dir/evidence"
  oracle_dir="$evidence_dir/oracle"
  logs_dir="$evidence_dir/logs"
  reference_wav="$work_dir/ecapa-fixture.wav"
  checkpoint="$input_dir/upstream/$UPSTREAM_CHECKPOINT"
  corrupt_gguf="$input_dir/corrupt/$CORRUPT_FILE"
  prepared_path="$work_dir/prepared/ecapa_tdnn.safetensors"
  gguf_path="$work_dir/ecapa-tdnn-corrected.gguf"
  cli_embedding="$work_dir/ecapa-embedding.f32"
  run_log="$logs_dir/run.log"
  env_log="$logs_dir/environment.txt"
  parity_log="$logs_dir/parity.log"
  oracle_log="$logs_dir/oracle.log"
  cli_log="$logs_dir/cli.log"
  workspace_log="$logs_dir/workspace.log"
  clippy_log="$logs_dir/clippy.log"
  summary_file="$evidence_dir/summary.txt"
  mkdir -p "$logs_dir" "$oracle_dir" "$(dirname "$checkpoint")" "$(dirname "$corrupt_gguf")" "$(dirname "$prepared_path")"
  export UV_CACHE_DIR="$VOKRA_SCRATCH/uv-cache-ecapa-tdnn"
  exec > >(tee -a "$run_log") 2>&1
  # shellcheck disable=SC2154
  trap 'rc=$?; if [[ -n "${summary_file:-}" && ! -f "$summary_file" ]]; then printf "execution_status=FAIL\nexit_code=%s\n" "$rc" > "$summary_file"; fi; exit "$rc"' EXIT

  step "Record VAST environment"
  record_environment "$env_log"

  step "Download and authenticate official SpeechBrain checkpoint"
  download_hf_file "$UPSTREAM_REPO" "$UPSTREAM_REVISION" "$UPSTREAM_CHECKPOINT" "$(dirname "$checkpoint")"
  verify_file "$checkpoint" "$UPSTREAM_CHECKPOINT_SHA256"
  verify_file "$JFK_WAV" "$JFK_SHA256" "$JFK_BYTES"
  verify_file "$FIXTURE_PCM" "$FIXTURE_PCM_SHA256" 208692
  verify_file "$FIXTURE_FEATURES" "$FIXTURE_FEATURES_SHA256" 104640
  verify_file "$FIXTURE_EMBEDDING" "$FIXTURE_EMBEDDING_SHA256" 768
  materialize_fixture_wav "$reference_wav"
  verify_file "$reference_wav" "$FIXTURE_WAV_SHA256" "$FIXTURE_WAV_BYTES"

  step "Authenticate the malformed public artifact being replaced"
  download_hf_file "$CORRUPT_REPO" "$CORRUPT_REVISION" "$CORRUPT_FILE" "$(dirname "$corrupt_gguf")"
  verify_file "$corrupt_gguf" "$CORRUPT_SHA256" "$CORRUPT_BYTES"

  step "Generate independent official SpeechBrain reference"
  uv run --project "$PARITY_PROJECT" --frozen --python 3.12 python "$PARITY_DUMPER" \
    --output-dir "$oracle_dir" --wav "$reference_wav" --source "$UPSTREAM_REPO" \
    --revision "$UPSTREAM_REVISION" --savedir "$input_dir/oracle-cache" \
    2>&1 | tee "$oracle_log"
  grep -Fq '"revision": "0f99f2d0ebe89ac095bcc5903c4dd8f72b367286"' "$oracle_dir/manifest.json" \
    || die "oracle manifest revision is not pinned"
  grep -Fq '"checkpoint_sha256": "0575cb64845e6b9a10db9bcb74d5ac32b326b8dc90352671d345e2ee3d0126a2"' \
    "$oracle_dir/manifest.json" || die "oracle manifest checkpoint identity is not pinned"
  grep -Fq '"speechbrain": "1.0.3"' "$oracle_dir/manifest.json" \
    || die "oracle did not use SpeechBrain 1.0.3"
  verify_committed_oracle "$oracle_dir"

  step "Prepare the safe 200-tensor checkpoint"
  uv run --project "$PARITY_PROJECT" --frozen --python 3.12 python "$PREPARER" \
    --input "$checkpoint" --output "$prepared_path"
  [[ -s "$prepared_path" ]] || die "checkpoint preparation emitted no safetensors"

  step "Build converter and create strict corrected-provenance GGUF"
  cargo build --locked --release -p vokra-cli 2>&1 | tee "$workspace_log"
  target/release/vokra-cli convert --model "$MODEL_KIND" \
    --input "$prepared_path" --output "$gguf_path" --license "$LICENSE_SPDX" \
    2>&1 | tee -a "$workspace_log"
  [[ -s "$gguf_path" ]] || die "strict converter emitted no GGUF"

  export "$GGUF_ENV=$gguf_path"
  step "Run real ECAPA-TDNN CPU parity"
  cargo test --locked -p vokra-models --test parity_ecapa_tdnn_real "$PARITY_TEST" \
    -- --nocapture 2>&1 | tee "$parity_log"
  require_cargo_result "$parity_log" "$PARITY_TEST"
  grep -Fq 'ECAPA-TDNN CPU embedding' "$parity_log" \
    || die "real ECAPA parity did not emit the CPU embedding sentinel"

  step "Run CLI speaker embedding e2e"
  target/release/vokra-cli run --model "$gguf_path" --input "$JFK_WAV" \
    --compare "$JFK_WAV" --backend cpu --output "$cli_embedding" 2>&1 | tee "$cli_log"
  [[ -s "$cli_embedding" ]] || die "CLI speaker e2e emitted no embedding"
  [[ "$(wc -c < "$cli_embedding" | tr -d '[:space:]')" == 768 ]] \
    || die "CLI embedding size is not the pinned 192-f32 width"
  grep -Fq 'cosine_similarity=' "$cli_log" || die "CLI speaker compare sentinel missing"

  step "Run workspace verification gates on VAST"
  bash "$VOKRA_ROOT/scripts/check-forbidden-symbols.sh" 2>&1 | tee -a "$workspace_log"
  bash "$VOKRA_ROOT/scripts/check-zero-deps.sh" 2>&1 | tee -a "$workspace_log"
  bash "$VOKRA_ROOT/scripts/check-bound-arch-coverage.sh" 2>&1 | tee -a "$workspace_log"
  cargo fmt --all -- --check 2>&1 | tee -a "$workspace_log"
  cargo test --locked --workspace 2>&1 | tee -a "$workspace_log"
  cargo clippy --locked --workspace --all-targets -- -D warnings 2>&1 | tee "$clippy_log"
  cargo deny check licenses advisories bans 2>&1 | tee -a "$workspace_log"
  cargo audit 2>&1 | tee -a "$workspace_log"

  {
    echo "execution_status=PASS"
    echo "git_commit=$(git -C "$VOKRA_ROOT" rev-parse HEAD)"
    echo "upstream_repo=$UPSTREAM_REPO"
    echo "upstream_revision=$UPSTREAM_REVISION"
    echo "upstream_checkpoint_sha256=$(sha256_file "$checkpoint")"
    echo "corrupt_target_sha256=$(sha256_file "$corrupt_gguf")"
    echo "prepared_safetensors_sha256=$(sha256_file "$prepared_path")"
    echo "corrected_gguf_sha256=$(sha256_file "$gguf_path")"
    echo "oracle_manifest_sha256=$(sha256_file "$oracle_dir/manifest.json")"
    echo "cli_embedding_sha256=$(sha256_file "$cli_embedding")"
    echo "real_parity=$PARITY_TEST:PASS"
    echo "cli_speaker_e2e=PASS"
    echo "workspace_gates=PASS"
    echo "publication=NOT_RUN"
  } | tee "$summary_file"
  trap - EXIT
  log "PASS: pull $evidence_dir and logs before destroying the VAST instance"
}

main "$@"
