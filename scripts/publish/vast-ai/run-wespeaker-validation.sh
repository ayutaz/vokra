#!/usr/bin/env bash
# VAST-only WeSpeaker official-checkpoint validation worker.
# Stages immutable inputs, creates corrected provenance, runs parity and gates.
# There is deliberately no upload, publish, or Git-push path.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VOKRA_ROOT="${VOKRA_ROOT:-$DEFAULT_ROOT}"
VOKRA_SCRATCH="${VOKRA_SCRATCH:-$HOME/scratchpad}"
PARITY_PROJECT="$VOKRA_ROOT/tools/parity/wespeaker"
PARITY_DUMPER="$VOKRA_ROOT/tools/parity/wespeaker_dump_reference.py"
PREPARER="$VOKRA_ROOT/tools/parity/wespeaker_prepare_checkpoint.py"
PREFLIGHT_GATE="$PARITY_PROJECT/preflight_gate.py"
PREFLIGHT_MANIFEST="$PARITY_PROJECT/license_gate_manifest.json"
JFK_WAV="$VOKRA_ROOT/tests/fixtures/audio/jfk-30s.wav"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

# All identity values are repository-recorded. Do not replace a missing
# digest with a newly observed value in this worker.
UPSTREAM_HF="Wespeaker/wespeaker-voxceleb-resnet34-LM"
UPSTREAM_REVISION="f0c48c298fd835726c27956a5d617bad7115627e"
UPSTREAM_CHECKPOINT="avg_model.pt"
UPSTREAM_CHECKPOINT_SHA256="9872b375f2c6a3851ca471cbbf59e06efd23a627d78bf5872e1f0269fd298449"
UPSTREAM_CHECKPOINT_BYTES=45053131
UPSTREAM_CHECKPOINT_GIT_OID="7f92ddd059d244c7d2653650d3be85de9f136c41"
UPSTREAM_CONFIG="config.yaml"
UPSTREAM_CONFIG_BYTES=1673
UPSTREAM_CONFIG_SHA256="3cf7d3243464cd939083e29d2be65c2abcdd954c1a64559bad73b74ffdb0db3e"
UPSTREAM_CONFIG_GIT_OID="1941982501edc3909a56c9bca025fecf10cf28d2"
SOURCE_REPOSITORY="https://github.com/wenet-e2e/wespeaker.git"
SOURCE_REVISION="45941e7cba2c3ea99e232d02bedf617fc71b0dad"
SOURCE_LICENSE_BYTES=11357
SOURCE_LICENSE_SHA256="c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"
SOURCE_LICENSE_GIT_BLOB="261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64"
SOURCE_RESNET_BYTES=9564
SOURCE_RESNET_SHA256="6f3c8219be2c9a8b9eabed8169c1abaec3e48670be7aaf1e792138b2b20e68c4"
SOURCE_RESNET_GIT_BLOB="17607e6d2c72627e15db4214cacfa9d7b89ca945"
SOURCE_POOLING_BYTES=10255
SOURCE_POOLING_SHA256="768910f8e88cb47e742274563339d7e780cb9d56c629c4d4124605296686f0f9"
SOURCE_POOLING_GIT_BLOB="47120eead47a511939267470496539804c17b7d3"

# Canonical 182-tensor artifact used by the existing real parity test.
PUBLIC_REPO="vokra/pyannote-wespeaker-voxceleb-resnet34-lm"
PUBLIC_REVISION="8e27acd8a875088f1a7321f40610397bf964a446"
PUBLIC_FILE="pyannote-wespeaker.restamped.gguf"
PUBLIC_BYTES=26584064
PUBLIC_SHA256="6dccbc026e9c32a8f99f3441e64f1ff52e36afb055442595c86cda8021c78c39"
JFK_BYTES=352078
JFK_SHA256="58adb4ea501d955fcd40bfbb69128f8f40428b81d8716b9ed337949773be253f"

MIN_VAST_MEM_KIB=67108864
MIN_FREE_DISK_KIB=150000000

log() { printf '[wespeaker-vast] %s\n' "$*" >&2; }
step() { printf '\n[wespeaker-vast] ==== %s ====\n' "$*" >&2; }
die() { log "ERROR: $*"; return 2; }

usage() {
  cat <<'EOF' >&2
usage: run-wespeaker-validation.sh --approval-evidence <json> [--work-dir <absent-dir>]
       run-wespeaker-validation.sh --self-test

VAST-only, no-upload WeSpeaker worker. It stages the exact official avg_model
and source revision, audits it with wespeaker_prepare_checkpoint.py, creates the
strict official-combined-bare-219 corrected-provenance GGUF, runs the
independent Wespeaker oracle, existing real parity and CLI smoke tests, and
workspace gates.

The 219-tensor bridge is fail-closed: only the exact state dict with 36 scalar
int64 BatchNorm counters is accepted. There is no upload, publish, push option, or
Git-push operation. Pull only logs/manifests before destroying the instance.
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
  local path="$1" expected_hash="$2" expected_bytes="${3:-}" actual_hash actual_bytes=""
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

verify_source_file() {
  local root="$1" relative="$2" expected_bytes="$3" expected_sha="$4" expected_blob="$5" path actual_blob
  path="$root/$relative"
  [[ -f "$path" && ! -L "$path" ]] || { die "source file is missing or symlinked: $relative"; return 2; }
  [[ "$(wc -c < "$path" | tr -d '[:space:]')" == "$expected_bytes" ]] || { die "source byte size mismatch: $relative"; return 2; }
  [[ "$(sha256_file "$path")" == "$expected_sha" ]] || { die "source SHA mismatch: $relative"; return 2; }
  actual_blob="$(git -C "$root" hash-object "$relative")"
  [[ "$actual_blob" == "$expected_blob" ]] || { die "source Git blob mismatch: $relative"; return 2; }
}

canonicalize_uncreated() {
  local path="$1" suffix='' name parent
  local scan rest component
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
    parent="${path%/*}"; [[ "$parent" == "$path" ]] && parent='.'
    [[ -n "$parent" ]] || parent='/'; path="$parent"
    [[ ! -L "$path" ]] || return 1
  done
  (cd -P "$path" && printf '%s%s\n' "$PWD" "$suffix")
}

paths_overlap() { [[ "$1" == "$2" || "$1" == "$2"/* || "$2" == "$1"/* ]]; }

require_absent_work_dir() {
  local target="$1" approval="$2" canonical protected other
  [[ ! -e "$target" && ! -L "$target" ]] || { die "--work-dir must be absent and non-symlink: $target"; return 2; }
  canonical="$(canonicalize_uncreated "$target")" || { die "cannot canonicalize --work-dir: $target"; return 2; }
  for protected in "$VOKRA_ROOT" "$PARITY_PROJECT" "$PREFLIGHT_GATE" "$PREFLIGHT_MANIFEST" "$approval" "$JFK_WAV"; do
    [[ -e "$protected" || -L "$protected" ]] || continue
    [[ ! -L "$protected" ]] || { die "protected path is symlinked: $protected"; return 2; }
    other="$(canonicalize_uncreated "$protected")" || { die "cannot canonicalize protected path: $protected"; return 2; }
    paths_overlap "$canonical" "$other" && { die "--work-dir overlaps protected path: $protected"; return 2; }
  done
  return 0
}

require_one_cargo_result() {
  local log_path="$1" test_name="$2"
  [[ "$(grep -Ec "^test $test_name \\.\\.\\. ok$" "$log_path" || true)" == 1 ]] || die "named Cargo test did not pass exactly once"
  [[ "$(grep -Ec '^test [^ ]+ \\.\\.\\.' "$log_path" || true)" == 1 ]] || die "Cargo emitted extra or missing test lines"
  [[ "$(grep -Ec '^test result:' "$log_path" || true)" == 1 ]] || die "Cargo emitted extra or missing result lines"
  grep -Eq '^test result: ok\\. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out(; finished in [0-9]+\\.[0-9]+s)?$' "$log_path" \
    || die "Cargo result is not the exact one-pass result"
}

require_official_cpu_sentinel() {
  local log_path="$1"
  [[ "$(grep -Ec '^WESPEAKER_OFFICIAL_COMBINED_CPU_VS_UPSTREAM PASS$' "$log_path" || true)" == 1 ]] \
    || die "official CPU parity sentinel missing or duplicated"
  [[ "$(grep -Ec '^WESPEAKER_OFFICIAL_COMBINED_(CPU_VS_UPSTREAM|METAL_VS_CPU) (PASS|FAIL)$' "$log_path" || true)" == 1 ]] \
    || die "official parity sentinel family contains extra or malformed lines"
}

write_apple_args() {
  local output="$1" gguf_sha="$2" reference_sha="$3"
  {
    printf '# Generated for the separate no-upload Apple WeSpeaker validation.\n'
    printf "scripts/verify/apple-silicon-wespeaker.sh \\\n"
    printf "  --gguf '%s' \\\n" '<APPLE_WESPEAKER_GGUF_PATH>'
    printf "  --gguf-sha256 '%s' \\\n" "$gguf_sha"
    printf "  --reference '%s' \\\n" '<APPLE_WESPEAKER_REFERENCE_DIR>'
    printf "  --reference-manifest-sha256 '%s' \\\n" "$reference_sha"
    printf "  --approval-evidence '%s' \\\n" '<APPLE_WESPEAKER_APPROVAL_EVIDENCE>'
    printf "  --evidence-dir '%s'\n" '<APPLE_EMPTY_EVIDENCE_DIR>'
  } > "$output"
}

download_hf_file() {
  local repository="$1" revision="$2" filename="$3" output_dir="$4"
  mkdir -p "$output_dir"
  local url="https://huggingface.co/$repository/resolve/$revision/$filename?download=true"
  if [[ -n "${HF_TOKEN:-${HF:-}}" ]]; then
    curl -fsSL --retry 3 -H "Authorization: Bearer ${HF_TOKEN:-${HF:-}}" "$url" -o "$output_dir/$filename"
  else
    curl -fsSL --retry 3 "$url" -o "$output_dir/$filename"
  fi
  [[ -f "$output_dir/$filename" ]] || die "download did not produce $output_dir/$filename"
}

pre_sync_gate() {
  local evidence="$1"
  command -v uv >/dev/null 2>&1 || die "uv is required before the WeSpeaker gate"
  [[ -f "$PARITY_PROJECT/uv.lock" && ! -L "$PARITY_PROJECT/uv.lock" && -f "$PARITY_PROJECT/pyproject.toml" && ! -L "$PARITY_PROJECT/pyproject.toml" && -f "$PREFLIGHT_GATE" && ! -L "$PREFLIGHT_GATE" && -f "$PREFLIGHT_MANIFEST" && ! -L "$PREFLIGHT_MANIFEST" ]] || die "WeSpeaker pre-sync gate inputs are missing"
  [[ -f "$evidence" && -s "$evidence" && ! -L "$evidence" ]] || die "external approval evidence must be a non-empty regular non-symlink file"
  step "Validate exact WeSpeaker closure before host/tooling/scratch/network work"
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python "$PREFLIGHT_GATE" --project "$PARITY_PROJECT" --manifest "$PREFLIGHT_MANIFEST" --approval-evidence "$evidence"
}

checkout_exact_source() {
  local repository="$1" revision="$2" output="$3"
  [[ ! -e "$output" ]] || die "source target already exists: $output"
  mkdir -p "$output"
  git -C "$output" init -q
  git -C "$output" remote add origin "$repository"
  git -C "$output" fetch -q --depth=1 origin "$revision"
  git -C "$output" checkout -q --detach FETCH_HEAD
  [[ "$(git -C "$output" rev-parse HEAD)" == "$revision" ]] || die "source checkout did not land on $revision"
  verify_source_file "$output" LICENSE "$SOURCE_LICENSE_BYTES" "$SOURCE_LICENSE_SHA256" "$SOURCE_LICENSE_GIT_BLOB"
  verify_source_file "$output" wespeaker/models/resnet.py "$SOURCE_RESNET_BYTES" "$SOURCE_RESNET_SHA256" "$SOURCE_RESNET_GIT_BLOB"
  verify_source_file "$output" wespeaker/models/pooling_layers.py "$SOURCE_POOLING_BYTES" "$SOURCE_POOLING_SHA256" "$SOURCE_POOLING_GIT_BLOB"
  [[ -z "$(git -C "$output" status --porcelain --untracked-files=all)" ]] || die "source checkout is not clean"
}

require_vast_host() {
  local mem_kib free_kib disk_path parent
  [[ "${VOKRA_PUBLISH_ON_VAST:-0}" == "1" ]] || die "VOKRA_PUBLISH_ON_VAST=1 is absent; run provision.sh first"
  [[ "$(uname -s)" == "Linux" ]] || die "model work is Linux/VAST-only; refusing host $(uname -s)"
  mem_kib="$(awk '$1 == "MemTotal:" {print $2; exit}' /proc/meminfo)"
  [[ "$mem_kib" =~ ^[0-9]+$ ]] || die "could not read MemTotal"
  (( mem_kib >= MIN_VAST_MEM_KIB )) || die "MemTotal=$mem_kib KiB is below the VAST 64-GiB guard (67108864 KiB)"
  disk_path="$VOKRA_SCRATCH"
  while [[ ! -e "$disk_path" ]]; do
    parent="$(dirname "$disk_path")"
    [[ "$parent" != "$disk_path" ]] || die "scratch parent cannot be resolved"
    disk_path="$parent"
  done
  [[ -d "$disk_path" && ! -L "$disk_path" ]] || die "scratch filesystem path is not a real directory"
  free_kib="$(df -Pk "$disk_path" | awk 'NR == 2 {print $4}')"
  [[ "$free_kib" =~ ^[0-9]+$ ]] || die "could not read free disk"
  (( free_kib >= MIN_FREE_DISK_KIB )) || die "free disk=$free_kib KiB is below the 150-GB guard (150000000 KiB)"
}

require_tooling() {
  local tool
  for tool in uv cargo rustc rustup git awk grep find tee wc tr curl; do
    command -v "$tool" >/dev/null 2>&1 || die "required tool missing: $tool"
  done
  [[ -d "$VOKRA_ROOT/.git" ]] || die "$VOKRA_ROOT is not a git checkout"
  [[ -f "$PARITY_PROJECT/uv.lock" ]] || die "parity uv.lock is missing"
  [[ -f "$PARITY_DUMPER" ]] || die "official WeSpeaker dumper is missing"
  [[ -f "$PREPARER" ]] || die "dedicated WeSpeaker preparer is missing"
  [[ -f "$JFK_WAV" ]] || die "committed JFK fixture is missing"
  [[ -z "$(git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all)" ]] || die "VAST checkout must be clean"
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
    echo "uname=$(uname -a)"
    echo "cpu_model=$cpu_model"
    echo "cpu_flags=$cpu_flags"
    echo "nproc=$(command -v nproc >/dev/null 2>&1 && nproc || echo unavailable)"
    awk '$1 == "MemTotal:" {print "mem_total_kib=" $2; exit}' /proc/meminfo
    rustc --version --verbose
    cargo --version
    uv --version
    UV_NO_CACHE=1 uv run --no-cache --project "$PARITY_PROJECT" --frozen --python 3.12 python -c \
      'import platform, torch, torchaudio; print(f"python={platform.python_version()}"); print(f"torch={torch.__version__}"); print(f"torchaudio={torchaudio.__version__}")'
  } | tee "$output"
}

# shellcheck disable=SC2016
run_self_test() {
  local script_path="${BASH_SOURCE[0]}" tmp payload expected actual fail=0 cases=0 required unsafe_loader_bad legacy_preparer
  tmp="$(mktemp -d)"
  cleanup_self_test() { rm -rf -- "$tmp"; }
  trap cleanup_self_test EXIT
  payload="$tmp/payload"
  printf 'wespeaker-worker-self-test\n' > "$payload"
  expected="$(sha256_file "$payload")"
  actual="$(sha256_file "$payload")"
  cases=$((cases + 1))
  [[ "$actual" == "$expected" ]] || { log "self-test FAIL: local hash identity"; fail=1; }

  cases=$((cases + 1))
  for required in "$UPSTREAM_HF" "$UPSTREAM_REVISION" "$UPSTREAM_CHECKPOINT" \
    "$UPSTREAM_CHECKPOINT_SHA256" "$SOURCE_REPOSITORY" "$SOURCE_REVISION" \
    "$PUBLIC_REPO" "$PUBLIC_REVISION" "$PUBLIC_SHA256" \
    "tools/parity/wespeaker_prepare_checkpoint.py" "tools/parity/wespeaker_dump_reference.py" \
    "parity_wespeaker_real" "public_pyannote_artifact_matches_upstream_wespeaker" \
    "official_combined_artifact_matches_upstream_wespeaker" \
    "run::tests::speaker_real_gguf_e2e_identical_inputs_gated" "--frozen --python 3.12" \
    "VOKRA_PUBLISH_ON_VAST" "uname -s" "MIN_VAST_MEM_KIB=67108864" "MemTotal:" \
    "MIN_FREE_DISK_KIB=150000000" "df -Pk" 'git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all' \
    "cargo test --locked --workspace" "cargo clippy --locked --workspace --all-targets -- -D warnings"; do
    if ! grep -Fq -- "$required" "$script_path"; then
      log "self-test FAIL: worker contract lost token: $required"
      fail=1
    fi
  done

  cases=$((cases + 1))
  if grep -En '^[[:space:]]*(python3|python|pip)([[:space:]]|$)' "$script_path" >/dev/null; then
    log "self-test FAIL: direct Python/pip command found"; fail=1
  fi
  if grep -En '(^|[[:space:]])(git[[:space:]]+push|[^[:space:]]*upload\.sh|[^[:space:]]*publish-one\.sh)([[:space:]]|$)' "$script_path" >/dev/null; then
    log "self-test FAIL: publication command found"; fail=1
  fi
  unsafe_loader_bad="weights_only"
  unsafe_loader_bad+="=False"
  legacy_preparer="nemo_pt_to"
  legacy_preparer+="_safetensors.py"
  if grep -Fq -- "$legacy_preparer" "$script_path" || grep -Fq -- "$unsafe_loader_bad" "$script_path"; then
    log "self-test FAIL: unsafe generic checkpoint loader found"; fail=1
  fi
  if [[ ! -f "$PREPARER" ]] \
    || ! grep -Fq -- 'weights_only=True' "$PREPARER" \
    || grep -Fq -- "$unsafe_loader_bad" "$PREPARER"; then
    log "self-test FAIL: dedicated preparer is not safe-loader-only"; fail=1
  fi
  local apple_args="$tmp/apple.args.sh"
  write_apple_args "$apple_args" "$PUBLIC_SHA256" "$PUBLIC_SHA256"
  bash -n "$apple_args" || { log "self-test FAIL: generated Apple args are not shell syntax"; fail=1; }
  for placeholder in APPLE_WESPEAKER_GGUF_PATH APPLE_WESPEAKER_REFERENCE_DIR APPLE_WESPEAKER_APPROVAL_EVIDENCE APPLE_EMPTY_EVIDENCE_DIR; do
    grep -Fq -- "'<${placeholder}>'" "$apple_args" || { log "self-test FAIL: Apple placeholder is not quoted: $placeholder"; fail=1; }
  done
  if grep -F "$VOKRA_SCRATCH" "$apple_args" >/dev/null 2>&1; then
    log "self-test FAIL: generated Apple args contain a VAST-local path"; fail=1
  fi
  cases=$((cases + 1))
  if "$script_path" --self-test --work-dir "$tmp/other" >/dev/null 2>&1; then
    log "self-test FAIL: extra --self-test argument accepted"; fail=1
  fi
  if "$script_path" --unknown-self-test-flag >/dev/null 2>&1; then
    log "self-test FAIL: unknown argument accepted"; fail=1
  fi
  if "$script_path" --work-dir -bad >/dev/null 2>&1 || "$script_path" --approval-evidence -bad >/dev/null 2>&1; then
    log "self-test FAIL: leading-dash option value accepted"; fail=1
  fi
  printf '{}' > "$tmp/evidence.json"
  set +e
  VOKRA_SCRATCH="$tmp/scratch" VOKRA_PUBLISH_ON_VAST=1 "$script_path" \
    --approval-evidence "$tmp/evidence.json" --work-dir "$tmp/work" >"$tmp/gate.log" 2>&1
  local gate_rc=$?
  set -e
  if [[ $gate_rc -ne 2 || -e "$tmp/scratch" || -e "$tmp/work" || -e "$tmp/uv-cache-wespeaker" ]]; then
    log "self-test FAIL: production gate did not stop before work/cache creation"; fail=1
  fi
  printf '{}\n' > "$tmp/evidence.json"
  require_absent_work_dir "$tmp/new-work" "$tmp/evidence.json" || fail=1
  mkdir "$tmp/empty-work"
  if require_absent_work_dir "$tmp/empty-work" "$tmp/evidence.json" >/dev/null 2>&1; then fail=1; fi
  ln -s "$tmp/missing-work" "$tmp/dangling-work"
  if require_absent_work_dir "$tmp/dangling-work" "$tmp/evidence.json" >/dev/null 2>&1; then fail=1; fi
  if require_absent_work_dir "$VOKRA_ROOT/wespeaker-self-test-work" "$tmp/evidence.json" >/dev/null 2>&1; then fail=1; fi
  if require_absent_work_dir "$tmp/evidence.json/child" "$tmp/evidence.json" >/dev/null 2>&1; then fail=1; fi
  if require_absent_work_dir "$JFK_WAV/child" "$tmp/evidence.json" >/dev/null 2>&1; then fail=1; fi
  mkdir -p "$tmp/real-parent/child"
  ln -s "$tmp/real-parent" "$tmp/link-parent"
  if require_absent_work_dir "$tmp/link-parent/child/new-work" "$tmp/evidence.json" >/dev/null 2>&1; then fail=1; fi
  rm -rf -- "$tmp"
  trap - EXIT
  if [[ $fail -eq 0 ]]; then
    echo "run-wespeaker-validation.sh self-test: OK ($cases cases)"
    return 0
  fi
  return 1
}

main() {
  local self_test=0 requested_work_dir="" approval_evidence="" run_stamp work_dir
  local inputs_dir sources_dir logs_dir reference_dir public_dir
  local checkpoint checkpoint_config public_gguf source_dir prepared_safetensors
  local corrected_gguf run_log env_log public_log parity_log cli_log workspace_log summary_file
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --work-dir)
        [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || { die "--work-dir requires a directory"; return 2; }
        [[ -z "$requested_work_dir" ]] || { die "duplicate --work-dir"; return 2; }
        requested_work_dir="$2"; shift 2 ;;
      --approval-evidence)
        [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || { die "--approval-evidence requires a file"; return 2; }
        [[ -z "$approval_evidence" ]] || { die "duplicate --approval-evidence"; return 2; }
        approval_evidence="$2"; shift 2 ;;
      --self-test)
        self_test=1; shift ;;
      -h|--help)
        usage; return 0 ;;
      *)
        die "unknown argument: $1"; usage; return 2 ;;
    esac
  done
  if [[ $self_test -eq 1 ]]; then
    [[ -z "$requested_work_dir$approval_evidence" ]] || { die "--self-test accepts no other arguments"; return 2; }
    run_self_test
    return $?
  fi
  [[ -n "$approval_evidence" ]] || { die "--approval-evidence is required"; return 2; }
  pre_sync_gate "$approval_evidence"
  require_tooling
  cd "$VOKRA_ROOT"
  run_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  work_dir="${requested_work_dir:-$VOKRA_SCRATCH/wespeaker-validation/$run_stamp}"
  require_absent_work_dir "$work_dir" "$approval_evidence"
  require_vast_host
  inputs_dir="$work_dir/inputs"; sources_dir="$work_dir/sources"
  logs_dir="$work_dir/logs"; reference_dir="$work_dir/reference"; public_dir="$inputs_dir/public-pyannote"
  checkpoint="$inputs_dir/upstream/$UPSTREAM_CHECKPOINT"
  checkpoint_config="$inputs_dir/upstream/$UPSTREAM_CONFIG"
  source_dir="$sources_dir/wespeaker"
  prepared_safetensors="$work_dir/prepared/official-combined-bare-219.safetensors"
  corrected_gguf="$work_dir/wespeaker-official-combined-corrected-provenance.gguf"
  public_gguf="$public_dir/$PUBLIC_FILE"
  run_log="$logs_dir/run.log"; env_log="$logs_dir/environment.txt"
  public_log="$logs_dir/public-parity.log"; parity_log="$logs_dir/parity.log"; cli_log="$logs_dir/cli.log"
  workspace_log="$logs_dir/workspace-gates.log"; summary_file="$logs_dir/summary.txt"
  mkdir -p "$logs_dir" "$reference_dir" "$public_dir" "$(dirname "$checkpoint")" "$(dirname "$prepared_safetensors")"
  export UV_CACHE_DIR="$VOKRA_SCRATCH/uv-cache-wespeaker"
  exec > >(tee -a "$run_log") 2>&1
  # shellcheck disable=SC2154
  trap 'rc=$?; if [[ -n "${summary_file:-}" && ! -f "$summary_file" ]]; then printf "execution_status=FAIL\nexit_code=%s\n" "$rc" > "$summary_file"; fi; exit "$rc"' EXIT

  step "Sync locked Python 3.12 reference environment"
  UV_NO_CACHE=1 uv sync --no-cache --project "$PARITY_PROJECT" --frozen --python 3.12
  step "Stage and verify exact official checkpoint and public parity artifact"
  download_hf_file "$UPSTREAM_HF" "$UPSTREAM_REVISION" "$UPSTREAM_CHECKPOINT" "$(dirname "$checkpoint")"
  download_hf_file "$UPSTREAM_HF" "$UPSTREAM_REVISION" "$UPSTREAM_CONFIG" "$(dirname "$checkpoint_config")"
  verify_file "$checkpoint" "$UPSTREAM_CHECKPOINT_SHA256" "$UPSTREAM_CHECKPOINT_BYTES"
  verify_file "$checkpoint_config" "$UPSTREAM_CONFIG_SHA256" "$UPSTREAM_CONFIG_BYTES"
  download_hf_file "$PUBLIC_REPO" "$PUBLIC_REVISION" "$PUBLIC_FILE" "$public_dir"
  verify_file "$public_gguf" "$PUBLIC_SHA256" "$PUBLIC_BYTES"
  verify_file "$JFK_WAV" "$JFK_SHA256" "$JFK_BYTES"
  step "Check out exact independent WeSpeaker source"
  checkout_exact_source "$SOURCE_REPOSITORY" "$SOURCE_REVISION" "$source_dir"
  step "Record VAST environment"
  record_environment "$env_log"
  step "Generate independent official WeSpeaker reference"
  UV_NO_CACHE=1 uv run --no-cache --project "$PARITY_PROJECT" --frozen --python 3.12 python "$PARITY_DUMPER" \
    --output-dir "$reference_dir" --checkpoint "$checkpoint" --wespeaker-source "$source_dir"
  cp "$reference_dir/manifest.json" "$logs_dir/reference-manifest.json"
  step "Prepare exact official combined 219-tensor input"
  UV_NO_CACHE=1 uv run --no-cache --project "$PARITY_PROJECT" --frozen --python 3.12 python "$PREPARER" \
    --checkpoint "$checkpoint" --output "$prepared_safetensors"
  cp "$prepared_safetensors.manifest.json" "$logs_dir/prepared-manifest.json"
  step "Build converter and CLI"
  cargo build --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release -p vokra-cli 2>&1 | tee "$workspace_log"
  target/release/vokra-cli convert --model wespeaker --input "$prepared_safetensors" --output "$corrected_gguf" 2>&1 | tee -a "$workspace_log"
  [[ -s "$corrected_gguf" ]] || die "converter did not produce corrected GGUF"
  write_apple_args "$logs_dir/apple-silicon-wespeaker.args.sh" \
    "$(sha256_file "$corrected_gguf")" "$(sha256_file "$reference_dir/manifest.json")"
  step "Run existing real WeSpeaker parity against canonical public artifact"
  VOKRA_WESPEAKER_GGUF="$public_gguf" cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release \
    -p vokra-models --test parity_wespeaker_real public_pyannote_artifact_matches_upstream_wespeaker \
    -- --exact --nocapture 2>&1 | tee "$public_log"
  require_one_cargo_result "$public_log" public_pyannote_artifact_matches_upstream_wespeaker
  grep -F "WeSpeaker CPU end-to-end" "$public_log" >/dev/null || die "existing parity sentinel missing"
  step "Run corrected official GGUF CLI smoke and gated speaker e2e"
  target/release/vokra-cli run --model "$corrected_gguf" --input "$JFK_WAV" --backend cpu 2>&1 | tee "$cli_log"
  VOKRA_WESPEAKER_OFFICIAL_GGUF="$corrected_gguf" \
    cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release \
      -p vokra-models --test parity_wespeaker_real \
      official_combined_artifact_matches_upstream_wespeaker -- --exact --nocapture \
      2>&1 | tee -a "$parity_log"
  require_one_cargo_result "$parity_log" official_combined_artifact_matches_upstream_wespeaker
  require_official_cpu_sentinel "$parity_log"
  VOKRA_WESPEAKER_GGUF="$corrected_gguf" cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release \
    -p vokra-cli run::tests::speaker_real_gguf_e2e_identical_inputs_gated -- --exact --nocapture 2>&1 | tee -a "$cli_log"
  grep -F "test run::tests::speaker_real_gguf_e2e_identical_inputs_gated ... ok" "$cli_log" >/dev/null \
    || die "CLI speaker e2e did not run exactly one passing test"
  step "Run workspace verification gates on VAST"
  cargo fmt --all -- --check 2>&1 | tee -a "$workspace_log"
  bash "$VOKRA_ROOT/scripts/check-forbidden-symbols.sh" 2>&1 | tee -a "$workspace_log"
  bash "$VOKRA_ROOT/scripts/check-zero-deps.sh" 2>&1 | tee -a "$workspace_log"
  bash "$VOKRA_ROOT/scripts/check-bound-arch-coverage.sh" 2>&1 | tee -a "$workspace_log"
  cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --workspace 2>&1 | tee -a "$workspace_log"
  cargo clippy --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --workspace --all-targets -- -D warnings 2>&1 | tee -a "$workspace_log"
  {
    echo "execution_status=PASS"
    echo "pyannote_182_cpu_parity=PASS"
    echo "official_combined_219_cpu_parity=PASS"
    echo "official_combined_gguf=generated_corrected_provenance"
    echo "git_commit=$(git -C "$VOKRA_ROOT" rev-parse HEAD)"
    echo "upstream_hf=$UPSTREAM_HF"; echo "upstream_revision=$UPSTREAM_REVISION"
    echo "upstream_checkpoint_sha256=$UPSTREAM_CHECKPOINT_SHA256"; echo "source_revision=$SOURCE_REVISION"
    echo "upstream_checkpoint_git_oid=$UPSTREAM_CHECKPOINT_GIT_OID"; echo "upstream_config_git_oid=$UPSTREAM_CONFIG_GIT_OID"
    echo "public_revision=$PUBLIC_REVISION"; echo "public_gguf_sha256=$PUBLIC_SHA256"
    echo "reference_manifest_sha256=$(sha256_file "$reference_dir/manifest.json")"
    echo "corrected_gguf_sha256=$(sha256_file "$corrected_gguf")"
    echo "workspace_gates=PASS"; echo "upload=NOT_RUN"
  } | tee "$summary_file"
  trap - EXIT
  log "PASS: pull $logs_dir and $reference_dir; do not pull model artifacts; destroy the VAST instance"
}

main "$@"
