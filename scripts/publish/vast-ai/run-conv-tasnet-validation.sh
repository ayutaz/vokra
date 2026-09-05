#!/usr/bin/env bash
# VAST/Linux-only Conv-TasNet validation.  The upstream license declarations
# conflict, so this worker permits an explicit research policy only and never
# uploads or publishes a model.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VOKRA_ROOT="${VOKRA_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
PARITY_PROJECT="$VOKRA_ROOT/tools/parity/conv_tasnet"
PREFLIGHT_GATE="$PARITY_PROJECT/license_gate.py"
PREFLIGHT_MANIFEST="$PARITY_PROJECT/license_gate_manifest.json"
PREPARER="$VOKRA_ROOT/tools/parity/conv_tasnet_prepare_checkpoint.py"
DUMPER="$VOKRA_ROOT/tools/parity/conv_tasnet_dump_reference.py"
UPSTREAM_REPO="JorisCos/ConvTasNet_Libri1Mix_enhsingle_16k"
UPSTREAM_REVISION="bb8a876bc157b5cf3c405994accb798c49146016"
CHECKPOINT_FILE="pytorch_model.bin"
CHECKPOINT_SHA256="dd8ddefe95a35761f8a48643a618eba908572d04d33208a8ed5451fb5a4378d0"
MIN_VAST_MEM_KIB=$((64 * 1024 * 1024))
MIN_FREE_DISK_KIB=$((150 * 1024 * 1024))
CONVTASNET_UV_CACHE_DIR="${CONVTASNET_UV_CACHE_DIR:-/tmp/vokra-conv-tasnet-uv-cache}"

log() { printf '[conv-tasnet-vast] %s\n' "$*" >&2; }
die() { log "ERROR: $*"; return 2; }

usage() {
  cat <<'EOF'
usage: run-conv-tasnet-validation.sh --checkpoint-sha256 <64-hex> --approval-evidence <file> [--work-dir <absent-dir>]
       run-conv-tasnet-validation.sh --self-test

VAST/Linux-only validation of the pinned Asteroid Conv-TasNet checkpoint.
The worker uses the safe weights-only preparer, the independent Asteroid
oracle, and an explicit CompliancePolicy research opt-in for the upstream
license conflict. It requires exact committed fixture hashes, runs native
CPU parity on VAST, and performs no upload or publication.
EOF
}

sha256_file() {
  sha256sum "$1" | awk '{print $1}'
}

self_test() {
  local path="${BASH_SOURCE[0]}" fail=0 token temporary log_file
  for token in \
    'VOKRA_PUBLISH_ON_VAST=1' 'Linux' 'x86_64' 'MIN_VAST_MEM_KIB' '/proc/meminfo' \
    'df -Pk' 'CARGO_BUILD_JOBS=1' 'cargo fmt --all -- --check' \
    'cargo metadata --no-deps --format-version 1' 'hf_hub_download' \
    "$UPSTREAM_REPO" "$UPSTREAM_REVISION" "$CHECKPOINT_FILE" \
    'conv_tasnet_prepare_checkpoint.py' 'weights_only=True' \
    'conv_tasnet_dump_reference.py' 'weights_only=False' \
    'fixture hash' 'CompliancePolicy' 'with_research_license(true)' \
    'MEASURED_NOT_GATED' 'NO_UPLOAD' 'git status --porcelain' \
    'conv_tasnet_dump_reference.py --self-test' 'license_gate.py' '--approval-evidence' 'UV_NO_CACHE=1' \
    'VOKRA_REMOTE_APPLE_SILICON=1'; do
    if ! grep -Fq -- "$token" "$path"; then
      log "self-test FAIL: missing contract token: $token"
      fail=1
    fi
  done
  if grep -En '(^|[[:space:]])(git[[:space:]]+push|.*upload\.sh|.*publish-one\.sh|--push|--upload)([[:space:]]|$)' "$path" >/dev/null; then
    log 'self-test FAIL: publication command found'
    fail=1
  fi
  if "$path" --self-test --checkpoint-sha256 "$(printf '0%.0s' {1..64})" >/dev/null 2>&1; then
    log 'self-test FAIL: extra argument accepted'
    fail=1
  fi
  if "$path" --checkpoint-sha256 0123 >/dev/null 2>&1; then
    log 'self-test FAIL: short checkpoint digest accepted'
    fail=1
  fi
  if "$path" --checkpoint-sha256 "$CHECKPOINT_SHA256" --checkpoint-sha256 "$CHECKPOINT_SHA256" >/dev/null 2>&1; then
    log 'self-test FAIL: duplicate checkpoint option accepted'
    fail=1
  fi
  if "$path" --checkpoint-sha256 "$CHECKPOINT_SHA256" >/dev/null 2>&1; then
    log 'self-test FAIL: missing approval option accepted'
    fail=1
  fi
  temporary="$(cd -P "$(mktemp -d)" && pwd)"
  trap 'rm -rf "$temporary"' RETURN
  local old_root="$VOKRA_ROOT"
  VOKRA_ROOT="$temporary/checkout"
  mkdir -p "$VOKRA_ROOT/.git" "$temporary/real/existing"
  printf '%s\n' approval > "$temporary/approval.json"
  require_absent_work_dir "$temporary/nested/new/work" "$temporary/approval.json" || { log 'self-test FAIL: nested absent work path rejected'; fail=1; }
  ln -s "$temporary/real" "$temporary/link"
  if require_absent_work_dir "$temporary/link/existing/nested/work" "$temporary/approval.json" >/dev/null 2>&1; then
    log 'self-test FAIL: intermediate symlink ancestor accepted'; fail=1
  fi
  printf '%s\n' approval > "$temporary/real/approval.json"
  if require_absent_work_dir "$temporary/new-work" "$temporary/link/approval.json" >/dev/null 2>&1; then
    log 'self-test FAIL: symlinked approval ancestor accepted'; fail=1
  fi
  ln -s "$temporary/missing" "$temporary/dangling"
  if require_absent_work_dir "$temporary/dangling/work" "$temporary/approval.json" >/dev/null 2>&1; then
    log 'self-test FAIL: dangling symlink ancestor accepted'; fail=1
  fi
  mkdir "$temporary/empty"
  if require_absent_work_dir "$temporary/empty" "$temporary/approval.json" >/dev/null 2>&1; then
    log 'self-test FAIL: existing empty work path accepted'; fail=1
  fi
  if require_absent_work_dir "$VOKRA_ROOT/nested" "$temporary/approval.json" >/dev/null 2>&1; then
    log 'self-test FAIL: checkout overlap accepted'; fail=1
  fi
  if require_absent_work_dir "$temporary/approval.json/child" "$temporary/approval.json" >/dev/null 2>&1; then
    log 'self-test FAIL: approval overlap accepted'; fail=1
  fi
  if ! require_validation_paths "$temporary/new-validation" "$temporary/approval.json"; then
    log 'self-test FAIL: disjoint absent validation paths rejected'; fail=1
  fi
  if require_validation_paths "$temporary/empty" "$temporary/approval.json" >/dev/null 2>&1; then
    log 'self-test FAIL: pre-existing validation root accepted'; fail=1
  fi
  if require_validation_paths "$temporary/link/new-validation" "$temporary/approval.json" >/dev/null 2>&1; then
    log 'self-test FAIL: symlinked validation ancestor accepted'; fail=1
  fi
  VOKRA_ROOT="$old_root"
  printf '%s\n' invalid > "$temporary/invalid-approval.json"
  printf '%s\n' '{"decision":"APPROVED","decision":"REJECTED"}' > "$temporary/duplicate-approval.json"
  printf '%s\n' '{"schema":"conv-tasnet-approval-v1","decision":"APPROVED"}' > "$temporary/tampered-approval.json"
  local approval_case approval_rc
  for approval_case in invalid duplicate tampered; do
    if "$path" --checkpoint-sha256 "$CHECKPOINT_SHA256" \
      --approval-evidence "$temporary/${approval_case}-approval.json" \
      --work-dir "$temporary/no-side-effect/$approval_case" >/dev/null 2>&1; then
      log "self-test FAIL: $approval_case approval was accepted"; fail=1
    else
      approval_rc=$?
      [[ "$approval_rc" == 2 ]] || { log "self-test FAIL: $approval_case approval returned $approval_rc, expected 2"; fail=1; }
    fi
    if [[ -e "$temporary/no-side-effect/$approval_case" || -L "$temporary/no-side-effect/$approval_case" ]]; then
      log "self-test FAIL: $approval_case approval created work"; fail=1
    fi
  done
  log_file="$temporary/parity.log"
  printf '%s\n' \
    'test converted_official_checkpoint_matches_asteroid ... ok' \
    'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s' > "$log_file"
  require_test_evidence "$log_file" || { log 'self-test FAIL: valid Cargo evidence rejected'; fail=1; }
  printf '%s\n' 'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out' >> "$log_file"
  if require_test_evidence "$log_file" >/dev/null 2>&1; then log 'self-test FAIL: duplicate result accepted'; fail=1; fi
  local command_block
  command_block="$(sed -n '/^[[:space:]]*exec scripts\/verify\/apple-silicon-conv-tasnet\.sh/,/^EOF$/p' "$path")"
  [[ "$(grep -Eoc '<APPLE_CONV_TASNET_GGUF>|<APPLE_CONV_TASNET_REFERENCE_DIR>|<APPLE_APPROVAL_EVIDENCE>|<APPLE_CONV_TASNET_EVIDENCE_DIR>' <<< "$command_block")" == 4 ]] || { log 'self-test FAIL: portable Apple command placeholders drifted'; fail=1; }
  # shellcheck disable=SC2016 # these are literal path-leak needles
  if grep -Eq '/workspace|\$VOKRA_ROOT|\$work_dir' <<< "$command_block"; then log 'self-test FAIL: VAST path leaked into Apple command'; fail=1; fi
  trap - RETURN
  rm -rf "$temporary"
  local gate_line host_line mkdir_line
  gate_line="$(grep -n 'PREFLIGHT_GATE' "$path" | tail -n 1 | cut -d: -f1)"
  host_line="$(grep -n 'uname -s' "$path" | tail -n 1 | cut -d: -f1)"
  # shellcheck disable=SC2016 # match the literal source token, not its value
  mkdir_line="$(grep -n 'mkdir -p "\$work_dir/input"' "$path" | tail -n 1 | cut -d: -f1)"
  [[ "$gate_line" =~ ^[0-9]+$ && "$host_line" =~ ^[0-9]+$ && "$mkdir_line" =~ ^[0-9]+$ && "$gate_line" -lt "$host_line" && "$gate_line" -lt "$mkdir_line" ]] || {
    log 'self-test FAIL: preflight gate is not before host or scratch work'; fail=1;
  }
  if ! UV_CACHE_DIR="$CONVTASNET_UV_CACHE_DIR" uv run --no-cache --no-project --offline --python 3.12 "$PREFLIGHT_GATE" --self-test >/dev/null; then
    log 'self-test FAIL: safe Conv-TasNet gate self-test failed'
    fail=1
  fi
  if ! UV_CACHE_DIR="$CONVTASNET_UV_CACHE_DIR" uv run --no-cache --no-project --offline --python 3.12 python "$DUMPER" --self-test >/dev/null; then
    log 'self-test FAIL: safe Asteroid dumper self-test failed'
    fail=1
  fi
  (( fail == 0 )) || return 1
  log 'self-test PASS'
}

require_test_evidence() {
  local log_file="$1" test_count named_count result_count ok_count
  test_count="$(awk '/^test / && $0 !~ /^test result:/ {count++} END {print count + 0}' "$log_file")"
  named_count="$(grep -Ecx '^test converted_official_checkpoint_matches_asteroid \.\.\. ok$' "$log_file" || true)"
  result_count="$(grep -Ecx '^test result: ok\. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out(; finished in [0-9]+\.[0-9]+s)?$' "$log_file" || true)"
  ok_count="$(grep -Ec '^test result:' "$log_file" || true)"
  [[ "$test_count" == 1 && "$named_count" == 1 && "$result_count" == 1 && "$ok_count" == 1 ]] || { die 'Cargo test evidence is not one exact named pass/result'; return 2; }
}

paths_overlap() {
  local left="$1" right="$2"
  [[ "$left" == "$right" || "$left/" == "$right/"* || "$right/" == "$left/"* ]]
}

canonical_absent_path() {
  local target="$1" current suffix component real lexical
  [[ "$target" = /* ]] || target="$PWD/$target"
  lexical="${target#/}"; current="/"
  while [[ -n "$lexical" ]]; do
    component="${lexical%%/*}"
    if [[ "$lexical" == "$component" ]]; then lexical=""; else lexical="${lexical#*/}"; fi
    [[ "$component" == "." || -z "$component" ]] && continue
    [[ "$component" != ".." ]] || { die 'work-dir path contains ..'; return 2; }
    current="${current%/}/$component"
    [[ ! -L "$current" ]] || { die 'work-dir path contains a symlinked component'; return 2; }
  done
  current="$target"; suffix=""
  while [[ ! -e "$current" && ! -L "$current" ]]; do
    component="$(basename "$current")"; suffix="/$component$suffix"; current="$(dirname "$current")"
  done
  [[ -d "$current" && ! -L "$current" ]] || { die 'work-dir parent is missing or symlinked'; return 2; }
  real="$(cd -P "$current" 2>/dev/null && pwd)" || { die 'work-dir parent is inaccessible'; return 2; }
  printf '%s%s\n' "$real" "$suffix"
}

canonical_existing_path() {
  local target="$1" lexical current="/" component parent base
  [[ "$target" = /* ]] || target="$PWD/$target"
  lexical="${target#/}"
  while [[ -n "$lexical" ]]; do
    component="${lexical%%/*}"
    if [[ "$lexical" == "$component" ]]; then lexical=""; else lexical="${lexical#*/}"; fi
    [[ "$component" == "." || -z "$component" ]] && continue
    [[ "$component" != ".." ]] || { die 'protected path contains ..'; return 2; }
    current="${current%/}/$component"
    [[ ! -L "$current" ]] || { die 'protected path contains a symlinked component'; return 2; }
  done
  [[ -e "$target" && ! -L "$target" ]] || { die 'protected path is missing or symlinked'; return 2; }
  if [[ -d "$target" ]]; then
    (cd -P "$target" && pwd) || { die 'protected directory is inaccessible'; return 2; }
  else
    parent="$(dirname "$target")"; base="$(basename "$target")"
    parent="$(cd -P "$parent" 2>/dev/null && pwd)" || { die 'protected parent is inaccessible'; return 2; }
    printf '%s/%s\n' "$parent" "$base"
  fi
}

require_absent_work_dir() {
  local work="$1" approval="$2" candidate root_real approval_real
  [[ ! -e "$work" && ! -L "$work" ]] || { die 'work-dir must be absent before validation'; return 2; }
  candidate="$(canonical_absent_path "$work")" || return 2
  root_real="$(canonical_existing_path "$VOKRA_ROOT")" || return 2
  approval_real="$(canonical_existing_path "$approval")" || return 2
  paths_overlap "$candidate" "$root_real" && { die 'work-dir overlaps checkout'; return 2; }
  paths_overlap "$candidate" "$approval_real" && { die 'work-dir overlaps approval'; return 2; }
  return 0
}

require_validation_paths() {
  local work="$1" approval="$2" work_real input_real evidence_real fixtures_real
  local root_real approval_real candidate
  # This path-only validation is called immediately after the license gate and
  # before any download, cache setup, mkdir, or evidence write.
  [[ ! -e "$work" && ! -L "$work" ]] || { die 'work-dir must be absent before validation'; return 2; }
  work_real="$(canonical_absent_path "$work")" || return 2
  input_real="$(canonical_absent_path "$work/input")" || return 2
  evidence_real="$(canonical_absent_path "$work/evidence")" || return 2
  fixtures_real="$(canonical_absent_path "$work/fixtures")" || return 2
  root_real="$(canonical_existing_path "$VOKRA_ROOT")" || return 2
  approval_real="$(canonical_existing_path "$approval")" || return 2

  for candidate in "$work_real" "$input_real" "$evidence_real" "$fixtures_real"; do
    paths_overlap "$candidate" "$root_real" && { die 'validation path overlaps checkout'; return 2; }
    paths_overlap "$candidate" "$approval_real" && { die 'validation path overlaps approval'; return 2; }
  done
  # The work root intentionally contains its three child directories.  The
  # children themselves must remain pairwise disjoint so evidence can never be
  # written into an input tree (or vice versa).
  paths_overlap "$input_real" "$evidence_real" && { die 'input and evidence paths overlap'; return 2; }
  paths_overlap "$input_real" "$fixtures_real" && { die 'input and fixture paths overlap'; return 2; }
  paths_overlap "$evidence_real" "$fixtures_real" && { die 'evidence and fixture paths overlap'; return 2; }
  return 0
}

work_dir="/workspace/vokra-conv-tasnet-validation"
checkpoint_sha256=""
approval_evidence=""
self=0
seen_checkpoint=0
seen_approval=0
seen_work=0
while (($#)); do
  case "$1" in
    --self-test) (( self == 0 )) || die 'duplicate --self-test'; self=1; shift ;;
    --checkpoint-sha256) (( seen_checkpoint == 0 )) || die 'duplicate --checkpoint-sha256'; (( $# >= 2 )) && [[ -n "$2" && "$2" != -* ]] || die '--checkpoint-sha256 requires a nonempty value'; seen_checkpoint=1; checkpoint_sha256="$2"; shift 2 ;;
    --approval-evidence) (( seen_approval == 0 )) || die 'duplicate --approval-evidence'; (( $# >= 2 )) && [[ -n "$2" && "$2" != -* ]] || die '--approval-evidence requires a nonempty value'; seen_approval=1; approval_evidence="$2"; shift 2 ;;
    --work-dir) (( seen_work == 0 )) || die 'duplicate --work-dir'; (( $# >= 2 )) && [[ -n "$2" && "$2" != -* ]] || die '--work-dir requires a nonempty path'; seen_work=1; work_dir="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done
if (( self )); then
  [[ "$seen_checkpoint" == 0 && "$seen_approval" == 0 && "$seen_work" == 0 ]] || die '--self-test accepts no other arguments'
  self_test
  exit $?
fi

[[ "$seen_approval" == 1 ]] || die '--approval-evidence is required'
[[ "$seen_checkpoint" == 1 && "$checkpoint_sha256" =~ ^[0-9a-f]{64}$ ]] || die '--checkpoint-sha256 must be exactly 64 lowercase hexadecimal characters'
[[ "$checkpoint_sha256" == "$CHECKPOINT_SHA256" ]] || die 'checkpoint digest is not the fixed authenticated identity'
UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 "$PREFLIGHT_GATE" \
  --lock "$PARITY_PROJECT/uv.lock" --project "$PARITY_PROJECT/pyproject.toml" \
  --manifest "$PREFLIGHT_MANIFEST" --evidence "$approval_evidence"
require_validation_paths "$work_dir" "$approval_evidence"
command -v sha256sum >/dev/null 2>&1 || die 'missing tool: sha256sum'
approval_evidence_sha256="$(sha256_file "$approval_evidence")"
[[ "$(uname -s)" == Linux ]] || die 'Conv-TasNet checkpoint work is VAST/Linux-only'
[[ "$(uname -m)" == x86_64 ]] || die 'VAST host must be x86_64'
[[ "${VOKRA_PUBLISH_ON_VAST:-0}" == 1 ]] || die 'VOKRA_PUBLISH_ON_VAST=1 is absent'
[[ -f "$VOKRA_ROOT/Cargo.toml" && -d "$VOKRA_ROOT/.git" ]] || die 'not a Vokra checkout'
[[ -z "$(git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all)" ]] || die 'VAST checkout must be clean'
[[ -f "$PARITY_PROJECT/pyproject.toml" && -f "$PARITY_PROJECT/uv.lock" ]] || die 'locked parity project is missing'
[[ -f "$PREPARER" && -f "$DUMPER" ]] || die 'Conv-TasNet parity tools are missing'

mem_kib="$(awk '$1 == "MemTotal:" {print $2; exit}' /proc/meminfo)"
[[ "$mem_kib" =~ ^[0-9]+$ ]] || die 'VAST memory value is invalid'
(( mem_kib >= MIN_VAST_MEM_KIB )) || die 'VAST memory guard failed'
work_parent="$work_dir"
while [[ ! -e "$work_parent" && ! -L "$work_parent" ]]; do work_parent="$(dirname "$work_parent")"; done
free_kib="$(df -Pk "$work_parent" | awk 'NR == 2 {print $4}')"
[[ "$free_kib" =~ ^[0-9]+$ ]] || die 'VAST disk value is invalid'
(( free_kib >= MIN_FREE_DISK_KIB )) || die 'VAST disk guard failed'
for tool in cargo git uv sha256sum awk find df cmp; do command -v "$tool" >/dev/null 2>&1 || die "missing tool: $tool"; done

mkdir -p "$work_dir/input" "$work_dir/evidence" "$work_dir/fixtures"
work_dir="$(cd "$work_dir" && pwd)"
export CARGO_BUILD_JOBS=1
export UV_CACHE_DIR="$CONVTASNET_UV_CACHE_DIR"
{
  printf 'upstream_repo=%s\nupstream_revision=%s\ncheckpoint=%s\ncheckpoint_sha256=%s\n' \
    "$UPSTREAM_REPO" "$UPSTREAM_REVISION" "$CHECKPOINT_FILE" "${checkpoint_sha256,,}"
  printf 'approval_evidence_sha256=%s\n' "$approval_evidence_sha256"
  echo 'runtime_status=MEASURED_NOT_GATED'
  echo 'parity_status=MEASURED_NOT_GATED'
  echo 'publication=NO_UPLOAD'
} > "$work_dir/evidence/validation.log"

cargo fmt --all -- --check >> "$work_dir/evidence/validation.log" 2>&1
cargo metadata --no-deps --format-version 1 >> "$work_dir/evidence/validation.log" 2>&1

checkpoint="$work_dir/input/$CHECKPOINT_FILE"
uv run --frozen --project "$PARITY_PROJECT" --python 3.12 python - \
  "$UPSTREAM_REPO" "$UPSTREAM_REVISION" "$CHECKPOINT_FILE" "$work_dir/input" <<'PY'
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download

repo, revision, filename, output_dir = sys.argv[1:]
path = Path(hf_hub_download(repo_id=repo, revision=revision, filename=filename, local_dir=output_dir))
if path.name != filename:
    raise SystemExit(f"unexpected checkpoint path: {path}")
PY
[[ -f "$checkpoint" && ! -L "$checkpoint" ]] || die 'downloaded checkpoint is missing, non-regular, or symlinked'
[[ "$(sha256_file "$checkpoint")" == "${checkpoint_sha256,,}" ]] || die 'checkpoint SHA-256 mismatch'

uv run --frozen --project "$PARITY_PROJECT" --python 3.12 python "$PREPARER" \
  --checkpoint "$checkpoint" --output "$work_dir/conv-tasnet.safetensors" \
  --manifest "$work_dir/evidence/prepared-manifest.json" >> "$work_dir/evidence/validation.log"

uv run --frozen --project "$PARITY_PROJECT" --python 3.12 python "$DUMPER" \
  --checkpoint "$checkpoint" --output-dir "$work_dir/fixtures" >> "$work_dir/evidence/validation.log"

fixture_dir="$VOKRA_ROOT/crates/vokra-models/tests/fixtures/conv_tasnet"
declare -A expected_hashes=(
  [pcm.f32.bin]=9afd5a533c5834708f3c10e0019b0802d354771f040fe42cef96564e30410455
  [encoder.f32.bin]=de8a252b60fb37b2232d0855a578b81c3ed3d0d5e4c97a0addf78596d7f07561
  [bottleneck.f32.bin]=54a7b62a68bd8d8f93cd5ca6860f6853ba542a97df95a2fac3583d9adeed173f
  [mask.f32.bin]=dec7369da1040f54183c80e15616ef1a3a8a91eb4fb42c74f3073bc7524bf255
  [separated.f32.bin]=c9d63ed4633c73487d7c4f4a0bbffae68478dcbd1b71f363e408d539a2d55f9b
)
for name in "${!expected_hashes[@]}"; do
  [[ -s "$work_dir/fixtures/$name" ]] || die "oracle fixture missing: $name"
  [[ "$(sha256_file "$work_dir/fixtures/$name")" == "${expected_hashes[$name]}" ]] \
    || die "fixture hash drift: $name"
  cmp "$work_dir/fixtures/$name" "$fixture_dir/$name" || die "fixture differs from committed: $name"
done
echo 'fixture hash: all committed Conv-TasNet fixtures match' | tee -a "$work_dir/evidence/validation.log"

cargo build --locked --release -p vokra-cli >> "$work_dir/evidence/validation.log" 2>&1
target/release/vokra-cli convert --model conv-tasnet-libri1mix \
  --input "$work_dir/conv-tasnet.safetensors" --output "$work_dir/conv-tasnet.gguf" \
  --license unknown >> "$work_dir/evidence/validation.log" 2>&1
[[ -s "$work_dir/conv-tasnet.gguf" ]] || die 'converter emitted no GGUF'
gguf_sha256="$(sha256_file "$work_dir/conv-tasnet.gguf")"
reference_sha256="$(sha256_file "$work_dir/fixtures/manifest.json")"
cat > "$work_dir/evidence/apple-verifier-command.sh" <<EOF
#!/usr/bin/env bash
export VOKRA_REMOTE_APPLE_SILICON=1
exec scripts/verify/apple-silicon-conv-tasnet.sh \\
  --gguf '<APPLE_CONV_TASNET_GGUF>' \\
  --gguf-sha256 '$gguf_sha256' \\
  --reference-dir '<APPLE_CONV_TASNET_REFERENCE_DIR>' \\
  --reference-sha256 '$reference_sha256' \\
  --approval-evidence '<APPLE_APPROVAL_EVIDENCE>' \\
  --evidence-dir '<APPLE_CONV_TASNET_EVIDENCE_DIR>'
EOF
chmod +x "$work_dir/evidence/apple-verifier-command.sh"
export VOKRA_CONV_TASNET_GGUF="$work_dir/conv-tasnet.gguf"
cargo test --locked -p vokra-models --test parity_conv_tasnet_real -- --nocapture \
  >> "$work_dir/evidence/validation.log" 2>&1
require_test_evidence "$work_dir/evidence/validation.log"
[[ "$(sha256_file "$approval_evidence")" == "$approval_evidence_sha256" ]] || die 'approval evidence changed during validation'
{
  printf 'approval_evidence_sha256=%s\n' "$approval_evidence_sha256"
  echo 'runtime_status=MEASURED_NOT_GATED'
  echo 'parity_status=MEASURED_NOT_GATED'
  echo 'cpu_official_gate=PASS_WITH_EXISTING_MEASURED_BOUNDS'
  echo 'verdict=NO_UPLOAD'
} | tee -a "$work_dir/evidence/validation.log"
log "VAST Conv-TasNet validation complete: evidence=$work_dir; no upload performed"
