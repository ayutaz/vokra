#!/usr/bin/env bash
# Model/source/checkpoint-free Canary dependency audit.
# This wrapper is deliberately separate from the real-weight workers: it only
# installs the dedicated frozen Python closure and records package facts.

set -euo pipefail

die() {
  echo "audit-canary-1b-dependencies: $*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Usage:
  audit-canary-1b-dependencies.sh --repo-root <checkout> \
    --expected-head <40-hex> --evidence-dir <existing-empty-dir>
  audit-canary-1b-dependencies.sh --self-test

The normal path performs only a frozen uv sync and the standard-library
dependency/native/license audit. It must exit 2 because owner review remains
pending. It never accepts checkpoint/source/model paths and never invokes
Cargo, git push, or an upload command.
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT_DEFAULT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PROJECT_DIR="$REPO_ROOT_DEFAULT/tools/parity/canary_1b_reference"
AUDITOR="$PROJECT_DIR/dependency_audit.py"
HEX40='^[0-9a-f]{40}$'
HEX64='^[0-9a-f]{64}$'
MIN_MEMORY_KIB=67108864

require_absent() {
  local path="$1"
  [[ ! -e "$path" && ! -L "$path" ]] || die "path must be absent: $path"
}

require_clean_head() {
  local root="$1" expected="$2" actual
  [[ "$expected" =~ $HEX40 ]] || die "expected HEAD must be lowercase 40-hex"
  [[ -d "$root" && ! -L "$root" && -e "$root/.git" && ! -L "$root/.git" ]] || die "repository root is missing or symlinked"
  [[ -z "$(git -C "$root" status --porcelain --untracked-files=all)" ]] || die "repository must be clean"
  actual="$(git -C "$root" rev-parse HEAD)" || die "cannot read repository HEAD"
  [[ "$actual" == "$expected" ]] || die "HEAD $actual != expected $expected"
}

run_self_test() {
  local script_path="${BASH_SOURCE[0]}" required
  for required in \
    'tools/parity/canary_1b_reference' \
    'uv sync --frozen' \
    'dependency_audit.py' \
    '--archive-dir' '--project-sha256' '--lock-sha256' '--audit-sha256' \
    'BLOCKED_UNREVIEWED_TRANSITIVE' 'NO_UPLOAD' 'audit_exit=2' \
    'git status --porcelain --untracked-files=all'; do
    grep -Fq -- "$required" "$script_path" || die "self-test contract token missing: $required"
  done
  if grep -En '^[[:space:]]*(python3?|pip)([[:space:]]|$)|^[[:space:]]*git[[:space:]]+push|publish-one\.sh|upload\.sh' "$script_path" >/dev/null; then
    die "self-test found forbidden direct Python/publish command"
  fi
  if grep -Eq 'tools/parity[[:space:]]+--(extra|project)' "$script_path"; then
    die "self-test found generic parity fallback"
  fi
  echo "audit-canary-1b-dependencies.sh self-test: PASS"
}

repo_root=""
expected_head=""
evidence_dir=""
self_test=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --self-test) (( self_test == 0 )) || die "duplicate --self-test"; self_test=1; shift ;;
    --repo-root) [[ $# -ge 2 && -n "$2" ]] || die "--repo-root requires a path"; repo_root="$2"; shift 2 ;;
    --expected-head) [[ $# -ge 2 && "$2" =~ $HEX40 ]] || die "--expected-head requires 40 lowercase hex"; expected_head="$2"; shift 2 ;;
    --evidence-dir) [[ $# -ge 2 && -n "$2" ]] || die "--evidence-dir requires a path"; evidence_dir="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

if (( self_test )); then
  [[ -z "$repo_root$expected_head$evidence_dir" ]] || die "--self-test accepts no other arguments"
  run_self_test
  exit 0
fi

[[ -n "$repo_root" && -n "$expected_head" && -n "$evidence_dir" ]] || die "normal run requires repo-root, expected-head, and evidence-dir"
[[ "$(uname -s)" == "Linux" && "$(uname -m)" == "x86_64" ]] || die "audit requires Linux x86_64"
[[ "${VOKRA_PUBLISH_ON_VAST:-0}" == "1" ]] || die "VOKRA_PUBLISH_ON_VAST=1 is required"
memory_kib="$(awk '/^MemTotal:/{print $2; exit}' /proc/meminfo)"
[[ "$memory_kib" =~ ^[0-9]+$ && "$memory_kib" -ge "$MIN_MEMORY_KIB" ]] || die "VAST RAM is below 64 GiB"
require_clean_head "$repo_root" "$expected_head"
if [[ -e "$evidence_dir" || -L "$evidence_dir" ]]; then
  [[ -d "$evidence_dir" && ! -L "$evidence_dir" ]] || die "evidence-dir is not a regular directory"
else
  evidence_parent="$(dirname "$evidence_dir")"
  [[ -d "$evidence_parent" && ! -L "$evidence_parent" ]] || die "evidence-dir parent is missing or symlinked"
  mkdir "$evidence_dir"
fi
project_file="$PROJECT_DIR/pyproject.toml"
lock_file="$PROJECT_DIR/uv.lock"
for path in "$project_file" "$lock_file" "$AUDITOR"; do
  [[ -f "$path" && ! -L "$path" ]] || die "dedicated project file is missing or symlinked: $path"
done
output="$evidence_dir/dependency-audit.json"
archive_dir="$evidence_dir/dependency-licenses"
audit_log="$evidence_dir/dependency-audit.log"
require_absent "$output"
require_absent "$archive_dir"
require_absent "$audit_log"
mkdir "$archive_dir"
project_sha256="$(sha256sum "$project_file" | awk '{print $1}')"
lock_sha256="$(sha256sum "$lock_file" | awk '{print $1}')"
audit_sha256="$(sha256sum "$AUDITOR" | awk '{print $1}')"

# This is the only environment synchronization in the wrapper. It occurs
# after all host/head/output guards and targets no project except the dedicated
# Canary reference closure.
UV_CACHE_DIR="${UV_CACHE_DIR:-/root/.cache/uv}" /root/.local/bin/uv sync \
  --project "$PROJECT_DIR" --frozen --python 3.12

set +e
UV_CACHE_DIR="${UV_CACHE_DIR:-/root/.cache/uv}" /root/.local/bin/uv run \
  --project "$PROJECT_DIR" --frozen --offline --python 3.12 python "$AUDITOR" \
  --project "$project_file" --lock "$lock_file" --repo-root "$repo_root" \
  --expected-head "$expected_head" --output "$output" --archive-dir "$archive_dir" \
  --project-sha256 "$project_sha256" --lock-sha256 "$lock_sha256" \
  --audit-sha256 "$audit_sha256" > "$audit_log" 2>&1
audit_exit=$?
set -e
[[ "$audit_exit" == 2 ]] || die "audit_exit=$audit_exit (expected 2)"
grep -Fq 'BLOCKED_UNREVIEWED_TRANSITIVE' "$output" || die "audit status is not blocked"
grep -Fq 'NO_UPLOAD' "$output" || die "audit publication is not NO_UPLOAD"
grep -Fq '"clean":true' "$output" || die "audit did not bind clean HEAD"
[[ -s "$output" && -s "$audit_log" && -f "$evidence_dir/SHA256SUMS" ]] || die "small audit evidence is incomplete"
echo "audit_exit=2"
echo "status=BLOCKED_UNREVIEWED_TRANSITIVE"
echo "publication=NO_UPLOAD"
echo "project_sha256=$project_sha256"
echo "lock_sha256=$lock_sha256"
echo "audit_sha256=$audit_sha256"
exit 2
