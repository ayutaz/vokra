#!/usr/bin/env bash
# VAST-only, model-free evidence collector for the microWakeWord LiteRT
# reference lock.  The only network-capable step is the frozen dependency sync;
# no model source, TFLite file, Cargo command, or publication command is used.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PROJECT="$ROOT/tools/parity/microwakeword-reference"
COLLECTOR="$PROJECT/audit_closure.py"
PROJECT_SHA256="2438d719428e497cc7f101429ba31fb5016e72737659d55aa0269d0824b1183d"
LOCK_SHA256="736fca6145c24984531ef11258cd64aebbb188fa8830300b09232cac0fe567f3"

die() { echo "run-microwakeword-reference-audit: $*" >&2; exit 2; }

usage() {
  cat <<'EOF'
Usage:
  run-microwakeword-reference-audit.sh --work-dir ABSENT_DIR --evidence-dir ABSENT_DIR
  run-microwakeword-reference-audit.sh --self-test

The real route is Linux x86_64 VAST-only and requires VOKRA_PUBLISH_ON_VAST=1
and a clean committed checkout.  It performs `uv sync --frozen` into the new
work directory, then records installed metadata, bounded license candidates,
RECORD identity, and native payload/readelf evidence.  It never acquires or
imports a model, executes LiteRT, runs Cargo, or uploads/publishes anything.
The report is always EVIDENCE_COLLECTED_OWNER_REVIEW_REQUIRED when collection
is complete; license classification and fixture/publication permission remain
outside this worker.
EOF
}

canonical_absent_path() {
  local path="$1" label="$2" parent candidate cursor component rest
  [[ "$path" == /* ]] || die "$label must be absolute: $path"
  rest="${path#/}"
  cursor=""
  while [[ -n "$rest" ]]; do
    component="${rest%%/*}"
    rest="${rest#*/}"
    [[ "$component" != "$rest" ]] || rest=""
    [[ -n "$component" && "$component" != "." && "$component" != ".." ]] || die "$label contains an unsafe path component"
    cursor="$cursor/$component"
    [[ ! -L "$cursor" ]] || die "$label has a symlink ancestor: $cursor"
  done
  [[ ! -e "$path" && ! -L "$path" ]] || die "$label must be absent: $path"
  parent="$(dirname "$path")"
  [[ -d "$parent" && ! -L "$parent" ]] || die "$label parent must be an existing real directory: $parent"
  candidate="$(cd -P "$parent" && printf '%s/%s' "$PWD" "$(basename "$path")")"
  [[ "$candidate" == "$path" ]] || die "$label is not a canonical path: $path -> $candidate"
  printf '%s\n' "$candidate"
}

paths_overlap() {
  [[ "$1" == "$2" || "$1" == "$2"/* || "$2" == "$1"/* ]]
}

check_checkout() {
  [[ -d "$ROOT/.git" ]] || die "Vokra checkout is missing"
  [[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] || die "clean committed checkout is required"
  git -C "$ROOT" rev-parse --verify 'HEAD^{commit}' >/dev/null 2>&1 || die "HEAD is not a committed revision"
  [[ -f "$PROJECT/pyproject.toml" && ! -L "$PROJECT/pyproject.toml" ]] || die "pyproject.toml is absent or symlinked"
  [[ -f "$PROJECT/uv.lock" && ! -L "$PROJECT/uv.lock" ]] || die "uv.lock is absent or symlinked"
  [[ -f "$COLLECTOR" && ! -L "$COLLECTOR" ]] || die "audit_closure.py is absent or symlinked"
  [[ "$(sha256sum "$PROJECT/pyproject.toml" | awk '{print $1}')" == "$PROJECT_SHA256" ]] || die "pyproject digest mismatch"
  [[ "$(sha256sum "$PROJECT/uv.lock" | awk '{print $1}')" == "$LOCK_SHA256" ]] || die "uv.lock digest mismatch"
}

self_test() {
  local required
  command -v uv >/dev/null 2>&1 || die "uv is required for self-test"
  for required in \
    "VOKRA_PUBLISH_ON_VAST=1" "Linux" "x86_64" "clean committed checkout" \
    "uv sync --frozen" "-I" "audit_closure.py" "--environment-root" "--site-packages" \
    "EVIDENCE_COLLECTED_OWNER_REVIEW_REQUIRED" "fixture/publication" \
    "NO_UPLOAD" "model" "LiteRT" "Cargo" "git"; do
    grep -Fq -- "$required" "${BASH_SOURCE[0]}" || die "worker contract missing: $required"
  done
  if grep -En '^[[:space:]]*(python|python3|pip)([[:space:]]|$)' "${BASH_SOURCE[0]}" >/dev/null; then
    die "raw Python/pip invocation found"
  fi
  if grep -En '^[[:space:]]*(git[[:space:]]+push|cargo([[:space:]]|$)|publish-one\.sh|upload\.sh|vokra-cli)' "${BASH_SOURCE[0]}" >/dev/null; then
    die "Cargo or publication command found"
  fi
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python "$COLLECTOR" --self-test
  echo "run-microwakeword-reference-audit.sh self-test: PASS (fake dist-info only)"
}

work_dir=""
evidence_dir=""
self=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --work-dir)
      [[ $# -ge 2 && -n "$2" ]] || die "--work-dir requires a path"
      work_dir="$2"
      shift 2
      ;;
    --evidence-dir)
      [[ $# -ge 2 && -n "$2" ]] || die "--evidence-dir requires a path"
      evidence_dir="$2"
      shift 2
      ;;
    --self-test)
      ((self == 0)) || die "duplicate --self-test"
      self=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *) die "unknown argument: $1" ;;
  esac
done

if (( self )); then
  [[ -z "$work_dir" && -z "$evidence_dir" ]] || die "--self-test accepts no paths"
  self_test
  exit 0
fi

[[ -n "$work_dir" && -n "$evidence_dir" ]] || { usage >&2; die "both --work-dir and --evidence-dir are required"; }
[[ "$(uname -s)" == Linux ]] || die "Linux VAST host required"
[[ "$(uname -m)" == x86_64 ]] || die "x86_64 VAST host required"
[[ "${VOKRA_PUBLISH_ON_VAST:-0}" == 1 ]] || die "VOKRA_PUBLISH_ON_VAST=1 is required"
for command in git uv sha256sum awk realpath; do command -v "$command" >/dev/null 2>&1 || die "missing tool: $command"; done

check_checkout
work_canonical="$(canonical_absent_path "$work_dir" work-dir)"
evidence_canonical="$(canonical_absent_path "$evidence_dir" evidence-dir)"
root_canonical="$(cd -P "$ROOT" && pwd)"
paths_overlap "$work_canonical" "$root_canonical" && die "work-dir overlaps checkout"
paths_overlap "$evidence_canonical" "$root_canonical" && die "evidence-dir overlaps checkout"
paths_overlap "$work_canonical" "$evidence_canonical" && die "work-dir and evidence-dir must be disjoint"

# The sync is intentionally the first operation that may access package
# indexes.  Its environment is outside the checkout and is never imported by
# this worker other than through the interpreter's stdlib startup.
mkdir "$work_canonical"
mkdir "$evidence_canonical"
UV_PROJECT_ENVIRONMENT="$work_canonical/.venv" uv sync --frozen --project "$PROJECT" --python 3.12
venv_python="$work_canonical/.venv/bin/python"
[[ -x "$venv_python" ]] || die "frozen sync did not create the expected Python environment"

environment_root_reported="$work_canonical/.venv"
environment_root="$(realpath -e -- "$environment_root_reported")" || die "environment root cannot be canonicalized"
[[ "$environment_root" == "$environment_root_reported" ]] || die "environment root has a symlink/canonical alias"
site_packages_reported="$("$venv_python" -I -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
[[ "$site_packages_reported" == /* ]] || die "stdlib sysconfig returned a non-absolute site-packages path"
site_packages="$(realpath -e -- "$site_packages_reported")" || die "site-packages path cannot be canonicalized"
[[ "$site_packages" == "$site_packages_reported" ]] || die "site-packages path has a symlink/canonical alias"
case "$site_packages/" in
  "$environment_root/"*) ;;
  *) die "site-packages is outside the synchronized environment" ;;
esac

set +e
"$venv_python" -I "$COLLECTOR" \
  --project "$PROJECT/pyproject.toml" \
  --lock "$PROJECT/uv.lock" \
  --environment-root "$environment_root" \
  --site-packages "$site_packages" \
  --output "$evidence_canonical/dependency-evidence.json"
collector_status=$?
set -e
[[ "$collector_status" == 0 ]] || die "dependency evidence collection failed; inspect the fail-closed report"
echo "microWakeWord dependency evidence: $evidence_canonical/dependency-evidence.json"
echo "publication: NO_UPLOAD; owner review remains required"
