#!/usr/bin/env bash
# VAST-only, report-only XY dependency license/native evidence collector.
# It downloads only exact uv.lock artifacts, never model/checkpoint payloads,
# and always leaves publication blocked pending external owner sign-off.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PROJECT="$ROOT/tools/parity/xy_tokenizer_reference"
AUDIT="$PROJECT/audit.py"
COLLECTOR="$PROJECT/collect_evidence.py"
LOCK_SHA256="ba26854d2cd1d695195fc906dde3d02f1fbf7ccc1d154e6015aaaa0aec44c049"

die() { echo "xy-tokenizer-dependency-audit: ERROR: $*" >&2; exit 2; }

usage() {
  cat >&2 <<'EOF'
usage: run-xy-tokenizer-dependency-audit.sh --evidence-output ABSENT_DIR
       run-xy-tokenizer-dependency-audit.sh --self-test

The worker is Linux/x86_64 VAST-only and requires VOKRA_PUBLISH_ON_VAST=1.
It downloads only the 57 exact uv.lock dependency artifacts and stores raw
license bytes plus bounded native evidence in the evidence directory. The
tracked project license_evidence.json is never modified; final audit uses an
explicit evidence override. It never downloads models,
checkpoints, or publishes/uploads anything. Owner sign-off remains required.
EOF
}

self_test() {
  command -v uv >/dev/null 2>&1 || die "uv is required"
  [[ -f "$AUDIT" && ! -L "$AUDIT" ]] || die "audit.py is missing or symlinked"
  [[ -f "$COLLECTOR" && ! -L "$COLLECTOR" ]] || die "collector is missing or symlinked"
  UV_CACHE_DIR="${XY_UV_CACHE_DIR:-/private/tmp/vokra-xy-uv-cache}" \
    uv run --offline --no-project --python 3.12 python "$AUDIT" --self-test
  UV_CACHE_DIR="${XY_UV_CACHE_DIR:-/private/tmp/vokra-xy-uv-cache}" \
    uv run --offline --no-project --python 3.12 python "$COLLECTOR" --self-test
  grep -Fq -- "$LOCK_SHA256" "$PROJECT/dependency_audit.json" \
    || die "tracked lock digest contract is missing"
  grep -Fq -- 'license-bytes' "$COLLECTOR" \
    || die "collector raw license-byte output contract is missing"
  grep -Fq -- 'native_payloads' "$COLLECTOR" \
    || die "collector native payload evidence contract is missing"
  if grep -Fq -- "mkdir \"\$evidence_output\"" "${BASH_SOURCE[0]}"; then
    die "worker must not pre-create collector output"
  fi
  grep -Fq -- '--license-evidence' "${BASH_SOURCE[0]}" \
    || die "audit evidence override contract is missing"
  grep -Fq -- 'collector_status' "${BASH_SOURCE[0]}" \
    || die "partial collection status contract is missing"
  grep -Fq -- 'final_report' "${BASH_SOURCE[0]}" \
    || die "partial final-audit contract is missing"
  [[ "$(grep -Fc -- 'uv run --offline' "${BASH_SOURCE[0]}")" -ge 5 ]] \
    || die "all local uv launches must be offline"
  grep -Fq -- 'VOKRA_PUBLISH_ON_VAST=1' "${BASH_SOURCE[0]}" \
    || die "VAST gate contract is missing"
  echo "run-xy-tokenizer-dependency-audit.sh self-test: OK (model-free/fake archive only)"
}

evidence_output=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --evidence-output) [[ $# -ge 2 ]] || die "--evidence-output requires a path"; evidence_output="$2"; shift 2 ;;
    --self-test) [[ $# == 1 ]] || die "--self-test accepts no arguments"; self_test; exit 0 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done
[[ -n "$evidence_output" ]] || { usage; exit 2; }
[[ "$evidence_output" = /* ]] || die "evidence output must be absolute"
[[ ! -L "$PROJECT" && -d "$PROJECT" ]] || die "dependency project is missing or symlinked"
[[ ! -L "$evidence_output" && ! -e "$evidence_output" ]] \
  || die "evidence output must be an absent non-symlink directory"
[[ -d "$(dirname "$evidence_output")" && ! -L "$(dirname "$evidence_output")" ]] \
  || die "evidence output parent must be an existing non-symlink directory"
[[ "$(uname -s)" == "Linux" ]] || die "dependency artifact collection is VAST/Linux-only"
[[ "$(uname -m)" == "x86_64" ]] || die "dependency artifact collection requires x86_64"
[[ "${VOKRA_PUBLISH_ON_VAST:-0}" == "1" ]] || die "VOKRA_PUBLISH_ON_VAST=1 is absent"
[[ -f "$PROJECT/uv.lock" && ! -L "$PROJECT/uv.lock" ]] || die "exact uv.lock is absent or symlinked"
[[ "$(sha256sum "$PROJECT/uv.lock" | awk '{print $1}')" == "$LOCK_SHA256" ]] \
  || die "tracked uv.lock digest mismatch"
[[ -f "$PROJECT/pyproject.toml" && ! -L "$PROJECT/pyproject.toml" ]] \
  || die "pyproject.toml is absent or symlinked"
[[ -f "$PROJECT/license_evidence.json" && ! -L "$PROJECT/license_evidence.json" ]] \
  || die "license_evidence.json is absent or symlinked"
[[ -d "$ROOT/.git" ]] || die "not a Vokra checkout"
[[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] \
  || die "VAST checkout must be clean"
command -v realpath >/dev/null 2>&1 || die "realpath is required"
project_real="$(realpath -e "$PROJECT")"
root_real="$(realpath -e "$ROOT")" || die "repository root cannot be canonicalized"
output_parent_real="$(realpath -e "$(dirname "$evidence_output")")" \
  || die "evidence output parent cannot be canonicalized"
evidence_real="$output_parent_real/$(basename "$evidence_output")"
[[ "$evidence_real" != "$project_real" && "$evidence_real" != "$project_real"/* \
  && "$project_real" != "$evidence_real"/* \
  && "$evidence_real" != "$root_real" && "$evidence_real" != "$root_real"/* \
  && "$root_real" != "$evidence_real"/* ]] \
  || die "evidence output must be disjoint from the dependency project"

# Parse the lock and current evidence before any artifact download. The report
# is intentionally BLOCKED because the tracked evidence template is empty.
preflight_parent="$(mktemp -d /tmp/vokra-xy-dependency-preflight.XXXXXX)"
cleanup() { rm -rf -- "$preflight_parent"; }
trap cleanup EXIT
  UV_CACHE_DIR="${XY_UV_CACHE_DIR:-/private/tmp/vokra-xy-uv-cache}" \
  uv run --offline --no-project --python 3.12 python "$AUDIT" \
  --project "$PROJECT" --output "$preflight_parent/report" >/dev/null

# The collector creates the absent output only after every model-free gate.
# It tries all 57 rows and exits 2 when the report is partial/BLOCKED.
set +e
UV_CACHE_DIR="${XY_UV_CACHE_DIR:-/private/tmp/vokra-xy-uv-cache}" \
  uv run --offline --no-project --python 3.12 python "$COLLECTOR" \
  --project "$PROJECT" --output "$evidence_output"
collector_status=$?
set -e
if [[ "$collector_status" -ne 0 ]]; then
  echo "XY dependency collection BLOCKED; report: $evidence_output/collection_report.json" >&2
fi

final_report="$output_parent_real/$(basename "$evidence_output")-report"
UV_CACHE_DIR="${XY_UV_CACHE_DIR:-/private/tmp/vokra-xy-uv-cache}" \
  uv run --offline --no-project --python 3.12 python "$AUDIT" \
  --project "$PROJECT" --output "$final_report" \
  --license-evidence "$evidence_output/license_evidence.json" >/dev/null
echo "XY dependency evidence collected: $evidence_output"
echo "XY audit report (BLOCKED/NO_UPLOAD pending owner sign-off): $final_report"
if [[ "$collector_status" -ne 0 ]]; then
  exit 2
fi
