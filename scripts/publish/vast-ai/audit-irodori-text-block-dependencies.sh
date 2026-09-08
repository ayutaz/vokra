#!/usr/bin/env bash
# VAST/Linux-only dependency, license, and native-payload audit for the
# isolated Irodori TextBlock project.  It never fetches Irodori source or
# weights, imports model code, constructs a model, invokes Cargo, or uploads.
set -euo pipefail

ROOT_DIR="${VOKRA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
PROJECT="$ROOT_DIR/tools/parity/irodori_text_block_reference"
AUDIT="$PROJECT/dependency_audit.py"
UV_GATE=(uv run --no-cache --no-project --offline --python 3.12 python)

die() { echo "irodori dependency audit: $*" >&2; exit 2; }

self_test() {
  cd "$ROOT_DIR"
  grep -Fq 'vokra-irodori-text-block-dependency-audit-v1' "$AUDIT"
  grep -Fq 'PENDING_OWNER_APPROVAL' "$AUDIT"
  grep -Fq 'publisher_license_files' "$AUDIT"
  grep -Fq 'native_payloads' "$AUDIT"
  grep -Fq 'NO_IRODORI_SOURCE_OR_MODEL_OR_CHECKPOINT_REQUESTS' "$AUDIT"
  grep -Fq 'VOKRA_VAST_AUDIT=1' "$0"
  grep -Fq -- '--validate-contract' "$0"
  grep -Fq -- '--validate-output' "$0"
  grep -Fq -- '--frozen' "$0"
  grep -Fq -- '--no-dev' "$0"
  if grep -En '(^|[;&|][[:space:]]*)git[[:space:]]+clone([[:space:]]|$)|(^|[;&|][[:space:]]*)curl([[:space:]]|$)|(^|[;&|][[:space:]]*)wget([[:space:]]|$)|(^|[;&|][[:space:]]*)snapshot_download([[:space:]]|$)|(^|[;&|][[:space:]]*)git[[:space:]]+push([[:space:]]|$)|(^|[;&|][[:space:]]*)vokra-cli[[:space:]]+convert([[:space:]]|$)|(^|[;&|][[:space:]]*)cargo([[:space:]]|$)' "$0" >/dev/null; then
    echo 'irodori dependency audit self-test: forbidden source/model/Cargo/publication marker' >&2
    return 1
  fi
  UV_NO_CACHE=1 "${UV_GATE[@]}" - <<'PY'
import json

def reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result

try:
    json.loads('{"status":"BLOCKED","status":"APPROVED"}', object_pairs_hook=reject_duplicate_keys)
except ValueError:
    pass
else:
    raise SystemExit("self-test accepted duplicate report key")
PY
  UV_NO_CACHE=1 "${UV_GATE[@]}" "$AUDIT" --self-test
  echo 'irodori dependency audit worker self-test: OK'
}

if [[ ${1:-} == --self-test ]]; then
  [[ $# == 1 ]] || die '--self-test accepts no arguments'
  self_test
  exit 0
fi

expected_head=''; output=''; seen_head=0; seen_output=0
while (($#)); do
  case "$1" in
    --expected-head)
      (( seen_head == 0 )) || die 'duplicate --expected-head'
      [[ $# -ge 2 && "$2" =~ ^[0-9a-f]{40}$ ]] || die '--expected-head requires lowercase 40-hex'
      expected_head="$2"; seen_head=1; shift 2 ;;
    --output)
      (( seen_output == 0 )) || die 'duplicate --output'
      [[ $# -ge 2 && "$2" == /* && "$2" != */ && "$2" != *'/./'* && "$2" != *'/../'* ]] || die '--output requires a canonical absolute path'
      output="$2"; seen_output=1; shift 2 ;;
    *) die "unknown argument: $1" ;;
  esac
done
(( seen_head == 1 && seen_output == 1 )) || die '--expected-head and --output are required'
[[ ${VOKRA_VAST_AUDIT:-} == 1 ]] || die 'set VOKRA_VAST_AUDIT=1 on disposable VAST'
[[ $(uname -s) == Linux && $(uname -m) == x86_64 ]] || die 'requires Linux x86_64 VAST'
[[ -d "$ROOT_DIR" && "$ROOT_DIR" == /* && "$ROOT_DIR" != */ && "$ROOT_DIR" != *'/./'* && "$ROOT_DIR" != *'/../'* ]] || die 'checkout path is not canonical'
[[ $(cd -P "$ROOT_DIR" && pwd) == "$ROOT_DIR" ]] || die 'checkout path resolves through a symlink'
[[ -f "$AUDIT" && ! -L "$AUDIT" ]] || die 'dependency audit is missing or symlinked'
[[ ! -e "$output" && ! -L "$output" ]] || die 'output must be absent'
output_parent="$(dirname "$output")"
[[ -d "$output_parent" && ! -L "$output_parent" ]] || die 'output parent must be an existing real directory'
[[ "$(cd -P "$output_parent" && pwd)" == "$output_parent" ]] || die 'output parent resolves through a symlink'
ancestor="$output_parent"
while :; do
  [[ ! -L "$ancestor" ]] || die "output ancestor is a symlink: $ancestor"
  [[ "$ancestor" == / ]] && break
  ancestor="$(dirname "$ancestor")"
done
cd "$ROOT_DIR"
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || die 'worktree is not clean'
[[ $(git rev-parse HEAD) == "$expected_head" ]] || die 'checkout HEAD does not match --expected-head'

# Validate the frozen project before any dependency-aware operation.  The
# normal route installs only this small CPU dependency closure into its uv
# environment; it has no Irodori checkout or model path.
UV_NO_CACHE=1 "${UV_GATE[@]}" "$AUDIT" --validate-contract >/dev/null || die 'Irodori dependency contract is invalid'
env -u HF_TOKEN -u HUGGINGFACE_HUB_TOKEN uv sync --frozen --project "$PROJECT" --no-dev || die 'frozen dependency sync failed'

set +e
env -u HF_TOKEN -u HUGGINGFACE_HUB_TOKEN uv run --no-cache --frozen --project "$PROJECT" --python 3.12 python "$AUDIT" --project "$PROJECT" --output "$output" --expected-head "$expected_head"
audit_status=$?
set -e
[[ $audit_status == 2 ]] || die "dependency audit returned unexpected status: $audit_status"
[[ -s "$output" ]] || die 'dependency audit report was not written'
if ! UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python - "$output" <<'PY'
import json
import pathlib
import sys

def reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result

path = pathlib.Path(sys.argv[1])
try:
    data = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicate_keys)
except (OSError, UnicodeError, TypeError, ValueError) as exc:
    raise SystemExit(f"malformed dependency audit report: {exc}")
if not isinstance(data, dict):
    raise SystemExit("malformed dependency audit report: root must be an object")
if data.get("schema") != "vokra-irodori-text-block-dependency-audit-v1":
    raise SystemExit("unexpected Irodori dependency audit schema")
if data.get("status") not in {"BLOCKED_FACTUAL_AUDIT", "BLOCKED_OWNER_REVIEW"}:
    raise SystemExit("dependency audit did not remain fail-closed")
if data.get("review") != "PENDING_OWNER_APPROVAL" or data.get("publication") != "NO_UPLOAD":
    raise SystemExit("dependency audit publication/approval gate drifted")
env = data.get("environment", {})
if not isinstance(env, dict):
    raise SystemExit("malformed dependency audit report: environment must be an object")
for key in ("irodori_source_fetched", "irodori_source_imported", "model_code_imported", "weights_acquired", "weights_imported", "weights_executed", "cargo_invoked"):
    if env.get(key) is not False:
        raise SystemExit(f"unsafe environment marker: {key}")
PY
then
  die 'dependency audit report envelope validation failed'
fi
UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python "$AUDIT" \
  --validate-output --project "$PROJECT" --output "$output" --expected-head "$expected_head" \
  >/dev/null || die 'dependency audit report schema/identity validation failed'
echo 'irodori dependency audit: report preserved; owner review required; no source/model/checkpoint/import/construction/upload attempted' >&2
exit 2
