#!/usr/bin/env bash
# VAST/Linux-only official HT-Demucs report-only parity worker.
# It remains blocked until the dedicated lock, package primary-byte audit,
# weight terms, and dataset provenance have owner approval.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VOKRA_ROOT="${VOKRA_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
PROJECT="$VOKRA_ROOT/tools/parity/htdemucs_multi"
AUDIT="$PROJECT/audit.py"
DUMPER="$PROJECT/dump_reference.py"
HTDEMUCS_UV_CACHE_DIR="${HTDEMUCS_UV_CACHE_DIR:-/tmp/vokra-htdemucs-uv-cache}"
export UV_CACHE_DIR="$HTDEMUCS_UV_CACHE_DIR"
export PYTHONDONTWRITEBYTECODE=1
UPSTREAM_URL="https://github.com/facebookresearch/demucs.git"
UPSTREAM_REVISION="e976d93ecc3865e5757426930257e200846a520a"
WEIGHT_ROOT="https://dl.fbaipublicfiles.com/demucs/hybrid_transformer"
MIN_VAST_MEM_KIB=$((64 * 1024 * 1024))
# 32 GiB leaves room for roughly 7 GiB of five checkpoints, <=~1 GiB of
# bounded raw taps, and the official CPU reference working set on a 62 GiB
# /dev/shm host; the worker still fails closed before creating anything below.
MIN_FREE_DISK_KIB=$((32 * 1024 * 1024))
AUDIO_FIXTURE_REL="tests/fixtures/audio/jfk-30s.wav"
AUDIO_FIXTURE_SHA256="58adb4ea501d955fcd40bfbb69128f8f40428b81d8716b9ed337949773be253f"
MEMBERS=(
  f7e0c4bc-ba3fe64a.th
  d12395a8-e57c48e6.th
  92cfc3b6-ef3bcb9c.th
  04573f0d-f3cf25b2.th
  5c90dfd2-34c22ccb.th
)

log() { printf '[htdemucs-multi-parity] %s\n' "$*" >&2; }
die() { log "ERROR: $*"; return 2; }

usage() {
  cat <<'EOF'
usage: run-htdemucs-multi-parity.sh [--work-dir /dev/shm/absent-dir]
       run-htdemucs-multi-parity.sh --self-test

VAST/Linux-only, report-only official HT-Demucs parity worker. It requires the
dedicated dependency/license gate to be approved before cloning, resolving,
downloading, or executing anything. It never converts, uploads, or publishes.
EOF
}

self_test() {
  local path="${BASH_SOURCE[0]}" token fail=0
  for token in \
    'VOKRA_PUBLISH_ON_VAST=1' 'Linux' 'x86_64' 'MIN_VAST_MEM_KIB' '/dev/shm' \
    'MIN_FREE_DISK_KIB' '32 * 1024 * 1024' 'uv.lock' 'license_gate_manifest.json' 'audit.py' \
    'dump_reference.py' 'weights_only=True' '--raw-dir' \
    'PYTHONDONTWRITEBYTECODE' 'NO_UPLOAD' 'REPORT_ONLY' \
    'e976d93ecc3865e5757426930257e200846a520a' 'jfk-30s.wav' \
    '58adb4ea501d955fcd40bfbb69128f8f40428b81d8716b9ed337949773be253f' \
    'publication' 'MUSDB18' 'provenance_status'; do
    if ! grep -Fq -- "$token" "$path"; then
      log "self-test FAIL: missing contract token: $token"
      fail=1
    fi
  done
  for token in 'BagOfModels' 'apply_model' 'raw_f32' 'terminal_tap'; do
    if ! grep -Fq -- "$token" "$DUMPER"; then
      log "self-test FAIL: dumper missing contract token: $token"
      fail=1
    fi
  done
  if grep -En '(^|[[:space:]])(git[[:space:]]+push|.*upload\.sh|.*publish-one\.sh|--push|--upload)([[:space:]]|$)' "$path" >/dev/null; then
    log 'self-test FAIL: publication command found'
    fail=1
  fi
  if ! uv run --no-project --offline --python 3.12 python "$AUDIT" --self-test >/dev/null; then
    log 'self-test FAIL: audit self-test failed'
    fail=1
  fi
  if ! uv run --no-project --offline --python 3.12 python - "$path" "$PROJECT/license_gate_manifest.json" <<'PY'
import json, re, sys
from pathlib import Path
script = Path(sys.argv[1]).read_text(encoding="utf-8")
body = re.search(r"MEMBERS=\(\n(.*?)\n\)", script, re.S)
if body is None:
    raise SystemExit("MEMBERS array missing")
members = tuple(re.findall(r"^\s+(\S+\.th)\s*$", body.group(1), re.M))
gate = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
expected = tuple(row["filename"] for row in gate["weights"])
if members != expected:
    raise SystemExit(f"worker/gate member mismatch: {members!r} != {expected!r}")
PY
  then
    log 'self-test FAIL: worker/gate member set mismatch'
    fail=1
  fi
  if ! uv run --no-project --offline --python 3.12 python "$DUMPER" --self-test >/dev/null; then
    log 'self-test FAIL: dumper self-test failed'
    fail=1
  fi
  (( fail == 0 )) || return 1
  log 'self-test PASS'
}

work_dir="/dev/shm/vokra-htdemucs-multi-parity"
self=0
while (($#)); do
  case "$1" in
    --self-test) self=1; shift ;;
    --work-dir) (($# >= 2)) || die '--work-dir requires a path'; work_dir="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done
if (( self )); then
  [[ "$work_dir" == "/dev/shm/vokra-htdemucs-multi-parity" ]] || die '--self-test accepts no other arguments'
  self_test
  exit $?
fi

[[ "$(uname -s)" == Linux ]] || die 'HT-Demucs parity is VAST/Linux-only'
[[ "$(uname -m)" == x86_64 ]] || die 'VAST host must be x86_64'
[[ "${VOKRA_PUBLISH_ON_VAST:-0}" == 1 ]] || die 'VOKRA_PUBLISH_ON_VAST=1 is absent'
[[ -f "$VOKRA_ROOT/Cargo.toml" && -d "$VOKRA_ROOT/.git" ]] || die 'not a Vokra checkout'
[[ -z "$(git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all)" ]] || die 'VAST checkout must be clean'
[[ "$work_dir" == /dev/shm/* ]] || die 'work-dir must be under /dev/shm (tmpfs)'
[[ ! -e "$work_dir" && ! -L "$work_dir" ]] || die 'work-dir must be absent and non-symlink'
mem_kib="$(awk '$1 == "MemTotal:" {print $2; exit}' /proc/meminfo)"
[[ "$mem_kib" =~ ^[0-9]+$ ]] || die 'VAST memory value is invalid'
(( mem_kib >= MIN_VAST_MEM_KIB )) || die 'VAST memory guard failed'
free_kib="$(df -Pk /dev/shm | awk 'NR == 2 {print $4}')"
[[ "$free_kib" =~ ^[0-9]+$ ]] || die 'VAST /dev/shm free-space guard failed'
(( free_kib >= MIN_FREE_DISK_KIB )) || die 'VAST /dev/shm free-space guard failed'
for tool in git uv curl sha256sum awk find df; do command -v "$tool" >/dev/null 2>&1 || die "missing tool: $tool"; done

# This gate must pass before any source, checkpoint, dependency, or model work.
uv run --no-project --offline --python 3.12 python "$AUDIT" --dependency-gate >/dev/null || die 'dependency/license gate is blocked'

mkdir "$work_dir"
mkdir "$work_dir/source" "$work_dir/weights" "$work_dir/evidence" "$work_dir/raw"
git clone --filter=blob:none --no-checkout "$UPSTREAM_URL" "$work_dir/source/repo"
git -C "$work_dir/source/repo" checkout --detach "$UPSTREAM_REVISION"
uv run --no-project --offline --python 3.12 python "$AUDIT" --source-dir "$work_dir/source/repo" >/dev/null

for member in "${MEMBERS[@]}"; do
  curl --fail --location --retry 3 --silent --show-error \
    "$WEIGHT_ROOT/$member" --output "$work_dir/weights/$member"
done

for member in "${MEMBERS[@]}"; do
  expected="$(uv run --no-project --offline --python 3.12 python - "$PROJECT/license_gate_manifest.json" "$member" <<'PY'
import json, sys
from pathlib import Path
gate = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
for row in gate['weights']:
    if row['filename'] == sys.argv[2]:
        print(row['sha256'])
        break
else:
    raise SystemExit('weight row missing')
PY
)"
  actual="$(sha256sum "$work_dir/weights/$member" | awk '{print $1}')"
  [[ "$actual" == "$expected" ]] || die "weight SHA-256 mismatch: $member"
done

audio_fixture="$VOKRA_ROOT/$AUDIO_FIXTURE_REL"
[[ -f "$audio_fixture" && ! -L "$audio_fixture" ]] || die 'fixed audio fixture is missing or symlinked'
[[ "$(sha256sum "$audio_fixture" | awk '{print $1}')" == "$AUDIO_FIXTURE_SHA256" ]] || die 'audio fixture SHA-256 mismatch'

for variant in htdemucs_ft htdemucs_6s; do
  uv run --frozen --project "$PROJECT" --python 3.12 python "$DUMPER" \
    --source-dir "$work_dir/source/repo" --weights-dir "$work_dir/weights" \
    --audio-fixture "$audio_fixture" --audio-sha256 "$AUDIO_FIXTURE_SHA256" \
    --variant "$variant" --output "$work_dir/evidence/$variant.json" \
    --raw-dir "$work_dir/raw/$variant"
done
log "report-only parity evidence written under $work_dir/evidence; no upload performed"
