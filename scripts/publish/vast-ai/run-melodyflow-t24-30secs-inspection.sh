#!/usr/bin/env bash
# VAST-only, no-upload MelodyFlow inspection preflight.
# shellcheck disable=SC1105,SC2215,SC2155
set -euo pipefail
export UV_NO_CACHE=1
if [[ "${1:-}" == --self-test ]]; then
  mktemp(){ local created; created="$(command mktemp "$@")" || return; if [[ "${1:-}" == -d ]]; then (cd -P "$created" && pwd -P); else printf '%s\n' "$created"; fi; }
fi
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"; PROJECT="$ROOT/tools/parity/melodyflow_t24_30secs"; GATE="$ROOT/tools/parity/audiocraft_safe_gate.py"
HF_REPOSITORY="facebook/melodyflow-t24-30secs"; HF_REVISION="77bcfce24371bf29a06152c72169162c6f2791de"
SOURCE_URL="https://github.com/facebook/MelodyFlow.git"
SOURCE_REVISION="9d0d223e9a63bbb8c20b9f57c5afcb4de297e6da"; README_BYTES=6560; README_SHA256="ab790ac275d6035184dabfa467be8ec8aa08a762ee3610cf43a061db45a8f0a1"; COMPRESSION_BYTES=238776630; COMPRESSION_SHA256="c075ee7c5b13d50937d1e4f197f3e940c3f3b74207857cb0e1e17891010fdc6d"; STATE_BYTES=3849817990; STATE_SHA256="e9f95857aa1e0906fb44017ca2e4e8205395599693d6e80e5c3b8b7fc16498ef"
die(){ echo "melodyflow-t24-30secs: BLOCKED: $*" >&2; exit 2; }
self_test_work_paths(){ local tmp; tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' RETURN; mkdir -p "$tmp/approval-dir" "$tmp/real-parent/existing"; printf '{}' >"$tmp/approval-dir/evidence.json"; validate_work "$tmp/a/b/c/work" "$tmp/approval-dir/evidence.json"; mkdir -p "$tmp/empty"; if (validate_work "$tmp/empty" "$tmp/approval-dir/evidence.json"); then die 'existing empty work-dir accepted'; fi; ln -s "$tmp/real-parent" "$tmp/link-parent"; if (validate_work "$tmp/link-parent/existing/nested/work" "$tmp/approval-dir/evidence.json"); then die 'existing descendant under symlink ancestor accepted'; fi; ln -s "$tmp/missing" "$tmp/dangling"; if (validate_work "$tmp/dangling" "$tmp/approval-dir/evidence.json"); then die 'dangling work symlink accepted'; fi; if (validate_work "$ROOT/child" "$tmp/approval-dir/evidence.json"); then die 'checkout descendant accepted'; fi; if (validate_work "$PROJECT/child" "$tmp/approval-dir/evidence.json"); then die 'project descendant accepted'; fi; if (validate_work "$tmp/approval-dir/work" "$tmp/approval-dir/evidence.json"); then die 'approval descendant accepted'; fi; }
log(){ echo "melodyflow-t24-30secs: $*" >&2; }
canonical_candidate(){ local value="$1" suffix='' parent; [[ "$value" = /* ]] || value="$PWD/$value"; value="${value%/}"; [[ -n "$value" ]] || die 'work-dir path is empty'; parent="$value"; while [[ "$parent" != / ]]; do [[ ! -L "$parent" ]] || die 'work-dir path contains symlink ancestor'; parent="$(dirname "$parent")"; done; while [[ ! -e "$value" && ! -L "$value" ]]; do parent="$(dirname "$value")"; suffix="/$(basename "$value")$suffix"; [[ "$parent" != "$value" ]] || die 'work-dir has no canonical parent'; value="$parent"; done; [[ -d "$value" && ! -L "$value" ]] || die 'work-dir parent is not a real directory'; (cd -P "$value" && printf '%s%s\n' "$PWD" "$suffix"); }
paths_overlap(){ local left="${1%/}" right="${2%/}"; [[ "$left" == "$right" || "$left" == "$right"/* || "$right" == "$left"/* ]]; }
validate_work(){ local work="$1" approval="$2" canonical_work approval_dir; [[ ! -e "$work" && ! -L "$work" ]] || die 'work-dir must be absent/nonexistent'; canonical_work="$(canonical_candidate "$work")"; approval_dir="$(cd -P "$(dirname "$approval")" && pwd -P)"; paths_overlap "$canonical_work" "$(canonical_candidate "$ROOT")" && die 'work-dir overlaps checkout'; paths_overlap "$canonical_work" "$(canonical_candidate "$PROJECT")" && die 'work-dir overlaps project'; paths_overlap "$canonical_work" "$approval_dir" && die 'work-dir overlaps approval'; return 0; }
verify_model_snapshot(){ local snapshot="$1"; UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python - "$snapshot" "$README_BYTES" "$README_SHA256" "$COMPRESSION_BYTES" "$COMPRESSION_SHA256" "$STATE_BYTES" "$STATE_SHA256" <<'PY'
import hashlib,sys
from pathlib import Path
r=Path(sys.argv[1]); expected=[(int(sys.argv[2]),sys.argv[3]),(int(sys.argv[4]),sys.argv[5]),(int(sys.argv[6]),sys.argv[7])]
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda: f.read(1024 * 1024), b''): h.update(chunk)
 return h.hexdigest()
expected_files={'README.md':expected[0],'compression_state_dict.bin':expected[1],'state_dict.bin':expected[2]}; seen=set()
for x in r.rglob('*'):
 rel=str(x.relative_to(r))
 if x.is_symlink() or not x.is_file() or rel not in expected_files: raise SystemExit(f'unexpected model snapshot entry: {rel}')
 size,want=expected_files[rel]
 if x.stat().st_size!=size or digest(x)!=want: raise SystemExit(f'fixed model identity mismatch: {rel}')
 seen.add(rel)
if seen != set(expected_files): raise SystemExit('fixed model snapshot file set is incomplete')
PY
}
verify_source(){ local source="$1"; [[ "$(git -C "$source" status --porcelain --untracked-files=all)" == '' ]] || die 'source checkout is dirty'; UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python - "$source" <<'PY'
import hashlib,sys
from pathlib import Path
r=Path(sys.argv[1]); expected={'LICENSE':(1088,'da6d3703ed11cbe42bd212c725957c98da23cbff1998c05fa4b3d976d1a58e93'),'LICENSE_weights':(19333,'336255dc30193e8e15d689d9481bb05673d89055718f3a96923a7ffb99adbbaf'),'models/__init__.py':(735,'79a3de6fa1f606bb058150aa0a8959b0c0d2fef84d7fc5b4b72bcd80becfa866'),'flow.py':(21274,'06d148fd8e40ada00a034c6d5189a35b5cb3b01f7ee140b6b6a94e02905a9892'),'loaders.py':(9509,'d4cdc731e145fb2c5257f8691b2330034d99b604bfdcb4da49f9fc7cc718f31b'),'melodyflow.py':(13083,'655c9d697b698b8c2dcdc560062dcbc32423b4468912bab3e5f4094b08071a43'),'requirements.txt':(375,'6766a3bb39e304094e1ca651b3acbeccf53130221afbfe933b31f1b25e2aa35')}
for name,(size,want) in expected.items():
 p=r/name
 if p.is_symlink() or not p.is_file() or p.stat().st_size!=size or hashlib.sha256(p.read_bytes()).hexdigest()!=want: raise SystemExit(f'source identity mismatch: {name}')
PY
}
self_test(){ [[ $# == 1 ]] || die '--self-test accepts no arguments'; grep -Fq "$HF_REVISION" "$0" || die 'fixed model revision missing'; grep -Eq 'python .*GATE.*--project' "$0" || die 'stdlib gate invocation missing'; grep -Fq -- '--no-project' "$0" || die 'gate must use no-project'; grep -Fq 'snapshot_download' "$0" || die 'collection stage missing'; if "$0" --approval-evidence >/dev/null 2>&1 || "$0" --approval-evidence -bad >/dev/null 2>&1 || "$0" --approval-evidence one --approval-evidence two >/dev/null 2>&1 || "$0" --self-test x >/dev/null 2>&1; then die 'invalid option accepted'; fi; UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python "$GATE" --self-test; tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT; printf '{}' >"$tmp/evidence.json"; if "$0" --approval-evidence "$tmp/evidence.json" --work-dir "$tmp/work" >/dev/null 2>&1; then die 'pending production gate unexpectedly passed'; fi; [[ ! -e "$tmp/work" && ! -L "$tmp/work" ]] || die 'blocked gate created work-dir'; echo 'run-melodyflow-t24-30secs-inspection.sh self-test: PASS'; }
check_default_work(){ local t="$(mktemp -d)"; trap 'rm -rf "$t"' RETURN; printf '{}' >"$t/evidence.json"; if AUDIOCRAFT_WORK_DIR="$t/default-work" "$0" --approval-evidence "$t/evidence.json" >/dev/null 2>&1; then die 'pending default gate unexpectedly passed'; fi; [[ ! -e "$t/default-work" && ! -L "$t/default-work" ]] || die 'blocked default gate created work-dir'; }
if [[ "${1:-}" == --self-test ]]; then self_test_work_paths; check_default_work; self_test "$@"; exit 0; fi
approval=''; work="${AUDIOCRAFT_WORK_DIR:-/tmp/vokra-audiocraft-work/melodyflow-t24-30secs}"; seen_approval=0; seen_work=0
while (($#)); do case "$1" in --approval-evidence) (( $# >= 2 )) && [[ -n "${2:-}" && "${2:-}" != -* ]] || die 'approval evidence value is required'; ((seen_approval == 0)) || die 'duplicate approval evidence'; approval="$2"; seen_approval=1; shift 2;; --work-dir) (( $# >= 2 )) && [[ -n "${2:-}" && "${2:-}" != -* ]] || die 'work-dir value is required'; ((seen_work == 0)) || die 'duplicate work-dir'; work="$2"; seen_work=1; shift 2;; *) die 'unknown or trailing argument';; esac; done
[[ "$seen_approval" == 1 && -s "$approval" && -f "$approval" && ! -L "$approval" ]] || die 'usage: --approval-evidence <external-evidence.json> [--work-dir DIR]'
UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python "$GATE" --project "$PROJECT" --approval-evidence "$approval"
validate_work "$work" "$approval"
[[ "${AUDIOCRAFT_SOURCE_URL:-}" == "$SOURCE_URL" ]] || die 'AUDIOCRAFT_SOURCE_URL must equal the fixed source origin'
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || die 'Linux x86_64 VAST required'; [[ "${VOKRA_PUBLISH_ON_VAST:-0}" == 1 ]] || die 'VOKRA_PUBLISH_ON_VAST=1 required'; mkdir -p "$work/model-snapshot" "$work/source" "$work/hf-cache"; uv sync --project "$PROJECT" --frozen --python 3.12; uv run --project "$PROJECT" --frozen --python 3.12 python -c 'import sys; from huggingface_hub import snapshot_download; snapshot_download(repo_id=sys.argv[1],revision=sys.argv[2],local_dir=sys.argv[3],cache_dir=sys.argv[4],local_dir_use_symlinks=False,allow_patterns=["README.md","compression_state_dict.bin","state_dict.bin"])' "$HF_REPOSITORY" "$HF_REVISION" "$work/model-snapshot" "$work/hf-cache"; verify_model_snapshot "$work/model-snapshot"; git -C "$work/source" init; git -C "$work/source" remote add origin "$AUDIOCRAFT_SOURCE_URL"; git -C "$work/source" fetch --depth 1 origin "$SOURCE_REVISION"; git -C "$work/source" checkout --detach FETCH_HEAD; [[ "$(git -C "$work/source" rev-parse HEAD)" == "$SOURCE_REVISION" ]] || die 'source revision mismatch'; verify_source "$work/source"; log 'INSPECTION_ONLY: authenticated model/source identities collected; no conversion or parity result is claimed'
