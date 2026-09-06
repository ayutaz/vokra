#!/usr/bin/env bash
# VAST-only real-validation staging for VibeVoice-1.5B.
#
# This worker is deliberately no-upload: it authenticates the fixed public
# composite GGUF and feeds that already-reviewed artifact to the ignored
# native test. It never converts or publishes the upstream shards.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VOKRA_ROOT="${VOKRA_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
PARITY_PROJECT="$VOKRA_ROOT/tools/parity"
INSPECTOR="$PARITY_PROJECT/vibevoice_1_5b_inspect.py"
REFERENCE="$PARITY_PROJECT/vibevoice_1_5b_dump_reference.py"
HF_REPOSITORY="microsoft/VibeVoice-1.5B"
HF_REVISION="142f4a5dda029212cda8b118e9d99c3da27018d8"
QWEN_REPOSITORY="Qwen/Qwen2.5-1.5B"
QWEN_REVISION="8faed761d45a263340a0528343f099c05c9a4323"
SOURCE_REPOSITORY="https://github.com/microsoft/VibeVoice.git"
SOURCE_REVISION="2f9a3d79a0e51bd1cf2ab40d36884c8948e6bb9c"
TRANSFORMERS_REPOSITORY="https://github.com/huggingface/transformers.git"
TRANSFORMERS_REVISION="5f4ecf2d9f867a1255131d2461d75793c0cf1db2"
PUBLIC_REPOSITORY="vokra/vibevoice-1.5b"
PUBLIC_REVISION="dec190628f58928fc247b1205b9da2dabc58b9da"
PUBLIC_BYTES=5408160960
PUBLIC_SHA256="8ef5f259dfab0b048151ce52d27468040f72b35b6909528e6db7fbb332ccaeac"
REFERENCE_LOCK_SHA256="ba80c08b17b2d04356264b9f9d42393e9c8be66bc0cd9fda6139dc007d943909"
REFERENCE_PACKAGE_ROWS_SHA256="1ea002fe37f4ddc4df9f7535b5ae3a42661fc1eaa0a28e8ae6dbba0fa7e9649b"
REFERENCE_LICENSE_ROWS_SHA256="987a1f7204c2d7f2baa1c537ebaa06ca4bc872d2aae60f25a78393967da7bf8c"
MIN_MEM_KIB=$((128 * 1024 * 1024))
MIN_SCRATCH_KIB=$((40 * 1024 * 1024))
UV=(uv run --frozen --project "$PARITY_PROJECT" --python 3.12 python)
REFERENCE_PROJECT="$VOKRA_ROOT/tools/parity/vibevoice_1_5b_reference"
REFERENCE_UV=(uv run --frozen --project "$REFERENCE_PROJECT" --python 3.12 python)
REFERENCE_AUDIT_UV=(uv run --no-cache --no-project --offline --python 3.12 python)

log() { printf '[vibevoice-1.5b-vast] %s\n' "$*" >&2; }
die() { log "ERROR: $*"; exit 2; }
sha256_file() { sha256sum "$1" | awk '{print $1}'; }

canonical_existing_path() {
  local path="$1" rest component current=/ parent base
  [[ "$path" == /* && -e "$path" && ! -L "$path" ]] || return 1
  rest="${path#/}"
  while [[ -n "$rest" ]]; do
    component="${rest%%/*}"; [[ "$rest" == "$component" ]] && rest="" || rest="${rest#*/}"
    [[ -n "$component" && "$component" != . && "$component" != .. ]] || return 1
    current="${current%/}/$component"; [[ ! -L "$current" ]] || return 1
  done
  if [[ -d "$path" ]]; then (cd -P "$path" && pwd); else
    parent="$(dirname "$path")"; base="$(basename "$path")"
    parent="$(cd -P "$parent" && pwd)" || return 1; printf '%s/%s\n' "$parent" "$base"
  fi
}

canonical_absent_path() {
  local path="$1" target="$1" rest component current=/ suffix='' real
  [[ "$path" == /* && ! -e "$path" && ! -L "$path" ]] || return 1
  rest="${path#/}"
  while [[ -n "$rest" ]]; do
    component="${rest%%/*}"; [[ "$rest" == "$component" ]] && rest="" || rest="${rest#*/}"
    [[ -n "$component" && "$component" != . && "$component" != .. ]] || return 1
    current="${current%/}/$component"; [[ ! -L "$current" ]] || return 1
  done
  while [[ ! -e "$target" && ! -L "$target" ]]; do
    component="$(basename "$target")"; suffix="/$component$suffix"; target="$(dirname "$target")"
  done
  [[ -d "$target" && ! -L "$target" ]] || return 1
  real="$(cd -P "$target" && pwd)" || return 1; printf '%s%s\n' "$real" "$suffix"
}

require_clean_expected_head() {
  local expected="$1" actual
  [[ "$expected" =~ ^[0-9a-f]{40}$ ]] || die '--expected-head must be lowercase 40-hex'
  [[ -d "$VOKRA_ROOT/.git" && -f "$VOKRA_ROOT/Cargo.toml" ]] || die 'Vokra checkout is missing'
  [[ -z "$(git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all)" ]] || die 'clean checkout required'
  actual="$(git -C "$VOKRA_ROOT" rev-parse HEAD)" || die 'cannot read checkout HEAD'
  [[ "$actual" == "$expected" ]] || die "checkout HEAD $actual differs from --expected-head $expected"
}

require_absent_path() {
  local target="$1"
  [[ "$target" == /* ]] || die 'work directory must be absolute'
  [[ ! -e "$target" && ! -L "$target" ]] || die 'work directory must be absent and not a symlink'
  canonical_absent_path "$target" >/dev/null || die 'work path contains dot or symlink ancestry'
}

validate_approval() {
  local path="$1" expected_head="$2" supplied_sha="$3"
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python - "$path" "$expected_head" "$supplied_sha" "$HF_REPOSITORY" "$HF_REVISION" "$QWEN_REPOSITORY" "$QWEN_REVISION" "$SOURCE_REPOSITORY" "$SOURCE_REVISION" "$TRANSFORMERS_REPOSITORY" "$TRANSFORMERS_REVISION" "$PUBLIC_REPOSITORY" "$PUBLIC_REVISION" <<'PY'
import hashlib, json, pathlib, sys
path, expected_head, supplied_sha, hf_repo, hf_rev, qwen_repo, qwen_rev, source_repo, source_rev, transformers_repo, transformers_rev, public_repo, public_rev = sys.argv[1:]
raw = pathlib.Path(path).read_bytes()
if hashlib.sha256(raw).hexdigest() != supplied_sha:
    raise SystemExit("approval bytes changed")
def pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("duplicate approval key: " + key)
        result[key] = value
    return result
data = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs)
keys = {"schema", "status", "disposition", "expected_head", "hf_repository", "hf_revision", "qwen_repository", "qwen_revision", "source_repository", "source_revision", "transformers_repository", "transformers_revision", "public_repository", "public_revision", "no_upload", "scope_sha256"}
if set(data) != keys:
    raise ValueError("approval schema is not exact")
expected = {
    "schema": "vokra-vibevoice-1-5b-approval-v1", "status": "BLOCKED", "disposition": "INSPECTION_ONLY",
    "expected_head": expected_head, "hf_repository": hf_repo, "hf_revision": hf_rev,
    "qwen_repository": qwen_repo, "qwen_revision": qwen_rev, "source_repository": source_repo,
    "source_revision": source_rev, "transformers_repository": transformers_repo,
    "transformers_revision": transformers_rev, "public_repository": public_repo,
    "public_revision": public_rev, "no_upload": True,
}

if any(data[key] != value for key, value in expected.items()):
    raise ValueError("approval identity/status mismatch")
scope = {key: data[key] for key in ("disposition", "expected_head", "hf_repository", "hf_revision", "no_upload", "public_repository", "public_revision", "qwen_repository", "qwen_revision", "source_repository", "source_revision", "status", "transformers_repository", "transformers_revision")}
if data["scope_sha256"] != hashlib.sha256(json.dumps(scope, sort_keys=True, separators=(",", ":")).encode()).hexdigest():
    raise ValueError("approval scope mismatch")
PY
}

require_cargo_singleton() {
  local log_file="$1" backend="$2" named result tests
  named="$(grep -Ec '^test vibevoice_1_5b_real_cpu_matches_official_reference \.\.\. ok$' "$log_file" || true)"
  result="$(grep -Ec '^test result:' "$log_file" || true)"
  tests="$(grep -Ec '^test ' "$log_file" || true)"
  (( named == 1 && result == 1 && tests - result == 1 )) || die "$backend Cargo output is not one exact named test/result"
  grep -Eq '^test result: ok\. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out(; finished in [0-9]+(\.[0-9]+)?s)?$' "$log_file" || die "$backend Cargo result is not an exact singleton pass"
  [[ "$(grep -Fc "VIBEVOICE_${backend}_TOKENS_MEASURED exact=true" "$log_file" || true)" == 1 ]] || die "$backend token marker is not singleton"
  [[ "$(grep -Fc "VIBEVOICE_${backend}_PCM_MEASURED" "$log_file" || true)" == 1 ]] || die "$backend PCM marker is not singleton"
  [[ "$(grep -Fc "VIBEVOICE_${backend}_OFFICIAL_DIFFUSION_LATENTS_CAPTURED" "$log_file" || true)" == 1 ]] || die "$backend diffusion marker is not singleton"
}

self_test() {
  local fail=0 token
  for token in "$HF_REPOSITORY" "$HF_REVISION" "$QWEN_REPOSITORY" "$QWEN_REVISION" "$SOURCE_REPOSITORY" "$SOURCE_REVISION" "$TRANSFORMERS_REVISION" "$PUBLIC_REPOSITORY" "$PUBLIC_REVISION" "$PUBLIC_SHA256" "$REFERENCE_LOCK_SHA256" "$REFERENCE_PACKAGE_ROWS_SHA256" "$REFERENCE_LICENSE_ROWS_SHA256" "package-resolution-and-dependency-markers-v2" "vibevoice_1_5b_inspect.py" "vibevoice_1_5b_dump_reference.py" "vibevoice_1_5b_reference" "uv.lock" "--license-audit" "--no-project" "BLOCKED_UNREVIEWED_TRANSITIVE" "BLOCKED_UNVERIFIED_API_SMOKE" "GHSA-xrqw-3rrv-vx5w" "reference_environment_identity" "local_dir" "RepoFile" "RepoFolder" "AUTHENTICATED_EVIDENCE_COMPLETE" "REFERENCE_EVIDENCE_COMPLETE" "INSPECTION_ERROR" "official_pcm.f32le" "diffusion_initial_native.f32le" "NO_UPLOAD" "--expected-head" "--approval-evidence" "--approval-sha256" "--reference-packet" "--reference-packet-sha256" "vibevoice-apple-transfer-v1" "apple-packet" "--offline" "--test-threads=1" "vibevoice_1_5b_real_cpu_matches_official_reference"; do
    if ! grep -Fq -- "$token" "$0" && ! grep -Fq -- "$token" "$INSPECTOR" && ! grep -Fq -- "$token" "$REFERENCE"; then
      log "self-test missing contract token: $token"; fail=1
    fi
  done
  for token in 'uname -s' 'uname -m' 'VOKRA_PUBLISH_ON_VAST' 'git status --porcelain --untracked-files=all' 'findmnt' 'CARGO_BUILD_JOBS=1' 'vokra/vibevoice-1.5b' 'model.gguf' 'vokra-models --test parity_vibevoice_1_5b_real'; do
    grep -Fq -- "$token" "$0" || { log "self-test missing VAST gate: $token"; fail=1; }
  done
  if grep -En '(^|[;&|][[:space:]]*)git[[:space:]]+push|(^|[;&|][[:space:]]*)(curl|wget|huggingface-cli)[[:space:]]' "$0" >/dev/null; then
    log 'self-test found publication command'; fail=1
  fi
  local forbidden_cli='vokra-cli"'
  forbidden_cli+=' convert'
  local forbidden_build='cargo build'
  forbidden_build+=' --manifest-path'
  if grep -Fq "$forbidden_cli" "$0" || grep -Fq "$forbidden_build" "$0"; then
    log 'self-test found forbidden conversion path'; fail=1
  fi
  if grep -En '(^|[;&|][[:space:]]*)(python3?|pip)([[:space:]]|$)' "$0" >/dev/null; then
    log 'self-test found bare Python command'; fail=1
  fi
  local gate_line line
  gate_line="$(awk '/if ! license_audit_preflight; then/{print NR; exit}' "$0")"
  [[ "$gate_line" =~ ^[0-9]+$ ]] || { log 'self-test cannot locate license gate'; fail=1; }
  if ! awk -v gate="$gate_line" '/\$\{REFERENCE_UV\[@\]\}/ && NR <= gate {bad=1} END {exit bad}' "$0"; then
    log 'dedicated reference invocation precedes license gate'; fail=1
  fi
  for token in 'mkdir -p ' 'snapshot_download' 'git init ' 'git clone --no-tags' 'cargo test --manifest-path'; do
    line="$(awk -v gate="$gate_line" -v token="$token" 'NR > gate && index($0, token) {print NR; exit}' "$0")"
    [[ "$line" =~ ^[0-9]+$ && "$line" -gt "$gate_line" ]] || { log "pre-download/Cargo operation is not after license gate: $token"; fail=1; }
  done
  if grep -En '(^|[[:space:]])uv[[:space:]]+sync([[:space:]]|$)' "$0" >/dev/null; then
    log 'self-test found an implicit uv sync'; fail=1
  fi
  if "$0" --expected-head bad >/dev/null 2>&1 || \
    "$0" --self-test --self-test >/dev/null 2>&1 || \
    "$0" --expected-head "$(printf '0%.0s' {1..40})" --expected-head "$(printf '1%.0s' {1..40})" >/dev/null 2>&1 || \
    "$0" --approval-sha256 bad >/dev/null 2>&1 || \
    "$0" --work-dir /tmp/a --work-dir /tmp/b >/dev/null 2>&1; then
    log 'self-test accepted malformed or duplicate CLI options'; fail=1
  fi
  if ! UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python - "$0" <<'PY'
import re, sys
source = open(sys.argv[1], encoding="utf-8").read()
blocks = re.findall(r"<<'PY'\n(.*?)\nPY", source, re.S)
if not blocks:
    raise SystemExit("no inline Python blocks found")
for index, block in enumerate(blocks):
    compile(block, f"<inline-python-{index}>", "exec")
PY
  then
    log 'inline Python syntax validation failed'; fail=1
  fi
  (( fail == 0 )) && log 'self-test: OK' || return 1
}

require_host() {
  [[ "${VOKRA_PUBLISH_ON_VAST:-0}" == 1 ]] || die 'VOKRA_PUBLISH_ON_VAST=1 is required'
  [[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || die 'Linux x86_64 VAST is required'
  local memory scratch
  memory="$(awk '$1 == "MemTotal:" {print $2; exit}' /proc/meminfo)"
  [[ "$memory" =~ ^[0-9]+$ && "$memory" -ge "$MIN_MEM_KIB" ]] || die 'at least 128 GiB RAM is required'
  scratch="$(df -Pk "$WORK_PARENT" | awk 'NR == 2 {print $4}')"
  [[ "$scratch" =~ ^[0-9]+$ && "$scratch" -ge "$MIN_SCRATCH_KIB" ]] || die 'at least 40 GiB free scratch is required'
  [[ "$(findmnt -T "$WORK_PARENT" -no FSTYPE 2>/dev/null || true)" == tmpfs ]] || die 'scratch parent must be tmpfs'
}

require_tools() {
  local tool
  for tool in git uv cargo sha256sum findmnt; do command -v "$tool" >/dev/null 2>&1 || die "missing tool: $tool"; done
  [[ -f "$PARITY_PROJECT/uv.lock" && -f "$REFERENCE_PROJECT/uv.lock" && -f "$REFERENCE_PROJECT/pyproject.toml" && -f "$INSPECTOR" && -f "$REFERENCE" ]] || die 'locked parity/reference tools are missing'
  [[ -z "$(git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all)" ]] || die 'clean checkout required'
}

license_audit_preflight() {
  local audit_output audit_rc
  set +e
  audit_output="$("${REFERENCE_AUDIT_UV[@]}" "$REFERENCE" --license-audit 2>&1)"
  audit_rc=$?
  set -e
  [[ "$audit_rc" == 2 ]] || die "license audit command returned $audit_rc"
  [[ "$audit_output" == *"$REFERENCE_LOCK_SHA256"* ]] || die 'license audit lock identity is missing'
  [[ "$audit_output" == *"$REFERENCE_PACKAGE_ROWS_SHA256"* ]] || die 'license audit package-row identity is missing'
  [[ "$audit_output" == *"$REFERENCE_LICENSE_ROWS_SHA256"* ]] || die 'license audit conclusion-row identity is missing'
  [[ "$audit_output" == *"package-resolution-and-dependency-markers-v2"* ]] || die 'license audit package-row schema is missing'
  [[ "$audit_output" == *"BLOCKED_UNREVIEWED_TRANSITIVE"* ]] || die 'license audit did not remain explicitly blocked'
  log 'dependency/license gate: BLOCKED_UNREVIEWED_TRANSITIVE (no sync, acquisition, or reference execution)'
  return 1
}

usage() {
  echo "usage: run-vibevoice-1-5b-validation.sh --expected-head HEX40 --approval-evidence FILE --approval-sha256 HEX64 --reference-packet FILE --reference-packet-sha256 HEX64 [--work-dir ABSENT_DIR] | --self-test"
}

main() {
  local self=0 expected_head='' approval='' approval_sha='' reference_packet='' reference_packet_sha='' requested="/dev/shm/vokra-vibevoice-1-5b-validation"
  local seen_head=0 seen_approval=0 seen_approval_sha=0 seen_packet=0 seen_packet_sha=0 seen_work=0
  while (( $# )); do
    case "$1" in
      --self-test) (( self == 0 )) || die 'duplicate --self-test'; self=1; shift ;;
      --expected-head) (( seen_head == 0 )) || die 'duplicate --expected-head'; [[ $# -ge 2 && "$2" =~ ^[0-9a-f]{40}$ ]] || die '--expected-head requires lowercase 40-hex'; seen_head=1; expected_head="$2"; shift 2 ;;
      --approval-evidence) (( seen_approval == 0 )) || die 'duplicate --approval-evidence'; [[ $# -ge 2 && "$2" == /* && "$2" != -* ]] || die '--approval-evidence requires an absolute path'; seen_approval=1; approval="$2"; shift 2 ;;
      --approval-sha256) (( seen_approval_sha == 0 )) || die 'duplicate --approval-sha256'; [[ $# -ge 2 && "$2" =~ ^[0-9a-f]{64}$ ]] || die '--approval-sha256 requires lowercase 64-hex'; seen_approval_sha=1; approval_sha="$2"; shift 2 ;;
      --reference-packet) (( seen_packet == 0 )) || die 'duplicate --reference-packet'; [[ $# -ge 2 && "$2" == /* && "$2" != -* ]] || die '--reference-packet requires an absolute path'; seen_packet=1; reference_packet="$2"; shift 2 ;;
      --reference-packet-sha256) (( seen_packet_sha == 0 )) || die 'duplicate --reference-packet-sha256'; [[ $# -ge 2 && "$2" =~ ^[0-9a-f]{64}$ ]] || die '--reference-packet-sha256 requires lowercase 64-hex'; seen_packet_sha=1; reference_packet_sha="$2"; shift 2 ;;
      --work-dir) (( seen_work == 0 )) || die 'duplicate --work-dir'; [[ $# -ge 2 && "$2" == /* && "$2" != -* ]] || die '--work-dir requires an absolute path'; seen_work=1; requested="$2"; shift 2 ;;
      -h|--help) usage; return 0 ;;
      *) die "unknown argument: $1" ;;
    esac
  done
  if (( self )); then
    [[ "$seen_head$seen_approval$seen_approval_sha$seen_packet$seen_packet_sha$seen_work" == 000000 ]] || die '--self-test accepts no other arguments'
    self_test; return 0
  fi
  [[ "$seen_head$seen_approval$seen_approval_sha$seen_packet$seen_packet_sha" == 11111 ]] || { usage; die 'expected-head, approval, and reference packet arguments are required'; }
  [[ -f "$approval" && ! -L "$approval" && -s "$approval" ]] || die 'approval evidence is missing, empty, or symlinked'
  [[ -f "$reference_packet" && ! -L "$reference_packet" && -s "$reference_packet" ]] || die 'reference packet is missing, empty, or symlinked'
  canonical_existing_path "$approval" >/dev/null || die 'approval path has unsafe ancestry'
  canonical_existing_path "$reference_packet" >/dev/null || die 'reference packet path has unsafe ancestry'
  [[ "$(sha256_file "$approval")" == "$approval_sha" ]] || die 'approval evidence SHA-256 mismatch'
  [[ "$(sha256_file "$reference_packet")" == "$reference_packet_sha" ]] || die 'reference packet SHA-256 mismatch'
  validate_approval "$approval" "$expected_head" "$approval_sha"
  require_clean_expected_head "$expected_head"
  require_absent_path "$requested"
  local work_real approval_real packet_real
  work_real="$(canonical_absent_path "$requested")" || die 'work path cannot be canonicalized'
  approval_real="$(canonical_existing_path "$approval")"; packet_real="$(canonical_existing_path "$reference_packet")"
  [[ "$work_real" != "$approval_real" && "$work_real" != "$approval_real"/* && "$approval_real" != "$work_real"/* ]] || die 'work directory overlaps approval evidence'
  [[ "$work_real" != "$packet_real" && "$work_real" != "$packet_real"/* && "$packet_real" != "$work_real"/* ]] || die 'work directory overlaps reference packet'
  if ! license_audit_preflight; then die 'dependency/license gate is unresolved; no VibeVoice model/source acquisition is permitted'; fi
  WORK_DIR="$requested"; WORK_PARENT="$(dirname "$WORK_DIR")"
  require_host; require_tools
  require_absent_path "$WORK_DIR"
  mkdir "$WORK_DIR" || die 'work directory was concurrently claimed'
  WORK_DIR="$(cd "$WORK_DIR" && pwd)"
  export CARGO_BUILD_JOBS=1
  local cache="$WORK_DIR/cache" snapshot="$WORK_DIR/model" qwen="$WORK_DIR/qwen" source="$WORK_DIR/source" transformers="$WORK_DIR/transformers" evidence="$WORK_DIR/evidence" packet="$WORK_DIR/packet.json"
  mkdir -p "$cache" "$snapshot" "$qwen" "$evidence"
  {
    uname -a
    uv --version
    cargo --version
    git -C "$VOKRA_ROOT" rev-parse HEAD
    printf 'CARGO_BUILD_JOBS=%s\n' "$CARGO_BUILD_JOBS"
  } > "$evidence/environment.txt"
  cp -- "$reference_packet" "$packet"
  [[ "$(sha256_file "$packet")" == "$reference_packet_sha" ]] || die 'reference packet copy changed'
  "${UV[@]}" - "$HF_REPOSITORY" "$HF_REVISION" "$QWEN_REPOSITORY" "$QWEN_REVISION" "$cache" "$snapshot" "$qwen" "$WORK_DIR/model-tree.json" "$WORK_DIR/qwen-tree.json" <<'PY'
import hashlib, json, os, sys
from pathlib import Path
from huggingface_hub import HfApi, RepoFile, RepoFolder, snapshot_download

repo, rev, qrepo, qrev, cache, model, qwen, packet, qpacket = sys.argv[1:]
api = HfApi()

def packet_for(repository, revision, destination, output, patterns):
    info = api.model_info(repository, revision=revision)
    if info.sha != revision:
        raise SystemExit(f"resolved revision drift: {repository} {info.sha}")
    snapshot_download(repo_id=repository, revision=revision, cache_dir=cache, local_dir=destination, allow_patterns=patterns)
    rows = []
    selected = set(patterns)
    seen = set()
    for item in api.list_repo_tree(repository, revision=revision, recursive=True, expand=True):
        if isinstance(item, RepoFolder):
            continue
        if not isinstance(item, RepoFile) or getattr(item, "type", None) not in {None, "file"}:
            raise SystemExit(f"unsupported server member: {item}")
        lfs = getattr(item, "lfs", None)
        if isinstance(lfs, dict):
            lfs_sha = lfs.get("sha256")
            lfs_size = lfs.get("size")
        else:
            lfs_sha = getattr(lfs, "sha256", None)
            lfs_size = getattr(lfs, "size", None)
        blob = getattr(item, "blob_id", None)
        if not isinstance(item.path, str) or not item.path or "\\" in item.path or item.path.startswith("/") or ".." in item.path.split("/") or item.path in seen:
            raise SystemExit(f"unsafe/duplicate server path: {item}")
        if not isinstance(item.size, int) or isinstance(item.size, bool) or item.size < 0 or not isinstance(blob, str) or not __import__("re").fullmatch(r"[0-9a-f]{40}", blob):
            raise SystemExit(f"untyped server item: {item}")
        if lfs_sha is not None and (not isinstance(lfs_sha, str) or not __import__("re").fullmatch(r"[0-9a-f]{64}", lfs_sha) or not isinstance(lfs_size, int) or isinstance(lfs_size, bool) or lfs_size < 0 or lfs_size != item.size):
            raise SystemExit(f"invalid LFS identity: {item.path}")
        if item.path not in selected:
            if lfs_sha is not None:
                pointer = f"version https://git-lfs.github.com/spec/v1\noid sha256:{lfs_sha}\nsize {item.size}\n".encode()
                if hashlib.sha1(f"blob {len(pointer)}\0".encode() + pointer).hexdigest() != blob:
                    raise SystemExit(f"LFS pointer Git blob mismatch: {item.path}")
            seen.add(item.path)
            continue
        if lfs_sha is None:
            rows.append({"path": item.path, "type": "file", "size": item.size, "git_blob_sha1": blob, "lfs_pointer_git_blob_sha1": None, "lfs_payload_sha256": None, "lfs_payload_size": None})
        else:
            import hashlib
            pointer = f"version https://git-lfs.github.com/spec/v1\noid sha256:{lfs_sha}\nsize {item.size}\n".encode()
            pointer_git = hashlib.sha1(f"blob {len(pointer)}\0".encode() + pointer).hexdigest()
            if pointer_git != blob:
                raise SystemExit(f"LFS pointer Git blob mismatch: {item.path}")
            rows.append({"path": item.path, "type": "file", "size": item.size, "git_blob_sha1": None, "lfs_pointer_git_blob_sha1": blob, "lfs_payload_sha256": lfs_sha, "lfs_payload_size": item.size})
        seen.add(item.path)
    if not rows:
        raise SystemExit(f"empty server tree: {repository}")
    if {row["path"] for row in rows} != set(patterns):
        raise SystemExit(f"fixed selected tree drift: {repository}")
    Path(output).write_text(json.dumps({"repository": repository, "requested_revision": revision, "resolved_revision": info.sha, "walk": "recursive_file_only", "files": sorted(rows, key=lambda x: x["path"])}, sort_keys=True, indent=2) + "\n", encoding="utf-8")

packet_for(repo, rev, model, packet, [".gitattributes", "README.md", "config.json", "preprocessor_config.json", "model.safetensors.index.json", "model-00001-of-00003.safetensors", "model-00002-of-00003.safetensors", "model-00003-of-00003.safetensors"])
packet_for(qrepo, qrev, qwen, qpacket, ["LICENSE", "tokenizer_config.json", "tokenizer.json", "vocab.json", "merges.txt"])
PY
  git init "$source" >/dev/null 2>&1 || die 'source init failed'
  git -C "$source" remote add origin "$SOURCE_REPOSITORY" || die 'source origin failed'
  git -C "$source" fetch --no-tags --filter=blob:none --depth=1 origin "$SOURCE_REVISION" >/dev/null 2>&1 || die 'source fetch failed'
  git -C "$source" checkout --detach FETCH_HEAD >/dev/null 2>&1 || die 'source checkout failed'
  git clone --no-tags --filter=blob:none "$TRANSFORMERS_REPOSITORY" "$transformers" >/dev/null 2>&1 || die 'Transformers clone failed'
  git -C "$transformers" checkout --detach "$TRANSFORMERS_REVISION" >/dev/null 2>&1 || die 'Transformers checkout failed'
  set +e
  "${UV[@]}" "$INSPECTOR" --snapshot "$snapshot" --source "$source" --transformers "$transformers" --server-tree "$WORK_DIR/model-tree.json" --qwen-snapshot "$qwen" --qwen-tree "$WORK_DIR/qwen-tree.json" --output "$evidence/inspection" > "$WORK_DIR/inspection.log" 2>&1
  local inspection_rc=$?
  set -e
  [[ "$inspection_rc" == 2 ]] || die "inspector exit was $inspection_rc"
  "${UV[@]}" - "$evidence/inspection/manifest.json" <<'PY'
import json, sys
def no_dupes(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise SystemExit(f"duplicate manifest key: {key}")
        out[key] = value
    return out
manifest = json.loads(open(sys.argv[1], encoding="utf-8").read(), object_pairs_hook=no_dupes)
if manifest.get("status") != "BLOCKED" or manifest.get("evidence_stage") != "INSPECTION_ONLY" or manifest.get("publication") != "NO_UPLOAD":
    raise SystemExit("inspection manifest is not fail-closed")
if manifest.get("inspection_status") != "AUTHENTICATED_EVIDENCE_COMPLETE" or manifest.get("collection_status") != "AUTHENTICATED":
    raise SystemExit(f"inspection evidence incomplete: {manifest.get('inspection_status')}")
PY
  cp -- "$evidence/inspection/manifest.json" "$evidence/inspection-manifest.json"
  set +e
  "${REFERENCE_UV[@]}" "$REFERENCE" --source "$source" --snapshot "$snapshot" --qwen-snapshot "$qwen" --packet "$packet" --output "$evidence/reference" > "$WORK_DIR/reference.log" 2>&1
  local reference_rc=$?
  set -e
  [[ "$reference_rc" == 0 ]] || die "official reference failed with exit $reference_rc"
  "${UV[@]}" - "$evidence/reference/manifest.json" <<'PY'
import json, sys
def no_dupes(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise SystemExit(f"duplicate manifest key: {key}")
        out[key] = value
    return out
manifest = json.loads(open(sys.argv[1], encoding="utf-8").read(), object_pairs_hook=no_dupes)
if manifest.get("reference_status") != "REFERENCE_EVIDENCE_COMPLETE":
    raise SystemExit("official reference did not complete; no native test is allowed")
PY
  "${UV[@]}" - "$evidence/inspection/manifest.json" "$evidence/reference/manifest.json" "$evidence/manifest.json" <<'PY'
import json, sys
def no_dupes(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise SystemExit(f"duplicate manifest key: {key}")
        out[key] = value
    return out
inspection = json.loads(open(sys.argv[1], encoding="utf-8").read(), object_pairs_hook=no_dupes)
reference = json.loads(open(sys.argv[2], encoding="utf-8").read(), object_pairs_hook=no_dupes)
if inspection.get("inspection_status") != "AUTHENTICATED_EVIDENCE_COMPLETE" or inspection.get("collection_status") != "AUTHENTICATED":
    raise SystemExit("inspection evidence is incomplete")
if reference.get("reference_status") != "REFERENCE_EVIDENCE_COMPLETE":
    raise SystemExit("official reference evidence is incomplete")
reference["inspection_status"] = inspection["inspection_status"]
reference["collection_status"] = inspection["collection_status"]
reference["inspection_manifest_sha256"] = __import__("hashlib").sha256(open(sys.argv[1], "rb").read()).hexdigest()
open(sys.argv[3], "w", encoding="utf-8").write(json.dumps(reference, sort_keys=True, indent=2) + "\n")
PY
  for artifact in token_ids.u32le prompt_pcm.f32le prompt_latent.f32le diffusion_initial.f32le diffusion_initial_native.f32le speech_input_mask.u8 speech_masks.u8 speech_replacement_positions.u32le generated_tokens.u32le packet.json guidance-scale.txt max-generated-tokens.txt official_pcm.f32le official_diffusion_latents.f32le; do
    cp -- "$evidence/reference/$artifact" "$evidence/$artifact"
  done
  local public_dir="$WORK_DIR/public" gguf="$evidence/vibevoice-1.5b.gguf"
  mkdir -p "$public_dir"
  # This is the already-reviewed public composite artifact. Download it
  # directly; upstream conversion is intentionally not part of this worker.
  "${UV[@]}" - "$PUBLIC_REPOSITORY" "$PUBLIC_REVISION" "$public_dir" "$gguf" "$evidence/public-artifact.json" "$PUBLIC_BYTES" "$PUBLIC_SHA256" <<'PY'
import hashlib, json, shutil, sys
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

repo, rev, destination, output, evidence, expected_bytes, expected_sha = sys.argv[1:]
expected_bytes, expected_sha = int(expected_bytes), str(expected_sha)
destination, output, evidence = map(Path, (destination, output, evidence))
api = HfApi()
info = api.model_info(repo, revision=rev)
if info.sha != rev:
    raise SystemExit(f"public revision drift: {info.sha}")
snapshot_download(repo_id=repo, revision=rev, local_dir=destination)
remote = None
for item in api.list_repo_tree(repo, revision=rev, recursive=True, expand=True):
    if getattr(item, "type", None) in {"directory", "folder"}:
        continue
    if getattr(item, "path", None) == "model.gguf":
        remote = item
        break
if remote is None or getattr(remote, "type", None) != "file" or remote.size != expected_bytes:
    raise SystemExit("fixed public model.gguf identity is missing or size-drifted")
lfs = getattr(remote, "lfs", None)
lfs_sha = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
if lfs_sha != expected_sha:
    raise SystemExit("fixed public LFS identity drift")
source = destination / "model.gguf"
if source.stat().st_size != expected_bytes:
    raise SystemExit("local public model size drift")
hash_value = hashlib.sha256()
with source.open("rb") as stream:
    for block in iter(lambda: stream.read(1 << 20), b""):
        hash_value.update(block)
if hash_value.hexdigest() != expected_sha:
    raise SystemExit("local public model content drift")
shutil.copyfile(source, output)
evidence.write_text(json.dumps({
    "repository": repo, "revision": rev, "resolved_revision": info.sha,
    "path": "model.gguf", "bytes": expected_bytes,
    "sha256": hash_value.hexdigest(), "tensor_count": 1204,
    "manifest_sha256": "45cb011420fdb114c7ad61d80888663bcc861e33b7945873836aee2450eb5702",
    "lfs_sha256": lfs_sha,
}, sort_keys=True, indent=2) + "\n", encoding="utf-8")
PY
  sha256sum "$gguf" > "$evidence/artifact-sha256.txt"
  "${UV[@]}" - "$evidence/manifest.json" "$evidence/public-artifact.json" <<'PY'
import json, sys
manifest_path, artifact_path = sys.argv[1:]
manifest = json.loads(open(manifest_path, encoding="utf-8").read())
manifest["public_artifact"] = json.loads(open(artifact_path, encoding="utf-8").read())
open(manifest_path, "w", encoding="utf-8").write(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
PY
  local native_reference="$WORK_DIR/native-reference" reference_file
  [[ ! -e "$native_reference" && ! -L "$native_reference" ]] || die 'native reference path must be absent'
  mkdir "$native_reference" || die 'native reference path was concurrently claimed'
  for reference_file in manifest.json packet.json token_ids.u32le prompt_pcm.f32le prompt_latent.f32le diffusion_initial.f32le diffusion_initial_native.f32le speech_input_mask.u8 speech_masks.u8 speech_replacement_positions.u32le generated_tokens.u32le guidance-scale.txt max-generated-tokens.txt official_pcm.f32le official_diffusion_latents.f32le; do
    cp -- "$evidence/$reference_file" "$native_reference/$reference_file"
  done
  local test_selector='vibevoice_1_5b_real_cpu_matches_official_reference'
  VOKRA_VIBEVOICE_GGUF="$gguf" VOKRA_VIBEVOICE_REFERENCE_DIR="$native_reference" VOKRA_VIBEVOICE_BACKEND=cpu CARGO_BUILD_JOBS=1 CARGO_NET_OFFLINE=true \
    cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml" --offline --locked --release -p vokra-models --test parity_vibevoice_1_5b_real "$test_selector" -- --ignored --exact --nocapture --test-threads=1 \
    > "$WORK_DIR/native-cpu.log" 2>&1 || die 'native CPU real-weight validation failed'
  require_cargo_singleton "$WORK_DIR/native-cpu.log" CPU
  cp -- "$WORK_DIR/native-cpu.log" "$evidence/native-cpu.log"
  "${UV[@]}" - "$evidence/manifest.json" <<'PY'
import json, sys
path = sys.argv[1]
data = json.loads(open(path, encoding="utf-8").read())
data["cpu_status"] = "MEASURED_NOT_GATED"
data["validation_status"] = "CPU_NATIVE_REFERENCE_EXECUTED"
data["native_test"] = {"tokens_exact": True, "pcm_status": "MEASURED_NOT_GATED"}
open(path, "w", encoding="utf-8").write(json.dumps(data, sort_keys=True, indent=2) + "\n")
PY
  sha256sum "$packet" "$evidence/manifest.json" "$evidence/generated_tokens.u32le" > "$evidence/reference-sha256.txt"
  local apple_packet="$WORK_DIR/apple-packet"
  [[ ! -e "$apple_packet" && ! -L "$apple_packet" ]] || die 'Apple packet path must be absent'
  mkdir "$apple_packet" || die 'Apple packet was concurrently claimed'
  local transfer_file
  for transfer_file in manifest.json inspection-manifest.json token_ids.u32le prompt_pcm.f32le prompt_latent.f32le diffusion_initial.f32le diffusion_initial_native.f32le speech_input_mask.u8 speech_masks.u8 speech_replacement_positions.u32le generated_tokens.u32le guidance-scale.txt max-generated-tokens.txt official_pcm.f32le official_diffusion_latents.f32le packet.json public-artifact.json artifact-sha256.txt reference-sha256.txt native-cpu.log vibevoice-1.5b.gguf; do
    cp -- "$evidence/$transfer_file" "$apple_packet/$transfer_file"
  done
  "${UV[@]}" - "$apple_packet" "$expected_head" "$approval_sha" "$apple_packet/packet.json" "$apple_packet/vibevoice-1.5b.gguf" "$apple_packet/apple-transfer-manifest.json" <<'PY'
import hashlib, json, pathlib, sys
root, expected_head, approval_sha, packet, gguf, output = map(pathlib.Path, sys.argv[1:])
expected_head, approval_sha = str(expected_head), str(approval_sha)
names = [
    "manifest.json", "inspection-manifest.json", "token_ids.u32le", "prompt_pcm.f32le",
    "prompt_latent.f32le", "diffusion_initial.f32le", "diffusion_initial_native.f32le",
    "speech_input_mask.u8", "speech_masks.u8", "speech_replacement_positions.u32le",
    "generated_tokens.u32le", "guidance-scale.txt", "max-generated-tokens.txt",
    "official_pcm.f32le", "official_diffusion_latents.f32le", "packet.json",
    "public-artifact.json", "artifact-sha256.txt", "reference-sha256.txt", "native-cpu.log",
    "vibevoice-1.5b.gguf",
]
def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()
if set(pathlib.Path(root).iterdir()) - {pathlib.Path(root) / name for name in names}:
    raise SystemExit("unexpected transfer artifact")
files = []
for name in names:
    path = root / name
    if not path.is_file() or path.is_symlink():
        raise SystemExit(f"missing or symlinked transfer artifact: {name}")
    files.append({"path": name, "bytes": path.stat().st_size, "sha256": digest(path)})
files.sort(key=lambda row: row["path"])
manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
if manifest.get("status") != "BLOCKED" or manifest.get("publication") != "NO_UPLOAD":
    raise SystemExit("combined VibeVoice manifest is not fail-closed")
payload = {
    "schema": "vibevoice-apple-transfer-v1", "expected_head": expected_head,
    "approval_evidence_sha256": approval_sha, "status": "MEASURED_NOT_GATED", "publication": "NO_UPLOAD",
    "reference_manifest_sha256": next(row["sha256"] for row in files if row["path"] == "manifest.json"),
    "input_packet_sha256": next(row["sha256"] for row in files if row["path"] == "packet.json"),
    "gguf_sha256": next(row["sha256"] for row in files if row["path"] == "vibevoice-1.5b.gguf"),
    "native_cpu_log_sha256": next(row["sha256"] for row in files if row["path"] == "native-cpu.log"),
    "files": files,
}
output.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
PY
  sha256sum "$apple_packet/apple-transfer-manifest.json" > "$apple_packet/apple-transfer-manifest.sha256"
  {
    printf '%q ' scripts/verify/apple-silicon-vibevoice-1-5b.sh \
      --expected-head "$expected_head" \
      --approval-evidence '<VIBEVOICE_APPROVAL_EVIDENCE>' \
      --approval-evidence-sha256 "$approval_sha" \
      --transfer-manifest '<VIBEVOICE_BUNDLE>/apple-transfer-manifest.json' \
      --transfer-manifest-sha256 "$(sha256_file "$apple_packet/apple-transfer-manifest.json")" \
      --bundle '<VIBEVOICE_BUNDLE>' --evidence-dir '<VIBEVOICE_EVIDENCE_DIR>'
    printf '\n'
  } > "$apple_packet/apple-transfer-args.txt"
  require_clean_expected_head "$expected_head"
}

main "$@"
