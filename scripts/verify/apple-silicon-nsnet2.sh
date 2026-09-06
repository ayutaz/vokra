#!/usr/bin/env bash
# Exact NSNet2 CPU/reference/Metal parity on a disposable Apple Silicon host.
# The GGUF and both WAVs must already have been produced/staged elsewhere.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VOKRA_ROOT="${VOKRA_ROOT:-$DEFAULT_ROOT}"

MIN_MEMORY_BYTES=32000000000
MIN_FREE_DISK_KIB=10000000
TEST_NAME="parity_nsnet2_gguf_smoke"
TEST_TARGET="parity_nsnet2"
GGUF_ENV="VOKRA_NSNET2_REAL_GGUF"
WAV_ENV="VOKRA_NSNET2_REAL_WAV"
REFERENCE_WAV_ENV="VOKRA_NSNET2_REFERENCE_WAV"
PARITY_SOURCE="$VOKRA_ROOT/crates/vokra-models/tests/parity_nsnet2.rs"

log() { printf '[nsnet2-apple] %s\n' "$*" >&2; }
die() { log "ERROR: $*"; return 2; }

usage() {
  cat <<'EOF' >&2
usage: apple-silicon-nsnet2.sh --gguf <corrected.gguf> \
  --gguf-sha256 <sha256> --input <input.wav> --input-sha256 <sha256> \
  --reference <reference.wav> --reference-sha256 <sha256> \
  --expected-head <exact-40-hex-git-commit> --approval-evidence <owner-approval.json> \
  --approval-sha256 <sha256> \
  --packet-manifest <manifest.json> --packet-manifest-sha256 <sha256> \
  --evidence-dir <absent-dir>
       apple-silicon-nsnet2.sh --self-test

Runs the exact existing parity_nsnet2_gguf_smoke test with the VAST-produced
corrected NSNet2 GGUF, input WAV, and independent reference WAV. It requires
VOKRA_REMOTE_APPLE_SILICON=1, Darwin arm64, at least 32 GB physical memory,
10 GB free disk, a clean checkout, and the Xcode Metal compiler.

This verifier performs no download, conversion, upload, publish, or model
mutation. --self-test is pure offline and performs no Cargo invocation.
EOF
}

sha256_file() {
  shasum -a 256 "$1" | awk '{print $1}'
}

require_file() {
  local label="$1" path="$2"
  [[ -f "$path" && ! -L "$path" && -s "$path" ]] || die "$label is missing, empty, symlinked, or non-regular: $path"
}

canonical_absent_path() {
  local path="$1" suffix='' rest component scan name parent
  [[ "$path" == /* ]] || path="$PWD/$path"; rest="${path#/}"; scan=''
  while [[ -n "$rest" ]]; do component="${rest%%/*}"; rest="${rest#*/}"; [[ "$component" == "$rest" ]] && rest=''; [[ -n "$component" ]] || continue; [[ "$component" != . && "$component" != .. ]] || return 1; scan="$scan/$component"; [[ ! -L "$scan" || "$scan" == "/var" ]] || return 1; done
  while [[ ! -d "$path" || -L "$path" ]]; do name="${path##*/}"; [[ -n "$name" ]] && suffix="/$name$suffix"; parent="${path%/*}"; [[ "$parent" == "$path" ]] && parent=/; path="$parent"; done
  (cd -P "$path" && printf '%s%s\n' "$PWD" "$suffix")
}
paths_overlap() { [[ "$1" == "$2" || "$1" == "$2"/* || "$2" == "$1"/* ]]; }
require_absent_evidence_dir() {
  local directory="$1" candidate protected other
  shift
  [[ ! -e "$directory" && ! -L "$directory" ]] || { die "evidence directory must be absent and non-symlink: $directory"; return 2; }
  candidate="$(canonical_absent_path "$directory")" || { die 'evidence directory has a symlinked ancestor'; return 2; }
  for protected in "$VOKRA_ROOT" "$@"; do
    [[ -e "$protected" && ! -L "$protected" ]] || { die "protected path is missing or symlinked"; return 2; }
    other="$(canonical_absent_path "$protected")" || { die 'protected path cannot be canonicalized'; return 2; }
    paths_overlap "$candidate" "$other" && { die 'evidence overlaps protected path'; return 2; }
  done
  return 0
}

require_cargo_result() {
  local file="$1" named tests results
  named="$(grep -Ec "^test $TEST_NAME \.\.\. ok$" "$file" || true)"
  tests="$(grep -Ec '^test [^ ]+ \.\.\.' "$file" || true)"
  results="$(grep -Ec '^test result:' "$file" || true)"
  [[ "$named" == 1 && "$tests" == 1 && "$results" == 1 ]] || die 'Cargo evidence has duplicate/missing test or result lines'
  grep -Eq '^test result: ok\. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out(; finished in [0-9]+\.[0-9]+s)?$' "$file" || die 'Cargo result is not the exact one-pass result'
}

require_metric_sentinel() {
  local file="$1" label="$2"
  [[ "$(grep -Ec "^NSNet2 real ${label} PCM max_abs=[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?$" "$file" || true)" == 1 ]] || die "NSNet2 ${label} metric sentinel is missing, malformed, or duplicated"
}

license_preflight() {
  local approval="$1" expected_head="$2" approval_sha="$3" project="$VOKRA_ROOT/tools/parity/pyproject.toml" lock="$VOKRA_ROOT/tools/parity/uv.lock" project_sha lock_sha
  [[ -f "$project" && ! -L "$project" && -f "$lock" && ! -L "$lock" ]] || die 'locked parity project is missing or symlinked'
  [[ -f "$approval" && ! -L "$approval" && -s "$approval" ]] || die 'approval evidence must be a nonempty regular non-symlink file'
  project_sha="$(shasum -a 256 "$project" | awk '{print $1}')"; lock_sha="$(shasum -a 256 "$lock" | awk '{print $1}')"
  [[ "$(sha256_file "$approval")" == "$approval_sha" ]] || die 'approval evidence SHA-256 changed before preflight'
  if UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python - "$approval" "$project_sha" "$lock_sha" "$expected_head" "$approval_sha" <<'PY'
import hashlib, json, pathlib, sys
def hook(pairs):
    result = {}
    for key, value in pairs:
        if key in result: raise ValueError('duplicate JSON key: ' + key)
        result[key] = value
    return result
try:
    raw=pathlib.Path(sys.argv[1]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != sys.argv[5]: raise ValueError('approval SHA-256 mismatch')
    d=json.loads(raw.decode('utf-8'), object_pairs_hook=hook)
    keys={'schema','model','upstream_repo','upstream_revision','license_spdx','project_sha256','lock_sha256','expected_head','no_upload','decision','signer','scope_sha256'}
    if set(d)!=keys: raise ValueError('approval schema is not exact')
    if (d['schema'],d['model'],d['upstream_repo'],d['upstream_revision'],d['license_spdx']) != ('vokra-validation-approval-v1','nsnet2','microsoft/DNS-Challenge','8b87a33b2892f147b5c7ad39ea978453730db269','cc-by-4.0'): raise ValueError('approval identity mismatch')
    if d['project_sha256']!=sys.argv[2] or d['lock_sha256']!=sys.argv[3] or d['expected_head']!=sys.argv[4] or d['no_upload'] is not True or d['decision']!='APPROVED': raise ValueError('approval facts mismatch')
    if not isinstance(d['signer'],str) or not d['signer'].strip() or d['signer'].strip().upper() in {'TBD','TODO','UNKNOWN','PENDING','UNRESOLVED','OWNER_SIGNOFF_REQUIRED'}: raise ValueError('approval signer unresolved')
    scope={'expected_head':sys.argv[4],'license_spdx':d['license_spdx'],'lock_sha256':sys.argv[3],'model':d['model'],'no_upload':True,'project_sha256':sys.argv[2],'upstream_repo':d['upstream_repo'],'upstream_revision':d['upstream_revision']}
    if d['scope_sha256'] != hashlib.sha256(json.dumps(scope,sort_keys=True,separators=(',',':')).encode()).hexdigest(): raise ValueError('approval scope digest mismatch')
except (OSError,TypeError,ValueError,json.JSONDecodeError) as exc: raise SystemExit('approval gate BLOCKED: '+str(exc))
PY
  then :; else die 'approval evidence is invalid or offline Python is unavailable'; fi
  if UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python "$VOKRA_ROOT/scripts/publish/signoff_match.py" --check-repo nsnet2 --audit "$VOKRA_ROOT/docs/license-audit.md"
  then :; else die 'repository signoff is unresolved'; fi
}

validate_transfer_manifest() {
  local manifest="$1" manifest_sha="$2" expected_head="$3" gguf_sha="$4" input_sha="$5" reference_sha="$6" approval_sha="$7"
  require_file 'NSNet2 transfer manifest' "$manifest"
  [[ "$manifest_sha" =~ ^[0-9a-f]{64}$ ]] || die 'packet manifest SHA-256 must be lowercase 64-hex'
  [[ "$(sha256_file "$manifest")" == "$manifest_sha" ]] || die 'packet manifest SHA-256 mismatch'
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python - \
    "$manifest" "$manifest_sha" "$expected_head" "$gguf_sha" "$input_sha" "$reference_sha" "$approval_sha" <<'PY'
import hashlib, json, pathlib, sys
manifest_path = pathlib.Path(sys.argv[1])
expected_manifest_sha, expected_head, expected_gguf, expected_input, expected_reference, expected_approval = sys.argv[2:]
raw = manifest_path.read_bytes()
if hashlib.sha256(raw).hexdigest() != expected_manifest_sha: raise SystemExit('manifest changed during validation')
def hook(pairs):
    out = {}
    for key, value in pairs:
        if key in out: raise ValueError('duplicate JSON key: ' + key)
        out[key] = value
    return out
try:
    data = json.loads(raw.decode('utf-8'), object_pairs_hook=hook)
except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
    raise SystemExit('invalid transfer manifest: ' + str(exc))
if set(data) != {'schema','expected_head','git_commit','license_spdx','no_upload','status','artifacts'}: raise SystemExit('transfer manifest schema is not exact')
if (data['schema'], data['expected_head'], data['git_commit'], data['license_spdx'], data['no_upload'], data['status']) != ('vokra-nsnet2-transfer-v1', expected_head, expected_head, 'cc-by-4.0', True, 'CPU_PASS_METAL_NOT_RUN'): raise SystemExit('transfer manifest identity mismatch')
expected = {'gguf': ('nsnet2.gguf', expected_gguf), 'input_wav': ('input.wav', expected_input), 'reference_wav': ('reference.wav', expected_reference), 'approval': ('approval.json', expected_approval), 'cpu_parity': ('parity.log', None)}
artifacts = data['artifacts']
if not isinstance(artifacts, dict) or set(artifacts) != set(expected): raise SystemExit('transfer artifact set is not exact')
for key, (name, digest) in expected.items():
    row = artifacts[key]
    required = {'filename','sha256'} if key != 'cpu_parity' else {'filename','sha256','marker'}
    if set(row) != required or row['filename'] != name or not isinstance(row['sha256'], str) or len(row['sha256']) != 64 or any(ch not in '0123456789abcdef' for ch in row['sha256']): raise SystemExit('transfer artifact row mismatch')
    if digest is not None and row['sha256'] != digest: raise SystemExit('transfer artifact digest mismatch')
    path = manifest_path.parent / name
    if not path.is_file() or path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']: raise SystemExit('transfer packet file mismatch: ' + name)
    if key == 'cpu_parity' and row['marker'] != 'NSNet2_PARITY cpu_reference=PASS': raise SystemExit('CPU marker mismatch')
if (manifest_path.parent / 'manifest.sha256').read_text(encoding='utf-8').strip() != f'{expected_manifest_sha}  manifest.json': raise SystemExit('manifest sidecar mismatch')
names = {p.name for p in manifest_path.parent.iterdir()}
if names != {'manifest.json','manifest.sha256','nsnet2.gguf','input.wav','reference.wav','approval.json','parity.log'}: raise SystemExit('transfer packet closure is not exact')
if any(not p.is_file() or p.is_symlink() for p in manifest_path.parent.iterdir()): raise SystemExit('transfer packet contains non-regular entry')
PY
}

require_remote_apple_host() {
  local memory_bytes free_disk_kib
  [[ "${VOKRA_REMOTE_APPLE_SILICON:-0}" == "1" ]] \
    || die "VOKRA_REMOTE_APPLE_SILICON=1 is absent; refusing possible maintainer-Mac execution"
  [[ "$(uname -s)" == "Darwin" ]] || die "real Metal parity requires Darwin"
  [[ "$(uname -m)" == "arm64" ]] || die "real Metal parity requires Apple arm64"
  memory_bytes="$(sysctl -n hw.memsize)"
  [[ "$memory_bytes" =~ ^[0-9]+$ ]] || die "could not read hw.memsize"
  (( memory_bytes >= MIN_MEMORY_BYTES )) \
    || die "physical memory $memory_bytes bytes is below the 32-GB guard"
  free_disk_kib="$(df -Pk "$VOKRA_ROOT" | awk 'NR == 2 {print $4}')"
  [[ "$free_disk_kib" =~ ^[0-9]+$ ]] || die "could not read free disk"
  (( free_disk_kib >= MIN_FREE_DISK_KIB )) \
    || die "free disk $free_disk_kib KiB is below the 10-GB guard"
}

require_tooling() {
  local tool
  for tool in cargo rustc git shasum awk find tee grep sysctl sw_vers uv \
    system_profiler xcrun; do
    command -v "$tool" >/dev/null 2>&1 || die "required tool missing: $tool"
  done
  [[ -d "$VOKRA_ROOT/.git" && -f "$VOKRA_ROOT/Cargo.toml" ]] \
    || die "$VOKRA_ROOT is not a Vokra git checkout"
  [[ -f "$PARITY_SOURCE" ]] || die "NSNet2 parity source is missing: $PARITY_SOURCE"
  grep -Fq 'env::var(REFERENCE_WAV_ENV).unwrap_or_else' "$PARITY_SOURCE" \
    || die "NSNet2 parity source lacks the reference-WAV leg"
  grep -Fq 'NSNet2 real CPU/Metal PCM max_abs=' "$PARITY_SOURCE" \
    || die "NSNet2 parity source lacks the CPU/Metal sentinel"
  [[ -z "$(git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all)" ]] \
    || die "remote Apple checkout must be clean so evidence names one exact commit"
  xcrun -f metal >/dev/null 2>&1 || die "Xcode Metal compiler is unavailable"
}

record_environment() {
  local output="$1"
  {
    echo "utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "git_commit=$(git -C "$VOKRA_ROOT" rev-parse HEAD)"
    echo "git_branch=$(git -C "$VOKRA_ROOT" branch --show-current)"
    echo "uname=$(uname -a)"
    echo "memory_bytes=$(sysctl -n hw.memsize)"
    echo "physical_cpu=$(sysctl -n hw.physicalcpu)"
    echo "logical_cpu=$(sysctl -n hw.logicalcpu)"
    sw_vers
    rustc --version --verbose
    cargo --version
    echo "metal_compiler=$(xcrun -f metal)"
    system_profiler SPDisplaysDataType
  } > "$output"
}

# shellcheck disable=SC2016
run_self_test() (
  local script_path="${BASH_SOURCE[0]}" temporary fail=0 required
  temporary="$(mktemp -d "${TMPDIR:-/tmp}/vokra-nsnet2-apple.XXXXXX")"
  trap 'rm -rf -- "$temporary"' EXIT
  printf 'abc' > "$temporary/value"
  [[ "$(sha256_file "$temporary/value")" == \
    "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad" ]] \
    || die "SHA-256 helper self-test failed"
  for required in \
    'VOKRA_REMOTE_APPLE_SILICON' 'Darwin' 'arm64' 'MIN_MEMORY_BYTES=32000000000' \
    'MIN_FREE_DISK_KIB=10000000' 'xcrun -f metal' \
    'VOKRA_NSNET2_REAL_GGUF' 'VOKRA_NSNET2_REAL_WAV' \
    'VOKRA_NSNET2_REFERENCE_WAV' 'parity_nsnet2_gguf_smoke' \
    '--gguf-sha256' '--input-sha256' '--reference-sha256' '--expected-head' '--approval-sha256' '--packet-manifest' '--packet-manifest-sha256' \
    'expected_head' 'CARGO_NET_OFFLINE=true' \
    '--features metal --test parity_nsnet2' '-- --exact --nocapture' \
    'NSNet2 real CPU/Metal PCM max_abs=' 'NSNet2 real CPU/reference PCM max_abs=' \
    'NSNet2 real Metal/reference PCM max_abs=' 'NSNet2_PARITY cpu_reference=PASS' \
    'NSNet2_PARITY metal_vs_reference=PASS' 'NSNet2_PARITY metal_vs_cpu=PASS' \
    '--ignored --exact --nocapture --test-threads=1' \
    'git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all' \
    'cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml"'; do
    if ! grep -Fq -- "$required" "$script_path"; then
      log "self-test FAIL: contract token missing: $required"
      fail=1
    fi
  done
  if grep -En '^[[:space:]]*(curl|wget|python3?|pip|.*convert|git[[:space:]]+(clone|fetch|pull)|.*(upload|publish))([[:space:]]|$)' \
    "$script_path" >/dev/null; then
    log "self-test FAIL: forbidden acquisition/conversion/publication command found"
    fail=1
  fi
  if "$script_path" --self-test --gguf "$temporary/model.gguf" >/dev/null 2>&1; then
    log "self-test FAIL: extra --self-test argument accepted"
    fail=1
  fi
  if "$script_path" --gguf >/dev/null 2>&1; then
    log "self-test FAIL: missing --gguf value accepted"
    fail=1
  fi
  if "$script_path" --unknown-flag >/dev/null 2>&1; then
    log "self-test FAIL: unknown argument accepted"
    fail=1
  fi
  if "$script_path" --approval-sha256 bad >/dev/null 2>&1 || \
    "$script_path" --approval-sha256 "$(printf '0%.0s' {1..64})" --approval-sha256 "$(printf '1%.0s' {1..64})" >/dev/null 2>&1; then
    log "self-test FAIL: malformed or duplicate --approval-sha256 accepted"
    fail=1
  fi
  if "$script_path" --gguf -bad >/dev/null 2>&1 || "$script_path" --gguf a --gguf b >/dev/null 2>&1 || "$script_path" --gguf-sha256 bad >/dev/null 2>&1 || "$script_path" --packet-manifest-sha256 bad >/dev/null 2>&1 || "$script_path" --packet-manifest a --packet-manifest b >/dev/null 2>&1 || "$script_path" --expected-head bad >/dev/null 2>&1 || "$script_path" --expected-head 0000000000000000000000000000000000000000 --expected-head 1111111111111111111111111111111111111111 >/dev/null 2>&1 || "$script_path" --approval-evidence >/dev/null 2>&1 || "$script_path" --self-test --approval-evidence x >/dev/null 2>&1; then
    log "self-test FAIL: malformed or duplicate options accepted"
    fail=1
  fi
  require_absent_evidence_dir "$temporary/new/nested/evidence" "$temporary/value" || { log 'self-test FAIL: nested absent evidence rejected'; fail=1; }
  if require_absent_evidence_dir "$temporary/new/../dotdot-evidence" "$temporary/value" >/dev/null 2>&1; then log 'self-test FAIL: dot-dot evidence path accepted'; fail=1; fi
  mkdir "$temporary/empty-evidence"
  if require_absent_evidence_dir "$temporary/empty-evidence" "$temporary/value" >/dev/null 2>&1; then log 'self-test FAIL: existing empty evidence accepted'; fail=1; fi
  ln -s "$temporary/missing-evidence" "$temporary/dangling-evidence"
  if require_absent_evidence_dir "$temporary/dangling-evidence" "$temporary/value" >/dev/null 2>&1; then log 'self-test FAIL: dangling evidence symlink accepted'; fail=1; fi
  local packet="$temporary/packet" head approval_digest gguf_digest input_digest reference_digest manifest_digest
  mkdir "$packet"
  printf 'gguf' > "$packet/nsnet2.gguf"
  printf 'input' > "$packet/input.wav"
  printf 'reference' > "$packet/reference.wav"
  printf 'approval' > "$packet/approval.json"
  printf 'NSNet2_PARITY cpu_reference=PASS\n' > "$packet/parity.log"
  head='0000000000000000000000000000000000000000'
  approval_digest="$(sha256_file "$packet/approval.json")"
  gguf_digest="$(sha256_file "$packet/nsnet2.gguf")"
  input_digest="$(sha256_file "$packet/input.wav")"
  reference_digest="$(sha256_file "$packet/reference.wav")"
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python - "$packet" "$head" "$gguf_digest" "$input_digest" "$reference_digest" "$approval_digest" <<'PY'
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
head, gguf, input_digest, reference, approval = sys.argv[2:]
names = {'gguf': 'nsnet2.gguf', 'input_wav': 'input.wav', 'reference_wav': 'reference.wav', 'approval': 'approval.json', 'cpu_parity': 'parity.log'}
rows = {key: {'filename': name, 'sha256': hashlib.sha256((root / name).read_bytes()).hexdigest()} for key, name in names.items()}
rows['cpu_parity']['marker'] = 'NSNet2_PARITY cpu_reference=PASS'
data = {'schema': 'vokra-nsnet2-transfer-v1', 'expected_head': head, 'git_commit': head, 'license_spdx': 'cc-by-4.0', 'no_upload': True, 'status': 'CPU_PASS_METAL_NOT_RUN', 'artifacts': rows}
(root / 'manifest.json').write_text(json.dumps(data, sort_keys=True, separators=(',', ':')) + '\n', encoding='utf-8')
(root / 'manifest.sha256').write_text(hashlib.sha256((root / 'manifest.json').read_bytes()).hexdigest() + '  manifest.json\n', encoding='ascii')
PY
  manifest_digest="$(sha256_file "$packet/manifest.json")"
  validate_transfer_manifest "$packet/manifest.json" "$manifest_digest" "$head" "$gguf_digest" "$input_digest" "$reference_digest" "$approval_digest" || { log 'self-test FAIL: valid transfer packet rejected'; fail=1; }
  touch "$packet/unexpected"
  if validate_transfer_manifest "$packet/manifest.json" "$manifest_digest" "$head" "$gguf_digest" "$input_digest" "$reference_digest" "$approval_digest" >/dev/null 2>&1; then log 'self-test FAIL: extra transfer packet entry accepted'; fail=1; fi
  (( fail == 0 )) || return 1
  log "self-test PASS"
)

main() {
  local gguf='' input='' reference='' approval='' approval_sha='' packet_manifest='' packet_manifest_sha='' evidence_dir='' expected_head='' expected_gguf_sha='' expected_input_sha='' expected_reference_sha='' self_test=0 gguf_sha actual_head
  local seen_gguf=0 seen_input=0 seen_reference=0 seen_approval=0 seen_approval_sha=0 seen_packet=0 seen_packet_sha=0 seen_evidence=0 seen_head=0 seen_gguf_sha=0 seen_input_sha=0 seen_reference_sha=0 seen_self=0
  while (( $# > 0 )); do
    case "$1" in
      --gguf)
        (( seen_gguf == 0 )) || die 'duplicate --gguf'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die '--gguf requires a nonempty path'; seen_gguf=1
        gguf="$2"; shift 2 ;;
      --gguf-sha256)
        (( seen_gguf_sha == 0 )) || die 'duplicate --gguf-sha256'; [[ $# -ge 2 && "$2" =~ ^[0-9a-f]{64}$ ]] || die '--gguf-sha256 requires a lowercase 64-hex digest'; seen_gguf_sha=1
        expected_gguf_sha="$2"; shift 2 ;;
      --input)
        (( seen_input == 0 )) || die 'duplicate --input'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die '--input requires a nonempty path'; seen_input=1
        input="$2"; shift 2 ;;
      --input-sha256)
        (( seen_input_sha == 0 )) || die 'duplicate --input-sha256'; [[ $# -ge 2 && "$2" =~ ^[0-9a-f]{64}$ ]] || die '--input-sha256 requires a lowercase 64-hex digest'; seen_input_sha=1
        expected_input_sha="$2"; shift 2 ;;
      --reference)
        (( seen_reference == 0 )) || die 'duplicate --reference'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die '--reference requires a nonempty path'; seen_reference=1
        reference="$2"; shift 2 ;;
      --reference-sha256)
        (( seen_reference_sha == 0 )) || die 'duplicate --reference-sha256'; [[ $# -ge 2 && "$2" =~ ^[0-9a-f]{64}$ ]] || die '--reference-sha256 requires a lowercase 64-hex digest'; seen_reference_sha=1
        expected_reference_sha="$2"; shift 2 ;;
      --expected-head)
        (( seen_head == 0 )) || die 'duplicate --expected-head'; [[ $# -ge 2 && "$2" =~ ^[0-9a-f]{40}$ ]] || die '--expected-head requires a lowercase 40-hex commit'; seen_head=1
        expected_head="$2"; shift 2 ;;
      --evidence-dir)
        (( seen_evidence == 0 )) || die 'duplicate --evidence-dir'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die '--evidence-dir requires a nonempty path'; seen_evidence=1
        evidence_dir="$2"; shift 2 ;;
      --approval-evidence)
        (( seen_approval == 0 )) || die 'duplicate --approval-evidence'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die '--approval-evidence requires a nonempty path'; seen_approval=1
        approval="$2"; shift 2 ;;
      --approval-sha256)
        (( seen_approval_sha == 0 )) || die 'duplicate --approval-sha256'; [[ $# -ge 2 && "$2" =~ ^[0-9a-f]{64}$ ]] || die '--approval-sha256 requires a lowercase 64-hex digest'; seen_approval_sha=1
        approval_sha="$2"; shift 2 ;;
      --packet-manifest)
        (( seen_packet == 0 )) || die 'duplicate --packet-manifest'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die '--packet-manifest requires a nonempty path'; seen_packet=1
        packet_manifest="$2"; shift 2 ;;
      --packet-manifest-sha256)
        (( seen_packet_sha == 0 )) || die 'duplicate --packet-manifest-sha256'; [[ $# -ge 2 && "$2" =~ ^[0-9a-f]{64}$ ]] || die '--packet-manifest-sha256 requires a lowercase 64-hex digest'; seen_packet_sha=1
        packet_manifest_sha="$2"; shift 2 ;;
      --self-test)
        (( seen_self == 0 )) || die 'duplicate --self-test'; seen_self=1; self_test=1; shift ;;
      -h|--help)
        usage; return 0 ;;
      *)
        usage; die "unknown argument $1"; return 2 ;;
    esac
  done

  if (( self_test == 1 )); then
    [[ -z "$gguf$input$reference$approval$approval_sha$packet_manifest$packet_manifest_sha$evidence_dir$expected_head$expected_gguf_sha$expected_input_sha$expected_reference_sha" ]] \
      || die "--self-test accepts no other arguments"
    run_self_test
    return
  fi
  [[ -n "$gguf" && -n "$input" && -n "$reference" && -n "$approval" && -n "$approval_sha" && -n "$packet_manifest" && -n "$packet_manifest_sha" && -n "$evidence_dir" && -n "$expected_head" && -n "$expected_gguf_sha" && -n "$expected_input_sha" && -n "$expected_reference_sha" ]] \
    || { usage; die "all artifact hashes, --expected-head, paths, --approval-evidence and --evidence-dir are required"; }

  cd "$VOKRA_ROOT"
  actual_head="$(git rev-parse HEAD)" || die 'could not read checkout HEAD'
  [[ "$actual_head" == "$expected_head" ]] || die "checkout HEAD $actual_head does not match --expected-head $expected_head"
  [[ -z "$(git status --porcelain --untracked-files=all)" ]] || die 'checkout must be clean before approval and model processing'
  require_absent_evidence_dir "$evidence_dir" "$gguf" "$input" "$reference" "$approval" "$packet_manifest"
  require_remote_apple_host
  require_tooling
  require_file "corrected NSNet2 GGUF" "$gguf"
  require_file "NSNet2 input WAV" "$input"
  require_file "independent NSNet2 reference WAV" "$reference"
  validate_transfer_manifest "$packet_manifest" "$packet_manifest_sha" "$expected_head" "$expected_gguf_sha" "$expected_input_sha" "$expected_reference_sha" "$approval_sha"
  license_preflight "$approval" "$expected_head" "$approval_sha"
  # The preflight requires an absent target; claim it atomically with mkdir so
  # a concurrent creator cannot turn this evidence path into a shared tree.
  [[ -d "$(dirname "$evidence_dir")" && ! -L "$(dirname "$evidence_dir")" ]] || die 'evidence parent must already exist'
  mkdir "$evidence_dir"
  gguf_sha="$(sha256_file "$gguf")"
  [[ "$gguf_sha" == "$expected_gguf_sha" ]] || die 'GGUF SHA-256 does not match expected digest'
  [[ "$(sha256_file "$input")" == "$expected_input_sha" ]] || die 'input WAV SHA-256 does not match expected digest'
  [[ "$(sha256_file "$reference")" == "$expected_reference_sha" ]] || die 'reference WAV SHA-256 does not match expected digest'
  record_environment "$evidence_dir/environment.txt"
  {
    echo "gguf=$gguf"
    echo "gguf_sha256=$gguf_sha"
    echo "gguf_sha256_expected=$expected_gguf_sha"
    echo "expected_head=$expected_head"
    echo "actual_head=$actual_head"
    echo "input_wav=$input"
    echo "input_wav_sha256=$(sha256_file "$input")"
    echo "reference_wav=$reference"
    echo "reference_wav_sha256=$(sha256_file "$reference")"
    echo "approval_sha256=$approval_sha"
    echo "packet_manifest=$packet_manifest"
    echo "packet_manifest_sha256=$packet_manifest_sha"
  } > "$evidence_dir/input-hashes.txt"

  log "running exact NSNet2 CPU/reference/Metal parity"
  env \
    "$GGUF_ENV=$gguf" \
    "$WAV_ENV=$input" \
    "$REFERENCE_WAV_ENV=$reference" \
    RUST_TEST_THREADS=1 CARGO_NET_OFFLINE=true \
    cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --offline --release \
      -p vokra-models --features metal --test "$TEST_TARGET" "$TEST_NAME" \
      -- --ignored --exact --nocapture --test-threads=1 2>&1 | tee "$evidence_dir/parity.log"

  require_cargo_result "$evidence_dir/parity.log"
  require_metric_sentinel "$evidence_dir/parity.log" 'CPU/Metal'
  require_metric_sentinel "$evidence_dir/parity.log" 'CPU/reference'
  require_metric_sentinel "$evidence_dir/parity.log" 'Metal/reference'
  for marker in cpu_reference metal_reference metal_vs_cpu; do
    case "$marker" in
      cpu_reference) expected_marker='NSNet2_PARITY cpu_reference=PASS' ;;
      metal_reference) expected_marker='NSNet2_PARITY metal_vs_reference=PASS' ;;
      metal_vs_cpu) expected_marker='NSNet2_PARITY metal_vs_cpu=PASS' ;;
    esac
    [[ "$(grep -Ec "^${expected_marker}$" "$evidence_dir/parity.log" || true)" == 1 ]] \
      || die "Rust parity test must emit exactly one $expected_marker marker"
  done
  actual_head="$(git -C "$VOKRA_ROOT" rev-parse HEAD)"
  [[ "$actual_head" == "$expected_head" ]] || die 'checkout HEAD changed before PASS publication'
  [[ -z "$(git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all)" ]] || die 'checkout became dirty before PASS publication'
  {
    echo "verdict=PASS"
    echo "git_commit=$(git -C "$VOKRA_ROOT" rev-parse HEAD)"
    echo "expected_head=$expected_head"
    echo "approval_sha256=$approval_sha"
    echo "gguf_sha256=$gguf_sha"
    echo "input_sha256_expected=$expected_input_sha"
    echo "reference_sha256_expected=$expected_reference_sha"
    echo "nsnet2_cpu_reference=PASS"
    echo "nsnet2_metal_reference=PASS"
    echo "nsnet2_metal_vs_cpu=PASS"
    echo "test=$TEST_TARGET::$TEST_NAME"
    echo "network=NOT_PERFORMED"
    echo "conversion=NOT_PERFORMED"
    echo "publication=NOT_PERFORMED"
  } > "$evidence_dir/summary.txt"
  log "PASS: pull only $evidence_dir, then remove staged inputs or destroy the remote worker"
}

main "$@"
