#!/usr/bin/env bash
set -euo pipefail
ROOT="${VOKRA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
VERIFY="$ROOT/tools/parity/ecapa_tdnn_verify_reference.py"
PARITY_PROJECT="$ROOT/tools/parity"
PROJECT_FILE="$PARITY_PROJECT/pyproject.toml"
LOCK_FILE="$PARITY_PROJECT/uv.lock"
TEST="public_artifact_matches_speechbrain"
ECAPA_UPSTREAM_REVISION="0f99f2d0ebe89ac095bcc5903c4dd8f72b367286"
ECAPA_CHECKPOINT_SHA256="0575cb64845e6b9a10db9bcb74d5ac32b326b8dc90352671d345e2ee3d0126a2"
ECAPA_WAV_SHA256="bf2dde5cb516939ff619d62fc07d4f4bec5b5d521aee3d07ae51828c9d93be0b"
ECAPA_PCM_SHA256="48aedc3a10b14b49ebe8da2efd1dd91cbe7dbbaf58278732e7fdb04f6d6cc1e9"
ECAPA_FEATURES_SHA256="6ea88148da19e1179e9c8bc27fa9b76c742d5b82376e9c4f153bf1da3cd6a191"
ECAPA_EMBEDDING_SHA256="f6b297f3c9e8746d0a2ceaded702b1ce5e741fd3957ce1879b07608f7bd082e4"
ECAPA_MANIFEST_JSON_SHA256="0cb074241201c3f14fff33c98bda9d1434e6c3ef1792db7310283a0113cf0b94"
die() { printf '[ecapa-apple] ERROR: %s\n' "$*" >&2; return 2; }
sha256() { shasum -a 256 "$1" | awk '{print $1}'; }
verify_head() {
  if ! [[ "$1" =~ ^[0-9a-f]{40}$ ]]; then die 'expected HEAD must be lowercase 40-hex'; return 2; fi
  if [[ -n "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]]; then die 'checkout must be clean'; return 2; fi
  if [[ "$(git -C "$ROOT" rev-parse HEAD)" != "$1" ]]; then die 'checkout HEAD mismatch'; return 2; fi
}
verify_reference() {
  local dir="$1" manifest="$2" packet="$3" pcm_sha="$4" features_sha="$5" embedding_sha="$6" manifest_json_sha="$7"
  [[ -d "$dir" && ! -L "$dir" ]] || die 'reference directory missing or symlinked'
  reject_symlink_ancestors "$dir"
  UV_NO_CACHE=1 uv run --no-project --offline --python 3.12 python "$VERIFY" \
    --directory "$dir" --revision "$ECAPA_UPSTREAM_REVISION" \
    --checkpoint-sha256 "$ECAPA_CHECKPOINT_SHA256" --wav-sha256 "$ECAPA_WAV_SHA256" \
    --manifest-sha256 "$manifest" --packet-sha256 "$packet" \
    --pcm-sha256 "$pcm_sha" --features-sha256 "$features_sha" \
    --embedding-sha256 "$embedding_sha" --manifest-json-sha256 "$manifest_json_sha" \
    || die 'reference authentication failed'
}
reject_symlink_ancestors() {
  local path="$1" component rest current
  [[ "$path" == /* ]] || die 'paths must be absolute'
  rest="${path#/}"; current="/"
  while [[ -n "$rest" ]]; do
    if [[ "$rest" == */* ]]; then component="${rest%%/*}"; rest="${rest#*/}"; else component="$rest"; rest=""; fi
    [[ -n "$component" ]] || continue
    current="$current$component"
    [[ ! -L "$current" ]] || die "path contains symlink ancestor: $path"
    current="$current/"
  done
}
canonical_absent_path() {
  local path="$1" suffix='' name parent
  reject_symlink_ancestors "$path"
  while [[ ! -d "$path" || -L "$path" ]]; do
    name="${path##*/}"; [[ -n "$name" ]] && suffix="/$name$suffix"
    parent="${path%/*}"; [[ "$parent" == "$path" ]] && parent=/; path="$parent"
  done
  (cd -P "$path" && printf '%s%s\n' "$PWD" "$suffix")
}
paths_overlap() { [[ "$1" == "$2" || "$1" == "$2"/* || "$2" == "$1"/* ]]; }
validate_evidence_path() {
  local evidence="$1" candidate root_real input input_real
  [[ ! -e "$evidence" && ! -L "$evidence" ]] || die 'evidence directory must be absent'
  candidate="$(canonical_absent_path "$evidence")" || die 'invalid evidence path'
  root_real="$(cd -P "$ROOT" && pwd)" || die 'cannot resolve checkout'
  paths_overlap "$candidate" "$root_real" && die 'evidence overlaps checkout'
  for input in "$gguf" "$reference" "$approval"; do
    input_real="$(cd -P "$(dirname "$input")" && pwd)/$(basename "$input")" || die 'cannot resolve input'
    paths_overlap "$candidate" "$input_real" && die 'evidence overlaps protected input'
  done
}
require_approval() {
  local file="$1" project_sha lock_sha
  [[ -f "$PROJECT_FILE" && ! -L "$PROJECT_FILE" && -f "$LOCK_FILE" && ! -L "$LOCK_FILE" ]] || die 'locked parity project missing or symlinked'
  [[ -f "$file" && ! -L "$file" && -s "$file" ]] || die 'approval evidence missing or symlinked'
  reject_symlink_ancestors "$file"
  project_sha="$(sha256 "$PROJECT_FILE")"; lock_sha="$(sha256 "$LOCK_FILE")"
  if UV_NO_CACHE=1 uv run --no-project --offline --python 3.12 python - "$file" "$project_sha" "$lock_sha" <<'PY'
import hashlib, json, pathlib, sys
def unique(pairs):
    out = {}
    for key, value in pairs:
        if key in out: raise ValueError('duplicate JSON key: ' + key)
        out[key] = value
    return out
try:
    data = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding='utf-8'), object_pairs_hook=unique)
    if not isinstance(data, dict): raise ValueError('approval root is not an object')
    keys = {'schema','model','upstream_repo','upstream_revision','license_spdx','project_sha256','lock_sha256','no_upload','decision','signer','scope_sha256'}
    if set(data) != keys: raise ValueError('approval schema is not exact')
    identity = ('vokra-validation-approval-v1','ecapa-tdnn','speechbrain/spkrec-ecapa-voxceleb','0f99f2d0ebe89ac095bcc5903c4dd8f72b367286','apache-2.0')
    if (data['schema'],data['model'],data['upstream_repo'],data['upstream_revision'],data['license_spdx']) != identity: raise ValueError('approval identity mismatch')
    if data['project_sha256'] != sys.argv[2] or data['lock_sha256'] != sys.argv[3] or data['no_upload'] is not True or data['decision'] != 'APPROVED': raise ValueError('approval facts mismatch')
    if not isinstance(data['signer'], str) or not data['signer'].strip() or data['signer'].strip().upper() in {'TODO','UNRESOLVED','OWNER_SIGNOFF_REQUIRED'}: raise ValueError('approval signer unresolved')
    scope = {'license_spdx':data['license_spdx'],'lock_sha256':sys.argv[3],'model':data['model'],'no_upload':True,'project_sha256':sys.argv[2],'upstream_repo':data['upstream_repo'],'upstream_revision':data['upstream_revision']}
    if data['scope_sha256'] != hashlib.sha256(json.dumps(scope,sort_keys=True,separators=(',',':')).encode()).hexdigest(): raise ValueError('approval scope digest mismatch')
except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
    raise SystemExit('approval gate BLOCKED: ' + str(exc))
PY
  then :; else die 'owner approval is invalid'; fi
  UV_NO_CACHE=1 uv run --no-project --offline --python 3.12 python "$ROOT/scripts/publish/signoff_match.py" --check-repo speechbrain-spkrec-ecapa-voxceleb --audit "$ROOT/docs/license-audit.md" || die 'repository license signoff unresolved'
}
require_cargo_singleton() {
  local log_file="$1" test_name="$2" named_count test_count result_count
  named_count="$(grep -Ec "^test ${test_name} \.\.\. ok$" "$log_file" || true)"
  test_count="$(grep -Ec '^test [^ ]+ \.\.\.' "$log_file" || true)"
  result_count="$(grep -Ec '^test result:' "$log_file" || true)"
  if ! [[ "$named_count" == 1 && "$test_count" == 1 && "$result_count" == 1 ]]; then
    die 'Cargo log is not one exact test/result'
    return 2
  fi
  if ! grep -Eq '^test result: ok\. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out(; finished in [0-9]+(\.[0-9]+)?s)?$' "$log_file"; then
    die 'Cargo result is not an exact singleton pass'
    return 2
  fi
}
self_test() {
  local self="${BASH_SOURCE[0]}" temporary reference_test name packet_sha manifest_sha synthetic_log
  temporary="$(mktemp -d "${TMPDIR:-/tmp}/vokra-ecapa-apple-self.XXXXXX")"
  temporary="$(cd -P "$temporary" && pwd)"
  trap '[[ -n ${temporary:-} ]] && rm -rf -- "$temporary"' EXIT
  reference_test="$temporary/reference"
  mkdir "$reference_test"
  for name in pcm.f32.bin features.f32.bin embedding.f32.bin manifest.json; do
    cp "$ROOT/crates/vokra-models/tests/fixtures/ecapa_tdnn/$name" "$reference_test/$name"
  done
  : > "$reference_test/reference-manifest.sha256"
  for name in pcm.f32.bin features.f32.bin embedding.f32.bin manifest.json; do
    printf '%s  %s\n' "$(sha256 "$reference_test/$name")" "$name" >> "$reference_test/reference-manifest.sha256"
  done
  packet_sha="$({ for name in pcm.f32.bin features.f32.bin embedding.f32.bin manifest.json; do cat "$reference_test/$name"; done; } | shasum -a 256 | awk '{print $1}')"
  printf '%s\n' "$packet_sha" > "$reference_test/reference-packet.sha256"
  manifest_sha="$(sha256 "$reference_test/reference-manifest.sha256")"
  verify_reference "$reference_test" "$manifest_sha" "$packet_sha" "$ECAPA_PCM_SHA256" "$ECAPA_FEATURES_SHA256" "$ECAPA_EMBEDDING_SHA256" "$ECAPA_MANIFEST_JSON_SHA256" || die 'reference verifier self-test packet failed'
  printf '\001' >> "$reference_test/pcm.f32.bin"
  if verify_reference "$reference_test" "$manifest_sha" "$packet_sha" "$ECAPA_PCM_SHA256" "$ECAPA_FEATURES_SHA256" "$ECAPA_EMBEDDING_SHA256" "$ECAPA_MANIFEST_JSON_SHA256" >/dev/null 2>&1; then
    die 'reference verifier self-test accepted tampered data'
  fi
  cp "$ROOT/crates/vokra-models/tests/fixtures/ecapa_tdnn/pcm.f32.bin" "$reference_test/pcm.f32.bin"
  touch "$reference_test/unexpected"
  if verify_reference "$reference_test" "$manifest_sha" "$packet_sha" "$ECAPA_PCM_SHA256" "$ECAPA_FEATURES_SHA256" "$ECAPA_EMBEDDING_SHA256" "$ECAPA_MANIFEST_JSON_SHA256" >/dev/null 2>&1; then
    die 'reference verifier self-test accepted an extra entry'
  fi
  synthetic_log="$temporary/cargo.log"
  printf '%s\n' 'test public_artifact_matches_speechbrain ... ok' 'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s' > "$synthetic_log"
  require_cargo_singleton "$synthetic_log" public_artifact_matches_speechbrain
  printf '%s\n' 'test unexpected ... ok' >> "$synthetic_log"
  if require_cargo_singleton "$synthetic_log" public_artifact_matches_speechbrain >/dev/null 2>&1; then
    die 'Cargo self-test accepted an extra test'
  fi
  printf '%s\n' 'test public_artifact_matches_speechbrain ... ok' 'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s' > "$synthetic_log"
  printf '%s\n' 'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s' >> "$synthetic_log"
  if require_cargo_singleton "$synthetic_log" public_artifact_matches_speechbrain >/dev/null 2>&1; then
    die 'Cargo self-test accepted an extra result'
  fi
  UV_NO_CACHE=1 uv run --no-project --offline --python 3.12 python "$VERIFY" --self-test >/dev/null \
    || die 'reference verifier self-test failed'
  for token in 'CARGO_NET_OFFLINE=true cargo test --offline' 'ECAPA-TDNN CPU_VS_UPSTREAM PASS' 'ECAPA-TDNN METAL_VS_UPSTREAM MEASUREMENT_ONLY' 'ECAPA-TDNN METAL_VS_CPU PASS' 'ecapa_tdnn_verify_reference.py' '--pcm-sha256' '--features-sha256' '--embedding-sha256' '--manifest-json-sha256' '--approval-sha256' 'approval gate BLOCKED' 'expected HEAD' 'require_cargo_singleton'; do
    grep -Fq -- "$token" "$self" || die "self-test contract missing: $token"
  done
  printf '[ecapa-apple] self-test PASS\n' >&2
}
usage() { printf '%s\n' 'usage: apple-silicon-ecapa-tdnn.sh --ecapa-gguf FILE --ecapa-gguf-sha256 SHA --reference DIR --reference-manifest-sha256 SHA --reference-packet-sha256 SHA --reference-pcm-sha256 SHA --reference-features-sha256 SHA --reference-embedding-sha256 SHA --reference-manifest-json-sha256 SHA --expected-head HEX40 --approval-evidence FILE --approval-sha256 SHA --evidence-dir DIR' >&2; }
gguf='' gguf_sha='' reference='' manifest_sha='' packet_sha='' pcm_sha='' features_sha='' embedding_sha='' manifest_json_sha='' expected_head='' approval='' approval_sha='' evidence='' self_test=0
seen_gguf=0 seen_gguf_sha=0 seen_reference=0 seen_manifest=0 seen_packet=0 seen_pcm=0 seen_features=0 seen_embedding=0 seen_manifest_json=0 seen_head=0 seen_approval=0 seen_approval_sha=0 seen_evidence=0 seen_self=0
while (( $# > 0 )); do
  case "$1" in
    --ecapa-gguf) (( seen_gguf == 0 && $# >= 2 )) || die 'duplicate/missing --ecapa-gguf'; seen_gguf=1; gguf="$2"; shift 2;;
    --ecapa-gguf-sha256) (( seen_gguf_sha == 0 && $# >= 2 )) || die 'duplicate/missing GGUF SHA'; [[ "$2" =~ ^[0-9a-f]{64}$ ]] || die 'malformed GGUF SHA'; seen_gguf_sha=1; gguf_sha="$2"; shift 2;;
    --reference) (( seen_reference == 0 && $# >= 2 )) || die 'duplicate/missing --reference'; seen_reference=1; reference="$2"; shift 2;;
    --reference-manifest-sha256) (( seen_manifest == 0 && $# >= 2 )) || die 'duplicate/missing manifest SHA'; [[ "$2" =~ ^[0-9a-f]{64}$ ]] || die 'malformed manifest SHA'; seen_manifest=1; manifest_sha="$2"; shift 2;;
    --reference-packet-sha256) (( seen_packet == 0 && $# >= 2 )) || die 'duplicate/missing packet SHA'; [[ "$2" =~ ^[0-9a-f]{64}$ ]] || die 'malformed packet SHA'; seen_packet=1; packet_sha="$2"; shift 2;;
    --reference-pcm-sha256) (( seen_pcm == 0 && $# >= 2 )) || die 'duplicate/missing PCM SHA'; [[ "$2" =~ ^[0-9a-f]{64}$ ]] || die 'malformed PCM SHA'; seen_pcm=1; pcm_sha="$2"; shift 2;;
    --reference-features-sha256) (( seen_features == 0 && $# >= 2 )) || die 'duplicate/missing features SHA'; [[ "$2" =~ ^[0-9a-f]{64}$ ]] || die 'malformed features SHA'; seen_features=1; features_sha="$2"; shift 2;;
    --reference-embedding-sha256) (( seen_embedding == 0 && $# >= 2 )) || die 'duplicate/missing embedding SHA'; [[ "$2" =~ ^[0-9a-f]{64}$ ]] || die 'malformed embedding SHA'; seen_embedding=1; embedding_sha="$2"; shift 2;;
    --reference-manifest-json-sha256) (( seen_manifest_json == 0 && $# >= 2 )) || die 'duplicate/missing manifest JSON SHA'; [[ "$2" =~ ^[0-9a-f]{64}$ ]] || die 'malformed manifest JSON SHA'; seen_manifest_json=1; manifest_json_sha="$2"; shift 2;;
    --expected-head) (( seen_head == 0 && $# >= 2 )) || die 'duplicate/missing expected HEAD'; [[ "$2" =~ ^[0-9a-f]{40}$ ]] || die 'malformed expected HEAD'; seen_head=1; expected_head="$2"; shift 2;;
    --approval-evidence) (( seen_approval == 0 && $# >= 2 )) || die 'duplicate/missing approval'; seen_approval=1; approval="$2"; shift 2;;
    --approval-sha256) (( seen_approval_sha == 0 && $# >= 2 )) || die 'duplicate/missing approval SHA'; [[ "$2" =~ ^[0-9a-f]{64}$ ]] || die 'malformed approval SHA'; seen_approval_sha=1; approval_sha="$2"; shift 2;;
    --evidence-dir) (( seen_evidence == 0 && $# >= 2 )) || die 'duplicate/missing evidence dir'; seen_evidence=1; evidence="$2"; shift 2;;
    --self-test) (( seen_self == 0 )) || die 'duplicate --self-test'; seen_self=1; self_test=1; shift;;
    -h|--help) usage; exit 0;;
    *) usage; die "unknown argument: $1";;
  esac
done
if (( self_test )); then
  [[ "$seen_gguf$seen_gguf_sha$seen_reference$seen_manifest$seen_packet$seen_pcm$seen_features$seen_embedding$seen_manifest_json$seen_head$seen_approval$seen_approval_sha$seen_evidence" == 0000000000000 ]] || die '--self-test accepts no other arguments'
  self_test; exit
fi
[[ "$seen_gguf$seen_gguf_sha$seen_reference$seen_manifest$seen_packet$seen_pcm$seen_features$seen_embedding$seen_manifest_json$seen_head$seen_approval$seen_approval_sha$seen_evidence" == 1111111111111 ]] || die 'all arguments are required'
verify_head "$expected_head"
require_approval "$approval"
[[ "$(sha256 "$approval")" == "$approval_sha" ]] || die 'approval SHA-256 mismatch'
[[ -f "$gguf" && ! -L "$gguf" && -s "$gguf" ]] || die 'GGUF missing/symlinked/empty'
reject_symlink_ancestors "$gguf"
[[ "$(sha256 "$gguf")" == "$gguf_sha" ]] || die 'GGUF hash mismatch'
[[ "$pcm_sha" == "$ECAPA_PCM_SHA256" && "$features_sha" == "$ECAPA_FEATURES_SHA256" && "$embedding_sha" == "$ECAPA_EMBEDDING_SHA256" && "$manifest_json_sha" == "$ECAPA_MANIFEST_JSON_SHA256" ]] || die 'reference fixture hash is not the committed packet'
verify_reference "$reference" "$manifest_sha" "$packet_sha" "$pcm_sha" "$features_sha" "$embedding_sha" "$manifest_json_sha"
validate_evidence_path "$evidence"
[[ "$(uname -s)" == Darwin && "$(uname -m)" == arm64 && "${VOKRA_REMOTE_APPLE_SILICON:-0}" == 1 ]] || die 'remote Apple arm64 guard failed'
if ! command -v xcrun >/dev/null 2>&1 || ! xcrun -f metal >/dev/null 2>&1; then
  die 'Metal compiler unavailable'
fi
mkdir -p "$evidence"
printf '%q ' "$0" --ecapa-gguf "$gguf" --ecapa-gguf-sha256 "$gguf_sha" --reference "$reference" --reference-manifest-sha256 "$manifest_sha" --reference-packet-sha256 "$packet_sha" --reference-pcm-sha256 "$pcm_sha" --reference-features-sha256 "$features_sha" --reference-embedding-sha256 "$embedding_sha" --reference-manifest-json-sha256 "$manifest_json_sha" --expected-head "$expected_head" --approval-evidence "$approval" --approval-sha256 "$approval_sha" --evidence-dir "$evidence" > "$evidence/execution-args.txt"
printf '\n' >> "$evidence/execution-args.txt"
{
  echo "expected_head=$expected_head"
  echo "gguf_sha256=$gguf_sha"
  echo "reference_manifest_sha256=$manifest_sha"
  echo "reference_packet_sha256=$packet_sha"
  echo "reference_pcm_sha256=$ECAPA_PCM_SHA256"
  echo "reference_features_sha256=$ECAPA_FEATURES_SHA256"
  echo "reference_embedding_sha256=$ECAPA_EMBEDDING_SHA256"
  echo "reference_manifest_json_sha256=$ECAPA_MANIFEST_JSON_SHA256"
  echo "approval_sha256=$approval_sha"
} > "$evidence/input-hashes.txt"
CARGO_NET_OFFLINE=true cargo test --offline --manifest-path "$ROOT/Cargo.toml" --locked --release -p vokra-models --features metal --test parity_ecapa_tdnn_real "$TEST" -- --exact --ignored --nocapture 2>&1 | tee "$evidence/ecapa-parity.log"
log_file="$evidence/ecapa-parity.log"
require_cargo_singleton "$log_file" "$TEST"
for marker in 'ECAPA-TDNN CPU_VS_UPSTREAM PASS' 'ECAPA-TDNN METAL_VS_UPSTREAM MEASUREMENT_ONLY' 'ECAPA-TDNN METAL_VS_CPU PASS'; do
  if [[ "$(grep -Fxc "$marker" "$log_file" || true)" != 1 ]]; then
    die "missing/non-singleton sentinel: $marker"
    exit 2
  fi
done
{ echo 'verdict=CPU_PASS_METAL_MEASUREMENT_ONLY'; echo "expected_head=$expected_head"; echo "gguf_sha256=$gguf_sha"; echo "reference_manifest_sha256=$manifest_sha"; echo "reference_packet_sha256=$packet_sha"; echo "reference_pcm_sha256=$ECAPA_PCM_SHA256"; echo "reference_features_sha256=$ECAPA_FEATURES_SHA256"; echo "reference_embedding_sha256=$ECAPA_EMBEDDING_SHA256"; echo "reference_manifest_json_sha256=$ECAPA_MANIFEST_JSON_SHA256"; echo "approval_sha256=$approval_sha"; echo 'cpu_vs_upstream=PASS'; echo 'metal_vs_upstream=MEASUREMENT_ONLY'; echo 'metal_vs_cpu=PASS'; echo 'metal_upstream_bound=UNREGISTERED'; echo 'upload=NOT_PERFORMED'; } > "$evidence/summary.txt"
