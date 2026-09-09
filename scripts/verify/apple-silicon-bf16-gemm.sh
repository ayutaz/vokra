#!/usr/bin/env bash
# Apple arm64 BFMMLA parity verifier. Inputs are the committed, independently
# generated PyTorch fixture packet; no model execution, network access,
# conversion, upload, or publication is performed; there is no upload.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VOKRA_ROOT="${VOKRA_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
FIXTURE_DIR="$VOKRA_ROOT/tests/parity/bf16_gemm"
TEST_NAME="apple_silicon_neon_bf16_gemm_matches_pytorch_reference"
MANIFEST_SHA256="b9e7b687ef6352b30f258b0b1c02695e724e32443665e74981cac66b025b1ba3"
FIXTURE_EXPECTED='README.md full_k32_m8_n64_a.f32 full_k32_m8_n64_b.f32 full_k32_m8_n64_output.f32 manifest.json manifest.sha256 tails_m3_n35_k65_a.f32 tails_m3_n35_k65_b.f32 tails_m3_n35_k65_output.f32 tails_m9_n33_k31_a.f32 tails_m9_n33_k31_b.f32 tails_m9_n33_k31_output.f32'

log() { printf '[bf16-gemm-apple] %s\n' "$*" >&2; }
die() { log "ERROR: $*"; return 2; }
usage() {
  cat >&2 <<'EOF'
usage: apple-silicon-bf16-gemm.sh --expected-head HEX40 --evidence-dir ABSENT_DIR
       apple-silicon-bf16-gemm.sh --self-test

Runs the exact ignored NeonBf16/BFMMLA parity test against the committed
PyTorch fixture packet on a Darwin arm64 host. The fixture and manifest are
validated before the serial release test. The evidence directory must not
exist before the run.
EOF
}

sha256_file() { shasum -a 256 "$1" | awk '{print $1}'; }
require_abs() { [[ "$1" == /* ]] || die "$2 must be absolute: $1"; }

require_bf16_support() {
  if [[ "${VOKRA_TEST_FORCE_NO_BF16:-0}" == 1 ]]; then
    die 'BF16 support was intentionally disabled'
    return 2
  fi
  local value
  value="$(sysctl -n hw.optional.arm.FEAT_BF16 2>/dev/null || true)"
  [[ "$value" == 1 ]] || die 'Darwin does not positively report hw.optional.arm.FEAT_BF16'
}

require_host() {
  [[ "${VOKRA_REMOTE_APPLE_SILICON:-0}" == 1 ]] || die 'VOKRA_REMOTE_APPLE_SILICON=1 is absent'
  [[ "$(uname -s)" == Darwin ]] || die 'requires Darwin'
  [[ "$(uname -m)" == arm64 ]] || die 'requires arm64'
  require_bf16_support
}

canonical_existing() {
  local path="$1" current=/ part rest
  [[ "$path" == /* && -e "$path" && ! -L "$path" ]] || return 1
  rest="${path#/}"
  while [[ -n "$rest" ]]; do
    if [[ "$rest" == */* ]]; then part="${rest%%/*}"; rest="${rest#*/}"; else part="$rest"; rest=''; fi
    [[ -n "$part" ]] || continue
    [[ "$part" != . && "$part" != .. ]] || return 1
    current="${current%/}/$part"
    [[ ! -L "$current" ]] || return 1
  done
  if [[ -d "$path" ]]; then (cd -P "$path" && pwd); else
    local parent
    parent="$(dirname "$path")"
    (cd -P "$parent" && printf '%s/%s\n' "$PWD" "$(basename "$path")")
  fi
}

canonical_absent() {
  local path="$1" current rest part suffix='' parent
  [[ "$path" == /* && ! -e "$path" && ! -L "$path" ]] || return 1
  rest="${path#/}"; current=/
  while [[ -n "$rest" ]]; do
    if [[ "$rest" == */* ]]; then part="${rest%%/*}"; rest="${rest#*/}"; else part="$rest"; rest=''; fi
    [[ -n "$part" ]] || continue
    [[ "$part" != . && "$part" != .. ]] || return 1
    current="${current%/}/$part"
    [[ ! -L "$current" ]] || return 1
  done
  while [[ ! -e "$path" ]]; do
    part="$(basename "$path")"; suffix="/$part$suffix"; parent="$(dirname "$path")"
    [[ "$parent" != "$path" ]] || return 1
    path="$parent"
  done
  [[ -d "$path" && ! -L "$path" ]] || return 1
  (cd -P "$path" && printf '%s%s\n' "$PWD" "$suffix")
}

require_evidence_path() {
  local evidence="$1" root_real parent_real
  require_abs "$evidence" 'evidence directory'
  [[ ! -e "$evidence" && ! -L "$evidence" ]] || die 'evidence directory must be absent before validation'
  root_real="$(canonical_existing "$VOKRA_ROOT")" || die 'checkout path is not canonical'
  parent_real="$(canonical_existing "$(dirname "$evidence")")" || die 'evidence parent is unavailable'
  local evidence_real
  evidence_real="$(canonical_absent "$evidence")" || { die 'evidence path contains dot components or symlink ancestry'; return 2; }
  [[ "$evidence_real" != "$root_real" && "$evidence_real" != "$root_real"/* ]] || die 'evidence overlaps checkout'
  [[ "$parent_real" != "$root_real"/* ]] || die 'evidence parent is inside checkout'
}

require_clean_checkout() {
  local expected_head="$1" actual_head
  [[ -d "$VOKRA_ROOT/.git" ]] || die 'checkout is not a git worktree'
  [[ -z "$(git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all)" ]] || die 'checkout is dirty'
  actual_head="$(git -C "$VOKRA_ROOT" rev-parse HEAD)" || die 'cannot read checkout HEAD'
  [[ "$actual_head" == "$expected_head" ]] || die "checkout HEAD $actual_head does not match --expected-head $expected_head"
}

require_fixture_entry_set() {
  local directory="$1" expected="$2" actual
  [[ -d "$directory" && ! -L "$directory" ]] || { die 'fixture directory missing or symlinked'; return 2; }
  actual="$(find -P "$directory" -mindepth 1 -maxdepth 1 -print | awk -F/ '{print $NF}' | sort | tr '\n' ' ' | sed 's/[[:space:]]*$//')"
  [[ "$actual" == "$expected" ]] || { die 'fixture directory entry set drifted'; return 2; }
}

require_fixture_entries_regular() {
  local directory="$1" entry
  for entry in $FIXTURE_EXPECTED; do
    canonical_existing "$directory/$entry" >/dev/null || { die "fixture entry has symlink ancestry: $entry"; return 2; }
    [[ -f "$directory/$entry" && ! -L "$directory/$entry" ]] || { die "fixture entry is not a regular file: $entry"; return 2; }
  done
}

require_fixture_contract() {
  [[ -d "$FIXTURE_DIR" && ! -L "$FIXTURE_DIR" ]] || die 'BF16 fixture directory missing or symlinked'
  require_fixture_entry_set "$FIXTURE_DIR" "$FIXTURE_EXPECTED"
  require_fixture_entries_regular "$FIXTURE_DIR"
  [[ "$(shasum -a 256 "$FIXTURE_DIR/manifest.json" | awk '{print $1}')" == "$MANIFEST_SHA256" ]] || die 'fixture manifest SHA-256 drifted'
  [[ "$(tr -d '\r\n' < "$FIXTURE_DIR/manifest.sha256")" == "$MANIFEST_SHA256  manifest.json" ]] || die 'manifest pin drifted'
  local name sha bytes
  for name in \
    full_k32_m8_n64_a.f32:282a02285486d78143da8c00793da1f93571c7efe35645d751de652b7f21c715:3072 \
    full_k32_m8_n64_b.f32:4bce8eddd102f18ae4ef169079ba12c395c8a28e3385f0a88425228c6de49036:24576 \
    full_k32_m8_n64_output.f32:e7247ba4fc0a8d5ef05e6fb31caa816603f4a490b5460add23b84822ef478325:2048 \
    tails_m3_n35_k65_a.f32:c312992e4e808f76e729ee8bcef8cd600cca75d923f5b31ae3461b3d436615cd:780 \
    tails_m3_n35_k65_b.f32:7d60d46524a18d252c591f3db963c655986b8e419c6ecdb1b5f14cda2bede0e3:9100 \
    tails_m3_n35_k65_output.f32:4485c00d80f7c94fe82c0d90bb7d006539d298ce0de0258aae421cc45299b627:420 \
    tails_m9_n33_k31_a.f32:8def0df1680e33c3cb34a3674a9b12bd4e1d77563d3c8ae0ae3490e7acb875b6:1116 \
    tails_m9_n33_k31_b.f32:864c37ad7a242e7ada61750cfe53d12f889e30892568251ef1f8578e38b8b35b:4092 \
    tails_m9_n33_k31_output.f32:f01cb1caff693b508f414fceb0d46be69b2ca14096c4a22e378945b2dcdc040f:1188; do
    IFS=: read -r name sha bytes <<<"$name"
    [[ "$(wc -c < "$FIXTURE_DIR/$name" | tr -d '[:space:]')" == "$bytes" ]] || die "fixture byte count drifted: $name"
    [[ "$(sha256_file "$FIXTURE_DIR/$name")" == "$sha" ]] || die "fixture SHA-256 drifted: $name"
  done
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python - "$FIXTURE_DIR/manifest.json" <<'PY' || { die 'fixture provenance/schema drifted'; return 2; }
import json, sys
from pathlib import Path
def reject(pairs):
    out = {}
    for key, value in pairs:
        if key in out: raise ValueError(f"duplicate key: {key}")
        out[key] = value
    return out
data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"), object_pairs_hook=reject)
if set(data) != {"cases", "comparison", "provenance", "schema"}: raise ValueError("top-level keys")
if data["schema"] != "vokra-bf16-gemm-parity-v1": raise ValueError("schema")
if data["comparison"] != {"atol": 0.001, "rtol": 0.0}: raise ValueError("comparison")
if data["provenance"] != {
    "byte_order":"little-endian", "device":"cpu", "dtype":"float32",
    "generator":"tools/parity/bf16_gemm/dump_reference.py",
    "generator_identity":"deterministic torch.matmul BF16-widened oracle",
    "oracle":"torch.matmul(a.to(torch.bfloat16).to(torch.float32), b.to(torch.bfloat16).to(torch.float32))",
    "randomness":"none", "torch_version":"2.13.0+cpu",
}: raise ValueError("provenance")
expected = {"full_k32_m8_n64":{"m":8,"n":64,"k":96}, "tails_m3_n35_k65":{"m":3,"n":35,"k":65}, "tails_m9_n33_k31":{"m":9,"n":33,"k":31}}
if set(data["cases"]) != set(expected): raise ValueError("cases")
for name, shape in expected.items():
    case = data["cases"][name]
    if set(case) != {"shape", "tensors"} or case["shape"] != shape: raise ValueError(name)
PY
}

parse_test_log() {
  local log_file="$1" marker metric
  [[ "$(grep -Ec "^test $TEST_NAME \.\.\. ok$" "$log_file" || true)" == 1 ]] || { die 'named Apple BF16 test did not pass exactly once'; return 2; }
  [[ "$(grep -Ec '^test result: ok\. 1 passed; 0 failed; 0 ignored; 0 measured; [0-9]+ filtered out(; finished in [0-9]+\.[0-9]+s)?$' "$log_file" || true)" == 1 ]] || { die 'test result was not exactly one pass'; return 2; }
  [[ "$(grep -Ec '^APPLE_BF16_GEMM_PASS$' "$log_file" || true)" == 1 ]] || { die 'BF16 PASS sentinel missing or duplicated'; return 2; }
  [[ "$(grep -Ec '^APPLE_BF16_GEMM backend=neon-bf16 cases=3 max_abs_error=[0-9]+\.[0-9]{9}e[+-][0-9]+ atol=1\.000000000e-3 rtol=0\.000000000e0$' "$log_file" || true)" == 1 ]] || { die 'BF16 metric marker malformed or duplicated'; return 2; }
  marker="$(grep '^APPLE_BF16_GEMM backend=' "$log_file")"
  metric="${marker#*max_abs_error=}"; metric="${metric%% *}"
  awk -v value="$metric" 'BEGIN { if (value != value || value < 0 || value > 1e-3) exit 1 }' || { die 'BF16 max error exceeds manifest atol'; return 2; }
}

self_test() {
  local script="${BASH_SOURCE[0]}" fail=0 token tmp good
  for token in 'VOKRA_REMOTE_APPLE_SILICON=1' 'Darwin' 'arm64' 'hw.optional.arm.FEAT_BF16' \
    '--expected-head' 'rev-parse HEAD' 'checkout HEAD' \
    'NeonBf16' 'BFMMLA' 'CARGO_BUILD_JOBS=1' 'CARGO_NET_OFFLINE=true' \
    "cargo test --manifest-path \"\$VOKRA_ROOT/Cargo.toml\" --locked --offline --release -p vokra-backend-cpu --test bf16_gemm_torch_parity" \
    '-- --ignored --exact --show-output --test-threads=1' 'APPLE_BF16_GEMM_PASS' 'max_abs_error=' \
    'status --porcelain --untracked-files=all' \
    'b9e7b687ef6352b30f258b0b1c02695e724e32443665e74981cac66b025b1ba3' 'no model' 'no upload'; do
    grep -Fq -- "$token" "$script" || { log "self-test missing contract token: $token"; fail=1; }
  done
  local parity_source="$VOKRA_ROOT/crates/vokra-backend-cpu/tests/bf16_gemm_torch_parity.rs"
  grep -Fq 'kernels::gemm_bf16_on' "$parity_source" || { log 'self-test missing forced BF16 kernel call'; fail=1; }
  grep -Fq 'IsaPath::NeonBf16' "$parity_source" || { log 'self-test missing explicit NeonBf16 path'; fail=1; }
  if grep -Eq 'best_bf16_isa[[:space:]]*\(' "$parity_source"; then log 'self-test found dispatch fallback in Apple test'; fail=1; fi
  grep -En '(^|[[:space:]])(curl|wget|git[[:space:]]+push|huggingface-cli|publish-one\.sh)([[:space:]]|$)' "$script" >/dev/null && { log 'self-test found network/publication command'; fail=1; } || true
  if "$script" --unknown >/dev/null 2>&1; then log 'self-test accepted unknown option'; fail=1; fi
  if "$script" --expected-head bad >/dev/null 2>&1 || \
    "$script" --expected-head "$(printf '0%.0s' {1..40})" --expected-head "$(printf '1%.0s' {1..40})" >/dev/null 2>&1; then
    log 'self-test accepted malformed or duplicate --expected-head'; fail=1
  fi
  if VOKRA_TEST_FORCE_NO_BF16=1 require_bf16_support >/dev/null 2>&1; then log 'self-test accepted missing BF16 support'; fail=1; fi
  tmp="$(mktemp -d "${TMPDIR:-/tmp}/bf16-apple-self-test.XXXXXX")"
  trap 'rm -rf "$tmp"' RETURN
  mkdir "$tmp/extra-dir"
  if require_fixture_entry_set "$tmp" "$FIXTURE_EXPECTED" >/dev/null 2>&1; then log 'self-test accepted extra directory'; fail=1; fi
  rmdir "$tmp/extra-dir"
  ln -s "$tmp/missing-target" "$tmp/extra-link"
  if require_fixture_entry_set "$tmp" "$FIXTURE_EXPECTED" >/dev/null 2>&1; then log 'self-test accepted extra symlink'; fail=1; fi
  mkdir "$tmp/expected"
  for entry in $FIXTURE_EXPECTED; do
    [[ "$entry" == manifest.json ]] || : > "$tmp/expected/$entry"
  done
  ln -s README.md "$tmp/expected/manifest.json"
  if require_fixture_entries_regular "$tmp/expected" >/dev/null 2>&1; then log 'self-test accepted expected manifest symlink'; fail=1; fi
  if require_evidence_path "$tmp/../dotdot-evidence" >/dev/null 2>&1; then log 'self-test accepted dotdot evidence path'; fail=1; fi
  good="$tmp/good.log"
  printf '%s\n' \
    "test $TEST_NAME ... ok" \
    'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s' \
    'APPLE_BF16_GEMM backend=neon-bf16 cases=3 max_abs_error=1.000000000e-4 atol=1.000000000e-3 rtol=0.000000000e0' \
    'APPLE_BF16_GEMM_PASS' > "$good"
  parse_test_log "$good" || fail=1
  for bad in duplicate malformed; do
    cp "$good" "$tmp/$bad.log"
    if [[ "$bad" == duplicate ]]; then printf '%s\n' 'APPLE_BF16_GEMM_PASS' >> "$tmp/$bad.log"; else sed 's/max_abs_error=1.000000000e-4/max_abs_error=nan/' "$good" > "$tmp/$bad.log"; fi
    if parse_test_log "$tmp/$bad.log" >/dev/null 2>&1; then log "self-test accepted $bad success log"; fail=1; fi
  done
  (( fail == 0 )) || return 1
  log 'self-test PASS'
}

EVIDENCE=''; EXPECTED_HEAD=''; SELF_TEST=0
while (($#)); do
  case "$1" in
    --self-test) ((SELF_TEST == 0)) || die 'duplicate --self-test'; SELF_TEST=1; shift;;
    --expected-head) [[ -z "$EXPECTED_HEAD" ]] || die 'duplicate --expected-head'; (($# >= 2)) || die '--expected-head requires a commit'; [[ "$2" =~ ^[0-9a-f]{40}$ ]] || die '--expected-head requires lowercase 40-hex'; EXPECTED_HEAD="$2"; shift 2;;
    --evidence-dir) (($# >= 2)) || die '--evidence-dir requires a path'; [[ -z "$EVIDENCE" ]] || die 'duplicate --evidence-dir'; EVIDENCE="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) usage; die "unknown option: $1";;
  esac
done
if ((SELF_TEST)); then [[ -z "$EVIDENCE$EXPECTED_HEAD" ]] || die '--self-test cannot accept run arguments'; self_test; exit $?; fi
[[ -n "$EXPECTED_HEAD" ]] || { usage; die '--expected-head is required'; }
[[ -n "$EVIDENCE" ]] || { usage; die '--evidence-dir is required'; }
require_host
cd "$VOKRA_ROOT"
require_clean_checkout "$EXPECTED_HEAD"
require_evidence_path "$EVIDENCE"
require_fixture_contract

log_file="$(mktemp "${TMPDIR:-/tmp}/bf16-apple-run.XXXXXX")"
trap 'rm -f -- "$log_file"' EXIT
CARGO_BUILD_JOBS=1 CARGO_NET_OFFLINE=true cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --offline --release -p vokra-backend-cpu --test bf16_gemm_torch_parity "$TEST_NAME" -- --ignored --exact --show-output --test-threads=1 2>&1 | tee "$log_file"
parse_test_log "$log_file"
actual_head="$(git -C "$VOKRA_ROOT" rev-parse HEAD)" || die 'cannot reread checkout HEAD before evidence'
[[ "$actual_head" == "$EXPECTED_HEAD" ]] || die 'checkout HEAD changed before evidence publication'
[[ -z "$(git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all)" ]] || die 'checkout became dirty before evidence publication'
mkdir "$EVIDENCE"
cp -p "$log_file" "$EVIDENCE/run.log"
{
  printf 'fixture_manifest_sha256=%s\nbackend=neon-bf16\nexpected_head=%s\ngit_commit=%s\nno_upload=NO_UPLOAD\n' "$MANIFEST_SHA256" "$EXPECTED_HEAD" "$actual_head"
  grep '^APPLE_BF16_GEMM backend=' "$log_file"
  grep '^APPLE_BF16_GEMM_PASS$' "$log_file"
} > "$EVIDENCE/backend.txt"
{
  echo "uname=$(uname -a)"
  echo "machine=$(sysctl -n hw.machine)"
  echo "physical_cpu=$(sysctl -n hw.physicalcpu)"
  echo "logical_cpu=$(sysctl -n hw.logicalcpu)"
  echo "memory_bytes=$(sysctl -n hw.memsize)"
  echo "bf16=$(sysctl -n hw.optional.arm.FEAT_BF16)"
  echo "expected_head=$EXPECTED_HEAD"
  echo "git_commit=$actual_head"
} > "$EVIDENCE/environment.txt"
actual_head="$(git -C "$VOKRA_ROOT" rev-parse HEAD)" || die 'cannot reread checkout HEAD before summary'
[[ "$actual_head" == "$EXPECTED_HEAD" ]] || die 'checkout HEAD changed before summary publication'
[[ -z "$(git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all)" ]] || die 'checkout became dirty before summary publication'
{
  echo 'status=PASS'
  echo 'verdict=PASS'
  echo "expected_head=$EXPECTED_HEAD"
  echo "git_commit=$actual_head"
  echo "fixture_manifest_sha256=$MANIFEST_SHA256"
  echo 'backend=neon-bf16'
  echo 'no_upload=NO_UPLOAD'
  grep '^APPLE_BF16_GEMM backend=' "$log_file"
  grep '^APPLE_BF16_GEMM_PASS$' "$log_file"
} > "$EVIDENCE/summary.txt"
log 'Apple arm64 NeonBf16/BFMMLA parity PASS'
