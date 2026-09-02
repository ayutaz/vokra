#!/usr/bin/env bash
# Apple Silicon SGMSE CPU/Metal parity verifier. Inputs are staged artifacts;
# this script performs no download, conversion, upload, or publication.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VOKRA_ROOT="${VOKRA_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
PARITY_PROJECT="$VOKRA_ROOT/tools/parity"
VERIFY_TOOL="$PARITY_PROJECT/sgmse_verify_reference.py"
TEST_NAME="sgmse_apple_cpu_metal_score_matches_reference"
MIN_MEMORY_BYTES=16000000000
MIN_FREE_DISK_KIB=5000000

log() { printf '[sgmse-apple] %s\n' "$*" >&2; }
die() { log "ERROR: $*"; return 2; }
usage() { cat >&2 <<'EOF'
usage: apple-silicon-sgmse.sh --gguf ABS --gguf-sha256 HEX64 \
  --reference ABS_DIR --evidence-dir ABSENT_DIR
       apple-silicon-sgmse.sh --self-test

Runs exactly one ignored vokra-models SGMSE CPU/Metal score parity test on a
disposable Scaleway Darwin/arm64 worker. The GGUF/reference packet must already
be staged. No download, conversion, upload, or publication is performed.
EOF
}

sha256_file() { shasum -a 256 "$1" | awk '{print $1}'; }
require_abs() { [[ "$1" == /* ]] || die "$2 must be absolute"; }
reject_symlink_ancestry() { local path="$1" label="$2" current="$1"; while :; do [[ ! -L "$current" ]] || die "$label has symlink ancestry: $current"; [[ "$current" == / ]] && break; current="$(dirname "$current")"; done; }
canonical_uncreated() { local path="$1" name parent suffix=''; while [[ ! -d "$path" || -L "$path" ]]; do name="${path##*/}"; parent="${path%/*}"; [[ "$parent" == "$path" ]] && parent=/; path="$parent"; [[ -n "$name" ]] && suffix="/${name}${suffix}"; done; (cd -P "$path" && printf '%s%s\n' "$PWD" "$suffix"); }
disjoint() { local a b; a="$(canonical_uncreated "$1")"; b="$(canonical_uncreated "$2")"; [[ "$a" != "$b" && "$a" != "$b"/* && "$b" != "$a"/* ]] || die "protected paths overlap"; }

require_host() {
  [[ "${VOKRA_REMOTE_APPLE_SILICON:-0}" == 1 ]] || die 'VOKRA_REMOTE_APPLE_SILICON=1 is absent'
  [[ "$(uname -s)" == Darwin ]] || die 'requires Darwin'
  [[ "$(uname -m)" == arm64 ]] || die 'requires arm64'
  local memory disk; memory="$(sysctl -n hw.memsize)"; disk="$(df -Pk "$VOKRA_ROOT" | awk 'NR==2 {print $4}')"
  [[ "$memory" =~ ^[0-9]+$ && "$memory" -ge "$MIN_MEMORY_BYTES" ]] || die 'insufficient physical memory'
  [[ "$disk" =~ ^[0-9]+$ && "$disk" -ge "$MIN_FREE_DISK_KIB" ]] || die 'insufficient free disk'
  xcrun -f metal >/dev/null 2>&1 || die 'Xcode Metal compiler unavailable'
}

run_self_test() {
  local script="${BASH_SOURCE[0]}" fail=0 token
  for token in 'VOKRA_REMOTE_APPLE_SILICON=1' 'Darwin' 'arm64' 'xcrun -f metal' \
    'sgmse_verify_reference.py' 'CARGO_BUILD_JOBS=1' 'SGMSE_APPLE_SCORE_PARITY' \
    'backend=cpu,metal' 'metal_device=present' 'atol=0.01' \
    'cargo test --locked --features metal -p vokra-models --test sgmse_apple_score' \
    '-- --ignored --exact --show-output' 'shasum -a 256' 'no download' 'no upload'; do
    grep -Fq -- "$token" "$script" || { log "self-test missing contract token: $token"; fail=1; }
  done
  grep -En 'git[[:space:]]+push|publish-one\.sh|huggingface-cli[[:space:]]+upload|--push|curl[[:space:]]|wget[[:space:]]' "$script" | grep -v 'grep -En' >/dev/null && { log 'self-test publication/network command found'; fail=1; } || true
  if VOKRA_REMOTE_APPLE_SILICON=1 "$script" --self-test --gguf /tmp/rejected >/dev/null 2>&1; then log 'self-test accepted extra argument'; fail=1; fi
  if "$script" --unknown >/dev/null 2>&1; then log 'self-test accepted unknown argument'; fail=1; fi
  (( fail == 0 )) || return 1
  log 'self-test PASS'
}

GGUF='' GGUF_SHA='' REFERENCE='' EVIDENCE='' SELF_TEST=0
while (($#)); do case "$1" in
  --self-test) ((SELF_TEST==0)) || die 'duplicate --self-test'; SELF_TEST=1; shift;;
  --gguf) (($#>=2)) || die '--gguf requires a path'; GGUF="$2"; shift 2;;
  --gguf-sha256) (($#>=2)) || die '--gguf-sha256 requires a digest'; GGUF_SHA="$2"; shift 2;;
  --reference) (($#>=2)) || die '--reference requires a path'; REFERENCE="$2"; shift 2;;
  --evidence-dir) (($#>=2)) || die '--evidence-dir requires a path'; EVIDENCE="$2"; shift 2;;
  -h|--help) usage; exit 0;; *) die "unknown argument: $1";; esac; done
if ((SELF_TEST)); then [[ -z "$GGUF$GGUF_SHA$REFERENCE$EVIDENCE" ]] || die '--self-test accepts no other arguments'; run_self_test; exit $?; fi
[[ -n "$GGUF" && -n "$GGUF_SHA" && -n "$REFERENCE" && -n "$EVIDENCE" ]] || { usage; exit 1; }
require_host
[[ -d "$VOKRA_ROOT/.git" && -f "$VOKRA_ROOT/Cargo.toml" ]] || die 'not a Vokra checkout'
[[ -z "$(git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all)" ]] || die 'checkout must be clean'
for command in cargo git shasum awk df sysctl xcrun uv find grep tee mktemp; do command -v "$command" >/dev/null 2>&1 || die "missing tool: $command"; done
[[ "$GGUF_SHA" =~ ^[0-9a-f]{64}$ ]] || die '--gguf-sha256 must be lowercase hex64'
require_abs "$GGUF" GGUF; require_abs "$REFERENCE" reference; require_abs "$EVIDENCE" evidence
reject_symlink_ancestry "$GGUF" GGUF; reject_symlink_ancestry "$REFERENCE" reference; assert_parent="$(dirname "$EVIDENCE")"; reject_symlink_ancestry "$assert_parent" evidence-parent
[[ -f "$GGUF" && ! -L "$GGUF" ]] || die 'GGUF missing or symlinked'; [[ -d "$REFERENCE" && ! -L "$REFERENCE" ]] || die 'reference missing or symlinked'; [[ ! -e "$EVIDENCE" && ! -L "$EVIDENCE" ]] || die 'evidence directory must be absent'; [[ -d "$assert_parent" ]] || die 'evidence parent missing'
disjoint "$GGUF" "$REFERENCE"; disjoint "$GGUF" "$EVIDENCE"; disjoint "$REFERENCE" "$EVIDENCE"
[[ "$(sha256_file "$GGUF")" == "$GGUF_SHA" ]] || die 'GGUF SHA-256 mismatch'
manifest_sha="$(sha256_file "$REFERENCE/manifest.json")"
UV_NO_CACHE=1 uv run --frozen --no-sync --project "$PARITY_PROJECT" --python 3.12 python "$VERIFY_TOOL" --manifest "$REFERENCE/manifest.json" --output-dir "$REFERENCE" --vokra-root "$VOKRA_ROOT" >/dev/null
log_file="$(mktemp "${TMPDIR:-/tmp}/sgmse-apple.XXXXXX")"; trap 'rm -f -- "$log_file"' EXIT
export VOKRA_SGMSE_GGUF="$GGUF" VOKRA_SGMSE_GGUF_SHA256="$GGUF_SHA" VOKRA_SGMSE_REFERENCE_DIR="$REFERENCE" VOKRA_SGMSE_REFERENCE_MANIFEST_SHA256="$manifest_sha" VOKRA_SGMSE_APPLE_EVIDENCE_DIR="$EVIDENCE" VOKRA_REMOTE_APPLE_SILICON=1
CARGO_BUILD_JOBS=1 cargo test --locked --features metal -p vokra-models --test sgmse_apple_score "$TEST_NAME" -- --ignored --exact --show-output 2>&1 | tee "$log_file"
[[ "$(grep -Ec "^test $TEST_NAME \.\.\. ok$" "$log_file" || true)" == 1 ]] || die 'Apple SGMSE named test did not pass exactly once'
[[ "$(grep -Ec '^test [^ ]+ \.\.\.' "$log_file" || true)" == 1 ]] || die 'Apple SGMSE Cargo emitted more than one test line'
[[ "$(grep -Ec '^test result:' "$log_file" || true)" == 1 ]] || die 'Apple SGMSE Cargo emitted more than one result line'
[[ "$(grep -Ec '^test result: ok\. 1 passed; 0 failed; 0 ignored; 0 measured; [0-9]+ filtered out' "$log_file" || true)" == 1 ]] || die 'Apple SGMSE Cargo result was not exactly one pass'
grep -Fq 'SGMSE_APPLE_SCORE_PARITY backend=cpu+metal' "$log_file" || die 'Apple SGMSE parity sentinel missing'
[[ -d "$EVIDENCE" ]] || die 'evidence directory was not created'
[[ "$(find "$EVIDENCE" -mindepth 1 -maxdepth 1 -exec basename {} \; | sort | tr '\n' ' ')" == 'backend.txt cpu_score_imag.f32 cpu_score_real.f32 metal_score_imag.f32 metal_score_real.f32 ' ]] || die 'evidence file set is not exact'
for evidence_file in cpu_score_real.f32 cpu_score_imag.f32 metal_score_real.f32 metal_score_imag.f32; do [[ "$(wc -c < "$EVIDENCE/$evidence_file" | tr -d '[:space:]')" == 65536 && ! -L "$EVIDENCE/$evidence_file" ]] || die "invalid evidence score file: $evidence_file"; done
[[ -f "$EVIDENCE/backend.txt" && ! -L "$EVIDENCE/backend.txt" ]] || die 'backend evidence is missing or symlinked'
grep -Fq "reference_manifest_sha256=$manifest_sha" "$EVIDENCE/backend.txt" || die 'manifest identity missing from evidence'
log 'SGMSE Apple CPU/Metal parity PASS'
