#!/usr/bin/env bash
# Apple Silicon Conv-TasNet verification using VAST-produced artifacts.
# CPU bounds are existing 2026-08-24 measurements; Metal is intentionally
# reported as MEASURED_NOT_GATED until a reviewed Metal bound exists.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VOKRA_ROOT="${VOKRA_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
PARITY_SOURCE="$VOKRA_ROOT/crates/vokra-models/tests/parity_conv_tasnet_real.rs"
PARITY_PROJECT="$VOKRA_ROOT/tools/parity/conv_tasnet"
PREFLIGHT_GATE="$PARITY_PROJECT/license_gate.py"
PREFLIGHT_MANIFEST="$PARITY_PROJECT/license_gate_manifest.json"

log() { printf '[conv-tasnet-apple] %s\n' "$*" >&2; }
die() { log "ERROR: $*"; return 2; }

usage() {
  cat <<'EOF'
usage: apple-silicon-conv-tasnet.sh --gguf <vast.gguf> --gguf-sha256 <64-hex> \
         --reference-dir <vast-fixtures> --reference-sha256 <64-hex> \
         --approval-evidence <file> --evidence-dir <absent-dir>
       apple-silicon-conv-tasnet.sh --self-test

Runs the existing CPU official bounds and explicit Metal execution against
VAST-produced artifacts on Darwin arm64. CPU is allowed to use the reviewed
2026-08-24 bounds; Metal remains MEASURED_NOT_GATED and no Metal PASS is
manufactured. This verifier never downloads, converts, uploads, or publishes.
EOF
}

sha256_file() { shasum -a 256 "$1" | awk '{print $1}'; }

require_absent_directory() {
  local directory="$1"
  [[ ! -e "$directory" && ! -L "$directory" ]] || die 'evidence directory must be absent before validation'
}

canonical_existing_path() {
  local target="$1" lexical current="/" component parent base
  [[ "$target" = /* ]] || target="$PWD/$target"
  lexical="${target#/}"
  while [[ -n "$lexical" ]]; do
    component="${lexical%%/*}"
    if [[ "$lexical" == "$component" ]]; then lexical=""; else lexical="${lexical#*/}"; fi
    [[ "$component" == "." || -z "$component" ]] && continue
    [[ "$component" != ".." ]] || { die 'input path contains ..'; return 2; }
    current="${current%/}/$component"
    [[ ! -L "$current" ]] || { die 'input path contains a symlinked component'; return 2; }
  done
  [[ -e "$target" && ! -L "$target" ]] || { die 'input is not a regular non-symlink path'; return 2; }
  if [[ -d "$target" ]]; then (cd -P "$target" && pwd) || { die 'input directory is inaccessible'; return 2; }
  else
    parent="$(dirname "$target")"; base="$(basename "$target")"
    parent="$(cd -P "$parent" 2>/dev/null && pwd)" || { die 'input parent is inaccessible'; return 2; }
    printf '%s/%s\n' "$parent" "$base"
  fi
}

paths_overlap() {
  local left="$1" right="$2"
  [[ "$left" == "$right" || "$left/" == "$right/"* || "$right/" == "$left/"* ]]
}

canonical_absent_path() {
  local target="$1" lexical current="/" component suffix="" real
  [[ "$target" = /* ]] || target="$PWD/$target"
  lexical="${target#/}"
  while [[ -n "$lexical" ]]; do
    component="${lexical%%/*}"
    if [[ "$lexical" == "$component" ]]; then lexical=""; else lexical="${lexical#*/}"; fi
    [[ "$component" == "." || -z "$component" ]] && continue
    [[ "$component" != ".." ]] || { die 'evidence path contains ..'; return 2; }
    current="${current%/}/$component"
    [[ ! -L "$current" ]] || { die 'evidence path contains a symlinked component'; return 2; }
  done
  current="$target"
  while [[ ! -e "$current" && ! -L "$current" ]]; do
    component="$(basename "$current")"; suffix="/$component$suffix"; current="$(dirname "$current")"
  done
  [[ -d "$current" && ! -L "$current" ]] || { die 'evidence parent is missing or symlinked'; return 2; }
  real="$(cd -P "$current" 2>/dev/null && pwd)" || { die 'evidence parent is inaccessible'; return 2; }
  printf '%s%s\n' "$real" "$suffix"
}

require_disjoint_evidence() {
  local evidence="$1" gguf_path="$2" reference="$3" approval="$4" root_real evidence_real
  root_real="$(canonical_existing_path "$VOKRA_ROOT")" || return 2
  canonical_existing_path "$gguf_path" >/dev/null || return 2
  canonical_existing_path "$reference" >/dev/null || return 2
  canonical_existing_path "$approval" >/dev/null || return 2
  evidence_real="$(canonical_absent_path "$evidence")" || return 2
  local protected
  for protected in "$root_real" "$(canonical_existing_path "$gguf_path")" "$(canonical_existing_path "$reference")" "$(canonical_existing_path "$approval")"; do
    if paths_overlap "$evidence_real" "$protected"; then die 'evidence directory overlaps a protected input'; return 2; fi
  done
}

require_reference_contract() {
  local directory="$1"
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python - "$directory" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected = {
    "manifest.json": None,
    "pcm.f32.bin": (16384, "9afd5a533c5834708f3c10e0019b0802d354771f040fe42cef96564e30410455", [4096]),
    "encoder.f32.bin": (522240, "de8a252b60fb37b2232d0855a578b81c3ed3d0d5e4c97a0addf78596d7f07561", [512, 255]),
    "bottleneck.f32.bin": (130560, "54a7b62a68bd8d8f93cd5ca6860f6853ba542a97df95a2fac3583d9adeed173f", [128, 255]),
    "mask.f32.bin": (522240, "dec7369da1040f54183c80e15616ef1a3a8a91eb4fb42c74f3073bc7524bf255", [512, 255]),
    "separated.f32.bin": (16384, "c9d63ed4633c73487d7c4f4a0bbffae68478dcbd1b71f363e408d539a2d55f9b", [4096]),
}
def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""): result.update(chunk)
    return result.hexdigest()
if {entry.name for entry in root.iterdir()} != set(expected): raise SystemExit("reference file set drifted")
for name, identity in expected.items():
    path = root / name
    if path.is_symlink() or not path.is_file(): raise SystemExit(f"{name}: not regular")
    if identity is not None and (path.stat().st_size != identity[0] or digest(path) != identity[1]): raise SystemExit(f"{name}: artifact identity drifted")
data = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
if not isinstance(data, dict): raise SystemExit("reference manifest must be an object")
if set(data) != {"format", "model_id", "revision", "checkpoint", "sample_rate", "pcm_samples", "shapes", "python", "numpy", "torch", "asteroid", "runtime_status", "parity_status", "tolerance", "artifacts"}: raise SystemExit("reference manifest schema drifted")
if data["format"] != "vokra-conv-tasnet-reference-v1" or data["model_id"] != "JorisCos/ConvTasNet_Libri1Mix_enhsingle_16k" or data["revision"] != "bb8a876bc157b5cf3c405994accb798c49146016": raise SystemExit("reference source identity drifted")
if not isinstance(data["checkpoint"], dict) or set(data["checkpoint"]) != {"path", "sha256", "identity"} or data["checkpoint"]["sha256"] != "dd8ddefe95a35761f8a48643a618eba908572d04d33208a8ed5451fb5a4378d0": raise SystemExit("checkpoint identity drifted")
if data["sample_rate"] != 16000 or data["pcm_samples"] != 4096 or not isinstance(data["python"], str) or not data["python"].startswith("3.12."): raise SystemExit("reference runtime contract drifted")
if data["numpy"] != "2.3.5" or data["torch"] != "2.9.1+cpu" or data["asteroid"] != "0.7.0": raise SystemExit("reference dependency versions drifted")
if data["runtime_status"] != "MEASURED_NOT_GATED" or data["parity_status"] != "MEASURED_NOT_GATED" or data["tolerance"] is not None: raise SystemExit("reference status drifted")
if data["shapes"] != {"encoded": [1, 512, 255], "bottleneck": [1, 128, 255], "masks": [1, 1, 512, 255], "decoded": [1, 1, 4096], "separated": [1, 1, 4096]}: raise SystemExit("reference shapes drifted")
if set(data["artifacts"]) != set(expected) - {"manifest.json"}: raise SystemExit("artifact manifest set drifted")
for name, identity in expected.items():
    if identity is None: continue
    size, sha, shape = identity
    row = data["artifacts"][name]
    if row != {"bytes": size, "sha256": sha, "shape": shape, "dtype": "float32-le"}: raise SystemExit(f"{name}: artifact manifest drifted")
PY
}

require_test_evidence() {
  local log_file="$1" test_count named_count result_count result_lines marker_family raw_marker_count mask_count waveform_count
  test_count="$(awk '/^test / && $0 !~ /^test result:/ {count++} END {print count + 0}' "$log_file")"
  named_count="$(grep -Ecx '^test converted_official_checkpoint_matches_asteroid \.\.\. ok$' "$log_file" || true)"
  result_count="$(grep -Ecx '^test result: ok\. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out(; finished in [0-9]+\.[0-9]+s)?$' "$log_file" || true)"
  result_lines="$(grep -Ec '^test result:' "$log_file" || true)"
  marker_family="$(grep -Ecx '^CONV_TASNET_METAL_CPU (mask|waveform) shape=[0-9]+ max_abs=[0-9]+\.[0-9]{9}e[+-][0-9]+ mean_abs=[0-9]+\.[0-9]{9}e[+-][0-9]+ relative_l1=[0-9]+\.[0-9]{9}e[+-][0-9]+ verdict=MEASURED_NOT_GATED$' "$log_file" || true)"
  raw_marker_count="$(grep -Ec 'CONV_TASNET_METAL_CPU' "$log_file" || true)"
  mask_count="$(grep -Ec '^CONV_TASNET_METAL_CPU mask shape=' "$log_file" || true)"
  waveform_count="$(grep -Ec '^CONV_TASNET_METAL_CPU waveform shape=' "$log_file" || true)"
  [[ "$test_count" == 1 && "$named_count" == 1 && "$result_count" == 1 && "$result_lines" == 1 && "$marker_family" == 2 && "$raw_marker_count" == 2 && "$mask_count" == 1 && "$waveform_count" == 1 ]] || { die 'test/result/Metal evidence is not exact'; return 2; }
}

self_test() {
  local path="${BASH_SOURCE[0]}" fail=0 token temporary log_file
  for token in 'VOKRA_REMOTE_APPLE_SILICON=1' 'Darwin' 'arm64' 'xcrun -f metal' \
    'MEASURED_NOT_GATED' '2026-08-24' 'parity_conv_tasnet_real.rs' \
    'cargo test --locked -p vokra-models --features metal' \
    'CONV_TASNET_METAL_CPU' 'verdict=MEASURED_NOT_GATED' \
    'cpu_official_gate=PASS_WITH_EXISTING_MEASURED_BOUNDS' \
    'metal_status=MEASURED_NOT_GATED' 'git status --porcelain'; do
    if ! grep -Fq -- "$token" "$path" && ! grep -Fq -- "$token" "$PARITY_SOURCE"; then
      log "self-test FAIL: missing contract token: $token"
      fail=1
    fi
  done
  if grep -En '(^|[[:space:]])(curl|wget|pip|.*convert|.*upload|.*publish|git[[:space:]]+push)([[:space:]]|$)' "$path" >/dev/null; then
    log 'self-test FAIL: acquisition/conversion/publication command found'
    fail=1
  fi
  if grep -En '^[[:space:]]*(printf|echo)[^#]*METAL[^#]*PASS' "$path" >/dev/null; then
    log 'self-test FAIL: verifier manufactures a Metal PASS marker'
    fail=1
  fi
  if "$path" --self-test --gguf /tmp/not-accepted >/dev/null 2>&1; then
    log 'self-test FAIL: extra argument accepted'
    fail=1
  fi
  if "$path" --unknown-flag >/dev/null 2>&1; then
    log 'self-test FAIL: unknown argument accepted'
    fail=1
  fi
  if "$path" --gguf a --gguf b >/dev/null 2>&1; then
    log 'self-test FAIL: duplicate GGUF option accepted'
    fail=1
  fi
  if "$path" --gguf a --gguf-sha256 0 >/dev/null 2>&1; then
    log 'self-test FAIL: incomplete required options accepted'
    fail=1
  fi
  temporary="$(cd -P "$(mktemp -d)" && pwd)"
  trap 'rm -rf "$temporary"' RETURN
  mkdir "$temporary/reference"; printf '%s\n' fixture > "$temporary/gguf"; printf '%s\n' approval > "$temporary/approval"
  if require_absent_directory "$temporary/reference" >/dev/null 2>&1; then
    log 'self-test FAIL: existing evidence directory accepted'; fail=1
  fi
  log_file="$temporary/parity.log"
  printf '%s\n' \
    'test converted_official_checkpoint_matches_asteroid ... ok' \
    'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s' \
    'CONV_TASNET_METAL_CPU mask shape=130560 max_abs=1.000000000e-03 mean_abs=1.000000000e-04 relative_l1=1.000000000e-04 verdict=MEASURED_NOT_GATED' \
    'CONV_TASNET_METAL_CPU waveform shape=4096 max_abs=1.000000000e-03 mean_abs=1.000000000e-04 relative_l1=1.000000000e-04 verdict=MEASURED_NOT_GATED' > "$log_file"
  if require_test_evidence "$log_file"; then :; else log 'self-test FAIL: valid test evidence rejected'; fail=1; fi
  printf '%s\n' 'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out' >> "$log_file"
  if require_test_evidence "$log_file" >/dev/null 2>&1; then log 'self-test FAIL: duplicate result accepted'; fail=1; fi
  sed 's/verdict=MEASURED_NOT_GATED$/verdict=FAIL/' "$log_file" > "$temporary/bad-marker.log"
  if require_test_evidence "$temporary/bad-marker.log" >/dev/null 2>&1; then log 'self-test FAIL: failed marker accepted'; fail=1; fi
  sed 's/^CONV_TASNET_METAL_CPU /prefix CONV_TASNET_METAL_CPU /' "$log_file" > "$temporary/prefix-marker.log"
  if require_test_evidence "$temporary/prefix-marker.log" >/dev/null 2>&1; then log 'self-test FAIL: prefixed marker accepted'; fail=1; fi
  trap - RETURN
  rm -rf "$temporary"
  (( fail == 0 )) || return 1
  log 'self-test PASS'
}

gguf=''
gguf_sha256=''
reference_dir=''
reference_sha256=''
approval_evidence=''
evidence_dir=''
self=0
seen_gguf=0; seen_gguf_sha=0; seen_reference=0; seen_reference_sha=0; seen_approval=0; seen_evidence=0
while (($#)); do
  case "$1" in
    --self-test) (( self == 0 )) || die 'duplicate --self-test'; self=1; shift ;;
    --gguf) (( seen_gguf == 0 )) || die 'duplicate --gguf'; (( $# >= 2 )) && [[ -n "$2" && "$2" != -* ]] || die '--gguf requires a nonempty path'; seen_gguf=1; gguf="$2"; shift 2 ;;
    --gguf-sha256) (( seen_gguf_sha == 0 )) || die 'duplicate --gguf-sha256'; (( $# >= 2 )) && [[ "$2" =~ ^[0-9a-f]{64}$ ]] || die '--gguf-sha256 requires lowercase 64-hex'; seen_gguf_sha=1; gguf_sha256="$2"; shift 2 ;;
    --reference-dir) (( seen_reference == 0 )) || die 'duplicate --reference-dir'; (( $# >= 2 )) && [[ -n "$2" && "$2" != -* ]] || die '--reference-dir requires a nonempty path'; seen_reference=1; reference_dir="$2"; shift 2 ;;
    --reference-sha256) (( seen_reference_sha == 0 )) || die 'duplicate --reference-sha256'; (( $# >= 2 )) && [[ "$2" =~ ^[0-9a-f]{64}$ ]] || die '--reference-sha256 requires lowercase 64-hex'; seen_reference_sha=1; reference_sha256="$2"; shift 2 ;;
    --approval-evidence) (( seen_approval == 0 )) || die 'duplicate --approval-evidence'; (( $# >= 2 )) && [[ -n "$2" && "$2" != -* ]] || die '--approval-evidence requires a nonempty path'; seen_approval=1; approval_evidence="$2"; shift 2 ;;
    --evidence-dir) (( seen_evidence == 0 )) || die 'duplicate --evidence-dir'; (( $# >= 2 )) && [[ -n "$2" && "$2" != -* ]] || die '--evidence-dir requires a nonempty path'; seen_evidence=1; evidence_dir="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done
if (( self )); then
  [[ "$seen_gguf$seen_gguf_sha$seen_reference$seen_reference_sha$seen_approval$seen_evidence" == 000000 ]] || die '--self-test accepts no other arguments'
  self_test
  exit $?
fi

[[ "$seen_gguf$seen_gguf_sha$seen_reference$seen_reference_sha$seen_approval$seen_evidence" == 111111 ]] || die '--gguf, hashes, approval, reference, and evidence paths are required'
UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 "$PREFLIGHT_GATE" \
  --lock "$PARITY_PROJECT/uv.lock" --project "$PARITY_PROJECT/pyproject.toml" \
  --manifest "$PREFLIGHT_MANIFEST" --evidence "$approval_evidence"
[[ "${VOKRA_REMOTE_APPLE_SILICON:-0}" == 1 ]] || die 'VOKRA_REMOTE_APPLE_SILICON=1 is absent'
[[ "$(uname -s)" == Darwin ]] || die 'real Metal verification requires Darwin'
[[ "$(uname -m)" == arm64 ]] || die 'real Metal verification requires Apple arm64'
command -v xcrun >/dev/null 2>&1 || die 'xcrun is unavailable'
xcrun -f metal >/dev/null 2>&1 || die 'Xcode Metal compiler is unavailable'
[[ -f "$gguf" && ! -L "$gguf" && -s "$gguf" ]] || die 'VAST GGUF is missing, empty, or symlinked'
[[ "$(sha256_file "$gguf")" == "$gguf_sha256" ]] || die 'VAST GGUF SHA-256 does not match the VAST argument'
[[ -d "$reference_dir" ]] || die 'VAST reference directory is missing'
[[ ! -L "$reference_dir" ]] || die 'VAST reference directory is symlinked'
[[ -f "$PARITY_SOURCE" ]] || die 'Conv-TasNet parity source is missing'
[[ -f "$VOKRA_ROOT/Cargo.toml" && -d "$VOKRA_ROOT/.git" ]] || die 'not a Vokra checkout'
[[ -z "$(git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all)" ]] || die 'Apple checkout must be clean'
require_absent_directory "$evidence_dir"
require_disjoint_evidence "$evidence_dir" "$gguf" "$reference_dir" "$approval_evidence"

for name in pcm.f32.bin encoder.f32.bin bottleneck.f32.bin mask.f32.bin separated.f32.bin; do
  [[ -s "$reference_dir/$name" ]] || die "reference fixture missing: $name"
done
[[ "$(sha256_file "$reference_dir/pcm.f32.bin")" == 9afd5a533c5834708f3c10e0019b0802d354771f040fe42cef96564e30410455 ]] || die 'PCM fixture hash drift'
[[ "$(sha256_file "$reference_dir/encoder.f32.bin")" == de8a252b60fb37b2232d0855a578b81c3ed3d0d5e4c97a0addf78596d7f07561 ]] || die 'encoder fixture hash drift'
[[ "$(sha256_file "$reference_dir/bottleneck.f32.bin")" == 54a7b62a68bd8d8f93cd5ca6860f6853ba542a97df95a2fac3583d9adeed173f ]] || die 'bottleneck fixture hash drift'
[[ "$(sha256_file "$reference_dir/mask.f32.bin")" == dec7369da1040f54183c80e15616ef1a3a8a91eb4fb42c74f3073bc7524bf255 ]] || die 'mask fixture hash drift'
[[ "$(sha256_file "$reference_dir/separated.f32.bin")" == c9d63ed4633c73487d7c4f4a0bbffae68478dcbd1b71f363e408d539a2d55f9b ]] || die 'separated fixture hash drift'
[[ -f "$reference_dir/manifest.json" && ! -L "$reference_dir/manifest.json" ]] || die 'reference manifest is missing or symlinked'
[[ "$(sha256_file "$reference_dir/manifest.json")" == "$reference_sha256" ]] || die 'reference manifest SHA-256 does not match the VAST argument'
require_reference_contract "$reference_dir"

export VOKRA_CONV_TASNET_GGUF="$gguf"
export CARGO_BUILD_JOBS=1
mkdir -p "$evidence_dir"
cargo test --locked -p vokra-models --features metal --test parity_conv_tasnet_real -- --nocapture \
  > "$evidence_dir/parity.log" 2>&1
require_test_evidence "$evidence_dir/parity.log"
{
  echo "git_commit=$(git -C "$VOKRA_ROOT" rev-parse HEAD)"
  echo "gguf_sha256=$(sha256_file "$gguf")"
  echo 'runtime_status=MEASURED_NOT_GATED'
  echo 'parity_status=MEASURED_NOT_GATED'
  echo 'cpu_official_gate=PASS_WITH_EXISTING_MEASURED_BOUNDS'
  echo 'metal_status=MEASURED_NOT_GATED'
  echo 'metal_pass_not_claimed=true'
} > "$evidence_dir/summary.txt"
log "Apple verification complete: CPU bounds exercised; Metal remains MEASURED_NOT_GATED"
