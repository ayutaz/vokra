#!/usr/bin/env bash
# Apple Silicon Charsiu CPU/reference/Metal parity verifier.
# Inputs are VAST-produced and authenticated; this worker performs no
# download, conversion, publication, upload, or implicit CPU fallback.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VOKRA_ROOT="${VOKRA_ROOT:-$DEFAULT_ROOT}"
TEST_SOURCE="$VOKRA_ROOT/crates/vokra-models/tests/charsiu_apple_cpu_metal.rs"
TEST_NAME="charsiu_apple_cpu_metal_matches_reference"
GGUF_ENV="VOKRA_CHARSIU_GGUF"
REFERENCE_ENV="VOKRA_CHARSIU_REFERENCE_DIR"
MIN_MEMORY_BYTES=16000000000
MIN_FREE_DISK_KIB=5000000
CHECKPOINT_SHA256="6dc8a18422db7c22e951d5f72dc2afc267b942eb0b8459ac6dcc0cf412536de1"
CONFIG_SHA256="7406aa4f917267640865688aa62f2337664a3abb9a49a2f204d932b53aeb6cb7"
UPSTREAM_REPO="charsiu/en_w2v2_fc_10ms"
UPSTREAM_REVISION="e9bf8dd314313fc57f6e4d0b5425bde4bbeac80f"
CPU_REFERENCE_ATOL="0.000200000"
METAL_ATOL="0.010000000"
METAL_ATOL_REGEX="${METAL_ATOL//./\\.}"

log() { printf '[charsiu-apple] %s\n' "$*" >&2; }
die() { log "ERROR: $*"; return 2; }

usage() {
  cat >&2 <<'EOF'
usage: apple-silicon-charsiu.sh --gguf ABS --gguf-sha256 HEX64 \
  --reference ABS_DIR --reference-manifest-sha256 HEX64 \
  --vast-evidence ABS_DIR --vast-evidence-sha256 HEX64 \
  --expected-head HEX40 --evidence-dir ABS_ABSENT_DIR
       apple-silicon-charsiu.sh --self-test

Consumes a VAST-produced Charsiu GGUF, official Transformers reference, and
CPU evidence, then runs one explicitly selected ignored real-weight CPU/Metal
parity test with --ignored --exact on a Darwin arm64 host. It never downloads,
converts, uploads, publishes, or uses a CPU fallback for the Metal leg.
EOF
}

sha256_file() { shasum -a 256 "$1" | awk '{print $1}'; }
require_file() {
  local label="$1" path="$2"
  [[ -f "$path" && ! -L "$path" && -s "$path" ]] \
    || die "$label is missing, empty, symlinked, or non-regular: $path"
}
require_absolute() { [[ "$2" == /* ]] || die "$1 must be absolute: $2"; }
reject_symlink_ancestry() {
  local path="$1" label="$2" current="$1"
  while :; do
    [[ ! -L "$current" ]] || die "$label has symlink ancestry: $current"
    [[ "$current" == / ]] && break
    current="$(dirname "$current")"
  done
}
canonical_existing() {
  local path="$1" parent
  [[ -e "$path" && ! -L "$path" ]] || return 1
  if [[ -d "$path" ]]; then (cd -P "$path" && pwd)
  else parent="$(dirname "$path")"; (cd -P "$parent" && printf '%s/%s\n' "$PWD" "$(basename "$path")"); fi
}
canonical_absent() {
  local path="$1" suffix='' name parent
  [[ ! -e "$path" && ! -L "$path" ]] || return 1
  while [[ ! -d "$path" || -L "$path" ]]; do
    name="$(basename "$path")"; [[ -n "$name" ]] && suffix="/$name$suffix"
    parent="$(dirname "$path")"; [[ "$parent" != "$path" ]] || return 1; path="$parent"
  done
  (cd -P "$path" && printf '%s%s\n' "$PWD" "$suffix")
}
paths_overlap() { [[ "$1" == "$2" || "$1" == "$2/"* || "$2" == "$1/"* ]]; }
reserve_evidence_dir() {
  local target="$1" target_real protected protected_real
  require_absolute evidence "$target"
  [[ ! -e "$target" && ! -L "$target" ]] || die 'evidence directory must be absent'
  [[ -d "$(dirname "$target")" ]] || die 'evidence parent directory is missing'
  reject_symlink_ancestry "$target" evidence
  target_real="$(canonical_absent "$target")" || die 'evidence path cannot be canonicalized'
  for protected in "$VOKRA_ROOT" "$GGUF" "$REFERENCE" "$VAST_EVIDENCE"; do
    protected_real="$(canonical_existing "$protected")" || die "protected input cannot be canonicalized: $protected"
    paths_overlap "$target_real" "$protected_real" && die "evidence overlaps protected input: $protected"
  done
  case "$target_real" in "$VOKRA_ROOT"/*|"$VOKRA_ROOT") die 'evidence must be outside the checkout' ;; esac
  mkdir -m 700 "$target" || die 'evidence directory appeared during reservation'
}
require_apple_host() {
  [[ "${VOKRA_REMOTE_APPLE_SILICON:-0}" == 1 ]] || die 'VOKRA_REMOTE_APPLE_SILICON=1 is required'
  [[ "$(uname -s)" == Darwin && "$(uname -m)" == arm64 ]] || die 'requires Darwin arm64'
  local memory disk; memory="$(sysctl -n hw.memsize)"; disk="$(df -Pk "$VOKRA_ROOT" | awk 'NR == 2 {print $4}')"
  [[ "$memory" =~ ^[0-9]+$ && "$memory" -ge "$MIN_MEMORY_BYTES" ]] || die 'insufficient physical memory'
  [[ "$disk" =~ ^[0-9]+$ && "$disk" -ge "$MIN_FREE_DISK_KIB" ]] || die 'insufficient free disk'
  xcrun -f metal >/dev/null 2>&1 || die 'Xcode Metal compiler is unavailable'
}
require_tooling() {
  local tool actual
  for tool in cargo rustc git shasum awk find grep tee sysctl sw_vers system_profiler xcrun uv df wc tr sort mktemp date \
    uname dirname basename mkdir cp sed pwd; do
    command -v "$tool" >/dev/null 2>&1 || die "required tool missing: $tool"
  done
  [[ -d "$VOKRA_ROOT/.git" && -f "$VOKRA_ROOT/Cargo.toml" && -f "$TEST_SOURCE" && ! -L "$TEST_SOURCE" ]] || die 'Vokra checkout or Charsiu Apple test source is missing'
  actual="$(git -C "$VOKRA_ROOT" rev-parse HEAD)"
  [[ "$actual" == "$EXPECTED_HEAD" ]] || die 'checkout HEAD does not match --expected-head'
  [[ -z "$(git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all)" ]] || die 'Apple checkout must be clean'
  xcrun -f metal >/dev/null 2>&1 || die 'Xcode Metal compiler is unavailable'
}

require_reference() {
  local directory="$1" entry name count=0
  [[ -d "$directory" && ! -L "$directory" ]] || die 'reference directory is missing or symlinked'
  reject_symlink_ancestry "$directory" reference
  while IFS= read -r entry; do
    name="$(basename "$entry")"
    case "$name" in manifest.json|pcm_400.f32.bin|logits_1x42.f32.bin) ;; *) die "unexpected reference entry: $name" ;; esac
    count=$((count + 1))
  done < <(find -P "$directory" -mindepth 1 -maxdepth 1 -print)
  [[ "$count" == 3 ]] || die 'reference must contain exactly three top-level files'
  require_file 'reference manifest' "$directory/manifest.json"
  require_file 'reference PCM' "$directory/pcm_400.f32.bin"
  require_file 'reference logits' "$directory/logits_1x42.f32.bin"
  [[ "$(sha256_file "$directory/manifest.json")" == "$REFERENCE_MANIFEST_SHA256" ]] || die 'reference manifest SHA-256 mismatch'
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python - \
    "$directory" "$REFERENCE_MANIFEST_SHA256" "$UPSTREAM_REPO" "$UPSTREAM_REVISION" "$CHECKPOINT_SHA256" "$CONFIG_SHA256" <<'PY'
import hashlib, json, math, struct, sys
from pathlib import Path
directory = Path(sys.argv[1]); expected_manifest_sha, upstream, revision, checkpoint, config = sys.argv[2:]
def reject_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result: raise ValueError(f"duplicate manifest key: {key}")
        result[key] = value
    return result
manifest_path = directory / "manifest.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates)
expected_keys = {"schema", "upstream", "revision", "checkpoint_sha256", "config_sha256", "reference_implementation", "torch_version", "transformers_version", "sample_rate", "pcm_file", "pcm_shape", "pcm_sha256", "logits_file", "logits_shape", "logits_sha256"}
if set(manifest) != expected_keys: raise ValueError("reference manifest key set drifted")
for key, value in {"schema":"vokra-charsiu-parity-v1", "upstream":upstream, "revision":revision, "checkpoint_sha256":checkpoint, "config_sha256":config, "reference_implementation":"transformers.Wav2Vec2ForCTC", "sample_rate":16000, "pcm_file":"pcm_400.f32.bin", "pcm_shape":[400], "logits_file":"logits_1x42.f32.bin", "logits_shape":[1,42]}.items():
    if manifest.get(key) != value: raise ValueError(f"reference manifest identity mismatch: {key}")
if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != expected_manifest_sha: raise ValueError("reference manifest digest changed during validation")
for name, expected_size, key in (("pcm_400.f32.bin",1600,"pcm_sha256"),("logits_1x42.f32.bin",168,"logits_sha256")):
    payload = (directory / name).read_bytes()
    if len(payload) != expected_size or hashlib.sha256(payload).hexdigest() != manifest[key]: raise ValueError(f"reference payload identity mismatch: {name}")
    if not all(math.isfinite(value) for value in struct.unpack("<{}f".format(expected_size // 4), payload)):
        raise ValueError(f"reference payload contains non-finite values: {name}")
print("Charsiu reference packet authenticated: official Transformers, 400 PCM samples, 1x42 logits")
PY
}

directory_digest() {
  local directory="$1" entry name
  while IFS= read -r entry; do
    name="$(basename "$entry")"
    printf '%s  %s\n' "$(sha256_file "$entry")" "$name"
  done < <(find -P "$directory" -mindepth 1 -maxdepth 1 -type f -print | sort) | shasum -a 256 | awk '{print $1}'
}

require_vast_evidence() {
  local directory="$1" entry name count=0
  [[ -d "$directory" && ! -L "$directory" ]] || die 'VAST evidence directory is missing or symlinked'
  reject_symlink_ancestry "$directory" VAST-evidence
  while IFS= read -r entry; do
    name="$(basename "$entry")"
    case "$name" in reference.json|reference-verified.log|bridge.log|bridge-verified.log|build.log|convert.log|gguf-verified.log|parity.log|parity-verified.log|fmt.log|zero-deps.log|summary.txt) ;; *) die "unexpected VAST evidence entry: $name" ;; esac
    count=$((count + 1)); require_file "VAST evidence $name" "$entry"
  done < <(find -P "$directory" -mindepth 1 -maxdepth 1 -print)
  [[ "$count" == 12 ]] || die 'VAST evidence must contain exactly twelve top-level files'
  [[ "$(directory_digest "$directory")" == "$VAST_EVIDENCE_SHA256" ]] || die 'VAST evidence digest mismatch'
  local summary="$directory/summary.txt"
  for line in 'execution_status=PASS' "upstream_repository=$UPSTREAM_REPO" "upstream_revision=$UPSTREAM_REVISION" "checkpoint_sha256=$CHECKPOINT_SHA256" "config_sha256=$CONFIG_SHA256" "gguf_sha256=$GGUF_SHA256" "reference_manifest_sha256=$REFERENCE_MANIFEST_SHA256" "fp32_atol=$CPU_REFERENCE_ATOL" 'cpu_parity=PASS' 'publication=NO_UPLOAD'; do
    grep -Fxc -- "$line" "$summary" | grep -Fxq 1 || die "VAST summary missing exact line: $line"
  done
  [[ "$(grep -Ec '^CHARSIU_OFFICIAL_PARITY_METRICS frames=[1-9][0-9]* logits=[1-9][0-9]* max_abs=[0-9]+\.[0-9]+ index=[0-9]+ rust=-?[0-9]+\.[0-9]+ transformers=-?[0-9]+\.[0-9]+ atol=0\.000200000$' "$directory/parity.log" || true)" == 1 ]] || die 'VAST CPU parity metrics are missing or duplicated'
  [[ "$(grep -Ec '^CHARSIU_OFFICIAL_PARITY PASS max_abs=[0-9]+\.[0-9]+ atol=0\.000200000 frames=[1-9][0-9]* reference=transformers\.Wav2Vec2ForCTC fixture=official_canned_pcm$' "$directory/parity.log" || true)" == 1 ]] || die 'VAST CPU parity PASS marker is missing or duplicated'
  awk '/^CHARSIU_OFFICIAL_PARITY_METRICS / { for (i = 1; i <= NF; i++) { split($i, pair, "="); if (pair[1] == "max_abs" && (pair[2] + 0) > 0.0002) exit 1 } }' "$directory/parity.log" || die 'VAST CPU parity metric exceeds the registered bound'
}

require_test_pass() {
  local path="$1"
  if ! awk -v expected="$TEST_NAME" '
    BEGIN {
      named = "test " expected " ..."
      inline = named " ok"
      prefixed_cpu = "^test " expected " [.][.][.] CHARSIU_APPLE_CPU_REFERENCE_METRICS frames=[1-9][0-9]* logits=[1-9][0-9]* max_abs=[0-9]+[.][0-9]{9} index=[0-9]+ atol=0[.]000200000$"
      tests = 0
      results = 0
      standalone_ok = 0
      invalid = 0
      state = "before"
      form = ""
    }
    /^test result:/ {
      results++
      if ($0 !~ /^test result: ok[.] 1 passed; 0 failed; 0 ignored; 0 measured; [0-9]+ filtered out(; finished in .+)?$/ || (state != "inline" && state != "split-ok")) {
        invalid = 1
      }
      state = "result"
      next
    }
    /^test / {
      tests++
      if ($0 == inline) {
        if (state != "before") invalid = 1
        state = "inline"
        form = "inline"
      } else if ($0 == named) {
        if (state != "before") invalid = 1
        state = "split"
        form = "split"
      } else if ($0 ~ prefixed_cpu) {
        if (state != "before") invalid = 1
        state = "split"
        form = "split-prefixed-cpu"
      } else {
        invalid = 1
      }
      next
    }
    /^ok$/ {
      standalone_ok++
      if (state == "split") state = "split-ok"
      else invalid = 1
      next
    }
    END {
      if (tests != 1 || results != 1 || invalid || state != "result") exit 1
      if (form == "inline" && standalone_ok == 0) exit 0
      if ((form == "split" || form == "split-prefixed-cpu") && standalone_ok == 1) exit 0
      exit 1
    }
  ' "$path"; then
    die 'Apple log must contain one passing named test in either inline or interleaved libtest form'
    return 2
  fi
  [[ "$(grep -Ec "^(CHARSIU_APPLE_CPU_REFERENCE_METRICS|test ${TEST_NAME} \.\.\. CHARSIU_APPLE_CPU_REFERENCE_METRICS) frames=[1-9][0-9]* logits=[1-9][0-9]* max_abs=[0-9]+\\.[0-9]{9} index=[0-9]+ atol=0\\.000200000$" "$path" || true)" == 1 ]] || { die 'CPU/reference metric is missing or malformed'; return 2; }
  [[ "$(grep -Ec "^CHARSIU_APPLE_METAL_REFERENCE_METRICS frames=[1-9][0-9]* logits=[1-9][0-9]* max_abs=[0-9]+\\.[0-9]{9} index=[0-9]+ atol=${METAL_ATOL_REGEX}$" "$path" || true)" == 1 ]] || { die 'Metal/reference metric is missing or malformed'; return 2; }
  [[ "$(grep -Ec "^CHARSIU_APPLE_METAL_CPU_METRICS frames=[1-9][0-9]* logits=[1-9][0-9]* max_abs=[0-9]+\\.[0-9]{9} index=[0-9]+ atol=${METAL_ATOL_REGEX}$" "$path" || true)" == 1 ]] || { die 'Metal/CPU metric is missing or malformed'; return 2; }
  [[ "$(grep -Fxc -- "CHARSIU_APPLE_PARITY PASS frames=1 cpu_reference_atol=${CPU_REFERENCE_ATOL} metal_atol=${METAL_ATOL} reference=transformers.Wav2Vec2ForCTC route=explicit_cpu_and_metal no_fallback=true publication=NO_UPLOAD" "$path" || true)" == 1 ]] || { die 'Apple Charsiu PASS sentinel is missing or malformed'; return 2; }
  awk -v expected="$TEST_NAME" '/^CHARSIU_APPLE_(CPU_REFERENCE|METAL_REFERENCE|METAL_CPU)_METRICS / || $0 ~ ("^test " expected " [.][.][.] CHARSIU_APPLE_CPU_REFERENCE_METRICS ") { for (i = 1; i <= NF; i++) { split($i, pair, "="); if (pair[1] == "max_abs" && (pair[2] + 0) > ((index($0, "CPU_REFERENCE") > 0) ? 0.0002 : 0.01)) exit 1 } }' "$path" || { die 'Apple Charsiu metric exceeds its registered bound'; return 2; }
}

record_fingerprint() {
  local path="$1"
  {
    echo "utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "git_commit=$(git -C "$VOKRA_ROOT" rev-parse HEAD)"
    echo "expected_head=$EXPECTED_HEAD"
    echo "uname=$(uname -a)"
    echo "machine=$(uname -m)"
    echo "memory_bytes=$(sysctl -n hw.memsize)"
    echo "physical_cpu=$(sysctl -n hw.physicalcpu)"
    echo "logical_cpu=$(sysctl -n hw.logicalcpu)"
    sw_vers
    rustc --version --verbose
    cargo --version
    echo "metal_compiler=$(xcrun -f metal)"
    system_profiler SPDisplaysDataType | sed -n '1,40p'
  } > "$path"
}

run_self_test() {
  local path="${BASH_SOURCE[0]}" fail=0 token
  for token in 'VOKRA_REMOTE_APPLE_SILICON=1' 'Darwin' 'arm64' 'xcrun -f metal' \
    'charsiu_apple_cpu_metal_matches_reference' 'from_gguf_with_backend' 'BackendKind::Metal' \
    'CHARSIU_APPLE_CPU_REFERENCE_METRICS' 'CHARSIU_APPLE_METAL_REFERENCE_METRICS' \
    'CHARSIU_APPLE_METAL_CPU_METRICS' 'no_fallback=true' 'publication=NO_UPLOAD' \
    'reference-manifest-sha256' 'vast-evidence-sha256' '--expected-head' \
    'VAST_EVIDENCE_SHA256' 'directory_digest' 'CARGO_BUILD_JOBS=1' '--ignored' \
    'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured' \
    'UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python'; do
    grep -Fq -- "$token" "$path" || { log "self-test missing contract: $token"; fail=1; }
  done
  grep -Fq 'fn charsiu_apple_cpu_metal_matches_reference' "$TEST_SOURCE" || { log 'self-test missing named Rust test'; fail=1; }
  grep -Fq '#[test]' "$TEST_SOURCE" || { log 'self-test missing Rust test attribute'; fail=1; }
  grep -Fq 'BackendKind::Cpu' "$TEST_SOURCE" || { log 'self-test missing explicit CPU selection'; fail=1; }
  grep -Fq 'BackendKind::Metal' "$TEST_SOURCE" || { log 'self-test missing explicit Metal selection'; fail=1; }
  grep -Fq '#[ignore' "$TEST_SOURCE" || { log 'self-test missing ignored Apple test attribute'; fail=1; }
  local model_source="$VOKRA_ROOT/crates/vokra-models/src/align/charsiu.rs"
  for token in 'pub const CHARSIU_HOT_OPS' 'Compute::for_backend' 'group_norm_groups_f32' 'softmax_with_compute' 'feature_projection_forward_with_compute' 'positional_conv_forward_with_compute' 'transformer_block_forward_with_valid_keys_and_compute' 'linear_forward_with_compute'; do
    grep -Fq -- "$token" "$model_source" || { log "self-test missing backend source contract: $token"; fail=1; }
  done
  if grep -Eq '^[[:space:]]*let features = vokra_ops::waveform_frontend' "$model_source"; then log 'self-test found scalar Charsiu waveform fallback'; fail=1; fi
  if grep -En 'git[[:space:]]+(push|clone|fetch|pull)|publish-one\.sh|huggingface-cli[[:space:]]+upload|--(push|upload|publish)|curl[[:space:]]|wget[[:space:]]' "$path" | grep -v 'grep -En' >/dev/null; then
    log 'self-test found forbidden network/publication command'; fail=1
  fi
  if grep -En '^[[:space:]]*(python3?|pip)([[:space:]]|$)' "$path" >/dev/null; then
    log 'self-test found direct Python invocation'; fail=1
  fi
  local temporary interleaved inline prefixed extra failure
  temporary="$(mktemp -d "${TMPDIR:-/tmp}/vokra-charsiu-apple-self-test.XXXXXX")"
  interleaved="$temporary/interleaved.log"
  inline="$temporary/inline.log"
  prefixed="$temporary/prefixed.log"
  extra="$temporary/extra.log"
  failure="$temporary/failure.log"
  printf '%s\n' \
    "test $TEST_NAME ..." \
    'CHARSIU_APPLE_CPU_REFERENCE_METRICS frames=1 logits=42 max_abs=0.000100000 index=0 atol=0.000200000' \
    'CHARSIU_APPLE_METAL_REFERENCE_METRICS frames=1 logits=42 max_abs=0.001000000 index=0 atol=0.010000000' \
    'CHARSIU_APPLE_METAL_CPU_METRICS frames=1 logits=42 max_abs=0.001000000 index=0 atol=0.010000000' \
    'CHARSIU_APPLE_PARITY PASS frames=1 cpu_reference_atol=0.000200000 metal_atol=0.010000000 reference=transformers.Wav2Vec2ForCTC route=explicit_cpu_and_metal no_fallback=true publication=NO_UPLOAD' \
    'ok' \
    'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s' > "$interleaved"
  awk -v expected="$TEST_NAME" '{ if ($0 == "test " expected " ...") print $0 " ok"; else if ($0 != "ok") print }' "$interleaved" > "$inline"
  awk -v expected="$TEST_NAME" '{ if ($0 == "test " expected " ...") { getline metric; print $0 " " metric } else if ($0 != "CHARSIU_APPLE_CPU_REFERENCE_METRICS frames=1 logits=42 max_abs=0.000100000 index=0 atol=0.000200000") print }' "$interleaved" > "$prefixed"
  cp "$interleaved" "$extra"
  printf '%s\n' 'test unrelated_extra_test ... ok' >> "$extra"
  printf '%s\n' \
    "test $TEST_NAME ... FAILED" \
    'test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s' > "$failure"
  if ! require_test_pass "$interleaved" >/dev/null 2>&1; then log 'self-test rejected valid interleaved libtest output'; fail=1; fi
  if ! require_test_pass "$inline" >/dev/null 2>&1; then log 'self-test rejected valid inline libtest output'; fail=1; fi
  if ! require_test_pass "$prefixed" >/dev/null 2>&1; then log 'self-test rejected valid prefixed interleaved libtest output'; fail=1; fi
  if require_test_pass "$extra" >/dev/null 2>&1; then log 'self-test accepted an extra named test'; fail=1; fi
  if require_test_pass "$failure" >/dev/null 2>&1; then log 'self-test accepted a failed test result'; fail=1; fi
  rm -f -- "$temporary"/*.log
  rmdir "$temporary"
  if "$path" --self-test --gguf /tmp/rejected >/dev/null 2>&1; then log 'self-test accepted an extra argument'; fail=1; fi
  if "$path" --unknown >/dev/null 2>&1; then log 'self-test accepted an unknown argument'; fail=1; fi
  (( fail == 0 )) || return 1
  echo 'apple-silicon-charsiu.sh self-test: OK'
}

GGUF='' GGUF_SHA256='' REFERENCE='' REFERENCE_MANIFEST_SHA256='' VAST_EVIDENCE='' VAST_EVIDENCE_SHA256='' EXPECTED_HEAD='' EVIDENCE='' SELF_TEST=0
seen_gguf=0 seen_gguf_sha=0 seen_reference=0 seen_reference_sha=0 seen_vast=0 seen_vast_sha=0 seen_head=0 seen_evidence=0 seen_self=0
while (($#)); do
  case "$1" in
    --self-test) ((seen_self == 0)) || die 'duplicate --self-test'; seen_self=1; SELF_TEST=1; shift ;;
    --gguf) ((seen_gguf == 0)) || die 'duplicate --gguf'; (($# >= 2)) && [[ -n "$2" && "$2" != -* ]] || die '--gguf requires a nonempty value'; seen_gguf=1; GGUF="$2"; shift 2 ;;
    --gguf-sha256) ((seen_gguf_sha == 0)) || die 'duplicate --gguf-sha256'; (($# >= 2)) && [[ "$2" =~ ^[0-9a-f]{64}$ ]] || die '--gguf-sha256 must be lowercase hex64'; seen_gguf_sha=1; GGUF_SHA256="$2"; shift 2 ;;
    --reference) ((seen_reference == 0)) || die 'duplicate --reference'; (($# >= 2)) && [[ -n "$2" && "$2" != -* ]] || die '--reference requires a nonempty value'; seen_reference=1; REFERENCE="$2"; shift 2 ;;
    --reference-manifest-sha256) ((seen_reference_sha == 0)) || die 'duplicate --reference-manifest-sha256'; (($# >= 2)) && [[ "$2" =~ ^[0-9a-f]{64}$ ]] || die '--reference-manifest-sha256 must be lowercase hex64'; seen_reference_sha=1; REFERENCE_MANIFEST_SHA256="$2"; shift 2 ;;
    --vast-evidence) ((seen_vast == 0)) || die 'duplicate --vast-evidence'; (($# >= 2)) && [[ -n "$2" && "$2" != -* ]] || die '--vast-evidence requires a nonempty value'; seen_vast=1; VAST_EVIDENCE="$2"; shift 2 ;;
    --vast-evidence-sha256) ((seen_vast_sha == 0)) || die 'duplicate --vast-evidence-sha256'; (($# >= 2)) && [[ "$2" =~ ^[0-9a-f]{64}$ ]] || die '--vast-evidence-sha256 must be lowercase hex64'; seen_vast_sha=1; VAST_EVIDENCE_SHA256="$2"; shift 2 ;;
    --expected-head) ((seen_head == 0)) || die 'duplicate --expected-head'; (($# >= 2)) && [[ "$2" =~ ^[0-9a-f]{40}$ ]] || die '--expected-head must be lowercase hex40'; seen_head=1; EXPECTED_HEAD="$2"; shift 2 ;;
    --evidence-dir) ((seen_evidence == 0)) || die 'duplicate --evidence-dir'; (($# >= 2)) && [[ -n "$2" && "$2" != -* ]] || die '--evidence-dir requires a nonempty value'; seen_evidence=1; EVIDENCE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) usage; die "unknown argument: $1" ;;
  esac
done

if ((SELF_TEST)); then
  [[ "$GGUF$GGUF_SHA256$REFERENCE$REFERENCE_MANIFEST_SHA256$VAST_EVIDENCE$VAST_EVIDENCE_SHA256$EXPECTED_HEAD$EVIDENCE" == '' ]] || die '--self-test accepts no other arguments'
  run_self_test
  exit $?
fi

((seen_gguf && seen_gguf_sha && seen_reference && seen_reference_sha && seen_vast && seen_vast_sha && seen_head && seen_evidence)) || { usage; die 'all required arguments must be supplied'; }
[[ "$GGUF_SHA256" =~ ^[0-9a-f]{64}$ && "$REFERENCE_MANIFEST_SHA256" =~ ^[0-9a-f]{64}$ && "$VAST_EVIDENCE_SHA256" =~ ^[0-9a-f]{64}$ ]] || die 'input digests must be lowercase hex64'
[[ "$EXPECTED_HEAD" =~ ^[0-9a-f]{40}$ ]] || die 'expected head must be lowercase hex40'
for path_label in GGUF REFERENCE VAST_EVIDENCE; do
  path_value="${!path_label}"
  require_absolute "$path_label" "$path_value"
  reject_symlink_ancestry "$path_value" "$path_label"
done
require_absolute evidence "$EVIDENCE"
require_file GGUF "$GGUF"
[[ "$(sha256_file "$GGUF")" == "$GGUF_SHA256" ]] || die 'GGUF SHA-256 mismatch'
require_apple_host
require_tooling
require_reference "$REFERENCE"
require_vast_evidence "$VAST_EVIDENCE"
reserve_evidence_dir "$EVIDENCE"
record_fingerprint "$EVIDENCE/fingerprint.txt"

APPLE_LOG="$(mktemp "${TMPDIR:-/tmp}/charsiu-apple.XXXXXX")"
trap 'rm -f -- "$APPLE_LOG"' EXIT
export "$GGUF_ENV=$GGUF" "$REFERENCE_ENV=$REFERENCE" VOKRA_REMOTE_APPLE_SILICON=1
CARGO_BUILD_JOBS=1 cargo test --locked --release --features metal -p vokra-models --test charsiu_apple_cpu_metal "$TEST_NAME" -- --ignored --exact --nocapture --test-threads=1 > "$APPLE_LOG" 2>&1
require_test_pass "$APPLE_LOG"
cp "$APPLE_LOG" "$EVIDENCE/parity.log"
[[ -f "$EVIDENCE/parity.log" && ! -L "$EVIDENCE/parity.log" ]] || die 'Apple parity log was not written safely'
printf '%s\n' "{\"schema\":\"vokra-charsiu-apple-v1\",\"status\":\"APPLE_CPU_REFERENCE_METAL_PASS\",\"publication\":\"NO_UPLOAD\",\"expected_head\":\"$EXPECTED_HEAD\",\"actual_head\":\"$(git -C "$VOKRA_ROOT" rev-parse HEAD)\",\"gguf_sha256\":\"$GGUF_SHA256\",\"reference_manifest_sha256\":\"$REFERENCE_MANIFEST_SHA256\",\"vast_evidence_sha256\":\"$VAST_EVIDENCE_SHA256\",\"test\":\"$TEST_NAME\",\"no_fallback\":true}" > "$EVIDENCE/validation-summary.json"
[[ -f "$EVIDENCE/validation-summary.json" && ! -L "$EVIDENCE/validation-summary.json" ]] || die 'validation summary was not written safely'
log 'PASS: Charsiu CPU/reference, Metal/reference, and Metal/CPU parity; publication=NO_UPLOAD'
