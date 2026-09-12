#!/usr/bin/env bash
# Apple Silicon verifier for the VAST-produced Distil-Whisper and
# Kotoba-Whisper v2.2 real-weight packets. It consumes an authenticated
# transfer packet and never downloads, converts, publishes, or manufactures a
# parity result.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VOKRA_ROOT="${VOKRA_ROOT:-$DEFAULT_ROOT}"
PRODUCER="$VOKRA_ROOT/tools/parity/whisper_extras/run_vast_validation.sh"
TEST_SOURCE="$VOKRA_ROOT/crates/vokra-models/tests/parity_whisper_extras.rs"
TEST_TARGET="parity_whisper_extras"
NO_UPLOAD=1

INPUT_AUDIO_SHA256="58adb4ea501d955fcd40bfbb69128f8f40428b81d8716b9ed337949773be253f"
DISTIL_REPO="distil-whisper/distil-large-v3.5"
DISTIL_REVISION="728a7691f3ff1d3d971528d3203a6e9559165d41"
DISTIL_LICENSE="MIT"
DISTIL_CHECKPOINT_BYTES=3025686376
DISTIL_CONFIG_BYTES=1249
DISTIL_GENERATION_BYTES=4249
DISTIL_TOKENIZER_BYTES=2480645
DISTIL_PREPROCESSOR_BYTES=340
DISTIL_CHECKPOINT_SHA256="76ec9f754fc4b4810845dc36b71d1897c1342e702810c179e1569690084cfb0c"
DISTIL_CONFIG_SHA256="515a10a9979258d3fc71cf79b2cd055c189f07d78879a15bd9bc282673308b85"
DISTIL_GENERATION_SHA256="b521c66612bd95be36c154f2d3904f6e4ea3be481a18a48f293d66791f60cf98"
DISTIL_TOKENIZER_SHA256="b3c8202bbf06d8ee4232c5984baa563784ac4737e2e7fdc42fa180200d3cfcdb"
DISTIL_PREPROCESSOR_SHA256="7ccc62c6f2765af1f3b46c00c9b5894426835a05021c8b9c01eecb6dfb542711"
KOTOBA_REPO="kotoba-tech/kotoba-whisper-v2.2"
KOTOBA_REVISION="9d33482a0eb9b57f1ad80708e8ac5538246d8355"
KOTOBA_LICENSE="Apache-2.0"
KOTOBA_CHECKPOINT_BYTES=3025686376
KOTOBA_CONFIG_BYTES=1499
KOTOBA_GENERATION_BYTES=3898
KOTOBA_TOKENIZER_BYTES=3931381
KOTOBA_PREPROCESSOR_BYTES=340
KOTOBA_CHECKPOINT_SHA256="e0ef3e7b379515f0c35d0e7885638ddef0fa9f5c8e3e3f88cbc6da9b39edd1e9"
KOTOBA_CONFIG_SHA256="75e0166afbf44308af4908793fc5ade1707890ed15760f0529b468dcfc378aec"
KOTOBA_GENERATION_SHA256="20d28b9169207ab6ca402ec9342393a88ec3f341d88c9da24e516e1da71c24de"
KOTOBA_TOKENIZER_SHA256="615928f5a25409279b47b47d87a4ca2aaee3bd09f65e1a3df6c9d23c718cfdb0"
KOTOBA_PREPROCESSOR_SHA256="7ccc62c6f2765af1f3b46c00c9b5894426835a05021c8b9c01eecb6dfb542711"

log() { printf '[whisper-extras-apple] %s\n' "$*" >&2; }
die() { log "ERROR: $*"; return 2; }

usage() {
  cat <<'EOF' >&2
usage: apple-silicon-whisper-extras.sh \
  --distil-gguf ABS --distil-gguf-sha256 HEX64 \
  --distil-reference ABS --distil-reference-manifest-sha256 HEX64 \
  --distil-reference-packet-sha256 HEX64 \
  --distil-cpu-log ABS --distil-cpu-log-sha256 HEX64 \
  --kotoba-gguf ABS --kotoba-gguf-sha256 HEX64 \
  --kotoba-reference ABS --kotoba-reference-manifest-sha256 HEX64 \
  --kotoba-reference-packet-sha256 HEX64 \
  --kotoba-cpu-log ABS --kotoba-cpu-log-sha256 HEX64 \
  --vast-head-file ABS --vast-head-sha256 HEX64 \
  --expected-head HEX40 --evidence-dir ABSENT_DIR
       apple-silicon-whisper-extras.sh --self-test

Consumes only the exact VAST real-weight packet. The test runs once per
variant on CPU and Metal through an explicit backend; this worker records no
upload and never creates PASS markers from input evidence.
EOF
}

sha256_file() { shasum -a 256 "$1" | awk '{print $1}'; }

require_regular() {
  local label="$1" path="$2"
  [[ -f "$path" && ! -L "$path" && -s "$path" ]] \
    || die "$label must be a non-empty regular file: $path"
}

reject_symlink_ancestors() {
  local path="$1" component rest current
  [[ "$path" == /* && "$path" != *'//' && "$path" != *'/./'* && \
    "$path" != *'/../'* && "$path" != */. && "$path" != */.. ]] \
    || { die "path is not canonical absolute: $path"; return 2; }
  rest="${path#/}"
  current="/"
  while [[ -n "$rest" ]]; do
    if [[ "$rest" == */* ]]; then
      component="${rest%%/*}"; rest="${rest#*/}"
    else
      component="$rest"; rest=""
    fi
    [[ -n "$component" ]] || continue
    current="${current%/}/$component"
    [[ ! -L "$current" ]] || { die "symlink ancestor is forbidden: $path"; return 2; }
  done
}

real_path() {
  local path="$1" parent leaf
  parent="$(dirname "$path")"; leaf="$(basename "$path")"
  printf '%s/%s\n' "$(cd -P "$parent" && pwd)" "$leaf"
}

require_disjoint_absent_evidence() {
  local evidence="$1" other candidate other_real
  reject_symlink_ancestors "$evidence" || return 2
  [[ ! -e "$evidence" && ! -L "$evidence" ]] \
    || { die "evidence directory must be absent: $evidence"; return 2; }
  candidate="$(real_path "$evidence")"
  shift
  for other in "$@"; do
    [[ "$other" == /* ]] || { die "input path must be absolute: $other"; return 2; }
    reject_symlink_ancestors "$other" || return 2
    other_real="$(real_path "$other")"
    if [[ "$candidate" == "$other_real" || "$candidate/" == "$other_real/"* || \
      "$other_real/" == "$candidate/"* ]]; then
      die "evidence directory overlaps input: $evidence / $other"
      return 2
    fi
  done
}

require_digest() {
  local label="$1" path="$2" expected="$3" actual
  [[ "$expected" =~ ^[0-9a-f]{64}$ ]] || die "$label digest must be lowercase HEX64"
  require_regular "$label" "$path"
  actual="$(sha256_file "$path")"
  [[ "$actual" == "$expected" ]] || die "$label digest mismatch: $actual != $expected"
}

packet_digest() {
  local directory="$1" name
  local -a names=(input_pcm.f32le encoder.f32le logits_last.f32le greedy_tokens.u32le manifest.json)
  for name in "${names[@]}"; do require_regular "reference $name" "$directory/$name"; done
  for name in "${names[@]}"; do
    printf '%s  %s\n' "$(sha256_file "$directory/$name")" "$name"
  done | shasum -a 256 | awk '{print $1}'
}

validate_manifest() {
  local variant="$1" directory="$2" expected_manifest="$3" expected_packet="$4"
  local repo revision license checkpoint config generation tokenizer preprocessor prefix checkpoint_bytes config_bytes generation_bytes tokenizer_bytes preprocessor_bytes
  case "$variant" in
    distil_whisper)
      repo="$DISTIL_REPO"; revision="$DISTIL_REVISION"; license="$DISTIL_LICENSE"
      checkpoint_bytes="$DISTIL_CHECKPOINT_BYTES"; config_bytes="$DISTIL_CONFIG_BYTES"; generation_bytes="$DISTIL_GENERATION_BYTES"; tokenizer_bytes="$DISTIL_TOKENIZER_BYTES"; preprocessor_bytes="$DISTIL_PREPROCESSOR_BYTES"
      checkpoint="$DISTIL_CHECKPOINT_SHA256"; config="$DISTIL_CONFIG_SHA256"
      generation="$DISTIL_GENERATION_SHA256"; tokenizer="$DISTIL_TOKENIZER_SHA256"
      preprocessor="$DISTIL_PREPROCESSOR_SHA256"; prefix='[50258, 50259, 50360, 50364]' ;;
    kotoba_whisper)
      repo="$KOTOBA_REPO"; revision="$KOTOBA_REVISION"; license="$KOTOBA_LICENSE"
      checkpoint_bytes="$KOTOBA_CHECKPOINT_BYTES"; config_bytes="$KOTOBA_CONFIG_BYTES"; generation_bytes="$KOTOBA_GENERATION_BYTES"; tokenizer_bytes="$KOTOBA_TOKENIZER_BYTES"; preprocessor_bytes="$KOTOBA_PREPROCESSOR_BYTES"
      checkpoint="$KOTOBA_CHECKPOINT_SHA256"; config="$KOTOBA_CONFIG_SHA256"
      generation="$KOTOBA_GENERATION_SHA256"; tokenizer="$KOTOBA_TOKENIZER_SHA256"
      preprocessor="$KOTOBA_PREPROCESSOR_SHA256"; prefix='[50258, 50266, 50360, 50364]' ;;
    *) die "unknown reference variant: $variant"; return 2 ;;
  esac
  [[ "$directory" == /* ]] || die "reference directory must be absolute"
  reject_symlink_ancestors "$directory"
  [[ -d "$directory" && ! -L "$directory" ]] || die "reference directory is invalid: $directory"
  require_digest "$variant reference manifest" "$directory/manifest.json" "$expected_manifest"
  [[ "$expected_packet" =~ ^[0-9a-f]{64}$ ]] || die "$variant packet digest must be lowercase HEX64"
  [[ "$(packet_digest "$directory")" == "$expected_packet" ]] \
    || die "$variant reference packet digest mismatch"
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python - \
    "$directory/manifest.json" "$repo" "$revision" "$license" "$checkpoint" \
    "$config" "$generation" "$tokenizer" "$preprocessor" "$prefix" \
    "$checkpoint_bytes" "$config_bytes" "$generation_bytes" "$tokenizer_bytes" "$preprocessor_bytes" \
    "$INPUT_AUDIO_SHA256" <<'PY'
import hashlib, json, pathlib, sys
path, repo, revision, license_name, checkpoint, config, generation, tokenizer, preprocessor, prefix, checkpoint_bytes, config_bytes, generation_bytes, tokenizer_bytes, preprocessor_bytes, audio_sha = sys.argv[1:]
def unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise SystemExit(f"duplicate manifest key: {key}")
        result[key] = value
    return result
data = json.loads(pathlib.Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique)
expected = {
    "schema": "vokra-whisper-extras-reference-v1",
    "model": "kotoba_whisper" if "kotoba" in repo else "distil_whisper",
    "upstream_repo": repo,
    "upstream_revision": revision,
    "weight_license_spdx": license_name,
    "reference_implementation": "transformers.WhisperForConditionalGeneration",
    "audio_sha256": audio_sha,
    "sample_rate": 16000, "pcm_samples": 480000,
    "language": "ja" if "kotoba" in repo else "en", "task": "transcribe",
    "no_timestamps": True, "decoder_prefix": json.loads(prefix),
    "checkpoint_sha256": checkpoint, "config_sha256": config,
    "generation_config_sha256": generation, "tokenizer_sha256": tokenizer,
    "preprocessor_sha256": preprocessor, "atol": 0.01,
    "checkpoint_bytes": int(checkpoint_bytes), "config_bytes": int(config_bytes),
    "generation_config_bytes": int(generation_bytes), "tokenizer_bytes": int(tokenizer_bytes),
    "preprocessor_bytes": int(preprocessor_bytes),
}
for key, value in expected.items():
    if data.get(key) != value:
        raise SystemExit(f"manifest {key} drift: {data.get(key)!r} != {value!r}")
files = data.get("files")
names = ["encoder.f32le", "greedy_tokens.u32le", "input_pcm.f32le", "logits_last.f32le"]
if not isinstance(files, dict) or set(files) != set(names):
    raise SystemExit("manifest file hash closure is incomplete or has extras")
root = pathlib.Path(path).parent
allowed = set(names) | {"manifest.json"}
entries = list(root.iterdir())
if {entry.name for entry in entries} != allowed or any(entry.is_symlink() or not entry.is_file() for entry in entries):
    raise SystemExit("reference packet directory contains an unexpected or unsafe entry")
for name in names:
    digest = files[name]
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise SystemExit(f"invalid packet digest for {name}")
    actual = hashlib.sha256(pathlib.Path(path).parent.joinpath(name).read_bytes()).hexdigest()
    if actual != digest:
        raise SystemExit(f"packet digest mismatch for {name}: {actual} != {digest}")
PY
}

require_cpu_log() {
  local variant="$1" path="$2" expected_sha="$3" test_name marker repo revision
  test_name="parity_whisper_extras_${variant}"
  marker="[parity_whisper_extras/${variant}] native CPU real-weight parity PASS"
  case "$variant" in
    distil_whisper) repo="$DISTIL_REPO"; revision="$DISTIL_REVISION" ;;
    kotoba_whisper) repo="$KOTOBA_REPO"; revision="$KOTOBA_REVISION" ;;
    *) die "unknown CPU log variant: $variant"; return 2 ;;
  esac
  require_digest "$variant CPU log" "$path" "$expected_sha" || return 2
  grep -Fq "[$variant] repo=$repo revision=$revision" "$path" || { die "$variant CPU log source pin is missing"; return 2; }
  ! grep -Eq '^test .* \.\.\. (ignored|skipped|FAILED|failed)([[:space:]]|$)' "$path" || { die "$variant CPU log contains an ignored, skipped, or failed test"; return 2; }
  [[ "$(grep -Ec "^test ${test_name} \.\.\." "$path" || true)" == 1 ]] || { die "$variant CPU test singleton missing"; return 2; }
  [[ "$(grep -Ev '^test result:' "$path" | grep -Ec '^test ' || true)" == 1 ]] || { die "$variant CPU log has extra or missing test result"; return 2; }
  [[ "$(grep -Ec '^ok$' "$path" || true)" == 1 ]] || { die "$variant CPU log has no exact standalone ok"; return 2; }
  [[ "$(grep -Ec '^test result: ok\. 1 passed; 0 failed; 0 ignored; 0 measured; [0-9]+ filtered out(; finished in [0-9]+\.[0-9]+s)?$' "$path" || true)" == 1 ]] || { die "$variant CPU result is not exactly one pass"; return 2; }
  [[ "$(grep -Fc "$marker" "$path" || true)" == 1 ]] || { die "$variant CPU parity sentinel missing or duplicated"; return 2; }
  ! grep -Eq 'FAILED|test result: FAILED|verdict=FAIL' "$path" || { die "$variant CPU log contains a failure marker"; return 2; }
}

require_apple_log() {
  local variant="$1" path="$2" expected_sha="$3" test_name marker
  test_name="parity_whisper_extras_${variant}_apple_cpu_metal"
  marker="WHISPER_EXTRAS_APPLE variant=${variant} cpu_reference=PASS metal_reference=PASS metal_cpu=PASS greedy_tokens=EXACT no_fallback=PASS verdict=PASS"
  require_digest "$variant Apple log" "$path" "$expected_sha" || return 2
  ! grep -Eq '^test .* \.\.\. (ignored|skipped|FAILED|failed)([[:space:]]|$)' "$path" || { die "$variant Apple log contains an ignored, skipped, or failed test"; return 2; }
  [[ "$(grep -Ec "^test ${test_name} \.\.\." "$path" || true)" == 1 ]] || { die "$variant Apple test singleton missing"; return 2; }
  [[ "$(grep -Ev '^test result:' "$path" | grep -Ec '^test ' || true)" == 1 ]] || { die "$variant Apple log has extra test results"; return 2; }
  [[ "$(grep -Ec '^ok$' "$path" || true)" == 1 ]] || { die "$variant Apple log has no exact standalone ok"; return 2; }
  [[ "$(grep -Ec '^test result: ok\. 1 passed; 0 failed; 0 ignored; 0 measured; [0-9]+ filtered out(; finished in [0-9]+\.[0-9]+s)?$' "$path" || true)" == 1 ]] || { die "$variant Apple result is not exactly one pass"; return 2; }
  [[ "$(grep -Fc "$marker" "$path" || true)" == 1 ]] || { die "$variant Apple parity sentinel missing or duplicated"; return 2; }
  ! grep -Eq 'FAILED|test result: FAILED|verdict=FAIL|BackendKind::Cpu.*Metal' "$path" || { die "$variant Apple log contains a failure or fallback marker"; return 2; }
}

require_remote_apple() {
  [[ "${VOKRA_REMOTE_APPLE_SILICON:-0}" == 1 ]] || die 'VOKRA_REMOTE_APPLE_SILICON=1 is required'
  [[ "$(uname -s)" == Darwin ]] || die 'requires Darwin'
  [[ "$(uname -m)" == arm64 ]] || die 'requires Apple arm64'
  command -v xcrun >/dev/null 2>&1 || die 'xcrun is unavailable'
  command -v uv >/dev/null 2>&1 || die 'uv is unavailable for manifest validation'
  xcrun -f metal >/dev/null 2>&1 || die 'Xcode Metal compiler is unavailable'
}

run_apple_test() {
  local variant="$1" gguf="$2" reference="$3" test_name="$4" output="$5" env_prefix
  case "$variant" in
    distil_whisper) env_prefix='DISTIL_WHISPER' ;;
    kotoba_whisper) env_prefix='KOTOBA_WHISPER' ;;
    *) die "unknown test variant: $variant"; return 2 ;;
  esac
  [[ "$(git -C "$VOKRA_ROOT" rev-parse HEAD)" == "$expected_head" ]] || die 'HEAD changed before Apple test'
  [[ -z "$(git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all)" ]] || die 'checkout became dirty'
  set +e
  env CARGO_BUILD_JOBS=1 CARGO_NET_OFFLINE=true RUST_TEST_THREADS=1 VOKRA_REMOTE_APPLE_SILICON=1 \
    "VOKRA_${env_prefix}_GGUF=$gguf" "VOKRA_${env_prefix}_REFDIR=$reference" \
    cargo test --locked --offline --release --features metal \
      --manifest-path "$VOKRA_ROOT/Cargo.toml" -p vokra-models --test "$TEST_TARGET" "$test_name" \
      -- --ignored --exact --nocapture --test-threads=1 2>&1 | tee "$output"
  local status=${PIPESTATUS[0]}
  set -e
  (( status == 0 )) || die "$variant Apple parity test failed; evidence log preserved"
  require_apple_log "$variant" "$output" "$(sha256_file "$output")"
}

self_test() {
  local self="${BASH_SOURCE[0]}" token temporary valid_cpu valid_apple mutated
  for token in 'VOKRA_REMOTE_APPLE_SILICON=1' 'Darwin' 'arm64' 'xcrun -f metal' 'NO_UPLOAD=1' \
    'CARGO_NET_OFFLINE=true' 'parity_whisper_extras_distil_whisper_apple_cpu_metal' \
    'parity_whisper_extras_kotoba_whisper_apple_cpu_metal' 'WHISPER_EXTRAS_APPLE variant=distil_whisper' \
    'WHISPER_EXTRAS_APPLE variant=kotoba_whisper' 'cpu_reference=PASS metal_reference=PASS metal_cpu=PASS' \
    'greedy_tokens=EXACT no_fallback=PASS' 'reference packet digest' 'transformers.WhisperForConditionalGeneration' \
    'vokra-whisper-extras-reference-v1' 'expected-head' 'vast-head-file' 'NO_UPLOAD'; do
    grep -Fq -- "$token" "$self" || die "self-test contract token missing: $token"
  done
  if grep -En '(^|[[:space:]])(curl|wget|git[[:space:]]+(clone|fetch|pull|push)|hf_hub_download|publish-one\.sh|upload\.sh|--push)([[:space:]]|$)' "$self" >/dev/null; then
    die 'acquisition or publication command found'
  fi
  bash -n "$self" || die 'shell syntax check failed'
  temporary="$(mktemp -d "${TMPDIR:-/tmp}/vokra-whisper-apple-selftest.XXXXXX")"
  valid_cpu="$temporary/cpu.log"
  valid_apple="$temporary/apple.log"
  printf '%s\n' \
    '[distil_whisper] repo=distil-whisper/distil-large-v3.5 revision=728a7691f3ff1d3d971528d3203a6e9559165d41' \
    'test parity_whisper_extras_distil_whisper ... [parity_whisper_extras] encoder hidden: max |Δ|=2.0e-5 at 1, atol=1.0e-2' \
    '[parity_whisper_extras/distil_whisper] native CPU real-weight parity PASS' \
    'ok' '' \
    'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 18 filtered out; finished in 45.44s' > "$valid_cpu"
  require_cpu_log distil_whisper "$valid_cpu" "$(sha256_file "$valid_cpu")"
  printf '%s\n' \
    'test parity_whisper_extras_distil_whisper_apple_cpu_metal ... [parity_whisper_extras] Apple CPU encoder/reference: max |Δ|=2.0e-5 at 1, atol=1.0e-2' \
    'WHISPER_EXTRAS_APPLE variant=distil_whisper cpu_reference=PASS metal_reference=PASS metal_cpu=PASS greedy_tokens=EXACT no_fallback=PASS verdict=PASS' \
    'ok' '' \
    'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 18 filtered out; finished in 95.44s' > "$valid_apple"
  require_apple_log distil_whisper "$valid_apple" "$(sha256_file "$valid_apple")"
  mutated="$temporary/mutated.log"
  for outcome in ignored skipped FAILED; do
    awk -v outcome="$outcome" 'BEGIN { replaced = 0 } /^test / && !replaced { sub(/\.\.\..*/, "... " outcome); replaced = 1 } { print }' "$valid_cpu" > "$mutated"
    if require_cpu_log distil_whisper "$mutated" "$(sha256_file "$mutated")" >/dev/null 2>&1; then die "CPU $outcome test was accepted"; fi
    awk -v outcome="$outcome" 'BEGIN { replaced = 0 } /^test / && !replaced { sub(/\.\.\..*/, "... " outcome); replaced = 1 } { print }' "$valid_apple" > "$mutated"
    if require_apple_log distil_whisper "$mutated" "$(sha256_file "$mutated")" >/dev/null 2>&1; then die "Apple $outcome test was accepted"; fi
  done
  cp "$valid_apple" "$mutated"
  printf '%s\n' 'test unexpected_extra ... ok' >> "$mutated"
  if require_apple_log distil_whisper "$mutated" "$(sha256_file "$mutated")" >/dev/null 2>&1; then die 'extra Apple test was accepted'; fi
  cp "$valid_apple" "$mutated"
  printf '%s\n' 'ok' >> "$mutated"
  if require_apple_log distil_whisper "$mutated" "$(sha256_file "$mutated")" >/dev/null 2>&1; then die 'duplicate standalone ok was accepted'; fi
  cp "$valid_apple" "$mutated"
  printf '%s\n' 'test result: FAILED. 1 passed; 1 failed; 0 ignored; 0 measured; 18 filtered out' >> "$mutated"
  if require_apple_log distil_whisper "$mutated" "$(sha256_file "$mutated")" >/dev/null 2>&1; then die 'failure result was accepted'; fi
  rm -rf "$temporary"
  if "$self" --self-test --expected-head 0123456789012345678901234567890123456789 >/dev/null 2>&1; then die '--self-test accepted extra arguments'; fi
  if "$self" --unknown-option >/dev/null 2>&1; then die 'unknown option accepted'; fi
  if "$self" --self-test --self-test >/dev/null 2>&1; then die 'duplicate --self-test accepted'; fi
  log 'self-test PASS'
}

mark_seen() {
  local key="$1"
  case " $seen_options " in
    *" $key "*) die "duplicate option: $key"; return 2 ;;
  esac
  seen_options="$seen_options $key"
}

distil_gguf=''; distil_gguf_sha=''; distil_reference=''; distil_manifest_sha=''; distil_packet_sha=''; distil_cpu_log=''; distil_cpu_sha=''
kotoba_gguf=''; kotoba_gguf_sha=''; kotoba_reference=''; kotoba_manifest_sha=''; kotoba_packet_sha=''; kotoba_cpu_log=''; kotoba_cpu_sha=''
vast_head_file=''; vast_head_sha=''; expected_head=''; evidence_dir=''; self=0
seen_options=''
while (($#)); do
  case "$1" in
    --self-test) mark_seen self; self=1; shift ;;
    --distil-gguf) mark_seen distil_gguf; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die 'invalid --distil-gguf'; distil_gguf="$2"; shift 2 ;;
    --distil-gguf-sha256) mark_seen distil_gguf_sha; [[ $# -ge 2 ]] || die 'invalid --distil-gguf-sha256'; distil_gguf_sha="$2"; shift 2 ;;
    --distil-reference) mark_seen distil_reference; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die 'invalid --distil-reference'; distil_reference="$2"; shift 2 ;;
    --distil-reference-manifest-sha256) mark_seen distil_manifest_sha; [[ $# -ge 2 ]] || die 'invalid --distil-reference-manifest-sha256'; distil_manifest_sha="$2"; shift 2 ;;
    --distil-reference-packet-sha256) mark_seen distil_packet_sha; [[ $# -ge 2 ]] || die 'invalid --distil-reference-packet-sha256'; distil_packet_sha="$2"; shift 2 ;;
    --distil-cpu-log) mark_seen distil_cpu_log; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die 'invalid --distil-cpu-log'; distil_cpu_log="$2"; shift 2 ;;
    --distil-cpu-log-sha256) mark_seen distil_cpu_sha; [[ $# -ge 2 ]] || die 'invalid --distil-cpu-log-sha256'; distil_cpu_sha="$2"; shift 2 ;;
    --kotoba-gguf) mark_seen kotoba_gguf; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die 'invalid --kotoba-gguf'; kotoba_gguf="$2"; shift 2 ;;
    --kotoba-gguf-sha256) mark_seen kotoba_gguf_sha; [[ $# -ge 2 ]] || die 'invalid --kotoba-gguf-sha256'; kotoba_gguf_sha="$2"; shift 2 ;;
    --kotoba-reference) mark_seen kotoba_reference; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die 'invalid --kotoba-reference'; kotoba_reference="$2"; shift 2 ;;
    --kotoba-reference-manifest-sha256) mark_seen kotoba_manifest_sha; [[ $# -ge 2 ]] || die 'invalid --kotoba-reference-manifest-sha256'; kotoba_manifest_sha="$2"; shift 2 ;;
    --kotoba-reference-packet-sha256) mark_seen kotoba_packet_sha; [[ $# -ge 2 ]] || die 'invalid --kotoba-reference-packet-sha256'; kotoba_packet_sha="$2"; shift 2 ;;
    --kotoba-cpu-log) mark_seen kotoba_cpu_log; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die 'invalid --kotoba-cpu-log'; kotoba_cpu_log="$2"; shift 2 ;;
    --kotoba-cpu-log-sha256) mark_seen kotoba_cpu_sha; [[ $# -ge 2 ]] || die 'invalid --kotoba-cpu-log-sha256'; kotoba_cpu_sha="$2"; shift 2 ;;
    --vast-head-file) mark_seen vast_head_file; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die 'invalid --vast-head-file'; vast_head_file="$2"; shift 2 ;;
    --vast-head-sha256) mark_seen vast_head_sha; [[ $# -ge 2 ]] || die 'invalid --vast-head-sha256'; vast_head_sha="$2"; shift 2 ;;
    --expected-head) mark_seen expected_head; [[ $# -ge 2 ]] || die 'invalid --expected-head'; expected_head="$2"; shift 2 ;;
    --evidence-dir) mark_seen evidence_dir; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die 'invalid --evidence-dir'; evidence_dir="$2"; shift 2 ;;
    -h|--help) [[ $# == 1 ]] || die '--help cannot be combined with arguments'; usage; exit 0 ;;
    *) usage; die "unknown option: $1" ;;
  esac
done

if (( self )); then
  [[ "$seen_options" == ' self' ]] || die '--self-test accepts no other arguments'
  self_test
  exit 0
fi

[[ "$expected_head" =~ ^[0-9a-f]{40}$ ]] || die '--expected-head must be lowercase HEX40'
[[ -n "$distil_gguf$distil_reference$distil_cpu_log$kotoba_gguf$kotoba_reference$kotoba_cpu_log$vast_head_file$evidence_dir" ]] || die 'all artifact, CPU evidence, head, and evidence paths are required'
for value in "$distil_gguf_sha" "$distil_manifest_sha" "$distil_packet_sha" "$distil_cpu_sha" "$kotoba_gguf_sha" "$kotoba_manifest_sha" "$kotoba_packet_sha" "$kotoba_cpu_sha" "$vast_head_sha"; do
  [[ "$value" =~ ^[0-9a-f]{64}$ ]] || die 'all supplied digests must be lowercase HEX64'
done
[[ -f "$VOKRA_ROOT/Cargo.toml" && -d "$VOKRA_ROOT/.git" && -f "$TEST_SOURCE" && -f "$PRODUCER" ]] || die 'required Vokra checkout, test, or VAST producer is missing'
[[ -z "$(git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all)" ]] || die 'Apple checkout must be clean'
[[ "$(git -C "$VOKRA_ROOT" rev-parse HEAD)" == "$expected_head" ]] || die 'checkout HEAD does not match --expected-head'
require_remote_apple
require_digest 'VAST clean-head evidence' "$vast_head_file" "$vast_head_sha"
[[ "$(tr -d '[:space:]' < "$vast_head_file")" == "$expected_head" ]] || die 'VAST clean-head evidence does not bind expected HEAD'

for pair in \
  "distil_whisper|$distil_gguf|$distil_gguf_sha|$distil_reference|$distil_manifest_sha|$distil_packet_sha|$distil_cpu_log|$distil_cpu_sha" \
  "kotoba_whisper|$kotoba_gguf|$kotoba_gguf_sha|$kotoba_reference|$kotoba_manifest_sha|$kotoba_packet_sha|$kotoba_cpu_log|$kotoba_cpu_sha"; do
  IFS='|' read -r variant gguf gguf_sha reference manifest_sha packet_sha cpu_log cpu_sha <<< "$pair"
  reject_symlink_ancestors "$gguf"; reject_symlink_ancestors "$reference"; reject_symlink_ancestors "$cpu_log"
  require_digest "$variant GGUF" "$gguf" "$gguf_sha"
  validate_manifest "$variant" "$reference" "$manifest_sha" "$packet_sha"
  require_cpu_log "$variant" "$cpu_log" "$cpu_sha"
done
require_disjoint_absent_evidence "$evidence_dir" "$VOKRA_ROOT" "$distil_gguf" "$distil_reference" "$distil_cpu_log" "$kotoba_gguf" "$kotoba_reference" "$kotoba_cpu_log" "$vast_head_file"
mkdir "$evidence_dir"
printf 'utc=%s\nexpected_head=%s\nactual_head=%s\nmetal_compiler=%s\npublication=NO_UPLOAD\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$expected_head" "$(git -C "$VOKRA_ROOT" rev-parse HEAD)" "$(xcrun -f metal)" > "$evidence_dir/environment.txt"
printf 'distil_gguf_sha256=%s\ndistil_reference_manifest_sha256=%s\ndistil_reference_packet_sha256=%s\ndistil_cpu_log_sha256=%s\nkotoba_gguf_sha256=%s\nkotoba_reference_manifest_sha256=%s\nkotoba_reference_packet_sha256=%s\nkotoba_cpu_log_sha256=%s\nvast_head_sha256=%s\n' "$distil_gguf_sha" "$distil_manifest_sha" "$distil_packet_sha" "$distil_cpu_sha" "$kotoba_gguf_sha" "$kotoba_manifest_sha" "$kotoba_packet_sha" "$kotoba_cpu_sha" "$vast_head_sha" > "$evidence_dir/input-hashes.txt"

run_apple_test distil_whisper "$distil_gguf" "$distil_reference" parity_whisper_extras_distil_whisper_apple_cpu_metal "$evidence_dir/distil_whisper-apple.log"
run_apple_test kotoba_whisper "$kotoba_gguf" "$kotoba_reference" parity_whisper_extras_kotoba_whisper_apple_cpu_metal "$evidence_dir/kotoba_whisper-apple.log"
{
  echo 'verdict=PASS'
  echo "expected_head=$expected_head"
  echo "actual_head=$(git -C "$VOKRA_ROOT" rev-parse HEAD)"
  echo 'distil_whisper_cpu_reference=PASS'
  echo 'distil_whisper_metal_reference=PASS'
  echo 'distil_whisper_metal_cpu=PASS'
  echo 'distil_whisper_greedy_tokens=EXACT'
  echo 'kotoba_whisper_cpu_reference=PASS'
  echo 'kotoba_whisper_metal_reference=PASS'
  echo 'kotoba_whisper_metal_cpu=PASS'
  echo 'kotoba_whisper_greedy_tokens=EXACT'
  echo 'no_fallback=PASS'
  echo "publication=NO_UPLOAD (NO_UPLOAD=$NO_UPLOAD)"
} > "$evidence_dir/summary.txt"
log "PASS: Apple evidence recorded at $evidence_dir; publication=NO_UPLOAD"
