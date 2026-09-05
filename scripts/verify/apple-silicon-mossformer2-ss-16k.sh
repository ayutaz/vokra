#!/usr/bin/env bash
# Real-weight MossFormer2-SS-16K CPU/Metal measurement on disposable Apple Silicon.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VOKRA_ROOT="${VOKRA_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
PARITY_PROJECT="$VOKRA_ROOT/tools/parity/mossformer2_ss_16k"
PREFLIGHT_GATE="$PARITY_PROJECT/preflight_gate.py"
PREFLIGHT_MANIFEST="$PARITY_PROJECT/license_gate_manifest.json"
REFERENCE_DUMPER="$VOKRA_ROOT/tools/parity/mossformer2_ss_16k_dump_reference.py"
PUBLIC_BYTES=223058240
PUBLIC_SHA256="822516b75873dbeb814dac72f7ca0b5fb75254dd051dfdfdda54987347330f0c"
MIN_MEMORY_BYTES=24000000000

log() { printf '[mossformer2-apple] %s\n' "$*" >&2; }
die() { log "ERROR: $*"; return 2; }
sha256_file() { shasum -a 256 "$1" | awk '{print $1}'; }

license_preflight() {
  local approval="$1"
  command -v uv >/dev/null 2>&1 || die "required tool missing: uv"
  [[ -f "$PREFLIGHT_GATE" && ! -L "$PREFLIGHT_GATE" && -f "$PREFLIGHT_MANIFEST" && ! -L "$PREFLIGHT_MANIFEST" ]] || die "MossFormer2 preflight inputs are missing"
  [[ -f "$approval" && ! -L "$approval" && -s "$approval" ]] || die "approval evidence is missing, symlinked, or empty"
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python "$PREFLIGHT_GATE" --project "$PARITY_PROJECT" --manifest "$PREFLIGHT_MANIFEST" --evidence "$approval"
}

require_disjoint_evidence() {
  local evidence="$1" parent candidate other other_parent real
  [[ ! -e "$evidence" && ! -L "$evidence" ]] || { die "evidence directory must be absent before validation"; return 2; }
  parent="$(cd -P "$(dirname "$evidence")" 2>/dev/null && pwd)" || { die "evidence parent is inaccessible"; return 2; }
  candidate="$parent/$(basename "$evidence")"
  shift
  for other in "$@"; do
    [[ ! -L "$other" ]] || { die "validation input is a symlink: $other"; return 2; }
    other_parent="$(cd -P "$(dirname "$other")" 2>/dev/null && pwd)" || { die "validation input is inaccessible: $other"; return 2; }
    real="$other_parent/$(basename "$other")"
    [[ "$candidate" != "$real" && "$candidate/" != "$real/"* && "$real/" != "$candidate/"* ]] || { die "evidence directory overlaps validation input"; return 2; }
  done
  mkdir -p "$evidence"
}

usage() {
  cat >&2 <<'EOF'
usage: apple-silicon-mossformer2-ss-16k.sh --gguf PATH --gguf-sha256 HEX64 \
  --reference DIR --reference-sha256 HEX64 --approval-evidence FILE --evidence-dir ABSENT_DIR
       apple-silicon-mossformer2-ss-16k.sh --self-test
EOF
}

require_file_set() {
  local directory="$1" entry name
  local expected='manifest.json pcm.f32.bin encoder.f32.bin attention_0.f32.bin fsmn_0.f32.bin mask.f32.bin separated.f32.bin'
  [[ -d "$directory" && ! -L "$directory" ]] || die "reference directory is missing or symlinked: $directory"
  while IFS= read -r entry; do
    name="$(basename "$entry")"
    [[ -f "$entry" && ! -L "$entry" ]] || { die "reference output is not regular: $name"; return 2; }
    case " $expected " in *" $name "*) ;; *) die "unexpected reference output: $name"; return 2 ;; esac
  done < <(find -P "$directory" -mindepth 1 -maxdepth 1 -print)
  for name in $expected; do [[ -f "$directory/$name" && ! -L "$directory/$name" ]] || { die "missing reference output: $name"; return 2; }; done
}

require_regular_nonempty_file() {
  local label="$1" path="$2"
  [[ -f "$path" && ! -L "$path" && -s "$path" ]] || die "$label is missing, empty, or symlinked: $path"
}

verify_reference() {
  local reference="$1" expected_manifest="$2"
  [[ "$expected_manifest" =~ ^[0-9a-f]{64}$ ]] || die "reference manifest hash is not lowercase hex"
  [[ "$(sha256_file "$reference/manifest.json")" == "$expected_manifest" ]] || die "reference manifest SHA-256 mismatch"
  [[ -f "$REFERENCE_DUMPER" ]] || die "reference validator is missing"
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python "$REFERENCE_DUMPER" \
    --validate-reference "$reference"
}

require_remote_host() {
  [[ "${VOKRA_REMOTE_APPLE_SILICON:-0}" == 1 ]] || die "VOKRA_REMOTE_APPLE_SILICON=1 is required"
  [[ "$(uname -s)" == Darwin && "$(uname -m)" == arm64 ]] || die "Apple Silicon Darwin is required"
  local memory_bytes; memory_bytes="$(sysctl -n hw.memsize)"
  [[ "$memory_bytes" =~ ^[0-9]+$ && $memory_bytes -ge $MIN_MEMORY_BYTES ]] || die "physical memory is below remote-worker guard"
}

require_tooling() {
  local tool
  for tool in cargo rustc git shasum awk find tee grep sysctl sw_vers xcrun uv wc tr; do
    command -v "$tool" >/dev/null 2>&1 || die "required tool missing: $tool"
  done
  [[ -d "$VOKRA_ROOT/.git" ]] || die "Vokra checkout is missing"
  [[ -z "$(git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all)" ]] || die "Apple checkout must be clean"
  xcrun -f metal >/dev/null 2>&1 || die "Metal compiler is unavailable"
}

require_one_result() {
  local log_path="$1" backend="$2" name='mossformer2_ss_16k::tests::measure_real_cpu_and_optional_metal_against_official'
  local test_count ok_count name_count result_count total_result marker_family_count marker_count
  test_count="$(grep -Ec '^test .* \.\.\.' "$log_path" || true)"
  ok_count="$(grep -Ec "^test ${name} \.\.\. ok$" "$log_path" || true)"
  name_count="$(grep -Ec "^test ${name} \.\.\." "$log_path" || true)"
  result_count="$(grep -Ec '^test result: ok\. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out(; finished in [0-9]+\.[0-9]+s)?$' "$log_path" || true)"
  total_result="$(grep -Ec '^test result:' "$log_path" || true)"
  marker_family_count="$(grep -Ec '^MOSSFORMER2_SS_16K_MEASUREMENT_ONLY ' "$log_path" || true)"
  marker_count="$(grep -Ec "^MOSSFORMER2_SS_16K_MEASUREMENT_ONLY backend=${backend} numeric_bounds=UNSET verdict=MEASURED_NOT_GATED max_abs=[0-9.e+-]+ rms=[0-9.e+-]+ relative_l1=[0-9.e+-]+ index=[0-9]+ (actual|metal)=[0-9.e+-]+ (reference|cpu)=[0-9.e+-]+$" "$log_path" || true)"
  [[ "$test_count" == 1 && "$ok_count" == 1 && "$name_count" == 1 && "$result_count" == 1 && "$total_result" == 1 && "$marker_family_count" == 1 && "$marker_count" == 1 ]] || { die "${backend} test/result/measurement lines are not exact singletons"; return 2; }
  ! grep -Eq '^MOSSFORMER2_SS_16K_MEASUREMENT_ONLY .*FAIL$' "$log_path" || { die "measurement FAIL marker present"; return 2; }
}

require_arguments() {
  [[ -n "$1" && -n "$2" && -n "$3" && -n "$4" && -n "$5" && -n "$6" ]]
}

run_self_test() {
  local tmp; tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
  mkdir "$tmp/reference"; for name in manifest.json pcm.f32.bin encoder.f32.bin attention_0.f32.bin fsmn_0.f32.bin mask.f32.bin separated.f32.bin; do : > "$tmp/reference/$name"; done
  require_file_set "$tmp/reference"; : > "$tmp/reference/extra"; if require_file_set "$tmp/reference" >/dev/null 2>&1; then die "extra reference file accepted"; fi
  printf 'gguf-self-test\n' > "$tmp/gguf"
  require_regular_nonempty_file "self-test GGUF" "$tmp/gguf"
  mkdir "$tmp/gguf-dir"
  if require_regular_nonempty_file "self-test directory GGUF" "$tmp/gguf-dir" >/dev/null 2>&1; then die "GGUF directory accepted"; fi
  ln -s "$tmp/gguf" "$tmp/gguf-link"
  if require_regular_nonempty_file "self-test symlink GGUF" "$tmp/gguf-link" >/dev/null 2>&1; then die "GGUF symlink accepted"; fi
  ln -s "$tmp/missing" "$tmp/gguf-dangling"
  if require_regular_nonempty_file "self-test dangling GGUF" "$tmp/gguf-dangling" >/dev/null 2>&1; then die "dangling GGUF symlink accepted"; fi
  : > "$tmp/gguf-empty"
  if require_regular_nonempty_file "self-test empty GGUF" "$tmp/gguf-empty" >/dev/null 2>&1; then die "empty GGUF accepted"; fi
  local name='mossformer2_ss_16k::tests::measure_real_cpu_and_optional_metal_against_official'
  printf '%s\n' "test $name ... ok" 'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s' 'MOSSFORMER2_SS_16K_MEASUREMENT_ONLY backend=cpu numeric_bounds=UNSET verdict=MEASURED_NOT_GATED max_abs=1.0e-3 rms=1.0e-4 relative_l1=1.0e-4 index=0 actual=1.0e-1 reference=1.0e-1' > "$tmp/log"
  require_one_result "$tmp/log" cpu
  for malformed in different_test failed_target duplicate_result duplicate_marker malformed_result malformed_marker; do
    cp "$tmp/log" "$tmp/$malformed.log"
    case "$malformed" in
      different_test) printf '%s\n' 'test another_test ... ok' >> "$tmp/$malformed.log" ;;
      failed_target) sed 's/\.\.\. ok$/... FAILED/' "$tmp/$malformed.log" > "$tmp/x" && mv "$tmp/x" "$tmp/$malformed.log" ;;
      duplicate_result) printf '%s\n' 'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out' >> "$tmp/$malformed.log" ;;
      duplicate_marker) printf '%s\n' 'MOSSFORMER2_SS_16K_MEASUREMENT_ONLY backend=cpu malformed' >> "$tmp/$malformed.log" ;;
      malformed_result) sed 's/filtered out; finished in 0.01s$/filtered out; finished in nonsense/' "$tmp/$malformed.log" > "$tmp/x" && mv "$tmp/x" "$tmp/$malformed.log" ;;
      malformed_marker) sed 's/numeric_bounds=UNSET /numeric_bounds=UNSET trailing /' "$tmp/$malformed.log" > "$tmp/x" && mv "$tmp/x" "$tmp/$malformed.log" ;;
    esac
    if require_one_result "$tmp/$malformed.log" cpu >/dev/null 2>&1; then die "accepted malformed $malformed Cargo/measurement log"; fi
  done
  grep -Fq -- '--features metal' "$0" || die "Metal run does not enable the metal feature"
  grep -Fq -- '--validate-reference' "$0" || die "Apple path does not use the strict shared reference validator"
  grep -Fq -- 'UV_NO_CACHE=1 uv run --no-cache' "$0" || die "Apple stdlib validation is allowed to create a UV cache"
  local valid_a='a' valid_b='b' valid_c='c' valid_d='d' valid_e='e' valid_f='f'
  for missing in 1 2 3 4 5 6; do
    case "$missing" in
      1) if require_arguments "" "$valid_b" "$valid_c" "$valid_d" "$valid_e" "$valid_f"; then die "missing gguf accepted"; fi ;;
      2) if require_arguments "$valid_a" "" "$valid_c" "$valid_d" "$valid_e" "$valid_f"; then die "missing gguf digest accepted"; fi ;;
      3) if require_arguments "$valid_a" "$valid_b" "" "$valid_d" "$valid_e" "$valid_f"; then die "missing reference accepted"; fi ;;
      4) if require_arguments "$valid_a" "$valid_b" "$valid_c" "" "$valid_e" "$valid_f"; then die "missing reference digest accepted"; fi ;;
      5) if require_arguments "$valid_a" "$valid_b" "$valid_c" "$valid_d" "" "$valid_f"; then die "missing approval accepted"; fi ;;
      6) if require_arguments "$valid_a" "$valid_b" "$valid_c" "$valid_d" "$valid_e" ""; then die "missing evidence directory accepted"; fi ;;
    esac
  done
  # shellcheck disable=SC2086 # Each case intentionally models argv tokenization.
  for bad_args in "--self-test --approval-evidence x" "--self-test --self-test" "--gguf x --gguf y" "--approval-evidence" "--unknown x"; do
    if bash "$0" $bad_args >/dev/null 2>&1; then die "accepted malformed parser case: $bad_args"; fi
  done
  trap - EXIT
  log 'self-test PASS'
}

main() {
  local gguf='' gguf_digest='' reference='' reference_digest='' approval='' evidence='' self_test=0
  local seen_gguf=0 seen_gguf_digest=0 seen_reference=0 seen_reference_digest=0 seen_approval=0 seen_evidence=0 seen_self_test=0
  while (($#)); do
    case "$1" in
      --gguf) (( seen_gguf == 0 )) || die 'duplicate --gguf'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die '--gguf requires a nonempty value'; seen_gguf=1; gguf="$2"; shift 2;;
      --gguf-sha256) (( seen_gguf_digest == 0 )) || die 'duplicate --gguf-sha256'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die '--gguf-sha256 requires a nonempty value'; seen_gguf_digest=1; gguf_digest="$2"; shift 2;;
      --reference) (( seen_reference == 0 )) || die 'duplicate --reference'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die '--reference requires a nonempty value'; seen_reference=1; reference="$2"; shift 2;;
      --reference-sha256) (( seen_reference_digest == 0 )) || die 'duplicate --reference-sha256'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die '--reference-sha256 requires a nonempty value'; seen_reference_digest=1; reference_digest="$2"; shift 2;;
      --approval-evidence) (( seen_approval == 0 )) || die 'duplicate --approval-evidence'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die '--approval-evidence requires a nonempty value'; seen_approval=1; approval="$2"; shift 2;;
      --evidence-dir) (( seen_evidence == 0 )) || die 'duplicate --evidence-dir'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die '--evidence-dir requires a nonempty value'; seen_evidence=1; evidence="$2"; shift 2;;
      --self-test) (( seen_self_test == 0 )) || die 'duplicate --self-test'; seen_self_test=1; self_test=1; shift;;
      -h|--help) usage; return 0;; *) usage; die "unknown argument $1";;
    esac
  done
  if ((self_test)); then
    [[ -z "$gguf$gguf_digest$reference$reference_digest$approval$evidence" ]] || die '--self-test accepts no other arguments'
    run_self_test; return
  fi
  require_arguments "$gguf" "$gguf_digest" "$reference" "$reference_digest" "$approval" "$evidence" || { usage; die "all arguments are required"; }
  [[ "$gguf_digest" =~ ^[0-9a-f]{64}$ && "$reference_digest" =~ ^[0-9a-f]{64}$ ]] || die "expected SHA-256 values must be lowercase 64-hex"
  license_preflight "$approval"
  require_remote_host; require_tooling; require_file_set "$reference"
  [[ "$gguf_digest" == "$PUBLIC_SHA256" ]] || die "public GGUF hash is not fixed identity"
  require_regular_nonempty_file "public GGUF" "$gguf"
  [[ "$(wc -c < "$gguf" | tr -d '[:space:]')" == "$PUBLIC_BYTES" ]] || die "public GGUF byte identity mismatch"
  [[ "$(sha256_file "$gguf")" == "$gguf_digest" ]] || die "public GGUF SHA-256 mismatch"
  verify_reference "$reference" "$reference_digest"
  require_disjoint_evidence "$evidence" "$VOKRA_ROOT" "$gguf" "$reference" "$approval"
  for backend in cpu metal; do
    log "running $backend measurement"
    if [[ "$backend" == metal ]]; then
      env VOKRA_MOSSFORMER2_GGUF="$gguf" VOKRA_MOSSFORMER2_REFERENCE_DIR="$reference" \
        VOKRA_MOSSFORMER2_METAL_MEASUREMENT=1 RUST_TEST_THREADS=1 \
        cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release -p vokra-models --lib \
          --features metal \
          mossformer2_ss_16k::tests::measure_real_cpu_and_optional_metal_against_official -- --ignored --exact --nocapture 2>&1 | tee "$evidence/measurement-$backend.log"
    else
      env VOKRA_MOSSFORMER2_GGUF="$gguf" VOKRA_MOSSFORMER2_REFERENCE_DIR="$reference" \
        RUST_TEST_THREADS=1 cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release -p vokra-models --lib \
          mossformer2_ss_16k::tests::measure_real_cpu_and_optional_metal_against_official -- --ignored --exact --nocapture 2>&1 | tee "$evidence/measurement-$backend.log"
    fi
    require_one_result "$evidence/measurement-$backend.log" "$backend"
  done
  printf 'gguf_sha256=%s\nreference_manifest_sha256=%s\nnumeric_verdict=MEASURED_NOT_GATED\nnumeric_bounds=UNSET\nupload=NOT_PERFORMED\n' "$gguf_digest" "$reference_digest" > "$evidence/input-hashes.txt"
}
main "$@"
