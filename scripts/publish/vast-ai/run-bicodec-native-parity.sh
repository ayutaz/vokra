#!/usr/bin/env bash
# VAST-only official-reference evidence worker for native BiCodec decode.
#
# This worker does not upload weights. The Python dumper authenticates the
# exact Spark-TTS source/checkpoint/config and records the official semantic
# latent, d-vector, prenet output, and waveform; the Rust test then applies
# the reviewed, stage-specific measured parity gate to those records.
set -euo pipefail

die() {
  echo "run-bicodec-native-parity: $*" >&2
  exit 1
}

require_cpu_parity_pass() {
  local log_path="$1" stage count
  [[ "$(grep -Fxc 'test bicodec::tests::official_reference_measured_parity ... ok' "$log_path" || true)" == 1 ]] \
    || die 'BiCodec CPU parity named test did not pass exactly once'
  [[ "$(grep -Ec '^test result: ok[.] 1 passed; 0 failed; 0 ignored; 0 measured; [0-9]+ filtered out(; finished in .+)?$' "$log_path" || true)" == 1 && "$(grep -Ec '^test result:' "$log_path" || true)" == 1 ]] \
    || die 'BiCodec CPU parity result was not exactly one pass'
  for stage in semantic_latent d_vector prenet_output waveform; do
    count="$(grep -Ec "^BICODEC_MEASURED_PARITY stage=$stage .* verdict=PASS$" "$log_path" || true)"
    [[ "$count" == 1 ]] || die "BiCodec CPU parity stage marker is not unique: $stage"
  done
  [[ "$(grep -Fxc 'BICODEC_MEASURED_PARITY_BACKEND backend=cpu verdict=PASS' "$log_path" || true)" == 1 ]] \
    || die 'BiCodec CPU backend sentinel was not emitted exactly once'
  ! grep -Eq '^BICODEC_MEASURED_PARITY .* verdict=FAIL$' "$log_path" \
    || die 'BiCodec CPU parity log contains a failed stage marker'
}

run_self_test() (
  local temporary project lock
  temporary="$(mktemp -d "${TMPDIR:-/tmp}/vokra-bicodec-vast.XXXXXX")"
  trap 'rm -rf -- "$temporary"' EXIT
  project="$(cd "$(dirname "$0")/../../.." && pwd)/tools/parity/pyproject.toml"
  lock="$(dirname "$project")/uv.lock"
  [[ -f "$project" && -f "$lock" ]] || die 'parity project or lock is missing'
  grep -Fq 'reference-only' "$project" || die 'reference-only dependency posture is missing'
  grep -Fq 'no-upload' "$project" || die 'no-upload dependency posture is missing'
  grep -Fq '"einx==0.4.3"' "$project" || die 'exact einx dependency is missing'
  grep -Fq 'name = "einx"' "$lock" || die 'exact einx lock row is missing'
  grep -Fq 'hash = "sha256:be7d81ea1908b9f00e4a467840998fc483c33aa32aaaaa3ada6c8386f693edf9"' "$lock" || die 'einx sdist digest is not pinned'
  grep -Fq 'hash = "sha256:47ce54a0144f6dffcfacdd8fe2cc9e2e5e6485dda2471330ab75ee747dd22f39"' "$lock" || die 'einx wheel digest is not pinned'
  grep -Fq 'name = "frozendict"' "$lock" || die 'exact frozendict lock row is missing'
  grep -Fq 'hash = "sha256:e478fb2a1391a56c8a6e10cc97c4a9002b410ecd1ac28c18d780661762e271bd"' "$lock" || die 'frozendict sdist digest is not pinned'
  grep -Fq 'hash = "sha256:972af65924ea25cf5b4d9326d549e69a9a4918d8a76a9d3a7cd174d98b237550"' "$lock" || die 'frozendict wheel digest is not pinned'
  printf '%s\n' \
    'test bicodec::tests::official_reference_measured_parity ... ok' \
    'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 2975 filtered out' \
    'BICODEC_MEASURED_PARITY stage=semantic_latent elements=1 max_abs=0 rmse=0 verdict=PASS' \
    'BICODEC_MEASURED_PARITY stage=d_vector elements=1 max_abs=0 rmse=0 verdict=PASS' \
    'BICODEC_MEASURED_PARITY stage=prenet_output elements=1 max_abs=0 rmse=0 verdict=PASS' \
    'BICODEC_MEASURED_PARITY stage=waveform elements=1 max_abs=0 rmse=0 verdict=PASS' \
    'BICODEC_MEASURED_PARITY_BACKEND backend=cpu verdict=PASS' > "$temporary/valid.log"
  require_cpu_parity_pass "$temporary/valid.log"
  cp "$temporary/valid.log" "$temporary/duplicate.log"
  printf '%s\n' 'BICODEC_MEASURED_PARITY stage=waveform elements=1 max_abs=0 rmse=0 verdict=PASS' >> "$temporary/duplicate.log"
  if (require_cpu_parity_pass "$temporary/duplicate.log") >/dev/null 2>&1; then die 'duplicate stage marker accepted'; fi
  cp "$temporary/valid.log" "$temporary/failure.log"
  printf '%s\n' 'BICODEC_MEASURED_PARITY stage=waveform elements=1 max_abs=1 rmse=1 verdict=FAIL' >> "$temporary/failure.log"
  if (require_cpu_parity_pass "$temporary/failure.log") >/dev/null 2>&1; then die 'failure marker accepted'; fi
  sed '/BICODEC_MEASURED_PARITY_BACKEND/d' "$temporary/valid.log" > "$temporary/missing-sentinel.log"
  if (require_cpu_parity_pass "$temporary/missing-sentinel.log") >/dev/null 2>&1; then die 'missing backend sentinel accepted'; fi
  grep -Fq -- 'VOKRA_BICODEC_PARITY_BACKEND=cpu' "$0" || die 'CPU selector missing from production command'
  grep -Fq -- 'cargo test --locked --lib -p vokra-models' "$0" || die 'production command lacks --lib'
  grep -Fq -- '-- --ignored --exact --show-output' "$0" || die 'production command lacks harness --exact/show-output'
  echo 'run-bicodec-native-parity.sh self-test: OK'
)

usage() {
  cat <<'EOF'
Usage:
  run-bicodec-native-parity.sh --source-dir <checkout> --model-dir <BiCodec> \
    --output <empty-tmpfs-dir>
  run-bicodec-native-parity.sh --self-test

Requires Linux x86_64, VAST, and the repository Python environment. Inputs are
authenticated by bicodec_dump_reference.py. The worker never publishes or
uploads model artifacts. --self-test is hermetic and performs no Cargo,
network, model, or checkpoint operation.
EOF
}

source_dir=""
model_dir=""
output=""
seen_self_test=0
self_test=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --self-test) (( seen_self_test == 0 )) || die 'duplicate --self-test'; seen_self_test=1; self_test=1; shift ;;
    --source-dir) [[ $# -ge 2 ]] || die "--source-dir requires a path"; source_dir="$2"; shift 2 ;;
    --model-dir) [[ $# -ge 2 ]] || die "--model-dir requires a path"; model_dir="$2"; shift 2 ;;
    --output) [[ $# -ge 2 ]] || die "--output requires a path"; output="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done
if (( self_test )); then
  [[ -z "$source_dir$model_dir$output" ]] || die '--self-test accepts no other arguments'
  run_self_test
  exit 0
fi
[[ -n "$source_dir" && -n "$model_dir" && -n "$output" ]] || { usage >&2; exit 1; }
[[ "$(uname -s)" == "Linux" ]] || die "official reference runs on Linux/VAST only"
[[ "$(uname -m)" == "x86_64" ]] || die "official reference requires x86_64"
[[ "${VOKRA_PUBLISH_ON_VAST:-0}" == "1" ]] || die "VOKRA_PUBLISH_ON_VAST=1 is absent"
[[ -f Cargo.toml && -d tools/parity ]] || die "run from a Vokra checkout"
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || die "worktree is not clean"
command -v uv >/dev/null 2>&1 || die "uv is required"
command -v findmnt >/dev/null 2>&1 || die "findmnt is required"
command -v cargo >/dev/null 2>&1 || die "cargo is required"
[[ "$(findmnt -T "$(dirname "$output")" -no FSTYPE 2>/dev/null || true)" == "tmpfs" ]] \
  || die "output parent must be tmpfs/RAM disk"

mkdir -p "$output"
[[ -z "$(find "$output" -mindepth 1 -maxdepth 1 -print -quit)" ]] || die "output must be empty"
uv run --frozen --project tools/parity --python 3.12 python \
  tools/parity/bicodec_dump_reference.py \
  --source-dir "$source_dir" --model-dir "$model_dir" --output "$output"
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-1}"
cargo build --locked --release -p vokra-cli
gguf_path="$output/bicodec.gguf"
target/release/vokra-cli convert \
  --model bicodec \
  --input "$model_dir/model.safetensors" \
  --output "$gguf_path" \
  --license cc-by-nc-sa-4.0
[[ -s "$gguf_path" ]] || die "authenticated BiCodec conversion produced no GGUF"
VOKRA_BICODEC_PARITY_GGUF="$gguf_path" \
VOKRA_BICODEC_PARITY_REFERENCE="$output" \
VOKRA_BICODEC_PARITY_BACKEND=cpu \
  cargo test --locked --lib -p vokra-models \
    bicodec::tests::official_reference_measured_parity -- --ignored --exact --show-output 2>&1 | tee "$output/parity-cpu.log"
require_cpu_parity_pass "$output/parity-cpu.log"
echo "BiCodec official reference evidence: $output"
