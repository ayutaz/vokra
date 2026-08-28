#!/usr/bin/env bash
# Exact public MusicGen Medium/Large companion routes on remote Apple Silicon.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VOKRA_ROOT="${VOKRA_ROOT:-$DEFAULT_ROOT}"

SMALL_BYTES=2364405568
SMALL_SHA256="be0a1d823cd4b4570e39cb87ce05a707959ffdffdc0aef23eb90fffa5c084a98"
MEDIUM_BYTES=3677520768
MEDIUM_SHA256="574072a7058c4a7bd5f60b7a773e219f659a029dc35cc6a4fd167b08e62fbc1c"
LARGE_BYTES=6513958784
LARGE_SHA256="d015b2dbe60b1ab85d0778d98c818413f46e71c91f9dcf04b2ff1088a9bc6ca9"

MIN_MEMORY_BYTES=48000000000
MIN_FREE_DISK_KIB=30000000

log() { printf '[musicgen-companion-apple] %s\n' "$*" >&2; }
step() { printf '\n[musicgen-companion-apple] ==== %s ====\n' "$*" >&2; }
die() { log "ERROR: $*"; return 2; }

usage() {
  cat <<'EOF' >&2
usage: apple-silicon-musicgen-companion.sh \
  --small <musicgen-small.gguf> --medium <musicgen-medium.gguf> \
  --large <musicgen-large.gguf> --evidence-dir <empty-dir>
       apple-silicon-musicgen-companion.sh --self-test

Runs complete CPU and real Metal routes for the exact public MusicGen Medium
and Large LM-only GGUFs with the exact public Small T5/EnCodec companion. It
also exercises the Medium CLI on Metal. The host must be a clean, disposable
Darwin/arm64 checkout with VOKRA_REMOTE_APPLE_SILICON=1, at least 48 GB RAM,
and pre-staged inputs matching the fixed byte sizes and SHA-256 identities.

This is a finite/non-zero real-weight route gate. It does not invent or claim
an independent AudioCraft numerical tolerance. The script does not download,
upload, convert, publish, delete, stop or destroy anything. Pull the evidence
directory, remove the separately staged model files, then destroy the worker.
EOF
}

sha256_file() {
  shasum -a 256 "$1" | awk '{print $1}'
}

verify_file() {
  local label="$1" path="$2" expected_bytes="$3" expected_hash="$4"
  local actual_bytes actual_hash
  [[ -f "$path" && -s "$path" ]] || die "$label is missing or empty: $path"
  actual_bytes="$(wc -c < "$path" | tr -d '[:space:]')"
  [[ "$actual_bytes" == "$expected_bytes" ]] \
    || die "$label byte size $actual_bytes != exact public artifact $expected_bytes"
  actual_hash="$(sha256_file "$path")"
  [[ "$actual_hash" == "$expected_hash" ]] \
    || die "$label SHA-256 $actual_hash != exact public artifact $expected_hash"
  log "identity OK: $label bytes=$actual_bytes sha256=$actual_hash"
}

require_empty_directory() {
  local directory="$1"
  if [[ -e "$directory" ]]; then
    [[ -d "$directory" ]] || die "evidence path is not a directory: $directory"
    [[ -z "$(find "$directory" -mindepth 1 -maxdepth 1 -print -quit)" ]] \
      || die "evidence directory must be empty: $directory"
  else
    mkdir -p "$directory"
  fi
}

require_remote_apple_host() {
  local memory_bytes free_disk_kib
  [[ "${VOKRA_REMOTE_APPLE_SILICON:-0}" == "1" ]] \
    || die "VOKRA_REMOTE_APPLE_SILICON=1 is absent; refusing possible maintainer-Mac execution"
  [[ "$(uname -s)" == "Darwin" ]] || die "real Metal validation requires Darwin"
  [[ "$(uname -m)" == "arm64" ]] || die "real Metal validation requires Apple arm64"
  memory_bytes="$(sysctl -n hw.memsize)"
  [[ "$memory_bytes" =~ ^[0-9]+$ ]] || die "could not read hw.memsize"
  if (( memory_bytes < MIN_MEMORY_BYTES )); then
    die "physical memory $memory_bytes bytes is below the 48-GB remote-worker guard"
  fi
  free_disk_kib="$(df -Pk "$VOKRA_ROOT" | awk 'NR == 2 {print $4}')"
  [[ "$free_disk_kib" =~ ^[0-9]+$ ]] || die "could not read free disk"
  if (( free_disk_kib < MIN_FREE_DISK_KIB )); then
    die "free disk $free_disk_kib KiB is below the 30-GB run guard"
  fi
}

require_tooling() {
  local tool
  for tool in cargo rustc git shasum awk find tee grep sysctl sw_vers \
    system_profiler xcrun wc tr; do
    command -v "$tool" >/dev/null 2>&1 || die "required tool missing: $tool"
  done
  [[ -d "$VOKRA_ROOT/.git" && -f "$VOKRA_ROOT/Cargo.toml" ]] \
    || die "$VOKRA_ROOT is not a repository checkout"
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

run_self_test() (
  local temporary payload_hash script_path
  temporary="$(mktemp -d "${TMPDIR:-/tmp}/vokra-musicgen-companion-apple.XXXXXX")"
  trap 'rm -rf "$temporary"' EXIT
  printf 'vokra-musicgen-companion-apple\n' > "$temporary/value"
  payload_hash="$(sha256_file "$temporary/value")"
  verify_file "self-test payload" "$temporary/value" \
    "$(wc -c < "$temporary/value" | tr -d '[:space:]')" "$payload_hash"
  require_empty_directory "$temporary/evidence"

  script_path="${BASH_SOURCE[0]}"
  for required in "$SMALL_SHA256" "$MEDIUM_SHA256" "$LARGE_SHA256" \
    "MUSICGEN_COMPANION_ROUTE backend=cpu" \
    "MUSICGEN_COMPANION_ROUTE backend=metal" \
    "public_musicgen_lm_only_companion_generates_finite_pcm" \
    "numerical_parity=NOT_CLAIMED"; do
    grep -Fq -- "$required" "$script_path" \
      || die "remote Apple contract lost token: $required"
  done
  log "self-test PASS"
)

run_route() {
  local backend="$1" small="$2" medium="$3" large="$4" output="$5"
  local feature_args=()
  if [[ "$backend" == "metal" ]]; then
    feature_args=(--features metal)
  fi
  env \
    VOKRA_MUSICGEN_SMALL_GGUF="$small" \
    VOKRA_MUSICGEN_MEDIUM_GGUF="$medium" \
    VOKRA_MUSICGEN_LARGE_GGUF="$large" \
    VOKRA_MUSICGEN_ROUTE_BACKEND="$backend" \
    CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-2}" \
    RUST_TEST_THREADS=1 \
    cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release \
      -p vokra-models "${feature_args[@]}" --test musicgen_public_contract \
      public_musicgen_lm_only_companion_generates_finite_pcm \
      -- --exact --ignored --nocapture --test-threads=1 \
      2>&1 | tee "$output"

  for target in musicgen-medium musicgen-large; do
    grep -F "MUSICGEN_COMPANION_ROUTE backend=$backend target=$target" "$output" \
      | grep -F "verdict=PASS" >/dev/null \
      || die "$target $backend route PASS marker is absent"
  done
}

main() {
  local small='' medium='' large='' evidence_dir='' self_test=0
  while (( $# > 0 )); do
    case "$1" in
      --small)
        [[ $# -ge 2 ]] || { usage; return 2; }
        small="$2"; shift 2 ;;
      --medium)
        [[ $# -ge 2 ]] || { usage; return 2; }
        medium="$2"; shift 2 ;;
      --large)
        [[ $# -ge 2 ]] || { usage; return 2; }
        large="$2"; shift 2 ;;
      --evidence-dir)
        [[ $# -ge 2 ]] || { usage; return 2; }
        evidence_dir="$2"; shift 2 ;;
      --self-test) self_test=1; shift ;;
      -h|--help) usage; return 0 ;;
      *) usage; die "unknown argument $1" ;;
    esac
  done

  if (( self_test == 1 )); then
    [[ -z "$small$medium$large$evidence_dir" ]] \
      || die "--self-test accepts no other arguments"
    run_self_test
    return
  fi
  [[ -n "$small" && -n "$medium" && -n "$large" && -n "$evidence_dir" ]] \
    || { usage; die "--small, --medium, --large and --evidence-dir are required"; }

  require_remote_apple_host
  require_tooling
  verify_file "MusicGen Small" "$small" "$SMALL_BYTES" "$SMALL_SHA256"
  verify_file "MusicGen Medium" "$medium" "$MEDIUM_BYTES" "$MEDIUM_SHA256"
  verify_file "MusicGen Large" "$large" "$LARGE_BYTES" "$LARGE_SHA256"
  require_empty_directory "$evidence_dir"
  record_environment "$evidence_dir/environment.txt"
  {
    echo "small_bytes=$SMALL_BYTES"
    echo "small_sha256=$SMALL_SHA256"
    echo "medium_bytes=$MEDIUM_BYTES"
    echo "medium_sha256=$MEDIUM_SHA256"
    echo "large_bytes=$LARGE_BYTES"
    echo "large_sha256=$LARGE_SHA256"
  } > "$evidence_dir/input-identities.txt"

  step "Run exact Medium/Large companion routes on Apple CPU"
  run_route cpu "$small" "$medium" "$large" "$evidence_dir/cpu-route.log"

  step "Run exact Medium/Large companion routes on real Metal"
  run_route metal "$small" "$medium" "$large" "$evidence_dir/metal-route.log"

  step "Exercise the Medium CLI on real Metal"
  CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-2}" \
    cargo build --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release \
      -p vokra-cli --features metal 2>&1 | tee "$evidence_dir/cli-build.log"
  VOKRA_ALLOW_RESEARCH_LICENSE=1 \
    "$VOKRA_ROOT/target/release/vokra-cli" run \
      --model "$medium" --backend metal --musicgen-companion "$small" \
      --token-ids 1 --music-unconditional-token-ids 0 \
      --music-frames 3 --music-seed 0 --output "$evidence_dir/medium-metal.wav" \
      2>&1 | tee "$evidence_dir/cli-metal.log"
  [[ -s "$evidence_dir/medium-metal.wav" ]] \
    || die "Medium Metal CLI output WAV is absent or empty"
  grep -F "musicgen: wrote 1920 samples @ 32000 Hz" "$evidence_dir/cli-metal.log" >/dev/null \
    || die "Medium Metal CLI output contract is absent"

  {
    echo "execution_status=PASS"
    echo "git_commit=$(git -C "$VOKRA_ROOT" rev-parse HEAD)"
    echo "small_sha256=$SMALL_SHA256"
    echo "medium_sha256=$MEDIUM_SHA256"
    echo "large_sha256=$LARGE_SHA256"
    echo "medium_cpu_route=FINITE_NONZERO_PASS"
    echo "large_cpu_route=FINITE_NONZERO_PASS"
    echo "medium_metal_route=FINITE_NONZERO_PASS"
    echo "large_metal_route=FINITE_NONZERO_PASS"
    echo "medium_metal_cli_route=PASS"
    echo "medium_metal_wav_sha256=$(sha256_file "$evidence_dir/medium-metal.wav")"
    echo "numerical_parity=NOT_CLAIMED"
    echo "upload=NOT_PERFORMED"
  } > "$evidence_dir/summary.txt"
  log "PASS: pull $evidence_dir, remove staged inputs, then destroy the remote worker"
}

main "$@"
