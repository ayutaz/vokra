#!/usr/bin/env bash
# VAST-only inspection worker for Meta omniASR-CTC-1B.
#
# This worker downloads only the pinned upstream checkpoint and tokenizer,
# prepares the checkpoint through the safe, audited bridge, and converts it to
# a local GGUF for header/tensor inspection.  It deliberately stops before
# numerical parity or runtime transcription: the native omniASR binder still
# has synthesized weights and `transcribe` remains incomplete.  It never
# uploads, publishes, pushes, stops, or destroys an instance.

set -euo pipefail

UPSTREAM_REPO="facebook/omniASR-CTC-1B"
UPSTREAM_REVISION="8c22e3ffdaa4aab6431b128b84b991a7d9c2515c"
CHECKPOINT_FILENAME="omniASR-CTC-1B.pt"
TOKENIZER_FILENAME="omniASR_tokenizer.model"
MODEL_KIND="omniasr-ctc"
EXPECTED_TENSOR_COUNT=807
PREPARER="tools/parity/omniasr_ctc_prepare_checkpoint.py"
UV_CONTRACT="uv run --frozen --project tools/parity --python 3.12 python"
MIN_VAST_MEM_KIB=$((64 * 1024 * 1024))
MIN_FREE_DISK_KIB=$((150 * 1024 * 1024))

usage() {
  cat <<'EOF'
Usage:
  run-omniasr-ctc-inspection.sh [--work-dir <dir>]
  run-omniasr-ctc-inspection.sh --self-test

The inspection path is VAST-only.  It requires Linux, a provisioned host with
VOKRA_PUBLISH_ON_VAST=1, a clean repository checkout, uv's frozen
tools/parity environment, and the Rust build/check tools.  It downloads only
the pinned omniASR-CTC-1B checkpoint and SentencePiece tokenizer, then records
source/prepared/GGUF hashes and the 807-tensor preparation manifest.

This is not a parity or full-runtime validation worker.  No upload or
publication is performed.
EOF
}

die() {
  echo "run-omniasr-ctc-inspection: $*" >&2
  exit 1
}

run_self_test() {
  local script_path="${BASH_SOURCE[0]}" worker_path fail=0 cases=0 required
  worker_path="$(cd "$(dirname "$script_path")/../../.." && pwd)/$PREPARER"
  [[ -f "$worker_path" ]] || die "safe preparer is missing: $worker_path"

  cases=$((cases + 1))
  for required in \
    "$UPSTREAM_REPO" "$UPSTREAM_REVISION" "$CHECKPOINT_FILENAME" \
    "$TOKENIZER_FILENAME" "$MODEL_KIND" "$EXPECTED_TENSOR_COUNT" \
    "$PREPARER" "$UV_CONTRACT" \
    'allow_patterns=["omniASR-CTC-1B.pt", "omniASR_tokenizer.model"]' \
    "snapshot_download" "source_sha256" "prepared_sha256" "gguf_sha256" \
    "tensor_manifest" "runtime_status=INCOMPLETE" "parity_status=NOT_RUN" \
    "verdict=INSPECTION_ONLY" "MIN_VAST_MEM_KIB" "MIN_FREE_DISK_KIB" \
    "/proc/meminfo" "df -Pk" "mindepth 1" \
    "grep -Eq \"^converted \${MODEL_KIND}: \${EXPECTED_TENSOR_COUNT} tensors,\""; do
    if ! grep -Fq -- "$required" "$script_path"; then
      echo "run-omniasr-ctc-inspection: self-test FAIL: missing contract: $required" >&2
      fail=1
    fi
  done

  cases=$((cases + 1))
  for required in \
    'uname -s' 'VOKRA_PUBLISH_ON_VAST' \
    'git status --porcelain --untracked-files=all' \
    'cargo fmt --all -- --check' 'check-forbidden-symbols.sh' \
    'check-zero-deps.sh' 'check-bound-arch-coverage.sh' \
    'cargo build --locked --release -p vokra-cli'; do
    if ! grep -Fq -- "$required" "$script_path"; then
      echo "run-omniasr-ctc-inspection: self-test FAIL: missing VAST gate: $required" >&2
      fail=1
    fi
  done

  cases=$((cases + 1))
  if grep -En 'weights_only=False|torch\.load\([^)]*False' "$worker_path" >/dev/null; then
    echo "run-omniasr-ctc-inspection: self-test FAIL: unsafe pickle loader in preparer" >&2
    fail=1
  fi
  if ! grep -Fq -- 'torch.load(str(args.input), map_location="cpu", weights_only=True)' "$worker_path"; then
    echo "run-omniasr-ctc-inspection: self-test FAIL: safe loader contract missing" >&2
    fail=1
  fi
  if grep -En '^[[:space:]]*(git[[:space:]]+push|.*publish-one\.sh|.*upload\.sh)([[:space:]]|$)' \
    "$script_path" >/dev/null; then
    echo "run-omniasr-ctc-inspection: self-test FAIL: publication command found" >&2
    fail=1
  fi
  if grep -En '^[[:space:]]*run_logged[[:space:]]+.*cargo[[:space:]]+test' "$script_path" >/dev/null; then
    echo "run-omniasr-ctc-inspection: self-test FAIL: parity/test execution found" >&2
    fail=1
  fi

  cases=$((cases + 1))
  if "$script_path" --self-test --work-dir /tmp/omniasr-self-test >/dev/null 2>&1; then
    echo "run-omniasr-ctc-inspection: self-test FAIL: extra argument accepted" >&2
    fail=1
  fi
  if "$script_path" --unknown-self-test-flag >/dev/null 2>&1; then
    echo "run-omniasr-ctc-inspection: self-test FAIL: unknown argument accepted" >&2
    fail=1
  fi

  if (( fail == 0 )); then
    echo "run-omniasr-ctc-inspection.sh self-test: OK ($cases cases)"
    return 0
  fi
  return 1
}

work_dir="/workspace/vokra-omniasr-ctc-inspection"
self_test=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --self-test)
      self_test=1
      shift
      ;;
    --work-dir)
      [[ $# -ge 2 ]] || die "--work-dir requires a path"
      work_dir="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

if [[ $self_test -eq 1 ]]; then
  [[ "$work_dir" == "/workspace/vokra-omniasr-ctc-inspection" ]] \
    || die "--self-test accepts no other arguments"
  run_self_test
  exit $?
fi

[[ "$(uname -s)" == "Linux" ]] || die "actual inspection is Linux/VAST-only"
[[ "${VOKRA_PUBLISH_ON_VAST:-0}" == "1" ]] \
  || die "VOKRA_PUBLISH_ON_VAST=1 is absent; run provision.sh first"
[[ -f Cargo.toml && -d crates/vokra-models ]] \
  || die "run from the Vokra repository root"
[[ -z "$(git status --porcelain --untracked-files=all)" ]] \
  || die "worktree changes or untracked files are present; use a clean committed git-bundle checkpoint"

work_parent="$(dirname "$work_dir")"
[[ -d "$work_parent" ]] || die "work-dir parent does not exist: $work_parent"
if [[ -e "$work_dir" ]]; then
  [[ -d "$work_dir" ]] || die "work-dir exists but is not a directory: $work_dir"
  [[ -z "$(find "$work_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]] \
    || die "work-dir must be absent or empty; refusing stale inspection output: $work_dir"
fi

[[ -r /proc/meminfo ]] || die "/proc/meminfo is unavailable"
mem_kib="$(awk '$1 == "MemTotal:" {print $2; exit}' /proc/meminfo)"
[[ "$mem_kib" =~ ^[0-9]+$ ]] || die "could not read MemTotal"
(( mem_kib >= MIN_VAST_MEM_KIB )) \
  || die "MemTotal=${mem_kib} KiB is below the 64-GiB guard"
free_kib="$(df -Pk "$work_parent" | awk 'NR == 2 {print $4}')"
[[ "$free_kib" =~ ^[0-9]+$ ]] || die "could not read free disk"
(( free_kib >= MIN_FREE_DISK_KIB )) \
  || die "free disk=${free_kib} KiB is below the 150-GB run guard"

for command in cargo rustfmt uv sha256sum; do
  command -v "$command" >/dev/null 2>&1 \
    || die "required VAST tool is missing: $command"
done
cargo clippy --version >/dev/null 2>&1 \
  || die "the clippy component is missing from the VAST Rust toolchain"

mkdir -p "$work_dir"
work_dir="$(cd "$work_dir" && pwd)"
log_path="$work_dir/inspection.log"
cache_dir="$work_dir/hf-cache"
assets_dir="$work_dir/assets"
prepared_dir="$work_dir/prepared"
evidence_dir="$work_dir/evidence"
snapshot_path_file="$evidence_dir/hf-snapshot-path.txt"
manifest_path="$prepared_dir/${CHECKPOINT_FILENAME%.pt}.safetensors.manifest.json"
prepared_path="$prepared_dir/${CHECKPOINT_FILENAME%.pt}.safetensors"
gguf_path="$work_dir/omniasr-ctc-1b.gguf"
mkdir -p "$cache_dir" "$assets_dir" "$prepared_dir" "$evidence_dir"

run_logged() {
  echo "+ $*" | tee -a "$log_path"
  "$@" 2>&1 | tee -a "$log_path"
}

export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-1}"
export RUST_BACKTRACE=1
UV_CMD=(uv run --frozen --project tools/parity --python 3.12 python)

run_logged cargo fmt --all -- --check
run_logged bash scripts/check-forbidden-symbols.sh
run_logged bash scripts/check-zero-deps.sh
run_logged bash scripts/check-bound-arch-coverage.sh

# Resolve the exact revision with the frozen repository-side Hugging Face
# tooling, but copy only the two canonical files into the working staging
# directory.  The cache may contain hub bookkeeping; the staged assets do not.
run_logged uv run --frozen --project tools/parity --python 3.12 python - \
  "$UPSTREAM_REPO" "$UPSTREAM_REVISION" "$cache_dir" "$snapshot_path_file" <<'PY'
import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

repo, revision, cache_dir, output = sys.argv[1:]
path = snapshot_download(
    repo_id=repo,
    revision=revision,
    cache_dir=cache_dir,
    allow_patterns=["omniASR-CTC-1B.pt", "omniASR_tokenizer.model"],
    token=os.environ.get("HF_TOKEN") or os.environ.get("HF"),
)
resolved = Path(path)
if resolved.name != revision:
    raise SystemExit(f"snapshot revision drift: {resolved.name!r} != {revision!r}")
for filename in ("omniASR-CTC-1B.pt", "omniASR_tokenizer.model"):
    if not (resolved / filename).is_file():
        raise SystemExit(f"canonical file missing from pinned snapshot: {filename}")
Path(output).write_text(str(resolved) + "\n", encoding="utf-8")
PY

snapshot_path="$(< "$snapshot_path_file")"
cp "$snapshot_path/$CHECKPOINT_FILENAME" "$assets_dir/$CHECKPOINT_FILENAME"
cp "$snapshot_path/$TOKENIZER_FILENAME" "$assets_dir/$TOKENIZER_FILENAME"
asset_count="$(find "$assets_dir" -type f -print | wc -l | tr -d ' ')"
[[ "$asset_count" == "2" ]] || die "staged asset count=$asset_count, expected only checkpoint + tokenizer"
source_sha256="$(sha256sum "$assets_dir/$CHECKPOINT_FILENAME" | awk '{print $1}')"
tokenizer_sha256="$(sha256sum "$assets_dir/$TOKENIZER_FILENAME" | awk '{print $1}')"

run_logged "${UV_CMD[@]}" "$PREPARER" \
  --input "$assets_dir/$CHECKPOINT_FILENAME" \
  --output "$prepared_path"
[[ -f "$manifest_path" ]] || die "safe preparer manifest missing: $manifest_path"

run_logged "${UV_CMD[@]}" - "$manifest_path" "$EXPECTED_TENSOR_COUNT" "$source_sha256" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = int(sys.argv[2])
source_sha256 = sys.argv[3]
if manifest.get("model_id") != "facebook/omniASR-CTC-1B":
    raise SystemExit(f"manifest model mismatch: {manifest.get('model_id')!r}")
if manifest.get("hf_revision") != "8c22e3ffdaa4aab6431b128b84b991a7d9c2515c":
    raise SystemExit("manifest HF revision mismatch")
if manifest.get("source_sha256") != source_sha256:
    raise SystemExit("manifest source SHA-256 mismatch")
if manifest.get("tensor_count") != expected or len(manifest.get("tensors", [])) != expected:
    raise SystemExit(f"manifest tensor count mismatch: {manifest.get('tensor_count')}")
if manifest.get("integer_tensor_count") != 0 or manifest.get("unknown_dtype_count") != 0:
    raise SystemExit("manifest contains rejected dtype classes")
print(f"manifest inspection: {expected} floating tensors; integer/unknown=0")
PY

run_logged cargo build --locked --release -p vokra-cli
run_logged target/release/vokra-cli convert \
  --model "$MODEL_KIND" \
  --input "$prepared_path" \
  --output "$gguf_path"
grep -Eq "^converted ${MODEL_KIND}: ${EXPECTED_TENSOR_COUNT} tensors," "$log_path" \
  || die "converter did not report the expected ${EXPECTED_TENSOR_COUNT}-tensor conversion"

cp "$manifest_path" "$evidence_dir/omniasr-ctc-1b.tensor-manifest.json"
{
  echo "upstream_repo=$UPSTREAM_REPO"
  echo "hf_revision=$UPSTREAM_REVISION"
  echo "checkpoint_filename=$CHECKPOINT_FILENAME"
  echo "tokenizer_filename=$TOKENIZER_FILENAME"
  echo "source_sha256=$source_sha256"
  echo "tokenizer_sha256=$tokenizer_sha256"
  echo "prepared_sha256=$(sha256sum "$prepared_path" | awk '{print $1}')"
  echo "gguf_sha256=$(sha256sum "$gguf_path" | awk '{print $1}')"
  echo "tensor_manifest=$evidence_dir/omniasr-ctc-1b.tensor-manifest.json"
  echo "tensor_count=$EXPECTED_TENSOR_COUNT"
  echo "runtime_status=INCOMPLETE"
  echo "parity_status=NOT_RUN"
  echo "publication_status=NOT_REQUESTED"
  echo "verdict=INSPECTION_ONLY"
} > "$evidence_dir/inspection-summary.txt"

echo "run-omniasr-ctc-inspection: INSPECTION_ONLY"
echo "Pull before external VAST teardown: $evidence_dir and $log_path"
echo "Do not pull the multi-GB checkpoint, safetensors, or GGUF to the maintainer Mac."
