#!/usr/bin/env bash
# VAST-only, NO_UPLOAD real-weight parity worker for omniASR-CTC-1B.
#
# The older run-omniasr-ctc-inspection.sh remains inspection-only.  This worker
# is deliberately separate: it performs the authenticated conversion and the
# independent official reference dump, then records a CPU parity result.  It
# never uploads, publishes, pushes, or destroys a VAST instance.
set -euo pipefail

MODEL_ID="facebook/omniASR-CTC-1B"
HF_REVISION="8c22e3ffdaa4aab6431b128b84b991a7d9c2515c"
CHECKPOINT_FILENAME="omniASR-CTC-1B.pt"
TOKENIZER_FILENAME="omniASR_tokenizer.model"
CHECKPOINT_SHA256="e8564fa59dab7caedbcdb54ab7fb9bd6c96989f4d19add2ad81ddd969716952c"
PREPARED_SHA256="cda8d7dd7cad2a0361b6946c42342b85ef7b0a8d672b99631dc75b4c3123dbc5"
OMNI_REPOSITORY="https://github.com/facebookresearch/omnilingual-asr"
OMNI_REVISION="a7fb36017a46eee8953f76bd628c174d51aefeef"
FAIRSEQ2_REPOSITORY="https://github.com/facebookresearch/fairseq2"
FAIRSEQ2_REVISION="8ae890e1b4d3e36307d0ba5fb695f0fc4815ecca"
PREPARER="tools/parity/omniasr_ctc_prepare_checkpoint.py"
DUMPER="tools/parity/omniasr_ctc_dump_reference.py"
PARITY_TEST="real_omniasr_ctc_encoder_logits_and_tokens_match_official"
AUDIO_FIXTURE_PATH="tests/fixtures/audio/jfk-30s.wav"
AUDIO_FIXTURE_SHA256="58adb4ea501d955fcd40bfbb69128f8f40428b81d8716b9ed337949773be253f"
AUDIO_FIXTURE_BYTES=352078
MIN_MEM_KIB=$((64 * 1024 * 1024))
MIN_DISK_KIB=$((150 * 1024 * 1024))
DEFAULT_WORK_DIR="/workspace/vokra-omniasr-ctc-validation"
SNAPSHOT_LOCAL_DIR_NAME="hf-materialized"

die() { echo "run-omniasr-ctc-validation: $*" >&2; exit 2; }

usage() {
  cat <<'EOF'
Usage:
  run-omniasr-ctc-validation.sh [--work-dir /absolute/absent/path]
  run-omniasr-ctc-validation.sh --self-test

The real route is Linux x86_64/VAST-only, requires a clean committed checkout,
64 GiB RAM, 150 GiB free disk, and VOKRA_PUBLISH_ON_VAST=1.  It stages the
pinned HF checkpoint/tokenizer, verifies the known checkpoint and prepared
SHA-256 values, clones both official source repositories at fixed commits,
converts the exact 807-F32 manifest, dumps an official fairseq2 oracle, and
leaves an authenticated CPU-parity packet.  Publication is permanently
NO_UPLOAD.
EOF
}

canonical_absent_path() {
  local path="$1" label="$2" parent candidate cursor
  [[ "$path" = /* ]] || die "$label must be absolute: $path"
  cursor="$path"
  while [[ "$cursor" != "/" ]]; do
    [[ ! -L "$cursor" ]] || die "$label has a symlink ancestor: $cursor"
    cursor="$(dirname "$cursor")"
  done
  parent="$(dirname "$path")"
  [[ -d "$parent" ]] || die "$label parent is missing: $parent"
  candidate="$(cd "$parent" && pwd -P)/$(basename "$path")"
  [[ "$candidate" == "$path" ]] || die "$label is a lexical/canonical alias: $path -> $candidate"
  [[ ! -e "$path" ]] || die "$label must be absent before the gate: $path"
}

canonical_existing_path() {
  local path="$1" label="$2" cursor canonical
  [[ "$path" = /* ]] || die "$label must be absolute: $path"
  [[ -e "$path" ]] || die "$label is missing: $path"
  cursor="$path"
  while [[ "$cursor" != "/" ]]; do
    [[ ! -L "$cursor" ]] || die "$label has a symlink ancestor: $cursor"
    cursor="$(dirname "$cursor")"
  done
  canonical="$(cd "$path" && pwd -P)"
  [[ "$canonical" == "$path" ]] || die "$label is not canonical: $path -> $canonical"
  printf '%s\n' "$canonical"
}

canonical_existing_file() {
  local path="$1" label="$2" cursor canonical
  [[ "$path" = /* ]] || die "$label must be absolute: $path"
  [[ -f "$path" && ! -L "$path" ]] || die "$label must be a regular non-symlink file: $path"
  cursor="$path"
  while [[ "$cursor" != "/" ]]; do
    [[ ! -L "$cursor" ]] || die "$label has a symlink ancestor: $cursor"
    cursor="$(dirname "$cursor")"
  done
  canonical="$(cd "$(dirname "$path")" && pwd -P)/$(basename "$path")"
  [[ "$canonical" == "$path" ]] || die "$label is not canonical: $path -> $canonical"
  printf '%s\n' "$canonical"
}

run_self_test() {
  local fail=0 required
  for required in \
    "$MODEL_ID" "$HF_REVISION" "$CHECKPOINT_SHA256" "$PREPARED_SHA256" \
    "$OMNI_REPOSITORY" "$OMNI_REVISION" "$FAIRSEQ2_REPOSITORY" \
    "$FAIRSEQ2_REVISION" "$PREPARER" "$DUMPER" "$AUDIO_FIXTURE_PATH" \
    "--audio" "$AUDIO_FIXTURE_SHA256" "$AUDIO_FIXTURE_BYTES" "NO_UPLOAD" \
    "canonical_absent_path" "canonical_existing_path" "canonical_existing_file" "symlink ancestor" \
    "SNAPSHOT_LOCAL_DIR_NAME" "local_dir=destination" "destination.is_symlink()" \
    "materialized snapshot" \
    'manifest="$prepared.manifest.json"' \
    "dumper owns creation of the reference output directory" \
    "git status --porcelain --untracked-files=all" "MemTotal" "df -Pk" \
    "807" "$PARITY_TEST" "OMNIASR_REAL_PARITY_PASS" "parity_status=CPU_PASS" \
    "token_exact=true" "reference_manifest_sha256" "max_abs" \
    "reference_manifest=reference/manifest.json" "frontend_atol" \
    "REMOTE_BUNDLE_NO_LOCAL_PULL" \
    "models/transformer/encoder_layer.py" "models/transformer/ffn.py"; do
    grep -Fq -- "$required" "$0" || { echo "self-test missing: $required" >&2; fail=1; }
  done
  local doubled_manifest='manifest="$prepared.safetensors.'
  doubled_manifest+='manifest.json"'
  if grep -Fq -- "$doubled_manifest" "$0"; then
    echo "self-test found doubled safetensors manifest suffix" >&2
    fail=1
  fi
  local reference_mkdir='mkdir "$ref_'
  reference_mkdir+='dir"'
  if grep -Fq -- "$reference_mkdir" "$0"; then
    echo "self-test found reference output pre-creation" >&2
    fail=1
  fi
  if grep -En '(^|[;&|][[:space:]]*)(git[[:space:]]+push|.*publish-one\.sh|.*upload\.sh)([[:space:]]|$)' "$0" >/dev/null; then
    echo "self-test found publication command" >&2; fail=1
  fi
  if (( fail == 0 )); then
    echo "run-omniasr-ctc-validation.sh self-test: OK"
    return 0
  fi
  return 1
}

work_dir="$DEFAULT_WORK_DIR"
self_test=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --self-test) self_test=1; shift ;;
    --work-dir) [[ $# -ge 2 ]] || die "--work-dir requires a path"; work_dir="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done
if (( self_test == 1 )); then
  [[ "$work_dir" == "$DEFAULT_WORK_DIR" ]] || die "--self-test accepts no work-dir"
  [[ $# == 0 ]] || die "--self-test accepts no extra arguments"
  run_self_test
  exit $?
fi

[[ "$(uname -s)" == Linux ]] || die "Linux/VAST is required"
[[ "$(uname -m)" == x86_64 ]] || die "x86_64 VAST host is required"
[[ "${VOKRA_PUBLISH_ON_VAST:-0}" == 1 ]] || die "VOKRA_PUBLISH_ON_VAST=1 is required"
[[ -f Cargo.toml && -d crates/vokra-models ]] || die "run from repository root"
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || die "clean committed checkout is required"
repo_root="$(canonical_existing_path "$(pwd -P)" repository)"
canonical_absent_path "$work_dir" work-dir
work_parent="$(dirname "$work_dir")"
work_canonical="$(cd "$work_parent" && pwd -P)/$(basename "$work_dir")"
case "$work_canonical/" in "$repo_root/"* ) die "work-dir overlaps checkout" ;; esac
case "$repo_root/" in "$work_canonical/"* ) die "checkout overlaps work-dir" ;; esac

[[ -r /proc/meminfo ]] || die "/proc/meminfo unavailable"
mem_kib="$(awk '$1 == "MemTotal:" {print $2; exit}' /proc/meminfo)"
[[ "$mem_kib" =~ ^[0-9]+$ && "$mem_kib" -ge "$MIN_MEM_KIB" ]] || die "at least 64 GiB RAM required"
free_kib="$(df -Pk "$work_parent" | awk 'NR == 2 {print $4}')"
[[ "$free_kib" =~ ^[0-9]+$ && "$free_kib" -ge "$MIN_DISK_KIB" ]] || die "at least 150 GiB free disk required"
for command in git cargo uv sha256sum; do command -v "$command" >/dev/null 2>&1 || die "missing tool: $command"; done
[[ -f tools/parity/omniasr_ctc/uv.lock ]] || die "dedicated OmniASR uv.lock is required; generate/review it on VAST"

# Authenticate the committed fixture before any source clone, model download,
# conversion, or reference execution begins.
audio_fixture="$(canonical_existing_file "$repo_root/$AUDIO_FIXTURE_PATH" audio-fixture)"
audio_bytes="$(stat -c '%s' "$audio_fixture")"
[[ "$audio_bytes" == "$AUDIO_FIXTURE_BYTES" ]] || die "audio fixture byte count drift: $audio_bytes"
audio_sha256="$(sha256sum "$audio_fixture" | awk '{print $1}')"
[[ "$audio_sha256" == "$AUDIO_FIXTURE_SHA256" ]] || die "audio fixture SHA-256 drift: $audio_sha256"

# No scratch/cache is created before all host/path gates above pass.
mkdir "$work_dir"
work_dir="$(cd "$work_dir" && pwd -P)"
snapshot_local_dir="$work_dir/$SNAPSHOT_LOCAL_DIR_NAME"
canonical_absent_path "$snapshot_local_dir" snapshot-local-dir
assets="$work_dir/assets"
prepared="$work_dir/prepared/$CHECKPOINT_FILENAME.safetensors"
prepared_dir="$(dirname "$prepared")"
evidence="$work_dir/evidence"
sources="$work_dir/sources"
mkdir -p "$assets" "$prepared_dir" "$evidence" "$sources"
log="$evidence/validation.log"
run_logged() { echo "+ $*" | tee -a "$log"; "$@" 2>&1 | tee -a "$log"; }

run_logged git clone --no-checkout "$OMNI_REPOSITORY" "$sources/omnilingual-asr"
run_logged git -C "$sources/omnilingual-asr" checkout --detach "$OMNI_REVISION"
run_logged git clone --no-checkout "$FAIRSEQ2_REPOSITORY" "$sources/fairseq2"
run_logged git -C "$sources/fairseq2" checkout --detach "$FAIRSEQ2_REVISION"
[[ "$(git -C "$sources/omnilingual-asr" remote get-url origin)" == "$OMNI_REPOSITORY" ]] || die "Omnilingual origin drift"
[[ "$(git -C "$sources/fairseq2" remote get-url origin)" == "$FAIRSEQ2_REPOSITORY" ]] || die "fairseq2 origin drift"
[[ "$(git -C "$sources/omnilingual-asr" rev-parse HEAD)" == "$OMNI_REVISION" ]] || die "Omnilingual checkout revision drift"
[[ "$(git -C "$sources/fairseq2" rev-parse HEAD)" == "$FAIRSEQ2_REVISION" ]] || die "fairseq2 checkout revision drift"
[[ -z "$(git -C "$sources/omnilingual-asr" status --porcelain)" ]] || die "Omnilingual checkout is dirty"
[[ -z "$(git -C "$sources/fairseq2" status --porcelain)" ]] || die "fairseq2 checkout is dirty"

run_logged uv run --frozen --project tools/parity --python 3.12 python - \
  "$MODEL_ID" "$HF_REVISION" "$work_dir/hf-cache" "$snapshot_local_dir" \
  "$evidence/hf-snapshot-path.txt" <<'PY'
import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download
repo, revision, cache_dir, local_dir, output = sys.argv[1:]
destination = Path(local_dir)
if not destination.is_absolute():
    raise SystemExit(f"materialized snapshot destination must be absolute: {destination}")
if destination.exists() or destination.is_symlink():
    raise SystemExit(f"materialized snapshot destination must be absent and non-symlink: {destination}")
if not destination.parent.is_dir() or destination.parent.is_symlink():
    raise SystemExit(f"materialized snapshot parent must be a regular directory: {destination.parent}")
if destination.parent.resolve(strict=True) != destination.parent:
    raise SystemExit(f"materialized snapshot parent has a symlink ancestor: {destination.parent}")
downloaded = Path(snapshot_download(repo_id=repo, revision=revision, cache_dir=cache_dir,
                                    local_dir=destination,
                                    allow_patterns=["omniASR-CTC-1B.pt", "omniASR_tokenizer.model"],
                                    token=os.environ.get("HF_TOKEN") or os.environ.get("HF")))
canonical_destination = destination.resolve(strict=True)
if canonical_destination != destination:
    raise SystemExit(f"materialized snapshot destination is not canonical: {destination} -> {canonical_destination}")
if downloaded.resolve(strict=True) != canonical_destination:
    raise SystemExit(f"snapshot_download returned an unexpected materialized path: {downloaded}")
for name in ("omniASR-CTC-1B.pt", "omniASR_tokenizer.model"):
    asset = canonical_destination / name
    if not asset.is_file() or asset.is_symlink():
        raise SystemExit(f"missing canonical materialized snapshot asset: {name}")
Path(output).write_text(str(canonical_destination), encoding="utf-8")
PY
snapshot="$(< "$evidence/hf-snapshot-path.txt")"
cp "$snapshot/$CHECKPOINT_FILENAME" "$assets/$CHECKPOINT_FILENAME"
cp "$snapshot/$TOKENIZER_FILENAME" "$assets/$TOKENIZER_FILENAME"
source_sha="$(sha256sum "$assets/$CHECKPOINT_FILENAME" | awk '{print $1}')"
[[ "$source_sha" == "$CHECKPOINT_SHA256" ]] || die "checkpoint SHA-256 drift: $source_sha"
tokenizer_sha="$(sha256sum "$assets/$TOKENIZER_FILENAME" | awk '{print $1}')"

run_logged uv run --frozen --project tools/parity --python 3.12 python "$PREPARER" \
  --input "$assets/$CHECKPOINT_FILENAME" --output "$prepared"
prepared_sha="$(sha256sum "$prepared" | awk '{print $1}')"
[[ "$prepared_sha" == "$PREPARED_SHA256" ]] || die "prepared SHA-256 drift: $prepared_sha"
manifest="$prepared.manifest.json"
[[ -f "$manifest" ]] || die "prepared tensor manifest missing"

run_logged cargo build --locked --release -p vokra-cli
gguf="$work_dir/omniasr-ctc-1b.gguf"
run_logged target/release/vokra-cli convert --model omniasr-ctc --input "$prepared" --output "$gguf"
grep -Eq '^converted omniasr-ctc: 807 tensors,' "$log" || die "converter did not report exact 807 tensors"

# The dumper owns creation of the reference output directory (exist_ok=False);
# this path must remain absent until its invocation.
ref_dir="$evidence/reference"
run_logged env PYTHONPATH="$sources/omnilingual-asr/src:$sources/fairseq2/src${PYTHONPATH:+:$PYTHONPATH}" \
  uv run --frozen --project tools/parity/omniasr_ctc --python 3.12 \
  python "$DUMPER" --checkpoint "$assets/$CHECKPOINT_FILENAME" --tokenizer "$assets/$TOKENIZER_FILENAME" \
  --audio "$repo_root/$AUDIO_FIXTURE_PATH" \
  --omnilingual-src "$sources/omnilingual-asr" --fairseq2-src "$sources/fairseq2" --output-dir "$ref_dir"
run_logged env PYTHONPATH="$sources/omnilingual-asr/src:$sources/fairseq2/src${PYTHONPATH:+:$PYTHONPATH}" \
  uv run --frozen --project tools/parity/omniasr_ctc --python 3.12 \
  python "$DUMPER" --validate-manifest "$ref_dir/manifest.json"

cp "$gguf" "$evidence/omniasr-ctc-1b.gguf"
cp "$manifest" "$evidence/omniasr-ctc-1b.tensor-manifest.json"
gguf_sha="$(sha256sum "$gguf" | awk '{print $1}')"

# The ignored consumer test is intentionally invoked only on this VAST host.
# It reads the hash-bound GGUF/reference packet; absence or a partial packet is
# a hard error and never becomes a skip.
run_logged env VOKRA_OMNIASR_GGUF="$gguf" VOKRA_OMNIASR_REFERENCE_DIR="$ref_dir" \
  cargo test --locked -p vokra-models --test parity_omniasr_ctc_real -- \
  --ignored --exact "$PARITY_TEST" --nocapture
[[ "$(grep -Ec "^test ${PARITY_TEST} \.\.\. ok$" "$log")" == 1 ]] || \
  die "expected exactly one successful named OmniASR parity test result"
[[ "$(grep -Fxc 'OMNIASR_REAL_PARITY_PASS' "$log")" == 1 ]] || \
  die "expected exactly one OmniASR parity sentinel"

reference_manifest_sha="$(sha256sum "$ref_dir/manifest.json" | awk '{print $1}')"
cat > "$evidence/validation-summary.txt" <<EOF
schema=omniasr-ctc-validation-v1
status=CPU_PARITY_COMPLETE
publication=NO_UPLOAD
model_id=$MODEL_ID
hf_revision=$HF_REVISION
checkpoint_sha256=$source_sha
tokenizer_sha256=$tokenizer_sha
prepared_sha256=$prepared_sha
gguf_sha256=$gguf_sha
omnilingual_repository=$OMNI_REPOSITORY
omnilingual_revision=$OMNI_REVISION
fairseq2_repository=$FAIRSEQ2_REPOSITORY
fairseq2_revision=$FAIRSEQ2_REVISION
tensor_count=807
reference_manifest=reference/manifest.json
reference_manifest_sha256=$reference_manifest_sha
parity_test=$PARITY_TEST
parity_status=CPU_PASS
token_exact=true
max_abs=recorded-by-parity-test
metal_status=NOT_RUN_APPLE_WORKER
EOF
echo "run-omniasr-ctc-validation: CPU_PARITY_COMPLETE; NO_UPLOAD; REMOTE_BUNDLE_NO_LOCAL_PULL"
echo "Keep the GGUF/reference bundle remote; transfer VAST->authenticated Apple/Scaleway only. Return only small logs/manifests: $evidence"
