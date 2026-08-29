#!/usr/bin/env bash
# Convert and validate the two exact MOSS-Audio Instruct releases. VAST-only.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VOKRA_ROOT="${VOKRA_ROOT:-$DEFAULT_ROOT}"
VOKRA_SCRATCH="${VOKRA_SCRATCH:-$HOME/scratchpad}"
PARITY_PROJECT="$VOKRA_ROOT/tools/parity/moss_audio"
REFERENCE_DUMPER="$PARITY_PROJECT/dump_reference.py"
PREFLIGHT_GATE="$PARITY_PROJECT/preflight_gate.py"
PREFLIGHT_MANIFEST="$PARITY_PROJECT/license_gate_manifest.json"
REFERENCE_AUDIO="$VOKRA_ROOT/tests/parity/utmos/ref-clip.wav"
SOURCE_REPO="https://github.com/OpenMOSS/MOSS-Audio.git"
SOURCE_REVISION="5cbb1d823937cd5b5de3d8fa4d3a7253ebd3b883"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

# The 8B FP32 official oracle and the shard merger can each retain tens of GB.
# A 128-GB-class CPU host leaves headroom for allocator and Cargo peaks.
MIN_VAST_MEM_KIB=120000000
MIN_FREE_DISK_KIB=150000000
REFERENCE_AUDIO_SHA256="241c0d93cc7ed8792c85c525d1e02b8c33850b791902a5e75b79c2d500e71a1a"

log() { printf '[moss-audio-vast] %s\n' "$*" >&2; }
step() { printf '\n[moss-audio-vast] ==== %s ====\n' "$*" >&2; }
die() { log "ERROR: $*"; return 2; }

usage() {
  cat <<'EOF' >&2
usage: run-moss-audio-validation.sh --variant <4b|8b|all> --approval-evidence <file> [--work-dir <absent-dir>]
       run-moss-audio-validation.sh --self-test

VAST-only, non-publishing gate for the two pinned MOSS-Audio Instruct
releases. It downloads exact immutable source/model revisions, merges the
shards, converts a self-contained GGUF, generates an independent FP32 CPU
reference through the official OpenMOSS model and processor, and compares
Vokra CPU audio projections, prompt ids, greedy ids and decoded text. It then
runs workspace and Apple Metal cross-build verification once.

There is no publishing option or artifact-upload path. Pull only the small
evidence directory, never the snapshots, merged checkpoints or GGUFs. Destroy
the VAST instance after the evidence is recovered; do not merely stop it.
EOF
}

variant_repo() {
  case "$1" in
    4b) printf '%s\n' 'OpenMOSS-Team/MOSS-Audio-4B-Instruct' ;;
    8b) printf '%s\n' 'OpenMOSS-Team/MOSS-Audio-8B-Instruct' ;;
    *) die "unknown variant $1" ;;
  esac
}

variant_revision() {
  case "$1" in
    4b) printf '%s\n' '6907a499dc0e87cc77c8ae0fe23fd0eb5476a02d' ;;
    8b) printf '%s\n' '6521a39181b47a18f2d9f4b3acfb5bca7b76b57f' ;;
    *) die "unknown variant $1" ;;
  esac
}

variant_model_kind() {
  case "$1" in
    4b) printf '%s\n' 'moss-audio-4b-instruct' ;;
    8b) printf '%s\n' 'moss-audio-8b-instruct' ;;
    *) die "unknown variant $1" ;;
  esac
}

variant_config_sha256() {
  case "$1" in
    4b) printf '%s\n' 'e528a941446f4443f1b9fede12ea484e58a79d494c28d21ef1e73b5148abfbfa' ;;
    8b) printf '%s\n' '535154c2a5bcbd0e18e2f92bcf370ac74b530eec97ad4fd9317993ba0a316536' ;;
    *) die "unknown variant $1" ;;
  esac
}

variant_preparer() {
  case "$1" in
    4b) printf '%s\n' "$VOKRA_ROOT/tools/parity/moss_audio_4b_instruct_prepare_checkpoint.py" ;;
    8b) printf '%s\n' "$VOKRA_ROOT/tools/parity/moss_audio_8b_instruct_prepare_checkpoint.py" ;;
    *) die "unknown variant $1" ;;
  esac
}

variant_test() {
  case "$1" in
    4b) printf '%s\n' 'moss_audio_4b_cpu_matches_official_reference' ;;
    8b) printf '%s\n' 'moss_audio_8b_cpu_matches_official_reference' ;;
    *) die "unknown variant $1" ;;
  esac
}

variant_gguf_env() {
  case "$1" in
    4b) printf '%s\n' 'VOKRA_MOSS_AUDIO_4B_GGUF' ;;
    8b) printf '%s\n' 'VOKRA_MOSS_AUDIO_8B_GGUF' ;;
    *) die "unknown variant $1" ;;
  esac
}

variant_reference_env() {
  case "$1" in
    4b) printf '%s\n' 'VOKRA_MOSS_AUDIO_4B_REFERENCE_DIR' ;;
    8b) printf '%s\n' 'VOKRA_MOSS_AUDIO_8B_REFERENCE_DIR' ;;
    *) die "unknown variant $1" ;;
  esac
}

sha256_file() {
  local path="$1"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$path" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$path" | awk '{print $1}'
  else
    die "neither sha256sum nor shasum is available"
  fi
}

require_vast_host() {
  local mem_kib free_kib
  [[ "${VOKRA_PUBLISH_ON_VAST:-0}" == "1" ]] \
    || die "VOKRA_PUBLISH_ON_VAST=1 is absent; run provision.sh first"
  [[ "$(uname -s)" == "Linux" ]] \
    || die "MOSS-Audio checkpoint work is VAST/Linux-only; refusing $(uname -s)"
  [[ -r /proc/meminfo ]] || die "/proc/meminfo is unavailable"
  mem_kib="$(awk '$1 == "MemTotal:" {print $2; exit}' /proc/meminfo)"
  [[ "$mem_kib" =~ ^[0-9]+$ ]] || die "could not read MemTotal"
  if (( mem_kib < MIN_VAST_MEM_KIB )); then
    die "MemTotal=${mem_kib} KiB is below the 128-GB-class guard"
  fi
  mkdir -p "$VOKRA_SCRATCH"
  free_kib="$(df -Pk "$VOKRA_SCRATCH" | awk 'NR == 2 {print $4}')"
  [[ "$free_kib" =~ ^[0-9]+$ ]] || die "could not read free disk"
  if (( free_kib < MIN_FREE_DISK_KIB )); then
    die "free disk=${free_kib} KiB is below the 150-GB run guard"
  fi
}

require_tooling() {
  local tool preparer
  for tool in uv cargo rustc rustup git awk find tee grep wc df; do
    command -v "$tool" >/dev/null 2>&1 || die "required tool missing: $tool"
  done
  [[ -d "$VOKRA_ROOT/.git" ]] || die "$VOKRA_ROOT is not a git checkout"
  [[ -f "$PARITY_PROJECT/uv.lock" ]] || die "MOSS-Audio parity uv.lock is missing"
  [[ -f "$REFERENCE_DUMPER" ]] || die "official reference dumper is missing"
  [[ -f "$REFERENCE_AUDIO" ]] || die "reference audio is missing"
  for preparer in "$(variant_preparer 4b)" "$(variant_preparer 8b)"; do
    [[ -f "$preparer" ]] || die "checkpoint preparer is missing: $preparer"
  done
  local audio_hash
  audio_hash="$(sha256_file "$REFERENCE_AUDIO")"
  [[ "$audio_hash" == "$REFERENCE_AUDIO_SHA256" ]] \
    || die "reference audio SHA-256 drift: $audio_hash"
  if [[ -n "$(git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all)" ]]; then
    die "VAST checkout must be clean so evidence names one exact commit"
  fi
}

pre_sync_gate() {
  local approval="$1"
  command -v uv >/dev/null 2>&1 || die "uv is required before the MOSS-Audio gate"
  [[ -f "$PARITY_PROJECT/uv.lock" && -f "$PARITY_PROJECT/pyproject.toml" && \
    -f "$PREFLIGHT_GATE" && -f "$PREFLIGHT_MANIFEST" ]] \
    || die "MOSS-Audio pre-sync gate inputs are missing"
  step "Validate exact MOSS-Audio closure before host/tooling/scratch/network work"
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python "$PREFLIGHT_GATE" \
      --project "$PARITY_PROJECT" --manifest "$PREFLIGHT_MANIFEST" \
      --evidence "$approval"
}

require_disjoint_work_dir() {
  local work="$1" approval="$2" candidate root_real approval_parent approval_real
  candidate="$(canonical_absent_path "$work")" || return 2
  root_real="$(cd -P "$VOKRA_ROOT" 2>/dev/null && pwd)" || die "Vokra checkout is inaccessible"
  approval_parent="$(cd -P "$(dirname "$approval")" 2>/dev/null && pwd)" || die "approval parent is inaccessible"
  approval_real="$approval_parent/$(basename "$approval")"
  [[ "$candidate" != "$root_real" && "$candidate/" != "$root_real/"* && "$root_real/" != "$candidate/"* ]] || die "work-dir overlaps the checkout"
  [[ "$candidate" != "$approval_real" && "$candidate/" != "$approval_real/"* && "$approval_real/" != "$candidate/"* ]] || die "work-dir overlaps approval evidence"
}

canonical_absent_path() {
  local target="$1" current suffix component real lexical
  [[ "$target" = /* ]] || target="$PWD/$target"
  lexical="${target#/}"; current="/"
  while [[ -n "$lexical" ]]; do
    component="${lexical%%/*}"
    if [[ "$lexical" == "$component" ]]; then lexical=""; else lexical="${lexical#*/}"; fi
    [[ "$component" == "." || -z "$component" ]] && continue
    [[ "$component" != ".." ]] || { die "work-dir path contains .."; return 2; }
    current="${current%/}/$component"
    if [[ -L "$current" ]]; then
      real="$(cd -P "$current" 2>/dev/null && pwd)" || { die "work-dir path contains an inaccessible component"; return 2; }
      case "$current:$real" in
        /var:/private/var|/tmp:/private/tmp) current="$real" ;;
        *) die "work-dir path contains a symlinked component"; return 2 ;;
      esac
    fi
  done
  current="$target"; suffix=""
  while [[ ! -e "$current" && ! -L "$current" ]]; do
    component="$(basename "$current")"; suffix="/$component$suffix"; current="$(dirname "$current")"
  done
  [[ -d "$current" && ! -L "$current" ]] || { die "work-dir has an inaccessible or symlinked existing parent"; return 2; }
  real="$(cd -P "$current" 2>/dev/null && pwd)" || { die "work-dir parent is inaccessible"; return 2; }
  printf '%s%s\n' "$real" "$suffix"
}

require_absent_work_dir() {
  local work="$1" approval="$2"
  require_disjoint_work_dir "$work" "$approval" || return 2
  [[ ! -e "$work" && ! -L "$work" ]] || { die "--work-dir must be absent before validation: $work"; return 2; }
}

require_one_named_test_passed() {
  local log_path="$1" test_name="$2" test_count named_count total_test_count result_count total_result_count
  test_count="$(grep -Ec "^test ${test_name} \.\.\. ok$" "$log_path" || true)"
  named_count="$(grep -Ec "^test ${test_name} \.\.\." "$log_path" || true)"
  total_test_count="$(grep -Ec '^test [^ ]+ \.\.\.' "$log_path" || true)"
  result_count="$(grep -Ec '^test result: ok\. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out(; finished in [0-9]+\.[0-9]+s)?$' "$log_path" || true)"
  total_result_count="$(grep -Ec '^test result:' "$log_path" || true)"
  [[ "$test_count" == 1 ]] || { die "expected exactly one passing $test_name"; return 2; }
  [[ "$named_count" == 1 ]] || { die "expected exactly one total $test_name line"; return 2; }
  [[ "$total_test_count" == 1 ]] || { die "expected exactly one total Cargo test line"; return 2; }
  [[ "$result_count" == 1 ]] || { die "expected one standard Cargo result for $test_name"; return 2; }
  [[ "$total_result_count" == 1 ]] || { die "expected exactly one total Cargo result line"; return 2; }
}

require_exact_cpu_sentinel() {
  local log_path="$1" model_kind="$2" count family_count
  family_count="$(grep -Ec "^MOSS_AUDIO_PARITY ${model_kind} CPU_vs_official " "$log_path" || true)"
  count="$(grep -Ec "^MOSS_AUDIO_PARITY ${model_kind} CPU_vs_official token_ids=exact text=exact PASS$" "$log_path" || true)"
  [[ "$family_count" == 1 && "$count" == 1 ]] || { die "expected exactly one complete CPU sentinel family for $model_kind"; return 2; }
}

record_environment() {
  local output="$1" cpu_model cpu_flags
  cpu_model="$(awk -F ':' '$1 ~ /model name/ {sub(/^[[:space:]]+/, "", $2); print $2; exit}' /proc/cpuinfo)"
  cpu_flags="$(awk -F ':' '$1 ~ /flags/ {sub(/^[[:space:]]+/, "", $2); print $2; exit}' /proc/cpuinfo)"
  {
    echo "utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "git_commit=$(git -C "$VOKRA_ROOT" rev-parse HEAD)"
    echo "git_branch=$(git -C "$VOKRA_ROOT" branch --show-current)"
    echo "uname=$(uname -a)"
    echo "cpu_model=$cpu_model"
    echo "cpu_flags=$cpu_flags"
    echo "nproc=$(nproc)"
    awk '$1 == "MemTotal:" {print "mem_total_kib=" $2; exit}' /proc/meminfo
    if command -v nvidia-smi >/dev/null 2>&1; then
      nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    else
      echo "gpu=unavailable (reference is intentionally CPU FP32)"
    fi
    rustc --version --verbose
    cargo --version
    uv --version
  } > "$output"
}

checkout_official_source() {
  local output="$1"
  git clone --filter=blob:none --no-checkout "$SOURCE_REPO" "$output"
  git -C "$output" checkout --detach "$SOURCE_REVISION"
  [[ "$(git -C "$output" rev-parse HEAD)" == "$SOURCE_REVISION" ]] \
    || die "official source checkout revision mismatch"
}

download_snapshot() {
  local repo="$1" revision="$2" output="$3"
  mkdir -p "$output"
  (
    UV_NO_CACHE=1 uv run --no-cache --project "$PARITY_PROJECT" --frozen --python 3.12 python -c \
      'import os,sys
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id=sys.argv[1],
    revision=sys.argv[2],
    local_dir=sys.argv[3],
    allow_patterns=["LICENSE", "config.json", "*.safetensors", "model.safetensors.index.json", "vocab.json", "merges.txt", "tokenizer_config.json", "chat_template.jinja", "generation_config.json", "processor_config.json"],
    token=os.environ.get("HF_TOKEN") or os.environ.get("HF"),
)' "$repo" "$revision" "$output"
  )
}

run_variant() {
  local variant="$1" work_dir="$2" evidence_dir="$3" source_dir="$4"
  local repo revision model_kind preparer snapshot merged gguf reference_dir
  local test_name gguf_env reference_env reference_threads parity_log
  repo="$(variant_repo "$variant")"
  revision="$(variant_revision "$variant")"
  model_kind="$(variant_model_kind "$variant")"
  preparer="$(variant_preparer "$variant")"
  snapshot="$work_dir/source-$variant"
  merged="$snapshot/model.merged.safetensors"
  gguf="$work_dir/$model_kind.gguf"
  reference_dir="$evidence_dir/reference-$variant"
  test_name="$(variant_test "$variant")"
  gguf_env="$(variant_gguf_env "$variant")"
  reference_env="$(variant_reference_env "$variant")"
  reference_threads="${VOKRA_REFERENCE_TORCH_THREADS:-8}"
  parity_log="$evidence_dir/parity-$variant.log"

  step "Download $repo@$revision"
  download_snapshot "$repo" "$revision" "$snapshot"
  step "Verify the exact downloaded $variant snapshot before any import or conversion"
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python "$PREFLIGHT_GATE" \
      --verify-snapshot --snapshot "$snapshot" --variant "$variant"

  step "Merge the pinned $variant sharded checkpoint"
  UV_NO_CACHE=1 uv run --no-cache --project "$PARITY_PROJECT" --frozen --python 3.12 python \
    "$preparer" --input-dir "$snapshot" --output "$merged" --strict \
    2>&1 | tee "$evidence_dir/prepare-$variant.log"
  [[ -s "$merged" ]] || die "checkpoint merger emitted no file: $merged"

  step "Convert $model_kind without quantization or publication"
  local license_spdx
  license_spdx="$(UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python "$PREFLIGHT_GATE" --license-spdx --variant "$variant")" \
    || die "fixed model license SPDX is unresolved for $variant"
  "$VOKRA_ROOT/target/release/vokra-cli" convert \
    --model "$model_kind" \
    --input "$merged" \
    --output "$gguf" \
    --license "$license_spdx" \
    2>&1 | tee "$evidence_dir/convert-$variant.log"
  [[ -s "$gguf" ]] || die "converter emitted no GGUF: $gguf"

  step "Generate independent official FP32 CPU reference for $variant"
  VOKRA_REFERENCE_TORCH_THREADS="$reference_threads" \
    UV_NO_CACHE=1 uv run --no-cache --project "$PARITY_PROJECT" --frozen --python 3.12 python \
      "$REFERENCE_DUMPER" \
      --variant "$variant" \
      --model-dir "$snapshot" \
      --source-dir "$source_dir" \
      --audio "$REFERENCE_AUDIO" \
      --output "$reference_dir" \
      --max-new-tokens 4 \
      2>&1 | tee "$evidence_dir/reference-$variant.log"

  step "Compare Vokra CPU with official reference for $variant"
  env "$gguf_env=$gguf" "$reference_env=$reference_dir" RUST_TEST_THREADS=1 \
  cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release \
      -p vokra-models --test moss_audio_real "$test_name" -- --exact --nocapture \
      2>&1 | tee "$parity_log"
  require_one_named_test_passed "$parity_log" "$test_name"
  require_exact_cpu_sentinel "$parity_log" "$model_kind"

  {
    echo "variant=$variant"
    echo "upstream_repo=$repo"
    echo "upstream_revision=$revision"
    echo "source_repo=$SOURCE_REPO"
    echo "source_revision=$SOURCE_REVISION"
    echo "config_sha256=$(variant_config_sha256 "$variant")"
    echo "checkpoint_file_identity=UNRESOLVED_REVIEW"
    echo "gguf_sha256=$(sha256_file "$gguf")"
    echo "reference_manifest_sha256=$(sha256_file "$reference_dir/manifest.txt")"
    echo "numeric_bound=0.01"
    echo "greedy_ids=exact"
    echo "text=exact"
    echo "verdict=PASS"
  } > "$evidence_dir/summary-$variant.txt"
}

write_apple_args() {
  local output="$1" gguf_4b_sha="$2" reference_4b_sha="$3" gguf_8b_sha="$4" reference_8b_sha="$5"
  {
    printf '# Generated for the separate no-upload Apple validation step.\n'
    printf '%q \\\n' 'scripts/verify/apple-silicon-moss-audio.sh'
    printf "  --gguf-4b '%s' \\\n" '<APPLE_GGUF_4B_PATH>'
    printf '  --gguf-4b-sha256 %q \\\n' "$gguf_4b_sha"
    printf "  --reference-4b '%s' \\\n" '<APPLE_REFERENCE_4B_DIR>'
    printf '  --reference-4b-sha256 %q \\\n' "$reference_4b_sha"
    printf "  --gguf-8b '%s' \\\n" '<APPLE_GGUF_8B_PATH>'
    printf '  --gguf-8b-sha256 %q \\\n' "$gguf_8b_sha"
    printf "  --reference-8b '%s' \\\n" '<APPLE_REFERENCE_8B_DIR>'
    printf '  --reference-8b-sha256 %q \\\n' "$reference_8b_sha"
    printf "  --approval-evidence '<APPLE_APPROVAL_EVIDENCE>' \\\n"
    printf "  --evidence-dir '<APPLE_EVIDENCE_DIR>'\n"
  } > "$output"
}

run_self_test() {
  local failed=0 temporary script_path="${BASH_SOURCE[0]}"
  temporary="$(mktemp -d)"
  trap 'rm -rf "$temporary"' EXIT
  printf '{}\n' > "$temporary/path-approval.json"
  mkdir -p "$temporary/nested-parent"
  require_absent_work_dir "$temporary/nested-parent/model/work" "$temporary/path-approval.json" || failed=1
  mkdir -p "$temporary/intermediate"
  ln -s "$VOKRA_ROOT" "$temporary/intermediate/checkout-link"
  if require_absent_work_dir "$temporary/intermediate/checkout-link/work" "$temporary/path-approval.json" >/dev/null 2>&1; then failed=1; fi
  mkdir -p "$temporary/real/existing"
  ln -s "$temporary/real" "$temporary/ancestor-link"
  if require_absent_work_dir "$temporary/ancestor-link/existing/nested/new" "$temporary/path-approval.json" >/dev/null 2>&1; then failed=1; fi
  ln -s "$temporary/missing-target" "$temporary/dangling-work"
  if require_absent_work_dir "$temporary/dangling-work" "$temporary/path-approval.json" >/dev/null 2>&1; then failed=1; fi
  if require_absent_work_dir "$VOKRA_ROOT/tools" "$temporary/path-approval.json" >/dev/null 2>&1; then failed=1; fi
  if require_absent_work_dir "$temporary/path-approval.json/child" "$temporary/path-approval.json" >/dev/null 2>&1; then failed=1; fi
  mkdir "$temporary/existing-empty"
  if require_absent_work_dir "$temporary/existing-empty" "$temporary/path-approval.json" >/dev/null 2>&1; then failed=1; fi
  [[ "$(variant_repo 4b)" == "OpenMOSS-Team/MOSS-Audio-4B-Instruct" ]] || failed=1
  [[ "$(variant_revision 8b)" =~ ^[0-9a-f]{40}$ ]] || failed=1
  [[ "$(variant_model_kind 8b)" == "moss-audio-8b-instruct" ]] || failed=1
  [[ "$(variant_config_sha256 4b)" =~ ^[0-9a-f]{64}$ ]] || failed=1
  [[ "$(variant_test 4b)" == "moss_audio_4b_cpu_matches_official_reference" ]] || failed=1
  [[ "$(variant_gguf_env 8b)" == "VOKRA_MOSS_AUDIO_8B_GGUF" ]] || failed=1
  if variant_repo bad >/dev/null 2>&1; then
    failed=1
  fi
  for required in pre_sync_gate PREFLIGHT_MANIFEST "--gguf-4b-sha256" \
    "--reference-4b-sha256" "--gguf-8b-sha256" "--reference-8b-sha256" \
    "<APPLE_GGUF_4B_PATH>" "<APPLE_REFERENCE_4B_DIR>" "<APPLE_GGUF_8B_PATH>" \
    "<APPLE_REFERENCE_8B_DIR>" "<APPLE_APPROVAL_EVIDENCE>" "<APPLE_EVIDENCE_DIR>" "--approval-evidence" "--verify-source" "--verify-snapshot" \
    "--no-cache"; do
    grep -F -- "$required" "$script_path" >/dev/null || failed=1
  done
  local log="$temporary/cargo.log"
  printf '%s\n%s\n' \
    'test moss_audio_4b_cpu_matches_official_reference ... ok' \
    'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out' > "$log"
  require_one_named_test_passed "$log" moss_audio_4b_cpu_matches_official_reference
  printf '%s\n%s\n' \
    'test moss_audio_4b_cpu_matches_official_reference ... FAILED' \
    'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out' > "$log"
  if require_one_named_test_passed "$log" moss_audio_4b_cpu_matches_official_reference >/dev/null 2>&1; then failed=1; fi
  printf '%s\n%s\n%s\n' \
    'test moss_audio_4b_cpu_matches_official_reference ... ok' \
    'test moss_audio_8b_cpu_matches_official_reference ... ok' \
    'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out' > "$log"
  if require_one_named_test_passed "$log" moss_audio_4b_cpu_matches_official_reference >/dev/null 2>&1; then failed=1; fi
  printf '%s\n%s\n%s\n' \
    'test moss_audio_4b_cpu_matches_official_reference ... ok' \
    'test moss_audio_4b_cpu_matches_official_reference ... ok' \
    'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out' > "$log"
  if require_one_named_test_passed "$log" moss_audio_4b_cpu_matches_official_reference >/dev/null 2>&1; then failed=1; fi
  printf '%s\n%s\n%s\n' \
    'test moss_audio_4b_cpu_matches_official_reference ... ok' \
    'test result: ok. 1 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out' \
    'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out' > "$log"
  if require_one_named_test_passed "$log" moss_audio_4b_cpu_matches_official_reference >/dev/null 2>&1; then failed=1; fi
  printf '%s\n%s\n' \
    'test moss_audio_4b_cpu_matches_official_reference ... ok' \
    'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.2x' > "$log"
  if require_one_named_test_passed "$log" moss_audio_4b_cpu_matches_official_reference >/dev/null 2>&1; then failed=1; fi
  local sentinel="MOSS_AUDIO_PARITY moss-audio-4b-instruct CPU_vs_official token_ids=exact text=exact PASS"
  printf '%s\n' "$sentinel" > "$log"
  require_exact_cpu_sentinel "$log" moss-audio-4b-instruct
  printf '%s\n%s\n' "$sentinel" "$sentinel" > "$log"
  if require_exact_cpu_sentinel "$log" moss-audio-4b-instruct >/dev/null 2>&1; then failed=1; fi
  printf 'prefix%s\n' "$sentinel" > "$log"
  if require_exact_cpu_sentinel "$log" moss-audio-4b-instruct >/dev/null 2>&1; then failed=1; fi
  printf '%s suffix\n' "$sentinel" > "$log"
  if require_exact_cpu_sentinel "$log" moss-audio-4b-instruct >/dev/null 2>&1; then failed=1; fi
  printf '%s\n%s\n' "$sentinel" "${sentinel/PASS/FAIL}" > "$log"
  if require_exact_cpu_sentinel "$log" moss-audio-4b-instruct >/dev/null 2>&1; then failed=1; fi

  local fake_root="$temporary/fake-checkout" fake_home="$temporary/fake-home"
  local fake_bin="$fake_home/.local/bin" trace="$temporary/trace.log"
  local fake_scratch="$temporary/scratch" fake_work="$fake_root/work" rc real_uv
  real_uv="$(command -v uv)"
  mkdir -p "$fake_root/tools/parity/moss_audio" "$fake_bin"
  cp "$PARITY_PROJECT/uv.lock" "$fake_root/tools/parity/moss_audio/uv.lock"
  cp "$PARITY_PROJECT/pyproject.toml" "$fake_root/tools/parity/moss_audio/pyproject.toml"
  cp "$PREFLIGHT_GATE" "$fake_root/tools/parity/moss_audio/preflight_gate.py"
  cp "$PREFLIGHT_MANIFEST" "$fake_root/tools/parity/moss_audio/license_gate_manifest.json"
  printf 'invalid approval\n' > "$fake_root/approval.json"
  cp "$script_path" "$fake_root/run-worker.sh"
  cat > "$fake_bin/uv" <<'EOF'
#!/usr/bin/env bash
printf 'uv %s\n' "$*" >> "${MOSS_AUDIO_TRACE:?}"
exec "${MOSS_AUDIO_REAL_UV:?}" "$@"
EOF
  chmod +x "$fake_bin/uv"
  git -C "$fake_root" init -q
  git -C "$fake_root" config user.email self-test@example.invalid
  git -C "$fake_root" config user.name self-test
  git -C "$fake_root" add .
  git -C "$fake_root" commit -qm baseline
  printf 'dirty checkout must not outrank the gate\n' > "$fake_root/dirty.txt"
  set +e
  HOME="$fake_home" PATH="$fake_bin:$PATH" MOSS_AUDIO_TRACE="$trace" \
    MOSS_AUDIO_REAL_UV="$real_uv" VOKRA_ROOT="$fake_root" \
    VOKRA_SCRATCH="$fake_scratch" VOKRA_PUBLISH_ON_VAST=1 \
    bash "$fake_root/run-worker.sh" --variant 4b --work-dir "$fake_work" \
      --approval-evidence "$fake_root/approval.json" >/dev/null 2>&1
  rc=$?
  set -e
  [[ "$rc" == 2 && ! -e "$fake_work" && ! -e "$fake_scratch" ]] || failed=1
  grep -F 'uv run --no-cache --no-project --offline --python 3.12 python' "$trace" >/dev/null || failed=1
  if grep -Eq 'uv sync|git clone|snapshot_download|cargo |cuda' "$trace"; then failed=1; fi
  local args_file="$temporary/apple.args"
  local hash_4b hash_ref_4b hash_8b hash_ref_8b
  hash_4b="$(printf 'a%.0s' {1..64})"
  hash_ref_4b="$(printf 'b%.0s' {1..64})"
  hash_8b="$(printf 'c%.0s' {1..64})"
  hash_ref_8b="$(printf 'd%.0s' {1..64})"
  write_apple_args "$args_file" "$hash_4b" "$hash_ref_4b" "$hash_8b" "$hash_ref_8b"
  grep -F '<APPLE_GGUF_4B_PATH>' "$args_file" >/dev/null || failed=1
  grep -F '<APPLE_REFERENCE_4B_DIR>' "$args_file" >/dev/null || failed=1
  grep -F '<APPLE_GGUF_8B_PATH>' "$args_file" >/dev/null || failed=1
  grep -F '<APPLE_REFERENCE_8B_DIR>' "$args_file" >/dev/null || failed=1
  grep -F -- "--gguf-4b '<APPLE_GGUF_4B_PATH>'" "$args_file" >/dev/null || failed=1
  grep -F -- "--reference-4b '<APPLE_REFERENCE_4B_DIR>'" "$args_file" >/dev/null || failed=1
  grep -F -- "--gguf-8b '<APPLE_GGUF_8B_PATH>'" "$args_file" >/dev/null || failed=1
  grep -F -- "--reference-8b '<APPLE_REFERENCE_8B_DIR>'" "$args_file" >/dev/null || failed=1
  bash -n "$args_file" || failed=1
  grep -F -- "$hash_4b" "$args_file" >/dev/null || failed=1
  grep -F -- "$hash_ref_4b" "$args_file" >/dev/null || failed=1
  grep -F -- "$hash_8b" "$args_file" >/dev/null || failed=1
  grep -F -- "$hash_ref_8b" "$args_file" >/dev/null || failed=1
  if grep -F "$fake_root" "$args_file" >/dev/null || grep -F "$temporary" "$args_file" >/dev/null; then failed=1; fi
  # shellcheck disable=SC2086 # Each case intentionally models argv tokenization.
  for bad_args in \
    "--self-test --approval-evidence x" \
    "--self-test --self-test" \
    "--variant 4b --variant 8b" \
    "--approval-evidence" \
    "--approval-evidence --work-dir x" \
    "--unknown x"; do
    if bash "$script_path" $bad_args >/dev/null 2>&1; then failed=1; fi
  done
  rm -rf "$temporary"
  trap - EXIT
  if (( failed != 0 )); then
    log "self-test FAIL"
    return 1
  fi
  log "self-test PASS"
}

main() {
  local selection='' work_dir='' approval_evidence='' self_test=0
  local seen_variant=0 seen_work_dir=0 seen_approval=0 seen_self_test=0
  while (( $# > 0 )); do
    case "$1" in
      --variant)
        (( seen_variant == 0 )) || die "duplicate --variant"
        [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || { usage; die "--variant requires a nonempty value"; }
        seen_variant=1
        selection="$2"
        shift 2
        ;;
      --work-dir)
        (( seen_work_dir == 0 )) || die "duplicate --work-dir"
        [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || { usage; die "--work-dir requires a nonempty value"; }
        seen_work_dir=1
        work_dir="$2"
        shift 2
        ;;
      --approval-evidence)
        (( seen_approval == 0 )) || die "duplicate --approval-evidence"
        [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || { usage; die "--approval-evidence requires a nonempty value"; }
        seen_approval=1
        approval_evidence="$2"
        shift 2
        ;;
      --self-test)
        (( seen_self_test == 0 )) || die "duplicate --self-test"
        seen_self_test=1
        self_test=1
        shift
        ;;
      -h|--help)
        usage
        return 0
        ;;
      *)
        usage
        die "unknown argument $1"
        ;;
    esac
  done
  if (( self_test == 1 )); then
    [[ -z "$selection$work_dir$approval_evidence" ]] || die "--self-test accepts no other arguments"
    run_self_test
    return
  fi
  [[ -n "$approval_evidence" ]] || { usage; die "--approval-evidence is required"; }
  [[ -f "$approval_evidence" && ! -L "$approval_evidence" && -s "$approval_evidence" ]] || die "approval evidence must be a nonempty regular file"
  case "$selection" in
    4b|8b|all) ;;
    *) usage; die "--variant must be 4b, 8b, or all" ;;
  esac

  pre_sync_gate "$approval_evidence"
  require_vast_host
  require_tooling
  if [[ -z "$work_dir" ]]; then
    work_dir="$VOKRA_SCRATCH/moss-audio-validation-$(git -C "$VOKRA_ROOT" rev-parse --short=12 HEAD)"
  fi
  require_absent_work_dir "$work_dir" "$approval_evidence"
  mkdir -p "$work_dir"
  local evidence_dir="$work_dir/evidence"
  local source_dir="$work_dir/official-source"
  mkdir -p "$evidence_dir"
  record_environment "$evidence_dir/environment.txt"

  step "Install the locked official reference environment"
  UV_NO_CACHE=1 uv sync --no-cache --project "$PARITY_PROJECT" --frozen --python 3.12 \
    2>&1 | tee "$evidence_dir/uv-sync.log"

  step "Checkout the immutable official OpenMOSS source"
  checkout_official_source "$source_dir" \
    2>&1 | tee "$evidence_dir/source-checkout.log"
  step "Verify the exact checked-out official source before imports"
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python "$PREFLIGHT_GATE" \
    --verify-source --source "$source_dir"

  step "Build the current Vokra CLI on VAST"
  cargo build --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release -p vokra-cli \
    2>&1 | tee "$evidence_dir/build-cli.log"

  if [[ "$selection" == "4b" || "$selection" == "all" ]]; then
    run_variant 4b "$work_dir" "$evidence_dir" "$source_dir"
  fi
  if [[ "$selection" == "8b" || "$selection" == "all" ]]; then
    run_variant 8b "$work_dir" "$evidence_dir" "$source_dir"
  fi

  if [[ "$selection" == "all" ]]; then
    local apple_args="$evidence_dir/apple-silicon-moss-audio.args.sh"
    write_apple_args "$apple_args" \
      "$(sha256_file "$work_dir/moss-audio-4b-instruct.gguf")" \
      "$(sha256_file "$evidence_dir/reference-4b/manifest.txt")" \
      "$(sha256_file "$work_dir/moss-audio-8b-instruct.gguf")" \
      "$(sha256_file "$evidence_dir/reference-8b/manifest.txt")"
  fi

  step "Run repository gates and full workspace verification on VAST"
  cargo fmt --manifest-path "$VOKRA_ROOT/Cargo.toml" --all -- --check
  bash "$VOKRA_ROOT/scripts/check-zero-deps.sh"
  bash "$VOKRA_ROOT/scripts/check-forbidden-symbols.sh"
  bash "$VOKRA_ROOT/scripts/check-bound-arch-coverage.sh"
  bash "$VOKRA_ROOT/scripts/check-arch-handshake.sh"
  cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --workspace \
    2>&1 | tee "$evidence_dir/workspace-test.log"
  cargo clippy --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --workspace \
    --all-targets -- -D warnings 2>&1 | tee "$evidence_dir/workspace-clippy.log"

  step "Cross-check Apple Metal feature compilation"
  rustup target add aarch64-apple-darwin
  cargo check --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked \
    -p vokra-models --features metal --target aarch64-apple-darwin \
    2>&1 | tee "$evidence_dir/apple-metal-cross-check.log"

  {
    echo "verdict=PASS"
    echo "selection=$selection"
    echo "git_commit=$(git -C "$VOKRA_ROOT" rev-parse HEAD)"
    echo "workspace_test=PASS"
    echo "workspace_clippy=PASS"
    echo "apple_metal_cross_compile=PASS"
    echo "apple_real_weight_runtime=PENDING_SEPARATE_APPLE_SILICON_RUN"
    echo "upload=NOT_PERFORMED"
  } > "$evidence_dir/summary.txt"
  log "PASS: pull only $evidence_dir (including reference-*), never source-* or *.gguf"
  log "After evidence is pulled, destroy the VAST instance; do not merely stop it"
}

main "$@"
