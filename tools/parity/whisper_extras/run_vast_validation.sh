#!/usr/bin/env bash
# VAST-only real-weight CPU parity runner. NO_UPLOAD by construction.
# The GitHub workflow is model-free; this is the authoritative >=2 GiB route.
set -euo pipefail

DISTIL_REPO="distil-whisper/distil-large-v3.5"
DISTIL_REVISION="728a7691f3ff1d3d971528d3203a6e9559165d41"
KOTOBA_REPO="kotoba-tech/kotoba-whisper-v2.2"
KOTOBA_REVISION="9d33482a0eb9b57f1ad80708e8ac5538246d8355"
AUDIO="tests/fixtures/audio/jfk-30s.wav"
AUDIO_SHA256="58adb4ea501d955fcd40bfbb69128f8f40428b81d8716b9ed337949773be253f"
AUDIO_BYTES=352078
NO_UPLOAD=1

# Fixed identity pins are verified before conversion on VAST.
DISTIL_MODEL_BYTES=3025686376
DISTIL_MODEL_SHA256="76ec9f754fc4b4810845dc36b71d1897c1342e702810c179e1569690084cfb0c"
KOTOBA_MODEL_BYTES=3025686376
KOTOBA_MODEL_SHA256="e0ef3e7b379515f0c35d0e7885638ddef0fa9f5c8e3e3f88cbc6da9b39edd1e9"
DISTIL_CONFIG_BYTES=1249
DISTIL_CONFIG_SHA256="515a10a9979258d3fc71cf79b2cd055c189f07d78879a15bd9bc282673308b85"
DISTIL_GENERATION_BYTES=4249
DISTIL_GENERATION_SHA256="b521c66612bd95be36c154f2d3904f6e4ea3be481a18a48f293d66791f60cf98"
DISTIL_TOKENIZER_BYTES=2480645
DISTIL_TOKENIZER_SHA256="b3c8202bbf06d8ee4232c5984baa563784ac4737e2e7fdc42fa180200d3cfcdb"
KOTOBA_CONFIG_BYTES=1499
KOTOBA_CONFIG_SHA256="75e0166afbf44308af4908793fc5ade1707890ed15760f0529b468dcfc378aec"
KOTOBA_GENERATION_BYTES=3898
KOTOBA_GENERATION_SHA256="20d28b9169207ab6ca402ec9342393a88ec3f341d88c9da24e516e1da71c24de"
KOTOBA_TOKENIZER_BYTES=3931381
KOTOBA_TOKENIZER_SHA256="615928f5a25409279b47b47d87a4ca2aaee3bd09f65e1a3df6c9d23c718cfdb0"

die() { echo "run_vast_validation: $*" >&2; exit 2; }

# Keep every Cargo invocation serial and locked, including the existing
# evidence-producing commands below. `command` bypasses this wrapper.
cargo() { CARGO_BUILD_JOBS=1 command cargo --locked "$@"; }

file_bytes() {
  if stat -c '%s' "$1" >/dev/null 2>&1; then
    stat -c '%s' "$1"
  else
    stat -f '%z' "$1"
  fi
}

verify_file_identity() {
  local path="$1" expected_bytes="$2" expected_sha="$3" label="$4"
  [[ -f "$path" ]] || die "$label missing: $path"
  local actual_bytes actual_sha
  actual_bytes="$(file_bytes "$path")"
  [[ "$actual_bytes" == "$expected_bytes" ]] || die "$label byte count drift: $actual_bytes != $expected_bytes"
  if command -v sha256sum >/dev/null; then
    actual_sha="$(sha256sum "$path" | awk '{print $1}')"
  else
    actual_sha="$(shasum -a 256 "$path" | awk '{print $1}')"
  fi
  [[ "$actual_sha" == "$expected_sha" ]] || die "$label SHA-256 drift: $actual_sha != $expected_sha"
}

verify_license_signoffs() {
  local audit="docs/license-audit.md" matcher="scripts/publish/signoff_match.py"
  [[ -f "$audit" && -f "$matcher" ]] || die "license audit/signoff matcher missing"
  local state row count line
  state="$(uv run --no-project --python 3.12 python - "$matcher" "$audit" <<'PY'
import runpy
import sys
from pathlib import Path

module = runpy.run_path(sys.argv[1])
rows = module["parse_signoff_rows"](Path(sys.argv[2]))
expected = {
    "distil-whisper/distil-large-v3.5": True,
    "kotoba-tech/kotoba-whisper-v2.2": True,
}
for name, approved in expected.items():
    if rows.get(name) is not approved:
        raise SystemExit(f"{name}: signed §3.1 row is missing or not approved")
print("APPROVED")
PY
  )"
  [[ "$state" == "APPROVED" ]] || die "license signoff verification failed: $state"
  # Require one complete canonical row per artifact, not a substring fragment.
  for row in "distil-whisper/distil-large-v3.5" "kotoba-tech/kotoba-whisper-v2.2"; do
    count="$(awk -v needle="| **$row** |" 'index($0, needle) == 1 { n++ } END { print n + 0 }' "$audit")"
    [[ "$count" == 1 ]] || die "license audit must contain exactly one complete row: $row (count=$count)"
    line="$(awk -v needle="| **$row** |" 'index($0, needle) == 1 { print; exit }' "$audit")"
    [[ "$line" == *"| MIT |"* || "$line" == *"| Apache-2.0 |"* ]] || die "license cell drift: $row"
    [[ "$line" == *"| ☑ Commercial / ☐ Research-only / ☐ Rejected |"* ]] || die "commercial signoff drift: $row"
    [[ "$line" == *"yousan"* ]] || die "approver missing from signed row: $row"
  done
}

work_dir=""
self_test=0
while (($#)); do
  case "$1" in
    --self-test)
      ((self_test == 0)) || die "--self-test specified more than once"
      self_test=1
      shift
      ;;
    --work-dir)
      [[ -n "${2:-}" && "$2" != -* ]] || die "--work-dir requires a value"
      [[ -z "$work_dir" ]] || die "--work-dir specified more than once"
      work_dir="$2"
      shift 2
      ;;
    *)
      die "unknown argument '$1'; use --self-test or --work-dir <absolute absent directory>"
      ;;
  esac
done

if ((self_test)); then
  [[ -z "$work_dir" ]] || die "--self-test cannot be combined with --work-dir"
  if [[ -f "$AUDIO" ]]; then
    selftest_bytes="$(wc -c < "$AUDIO" | tr -d '[:space:]')"
    [[ "$selftest_bytes" == "$AUDIO_BYTES" ]] || die "audio fixture byte count drift"
    if command -v sha256sum >/dev/null; then
      selftest_sha="$(sha256sum "$AUDIO" | awk '{print $1}')"
    else
      selftest_sha="$(shasum -a 256 "$AUDIO" | awk '{print $1}')"
    fi
    [[ "$selftest_sha" == "$AUDIO_SHA256" ]] || die "audio fixture SHA-256 drift"
  fi
  PYTHONDONTWRITEBYTECODE=1 uv run --no-project --python 3.12 python \
    tools/parity/whisper_extras/dump_reference.py --self-test
  selftest_identity_bytes="$(wc -c < "$0" | tr -d '[:space:]')"
  if command -v sha256sum >/dev/null; then
    selftest_identity_sha="$(sha256sum "$0" | awk '{print $1}')"
  else
    selftest_identity_sha="$(shasum -a 256 "$0" | awk '{print $1}')"
  fi
  verify_file_identity "$0" "$selftest_identity_bytes" "$selftest_identity_sha" "runner self-test"
  if (verify_file_identity "$0" 0 "$selftest_identity_sha" "runner negative self-test") 2>/dev/null; then
    die "identity verifier accepted a wrong byte count"
  fi
  for needle in "$DISTIL_REPO" "$DISTIL_REVISION" "$KOTOBA_REPO" "$KOTOBA_REVISION" \
    "$AUDIO_SHA256" "$AUDIO_BYTES" "dump_reference.py" "NO_UPLOAD" \
    "--work-dir" "evidence" "git rev-parse HEAD" "tar -czf" \
    "cargo test --release -p vokra-models" "--locked" "CARGO_BUILD_JOBS=1" \
    "verify_file_identity" "verify_license_signoffs" "DISTIL_MODEL_SHA256" \
    "KOTOBA_MODEL_SHA256" "DISTIL_CONFIG_SHA256" "KOTOBA_CONFIG_SHA256" \
    "DISTIL_GENERATION_SHA256" "KOTOBA_GENERATION_SHA256" \
    "DISTIL_TOKENIZER_SHA256" "KOTOBA_TOKENIZER_SHA256" \
    "snapshot-identity.txt" "--locked" \
    "CARGO_BUILD_JOBS=1"; do
    grep -Fq -- "$needle" "$0" || die "self-test missing $needle"
  done
  verify_license_signoffs
  ! grep -Eq 'git[[:space:]]+push|publish-one\.sh|upload\.sh' "$0" || die "publication command present"
  echo "run_vast_validation self-test: OK"
  exit 0
fi

[[ -n "$work_dir" ]] || die "--work-dir <absolute absent directory> is required"
[[ "$work_dir" = /* ]] || die "work directory must be an absolute path"
[[ ! -e "$work_dir" && ! -L "$work_dir" ]] || die "work directory must be absent"
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || die "Linux x86_64 VAST required"
[[ -f Cargo.toml && -f "$AUDIO" ]] || die "run from a clean repository root"
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || die "clean committed checkout required"
verify_license_signoffs
[[ "$(stat -c '%s' "$AUDIO")" == "$AUDIO_BYTES" ]] || die "audio fixture byte count drift"
[[ "$(sha256sum "$AUDIO" | awk '{print $1}')" == "$AUDIO_SHA256" ]] || die "audio fixture SHA-256 drift"
mem_kib="$(awk '$1 == "MemTotal:" {print $2; exit}' /proc/meminfo)"
[[ "$mem_kib" =~ ^[0-9]+$ && "$mem_kib" -ge $((64 * 1024 * 1024)) ]] || die "64 GiB RAM required"
free_kib="$(df -Pk "$(dirname "$work_dir")" | awk 'NR == 2 {print $4}')"
[[ "$free_kib" =~ ^[0-9]+$ && "$free_kib" -ge $((150 * 1024 * 1024)) ]] || die "150 GiB free disk required"
for command in cargo uv sha256sum tar tee; do command -v "$command" >/dev/null || die "missing $command"; done

mkdir "$work_dir"
mkdir "$work_dir/snapshots" "$work_dir/gguf" "$work_dir/reference" "$work_dir/evidence"
evidence="$work_dir/evidence"
git rev-parse HEAD > "$evidence/git-head.txt"
{
  echo "audio=$AUDIO"
  echo "audio_bytes=$AUDIO_BYTES"
  echo "audio_sha256=$AUDIO_SHA256"
  echo "distil_repo=$DISTIL_REPO"
  echo "distil_revision=$DISTIL_REVISION"
  echo "kotoba_repo=$KOTOBA_REPO"
  echo "kotoba_revision=$KOTOBA_REVISION"
  echo "distil_model_bytes=$DISTIL_MODEL_BYTES"
  echo "distil_model_sha256=$DISTIL_MODEL_SHA256"
  echo "kotoba_model_bytes=$KOTOBA_MODEL_BYTES"
  echo "kotoba_model_sha256=$KOTOBA_MODEL_SHA256"
  echo "NO_UPLOAD=$NO_UPLOAD"
} > "$evidence/input-sha256.txt"

uv sync --frozen --project tools/parity/whisper_extras 2>&1 | tee "$evidence/setup.log"
cargo build --release -p vokra-cli 2>&1 | tee -a "$evidence/setup.log"

download() {
  local repo="$1" revision="$2" slug="$3" destination="$work_dir/snapshots/$3"
  uv run --frozen --project tools/parity/whisper_extras python - "$repo" "$revision" "$destination" <<'PY'
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

repo, revision, destination = sys.argv[1:]
path = Path(destination)
if path.exists() or path.is_symlink():
    raise SystemExit(f"snapshot destination must be absent: {path}")
snapshot_download(
    repo_id=repo,
    revision=revision,
    local_dir=path,
    allow_patterns=[
        "config.json", "generation_config.json", "preprocessor_config.json",
        "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
        "vocab.json", "merges.txt", "model.safetensors", "*.txt",
    ],
)
if not (path / "config.json").is_file() or not (path / "model.safetensors").is_file():
    raise SystemExit(f"snapshot incomplete: {path}")
PY
  case "$slug" in
    distil_whisper)
      verify_file_identity "$destination/model.safetensors" "$DISTIL_MODEL_BYTES" "$DISTIL_MODEL_SHA256" "$slug model"
      verify_file_identity "$destination/config.json" "$DISTIL_CONFIG_BYTES" "$DISTIL_CONFIG_SHA256" "$slug config"
      verify_file_identity "$destination/generation_config.json" "$DISTIL_GENERATION_BYTES" "$DISTIL_GENERATION_SHA256" "$slug generation_config"
      verify_file_identity "$destination/tokenizer.json" "$DISTIL_TOKENIZER_BYTES" "$DISTIL_TOKENIZER_SHA256" "$slug tokenizer"
      ;;
    kotoba_whisper)
      verify_file_identity "$destination/model.safetensors" "$KOTOBA_MODEL_BYTES" "$KOTOBA_MODEL_SHA256" "$slug model"
      verify_file_identity "$destination/config.json" "$KOTOBA_CONFIG_BYTES" "$KOTOBA_CONFIG_SHA256" "$slug config"
      verify_file_identity "$destination/generation_config.json" "$KOTOBA_GENERATION_BYTES" "$KOTOBA_GENERATION_SHA256" "$slug generation_config"
      verify_file_identity "$destination/tokenizer.json" "$KOTOBA_TOKENIZER_BYTES" "$KOTOBA_TOKENIZER_SHA256" "$slug tokenizer"
      ;;
    *) die "unknown snapshot slug: $slug" ;;
  esac
  {
    for file in model.safetensors config.json generation_config.json tokenizer.json; do
      printf '%s bytes=%s sha256=%s\n' "$file" "$(stat -c '%s' "$destination/$file")" "$(sha256sum "$destination/$file" | awk '{print $1}')"
    done
  } > "$evidence/$slug-snapshot-identity.txt"
}

run_one() {
  local slug="$1" repo="$2" revision="$3" model="distil-whisper"
  [[ "$slug" == distil_whisper ]] || model="kotoba-whisper-v2.2"
  local snapshot="$work_dir/snapshots/$slug" gguf="$work_dir/gguf/$slug.gguf"
  local reference="$work_dir/reference/$slug" log="$evidence/$slug.log"
  {
    echo "[$slug] repo=$repo revision=$revision"
    download "$repo" "$revision" "$slug"
    ./target/release/vokra-cli convert --model "$model" --input "$snapshot/model.safetensors" --output "$gguf"
    uv run --frozen --project tools/parity/whisper_extras python \
      tools/parity/whisper_extras/dump_reference.py \
      --model "$slug" --checkpoint-dir "$snapshot" --audio "$AUDIO" --output-dir "$reference"
    local upper
    upper="$(printf '%s' "$slug" | tr '[:lower:]' '[:upper:]')"
    env "VOKRA_${upper}_GGUF=$gguf" "VOKRA_${upper}_REFDIR=$reference" \
      cargo test --release -p vokra-models --test parity_whisper_extras \
      "parity_whisper_extras_${slug}" -- --nocapture --test-threads=1
    cp "$reference/manifest.json" "$evidence/${slug}-manifest.json"
    sha256sum "$reference"/* > "$evidence/${slug}-packet-sha256.txt"
    echo "[$slug] CPU parity PASS"
  } 2>&1 | tee "$log"
}

run_one distil_whisper "$DISTIL_REPO" "$DISTIL_REVISION"
run_one kotoba_whisper "$KOTOBA_REPO" "$KOTOBA_REVISION"
{
  echo "status=CPU real-weight parity PASS"
  echo "NO_UPLOAD=$NO_UPLOAD"
  echo "authoritative_route=VAST"
  echo "destroy_instance_after_evidence=true"
} > "$evidence/result.txt"
tar -czf "$work_dir/whisper-extras-evidence.tar.gz" -C "$work_dir" evidence
echo "CPU real-weight parity PASS; evidence=$work_dir/whisper-extras-evidence.tar.gz; NO_UPLOAD; destroy this VAST instance after retrieval"
