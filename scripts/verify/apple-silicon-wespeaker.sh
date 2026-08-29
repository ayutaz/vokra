#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VOKRA_ROOT="${VOKRA_ROOT:-$DEFAULT_ROOT}"
PARITY_PROJECT="$VOKRA_ROOT/tools/parity/wespeaker"
PREFLIGHT_GATE="$PARITY_PROJECT/preflight_gate.py"
PREFLIGHT_MANIFEST="$PARITY_PROJECT/license_gate_manifest.json"
SOURCE_REVISION="45941e7cba2c3ea99e232d02bedf617fc71b0dad"
MODEL_REVISION="f0c48c298fd835726c27956a5d617bad7115627e"
CHECKPOINT_SHA256="9872b375f2c6a3851ca471cbbf59e06efd23a627d78bf5872e1f0269fd298449"
REFERENCE_FILES=(manifest.json pcm.f32.bin features.f32.bin embedding.f32.bin)
REFERENCE_KEYS=(bytes_embedding_f32_bin bytes_features_f32_bin bytes_pcm_f32_bin checkpoint_sha256 device embedding_dtype embedding_shape feature_shape features_dtype format model_id model_revision numpy pcm_dtype pcm_samples python runtime sample_rate sha256_embedding_f32_bin sha256_features_f32_bin sha256_pcm_f32_bin source_revision torch torchaudio)
TEST_NAME="official_combined_artifact_matches_upstream_wespeaker"
CPU_SENTINEL="WESPEAKER_OFFICIAL_COMBINED_CPU_VS_UPSTREAM PASS"
METAL_SENTINEL="WESPEAKER_OFFICIAL_COMBINED_METAL_VS_CPU PASS"
log() { printf '[wespeaker-apple] %s\n' "$*" >&2; }
die() { log "ERROR: $*"; return 2; }
sha256_file() { shasum -a 256 "$1" | awk '{print $1}'; }
drop_last_line() {
  local file="$1" lines
  lines="$(wc -l < "$file" | tr -d '[:space:]')"
  sed -n "1,$((lines - 1))p" "$file" > "$file.tmp"
  mv "$file.tmp" "$file"
}
usage() { printf '%s\n' 'usage: apple-silicon-wespeaker.sh --gguf PATH --gguf-sha256 HEX64 --reference DIR --reference-manifest-sha256 HEX64 --approval-evidence JSON --evidence-dir ABSENT_DIR' '       apple-silicon-wespeaker.sh --self-test' >&2; }
require_file() {
  local label="$1" path="$2"
  [[ -f "$path" && ! -L "$path" ]] || { die "$label is missing, symlinked, or not regular"; return 2; }
}
require_hash() {
  local label="$1" path="$2" expected="$3" actual
  [[ "$expected" =~ ^[0-9a-f]{64}$ ]] || { die "$label expected hash is malformed"; return 2; }
  require_file "$label" "$path" || return 2
  actual="$(sha256_file "$path")"
  [[ "$actual" == "$expected" ]] || { die "$label hash mismatch"; return 2; }
}
manifest_value() {
  local file="$1" key="$2" value="$3"
  [[ "$(grep -Ec "^  \"$key\": $value,?$" "$file" || true)" == 1 ]] || { die "manifest field $key is missing, duplicated, or wrong"; return 2; }
}
require_manifest_keys() {
  local file="$1" actual expected
  actual="$(sed -n 's/^  "\([^"]*\)":.*/\1/p' "$file" | sort)"
  expected="$(printf '%s\n' "${REFERENCE_KEYS[@]}" | sort)"
  [[ "$actual" == "$expected" && -z "$(printf '%s\n' "$actual" | uniq -d)" ]] || { die "manifest key set is not exact"; return 2; }
}
require_reference() {
  local directory="$1" name expected actual manifest key
  [[ -d "$directory" && ! -L "$directory" ]] || { die "reference root is missing or symlinked"; return 2; }
  expected="$(printf '%s\n' "${REFERENCE_FILES[@]}" | sort)"
  actual="$(find "$directory" -mindepth 1 -maxdepth 1 -exec basename {} \; | sort)"
  [[ "$actual" == "$expected" ]] || { die "reference contains extra or missing entries"; return 2; }
  for name in "${REFERENCE_FILES[@]}"; do require_file "reference $name" "$directory/$name" || return 2; done
  manifest="$directory/manifest.json"
  require_manifest_keys "$manifest" || return 2
  manifest_value "$manifest" format '"vokra-wespeaker-reference-v1"' || return 2
  manifest_value "$manifest" model_id '"Wespeaker/wespeaker-voxceleb-resnet34-LM"' || return 2
  manifest_value "$manifest" model_revision "\"$MODEL_REVISION\"" || return 2
  manifest_value "$manifest" source_revision "\"$SOURCE_REVISION\"" || return 2
  manifest_value "$manifest" checkpoint_sha256 "\"$CHECKPOINT_SHA256\"" || return 2
  manifest_value "$manifest" sample_rate 16000 || return 2
  manifest_value "$manifest" pcm_samples 32000 || return 2
  manifest_value "$manifest" feature_shape '\[198, 80\]' || return 2
  manifest_value "$manifest" embedding_shape '\[1, 256\]' || return 2
  manifest_value "$manifest" runtime '"torch-cpu"' || return 2
  manifest_value "$manifest" device '"cpu"' || return 2
  manifest_value "$manifest" pcm_dtype '"float32-le"' || return 2
  manifest_value "$manifest" features_dtype '"float32-le"' || return 2
  manifest_value "$manifest" embedding_dtype '"float32-le"' || return 2
  for name in "${REFERENCE_FILES[@]:1}"; do
    key="sha256_${name//./_}"
    expected="$(grep -E "^  \"$key\":" "$manifest" | sed -E 's/.*: "([^"]*)",?$/\1/')"
    require_hash "reference $name" "$directory/$name" "$expected" || return 2
    key="bytes_${name//./_}"
    expected="$(grep -E "^  \"$key\":" "$manifest" | sed -E 's/.*: ([0-9]+),?$/\1/')"
    [[ "$expected" =~ ^[0-9]+$ && "$(wc -c < "$directory/$name" | tr -d '[:space:]')" == "$expected" ]] || { die "reference $name byte count mismatch"; return 2; }
  done
}
require_cargo_result() {
  local file="$1" named total results
  named="$(grep -Ec "^test $TEST_NAME \.\.\. ok$" "$file" || true)"
  total="$(grep -Ec '^test [^ ]+ \.\.\.' "$file" || true)"
  results="$(grep -Ec '^test result:' "$file" || true)"
  [[ "$named" == 1 && "$total" == 1 && "$results" == 1 ]] || { die "Cargo named/total test count is not exactly one"; return 2; }
  grep -Eq '^test result: ok\. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out(; finished in [0-9]+\.[0-9]+s)?$' "$file" || { die "Cargo result is not exact"; return 2; }
}
require_sentinel() {
  local file="$1" expected="$2" family count
  family="$(grep -Ec '^WESPEAKER_OFFICIAL_COMBINED_(CPU_VS_UPSTREAM|METAL_VS_CPU) (PASS|FAIL)$' "$file" || true)"
  count="$(grep -Ec "^${expected// /[[:space:]]+}$" "$file" || true)"
  [[ "$family" == 1 && "$count" == 1 ]] || { die "sentinel family is not one exact PASS"; return 2; }
}
require_both_sentinels() {
  local file="$1"
  [[ "$(grep -Ec '^WESPEAKER_OFFICIAL_COMBINED_(CPU_VS_UPSTREAM|METAL_VS_CPU) PASS$' "$file" || true)" == 2 ]] || { die "CPU/Metal sentinel family is incomplete"; return 2; }
  local expected
  for expected in "$CPU_SENTINEL" "$METAL_SENTINEL"; do
    [[ "$(grep -Ec "^${expected// /[[:space:]]+}$" "$file" || true)" == 1 ]] || { die "missing or duplicate sentinel"; return 2; }
  done
}
require_host() {
  [[ "${VOKRA_REMOTE_APPLE_SILICON:-0}" == 1 && "$(uname -s)" == Darwin && "$(uname -m)" == arm64 ]] || die "requires VOKRA_REMOTE_APPLE_SILICON=1 on Darwin arm64"
  [[ -z "${WESPEAKER_ATOL:-}" && -z "${WESPEAKER_RTOL:-}" ]] || die "numeric overrides are not accepted"
}
license_preflight() {
  local approval="$1"
  command -v uv >/dev/null 2>&1 || die "uv is required before the WeSpeaker gate"
  [[ -f "$PARITY_PROJECT/uv.lock" && ! -L "$PARITY_PROJECT/uv.lock" && -f "$PARITY_PROJECT/pyproject.toml" && ! -L "$PARITY_PROJECT/pyproject.toml" && -f "$PREFLIGHT_GATE" && ! -L "$PREFLIGHT_GATE" && -f "$PREFLIGHT_MANIFEST" && ! -L "$PREFLIGHT_MANIFEST" ]] || die "WeSpeaker preflight inputs are missing or symlinked"
  [[ -f "$approval" && -s "$approval" && ! -L "$approval" ]] || die "approval evidence must be a non-empty regular non-symlink file"
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python "$PREFLIGHT_GATE" --project "$PARITY_PROJECT" --manifest "$PREFLIGHT_MANIFEST" --approval-evidence "$approval"
}
canonicalize_uncreated() {
  local path="$1" suffix='' name parent scan rest component
  [[ "$path" == /* ]] || path="$PWD/$path"
  rest="${path#/}"; scan=''
  while [[ -n "$rest" ]]; do
    component="${rest%%/*}"; rest="${rest#*/}"
    [[ "$component" == "$rest" ]] && rest=''
    [[ -n "$component" && "$component" != . && "$component" != .. ]] || continue
    scan="$scan/$component"
    [[ ! -L "$scan" || "$scan" == "/var" ]] || return 1
  done
  while [[ ! -d "$path" || -L "$path" ]]; do
    name="${path##*/}"; [[ -n "$name" ]] && suffix="/$name$suffix"
    parent="${path%/*}"; [[ "$parent" == "$path" ]] && parent='.'
    [[ -n "$parent" ]] || parent='/'; path="$parent"
  done
  (cd -P "$path" && printf '%s%s\n' "$PWD" "$suffix")
}
paths_overlap() { [[ "$1" == "$2" || "$1" == "$2"/* || "$2" == "$1"/* ]]; }
require_disjoint_path() {
  local candidate="$1" other="$2" candidate_abs other_abs
  candidate_abs="$(canonicalize_uncreated "$candidate")" || { die "cannot canonicalize evidence path"; return 2; }
  other_abs="$(canonicalize_uncreated "$other")" || { die "cannot canonicalize protected path"; return 2; }
  if paths_overlap "$candidate_abs" "$other_abs"; then
    die "evidence path is not disjoint"
    return 2
  fi
  return 0
}
require_absent_evidence_dir() {
  local target="$1"; shift
  [[ ! -e "$target" && ! -L "$target" ]] || { die "evidence directory must be absent and non-symlink"; return 2; }
  [[ -d "$(dirname "$target")" && ! -L "$(dirname "$target")" ]] || { die "evidence parent directory is missing or symlinked"; return 2; }
  local protected
  for protected in "$VOKRA_ROOT" "$@"; do
    require_disjoint_path "$target" "$protected" || return 2
  done
  return 0
}
run_self_test() {
  local tmp="$TMPDIR/wespeaker-apple-selftest.$$"
  SELFTEST_TMP="$tmp"
  mkdir "$tmp"; trap 'rm -rf -- "$SELFTEST_TMP"' EXIT
  mkdir "$tmp/reference"
  for name in "${REFERENCE_FILES[@]}"; do [[ "$name" == manifest.json ]] || printf abc > "$tmp/reference/$name"; done
  cat > "$tmp/reference/manifest.json" <<EOF
{
  "bytes_embedding_f32_bin": 3,
  "bytes_features_f32_bin": 3,
  "bytes_pcm_f32_bin": 3,
  "checkpoint_sha256": "$CHECKPOINT_SHA256",
  "embedding_shape": [1, 256],
  "feature_shape": [198, 80],
  "format": "vokra-wespeaker-reference-v1",
  "device": "cpu",
  "embedding_dtype": "float32-le",
  "features_dtype": "float32-le",
  "model_id": "Wespeaker/wespeaker-voxceleb-resnet34-LM",
  "model_revision": "$MODEL_REVISION",
  "numpy": "2.3.5",
  "pcm_samples": 32000,
  "python": "3.12",
  "pcm_dtype": "float32-le",
  "runtime": "torch-cpu",
  "sample_rate": 16000,
  "sha256_embedding_f32_bin": "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
  "sha256_features_f32_bin": "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
  "sha256_pcm_f32_bin": "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
  "source_revision": "$SOURCE_REVISION",
  "torch": "2.9.1",
  "torchaudio": "2.9.1"
}
EOF
  require_reference "$tmp/reference"
  printf 'extra' > "$tmp/reference/extra"
  if require_reference "$tmp/reference" >/dev/null 2>&1; then die "extra reference entry accepted"; fi
  rm "$tmp/reference/extra"
  rm "$tmp/reference/pcm.f32.bin"; ln -s features.f32.bin "$tmp/reference/pcm.f32.bin"
  if require_reference "$tmp/reference" >/dev/null 2>&1; then die "expected reference symlink accepted"; fi
  rm "$tmp/reference/pcm.f32.bin"; printf abc > "$tmp/reference/pcm.f32.bin"
  printf '  "format": "vokra-wespeaker-reference-v1",\n' >> "$tmp/reference/manifest.json"
  if require_reference "$tmp/reference" >/dev/null 2>&1; then die "duplicate manifest key accepted"; fi
  drop_last_line "$tmp/reference/manifest.json"
  ln -s "$tmp/reference" "$tmp/root-link"
  if require_reference "$tmp/root-link" >/dev/null 2>&1; then die "root symlink accepted"; fi
  printf '{}' > "$tmp/approval.json"
  printf abc > "$tmp/value"
  require_absent_evidence_dir "$tmp/new-evidence" "$tmp/value" "$tmp/approval.json"
  mkdir "$tmp/empty-evidence"
  if require_absent_evidence_dir "$tmp/empty-evidence" "$tmp/value" "$tmp/approval.json" >/dev/null 2>&1; then die "existing empty evidence accepted"; fi
  ln -s "$tmp/missing-evidence" "$tmp/dangling-evidence"
  if require_absent_evidence_dir "$tmp/dangling-evidence" "$tmp/value" "$tmp/approval.json" >/dev/null 2>&1; then die "dangling evidence symlink accepted"; fi
  if require_absent_evidence_dir "$VOKRA_ROOT/wespeaker-apple-self-test" "$tmp/value" "$tmp/approval.json" >/dev/null 2>&1; then die "checkout-overlap evidence accepted"; fi
  if require_absent_evidence_dir "$tmp/value/child" "$tmp/value" "$tmp/approval.json" >/dev/null 2>&1; then die "input-overlap evidence accepted"; fi
  if require_absent_evidence_dir "$tmp/approval.json/child" "$tmp/value" "$tmp/approval.json" >/dev/null 2>&1; then die "approval-overlap evidence accepted"; fi
  mkdir -p "$tmp/real-parent/child"
  ln -s "$tmp/real-parent" "$tmp/link-parent"
  if require_absent_evidence_dir "$tmp/link-parent/child/new" "$tmp/value" "$tmp/approval.json" >/dev/null 2>&1; then die "intermediate evidence symlink accepted"; fi
  printf 'test %s ... ok\n' "$TEST_NAME" > "$tmp/log"
  printf '%s\n%s\n' 'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out' "$CPU_SENTINEL" >> "$tmp/log"
  require_cargo_result "$tmp/log"
  printf '%s\n' 'test malformed ... ok' >> "$tmp/log"
  if require_cargo_result "$tmp/log" >/dev/null 2>&1; then die "extra Cargo test line accepted"; fi
  drop_last_line "$tmp/log"
  printf '%s\n' 'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.2x' >> "$tmp/log"
  if require_cargo_result "$tmp/log" >/dev/null 2>&1; then die "malformed Cargo timing accepted"; fi
  drop_last_line "$tmp/log"
  if require_both_sentinels "$tmp/log" >/dev/null 2>&1; then die "missing Metal sentinel accepted"; fi
  printf '%s\n' "$METAL_SENTINEL" >> "$tmp/log"
  require_both_sentinels "$tmp/log"
  printf '%s\n' "$METAL_SENTINEL" >> "$tmp/log"
  if require_both_sentinels "$tmp/log" >/dev/null 2>&1; then die "duplicate sentinel accepted"; fi
  echo "apple-silicon-wespeaker self-test: PASS"
}
main() {
  if [[ "${1:-}" == --self-test ]]; then [[ $# == 1 ]] || { usage; return 2; }; run_self_test; return; fi
  local gguf='' gguf_sha='' reference='' reference_sha='' approval='' evidence='' arg seen=''
  while [[ $# -gt 0 ]]; do
    arg="$1"; shift
    case "$arg" in
      --gguf|--gguf-sha256|--reference|--reference-manifest-sha256|--approval-evidence|--evidence-dir)
        [[ "$seen" != *"|$arg|"* ]] || { usage; die "duplicate argument: $arg"; }
        seen+="|$arg|"
        [[ $# -gt 0 && -n "$1" && "$1" != -* ]] || { usage; return 2; }
        case "$arg" in --gguf) gguf="$1";; --gguf-sha256) gguf_sha="$1";; --reference) reference="$1";; --reference-manifest-sha256) reference_sha="$1";; --approval-evidence) approval="$1";; --evidence-dir) evidence="$1";; esac
        shift;;
      -h|--help) usage; return 0;; *) usage; die "unknown argument: $arg";;
    esac
  done
  [[ -n "$gguf" && -n "$gguf_sha" && -n "$reference" && -n "$reference_sha" && -n "$approval" && -n "$evidence" ]] || { usage; return 2; }
  license_preflight "$approval"
  require_host
  require_hash "WeSpeaker GGUF" "$gguf" "$gguf_sha"
  require_hash "reference manifest" "$reference/manifest.json" "$reference_sha"
  require_reference "$reference"
  require_absent_evidence_dir "$evidence" "$gguf" "$reference" "$approval"
  [[ -d "$VOKRA_ROOT/.git" && -z "$(git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all)" ]] || die "checkout must be clean"
  mkdir -p "$evidence"
  local log_file="$evidence/parity.log"
  env VOKRA_WESPEAKER_OFFICIAL_GGUF="$gguf" RUST_TEST_THREADS=1 cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release -p vokra-models --features metal --test parity_wespeaker_real "$TEST_NAME" -- --exact --nocapture 2>&1 | tee "$log_file"
  require_cargo_result "$log_file"; require_both_sentinels "$log_file"
  printf 'verdict=PASS\ngguf_sha256=%s\nreference_manifest_sha256=%s\nupload=NOT_PERFORMED\n' "$gguf_sha" "$reference_sha" > "$evidence/summary.txt"
}
main "$@"
