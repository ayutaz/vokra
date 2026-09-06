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

require_output_closure() {
  local output="$1" entry relative name
  local expected='bicodec.gguf input-hashes.txt apple-transfer-args.txt parity-cpu.log summary.txt reference reference/manifest.json reference/semantic_latent.f32 reference/d_vector.f32 reference/prenet_output.f32 reference/waveform.f32'
  while IFS= read -r entry; do
    relative="${entry#"$output/"}"
    [[ "$entry" != *'/'./* && "$entry" != *'/'../* ]] || die "output contains lexical dot path: $relative"
    [[ -e "$entry" && ! -L "$entry" ]] || die "output contains missing or symlinked entry: $relative"
    case " $expected " in *" $relative "*) ;; *) die "output contains unexpected entry: $relative" ;; esac
  done < <(find -P "$output" -mindepth 1 -print)
  for relative in $expected; do
    entry="$output/$relative"
    [[ -e "$entry" && ! -L "$entry" ]] || die "output closure is missing: $relative"
    if [[ "$relative" == reference ]]; then
      [[ -d "$entry" ]] || die "reference closure is not a directory"
    else
      [[ -f "$entry" && -s "$entry" ]] || die "output closure entry is not a non-empty file: $relative"
    fi
  done
}

write_apple_transfer_args() {
  local output="$1" gguf_sha="$2" reference_sha="$3" expected_head="$4"
  {
    printf '%q ' apple-silicon-bicodec.sh \
      --gguf '<BICODEC_GGUF>' --gguf-sha256 "$gguf_sha" \
      --reference '<BICODEC_REFERENCE_DIR>' --reference-sha256 "$reference_sha" \
      --approval-evidence '<BICODEC_APPROVAL_JSON>' --evidence-dir '<BICODEC_EVIDENCE_DIR>' \
      --expected-head "$expected_head"
    printf '\n'
  } > "$output"
}

require_approval() {
  local approval="$1" expected_head="$2"
  [[ "$approval" == /* && -f "$approval" && ! -L "$approval" ]] || die 'approval evidence must be an absolute regular non-symlink file'
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python - "$approval" "$expected_head" <<'PY'
import hashlib, json, sys
from pathlib import Path

def reject(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key: {key}")
        result[key] = value
    return result

try:
    value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"), object_pairs_hook=reject)
    expected_keys = {"schema", "model", "upstream_repo", "upstream_revision", "upstream_hf_revision", "license_spdx", "checkpoint_sha256", "config_sha256", "git_commit", "no_upload", "decision", "signer", "scope_sha256"}
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise ValueError("approval schema is not exact")
    expected = {
        "schema": "vokra-bicodec-approval-v1",
        "model": "SparkAudio/Spark-TTS-0.5B",
        "upstream_repo": "https://github.com/SparkAudio/Spark-TTS",
        "upstream_revision": "2f1ea9082400547242641f5271b6f941c9f439d1",
        "upstream_hf_revision": "642071559bfc6346c2359d19dcb6be3f9dd8a05d",
        "license_spdx": "cc-by-nc-sa-4.0",
        "checkpoint_sha256": "e9940cd48d4446e4340ced82d234bf5618350dd9f5db900ebe47a4fdb03867ec",
        "config_sha256": "744f4093ae2381a2eb44ea8c4a5268a8d1e581498e9bf0808c034d1b076429be",
        "git_commit": sys.argv[2], "no_upload": True, "decision": "RESEARCH_ONLY",
    }
    for key, expected_value in expected.items():
        if value[key] != expected_value:
            raise ValueError(f"approval identity drift: {key}")
    if (type(value["no_upload"]) is not bool or not isinstance(value["signer"], str)
            or not value["signer"].strip()
            or value["signer"].strip().upper() in {"TODO", "TBD", "UNRESOLVED", "UNKNOWN", "PENDING"}):
        raise ValueError("approval signer/no_upload is invalid")
    if not isinstance(value["git_commit"], str) or len(value["git_commit"]) != 40 or any(c not in "0123456789abcdef" for c in value["git_commit"]):
        raise ValueError("approval git_commit is not lowercase 40-hex")
    scope = {key: value[key] for key in expected_keys if key not in {"scope_sha256", "signer"}}
    digest = hashlib.sha256(json.dumps(scope, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if value["scope_sha256"] != digest:
        raise ValueError("approval scope digest mismatch")
except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
    raise SystemExit(f"approval gate BLOCKED: {error}")
PY
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 \
    python scripts/publish/signoff_match.py --check-repo bicodec --audit docs/license-audit.md \
    >/dev/null || die 'repository BiCodec owner signoff is not approved'
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
  mkdir "$temporary/closure" "$temporary/closure/reference"
  for name in bicodec.gguf input-hashes.txt apple-transfer-args.txt parity-cpu.log summary.txt; do printf x > "$temporary/closure/$name"; done
  for name in manifest.json semantic_latent.f32 d_vector.f32 prenet_output.f32 waveform.f32; do printf x > "$temporary/closure/reference/$name"; done
  require_output_closure "$temporary/closure"
  mkdir "$temporary/closure/unexpected-dir"
  if (require_output_closure "$temporary/closure") >/dev/null 2>&1; then die 'unexpected output directory accepted'; fi
  grep -Fq -- 'VOKRA_BICODEC_PARITY_BACKEND=cpu' "$0" || die 'CPU selector missing from production command'
  grep -Fq -- '--expected-head' "$0" || die 'exact checkout HEAD gate is missing'
  grep -Fq -- '--approval-evidence' "$0" || die 'owner approval gate is missing'
  grep -Fq -- '--evidence-dir' "$0" || die 'Apple transfer args lack evidence directory'
  grep -Fq -- "reference_dir=\"\$output/reference\"" "$0" || die 'reference packet is not isolated from evidence output'
  grep -Fq -- 'verdict=CPU_PASS_METAL_NOT_RUN' "$0" || die 'CPU-only verdict is not explicit'
  grep -Fq -- 'signoff_match.py --check-repo bicodec' "$0" || die 'repository owner signoff gate is missing'
  if bash "$0" --self-test --self-test >/dev/null 2>&1; then die 'duplicate --self-test accepted'; fi
  grep -Fq -- 'cargo test --locked --offline --lib -p vokra-models' "$0" || die 'production command lacks locked offline --lib'
  grep -Fq -- '-- --ignored --exact --show-output' "$0" || die 'production command lacks harness --exact/show-output'
  write_apple_transfer_args "$temporary/transfer-args.txt" \
    'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa' \
    'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb' \
    'cccccccccccccccccccccccccccccccccccccccc'
  [[ "$(wc -l < "$temporary/transfer-args.txt" | tr -d ' ')" == 1 ]] || die 'Apple transfer args are not one line'
  read -r -a transfer_args < "$temporary/transfer-args.txt"
  for flag in --gguf --gguf-sha256 --reference --reference-sha256 --approval-evidence --evidence-dir --expected-head; do
    printf '%s\n' "${transfer_args[@]}" | grep -Fqx -- "$flag" || die "Apple transfer args omit $flag"
  done
  echo 'run-bicodec-native-parity.sh self-test: OK'
)

usage() {
  cat <<'EOF'
Usage:
  run-bicodec-native-parity.sh --source-dir <checkout> --model-dir <BiCodec> \
    --output <empty-tmpfs-dir> --approval-evidence <json> --expected-head <40-lowercase-hex>
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
approval=""
expected_head=""
seen_self_test=0
self_test=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --self-test) (( seen_self_test == 0 )) || die 'duplicate --self-test'; seen_self_test=1; self_test=1; shift ;;
    --source-dir) [[ $# -ge 2 && -z "$source_dir" ]] || die "--source-dir requires one path"; source_dir="$2"; shift 2 ;;
    --model-dir) [[ $# -ge 2 && -z "$model_dir" ]] || die "--model-dir requires one path"; model_dir="$2"; shift 2 ;;
    --output) [[ $# -ge 2 && -z "$output" ]] || die "--output requires one path"; output="$2"; shift 2 ;;
    --approval-evidence) [[ $# -ge 2 && -z "$approval" ]] || die "--approval-evidence requires one path"; approval="$2"; shift 2 ;;
    --expected-head) [[ $# -ge 2 && "$2" =~ ^[0-9a-f]{40}$ && -z "$expected_head" ]] || die "--expected-head requires one lowercase 40-hex commit"; expected_head="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done
if (( self_test )); then
  [[ -z "$source_dir$model_dir$output$approval$expected_head" ]] || die '--self-test accepts no other arguments'
  run_self_test
  exit 0
fi
[[ -n "$source_dir" && -n "$model_dir" && -n "$output" && -n "$approval" && -n "$expected_head" ]] || { usage >&2; exit 1; }
for path in "$source_dir" "$model_dir" "$output"; do [[ "$path" == /* ]] || die 'source, model, and output paths must be absolute'; done
[[ "$(uname -s)" == "Linux" ]] || die "official reference runs on Linux/VAST only"
[[ "$(uname -m)" == "x86_64" ]] || die "official reference requires x86_64"
[[ "${VOKRA_PUBLISH_ON_VAST:-0}" == "1" ]] || die "VOKRA_PUBLISH_ON_VAST=1 is absent"
[[ -f Cargo.toml && -d tools/parity ]] || die "run from a Vokra checkout"
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || die "worktree is not clean"
actual_head="$(git rev-parse HEAD)"
[[ "$actual_head" == "$expected_head" ]] || die "checkout HEAD $actual_head does not match --expected-head $expected_head"
require_approval "$approval" "$expected_head"
command -v uv >/dev/null 2>&1 || die "uv is required"
command -v findmnt >/dev/null 2>&1 || die "findmnt is required"
command -v cargo >/dev/null 2>&1 || die "cargo is required"
[[ "$(findmnt -T "$(dirname "$output")" -no FSTYPE 2>/dev/null || true)" == "tmpfs" ]] \
  || die "output parent must be tmpfs/RAM disk"
[[ ! -e "$output" && ! -L "$output" ]] || die "output must be absent (no-clobber)"
reference_dir="$output/reference"
uv run --frozen --project tools/parity --python 3.12 python \
  tools/parity/bicodec_dump_reference.py \
  --source-dir "$source_dir" --model-dir "$model_dir" --output "$reference_dir"
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-1}"
cargo build --locked --offline --release -p vokra-cli
gguf_path="$output/bicodec.gguf"
target/release/vokra-cli convert \
  --model bicodec \
  --input "$model_dir/model.safetensors" \
  --output "$gguf_path" \
  --license cc-by-nc-sa-4.0
[[ -s "$gguf_path" ]] || die "authenticated BiCodec conversion produced no GGUF"
for artifact in manifest.json semantic_latent.f32 d_vector.f32 prenet_output.f32 waveform.f32; do
  [[ -f "$reference_dir/$artifact" && ! -L "$reference_dir/$artifact" && -s "$reference_dir/$artifact" ]] || die "reference artifact missing or symlinked: $artifact"
done
[[ "$(find -P "$reference_dir" -mindepth 1 -maxdepth 1 -print | wc -l | tr -d ' ')" == 5 ]] || die 'reference packet contains an unexpected entry'
gguf_sha256="$(sha256sum "$gguf_path" | awk '{print $1}')"
reference_manifest_sha256="$(sha256sum "$reference_dir/manifest.json" | awk '{print $1}')"
{
  echo "expected_head=$expected_head"
  echo "gguf_sha256=$gguf_sha256"
  echo "reference_manifest_sha256=$reference_manifest_sha256"
  for artifact in semantic_latent.f32 d_vector.f32 prenet_output.f32 waveform.f32; do
    echo "${artifact}_sha256=$(sha256sum "$reference_dir/$artifact" | awk '{print $1}')"
  done
  echo 'publication=NO_UPLOAD'
} > "$output/input-hashes.txt"
write_apple_transfer_args "$output/apple-transfer-args.txt" "$gguf_sha256" "$reference_manifest_sha256" "$expected_head"
VOKRA_BICODEC_PARITY_GGUF="$gguf_path" \
VOKRA_BICODEC_PARITY_REFERENCE="$reference_dir" \
VOKRA_BICODEC_PARITY_BACKEND=cpu \
  cargo test --locked --offline --lib -p vokra-models \
    bicodec::tests::official_reference_measured_parity -- --ignored --exact --show-output 2>&1 | tee "$output/parity-cpu.log"
require_cpu_parity_pass "$output/parity-cpu.log"
printf '%s\n' "verdict=CPU_PASS_METAL_NOT_RUN" "expected_head=$expected_head" "gguf_sha256=$gguf_sha256" "reference_manifest_sha256=$reference_manifest_sha256" 'cpu_vs_official=PASS' 'metal_vs_official=NOT_RUN' 'metal_vs_cpu=NOT_RUN' 'publication=NO_UPLOAD' > "$output/summary.txt"
require_output_closure "$output"
echo "BiCodec official reference evidence: $output"
