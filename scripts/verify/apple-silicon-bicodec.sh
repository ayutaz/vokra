#!/usr/bin/env bash
# Real-weight SparkAudio BiCodec CPU/Metal parity on a disposable Apple host.
# Inputs are VAST-produced/authenticated; this verifier never downloads,
# converts, uploads, publishes, or silently falls back to CPU.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VOKRA_ROOT="${VOKRA_ROOT:-$DEFAULT_ROOT}"
TEST_SOURCE="$VOKRA_ROOT/crates/vokra-models/src/bicodec/mod.rs"
TEST_NAME="official_reference_measured_parity"
TEST_SELECTOR="bicodec::tests::$TEST_NAME"
MIN_MEMORY_BYTES=24000000000
MIN_FREE_DISK_KIB=12000000
UPSTREAM_HF_REVISION="642071559bfc6346c2359d19dcb6be3f9dd8a05d"
CHECKPOINT_SHA256="e9940cd48d4446e4340ced82d234bf5618350dd9f5db900ebe47a4fdb03867ec"
CONFIG_SHA256="744f4093ae2381a2eb44ea8c4a5268a8d1e581498e9bf0808c034d1b076429be"
SOURCE_REPOSITORY="https://github.com/SparkAudio/Spark-TTS"
SOURCE_REVISION="2f1ea9082400547242641f5271b6f941c9f439d1"

log() { printf '[bicodec-apple] %s\n' "$*" >&2; }
die() { log "ERROR: $*"; return 2; }
sha256_file() { shasum -a 256 "$1" | awk '{print $1}'; }
require_absolute() { [[ "$2" == /* ]] || die "$1 must be absolute: $2"; }
require_file() { [[ -f "$2" && ! -L "$2" && -s "$2" ]] || die "$1 is missing, empty, or symlinked: $2"; }

usage() {
  cat <<'EOF' >&2
usage: apple-silicon-bicodec.sh --gguf <file> --gguf-sha256 <64-hex> \
       --reference <directory> --reference-sha256 <64-hex> \
       --approval-evidence <file> --evidence-dir <absent-dir>
       apple-silicon-bicodec.sh --self-test

Runs authenticated BiCodec official-reference parity once on CPU and once on
real Metal on a disposable Apple arm64 host. It performs no download,
conversion, upload, publication, or CPU fallback.
EOF
}

canonical_existing() {
  local path="$1" rest component scan parent
  [[ "$path" == /* && -e "$path" && ! -L "$path" ]] || return 1
  rest="${path#/}"; scan=''
  while [[ -n "$rest" ]]; do
    if [[ "$rest" == */* ]]; then component="${rest%%/*}"; rest="${rest#*/}"; else component="$rest"; rest=''; fi
    [[ -n "$component" ]] || continue
    [[ "$component" != . && "$component" != .. ]] || return 1
    scan="$scan/$component"
    [[ ! -L "$scan" || "$scan" == "/var" ]] || return 1
  done
  if [[ -d "$path" ]]; then (cd -P "$path" && pwd); else parent="$(dirname "$path")"; (cd -P "$parent" && printf '%s/%s\n' "$PWD" "$(basename "$path")"); fi
}

canonical_absent() {
  local path="$1" rest component scan name suffix='' parent
  [[ "$path" == /* && ! -e "$path" && ! -L "$path" ]] || return 1
  rest="${path#/}"; scan=''
  while [[ -n "$rest" ]]; do
    if [[ "$rest" == */* ]]; then component="${rest%%/*}"; rest="${rest#*/}"; else component="$rest"; rest=''; fi
    [[ -n "$component" ]] || continue
    [[ "$component" != . && "$component" != .. ]] || return 1
    scan="$scan/$component"
    [[ ! -L "$scan" || "$scan" == "/var" ]] || return 1
  done
  while [[ ! -e "$path" ]]; do name="$(basename "$path")"; [[ -n "$name" ]] && suffix="/$name$suffix"; parent="$(dirname "$path")"; [[ "$parent" != "$path" ]] || return 1; path="$parent"; done
  [[ -d "$path" && ! -L "$path" ]] || return 1
  (cd -P "$path" && printf '%s%s\n' "$PWD" "$suffix")
}

paths_overlap() { [[ "$1" == "$2" || "$1" == "$2/"* || "$2" == "$1/"* ]]; }
require_disjoint_evidence() {
  local evidence="$1"; shift; local evidence_real protected protected_real
  require_absolute 'evidence directory' "$evidence"
  [[ ! -e "$evidence" && ! -L "$evidence" ]] || { die "evidence directory must be absent: $evidence"; return 2; }
  [[ -d "$(dirname "$evidence")" ]] || { die 'evidence parent is unavailable'; return 2; }
  evidence_real="$(canonical_absent "$evidence")" || { die 'evidence path cannot be canonicalized'; return 2; }
  for protected in "$@"; do
    [[ -e "$protected" || -L "$protected" ]] || { die "protected input missing: $protected"; return 2; }
    protected_real="$(canonical_existing "$protected")" || { die "protected input cannot be canonicalized: $protected"; return 2; }
    paths_overlap "$evidence_real" "$protected_real" && { die "evidence overlaps protected input: $protected"; return 2; }
  done
  mkdir -p "$evidence"
}

require_reference_files() {
  local directory="$1" entry name
  local expected='manifest.json semantic_latent.f32 d_vector.f32 prenet_output.f32 waveform.f32'
  require_absolute 'reference directory' "$directory"
  [[ -d "$directory" && ! -L "$directory" ]] || { die "reference is not a directory: $directory"; return 2; }
  while IFS= read -r entry; do
    name="$(basename "$entry")"
    [[ -f "$entry" && ! -L "$entry" ]] || { die "reference entry is not regular/non-symlink: $name"; return 2; }
    case " $expected " in *" $name "*) ;; *) die "unexpected reference entry: $name"; return 2 ;; esac
  done < <(find -P "$directory" -mindepth 1 -maxdepth 1 -print)
  for name in $expected; do require_file "reference $name" "$directory/$name"; done
}

require_reference_manifest() {
  local directory="$1" manifest="$1/manifest.json"
  require_reference_files "$directory"
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python - "$manifest" "$directory" <<'PY'
import hashlib, json, re, sys
from pathlib import Path
manifest_path, directory = map(Path, sys.argv[1:])
def reject(pairs):
    result = {}
    for key, value in pairs:
        if key in result: raise ValueError(f"duplicate key: {key}")
        result[key] = value
    return result
def keys(value, expected, label):
    if not isinstance(value, dict) or set(value) != set(expected): raise ValueError(f"{label} keys drifted")
def string(value, expected, label):
    if type(value) is not str or value != expected: raise ValueError(f"{label} drifted")
data = json.loads(manifest_path.read_text(encoding="utf-8"), object_pairs_hook=reject)
keys(data, {"schema","provenance","contract","tokens","token_contract","tensors"}, "manifest")
string(data["schema"], "vokra-bicodec-official-reference-v1", "schema")
provenance = data["provenance"]
keys(provenance, {"oracle","source_repository","source_revision","upstream_hf_revision","checkpoint_sha256","config_sha256","randomness","upload"}, "provenance")
for key, value in {
    "oracle":"SparkAudio/Spark-TTS BiCodec official source",
    "source_repository":"https://github.com/SparkAudio/Spark-TTS",
    "source_revision":"2f1ea9082400547242641f5271b6f941c9f439d1",
    "upstream_hf_revision":"642071559bfc6346c2359d19dcb6be3f9dd8a05d",
    "checkpoint_sha256":"e9940cd48d4446e4340ced82d234bf5618350dd9f5db900ebe47a4fdb03867ec",
    "config_sha256":"744f4093ae2381a2eb44ea8c4a5268a8d1e581498e9bf0808c034d1b076429be",
    "randomness":"none (fixed literal token vectors)", "upload":"none",
}.items(): string(provenance[key], value, f"provenance.{key}")
contract = data["contract"]
keys(contract, {"sample_rate","frame_hop","semantic_vocab","semantic_codebook_dim","semantic_latent_dim","global_vocab","global_tokens"}, "contract")
for key, value in {"sample_rate":16000,"frame_hop":320,"semantic_vocab":8192,"semantic_codebook_dim":8,"semantic_latent_dim":1024,"global_vocab":4096,"global_tokens":32}.items():
    if type(contract[key]) is not int or contract[key] != value: raise ValueError(f"contract.{key} drifted")
tokens = data["tokens"]; keys(tokens, {"semantic","global"}, "tokens")
semantic, global_values = [0,1,4096,8191], [0,1,4095,16,255,1024,2048,3072] * 4
if tokens["semantic"] != semantic or tokens["global"] != global_values: raise ValueError("fixed tokens drifted")
token_contract = data["token_contract"]; keys(token_contract, {"semantic_csv","global_csv"}, "token_contract")
if token_contract["semantic_csv"] != ",".join(map(str, semantic)) or token_contract["global_csv"] != ",".join(map(str, global_values)): raise ValueError("token CSV drifted")
expected = {"semantic_latent":([1,1024,4],16384),"d_vector":([1,1024],4096),"prenet_output":([1,1024,4],16384),"waveform":([1,1,1280],5120)}
keys(data["tensors"], expected, "tensors")
for role, (shape, size) in expected.items():
    row = data["tensors"][role]; keys(row, {"path","shape","dtype","bytes","sha256"}, f"tensors.{role}")
    if row["path"] != f"{role}.f32" or row["shape"] != shape or row["dtype"] != "F32" or type(row["bytes"]) is not int or row["bytes"] != size or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"]): raise ValueError(f"tensor record drifted: {role}")
    artifact = directory / row["path"]
    if artifact.is_symlink() or not artifact.is_file() or artifact.stat().st_size != size or hashlib.sha256(artifact.read_bytes()).hexdigest() != row["sha256"]: raise ValueError(f"tensor artifact drifted: {role}")
PY
}

require_approval() {
  local approval="$1"
  require_absolute 'approval evidence' "$approval"; require_file 'approval evidence' "$approval"
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 \
    python - "$approval" "$VOKRA_EXPECTED_COMMIT" "$CHECKPOINT_SHA256" "$CONFIG_SHA256" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

def reject_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate approval key: {key}")
        result[key] = value
    return result

try:
    approval = json.loads(
        Path(sys.argv[1]).read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicates,
    )
    expected_keys = {
        "schema", "model", "upstream_repo", "upstream_revision",
        "upstream_hf_revision", "license_spdx", "checkpoint_sha256",
        "config_sha256", "git_commit", "no_upload", "decision", "signer",
        "scope_sha256",
    }
    if not isinstance(approval, dict) or set(approval) != expected_keys:
        raise ValueError("approval schema is not exact")
    expected = {
        "schema": "vokra-bicodec-approval-v1",
        "model": "SparkAudio/Spark-TTS-0.5B",
        "upstream_repo": "https://github.com/SparkAudio/Spark-TTS",
        "upstream_revision": "2f1ea9082400547242641f5271b6f941c9f439d1",
        "upstream_hf_revision": "642071559bfc6346c2359d19dcb6be3f9dd8a05d",
        "license_spdx": "cc-by-nc-sa-4.0",
        "checkpoint_sha256": sys.argv[3],
        "config_sha256": sys.argv[4],
        "git_commit": sys.argv[2],
        "no_upload": True,
        "decision": "RESEARCH_ONLY",
    }
    for key, value in expected.items():
        if approval[key] != value:
            raise ValueError(f"approval identity drift: {key}")
    if type(approval["no_upload"]) is not bool:
        raise ValueError("approval no_upload must be a JSON boolean")
    if not isinstance(approval["signer"], str) or not approval["signer"].strip():
        raise ValueError("approval signer is unresolved")
    if len(approval["git_commit"]) != 40 or any(c not in "0123456789abcdef" for c in approval["git_commit"]):
        raise ValueError("approval git commit is not lowercase 40-hex")
    for key in ("checkpoint_sha256", "config_sha256"):
        if (not isinstance(approval[key], str) or len(approval[key]) != 64
                or any(c not in "0123456789abcdef" for c in approval[key])):
            raise ValueError(f"approval {key} is not lowercase 64-hex")
    scope = {key: approval[key] for key in expected_keys if key not in {"scope_sha256", "signer"}}
    scope_digest = hashlib.sha256(
        json.dumps(scope, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if approval["scope_sha256"] != scope_digest:
        raise ValueError("approval scope digest mismatch")
except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
    raise SystemExit(f"approval gate BLOCKED: {exc}")
PY
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 \
    python "$VOKRA_ROOT/scripts/publish/signoff_match.py" \
    --check-repo bicodec --audit "$VOKRA_ROOT/docs/license-audit.md" \
    >/dev/null || die 'repository bicodec signoff is not APPROVED'
}

require_remote_host() {
  local memory_bytes free_disk_kib
  [[ "${VOKRA_REMOTE_APPLE_SILICON:-0}" == 1 ]] || die 'VOKRA_REMOTE_APPLE_SILICON=1 is absent'
  [[ "$(uname -s)" == Darwin && "$(uname -m)" == arm64 ]] || die 'BiCodec Metal validation requires Darwin arm64'
  memory_bytes="$(sysctl -n hw.memsize)"; [[ "$memory_bytes" =~ ^[0-9]+$ ]] || die 'invalid hw.memsize'
  (( memory_bytes >= MIN_MEMORY_BYTES )) || die 'physical memory is below the 24-GB remote-worker guard'
  free_disk_kib="$(df -Pk "$VOKRA_ROOT" | awk 'NR == 2 {print $4}')"; [[ "$free_disk_kib" =~ ^[0-9]+$ ]] || die 'invalid free disk'
  (( free_disk_kib >= MIN_FREE_DISK_KIB )) || die 'free disk is below the 12-GB run guard'
}

require_tooling() {
  local tool actual_commit
  for tool in cargo rustc git shasum awk find tee grep sysctl sw_vers system_profiler xcrun uv; do command -v "$tool" >/dev/null 2>&1 || die "required tool missing: $tool"; done
  [[ -d "$VOKRA_ROOT/.git" && -f "$VOKRA_ROOT/Cargo.toml" && -f "$TEST_SOURCE" && ! -L "$TEST_SOURCE" ]] || die 'Vokra checkout or BiCodec test source is missing'
  [[ -n "${VOKRA_EXPECTED_COMMIT:-}" && "$VOKRA_EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]] || die 'VOKRA_EXPECTED_COMMIT must be the exact lowercase 40-hex checkout commit'
  actual_commit="$(git -C "$VOKRA_ROOT" rev-parse HEAD)"
  [[ "$actual_commit" == "$VOKRA_EXPECTED_COMMIT" ]] || die 'checkout commit does not match VOKRA_EXPECTED_COMMIT'
  [[ -z "$(git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all)" ]] || die 'Apple checkout must be clean'
  xcrun -f metal >/dev/null 2>&1 || die 'Xcode Metal compiler is unavailable'
}

require_backend_contract() {
  grep -Fq "fn $TEST_NAME" "$TEST_SOURCE" || die "named BiCodec test is missing: $TEST_NAME"
  grep -Fq 'VOKRA_BICODEC_PARITY_BACKEND' "$TEST_SOURCE" || die 'BiCodec test lacks explicit backend selector; refusing CPU-as-Metal labeling'
  grep -Fq 'BackendKind::Metal' "$TEST_SOURCE" || die 'BiCodec test lacks Metal arm; refusing silent CPU fallback'
  grep -Fq 'from_gguf_with_backend' "$TEST_SOURCE" || die 'BiCodec test does not bind selected backend explicitly'
  grep -Fq 'BICODEC_MEASURED_PARITY_BACKEND' "$TEST_SOURCE" || die 'BiCodec test lacks backend-specific PASS sentinel'
}

record_environment() {
  local output="$1"
  {
    echo "utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "git_commit=$(git -C "$VOKRA_ROOT" rev-parse HEAD)"
    echo "expected_commit=$VOKRA_EXPECTED_COMMIT"
    echo "uname=$(uname -a)"; echo "machine=$(uname -m)"
    echo "memory_bytes=$(sysctl -n hw.memsize)"; echo "physical_cpu=$(sysctl -n hw.physicalcpu)"; echo "logical_cpu=$(sysctl -n hw.logicalcpu)"
    sw_vers; rustc --version --verbose; cargo --version; echo "metal_compiler=$(xcrun -f metal)"
    system_profiler SPDisplaysDataType | sed -n '1,40p'
  } > "$output"
}

require_test_pass() {
  local log_path="$1" backend="$2" count stage
  [[ "$(grep -Fxc "test $TEST_SELECTOR ... ok" "$log_path" || true)" == 1 ]] || { die "$backend log lacks one exact passing named test"; return 2; }
  [[ "$(grep -Ec '^test result: ok[.] 1 passed; 0 failed; 0 ignored; 0 measured; [0-9]+ filtered out(; finished in .+)?$' "$log_path" || true)" == 1 && "$(grep -Ec '^test result:' "$log_path" || true)" == 1 ]] || { die "$backend log lacks one exact result"; return 2; }
  for stage in semantic_latent d_vector prenet_output waveform; do
    count="$(grep -Ec "^BICODEC_MEASURED_PARITY stage=$stage .* verdict=PASS$" "$log_path" || true)"
    [[ "$count" == 1 ]] || { die "$backend log lacks one PASS marker for $stage"; return 2; }
  done
  [[ "$(grep -Fxc "BICODEC_MEASURED_PARITY_BACKEND backend=$backend verdict=PASS" "$log_path" || true)" == 1 ]] \
    || { die "$backend log lacks one backend-specific PASS sentinel"; return 2; }
  ! grep -Eq '^BICODEC_MEASURED_PARITY .* verdict=FAIL$' "$log_path" || die "$backend log contains failed parity marker"
}

run_parity() {
  local backend="$1" gguf="$2" reference="$3" log_path="$4"
  env VOKRA_BICODEC_PARITY_GGUF="$gguf" VOKRA_BICODEC_PARITY_REFERENCE="$reference" VOKRA_BICODEC_PARITY_BACKEND="$backend" CARGO_BUILD_JOBS=1 RUST_TEST_THREADS=1 \
    cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release --lib -p vokra-models --features metal "$TEST_SELECTOR" -- --ignored --exact --show-output --test-threads=1 2>&1 | tee "$log_path"
}

run_self_test() (
  local temporary name forbidden_publication
  temporary="$(mktemp -d "${TMPDIR:-/tmp}/vokra-bicodec-apple.XXXXXX")"; trap 'rm -rf -- "$temporary"' EXIT
  printf abc > "$temporary/value"
  [[ "$(sha256_file "$temporary/value")" == ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad ]] || die 'SHA helper self-test failed'
  mkdir "$temporary/reference"
  for name in manifest.json semantic_latent.f32 d_vector.f32 prenet_output.f32 waveform.f32; do printf x > "$temporary/reference/$name"; done
  require_reference_files "$temporary/reference"
  printf x > "$temporary/reference/unexpected.bin"; if require_reference_files "$temporary/reference" >/dev/null 2>&1; then die 'extra reference entry accepted'; fi; rm "$temporary/reference/unexpected.bin"
  mkdir "$temporary/existing"; if require_disjoint_evidence "$temporary/existing" "$temporary/value" >/dev/null 2>&1; then die 'existing evidence accepted'; fi; rmdir "$temporary/existing"
  require_disjoint_evidence "$temporary/new-evidence" "$temporary/value"
  ln -s value "$temporary/value-link"; if require_file symlink "$temporary/value-link" >/dev/null 2>&1; then die 'symlink input accepted'; fi
  mkdir "$temporary/real-parent"; ln -s "$temporary/real-parent" "$temporary/link-parent"
  if require_disjoint_evidence "$temporary/link-parent/new-evidence" "$temporary/value" >/dev/null 2>&1; then die 'symlink ancestor accepted'; fi
  printf '%s\n' "test $TEST_SELECTOR ... ok" 'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out' \
    'BICODEC_MEASURED_PARITY stage=semantic_latent elements=1 max_abs=0 rmse=0 verdict=PASS' \
    'BICODEC_MEASURED_PARITY stage=d_vector elements=1 max_abs=0 rmse=0 verdict=PASS' \
    'BICODEC_MEASURED_PARITY stage=prenet_output elements=1 max_abs=0 rmse=0 verdict=PASS' \
    'BICODEC_MEASURED_PARITY stage=waveform elements=1 max_abs=0 rmse=0 verdict=PASS' \
    'BICODEC_MEASURED_PARITY_BACKEND backend=cpu verdict=PASS' > "$temporary/valid.log"
  require_test_pass "$temporary/valid.log" cpu
  cp "$temporary/valid.log" "$temporary/duplicate.log"; printf '%s\n' 'BICODEC_MEASURED_PARITY stage=waveform elements=1 max_abs=0 rmse=0 verdict=PASS' >> "$temporary/duplicate.log"
  if require_test_pass "$temporary/duplicate.log" cpu >/dev/null 2>&1; then die 'duplicate marker accepted'; fi
  sed 's/0 filtered out/2975 filtered out/' "$temporary/valid.log" > "$temporary/filtered.log"
  require_test_pass "$temporary/filtered.log" cpu
  if bash "$0" --help >"$temporary/help.txt" 2>&1; then :; else die 'help invocation failed'; fi
  grep -Fq 'usage: apple-silicon-bicodec.sh' "$temporary/help.txt" || die 'help output is incomplete'
  for token in 'run_parity cpu' 'run_parity metal' 'VOKRA_BICODEC_PARITY_BACKEND' 'backend=metal verdict=PASS'; do
    grep -Fq -- "$token" "$0" || die "self-test missing backend contract: $token"
  done
  grep -Fq "require_file 'GGUF'" "$0" || die 'self-test missing regular-file helper call'
  local legacy_helper='require_regular_'
  if grep -Fq "${legacy_helper}file" "$0"; then die 'self-test found undefined legacy helper'; fi
  for bad_args in '--self-test --self-test' '--self-test --gguf x' '--gguf x --gguf y' '--unknown x'; do
    # shellcheck disable=SC2086
    if bash "$0" $bad_args >/dev/null 2>&1; then die "accepted malformed parser case: $bad_args"; fi
  done
  forbidden_publication='git[[:space:]]+push|publish-one[.]sh|--'"push"
  if grep -En "$forbidden_publication" "$0" >/dev/null; then die 'publication command found'; fi
  echo 'apple-silicon-bicodec.sh self-test: OK'
)

main() {
  local gguf='' gguf_digest='' reference='' reference_digest='' approval='' evidence='' self_test=0 pair label value
  local seen_gguf=0 seen_gguf_digest=0 seen_reference=0 seen_reference_digest=0 seen_approval=0 seen_evidence=0 seen_self_test=0
  while (( $# > 0 )); do
    case "$1" in
      --self-test) (( seen_self_test == 0 )) || die 'duplicate --self-test'; seen_self_test=1; self_test=1; shift ;;
      --gguf) (( seen_gguf == 0 )) || die 'duplicate --gguf'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die '--gguf requires a nonempty value'; seen_gguf=1; gguf="$2"; shift 2 ;;
      --gguf-sha256) (( seen_gguf_digest == 0 )) || die 'duplicate --gguf-sha256'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die '--gguf-sha256 requires a nonempty value'; seen_gguf_digest=1; gguf_digest="$2"; shift 2 ;;
      --reference) (( seen_reference == 0 )) || die 'duplicate --reference'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die '--reference requires a nonempty value'; seen_reference=1; reference="$2"; shift 2 ;;
      --reference-sha256) (( seen_reference_digest == 0 )) || die 'duplicate --reference-sha256'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die '--reference-sha256 requires a nonempty value'; seen_reference_digest=1; reference_digest="$2"; shift 2 ;;
      --approval-evidence) (( seen_approval == 0 )) || die 'duplicate --approval-evidence'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die '--approval-evidence requires a nonempty value'; seen_approval=1; approval="$2"; shift 2 ;;
      --evidence-dir) (( seen_evidence == 0 )) || die 'duplicate --evidence-dir'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die '--evidence-dir requires a nonempty value'; seen_evidence=1; evidence="$2"; shift 2 ;;
      -h|--help) usage; return 0 ;;
      *) usage; die "unknown argument: $1" ;;
    esac
  done
  if (( self_test )); then [[ -z "$gguf$gguf_digest$reference$reference_digest$approval$evidence" ]] || die '--self-test accepts no other arguments'; run_self_test; return; fi
  [[ -n "$gguf$gguf_digest$reference$reference_digest$approval$evidence" ]] || { usage; die 'all inputs are required'; }
  for pair in "GGUF path|$gguf" "reference directory|$reference" "approval evidence|$approval" "evidence directory|$evidence"; do label="${pair%%|*}"; value="${pair#*|}"; require_absolute "$label" "$value"; done
  [[ "$gguf_digest" =~ ^[0-9a-f]{64}$ && "$reference_digest" =~ ^[0-9a-f]{64}$ ]] || die 'input hashes must be lowercase 64-hex'
  require_tooling; require_remote_host; require_backend_contract
  require_file 'GGUF' "$gguf"; [[ "$(sha256_file "$gguf")" == "$gguf_digest" ]] || die 'GGUF SHA-256 mismatch'
  require_reference_manifest "$reference"; [[ "$(sha256_file "$reference/manifest.json")" == "$reference_digest" ]] || die 'reference manifest SHA-256 mismatch'
  require_approval "$approval"; require_disjoint_evidence "$evidence" "$VOKRA_ROOT" "$gguf" "$reference" "$approval"
  record_environment "$evidence/environment.txt"
  printf '%s\n' "gguf_sha256=$gguf_digest" "reference_manifest_sha256=$reference_digest" "checkpoint_sha256=$CHECKPOINT_SHA256" "config_sha256=$CONFIG_SHA256" "upstream_hf_revision=$UPSTREAM_HF_REVISION" "source_repository=$SOURCE_REPOSITORY" "source_revision=$SOURCE_REVISION" > "$evidence/input-hashes.txt"
  log 'running authenticated BiCodec CPU parity'; run_parity cpu "$gguf" "$reference" "$evidence/parity-cpu.log"; require_test_pass "$evidence/parity-cpu.log" cpu
  log 'running authenticated BiCodec Metal parity'; run_parity metal "$gguf" "$reference" "$evidence/parity-metal.log"; require_test_pass "$evidence/parity-metal.log" metal
  printf '%s\n' 'verdict=PASS' "git_commit=$(git -C "$VOKRA_ROOT" rev-parse HEAD)" "gguf_sha256=$gguf_digest" "reference_manifest_sha256=$reference_digest" 'cpu_vs_official=PASS' 'metal_vs_official=PASS' 'publication=NO_UPLOAD' 'fallback=FORBIDDEN' > "$evidence/summary.txt"
  log "PASS: evidence written to $evidence; remove staged inputs or destroy worker"
}
main "$@"
