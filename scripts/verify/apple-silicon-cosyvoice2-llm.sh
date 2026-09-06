#!/usr/bin/env bash
# Disposable Apple Silicon CPU/Metal parity for the standalone CosyVoice2 LLM.
# All inputs are staged and authenticated before the exact ignored test runs;
# this script never downloads, converts, uploads, publishes, or acquires data.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VOKRA_ROOT="${VOKRA_ROOT:-$DEFAULT_ROOT}"
PARITY_PROJECT="$VOKRA_ROOT/tools/parity/cosyvoice2_llm_reference"
LICENSE_GATE="$PARITY_PROJECT/preflight_gate.py"
TEST_SOURCE="$VOKRA_ROOT/crates/vokra-models/tests/parity_cosyvoice2_llm_component_real.rs"
TEST_NAME="parity_cosyvoice2_llm_component_real_apple_cpu_metal"
ATOL="3e-4"
REFERENCE_ROWS=11
MIN_MEMORY_BYTES=24000000000
MIN_FREE_DISK_KIB=12000000

log() { printf '[cosyvoice2-llm-apple] %s\n' "$*" >&2; }
die() { log "ERROR: $*"; return 2; }

usage() {
  cat >&2 <<'EOF'
usage: apple-silicon-cosyvoice2-llm.sh \
  --gguf ABS --gguf-sha256 HEX64 \
  --reference ABS --reference-manifest-sha256 HEX64 \
  --license-manifest OWNER_SIGNED_ABS --license-manifest-sha256 HEX64 \
  --evidence-dir ABSENT_DIR
       apple-silicon-cosyvoice2-llm.sh --self-test

Runs one exact ignored real-weight test on a disposable Darwin/arm64 host. The
test binds the authenticated GGUF twice, once as CPU and once as Metal, and
compares both backends with the independent Transformers reference plus
Metal/CPU logits and exact argmax. The license manifest is intentionally an
explicit external owner-signed input; the checked-in manifest is never used.
No network, model acquisition, conversion, upload, publication, or CPU
fallback is permitted.
EOF
}

sha256_file() { shasum -a 256 "$1" | awk '{print $1}'; }

require_abs() { [[ "$1" == /* ]] || die "$2 must be an absolute path"; }

reject_symlink_ancestors() {
  local path="$1" label="$2" current="$1" rest component
  require_abs "$path" "$label"
  rest="${path#/}"
  while [[ -n "$rest" ]]; do
    if [[ "$rest" == */* ]]; then component="${rest%%/*}"; rest="${rest#*/}"; else component="$rest"; rest=''; fi
    [[ "$component" != . && "$component" != .. ]] \
      || die "$label contains a dot path component: $path"
  done
  current="$path"
  while :; do
    [[ ! -L "$current" ]] || die "$label has symlink ancestry: $current"
    [[ "$current" == / ]] && break
    current="$(dirname "$current")"
  done
}

require_file() {
  local label="$1" path="$2"
  reject_symlink_ancestors "$path" "$label"
  [[ -f "$path" && ! -L "$path" && -s "$path" ]] \
    || die "$label must be a non-empty regular non-symlink file: $path"
}

canonicalize_uncreated() {
  local path="$1" suffix='' name parent
  reject_symlink_ancestors "$path" path || return 1
  while [[ ! -d "$path" || -L "$path" ]]; do
    name="${path##*/}"
    [[ -n "$name" ]] && suffix="/$name$suffix"
    parent="${path%/*}"
    [[ "$parent" == "$path" ]] && parent=/
    path="$parent"
  done
  (cd -P "$path" && printf '%s%s\n' "$PWD" "$suffix")
}

paths_overlap() {
  local left right
  left="$(canonicalize_uncreated "$1")" || return 1
  right="$(canonicalize_uncreated "$2")" || return 1
  [[ "$left" == "$right" || "$left" == "$right"/* || "$right" == "$left"/* ]]
}

require_absent_evidence_dir() {
  local target="$1" protected canonical parent
  shift
  reject_symlink_ancestors "$target" evidence
  [[ ! -e "$target" && ! -L "$target" ]] \
    || die "evidence directory must be absent and non-symlink: $target"
  parent="$(dirname "$target")"
  [[ -d "$parent" && ! -L "$parent" ]] \
    || die "evidence parent must already exist and be non-symlink: $parent"
  canonical="$(canonicalize_uncreated "$target")" \
    || die "cannot canonicalize evidence directory: $target"
  for protected in "$@"; do
    [[ -e "$protected" || -L "$protected" ]] || continue
    reject_symlink_ancestors "$protected" protected
    if paths_overlap "$canonical" "$protected"; then
      die "evidence directory overlaps protected input: $protected"
      return 2
    fi
  done
}

require_pairwise_disjoint() {
  local -a paths=("$@")
  local left_index right_index
  for ((left_index = 0; left_index < ${#paths[@]}; left_index++)); do
    for ((right_index = left_index + 1; right_index < ${#paths[@]}; right_index++)); do
      if paths_overlap "${paths[left_index]}" "${paths[right_index]}"; then
        die "protected input paths overlap: ${paths[left_index]} and ${paths[right_index]}"
        return 2
      fi
    done
  done
}

is_checked_in_license_manifest() {
  local candidate="$1" checked_in="$PARITY_PROJECT/license_gate_manifest.json"
  [[ "$(canonicalize_uncreated "$candidate")" == "$(canonicalize_uncreated "$checked_in")" ]]
}

require_expected_sha256() {
  local label="$1" expected="$2" path="$3" actual
  [[ "$expected" =~ ^[0-9a-f]{64}$ ]] || die "$label expected SHA-256 is malformed"
  require_file "$label" "$path"
  actual="$(sha256_file "$path")"
  [[ "$actual" == "$expected" ]] \
    || die "$label SHA-256 mismatch: expected $expected, got $actual"
}

require_host() {
  local memory disk
  [[ "${VOKRA_REMOTE_APPLE_SILICON:-0}" == 1 ]] \
    || die 'VOKRA_REMOTE_APPLE_SILICON=1 is absent'
  [[ "$(uname -s)" == Darwin ]] || die 'real Metal parity requires Darwin'
  [[ "$(uname -m)" == arm64 ]] || die 'real Metal parity requires arm64'
  memory="$(sysctl -n hw.memsize)"
  [[ "$memory" =~ ^[0-9]+$ && "$memory" -ge "$MIN_MEMORY_BYTES" ]] \
    || die 'physical memory is below the 24-GB remote-worker guard'
  disk="$(df -Pk "$VOKRA_ROOT" | awk 'NR == 2 {print $4}')"
  [[ "$disk" =~ ^[0-9]+$ && "$disk" -ge "$MIN_FREE_DISK_KIB" ]] \
    || die 'free disk is below the 12-GB run guard'
  xcrun -f metal >/dev/null 2>&1 \
    || die 'Xcode Metal compiler is unavailable'
}

require_tooling() {
  local tool
  for tool in cargo rustc git uv shasum awk find grep sed tee sysctl df xcrun \
    uname wc sort tr; do
    command -v "$tool" >/dev/null 2>&1 || die "required tool missing: $tool"
  done
  [[ -d "$VOKRA_ROOT/.git" && -f "$VOKRA_ROOT/Cargo.toml" && -f "$TEST_SOURCE" ]] \
    || die 'Vokra checkout or exact real test source is missing'
  [[ -f "$PARITY_PROJECT/pyproject.toml" && -f "$PARITY_PROJECT/uv.lock" && -f "$LICENSE_GATE" ]] \
    || die 'dedicated CosyVoice2 LLM gate inputs are missing'
  [[ -z "$(git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all)" ]] \
    || die 'remote Apple checkout must be clean'
}

require_reference() {
  local reference="$1"
  # Parse the complete bundle with Python stdlib before any model/Cargo work.
  # object_pairs_hook rejects duplicate keys; every identity and NPY header is
  # checked here as well as in the direct Rust test.
  UV_NO_CACHE=1 uv run --no-project --offline --python 3.12 python - "$reference" <<'PY'
import hashlib
import json
import math
import pathlib
import re
import struct
import sys

root = pathlib.Path(sys.argv[1])
EXPECTED_NAMES = {"diagnostics.json", "manifest.json", "token-ids.json", "true_hf_logits.npy"}
MODEL = {"repository": "FunAudioLLM/CosyVoice2-0.5B", "revision": "eec1ae6c79877dbd9379285cf8789c9e0879293d", "path": "llm.pt", "bytes": 2023316821, "sha256": "b144ef55b51ce8cfb79a73c90dbba0bdaba4e451c0ebcfab20f769264f84a608"}
QWEN = {"path": "CosyVoice-BlankEN/config.json", "bytes": 659, "sha256": "168aa1bd401abc3bc262ba15ba4e499627a8b4e006e9d050b47c22de20660185", "git_blob_sha1": "463b055262b6c66c4629a74a4b300bfe2ed31d3c", "verification": "ACQUIRED_AND_HASH_VERIFIED"}
QWEN_FIELDS = {"hidden_size": 896, "intermediate_size": 4864, "num_hidden_layers": 24, "num_attention_heads": 14, "num_key_value_heads": 2, "max_position_embeddings": 32768, "rope_theta": 1000000.0, "rms_norm_eps": 1e-6, "vocab_size": 151936, "tie_word_embeddings": True}
SOURCE = "https://github.com/FunAudioLLM/CosyVoice.git"
SOURCE_REV = "8555549e882236e6541748b1042d95693caa82ba"
ROLES = {
    "cosyvoice/cli/cosyvoice.py": ("cc443bed44c651a47492fc7e2142e3a88fb47627", "8e44f0f0144378561a00ebc065fdb15a843bc4650e68683bebb6624827731859", "CosyVoice2"),
    "cosyvoice/llm/llm.py": ("59ebd48fde1f1b69240391fdac6e2afc1035e123", "6439d57fcf78bcdcad6d31812f3f4b02bd34f513333711ee317d71d1fd14d2de", "Qwen2LM"),
    "cosyvoice/tokenizer/tokenizer.py": ("43fb39a2b543cc7ba4ec95fca9327596c34dcff0", "94340fc7cdf270c69a3aeb63290c5241044e20714e01fea736f361f9e5a56df2", "Qwen"),
}

def fail(msg):
    raise SystemExit(msg)

def unique(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate key {key}")
        out[key] = value
    return out

def load(path, label):
    try:
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique)
    except Exception as error:
        fail(f"{label} JSON: {error}")

def keys(obj, expected, label):
    if not isinstance(obj, dict) or set(obj) != set(expected):
        fail(f"{label} schema")

def regular(path, label):
    if path.is_symlink() or not path.is_file():
        fail(f"{label} is not a regular file")

def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

def npy_shape(path):
    data = path.read_bytes()
    if data[:6] != b"\x93NUMPY" or len(data) < 10:
        fail("bad NPY magic")
    version = data[6]
    if version == 1:
        start, size = 10, struct.unpack_from("<H", data, 8)[0]
    elif version in (2, 3):
        if len(data) < 12:
            fail("truncated NPY header")
        start, size = 12, struct.unpack_from("<I", data, 8)[0]
    else:
        fail("unsupported NPY version")
    end = start + size
    if end > len(data):
        fail("truncated NPY header")
    try:
        header = data[start:end].decode("ascii")
    except UnicodeDecodeError:
        fail("NPY header encoding")
    if header.count("'descr': '<f4'") != 1 or header.count("'fortran_order': False") != 1:
        fail("NPY dtype/order")
    if not re.fullmatch(r"\{\s*'descr':\s*'<f4',\s*'fortran_order':\s*False,\s*'shape':\s*\(\s*11,\s*151936\s*,?\s*\),?\s*\}\s*", header):
        fail("NPY shape/header")
    if len(data) - end != 11 * 151936 * 4:
        fail("NPY payload length")

if not root.is_absolute() or root.is_symlink() or not root.is_dir():
    fail("reference directory")
if set(path.name for path in root.iterdir()) != EXPECTED_NAMES:
    fail("reference artifact set")
for name in EXPECTED_NAMES:
    regular(root / name, f"reference {name}")
manifest_path = root / "manifest.json"
manifest = load(manifest_path, "reference manifest")
keys(manifest, ["format", "status", "component", "model", "qwen_config", "source", "state", "execution", "artifacts"], "reference manifest")
if (manifest["format"], manifest["status"], manifest["component"]) != ("vokra-cosyvoice2-llm-reference-v1", "REFERENCE_READY", "llm"):
    fail("reference identity")
keys(manifest["model"], MODEL, "model")
if manifest["model"] != MODEL:
    fail("model identity")
keys(manifest["qwen_config"], [*QWEN, "fields"], "Qwen config")
if {key: manifest["qwen_config"][key] for key in QWEN} != QWEN or manifest["qwen_config"]["fields"] != QWEN_FIELDS:
    fail("Qwen config identity")
source = manifest["source"]
keys(source, ["repository", "revision", "roles", "license"], "source")
if (source["repository"], source["revision"]) != (SOURCE, SOURCE_REV):
    fail("source identity")
keys(source["roles"], ROLES, "source roles")
for name, (blob, sha, marker) in ROLES.items():
    row = source["roles"][name]
    keys(row, ["blob_sha1", "sha256", "marker", "bytes"], "source role")
    if (row["blob_sha1"], row["sha256"], row["marker"]) != (blob, sha, marker) or type(row["bytes"]) is not int or row["bytes"] <= 0:
        fail("source role identity")
license = source["license"]
keys(license, ["path", "bytes", "sha256", "git_blob_sha1", "declared"], "source license")
if license != {"path": "LICENSE", "bytes": 11357, "sha256": "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4", "git_blob_sha1": "261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64", "declared": "Apache-2.0"}:
    fail("source license identity")
if manifest["state"] != {"tensor_count": 295, "manifest_sha256": "07cf10ae088c27a7c88e1c08fb231d00b01bba0c13f312a74d2fd4b35403bda2"}:
    fail("state identity")
if manifest["execution"] != {"implementation": "transformers.Qwen2ForCausalLM", "attention": "eager", "dtype": "F32", "threads": 1, "deterministic_algorithms": True, "model_execution": "RUN", "publication": "NO_UPLOAD"}:
    fail("execution identity")
artifacts = manifest["artifacts"]
keys(artifacts, ["true_hf_logits.npy", "token-ids.json", "diagnostics.json"], "artifacts")
for name in artifacts:
    row = artifacts[name]
    keys(row, ["bytes", "sha256"], f"artifact {name}")
    path = root / name
    if type(row["bytes"]) is not int or row["bytes"] <= 0 or row["bytes"] != path.stat().st_size or row["sha256"] != digest(path):
        fail(f"artifact identity {name}")
tokens = load(root / "token-ids.json", "token IDs")
keys(tokens, ["ids", "text", "provenance"], "token IDs")
if tokens != {"ids": [151643, 785, 3974, 13876, 38835, 34208, 916, 279, 15678, 5562, 13], "text": "The quick brown fox jumps over the lazy dog.", "provenance": "fixed documented ids; no tokenizer download"}:
    fail("token identity")
diag = load(root / "diagnostics.json", "diagnostics")
keys(diag, ["lm_head_vs_embed_max_abs_delta", "torch", "transformers", "numpy", "threads", "deterministic_algorithms", "attention_implementation", "state", "source"], "diagnostics")
if type(diag["lm_head_vs_embed_max_abs_delta"]) not in (int, float) or not math.isfinite(diag["lm_head_vs_embed_max_abs_delta"]):
    fail("diagnostic delta")
if any(not isinstance(diag[key], str) or not diag[key].strip() for key in ("torch", "transformers", "numpy")):
    fail("diagnostic versions")
if diag["threads"] != 1 or diag["deterministic_algorithms"] is not True or diag["attention_implementation"] != "eager" or diag["state"] != manifest["state"] or diag["source"] != source:
    fail("diagnostic linkage")
npy_shape(root / "true_hf_logits.npy")
PY
}

license_preflight() {
  local license_manifest="$1"
  require_file 'external owner-signed license manifest' "$license_manifest"
  # The project is Linux-reference-only, so run the standard-library gate as
  # an isolated uv invocation on Apple; --offline makes this check non-network.
  UV_NO_CACHE=1 uv run --no-project --offline --python 3.12 python "$LICENSE_GATE" \
    --project "$PARITY_PROJECT/pyproject.toml" \
    --lock "$PARITY_PROJECT/uv.lock" \
    --license-manifest "$license_manifest" >/dev/null \
    || die 'external owner-signed CosyVoice2 LLM license gate did not pass'
}

require_cargo_singleton() {
  local log_file="$1" test_count result_count test_lines
  test_count="$(grep -Ec "^test $TEST_NAME \.\.\. ok$" "$log_file" || true)"
  result_count="$(grep -Ec '^test result:' "$log_file" || true)"
  test_lines="$(grep -Ec '^test ' "$log_file" || true)"
  [[ "$test_count" == 1 && "$result_count" == 1 && "$test_lines" == 2 ]] \
    || { die 'Cargo output is not exactly one named test plus one result'; return 2; }
  grep -Eq '^test result: ok\. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out(; finished in [0-9]+(\.[0-9]+)?s)?$' "$log_file" \
    || { die 'Cargo output is not an exact singleton pass'; return 2; }
}

require_apple_sentinel() {
  local log_file="$1" line count key value
  local number='[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?'
  count="$(grep -Ec '^COSYVOICE2_LLM_APPLE_PARITY ' "$log_file" || true)"
  [[ "$count" == 1 ]] || { die 'Apple parity sentinel is missing or duplicated'; return 2; }
  line="$(grep -E '^COSYVOICE2_LLM_APPLE_PARITY ' "$log_file")"
  [[ "$line" =~ ^COSYVOICE2_LLM_APPLE_PARITY\ backend=cpu,metal\ metal_device=present\ cpu_reference_max_abs=$number\ metal_reference_max_abs=$number\ metal_cpu_max_abs=$number\ cpu_reference_mean_abs=$number\ metal_reference_mean_abs=$number\ metal_cpu_mean_abs=$number\ argmax_matches=$REFERENCE_ROWS/$REFERENCE_ROWS\ atol=$ATOL\ verdict=PASS$ ]] \
    || { die 'Apple parity sentinel has malformed fields or metrics'; return 2; }
  for key in cpu_reference_max_abs metal_reference_max_abs metal_cpu_max_abs \
    cpu_reference_mean_abs metal_reference_mean_abs metal_cpu_mean_abs; do
    value="$(printf '%s\n' "$line" | sed -n "s/.* $key=\([^ ]*\).*/\1/p")"
    awk -v value="$value" -v bound="$ATOL" 'BEGIN { if ((value + 0) != (value + 0) || value + 0 > bound) exit 1 }' \
      || { die "Apple parity metric exceeds ATOL: $key=$value"; return 2; }
  done
}

run_self_test() (
  local script="${BASH_SOURCE[0]}" temporary log_file fail=0
  temporary="$(mktemp -d "${TMPDIR:-/tmp}/vokra-cosyvoice2-llm-apple.XXXXXX")"
  trap 'rm -rf "$temporary"' EXIT

  # A model-free result fixture exercises the exact singleton, sentinel,
  # duplicate, malformed, and over-bound marker rejection paths.
  log_file="$temporary/result.log"
  printf '%s\n' \
    "test $TEST_NAME ... ok" \
    'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.25s' \
    'COSYVOICE2_LLM_APPLE_PARITY backend=cpu,metal metal_device=present cpu_reference_max_abs=1.0e-5 metal_reference_max_abs=2.0e-5 metal_cpu_max_abs=3.0e-5 cpu_reference_mean_abs=1.0e-6 metal_reference_mean_abs=2.0e-6 metal_cpu_mean_abs=3.0e-6 argmax_matches=11/11 atol=3e-4 verdict=PASS' > "$log_file"
  require_cargo_singleton "$log_file" || fail=1
  require_apple_sentinel "$log_file" || fail=1
  cat "$log_file" "$log_file" > "$temporary/duplicate.log"
  if require_cargo_singleton "$temporary/duplicate.log" >/dev/null 2>&1; then fail=1; fi
  if require_apple_sentinel "$temporary/duplicate.log" >/dev/null 2>&1; then fail=1; fi
  sed 's/metal_cpu_max_abs=3.0e-5/metal_cpu_max_abs=4.0e-4/' "$log_file" > "$temporary/over-bound.log"
  if require_apple_sentinel "$temporary/over-bound.log" >/dev/null 2>&1; then fail=1; fi
  sed 's/argmax_matches=11\/11/argmax_matches=10\/11/' "$log_file" > "$temporary/malformed.log"
  if require_apple_sentinel "$temporary/malformed.log" >/dev/null 2>&1; then fail=1; fi

  # Path and CLI guards are behavioral: equality and nesting are rejected,
  # duplicate options are rejected by the parser, and a lexical alias of the
  # checked-in manifest is still refused by canonical identity.
  if require_pairwise_disjoint "$temporary/equal" "$temporary/equal" >/dev/null 2>&1; then fail=1; fi
  if require_pairwise_disjoint "$temporary/nested" "$temporary/nested/child" >/dev/null 2>&1; then fail=1; fi
  if is_checked_in_license_manifest "$PARITY_PROJECT//license_gate_manifest.json"; then :; else fail=1; fi
  if "$script" --gguf /tmp/cosyvoice2-a --gguf /tmp/cosyvoice2-b >/dev/null 2>&1; then fail=1; fi
  if "$script" --license-manifest-sha256 "$(printf 'a%.0s' {1..64})" --license-manifest-sha256 "$(printf 'b%.0s' {1..64})" >/dev/null 2>&1; then fail=1; fi
  local malformed_reference="$temporary/reference"
  mkdir "$malformed_reference"
  printf '%s\n' '{"format": 1, "format": 2}' > "$malformed_reference/manifest.json"
  printf '%s\n' x > "$malformed_reference/token-ids.json"
  printf '%s\n' x > "$malformed_reference/true_hf_logits.npy"
  printf '%s\n' x > "$malformed_reference/diagnostics.json"
  if require_reference "$malformed_reference" >/dev/null 2>&1; then fail=1; fi

  # The preflight must precede the clean checkout check and any Cargo launch;
  # the worker must not contain network/publication/model acquisition verbs.
  local gate_line cargo_line
  gate_line="$(grep -nF "license_preflight \"\$license_manifest\"" "$script" | head -n1 | cut -d: -f1 || true)"
  cargo_line="$(grep -nE '^[[:space:]]+CARGO_NET_OFFLINE=true cargo test --offline --locked --release --features metal' "$script" | head -n1 | cut -d: -f1 || true)"
  [[ "$gate_line" =~ ^[0-9]+$ && "$cargo_line" =~ ^[0-9]+$ && "$gate_line" -lt "$cargo_line" ]] || fail=1
  if grep -En '(^|[[:space:];|])(curl|wget|git[[:space:]]+(clone|fetch|pull|push)|snapshot_download|huggingface-cli|upload\.sh|publish-one\.sh|--push)([[:space:]]|$)' "$script" | grep -vE 'grep -En|never downloads|No network'; then fail=1; fi
  for token in 'external owner-signed license manifest' '--license-manifest' '--no-project --offline' \
    'VOKRA_REMOTE_APPLE_SILICON=1' 'xcrun -f metal' 'metal_device=present' \
    'COSYVOICE2_LLM_APPLE_PARITY' 'verdict=PASS' 'CARGO_BUILD_JOBS=1' \
    '--license-manifest-sha256' \
    'CPU fallback' 'git status --porcelain' 'CARGO_NET_OFFLINE=true' \
    'require_pairwise_disjoint' 'dot path component' \
    'checked-in license manifest cannot be used'; do
    grep -Fq -- "$token" "$script" || fail=1
  done
  (( fail == 0 )) || return 1
  log 'self-test PASS'
)

main() {
  local self_test=0 gguf='' gguf_sha='' reference='' reference_sha='' license_manifest='' license_sha='' evidence='' key value seen=''
  while (( $# > 0 )); do
    key="$1"
    if [[ "$key" == --self-test ]]; then
      (( self_test == 0 )) || die 'duplicate --self-test'
      self_test=1; shift; continue
    fi
    case "$key" in
      --gguf) value=gguf;;
      --gguf-sha256) value=gguf_sha;;
      --reference) value=reference;;
      --reference-manifest-sha256) value=reference_sha;;
      --license-manifest) value=license_manifest;;
      --license-manifest-sha256) value=license_sha;;
      --evidence-dir) value=evidence;;
      -h|--help) [[ $# == 1 && $self_test == 0 ]] || die '--help cannot be combined'; usage; return 0;;
      *) usage; die "unknown argument: $key";;
    esac
    [[ "$seen" != *"|$key|"* ]] || die "duplicate argument: $key"
    seen+="|$key|"
    [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die "$key requires a non-empty value"
    printf -v "$value" '%s' "$2"
    shift 2
  done
  if (( self_test )); then
    [[ -z "$gguf$gguf_sha$reference$reference_sha$license_manifest$license_sha$evidence$seen" ]] \
      || die '--self-test accepts no other arguments'
    run_self_test
    return
  fi
  [[ -n "$gguf" && -n "$gguf_sha" && -n "$reference" && -n "$reference_sha" && -n "$license_manifest" && -n "$license_sha" && -n "$evidence" ]] \
    || { usage; return 2; }
  require_tooling
  require_abs "$gguf" GGUF; require_abs "$reference" reference; require_abs "$license_manifest" license-manifest; require_abs "$evidence" evidence
  if is_checked_in_license_manifest "$license_manifest"; then
    die 'the checked-in license manifest cannot be used as external approval'
  fi
  require_expected_sha256 'CosyVoice2 LLM GGUF' "$gguf_sha" "$gguf"
  require_expected_sha256 'reference manifest' "$reference_sha" "$reference/manifest.json"
  require_expected_sha256 'external owner-signed license manifest' "$license_sha" "$license_manifest"
  require_reference "$reference"
  require_pairwise_disjoint "$gguf" "$reference" "$license_manifest" "$evidence"
  require_absent_evidence_dir "$evidence" "$gguf" "$reference" "$license_manifest" "$TEST_SOURCE"
  # This is intentionally the first operation that can validate the owner
  # approval and it happens before any model binding or Cargo execution.
  license_preflight "$license_manifest"
  require_host
  (umask 077 && mkdir -m 700 "$evidence") \
    || die 'evidence directory creation raced or failed'
  local log_file="$evidence/parity.log" summary_file="$evidence/summary.txt"
  (set -o noclobber; : > "$log_file") \
    || die 'parity log already exists or cannot be created without clobbering'
  (set -o noclobber; : > "$summary_file") \
    || die 'evidence summary already exists or cannot be created without clobbering'
  env VOKRA_COSYVOICE2_LLM_COMPONENT_GGUF="$gguf" \
    VOKRA_COSYVOICE2_LLM_COMPONENT_GGUF_SHA256="$gguf_sha" \
    VOKRA_COSYVOICE2_LLM_LICENSE_MANIFEST="$license_manifest" \
    VOKRA_COSYVOICE2_LLM_LICENSE_MANIFEST_SHA256="$license_sha" \
    VOKRA_COSYVOICE2_LLM_REFERENCE="$reference" \
    VOKRA_COSYVOICE2_LLM_REFERENCE_MANIFEST_SHA256="$reference_sha" \
    VOKRA_REMOTE_APPLE_SILICON=1 CARGO_BUILD_JOBS=1 \
    CARGO_NET_OFFLINE=true cargo test --offline --locked --release --features metal -p vokra-models \
      --test parity_cosyvoice2_llm_component_real "$TEST_NAME" \
      -- --ignored --exact --nocapture 2>&1 | tee -a "$log_file"
  require_cargo_singleton "$log_file"
  require_apple_sentinel "$log_file"
  [[ -f "$log_file" && ! -L "$log_file" ]] || die 'parity log is missing or symlinked'
  printf '%s\n' \
    'format=vokra-cosyvoice2-llm-apple-evidence-v1' \
    'backend=cpu,metal' \
    'metal_device=present' \
    "gguf_sha256=$gguf_sha" \
    "reference_manifest_sha256=$reference_sha" \
    "license_manifest_sha256=$license_sha" \
    "git_commit=$(git -C "$VOKRA_ROOT" rev-parse HEAD)" \
    "test=$TEST_NAME" \
    "atol=$ATOL" \
    'publication=NO_UPLOAD' \
    'network=NOT_PERFORMED' \
    'verdict=PASS' >> "$summary_file"
  log 'CosyVoice2 LLM Apple CPU/Metal parity PASS'
}

main "$@"
