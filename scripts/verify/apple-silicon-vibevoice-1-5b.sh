#!/usr/bin/env bash
# Disposable Darwin arm64 VibeVoice measurement worker.
# Inputs are authenticated by VAST; this script never downloads, converts,
# publishes, uploads, or fabricates a parity result.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VOKRA_ROOT="${VOKRA_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
REFERENCE="$VOKRA_ROOT/tools/parity/vibevoice_1_5b_dump_reference.py"
HF_REPOSITORY="microsoft/VibeVoice-1.5B"
HF_REVISION="142f4a5dda029212cda8b118e9d99c3da27018d8"
QWEN_REPOSITORY="Qwen/Qwen2.5-1.5B"
QWEN_REVISION="8faed761d45a263340a0528343f099c05c9a4323"
SOURCE_REPOSITORY="https://github.com/microsoft/VibeVoice.git"
SOURCE_REVISION="2f9a3d79a0e51bd1cf2ab40d36884c8948e6bb9c"
TRANSFORMERS_REPOSITORY="https://github.com/huggingface/transformers.git"
TRANSFORMERS_REVISION="5f4ecf2d9f867a1255131d2461d75793c0cf1db2"
PUBLIC_REPOSITORY="vokra/vibevoice-1.5b"
PUBLIC_REVISION="dec190628f58928fc247b1205b9da2dabc58b9da"
REFERENCE_AUDIT_UV=(uv run --no-cache --no-project --offline --python 3.12 python)

die() { printf '[vibevoice-apple] ERROR: %s\n' "$*" >&2; exit 2; }
sha256_file() { shasum -a 256 "$1" | awk '{print $1}'; }

canonical_existing_path() {
  local path="$1" rest component current=/ parent base
  [[ "$path" == /* && -e "$path" && ! -L "$path" ]] || return 1
  rest="${path#/}"
  while [[ -n "$rest" ]]; do
    component="${rest%%/*}"; [[ "$rest" == "$component" ]] && rest="" || rest="${rest#*/}"
    [[ -n "$component" && "$component" != . && "$component" != .. ]] || return 1
    current="${current%/}/$component"; [[ ! -L "$current" ]] || return 1
  done
  if [[ -d "$path" ]]; then (cd -P "$path" && pwd); else
    parent="$(dirname "$path")"; base="$(basename "$path")"
    parent="$(cd -P "$parent" && pwd)" || return 1; printf '%s/%s\n' "$parent" "$base"
  fi
}

canonical_absent_path() {
  local path="$1" target="$1" rest component current=/ suffix='' real
  [[ "$path" == /* && ! -e "$path" && ! -L "$path" ]] || return 1
  rest="${path#/}"
  while [[ -n "$rest" ]]; do
    component="${rest%%/*}"; [[ "$rest" == "$component" ]] && rest="" || rest="${rest#*/}"
    [[ -n "$component" && "$component" != . && "$component" != .. ]] || return 1
    current="${current%/}/$component"; [[ ! -L "$current" ]] || return 1
  done
  while [[ ! -e "$target" && ! -L "$target" ]]; do
    component="$(basename "$target")"; suffix="/$component$suffix"; target="$(dirname "$target")"
  done
  [[ -d "$target" && ! -L "$target" ]] || return 1
  real="$(cd -P "$target" && pwd)" || return 1; printf '%s%s\n' "$real" "$suffix"
}

self_test() {
  local fail=0 token
  for token in Darwin arm64 VOKRA_REMOTE_APPLE_SILICON=1 VOKRA_VIBEVOICE_GGUF VOKRA_VIBEVOICE_REFERENCE_DIR \
    parity_vibevoice_1_5b_real VIBEVOICE_CPU_TOKENS_MEASURED VIBEVOICE_METAL_TOKENS_MEASURED \
    VIBEVOICE_CPU_OFFICIAL_DIFFUSION_LATENTS_CAPTURED VIBEVOICE_METAL_OFFICIAL_DIFFUSION_LATENTS_CAPTURED \
    exact=true MEASURED_NOT_GATED official_pcm.f32le packet.json vibevoice-apple-summary.json NO_UPLOAD apple-transfer-manifest.sha256 \
    reference_environment license_audit BLOCKED_UNREVIEWED_TRANSITIVE BLOCKED_UNVERIFIED_API_SMOKE GHSA-xrqw-3rrv-vx5w AUTHENTICATED_CLEAR --license-audit --no-project package-resolution-and-dependency-markers-v2 1ea002fe37f4ddc4df9f7535b5ae3a42661fc1eaa0a28e8ae6dbba0fa7e9649b 987a1f7204c2d7f2baa1c537ebaa06ca4bc872d2aae60f25a78393967da7bf8c ba80c08b17b2d04356264b9f9d42393e9c8be66bc0cd9fda6139dc007d943909 \
    'VOKRA_VIBEVOICE_BACKEND=cpu' 'VOKRA_VIBEVOICE_BACKEND=metal' '--ignored --exact --nocapture' '--expected-head' '--approval-evidence' '--approval-evidence-sha256' '--transfer-manifest' '--transfer-manifest-sha256' '--bundle' '--evidence-dir' '--offline' '--test-threads=1' 'vibevoice-apple-transfer-v1'; do
    grep -Fq -- "$token" "$0" || { printf 'self-test missing %s\n' "$token" >&2; fail=1; }
  done
  if grep -En '(^|[;&|][[:space:]]*)git[[:space:]]+push|(^|[;&|][[:space:]]*)(curl|wget|snapshot_download)[[:space:]]' "$0" >/dev/null; then
    printf 'self-test found download/publication command\n' >&2; fail=1
  fi
  local gate_line line
  gate_line="$(awk '/^[[:space:]]+license_audit_preflight$/{print NR; exit}' "$0")"
  [[ "$gate_line" =~ ^[0-9]+$ ]] || { printf 'self-test cannot locate license gate\n' >&2; fail=1; }
  for token in 'local selector=' 'cargo test --manifest-path'; do
    line="$(awk -v gate="$gate_line" -v token="$token" 'NR > gate && index($0, token) {print NR; exit}' "$0")"
    [[ "$line" =~ ^[0-9]+$ && "$line" -gt "$gate_line" ]] || { printf 'self-test operation precedes license gate: %s\n' "$token" >&2; fail=1; }
  done
  for token in 'require_transfer_manifest ' 'require_bundle '; do
    line="$(awk -v gate="$gate_line" -v token="$token" 'NR > gate && index($0, token) {print NR; exit}' "$0")"
    [[ "$line" =~ ^[0-9]+$ && "$line" -gt "$gate_line" ]] || { printf 'self-test bundle authentication is not after factual license gate: %s\n' "$token" >&2; fail=1; }
  done
  grep -Fq -- 'REFERENCE_AUDIT_UV=(uv run --no-cache --no-project --offline --python 3.12 python)' "$0" || { printf 'self-test missing no-cache audit command\n' >&2; fail=1; }
  if grep -En '(^|[[:space:]])uv[[:space:]]+sync([[:space:]]|$)' "$0" >/dev/null; then
    printf 'self-test found an implicit uv sync\n' >&2; fail=1
  fi
  if "$0" --self-test --self-test >/dev/null 2>&1 || \
    "$0" --expected-head bad >/dev/null 2>&1 || \
    "$0" --expected-head "$(printf '0%.0s' {1..40})" --expected-head "$(printf '1%.0s' {1..40})" >/dev/null 2>&1 || \
    "$0" --approval-evidence one --approval-evidence two >/dev/null 2>&1 || \
    "$0" --transfer-manifest-sha256 bad >/dev/null 2>&1; then
    printf 'self-test accepted malformed or duplicate CLI option\n' >&2; fail=1
  fi
  if ! UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python - "$0" <<'PY'
import re, sys
source = open(sys.argv[1], encoding="utf-8").read()
blocks = re.findall(r"<<'PY'\n(.*?)\nPY", source, re.S)
if not blocks:
    raise SystemExit("no inline Python blocks found")
for index, block in enumerate(blocks):
    compile(block, f"<inline-python-{index}>", "exec")
PY
  then
    printf 'inline Python syntax validation failed\n' >&2; fail=1
  fi
  if ! (
    set -euo pipefail
    fixture="$(mktemp -d "${TMPDIR:-/tmp}/vibevoice-apple-transfer.XXXXXX")"
    cleanup_fixture() { rm -r -- "$fixture"; }
    trap cleanup_fixture EXIT
    names=(manifest.json inspection-manifest.json token_ids.u32le prompt_pcm.f32le prompt_latent.f32le diffusion_initial.f32le diffusion_initial_native.f32le speech_input_mask.u8 speech_masks.u8 speech_replacement_positions.u32le generated_tokens.u32le guidance-scale.txt max-generated-tokens.txt official_pcm.f32le official_diffusion_latents.f32le packet.json public-artifact.json artifact-sha256.txt reference-sha256.txt native-cpu.log vibevoice-1.5b.gguf)
    for name in "${names[@]}"; do printf x > "$fixture/$name"; done
    fixture_head="$(printf '0%.0s' {1..40})"
    fixture_approval="$(printf '1%.0s' {1..64})"
    UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python - "$fixture" "$fixture_head" "$fixture_approval" <<'PY'
import hashlib, json, pathlib, sys
root, expected_head, approval_sha = map(pathlib.Path, sys.argv[1:])
expected_head, approval_sha = str(expected_head), str(approval_sha)
def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()
names = sorted(path.name for path in root.iterdir())
rows = [{"path": name, "bytes": (root / name).stat().st_size, "sha256": digest(root / name)} for name in names]
payload = {
    "schema": "vibevoice-apple-transfer-v1", "expected_head": expected_head,
    "approval_evidence_sha256": approval_sha, "status": "MEASURED_NOT_GATED", "publication": "NO_UPLOAD",
    "reference_manifest_sha256": next(row["sha256"] for row in rows if row["path"] == "manifest.json"),
    "input_packet_sha256": next(row["sha256"] for row in rows if row["path"] == "packet.json"),
    "gguf_sha256": next(row["sha256"] for row in rows if row["path"] == "vibevoice-1.5b.gguf"),
    "native_cpu_log_sha256": next(row["sha256"] for row in rows if row["path"] == "native-cpu.log"),
    "files": rows,
}
(root / "apple-transfer-manifest.json").write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
PY
    (cd "$fixture" && shasum -a 256 apple-transfer-manifest.json > apple-transfer-manifest.sha256)
    printf x > "$fixture/apple-transfer-args.txt"
    require_transfer_manifest "$fixture" "$fixture/apple-transfer-manifest.json" "$(sha256_file "$fixture/apple-transfer-manifest.json")" "$fixture_head" "$fixture_approval"
  ); then
    printf 'transfer producer/consumer closure self-test failed\n' >&2; fail=1
  fi
  (( fail == 0 )) && printf '[vibevoice-apple] self-test: OK\n' || return 1
}

usage() {
  cat <<'EOF'
usage: apple-silicon-vibevoice-1-5b.sh --expected-head HEX40 \
  --approval-evidence FILE --approval-evidence-sha256 HEX64 \
  --transfer-manifest FILE --transfer-manifest-sha256 HEX64 \
  --bundle BUNDLE --evidence-dir ABSENT_DIR
       apple-silicon-vibevoice-1-5b.sh --self-test
EOF
}

validate_approval() {
  local path="$1" expected_head="$2" supplied_sha="$3"
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python - "$path" "$expected_head" "$supplied_sha" "$HF_REPOSITORY" "$HF_REVISION" "$QWEN_REPOSITORY" "$QWEN_REVISION" "$SOURCE_REPOSITORY" "$SOURCE_REVISION" "$TRANSFORMERS_REPOSITORY" "$TRANSFORMERS_REVISION" "$PUBLIC_REPOSITORY" "$PUBLIC_REVISION" <<'PY'
import hashlib, json, pathlib, sys
path, expected_head, supplied_sha, hf_repo, hf_rev, qwen_repo, qwen_rev, source_repo, source_rev, transformers_repo, transformers_rev, public_repo, public_rev = sys.argv[1:]
raw = pathlib.Path(path).read_bytes()
if hashlib.sha256(raw).hexdigest() != supplied_sha:
    raise SystemExit("approval bytes changed")
def pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("duplicate approval key: " + key)
        result[key] = value
    return result
data = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs)
keys = {"schema", "status", "disposition", "expected_head", "hf_repository", "hf_revision", "qwen_repository", "qwen_revision", "source_repository", "source_revision", "transformers_repository", "transformers_revision", "public_repository", "public_revision", "no_upload", "scope_sha256"}
if set(data) != keys:
    raise ValueError("approval schema is not exact")
expected = {"schema": "vokra-vibevoice-1-5b-approval-v1", "status": "BLOCKED", "disposition": "INSPECTION_ONLY", "expected_head": expected_head, "hf_repository": hf_repo, "hf_revision": hf_rev, "qwen_repository": qwen_repo, "qwen_revision": qwen_rev, "source_repository": source_repo, "source_revision": source_rev, "transformers_repository": transformers_repo, "transformers_revision": transformers_rev, "public_repository": public_repo, "public_revision": public_rev, "no_upload": True}
if any(data[key] != value for key, value in expected.items()):
    raise ValueError("approval identity/status mismatch")
scope = {key: data[key] for key in ("disposition", "expected_head", "hf_repository", "hf_revision", "no_upload", "public_repository", "public_revision", "qwen_repository", "qwen_revision", "source_repository", "source_revision", "status", "transformers_repository", "transformers_revision")}
if data["scope_sha256"] != hashlib.sha256(json.dumps(scope, sort_keys=True, separators=(",", ":")).encode()).hexdigest():
    raise ValueError("approval scope mismatch")
PY
}

require_transfer_manifest() {
  local bundle="$1" path="$2" expected_sha="$3" expected_head="$4" approval_sha="$5"
  [[ -f "$path" && ! -L "$path" && -s "$path" ]] || die 'transfer manifest is missing or symlinked'
  [[ "$(sha256_file "$path")" == "$expected_sha" ]] || die 'transfer manifest SHA-256 mismatch'
  [[ -f "$bundle/apple-transfer-manifest.sha256" && ! -L "$bundle/apple-transfer-manifest.sha256" ]] || die 'transfer manifest sidecar is missing or symlinked'
  [[ "$(<"$bundle/apple-transfer-manifest.sha256")" == "$expected_sha  apple-transfer-manifest.json" ]] || die 'transfer manifest sidecar mismatch'
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python - "$bundle" "$path" "$expected_head" "$approval_sha" <<'PY'
import hashlib, json, pathlib, sys
bundle, manifest_path, expected_head, approval_sha = sys.argv[1:]
bundle, manifest_path = pathlib.Path(bundle), pathlib.Path(manifest_path)
def pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("duplicate transfer-manifest key: " + key)
        result[key] = value
    return result
manifest = json.loads(manifest_path.read_text(encoding="utf-8"), object_pairs_hook=pairs)
required = {"schema", "expected_head", "approval_evidence_sha256", "status", "publication", "reference_manifest_sha256", "input_packet_sha256", "gguf_sha256", "native_cpu_log_sha256", "files"}
if set(manifest) != required or manifest["schema"] != "vibevoice-apple-transfer-v1" or manifest["expected_head"] != expected_head or manifest["approval_evidence_sha256"] != approval_sha or manifest["status"] != "MEASURED_NOT_GATED" or manifest["publication"] != "NO_UPLOAD":
    raise SystemExit("transfer manifest identity/status mismatch")
names = {"manifest.json", "inspection-manifest.json", "token_ids.u32le", "prompt_pcm.f32le", "prompt_latent.f32le", "diffusion_initial.f32le", "diffusion_initial_native.f32le", "speech_input_mask.u8", "speech_masks.u8", "speech_replacement_positions.u32le", "generated_tokens.u32le", "guidance-scale.txt", "max-generated-tokens.txt", "official_pcm.f32le", "official_diffusion_latents.f32le", "packet.json", "public-artifact.json", "artifact-sha256.txt", "reference-sha256.txt", "native-cpu.log", "vibevoice-1.5b.gguf"}
rows = manifest["files"]
if not isinstance(rows, list) or {row.get("path") for row in rows if isinstance(row, dict)} != names or len(rows) != len(names):
    raise SystemExit("transfer manifest file closure is not exact")
def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()
seen = set()
for row in rows:
    if not isinstance(row, dict) or set(row) != {"path", "bytes", "sha256"} or row["path"] in seen or row["path"] not in names or not isinstance(row["bytes"], int) or isinstance(row["bytes"], bool) or not isinstance(row["sha256"], str) or len(row["sha256"]) != 64 or any(char not in "0123456789abcdef" for char in row["sha256"]):
        raise SystemExit("transfer manifest row is malformed")
    seen.add(row["path"])
    path = bundle / row["path"]
    if not path.is_file() or path.is_symlink() or path.stat().st_size != row["bytes"] or digest(path) != row["sha256"]:
        raise SystemExit(f"transfer artifact hash mismatch: {row['path']}")
top = {entry.name for entry in bundle.iterdir()}
if top != names | {"apple-transfer-manifest.json", "apple-transfer-manifest.sha256", "apple-transfer-args.txt"}:
    raise SystemExit("bundle top-level closure is not exact")
by_name = {row["path"]: row["sha256"] for row in rows}
if manifest["reference_manifest_sha256"] != by_name["manifest.json"] or manifest["input_packet_sha256"] != by_name["packet.json"] or manifest["gguf_sha256"] != by_name["vibevoice-1.5b.gguf"] or manifest["native_cpu_log_sha256"] != by_name["native-cpu.log"]:
    raise SystemExit("transfer manifest summary hashes mismatch")
PY
}

require_absent_evidence() {
  local evidence="$1"; shift
  local candidate root_real protected protected_real
  candidate="$(canonical_absent_path "$evidence")" || die 'evidence directory has unsafe ancestry'
  root_real="$(canonical_existing_path "$VOKRA_ROOT")" || die 'checkout path is unsafe'
  for protected in "$@"; do
    protected_real="$(canonical_existing_path "$protected")" || die 'protected input path is unsafe'
    [[ "$candidate" != "$protected_real" && "$candidate" != "$protected_real"/* && "$protected_real" != "$candidate"/* ]] || die 'evidence directory overlaps a protected input'
  done
  [[ "$candidate" != "$root_real" && "$candidate" != "$root_real"/* && "$root_real" != "$candidate"/* ]] || die 'evidence directory overlaps checkout'
  mkdir "$evidence" || die 'evidence directory was concurrently claimed'
}

require_cargo_singleton() {
  local log_file="$1" backend="$2" named result tests marker
  named="$(grep -Ec '^test vibevoice_1_5b_real_cpu_matches_official_reference \.\.\. ok$' "$log_file" || true)"
  result="$(grep -Ec '^test result:' "$log_file" || true)"
  tests="$(grep -Ec '^test ' "$log_file" || true)"
  (( named == 1 && result == 1 && tests - result == 1 )) || die "$backend Cargo output is not one exact named test/result"
  grep -Eq '^test result: ok\. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out(; finished in [0-9]+(\.[0-9]+)?s)?$' "$log_file" || die "$backend Cargo result is not an exact singleton pass"
  marker="VIBEVOICE_${backend}_TOKENS_MEASURED exact=true"
  [[ "$(grep -Fc "$marker" "$log_file" || true)" == 1 ]] || die "$backend token marker is not singleton"
  [[ "$(grep -Fc "VIBEVOICE_${backend}_PCM_MEASURED" "$log_file" || true)" == 1 ]] || die "$backend PCM marker is not singleton"
  [[ "$(grep -Fc "VIBEVOICE_${backend}_OFFICIAL_DIFFUSION_LATENTS_CAPTURED" "$log_file" || true)" == 1 ]] || die "$backend diffusion marker is not singleton"
}

require_bundle() {
  local bundle="$1"
  [[ -d "$bundle" ]] || die "bundle missing: $bundle"
  for file in manifest.json inspection-manifest.json token_ids.u32le prompt_pcm.f32le \
    prompt_latent.f32le diffusion_initial.f32le diffusion_initial_native.f32le \
    speech_input_mask.u8 speech_masks.u8 \
    speech_replacement_positions.u32le generated_tokens.u32le official_pcm.f32le \
    official_diffusion_latents.f32le \
    packet.json vibevoice-1.5b.gguf; do
    [[ -f "$bundle/$file" && ! -L "$bundle/$file" ]] || die "bundle input missing or symlinked: $file"
  done
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python - "$bundle/manifest.json" "$bundle/inspection-manifest.json" "$bundle/packet.json" <<'PY'
import hashlib, json, sys
for name in sys.argv[1:]:
    if name.endswith("packet.json"):
        continue
    data = json.loads(open(name, encoding="utf-8").read())
    required = {
        "status": "BLOCKED",
        "evidence_stage": "INSPECTION_ONLY",
        "runtime_status": "NOT_IMPLEMENTED_FAIL_CLOSED",
        "metal_status": "BLOCKED_BY_CPU",
        "publication": "NO_UPLOAD",
    }
    for key, expected in required.items():
        if data.get(key) != expected:
            raise SystemExit(f"{name}: {key} is not fail-closed")
    expected_cpu = "MEASURED_NOT_GATED" if name == sys.argv[1] else "UNSUPPORTED"
    if data.get("cpu_status") != expected_cpu:
        raise SystemExit(f"{name}: unexpected CPU validation status")
    expected_parity = "MEASURED_NOT_GATED" if name == sys.argv[1] else "NOT_RUN"
    if data.get("parity_status") != expected_parity:
        raise SystemExit(f"{name}: unexpected parity validation status")
    if name == sys.argv[2] and data.get("inspection_status") != "AUTHENTICATED_EVIDENCE_COMPLETE":
        raise SystemExit(f"{name}: inspection evidence is not complete")
    if name == sys.argv[1] and data.get("inspection_status") != "AUTHENTICATED_EVIDENCE_COMPLETE":
        raise SystemExit(f"{name}: combined manifest lost inspection evidence")
    if name == sys.argv[1] and data.get("reference_status") != "REFERENCE_EVIDENCE_COMPLETE":
        raise SystemExit(f"{name}: official reference evidence is not complete")
    if name == sys.argv[1] and data.get("validation_status") != "CPU_NATIVE_REFERENCE_EXECUTED":
        raise SystemExit(f"{name}: native CPU validation is not recorded")
    if data.get("upstream", {}).get("revision") != "142f4a5dda029212cda8b118e9d99c3da27018d8":
        raise SystemExit(f"{name}: fixed HF revision mismatch")
    if name == sys.argv[1] and data.get("source", {}).get("revision") != "2f9a3d79a0e51bd1cf2ab40d36884c8948e6bb9c":
        raise SystemExit(f"{name}: fixed source revision mismatch")
    if data.get("reference_status") == "REFERENCE_ERROR":
        raise SystemExit(f"{name}: official reference failed")
combined = json.loads(open(sys.argv[1], encoding="utf-8").read())
environment = combined.get("reference_environment")
if not isinstance(environment, dict):
    raise SystemExit("combined manifest is missing reference environment identity")
lock = environment.get("lock")
audit = environment.get("license_audit")
if not isinstance(lock, dict) or lock.get("sha256") != "ba80c08b17b2d04356264b9f9d42393e9c8be66bc0cd9fda6139dc007d943909":
    raise SystemExit("combined manifest has an unreviewed VibeVoice lock")
if lock.get("package_rows_schema") != "package-resolution-and-dependency-markers-v2" or lock.get("package_rows_sha256") != "1ea002fe37f4ddc4df9f7535b5ae3a42661fc1eaa0a28e8ae6dbba0fa7e9649b":
    raise SystemExit("combined manifest has unreviewed VibeVoice dependency qualifiers")
if not isinstance(audit, dict) or audit.get("status") != "AUTHENTICATED_CLEAR":
    raise SystemExit("combined manifest has an unexpected VibeVoice license status")
if audit.get("license_audit_rows_sha256") != "987a1f7204c2d7f2baa1c537ebaa06ca4bc872d2aae60f25a78393967da7bf8c":
    raise SystemExit("combined manifest has unreviewed VibeVoice license rows")
security = environment.get("transformers_security")
if not isinstance(security, dict) or security.get("transformers_security_advisory") != "GHSA-xrqw-3rrv-vx5w" or security.get("transformers_security_patched_minimum") != "5.10.0" or security.get("isolated_transformers_pin") != "transformers==5.10.4" or security.get("transformers_compatibility_status") != "AUTHENTICATED_API_SMOKE":
    raise SystemExit("combined manifest has an unverified Transformers security closure")
packet_hash = hashlib.sha256(open(sys.argv[3], "rb").read()).hexdigest()
if combined.get("input_packet_sha256") != packet_hash:
    raise SystemExit("caller-owned packet hash mismatch")
PY
}

license_audit_preflight() {
  local audit_output audit_rc
  [[ -f "$REFERENCE" ]] || die 'VibeVoice reference gate is missing'
  set +e
  audit_output="$("${REFERENCE_AUDIT_UV[@]}" "$REFERENCE" --license-audit 2>&1)"
  audit_rc=$?
  set -e
  if [[ "$audit_rc" == 2 ]]; then
    printf '%s\n' "$audit_output" >&2
    die 'dependency/license gate is unresolved; no Apple bundle or Cargo execution is permitted'
  fi
  [[ "$audit_rc" == 0 ]] || die "license audit command returned $audit_rc"
  [[ "$audit_output" == *"AUTHENTICATED_CLEAR"* ]] || die 'license audit did not return authenticated clearance'
  [[ "$audit_output" == *"ba80c08b17b2d04356264b9f9d42393e9c8be66bc0cd9fda6139dc007d943909"* ]] || die 'license audit lock identity is missing'
}

parse_args() {
  self=0; expected_head=''; approval=''; approval_sha=''; transfer=''; transfer_sha=''; bundle=''; evidence=''; seen=''; self_seen=0
  while (( $# )); do
    key="$1"
    if [[ "$key" == --self-test ]]; then (( self_seen == 0 )) || die 'duplicate --self-test'; self_seen=1; self=1; shift; continue; fi
    case "$key" in
      --expected-head) value_name=expected_head;; --approval-evidence) value_name=approval;; --approval-evidence-sha256) value_name=approval_sha;;
      --transfer-manifest) value_name=transfer;; --transfer-manifest-sha256) value_name=transfer_sha;; --bundle) value_name=bundle;; --evidence-dir) value_name=evidence;;
      -h|--help) [[ $# == 1 && $self == 0 ]] || die '--help cannot be combined'; usage; exit 0;; *) die "unknown argument: $key";;
    esac
    [[ " $seen " != *" $value_name "* ]] || die "duplicate $key"
    [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die "$key requires a nonempty value"
    printf -v "$value_name" '%s' "$2"; seen="$seen $value_name"; shift 2
  done
}

main() {
  parse_args "$@"
  if (( self )); then [[ -z "$seen" ]] || die '--self-test accepts no other arguments'; self_test; return 0; fi
  [[ "$seen" == *' expected_head '* && "$seen" == *' approval '* && "$seen" == *' approval_sha '* && "$seen" == *' transfer '* && "$seen" == *' transfer_sha '* && "$seen" == *' bundle '* && "$seen" == *' evidence '* ]] || { usage; die 'all approval, transfer, bundle, and evidence arguments are required'; }
  [[ "$expected_head" =~ ^[0-9a-f]{40}$ ]] || die '--expected-head must be lowercase 40-hex'
  [[ "$approval_sha" =~ ^[0-9a-f]{64}$ && "$transfer_sha" =~ ^[0-9a-f]{64}$ ]] || die 'approval and transfer SHA must be lowercase 64-hex'
  [[ "$approval" == /* && "$transfer" == /* && "$bundle" == /* && "$evidence" == /* ]] || die 'all paths must be absolute'
  [[ -f "$approval" && ! -L "$approval" && -s "$approval" ]] || die 'approval evidence is missing, empty, or symlinked'
  [[ -d "$bundle" && ! -L "$bundle" ]] || die 'bundle must be a regular directory'
  canonical_existing_path "$approval" >/dev/null || die 'approval path has unsafe ancestry'
  canonical_existing_path "$bundle" >/dev/null || die 'bundle path has unsafe ancestry'
  [[ "$(canonical_existing_path "$transfer")" == "$(canonical_existing_path "$bundle")/apple-transfer-manifest.json" ]] || die 'transfer manifest must be the bundle-owned manifest'
  [[ "$(sha256_file "$approval")" == "$approval_sha" ]] || die 'approval evidence SHA-256 mismatch'
  validate_approval "$approval" "$expected_head" "$approval_sha"
  [[ -d "$VOKRA_ROOT/.git" && -z "$(git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all)" ]] || die 'clean checkout required'
  [[ "$(git -C "$VOKRA_ROOT" rev-parse HEAD)" == "$expected_head" ]] || die 'checkout HEAD differs from --expected-head'
  # This is deliberately before transfer/GGUF hashing and before any host or
  # Cargo work. The current approval is inspection-only, not execution assent.
  license_audit_preflight
  die 'BLOCKED_APPROVAL/INSPECTION_ONLY: this approval cannot authorize VibeVoice Apple execution'
  require_transfer_manifest "$bundle" "$transfer" "$transfer_sha" "$expected_head" "$approval_sha"
  require_bundle "$bundle"
  [[ "${VOKRA_REMOTE_APPLE_SILICON:-0}" == 1 ]] || die 'VOKRA_REMOTE_APPLE_SILICON=1 is required'
  [[ "$(uname -s)" == Darwin && "$(uname -m)" == arm64 ]] || die 'disposable Darwin arm64 is required'
  local mem
  mem="$(sysctl -n hw.memsize 2>/dev/null || true)"
  [[ "$mem" =~ ^[0-9]+$ && "$mem" -ge 34359738368 ]] || die 'at least 32 GiB RAM is required'
  [[ -d "$VOKRA_ROOT" && -z "$(git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all)" ]] || die 'clean checkout required'
  command -v uv >/dev/null 2>&1 || die 'uv is required'
  command -v cargo >/dev/null 2>&1 || die 'cargo is required'
  command -v xcrun >/dev/null 2>&1 || die 'xcrun is required'
  xcrun -sdk macosx metal -v >/dev/null 2>&1 || die 'Metal compiler unavailable'
  require_absent_evidence "$evidence" "$bundle" "$approval" "$transfer"
  local reference_dir="$evidence/reference" reference_file
  mkdir "$reference_dir" || die 'reference staging directory was concurrently claimed'
  for reference_file in manifest.json packet.json token_ids.u32le prompt_pcm.f32le prompt_latent.f32le diffusion_initial.f32le diffusion_initial_native.f32le speech_input_mask.u8 speech_masks.u8 speech_replacement_positions.u32le generated_tokens.u32le guidance-scale.txt max-generated-tokens.txt official_pcm.f32le official_diffusion_latents.f32le; do
    cp -- "$bundle/$reference_file" "$reference_dir/$reference_file"
  done
  local selector='vibevoice_1_5b_real_cpu_matches_official_reference'
  local cpu_log="$evidence/vibevoice-cpu.log" metal_log="$evidence/vibevoice-metal.log"
  VOKRA_VIBEVOICE_GGUF="$bundle/vibevoice-1.5b.gguf" \
    VOKRA_VIBEVOICE_REFERENCE_DIR="$reference_dir" \
    VOKRA_VIBEVOICE_BACKEND=cpu CARGO_BUILD_JOBS=1 CARGO_NET_OFFLINE=true \
    cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml" --offline --locked --release --features metal \
      -p vokra-models --test parity_vibevoice_1_5b_real "$selector" -- --ignored --exact --nocapture --test-threads=1 \
      2>&1 | tee "$cpu_log"
  require_cargo_singleton "$cpu_log" CPU
  VOKRA_VIBEVOICE_GGUF="$bundle/vibevoice-1.5b.gguf" \
    VOKRA_VIBEVOICE_REFERENCE_DIR="$reference_dir" \
    VOKRA_VIBEVOICE_BACKEND=metal CARGO_BUILD_JOBS=1 CARGO_NET_OFFLINE=true \
    cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml" --offline --locked --release --features metal \
      -p vokra-models --test parity_vibevoice_1_5b_real "$selector" -- --ignored --exact --nocapture --test-threads=1 \
      2>&1 | tee "$metal_log"
  require_cargo_singleton "$metal_log" METAL
  [[ -z "$(git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all)" && "$(git -C "$VOKRA_ROOT" rev-parse HEAD)" == "$expected_head" ]] || die 'checkout changed before Apple summary'
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python - "$evidence/vibevoice-apple-summary.json" "$cpu_log" "$metal_log" <<'PY'
import hashlib
import json
import sys

summary, cpu, metal = sys.argv[1:]

def digest(path):
    value = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()

with open(summary, "w", encoding="utf-8") as stream:
    json.dump({
        "status": "BLOCKED",
        "evidence_stage": "INSPECTION_ONLY",
        "cpu_status": "MEASURED_NOT_GATED",
        "metal_status": "MEASURED_NOT_GATED",
        "parity_status": "MEASURED_NOT_GATED",
        "publication": "NO_UPLOAD",
        "cpu_log_sha256": digest(cpu),
        "metal_log_sha256": digest(metal),
    }, stream, indent=2, sort_keys=True)
    stream.write("\n")
PY
  printf '[vibevoice-apple] CPU and Metal executions completed; PCM remains MEASURED_NOT_GATED; no upload.\n' >&2
  return 2
}

main "$@"
