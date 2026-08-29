#!/usr/bin/env bash
set -euo pipefail

# Two-phase, no-upload GigaAM v3 worker.  Measure authenticates the fixed
# remote source and records prepared/reference digests.  Parity is unavailable
# until the Sol-reviewed prepared SHA is written into the Rust constant.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
die() { echo "gigaam-v3-vast: BLOCKED: $*" >&2; exit 2; }
require_tool() { command -v "$1" >/dev/null 2>&1 || die "required tool missing: $1"; }
reject_symlink_ancestry() {
  local candidate="$1"
  [[ "$candidate" == /* ]] || die "path must be absolute: $candidate"
  while [[ "$candidate" != "/" ]]; do
    [[ ! -L "$candidate" ]] || die "symlink ancestry forbidden: $candidate"
    candidate="$(dirname "$candidate")"
  done
}
disjoint() {
  local left="$1" right="$2"
  case "$left" in "$right"|"$right"/*) die "path overlap: $left / $right";; esac
  case "$right" in "$left"|"$left"/*) die "path overlap: $left / $right";; esac
}

validate_parity_log() {
  local log_file="$1"
  [[ -f "$log_file" ]] || return 1
  UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/vokra-gigaam-uv-cache}" uv run --no-project \
    --python 3.12 python - "$log_file" <<'PY'
import re
import sys
from pathlib import Path

lines = Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()
target_prefix = "test real_gigaam_v3_cpu_trace_matches_official ... "
target_lines = [line for line in lines if line.startswith(target_prefix)]
if len(target_lines) != 1:
    raise SystemExit("parity log must contain exactly one target test start")
target_line = target_lines[0]

test_lines = [line for line in lines if line.startswith("test ")]
if len(test_lines) != 2 or sum(line.startswith("test result: ") for line in test_lines) != 1:
    raise SystemExit("parity log contains an unexpected named test line")

if lines.count("ok") != 1:
    raise SystemExit("parity log must contain one isolated test completion")

summary_re = re.compile(
    r"^test result: ok\. 1 passed; 0 failed; 0 ignored; 0 measured; "
    r"([0-9]+) filtered out; finished in [0-9]+(?:\.[0-9]+)?s$"
)
summaries = [summary_re.fullmatch(line) for line in lines if line.startswith("test result: ")]
if len(summaries) != 1 or summaries[0] is None:
    raise SystemExit("parity log result is not an exact one-pass summary")
running_re = re.compile(r"^running 1 test$")
running = [running_re.fullmatch(line) for line in lines if line.startswith("running ")]
if len(running) != 1:
    raise SystemExit("parity log must contain exactly one running 1 test line")

markers = []
for line in lines:
    marker = "GIGAAM_V3_PARITY"
    if marker in line:
        markers.append(line[line.index(marker):])

metric_re = re.compile(
    r"GIGAAM_V3_PARITY (log_mel|encoded|rnnt_logits) "
    r"max_abs=[0-9]+(?:\.[0-9]+)?e[+-][0-9]+ "
    r"mean_abs=[0-9]+(?:\.[0-9]+)?e[+-][0-9]+"
)
if len(markers) != 4:
    raise SystemExit("parity log must contain exactly four GigaAM markers")
metrics = []
for marker in markers:
    if marker == "GIGAAM_V3_PARITY CPU PASS; Metal OPEN_UNSUPPORTED; publication NO_UPLOAD":
        continue
    match = metric_re.fullmatch(marker)
    if match is None:
        raise SystemExit("parity log contains a malformed or spoofed GigaAM marker")
    metrics.append(match.group(1))
if sorted(metrics) != ["encoded", "log_mel", "rnnt_logits"]:
    raise SystemExit("parity log metric markers are missing or duplicated")

if not target_line.startswith(target_prefix + "GIGAAM_V3_PARITY log_mel "):
    raise SystemExit("target test start is not interleaved with its first metric")
PY
}

if [[ "${1:-}" == --self-test ]]; then
  [[ $# == 1 ]] || die "--self-test accepts no arguments"
  for path in \
    tools/parity/sber_gigaam_v3_dump_reference.py \
    tools/parity/sber_gigaam_v3_prepare_checkpoint.py \
    tools/parity/gigaam_v3_validation.py \
    crates/vokra-convert/src/models/sber_gigaam_v3.rs \
    crates/vokra-models/src/gigaam/v3.rs; do
    [[ -f "$ROOT/$path" ]] || die "missing contract: $path"
  done
  rg -n -- 'AUTHENTICATED_PREPARED_SHA256|--exact --ignored --nocapture --test-threads=1|OPEN_UNSUPPORTED|GigaAM-v3' "$ROOT/scripts/publish/vast-ai/run-gigaam-v3-validation.sh" >/dev/null || die "phase contract missing"
  rg -n -- 'cargo run --locked -p vokra-cli -- convert --model sber-gigaam-v3|export GIGAAM_V3_GGUF|export GIGAAM_V3_REFERENCE_DIR|CONVERTER_APPROVED_SHA|RUNTIME_APPROVED_SHA' "$ROOT/scripts/publish/vast-ai/run-gigaam-v3-validation.sh" >/dev/null || die "converter/env approval chain missing"
  rg -n -- 'decision_argmax\.u32le|decision_frames.*decision_symbols|joint_output.*log_softmax' "$ROOT/tools/parity/sber_gigaam_v3_dump_reference.py" "$ROOT/tools/parity/gigaam_v3_validation.py" >/dev/null || die "decision trace contract missing"
  if rg -n -- 'git push|upload\.sh|publish-one\.sh' "$ROOT/scripts/publish/vast-ai/run-gigaam-v3-validation.sh" | grep -v 'if rg -n' >/dev/null; then die "upload command found"; fi
  UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/vokra-gigaam-uv-cache}" uv run --frozen --project "$ROOT/tools/parity" --python 3.12 python "$ROOT/tools/parity/sber_gigaam_v3_dump_reference.py" --self-test >/dev/null || die "reference dumper self-test failed"
  UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/vokra-gigaam-uv-cache}" uv run --frozen --project "$ROOT/tools/parity" --python 3.12 python "$ROOT/tools/parity/gigaam_v3_validation.py" --self-test >/dev/null || die "validator self-test failed"
  UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/vokra-gigaam-uv-cache}" uv run --frozen --project "$ROOT/tools/parity" --python 3.12 python "$ROOT/tools/parity/sber_gigaam_v3_prepare_checkpoint.py" --self-test >/dev/null || die "preparer self-test failed"
  UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/vokra-gigaam-uv-cache}" uv run --frozen --project "$ROOT/tools/parity" --python 3.12 python - <<'PY'
import re

PATTERN = re.compile(
    r'\bpub\s+const\s+AUTHENTICATED_PREPARED_SHA256\s*:\s*'
    r'Option\s*<\s*&str\s*>\s*=\s*Some\s*\(\s*"([0-9a-f]{64})"\s*\)\s*;',
    re.DOTALL,
)


def extract(source):
    matches = PATTERN.findall(source)
    if len(matches) != 1:
        raise ValueError("AUTHENTICATED_PREPARED_SHA256 must be exactly one Some(lowercase SHA-256)")
    return matches[0]


sha = "0123456789abcdef" * 4
assert extract(
    'pub const AUTHENTICATED_PREPARED_SHA256: Option<&str> =\n'
    f'    Some("{sha}");\n'
) == sha
for invalid in (
    'pub const AUTHENTICATED_PREPARED_SHA256: Option<&str> = None;\n',
    f'pub const AUTHENTICATED_PREPARED_SHA256: Option<&str> = Some("{sha}");\n'
    f'pub const AUTHENTICATED_PREPARED_SHA256: Option<&str> = Some("{sha}");\n',
    'pub const AUTHENTICATED_PREPARED_SHA256: Option<&str> = Some("short");\n',
):
    try:
        extract(invalid)
    except ValueError:
        pass
    else:
        raise SystemExit("invalid AUTHENTICATED_PREPARED_SHA256 layout was accepted")
PY
  parity_log_test_dir="$(mktemp -d)"
  trap 'rm -rf "$parity_log_test_dir"' EXIT
  cat > "$parity_log_test_dir/good.log" <<'EOF'
running 1 test

test real_gigaam_v3_cpu_trace_matches_official ... GIGAAM_V3_PARITY log_mel max_abs=1.000000000e-03 mean_abs=2.000000000e-04
GIGAAM_V3_PARITY encoded max_abs=1.000000000e-03 mean_abs=2.000000000e-04
GIGAAM_V3_PARITY rnnt_logits max_abs=1.000000000e-03 mean_abs=2.000000000e-04
GIGAAM_V3_PARITY CPU PASS; Metal OPEN_UNSUPPORTED; publication NO_UPLOAD
ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 1 filtered out; finished in 0.01s
EOF
  validate_parity_log "$parity_log_test_dir/good.log" || die "valid interleaved parity log was rejected"
  cat > "$parity_log_test_dir/duplicate-test.log" <<'EOF'
running 1 test
test real_gigaam_v3_cpu_trace_matches_official ... GIGAAM_V3_PARITY log_mel max_abs=1.000000000e-03 mean_abs=2.000000000e-04
test real_gigaam_v3_cpu_trace_matches_official ... GIGAAM_V3_PARITY log_mel max_abs=1.000000000e-03 mean_abs=2.000000000e-04
ok
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 2 filtered out; finished in 0.01s
EOF
  if validate_parity_log "$parity_log_test_dir/duplicate-test.log"; then die "duplicate target test was accepted"; fi
  cat > "$parity_log_test_dir/extra-test.log" <<'EOF'
running 1 test
test another_test ... ok
test real_gigaam_v3_cpu_trace_matches_official ... GIGAAM_V3_PARITY log_mel max_abs=1.000000000e-03 mean_abs=2.000000000e-04
ok
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 2 filtered out; finished in 0.01s
EOF
  if validate_parity_log "$parity_log_test_dir/extra-test.log"; then die "extra named test was accepted"; fi
  cat > "$parity_log_test_dir/duplicate-marker.log" <<'EOF'
running 1 test
test real_gigaam_v3_cpu_trace_matches_official ... GIGAAM_V3_PARITY log_mel max_abs=1.000000000e-03 mean_abs=2.000000000e-04
GIGAAM_V3_PARITY encoded max_abs=1.000000000e-03 mean_abs=2.000000000e-04
GIGAAM_V3_PARITY rnnt_logits max_abs=1.000000000e-03 mean_abs=2.000000000e-04
GIGAAM_V3_PARITY CPU PASS; Metal OPEN_UNSUPPORTED; publication NO_UPLOAD
GIGAAM_V3_PARITY CPU PASS; Metal OPEN_UNSUPPORTED; publication NO_UPLOAD
ok
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 1 filtered out; finished in 0.01s
EOF
  if validate_parity_log "$parity_log_test_dir/duplicate-marker.log"; then die "duplicate CPU marker was accepted"; fi
  cat > "$parity_log_test_dir/malformed-marker.log" <<'EOF'
running 1 test
test real_gigaam_v3_cpu_trace_matches_official ... GIGAAM_V3_PARITY log_mel max_abs=nan mean_abs=2.000000000e-04
GIGAAM_V3_PARITY encoded max_abs=1.000000000e-03 mean_abs=2.000000000e-04
GIGAAM_V3_PARITY rnnt_logits max_abs=1.000000000e-03 mean_abs=2.000000000e-04
GIGAAM_V3_PARITY CPU PASS; Metal OPEN_UNSUPPORTED; publication NO_UPLOAD
ok
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 1 filtered out; finished in 0.01s
EOF
  if validate_parity_log "$parity_log_test_dir/malformed-marker.log"; then die "malformed metric marker was accepted"; fi
  cat > "$parity_log_test_dir/malformed-summary.log" <<'EOF'
running 1 test
test real_gigaam_v3_cpu_trace_matches_official ... GIGAAM_V3_PARITY log_mel max_abs=1.000000000e-03 mean_abs=2.000000000e-04
GIGAAM_V3_PARITY encoded max_abs=1.000000000e-03 mean_abs=2.000000000e-04
GIGAAM_V3_PARITY rnnt_logits max_abs=1.000000000e-03 mean_abs=2.000000000e-04
GIGAAM_V3_PARITY CPU PASS; Metal OPEN_UNSUPPORTED; publication NO_UPLOAD
ok
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; -1 filtered out; finished in 0.01s
EOF
  if validate_parity_log "$parity_log_test_dir/malformed-summary.log"; then die "malformed summary was accepted"; fi
  rm -rf "$parity_log_test_dir"
  trap - EXIT
  echo "run-gigaam-v3-validation.sh self-test: OK (NO_UPLOAD; parity OPEN)"
  exit 0
fi

[[ $# == 2 && "${1:-}" == --phase ]] || die "usage: $0 --phase {measure|parity}"
PHASE="$2"
[[ "$PHASE" == measure || "$PHASE" == parity ]] || die "usage: $0 --phase {measure|parity}"
for tool in git realpath sha256sum stat free df awk uv; do require_tool "$tool"; done
[[ "$(free -b | awk '/^Mem:/ {print $2}')" =~ ^[0-9]+$ ]] || die "cannot read RAM"
[[ "$(free -b | awk '/^Mem:/ {print $2}')" -ge 17179869184 ]] || die "at least 16 GiB RAM is required"
[[ "$(df -Pk "$ROOT" | awk 'NR==2 {print $4}')" =~ ^[0-9]+$ ]] || die "cannot read free disk"
[[ "$(df -Pk "$ROOT" | awk 'NR==2 {print $4}')" -ge 20971520 ]] || die "at least 20 GiB free disk is required"
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || die "VAST requires Linux x86_64"
[[ "${VOKRA_PUBLISH_ON_VAST:-0}" == 1 ]] || die "VOKRA_PUBLISH_ON_VAST=1 is absent"
[[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] || die "checkout must be clean"
# The broad parity environment is not an approved production dependency
# closure. Keep both phases fail-closed until this dedicated project and lock
# are reviewed and committed.
[[ -f "$ROOT/tools/parity/gigaam_v3/pyproject.toml" && -f "$ROOT/tools/parity/gigaam_v3/uv.lock" ]] || die "dedicated GigaAM v3 pyproject.toml+uv.lock are not reviewed"
V3_PROJECT="$ROOT/tools/parity/gigaam_v3"

[[ -n "${GIGAAM_EVIDENCE_DIR:-}" ]] || die "set evidence path"
reject_symlink_ancestry "$GIGAAM_EVIDENCE_DIR"
[[ ! -e "$GIGAAM_EVIDENCE_DIR" && ! -L "$GIGAAM_EVIDENCE_DIR" ]] || die "evidence directory must be absent"
EVIDENCE_REAL="$(realpath -m "$GIGAAM_EVIDENCE_DIR")"
ROOT_REAL="$(realpath "$ROOT")"
case "$EVIDENCE_REAL" in "$ROOT_REAL"|"$ROOT_REAL"/*) die "evidence must be outside checkout";; esac

if [[ "$PHASE" == measure ]]; then
  [[ -n "${GIGAAM_MODEL_DIR:-}" && -n "${GIGAAM_REFERENCE_DIR:-}" && -n "${GIGAAM_WORK_DIR:-}" ]] || die "set model, reference, and absent work paths"
  reject_symlink_ancestry "$GIGAAM_MODEL_DIR"
  reject_symlink_ancestry "$GIGAAM_REFERENCE_DIR"
  reject_symlink_ancestry "$GIGAAM_WORK_DIR"
  [[ -d "$GIGAAM_MODEL_DIR" && ! -L "$GIGAAM_MODEL_DIR" ]] || die "model snapshot is missing"
  [[ ! -e "$GIGAAM_REFERENCE_DIR" && ! -L "$GIGAAM_REFERENCE_DIR" ]] || die "reference output must be absent"
  [[ ! -e "$GIGAAM_WORK_DIR" && ! -L "$GIGAAM_WORK_DIR" ]] || die "work directory must be absent"
  WORK_REAL="$(realpath -m "$GIGAAM_WORK_DIR")"
  disjoint "$EVIDENCE_REAL" "$WORK_REAL"
  disjoint "$EVIDENCE_REAL" "$(realpath -m "$GIGAAM_REFERENCE_DIR")"
  case "$WORK_REAL" in "$ROOT_REAL"|"$ROOT_REAL"/*) die "work must be outside checkout";; esac
  mkdir "$GIGAAM_WORK_DIR"
  GIGAAM_PREPARED_SAFETENSORS="$GIGAAM_WORK_DIR/prepared.safetensors"
  uv run --frozen --project "$V3_PROJECT" --python 3.12 python "$ROOT/tools/parity/sber_gigaam_v3_prepare_checkpoint.py" --input "$GIGAAM_MODEL_DIR/pytorch_model.bin" --output "$GIGAAM_PREPARED_SAFETENSORS"
else
  [[ -n "${GIGAAM_PREPARED_SAFETENSORS:-}" ]] || die "set prepared path"
  reject_symlink_ancestry "$GIGAAM_PREPARED_SAFETENSORS"
  [[ -f "$GIGAAM_PREPARED_SAFETENSORS" && ! -L "$GIGAAM_PREPARED_SAFETENSORS" ]] || die "prepared safetensors must be regular"
fi
PREPARED_REAL="$(realpath "$GIGAAM_PREPARED_SAFETENSORS")"
case "$PREPARED_REAL" in "$ROOT_REAL"|"$ROOT_REAL"/*) die "prepared file must be outside checkout";; esac
disjoint "$EVIDENCE_REAL" "$PREPARED_REAL"
SIDECAR="${GIGAAM_PREPARED_SAFETENSORS%.*}.manifest.json"
reject_symlink_ancestry "$SIDECAR"
[[ -f "$SIDECAR" && ! -L "$SIDECAR" ]] || die "prepared sidecar must be regular"
disjoint "$EVIDENCE_REAL" "$(realpath "$SIDECAR")"
PREPARED_SHA256="$(sha256sum "$GIGAAM_PREPARED_SAFETENSORS" | awk '{print $1}')"
[[ "$PREPARED_SHA256" =~ ^[0-9a-f]{64}$ ]] || die "prepared digest is not lowercase SHA-256"
SIDECAR_SHA256="$(sha256sum "$SIDECAR" | awk '{print $1}')"
[[ "$SIDECAR_SHA256" =~ ^[0-9a-f]{64}$ ]] || die "sidecar digest is not lowercase SHA-256"
SIDECAR_PREPARED_SHA256="$(uv run --frozen --project "$V3_PROJECT" --python 3.12 python - "$SIDECAR" <<'PY'
import json
import sys
from pathlib import Path
doc = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
value = doc.get("prepared_sha256")
if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
    raise SystemExit("sidecar prepared_sha256 is not lowercase SHA-256")
print(value)
PY
)"
[[ "$SIDECAR_PREPARED_SHA256" == "$PREPARED_SHA256" ]] || die "sidecar prepared SHA does not match input bytes"

if [[ "$PHASE" == measure ]]; then
  [[ -n "${GIGAAM_MODEL_DIR:-}" && -n "${GIGAAM_REFERENCE_DIR:-}" ]] || die "set model snapshot and reference paths"
  reject_symlink_ancestry "$GIGAAM_MODEL_DIR"
  reject_symlink_ancestry "$GIGAAM_REFERENCE_DIR"
  [[ -d "$GIGAAM_MODEL_DIR" && ! -L "$GIGAAM_MODEL_DIR" ]] || die "model snapshot is missing"
  REFERENCE_DIR="$GIGAAM_REFERENCE_DIR"
  [[ ! -e "$REFERENCE_DIR" && ! -L "$REFERENCE_DIR" ]] || die "reference output must be absent"
  REFERENCE_REAL="$(realpath -m "$REFERENCE_DIR")"
  disjoint "$EVIDENCE_REAL" "$REFERENCE_REAL"
  disjoint "$PREPARED_REAL" "$REFERENCE_REAL"
  uv run --frozen --project "$V3_PROJECT" --python 3.12 python "$ROOT/tools/parity/sber_gigaam_v3_dump_reference.py" --model-dir "$GIGAAM_MODEL_DIR" --output "$REFERENCE_DIR"
  uv run --frozen --project "$V3_PROJECT" --python 3.12 python "$ROOT/tools/parity/gigaam_v3_validation.py" "$REFERENCE_DIR"
  REFERENCE_MANIFEST_SHA256="$(sha256sum "$REFERENCE_DIR/manifest.json" | awk '{print $1}')"
  [[ "$REFERENCE_MANIFEST_SHA256" =~ ^[0-9a-f]{64}$ ]] || die "reference manifest digest is invalid"
  python_commit="$(git -C "$ROOT" rev-parse HEAD)"
  [[ "$python_commit" =~ ^[0-9a-f]{40}$ ]] || die "invalid commit identity"
  mkdir "$GIGAAM_EVIDENCE_DIR"
  cat > "$GIGAAM_EVIDENCE_DIR/prepared-digest.json" <<EOF
{"format":"vokra-gigaam-v3-prepared-digest-v1","prepared_path":"$PREPARED_REAL","prepared_sha256":"$PREPARED_SHA256","prepared_bytes":$(stat -c '%s' "$GIGAAM_PREPARED_SAFETENSORS"),"sidecar_sha256":"$SIDECAR_SHA256","reference_manifest_sha256":"$REFERENCE_MANIFEST_SHA256","git_commit":"$python_commit","status":"MEASURED_OPEN_NO_UPLOAD"}
EOF
  echo "GigaAM v3 measure: prepared_sha256=$PREPARED_SHA256 reference=$REFERENCE_DIR status=MEASURED_OPEN_NO_UPLOAD"
  exit 0
fi

[[ -n "${GIGAAM_REFERENCE_DIR:-}" && -n "${GIGAAM_GGUF:-}" ]] || die "set reference and absent GGUF paths"
reject_symlink_ancestry "$GIGAAM_REFERENCE_DIR"
reject_symlink_ancestry "$GIGAAM_GGUF"
[[ -d "$GIGAAM_REFERENCE_DIR" && ! -L "$GIGAAM_REFERENCE_DIR" ]] || die "reference directory is missing"
[[ ! -e "$GIGAAM_GGUF" && ! -L "$GIGAAM_GGUF" ]] || die "GGUF output must be absent"
REFERENCE_REAL="$(realpath "$GIGAAM_REFERENCE_DIR")"
GGUF_REAL="$(realpath -m "$GIGAAM_GGUF")"
disjoint "$EVIDENCE_REAL" "$REFERENCE_REAL"
disjoint "$EVIDENCE_REAL" "$GGUF_REAL"
disjoint "$PREPARED_REAL" "$REFERENCE_REAL"
disjoint "$PREPARED_REAL" "$GGUF_REAL"
APPROVED_SHAS="$(uv run --frozen --project "$V3_PROJECT" --python 3.12 python - \
  "$ROOT/crates/vokra-convert/src/models/sber_gigaam_v3.rs" \
  "$ROOT/crates/vokra-models/src/gigaam/v3.rs" <<'PY'
import re
import sys
from pathlib import Path

pattern = re.compile(
    r'\bpub\s+const\s+AUTHENTICATED_PREPARED_SHA256\s*:\s*'
    r'Option\s*<\s*&str\s*>\s*=\s*Some\s*\(\s*"([0-9a-f]{64})"\s*\)\s*;',
    re.DOTALL,
)
for filename in sys.argv[1:]:
    source = Path(filename).read_text(encoding="utf-8")
    matches = pattern.findall(source)
    if len(matches) != 1:
        raise SystemExit(
            f"{filename}: AUTHENTICATED_PREPARED_SHA256 must be exactly one "
            "Some(lowercase SHA-256)"
        )
    print(matches[0])
PY
)" || die "approved prepared SHA cannot be parsed from converter/runtime"
[[ "$APPROVED_SHAS" == *$'\n'* ]] || die "approved prepared SHA parser returned fewer than two values"
CONVERTER_APPROVED_SHA="${APPROVED_SHAS%%$'\n'*}"
RUNTIME_APPROVED_SHA="${APPROVED_SHAS#*$'\n'}"
[[ "$CONVERTER_APPROVED_SHA" =~ ^[0-9a-f]{64}$ && "$RUNTIME_APPROVED_SHA" =~ ^[0-9a-f]{64}$ ]] || die "approved prepared SHA is not stamped in converter/runtime"
[[ "$CONVERTER_APPROVED_SHA" == "$RUNTIME_APPROVED_SHA" && "$CONVERTER_APPROVED_SHA" == "$PREPARED_SHA256" ]] || die "approved SHA does not equal prepared bytes"
uv run --frozen --project "$V3_PROJECT" --python 3.12 python "$ROOT/tools/parity/gigaam_v3_validation.py" "$GIGAAM_REFERENCE_DIR"
[[ -d "$(dirname "$GIGAAM_GGUF")" ]] || die "GGUF parent must already exist"
export GIGAAM_V3_GGUF="$GIGAAM_GGUF"
export GIGAAM_V3_REFERENCE_DIR="$GIGAAM_REFERENCE_DIR"
GIGAAM_V3_REFERENCE_MANIFEST_SHA256="$(sha256sum "$GIGAAM_REFERENCE_DIR/manifest.json" | awk '{print $1}')"
export GIGAAM_V3_REFERENCE_MANIFEST_SHA256
[[ "$GIGAAM_V3_REFERENCE_MANIFEST_SHA256" =~ ^[0-9a-f]{64}$ ]] || die "reference manifest digest is invalid"
cargo run --locked -p vokra-cli -- convert --model sber-gigaam-v3 --input "$GIGAAM_PREPARED_SAFETENSORS" --output "$GIGAAM_GGUF" --license mit
[[ -f "$GIGAAM_GGUF" && ! -L "$GIGAAM_GGUF" ]] || die "converter did not create GGUF"
mkdir "$GIGAAM_EVIDENCE_DIR"
cargo test --locked -p vokra-models --test parity_gigaam_v3_real real_gigaam_v3_cpu_trace_matches_official -- --exact --ignored --nocapture --test-threads=1 > "$GIGAAM_EVIDENCE_DIR/parity.log" 2>&1
validate_parity_log "$GIGAAM_EVIDENCE_DIR/parity.log" || die "parity log shape or PASS markers are invalid"
GGUF_SHA256="$(sha256sum "$GIGAAM_GGUF" | awk '{print $1}')"
[[ "$GGUF_SHA256" =~ ^[0-9a-f]{64}$ ]] || die "GGUF digest is invalid"
cat > "$GIGAAM_EVIDENCE_DIR/validation-summary.json" <<EOF
{"format":"vokra-gigaam-v3-validation-v1","phase":"parity","status":"CPU_PARITY_PASS","publication":"NO_UPLOAD","prepared_sha256":"$PREPARED_SHA256","sidecar_sha256":"$SIDECAR_SHA256","gguf_sha256":"$GGUF_SHA256","reference_manifest_sha256":"$GIGAAM_V3_REFERENCE_MANIFEST_SHA256","metal_apple_status":"OPEN_UNSUPPORTED"}
EOF
echo "GigaAM v3 parity: CPU PASS; Metal/Apple OPEN_UNSUPPORTED; publication NO_UPLOAD"
