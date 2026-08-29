#!/usr/bin/env bash
# VAST-only RMVPE replacement validation worker.
#
# This worker authenticates the historical public target, fetches the exact
# yxlllc/RMVPE release, prepares the pickle with a weights-only bridge, creates
# a strict unknown-provenance GGUF, imports the fixed upstream implementation
# for independent fixtures, and runs the real CPU/parity gates.  It never
# uploads, publishes, pushes, or mutates the public artifact.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VOKRA_ROOT="${VOKRA_ROOT:-$DEFAULT_ROOT}"
VOKRA_SCRATCH="${VOKRA_SCRATCH:-$HOME/scratchpad}"
PARITY_PROJECT="$VOKRA_ROOT/tools/parity"
RMVPE_PROJECT="$VOKRA_ROOT/tools/parity/rmvpe"
RMVPE_INSPECTOR="$PARITY_PROJECT/rmvpe_inspect.py"
FETCH_HELPER="$RMVPE_PROJECT/fetch_rmvpe_pt.sh"
PARITY_DUMPER="$RMVPE_PROJECT/dump_reference.py"
CLI_INPUT="$VOKRA_ROOT/tests/parity/silero_vad/test_16k.wav"

UPSTREAM_REPO="yxlllc/RMVPE"
UPSTREAM_URL="https://github.com/yxlllc/RMVPE.git"
UPSTREAM_REVISION="0aabafba18289ca938a73af0b0297686abf4922d"
UPSTREAM_RELEASE_URL="https://github.com/yxlllc/RMVPE/releases/download/230917/rmvpe.zip"
CHECKPOINT_FILE="model.pt"
MODEL_KIND="rmvpe"
LICENSE_SPDX="unknown"

# The release payload digest was not present in the audited handoff.  Do not
# invent one: the operator must supply the independently recorded digest.
CHECKPOINT_SHA256=""

# Historical public target.  It is downloaded only to authenticate what the
# replacement would supersede; its incorrect MIT/permissive provenance is not
# used as a parity input and is intentionally rejected by the strict loader.
PUBLIC_REPO="vokra/rmvpe"
PUBLIC_REVISION="3eb5fa8946f1074ba3959074c5cde95ec22b8c91"
PUBLIC_FILE="rmvpe.gguf"
PUBLIC_BYTES=181010688
PUBLIC_SHA256="208fc73819586b4546f2cba7a829033c5900c44af1ad48fe9d3e727cc1a932fb"

MIN_VAST_MEM_KIB=67108864
MIN_FREE_DISK_KIB=150000000

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-1}"

log() { printf '[rmvpe-vast] %s\n' "$*" >&2; }
step() { printf '\n[rmvpe-vast] ==== %s ====\n' "$*" >&2; }
die() { log "ERROR: $*"; return 2; }

usage() {
  cat <<'EOF' >&2
usage: run-rmvpe-validation.sh --checkpoint-sha256 <64-hex> \
         --approval-evidence <file> [--work-dir <absent-dir>]
       run-rmvpe-validation.sh --self-test

VAST-only, non-publishing RMVPE validation worker.  It authenticates the
historical vokra/rmvpe replacement target, fetches the exact yxlllc/RMVPE
release, uses a torch.load(weights_only=True) bridge to make safetensors,
converts the strict `rmvpe` model with `unknown` provenance, imports the exact
upstream source for independent fixtures, runs all real CPU parity paths and a
CLI F0 smoke, then runs repository workspace/clippy/license gates.

Normal runs require Linux x86_64, VOKRA_PUBLISH_ON_VAST=1, at least 64 GiB
RAM, 150 GB free disk, and an independently recorded checkpoint SHA-256.
--self-test is hermetic: it performs no network, model download, Python,
Cargo, credentials, or publication action.
EOF
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

verify_file() {
  local path="$1" expected_hash="$2" expected_bytes="${3:-}"
  local actual_hash actual_bytes=""
  [[ -f "$path" && ! -L "$path" ]] || { die "missing, symlinked, or non-regular pinned input: $path"; return 2; }
  if [[ -n "$expected_bytes" ]]; then
    actual_bytes="$(wc -c < "$path" | tr -d '[:space:]')"
    [[ "$actual_bytes" == "$expected_bytes" ]] || {
      die "byte-size mismatch for $path: got $actual_bytes, expected $expected_bytes"
      return 2
    }
  fi
  actual_hash="$(sha256_file "$path")"
  [[ "$actual_hash" == "$expected_hash" ]] || {
    die "SHA-256 mismatch for $path: got $actual_hash, expected $expected_hash"
    return 2
  }
  log "identity OK: $path sha256=$actual_hash${actual_bytes:+ bytes=$actual_bytes}"
}

license_preflight() {
  local approval="$1" checkpoint_sha="$2" project_sha lock_sha
  [[ -f "$approval" && ! -L "$approval" && -s "$approval" ]] || die '--approval-evidence must be a nonempty regular non-symlink file'
  project_sha="$(sha256_file "$RMVPE_PROJECT/pyproject.toml")"; lock_sha="$(sha256_file "$RMVPE_PROJECT/uv.lock")"
  if UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python - "$approval" "$project_sha" "$lock_sha" "$checkpoint_sha" <<'PY'
import hashlib, json, pathlib, sys
def reject(pairs):
    d = {}
    for k, v in pairs:
        if k in d: raise ValueError("duplicate JSON key: " + k)
        d[k] = v
    return d
try:
    d = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"), object_pairs_hook=reject)
    keys = {"schema", "model", "upstream_repo", "upstream_revision", "license_spdx", "project_sha256", "lock_sha256", "checkpoint_sha256", "no_upload", "decision", "signer", "scope_sha256"}
    if set(d) != keys: raise ValueError("approval schema is not exact")
    if d["schema"] != "vokra-validation-approval-v1" or d["model"] != "rmvpe" or d["upstream_repo"] != "yxlllc/RMVPE" or d["upstream_revision"] != "0aabafba18289ca938a73af0b0297686abf4922d": raise ValueError("RMVPE identity mismatch")
    if d["license_spdx"] != "unknown" or d["project_sha256"] != sys.argv[2] or d["lock_sha256"] != sys.argv[3] or d["checkpoint_sha256"] != sys.argv[4] or d["no_upload"] is not True or d["decision"] != "APPROVED": raise ValueError("approval facts mismatch")
    if not isinstance(d["signer"], str) or not d["signer"].strip() or d["signer"].strip().upper() in {"TODO", "UNRESOLVED", "OWNER_SIGNOFF_REQUIRED"}: raise ValueError("approval signer unresolved")
    scope = {"checkpoint_sha256": sys.argv[4], "license_spdx": d["license_spdx"], "lock_sha256": sys.argv[3], "model": d["model"], "no_upload": True, "project_sha256": sys.argv[2], "upstream_repo": d["upstream_repo"], "upstream_revision": d["upstream_revision"]}
    if d["scope_sha256"] != hashlib.sha256(json.dumps(scope, sort_keys=True, separators=(",", ":")).encode()).hexdigest(): raise ValueError("approval scope digest mismatch")
except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
    raise SystemExit("approval gate BLOCKED: " + str(exc))
PY
  then :; else die 'approval evidence is invalid or offline Python is unavailable'; fi
}

canonical_absent_path() {
  local path="$1" suffix='' rest component scan name parent
  [[ "$path" == /* ]] || path="$PWD/$path"; rest="${path#/}"; scan=''
  while [[ -n "$rest" ]]; do
    component="${rest%%/*}"; rest="${rest#*/}"; [[ "$component" == "$rest" ]] && rest=''
    [[ -n "$component" && "$component" != . && "$component" != .. ]] || continue
    scan="$scan/$component"; [[ ! -L "$scan" || "$scan" == /var ]] || return 1
  done
  while [[ ! -d "$path" || -L "$path" ]]; do
    name="${path##*/}"; [[ -n "$name" ]] && suffix="/$name$suffix"; parent="${path%/*}"; [[ "$parent" == "$path" ]] && parent=/; path="$parent"
  done
  (cd -P "$path" && printf '%s%s\n' "$PWD" "$suffix")
}
paths_overlap() { [[ "$1" == "$2" || "$1" == "$2"/* || "$2" == "$1"/* ]]; }
require_absent_work_dir() {
  local target="$1" approval="$2" candidate protected other
  [[ ! -e "$target" && ! -L "$target" ]] || { die 'work-dir must be absent and non-symlink'; return 2; }
  candidate="$(canonical_absent_path "$target")" || { die 'work-dir has a symlinked ancestor'; return 2; }
  for protected in "$VOKRA_ROOT" "$RMVPE_PROJECT" "$approval"; do
    [[ ! -L "$protected" ]] || { die 'protected input is symlinked'; return 2; }
    other="$(canonical_absent_path "$protected")" || return 2
    paths_overlap "$candidate" "$other" && { die 'work-dir overlaps protected input'; return 2; }
  done
}

download_hf_file() {
  local repository="$1" revision="$2" filename="$3" output_dir="$4"
  mkdir -p "$output_dir"
  uv run --project "$PARITY_PROJECT" --frozen --python 3.12 python -c \
    'import sys; from huggingface_hub import hf_hub_download; print(hf_hub_download(repo_id=sys.argv[1], revision=sys.argv[2], filename=sys.argv[3], local_dir=sys.argv[4]))' \
    "$repository" "$revision" "$filename" "$output_dir"
  [[ -f "$output_dir/$filename" ]] || die "download did not produce $output_dir/$filename"
}

require_vast_host() {
  local mem_kib free_kib
  [[ "${VOKRA_PUBLISH_ON_VAST:-0}" == "1" ]] \
    || die "VOKRA_PUBLISH_ON_VAST=1 is absent; run provision.sh first"
  [[ "$(uname -s)" == "Linux" ]] \
    || die "RMVPE checkpoint work is VAST/Linux-only; refusing $(uname -s)"
  [[ "$(uname -m)" == "x86_64" ]] \
    || die "locked RMVPE reference targets Linux x86_64, got $(uname -m)"
  [[ -r /proc/meminfo ]] || die "/proc/meminfo is unavailable"
  mem_kib="$(awk '$1 == "MemTotal:" {print $2; exit}' /proc/meminfo)"
  [[ "$mem_kib" =~ ^[0-9]+$ ]] || die "could not read MemTotal"
  (( mem_kib >= MIN_VAST_MEM_KIB )) \
    || die "MemTotal=${mem_kib} KiB is below the exact 64-GiB guard"
  mkdir -p "$VOKRA_SCRATCH"
  free_kib="$(df -Pk "$VOKRA_SCRATCH" | awk 'NR == 2 {print $4}')"
  [[ "$free_kib" =~ ^[0-9]+$ ]] || die "could not read free disk"
  (( free_kib >= MIN_FREE_DISK_KIB )) \
    || die "free disk=${free_kib} KiB is below the exact 150-GB guard"
}

require_tooling() {
  local tool path
  for tool in uv cargo rustc git curl awk grep find tee wc tr df nproc unzip rustfmt cargo-deny cargo-audit; do
    command -v "$tool" >/dev/null 2>&1 || die "required VAST tool missing: $tool"
  done
  cargo clippy --version >/dev/null 2>&1 \
    || die "the clippy component is missing on the VAST host"
  [[ -d "$VOKRA_ROOT/.git" && -f "$VOKRA_ROOT/Cargo.toml" ]] \
    || die "$VOKRA_ROOT is not the repository checkout"
  [[ -f "$PARITY_PROJECT/pyproject.toml" && -f "$PARITY_PROJECT/uv.lock" ]] \
    || die "tools/parity locked Python project is missing"
  [[ -f "$RMVPE_PROJECT/pyproject.toml" && -f "$RMVPE_PROJECT/uv.lock" ]] \
    || die "RMVPE parity project is missing: $RMVPE_PROJECT"
  [[ -f "$RMVPE_INSPECTOR" ]] \
    || die "RMVPE pre-download inspector is missing: $RMVPE_INSPECTOR"
  [[ -f "$VOKRA_ROOT/tools/audit/gguf_manifest.py" ]] \
    || die "GGUF metadata audit helper is missing"
  for path in "$FETCH_HELPER" "$PARITY_DUMPER" "$CLI_INPUT"; do
    [[ -f "$path" ]] || die "required RMVPE validation input is missing: $path"
  done
  grep -Fq "$UPSTREAM_RELEASE_URL" "$FETCH_HELPER" \
    || die "fetch helper does not retain the audited RMVPE release URL"
  grep -Fq '_weights_only_torch_load' "$PARITY_DUMPER" \
    || die "reference dumper lacks its fail-closed torch.load wrapper"
  grep -Fq 'weights_only=True' "$PARITY_DUMPER" \
    || die "reference dumper does not force weights_only=True"
  grep -Fq 'weights_only=False' "$PARITY_DUMPER" \
    || die "reference dumper does not reject contradictory unsafe loading"
  [[ -z "$(git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all)" ]] \
    || die "VAST checkout must be clean so evidence names one exact commit"
}

verify_locked_projects() {
  step "Verify exact locked Python projects offline"
  uv lock --check --offline --project "$PARITY_PROJECT" \
    || die "parent parity uv.lock is not reproducible offline"
  uv lock --check --offline --project "$RMVPE_PROJECT" \
    || die "dedicated RMVPE uv.lock is not reproducible offline"
}

run_dependency_gate() {
  [[ -f "$RMVPE_INSPECTOR" ]] \
    || { die "RMVPE dependency gate is missing: $RMVPE_INSPECTOR"; return 2; }
  command -v uv >/dev/null 2>&1 \
    || { die "uv is required to run the RMVPE dependency gate"; return 2; }
  log "running stdlib-only RMVPE dependency/license gate before any acquisition"
  # The inspector's exit 2 is intentional while the native/bundled notices,
  # source license, and checkpoint terms remain unresolved.  Keep this call
  # before host setup, uv execution, source cloning, or model acquisition so
  # a blocked route cannot reach an effectful action.
  UV_CACHE_DIR="${UV_CACHE_DIR:-${TMPDIR:-/tmp}/vokra-rmvpe-uv-cache}"
  export UV_CACHE_DIR
  if UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python "$RMVPE_INSPECTOR" --dependency-gate; then
    return 0
  else
    local rc=$?
    log "RMVPE dependency/license gate blocked execution (exit $rc)"
    return "$rc"
  fi
}

record_environment() {
  local output="$1" cpu_model cpu_flags
  cpu_model="$(awk -F ':' '$1 ~ /model name/ {sub(/^[[:space:]]+/, "", $2); print $2; exit}' /proc/cpuinfo)"
  cpu_flags="$(awk -F ':' '$1 ~ /flags/ {sub(/^[[:space:]]+/, "", $2); print $2; exit}' /proc/cpuinfo)"
  [[ -n "$cpu_model" && -n "$cpu_flags" ]] || die "CPU provenance unavailable"
  {
    echo "utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "git_commit=$(git -C "$VOKRA_ROOT" rev-parse HEAD)"
    echo "git_branch=$(git -C "$VOKRA_ROOT" branch --show-current)"
    echo "upstream_repo=$UPSTREAM_REPO"
    echo "upstream_url=$UPSTREAM_URL"
    echo "upstream_revision=$UPSTREAM_REVISION"
    echo "upstream_release_url=$UPSTREAM_RELEASE_URL"
    echo "checkpoint_sha256=$CHECKPOINT_SHA256"
    echo "public_repo=$PUBLIC_REPO"
    echo "public_revision=$PUBLIC_REVISION"
    echo "public_file=$PUBLIC_FILE"
    echo "public_sha256=$PUBLIC_SHA256"
    echo "uname=$(uname -a)"
    echo "cpu_model=$cpu_model"
    echo "cpu_flags=$cpu_flags"
    echo "nproc=$(nproc)"
    awk '$1 == "MemTotal:" {print "mem_total_kib=" $2; exit}' /proc/meminfo
    rustc --version --verbose
    cargo --version
    uv --version
    uv run --project "$RMVPE_PROJECT" --frozen --python 3.12 python -c \
      'import platform, torch; print(f"python={platform.python_version()}"); print(f"torch={torch.__version__}")'
  } | tee "$output"
}

prepare_checkpoint_safely() {
  local source="$1" output="$2"
  # This bridge intentionally accepts only plain tensor dictionaries.  A
  # weights-only load failure is fatal; there is no unrestricted pickle fallback.
  uv run --project "$RMVPE_PROJECT" --frozen --python 3.12 python - \
    "$source" "$output" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import save_file


source = Path(sys.argv[1])
output = Path(sys.argv[2])
checkpoint = torch.load(source, map_location="cpu", weights_only=True)


def tensor_dict(value: object) -> dict[str, torch.Tensor] | None:
    if not isinstance(value, dict):
        return None
    tensors = {
        name: tensor
        for name, tensor in value.items()
        if isinstance(name, str) and isinstance(tensor, torch.Tensor)
    }
    if tensors:
        return tensors
    for key in ("state_dict", "model", "module"):
        if key in value:
            nested = tensor_dict(value[key])
            if nested is not None:
                return nested
    return None


state = tensor_dict(checkpoint)
if state is None:
    raise SystemExit("RMVPE checkpoint is not a plain tensor state_dict")

kept: dict[str, torch.Tensor] = {}
for name, tensor in state.items():
    if tensor.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        if name.endswith(".num_batches_tracked"):
            continue
        raise SystemExit(
            f"unsupported non-floating RMVPE tensor {name!r} ({tensor.dtype}); "
            "refusing implicit conversion"
        )
    if tensor.ndim == 0 or any(size == 0 for size in tensor.shape):
        raise SystemExit(f"invalid RMVPE tensor shape {name!r}: {tuple(tensor.shape)}")
    kept[name] = tensor.detach().cpu().contiguous().clone()

if not any(name.startswith("unet.") for name in kept):
    raise SystemExit("RMVPE state_dict lacks the canonical unet.* tensor namespace")
if not any(name.startswith("fc.") for name in kept):
    raise SystemExit("RMVPE state_dict lacks the canonical fc.* tensor namespace")

output.parent.mkdir(parents=True, exist_ok=True)
save_file(kept, str(output), metadata={"source": "yxlllc/RMVPE"})
print(f"prepared {len(kept)} floating tensors with weights_only=True")
PY
  [[ -s "$output" ]] || die "safe checkpoint preparation emitted no safetensors"
}

verify_unknown_provenance() {
  local gguf="$1" metadata="$2"
  uv run --project "$PARITY_PROJECT" --frozen --python 3.12 python \
    "$VOKRA_ROOT/tools/audit/gguf_manifest.py" "$gguf" --metadata-only > "$metadata"
  uv run --project "$PARITY_PROJECT" --frozen --python 3.12 python - "$metadata" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path


metadata = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for key in ("vokra.provenance.license", "vokra.provenance.weight_license"):
    if metadata.get(key) != "unknown":
        raise SystemExit(
            f"corrected RMVPE GGUF must retain fail-closed {key}=unknown; "
            f"got {metadata.get(key)!r}"
        )
print("RMVPE provenance OK: license=unknown weight_license=unknown")
PY
}

run_self_test() (
  local script_path="${BASH_SOURCE[0]}" temporary fail=0 required
  temporary="$(mktemp -d "${TMPDIR:-/tmp}/vokra-rmvpe-vast.XXXXXX")"
  trap 'rm -rf "$temporary"' EXIT
  printf 'abc' > "$temporary/value"
  [[ "$(sha256_file "$temporary/value")" == \
    "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad" ]] \
    || die "SHA-256 helper self-test failed"

  # shellcheck disable=SC2016 # literal strings are contract tokens
  for required in \
    "$UPSTREAM_REPO" "$UPSTREAM_URL" "$UPSTREAM_REVISION" "$UPSTREAM_RELEASE_URL" \
    "$PUBLIC_REPO" "$PUBLIC_REVISION" "$PUBLIC_FILE" "$PUBLIC_BYTES" "$PUBLIC_SHA256" \
    "$MODEL_KIND" "$LICENSE_SPDX" "fetch_rmvpe_pt.sh" "dump_reference.py" \
    "tools/audit/gguf_manifest.py" "_weights_only_torch_load" 'weights_only=False' \
    'verify_unknown_provenance' 'unzip' 'CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-1}"' \
    'torch.load(source, map_location="cpu", weights_only=True)' \
    'there is no unrestricted pickle fallback' \
    'rmvpe_inspect.py' 'run_dependency_gate' 'verify_locked_projects' 'prepare_checkpoint_safely' '--dependency-gate' \
    'reference fixture output must start absent' \
    'src.inference.RMVPE / src.model.E2E0' \
    'uv run --project "$RMVPE_PROJECT" --frozen --python 3.12 python' \
    'target/release/vokra-cli convert' ' --model "$MODEL_KIND"' \
    ' --license "$LICENSE_SPDX"' \
    'parity_rmvpe_gguf_smoke' 'parity_rmvpe_full_upstream_f0' \
    'parity_rmvpe_from_hidden_argmax_match_rate'; do
    if ! grep -Fq -- "$required" "$script_path"; then
      log "self-test FAIL: worker contract lost token: $required"
      fail=1
    fi
  done

  # shellcheck disable=SC2016 # literal strings are contract tokens
  for required in \
    'uname -s' 'uname -m' 'VOKRA_PUBLISH_ON_VAST' 'MIN_VAST_MEM_KIB=67108864' \
    'MIN_FREE_DISK_KIB=150000000' 'MemTotal=' 'df -Pk' \
    'git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all' \
    'cargo fmt --all -- --check' 'cargo test --locked --workspace' \
    'cargo clippy --locked --workspace --all-targets -- -D warnings' \
    'cargo deny check licenses advisories bans' 'cargo audit' \
    'full RMVPE parity:' 'jointly voiced' 'path-B:'; do
    if ! grep -Fq -- "$required" "$script_path"; then
      log "self-test FAIL: fail-closed gate lost token: $required"
      fail=1
    fi
  done

  if grep -En '^[[:space:]]*(python3|python|pip)([[:space:]]|$)' "$script_path" >/dev/null; then
    log "self-test FAIL: direct Python/pip command found"
    fail=1
  fi
  if grep -En -- '(^|[[:space:]])(git[[:space:]]+push|.*upload\.sh|.*publish-one\.sh|--push|--upload|--publish)([[:space:]]|$)' \
    "$script_path" >/dev/null; then
    log "self-test FAIL: publication command found"
    fail=1
  fi
  if "$script_path" --self-test --work-dir "$temporary/other" >/dev/null 2>&1; then
    log "self-test FAIL: extra --self-test argument accepted"
    fail=1
  fi
  if "$script_path" --work-dir >/dev/null 2>&1; then
    log "self-test FAIL: missing --work-dir value accepted"
    fail=1
  fi
  if "$script_path" --checkpoint-sha256 0123 >/dev/null 2>&1; then
    log "self-test FAIL: short checkpoint digest accepted"
    fail=1
  fi
  for bad in '--checkpoint-sha256' '--checkpoint-sha256 -bad' '--checkpoint-sha256 0123 --checkpoint-sha256 4567' '--approval-evidence' '--approval-evidence -bad' '--approval-evidence a --approval-evidence b'; do
    if eval "\"$script_path\" $bad" >/dev/null 2>&1; then
      log "self-test FAIL: malformed or duplicate option accepted: $bad"
      fail=1
    fi
  done
  (( fail == 0 )) || return 1
  echo "run-rmvpe-validation.sh self-test: OK"
)

main() {
  local self_test=0 requested_work_dir="" approval_evidence="" run_stamp work_dir
  local checkpoint_sha256="" checkpoint_sha256_seen=0
  local approval_seen=0 work_seen=0 self_seen=0
  local input_dir public_dir upstream_dir fixture_dir evidence_dir
  local checkpoint public_gguf prepared_path gguf_path cli_log
  local gguf_metadata
  local run_log env_log parity_log workspace_log clippy_log summary_file

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --checkpoint-sha256)
        (( checkpoint_sha256_seen == 0 )) || { die 'duplicate --checkpoint-sha256'; return 2; }
        [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || { usage; return 2; }
        checkpoint_sha256="$2"
        checkpoint_sha256_seen=1
        shift 2
        ;;
      --work-dir)
        (( work_seen == 0 )) || { die 'duplicate --work-dir'; return 2; }
        [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || { die "--work-dir requires a directory"; return 2; }
        work_seen=1
        requested_work_dir="$2"
        shift 2
        ;;
      --approval-evidence)
        (( approval_seen == 0 )) || { die 'duplicate --approval-evidence'; return 2; }
        [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || { die '--approval-evidence requires a nonempty path'; return 2; }
        approval_seen=1; approval_evidence="$2"; shift 2
        ;;
      --self-test)
        (( self_seen == 0 )) || { die 'duplicate --self-test'; return 2; }
        self_seen=1
        self_test=1
        shift
        ;;
      -h|--help)
        usage
        return 0
        ;;
      *)
        usage
        die "unknown argument: $1"
        return 2
        ;;
    esac
  done

  if (( self_test == 1 )); then
    [[ -z "$requested_work_dir$checkpoint_sha256$approval_evidence" ]] \
      || { die "--self-test accepts no other arguments"; return 2; }
    run_self_test
    return $?
  fi
  (( checkpoint_sha256_seen == 1 )) \
    || { usage; die "--checkpoint-sha256 is required; the audited handoff contains no digest"; return 2; }
  [[ "$checkpoint_sha256" =~ ^[0-9a-fA-F]{64}$ ]] \
    || { die "--checkpoint-sha256 must be exactly 64 hexadecimal characters"; return 2; }
  CHECKPOINT_SHA256="$(printf '%s' "$checkpoint_sha256" | tr '[:upper:]' '[:lower:]')"
  (( approval_seen == 1 )) || { die '--approval-evidence is required'; return 2; }

  license_preflight "$approval_evidence" "$CHECKPOINT_SHA256" || return $?
  run_dependency_gate || return $?
  require_vast_host
  require_tooling
  verify_locked_projects
  cd "$VOKRA_ROOT"
  run_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  work_dir="${requested_work_dir:-$VOKRA_SCRATCH/rmvpe-validation/$run_stamp}"
  require_absent_work_dir "$work_dir" "$approval_evidence" || return $?

  input_dir="$work_dir/input"
  public_dir="$input_dir/public"
  upstream_dir="$input_dir/upstream"
  fixture_dir="$work_dir/fixtures"
  evidence_dir="$work_dir/evidence"
  checkpoint="$upstream_dir/$CHECKPOINT_FILE"
  public_gguf="$public_dir/$PUBLIC_FILE"
  prepared_path="$work_dir/rmvpe.safetensors"
  gguf_path="$work_dir/rmvpe-corrected-unknown.gguf"
  cli_log="$evidence_dir/cli.log"
  run_log="$evidence_dir/run.log"
  env_log="$evidence_dir/environment.txt"
  parity_log="$evidence_dir/parity.log"
  workspace_log="$evidence_dir/workspace.log"
  clippy_log="$evidence_dir/clippy.log"
  summary_file="$evidence_dir/summary.txt"
  gguf_metadata="$evidence_dir/gguf-metadata.json"
  mkdir -p "$evidence_dir" "$public_dir" "$upstream_dir"
  [[ ! -e "$fixture_dir" && ! -L "$fixture_dir" ]] || die 'reference fixture output must start absent'
  export UV_CACHE_DIR="$VOKRA_SCRATCH/uv-cache-rmvpe"
  exec > >(tee -a "$run_log") 2>&1
  # shellcheck disable=SC2154 # rc is assigned by the EXIT trap itself
  trap 'rc=$?; if [[ -n "${summary_file:-}" && ! -f "$summary_file" ]]; then printf "execution_status=FAIL\nexit_code=%s\n" "$rc" > "$summary_file"; fi; exit "$rc"' EXIT

  step "Record VAST environment"
  record_environment "$env_log"

  step "Authenticate the historical public replacement target"
  download_hf_file "$PUBLIC_REPO" "$PUBLIC_REVISION" "$PUBLIC_FILE" "$public_dir"
  verify_file "$public_gguf" "$PUBLIC_SHA256" "$PUBLIC_BYTES"

  step "Fetch and authenticate the exact upstream RMVPE checkpoint"
  bash "$FETCH_HELPER" --output "$checkpoint" --sha256 "$CHECKPOINT_SHA256"
  verify_file "$checkpoint" "$CHECKPOINT_SHA256"

  step "Clone and pin the exact upstream source"
  git clone --no-checkout "$UPSTREAM_URL" "$upstream_dir/source"
  git -C "$upstream_dir/source" checkout --detach "$UPSTREAM_REVISION"
  [[ "$(git -C "$upstream_dir/source" rev-parse HEAD)" == "$UPSTREAM_REVISION" ]] \
    || die "upstream checkout revision mismatch"
  [[ -z "$(git -C "$upstream_dir/source" status --porcelain --untracked-files=all)" ]] \
    || die "upstream source checkout is dirty"

  step "Prepare the real checkpoint with a weights-only bridge"
  prepare_checkpoint_safely "$checkpoint" "$prepared_path"

  step "Verify the independent dumper safe-load contract"
  uv run --project "$RMVPE_PROJECT" --frozen --python 3.12 python \
    "$PARITY_DUMPER" --self-test 2>&1 | tee "$evidence_dir/dumper-self-test.log"

  step "Generate independent official upstream fixtures"
  uv run --project "$RMVPE_PROJECT" --frozen --python 3.12 python "$PARITY_DUMPER" \
    --pt-path "$checkpoint" --upstream-src "$upstream_dir/source" \
    --canned --out-dir "$fixture_dir" 2>&1 | tee "$evidence_dir/dumper.log"
  for path in pcm.f32 hidden.f32 probabilities.f32 argmax.u32 f0.f32 meta.json; do
    [[ -s "$fixture_dir/$path" ]] || die "RMVPE dumper omitted fixture: $path"
  done
  grep -Fq '"upstream_revision": "'"$UPSTREAM_REVISION"'"' "$fixture_dir/meta.json" \
    || die "fixture metadata is not pinned to the exact upstream revision"
  grep -Fq '"feature_dim": 384' "$fixture_dir/meta.json" \
    || die "fixture metadata does not prove the 384-wide post-CNN state"
  grep -Fq '"n_class": 360' "$fixture_dir/meta.json" \
    || die "fixture metadata does not prove the 360-class head"

  step "Convert strict provenance-corrected RMVPE GGUF"
  cargo build --locked --release -p vokra-cli 2>&1 | tee "$workspace_log"
  target/release/vokra-cli convert --model "$MODEL_KIND" \
    --input "$prepared_path" --output "$gguf_path" --license "$LICENSE_SPDX" \
    2>&1 | tee -a "$workspace_log"
  [[ -s "$gguf_path" ]] || die "strict RMVPE converter emitted no GGUF"
  step "Verify corrected GGUF fail-closed provenance"
  verify_unknown_provenance "$gguf_path" "$gguf_metadata" \
    2>&1 | tee "$evidence_dir/provenance.log"

  export VOKRA_RMVPE_REAL_GGUF="$gguf_path"
  export VOKRA_RMVPE_REAL_PCM="$fixture_dir/pcm.f32"
  export VOKRA_RMVPE_REAL_HIDDEN="$fixture_dir/hidden.f32"
  export VOKRA_RMVPE_REAL_HIDDEN_FEATURE_DIM=384
  export VOKRA_RMVPE_REAL_ARGMAX="$fixture_dir/argmax.u32"
  export VOKRA_RMVPE_REAL_F0="$fixture_dir/f0.f32"
  step "Run all real RMVPE CPU parity paths"
  cargo test --locked -p vokra-models --test parity_rmvpe -- --nocapture \
    2>&1 | tee "$parity_log"
  for test_name in parity_rmvpe_gguf_smoke parity_rmvpe_full_upstream_f0 \
    parity_rmvpe_from_hidden_argmax_match_rate; do
    grep -Fq "test $test_name ... ok" "$parity_log" \
      || die "real RMVPE parity test did not pass: $test_name"
  done
  grep -Eq 'full RMVPE parity: .*\([1-9][0-9]*/[1-9][0-9]*\)' "$parity_log" \
    || die "full CPU/upstream parity log lacks a nonzero voiced comparison"
  grep -Eq 'path-B: [1-9][0-9]* / [1-9][0-9]* voiced frames' "$parity_log" \
    || die "post-CNN parity log lacks a nonzero voiced comparison"

  step "Run real RMVPE CLI F0 smoke"
  target/release/vokra-cli run --model "$gguf_path" --input "$CLI_INPUT" \
    --backend cpu 2>&1 | tee "$cli_log"
  grep -Eq 'f0: [1-9][0-9]* frames, voiced_frames=' "$cli_log" \
    || die "RMVPE CLI did not emit a nonzero F0 track summary"

  step "Run workspace, clippy, and license gates on VAST"
  bash "$VOKRA_ROOT/scripts/check-forbidden-symbols.sh" 2>&1 | tee -a "$workspace_log"
  bash "$VOKRA_ROOT/scripts/check-zero-deps.sh" 2>&1 | tee -a "$workspace_log"
  bash "$VOKRA_ROOT/scripts/check-bound-arch-coverage.sh" 2>&1 | tee -a "$workspace_log"
  cargo fmt --all -- --check 2>&1 | tee -a "$workspace_log"
  cargo test --locked --workspace 2>&1 | tee -a "$workspace_log"
  cargo clippy --locked --workspace --all-targets -- -D warnings 2>&1 | tee "$clippy_log"
  cargo deny check licenses advisories bans 2>&1 | tee -a "$workspace_log"
  cargo audit 2>&1 | tee -a "$workspace_log"

  {
    echo "execution_status=PASS"
    echo "git_commit=$(git -C "$VOKRA_ROOT" rev-parse HEAD)"
    echo "upstream_repo=$UPSTREAM_REPO"
    echo "upstream_revision=$UPSTREAM_REVISION"
    echo "checkpoint_sha256=$(sha256_file "$checkpoint")"
    echo "public_target_sha256=$(sha256_file "$public_gguf")"
    echo "prepared_safetensors_sha256=$(sha256_file "$prepared_path")"
    echo "corrected_gguf_sha256=$(sha256_file "$gguf_path")"
    echo "fixture_meta_sha256=$(sha256_file "$fixture_dir/meta.json")"
    echo "real_cpu_parity=PASS"
    echo "cli_f0_smoke=PASS"
    echo "workspace_clippy_license_gates=PASS"
    echo "provenance_license=$LICENSE_SPDX"
    echo "provenance_weight_license=unknown"
    echo "publication=NOT_PERFORMED"
  } | tee "$summary_file"
  trap - EXIT
  log "PASS: pull only $evidence_dir and evidence logs; do not pull model artifacts"
}

main "$@"
