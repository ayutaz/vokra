#!/usr/bin/env bash
# Prepare the VAST-only Dia reference environment.
#
# The frozen project deliberately omits optional audio dependencies.  NumPy is
# built from its exact locked sdist with BLAS/LAPACK disabled, then installed
# as the only NumPy distribution.  The caller must use ``uv run --no-sync``.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VOKRA_ROOT="${VOKRA_ROOT:-$DEFAULT_ROOT}"
PROJECT="$VOKRA_ROOT/tools/parity/dia_1_6b_reference"
BUILD_CONSTRAINTS="$PROJECT/numpy-build-constraints.txt"
LOCK_SHA256="58218102471c94979b1e9147759abf50fa3784793c193ff30cdde908400650dc"
PYPROJECT_SHA256="fa675f2c7542bd9eebedcc6ba29963f49093305c7a518542d71fad424449e77b"
NUMPY_SDIST_URL="https://files.pythonhosted.org/packages/dc/b2/ce4b867d8cd9c0ee84938ae1e6a6f7926ebf928c9090d036fc3c6a04f946/numpy-2.2.5.tar.gz"
NUMPY_SDIST_SHA256="a9c0d994680cd991b1cb772e8b297340085466a6fe964bc9d4e80f5e2f43c291"
NUMPY_SDIST_BYTES=20273920
BUILD_CONSTRAINTS_SHA256="3cfa1e8fcf7fc4ef9aeaa3a707b21f92b1d205c22b4e7f4231b99d85ecad8e3f"
MIN_VAST_MEM_KIB=60000000

log() { printf '[dia-reference-prep] %s\n' "$*" >&2; }
die() { log "ERROR: $*"; return 2; }

paths_overlap() { [[ "$1" == "$2" || "$1" == "$2"/* || "$2" == "$1"/* ]]; }

canonicalize_uncreated() {
  local path="$1" suffix='' name parent component rest scan
  [[ "$path" == /* ]] || return 1
  rest="${path#/}"; scan=''
  while [[ -n "$rest" ]]; do
    component="${rest%%/*}"
    [[ "$rest" == "$component" ]] && rest='' || rest="${rest#*/}"
    [[ -n "$component" && "$component" != . && "$component" != .. ]] || return 1
    scan="$scan/$component"
    [[ ! -L "$scan" ]] || return 1
  done
  while [[ ! -d "$path" || -L "$path" ]]; do
    name="${path##*/}"; [[ -n "$name" ]] && suffix="/$name$suffix"
    parent="${path%/*}"; [[ "$parent" == "$path" ]] && parent='/'
    path="$parent"; [[ ! -L "$path" ]] || return 1
  done
  (cd -P "$path" && printf '%s%s\n' "$PWD" "$suffix")
}

require_vast() {
  local memory
  [[ "${VOKRA_PUBLISH_ON_VAST:-0}" == 1 ]] || { die 'VOKRA_PUBLISH_ON_VAST=1 is required'; return 2; }
  [[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || { die 'Linux x86_64 VAST host required'; return 2; }
  for tool in uv sha256sum tar readelf cc ninja; do command -v "$tool" >/dev/null 2>&1 || { die "$tool is required"; return 2; }; done
  memory="$(awk '$1 == "MemTotal:" {print $2; exit}' /proc/meminfo)"
  [[ "$memory" =~ ^[0-9]+$ && "$memory" -ge "$MIN_VAST_MEM_KIB" ]] || { die 'VAST RAM is below the 60-GB guard'; return 2; }
}

require_contract() {
  [[ -d "$VOKRA_ROOT/.git" || -f "$VOKRA_ROOT/.git" ]] || { die 'VOKRA_ROOT is not a Vokra checkout'; return 2; }
  [[ -z "$(git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all)" ]] || { die 'VAST checkout must be clean'; return 2; }
  [[ -f "$PROJECT/pyproject.toml" && ! -L "$PROJECT/pyproject.toml" ]] || { die 'missing Dia pyproject'; return 2; }
  [[ -f "$PROJECT/uv.lock" && ! -L "$PROJECT/uv.lock" ]] || { die 'missing Dia uv.lock'; return 2; }
  [[ "$(sha256sum "$PROJECT/uv.lock" | awk '{print $1}')" == "$LOCK_SHA256" ]] || { die 'Dia uv.lock identity mismatch'; return 2; }
  [[ "$(sha256sum "$PROJECT/pyproject.toml" | awk '{print $1}')" == "$PYPROJECT_SHA256" ]] || { die 'Dia pyproject identity mismatch'; return 2; }
  [[ -f "$BUILD_CONSTRAINTS" && ! -L "$BUILD_CONSTRAINTS" ]] || { die 'NumPy build constraints are missing'; return 2; }
  [[ "$(sha256sum "$BUILD_CONSTRAINTS" | awk '{print $1}')" == "$BUILD_CONSTRAINTS_SHA256" ]] || { die 'NumPy build constraints identity mismatch'; return 2; }
}

download_sdist() {
  local destination="$1"
  UV_NO_CACHE=1 uv run --no-cache --no-project --python 3.12 python - "$destination" "$NUMPY_SDIST_URL" "$NUMPY_SDIST_SHA256" "$NUMPY_SDIST_BYTES" <<'PY'
import hashlib
import pathlib
import sys
import urllib.request

destination, url, expected_sha, expected_size = sys.argv[1:]
target = pathlib.Path(destination)
with urllib.request.urlopen(url, timeout=60) as response:
    data = response.read()
if len(data) != int(expected_size) or hashlib.sha256(data).hexdigest() != expected_sha:
    raise SystemExit("locked NumPy sdist identity mismatch")
target.write_bytes(data)
PY
}

inspect_wheel() {
  local wheel="$1"
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python - "$wheel" <<'PY'
import sys
import zipfile

wheel = sys.argv[1]
with zipfile.ZipFile(wheel) as archive:
    names = archive.namelist()
for name in names:
    lowered = name.casefold()
    if "numpy.libs/" in lowered or any(token in lowered for token in ("libgfortran", "libquadmath", "openblas")):
        raise SystemExit(f"forbidden NumPy wheel payload: {name}")
    if any(token in lowered for token in ("gpl", "lgpl")):
        raise SystemExit(f"GPL/LGPL NumPy wheel payload: {name}")
    if name.casefold().split("/")[-1] in {"license", "licence", "copying", "notice"}:
        body = archive.read(name).casefold()
        if b"gnu general public license" in body or b"gnu lesser general public license" in body:
            raise SystemExit(f"GPL/LGPL NumPy license payload: {name}")
PY
}

assert_build_requirements() {
  local source="$1"
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python - "$source/pyproject.toml" <<'PY'
import sys
import tomllib

with open(sys.argv[1], "rb") as stream:
    build = tomllib.load(stream).get("build-system", {})
if build.get("build-backend") != "mesonpy" or set(build.get("requires", ())) != {"meson-python>=0.15.0", "Cython>=3.0.6"}:
    raise SystemExit("NumPy sdist build-system requirements drifted from the bound closure")
PY
}

inspect_numpy_install() {
  local environment="$1"
  UV_PROJECT_ENVIRONMENT="$environment" UV_NO_CACHE=1 uv run --project "$PROJECT" --frozen --no-sync --python 3.12 python - <<'PY'
import pathlib
import subprocess

import numpy
from importlib import metadata

root = pathlib.Path(numpy.__file__).resolve().parent
dist = metadata.distribution("numpy")
native = []
for relative in dist.files or ():
    path = pathlib.Path(dist.locate_file(relative))
    if path.suffix not in {".so", ".dylib", ".dll", ".pyd"}:
        continue
    if not path.is_file() or path.is_symlink():
        raise SystemExit(f"NumPy native file is not regular: {path}")
    output = subprocess.run(["readelf", "-d", str(path)], check=True, capture_output=True, text=True).stdout
    needed = [line.split("[")[1].split("]", 1)[0] for line in output.splitlines() if "(NEEDED)" in line and "[" in line]
    lowered = " ".join(needed).casefold()
    if any(token in lowered for token in ("gfortran", "quadmath", "openblas")):
        raise SystemExit(f"forbidden NumPy native dependency: {path}: {needed}")
    native.append({"path": str(path.relative_to(root.parent)), "needed": needed})
if not native:
    raise SystemExit("prepared NumPy has no native payload to inspect")
if any("numpy.libs" in item["path"] for item in native):
    raise SystemExit("prepared NumPy unexpectedly contains numpy.libs")
PY
}

assert_optional_audio_absent() {
  local environment="$1"
  UV_PROJECT_ENVIRONMENT="$environment" UV_NO_CACHE=1 uv run --project "$PROJECT" --frozen --no-sync --python 3.12 python - <<'PY'
from importlib import metadata

for package in ("soundfile", "torchaudio"):
    try:
        metadata.version(package)
    except metadata.PackageNotFoundError:
        continue
    raise SystemExit(f"forbidden optional audio distribution is installed: {package}")
PY
}

write_config_evidence() {
  local environment="$1" destination="$2"
  UV_PROJECT_ENVIRONMENT="$environment" UV_NO_CACHE=1 uv run --project "$PROJECT" --frozen --no-sync --python 3.12 python - "$destination" <<'PY'
import contextlib
import io
import json
import platform
import sys
from importlib import metadata

import numpy

stream = io.StringIO()
with contextlib.redirect_stdout(stream):
    numpy.show_config()
config = stream.getvalue()
lowered = config.casefold()
if any(token in lowered for token in ("openblas", "libgfortran", "libquadmath")):
    raise SystemExit("NumPy config exposes a forbidden BLAS/native payload")
if "blas" not in lowered or "none" not in lowered:
    raise SystemExit("NumPy config does not prove BLAS/LAPACK none")
payload = {
    "numpy": {"name": metadata.distribution("numpy").metadata["Name"], "version": metadata.version("numpy")},
    "python": sys.version,
    "platform": platform.platform(),
    "config": config,
}
with open(sys.argv[1], "w", encoding="utf-8") as stream:
    json.dump(payload, stream, sort_keys=True, indent=2)
    stream.write("\n")
PY
}

write_build_dependency_evidence() {
  local builder="$1" destination="$2"
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python "$builder/bin/python" python - "$destination" "$BUILD_CONSTRAINTS" <<'PY'
import hashlib
import json
import sys
from importlib import metadata
from pathlib import Path

destination, constraints = sys.argv[1:]
names = ("Cython", "meson", "meson-python", "packaging", "pyproject-metadata")
rows = []
for name in names:
    dist = metadata.distribution(name)
    license_files = []
    for relative in dist.files or ():
        if Path(relative).name.casefold() not in {"license", "licence", "copying", "notice"}:
            continue
        path = Path(dist.locate_file(relative))
        if path.is_file() and not path.is_symlink():
            license_files.append({"path": str(relative), "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    license_metadata = dist.metadata.get("License")
    if not license_metadata and not license_files:
        raise SystemExit(f"build dependency has no publisher license fact: {name}")
    rows.append({"name": dist.metadata["Name"], "version": dist.version, "license_metadata": license_metadata, "license_files": license_files})
payload = {"schema": "vokra-dia-build-dependency-evidence-v1", "constraints": {"path": str(constraints), "sha256": hashlib.sha256(Path(constraints).read_bytes()).hexdigest()}, "packages": rows}
with open(destination, "w", encoding="utf-8") as stream:
    json.dump(payload, stream, sort_keys=True, indent=2)
    stream.write("\n")
PY
}

prepare() {
  local output="$1" output_real sdist src wheel_dir wheel environment wheel_path builder
  require_vast || return 2
  require_contract || return 2
  [[ "$output" == /* && ! -e "$output" && ! -L "$output" ]] || { die 'preparation output must be an absent absolute path'; return 2; }
  output_real="$(canonicalize_uncreated "$output")" || { die 'preparation output ancestry is unsafe'; return 2; }
  local root_real project_real
  root_real="$(canonicalize_uncreated "$VOKRA_ROOT")" || return 2
  project_real="$(canonicalize_uncreated "$PROJECT")" || return 2
  paths_overlap "$output_real" "$root_real" && { die 'preparation output overlaps checkout'; return 2; }
  paths_overlap "$output_real" "$project_real" && { die 'preparation output overlaps project'; return 2; }
  mkdir -p "$output_real" "$output_real/wheelhouse"
  sdist="$output_real/numpy-2.2.5.tar.gz"
  src="$output_real/src"
  wheel_dir="$output_real/wheelhouse"
  environment="$output_real/venv"
  builder="$output_real/build-venv"
  download_sdist "$sdist"
  tar -xzf "$sdist" -C "$output_real"
  [[ -d "$output_real/numpy-2.2.5" ]] || { die 'NumPy sdist extracted to an unexpected directory'; return 2; }
  mv "$output_real/numpy-2.2.5" "$src"
  assert_build_requirements "$src"
  log 'Installing every locked dependency except NumPy'
  UV_PROJECT_ENVIRONMENT="$environment" UV_NO_CACHE=1 UV_CACHE_DIR="${DIA_REFERENCE_UV_CACHE_DIR:-/tmp/vokra-dia-reference-uv-cache}" \
  uv sync --project "$PROJECT" --frozen --no-install-project --no-install-package numpy --python 3.12
  assert_optional_audio_absent "$environment"
  uv venv --python 3.12 "$builder"
  UV_NO_CACHE=1 uv pip install --python "$builder/bin/python" --require-hashes --no-deps -r "$BUILD_CONSTRAINTS"
  write_build_dependency_evidence "$builder" "$output_real/build-dependency-evidence.json"
  log 'Building exact NumPy sdist with BLAS/LAPACK disabled'
  PATH="$builder/bin:$PATH" uv build --no-build-isolation --python "$builder/bin/python" --wheel --out-dir "$wheel_dir" \
    -C setup-args=-Dblas=none -C setup-args=-Dlapack=none -C setup-args=-Dallow-noblas=true "$src"
  wheel_path="$(find "$wheel_dir" -maxdepth 1 -type f -name 'numpy-2.2.5-*.whl' -print -quit)"
  [[ -n "$wheel_path" && -f "$wheel_path" ]] || { die 'NumPy wheel was not produced'; return 2; }
  inspect_wheel "$wheel_path"
  log 'Installing only the prepared NumPy wheel without dependency resolution'
  UV_NO_CACHE=1 uv pip install --python "$environment/bin/python" --no-deps --force-reinstall "$wheel_path"
  assert_optional_audio_absent "$environment"
  inspect_numpy_install "$environment"
  write_config_evidence "$environment" "$output_real/numpy-config.json"
  wheel="$(sha256sum "$wheel_path" | awk '{print $1}')"
  uv_version="$(uv --version)"
  compiler_version="$(cc --version | head -n 1)"
  meson_version="$("$builder/bin/meson" --version)"
  ninja_version="$(ninja --version)"
  UV_PROJECT_ENVIRONMENT="$environment" UV_NO_CACHE=1 uv run --project "$PROJECT" --frozen --no-sync --python 3.12 python - "$output_real/preparation.json" "$wheel_path" "$wheel" "$environment" "$NUMPY_SDIST_URL" "$NUMPY_SDIST_SHA256" "$NUMPY_SDIST_BYTES" "$uv_version" "$compiler_version" "$meson_version" "$ninja_version" <<'PY'
import json
import os
import platform
import sys
from importlib import metadata
from pathlib import Path

destination, wheel_path, wheel_sha, environment, sdist_url, sdist_sha, sdist_bytes, uv_version, compiler_version, meson_version, ninja_version = sys.argv[1:]
dist = metadata.distribution("numpy")
payload = {
    "schema": "vokra-dia-reference-preparation-v1",
    "status": "PREPARED_NO_BLAS",
    "publication": "NO_UPLOAD",
    "sdist": {"name": "numpy", "version": "2.2.5", "url": sdist_url, "sha256": sdist_sha, "bytes": int(sdist_bytes)},
    "build": {"arguments": ["-C", "setup-args=-Dblas=none", "-C", "setup-args=-Dlapack=none", "-C", "setup-args=-Dallow-noblas=true"], "python": sys.version, "platform": platform.platform(), "uv": uv_version, "compiler": compiler_version, "meson": meson_version, "ninja": ninja_version, "isolation": "no-build-isolation; builder venv preinstalled from hash-pinned constraints"},
    "wheel": {"path": wheel_path, "sha256": wheel_sha, "bytes": Path(wheel_path).stat().st_size},
    "installed_distribution": {"name": dist.metadata["Name"], "version": dist.version, "location": str(dist.locate_file("")), "files": len(tuple(dist.files or ()))},
    "runtime": {"environment": environment, "uv_run_mode": "--no-sync", "soundfile_installed": False, "torchaudio_installed": False},
    "numpy_config": "numpy-config.json",
}
if payload["installed_distribution"]["version"] != "2.2.5":
    raise SystemExit("prepared NumPy distribution version mismatch")
with open(destination, "w", encoding="utf-8") as stream:
    json.dump(payload, stream, sort_keys=True, indent=2)
    stream.write("\n")
PY
  (cd "$output_real" && sha256sum preparation.json build-dependency-evidence.json numpy-config.json numpy-2.2.5.tar.gz "$wheel_path") > "$output_real/SHA256SUMS"
  log "Prepared environment: $environment"
}

self_test() {
  local failed=0 required_tools_line
  for token in 'VOKRA_PUBLISH_ON_VAST=1' 'uv sync --project' '--no-install-package numpy' '--require-hashes' '--no-build-isolation' '--no-deps --force-reinstall' '--no-sync' 'Dblas' 'NUMPY_SDIST_SHA256' 'readelf' 'compiler' 'build-dependency-evidence.json' 'numpy-config.json' 'NO_UPLOAD' 'soundfile_installed'; do
    grep -Fq -- "$token" "$0" || failed=1
  done
  required_tools_line="$(grep -F 'for tool in' "$0" | head -n 1)"
  [[ "$required_tools_line" == '  for tool in uv sha256sum tar readelf cc ninja; do command -v "$tool" >/dev/null 2>&1 || { die "$tool is required"; return 2; }; done' ]] || failed=1
  grep -Fq -- 'PATH="$builder/bin:$PATH" uv build --no-build-isolation' "$0" || failed=1
  grep -Fq -- '"$builder/bin/meson" --version' "$0" || failed=1
  grep -Fq -- 'no-build-isolation; builder venv preinstalled from hash-pinned constraints' "$0" || failed=1
  if grep -En '^[[:space:]]*(python3?|pip)([[:space:]]|$)' "$0" | grep -v 'grep -En' >/dev/null; then failed=1; fi
  if grep -En 'snapshot_download|git[[:space:]]+clone|cargo[[:space:]]+(build|test|check|clippy)|publish-one\.sh|--push|--upload' "$0" | grep -v 'grep -En' >/dev/null; then failed=1; fi
  (( failed == 0 )) || { log 'self-test FAIL'; return 1; }
  echo 'prepare-dia-1-6b-reference.sh self-test: PASS (model-free, VAST-only, NO_UPLOAD)'
}

main() {
  local output='' self=0
  while (($#)); do
    case "$1" in
      --output-dir) [[ $# -eq 2 && -z "$output" && "$2" != -* ]] || { die 'invalid or duplicate --output-dir'; return 2; }; output="$2"; shift 2 ;;
      --self-test) (( self++ == 0 )) || { die 'duplicate --self-test'; return 2; }; shift ;;
      -h|--help) echo 'usage: prepare-dia-1-6b-reference.sh --output-dir ABSENT_DIR | --self-test' >&2; return 0 ;;
      *) die "unknown argument: $1"; return 2 ;;
    esac
  done
  if (( self )); then [[ -z "$output" ]] || { die '--self-test accepts no output'; return 2; }; self_test; return $?; fi
  [[ -n "$output" ]] || { die '--output-dir is required'; return 2; }
  prepare "$output"
}

main "$@"
