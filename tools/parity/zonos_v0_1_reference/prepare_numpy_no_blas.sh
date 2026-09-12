#!/usr/bin/env bash
# VAST-only preparation of the exact locked Zonos NumPy sdist with BLAS disabled.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$SCRIPT_DIR"
REPOSITORY_ROOT="$(cd "$PROJECT/../../.." && pwd)"
SCRIPT_PATH="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"
CONSTRAINTS="$PROJECT/numpy-build-constraints.txt"
LOCK_SHA256="40fa51a7cffcfed126e073ecf0813fcbdb0935ea1bef05f51be1e75585fbcf76"
PYPROJECT_SHA256="5cb58da85195f8f0812aa18bedd6a320226c7a3ef94e64c33e5782414c115b29"
CONSTRAINTS_SHA256="812ab3e215d7756738ab9a9aca7b8c94b53a3b10e8f1e43228e1b74965be020a"
NUMPY_SDIST_URL="https://files.pythonhosted.org/packages/ec/d0/c12ddfd3a02274be06ffc71f3efc6d0e457b0409c4481596881e748cb264/numpy-2.2.2.tar.gz"
NUMPY_SDIST_SHA256="ed6906f61834d687738d25988ae117683705636936cc605be0bb208b23df4d8f"
NUMPY_SDIST_BYTES=20233295
MIN_VAST_MEM_KIB=60000000

log() { printf '[zonos-numpy-no-blas] %s\n' "$*" >&2; }
die() { log "ERROR: $*"; return 2; }

canonicalize_uncreated() {
  local path="$1" suffix='' name parent component rest scan
  [[ "$path" == /* ]] || return 1
  rest="${path#/}"; scan=''
  while [[ -n "$rest" ]]; do
    component="${rest%%/*}"
    [[ "$rest" == "$component" ]] && rest='' || rest="${rest#*/}"
    [[ -n "$component" && "$component" != . && "$component" != .. ]] || return 1
    scan="$scan/$component"; [[ ! -L "$scan" ]] || return 1
  done
  while [[ ! -d "$path" || -L "$path" ]]; do
    name="${path##*/}"; [[ -n "$name" ]] && suffix="/$name$suffix"
    parent="${path%/*}"; [[ "$parent" == "$path" ]] && parent='/'; path="$parent"
    [[ ! -L "$path" ]] || return 1
  done
  (cd -P "$path" && printf '%s%s\n' "$PWD" "$suffix")
}

require_vast() {
  local memory
  [[ "${VOKRA_PUBLISH_ON_VAST:-0}" == 1 ]] || { die 'VOKRA_PUBLISH_ON_VAST=1 is required'; return 2; }
  [[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || { die 'Linux x86_64 VAST host required'; return 2; }
  for tool in uv sha256sum tar readelf cc ninja; do command -v "$tool" >/dev/null 2>&1 || { die "$tool is required"; return 2; }; done
  memory="$(awk '$1 == "MemTotal:" {print $2; exit}' /proc/meminfo)"
  [[ "$memory" =~ ^[0-9]+$ && "$memory" -ge "$MIN_VAST_MEM_KIB" ]] || { die 'VAST RAM is below the 60-GiB guard'; return 2; }
}

require_contract() {
  [[ -f "$PROJECT/pyproject.toml" && ! -L "$PROJECT/pyproject.toml" ]] || { die 'Zonos pyproject is missing or symlinked'; return 2; }
  [[ -f "$PROJECT/uv.lock" && ! -L "$PROJECT/uv.lock" ]] || { die 'Zonos uv.lock is missing or symlinked'; return 2; }
  [[ -f "$CONSTRAINTS" && ! -L "$CONSTRAINTS" ]] || { die 'NumPy builder constraints are missing or symlinked'; return 2; }
  [[ "$(sha256sum "$PROJECT/pyproject.toml" | awk '{print $1}')" == "$PYPROJECT_SHA256" ]] || { die 'Zonos pyproject identity mismatch'; return 2; }
  [[ "$(sha256sum "$PROJECT/uv.lock" | awk '{print $1}')" == "$LOCK_SHA256" ]] || { die 'Zonos uv.lock identity mismatch'; return 2; }
  [[ "$(sha256sum "$CONSTRAINTS" | awk '{print $1}')" == "$CONSTRAINTS_SHA256" ]] || { die 'NumPy builder constraints identity mismatch'; return 2; }
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
digest = hashlib.sha256()
size = 0
with urllib.request.urlopen(url, timeout=60) as response, target.open("xb") as output:
    if response.geturl() != url:
        raise SystemExit("locked NumPy sdist redirect is forbidden")
    content_length = response.headers.get("Content-Length")
    if content_length is not None and content_length != expected_size:
        raise SystemExit("locked NumPy sdist Content-Length mismatch")
    for block in iter(lambda: response.read(1 << 20), b""):
        size += len(block)
        if size > int(expected_size):
            raise SystemExit("locked NumPy sdist exceeds locked size")
        digest.update(block)
        output.write(block)
if size != int(expected_size) or digest.hexdigest() != expected_sha:
    raise SystemExit("locked NumPy sdist identity mismatch")
PY
}

inspect_wheel() {
  local wheel="$1"
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python - "$wheel" <<'PY'
import sys
import zipfile

with zipfile.ZipFile(sys.argv[1]) as archive:
    for name in archive.namelist():
        lowered = name.casefold()
        if any(token in lowered for token in ("openblas", "libgfortran", "libquadmath", "gpl", "lgpl")):
            raise SystemExit(f"forbidden no-BLAS NumPy wheel payload: {name}")
        if lowered.split("/")[-1] in {"license", "licence", "copying", "notice"}:
            body = archive.read(name).lower()
            if b"gnu general public license" in body or b"gnu lesser general public license" in body:
                raise SystemExit(f"GPL/LGPL no-BLAS NumPy wheel license: {name}")
PY
}

extract_sdist() {
  local archive="$1" destination="$2"
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python - "$archive" "$destination" <<'PY'
import pathlib
import sys
import tarfile
from pathlib import PurePosixPath

archive, destination = map(pathlib.Path, sys.argv[1:])
# The locked NumPy 2.2.2 sdist has 7,758 unique regular-file members. Keep a
# small, explicit margin for that exact archive while retaining a hard cap.
maximum_members = 8192
maximum_member_bytes = 16 * 1024 * 1024
maximum_bytes = 512 * 1024 * 1024
members_seen = 0
bytes_seen = 0
seen_paths = set()
with tarfile.open(archive, "r:*") as package:
    members = package.getmembers()
    if len(members) > maximum_members:
        raise SystemExit("NumPy sdist has too many members")
    for member in members:
        members_seen += 1
        member_path = PurePosixPath(member.name)
        if (
            not member.name
            or "\\" in member.name
            or member_path.is_absolute()
            or ".." in member_path.parts
        ):
            raise SystemExit(f"unsafe NumPy sdist member path: {member.name!r}")
        if member.name in seen_paths:
            raise SystemExit(f"duplicate NumPy sdist member: {member.name!r}")
        seen_paths.add(member.name)
        target = destination.joinpath(*member_path.parts)
        if member.isdir():
            if target.exists() and (target.is_symlink() or not target.is_dir()):
                raise SystemExit(f"NumPy sdist directory collision: {target}")
            target.mkdir(parents=True, exist_ok=True)
            continue
        if member.issym() or member.islnk() or not member.isfile():
            raise SystemExit(f"unsafe NumPy sdist member type: {member.name!r}")
        if member.size < 0 or member.size > maximum_member_bytes or bytes_seen + member.size > maximum_bytes:
            raise SystemExit("NumPy sdist extracted bytes exceed the bound")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            raise SystemExit(f"NumPy sdist file collision: {target}")
        extracted = 0
        source = package.extractfile(member)
        if source is None:
            raise SystemExit(f"NumPy sdist member cannot be read: {member.name!r}")
        with source, target.open("xb") as output:
            for block in iter(lambda: source.read(1 << 20), b""):
                extracted += len(block)
                if extracted > member.size or bytes_seen + extracted > maximum_bytes:
                    target.unlink(missing_ok=True)
                    raise SystemExit("NumPy sdist extracted bytes exceed the bound")
                output.write(block)
        if extracted != member.size:
            target.unlink(missing_ok=True)
            raise SystemExit(f"NumPy sdist member size mismatch: {member.name!r}")
        bytes_seen += extracted
if members_seen == 0:
    raise SystemExit("NumPy sdist is empty")
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
    raise SystemExit("locked NumPy sdist build-system drifted from the reviewed no-BLAS constraints")
PY
}

prepare() {
  local output="$1" output_real sdist src wheel_dir environment builder wheel_path
  require_vast || return 2
  require_contract || return 2
  [[ "$output" == /* && ! -e "$output" && ! -L "$output" ]] || { die 'preparation output must be absent absolute path'; return 2; }
  output_real="$(canonicalize_uncreated "$output")" || { die 'preparation output ancestry is unsafe'; return 2; }
  local repo_real
  repo_real="$(canonicalize_uncreated "$REPOSITORY_ROOT")" || return 2
  [[ "$output_real" != "$repo_real" && "$output_real" != "$repo_real"/* ]] || { die 'preparation output overlaps checkout'; return 2; }
  mkdir "$output_real" "$output_real/wheelhouse"
  sdist="$output_real/numpy-2.2.2.tar.gz"; src="$output_real/src"; wheel_dir="$output_real/wheelhouse"
  environment="$output_real/venv"; builder="$output_real/build-venv"
  download_sdist "$sdist"
  extract_sdist "$sdist" "$output_real"
  [[ -d "$output_real/numpy-2.2.2" ]] || { die 'NumPy sdist extracted to an unexpected directory'; return 2; }
  mv "$output_real/numpy-2.2.2" "$src"
  assert_build_requirements "$src"
  UV_PROJECT_ENVIRONMENT="$environment" UV_NO_CACHE=1 uv sync --project "$PROJECT" --frozen --no-install-project --no-install-package numpy --python 3.12
  uv venv --python 3.12 "$builder"
  UV_NO_CACHE=1 uv pip install --python "$builder/bin/python" --require-hashes --no-deps -r "$CONSTRAINTS"
  log 'Building exact locked NumPy 2.2.2 with BLAS/LAPACK disabled'
  PATH="$builder/bin:$PATH" uv build --no-build-isolation --python "$builder/bin/python" --wheel --out-dir "$wheel_dir" \
    -C setup-args=-Dblas=none -C setup-args=-Dlapack=none -C setup-args=-Dallow-noblas=true "$src"
  wheel_path="$(find "$wheel_dir" -maxdepth 1 -type f -name 'numpy-2.2.2-*.whl' -print -quit)"
  [[ -n "$wheel_path" && -f "$wheel_path" ]] || { die 'NumPy no-BLAS wheel was not produced'; return 2; }
  inspect_wheel "$wheel_path"
  UV_NO_CACHE=1 uv pip install --python "$environment/bin/python" --no-deps --force-reinstall "$wheel_path"
  local wheel_sha wheel_bytes venv_identity preparer_sha
  wheel_sha="$(sha256sum "$wheel_path" | awk '{print $1}')"
  wheel_bytes="$(stat -c '%s' "$wheel_path")"
  preparer_sha="$(sha256sum "$SCRIPT_PATH" | awk '{print $1}')"
  venv_identity="$(UV_NO_CACHE=1 uv run --no-cache --no-project --python "$environment/bin/python" python - "$environment" <<'PY'
import json
import pathlib
import platform
import sys

expected = pathlib.Path(sys.argv[1]).resolve()
prefix = pathlib.Path(sys.prefix).resolve()
executable = pathlib.Path(sys.executable).absolute()
if prefix != expected or executable != expected / "bin" / "python":
    raise SystemExit("NumPy preparation interpreter is not inside the prepared venv")
print(json.dumps({
    "executable": str(executable),
    "prefix": str(prefix),
    "implementation": platform.python_implementation(),
    "version": platform.python_version(),
}, sort_keys=True, separators=(",", ":")))
PY
)"
  UV_NO_CACHE=1 uv run --no-cache --no-project --python 3.12 python - "$output_real" "$sdist" "$wheel_path" "$environment" "$wheel_sha" "$wheel_bytes" "$preparer_sha" "$venv_identity" <<'PY'
import json
import pathlib
import sys

output, sdist, wheel, venv, wheel_sha, wheel_bytes, preparer_sha, identity = sys.argv[1:]
root = pathlib.Path(output).resolve()
venv_path = pathlib.Path(venv).resolve()
if venv_path.parent != root:
    raise SystemExit("prepared venv is outside preparation output")
document = {
    "schema": "vokra-zonos-numpy-no-blas-preparation-v2",
    "status": "PREPARED_NO_BLAS",
    "publication": "NO_UPLOAD",
    "project": {
        "name": pathlib.Path("tools/parity/zonos_v0_1_reference").name,
        "pyproject_sha256": "__PYPROJECT_SHA256__",
        "uv_lock_sha256": "__LOCK_SHA256__",
        "constraints_sha256": "__CONSTRAINTS_SHA256__",
    },
    "preparer_sha256": preparer_sha,
    "sdist": {
        "url": "__SDIST_URL__",
        "sha256": "__SDIST_SHA256__",
        "bytes": int("__SDIST_BYTES__"),
    },
    "wheel": {
        "basename": pathlib.Path(wheel).name,
        "sha256": wheel_sha,
        "bytes": int(wheel_bytes),
    },
    "build": {
        "no_build_isolation": True,
        "arguments": ["-Dblas=none", "-Dlapack=none", "-Dallow-noblas=true"],
    },
    "venv": {
        "path": str(venv_path),
        "interpreter": json.loads(identity),
    },
}
(root / "preparation.json").write_text(
    json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY
  sed -i \
    -e "s/__PYPROJECT_SHA256__/$PYPROJECT_SHA256/" \
    -e "s/__LOCK_SHA256__/$LOCK_SHA256/" \
    -e "s/__CONSTRAINTS_SHA256__/$CONSTRAINTS_SHA256/" \
    -e "s#__SDIST_URL__#$NUMPY_SDIST_URL#" \
    -e "s/__SDIST_SHA256__/$NUMPY_SDIST_SHA256/" \
    -e "s/__SDIST_BYTES__/$NUMPY_SDIST_BYTES/" \
    "$output_real/preparation.json"
  {
    sha256sum preparation.json "$(basename "$sdist")" "wheelhouse/$(basename "$wheel_path")"
  } > "$output_real/SHA256SUMS"
  log "Prepared no-BLAS NumPy environment: $environment"
}

self_test() {
  local failed=0 token temporary diagnostic
  for token in 'uv sync --project' '--no-install-package numpy' '--require-hashes' '--no-build-isolation' '-Dblas=none' '-Dlapack=none' '-Dallow-noblas=true' 'PREPARED_NO_BLAS' 'NO_UPLOAD' 'NUMPY_SDIST_SHA256' 'preparation.json' 'SHA256SUMS' 'redirect is forbidden' 'assert_build_requirements' 'member.issym()' 'member.islnk()' 'maximum_members' 'maximum_bytes'; do
    grep -Fq -- "$token" "$0" || failed=1
  done
  if temporary="$(mktemp -d "${TMPDIR:-/tmp}/zonos-numpy-selftest.XXXXXX")"; then
    UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python - "$temporary" <<'PY' || failed=1
import pathlib
import sys
import zipfile

root = pathlib.Path(sys.argv[1])
with zipfile.ZipFile(root / "valid.whl", "w") as archive:
    archive.writestr("numpy/core/_multiarray_umath.so", b"ELF")
with zipfile.ZipFile(root / "bad.whl", "w") as archive:
    archive.writestr("numpy.libs/libopenblas.so.0", b"ELF")
PY
    inspect_wheel "$temporary/valid.whl" || failed=1
    if diagnostic="$(inspect_wheel "$temporary/bad.whl" 2>&1)"; then failed=1; elif [[ "$diagnostic" != *'forbidden no-BLAS'* ]]; then failed=1; fi
    UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python - "$temporary" <<'PY' || failed=1
import io
import pathlib
import sys
import tarfile

root = pathlib.Path(sys.argv[1])
maximum_members = 8192
def write_tar(path, member, body):
    with tarfile.open(path, "w:gz") as archive:
        member.size = len(body)
        archive.addfile(member, io.BytesIO(body))

write_tar(root / "safe.tar.gz", tarfile.TarInfo("numpy-2.2.2/LICENSE"), b"x")
with tarfile.open(root / "duplicate.tar.gz", "w:gz") as archive:
    for body in (b"x", b"y"):
        member = tarfile.TarInfo("numpy-2.2.2/LICENSE")
        member.size = 1
        archive.addfile(member, io.BytesIO(body))
write_tar(root / "traversal.tar.gz", tarfile.TarInfo("../LICENSE"), b"x")
link = tarfile.TarInfo("numpy-2.2.2/LICENSE-link")
link.type = tarfile.SYMTYPE
link.linkname = "LICENSE"
with tarfile.open(root / "link.tar.gz", "w:gz") as archive:
    archive.addfile(link)
oversized = tarfile.TarInfo("numpy-2.2.2/LICENSE-big")
write_tar(root / "oversized.tar.gz", oversized, b"x" * (16 * 1024 * 1024 + 1))
def write_member_count(path, count):
    with tarfile.open(path, "w:gz") as archive:
        for index in range(count):
            member = tarfile.TarInfo(f"numpy-2.2.2/member-{index}")
            member.size = 0
            archive.addfile(member)
write_member_count(root / "member-limit.tar.gz", maximum_members)
write_member_count(root / "member-over-limit.tar.gz", maximum_members + 1)
PY
    extract_sdist "$temporary/safe.tar.gz" "$temporary/safe-out" || failed=1
    if extract_sdist "$temporary/duplicate.tar.gz" "$temporary/duplicate-out" >/dev/null 2>&1; then failed=1; fi
    if extract_sdist "$temporary/traversal.tar.gz" "$temporary/traversal-out" >/dev/null 2>&1; then failed=1; fi
    if extract_sdist "$temporary/link.tar.gz" "$temporary/link-out" >/dev/null 2>&1; then failed=1; fi
    if extract_sdist "$temporary/oversized.tar.gz" "$temporary/oversized-out" >/dev/null 2>&1; then failed=1; fi
    extract_sdist "$temporary/member-limit.tar.gz" "$temporary/member-limit-out" || failed=1
    if extract_sdist "$temporary/member-over-limit.tar.gz" "$temporary/member-over-limit-out" >/dev/null 2>&1; then failed=1; fi
    rm -rf "$temporary"
  else
    failed=1
  fi
  (( failed == 0 )) || { log 'self-test FAIL'; return 1; }
  echo 'prepare_numpy_no_blas.sh self-test: OK (model-free, VAST-only, NO_UPLOAD)'
}

if [[ "${1:-}" == --self-test ]]; then
  [[ $# == 1 ]] || die '--self-test accepts no arguments'
  self_test
  exit 0
fi
[[ $# == 2 && "$1" == --output-dir ]] || die 'usage: prepare_numpy_no_blas.sh --output-dir ABSENT_DIR'
prepare "$2"
