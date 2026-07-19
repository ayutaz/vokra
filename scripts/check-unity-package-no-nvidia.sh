#!/usr/bin/env bash
# check-unity-package-no-nvidia.sh
#
# Mechanical NVIDIA-runtime non-bundle scanner for the com.vokra.unity UPM
# package (M2-11-T12 / plan D7 / risk R3).
#
# Enforces NVIDIA CUDA EULA "installed only in a private (non-shared) directory
# location" — Vokra loads CUDA driver dynamically via dlopen("libcuda.so") /
# LoadLibrary("nvcuda.dll") at runtime; the Unity Asset Store shared plugins
# directory MUST NOT contain cudart/cudnn/cublas/nvrtc/nvcuda/libcuda binaries.
#
# Checks (all failures exit 1):
#   (1) find scan for filenames matching cudart*/cudnn*/cublas*/nvcuda*/
#       libcuda.so*/nvrtc*.dll under Plugins/.
#   (2) For every .dll/.so/.dylib: nm --undefined-only (Linux) / nm -u (macOS)
#       to catch cudart/cudnn/cublasLt symbol references not visible in `strings`.
#   (3) Linux only: readelf -d for DT_NEEDED cudart*.
#
# Usage: bash scripts/check-unity-package-no-nvidia.sh [PKG_DIR]
#   PKG_DIR defaults to bindings/unity/com.vokra.unity.

set -euo pipefail

PKG_DIR="${1:-bindings/unity/com.vokra.unity}"

if [ ! -d "$PKG_DIR" ]; then
    echo "FAIL: package directory not found: $PKG_DIR" >&2
    exit 1
fi

PLUGINS_DIR="$PKG_DIR/Plugins"

if [ ! -d "$PLUGINS_DIR" ]; then
    echo "OK: no NVIDIA runtime bundled (Plugins/ absent — nothing to scan)"
    exit 0
fi

fail=0
os_name="$(uname -s)"

# --- (1) Filename scan ------------------------------------------------------
bad_files="$(find "$PLUGINS_DIR" -type f \
    \( -iname 'cudart*' \
    -o -iname 'cudnn*' \
    -o -iname 'cublas*' \
    -o -iname 'nvcuda*' \
    -o -iname 'libcuda.so*' \
    -o -iname 'nvrtc*.dll' \
    -o -iname 'nvrtc*.so*' \
    -o -iname 'nvrtc*.dylib' \) 2>/dev/null || true)"

if [ -n "$bad_files" ]; then
    echo "FAIL: bundled NVIDIA runtime file(s) detected under $PLUGINS_DIR:" >&2
    printf '  %s\n' $bad_files >&2
    fail=1
fi

# --- (2) Symbol scan --------------------------------------------------------
# Iterate every shared/loadable native library in the package and inspect its
# undefined-symbol table. `strings` catches dlopen path literals, which is
# noise for us (Vokra dlopens libcuda intentionally). We care about *link-time*
# references — those show up as undefined symbols and indicate the shipped
# binary expects a bundled cudart/cudnn/cublasLt to resolve at load time.

# Collect libraries safely (NUL-separated) so paths with spaces survive.
libs=()
while IFS= read -r -d '' f; do
    libs+=("$f")
done < <(find "$PLUGINS_DIR" -type f \
    \( -name '*.dll' -o -name '*.so' -o -name '*.so.*' -o -name '*.dylib' \) \
    -print0 2>/dev/null)

# nvidia_symbol_hits <lib> — prints offending undefined symbols (one per line).
#
# The undefined-symbol dump is captured *once* into a variable and then filtered
# with a plain `grep -E` (no `-q`). Do NOT "simplify" this back to:
#
#     if nm -u "$f" 2>/dev/null | grep -qE '_(cudart|...)'; then
#
# Under `set -o pipefail` that idiom FAILS OPEN. `grep -q` exits at the first
# match; `nm` is still writing, so it dies of SIGPIPE (141); pipefail promotes
# 141 to the pipeline status; and the `if` therefore takes the *false* branch —
# the scanner reports "no NVIDIA runtime bundled" for a library that *does*
# reference cudart. It only triggers once nm's output exceeds the ~64 KiB pipe
# buffer, so small libraries are scanned correctly and large ones silently pass.
# `grep -E` without -q reads to EOF, so the producer never sees SIGPIPE.
#
# Regression test: scripts/compliance/test-nvidia-scanner-sigpipe.sh
# Same idiom as scan_symbols() in compliance/check-cpu-vulkan-only-no-nvidia.sh.
nvidia_symbol_hits() {
    local f="$1" dump="" re=""
    case "$os_name" in
        Darwin)
            # macOS nm; -u lists undefined symbols. Mangled C symbols on
            # Mach-O carry a leading underscore.
            dump="$(nm -u "$f" 2>/dev/null || true)"
            re='_(cudart|cudnn|cublasLt|cublas)'
            ;;
        Linux | *)
            # GNU nm; --undefined-only prints only undefined symbols.
            command -v nm >/dev/null 2>&1 || return 0
            dump="$(nm --undefined-only "$f" 2>/dev/null || true)"
            re='(cudart|cudnn|cublasLt|cublas)'
            ;;
    esac
    [ -n "$dump" ] || return 0
    printf '%s\n' "$dump" | grep -E "$re" || true
}

for f in "${libs[@]:-}"; do
    [ -z "$f" ] && continue
    # No `strings` prefilter: Mach-O symbol tables (LC_SYMTAB) are not exposed
    # by macOS `strings` even with -a, so a strings-based skip would silently
    # miss real cudart symbol references. Run nm on every native library
    # unconditionally — UPM packages carry only a handful of libs, so cost
    # is negligible.
    hits="$(nvidia_symbol_hits "$f")"
    if [ -n "$hits" ]; then
        echo "FAIL: $f references NVIDIA runtime symbols (undefined-symbol scan)" >&2
        # `sed -n '1,5p'` not `head -5`: head exits early, which would SIGPIPE
        # the printf and abort this script under `set -e` + pipefail.
        printf '%s\n' "$hits" | sed -n '1,5p' >&2
        fail=1
    fi
done

# --- (3) readelf DT_NEEDED (Linux only) ------------------------------------
if [ "$os_name" = "Linux" ] && command -v readelf >/dev/null 2>&1; then
    for f in "${libs[@]:-}"; do
        [ -z "$f" ] && continue
        case "$f" in
            *.so|*.so.*)
                # Captured, not `| grep -q` — see nvidia_symbol_hits() above for
                # why a pipefail'd `grep -q` fails open. `readelf -d` output is
                # small enough that it is unlikely to trip in practice, but this
                # gate must not have a size-dependent blind spot anywhere.
                needed="$(readelf -d "$f" 2>/dev/null | grep -E 'NEEDED' || true)"
                dt_hits="$(printf '%s\n' "$needed" | grep -E 'cudart|cudnn|cublas' || true)"
                if [ -n "$dt_hits" ]; then
                    echo "FAIL: $f has DT_NEEDED entry for NVIDIA runtime" >&2
                    printf '%s\n' "$dt_hits" >&2
                    fail=1
                fi
                ;;
        esac
    done
fi

if [ "$fail" -ne 0 ]; then
    echo "" >&2
    echo "NVIDIA CUDA runtime libraries MUST NOT be bundled in com.vokra.unity." >&2
    echo "Vokra loads libcuda / nvcuda dynamically at runtime; consumers must" >&2
    echo "install the CUDA Toolkit separately per NVIDIA EULA. See NOTICE." >&2
    exit 1
fi

echo "OK: no NVIDIA runtime bundled (scanned ${#libs[@]} native libraries under $PLUGINS_DIR)"
