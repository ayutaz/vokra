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

for f in "${libs[@]:-}"; do
    [ -z "$f" ] && continue
    # No `strings` prefilter: Mach-O symbol tables (LC_SYMTAB) are not exposed
    # by macOS `strings` even with -a, so a strings-based skip would silently
    # miss real cudart symbol references. Run nm on every native library
    # unconditionally — UPM packages carry only a handful of libs, so cost
    # is negligible.
    case "$os_name" in
        Darwin)
            # macOS nm; -u lists undefined symbols. Mangled C symbols on
            # Mach-O carry a leading underscore.
            if nm -u "$f" 2>/dev/null | grep -qE '_(cudart|cudnn|cublasLt|cublas)'; then
                echo "FAIL: $f references NVIDIA runtime symbols (nm -u)" >&2
                nm -u "$f" 2>/dev/null | grep -E '_(cudart|cudnn|cublasLt|cublas)' | head -5 >&2
                fail=1
            fi
            ;;
        Linux|*)
            # GNU nm; --undefined-only prints only undefined symbols.
            if command -v nm >/dev/null 2>&1; then
                if nm --undefined-only "$f" 2>/dev/null | grep -qE '(cudart|cudnn|cublasLt|cublas)'; then
                    echo "FAIL: $f references NVIDIA runtime symbols (nm --undefined-only)" >&2
                    nm --undefined-only "$f" 2>/dev/null | grep -E '(cudart|cudnn|cublasLt|cublas)' | head -5 >&2
                    fail=1
                fi
            fi
            ;;
    esac
done

# --- (3) readelf DT_NEEDED (Linux only) ------------------------------------
if [ "$os_name" = "Linux" ] && command -v readelf >/dev/null 2>&1; then
    for f in "${libs[@]:-}"; do
        [ -z "$f" ] && continue
        case "$f" in
            *.so|*.so.*)
                if readelf -d "$f" 2>/dev/null | grep -E 'NEEDED' | grep -qE 'cudart|cudnn|cublas'; then
                    echo "FAIL: $f has DT_NEEDED entry for NVIDIA runtime" >&2
                    readelf -d "$f" 2>/dev/null | grep -E 'NEEDED' | grep -E 'cudart|cudnn|cublas' >&2
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
