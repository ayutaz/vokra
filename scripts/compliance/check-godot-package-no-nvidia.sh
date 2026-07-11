#!/usr/bin/env bash
# check-godot-package-no-nvidia.sh
#
# Mechanical NVIDIA-runtime non-bundle scanner + LICENSE / NOTICE presence
# check for the Godot 4.x AssetLib package produced by
# scripts/build-godot-gdextension.sh (M3-11-T18 / ADR-0011 §D7 / NFR-LC-01).
#
# Enforces NVIDIA CUDA EULA "installed only in a private (non-shared) directory
# location" — Vokra loads CUDA driver dynamically via dlopen("libcuda.so") /
# LoadLibrary("nvcuda.dll") at runtime; the Godot AssetLib addons/ directory
# is a shared consumer plugin location and MUST NOT contain
# cudart/cudnn/cublas/nvrtc/nvcuda/libcuda binaries.
#
# Mirror of scripts/check-unity-package-no-nvidia.sh (M2-11-T12), adapted for
# the Godot layout described in ADR-0011 §D9:
#     addons/vokra/vokra.gdextension
#     addons/vokra/bin/<platform>/<arch>/{libvokra_godot.{so,dylib},vokra_godot.dll}
#     addons/vokra/LICENSE     (Apache-2.0 verbatim)
#     addons/vokra/NOTICE      (dep NOTICE roll-up + Godot MIT credit)
#
# Checks (all failures exit 1):
#   (1) find scan for filenames matching cudart*/cudnn*/cublas*/nvcuda*/
#       libcuda.so*/nvrtc*.dll under addons/vokra/bin/.
#   (2) For every .dll/.so/.dylib: nm --undefined-only (Linux) / nm -u (macOS)
#       to catch cudart/cudnn/cublasLt symbol references not visible in `strings`.
#   (3) Linux only: readelf -d for DT_NEEDED cudart*.
#   (4) LICENSE + NOTICE presence under addons/vokra/ (NFR-LC-01,
#       ADR-0011 §D9). Empty LICENSE / NOTICE = fail (a license-less
#       AssetLib package would infect downstream projects with
#       undefined redistribution terms).
#
# Usage: bash scripts/compliance/check-godot-package-no-nvidia.sh [PKG_DIR]
#   PKG_DIR defaults to dist/godot/vokra-godot.

set -euo pipefail

PKG_DIR="${1:-dist/godot/vokra-godot}"

if [ ! -d "$PKG_DIR" ]; then
    echo "FAIL: package directory not found: $PKG_DIR" >&2
    exit 1
fi

ADDON_DIR="$PKG_DIR/addons/vokra"

if [ ! -d "$ADDON_DIR" ]; then
    echo "FAIL: expected addons/vokra/ tree missing under $PKG_DIR" >&2
    echo "      (build-godot-gdextension.sh --pack should have created it)" >&2
    exit 1
fi

BIN_DIR="$ADDON_DIR/bin"

fail=0
os_name="$(uname -s)"

# --- (1) Filename scan ------------------------------------------------------
# If bin/ is absent nothing is bundled — skip the binary scan but still
# validate LICENSE / NOTICE below. A missing bin/ is legitimate when the
# package was built in --no-build mode against a fresh checkout.
if [ -d "$BIN_DIR" ]; then
    # Match both Windows-style (`cudart64_12.dll`) and Unix-style
    # (`libcudart.so.12`, `libcudart.12.dylib`) filenames. The Unity mirror's
    # `cudart*` alone would silently miss the `lib`-prefixed Linux/macOS
    # names — a real bug we do NOT propagate here (M3-11-T18 spec: "パッケージ
    # LICENSE + NOTICE が正しく配置される" — same test rigor applied to the
    # NVIDIA scan). We list explicit `lib`-prefixed alternates rather than
    # `*cudart*` glob-on-both-sides because a broad substring match would
    # false-positive on unrelated files (e.g. `mycudartool.dll`).
    bad_files="$(find "$BIN_DIR" -type f \
        \( -iname 'cudart*'       -o -iname 'libcudart*' \
        -o -iname 'cudnn*'        -o -iname 'libcudnn*' \
        -o -iname 'cublas*'       -o -iname 'libcublas*' \
        -o -iname 'nvcuda*' \
        -o -iname 'libcuda.so*' \
        -o -iname 'nvrtc*.dll'    -o -iname 'libnvrtc*' \
        -o -iname 'nvrtc*.so*' \
        -o -iname 'nvrtc*.dylib' \) 2>/dev/null || true)"

    if [ -n "$bad_files" ]; then
        echo "FAIL: bundled NVIDIA runtime file(s) detected under $BIN_DIR:" >&2
        printf '  %s\n' $bad_files >&2
        fail=1
    fi

    # --- (2) Symbol scan ----------------------------------------------------
    # Iterate every shared/loadable native library in the addon bin/ tree and
    # inspect its undefined-symbol table. `strings` catches dlopen path
    # literals, which is noise for us (Vokra dlopens libcuda intentionally).
    # We care about *link-time* references — those show up as undefined
    # symbols and indicate the shipped binary expects a bundled
    # cudart/cudnn/cublasLt to resolve at load time.
    libs=()
    while IFS= read -r -d '' f; do
        libs+=("$f")
    done < <(find "$BIN_DIR" -type f \
        \( -name '*.dll' -o -name '*.so' -o -name '*.so.*' -o -name '*.dylib' \) \
        -print0 2>/dev/null)

    for f in "${libs[@]:-}"; do
        [ -z "$f" ] && continue
        # No `strings` prefilter: Mach-O symbol tables (LC_SYMTAB) are not
        # exposed by macOS `strings` even with -a, so a strings-based skip
        # would silently miss real cudart symbol references. Run nm on every
        # native library unconditionally — AssetLib packages carry only a
        # handful of libs, so cost is negligible.
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

    # --- (3) readelf DT_NEEDED (Linux only) ---------------------------------
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
    lib_count="${#libs[@]:-0}"
else
    echo "note: no bin/ under $ADDON_DIR (--no-build or empty stage) — skipping binary scan"
    lib_count=0
fi

# --- (4) LICENSE + NOTICE presence (NFR-LC-01 / ADR-0011 §D9) --------------
# The AssetLib package MUST ship LICENSE (Apache-2.0) + NOTICE (dep NOTICE
# roll-up + Godot MIT credit) alongside the .gdextension config. A missing
# or zero-byte LICENSE / NOTICE would make the package license-less and
# unsafe to redistribute.
for f in LICENSE NOTICE; do
    if [ ! -f "$ADDON_DIR/$f" ]; then
        echo "FAIL: $ADDON_DIR/$f missing (NFR-LC-01)" >&2
        fail=1
    elif [ ! -s "$ADDON_DIR/$f" ]; then
        echo "FAIL: $ADDON_DIR/$f is empty (NFR-LC-01)" >&2
        fail=1
    fi
done

# vokra.gdextension is the config Godot reads on load; a missing config
# means the addon doesn't register any class in the Editor. Not strictly a
# license check, but bundled here so the whole compliance gate is a single
# invocation from CI.
if [ ! -f "$ADDON_DIR/vokra.gdextension" ]; then
    echo "FAIL: $ADDON_DIR/vokra.gdextension missing (ADR-0011 §D9)" >&2
    fail=1
fi

if [ "$fail" -ne 0 ]; then
    echo "" >&2
    echo "NVIDIA CUDA runtime libraries MUST NOT be bundled in the Vokra Godot" >&2
    echo "AssetLib package. Vokra loads libcuda / nvcuda dynamically at runtime;" >&2
    echo "consumers must install the CUDA Toolkit separately per NVIDIA EULA." >&2
    echo "See NOTICE + third_party/NVIDIA-EULA.md." >&2
    exit 1
fi

echo "OK: no NVIDIA runtime bundled + LICENSE / NOTICE / vokra.gdextension present (scanned ${lib_count} native libraries under $BIN_DIR)"
