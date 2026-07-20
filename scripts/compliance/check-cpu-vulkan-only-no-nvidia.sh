#!/usr/bin/env bash
# check-cpu-vulkan-only-no-nvidia.sh
#
# Mechanical exclusion scanner for the CPU + Vulkan-only build target
# (M4-15-T04, ADR M4-15, FR-BE-09 partial: build target only; NFR-LG-04,
# NFR-LC-01). Verifies that the assembled artifact directory (cdylib +
# LICENSE + NOTICE variant) contains NO trace of the excluded backends:
#
#   * NVIDIA runtime (cudart / cudnn / cublas / nvrtc / nvcuda / libcuda) —
#     neither as bundled files nor as link-time symbol references. The
#     dlopen install model means a correct Vokra artifact NEVER carries an
#     undefined NVIDIA symbol; in the CPU + Vulkan-only target the CUDA
#     backend is not even compiled, so any hit is a build-system bug.
#   * Metal / Objective-C runtime (objc_msgSend / MTL*) — the Metal backend
#     is equally excluded from this target.
#
# Mirror of scripts/check-unity-package-no-nvidia.sh (M2-11-T12) and
# scripts/compliance/check-godot-package-no-nvidia.sh (M3-11-T18), adapted
# to the flat artifact-dir layout assembled by the `build-target-vulkan-only`
# CI job:
#     <ARTIFACT_DIR>/{libvokra.so|libvokra.dylib|vokra.dll}
#     <ARTIFACT_DIR>/LICENSE            (Apache-2.0, common to all targets)
#     <ARTIFACT_DIR>/NOTICE             (T05 variant — NVIDIA section dropped)
#     <ARTIFACT_DIR>/*.spdx.json        (T06/T07 SBOM, optional at scan time)
#
# The filename tier inherits the Godot scanner's fix for the Unity mirror's
# latent gap: `cudart*` alone silently misses the `lib`-prefixed Linux /
# macOS names (libcudart.so.12), so the lib-prefixed alternates are matched
# explicitly.
#
# Checks (any failure exits 1 — FR-EX-08 rigor, no silent pass):
#   (1) filename scan: no NVIDIA runtime file bundled anywhere in the dir.
#   (2) undefined-symbol scan (`nm -u` on macOS / `nm --undefined-only`
#       elsewhere) on every native library:
#         - NVIDIA: cudart* / cuda[A-Z]* (runtime API) / cu[A-Z]* (driver
#           API) / cudnn* / cublas* (incl. cublasLt) / nvrtc*
#         - Metal/ObjC: objc_msgSend* / MTL[A-Z]* / OBJC_CLASS_$_MTL*
#       Symbols are matched against the LAST whitespace field of the nm
#       output, anchored at the start (optional Mach-O leading underscore),
#       so unrelated symbols that merely contain "cu" cannot false-positive.
#   (3) Linux only: readelf -d DT_NEEDED must not name
#       libcuda*/libcudart*/libcudnn*/libcublas*/libnvrtc*/libmetal*
#       (dlopen never produces DT_NEEDED — belt-and-suspenders).
#   (4) LICENSE + NOTICE presence, non-empty, and NOTICE CONTENT:
#         - must NOT contain the root NOTICE's NVIDIA runtime section
#           heading (the variant, not the root file, ships here),
#         - must NOT reference "NVIDIA-EULA",
#         - must retain the backend-independent attributions
#           (BigVGAN / pocketfft / piper-plus / Mimi / EnCodec / SpeexDSP).
#       At least one native library must be present — an empty stage is a
#       broken pipeline, not a pass.
#
# Usage: bash scripts/compliance/check-cpu-vulkan-only-no-nvidia.sh [ARTIFACT_DIR]
#   ARTIFACT_DIR defaults to dist/cpu-vulkan-only.

set -euo pipefail

ART_DIR="${1:-dist/cpu-vulkan-only}"

if [ ! -d "$ART_DIR" ]; then
    echo "FAIL: artifact directory not found: $ART_DIR" >&2
    exit 1
fi

fail=0
os_name="$(uname -s)"

# --- (1) Filename scan ------------------------------------------------------
bad_files="$(find "$ART_DIR" -type f \
    \( -iname 'cudart*' -o -iname 'libcudart*' \
    -o -iname 'cudnn*' -o -iname 'libcudnn*' \
    -o -iname 'cublas*' -o -iname 'libcublas*' \
    -o -iname 'nvcuda*' \
    -o -iname 'libcuda.so*' \
    -o -iname 'nvrtc*.dll' -o -iname 'libnvrtc*' \
    -o -iname 'nvrtc*.so*' \
    -o -iname 'nvrtc*.dylib' \) 2>/dev/null || true)"

if [ -n "$bad_files" ]; then
    echo "FAIL: bundled NVIDIA runtime file(s) detected under $ART_DIR:" >&2
    printf '  %s\n' $bad_files >&2
    fail=1
fi

# --- collect native libraries ------------------------------------------------
libs=()
while IFS= read -r -d '' f; do
    libs+=("$f")
done < <(find "$ART_DIR" -type f \
    \( -name '*.dll' -o -name '*.so' -o -name '*.so.*' -o -name '*.dylib' \) \
    -print0 2>/dev/null)

if [ "${#libs[@]}" -eq 0 ]; then
    echo "FAIL: no native library (*.so / *.dylib / *.dll) under $ART_DIR —" >&2
    echo "      the CPU + Vulkan-only artifact stage is empty (broken assembly)." >&2
    fail=1
fi

# --- (2) Undefined-symbol scan ----------------------------------------------
# Anchored at the start of the symbol name (last nm field), with the
# optional Mach-O leading underscore. cu[A-Z] catches the CUDA Driver API
# (cuInit / cuLaunchKernel), cuda[A-Z] the runtime API (cudaMalloc);
# neither matches innocent symbols that merely contain "cu".
NVIDIA_SYM_RE='^_?(cu[A-Z]|cuda[A-Z]|cudart|cudnn|cublas|nvrtc)'
METAL_SYM_RE='^_?(objc_msgSend|MTL[A-Z]|OBJC_CLASS_\$_MTL)'
# CoreML (M5-01, FR-BE-09): the CoreML delegate backend is framework-linked
# like Metal — unlike the dlopen'd CUDA — so a CPU + Vulkan-only build that
# accidentally pulled it in would leave undefined CoreML symbols behind, not a
# runtime dlopen. Match the CoreML C entry point we call (MLAllComputeDevices)
# and the ObjC class symbols (`_OBJC_CLASS_$_ML<Upper>`). `ML[A-Z]` needs M then
# L then an upper — disjoint from Metal's `MTL` (M then T), so the two tiers do
# not overlap. `objc_msgSend` is already caught by METAL_SYM_RE (both backends
# share the ObjC runtime), so it is not repeated here.
COREML_SYM_RE='^_?(MLAllComputeDevices|OBJC_CLASS_\$_ML[A-Z])'

scan_symbols() {
    # scan_symbols <lib> — prints offending symbols (one per line).
    local f="$1" dump=""
    case "$os_name" in
        Darwin) dump="$(nm -u "$f" 2>/dev/null || true)" ;;
        *) dump="$(nm --undefined-only "$f" 2>/dev/null || true)" ;;
    esac
    [ -n "$dump" ] || return 0
    printf '%s\n' "$dump" | awk '{print $NF}' \
        | grep -E "$NVIDIA_SYM_RE|$METAL_SYM_RE|$COREML_SYM_RE" || true
}

for f in "${libs[@]:-}"; do
    [ -z "$f" ] && continue
    hits="$(scan_symbols "$f")"
    if [ -n "$hits" ]; then
        echo "FAIL: $f references excluded-backend symbols (undefined-symbol scan):" >&2
        printf '%s\n' "$hits" | head -5 >&2
        fail=1
    fi
done

# --- (3) readelf DT_NEEDED (Linux only) ---------------------------------------
if [ "$os_name" = "Linux" ] && command -v readelf >/dev/null 2>&1; then
    for f in "${libs[@]:-}"; do
        [ -z "$f" ] && continue
        case "$f" in
            *.so | *.so.*)
                # Captured, not `| grep -q`: under `set -o pipefail` a `grep -q`
                # that matches early can SIGPIPE the producer, whose 141 becomes
                # the pipeline status and flips the `if` to false — i.e. the gate
                # would report "clean" for a library that *does* link NVIDIA.
                # scan_symbols() above already uses the capture idiom; this is
                # the same rule applied to the DT_NEEDED leg.
                needed="$(readelf -d "$f" 2>/dev/null | grep -E 'NEEDED' || true)"
                dt_hits="$(printf '%s\n' "$needed" \
                    | grep -E 'libcuda|libcudart|libcudnn|libcublas|libnvrtc|libmetal' || true)"
                if [ -n "$dt_hits" ]; then
                    echo "FAIL: $f has a DT_NEEDED entry for an excluded backend:" >&2
                    printf '%s\n' "$dt_hits" >&2
                    fail=1
                fi
                ;;
        esac
    done
fi

# --- (4) LICENSE + NOTICE presence and NOTICE variant content ------------------
for f in LICENSE NOTICE; do
    if [ ! -f "$ART_DIR/$f" ]; then
        echo "FAIL: $ART_DIR/$f missing (NFR-LC-01)" >&2
        fail=1
    elif [ ! -s "$ART_DIR/$f" ]; then
        echo "FAIL: $ART_DIR/$f is empty (NFR-LC-01)" >&2
        fail=1
    fi
done

if [ -s "$ART_DIR/NOTICE" ]; then
    if grep -q "NVIDIA CUDA / cuDNN / cuBLAS runtime dependencies" "$ART_DIR/NOTICE"; then
        echo "FAIL: $ART_DIR/NOTICE still carries the NVIDIA runtime section —" >&2
        echo "      the root NOTICE was shipped instead of the T05 variant" >&2
        echo "      (scripts/gen-notice-cpu-vulkan-only.sh)." >&2
        fail=1
    fi
    if grep -q "NVIDIA-EULA" "$ART_DIR/NOTICE"; then
        echo "FAIL: $ART_DIR/NOTICE references NVIDIA-EULA — inapplicable to the" >&2
        echo "      CPU + Vulkan-only build target (ADR M4-15 §(g))." >&2
        fail=1
    fi
    for marker in "BigVGAN" "pocketfft" "piper-plus" "Mimi codec" "EnCodec" "SpeexDSP echo canceller"; do
        if ! grep -q "$marker" "$ART_DIR/NOTICE"; then
            echo "FAIL: $ART_DIR/NOTICE lost the backend-independent attribution: $marker" >&2
            fail=1
        fi
    done
fi

if [ "$fail" -ne 0 ]; then
    echo "" >&2
    echo "The CPU + Vulkan-only build target must contain no NVIDIA runtime and" >&2
    echo "no Metal/ObjC linkage (both backends are excluded by feature selection" >&2
    echo "— ADR M4-15 §(a)). A hit here means the feature plumbing or the" >&2
    echo "artifact assembly regressed. See NOTICE (variant) and ADR M4-15." >&2
    exit 1
fi

echo "OK: no NVIDIA / Metal linkage + LICENSE / NOTICE variant present (scanned ${#libs[@]} native libraries under $ART_DIR)"
