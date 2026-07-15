#!/usr/bin/env bash
# gen-notice-cpu-vulkan-only.sh — derive the NOTICE variant bundled with the
# CPU + Vulkan-only build target (M4-15-T05, ADR M4-15 §(g), NFR-LG-04,
# NFR-PT-04).
#
# WHY A DERIVED FILE (and not a second committed NOTICE):
#   NFR-PT-04 fixes the single-source principle — Full / build-target
#   variants derive from one source by feature flags, never by a parallel
#   tree. A committed NOTICE.cpu-vulkan-only would silently drift every
#   time the root NOTICE gains a section. This script therefore derives
#   the variant AT BUILD TIME from the root NOTICE:
#
#     1. DROP the "NVIDIA CUDA / cuDNN / cuBLAS runtime dependencies"
#        section — the CUDA backend is not compiled into this build
#        target (`--no-default-features --features vulkan`), so the
#        NVIDIA runtime record (and its pointer to the EULA file under
#        third_party/) is structurally inapplicable and MUST not ship
#        with the artifact (NFR-LG-04 applies to CUDA-capable builds).
#     2. KEEP every other section verbatim (model / algorithm
#        attributions are backend-independent) and renumber contiguously.
#     3. PREPEND a variant note naming the build target in the neutral
#        wording "CPU + Vulkan-only build target" (ADR M4-15 §(e)).
#
# FAIL-LOUD (FR-EX-08 spirit): if the input NOTICE does not contain the
# NVIDIA section, or no numbered sections parse at all, the derivation
# rule has gone stale — exit 1 instead of writing something plausible.
#
# The output is a pure function of the input file (no timestamps, no
# environment), so two runs are byte-identical — the same reproducibility
# property the SBOM generator guarantees (ADR M4-15 §(c)).
#
# Usage:
#   scripts/gen-notice-cpu-vulkan-only.sh [--notice PATH] [--out PATH]
#     --notice PATH   input NOTICE (default: <repo root>/NOTICE)
#     --out PATH      output file (default: stdout)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NOTICE_IN="$ROOT/NOTICE"
OUT=""

while [ $# -gt 0 ]; do
    case "$1" in
        --notice)
            [ $# -ge 2 ] || {
                echo "error: --notice needs a path" >&2
                exit 1
            }
            NOTICE_IN="$2"
            shift 2
            ;;
        --out)
            [ $# -ge 2 ] || {
                echo "error: --out needs a path" >&2
                exit 1
            }
            OUT="$2"
            shift 2
            ;;
        -h | --help)
            sed -n '2,36p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "error: unknown argument: $1 (see --help)" >&2
            exit 1
            ;;
    esac
done

if [ ! -f "$NOTICE_IN" ]; then
    echo "error: input NOTICE not found: $NOTICE_IN" >&2
    exit 1
fi

# The dropped section is identified by its exact heading text in the root
# NOTICE. index()-based literal match (no regex metacharacter surprises).
DROP_TITLE="NVIDIA CUDA / cuDNN / cuBLAS runtime dependencies"

derived="$(awk -v drop_title="$DROP_TITLE" '
    # A section header is:  <dash-line> / "N. Title" / <dash-line>.
    function is_dashline(s) { return s ~ /^-{10,}$/ }

    { lines[NR] = $0 }

    END {
        n = NR
        # ---- locate section headers -----------------------------------
        nsec = 0
        for (i = 1; i + 2 <= n; i++) {
            if (is_dashline(lines[i]) && lines[i + 1] ~ /^[0-9]+\. / \
                && is_dashline(lines[i + 2])) {
                nsec++
                sec_start[nsec] = i
                sec_title[nsec] = lines[i + 1]
            }
        }
        if (nsec == 0) {
            print "error: no numbered sections parsed from input NOTICE" > "/dev/stderr"
            exit 1
        }
        for (s = 1; s <= nsec; s++)
            sec_end[s] = (s < nsec) ? sec_start[s + 1] - 1 : n

        # ---- find the section to drop (fail loud if missing) ----------
        drop = 0
        for (s = 1; s <= nsec; s++)
            if (index(sec_title[s], drop_title) > 0) drop = s
        if (drop == 0) {
            print "error: NVIDIA runtime section not found in input NOTICE —" > "/dev/stderr"
            print "       the M4-15 derivation rule is stale; refusing to emit" > "/dev/stderr"
            print "       a variant silently (ADR M4-15 §(g) fail-loud)." > "/dev/stderr"
            exit 1
        }

        # ---- preamble (everything before the first section) -----------
        for (i = 1; i < sec_start[1]; i++) print lines[i]

        # ---- variant note (neutral wording — ADR M4-15 §(e)) ----------
        print "--------------------------------------------------------------------------"
        print "About this NOTICE variant (CPU + Vulkan-only build target)"
        print "--------------------------------------------------------------------------"
        print "This NOTICE accompanies artifacts of the CPU + Vulkan-only build"
        print "target (built with `--no-default-features --features vulkan`). The"
        print "CUDA backend is not compiled into these artifacts, so the root"
        print "NOTICE section recording NVIDIA runtime dependency decisions does"
        print "not apply here and is omitted: no NVIDIA runtime component is"
        print "linked, loaded, or bundled by this build. All remaining sections"
        print "are backend-independent model and algorithm attributions and apply"
        print "unchanged. The Vulkan loader is likewise not bundled — it is"
        print "loaded at runtime from the system installation — and the SPIR-V"
        print "compute shaders are first-party sources from this repository."
        print ""

        # ---- remaining sections, renumbered ----------------------------
        outno = 0
        for (s = 1; s <= nsec; s++) {
            if (s == drop) continue
            outno++
            for (i = sec_start[s]; i <= sec_end[s]; i++) {
                if (i == sec_start[s] + 1) {
                    title = lines[i]
                    sub(/^[0-9]+\./, outno ".", title)
                    print title
                } else {
                    print lines[i]
                }
            }
        }
    }
' "$NOTICE_IN")"

if [ -n "$OUT" ]; then
    printf '%s\n' "$derived" >"$OUT"
else
    printf '%s\n' "$derived"
fi
