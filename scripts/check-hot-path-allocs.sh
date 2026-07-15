#!/usr/bin/env bash
# check-hot-path-allocs.sh — guard the malloc-free decode/encode hot path
# (M1-04 sub-part 3, FR-EX-05).
#
# The Whisper forward pass reuses per-step / per-layer scratch buffers
# (crates/vokra-models/src/whisper/scratch.rs) so the autoregressive decode loop
# allocates nothing at steady state. The functions that implement that loop are
# bracketed with `// ZERO-ALLOC-BEGIN` … `// ZERO-ALLOC-END`. This script fails
# if an allocating construct appears inside any such region:
#
#     vec![ ]        Vec::with_capacity        .to_vec()        .collect()
#
# It is a lightweight regression guard alongside the runtime capacity-stability
# oracle (`scratch_capacity_is_stable_across_decode_steps` in whisper::decoder),
# which is the authoritative proof. Error paths inside a region may still build a
# `format!` string — that is not on the forbidden list (errors are rare, off the
# hot path). Exit code: 0 = clean, 1 = a forbidden construct (or no regions).

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

FILES=(
    "crates/vokra-models/src/whisper/nn.rs"
    "crates/vokra-models/src/whisper/decoder.rs"
    "crates/vokra-models/src/whisper/encoder.rs"
    # M4-03: the AEC process()/DSP-kernel regions (FR-EX-05; the counting-
    # allocator proof lives in crates/vokra-ops/tests/aec_hot_path_alloc.rs).
    "crates/vokra-ops/src/aec.rs"
)

# The forbidden allocating constructs, as a grep ERE. (Matching is done by grep,
# not awk, so the bracket/paren escapes are handled portably; awk only tracks
# which lines fall inside a marked region.)
FORBIDDEN='vec!|Vec::with_capacity|\.to_vec\(\)|\.collect\(\)'

status=0
regions=0

for rel in "${FILES[@]}"; do
    f="$ROOT/$rel"
    if [ ! -f "$f" ]; then
        echo "error: expected source file not found: $rel" >&2
        exit 1
    fi

    # Count marker pairs so an unbalanced BEGIN/END (a refactor that dropped a
    # marker) is caught rather than silently disabling the guard.
    begins="$(grep -c 'ZERO-ALLOC-BEGIN' "$f" || true)"
    ends="$(grep -c 'ZERO-ALLOC-END' "$f" || true)"
    if [ "$begins" != "$ends" ]; then
        echo "error: $rel has $begins ZERO-ALLOC-BEGIN vs $ends ZERO-ALLOC-END markers" >&2
        exit 1
    fi
    regions=$((regions + begins))

    # Emit "file:line: text" for every *code* line inside a marked region (the
    # marker lines themselves only toggle state; pure `//` comment lines are
    # skipped — comments allocate nothing and the markers describe the forbidden
    # tokens in prose), then let grep flag the forbidden constructs among them.
    offenders="$(
        awk -v file="$rel" '
            /ZERO-ALLOC-BEGIN/ { inside = 1; next }
            /ZERO-ALLOC-END/   { inside = 0; next }
            inside && $0 !~ /^[[:space:]]*\/\// { printf "%s:%d: %s\n", file, NR, $0 }
        ' "$f" | grep -E "$FORBIDDEN" || true
    )"
    if [ -n "$offenders" ]; then
        printf '  %s\n' "$offenders"
        status=1
    fi
done

if [ "$regions" -eq 0 ]; then
    echo "error: no ZERO-ALLOC-BEGIN/END regions found — the hot-path guard is not wired" >&2
    exit 1
fi

if [ "$status" -ne 0 ]; then
    echo "error: allocating construct(s) inside a zero-alloc hot region (listed above)." >&2
    echo "       These forward-pass fns must reuse scratch buffers, not allocate" >&2
    echo "       (M1-04 sub-part 3, FR-EX-05). Move the allocation out of the region," >&2
    echo "       or route it through a reused super::scratch field." >&2
    exit 1
fi

echo "check-hot-path-allocs: OK ($regions marked region(s), no forbidden allocations)"
