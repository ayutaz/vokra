#!/usr/bin/env bash
# check-callback-pattern.sh — Grep-based static-analysis for the IL2CPP-safe
# native→C# callback pattern (M2-11-T10 / D5 / R1).
#
# Rules enforced (all violations fail the run):
#   R-STATIC   Every [MonoPInvokeCallback(...)] annotates a `static` method.
#   R-PRESERVE Every [MonoPInvokeCallback(...)] is paired with a [Preserve] attribute
#              on the same method.
#   R-ROOTED   For every delegate T declared with [UnmanagedFunctionPointer] in the
#              file, the file has a `static readonly T` field — the field anchors
#              the delegate instance against the GC (paired with delegate-typed
#              P/Invoke parameters using the same T).
#   R-PAIRED   Every `GCHandle.Alloc(` in a file is paired with at least one
#              `GCHandle.Free(` OR `.Free()` reference in the same file
#              (Dispose / finally / matched teardown).
#
# Portable across macOS bash 3.2 (no mapfile) and BSD awk (no gawk match-with-array).

set -euo pipefail

ROOT="${1:-bindings/unity/com.vokra.unity/Runtime}"

if [ ! -d "$ROOT" ]; then
    echo "check-callback-pattern.sh: root not found: $ROOT" >&2
    exit 2
fi

FAIL=0
COUNT=0

# Portable file iteration — no mapfile.
while IFS= read -r f; do
    COUNT=$((COUNT + 1))

    # Skip files with none of the tracked constructs.
    if ! grep -qE 'MonoPInvokeCallback|UnmanagedFunctionPointer|GCHandle\.Alloc' "$f"; then
        continue
    fi

    # R-STATIC + R-PRESERVE: awk over attribute blocks. Portable to BSD awk.
    # Comment lines (leading `//`) are ignored so doc-comment mentions of the
    # attribute name don't trigger phantom method scans.
    awk -v file="$f" '
        BEGIN { has_mono = 0; has_preserve = 0; ec = 0 }
        /^[[:space:]]*\/\// { next }
        /\[MonoPInvokeCallback[[:space:]]*\(/ {
            has_mono = 1
            has_preserve = 0
            if ($0 ~ /\[Preserve\]/) has_preserve = 1
            next
        }
        has_mono == 1 {
            if ($0 ~ /^[[:space:]]*$/) next
            if ($0 ~ /^[[:space:]]*\/\//) next
            if ($0 ~ /^[[:space:]]*\[/) {
                if ($0 ~ /\[Preserve\]/) has_preserve = 1
                next
            }
            # First non-attribute, non-blank line = method declaration.
            method_line = $0
            if (method_line !~ /[[:space:]]static[[:space:]]/) {
                printf "FAIL[R-STATIC] %s: [MonoPInvokeCallback] on non-static method: %s\n", file, method_line
                ec = 1
            }
            if (has_preserve == 0) {
                printf "FAIL[R-PRESERVE] %s: [MonoPInvokeCallback] missing [Preserve] on: %s\n", file, method_line
                ec = 1
            }
            has_mono = 0
        }
        END { exit ec }
    ' "$f" || FAIL=1

    # R-ROOTED: extract delegate identifiers named after [UnmanagedFunctionPointer].
    # BSD-awk-safe: split() over spaces, find "delegate" then the identifier two
    # tokens later (i.e., "delegate <return-type> <Name>(...)").
    delegates=$(awk '
        /^[[:space:]]*\/\// { next }
        /\[UnmanagedFunctionPointer/ { pending = 1; next }
        pending == 1 && /delegate[[:space:]]/ {
            n = split($0, toks, /[[:space:]]+|\(/)
            for (i = 1; i <= n; i++) {
                if (toks[i] == "delegate" && i + 2 <= n) {
                    name = toks[i + 2]
                    # Strip any trailing punctuation just in case.
                    gsub(/[^A-Za-z0-9_]/, "", name)
                    if (name != "") print name
                    break
                }
            }
            pending = 0
        }
    ' "$f")

    if [ -n "$delegates" ]; then
        # Strip line-comments before the anchor grep so a doc mention like
        # "// no static readonly MyCb field" cannot satisfy the check.
        code_only=$(grep -vE '^[[:space:]]*//' "$f")
        while IFS= read -r delegate_name; do
            [ -z "$delegate_name" ] && continue
            if ! printf '%s\n' "$code_only" | grep -qE "static[[:space:]]+readonly[[:space:]]+${delegate_name}([[:space:]]|;|=)"; then
                echo "FAIL[R-ROOTED] $f: delegate '$delegate_name' has no 'static readonly $delegate_name' anchor field" >&2
                FAIL=1
            fi
        done <<EOF
$delegates
EOF
    fi

    # R-PAIRED: any GCHandle.Alloc requires at least one Free reference in the same file.
    if grep -qE 'GCHandle\.Alloc[[:space:]]*\(' "$f"; then
        if ! grep -qE '\.Free[[:space:]]*\(\)|GCHandle\.Free' "$f"; then
            echo "FAIL[R-PAIRED] $f: GCHandle.Alloc(...) with no matching .Free() in same file" >&2
            FAIL=1
        fi
    fi
done < <(find "$ROOT" -type f -name '*.cs' | sort)

if [ "$COUNT" -eq 0 ]; then
    echo "check-callback-pattern.sh: no C# files under $ROOT" >&2
    exit 2
fi

if [ "$FAIL" -ne 0 ]; then
    echo "check-callback-pattern.sh: violations detected" >&2
    exit 1
fi

echo "check-callback-pattern.sh: OK ($COUNT file(s) scanned under $ROOT)"
