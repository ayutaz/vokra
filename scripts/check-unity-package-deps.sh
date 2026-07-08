#!/usr/bin/env bash
# check-unity-package-deps.sh — assert the Unity UPM package declares no
# external runtime dependencies (M2-11-T13, plan §R2).
#
# The Vokra Unity package is a single-cdylib runtime plus C# wrapper. It
# MUST NOT pull in any external UPM package at install time — most
# critically NOT `com.unity.sentis` (competing ML runtime) or
# `com.unity.ml-agents`, both of which would change Vokra's competitive
# story ("standalone alternative to Sentis") and drag in transitive deps
# that Vokra deliberately avoids (Barracuda / ONNX assemblies, per
# NFR-DS-02 and CLAUDE.md).
#
# Two assertions:
#   1. `dependencies` field is either absent or an empty object `{}`.
#      An empty object is the canonical form; absent is equivalent because
#      UPM defaults to `{}` for a missing `dependencies` key.
#   2. No mention of `com.unity.sentis`, `com.unity.barracuda`,
#      `com.unity.ml-agents`, or `onnxruntime` anywhere in package.json —
#      catches drift where someone adds a dependency via a different
#      spelling (e.g. inside `samples[]` or a stray comment).
#
# Companion of `scripts/check-unity-package-no-nvidia.sh` (which does the
# NVIDIA-runtime NON-bundling check under Plugins/*).
#
# Usage:
#   scripts/check-unity-package-deps.sh <path/to/com.vokra.unity>
#
# Exit 0 = both assertions hold; non-zero on the first failure.

set -euo pipefail

PKG="${1:-bindings/unity/com.vokra.unity}"
PKG_JSON="$PKG/package.json"

if [ ! -f "$PKG_JSON" ]; then
    echo "check-unity-package-deps: FAIL missing $PKG_JSON" >&2
    exit 1
fi

echo "== check-unity-package-deps $PKG_JSON =="

# (1) dependencies is absent or {} — read via python3's json module because
#     jq is not guaranteed on GitHub-hosted Windows runners and the value
#     needs semantic (not syntactic) comparison.
if ! command -v python3 >/dev/null 2>&1; then
    echo "check-unity-package-deps: FAIL python3 is required to parse package.json" >&2
    exit 1
fi

DEP_STATE="$(python3 - "$PKG_JSON" <<'PY'
import json, sys
with open(sys.argv[1], "r", encoding="utf-8") as f:
    doc = json.load(f)
deps = doc.get("dependencies")
if deps is None:
    print("absent")
elif isinstance(deps, dict) and len(deps) == 0:
    print("empty")
else:
    print("nonempty:" + json.dumps(deps))
PY
)"

case "$DEP_STATE" in
    absent)
        echo "  dependencies: absent (equivalent to {}); OK"
        ;;
    empty)
        echo "  dependencies: {} (canonical empty); OK"
        ;;
    nonempty:*)
        echo "check-unity-package-deps: FAIL $PKG_JSON declares external UPM deps: ${DEP_STATE#nonempty:}" >&2
        echo "  Vokra UPM must remain standalone (no com.unity.sentis, no com.unity.ml-agents, etc.)." >&2
        echo "  Plan §R2 / NFR-DS-02: no external UPM runtime dependencies." >&2
        exit 1
        ;;
    *)
        echo "check-unity-package-deps: FAIL unexpected DEP_STATE '$DEP_STATE' (parser bug?)" >&2
        exit 1
        ;;
esac

# (2) forbidden-substring scan across the whole package.json — catches
#     drift where a dep is smuggled in via samples[], custom fields, or a
#     comment.
FORBIDDEN=(
    "com.unity.sentis"
    "com.unity.barracuda"
    "com.unity.ml-agents"
    "onnxruntime"
    "onnx-runtime"
    "Microsoft.ML"
)

fail=0
for needle in "${FORBIDDEN[@]}"; do
    if grep -qF "$needle" "$PKG_JSON"; then
        echo "check-unity-package-deps: FAIL $PKG_JSON mentions '$needle'" >&2
        grep -nF "$needle" "$PKG_JSON" >&2 || true
        fail=1
    fi
done

if [ "$fail" -ne 0 ]; then
    echo "  Vokra is positioned as an alternative to Sentis / ONNX Runtime — the UPM package must not name them." >&2
    exit 1
fi
echo "  no forbidden substrings (${#FORBIDDEN[@]} names checked): OK"

echo "check-unity-package-deps: all checks passed"
