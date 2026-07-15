#!/usr/bin/env bash
# check-unity-webgl-traps.sh — keep the two VokraAndroidAssets WebGL latent
# traps fixed (M4-02-T07, spec 起票時発見 1/2).
#
# Trap 1: the non-Android fallthrough of EnsureLocalCopy used to return
#   Application.streamingAssetsPath verbatim — on WebGL that is an HTTP URL
#   which fopen cannot open. The fix: WebGL has its own explicit branch.
# Trap 2: the Android synchronous busy-wait (`while (!op.isDone) { }`) would
#   deadlock the WebGL main thread if copied into a WebGL branch. The fix:
#   the synchronous EnsureLocalCopy throws NotSupportedException on WebGL
#   (loud fail, FR-EX-08 — never a hang), and only async paths fetch.
#
# This lint pins the fixed shape so a refactor cannot silently reintroduce
# either trap. Companion asserts: the model-bytes API (ReadBytesAsync) exists
# for the CreateFromBytes path (ADR M4-02 §3).
#
# Usage: bash scripts/check-unity-webgl-traps.sh
# Exit 0 = shape intact; 1 = a trap fix regressed.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FILE="$REPO_ROOT/bindings/unity/com.vokra.unity/Runtime/Vokra/VokraAndroidAssets.cs"

if [ ! -f "$FILE" ]; then
  echo "check-unity-webgl-traps: VokraAndroidAssets.cs not found at $FILE" >&2
  exit 2
fi

fail=0

# 1) A WebGL compile branch must exist (trap 1: no more silent fallthrough).
if ! grep -qE '#if[[:space:]]+UNITY_WEBGL[[:space:]]*&&[[:space:]]*!UNITY_EDITOR' "$FILE"; then
  echo "FAIL: VokraAndroidAssets.cs has no '#if UNITY_WEBGL && !UNITY_EDITOR' branch (trap 1 regressed)" >&2
  fail=1
fi

# 2) The synchronous path must throw NotSupportedException on WebGL (trap 2:
#    loud fail instead of a main-thread deadlock).
if ! grep -q 'NotSupportedException' "$FILE"; then
  echo "FAIL: VokraAndroidAssets.cs no longer throws NotSupportedException on the WebGL sync path (trap 2 regressed)" >&2
  fail=1
fi

# 3) Every synchronous busy-wait must stay confined to the Android branch:
#    the file may contain at most the one Android `while (!op.isDone) { }`.
busy_count="$(grep -cE 'while[[:space:]]*\(![a-zA-Z_]+\.isDone\)[[:space:]]*\{[[:space:]]*\}' "$FILE" || true)"
if [ "$busy_count" -gt 1 ]; then
  echo "FAIL: $busy_count synchronous busy-waits in VokraAndroidAssets.cs (only the Android one is allowed; a WebGL busy-wait deadlocks the main thread)" >&2
  fail=1
fi

# 4) The model-bytes API must exist (WebGL primary path pairs with
#    VokraSession.CreateFromBytes — ADR M4-02 §3).
if ! grep -q 'ReadBytesAsync' "$FILE"; then
  echo "FAIL: VokraAndroidAssets.cs is missing ReadBytesAsync (WebGL model-bytes path)" >&2
  fail=1
fi

# 5) The header comment must not claim "non-Android platforms have a real
#    filesystem" without mentioning WebGL (the doc half of trap 1).
if ! grep -qi 'WebGL' "$FILE"; then
  echo "FAIL: VokraAndroidAssets.cs header/doc never mentions WebGL" >&2
  fail=1
fi

if [ "$fail" -ne 0 ]; then
  echo "check-unity-webgl-traps: FAILED" >&2
  exit 1
fi

echo "check-unity-webgl-traps: OK (WebGL sync=throws / async branch present / bytes API present)"
