#!/usr/bin/env bash
# check-native-methods.sh — Vokra C# P/Invoke lint (M2-11-T09, ADR-0007 D4).
#
# Enforces that every [DllImport] in the com.vokra.unity Runtime asmdef routes
# through NativeMethods.Lib. Hardcoding "vokra" or "__Internal" in a DllImport
# would defeat the platform switch (D4): iOS device builds must resolve symbols
# via "__Internal" (static libvokra.a, NFR-RL-03 — dlopen of a custom dylib is
# forbidden by App Store review) while every other target must resolve the
# shared library "vokra" (Unity strips the platform prefix/suffix).
#
# Also asserts that the Lib constant itself carries both branches so a stray
# rewrite that drops the iOS #if would fail loudly here instead of at App Store
# submission time.
#
# Usage:
#   bash scripts/check-native-methods.sh                     # scans default UPM path
#   bash scripts/check-native-methods.sh path/to/Runtime     # custom root
#
# Exit 0 on success; non-zero with a human-readable diagnostic on any violation.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEFAULT_ROOT="$REPO_ROOT/bindings/unity/com.vokra.unity/Runtime"
ROOT="${1:-$DEFAULT_ROOT}"

if [ ! -d "$ROOT" ]; then
  echo "check-native-methods: root not found: $ROOT" >&2
  exit 2
fi

NATIVE_METHODS_FILE="$ROOT/Vokra/NativeMethods.cs"
if [ ! -f "$NATIVE_METHODS_FILE" ]; then
  echo "check-native-methods: NativeMethods.cs not found at $NATIVE_METHODS_FILE" >&2
  exit 2
fi

fail=0

# 1) Every [DllImport(...)] in the Runtime tree must use the Lib symbol, not a
#    hardcoded string literal ("vokra" / "__Internal" / "libvokra" / …).
#    grep -E over all .cs files under Runtime; skip nothing.
bad_literal="$(grep -REn --include='*.cs' \
  'DllImport[[:space:]]*\([[:space:]]*"' "$ROOT" || true)"
if [ -n "$bad_literal" ]; then
  echo "FAIL: DllImport with a hardcoded string literal (must reference NativeMethods.Lib):" >&2
  echo "$bad_literal" >&2
  fail=1
fi

# 2) Belt-and-braces: no bare "vokra" / "__Internal" identifier passed to
#    DllImport, even without quotes (e.g. `DllImport(vokra, ...)`).
bad_ident="$(grep -REn --include='*.cs' \
  'DllImport[[:space:]]*\([[:space:]]*(vokra|__Internal)([[:space:]]*,|[[:space:]]*\))' \
  "$ROOT" || true)"
if [ -n "$bad_ident" ]; then
  echo "FAIL: DllImport with a bare vokra/__Internal identifier (must be NativeMethods.Lib):" >&2
  echo "$bad_ident" >&2
  fail=1
fi

# 3) Every [DllImport] must route through the Lib symbol. Count DllImport
#    attributes and DllImport(Lib …) references; they must match.
total="$(grep -REn --include='*.cs' -c 'DllImport[[:space:]]*\(' "$ROOT" \
  | awk -F: '{ s += $NF } END { print s + 0 }')"
routed="$(grep -REn --include='*.cs' -c 'DllImport[[:space:]]*\(Lib' "$ROOT" \
  | awk -F: '{ s += $NF } END { print s + 0 }')"
if [ "$total" -ne "$routed" ]; then
  echo "FAIL: $total DllImport attribute(s) found but only $routed route through Lib" >&2
  grep -REn --include='*.cs' 'DllImport[[:space:]]*\(' "$ROOT" \
    | grep -v 'DllImport[[:space:]]*(Lib' >&2 || true
  fail=1
fi

# 4) NativeMethods.cs must define Lib with all three states of the platform
#    switch (M4-02-T05): the static-link targets (iOS AND WebGL -> "__Internal",
#    NFR-RL-03 + its WebGL 準用) and the shared-library default ("vokra").
#    Guards against a rewrite that collapses the platform switch or drops one
#    static-link platform — an iOS-branch drop would fail at App Store
#    submission, a WebGL-branch drop at the Unity WebGL link; both fail HERE
#    instead.
if ! grep -qE '#if[[:space:]]+\(UNITY_IOS[[:space:]]*\|\|[[:space:]]*UNITY_WEBGL\)[[:space:]]*&&[[:space:]]*!UNITY_EDITOR' \
     "$NATIVE_METHODS_FILE"; then
  echo "FAIL: NativeMethods.cs is missing the '#if (UNITY_IOS || UNITY_WEBGL) && !UNITY_EDITOR' guard" >&2
  fail=1
fi
if ! grep -qE 'internal const string Lib = "__Internal"' "$NATIVE_METHODS_FILE"; then
  echo "FAIL: NativeMethods.cs is missing the iOS/WebGL Lib = \"__Internal\" branch" >&2
  fail=1
fi
if ! grep -qE 'internal const string Lib = "vokra"' "$NATIVE_METHODS_FILE"; then
  echo "FAIL: NativeMethods.cs is missing the default Lib = \"vokra\" branch" >&2
  fail=1
fi

if [ "$fail" -ne 0 ]; then
  echo "check-native-methods: FAILED" >&2
  exit 1
fi

echo "check-native-methods: OK (all DllImport route through NativeMethods.Lib; iOS+WebGL __Internal / default vokra branches present)"
