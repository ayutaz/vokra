#!/usr/bin/env bash
# check-plugin-meta.sh — validate hand-authored Unity PluginImporter .meta files
# for the desktop native libs in com.vokra.unity.
#
# For each of the three desktop plugin .meta files, this script asserts:
#   1. YAML parses cleanly (via python3 yaml.safe_load).
#   2. The file contains exactly one `enabled: 1` line — i.e. exactly one
#      per-platform block enabled, matching the folder the .meta lives in
#      (macOS -> OSXUniversal, Windows -> Win64, Linux -> Linux64).
#   3. That `enabled: 1` sits under the platform token that maps to the
#      folder name (grep -B 3 window).
#
# Used by:
#   * local dev loop after `scripts/build-unity-plugin.sh`
#   * CI unity-package job (M2-11-T13)
#
# See docs/tickets/m2/ (M2-11 WP) for context.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PKG_DIR="$REPO_ROOT/bindings/unity/com.vokra.unity"

# folder-relative .meta path : expected Unity platform token
CASES=(
  "Plugins/macOS/libvokra.dylib.meta:OSXUniversal"
  "Plugins/Windows/x86_64/vokra.dll.meta:Win64"
  "Plugins/Linux/x86_64/libvokra.so.meta:Linux64"
)

fail() { echo "FAIL: $*" >&2; exit 1; }

if ! command -v python3 >/dev/null 2>&1; then
  fail "python3 not found (required for YAML parse)"
fi

# PyYAML available?
if ! python3 -c 'import yaml' >/dev/null 2>&1; then
  fail "PyYAML not installed (pip install pyyaml)"
fi

for entry in "${CASES[@]}"; do
  rel="${entry%%:*}"
  platform="${entry##*:}"
  path="$PKG_DIR/$rel"

  [ -f "$path" ] || fail "$rel: file missing at $path"

  # 1. YAML parse.
  if ! python3 -c "import sys, yaml; yaml.safe_load(open(sys.argv[1]))" "$path" \
       >/dev/null 2>&1; then
    fail "$rel: not valid YAML"
  fi

  # 2. Exactly one 'enabled: 1'.
  count=$(grep -Ec '^[[:space:]]*enabled: 1[[:space:]]*$' "$path" || true)
  [ "$count" -eq 1 ] || fail "$rel: expected exactly one 'enabled: 1', got $count"

  # 3. `enabled: 1` sits under the expected platform block (within 3 lines).
  if ! grep -B 3 -E '^[[:space:]]*enabled: 1[[:space:]]*$' "$path" \
       | grep -q "$platform"; then
    fail "$rel: 'enabled: 1' not tied to platform '$platform'"
  fi

  echo "OK: $rel (enabled for $platform)"
done

echo "All plugin .meta files valid."
