#!/usr/bin/env bash
# test-nvidia-scanner-sigpipe.sh
#
# Regression test for a fail-open defect in the NVIDIA non-bundle compliance
# scanners (check-unity-package-no-nvidia.sh / check-godot-package-no-nvidia.sh
# / check-cpu-vulkan-only-no-nvidia.sh) and in verify-ios-xcframework.sh.
#
# THE DEFECT
# ----------
# Those scripts run under `set -euo pipefail` and used to test for banned
# symbols with:
#
#     if nm -u "$lib" 2>/dev/null | grep -qE '_(cudart|cudnn|cublas)'; then
#         ... FAIL ...
#     fi
#
# `grep -q` exits as soon as it matches. If `nm` still has output buffered at
# that moment it dies of SIGPIPE (exit 141). Under `pipefail` the *pipeline*
# status becomes 141, so the `if` takes the FALSE branch — the scanner reports
# "no NVIDIA runtime bundled" for a package that demonstrably does bundle it.
#
# The producer only survives if it can push its entire output into the ~64 KiB
# pipe buffer before grep leaves. So the gate is correct for small libraries and
# fails open for large ones — and is outright NON-DETERMINISTIC near the
# boundary. That is the worst possible failure mode for a compliance gate: it
# passes precisely the artifacts big enough to be real.
#
# THE FIX (what this test locks in)
# ---------------------------------
# Capture the dump once, then filter it with a plain `grep -E` (no `-q`). grep
# without -q reads to EOF, so the producer never sees SIGPIPE and the pipeline
# status is honest.
#
# Usage: bash scripts/compliance/test-nvidia-scanner-sigpipe.sh
# Exits 0 if every scanner correctly FAILS on a large bundling package.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

pass=0
fail=0

ok() {
    echo "  PASS: $1"
    pass=$((pass + 1))
}
bad() {
    echo "  FAIL: $1" >&2
    fail=$((fail + 1))
}

# --- fixture generation -------------------------------------------------------
# A synthetic `nm` dump: the banned symbol on LINE 1 (worst case for the bug —
# grep matches immediately and leaves while the producer is still writing),
# followed by `$1` bytes of innocuous filler.
gen_dump() { # gen_dump <filler_bytes> <outfile> <first_symbol>
    python3 - "$@" <<'PY'
import sys
filler_bytes, out, first = int(sys.argv[1]), sys.argv[2], sys.argv[3]
with open(out, "w") as f:
    f.write("                 U %s\n" % first)
    written = i = 0
    while written < filler_bytes:
        line = "                 U _benign_filler_symbol_%08d\n" % i
        f.write(line)
        written += len(line)
        i += 1
PY
}

mkdir -p "$TMP/bin"

# `nm` shim: ignores its arguments and cats the fixture pointed at by
# $NM_FIXTURE. `cat` dies of SIGPIPE exactly the way a real `nm` would, which is
# the part of the reproduction that matters.
cat > "$TMP/bin/nm" <<'EOF'
#!/bin/sh
exec cat "$NM_FIXTURE"
EOF
chmod +x "$TMP/bin/nm"

# `uname` shim so the Linux code path (nm --undefined-only) can be exercised
# from macOS and vice versa. Both branches carried the same defect.
cat > "$TMP/bin/uname" <<'EOF'
#!/bin/sh
if [ "$1" = "-s" ] && [ -n "${FAKE_UNAME_S:-}" ]; then
    echo "$FAKE_UNAME_S"
    exit 0
fi
exec /usr/bin/uname "$@"
EOF
chmod +x "$TMP/bin/uname"

# ~5 MB: far beyond any pipe buffer, so the bug is deterministic here.
gen_dump 5000000 "$TMP/large-dirty.txt" "_cudart_launch_kernel"
# ~1 KB: fits in the pipe buffer, so even the buggy code caught this. Control
# case — proves the test's package fixture and shims are wired up correctly and
# that a detection is not an artifact of the fix.
gen_dump 1000 "$TMP/small-dirty.txt" "_cudart_launch_kernel"
# ~5 MB with NO banned symbol: guards against over-correcting into a scanner
# that fails on everything (a gate that always fails gets deleted).
gen_dump 5000000 "$TMP/large-clean.txt" "_malloc"

# --- package fixtures ---------------------------------------------------------
UNITY_PKG="$TMP/unity"
mkdir -p "$UNITY_PKG/Plugins"
printf 'placeholder' > "$UNITY_PKG/Plugins/libvokra.dylib"
printf 'placeholder' > "$UNITY_PKG/Plugins/libvokra.so"

GODOT_PKG="$TMP/godot"
mkdir -p "$GODOT_PKG/addons/vokra/bin/linuxbsd/x86_64"
printf 'placeholder' > "$GODOT_PKG/addons/vokra/bin/linuxbsd/x86_64/libvokra_godot.so"
printf 'Apache-2.0 placeholder\n' > "$GODOT_PKG/addons/vokra/LICENSE"
printf 'NOTICE placeholder\n' > "$GODOT_PKG/addons/vokra/NOTICE"
printf 'placeholder\n' > "$GODOT_PKG/addons/vokra/vokra.gdextension"

# --- runner -------------------------------------------------------------------
run_scanner() { # run_scanner <script> <pkgdir> <fixture> <fake_uname>
    local script="$1" pkg="$2" fixture="$3" os="$4" rc=0
    NM_FIXTURE="$fixture" FAKE_UNAME_S="$os" PATH="$TMP/bin:$PATH" \
        bash "$script" "$pkg" >/dev/null 2>&1 || rc=$?
    echo "$rc"
}

expect_detects() { # expect_detects <label> <script> <pkg> <fixture> <os>
    local label="$1" rc
    rc="$(run_scanner "$2" "$3" "$4" "$5")"
    if [ "$rc" -ne 0 ]; then
        ok "$label — scanner rejected the bundling package (exit $rc)"
    else
        bad "$label — SCANNER FAILED OPEN: exit 0 on a package whose nm dump contains _cudart_launch_kernel on line 1"
    fi
}

expect_clean() { # expect_clean <label> <script> <pkg> <fixture> <os>
    local label="$1" rc
    rc="$(run_scanner "$2" "$3" "$4" "$5")"
    if [ "$rc" -eq 0 ]; then
        ok "$label — clean package accepted (exit 0)"
    else
        bad "$label — false positive: exit $rc on a package with no NVIDIA symbols"
    fi
}

echo "test-nvidia-scanner-sigpipe: unity scanner"
expect_detects "unity/Darwin  small dump (control, ~1 KB)" \
    scripts/check-unity-package-no-nvidia.sh "$UNITY_PKG" "$TMP/small-dirty.txt" Darwin
expect_detects "unity/Darwin  LARGE dump (~5 MB, the regression)" \
    scripts/check-unity-package-no-nvidia.sh "$UNITY_PKG" "$TMP/large-dirty.txt" Darwin
expect_detects "unity/Linux   LARGE dump (~5 MB, the regression)" \
    scripts/check-unity-package-no-nvidia.sh "$UNITY_PKG" "$TMP/large-dirty.txt" Linux
expect_clean "unity/Darwin  LARGE clean dump (~5 MB)" \
    scripts/check-unity-package-no-nvidia.sh "$UNITY_PKG" "$TMP/large-clean.txt" Darwin

echo "test-nvidia-scanner-sigpipe: godot scanner"
expect_detects "godot/Darwin  LARGE dump (~5 MB, the regression)" \
    scripts/compliance/check-godot-package-no-nvidia.sh "$GODOT_PKG" "$TMP/large-dirty.txt" Darwin
expect_detects "godot/Linux   LARGE dump (~5 MB, the regression)" \
    scripts/compliance/check-godot-package-no-nvidia.sh "$GODOT_PKG" "$TMP/large-dirty.txt" Linux
expect_clean "godot/Darwin  LARGE clean dump (~5 MB)" \
    scripts/compliance/check-godot-package-no-nvidia.sh "$GODOT_PKG" "$TMP/large-clean.txt" Darwin

# --- static lint: the idiom must not come back --------------------------------
# Behavioural tests only cover the scanners we can cheaply fake a package for.
# This lint covers the rest of the tree, including scripts (verify-ios-
# xcframework.sh) whose fixtures are too expensive to synthesise here.
echo "test-nvidia-scanner-sigpipe: static lint (unbounded producer | grep -q)"
lint_out="$(python3 scripts/compliance/lint-pipefail-grep-q.py 2>&1)" && lint_rc=0 || lint_rc=$?
if [ "$lint_rc" -eq 0 ]; then
    ok "no 'unbounded-producer | grep -q' pipelines under pipefail"
else
    bad "static lint found reintroduced fail-open pipeline(s):"
    printf '%s\n' "$lint_out" >&2
fi

echo ""
if [ "$fail" -ne 0 ]; then
    echo "test-nvidia-scanner-sigpipe: $fail FAILED, $pass passed" >&2
    exit 1
fi
echo "test-nvidia-scanner-sigpipe: OK ($pass passed)"
