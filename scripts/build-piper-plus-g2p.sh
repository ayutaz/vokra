#!/usr/bin/env bash
# build-piper-plus-g2p.sh — build the OUT-OF-WORKSPACE piper-plus G2P reuse
# integration crate, and prove it never leaks into the zero-dependency core
# (M1-01-A; FR-TL-04, FR-API-06, NFR-DS-02).
#
# WHY OUT OF THE WORKSPACE (the load-bearing design decision):
#   The 8-language G2P is *reused* from the existing piper-plus (MIT), not
#   reimplemented (client decision 2026-07-02). The obvious "optional cargo
#   feature on a workspace member" does NOT satisfy this repo's zero-dependency
#   gate: `scripts/check-zero-deps.sh` scans the FULL Cargo.lock, and an
#   optional dependency still appears there — so pulling `piper-plus-g2p` into a
#   `crates/*` member would break CI. The resolution is a physically separate
#   crate at `integrations/vokra-piper-g2p/` that is NOT under `crates/*`, has
#   its OWN `[workspace]` table (an isolated Cargo.lock), and only path-depends
#   on `vokra-piper-plus` for the `Phonemizer` trait. Downstreams (server /
#   Unity / CLI) opt into that crate and inject its `Phonemizer` via
#   `PiperPlusTts::synthesize_with`; the default core build stays std-only and
#   Apache-2.0. See `docs/piper-plus-integration.md` §7/§8.
#
# STATUS: this is a SCAFFOLD. The real integration crate + version pin is
# M1-01-B, BLOCKED on the client's T04 G2P-reuse-form decision (§8 B-7/B-8).
# Until it exists this script still does useful work: it runs the isolation
# guard so the invariant is enforced from day one. Once the crate lands it also
# builds it with per-language features and runs its own license lane.
#
# Exit code: 0 = isolation holds (and, if present, the integration crate built);
#            non-zero = the zero-dependency invariant was violated or a build
#            failure occurred.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
INTEGRATION_DIR="$ROOT/integrations/vokra-piper-g2p"
INTEGRATION_MANIFEST="$INTEGRATION_DIR/Cargo.toml"

# --- 1. Isolation guard (always) --------------------------------------------
# The reuse boundary is only sound if the root lockfile never gains a non-vokra
# crate. This is the same invariant the pre-commit hook and CI enforce; run it
# here too so `build-piper-plus-g2p` fails loudly if anyone wires the G2P in as
# a workspace dependency by mistake.
echo "build-piper-plus-g2p: verifying the zero-dependency core (root Cargo.lock)…"
bash "$ROOT/scripts/check-zero-deps.sh"

# --- 2. The integration crate (when it exists) ------------------------------
if [ ! -f "$INTEGRATION_MANIFEST" ]; then
    cat >&2 <<EOF
build-piper-plus-g2p: integration crate not present yet (scaffold).
  expected: $INTEGRATION_MANIFEST
  This is M1-01-B, blocked on the client's T04 G2P-reuse-form + version-pin
  decision (docs/piper-plus-integration.md §8 B-7/B-8). When it lands it MUST:
    - live outside crates/* with its own [workspace] table (isolated lockfile),
    - path-depend on vokra-piper-plus for the Phonemizer trait,
    - pin piper-plus-g2p (MIT, ort-free) behind per-language features
      (japanese OFF by default; the ~20MB naist-jdic is desktop-only, §8 B-10),
    - carry its own 'cargo deny check licenses' lane.
  The zero-dependency core invariant is already enforced above.
EOF
    exit 0
fi

# The integration crate resolves against its OWN lockfile (its own [workspace]).
# Never pass --manifest-path from the root and never touch the root lockfile.
FEATURES="${VOKRA_G2P_FEATURES:-}" # e.g. "en,zh"; japanese is opt-in (naist-jdic size)
echo "build-piper-plus-g2p: building integration crate (features='${FEATURES}')…"
BUILD_ARGS=(build --release --manifest-path "$INTEGRATION_MANIFEST")
if [ -n "$FEATURES" ]; then
    BUILD_ARGS+=(--no-default-features --features "$FEATURES")
fi
cargo "${BUILD_ARGS[@]}"

# License lane for the reused transitive data/deps (naist-jdic BSD-3, CMUdict
# BSD-style, jpreprocess/pinyin MIT — all non-GPL, ort-free; §10 / license-audit).
if command -v cargo-deny >/dev/null 2>&1; then
    echo "build-piper-plus-g2p: cargo deny check licenses (integration crate)…"
    ( cd "$INTEGRATION_DIR" && cargo deny check licenses )
else
    echo "build-piper-plus-g2p: WARN cargo-deny not installed; skipping license lane" >&2
fi

# --- 3. Re-assert isolation after building ----------------------------------
# Building the integration crate must not have perturbed the ROOT lockfile.
INTEGRATION_LOCK="$INTEGRATION_DIR/Cargo.lock"
if [ -f "$INTEGRATION_LOCK" ] && grep -qE '^name = "piper-plus-g2p"' "$ROOT/Cargo.lock" 2>/dev/null; then
    echo "build-piper-plus-g2p: ERROR piper-plus-g2p leaked into the ROOT Cargo.lock" >&2
    exit 1
fi
bash "$ROOT/scripts/check-zero-deps.sh"
echo "build-piper-plus-g2p: OK (integration crate built; core stays zero-dependency)"
