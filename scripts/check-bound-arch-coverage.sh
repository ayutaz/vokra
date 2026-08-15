#!/usr/bin/env bash
# check-bound-arch-coverage.sh — bound-arch registry COMPLETENESS gate.
#
# WHY THIS GATE EXISTS
#   crates/vokra-cli/src/engine.rs carries BOUND_ARCHES: the registry that
#   turns "unsupported model arch `X`" — a lie, when vokra-models has a binder
#   for X — into a diagnostic naming the binding module, the library entry
#   point and the class of blocker. Its own doc comment says:
#
#       **Adding a binder?** Add a row here in the same commit.
#
#   Nothing enforced that. The in-crate test
#   `bound_arch_registry_is_disjoint_from_the_routed_arches` checks the rows
#   that EXIST (no duplicates, no shadowing of a routed arch, module path and
#   entry non-empty) — it cannot see a row that was never written. Five binders
#   (chattts, deepfake_detection, lang_id_ecapa, dtln_aec, nkf_aec) went missing
#   that way, three of them in the very commit that wrote the rule, and every
#   one of their users got the blanket "unsupported model arch" instead.
#
#   This gate closes that hole from the other direction: it starts from the
#   BINDERS, not from the registry.
#
# THE INVARIANT
#   Every `pub const ARCH: &str = "…"` (and its `pub const ARCH_<SUFFIX>: &str`
#   siblings, e.g. gigaam's ARCH_V3 / ARCH_MULTILINGUAL) declared anywhere under
#   crates/vokra-models/src/ must be accounted for in engine.rs by EITHER
#     (a) a routed `const ARCH_*: &str = "…"` constant — the dispatch runs it, or
#     (b) an `arch: "…"` row inside `const BOUND_ARCHES` — bound, not runnable.
#   An arch in neither place is a binder whose users are told it does not exist.
#
# DIRECTION IS DELIBERATELY ONE-WAY (binders -> engine.rs)
#   The reverse direction would be wrong: plenty of legitimate BOUND_ARCHES rows
#   name arches whose module declares no `pub const ARCH` at all (canary,
#   distil-whisper, zonos, …), so a registry->models check would fire on correct
#   rows. Duplicate / shadowed / malformed rows are already the in-crate test's
#   job; this script only answers "is any binder missing?".
#
# Zero-dep: bash + python3 stdlib only (no jq, no pip, no cargo). Not a Vokra
# runtime dep.
# Exit: 0 = every binder accounted for, 1 = an unaccounted binder / bad arg.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODELS_DEFAULT="$ROOT/crates/vokra-models/src"
ENGINE_DEFAULT="$ROOT/crates/vokra-cli/src/engine.rs"

usage() {
    cat <<'USAGE'
check-bound-arch-coverage.sh — bound-arch registry completeness gate

Usage:
  bash scripts/check-bound-arch-coverage.sh
  bash scripts/check-bound-arch-coverage.sh --help
  bash scripts/check-bound-arch-coverage.sh --self-test

Checks that every model arch constant declared under crates/vokra-models/src/
is accounted for in crates/vokra-cli/src/engine.rs — either as a routed
`const ARCH_*` the dispatch runs, or as a row in `const BOUND_ARCHES`. A binder
in neither place makes `vokra-cli run` report "unsupported model arch" for a
model this build actually binds. Exit 1 on any unaccounted binder.
USAGE
}

# The coverage checker. Args: <vokra-models/src dir> <engine.rs path>.
# stdlib only. Reused verbatim by the main path and --self-test.
run_check() {
    python3 - "$1" "$2" <<'PY'
import os, re, sys

models_dir, engine_path = sys.argv[1], sys.argv[2]

# ---- 1. binders: every arch constant under vokra-models/src ----------------
# Matches `pub const ARCH: &str = "x";` and `pub const ARCH_V3: &str = "x";`.
# `&'static str` is accepted too so a future spelling change does not silently
# drop a binder out of the scan.
ARCH_CONST = re.compile(
    r'^\s*pub\s+const\s+(ARCH(?:_[A-Z0-9_]+)?)\s*:\s*&(?:\'static\s+)?str\s*=\s*"([^"]+)"\s*;'
)

binders = []  # (arch, "relpath:lineno", const_name)
for dirpath, dirnames, filenames in os.walk(models_dir):
    dirnames.sort()
    for fn in sorted(filenames):
        if not fn.endswith(".rs"):
            continue
        path = os.path.join(dirpath, fn)
        rel = os.path.relpath(path, models_dir)
        with open(path, encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                m = ARCH_CONST.match(line)
                if m:
                    binders.append((m.group(2), f"{rel}:{lineno}", m.group(1)))

# ---- 2. engine.rs: routed constants + BOUND_ARCHES rows --------------------
ROUTED_CONST = re.compile(
    r'^\s*(?:pub\s+)?const\s+ARCH_[A-Z0-9_]+\s*:\s*&(?:\'static\s+)?str\s*=\s*"([^"]+)"\s*;'
)
REGISTRY_START = re.compile(r'^\s*(?:pub\s+)?const\s+BOUND_ARCHES\s*:')
REGISTRY_ROW = re.compile(r'^\s*arch\s*:\s*"([^"]+)"\s*,?\s*$')

routed = set()
registry = set()
in_registry = False
registry_seen = False
with open(engine_path, encoding="utf-8") as fh:
    for line in fh:
        if not in_registry and REGISTRY_START.match(line):
            in_registry = True
            registry_seen = True
            continue
        if in_registry:
            # The registry literal closes on a column-0 `];`.
            if line.rstrip() == "];":
                in_registry = False
                continue
            m = REGISTRY_ROW.match(line)
            if m:
                registry.add(m.group(1))
            continue
        m = ROUTED_CONST.match(line)
        if m:
            routed.add(m.group(1))

errors = []

# ---- 3. parser guards -----------------------------------------------------
# A checker that silently scanned nothing would pass every run — the exact
# fabricated-pass shape this gate exists to prevent. Each guard fires only if
# the source layout moved out from under the parser.
if not binders:
    errors.append(
        f"no `pub const ARCH...: &str = \"…\"` found anywhere under {models_dir} — the "
        f"walk or the constant spelling changed; the scan covered nothing, so a pass "
        f"here would be vacuous."
    )
if not routed:
    errors.append(
        f"no routed `const ARCH_*: &str = \"…\"` found in {engine_path} — the dispatch's "
        f"arch constants moved or were renamed; without them every routed arch would "
        f"read as unaccounted."
    )
if not registry_seen:
    errors.append(
        f"`const BOUND_ARCHES` not found in {engine_path} — the registry was renamed or "
        f"moved; this gate cannot verify completeness against a registry it cannot find."
    )
elif not registry:
    errors.append(
        f"`const BOUND_ARCHES` in {engine_path} parsed to ZERO rows — the row shape "
        f"changed (expected `arch: \"…\",` one per line)."
    )
if in_registry:
    errors.append(
        f"`const BOUND_ARCHES` in {engine_path} never closed on a column-0 `];` — the "
        f"registry literal was reformatted and the row scan may have run past its end."
    )

# ---- 4. the invariant: every binder is routed or registered ----------------
covered = routed | registry
for arch, where, const_name in binders:
    if arch not in covered:
        errors.append(
            f"binder arch `{arch}` ({const_name} at vokra-models/src/{where}) appears in "
            f"NEITHER the routed `const ARCH_*` constants NOR `const BOUND_ARCHES` in "
            f"crates/vokra-cli/src/engine.rs. `vokra-cli run` therefore reports "
            f"\"unsupported model arch `{arch}`\" for a model this build actually binds. "
            f"Fix: add a BOUND_ARCHES row (arch / module / entry / reason / probe) — or, "
            f"if the dispatch runs it, a routed `const ARCH_*` constant."
        )

if errors:
    print(f"FAIL: {len(errors)} bound-arch coverage problem(s):")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)

archs = {a for a, _, _ in binders}
via_routed = sum(1 for a in archs if a in routed)
via_registry = sum(1 for a in archs if a in registry and a not in routed)
print(
    f"OK: {len(binders)} arch constant(s) / {len(archs)} distinct arch(es) declared under "
    f"vokra-models are all accounted for in engine.rs "
    f"({via_routed} routed by the dispatch, {via_registry} in BOUND_ARCHES); "
    f"registry carries {len(registry)} row(s), dispatch {len(routed)} routed constant(s)."
)
PY
}

self_test() {
    local status=0
    local tmp
    tmp="$(mktemp -d)"
    trap 'rm -rf "$tmp"' RETURN

    mkdir -p "$tmp/models/alpha" "$tmp/models/beta" "$tmp/models/nested/gamma"
    printf 'pub const ARCH: &str = "alpha";\n' >"$tmp/models/alpha/mod.rs"
    printf 'pub const ARCH: &str = "beta";\n' >"$tmp/models/beta/mod.rs"
    # Nested + suffixed constant: proves the walk recurses and that the
    # `ARCH_<SUFFIX>` spelling (gigaam / hifigan / magnet) is in scope.
    printf 'pub const ARCH_V2: &str = "gamma";\n' >"$tmp/models/nested/gamma/mod.rs"

    # write_engine <registry-arch...> — one engine.rs with `alpha` routed and
    # the named arches as BOUND_ARCHES rows.
    write_engine() {
        {
            printf 'const ARCH_ALPHA: &str = "alpha";\n\n'
            printf 'const BOUND_ARCHES: &[BoundArch] = &[\n'
            local a
            for a in "$@"; do
                printf '    BoundArch {\n        arch: "%s",\n        module: "vokra_models::x",\n    },\n' "$a"
            done
            printf '];\n'
        } >"$tmp/engine.rs"
    }
    passes() { run_check "$tmp/models" "$tmp/engine.rs" >/dev/null 2>&1; }

    # Fixture 1: every binder accounted for (alpha routed, beta + gamma rows).
    write_engine beta gamma
    if passes; then
        echo "self-test PASS: a complete registry passes (routed + rows, nested walk)"
    else
        echo "self-test FAIL: a complete registry should pass" >&2; status=1
    fi

    # Fixture 2: gamma's row dropped -> the defect this gate exists to catch.
    write_engine beta
    if passes; then
        echo "self-test FAIL: a missing BOUND_ARCHES row should fail" >&2; status=1
    else
        echo "self-test PASS: a binder with no row and no routing fails"
    fi

    # Fixture 3: routed coverage counts too (no row needed when the dispatch
    # actually runs the arch).
    {
        printf 'const ARCH_ALPHA: &str = "alpha";\n'
        printf 'const ARCH_BETA: &str = "beta";\n'
        printf 'const ARCH_GAMMA: &str = "gamma";\n\n'
        printf 'const BOUND_ARCHES: &[BoundArch] = &[\n    BoundArch {\n        arch: "unrelated",\n    },\n];\n'
    } >"$tmp/engine.rs"
    if passes; then
        echo "self-test PASS: a routed arch needs no registry row"
    else
        echo "self-test FAIL: routed coverage should satisfy the invariant" >&2; status=1
    fi

    # Fixture 4: registry renamed away -> parser guard, not a vacuous pass.
    {
        printf 'const ARCH_ALPHA: &str = "alpha";\n'
        printf 'const ARCH_BETA: &str = "beta";\n'
        printf 'const ARCH_GAMMA: &str = "gamma";\n'
    } >"$tmp/engine.rs"
    if passes; then
        echo "self-test FAIL: a missing BOUND_ARCHES literal should fail the guard" >&2; status=1
    else
        echo "self-test PASS: a renamed/absent registry fails the parser guard"
    fi

    # Fixture 5: a models tree the walk finds nothing in -> parser guard.
    write_engine beta gamma
    mkdir -p "$tmp/empty"
    if run_check "$tmp/empty" "$tmp/engine.rs" >/dev/null 2>&1; then
        echo "self-test FAIL: scanning zero binders should fail the guard" >&2; status=1
    else
        echo "self-test PASS: a scan that found no binders fails rather than passing vacuously"
    fi

    if [ "$status" -eq 0 ]; then
        echo "check-bound-arch-coverage --self-test: OK"
    fi
    return "$status"
}

case "${1:-}" in
    --help | -h)
        usage
        exit 0
        ;;
    --self-test)
        self_test
        exit $?
        ;;
    "")
        if [ ! -d "$MODELS_DEFAULT" ]; then
            echo "error: required directory not found: $MODELS_DEFAULT" >&2
            exit 1
        fi
        if [ ! -f "$ENGINE_DEFAULT" ]; then
            echo "error: required file not found: $ENGINE_DEFAULT" >&2
            exit 1
        fi
        run_check "$MODELS_DEFAULT" "$ENGINE_DEFAULT"
        ;;
    *)
        echo "error: unknown argument '$1'" >&2
        usage >&2
        exit 1
        ;;
esac
