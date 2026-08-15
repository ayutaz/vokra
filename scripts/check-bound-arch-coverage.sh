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
#   Every `pub const ARCH: &str = "…"`, its `pub const ARCH_<SUFFIX>: &str`
#   siblings (gigaam's ARCH_V3 / ARCH_MULTILINGUAL) and every
#   `pub const EXPECTED_ARCH: &str` declared anywhere under
#   crates/vokra-models/src/ must be accounted for in engine.rs by EITHER
#     (a) a routed `const ARCH_*: &str = "…"` constant — the dispatch runs it, or
#     (b) an `arch: "…"` row inside `const BOUND_ARCHES` — bound, not runnable.
#   An arch in neither place is a binder whose users are told it does not exist.
#
#   The EXPECTED_ARCH half of that population was invisible here until
#   2026-08-15 — see the ARCH_CONST comment below. Being green over a
#   population you never scanned is worse than having no gate at all, so a
#   guard now fails the run if any arch-shaped `&str` constant on disk falls
#   outside the discovery regex.
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
(`pub const ARCH`, `pub const ARCH_<SUFFIX>`, `pub const EXPECTED_ARCH`, all
typed &str) is accounted for in crates/vokra-cli/src/engine.rs — either as a
routed `const ARCH_*` the dispatch runs, or as a row in `const BOUND_ARCHES`. A
binder in neither place makes `vokra-cli run` report "unsupported model arch"
for a model this build actually binds. Exit 1 on any unaccounted binder.

A parser guard also fails the run if any arch-shaped &str constant on disk
falls outside that set, so a discovery regex that silently stops matching
cannot report a smaller, clean-looking population.
USAGE
}

# The coverage checker. Args: <vokra-models/src dir> <engine.rs path>.
# stdlib only. Reused verbatim by the main path and --self-test.
run_check() {
    python3 - "$1" "$2" <<'PY'
import os, re, sys

models_dir, engine_path = sys.argv[1], sys.argv[2]

# ---- 1. binders: every arch constant under vokra-models/src ----------------
# Matches `pub const ARCH: &str = "x";`, `pub const ARCH_V3: &str = "x";` and
# `pub const EXPECTED_ARCH: &str = "x";`. `&'static str` is accepted too.
#
# The EXPECTED_ARCH alternative was added 2026-08-15. Its absence was not a
# cosmetic omission: 29 of the 89 arch constants under vokra-models/src use
# that spelling (charsiu, csm, moshi, silero-vad, voxtral, zonos, the whole
# chatterbox family, …), so this gate had been reporting "60 arch constants,
# all accounted for" while never looking at a third of the binders. The
# sibling scripts/check-arch-handshake.sh had the identical blind spot, which
# is why `charsiu` — a binder with no converter at all — survived four audit
# rounds behind two green gates. LOOSE_ARCH_CONST below exists so the next
# new spelling fails loudly instead of shrinking the population in silence.
ARCH_CONST = re.compile(
    r'^\s*pub\s+const\s+((?:EXPECTED_)?ARCH(?:_[A-Z0-9_]+)?)\s*:\s*&(?:\'static\s+)?str\s*=\s*"([^"]+)"\s*;'
)
# Deliberately sloppy twin of ARCH_CONST: ANY `pub const <name>: &str = "…";`
# whose name contains `ARCH`. Never used for discovery — only to prove
# discovery saw everything on disk that looks like an arch declaration.
LOOSE_ARCH_CONST = re.compile(
    r'^\s*pub\s+const\s+([A-Z0-9_]*ARCH[A-Z0-9_]*)\s*:\s*&(?:\'static\s+)?str\s*=\s*"([^"]+)"\s*;'
)

binders = []  # (arch, "relpath:lineno", const_name)
unseen_spellings = {}  # const_name -> "relpath:lineno"
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
                    continue
                loose = LOOSE_ARCH_CONST.match(line)
                if loose:
                    unseen_spellings.setdefault(loose.group(1), f"{rel}:{lineno}")

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
# The discovery-coverage guard. The guards around it notice a scan that found
# NOTHING; this one notices a scan that found SOME — the far more dangerous
# shape, because a partial population still prints a confident count and a
# clean verdict. A count band would need a threshold and modules legitimately
# declare zero or several arch constants, so any band loose enough to avoid
# false alarms would have accepted the 60-of-89 this gate actually shipped.
# Matching the strict discovery regex against a deliberately sloppy twin needs
# no threshold and names the exact unmatched spelling.
#
# SCOPE LIMIT, measured rather than assumed [2026-08-15]: LOOSE_ARCH_CONST
# keys on the constant NAME but still requires `pub `, so a declaration hidden
# by VISIBILITY is invisible to it too — 3 models-side `EXPECTED_ARCH`
# constants are private (cosyvoice2, piper_plus, kokoro) and 71 converter-side
# ones are `pub(crate)`. Those were not triaged here, and at least one is a
# live defect (`crisper-whisper`; see the same note in
# scripts/check-arch-handshake.sh). A pass from this guard means "no unknown
# SPELLING", not "every arch constant is covered".
for _name, _where in sorted(unseen_spellings.items()):
    errors.append(
        f"`pub const {_name}` at vokra-models/src/{_where} declares an arch-shaped `&str` "
        f"constant that the discovery regex does NOT match, so this gate never looked at it "
        f"— nor at any other binder spelled `{_name}`. That is exactly how 29 "
        f"`EXPECTED_ARCH` binders stayed outside this gate and its sibling until 2026-08-15 "
        f"while both reported a clean green. Fix: widen ARCH_CONST here AND in "
        f"scripts/check-arch-handshake.sh (they discover independently, so both must "
        f"change), then re-run both and expect NEW findings — or rename the constant to an "
        f"already-covered spelling."
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
# The spellings are printed so a future shrink of the scanned population is
# visible in a CI log rather than hiding behind a still-green verdict.
spellings = sorted({n for _, _, n in binders})
print(
    f"OK: {len(binders)} arch constant(s) / {len(archs)} distinct arch(es) declared under "
    f"vokra-models are all accounted for in engine.rs "
    f"({via_routed} routed by the dispatch, {via_registry} in BOUND_ARCHES); "
    f"registry carries {len(registry)} row(s), dispatch {len(routed)} routed constant(s); "
    f"discovery covered the spelling(s): {', '.join(spellings)}."
)
PY
}

self_test() {
    local status=0
    local out
    local tmp
    tmp="$(mktemp -d)"
    trap 'rm -rf "$tmp"' RETURN

    mkdir -p "$tmp/models/alpha" "$tmp/models/beta" "$tmp/models/nested/gamma" \
        "$tmp/models/nested/delta"
    printf 'pub const ARCH: &str = "alpha";\n' >"$tmp/models/alpha/mod.rs"
    printf 'pub const ARCH: &str = "beta";\n' >"$tmp/models/beta/mod.rs"
    # Nested + suffixed constant: proves the walk recurses and that the
    # `ARCH_<SUFFIX>` spelling (gigaam / hifigan / magnet) is in scope.
    printf 'pub const ARCH_V2: &str = "gamma";\n' >"$tmp/models/nested/gamma/mod.rs"
    # The `EXPECTED_ARCH` spelling — 29 real binders use it, and it was
    # invisible to this gate until 2026-08-15. Present in the BASE fixture so
    # every case below exercises it, not just the one that names it.
    printf 'pub const EXPECTED_ARCH: &str = "delta";\n' >"$tmp/models/nested/delta/mod.rs"

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

    # Fixture 1: every binder accounted for (alpha routed, beta + gamma +
    # delta rows).
    write_engine beta gamma delta
    if passes; then
        echo "self-test PASS: a complete registry passes (routed + rows, nested walk)"
    else
        echo "self-test FAIL: a complete registry should pass" >&2; status=1
    fi

    # Fixture 2: gamma's row dropped -> the defect this gate exists to catch.
    write_engine beta delta
    if passes; then
        echo "self-test FAIL: a missing BOUND_ARCHES row should fail" >&2; status=1
    else
        echo "self-test PASS: a binder with no row and no routing fails"
    fi

    # Fixture 2b: delta's row dropped. Same defect, but on the EXPECTED_ARCH
    # spelling — if discovery ever narrows back to `ARCH` only, this case
    # passes vacuously, so it is the regression test for the 2026-08-15 bug.
    write_engine beta gamma
    if out="$(run_check "$tmp/models" "$tmp/engine.rs" 2>&1)"; then
        echo "self-test FAIL: an unaccounted EXPECTED_ARCH binder should fail" >&2; status=1
    elif grep -q 'binder arch `delta`' <<<"$out"; then
        echo "self-test PASS: a \`pub const EXPECTED_ARCH\` binder is discovered and checked"
    else
        echo "self-test FAIL: the failure did not name \`delta\`" >&2
        printf '%s\n' "$out" >&2; status=1
    fi

    # Fixture 3: routed coverage counts too (no row needed when the dispatch
    # actually runs the arch).
    {
        printf 'const ARCH_ALPHA: &str = "alpha";\n'
        printf 'const ARCH_BETA: &str = "beta";\n'
        printf 'const ARCH_GAMMA: &str = "gamma";\n'
        printf 'const ARCH_DELTA: &str = "delta";\n\n'
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
        printf 'const ARCH_DELTA: &str = "delta";\n'
    } >"$tmp/engine.rs"
    if passes; then
        echo "self-test FAIL: a missing BOUND_ARCHES literal should fail the guard" >&2; status=1
    else
        echo "self-test PASS: a renamed/absent registry fails the parser guard"
    fi

    # Fixture 5: a models tree the walk finds nothing in -> parser guard.
    write_engine beta gamma delta
    mkdir -p "$tmp/empty"
    if run_check "$tmp/empty" "$tmp/engine.rs" >/dev/null 2>&1; then
        echo "self-test FAIL: scanning zero binders should fail the guard" >&2; status=1
    else
        echo "self-test PASS: a scan that found no binders fails rather than passing vacuously"
    fi

    # Fixture 6: the guard that would have CAUGHT the 2026-08-15 bug. An
    # arch-shaped `&str` constant whose NAME the discovery regex does not
    # match must fail loudly and name the spelling — the fixture 5 guard only
    # notices a scan that found NOTHING, and this gate's real failure was a
    # scan that found SOME and reported the remainder as clean.
    printf 'pub const LEGACY_ARCH: &str = "invisible";\n' >"$tmp/models/nested/legacy.rs"
    if out="$(run_check "$tmp/models" "$tmp/engine.rs" 2>&1)"; then
        echo "self-test FAIL: an undiscoverable arch spelling should fail the guard" >&2
        status=1
    elif grep -q 'LEGACY_ARCH' <<<"$out"; then
        echo "self-test PASS: an arch-shaped constant outside the discovery regex fails the guard"
    else
        echo "self-test FAIL: the guard did not name \`LEGACY_ARCH\`" >&2
        printf '%s\n' "$out" >&2; status=1
    fi
    rm -f "$tmp/models/nested/legacy.rs"

    # Fixture 7: the guard must not fire on non-`&str` arch-ish constants —
    # `ACCEPTED_ARCHS: &[&str]` (gigaam, whisper) and
    # `SIBLING_EVAL_ARCHES: [&str; 4]` (squim, utmosv2) are real shapes in
    # this tree and are aggregates of arches, not declarations of one. A
    # guard that cried wolf on them would be turned off within a week.
    {
        printf 'pub const ACCEPTED_ARCHS: &[&str] = &[ARCH, ARCH_V2];\n'
        printf 'pub const SIBLING_EVAL_ARCHES: [&str; 2] = ["alpha", "beta"];\n'
    } >"$tmp/models/nested/aggregates.rs"
    if passes; then
        echo "self-test PASS: aggregate arch constants (&[&str] / [&str; N]) do not trip the guard"
    else
        echo "self-test FAIL: the guard fired on a non-declaration aggregate" >&2
        run_check "$tmp/models" "$tmp/engine.rs" >&2 || true
        status=1
    fi
    rm -f "$tmp/models/nested/aggregates.rs"

    if [ "$status" -eq 0 ]; then
        echo "check-bound-arch-coverage --self-test: OK (8 cases)"
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
