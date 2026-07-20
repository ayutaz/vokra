#!/usr/bin/env bash
# check-zoo-manifest-complete.sh — GA DoD item 2 zoo-completeness gate
# (M5-12-T09 catalog->manifest, + M5-12-T10 manifest->audit license leg).
#
# WHY THIS GATE EXISTS
#   DoD item 2 claims the runner (vokra_eval::dod) covers "all zoo models". A
#   manifest that silently drops a model would make that claim vacuously true
#   for the missing model — a partial coverage laundered into a full pass. This
#   gate makes dropping a model IMPOSSIBLE-to-miss: it is a hard failure unless
#   the model is either present or carries an explicit `excluded_reason`.
#
# THE MACHINE SoT IS docs/license-audit.md (tracked), NOT deliverables.md
#   deliverables.md §3.5 is gitignore-local — absent from a fresh clone / CI
#   runner — so it cannot be the catalog a gate reads. The tracked license
#   audit's "Vokra 公式配布" column (★ 公式 zoo / ⚠ 保留) is the machine SoT.
#
# TWO DIRECTIONS (both are needed to honestly say "all models, license-clean"):
#   (a) catalog -> manifest (completeness, T09): every ★/⚠ zoo row in the audit
#       is claimed by a manifest record (normal OR excluded_reason). An
#       unaccounted row is a hard error.
#   (b) manifest -> audit (no-orphan + no-NC, T10 leg): every manifest record's
#       `audit_name` is a real audit row (no phantom models), and no record
#       carries a CC-BY-NC weight (deliverables.md §3.5 "含めないもの").
#   Co-located here because both directions read the same two files; a Rust test
#   reading repo-root docs from a crate CWD would be fragile.
#
# Plus a bundle-expansion count assert: the Whisper audit row (one row, five
# sizes) must be claimed by exactly 5 records, and the four codecs must each be
# an individual record.
#
# Zero-dep: bash + python3 stdlib only (no jq, no pip). Not a Vokra runtime dep.
# Exit: 0 = complete & license-clean, 1 = a gap / orphan / NC weight / bad arg.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
AUDIT_DEFAULT="$ROOT/docs/license-audit.md"
MANIFEST_DEFAULT="$ROOT/crates/vokra-eval/data/zoo/manifest.txt"

usage() {
    cat <<'USAGE'
check-zoo-manifest-complete.sh — GA DoD item 2 zoo-completeness gate (M5-12)

Usage:
  bash scripts/check-zoo-manifest-complete.sh
  bash scripts/check-zoo-manifest-complete.sh --help
  bash scripts/check-zoo-manifest-complete.sh --self-test

Checks that every ★ 公式 zoo / ⚠ 保留 row in docs/license-audit.md is claimed by
a record (or an excluded_reason record) in the zoo manifest, that no manifest
record is a phantom or carries a CC-BY-NC weight, and that the Whisper bundle
expands to 5 sizes. Exit 1 on any gap.
USAGE
}

# The completeness + license checker. Args: <license-audit.md> <manifest.txt>.
# stdlib only. Reused verbatim by the main path and --self-test.
run_check() {
    python3 - "$1" "$2" <<'PY'
import sys, re

audit_path, manifest_path = sys.argv[1], sys.argv[2]

# ---- 1. audit ★/⚠ zoo rows: (model-name, marker) --------------------------
# A model-table row has 6 columns; the 5th ("Vokra 公式配布") carries the ★/⚠
# marker. The §3.1 sign-off table mentions "★ 公式 zoo" only in its NOTES
# column (6th), so filtering on the 5th column excludes it — a naive substring
# grep would wrongly pull those 4 rows in (they are duplicates of §3 rows).
def strip_md_name(cell):
    c = cell.strip()
    if c.startswith("**") and c.endswith("**") and len(c) >= 4:
        c = c[2:-2].strip()
    return c

audit_models = []  # list of (name, marker)
for raw in open(audit_path, encoding="utf-8"):
    line = raw.rstrip("\n")
    if not line.lstrip().startswith("|"):
        continue
    fields = line.split("|")
    # ['', c1, c2, c3, c4, c5, c6, ''] for a 6-column row.
    if len(fields) < 7:
        continue
    # The marker may be bold-wrapped (e.g. `**⚠ 保留**（…）`); drop `**` before
    # matching so a held row is not missed and mistaken for a phantom.
    col5 = fields[5].replace("**", "").strip()
    if col5.startswith("★ 公式 zoo") or col5.startswith("⚠ 保留"):
        name = strip_md_name(fields[1])
        marker = "star" if col5.startswith("★") else "hold"
        audit_models.append((name, marker))

# ---- 2. manifest records: audit_name / license / name / excluded ----------
records = []
cur = {}
def flush():
    global cur
    if cur:
        records.append(cur)
        cur = {}
for raw in open(manifest_path, encoding="utf-8"):
    s = raw.strip()
    if not s:
        flush(); continue
    if s.startswith("#"):
        continue
    if "=" in s:
        k, v = s.split("=", 1)
        cur[k.strip()] = v.strip()
flush()

manifest_audit_names = {}  # audit_name -> count
for r in records:
    an = r.get("audit_name", "")
    if an:
        manifest_audit_names[an] = manifest_audit_names.get(an, 0) + 1

errors = []

# ---- 3. (a) catalog -> manifest: every ★/⚠ row is claimed ------------------
if not audit_models:
    errors.append(
        "no ★ 公式 zoo / ⚠ 保留 rows found in the audit — the parser or the SoT "
        "layout changed (expected the 5th table column to carry the marker)."
    )
for name, marker in audit_models:
    if name not in manifest_audit_names:
        errors.append(
            f"audit row '{name}' ({marker}) is NOT claimed by any manifest record "
            f"(add a record with `audit_name = {name}`, or an excluded_reason record). "
            f"A silently-dropped model makes DoD item 2 vacuously true for it."
        )

# ---- 4. (b) manifest -> audit: no phantom audit_name ----------------------
audit_name_set = {n for n, _ in audit_models}
for an in manifest_audit_names:
    if an not in audit_name_set:
        errors.append(
            f"manifest audit_name '{an}' matches no ★/⚠ row in docs/license-audit.md "
            f"(a phantom model, or the audit row was renamed — reconcile the two)."
        )

# ---- 5. bundle-expansion count asserts ------------------------------------
WHISPER = "Whisper base/small/medium/large-v3/turbo"
if WHISPER in audit_name_set:
    n = manifest_audit_names.get(WHISPER, 0)
    if n != 5:
        errors.append(
            f"the Whisper bundle audit row must expand to exactly 5 size records, got {n} "
            f"(base/small/medium/large-v3/turbo)."
        )
for codec in ["DAC (Descript)", "Mimi codec (Kyutai)", "WavTokenizer", "X-Codec 2 (Llasa)"]:
    if codec in audit_name_set and codec not in manifest_audit_names:
        errors.append(f"codec '{codec}' must be its own manifest record (individual, not bundled).")

# ---- 6. no CC-BY-NC weight (deliverables.md §3.5 "含めないもの") ------------
for r in records:
    lic = r.get("license", "").upper()
    if "-NC" in lic or "NONCOMMERCIAL" in lic or "NON-COMMERCIAL" in lic:
        errors.append(
            f"manifest record '{r.get('name','?')}' has an NC license '{r.get('license')}' — "
            f"CC-BY-NC weights must never be in the official zoo (X-03 / NFR-LC-04)."
        )
FORBIDDEN = ("f5-tts", "fish-speech", "encodec")
for r in records:
    nm = r.get("name", "").lower()
    if any(f in nm for f in FORBIDDEN):
        errors.append(f"manifest record '{r.get('name')}' is a non-commercial-weight model that must stay research-flagged, not in the zoo.")

if errors:
    print(f"FAIL: {len(errors)} zoo-manifest completeness/license problem(s):")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)

gated = sum(1 for r in records if "excluded_reason" not in r)
excluded = len(records) - gated
print(
    f"OK: {len(audit_models)} ★/⚠ audit rows all accounted for by {len(records)} "
    f"manifest records ({gated} gated / {excluded} excluded); Whisper=5; codecs individual; "
    f"no NC weights."
)
PY
}

self_test() {
    local status=0
    local tmp
    tmp="$(mktemp -d)"
    trap 'rm -rf "$tmp"' RETURN

    # A minimal, self-consistent audit: one ★ row + one ⚠ row + a Whisper
    # bundle. The last row is a §3.1-style sign-off row that mentions the ★
    # marker only in its NOTES column — it must NOT be picked up (5th-column
    # filter), else it would demand a phantom "DAC 24khz" manifest record.
    cat >"$tmp/audit.md" <<'MD'
| モデル | Code License | Weight License | 商用可 | Vokra 公式配布 | 備考 |
|---|---|---|---|---|---|
| **Silero VAD v5** | MIT | MIT | ○ | ★ 公式 zoo | note |
| **Whisper base/small/medium/large-v3/turbo** | MIT | MIT | ○ | ★ 公式 zoo | note |
| **X-Codec 2 (Llasa)** | MIT | ⚠ | ⚠ | **⚠ 保留**（bold-wrapped marker） | held |
| **DAC 24khz (Descript)** | MIT | 2026-07-15 | ___ | ☐ Commercial | mentions ★ 公式 zoo in notes only |
MD

    # Record emitters (blank-line separated by build_manifest — the parser, like
    # crates/vokra-eval/src/manifest.rs, splits records on blank lines).
    _silero() { printf 'name = silero-vad-v5\naudit_name = Silero VAD v5\nlicense = %s\n' "${1:-MIT}"; }
    _whisper() { printf 'name = whisper-%s\naudit_name = Whisper base/small/medium/large-v3/turbo\nlicense = MIT\n' "$1"; }
    _xcodec2() { printf 'name = xcodec2\naudit_name = X-Codec 2 (Llasa)\nlicense = pending\nexcluded_reason = held\n'; }
    _phantom() { printf 'name = phantom\naudit_name = Not A Real Audit Row\nlicense = MIT\nexcluded_reason = whatever\n'; }
    # build_manifest <emitter...> — writes records blank-separated to manifest.txt
    build_manifest() {
        : >"$tmp/manifest.txt"
        local first=1 e
        for e in "$@"; do
            [ "$first" -eq 1 ] || printf '\n' >>"$tmp/manifest.txt"
            first=0
            eval "$e" >>"$tmp/manifest.txt"
        done
    }
    passes() { run_check "$tmp/audit.md" "$tmp/manifest.txt" >/dev/null 2>&1; }

    # Fixture 1: complete manifest (Silero + Whisper x5 + xcodec2 excluded) -> PASS.
    build_manifest '_silero' '_whisper base' '_whisper small' '_whisper medium' \
        '_whisper large-v3' '_whisper turbo' '_xcodec2'
    if passes; then
        echo "self-test PASS: complete manifest passes (bold ⚠ marker recognised)"
    else
        echo "self-test FAIL: complete manifest should pass" >&2; status=1
    fi

    # Fixture 2: drop Silero -> completeness (audit->manifest) FAIL.
    build_manifest '_whisper base' '_whisper small' '_whisper medium' \
        '_whisper large-v3' '_whisper turbo' '_xcodec2'
    if passes; then
        echo "self-test FAIL: dropping Silero should fail completeness" >&2; status=1
    else
        echo "self-test PASS: a dropped model fails completeness (audit->manifest)"
    fi

    # Fixture 3: Whisper only 4 sizes -> bundle-expansion count FAIL.
    build_manifest '_silero' '_whisper base' '_whisper small' '_whisper medium' \
        '_whisper large-v3' '_xcodec2'
    if passes; then
        echo "self-test FAIL: 4-size Whisper should fail the bundle count" >&2; status=1
    else
        echo "self-test PASS: Whisper != 5 fails the bundle-expansion assert"
    fi

    # Fixture 4: a phantom audit_name -> no-orphan (manifest->audit) FAIL.
    build_manifest '_silero' '_whisper base' '_whisper small' '_whisper medium' \
        '_whisper large-v3' '_whisper turbo' '_xcodec2' '_phantom'
    if passes; then
        echo "self-test FAIL: a phantom audit_name should fail no-orphan" >&2; status=1
    else
        echo "self-test PASS: a phantom manifest audit_name fails (manifest->audit)"
    fi

    # Fixture 5: an NC weight -> no-NC license leg FAIL.
    build_manifest '_silero CC-BY-NC-4.0' '_whisper base' '_whisper small' '_whisper medium' \
        '_whisper large-v3' '_whisper turbo' '_xcodec2'
    if passes; then
        echo "self-test FAIL: a CC-BY-NC weight should fail the license leg" >&2; status=1
    else
        echo "self-test PASS: a CC-BY-NC weight fails the no-NC license leg"
    fi

    if [ "$status" -eq 0 ]; then
        echo "check-zoo-manifest-complete --self-test: OK"
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
        for f in "$AUDIT_DEFAULT" "$MANIFEST_DEFAULT"; do
            if [ ! -f "$f" ]; then
                echo "error: required file not found: $f" >&2
                exit 1
            fi
        done
        run_check "$AUDIT_DEFAULT" "$MANIFEST_DEFAULT"
        ;;
    *)
        echo "error: unknown argument '$1'" >&2
        usage >&2
        exit 1
        ;;
esac
