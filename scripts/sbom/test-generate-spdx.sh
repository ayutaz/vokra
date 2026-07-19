#!/usr/bin/env bash
# test-generate-spdx.sh — self-test for the first-party SPDX SBOM generator
# (M4-15-T06/T07, ADR M4-15 §(b)(c)).
#
# Needs the Rust toolchain (`cargo tree`) + python3 — the same environment
# as the `build-target-vulkan-only` CI job and the local dev machine. It
# exercises the REAL generator against the REAL workspace (no fixtures):
#
#   b1  well-formed SPDX 2.3 JSON with the required fields
#   b2  feature exclusion is visible in the SBOM: vulkan build lists
#       vokra-backend-vulkan and does NOT list vokra-backend-metal/-cuda
#       (FR-BE-09 "SBOM で明示")
#   b3  every package is first-party vokra-* (NFR-DS-02 reflected)
#   b4  deterministic: two runs at the same commit are byte-identical
#       (T10 reproducible-build verify precondition)
#   b5  SOURCE_DATE_EPOCH is honored (deterministic `created`)
#   b6  relationships reference declared SPDXIDs; DESCRIBES present
#   b7  control: a metal feature selection DOES list vokra-backend-metal
#       (proves the SBOM reflects the feature resolve, not a hardcode)
#   b8  generation leaves the root Cargo.lock untouched (zero-dep gate)
#
# Usage: bash scripts/sbom/test-generate-spdx.sh
# Exit:  0 = all pass, 1 = any failure.

set -uo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
GEN="$ROOT/scripts/sbom/generate_spdx.py"

SCRATCH="$(mktemp -d "${TMPDIR:-/tmp}/m4-15-sbom.XXXXXX")"
trap 'rm -rf "$SCRATCH"' EXIT

pass=0
fail=0
ok() {
    pass=$((pass + 1))
    echo "  ok:   $1"
}
bad() {
    fail=$((fail + 1))
    echo "  FAIL: $1" >&2
}

echo "[T06] generate_spdx.py"

if [ ! -f "$GEN" ]; then
    bad "generator missing: $GEN"
    echo ""
    echo "sbom self-test: $pass passed, $fail failed"
    exit 1
fi

SBOM="$SCRATCH/vokra-capi.spdx.json"

# b1 — generate + structural validation.
if python3 "$GEN" \
    --package vokra-capi \
    --no-default-features \
    --features vulkan \
    --doc-name vokra-capi-cpu-vulkan-only \
    --output "$SBOM" >/dev/null 2>"$SCRATCH/gen.err" \
    && [ -s "$SBOM" ]; then
    ok "b1a generator produced a non-empty document"
else
    bad "b1a generator failed: $(cat "$SCRATCH/gen.err" 2>/dev/null | head -3)"
fi

if python3 - "$SBOM" <<'EOF' >/dev/null 2>&1; then
import json, sys
doc = json.load(open(sys.argv[1]))
required = ["spdxVersion", "dataLicense", "SPDXID", "name",
            "documentNamespace", "creationInfo", "packages", "relationships"]
missing = [k for k in required if k not in doc]
assert not missing, f"missing top-level fields: {missing}"
assert doc["spdxVersion"] == "SPDX-2.3", doc["spdxVersion"]
assert doc["dataLicense"] == "CC0-1.0", doc["dataLicense"]
assert doc["SPDXID"] == "SPDXRef-DOCUMENT"
assert "created" in doc["creationInfo"] and "creators" in doc["creationInfo"]
for p in doc["packages"]:
    for k in ["SPDXID", "name", "versionInfo", "downloadLocation",
              "filesAnalyzed", "licenseConcluded", "licenseDeclared"]:
        assert k in p, f"package {p.get('name')} missing {k}"
EOF
    ok "b1b required SPDX 2.3 fields present"
else
    bad "b1b SPDX structure validation failed"
fi

# b2 — exclusion is explicit in the SBOM.
names="$(python3 -c 'import json,sys; print("\n".join(sorted(p["name"] for p in json.load(open(sys.argv[1]))["packages"])))' "$SBOM" 2>/dev/null)"
if echo "$names" | grep -qx "vokra-backend-vulkan" \
    && echo "$names" | grep -qx "vokra-capi" \
    && ! echo "$names" | grep -qx "vokra-backend-metal" \
    && ! echo "$names" | grep -qx "vokra-backend-cuda"; then
    ok "b2 SBOM lists vokra-backend-vulkan and excludes metal / cuda"
else
    bad "b2 SBOM package set wrong: $(echo "$names" | tr '\n' ' ')"
fi

# b3 — zero-dep reflected: every package is vokra-*.
if [ -n "$names" ] && ! echo "$names" | grep -qv '^vokra'; then
    ok "b3 every SBOM package is first-party vokra-*"
else
    bad "b3 non-vokra package in the SBOM (or empty set): $(echo "$names" | grep -v '^vokra' | tr '\n' ' ')"
fi

# b4 — determinism at the same commit.
python3 "$GEN" --package vokra-capi --no-default-features --features vulkan \
    --doc-name vokra-capi-cpu-vulkan-only --output "$SCRATCH/second.spdx.json" >/dev/null 2>&1
if cmp -s "$SBOM" "$SCRATCH/second.spdx.json"; then
    ok "b4 two runs are byte-identical"
else
    bad "b4 output differs between two runs at the same commit"
fi

# b5 — SOURCE_DATE_EPOCH honored.
SOURCE_DATE_EPOCH=0 python3 "$GEN" --package vokra-capi --no-default-features \
    --features vulkan --doc-name vokra-capi-cpu-vulkan-only \
    --output "$SCRATCH/epoch0.spdx.json" >/dev/null 2>&1
created="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["creationInfo"]["created"])' "$SCRATCH/epoch0.spdx.json" 2>/dev/null)"
if [ "$created" = "1970-01-01T00:00:00Z" ]; then
    ok "b5 SOURCE_DATE_EPOCH=0 yields created=1970-01-01T00:00:00Z"
else
    bad "b5 created not derived from SOURCE_DATE_EPOCH: got '$created'"
fi

# b6 — relationship graph integrity.
if python3 - "$SBOM" <<'EOF' >/dev/null 2>&1; then
import json, sys
doc = json.load(open(sys.argv[1]))
ids = {p["SPDXID"] for p in doc["packages"]} | {"SPDXRef-DOCUMENT"}
rels = doc["relationships"]
assert any(r["relationshipType"] == "DESCRIBES" for r in rels)
for r in rels:
    assert r["spdxElementId"] in ids, r
    assert r["relatedSpdxElement"] in ids, r
EOF
    ok "b6 relationships reference declared SPDXIDs (DESCRIBES present)"
else
    bad "b6 relationship graph is inconsistent"
fi

# b7 — control: metal feature selection surfaces vokra-backend-metal.
python3 "$GEN" --package vokra-models --features metal \
    --doc-name vokra-models-metal-control --output "$SCRATCH/metal.spdx.json" >/dev/null 2>&1
if python3 -c 'import json,sys; names={p["name"] for p in json.load(open(sys.argv[1]))["packages"]}; sys.exit(0 if "vokra-backend-metal" in names else 1)' "$SCRATCH/metal.spdx.json" 2>/dev/null; then
    ok "b7 metal control SBOM lists vokra-backend-metal (feature resolve is live)"
else
    bad "b7 metal control SBOM did not list vokra-backend-metal"
fi

# b8 — root Cargo.lock untouched by generation.
if git -C "$ROOT" diff --exit-code Cargo.lock >/dev/null 2>&1; then
    ok "b8 root Cargo.lock unchanged (zero-dep isolation held)"
else
    bad "b8 root Cargo.lock was modified by SBOM generation"
fi

# b9 — ANSI-proofing. CI exports CARGO_TERM_COLOR=always, which makes `cargo
# tree` wrap the dedup marker in escapes ("(/path) \x1b[33m\x1b[2m(*)"); the
# generator's TREE_LINE then rejected the line as format drift and the SBOM job
# died. resolve_graph() pins `--color never`, but nothing re-exercised the
# hostile setting, and the bug had already stayed latent until vokra-mmap became
# a shared dependency (a package must appear TWICE for a dedup marker to exist).
# Regenerate under CARGO_TERM_COLOR=always and demand byte-identical output.
if CARGO_TERM_COLOR=always python3 "$GEN" --package vokra-capi \
    --no-default-features --features vulkan \
    --doc-name vokra-capi-cpu-vulkan-only \
    --output "$SCRATCH/color.spdx.json" >/dev/null 2>&1 \
    && cmp -s "$SBOM" "$SCRATCH/color.spdx.json"; then
    ok "b9 CARGO_TERM_COLOR=always yields a byte-identical SBOM"
else
    bad "b9 CARGO_TERM_COLOR=always changed or broke generation (ANSI leaked into \`cargo tree\` parsing)"
fi

echo ""
echo "sbom self-test: $pass passed, $fail failed"
[ "$fail" -eq 0 ] || exit 1
exit 0
