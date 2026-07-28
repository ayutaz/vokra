#!/usr/bin/env bash
# publish-one.sh — take one model from a re-converted GGUF to a live HF repo.
#
# Chains the pieces that were proven by hand on piper-plus, so the repetitive
# 11-model run does not re-type (and mis-type) them:
#   1. stage via upload.sh (card from artifact + §3.1 sign-off gate + NOTICE/SOURCE)
#   2. fetch the correct LICENSE text (fetch_license.sh)
#   3. T3 (Copyleft) gates 6a-6e (this script — see below)
#   4. push via upload.sh --push
#
# The GGUF must ALREADY be re-converted with provenance stamped (upload.sh
# refuses an unstamped artifact). This script does not convert — conversion is
# memory-bound and model-specific, so it stays in the caller's hands.
#
# DRY-RUN by default; --push publishes. Publishing is irreversible.
#
# T3 (Copyleft) gates — 6a-6e
# ----------------------------
# make_model_card.py already lets a `LicenseClass::Copyleft` weight (AGPL /
# GPL / LGPL / CC-BY-SA — crates/vokra-core/src/compliance/license_class.rs)
# publish with no extra flag, because the class itself is `redistributable()`.
# What it does NOT check is whether the *bundle this script assembles* — the
# files upload.sh and fetch_license.sh write one step at a time — actually
# discharges the obligation the class carries
# (`requires_license_preserved()`). A partial run, or a future reordering of
# the steps above, could otherwise leave a copyleft weight staged (or
# published) without them. Fail-closed, mirroring upload.sh's own
# belt-and-suspenders LICENSE check immediately before --push:
#   6a. LICENSE must be bundled.
#   6b. NOTICE must be bundled.
#   6c. AGPL specifically also requires SOURCE.md (the network-use
#       source-availability pointer; CC-BY-SA carries no such term).
#   6d. Any copyleft SKU requires --acknowledge-copyleft — a conscious,
#       per-invocation opt-in, mirroring --allow-noncommercial.
#   6e. The card's HF front-matter `license:` tag must match
#       --license-spdx — a mismatch means the LICENSE text about to ship is
#       not the one the weight's own metadata says it is.
# Copyleft is detected from the artifact's own `vokra.provenance.*` metadata
# (the LicenseClass vokra-convert already stamped), not by re-deriving a
# verdict from a license string by hand.
#
# Usage:
#   publish-one.sh --gguf <file> --repo vokra/<name> \
#     ( --license-url <raw-url> | --license-spdx <spdx> ) \
#     [--push] [--allow-noncommercial] [--acknowledge-copyleft]
#   publish-one.sh --self-test
#
# HF token: HF_TOKEN or HF in the environment.

set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- T3 (Copyleft) gate helpers ------------------------------------------

license_meta() {
  # Prints two lines: the class (`vokra.provenance.weight_license`, e.g.
  # "copyleft") then the raw upstream string (`vokra.provenance.license`,
  # e.g. "AGPL-3.0"). Reuses GgufReader from make_model_card.py — a
  # header-only parser already exercised by that script's own self-test —
  # instead of hand-rolling a second GGUF reader here.
  python3 - "$1" "$here/make_model_card.py" <<'PY'
import sys, importlib.util
gguf_path, mmc_path = sys.argv[1], sys.argv[2]
spec = importlib.util.spec_from_file_location("mmc", mmc_path)
mmc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mmc)
g = mmc.GgufReader(gguf_path)
print((g.get("vokra.provenance.weight_license") or "").strip().lower())
print((g.get("vokra.provenance.license") or "").strip())
PY
}

hf_tag_for() {
  # Mirrors make_model_card.py::hf_license_tag exactly — imported, not
  # re-implemented, so a future change to the accepted-tag set cannot make
  # the two silently disagree.
  python3 - "$1" "$here/make_model_card.py" <<'PY'
import sys, importlib.util
val, mmc_path = sys.argv[1], sys.argv[2]
spec = importlib.util.spec_from_file_location("mmc", mmc_path)
mmc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mmc)
print(mmc.hf_license_tag(val))
PY
}

# Gate 6d. Must be acknowledged before anything else happens on this SKU's
# behalf (network fetch, a card that will say "copyleft" in public, etc.) —
# mirrors --allow-noncommercial, which gates make_model_card.py the same way.
#   $1 weight_class  e.g. "copyleft" (from vokra.provenance.weight_license)
#   $2 raw_license   e.g. "AGPL-3.0" (from vokra.provenance.license) — message only
#   $3 ack           "true" if --acknowledge-copyleft was passed
require_copyleft_ack() {
  local weight_class="$1" raw_license="$2" ack="$3"
  [[ "$weight_class" == "copyleft" ]] || return 0
  if [[ "$ack" != "true" ]]; then
    echo "publish-one: gate 6d REFUSE — Copyleft SKU (license class 'copyleft', raw licence '${raw_license:-unrecorded}') requires --acknowledge-copyleft" >&2
    return 1
  fi
  return 0
}

# Gates 6a/6b/6c/6e. The bundle this script assembles must actually carry
# what the class obliges — a partial run, or a future reordering of the
# staging steps, could otherwise leave a copyleft weight published without
# them. Pure: only reads local files + strings, no network, so it is
# exercised directly (no subprocess, no live fetch) from --self-test.
#   $1 outdir       staged directory (README.md / LICENSE / NOTICE / SOURCE.md)
#   $2 weight_class e.g. "copyleft"
#   $3 raw_license  e.g. "AGPL-3.0" — decides whether 6c (SOURCE.md) applies
#   $4 lspdx        --license-spdx value, or "" if --license-url was used
copyleft_bundle_gates() {
  local outdir="$1" weight_class="$2" raw_license="$3" lspdx="$4"
  [[ "$weight_class" == "copyleft" ]] || return 0

  local raw_lc is_agpl
  raw_lc="$(printf '%s' "$raw_license" | tr '[:upper:]' '[:lower:]')"
  is_agpl=0
  [[ "$raw_lc" == *agpl* ]] && is_agpl=1

  # 6a — LICENSE.
  if [[ ! -f "$outdir/LICENSE" ]]; then
    echo "publish-one: gate 6a REFUSE — Copyleft SKU requires LICENSE bundled (missing at $outdir/LICENSE)" >&2
    return 1
  fi

  # 6b — NOTICE.
  if [[ ! -f "$outdir/NOTICE" ]]; then
    echo "publish-one: gate 6b REFUSE — Copyleft SKU requires NOTICE bundled (missing at $outdir/NOTICE)" >&2
    return 1
  fi

  # 6c — AGPL specifically also requires SOURCE.md (the network-use
  # source-availability pointer; CC-BY-SA carries no such term).
  if [[ $is_agpl -eq 1 && ! -f "$outdir/SOURCE.md" ]]; then
    echo "publish-one: gate 6c REFUSE — AGPL SKU requires SOURCE.md bundled (missing at $outdir/SOURCE.md)" >&2
    return 1
  fi

  # 6e — the card's HF front-matter tag must match the SPDX we are actually
  # fetching. A mismatch means the LICENSE text about to ship is not the one
  # the weight's own metadata says it is. Only checked when --license-spdx
  # was given (an opaque --license-url has no SPDX to compare against); a
  # copyleft SKU whose README.md this gate cannot parse REFUSES rather than
  # silently skipping a check it cannot perform.
  if [[ -n "$lspdx" ]]; then
    if [[ ! -f "$outdir/README.md" ]]; then
      echo "publish-one: gate 6e REFUSE — cannot verify the card licence tag: $outdir/README.md is missing" >&2
      return 1
    fi
    local card_tag expected_tag
    card_tag="$(sed -n 's/^license:[[:space:]]*//p' "$outdir/README.md" | head -1 | tr -d '\r')"
    if [[ -z "$card_tag" ]]; then
      echo "publish-one: gate 6e REFUSE — cannot verify the card licence tag: no 'license:' front-matter line in $outdir/README.md" >&2
      return 1
    fi
    expected_tag="$(hf_tag_for "$lspdx")"
    if [[ "$card_tag" != "$expected_tag" ]]; then
      echo "publish-one: gate 6e REFUSE — card licence tag '$card_tag' does not match --license-spdx '$lspdx' (normalized '$expected_tag')" >&2
      return 1
    fi
  fi

  return 0
}

_selftest_build_gguf() {
  # Writes a header-only GGUF (0 tensors, string-only metadata) at $1 from
  # the key/value pairs in $2... — same byte layout make_model_card.py's own
  # self-test uses, so the two can never silently disagree about the format.
  local out="$1"; shift
  python3 - "$out" "$here/make_model_card.py" "$@" <<'PY'
import struct, sys, importlib.util
out, mmc_path, kvs = sys.argv[1], sys.argv[2], sys.argv[3:]
spec = importlib.util.spec_from_file_location("mmc", mmc_path)
mmc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mmc)
buf = bytearray(b"GGUF")
buf += struct.pack("<I", 3) + struct.pack("<Q", 0)
buf += struct.pack("<Q", len(kvs) // 2)
for i in range(0, len(kvs), 2):
    k, v = kvs[i], kvs[i + 1]
    buf += struct.pack("<Q", len(k)) + k.encode()
    buf += struct.pack("<I", mmc.STR)
    buf += struct.pack("<Q", len(v)) + v.encode()
open(out, "wb").write(bytes(buf))
PY
}

run_self_test() {
  # Exercises the gate functions above directly (no subprocess, no network,
  # no dependency on docs/license-audit.md or a real HF token) — matching
  # upload.sh --self-test's own "verify the refusals that matter, without
  # touching the network" design. Both directions matter: refusing what must
  # be refused, and *passing* what is compliant — an over-strict gate is a
  # silent failure too.
  local tmp fail cases out
  tmp="$(mktemp -d)"
  # Expand $tmp NOW: the trap fires after this function's locals are gone
  # (at real process exit), so a single-quoted 'rm -rf "$tmp"' would defer
  # the lookup to then, when $tmp is out of scope and set -u makes it a
  # hard abort instead of a cleanup. Same pattern as check-capi-thread-free.sh.
  # shellcheck disable=SC2064
  trap "rm -rf '$tmp'" EXIT
  fail=0; cases=0

  # --- detection: a real copyleft/AGPL GGUF round-trips through license_meta
  local g meta
  g="$tmp/agpl.gguf"
  _selftest_build_gguf "$g" \
    vokra.model.arch test-copyleft-agpl \
    vokra.provenance.weight_license copyleft \
    vokra.provenance.license AGPL-3.0
  cases=$((cases + 1))
  if ! meta="$(license_meta "$g")"; then
    echo "self-test FAIL: license_meta crashed on a well-formed GGUF" >&2; fail=1
  elif [[ "$(sed -n '1p' <<<"$meta")" != "copyleft" || "$(sed -n '2p' <<<"$meta")" != "AGPL-3.0" ]]; then
    echo "self-test FAIL: license_meta did not read back copyleft/AGPL-3.0 from a real GGUF (got: $meta)" >&2
    fail=1
  fi

  # --- hf_tag_for mirrors make_model_card.py's normalizer ------------------
  cases=$((cases + 1))
  [[ "$(hf_tag_for AGPL-3.0)" == "agpl-3.0" ]] || { echo "self-test FAIL: hf_tag_for AGPL-3.0 != agpl-3.0" >&2; fail=1; }
  cases=$((cases + 1))
  [[ "$(hf_tag_for cc-by-sa-4.0)" == "cc-by-sa-4.0" ]] || { echo "self-test FAIL: hf_tag_for cc-by-sa-4.0" >&2; fail=1; }

  # --- 6d: acknowledgement is required for copyleft, never for others -----
  cases=$((cases + 1))
  if require_copyleft_ack "copyleft" "AGPL-3.0" "false" 2>/dev/null; then
    echo "self-test FAIL: gate 6d did not refuse a copyleft SKU with no --acknowledge-copyleft" >&2; fail=1
  fi
  cases=$((cases + 1))
  require_copyleft_ack "copyleft" "AGPL-3.0" "true" \
    || { echo "self-test FAIL: gate 6d refused an acknowledged copyleft SKU" >&2; fail=1; }
  cases=$((cases + 1))
  require_copyleft_ack "permissive" "MIT" "false" \
    || { echo "self-test FAIL: gate 6d fired for a non-copyleft (permissive) SKU" >&2; fail=1; }

  # --- 6a: a copyleft bundle missing LICENSE must refuse, naming gate 6a --
  mkdir -p "$tmp/neg-license"
  printf -- '---\nlicense: agpl-3.0\n---\n' > "$tmp/neg-license/README.md"
  : > "$tmp/neg-license/NOTICE"
  : > "$tmp/neg-license/SOURCE.md"
  cases=$((cases + 1))
  if out="$(copyleft_bundle_gates "$tmp/neg-license" "copyleft" "AGPL-3.0" "agpl-3.0" 2>&1)"; then
    echo "self-test FAIL: gate 6a did not refuse a copyleft bundle missing LICENSE" >&2; fail=1
  elif [[ "$out" != *"gate 6a"* ]]; then
    echo "self-test FAIL: refusal did not name gate 6a: $out" >&2; fail=1
  fi

  # --- 6b: LICENSE present, NOTICE missing -> refuse naming gate 6b -------
  mkdir -p "$tmp/neg-notice"
  cp "$tmp/neg-license/README.md" "$tmp/neg-notice/"
  : > "$tmp/neg-notice/LICENSE"
  : > "$tmp/neg-notice/SOURCE.md"
  cases=$((cases + 1))
  if out="$(copyleft_bundle_gates "$tmp/neg-notice" "copyleft" "AGPL-3.0" "agpl-3.0" 2>&1)"; then
    echo "self-test FAIL: gate 6b did not refuse a copyleft bundle missing NOTICE" >&2; fail=1
  elif [[ "$out" != *"gate 6b"* ]]; then
    echo "self-test FAIL: refusal did not name gate 6b: $out" >&2; fail=1
  fi

  # --- 6c: AGPL missing SOURCE.md refuses; CC-BY-SA missing it is fine ----
  mkdir -p "$tmp/neg-source"
  cp "$tmp/neg-license/README.md" "$tmp/neg-source/"
  : > "$tmp/neg-source/LICENSE"
  : > "$tmp/neg-source/NOTICE"
  cases=$((cases + 1))
  if out="$(copyleft_bundle_gates "$tmp/neg-source" "copyleft" "AGPL-3.0" "agpl-3.0" 2>&1)"; then
    echo "self-test FAIL: gate 6c did not refuse an AGPL bundle missing SOURCE.md" >&2; fail=1
  elif [[ "$out" != *"gate 6c"* ]]; then
    echo "self-test FAIL: refusal did not name gate 6c: $out" >&2; fail=1
  fi

  mkdir -p "$tmp/sa-no-source"
  printf -- '---\nlicense: cc-by-sa-4.0\n---\n' > "$tmp/sa-no-source/README.md"
  : > "$tmp/sa-no-source/LICENSE"
  : > "$tmp/sa-no-source/NOTICE"
  cases=$((cases + 1))
  copyleft_bundle_gates "$tmp/sa-no-source" "copyleft" "CC-BY-SA-4.0" "cc-by-sa-4.0" \
    || { echo "self-test FAIL: gate 6c wrongly required SOURCE.md for a non-AGPL copyleft SKU" >&2; fail=1; }

  # --- 6e: a declared SPDX that disagrees with the card must refuse -------
  mkdir -p "$tmp/mismatch"
  printf -- '---\nlicense: apache-2.0\n---\n' > "$tmp/mismatch/README.md"
  : > "$tmp/mismatch/LICENSE"
  : > "$tmp/mismatch/NOTICE"
  : > "$tmp/mismatch/SOURCE.md"
  cases=$((cases + 1))
  if out="$(copyleft_bundle_gates "$tmp/mismatch" "copyleft" "AGPL-3.0" "agpl-3.0" 2>&1)"; then
    echo "self-test FAIL: gate 6e did not refuse a card/--license-spdx mismatch" >&2; fail=1
  elif [[ "$out" != *"gate 6e"* ]]; then
    echo "self-test FAIL: refusal did not name gate 6e: $out" >&2; fail=1
  fi

  # --- positive: acknowledged + all four files + agreeing tag -> proceeds -
  mkdir -p "$tmp/pos"
  printf -- '---\nlicense: agpl-3.0\n---\n' > "$tmp/pos/README.md"
  : > "$tmp/pos/LICENSE"
  : > "$tmp/pos/NOTICE"
  : > "$tmp/pos/SOURCE.md"
  cases=$((cases + 1))
  if ! { require_copyleft_ack "copyleft" "AGPL-3.0" "true" \
      && copyleft_bundle_gates "$tmp/pos" "copyleft" "AGPL-3.0" "agpl-3.0"; }; then
    echo "self-test FAIL: a fully-compliant, acknowledged AGPL bundle was refused" >&2; fail=1
  fi

  # --- non-copyleft SKUs are never gated, regardless of what is missing ---
  mkdir -p "$tmp/mit-empty"
  cases=$((cases + 1))
  copyleft_bundle_gates "$tmp/mit-empty" "permissive" "MIT" "mit" \
    || { echo "self-test FAIL: gates 6a-6e fired for a permissive (non-copyleft) SKU" >&2; fail=1; }

  if [[ $fail -eq 0 ]]; then
    echo "publish-one self-test: OK ($cases cases)"
    return 0
  fi
  return 1
}

gguf=""; repo=""; lurl=""; lspdx=""; push=0; nc=0; ack_copyleft=0; self_test=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gguf) gguf="$2"; shift 2 ;;
    --repo) repo="$2"; shift 2 ;;
    --license-url) lurl="$2"; shift 2 ;;
    --license-spdx) lspdx="$2"; shift 2 ;;
    --push) push=1; shift ;;
    --allow-noncommercial) nc=1; shift ;;
    --acknowledge-copyleft) ack_copyleft=1; shift ;;
    --self-test) self_test=1; shift ;;
    *) echo "publish-one: unexpected arg $1" >&2; exit 2 ;;
  esac
done

if [[ $self_test -eq 1 ]]; then
  run_self_test
  exit $?
fi

[[ -f "$gguf" ]] || { echo "publish-one: --gguf must be an existing file" >&2; exit 2; }
[[ -n "$repo" ]] || { echo "publish-one: --repo is required" >&2; exit 2; }
[[ -n "$lurl" || -n "$lspdx" ]] || { echo "publish-one: one of --license-url / --license-spdx is required" >&2; exit 2; }

model_name="${repo##*/}"
outdir="$(cd "$(git -C "$here" rev-parse --show-toplevel)" && pwd)/target/publish/$model_name"

# T3 (Copyleft) gate 6d — read the class before any side effect (network
# fetch, a card that will say "copyleft" in public) happens on this SKU's
# behalf.
if ! meta="$(license_meta "$gguf")"; then
  echo "publish-one: could not read licence metadata from $gguf — is it a valid Vokra GGUF?" >&2
  exit 2
fi
weight_class="$(sed -n '1p' <<<"$meta")"
raw_license="$(sed -n '2p' <<<"$meta")"
ack_flag="false"; [[ $ack_copyleft -eq 1 ]] && ack_flag="true"
require_copyleft_ack "$weight_class" "$raw_license" "$ack_flag" || exit 1

nc_flag=(); [[ $nc -eq 1 ]] && nc_flag=(--allow-noncommercial)

echo "############ $repo ############"
echo "== stage (dry-run) =="
"$here/upload.sh" "$gguf" --repo "$repo" --out "$outdir" ${nc_flag[@]+"${nc_flag[@]}"}

echo "== LICENSE =="
if [[ -n "$lurl" ]]; then
  "$here/fetch_license.sh" --url "$lurl" "$outdir/LICENSE"
else
  "$here/fetch_license.sh" --spdx "$lspdx" "$outdir/LICENSE"
fi

echo "== T3 (Copyleft) gates (6a-6e) =="
copyleft_bundle_gates "$outdir" "$weight_class" "$raw_license" "$lspdx" || exit 1

if [[ $push -eq 0 ]]; then
  echo "== DRY RUN complete — staged in $outdir. Re-run with --push to publish. =="
  exit 0
fi

echo "== push =="
"$here/upload.sh" "$gguf" --repo "$repo" --out "$outdir" --push ${nc_flag[@]+"${nc_flag[@]}"}
echo "== done: https://huggingface.co/$repo =="
