#!/usr/bin/env bash
# check-catalog-reality.sh — every model the catalog advertises as officially
# distributed must actually be implemented.
#
# WHY THIS EXISTS
#
# `docs/license-audit.md` §3 is the public-facing model table. Its distribution
# column is what a reader takes as "Vokra ships this". On 2026-07-22, while
# preparing to publish converted weights to a public hub, an audit found EIGHT
# rows marked `★ 公式 zoo` with no implementation whatsoever — no model module,
# no converter, nothing but the row: WavTokenizer, openWakeWord, ECAPA-TDNN,
# WeSpeaker, RNNoise, GTCRN, AudioSeal, Vocos.
#
# Nothing caught it because the existing gate
# (`scripts/compliance/check-zoo-manifest-complete.sh`) checks the *inverse*
# direction: that every advertised row is claimed by a zoo-manifest record. A
# record can legitimately be an `excluded_reason` placeholder, so a model with
# zero code satisfies that gate perfectly.
#
# A catalog that overstates coverage is a credibility problem the moment the
# model hub goes public, and it is exactly the class of drift that reappears
# unless a machine watches it. This gate closes that direction: advertised
# implies implemented.
#
# WHAT COUNTS AS IMPLEMENTED
#
# A row is satisfied when EITHER
#   (a) a runtime module exists for it under `crates/vokra-models/src/`, or
#   (b) an operator implements it under `crates/vokra-ops/src/`, or
#   (c) a converter exists under `crates/vokra-convert/src/models/`.
# (b) is included deliberately: DAC and DeepFilterNet3 live in `vokra-ops`
# rather than `vokra-models`, and they are genuinely shipped.
#
# Rows whose distribution cell is anything other than a plain `★ 公式 zoo` —
# `⚠ 保留`, `要 owner sign-off`, `✕`, `★ post-v1.0 GA` — are NOT checked: those
# already say "not shipped today" to a reader.
#
# EXPECTED-GAP LEDGER
#
# Known-unimplemented rows are listed in `EXPECTED_GAPS` below with the reason.
# The gate fails on a gap that is NOT in the ledger (new drift), and ALSO fails
# on a ledger entry that has since been implemented (stale ledger) — so the file
# cannot rot in either direction.
#
# Usage: scripts/publish/check-catalog-reality.sh [--self-test]

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
audit="$repo_root/docs/license-audit.md"

# name-in-catalog -> why it is advertised without an implementation.
# Keep the reason specific enough that a reader can act on it.
declare -a EXPECTED_GAPS=(
  "WavTokenizer|op-only (wavtokenizer_vq in vokra-ops/src/fsq_codec.rs); no model binder or converter — model WP not started"
  "openWakeWord|kws (FR-OP-51) is unimplemented; the catalog row precedes the op"
  "ECAPA-TDNN (SpeechBrain)|speaker_encode variant; anchor-only in m5_residual_ops.rs (CAM++ covers the task)"
  "WeSpeaker|speaker_encode variant; anchor-only in m5_residual_ops.rs (CAM++ covers the task)"
  "RNNoise|denoise alternative candidate; DeepFilterNet3 is the implemented first choice"
  "GTCRN|denoise alternative candidate; DeepFilterNet3 is the implemented first choice"
  "AudioSeal (Meta)|watermark embedding is Deferred (2026-07-04 drop); config surface only"
  "Vocos|vocoder head component; min-dtype anchor only, no kernel"
)

if [[ "${1:-}" == "--self-test" ]]; then
  # A synthetic catalog: one implemented row, one undeclared gap.
  tmp="$(mktemp -d)"
  trap 'rm -rf "$tmp"' EXIT
  cat >"$tmp/audit.md" <<'EOF'
| モデル | Code | Weight | 商用可 | Vokra 公式配布 | 備考 |
|---|---|---|---|---|---|
| **Whisper base** | MIT | MIT | ○ | ★ 公式 zoo | implemented |
| **TotallyFakeModel** | MIT | MIT | ○ | ★ 公式 zoo | no code anywhere |
| **HeldModel** | MIT | ? | △ | ⚠ 保留 | not checked |
EOF
  if out="$("$0" --audit "$tmp/audit.md" 2>&1)"; then
    echo "check-catalog-reality self-test: FAIL (undeclared gap was not caught)" >&2
    exit 1
  fi
  if ! grep -q "TotallyFakeModel" <<<"$out"; then
    echo "check-catalog-reality self-test: FAIL (did not name the offending row)" >&2
    printf '%s\n' "$out" >&2
    exit 1
  fi
  if grep -q "HeldModel" <<<"$out"; then
    echo "check-catalog-reality self-test: FAIL (a 保留 row must not be checked)" >&2
    exit 1
  fi
  echo "check-catalog-reality self-test: OK (3 cases)"
  exit 0
fi

if [[ "${1:-}" == "--audit" ]]; then
  audit="$2"
fi

# Slugs to try when looking for an implementation, derived from the catalog
# name: lowercase, non-alphanumerics -> separators, plus a few known aliases.
slugs_for() {
  local name="$1"
  local base
  base="$(printf '%s' "$name" \
    | tr '[:upper:]' '[:lower:]' \
    | sed -E 's/\(.*//; s/[^a-z0-9]+/ /g; s/^ +| +$//g')"
  # first token, and the whole thing with _ and -
  printf '%s\n' "$base" | tr ' ' '_'
  printf '%s\n' "$base" | tr ' ' '-'
  printf '%s\n' "${base%% *}"
  case "$base" in
    "cam speaker embedding"|"cam"*) printf 'campplus\ncamplus\n' ;;
    "piper plus"*) printf 'piper_plus\n' ;;
    "sesame csm"*) printf 'csm\n' ;;
    "moshi"*) printf 'moshi\n' ;;
    "mimi codec"*) printf 'mimi\n' ;;
    "dac"*) printf 'dac\n' ;;
    "silero vad"*) printf 'silero\n' ;;
    "deepfilternet"*) printf 'denoise\ndeepfilternet\n' ;;
    "cosyvoice"*) printf 'cosyvoice2\n' ;;
    "utmos"*) printf 'utmos\n' ;;
    "x codec 2"*|"xcodec"*) printf 'fsq_codec\nxcodec2\n' ;;
    "wavtokenizer"*) printf 'wavtokenizer\n' ;;
  esac
}

implemented() {
  local name="$1" slug
  while read -r slug; do
    [[ -z "$slug" ]] && continue
    for dir in crates/vokra-models/src crates/vokra-ops/src crates/vokra-convert/src/models; do
      if compgen -G "$repo_root/$dir/$slug"* >/dev/null 2>&1; then
        return 0
      fi
    done
  done < <(slugs_for "$name")
  return 1
}

in_ledger() {
  local name="$1" e
  for e in "${EXPECTED_GAPS[@]}"; do
    [[ "${e%%|*}" == "$name" ]] && return 0
  done
  return 1
}

undeclared=()
stale=()
checked=0

while IFS= read -r line; do
  # Only 6-column model rows.
  IFS='|' read -r -a f <<<"$line"
  [[ ${#f[@]} -lt 7 ]] && continue
  dist="${f[5]//\*\*/}"
  dist="$(printf '%s' "$dist" | sed -E 's/^ +| +$//g')"
  # Only rows that advertise plain official distribution.
  [[ "$dist" == "★ 公式 zoo"* ]] || continue
  # `★ post-v1.0 GA` and friends are a different claim; the prefix test above
  # already excludes them because they do not start with "★ 公式 zoo".
  name="$(printf '%s' "${f[1]}" | sed -E 's/\*\*//g; s/^ +| +$//g')"
  [[ -z "$name" ]] && continue
  checked=$((checked + 1))
  if implemented "$name"; then
    in_ledger "$name" && stale+=("$name")
  else
    in_ledger "$name" || undeclared+=("$name")
  fi
done < "$audit"

status=0
if ((${#undeclared[@]})); then
  status=1
  echo "check-catalog-reality: FAIL — advertised as 公式 zoo but not implemented," >&2
  echo "  and not present in the expected-gap ledger:" >&2
  for n in "${undeclared[@]}"; do echo "    - $n" >&2; done
  echo "  Either implement it, change its distribution cell to something that does" >&2
  echo "  not claim shipping today, or add it to EXPECTED_GAPS with a reason." >&2
fi
if ((${#stale[@]})); then
  status=1
  echo "check-catalog-reality: FAIL — listed as an expected gap but an" >&2
  echo "  implementation now exists (remove it from EXPECTED_GAPS):" >&2
  for n in "${stale[@]}"; do echo "    - $n" >&2; done
fi

if ((status == 0)); then
  echo "check-catalog-reality: OK ($checked rows advertised as 公式 zoo; ${#EXPECTED_GAPS[@]} known gaps declared)"
fi
exit "$status"
