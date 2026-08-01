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
  # WavTokenizer entry removed 2026-08-01: Wave 3 codec add landed the
  # converter (`crates/vokra-convert/src/models/wavtokenizer.rs`) +
  # `ModelKind::WavTokenizer`. The row is now considered implemented
  # (`implemented("WavTokenizer")` matches the `wavtokenizer` slug in
  # `crates/vokra-convert/src/models/`), and keeping the entry would trip
  # the double-sided "stale ledger" check. Runtime forward already exists
  # via the M4-16 `wavtokenizer_vq` op (`vokra-ops/src/fsq_codec.rs`);
  # real-weight parity is a §3.1 sign-off follow-up.
  "openWakeWord|kws (FR-OP-51) is unimplemented; the catalog row precedes the op"
  # ECAPA-TDNN and WeSpeaker previously listed here as anchor-only; converter
  # implementations landed in the SoTA Phase 1-4 wave (commit 7ed0548) at
  # crates/vokra-convert/src/models/{ecapa_tdnn.rs,wespeaker.rs}. The stale
  # entries were removed 2026-07-31 as part of the FQ-03 CI promotion — the
  # production run now runs per-PR, so future stale/undeclared drift is caught
  # at PR time rather than at owner publish time.
  "RNNoise|denoise alternative candidate; DeepFilterNet3 is the implemented first choice"
  "GTCRN|denoise alternative candidate; DeepFilterNet3 is the implemented first choice"
  "AudioSeal (Meta)|watermark embedding is Deferred (2026-07-04 drop); config surface only"
  # Vocos previously listed here as anchor-only; converter implementation
  # landed in the vocos wave (2026-08-01) at
  # crates/vokra-convert/src/models/vocos.rs (BF16 pass-through skeleton,
  # mel-24khz + encodec-24khz VocosVariant, runtime binder deferred to
  # owner §3.1 sign-off — mel-24khz + encodec-24khz rows signed
  # 2026-08-01 yousan). The stale entry was removed 2026-08-01 as part
  # of this wave; the FQ-03 CI production run catches undeclared drift
  # per-PR (same rule as ECAPA-TDNN / WeSpeaker precedent above).
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
    # 2026-08-01 Wave 3: MOSS-Audio-Tokenizer — the codec half of the
    # MOSS-TTS pipeline (Full + Nano, OpenMOSS-Team, apache-2.0). The
    # auto-derived `moss_audio_tokenizer` slug already resolves via the
    # module file `crates/vokra-convert/src/models/moss_audio_tokenizer.rs`,
    # but this explicit alias catches display-name variants ("MOSS Audio
    # Tokenizer" / "MOSS-Audio-Tokenizer (Full)" 等) that would tokenize
    # differently.
    "moss audio tokenizer"*|"moss-audio-tokenizer"*) printf 'moss_audio_tokenizer\n' ;;
    # 2026-08-01 Wave 4 slug-only add: OpenMOSS Team MOSS-VoiceGenerator
    # (`OpenMOSS-Team/MOSS-VoiceGenerator`, apache-2.0). The catalog
    # display name "MOSS-VoiceGenerator" tokenizes to "moss
    # voicegenerator" → first-token "moss" which does NOT match any
    # `crates/vokra-convert/src/models/moss*.rs` module (the auto slug
    # walk would land on the ambiguous `moss` token). This explicit
    # alias points every display-name variant back at the sibling
    # `moss_tts` module — the same converter dispatch the
    # `moss-voice-generator` CLI slug is routed through, per the
    # slug-only landing decision recorded in §3.1 and
    # `scripts/publish/signoff_match.py::REPO_TO_SIGNOFF_ROWS`. Landed
    # in the same wave as the §3.1 row so a hypothetical future §3
    # catalog row for MOSS-VoiceGenerator would resolve
    # `implemented("MOSS-VoiceGenerator")` = true without a follow-up
    # to this file.
    "moss voice generator"*|"moss-voice-generator"*|"moss voicegenerator"*|"moss-voicegenerator"*) printf 'moss_tts\n' ;;
    # 2026-08-01 Wave 3: Amphion NaturalSpeech 3 FACodec — factorized VQ
    # codec (apache-2.0). The catalog display name "NaturalSpeech 3
    # FACodec (Amphion)" tokenizes to "naturalspeech 3 facodec" ->
    # first-token "naturalspeech", which does NOT match the module file
    # `crates/vokra-convert/src/models/naturalspeech3_facodec.rs`. This
    # explicit alias maps every display-name variant to the actual
    # module slug so the double-sided catalog-vs-implementation gate
    # sees the converter as implemented.
    "naturalspeech 3 facodec"*|"naturalspeech3 facodec"*|"naturalspeech3-facodec"*|"facodec"*|"ns3 facodec"*|"ns3-facodec"*)
      printf 'naturalspeech3_facodec\nfacodec\n' ;;
    # 2026-08-01 Wave 3 sibling-pair: YuE bundle
    # (`m-a-p/YuE-upsampler` = vocoder / `m-a-p/xcodec_mini_infer` =
    # codec, both apache-2.0). The catalog display names ("YuE-upsampler" /
    # "YuE xcodec-mini") tokenize to "yue upsampler" / "yue xcodec mini"
    # → first-token "yue", which does NOT match the shared module file
    # `crates/vokra-convert/src/models/yue_bundle.rs`. This explicit
    # alias maps every display-name variant to the actual module slug
    # `yue_bundle` so the double-sided catalog-vs-implementation gate
    # sees the converter as implemented for both sibling rows.
    "yue upsampler"*|"yue-upsampler"*|"yue xcodec mini"*|"yue-xcodec-mini"*|"yue xcodec-mini"*|"yue"*)
      printf 'yue_bundle\n' ;;
    # 2026-08-01 Wave 5 music-generation add: Meta AudioCraft MusicGen family
    # (`facebook/musicgen-{medium,large}`, cc-by-nc-4.0). First
    # `category = "music"` entry in the tree. Two distinct HF repos + two
    # distinct sibling files (`musicgen_medium.rs` + `musicgen_large.rs`,
    # the chatterbox / chatterbox_turbo / chatterbox_nano split) rather
    # than a shared `musicgen.rs` variant enum. Bash case patterns are
    # evaluated top-to-bottom, first match wins — the specific size
    # patterns (medium / large) catch first so their display names route
    # to the correct sibling file. The generic `musicgen` catch-all
    # (used when only the bare arch tag appears in a catalog display
    # name) prints BOTH sibling slugs so `implemented()` returns true if
    # any sibling file exists (either sibling standing in for the
    # family is a valid answer to "is the MusicGen family
    # implemented?"). Future family variants
    # (`musicgen-small` / `musicgen-melody` / `musicgen-stereo-*`) will
    # add their own specific pattern above the generic catch-all.
    "musicgen medium"*|"musicgen-medium"*) printf 'musicgen_medium\n' ;;
    "musicgen large"*|"musicgen-large"*) printf 'musicgen_large\n' ;;
    "musicgen"*|"musicgen-"*) printf 'musicgen_medium\nmusicgen_large\n' ;;
    # 2026-08-01 Wave 5 music-generation add: AudioLDM 2
    # (`cvssp/audioldm2`, cc-by-nc-sa-4.0). First non-AR audio-
    # generation converter (sibling musicgen family is AR + RVQ;
    # AudioLDM 2 is latent-diffusion + VAE, distinct topology / arch
    # tag). Present in this alias table so a future `★ 公式 zoo`
    # promotion can immediately resolve the display name to the
    # `audioldm2` module slug — but note the row is currently
    # unpublishable (NC + SA cascade requires an owner ADR before the
    # §3.1 sign-off is filled and a `REPO_TO_SIGNOFF_ROWS` entry is
    # added), so today's row does NOT carry the `★ 公式 zoo` marker
    # and this scanner never runs `implemented("AudioLDM 2")` against
    # it. Future family variants (`audioldm2-music` /
    # `audioldm2-large` / `audioldm2-music-665k`) will add their own
    # specific pattern above this catch-all if a shared enum split
    # doesn't collapse them into the single `audioldm2` slug.
    "audioldm 2"*|"audioldm-2"*|"audioldm2"*|"audio-ldm-2"*) printf 'audioldm2\n' ;;
    # 2026-08-01 Wave 5 music-separation add: BS-Roformer /
    # Mel-Band Roformer (`chenmozhijin/BSRoformer-GGUF` third-
    # party mirror, **weight provenance unclear** — first music-
    # source-separation converter, Lu et al. 2023
    # arXiv:2310.01809). Present in this alias table so a future
    # `★ 公式 zoo` promotion can immediately resolve the display
    # name to the `bs_roformer` module slug — but note the row is
    # currently unpublishable (weight provenance unclear ⇒
    # `LicenseClass::RedistributionForbidden` fail-closed default
    # + no `REPO_TO_SIGNOFF_ROWS` entry ⇒ `UNKNOWN_REPO` at
    # publish gate time). Today's row does NOT carry the
    # `★ 公式 zoo` marker and this scanner never runs
    # `implemented("BS-Roformer …")` against it. Aliases cover
    # the arch tag spellings and the family-name variants
    # ("BS-Roformer" / "Mel-Band Roformer" / "MelBand Roformer" /
    # "BSRoformer" — first tokens all resolve into different
    # buckets so the family-name catch-all matters).
    "bs roformer"*|"bs-roformer"*|"bs_roformer"*|"bsroformer"*|\
"mel band roformer"*|"mel-band-roformer"*|"melband roformer"*|"melband-roformer"*)
      printf 'bs_roformer\n' ;;
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
