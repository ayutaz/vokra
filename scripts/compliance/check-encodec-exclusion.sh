#!/usr/bin/env bash
# check-encodec-exclusion.sh — enforce FR-OP-32 (permanent EnCodec weight
# exclusion from the official model zoo).
#
# EnCodec's pretrained weights are CC-BY-NC 4.0 (non-commercial). The M2-13
# compliance gate refuses to *load* them without an explicit research flag
# (see `crates/vokra-core/src/compliance/`). This script is the
# *distribution*-side complement: it fails release CI if any EnCodec weight
# file has slipped into the release tarball or the working tree's shipping
# paths.
#
# What counts as an EnCodec weight:
#   * `*.safetensors` / `*.gguf` / `*.pth` / `*.bin` whose *filename* matches
#     the case-insensitive substring "encodec".
#   * A GGUF whose `vokra.model.arch` or `vokra.provenance.model_id` says
#     "encodec" is caught by the M2-13 runtime gate, not this script — this
#     script only looks at the file *system*, so it stays zero-dep (no GGUF
#     parser needed).
#
# What is NOT caught:
#   * The `mimi_rvq` *op* itself, which is codebook-shaped and can host an
#     EnCodec code path — that is expected to exist and is not weight data.
#   * A developer's local checkout of an EnCodec checkpoint under a directory
#     ignored by `.gitignore` (e.g. their own cache). This script only checks
#     paths that could conceivably ship — the repo tree and, optionally, a
#     release-tarball path passed on the command line.
#
# Zero-dependency: pure `bash` + `find` + `grep`. No Python, no cargo, no jq.
#
# Exit code: 0 = clean, 1 = a candidate EnCodec weight was found.
#
# Wire this into release CI (final gate before publishing artefacts) and
# have the M2-13 compliance job invoke it as well so a stray weight fails
# the PR long before a release is cut.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
# Default search root = the repo, minus tree paths that never ship (build
# artefacts, target/, python venvs, tools that only exist for parity dumps
# and are not part of the wheel or the C ABI tarball).
SEARCH_ROOTS=("$ROOT/crates" "$ROOT/model_zoo" "$ROOT/docs")

# Extra path (optional first arg): a release-tarball directory to also scan.
if [ "${1:-}" != "" ]; then
    if [ ! -d "$1" ]; then
        echo "error: extra scan path '$1' is not a directory" >&2
        exit 2
    fi
    SEARCH_ROOTS+=("$1")
fi

# Build a `-path`-safe list of existing roots (skip missing directories
# quietly so the script works in a shallow checkout).
EXISTING_ROOTS=()
for r in "${SEARCH_ROOTS[@]}"; do
    if [ -e "$r" ]; then
        EXISTING_ROOTS+=("$r")
    fi
done
if [ ${#EXISTING_ROOTS[@]} -eq 0 ]; then
    echo "check-encodec-exclusion: no search roots exist — nothing to scan." >&2
    exit 0
fi

# Weight-file extensions we ever ship, ranked by prevalence in the ecosystem.
EXTENSIONS=("safetensors" "gguf" "pth" "bin")

matches=""
for ext in "${EXTENSIONS[@]}"; do
    # `find -iname` matches case-insensitively, catching "EnCodec" /
    # "encodec_24khz.safetensors" / etc.
    while IFS= read -r hit; do
        [ -z "$hit" ] && continue
        matches+="$hit"$'\n'
    done < <(find "${EXISTING_ROOTS[@]}" -type f -iname "*encodec*.$ext" 2>/dev/null || true)
done

if [ -n "$matches" ]; then
    echo "check-encodec-exclusion: FR-OP-32 violation — EnCodec weight file(s) found in the shipping tree:" >&2
    echo "$matches" >&2
    echo "" >&2
    echo "EnCodec pretrained weights are CC-BY-NC 4.0 (non-commercial) and are" >&2
    echo "permanently excluded from the official model zoo. See:" >&2
    echo "  - docs/license-audit.md §3 (EnCodec row)" >&2
    echo "  - docs/adr/M3-06-mimi-rvq.md §D2" >&2
    echo "" >&2
    echo "Commercial codec alternatives: DAC (MIT) / Mimi (CC-BY 4.0) /" >&2
    echo "WavTokenizer (MIT) / X-Codec 2 (MIT)." >&2
    exit 1
fi

# Extra defence: grep the vokra-convert crate to confirm no code-path emits
# an EnCodec GGUF into the permissive/attribution model-zoo publish branch.
# This mirrors `docs/license-audit.md` §3's "grep 検証" line.
#
# A hit is only an alarm if it appears outside:
#   (a) a `// …` comment line, or
#   (b) a `#[cfg(test)]` module (Rust convention: tests live at end of file
#       inside `#[cfg(test)] mod tests { … }`; we skip everything after the
#       *first* `#[cfg(test)]` marker in a file), or
#   (c) an explicit substring allowlist for third-party slugs that happen
#       to embed the token "encodec" without being Meta's EnCodec codec.
#       Each entry MUST document a distinct upstream project and why its
#       slug cannot be renamed (provenance breadcrumb integrity). Adding
#       an entry is a compliance-review action, not a routine change.
#
# Slug allowlist (line-substring match against the raw grep hit).
# Format: exact substring that MUST appear on the source line for the hit
# to be exempted; anything else remains scanner-flagged.
#
# The FunCodec entry documents: the upstream ModelScope release
# `alibaba-damo/audio_codec-encodec-zh_en-general-16k-nq32ds320-pytorch`
# is FunCodec (Alibaba DAMO Academy) — a *separate* codec from Meta's
# EnCodec. The literal token "encodec" in the slug is unfortunate
# ModelScope naming, not the CC-BY-NC-restricted Meta codec. We keep the
# slug byte-exact because it is what `vokra.provenance.upstream_hf` is
# stamped with, and the M2-13 runtime gate cross-checks byte-for-byte
# against upstream. See:
#   - crates/vokra-convert/src/models/funcodec.rs (converter)
#   - docs/license-audit.md §3.1 (FunCodec sign-off row, MIT)
#   - PR #20 (SoTA Phase 1-4 + JA + BF16 fleet)
SLUG_ALLOWLIST=(
    'audio_codec-encodec-zh_en'
    'audio_codec-encodec-zh-en'
    'funcodec-encodec-zh-en'
    'funcodec-encodec-zh_en'
)

if [ -d "$ROOT/crates/vokra-convert/src" ]; then
    grep_hits="$(grep -rniE 'encodec' "$ROOT/crates/vokra-convert/src/" 2>/dev/null || true)"
    if [ -n "$grep_hits" ]; then
        offending=""
        while IFS= read -r hit; do
            [ -z "$hit" ] && continue
            # Format: file:linenum:content
            file="${hit%%:*}"
            rest="${hit#*:}"
            linenum="${rest%%:*}"
            content="${rest#*:}"
            # Skip pure comment lines.
            case "$content" in
                *//*) # rough — a `//` anywhere on the line. Refine only if
                      # ever a false positive shows up.
                    # Only skip if the // starts before the "encodec" token.
                    before="${content%%encodec*}"
                    case "$before" in
                        *//*) continue ;;
                    esac
                    ;;
            esac
            # Skip lines inside a `#[cfg(test)]` block (all lines after the
            # first cfg(test) marker in that file).
            cfg_test_line="$(awk '/^[[:space:]]*#\[cfg\(test\)\]/ { print NR; exit }' "$file" 2>/dev/null || true)"
            if [ -n "$cfg_test_line" ] && [ "$linenum" -gt "$cfg_test_line" ]; then
                continue
            fi
            # Slug allowlist — exempt third-party projects whose upstream
            # slug happens to embed "encodec" without being Meta's EnCodec.
            allowlisted=""
            for slug in "${SLUG_ALLOWLIST[@]}"; do
                case "$content" in
                    *"$slug"*)
                        allowlisted="1"
                        break
                        ;;
                esac
            done
            if [ -n "$allowlisted" ]; then
                continue
            fi
            offending+="$hit"$'\n'
        done <<< "$grep_hits"

        if [ -n "$offending" ]; then
            echo "check-encodec-exclusion: vokra-convert code path references 'encodec' outside comments and #[cfg(test)]:" >&2
            printf '%s' "$offending" >&2
            echo "" >&2
            echo "Move the reference into a comment (documenting the exclusion), or" >&2
            echo "into a #[cfg(test)] block asserting the M2-13 gate refuses it," >&2
            echo "or (only for third-party projects whose upstream slug embeds" >&2
            echo "'encodec' without being Meta's codec) extend SLUG_ALLOWLIST above" >&2
            echo "with a compliance-review-approved substring." >&2
            exit 1
        fi
    fi
fi

echo "check-encodec-exclusion: OK (no EnCodec weight files found; vokra-convert clean)."
