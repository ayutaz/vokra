#!/usr/bin/env bash
# verify-piper-voices.sh — per-language piper-plus voice conversion + synthesis
# smoke harness (M1-01-E). Records the coverage matrix M1-01-F fills in.
#
# For each distributed voice (ONNX + config.json) it: (1) converts to GGUF with
# `vokra-convert` (M0-03/M0-07), capturing the derived iSTFT/PQMF params and any
# over-range / defaulted-param warnings (M1-01-E); (2) runs the voice-agnostic
# `session_tts_api_smoke` (loads the GGUF and synthesizes deterministic audio
# through the placeholder tokenizer — no reference fixtures, so it works for any
# voice); (3) optionally records the RTF from the M1-01-D bench. Rows are
# appended to a matrix doc.
#
# Only 6 languages (ja/en/zh/es/fr/pt) have a DISTRIBUTED voice; ko/sv voices
# are not distributed (docs/piper-plus-integration.md §2.6), so a true
# 8-language result needs client-provided ko/sv checkpoints — this harness is
# buildable now, the 8-language RESULT is M1-01-F (client/HW-blocked).
#
# Like the parity tests, it DEGRADES TO A SKIP when no voices are supplied — it
# never invents inputs or numbers (CLAUDE.md hallucination red line).
#
# Usage:
#   scripts/verify-piper-voices.sh [VOICES_DIR]
#   VOKRA_PIPER_VOICES=<dir>   voices dir (each <name>.onnx + its config json)
#   VOKRA_PIPER_MATRIX=<file>  output matrix (default: target/piper-voice-matrix.md)
#   VOKRA_PIPER_RTF=1          also run the M1-01-D RTF bench per voice (slow)
#
# Exit code: 0 = ran what was available (a missing voices dir is a skip, a
#            per-voice failure is recorded, not fatal); non-zero = a harness
#            error (bad args, unwritable output).

# The backticks in the printf format strings below are LITERAL markdown code
# spans for the matrix cells, not command substitution — they are intentionally
# single-quoted.
# shellcheck disable=SC2016
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VOICES_DIR="${1:-${VOKRA_PIPER_VOICES:-}}"
MATRIX="${VOKRA_PIPER_MATRIX:-$ROOT/target/piper-voice-matrix.md}"
WORK="$ROOT/target/piper-voices"

if [ -z "$VOICES_DIR" ] || [ ! -d "$VOICES_DIR" ]; then
    echo "verify-piper-voices: no voices dir (set VOKRA_PIPER_VOICES=<dir> or pass it as \$1) — skipped"
    echo "  a voice = <name>.onnx plus its config json (<name>.onnx.json | <name>.json | config.json)"
    exit 0
fi

mkdir -p "$WORK" "$(dirname "$MATRIX")"

# Find the config.json paired with an ONNX file (piper ships <name>.onnx.json).
find_config() {
    local onnx="$1" stem
    stem="${onnx%.onnx}"
    for cand in "$onnx.json" "$stem.json" "$(dirname "$onnx")/config.json"; do
        if [ -f "$cand" ]; then
            printf '%s' "$cand"
            return 0
        fi
    done
    return 1
}

{
    echo "# piper-plus per-language voice matrix (M1-01-E harness)"
    echo
    echo "- Generated: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    echo "- Voices dir: \`$VOICES_DIR\`"
    echo "- Host: $(uname -sm)"
    echo
    echo "| voice | convert | converter note | synth smoke | RTF |"
    echo "|-------|---------|----------------|-------------|-----|"
} > "$MATRIX"

shopt -s nullglob
found=0
for onnx in "$VOICES_DIR"/*.onnx; do
    found=1
    name="$(basename "${onnx%.onnx}")"
    gguf="$WORK/$name.gguf"

    if ! config="$(find_config "$onnx")"; then
        echo "verify-piper-voices: $name — no config json alongside $onnx" >&2
        printf '| `%s` | NO CONFIG | — | — | — |\n' "$name" >> "$MATRIX"
        continue
    fi

    # (1) convert — capture the summary/warning lines for the matrix cell.
    if convert_out="$(cargo run --quiet -p vokra-convert -- \
        --model piper-plus --input "$onnx" --config "$config" --output "$gguf" 2>&1)"; then
        convert_cell="PASS"
    else
        convert_cell="FAIL"
    fi
    note="$(printf '%s' "$convert_out" | grep -iE 'piper-plus:|over |default|warn' | head -1 | tr '|' '/' )"
    note="${note:-—}"

    if [ "$convert_cell" != "PASS" ] || [ ! -f "$gguf" ]; then
        printf '| `%s` | %s | %s | — | — |\n' "$name" "$convert_cell" "$note" >> "$MATRIX"
        continue
    fi

    # (2) synthesis smoke — voice-agnostic (placeholder tokenizer, no fixtures).
    if VOKRA_PIPER_GGUF="$gguf" cargo test --quiet -p vokra-models -- \
        session_tts_api_smoke >/dev/null 2>&1; then
        smoke_cell="PASS"
    else
        smoke_cell="FAIL"
    fi

    # (3) optional RTF (M1-01-D bench), when explicitly requested.
    rtf_cell="—"
    if [ "${VOKRA_PIPER_RTF:-0}" != "0" ]; then
        rtf_line="$(VOKRA_PIPER_GGUF="$gguf" cargo bench --quiet -p vokra-models \
            --bench piper_rtf 2>/dev/null | grep -oE 'RTF [0-9.]+' | head -1 || true)"
        rtf_cell="${rtf_line:-n/a}"
    fi

    printf '| `%s` | %s | %s | %s | %s |\n' \
        "$name" "$convert_cell" "$note" "$smoke_cell" "$rtf_cell" >> "$MATRIX"
    echo "verify-piper-voices: $name — convert=$convert_cell smoke=$smoke_cell rtf=$rtf_cell"
done

if [ "$found" -eq 0 ]; then
    echo "verify-piper-voices: no *.onnx in $VOICES_DIR — skipped"
    exit 0
fi

echo "verify-piper-voices: matrix written to $MATRIX"
echo "  (ko/sv rows require client-provided checkpoints — see docs §2.6 / M1-01-F)"
