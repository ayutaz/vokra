#!/usr/bin/env bash
# run-capi-smoke.sh — build the C ABI and run the C smoke tests (M0-09-T12).
#
# Ticket: M0-09-T12 (local stand-in for the CI `capi` job — CI YAML wiring is a
# followup). Steps:
#   1. cargo build -p vokra-capi --release   (libvokra cdylib + header inputs)
#   2. scripts/gen-c-abi.sh --check          (vokra.h has no drift — WP cond.)
#   3. exported-symbol check                 (only vokra_* symbols are public)
#   4. cc-build + run tests/capi/smoke_{vad,asr,tts}.c against <vokra.h>
#
# VAD uses the committed 2 MB Silero fixture. ASR/TTS are ENV-GATED: they SKIP
# cleanly unless VOKRA_WHISPER_GGUF / VOKRA_PIPER_GGUF point at the (uncommitted)
# Whisper / piper GGUFs.
#
# Exit code: 0 = all pass/skip, non-zero = a build or test failure.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REL="$ROOT/target/release"
CC="${CC:-cc}"

case "$(uname -s)" in
    Darwin)
        LIB="$REL/libvokra.dylib"
        NM_ARGS="-gU"
        export DYLD_LIBRARY_PATH="$REL:${DYLD_LIBRARY_PATH:-}"
        ;;
    Linux)
        LIB="$REL/libvokra.so"
        NM_ARGS="-D --defined-only"
        export LD_LIBRARY_PATH="$REL:${LD_LIBRARY_PATH:-}"
        ;;
    *)
        echo "run-capi-smoke: unsupported OS $(uname -s)" >&2
        exit 1
        ;;
esac

echo "== 1. cargo build -p vokra-capi --release =="
( cd "$ROOT" && cargo build -p vokra-capi --release )

echo "== 2. gen-c-abi.sh --check (header drift) =="
bash "$ROOT/scripts/gen-c-abi.sh" --check

echo "== 3. exported-symbol check =="
if [ ! -f "$LIB" ]; then
    echo "run-capi-smoke: FAIL missing $LIB" >&2
    exit 1
fi
# Every exported symbol must be vokra_-prefixed (after stripping a leading '_').
syms="$(nm $NM_ARGS "$LIB" 2>/dev/null | awk 'NF>=2 {print $NF}' | sed 's/^_//')"
unexpected="$(printf '%s\n' "$syms" | grep -vE '^vokra_' | grep -vE '^$' || true)"
if [ -n "$unexpected" ]; then
    echo "run-capi-smoke: FAIL unexpected exported symbols:" >&2
    printf '  %s\n' "$unexpected" >&2
    exit 1
fi
count="$(printf '%s\n' "$syms" | grep -cE '^vokra_' || true)"
echo "run-capi-smoke: $count vokra_* symbols exported, no leaks"

echo "== 4. build + run C smoke tests =="
TMP="$(mktemp -d -t vokra-capi-smoke.XXXXXX)"
trap 'rm -rf "$TMP"' EXIT
CFLAGS="-std=c11 -Wall -Wextra -Werror -I$ROOT/include"
LDFLAGS="-L$REL -lvokra -Wl,-rpath,$REL"

build_one() {
    "$CC" $CFLAGS "$ROOT/tests/capi/smoke_$1.c" $LDFLAGS -o "$TMP/smoke_$1"
}

status=0
build_one vad
"$TMP/smoke_vad" "$ROOT/tests/parity/silero_vad/silero-vad-v5.gguf" \
    "$ROOT/tests/capi/fixtures/vad_input_16k.f32" || status=1

build_one asr
"$TMP/smoke_asr" "$ROOT/tests/capi/fixtures/asr_input_16k.f32" || status=1

build_one tts
"$TMP/smoke_tts" || status=1

if [ "$status" -eq 0 ]; then
    echo "run-capi-smoke: OK"
else
    echo "run-capi-smoke: FAILED" >&2
fi
exit "$status"
