#!/usr/bin/env bash
# fetch-demo-models.sh — stage the Unity demo's models and test audio (M0-10-T09).
#
# Places into examples/unity-demo/Assets/StreamingAssets/:
#   models/silero-vad-v5.gguf   <- committed fixture (VAD works out of the box)
#   test_16k.wav                <- generated from the committed VAD f32 fixture
#   models/whisper-base.gguf    <- from $VOKRA_WHISPER_GGUF if set (else: instructions)
#   models/voice.gguf           <- from $VOKRA_PIPER_GGUF   if set (else: instructions)
#
# ASR/TTS checkpoints are large and uncommitted (like the M0-09 C-smoke gating);
# convert them with `vokra-convert` (M0-03) from your own Whisper base / piper-plus
# voice, or point the env vars at ready-made GGUFs. No network downloads and no
# invented URLs here.
#
# Exit code: 0 = staged what is available (missing large models are a note, not a
# failure), non-zero = a copy/generate error.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SA="$ROOT/examples/unity-demo/Assets/StreamingAssets"
MODELS="$SA/models"
mkdir -p "$MODELS"

# --- VAD: committed fixture -------------------------------------------------
SILERO_SRC="$ROOT/tests/parity/silero_vad/silero-vad-v5.gguf"
if [ -f "$SILERO_SRC" ]; then
    cp -f "$SILERO_SRC" "$MODELS/silero-vad-v5.gguf"
    echo "fetch-demo-models: placed models/silero-vad-v5.gguf"
else
    echo "fetch-demo-models: WARN committed Silero fixture missing: $SILERO_SRC" >&2
fi

# --- test WAV: wrap the committed raw-f32 VAD fixture as a 16 kHz mono WAV ---
F32_SRC="$ROOT/tests/capi/fixtures/vad_input_16k.f32"
WAV_DST="$SA/test_16k.wav"
if [ -f "$F32_SRC" ]; then
    if command -v python3 >/dev/null 2>&1; then
        python3 - "$F32_SRC" "$WAV_DST" <<'PY'
import struct, sys
src, dst = sys.argv[1], sys.argv[2]
raw = open(src, "rb").read()
rate = 16000  # the fixture is 16 kHz mono float32 (tests/capi/README.md)
with open(dst, "wb") as f:
    f.write(b"RIFF"); f.write(struct.pack("<I", 36 + len(raw))); f.write(b"WAVE")
    f.write(b"fmt "); f.write(struct.pack("<I", 16))
    # audioFormat=3 (IEEE float), channels=1, rate, byteRate, blockAlign=4, bits=32
    f.write(struct.pack("<HHIIHH", 3, 1, rate, rate * 4, 4, 32))
    f.write(b"data"); f.write(struct.pack("<I", len(raw))); f.write(raw)
print("fetch-demo-models: generated test_16k.wav (%d samples)" % (len(raw) // 4))
PY
    else
        echo "fetch-demo-models: WARN python3 not found; skipping test_16k.wav generation." >&2
        echo "  Provide your own 16 kHz mono WAV at $WAV_DST, or read the raw .f32 directly." >&2
    fi
else
    echo "fetch-demo-models: WARN VAD f32 fixture missing: $F32_SRC" >&2
fi

# --- ASR / TTS: large, uncommitted ------------------------------------------
stage_env() {
    local name="$1" env_val="$2" dst="$3"
    if [ -n "$env_val" ]; then
        if [ -f "$env_val" ]; then
            cp -f "$env_val" "$dst"
            echo "fetch-demo-models: placed $(basename "$dst") from \$$name"
        else
            echo "fetch-demo-models: WARN \$$name=$env_val does not exist" >&2
        fi
    else
        echo "fetch-demo-models: $(basename "$dst") not staged (set \$$name or convert with vokra-convert)"
    fi
}

stage_env VOKRA_WHISPER_GGUF "${VOKRA_WHISPER_GGUF:-}" "$MODELS/whisper-base.gguf"
stage_env VOKRA_PIPER_GGUF   "${VOKRA_PIPER_GGUF:-}"   "$MODELS/voice.gguf"

echo
echo "fetch-demo-models: done."
echo "  VAD runs from the committed fixture; ASR/TTS light up once their GGUFs are placed."
echo "  See examples/unity-demo/Assets/StreamingAssets/models/README.md for conversion."
