#!/usr/bin/env bash
# M3-11 T19 — headless verification of the Vokra GDExtension against a real
# Godot binary.
#
# What this proves that `cargo test -p vokra-godot` cannot: the crate's unit
# tests drive a mock ClassDB, so they validate the shape we *intend* to hand
# Godot. This leg loads the built cdylib into a real Godot, which exercises
# Godot's own ClassDB, Variant marshalling, instance binding and method
# dispatch. Three defects that every mock-level test passed were caught here:
#
#   1. `GDExtensionPropertyInfo::{class_name,hint_string}` were NULL. Godot
#      dereferences both unconditionally → SIGSEGV inside
#      `GDExtension::_register_extension_class_method`, i.e. the extension
#      killed the host the first time any method was registered.
#   2. `create_instance_func` returned the bare `Box::into_raw` pointer.
#      Godot `dynamic_cast`s that as `Object *` → SIGSEGV on the first
#      `VokraSession.new()`.
#   3. This harness's own negative leg hung forever: a GDScript CallError
#      aborts `_initialize()` without stopping the headless main loop, so
#      the process never exited. Worse, it looked like a PASS — the expected
#      error text is printed before the stall, so a killed run's partial
#      output still matched. Fixed by arming `quit()` ahead of the raising
#      call (verify_unloaded.gd) and by putting every Godot invocation under
#      `run_limited`, which reports a timeout instead of matching text from a
#      process that never finished.
#
# 1 and 2 reproduce identically on Godot 4.3-stable and 4.7.1-stable; all
# three are fixed, and this script is the regression gate.
#
# Usage:
#   GODOT=/path/to/Godot scripts/verify-godot-headless.sh                    # asset-free
#   GODOT=/path/to/Godot scripts/verify-godot-headless.sh MODEL.gguf AUDIO.wav
#
# `GODOT` may point at the binary itself or at a macOS `Godot.app` bundle.
# Godot is a free download (https://godotengine.org/download) — no account,
# no license key. If GODOT is unset the script exits 2 with an explicit
# message: an absent verifier is announced, never silently treated as a pass
# (FR-EX-08 discipline applied to the harness itself).
#
# Scope: this is the CC half of M3-11 T19. Owner tasks that remain are the
# interactive editor confirmation (the demo scenes under demos/, driven by a
# human in the Godot GUI) and the T20 WP-close PR.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CRATE_DIR="$REPO_ROOT/integrations/vokra-godot"
MODEL_PATH="${1:-}"
AUDIO_PATH="${2:-}"
# Per-Godot-invocation watchdog. Generous enough for a cold whisper-base
# transcription on a loaded machine; short enough that a hang is reported
# rather than waited out.
GODOT_TIMEOUT="${GODOT_TIMEOUT:-600}"

# Run a command under a wall-clock limit, portably.
#
# `timeout(1)` is GNU coreutils and is NOT present on a stock macOS, which is
# the primary dev machine for this crate — so the watchdog is built from a
# background killer instead. It matters: a GDScript runtime error aborts the
# script function without stopping the headless main loop, so a Godot leg CAN
# hang indefinitely (it did, before verify_unloaded.gd learned to arm quit()
# ahead of the raising call). Exit 124 mirrors timeout(1)'s convention.
run_limited() {
  local secs="$1"; shift
  "$@" &
  local pid=$!
  ( sleep "$secs"; kill -9 "$pid" 2>/dev/null ) &
  local killer=$!
  local rc=0
  wait "$pid" 2>/dev/null || rc=$?
  # Retiring the killer keeps it from reaping an unrelated recycled PID.
  kill "$killer" 2>/dev/null || true
  wait "$killer" 2>/dev/null || true
  # 137 = SIGKILL, i.e. the watchdog fired.
  if [[ "$rc" -eq 137 ]]; then
    echo "  TIMEOUT after ${secs}s: $*" >&2
    return 124
  fi
  return "$rc"
}

# --- locate the Godot binary ------------------------------------------------
if [[ -z "${GODOT:-}" ]]; then
  echo "ERROR: set GODOT to a Godot 4.1+ binary (or Godot.app bundle)." >&2
  echo "       Download: https://godotengine.org/download" >&2
  echo "       This leg is SKIPPED, not passed." >&2
  exit 2
fi
GODOT_BIN="$GODOT"
if [[ -d "$GODOT_BIN" ]]; then
  # macOS .app bundle → the executable inside it.
  GODOT_BIN="$GODOT_BIN/Contents/MacOS/Godot"
fi
if [[ ! -x "$GODOT_BIN" ]]; then
  echo "ERROR: '$GODOT_BIN' is not an executable Godot binary." >&2
  exit 2
fi
echo "Godot binary : $GODOT_BIN"
"$GODOT_BIN" --headless --version

# --- build the cdylib -------------------------------------------------------
echo "Building vokra-godot cdylib (release) ..."
( cd "$CRATE_DIR" && cargo build --release )

case "$(uname -s)" in
  Darwin) LIB_NAME="libvokra_godot.dylib"; PLATFORM_DIR="macos" ;;
  Linux)  LIB_NAME="libvokra_godot.so";    PLATFORM_DIR="linux" ;;
  *) echo "ERROR: unsupported host '$(uname -s)' for the headless leg." >&2; exit 2 ;;
esac
case "$(uname -m)" in
  arm64|aarch64) ARCH_DIR="arm64" ;;
  x86_64|amd64)  ARCH_DIR="x86_64" ;;
  *) echo "ERROR: unsupported arch '$(uname -m)'." >&2; exit 2 ;;
esac
LIB_PATH="$CRATE_DIR/target/release/$LIB_NAME"
[[ -f "$LIB_PATH" ]] || { echo "ERROR: $LIB_PATH not produced." >&2; exit 1; }

# --- assemble a throwaway Godot project ------------------------------------
PROJECT_DIR="$(mktemp -d)"
trap 'rm -rf "$PROJECT_DIR"' EXIT
mkdir -p "$PROJECT_DIR/addons/vokra/bin/$PLATFORM_DIR/$ARCH_DIR" "$PROJECT_DIR/.godot"
cp "$CRATE_DIR/vokra.gdextension" "$PROJECT_DIR/addons/vokra/"
cp "$LIB_PATH" "$PROJECT_DIR/addons/vokra/bin/$PLATFORM_DIR/$ARCH_DIR/"
cp "$CRATE_DIR/tests/headless/verify.gd" "$CRATE_DIR/tests/headless/verify_unloaded.gd" "$PROJECT_DIR/"
# Godot only dlopens extensions listed here; the editor normally writes it on
# first scan, which a headless run never performs.
printf 'res://addons/vokra/vokra.gdextension\n' > "$PROJECT_DIR/.godot/extension_list.cfg"
cat > "$PROJECT_DIR/project.godot" <<'PROJ'
[application]
config/name="vokra-godot-headless-verify"
config/features=PackedStringArray("4.1")
PROJ

status=0

# --- leg 1: positive path ---------------------------------------------------
echo
if [[ -n "$MODEL_PATH" && -n "$AUDIO_PATH" ]]; then
  [[ -f "$MODEL_PATH" ]] || { echo "ERROR: model '$MODEL_PATH' not found." >&2; exit 2; }
  [[ -f "$AUDIO_PATH" ]] || { echo "ERROR: audio '$AUDIO_PATH' not found." >&2; exit 2; }
  run_limited "$GODOT_TIMEOUT" "$GODOT_BIN" --headless --path "$PROJECT_DIR" \
    --script res://verify.gd -- "$MODEL_PATH" "$AUDIO_PATH" || status=1
else
  echo "No model/audio given → asset-free checks only."
  run_limited "$GODOT_TIMEOUT" "$GODOT_BIN" --headless --path "$PROJECT_DIR" \
    --script res://verify.gd || status=1
fi

# --- leg 2: negative path ---------------------------------------------------
# An unloaded session must REFUSE the call. Godot renders our
# InvalidMethod CallError as "Nonexistent function"; matching that text is
# how we assert the refusal from outside the GDScript VM.
echo
echo "== negative path: transcribe on an unloaded session =="
# Output goes to a file rather than a command substitution so the run can sit
# under the watchdog. A hang here used to be indistinguishable from a slow
# pass: the CallError text is printed BEFORE the loop stalls, so grepping a
# killed process's partial output reported a false PASS. The timeout is now
# checked first, and separately from the text match.
neg_log="$PROJECT_DIR/negative.log"
neg_rc=0
run_limited "$GODOT_TIMEOUT" "$GODOT_BIN" --headless --path "$PROJECT_DIR" \
  --script res://verify_unloaded.gd >"$neg_log" 2>&1 || neg_rc=$?
if [[ "$neg_rc" -eq 124 ]]; then
  echo "  FAIL  negative path did not terminate within ${GODOT_TIMEOUT}s"
  echo "        (a GDScript error aborts _initialize() without stopping the"
  echo "         headless loop — verify_unloaded.gd must arm quit() first)"
  cat "$neg_log"
  status=1
elif grep -q "Nonexistent function 'transcribe'" "$neg_log"; then
  echo "  PASS  unloaded session refuses transcribe with an explicit CallError"
elif grep -q "MARKER: unexpected-success" "$neg_log"; then
  echo "  FAIL  unloaded session returned a transcript (FR-EX-08 violation)"
  cat "$neg_log"
  status=1
else
  echo "  FAIL  negative path produced neither the expected CallError nor a transcript"
  cat "$neg_log"
  status=1
fi

echo
if [[ "$status" -eq 0 ]]; then
  echo "verify-godot-headless: PASS"
else
  echo "verify-godot-headless: FAIL"
fi
exit "$status"
