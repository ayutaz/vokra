# M3-11 T19 — negative path: methods on a session that never loaded a model.
#
# `SessionInstance::inner == None` makes the trampolines report
# `GDEXTENSION_CALL_ERROR_INVALID_METHOD` (see
# trampoline::dispatch_session_transcribe step 4). That is the FR-EX-08
# posture: an unusable session refuses the call rather than fabricating an
# empty transcript.
#
# Godot renders that CallError as the runtime error
#   "Invalid call. Nonexistent function 'transcribe' in base 'VokraSession'."
# and aborts the calling function, so the assertion cannot be made from
# inside GDScript — scripts/verify-godot-headless.sh matches that text on
# this script's output instead.
#
# This script therefore only has to REACH the call. If the transcript ever
# came back successfully, the `print` below would run and the shell wrapper
# would fail the leg on the unexpected marker.
#
# TERMINATION (this bit is load-bearing — getting it wrong hangs the run)
# ----------------------------------------------------------------------
# The CallError aborts `_initialize()` at the `transcribe` line, so ANY
# `quit()` written after that line is dead code and the headless SceneTree
# spins forever with no window to close. That is not hypothetical: it hung a
# verification run for >3 minutes until killed, and the leg only appeared to
# pass because killing it left the already-printed error text in the buffer
# the wrapper greps.
#
# The fix is to arm the exit BEFORE the raising call. `SceneTree.quit()` sets
# a flag consumed at the end of the current frame rather than returning
# immediately, so arming it early is honoured even when the frame's script
# aborts midway. scripts/verify-godot-headless.sh additionally wraps every
# Godot invocation in a watchdog, so a future regression here degrades to a
# reported timeout instead of an indefinite stall.
extends SceneTree

func _initialize() -> void:
	print("== unloaded-session negative path ==")
	if not ClassDB.class_exists("VokraSession"):
		print("MARKER: extension-not-loaded")
		quit(2)
		return
	var session: Object = ClassDB.instantiate("VokraSession")
	if session == null:
		print("MARKER: instantiate-failed")
		quit(2)
		return

	# Arm the exit before the call that is expected to raise (see TERMINATION).
	quit(0)

	# No load_model call: `inner` is None.
	var text: String = session.transcribe(PackedFloat32Array([0.0, 0.0]), 16000)

	# Unreachable when the contract holds. Re-arm with a failing code so a
	# successful transcribe is both marked in the output and reflected in the
	# exit status.
	print("MARKER: unexpected-success len=%d" % text.length())
	quit(1)
