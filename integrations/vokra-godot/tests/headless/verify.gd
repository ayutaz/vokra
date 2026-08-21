# M3-11 T19 — headless verification driver for the Vokra GDExtension.
#
# Driven by scripts/verify-godot-headless.sh; not run by `cargo test` (this is
# a GDScript file executed by a real Godot binary, which is what makes it
# evidence: it exercises Godot's own ClassDB, Variant marshalling and method
# dispatch rather than a mock of them).
#
# Two modes:
#   * asset-free   — no arguments. Runs the checks that need no model:
#                    ClassDB registration, method binding, instantiation, and
#                    the load_model failure path. Suitable for CI.
#   * ASR chain    — `-- <model.gguf> <audio.wav>`. Adds a real GGUF load and
#                    a real transcription through the registered trampolines.
#   * VAD chain    — `-- --vad <model.gguf> <audio.f32>`. Opens the returned
#                    VokraStream Object, pushes real PCM, polls probabilities,
#                    interrupts/reset it, and repeats the stream deterministically.
#
# Exits 0 only when every check passes.
extends SceneTree

var _failures: int = 0

func _check(ok: bool, label: String, detail: String = "") -> void:
	if ok:
		print("  PASS  ", label, ("" if detail == "" else "  [" + detail + "]"))
	else:
		_failures += 1
		print("  FAIL  ", label, ("" if detail == "" else "  [" + detail + "]"))


# Walk RIFF chunks rather than assuming a 44-byte header: real corpora carry
# a LIST/INFO chunk between `fmt ` and `data` (the JFK fixture does).
func _load_wav_16k_mono(path: String) -> PackedFloat32Array:
	var f := FileAccess.open(path, FileAccess.READ)
	if f == null:
		push_error("cannot open " + path)
		return PackedFloat32Array()
	var b := f.get_buffer(f.get_length())
	f.close()
	if b.size() < 12 or b.slice(0, 4) != "RIFF".to_ascii_buffer() \
			or b.slice(8, 12) != "WAVE".to_ascii_buffer():
		push_error("not a RIFF/WAVE file: " + path)
		return PackedFloat32Array()

	var pos := 12
	var channels := 0
	var rate := 0
	var bits := 0
	var out := PackedFloat32Array()
	while pos + 8 <= b.size():
		var cid := b.slice(pos, pos + 4).get_string_from_ascii()
		var csz := b.decode_u32(pos + 4)
		var body := pos + 8
		if cid == "fmt ":
			channels = b.decode_u16(body + 2)
			rate = b.decode_u32(body + 4)
			bits = b.decode_u16(body + 14)
		elif cid == "data":
			var n := csz / 2
			out.resize(n)
			for i in range(n):
				out[i] = float(b.decode_s16(body + i * 2)) / 32768.0
		# Chunks are word-aligned: an odd size carries a pad byte.
		pos = body + csz + (csz & 1)

	if channels != 1 or rate != 16000 or bits != 16:
		push_error("expected 16 kHz mono s16; got %d ch / %d Hz / %d bit" % [channels, rate, bits])
		return PackedFloat32Array()
	return out


# The C-ABI VAD fixture is raw little-endian f32. Decode it explicitly rather
# than relying on host-endian PackedByteArray casts, so the headless contract
# is identical on every supported desktop runner.
func _load_raw_f32(path: String) -> PackedFloat32Array:
	var f := FileAccess.open(path, FileAccess.READ)
	if f == null:
		push_error("cannot open " + path)
		return PackedFloat32Array()
	var b := f.get_buffer(f.get_length())
	f.close()
	if b.size() == 0 or b.size() % 4 != 0:
		push_error("expected non-empty little-endian f32 PCM: " + path)
		return PackedFloat32Array()
	var out := PackedFloat32Array()
	out.resize(b.size() / 4)
	for i in range(out.size()):
		out[i] = b.decode_float(i * 4)
	return out


func _run_vad_stream(stream: Object, pcm: PackedFloat32Array) -> PackedFloat32Array:
	var out := PackedFloat32Array()
	var offset := 0
	while offset < pcm.size():
		var end: int = min(offset + 2048, pcm.size())
		stream.push_pcm(pcm.slice(offset, end))
		out.append_array(stream.poll(64))
		offset = end
	# A final non-blocking drain proves poll remains callable when the ring is
	# already empty; complete frames were drained after every push above.
	out.append_array(stream.poll(64))
	return out


func _initialize() -> void:
	var argv := OS.get_cmdline_user_args()
	var vad_mode: bool = argv.size() > 0 and argv[0] == "--vad"
	var arg_offset: int = 1 if vad_mode else 0
	var model_path: String = argv[arg_offset] if argv.size() > arg_offset else ""
	var audio_path: String = argv[arg_offset + 1] if argv.size() > arg_offset + 1 else ""
	var full_chain: bool = model_path != "" and audio_path != ""

	print("== Vokra GDExtension headless verification (M3-11 T19) ==")
	print("Godot   : ", Engine.get_version_info().string)
	print("mode    : ", ("VAD stream" if vad_mode else "ASR full chain") if full_chain else "asset-free")
	if full_chain:
		print("model   : ", model_path)
		print("audio   : ", audio_path)
	print("")

	# --- 1. ClassDB registration (registry.rs `register`) ------------------
	print("[1] ClassDB registration")
	_check(ClassDB.class_exists("VokraSession"), "VokraSession registered")
	_check(ClassDB.class_exists("VokraStream"), "VokraStream registered")
	if not ClassDB.class_exists("VokraSession"):
		print("\nRESULT: FAIL (extension did not load)")
		quit(1)
		return

	# --- 2. Method bindings ------------------------------------------------
	# Each entry is [name, arg count, return Variant type]. The return types
	# are load-bearing: the demos bind `load_model`'s result to a statically
	# typed `int` and `transcribe`'s to a `String`.
	print("[2] Method bindings on VokraSession")
	var expected := {
		"load_model": [1, TYPE_INT],
		"transcribe": [2, TYPE_STRING],
		"synthesize": [1, TYPE_DICTIONARY],
		"vad_open_stream": [1, TYPE_OBJECT],
	}
	var seen := {}
	for m in ClassDB.class_get_method_list("VokraSession", true):
		seen[m["name"]] = m
	for want in expected:
		if not seen.has(want):
			_check(false, "method " + want, "MISSING")
			continue
		var m: Dictionary = seen[want]
		var argc: int = (m["args"] as Array).size()
		var ret: int = int(m.get("return", {}).get("type", TYPE_NIL))
		var spec: Array = expected[want]
		_check(argc == spec[0] and ret == spec[1], "method " + want,
			"args=%d ret=%d" % [argc, ret])

	# --- 3. Instantiate ----------------------------------------------------
	# Regression anchor: `create_instance_func` must return a Godot Object
	# built by `classdb_construct_object` with the Rust instance attached via
	# `object_set_instance`. Returning the bare Box pointer segfaults Godot
	# here (it dynamic_casts the result as `Object *`).
	print("[3] Instantiation")
	var session: Object = ClassDB.instantiate("VokraSession")
	_check(session != null, "VokraSession.new()")
	if session == null:
		print("\nRESULT: FAIL")
		quit(1)
		return
	_check(session.has_method("load_model"), "instance exposes load_model")

	# --- 4. load_model failure is an in-band non-zero status ---------------
	# FR-EX-08: an unreadable model reports a VokraStatus code, never a
	# silent success. (Argument-shape errors take the CallError channel
	# instead — see trampoline::session_load_model.)
	print("[4] load_model error path")
	var bad_status: int = session.load_model("/nonexistent/vokra-headless-verify.gguf")
	_check(bad_status != 0, "missing file yields non-zero status",
		"status=%d" % bad_status)

	if not full_chain:
		session.free()
		print("")
		if _failures == 0:
			print("RESULT: PASS (asset-free checks green; pass <model.gguf> <audio.wav> for the full chain)")
			quit(0)
		else:
			print("RESULT: FAIL (%d check(s) failed)" % _failures)
			quit(1)
		return

	# --- 5. load_model success path ---------------------------------------
	print("[5] load_model success path")
	var t0: int = Time.get_ticks_msec()
	var status: int = session.load_model(model_path)
	var load_ms: int = Time.get_ticks_msec() - t0
	_check(status == 0, "real GGUF loads", "status=%d, %d ms" % [status, load_ms])
	if status != 0:
		print("\nRESULT: FAIL (model did not load)")
		session.free()
		quit(1)
		return

	# --- 6. task path through the registered trampolines -------------------
	print("[6] ", "VAD stream" if vad_mode else "transcribe")
	var pcm: PackedFloat32Array = _load_raw_f32(audio_path) if vad_mode else _load_wav_16k_mono(audio_path)
	_check(pcm.size() > 0, "PCM decoded", "%d samples" % pcm.size())
	if pcm.size() == 0:
		print("\nRESULT: FAIL")
		session.free()
		quit(1)
		return

	if vad_mode:
		var stream: Object = session.vad_open_stream(16000)
		_check(stream != null, "vad_open_stream returns an Object")
		if stream == null:
			print("\nRESULT: FAIL")
			session.free()
			quit(1)
			return
		_check(stream.is_class("VokraStream"), "returned Object is VokraStream", stream.get_class())
		_check(stream.has_method("push_pcm") and stream.has_method("poll") and stream.has_method("interrupt"),
			"stream methods are callable")

		t0 = Time.get_ticks_msec()
		var probs: PackedFloat32Array = _run_vad_stream(stream, pcm)
		var vad_ms: int = Time.get_ticks_msec() - t0
		_check(probs.size() == pcm.size() / 512, "one probability per complete frame",
			"%d probs, %d ms" % [probs.size(), vad_ms])
		var in_range: bool = probs.size() > 0
		for prob in probs:
			if not (prob >= 0.0 and prob <= 1.0):
				in_range = false
				break
		_check(in_range, "all probabilities are finite and within [0,1]")

		stream.interrupt()
		_check(stream.poll(64).is_empty(), "interrupt drains pending output")
		var probs_after_reset: PackedFloat32Array = _run_vad_stream(stream, pcm)
		_check(probs_after_reset == probs, "interrupt resets hidden/context state deterministically")
		stream.free()
		session.free()
		print("")
		if _failures == 0:
			print("RESULT: PASS (real VAD stream checks green)")
			quit(0)
		else:
			print("RESULT: FAIL (%d check(s) failed)" % _failures)
			quit(1)
		return

	t0 = Time.get_ticks_msec()
	var text: String = session.transcribe(pcm, 16000)
	var asr_ms := Time.get_ticks_msec() - t0
	_check(text.length() > 0, "transcript is non-empty",
		"%d chars, %d ms" % [text.length(), asr_ms])
	print("  transcript: ", text)

	# --- 7. determinism ----------------------------------------------------
	print("[7] determinism")
	var text2: String = session.transcribe(pcm, 16000)
	_check(text2 == text, "second transcribe returns an identical transcript")

	# --- 8. reload + instance teardown -------------------------------------
	print("[8] reload / teardown")
	var reload: int = session.load_model(model_path)
	_check(reload == 0, "model reloads over a live session", "status=%d" % reload)
	var text3: String = session.transcribe(pcm, 16000)
	_check(text3 == text, "transcript survives a reload")
	# Exercises free_instance_func: Godot frees the Object, which must run the
	# paired `Box::from_raw` reclaim. There is nothing to assert afterwards —
	# a double-free or a bad reclaim aborts the process here, so *reaching*
	# the next line is the whole result. Recorded as a survival check rather
	# than dressed up as an assertion.
	session.free()
	print("  SURVIVED  VokraSession.free() returned without aborting")

	print("")
	if _failures == 0:
		print("RESULT: PASS (all checks green)")
		quit(0)
	else:
		print("RESULT: FAIL (%d check(s) failed)" % _failures)
		quit(1)
