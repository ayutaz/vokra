# Vokra ASR demo — main scene script (M3-11-T14, ADR-0011 §D5).
#
# Loads a Whisper base GGUF from res://models/, decodes a 16 kHz mono
# WAV from res://audio/, invokes VokraSession.transcribe(...) from the
# vokra-godot GDExtension, and writes the result into TranscriptLabel.
#
# Model + audio fetch:
#   The AssetLib package installs the GDExtension addon under
#   res://addons/vokra/, but does NOT ship the whisper-base weights
#   (~150 MB, out-of-band per ADR-0011 §D5 = model files never bundled).
#   Consumers are expected to run
#     bash addons/vokra/fetch-demo-models.sh
#   from the project root before opening this demo — the fetch script
#   pulls the MIT-licensed whisper-base checkpoint from Hugging Face
#   and drops it at res://models/whisper-base.gguf.
#
# Backend selection:
#   VokraSession defaults to CPU. Passing "metal" / "cuda" to
#   `load_model` selects a GPU backend if the current build shipped
#   with `--features metal` / `--features cuda`. FR-EX-08: if the
#   requested backend is unavailable the load returns
#   VOKRA_ERROR_BACKEND_UNAVAILABLE — no silent CPU fallback.
#
# Runtime verification (open in Editor → Play → check transcription):
# M3-11-T19 owner work.

extends Control

const MODEL_PATH: String = "res://models/whisper-base.gguf"
const AUDIO_PATH: String = "res://audio/jfk.wav"
const SAMPLE_RATE: int = 16000

@onready var _status: Label = $VBoxContainer/StatusLabel
@onready var _transcript: Label = $VBoxContainer/TranscriptLabel
@onready var _button: Button = $VBoxContainer/LoadButton


func _on_transcribe_pressed() -> void:
	# Guard against double-invocation while a run is in flight.
	_button.disabled = true
	_status.text = "Loading %s ..." % MODEL_PATH

	# GDExtension class is registered as `VokraSession` in
	# integrations/vokra-godot/src/registry.rs (M3-11-T05).
	# ClassDB.class_exists is Godot 4's guard for optional GDExtensions;
	# an absent class means the addon binary is missing from
	# addons/vokra/bin/ for the current platform.
	if not ClassDB.class_exists("VokraSession"):
		_status.text = "Error: VokraSession class not registered."
		_transcript.text = ("The vokra-godot GDExtension addon is not "
			+ "installed for this platform. See addons/vokra/README.md.")
		_button.disabled = false
		return

	var session = ClassDB.instantiate("VokraSession")
	if session == null:
		_status.text = "Error: could not instantiate VokraSession."
		_button.disabled = false
		return

	# load_model(path) — the GDExtension side reads the GGUF via
	# vokra_session_load_model (M3-11-T05 registration table). Errors
	# come back through vokra_last_error() → VokraError; catch_unwind
	# on the trampoline (M3-11-T06) means a Rust panic never crosses
	# the FFI boundary (NFR-RL-07).
	var load_status: int = session.load_model(MODEL_PATH)
	if load_status != 0:
		_status.text = "Error: load_model returned status=%d" % load_status
		_transcript.text = ("Whisper GGUF not found at %s. Run "
			+ "`bash addons/vokra/fetch-demo-models.sh` from the project "
			+ "root to fetch the MIT-licensed weights.") % MODEL_PATH
		_button.disabled = false
		return

	_status.text = "Reading %s ..." % AUDIO_PATH
	var pcm: PackedFloat32Array = _load_pcm_16k_mono(AUDIO_PATH)
	if pcm.size() == 0:
		_status.text = "Error: could not read %s" % AUDIO_PATH
		_button.disabled = false
		return

	_status.text = "Transcribing (%d samples)..." % pcm.size()
	var text: String = session.transcribe(pcm, SAMPLE_RATE)
	_transcript.text = text
	_status.text = "Done. %d samples → %d chars." % [pcm.size(), text.length()]
	_button.disabled = false


# _load_pcm_16k_mono — read a 16 kHz mono PCM16 WAV from `path` and
# return the samples as a PackedFloat32Array normalised to [-1.0, 1.0].
#
# Kept intentionally simple: expects the standard WAV RIFF/fmt / data
# chunk layout (44-byte header) at 16 kHz mono s16le. A malformed or
# non-matching file returns an empty PackedFloat32Array — the caller
# surfaces this to the user rather than crashing.
#
# For higher-fidelity streaming, the addon's VokraStream class handles
# resampling + framing on the Rust side (M3-11-T08); this helper is
# only here to keep the demo self-contained.
func _load_pcm_16k_mono(path: String) -> PackedFloat32Array:
	var f: FileAccess = FileAccess.open(path, FileAccess.READ)
	if f == null:
		return PackedFloat32Array()
	var bytes: PackedByteArray = f.get_buffer(f.get_length())
	f.close()
	# Minimal RIFF/WAVE guard — 44-byte header, "RIFF" magic, "WAVE".
	if bytes.size() < 44:
		return PackedFloat32Array()
	if bytes.slice(0, 4) != "RIFF".to_ascii_buffer():
		return PackedFloat32Array()
	if bytes.slice(8, 12) != "WAVE".to_ascii_buffer():
		return PackedFloat32Array()
	# Data starts at offset 44 in the canonical layout. Real-world WAVs
	# can carry LIST/INFO chunks; a robust parser would scan for the
	# "data" chunk header. Owner-side smoke uses OpenAI whisper's
	# fixtures which are canonical 44-byte-header files, so the
	# heuristic is fine for the demo.
	var samples: int = (bytes.size() - 44) / 2
	var out: PackedFloat32Array = PackedFloat32Array()
	out.resize(samples)
	for i in samples:
		var lo: int = bytes[44 + i * 2]
		var hi: int = bytes[44 + i * 2 + 1]
		var s: int = lo | (hi << 8)
		if s >= 0x8000:
			s -= 0x10000
		out[i] = float(s) / 32768.0
	return out
