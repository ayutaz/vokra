# Vokra TTS demo — main scene script (M3-11-T15, ADR-0011 §D5).
#
# Loads a piper-plus voice GGUF from res://models/, synthesizes user
# text via VokraSession.synthesize(...), and streams the resulting
# PCM buffer through an AudioStreamGenerator so the user hears it.
#
# Model fetch:
#   The AssetLib package installs the GDExtension addon under
#   res://addons/vokra/, but does NOT ship the piper-plus voices
#   (~50-80 MB each, out-of-band per ADR-0011 §D5). Consumers run
#     bash addons/vokra/fetch-demo-models.sh
#   from the project root before opening this demo — the fetch script
#   pulls the MIT-licensed en_US-amy voice from Hugging Face and drops
#   it at res://models/piper-en-amy.gguf. Voice ID "en_us_amy_medium"
#   matches the metadata inside the checkpoint.
#
# Backend selection:
#   VokraSession defaults to CPU. Passing "metal" / "cuda" to
#   `load_model` selects a GPU backend if the current build shipped
#   with `--features metal` / `--features cuda`. FR-EX-08: if the
#   requested backend is unavailable the load returns
#   VOKRA_ERROR_BACKEND_UNAVAILABLE — no silent CPU fallback.
#
# Runtime verification (open in Editor → Play → hear voice output):
# M3-11-T19 owner work.

extends Control

const MODEL_PATH: String = "res://models/piper-en-amy.gguf"
const VOICE_ID: String = "en_us_amy_medium"
# piper-plus native TTS emits 22 050 Hz mono PCM by default; the voice
# metadata inside the GGUF records the actual sample rate — the
# GDExtension surface returns it as the trailing element of the
# synthesize() result. This constant is only the AudioStreamGenerator
# fallback for the case where the addon isn't loaded.
const FALLBACK_SAMPLE_RATE: float = 22050.0

@onready var _status: Label = $VBoxContainer/StatusLabel
@onready var _input: LineEdit = $VBoxContainer/TextInput
@onready var _button: Button = $VBoxContainer/SpeakButton
@onready var _player: AudioStreamPlayer = $AudioStreamPlayer


func _on_speak_pressed() -> void:
	# Guard against double-invocation while a synthesis run is in flight.
	_button.disabled = true
	var text: String = _input.text.strip_edges()
	if text.is_empty():
		_status.text = "Please enter some text to synthesize."
		_button.disabled = false
		return

	_status.text = "Loading %s ..." % MODEL_PATH

	# GDExtension class is registered as `VokraSession` in
	# integrations/vokra-godot/src/registry.rs (M3-11-T05).
	# ClassDB.class_exists is Godot 4's guard for optional GDExtensions;
	# an absent class means the addon binary is missing from
	# addons/vokra/bin/ for the current platform.
	if not ClassDB.class_exists("VokraSession"):
		_status.text = "Error: VokraSession class not registered."
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
		_button.disabled = false
		return

	_status.text = "Synthesizing (%d chars)..." % text.length()
	# synthesize(text, voice_id) — returns a PackedFloat32Array of
	# mono PCM samples in [-1.0, 1.0]. Sample rate is voice-specific
	# and encoded in the GGUF metadata; for the piper-plus en_US-amy
	# medium voice it is 22 050 Hz.
	var pcm: PackedFloat32Array = session.synthesize(text, VOICE_ID)
	if pcm.size() == 0:
		_status.text = "Error: synthesize returned no samples."
		_button.disabled = false
		return

	_status.text = "Playing %d samples ..." % pcm.size()
	_play_pcm(pcm, FALLBACK_SAMPLE_RATE)
	_status.text = "Done. %d samples spoken." % pcm.size()
	_button.disabled = false


# _play_pcm — push a PackedFloat32Array of mono samples into the
# AudioStreamPlayer via an AudioStreamGenerator, so the user hears
# the synthesized voice.
#
# AudioStreamGenerator is the Godot 4 primitive for pushing PCM
# buffers from a script (see
# https://docs.godotengine.org/en/stable/classes/class_audiostreamgenerator.html).
# It streams at `mix_rate` — we set it from the voice's sample rate
# so no resampling is needed. The demo pushes the whole buffer at
# once for simplicity; a real integration would push in chunks as
# the flow-matching stream emits them (see the M3-14 barge-in path).
func _play_pcm(pcm: PackedFloat32Array, sample_rate: float) -> void:
	var gen: AudioStreamGenerator = AudioStreamGenerator.new()
	gen.mix_rate = sample_rate
	# Buffer length in seconds. Room for the whole utterance so we
	# don't underrun; longer utterances would need chunked push.
	var seconds: float = float(pcm.size()) / sample_rate + 0.5
	gen.buffer_length = maxf(seconds, 0.5)
	_player.stream = gen
	_player.play()
	# get_stream_playback returns AudioStreamGeneratorPlayback; the
	# push_buffer(pcm) call enqueues mono samples in [-1.0, 1.0].
	var playback: AudioStreamGeneratorPlayback = _player.get_stream_playback()
	if playback == null:
		return
	# push_frames wants stereo Vector2 pairs, push_buffer wants mono
	# Vector2 with .x = left, .y = right. Fill both channels from the
	# same mono sample to hear it on both speakers.
	var stereo: PackedVector2Array = PackedVector2Array()
	stereo.resize(pcm.size())
	for i in pcm.size():
		var s: float = pcm[i]
		stereo[i] = Vector2(s, s)
	playback.push_buffer(stereo)
