// demo_scaffold.rs — structural sanity checks for the ASR / TTS demo
// projects introduced in M3-11-T14 + T15.
//
// Runtime verification (opening the demos in the Godot 4.x Editor + hitting
// Play) is M3-11-T19 owner work. These tests only guarantee that the
// scaffold files exist and carry the Godot 4.x format markers Godot
// needs to load a project — a missing `config_version=5`, a missing
// `.tscn` `format=3` header, or a stray `extends`-less GDScript would
// mean the demo can't even open, so we catch that here instead of
// waiting for the M3-11-T19 owner smoke to report it.
//
// Zero-dep unchanged (NFR-DS-02): these tests read files under
// `CARGO_MANIFEST_DIR` via `std::fs`; no new crate is added.

use std::fs;
use std::path::{Path, PathBuf};

fn demos_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("demos")
}

fn read_to_string(p: &Path) -> String {
    fs::read_to_string(p).unwrap_or_else(|e| panic!("failed to read {}: {e}", p.display()))
}

// ---- ASR demo ---------------------------------------------------------------

#[test]
fn asr_demo_project_godot_carries_config_version_and_main_scene() {
    let src = read_to_string(&demos_root().join("asr_demo").join("project.godot"));
    // Godot 4.0+ INI uses `config_version=5`; anything lower is 3.x and
    // will refuse to open in a 4.x Editor.
    assert!(
        src.contains("config_version=5"),
        "asr_demo/project.godot missing `config_version=5` header",
    );
    // `run/main_scene` must point at the .tscn we ship, otherwise Play
    // starts nothing.
    assert!(
        src.contains("run/main_scene=\"res://main.tscn\""),
        "asr_demo/project.godot missing `run/main_scene=res://main.tscn`",
    );
    // ADR-0011 §D4 pins compatibility_minimum = 4.1; the feature tag
    // list is what surfaces that requirement to the Editor.
    assert!(
        src.contains("\"4.1\""),
        "asr_demo/project.godot missing the `4.1` feature tag",
    );
}

#[test]
fn asr_demo_main_tscn_declares_gd_scene_and_control_root() {
    let src = read_to_string(&demos_root().join("asr_demo").join("main.tscn"));
    // Godot 4 .tscn format string.
    assert!(
        src.contains("[gd_scene"),
        "asr_demo/main.tscn missing `[gd_scene ...]` header",
    );
    assert!(
        src.contains("format=3"),
        "asr_demo/main.tscn missing `format=3` (Godot 4.x scene format)",
    );
    // The main node must be a Control descendant so the Label + Button
    // children render.
    assert!(
        src.contains("[node name=\"Main\" type=\"Control\"]"),
        "asr_demo/main.tscn missing `Main` Control root",
    );
    // Button.pressed → _on_transcribe_pressed wire-up is what triggers
    // the ASR call; a missing connection means the demo is silent.
    assert!(
        src.contains("method=\"_on_transcribe_pressed\""),
        "asr_demo/main.tscn missing the Button.pressed → _on_transcribe_pressed connection",
    );
}

#[test]
fn asr_demo_main_gd_extends_control_and_uses_vokrasession() {
    let src = read_to_string(&demos_root().join("asr_demo").join("main.gd"));
    // GDScript classes must declare a base with `extends`. A missing
    // `extends Control` means the script cannot attach to the .tscn root.
    assert!(
        src.contains("extends Control"),
        "asr_demo/main.gd missing `extends Control`",
    );
    // The demo's entry point is the ClassDB-instantiated VokraSession
    // — that string must be present because the ClassDB lookup uses the
    // exact class name registered by src/registry.rs.
    assert!(
        src.contains("VokraSession"),
        "asr_demo/main.gd never references VokraSession",
    );
    // Callback matches the .tscn connection.
    assert!(
        src.contains("func _on_transcribe_pressed"),
        "asr_demo/main.gd missing _on_transcribe_pressed callback",
    );
    // Sample rate must match Whisper base's fixed 16 kHz frontend.
    assert!(
        src.contains("SAMPLE_RATE") && src.contains("16000"),
        "asr_demo/main.gd missing SAMPLE_RATE = 16000",
    );
}

// ---- TTS demo ---------------------------------------------------------------

#[test]
fn tts_demo_project_godot_carries_config_version_and_main_scene() {
    let src = read_to_string(&demos_root().join("tts_demo").join("project.godot"));
    assert!(
        src.contains("config_version=5"),
        "tts_demo/project.godot missing `config_version=5` header",
    );
    assert!(
        src.contains("run/main_scene=\"res://main.tscn\""),
        "tts_demo/project.godot missing `run/main_scene=res://main.tscn`",
    );
    assert!(
        src.contains("\"4.1\""),
        "tts_demo/project.godot missing the `4.1` feature tag",
    );
}

#[test]
fn tts_demo_main_tscn_declares_scene_root_and_audio_player() {
    let src = read_to_string(&demos_root().join("tts_demo").join("main.tscn"));
    assert!(
        src.contains("[gd_scene"),
        "tts_demo/main.tscn missing `[gd_scene ...]` header",
    );
    assert!(
        src.contains("format=3"),
        "tts_demo/main.tscn missing `format=3` (Godot 4.x scene format)",
    );
    assert!(
        src.contains("[node name=\"Main\" type=\"Control\"]"),
        "tts_demo/main.tscn missing `Main` Control root",
    );
    // AudioStreamPlayer is how the TTS PCM reaches the speakers. Without
    // it the demo can synthesize but produces silence.
    assert!(
        src.contains("type=\"AudioStreamPlayer\""),
        "tts_demo/main.tscn missing AudioStreamPlayer node",
    );
    assert!(
        src.contains("method=\"_on_speak_pressed\""),
        "tts_demo/main.tscn missing the Button.pressed → _on_speak_pressed connection",
    );
}

#[test]
fn tts_demo_main_gd_extends_control_and_uses_vokrasession() {
    let src = read_to_string(&demos_root().join("tts_demo").join("main.gd"));
    assert!(
        src.contains("extends Control"),
        "tts_demo/main.gd missing `extends Control`",
    );
    assert!(
        src.contains("VokraSession"),
        "tts_demo/main.gd never references VokraSession",
    );
    assert!(
        src.contains("func _on_speak_pressed"),
        "tts_demo/main.gd missing _on_speak_pressed callback",
    );
    // synthesize(...) is the piper-plus TTS entry point registered by
    // src/registry.rs. A missing reference means the demo would compile
    // but never call the TTS engine.
    assert!(
        src.contains(".synthesize("),
        "tts_demo/main.gd never calls .synthesize(...)",
    );
    // AudioStreamGenerator is the primitive that lets GDScript push raw
    // PCM buffers — required for the play path.
    assert!(
        src.contains("AudioStreamGenerator"),
        "tts_demo/main.gd never uses AudioStreamGenerator",
    );
}
