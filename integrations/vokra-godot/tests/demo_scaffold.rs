// demo_scaffold.rs — structural sanity checks for the ASR / TTS demo
// projects introduced in M3-11-T14 + T15, plus the addons/vokra fetch
// script that both demos reference on their first run.
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

fn addons_vokra_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("addons")
        .join("vokra")
}

fn repo_root() -> PathBuf {
    // integrations/vokra-godot -> integrations -> <repo root>
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..")
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

// ---- addons/vokra/fetch-demo-models.sh --------------------------------------
//
// The M3-11 Godot mirror of the Unity Samples fetch script
// (bindings/unity/com.vokra.unity/Samples~/VadAsrTts/Scripts/fetch-demo-models.sh).
// Both demo GDScripts document this exact path
// (`bash addons/vokra/fetch-demo-models.sh` in demos/*/main.gd), so a
// missing script or a broken red-line is a functional gap owner would hit
// on first play. These tests guard the invariant.
//
// Not a shell interpreter test: we assert the script's structural
// contract (shebang, downloader probe, filenames the demos reference,
// MIT-only red-line) via string containment. A shell-level e2e
// invocation would need real network access + the maintainer's release
// assets; that's the M3-11-T19 owner smoke path, not this file's job.

fn fetch_script_path() -> PathBuf {
    addons_vokra_root().join("fetch-demo-models.sh")
}

#[test]
fn fetch_demo_models_script_exists_and_is_bash() {
    let p = fetch_script_path();
    assert!(
        p.exists(),
        "addons/vokra/fetch-demo-models.sh missing — demos/*/main.gd:12 \
         reference this path but the file was never landed",
    );
    let src = read_to_string(&p);
    // A `#!/usr/bin/env bash` shebang so `./addons/vokra/fetch-demo-models.sh`
    // works on any POSIX host without pinning /bin/bash absolute path.
    assert!(
        src.starts_with("#!/usr/bin/env bash\n"),
        "fetch-demo-models.sh must start with `#!/usr/bin/env bash` shebang",
    );
    // `set -euo pipefail` — fail-loud invariant matching the Unity mirror
    // and every other Vokra shell script.
    assert!(
        src.contains("set -euo pipefail"),
        "fetch-demo-models.sh must set -euo pipefail",
    );
}

#[test]
fn fetch_demo_models_script_probes_curl_then_wget() {
    let src = read_to_string(&fetch_script_path());
    // curl-first / wget-fallback matches the Unity mirror + Vokra shell
    // script convention: curl is universally available on macOS/Linux CI
    // runners and wget covers minimal container images.
    assert!(
        src.contains("command -v curl"),
        "fetch-demo-models.sh must probe curl availability",
    );
    assert!(
        src.contains("command -v wget"),
        "fetch-demo-models.sh must probe wget as a fallback",
    );
    let curl_pos = src.find("command -v curl").unwrap();
    let wget_pos = src.find("command -v wget").unwrap();
    assert!(
        curl_pos < wget_pos,
        "fetch-demo-models.sh must probe curl BEFORE wget (Unity mirror parity)",
    );
    // FR-EX-08 spirit: no silent fallback — if neither is present, die
    // loudly with a stderr message.
    assert!(
        src.contains("neither curl nor wget"),
        "fetch-demo-models.sh must fail loudly when no downloader is present",
    );
}

#[test]
fn fetch_demo_models_script_env_overrides_are_present() {
    let src = read_to_string(&fetch_script_path());
    // Env overrides let the maintainer point at unreleased builds
    // (e.g., a private staging URL) without editing the script. Mirror
    // of the Unity script's VOKRA_MODELS_DIR / VOKRA_*_URL knobs.
    for env_var in ["VOKRA_MODELS_DIR", "VOKRA_WHISPER_URL", "VOKRA_PIPER_URL"] {
        assert!(
            src.contains(env_var),
            "fetch-demo-models.sh must honor ${env_var} env override",
        );
    }
}

#[test]
fn fetch_demo_models_script_targets_the_demo_filenames() {
    let src = read_to_string(&fetch_script_path());
    // Filename pins MUST match `demos/asr_demo/main.gd` MODEL_PATH and
    // `demos/tts_demo/main.gd` MODEL_PATH exactly — otherwise the demos
    // would open a stub .gguf that doesn't exist.
    assert!(
        src.contains("whisper-base.gguf"),
        "fetch-demo-models.sh must write whisper-base.gguf \
         (asr_demo/main.gd:29 res://models/whisper-base.gguf)",
    );
    assert!(
        src.contains("piper-en-amy.gguf"),
        "fetch-demo-models.sh must write piper-en-amy.gguf \
         (tts_demo/main.gd:29 res://models/piper-en-amy.gguf)",
    );
    // Destination MUST resolve to `<project_root>/models/` so the
    // demo's `res://models/…` const finds the weights. The script uses
    // `${SCRIPT_DIR}/../..` (two levels: addons/vokra → addons → project
    // root) + `models` — assert both anchors are present.
    assert!(
        src.contains("SCRIPT_DIR}/../.."),
        "fetch-demo-models.sh must compute PROJECT_ROOT via `${{SCRIPT_DIR}}/../..`",
    );
    assert!(
        src.contains("PROJECT_ROOT}/models"),
        "fetch-demo-models.sh must default DEST_DIR to `${{PROJECT_ROOT}}/models`",
    );
}

#[test]
fn fetch_demo_models_script_enforces_mit_only_red_line() {
    let src = read_to_string(&fetch_script_path());
    // BR-10 / M2-13 provenance contract: the AssetLib package channel is
    // MIT-only. If someone adds a CC-BY-NC model here it must fail this
    // test before it lands. We check the disclaimer copy AND the
    // specific model names that MUST NOT appear as fetch_one targets.
    assert!(
        src.contains("MIT-only"),
        "fetch-demo-models.sh must document its MIT-only policy",
    );
    assert!(
        src.contains("CC-BY-NC"),
        "fetch-demo-models.sh must call out the CC-BY-NC exclusion",
    );
    assert!(
        src.contains("M2-13"),
        "fetch-demo-models.sh must reference the M2-13 compliance gate",
    );
    // Explicit red-line filenames — a CC-BY-NC weight added as a
    // `fetch_one … "…/${dest}"` line would trip this check. We look
    // for these strings anywhere in the script so a stray "F5-TTS" in
    // an excluded-list comment is fine, but a fetch_one line targeting
    // one of these files is not.
    //
    // We can't easily distinguish "mentioned in a rejection comment"
    // vs "actually downloaded" via a substring test alone. So we assert
    // that any occurrence sits inside an EXCLUDED-list block: the token
    // "EXCLUDED" or "excluded" must appear within a small window of the
    // model name. This is looser than a full parser but tight enough to
    // catch the "someone pasted an F5 URL into fetch_one" mistake.
    for cc_by_nc_model in ["F5-TTS", "Fish-Speech", "EnCodec"] {
        if let Some(pos) = src.find(cc_by_nc_model) {
            // Look 200 chars in each direction for an "exclud" marker.
            // 200 chars is longer than a fetch_one line but shorter than
            // a policy paragraph, so a real fetch_one line targeting the
            // model would fail this window check.
            let start = pos.saturating_sub(200);
            let end = (pos + 200).min(src.len());
            let window = &src[start..end];
            let mentioned_in_exclusion = window.contains("EXCLUDED")
                || window.contains("excluded")
                || window.contains("EXCLUDE")
                || window.contains("exclude");
            assert!(
                mentioned_in_exclusion,
                "fetch-demo-models.sh mentions {cc_by_nc_model} outside an \
                 EXCLUDED context — MIT-only red-line violated",
            );
        }
    }
    // Voice-clone models are ALSO categorically banned from the demo
    // channel per BR-08 (Vokra core scope: no voice clone).
    for voice_clone_model in ["RVC", "GPT-SoVITS"] {
        if let Some(pos) = src.find(voice_clone_model) {
            let start = pos.saturating_sub(200);
            let end = (pos + 200).min(src.len());
            let window = &src[start..end];
            let mentioned_in_exclusion = window.contains("EXCLUDED")
                || window.contains("excluded")
                || window.contains("EXCLUDE")
                || window.contains("exclude");
            assert!(
                mentioned_in_exclusion,
                "fetch-demo-models.sh mentions {voice_clone_model} outside an \
                 EXCLUDED context — voice-clone core-exclusion violated",
            );
        }
    }
}

#[test]
fn fetch_demo_models_script_exposes_help_and_license_flags() {
    let src = read_to_string(&fetch_script_path());
    // The Unity mirror ships `--help` + `--license`; matching those flags
    // here keeps operator muscle memory identical across the two channels.
    assert!(
        src.contains("-h|--help)"),
        "fetch-demo-models.sh must handle `-h` / `--help`",
    );
    assert!(
        src.contains("--license)"),
        "fetch-demo-models.sh must handle `--license`",
    );
    // license_manifest() is the function both flags emit; a stub with
    // an empty body would break the CI-side license attribution check.
    assert!(
        src.contains("license_manifest()"),
        "fetch-demo-models.sh must define license_manifest()",
    );
}

#[test]
fn demos_reference_the_addons_fetch_script_path() {
    // Cross-check invariant: both demo GDScripts document the exact
    // invocation `bash addons/vokra/fetch-demo-models.sh`. If we ever
    // move or rename the fetch script, this test forces the demo docs
    // to move in lock-step. (M3-11-T14 + T15 → the addons/vokra path is
    // the AssetLib install target, ADR-0011 §D9.)
    let asr_gd = read_to_string(&demos_root().join("asr_demo").join("main.gd"));
    let tts_gd = read_to_string(&demos_root().join("tts_demo").join("main.gd"));
    for (name, src) in [("asr_demo/main.gd", &asr_gd), ("tts_demo/main.gd", &tts_gd)] {
        assert!(
            src.contains("addons/vokra/fetch-demo-models.sh"),
            "{name} must reference `addons/vokra/fetch-demo-models.sh` \
             (invocation path AssetLib consumers will follow)",
        );
    }
}

// ---- build-godot-gdextension.sh wire-up -------------------------------------
//
// The AssetLib package is assembled by `scripts/build-godot-gdextension.sh`.
// For consumers to run `bash addons/vokra/fetch-demo-models.sh` after
// unzipping vokra-godot-<version>.zip, the packaging step must include
// the fetch script alongside LICENSE / NOTICE / vokra.gdextension.
//
// This test guards against the wire-up regressing (someone dropping the
// `cp -f "$FETCH_SCRIPT_SRC"` line while refactoring the LICENSE loop).

#[test]
fn build_godot_gdextension_copies_fetch_demo_models() {
    let build_script = repo_root()
        .join("scripts")
        .join("build-godot-gdextension.sh");
    assert!(
        build_script.exists(),
        "scripts/build-godot-gdextension.sh missing — cannot verify \
         fetch-demo-models.sh wire-up",
    );
    let src = read_to_string(&build_script);
    // The build script must reference the fetch script's source path
    // AND copy it into the AssetLib addons/ tree.
    assert!(
        src.contains("fetch-demo-models.sh"),
        "build-godot-gdextension.sh must copy addons/vokra/fetch-demo-models.sh \
         into the AssetLib package",
    );
    // The `cp` invocation should preserve or explicitly set the exec bit
    // so consumers don't have to `chmod +x` after unzip. We look for
    // either `cp -p` (preserve) or `chmod +x` (explicit stamp).
    assert!(
        src.contains("cp -p -f \"$FETCH_SCRIPT_SRC\"")
            || src.contains("cp -pf \"$FETCH_SCRIPT_SRC\"")
            || src.contains("chmod +x \"$ADDONS_DIR/fetch-demo-models.sh\""),
        "build-godot-gdextension.sh must preserve OR explicitly set the exec bit \
         on the copied fetch-demo-models.sh (`cp -p` or `chmod +x`)",
    );
}
