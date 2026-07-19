//! M4-07-T13: `tools/parity/cuda_rtf_variance.sh --fa-mode` e2e fixture tests
//! (the "local fake vokra-cli stub" discipline of M2 item 2) + analyzer
//! backward compatibility against the recorded 2026-07-10 JSONL.
//!
//! The shell harness cannot be compiler-checked against the Rust env-toggle
//! surface, so these tests drive the *real* script with a fake `vokra-cli`
//! that reports which FA env vars it saw, then assert (a) the injected env
//! per mode, (b) the JSONL `fa_mode` + legacy `fa_v2_mode` fields, (c) the
//! `--fa-v2` alias mapping and the both-flags conflict error, and (d) that
//! `cuda_rtf_analyze.py` (stdlib-only) still parses both the NEW JSONL and
//! the OLD committed baselines (`docs/bench-baselines/vast-2026-07-10/`).
//!
//! Host requirements: `bash` + `python3` (the harness's own requirements —
//! present on the dev Macs and CI runners). Missing either → clean skip.
//! No CUDA / GPU involvement anywhere: the fake CLI never touches a device.

#![cfg(unix)] // the harness is a bash script; Windows runs it under WSL/CI Linux

use std::path::{Path, PathBuf};
use std::process::Command;

fn repo_root() -> PathBuf {
    // crates/vokra-backend-cuda -> repo root.
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("crate lives two levels under the repo root")
        .to_path_buf()
}

fn have(cmd: &str) -> bool {
    Command::new(cmd)
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// A scratch dir under the target dir (std-only; no tempfile crate).
fn scratch_dir(tag: &str) -> PathBuf {
    let d = repo_root()
        .join("target")
        .join("fa-v3-harness-tests")
        .join(format!("{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&d).expect("create scratch dir");
    d
}

/// Writes the fake `vokra-cli`: emits one bench-report JSON line whose
/// fields echo the FA env vars it observed (so the JSONL `bench` envelope
/// carries the ground truth of what the harness injected).
fn write_fake_cli(dir: &Path) -> PathBuf {
    let path = dir.join("fake-vokra-cli");
    let script = r#"#!/usr/bin/env bash
# Fake vokra-cli for the M4-07-T13 harness fixture tests: no GPU, no model.
# Prints a canned bench report whose fields echo the observed FA env vars.
d2="${VOKRA_CUDA_DISABLE_FA_V2:-unset}"
d3="${VOKRA_CUDA_DISABLE_FA_V3:-unset}"
enc="${VOKRA_CUDA_FA_V3_ENCODER:-unset}"
printf '{"task":"asr","rtf":0.5,"latency_ms":{"mean":100.0},"env_disable_fa_v2":"%s","env_disable_fa_v3":"%s","env_fa_v3_encoder":"%s"}\n' "$d2" "$d3" "$enc"
"#;
    std::fs::write(&path, script).expect("write fake cli");
    // SAFETY-free chmod via std: mark executable.
    use std::os::unix::fs::PermissionsExt;
    let mut perm = std::fs::metadata(&path).expect("stat").permissions();
    perm.set_mode(0o755);
    std::fs::set_permissions(&path, perm).expect("chmod fake cli");
    path
}

/// Runs the harness for `args` and returns (exit ok, stdout+stderr, JSONL).
fn run_harness(tag: &str, extra_args: &[&str]) -> (bool, String, String) {
    let root = repo_root();
    let dir = scratch_dir(tag);
    let cli = write_fake_cli(&dir);
    let gguf = dir.join("fake.gguf");
    let audio = dir.join("fake.wav");
    std::fs::write(&gguf, b"gguf-placeholder").unwrap();
    std::fs::write(&audio, b"wav-placeholder").unwrap();
    let out = dir.join("out.jsonl");

    let mut cmd = Command::new("bash");
    cmd.arg(root.join("tools/parity/cuda_rtf_variance.sh"))
        .arg("--gguf")
        .arg(&gguf)
        .arg("--audio")
        .arg(&audio)
        .arg("--iters")
        .arg("2")
        .arg("--warmup")
        .arg("0")
        .arg("--vokra-cli")
        .arg(&cli)
        .arg("--output")
        .arg(&out)
        .args(extra_args);
    let o = cmd.output().expect("run harness");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&o.stdout),
        String::from_utf8_lossy(&o.stderr)
    );
    let jsonl = std::fs::read_to_string(&out).unwrap_or_default();
    (o.status.success(), text, jsonl)
}

/// The three modes inject exactly the documented env-var sets and stamp the
/// matching `fa_mode` + legacy `fa_v2_mode` fields on every JSONL line.
#[test]
fn fa_mode_three_values_inject_documented_env_sets() {
    if !have("bash") || !have("python3") {
        eprintln!("skipping: bash/python3 not available on this host");
        return;
    }
    // (mode, expect disable_v2, disable_v3, encoder, legacy fa_v2_mode)
    let cases = [
        ("decomposed", "1", "1", "unset", "off"),
        ("v2", "unset", "1", "unset", "on"),
        ("v3", "unset", "unset", "1", "on"),
    ];
    for (mode, d2, d3, enc, legacy) in cases {
        let (ok, log, jsonl) = run_harness(&format!("mode-{mode}"), &["--fa-mode", mode]);
        assert!(ok, "harness must exit 0 for --fa-mode {mode}: {log}");
        let mut ok_lines = 0;
        for line in jsonl.lines() {
            assert!(
                line.contains(&format!("\"fa_mode\":\"{mode}\"")),
                "every line stamps fa_mode={mode}: {line}"
            );
            assert!(
                line.contains(&format!("\"fa_v2_mode\":\"{legacy}\"")),
                "every line keeps the legacy field ({legacy}): {line}"
            );
            if line.contains("\"status\":\"ok\"") {
                ok_lines += 1;
                assert!(
                    line.contains(&format!("\"env_disable_fa_v2\":\"{d2}\"")),
                    "{mode}: DISABLE_FA_V2 must be {d2}: {line}"
                );
                assert!(
                    line.contains(&format!("\"env_disable_fa_v3\":\"{d3}\"")),
                    "{mode}: DISABLE_FA_V3 must be {d3}: {line}"
                );
                assert!(
                    line.contains(&format!("\"env_fa_v3_encoder\":\"{enc}\"")),
                    "{mode}: FA_V3_ENCODER must be {enc}: {line}"
                );
            }
        }
        assert_eq!(
            ok_lines, 2,
            "{mode}: both iterations must succeed:\n{jsonl}"
        );
    }
}

/// The legacy `--fa-v2` alias maps on→v2 / off→decomposed, the default (no
/// flag) is the historical `v2` leg, and passing both flags is an explicit
/// usage error (no silent precedence).
#[test]
fn fa_v2_alias_and_conflict_semantics() {
    if !have("bash") || !have("python3") {
        eprintln!("skipping: bash/python3 not available on this host");
        return;
    }
    let (ok, _log, jsonl) = run_harness("alias-on", &["--fa-v2", "on"]);
    assert!(ok);
    assert!(
        jsonl.contains("\"fa_mode\":\"v2\""),
        "alias on -> v2:\n{jsonl}"
    );
    assert!(jsonl.contains("\"fa_v2_mode\":\"on\""));

    let (ok, _log, jsonl) = run_harness("alias-off", &["--fa-v2", "off"]);
    assert!(ok);
    assert!(
        jsonl.contains("\"fa_mode\":\"decomposed\""),
        "alias off -> decomposed:\n{jsonl}"
    );
    assert!(jsonl.contains("\"fa_v2_mode\":\"off\""));

    let (ok, _log, jsonl) = run_harness("default", &[]);
    assert!(ok);
    assert!(
        jsonl.contains("\"fa_mode\":\"v2\""),
        "default = historical --fa-v2 on leg:\n{jsonl}"
    );

    let (ok, log, _jsonl) = run_harness("conflict", &["--fa-mode", "v3", "--fa-v2", "on"]);
    assert!(!ok, "both flags at once must be a usage error");
    assert!(
        log.contains("not both"),
        "the error must name the conflict: {log}"
    );

    let (ok, log, _jsonl) = run_harness("bad-mode", &["--fa-mode", "v4"]);
    assert!(!ok, "unknown mode must be a usage error");
    assert!(log.contains("--fa-mode"), "must name the flag: {log}");
}

/// `cuda_rtf_analyze.py` parses BOTH the new 3-mode JSONL and the committed
/// pre-M4-07 baselines (legacy `fa_v2_mode`-only lines mapped on→v2 /
/// off→decomposed) — the recorded vast.ai 2026-07-10 collection must stay
/// analyzable byte-for-byte (M4-07-T13 completion condition).
#[test]
fn analyzer_handles_new_and_legacy_jsonl() {
    if !have("bash") || !have("python3") {
        eprintln!("skipping: bash/python3 not available on this host");
        return;
    }
    let root = repo_root();
    let analyzer = root.join("tools/parity/cuda_rtf_analyze.py");

    // New-format JSONL from a real harness run against the fake CLI.
    let (ok, _log, jsonl) = run_harness("analyze-new", &["--fa-mode", "v3"]);
    assert!(ok);
    let dir = scratch_dir("analyze-new-out");
    let new_jsonl = dir.join("new.jsonl");
    std::fs::write(&new_jsonl, &jsonl).unwrap();
    let o = Command::new("python3")
        .arg(&analyzer)
        .arg(&new_jsonl)
        .output()
        .expect("run analyzer on new JSONL");
    let report = String::from_utf8_lossy(&o.stdout).into_owned();
    assert!(o.status.success(), "analyzer must accept the new JSONL");
    assert!(
        report.contains("| fa_mode | `v3` |"),
        "report must surface fa_mode=v3:\n{report}"
    );

    // Legacy committed baseline (pre-M4-07: fa_v2_mode only).
    let legacy = root.join("docs/bench-baselines/vast-2026-07-10/rtf-decomposed.jsonl");
    assert!(legacy.exists(), "recorded baseline must remain in-tree");
    let o = Command::new("python3")
        .arg(&analyzer)
        .arg(&legacy)
        .output()
        .expect("run analyzer on legacy JSONL");
    let report = String::from_utf8_lossy(&o.stdout).into_owned();
    assert!(
        o.status.success(),
        "analyzer must keep accepting legacy JSONL"
    );
    assert!(
        report.contains("| fa_mode | `decomposed` |"),
        "legacy off must map to decomposed:\n{report}"
    );
    assert!(
        report.contains("mean"),
        "stats must still be computed for legacy data"
    );
}
