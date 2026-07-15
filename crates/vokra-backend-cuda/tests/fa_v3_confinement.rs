//! M4-07-T15: fixture tests for `scripts/check-fa-v3-confinement.sh` — the
//! machine form of the "FA v3 only inside `crates/vokra-backend-cuda`"
//! red-line (supersedes the M3-01 ADR §2-(b) all-tree grep gate).
//!
//! Three verdicts are pinned:
//!   1. the REAL tree passes (the gate is live and the current landing is
//!      confined);
//!   2. a scratch tree with a deliberate non-comment leak in another crate
//!      FAILS (the gate actually bites);
//!   3. a scratch tree where the symbols appear only on comment lines —
//!      the vokra-backend-vulkan "no Hopper WGMMA/TMA equivalent" doc
//!      pattern — passes (zero false positives on the known mention).
//!
//! Host requirement: `bash` (the script's own requirement); missing → skip.

#![cfg(unix)]

use std::path::{Path, PathBuf};
use std::process::Command;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("crate lives two levels under the repo root")
        .to_path_buf()
}

fn have_bash() -> bool {
    Command::new("bash")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn scratch_tree(tag: &str) -> PathBuf {
    let d = repo_root()
        .join("target")
        .join("fa-v3-confinement-tests")
        .join(format!("{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).expect("create scratch tree");
    d
}

fn run_gate(crates_dir: &Path) -> (bool, String) {
    let script = repo_root().join("scripts/check-fa-v3-confinement.sh");
    let o = Command::new("bash")
        .arg(&script)
        .env("CRATES_DIR", crates_dir)
        .output()
        .expect("run confinement gate");
    (
        o.status.success(),
        format!(
            "{}{}",
            String::from_utf8_lossy(&o.stdout),
            String::from_utf8_lossy(&o.stderr)
        ),
    )
}

/// (1) The real tree is confined — the gate is green right now, i.e. the
/// M4-07 landing keeps every FA v3 symbol inside vokra-backend-cuda.
#[test]
fn real_tree_is_confined() {
    if !have_bash() {
        eprintln!("skipping: bash not available");
        return;
    }
    let (ok, log) = run_gate(&repo_root().join("crates"));
    assert!(ok, "the real tree must pass the confinement gate:\n{log}");
    assert!(log.contains("OK: FA v3 symbols confined"));
}

/// (2) A deliberate leak (non-comment `wgmma` in a sibling crate) trips the
/// gate with the offending line printed.
#[test]
fn deliberate_leak_trips_the_gate() {
    if !have_bash() {
        eprintln!("skipping: bash not available");
        return;
    }
    let tree = scratch_tree("leak");
    let other = tree.join("vokra-backend-metal/src");
    std::fs::create_dir_all(&other).unwrap();
    std::fs::write(
        other.join("lib.rs"),
        "pub fn oops() {\n    let _ = \"wgmma.mma_async\"; // leaked mnemonic in code\n}\n",
    )
    .unwrap();
    // The allowed subtree stays legal even in the scratch world.
    let cuda = tree.join("vokra-backend-cuda/src");
    std::fs::create_dir_all(&cuda).unwrap();
    std::fs::write(cuda.join("lib.rs"), "pub const K: &str = \"wgmma\";\n").unwrap();

    let (ok, log) = run_gate(&tree);
    assert!(
        !ok,
        "a non-comment leak outside vokra-backend-cuda must fail"
    );
    assert!(
        log.contains("vokra-backend-metal") && log.contains("wgmma"),
        "the offending file + symbol must be printed:\n{log}"
    );
    assert!(
        !log.contains("vokra-backend-cuda/src/lib.rs"),
        "the allowed subtree must not be reported:\n{log}"
    );
}

/// (3) Comment-only mentions — the documented-prohibition pattern — do not
/// trip the gate: `//!`, `///`, `//`, and block-comment continuation lines
/// are all excluded (matching the vokra-backend-vulkan doc mention and the
/// vokra-backend-cuda historical rustdoc notes in other crates).
#[test]
fn comment_only_mentions_do_not_trip() {
    if !have_bash() {
        eprintln!("skipping: bash not available");
        return;
    }
    let tree = scratch_tree("comments");
    let other = tree.join("vokra-backend-vulkan/src");
    std::fs::create_dir_all(&other).unwrap();
    std::fs::write(
        other.join("lib.rs"),
        concat!(
            "//! has no Hopper WGMMA/TMA equivalent, but the attention path\n",
            "/// FA v3 (wgmma / compute_90a / TMA_) is forbidden here.\n",
            "// flash_attn_v3 must not be implemented in this crate.\n",
            "/* block comment naming WGMMA too\n",
            " * and a continuation line with wgmma */\n",
            "pub fn clean() {}\n",
        ),
    )
    .unwrap();

    let (ok, log) = run_gate(&tree);
    assert!(
        ok,
        "comment-only mentions must not trip the confinement gate:\n{log}"
    );
}
