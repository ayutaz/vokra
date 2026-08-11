//! OOM-STOPGAP-CLEANUP Phase 2 regression pin (2026-08-09).
//!
//! `SbV2Model::synthesize` used to carry a
//! `PER_PHONEME_DURATION_CEILING = 500` per-phoneme duration cap + a
//! matching `[sbv2-synth-warn] SbV2SDP produced runaway durations` stderr
//! message, both landed 2026-08-08 as a temporary OOM safety fuse while
//! Wave-2 SBV2-BUG4 (upstream `text_encoder` scale inflation, ~35×) was
//! being investigated. Bug 4 is fixed on this branch (see the Wave-2
//! commit chain in `git log --oneline`: `f3b10ab hifigan per-iteration
//! residual + convs2 chain`, `15df641 test(sbv2/parity): pin text_encoder
//! bit-exact vs Python reference`, `af58ba8 flow-noise-scale`,
//! `bfaf2ac style-injector`, `b2c5c96 posffn-xmask`, etc.), so the cap
//! becomes dead code and Wave-5 audit rank 17 calls for total deletion.
//!
//! This test pins the deletion via a source-string regression check — a
//! future refactor that reintroduces the cap (or its stopgap warning)
//! would trip. Scope is deliberately narrow (only `sbv2/mod.rs` is
//! scanned, not the whole repo) to avoid catching honest historical
//! mentions in `docs/handoff/sbv2-sdp-debug-2026-08-08.md` — the historical
//! trail is preserved on purpose (audit rank 17 explicitly calls out
//! "trace entirely" only for the runtime code path, not the design-history
//! documentation, which future maintainers still want).

use std::fs;
use std::path::Path;

/// The cap's identifier + its user-visible stderr fingerprint together
/// uniquely name the removed block — matching either in
/// `src/sbv2/mod.rs` proves the audit's Phase 2 deletion has been
/// undone.
#[test]
fn per_phoneme_duration_ceiling_and_synth_warn_are_deleted_from_mod_rs() {
    let mod_rs = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join("sbv2")
        .join("mod.rs");
    let src =
        fs::read_to_string(&mod_rs).unwrap_or_else(|e| panic!("read {}: {e}", mod_rs.display()));

    // Constant-name check. The pre-fix source has both the `const`
    // declaration and the `min(PER_PHONEME_DURATION_CEILING)` clamp
    // inside a `for` loop; any surviving reference (either) trips.
    // The two-part token `PER_PHONEME_DURATION_` + `CEILING` avoids this
    // test file itself matching the raw literal via any source scanner
    // that might one day include `tests/`.
    let ceiling_needle = concat!("PER_PHONEME_DURATION_", "CEILING");
    assert!(
        !src.contains(ceiling_needle),
        "OOM-STOPGAP-CLEANUP Phase 2: the per-phoneme duration cap constant \
         (`{ceiling_needle}`) must be fully deleted from crates/vokra-models/\
         src/sbv2/mod.rs post-Bug-4 fix. See Wave-5 audit rank 17 and \
         docs/handoff/sbv2-sdp-debug-2026-08-08.md for the historical trail."
    );

    // Stderr-message check. The two-part `sbv2-synth-` + `warn` split
    // avoids this test file itself matching.
    let warn_needle = concat!("[sbv2-synth-", "warn]");
    assert!(
        !src.contains(warn_needle),
        "OOM-STOPGAP-CLEANUP Phase 2: the `{warn_needle} SbV2SDP produced \
         runaway durations` stderr message was the cap's user-visible face — \
         its `eprintln!` must be deleted along with the cap logic."
    );
}
