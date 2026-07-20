//! Mimi GGUF dual-write guard (M4-RESIDUAL-B (B), T14/T15).
//!
//! The Mimi converter writes **two coexisting tensor name groups** by design
//! (`vokra-convert/src/models/mimi.rs` module docs §5):
//!
//! 1. **verbatim pass-through** — every upstream tensor under its original name
//!    (`encoder.model.*` / `decoder.model.*` / `quantizer.*`), retained so an
//!    M4-05/06 (CSM / Moshi) or external tool can consume the raw checkpoint
//!    from the same GGUF without re-running the converter;
//! 2. **derived structural** — `mimi.enc.*` / `mimi.dec.*` (the SEANet chain in
//!    the runtime binder layout, read by [`MimiEncoder`] / [`MimiNeuralDecoder`])
//!    plus the effective codebook tables `vokra.mimi.codebook_tables` (read by
//!    [`MimiCodecGguf`] — the RVQ path).
//!
//! This is **not** a bug to dedup: the two groups feed *different* consumers.
//! This guard converts the **real** checkpoint with the **current** converter
//! and asserts both groups exist in the output — so a future dedup (size
//! optimization, deferred by this WP) that silently drops one group turns the
//! guard **red** rather than breaking a consumer at load.
//!
//! Gated on `VOKRA_MIMI_SAFETENSORS` (the moshi-native
//! `tokenizer-e351c8d8-checkpoint125.safetensors`, ~385 MB — uncommittable). It
//! skips cleanly when unset (CI). Optionally also parses `VOKRA_MIMI_GGUF` to
//! record whether an existing cache GGUF still carries the structural group (a
//! pre-T29 cache is stale = structural-empty and cannot feed the neural binder).
//!
//! ```text
//! VOKRA_MIMI_SAFETENSORS=~/.cache/vokra-eval/weights/mimi/tokenizer-e351c8d8-checkpoint125.safetensors \
//!     cargo test -p vokra-models --test mimi_dual_write_guard -- --nocapture
//! ```

use std::path::PathBuf;

use vokra_convert::{ModelKind, convert_file};
use vokra_core::gguf::GgufFile;
use vokra_models::codec::MimiCodecGguf;
use vokra_models::mimi::{MimiEncoder, MimiNeuralConfig, MimiNeuralDecoder};

fn tensor_names(file: &GgufFile) -> Vec<String> {
    file.tensors().iter().map(|t| t.name.clone()).collect()
}

fn count_prefix(names: &[String], prefix: &str) -> usize {
    names.iter().filter(|n| n.starts_with(prefix)).count()
}

#[test]
fn mimi_converter_dual_writes_verbatim_and_structural_groups() {
    let Ok(src) = std::env::var("VOKRA_MIMI_SAFETENSORS") else {
        eprintln!("skip: VOKRA_MIMI_SAFETENSORS unset (real-checkpoint gated)");
        return;
    };
    let src = PathBuf::from(src);
    assert!(src.exists(), "source safetensors {src:?} does not exist");

    // Run the CURRENT converter (the dual-write is a converter behaviour, not a
    // property of any pre-existing GGUF — a chain-carrying checkpoint triggers
    // the presence-driven structural adapter, `mimi.rs:360-362`).
    let out =
        std::env::temp_dir().join(format!("vokra_mimi_dualwrite_{}.gguf", std::process::id()));
    let summary = convert_file(ModelKind::Mimi, &src, &out).expect("convert mimi checkpoint");
    let _ = summary; // ConvertSummary path/notes not asserted here.

    let file = GgufFile::open(&out).expect("open freshly-generated GGUF");
    let names = tensor_names(&file);

    // --- Group 1: verbatim pass-through (retained upstream names). ----------
    let verbatim_encoder = count_prefix(&names, "encoder.");
    let verbatim_decoder = count_prefix(&names, "decoder.");
    let verbatim_quantizer = count_prefix(&names, "quantizer.");
    assert!(
        names
            .iter()
            .any(|n| n == "encoder.model.0.conv.conv.weight"),
        "verbatim pass-through missing `encoder.model.0.conv.conv.weight` \
         (this is the chain-presence key the structural adapter also gates on)"
    );
    assert!(
        verbatim_decoder > 0,
        "verbatim `decoder.*` pass-through absent"
    );
    assert!(
        verbatim_quantizer > 0,
        "verbatim `quantizer.*` pass-through absent"
    );

    // --- Group 2: derived structural + effective tables. -------------------
    let structural_enc = count_prefix(&names, "mimi.enc.");
    let structural_dec = count_prefix(&names, "mimi.dec.");
    assert!(
        structural_enc > 0,
        "structural `mimi.enc.*` absent — dual-write broken (dedup must not \
         remove the group the neural binder reads)"
    );
    assert!(
        structural_dec > 0,
        "structural `mimi.dec.*` absent — dual-write broken"
    );
    assert!(
        names.iter().any(|n| n == "vokra.mimi.codebook_tables"),
        "derived `vokra.mimi.codebook_tables` absent (the RVQ consumer's input)"
    );

    // --- Both consumers actually BIND from this one GGUF (T15). -------------
    // RVQ path (derived tables) + neural chain (structural names) — proving the
    // dual-write keeps *both* live, not just present as bytes.
    let codec = MimiCodecGguf::from_gguf(&file).expect("RVQ consumer binds (codebook_tables)");
    assert!(codec.attrs.n_codebooks > 0);
    let cfg = MimiNeuralConfig::from_gguf(&file).expect("vokra.mimi.* config chunk reads back");
    MimiEncoder::from_gguf(&file, &cfg).expect("structural encoder consumer binds (mimi.enc.*)");
    MimiNeuralDecoder::from_gguf(&file, &cfg)
        .expect("structural decoder consumer binds (mimi.dec.*)");

    // --- Measured accounting (T15) — pinned from THIS output, not intake. ---
    let bytes = std::fs::metadata(&out).expect("stat GGUF").len();
    eprintln!(
        "mimi dual-write GGUF: {bytes} bytes | verbatim: encoder={verbatim_encoder} \
         decoder={verbatim_decoder} quantizer={verbatim_quantizer} | \
         structural: mimi.enc.*={structural_enc} mimi.dec.*={structural_dec}"
    );

    let _ = std::fs::remove_file(&out);
}

/// Records whether an existing `VOKRA_MIMI_GGUF` cache still carries the
/// structural group. A **pre-T29 cache is stale** (structural-empty) and would
/// fail the neural binder ([`MimiNeuralDecoder::from_gguf`] hard-requires
/// `mimi.dec.*`); this is an observation for the record, not a gate on the
/// (transient) cache contents.
#[test]
fn existing_mimi_cache_structural_group_observation() {
    let Ok(path) = std::env::var("VOKRA_MIMI_GGUF") else {
        eprintln!("skip: VOKRA_MIMI_GGUF unset (cache observation)");
        return;
    };
    let file = GgufFile::open(&path).expect("open cache GGUF");
    let names = tensor_names(&file);
    let structural = count_prefix(&names, "mimi.enc.") + count_prefix(&names, "mimi.dec.");
    let verbatim = count_prefix(&names, "encoder.") + count_prefix(&names, "decoder.");
    eprintln!(
        "existing VOKRA_MIMI_GGUF: {} tensors, verbatim(enc+dec)={verbatim}, structural(enc+dec)={structural} — {}",
        names.len(),
        if structural == 0 {
            "STALE (pre-T29): structural-empty, cannot feed MimiNeuralDecoder::from_gguf"
        } else {
            "carries the structural group (post-T29)"
        }
    );
    // A structural-empty cache is a real, provable state; the neural binder
    // rejects it loudly (FR-EX-08). Assert that consequence when it applies.
    if structural == 0 {
        let cfg = MimiNeuralConfig::from_gguf(&file);
        if let Ok(cfg) = cfg {
            assert!(
                MimiNeuralDecoder::from_gguf(&file, &cfg).is_err(),
                "a structural-empty GGUF must fail the neural binder, not half-load"
            );
        }
    }
}
