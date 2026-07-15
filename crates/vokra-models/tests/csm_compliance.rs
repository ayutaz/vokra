//! M4-05 T26 — WatermarkConfig default-ON (config surface only) + M2-13
//! compliance-gate tests for CSM GGUFs.
//!
//! # Watermark posture (NFR-LG-01 restatement — ADR M4-05 §D1-(f))
//!
//! The embedding backend is **Deferred** (2026-07-04 依頼者ドロップ;
//! 復活 = v1.0.x patch or M5-06, owner judgement — 2026-07-14 見送り確定).
//! The config is default **ON** (opt-out surface preserved) and
//! `backend_status()` says `Deferred` honestly — no fake marker is ever
//! attached. **Deployer-side visible disclosure stays a MUST**
//! (docs/legal-compliance.md §1.4) for EU AI Act Article 50
//! detectability during the deferral.
//!
//! # Compliance gate (M2-13)
//!
//! `sesame-csm` / `csm-1b` are registered `Permissive` (Apache 2.0 /
//! Apache 2.0 — docs/license-audit.md); the tests pin: a provenance-
//! stamped CSM GGUF loads without a research flag, a registry id resolves
//! permissive, and a GGUF with no weight-license information is refused
//! fail-closed.

use std::sync::Arc;

use vokra_core::gguf::{GgufArray, GgufBuilder, GgufMetadataValue, GgufValueType};
use vokra_core::{
    CompliancePolicy, LicenseClass, VokraError, WatermarkBackendStatus, WatermarkConfig,
    check_weight_license, registry_lookup, stamp_provenance,
};
use vokra_models::csm::{CsmConfig, CsmEngine, FixtureByteTokenizer};
use vokra_models::mimi::MimiNeuralConfig;

fn fixture_gguf(with_provenance: bool) -> Vec<u8> {
    let cfg = CsmConfig::tiny_for_tests();
    let mut mimi_cfg = MimiNeuralConfig::tiny_for_tests();
    mimi_cfg.quantizer.n_q = cfg.n_codebooks;
    mimi_cfg.quantizer.bins = cfg.audio_vocab_size;
    let mut fixed = cfg;
    fixed.sample_rate = mimi_cfg.sample_rate;
    fixed.frame_rate_mhz = mimi_cfg.frame_rate_mhz;
    let mut b = GgufBuilder::new();
    b.add_string("vokra.model.arch", "csm");
    if with_provenance {
        stamp_provenance(
            &mut b,
            LicenseClass::Permissive,
            "Apache-2.0",
            Some("sesame/csm-1b"),
            Some("huggingface"),
        );
    }
    fixed.write_gguf_metadata(&mut b);
    mimi_cfg.write_gguf_metadata(&mut b);
    b.add_metadata(
        "vokra.tokenizer.model",
        GgufMetadataValue::Array(GgufArray {
            element_type: GgufValueType::U8,
            values: vec![GgufMetadataValue::U8(7)],
        }),
    );
    b.to_bytes().expect("serialize")
}

#[test]
fn watermark_config_is_default_on_with_deferred_backend() {
    let engine = CsmEngine::synthesized_fixture(1).unwrap();
    let w = engine.watermark();
    assert!(
        !w.audioseal_opted_out(),
        "FR-CP-01 design intent: AudioSeal default ON (opt-out, never opt-in)"
    );
    assert_eq!(
        w.backend_status(),
        WatermarkBackendStatus::Deferred,
        "no embedding backend — the audio is NOT watermarked and the config \
         never pretends otherwise (deployer-side disclosure MUST, \
         docs/legal-compliance.md §1.4)"
    );
}

#[test]
fn watermark_opt_out_surface_is_preserved() {
    let engine = CsmEngine::synthesized_fixture(1)
        .unwrap()
        .with_watermark(WatermarkConfig {
            audioseal: false, // the FR-CP-01 opt-out path (warning hook fires at synthesis)
            ..WatermarkConfig::default()
        });
    assert!(engine.watermark().audioseal_opted_out());
    // Opting out changes the *setting* only; the backend stays Deferred
    // either way.
    assert_eq!(
        engine.watermark().backend_status(),
        WatermarkBackendStatus::Deferred
    );
}

#[test]
fn registry_knows_the_csm_ids_as_permissive() {
    assert_eq!(
        registry_lookup("sesame-csm"),
        Some(LicenseClass::Permissive)
    );
    assert_eq!(registry_lookup("csm-1b"), Some(LicenseClass::Permissive));
}

#[test]
fn provenance_stamped_csm_gguf_passes_the_strict_gate() {
    let bytes = fixture_gguf(true);
    let file = vokra_core::gguf::GgufFile::parse(bytes.clone()).unwrap();
    let res = check_weight_license(&file, &CompliancePolicy::strict()).expect("gate passes");
    assert_eq!(res.class, LicenseClass::Permissive);
    // And the full engine load path agrees (T04 provenance → T26 gate).
    let engine = CsmEngine::from_gguf_with_policy(&bytes, &CompliancePolicy::strict())
        .expect("stock CSM GGUF loads without a research flag");
    // The GGUF tokenizer is the honest T29 stub; swapping in the explicit
    // fixture tokenizer keeps the engine usable for smoke flows.
    let vocab = engine.config().text_vocab_size;
    engine
        .with_tokenizer(Arc::new(FixtureByteTokenizer::new(vocab).unwrap()))
        .expect("fixture tokenizer vocab matches");
}

#[test]
fn license_less_gguf_is_refused_fail_closed() {
    let bytes = fixture_gguf(false);
    let err = CsmEngine::from_gguf_with_policy(&bytes, &CompliancePolicy::strict()).unwrap_err();
    assert!(
        matches!(err, VokraError::ResearchLicenseRequired { .. }),
        "a weight-license-less GGUF must be refused (fail-closed M2-13 gate), got {err:?}"
    );
}
