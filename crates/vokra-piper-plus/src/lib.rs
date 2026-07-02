//! # vokra-piper-plus
//!
//! piper-plus integration layer for Vokra (SRS §1.3: "G2P 流用ブリッジ +
//! voice model 変換補助" — G2P reuse bridge + voice-model conversion
//! helpers).
//!
//! # Scope boundary (client decision 2026-07-02; SRS §5-(9) revision, FR-MD-03)
//!
//! **This crate is the G2P reuse bridge plus voice-model conversion helpers
//! only. The piper-plus inference core (MB-iSTFT-VITS2) is natively
//! implemented in `vokra-models` as Vokra's first native TTS** — the former
//! "wrap piper-plus" positioning was abolished by the client decision of
//! 2026-07-02. **The inference path never brings in onnxruntime**: piper-plus
//! voice models (ONNX) are converted to GGUF by the *offline* conversion
//! tooling, and the runtime never loads ONNX (FR-LD-05, permanent
//! constraint).
//!
//! Concretely:
//!
//! - **G2P bridge**: the 8-language text preprocessing is reused from the
//!   existing piper-plus (MIT) implementation for the time being (a Rust
//!   port is future work, to be re-evaluated); implemented in **M0-07**.
//! - **Voice-model conversion helpers**: assist the offline ONNX → GGUF
//!   conversion of piper-plus voices (offline tooling side; the runtime
//!   loader is M0-03).
//!
//! M0-02 ships only this crate skeleton; the G2P bridge and the native
//! inference implementation are M0-07 work.

#[cfg(test)]
mod tests {
    #[test]
    fn links_against_vokra_core() {
        // Smoke test for the crate wiring (M0-02-T02).
        let kind = vokra_core::BackendKind::Cpu;
        assert_eq!(kind, vokra_core::BackendKind::Cpu);
    }
}
