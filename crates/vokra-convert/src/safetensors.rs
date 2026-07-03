//! Re-export of the safetensors reader, now hosted in
//! [`vokra_core::safetensors`](vokra_core::safetensors).
//!
//! Promoted from `vokra-convert` to `vokra-core` in M1-02 so the runtime can
//! direct-load safetensors as a weight provider (FR-LD-04 / IF-06). The reader
//! is still hand-written and std-only (zero external dependencies). The
//! converter's Whisper path consumes the same [`SafetensorsFile`] /
//! [`SafeTensorInfo`] types through this re-export.

pub(crate) use vokra_core::safetensors::*;
