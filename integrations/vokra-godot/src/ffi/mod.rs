//! Hand-written `extern "C"` bindings for the GDExtension ABI + the Vokra C
//! ABI (ADR-0011 §D3). No `godot-cpp`, no `gdext-rs`, no `bindgen` — the
//! symbols and struct layouts declared here mirror
//! `godot/core/extension/gdextension_interface.h` (MIT, Godot 4.1+) and
//! `include/vokra.h` (cbindgen-generated from `crates/vokra-capi`).
//!
//! Only the subset that Vokra actually calls is bound. Adding a new
//! GDExtension API MUST come with a `// SAFETY:` comment at every call site
//! (workspace lint `undocumented_unsafe_blocks`, NFR-RL-07).
//!
//! # Layout stability
//!
//! GDExtension's public ABI is versioned by `godot_version.h`. This crate
//! targets Godot 4.1+ (`compatibility_minimum = "4.1"` in `vokra.gdextension`,
//! ADR-0011 §D9) and MUST be re-audited when Godot bumps a major version.

pub mod capi;
pub mod gdextension;
