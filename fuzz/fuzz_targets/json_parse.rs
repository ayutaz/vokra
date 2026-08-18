#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // This std-only parser consumes untrusted safetensors metadata and model
    // configuration. It must reject arbitrary bytes without panicking.
    let _ = vokra_core::json::parse(data);
});
