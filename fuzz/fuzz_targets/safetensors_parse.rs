#![no_main]

use libfuzzer_sys::fuzz_target;
use vokra_core::safetensors::SafetensorsFile;

fuzz_target!(|data: &[u8]| {
    // Model metadata, tensor shapes, and offsets are attacker-controlled at
    // this boundary. Invalid inputs must remain recoverable parse errors.
    let _ = SafetensorsFile::parse(data.to_vec());
});
