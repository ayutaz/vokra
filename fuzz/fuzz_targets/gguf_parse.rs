#![no_main]

use libfuzzer_sys::fuzz_target;
use vokra_core::gguf::GgufFile;

fuzz_target!(|data: &[u8]| {
    // A malformed model is an ordinary parse error. Panics, OOM-amplifying
    // length handling, or sanitizer findings are bugs at this trust boundary.
    let _ = GgufFile::parse(data.to_vec());
});
