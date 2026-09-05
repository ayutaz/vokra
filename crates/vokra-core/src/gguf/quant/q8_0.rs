//! `Q8_0` dequantization (ggml type tag 8).
//!
//! One block contains an FP16 scale followed by 32 signed bytes, for a total
//! of 34 bytes. The value at each position is `scale * quantized_byte`.

use super::f16_to_f32;
// M5-03-T05: `Vec` is an `alloc` type in the no_std subset.
#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

const BLOCK_SIZE: usize = 32;
const BLOCK_BYTES: usize = 34;

/// Dequantizes whole `Q8_0` blocks. Payload length and block alignment have
/// already been checked by the dispatch in [`super::dequantize`].
pub(super) fn dequantize(bytes: &[u8], n_elements: usize) -> Vec<f32> {
    let mut output = vec![0.0; n_elements];
    for (block_index, output_block) in output.chunks_exact_mut(BLOCK_SIZE).enumerate() {
        let block = &bytes[block_index * BLOCK_BYTES..(block_index + 1) * BLOCK_BYTES];
        let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        for (value, &quantized) in output_block.iter_mut().zip(&block[2..]) {
            *value = scale * f32::from(quantized as i8);
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::tensor::GgmlType;

    #[test]
    fn decodes_scale_and_signed_values() {
        let mut block = vec![0u8; BLOCK_BYTES];
        block[..2].copy_from_slice(&0x3C00u16.to_le_bytes()); // 1.0
        block[2] = 127;
        block[3] = 0x80; // -128
        let output = super::dequantize(&block, BLOCK_SIZE);
        assert_eq!(output[0], 127.0);
        assert_eq!(output[1], -128.0);
        assert!(output[2..].iter().all(|&value| value == 0.0));
    }

    #[test]
    fn dispatch_accepts_q8_0_wire_layout() {
        let bytes = vec![0u8; GgmlType::Q8_0.type_size()];
        let output = super::super::dequantize(GgmlType::Q8_0, &bytes, BLOCK_SIZE).unwrap();
        assert_eq!(output, vec![0.0; BLOCK_SIZE]);
    }
}
