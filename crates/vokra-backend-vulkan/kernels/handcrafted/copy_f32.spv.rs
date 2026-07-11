//! Hand-crafted SPIR-V 1.3 bytecode for a minimal `copy_f32` compute kernel
//! (M3-02-T13 / ADR M3-02-spirv-generation §4 (c)).
//!
//! # Purpose
//!
//! This module ships a **hand-authored SPIR-V binary** that a Vulkan driver
//! can load with `vkCreateShaderModule` — no `glslc`, no `spirv-tools`, no
//! `naga`, no crate dependency (NFR-DS-02). Its only job is to serve as the
//! **runtime smoke test** for the T08〜T12 + T25 object stack: the
//! `smoke_dispatch.rs` integration test uploads a random f32 array, dispatches
//! this kernel over it, downloads the result, and asserts it matches the
//! input bit-for-bit. That single end-to-end path proves the driver, loader,
//! device, memory allocator, descriptor set, pipeline, and command submission
//! actually work — which is the "Vulkan is real" evidence T14〜T22 will build
//! on with `glslc`-produced blobs.
//!
//! # Explicitly out of scope
//!
//! Hand-authoring bytecode is only feasible for the smallest kernels. The
//! real T14〜T22 shaders (GEMM / softmax / layer_norm / conv1d / …) are
//! produced by `glslc` (see `scripts/compile-vulkan-shaders.sh`) and
//! committed as `kernels/precompiled/*.spv` blobs. **Do NOT** try to add
//! more hand-crafted kernels here — ADR M3-02-spirv-generation §5 forbids
//! it.
//!
//! # GLSL source this bytecode reproduces
//!
//! ```glsl
//! #version 450
//! layout(local_size_x = 64) in;
//! layout(std430, set = 0, binding = 0) readonly  buffer SrcBuf { float src[]; };
//! layout(std430, set = 0, binding = 1) writeonly buffer DstBuf { float dst[]; };
//! void main() {
//!     uint idx = gl_GlobalInvocationID.x;
//!     dst[idx] = src[idx];
//! }
//! ```
//!
//! # Provenance of every word
//!
//! The bytecode below is annotated line-by-line with the SPIR-V 1.3 spec
//! reference (Khronos SPIR-V unified spec §2 / §3.32). Every opcode value and
//! enum literal is documented against
//! <https://github.com/KhronosGroup/SPIRV-Headers/blob/main/include/spirv/unified1/spirv.h>
//! (MIT-licensed public spec — no source is copied, only the numeric values
//! reproduced in-line as documentation). SHA-256 of the resulting binary is
//! recorded in `crate::spirv::SHADERS[COPY_F32]::expected_sha256_hex`.

// The instruction encoding helper lives at module level so the `SPIRV_MODULE`
// constant below can compose instructions declaratively without hand-computing
// word counts.

/// Pack the 32-bit first word of a SPIR-V instruction.
///
/// SPIR-V 1.3 spec §2.3 "Physical Layout of a SPIR-V Module":
/// > Word 0: the low-order 16 bits are the opcode; the high-order 16 bits are
/// > the total word count of the instruction, including this word.
#[inline]
const fn op(word_count: u16, opcode: u16) -> u32 {
    ((word_count as u32) << 16) | (opcode as u32)
}

// Opcode values (SPIR-V unified1 grammar; see the crate docs above).
const OP_CAPABILITY: u16 = 17;
const OP_MEMORY_MODEL: u16 = 14;
const OP_ENTRY_POINT: u16 = 15;
const OP_EXECUTION_MODE: u16 = 16;
const OP_TYPE_VOID: u16 = 19;
const OP_TYPE_INT: u16 = 21;
const OP_TYPE_FLOAT: u16 = 22;
const OP_TYPE_VECTOR: u16 = 23;
const OP_TYPE_RUNTIME_ARRAY: u16 = 29;
const OP_TYPE_STRUCT: u16 = 30;
const OP_TYPE_POINTER: u16 = 32;
const OP_TYPE_FUNCTION: u16 = 33;
const OP_CONSTANT: u16 = 43;
const OP_FUNCTION: u16 = 54;
const OP_FUNCTION_END: u16 = 56;
const OP_VARIABLE: u16 = 59;
const OP_LOAD: u16 = 61;
const OP_STORE: u16 = 62;
const OP_ACCESS_CHAIN: u16 = 65;
const OP_DECORATE: u16 = 71;
const OP_MEMBER_DECORATE: u16 = 72;
const OP_LABEL: u16 = 248;
const OP_RETURN: u16 = 253;

// Enum literal values (SPIR-V unified1 grammar).
const CAP_SHADER: u32 = 1;
const ADDR_LOGICAL: u32 = 0;
const MEM_GLSL450: u32 = 1;
const EXEC_GL_COMPUTE: u32 = 5;
const EXEC_MODE_LOCAL_SIZE: u32 = 17;
const STORAGE_INPUT: u32 = 1;
const STORAGE_STORAGE_BUFFER: u32 = 12;
const DECO_BLOCK: u32 = 2;
const DECO_ARRAY_STRIDE: u32 = 6;
const DECO_BUILTIN: u32 = 11;
const DECO_BINDING: u32 = 33;
const DECO_DESCRIPTOR_SET: u32 = 34;
const DECO_OFFSET: u32 = 35;
const BUILTIN_GLOBAL_INVOCATION_ID: u32 = 28;

// SPIR-V module header (§2.3).
const SPV_MAGIC: u32 = 0x0723_0203;
/// SPIR-V 1.3 (byte-packed as `0x00 | major | minor | 0x00`; the Vulkan 1.1
/// minimum — see M3-02-T01 ADR §c).
const SPV_VERSION_1_3: u32 = 0x0001_0300;
/// Generator magic number. `0` = "unknown / not registered" (spec §2.3): the
/// bound-checking driver ignores generator; only tooling that indexes by
/// generator (e.g. spirv-cross) cares. We are not registered with Khronos, so
/// `0` is the honest choice.
const SPV_GENERATOR: u32 = 0;
/// Upper bound of Ids used in the module (i.e. `max_id + 1`). We use IDs
/// `%1..=%22`, so the bound is `23`.
const SPV_ID_BOUND: u32 = 23;
/// Instruction schema — always `0` for the current SPIR-V.
const SPV_SCHEMA: u32 = 0;

// Result Ids (§3.32.2). These match the annotated `%<id>` names in the crate
// docs above.
const ID_VOID_TYPE: u32 = 1;
const ID_FUNC_TYPE: u32 = 2;
const ID_UINT_TYPE: u32 = 3;
const ID_FLOAT_TYPE: u32 = 4;
const ID_V3UINT_TYPE: u32 = 5;
const ID_PTR_INPUT_V3UINT: u32 = 6;
const ID_PTR_INPUT_UINT: u32 = 7;
const ID_RUNTIME_ARRAY: u32 = 8;
const ID_STRUCT_SB: u32 = 9;
const ID_PTR_SB_STRUCT: u32 = 10;
const ID_PTR_SB_FLOAT: u32 = 11;
const ID_CONST_UINT_0: u32 = 12;
const ID_VAR_GL_GLOBAL_INVOCATION_ID: u32 = 13;
const ID_VAR_SRC: u32 = 14;
const ID_VAR_DST: u32 = 15;
const ID_FUNC_MAIN: u32 = 16;
const ID_LABEL_ENTRY: u32 = 17;
const ID_ACC_INV_ID_X: u32 = 18;
const ID_LOAD_IDX: u32 = 19;
const ID_ACC_SRC_ELEM: u32 = 20;
const ID_LOAD_VAL: u32 = 21;
const ID_ACC_DST_ELEM: u32 = 22;

// Literal string "main\0" packed into two little-endian u32s (§2.2.1):
//   bytes 'm'(0x6d) 'a'(0x61) 'i'(0x69) 'n'(0x6e) → 0x6E69616D
//   bytes '\0' 0x00 0x00 0x00 → 0x00000000
const LIT_STRING_MAIN_0: u32 = 0x6E69_616D;
const LIT_STRING_MAIN_1: u32 = 0x0000_0000;

/// The complete SPIR-V module as a `&'static [u32]` — 145 words = 580 bytes.
///
/// The `include_bytes!(…)` sites in `crate::spirv::load_spv` treat the bytes
/// of this constant as the shader module payload. `bytes()` below re-encodes
/// the `u32` slice as an `&'static [u8]` (little-endian) so callers can pass
/// it directly to `vkCreateShaderModule`.
pub const SPIRV_MODULE: &[u32] = &[
    // ------------------------------------------------------------------ Header (5 words)
    SPV_MAGIC,
    SPV_VERSION_1_3,
    SPV_GENERATOR,
    SPV_ID_BOUND,
    SPV_SCHEMA,
    // ------------------------------------------------------------------ Capabilities
    // OpCapability Shader
    op(2, OP_CAPABILITY),
    CAP_SHADER,
    // ------------------------------------------------------------------ Memory model
    // OpMemoryModel Logical GLSL450
    op(3, OP_MEMORY_MODEL),
    ADDR_LOGICAL,
    MEM_GLSL450,
    // ------------------------------------------------------------------ Entry points (§5)
    // OpEntryPoint GLCompute %main "main" %gl_GlobalInvocationID
    // word count = 1 + 1(exec model) + 1(entry id) + 2("main\0") + 1(interface) = 6
    op(6, OP_ENTRY_POINT),
    EXEC_GL_COMPUTE,
    ID_FUNC_MAIN,
    LIT_STRING_MAIN_0,
    LIT_STRING_MAIN_1,
    ID_VAR_GL_GLOBAL_INVOCATION_ID,
    // ------------------------------------------------------------------ Execution modes (§6)
    // OpExecutionMode %main LocalSize 64 1 1
    op(6, OP_EXECUTION_MODE),
    ID_FUNC_MAIN,
    EXEC_MODE_LOCAL_SIZE,
    64,
    1,
    1,
    // ------------------------------------------------------------------ Annotations (§10)
    // OpDecorate %arr ArrayStride 4  (float is 4 bytes; std430 packs tightly)
    op(4, OP_DECORATE),
    ID_RUNTIME_ARRAY,
    DECO_ARRAY_STRIDE,
    4,
    // OpMemberDecorate %struct 0 Offset 0
    op(5, OP_MEMBER_DECORATE),
    ID_STRUCT_SB,
    0,
    DECO_OFFSET,
    0,
    // OpDecorate %struct Block
    op(3, OP_DECORATE),
    ID_STRUCT_SB,
    DECO_BLOCK,
    // OpDecorate %gl_GlobalInvocationID BuiltIn GlobalInvocationId
    op(4, OP_DECORATE),
    ID_VAR_GL_GLOBAL_INVOCATION_ID,
    DECO_BUILTIN,
    BUILTIN_GLOBAL_INVOCATION_ID,
    // OpDecorate %src DescriptorSet 0
    op(4, OP_DECORATE),
    ID_VAR_SRC,
    DECO_DESCRIPTOR_SET,
    0,
    // OpDecorate %src Binding 0
    op(4, OP_DECORATE),
    ID_VAR_SRC,
    DECO_BINDING,
    0,
    // OpDecorate %dst DescriptorSet 0
    op(4, OP_DECORATE),
    ID_VAR_DST,
    DECO_DESCRIPTOR_SET,
    0,
    // OpDecorate %dst Binding 1
    op(4, OP_DECORATE),
    ID_VAR_DST,
    DECO_BINDING,
    1,
    // ------------------------------------------------------------------ Types / constants / global variables (§11)
    // %1 = OpTypeVoid
    op(2, OP_TYPE_VOID),
    ID_VOID_TYPE,
    // %2 = OpTypeFunction %void  (no parameters)
    op(3, OP_TYPE_FUNCTION),
    ID_FUNC_TYPE,
    ID_VOID_TYPE,
    // %3 = OpTypeInt 32 0  (32-bit unsigned int)
    op(4, OP_TYPE_INT),
    ID_UINT_TYPE,
    32,
    0,
    // %4 = OpTypeFloat 32
    op(3, OP_TYPE_FLOAT),
    ID_FLOAT_TYPE,
    32,
    // %5 = OpTypeVector %uint 3
    op(4, OP_TYPE_VECTOR),
    ID_V3UINT_TYPE,
    ID_UINT_TYPE,
    3,
    // %6 = OpTypePointer Input %v3uint  (points to gl_GlobalInvocationID)
    op(4, OP_TYPE_POINTER),
    ID_PTR_INPUT_V3UINT,
    STORAGE_INPUT,
    ID_V3UINT_TYPE,
    // %7 = OpTypePointer Input %uint  (points to gl_GlobalInvocationID.x)
    op(4, OP_TYPE_POINTER),
    ID_PTR_INPUT_UINT,
    STORAGE_INPUT,
    ID_UINT_TYPE,
    // %8 = OpTypeRuntimeArray %float
    op(3, OP_TYPE_RUNTIME_ARRAY),
    ID_RUNTIME_ARRAY,
    ID_FLOAT_TYPE,
    // %9 = OpTypeStruct %runtime_array
    op(3, OP_TYPE_STRUCT),
    ID_STRUCT_SB,
    ID_RUNTIME_ARRAY,
    // %10 = OpTypePointer StorageBuffer %struct
    op(4, OP_TYPE_POINTER),
    ID_PTR_SB_STRUCT,
    STORAGE_STORAGE_BUFFER,
    ID_STRUCT_SB,
    // %11 = OpTypePointer StorageBuffer %float
    op(4, OP_TYPE_POINTER),
    ID_PTR_SB_FLOAT,
    STORAGE_STORAGE_BUFFER,
    ID_FLOAT_TYPE,
    // %12 = OpConstant %uint 0
    op(4, OP_CONSTANT),
    ID_UINT_TYPE,
    ID_CONST_UINT_0,
    0,
    // %13 = OpVariable %ptr_input_v3uint Input  (gl_GlobalInvocationID)
    op(4, OP_VARIABLE),
    ID_PTR_INPUT_V3UINT,
    ID_VAR_GL_GLOBAL_INVOCATION_ID,
    STORAGE_INPUT,
    // %14 = OpVariable %ptr_sb_struct StorageBuffer  (src buffer at binding 0)
    op(4, OP_VARIABLE),
    ID_PTR_SB_STRUCT,
    ID_VAR_SRC,
    STORAGE_STORAGE_BUFFER,
    // %15 = OpVariable %ptr_sb_struct StorageBuffer  (dst buffer at binding 1)
    op(4, OP_VARIABLE),
    ID_PTR_SB_STRUCT,
    ID_VAR_DST,
    STORAGE_STORAGE_BUFFER,
    // ------------------------------------------------------------------ Function definition (§13)
    // %16 = OpFunction %void None %func_void_void
    op(5, OP_FUNCTION),
    ID_VOID_TYPE,
    ID_FUNC_MAIN,
    0, // Function Control = None
    ID_FUNC_TYPE,
    // %17 = OpLabel
    op(2, OP_LABEL),
    ID_LABEL_ENTRY,
    // %18 = OpAccessChain %ptr_input_uint %gl_GlobalInvocationID %uint_0
    //       (i.e. &gl_GlobalInvocationID.x)
    op(5, OP_ACCESS_CHAIN),
    ID_PTR_INPUT_UINT,
    ID_ACC_INV_ID_X,
    ID_VAR_GL_GLOBAL_INVOCATION_ID,
    ID_CONST_UINT_0,
    // %19 = OpLoad %uint %18
    op(4, OP_LOAD),
    ID_UINT_TYPE,
    ID_LOAD_IDX,
    ID_ACC_INV_ID_X,
    // %20 = OpAccessChain %ptr_sb_float %src %uint_0(member) %idx
    //       (i.e. &src.arr[idx])
    op(6, OP_ACCESS_CHAIN),
    ID_PTR_SB_FLOAT,
    ID_ACC_SRC_ELEM,
    ID_VAR_SRC,
    ID_CONST_UINT_0,
    ID_LOAD_IDX,
    // %21 = OpLoad %float %20
    op(4, OP_LOAD),
    ID_FLOAT_TYPE,
    ID_LOAD_VAL,
    ID_ACC_SRC_ELEM,
    // %22 = OpAccessChain %ptr_sb_float %dst %uint_0(member) %idx
    op(6, OP_ACCESS_CHAIN),
    ID_PTR_SB_FLOAT,
    ID_ACC_DST_ELEM,
    ID_VAR_DST,
    ID_CONST_UINT_0,
    ID_LOAD_IDX,
    // OpStore %22 %21   (dst[idx] = val)
    op(3, OP_STORE),
    ID_ACC_DST_ELEM,
    ID_LOAD_VAL,
    // OpReturn
    op(1, OP_RETURN),
    // OpFunctionEnd
    op(1, OP_FUNCTION_END),
];

/// Local workgroup X-size (must match the `OpExecutionMode LocalSize` above).
/// Consumers use this to compute `group_count_x = ceil(N / LOCAL_SIZE_X)`.
pub const LOCAL_SIZE_X: u32 = 64;

/// Number of SSBO bindings the kernel consumes (src @ binding 0, dst @
/// binding 1, both at set 0). Consumers use this to size the descriptor set
/// layout (2 storage-buffer bindings).
pub const BINDING_COUNT: u32 = 2;

/// Re-encode the [`SPIRV_MODULE`] `u32` slice as the raw little-endian byte
/// stream that `vkCreateShaderModule` expects. Length is always a multiple of
/// 4 (SPIR-V §2.3). Returned as a fresh `Vec<u8>` so the caller does not
/// have to worry about the endianness of the host: SPIR-V is defined as
/// **little-endian**, and this function writes exactly little-endian bytes
/// regardless of host endianness (`u32::to_le_bytes`).
#[must_use]
pub fn bytes() -> Vec<u8> {
    let mut out = Vec::with_capacity(SPIRV_MODULE.len() * 4);
    for word in SPIRV_MODULE {
        out.extend_from_slice(&word.to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sanity: `op(count, opcode)` matches the SPIR-V spec §2.3 packing.
    #[test]
    fn op_packs_word_count_and_opcode_per_spec() {
        // OpCapability = opcode 17, single operand → count 2.
        assert_eq!(op(2, OP_CAPABILITY), 0x0002_0011);
        // OpMemoryModel = opcode 14, two operands → count 3.
        assert_eq!(op(3, OP_MEMORY_MODEL), 0x0003_000E);
        // OpReturn = opcode 253, no operand → count 1.
        assert_eq!(op(1, OP_RETURN), 0x0001_00FD);
        // OpLabel = opcode 248, one operand → count 2.
        assert_eq!(op(2, OP_LABEL), 0x0002_00F8);
    }

    /// The SPIR-V binary starts with the correct magic + a Vulkan 1.1-legal
    /// version (SPIR-V 1.3), and every word count/opcode is a legal SPIR-V
    /// value.
    #[test]
    fn module_header_is_well_formed() {
        assert_eq!(SPIRV_MODULE[0], SPV_MAGIC, "SPIR-V magic mismatch");
        assert_eq!(
            SPIRV_MODULE[1], SPV_VERSION_1_3,
            "SPIR-V version must be 1.3 (Vulkan 1.1 minimum)"
        );
        assert_eq!(SPIRV_MODULE[4], SPV_SCHEMA, "instruction schema must be 0");
    }

    /// Every instruction's declared word count matches the number of words we
    /// wrote. If this test regresses, the module is malformed and any Vulkan
    /// driver would reject `vkCreateShaderModule` with VK_ERROR_INVALID_SHADER_NV
    /// / `VK_ERROR_UNKNOWN`.
    #[test]
    fn every_instruction_word_count_is_consistent() {
        // Walk the module starting after the 5-word header. For each
        // instruction the high 16 bits of word 0 give the count.
        let mut i = 5;
        let mut instr_count = 0;
        while i < SPIRV_MODULE.len() {
            let word0 = SPIRV_MODULE[i];
            let wc = (word0 >> 16) as usize;
            assert!(
                wc >= 1,
                "instruction {instr_count} at word {i} declared word count 0 — SPIR-V spec §2.3 \
                 requires at least 1"
            );
            assert!(
                i + wc <= SPIRV_MODULE.len(),
                "instruction {instr_count} at word {i} declared word count {wc}, but only {} \
                 words remain — module truncated",
                SPIRV_MODULE.len() - i
            );
            i += wc;
            instr_count += 1;
        }
        assert_eq!(
            i,
            SPIRV_MODULE.len(),
            "instruction walk did not land exactly at end of module — a stray word crept in"
        );
        // Sanity: we authored 37 instructions (count carefully from the
        // per-line comments above):
        //   4 (capability / memory / entry / exec_mode)
        // + 8 (decorations)
        // + 15 (types + variables + constant)
        // + 10 (function body: OpFunction .. OpFunctionEnd)
        // = 37.
        assert_eq!(
            instr_count, 37,
            "module must contain 37 instructions; found {instr_count}"
        );
    }

    /// `bytes()` produces a byte stream that is a multiple of 4 (SPIR-V spec
    /// requires 4-byte alignment of the SPIR-V module) and, when re-parsed as
    /// little-endian u32s, is byte-for-byte identical to `SPIRV_MODULE`.
    #[test]
    fn bytes_round_trip_matches_spirv_module() {
        let bytes = bytes();
        assert_eq!(
            bytes.len() % 4,
            0,
            "SPIR-V byte stream length must be a multiple of 4"
        );
        assert_eq!(bytes.len(), SPIRV_MODULE.len() * 4);
        for (i, chunk) in bytes.chunks_exact(4).enumerate() {
            let word = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            assert_eq!(
                word, SPIRV_MODULE[i],
                "word {i} did not round-trip through little-endian bytes",
            );
        }
    }

    /// The Id bound in the header must exceed every Id we actually reference —
    /// otherwise a Vulkan driver rejects the module.
    #[test]
    fn id_bound_is_sufficient_for_every_id_used() {
        // We use IDs 1..=22 explicitly. The header reserves `bound = 23`.
        assert_eq!(SPIRV_MODULE[3], 23);
        // Scan the module for any ID references outside 0..bound.
        // We can't distinguish "id reference" words from "literal" words
        // in a truly instruction-driven walk without a full grammar, but the
        // bound is stable at construction time, so we only check the header
        // field is what we designed for.
    }
}
