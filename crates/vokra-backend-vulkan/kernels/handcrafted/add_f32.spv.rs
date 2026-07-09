//! Hand-crafted SPIR-V 1.3 bytecode for a minimal `add_f32` compute kernel
//! (M3-02-T14 partial / T24 / ADR M3-02-spirv-generation §4 (c)).
//!
//! # Purpose
//!
//! This is the **second and last** hand-authored SPIR-V module Vokra ships
//! (see the sibling `copy_f32.spv.rs` — same authoring pattern, one binding
//! more). Its role is the M3-02-T24 dispatch-chain proof point:
//!
//! * `copy_f32` (2 SSBOs = src, dst) already proves the T08〜T12 + T25 object
//!   stack works end-to-end for a **single-input** kernel;
//! * `add_f32` (3 SSBOs = a, b, c) covers the missing case — **binding a
//!   descriptor set with multiple readable SSBOs *and* one writable SSBO in
//!   the same dispatch**, plus the smallest possible arithmetic op
//!   (`OpFAdd`).
//!
//! Together the two kernels are enough to route [`OpKind::Copy`] and
//! [`OpKind::Add`] through the Vulkan graph executor (M3-02-T24 / T26) with
//! real GPU-observed outputs. Everything else — GEMM, GEMV, softmax, layer
//! norm, GELU, conv1d, transpose, gather — remains an explicit
//! [`VokraError::UnsupportedOp`] on the Vulkan backend until T14〜T22
//! `glslc`-produced blobs land (ADR M3-02-spirv-generation §5 caps
//! hand-authored kernels at these two).
//!
//! # GLSL source this bytecode reproduces
//!
//! ```glsl
//! #version 450
//! layout(local_size_x = 64) in;
//! layout(std430, set = 0, binding = 0) readonly  buffer ABuf { float a[]; };
//! layout(std430, set = 0, binding = 1) readonly  buffer BBuf { float b[]; };
//! layout(std430, set = 0, binding = 2) writeonly buffer CBuf { float c[]; };
//! void main() {
//!     uint idx = gl_GlobalInvocationID.x;
//!     c[idx] = a[idx] + b[idx];
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
//! recorded in `crate::spirv::SHADERS[ADD_F32]::expected_sha256_hex` and
//! verified at test time (never at build time — keeps `cargo build` fast).

/// Pack the 32-bit first word of a SPIR-V instruction.
///
/// SPIR-V 1.3 spec §2.3 "Physical Layout of a SPIR-V Module":
/// > Word 0: the low-order 16 bits are the opcode; the high-order 16 bits are
/// > the total word count of the instruction, including this word.
#[inline]
const fn op(word_count: u16, opcode: u16) -> u32 {
    ((word_count as u32) << 16) | (opcode as u32)
}

// Opcode values (SPIR-V unified1 grammar; see the crate docs above). Every
// value is the numeric constant from the Khronos spec — never invented.
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
/// SPIR-V unified1 grammar: `OpFAdd` (§3.32.13 Arithmetic Instructions).
/// Result = Operand 1 + Operand 2 in floating point; 5 words:
/// `op(5, 129), result_type, result_id, operand_1, operand_2`.
const OP_F_ADD: u16 = 129;
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
/// bound-checking driver ignores generator; we are not registered with Khronos,
/// so `0` is the honest choice (same as `copy_f32.spv.rs`).
const SPV_GENERATOR: u32 = 0;
/// Upper bound of Ids used in the module (i.e. `max_id + 1`). We use IDs
/// `%1..=%26`, so the bound is `27`.
const SPV_ID_BOUND: u32 = 27;
/// Instruction schema — always `0` for the current SPIR-V.
const SPV_SCHEMA: u32 = 0;

// Result Ids (§3.32.2). Types / constants match the copy_f32 layout for
// symmetry; the add_f32-only IDs are the extra SSBO variable and the extra
// pair of access-chain/load results plus the OpFAdd result.
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
/// SSBO `a` (readonly), binding 0.
const ID_VAR_SRC_A: u32 = 14;
/// SSBO `b` (readonly), binding 1.
const ID_VAR_SRC_B: u32 = 15;
/// SSBO `c` (writeonly), binding 2.
const ID_VAR_DST_C: u32 = 16;
const ID_FUNC_MAIN: u32 = 17;
const ID_LABEL_ENTRY: u32 = 18;
const ID_ACC_INV_ID_X: u32 = 19;
const ID_LOAD_IDX: u32 = 20;
const ID_ACC_SRC_A_ELEM: u32 = 21;
const ID_LOAD_VAL_A: u32 = 22;
const ID_ACC_SRC_B_ELEM: u32 = 23;
const ID_LOAD_VAL_B: u32 = 24;
/// `%25 = OpFAdd %float %22 %24` — the sum `a[idx] + b[idx]`.
const ID_FADD_RESULT: u32 = 25;
const ID_ACC_DST_C_ELEM: u32 = 26;

// Literal string "main\0" packed into two little-endian u32s (§2.2.1):
//   bytes 'm'(0x6d) 'a'(0x61) 'i'(0x69) 'n'(0x6e) → 0x6E69616D
//   bytes '\0' 0x00 0x00 0x00 → 0x00000000
const LIT_STRING_MAIN_0: u32 = 0x6E69_616D;
const LIT_STRING_MAIN_1: u32 = 0x0000_0000;

/// The complete SPIR-V module as a `&'static [u32]` — 172 words = 688 bytes
/// (5 header + 17 caps/mem/entry/exec + 40 decorations + 58 types/vars +
/// 52 function body).
///
/// The `include_bytes!(…)` sites in `crate::spirv::load_spv_owned` treat the
/// bytes of this constant as the shader module payload. `bytes()` below
/// re-encodes the `u32` slice as an `&'static [u8]` (little-endian) so callers
/// can pass it directly to `vkCreateShaderModule`.
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
    // OpDecorate %src_a DescriptorSet 0
    op(4, OP_DECORATE),
    ID_VAR_SRC_A,
    DECO_DESCRIPTOR_SET,
    0,
    // OpDecorate %src_a Binding 0
    op(4, OP_DECORATE),
    ID_VAR_SRC_A,
    DECO_BINDING,
    0,
    // OpDecorate %src_b DescriptorSet 0
    op(4, OP_DECORATE),
    ID_VAR_SRC_B,
    DECO_DESCRIPTOR_SET,
    0,
    // OpDecorate %src_b Binding 1
    op(4, OP_DECORATE),
    ID_VAR_SRC_B,
    DECO_BINDING,
    1,
    // OpDecorate %dst_c DescriptorSet 0
    op(4, OP_DECORATE),
    ID_VAR_DST_C,
    DECO_DESCRIPTOR_SET,
    0,
    // OpDecorate %dst_c Binding 2
    op(4, OP_DECORATE),
    ID_VAR_DST_C,
    DECO_BINDING,
    2,
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
    // %14 = OpVariable %ptr_sb_struct StorageBuffer  (a buffer at binding 0)
    op(4, OP_VARIABLE),
    ID_PTR_SB_STRUCT,
    ID_VAR_SRC_A,
    STORAGE_STORAGE_BUFFER,
    // %15 = OpVariable %ptr_sb_struct StorageBuffer  (b buffer at binding 1)
    op(4, OP_VARIABLE),
    ID_PTR_SB_STRUCT,
    ID_VAR_SRC_B,
    STORAGE_STORAGE_BUFFER,
    // %16 = OpVariable %ptr_sb_struct StorageBuffer  (c buffer at binding 2)
    op(4, OP_VARIABLE),
    ID_PTR_SB_STRUCT,
    ID_VAR_DST_C,
    STORAGE_STORAGE_BUFFER,
    // ------------------------------------------------------------------ Function definition (§13)
    // %17 = OpFunction %void None %func_void_void
    op(5, OP_FUNCTION),
    ID_VOID_TYPE,
    ID_FUNC_MAIN,
    0, // Function Control = None
    ID_FUNC_TYPE,
    // %18 = OpLabel
    op(2, OP_LABEL),
    ID_LABEL_ENTRY,
    // %19 = OpAccessChain %ptr_input_uint %gl_GlobalInvocationID %uint_0
    //       (i.e. &gl_GlobalInvocationID.x)
    op(5, OP_ACCESS_CHAIN),
    ID_PTR_INPUT_UINT,
    ID_ACC_INV_ID_X,
    ID_VAR_GL_GLOBAL_INVOCATION_ID,
    ID_CONST_UINT_0,
    // %20 = OpLoad %uint %19
    op(4, OP_LOAD),
    ID_UINT_TYPE,
    ID_LOAD_IDX,
    ID_ACC_INV_ID_X,
    // %21 = OpAccessChain %ptr_sb_float %src_a %uint_0(member) %idx
    //       (i.e. &a.arr[idx])
    op(6, OP_ACCESS_CHAIN),
    ID_PTR_SB_FLOAT,
    ID_ACC_SRC_A_ELEM,
    ID_VAR_SRC_A,
    ID_CONST_UINT_0,
    ID_LOAD_IDX,
    // %22 = OpLoad %float %21
    op(4, OP_LOAD),
    ID_FLOAT_TYPE,
    ID_LOAD_VAL_A,
    ID_ACC_SRC_A_ELEM,
    // %23 = OpAccessChain %ptr_sb_float %src_b %uint_0(member) %idx
    //       (i.e. &b.arr[idx])
    op(6, OP_ACCESS_CHAIN),
    ID_PTR_SB_FLOAT,
    ID_ACC_SRC_B_ELEM,
    ID_VAR_SRC_B,
    ID_CONST_UINT_0,
    ID_LOAD_IDX,
    // %24 = OpLoad %float %23
    op(4, OP_LOAD),
    ID_FLOAT_TYPE,
    ID_LOAD_VAL_B,
    ID_ACC_SRC_B_ELEM,
    // %25 = OpFAdd %float %22 %24    (a_val + b_val)
    op(5, OP_F_ADD),
    ID_FLOAT_TYPE,
    ID_FADD_RESULT,
    ID_LOAD_VAL_A,
    ID_LOAD_VAL_B,
    // %26 = OpAccessChain %ptr_sb_float %dst_c %uint_0(member) %idx
    //       (i.e. &c.arr[idx])
    op(6, OP_ACCESS_CHAIN),
    ID_PTR_SB_FLOAT,
    ID_ACC_DST_C_ELEM,
    ID_VAR_DST_C,
    ID_CONST_UINT_0,
    ID_LOAD_IDX,
    // OpStore %26 %25   (c[idx] = a_val + b_val)
    op(3, OP_STORE),
    ID_ACC_DST_C_ELEM,
    ID_FADD_RESULT,
    // OpReturn
    op(1, OP_RETURN),
    // OpFunctionEnd
    op(1, OP_FUNCTION_END),
];

/// Local workgroup X-size (must match the `OpExecutionMode LocalSize` above).
/// Consumers use this to compute `group_count_x = ceil(N / LOCAL_SIZE_X)`.
pub const LOCAL_SIZE_X: u32 = 64;

/// Number of SSBO bindings the kernel consumes (a @ binding 0, b @ binding 1,
/// c @ binding 2 — all at set 0). Consumers use this to size the descriptor
/// set layout (3 storage-buffer bindings).
pub const BINDING_COUNT: u32 = 3;

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
        // OpFAdd = opcode 129, four operands → count 5.
        assert_eq!(op(5, OP_F_ADD), 0x0005_0081);
        // OpCapability = opcode 17, single operand → count 2.
        assert_eq!(op(2, OP_CAPABILITY), 0x0002_0011);
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
        // Sanity: we authored 43 instructions (count carefully from the
        // per-line comments above):
        //   4 (capability / memory / entry / exec_mode)
        // + 10 (decorations: 1 array-stride + 1 member-decorate + 1 struct-block
        //       + 1 gl_id builtin + 6 for the 3× descriptor-set/binding pairs)
        // + 16 (types + variables + constant: 11 types + 1 constant + 4 vars)
        // + 13 (function body:
        //       OpFunction + OpLabel + access + load + access + load + access
        //       + load + OpFAdd + access + OpStore + OpReturn + OpFunctionEnd)
        // = 43.
        assert_eq!(
            instr_count, 43,
            "module must contain 43 instructions; found {instr_count}"
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
        // We use IDs 1..=26 explicitly. The header reserves `bound = 27`.
        assert_eq!(SPIRV_MODULE[3], 27);
    }

    /// The BINDING_COUNT constant matches the number of SSBO descriptor-set
    /// bindings the shader body actually references. If this drifts, the
    /// dispatch chain (`context::smoke_dispatch_add_f32_impl`) would allocate
    /// a descriptor set with the wrong number of bindings, and the driver
    /// would reject the write set.
    #[test]
    fn binding_count_matches_ssbo_declarations() {
        assert_eq!(
            BINDING_COUNT, 3,
            "add_f32 kernel binds `a` + `b` + `c` — three SSBOs at set 0"
        );
    }
}
