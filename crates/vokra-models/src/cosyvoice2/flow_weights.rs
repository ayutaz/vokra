//! Authenticated, staged CosyVoice2 flow-component GGUF binder.
//!
//! This module intentionally stops at a strict header/schema binding seam.
//! It does not enable a public flow runtime, synthesize payloads, or make
//! CPU/Metal/parity/publication claims. Tensor names and shapes below are
//! generated from the authenticated component evidence, not inferred from a
//! model implementation.
//!
//! Flow tensor roots are accepted as a narrow staged contract. A future
//! composite converter must reserve or prefix these roots so unrelated
//! tokenizer/speaker tensors cannot collide; those namespaces are not
//! admitted here. LLM/HiFT names remain outside this component contract.

use std::collections::BTreeSet;

use vokra_core::gguf::{GgmlType, GgufFile, GgufMetadataValue, chunks};
use vokra_core::{Result, VokraError};

use crate::strict_checkpoint::sha256_bytes;

pub(crate) const FLOW_TENSOR_COUNT: usize = 1_121;
pub(crate) const FLOW_MANIFEST_SHA256: &str =
    "68abcdcdc961091b9c9e6456146ac9da06cda9f86e45a8faaac22a3b0b26217d";
pub(crate) const FLOW_ARTIFACT_FILE: &str = "flow.pt";
pub(crate) const FLOW_ARTIFACT_BYTES: u32 = 450_575_567;
pub(crate) const FLOW_ARTIFACT_SHA256: &str =
    "ff4c2f867674411e0a08cee702996df13fa67c1cd864c06108da88d16d088541";
pub(crate) const FLOW_UPSTREAM_HF: &str = "FunAudioLLM/CosyVoice2-0.5B";
pub(crate) const FLOW_UPSTREAM_REVISION: &str = "eec1ae6c79877dbd9379285cf8789c9e0879293d";
pub(crate) const FLOW_CONFIG_FILE: &str = "cosyvoice2.yaml";
pub(crate) const FLOW_CONFIG_BYTES: u32 = 7_330;
pub(crate) const FLOW_CONFIG_SHA256: &str =
    "0af2c0d010c477187c39f3e8fd5f1ae2e4e6f90ad03ba37c10ed6c6a87b05959";
pub(crate) const FLOW_CONFIG_GIT_BLOB_SHA1: &str = "bc19267bbfd373c9a760b7667a74349ddd487db1";
pub(crate) const FLOW_SOURCE_REPOSITORY: &str = "https://github.com/FunAudioLLM/CosyVoice.git";
pub(crate) const FLOW_SOURCE_REVISION: &str = "8555549e882236e6541748b1042d95693caa82ba";
pub(crate) const FLOW_SOURCE_LICENSE_FILE: &str = "LICENSE";
pub(crate) const FLOW_SOURCE_LICENSE_BYTES: u32 = 11_357;
pub(crate) const FLOW_SOURCE_LICENSE_SHA256: &str =
    "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4";
pub(crate) const FLOW_SOURCE_LICENSE_GIT_BLOB_SHA1: &str =
    "261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64";
pub(crate) const FLOW_SOURCE_LICENSE_DECLARED: &str = "Apache-2.0";
pub(crate) const FLOW_DATA_PICKLE_SHA256: &str =
    "d8977bfb852a57439b8e6ca0a656b69bf171c6f86e434bce2210d2c05f0a2448";
pub(crate) const FLOW_STORAGE_MANIFEST_SHA256: &str =
    "e2ec1a5009a0bf4f63eaebe82d4a1bc037f1cb26e82360f7d40c4c9700fc68ee";

const FLOW_ARCH: &str = "cosyvoice2";
const FLOW_COMPONENT: &str = "flow";
const FLOW_COMPOSITE_STATUS: &str = "INSPECTION_ONLY";
const KEY_ARCH: &str = chunks::KEY_MODEL_ARCH;
const KEY_COMPONENT: &str = "vokra.cosyvoice2_flow.component";
const KEY_COMPOSITE_STATUS: &str = "vokra.cosyvoice2.composite_status";
const KEY_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";
const KEY_UPSTREAM_REVISION: &str = "vokra.provenance.upstream_revision";
const KEY_CHECKPOINT_SHA256: &str = "vokra.provenance.checkpoint_sha256";
const KEY_COMPONENT_UPSTREAM_REVISION: &str = "vokra.cosyvoice2_flow.upstream_revision";
const KEY_CHECKPOINT_FILE: &str = "vokra.cosyvoice2_flow.checkpoint_file";
const KEY_CHECKPOINT_BYTES: &str = "vokra.cosyvoice2_flow.checkpoint_bytes";
const KEY_CHECKPOINT_COMPONENT_SHA256: &str = "vokra.cosyvoice2_flow.checkpoint_sha256";
const KEY_CONFIG_FILE: &str = "vokra.cosyvoice2_flow.config_file";
const KEY_CONFIG_BYTES: &str = "vokra.cosyvoice2_flow.config_bytes";
const KEY_CONFIG_SHA256: &str = "vokra.cosyvoice2_flow.config_sha256";
const KEY_CONFIG_GIT_BLOB_SHA1: &str = "vokra.cosyvoice2_flow.config_git_blob_sha1";
const KEY_SOURCE_REPOSITORY: &str = "vokra.cosyvoice2_flow.source_repo";
const KEY_SOURCE_REVISION: &str = "vokra.cosyvoice2_flow.source_revision";
const KEY_SOURCE_LICENSE_FILE: &str = "vokra.cosyvoice2_flow.source_license_file";
const KEY_SOURCE_LICENSE_BYTES: &str = "vokra.cosyvoice2_flow.source_license_bytes";
const KEY_SOURCE_LICENSE_SHA256: &str = "vokra.cosyvoice2_flow.source_license_sha256";
const KEY_SOURCE_LICENSE_GIT_BLOB_SHA1: &str = "vokra.cosyvoice2_flow.source_license_git_blob_sha1";
const KEY_SOURCE_LICENSE_DECLARED: &str = "vokra.cosyvoice2_flow.source_license_declared";
const KEY_MANIFEST_SHA256: &str = "vokra.cosyvoice2_flow.tensor_manifest_sha256";
const KEY_DATA_PICKLE_SHA256: &str = "vokra.cosyvoice2_flow.data_pickle_sha256";
const KEY_STORAGE_MANIFEST_SHA256: &str = "vokra.cosyvoice2_flow.storage_manifest_sha256";
const SOURCE_ROLES: &[(&str, &str, &str)] = &[
    (
        "cosyvoice/cli/cosyvoice.py",
        "8e44f0f0144378561a00ebc065fdb15a843bc4650e68683bebb6624827731859",
        "cc443bed44c651a47492fc7e2142e3a88fb47627",
    ),
    (
        "cosyvoice/flow/decoder.py",
        "ef5eceb9db7f63ddda1d5bca6bfa6b28b8ea11656c4b1f9c109d28f656cbbf29",
        "97768a459fbb89a2c99f98de302628d8ccafda67",
    ),
    (
        "cosyvoice/flow/flow.py",
        "a8497feb58336e7566b1f085d11acff9cb4f1a24949abd2c244fbf97c76f9b6d",
        "a068288f889aff4079b0c54c612897d31d08882a",
    ),
    (
        "cosyvoice/flow/flow_matching.py",
        "b1ad671fe37f872c034bde8f75cc19c1b88758d54e375fa2b54e14a088addfe6",
        "7f92df5d24690fe89fc548ab60f37483f91b03a6",
    ),
    (
        "cosyvoice/transformer/upsample_encoder.py",
        "a8003c212ce64697ce43001f776902ee60696a3b7e373935479029dccaf7d569",
        "6ffda6acad25cc0cfcf1bc07b9211c326ca8d49f",
    ),
];

pub(crate) const FLOW_MANIFEST_DIGEST: [u8; 32] = [
    0x68, 0xab, 0xcd, 0xcd, 0xc9, 0x61, 0x09, 0x1b, 0x9c, 0x9e, 0x45, 0x61, 0x46, 0xac, 0x9d, 0xa0,
    0x6c, 0xda, 0x9f, 0x86, 0xe4, 0x5a, 0x8f, 0xaa, 0xc2, 0x2a, 0x3b, 0x0b, 0x26, 0x21, 0x7d,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FlowTensorSpec {
    pub(crate) name: &'static str,
    pub(crate) shape: &'static [usize],
    pub(crate) dtype: GgmlType,
}

/// Complete authenticated flow tensor schema, in canonical name order.
pub(crate) static FLOW_SCHEMA: &[FlowTensorSpec] = &[
    FlowTensorSpec {
        name: "decoder.estimator.down_blocks.0.0.block1.block.0.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.down_blocks.0.0.block1.block.0.weight",
        shape: &[256, 320, 3],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.down_blocks.0.0.block1.block.2.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.down_blocks.0.0.block1.block.2.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.down_blocks.0.0.block2.block.0.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.down_blocks.0.0.block2.block.0.weight",
        shape: &[256, 256, 3],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.down_blocks.0.0.block2.block.2.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.down_blocks.0.0.block2.block.2.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.down_blocks.0.0.mlp.1.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.down_blocks.0.0.mlp.1.weight",
        shape: &[256, 1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.down_blocks.0.0.res_conv.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.down_blocks.0.0.res_conv.weight",
        shape: &[256, 320, 1],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.down_blocks.0.1.0.attn1.to_k.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.down_blocks.0.1.0.attn1.to_out.0.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.down_blocks.0.1.0.attn1.to_out.0.weight",
        shape: &[256, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.down_blocks.0.1.0.attn1.to_q.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.down_blocks.0.1.0.attn1.to_v.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.down_blocks.0.1.0.ff.net.0.proj.bias",
        shape: &[1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.down_blocks.0.1.0.ff.net.0.proj.weight",
        shape: &[1024, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.down_blocks.0.1.0.ff.net.2.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.down_blocks.0.1.0.ff.net.2.weight",
        shape: &[256, 1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.down_blocks.0.1.0.norm1.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.down_blocks.0.1.0.norm1.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.down_blocks.0.1.0.norm3.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.down_blocks.0.1.0.norm3.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.down_blocks.0.1.1.attn1.to_k.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.down_blocks.0.1.1.attn1.to_out.0.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.down_blocks.0.1.1.attn1.to_out.0.weight",
        shape: &[256, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.down_blocks.0.1.1.attn1.to_q.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.down_blocks.0.1.1.attn1.to_v.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.down_blocks.0.1.1.ff.net.0.proj.bias",
        shape: &[1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.down_blocks.0.1.1.ff.net.0.proj.weight",
        shape: &[1024, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.down_blocks.0.1.1.ff.net.2.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.down_blocks.0.1.1.ff.net.2.weight",
        shape: &[256, 1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.down_blocks.0.1.1.norm1.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.down_blocks.0.1.1.norm1.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.down_blocks.0.1.1.norm3.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.down_blocks.0.1.1.norm3.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.down_blocks.0.1.2.attn1.to_k.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.down_blocks.0.1.2.attn1.to_out.0.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.down_blocks.0.1.2.attn1.to_out.0.weight",
        shape: &[256, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.down_blocks.0.1.2.attn1.to_q.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.down_blocks.0.1.2.attn1.to_v.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.down_blocks.0.1.2.ff.net.0.proj.bias",
        shape: &[1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.down_blocks.0.1.2.ff.net.0.proj.weight",
        shape: &[1024, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.down_blocks.0.1.2.ff.net.2.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.down_blocks.0.1.2.ff.net.2.weight",
        shape: &[256, 1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.down_blocks.0.1.2.norm1.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.down_blocks.0.1.2.norm1.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.down_blocks.0.1.2.norm3.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.down_blocks.0.1.2.norm3.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.down_blocks.0.1.3.attn1.to_k.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.down_blocks.0.1.3.attn1.to_out.0.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.down_blocks.0.1.3.attn1.to_out.0.weight",
        shape: &[256, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.down_blocks.0.1.3.attn1.to_q.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.down_blocks.0.1.3.attn1.to_v.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.down_blocks.0.1.3.ff.net.0.proj.bias",
        shape: &[1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.down_blocks.0.1.3.ff.net.0.proj.weight",
        shape: &[1024, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.down_blocks.0.1.3.ff.net.2.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.down_blocks.0.1.3.ff.net.2.weight",
        shape: &[256, 1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.down_blocks.0.1.3.norm1.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.down_blocks.0.1.3.norm1.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.down_blocks.0.1.3.norm3.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.down_blocks.0.1.3.norm3.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.down_blocks.0.2.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.down_blocks.0.2.weight",
        shape: &[256, 256, 3],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.final_block.block.0.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.final_block.block.0.weight",
        shape: &[256, 256, 3],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.final_block.block.2.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.final_block.block.2.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.final_proj.bias",
        shape: &[80],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.final_proj.weight",
        shape: &[80, 256, 1],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.0.0.block1.block.0.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.0.0.block1.block.0.weight",
        shape: &[256, 256, 3],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.0.0.block1.block.2.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.0.0.block1.block.2.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.0.0.block2.block.0.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.0.0.block2.block.0.weight",
        shape: &[256, 256, 3],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.0.0.block2.block.2.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.0.0.block2.block.2.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.0.0.mlp.1.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.0.0.mlp.1.weight",
        shape: &[256, 1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.0.0.res_conv.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.0.0.res_conv.weight",
        shape: &[256, 256, 1],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.0.1.0.attn1.to_k.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.0.1.0.attn1.to_out.0.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.0.1.0.attn1.to_out.0.weight",
        shape: &[256, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.0.1.0.attn1.to_q.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.0.1.0.attn1.to_v.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.0.1.0.ff.net.0.proj.bias",
        shape: &[1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.0.1.0.ff.net.0.proj.weight",
        shape: &[1024, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.0.1.0.ff.net.2.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.0.1.0.ff.net.2.weight",
        shape: &[256, 1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.0.1.0.norm1.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.0.1.0.norm1.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.0.1.0.norm3.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.0.1.0.norm3.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.0.1.1.attn1.to_k.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.0.1.1.attn1.to_out.0.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.0.1.1.attn1.to_out.0.weight",
        shape: &[256, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.0.1.1.attn1.to_q.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.0.1.1.attn1.to_v.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.0.1.1.ff.net.0.proj.bias",
        shape: &[1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.0.1.1.ff.net.0.proj.weight",
        shape: &[1024, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.0.1.1.ff.net.2.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.0.1.1.ff.net.2.weight",
        shape: &[256, 1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.0.1.1.norm1.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.0.1.1.norm1.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.0.1.1.norm3.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.0.1.1.norm3.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.0.1.2.attn1.to_k.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.0.1.2.attn1.to_out.0.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.0.1.2.attn1.to_out.0.weight",
        shape: &[256, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.0.1.2.attn1.to_q.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.0.1.2.attn1.to_v.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.0.1.2.ff.net.0.proj.bias",
        shape: &[1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.0.1.2.ff.net.0.proj.weight",
        shape: &[1024, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.0.1.2.ff.net.2.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.0.1.2.ff.net.2.weight",
        shape: &[256, 1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.0.1.2.norm1.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.0.1.2.norm1.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.0.1.2.norm3.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.0.1.2.norm3.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.0.1.3.attn1.to_k.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.0.1.3.attn1.to_out.0.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.0.1.3.attn1.to_out.0.weight",
        shape: &[256, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.0.1.3.attn1.to_q.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.0.1.3.attn1.to_v.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.0.1.3.ff.net.0.proj.bias",
        shape: &[1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.0.1.3.ff.net.0.proj.weight",
        shape: &[1024, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.0.1.3.ff.net.2.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.0.1.3.ff.net.2.weight",
        shape: &[256, 1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.0.1.3.norm1.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.0.1.3.norm1.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.0.1.3.norm3.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.0.1.3.norm3.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.1.0.block1.block.0.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.1.0.block1.block.0.weight",
        shape: &[256, 256, 3],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.1.0.block1.block.2.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.1.0.block1.block.2.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.1.0.block2.block.0.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.1.0.block2.block.0.weight",
        shape: &[256, 256, 3],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.1.0.block2.block.2.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.1.0.block2.block.2.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.1.0.mlp.1.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.1.0.mlp.1.weight",
        shape: &[256, 1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.1.0.res_conv.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.1.0.res_conv.weight",
        shape: &[256, 256, 1],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.1.1.0.attn1.to_k.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.1.1.0.attn1.to_out.0.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.1.1.0.attn1.to_out.0.weight",
        shape: &[256, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.1.1.0.attn1.to_q.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.1.1.0.attn1.to_v.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.1.1.0.ff.net.0.proj.bias",
        shape: &[1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.1.1.0.ff.net.0.proj.weight",
        shape: &[1024, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.1.1.0.ff.net.2.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.1.1.0.ff.net.2.weight",
        shape: &[256, 1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.1.1.0.norm1.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.1.1.0.norm1.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.1.1.0.norm3.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.1.1.0.norm3.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.1.1.1.attn1.to_k.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.1.1.1.attn1.to_out.0.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.1.1.1.attn1.to_out.0.weight",
        shape: &[256, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.1.1.1.attn1.to_q.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.1.1.1.attn1.to_v.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.1.1.1.ff.net.0.proj.bias",
        shape: &[1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.1.1.1.ff.net.0.proj.weight",
        shape: &[1024, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.1.1.1.ff.net.2.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.1.1.1.ff.net.2.weight",
        shape: &[256, 1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.1.1.1.norm1.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.1.1.1.norm1.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.1.1.1.norm3.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.1.1.1.norm3.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.1.1.2.attn1.to_k.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.1.1.2.attn1.to_out.0.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.1.1.2.attn1.to_out.0.weight",
        shape: &[256, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.1.1.2.attn1.to_q.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.1.1.2.attn1.to_v.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.1.1.2.ff.net.0.proj.bias",
        shape: &[1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.1.1.2.ff.net.0.proj.weight",
        shape: &[1024, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.1.1.2.ff.net.2.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.1.1.2.ff.net.2.weight",
        shape: &[256, 1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.1.1.2.norm1.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.1.1.2.norm1.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.1.1.2.norm3.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.1.1.2.norm3.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.1.1.3.attn1.to_k.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.1.1.3.attn1.to_out.0.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.1.1.3.attn1.to_out.0.weight",
        shape: &[256, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.1.1.3.attn1.to_q.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.1.1.3.attn1.to_v.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.1.1.3.ff.net.0.proj.bias",
        shape: &[1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.1.1.3.ff.net.0.proj.weight",
        shape: &[1024, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.1.1.3.ff.net.2.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.1.1.3.ff.net.2.weight",
        shape: &[256, 1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.1.1.3.norm1.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.1.1.3.norm1.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.1.1.3.norm3.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.1.1.3.norm3.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.10.0.block1.block.0.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.10.0.block1.block.0.weight",
        shape: &[256, 256, 3],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.10.0.block1.block.2.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.10.0.block1.block.2.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.10.0.block2.block.0.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.10.0.block2.block.0.weight",
        shape: &[256, 256, 3],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.10.0.block2.block.2.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.10.0.block2.block.2.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.10.0.mlp.1.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.10.0.mlp.1.weight",
        shape: &[256, 1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.10.0.res_conv.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.10.0.res_conv.weight",
        shape: &[256, 256, 1],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.10.1.0.attn1.to_k.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.10.1.0.attn1.to_out.0.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.10.1.0.attn1.to_out.0.weight",
        shape: &[256, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.10.1.0.attn1.to_q.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.10.1.0.attn1.to_v.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.10.1.0.ff.net.0.proj.bias",
        shape: &[1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.10.1.0.ff.net.0.proj.weight",
        shape: &[1024, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.10.1.0.ff.net.2.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.10.1.0.ff.net.2.weight",
        shape: &[256, 1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.10.1.0.norm1.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.10.1.0.norm1.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.10.1.0.norm3.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.10.1.0.norm3.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.10.1.1.attn1.to_k.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.10.1.1.attn1.to_out.0.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.10.1.1.attn1.to_out.0.weight",
        shape: &[256, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.10.1.1.attn1.to_q.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.10.1.1.attn1.to_v.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.10.1.1.ff.net.0.proj.bias",
        shape: &[1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.10.1.1.ff.net.0.proj.weight",
        shape: &[1024, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.10.1.1.ff.net.2.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.10.1.1.ff.net.2.weight",
        shape: &[256, 1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.10.1.1.norm1.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.10.1.1.norm1.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.10.1.1.norm3.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.10.1.1.norm3.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.10.1.2.attn1.to_k.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.10.1.2.attn1.to_out.0.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.10.1.2.attn1.to_out.0.weight",
        shape: &[256, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.10.1.2.attn1.to_q.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.10.1.2.attn1.to_v.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.10.1.2.ff.net.0.proj.bias",
        shape: &[1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.10.1.2.ff.net.0.proj.weight",
        shape: &[1024, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.10.1.2.ff.net.2.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.10.1.2.ff.net.2.weight",
        shape: &[256, 1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.10.1.2.norm1.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.10.1.2.norm1.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.10.1.2.norm3.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.10.1.2.norm3.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.10.1.3.attn1.to_k.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.10.1.3.attn1.to_out.0.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.10.1.3.attn1.to_out.0.weight",
        shape: &[256, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.10.1.3.attn1.to_q.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.10.1.3.attn1.to_v.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.10.1.3.ff.net.0.proj.bias",
        shape: &[1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.10.1.3.ff.net.0.proj.weight",
        shape: &[1024, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.10.1.3.ff.net.2.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.10.1.3.ff.net.2.weight",
        shape: &[256, 1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.10.1.3.norm1.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.10.1.3.norm1.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.10.1.3.norm3.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.10.1.3.norm3.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.11.0.block1.block.0.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.11.0.block1.block.0.weight",
        shape: &[256, 256, 3],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.11.0.block1.block.2.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.11.0.block1.block.2.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.11.0.block2.block.0.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.11.0.block2.block.0.weight",
        shape: &[256, 256, 3],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.11.0.block2.block.2.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.11.0.block2.block.2.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.11.0.mlp.1.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.11.0.mlp.1.weight",
        shape: &[256, 1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.11.0.res_conv.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.11.0.res_conv.weight",
        shape: &[256, 256, 1],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.11.1.0.attn1.to_k.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.11.1.0.attn1.to_out.0.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.11.1.0.attn1.to_out.0.weight",
        shape: &[256, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.11.1.0.attn1.to_q.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.11.1.0.attn1.to_v.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.11.1.0.ff.net.0.proj.bias",
        shape: &[1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.11.1.0.ff.net.0.proj.weight",
        shape: &[1024, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.11.1.0.ff.net.2.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.11.1.0.ff.net.2.weight",
        shape: &[256, 1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.11.1.0.norm1.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.11.1.0.norm1.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.11.1.0.norm3.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.11.1.0.norm3.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.11.1.1.attn1.to_k.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.11.1.1.attn1.to_out.0.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.11.1.1.attn1.to_out.0.weight",
        shape: &[256, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.11.1.1.attn1.to_q.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.11.1.1.attn1.to_v.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.11.1.1.ff.net.0.proj.bias",
        shape: &[1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.11.1.1.ff.net.0.proj.weight",
        shape: &[1024, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.11.1.1.ff.net.2.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.11.1.1.ff.net.2.weight",
        shape: &[256, 1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.11.1.1.norm1.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.11.1.1.norm1.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.11.1.1.norm3.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.11.1.1.norm3.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.11.1.2.attn1.to_k.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.11.1.2.attn1.to_out.0.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.11.1.2.attn1.to_out.0.weight",
        shape: &[256, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.11.1.2.attn1.to_q.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.11.1.2.attn1.to_v.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.11.1.2.ff.net.0.proj.bias",
        shape: &[1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.11.1.2.ff.net.0.proj.weight",
        shape: &[1024, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.11.1.2.ff.net.2.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.11.1.2.ff.net.2.weight",
        shape: &[256, 1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.11.1.2.norm1.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.11.1.2.norm1.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.11.1.2.norm3.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.11.1.2.norm3.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.11.1.3.attn1.to_k.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.11.1.3.attn1.to_out.0.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.11.1.3.attn1.to_out.0.weight",
        shape: &[256, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.11.1.3.attn1.to_q.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.11.1.3.attn1.to_v.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.11.1.3.ff.net.0.proj.bias",
        shape: &[1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.11.1.3.ff.net.0.proj.weight",
        shape: &[1024, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.11.1.3.ff.net.2.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.11.1.3.ff.net.2.weight",
        shape: &[256, 1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.11.1.3.norm1.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.11.1.3.norm1.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.11.1.3.norm3.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.11.1.3.norm3.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.2.0.block1.block.0.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.2.0.block1.block.0.weight",
        shape: &[256, 256, 3],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.2.0.block1.block.2.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.2.0.block1.block.2.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.2.0.block2.block.0.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.2.0.block2.block.0.weight",
        shape: &[256, 256, 3],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.2.0.block2.block.2.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.2.0.block2.block.2.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.2.0.mlp.1.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.2.0.mlp.1.weight",
        shape: &[256, 1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.2.0.res_conv.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.2.0.res_conv.weight",
        shape: &[256, 256, 1],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.2.1.0.attn1.to_k.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.2.1.0.attn1.to_out.0.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.2.1.0.attn1.to_out.0.weight",
        shape: &[256, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.2.1.0.attn1.to_q.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.2.1.0.attn1.to_v.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.2.1.0.ff.net.0.proj.bias",
        shape: &[1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.2.1.0.ff.net.0.proj.weight",
        shape: &[1024, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.2.1.0.ff.net.2.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.2.1.0.ff.net.2.weight",
        shape: &[256, 1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.2.1.0.norm1.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.2.1.0.norm1.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.2.1.0.norm3.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.2.1.0.norm3.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.2.1.1.attn1.to_k.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.2.1.1.attn1.to_out.0.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.2.1.1.attn1.to_out.0.weight",
        shape: &[256, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.2.1.1.attn1.to_q.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.2.1.1.attn1.to_v.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.2.1.1.ff.net.0.proj.bias",
        shape: &[1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.2.1.1.ff.net.0.proj.weight",
        shape: &[1024, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.2.1.1.ff.net.2.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.2.1.1.ff.net.2.weight",
        shape: &[256, 1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.2.1.1.norm1.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.2.1.1.norm1.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.2.1.1.norm3.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.2.1.1.norm3.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.2.1.2.attn1.to_k.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.2.1.2.attn1.to_out.0.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.2.1.2.attn1.to_out.0.weight",
        shape: &[256, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.2.1.2.attn1.to_q.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.2.1.2.attn1.to_v.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.2.1.2.ff.net.0.proj.bias",
        shape: &[1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.2.1.2.ff.net.0.proj.weight",
        shape: &[1024, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.2.1.2.ff.net.2.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.2.1.2.ff.net.2.weight",
        shape: &[256, 1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.2.1.2.norm1.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.2.1.2.norm1.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.2.1.2.norm3.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.2.1.2.norm3.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.2.1.3.attn1.to_k.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.2.1.3.attn1.to_out.0.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.2.1.3.attn1.to_out.0.weight",
        shape: &[256, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.2.1.3.attn1.to_q.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.2.1.3.attn1.to_v.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.2.1.3.ff.net.0.proj.bias",
        shape: &[1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.2.1.3.ff.net.0.proj.weight",
        shape: &[1024, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.2.1.3.ff.net.2.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.2.1.3.ff.net.2.weight",
        shape: &[256, 1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.2.1.3.norm1.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.2.1.3.norm1.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.2.1.3.norm3.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.2.1.3.norm3.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.3.0.block1.block.0.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.3.0.block1.block.0.weight",
        shape: &[256, 256, 3],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.3.0.block1.block.2.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.3.0.block1.block.2.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.3.0.block2.block.0.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.3.0.block2.block.0.weight",
        shape: &[256, 256, 3],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.3.0.block2.block.2.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.3.0.block2.block.2.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.3.0.mlp.1.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.3.0.mlp.1.weight",
        shape: &[256, 1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.3.0.res_conv.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.3.0.res_conv.weight",
        shape: &[256, 256, 1],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.3.1.0.attn1.to_k.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.3.1.0.attn1.to_out.0.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.3.1.0.attn1.to_out.0.weight",
        shape: &[256, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.3.1.0.attn1.to_q.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.3.1.0.attn1.to_v.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.3.1.0.ff.net.0.proj.bias",
        shape: &[1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.3.1.0.ff.net.0.proj.weight",
        shape: &[1024, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.3.1.0.ff.net.2.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.3.1.0.ff.net.2.weight",
        shape: &[256, 1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.3.1.0.norm1.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.3.1.0.norm1.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.3.1.0.norm3.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.3.1.0.norm3.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.3.1.1.attn1.to_k.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.3.1.1.attn1.to_out.0.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.3.1.1.attn1.to_out.0.weight",
        shape: &[256, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.3.1.1.attn1.to_q.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.3.1.1.attn1.to_v.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.3.1.1.ff.net.0.proj.bias",
        shape: &[1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.3.1.1.ff.net.0.proj.weight",
        shape: &[1024, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.3.1.1.ff.net.2.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.3.1.1.ff.net.2.weight",
        shape: &[256, 1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.3.1.1.norm1.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.3.1.1.norm1.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.3.1.1.norm3.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.3.1.1.norm3.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.3.1.2.attn1.to_k.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.3.1.2.attn1.to_out.0.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.3.1.2.attn1.to_out.0.weight",
        shape: &[256, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.3.1.2.attn1.to_q.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.3.1.2.attn1.to_v.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.3.1.2.ff.net.0.proj.bias",
        shape: &[1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.3.1.2.ff.net.0.proj.weight",
        shape: &[1024, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.3.1.2.ff.net.2.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.3.1.2.ff.net.2.weight",
        shape: &[256, 1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.3.1.2.norm1.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.3.1.2.norm1.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.3.1.2.norm3.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.3.1.2.norm3.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.3.1.3.attn1.to_k.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.3.1.3.attn1.to_out.0.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.3.1.3.attn1.to_out.0.weight",
        shape: &[256, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.3.1.3.attn1.to_q.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.3.1.3.attn1.to_v.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.3.1.3.ff.net.0.proj.bias",
        shape: &[1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.3.1.3.ff.net.0.proj.weight",
        shape: &[1024, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.3.1.3.ff.net.2.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.3.1.3.ff.net.2.weight",
        shape: &[256, 1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.3.1.3.norm1.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.3.1.3.norm1.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.3.1.3.norm3.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.3.1.3.norm3.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.4.0.block1.block.0.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.4.0.block1.block.0.weight",
        shape: &[256, 256, 3],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.4.0.block1.block.2.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.4.0.block1.block.2.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.4.0.block2.block.0.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.4.0.block2.block.0.weight",
        shape: &[256, 256, 3],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.4.0.block2.block.2.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.4.0.block2.block.2.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.4.0.mlp.1.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.4.0.mlp.1.weight",
        shape: &[256, 1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.4.0.res_conv.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.4.0.res_conv.weight",
        shape: &[256, 256, 1],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.4.1.0.attn1.to_k.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.4.1.0.attn1.to_out.0.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.4.1.0.attn1.to_out.0.weight",
        shape: &[256, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.4.1.0.attn1.to_q.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.4.1.0.attn1.to_v.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.4.1.0.ff.net.0.proj.bias",
        shape: &[1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.4.1.0.ff.net.0.proj.weight",
        shape: &[1024, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.4.1.0.ff.net.2.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.4.1.0.ff.net.2.weight",
        shape: &[256, 1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.4.1.0.norm1.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.4.1.0.norm1.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.4.1.0.norm3.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.4.1.0.norm3.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.4.1.1.attn1.to_k.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.4.1.1.attn1.to_out.0.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.4.1.1.attn1.to_out.0.weight",
        shape: &[256, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.4.1.1.attn1.to_q.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.4.1.1.attn1.to_v.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.4.1.1.ff.net.0.proj.bias",
        shape: &[1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.4.1.1.ff.net.0.proj.weight",
        shape: &[1024, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.4.1.1.ff.net.2.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.4.1.1.ff.net.2.weight",
        shape: &[256, 1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.4.1.1.norm1.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.4.1.1.norm1.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.4.1.1.norm3.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.4.1.1.norm3.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.4.1.2.attn1.to_k.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.4.1.2.attn1.to_out.0.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.4.1.2.attn1.to_out.0.weight",
        shape: &[256, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.4.1.2.attn1.to_q.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.4.1.2.attn1.to_v.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.4.1.2.ff.net.0.proj.bias",
        shape: &[1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.4.1.2.ff.net.0.proj.weight",
        shape: &[1024, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.4.1.2.ff.net.2.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.4.1.2.ff.net.2.weight",
        shape: &[256, 1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.4.1.2.norm1.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.4.1.2.norm1.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.4.1.2.norm3.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.4.1.2.norm3.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.4.1.3.attn1.to_k.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.4.1.3.attn1.to_out.0.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.4.1.3.attn1.to_out.0.weight",
        shape: &[256, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.4.1.3.attn1.to_q.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.4.1.3.attn1.to_v.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.4.1.3.ff.net.0.proj.bias",
        shape: &[1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.4.1.3.ff.net.0.proj.weight",
        shape: &[1024, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.4.1.3.ff.net.2.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.4.1.3.ff.net.2.weight",
        shape: &[256, 1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.4.1.3.norm1.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.4.1.3.norm1.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.4.1.3.norm3.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.4.1.3.norm3.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.5.0.block1.block.0.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.5.0.block1.block.0.weight",
        shape: &[256, 256, 3],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.5.0.block1.block.2.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.5.0.block1.block.2.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.5.0.block2.block.0.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.5.0.block2.block.0.weight",
        shape: &[256, 256, 3],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.5.0.block2.block.2.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.5.0.block2.block.2.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.5.0.mlp.1.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.5.0.mlp.1.weight",
        shape: &[256, 1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.5.0.res_conv.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.5.0.res_conv.weight",
        shape: &[256, 256, 1],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.5.1.0.attn1.to_k.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.5.1.0.attn1.to_out.0.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.5.1.0.attn1.to_out.0.weight",
        shape: &[256, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.5.1.0.attn1.to_q.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.5.1.0.attn1.to_v.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.5.1.0.ff.net.0.proj.bias",
        shape: &[1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.5.1.0.ff.net.0.proj.weight",
        shape: &[1024, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.5.1.0.ff.net.2.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.5.1.0.ff.net.2.weight",
        shape: &[256, 1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.5.1.0.norm1.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.5.1.0.norm1.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.5.1.0.norm3.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.5.1.0.norm3.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.5.1.1.attn1.to_k.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.5.1.1.attn1.to_out.0.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.5.1.1.attn1.to_out.0.weight",
        shape: &[256, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.5.1.1.attn1.to_q.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.5.1.1.attn1.to_v.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.5.1.1.ff.net.0.proj.bias",
        shape: &[1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.5.1.1.ff.net.0.proj.weight",
        shape: &[1024, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.5.1.1.ff.net.2.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.5.1.1.ff.net.2.weight",
        shape: &[256, 1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.5.1.1.norm1.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.5.1.1.norm1.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.5.1.1.norm3.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.5.1.1.norm3.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.5.1.2.attn1.to_k.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.5.1.2.attn1.to_out.0.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.5.1.2.attn1.to_out.0.weight",
        shape: &[256, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.5.1.2.attn1.to_q.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.5.1.2.attn1.to_v.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.5.1.2.ff.net.0.proj.bias",
        shape: &[1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.5.1.2.ff.net.0.proj.weight",
        shape: &[1024, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.5.1.2.ff.net.2.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.5.1.2.ff.net.2.weight",
        shape: &[256, 1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.5.1.2.norm1.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.5.1.2.norm1.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.5.1.2.norm3.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.5.1.2.norm3.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.5.1.3.attn1.to_k.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.5.1.3.attn1.to_out.0.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.5.1.3.attn1.to_out.0.weight",
        shape: &[256, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.5.1.3.attn1.to_q.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.5.1.3.attn1.to_v.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.5.1.3.ff.net.0.proj.bias",
        shape: &[1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.5.1.3.ff.net.0.proj.weight",
        shape: &[1024, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.5.1.3.ff.net.2.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.5.1.3.ff.net.2.weight",
        shape: &[256, 1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.5.1.3.norm1.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.5.1.3.norm1.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.5.1.3.norm3.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.5.1.3.norm3.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.6.0.block1.block.0.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.6.0.block1.block.0.weight",
        shape: &[256, 256, 3],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.6.0.block1.block.2.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.6.0.block1.block.2.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.6.0.block2.block.0.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.6.0.block2.block.0.weight",
        shape: &[256, 256, 3],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.6.0.block2.block.2.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.6.0.block2.block.2.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.6.0.mlp.1.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.6.0.mlp.1.weight",
        shape: &[256, 1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.6.0.res_conv.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.6.0.res_conv.weight",
        shape: &[256, 256, 1],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.6.1.0.attn1.to_k.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.6.1.0.attn1.to_out.0.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.6.1.0.attn1.to_out.0.weight",
        shape: &[256, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.6.1.0.attn1.to_q.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.6.1.0.attn1.to_v.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.6.1.0.ff.net.0.proj.bias",
        shape: &[1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.6.1.0.ff.net.0.proj.weight",
        shape: &[1024, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.6.1.0.ff.net.2.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.6.1.0.ff.net.2.weight",
        shape: &[256, 1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.6.1.0.norm1.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.6.1.0.norm1.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.6.1.0.norm3.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.6.1.0.norm3.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.6.1.1.attn1.to_k.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.6.1.1.attn1.to_out.0.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.6.1.1.attn1.to_out.0.weight",
        shape: &[256, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.6.1.1.attn1.to_q.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.6.1.1.attn1.to_v.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.6.1.1.ff.net.0.proj.bias",
        shape: &[1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.6.1.1.ff.net.0.proj.weight",
        shape: &[1024, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.6.1.1.ff.net.2.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.6.1.1.ff.net.2.weight",
        shape: &[256, 1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.6.1.1.norm1.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.6.1.1.norm1.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.6.1.1.norm3.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.6.1.1.norm3.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.6.1.2.attn1.to_k.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.6.1.2.attn1.to_out.0.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.6.1.2.attn1.to_out.0.weight",
        shape: &[256, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.6.1.2.attn1.to_q.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.6.1.2.attn1.to_v.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.6.1.2.ff.net.0.proj.bias",
        shape: &[1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.6.1.2.ff.net.0.proj.weight",
        shape: &[1024, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.6.1.2.ff.net.2.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.6.1.2.ff.net.2.weight",
        shape: &[256, 1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.6.1.2.norm1.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.6.1.2.norm1.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.6.1.2.norm3.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.6.1.2.norm3.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.6.1.3.attn1.to_k.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.6.1.3.attn1.to_out.0.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.6.1.3.attn1.to_out.0.weight",
        shape: &[256, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.6.1.3.attn1.to_q.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.6.1.3.attn1.to_v.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.6.1.3.ff.net.0.proj.bias",
        shape: &[1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.6.1.3.ff.net.0.proj.weight",
        shape: &[1024, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.6.1.3.ff.net.2.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.6.1.3.ff.net.2.weight",
        shape: &[256, 1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.6.1.3.norm1.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.6.1.3.norm1.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.6.1.3.norm3.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.6.1.3.norm3.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.7.0.block1.block.0.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.7.0.block1.block.0.weight",
        shape: &[256, 256, 3],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.7.0.block1.block.2.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.7.0.block1.block.2.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.7.0.block2.block.0.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.7.0.block2.block.0.weight",
        shape: &[256, 256, 3],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.7.0.block2.block.2.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.7.0.block2.block.2.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.7.0.mlp.1.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.7.0.mlp.1.weight",
        shape: &[256, 1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.7.0.res_conv.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.7.0.res_conv.weight",
        shape: &[256, 256, 1],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.7.1.0.attn1.to_k.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.7.1.0.attn1.to_out.0.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.7.1.0.attn1.to_out.0.weight",
        shape: &[256, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.7.1.0.attn1.to_q.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.7.1.0.attn1.to_v.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.7.1.0.ff.net.0.proj.bias",
        shape: &[1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.7.1.0.ff.net.0.proj.weight",
        shape: &[1024, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.7.1.0.ff.net.2.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.7.1.0.ff.net.2.weight",
        shape: &[256, 1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.7.1.0.norm1.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.7.1.0.norm1.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.7.1.0.norm3.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.7.1.0.norm3.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.7.1.1.attn1.to_k.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.7.1.1.attn1.to_out.0.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.7.1.1.attn1.to_out.0.weight",
        shape: &[256, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.7.1.1.attn1.to_q.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.7.1.1.attn1.to_v.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.7.1.1.ff.net.0.proj.bias",
        shape: &[1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.7.1.1.ff.net.0.proj.weight",
        shape: &[1024, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.7.1.1.ff.net.2.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.7.1.1.ff.net.2.weight",
        shape: &[256, 1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.7.1.1.norm1.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.7.1.1.norm1.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.7.1.1.norm3.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.7.1.1.norm3.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.7.1.2.attn1.to_k.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.7.1.2.attn1.to_out.0.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.7.1.2.attn1.to_out.0.weight",
        shape: &[256, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.7.1.2.attn1.to_q.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.7.1.2.attn1.to_v.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.7.1.2.ff.net.0.proj.bias",
        shape: &[1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.7.1.2.ff.net.0.proj.weight",
        shape: &[1024, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.7.1.2.ff.net.2.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.7.1.2.ff.net.2.weight",
        shape: &[256, 1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.7.1.2.norm1.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.7.1.2.norm1.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.7.1.2.norm3.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.7.1.2.norm3.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.7.1.3.attn1.to_k.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.7.1.3.attn1.to_out.0.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.7.1.3.attn1.to_out.0.weight",
        shape: &[256, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.7.1.3.attn1.to_q.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.7.1.3.attn1.to_v.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.7.1.3.ff.net.0.proj.bias",
        shape: &[1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.7.1.3.ff.net.0.proj.weight",
        shape: &[1024, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.7.1.3.ff.net.2.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.7.1.3.ff.net.2.weight",
        shape: &[256, 1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.7.1.3.norm1.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.7.1.3.norm1.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.7.1.3.norm3.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.7.1.3.norm3.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.8.0.block1.block.0.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.8.0.block1.block.0.weight",
        shape: &[256, 256, 3],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.8.0.block1.block.2.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.8.0.block1.block.2.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.8.0.block2.block.0.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.8.0.block2.block.0.weight",
        shape: &[256, 256, 3],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.8.0.block2.block.2.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.8.0.block2.block.2.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.8.0.mlp.1.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.8.0.mlp.1.weight",
        shape: &[256, 1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.8.0.res_conv.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.8.0.res_conv.weight",
        shape: &[256, 256, 1],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.8.1.0.attn1.to_k.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.8.1.0.attn1.to_out.0.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.8.1.0.attn1.to_out.0.weight",
        shape: &[256, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.8.1.0.attn1.to_q.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.8.1.0.attn1.to_v.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.8.1.0.ff.net.0.proj.bias",
        shape: &[1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.8.1.0.ff.net.0.proj.weight",
        shape: &[1024, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.8.1.0.ff.net.2.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.8.1.0.ff.net.2.weight",
        shape: &[256, 1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.8.1.0.norm1.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.8.1.0.norm1.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.8.1.0.norm3.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.8.1.0.norm3.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.8.1.1.attn1.to_k.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.8.1.1.attn1.to_out.0.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.8.1.1.attn1.to_out.0.weight",
        shape: &[256, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.8.1.1.attn1.to_q.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.8.1.1.attn1.to_v.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.8.1.1.ff.net.0.proj.bias",
        shape: &[1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.8.1.1.ff.net.0.proj.weight",
        shape: &[1024, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.8.1.1.ff.net.2.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.8.1.1.ff.net.2.weight",
        shape: &[256, 1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.8.1.1.norm1.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.8.1.1.norm1.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.8.1.1.norm3.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.8.1.1.norm3.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.8.1.2.attn1.to_k.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.8.1.2.attn1.to_out.0.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.8.1.2.attn1.to_out.0.weight",
        shape: &[256, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.8.1.2.attn1.to_q.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.8.1.2.attn1.to_v.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.8.1.2.ff.net.0.proj.bias",
        shape: &[1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.8.1.2.ff.net.0.proj.weight",
        shape: &[1024, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.8.1.2.ff.net.2.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.8.1.2.ff.net.2.weight",
        shape: &[256, 1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.8.1.2.norm1.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.8.1.2.norm1.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.8.1.2.norm3.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.8.1.2.norm3.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.8.1.3.attn1.to_k.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.8.1.3.attn1.to_out.0.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.8.1.3.attn1.to_out.0.weight",
        shape: &[256, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.8.1.3.attn1.to_q.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.8.1.3.attn1.to_v.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.8.1.3.ff.net.0.proj.bias",
        shape: &[1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.8.1.3.ff.net.0.proj.weight",
        shape: &[1024, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.8.1.3.ff.net.2.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.8.1.3.ff.net.2.weight",
        shape: &[256, 1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.8.1.3.norm1.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.8.1.3.norm1.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.8.1.3.norm3.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.8.1.3.norm3.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.9.0.block1.block.0.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.9.0.block1.block.0.weight",
        shape: &[256, 256, 3],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.9.0.block1.block.2.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.9.0.block1.block.2.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.9.0.block2.block.0.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.9.0.block2.block.0.weight",
        shape: &[256, 256, 3],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.9.0.block2.block.2.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.9.0.block2.block.2.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.9.0.mlp.1.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.9.0.mlp.1.weight",
        shape: &[256, 1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.9.0.res_conv.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.9.0.res_conv.weight",
        shape: &[256, 256, 1],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.9.1.0.attn1.to_k.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.9.1.0.attn1.to_out.0.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.9.1.0.attn1.to_out.0.weight",
        shape: &[256, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.9.1.0.attn1.to_q.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.9.1.0.attn1.to_v.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.9.1.0.ff.net.0.proj.bias",
        shape: &[1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.9.1.0.ff.net.0.proj.weight",
        shape: &[1024, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.9.1.0.ff.net.2.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.9.1.0.ff.net.2.weight",
        shape: &[256, 1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.9.1.0.norm1.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.9.1.0.norm1.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.9.1.0.norm3.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.9.1.0.norm3.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.9.1.1.attn1.to_k.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.9.1.1.attn1.to_out.0.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.9.1.1.attn1.to_out.0.weight",
        shape: &[256, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.9.1.1.attn1.to_q.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.9.1.1.attn1.to_v.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.9.1.1.ff.net.0.proj.bias",
        shape: &[1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.9.1.1.ff.net.0.proj.weight",
        shape: &[1024, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.9.1.1.ff.net.2.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.9.1.1.ff.net.2.weight",
        shape: &[256, 1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.9.1.1.norm1.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.9.1.1.norm1.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.9.1.1.norm3.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.9.1.1.norm3.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.9.1.2.attn1.to_k.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.9.1.2.attn1.to_out.0.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.9.1.2.attn1.to_out.0.weight",
        shape: &[256, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.9.1.2.attn1.to_q.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.9.1.2.attn1.to_v.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.9.1.2.ff.net.0.proj.bias",
        shape: &[1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.9.1.2.ff.net.0.proj.weight",
        shape: &[1024, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.9.1.2.ff.net.2.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.9.1.2.ff.net.2.weight",
        shape: &[256, 1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.9.1.2.norm1.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.9.1.2.norm1.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.9.1.2.norm3.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.9.1.2.norm3.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.9.1.3.attn1.to_k.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.9.1.3.attn1.to_out.0.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.9.1.3.attn1.to_out.0.weight",
        shape: &[256, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.9.1.3.attn1.to_q.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.9.1.3.attn1.to_v.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.9.1.3.ff.net.0.proj.bias",
        shape: &[1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.9.1.3.ff.net.0.proj.weight",
        shape: &[1024, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.9.1.3.ff.net.2.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.9.1.3.ff.net.2.weight",
        shape: &[256, 1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.9.1.3.norm1.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.9.1.3.norm1.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.9.1.3.norm3.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.mid_blocks.9.1.3.norm3.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.time_mlp.linear_1.bias",
        shape: &[1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.time_mlp.linear_1.weight",
        shape: &[1024, 320],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.time_mlp.linear_2.bias",
        shape: &[1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.time_mlp.linear_2.weight",
        shape: &[1024, 1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.up_blocks.0.0.block1.block.0.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.up_blocks.0.0.block1.block.0.weight",
        shape: &[256, 512, 3],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.up_blocks.0.0.block1.block.2.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.up_blocks.0.0.block1.block.2.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.up_blocks.0.0.block2.block.0.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.up_blocks.0.0.block2.block.0.weight",
        shape: &[256, 256, 3],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.up_blocks.0.0.block2.block.2.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.up_blocks.0.0.block2.block.2.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.up_blocks.0.0.mlp.1.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.up_blocks.0.0.mlp.1.weight",
        shape: &[256, 1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.up_blocks.0.0.res_conv.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.up_blocks.0.0.res_conv.weight",
        shape: &[256, 512, 1],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.up_blocks.0.1.0.attn1.to_k.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.up_blocks.0.1.0.attn1.to_out.0.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.up_blocks.0.1.0.attn1.to_out.0.weight",
        shape: &[256, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.up_blocks.0.1.0.attn1.to_q.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.up_blocks.0.1.0.attn1.to_v.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.up_blocks.0.1.0.ff.net.0.proj.bias",
        shape: &[1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.up_blocks.0.1.0.ff.net.0.proj.weight",
        shape: &[1024, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.up_blocks.0.1.0.ff.net.2.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.up_blocks.0.1.0.ff.net.2.weight",
        shape: &[256, 1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.up_blocks.0.1.0.norm1.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.up_blocks.0.1.0.norm1.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.up_blocks.0.1.0.norm3.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.up_blocks.0.1.0.norm3.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.up_blocks.0.1.1.attn1.to_k.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.up_blocks.0.1.1.attn1.to_out.0.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.up_blocks.0.1.1.attn1.to_out.0.weight",
        shape: &[256, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.up_blocks.0.1.1.attn1.to_q.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.up_blocks.0.1.1.attn1.to_v.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.up_blocks.0.1.1.ff.net.0.proj.bias",
        shape: &[1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.up_blocks.0.1.1.ff.net.0.proj.weight",
        shape: &[1024, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.up_blocks.0.1.1.ff.net.2.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.up_blocks.0.1.1.ff.net.2.weight",
        shape: &[256, 1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.up_blocks.0.1.1.norm1.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.up_blocks.0.1.1.norm1.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.up_blocks.0.1.1.norm3.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.up_blocks.0.1.1.norm3.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.up_blocks.0.1.2.attn1.to_k.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.up_blocks.0.1.2.attn1.to_out.0.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.up_blocks.0.1.2.attn1.to_out.0.weight",
        shape: &[256, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.up_blocks.0.1.2.attn1.to_q.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.up_blocks.0.1.2.attn1.to_v.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.up_blocks.0.1.2.ff.net.0.proj.bias",
        shape: &[1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.up_blocks.0.1.2.ff.net.0.proj.weight",
        shape: &[1024, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.up_blocks.0.1.2.ff.net.2.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.up_blocks.0.1.2.ff.net.2.weight",
        shape: &[256, 1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.up_blocks.0.1.2.norm1.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.up_blocks.0.1.2.norm1.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.up_blocks.0.1.2.norm3.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.up_blocks.0.1.2.norm3.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.up_blocks.0.1.3.attn1.to_k.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.up_blocks.0.1.3.attn1.to_out.0.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.up_blocks.0.1.3.attn1.to_out.0.weight",
        shape: &[256, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.up_blocks.0.1.3.attn1.to_q.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.up_blocks.0.1.3.attn1.to_v.weight",
        shape: &[512, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.up_blocks.0.1.3.ff.net.0.proj.bias",
        shape: &[1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.up_blocks.0.1.3.ff.net.0.proj.weight",
        shape: &[1024, 256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.up_blocks.0.1.3.ff.net.2.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.up_blocks.0.1.3.ff.net.2.weight",
        shape: &[256, 1024],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.up_blocks.0.1.3.norm1.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.up_blocks.0.1.3.norm1.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.up_blocks.0.1.3.norm3.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.up_blocks.0.1.3.norm3.weight",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.up_blocks.0.2.bias",
        shape: &[256],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "decoder.estimator.up_blocks.0.2.weight",
        shape: &[256, 256, 3],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.after_norm.bias",
        shape: &[512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.after_norm.weight",
        shape: &[512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.embed.out.0.bias",
        shape: &[512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.embed.out.0.weight",
        shape: &[512, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.embed.out.1.bias",
        shape: &[512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.embed.out.1.weight",
        shape: &[512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.encoders.0.feed_forward.w_1.bias",
        shape: &[2048],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.encoders.0.feed_forward.w_1.weight",
        shape: &[2048, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.encoders.0.feed_forward.w_2.bias",
        shape: &[512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.encoders.0.feed_forward.w_2.weight",
        shape: &[512, 2048],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.encoders.0.norm_ff.bias",
        shape: &[512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.encoders.0.norm_ff.weight",
        shape: &[512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.encoders.0.norm_mha.bias",
        shape: &[512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.encoders.0.norm_mha.weight",
        shape: &[512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.encoders.0.self_attn.linear_k.bias",
        shape: &[512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.encoders.0.self_attn.linear_k.weight",
        shape: &[512, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.encoders.0.self_attn.linear_out.bias",
        shape: &[512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.encoders.0.self_attn.linear_out.weight",
        shape: &[512, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.encoders.0.self_attn.linear_pos.weight",
        shape: &[512, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.encoders.0.self_attn.linear_q.bias",
        shape: &[512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.encoders.0.self_attn.linear_q.weight",
        shape: &[512, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.encoders.0.self_attn.linear_v.bias",
        shape: &[512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.encoders.0.self_attn.linear_v.weight",
        shape: &[512, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.encoders.0.self_attn.pos_bias_u",
        shape: &[8, 64],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.encoders.0.self_attn.pos_bias_v",
        shape: &[8, 64],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.encoders.1.feed_forward.w_1.bias",
        shape: &[2048],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.encoders.1.feed_forward.w_1.weight",
        shape: &[2048, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.encoders.1.feed_forward.w_2.bias",
        shape: &[512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.encoders.1.feed_forward.w_2.weight",
        shape: &[512, 2048],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.encoders.1.norm_ff.bias",
        shape: &[512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.encoders.1.norm_ff.weight",
        shape: &[512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.encoders.1.norm_mha.bias",
        shape: &[512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.encoders.1.norm_mha.weight",
        shape: &[512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.encoders.1.self_attn.linear_k.bias",
        shape: &[512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.encoders.1.self_attn.linear_k.weight",
        shape: &[512, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.encoders.1.self_attn.linear_out.bias",
        shape: &[512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.encoders.1.self_attn.linear_out.weight",
        shape: &[512, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.encoders.1.self_attn.linear_pos.weight",
        shape: &[512, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.encoders.1.self_attn.linear_q.bias",
        shape: &[512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.encoders.1.self_attn.linear_q.weight",
        shape: &[512, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.encoders.1.self_attn.linear_v.bias",
        shape: &[512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.encoders.1.self_attn.linear_v.weight",
        shape: &[512, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.encoders.1.self_attn.pos_bias_u",
        shape: &[8, 64],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.encoders.1.self_attn.pos_bias_v",
        shape: &[8, 64],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.encoders.2.feed_forward.w_1.bias",
        shape: &[2048],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.encoders.2.feed_forward.w_1.weight",
        shape: &[2048, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.encoders.2.feed_forward.w_2.bias",
        shape: &[512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.encoders.2.feed_forward.w_2.weight",
        shape: &[512, 2048],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.encoders.2.norm_ff.bias",
        shape: &[512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.encoders.2.norm_ff.weight",
        shape: &[512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.encoders.2.norm_mha.bias",
        shape: &[512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.encoders.2.norm_mha.weight",
        shape: &[512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.encoders.2.self_attn.linear_k.bias",
        shape: &[512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.encoders.2.self_attn.linear_k.weight",
        shape: &[512, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.encoders.2.self_attn.linear_out.bias",
        shape: &[512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.encoders.2.self_attn.linear_out.weight",
        shape: &[512, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.encoders.2.self_attn.linear_pos.weight",
        shape: &[512, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.encoders.2.self_attn.linear_q.bias",
        shape: &[512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.encoders.2.self_attn.linear_q.weight",
        shape: &[512, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.encoders.2.self_attn.linear_v.bias",
        shape: &[512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.encoders.2.self_attn.linear_v.weight",
        shape: &[512, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.encoders.2.self_attn.pos_bias_u",
        shape: &[8, 64],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.encoders.2.self_attn.pos_bias_v",
        shape: &[8, 64],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.encoders.3.feed_forward.w_1.bias",
        shape: &[2048],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.encoders.3.feed_forward.w_1.weight",
        shape: &[2048, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.encoders.3.feed_forward.w_2.bias",
        shape: &[512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.encoders.3.feed_forward.w_2.weight",
        shape: &[512, 2048],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.encoders.3.norm_ff.bias",
        shape: &[512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.encoders.3.norm_ff.weight",
        shape: &[512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.encoders.3.norm_mha.bias",
        shape: &[512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.encoders.3.norm_mha.weight",
        shape: &[512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.encoders.3.self_attn.linear_k.bias",
        shape: &[512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.encoders.3.self_attn.linear_k.weight",
        shape: &[512, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.encoders.3.self_attn.linear_out.bias",
        shape: &[512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.encoders.3.self_attn.linear_out.weight",
        shape: &[512, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.encoders.3.self_attn.linear_pos.weight",
        shape: &[512, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.encoders.3.self_attn.linear_q.bias",
        shape: &[512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.encoders.3.self_attn.linear_q.weight",
        shape: &[512, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.encoders.3.self_attn.linear_v.bias",
        shape: &[512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.encoders.3.self_attn.linear_v.weight",
        shape: &[512, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.encoders.3.self_attn.pos_bias_u",
        shape: &[8, 64],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.encoders.3.self_attn.pos_bias_v",
        shape: &[8, 64],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.encoders.4.feed_forward.w_1.bias",
        shape: &[2048],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.encoders.4.feed_forward.w_1.weight",
        shape: &[2048, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.encoders.4.feed_forward.w_2.bias",
        shape: &[512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.encoders.4.feed_forward.w_2.weight",
        shape: &[512, 2048],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.encoders.4.norm_ff.bias",
        shape: &[512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.encoders.4.norm_ff.weight",
        shape: &[512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.encoders.4.norm_mha.bias",
        shape: &[512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.encoders.4.norm_mha.weight",
        shape: &[512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.encoders.4.self_attn.linear_k.bias",
        shape: &[512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.encoders.4.self_attn.linear_k.weight",
        shape: &[512, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.encoders.4.self_attn.linear_out.bias",
        shape: &[512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.encoders.4.self_attn.linear_out.weight",
        shape: &[512, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.encoders.4.self_attn.linear_pos.weight",
        shape: &[512, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.encoders.4.self_attn.linear_q.bias",
        shape: &[512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.encoders.4.self_attn.linear_q.weight",
        shape: &[512, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.encoders.4.self_attn.linear_v.bias",
        shape: &[512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.encoders.4.self_attn.linear_v.weight",
        shape: &[512, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.encoders.4.self_attn.pos_bias_u",
        shape: &[8, 64],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.encoders.4.self_attn.pos_bias_v",
        shape: &[8, 64],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.encoders.5.feed_forward.w_1.bias",
        shape: &[2048],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.encoders.5.feed_forward.w_1.weight",
        shape: &[2048, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.encoders.5.feed_forward.w_2.bias",
        shape: &[512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.encoders.5.feed_forward.w_2.weight",
        shape: &[512, 2048],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.encoders.5.norm_ff.bias",
        shape: &[512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.encoders.5.norm_ff.weight",
        shape: &[512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.encoders.5.norm_mha.bias",
        shape: &[512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.encoders.5.norm_mha.weight",
        shape: &[512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.encoders.5.self_attn.linear_k.bias",
        shape: &[512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.encoders.5.self_attn.linear_k.weight",
        shape: &[512, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.encoders.5.self_attn.linear_out.bias",
        shape: &[512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.encoders.5.self_attn.linear_out.weight",
        shape: &[512, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.encoders.5.self_attn.linear_pos.weight",
        shape: &[512, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.encoders.5.self_attn.linear_q.bias",
        shape: &[512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.encoders.5.self_attn.linear_q.weight",
        shape: &[512, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.encoders.5.self_attn.linear_v.bias",
        shape: &[512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.encoders.5.self_attn.linear_v.weight",
        shape: &[512, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.encoders.5.self_attn.pos_bias_u",
        shape: &[8, 64],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.encoders.5.self_attn.pos_bias_v",
        shape: &[8, 64],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.pre_lookahead_layer.conv1.bias",
        shape: &[512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.pre_lookahead_layer.conv1.weight",
        shape: &[512, 512, 4],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.pre_lookahead_layer.conv2.bias",
        shape: &[512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.pre_lookahead_layer.conv2.weight",
        shape: &[512, 512, 3],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.up_embed.out.0.bias",
        shape: &[512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.up_embed.out.0.weight",
        shape: &[512, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.up_embed.out.1.bias",
        shape: &[512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.up_embed.out.1.weight",
        shape: &[512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.up_encoders.0.feed_forward.w_1.bias",
        shape: &[2048],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.up_encoders.0.feed_forward.w_1.weight",
        shape: &[2048, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.up_encoders.0.feed_forward.w_2.bias",
        shape: &[512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.up_encoders.0.feed_forward.w_2.weight",
        shape: &[512, 2048],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.up_encoders.0.norm_ff.bias",
        shape: &[512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.up_encoders.0.norm_ff.weight",
        shape: &[512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.up_encoders.0.norm_mha.bias",
        shape: &[512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.up_encoders.0.norm_mha.weight",
        shape: &[512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.up_encoders.0.self_attn.linear_k.bias",
        shape: &[512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.up_encoders.0.self_attn.linear_k.weight",
        shape: &[512, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.up_encoders.0.self_attn.linear_out.bias",
        shape: &[512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.up_encoders.0.self_attn.linear_out.weight",
        shape: &[512, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.up_encoders.0.self_attn.linear_pos.weight",
        shape: &[512, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.up_encoders.0.self_attn.linear_q.bias",
        shape: &[512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.up_encoders.0.self_attn.linear_q.weight",
        shape: &[512, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.up_encoders.0.self_attn.linear_v.bias",
        shape: &[512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.up_encoders.0.self_attn.linear_v.weight",
        shape: &[512, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.up_encoders.0.self_attn.pos_bias_u",
        shape: &[8, 64],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.up_encoders.0.self_attn.pos_bias_v",
        shape: &[8, 64],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.up_encoders.1.feed_forward.w_1.bias",
        shape: &[2048],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.up_encoders.1.feed_forward.w_1.weight",
        shape: &[2048, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.up_encoders.1.feed_forward.w_2.bias",
        shape: &[512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.up_encoders.1.feed_forward.w_2.weight",
        shape: &[512, 2048],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.up_encoders.1.norm_ff.bias",
        shape: &[512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.up_encoders.1.norm_ff.weight",
        shape: &[512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.up_encoders.1.norm_mha.bias",
        shape: &[512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.up_encoders.1.norm_mha.weight",
        shape: &[512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.up_encoders.1.self_attn.linear_k.bias",
        shape: &[512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.up_encoders.1.self_attn.linear_k.weight",
        shape: &[512, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.up_encoders.1.self_attn.linear_out.bias",
        shape: &[512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.up_encoders.1.self_attn.linear_out.weight",
        shape: &[512, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.up_encoders.1.self_attn.linear_pos.weight",
        shape: &[512, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.up_encoders.1.self_attn.linear_q.bias",
        shape: &[512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.up_encoders.1.self_attn.linear_q.weight",
        shape: &[512, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.up_encoders.1.self_attn.linear_v.bias",
        shape: &[512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.up_encoders.1.self_attn.linear_v.weight",
        shape: &[512, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.up_encoders.1.self_attn.pos_bias_u",
        shape: &[8, 64],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.up_encoders.1.self_attn.pos_bias_v",
        shape: &[8, 64],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.up_encoders.2.feed_forward.w_1.bias",
        shape: &[2048],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.up_encoders.2.feed_forward.w_1.weight",
        shape: &[2048, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.up_encoders.2.feed_forward.w_2.bias",
        shape: &[512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.up_encoders.2.feed_forward.w_2.weight",
        shape: &[512, 2048],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.up_encoders.2.norm_ff.bias",
        shape: &[512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.up_encoders.2.norm_ff.weight",
        shape: &[512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.up_encoders.2.norm_mha.bias",
        shape: &[512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.up_encoders.2.norm_mha.weight",
        shape: &[512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.up_encoders.2.self_attn.linear_k.bias",
        shape: &[512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.up_encoders.2.self_attn.linear_k.weight",
        shape: &[512, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.up_encoders.2.self_attn.linear_out.bias",
        shape: &[512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.up_encoders.2.self_attn.linear_out.weight",
        shape: &[512, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.up_encoders.2.self_attn.linear_pos.weight",
        shape: &[512, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.up_encoders.2.self_attn.linear_q.bias",
        shape: &[512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.up_encoders.2.self_attn.linear_q.weight",
        shape: &[512, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.up_encoders.2.self_attn.linear_v.bias",
        shape: &[512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.up_encoders.2.self_attn.linear_v.weight",
        shape: &[512, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.up_encoders.2.self_attn.pos_bias_u",
        shape: &[8, 64],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.up_encoders.2.self_attn.pos_bias_v",
        shape: &[8, 64],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.up_encoders.3.feed_forward.w_1.bias",
        shape: &[2048],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.up_encoders.3.feed_forward.w_1.weight",
        shape: &[2048, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.up_encoders.3.feed_forward.w_2.bias",
        shape: &[512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.up_encoders.3.feed_forward.w_2.weight",
        shape: &[512, 2048],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.up_encoders.3.norm_ff.bias",
        shape: &[512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.up_encoders.3.norm_ff.weight",
        shape: &[512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.up_encoders.3.norm_mha.bias",
        shape: &[512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.up_encoders.3.norm_mha.weight",
        shape: &[512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.up_encoders.3.self_attn.linear_k.bias",
        shape: &[512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.up_encoders.3.self_attn.linear_k.weight",
        shape: &[512, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.up_encoders.3.self_attn.linear_out.bias",
        shape: &[512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.up_encoders.3.self_attn.linear_out.weight",
        shape: &[512, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.up_encoders.3.self_attn.linear_pos.weight",
        shape: &[512, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.up_encoders.3.self_attn.linear_q.bias",
        shape: &[512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.up_encoders.3.self_attn.linear_q.weight",
        shape: &[512, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.up_encoders.3.self_attn.linear_v.bias",
        shape: &[512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.up_encoders.3.self_attn.linear_v.weight",
        shape: &[512, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.up_encoders.3.self_attn.pos_bias_u",
        shape: &[8, 64],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.up_encoders.3.self_attn.pos_bias_v",
        shape: &[8, 64],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.up_layer.conv.bias",
        shape: &[512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder.up_layer.conv.weight",
        shape: &[512, 512, 5],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder_proj.bias",
        shape: &[80],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "encoder_proj.weight",
        shape: &[80, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "input_embedding.weight",
        shape: &[6561, 512],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "spk_embed_affine_layer.bias",
        shape: &[80],
        dtype: GgmlType::F32,
    },
    FlowTensorSpec {
        name: "spk_embed_affine_layer.weight",
        shape: &[80, 192],
        dtype: GgmlType::F32,
    },
];

#[derive(Debug)]
pub(crate) struct FlowWeights<'a> {
    file: &'a GgufFile,
}

impl<'a> FlowWeights<'a> {
    /// Binds only an authenticated flow component header. No tensor payload is
    /// decoded during binding; callers explicitly load each checked tensor.
    pub(crate) fn bind(file: &'a GgufFile) -> Result<Self> {
        validate_metadata(file)?;
        validate_tensor_schema(file)?;
        Ok(Self { file })
    }

    /// Decode one expected F32 tensor after the complete schema has bound.
    /// Composite LLM/HiFT namespaces are never accepted by this lookup.
    pub(crate) fn tensor(&self, name: &str) -> Result<Vec<f32>> {
        let spec = FLOW_SCHEMA
            .iter()
            .find(|spec| spec.name == name)
            .ok_or_else(|| {
                VokraError::ModelLoad(format!(
                    "cosyvoice2_flow: tensor {name} is outside the authenticated schema"
                ))
            })?;
        let info = self.file.tensor_info(name).ok_or_else(|| {
            VokraError::ModelLoad(format!(
                "cosyvoice2_flow: required tensor {name} is missing"
            ))
        })?;
        if info.dtype != spec.dtype
            || info
                .dimensions
                .iter()
                .copied()
                .any(|d| d > usize::MAX as u64)
            || info
                .dimensions
                .iter()
                .map(|&d| d as usize)
                .collect::<Vec<_>>()
                != spec.shape
        {
            return Err(VokraError::ModelLoad(format!(
                "cosyvoice2_flow: tensor {name} has dtype {:?}, shape {:?}; expected {:?}, {:?}",
                info.dtype, info.dimensions, spec.dtype, spec.shape
            )));
        }
        let expected = spec
            .shape
            .iter()
            .try_fold(1usize, |count, &dim| count.checked_mul(dim))
            .ok_or_else(|| {
                VokraError::ModelLoad(format!(
                    "cosyvoice2_flow: tensor {name} element count overflows usize"
                ))
            })?;
        let values = self.file.tensor_f32(name).map_err(|error| {
            VokraError::ModelLoad(format!(
                "cosyvoice2_flow: tensor {name} decode failed: {error}"
            ))
        })?;
        if values.len() != expected {
            return Err(VokraError::ModelLoad(format!(
                "cosyvoice2_flow: tensor {name} decoded {} values, expected {expected}",
                values.len()
            )));
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err(VokraError::ModelLoad(format!(
                "cosyvoice2_flow: tensor {name} contains non-finite values"
            )));
        }
        Ok(values)
    }

    pub(crate) fn file(&self) -> &'a GgufFile {
        self.file
    }
}

fn require_string(file: &GgufFile, key: &str, expected: &str) -> Result<()> {
    match file.get(key).and_then(GgufMetadataValue::as_str) {
        Some(value) if value == expected => Ok(()),
        Some(value) => Err(VokraError::ModelLoad(format!(
            "cosyvoice2_flow: metadata {key}={value:?}, expected {expected:?}"
        ))),
        None => Err(VokraError::ModelLoad(format!(
            "cosyvoice2_flow: missing/non-string metadata {key}"
        ))),
    }
}

fn require_u32(file: &GgufFile, key: &str, expected: u32) -> Result<()> {
    match file.get(key) {
        Some(GgufMetadataValue::U32(value)) if *value == expected => Ok(()),
        Some(value) => Err(VokraError::ModelLoad(format!(
            "cosyvoice2_flow: metadata {key}={value:?}, expected U32({expected})"
        ))),
        None => Err(VokraError::ModelLoad(format!(
            "cosyvoice2_flow: missing/non-u32 metadata {key}"
        ))),
    }
}

fn validate_metadata(file: &GgufFile) -> Result<()> {
    for (key, expected) in [
        (KEY_ARCH, FLOW_ARCH),
        (KEY_COMPONENT, FLOW_COMPONENT),
        (KEY_COMPOSITE_STATUS, FLOW_COMPOSITE_STATUS),
        (KEY_UPSTREAM_HF, FLOW_UPSTREAM_HF),
        (KEY_UPSTREAM_REVISION, FLOW_UPSTREAM_REVISION),
        (KEY_COMPONENT_UPSTREAM_REVISION, FLOW_UPSTREAM_REVISION),
        (KEY_CHECKPOINT_SHA256, FLOW_ARTIFACT_SHA256),
        (KEY_CHECKPOINT_FILE, FLOW_ARTIFACT_FILE),
        (KEY_CHECKPOINT_COMPONENT_SHA256, FLOW_ARTIFACT_SHA256),
        (KEY_CONFIG_FILE, FLOW_CONFIG_FILE),
        (KEY_CONFIG_SHA256, FLOW_CONFIG_SHA256),
        (KEY_CONFIG_GIT_BLOB_SHA1, FLOW_CONFIG_GIT_BLOB_SHA1),
        (KEY_SOURCE_REPOSITORY, FLOW_SOURCE_REPOSITORY),
        (KEY_SOURCE_REVISION, FLOW_SOURCE_REVISION),
        (KEY_SOURCE_LICENSE_FILE, FLOW_SOURCE_LICENSE_FILE),
        (KEY_SOURCE_LICENSE_SHA256, FLOW_SOURCE_LICENSE_SHA256),
        (
            KEY_SOURCE_LICENSE_GIT_BLOB_SHA1,
            FLOW_SOURCE_LICENSE_GIT_BLOB_SHA1,
        ),
        (KEY_SOURCE_LICENSE_DECLARED, FLOW_SOURCE_LICENSE_DECLARED),
        (KEY_MANIFEST_SHA256, FLOW_MANIFEST_SHA256),
        (KEY_DATA_PICKLE_SHA256, FLOW_DATA_PICKLE_SHA256),
        (KEY_STORAGE_MANIFEST_SHA256, FLOW_STORAGE_MANIFEST_SHA256),
    ] {
        require_string(file, key, expected)?;
    }
    for (path, sha256, blob) in SOURCE_ROLES {
        require_string(
            file,
            &format!("vokra.cosyvoice2_flow.source.{path}.sha256"),
            sha256,
        )?;
        require_string(
            file,
            &format!("vokra.cosyvoice2_flow.source.{path}.git_blob_sha1"),
            blob,
        )?;
    }
    require_u32(file, KEY_SOURCE_LICENSE_BYTES, FLOW_SOURCE_LICENSE_BYTES)?;
    require_u32(file, KEY_CHECKPOINT_BYTES, FLOW_ARTIFACT_BYTES)?;
    require_u32(file, KEY_CONFIG_BYTES, FLOW_CONFIG_BYTES)?;
    Ok(())
}

fn is_flow_tensor_name(name: &str) -> bool {
    [
        "decoder.",
        "encoder.",
        "encoder_proj.",
        "input_embedding.",
        "spk_embed_affine_layer.",
    ]
    .iter()
    .any(|prefix| name.starts_with(prefix))
}

fn schema_spec(name: &str) -> Option<&'static FlowTensorSpec> {
    FLOW_SCHEMA.iter().find(|spec| spec.name == name)
}

fn validate_tensor_schema(file: &GgufFile) -> Result<()> {
    if FLOW_SCHEMA.len() != FLOW_TENSOR_COUNT {
        return Err(VokraError::ModelLoad(format!(
            "cosyvoice2_flow: compiled schema count {}, expected {FLOW_TENSOR_COUNT}",
            FLOW_SCHEMA.len()
        )));
    }
    if canonical_manifest_digest() != FLOW_MANIFEST_DIGEST {
        return Err(VokraError::ModelLoad(
            "cosyvoice2_flow: compiled schema digest does not match authenticated evidence"
                .to_owned(),
        ));
    }
    let mut expected_names = BTreeSet::new();
    for spec in FLOW_SCHEMA {
        if !expected_names.insert(spec.name) {
            return Err(VokraError::ModelLoad(format!(
                "cosyvoice2_flow: duplicate compiled schema name {}",
                spec.name
            )));
        }
    }
    let mut seen = BTreeSet::new();
    for info in file.tensors() {
        if !is_flow_tensor_name(&info.name) {
            continue;
        }
        if !seen.insert(info.name.as_str()) {
            return Err(VokraError::ModelLoad(format!(
                "cosyvoice2_flow: duplicate tensor {}",
                info.name
            )));
        }
        let spec = schema_spec(&info.name).ok_or_else(|| {
            VokraError::ModelLoad(format!(
                "cosyvoice2_flow: unexpected flow tensor {}",
                info.name
            ))
        })?;
        let shape_matches = info.dimensions.len() == spec.shape.len()
            && info
                .dimensions
                .iter()
                .zip(spec.shape)
                .all(|(&actual, &expected)| actual == expected as u64);
        if info.dtype != spec.dtype || !shape_matches {
            return Err(VokraError::ModelLoad(format!(
                "cosyvoice2_flow: tensor {} dtype {:?}, shape {:?}; expected {:?}, {:?}",
                info.name, info.dtype, info.dimensions, spec.dtype, spec.shape
            )));
        }
    }
    if seen.len() != FLOW_TENSOR_COUNT {
        let missing = FLOW_SCHEMA
            .iter()
            .find(|spec| !seen.contains(spec.name))
            .map_or("unknown", |spec| spec.name);
        return Err(VokraError::ModelLoad(format!(
            "cosyvoice2_flow: saw {} flow tensors, expected {}; first missing {}",
            seen.len(),
            FLOW_TENSOR_COUNT,
            missing
        )));
    }
    Ok(())
}

fn canonical_manifest_digest() -> [u8; 32] {
    let mut canonical = Vec::new();
    for spec in FLOW_SCHEMA {
        canonical.extend_from_slice(spec.name.as_bytes());
        canonical.push(0);
        canonical.extend_from_slice(&(spec.shape.len() as u64).to_le_bytes());
        for &dimension in spec.shape {
            canonical.extend_from_slice(&(dimension as u64).to_le_bytes());
        }
    }
    sha256_bytes(&canonical)
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::{GgmlType, GgufBuilder};

    fn metadata_builder() -> GgufBuilder {
        let mut builder = GgufBuilder::new();
        for (key, value) in [
            (KEY_ARCH, FLOW_ARCH),
            (KEY_COMPONENT, FLOW_COMPONENT),
            (KEY_COMPOSITE_STATUS, FLOW_COMPOSITE_STATUS),
            (KEY_UPSTREAM_HF, FLOW_UPSTREAM_HF),
            (KEY_UPSTREAM_REVISION, FLOW_UPSTREAM_REVISION),
            (KEY_COMPONENT_UPSTREAM_REVISION, FLOW_UPSTREAM_REVISION),
            (KEY_CHECKPOINT_SHA256, FLOW_ARTIFACT_SHA256),
            (KEY_CHECKPOINT_FILE, FLOW_ARTIFACT_FILE),
            (KEY_CHECKPOINT_COMPONENT_SHA256, FLOW_ARTIFACT_SHA256),
            (KEY_CONFIG_FILE, FLOW_CONFIG_FILE),
            (KEY_CONFIG_SHA256, FLOW_CONFIG_SHA256),
            (KEY_CONFIG_GIT_BLOB_SHA1, FLOW_CONFIG_GIT_BLOB_SHA1),
            (KEY_SOURCE_REPOSITORY, FLOW_SOURCE_REPOSITORY),
            (KEY_SOURCE_REVISION, FLOW_SOURCE_REVISION),
            (KEY_SOURCE_LICENSE_FILE, FLOW_SOURCE_LICENSE_FILE),
            (KEY_SOURCE_LICENSE_SHA256, FLOW_SOURCE_LICENSE_SHA256),
            (
                KEY_SOURCE_LICENSE_GIT_BLOB_SHA1,
                FLOW_SOURCE_LICENSE_GIT_BLOB_SHA1,
            ),
            (KEY_SOURCE_LICENSE_DECLARED, FLOW_SOURCE_LICENSE_DECLARED),
            (KEY_MANIFEST_SHA256, FLOW_MANIFEST_SHA256),
            (KEY_DATA_PICKLE_SHA256, FLOW_DATA_PICKLE_SHA256),
            (KEY_STORAGE_MANIFEST_SHA256, FLOW_STORAGE_MANIFEST_SHA256),
        ] {
            builder.add_string(key, value);
        }
        for (path, sha256, blob) in SOURCE_ROLES {
            builder
                .add_string(
                    &format!("vokra.cosyvoice2_flow.source.{path}.sha256"),
                    sha256,
                )
                .add_string(
                    &format!("vokra.cosyvoice2_flow.source.{path}.git_blob_sha1"),
                    blob,
                );
        }
        builder
            .add_u32(KEY_SOURCE_LICENSE_BYTES, FLOW_SOURCE_LICENSE_BYTES)
            .add_u32(KEY_CHECKPOINT_BYTES, FLOW_ARTIFACT_BYTES)
            .add_u32(KEY_CONFIG_BYTES, FLOW_CONFIG_BYTES);
        builder
    }

    fn tiny_file(mut builder: GgufBuilder) -> GgufFile {
        builder
            .add_tensor(FLOW_SCHEMA[0].name, GgmlType::F32, vec![1], vec![0; 4])
            .expect("tiny tensor");
        GgufFile::parse(builder.to_bytes().expect("GGUF")).expect("parse")
    }

    #[test]
    fn authenticated_schema_count_digest_and_names_are_pinned() {
        assert_eq!(FLOW_SCHEMA.len(), FLOW_TENSOR_COUNT);
        assert_eq!(canonical_manifest_digest(), FLOW_MANIFEST_DIGEST);
        let names = FLOW_SCHEMA.iter().map(|spec| spec.name).collect::<Vec<_>>();
        let mut sorted = names.clone();
        sorted.sort_unstable();
        assert_eq!(names, sorted);
        assert!(FLOW_SCHEMA.iter().all(|spec| spec.dtype == GgmlType::F32));
        assert!(
            FLOW_SCHEMA
                .windows(2)
                .all(|window| window[0].name != window[1].name)
        );
    }

    #[test]
    fn metadata_is_required_before_schema_or_payload() {
        let file = tiny_file(GgufBuilder::new());
        let error = FlowWeights::bind(&file).expect_err("missing identity metadata");
        assert!(error.to_string().contains("metadata"));
    }

    #[test]
    fn missing_flow_tensor_fails_closed_after_metadata() {
        let file = GgufFile::parse(metadata_builder().to_bytes().expect("GGUF")).expect("parse");
        let error = FlowWeights::bind(&file).expect_err("missing flow tensors");
        assert!(error.to_string().contains("flow tensors"));
    }

    #[test]
    fn metadata_tamper_rejects_before_tensor_schema() {
        let mut builder = metadata_builder();
        builder.add_string(KEY_COMPONENT, "llm");
        let file = tiny_file(builder);
        let error = FlowWeights::bind(&file).expect_err("tampered component metadata");
        assert!(error.to_string().contains("metadata"));
        assert!(!error.to_string().contains("flow tensors"));
    }

    #[test]
    fn wrong_shape_fails_closed_without_payload_allocation() {
        let mut builder = metadata_builder();
        builder
            .add_tensor(FLOW_SCHEMA[0].name, GgmlType::F32, vec![2], vec![0; 8])
            .expect("tiny tensor");
        let file = GgufFile::parse(builder.to_bytes().expect("GGUF")).expect("parse");
        let error = FlowWeights::bind(&file).expect_err("wrong shape");
        assert!(error.to_string().contains("shape"));
    }

    #[test]
    fn wrong_dtype_fails_closed_without_payload_allocation() {
        let mut builder = metadata_builder();
        builder
            .add_tensor(FLOW_SCHEMA[0].name, GgmlType::F16, vec![1], vec![0; 2])
            .expect("tiny tensor");
        let file = GgufFile::parse(builder.to_bytes().expect("GGUF")).expect("parse");
        let error = FlowWeights::bind(&file).expect_err("wrong dtype");
        assert!(error.to_string().contains("dtype"));
    }

    #[test]
    fn unexpected_flow_namespace_fails_but_composite_names_are_outside_scope() {
        assert!(is_flow_tensor_name("decoder.unexpected.weight"));
        assert!(!is_flow_tensor_name("llm.model.layers.0.weight"));
        assert!(!is_flow_tensor_name("conv_pre.weight"));

        let mut builder = metadata_builder();
        builder
            .add_tensor(
                "decoder.unexpected.weight",
                GgmlType::F32,
                vec![1],
                vec![0; 4],
            )
            .expect("tiny tensor");
        let file = GgufFile::parse(builder.to_bytes().expect("GGUF")).expect("parse");
        let error = FlowWeights::bind(&file).expect_err("unexpected flow tensor");
        assert!(error.to_string().contains("unexpected flow tensor"));
    }
}
