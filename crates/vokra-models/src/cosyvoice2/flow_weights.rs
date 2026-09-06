//! Authenticated, staged CosyVoice2 flow-component GGUF binder.
//!
//! This module intentionally stops at a strict header/schema binding seam.
//! It does not enable a public flow runtime, synthesize payloads, or make
//! CPU/Metal/parity/publication claims. Tensor names and shapes below are
//! generated from the authenticated component evidence, not inferred from a
//! model implementation. The fixed Matcha gitlink is authenticated by the
//! source-closure gate, while this binder remains fail-closed/staged.
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
pub(crate) const FLOW_SOURCE_CLOSURE_MANIFEST_SHA256: &str =
    "cc391a4a63e95b9cdf6373b5b41ac94239a89d6594a235bc142a17c542532673";

const FLOW_ARCH: &str = "cosyvoice2";
const FLOW_NAME: &str = "cosyvoice2-0.5b-flow";
const FLOW_COMPONENT: &str = "flow";
const FLOW_COMPOSITE_STATUS: &str = "INSPECTION_ONLY";
const KEY_ARCH: &str = chunks::KEY_MODEL_ARCH;
const KEY_NAME: &str = chunks::KEY_MODEL_NAME;
const KEY_PROVENANCE_WEIGHT_LICENSE: &str = chunks::KEY_PROVENANCE_WEIGHT_LICENSE;
const KEY_PROVENANCE_LICENSE: &str = chunks::KEY_PROVENANCE_LICENSE;
const KEY_PROVENANCE_MODEL_ID: &str = chunks::KEY_PROVENANCE_MODEL_ID;
const KEY_PROVENANCE_SOURCE: &str = chunks::KEY_PROVENANCE_SOURCE;
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
const KEY_SOURCE_CLOSURE_MANIFEST_SHA256: &str =
    "vokra.cosyvoice2_flow.source_closure_manifest_sha256";
const KEY_PREPARED_BYTES: &str = "vokra.cosyvoice2_flow.prepared_input.bytes";
const KEY_PREPARED_SHA256: &str = "vokra.cosyvoice2_flow.prepared_input.sha256";
const KEY_PREPARED_STATUS: &str = "vokra.cosyvoice2_flow.prepared_input.authentication_status";
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
    (
        "cosyvoice/transformer/activation.py",
        "a4a96caea9f7b05111286de5ea760bdfdf9c4e53448f53420857dc04a50440c2",
        "8cea54816385d3b6585ccc2417bc71630d578177",
    ),
    (
        "cosyvoice/transformer/attention.py",
        "41d269b2d5e3a3952c0a538f46480aff8570767eb88e80452f367436cef628b1",
        "8c0c0983a833a6a91cae306a8198ebb2ca82696f",
    ),
    (
        "cosyvoice/transformer/convolution.py",
        "eb5f41f324a97225c73b9c430af8bb231bd77386ef18450bb0fd329f469ab4e1",
        "4d5d96149154776000991a681a666fbe55e562fe",
    ),
    (
        "cosyvoice/transformer/embedding.py",
        "b50c30be5c66c39c95e0db1015814c545fcd062a936c0ea7ad4c01508b2595dc",
        "ba20d71bec11baa7081bed522399cf3a2046ef56",
    ),
    (
        "cosyvoice/transformer/encoder_layer.py",
        "1920582be2c9b7cf7835e7b539118ec1eb9da8f2c711c6257f8598b3792951b9",
        "efbb12dd365770bebe8bca75276fe63be260a08f",
    ),
    (
        "cosyvoice/transformer/positionwise_feed_forward.py",
        "6e8038e3bcc8ca086ddca508fb9b9d40bf3ab98bebabcf5c94935d032329d0f8",
        "b7a2cf6e7315e3a5ed2794423daff0a59cc5b208",
    ),
    (
        "cosyvoice/transformer/subsampling.py",
        "31fc0347a851abc1205dfd4cf68099c0c416589c3e99a07b250e8630cece19d0",
        "e17c2e324e3afb24e1b619effe29cef07c9c5b3a",
    ),
    (
        "cosyvoice/utils/class_utils.py",
        "75d6977f7574304f8464cdb50c889564ca557a99a112419918b2981f0e8dcfc2",
        "c49de00c873340e6c45dd05299b684d81f19c5a4",
    ),
    (
        "cosyvoice/utils/common.py",
        "6161a8d90d7beb0766f6d2de67ccce35a754af43930c28af37c4d19b65eef897",
        "6f5a3dd8b7ae99601783c3a4ed91b3b64270fab3",
    ),
    (
        "cosyvoice/utils/mask.py",
        "852c6e4b14201a238ab076396db599c157b2ee5a5a2aacf0f277ddce8c648976",
        "5d3dfd6ca6cd84e95f237a9aef4467cb7d2d4c33",
    ),
];

pub(crate) const FLOW_MANIFEST_DIGEST: [u8; 32] = [
    0x68, 0xab, 0xcd, 0xcd, 0xc9, 0x61, 0x09, 0x1b, 0x9c, 0x9e, 0x64, 0x56, 0x14, 0x6a, 0xc9, 0xda,
    0x06, 0xcd, 0xa9, 0xf8, 0x6e, 0x45, 0xa8, 0xfa, 0xaa, 0xc2, 0x2a, 0x3b, 0x0b, 0x26, 0x21, 0x7d,
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

fn require_positive_u64(file: &GgufFile, key: &str) -> Result<()> {
    match file.get(key) {
        Some(GgufMetadataValue::U64(value)) if *value > 0 => Ok(()),
        Some(value) => Err(VokraError::ModelLoad(format!(
            "cosyvoice2_flow: metadata {key}={value:?}, expected positive U64"
        ))),
        None => Err(VokraError::ModelLoad(format!(
            "cosyvoice2_flow: missing/non-u64 metadata {key}"
        ))),
    }
}

fn require_lowercase_sha256(file: &GgufFile, key: &str) -> Result<()> {
    match file.get(key).and_then(GgufMetadataValue::as_str) {
        Some(value)
            if value.len() == 64
                && value.bytes().all(|byte| byte.is_ascii_hexdigit())
                && value.bytes().all(|byte| !byte.is_ascii_uppercase()) =>
        {
            Ok(())
        }
        Some(value) => Err(VokraError::ModelLoad(format!(
            "cosyvoice2_flow: metadata {key}={value:?}, expected lowercase SHA-256"
        ))),
        None => Err(VokraError::ModelLoad(format!(
            "cosyvoice2_flow: missing/non-string metadata {key}"
        ))),
    }
}

fn validate_metadata(file: &GgufFile) -> Result<()> {
    for (key, expected) in [
        (KEY_ARCH, FLOW_ARCH),
        (KEY_NAME, FLOW_NAME),
        (KEY_PROVENANCE_WEIGHT_LICENSE, "permissive"),
        (KEY_PROVENANCE_LICENSE, FLOW_SOURCE_LICENSE_DECLARED),
        (KEY_PROVENANCE_MODEL_ID, FLOW_NAME),
        (KEY_PROVENANCE_SOURCE, FLOW_SOURCE_REPOSITORY),
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
        (
            KEY_SOURCE_CLOSURE_MANIFEST_SHA256,
            FLOW_SOURCE_CLOSURE_MANIFEST_SHA256,
        ),
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
    require_positive_u64(file, KEY_PREPARED_BYTES)?;
    require_lowercase_sha256(file, KEY_PREPARED_SHA256)?;
    require_string(
        file,
        KEY_PREPARED_STATUS,
        "PREPARED_INPUT_DIGEST_RECORDED_NOT_PINNED",
    )?;
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
            return Err(VokraError::ModelLoad(format!(
                "cosyvoice2_flow: tensor {} is outside the authenticated flow component",
                info.name
            )));
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
        metadata_builder_with_prepared(true)
    }

    fn metadata_builder_with_prepared(include_prepared: bool) -> GgufBuilder {
        let mut builder = GgufBuilder::new();
        for (key, value) in [
            (KEY_ARCH, FLOW_ARCH),
            (KEY_NAME, FLOW_NAME),
            (KEY_PROVENANCE_WEIGHT_LICENSE, "permissive"),
            (KEY_PROVENANCE_LICENSE, FLOW_SOURCE_LICENSE_DECLARED),
            (KEY_PROVENANCE_MODEL_ID, FLOW_NAME),
            (KEY_PROVENANCE_SOURCE, FLOW_SOURCE_REPOSITORY),
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
            (
                KEY_SOURCE_CLOSURE_MANIFEST_SHA256,
                FLOW_SOURCE_CLOSURE_MANIFEST_SHA256,
            ),
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
        if include_prepared {
            builder
                .add_metadata(KEY_PREPARED_BYTES, GgufMetadataValue::U64(1))
                .add_string(KEY_PREPARED_SHA256, &"a".repeat(64))
                .add_string(
                    KEY_PREPARED_STATUS,
                    "PREPARED_INPUT_DIGEST_RECORDED_NOT_PINNED",
                );
        }
        builder
    }

    fn tiny_file(mut builder: GgufBuilder) -> GgufFile {
        builder
            .add_tensor(FLOW_SCHEMA[0].name, GgmlType::F32, vec![1], vec![0; 4])
            .expect("tiny tensor");
        GgufFile::parse(builder.to_bytes().expect("GGUF")).expect("parse")
    }

    #[test]
    fn manifest_digest_string_matches_byte_array() {
        let rendered = FLOW_MANIFEST_DIGEST
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        assert_eq!(rendered, FLOW_MANIFEST_SHA256);
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
    fn prepared_input_metadata_is_required_and_strictly_typed() {
        let missing = tiny_file(metadata_builder_with_prepared(false));
        let error = FlowWeights::bind(&missing).expect_err("prepared metadata is required");
        assert!(error.to_string().contains(KEY_PREPARED_BYTES));

        let mut wrong_type = metadata_builder();
        wrong_type.add_metadata(KEY_PREPARED_BYTES, GgufMetadataValue::U32(1));
        let error =
            FlowWeights::bind(&tiny_file(wrong_type)).expect_err("prepared bytes must remain U64");
        assert!(error.to_string().contains("positive U64"));

        let mut zero = metadata_builder();
        zero.add_metadata(KEY_PREPARED_BYTES, GgufMetadataValue::U64(0));
        let error = FlowWeights::bind(&tiny_file(zero)).expect_err("zero bytes must fail");
        assert!(error.to_string().contains("positive U64"));

        let mut malformed_digest = metadata_builder();
        malformed_digest.add_string(KEY_PREPARED_SHA256, "not-a-sha");
        let error = FlowWeights::bind(&tiny_file(malformed_digest))
            .expect_err("prepared digest must be lowercase SHA-256");
        assert!(error.to_string().contains(KEY_PREPARED_SHA256));

        let mut malformed_status = metadata_builder();
        malformed_status.add_string(KEY_PREPARED_STATUS, "APPROVED");
        let error = FlowWeights::bind(&tiny_file(malformed_status))
            .expect_err("prepared status must remain unpinned");
        assert!(error.to_string().contains(KEY_PREPARED_STATUS));
    }

    #[test]
    fn standard_identity_metadata_is_required() {
        let mut builder = metadata_builder();
        builder.add_string(KEY_NAME, "wrong-name");
        let error = FlowWeights::bind(&tiny_file(builder)).expect_err("name must be authenticated");
        assert!(error.to_string().contains(KEY_NAME));
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

    #[test]
    fn foreign_extra_tensor_is_rejected_in_standalone_component() {
        let mut builder = metadata_builder();
        builder
            .add_tensor(
                "llm.model.layers.0.weight",
                GgmlType::F32,
                vec![1],
                vec![0; 4],
            )
            .expect("foreign tensor");
        let file = GgufFile::parse(builder.to_bytes().expect("GGUF")).expect("parse");
        let error = FlowWeights::bind(&file).expect_err("foreign tensor must fail closed");
        assert!(
            error
                .to_string()
                .contains("outside the authenticated flow component")
        );
    }

    fn required_vast_gguf() -> GgufFile {
        let value = std::env::var("VOKRA_COSYVOICE2_FLOW_GGUF")
            .expect("VOKRA_COSYVOICE2_FLOW_GGUF is required");
        let path = std::path::PathBuf::from(value);
        assert!(path.is_absolute(), "flow GGUF path must be absolute");
        let metadata = std::fs::symlink_metadata(&path).expect("flow GGUF must exist");
        assert!(metadata.file_type().is_file(), "flow GGUF must be regular");
        assert!(
            !metadata.file_type().is_symlink(),
            "flow GGUF must not be a symlink"
        );
        GgufFile::open(path).expect("open VAST flow GGUF")
    }

    /// VAST-only: bind the real converted component without executing it.
    #[test]
    #[ignore = "requires the authenticated VAST CosyVoice2 flow GGUF"]
    fn vast_real_flow_gguf_binds_exact_component() {
        let file = required_vast_gguf();
        assert_eq!(file.tensors().len(), FLOW_TENSOR_COUNT);
        let bound = FlowWeights::bind(&file).expect("strict flow bind");
        assert_eq!(bound.file().tensors().len(), FLOW_TENSOR_COUNT);
    }
}
