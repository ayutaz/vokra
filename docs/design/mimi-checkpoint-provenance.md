# Mimi checkpoint input provenance

The standalone Mimi safetensors converter records the identity of the exact
input byte buffer supplied to `crates/vokra-convert/src/models/mimi.rs`.
Before safetensors parsing consumes that buffer, the converter measures:

| GGUF key | Type | Meaning |
| --- | --- | --- |
| `vokra.provenance.checkpoint_sha256` | `string` | Lowercase, 64-character SHA-256 of the exact input bytes. |
| `vokra.provenance.checkpoint_bytes` | `u64` | Exact byte length of that input buffer. |

These fields are additive to the existing generic license and attribution
stamp. They do not change tensor mapping, quantizer derivation, structural
metadata, or pass-through behavior.

The digest is a measured input identity, not an independent conversion proof,
source-origin assertion, revision or filename claim, owner approval, or
license sign-off. This converter does not infer any of those facts. The
converter also cannot self-embed the final GGUF whole-file hash without a
circular definition; an external transfer packet must authenticate the output
bytes separately.

The synthetic converter tests check the exact input digest and length and show
that a structurally valid source-byte change receives a different identity.
They do not execute a model, compare numerical outputs, or establish rights to
any real checkpoint. Runtime binding and publication gates remain unchanged.
