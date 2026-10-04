# Source-contract fixture provenance

`source_contract_primary_packet.json` records the source-only contract shape
returned by `authenticate_streaming_source_contract` in
`tools/parity/kyutai_stt_decoder_dump_reference.py` (the helper's constants,
role order, contract names, expressions, and blocker boundary).  The role
sizes and Git blob IDs come from the retained primary packet's `dsm-tree.json`
and `moshi-tree.json`; SHA-256 values for the roles present in
`source-receipt.json` are retained byte-for-byte.

The primary packet does not contain the config bytes, but the independent
fixed-revision GitHub API receipt supplies its exact 1016-byte identity:
Git blob `75382f8dabeb31832f28c8754afaaf63aaa7b158` and SHA-256
`81f77d642689e1acb276089f62064dab2e71a5532fcac2d3c12563cf4946552c`.
The primary source URL is
`https://github.com/kyutai-labs/delayed-streams-modeling/blob/4c4f65e147df056adf3346290d64c7b9649b18c9/configs/config-stt-en-hf.toml`.
`generate_source_contract_fixture.py` invokes the existing helper against AST
bytes from the retained packet; its only mock boundary is `git_identity`,
which injects that reviewed metadata because the packet lacks the config body.
The generated JSON is a positive schema fixture, not clean-checkout,
runtime, numerical, dependency, or legal approval evidence.

The generator uses decoder helper SHA-256
`a028ff8e27b8e179c00e1988854d0bdf592d6e3a750c9b18631ea30c10e0e814`, the
stdlib contract helper SHA-256
`092a4cd77d1f1e6d068ee8f130b384b2ae1af25f435d94730eafcdd706be4944`, and
requires packet manifest SHA-256
`85223a7ac8b947eaeafa7b2f337a1ac60dea84a75d44ee0c607df72ad25d6342`.
Regenerate without project synchronization or third-party imports:

```text
UV_CACHE_DIR=/private/tmp/vokra-uv-cache uv run --no-project --no-sync --python 3.12 python -S fixtures/generate_source_contract_fixture.py --packet /path/to/packet
```
