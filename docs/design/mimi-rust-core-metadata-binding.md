# Moshi e6 core metadata binding

Status: `OPEN / SOURCE-ONLY / VAST-ONLY`

`tools/parity/mimi_rust_core_metadata_binding.py` reuses the accepted
`mimi_rust_core_source_audit` collector to prepare a disposable single-member
workspace and records the exact `cargo metadata` invocation. It does not fetch
packages, rewrite the upstream lock, build Rust, access checkpoints, or claim
a CPU/Metal/parity result. The normal path is VAST-only; missing authenticated
toolchain binaries, source identity, or a clean copied lock remain BLOCKED.

The source identity is fixed to
`e6a55d2722a65870ef52a6c9f6ecfc0e90f38362` and the upstream lock SHA-256 is
`bc4348116cdf1408583311954c1baaee6bde3b4cf54cd160af50b278aa4e44ab`.
The source package is `moshi@0.6.4`; `moshi-core` is its directory. The
derived workspace copies `rust/README.md` to the workspace-root `README.md`
because the member manifest references `../README.md`.

The only metadata command is the bounded, offline CPU candidate:

```text
cargo metadata --locked --offline --format-version 1 \
  --filter-platform x86_64-unknown-linux-gnu --no-default-features \
  --manifest-path <derived-workspace>/Cargo.toml
```

`target_filter` is not invented in or required from Cargo's raw JSON. The
target is authenticated by the exact argv/environment recorded in the packet;
the raw metadata bytes remain unchanged.

The accepted source inventory is rechecked by its canonical payload digest
(the digest field itself excluded), and its source facts, pinned lock, derived
manifest, copied lock, and every copied member are compared before and after
the metadata process. Packet output is no-clobber and rejects symlink members.

Failure is retained as `BLOCKED_METADATA_COMMAND` (or a more specific
`BLOCKED` packet) with bounded raw stdout, stderr, exit status, argv, and
environment files. There is no network, fallback resolution, lock
replacement, or feature expansion. The cargo and rustc paths are bound via
`rustup which`, streamed-hashed as regular files, version-checked, and
rechecked after the command; Cargo is invoked by authenticated absolute path.
Reachability from `resolve.root` is reported separately from target/feature
activation, and complete `dep_kinds` remain evidence rather than an implicit
CPU claim. Activated or ambiguous CUDA/Metal/flash-attn/native markers retain
their BLOCKED status and are never overwritten by a later OPEN status.

Registry archive, native payload, license/NOTICE/build-script, owner/legal,
checkpoint, independent reference, real-weight CPU, and Apple Metal gates are
all explicitly OPEN. A green metadata command alone cannot promote any of
those states.
