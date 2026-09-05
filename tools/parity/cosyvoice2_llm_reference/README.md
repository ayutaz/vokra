# CosyVoice2 LLM official reference

This is a separate Linux x86_64 VAST-only CPU reference project for the
CosyVoice2 LLM component. It uses the official Transformers `Qwen2ForCausalLM`
eager implementation and fixed token IDs; it does not import the CosyVoice
front-end and therefore does not pull the forbidden `librosa`/`soxr` closure.

The checked-in license gate is intentionally `PENDING_REVIEW` and
`NO_UPLOAD`. A dedicated `uv.lock` must be generated with `uv add`/`uv lock`
on an approved networked worker and then audited for every package and native
payload before this project may sync or acquire a model. Until that happens,
the reference dumper fails closed before model acquisition.

Pinned inputs are the `llm.pt` component at the immutable CosyVoice2 revision,
the exact `CosyVoice-BlankEN/config.json` Qwen sidecar, and the clean official
CosyVoice source revision. No tokenizer download, model upload, or fallback
implementation is permitted.
