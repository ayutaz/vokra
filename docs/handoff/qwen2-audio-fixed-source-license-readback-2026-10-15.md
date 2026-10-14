# Qwen2 Audio fixed-source license readback — 2026-10-15 JST

Read-only investigation at accepted Vokra main
`f0130b5de5c37e4441612a85ecaa4e39ae0344f9`. This is a bounded primary-source
inspection, not a completed production history-audit receipt or legal approval.

## Separate repository identities

The public catalog row is `vokra/qwen2-audio-7b-instruct` at
`57421d5970ee83f2d5561888fd45476765ed27dd`. Its inspector separately identifies
the upstream model `Qwen/Qwen2-Audio-7B-Instruct` at
`0a095220c30b7b31434169c3086508ef3ea5bf0a` and the official execution source
`QwenLM/Qwen2-Audio` at `595360e82b5839c1507492ec83cae5bda6d5c7d4`.
These are different repositories. Neither the model's license declaration nor
the separate Transformers source can grant rights to Qwen2-Audio source code.

## Actual fixed-tree metadata and its limits

The agent's direct recursive API request failed DNS resolution. The manager
subsequently retrieved the official API through the GitHub connector at
approximately 16:11 UTC on 2026-10-14:

- [Fixed commit](https://api.github.com/repos/QwenLM/Qwen2-Audio/commits/595360e82b5839c1507492ec83cae5bda6d5c7d4)
  binds Git tree `0d66378d9e4b1a4d6bdba3969aa87ef777f487cd`.
- [That exact recursive tree](https://api.github.com/repos/QwenLM/Qwen2-Audio/git/trees/0d66378d9e4b1a4d6bdba3969aa87ef777f487cd?recursive=1)
  returns `truncated=false`, 23 entries and 19 blobs. No path matches
  license/licence, copying, notice, copyright or EULA filename terms.
- The [fixed README](https://raw.githubusercontent.com/QwenLM/Qwen2-Audio/595360e82b5839c1507492ec83cae5bda6d5c7d4/README.md)
  routes model-license questions to each model's repository. This does not
  establish an explicit source-code grant.

Filename absence is not proof that every source body, every ancestor, every
branch or every release lacks a grant. A root `LICENSE` 404 alone would establish
only that one path is absent. This investigation did not run the production
collector's complete reachable-history, branch-tip, release/tag and clean-HEAD
binding contract; it must not be substituted for that receipt. No license-body
hash or new authorized source route is invented.

## Verification and remaining gates

The source audit's offline UV Python 3.12 self-test passes, proving synthetic
history/rejection behavior only. The main inspector's self-test could not run
without `safetensors`; no installation was attempted to obtain a passing result.
Existing production source-license and dependency/dataset/native/parity gates
remain fail-closed. Source grant, exact owner scope, dependency/native review,
complete audio-language implementation and independent real-weight CPU parity
remain unresolved. The 58-row denominator is unchanged.

No file/manifest implementation changed, approval was signed, model/config/
tokenizer was acquired or imported, inference or heavy Cargo was run, paid
worker was created, or artifact was published. The protected CosyVoice2 manifest
was not accessed. Continue independent source work, without repeating a paid
model audit to resolve an upstream rights question.
