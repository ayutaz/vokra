# Agent guidance coverage audit — 2026-10-23

This is a dated, read-only coverage record for the repository agent guidance
used during the documentation refresh. It records the actual skill paths read,
the authority relationships checked, and the deliberately narrow edits. It is
not an implementation, model, parity, license, publication, or hardware
approval record.

## Scope and method

- `AGENTS.md` and all 16 repository skill files listed below were read in full.
  The 16 files are eight `.agents/skills/*/SKILL.md` files and their eight
  `.claude/skills/*/SKILL.md` twins.
- The `skill-creator` guidance and its `quick_validate.py` entry point were
  read before editing. Frontmatter (`name` and `description`) was preserved.
- The eight `.agents`/`.claude` pairs were compared before editing. The pairs
  were byte-identical at the start; the two pairs named below were edited with
  identical content, and the other six pairs remain byte-identical.
- Read-only authority checks covered `AGENTS.md`, the current numerical-parity
  guidance, `.codex/hooks.json`, and the hook/permission references named by
  the two edited skills. No runtime, model, package, Cargo-heavy, cloud, or
  device command was run.

## Per-path coverage

| Path | Status | Actual source(s) read and relation checked | Limits |
|---|---|---|---|
| `AGENTS.md` | Read, unchanged | Project delegation, memory, VAST, verification, and skill-routing rules | Policy was not changed. |
| `.agents/skills/add-audio-operator/SKILL.md` | Unchanged | Full skill; CPU/Metal/CUDA, GPL/codec, parity, and source-first routing | No operator implementation or backend run. |
| `.claude/skills/add-audio-operator/SKILL.md` | Unchanged | Full twin; compared with the `.agents` path | Pair remains byte-identical. |
| `.agents/skills/add-speech-model/SKILL.md` | **Changed** | Full skill; `AGENTS.md` and `numerical-parity` independent-reference/GPU section | §1 now says CPU comparison follows independent upstream-reference validation and GPU/CPU parity alone is not completion; thresholds, metadata, and permissions unchanged. |
| `.claude/skills/add-speech-model/SKILL.md` | **Changed** | Full twin; compared with edited `.agents` path | Identical twin of the bounded §1 clarification. |
| `.agents/skills/complete-mac-cpu-metal/SKILL.md` | Unchanged | Full skill; VAST/Scaleway boundary, owner/legal gates, and protected-manifest references | No Apple, VAST, or Scaleway execution. |
| `.claude/skills/complete-mac-cpu-metal/SKILL.md` | Unchanged | Full twin; compared with the `.agents` path | Pair remains byte-identical. |
| `.agents/skills/license-audit/SKILL.md` | Unchanged | Full skill; `docs/license-audit.md`, `deny.toml`, sign-off, and fail-closed publication relation | No license decision or dependency audit was performed. |
| `.claude/skills/license-audit/SKILL.md` | Unchanged | Full twin; compared with the `.agents` path | Pair remains byte-identical. |
| `.agents/skills/numerical-parity/SKILL.md` | Unchanged | Full skill; independent upstream reference, GPU/CPU distinction, device gate, and exact-scope completion rules | No reference generation or parity run. |
| `.claude/skills/numerical-parity/SKILL.md` | Unchanged | Full twin; compared with the `.agents` path | Pair remains byte-identical. |
| `.agents/skills/publish-model-to-hf/SKILL.md` | Unchanged | Full skill; `publish-one.sh` as source of truth, five gates, and separate upload permission | No token, model, or publication action. |
| `.claude/skills/publish-model-to-hf/SKILL.md` | Unchanged | Full twin; compared with the `.agents` path | Pair remains byte-identical. |
| `.agents/skills/refresh-vokra-docs/SKILL.md` | Unchanged | Full skill; evidence-first refresh, protected paths, static gates, and handoff boundary | No commit, push, or PR action. |
| `.claude/skills/refresh-vokra-docs/SKILL.md` | Unchanged | Full twin; compared with the `.agents` path | Pair remains byte-identical. |
| `.agents/skills/vast-ai-workflow/SKILL.md` | **Changed** | Full skill; `AGENTS.md`, `.codex/hooks.json`, and `guard-local-memory.sh` boundary references | Removed only the copied `43` case count and pointed readers to the current `--self-test`; hook names and `VOKRA_ALLOW_LOCAL_HEAVY=1` authorization boundary unchanged. |
| `.claude/skills/vast-ai-workflow/SKILL.md` | **Changed** | Full twin; compared with edited `.agents` path | Identical twin of the bounded self-test wording update. |

## Validation and remaining limits

The edited pairs were re-compared with `cmp`; both comparisons passed. The
skill-creator quick validator was attempted for all four edited skill
directories through the repository's offline UV 3.12 path, but each invocation
stopped before validation with `ModuleNotFoundError: No module named 'yaml'`.
Per the no-install boundary, PyYAML was not installed and no sync/network
fallback was attempted. `git diff --check` and the documentation-reference
self-test were run for the documentation/skill scope.
No source implementation, hook behavior, thresholds, oracle implementation,
execution permission, frontmatter, model artifact, or public artifact was
changed. Any future hook-case count must come from the hook's current
`guard-local-memory.sh --self-test`, not from this documentation.
