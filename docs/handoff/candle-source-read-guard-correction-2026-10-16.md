# Candle source-read guard correction — 2026-10-16 JST

## Authorization and cause

The owner permits a limited correction of the read-only false positive while
retaining the maintainer Mac's ban on model acquisition/execution. The refused
command is a literal `rg -n` inventory of the CPU compatibility checker. Its
single-quoted search pattern contains regex alternation pipes. The existing
defensive segment splitter treats those bytes as shell separators and
misclassifies a later segment containing the parity-source path as execution.

Only `.codex/hooks/guard-local-models.sh` changes: one byte-exact command
constant, an equality check before generic segmentation and nine regression
cases (21 added lines total). No general `rg` exception, generic shell parser,
configuration/trust change, hook disablement or local-heavy override is added.
All original model-execution, model-download and protected-path rules remain.
[Official OpenAI hook documentation](https://learn.chatgpt.com/docs/hooks#pretooluse)
defines command-input inspection and blocking at PreToolUse; this repair
changes this repository's classifier, not the product permission settings.

## Reviewed evidence

Luna applies the same patch to the canonical repository's isolated
`codex/fix-candle-readonly-guard-20261016` branch and the private Candle
candidate clone. Manager reviews the complete diff, independently reruns the
guard self-test and executes the original previously refused read-only
command successfully. No model/Python/Cargo invocation is part of that read.

Both copies change SHA-256 from
`0b8f3b1a343e1a8fd10e47f47d09dc65cbf9dc43aabb53bbeb3b68c9611edc62`
to `45e2c0852dadf8c54fed7d67a06a4a61a5c4ccd565044d5accd315cb62e15e41`.
Manager's byte comparison and whitespace check pass. Luna reports passing
shell syntax, both guard self-tests, both full Codex-hook contract gates and
ShellCheck at error severity. Ordinary ShellCheck retains pre-existing
warnings; no warning suppression is introduced.

Negative regressions reject appended model commands using `&&`, semicolon,
pipe or newline, changed/unquoted patterns and substitution variants. Tests
classify inert strings; none executes the model command it describes.

## Remaining boundary

This resolves the source-read blocker, not CPU compatibility. Review of the
new checker resumes and identifies toolchain-environment/cache-snapshot and
subprocess-cleanup defects requiring correction before its one permitted
remote job. No compatibility job, model download/run, paid-worker allocation,
owner/legal signature, Apple result or publication follows from this hook
change. The original full non-Apple, real-weight CPU and final clean-HEAD /
fresh Apple-packet objective remains unfinished.
