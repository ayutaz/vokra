# MOSS API-smoke urllib3 candidate reconciliation — 2026-10-15 JST

Source-only candidate based on accepted main
`0f6750fa401b613da7155e894065fdc70fb172bc`. The separate API-smoke project
has **45 full package records**, not the main MOSS lock's 48. Its changes
normally include Dependabot #246 head
`f5f9fc7fb59a40d0fcfbf3959bf0a0196a570528`; this record does not establish
PR acceptance or fresh integrated-head CI.

## Cause and bounded correction

The bot updates urllib3 2.7.0 to 2.8.0 but also removes 60 dependency-edge
Linux x86_64 markers and nine root-dependency markers. The corrected candidate
preserves all original markers and changes only urllib3's version/artifact
record in the lock. The API-smoke tool's fixed lock digest is updated with
the actual corrected bytes:

| Input | SHA-256 |
|---|---|
| Accepted API-smoke lock | `f64849e762adac41eebddf54debb64ef6f33a935b8a5931dc3ec9feb3390064c` |
| Original bot lock | `29abe68bf0881760b6b3c1cf89f3d409066d270a72bf81c96f643794ea8d69f4` |
| Corrected API-smoke lock | `517ed7226440a5b6b6d2e462619ad705ba24bc7bc8e87cdecc59317a77b1d4be` |
| Unchanged API-smoke project | `17431a2f3ddfb6a9853c47c51f2e99209fcf8c0497e0675c5eb4016cbed58aec` |

Manager stdlib TOML/AST/hash comparison confirms complete equality of all
other package records and top-level lock fields, correct actual project/lock
bindings, and equality of the urllib3 record to the separately accepted #242
main-MOSS metadata. This compares declarations, not downloaded archive bytes
or installed-package/API compatibility. No package or model is imported.

Luna retains the bot ancestry through an ordinary merge, then restores the
reviewed marker semantics. Both genuine `--self-test` commands for API smoke
and preflight pass, as do metadata comparison, diff hygiene and all five
ordinary pre-commit checks. Manager independently passes both self-tests and
the complete metadata comparison. UV uses Python 3.12/no-project/offline;
the reviewed stdlib-only hook avoids parity-environment synchronization.

## Separate pre-existing bridge hold

The [main preflight bridge](../../tools/parity/moss_audio/preflight_gate.py)
still declares project digest `af7486c52182a23cc64185e6ad95c42b8863673dc4e032fda79637a9ca0d427e`,
lock digest `fd8c1f2342da9512d1fb1e97ed0e7640ffa4a83db279aeaba655ce6cf89c1085`,
and Torch/TorchAudio 2.9.1. Current API-smoke declarations instead use the
project/corrected-lock digests above and Torch 2.13.0/TorchAudio 2.11.0.
This within-contract bridge mismatch predates #246. This minimal dependency
update does not alter it, restamp historical evidence or approve a new scope.
Any later coordinated bridge correction must preserve rejection of stale
hash-bound scope/evidence and obtain genuine approvals for the new inputs.

The [main-MOSS reconciliation](moss-audio-current-lock-reconciliation-2026-10-14.md)
is a separate contract. Source/model grants, package/native/API/owner review,
sidecars and real-weight CPU parity remain unresolved. No READY, Apple pass,
publication, owner signature or downloaded-artifact authentication is inferred.
No model/config/tokenizer acquisition, dependency sync/install, heavy Cargo,
paid worker or protected CosyVoice2 manifest access occurred. The 58 unresolved
public rows, frozen 63-row owner packet and final Apple packet remain unchanged.
