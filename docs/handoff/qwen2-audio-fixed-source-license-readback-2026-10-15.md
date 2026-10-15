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

- [Fixed commit](https://github.com/QwenLM/Qwen2-Audio/commit/595360e82b5839c1507492ec83cae5bda6d5c7d4)
  binds Git tree `0d66378d9e4b1a4d6bdba3969aa87ef777f487cd`.
- [That exact recursive tree](https://api.github.com/repos/QwenLM/Qwen2-Audio/git/trees/0d66378d9e4b1a4d6bdba3969aa87ef777f487cd?recursive=1)
  returns `truncated=false`, 23 entries and 19 blobs. No path matches
  license/licence, copying, notice, copyright or EULA filename terms.
- The [fixed README](https://raw.githubusercontent.com/QwenLM/Qwen2-Audio/595360e82b5839c1507492ec83cae5bda6d5c7d4/README.md)
  routes model-license questions to each model's repository. This does not
  establish an explicit source-code grant.

At 08:01 UTC on 2026-10-15, PR #254's link checker received HTTP 403
for this commit's REST endpoint. A fresh connector read independently
returned the same full commit identity and its official HTML permalink;
the HTML page was also inspected. The reader-facing commit link above
therefore uses that exact permalink rather than the API endpoint. The
original API-derived tree facts, license uncertainty and all other links
remain unchanged. This is not a general 403 exemption or a legal approval.

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

## Separate Transformers-only candidate, not a grant substitution

Subsequent read-only investigation at Vokra main `627a641f` distinguishes
source-role inspection from reference execution. The current inspector checks
QwenLM README/demo/eval files as source evidence; it does not import or execute
them. There is no complete Qwen2 Audio reference dumper, and the native
converter remains an inspection-only placeholder. The worker still stops at
the source-license gate before source/model acquisition.

The manager and Luna retrieved exact-revision files using the GitHub connector
for `huggingface/transformers@2ef31dec1676249d26044a8aa8abe33dbecf0d10`
(`v4.45.0`), not mutable `main` documentation:

| Path relative to `src/transformers/models/` | UTF-8 bytes | Git blob SHA |
|---|---:|---|
| `qwen2_audio/configuration_qwen2_audio.py` | 8,530 | `deb276f334723c5c8918cfd4695211ac85e0b73e` |
| `qwen2_audio/modeling_qwen2_audio.py` | 69,111 | `bf48e1c6a97ef97d30aebf73d30b8ac354c65c11` |
| `qwen2_audio/processing_qwen2_audio.py` | 8,633 | `eabf5b7069f2008604149c0a3f8c53d89c871de0` |
| `whisper/feature_extraction_whisper.py` | 14,895 | `a79eeedd0a295c744ee16405582002a0cd77ed19` |
| `qwen2/configuration_qwen2.py` | 10,100 | `20ebfb0e28572606c2a191c6c00ee7f0ba51c416` |
| `qwen2/modeling_qwen2.py` | 70,051 | `10c0b6f38669da83bc4e9da6ff5dad5122cdd051` |
| `qwen2/tokenization_qwen2.py` | 13,913 | `be2685430f649eab8bde99f217597afd282337c5` |
| `qwen2/tokenization_qwen2_fast.py` | 5,182 | `fcfc4ab764da45018634ff3e9eb16ef9f186643f` |

The separately retrieved [fixed root LICENSE](https://github.com/huggingface/transformers/blob/2ef31dec1676249d26044a8aa8abe33dbecf0d10/LICENSE)
is 11,418 UTF-8 bytes, Git blob
`68b7d66c97d66c58de883ed0c451af2b3183e6f3`. Apache-2.0 source headers and
that license apply to these Transformers sources only, not the QwenLM tree.
Blob identity/byte counts are not newly generated SHA-256 archive receipts.

The fixed modeling source has `Qwen2AudioEncoder`,
`Qwen2AudioMultiModalProjector` and `Qwen2AudioForConditionalGeneration`; it
does **not** define `Qwen2AudioModel`. The production inspector requires only
the conditional-generation marker for this modeling role. A preliminary agent
class-list description was corrected after exact-source manager review; it
was never a production requirement or a defect in the inspector.

The constructor selects its audio tower through `AutoModel.from_config`,
creates the multimodal projector, and selects the language model through
`AutoModelForCausalLM.from_config`. The processor/loader uses the audio feature
extractor and tokenizer paths. Auto registries, PyTorch, NumPy, audio utilities
and fast-tokenizer dependencies therefore remain part of the candidate closure;
the six inspected role files alone are not a full executable environment.

This supports investigation of a different official source provider, not a
completed or approved reference. Exact model/config/tokenizer identities,
prompt/chat and preprocessing semantics, security of the fixed source version,
full dependency/native review, owner approval, native conversion/runtime and
independent real-weight parity remain unresolved. Do not silently remove the
QwenLM source role or infer execution rights from this candidate. No package
or source body was imported/executed, model asset acquired, approval changed,
paid worker started, or row promoted by these additional source reads.

## Known version advisories are a separate admission hold

On 2026-10-15 JST, Luna searched primary GitHub advisory/Hugging Face security
sources; the manager independently read the following reviewed advisories.
Their reported package-version ranges include the declared source version
4.45.0. This is not an installed-environment or exploitation result:

| Primary advisory | Reported affected / fixed version | Scope and retained uncertainty |
|---|---|---|
| [GHSA-29pf-2h5f-8g72 / CVE-2026-4372](https://github.com/advisories/GHSA-29pf-2h5f-8g72) | `<5.3.0` / `5.3.0` | Describes malicious configuration influencing downloaded/executed attention code through standard model loading, including a remote-code trust bypass. Actual reachability through the prospective fixed Qwen2 Audio constructor/loader remains unreviewed; do not claim a safetensors-only policy removes this configuration risk. |
| [GHSA-69w3-r845-3855 / CVE-2026-1839](https://github.com/advisories/GHSA-69w3-r845-3855) | `<5.0.0rc3` / `5.0.0rc3` | Describes Trainer RNG-state checkpoint loading, with PyTorch-version and operation conditions. It is not proof that a Qwen2 Audio inference-only candidate uses Trainer or reaches that operation. |

The [official security policy](https://github.com/huggingface/transformers/security)
recommends explicit safetensors loading and fixed revisions for remote-code
review. These controls do not by themselves close configuration, regex, path
or dynamic-code questions. An empty repository advisory UI also does not negate
reviewed records in the separate GitHub Advisory Database.

Advisory coverage and the full candidate import/call closure were not proved
complete. Each listed fixed version addresses that particular record; it is
not a recommendation or approval of a final dependency version. Do not enable
the old candidate, silently bump/retarget its source, weaken the gate or run a
paid model test to replace this review. Authenticate any proposed maintained
source/loader route and its changed package/native/legal/API scope first, then
obtain exact approvals and independent measurements. The current worker still
refuses before model/source acquisition; no candidate reference installation,
model execution, exploit test or new security exception was performed. The
ordinary documentation pre-commit hook later synchronized its cached software
environment; see the [OWSM management-hook boundary](owsm-audit-admission-readback-2026-10-14.md#fixed-frontend-has-no-native-complex-selection-flag--2026-10-15-jst).
