# Mac CPU / Metal owner and legal decision record (2026-09-09)

This record is the exact-scope owner/legal disposition for the post-PR #79
campaign.  It is separate from model execution evidence, VAST results,
publication approval, and any public-repository withdrawal.  A decision below
does not authorize Hugging Face upload.

The owner explicitly authorized the agent to record these decisions using
`signer=yousan` at `2026-09-09T03:49:54Z` UTC
(`依頼者許可 = agent 判断`). Each decision is bound to the complete canonical
scope SHA-256; changing a source revision, model identity, dependency closure,
native payload, or manifest requires a new decision.

## Decisions

| Family / scope | Decision | Signer / authorization timestamp (UTC) | Exact scope SHA-256 | Boundary and reason |
|---|---|---|---|---|
| SpeechT5-TTS + HiFi-GAN | `APPROVE_COMMERCIAL_EXECUTION` | `yousan` / `2026-09-09T03:49:54Z` | `99116b392c560ec40c574589305492f35d9d30e8e2f44a9c03392885c77e85ba` | The fixed Microsoft SpeechT5 and HiFi-GAN primary model sources are MIT, and the reviewed source/tool boundary is Apache-2.0 or first-party. The operator approval is exact-scope only. The Transformers route remains `BLOCKED_UNVERIFIED_API_SMOKE`; this record does not make the family `VAST_READY` and does not authorize publication. Primary sources: <https://huggingface.co/api/models/microsoft/speecht5_tts/revision/30fcde30f19b87502b8435427b5f5068e401d5f6> and <https://huggingface.co/api/models/microsoft/speecht5_hifigan/revision/bb6f429406e86a9992357a972c0698b22043307d>. |
| BiCodec / Spark-TTS 0.5B | `APPROVE_RESEARCH_ONLY_EXECUTION` | `yousan` / `2026-09-09T03:49:54Z` | `bicodec-native-parity-owner-scope-v1` (immutable identity below) | The fixed checkpoint is CC-BY-NC-SA-4.0. Execution is research-only and no-upload; the existing runtime scope is decode-only and does not approve a missing PCM encode route or replacement publication. The exact-head approval JSON is generated after the final clean commit by `tools/parity/bicodec_owner_approval.py`, then checked by `run-bicodec-native-parity.sh`. Primary sources: <https://huggingface.co/api/models/SparkAudio/Spark-TTS-0.5B/revision/642071559bfc6346c2359d19dcb6be3f9dd8a05d> and <https://github.com/SparkAudio/Spark-TTS/blob/2f1ea9082400547242641f5271b6f941c9f439d1/LICENSE>. |
| Qwen3-ASR 0.6B / 1.7B | `WITHHOLD_EXECUTION` | `yousan` / `2026-09-09T03:49:54Z` | `da581832351b223b890814c0bf45ba036174da24dd7fd47a58236c1dde33ced1` | The family model license is not being re-decided here. The exact execution scope still has unresolved dependency/package rows and therefore must not acquire, convert, or execute real weights. No upload or withdrawal is authorized. |
| Qwen3-TTS four variants | `WITHHOLD_EXECUTION` | `yousan` / `2026-09-09T03:49:54Z` | `44daa1a9191e73a615e3ca32134a66a3619704b125587784afb4186a42b4f989` | The candidate loader and dependency closure require a fresh exact-head audit; stale/unresolved rows remain. No real-weight execution, upload, or withdrawal is authorized. |
| MOSS Audio 4B / 8B | `WITHHOLD_EXECUTION` | `yousan` / `2026-09-09T03:49:54Z` | `cdda6be3a60e4b703c0d26d249a21703b69c2024f10bfabedf4896e2d7bb2b9a` | Fixed source/model revisions do not provide a complete license file/redistribution closure and dependency/operator rows remain unresolved. The card's `apache-2.0` metadata is provenance, not sufficient approval. |
| Ultravox + Meta Llama companion | `WITHHOLD_EXECUTION` | `yousan` / `2026-09-09T03:49:54Z` | `35e73acdfdffa729464400a11cdc2f890b216dc59476a79acebd24bfe8ae555b` | The adapter's MIT declaration does not settle the gated Meta Llama companion's conditional license or the complete payload/dependency closure. Do not execute the composite. |
| WeSpeaker corrected replacement | `WITHHOLD_EXECUTION` | `yousan` / `2026-09-09T03:49:54Z` | `0133cb13d4869903f89d6bbcaee9e784a69cf158804ae76bf4a52c7a0ca3efd0` | Replacement/source/checkpoint rows remain pending and the exact replacement-versus-withdrawal disposition is not complete. |
| YuE XCodec Mini | `WITHHOLD_EXECUTION` | `yousan` / `2026-09-09T03:49:54Z` | `6f8378213db1ef19924c42cb76a194ed013093c048aba910808eb14e2dcef262` | The source license and mixed MIT / CC-BY-NC RepCodec boundary remain unresolved, including the public artifact and package rows. |
| Parler-TTS English / Multilingual external/documented scope | `WITHHOLD_EXECUTION` | `yousan` / `2026-09-09T03:49:54Z` | `18876e4d8198e76da3bd220e44033ca6754cca93dfd226a6b44c5551c9af329e` | Model-free closure evidence is not an operator decision; the exact source/model/DAC review and operator disposition remain pending. |
| BigVGAN four variants | `WITHHOLD_EXECUTION` | `yousan` / `2026-09-09T03:49:54Z` | `73f8b60a0f71be420dfbaf1fc7213743701816a301303ff98ba46bbf2d09bce4` | The hash-bound Linux/Darwin package, license, and native-payload review still requires an explicit operator decision. Existing MIT model rows do not approve this wider execution scope. |

`WITHHOLD_EXECUTION` keeps each row in the campaign denominator and is not a
withdrawal. It authorizes neither deletion nor modification of any public
repository or artifact.

## BiCodec immutable approval identity

The external JSON generated for an exact clean HEAD has the following fixed
fields. `git_commit` and the resulting `scope_sha256` are intentionally
materialized outside the repository after the final commit so the approval
cannot become stale by committing itself:

```text
schema=vokra-bicodec-approval-v1
model=SparkAudio/Spark-TTS-0.5B
upstream_repo=https://github.com/SparkAudio/Spark-TTS
upstream_revision=2f1ea9082400547242641f5271b6f941c9f439d1
upstream_hf_revision=642071559bfc6346c2359d19dcb6be3f9dd8a05d
license_spdx=cc-by-nc-sa-4.0
checkpoint_sha256=e9940cd48d4446e4340ced82d234bf5618350dd9f5db900ebe47a4fdb03867ec
config_sha256=744f4093ae2381a2eb44ea8c4a5268a8d1e581498e9bf0808c034d1b076429be
no_upload=true
decision=RESEARCH_ONLY
signer=yousan
```

Generate and validate it only after fixing the final clean HEAD:

```text
uv run --no-project --python 3.12 python tools/parity/bicodec_owner_approval.py \
  --repo-root /absolute/path/to/vokra \
  --output /absolute/path/to/external/bicodec-approval.json
uv run --no-project --python 3.12 python tools/parity/bicodec_owner_approval.py \
  --validate /absolute/path/to/external/bicodec-approval.json \
  --expected-head <exact-clean-head>
```

The VAST worker still independently validates this schema, every immutable
identity, `git_commit`, `scope_sha256`, `signer`, `no_upload`, and the existing
repository BiCodec sign-off. A generated approval JSON is evidence for the
authorized VAST run only; it is not an upload permission.

## Remaining 63-row denominator

The 63-row figure is the canonical audit denominator at baseline
`1787818e702bdaba488d52aa1666fd5f08c5ae16` on repository `vokra`, branch
`feat/mac-cpu-metal-post-79`. The exact ledger SHA-256 is
`69799be46b64b65f715faf07f42220a84a82cc515e655a3e025d48aece4fd3bf`. The row
identity and fixed source, artifact, or repository boundary remain those recorded in
[`mac-pre-scaleway-remaining-tasks-2026-09-05.md`](mac-pre-scaleway-remaining-tasks-2026-09-05.md).

### Aggregate owner decision

For this immutable ledger only, the default disposition is
`WITHHOLD_EXECUTION` for every row not listed in the exact exception set below.
This is not a claim that all 63 rows have the same technical or legal defect:
the exceptions retain their own explicitly recorded state and are not newly
withheld.

| Exact ledger row | Aggregate exception disposition | Boundary retained |
|---|---|---|
| `vokra/bicodec` | `APPROVE_RESEARCH_ONLY_EXECUTION` | CC-BY-NC-SA-4.0 research-only, decode/native parity scope, `NO_UPLOAD`; the separate exact-head approval JSON is required before VAST. |
| `vokra/sgmse-voicebank` | `RETAIN_EXISTING_CPU_PASS_METAL_NOT_RUN` | Existing exact VAST CPU/reference evidence remains valid for its recorded scope; only the final Apple CPU/Metal/no-fallback evidence remains. |
| `vokra/voice-gender-classifier` | `RETAIN_EXISTING_CPU_PASS_METAL_NOT_RUN` | Existing corrected artifact and exact VAST CPU parity remain valid; regenerate the Apple packet and run the final Apple evidence. |
| `vokra/reazonspeech-nemo-v2` | `RETAIN_EXISTING_HISTORICAL_CPU_PACKET` | The historical CPU packet/state is retained as ledger evidence; the current exact-head VAST packet and Apple evidence remain open. |

The prepared Apple-worker set outside the 63-row aggregate retains its existing
state and is not newly withheld: **GigaAM v3, GigaAM Multilingual, OmniASR CTC
1B, ReazonSpeech NeMo v2, BiCodec, Voice Gender, and SGMSE**. ReazonSpeech,
BiCodec, Voice Gender, and SGMSE are named above because they also occur in the
ledger or have an explicit current exception; the other prepared entries are
outside this 63-row denominator.

The aggregate decision is reproducible from this canonical JSON object. The
scope hash is SHA-256 of its UTF-8 JSON serialization with lexicographically
sorted keys and no insignificant whitespace; the `aggregate_scope_sha256` field
is excluded from the hashed object:

```json
{
  "baseline_commit": "1787818e702bdaba488d52aa1666fd5f08c5ae16",
  "exceptions": [
    {"disposition": "APPROVE_RESEARCH_ONLY_EXECUTION", "id": "vokra/bicodec"},
    {"disposition": "RETAIN_EXISTING_HISTORICAL_CPU_PACKET", "id": "vokra/reazonspeech-nemo-v2"},
    {"disposition": "RETAIN_EXISTING_CPU_PASS_METAL_NOT_RUN", "id": "vokra/sgmse-voicebank"},
    {"disposition": "RETAIN_EXISTING_CPU_PASS_METAL_NOT_RUN", "id": "vokra/voice-gender-classifier"}
  ],
  "ledger_path": "docs/handoff/mac-pre-scaleway-remaining-tasks-2026-09-05.md",
  "ledger_sha256": "69799be46b64b65f715faf07f42220a84a82cc515e655a3e025d48aece4fd3bf",
  "no_upload": true,
  "no_withdrawal": true,
  "rule": "Every other row in the exact 63-row ledger is WITHHOLD_EXECUTION; listed exceptions retain their explicit disposition and are not newly withheld.",
  "signer": "yousan",
  "timestamp_utc": "2026-09-09T03:49:54Z"
}
```

The resulting `aggregate_scope_sha256` is
`f691c330cdb5017f76b213e0d42ed318d50ce3e623e4be816692bfadc2dc579a`.
This aggregate decision is `NO_UPLOAD` and `NO_WITHDRAWAL`; a public artifact
update or repository withdrawal still requires separate authorization.

| Ledger class | Count | Accounting boundary |
|---|---:|---|
| Public-artifact-specific blocker | 27 | Recorded `vokra/<slug>` identity remains blocked by its specific provenance, artifact, unsafe-loader, or license boundary. |
| Bound but incomplete runtime | 19 | Recorded fixed repository/source identity remains inspection-only until its native composite, dependency/license closure, independent reference, and VAST CPU evidence exist. |
| Generic no-runtime-binder | 14 | Recorded repository remains unsupported for execution; no converter, shape, source, dataset, or license fact is inferred from a sibling. |
| Routed but intentionally partial composite | 2 | `vokra/csm-1b` and `vokra/ultravox-v0-5-llama-3-2-1b` remain partial composites. |
| Non-artifact repository | 1 | `vokra/seamless-m4t-v2-large` remains empty; replacement or withdrawal requires separate repository-scoped authorization. |

The individual exact decisions above take precedence over this denominator
table. In particular, BiCodec is research-only approved, and existing
separately evidenced `CPU_PASS_METAL_NOT_RUN` rows such as SGMSE and Voice
Gender, together with the prepared Apple-worker set, are not newly withheld by
this accounting section. Only rows with an unresolved boundary retain a
withhold decision; no claim is made that all 63 rows share that decision.
