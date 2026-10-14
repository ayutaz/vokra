# Current 58-row prerequisite inventory

Read-only management inventory, 2026-10-14. No model/archive was acquired,
loaded or executed; no cloud worker was created; no approval, owner field or
protected manifest was changed or read.

## Boundary and verdict vocabulary

> Subsequent [exact software finding](wespeaker-setuptools84-static-finding-2026-10-14.md)
> confirmed a vendored LGPL payload in the locked setuptools wheel used by the
> current WeSpeaker reference closure. That known dependency-admission blocker
> supplements, rather than replaces, the initial per-row prerequisite snapshot
> below. It does not decide public withdrawal, replacement or owner approval.

The source is `/private/tmp/vokra-coverage-20261014.wkUg7j/coverage.tsv`,
SHA-256
`9ef3eb07f12a98de7f425dcd75b98f4cd1be45ab1d3907aafeca2626196cdd56`, with
194 data rows: 136 `full`, 43 `partial`, 14 `no-runtime-binder`, and one
`not-artifact`. Its revision column identifies each public repository's exact
metadata revision (retained in the tables below). The source classification
and preceding execution readback were captured at source HEAD
`bf1f1eb28361403366c882dda8577d60e0fdb271`; the checkout inspected during
this read-only pass was `d4ce1040b793fba1c75d7650caea503fac18fe29`, whose
post-`bf1f` changes are documentation-only. The denominator is therefore the
58 non-full rows in that exact TSV snapshot, not a fresh HF retrieval.

`NOT_PROVEN` below means the row's current exact execution approval and/or
VAST evidence was not established by this inventory. It is deliberately not a
claim that the row has been proven `READY=0`, nor does a source checker,
synthetic test, historical CPU result, or `full` catalog classification become
`VAST_READY` or `CPU_PASS_METAL_NOT_RUN` by association. `UNKNOWN_EXCLUDED` is
used only where a protected input was intentionally not inspected.

The immutable cross-row references used here are:

- Current metadata ledger: `docs/api-reference.md` (194/136/58 counts) and
  `docs/handoff/execution-readback-2026-10-14.md` (same TSV SHA and the
  distinction between catalog facts and execution readiness).
- Row-specific prerequisite ledger/checkers:
  `docs/handoff/mac-pre-scaleway-remaining-tasks-2026-09-05.md` (artifact-
  specific, bound-incomplete, generic no-binder and non-artifact sections),
  `docs/handoff/mac-cpu-metal-owner-disposition-packet-2026-09-07.md` (frozen
  owner-scope evidence, not current approval), and the exact paths listed in
  the tables below.
- Protected exclusion: the CosyVoice2 LLM manifest
  `tools/parity/cosyvoice2_llm_reference/license_gate_manifest.json` was not
  opened, hashed, edited, staged or used to infer a decision. The CosyVoice2
  row is consequently `UNKNOWN_EXCLUDED`, not `NOT_PROVEN` from manifest
  inspection.

## Summary

| Catalog class | Rows | First prerequisite pattern | Current execution conclusion |
|---|---:|---|---|
| `partial`, route/binder exists but released CPU forward/composite is incomplete | 22 | Complete the exact source/component/forward contract, then independent reference and a new execution-approved VAST scope | `NOT_PROVEN`; source route is not execution approval |
| `partial`, public artifact/replacement or provenance boundary | 21 | Obtain an authorized exact replacement or primary provenance/license/companion evidence; do not restamp the live artifact | `NOT_PROVEN`; old artifact is deliberately rejected |
| `no-runtime-binder` | 14 | Pin primary source/license and complete tensor/component contract, native binder/CLI and dependency review | `NOT_PROVEN`; no complete runtime route is established |
| `not-artifact` | 1 | Owner-authorized real artifact or exact withdrawal decision | `NOT_PROVEN`; no model artifact exists |
| **Total** | **58** |  | **No `VAST_READY` or current execution approval is inferred** |

## A. Partial: route/binder exists, but CPU forward/composite is incomplete (22)

The listed checker is a current source/preflight location only. None of these
locations is an owner approval or real-weight result.

| # | Repository | Existing exact checker/readiness location | First actionable prerequisite | Verdict |
|---:|---|---|---|---|
| 1 | `vokra/audioldm2` @ `5bbf9df2c5580afc62816a1349f178ea96a1caab` | `tools/parity/audioldm2_inspect.py`; `scripts/publish/vast-ai/run-audioldm2-inspection.sh` | Authenticate complete component tree/tensor manifest, dedicated reference lock, license/owner scope, then implement full diffusion composite | `NOT_PROVEN` |
| 2 | `vokra/audioldm2-large` @ `1f549d873359b09a1d2cad44dd63888eb29b1d73` | `tools/parity/audioldm2_inspect.py`; `tools/parity/audioldm2_large_prepare_checkpoint.py` | Bind Large-specific component/tensor identity and reference; do not reuse Base facts | `NOT_PROVEN` |
| 3 | `vokra/canary-qwen-2.5b` @ `4a894c430ed793144acdaa1b07410b071501cb82` | `tools/parity/canary_qwen_2_5b_inspect.py` | Authenticate FastConformer→adapter→Qwen shapes, tokenizer semantics, dataset/dependency/license scope | `NOT_PROVEN` |
| 4 | `vokra/chatterbox-multilingual-v3` @ `95c8bf4409c237de930c2eec0274fb2b99a21a09` | `tools/parity/chatterbox_family_inspect.py`; `tools/parity/test_chatterbox_worker_contract.py` | Authenticate full voice-encoder/S3/HiFT/watermark component manifest and policy/license boundary | `NOT_PROVEN` |
| 5 | `vokra/chatterbox-nano-v1` @ `49b2f3612ec3e479eb64ce49ab27ae82cbf0b206` | `tools/parity/chatterbox_family_inspect.py`; `tools/parity/test_chatterbox_worker_contract.py` | Same component closure, separately bound to Nano; no variant aliasing | `NOT_PROVEN` |
| 6 | `vokra/chatterbox-turbo-v1` @ `10fee774c6c5ed890e39cea76d0ae1a320f7a4eb` | `tools/parity/chatterbox_family_inspect.py`; `tools/parity/test_chatterbox_worker_contract.py` | Same component closure, separately bound to Turbo; no variant aliasing | `NOT_PROVEN` |
| 7 | `vokra/chattts` @ `2202eb0a2faa43e378af5fba01996e76860eacc5` | `tools/parity/chattts_inspect.py`; `tools/parity/chattts/uv.lock`; `tools/parity/chattts/pyproject.toml` | Complete GPT/Embed/DVAE/decoder/Vocos role manifest, native composite, AGPL/CC-BY-NC/dependency policy | `NOT_PROVEN` |
| 8 | `vokra/clap-htsat-fused` @ `b6c90b7eb353ba7fa6f25c5f2a278c555ef0d939` | `tools/parity/clap/clap_model_free_audit.py`; `tools/parity/clap/license_gate.py`; `scripts/publish/vast-ai/run-clap-htsat-fused-validation.sh` | Authenticate real state-dict role/shape manifest and native preprocessing/forward; then obtain dedicated dependency/owner review | `NOT_PROVEN` |
| 9 | `vokra/cosyvoice2-0.5b` @ `d707e0277e2a29e8fbd3972aec1b6a1cba0192ea` | Protected LLM manifest excluded; non-protected `scripts/verify/apple-silicon-cosyvoice2.sh` is not treated as closure evidence | `UNKNOWN_EXCLUDED`: obtain separately authorized model-free closure and owner approval without opening the protected manifest | `UNKNOWN_EXCLUDED` |
| 10 | `vokra/csm-1b` @ `81613fc840fa995f4c8f1c48749fd731ed6424b8` | `tools/parity/csm_1b_gate.py`; `tools/parity/csm_1b_inspect.py` | Prove codec tensor roles and CSM+Mimi composite, official API/reference, dependency/native and redistribution scope | `NOT_PROVEN` |
| 11 | `vokra/dia-1.6b` @ `dd1df2a129fed7d15c365caeabaae227ccfe8537` | `tools/parity/dia_1_6b_source_contract.py`; `tools/parity/dia_1_6b_reference/dependency_approval.py`; `tools/parity/dia_1_6b_validate_evidence.py` | Authenticate six-file tree, safe checkpoint mapping, 343-tensor roles and DAC composition before real parity | `NOT_PROVEN` |
| 12 | `vokra/firered-asr-aed-l` @ `3796c285b9e9f68ec9dc70cfe06e6b2b8f06cfb3` | `tools/parity/firered_asr_aed_l_gate.py`; `tools/parity/firered_asr_aed_l/license_gate.py`; `scripts/publish/vast-ai/run-firered-asr-aed-l-inspection.sh` | Close released-weight AED decoding/config/dependency provenance and independent CPU reference | `NOT_PROVEN` |
| 13 | `vokra/fun-cosyvoice3-0.5b-2512` @ `37e7d22a665d96dd7eb2e10e43ff4571783670cc` | `tools/parity/cosyvoice3_gate.py`; `tools/parity/cosyvoice3_source_audit.py`; `tools/parity/cosyvoice3_validate_reference.py` | Resolve the exact `soxr` closure and complete the full composite route | `NOT_PROVEN` |
| 14 | `vokra/irodori-tts-500m-v3` @ `28e3efaf41f0890784d88f4744c34269e80bdd41` | `tools/parity/irodori_inspect.py`; `tools/parity/irodori_text_block_reference/dependency_audit.py` | Find a fixed official codec/source route without the currently blocked dependency closure; then license/reference review | `NOT_PROVEN` |
| 15 | `vokra/kyutai-stt-2.6b-en` @ `c8f5779f1471f34734aafe1999082ca33862bc5e` | `tools/parity/kyutai_stt_2_6b_en_inspect.py`; `tools/parity/kyutai_stt_pytorch_pcm_source_receipt.json` | Join PCM Mimi encoder, tokenizer, KV/state lifecycle and scheduling semantics to the strict decoder | `NOT_PROVEN` |
| 16 | `vokra/moss-tts-local-transformer-v1.5` @ `1a613a7e0ef36d02e51f7b990bce05e4627ea8df` | `tools/parity/moss_tts_local/preflight_gate.py`; `tools/parity/moss_tts_local/model_free_audit.py`; `tools/parity/moss_tts_local/reference_validator.py`; `tools/parity/moss_tts_local/license_gate_manifest.json` | Bind the exact 48-kHz stereo Tokenizer-v2 companion and complete package/native review | `NOT_PROVEN` |
| 17 | `vokra/owsm-v4-medium-1b` @ `1c491e291b29ac922c5d660c05c134c1e3d67314` | `tools/parity/owsm_v4_medium_1b_inspect.py`; `tools/parity/owsm_v4_medium_1b_reference/dependency_license_gate.py` | Resolve authoritative `torch-complex` license evidence before frontend/full decoder reference work | `NOT_PROVEN` |
| 18 | `vokra/sortformer-diar-4spk-v1` @ `8cc3ad79fec93d1fa69cf04c56591d6bc26d804a` | `tools/parity/sortformer_diar_4spk_v1_inspect.py`; `scripts/publish/vast-ai/run-sortformer-diar-4spk-v1-inspection.sh` | Resolve mutable NeMo build provenance and bind full diarization topology/reference | `NOT_PROVEN` |
| 19 | `vokra/ultravox-v0-5-llama-3-2-1b` @ `ddbbeec5bfcb09c71a1f88971b794e3e5da811f9` | `tools/parity/ultravox/dependency_audit.py`; `tools/parity/ultravox/license_gate_manifest.json`; `tools/parity/ultravox/dump_reference.py` | Authenticate Meta companion digest/tokenizer/chat boundary and license/dependency closure | `NOT_PROVEN` |
| 20 | `vokra/vibevoice-1.5b` @ `dec190628f58928fc247b1205b9da2dabc58b9da` | `tools/parity/vibevoice_1_5b_inspect.py`; `tools/parity/vibevoice_1_5b_dump_reference.py` | Close Qwen tokenizer/prefill, streaming tokenizer, DPMSolver, full composite and independent real parity | `NOT_PROVEN` |
| 21 | `vokra/vibevoice-realtime-0.5b` @ `59831601f7f88c9ac0137a15a232ef850afb8781` | `tools/parity/vibevoice_realtime_0_5b_inspect.py`; `tools/parity/vibevoice_realtime_0_5b_reference/run_reference.py` | Authenticate streaming state/CFG/diffusion/acoustic decoder/tokenizer contracts and independent reference | `NOT_PROVEN` |
| 22 | `vokra/voxcpm-0.5b` @ `ee0ca6d6728c947ecf170e6711bdfbd6decaf0d5` | `tools/parity/voxcpm_0_5b_gate.py`; `tools/parity/voxcpm_0_5b_inspect.py`; `tools/parity/voxcpm_0_5b_reference.py` | Authenticate AudioVAE tensors, tokenizer and MiniCPM+DiT+CFM composite before VAST reference | `NOT_PROVEN` |

## B. Partial: public artifact/replacement or provenance boundary (21)

These are not “withheld” by inference. The live artifact is known to fail a
specific identity/provenance/companion gate, or the source facts are not yet
enough for a replacement. A replacement must be separately authorized and
rebound to current hashes; historical owner decisions do not transfer.

| # | Repository | Existing exact checker/readiness location | First actionable prerequisite | Verdict |
|---:|---|---|---|---|
| 23 | `vokra/audiogen-medium` @ `d537e1f92fc1665768be050c0e615c603717dc4e` | `tools/parity/audiogen_medium_reference/model_free_gate.py`; `tools/parity/audiogen_medium_reference/metadata_audit.py`; `tools/parity/audiogen_medium_reference/companion_contract.py` | Authenticate immutable T5-large and 16-kHz EnCodec companion/build identities, then replacement composite | `NOT_PROVEN` |
| 24 | `vokra/canary-1b-flash` @ `bc252b830cd34f732ea5d730576c783c60048189` | `tools/parity/canary_1b/preflight_gate.py`; `tools/parity/canary_1b/verify_reference_packet.py`; `tools/parity/canary_1b/license_gate_manifest.json` | Produce authorized complete 1,374-tensor replacement (AED decoder + tokenizer) from pinned checkpoint | `NOT_PROVEN` |
| 25 | `vokra/canary-1b-v2` @ `a2ed1680cfc8eb7523744c48695cdbe16f68976f` | `tools/parity/canary_1b/preflight_gate.py`; `tools/parity/canary_1b_v2_dump_reference.py` | Reject duplicate timestamp-auxiliary artifact; create authorized complete conversion with decoder/tokenizer | `NOT_PROVEN` |
| 26 | `vokra/conv-tasnet-libri1mix` @ `a030c16c8827e6580f639569b90f114ab762c7bb` | `tools/parity/conv_tasnet/license_gate.py`; `tools/parity/conv_tasnet/license_gate_manifest.json`; `tools/parity/conv_tasnet_dump_reference.py` | Resolve CC-BY-SA/WHAM license conflict and corrected kernel/stride provenance before replacement | `NOT_PROVEN` |
| 27 | `vokra/lang-id-voxlingua107` @ `0076abe0e94a3e7389137516df107d063238338d` | `tools/parity/speechbrain_lang_id/preflight_gate.py`; `tools/parity/speechbrain_lang_id/model_free_audit.py`; `tools/parity/speechbrain_lang_id/license_gate_manifest.json` | Regenerate authorized artifact containing ECAPA + XVector classifier + ordered 107 labels and exact axes | `NOT_PROVEN` |
| 28 | `vokra/mms-1b-all-base` @ `b3cde35ad11e0c67549ae4d45ab3f0c2dea026e5` | `tools/parity/mms_1b_all/license_gate.py`; `tools/parity/mms_1b_all/api_model_free_inspector.py`; `tools/parity/mms_1b_all/license_gate_manifest.json` | Obtain owner-approved language, complete backbone/adapter/vocab identities and composition contract; adapter-only public file cannot pass | `NOT_PROVEN` |
| 29 | `vokra/moss-audio-4b-instruct` @ `dbceb26a30dc4c187ea5da2ae607d14f2463e2ca` | `tools/parity/moss_audio/identity_audit.py`; `tools/parity/moss_audio/license_gate_manifest.json`; `tools/parity/moss_audio/dump_reference.py` | Correct 4B artifact stamp and authenticate tokenizer/chat/processor sidecars; then owner-approved real parity | `NOT_PROVEN` |
| 30 | `vokra/moss-audio-8b-instruct` @ `e194997c8d3a225fcfd6f9359ffbe7ce8668b871` | `tools/parity/moss_audio/identity_audit.py`; `tools/parity/moss_audio/license_gate_manifest.json`; `tools/parity/moss_audio/dump_reference.py` | Correct 8B artifact stamp and authenticate tokenizer/chat/processor sidecars; then owner-approved real parity | `NOT_PROVEN` |
| 31 | `vokra/moss-audio-tokenizer-nano` @ `0e1b26b38ad963f7787db63c9ea0799cf310373f` | `tools/parity/moss_audio_tokenizer_nano/license_gate.py`; `tools/parity/moss_audio_tokenizer_nano/source_contract_inspector.py`; `tools/parity/moss_audio_tokenizer_nano/license_gate_manifest.json` | Correct Nano provenance/variant artifact through authorized replacement; close dependency/license and first real-weight gates | `NOT_PROVEN` |
| 32 | `vokra/nsnet2` @ `983e1cc1397810201f93a121a9daf60cf247813b` | `tools/parity/nsnet2_dump_reference.py`; `tools/parity/nsnet2_reference/uv.lock`; `scripts/publish/vast-ai/run-nsnet2-validation.sh` | Resolve exact Microsoft source CC-BY-4.0 versus live MIT stamp, then authorized replacement | `NOT_PROVEN` |
| 33 | `vokra/qwen3-asr-0.6b` @ `dafd780826d041323b79a0444dca343fc0034aa2` | `tools/parity/qwen3_asr/preflight_gate.py`; `tools/parity/qwen3_asr/wheel_audit.py`; `tools/parity/qwen3_asr/license_gate_manifest.json` | Regenerate artifact with three execution keys and five authenticated sidecars under current lock/package scope | `NOT_PROVEN` |
| 34 | `vokra/qwen3-asr-1.7b` @ `7d160c2a319bfd2bf4b35274ba52af12b302ed62` | `tools/parity/qwen3_asr/preflight_gate.py`; `tools/parity/qwen3_asr/wheel_audit.py`; `tools/parity/qwen3_asr/license_gate_manifest.json` | Same contract, separately bound to 1.7B; do not reuse 0.6B artifact evidence | `NOT_PROVEN` |
| 35 | `vokra/qwen3-tts-12hz-0.6b-base` @ `69de218e2a6deafb67acaee565ad0e53e756f04e` | `tools/parity/qwen3_tts/dependency_audit.py`; `tools/parity/qwen3_tts/test_lock_pending_review.py`; `tools/parity/qwen3_tts/license_gate_manifest.json` | Current lock/package review and owner/operator decision, then authorized conversion with variant metadata/BPE sidecars | `NOT_PROVEN` |
| 36 | `vokra/qwen3-tts-12hz-0.6b-customvoice` @ `3f605b47383b79b82dc6ac0cff4dee7894b3baa2` | `tools/parity/qwen3_tts/dependency_audit.py`; `tools/parity/qwen3_tts/license_gate_manifest.json` | Correct CustomVoice name/variant metadata and BPE sidecars under a new exact scope; no old approval carryover | `NOT_PROVEN` |
| 37 | `vokra/qwen3-tts-12hz-1.7b-base` @ `b73ee6dd5eeef69ce4a797a48d8367cfed7e393d` | `tools/parity/qwen3_tts/dependency_audit.py`; `tools/parity/qwen3_tts/license_gate_manifest.json` | Correct 1.7B Base model/topology metadata and BPE sidecars under current lock and pending review | `NOT_PROVEN` |
| 38 | `vokra/qwen3-tts-12hz-1.7b-customvoice` @ `2f528f51b9ab27273929d12c3c2e22ec26315ee4` | `tools/parity/qwen3_tts/dependency_audit.py`; `tools/parity/qwen3_tts/license_gate_manifest.json` | Correct 1.7B CustomVoice metadata and BPE sidecars under current lock and pending review | `NOT_PROVEN` |
| 39 | `vokra/rmvpe` @ `3eb5fa8946f1074ba3959074c5cde95ec22b8c91` | `tools/parity/rmvpe_inspect.py`; `tools/parity/rmvpe/dump_reference.py`; `scripts/publish/vast-ai/run-rmvpe-validation.sh` | Resolve absent exact-source license; do not infer permissive MIT from live stamp | `NOT_PROVEN` |
| 40 | `vokra/sbv2-v2-jp-extra-base` @ `1234c6d41925f79307a1238c34c3188e562bcec9` | `tools/parity/sbv2_approval_preflight.py`; `tools/parity/sbv2_jp_extra/validate_contract.py`; `scripts/publish/vast-ai/run-sbv2-jp-extra-validation.sh` | Authenticate official bundle/production Japanese G2P and corrected metadata before replacement | `NOT_PROVEN` |
| 41 | `vokra/speechbrain-spkrec-ecapa-voxceleb` @ `3dc7704b5861e348edafc0d800291887a284a476` | `tools/parity/ecapa_tdnn_verify_reference.py`; `scripts/publish/vast-ai/run-ecapa-tdnn-validation.sh` | Replace corrupt out-of-bounds public artifact, then rerun strict replacement parity | `NOT_PROVEN` |
| 42 | `vokra/wespeaker` @ `a20ec15a61be1b5c5cb0f4805dbf72bb341e946f` | `tools/parity/wespeaker/preflight_gate.py`; `tools/parity/wespeaker/license_gate_manifest.json`; `scripts/publish/vast-ai/run-wespeaker-validation.sh` | Resolve Apache-vs-CC-BY-4.0 provenance/attribution and authorize exact replacement; current static collector is only a candidate | `NOT_PROVEN` |
| 43 | `vokra/yue-xcodec-mini` @ `83c14a67ed792a0d5b3b61fff8ae35a04c6da8fa` | `tools/parity/yue_xcodec_mini/preflight_gate.py`; `tools/parity/yue_xcodec_mini/reference_validator.py`; `tools/parity/yue_xcodec_mini/license_gate_manifest.json` | Obtain safe tensor-only/approved bundle route, acoustic/HuBERT/fusion/encode reference and RepCodec license decision | `NOT_PROVEN` |

## C. No runtime binder (14)

The common first prerequisite is a pinned primary source/license and complete
component/tensor contract. Existing inspectors/gates are fail-closed
preflight, not runtime approval.

| # | Repository | Existing exact checker/readiness location | First actionable prerequisite | Verdict |
|---:|---|---|---|---|
| 44 | `vokra/ace-step-1.5` @ `90715dd335e1998d31731f18ddbaf8b392cc71f4` | `tools/parity/ace_step_v15_gate.py`; `tools/parity/ace_step_v15_inspect.py` | Close license/dependency/dataset decisions and native DiT/LM/VAE component contract | `NOT_PROVEN` |
| 45 | `vokra/baichuan-audio` @ `5d41a238b74221c43b8a2fe1927c798e4c6ef8e0` | `tools/parity/baichuan_audio_instruct_gate.py`; `tools/parity/baichuan_audio_instruct_inspect.py` | Pin Matcha/source roles, MPL/PSF dependency policy and native composite | `NOT_PROVEN` |
| 46 | `vokra/granite-speech-4.1-2b` @ `3e1984c8a8e3278024768c8e399dc4846654c428` | `tools/parity/granite_speech_4_1_2b_gate.py`; `tools/parity/granite_speech_4_1_2b_inspect.py`; `scripts/publish/vast-ai/run-granite-speech-4-1-2b-inspection.sh` | Verify source/signature evidence, dependencies/datasets and native speech-language composite | `NOT_PROVEN` |
| 47 | `vokra/hibiki-2b` @ `19c22b7fb19c567d99862eff4671991ab8cd189d` | `tools/parity/hibiki_2b_gate.py`; `tools/parity/hibiki_2b_inspect.py` | Authenticate tensor roles and implement translation/demux/Moshi/Mimi contract | `NOT_PROVEN` |
| 48 | `vokra/htdemucs-multi` @ `23d1e2c9bd4662172bef1daf73dc0dcee58857a6` | `tools/parity/htdemucs_multi/audit.py`; `tools/parity/htdemucs_multi/dependency_audit.json`; `tools/parity/htdemucs_multi/license_gate_manifest.json` | Obtain tensor-only/state-dict route and owner disposition for native/data/license boundaries | `NOT_PROVEN` |
| 49 | `vokra/kimi-audio` @ `a9d3c39cf9c9e546be37bd4da595adb2d45a027f` | `tools/parity/kimi_audio_7b_instruct_gate.py`; `tools/parity/kimi_audio_7b_instruct_inspect.py` | Authenticate and bind the multi-component release; all model work remains VAST-only | `NOT_PROVEN` |
| 50 | `vokra/kyutai-tts-1.6b-en-fr` @ `f9620b81e2fafd4d1f6da6f75eea9f410ff6117b` | `tools/parity/kyutai_tts_1_6b_en_fr_inspect.py` | Authenticate 418 tensor roles/shapes, then bind demux/conditioners/Mimi and separate licenses | `NOT_PROVEN` |
| 51 | `vokra/qwen2-5-omni-7b` @ `ba080198d380554356f68501cd96cd25eb4e149e` | `tools/parity/qwen2_5_omni_7b_gate.py`; `tools/parity/qwen2_5_omni_7b_inspect.py` | Complete native multimodal streaming, dependency/license and independent parity contracts | `NOT_PROVEN` |
| 52 | `vokra/qwen2-audio-7b-instruct` @ `57421d5970ee83f2d5561888fd45476765ed27dd` | `tools/parity/qwen2_audio_7b_instruct_inspect.py`; `tools/parity/qwen2_audio_source_license_audit.py` | Resolve exact source-repository license and implement native audio-language path | `NOT_PROVEN` |
| 53 | `vokra/step-audio2-mini` @ `33cf2499217070b145934d75f05a0788d703c509` | `tools/parity/step_audio2_mini_gate.py`; `tools/parity/step_audio2_mini_inspect.py`; `scripts/publish/vast-ai/run-step-audio2-mini-inspection.sh` | Complete multi-component native S2S plus source/weight/dependency license closure | `NOT_PROVEN` |
| 54 | `vokra/vibevoice-asr` @ `b9715cd06abf7a0d837bc8eacea9ddfe0ce25cd3` | `tools/parity/vibevoice_asr_gate.py`; `tools/parity/vibevoice_asr_inspect_reference.py` | Close dataset/timestamp semantics, native ASR route and independent reference parity | `NOT_PROVEN` |
| 55 | `vokra/vieneu-tts-v3-turbo` @ `a7902f28e8e5e7527de5f21ca7ee30e91a383b56` | `tools/parity/vieneu_v3_turbo_gate.py`; `tools/parity/vieneu_v3_turbo_inspect.py`; `scripts/publish/vast-ai/run-vieneu-v3-turbo-inspection.sh` | Complete dependency/license/voice-cloning policy and composite runtime | `NOT_PROVEN` |
| 56 | `vokra/xtts-v2` @ `0ce1bd209ffa00f19ead08e54403d9ca2721b408` | `tools/parity/xtts_v2_gate.py`; `tools/parity/xtts_v2_inspect.py`; `scripts/publish/vast-ai/run-xtts-v2-inspection.sh` | Resolve CPML/source/dependency/consent gates and safe checkpoint layout before native composition | `NOT_PROVEN` |
| 57 | `vokra/xy-tokenizer` @ `20b5ddb81b038567485bd2b92e53ff5df536d323` | `tools/parity/xy_tokenizer_reference/audit.py`; `tools/parity/xy_tokenizer_reference/collect_evidence.py`; `tools/parity/xy_tokenizer_reference/license_gate_manifest.json` | Authenticate a real tensor manifest/topology, implement native route and regenerate the metadata-only artifact; current reference manifest is `BLOCKED`/`NO_UPLOAD` | `NOT_PROVEN` |

## D. Non-artifact (1)

| # | Repository | Existing ledger/readiness location | First actionable prerequisite | Verdict |
|---:|---|---|---|---|
| 58 | `vokra/seamless-m4t-v2-large` @ `75105667d1df57e8a4ab75239a9bfd4e6c2e84d7` | `docs/handoff/mac-pre-scaleway-remaining-tasks-2026-09-05.md` non-artifact section; no artifact checker can prove a missing GGUF | Owner-authorized real gated artifact or exact repository withdrawal; both are public-state changes and require separate authorization | `NOT_PROVEN` |

The report numbers are management identifiers, not coverage file row numbers;
the source TSV remains authoritative for repository/revision/class/reason text.

## Existing candidates outside the 58 denominator

Several named candidates are in the 136 `full` code/artifact catalog rows and
therefore are not members of this 58-row table: SGMSE, Voice Gender Classifier,
ReazonSpeech NeMo v2, BigVGAN, Parler-TTS, SpeechT5 and the corrected WeSpeaker
repository. Their existing checkers/receipts remain useful facts but do not
make them current Apple or owner `READY` rows:

- SGMSE: `scripts/publish/vast-ai/run-sgmse-voicebank-validation.sh` and
  `scripts/verify/apple-silicon-sgmse.sh`; ledger says VAST/source evidence is
  separate from final Apple CPU/Metal/no-fallback evidence.
- Voice Gender Classifier: `tools/parity/voice_gender_classifier_dump_reference.py`
  and the dated ledger entry record corrected CPU evidence; final Apple packet
  is still a separate gate.
- ReazonSpeech: `tools/parity/reazonspeech_nemo_v2/license_gate_manifest.json`
  and `tools/parity/reazonspeech_nemo_v2_dump_reference.py`; code/artifact-full
  is not an owner/Apple result.
- BigVGAN: `tools/parity/bigvgan/license_gate_manifest.json`,
  `tools/parity/bigvgan/vast_torch213_linux_audit.json`, and the corresponding
  Darwin audit; its manifest remains `OWNER_SIGNOFF_REQUIRED`/`NO_UPLOAD`.
- Parler-TTS: `tools/parity/parler_tts/license_gate_manifest.json` and
  `tools/parity/parler_tts/dependency_audit.py`; model/dependency rows remain
  pending operator decision.
- SpeechT5: `tools/parity/speecht5_tts/license_gate_manifest.json` and
  `tools/parity/speecht5_tts/dependency_audit.py`; dependency facts do not close
  operator/API smoke/owner approval.

Historical `CPU_PASS_METAL_NOT_RUN`, old owner scopes, and source-only green
checkers must be rebound to the exact current source/artifact/lock/API scope
before any real-weight request. This inventory therefore reports no current
`VAST_READY` row and no current `CPU_PASS_METAL_NOT_RUN` promotion, while
preserving the distinction between `NOT_PROVEN` rows and the intentionally
`UNKNOWN_EXCLUDED` CosyVoice2 row.
