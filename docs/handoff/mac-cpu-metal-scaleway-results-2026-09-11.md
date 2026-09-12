# Mac CPU / Apple Metal Scaleway results (2026-09-11)

**Status:** the bounded, authorized execution batch is complete. Every Apple
CPU/reference, Metal/reference, and Metal/CPU check listed below passed without
a CPU fallback. Operational cleanup and the follow-up PR are recorded
separately from this numerical verdict.

This dated record captures the bounded post-PR #79 VAST and Scaleway run.  It
does not claim that all public repositories are Mac-complete, does not change
the canonical 63-row unresolved-public denominator, and does not authorize a
Hugging Face upload or repository withdrawal.

## Scope and immutable boundaries

- PR #79 was merged as `1787818e702bdaba488d52aa1666fd5f08c5ae16` on
  2026-09-09.  The post-merge implementation branch was rebased onto GitHub
  `main` `38dfbc0d34ebd86a932634d286f27b4f5276e592`.
- The common runtime/Metal verification head is
  `58f8326242aefd72d845c3496ea7a49436f3248b`. SGMSE's final Apple consumer is
  at `bed6cc3f059adf77f4d7d7dbf1307aac13cf85d1`. Between those heads,
  `09f9b66d` only isolates Codex hook self-test fixtures, while `bed6cc3f` makes
  the Rust fixture consumer accept the same fixed, reviewed VAST
  producer/runtime/score-pair allow-list as the independent Python verifier;
  neither changes the SGMSE runtime implementation. After the successful
  SGMSE run exposed a shell-only sentinel spelling mismatch, `955f921d` aligned
  that verifier with the Rust test and added a drift regression; it did not
  change runtime code, weights, references, numerical bounds, or the accepted
  `bed6cc3f` result. The later results-documentation commit is not a new runtime
  or artifact identity.
- The owner and legal decisions are the exact-scope dispositions in
  [the 2026-09-09 decision record](mac-cpu-metal-owner-decision-record-2026-09-09.md).
  Withheld families were not executed.  BiCodec remained research-only and
  `NO_UPLOAD`; SpeechT5 execution remained separate from publication.
- No model was downloaded, converted, or executed on the maintainer Mac.
  Large-model and workspace work ran on disposable VAST workers; Apple
  execution ran on the disposable Scaleway host.
- No public artifact was uploaded or replaced during this batch.

## Exact VAST evidence

The full Linux workspace gate completed at
`b94977f2c8166f81966f6658a7666337ed0f4f63`.  It covered workspace all-target
tests, all-target Clippy with warnings denied, `cargo deny`, `cargo audit`,
formatting, all 52 Apple-worker contract self-tests, zero-dependency,
forbidden-symbol, fixture-EOL, and pipefail checks.  The recovered log SHA-256
is `9b77a16eb91b68d5a196446dd577f13327a9eabfc640d5c0c25d73877adc285e`.

The only tree change from that gate to the final code head was
`scripts/verify/apple-silicon-speecht5-tts.sh`.  At `58f83262`, its syntax and
self-test, all 52 Apple-worker contracts, formatting, zero-dependency,
forbidden-symbol, fixture-EOL, pipefail and diff hygiene passed again.  The
recovered final-static log SHA-256 is
`1bdd79e810ac7c8bfaeaeaead486299d493c6d9ff411d3a144945074f830ba55`.

The SGMSE producer-identity fix was then compiled and tested on VAST at exact
clean head `bed6cc3f059adf77f4d7d7dbf1307aac13cf85d1`: all three selected Linux
tests passed, including the reviewed CPU-79 and unknown-producer rejection
test. All 52 Apple-worker syntax/self-test contracts, formatting,
zero-dependency, forbidden-symbol, fixture-EOL and pipefail gates passed. The
recovered combined log SHA-256 is
`e3c0304e76db0a63d13c2adf535f6fbf8e5b23f8f4f6b3cf168c07cbc22a6e80`.

The Apple run itself then found a verifier-only mismatch: the successful Rust
enhancement test emitted `backend=cpu+metal`, while the shell post-check looked
for `backend=cpu,metal`. The underlying test completed one exact named pass and
wrote the complete evidence set, but the outer runner correctly remained
nonzero until the discrepancy was reviewed. At verifier-only head `955f921d`,
the corrected self-test and all 52 Apple-worker contracts passed on the same
Scaleway host. The recovered contract log SHA-256 is
`bc65d030c55ed04a1e41f3230e7281929036c28fa4af3329acb912648386d1c0`.
The fixed sentinel occurs exactly once in the preserved successful run log;
the obsolete spelling occurs zero times.

The clean final bundle was
`vokra-58f83262-from-1787818e.bundle`, requiring base
`1787818e702bdaba488d52aa1666fd5f08c5ae16`, with SHA-256
`a84c53c8d8fe9110ef3d5d58874585de1b9c63986d587fd6584a78e51a9a45fa`.
It was verified before each remote checkout.

The later SGMSE test-contract delta was
`vokra-bed6cc3f-from-58f83262.bundle`, requiring base
`58f8326242aefd72d845c3496ea7a49436f3248b`, with SHA-256
`dc50724b00b1dbab8982900d76e7715081ce472ecf9cd4d1e5ea3740f411edaa`.
It was verified on VAST before the public PR branch was used as the
Scaleway-side transport.

SpeechT5's authenticated Transformers 5.10.4 API smoke ran at the exact final
head and remained `NO_UPLOAD`.  Its evidence JSON SHA-256 is
`3d9bd68db8f2f1db288a0857f1f8d4aa840797f4b7b8edc5195899db0eabbdfe`.

SGMSE retained the earlier strict Linux enhancement execution at unchanged
runtime source head `855833c6`, where its authenticated 4,096-sample,
61-noise-call consumer passed with maximum absolute error
`3.332793712615967e-4` and RMSE `6.269547446627377e-5`. The current VAST host
completed the same native output before its attached shell timed out; an
independent comparison against the freshly generated exact-head packet passed
with maximum absolute error `6.735324859619141e-6` and RMSE
`7.381020032645416e-7`, at bound `0.01`. Score CPU/reference also passed with
real and imaginary maximum absolute errors `1.068115234375e-4` and
`8.96453857421875e-5`. The final score manifest SHA-256 is
`8f56f449a4da4b90161dbc6b719b522fbb95866da9010f624d6420433906bdcd`;
the `bed6cc3f` enhancement manifest SHA-256 is
`c2f3f14dc81cef73a53f693089031536af9c276bae2458d307d1166a1cef216c`.
All of these paths remained `NO_UPLOAD`.

## Apple host identity

The Apple worker was Scaleway Apple Silicon instance
`d2ce6428-82c9-438c-9640-bca76620b50c` (`M4-M`).  The captured environment was
macOS 26.6.1 build 25G76 / Darwin 25.6.0, Apple M4 with 10 physical and 10
logical CPU cores, 32 GiB RAM, a 10-core GPU, Metal 4, and Rust/Cargo 1.98.1.
The host checkout was clean for each accepted run.

## Accepted Apple results

| Scope | Exact code head | Result | Evidence identity |
|---|---|---|---|
| Metal backend package | `58f83262` | 21 library tests passed with one hardware-only ignore; BigVGAN 2, graph 8, kernel 53, and parity 5 passed.  The ignored SineGen device case was then selected by its fully qualified name and passed.  Unsupported graph operations remained explicit. | package log `6818eeea7759197fdb4633f291315bf359032251293bd36260dd8ec8398be0e0`; SineGen log `a7060f3cefe906f39017384ee3554cc3347a5a3015a42df8e8fa2a017cb91ea9` |
| Apple BF16 GEMM | `58f83262` | PASS; three cases, maximum absolute error `7.629394531e-6` against bound `1.0e-3`. | summary `73bb7df7c13f714278db7aaeb63f1b33949174b4eed72b024760d2abef893dfd` |
| SpeechT5-TTS | `58f83262` | CPU/official, Metal/official and Metal/CPU PASS at bound `0.01`; no upload. | summary `8ad0a6294dd4a2f309ae9125d6b802a5dd7817a7d79dfdc0d4a490589e872cd0`; parity log `57dd1b0a631fc5357fa1e10a6d2b9da6d87949ba87212dfbcfc413ab2eecfdcc` |
| ReazonSpeech NeMo v2 | `58f83262` | CPU/official encoder max abs `9.536743164e-6`; Metal/official `7.629394531e-6`; Metal/CPU `2.861022949e-6`; token and text decisions exact. | summary `20ff13e2a84b08a37866387d538b59736172cce7410e015fd8cb89332cfc79d9` |
| Voice Gender Classifier | `58f83262` | CPU/official, Metal/official, Metal/CPU and argmax-label agreement PASS at bound `0.01`. | summary `c2ced431905ca6575c6faac2d04f9443013721d30b15019c734e998108595dad`; parity log `719d65decf7937f8db20ac4a4186d902cd30d1d056bb91fc5bed3f837444a29f` |
| OmniASR CTC 1B | `58f83262` | CPU and Metal exact packet tests PASS; `NO_UPLOAD`. | result `a21fb479aed6d500f4e0b7d3617e2e19801881df44b191cd14b1975586117ae5`; CPU log `32fcbd81fc337b449d047247ef43f58a01088b463254f9e5deb1774ff4c67a34`; Metal log `45b982783b12c38761ddb62f3622ac1a739b8b8b75ec48ffea4795c2eb330b4e` |
| BiCodec / Spark-TTS 0.5B | `58f83262` | Research-only CPU/reference and Metal/reference passed all four measured stages; Metal/CPU passed for semantic latent, d-vector, prenet output and waveform.  CPU fallback was forbidden. | exact-head approval `e16db34a9a04ae5054f0ae24c1fbdaa45980220106cf3eb5acd3bc2394c77516`; summary `9a89214c337095c7da594adda99f332f669cec253bbe3f0408bf0de3801c3e27` |
| GigaAM v3 | `141e9319` | Approval-bound CPU and Metal validations PASS; publication `NO_UPLOAD`. | CPU summary `52b406f59a519dea4f912be9ccb78b103e28226d4322a5f5391a0658bb978ec3`; Metal summary `2267ab7c62924c830958f8cdf536738c3a1ccaee74d9a528ddc66e95ad276317` |
| GigaAM Multilingual | `141e9319` | Approval-bound CPU and Metal validations PASS; publication `NO_UPLOAD`. | CPU summary `9cab9fae8d9b75ff5cb1bcbb1abaeedff18695e883f04c9292ff037c6e43f5e9`; Metal summary `030556f3ed0bec0c5ae9d539bb6a2db887f6d3cce60d278129c053e94d750644` |
| SGMSE VoiceBank | `bed6cc3f` | Score and the 4,096-sample, 61-call full enhancement passed CPU/reference, Metal/reference and Metal/CPU at bound `0.01`; Metal was explicitly present. Enhancement maximum absolute errors were `6.794929504e-6`, `6.705522537e-6`, and `1.192092896e-7`, respectively. | combined run log `d4ff18a96557004d89a03557b94a5e8d534a064783f33433204364ae52848e7a`; score backend `19d8f813c9ceea08cdc7daebdc0442eb7204970f432fae8341474372bb1534d7`; enhancement backend `546ce4370aa712c3c09a82e6d81905a752cc4b8ede41288b635ccbf3cd101b49` |

GigaAM is deliberately reported at the exact approval-bound head recorded in
its input JSON rather than being restamped as a final-head result.  The later
code-head evidence above is likewise not retroactively attached to an older
artifact or approval packet.

SpeechT5 used public GGUF SHA-256
`f26019f5e2f7106d834b0b1fd4f66286839e000350caad169388467452c8dde0`
and official-reference manifest SHA-256
`92ae255ef216c9c346adcf624affae4775e57d401a65748bb35517b2775a44c4`.
Its measured maximum absolute differences were `1.197576523e-3` before and
`1.204490662e-3` after the postnet on CPU/official, `3.781318665e-4` and
`3.811120987e-4` on Metal/official, and `1.585602760e-3` Metal/CPU.

SGMSE score maximum absolute differences were `1.077651978e-4` and
`8.869171143e-5` for CPU/reference real and imaginary planes,
`1.096725464e-4` and `8.964538574e-5` for Metal/reference, and
`5.722045898e-6` and `6.482005119e-6` for Metal/CPU. The enhancement evidence
PCM SHA-256 values are
`2ce3af6242eca6a961323f0cb683c7b4d3b533d687f7f128d3f0e77dbad361c4`
for CPU and
`4dcaec50b2051cbbaa3babc12e84bdb819b36bc5e3c2c03834b4dee8b01415a1`
for Metal.

## Resource lifecycle

The final SGMSE transfer worker was VAST instance `50481005` with 200 GB of
storage. After packet transfer and small-evidence recovery, its temporary
Scaleway SSH key was removed, the instance was destroyed with its data, the
individual API returned `instances: null`, and its old SSH endpoint refused
connections. A full VAST inventory then found no remaining Vokra-labelled
instance or retained Vokra storage.

The Scaleway M4-M evidence was recovered and matched against the remote
SHA-256 values before cleanup. The console then reported successful permanent
deletion of instance `d2ce6428-82c9-438c-9640-bca76620b50c` and returned an
empty Apple Silicon server list. After the control-plane change propagated, a
fresh non-multiplexed SSH connection to the former endpoint timed out. No
Scaleway compute resource remains for this batch.

## Accounting and remaining work

These results close the prepared, authorized execution batch; they do not make
the entire public catalog complete.  The 63-row ledger continues to include
withheld, provenance-blocked, incomplete-runtime, no-binder, partial-composite
and non-artifact rows.  No withheld row was reclassified from source-only or
approval-blocked evidence, and a successful candidate packet is not treated as
a corrected public artifact before the separately authorized publication
workflow runs.

Numerical execution, evidence recovery, VAST destruction, Scaleway deletion,
and follow-up PR creation are complete. The follow-up is PR #88. Public
artifact reconciliation remained a separate owner-approved action at the time
this Apple execution record was first captured.

## Post-batch public artifact reconciliation (2026-09-12)

A later, explicit repository-scoped approval authorized the existing local
Hugging Face token to be passed only over encrypted SSH standard input to VAST
instance `50600828`. The token was not placed in a command argument, log, or
remote file. All four uploads used `scripts/publish/publish-one.sh --push`
after that script's dry-run completed successfully; no manual upload path was
used.

Before publication, SGMSE was replayed at exact clean consumer head
`bed6cc3f059adf77f4d7d7dbf1307aac13cf85d1`. Its strict 647-tensor GGUF bind,
independent score reference, native CPU score parity, official 4,096-sample
enhancement reference, and 61-call native CPU enhancement parity passed. The
enhancement comparison reported maximum absolute error
`1.3494491577148438e-4` and RMSE `1.569286825642952e-5` at bound `0.01`.
The package test reported 1,101 passed and 16 ignored; the workspace all-target
test, all-target Clippy, deny, audit, formatting, zero-dependency, forbidden
symbol, fixture-pin, bound-architecture, and no-dynamic-load gates also
passed. The SGMSE summary SHA-256 is
`f601ab76db543e3832594051bcac293d8d13671adceab6dcb978661f9840524e`,
and the orchestration log SHA-256 is
`5d4bc60a832e6ec8b5ba5e63668694451882add0c2b788cd8011eeb047dd80ba`.
No model was executed on the maintainer Mac.

The published identities independently returned by the Hugging Face model API
were:

| Repository | Published revision | Exact GGUF | Bytes | LFS SHA-256 | Live audit result |
|---|---|---|---:|---|---|
| `vokra/reazonspeech-nemo-v2` | `d626a5dc5ca3bf17ea4582f8f1641f93e35477c4` | `reazonspeech-nemo-v2.gguf` | 2,477,292,896 | `ff761a7bc04bed0f45d47535fcfc54a929d4b6aa2fb04c03160be60ec75ca35a` | CPU `full`; Metal `full` |
| `vokra/voice-gender-classifier` | `f1bb0985d62504dcead1012460ee045220f821a3` | `voice-gender-classifier.restamped.gguf` | 61,899,328 | `afb03696d8a640d5d701ea0c136bb065cac648cbfe905a5dcc4eae04e0769b1a` | CPU `full`; Metal `full` |
| `vokra/bicodec` | `9760a082df544265b2b6410581c5e4a3945c93e8` | `model.gguf` | 625,491,648 | `ed0ba92cac023a4bc8cb20d9c8328272e03336c9b9da0dfe1c97ec2f41092f84` | CPU `partial`; Metal `blocked-by-cpu` (CLI remains bounded) |
| `vokra/sgmse-voicebank` | `c37e93159b4129b2c582c44f8170b44cf6e3e531` | `sgmse-voicebank.gguf` | 262,470,272 | `173e4079c5af65eab1fda027ea55aad502cbd9012ee33f00c484167e72d36e8a` | CPU `partial`; Metal `blocked-by-cpu` (CLI remains bounded) |

The API snapshot log SHA-256 is
`2d030fedda3ed6661df3beaf3fd35976facd245cd65ac868ef191d144391fc7d`.
Dry-run log SHA-256 values were `1a8a1d5d658fc055c00caa1ac32aa2fae66f1e602b9bd467aef87f241933501c`
(ReazonSpeech), `20d00d21dd287dec8dc3c0bb7525bbe03dfac70c01030440ad8ed065df24ab8c`
(Voice Gender), `e7ca6db89c9c151e1ba09c3451b091d10445b476d540525ac9093bad0fc22f94`
(BiCodec), and `e5a65eeaaeb5e890870cf30d46831e963d4e24df5d0221da809c7493a57bdfd7`
(SGMSE). Push log SHA-256 values were
`19033f71737b1746f96d54f22bdd35185451c9f0ef4bf710b98fc52e319cd5d5`,
`83269fd88648504f2bb80cf80e4fe0144e64a82c2c08912be040ebb28065b1cc`,
`3a025577c65d57bce915fadc846129ffb8a533a12723da7bdfd966e539228e8c`,
and `b4c20d23fc972efde37e783d1a03f6eb2911d0c0b6673daf1e48d6560cee74a9`
in the same order; all four push status files contained `0`.

The revision-aware audit update is commit
`67ba18ecc2684f0e48f36a525f9f1ff70dab39cf`. It keeps each historical
artifact failure attached to its exact old revision, allows only the reviewed
replacement revision, and fails closed for an unknown revision or unexpected
GGUF filename. At that exact clean head, the focused SGMSE CLI diagnostic test
passed once, all 13 audit unit tests passed, and the live audit returned:

```text
public_repos=194
gguf_repos=193
gguf_files=198
cpu_code=full:133,no-runtime-binder:15,not-artifact:1,partial:45
metal_code=blocked-by-cpu:60,full:133,not-artifact:1
```

The exact-head status, CLI, Python, and live-audit log SHA-256 values are
`55326d648fa7c3f36c6a35e044ba0ae04535f147236ef0da46f1df6ac025b5d2`,
`b6068e8a55637c9cdcfce9286c9461fba8202d55c14b2858671c59c00755409c`,
`eb55bdeb232a5345b812b14974fa3803a4265c20b50cb28f1fb75acf884da36e`,
and `b67a5bd8d6cbc211fd4fa99759ebee1a834f8b54d07917ae96b41c78df19e07a`,
respectively. Publication corrected artifact identity; it did not invent a CLI
route. Consequently ReazonSpeech and Voice Gender move to `full`, while
BiCodec and SGMSE remain `partial` until their explicit bounded CLI work is
implemented.

The final small-evidence archive is
`/private/tmp/vokra-vast-50600828-evidence/vokra-public-evidence-final.tar.gz`
(470,788 bytes), SHA-256
`dd0bb555da0b3cf10561906f42201ef389eaeaf7b9e09e608cd44f44419ec36d`;
its local digest matched the remote digest before cleanup. VAST instance
`50600828` (`vokra-public-artifact-reconcile-20260911`) was then destroyed
with its 200 GB instance storage. Its individual API returned
`instances: null`; the full instance and standard volume inventories both
returned `[]`. The available API key lacks the `machine_read` permission needed
to enumerate network disks, so this record does not claim a network-disk
inventory result. The previously deleted Scaleway M4-M instance was not
reprovisioned for publication.
