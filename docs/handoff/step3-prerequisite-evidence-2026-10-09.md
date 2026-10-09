# Step 3 prerequisite evidence — 2026-10-09

この記録は「3まで」の進捗であり、実weight変換・独立reference・CPU parityの
完了記録ではありません。参照baselineは受理済みmain
`7025f17f8d18883bec5a3d3a8c1494a2e81b0993`です。

> **2026-10-09 03:42:53 UTC acceptance supersession:** #220は通常のsquash
> mergeで受理され、mainは `2c9702d4b2dc602c0afc60ab6c75a442003a3d59` に
> 進みました。fresh exact-head CIは69 SUCCESS / 1 SKIPPED、failed/pendingなし。
> strictな16 required context/application pairsとadmin保護を確認しました。
> reviewed head `e0b9108de100ba6af1052149297591338160bb21` とaccepted mainの
> treeはともに `fc23c1003a1337f5ac68eb154f9085a74af4c993`。root mainは
> fast-forward後cleanです。以下の「#220 pending」は取得時点の履歴であり、
> 現在の未受理状態ではありません。#152の更新・実weight/owner gatesは別途未完です。

## PR acceptance and current CI

- [#218](https://github.com/ayutaz/vokra/pull/218)は2026-10-09 03:05:11 UTCに
  通常のsquash mergeで受理されました。reviewed candidate
  `92109d8bee9fe7337b81d4133bf0ef9195acdc9c`のCIは69 SUCCESS / 1 SKIPPED、
  failed/pendingなし。strictな16 required context/application pairsが成功し、
  admin bypassや保護設定の変更はありません。
- accepted mainとcandidateのtreeはともに
  `b11b0825a646b97e76e70274b846c3c1c9e59688`。
  [#219](https://github.com/ayutaz/vokra/pull/219#issuecomment-6073459185)は
  取り込み済みとしてCLOSED、個別mergeではありません。元HEADのmain ancestryを
  squash結果から推測しません。
- root checkoutはこの受理済みmainへfast-forwardしcleanです。既存の計画文書差分は
  accepted blobと一致し、同じ内容のscoped stashを保全しました。
- [#182の既存コメント](https://github.com/ayutaz/vokra/pull/182#issuecomment-6073128553)
  に、`a3afc1ecc38db67434c564bbea424e0366e0cdd9`のfresh CI結果を追記しました。
  RESTの118 checks（100 + 18）は105 SUCCESS / 13 SKIPPED、failed/pendingなし。
  35 collector unittestと5 direct lock checksはmanager-localのsynthetic evidenceで、
  hosted `python-parity-oracles`がそのnested suiteを実行したという主張ではありません。
  #182はDraft、#169もOPENを維持し、closure/license/owner/real-weight gatesは未完です。
- 新着[#220](https://github.com/ayutaz/vokra/pull/220#issuecomment-6073610368)は
  head `e0b9108de100ba6af1052149297591338160bb21`のurllib3-only lock更新です。
  reviewed preparation時点でfresh CIは実行中、まだ受理していません。
  Draft #152の古い2.7.0-bound evidenceを2.8.0の証拠とみなしません。

## Authentic software wheel facts, not model execution

Qwen3-TTSのcurrent lock SHA-256は
`98cef03a391c9a31116b0b63c1dca4bb456c1a58ec407a4567e78e648eaf8857`。
以下4件の公式software wheelを通常curl（config/proxy/redirectなし）で取得し、
固定lock hash、HTTP size、ZIP、METADATA/WHEEL、全RECORDのhash/sizeを検証しました。
archiveを展開・install・importせず、native codeやモデルを実行していません。

| Locked wheel | Bytes | SHA-256 |
|---|---:|---|
| Torch 2.13.0 / cp312 macOS arm64 | 111,213,045 | `2fe228aba290d14b9f31b049be550dbd469c3fd3013d7a19705b30454da97027` |
| Torch 2.13.0+cpu / cp312 Linux x86_64 | 191,817,609 | `4ca4a9394b0c771238a4f73590fdbbc4debad85ed0fa63d026ae1b085da7d6e2` |
| TorchAudio 2.11.0 / cp312 macOS arm64 | 684,226 | `a1cf1acc883bee9cb906a933572fed6a8a933f86ef34e9ea7d803f72317e8c1b` |
| TorchAudio 2.11.0+cpu / cp312 Linux x86_64 | 341,338 | `2354248848d06a9ae1e7a12165f800f0dda7df60ecac9fca892322b722b922c0` |

bounded local reportは7,882,939 bytes、SHA-256
`cf7f6cc835173265d7757522925f7710d12f71e02647b1e740b754c4700351e2`。
内容はprimary license本文、member/native hashとRECORD factsです。
archive合計304,056,218 bytesは検査後に削除し、root readbackでowned archive directoryが
空、partial reportなしと確認しました。full runnerのDarwin peak RSSは60,522,496 bytes。
managerの32 synthetic tests、因果assertion強化後の11 transport testsもPASSです。

stdlib urllibのHEADはHTTP403、同じprimary URLへの通常curl HEADは200でした。
今回のcurl full GET成功は取得経路を前進させますが、UA/TLS等の原因確定でも、
別のVAST実行のHTTP403解消でもありません。古い失敗receiptは保全しました。

## Remaining exact-scope gates

- Torch root LICENSEのBSD-family本文、TorchAudioのBSD-2-Clause本文を取得しました。
  Torchのcomposite metadata expressionだけでvendored/native全体を承認しません。
- Torch wheelには `cpr/test/LICENSE` のGPL-3.0本文があります。そのファイルだけで
  Torch全体がGPL、またはnativeにGPL codeがlinkedしているとは断定しません。
  cpr runtime/testのsource・build境界と適用条件は別途確定が必要です。
- Darwin TorchにはLLVM/conda-forgeのOpenMP license本文と `libomp.dylib` があり、
  Linux Torchには `libgomp.so.1` があります。filenameはexact binaryの由来や
  GCC runtime exceptionの適用、ELF NEEDED/Mach-O load commandsを証明しません。
- Darwin TorchAudioの `torchaudio/.dylibs/libc++.1.0.dylib` は1,171,312 bytes、
  hash `80f827fe528f138289288d678b6940df40babee1c9105fa5420ea524537f42c9`。
  packageのBSD本文だけでは、このvendored binaryのprovenance/noticeを確定できません。
- current Qwen manifest scope
  `c98c156466fed7fefa54d90c7036b7d7b21e5fb957e9387607d404d50af82f38`の
  4 rowsは `BLOCKED_UNRESOLVED_REVIEW` のまま。manifest/approvalは変更していません。
  既存Linux 55-row factual auditは取得済みですが、旧Torch/TorchAudio 2.7.1のoperator
  scopeはcurrent 2.13.0/2.11.0を承認しません。旧4/4 CPU結果もcurrentへ流用しません。

## Consequence for step 3

### Source and archive delta clarification

公式PyTorch release source `cf30153c4c131c8164ee7798e5022d810682e2cb` の
gitlinkから、Kineto `094d3c1d072362d0a919a77299459eee94f97931`、Dynolog
`d2ffe0a4e3acace628db49974246b66fc3e85fb1`、cpr
`871ed52d350214a034f6ef8a3b8f51c5ce1bd400` を追跡しました。
そのcpr sourceではruntimeとtestが分離され、test/LICENSEはtest subtreeへの
GPLv3適用を明記します。これは公開source境界の確定であり、4 wheelのbuild identityや
native linkageの証明ではありません。上のsource/build未確定事項をこの範囲だけ補足します。

XCodec2の[Draft #152 dated archive disposition](https://github.com/ayutaz/vokra/blob/fe1f5fe27fe4016d6a6fd76fe131fa3d62e1f307/docs/handoff/xcodec2-dependency-license-disposition-2026-10-04.md)
は同一Linux Torch wheel/native digestについて過去のELF NEEDED結果を記録しています。
今回の新archive report自身はELF/Mach-O解析を実行していません。過去の観測を新しい
installed closure、build provenance、policy approvalへ昇格させません。

#152候補とA7 lockの行別比較では、Linux external archive 62行のうち60行が
version/source/URL/hash/size/upload-time一致、変更はmultidict 6.7.1→6.9.1と
urllib3 2.7.0→2.8.0の2行です。不変archive factsの再利用を検討し、全62行を
盲目的に再取得しません。ただしcurrent clean HEADへのbinding、installed RECORD、
native/build reviewとowner approvalは別の未完gateです。

最新metadata-only inventoryは194 repositories / 193 GGUF-bearing / 198 files、
136 code/artifact-full / 58 unresolved（43 partial / 14 no-runtime-binder /
1 not-artifact）。136はApple hardware-passの件数ではありません。
今回の58行auditではcurrent `VAST_READY` を証明できた行はまだありません。
native seamsが全て存在しない、という意味ではなく、行別gateが揃っていない状態です。

独立に進められるDraft source/依存/reference作業を続け、current exact scopeの
source・license/native・operator gatesが揃った行だけVASTで実weight変換、独立reference、
CPU parityを実行します。小さい証拠回収後にowned worker/storageをdestroyします。
この記録の作業ではVAST workerを作成していません。Scaleway、Apple検証、公開artifact
更新は後段であり、今回のsoftware factsから完了・承認を推定しません。

保護対象CosyVoice2 LLM manifestは閲覧・hash・編集・stage・discardしていません。
凍結済み63-row owner packetと既存測定は変更しません。
