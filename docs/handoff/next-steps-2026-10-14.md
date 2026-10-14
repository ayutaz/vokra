# 残タスクと実行計画 — 2026-10-14 JST

## ゴールと現在地

> **17:46:55 UTCの統合受理readback（2026-10-15 JST）:** #247 exact
> `50ea0dec`／base `0f6750fa`は全73 checks（71 success / 2 intentional skips）、
> 必須16 context/appと開始した全10 workflowsのterminal successを確認して通常受理した。
> 遅いUnity packagingも完了後に受理し、main／ローカルmainは`d1cbed72`でclean。
> #243／#245は統合によるsupersededとしてcloseし、個別mergeとは記録しない。
> Diaの無条件実行拒否、source/staticと実weight結果の境界、元の安全ルールは保持した。
> #246の[API-smoke lock修正](moss-api-smoke-urllib3-reconciliation-2026-10-15.md)と
> [bridgeの独立lock照合](moss-api-smoke-bridge-binding-2026-10-15.md)を同じ候補へまとめ、
> 最新mainを通常mergeした`d039e4dd`でfocused metadata checksは成功した。
> 45 full records／44 whole-lock registry rowsはinstalled active dependenciesではなく、
> main MOSSの48 pending rowsとは別契約。新しいpublished HEADのfresh CI／通常受理は別に必要。
> 次は#246受理と独立したsource前提の解決であり、旧承認の再刻印や未承認の有料監査はしない。
> 58未解決行、凍結63-row owner scope、実weight CPU、最終clean HEAD／Apple packetは未完了。
> 新規cloud／ローカルモデル実行／公開は開始していない。以下の受理snapshotは履歴。

> **16:55:26 UTCのMOSS受理と検証集約（2026-10-15 JST）:** PR #242はexact
> `934e61ac`／base `627a641f`の全70 checks（69 success / 1 skipped）、必須16
> context/app成功、開始した全9 workflowsのterminal successを確認して通常受理した。
> 最新main／ローカルmainは`0f6750fa`でclean。48 dependency rows、source/model grant、
> native/API/owner review、実weight CPU／Appleの承認や完了を意味しない。
> 後続#245 Dia `cd90b94a`、#243 docsのprivate追記`f9feaabb`、stdlib pre-commit
> 修正`e22a81eb`を通常mergeした候補`012ccaf4`に集約し、順次main更新によるCIの再無効化を
> 減らす。元のPRは統合内容のfresh CI／受理が済むまで残し、古いCIを流用して受理しない。
> [軽量検査の修正](pre-commit-stdlib-invocation-2026-10-15.md)は同じ5検査を保持し、
> stdlib scannerの不要な146-package／約1.3G environment再生成を避ける。parser／memory
> guards／owner gatesは不変で、通常commitと隔離negative-controlも成功した。
> [OWSM follow-up](owsm-audit-admission-readback-2026-10-14.md#fixed-frontend-has-no-native-complex-selection-flag--2026-10-15-jst)
> は固定公式frontendに依存回避flagがないことを確認し、[Qwen2候補調査](qwen2-audio-fixed-source-license-readback-2026-10-15.md)
> は別providerのsource factsと既知advisoryの範囲を分離した。実行や署名は解除しない。
> 58未解決行、凍結63-row owner scope、実weight CPU、最終clean HEAD／Apple packetは
> 未完了。以下の受理snapshotは履歴として保持する。

> **16:13:50 UTCのsource受理readback（2026-10-15 JST）:** PR #241はexact
> `6bf092c4`／base `f0130b5d`の全70 checks（69 success / 1 skipped）、必須16
> context/app成功、開始した全9 workflowsのterminal successを確認して通常受理した。
> 最後のUnity Android/WebGL/UPMも成功し、最新main／ローカルmainは`627a641f`でclean。
> source/staging修正だけの受理であり、依存/native/API/license/owner承認ではない。
> MOSS #242とDia #245はこのmainの通常mergeとfresh CIが必要。Diaのdumper/validator
> 現行contract修正`0c6bfd62`はpending #245 `cd90b94a`に含め、31 full records／27 active Linux
> packagesを分離、未review／未検証seamの表現と無条件実行拒否を保持した。これはまだ
> 公開headのCIで検証済みとは扱わない。[MMS current evidence readback](mms-current-evidence-admission-readback-2026-10-15.md)
> は既存wheel-size receiptによる正当な省略を確認し、不必要なlock変更／有料再監査をしない。
> [Qwen2 Audio fixed-tree readback](qwen2-audio-fixed-source-license-readback-2026-10-15.md)
> もprimary metadataの限定的確認であり、complete history receiptやsource grantではない。
> 58未解決行、凍結63-row owner scope、実weight CPU、最終clean HEAD／Apple packetは
> 未完了。新規VAST／Scaleway／公開／ローカルモデル実行は開始していない。

> **2026-10-15 JSTの行別静的再確認:** [58-row readback](current-58-static-prerequisite-readback-2026-10-15.md)
> で先行10行と残り48 report IDsを整理した。57行のsource/static observationsと保護対象1行の
> `UNKNOWN_EXCLUDED`であり、HFのfresh全行取得・production READY判定・実weight結果ではない。
> public/upstreamの別repository revisionや未受理候補との差だけを不整合と扱わず、
> 現行MOSS lock48件／review49件と一致するlock hashを独立確認した。
> source/consumerの独立作業とowner/legal/upstreamの外部前提を分離し、全行non-Apple完了・
> 最終clean HEAD／Apple packetというゴールを変えない。

> **14:53:31 UTCのsecurity受理readback（2026-10-15 JST）:** PR #244はexact
> `71cc8017`の全70 checks（69 success / 1 skipped）、必須16 context/app全成功、
> 開始した全9 workflowsの完了を確認して通常受理した。main／ローカルmainは
> `f0130b5d`でclean。変更はisolated server lockのrustls 0.23.45→0.23.46のみで、
> Linux muslのadvisory/build/tests、macOS/Windows serverとdeploymentも成功した。
> #241旧headと#243旧headの同じ新規RustSec失敗をrerun／ignoreせず原因修正した。
> WeSpeaker `6bf092c4`、MOSS `33cb9526`、Dia `d33d7843`はこのmainを通常mergeした
> 未受理候補で、focused model-free checksは成功、fresh exact-head CIは別に必要。
> MOSSのcurrent-lock bindingとDiaの27-package／147件 publisher記録の契約修正は、
> pending／unsigned／NO_UPLOADを維持し、実依存・API・owner approvalを解除しない。
> 58未解決行・凍結63-row owner scope・実weight CPU・最終clean HEAD／Apple packetは
> 未完了のまま。以下の13:43等のsnapshotは履歴として保持する。

> **13:43:42 UTCの受理・監査readback:** PR #240はexact `e3fadc73`の
> 全73 checks（72 success / 1 skipped）と必須16 context/app全成功を確認して
> 通常受理した。main／ローカルmainは`fb1a692a`でclean。PR #237/#238/#239も
> 受理済みだが、実weightやApple完了の判定には使わない。PR #241は上流fixtureの
> 元の綴りを保持する単一ファイルのtypos除外と、最新mainの通常mergeを含む
> `5199643d`をpushし、fresh CI待ち。旧headの失敗／Linux成功は履歴である。
> [Candleの実source-only capture](kyutai-candle-reference-readback-2026-10-14.md#actual-source-only-capture-and-retained-lock-rejection)
> は射影lockの`backtrace`完全record不一致で停止し、owned `55932858`と保存データを
> 13:28:48 UTCにdestroy、個別null readbackまで確認した。違うfieldは未確定で、
> 許容条件を緩めず診断だけを準備する。[RMVPE一次source調査](rmvpe-primary-license-readback-2026-10-14.md)
> も固定source／weightの明示的grantを発見できず、同じlicense holdを維持する。
> 58行・実weight CPU・最終clean HEAD／Apple packetは未完了のまま。

> **11:01:31 UTCの受理readback:** PR #236はexact `afbaf0f9`の全70 checks
> （69 success / 1 skipped、pending・失敗なし）と必須16 context/appの全成功を
> 確認して通常受理した。Unityの全24 stepも成功を個別確認し、古いpending情報による
> 自動審査の拒否は新しい証拠で解消した。mainは`99bfa77a`、ローカルmainもFF更新して
> clean。これでsource-only observer準備は受理済みだが、Rust/API・独立PCM oracle・
> 実weight CPUは未実施。続く[FireRedの設定readback](firered-checkpoint-config-source-readback-2026-10-14.md)
> はcheckpoint内`args`の契約と、tensor値を読まない小さなmetadata readerを準備した。
> 27件の合成安全性テストと940 descriptorの容量確認は、実モデルの設定・940 tensorの
> 実inventory・依存承認・parityの証拠ではない。58行と最終Apple packetの完了は未判定のまま。

> **10:17:38 UTCの受理readback:** PR #234はexact `54340ba5`の69 success /
> 1 skipped、pending・失敗なし、必須16 context/appの全成功を確認して通常受理した。
> 最新mainは`ab653152`、ローカルmainもFF更新してclean。observer準備は`c2b85c24`
> として保存し、このmainを通常mergeした候補は`c90eeac5`。以下のCI待ち記録は履歴。
> VAST検索の単位ずれを確認し、採用条件のraw `cpu_ram>=65536`を変えずCLI検索を
> `cpu_ram>=65.536`（1000倍）へ揃えた。10:17:16 UTCのread-only検索は0 quotesで、
> inventory照会・作成・SSH・モデル実行なし。これは検索条件内の結果であり、VAST全体の
> 在庫切れという判定ではない。費用上限・期限・削除条件は変更せず、58行の実weight
> CPU・最終Apple packetは未完了。詳細は[observer準備記録](kyutai-candle-observer-source-preparation-2026-10-14.md)。

> **09:39:44 UTCの受理readback:** PR #235はexact `64422404`の69 success /
> 1 skippedと必須16 context/appの全成功を確認して通常受理した。mainは`d4a54d62`、
> ローカルmainもFF更新済み。PR #234はその後behindと判定され、最新mainを通常mergeした
> `54340ba5`でfresh CIを要求する。旧headの成功は新headのCIと扱わない。
> [Kyutai observer source準備](kyutai-candle-observer-source-preparation-2026-10-14.md)
> は17件のstdlibテストと原本復元を確認した別候補で、Rust/API・full PCM reference・
> 実weight CPU・最終packetの成功ではない。新規VAST/Scalewayは開始していない。

> **09:03:54 UTCの受理readback:** PR #232はexact `89f0e2e5`の70 checks
> （69 success / 1 skipped、失敗・pendingなし）と必須16 context/appの全成功を
> 確認して通常squash mergeした。受理mainは`fa42f112`、ローカルmainもFF更新済み。
> OWSM修正`c86f72ab`は保存し、最新mainを通常mergeした候補は`b92665f4`。
> 以下のPR #232未受理記録は、その時点の履歴として保持する。
> 新しいVAST source-metadata-only候補検索は64 quotesを返し、上限内3件は全て
> RAM 65,536 MiB未満のため作成前停止。料金・RAM・削除条件は緩めず、新規leaseなし。
> [Candle source-only admission記録](kyutai-candle-reference-readback-2026-10-14.md#source-metadata-only-offer-admission-readback)
> を参照。実weight CPU・最終packet・58行の完了判定は未完了のまま。

> **後続の受理と独立作業:** PR #233 は56 success / 1 skippedと必須16 context/app
> の全成功を確認して08:24:33 UTCに通常受理され、mainは`46d2c9dd`。
> PR #232の新候補`89f0e2e5`はこのmainを通常mergeしてfresh CI待ちで、未受理。
> [OWSM admission修正](owsm-audit-admission-readback-2026-10-14.md)は、証拠に未束縛の
> 監査フラグを拒否する。実際に使われるComplexTensorのlicense holdは解除しない。
> この追加の静的作業も、58行の実weight CPU完了や最終packet生成とは区別する。

> **PR受理後の最新readback:** PR #230 は 06:19:07 UTC に通常受理され、main は
> `f8804b7429691bd90520aec801869262ba53573b`。以下のPR準備時点の記録は履歴として
> 保持する。[Kyutai reference調査](kyutai-candle-reference-readback-2026-10-14.md)では
> 公式PyTorch/Candleのcheckpoint識別が一致したが、固定Rust lockのmemmap2 0.9.5に
> RustSec問題を確認。未修正lockの有料再検証はせず、依存scopeと実行契約を先に解決する。
> 58行の実weight／owner前提・最終packetは未完了で、新規leaseは開始していない。

> **追加readback:** [2026-10-14 execution record](execution-readback-2026-10-14.md)
> は同日のfresh metadata、Qwen修正commit、OpenMP source/native facts、停止した
> VAST workspace attemptとowned resource削除を記録する。remote Cargo、実weight CPU、
> 最終packetの成功とは扱わず、以下の計画準備時点の値・履歴も保持する。

> **後続の実測:** 同じexecution recordに、exact `005ac126`のRust fmt/Clippy/
> workspace tests（8,218 passed / 0 failed / 105 ignored）と後続shell gate失敗を
> 追記した。Linux tempdirの1行修正は`9e528d27`、失敗・PR readbackは`5d954727`。
> SSH待機controllerの固定hash回帰は合格したが、その後のVAST admissionは料金上限
> 内の候補がなく作成前停止。全体remote green、push/PR受理、実weight、最終packetは
> 依然未完了で、古いHEADの部分成功を新HEADの全体成功に流用しない。

ゴールは公開モデルの各行を、exact source/artifact・依存/license・owner判断・
strict converter/binder/native/CLI・独立reference・実weight CPU parity・
Apple CPU/Metal/no-fallbackの証拠で完了、またはexact owner-approved dispositionで
説明可能にすること。公開artifactの更新と不要なowned cloud資源の破棄も個別に記録する。
静的監査、synthetic test、ソースCIだけでは後段の完了としない。

> **最新の継続結果:** FireRedの固定幅beam/cache/PAD-key修正は`3a7cc5c7`、
> private helperのtest-only化は`9e5dc575`。公式proxy経由で実ソース受理まで進んだが、
> 前者のVASTはClippyで停止し、テストは未開始。原因を修正した後者は料金上限内の
> 候補がなく作成前停止。04:38 UTCのCPU floor 4を含むread-only調査も上限内0件。
> owned worker/storageは削除確認済み。次は修正後clean HEADのremote greenを取得し、
> 通常PR/CIへ進む。58行の実weight/owner前提・最終packet・Scalewayは未完了のまま。
> 詳細は[継続readback](execution-readback-2026-10-14.md#corrected-helper-head-and-current-no-lease-boundary)。

現在の「計画の4まで」という実行ゴールは、元の順序の4（最終clean HEADを固定して
Apple packetを再生成）までを指す。以下の表は安全整理・PR準備を展開しているため、
この到達点は表の工程5に相当する。工程4のCPU作業だけでゴール達成とせず、Scaleway
実行と公開はこの到達点の後に置く。

> **Kyutaiの先行作業:** [固定sourceの照合](kyutai-stt-pcm-contract-reconciliation-2026-10-14.md)
> により、非公開PCM接続は既に存在し、追加接続よりも契約の混在修正が必要と判明した。
> MLXのpaddingとPyTorchのLM方式を分離し、非公開PCM経路を固定DSM/PyTorchへ整合する。
> 独立PCM oracle・実weight CPU parityは未取得で、完了件数や最終packetは進めない。

> **最新source検証:** clean `4263f416` のVAST replayはClippy・workspace tests・
> 最終shell gateまで合格（8,223 passed / 0 failed / 105 ignored）。private Kyutai
> PCM 20/20、FireRed native 29/29を確認し、`55867942`と保存データはdestroy後の
> 個別null readbackまで完了した。次は通常PRのCI／受理。独立PCM reference、
> 58行の実weight／owner前提、最終Apple packetは未完了。
> [exact実行と回収hash](execution-readback-2026-10-14.md#exact-source-replay-passed-and-owned-worker-destroyed)。

- 計画準備時のローカル `main` はドキュメントcommit
  `1a63a8d162b2285650d858ea5d174573588b3f13`。2026-10-14のGitHub branch APIで
  remote `main` は当初 `7d38061c3dcb56f6a16bcf4b8085d966b3cdab09` と確認し、
  PR #229受理後は `134eb9c7dbbfdafd2e4c1bdddac2e97253f9eea4` をreadbackした。
  文書commitはローカルのみで、push/PR/merge済みとは扱わない。
- 未コミットの実装差分は `.codex/hooks/guard-protected-path.sh`、
  `scripts/check-codex-hooks.sh`、新規 `scripts/check-protected-path-metadata.sh`
  の3件。文書commitには含めていない。
- [10月13日のmetadata-only監査](documentation-refresh-2026-10-13.md)は
  194 repositories / 193 GGUF-bearing / 198 files、136 code/artifact-full /
  **58 unresolved**。これはApple hardware-pass件数ではなく、10月14日にHF全行を
  再監査したという主張でもない。凍結63-row owner scope、M5の53 checked /
  29 uncheckedは別の母数として維持する。
- [VibeVoiceの41-wheel静的監査](vibevoice-realtime-static-closure-audit-2026-10-13.md)
  は完了。controller PASSに対してcollectorは
  `OWNER_REVIEW_REQUIRED_NO_UPLOAD`。private candidateの監査であり、mainの
  project/lock/runtimeを検証したものではない。同じ入力の有料再監査は不要。
  owned worker `55648877` と保存データは破棄確認済み。これはVASTアカウント全体の
  空き状態の保証ではない。

## 優先順位・成果物・完了条件

### 実行readback — 2026-10-14 00:16 UTC以降

文書は`0379607c`、安全フック3件は`9de9409c`として別の論理commitにした。
受理済みmain `134eb9c7`を通常mergeした検証候補は
`6e1a4729f675d0094cbbc1deeedd85c1d3f2e661`で、この照会時点のworktreeはclean。
これは検証準備のHEADであり、全行non-Apple完了後の最終Apple packet用HEADではない。
文書／安全フックのpush、PR、remote workspace greenはまだ確認していない。
GitHub APIの再照会でもmainは`134eb9c7`、openはDraft #198/#191/#182/#169/#152の5件。
先行する「未コミット3件」は計画準備時点の状態として読み分ける。

[source-license follow-up](vibevoice-source-license-reconciliation-2026-10-14.md)で
safetensors 0.5.3とtokenizers 0.22.2の固定tag／commit／LICENSE blobを確認した。
source-license factsの補完であり、wheel/native/owner/API gateの解除ではない。
古いprivate static auditのraw receipt directoryは新しいfilesystem照会で不在だった。
tracked summaryと記録hashは保持し、回収済みの原本と主張しない。source LICENSEを
見つけたことを理由に、同じ入力の有料再監査や実モデル実行を開始しない。

[Qwen3-TTS current-lock reconciliation](qwen3-tts-current-lock-reconciliation-2026-10-14.md)
では、PR #223後のlockと旧gate digestの不一致を修正した。旧audit/owner証拠は
書き換えず、新scopeのapproval fieldsを未承認へ戻す。実際のgateを呼ぶ静的fixtureで、
現行lockの入力確認後に旧dependency auditを拒否することを確認した。
これは実weight実行やAPI importの許可ではない。source/LICENSE照合とこの修正は
静的な前提整備であり、全58行のnon-Apple完了や最終packet固定とは区別する。

| 順序 | 作業 | 完了条件／次段への条件 |
|---|---|---|
| 0 | 不要ローカルモデルの安全整理 | exact path・帰属・再取得／別コピー・未使用を確認したものだけ削除。fixture、未公開唯一成果物、receipt/owner証拠は保持。削除結果と空き容量を記録 |
| 1 | 安全フック3差分の恒久修正を独立レビュー | 元のモデル安全ルールを保持し、metadata 35ケースとfull hook gateをexact diffで確認。実装担当Luna→manager review→独立した論理commit/PR/CI。133行の期限付き例外は復活させない |
| 2 | 文書commitと新着PRの受理準備 | ローカル文書を通常のレビュー経路へ載せる。#229の固定action差分・必須CI・保護条件を確認して受理する。変更後は新HEADのCIを取り直し、mainの受理結果をreadback |
| 3 | Draftと非Apple前提を行別に整理・解決 | source、exact lock/archive/native、license/provenance、owner scope、upstream/API、converter/binder/native/CLIの不足を分類。外部待ちは保持して独立作業へ移り、実際に揃った行だけ `VAST_READY` |
| 4 | VASTで実weight変換・独立reference・CPU parity | execution-approved exact scope・clean source HEAD・モデルフリー前提がgreen。固定upstreamをoracleとして実測、ログ/hashを回収。失敗原因を解決してから再実行し、owned worker/dataをdestroy/readback |
| 5 | 最終clean HEADとApple packet固定 | non-Apple前提を全行で完了、またはexact owner-approved withholding/withdrawal等でaccount。必要なremote workspace gatesもgreen。旧packetを再利用せず、最終HEAD・weight/reference/hash-bound manifestを再生成 |
| 6 | 最後にScalewayでApple実機検証 | Apple CPU/reference・Metal/reference・Metal/CPUを個別に測定、明示的no-fallback。失敗時は実装/VAST工程へ戻し、Scalewayだけで不足したsource/licenseを埋めない |
| 7 | 結果記録・公開・後片付け | exact結果をfollow-up PRへ記録しCI／レビュー受理。公開はrepository/artifact-specific承認と `publish-one.sh` のgate経由。live inventoryを再照合、不要なowned instance/storageを破棄 |

この計画更新自体はcommit/push/PR mutation/mergeや実装、モデル実行、新規cloud leaseを
開始するものではない。実行時は既承認のexact scopeを確認し、静的監査限定の承認・
費用上限・30分枠を実weight実行へ流用しない。予算・時間・cleanup条件をrunごとに固定する。

## PRキュー：2026-10-14の確認と受理

GitHub連携のopen searchは次の6件を返した。#229のexact HEAD check-runs APIは
total 77、74 success / 3 skipped、pending/failureなし。legacy commit-status APIの
空配列を「CIなし」や「PASS」と解釈していない。受理前の#229は
open/non-draft/mergeable、head `9db7d7b512888d3ea9834b10aaca767881309ae1`。
通常マージ後のPR／branch readbackはmerged=true、accepted main
`134eb9c7dbbfdafd2e4c1bdddac2e97253f9eea4`。受理日時は2026-10-13
15:30:40 UTC（2026-10-14 00:30:40 JST）。

| PR | 現行の計画上の扱い |
|---|---|
| [#229](https://github.com/ayutaz/vokra/pull/229) | レビュー・通常受理済み。install-actionの固定SHA更新、4 files / 5 additions / 5 deletions。exact headの必須16 context/application pairsは全成功。accepted main `134eb9c7` |
| [#182](https://github.com/ayutaz/vokra/pull/182) | Realtime runtime/referenceの統合候補。10/13 static receiptをexact private候補／lockにbindし、license/native/owner/compatibilityとfull E2E／streaming/KV責務を残す |
| [#169](https://github.com/ayutaz/vokra/pull/169) | Torch CPU reference closure。#182への統合／supersessionは実際のaccepted-mainと責務coverageを確認してから判断。先に閉じない |
| [#152](https://github.com/ayutaz/vokra/pull/152) | XCodec2 pre-import gateとdecoder tooling。current lockとの差分、installed/native証拠、owner approvalをそろえる。旧lockの成功を新lockにrestampしない |
| [#191](https://github.com/ayutaz/vokra/pull/191) | Kyutai DSM PCM schedule/reference metadata。source-only認証を再取得せず、残る依存・API・実weight独立composite reference・CPU parityへ進める |
| [#198](https://github.com/ayutaz/vokra/pull/198) | dated法務source reconciliation。日付・一次source・owner判断と範囲をレビュー。PR受理をモデル／依存の実行・公開承認に読み替えない |

残る5件は10/13時点でDraft。今回はopenを再確認したが、各Draftのfresh全CI結果や
全diffレビューを完了したとは主張しない。受理準備時にhead・base・全必須CIを再確認する。
GitHub CLIはTLS certificate verificationで失敗したため、検証を無効化せず
GitHub連携GETでbranch、open PRs、#229 metadata/check-runsを読んだ。

### この実行で完了したPRレビュー

#229は5箇所の`taiki-e/install-action`を
`4cef1412cce204788f482e778a0b9187f9626a29`から公式v2.87.25の
`183e4297cca2404691e9380e1307288dced5c82a`へ更新する限定差分。
workflow permission／引数／runtime依存の変更はない。公式upstream compareも
照合し、action entrypoint／実行scriptと使用4種のcargo tool manifestの変更が
ないことを確認した。upstream codegenの変更はsyft signature bundle対応であり、
Vokra jobのruntime変更とは扱わない。その他のmanifest更新があるため、単に
同一actionと主張せず、exact head CIを併用した。

merge直前にprotected branchから16 required context/application pairsを再取得し、
全件completed/successを確認。expected-head付き通常squash merge後、PRの
merged状態とmain SHAをreadbackした。管理者overrideや保護ルールの変更は行って
いない。管理権限を要する全protection endpointは取得できないため、新しくstrict／
admin／linear／conversation全設定を監査したとは主張しない。

安全フックはLunaとmanagerが35永続ケース（21既存＋14生成）とfull hook gateを
それぞれ再実行してPASSを確認。論理commit／remote verification／PR受理は別の残件であり、
この文書変更に未検証の実装3ファイルを混ぜない。

## 先に解決する具体的な前提

1. **VibeVoice Realtime**：safetensors 0.5.3／tokenizers 0.22.2のbundled license
   不足を、exact-version primary license・build/native provenanceと分離して解決する。
   10 native distributionsのrelease可否とexact owner scopeを確定する。
   static installation成功はimport/API/full-generation互換性ではないため、必要な
   source/API検証とruntime parityを別の限定scopeで準備する。
2. **Qwen3-TTS／共通Torch**：[10/09 prerequisite record](step3-prerequisite-evidence-2026-10-09.md)
   のTorch 2.13.0／TorchAudio 2.11.0 archive事実を再利用できる範囲だけ使う。
   cpr test/runtime境界、libomp/libgomp、vendored libc++のexact binary/build/notice、
   current operator scopeを解決する。旧2.7.1承認・旧4/4 CPUをcurrent結果としない。
3. **残る全行**：[行別ledger](mac-pre-scaleway-remaining-tasks-2026-09-05.md)と
   [owner packet](mac-cpu-metal-owner-disposition-packet-2026-09-07.md)を現行sourceと照合。
   topology/tokenizer/codec・独立referenceが欠ける行はsource workを先に行う。
   source CIがgreenでも `VAST_READY` を推定せず、signableでないscopeをownerへ
   形式的に承認依頼しない。保護対象CosyVoice2 manifestは閲覧・hash・編集等の対象外。
4. **別のsecurity queue**：10/13の199 Dependabot／8 Code Scanningはdated snapshot。
   CodeQL #68–#71の実データフローと4 Scorecard項目をtriageする。これらはモデル
   数値検証とは別であり、テストファイルという理由だけでfalse positiveとして閉じない。

GPUが実測で速くなる工程は、承認済みremote scopeで利用を検討する。ただしGPU reference
の成功でCPU parityを省略せず、CPU／Metalの対象backendの結果を個別に残す。

## リソースと安全ルール

### ローカルモデル整理の実測

managerと2名の独立したread-only調査で、repo内と `/private/tmp/vokra*` に
削除可能と確定できる不要モデルは見つからなかった。**今回のモデル削除は0件**。
repoは約3.7GB、再確認した対象filesystemの空きは約168GiBという照会時点の値であり、
今回削除によって空いた容量ではない。

- `tests/parity/silero_vad/silero-vad-v5.gguf` は2,183,424 bytesのGit-tracked
  fixture。CI/CLI/C API/server/parityが参照するため保持。
- Vokra名の一時領域にも実モデル候補は確認できず、private candidate・venv・
  Git pack・静的監査receipt等は別の用途があるため保持。
- repo内parity環境約2.7GB、LibriSpeech archive約322MB、bytecode cache、
  fuzz buildはモデルではない。今回のモデル削除と混ぜて削除しない。
- 共用HF cache約2.5GBのうちruri model約1.2GB、Torch cache約504MBは、
  他projectの利用／不要性を確定できていないため保持。
- `/Users/inamotoyuuta/.local/share/ralomi/private-backups` は約118GB。
  別projectのdataset・model backup等であり、保存先／復元可能性を未検証。
  これはVokraの不要モデルや再生成可能cacheとして扱わない。

上記はmodel download/load/forwardをせずpath・サイズ・Git追跡・参照を調べた結果。
別projectのbackupや共有cacheを対象に広げる場合は、帰属と唯一コピーでないことを
別途検証してから削除可否を判断する。

### 継続時の制約

- ローカルではモデルdownload/load/forward/変換をしない。広域／`vokra-models` Cargoは
  VASTへ。文書・静的検査・既知の軽量検証だけを行う。
- 今回のローカル整理ではrepoの容量と既知cacheの帰属を確認。共用HF/Torch cache、
  特に別projectのRalomi private-backupsは、不要／代替コピーが証明されるまで残す。
  「この作業で使わない」だけを削除理由にしない。
- 同じ失敗の有料再試行はしない。小さいログ／receiptから原因、修正diff、focused
  regression、actual leafの検証をそろえてから次のbounded runを計画する。
- 停止だけでstorageを残さず、owned resourceをexact-IDでdestroy/readback。
  他projectの資源は操作しない。今日の全cloud inventoryは照会していないため、
  アカウント全体の「instanceゼロ」を主張しない。

完了予測は未解決scope・実weight・Apple実測が揃うまで固定しない。
まず工程0–3の、料金を発生させず進められる作業を優先する。

## この計画更新の検証

文書reference・community（日英整合とリンク）・runbook path citations・
文書example checker・`git diff --check`はPASS。example checkerの30 Tier C
ブロックは引き続き未検証。新規計画を含む5既存＋1新規文書の追加相対リンク11件は
path存在チェックPASS。既存の安全フック3差分のSHA256は変更前後で同一。
当初の計画更新ではモデル実行・削除・新cloud・PR mutation・commit/pushは行って
いない。その後の実行で#229の通常マージと受理readbackを完了した。実weight CPU、
最終clean HEAD／Apple packet、Scaleway検証は未完了のままである。
