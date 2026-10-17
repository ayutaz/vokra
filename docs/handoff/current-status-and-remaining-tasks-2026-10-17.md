# 現況と残タスク — 2026-10-17

## 読み方と確認範囲

この記録は、Mac CPU／Apple Metal対応の現況・PRキュー・次の判断をまとめる
management-only snapshotである。過去の実測値、承認、公開条件は変更しない。
以前のdated recordと状態が異なる場合は、この日付の現況と過去の実行証跡を
分けて読む。実行証跡そのものを新しいHEADの結果に読み替えない。

- 確認したGitHub `main`／更新開始時のローカルbaseline:
  `5f1dfc1ec448624ff8745706f9747d67ff3a0bff`。更新開始時のworktreeはclean。
- GitHubの公開PR状態、exact-head check runs、必須context/application、Actions
  workflow結果を2026-10-17 UTCにread-onlyで再確認した。
- モデル一覧は[2026-10-16に回収した194行TSV](public-model-metadata-2026-10-16.tsv)を
  再ハッシュして照合した。SHA-256は
  `9ef3eb07f12a98de7f425dcd75b98f4cd1be45ab1d3907aafeca2626196cdd56`。
  **今回はHF inventoryを再取得していない。**
- ローカルのmodel/config/tokenizer/weight取得・実行、workspace Cargo、
  provider/HF credential参照、有料worker起動、artifact uploadは行っていない。
  providerの全アカウントにinstance/storageが存在しないという監査でもない。

## 完了したことと、完了していないこと

[PR #255受理記録](candle-pr255-acceptance-2026-10-16.md)と
[実CPU/API確認](kyutai-candle-cpu-confirmation-result-2026-10-16.md)により、
元のCandle model-free互換性確認と通常PR受理（六工程の1・2）は完了している。
これは追加observerパッチのコンパイル、独立full-PCM reference、実weight CPU、
Apple実機、owner/legal approvalの完了ではない。

| 元の工程 | 現況 | 次の完了条件 |
| --- | --- | --- |
| 1. 修正版model-free CPU/API確認 | 元のscopeは完了 | #259の追加observerパッチは別scopeとして実コンパイル/API確認が必要 |
| 2. CI・#255通常マージ | 完了 | 新しいPRには、それぞれ新しいexact-head CIとレビューが必要 |
| 3. 全対象の非Apple前提 | 未完了 | source・依存/native/license・converter/binder/forward・独立referenceを対象別に解決し、必要なowner判断を記録 |
| 4. 実weight VAST CPU検証 | 未完了 | 承認済みscopeで変換・独立upstream reference・CPU数値照合、証跡回収、worker/storage削除 |
| 5. 最終packet・Apple実機 | 未完了 | 最終clean HEADとfresh packetを固定後、ScalewayでApple CPU/reference・Metal/reference・Metal/CPUとexplicit no-fallbackを確認 |
| 6. 結果記録・完了監査 | 未完了 | 全対象の結果または正式な保留判断、結果PR/CI、不要データとinstance/storageのcleanupを照合 |

Scalewayは最後のcompute工程であり、source・ライセンス・実装・独立referenceの
不足を代替しない。公開artifactの更新・withdrawalは、この六工程の成功から
権限を推定せず、別のrepository-scoped authorizationと公開gateに従う。

## モデル一覧の残件

10月16日のmetadata snapshotは194 repositories／193 GGUF-bearing／198 files、
136 code/artifact-full、58 non-fullである。136は実機検証完了数ではなく、
各対象のexact source/artifact/lockに対応したCPU／Apple証跡を別途確認する。

| 58行の分類 | 件数 | 次に必要なこと |
| --- | ---: | --- |
| 実行経路はあるがCPU forward／複合処理が不完全 | 22 | component/tensor contract、native composite、独立referenceを完成 |
| artifact差し替え・来歴・companion境界 | 21 | primary provenance/licenseと正しいartifact contract、必要な差し替え許可 |
| runtime binderなし | 14 | 固定source・tensor/component contract、native binder/CLI、依存レビュー |
| 実artifactなし | 1 | 承認済み実artifact、またはexact owner-approved withdrawal判断 |

対象名・revision・checker・最初の不足条件は
[58行の対象別一覧](current-58-prerequisite-inventory-2026-10-14.md)と
[非Apple残件台帳](mac-pre-scaleway-remaining-tasks-2026-09-05.md)を参照する。
`NOT_PROVEN`は未証明であり、全行が実行不可と判定済みという意味ではない。
保護対象CosyVoice2 LLMのexecution readinessは`UNKNOWN_EXCLUDED`のまま。
凍結された[63行owner packet](mac-cpu-metal-owner-disposition-packet-2026-09-07.md)は
別の分母であり、58行へ書き換えず、署名・scope hash・判断も変更しない。

## 未マージPR：7件

以下はこのsnapshot時点の状態である。GitHubの`DIRTY`はmainとのmerge conflictであり、
ローカルworktreeの未コミット差分を意味しない。CIがgreenでも保留条件は消えない。

| PR | head | 状態 | 残っている作業 |
| --- | --- | --- | --- |
| [#258 urllib3](https://github.com/ayutaz/vokra/pull/258) | `b2846466` | 非Draft・BEHIND | Canaryのlock変更と承認hash/reference consumerへの影響レビュー、main追従、新CI、通常マージ判断 |
| [#259 Candle observer](https://github.com/ayutaz/vokra/pull/259) | `481ab7e2` | Draft・CLEAN | 新パッチを適用したupstream Rust library/APIの実互換性検証と結果レビュー |
| [#198 法務資料](https://github.com/ayutaz/vokra/pull/198) | `5614a92c` | Draft・BEHIND | 一次資料アクセス、版・適用条件、owner/legal判断。無変更rerunやリンク削除で通過させない |
| [#191 Kyutai](https://github.com/ayutaz/vokra/pull/191) | `63cadf3c` | Draft・DIRTY | 競合解消に加え、supported dependency/native/license、独立PCM/KV/reset/eviction、実weight CPU |
| [#182 VibeVoice](https://github.com/ayutaz/vokra/pull/182) | `a3afc1ec` | Draft・DIRTY | 競合解消、41-row installed/native/license closure、owner・voice-rights、独立実weight検証 |
| [#169 Torch監査](https://github.com/ayutaz/vokra/pull/169) | `b3e9d5c1` | Draft・DIRTY | #182が実際にmainへ入り、強いsecurity/collector責務の統合を確認するまで保持 |
| [#152 XCodec2](https://github.com/ayutaz/vokra/pull/152) | `a9f54e1a` | Draft・DIRTY | 競合解消、依存/native/license/API/owner、実weight CPU、Apple検証 |

### #259の通常CIは完了、追加パッチの実コンパイルは未実施

exact head `481ab7e2b4fc471d9b6edc037f0778501b6bc5ba`、base `5f1dfc1e`で、
72 SUCCESS／1 intentional SKIPPED／0 FAILURE／0 live checks。
skipは`gitleaks full history`。strict branch protectionの16 context/application
pairを照合し、すべてsuccess。外部`CodeQL` application `57789`も含む。
開始した10 workflowsはすべてattempt 1でsuccess。

- [source tests run 38999482568](https://github.com/ayutaz/vokra/actions/runs/38999482568):
  Ubuntuは15 observer＋17 ASR＋27 mocked checker＋8 workflow contract＝67 tests。
  macOS／Windowsはそれぞれ15＋17＝32 tests。各OSのexplicit self-testも成功。
  これはsource/fixture処理であり、新パッチを適用したupstream Rust実行ではない。
- [通常CI run 38999482767](https://github.com/ayutaz/vokra/actions/runs/38999482767):
  最後のUnity package jobは2026-10-17 03:58:51 UTCにsuccessで完了。
- 新しいmodel-free compiler workflowは準備済みだが未起動。
  activation labelはなく、将来の8-file input deliveryも未配置。
  過去の#255 one-shot実行許可は再利用しない。

追加パッチのcompile/API/borrow/type確認、observer runtime、raw K/V収集、
正しくCandleに結び付いたproducer/consumer、full PCM reference、real-weight CPUは
それぞれ別の未完了条件。通常CIだけを理由にDraft解除・マージ・Apple PASSへ進めない。

## 直近の個別確認待ち

以下は前回提示したscopeの未回答事項であり、この文書更新は許可を代替しない。

1. #258のため、Canary関連4ディレクトリのlock／approval manifest／hash-reference
   codeをread-onlyで静的レビューする範囲。対象は`canary_1b`、
   `canary_1b_reference`、`canary_1b_v2_reference`、`canary_qwen_reference`。
   モデル実行や承認manifest変更は含めない。
2. #259のため、回収済み原本証跡8ファイル（合計4,333,048 bytes）を公開PRの検証入力として
   配置し、GitHub-hosted Linuxでmodel-free互換性検証を1回、最大20分行う範囲。
   upstream build scripts／proc macrosの実行を含む。モデル取得・構築・推論、
   VAST／Scaleway起動、HF公開は含めない。

元のGitHub input artifactはexpired／404であり、同じdownloadの無変更再試行では
解消しない。[原本8ファイルの実適用記録](kyutai-candle-paste-application-result-2026-10-16.md)と
回収済みprivate evidenceの一致は確認済みだが、公開deliveryの許可とは別である。

## 以降の実行順序と終了条件

1. 既存PRの実際の不足条件を先に解消する。#258はscope review→main追従→新CI、
   #259は個別scope確定→実compile/API→証跡レビュー→通常マージ判断。
   source-only CIやmergeabilityだけで他のDraftを一括マージしない。
2. 全対象のsource／依存／license／implementation／独立referenceを対象別に整理し、
   agentが解消できる事実とowner/legal/upstream待ちを分離する。
   外部待ちの行を黙って完了扱い・分母から除外しない。
3. 準備・承認が揃ったexact scopeからVASTでno-upload変換、独立reference、実CPU parity。
   失敗は原因を特定して修正し、同じ失敗の無変更再試行や許容誤差の緩和で隠さない。
   小さい証跡を回収後、所有workerと保存データをdestroyする。
4. 必要な非Apple前提とCPU legを完了、または正式なowner-approved dispositionで
   accountedにしてから最終clean HEAD／fresh transfer packetを生成する。
5. Scalewayを最後に起動し、exact model／HEAD／packetに対するApple CPU/reference、
   Metal/reference、Metal/CPUをexplicit no-fallbackで確認する。
6. 実結果と正式な保留判断をfollow-up PRへ記録し、fresh CI・対象別証跡・cleanupを監査する。
   ローカル不要モデルの削除は必要物を同定した別の安全確認を経て行い、
   保護対象や唯一の検証証跡をcleanupに混ぜない。

この文書は残タスクの整理であり、工程3〜6の完了宣言、モデル／有料実行許可、
owner署名、公開artifact更新許可ではない。

## このdocumentation refreshの検証範囲

更新はREADME・API索引・backend guideの日英版、docs index、platform support、
owner checklist、ABI/license索引、governance、この現況記録と関連する
dated plan/ledgerへのsupersession noteの計22ファイル。実装・workflow・
承認manifest・既存TSVは変更していない。ABI/license/governanceでは現況への入口だけを
更新し、既存判断や測定、署名・checkboxは変更しない。

- 文書コード例：144 blocks／26 docsを検査し成功。30のtier-C deferred項目は
  未検証のままであり、成功数に読み替えない。checker self-testも成功。
- 文書参照：110 requirement IDs、33 architecture anchors、日英6組、入口リンクを
  検査し成功。17-case self-testも成功。
- 変更22文書内の相対リンク563件とfenceを明示的に検査し、成功。
  新規untracked文書も含めた。この結果を全体gate成功とは読み替えない。
- community docs、platform support（53 anchors）、ABI changelog、workflow hygiene
  （56 workflows／36 crons）は成功。API/header生成やRust workspace実行ではない。
- `git diff --check`は成功。historical実行証跡・署名行を上書きしていない。
- 全体runbook-path／owner-checklist gateは未実施。既存checkerは保護対象を含む
  parity README／tracked Markdown全体を読むため、この更新では呼び出さない。
  変更文書の相対リンク検査を、この二つの全体gate成功とは主張しない。
- Python checkerはすべてoffline UV／Python 3.12経由。ローカルモデル実行、
  有料worker、HF公開、コミット／push／PR作成・マージはこの更新に含めない。
