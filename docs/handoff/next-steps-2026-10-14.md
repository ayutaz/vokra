# 残タスクと実行計画 — 2026-10-14 JST

## ゴールと現在地

> **追加readback:** [2026-10-14 execution record](execution-readback-2026-10-14.md)
> は同日のfresh metadata、Qwen修正commit、OpenMP source/native facts、停止した
> VAST workspace attemptとowned resource削除を記録する。remote Cargo、実weight CPU、
> 最終packetの成功とは扱わず、以下の計画準備時点の値・履歴も保持する。

ゴールは公開モデルの各行を、exact source/artifact・依存/license・owner判断・
strict converter/binder/native/CLI・独立reference・実weight CPU parity・
Apple CPU/Metal/no-fallbackの証拠で完了、またはexact owner-approved dispositionで
説明可能にすること。公開artifactの更新と不要なowned cloud資源の破棄も個別に記録する。
静的監査、synthetic test、ソースCIだけでは後段の完了としない。

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
