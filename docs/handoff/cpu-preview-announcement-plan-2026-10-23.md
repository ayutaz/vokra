# CPUプレビュー告知までの実行計画 — 2026-10-23

## 最小checkout回帰とPR gate — 2026-10-23 10:57:05 UTC

統合先#293の受入対象は `80eff153f90e7158c9a85a78b36f8f18f9ab1ab0` の12ファイル。
手動workflowの回帰が読むworkflow自身もsparse checkoutへ含め、モデルなしの
必要5ファイルだけで全8回帰が成功することを確認した。workflowを省く負例では、
実際の既存テストがFileNotFoundErrorになることも確認済みである。
同じ回帰を既存repo-hygieneジョブへ追加し、今後はPR時点でも不足を検出する。
job・権限・trigger・依存追加やgate緩和は行っていない。新HEADのfresh CI受入、
mainへの通常マージ、genuine hostedでの公開CLI実測、文書受入という終了条件は維持する。

## 検証workflowの規約修正 — 2026-10-23 10:14 UTC

統合先#293の受入対象HEADは `525b2a222828adf3b2205cb8a99b44ce10a9860b`。
旧HEAD `8f2f9433`ではCUDA・MetalのCIジョブが成功した一方、repo-hygieneが
setup-uvをhelper経由だけで使用する構成を拒否した。workflow内で既存7件の
モデルなし回帰を直接UV実行するステップと、そのテストのsparse checkoutを追加した。
修正候補そのものへのcanonical hygiene検査、回帰、actionlintは成功済み。
新HEADのfresh CI受入・mainへの通常マージ・実モデルsmokeという残条件は維持する。
後段のHEAD・CI値は各確認時点の履歴である。

## 実行時の統合方針 — 2026-10-23 09:06 UTC

Whisper関連の#291（回帰・監査・公開証拠）、#292（CI入力identity修正）、
#293（公開CLI検証workflow）を、[PR #293](https://github.com/ayutaz/vokra/pull/293)へ
統合した。受入対象はHEAD `8f2f9433919daed9f70b1d6d7d4eb9220282bfd6` の11ファイル。
#291の4ファイルと#292の2ファイルはレビュー済みblobをそのまま保持し、
検証workflowの5ファイルを合わせた。元PRを個別にマージするたびにstrictなbase更新と
同じsource差分のCIを繰り返す待ち時間を減らすための統合であり、検証の省略ではない。

この統合PRのfresh CIと通常マージが、当初の手順2と手順3のworkflow受入を担う。
#291/#292は#293がmainへ受入された後に、取り込み先を記録して整理する。
旧HEADの失敗やcancelを成功へ書き換えない。独立したPillow更新#287は別PRのまま、
受入後mainへ更新してfresh CI・通常マージを行う。#290等の保留条件は維持する。

順序は「統合#293の受入 → genuine hosted公開CLI smokeと#287更新を並行 →
成功証拠・英日導入文書と現況の受入 → 最新main」。実測・文書更新・cleanupという
手順4までの終了条件は変更しない。下記08:33/07:40の個別マージ順は当時の計画である。

## 最新の実行順 — 2026-10-23 08:33 UTC

以下を現在の進行計画とする。後段の07:40 UTC時点の調査・計画は履歴として残す。
現在のmainは引き続き `92a336318a54f659fa991c129aaa991f46ea6231`。
公開中のPRは10件（通常4件、Draft 6件）。この更新は計画の更新であり、
PRのマージ、モデル実行、新規Releaseの公開やX投稿の完了を意味しない。

目標は既存 `v0.4.0-rc.1` のCPUプレビュー告知である。Linuxの配布CLI本体と
固定Whisper ASR / Silero VADの動作を証拠付きで案内できることを終了条件とする。
全モデル・全OS・Apple Metal対応完了は今回の告知範囲に含めない。

### 現在のPRの扱い

| PR | 最新の状態・扱い | 告知までの処理 |
|---|---|---|
| [#291](https://github.com/ayutaz/vokra/pull/291) | Whisper provenance回帰・監査・公開成功証拠を4ファイルでPR化済み。HEAD `41c445f5`。CIは成功67、失敗1、待機13、skip 5 | 告知の必須PR。共通CI障害の解決、最新mainへの整合、fresh CI後にマージ |
| [#287](https://github.com/ayutaz/vokra/pull/287) | Pillow 13.0.0のMIT-CMU一次資料・正準hashへ修正済み。HEAD `1ffe4d9c`を承認済み。CIは成功45、失敗1、待機21、skip 1 | 優先処理。共通CI障害の解決後、必要な更新とfresh CIを経てマージ。Whisper導入経路のモデル検証とは別 |
| [#290](https://github.com/ayutaz/vokra/pull/290) | HEAD `6d1f02dd`へ変更要求済み。Torch更新はCUDA依存closureも変更し、既存のAPI/導入済み依存監査証拠を流用できない | 保留。新しい互換性・native/license証拠とCPU index契約の整理が必要。告知の待ち条件にはしない |
| [#278](https://github.com/ayutaz/vokra/pull/278) | Draft。artifact展開上限とmusl server-compatの失敗が残る | 独立して原因修正・検証。告知後続へ回せる |
| [#271](https://github.com/ayutaz/vokra/pull/271) | production Actions更新。Candle producer artifact未発見の課題 | #278との前提関係を解決してから受入。上限超過と同一原因扱いしない。告知後続へ回せる |
| [#198](https://github.com/ayutaz/vokra/pull/198) | Draft。法務一次資料・版・owner判断が未確定 | 保留。今回の告知で法令適合を主張しない |
| [#191](https://github.com/ayutaz/vokra/pull/191) | Draft、mainとの競合あり。Kyutai依存/upstream・独立reference等が残る | 別モデルcampaignとして保留 |
| [#182](https://github.com/ayutaz/vokra/pull/182) | Draft、mainとの競合あり。VibeVoice native/license/owner・実weight等が残る | #290/#169との責務を整理する別campaign |
| [#169](https://github.com/ayutaz/vokra/pull/169) | Draft、mainとの競合あり。#182への責務移管待ち | #182受入と責務包含を確認するまで閉じない |
| [#152](https://github.com/ayutaz/vokra/pull/152) | Draft、mainとの競合あり。XCodec2依存/native/license/owner・実weight/Appleが残る | 別モデルcampaignとして保留 |

CI件数はこの時点の観測であり、後から起動するjobを含む最終件数ではない。
全PRの一括マージは告知の前提にせず、各保留の解除条件を維持する。

### 告知へ向けた順序と終了条件

1. **共通CI障害を先に直す。** #287と#291のCUDA jobはRustテストの後、
   GGUF identity確認で失敗した。ログではWhisperをHFの可変 `main` から取得する一方、
   期待hashとcache keyが旧公開物 `7e774425…` のままである。
   原因確認と修正では、検証済みの固定revision・hash・cache契約へ揃える。
   hashチェックを削除したり、再実行だけで解決扱いしない。
   実装・workflow変更はLuna担当、管理側が差分と検証証拠をレビューする。
   必要なら独立したCI修正PRを先行させ、受入後に後続PRを最新mainへ更新する。
2. **#287、#291を受入する。** 受入可能な#287を優先し、#291の必須修正・証拠を
   mainへ入れる。各マージ後に後続のbaseを更新し、変更後のHEADでCI・レビュー・
   必須checkと会話解決を確認する。既存dirty guardや保護manifestを混ぜない。
3. **公開CLIの新モデル組み合わせを実行する仕組みを受入し、実行する。**
   既存driverは本物のGitHub Actions環境を要求するため、新しい手動専用workflowを
   独立PRでレビュー・CI・mainへ受入してから、実際のGitHub-hosted runnerで実行する。
   候補5ファイルは一時領域にあり、まだPR化・マージされていない。
   公開RC1 CLI・sidecar・manifest、新Whisper固定pin、固定Silero、音声fixtureを照合し、
   Whisper ASRとSilero speech/silenceの成功を小さなJSONへ記録する。
   VAST実行時の `GITHUB_ACTIONS=true` 要求による拒否は失敗記録として保持し、
   環境を偽装しない。VAST worker `57343389` は削除後の個別不在と全件0を確認済み。
4. **英日導入ガイド・現況・新しい成功証拠をPR化する。**
   step 3が成功した組み合わせだけを案内し、旧pin・過去の結果は履歴として残す。
   現在の共有保留警告は、新しい成功証拠が揃うまで解除しない。
   文書の関連gateとfresh CIを通し、マージ後は最新mainへ戻す。
5. **告知の最終確認と原稿。** 配布URL、導入手順、固定モデル、確認済み環境と制約を
   確認し、CPUプレビューとしてX原稿を作る。投稿操作は原稿・readinessと別に扱う。
   RC1本体の不具合が見つかった場合だけ、修正とRC2の公開を追加してから告知する。

手動smoke workflowの候補レビューと文書準備は、既存PRのCI待ちと並行できる。
最短経路は「共通CI修正 → 必須PRと手動workflow受入 → 配布CLI実測 → ガイド受入 → 告知」。
Scalewayや全catalogの完了はこの経路の待ち条件ではない。

この時点では新しい公開CLI＋新Whisperの機能smokeは未成功であり、告知可能とは判定しない。
詳細な経過と失敗・後続結果は
[実行記録](cpu-preview-announcement-execution-2026-10-23.md)へ追記する。

## 07:40 UTC時点の調査・当初計画

2026-10-23 07:40 UTC のGitHub readbackを基準とする計画。
受入済みmainは `92a336318a54f659fa991c129aaa991f46ea6231`。
開いているPRは9件（通常3件・Draft 6件）。PRのHEAD、レビューとCIは実行時に再取得する。
この計画作成ではcommit、push、PR更新・マージ、CI再実行、有料worker起動、モデル実行、
Release更新、X投稿を行っていない。

## 告知のゴールと現在地

既存の [v0.4.0-rc.1](https://github.com/ayutaz/vokra/releases/tag/v0.4.0-rc.1) を、
CPU-only desktop CLI/C ABIのプレビューとして紹介できる状態にする。
Release ID `416018534`、`draft=false`、`prerelease=true`、22 assetsを再確認済み。
公開ソース/tagは `d84d311e98d0abb2987b5c4e96157ae8f2068e1a` で、現在のmainとは別。
新しい安定版やRCを必須条件にはせず、配布CLIの修正が必要と判明した場合にRC2を検討する。
既存のRelease・tag・assetと過去の検証receiptを上書きしない。

Whisper baseのlicense/source修正と公開は完了している。
新しい固定revisionは `bf440f6064ffc882d53d3694729dc4616dc2cfa7`、
`whisper-base.gguf` は290,859,488 bytes、SHA-256
`5eea7ff64a6ea80a6145e680e7451dc1fcdc92fe049f8aa96b4c60b8c5b86014`。
全245テンソルの固定上流との一致・再刻印前後の一致、回帰、dry-run、公開後readbackは成功。
[修正記録](whisper-base-provenance-repair-2026-10-23.md)と
[成功receipt](evidence/whisper-base-provenance-repair-2026-10-23.json)を根拠とする。
この修正のコード・監査文書・証拠は未コミットで、PRは未作成。

残る告知上の中心課題は、新しい固定モデルと**配布済みCLI本体**の機能確認、および
英日ガイドの更新とそのPR受入である。現行ガイドには旧Whisper pinと共有保留警告が残る。
新しいモデルの公開だけでその警告を消したり、過去のsmoke結果を新しい組み合わせへ転記しない。

## 既存9 PRの処理方針

以下は現在のPR・レビュー・CIと変更ファイルの調査結果であり、各Draftの古い本文を
現在のupstream状態、法的判断、実weight成功へ読み替えない。
表示されたCI件数にはskipもあり、件数だけで受入を決めない。

| PR・確認したHEAD | 現況 | 次の処理 | CPUプレビュー告知との関係 |
|---|---|---|---|
| [#290](https://github.com/ayutaz/vokra/pull/290) `6d1f02dd` | Torch 2.7.1→2.13.0。CI成功69・skip 1、mergeable/clean | lock・依存互換・既存監査証拠の適用範囲をレビュー。#169/#182のCPU index指定との違いを整理してから受入判断 | 最初にレビューする通常PR。CLI同梱依存ではないため、未解決のreference課題を告知の待ち条件に増やさない |
| [#287](https://github.com/ayutaz/vokra/pull/287) `efaf7d02` | Pillow更新。CI成功69・skip 1だがCHANGES_REQUESTED、mainより遅れている | 正準lock/package/license hashとPillowの版別出典を更新。Diffusers 0.32.2の利用箇所を確認し、レビュー解決・main取込み後のfresh CIを通す | 優先する通常PR。古いconsumer証拠や別モデルの試験で代用しない |
| [#278](https://github.com/ayutaz/vokra/pull/278) `11e33a10` | Draft。Candle applied advisoryとmusl server-compatが失敗 | artifact展開上限の契約不一致とserver-compat失敗を個別に診断。保護されたレビュー対象の境界を守り、上限緩和・読取迂回をしない | #271のsource/workflow前提。修正・CI・レビューが揃うまで保留できる |
| [#271](https://github.com/ayutaz/vokra/pull/271) `d777dd71` | production actions 9件更新。Candle applied advisoryが失敗 | #278受入後に最新mainへ整合。producer artifact未発見を独立に確認し、更新workflow全体のfresh CIを確認する | #278→#271の順。既存Releaseの配布物を変更するための必須PRではない |
| [#198](https://github.com/ayutaz/vokra/pull/198) `5614a92c` | Draft、documentation-links失敗。一次資料アクセス・法務判断の保留記録あり | 正確な一次資料・版・適用範囲・owner/legal判断を解決してから再評価。403の一律許可や引用削除はしない | 法令適合やTTS/voice完了を今回の告知で主張しない。全面解決待ちにしない |
| [#191](https://github.com/ayutaz/vokra/pull/191) `63cadf3c` | Draft、documentation-links失敗。Kyutai依存/upstream・独立reference等の責務が残る | 実行時にupstream契約とpatched依存を再確認し、独立PCM/KV/reset等を受入条件として維持 | Whisper/SileroのCPU導入経路とは別のモデルcampaignとして保持 |
| [#182](https://github.com/ayutaz/vokra/pull/182) `a3afc1ec` | Draft。観測CIに失敗なしだがnative/license/owner・実weight等の責務あり | #290/#169とCPU lock・collector責務を整合し、必要な証拠・レビューを揃える。green/skipをモデル完了としない | VibeVoice Realtimeは今回の告知実証対象に追加しない |
| [#169](https://github.com/ayutaz/vokra/pull/169) `b3e9d5c1` | Draft、documentation-links失敗。#182への責務移管待ち | #182がmainに受入済みで、より強いcollector/security責務の包含を確認した後に整理する | #182受入前に重複扱いで閉じない |
| [#152](https://github.com/ayutaz/vokra/pull/152) `a9f54e1a` | Draft。観測CIに失敗なしだが依存/native/license/API/owner・実weight/Apple gateが残る | 保留条件を維持し、XCodec2の別campaignとして進める | CPUプレビュー告知の前提に追加しない |

特に#271のproducer artifact未発見と#278の展開サイズ超過を同じ原因として扱わない。
#290は単純な版変更、#169/#182は公式CPU indexと `2.13.0+cpu` の契約を含み、
同じバージョン文字列でも同一の依存closureではない。重複・置換判断には内容比較が必要。
#287の既存変更要求はGitHubレビューでも確認済み。CI greenだけで解除しない。
GitHubがmergeabilityをUNKNOWNと返したPRは、競合なしとも競合ありとも断定しない。

## 実行順序

1. **通常PR #287・#290のレビューと処理を先行する。**
   #287は具体的な変更要求を修正し、#290はCPU lock等との整合を確認する。
   修正が必要な実装・テスト・lock/configはLunaが担当し、Solが差分・証拠をレビューする。
   正しいsource/license/guard状態を保持したうえで、受入可能なPRを一件ずつ通常マージする。
   main更新で厳密なbase条件が変わるため、後続PRのHEADとCIを改めて確認する。
   解決できない課題は具体的な理由と次の条件を残して保留し、告知の必須経路を進める。

2. **未コミットのWhisper修正をPRとして受入する。**
   最新の受入済みmainへ整合し、回帰・一次資料の訂正・成功receiptを論理単位でcommitする。
   既存dirty guardを混ぜず、保護manifestを読み取り・変更・stageしない。
   変更後の検証が必要なら該当範囲だけ遠隔で確認し、PRのfresh exact-head CI、
   必須context/app、レビュー・会話解決等を満たして通常マージする。
   先のVAST結果を変更したコードへ付け替えない。

   公開CLI・モデル・fixtureの固定契約が確定していれば、このPRのCI待ちに次の遠隔smokeを
   並行できる。検証入力が変わった場合は、その結果を新しい入力へ流用しない。

3. **新Whisper pinと公開済みRC1の組み合わせを遠隔で確認する。**
   VAST等の許可された遠隔環境で、公開ReleaseのCLI本体・sidecar・manifestのidentityを
   確認し、上記の新しいWhisper revision/hashと固定音声fixtureでCPU ASRを実行する。
   Silero VAD v5も固定pinを維持してspeech/silenceを確認する。既存の検証driver・
   `--cli`経路を確認して再利用し、model pin・binary・fixture・command・exit・出力判定を
   小さなreceiptへ記録する。新しいビルドの成功だけで配布バイナリの合格としない。
   maintainer Macではモデル取得・実行をしない。workerは結果回収後、保存データごと削除する。
   RC1に実際の不具合がある場合は告知を止め、修正・CI・新RC2公開の計画へ切り替える。

4. **導入ガイドと現在情報のPRを受入する。**
   英日ガイドの新pin・SHA・新しい機能smoke証拠を整合し、確認できた範囲で保留警告を更新。
   `docs/current-status.md` / `.ja.md`、文書入口と告知readinessにも日付付き現況を反映する。
   旧pin・旧結果は日付付き履歴として保持し、過去のreceiptは上書きしない。
   pin/configやchecker変更が必要ならLunaへ担当を割り当てる。
   公開Releaseの既存tagにある古い固定pinを参照するworkflowと、新しい導入経路を区別する。
   文書・リンク・pin/CLI example等の関連gateとfresh CIを通し、通常マージして最新mainへ戻す。

5. **告知対象の最終確認とX原稿を作る。**
   公開URL、導入先、各固定identity、現在の重大警告の到達範囲、マージ結果を再確認する。
   「CPUプレビュー」「Linuxで確認したWhisper ASR/Silero VAD」と配布OSの範囲を明記する。
   macOS/Windowsの配布・packaging結果と実モデルの検証範囲を区別する。
   全catalog完成、全モデルMetal完了、独立数値parity、品質/速度保証、全面的security-clean、
   法令適合、外部package registry公開を今回の成果として主張しない。
   Xの公開操作は原稿作成と分け、ユーザーの投稿方針に従う。

## 告知の終了条件

- Whisper修正と新しい導入手順がmainへマージ済みで、該当PRのfresh CIが受入済み。
- 配布済みRC1 CLI＋新Whisper pin、固定Silero pin＋speech/silenceの遠隔smokeが成功し、
  source/build/run・入力identity・結果を結び付けた新receiptを回収している。
- 英日ガイドを辿って、公開済みassetと正しい固定モデルへ到達できる。
- 告知経路へ到達する重大な未解決問題が見つかった場合は解決済み、または告知を保留している。
  reference環境の全警告を解消したとは主張しない。
- 上記9 PRの受入・保留・置換条件が明確で、失敗したPRをgreen扱いしていない。
- 作業用workerと保存データを削除し、個別不在を確認している。

#278→#271、#182→#169整理、#191/#152の実weight/Apple campaign、#198の法務出典解決は
継続課題として残せる。今回のCPU告知のために全Draftをマージしたり、Scalewayで全モデルを
検証する計画にはしない。Scalewayは全catalogのApple CPU/Metal/no-fallback campaignの最後。

## 調査と権限の限界

PR本文・変更ファイル一覧・必要なmetadata patch・レビューとGitHub CIを読み取った。
通常3 PRのうち、#290はclean、#287はbehindかつchanges requested、#271はCandle失敗が残る。
一部のcheck rollupは100件のため、上表は全workflow成功を保証する件数表ではない。
マージを実行する段階では、ページングしたchecks、必要な実run/jobと保護設定を確認する。

ローカル安全フックはworkflow/pin/driver等をまとめた読み取りコマンドも拒否した。
その要求を別形式・別ツール・遠隔へ移して再取得せず、既取得の情報とPR/API記録で計画した。
#278のPR本文に記録された保護対象レビュー境界も維持している。実装時に必要な正確な
レビュー範囲を確定し、保護解除やgate緩和を計画の前提にしない。
