# プロジェクトの現状

[English](current-status.md) | **日本語**

**確認日: 2026-10-23 UTC**。この文書更新前の実測・ソース確認の基準は、受入済みGitHub main
[`0d51bb01e56da6ce5f72375cc5fdddc599beac3c`](https://github.com/ayutaz/vokra/commit/0d51bb01e56da6ce5f72375cc5fdddc599beac3c)
です。最終モデル実測はこのソースで行い、本更新はruntime codeを維持しています。
このページを現行情報の入口とし、過去のhandoff・測定結果は、そのcommit・
承認範囲・環境・制限に限定した履歴として読みます。

## リリースとソース

| 対象 | 確認した状態 |
|---|---|
| workspaceのソース版 | `Cargo.toml`は`0.4.0-rc.1`。現在のmainには公開後の保守変更も含む |
| 最新安定版GitHub Release | [`v0.3.0`](https://github.com/ayutaz/vokra/releases/tag/v0.3.0)、2026-09-20公開、18 assets |
| 公開済みGitHub preview | [`v0.4.0-rc.1`](https://github.com/ayutaz/vokra/releases/tag/v0.4.0-rc.1)、2026-10-17公開、22 assets。安定版ではない |
| previewの正確なソース | `d84d311e98d0abb2987b5c4e96157ae8f2068e1a`。現在のmainとは別 |
| previewの範囲 | CPU-only desktop CLI/C ABI、hosted Linux Whisper/Silero機能smokeとnative packaging検証。全モデル・Apple/Metal parityではない |
| 外部package registry | 別途gateされた公開先。GitHub preview公開はcrates.io・PyPI・npm・OpenUPM・Godot AssetLib公開を意味しない |

そのreleaseの利用には[preview導入ガイド](release-0.4.0-rc1.ja.md)を参照してください。
Python・Unity・Godot・webは配布状況が別々です。workspaceの版から配布済みと推定せず、
各bindingのREADMEを確認します。pre-1.0のAPI/ABI安定性や全catalog完了は保証しません。

Whisperのprovenance・MITを修正したartifactは、固定revision
`bf440f6064ffc882d53d3694729dc4616dc2cfa7`で公開済みです。
[修正証拠](handoff/whisper-base-provenance-repair-2026-10-23.md)を参照してください。
[受入済みmainの最終実測](https://github.com/ayutaz/vokra/actions/runs/39953770618)と
[保存したreceipt](handoff/evidence/published-rc1-cli-pairing-final-2026-10-23.json)で、
公開RC1 Linux CLIと修正済みWhisper・固定Sileroの組み合わせを確認しました。
CPU明示の3コマンドがすべて終了コード0で、Whisper keyphraseが一致、Sileroは
音声214/343 frames・無音0/343でした。先の入力不足2件とsourceの先行成功結果は、
[実行記録](handoff/cpu-preview-announcement-execution-2026-10-23.md)に別々に保持しています。
これは固定した組み合わせの機能確認で、全モデル・他OSの実モデル推論・数値parity・
Metal・性能の検証ではありません。

## mainのCIと未完了PR

Whisper統合の[#293](https://github.com/ayutaz/vokra/pull/293)と、レビュー済みPillow metadata更新の
[#287](https://github.com/ayutaz/vokra/pull/287)、fixtureパス修正の
[#294](https://github.com/ayutaz/vokra/pull/294)・[#295](https://github.com/ayutaz/vokra/pull/295)は
通常マージ済みです。それぞれレビューした
正確なsource HEADで必須context/app 16組と関連監査が成功しました。
PR受入、その後のmain push CI、成功した手動機能実測は区別します。各HEADの件数と
optionalジョブの観測は実行記録に日付付きで残し、全現行workflow成功という判断へ
読み替えません。無関係なdirty guardも別の差分です。mainの必須context/app 16組、strict更新要件、
管理者への強制、linear history、会話解決要件は変更していません。

保留中の後続PRは8件（うちdraft 6件）です。

| PR | head先頭 | 残る責務 |
|---|---|---|
| [#290](https://github.com/ayutaz/vokra/pull/290) | `6d1f02dd` | Torch/CUDA reference closure変更。新しい互換性・native/license証拠とCPU index責務の整理が必要。変更要求中 |
| [#278](https://github.com/ayutaz/vokra/pull/278) | `11e33a10` | draft。Candleのsame-run producer・receipt・consumer契約。サイズ／認証gateは緩和しない |
| [#271](https://github.com/ayutaz/vokra/pull/271) | `d777dd71` | production action 9件の更新。producer artifact未発見は#278の展開サイズ超過と別問題 |
| [#198](https://github.com/ayutaz/vokra/pull/198) | `5614a92c` | draft。日付付き法令出典の照合。法務sign-off・マージ済みpolicyではない |
| [#191](https://github.com/ayutaz/vokra/pull/191) | `63cadf3c` | draft。Kyutai DSM PCM/referenceの整合 |
| [#182](https://github.com/ayutaz/vokra/pull/182) | `a3afc1ec` | draft。別モデルであるVibeVoice Realtime runtime/referenceの統合 |
| [#169](https://github.com/ayutaz/vokra/pull/169) | `b3e9d5c1` | draft。VibeVoice CPU reference依存closure |
| [#152](https://github.com/ayutaz/vokra/pull/152) | `a9f54e1a` | draft。XCodec2のpre-import・依存・証拠gate |

#284・#286・#288・#293・#287・#294・#295は受入済みmainへマージされています。
#291/#292は#293へ取り込んでcloseし、個別マージとは区別します。通常CIがgreenでも、source・
license・owner・実weight gateの解決にはなりません。
[告知計画](handoff/cpu-preview-announcement-plan-2026-10-23.md)で必須経路と保留中の
モデル・CI・法務課題を分けています。

## モデルとApple対応

2026-10-23に公開APIとmodel cardだけを再取得し、**公開194 repositories／GGUFあり193／
GGUF 198 files**を確認しました。[全行のsnapshot](handoff/model-inventory-2026-10-23.tsv)の
SHA-256は`9ef3eb07f12a98de7f425dcd75b98f4cd1be45ab1d3907aafeca2626196cdd56`です。
分類はソース・artifact到達性であり、新たな実モデルparityではありません。

| 分類 | 件数 |
|---|---:|
| CPU code/artifact full | 136 |
| CPU partial | 43 |
| 完全なruntime binderなし | 14 |
| artifactなし | 1 |
| Metal code/artifact full | 136 |
| CPU未完了によりMetal blocked | 57 |
| Metal artifactなし | 1 |

現在の公開分類では**未完了58件**で、CPU fullなのにMetal未対応という分類は0件です。
凍結された**owner判断packetの63行**は別の日付付き・hash-bound範囲であり、58へ書き換えません。
過去の[Apple結果](handoff/mac-cpu-metal-scaleway-results-2026-09-11.md)・個別許可された
artifact公開は記録された範囲だけに有効で、全catalogは**完成していません**。

残りの順番は、PR/source/依存/license/owner前提の解決 → 許可されたVAST実weight変換・
固定した独立upstream reference・CPU parity → レビュー済みclean HEADと新しい認証packet →
**最後にScaleway**でApple CPU/reference・Metal/reference・Metal/CPU・明示no-fallback →
結果の受入とcleanup → 別途許可された公開です。実BF16 checkpoint・HiFTNet・
**BigVGAN全4 variant**も引き続き必須です。
[実行計画](handoff/mac-cpu-metal-execution-plan-2026-09-07.md)・
[残作業ledger](handoff/mac-pre-scaleway-remaining-tasks-2026-09-05.md)・
[owner packet](handoff/mac-cpu-metal-owner-disposition-packet-2026-09-07.md)を参照してください。

## セキュリティ・法務・運用の境界

2026-10-23の最終GitHub全ページ再確認で、Dependabot未解決は**599件**（critical 90／high 236／
medium 189／low 84、修正版の指定あり575／なし24）です。すべて`tools/parity/`または
`integrations/`配下のpip警告で、critical/highは326件です。
[日付付きmetadata集計receipt](handoff/evidence/announcement-security-final-readback-2026-10-23.json)に
件数とmanifest分布を記録し、以前の572件・638件のsnapshotは履歴として保持します。
Code Scanning未解決は8件で、
Scorecard 4件とserverテスト中のCodeQL `rust/log-injection` #68–#71です。
テスト中という理由だけでfalse positiveとせず、dismissもしていません。repository全体・
reference環境等のalertであり、first-party-onlyのroot runtimeへ外部依存を追加したものでは
ありませんが、セキュリティ上の全面的なclean判定もできません。

[license audit](license-audit.md)の既存判断と署名は維持します。
[法務ガイド](legal-compliance.md)では出典の取得と現行法の確認を区別します。EUR-Lexの
bot確認とCaliforniaのHTTP 403により、法令全体を新たに確認済みとはしていません。
通常生成経路への自動watermark接続は`Deferred`のままで、standalone AudioSealは法令適合の
保証ではありません。

maintainer Macでのモデル取得・実行、workspace全体／`vokra-models`のCargoは禁止のままです。
文書checkはstatic/offlineに限定します。告知作業では削除済みVAST workerとgenuine hosted
Actionsを使い、それぞれの失敗・cleanup証拠を
[実行記録](handoff/cpu-preview-announcement-execution-2026-10-23.md)へ保存しています。
手順1–4には新Release・registry・HF・X公開を含めません。過去の削除済みIDを現在のworkerや
転送元として扱いません。

前の作業のlocal-model guard修正は未コミットの別差分として保持しています。オフライン検証済み
ですが、mainへマージ済み・モデル対応完了という意味ではありません。

## 文書監査

[2026-10-23のcoverage・検証記録](handoff/documentation-refresh-2026-10-23.md)に、対象・根拠・
保存した履歴・static gate結果・実行未検証の軸を記録します。
[文書index](README.md)を入口に、古いcheckpointを現在の実行手順と取り違えないでください。
