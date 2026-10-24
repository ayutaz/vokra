# CPUプレビュー告知原稿と最終確認 — 2026-10-24

後続のPR処理は[07:44 UTCの記録](pr-processing-readback-2026-10-24.md)を参照。
以下の03:33–03:38 UTC時点の確認値は、その時点の記録として保持する。
後続のmain更新は配布済みRC1のsource・モデルidentityや告知範囲を変更しない。

## 対象と確認結果

既存の [v0.4.0-rc.1](https://github.com/ayutaz/vokra/releases/tag/v0.4.0-rc.1)
を紹介するための原稿。新Releaseの作成やXへの投稿を実施した記録ではない。
受入済みmainは `23a6d31397331ed36ed109f2f4863c64858c57be`。
英日ガイドと成功証拠の文書PR [#296](https://github.com/ayutaz/vokra/pull/296)は
通常マージ済みで、ローカルmain・origin/mainも同commitへ更新済みである。

2026-10-24 03:33–03:38 UTCのGitHub metadata確認では、Releaseは
`draft=false`、`prerelease=true`、22 assets、公開日2026-10-17である。
配布版sourceは `d84d311e98d0abb2987b5c4e96157ae8f2068e1a` であり、
現在のmainを配布済みと説明しない。

告知の根拠は、受入済みsource `0d51bb01e56da6ce5f72375cc5fdddc599beac3c` の
[最終Linux CPU run](https://github.com/ayutaz/vokra/actions/runs/39953770618)と
[保存したraw receipt](evidence/published-rc1-cli-pairing-final-2026-10-23.json)。
公開済みRC1 CLIにCPUを明示し、修正済みWhisper baseのASR、Silero VAD v5の音声・無音の
3コマンドがすべてexit 0だった。これは機能smokeの証拠である。

| 対象 | 検証済みの固定identity |
|---|---|
| Linux glibc CLI SHA-256 | `2a5d4ba5ca543b749f6738bd92d59da0e884a6693ab7e222b8b38ce81a91d8c4` |
| Whisper revision | `bf440f6064ffc882d53d3694729dc4616dc2cfa7` |
| Whisper GGUF SHA-256 | `5eea7ff64a6ea80a6145e680e7451dc1fcdc92fe049f8aa96b4c60b8c5b86014` |
| Silero revision | `3670dfed054980307eeb1711c6329c29b80bdf14` |
| Silero GGUF SHA-256 | `23b19eadba72fb9726fa73a0bfd095afcba2f8c5bcb4f03a1b5e57a67eb13df6` |

英日導入手順は、[English](../release-0.4.0-rc1.md) /
[日本語](../release-0.4.0-rc1.ja.md)で同じ固定identityと検証範囲を説明する。
過去の入力不足による2回の失敗は[実行記録](cpu-preview-announcement-execution-2026-10-23.md)に
別々に保持され、成功へ書き換えていない。

10月24日のHF固定カードへのweb取得はtool上で失敗し、ローカルのmetadata-only API GETも
model-download guardに拒否されたため停止した。その後、既存guardが明示許可するHEADだけで、
両モデルの固定カードURLと固定GGUF配布URLの計4件がHTTP 200になることを確認した。
本文を取得する別toolやremoteへの迂回、guard変更はしていない。
HEADは当日のリンク到達性の証拠であり、上のsize/hashを再測定した証拠ではない。
固定identityは10月23日の受入済み証拠に基づき、今回モデル本体は取得・実行していない。

## X本文案

音声推論ランタイム Vokra v0.4.0-rc.1 のCPUプレビューを公開しています。
Whisper baseの音声認識とSilero VAD v5のLinux CPU動作を公開CLIで確認しました。
導入手順・検証範囲：
https://github.com/ayutaz/vokra/blob/main/docs/release-0.4.0-rc1.ja.md

## 補足返信案

検証済みの範囲は固定したモデルとLinux CPUです。プレビュー版のため、全モデル・他OSでの
実モデル動作・Metal・数値parity・性能の検証完了を意味しません。
配布物とchecksumはこちら：
https://github.com/ayutaz/vokra/releases/tag/v0.4.0-rc.1

## 投稿前の取り扱いと後続作業

実際のX投稿は未実施。上記は原稿であり、投稿先アカウントや送信操作を確定するものではない。
現在の導入先は公開CLIのガイドで、Python参考環境のinstallを告知の手順に加えない。
新Release、HFモデル更新、package registry公開はこの原稿の作成に含めない。

2026-10-24 03:33 UTCの未完了PRは9件（#297、#290、#278、#271、#198、#191、#182、#169、#152）。
前日の8件に新規Pillow更新#297が加わった。その後、CUDA構成の#290は未マージでcloseし、
CPU専用の依存修正とその受入責務を#169/#182に残した。mainの依存修正が済んだという意味ではない。
これらは個別のsource・CI・依存・ライセンス・
owner・実weight条件で処理し、全9件のマージを既存CPUプレビュー紹介の条件にしない。
全catalogとApple CPU/Metal/no-fallbackの残作業は[現況](../current-status.ja.md)の
別の完了条件に従う。セキュリティ警告をすべて解消済みとも説明しない。
