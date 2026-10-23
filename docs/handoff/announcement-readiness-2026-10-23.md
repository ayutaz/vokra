# CPUプレビュー告知前確認 — 2026-10-23

## 新しい組み合わせの先行実測成功 — 2026-10-23 12:12 UTC

#294は通常マージ済み。sidecar不足を修正する#295のレビュー済みsource
`e17f1446`で、[genuine hosted run 39951135015](https://github.com/ayutaz/vokra/actions/runs/39951135015)が成功した。
[raw receipt](evidence/published-rc1-cli-pairing-preflight-2026-10-23.json)のartifact digest・
source/run・公開バイナリと固定モデルのidentity・3経路のCPU指定と実出力を独立確認した。
Whisper keyphrase、Silero音声214/343 frames・無音0/343の結果は当該runの実測である。

**最終告知判定は引き続き保留。** 先行確認を最終main結果の代用にせず、#295のfresh CI・
通常マージ・tree照合後のmain実測、成功証拠と英日ガイド/README・現況の文書PR受入を残条件とする。
旧失敗・旧pin・過去の結果は下記に履歴として保持する。

## 受入後の実測と現在の残条件 — 2026-10-23 11:40 UTC

Whisper provenance修正と監査・成功証拠・検証workflowを含む#293はmainへ受入済み。
#291/#292は取り込み先を記録してcloseし、Pillow metadata修正#287も通常マージ済みである。
受入済みmainは `4d6efd02a95a3585b98ca718ddc0a6363b634ef3`。
新WhisperのMIT/provenance修正済み公開状態は、下記の修正記録と公開成功receiptに従う。

一方、genuine Actionsの[公開CLI実測run](https://github.com/ayutaz/vokra/actions/runs/39944971172)は、
wrapperが一時音声パスを渡したことにより、元driverの固定fixtureパスgateで停止した。
[raw失敗receipt](evidence/published-rc1-cli-pairing-failure-2026-10-23.json)のモデル・コマンド欄は空。
既存RC1 binaryと入力identityの照合成功は、推論成功とは区別する。
修正[#294](https://github.com/ayutaz/vokra/pull/294)はレビュー済み2ファイルと
オフライン回帰9件でPR化済みだが、fresh CI・通常マージ・新しい実測は未完了。

**新Whisperと公開済みRC1 CLIを案内する告知は引き続き保留。**
#294受入、新しい3経路の成功receipt回収・独立確認、英日ガイド/現況の文書PR受入を残条件とする。
現在のセキュリティ再確認は
[638 Dependabot／8 Code Scanning](evidence/announcement-security-readback-2026-10-23.json)で、
後段の572件は以前の観測である。全pip警告はtools/parityまたはintegrations配下に属し、
配布CLIへ同梱される依存とは区別するが、全面的なsecurity-clean判定には使わない。

## 後続の修正と現在の受入待ち — 2026-10-23 09:29 UTC

以下の当初判定は、旧Whisper pinの調査時点の記録として保持する。
その後、[Whisper修正記録](whisper-base-provenance-repair-2026-10-23.md)と
[公開成功receipt](evidence/whisper-base-provenance-repair-2026-10-23.json)に従って、
MITの一次資料・具体的なsource revision・テンソル内容を確認した新しいartifactを公開した。
現在の修正済み固定revisionは `bf440f6064ffc882d53d3694729dc4616dc2cfa7`。
`whisper-base.gguf` は290,859,488 bytes、SHA-256
`5eea7ff64a6ea80a6145e680e7451dc1fcdc92fe049f8aa96b4c60b8c5b86014` である。
旧pinのlicense/provenance記述を、この新artifactの判断へ流用しない。

回帰・監査・公開証拠、固定GGUFを使うCI修正、配布済みCLIを試す手動workflowは、
[PR #293](https://github.com/ayutaz/vokra/pull/293)の
`8f2f9433919daed9f70b1d6d7d4eb9220282bfd6`へ統合した。
受入済みmainは引き続き `92a336318a54f659fa991c129aaa991f46ea6231`。
この時点で統合HEADのCIはGitHubのrunner割当待ちで、PRは未マージである。

**新Whisperと公開済みRC1 CLIの組み合わせは、まだ機能smoke未完了。**
モデルの公開修正だけで配布CLIとの互換性を成功扱いせず、手動workflowの受入・
実際のhosted実行・結果回収と英日ガイド更新を告知前の残条件として維持する。
現在の進行順は[告知計画](cpu-preview-announcement-plan-2026-10-23.md)、
実行・失敗・cleanupの事実は[実行記録](cpu-preview-announcement-execution-2026-10-23.md)を参照する。
X投稿、新規Release、全モデル・Apple/Metal完了はこの追記の成果ではない。

## 対象と判定

依頼された計画の1（文書push・PR・CI・通常マージ）と2（公開対象の最終確認）を扱う。
X投稿、新規Release、モデル更新、HF公開、Scaleway検証はこの作業の対象外。
公開対象は既存のGitHub prerelease `v0.4.0-rc.1` に限定する。

**確認時点の判定: Whisper GGUFのlicense/provenance説明に未解決の不整合があるため、
Whisperを試せるという積極的な告知は保留。** バイナリ配布・既存の機能smoke・
セキュリティ警告の到達範囲は下記で区別する。これは一括したsecurity-clean、
法務sign-off、全モデル対応完了の判断ではない。

ソース基準は受入済みmain `151d94a8613d5bd4acb1671a461f59742c37b8fc`。
文書更新は `efc6e952fe429095a56ebde60ce9a530cc5ff9cb`、文書push判定修正は
`bf53e3605debf8dc402320923a882615ecaf8c2e`。これらの新規受入状況は末尾で分離する。
公開RCのsource/tagはこれらとは異なる `d84d311e98d0abb2987b5c4e96157ae8f2068e1a`。

## 配布物と既存検証の再確認

| 項目 | 確認した証拠 | 限界 |
|---|---|---|
| 公開状態 | [Release](https://github.com/ayutaz/vokra/releases/tag/v0.4.0-rc.1) ID `416018534`、`draft=false`、`prerelease=true`、22 assets | 安定版Latestは`v0.3.0`。新たな公開・差替ではない |
| tag/source | GitHub git-ref APIは `d84d311e98d0abb2987b5c4e96157ae8f2068e1a` のcommitへ直接解決 | 現在のmainの保守変更を配布版へ遡及しない |
| 元の公開CI | [run 39062234813](https://github.com/ayutaz/vokra/actions/runs/39062234813)、attempt 1、10 jobs全てcompleted/success | 今回新たにbuild・モデル実行した結果ではない |
| manifest | 公開JSON 2,453 bytes、SHA-256 `32b03f537f48ff6f2bb7b12fda1a9426090debbd59da361c0ffa59e8f7a60e92`。10 payload全件のsize/digestが現在のasset APIと一致 | バイナリ本体は今回取得せず、公開metadataと元の公開検証を照合 |
| checksum | `.sha256`テキスト11件を取得。テキスト自体のAPI digestと、対象ファイル名・payload digestを全件照合 | 新たな全バイナリ再ダウンロードではない |
| CLI SBOM | 7,706 bytes、SHA-256 `2e56df7b51859bd2f61b2fc3f9fd942be3f8c0b6db8f298843e5f7f4d605ba0a`、first-party 12 packages | normal runtime Cargo closure。OS・toolchain・全ソースの安全性を保証しない |
| C ABI SBOM | 6,545 bytes、SHA-256 `cb4c13f199f9d7bdf4059db645541f0e6720d53b851c833c0ef9030922bb1a24`、first-party 10 packages | 同上。Python参考環境や独立integrationはこのclosureに含まれない |
| 元のCPU receipt | [公開workflow run 39062234813](https://github.com/ayutaz/vokra/actions/runs/39062234813) のartifact ID `12111439059`から認証済みREST APIで`receipt.json` 5,395 bytesを再取得。SHA-256 `d4b4656fbfa67c8f10b7dd5f6642fc896780ab11da4bc4db80d7922bb2348697` | artifactの直接UI/downloadには認証が必要。小さなJSONだけをメモリ上で確認し、モデル・バイナリ・音声本体は取得・実行しない |

receiptは元のassembled Linux CLI 13,344,272 bytes / SHA-256
`2a5d4ba5ca543b749f6738bd92d59da0e884a6693ab7e222b8b38ce81a91d8c4`、
source/run/attempt/manifestと一致する。ASRはJFK keyphrase、VADはspeech 214/343 frames、
silence 0/343 framesを記録し、3コマンドのreturncodeはいずれも0。
これは機能smokeだけであり、独立数値parity・WER/CER・性能・Apple/Metal結果ではない。

## 重大警告と導入経路

GitHub Dependabot APIを全ページ読み取り、572 open alerts（critical 93、high 247、
medium 173、low 59）を確認した。**全572件のecosystemはpip**。
critical/highの340件をmanifestパスごとに確認し、`tools/parity/**` と
`integrations/vokra-misaki-g2p/uv.lock` に属することを確認した。
全severityでも `tools/parity/**` / `integrations/**` 以外の対象パスは0件。
これは検出済み警告の対象を分類した結果であり、未知の脆弱性がないという証拠ではない。

公開RCの上記SBOMとtagのroot lockfileはfirst-party `vokra-*` のみ。
告知の導入先を[公開済みCPUプレビューガイド](../release-0.4.0-rc1.ja.md)へ限定すると、
利用者は同梱済みCLI・checksum・固定GGUF・固定音声fixtureを使用し、
`uv sync`、Python参考環境、変換環境、サーバーやG2P integrationの導入を必要としない。
従って今回確認したpip警告はその配布CLIの同梱依存ではない。
参考環境の安全性・導入推奨や全警告の解消を、この到達性の分類で代用しない。

Code Scanningは8件未解決のまま：serverテストのCodeQL `rust/log-injection`
#68–#71（medium）とScorecard 4件。テスト場所だけを理由にfalse positiveとせず、
dismiss・例外追加・保護設定の緩和はしない。生のAPI取得失敗は0 alertsとして扱わない。

## 固定モデルの確認と告知保留理由

公開HF revision API（ファイル名・size・LFS SHA-256だけ）と固定カードを確認した。
モデル本体は取得していない。

| モデル | 固定revision・file identity | カードのlicense |
|---|---|---|
| Whisper base | `dddd93ff6301a82aa3750d398cb2e0a599cd9e83`、290,859,360 bytes、SHA-256 `7e774425585d6e9ba58ac5337b406522b1408ab6ff6765d2938b092aef4c8e27` | `apache-2.0`（要照合） |
| Silero VAD v5 | `3670dfed054980307eeb1711c6329c29b80bdf14`、2,183,744 bytes、SHA-256 `23b19eadba72fb9726fa73a0bfd095afcba2f8c5bcb4f03a1b5e57a67eb13df6` | `mit` |

双方とも元のCPU receipt / pin manifestのsize・SHA・revisionと一致する。
ただし[Whisper固定カード](https://huggingface.co/vokra/whisper-base/blob/dddd93ff6301a82aa3750d398cb2e0a599cd9e83/README.md)は
upstream sourceを具体的なrepo/revisionに結び付けず、upstream licenseもApache-2.0とする。
一方、[OpenAIの一次LICENSE](https://github.com/openai/whisper/blob/main/LICENSE)はMITであり、
既存の[license audit](../license-audit.md)でもWhisper weight/codeをMITとして記録している。
この違いを、両方permissiveだから問題なしと判断しない。
カードはGGUF metadata由来とも説明するが、今回GGUF headerを読んでいないため、
artifact内部まで確認済みとは主張しない。

必要な次の判断は、正しいupstream revision・license・attributionと既存artifactの
provenanceを確定し、カードだけの修正で足りるのか、artifactの再刻印・公開が必要かを
分けることである。HFの変更やartifact更新は別の公開権限・gated workflowを必要とする。
今回の許可から`publish-one.sh --push`、weight差替、owner sign-offを推定しない。
保護対象manifestには触れない。

## 文書push修正と受入の状態

`bf53e360`の修正は、`tools/**`・`integrations/**`・fixture以外の`tests/**`の
Markdown文書をdocs fast-pathへ正しく分類するもの。
fixture、コード、script、build/config、hook自身の変更、未解決base、空diff、
`VOKRA_HOOK_DEEP=1`は従来どおりdeepとなる。maintainer Macのdeep拒否は維持。
既存`.codex/hooks/guard-local-models.sh`差分はこの作業に含めない。

ローカルのclassifier＋production-hook回帰は63/63成功。Cargo/compliance境界をmockする
回帰であって、ローカルモデル実行・workspace Cargo・実weight検証ではない。
元の文書commitのpre-commit、文書参照、runbook引用、diff hygieneも成功済み。

この記録を書いた時点では、修正コードのVAST静的回帰・cleanup証拠、push、
新規PRのfresh exact-head CIと通常マージはまだ未完了。
それらの実結果はPR/最終引渡しで別途確認し、この時点の状態を成功扱いしない。
全モデルcampaignやScalewayは告知準備とは別の残作業である。

### 遠隔静的回帰とcleanupの追記（2026-10-23 03:58 UTC）

hookを含む変更は引き続きdeepに分類されるため、Macで重いpre-pushを実行せず、
VASTで変更対象に限定したモデルなしshell回帰を実施した。Rust source、Cargo manifest、
root lockfileの変更はなく、workspace Cargoや実weightの検証を代替する主張ではない。

- 対象source: `bf53e3605debf8dc402320923a882615ecaf8c2e`。
- 転送archiveは下記3ファイルのみ。SHA-256
  `30501ef5fba2ba6316c5b2516427cafb853c255f352411633f739720e510d681`。
  `.env`、HF token、モデル、保護manifest、既存dirty guardは転送していない。
- `.githooks/pre-push`:
  `c2965376708c09ff1bad1397103c751332761d01ee2b99cda8be7195deb5dc02`。
- `.githooks/lib-fastpath.sh`:
  `0476c336440553a66b3153c0ba78b4747800daad530745df49ce5acfe315cdcc`。
- `scripts/test-pre-push-fastpath.sh`:
  `675012335ddca467c1ccfd66ec8b39ef1064cc3228994ed7574d947e0163b072`。

最初のinstance `57317301`（固有label `vokra-hook-static-444e8928d2ff`、
storage込みUSD 0.135556/hour）は、provider応答内の料金集計をinstance本体と誤認して
転送前に停止した。実応答形状をoffline回帰へ追加して修正し、rootが個別ID/labelを
確認してdestroyした。個別readbackは `instances: null`。

再試行instance `57317485`（固有label `vokra-hook-static-505ddac7cc56`、
storage込みUSD 0.122223/hour）は、所有確認、SSH転送、remote shell検証の各段階で
exit 0を記録した。remote shellはarchiveと3ファイルのhash、tar member allowlist、
`bash -n`、実回帰script成功、`OK` case数63、出力64 KiB以下をすべて強制する。
Cargo/compliance境界は回帰fixture内のmockであり、モデルやCargoは実行していない。

controller自体はdestroyの成功テキストをJSONとして解析して失敗し、非ゼロ終了した。
このため自動lifecycle全体をgreenとは記録せず、ローカルのcleanup-unconfirmed receiptも
成功receiptへ書き換えていない。remote native出力本文・digestは最終receiptとして
回収できておらず、回収済みとは主張しない。上記はremote shell exit 0と、その固定した
検査script・source契約に基づく静的回帰結果である。
使用した一時controllerのSHA-256は
`2ee5d02e0484d42d38fd49c610585041f4e3d000864133aad0cf68d5303d9e4c`。

rootが削除要求後の `57317485` を独立に個別照会し、`instances: null`を確認した。
どちらも今回作成したinstanceと保存データをdestroy済み。再試行の追加作成はしない。
全ページのinstance再照会も `success: true` / `total_instances: 0` / 空配列だった。
これはその時点のinstance inventoryであり、別建てvolumeの全件監査ではない。
両試行はそれぞれUSD 0.20/hour以下・30分以内で終了した。既存guardは不変。
実compliance scannerのローカル回帰も8/8成功した。

この限定した遠隔検証と手動cleanup照合を根拠に、対応するcode pushだけに既存の
`VOKRA_SKIP_HOOKS=1`運用を適用する。PR上のfresh exact-head CI、保護された通常マージ、
Whisperのlicense/provenance問題は、依然として別の受入項目である。

### PRのリンク検査修正

[PR #289](https://github.com/ayutaz/vokra/pull/289) の初回head `a2d73778`では、
必須`documentation-links`が失敗した。reportは5件のerrorを記録し、内訳は
法務文書の3種類の公式URL（US Codeの同一URLを2箇所、California条文、Congress議案）と、
本書のartifact直接UIリンクである。他のチェック成功でこの失敗を免除しない。

artifactは認証済みAPIで存在・未失効・元のrun/sourceの一致を再確認した。
直接UI/downloadへの認証要件を明示し、読者向けのリンクを公開workflow runへ修正した。
artifact ID、元のreceiptのsize/SHA、元の実行結果は変更しない。
法務文書の修正も公式資料への案内と取得境界を扱い、法的結論・過去の確認日・owner判断や
CIの除外設定を変更しない。修正後HEADのCI受入は、初回失敗と別に確認する。
