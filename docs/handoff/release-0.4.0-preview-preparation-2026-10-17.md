# 次版preview準備：工程1〜3 — 2026-10-17

## 範囲と非対象

ユーザー依頼は「次版の対象固定」「バージョン・変更履歴整理」「代表機能の再検証」
まで。speech.cpp等との差別化、特に告知で何を言うかも対象とする。
GitHub Release公開、tag、外部registry、HF upload、X投稿は実行しない。
保護対象の承認manifest・fixture tree、旧one-shot許可、owner署名を変更・流用しない。

開始baselineはclean `main` `d15ce4cba388d687ead5b2ed3829427f77f1a0c6`。
作業branchは`codex/release-040-preview-preparation-20261017`。
既存v0.3.0から106 commitsあるが、commit数を機能の完成数とは扱わない。
v0.3.0は2026-09-20に公開済みの唯一のReleaseで、18 assets。
単体desktop CLI release assetはまだなく、release toolingの新実装とは区別する。

## 工程1：次版対象の固定

| 区分 | 判断 |
| --- | --- |
| ソースの取り込み範囲 | baseline mainに通常マージ済みの変更と、このpreviewの検証・文書整備。未マージPRを完成機能として含めない |
| 告知に使う代表経路 | Whisper baseの実音声ASR、Silero VAD v5のspeech/silence。固定モデルidentityと実結果が揃うまで未検証扱い |
| 初期環境 | cloud／承認済みCIのLinux CPU。ローカルMacではモデル取得・実行なし |
| 将来の配布 | GitHub Release中心。新しいmodel weightや外部package registryの公開は別条件 |
| 独立した未完了事項 | 残モデルのnon-Apple/owner、独立実weight parity、最終packet、Scaleway Apple CPU/Metal/no-fallback。previewで完了扱いにしない |

最新PR readbackではopenは6件（#258/#198/#191/#182/#169/#152）。
#259は2026-10-17 06:14:21 UTCにClosed、merge commitなし、head `481ab7e2`。
理由を推測せず、追加observerの実compile/API、full PCM等が完了したとも扱わない。
以前の「7 open／#259 Draft」記録はその観測時点の履歴として保持する。
#258のCanary静的レビュー確認は別の未解決scopeで、この依頼から暗黙に許可しない。
必要なsecurity修正とpreview対象のsourceを区別し、既知の危険な実行環境は使わない。

## 工程2：バージョンと変更履歴

**次版targetは`v0.4.0`、最初の候補は`v0.4.0-rc.1`とする。まだ公開版ではない。**
v0.3.1ではなく次のminorを選ぶ理由は、Mimi compute seamやKyutai streaming-LM等の
public Rust API／機能追加が含まれるため。0.xの開発版として、API安定やABI freezeを
宣言しない。

read-only差分確認ではroot `Cargo.toml`／`Cargo.lock`、C header、
三crateのRust public-API snapshotはv0.3.0から変化していない。
一方、`vokra-models`側には新しいpublic APIがある。snapshotの対象は
core/ops/capiだけなので、「snapshot差分なし＝全Rust API互換」とは結論しない。
全体のsemver互換性解析やworkspace Cargoをローカルで実行したわけではない。

`CHANGELOG.md`の`Unreleased`に実際のmerged changeをAdded／Fixed／Security／
Documentationへ整理する。過去の0.3.0節は書き換えず、新版の公開日も発明しない。
現在のworkspace versionは0.3.0のまま。manifest／path-dependency／binding versionの
同期とchangelog rollは、次の配布工程で候補を確定するときの一つの変更として行う。
タグだけを先行させない。release workflowのtag/workspace一致gateは保持する。
さらに、`release.yml`のGitHub Release作成commandは現状`--prerelease`を明示していない。
RCというtag名だけで公開先のpre-release／Latest扱いまで保証したことにせず、
配布工程で明示設定と結果を検証する。今ここでeffectful releaseを試さない。

この整理後、changelog integrity check、changelog oracle 13件、version-contract
oracle 17件、candidate dispatch versionの解決（`0.4.0-rc.1`）は成功した。
これはタグ発行・version bump・配布物の実ビルド結果ではない。

## 工程3：代表経路の受入条件

既存`parity-whisper-real` workflowはTorch 2.4系列を指定しているため、
新しい検証のため無変更起動しない。Torch/reference環境を使わない限定した
CLI functional smokeを準備し、独立数値parityの代替とは扱わない。

| 項目 | 実行前に固定するもの | 合格に必要な証拠 |
| --- | --- | --- |
| 共通 | reviewed source HEAD、fixed repo revision／GGUF filename／size／SHA-256、CPU-only build、入力audio hash | 実際に回収したmodel/input hash、native CLI argv/exit、bounded stdout/stderr、skipなし |
| ASR | Whisper base、committed JFK実音声 | 実際のtranscriptと既知のJFK keyphrase。空文字・token IDだけ・synthetic入力で成功にしない |
| VAD | Silero VAD v5、同じJFK実音声と同じdurationのsilence | 有限なframe/probability summary、speech入力のspeech frame、silenceとの区別。精度benchmarkや独立reference一致とは呼ばない |
| 後処理 | small receipt／textだけ回収、model artifactは回収・公開しない | remote jobの終了と取得物を確認。VASTを使用した場合は今回のworker/storageのdestroy readbackも必要 |

audioは`tests/fixtures/audio/jfk-30s.wav`。
既存sidecarのSHA-256は
`58adb4ea501d955fcd40bfbb69128f8f40428b81d8716b9ed337949773be253f`。
名前は30秒だが、実clipは約11秒のPublic Domain JFK音声で、Whisper処理windowと
実audio durationを混同しない。

### 現在の実施状態

- **2026-10-17継続時の許可:** ユーザーは、二件の公開JSON参照、mockしたoffline回帰、
  準備PRの作成・CI後マージ、GitHub-hosted Linuxの実CPU検証一回・最大30分を
  明示許可した。以下の「確認中」は許可前の経緯であり、この許可を取り消さない。
  有料worker、localモデル、tag／Release／HF upload／X投稿は引き続き対象外。
- 明示許可後もlocal testは通常／正式なescalationの両方でhookに拒否された。
  公開JSONもweb取得はtool非対応、承認理由付きcurlはlocal model-download guardに
  拒否された。設定・hook・command wrapperを変えて回避していない。
- そのためmodel-freeな準備PRのhosted CIでoffline回帰と固定revisionの二件のAPI JSON
  検査を行う方針に変更した。これはlocal禁止の回避ではなくremoteへの移管であり、
  GGUF取得・Cargo・実モデル実行を準備CIに含めない。実モデルjobはmain登録後に一回。
- 通常pre-commitの既存fixture EOL gateが保護対象Silero fixture treeのmanifestを
  読むことをsource inspectionで確認した。hookをskip／変更せず、その読み取りだけを
  個別確認し、ユーザーは既存checkによる読み取りを明示許可した。通常hookを保持して
  commitする。CosyVoice2 owner manifestやfixtureを手作業で読まない。
- 固定model revisionの既存metadata記録はあるが、実行用のfilename／size／content SHAは
  この準備段階で未確定。推測で埋めない。
- 2モデルの公開JSON API metadata参照は安全hookに拒否された。要求はJSONのみだったが、
  同じ取得を別経路で再試行せず、read-only metadataのexact scopeをユーザーへ確認中。
- この段階ではmodel/config/tokenizer/weightを取得・実行していない。
  remote実行・受入証跡が揃うまで工程3は**未完了**。
- 新しいsmoke runner／workflowはLuna担当で準備し、manager reviewとoffline回帰を先行。
  未配置のpinsやmockedテストを実モデルの成功に読み替えない。

### 継続レビューと実行境界

managerはrunner・test・workflowの初版を全行レビューした。CLI出力の上限が
取得後の判定だった点、JSONの重複key／サイズ制限、実行前pin検査、生成silenceの
identity、失敗時receiptを確かめる回帰の不足をLunaへ差し戻した。
初版の7件成功という実装担当の報告は、修正版の受理結果ではない。

managerによる次のoffline回帰の実行は安全hookに拒否された。

```sh
env UV_CACHE_DIR=/private/tmp/vokra-uv-check UV_OFFLINE=1 \
  uv run --no-project --offline --python 3.12 python \
  tools/parity/test_release_candidate_cpu_smoke.py
```

実モデルを使わないtestではあるが、別command／agent／toolで再試行せず、
このtestの個別実行範囲を確認中。release workflow内のread-only文字列検索も
heavy-Cargo guardに拒否されたため再試行していない。これらの拒否を
回帰成功・配布gate成功と扱わない。hook・例外・安全設定は変更していない。

工程3の実行に向けて、今回の変更の専用branchへのcommit／pushと、固定した
公開GGUF二件（合計2GB未満）のGitHub-hosted Linux CPU検証一回・最大30分も
個別確認中。これは実行済みのjobや、新しい公開権限ではない。有料worker、
localモデル、tag、Release、HF upload、X投稿は含めない。

この継続で再実行した`check-zero-deps.sh`と変更済み管理文書の
`git diff --check`は成功。比較資料には固定した五つの一次資料、Vokra側の
設計上の選択、現時点の弱点、告知案と禁止すべき未検証の主張を分けて記録した。
代表実モデルの成功と、修正版runnerの受理は引き続き未完了。

新しいmanual-only workflowは、branchへのpushだけで初回dispatchできるとは
想定しない。[GitHubの公式手順](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/manually-run-a-workflow)
ではdefault branch上のworkflowを前提としている。実行許可が得られた後も、
まずsource準備PRのレビュー・通常CI・main登録を確かめ、実モデルの手動実行は
別に一回起動する。`pull_request`／`push`等を追加してモデル実行を自動発火させる
迂回は行わない。source準備PRの受理だけで工程3を完了とはしない。

追加レビューでは、receipt用の16KiB切り詰めをsemantic判定にも使っていた点と、
同じ時計fixtureによりtimeoutが先に成立してoverflow／read-errorを通らないtestを
修正対象とした。修正版はsemantic判定に64KiB上限内の全stdoutを使い、
production captureの各失敗原因をmockで分ける。fake processの異常系から
実processへsignalを送らないよう、testはterminatorもmockする。

rootが実行した`actionlint`とworkflow hygieneは成功（57 workflows／36 crons）。
後者は最初に通常のUV cacheのsandbox権限で失敗したが、許可済みtemporary cacheを
指定した実行で成功した。changelog 13件／version-contract 17件と文書参照も再確認済み。
これらはrunnerのPython回帰や実モデルsmokeの実行許可・成功を代替しない。
download deadlineはread間で確認するもので、通信単独の厳密なwall-clock上限は
主張しない。実モデルworkflow全体には30分の上限がある。

許可後のsource準備は`release-candidate-source-tests.yml`へ分離した。
exact sparse checkoutで保護treeを除外し、Python 3.12の用意後にmocked回帰を実行、
固定した二件のAPI JSONだけを取得する。ModelInfoのLFS欄は公式clientと照合した
`sha256`を使い、tree APIの`oid`と混同しない。redirect・不正なJSON・revisionや
sizeの不一致・曖昧なGGUF選択を拒否し、小さいpins／source identityだけを回収する。
この準備CIにはCargo・resolver／weight取得・model実行を含めない。
rootは追加したmetadata helper／test／source workflowも全行レビューし、二workflowの
actionlint、workflow hygiene（58 workflows／36 crons）、zero-depsとdiff checkは成功。
ただしsource準備PRはまだ作成しておらず、Python回帰・実API参照・実モデルは未実行。

## 差別化と告知

[競合比較と告知方針](competitive-positioning-2026-10-17.md)でspeech.cpp、whisper.cpp、
sherpa-onnx、Candle、NeMo-Speech.cppを一次資料に基づき比較した。
Rust-native／デフォルトruntimeの外部Cargo依存ゼロ／音声処理・状態管理の境界を
軸にする。Python不要・GGUF・C API・Metal・音声全部入りだけを独自性にしない。

代表経路の実結果と導入手順が確認できるまでは、告知は「開発中／preview準備中」。
速度優位、全136モデル動作、Metal完全対応、production-readyは主張しない。
今回の工程3が成功しても、競合benchmark、モデルの全品質検証や次のrelease公開の
完了ではない。
