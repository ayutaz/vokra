# Good first tasks

[English](good-first-tasks.md) | **日本語**

Vokra への最初のコントリビューションに向いた、独立して完結するタスクの一覧
です。各項目には file:line アンカーまたは再現コマンド、自分で確認できる受け
入れ条件、おおよその規模を付けています。着手前に「自分の時間を使う価値がある
か」を判断できるようにするためです。

**最終確認日: 2026-07-20。**

## この一覧の使い方

- **担当宣言の儀式はありません。** 手元にものができたら PR を出してください。
  規模の大きい項目で作業の重複を避けたい場合は、先に *Question* issue を
  立ててください。
- **これらは GitHub Issue で管理していません。** 作業項目は作者のチケット
  ツリーにあり、参照すべき issue 番号は存在しません
  （[CONTRIBUTING.md](../CONTRIBUTING.md) §1）。
- **緑のビルドから始めてください。** 何かを変更する前に
  `cargo test --workspace` が通ることを確認してください。「正常な状態」の
  実測時間は CONTRIBUTING に記載しています。
- **規模** は目安です: **XS** = 1 時間未満、**S** = 1〜2 時間、
  **M** = 半日、**L** = それ以上。

## ここに *載せない* もの

この一覧は意図的に短く保っています。数を揃えるためではなく、本当に独立して
完結するタスクが存在するときだけ追加します。次の 2 種類は方針として除外して
いるため、探しても見つかりません:

- **数値 parity の許容値に触れるもの。** 許容値は導出された architectural
  bound であってつまみではありません。これを変更する作業が最初のタスクに
  なることはありません。
- **コントリビュータ自身が検証できないハードウェアを要するもの。** 例えば
  `crates/vokra-backend-cuda/src/sys.rs:284` の CUDA ドライバ版数の TODO は
  実 NVIDIA GPU がないと確認できないため、「終わらせられないタスク」に
  ならないよう作者側に残しています。

同じ理由で、zero-dependency 不変条件・C ABI 面・GPU kernel も最初のタスクの
対象外です。

---

## 1. `compute.rs` の古いドキュメント参照を直す — XS

**場所**: `crates/vokra-models/src/compute.rs:2`

`Compute` seam のモジュールドキュメントが、リポジトリに存在しない設計メモを
参照しています:

```rust
//! Imperative compute dispatcher for the native models (Phase 3 of the GPU
//! execution architecture; see `scratchpad/graph-engine-plan.md` §3).
```

`scratchpad/` は tracked ではなく（`git ls-files scratchpad/` は何も返しま
せん）、この参照を辿った人は何も見つけられません。参照先の設計は現在
[architecture.ja.md](architecture.ja.md) §2 で公開されています。

**やること**: dangling な参照を、公開されている記述へのポインタに置き換え
ます。ツリー内で該当する参照はこの 1 箇所だけです
（`grep -rn "scratchpad/" crates/ --include="*.rs"` はちょうど 1 行を返します）。

**受け入れ条件**

- `grep -rn "scratchpad/" crates/ --include="*.rs"` が何も返さない
- `cargo doc -p vokra-models` が引き続きビルドできる
- `cargo fmt --all -- --check` と `cargo clippy --all-targets -- -D warnings` が通る

**なぜやる価値があるか**: 2 行の変更で、モデル層で最もアーキテクチャ上重要な
モジュールを、非公開の計画ツリーを持たない人にも読めるようにできます。
`architecture.md` が存在する理由そのものです。

---

## 2. `--help` を持たない check スクリプトに追加する — S

**場所**: `scripts/check-*.sh`

このリポジトリには「check スクリプトは `--help` に対応し、目的・モード・
終了コードを表示する」という確立した慣習があります。`check-zero-deps.sh` /
`check-forbidden-symbols.sh` ほか 11 本がまだ従っていません。

一覧の再現:

```bash
for s in scripts/check-*.sh; do grep -q -- '--help' "$s" || echo "$s"; done
```

執筆時点で **13 本**が出力されます。

**やること**: `scripts/check-platform-support.sh` と
`scripts/check-doc-references.sh` が既に使っている形（ヘッダのコメント
ブロックを再表示する `usage()` と、末尾のフラグ分岐）に揃えます。
**各 check の挙動と終了コードは一切変えないでください** — 目的は発見しやすさ
だけです。

2〜3 本だけの PR でも十分です。13 本すべてを対象にする必要はありません。

**受け入れ条件**

- 変更した各スクリプトが `--help` で有用なテキストを表示し、0 で終了する
- 引数なしの実行が以前とまったく同じ挙動・同じ終了コードになる
  （`git stash` で突き合わせて確認）
- 未知のフラグは、黙って check を実行せず非ゼロで終了する

---

## 3. `README-swift-package.md` を日本語化する — S

**場所**: `README-swift-package.md`（約 1.8 KB）→ 新規 `README-swift-package.ja.md`

Vokra は利用者向け文書の英語版・日本語版を維持していますが、Swift Package の
README は英語のみです。

**やること**: 日本語版を追加し、既存の慣習に従って相互リンクします
（[getting-started.md](getting-started.md) と
[getting-started.ja.md](getting-started.ja.md) の 3 行目を参照）。

**受け入れ条件**

- `README-swift-package.ja.md` が存在し、英語版へリンクしている
- 英語版から日本語版へのリンクがある
- 2 ファイルの見出し構成が一致している

**補足**: 翻訳タスクの中では最小で、より大きな項目に取り掛かる前に PR の流れ
に慣れるのに向いています。

---

## 4. Python binding の README を日本語化する — M

**場所**: `bindings/python/README.md`（約 5.9 KB）→ 新規 `bindings/python/README.ja.md`

タスク 3 と同じ慣習です。分量が多く、コード例を含みます。

**やること**: 散文を翻訳し、**コードブロック・API 名・コマンドラインはその
まま残してください**。説明を翻訳したコマンドについては、その説明どおりの動作
をするか確認してください。

**受け入れ条件**

- 日本語版が存在し、双方向にリンクしている
- コードブロックが英語版とバイト単位で同一
- 2 ファイルの見出し構成が一致している

---

## 5. Unity binding の README を日本語化する — M

**場所**:

- `bindings/unity/com.vokra.unity/README.md`（約 4.4 KB）
- `bindings/unity/com.vokra.unity/Samples~/VadAsrTts/README.md`（約 4.1 KB）

関連する 2 ファイルです。1 つの PR にまとめるのが自然ですが、片方だけでも
構いません。

**受け入れ条件**: 翻訳した各ファイルについてタスク 4 と同じ。

---

## 6. `CONTRIBUTING.md` を日本語化する — L

**場所**: `CONTRIBUTING.md`（約 11 KB）→ 新規 `CONTRIBUTING.ja.md`

翻訳の空白として最大のもので、**意図的に**作者ではなくコントリビュータ向けに
残しています。コントリビューションガイドの翻訳は「意味の通らない箇所」を
見つける実際的な手段であり、それを報告してもらうことは翻訳そのものと同じくらい
価値があります。

**着手前に読んでください**: 英語版は 2026-07-20 に大きく書き換わりました
（§1 / §2 / §6 とクイックスタート節の新設）。古いコピーではなく現在の `main`
を起点にしてください。

**やること**: 翻訳して相互リンクを張り、**不明瞭・誤りに見える箇所は翻訳で
均さずに Question issue を立ててください**。

**受け入れ条件**

- `CONTRIBUTING.ja.md` が存在し、双方向にリンクしている
- 2 ファイルの見出し構成が一致している
- 要件 ID・コマンドライン・ファイルパスはそのまま残す
- リンクが解決する —
  `bash scripts/check-community-docs.sh` が新たなリンク切れを報告しない

---

## この一覧の維持

- **担当**: 作者。四半期ごとの Go/No-go レビューで見直します
  （[governance/quarterly-reviews/](governance/quarterly-reviews/) 参照）。
- **項目が完了したら**: 完了させた PR と同じ PR で項目を削除し、上部の
  *最終確認日* を更新します。完了項目は履歴として残しません — 履歴は git log
  であり、消化済み項目が並ぶとページが使いにくくなるためです。
- **項目を追加するとき**: [CONTRIBUTING.md](../CONTRIBUTING.md) の全基準に
  加えて、file:line アンカーまたは再現コマンド、明示的な受け入れ条件、規模を
  満たすこと。受け入れ条件を書けない項目は、まだ掲載できる状態にありません。
- **この一覧が空でも**それは正常な状態で、独立して完結するタスクが今は存在
  しないことを意味します。埋めるべき欠落ではありません。
- 英語版 [good-first-tasks.md](good-first-tasks.md) は同じ PR で更新して
  ください。乖離すると `scripts/check-community-docs.sh` が失敗します。
