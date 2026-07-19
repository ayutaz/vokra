# 要件 ID 用語集

[English](requirement-ids.md) | **日本語**

Vokra の公開文書（`README.md` / `CONTRIBUTING.md` および `docs/` 配下）は、
`FR-EX-08` や `NFR-DS-02` といった要件 ID を、確定した設計方針の略号として
引用しています。**それらの ID を定義している文書は公開していません。**
要求定義・要件定義・成果物定義・マイルストーン計画は作者が非公開で管理して
いるため、公開リポジトリだけを見ている読者には引用された ID を解決する手段が
ありません。

本ページはその欠落を埋めます。公開文書に出現するすべての要件 ID について、
**「その項目が何を規定しているか」を一行で**示します。コードコメント・PR の
説明・レビュー指摘を、推測せずに読むにはこれで足ります。

## 本ページでないもの

- **要件定義書ではありません。** 各項は要件の *主題* を示すだけで、条文・
  受け入れ基準・数値閾値・対象リリースは含みません。それらは非公開の計画
  文書にあります。
- **存在するすべての ID の索引ではありません。** 公開文書に引用されている
  ID のみを対象とします。内部の計画ではより多くの ID を使っています。
- **連番ではありません。** family 内の番号はしばしば飛びます（`FR-MD` は
  02 → 09 → 10 → 13）。これは想定どおりで、飛んだ番号は「存在するが公開文書
  では引用されない ID」であり、**記載漏れではありません**。

## 本ページの維持

**最終確認日: 2026-07-20 — 公開文書に出現する ID は 91 種類。**

本ページが網羅すべき集合は手作業ではなく機械的に導出します。再生成コマンド:

```bash
git ls-files 'README.md' 'README.ja.md' 'CONTRIBUTING.md' 'docs/*.md' 'docs/**/*.md' \
  | xargs grep -IohE '\b(BR|FR|NFR|IF)-([A-Z]{2}-)?[0-9]+' \
  | sort -u
```

このコマンドの 2 点は load-bearing です:

- 中段の `[A-Z]{2}-` は **optional**。2 セグメント形の ID（`BR-02`、`IF-01`
  など）が実在し、中段を必須にした正規表現はそのうち 5 件を黙って取りこぼし
  ます。
- 母集合は `git ls-files` = **tracked ファイルのみ**で、README 2 本 /
  `CONTRIBUTING.md` / `docs/` を対象とします。untracked / ignore された作業
  ファイルは意図的に対象外です。

**更新責任**: 公開文書に要件 ID を新しく書いた人が、**同じ PR で**該当行を
追加します。定期レビュー任せにはしません —
`scripts/check-doc-references.sh` が引用集合と下記の掲載集合を**双方向で**
比較するため、未掲載の ID も、引用が消えた古い行も、どちらも検査で落ちます。
CI では advisory として実行され、ローカルでも実行できます:

```bash
bash scripts/check-doc-references.sh          # 検証
bash scripts/check-doc-references.sh --list   # 解決された集合を表示
```

英語版 [requirement-ids.md](requirement-ids.md) と掲載 ID 集合が一致している
ことも、同じスクリプトが検査します。

## Family 一覧

| Prefix | 領域 |
|---|---|
| `BR` | ビジネス要求 — プロジェクトの存在理由 |
| `FR-LD` | モデルロード（GGUF / safetensors / metadata） |
| `FR-EX` | 実行エンジンと IR |
| `FR-OP` | 音声オペレータ（vocoder / codec / decode / 前処理 / 話者） |
| `FR-BE` | バックエンド（CPU / GPU / NPU / ビルド SKU） |
| `FR-MD` | モデル対応とモデル追加のプロセス |
| `FR-QT` | 量子化 policy と検証 |
| `FR-SV` | サーバ / API 互換 |
| `FR-ST` | ストリーミング挙動 |
| `FR-CP` | コンプライアンス（watermark / provenance / research flag） |
| `FR-TL` | ツール（converter / CLI / eval / build script） |
| `NFR-DS` | 配布とサイズ |
| `NFR-PF` | 性能 |
| `NFR-QL` | 数値品質・音声品質 |
| `NFR-RL` | プラットフォーム制約 |
| `NFR-MT` | 保守・CI/CD・コミュニティ運営 |
| `NFR-LC` | 依存ライセンス |
| `NFR-LG` | 法務・規制 |
| `NFR-PT` | プラットフォーム網羅 |
| `IF` | 外部インタフェース |

---

## BR

ビジネス要求。公開文書に引用される 2 件のみを掲載します。

| ID | 何を規定するか |
|---|---|
| `BR-02` | **全プラットフォーム対応を前提とする**マルチプラットフォーム安定性。Windows / macOS / Linux / Android / iOS / Web (WASM) とサーバを、単一 API・単一バイナリで扱う。特定プラットフォームの切り捨ては行わず、ロードマップの段階化は GPU/NPU 高速化の順序であってプラットフォーム対応の選別ではない。 |
| `BR-04` | Unity / Godot から使えること。C ABI、単一バイナリ、Apache-2.0（GPL 回避）、IL2CPP / GDExtension 対応。 |

## FR-LD

モデルロード。

| ID | 何を規定するか |
|---|---|
| `FR-LD-01` | GGUF を直接ロードすること。`mmap` による zero-copy weight ロードで、cold start が全読み込みのコストを払わないようにする。 |
| `FR-LD-02` | 音声固有の metadata を Vokra 独自の `vokra.*` prefix 付き GGUF chunk として読み書きすること。llama.cpp 本体のキーと衝突しない命名にする。 |
| `FR-LD-03` | 特徴量フロントエンドの記述（`vokra.frontend.*`）を必須 chunk として扱い、runtime が検査すること。mel フロントエンドを再導出せず再現可能にする。 |
| `FR-LD-04` | safetensors を直接ロードすること。上流 checkpoint のみを使い、pickle は使わない。 |
| `FR-LD-05` | **恒久制約。** ONNX はオフライン変換ツールでのみ扱い、runtime には ONNX ローダーも protobuf / abseil / onnx 依存も含めない。 |
| `FR-LD-06` | Silero VAD を汎用オペレータに書き換えず、1:1 保存の専用サブグラフとしてロードすること。 |
| `FR-LD-07` | 量子化 weight（K-quant 系）を GGUF から直接ロードすること。 |

## FR-EX

実行エンジンと IR。

| ID | 何を規定するか |
|---|---|
| `FR-EX-01` | 外部のグラフ形式を採らず、Vokra 独自の IR（audio graph descriptor）を持つこと。 |
| `FR-EX-08` | **恒久制約。** 全バックエンドで同一の op coverage を保証する。バックエンドが実行できない op は**明示エラー**であり、silent CPU fallback をデフォルトにしない。リポジトリ中で最も多く引用される ID。 |
| `FR-EX-10` | sampler / beam search / CFG をモデルグラフに埋め込まず runtime 関数として提供すること。デコード設定の変更に再変換を要さない。 |

## FR-OP

音声オペレータ。汎用テンソル op の合成ではなく native 実装する。

| ID | 何を規定するか |
|---|---|
| `FR-OP-10` | `hifigan_generator` vocoder op、および低精度化を許容する条件。 |
| `FR-OP-11` | `bigvgan_generator` op。reference 実装が商用利用可能なライセンスでないため、論文からの再実装とする。 |
| `FR-OP-12` | `vocos_head` op。iSTFTNet head と区別し、config で引き下げられない最低精度を持つ。 |
| `FR-OP-13` | `snake_activation` op と、その内部精度 attribute。 |
| `FR-OP-31` | FSQ 系 codec op（`wavtokenizer_vq` / `xcodec2_fsq`）。コスト特性が異なるため RVQ とは別サブグラフとして実装する。 |
| `FR-OP-32` | EnCodec の扱い。エンジンは op に対応するが、weight は公式 model zoo から除外する。 |
| `FR-OP-40` | `beam_search` op — beam 幅、length normalization、early stopping、n-best 出力、word-level timestamps。ホスト側関数として提供する。 |
| `FR-OP-41` | `ctc_decode` op。言語モデル融合と hotword boost を含む。 |
| `FR-OP-42` | `rnnt_decode` op と、選択可能なデコード戦略。 |
| `FR-OP-60` | `aec`（音響エコーキャンセル）op と、それが必要とする runtime 管理・時間タグ付きの参照信号 queue。full-duplex な S2S の前提条件。 |
| `FR-OP-61` | `denoise`（音声強調）op。 |
| `FR-OP-62` | 収音側パイプラインの `agc` / `hpf` op。 |
| `FR-OP-63` | `loudness_norm` op（LUFS / EBU R128）。 |
| `FR-OP-80` | `speaker_encode` op — 複数の話者埋め込みアーキテクチャを 1 つの API で扱う。zero-shot TTS が依存するため core に残す。 |
| `FR-OP-81` | `speaker_verify` op（類似度による話者照合）。 |
| `FR-OP-82` | `diarize` op。optional feature flag の背後に置く。 |
| `FR-OP-93` | 評価メトリクス（mel loss / UTMOS / DNSMOS / WER / CER）を runtime に内蔵し、量子化の検証を自動化できるようにする。 |

## FR-BE

バックエンド。公開文書に引用される ID のみ掲載します（各バックエンドは個別の
ID を持ちます）。

| ID | 何を規定するか |
|---|---|
| `FR-BE-01` | CPU バックエンドを第一級バックエンドとすること。x86-64 / ARM64 / RISC-V / WASM にまたがる runtime ISA dispatch の階層を持つ。 |
| `FR-BE-05` | WebGPU バックエンド。binding crate ではなく手書きの extern-import shim として実装し、zero-dependency 不変条件を保つ。 |
| `FR-BE-09` | critical-safe ビルド SKU。ベンダー GPU/NPU 経路をコンパイル時に除外し、その結果を SBOM に明示する。 |

## FR-MD

モデル対応。

| ID | 何を規定するか |
|---|---|
| `FR-MD-02` | Whisper base（ASR）の native 再実装 — encoder / decoder / beam search。 |
| `FR-MD-09` | Moshi（full-duplex S2S）。weight のライセンス上、attribution 表示機能を要する。 |
| `FR-MD-10` | F5-TTS / Fish-Speech。エンジン対応のみとし、非商用 weight は research flag で分離する。 |
| `FR-MD-13` | **恒久プロセス。** モデル対応を追加する PR は、同じ PR でライセンス監査を更新し legal-compliance チェックリストを通すこと。[CONTRIBUTING.md](../CONTRIBUTING.md) §4 参照。 |

## FR-QT

量子化。

| ID | 何を規定するか |
|---|---|
| `FR-QT-02` | 層別量子化 policy を hard-code せず config 駆動にすること。 |
| `FR-QT-03` | 強い量子化に耐えない op に最低精度を強制し、config から引き下げられないようにすること。 |
| `FR-QT-04` | 量子化後の品質検証をパイプラインの標準段階とし、劣化を自動検出すること。 |
| `FR-QT-05` | KV cache の量子化。 |

## FR-SV

サーバと API 互換。

| ID | 何を規定するか |
|---|---|
| `FR-SV-01` | `vokra-server` を独立した単一バイナリとして提供すること。Docker は必須ではなく任意とする。 |
| `FR-SV-02` | OpenAI 互換の音声書き起こしエンドポイント。既存クライアントが無改造で動く。 |
| `FR-SV-04` | piper-plus 互換の HTTP TTS エンドポイント。Home Assistant / Rhasspy エコシステム向け。 |
| `FR-SV-05` | Wyoming Protocol サーバ実装。 |
| `FR-SV-06` | paged KV cache による複数同時セッションの処理。 |

## FR-ST

ストリーミング。

| ID | 何を規定するか |
|---|---|
| `FR-ST-03` | barge-in。生成中の処理を中断し、バッファ済み音声を即座に flush できること。 |

## FR-CP

コンプライアンス。

| ID | 何を規定するか |
|---|---|
| `FR-CP-01` | TTS / VC 出力へのデフォルト watermark 付与と、opt-out を明示的にすること。現況は [docs/legal-compliance.md](legal-compliance.md) を参照。 |
| `FR-CP-02` | C2PA manifest の付与と検証。 |
| `FR-CP-03` | 非商用ライセンスの weight を明示的な research flag 経由でのみロード可能とし、デフォルト経路から外すこと。 |
| `FR-CP-05` | モデルの provenance・ライセンス・フロントエンド記述を GGUF metadata として公開し、下流が検査できるようにすること。 |
| `FR-CP-06` | コンプライアンス設定 API。 |

## FR-TL

ツール。

| ID | 何を規定するか |
|---|---|
| `FR-TL-01` | checkpoint → GGUF のオフライン変換ツール。ONNX の取り扱いが許される唯一の場所（`FR-LD-05` 参照）。 |
| `FR-TL-02` | `vokra-cli`: `run` / `convert` / `bench`。 |
| `FR-TL-03` | `vokra-eval`: `FR-OP-93` の品質メトリクスを 1 コマンドで実行する。 |
| `FR-TL-04` | C ヘッダおよびエンジン / binding パッケージを生成するビルドスクリプト群。 |
| `FR-TL-05` | **廃止済。** 競合 changelog の自動監視ワークフローだったが、依頼者決定で廃止し、`NFR-MT-05` の手動四半期レビューに全面的に読み替えた。廃止の事実が今も参照されるため掲載している。 |

## NFR-DS

配布とサイズ。

| ID | 何を規定するか |
|---|---|
| `NFR-DS-01` | core runtime バイナリのサイズ予算（モバイル向けはより厳しい）。 |
| `NFR-DS-02` | **zero-dependency 不変条件。** runtime は protobuf / abseil / onnx を持たず、実際には third-party crate を一切持たない。解決後の root `Cargo.lock` は first-party の `vokra-*` crate のみを含み、静的リンクで単一ファイル配布ができる。ローカルと CI の `scripts/check-zero-deps.sh` が強制する。これを壊さずに機能を足す 2 つの正規手段は [CONTRIBUTING.md](../CONTRIBUTING.md) §7 と [architecture.ja.md](architecture.ja.md) に記載。 |
| `NFR-DS-03` | Vokra を配布するパッケージチャネル。 |
| `NFR-DS-04` | モデルをバイナリと分離して配布すること — metadata を内蔵した単一 GGUF ファイル。 |

## NFR-PF

性能。各項は「何の性能を縛るか」を示します。数値目標そのものは非公開の要件
定義側にあります。

| ID | 何を規定するか |
|---|---|
| `NFR-PF-03` | iOS 上の Whisper base の実時間係数（RTF）目標。 |
| `NFR-PF-04` | CUDA 上の Whisper large-v3 の RTF 目標。 |
| `NFR-PF-05` | サーバ側 TTS のレイテンシ目標。 |
| `NFR-PF-06` | Android 上の Whisper base の RTF 目標。 |
| `NFR-PF-08` | Web（WASM / WebGPU）ターゲットの動作目標。 |
| `NFR-PF-11` | cold start。`mmap` ベースのロードでモデルロード時間をほぼゼロに保つ。本プロジェクトが回避しようとした失敗モードそのもの。 |
| `NFR-PF-13` | 性能 regression ゲート。PR ごとに RTF / TTFA / レイテンシを計測し、regression は黙って merge せず justify を要する。 |

## NFR-QL

数値品質・音声品質。

| ID | 何を規定するか |
|---|---|
| `NFR-QL-01` | PyTorch reference との数値 parity を PR ごとに CI 検証すること。モデル別の許容値はグローバル定数ではなく [`tests/parity/`](../tests/parity/) に明示する — 許容値は architectural bound であり、CI が赤いときに緩めるつまみではない。 |
| `NFR-QL-02` | PyTorch reference に対する音声品質の劣化上限。 |
| `NFR-QL-04` | 公開評価データ subset に対する nightly の音声品質 regression 実行。閾値割れは blocking な欠陥として扱う。 |

## NFR-RL

プラットフォーム制約。先に避ければ安く、後から直すと非常に高くつく失敗モード
を明文化したもの。

| ID | 何を規定するか |
|---|---|
| `NFR-RL-03` | iOS が動的ライブラリロードを禁じること。ゆえに静的リンク。 |
| `NFR-RL-04` | Android の `StreamingAssets` jar URL 問題。ゆえに展開ヘルパーを内蔵する。 |
| `NFR-RL-05` | JIT を一切使わないこと（iOS の W^X）。高速化は runtime dispatch のみで行う。 |
| `NFR-RL-06` | GPU バックエンドの非互換を silent fallback ではなく明示エラーとして表面化すること。`FR-EX-08` のバックエンド版。 |
| `NFR-RL-07` | メモリ安全性。core は Rust とし、オペレータ内部の `unsafe` + SIMD intrinsics は許可するが、API 境界は安全に保つ。opt-out できる crate は [architecture.ja.md](architecture.ja.md) に列挙。 |

## NFR-MT

保守・CI/CD・コミュニティ運営。

| ID | 何を規定するか |
|---|---|
| `NFR-MT-01` | エンジン実装 / CI / ドキュメント / パッケージング / リリース工学 / コミュニティ運営への開発時間配分。単一メンテナのプロジェクトで痩せ細りがちな非コード作業に、意図的に大きな比率を割り当てる。 |
| `NFR-MT-02` | CI matrix の階層化。どのプラットフォームを毎 PR / nightly / weekly で回すか。 |
| `NFR-MT-03` | リリースプロセス — release train、semantic versioning、changelog 自動化、reproducible build、SBOM 生成。 |
| `NFR-MT-05` | プロジェクトの撤退条件に対する四半期 Go/No-go レビューをリリースプロセスに組み込むこと。`FR-TL-05` 廃止後、唯一残る監視機構。 |
| `NFR-MT-06` | オープン開発。実装開始と同時にリポジトリを public 化し、Issue / PR / CI 結果 / ベンチマークを公開して品質を外部から検証可能に保つ。 |
| `NFR-MT-07` | CI 品質ゲート。`main` は branch protection + PR 必須とし、ビルド / テスト / フォーマット / lint / 数値 parity / 性能 regression / ライセンス・脆弱性 / ドキュメント内コード例の検査を required check とする。現在の required 集合は [CONTRIBUTING.md](../CONTRIBUTING.md) §2 に記載。 |
| `NFR-MT-08` | リリース物の自動発行。リリース成果物は CI がビルド・発行し、手動ビルドを配布しない。 |

## NFR-LC

依存ライセンス。

| ID | 何を規定するか |
|---|---|
| `NFR-LC-02` | 許可する依存ライセンス（Apache-2.0 / MIT / BSD 系）、GPL・LGPL の禁止、MPL-2.0 の個別評価。 |
| `NFR-LC-04` | CI のライセンス検査により、GPL/LGPL 依存の混入を PR ブロッカーとすること。 |

## NFR-LG

法務・規制。

| ID | 何を規定するか |
|---|---|
| `NFR-LG-01` | EU AI Act Article 50 — AI 生成音声への machine-readable marking を設計要件として保持すること。実装の現況は [docs/legal-compliance.md](legal-compliance.md) にあり、本行から推測しないこと。 |
| `NFR-LG-02` | California SB 942 / Tennessee ELVIS Act / 審議中の NO FAKES Act。voice cloning の分離と consent 管理で対応する。状況の詳細は同じく [docs/legal-compliance.md](legal-compliance.md)。 |

## NFR-PT

プラットフォーム網羅。

| ID | 何を規定するか |
|---|---|
| `NFR-PT-01` | 全プラットフォーム対応を前提とすること。単一プラットフォームでしか成立しない必須依存を導入せず、全ターゲットへのクロスビルド可能性を CI で継続検証する。バックエンドの導入順序は高速化の段階化であり、対応プラットフォームの選別ではない。 |
| `NFR-PT-02` | CPU 対応の広さ。x86-64 と ARM64 で前提とする命令セットの baseline として表現される。 |

## IF

外部インタフェース。

| ID | 何を規定するか |
|---|---|
| `IF-01` | C ABI 利用者向けインタフェース — 単一の `include/vokra.h`、opaque handle、thread-local なエラー状態、および ABI 安定性のコミットメント。Unity / Godot / Swift / Kotlin / Python / JS の binding がすべてこの上に乗るため、本ページで 2 番目に多く引用される ID。 |
| `IF-05` | piper-plus HTTP / Home Assistant 向けインタフェース（`FR-SV-04` / `FR-SV-05` 参照）。 |
| `IF-07` | GGUF エコシステム向けインタフェース。GGUF 準拠に加え、llama.cpp 本体のキーと衝突しない `vokra.*` prefix chunk を持つ。 |

---

## 関連文書

- [architecture.ja.md](architecture.ja.md) — 用語ではなく構造を知りたい読者
  向けの crate マップ / 実行モデル / 設計 red line。
- [CONTRIBUTING.md](../CONTRIBUTING.md) — PR、required check、依存ポリシー、
  レビュー規則としての red line。
