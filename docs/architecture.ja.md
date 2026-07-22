# アーキテクチャ

[English](architecture.md) | **日本語**

Vokra のソースツリーを読む・拡張する人のための見取り図です。各 crate が何の
ためにあるか、モデルが実際にどう実行されるか、そしてどの設計判断が確定済みで
レビューで蒸し返さないかを示します。

ここで引用する要件 ID は [requirement-ids.ja.md](requirement-ids.ja.md) を、
レビュー規則は [CONTRIBUTING.md](../CONTRIBUTING.md) を参照してください。

---

## 1. crate マップ

本リポジトリは Cargo の **virtual workspace** です。member は `crates/*` と
テスト専用 2 crate で、`integrations/` は意図的に除外されています。

<!-- anchor: Cargo.toml -->

### 1.1 runtime グラフ

`crates/` 配下に 14 crate があります。すべてが first-party の `vokra-*`
crate であり、それが zero-dependency 不変条件（`NFR-DS-02`）を成立させて
います。

| Crate | 役割 |
|---|---|
| `vokra-core` <!-- anchor: crates/vokra-core/src/lib.rs --> | IR と実行エンジン。audio graph descriptor（`FR-EX-01`）、グラフ評価器、各バックエンドが実装する `Backend` trait、GGUF ローダー、タスクレベルの engine trait を持つ。**`unsafe` を一切含まない。** |
| `vokra-ops` <!-- anchor: crates/vokra-ops/src/lib.rs --> | 音声オペレータ。STFT / iSTFT / mel filterbank / MFCC / DCT ほか audio dialect（`FR-OP-*`）と、CPU 側 FFT lowering — C ライブラリの binding ではなく pocketfft アルゴリズムの Rust 再実装。 |
| `vokra-backend-cpu` <!-- anchor: crates/vokra-backend-cpu/src/lib.rs --> | 第一級の CPU バックエンド（`FR-BE-01`）。f32 の compute kernel と、実行ホストの CPU に応じて実装を選ぶ単一バイナリ runtime ISA dispatch。 |
| `vokra-backend-metal` <!-- anchor: crates/vokra-backend-metal/src/lib.rs --> | macOS / iOS の GPU バックエンド。**手書きの生 Objective-C runtime + Metal FFI** で実装し、`metal` / `objc2` binding crate は使わない。 |
| `vokra-backend-cuda` <!-- anchor: crates/vokra-backend-cuda/src/lib.rs --> | NVIDIA GPU バックエンド。CUDA Driver API + NVRTC を **実行時に `dlopen` / `LoadLibrary` でロード**する。`cudarc` / `cust` / `rustacuda` は使わず、CUDA ライブラリを同梱もリンクもしない — 利用者のシステムインストールを実行時に検出する（配布物を NVIDIA の再配布条項から切り離すための構成）。 |
| `vokra-backend-vulkan` <!-- anchor: crates/vokra-backend-vulkan/src/lib.rs --> | Android / Linux（および非 NVIDIA デスクトップ）の GPU バックエンド。生 Vulkan FFI を同じく `dlopen` でロードし、SPIR-V シェーダは事前コンパイル済み。`ash` / `vulkano` / `erupt` は使わない。 |
| `vokra-backend-webgpu` <!-- anchor: crates/vokra-backend-webgpu/src/lib.rs --> | ブラウザ向けバックエンド（`FR-BE-05`）。WebGPU API への手書き WASM extern-import shim。`wgpu` / `wasm-bindgen` は使わない — instantiate 時の import object 解決が WASM における `dlopen` 相当。 |
| `vokra-models` <!-- anchor: crates/vokra-models/src/lib.rs --> | native なモデル実装群。whisper.cpp 型で Rust に再実装し、モデルの *定義* をここに置いて上流からは **checkpoint のみ**を取り込む。piper-plus TTS の推論本体もここにある。 |
| `vokra-piper-plus` <!-- anchor: crates/vokra-piper-plus/src/lib.rs --> | piper-plus の **G2P 流用ブリッジ**と voice model 変換補助 — それだけ。推論本体（MB-iSTFT-VITS2）は `vokra-models` に native 実装されており、かつての「piper-plus を wrap する」方針は依頼者決定で廃止された。 |
| `vokra-convert` <!-- anchor: crates/vokra-convert/src/lib.rs --> | オフラインの checkpoint → GGUF 変換器（`FR-TL-01`）。**ONNX / protobuf の取り扱いが存在してよい唯一の場所**（下記 red line R1）。 |
| `vokra-mmap` <!-- anchor: crates/vokra-mmap/src/lib.rs --> | 真の `mmap` による GGUF ロード（`FR-LD-01`、`NFR-PF-11`）。メモリマップに必要な `unsafe` を `vokra-core` の外に隔離するため独立 crate になっている。 |
| `vokra-capi` <!-- anchor: crates/vokra-capi/src/lib.rs --> | C ABI 面（`IF-01`、`BR-04`）。公開成果物は `extern "C"` シンボル群で、生成ヘッダは `include/vokra.h`。Unity / Godot / Swift / Kotlin / Python / JS はすべてこの上に乗る。 |
| `vokra-cli` <!-- anchor: crates/vokra-cli/src/main.rs --> | 総合コマンドラインツール（`FR-TL-02`）。`run` / `convert` / `bench`。**binary crate** であり、`src/lib.rs` を持たない唯一の crate。引数パースは手書き。 |
| `vokra-eval` <!-- anchor: crates/vokra-eval/src/lib.rs --> | 評価メトリクス（`FR-OP-93`、`FR-TL-03`）。mel loss / WER / CER を再利用可能なライブラリ + CLI として提供。 |

さらに 2 つの workspace member はテスト専用です:

- `tests/parity` <!-- anchor: tests/parity --> — 数値 parity ハーネス
  （`NFR-QL-01`）。crate 名は `vokra-parity`。required check の `parity` が
  実行するのはこれ。
- `tests/wasm-harness` <!-- anchor: tests/wasm-harness --> — WASM エントリ
  crate（`vokra-wasm-harness`）。ブラウザ側の `(ptr, len)` ABI を検証する。

### 1.2 依存の向き

dev-dependency を除いた通常依存のみを示します。グラフは非循環で、
`vokra-core` が根 — workspace 内にこれより上流のものはありません。

```
vokra-core            （依存なし = 根）
  ├── vokra-ops
  ├── vokra-mmap
  ├── vokra-piper-plus
  ├── vokra-backend-cpu
  ├── vokra-backend-{metal,cuda,vulkan,webgpu}
  ├── vokra-eval        → core, ops, backend-cpu
  ├── vokra-convert     → core, ops, mmap
  ├── vokra-models      → core, ops, backend-cpu, piper-plus, mmap,
  │                       backend-{metal,cuda,vulkan,webgpu}（optional feature）
  ├── vokra-capi        → core, models, ops, mmap
  └── vokra-cli         → core, models, ops, convert, mmap
```

いくつかの *dev*-dependency は逆向きに張られています（例: `vokra-ops` の
テストが `vokra-models` を使う）。これらはテスト専用の辺であり、ビルドグラフ
を循環させません。

### 1.3 `integrations/` — 意図的に不変条件の外側

`integrations/` は root workspace から **除外**されています。現在 5 crate:

| パス | 用途 |
|---|---|
| `integrations/vokra-server` <!-- anchor: integrations/vokra-server --> | HTTP サーババイナリ（`FR-SV-01` / `FR-SV-02` / `FR-SV-04` / `FR-SV-05`） |
| `integrations/vokra-piper-g2p` <!-- anchor: integrations/vokra-piper-g2p --> | 実 8 言語 G2P ブリッジ |
| `integrations/vokra-godot` <!-- anchor: integrations/vokra-godot --> | Godot GDExtension |
| `integrations/vokra-server-bench` <!-- anchor: integrations/vokra-server-bench --> | サーバレイテンシ計測ハーネス |
| `integrations/vokra-cli-bench-server` <!-- anchor: integrations/vokra-cli-bench-server --> | CLI 側ベンチマークサーバ |

**なぜここでは外部 crate を使ってよいか。** zero-dependency 不変条件は
*特定の 1 ファイル* についての主張です — root の `Cargo.lock` が `vokra-*`
crate のみに解決されること。`integrations/` の各 crate は独自の `Cargo.lock`
を持つ独立 workspace なので、そこが何に依存しても root の解決には入りません。
runtime からは trait 境界越しに到達し、runtime グラフにリンクすることは
ありません。`scripts/check-zero-deps.sh` が root lockfile を検査し、ローカル
でも CI でも hard gate として働きます。

<!-- anchor: scripts/check-zero-deps.sh -->

### 1.4 `unsafe` の境界

workspace は safe-by-default です。workspace レベルで
`unsafe_code = "deny"` を設定し、すべての `unsafe` ブロックに `// SAFETY:`
コメントを要求します（clippy が強制）。安全な Rust では満たせない理由を持つ
**9 crate** だけがローカルに opt-out します（`NFR-RL-07`）:

| Crate | `unsafe` を要する理由 |
|---|---|
| `vokra-ops` | オペレータ hot path の SIMD intrinsics |
| `vokra-backend-cpu` | SIMD intrinsics と ISA dispatch |
| `vokra-backend-metal` | Objective-C runtime + Metal FFI |
| `vokra-backend-cuda` | CUDA Driver API + NVRTC FFI |
| `vokra-backend-vulkan` | Vulkan FFI |
| `vokra-backend-webgpu` | WASM extern-import shim |
| `vokra-capi` | C ABI 境界そのもの |
| `vokra-mmap` | POSIX `mmap` / Win32 file mapping |
| `vokra-wasm-harness` | `(ptr, len)` の WASM ABI 境界 |

`vokra-core` はこのリストに **入っていません**し、今後も入れません。公開 API
境界は、この 9 crate を含めどの crate でも安全に保ちます。

---

## 2. 実行モデル

Vokra は **kernel を共有する 2 つの経路**でモデルを実行します。自分が今どちら
の経路にいるかを知ることが、このコードベースで最も役に立つ前提知識です。

### 2.1 経路 A — グラフ評価器

<!-- anchor: crates/vokra-core/src/runtime/mod.rs -->
<!-- anchor: crates/vokra-core/src/runtime/tensor.rs -->

`vokra-core` の `runtime` モジュールがデータ運搬型のグラフ評価器を持ちます。
audio graph IR は *descriptor*（tensor は shape を持つがデータを持たない）で
あり、`run_graph` が実データを node から node へ受け渡しながら、
`Backend::eval_op` 経由で 1 op ずつ駆動します。

その契約:

- **1 グラフ = 1 バックエンド、silent fallback なし**（`FR-EX-08`）。評価
  開始前に `run_graph` がグラフ中の *すべて* の op をそのバックエンドが
  カバーしているか検査する。1 つでも未対応なら明示エラー。op ごとの CPU
  fallback も、ONNX Runtime 型の execution provider によるグラフ分割も行わない。
- **決定的なスケジュール。** node は topological order（Kahn 法、独立 node
  については index 安定）で実行されるため、同じグラフは毎回同じ順序で評価
  される。
- **検証はエンジン側に置く。** `eval_op` は計算だけを行い、`run_graph` が
  出力の arity と shape を宣言された descriptor と突き合わせる。

この経路は、新規・fused・グラフ前提のモデルに適した形です。

### 2.2 経路 B — `Compute` seam

<!-- anchor: crates/vokra-models/src/compute.rs -->

先行して存在したモデル（Whisper / piper-plus / CAM++）は **命令的**に書かれて
います。呼び出し側が所有する scratch バッファを使い、zero-malloc の hot path
で compute kernel を直接呼びます。これらをグラフエンジンへ書き換えると op 面
が大きく増え、数値 parity を危険に晒す一方で速度上の利得はありません（下層は
どちらも同じ kernel だからです）。

そこで、それらの呼び出し箇所は `Compute` という薄い型付き seam を経由します。
enum の arm を 1 つ差し替えるだけで、同じ GEMM が CPU から GPU へ移ります。

### 2.3 なぜこれが二重実装にならないか

**`(backend, op)` の組に対して kernel は 1 つで、両経路がそれを呼びます。**
CPU arm の `Compute::gemm_f32` は `Backend::eval_op` が呼ぶのと同一の
`vokra_backend_cpu::kernels::gemm_f32` を呼び、Metal arm では同一の
`MetalContext::gemm_f32` を呼びます。同期を取るべき 2 つ目の kernel は存在せず、
命令的経路とグラフ経路は同一バックエンド上で bit-for-bit 一致します。

この seam はモデル粒度で `FR-EX-08` も強制します。`Compute::for_backend` は
モデルが *必要とする* hot op 集合を受け取り、そのすべてをカバーしないバック
エンドの構築を拒否します。CPU を選ぶのは呼び出し側の明示的な選択であって、
暗黙の格下げではありません。

> **stale pointer についての注記。** `compute.rs` の冒頭コメントは
> `scratchpad/graph-engine-plan.md` という設計メモを参照していますが、この
> ファイルはリポジトリに含まれていません。本ページがその設計の公開かつ
> canonical な記述です。dangling な参照よりこちらを優先してください。

### 2.4 バックエンドの構成 — 6-file pattern

GPU / FFI 系の 4 バックエンドは共通の骨格を持つため、1 つ読めば残りも辿れます:

| ファイル | 責務 |
|---|---|
| `sys.rs` | 生 FFI 宣言 — 手書きの binding 層 |
| `probe.rs` | 実行時検出。このデバイス / ドライバは本当に使えるか |
| `context.rs` | device / queue / buffer の生存期間と compute kernel 群 |
| `backend.rs` | `vokra-core` の `Backend` trait 実装。op coverage の正直な申告を含む |
| `eval.rs` | グラフ経路向けの `eval_op` dispatch |
| `lib.rs` | crate ドキュメント、`unsafe` の opt-out、re-export |

必要に応じて 6 ファイルの外にファイルを足します。CUDA は `fa_v3.rs` と
`session_pool.rs`、Vulkan は `kernels.rs` / `plan.rs` / `spirv.rs`、WebGPU は
`plan.rs` / `wgsl.rs` を持ちます。

**CPU バックエンドはこの pattern に従いません** — FFI バックエンドではなく、
probe すべきデバイスも持たないためです。構成は `dispatch.rs` / `eval.rs` /
`features.rs` / `kernels/` / `lib.rs` / `pool.rs` / `selftest.rs` です。
6-file の骨格を 5 バックエンド全体に一般化しないでください。

---

## 3. 設計 red line

以下は確定済みの判断です。これらを越える PR は、実装の出来に関わらず却下され
ます。アイデアが悪いからではなく、**その判断のコストは既に支払われており、
再検討する費用が得られる利益を上回る**からです。規則そのものは
[CONTRIBUTING.md](../CONTRIBUTING.md) §5 にあり、理由をここに置きます。

<!-- anchor: CONTRIBUTING.md -->

### R1 — runtime で ONNX グラフをロードしない（`FR-LD-05`）

ONNX モデルは **オフライン変換器**（`vokra-convert`）だけが扱います。runtime
は onnxruntime / onnx / protobuf のいずれにも依存しません。

*理由。* 本プロジェクトは ONNX 音声スタックの問題群を出発点にしています。
実行時に ONNX グラフをロードすれば protobuf / abseil / onnx を依存グラフへ
連れ戻すことになり、`NFR-DS-02` が壊れます。そして zero-dependency である
ことこそが、Unity / Godot / モバイルへの単一バイナリ配布を成立させている
性質です。ONNX を 1 つのオフライン crate に閉じ込めることが、ツリーの残りを
綺麗に保つ手段になっています。

### R2 — piper-plus 推論経路に onnxruntime を入れない

MB-iSTFT-VITS2 の推論スタックは Rust で native 再実装されています。piper-plus
から流用しているのは当面 G2P のテキスト前処理だけです。

*理由。* wrap すると、「onnxruntime の代替である」ことを主張の全体としている
プロジェクトの e2e 経路に onnxruntime が残ります。加えて native 実装によって
`istft` オペレータに実利用者ができ、audio dialect の設計が仕様止まりではなく
検証されました。

### R3 — core に eSpeak-NG を入れない

*理由。* GPL-3.0 だからです。Vokra は Unity / Godot をはじめとするプロプラ
エタリな組み込み用途を対象にしており、そこでは GPL は製品を出す側にとって
受け入れられません。G2P は piper-plus 自身の MIT 実装、または IPA 辞書ベース
の手法から得ます。同じ理由で soxr / rubberband も除外します（R5）。

### R4 — NNAPI バックエンドを作らない

*理由。* Google が Android 15 時点で deprecated としています。Android の高速化
戦略を、deprecated な単一ベンダー抽象に賭けると、性能ではなく移行プロジェクト
を買うことになります。Android の GPU 高速化は Vulkan で担当します。

### R5 — soxr / rubberband を使わない

*理由。* R3 と同じく GPL です。リサンプリングは speexdsp（BSD）の resampler
設計に基づいて native 実装しています。

### R6 — 未対応は必ずエラー、silent fallback にしない（`FR-EX-08`）

これは CONTRIBUTING の red line 一覧には「red line」として載っていませんが、
両実行経路とバックエンド probe の水準（`NFR-RL-06`）で同じ厳しさで強制されて
います。

*理由。* silent CPU fallback は、kernel の欠落を「性能の謎」に変えてしまい
ます。コードは動き、数値ももっともらしく、regression は数週間後に他人の環境
での説明のつかないレイテンシ変化として現れます。明示エラーなら bug report
1 件で済み、その調査が丸ごと不要になります。

同じ判断がテストの許容値にも及びます。parity の `atol` は architectural bound
から導くものであって、CI が緑になるまで調整するものではありません。検査を
通すために許容値を広げるのは silent fallback と同種の失敗であり、レビューでも
そう扱います。

### zero-dependency の 2 つの正規の抜け道

`NFR-DS-02` は厳格ですが、壁ではありません。通り抜ける道はちょうど 2 つ
あり、どちらも「root の `Cargo.lock` が `vokra-*` のみに解決される」性質を
保ちます:

1. **first-party の optional feature。** GPU バックエンドは手書き生 FFI を
   持つ通常の `vokra-*` crate で、Cargo feature の背後で既定 OFF になって
   おり、default ビルドはそれらを名指しすらしません。新しい GPU / NPU 経路
   はこの形で入れます。
2. **隔離された integration workspace。** 外部 crate が本当に必要なコードは
   `integrations/` 配下に独自 workspace + 独自 lockfile で置き、trait 境界越し
   に接続します（§1.3）。

どちらも要らない変更なら、新しい依存も要りません。

---

## 4. バックエンドの追加

出発点は §2.4 の 6-file pattern です。概略:

1. **`crates/vokra-backend-<name>/` を first-party crate として作る。**
   optional な Cargo feature で gate し、default ビルドに影響させない。
2. **`sys.rs` を手書きする。** binding crate を足すのではなく必要な FFI を
   自分で宣言する — それが `NFR-DS-02` を保つ手段。プラットフォームの
   ライブラリはリンク時ではなく実行時にロードする（`dlopen` /
   `LoadLibrary`、WASM なら import object 解決）ので、ドライバの無いマシン
   でもバイナリは動く。
3. **`probe.rs` を最後ではなく次に書く。** 検出は正直に — 期待込みで先へ
   進まず、使えないデバイスは使えないと報告する（`NFR-RL-06`）。
4. **`context.rs` の kernel は CPU バックエンドを基準に実装する。** CPU
   kernel が数値のオラクルであり、parity テストはそれと比較する（`NFR-QL-01`）。
5. **`backend.rs` の op coverage を正直に実装する。** 実装した op だけを
   申告すること。過少申告は明示エラーになるだけで正しい挙動だが、過大申告は
   誤った数値を生む（`FR-EX-08`）。
6. **グラフ経路用に `eval.rs` を配線し、命令的経路用に `Compute` の arm を
   足す。** 同じ kernel を再利用すること（§2.3）。

すべての `unsafe` ブロックに `// SAFETY:` コメントが必要で、crate は workspace
マニフェストの opt-out リストに追加します（§1.4）。

---

## 5. 関連文書

- [requirement-ids.ja.md](requirement-ids.ja.md) — ソース中の `FR-*` /
  `NFR-*` / `BR-*` / `IF-*` を解決する
- [CONTRIBUTING.md](../CONTRIBUTING.md) — PR プロセス、required check、
  依存ポリシー
- [getting-started.ja.md](getting-started.ja.md) — 5 分のビルド & 実行
- [design/m0-03-gguf-loader.md](design/m0-03-gguf-loader.md) — GGUF ローダー設計
- [design/vokra-gguf-chunks.md](design/vokra-gguf-chunks.md) — `vokra.*`
  metadata chunk（`FR-LD-02`、`IF-07`）
- [design/quantization-policy.md](design/quantization-policy.md) — 量子化
  policy（`FR-QT-02`）
- [design/size-budget.md](design/size-budget.md) — バイナリサイズ予算
  （`NFR-DS-01`）
- [license-audit.md](license-audit.md) — モデル / 依存ライセンス監査
