# バックエンドの追加

[English](backend-guide.md) | **日本語**

Vokra に**新しい計算バックエンド**を追加するための end-to-end ガイド —
全チェックリストを実ソースへの anchor 付きで示す。概念（crate マップ・実行
モデル・6-file パターン）は [architecture.md](architecture.ja.md) が
*俯瞰図*として提供する（§2.4 と §4 を先に読むこと）。本ページはその上で
「実際に 1 個作る手順」を担う。引用する `FR-*` / `NFR-*` / `IF-*` の意味は
[requirement-ids.md](requirement-ids.ja.md) で解決できる。

Vokra は現在 5 つの計算バックエンド（CPU / Metal / CUDA / Vulkan / WebGPU）
と 2 つの delegate scaffold（CoreML / QNN）を持つ。いずれも外部 binding
crate を使わない first-party な `vokra-*` crate であり、それが
zero-dependency 不変条件（`NFR-DS-02`）を保つ理由である。

## 1. 着手前: バックエンドが壊してはならない 2 つの不変条件

新規バックエンドは、`NFR-DS-02` を破らずに runtime グラフの外へ到達する
2 つの sanctioned な方法のうちの 1 つである（もう 1 つは隔離された
integration workspace。[CONTRIBUTING.md](../CONTRIBUTING.md) §7 参照）。
<!-- anchor: CONTRIBUTING.md -->
どのバックエンドでも次の 2 点が成り立つ:

- **外部依存ゼロ**。プラットフォーム FFI は `unsafe extern` ブロックに自分で
  宣言する。`metal` / `cudarc` / `ash` / `wgpu` の binding crate を足さない。
  root `Cargo.lock` は `vokra-*` のみを保つ（`scripts/check-zero-deps.sh` が
  強制）。
- **silent fallback 禁止（`FR-EX-08`）**。実行できない op、存在しない device
  は**明示エラー**であり、CPU への無言の差し替えは決してしない。coverage の
  過少申告は loud なエラー（正しい）、過大申告は誤った数値（誤り）。

## 2. 6-file パターン

GPU/FFI バックエンドは 6 module 構成を共有する。Metal バックエンドを
canonical なテンプレートとして使う:

| ファイル | 役割 |
|---|---|
| `sys.rs` <!-- anchor: crates/vokra-backend-metal/src/sys.rs --> | 手書きの生 FFI: `extern` 宣言とランタイムライブラリロード（`dlopen` / `LoadLibrary` または framework link）。binding crate 不使用。 |
| `probe.rs` <!-- anchor: crates/vokra-backend-metal/src/probe.rs --> | 正直な device 検出。device/driver が不在なら `VokraError::BackendUnavailable` を返す（`NFR-RL-06`）。 |
| `context.rs` <!-- anchor: crates/vokra-backend-metal/src/context.rs --> | 生きた device/queue/allocation と compute kernel 本体。 |
| `backend.rs` <!-- anchor: crates/vokra-backend-metal/src/backend.rs --> | `Backend` trait 実装: `supports()` + `eval_op()` を lock-step に保つ。 |
| `eval.rs` <!-- anchor: crates/vokra-backend-metal/src/eval.rs --> | `run_graph` が駆動する graph-executor の per-op arm。 |
| `lib.rs` <!-- anchor: crates/vokra-backend-metal/src/lib.rs --> | crate root・feature gating・re-export。 |

**正直な注記 — CPU バックエンドは 6-file ではない**。`vokra-backend-cpu` は
別構成（`dispatch.rs` / `features.rs` / `kernels/` / `pool.rs` /
`selftest.rs`）である。これは runtime ISA-dispatch バックエンド（`FR-BE-01`）
であり device FFI バックエンドではないためだ。6-file パターンは
**GPU/FFI/NPU** バックエンド向けのテンプレートであり、CPU をここに押し込め
ない。Vulkan と WebGPU も shader 事情でファイルを数個追加する
（`spirv.rs` / `wgsl.rs` / `plan.rs`）。

## 3. add-a-backend チェックリスト

手順 1–5 で crate と coverage 契約を立ち上げ、6–9 で実行へ配線して数値検証
する。

1. **新 crate `crates/vokra-backend-<x>/`**。既定 OFF の optional Cargo
   feature で gate し、default（および他プラットフォーム）ビルドが名前すら
   出さないようにする。実 FFI は `cfg(target_os = …)` /
   `cfg(target_arch = …)` で target-gate する。
2. **`sys.rs` — 生 FFI、実行時ロード**。プラットフォームライブラリは
   link-time ではなく `dlopen` / `LoadLibrary`（または wasm import-object
   解決）でロードし、driver が無い機体でもバイナリが動くようにする。binding
   crate 不使用（`NFR-DS-02`）。
3. **`probe.rs` は最後ではなく次に**。検出は正直に: device が使えなければ
   楽観的に進めず `VokraError::BackendUnavailable` を返す（`NFR-RL-06`）。
4. **`BackendKind` variant を追加**（`vokra-core`）。 <!-- anchor: crates/vokra-core/src/backend.rs --> enum は `#[non_exhaustive]`
   なので新 variant は後方互換。NNAPI だけは永久に追加しない variant である
   （`FR-BE-07`）。
5. **`Backend::supports()` + `Backend::eval_op()` を lock-step 実装**。default
   の `eval_op` は全 op で `UnsupportedOp` を返すので half-wired でもコンパイル
   は通る。実装した op だけを override し、`supports()` はちょうどその集合で
   `true` を返す（`FR-EX-08`）。
6. **graph-executor arm を配線**（`eval.rs`）。CUDA / Vulkan arm を手本に。 <!-- anchor: crates/vokra-backend-cuda/src/eval.rs -->
7. **`Compute` seam を配線**（`vokra-models`）。命令的モデルの hot path 用に、
   *同じ* kernel を再利用する — (backend, op) ごとに 1 実装、二重実装ゼロ。 <!-- anchor: crates/vokra-models/src/compute.rs -->
8. **`FR-EX-08` を end-to-end で保つ**。未 cover の op、device 不在は graph
   seam でも `Compute` seam でも明示エラー。
9. **CPU バックエンドとの parity**。CPU kernel が数値 oracle であり、
   differential test が新バックエンドを `atol = 0.01` で照合する
   （`NFR-QL-01`）。この許容は architectural bound であって調整つまみでは
   ない（§5 参照）。

## 4. worked example: 最新のバックエンド

最も新しく追加されたのは **CoreML**（Apple ANE）と **QNN**（Qualcomm
Hexagon）の *delegate*（`FR-BE-06`）である。これらは上記の crate-scaffold
手順 1–4 の最新例だ: `crates/vokra-backend-coreml/`
<!-- anchor: crates/vokra-backend-coreml/src/lib.rs --> は `sys.rs` /
`probe.rs` / `backend.rs` / `lib.rs` を持つ first-party crate で、既定 OFF の
`coreml` feature で gate され macOS / iOS に target-gate されている。

**delegate は 6-file な GPU バックエンドと異なり、ガイドはそれを正直に書く**。
delegate は宣言された submodel を vendor framework に渡し、ANE / GPU / CPU への
配置は *framework 側* が内部で行う。これは Vokra 側の op 分割（`Backend`
trait の uniform-coverage 規則が禁止）でも silent fallback でもない。よって:

- canonical な **6-file** テンプレートは依然として 5 つの GPU/FFI バックエンド
  （Metal / CUDA / Vulkan / WebGPU）である — 別の *kernel* バックエンドを足す
  ときはこれらを使う。
- CoreML / QNN は *delegate* のテンプレート。op 実行パスは model-supply ADR が
  批准された後に land するので、今は全 hot op が明示的な `UnsupportedOp`、
  到達可能な NPU が無い host は明示的な `BackendUnavailable` である。これは
  正直な scaffold 状態であってバグではない。

delegate の C レベル selector は v1.0-rc 期間中は意図的に **export しない**。
post-bakeoff の `IF-01` 決定までは Rust surface（`with_backend`）が唯一の選択
手段である。具体仕様は引用前に on-disk の crate を確認すること — 本節は各
バックエンド land ごとに再検証される（下記 meta ブロック参照）。

## 5. レッドラインと落とし穴

- **NNAPI は恒久非対応（`FR-BE-07`）**。Android GPU は Vulkan、Hexagon NPU は
  QNN（`FR-BE-06`）。NNAPI variant を足さない。
- **GPL/LGPL コード禁止**（codec / resampler を含む） — バックエンド内部でも
  成り立つ（[CONTRIBUTING.md](../CONTRIBUTING.md) §5）。
- **shader は precompile、JIT 禁止**。Vulkan バックエンドは GLSL を実行時
  コンパイルせず pre-compiled SPIR-V を commit する。そのモデルに従う。
- **全 `unsafe` ブロックに `// SAFETY:` コメント**。`vokra-core` 自身は
  `unsafe`-free を保つ — `unsafe` はバックエンド crate に置く。
- **緑にするために parity 許容を緩めない**。`atol` は architectural bound
  （`NFR-QL-01`）。parity 失敗は kernel が誤っているのであって、bound が厳し
  すぎるのではない。

## 6. owner / contributor 境界

本ガイドは*手順*を documentation する。device は回さない: 実機（Apple Neural
Engine・Hexagon device・Android 端末）上の実 GPU / NPU parity と soak は owner
タスクである。contributor は crate・coverage 契約・CPU-oracle parity harness
を land し、owner が実機で回して sign-off する。

## Keeping this page current

**最終確認日: 2026-07-21 — 稼働中の 5 バックエンド + CoreML / QNN delegate
scaffold に対して確認。**

- **更新責任**: 新バックエンドを land した者（または 6-file 構成・`Backend`
  trait を変えた者）が、同一 PR で本ページと英語版を更新し、上の「確認対象」
  リストを更新する。
- **review cadence**: 四半期 Go/No-go review（`NFR-MT-05`）で見直す。本ページ
  は op coverage が時間とともに増えるバックエンドを名指しするため。
- **事実の再取得**（crate 構成・trait 契約・enum variant）:

```sh
ls crates/vokra-backend-metal/src/     # canonical な 6-file 構成
sed -n '/pub trait Backend/,/^}/p' crates/vokra-core/src/backend.rs
sed -n '/pub enum BackendKind/,/^}/p' crates/vokra-core/src/backend.rs
```
