# API リファレンス

[English](api-reference.md) | **日本語**

Vokra の API surface と、各リファレンスの所在の索引。大半はソースから
**自動生成**される。本ページは薄いポインタであって手管理のコピーではない
（コピーは腐る）。何が生成で何が手書きかは §4 に記す。

## 1. Rust — docs.rs

Rust crate は `rustdoc` で documentation される。将来 crate が publish されると、
各 crate が自身のページへ auto-link される:

- `https://docs.rs/vokra-core` — IR・`Backend` trait・GGUF loader・engine
- `https://docs.rs/vokra-capi` — C ABI surface crate（`IF-01`）
- `https://docs.rs/vokra-models`、`.../vokra-ops`、および backend crate 群

feature-gated な GPU/NPU バックエンドは `[package.metadata.docs.rs]` を持ち、
docs.rs がそのプラットフォーム固有 API をビルドする（Metal / CoreML は Apple
target、WebGPU は wasm32、CUDA / Vulkan / QNN は各 feature 経由）。memory-safe な
単一 crate に絞ってローカルでビルドするには:

```sh
cargo doc -p vokra-core --no-deps --open
```

maintainer の workspace 全体 rustdoc は 16 GB の開発 Mac ではなく VAST / CI で
実行する。

## 2. C ABI — `include/vokra.h`

canonical な C リファレンスは生成済みヘッダ
[`include/vokra.h`](../include/vokra.h) である。`scripts/gen-c-abi.sh` が
`vokra-capi` crate から生成し、その doc コメントがリファレンス本文となる。CI の
drift check が Rust ソースとの同期を保つ。Unity / Godot / Swift / Kotlin /
Python / JS の全バインディングはこの 1 つのヘッダの上に乗る（`IF-01`）。Vokra は
通常の Cargo crate / 単一ライブラリとして配布されるので、このヘッダ +
ライブラリが統合 surface の全体である（`NFR-DS-03`）。

## 3. 言語バインディング

各バインディングは C ABI 上の慣用的な surface を自身で documentation する:

- **Unity（C#）** — [Unity チュートリアル](tutorials/unity.ja.md)
- **Python** — [`bindings/python/README.md`](../bindings/python/README.md)
- **Godot（GDScript）** — [Godot チュートリアル](tutorials/godot.ja.md)
- **Swift / iOS** — [`Package.swift`](../Package.swift) SwiftPM マニフェストと
  [iOS チュートリアル](tutorials/ios.ja.md)

## 4. 何が自動生成で、何がそうでないか

- **自動生成**: Rust docs（rustdoc → docs.rs）と C ヘッダ（`gen-c-abi.sh` →
  `include/vokra.h`）。ソースから再生成され、source of truth である。
- **手書きだが薄い**: 本索引とバインディングチュートリアル。生成リファレンスと
  動く例を指すだけで、API の 2 個目のコピーではない。
- **deferred（正直に）**: C ヘッダの HTML 化（doxygen）と言語別 HTML ジェネ
  レータ（C# / Python / Swift の doc ツール）は未配線 — 当面はヘッダコメントと
  チュートリアルがリファレンス。初回の docs.rs render は crates.io publish 後に
  owner が確認する。

## 5. 現行 0.3.0 release と Apple 検証 status

現行の workspace release line は `0.3.0` である。以下の parity 数値は文書更新前の
snapshot であり、PR #79 の `d8a93bc3` を `origin/main` の `41ce9ffd` と照合して
109 pass、13 expected skip を記録した時点のものである。live public audit の現行値は
194 repository（GGUF repository 193、GGUF file 198）。CPU coverage は `full=131`、
`partial=42`、
`no-runtime-binder=20`、`not-artifact=1`、Metal は `full=131`、
`blocked-by-cpu=62`、`not-artifact=1`、source-level CPU-only は 0 である。
現時点の release tag は 0、GitHub Release も 0 である。

GigaAM v3 / Multilingual は conservative な Metal code route が complete だが、
Apple hardware verdict は未取得。OmniASR も認証済み Scaleway run 待ちである。
CI Quality の `hf-mac-coverage-unit` と live advisory は最新 PR で green だが、
CI/audit 結果を Apple 実機 evidence の代用とはしない。

## Keeping this page current

**最終確認日: 2026-08-31 — GitHub `main`
`41ce9ffdd4b0959497f55afa5016822f77a8a7b6`、文書更新前に照合したコード
baseline branch `feat/mac-cpu-metal-full-coverage-2026-08-28` の
`9f69277d8a0d5df574c1ee95563bd1f005de91d0`、および `include/vokra.h` に対して
確認。** pre-alpha の Python generator と checked-in `ctypes` table は、生成 C
の全 57 function と完全に一致する。header は 15 typedef、4 enum、2 concrete
struct、9 opaque handle を持つ。高水準 Python package は、全 C handle に wrapper
class を持つのではなく、より小さい慣用的な surface のままである。

- **更新責任**: publish crate・新バインディング・C ABI 生成を変えた PR が、
  同一 PR で本索引と英語版を更新する。
- **review cadence**: 四半期 Go/No-go review（`NFR-MT-05`）。
- **生成 surface の再取得**:

```sh
scripts/gen-c-abi.sh
# maintainer の workspace rustdoc は開発 Mac ではなく VAST/CI で実行:
cargo doc --no-deps --workspace
```
