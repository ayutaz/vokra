# API リファレンス

[English](api-reference.md) | **日本語**

Vokra の API surface と、各リファレンスの所在の索引。大半はソースから
**自動生成**される。本ページは薄いポインタであって手管理のコピーではない
（コピーは腐る）。何が生成で何が手書きかは §4 に記す。

## 1. Rust — docs.rs

Rust crate は `rustdoc` で documentation される。crate が publish されると
（リリーストレイン X-07）、各 crate が自身のページへ auto-link される:

- `https://docs.rs/vokra-core` — IR・`Backend` trait・GGUF loader・engine
- `https://docs.rs/vokra-capi` — C ABI surface crate（`IF-01`）
- `https://docs.rs/vokra-models`、`.../vokra-ops`、および backend crate 群

feature-gated な GPU/NPU バックエンドは `[package.metadata.docs.rs]` を持ち、
docs.rs がそのプラットフォーム固有 API をビルドする（Metal / CoreML は Apple
target、WebGPU は wasm32、CUDA / Vulkan / QNN は各 feature 経由）。同じものを
ローカルでビルドするには:

```sh
cargo doc --no-deps --open
```

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
  チュートリアルがリファレンス。初回の docs.rs render は crates.io publish
  （X-07）後に owner が確認する。

## Keeping this page current

**最終確認日: 2026-07-21 — workspace publish set と `include/vokra.h` に対して
確認。**

- **更新責任**: publish crate・新バインディング・C ABI 生成を変えた PR が、
  同一 PR で本索引と英語版を更新する。
- **review cadence**: 四半期 Go/No-go review（`NFR-MT-05`）。
- **生成 surface の再取得**:

```sh
scripts/gen-c-abi.sh && cargo doc --no-deps --workspace
```
