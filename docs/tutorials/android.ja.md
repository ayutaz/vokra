# Android（native JNI）チュートリアル

[English](android.md) | **日本語**

これは **native** な Android パス: `libvokra.so` をビルドし、JNI 経由でロード
し、C ABI を直接呼ぶ。Unity 経由で統合する場合は [Unity チュートリアル](unity.ja.md)
を使う（本ページはそれを繰り返さない）。Android の GPU 高速化は **Vulkan**
である（`FR-BE-07`: NNAPI は恒久非対応）。

## 1. CPU-only ライブラリのビルド

リポジトリは `scripts/build-android.sh` <!-- anchor: scripts/build-android.sh -->
（M2-11 由来）を同梱する。Android NDK で JNI ライブラリをクロスコンパイルする。
`ANDROID_NDK_HOME` が必須で、API level 24（Android 7.0）を対象とし、
`target/aarch64-linux-android/release/libvokra.so` を生成する:

```sh
ANDROID_NDK_HOME=/path/to/ndk scripts/build-android.sh
```

**正直な scope 注記**。`build-android.sh` は `--no-default-features` の
**CPU-only** ビルド（ヘッダに「NO Metal, NO CUDA」）である。Vulkan パスは
ビルドしない — それは §2 の話だ。CPU-only ライブラリはどこでも動く baseline
SKU。GPU 高速化が欲しいときだけ Vulkan を足す。

## 2. Vulkan バックエンド付きビルド（任意）

素の `cargo build --target aarch64-linux-android` で「そのまま動く」ものは無い:
NDK linker が無いと link 段階で失敗する。NDK toolchain を明示設定し
（`godot-crossbuild` CI workflow と同じパターン。ビルド機に応じて `<host-tag>`
を選ぶ。例 `darwin-x86_64` / `linux-x86_64`）、`vulkan` feature 付きで
`vokra-capi` をビルドする:

```
export NDK="$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/<host-tag>/bin"
export CC_aarch64_linux_android="$NDK/aarch64-linux-android24-clang"
export CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER="$CC_aarch64_linux_android"
cargo build -p vokra-capi --no-default-features --features vulkan \
  --target aarch64-linux-android --release
```

Vulkan FFI は実行時に `dlopen` ロードされる（何も link・bundle しないので
zero-dependency 不変条件 `NFR-DS-02` は保たれる）。`libvulkan.so` は device 上
でロードされる。

## 3. モデルのロードと JNI からの呼び出し

モデルは**決して bundle しない**。`libvokra.so` を `jniLibs/arm64-v8a/` 下に
同梱し、GGUF をアプリの private storage に push し、その実ファイルパスを C ABI
に渡す。ライブラリをロードし extern を宣言する:

```
class VokraBridge {
    companion object { init { System.loadLibrary("vokra") } }
    external fun transcribe(modelPath: String, wavPath: String): String
}
```

JNI の `.c`/`.cpp` shim は `vokra.h` を include し、[getting-started](../getting-started.ja.md#5-min-call-from-c-abi)
の C 例が示すのと同じ session API を呼ぶ（`vokra_session_create_from_file` →
`vokra_asr_transcribe` → `vokra_string_free` → `vokra_session_destroy`）。
モデルは `vokra-cli convert` でオフライン GGUF 化し、アプリの `filesDir` 下の
パスにコピーする（C ABI が `mmap` できる実ファイルパス — `NFR-RL-04`）。

## 4. バックエンド選択: Vulkan、NNAPI は決して使わない

バックエンドは C ABI の session-builder surface で明示選択する。Vulkan の
loader/device 不在は明示的な `BackendUnavailable`、Vulkan バックエンドが cover
しない op は明示的な `UnsupportedOp` — Vokra は CPU へ無言で落ちない
（`FR-EX-08`）。**NNAPI は選択肢ではなく**、今後も追加しない（`FR-BE-07`;
Google は Android 15 で deprecated）。CPU バックエンドは完全サポートの*明示*
選択肢として残る。

## 5. 性能とオンデバイス検証

Android の目標は Whisper base で RTF < 0.7（`NFR-PF-06`）。この数値は**実機**
計測である: 本チュートリアルはビルドと呼び出しパスを documentation するが、
実機での RTF 計測（および Android Vulkan soak）は owner タスクだ —
`docs/m3-18-android-rtf-handover.md` 参照。ビルドの成功を RTF の成功と読み
替えないこと。両者は別物である。

## 6. トラブルシューティング

| 症状 | 原因 / 対処 |
|---|---|
| `error: ANDROID_NDK_HOME must be set` | `build-android.sh` は NDK root を要する。NDK r25+ を入れ `ANDROID_NDK_HOME` を export する。 |
| Vulkan ビルドで `error: linking with 'cc' failed` | NDK linker 未設定。§2 の `CC_*` / `CARGO_TARGET_*_LINKER` env が必須。 |
| 実行時に `UnsatisfiedLinkError: libvokra.so` | `.so` が `jniLibs/arm64-v8a/` に無いか、別 ABI をビルドした。 |
| device 上で `BackendUnavailable: vulkan` | 使える Vulkan loader/device が無い。CPU ビルドへ*明示的に*fall back する（`--backend cpu`）。無言では落ちない。 |

## Next steps

- [バックエンドの追加](../backend-guide.ja.md) — Vulkan バックエンドの配線
- [デスクトップ CLI](cli.ja.md) — 同梱する GGUF を作る `convert` 手順
- [Web（WASM / WebGPU）](web.ja.md) — この native パスのブラウザ版の兄弟

## Keeping this page current

**最終確認日: 2026-07-21 — `scripts/build-android.sh`（CPU-only, API 24）と
`godot-crossbuild` の NDK-linker 先例に対して確認。**

- **更新責任**: Android ビルドスクリプト・NDK floor・Vulkan feature 配線を
  変えた PR が、同一 PR で本ページと英語版を更新する。
- **review cadence**: 四半期 Go/No-go review（`NFR-MT-05`）。オンデバイス RTF
  は owner が再計測するもので、ここで捏造しない。
- **ビルド事実の再取得**:

```sh
sed -n '1,40p' scripts/build-android.sh
```
