# Vokra

[English](README.md) | **日本語**

**Vokra** は、オーディオ AI — TTS・ASR・Speech-to-Speech・話者識別・VAD・
音楽生成・音源分離・音響理解 — に特化した Rust 製推論ランタイムです。
オーディオワークロード向けの ONNX / ONNX Runtime 代替として構築されています。

汎用推論ランタイムは音声モデルに対して慢性的に力不足です。STFT / iSTFT の
ストリーミング状態、ボコーダの数値精度、ニューラルコーデック (RVQ / FSQ) の
デコード、Flow Matching サンプラ、ビーム / CTC / RNN-T デコーディング、
VAD、話者エンベディング — こういった要素はどれもグラフエクスポート時に壊れやすかったり、
ホスト側の泥臭い接着コードに追いやられがちです。Vokra はこれらを一級のネイティブ演算子
として扱います。

- **発音**: "vo-krah"（英語）/「ヴォクラ」（日本語）
- **ライセンス**: [Apache-2.0](LICENSE)（依存クロージャに GPL / LGPL は
  一切含みません）
- **リポジトリ**: <https://github.com/ayutaz/vokra>
- **モデルハブ**: <https://huggingface.co/vokra>

> API・ファイルフォーマット・モデル対応はいずれも pre-1.0 で、破壊的変更が
> 入り得ます。安定 C ABI は最初の安定リリースと同時に提供予定です。

## 目次

- [主要な特徴](#主要な特徴)
- [対応モデル](#対応モデル)
- [対応プラットフォームとバックエンド](#対応プラットフォームとバックエンド)
- [はじめに](#はじめに)
- [C ABI の利用](#c-abi-の利用)
- [アーキテクチャ概要](#アーキテクチャ概要)
- [バインディングと統合](#バインディングと統合)
- [モデル配布](#モデル配布)
- [piper-plus 統合](#piper-plus-統合)
- [ドキュメント](#ドキュメント)
- [関連プロジェクト](#関連プロジェクト)
- [コントリビューション](#コントリビューション)
- [法務・コンプライアンス](#法務コンプライアンス)
- [ライセンス](#ライセンス)

## 主要な特徴

- **音声モデルのネイティブ再実装**（whisper.cpp スタイル）: モデルコードは
  Rust で書き下し、上流の `safetensors` / `GGUF` チェックポイントを直接
  読み込みます。ランタイムは ONNX グラフを一切ロードしないので、`onnx` /
  `protobuf` / `abseil` の推移的依存を持ち込みません。
- **ゼロ外部依存の不変条件**: ルート `Cargo.lock` は first-party の
  `vokra-*` クレートのみで解決されます。GPU / NPU バックエンドは手書きの FFI
  を使い（`metal-rs` / `cudarc` / `ash` / `wgpu` に依存しない）、いずれも
  Cargo フィーチャで opt-in されます。デフォルトビルドは常に first-party
  だけで完結し、CI の [`scripts/check-zero-deps.sh`](scripts/check-zero-deps.sh)
  で強制されます。
- **音声ファーストな演算子セット**: 明示的な window / hop / normalization /
  RFFT 属性を持つ STFT / iSTFT、mel フィルタバンク、ポリフェーズリサンプラ、
  ボコーダチェーン（HiFi-GAN、BigVGAN、HiFTNet、Vocos 型 iSTFT ヘッド）、
  設定可能な CFG モードとスケジュールを持つ Flow Matching サンプラ、ニューラル
  コーデックデコーダ（DAC、Mimi、WavTokenizer、X-Codec 2）、ビームサーチ /
  CTC / RNN-T デコーディング、ストリーミング KV キャッシュ（paged、
  3D `[time, stream, codebook]`）、VAD、音声強調（AEC / AGC / HPF /
  loudness-norm / DeepFilterNet3 / WPE 残響除去）、話者エンベディング、
  F0 抽出（YIN / PyIN と neural extractor 群）、逆テキスト正規化、
  自己教師ありエンコーダ群が共有する ViT 音響エンコーダプリミティブ、
  客観品質メトリクス（UTMOS、SI-SNR / SI-SDR / SDR、STOI、WER / CER）を
  一級 op として持ちます。
- **一級バックエンドとしての CPU** とランタイム ISA ディスパッチ: x86-64 は
  SSE2 baseline から AVX2、AVX-512F/DQ/BW/VL、AVX-512 VNNI/BF16、AVX-VNNI
  256-bit、AMX まで。ARM64 は NEON、fp16 演算、dotprod (SDOT/UDOT)、i8mm、
  bf16。RVV 1.0 baseline。K-quants（Q4_K / Q5_K / Q6_K）と、数値的に脆弱な
  ボコーダに対して INT8 を拒否する minimum-dtype レジストリ付きのレイヤ別
  量子化ポリシーを含みます。
- **サイレントフォールバック禁止**: バックエンドが実装していない演算子は
  明示的な loud エラーです。GPU バックエンドが黙って CPU にドロップすること
  は絶対にありません。誤った回答よりもハードエラーを優先します。
- **音声メタデータ付き GGUF**: モデルファイルは GGUF に `vokra.*` チャンク
  （`vokra.frontend.*`、`vokra.whisper.*`、`vokra.piper.*`、
  `vokra.provenance.*`、`vokra.quant.*`、`vokra.schema.*`）を追加した形式
  で、フロントエンド仕様・量子化ポリシー・ライセンス provenance がすべて
  weight と一緒に運ばれ、bit-exact に再現可能です。
- **クロスプラットフォーム配布**: 単一ライブラリ、単一 C ABI ヘッダ、静的
  または動的リンク。iOS XCFramework + Swift Package、Unity UPM パッケージ、
  Godot GDExtension、Python `ctypes` wheel、OpenAI Whisper / vLLM /
  piper-plus / Wyoming Protocol 互換エンドポイントを提供する HTTP
  互換サーバ。
- **デフォルトで安全な Rust**: ワークスペース全体で `unsafe_code = "deny"`
  を設定。`unsafe` はバックエンドと FFI クレートでのみ許可され、
  `clippy::undocumented_unsafe_blocks = "deny"` によって `// SAFETY:`
  コメントが強制されます。
- **構造的ライセンス衛生**: コンプライアンスゲートが、明示的な research flag
  / `--allow-noncommercial` を指定しない限り、非商用ライセンスの weight
  （F5-TTS、Fish-Speech、EnCodec、X-Codec 2）をデフォルト経路で拒否します。

## 対応モデル

Vokra は以下のモデルのネイティブ実装を同梱します。forward pass は Rust で、
weight ローダとトークナイザはランタイムに組み込まれており、GGUF ファイルを
生成するコンバータは `vokra-convert` クレートに含まれます。受け付ける
`--model` kind の権威ある一覧は `vokra-cli convert --help` です。

もう一つ、より大きなアーキテクチャ群は、コンバータ・GGUF ローダ・厳格な
アーキテクチャ検証までが land 済みで forward pass のみ deferred です。これらは
[loud partial](#forward-pass-が-deferred-のアーキテクチャ) として別掲しており、
黙って誤答することはありません — 名前を挙げて拒否します。

**ASR**
- Whisper — `base`、`small`、`medium`、`large-v3`、`turbo`
- Voxtral — `Mini-3B`、`Small-24B`（大容量バリアント向けストリーミングローダ）
- Canary-Qwen-2.5B（FastConformer + Qwen デコーダ）
- omniASR-CTC — 300M / 7B バリアント
- Charsiu（wav2vec2 CTC）
- Kyutai STT
- distil-whisper large-v3.5 / kotoba-whisper（Whisper large-v3 エンコーダ +
  2 層蒸留デコーダ）
- Parakeet — TDT-0.6B-v3 / CTC-1.1B（NVIDIA FastConformer）
- Zipformer / E-Branchformer / Hybrid CTC-Attention デコーダ

**TTS**
- piper-plus（ネイティブ MB-iSTFT-VITS2、8 言語 G2P: JA / EN / ZH / ES / FR
  / PT / SV / KO）
- Kokoro-82M
- CosyVoice2（FSQ tokens + Qwen2.5-0.5B AR + chunk-aware CFM → mel → HiFTNet）
- Style-Bert-VITS2 v2 — 多言語（JA / EN / ZH）、言語別コンディショニング
  エンコーダ付き: DeBERTa v2（JA）、DeBERTa v3（EN）、
  Chinese-RoBERTa-wwm-ext（ZH）
- VoxCPM-0.5B / VoxCPM2-2B
- Qwen3-TTS 1.7B
- Fun-CosyVoice3-0.5B
- Chatterbox-Multilingual（23 言語ゼロショット）と Turbo / Nano バリアント
- StyleTTS 2、Dia-1.6B（text-to-dialog）、VibeVoice-1.5B（長尺・多話者）、
  Zonos-v0.1
- 日本語: Irodori-TTS（rectified-flow DiT）、ESPnet 系 VITS

**Speech-to-speech（フルデュプレックス）**
- Sesame CSM-1B
- Moshi（Helium + Mimi コーデック）

**VAD**
- Silero VAD v5（デフォルト）/ v6.2.1
- FSMN-VAD

**キーワードスポッティング / ウェイクワード**
- openWakeWord
- microWakeWord — INT8 forward は `no_std` な `vokra-kws-micro` クレートで
  動作（Cortex-M55 クロスビルド）。checkpoint からの chain ロードは未配線で、
  未設定の detector は「何も起きなかった」ではなく拒否を返します

**話者エンベディング / 検証**
- CAM++（192 次元、ゼロショット音声クローンの入力）
- TitaNet-L
- ECAPA-TDNN

**F0 / ピッチ**
- RMVPE（U-Net + BiGRU forward を実装済み。実 checkpoint に対する
  テンソル名の走査は未検証で、ブロックを 1 つも見つけられなければ
  loud に拒否します）
- FCPE（実 Conformer forward）
- CREPE（実 6-block CNN、5 モデルサイズ）
- YIN / PyIN — weight 不要の DSP、checkpoint 不要
  （`vokra-cli f0 --algo yin|pyin`）

**ニューラルコーデック**
- DAC（24 kHz）、Mimi、WavTokenizer、X-Codec 2（研究用途のみ、
  CC-BY-NC-4.0）

**音声強調**
- DeepFilterNet3、NSNet2、DTLN-AEC、AGC、HPF、loudness normalization
- WPE 残響除去（weight 不要、`fgnt/nara_wpe` からの転写）

**ボコーダ**
- HiFi-GAN、および `vokra-ops` 側の HiFTNet / iSTFT head・BigVGAN 系
  anti-aliased upsampling プリミティブ

**テキスト正規化・句読点復元**
- CT-Transformer 句読点復元
- WeTextProcessing — 逆テキスト正規化 / テキスト正規化

**ダイアライゼーション**
- pyannote segmentation-3.0（PyanNet VAD / 話者セグメンテーション backbone）

**客観品質**
- UTMOS22-strong

### forward pass が deferred のアーキテクチャ

以下のファミリはコンバータ・GGUF ローダ・厳格なアーキテクチャタグ検証まで
land 済みで、forward pass は **loud partial** です。もっともらしい誤答では
なく、欠けているプリミティブと、それを規定する上流ソースを名指しした
エラーを返します。プリミティブが揃い実 checkpoint で検証された時点で利用可能に
なります。黙って動くことはありません。

- **音楽生成・解析** — MusicGen、MAGNeT、MelodyFlow、JASCO、AudioGen、
  AudioLDM2、MT3 採譜、Beat-This ビートトラッキング
- **音源分離** — Demucs、SepFormer、Conv-TasNet
- **音響表現エンコーダ** — ATST / EAT / M2D / MAEST（Beat-This と併せて
  `vokra_ops::vit` プリミティブを共有する 5 件）、および W2V-BERT-2
  （Conformer 系で deferred 理由が別）、WavLM、CLAP
- **音響タグ付け・分類** — PANNs、ディープフェイク検出、言語識別、emotion2vec
- **品質評価** — UTMOSv2、NISQA、TorchAudio-SQUIM、DNSMOS P.808 / P.835
- **ASR** — Canary-1B-Flash、Parakeet-TDT-1.1B、GigaAM、Whisper-Medusa、
  FireRed-AED、Moonshine、SenseVoiceSmall
- **VAD・ターンテイキング** — TEN-VAD、FireRed-VAD、smart-turn
- **強調** — GTCRN、StoRM、facebook-denoiser
- **話者** — ReDimNet、3D-Speaker ERes2Net、Sortformer ダイアライゼーション
- **その他** — AudioSR 超解像、DiffSinger 歌声合成、ChatTTS、Voila、
  SNAC / Vocos / BigVGAN のコーデック・ボコーダヘッド

ライセンスの詳細は [`docs/license-audit.md`](docs/license-audit.md) を、
そこから導かれる配布規則は [`docs/legal-compliance.md`](docs/legal-compliance.md)
を参照してください。

## 対応プラットフォームとバックエンド

以下のプラットフォームはすべて単一ライブラリ・単一 C ABI で対応対象です。
バックエンドアクセラレーションは Cargo フィーチャで有効化するため、
デフォルトビルドはゼロ依存のままです。

| バックエンド | Cargo フィーチャ | 備考 |
|---|---|---|
| CPU（デフォルト） | — | x86-64 SSE2 → AVX2 → AVX-512F/DQ/BW/VL → AVX-512 VNNI/BF16 → AVX-VNNI 256 → AMX；ARM64 NEON → fp16 → dotprod → i8mm → bf16；RVV 1.0 |
| Metal（macOS / iOS） | `metal` | 手書きの生 `objc` + Metal FFI、MSL コンピュートカーネル |
| CUDA（Windows / Linux） | `cuda` | Driver API + NVRTC を `dlopen` / `LoadLibrary` で実行時ロード — NVIDIA ライブラリは同梱しません（NVIDIA EULA 準拠） |
| Vulkan（Android / Linux / Windows） | `vulkan` | dlopen + pre-compiled SPIR-V、subgroup と cooperative matrix + フォールバック |
| WebGPU / WASM（ブラウザ） | `webgpu`, target `wasm32-unknown-unknown` | wasm extern-import shim、`wgpu` / `wasm-bindgen` 非依存 |
| CoreML（Apple ANE） | `coreml` | opt-in delegate スキャフォールド |
| QNN（Qualcomm Hexagon） | `qnn` | opt-in delegate スキャフォールド |

**オペレーティングシステム**: Windows、macOS、Linux、Android、iOS、
モダンブラウザ（WebGPU / WASM SIMD128 + threads 経由）。

**明示的非対応**: NNAPI（Google が Android 15、2024-10 で deprecated 化）、
Piper `OHF-Voice/piper1-gpl`（GPL-3.0 + eSpeak-NG GPL-3.0 の推移的依存 —
Piper 系で対応するのは依頼者作の MIT ライセンス
[`piper-plus`](https://github.com/ayutaz/piper-plus) フォークのみ）。

## はじめに

### 前提

- Rust toolchain（edition 2024）。MSRV は `1.85`。`vokra-backend-cpu` の
  一部 AVX-512 intrinsic は `1.89` を要求します。<https://rustup.rs> から
  インストール。
- C コンパイラ（C ABI をリンクする場合のみ）。

### CLI をビルドする

```sh
git clone https://github.com/ayutaz/vokra.git
cd vokra
cargo build --release -p vokra-cli
# バイナリは target/release/vokra-cli
```

### モデルをダウンロードする

<https://huggingface.co/vokra> の各モデルカードに、取得すべき `.gguf`
ファイルが明記されています。例えば Whisper base の checkpoint を取得するには:

```sh
# curl / wget / huggingface-cli のいずれでも可 — ファイルは素の GGUF blob です。
huggingface-cli download vokra/whisper-base whisper-base.gguf --local-dir .
```

### 音声を文字起こしする

```sh
target/release/vokra-cli run whisper-base.gguf --input audio.wav
```

### GPU バックエンドで推論する

該当する Cargo フィーチャをビルド時に有効化してください:

```sh
# macOS / iOS
cargo build --release -p vokra-cli --features metal
target/release/vokra-cli run whisper-base.gguf --input audio.wav --backend metal

# Linux / Windows（NVIDIA GPU、開発者側 CUDA インストール必須）
cargo build --release -p vokra-cli --features cuda
target/release/vokra-cli run whisper-base.gguf --input audio.wav --backend cuda
```

### 上流 checkpoint を変換する

```sh
target/release/vokra-cli convert --model whisper \
  --input path/to/upstream/checkpoint --output whisper-base.gguf
```

より詳しい情報 — バックエンド別ガイド、プラットフォーム別チュートリアル
（CLI、Android、iOS、Godot、Unity、Python、server、web）、onnxruntime /
whisper.cpp からの移行ガイドは [`docs/`](docs) にあります。

## C ABI の利用

Vokra は単一の C ヘッダ [`include/vokra.h`](include/vokra.h) を公開して
います（cbindgen で生成、[`scripts/gen-c-abi.sh`](scripts/gen-c-abi.sh) で
再生成可能）。`vokra-capi` クレートをビルドすると共有・静的ライブラリが
生成されます:

```sh
cargo build -p vokra-capi --release
# -> target/release/libvokra.dylib | libvokra.so | vokra.dll  (+ libvokra.a)
```

セッションは GGUF モデルから生成されます。アーキテクチャは
`vokra.model.arch` メタデータから検出され、対応するタスクが自動的に配線
されます（Whisper → ASR、Silero VAD → VAD ストリーム、piper-plus → TTS）。
すべての関数は `vokra_status_t`（`VOKRA_OK` = `0`）を返し、エラー時は
`vokra_last_error()` でスレッドローカルなメッセージを取得できます。
Vokra が確保した出力は対応する `vokra_*_free` / `vokra_*_destroy` で
解放してください。

```c
#include "vokra.h"

vokra_session_t *session = NULL;
if (vokra_session_create_from_file("whisper-base.gguf", &session) != VOKRA_OK) {
    fprintf(stderr, "load failed: %s\n", vokra_last_error());
    return 1;
}

char *text = NULL;
if (vokra_asr_transcribe(session, pcm, num_samples, 16000, &text) == VOKRA_OK) {
    printf("%s\n", text);
    vokra_string_free(text);
}
vokra_session_destroy(session);
```

ヘッダに対してコンパイルし、共有ライブラリをリンクします:

```sh
cc app.c -I include -L target/release -lvokra -Wl,-rpath,target/release -o app
```

エンドツーエンドの実行可能サンプル（ASR / TTS / VAD）は
[`tests/capi/`](tests/capi) にあります。`scripts/run-capi-smoke.sh` で
ビルドと実行が可能です。

## アーキテクチャ概要

**ネイティブ再実装**。対応モデルはすべて、tokenizer・tensor レイアウト・
forward pass・decoder loop・streaming state を含む Rust モジュールで、
上流の `safetensors` または Vokra 向け `GGUF` ファイルを直接消費します。
これにより `torch.onnx.export` の音声モデルにおける慢性的な脆さを回避し、
`onnx` / `protobuf` / `abseil` をランタイムから排除し、バグレポートが
実行可能になります（単一の Rust ファイルであり、グラフエクスポートでは
ありません）。

**ゼロ外部依存**。ルートワークスペースの `Cargo.lock` は first-party
`vokra-*` クレートのみを解決します。この不変条件は CI 実行のたびに
`scripts/check-zero-deps.sh` で検証されます。この条件を破るもの — 8 言語
G2P ポート、Godot GDExtension、HTTP サーバ、ONNX コンバータ — は
`integrations/` の独立サブワークスペースにその crate 独自の `Cargo.lock`
とともに配置されます。

**手書き FFI、EULA 準拠の CUDA**。GPU / NPU バックエンドはバインディング
クレートに依存しません。Metal は生 `objc` ランタイム呼び出しと MSL
コンピュートカーネル。CUDA は開発者インストールされた CUDA に対して
NVIDIA Driver API と NVRTC を実行時に `dlopen` / `LoadLibrary`（何も
同梱しないため NVIDIA CUDA / cuDNN EULA と互換 — [`NOTICE`](NOTICE)
参照）。Vulkan は SPIR-V を事前コンパイルし loader を `dlopen`。
WebGPU は小さな wasm extern-import shim 経由でブラウザと会話します。

**GGUF + `vokra.*` メタデータ**。Weight ファイルは標準 GGUF に Vokra
所有のチャンク群（STFT / mel spec の `vokra.frontend.*`、モデル別
ハイパーパラメータの `vokra.<arch>.*`、量子化ポリシーの
`vokra.quant.*`、ライセンス provenance の `vokra.provenance.*`、
producer identity の `vokra.schema.version` / `vokra.schema.producer`）
を加えたものです。フロントエンド仕様が weight と一緒に運ばれるため、
ランタイムはトレーニング時と bit-exact 一致しない spec を持つ checkpoint
を拒否します — librosa vs torchaudio の Mel フィルタ差分に起因する
無音の drift はありません。

**Compute seam とグラフ実行エンジン**。各モデルは小さな `Compute` seam
（GEMM ホットパス用の per-backend `Cpu` / `Metal` / `Cuda` / `Vulkan`
arm）を通じてバックエンドに到達します。より長寿命なパイプライン — pre-norm
エンコーダスタック、デバイス常駐 KV キャッシュ付き自己回帰デコーダステップ、
コーデックチェーン — はデータ搬送グラフで表現され、中間結果はデバイス常駐
のまま、ホスト↔デバイス readback はステップあたり小さな定数に抑えられます。

**Loud エラー、決して誤った回答を返さない**。バックエンドが演算子を実装
していない場合、呼び出しは明示的な `VokraError::UnsupportedOp` で失敗します。
Vokra は決して別のバックエンドや dtype に silently リルートしません。

## バインディングと統合

- **iOS** — XCFramework + Swift Package
  ([`Package.swift`](Package.swift), [`scripts/build-ios.sh`](scripts/build-ios.sh)
  でビルド): arm64 デバイスと Simulator スライス、静的リンク、
  `DllImport("__Internal")` 互換。
- **Unity** — [`bindings/unity/`](bindings/unity) 配下の UPM パッケージ
  `com.vokra.unity`、[`scripts/build-unity-plugin.sh`](scripts/build-unity-plugin.sh)
  でビルド: IL2CPP セーフなコールバックマーシャリング、Android
  `persistentDataPath` ヘルパー、CUDA ライブラリの誤同梱を防ぐ
  非 NVIDIA-bundle スキャナ (`check-unity-package-no-nvidia.sh`)。
- **Godot** — [`integrations/vokra-godot/`](integrations/vokra-godot) の
  GDExtension、[`scripts/build-godot-gdextension.sh`](scripts/build-godot-gdextension.sh)
  でビルド: 5-target クロスビルドマトリクス（macOS Intel + Apple Silicon、
  Linux x64、Windows MSVC、Android arm64）、AssetLib 形状のリリース
  レイアウト。
- **Python** — [`bindings/python/`](bindings/python) の純粋 `ctypes`
  実装（`pyo3` 非使用）、`cibuildwheel` で PyPI wheel を発行。
- **HTTP サーバ** — [`integrations/vokra-server`](integrations/vokra-server):
  既存クライアントを無改変で置き換えられる 4 種類の互換 API を公開する
  独立ワークスペース。**OpenAI Whisper** (`/v1/audio/transcriptions`)、
  **vLLM** (`/v1/completions`、`/v1/chat/completions`)、
  **piper-plus HTTP** (`/api/tts`)、Home Assistant Voice バックエンド用の
  **Wyoming Protocol**。

## モデル配布

すぐに動かせる GGUF 変換物は Hugging Face 上の
[`vokra`](https://huggingface.co/vokra) 組織で公開しています。公開される
成果物はすべて以下を同梱します:

- モデルカード（メタデータから生成）。
- 上流ライセンス本文を含む `LICENSE`（publish 時に
  [`scripts/publish/fetch_license.sh`](scripts/publish/fetch_license.sh)
  が取得）。
- 上流が attribution を要求する場合の `NOTICE`（例: Mimi は CC-BY-4.0 で
  Kyutai へのクレジットが必要）。
- 上流 URL と再変換手順を記載した `SOURCE.md`。

すべての publish は
[`scripts/publish/publish-one.sh`](scripts/publish/publish-one.sh) を経由し、
fail-closed に設計された 5 段ゲートを通過します:

1. **Catalog reality** — tracked catalog に無いモデルの publish を拒否
   （未レビューのモデルが誤って publish されない）。
2. **Redistributability** — 契約上再配布が禁止されているコーパスを拒否
   （VOICEVOX、CSJ、JSUT、JVS）。
3. **Provenance stamp presence** — `vokra.schema.version` と
   `vokra.schema.producer` チャンクを必須化し、消費者が producer を
   識別可能にします。
4. **Owner sign-off** — [`docs/license-audit.md`](docs/license-audit.md)
   §3.1 の対応行への署名（source-of-truth リンク付き）を必須化。
5. **Non-commercial opt-in** — 研究用途 tier（例: X-Codec 2、CC-BY-NC-4.0）
   には明示的な `--allow-noncommercial` フラグを必須化。

低メモリ `restamp_provenance` 書き換え経路と組み合わせることで、
控えめなハードウェアからでも数 GB の checkpoint publish を日常的に
実行できます。

## piper-plus 統合

[piper-plus](https://github.com/ayutaz/piper-plus) は依頼者作の MIT
ライセンス Piper フォークです（eSpeak-NG 依存を排除した 8 言語 G2P、
MB-iSTFT-VITS2 デコーダ、CUDA / CoreML / DirectML 対応、Unity binding）。
Vokra はこれを標準 TTS レイヤ、かつ Vokra 初のネイティブ実装 TTS モデル
として統合しています:

- MB-iSTFT-VITS2 推論スタック — text encoder、duration predictor、flow、
  MB-iSTFT decoder — は Rust でネイティブに再実装されています。上流の
  ONNX ベース実装の wrap ではなく、Vokra のエンドツーエンド推論経路には
  `onnxruntime` は含まれません。
- Voice モデルはオフラインで GGUF に変換し、ランタイムは GGUF のみを
  ロードします。
- 8 言語 G2P（JA / EN / ZH / ES / FR / PT / SV / KO）は当面 piper-plus
  から流用します。Rust ポートは follow-up 項目です。

## ドキュメント

ユーザ向けドキュメントはすべて [`docs/`](docs) 配下にあります。トップ
レベル文書はすべて英語版 (`.md`) と日本語版 (`.ja.md`) が用意されています。

| ドキュメント | 内容 |
|---|---|
| [`docs/getting-started.md`](docs/getting-started.md) | 5 分クイックスタート |
| [`docs/architecture.md`](docs/architecture.md) | 内部アーキテクチャ、クレート構成、グラフ実行エンジン |
| [`docs/api-reference.md`](docs/api-reference.md) | C ABI + CLI リファレンス |
| [`docs/backend-guide.md`](docs/backend-guide.md) | CPU / Metal / CUDA / Vulkan / WebGPU / CoreML / QNN ガイド |
| [`docs/tutorials/`](docs/tutorials) | プラットフォーム別チュートリアル: CLI、Android、iOS、Godot、Unity、Python、server、web |
| [`docs/migration-guide.md`](docs/migration-guide.md) | onnxruntime / whisper.cpp / piper からの移行 |
| [`docs/license-audit.md`](docs/license-audit.md) | モデルと依存の license 監査 |
| [`docs/legal-compliance.md`](docs/legal-compliance.md) | EU AI Act Article 50、SB 942、ELVIS Act、C2PA |
| [`docs/good-first-tasks.md`](docs/good-first-tasks.md) | コントリビュータ向け入口 |
| [`docs/abi-changelog.md`](docs/abi-changelog.md) | C ABI 変更履歴 |
| [`NOTICE`](NOTICE) | Attribution 要件と bundling ポリシー |

## 関連プロジェクト

- **[piper-plus](https://github.com/ayutaz/piper-plus)** — Vokra が標準
  TTS レイヤとして統合している、依頼者作の MIT ライセンス Piper
  フォーク（上記参照）。

## コントリビューション

コントリビューションを歓迎します。大きめの PR を出す前に、まず issue を
開いてスコープと方針を早めに整合させてください。

- **入り口**:
  [`docs/good-first-tasks.md`](docs/good-first-tasks.md) — file:line
  アンカーや再現コマンド、自分で確認できる受け入れ基準、おおよそのサイズ
  付きの、自己完結タスクを掲載しています。
- **質問・議論**:
  [GitHub issue](https://github.com/ayutaz/vokra/issues) を開いてください。
- **プルリクエスト**: [`CONTRIBUTING.md`](CONTRIBUTING.md) を参照。
  変更はすべて CI 品質ゲート（build / tests / formatting /
  clippy `-D warnings` / zero-dependency 不変条件 / C ABI changelog /
  license audit）付きの PR 経由で入ります。

## 法務・コンプライアンス

- **EU AI Act Article 50** および **California SB 942**: TTS と
  voice-conversion 出力は合成音声とみなされ、開示義務があります。Vokra は
  building block として AudioSeal watermarking と C2PA manifest サポート
  （`c2pa-rs` 経由）を提供します。デプロイ側の開示義務は
  [`docs/legal-compliance.md`](docs/legal-compliance.md) に記載しています。
- **Voice-cloning 分離**: RVC v2、GPT-SoVITS などの voice-conversion
  「trigger」モデルは意図的にこのリポジトリに **含まれていません**。
  Tennessee ELVIS Act（2024-07-01）と連邦 NO FAKES Act のため、別プロジェクト
  `vokra-voiceclone-experimental` に分離しています。ゼロショット TTS 用の
  話者エンベディング（特徴抽出のみ、変換なし）は core に残します。
- **NVIDIA CUDA / cuDNN EULA**: Vokra は NVIDIA ライブラリを一切同梱しません。
  CUDA バックエンドは開発者がインストールしたシステム CUDA を実行時に
  `dlopen` します。[`NOTICE`](NOTICE) と
  [`docs/license-audit.md`](docs/license-audit.md) に記録されています。
- **非商用 weight**: F5-TTS (CC-BY-NC-4.0)、Fish-Speech (CC-BY-NC-SA-4.0)、
  EnCodec (CC-BY-NC-4.0)、X-Codec 2 (CC-BY-NC-4.0) はデフォルトの
  model zoo に含まれません。エンジンは明示的な research flag /
  `--allow-noncommercial` の指定で実行できます。
- **Piper (`OHF-Voice/piper1-gpl`) は非対応** です（GPL-3.0 + eSpeak-NG
  GPL-3.0 の推移的依存）。Vokra が対応する Piper 系統合は
  [piper-plus](https://github.com/ayutaz/piper-plus) のみです。

## ライセンス

Vokra は [Apache License, Version 2.0](LICENSE) の下でライセンスされています。

追加のライセンス通知および配布通知 — 例えばモデル別 attribution
（Mimi の CC-BY-4.0 attribution 義務、BigVGAN の attribution）、
NVIDIA ランタイム非同梱ポリシー — は [`NOTICE`](NOTICE) に記録
されています。
