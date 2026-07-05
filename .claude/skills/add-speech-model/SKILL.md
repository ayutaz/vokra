---
name: add-speech-model
description: Vokra に新しい音声モデル（TTS / ASR / S2S / VC / Speaker-ID / VAD）対応を追加するときに使う。native 自前再実装・GGUF 変換・数値 parity・ライセンス/法務/model-zoo ゲートまでの全手順とレッドラインを示す。
---

# 音声モデルを Vokra に追加する

新規モデル対応 PR の標準手順。**設計レッドライン（下記）を跨ぐ実装は品質に関わらず却下**（CONTRIBUTING.md §5）。単一事実源は `CLAUDE.md`。

## 0. 事前判断（着手前に必ず）

- **ライセンス audit を先に通す** → skill `license-audit`。weight が **CC-BY-NC / CC-BY-NC-SA / 学習データ権利不明**なら公式 model zoo から除外し、engine 対応のみ・research flag 分離（例: F5-TTS, Fish-Speech は weight 非配布）。
- 用途が **voice cloning（RVC / VC / speaker cloning）** なら core に入れない → 別リポジトリ `vokra-voiceclone-experimental`（ELVIS Act / NO FAKES Act、CLAUDE.md 設計判断 #8）。**speaker embedding 抽出は core に残す**（zero-shot TTS 必須）。

## 1. native 自前再実装（whisper.cpp 型）

- モデル定義を Rust で自前実装し、**上流の safetensors checkpoint のみ**を使う（`torch.onnx.export` の dynamo/scriptmodule 分裂に耐性）。既存例: `crates/vokra-models/src/whisper/`（base〜large-v3）, `.../piper_plus/`, `.../silero_vad/`, `.../speaker/`（CAM++ speaker encoder）。
- **GPU backend 対応は `Compute` seam 経由**（`vokra-models/src/compute.rs`）: モデルの hot op（GEMM/GEMV/softmax/layer_norm/gelu/conv1d）を CPU kernel 直呼びでなく `Compute` に通すと、feature=metal/cuda build 時に Metal/CUDA へ swap できる（既定 CPU、非対応 backend は明示 `UnsupportedOp` = silent CPU fallback 禁止 FR-EX-08）。Whisper / piper-plus / CAM++ は配線済み。GPU parity は device-gated で CPU を oracle にする（→ skill `numerical-parity`）。
- **runtime に ONNX を絶対に入れない**（FR-LD-05、恒久）。onnxruntime / onnx / protobuf / prost / ort への依存禁止（deny.toml で ban 済み）。
- **piper-plus 系は native 自前実装**（wrap 廃止・依頼者決定 2026-07-02）。G2P（8 言語）のみ当面 piper-plus 実装を流用。
- **eSpeak-NG 禁止**（GPL-3.0）。G2P は piper-plus 独自 or IPA 辞書ベース。
- ハイパラは **`vokra.*` GGUF metadata から読む**（ハードコード禁止、FR-LD-02 / FR-MD-02）。
- Silero VAD のような recurrent/学習済み前処理を持つモデルは **1:1 保存の専用 subgraph**にし、汎用 audio-dialect op に落とさない（FR-LD-06 / NFR-QL-05）。

## 2. GGUF オフライン変換（`vokra-convert`）

- 上流 checkpoint → GGUF 変換は `crates/vokra-convert/` に追加。ONNX / protobuf を扱うのはこの**オフラインツールのみ**（runtime 側には持ち込まない）。
- 音声固有 metadata は **`vokra.*` prefix の独自 chunk** で焼き込む（llama.cpp 本体との命名衝突回避、CLAUDE.md 設計判断 #3）。frontend を持つモデルは `vokra.frontend.*`（n_fft/hop/win_length/window_type/mel_norm/htk_mode/fmin/fmax/n_mels/pad_mode/sample_rate 等）を必須で書く（bit-exact 再現、レビュアー C 指摘 #2）。

## 3. 新規 op が要るか（gap analysis）

- 必要 op が既存（`vokra-ops` / `vokra-backend-cpu`）で揃うか確認。足りなければ skill `add-audio-operator`。Whisper base は gap ゼロだった（`whisper/mod.rs` の inventory 表を参照）。

## 4. 数値 parity（必須）

→ skill `numerical-parity`。PyTorch/onnxruntime reference と MEL loss / UTMOS / WER 等で照合。**モデル6種中3種以上で 5% 超劣化は品質ゲート違反（要調査・リリースブロック相当）**。fixtures は必ずオフライン生成・実データをコミット（捏造厳禁）。

## 5. TTS / VC は法務チェックリスト

- `docs/legal-compliance.md` を通す（EU AI Act Article 50 / California SB 942）。**watermark / C2PA 埋め込み（FR-CP-01 AudioSeal / FR-CP-02 C2PA）は 2026-07-04 依頼者ドロップで未実装**: `vokra-core` の `WatermarkConfig` は config 面のみで、`backend_status` は常に `Deferred`（埋め込み backend 未配線 — 偽の marker を付けない方針）。model-zoo 可否・weight license は下記 compliance gate（→ skill `license-audit`）で runtime 強制する。

## 6. ドキュメント更新（同一 PR 内）

- `docs/license-audit.md` に行追加（code/weight ライセンス・商用可否・学習データ由来）。
- attribution / 配布条件があれば `NOTICE` に追記（例: Mimi は CC-BY 4.0 で credit 要）。
- 対応表（CLAUDE.md の「対応モデル」）と対応時期を更新。
- 調査値・レイテンシ・パラメータ数は **出典必須**（ハルシネーション厳禁）。不明なら `docs/_research/*.md` を読み返す。

## 7. 検証してコミット

```
cargo test --workspace
cargo clippy --all-targets -- -D warnings
cargo fmt --all -- --check
bash scripts/check-forbidden-symbols.sh
bash scripts/check-zero-deps.sh
cargo deny check licenses advisories bans
```

CONTRIBUTING.md §4（Adding support for a new model）のチェックリストと突き合わせて漏れがないか最終確認する。
