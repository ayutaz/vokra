---
name: add-speech-model
description: Vokra に新しい音声モデル（TTS / ASR / S2S / VC / Speaker-ID / VAD / 音楽生成 / 音声分離 / 音声 LLM）対応を追加するときに使う。native 自前再実装・GGUF 変換・数値 parity・ライセンス/法務/model-zoo ゲートまでの全手順とレッドラインを示す。
---

# 音声モデルを Vokra に追加する

新規モデル対応 PR の標準手順。**設計レッドライン（下記）を跨ぐ実装は品質に関わらず却下**（CONTRIBUTING.md §5）。基本方針は `AGENTS.md` と本skillに従う。

## 0. 事前判断（着手前に必ず）

- **スコープ判定**: 音声 AI 全域が in-scope（2026-07-30 依頼者 override、旧「TTS/ASR/VAD/Speaker-ID のみ」の絞りは廃止）。**新たに含まれる**: 音楽生成（MusicGen / Stable Audio / ACE-Step 等）／音声分離（SepFormer / Demucs / TIGER 等）／音声 LLM（Voxtral / Qwen2-Audio / Moshi 等）。**out-of-scope 継続**: 汎用 LLM、CV、multimodal（vision 側）— [[project-scope-expansion-2026-07-30]]。
- **ライセンス audit を先に通す** → skill `license-audit`。weight が **CC-BY-NC / CC-BY-NC-SA / 学習データ権利不明**なら公式 model zoo から除外し、engine 対応のみ・research flag 分離（例: F5-TTS, Fish-Speech は weight 非配布）。
- 用途が **voice cloning（RVC / VC / speaker cloning の trigger 側）** なら core に入れない → 別リポジトリ `vokra-voiceclone-experimental`（境界の根拠: `docs/legal-compliance.md` §3/§4、および `docs/system-requirements.md` FR-CP-04）。**speaker embedding 抽出は core に残す**（zero-shot TTS 必須）。voice-clone の共通 op（F0 抽出等）は core に残し、trigger model のみ別リポへ。

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
- 音声固有 metadata は **`vokra.*` prefix の独自 chunk** で焼き込む（llama.cpp 本体との命名衝突回避。仕様は `docs/design/vokra-gguf-chunks.md`、キー定数は `crates/vokra-core/src/gguf/chunks.rs` 等のコードをSoTとする）。frontend を持つモデルは `vokra.frontend.*`（n_fft/hop/win_length/window_type/mel_norm/htk_mode/fmin/fmax/n_mels/pad_mode/sample_rate 等）を必須で書く（bit-exact 再現、レビュアー C 指摘 #2）。

### 2.1 事前 merge が要る checkpoint 形状

以下は **vokra-cli convert に直渡しできない** — Python sidecar で事前処理してから渡す。

- **sharded safetensors（`model.safetensors.index.json` + 複数 `model-*-of-*.safetensors`）**: `vokra-cli` は直渡し不可（"safetensors buffer truncated" で落ちる、例外は Voxtral の streaming reader だけ）。→ `tools/parity/<slug>_prepare_checkpoint.py` を書いて事前 merge。既存例は `kokoro_prepare_checkpoint.py` / `nemo_pt_to_safetensors.py`。**int64/f64/bool は f32/f16 に strip**（GGUF writer が受けない dtype を潰しておく）。[[project-vokra-cli-sharded-safetensors]]。
- **tied embedding（shared tensor）**: `safetensors.torch.save_file` は複数 name が同一 `data_ptr` を指すと `RuntimeError`。Bark / XTTS-v2 / MOSS 変種の MLM head や LM head が該当。→ dedup が必須:
  ```python
  seen: dict[int, str] = {}
  shared_pairs: list[tuple[str, str]] = []
  for n, t in list(kept.items()):
      ptr = t.data_ptr()
      if ptr in seen:
          shared_pairs.append((seen[ptr], n))
          kept[n] = t.clone().contiguous()  # 別領域にコピーして collision 解消
      else:
          seen[ptr] = n
  # shared_pairs は shared_pairs.json に audit trail として吐く（後で復元ロジックが要る）
  ```
  [[reference-safetensors-shared-tensor-dedup]]。
- **5D 以上の tensor**: GGUF writer は現状 4D まで（>4D は `"too many dimensions: 5"` で hard-error）。Qwen2.5-Omni 系 multimodal adapter が該当し publish blocked。回避 = writer 拡張 or `reshape(5D → 4D + metadata)`、判断は M6 investigation phase。着手前に上流 tensor shape を `uv run --project tools/parity python -c "import safetensors; ..."` で確認して 5D を含むなら **converter に着手しない**。[[project-gguf-5d-tensor-limit]]。

### 2.2 合計 2 GB 以上のモデル artefact は M1 iMac で処理しない

- 依頼者機（M1 iMac 16 GB）では、checkpoint / GGUF / shard 群の**合計が 2 GB 以上**なら convert・実 checkpoint 検証・publish を行わない。実測で動いたモデルを例外扱いせず、skill `vast-ai-workflow` で VAST へ送る。
- shard 単体ではなく対象ディレクトリ内の合計で判定する。Voxtral-Small-24B 48 GB では swap 40 GB 到達、ローカル `vokra-models` Cargo では macOS 再起動の実績がある。
- 既存 GGUF の provenance 差替のみなら `restamp_provenance`（mmap 読取 + `GgufStreamWriter` で tensor コピーせず metadata だけ差替、8.7 GB Voxtral を M1 16 GB で peak footprint 6.4 MB で実測）→ skill `publish-model-to-hf` §restamp。[[project-restamp-provenance]]。

## 3. 新規 op が要るか（gap analysis）

- 必要 op が既存（`vokra-ops` / `vokra-backend-cpu`）で揃うか確認。足りなければ skill `add-audio-operator`。Whisper base は gap ゼロだった（`whisper/mod.rs` の inventory 表を参照）。

## 4. 数値 parity（必須）

→ skill `numerical-parity`。PyTorch/onnxruntime reference と MEL loss / UTMOS / WER 等で照合。**モデル6種中3種以上で 5% 超劣化は品質ゲート違反（要調査・リリースブロック相当）**。fixtures は必ずオフライン生成・実データをコミット（捏造厳禁）。

## 5. TTS / VC は法務チェックリスト

- `docs/legal-compliance.md` を通す（EU AI Act Article 50 / California SB 942）。**watermark / C2PA 埋め込み（FR-CP-01 AudioSeal / FR-CP-02 C2PA）は 2026-07-04 依頼者ドロップで未実装**: `vokra-core` の `WatermarkConfig` は config 面のみで、`backend_status` は常に `Deferred`（埋め込み backend 未配線 — 偽の marker を付けない方針）。model-zoo 可否・weight license は下記 compliance gate（→ skill `license-audit`）で runtime 強制する。

## 6. ドキュメント更新（同一 PR 内）

- `docs/license-audit.md` に行追加（code/weight ライセンス・商用可否・学習データ由来）。
- attribution / 配布条件があれば `NOTICE` に追記（例: Mimi は CC-BY 4.0 で credit 要）。
- 対応モデルのstatusは `docs/license-audit.md` §3.1、現行M5/mac handoff（`docs/handoff/mac-cpu-metal-full-coverage-2026-08-28.md`）、および `scripts/publish/check-catalog-reality.sh` の実測を突合して更新。
- 調査値・レイテンシ・パラメータ数は **出典必須**（ハルシネーション厳禁）。不明なら `docs/_research/*.md` を読み返す。

## 7. 検証してコミット

```bash
# ローカル
cargo fmt --all -- --check
bash scripts/check-forbidden-symbols.sh
bash scripts/check-zero-deps.sh

# VAST（workspace / vokra-models Cargo）
cargo test --workspace
cargo clippy --all-targets -- -D warnings
cargo deny check licenses advisories bans
```

CONTRIBUTING.md §4（Adding support for a new model）のチェックリストと突き合わせて漏れがないか最終確認する。
