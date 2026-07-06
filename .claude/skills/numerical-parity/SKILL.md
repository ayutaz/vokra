---
name: numerical-parity
description: Vokra の op / モデルを PyTorch など reference 実装と数値照合する parity テストを追加・更新するときに使う。オフライン reference 生成 → fixtures コミット → Rust parity テストの手順と許容誤差、捏造禁止ルールを示す。
---

# 数値 parity テストを追加する

reference 実装との数値一致は Vokra の品質背骨（NFR-QL-01）。**CI で Python / onnxruntime は一切実行しない** — reference fixtures はオフラインで生成してコミットする。reference-fixture parity は `parity` job（`cargo test -p vokra-parity`）、GPU backend の parity は `gpu-backends` job と各 backend/models crate（下記「GPU backend parity」）。

## 大原則（捏造厳禁）

- reference の数値は **必ず実際の reference 実装で生成**する。ライブラリが手元に無いなら **数値を発明しない** — `#[ignore]` の shell テストとして比較ロジックだけ書き、fixtures 生成後に有効化する（既存例: `tests/parity/tests/reference_ignored.rs`）。
- 生成は **byte-reproducible**（固定 seed。既存の numpy 系は seed=1234）。版数を pin して再現性を担保。

## 手順

1. **reference をオフライン生成**。crate は `tests/parity`（test-only、`publish = false`、`vokra-parity`）。
   - 依存は `tests/parity/parity-requirements.txt` に pin（`numpy>=1.26,<3` が必須。`torch>=2.2` / `librosa>=0.10` / `scipy>=1.11` は optional）。**再生成時は版数を固定**。
   - op 系: `python3 tests/parity/gen_parity_fixtures.py {all|stft|mel|dct}`。
   - モデル系: 各 suite の `gen_reference.py`（例: `tests/parity/silero_vad/`, `tests/parity/piper_plus/`, `tools/parity/dump_whisper_reference.py`）。
2. **fixtures を `tests/parity/<suite>/` にコミット**（`.f32` raw / `.txt` / `manifest.txt` / `README.md`）。各 suite の README に **reference 実装・入力・許容誤差**を明記（NFR-QL-01）。
3. **Rust parity テストを書く**。
   - **FP32: `atol = 0.01`**（`vokra_parity::FP32_ATOL`）。INT8: `atol = 0.05`（量子化パス導入後）。reference は f64、vokra は f32 である点に留意。
   - モデル固有基準（MEL loss / UTMOS / WER / CER 等）は per-suite doc に記述。
4. **検証**: `cargo test -p vokra-parity`（committed fixtures のみで green）。ignore shell は `cargo test -p vokra-parity -- --ignored`。

## GPU backend parity（Metal / CUDA）

GPU backend は **reference fixtures を持たず、CPU backend を oracle にする**（同じ per-(backend, op) kernel の別実装なので、CPU parity が通っていれば CPU が真値）。

- **device-gate（skip、fail しない）**: Metal は `MetalContext::new` / `vokra_metal_probe`、CUDA は device probe で gate。device が無い host は skip（GGUF-gated の model parity と同じ「runner に実機が要る」方針）。CI の `gpu-backends` job は **Metal を Apple-silicon macOS runner で実行**、CUDA は GitHub に NVIDIA GPU が無いので build/lint のみ（実機 parity は **vast.ai RTX 4090**）。
- **許容誤差は CPU parity と同じ FP32 `atol = 0.01`**。`vokra-backend-{metal,cuda}/tests/parity_{metal,cuda}.rs`（GEMM）と `parity_kernels_{metal,cuda}.rs`（gemv/softmax/layer_norm/gelu/conv1d）が GPU 出力を CPU kernel 出力と比較（観測誤差は遥かに tight）。
- **fused op（`Compute::mlp_f32` / `attn_f32` / `encode_prenorm_stack` / decoder-step session）は per-op GPU 経路と bit-identical**（1 GPU submission で中間を device 常駐、readback 削減のみ＝新しい数値ではない）。CPU arm の `mlp_f32` は fusion 前の 3-kernel 列と bit-for-bit 一致（CPU parity 維持）。**causal fused attention は host mask+softmax と IEEE-754 bit-identical**（masked col が `exp(-inf)=0` で寄与ゼロ）。**device KV cache append は host project+concat と 7.15e-7 一致**（M1 実測）。
- **e2e（`vokra-models/tests/parity_whisper.rs`、GGUF-gated）**: encoder / decoder logits は `atol = 0.01` 内、**greedy token 列は CPU と完全一致（`assert_eq`）** を要求（最も強い e2e 判定）。**whisper base full e2e greedy は Metal M1 と CUDA RTX 4090 の双方で CPU と完全一致（5/5 tokens）を実証済み**（Phase 3b、encoder 1.32e-3 / decoder logits 4.29e-5）。

## 品質ゲート（忘れない）

- v0.1 MVP でモデル6種中 **3種以上** が PyTorch reference 比で **MEL loss / UTMOS 5% 超劣化**した場合は品質ゲート違反（要調査・リリースブロック相当）。parity 劣化は音声品質に直結する。

## 落とし穴

- **STFT ≠ FFT**、**frontend は bit-exact でない**（librosa/torchaudio/TF で Mel フィルタが差異）。frontend 系は `vokra.frontend.*` を検査（レビュアー C 指摘 #1/#2）。
- onnxruntime は内部で FP16→FP32 に cast することがある（piper parity は FP32 native で一致した実績）。reference の内部精度を確認してから許容誤差を決める。
