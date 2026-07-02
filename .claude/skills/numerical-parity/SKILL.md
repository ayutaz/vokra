---
name: numerical-parity
description: Vokra の op / モデルを PyTorch など reference 実装と数値照合する parity テストを追加・更新するときに使う。オフライン reference 生成 → fixtures コミット → Rust parity テストの手順と許容誤差、捏造禁止ルールを示す。
---

# 数値 parity テストを追加する

reference 実装との数値一致は Vokra の品質背骨（NFR-QL-01）。**CI は `cargo test -p vokra-parity` のみを走らせ、Python / onnxruntime は一切実行しない** — reference fixtures はオフラインで生成してコミットする。

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

## Kill switch（忘れない）

- **Kill switch I**: v0.1 MVP でモデル6種中 **3種以上** が PyTorch reference 比で **MEL loss / UTMOS 5% 超劣化**なら撤退条件。parity 劣化は品質だけでなく事業判断に直結する。

## 落とし穴

- **STFT ≠ FFT**、**frontend は bit-exact でない**（librosa/torchaudio/TF で Mel フィルタが差異）。frontend 系は `vokra.frontend.*` を検査（レビュアー C 指摘 #1/#2）。
- onnxruntime は内部で FP16→FP32 に cast することがある（piper parity は FP32 native で一致した実績）。reference の内部精度を確認してから許容誤差を決める。
