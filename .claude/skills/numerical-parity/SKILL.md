---
name: numerical-parity
description: Vokra の op / モデルを PyTorch など reference 実装と数値照合する parity テストを追加・更新するときに使う。**reference dumper を書く / parity の許容誤差 (atol/bound) を変更する / parity が落ちた** ときは必ず読む。オフライン reference 生成 → fixtures コミット → Rust parity テストの手順に加え、reference の独立性（自作 mirror は何も検証しない）と許容誤差を緩める前の判断基準を示す。
---

# 数値 parity テストを追加する

reference 実装との数値一致は Vokra の品質背骨（NFR-QL-01）。**CI で Python / onnxruntime は一切実行しない** — reference fixtures はオフラインで生成してコミットする。reference-fixture parity は `parity` job（`cargo test -p vokra-parity`）、GPU backend の parity は `gpu-backends` job と各 backend/models crate（下記「GPU backend parity」）。

## 大原則（捏造厳禁）

- reference の数値は **必ず実際の reference 実装で生成**する。ライブラリが手元に無いなら **数値を発明しない** — `#[ignore]` の shell テストとして比較ロジックだけ書き、fixtures 生成後に有効化する（既存例: `tests/parity/tests/reference_ignored.rs`）。
- 生成は **byte-reproducible**（固定 seed。既存の numpy 系は seed=1234）。版数を pin して再現性を担保。

## 大原則 2: reference は「独立」でなければ何も検証しない

**parity が緑でも、reference が独立でなければ検証はゼロ**。これは仮説ではなく実際に起きた（Kokoro、2026-07-16 / 修正 `92dbc92`）:

> dumper が upstream package ではなく **実装と同じ理解で書かれた再実装**を叩いていた。両者は同じ誤りを共有していたので **9/9 tensor PASS**。しかし実際に合成した音声は **round-trip WER 1.0 = 完全に意味不明**だった。原因は 11 箇所（LeakyReLU slope、ALBERT 層数、decoder 入力、F0/N 配線、duration 式…）。**parity suite は 1 つも捕まえていない。**

したがって dumper を書くときは:

1. **upstream package を import する**（`import kokoro` / `from transformers import X`）。自分で層を書き直して「同じはず」の値を出してはいけない。**それは reference ではなく 2 つ目の実装**。
2. **import できないなら loud abort**。「無ければ再実装に fallback」は最悪の分岐 — 静かに無意味な緑を作る（FR-EX-08 の精神）。
3. dumper header に **何を import して何処に hook したか**を書く（例: `tools/parity/dump_kokoro_reference.py`）。
4. upstream が無い / 使えない場合は **その旨を fixtures の README に明記**し、parity ではなく「self-consistency の pin」だと名乗る。名前を正しく付ければ、後任が過信しない。

**匂いのチェック**: その reference は「自分の実装が間違っていたら**違う**値を出す」か？ 出さないなら reference ではない。

## 大原則 3: 許容誤差は「広げる」方向に動かさない

**測定値が bound を超えたとき、既定の行動は bound を上げることではない。** これも今日踏みかけた（Kokoro pcm、2026-07-19 / 撤回 `1419964`）:

> Linux CI で pcm max が gate を 8.5% 超過。分布を見ると Linux は bulk ではむしろ高精度（mean が低い）で、違いは外れ値だけ — 「プラットフォーム差だから広げてよい」と判断しかけた。しかし **x86 の観測は 1 点しかなく**、その経路の x86 SIMD kernel は実機で走ったことがなかった。「プラットフォーム外れ値」と「局所的な kernel バグ」は集計統計では区別できない。撤回。

判断基準:

- **max ゲートは局所破損**を、**mean ゲートは系統ドリフト**を捕まえる。**max を緩めることは局所バグの検出力を捨てること**。
- 「機構が説明できる」は**仮説**であって診断ではない。**最悪 bin を実際に見に行く**（その位置の値は? 近傍は? 別プラットフォームでも同じ位置か?）。
- 1 プラットフォームの実測から導いた bound は**不完全**だが、それは「もう 1 点で広げてよい」を意味しない。**両方を説明する機構を特定してから**動かす。
- 緩めるより **OPEN として記録して赤のまま残す**方が誠実なことが多い（advisory な suite ならブロッカーですらない）。
- 動かす場合は **理論下限 × 1.5〜2**、rustdoc + ADR + CI YAML に **冗長に**根拠を記録（既存例: Kokoro decoder の branch-cut bound）。

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
- **合成 weight の parity は「構造」しか守らない**。実 checkpoint + 実音声で初めて出る欠陥が実測 9 系統あった（2026-07-16/17 の実 weight 評価: Silero の rolling context 欠落で実音声検出 0、Voxtral converter が weightless GGUF を exit 0 で出力、CosyVoice2 の attention bias 欠落…）。合成 fixtures が緑でも **flip-the-switch を早く回す**。記録: `docs/bench-baselines/m1-real-weight-eval-2026-07-16/`。
- **per-tensor の atol override は「理論下限」を騙りやすい**。Kokoro の `PROSODY_F0_ATOL = 0.05` は「architecturally 到達不可なので honest」と文書化されていたが、実際には **flawed reference の artifact** で、真の upstream に対しては 3.0e-3（default `ATOL = 0.01` の内側）だった。override を足す前に「reference は正しいか」を先に疑う。
