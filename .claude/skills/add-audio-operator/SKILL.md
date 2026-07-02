---
name: add-audio-operator
description: Vokra の audio-dialect オペレータ（STFT/vocoder/flow sampler/codec decode/beam search 等）を新規追加するときに使う。vokra-ops での定義・vokra-backend-cpu のカーネル・属性設計・parity・GPL codec 回避のパターンを示す。
---

# audio-dialect オペレータを追加する

「最新技術のオペレータ化」は Vokra の中核目標。op は**属性で挙動を明示**し、CPU を第一級 backend として必ず動かす。単一事実源は `CLAUDE.md`「音声特化オペレータ」節。

## 1. 定義は `vokra-ops`、カーネルは `vokra-backend-cpu`

- op の型・属性・shape 検査・reference forward を `crates/vokra-ops/` に定義。
- 実カーネル（SIMD）は `crates/vokra-backend-cpu/`。**runtime dispatch**（x86-64: SSE2 baseline→AVX2+FMA 主力、ARM64: NEON baseline→dotprod/i8mm）。RTF 最優先で `unsafe` + SIMD intrinsics を積極使用してよいが、**各 `unsafe` に `// SAFETY:` 必須**（`undocumented_unsafe_blocks = deny`）。公開 API 境界は safe に保つ。`unsafe` 許可は `vokra-ops` / `vokra-backend-cpu` / `vokra-capi` のみ（crate root で `#![allow(unsafe_code)]`）。

## 2. 属性を明示的に設計する（暗黙のデフォルト禁止）

音声 op は「暗黙の前提」が事故る。属性として明示する：

- **`stft` / `istft`**: window(Hann/Hamming/Blackman-Harris/Kaiser)、hop_length、n_fft、center-padding、pad_mode、normalization('forward'/'backward'/'ortho')、causal-mode、real_input(RFFT で2倍高速)。**STFT ≠ FFT** — framing + window + normalization + causal の設計が本質（レビュアー C 指摘 #1）。
- **vocoder**: `snake_activation` は internal_precision 属性（デフォルト FP32、BF16 mantissa 損失が問題）。**Vocos / BigVGAN は INT8 で崩壊 → fp16 必須**、HiFi-GAN は INT8 慎重。
- **flow/diffusion sampler**: cfg_mode ∈ {none, split_batch, dual_forward}、cfg_scale、nfe、schedule ∈ {linear, sway, epss}。
- **codec**: RVQ（paged block size 2-4）と FSQ（単段 GEMV）は別サブグラフ。
- **search**（`beam_search`/`ctc_decode`/`rnnt_decode`）は **host-side runtime 関数**（model graph に埋めない、"contrib op" アンチパターン回避、FR-OP-40）。`crates/vokra-core/src/decode/` に置く。

## 3. GPL / 非商用 codec を持ち込まない

- **soxr / rubberband（GPL）禁止** → resample は speexdsp(BSD) resampler 設計ベースの自前実装。
- FFT は pocketfft(BSD-3) アルゴリズムの自前 Rust 移植（`crates/vokra-ops/src/fft/`、`NOTICE` §3 に既記）。
- **BigVGAN は論文からスクラッチ再実装**（NVIDIA reference は Source Code License-NC で非商用、`NOTICE` §1）。
- **EnCodec weight は CC-BY-NC → 公式 zoo 除外**。DAC/Mimi/WavTokenizer/X-Codec2 が商用 OK 候補。
- 新 codec/依存の追加時は skill `license-audit` を通す。

## 4. frontend 系は bit-exact

- Mel フィルタは librosa/torchaudio/TF で bit-exact でない。frontend op は `vokra.frontend.*` metadata を検査し、不一致なら warn/fail（レビュアー C 指摘 #2）。Slaney/HTK 両対応。

## 5. parity と検証

→ skill `numerical-parity`（torch / scipy / librosa reference と照合、fixtures はオフライン生成をコミット）。最後に：

```
cargo test --workspace
cargo clippy --all-targets -- -D warnings
cargo fmt --all -- --check
bash scripts/check-zero-deps.sh
```
