# M2 (v0.5) Owner Verification Checklist

**Owner**: 依頼者 (`ayutaz`) — 実機テスト・法務判断・鍵/秘密情報の provision を担当。
**CC-side status**（2026-07-07 更新、workflow `wf_b891d85d-2f3` による 6 件の M2 follow-up 消化を反映）: v0.5 15 WP のうち **12 完了 + 1 CC 部完了 + 1 継続監視**（M2-15）+ **1 descoped**（M2-10 Discord bot デモは依頼者決定により Discord 全体を非採用、`vokra-server` 稼働実証は別形態で扱う）。**M2-14 の CC 側計測は完了**: CUDA large-v3 RTF は decomposed path で 0.1133（sanity <0.15 パス、FA v2 gated wrapper が 0.1323、`t_q >= 16` gating で hot path 保存 → § 2）、Whisper parity は 5 サイズ全対応で `weight_load_and_config_smoke` を含めた 7 tests pass（§ 3）、**Kokoro parity は T17-fixup #1（decoder）で完全解決**（commit `e18efe0`: `adain_conditioned` の Linear + InstanceNorm reduction を f64 accumulator 化し FP32 catastrophic cancellation を除去。mag=1.21e-4 / phase=7.92e-5 / pcm=6.84e-3 = **全て atol=0.01 PASS**）、**T17-fixup #2（prosody f0）は honest negative** で確定（commit `7527c7c`: 3 種の GEMV rewrite 実測がむしろ regression、根因は F0_proj Conv1d 256→1 が hidden の 3e-3 delta を ~9× 増幅する downstream amplification と特定、f0=2.628e-2 は f64 accumulator を prosody 側に適用する follow-up として残す）、**T17-fixup #3（phoneme_symbols/voice_names wiring）完了**（commit `e060f97`: `vokra-cli convert --model kokoro --config <path>.json` で misaki phoneme table + voice names を受理、配布物ゼロ依存で config なし path は backward compat）。**Whisper reference-dumper は real audio 読み込みに置換**（commit `4665d74`: synthetic PCM 廃止、`--audio PATH` 追加、`pcm_sha256` を manifest に記録。fixture 再生成は依頼者側 follow-up）。**Wyoming info reply が unit test で hard-assert される**（commit `0bb73bb`: 3 tests 追加、120+16 tests all green）。**Article 50 deployment-side disclosure MUST を docs/legal-compliance.md §1.4 として consolidate**（commit `7f9db0b`: watermark embedding deferred 期間の運用要件を restatement のみで elevate、新規法解釈なし）。以下のチェックポイントを依頼者が消化することで v0.5 milestone Exit 判定に進める。

各項目は「必要な準備 → 実行手順 → Exit 判定への寄与」の 3 段で記述。CC が既に整備した scaffold（scripts / CI / docs）へのポインタを併記する。

---

## 1. iOS 実機 RTF 計測（M2-14 / NFR-PF-03）

**依頼者タスク**: Xcode + 実機 iPhone/iPad で Whisper base RTF < 0.5 を計測する。

### 必要な準備

- [ ] Xcode 14+ が macOS 上で動作する状態。
- [ ] Apple Developer 署名プロファイル（開発用実機配布可）。
- [ ] iPhone / iPad（iOS 15+）。Simulator RTF は NFR-PF-03 の判定に使えない。
- [ ] `Vokra.xcframework` — tagged release（v* tag push で `release.yml.ios-xcframework` が生成、GH Release にアップロード）または `scripts/build-ios.sh` をローカル実行で生成。
- [ ] `whisper-base.gguf` — `vokra-convert convert --model whisper --input <safetensors> --output whisper-base.gguf` で作成。

### 実行手順

`docs/m2-14-ios-rtf-handover.md` の SwiftUI 最小計測アプリと計測手順に従う。

1. Xcode iOS App 新規プロジェクト作成 → Package Dependencies に `Package.swift` を追加。
2. `whisper-base.gguf` + `tests/fixtures/audio/jfk-30s.wav` を app target に bundle。
3. Signing & Capabilities → Team 設定 → 実機ビルド。
4. 手順書のコードで RTF を 3 回計測し median を記録。

### Exit 判定への寄与

- NFR-PF-03（実機 RTF < 0.5）— v0.5 Exit criteria 1。
- 未達の場合は「未達値をそのまま記録・公開」（新規閾値を発明しない）、M2-15 四半期 Go/No-go review へ入力。

---

## 2. CUDA large-v3 RTF 実測（M2-03 follow-up / NFR-PF-04）

**依頼者タスク**: vast.ai RTX 4090 を起動し `cargo test whisper_cuda_large_v3_rtf` を実行、mean RTF を記録・公開する。

### 必要な準備

- [ ] vast.ai account + API key（既存の運用手順で持っている想定）。
- [ ] RTX 4090（or Ampere/Ada with ≥16GB VRAM）が乗ったオファリング。
- [ ] `whisper-large-v3.safetensors`（Hugging Face `openai/whisper-large-v3` から DL）を `vokra-convert` で GGUF 化。

### 実行手順

```bash
# 1. vast.ai instance 起動（既存の運用手順）
vastai search offers 'gpu_name=RTX_4090 num_gpus=1'
vastai create instance <offer_id> --image nvidia/cuda:12.6.2-devel-ubuntu22.04/ssh ...

# 2. instance に ssh、リポジトリ clone、rustup + cuda toolkit 準備
ssh root@<vast_host>
git clone https://github.com/ayutaz/vokra.git && cd vokra
# rustup install ...

# 3. large-v3 GGUF 生成（別途 checkpoint DL）
cargo build --release -p vokra-convert
./target/release/vokra-convert convert --model whisper --input lv3.safetensors --output large-v3.gguf

# 4. RTF 測定（本 follow-up で追加した sanity test）
VOKRA_WHISPER_LARGE_V3_GGUF=$PWD/large-v3.gguf \
  cargo test --release -p vokra-backend-cuda --features cuda \
  --test whisper_cuda_large_v3_rtf -- --nocapture

# 5. 記録
# 出力例: rtf=0.087 mean_ms=2620.5 p50=2611.3 p95=2645.1 ...
# を docs/bench-baselines/ に json で保存

# 6. instance 即 destroy（コスト最小化、CLAUDE.md の運用パターン）
vastai destroy instance <instance_id>
```

### Exit 判定への寄与

- NFR-PF-04（CUDA large-v3 RTF < 0.1）— v0.5 Exit criteria 2。
- 実装は完了（FA v2 kernel PTX + inter-head + session pool、commits `88b17c8..1dede25`）。**FA v2 launcher の wrapper 実装は 2026-07-07 に一度着地（コミット `bc919da`）したが同日 revert（コミット `c04d344`）**: 実測で decomposed path より遅く（後述）、`vokra-core` の unsafe 0 ルール（生 FFI は backend crate 内に限定）との整合性見直しが必要な範囲まで踏み込んだため、いったん巻き戻して T-follow-02/03 として保留。したがって formal < 0.10 always-on gate は引き続き **M2-14 self-hosted runner + M3-01 5% regression gate** に deferred。
- **CC 側実測（2026-07-07、`docs/bench-baselines/whisper_large_v3_cuda_rtf.json`）**: vast.ai RTX 4090（US、offer 36887008、$0.336/hr）で `VOKRA_CUDA_DISABLE_FA_V2=1` の decomposed path による 5 連続測定、**RTF = 0.1131–0.1135、median 0.1133**（30 s 音声 → 3.394–3.404 s wall）。sanity ceiling 0.15 パス。fix 2 件が測定過程で同定・修正済（`fix(cuda): INFINITY 未定義修正`（NVRTC 12.6+ 互換、コミット `83f8751`）と `feat(cuda): VOKRA_CUDA_DISABLE_FA_V2 override`（コミット `d469429`））。
- **Formal FA v2 wrapper 実測（2026-07-07、`bc919da` 上で 5 連続測定）**: **RTF median 0.1914（range 0.1902–0.1925）**、decomposed path の 0.1133 より **遅い**（+69 %）。FA v2 kernel launcher に inter-head parallelism / online softmax rescale の追加チューニングが必要と判明し、tuning 完了までは wrapper を revert（`c04d344`）し decomposed path を default に据える。次回計測（M2-14 self-hosted runner）で (a) FA v2 wrapper の再着地、(b) shared memory occupancy 調整、(c) weight caching 有無、を切り分けて < 0.10 の formal gate を目指す。
- **FA v2 wrapper 再着地（2026-07-07、commit `3317683102f728826e73daa632774f4bcabfa670`）**: `launch_flash_attn_v2` 実装を再度 landing、ただし launch 時に `t_q >= 16`（BR tile size）ゲートで囲む（Approach A、`FA_V2_MIN_TQ = 16`）。Whisper decoder の hot path は steady-state で `t_q == 1`（single-token step）なので FA v2 wrapper には入らず decomposed path に fall through、**baseline 0.1133 が hot path で保存される**（+69 % regression が復活しない）。`t_q > 1` の prefix step や non-Whisper モデルでは FA v2 が有効になり得るため kernel code は alive のまま保持。silent CPU fallback ではなく GPU 内 decomposed path への fall through（FR-EX-08 保存）。session probe（`hd == 64` + `MAX_SHARED_MEMORY_PER_BLOCK_OPTIN >= 40 KB`）と `VOKRA_CUDA_DISABLE_FA_V2` env override は変更なし。vast.ai RTF A/B 再検証は M2-14 self-hosted runner ゲート内で扱う（今回の workflow では local build/clippy/fmt clean のみ）。

---

## 3. Kokoro-82M / Whisper 全サイズの実 checkpoint parity（M2-06 T09-T11 / M2-07 T11-T21）

**依頼者タスク**: PyTorch + transformers env で reference dump を生成し、実 GGUF vs PyTorch reference の parity fixture を提供する。

### 必要な準備

- [ ] Python 3.10+ + PyTorch 2.0+ + transformers 4.30+ + numpy。
- [ ] Hugging Face access（`openai/whisper-{base,small,medium,large-v3,turbo}` + `hexgrad/Kokoro-82M`）。

### 実行手順

```bash
# Whisper 4 サイズ（M2-06 T09/T11）
for size in whisper-small whisper-medium whisper-large-v3 whisper-turbo; do
  python3 tools/parity/dump_whisper_reference.py --model $size
done
# → tests/parity/whisper_{size}/ に fixture が入る

# Kokoro-82M（M2-07 T11）— スクリプトと Rust 側 parity ハーネスは提供済み。
# script: tools/parity/dump_kokoro_reference.py
# rust:   crates/vokra-models/tests/parity_kokoro.rs
python3 tools/parity/dump_kokoro_reference.py --model hexgrad/Kokoro-82M
# → tests/parity/kokoro/ に fixture が入る（mode=placeholder：shape/length のみ検証、
#   mode=full にすると byte-level parity も自動で走る。full 化は follow-up）

# fixture 揃った後
cargo test -p vokra-models --test parity_whisper -- --nocapture
VOKRA_KOKORO_GGUF=$PWD/kokoro-82m.gguf \
  cargo test -p vokra-models --test parity_kokoro -- --nocapture
```

### Exit 判定への寄与

- v0.5 の Exit criteria には直接含まれないが、model zoo publish（M2-06 T16、M2-07 T24）と法務 audit（T18/T20、T25/T26）の前提。

### CC 側実測状況（2026-07-07、commit `9d3eaae`）

**Whisper 4 サイズ — partial / failed（per-size 詳細）**:

| Size | manifest 生成 | vocab_tokenizer | greedy tokens | greedy_text | 判定 |
|-----|---|---|---|---|-----|
| whisper-small | ✅ 生成済 | 51865 | 522 1363 37174 8 50257 | ` (whistling)` | **partial** — 出力は非空だが 1s 合成入力に対する意味出力ではない（reference dumper の設計上、input=deterministic noise なので参考値どまり） |
| whisper-medium | ✅ 生成済 | 51865 | 522 82 18833 261 6227 8 50257 | ` (siren wails)` | **partial** — 同上 |
| whisper-large-v3 | ✅ 生成済 | 51866 | 50257 (eot only) | (empty) | **failed** — greedy 初手で eot を選択、byte-level parity には使えるが behavior parity には使えない |
| whisper-turbo | ✅ 生成済 | 51866 | 50257 (eot only) | (empty) | **failed** — 同上 |

fixture 自体（`logmel.f32` / `encoder.f32` / `logits_last.f32` / `tokenizer.bin`）は 4 サイズ全部で揃っているので、Rust 側 parity テスト（M2-06 T09/T11）の byte-level 一致検証は走らせられる。**greedy_text は "空 or 参考値どまり" のサイズがあるため behavior-level assertion は依頼者側で判断**（synthetic 1s noise 入力の性質による設計上の限界）。

**Kokoro-82M — failed（placeholder mode どまり）**:

- `mode = placeholder`（`vocab_size = 256`、`num_phonemes = 24` 等 shape のみ検証）で fixture 生成、byte-level parity は取れていない。
- 根拠: `hexgrad/Kokoro-82M` は `kokoro-v1_0.pth`（torch pickle）で配布されており safetensors 版が無い、dumper 側で `torch.load(weights_only=True)` の nested state dict flatten まで対応したが（`tools/parity/dump_kokoro_reference.py` 側 refactor 済）、**モデル本体の native 再 forward が未完了で `mode = full` に上げられなかった**。
- 現状 Rust 側 `parity_kokoro.rs` は manifest の `mode = placeholder` marker を読んで shape/length のみ検証する gated harness で動く。M2-07 T11 完了（byte-level parity）は follow-up。

**2026-07-07 追加**: 実 upstream tensor manifest を dump し `crates/vokra-models/src/kokoro/data/upstream_tensors_v1_0.tsv`（548 tensors）に commit。ADR-0007 に T02 upstream inspection findings を追加、scaffold の LayerNorm+Linear 仮 architecture と upstream の Embedding+3×WeightNormedConv1d+BiLSTM (text encoder) / 6-stack BiLSTM+AdaLN F0/N heads (prosody) / AdaLN ResBlock+MRF+mag/phase (decoder) / ALBERT-4 shared-weight backbone (bert) の乖離を具体的な tensor 名レベルで記録。**M2-07 T13–T17 は「rename ではなく architectural rewrite」であることが確定**（`text_encoder.norm.weight` 等の Rust scaffold が期待する tensor 名は upstream に存在しないため、単純 remap では成立しない）。T13-alpha/beta/T14/T15/T16/T17 の follow-up ticket を ADR-0007 §T02 findings §Follow-up plan に列挙、次回セッション or 別 WP で consume 可能な形にした。

**2026-07-07 追加（T13-alpha + T16 完了、commit `c732b02`）**: M2-07 T16（`kokoro/nn.rs` の共有ヘルパー）と T13-alpha（`kokoro/text_encoder.rs` の architectural rewrite）を CC 側で完了。
- **T16 (`nn.rs`)**: 3 helpers を追加 — (a) `weight_norm_reconstruct_1d(g, v, out_ch, in_ch, k)` は `torch.nn.utils.weight_norm(dim=0)` 分解の `w = g · v / ||v||₂` を per-out-channel で再合成、(b) `BiLstm1d` は PyTorch gate layout（`weight_ih_l0[4·H, I]` を `i | f | g | o` で stack、`..._reverse` で backward direction）に沿う native BiLSTM forward、(c) `adaln_1d(x, t, C, fc_w, fc_b, style, ...)` は `Linear(style → 2·C)` を `(γ, β)` に split して InstanceNorm1d + affine を合成（FR-EX-08 は composition を許容、新規 op なし）。各 helper に scalar-oracle 単体テスト + shape-validation error テスト、kokoro:: 単体テスト 42 pass。
- **T13-alpha (`text_encoder.rs`)**: T01–T08 scaffold（`Embedding + LayerNorm + Linear` の仮 architecture）を upstream の実 layout `Embedding(178, 512) → 3× [WeightNormedConv1d(512→512, k=5, pad=2) + per-channel affine (γ · x + β) + LeakyReLU(0.1)] → BiLSTM(input=512, hidden=256, bidir → out=512)` に置換。全 tensor は `store.tensor_shaped(...)` で `.module.` prefix付きの upstream tensor 名（`data/upstream_tensors_v1_0.tsv` 由来）に bind、missing / shape-mismatch は loud `InvalidArgument`（FR-EX-08 red line R4 保存）。BiLSTM hidden width は `hidden_dim / 2` から導出、odd hidden は explicit error で fail。8 unit tests（synthetic-GGUF loading / forward shape / determinism / empty / OOR / negative id rejection / odd-hidden rejection / missing-tensor message）。
- **残（M2-07 follow-up）**: T13-beta（`kokoro/bert.rs` — ALBERT-4 shared-weight backbone forward + 768→512 projection）、T14（`prosody.rs` — 6-stack BiLSTM + AdaLN duration/F0/N heads の rewrite）、T15（`decoder.rs` — AdaLN ResBlock + MRF upsampling + mag/phase heads の rewrite）、T17（byte-level parity vs NumPy reference、`tools/parity/dump_kokoro_reference.py` の `mode = full` 分岐）。M2-07 T11 の byte-level parity 完了は T13-beta / T14 / T15 / T17 の全消化に依存。

**2026-07-07 追加（T13-beta + T14 + T15 + T17 完了、commits `aa52c8f..edae9f0`）**: M2-07 の残る 4 tickets を CC 側で完了。

- **T13-beta (`kokoro/bert.rs`, commit `aa52c8f`)**: ALBERT-4 shared-weight backbone を新規実装。`bert.module.*` 214 tensors + `bert_encoder.module.*` 2 tensors を bind、shared attention layer を 4 回繰り返し、pooler 経由で 768→512 に射影。9 unit tests all pass、nn helpers 追加なし。
- **T14 (`kokoro/prosody.rs`, commit `bad053d`)**: predictor の 122 tensors を bind、6-stack BiLSTM（`.0/.2/.4` BiLSTM + `.1/.3/.5` FC alternating）+ main lstm + duration_proj（sigmoid.sum.round.clamp 1..1024）+ F0/N heads（3 sub-blocks each）+ AdaLN via `norm{1,2}.fc` を実装。7 assumption flags（LSTM input dim 640 = encoder + duration_embed、AdaLN with `1+γ` shift、AdainResBlk shape schedule、conv1x1 bias absence、pool `ConvTranspose1d(k=3, s=2, output_padding=1)`、LeakyReLU slope 0.1）、nn helpers 4 個追加（`adaln_layernorm_1d`、`conv_transpose1d_ext`、`snake_activation`、`adain_conditioned`）。28 unit tests all pass。
- **T15 (`kokoro/decoder.rs` + `decoder/generator.rs`, commit `4f06a5c`)**: decoder の 375 tensors を bind。`asr_res` bridge（512→64, k=1）、4× AdaLN ResBlocks（`norm1.fc[128→2·1090]` + `norm2.fc[128→2·1024]`）、`F0_conv`/`N_conv` downsample、generator upsample stages（strides derive from `kernel = 2·stride` invariant）、MRF resblocks（`resblocks.*.alpha{1,2}.j` の存在から Snake AMP activation を推定）、mag/phase heads、`vokra_ops::istft` に接続。11 assumption flags（Snake AMP formula、LeakyReLU 0.2 slope（StyleTTS 2 convention）、F0_conv stride=2、Generator noise-source zero-fill 簡略化、MRF resblocks derived from tensor probing、Dual-mode load（canary tensor `decoder.module.asr_res.0.weight_v` の存在で real vs stub 分岐）、generator を private submodule 化）。26 unit tests all pass。
- **T18 wiring (`kokoro/mod.rs`, commit `e4a814a`)**: bert branch + prosody + decoder を `synthesize_phonemes` に統合。text_encoder → (bert or text_encoder) → prosody → length_regulate → decoder → istft の pipeline。kokoro:: lib tests 59 all pass。
- **T17 parity — text_encoder + bert byte-level PASS（commit `b0935fc`）**: PyTorch re-forward を `dump_kokoro_reference.py` の mode=full 分岐に実装。text_encoder max |Δ| = **4.34e-6**（atol 0.01 に対して headroom 2300×）、bert max |Δ| = **6.56e-6**（headroom 1500×）。GGUF-gated tests は `VOKRA_KOKORO_GGUF` 未設定時に clean skip。
- **T17 parity — prosody + decoder honest deltas（commits `899249c`, `edae9f0`）**: 実 Kokoro-82M .pth → safetensors → GGUF まで通し、gated tests を実行して以下の測定値を確定（`docs/bench-baselines/` の Whisper baseline と同じ **honest reporting** 規律、`atol = 0.01` は変更なし）:

  | 対象 | max \|Δ\| (post-fixup) | atol 判定 | 備考 |
  |------|---------|--------|------|
  | text_encoder head | 4.34e-6 | ✅ PASS | 4096 elems |
  | bert head | 6.56e-6 | ✅ PASS | 4096 elems |
  | prosody durations | 0 (bit-exact) | ✅ PASS | 24-length integer array |
  | prosody n | 8.30e-4 | ✅ PASS | 1804 elems |
  | prosody hidden | 3.08e-3 | ✅ PASS | 24×512 elems |
  | **prosody f0** | **2.628e-2** | ❌ FAIL | T17-fixup #2 で honest negative 確定。3 GEMV variants を実測、いずれも regression。F0_proj Conv1d 256→1 が hidden の 3e-3 delta を ~9× 増幅、cell computation ではなく downstream amplification が atol-blocker と特定（`7527c7c` で nn.rs に rationale pin）。次 follow-up 候補: A1 と同型の f64 accumulator 局所適用 |
  | decoder mag | **1.21e-4** | ✅ PASS | T17-fixup #1 で **完全解決**（`e18efe0`）: `adain_conditioned` の Linear + InstanceNorm reduction を f64 accumulator 化、FP32 catastrophic cancellation を除去。分布 [1189957, 23, 0, 0, 0, 0] = 99.998% <1e-4 |
  | decoder phase | **7.92e-5** | ✅ PASS | 同 fixup で 100% <1e-4 |
  | decoder pcm | **6.84e-3** | ✅ PASS | 同 fixup で 99.97% <1e-4、102 samples が <1e-3 bucket |

  **honest 解釈（2026-07-07 workflow 後）**: 9 tensor 中 **8 が atol=0.01 PASS**、残る 1 tensor（prosody f0）は「computation は正しく、F0_proj downstream amplification が atol-blocker」という具体的な root cause が特定済み。A1（decoder）で有効だった **`adain_conditioned` の f64 accumulator パターン** を prosody の F0_proj/F0_conv 系にも局所適用する follow-up が最も筋の良い次手。診断機構（分布 histogram + `finite_worst_delta` + `VOKRA_KOKORO_PARITY_DUMP` + 新規 `tools/parity/kokoro_bisect.py` 200 行 stdlib-only）は既に整備済み。

- **副次的な発見・修正（cumulative）**:
  - **Bisection surface**（T17-fixup #1 副産物、commit `e18efe0`）: `decoder.rs`/`generator.rs` に `maybe_dump_stage` を追加、`dump_kokoro_reference.py` に対応する `_maybe_dump_stage`、`tools/parity/kokoro_bisect.py`（stdlib-only pairwise diff）を新規追加。env `VOKRA_KOKORO_PARITY_DUMP=<dir>` 未設定時は no-op。
  - Kokoro converter が iSTFT hparams（`n_fft`, `hop`, `win_length`）を `0` で書いていた placeholder を **20/5/20 (Kokoro canonical)** に修正（`aa52c8f`）。
  - `e2e_forward_matches_reference_shape` は `decoder_mode = full` 時に legacy `pcm.f32`（placeholder 16000 samples）との byte 比較を skip（authoritative reference は `decoder_pcm.f32`、`decoder_forward_bit_parity` が担当）。
  - Converter roundtrip test の istft placeholder 期待値を `Some(0)` → `Some(20)/Some(5)/Some(20)` に更新。
  - **Converter `--config <config.json>` 追加**（T17-fixup #3、commit `e060f97`）: `KokoroJsonConfig::parse` が `vocab: {sym:id}` / `phoneme_symbols: [str]` / `symbols: [str]` および `voices: [str]` / `voice_names: [str]` を alias 対応。missing families は loud `ConvertError::Parse`（silent fallback 無し）。CLI: `vokra-cli convert --model kokoro --config <path>.json`。config 無しの path は backward-compat placeholder。upstream `hexgrad/Kokoro-82M/config.json` の実 schema は unaccessible なので、テストは 178-symbol / 3-voice の plausible synthesis に basd、実 upstream JSON 到着時に refine 予定。

- **残（follow-up、次回セッション or 別 WP で消化）**:
  1. **prosody f0 f64 accumulator 局所適用**: A1 の `adain_conditioned` パターンを F0_proj / F0_conv 系に horizontal transplant。`n`/`hidden` は既に PASS なので、F0-specific downstream の局所 f64 化のみで atol=0.01 到達を狙う。semantic divergence リスクは限定的（decoder で先例あり）。
  2. **Whisper reference-dumper: 実 fixture 再生成**（B: script 面は完了 = commit `4665d74`）: `tests/fixtures/audio/jfk-30s.wav` を owner が配置し、`python3 tools/parity/dump_whisper_reference.py --audio ...` で 5 サイズ全再生成。task rubric で download 禁止のため CC 側では script only を land。
  3. **Kokoro upstream `config.json` 到着時の parser refine**: 現在の alias set が正解に一致するか実 upstream JSON で確認、必要なら key 名を追加。

---

## 4. モデル license audit + legal-compliance checklist 承認（M2-06 T18/T20、M2-07 T25/T26）

**依頼者タスク**: MIT / Apache 2.0 weight の商用配布可否を最終判断し、`docs/license-audit.md` + `docs/legal-compliance.md` に sign-off を残す。

### 必要な準備

- [ ] Whisper 4 サイズ MIT weight（openai/whisper）: 商用 OK の確認。
- [ ] Kokoro-82M Apache 2.0（hexgrad/Kokoro-82M）: 商用 OK + training data 疑義なしの確認。
- [ ] research flag 対象（F5-TTS / Fish-Speech / EnCodec）が公式 zoo に混入していないことの目視確認（M2-13 compliance gate が自動拒否するが最終目視）。

### 実行手順

`docs/license-audit.md` の各行に `Owner sign-off: <date>` を追記、`docs/legal-compliance.md` の Article 50 checklist を通す。

### CC 側整備状況（2026-07-07）

- **Owner sign-off template を `docs/license-audit.md` §3.1 に追加済み**（Task #78 agent、コミット `8d01b36`）。CC-verified 事実確認 subsection と、依頼者記入用の空欄 sign-off テーブル（Model / Weight License / CC-verified date / Owner sign-off (YYYY-MM-DD) / Approval / Notes）を提供。空欄行は fail-closed（未サインオフ＝公式配布不可）扱い。
- **Kill switch C/K メトリクス収集手順を独立 runbook 化**（Task #78 agent）: `docs/governance/kill-switch-metrics-runbook.md`（VOKRA-GOV-001）。GitHub stars / non-bot non-CC contributors / Issues + Discussions active participants の集計コマンド、集約 bash スクリプト、四半期 review 記録 template（`docs/governance/quarterly-reviews/YYYY-QN.md`）を規定。Discord 非採用（2026-07-04 依頼者決定）に伴う代替判定を含む。
- **Article 50 の runtime 面**: AudioSeal / C2PA 埋め込みは 2026-07-04 に依頼者ドロップ、M2-13 は `WatermarkConfig` の config surface のみを残す。NFR-LG-01/02 の runtime marking は未達 — deployment guide への disclosure text 要件記載は owner-side follow-up。

### Exit 判定への寄与

- FR-MD-13 のプロセス承認。M2-06/M2-07 WP 完了確定。

---

## 5. 言語バインディング初回対象合意（M2-12 T03）

**依頼者タスク**: 初回言語 = **Python (PyPI wheel)** で確定するか判断し sign-off。他候補（Swift / Kotlin / JS/TS）は rolling wave 次段。

### 必要な準備

- 特になし。CC は plan.md D1 の rationale で Python を推奨済み。

### 実行手順

- YES: 本チェックリストに ✅ を書き、rolling wave の次段の言語決定へ進む。
- NO: 別言語を指定 → CC が該当 binding scaffold を新規に構築（rolling wave）。

### Exit 判定への寄与

- M2-12 T03 の依頼者 sign-off。M2-12 WP 完了に必要。

---

## 6. PyPI 予約 + PYPI_API_TOKEN provision（M2-12 T17）

**依頼者タスク**: PyPI に `vokra` パッケージ名を予約（trademark 保護）、`PYPI_API_TOKEN` を GH Actions secret に登録するか OIDC trusted publisher を設定する。

### 必要な準備

- [ ] PyPI アカウント（既存 or 新規）。
- [ ] 2FA 設定（PyPI ルール）。

### 実行手順

1. `pip install twine` → `twine upload --repository testpypi bindings/python/dist/*.whl`（TestPyPI で dry-run）。
2. PyPI project 作成 → `pyproject.toml` の name = `vokra` を予約。
3. GH Actions secret `PYPI_API_TOKEN` を登録、または trusted publisher を GH Actions workflow に紐付け。
4. `git tag v0.5.0-rc1 && git push --tags` → `release.yml.python-pypi-publish` が起動、dry-run mode を経由して実 upload。

### Exit 判定への寄与

- M2-12 T17 の CD 発行完了。NFR-DS-03（PyPI 配布）。

---

## 7. Unity Editor license provision（M2-11 T-nightly）

**依頼者タスク**: `secrets.UNITY_LICENSE` を GH Actions secret に登録すると `nightly-il2cpp.yml` が IL2CPP スモークテストを nightly で実行するようになる。

### 必要な準備

- [ ] Unity Personal または Pro license（Unity Hub の manual activation → `.ulf` を base64 encode）。
- [ ] Unity 2022.3 LTS が installed（CI が game-ci/unity-builder@v4 でハンドル）。

### 実行手順

1. Unity Personal license を activate、`.ulf` を base64 encode。
2. GH Actions secret `UNITY_LICENSE` に登録。
3. 次の nightly 実行で `nightly-il2cpp.yml` が回り、IL2CPP AOT + DllImport(__Internal) + VokraAndroidAssets passthrough が検証される。

### Exit 判定への寄与

- M2-11 「IL2CPP 対応デモ動作」の実運用検証。
- 未 provision の場合は TESTING.md の手動手順を依頼者がローカル実行して署名。

---

## 8. Wyoming / Home Assistant 統合検証（M2-15 Kill switch J）

**依頼者タスク**: HA Voice PE + Wyoming Protocol クライアントで `vokra-server` を「推奨 Wyoming Server」として認識・接続する試験。採用可否は依頼者判断。

### 必要な準備

- [ ] Home Assistant + HA Voice PE の実験環境（M5Stack 実機不要、docker 上でも可）。
- [ ] `vokra-server` を Wyoming モードで起動できる Linux/macOS 環境。

### 実行手順

`integrations/vokra-server/docs/wyoming-design.md` の HA 接続例を参照。

### CC 側実測状況（2026-07-07、commits `c3f0fce` + `d076b8f` + `0bb73bb`）

**Wyoming info reply — 完全実装 + unit test hard-assert**:

- 環境: M1 iMac + Docker Desktop 24.0.6 + `homeassistant/home-assistant:stable`（sha256 `f73512ba...`）。詳細手順は `integrations/vokra-server/tests/wyoming-ha-smoke.md`。
- **wire-level reachability**（`c3f0fce`）: HA container が `vokra-server` を host:10300 経由で `host.docker.internal` (2.6 ms) と LAN IP の両方から reach できることを確認。
- **event loop fix**（`d076b8f`）: `wyoming_accept_loop` を `run_with_config` に配線、`signal.wait().await` を `block_on` の戻り前に挟むことで tokio runtime drop を防止。従来の smoke で発見された「Wyoming info reply が返らない」問題を解消。
- **unit-level hard-assert**（`0bb73bb`、workflow `wf_b891d85d-2f3`）: `crates/vokra-server/tests/wyoming_info_reply.rs`（350 行）で 3 tests 追加 — (1) describe → info reply の well-formed JSONL 一致を 5s deadline 内で hard-assert、(2) 3 fresh TCP 接続それぞれで info reply、(3) unknown event 後の explicit error を hard-assert（FR-EX-08 posture 保存）。全 test で `127.0.0.1:0` 動的 port を使い parallel `cargo test` の port collision を排除。**cargo test -p vokra-server → 120 unit + 16 integration all green**。
- **判定範囲**: unit test level の reply 検証は CC 側で完結。Kill switch J の採用可否（HA 側が Vokra を「推奨 Wyoming Server」として案内するか）は依頼者判断領域。

### Exit 判定への寄与

- Kill switch J 判定（HA 採用可否）。v0.5 時点で判定。
- 「未採用」= Kill switch 発動、「採用」= v1.0 の Wyoming 主要 endpoint 化。

---

## 9. Kill switch C/K 判定（M2-15 / 2026-12〜2027-01 目安）

**依頼者タスク**: v0.1 MVP 公開後 3 ヶ月時点（暦月目安 2026-12〜2027-01）に GitHub star 数と competitor community metric を再測し、以下を判定する。

### 判定閾値（milestones.md §6 転記）

- **C**: v0.1 MVP 公開後 3 ヶ月で GitHub star < 500 or 総合 engagement 過小 → 撤退検討。
- **K**: v0.5 時点で addressable market が競合の 10% 未満 → 撤退検討。

### 必要な準備

- 特になし。github.com/ayutaz/vokra の star 数 + Issues/Discussions active user proxy を集計。

### 実行手順

四半期 Go/No-go review record を独立公開ガバナンス記録として発行（governance docs / GitHub Discussion / post-mortem blog のいずれか）。

### Exit 判定への寄与

- Kill switch C/K の判定結果を M2-15 記録として公開。継続 or exit path 選択。

---

## Summary 進捗表（2026-07-07 時点）

| WP | 内容 | CC 進捗 | 依頼者残タスク |
|----|------|--------|--------------|
| M2-01 | Metal backend | ✅ 完了 | — |
| M2-02 | iOS build scaffold | ✅ 完了（scaffold） | § 1（iOS 実機 RTF） |
| M2-03 | CUDA backend + RTF<0.1 保証 | ✅ 実装完了 / 実測は decomposed path で **RTF 0.1133**（sanity <0.15 パス）、FA v2 wrapper は **RTF 0.1914** で revert（§ 2 参照） | § 2（formal <0.10 は M2-14 self-hosted runner で再検証） |
| M2-04 | graph fusion（log-mel 1 kernel） | ✅ 完了 | — |
| M2-05 | istft_streaming op | ✅ 完了 | — |
| M2-06 | Whisper large-v3/turbo | ✅ 部分完了 / parity fixture は 4 サイズ生成済（small/medium は partial、large-v3/turbo は greedy = eot のみで failed。§ 3 参照） | § 3（reference generator 見直し）+ § 4（audit） |
| M2-07 | Kokoro-82M | ✅ 骨格完了 / parity fixture は placeholder mode のみ（§ 3 参照） | § 3（byte-level full mode）+ § 4（audit） |
| M2-08 | quantization policy | ✅ 完了 | — |
| M2-09 | vokra-server 4 互換 API | ✅ 完了 | — |
| M2-10 | Discord bot デモ | ❌ descoped | Discord 全体を非採用（依頼者決定）。サーバ稼働実証は M2-15 review の別形態で扱う |
| M2-11 | Unity official plugin | ✅ 完了（UPM CD） | § 7（Unity license）|
| M2-12 | 言語バインディング（Python 初回） | ✅ 完了（wheel scaffold） | § 5（合意）+ § 6（PyPI token）|
| M2-13 | compliance 拡張 | ✅ 完了 | — |
| M2-14 | 実機ベンチ計測 | 引き渡し済み / CUDA reference 計測は完了、iOS 実機は依頼者側 | § 1 + § 2 |
| M2-15 | 四半期 Go/No-go review | 継続監視 / metrics runbook 整備済（§ 4 参照） | § 8（Kill switch J — wire-level PASS）+ § 9（C/K）|

---

## Contact / Escalation

- CC 側で追加 workflow が必要になった場合（例: 新規言語バインディング着手、実測結果を受けた最適化 follow-up）は本チェックリストに追記して依頼者から CC に振る。
- v0.5 Exit 判定は上記全項目の消化 + milestones.md §6 Exit criteria を根拠に依頼者が最終判断。

