# vast.ai handoff — vocoder / codec GPU kernel implementation wave

**Owner-triggered.** CC は本 doc 作成のみ。実 vast.ai instance の起動・NVRTC
kernel bakeoff・parity verification は owner が本 runbook を追いながら実行する。

**Related**:
- 本 runbook は `docs/handoff/vast-ai-large-model-publish.md`（総論）を **前提** と
  する。共通 provision + lifecycle は総論を参照し、本 doc は GPU kernel 実装 wave
  に固有の非対称性（Metal は M1 iMac で local 可 / CUDA は vast.ai owner 必須）
  のみを記述する。
- 本 wave は **model publish ではなく kernel implementation work item** である。
  他 2 handoff（`vast-ai-publish-voxcpm2-2b.md` / `vast-ai-publish-rmvpe.md`）は
  publish 手順で、本 doc は「Metal 半分 + CUDA 半分」の非対称性ゆえ CC 単発では
  完結しない follow-up をどう owner-hand-off するかを記述する。

## 1. Work item 情報

| 項目 | 値 |
|---|---|
| Category | GPU kernel implementation（vocoder + codec、CPU seam は既 real） |
| Scope | HiFTNet / BigVGAN / SNAC / Qwen3-TTS-codec の Metal MSL + CUDA NVRTC kernel |
| M-ticket 起源 | M4 spec の明示的 follow-up、CLAUDE.md M4 節末尾「GPU 実 kernel (codec/enhancement op の Metal MSL / CUDA NVRTC) は各 spec が明示的にスコープ外化した follow-up = 実装漏れではない」（M2-01 Metal / M2-03 CUDA / M3-06 mimi_rvq T14/T15 + M4-05 CSM + M4-16 FSQ / SoTA plan Phase 1-3 vocoder + codec 各種 spec のスコープ外項） |
| CPU arm 状態 | **既 real** — `vokra_ops::hiftnet` / `bigvgan_generator` / `snac_decode` / `qwen3_tts_codec` / `mimi_rvq` 等は CPU で機能完結、bit-exact parity harness 済み |
| Metal arm 状態 | **未実装** — `Compute::hifigan_f32` / `mimi_rvq_f32` 等の Metal arm は `VokraError::UnsupportedOp` を返す（FR-EX-08、never silent CPU fallback） |
| CUDA arm 状態 | **未実装** — 同上 CUDA arm も `VokraError::UnsupportedOp` |
| Vulkan arm 状態 | **未実装** — 同上 |
| 実装 blocker | Metal 側は M1 iMac 上で CC 可 / **CUDA 側は vast.ai + RTX 4090 で NVRTC compile + real-GPU bakeoff が必須** |

### 対象 op / モデル一覧

| op / モデル | primary source | CPU arm 状態 | GPU kernel M-ticket | 非対称性の理由 |
|---|---|---|---|---|
| `mimi_rvq_decode` (Mimi RVQ codec) | Kyutai Moshi / CSM upstream | ✅ real | M3-06 T14（Metal MSL）/ T15（NVRTC）→ M3-09 mimi_bridge upgrade past stub | Metal = M1 iMac 可 / CUDA = vast.ai 必須 |
| `hifigan` generator | HiFi-GAN paper + upstream | ✅ real（M3-07 land） | 未起票（M5+ follow-up） | 同上 |
| `hiftnet` generator（NSF + iSTFTNet） | CosyVoice2/3 upstream `cosyvoice/hifigan/generator.py:378` | ✅ real（SoTA plan Phase 1 land、`c3ff7b2`） | 未起票（M5+ follow-up、convolution stack + NSF sine gen が hot path） | 同上 |
| `bigvgan_generator`（AMP block + anti-aliased upsample） | NVIDIA/BigVGAN `bigvgan.py:206-354` MIT | ✅ real（SoTA plan Phase 3 land） | 未起票（M5+ follow-up、Snake activation + MRF が hot path） | 同上 |
| `snac_decode`（3-stage hierarchical RVQ） | hubertsiuzdak/SNAC MIT | ✅ real（SoTA plan Phase 3 land） | 未起票（M5+ follow-up） | 同上 |
| `qwen3_tts_codec`（continuous VAE + FSQ hybrid） | Qwen3-TTS upstream | ✅ real（SoTA plan Phase 3 land） | 未起票（M5+ follow-up） | 同上 |
| `dac_rvq`（DAC 24kHz） | descriptinc/descript-audio-codec MIT | ✅ real（M4-04 land） | 未起票（M5+ follow-up） | 同上 |
| `fsq_codec`（WavTokenizer + X-Codec 2） | WavTokenizer MIT / X-Codec 2 MIT | ✅ real（M4-16 land） | 未起票（M5+ follow-up） | 同上 |
| `denoise` （DeepFilterNet3 等） | DFN3 upstream | ✅ real（campaign-2 land、SI-SNR gap 2e-7 dB） | 未起票 | 同上 |
| `agc` / `hpf` / `loudness_norm` | WebRTC audio processing MIT | ✅ real（M4-20 land） | 未起票（低優先度） | 同上 |

## 2. 非対称性の詳細

### 2.1 Metal 半分 = M1 iMac 上で CC 可

**依頼者機 = Apple M1 iMac 16 GB**。Metal 実 GPU バックエンド（生 objc/Metal FFI +
MSL compute kernel）はローカル実行可能:

- 既存 pattern: `wave2b M4-06 Moshi` / `wave B Mimi` / `M2-01 Whisper full Metal
  e2e` 等で bit-identical vs CPU atol < 5e-4 実測（M1 iMac 上）
- 実装場所: `crates/vokra-backend-metal/src/` の MSL kernel 追加 +
  `crates/vokra-models/src/compute.rs` の Metal arm 実装
- Verification: `cargo test --features metal` で M1 iMac ローカル実行
- CC 単発の scope: MSL kernel を実装 → CPU との bit-identical parity で verify →
  commit（M4-20 T17 DFN3 Metal parity と同 pattern）

### 2.2 CUDA 半分 = vast.ai owner 必須

**依頼者機は CUDA 非搭載**（Apple M1）。CUDA バックエンド（生 FFI + NVRTC 実行時
コンパイル）は vast.ai 経由:

- 既存 pattern: `M2-03 CUDA full Whisper e2e`（vast.ai RTX 4090 で greedy 完全
  一致 5/5、encoder 1.32e-3 / decoder logits 4.29e-5）/ `M3-01 CUDA FA v2` /
  `Wave 14 vast.ai N=10 reference results 2026-07-10 collection`
  （`docs/bench-baselines/vast-2026-07-10/`）
- 実装場所: `crates/vokra-backend-cuda/src/` の NVRTC kernel source string 追加 +
  `crates/vokra-models/src/compute.rs` の CUDA arm 実装
- Verification: **vast.ai 上でしか実行できない** — NVRTC PTX compile + real GPU
  で bit-identical vs CPU の parity 実測が必要
- CC 単発の scope: kernel source string の drafting + CPU arm reference の pin
  まで（vast.ai bakeoff は owner 側で回して結果 report）

### 2.3 なぜ「別 WP」なのか（実装漏れではない）

**M4 spec が明示的にスコープ外化した follow-up 判断**（CLAUDE.md M4 節、
2026-07-15 terminal 到達時に investigation 3 round で 2 連続 0 CC ticket = terminal
判定の boundary 定義）:

1. **CPU arm は real で機能完結**: 全 op が CPU で bit-exact に動く、実 model 経路
   （CosyVoice2 mel synth / Moshi / CSM 実 weight parity 等）は全て CPU + Metal
   half で pass
2. **GPU 化 = 性能最適化**: kernel 実装は correctness ではなく latency / throughput
   の話ゆえ、CPU real + Metal half でも "Vokra が動く" は成立
3. **非対称性ゆえ CC 単発では完結しない**: Metal 半分 (M1 iMac) を CC が書いても
   CUDA 半分 (vast.ai) は owner 実行になるため、"landing" タイミングが分かれる
   （partial landing は verify-on-actual-HEAD 規律に沿わない、CLAUDE.md M4 教訓）

## 3. Instance recipe（vast.ai bakeoff）

Metal 半分は M1 iMac ローカル、CUDA 半分のみ vast.ai:

| 項目 | 推奨値 | 備考 |
|---|---|---|
| Image | `pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel` or `nvidia/cuda:12.4.0-devel-ubuntu22.04` | 総論 §2.2 と同じ |
| RAM | 32 GB 以上 | 大 model の GGUF load + NVRTC compile working set |
| Disk | 100 GB 以上 | 上流 model DL 数種類 + Vokra release build + PTX cache |
| GPU | **RTX 4090 or A100 or H100** | 既存 CUDA 検証も RTX 4090 実測（M2-03、M3-01 vast.ai N=10 reference）で pattern 一致。**H100 は FA v3 用ゆえ本 wave では不要** |
| Network | 非従量課金 or inclusive | 上下 ~5-10 GB out-bound（HF model DL 数種類） |
| 課金見込 | ~2-4 hours × $0.5-1.0/hr = **$1-4**（kernel 実装 wave 全体で複数 op を bakeoff、1-2 op 単位で分割起動も可） |

## 4. provision.sh gotcha（総論 §3）

総論 §3 の 4 gotcha 全て該当。`scripts/publish/vast-ai/provision.sh` を 1 コマンド
で実行（`docs/handoff/vast-ai-large-model-publish.md` §0 TL;DR 参照）。

## 5. Kernel implementation workflow

**注意**: 本 wave は publish command ではなく **kernel implementation +
bit-identical parity verify** のフロー。以下は Metal + CUDA 両側の pattern。

### 5.1 Metal MSL kernel（M1 iMac local、CC 単発可）

例: `mimi_rvq_decode` の Metal arm 実装（M3-06 T14）:

```bash
# M1 iMac local
cd ~/vokra

# 1. MSL kernel を追加
# crates/vokra-backend-metal/src/kernels/mimi_rvq_decode.metal
# （既存 kernel の pattern: fused_softmax_causal.metal 等）

# 2. compute.rs の Metal arm を実装
# crates/vokra-models/src/compute.rs
# impl Compute { fn mimi_rvq_f32 {
#     Compute::Metal(ctx) => ctx.mimi_rvq_f32(codes, weights, ...),  // 新規
#     ...
# }}

# 3. Bit-identical parity で verify
cargo test --features metal -p vokra-models mimi_rvq_metal_bit_identical
# atol < 5e-4 (M4-20 T17 DFN3 pattern の horizontal 展開)

# 4. Commit
git add crates/vokra-backend-metal/src/kernels/mimi_rvq_decode.metal
git add crates/vokra-models/src/compute.rs
git commit -m "feat(mimi-rvq): Metal MSL kernel + bit-identical vs CPU parity"
```

### 5.2 CUDA NVRTC kernel（vast.ai owner 必須）

例: 同じ `mimi_rvq_decode` の CUDA arm 実装（M3-06 T15）:

```bash
# vast.ai RTX 4090 SSH 接続後
export HF_TOKEN='hf_xxxxxx'
curl -sSL https://raw.githubusercontent.com/ayutaz/vokra/main/scripts/publish/vast-ai/provision.sh | bash
source ~/.bashrc

cd ~/vokra

# 1. NVRTC kernel source string を追加（既 CC が下書き済み前提）
# crates/vokra-backend-cuda/src/kernels/mimi_rvq_decode.cu.rs
# （既存 kernel の pattern: fa_v2_causal.cu 相当の const &'static str）

# 2. compute.rs の CUDA arm を実装
# crates/vokra-models/src/compute.rs
# impl Compute { fn mimi_rvq_f32 {
#     Compute::Cuda(ctx) => ctx.mimi_rvq_f32(codes, weights, ...),  // 新規
#     ...
# }}

# 3. NVRTC compile + real GPU で bit-identical parity
cargo build --release --features cuda
cargo test --release --features cuda -p vokra-models mimi_rvq_cuda_bit_identical
# NVRTC PTX compile + RTX 4090 実行、atol < 5e-4

# 4. RTF 計測（optional、baseline vs GPU）
./target/release/vokra-cli bench --backend cuda --model <target-model>
# 結果を docs/bench-baselines/vast-YYYY-MM-DD/mimi-rvq-gpu.jsonl に record

# 5. Commit → push（vast.ai から本 branch へ）
git add crates/vokra-backend-cuda/src/kernels/mimi_rvq_decode.cu.rs
git add crates/vokra-models/src/compute.rs
git add docs/bench-baselines/vast-YYYY-MM-DD/mimi-rvq-gpu.jsonl
git commit -m "feat(mimi-rvq): CUDA NVRTC kernel + bit-identical vs CPU (vast.ai)"
git push
```

### 5.3 Verify-on-actual-HEAD 規律（両側）

Metal + CUDA が **別 commit**（Metal は M1 iMac から / CUDA は vast.ai から）で
land する場合、片側が land した段階で **必ず全 gate 再走**（memory
[[project-m4-implementation]] の verify-on-actual-HEAD 規律を horizontal 展開）:

```bash
# M1 iMac 上（両 land 後の統合 HEAD 上）
cd ~/vokra
git pull origin <branch>
cargo test --workspace  # default
cargo test --workspace --features vulkan,metal  # M1 iMac は Metal 有効
cargo fmt --check
cargo clippy -- -D warnings
scripts/check-zero-deps.sh
scripts/gen-c-abi.sh --check  # C ABI drift 検出
```

CUDA 側の verify は vast.ai instance 上で追加実行（`--features cuda`）:

```bash
# vast.ai RTX 4090 上（両 land 後の統合 HEAD 上）
cargo test --release --features cuda,metal,vulkan  # all-features
```

## 6. §3.1 sign-off status（非該当）

本 wave は **model publish ではなく kernel implementation work item** ゆえ、§3.1
sign-off の対象外。上流 model の再配布は行わない（既 published `vokra/moshiko-7b`
/ `vokra/csm-1b` / `vokra/dac-24khz` / `vokra/mimi` / `vokra/cosyvoice2-0.5b` 等
の GGUF に kernel 側の変更は影響しない）。

## 7. 期待される artifacts

Kernel 実装 wave 完了後の artifacts:

| Path | 内容 |
|---|---|
| `crates/vokra-backend-metal/src/kernels/*.metal` | 新規 MSL kernel source files（op ごとに 1-2 file） |
| `crates/vokra-backend-cuda/src/kernels/*.cu.rs` | 新規 NVRTC kernel source string（`const &'static str`、実行時 compile） |
| `crates/vokra-models/src/compute.rs` | Metal / CUDA arm の実装追加 |
| `crates/vokra-models/tests/*_metal_bit_identical.rs` | M1 iMac local で bit-exact parity verify |
| `crates/vokra-models/tests/*_cuda_bit_identical.rs` | vast.ai 上で bit-exact parity verify |
| `docs/bench-baselines/vast-YYYY-MM-DD/*.jsonl` | RTF baseline（optional、GPU vs CPU speedup 記録） |
| `docs/abi-changelog.md` | Rust surface のみ additions（新規 C ABI ゼロ、既存 M4 pattern と同じ） |

**新規 C ABI = 0**（既 M4-06 で `vokra_s2s_duplex_*` 等が land 済み、本 wave は
kernel を追加するだけで public API 面には影響しない）。zero-dep NFR-DS-02 preserved
（Metal は生 objc/Metal FFI、CUDA は生 dlopen + NVRTC、binding crate 追加なし）。

## 8. Owner critical path

**優先度は依頼者判断**（本 wave は correctness ではなく性能最適化ゆえ v1.0 GA
blocking ではない、C ABI 凍結 M5-13 の precondition でもない）。実行するとしたら:

1. **CC がどの op から始めるか request** — owner が優先 op を指定（`mimi_rvq` /
   `hifigan` / `bigvgan` / `hiftnet` / `snac` / `qwen3_tts_codec` / `dac_rvq` /
   `fsq_codec` / `denoise` 等から選択、ちなみに **`mimi_rvq` (Moshi/CSM real-time
   critical) + `hiftnet` (CosyVoice2/3 hot path) が最優先候補**）
2. **CC 側 Metal 半分実装** — M1 iMac local で MSL kernel + bit-identical parity
   → commit（1 op 単位、bundle でも OK）
3. **Owner vast.ai instance 起動** — §3 recipe、~$1-4 / 1-2 op
4. **CC 側 CUDA kernel source string 下書き** — vast.ai bakeoff 前に CC が
   drafting（`const &'static str` の string literal ゆえ compile 検証は vast.ai 上
   で）
5. **Owner vast.ai 上で NVRTC compile + real GPU bakeoff** — §5.2 の workflow を
   実行、bit-identical vs CPU verify、RTF 計測（optional）
6. **Owner commit + push from vast.ai** — Metal + CUDA が両揃った時点で
   `docs/abi-changelog.md` に entry 追加（Rust surface additions のみ、新規 C ABI
   ゼロ）
7. **Verify-on-actual-HEAD** — §5.3 の統合 verify を M1 iMac 上で実行、all gate
   green 確認

## 9. Notes

- **本 wave は M5-13 C ABI 凍結の precondition ではない**: 新規 C ABI ゼロゆえ
  凍結タイミング（v1.0 GA タグ）と decouple。GA タグ後でも v1.0.x patch release
  で追加可能（M4-12 handoff §(e)-4 の "patch-release 条項" と同 pattern）。
- **本 wave は NPU bakeoff（M5-01 CoreML/ANE / M5-02 QNN/Hexagon）と別**: NPU
  delegate は実機 owner necessary で M5-13 blocking、GPU kernel は GPU 実機 owner
  necessary だが v1.0 GA blocking ではない。優先度は NPU > GPU（NPU は 2× gate
  = NFR-PF-12 の commercial pitch、GPU は latency 最適化）。
- **FA v3 (Hopper H100) は本 wave 対象外**: `docs/milestones.md` §5-(7) の FA v3
  confinement red-line を継承（M4-07 で先行 primitive 追加済、実 model 経路への
  wiring は M5 以降 spec ゆえ、本 wave では kernel を書かない）。
- **Vulkan 半分も同型 non-goal**: Metal + CUDA を先に land、Vulkan は M4-13 で
  base scaffold まで land 済み、実 kernel は v1.0 GA 後の major release で。
- **教訓の horizontal 展開**: 本 wave は M4-20 T17 DFN3 の Phase B gate + M2-01
  Metal M1 iMac 実測 pattern を全 vocoder / codec op に horizontal 展開する構成。
  Kernel 数は約 8-10 op × Metal + CUDA = 16-20 kernel、1 op あたり Metal 半日 +
  CUDA 半日 + vast.ai bakeoff 数時間の budget が現実的な見積り。

## 関連

- 総論: `docs/handoff/vast-ai-large-model-publish.md`
- CPU arm reference: `crates/vokra-ops/src/{hifigan,hiftnet,bigvgan_generator,snac_decode,qwen3_tts_codec,mimi_rvq,dac_rvq,fsq_codec}.rs`
- Compute seam: `crates/vokra-models/src/compute.rs`（`HotOp` enum + `for_backend` gate）
- 既存 GPU kernel pattern:
  - Metal: `crates/vokra-backend-metal/src/kernels/*.metal`（例: `fused_softmax_causal.metal`）
  - CUDA: `crates/vokra-backend-cuda/src/kernels/*.cu.rs`（例: `fa_v2_causal`）
- Metal M1 iMac real GPU parity 実測 precedent:
  - M2-01 Whisper full Metal e2e（greedy 完全一致 vs CPU 1.58e-6）
  - Wave B（2026-07-16 real weight campaign）Mimi / Moshi / CosyVoice2 Δ 4.8-7.2e-7
  - Wave D M4-20 T17 DFN3 SI-SNR gap 2e-7 dB
- CUDA vast.ai bakeoff 実測 precedent:
  - M2-03 full Whisper on CUDA e2e（vast.ai RTX 4090、greedy 5/5 完全一致）
  - M3-01 CUDA FA v2 primitive
  - Wave 14 vast.ai N=10 reference (`docs/bench-baselines/vast-2026-07-10/`)
- Memory: [[feedback-large-models-on-vast-ai]] / [[project-m4-implementation]] /
  [[project-real-weight-eval]]
- CLAUDE.md M4 節末尾「GPU 実 kernel (codec/enhancement op の Metal MSL / CUDA
  NVRTC) は各 spec が明示的にスコープ外化した follow-up = 実装漏れではない」
