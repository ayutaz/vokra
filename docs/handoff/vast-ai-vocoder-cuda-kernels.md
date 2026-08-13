# vast.ai handoff — Vocoder CUDA-half kernel implementation

**Date**: 2026-08-13（audit）→ 2026-08-14（本 handoff land）
**Branch**: `feat/post-audit-cc-gap-2026-08-13`
**Status**: **Owner-triggered.** CC は本 handoff の作成のみ。CUDA NVRTC kernel
の実装・vast.ai bakeoff・real-GPU parity verification は owner が本 runbook を
追いながら実行する。
**Related**:
- **前提**: `docs/handoff/vast-ai-vocoder-gpu-kernels.md`（総論 — Metal/CUDA
  非対称性の判断根拠、M4 spec の follow-up として scope 外化した rationale）
- **前提**: `docs/handoff/vast-ai-large-model-publish.md`（vast.ai provision +
  lifecycle 総論）
- **audit source**: post-audit 2026-08-13 wave 「Vocoder CUDA half」 finding
  （`docs/handoff/post-audit-2026-08-13-summary.md`）。**8 kernel いずれも
  `crates/vokra-backend-cuda/src/context.rs` に不在**（`grep -n "mimi_rvq|
  dac_rvq|wavtokenizer|xcodec2|snake_activation|snac_decode|denoise|
  qwen3_tts_codec|fsq_codec" crates/vokra-backend-cuda/src/context.rs` は
  hit 0）。全て Metal-only、`Compute::*_f32` の `Be::Cuda` arm は
  `VokraError::UnsupportedOp` を返す（FR-EX-08、never silent CPU fallback）。

**Why handoff-only（not speculative CUDA）**:
- CC は M1 iMac で `--features cuda` build も NVRTC kernel bakeoff もできない
  （cudarc 不在 + libcuda 不在 + Apple silicon）。CI 上の CUDA gate も
  self-hosted runner 未 provision（`docs/m2-cuda-rtf-variance-2026-07-08.md`
  M2-14 defer）。
- 未検証 CUDA を書いても owner debug 負担が増えるだけ = 「未 verify を実装
  漏れ扱いで land する」規律違反（CLAUDE.md M4 教訓「fake-complete より
  loud-partial の方が honest」）。
- 本 doc は Metal MSL の verbatim source + CUDA NVRTC 翻訳 template + launch
  config + parity target を全 8 kernel 分同梱するので、owner は「本 doc を
  読みながら 1 kernel ずつ実装 → NVRTC compile → CPU parity で verify → commit」
  の cycle を回せる。

---

## 1. Work item と対象 kernel 一覧

**Scope**: Metal-landed 8 kernel の CUDA NVRTC mirror。追加で 2026-08-14 land
の WF6 共通 vocoder primitives 3 種も CUDA 未実装（section 4 で扱う）。

### 1.1 audit 対象 8 kernel（本 doc の主対象）

| # | kernel name | landed Metal wave | 上流 CPU op | 用途 / 消費モデル |
|---|---|---|---|---|
| 1 | `vokra_mimi_rvq_gather_fold_f32` | WF1（M3-06 T14） | `vokra_ops::mimi_rvq_decode` | Mimi RVQ codec — Moshi, CSM-1B |
| 2 | `vokra_dac_rvq_gather_project_fold_f32` | WF2（M4-04） | `vokra_ops::dac_rvq_decode` | DAC 24kHz — descriptinc/descript-audio-codec |
| 3 | `vokra_wavtokenizer_vq_gather_f32` | WF2（M4-16） | `vokra_ops::fsq_codec::wavtokenizer_vq_decode` | WavTokenizer — 単一 codebook FSQ |
| 4 | `vokra_xcodec2_fsq_decode_f32` | WF2（M4-16） | `vokra_ops::fsq_codec::xcodec2_fsq_decode` | X-Codec 2 — grid-decompose FSQ |
| 5 | `vokra_snake_activation_f32` | WF2 | `vokra_ops::snake_activation_f32` | HiFTNet / Kokoro-82M 系 alias-free activation |
| 6 | `vokra_snac_decode_f32` | WF5 | `vokra_ops::snac_decode::SnacDecoder::decode` | SNAC 3-stage hierarchical RVQ — Orpheus, Maya1 |
| 7 | `vokra_denoise_apply_mask_f32` | WF5 | `vokra_ops::denoise::denoise_apply_mask_f32` | DFN3 / GTCRN / RNNoise 出力段 mask apply |
| 8 | `vokra_qwen3_tts_codec_decode_f32` | WF5 | `vokra_ops::qwen3_tts_codec::qwen3_tts_codec_decode` | Qwen3-TTS-Codec 12Hz — 全 Qwen3-TTS 声 |

### 1.2 WF6 common vocoder primitives（本 doc の付録対象、audit 8 に含まれず）

以下は本 branch 2026-08-14 land の Metal-only 3 追加 kernel。同一 pattern で
owner に mirror してもらう follow-up:

| # | kernel name | landed Metal wave | 上流 CPU op | 用途 |
|---|---|---|---|---|
| A1 | `vokra_snake_beta_f32` | WF6（commit `967131c`） | `vokra_ops::snake_beta_f32` | BigVGAN 系 α+β 分離 activation |
| A2 | `vokra_sinegen_deterministic_f32` | WF6（commit `967131c`） | `vokra_ops::sinegen_deterministic_f32` | NSF SineGen 決定パス — HiFTNet / CosyVoice2/3 |
| A3 | `vokra_anti_aliased_upsample_f32` | WF6（commit `967131c`） | `vokra_ops::anti_aliased_upsample_f32` | Kaiser polyphase upsample — BigVGAN / HiFTNet |

---

## 2. NVRTC 翻訳 template（一般規則）

Metal MSL → CUDA NVRTC の 1-to-1 対応 pattern。**全 8 kernel 共通** ゆえ、
per-kernel section では kernel-specific な部分（buffer 順・shape）だけ扱う。

### 2.1 Kernel 属性の翻訳表

| MSL | CUDA NVRTC | 補足 |
|---|---|---|
| `kernel void foo(...)` | `extern "C" __global__ void foo(...)` | `extern "C"` は C++ mangling 抑止（`cuModuleGetFunction("foo")` 解決の必須条件） |
| `device const float*  buf [[buffer(N)]]` | `const float* buf` | `[[buffer(N)]]` は Metal binding 属性、CUDA kernel arg は launch parameter list の順序で暗黙的に対応 |
| `device float*        buf [[buffer(N)]]` | `float* buf` | 同上（read-write buffer） |
| `constant Foo&        d   [[buffer(N)]]` | `unsigned int a, unsigned int b, ...`（構造体を展開）or `Foo d`（NVRTC で完全同定義を declare） | **推奨は個別 uint 展開**。既存 CUDA kernel（GEMV / softmax / layer_norm）は全て個別 arg で書かれており pattern 一致 |
| `uint2 gid [[thread_position_in_grid]]` | `unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;`<br>`unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;` | 2D 座標。`gid.x` → `col`, `gid.y` → `row`（あるいは MSL コメントの `t` / `d` 命名を尊重） |
| `uint gid [[thread_position_in_grid]]` | `unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;` | 1D 座標 |
| `threadgroup float smem[N]` | `__shared__ float smem[N];`<br>or `extern __shared__ float smem[];` | dynamic shared memory は `cuLaunchKernel` の `sharedMemBytes` に指定 |
| `threadgroup_barrier(mem_flags::mem_threadgroup)` | `__syncthreads();` | |
| `sin(x)` / `cos(x)` / `exp(x)` / `sqrt(x)` | `sinf(x)` / `cosf(x)` / `expf(x)` / `sqrtf(x)` | FP32 single-precision intrinsics。既存 CUDA kernel の precision red line に一致 |
| `floor(x)` | `floorf(x)` | 同上 |
| `1.0e-9f` | `1.0e-9f` | リテラルはそのまま |
| `select(a, b, cond)` (MSL) | `(cond) ? a : b` | CUDA は C ternary |

### 2.2 Launch geometry の翻訳

| MSL grid helper | 対応 CUDA launch | 備考 |
|---|---|---|
| `grid_2d(nx, ny)` = threadgroups `(nx.div_ceil(16), ny.div_ceil(16), 1)` × threads `(16, 16, 1)` | `grid = (nx.div_ceil(BLOCK) as c_uint, ny.div_ceil(BLOCK) as c_uint, 1)`<br>`block = (BLOCK, BLOCK, 1)` (with `BLOCK = 16`) | 全 codec kernel 共通。既存 CUDA `BLOCK = 16` 定数を再利用 |
| `grid_1d(count)` = threadgroups `(count.div_ceil(256), 1, 1)` × threads `(256, 1, 1)` | `grid = (count.div_ceil(BLOCK_1D as usize) as c_uint, 1, 1)`<br>`block = (BLOCK_1D, 1, 1)` (with `BLOCK_1D = 256`) | 既存 `vokra_gemv_f32` / `vokra_gelu_f32` 等の pattern |

### 2.3 NVRTC 特有の注意

- `KERNELS_CUDA` const string の末尾に本 8 kernel を **追記**（新 module 分割は
  不要 — 既存 pattern は `KERNELS_CUDA` 1 module に全 sibling を bundle）。
- `#ifndef INFINITY` guard は既に L98-100 で定義済（NVRTC の `<math.h>`
  不読込対策）ゆえ、`INFINITY` を新 kernel で使うときも追加 include 不要。
- **`Modules` struct（L5698 附近）に 8 個の `CUfunction` field を追加**、
  `load_modules`（L5785）で `get_function(driver, kernels_module.module, c"vokra_mimi_rvq_gather_fold_f32")?` の pattern で resolve。
- **`impl CudaContext` に 8 個の `pub fn mimi_rvq_f32(...)` を追加**、既存
  Metal wrapper（`crates/vokra-backend-metal/src/context.rs` L4480 附近）
  と shape 一致の Rust API。
- **`crates/vokra-models/src/compute.rs` の `Be::Cuda` arm を書き換え**:
  現行 `Err(VokraError::UnsupportedOp(...))` を `Be::Cuda(ctx) => ctx.mimi_rvq_f32(...)` に。既存の Metal arm と対称の host-side shape / index bound check を書き写す（silent GPU OOB 防止、FR-EX-08）。

### 2.4 Parity red line（audio-dialect rule）

- **FP32 accumulator throughout**（CLAUDE.md「BF16 mantissa loss is the real
  problem」）— NVRTC で `float` を使い、`__half` / `__nv_bfloat16` は禁止。
- **No cuBLAS / cuDNN 依存**（zero-dep NFR-DS-02、EULA install モデル）
  — NVRTC で self-contained に書く。既存 `KERNELS_CUDA` と同 pattern。
- **No TF32 fast path**（cuBLAS の `CUBLAS_TF32_TENSOR_OP_MATH` は使わない）
  — 既存 GEMM kernel の precision red line 継承。

---

## 3. Per-kernel implementation notes（8 kernel）

各 section 構成:
- (a) Metal MSL source（`crates/vokra-backend-metal/src/context.rs` verbatim
  抜粋。line 番号は 2026-08-14 の HEAD `653186c` 時点）
- (b) NVRTC 翻訳 template（`extern "C" __global__ void ...` signature を含む
  starter code。owner は body を MSL から transcribe する）
- (c) Launch config
- (d) Parity target
- (e) Compute seam wiring（`compute.rs` の `Be::Cuda` arm 差分ヒント）

### 3.1 `vokra_mimi_rvq_gather_fold_f32`（WF1 / M3-06 T14）

**Semantics**: `out[t, d] = Σ_cb tables[cb].row(codes[t, cb])[d]`

#### (a) Metal MSL source（`context.rs` L799-824）

```metal
struct MimiRvqDims {
    uint n_codebooks;
    uint codebook_size;
    uint d_model;
    uint time;
};

kernel void vokra_mimi_rvq_gather_fold_f32(
    device const uint*      codes  [[buffer(0)]],
    device const float*     tables [[buffer(1)]],
    device float*           out    [[buffer(2)]],
    constant MimiRvqDims&   d      [[buffer(3)]],
    uint2                   gid    [[thread_position_in_grid]])
{
    const uint t     = gid.y;
    const uint delem = gid.x;
    if (t >= d.time || delem >= d.d_model) {
        return;
    }
    const uint code_base = t * d.n_codebooks;
    const uint cb_stride = d.codebook_size * d.d_model;
    float acc = 0.0f;
    for (uint cb = 0; cb < d.n_codebooks; ++cb) {
        const uint idx       = codes[code_base + cb];
        const uint table_off = cb * cb_stride + idx * d.d_model + delem;
        acc += tables[table_off];
    }
    out[t * d.d_model + delem] = acc;
}
```

#### (b) CUDA NVRTC translation template

```cuda
extern "C" __global__ void vokra_mimi_rvq_gather_fold_f32(
    const unsigned int* codes,
    const float*        tables,
    float*              out,
    unsigned int        n_codebooks,
    unsigned int        codebook_size,
    unsigned int        d_model,
    unsigned int        time)
{
    unsigned int delem = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int t     = blockIdx.y * blockDim.y + threadIdx.y;
    if (t >= time || delem >= d_model) {
        return;
    }
    unsigned int code_base = t * n_codebooks;
    unsigned int cb_stride = codebook_size * d_model;
    float acc = 0.0f;
    for (unsigned int cb = 0; cb < n_codebooks; ++cb) {
        unsigned int idx       = codes[code_base + cb];
        unsigned int table_off = cb * cb_stride + idx * d_model + delem;
        acc += tables[table_off];
    }
    out[t * d_model + delem] = acc;
}
```

#### (c) Launch config

- grid: `(d_model.div_ceil(BLOCK) as c_uint, time.div_ceil(BLOCK) as c_uint, 1)`
- block: `(BLOCK, BLOCK, 1)` with `BLOCK = 16`
- shared memory: 0（no `__shared__`）
- Kernel arg 順（`cu_launch_kernel` の `params`）: `codes`, `tables`, `out`, `n_codebooks`, `codebook_size`, `d_model`, `time`

#### (d) Parity target

- **Bound**: `atol ≤ 5e-4`（FP32 GEMV-scale、fast-math 再結合を想定）
- **典型実測**: canonical Mimi shape（`n_codebooks=8`）で `|Δ| = 0` の
  bit-identical が達成される見込み（Metal 側実測、`vokra-models/tests/
  mimi_rvq_metal_bit_identical.rs` 参照）。
- **Reference**: `vokra_ops::mimi_rvq::rvq_fold_core`（strict left-to-right FP32 loop）

#### (e) Compute seam wiring

- Rust wrapper: `impl CudaContext { pub fn mimi_rvq_f32(&self, codes: &[u32], tables_flat: &[f32], n_codebooks: usize, codebook_size: usize, d_model: usize, time: usize) -> Result<Vec<f32>>` — Metal wrapper（`crates/vokra-backend-metal/src/context.rs::mimi_rvq_gather_fold_f32` L4480-4547）と shape 一致。
- host-side shape / index bound check は Metal wrapper と対称に書く（`vokra_ops::mimi_rvq::check_tables_shape` / `check_codes_shape` / `CodebookTable::row` を mirror）。
- `crates/vokra-models/src/compute.rs::mimi_rvq_f32` L1153-1160 の `Be::Cuda(_) => Err(VokraError::UnsupportedOp(...))` を `Be::Cuda(ctx) => ctx.mimi_rvq_f32(codes, tables_flat, attrs.n_codebooks, attrs.codebook_size, attrs.d_model, time)` に差替え。

---

### 3.2 `vokra_dac_rvq_gather_project_fold_f32`（WF2 / M4-04）

**Semantics**: `out[t, d] = Σ_cb (proj_biases[cb, d] + Σ_c proj_weights[cb, d, c] * low_tables[cb, codes[t, cb], c])`

#### (a) Metal MSL source（`context.rs` L864-897）

```metal
struct DacRvqDims {
    uint n_codebooks;
    uint codebook_size;
    uint codebook_dim;
    uint d_model;
    uint time;
};

kernel void vokra_dac_rvq_gather_project_fold_f32(
    device const uint*      codes        [[buffer(0)]],
    device const float*     low_tables   [[buffer(1)]],
    device const float*     proj_weights [[buffer(2)]],
    device const float*     proj_biases  [[buffer(3)]],
    device float*           out          [[buffer(4)]],
    constant DacRvqDims&    d            [[buffer(5)]],
    uint2                   gid          [[thread_position_in_grid]])
{
    const uint t     = gid.y;
    const uint delem = gid.x;
    if (t >= d.time || delem >= d.d_model) {
        return;
    }
    const uint code_base   = t * d.n_codebooks;
    const uint low_stride  = d.codebook_size * d.codebook_dim;
    const uint w_stride    = d.d_model * d.codebook_dim;
    float acc = 0.0f;
    for (uint cb = 0; cb < d.n_codebooks; ++cb) {
        const uint idx     = codes[code_base + cb];
        const uint low_off = cb * low_stride + idx * d.codebook_dim;
        const uint w_off   = cb * w_stride + delem * d.codebook_dim;
        float y = proj_biases[cb * d.d_model + delem];
        for (uint c = 0; c < d.codebook_dim; ++c) {
            y += proj_weights[w_off + c] * low_tables[low_off + c];
        }
        acc += y;
    }
    out[t * d.d_model + delem] = acc;
}
```

#### (b) CUDA NVRTC translation template

```cuda
extern "C" __global__ void vokra_dac_rvq_gather_project_fold_f32(
    const unsigned int* codes,
    const float*        low_tables,
    const float*        proj_weights,
    const float*        proj_biases,
    float*              out,
    unsigned int        n_codebooks,
    unsigned int        codebook_size,
    unsigned int        codebook_dim,
    unsigned int        d_model,
    unsigned int        time)
{
    unsigned int delem = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int t     = blockIdx.y * blockDim.y + threadIdx.y;
    if (t >= time || delem >= d_model) {
        return;
    }
    unsigned int code_base  = t * n_codebooks;
    unsigned int low_stride = codebook_size * codebook_dim;
    unsigned int w_stride   = d_model * codebook_dim;
    float acc = 0.0f;
    for (unsigned int cb = 0; cb < n_codebooks; ++cb) {
        unsigned int idx     = codes[code_base + cb];
        unsigned int low_off = cb * low_stride + idx * codebook_dim;
        unsigned int w_off   = cb * w_stride + delem * codebook_dim;
        float y = proj_biases[cb * d_model + delem];
        for (unsigned int c = 0; c < codebook_dim; ++c) {
            y += proj_weights[w_off + c] * low_tables[low_off + c];
        }
        acc += y;
    }
    out[t * d_model + delem] = acc;
}
```

#### (c) Launch config

- grid: `(d_model.div_ceil(BLOCK), time.div_ceil(BLOCK), 1)` with `BLOCK = 16`
- block: `(BLOCK, BLOCK, 1)`
- shared memory: 0
- Kernel arg 順: `codes, low_tables, proj_weights, proj_biases, out, n_codebooks, codebook_size, codebook_dim, d_model, time`

#### (d) Parity target

- **Bound**: `atol ≤ 5e-4`（FP32 GEMV-scale、inner `Σ_c` の fast-math 再結合ゆえ bit-identical は保証しない）
- **Reference**: `vokra_ops::dac_rvq::dac_rvq_decode`

#### (e) Compute seam wiring

- Metal wrapper: `vokra-backend-metal/src/context.rs::dac_rvq_gather_project_fold_f32` L4647 附近を mirror
- `compute.rs::dac_rvq_f32` の `Be::Cuda(_)` arm 差替え

---

### 3.3 `vokra_wavtokenizer_vq_gather_f32`（WF2 / M4-16）

**Semantics**: `out[t, d] = codebook_table[codes[t]].row[d]`（純 gather、bit-identical）

#### (a) Metal MSL source（`context.rs` L934-948）

```metal
struct WavTokenizerVqDims {
    uint vocab_size;
    uint d_model;
    uint time;
};

kernel void vokra_wavtokenizer_vq_gather_f32(
    device const uint*              codes  [[buffer(0)]],
    device const float*             table  [[buffer(1)]],
    device float*                   out    [[buffer(2)]],
    constant WavTokenizerVqDims&    d      [[buffer(3)]],
    uint2                           gid    [[thread_position_in_grid]])
{
    const uint t     = gid.y;
    const uint delem = gid.x;
    if (t >= d.time || delem >= d.d_model) {
        return;
    }
    const uint idx = codes[t];
    out[t * d.d_model + delem] = table[idx * d.d_model + delem];
}
```

#### (b) CUDA NVRTC translation template

```cuda
extern "C" __global__ void vokra_wavtokenizer_vq_gather_f32(
    const unsigned int* codes,
    const float*        table,
    float*              out,
    unsigned int        vocab_size,   // 未使用: 検証は host 側で行う
    unsigned int        d_model,
    unsigned int        time)
{
    unsigned int delem = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int t     = blockIdx.y * blockDim.y + threadIdx.y;
    if (t >= time || delem >= d_model) {
        return;
    }
    unsigned int idx = codes[t];
    out[t * d_model + delem] = table[idx * d_model + delem];
}
```

#### (c) Launch config

- grid: `(d_model.div_ceil(BLOCK), time.div_ceil(BLOCK), 1)` with `BLOCK = 16`
- block: `(BLOCK, BLOCK, 1)`
- shared memory: 0

#### (d) Parity target

- **Bound**: `atol = 0.0` — **bit-identical**（純 gather、no reduction、no FMA、no transcendental）。ここで実測 `|Δ|` が 0 でなければ kernel 実装 or launch 引数 bug。
- **Reference**: `vokra_ops::fsq_codec::wavtokenizer_vq_decode`

---

### 3.4 `vokra_xcodec2_fsq_decode_f32`（WF2 / M4-16）

**Semantics**: `codes[t]` を mixed-radix decompose → per-dim grid → `has_projection` なら Linear + bias、なければ Identity

#### (a) Metal MSL source（`context.rs` L999-1052）

```metal
struct Xcodec2FsqDims {
    uint d_model;         // output width (= FsqOutProj::d_model, or = n_dims for Identity)
    uint n_dims;          // len(levels) (= X-Codec 2's 8)
    uint time;
    uint has_projection;  // 0 = Identity (d_model == n_dims), 1 = GEMV
};

kernel void vokra_xcodec2_fsq_decode_f32(
    device const uint*          codes        [[buffer(0)]],
    device const uint*          levels       [[buffer(1)]],
    device const float*         proj_weight  [[buffer(2)]],
    device const float*         proj_bias    [[buffer(3)]],
    device float*               out          [[buffer(4)]],
    constant Xcodec2FsqDims&    d            [[buffer(5)]],
    uint2                       gid          [[thread_position_in_grid]])
{
    const uint t     = gid.y;
    const uint delem = gid.x;
    if (t >= d.time || delem >= d.d_model) {
        return;
    }

    if (d.has_projection != 0u) {
        uint rem = codes[t];
        float acc = proj_bias[delem];
        const uint w_base = delem * d.n_dims;
        for (uint k = 0; k < d.n_dims; ++k) {
            const uint level = levels[k];
            const uint level_index = rem % level;
            rem /= level;
            const uint half_width = level / 2u;  // >= 1: host validates level >= 2
            const float grid_val =
                (float)((int)level_index - (int)half_width) / (float)half_width;
            acc += proj_weight[w_base + k] * grid_val;
        }
        out[t * d.d_model + delem] = acc;
    } else {
        uint rem = codes[t];
        for (uint k = 0; k <= delem; ++k) {
            const uint level = levels[k];
            const uint level_index = rem % level;
            if (k == delem) {
                const uint half_width = level / 2u;
                out[t * d.d_model + delem] =
                    (float)((int)level_index - (int)half_width) / (float)half_width;
                return;
            }
            rem /= level;
        }
    }
}
```

#### (b) CUDA NVRTC translation template

```cuda
extern "C" __global__ void vokra_xcodec2_fsq_decode_f32(
    const unsigned int* codes,
    const unsigned int* levels,
    const float*        proj_weight,
    const float*        proj_bias,
    float*              out,
    unsigned int        d_model,
    unsigned int        n_dims,
    unsigned int        time,
    unsigned int        has_projection)
{
    unsigned int delem = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int t     = blockIdx.y * blockDim.y + threadIdx.y;
    if (t >= time || delem >= d_model) {
        return;
    }

    if (has_projection != 0u) {
        unsigned int rem = codes[t];
        float acc = proj_bias[delem];
        unsigned int w_base = delem * n_dims;
        for (unsigned int k = 0; k < n_dims; ++k) {
            unsigned int level = levels[k];
            unsigned int level_index = rem % level;
            rem /= level;
            unsigned int half_width = level / 2u;  // >= 1: host validates level >= 2
            float grid_val =
                (float)((int)level_index - (int)half_width) / (float)half_width;
            acc += proj_weight[w_base + k] * grid_val;
        }
        out[t * d_model + delem] = acc;
    } else {
        unsigned int rem = codes[t];
        for (unsigned int k = 0; k <= delem; ++k) {
            unsigned int level = levels[k];
            unsigned int level_index = rem % level;
            if (k == delem) {
                unsigned int half_width = level / 2u;
                out[t * d_model + delem] =
                    (float)((int)level_index - (int)half_width) / (float)half_width;
                return;
            }
            rem /= level;
        }
    }
}
```

#### (c) Launch config

- grid: `(d_model.div_ceil(BLOCK), time.div_ceil(BLOCK), 1)` with `BLOCK = 16`
- block: `(BLOCK, BLOCK, 1)`
- shared memory: 0
- **注意**: `proj_weight` / `proj_bias` は `has_projection == 0` の時 dummy `[0.0f]` を渡す（Metal 側と同じ pattern。CUDA では `cuMemAlloc(4)` で 4-byte 確保 → `cuMemsetD8` で 0 埋め）。

#### (d) Parity target

- **Bound**: `atol ≤ 5e-4`（GEMV path、fast-math 再結合）／Identity path は `atol = 0.0`（純算術、no reduction）
- **Reference**: `vokra_ops::fsq_codec::xcodec2_fsq_decode`

---

### 3.5 `vokra_snake_activation_f32`（WF2）

**Semantics**: `out[c, t] = x[c, t] + (1 / (alpha[c] + 1e-9)) * sin(alpha[c] * x[c, t])^2`

#### (a) Metal MSL source（`context.rs` L1200-1218）

```metal
struct SnakeActivationDims {
    uint channels;
    uint time;
};

kernel void vokra_snake_activation_f32(
    device const float*                x     [[buffer(0)]],
    device const float*                alpha [[buffer(1)]],
    device float*                      out   [[buffer(2)]],
    constant SnakeActivationDims&      d     [[buffer(3)]],
    uint2                              gid   [[thread_position_in_grid]])
{
    const uint t = gid.x;
    const uint c = gid.y;
    if (c >= d.channels || t >= d.time) {
        return;
    }
    const float a     = alpha[c];
    const float inv_a = 1.0f / (a + 1.0e-9f);
    const uint  idx   = c * d.time + t;
    const float v     = x[idx];
    const float s     = sin(a * v);
    out[idx] = v + inv_a * s * s;
}
```

#### (b) CUDA NVRTC translation template

```cuda
extern "C" __global__ void vokra_snake_activation_f32(
    const float* x,
    const float* alpha,
    float*       out,
    unsigned int channels,
    unsigned int time)
{
    unsigned int t = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int c = blockIdx.y * blockDim.y + threadIdx.y;
    if (c >= channels || t >= time) {
        return;
    }
    float a     = alpha[c];
    float inv_a = 1.0f / (a + 1.0e-9f);
    unsigned int idx = c * time + t;
    float v = x[idx];
    float s = sinf(a * v);          // MSL sin -> CUDA sinf (FP32)
    out[idx] = v + inv_a * s * s;
}
```

#### (c) Launch config

- grid: `(time.div_ceil(BLOCK), channels.div_ceil(BLOCK), 1)` with `BLOCK = 16` — **note**: MSL は `time` を `gid.x` に置いた（row-major fast axis）。CUDA も同じく `time` を `blockIdx.x` へ。
- block: `(BLOCK, BLOCK, 1)`
- shared memory: 0

#### (d) Parity target

- **Bound**: `atol ≤ 5e-4`（`sinf` の transcendental gap 由来。Metal MSL の `sin` と CUDA `sinf` は libdevice 実装が異なるので、bit-identical は保証しない — CPU との差はそれぞれ 5e-4 以下、両 GPU 間の差も同 order）
- **Reference**: `vokra_ops::snake_activation_f32`

---

### 3.6 `vokra_snac_decode_f32`（WF5）

**Semantics**: 3-stage hierarchical RVQ、各 stage の temporal stride `[4, 2, 1]`（24 kHz canonical）を `t_stage = t_out / stride` で index

#### (a) Metal MSL source（`context.rs` L1124-1158）

```metal
struct SnacDecodeDims {
    uint d_model;
    uint codebook_dim;
    uint codebook_size;
    uint t_expanded;
    uint strides[3];           // 24 kHz canonical = [4, 2, 1]
    uint stage_offsets[3];     // start of each stage in flat codes buffer
};

kernel void vokra_snac_decode_f32(
    device const uint*         codes         [[buffer(0)]],
    device const float*        codebooks     [[buffer(1)]],
    device const float*        proj_weights  [[buffer(2)]],
    device const float*        proj_biases   [[buffer(3)]],
    device float*              out           [[buffer(4)]],
    constant SnacDecodeDims&   d             [[buffer(5)]],
    uint2                      gid           [[thread_position_in_grid]])
{
    const uint t_out = gid.y;
    const uint d_out = gid.x;
    if (t_out >= d.t_expanded || d_out >= d.d_model) {
        return;
    }
    const uint cb_stride = d.codebook_size * d.codebook_dim;
    const uint w_stride  = d.d_model * d.codebook_dim;
    float acc = 0.0f;
    for (uint s = 0; s < 3u; ++s) {
        const uint stride_s = d.strides[s];
        const uint t_stage  = t_out / stride_s;
        const uint idx      = codes[d.stage_offsets[s] + t_stage];
        const uint low_off  = s * cb_stride + idx * d.codebook_dim;
        const uint w_off    = s * w_stride + d_out * d.codebook_dim;
        float y = proj_biases[s * d.d_model + d_out];
        for (uint c = 0; c < d.codebook_dim; ++c) {
            y += proj_weights[w_off + c] * codebooks[low_off + c];
        }
        acc += y;
    }
    out[t_out * d.d_model + d_out] = acc;
}
```

#### (b) CUDA NVRTC translation template

**注意**: MSL の `constant SnacDecodeDims& d [[buffer(5)]]` は 3-element uint 配列
2 個を含む。NVRTC で「struct 全定義」を渡すか「配列を個別 uint arg 6 個に展開」
の 2 択。既存 CUDA kernel は個別 arg pattern なので、以下は展開版:

```cuda
extern "C" __global__ void vokra_snac_decode_f32(
    const unsigned int* codes,
    const float*        codebooks,
    const float*        proj_weights,
    const float*        proj_biases,
    float*              out,
    unsigned int        d_model,
    unsigned int        codebook_dim,
    unsigned int        codebook_size,
    unsigned int        t_expanded,
    unsigned int        stride_0,
    unsigned int        stride_1,
    unsigned int        stride_2,
    unsigned int        stage_offset_0,
    unsigned int        stage_offset_1,
    unsigned int        stage_offset_2)
{
    unsigned int d_out = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int t_out = blockIdx.y * blockDim.y + threadIdx.y;
    if (t_out >= t_expanded || d_out >= d_model) {
        return;
    }
    unsigned int cb_stride = codebook_size * codebook_dim;
    unsigned int w_stride  = d_model * codebook_dim;

    // Stage tables — kept as local arrays so the loop can index them.
    unsigned int strides[3]       = { stride_0, stride_1, stride_2 };
    unsigned int stage_offsets[3] = { stage_offset_0, stage_offset_1, stage_offset_2 };

    float acc = 0.0f;
    for (unsigned int s = 0; s < 3u; ++s) {
        unsigned int stride_s = strides[s];
        unsigned int t_stage  = t_out / stride_s;
        unsigned int idx      = codes[stage_offsets[s] + t_stage];
        unsigned int low_off  = s * cb_stride + idx * codebook_dim;
        unsigned int w_off    = s * w_stride + d_out * codebook_dim;
        float y = proj_biases[s * d_model + d_out];
        for (unsigned int c = 0; c < codebook_dim; ++c) {
            y += proj_weights[w_off + c] * codebooks[low_off + c];
        }
        acc += y;
    }
    out[t_out * d_model + d_out] = acc;
}
```

#### (c) Launch config

- grid: `(d_model.div_ceil(BLOCK), t_expanded.div_ceil(BLOCK), 1)` with `BLOCK = 16`
- block: `(BLOCK, BLOCK, 1)`
- shared memory: 0
- **注意**: `strides` / `stage_offsets` は host 側で validate（SNAC 24kHz: strides `[4, 2, 1]`、stage_offsets は前 stage 累積長）→ 3 個ずつ scalar uint で渡す（15 args total）。

#### (d) Parity target

- **Bound**: `atol ≤ 5e-4`（inner GEMV の fast-math 再結合）
- **Reference**: `vokra_ops::snac_decode::SnacDecoder::decode`

---

### 3.7 `vokra_denoise_apply_mask_f32`（WF5）

**Semantics**: `out_re = spec_re * gain`, `out_im = spec_im * gain`（element-wise、reduction 無し）

#### (a) Metal MSL source（`context.rs` L1472-1490）

```metal
struct DenoiseApplyMaskDims {
    uint n_bins;
    uint n_frames;
};

kernel void vokra_denoise_apply_mask_f32(
    device const float*                spec_re [[buffer(0)]],
    device const float*                spec_im [[buffer(1)]],
    device const float*                gain    [[buffer(2)]],
    device float*                      out_re  [[buffer(3)]],
    device float*                      out_im  [[buffer(4)]],
    constant DenoiseApplyMaskDims&     d       [[buffer(5)]],
    uint2                              gid     [[thread_position_in_grid]])
{
    const uint f = gid.x;
    const uint t = gid.y;
    if (f >= d.n_bins || t >= d.n_frames) {
        return;
    }
    const uint  idx = t * d.n_bins + f;
    const float g   = gain[idx];
    out_re[idx] = spec_re[idx] * g;
    out_im[idx] = spec_im[idx] * g;
}
```

#### (b) CUDA NVRTC translation template

```cuda
extern "C" __global__ void vokra_denoise_apply_mask_f32(
    const float* spec_re,
    const float* spec_im,
    const float* gain,
    float*       out_re,
    float*       out_im,
    unsigned int n_bins,
    unsigned int n_frames)
{
    unsigned int f = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int t = blockIdx.y * blockDim.y + threadIdx.y;
    if (f >= n_bins || t >= n_frames) {
        return;
    }
    unsigned int idx = t * n_bins + f;
    float g = gain[idx];
    out_re[idx] = spec_re[idx] * g;
    out_im[idx] = spec_im[idx] * g;
}
```

#### (c) Launch config

- grid: `(n_bins.div_ceil(BLOCK), n_frames.div_ceil(BLOCK), 1)` with `BLOCK = 16`
- block: `(BLOCK, BLOCK, 1)`
- shared memory: 0

#### (d) Parity target

- **Bound**: `atol = 0.0` — **bit-identical**。IEEE-754 correctly-rounded multiply のみ、no reduction / no FMA opportunity / no transcendental。ここで `|Δ|` が 0 でなければ kernel 実装 or launch 引数 bug（早期発見の tripwire）。
- **Reference**: `vokra_ops::denoise::denoise_apply_mask_f32`

---

### 3.8 `vokra_qwen3_tts_codec_decode_f32`（WF5）

**Semantics**: hybrid semantic + acoustic RVQ、semantic quantizer は larger vocab（4096）、acoustic は 2048。FP32 fold across all quantizers

#### (a) Metal MSL source（`context.rs` L1551-1584）

```metal
struct Qwen3TtsCodecDims {
    uint num_quantizers;
    uint num_semantic_quantizers;
    uint semantic_codebook_size;
    uint codebook_size;
    uint codebook_dim;
    uint time;
};

kernel void vokra_qwen3_tts_codec_decode_f32(
    device const uint*                 codes            [[buffer(0)]],
    device const float*                semantic_tables  [[buffer(1)]],
    device const float*                acoustic_tables  [[buffer(2)]],
    device float*                      out              [[buffer(3)]],
    constant Qwen3TtsCodecDims&        d                [[buffer(4)]],
    uint2                              gid              [[thread_position_in_grid]])
{
    const uint t     = gid.y;
    const uint delem = gid.x;
    if (t >= d.time || delem >= d.codebook_dim) {
        return;
    }
    const uint code_base     = t * d.num_quantizers;
    const uint sem_cb_stride = d.semantic_codebook_size * d.codebook_dim;
    const uint ac_cb_stride  = d.codebook_size          * d.codebook_dim;
    float acc = 0.0f;
    for (uint q = 0; q < d.num_quantizers; ++q) {
        const uint idx = codes[code_base + q];
        if (q < d.num_semantic_quantizers) {
            const uint off = q * sem_cb_stride + idx * d.codebook_dim + delem;
            acc += semantic_tables[off];
        } else {
            const uint ac_q = q - d.num_semantic_quantizers;
            const uint off  = ac_q * ac_cb_stride + idx * d.codebook_dim + delem;
            acc += acoustic_tables[off];
        }
    }
    out[t * d.codebook_dim + delem] = acc;
}
```

#### (b) CUDA NVRTC translation template

```cuda
extern "C" __global__ void vokra_qwen3_tts_codec_decode_f32(
    const unsigned int* codes,
    const float*        semantic_tables,
    const float*        acoustic_tables,
    float*              out,
    unsigned int        num_quantizers,
    unsigned int        num_semantic_quantizers,
    unsigned int        semantic_codebook_size,
    unsigned int        codebook_size,
    unsigned int        codebook_dim,
    unsigned int        time)
{
    unsigned int delem = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int t     = blockIdx.y * blockDim.y + threadIdx.y;
    if (t >= time || delem >= codebook_dim) {
        return;
    }
    unsigned int code_base     = t * num_quantizers;
    unsigned int sem_cb_stride = semantic_codebook_size * codebook_dim;
    unsigned int ac_cb_stride  = codebook_size          * codebook_dim;
    float acc = 0.0f;
    for (unsigned int q = 0; q < num_quantizers; ++q) {
        unsigned int idx = codes[code_base + q];
        if (q < num_semantic_quantizers) {
            unsigned int off = q * sem_cb_stride + idx * codebook_dim + delem;
            acc += semantic_tables[off];
        } else {
            unsigned int ac_q = q - num_semantic_quantizers;
            unsigned int off  = ac_q * ac_cb_stride + idx * codebook_dim + delem;
            acc += acoustic_tables[off];
        }
    }
    out[t * codebook_dim + delem] = acc;
}
```

#### (c) Launch config

- grid: `(codebook_dim.div_ceil(BLOCK), time.div_ceil(BLOCK), 1)` with `BLOCK = 16`
- block: `(BLOCK, BLOCK, 1)`
- shared memory: 0
- **注意**: `semantic_tables` / `acoustic_tables` は片方が空でも 4-byte dummy を渡す（Metal buffer と同 rule。CUDA では `cuMemAlloc(4)` + `cuMemsetD8(ptr, 0, 4)`）。

#### (d) Parity target

- **Bound**: `atol ≤ 5e-4`（FP32 fold の fast-math 再結合）— canonical shape で `|Δ| = 0` になる場合が多い
- **Reference**: `vokra_ops::qwen3_tts_codec::qwen3_tts_codec_decode`

---

## 4. Additional WF6 primitives（付録、audit 8 に含まれず）

2026-08-14 land の `snake_beta` / `sinegen_deterministic` / `anti_aliased_upsample`
3 種も同 pattern で CUDA mirror 対象。以下は MSL source ポインタと NVRTC 翻訳
差分だけ抜粋（body は section 3 と同じ翻訳規則）。

### 4.1 `vokra_snake_beta_f32`（`context.rs` L1258-1278）

α と β を分離した SnakeBeta。Snake activation の 2 変数版:

```
out[c, t] = x[c, t] + (1 / (beta[c] + 1e-9)) * sin(alpha[c] * x[c, t])^2
```

NVRTC signature:
```cuda
extern "C" __global__ void vokra_snake_beta_f32(
    const float* x, const float* alpha, const float* beta, float* out,
    unsigned int channels, unsigned int time);
```
Launch: `(time.div_ceil(16), channels.div_ceil(16), 1)` × `(16, 16, 1)`。Parity: `atol ≤ 5e-4`。

### 4.2 `vokra_sinegen_deterministic_f32`（`context.rs` L1331-1354）

**注意**: MSL 側は `grid_1d(h1)`（1-D dispatch、`h1 = harmonic_num + 1`）で各
thread が「1 harmonic × time axis 全走査」を担当。**per-utterance で
`h1` は 1-10 order** ゆえ、CUDA 側も `grid = (h1.div_ceil(BLOCK_1D) as
c_uint, 1, 1)` × `block = (BLOCK_1D, 1, 1)` の 1-D dispatch。**MSL の
`SinegenDeterministicDims` struct は `float` を含む**（`samp_rate_f` /
`sine_amp` / `voiced_threshold`）ゆえ、NVRTC 展開時に `float` として arg 化:

```cuda
extern "C" __global__ void vokra_sinegen_deterministic_f32(
    const float* f0, float* out,
    unsigned int t, unsigned int h1,
    float samp_rate_f, float sine_amp, float voiced_threshold);
```

Kernel body の `sin(theta)` → `sinf(theta)`、`floor(cs)` → `floorf(cs)`。

Parity: `atol ≤ 5e-4`（`sinf` の transcendental gap）。

### 4.3 `vokra_anti_aliased_upsample_f32`（`context.rs` L1396-1428）

Polyphase upsample-then-FIR。Kernel は host が設計済の `kernel` taps を消費。
Bounded per-branch tap walk（`for (uint j = 0u; ; ++j)` with `break` on
`k_idx >= taps` or `j > t`）→ CUDA も same。

**注意**: MSL は buffer 変数名 `kernel [[buffer(1)]]` を使うが、これは CUDA
`__global__` 予約語衝突を避けるため NVRTC 側では `kernel_` に rename する
（MSL 側の `kernel_` naming と一致させる）:

```cuda
extern "C" __global__ void vokra_anti_aliased_upsample_f32(
    const float* x, const float* kernel_, float* out,
    unsigned int channels, unsigned int time_in, unsigned int time_out,
    unsigned int ratio, unsigned int taps);
```

Launch: `(time_out.div_ceil(16), channels.div_ceil(16), 1)` × `(16, 16, 1)`。
Parity: `atol ≤ 1e-4`（MSL コメント記載の bound。CPU 側は strict left-fold、
CUDA の fast-math が `fma` に融合するので ~1 ULP × taps drift）。

---

## 5. vast.ai setup（CUDA-half bakeoff 必須手順）

**総論** は `docs/handoff/vast-ai-large-model-publish.md` §2〜§4 に集約されている。
本 doc は kernel bakeoff に特有の gotcha だけ抜粋:

### 5.1 Instance recipe

| 項目 | 推奨値 | 備考 |
|---|---|---|
| Image | `nvidia/cuda:13.0.0-devel-ubuntu22.04` or `pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel` | CUDA >= 12.0 必須（既存 NVRTC pattern の compute_89 default に一致） |
| GPU | **RTX 4090**（既存 M2-03 / M3-01 vast.ai 検証と同型） | H100 は FA v3 用ゆえ本 wave では不要 |
| RAM | 16 GB 以上 | NVRTC compile + kernel bakeoff の working set は小さい（>10 GB モデル load は不要 — 本 wave は kernel の parity harness のみ、実 model 経路は含まない） |
| Disk | 50 GB 以上 | Vokra release build + PTX cache |
| Network | 非従量課金 | HF DL は不要（本 wave 対象 kernel はどれも synthetic parity で verify 可） |

### 5.2 必須の provision gotcha（既存 memory 参照）

1. **`hf_config.pth` shim の除去**（memory `[[reference-vast-ai-hf-config-pth-shim]]`）:
   - `nvidia/cuda:13.0.0` image は Python startup shim を仕込み、`HF_ENDPOINT`
     を malicious mirror `117.175.104.83:8081` に上書きする。本 wave で HF DL
     は不要だが、tools/parity Python サブツリーを uv 化する際に依然踏む risk。
   - `provision.sh` Wave 12 で対応（`rm /path/to/hf_config.pth` + certifi CA
     再植え付け）。
2. **`huggingface_hub<0.30` pin**（memory `[[reference-huggingface-hub-lt-030-vast-ai]]`）:
   - vast.ai 上のみ pin。1.x xet-token routing が mirror 404 を投げる。**本 wave
     で HF DL 不使用ゆえ実質不要**、但し `uv add huggingface_hub` を叩く時は
     `huggingface_hub<0.30` を明示。
3. **Python は uv で管理**（memory `[[feedback-python-uses-uv]]`）+ **Python 3.12
   pin**（memory `[[feedback-python-3-12]]`）:
   - `uv python pin 3.12` + `requires-python ">=3.12"` を per-tree に置く。

### 5.3 Verify build

```bash
# vast.ai 上、CUDA-only feature build（メモリセーフ = 1 crate at a time）
CARGO_BUILD_JOBS=1 cargo check -p vokra-backend-cuda --features cuda

# 続けて test（build 成功後）
CARGO_BUILD_JOBS=1 cargo test -p vokra-backend-cuda --features cuda -- --nocapture

# Compute seam の CUDA arm も verify
CARGO_BUILD_JOBS=1 cargo test -p vokra-models --features cuda -- --nocapture

# ⚠ M1 iMac（16 GB）では `--features cuda` の build 不可（cudarc 不在 + libcuda 不在 + Apple silicon）。
# ⚠ vast.ai 上でも `--workspace` や `--all-features` は禁止（Metal + CUDA 同時 = zero-dep
#    build 経路が 2 系統走り memory 危険）。per-crate + `--features cuda` 単独が正。
```

**Expected**:
- `cargo check`: 全 8 kernel を `KERNELS_CUDA` に足し `Modules` struct に
  `CUfunction` field を足し `load_modules` で resolve を書き足せば通る。
- `cargo test -p vokra-backend-cuda --features cuda`: NVRTC compile が成功
  すれば全 kernel の PTX 化 + module load が通る。既存 `parity_kernels_cuda.rs`
  pattern で新規 kernel の bit-identical vs CPU parity test を per-kernel 追加
  推奨（section 3 の parity target を tripwire として使う）。
- `cargo test -p vokra-models --features cuda`: `csm_gpu_session.rs` /
  `moshi_duplex.rs` / `parity_whisper.rs` 等の既存 CUDA gate は無変更で通る
  （追加した新 kernel は既存モデルの hot path に自動配線されない — Metal 半分と
  同様、モデル側の `Compute::mimi_rvq_f32` 経由呼び出しが新 CUDA arm を選ぶ）。

### 5.4 vast.ai lifecycle（trap-based auto-destroy）

`docs/handoff/vast-ai-large-model-publish.md` §4.3 の trap pattern を踏襲:

```bash
# rent → provision → work → destroy の 1-shot script
INSTANCE_ID=$(vastai launch instance ... --json | jq -r '.new_contract')
trap "vastai destroy instance $INSTANCE_ID" EXIT INT TERM
# ... provision + work ...
# 明示的 destroy は不要（trap が発火）
```

---

## 6. Test invocation on vast.ai

### 6.1 Full bakeoff cycle（推奨、per-kernel）

```bash
# 1. build check（NVRTC PTX compile が成功するか）
CARGO_BUILD_JOBS=1 cargo build -p vokra-backend-cuda --features cuda

# 2. 単体 kernel unit test（既存 parity_kernels_cuda.rs pattern を横展開）
CARGO_BUILD_JOBS=1 cargo test -p vokra-backend-cuda --features cuda \
    mimi_rvq -- --nocapture

# 3. Compute seam integration test
CARGO_BUILD_JOBS=1 cargo test -p vokra-models --features cuda \
    mimi_rvq -- --nocapture
```

### 6.2 Parity harness template

既存 Metal parity `crates/vokra-models/tests/mimi_rvq_metal_bit_identical.rs`
を CUDA 版に mirror:

```rust
// crates/vokra-models/tests/mimi_rvq_cuda_bit_identical.rs（新規）
#[cfg(not(all(feature = "cuda", any(unix, windows))))]
mod off_feature {
    use vokra_core::{BackendKind, VokraError};
    use vokra_models::compute::{Compute, HotOp};

    #[test]
    fn for_backend_cuda_mimi_rvq_off_feature_is_backend_unavailable() {
        let err = Compute::for_backend(BackendKind::Cuda, &[HotOp::MimiRvq])
            .expect_err("off-feature CUDA must fail explicitly, not silently CPU-substitute");
        assert!(
            matches!(err, VokraError::BackendUnavailable(_)),
            "expected BackendUnavailable off the cuda feature, got {err:?}",
        );
    }
}

#[cfg(all(feature = "cuda", any(unix, windows)))]
mod cuda_band {
    // Section 3.1 の parity target: atol ≤ 5e-4（canonical で bit-identical）
    const ATOL: f32 = 5e-4;
    // ... Metal 版と同じ tiny_shape / real-parity / negative_control 3 テスト ...
}
```

**Parity 表**（section 3 の atol summary、CI red-line 一覧）:

| kernel | atol | 由来 |
|---|---|---|
| `mimi_rvq` | 5e-4 | FP32 fold の fast-math 再結合 |
| `dac_rvq` | 5e-4 | 同上 + inner GEMV |
| `wavtokenizer_vq` | **0.0** | 純 gather（bit-identical tripwire） |
| `xcodec2_fsq` GEMV | 5e-4 | inner GEMV fast-math |
| `xcodec2_fsq` Identity | **0.0** | 純算術 |
| `snake_activation` | 5e-4 | `sinf` transcendental gap |
| `snac_decode` | 5e-4 | inner GEMV fast-math |
| `denoise_apply_mask` | **0.0** | element-wise multiply（bit-identical tripwire） |
| `qwen3_tts_codec_decode` | 5e-4 | FP32 fold fast-math |

**Note**: `0.0` bound は「bit-identical」を意味する（IEEE-754 correctly-rounded
op のみで構成）。ここが 0 でなければ実装 bug の tripwire。

---

## 7. CI scaffold template

**Note**: CUDA CI は self-hosted runner を要する（GitHub Actions Linux hosted
runner は GPU なし、Vokra は zero-dep で cudarc / cuBLAS 不使用ゆえ「NVRTC
compile → PTX → real GPU launch」の cycle が host GPU 必須）。**本 handoff の
scaffold は default branch に足しても runner 側で自動 skip される** 構造で書く。

### 7.1 `.github/workflows/parity-vocoder-cuda.yml`（新規、owner 側で `git add`）

```yaml
name: parity-vocoder-cuda
# CUDA vocoder kernel parity — self-hosted vast.ai runner が provision されて
# いるときのみ実行、GitHub-hosted runner では skip clean（never fabricated pass）。
#
# 発火:
#   - workflow_dispatch: owner が手動起動
#   - schedule: 週次（cron '0 5 * * 1' = 月曜 05:00 UTC）
#
# ready 判定: setup job が self-hosted CUDA runner の presence を probe し、
# 不在なら parity leg 全 kernel を "runner absent" annotation で clean skip
# （Kokoro parity CI と同 pattern）。

on:
  workflow_dispatch:
  schedule:
    - cron: '0 5 * * 1'

jobs:
  setup:
    runs-on: ubuntu-latest
    outputs:
      cuda_ready: ${{ steps.probe.outputs.cuda_ready }}
    steps:
      - name: Probe CUDA self-hosted runner availability
        id: probe
        run: |
          # Placeholder: 本物の probe は `curl` で self-hosted vast.ai runner の
          # heartbeat endpoint を叩く（owner-provisioned infra 依存）。
          # デフォルトは "cuda_ready=false" で clean skip。
          echo "cuda_ready=false" >> "$GITHUB_OUTPUT"

  parity-vocoder-cuda:
    needs: setup
    if: needs.setup.outputs.cuda_ready == 'true'
    runs-on: [self-hosted, linux, x64, cuda, rtx-4090]  # owner が provision したラベル
    strategy:
      fail-fast: false
      matrix:
        kernel:
          - mimi_rvq
          - dac_rvq
          - wavtokenizer_vq
          - xcodec2_fsq
          - snake_activation
          - snac_decode
          - denoise_apply_mask
          - qwen3_tts_codec_decode
    steps:
      - uses: actions/checkout@v4
      - name: Verify NVRTC compile
        run: |
          CARGO_BUILD_JOBS=1 cargo build -p vokra-backend-cuda --features cuda
      - name: Run per-kernel parity harness
        run: |
          CARGO_BUILD_JOBS=1 cargo test -p vokra-backend-cuda --features cuda \
              ${{ matrix.kernel }} -- --nocapture
      - name: Compute seam integration
        run: |
          CARGO_BUILD_JOBS=1 cargo test -p vokra-models --features cuda \
              ${{ matrix.kernel }} -- --nocapture

  clean-skip-report:
    needs: setup
    if: needs.setup.outputs.cuda_ready == 'false'
    runs-on: ubuntu-latest
    steps:
      - name: Emit "self-hosted CUDA runner absent" annotation
        run: |
          echo "::warning::parity-vocoder-cuda: self-hosted CUDA runner not \
              provisioned; skipping bakeoff (owner action required to enable)."
```

### 7.2 CI 政策（初期）

- **required check には入れない**（M2-14 self-hosted runner + M3-01 5%
  regression gate と同じ defer 政策、`docs/m2-cuda-rtf-variance-2026-07-08.md`
  §D6 の red line 継承）。数週の連続緑後に owner 判断で promote。
- **fabricated pass 禁止**: `cuda_ready=false` のとき matrix leg は run されず、
  代わりに `clean-skip-report` job が warning annotation を吐いて「本当に skip」
  であることを可視化。Kokoro CI（`parity-kokoro-real.yml`）+ Whisper CI
  （`parity-whisper-real.yml`）と同じ pattern。

---

## 8. Owner critical path checklist

以下は本 handoff が引き渡す owner 作業。順序推奨、per-kernel の依存はなく
independent parallel 可（8 kernel を 1 kernel ずつ land + verify で cycle）。

- [ ] **vast.ai instance provision**（section 5、既存 `docs/handoff/vast-ai-large-model-publish.md` §2 手順、trap-based auto-destroy 込み）
- [ ] `crates/vokra-backend-cuda/src/context.rs` の `KERNELS_CUDA` const string 末尾に 8 kernel（+ 付録 3 kernel）を追記
- [ ] 同ファイル `Modules` struct（L5698）に 8 個の `CUfunction` field 追加
- [ ] 同ファイル `load_modules`（L5785）で 8 個の `get_function(...)` 追加
- [ ] 同ファイル `impl CudaContext` に 8 個の `pub fn *_f32(...)` wrapper 追加（Metal wrapper と shape 一致、host-side shape / index bound check 込み）
- [ ] `crates/vokra-models/src/compute.rs` の 8 個の `Be::Cuda(_) => Err(VokraError::UnsupportedOp(...))` arm を `Be::Cuda(ctx) => ctx.*_f32(...)` に差替え
- [ ] 8 個の parity test（`crates/vokra-models/tests/*_cuda_bit_identical.rs`）を Metal 版 mirror として新規追加、section 3 の atol target を tripwire に使用
- [ ] vast.ai 上で `CARGO_BUILD_JOBS=1 cargo test -p vokra-backend-cuda --features cuda -- --nocapture` + `cargo test -p vokra-models --features cuda -- --nocapture` を実行し全 kernel bit-identical vs CPU 実測
- [ ] `docs/abi-changelog.md` に「CUDA arm 追加（Rust surface のみ、新規 C ABI ゼロ）」を additive entry として記録（既存 v1.0-rc baseline 33 fn + 11 typedef 不変）
- [ ] `.github/workflows/parity-vocoder-cuda.yml` を commit（section 7.1 scaffold）、初回 `workflow_dispatch` は owner が起動
- [ ] （optional）付録 WF6 primitives 3 種（`snake_beta` / `sinegen_deterministic` / `anti_aliased_upsample`）も同 pattern で mirror

**land 分割の推奨**: 8 kernel + 3 primitives = 11 kernel 一括ではなく、
「1 kernel 実装 → NVRTC compile 成功 → CPU parity 通過 → commit」の cycle を
per-kernel 回す（M4 教訓の verify-on-actual-HEAD 規律、post-terminal wave の
patch-to-scratchpad + full verify pattern）。

---

## 9. Non-goals（本 handoff で扱わない）

- **cuBLAS / cuDNN 依存**（zero-dep NFR-DS-02 違反、NVIDIA EULA install モデル
  破綻）— 全 kernel は NVRTC 上で self-contained に書く
- **TF32 fast path**（FP32 red line 破綻）— 明示 `float` intrinsics 使用
- **`__half` / `__nv_bfloat16` accumulator**（BF16 mantissa loss = audio-dialect
  red line 違反）— FP32 accumulator throughout
- **FA v3 / Hopper WGMMA/TMA**（v1.5+ 政策違反、CLAUDE.md 設計制約 §5-(7)、
  M2-03-followup-rtf ADR）— 本 wave は FA v2 世代の kernel のみ
- **Metal 半分の再実装**（既 land、本 wave 対象外）
- **CUDA arm 経由の実 model bakeoff**（本 wave は kernel parity のみ、実
  CosyVoice2 / Moshi / CSM full pipeline の CUDA 化は別 WP、`docs/handoff/
  vast-ai-vocoder-gpu-kernels.md` §2.3 の分離判断継承）
- **HF upload / publish**（owner-only、`docs/handoff/vast-ai-large-model-
  publish.md` 経由、本 wave 無関係）
- **手動 destroy 忘れ**（trap-based auto-destroy 必須、vast.ai lifecycle red
  line）

---

## 10. 追跡 metadata

- **Audit source**: post-audit 2026-08-13、`{"status":"actionable","notes":"NONE of the 8 vocoder kernels (mimi_rvq / dac_rvq / wavtokenizer_vq / xcodec2_fsq / snake / snac / denoise / qwen3_tts_codec) are present in crates/vokra-backend-cuda/src/context.rs — grep for those 8 names returns 0 hits."}`
- **本 handoff の CC scope**: doc-only（NO speculative CUDA kernel code, NO
  `crates/vokra-backend-cuda/src/context.rs` modification, NO `--features
  cuda` local build attempt = memory-safe on M1 iMac 16 GB）
- **本 handoff の owner scope**: kernel implementation on vast.ai + parity
  verification + CI scaffold enable
- **Non-goals（session-level absolute）**: Matcha-TTS / RVC in main /
  AudioSeal forced-embed / NNAPI / Piper1-gpl / ONNX graph / Bark 2 /
  watermark embed / voice-clone in main repo (ELVIS Act separation) / HF
  upload / publish (owner only) / Number fabrication (measured only)
