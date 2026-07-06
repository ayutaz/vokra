# 量子化ポリシー設計仕様書（Quantization Policy）

- **チケット**: M2-08-T15（WP 主成果物「quantization policy 設計 + CI 配線」）
- **日付**: 2026-07-06（初版、M2-08-T15）
- **正**: この文書は二次成果物。**SSOT はモジュール rustdoc**（`crates/vokra-core/src/quant/mod.rs`）およびキー定数（`crates/vokra-core/src/gguf/chunks.rs`）。乖離した場合はコードを修正して本書に合わせる。
- **要件トレース**: FR-QT-02（policy + resolve + `vokra.quant.*` chunk）／FR-QT-03（min-dtype registry、FR-OP-10/11/12/13 audit trail）／NFR-QL-01（FP16 atol=0.01）／NFR-QL-02（MEL loss 劣化 <5%）／NFR-DS-02（zero-dep）／FR-EX-08（silent fallback 禁止）。
- **CI ゲート**（M2-08-T15、`.github/workflows/ci.yml` の `parity` job）:
  - `cargo test -p vokra-core --test quant_parity`（K-quant dequant → GEMM の F32 参照値との一致）
  - `cargo test -p vokra-cli --test policy_e2e`（converter → runtime → 5% mel-loss ゲート）
  - `cargo-deny check` は既存の `license` job で継続必須（本 WP は新規依存なし → 継続 pass）

## 1. スコープと非スコープ

**スコープ**（M2-08 で実装）:
- `QuantScheme` enum（Fp32 / Fp16 / W4A16Q4K / W4A16Q5K / W4A16Q6K / W8A8Int8）と WxAy 表記の SSOT 化。
- `QuantPolicy`（default + rules）と first-match 解決。
- `vokra.quant.*` GGUF chunk 契約（policy の唯一の永続化パス、TOML/YAML/serde-JSON は非採用）。
- 変換器での per-tensor policy 適用（whisper のみ、他コンバータは follow-up）。
- ランタイム側の policy ロードと activation 精度検査（W8A8 は kernel 未実装のため明示エラー）。
- `MinDtypeRegistry`（FR-OP-10 HiFi-GAN / FR-OP-11 BigVGAN / FR-OP-12 Vocos / FR-OP-13 Snake の audit anchor）。
- HiFi-GAN INT8 opt-in ゲート（構造上 opt-in without calibration を表現不能に）。
- `vokra-eval::check_degradation` による 5% mel-loss ゲート。

**非スコープ**（follow-up issue で追跡）:
- W8A8 INT8 GEMM kernel の各バックエンド実装（CPU AVX-VNNI / ARM SDOT-i8mm / Metal / CUDA）。
- Vocos / BigVGAN / HiFi-GAN の kernel body（consumer モデルと同期配信、FR-OP-10/11/12）。
- KV cache 量子化（FR-QT-05、v1.0）。
- piper / CAM++ / Silero converter への policy 配線。
- UTMOS / DNSMOS weight 配信（M1-09b）。
- Snake activation の FP32 internal_precision の**強制**（audit anchor のみ、強制は kernel と同期）。

## 2. スキーム表記（WxAy）

`QuantScheme` は `#[non_exhaustive]`。将来の追加（W4A8、Q2_K、IQ2 等）に対して閉じない設計。

| Variant | as_str alias | 重み dtype | activation dtype | backend supported (M2-08) |
|---|---|---|---|---|
| `Fp32` | `"fp32"` | F32 | F32 | all |
| `Fp16` | `"fp16"` | F16 | F16 | all |
| `W4A16Q4K` | `"w4a16-q4k"` | Q4_K | F16 | all（重みは load 時 F32 dequant） |
| `W4A16Q5K` | `"w4a16-q5k"` | Q5_K | F16 | all |
| `W4A16Q6K` | `"w4a16-q6k"` | Q6_K | F16 | all |
| `W8A8Int8` | `"w8a8"` | Int8 | Int8 | **none**（kernel 未実装、validate で拒否） |

エイリアス規約:
- `"w4a16"` 単体は `W4A16Q4K`（デフォルト 4-bit tier）に解決。
- `as_str` ↔ `from_alias_str` は全 variant で往復可能（T02 テスト）。
- 未知 alias は `VokraError::UnknownQuantScheme { alias }` — panic なし。

**"W4A16" は 4-bit weight + 16-bit activation、"W8A8" は 8-bit weight + 8-bit activation** を意味する。M2-08 時点で activation dtype はメタデータ契約であり、実際の kernel は F32 で走る（NFR-QL-01 の FP16 atol=0.01 を満たすため）。FP16 kernel 化は follow-up。

## 3. ポリシーとルール解決

`QuantPolicy` は Rust builder API で構築（serde 不採用、NFR-DS-02）:

```rust
let policy = QuantPolicy::new(QuantScheme::W4A16Q4K)
    .with_rule(LayerPattern::Suffix(".bias".into()),        QuantScheme::Fp32)
    .with_rule(LayerPattern::Suffix(".weight_norm".into()), QuantScheme::Fp32);
```

**解決順序**（first-match）: rules を順番に走査、最初にマッチした scheme を返す。マッチなしなら `policy.default`。**hardcoded region → prefix の推論は禁止**（M0-06-T04 の tensor 命名規約を尊重、caller が name を渡す）。

**プリセット**:
- `QuantPolicy::default_vocoder_safe()` — `default=Fp16`、rules 空。全 vocoder / 未知モデルの安全側 default。
- `QuantPolicy::whisper_q4_k()` — `default=W4A16Q4K` + `.bias` / `.weight_norm` を `Fp32` 例外（現行 `is_quantizable` 挙動と一致）。

### 例 1: Whisper Q4_K プリセット

```rust
let p = QuantPolicy::whisper_q4_k();
assert_eq!(resolve(&p, "encoder.blocks.0.mlp.0.weight"), QuantScheme::W4A16Q4K);
assert_eq!(resolve(&p, "encoder.blocks.0.mlp.0.bias"),   QuantScheme::Fp32);
assert_eq!(resolve(&p, "encoder.ln_post.weight_norm"),   QuantScheme::Fp32);
```

### 例 2: Vocoder safe

```rust
let p = QuantPolicy::default_vocoder_safe();
assert_eq!(resolve(&p, "generator.ups.0.weight"),   QuantScheme::Fp16);
assert_eq!(resolve(&p, "snake.alpha"),              QuantScheme::Fp16);
```

### 例 3: HiFi-GAN INT8 opt-in（唯一の構築パス）

```rust
let cal = CalibrationRef::from_blob_handle("hifigan-int8-cal-v1");
let p = QuantPolicy::new(QuantScheme::Fp16)
    .with_rule(LayerPattern::Prefix("generator.".into()), QuantScheme::W8A8Int8)
    .with_hifigan_int8_opt_in(cal);  // opt_in + calibration をアトミックにセット
```

`with_hifigan_int8_opt_in(calibration)` **のみが** `opt_in=true` を立てる。単独の `opt_in` setter は存在しない → 「opt-in なのに calibration=None」を型で不可能に。

## 4. `vokra.quant.*` GGUF chunk

`QuantPolicy` の唯一の永続化パス。TOML/YAML/serde-JSON パーサは非採用（NFR-DS-02、CLI 引数は既存の hand-rolled `while` ループ + `--policy-preset` プリセット名で受け渡し）。

| キー | GGUF value type | 意味 |
|---|---|---|
| `vokra.quant.default_scheme` | STRING | `policy.default` の alias 文字列 |
| `vokra.quant.rule_count` | UINT64 | rules 数（0 可） |
| `vokra.quant.rule.{i}.pattern_kind` | STRING | `"exact" \| "prefix" \| "suffix" \| "glob"` |
| `vokra.quant.rule.{i}.pattern` | STRING | パターン payload |
| `vokra.quant.rule.{i}.scheme` | STRING | scheme alias |
| `vokra.quant.hifigan_int8_opt_in` | BOOL | HiFi-GAN INT8 opt-in（default false） |
| `vokra.quant.hifigan_int8_calibration_ref` | STRING（optional） | calibration blob handle |
| `vokra.quant.min_dtype_enforced` | ARRAY<STRING>（optional） | validate 済み op 名（監査記録） |

round-trip: `QuantPolicy::write_to_gguf_builder(&mut b)` → parse via `QuantPolicy::from_gguf(&gguf)` → equality。全 variant × 3 プリセットで T05 テスト。

chunk 不在の GGUF は `QuantPolicy::default_vocoder_safe()` として扱う（後方互換、既存 whisper GGUF がそのまま読める）。

## 5. `MinDtypeRegistry` — mechanism 先行

FR-OP-10/11/12/13 の kernel body が存在する前に**制約だけ**登録する。precedent: `crates/vokra-core/src/ir/fusion/patterns/snake.rs`（`Conv1dSnakePattern::new()` を空 impl で先出し）。

| op_name | min_activation | downgrade | fr_ref |
|---|---|---|---|
| `"hifigan_generator"` | Fp16 | `HifiganOptIn` | FR-OP-10 |
| `"bigvgan_generator"` | Fp16 | `Forbidden` | FR-OP-11 |
| `"vocos_head"` | Fp16 | `Forbidden` | FR-OP-12 |
| `"snake_activation"` | Fp32 | `Forbidden` | FR-OP-13（audit-only、強制は kernel と同期） |

**Kokoro は登録しない**。Kokoro decoder は iSTFTNet / StyleTTS 2 派生（CLAUDE.md レビュアー A 修正）で **Vocos ではない**。将来の Kokoro op-kind は `"kokoro_istft_head"` を予約。T08 テストで Kokoro 名 lookup が `None` を返すことを assert。

`validate_policy_against_model(policy, ops_in_use, registry)` は各 op について registry を参照し、resolved scheme の activation dtype が最低要件を満たすか検査。3 ケース:

1. Vocos / BigVGAN + W8A8 → `MinDtypeViolation`（`Forbidden` → opt-in flag に関わらず拒否）。
2. HiFi-GAN + W8A8 + `opt_in=false` → `MinDtypeViolation`。
3. HiFi-GAN + W8A8 + `opt_in=true` + `calibration=Some` → validate は pass、deployment 時に `check_degradation` の 5% ゲートで最終判定（T12）。

## 6. degradation ゲート（vokra-eval）

`vokra-eval::degradation::check_degradation(reference, quantized, sample_rate, threshold)` は `MelLoss::new(sample_rate, 1024, 256, 80)` を内部構築し、`(quant - ref) / max(ref, ε)` を `threshold`（NFR-QL-02 = 0.05）と比較。UTMOS は `#[cfg(feature = "utmos")]` で `NotImplemented`（M1-09b の weight 未配信）。`DegradationReport { mel_loss_ref, mel_loss_quant, relative_delta, passes_5pct_gate, mel_loss_only }` の `mel_loss_only=true` は「UTMOS 未配線」を下流に伝える partial gate flag。

HiFi-GAN INT8 opt-in（T12）:
- opt-in + eval pass → 許可。
- opt-in + eval 未実行 → `VokraError::HifiganInt8VerifyMissing`。
- opt-in + eval fail → `VokraError::HifiganInt8DegradationExceeded { delta, threshold }`。

## 7. FR-EX-08（silent fallback 禁止）の適用点

- 変換器: policy が Q4_K を要求したが `element_count % QK_K != 0` or rank < 2 → `VokraError::QuantPolicyInapplicable { tensor_name, scheme, reason }`（`is_quantizable` の silent-widen 挙動を置き換え）。
- ランタイム: `activation_dtype() == Int8` のセッション生成 → `VokraError::UnsupportedQuantPath { op, scheme, backend }`。
- validate: 前掲 `MinDtypeViolation` / `HifiganInt8VerifyMissing` / `HifiganInt8DegradationExceeded`。

## 8. CI ゲート

`.github/workflows/ci.yml` の `parity` job（既存の必須 check）に以下を追加（M2-08-T15）:

- `cargo test -p vokra-core --test quant_parity` — K-quant weight → dequantize → GEMM の F32 参照値との一致（FP16 atol=0.01）、INT8 branch は `MinDtypeViolation` / `UnsupportedQuantPath` を assert。fixture は `crates/vokra-core/tests/parity/fixtures/m2-08/*.gguf`（hand-generated、PyTorch 不使用）。
- `cargo test -p vokra-cli --test policy_e2e` — 変換 → ロード → `check_degradation` の 5% ゲート、Vocos/BigVGAN W8A8 拒否、HiFi-GAN opt-in verify 三分岐。

`license` job（cargo-deny check licenses / advisories / bans、`scripts/check-zero-deps.sh`、`scripts/check-forbidden-symbols.sh`）は本 WP で新規依存を導入しないため継続 pass。

## 9. follow-up

- W8A8 INT8 GEMM kernel（backend crates、`vokra-backend-cpu` / `-metal` / `-cuda`）。
- Vocos / BigVGAN / HiFi-GAN kernel body（consumer モデル同期、FR-OP-10/11/12）。
- KV cache 量子化（FR-QT-05、v1.0）。
- piper / CAM++ / Silero converter への policy 配線。
- UTMOS / DNSMOS weight 配信（M1-09b）と `check_degradation` の UTMOS 分岐。
- Snake activation FP32 internal_precision の**強制**（現状 audit anchor のみ、kernel と同期）。
