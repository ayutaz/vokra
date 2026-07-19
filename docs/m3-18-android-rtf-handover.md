# M3-18 Android Whisper base RTF Measurement — Handover

**Owner**: 依頼者 (physical Android arm64-v8a device required; CC cannot execute this WP).
**Predecessors**:
- M3-02 (Vulkan バックエンド新規実装 + Android arm64-v8a cross-build clean、~28% 残 = 実 kernel `.spv` 生成 + eval.rs 他 op arm) — 現時点 Copy/Add 以外の SPIR-V dispatch は `UnsupportedOp`。Whisper base full graph を Vulkan で走らせるには残 kernel の glslc 生成が必要。CPU fallback を認める場合は M2-06 (Whisper base) の CPU 経路をそのまま Android arm64-v8a build で動作させることも可能。
- M3-11 (Godot GDExtension T01-T18 = 100% CC 完成、T19 = 実 Godot editor での dispatch verify が依頼者側で残る) — 本 WP は Godot 経由ではなく直接 `libvokra.so` を JNI 経由で叩くのが一次ルート。

**Requirement under measurement**: NFR-PF-06 (Android arm64-v8a 実機で Whisper base RTF < 0.7).

> **Explicit boundary**: 本 WP は成果物を実機で走らせて数値を採る役割に徹する。バックエンド最適化・kernel チューニング・M3-02 Vulkan 残 kernel の glslc 生成は本 WP の対象外 (M3-02 側の follow-up)。閾値未達時は「未達値をそのまま記録・公開」= 新規の緩和目標を発明しない (M2-14 と同運用)。

## 1. Prerequisites checklist

- [ ] **Android arm64-v8a 実機** (Snapdragon 8 Gen 1 以降 / Google Tensor G3+ / Dimensity 9200+ 目安、Vulkan 1.1+ 対応)。Simulator / エミュレータの RTF は NFR-PF-06 判定には**使えない** (`docs/m3-owner-verification-checklist.md` §1)。
- [ ] **Android NDK r25+** (LLVM/clang、`scripts/build-android.sh` 前提)。`ANDROID_NDK_HOME` を実 NDK 展開先に設定できること。
- [ ] **Android Studio Ladybug 以降** (Java/Kotlin 側から JNI 経由で `libvokra.so` を叩く最小アプリを組む)。
- [ ] **`whisper-base.gguf`** — `vokra-cli convert --model whisper --input <safetensors> --output whisper-base.gguf` で生成 (MIT weight、M2-06 検証済み)。
- [ ] **`tests/fixtures/audio/jfk-30s.wav`** — repo に commit 済 (sha256 `58adb4ea…`、16 kHz mono PCM16 30 s pad、M2 item 4 で確定)。
- [ ] **`Vokra.aar` or `libvokra.so`** — 下記 §2 手順で生成。
- [ ] Apple 系不要 (本 WP は Android 単体)。

## 2. `libvokra.so` cross-build (arm64-v8a、CPU-only baseline)

現在 `scripts/build-android.sh` が確定している CPU-only baseline を採用する:

```bash
export ANDROID_NDK_HOME=/path/to/android-ndk-r26
bash scripts/build-android.sh
```

生成物:
- `target/aarch64-linux-android/release/libvokra.so` (CPU-only、SIMD = NEON、zero-dep NFR-DS-02 preserved)
- `bindings/unity/com.vokra.unity/Plugins/Android/libs/arm64-v8a/libvokra.so` (Unity Package staging)

> **訂正 2026-07-19**: 下記 (a)(b) のうち **(a) は実装済み** —
> `crates/vokra-capi/Cargo.toml:88` に `vulkan = ["vokra-models/vulkan"]` が
> 存在する。同様に「Copy/Add op のみ dispatch 可能」も解消済みで、12 kernel が
> `.spv` としてコミット済み（`81e1f3c`）。**残る owner タスクは Android 実機
> soak そのもの**（M4-13-T17 = WP の exit hard gate）。原文は下に保持。

**Vulkan 有効化は現時点 vokra-capi 側で feature 未 wire**: `crates/vokra-capi/Cargo.toml` は `cpu` feature のみ持ち、`metal` / `cuda` / `vulkan` は下流の `vokra-models` 側でのみ feature 定義されている。Android Vulkan 経路を走らせるには (a) `vokra-capi` に `vulkan = ["vokra-models/vulkan"]` を追加、(b) `scripts/build-android.sh` の `--no-default-features` 制約を Vulkan feature 用に緩和する、の 2 手が必要 (現時点は未着手 = M4 follow-up)。**実機測定 v0.9 gate は CPU baseline で NFR-PF-06 を通す方針で構わない** (Vulkan RTF は M4 で再測、CPU pass が v0.9 Exit 前提)。M3-02 が partial (~28% 残 = Copy/Add op のみ dispatch 可能、他 op は explicit `UnsupportedOp` = FR-EX-08 silent CPU fallback 禁止) の状態も併せて考慮。

## 3. Minimal measurement app (Android Studio)

Android Studio Ladybug 以降で新規 Kotlin project 作成 (min SDK API 24 = Android 7.0、Unity 2022.3 LTS の Android floor と一致):

1. `libvokra.so` を `app/src/main/jniLibs/arm64-v8a/libvokra.so` に配置。
2. `whisper-base.gguf` と `jfk-30s.wav` を `app/src/main/assets/` に配置。
3. JNI wrapper (Kotlin または Java) を作成。C ABI 呼び出しは `include/vokra.h` (cbindgen 生成) を参照。最低限:
   - `vokra_session_create(model_gguf_path, backend) -> handle`
   - `vokra_transcribe(handle, pcm_ptr, pcm_len, sample_rate, out_text_ptr, out_text_capacity) -> status`
   - `vokra_session_destroy(handle)`
4. `assets/` から `getFilesDir()` に extract (Android の `AssetManager` は真の filesystem パスを提供しないため、mmap 前提の C ABI では extract 必須。`NFR-RL-04` 準拠)。
5. `jfk-30s.wav` を PCM16 16 kHz mono として parse → `FloatArray` に変換 → `vokra_transcribe` に渡す。
6. 開始・終了時刻を `System.nanoTime()` で採測。RTF = `elapsed_ns / audio_duration_ns`。

## 4. 計測プロトコル

- [ ] 実機を USB debug モードで PC に接続、`adb install` でアプリを実機に配置。
- [ ] 実機の CPU governor / thermal を **同一条件で 3 回計測**。連続計測時は間に 30 秒以上の cooldown を挟む (thermal throttling 排除)。
- [ ] Backend は **CPU (NEON)** を primary、Vulkan は experimental フラグ扱い。
- [ ] iteration = 10 (warmup 2 除外)、median を採用。
- [ ] 各 run で以下を記録:
  - 実機モデル (device model + SoC)
  - Android バージョン
  - Backend (`cpu` or `vulkan`)
  - `whisper-base` version (`vokra-cli convert` した checkpoint の SHA-256)
  - RTF (median / p50 / p95 / max)
  - 転写結果 (JFK inaugural の該当部分と一致することを目視確認)

## 5. 判定基準 (NFR-PF-06)

- **PASS**: RTF < 0.7 が median で成立、かつ転写結果が JFK reference と一致 (word-level 完全一致は不要、句レベルの一致で可、audio fixture の 11 s 発声 = `And so, my fellow Americans, ask not what your country can do for you — ask what you can do for your country.` 相当)。
- **FAIL**: RTF ≥ 0.7 or 転写不一致。
- **INDETERMINATE**: 実機不足 / build failure / Vulkan `UnsupportedOp` 遭遇。「未達値をそのまま記録」= 新規閾値を発明しない (M2-14 と同運用)。

## 6. 結果報告テンプレート

計測結果を `docs/benchmarks/v0.9-device-benchmarks.md` (M3-18-T01 で CC scaffold 予定、現時点は本 handover 内に追記でも可) に追記:

```
### Android arm64-v8a Whisper base

| Device | SoC | Android | Backend | RTF (median) | Verdict |
|--------|-----|---------|---------|--------------|---------|
| Pixel 9 Pro | Tensor G4 | 15 | cpu | 0.__ | PASS/FAIL |

Measurement date: YYYY-MM-DD
Iterations: 10 (2 warmup discarded)
Fixture: tests/fixtures/audio/jfk-30s.wav (sha256 58adb4ea…, 11 s speech + 19 s pad)
whisper-base checkpoint SHA-256: ______________
libvokra.so build: scripts/build-android.sh (NDK r26, API 24, --no-default-features)
Transcription verified: ☐ Word-level match with JFK reference
```

## 7. Escalation

- **RTF が 0.7 を超えた場合**: 未達値を上記テンプレートにそのまま記録し、原因分析は M3-02 (Vulkan) or 個別 kernel 最適化として **別 issue に起票**。本 WP は Exit criteria 判定材料の提供のみ (`docs/milestones.md` §7.3 Exit criteria 1)。
- **Vulkan で `UnsupportedOp` に当たった場合**: M3-02 の partial 状態 (~28% 残) が原因。CPU 経路で NFR-PF-06 を通せば v0.9 Exit は成立。Vulkan は M4 で再測 (`docs/tickets/m3/M3-02-vulkan-backend.md`)。
- **M3-19 四半期 Go/No-go review** に本 WP の実測結果を材料として持ち込む (`docs/m3-owner-verification-checklist.md` §4)。

## 8. 参考

- **前提 handover**: `docs/m2-14-ios-rtf-handover.md` (iOS 側、M2 carry-over)
- **build script**: `scripts/build-android.sh` (M2-11-T07、NFR-RL-04)
- **C ABI**: `include/vokra.h` (cbindgen 生成、M0)
- **Vulkan status**: `docs/adr/M3-02-spirv-generation.md` (Wave 8 + 11 で ADR 正式 land)
- **JFK fixture**: `tests/fixtures/audio/jfk-30s.wav` (M2 item 4、`docs/m3-owner-verification-checklist.md` §1)
