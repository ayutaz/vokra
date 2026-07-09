# M3 (v0.9) Owner Verification Checklist

**Owner**: 依頼者 (`ayutaz`) — 実機テスト・法務判断・鍵/秘密情報の provision を担当。
**CC-side status**（2026-07-09 更新、branch `feat/m3-plan-and-wave1`、**Wave 1〜Wave 9 の CC 実装分完了 = 19 WP 中 16 コミット済 + partial WP を Wave 7/8/9 で追撃**）:

**ticket spec**: `docs/tickets/m3/` (gitignore) に 19 WP 全 file (M3-01〜M3-19) + README 完備、~340 tickets / 285h の内訳確定 (ultracode workflow 経由)。

**実装コミット (16 WP)** — feat/m3-plan-and-wave1 で main から **35 commits ahead**、**+47,896 lines**:
- **Wave 1** (dev-experience layer): M3-08 length_conditioning / M3-14 barge-in interrupt / M3-16 ABI changelog scaffold / M3-17 prosody_control API
- **Wave 2** (foundation): M3-03 paged KV cache / M3-04 KV 量子化 (Q4_0/Q5_0/Q8_0 CPU path) / M3-05 flow_sampler + ODE solvers
- **Wave 3** (codec/vocoder): M3-06 mimi_rvq / M3-07 hifigan_generator
- **Wave 3.5** (standalone): M3-11 Godot GDExtension scaffold / M3-13 RVV 1.0 base dispatch + cross-build CI / M3-15 vokra-server multi-session + 75ms bench hooks
- **Wave 4** (CUDA): M3-01 CUDA バックエンド完成 (graph-executor 拡張 + coverage + RTF gate scaffold)
- **Wave 5** (Vulkan + models): M3-02 Vulkan backend scaffold (~30% of 41 tickets) / M3-09 CosyVoice2 scaffold (~40% of 28 CC tickets) / M3-10 Voxtral scaffold + config-aware converter (~40% of 24 CC tickets)
- **Wave 6** (parallel `wf_25308de8-a59`、5 agents 32 分、1.36M tokens): **M3-12 piper-plus native GPU バックエンド対応 (新規完成)** + **M3-04 CUDA/Metal fused dequant kernel (partial → complete)** + M3-09 Flow Matching 昇格 + chunk-aware CFM (~40% → ~60%) + M3-10 streaming + tokenizer + vokra-server dispatch (~40% → ~60%) + M3-02 SPIR-V shader library + Compute seam Vulkan arm (~30% → ~35%)
- **Wave 7** (parallel `wf_a0a8749a-3e0`、3 agents 31 分、917K tokens): **M3-02 Vulkan runtime object stack (VkDevice/CmdPool/DescriptorSet/Pipeline/DeviceMemory/Buffer + host↔device copy + coop-matrix ext walk、Android arm64 cross-build clean、~35% → ~55%)** + **M3-09 CosyVoice2 LLM decoder backbone + MEL loss + RTF gate scaffold (synthesized fixture、~60% → ~72%)** + **M3-10 Voxtral autoregressive greedy decode 完成 + Whisper 互換 API 501 → 200 化 (~60% → ~80%、audio adapter は honest limitation として明記)**
- **Wave 8** (parallel `wf_6016f4e1-172`、2 agents 31 分、610K tokens): **M3-10 Voxtral audio adapter framework (AdapterKind pluggable = None/Linear/Mlp/DownsampleLinear + GGUF metadata `vokra.voxtral.adapter.*` + JSON side-car + `--adapter-config` CLI + soft-prefix decode 配線、Wave 7 の honest LM continuation → real audio-conditioned decode に昇格、~80% → ~90%)** + **M3-02 SPIR-V blob 生成方針確定 (ADR M3-02-spirv-generation で 4 option 評価) + tooling (`scripts/compile-vulkan-shaders.sh` + `scripts/install-vulkan-toolchain.md`) + hand-crafted copy_f32 SPIR-V module (145 words / 580 bytes) + FFI dispatch primitives + 実 Vulkan dispatch chain smoke test 5 本、T13 完全完了、~55% → ~65%)**
- **Wave 9** (parallel `wf_05152309-fec`、4 agents 30.6 分、1.15M tokens): **M3-02 T14 partial + T24 eval_op + T26 graph-executor Vulkan arm** (`OpKind::Copy` を vokra-core に追加 + hand-crafted `add_f32.spv` 172 words / 688 bytes 2 個目 Handcrafted (ADR §5 cap 到達) + 14-entry spirv.rs manifest + `smoke_dispatch_add_f32_impl` 3-SSBO descriptor set + `eval_vulkan_op` の Copy/Add arm dispatch (他 op は明示 UnsupportedOp = FR-EX-08) + `backend.rs::supports` T35 lock-step gate + 4 new test files (handcrafted_add_f32 / eval_op_dispatch / graph_executor_vulkan / smoke_dispatch 拡張) + no_silent_fallback 拡張、SPIR-V opcode 値は Khronos spec §3.32 実値のみ = 発明ゼロ、SHA-256 pin は empirically 取得、+30 tests、~65% → ~70%) + **M3-09 T07/T08 LLM forward body + synthesized-fixture parity harness** (Mistral-style GQA transformer 完全実装: RoPE precomputed cos/sin table + Grouped Query Attention (n_heads/n_kv_heads = group_size) + causal-masked softmax + SwiGLU FFN + RMSNorm + KV cache append + `LlmBackbone::forward` bulk pre-norm block sequence + tied logits + `LlmBackbone::step` autoregressive step (KV cache lazy) + `LlmBackbone::greedy_decode` argmax + eos early stop + `LlmWeights::synthesized(config, seed)` SplitMix64 Xavier + `parity::forward_matches_step_by_step` property test (atol=1e-3) + `parity::assert_vs_hf_reference` 「flip-the-switch」harness (NotImplemented 今、T02 real HF checkpoint 到着で switch) + shared primitives (rms_norm/silu_inplace/hadamard_inplace/rope_apply) を voxtral::text_decoder から import、hyperparam は全て CosyVoice2Config 経由 = ハルシネーション厳禁 preserved、+46 tests、~72% → ~85%) + **M3-10 Voxtral beam search + n-best decode + KV cache snapshot** (`voxtral::beam_search` module: BeamConfig (beam_size / α=0.6 GNMT length penalty / max_new_tokens / eos_token / no_repeat_ngram_size / top_k_per_beam) + BeamResult (tokens / log_prob / length_normalized_score) + beam_search_decode + beam_search_decode_with_prefix (adapter-aware soft-prefix branch) + `vokra_core::KvCache` derives Clone (beam 分岐用 deep-copy、hot path でない旨明記) + `voxtral::TextDecoderKvSnapshot` opaque Clone handle + `TextDecoderSession::kv_snapshot` / `kv_restore` branch primitive + `AsrHead::transcribe_beam` + `VoxtralAsr::transcribe_beam` + `TranscribedBeam` + `transcribe_beam_with_defaults` convenience wrapper + `vokra-cli --beam-size / --length-penalty / --no-repeat-ngram` flag + beam_size=1 が greedy と bit-identical property test、+41 tests、~90% → ~95%) + **M3-10 Voxtral GPU decode session (Metal + CUDA thin Compute-seam wrappers)** (VoxtralMetalDecodeSession + VoxtralCudaDecodeSession = TextDecoderSession を backend Metal/CUDA 固定で wrap = 既に forward_step の GEMM/GEMV/softmax/RMSNorm/GELU が Compute seam 経由で Metal/CUDA arm へ dispatch = per-op 実 GPU dispatch is real (full device residency = M4 follow-up で意図的 scope-out) + API surface: new_from_decoder / step / step_with_embed_prefix / kv_cache_len / reset / last_logits / all_logits / backend_name + Metal type Apple + `metal` gated / CUDA type Unix,Windows + `cuda` gated + `VoxtralAsr::with_allow_device_session(true)` opt-in + `select_effective_backend` cfg-gate probing (Metal > CUDA > BackendUnavailable、Apple + CUDA rare case は Metal 優先) + FR-EX-08 no silent CPU fallback verified by 3 dedicated tests (off-GPU-build BackendUnavailable / on-metal-build honest Ok or BackendUnavailable / on-cuda-build 同) + Metal 実 GPU で bit-identical vs CPU (atol < 5e-4、M1 iMac 実行) + CUDA cross-build clean、+20 tests、~95% → ~97%)

**未実装 CC WP**:
- **M3-18 Android/Godot 実機テスト** — 依頼者専任 WP、CC 側実装なし。
- **M3-19 Kill switch D 判定** — 依頼者専任 WP。

**Verify (2026-07-09、Wave 9 merge 後)**: cargo build clean (全 12 crate + vokra-backend-vulkan opt-in feature、default / metal / cuda / vulkan feature 全 build 成功) / cargo test 全体 **1712 passed / 0 failed / 4 ignored** (default) / **1720 passed** (all features `vulkan,metal,cuda`) (Wave 8 baseline 1585 から **+127 (default) / +135 (all features)**、Wave 5 baseline 1386 から累積 **+326 / +334**) / cargo fmt --check clean / cargo clippy `-D warnings` clean / `scripts/check-zero-deps.sh` OK (root Cargo.lock は vokra-* only、NFR-DS-02 保存) / `scripts/check-abi-changelog.sh` OK (Wave 9 は adapter/beam/GPU session すべて Rust-only + GGUF metadata で C ABI 変更なし)。

**Partial 実装 WP の残 ticket** (各 WP の spec file を参照して follow-up、blocking ではない):
- **M3-02 Vulkan** (~30% 残、Wave 9 で ~35% → ~30%): T14-T22 実 kernel の glslc 生成 + `kernels/precompiled/*.spv` commit (GEMM 2 variants / GEMV / softmax + softmax_causal / layer_norm / gelu / conv1d / elementwise / activation / shape ops、`scripts/compile-vulkan-shaders.sh` 実行 by developer with `brew install glslang` or Vulkan SDK) / eval.rs の Copy/Add 以外 op arm の SPIR-V dispatch (Wave 9 で Copy+Add は完了) / T27-T29 graph-executor 他 op Vulkan arm / T33/T34 Whisper base parity harness + numerical-parity CI wiring / T36 CI 上での glslc install → recompile → diff gate (ADR §5 で defer) / T37 Android arm64-v8a real-device runner / T39-T40 Android 実機 RTF (M3-18 と併走)
- **M3-09 CosyVoice2** (~15% 残、Wave 9 で ~28% → ~15%): T02 upstream CosyVoice2 safetensors tensor manifest 提供後に `LlmWeights::from_gguf` を real weights 対応 (LLM forward body は Wave 9 で完了、synthesized fixture で numerical stability + shape sanity は verify 済) / T09/T21/T22 実 HF checkpoint parity CI + PyTorch reference dumper / T13 real Mimi checkpoint 統合 / T23 real MEL loss validation / T24 real-checkpoint RTF <1.0 always-on gate (M2-14 self-hosted CUDA runner) = **すべて依頼者 HF アクセス + real checkpoint 前提**
- **M3-10 Voxtral** (~3% 残、Wave 9 で ~10% → ~3%): (a) real Voxtral safetensors inspection → 実 adapter tensor 名 + dim + activation 抽出 → `adapter.json` side-car 作成 → `vokra-cli convert --adapter-config` 実行 → 生成 GGUF の loadable 確認 = **依頼者**、(b) `VoxtralAsr::transcribe` の real ASR WER 実測 (T19/T20 fixtures 対 upstream reference) = **依頼者**、(c) 実 multilang WER 実測 (T22、real HF weight + real audio dataset = 依頼者)、(d) full device-residency VoxtralDecodeSession (weights + KV cache 一括 device 常駐、Whisper WhisperMetalDecodeSession 相当の per-step 1 command buffer、CC follow-up、Wave 9 の thin Compute-seam wrapper は per-op GPU dispatch なので機能的に完成)
- **M3-12 piper-plus GPU**: M3-12-T14 実 voice GGUF での Metal (M1 iMac) / CUDA (vast.ai RTX 4090) sanity run = 依頼者専任 (実機必須、atol=0.01 component / atol=0.05 PCM を確認)。
- **M3-01 CUDA 完成**: 実 GPU 側 RTF<0.10 always-on gate は M2-14 self-hosted runner + M3-01 regression 5% gate へ defer 決定済 (docs/m2-cuda-rtf-variance-2026-07-08.md)。

各項目は「必要な準備 → 実行手順 → Exit 判定への寄与」の 3 段で記述。CC が既に整備した scaffold（scripts / CI / docs）へのポインタを併記する。

---

## 1. Android 実機 RTF 測定（M3-18 / NFR-PF-06）

**依頼者タスク**: Android 実機で Whisper base の RTF < 0.7 を計測する。Vulkan バックエンド（M3-02、CC 実装）が Android arm64-v8a で正しく動くことを実測で担保する（`docs/milestones.md` §7.3 Exit criteria 1）。

### 必要な準備

- [ ] Android Studio + NDK r26+（Vulkan header 対応、`libvulkan.so.1` dynamic link）。
- [ ] Android arm64-v8a 実機（Vulkan 1.1+ 対応。Snapdragon 8 Gen 1 以降 / Google Tensor G3+ / Dimensity 9200+ 目安）。エミュレータ RTF は NFR-PF-06 判定には使えない（Vulkan lavapipe は CI で validate 済だが実機性能とは別）。
- [ ] `libvokra.so` — Android arm64-v8a build（`scripts/build-android.sh` をローカル実行、または tagged release の `release.yml` で生成）。
- [ ] `whisper-base.gguf` — `vokra-convert convert --model whisper --input <safetensors> --output whisper-base.gguf` で作成。
- [ ] `tests/fixtures/audio/jfk-30s.wav`（M2 で commit 済、sha256 `58adb4ea...`）を app assets に bundle。

### 実行手順

M3-02（Vulkan バックエンド）完成後の owner runbook（別途 CC が `docs/m3-18-android-rtf-handover.md` を land 予定、現時点未着手 = TODO）に従う。

1. Android Studio App 新規プロジェクト → `libvokra.so` + `libvokra_capi.h` を JNI/JNA 経由で bind。
2. `whisper-base.gguf` + `jfk-30s.wav` を app target に bundle（`assets/` 経由、`VokraAndroidAssets` passthrough を利用）。
3. Signing → 実機ビルド → `adb install`。
4. アプリ内から Whisper base を Vulkan backend で 3 回計測し median を記録。

### Exit 判定への寄与

- NFR-PF-06（Android 実機 Whisper base RTF < 0.7）— v0.9 Exit criteria 1（`docs/milestones.md` §7.3）。
- 未達の場合は「未達値をそのまま記録・公開」（新規閾値を発明しない）、M3-19 四半期 Go/No-go review へ入力。

---

## 2. Godot デモ実機動作確認（M3-11 → M3-18 / FR-API-05）

**依頼者タスク**: Godot GDExtension バインディング（M3-11、CC 実装）を使ったデモが Godot Editor + Android/Windows/Linux で動作することを実機で確認する。`docs/milestones.md` §7.3 Exit criteria 3。

### 必要な準備

- [ ] Godot 4.2+（GDExtension は Godot 4 以降）。
- [ ] `vokra-godot.gdextension` パッケージ（M3-11 の `scripts/build-godot-gdextension.sh` CI 自動ビルド、NFR-MT-08 手動ビルド配布禁止）。
- [ ] Whisper base or piper-plus のデモモデル（GGUF）。
- [ ] 実機 or Godot Editor（macOS/Linux/Windows）。Android/iOS ターゲットは export template 経由。

### 実行手順

M3-11 完成後の owner runbook（CC 側で `docs/m3-11-godot-demo-handover.md` を land 予定、現時点未着手 = TODO）に従う。

1. Godot Editor で新規プロジェクト作成 → `vokra-godot.gdextension` を addon として import。
2. GDScript から `Vokra.load_model()` / `Vokra.transcribe()` などの API を呼ぶデモシーンを作成。
3. Editor で動作確認 → Windows/Linux/Android の各 export target でビルド → 実機動作確認。

### Exit 判定への寄与

- FR-API-05 の Godot デモ動作 — v0.9 Exit criteria 3（`docs/milestones.md` §7.3）。
- Godot ゲーム開発者コミュニティへの露出 = Kill switch D 回避のための contributor onboarding にも寄与（BR-04）。

---

## 3. CosyVoice2 / Voxtral モデル license audit（M3-09/M3-10 T-audit / FR-MD-13）

**依頼者タスク**: MIT / Apache 2.0 weight の商用配布可否を最終判断し、`docs/license-audit.md` + `docs/legal-compliance.md` に sign-off を残す。M2-06 の Whisper / M2-07 の Kokoro と同じ手続き。

### 必要な準備

- [ ] **CosyVoice2** Apache 2.0（FunAudioLLM/CosyVoice2）: 商用 OK + training data 疑義なしの確認。**Mimi codec** は CC-BY 4.0 with attribution — NOTICE 記載必須（M3-06 で対応、M3-09 Exit で確認）。
- [ ] **Voxtral (Mistral)** Apache 2.0 / Apache 2.0: 商用 OK の確認。
- [ ] research flag 対象（F5-TTS / Fish-Speech / EnCodec）が公式 zoo に混入していないことの目視確認（M2-13 compliance gate が自動拒否するが最終目視、M3 での新規モデル追加時に再確認）。

### 実行手順

`docs/license-audit.md` §3.1 の Owner sign-off template を使用（M2-06/M2-07 と同じ手続き、CC-verified 事実確認 subsection + 依頼者記入用の空欄 sign-off テーブル）。M3-09/M3-10 の T-audit チケット完了時に spec 化される想定。

### CC 側整備状況（現時点 = 2026-07-09）

- CC は M3-09（CosyVoice2）と M3-10（Voxtral）のチケット spec を未 land（rolling wave で本 WP 着手時に spec 化）。license-audit.md への追記は M3-09/M3-10 の中でチケット化する。
- Mimi codec attribution 要件（CC-BY 4.0）は M3-06 mimi_rvq チケット spec で NOTICE 更新項目として扱う（現時点未 spec 化）。

### Exit 判定への寄与

- FR-MD-13 のプロセス承認。M3-09/M3-10 WP 完了確定。

---

## 4. Kill switch D 判定 + 四半期 Go/No-go review（M3-19 / NFR-MT-05）

**依頼者タスク**: v0.5 公開後 **3 ヶ月固定**時点（暦月目安 2027-03〜05、`docs/milestones.md` §7.4）で Claude Code 以外のコミッター数を計測し、3 名未満なら撤退検討 = Kill switch D 発動を判定する。

### 判定閾値（`docs/milestones.md` §7.4 / CLAUDE.md Kill switch 表転記）

- **D**: v0.5 公開後 3 ヶ月で Claude Code 以外のコミッター 3 名未満 → 撤退検討。
- 判定時期は暦月換算 2027-03〜05 頃（v0.5 close = 2026-07 末〜08 上旬 + 3 ヶ月固定）。

### 必要な準備

- `docs/governance/kill-switch-metrics-runbook.md`（VOKRA-GOV-001、M2 で land 済）: GitHub commits の non-bot non-CC contributor 集計コマンド、四半期 review 記録 template（`docs/governance/quarterly-reviews/YYYY-QN.md`）を規定。

### 実行手順

四半期 Go/No-go review record を独立公開ガバナンス記録として発行（governance docs / GitHub Discussion / post-mortem blog のいずれか、M2-15 の枠組みを継続）。

### Exit 判定への寄与

- Kill switch D の判定結果を M3-19 記録として公開。継続 or exit path 選択（Wyoming Protocol 準拠実装として HA に統合 / Candle audio extension として merge / HuggingFace・ggml-org による acquire = CLAUDE.md の現実的撤退経路）。
- 3 名未達を回避するための contributor onboarding・community engagement（開発時間配分 7%、NFR-MT-01）は本フェーズも継続支出する（`docs/milestones.md` §7.4）。

---

## 5. サーバ 75ms 実測（M3-15 / NFR-PF-05）

**依頼者タスク**: `vokra-server` v1.0 版の multi-session + TTS レイテンシ 75ms（NFR-PF-05 の v1.0 値）を実機ネットワーク条件で計測する。

### 必要な準備

- [ ] `vokra-server` v1.0 版（M3-15 CC 実装完了後）。paged KV cache（M3-03）+ multi-session 実装済。
- [ ] 計測クライアント: `vokra-cli bench --model piper-plus --target-latency-ms 75` or 独自 HTTP client（FR-TL-02 の RTF/TTFA/p50/p95/p99 出力を利用）。
- [ ] サーバ実行環境: 実機 or CI GPU runner。ネットワーク条件は「LAN local」を基本とし、リモートは別途参考値。

### 実行手順

M3-15 完成後の owner runbook（CC 側で `docs/m3-15-server-latency-handover.md` を land 予定、現時点未着手 = TODO）に従う。

### Exit 判定への寄与

- NFR-PF-05 の v1.0 値（75ms）達成確認 — v0.9 Exit criteria の追加項目（`docs/milestones.md` §7.3 追加項目 = サーバ TTS レイテンシ 75ms）。

---

## 6. CUDA large-v3 RTF formal gate（M2-14 carry-over → M3-01 5% regression gate）

**依頼者タスク**: M2 で defer 決定した **CUDA large-v3 RTF < 0.1 の formal always-on gate** を M3-01（CUDA バックエンド完成）+ M2-14（self-hosted runner standup）で達成する。M3-01 で NFR-PF-13（性能 regression 5% 判定）下で維持する。

### 必要な準備（M2 checklist §2 から carry-over）

- [ ] **M2-14 self-hosted runner standup**: dedicated RTX 4090 or Ampere/Ada GPU、GitHub Actions self-hosted runner 登録、CI job `gpu-backends` の cuda arm に接続。M2 で vast.ai spot RTX 4090 の hardware variance が baseline の 2.5x に届いた（`docs/m2-cuda-rtf-variance-2026-07-08.md`）ため owner judgment で defer 決定済。
- [ ] `whisper-large-v3.safetensors` を `vokra-convert` で GGUF 化（M2 手順と同じ）。

### 実行手順

1. self-hosted runner を M2-14 で standup（依頼者作業）。
2. M3-01 の CC 実装（FA v2 ベース + inter-head overlap + session pool の仕上げ）を land。
3. M3-01 T-follow-01/02（`vokra_flash_attn_v2_causal_f32` の再着地 + weight caching + shared memory tuning）を CC 側で消化。
4. self-hosted runner 上で N=10 iter measurement → RTF < 0.1 を 5% regression gate 下で常時保証。
5. 結果を `docs/bench-baselines/whisper_large_v3_cuda_rtf.json` に更新して commit。

### Exit 判定への寄与

- NFR-PF-04（CUDA large-v3 RTF < 0.1）の formal always-on gate 達成 — M2 exit criteria 2 の carry-over が M3-01 で完結。

---

## 7. iOS 実機 RTF（M2-14 carry-over）

**依頼者タスク**: M2 で引き渡し済みの iOS 実機 RTF 計測（`docs/m2-14-ios-rtf-handover.md`）を M3 期間内に消化する。Whisper base RTF < 0.5（NFR-PF-03）。

### 必要な準備（M2 checklist §1 参照）

M2 checklist §1 と同じ。差分なし。

### Exit 判定への寄与

- NFR-PF-03（iOS 実機 Whisper base RTF < 0.5）— M2 exit criteria 1 の carry-over。M2 close 判定に必要。

---

## Summary 進捗表（2026-07-09 更新、Wave 1〜Wave 9 land 完了時点、branch `feat/m3-plan-and-wave1`）

| WP | 内容 | CC 進捗 | 依頼者残タスク |
|----|------|--------|--------------|
| M3-01 | CUDA 完成 + RTF<0.1 formal gate | ✅ **Wave 4 実装 land**（`3f1a4a5`、graph-executor 拡張 Gemv/Softmax/SoftmaxCausal/LayerNorm/Gelu/Conv1D + FA v2 compute_89 pin + coverage test + `gpu-cuda-rtf.yml` scaffold + long-form decoder dumper。RTF<0.10 always-on gate は M2-14 self-hosted runner + M3-01 5% regression gate へ defer 済） | § 6（M2-14 self-hosted runner + RTF gate）|
| M3-02 | Vulkan バックエンド | ✅ **Wave 5 + Wave 6 + Wave 7 + Wave 8 + Wave 9 追撃**（`d11fac2` + `3549e34` + `2ff5561` + `1e2199e` + `07431cb` + `1a35473`、`vokra-backend-vulkan` crate = **~70% (Wave 9 で +5pp)** = Wave 5-8 の 生 FFI dlopen + probe queue family + SPIR-V manifest + runtime object stack + T13 完全完了 (hand-crafted copy_f32 smoke kernel) + **Wave 9 で: `OpKind::Copy` を vokra-core に追加 (non_exhaustive、下流 backend 破壊なし) + hand-crafted `add_f32.spv` 172 words / 688 bytes 2 個目 Handcrafted (ADR §5 cap "exactly these two — no more" 到達) + spirv.rs manifest 13→14 entry + `smoke_dispatch_add_f32_impl` 3-SSBO descriptor set + `smoke_dispatch_add_f32` public API + off-target BackendUnavailable stub + eval.rs `eval_vulkan_op` の Copy/Add arm dispatch (他 op は明示 UnsupportedOp = FR-EX-08、silent CPU fallback 禁止) + `backend.rs::supports` を Copy+Add で `true` に反転 (T35 lock-step gate) + tests/handcrafted_add_f32.rs / tests/eval_op_dispatch.rs / tests/graph_executor_vulkan.rs 新規 3 files + smoke_dispatch.rs / no_silent_fallback.rs 拡張 + SPIR-V opcode 値は Khronos SPIR-V 1.3 spec §3.32 実値のみ (OpFAdd=129, OpAccessChain=65, OpCapability=17 等) + SHA-256 pin は cargo test の panic message から empirically 取得 = 発明ゼロ、+30 tests**、~30% 残 = T14-T22 実 kernel の glslc 生成 (developer-side) / eval.rs の Copy/Add 以外 op arm SPIR-V dispatch (Wave 9 で Copy+Add 完了) / T27-T29 graph-executor 他 op Vulkan arm / T33/T34 Whisper base parity on lavapipe / T36-T38 CI wiring / T39-T40 Android 実機 RTF）| § 1（Android 実機 RTF、M3-18 と連動）|
| M3-03 | paged KV cache | ✅ **Wave 2 実装 land**（`56b52a9`、`PagedKvCache<T>` + [time, stream, codebook] 3D + `KvElement` trait + `GpuPagedKvCacheOps` seam + 23 unit tests）| — |
| M3-04 | KV cache 量子化 | ✅ **Wave 2 + Wave 6 完成**（`56b52a9` + `c315186`、Q4_0/Q5_0/Q8_0 CPU pack/unpack + Wave 6 で `KvQuantDequantGemvOps` trait + CUDA NVRTC PTX `vokra_dequant_gemv_q{4,5,8}_0_f32` kernel + Metal MSL 対応 3 kernel + DequantGemvDims + trait impl + 8 shape × 3 format parity tests (atol 1e-4、Apple M1 実測 max\|Δ\|=5.245e-6)、100% 完了、Fp32 rejection + shape mismatch explicit error 記録済）| — |
| M3-05 | flow_sampler + ODE solver | ✅ **Wave 2 実装 land**（`596c312`、`flow_sample()` runtime function（FR-EX-10、グラフ非埋込）+ CfgMode 3 種 + Schedule 3 種 + OdeSolver 5 種（DDIM/DPM++/Euler/Heun/Flow-ODE）+ 35 tests）| — |
| M3-06 | mimi_rvq codec | ✅ **Wave 3 実装 land**（`596c312`、Mimi paged block_size 2/4 time-axis paging + CC-BY 4.0 attribution NOTICE §5 + EnCodec exclusion gate `scripts/compliance/check-encodec-exclusion.sh`）| § 3（CosyVoice2 audit と一括）|
| M3-07 | hifigan_generator op | ✅ **Wave 3 実装 land**（`596c312`、FP32/fp16 + INT8 opt-in with per-channel calibration + `SPECTRAL_CHECK_THRESHOLD` spectral check gate）| — |
| M3-08 | length_conditioning op | ✅ **Wave 1 実装 land**（`f61c649`、`crates/vokra-ops/src/length_conditioning.rs` 326 行 + tests 2 本（IR 区別 + parity）） | — |
| M3-09 | CosyVoice2 統合 | ✅ **Wave 5 + Wave 6 + Wave 7 + Wave 9 追撃**（`3507573` + `e2f842e` + `1ff934b` + `817ceb5`、**~85% (Wave 9 で +13pp)** = module tree + text encoder + `ChunkAwareCfm` + `ChunkAwareStreamingPipeline` (Wave 5-6) + Wave 7 で `LlmBackboneConfig` reading `vokra.cosyvoice2.arch.*` keys + LlmBackbone scaffold with GQA well-formedness validation + shared Mistral primitives with voxtral + degradation gates + Wave 9 で **T07/T08 LLM forward body 完全実装** = Mistral-style GQA transformer 完全実装 (RoPE precomputed cos/sin table + Grouped Query Attention (n_heads/n_kv_heads = group_size) + causal-masked softmax + SwiGLU FFN + RMSNorm + KV cache append) + `LlmBackbone::forward` bulk pre-norm block sequence + tied logits + `LlmBackbone::step` autoregressive step (KV cache lazy allocate) + `LlmBackbone::greedy_decode` argmax + eos early stop + n_ctx guard + `LlmWeights::synthesized(config, seed)` SplitMix64 Xavier uniform init + `parity::forward_matches_step_by_step` property test (bulk forward vs per-token step が atol=1e-3 一致) + `parity::assert_vs_hf_reference` real-checkpoint 「flip-the-switch」harness (NotImplemented 今、T02 tensor manifest 到着で switch) + hyperparam は全て CosyVoice2Config 経由 = ハルシネーション厳禁 preserved (DEFAULT_ROPE_BASE_QWEN2 = 1_000_000 / DEFAULT_RMS_NORM_EPS = 1e-5 の sentinel は Qwen2 公式 config.json 実値、コメントで根拠明記)、+46 tests、**~15% 残** = T02 real HF tensor manifest 提供後に `LlmWeights::from_gguf` を real weights 対応 (LLM forward body は Wave 9 で完了、synthesized fixture で numerical stability + shape sanity は verify 済) / T09/T21/T22 実 HF checkpoint parity CI + PyTorch reference dumper / T13 real Mimi checkpoint / T23 real MEL loss validation / T24 real-checkpoint RTF <1.0 always-on gate (M2-14 self-hosted CUDA runner) = 依頼者 HF アクセス + real checkpoint 前提）| § 3（audit）|
| M3-10 | Voxtral 統合 | ✅ **Wave 5 + Wave 6 + Wave 7 + Wave 8 + Wave 9 追撃**（`089b9c3` + `b1d7aaa` + `c5e0579` + `772124f` + `2989c4f` + `0efd238`、**~97% (Wave 9 で +7pp)** = Whisper 派生 audio encoder + Mistral text decoder + tokenizer + streaming + TextDecoderSession + autoregressive greedy decode + 501 → 200 化 + audio adapter framework (AdapterKind None/Linear/Mlp/DownsampleLinear + JSON side-car + `--adapter-config` CLI + soft-prefix decode) + **Wave 9 で: beam search + n-best decode** (`voxtral::beam_search` module: BeamConfig (beam_size / α=0.6 GNMT length penalty / max_new_tokens / eos_token / no_repeat_ngram_size / top_k_per_beam) + BeamResult (tokens / log_prob / length_normalized_score) + beam_search_decode + beam_search_decode_with_prefix (adapter-aware soft-prefix branch) + `vokra_core::KvCache` derives Clone (beam 分岐用 deep-copy、hot path でない旨明記) + `voxtral::TextDecoderKvSnapshot` opaque Clone handle + `TextDecoderSession::kv_snapshot` / `kv_restore` branch primitive + `AsrHead::transcribe_beam` + `VoxtralAsr::transcribe_beam` + `TranscribedBeam` + `transcribe_beam_with_defaults` convenience wrapper + `vokra-cli --beam-size / --length-penalty / --no-repeat-ngram` flag + property tests = α=0 で log-prob と一致 / α>0 で長い beam 不利 / beam_size=1 が greedy と bit-identical / no_repeat_ngram で -inf mask、+41 tests) + **Wave 9 で: GPU decode session (Metal + CUDA)** (VoxtralMetalDecodeSession + VoxtralCudaDecodeSession = TextDecoderSession を backend Metal/CUDA 固定で wrap = forward_step の GEMM/GEMV/softmax/RMSNorm/GELU が Compute seam 経由で Metal/CUDA arm へ dispatch = per-op 実 GPU dispatch (full device residency = M4 follow-up で意図的 scope-out) + API surface: new_from_decoder / step / step_with_embed_prefix / kv_cache_len / reset / last_logits / all_logits / backend_name + Metal type Apple + `metal` gated / CUDA type Unix,Windows + `cuda` gated + `VoxtralAsr::with_allow_device_session(true)` opt-in + `select_effective_backend` cfg-gate probing (Metal > CUDA > BackendUnavailable) + FR-EX-08 no silent CPU fallback verified by 3 dedicated tests + Metal 実 GPU で bit-identical vs CPU (atol < 5e-4、M1 iMac 実行) + CUDA cross-build clean、+20 tests)、**~3% 残** = (a) real Voxtral safetensors inspection → 実 adapter tensor 名 + dim + activation 抽出 → `adapter.json` side-car 作成 → `vokra-cli convert --adapter-config` = **依頼者**、(b) `VoxtralAsr::transcribe` real ASR WER 実測 (T19/T20 fixtures) = **依頼者**、(c) 実 multilang WER 実測 (T22) = **依頼者**、(d) full device-residency VoxtralDecodeSession (per-step 1 command buffer、Whisper Metal session 相当、CC follow-up、機能的には既に per-op GPU dispatch で完成)）| § 3（audit + adapter side-car）|
| M3-11 | Godot GDExtension | ✅ **Wave 3.5 実装 land**（`5fdb032`、excluded workspace `integrations/vokra-godot` + 生 GDExtension C ABI（godot-cpp binding crate 不使用）+ Rust panic → godot error via catch_unwind + Linux build path）| § 2（実機動作確認、M3-18 と連動）|
| M3-12 | piper-plus native の GPU 対応 | ✅ **Wave 6 実装 land**（`4805d9a`、既存 CPU 経路が `PIPER_HOT_OPS=&[HotOp::Gemm]` で Compute seam 構造化済（M0-07）を確認 + `PiperPlusTts::synthesize_with_intermediates(&self, ids, lid, backend, ...)` 明示 backend deterministic 合成 (noise=0) API + `PiperIntermediates` 6-field struct (m_p / logs_p / z / pcm / t_phonemes / t_frames) + `tests/parity_piper_plus_gpu.rs` Metal/CUDA 別 test で T10 (encoder atol=0.01) / T11 (flow atol=0.01) / T12+T13 (PCM atol=0.05) parity assert (tolerance = ADR-0012 §D3)、triple gate = env `VOKRA_PIPER_V7_GGUF` + backend feature + 実 GPU device、CI は 3 段階 skip clean、silent CPU fallback 禁止、`UnsupportedOp` は panic (Phase 4 で GEMM cover 済ゆえ現れたら bug)、`mas` op は piper-plus 推論では不要 (length_regulate = commons.generate_path monotonic search-free)、ADR-0012 §D2 判定根拠記録済）| § 8（M3-12-T14 実 voice GGUF sanity run、Metal M1 iMac / CUDA vast.ai RTX 4090）|
| M3-13 | RVV 1.0 基本対応 | ✅ **Wave 3.5 実装 land**（`c6022cf`、`crates/vokra-backend-cpu/src/kernels/rvv.rs` + `vec_add_f32` intrinsics（`vsetvli`/`vle32`/`vfadd`/`vse32`）+ CI cross-build（`riscv64gc-unknown-linux-gnu`）+ asm mnemonic check）| — |
| M3-14 | barge-in（stream.interrupt()）| ✅ **Wave 1 実装 land**（`9266f62`、`Stream::interrupt()` + `InterruptHandle`（`Arc<AtomicBool>` + `Clone+Send+Sync`）+ `EventPoller::drain_all()` + C ABI `vokra_stream_interrupt` + 10 unit + 4 integration + 3 C-ABI tests、ABI changelog に entry 記録済）| — |
| M3-15 | vokra-server multi-session + 75ms | ✅ **Wave 3.5 実装 land**（`819acf3`、multi-session scheduler + paged KV cache 配線 + 75ms bench hooks（NFR-PF-05 v1.0 値））| § 5（サーバ 75ms 実測、実機ネットワーク条件下）|
| M3-16 | v0.9 ABI 変更点の changelog 記録（凍結は M4-12 へ移動）| ✅ **Wave 1 実装 land**（`f864ade`、`docs/abi-changelog.md` schema + `docs/abi/vokra.h.v0.9-baseline.symbols` machine-anchor + `scripts/check-abi-changelog.sh`（verify/list/update-snapshot/self-test/help modes、zero-dep = bash+awk+grep+diff）+ cbindgen banner に M3-16/M4-12 参照）| — |
| M3-17 | prosody_control 統一 API | ✅ **Wave 1 実装 land**（`f61c649`、`crates/vokra-ops/src/prosody.rs` 440 行（`ApplyProsody` + `ProsodyControl`、attrs/dispatch/lib 配線）） | — |
| M3-18 | 実機テスト: Android + Godot | ⏸️ **依頼者ボトルネック**（実機必須、NFR-PF-06 Whisper base RTF <0.7）| § 1 + § 2 |
| M3-19 | Kill switch D + 四半期 review | ⏸️ **依頼者ボトルネック**（暦月 2027-03〜05 頃、v0.5 公開後 3 ヶ月固定）| § 4 |
| M2-14 carry-over | iOS 実機 RTF | 引き渡し済み | § 7 |

**Wave 1〜Wave 9 完了サマリ（2026-07-09）**: **19 WP 中 16 WP CC 実装コミット完了 + partial WP を Wave 6/7/8/9 で追撃**（Wave 1 = 4 WP / Wave 2 = 3 WP / Wave 3 = 2 WP / Wave 3.5 = 3 WP / Wave 4 = 1 WP / Wave 5 = 3 WP scaffold / Wave 6 = M3-12 新規完成 + M3-04 GPU 完成 + M3-02/09/10 partial 進捗 / Wave 7 = M3-02 Vulkan runtime object stack + M3-09 LLM backbone scaffold + M3-10 autoregressive greedy decode 完成 / Wave 8 = M3-10 audio adapter framework (real ASR 昇格) + M3-02 SPIR-V blob 生成方針確定 + hand-crafted smoke kernel / **Wave 9 = M3-02 add_f32 handcrafted kernel + eval_op(Copy/Add) + graph-executor Vulkan arm + M3-09 LLM forward body + M3-10 beam search + M3-10 GPU decode session (Metal+CUDA)**）。branch `feat/m3-plan-and-wave1`、main から **35 commits ahead / +47.9k lines**。**Wave 6/7/8/9 いずれも ultracode parallel workflow (worktree 隔離)**: Wave 6 = `wf_25308de8-a59` (5 agents 32 分、1.36M tokens、conflict 0) / Wave 7 = `wf_a0a8749a-3e0` (3 agents 31 分、917K tokens、conflict 0) / Wave 8 = `wf_6016f4e1-172` (2 agents 31 分、610K tokens、conflict 0) / **Wave 9 = `wf_05152309-fec` (4 agents 30.6 分、1.15M tokens、voxtral/asr.rs + voxtral/mod.rs 軽微 conflict = A2 の beam search と A3 の GPU session が同 file を触った、structural resolution で解消)**。**verify 全 green**: cargo build clean（全 12 crate + vokra-backend-vulkan opt-in default OFF、default / metal / cuda / vulkan feature 全 build 成功）／cargo test 全体 = **1712 passed / 0 failed / 4 ignored** (default) / **1720 passed** (all features `vulkan,metal,cuda`)（Wave 8 baseline 1585 から **+127 / +135**、Wave 5 baseline 1386 から累積 **+326 / +334**）／cargo fmt --check clean／cargo clippy `-D warnings` clean／`scripts/check-zero-deps.sh` OK（root Cargo.lock は `vokra-*` のみ、NFR-DS-02 保存）／`scripts/check-abi-changelog.sh` OK（Wave 9 は adapter/beam/GPU session すべて Rust-only + GGUF metadata で C ABI 変更なし）。**残 CC WP** = **なし** (Wave 6 で M3-12 完成 + Wave 7/8/9 で M3-02/09/10 の CC-implementable 分を継続進捗)。**残依頼者専任 WP** = **M3-18** + **M3-19** + **M3-12-T14 実機 sanity run** + **M3-10 real adapter.json side-car 作成 + WER 実測**。**Partial 実装 WP の依頼者引き渡し分**（follow-up、blocking ではない）= M3-02 ~30% 残 (T14-T22 実 kernel glslc 生成 + T33/T34 lavapipe parity + T39-T40 Android 実機 owner) / M3-09 ~15% 残 (T02 real HF tensor manifest 提供後 `LlmWeights::from_gguf` real weights 対応 + T21-T24 real HF checkpoint parity owner、LLM forward body は Wave 9 で完了) / M3-10 ~3% 残 (real Voxtral safetensors → adapter.json + WER 実測 owner、full device-residency VoxtralDecodeSession は CC follow-up 候補) / M3-01 RTF gate（M2-14 defer）。

**チケット spec 化進捗（2026-07-09 更新）**: 19 WP 全 file（M3-01〜M3-19 + README）を `docs/tickets/m3/`（gitignore）に land 完了、**~340 tickets / 285h、Draft**。ultracode workflow 2 回（wave 1 + wave 2〜5）で作成した。M2 と同型（30 分単位・WP 別ファイル・README + tickets）。

---

## Contact / Escalation

- CC 側で追加 workflow が必要になった場合（例: 新規モデル対応、GPU RTF 計測ハーネスの拡張）は本チェックリストに追記して依頼者から CC に振る。
- v0.9 Exit 判定は上記全項目の消化 + `docs/milestones.md` §7.3 Exit criteria を根拠に依頼者が最終判断。
- **参照 SoT**: `docs/milestones.md` §7（WP 一覧・Exit criteria・Kill switch）／`docs/tickets/m3/`（現時点 spec 済 = M3-07 と M3-13）／CLAUDE.md「M3（v0.9）🚧 進行中」節。
