# M3 (v0.9) Owner Verification Checklist

**Owner**: 依頼者 (`ayutaz`) — 実機テスト・法務判断・鍵/秘密情報の provision を担当。
**CC-side status**（2026-07-09 時点、branch `feat/m3-plan-and-wave1`）: M3 は `docs/milestones.md` §7 に 19 WP (M3-01〜M3-19) が定義済。**チケット spec は現時点で M3-07 (hifigan_generator, 12 tickets / 6h) と M3-13 (RVV 1.0, 12 tickets / 6h) の 2 WP のみ Draft 済**、残 17 WP は rolling wave で着手ごとに spec 化する（M2 パターン、`docs/tickets/` はローカル gitignore 内部計画物）。**Wave 1 = M3-16 / M3-14 / M3-08 / M3-17**（docs-heavy + 小コード WP、依存が薄く並行着手可）の CC 実装を **working tree に land 済（本セッション時点で未 commit）** + **verify 5 面全 green**: cargo build succeeded（14.85s、全 12 crate）／cargo test 全体 = **1104 passed / 0 failed / 4 ignored**（4 ignored は scipy/librosa/torch fixture-gated parity で既存の internal-oracle 規律に沿った意図 skip）／cargo fmt --check clean／cargo clippy `-D warnings` clean（13 crate）／`scripts/check-zero-deps.sh` OK（root Cargo.lock は `vokra-*` のみ、NFR-DS-02 保存）。

**Wave 1 result payload の粒度差（正直な内訳、2026-07-09）**: orchestrator から詳細実装レポートが返ってきたのは **M3-14 と M3-16 のみ**（M3-14 = `Stream::interrupt()` / `InterruptHandle` の 10 in-crate unit tests + 4 hermetic integration tests + 3 C-ABI tests + `vokra_stream_interrupt` cbindgen export、M3-16 = `docs/abi-changelog.md` schema + `docs/abi/vokra.h.v0.9-baseline.symbols` machine-anchor + `scripts/check-abi-changelog.sh` の verify/list/update-snapshot/self-test/help modes と comment-strip + brace-aware `;`-splitter、cbindgen banner の M3-16/M4-12 参照、`gen-c-abi.sh --check` clean を報告）。**M3-08 / M3-17 は report text 未提供** — working tree の実ファイル存在（M3-08 = `crates/vokra-ops/src/length_conditioning.rs` 326 行 + tests `length_conditioning_ir_distinction.rs` 188 行 + `parity_length_conditioning.rs` 156 行 + attrs/dispatch/lib 配線 + `ir/graph.rs` 変更、M3-17 = `crates/vokra-ops/src/prosody.rs` 440 行 + attrs/dispatch/lib 配線）と ops crate export（`pub mod length_conditioning; pub use prosody::{ApplyProsody, ProsodyControl};`）と 1104 tests 全 pass で完了状態を裏付ける（証跡は `git status` + `cargo test` 出力）。**残る CC 側タスク**: 4 WP を分割コミット + push、Wave 1 4 WP の spec を `docs/tickets/m3/` に retro 追加（rolling wave 規律遵守）、Wave 2 着手。以下のチェックポイントは Wave 1 commit + push 後および M3 進行中に依頼者が消化する項目群。

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

## Summary 進捗表（2026-07-09 時点、Wave 1 実装未捕捉 = null）

| WP | 内容 | CC 進捗 | 依頼者残タスク |
|----|------|--------|--------------|
| M3-01 | CUDA 完成 + RTF<0.1 formal gate | ❌ 未着手 / spec 未 land | § 6（M2-14 self-hosted runner + RTF gate) |
| M3-02 | Vulkan バックエンド | ❌ 未着手 / spec 未 land | § 1（Android 実機 RTF）|
| M3-03 | paged KV cache | ❌ 未着手 / spec 未 land / **Wave 2 候補** | — |
| M3-04 | KV cache 量子化 | ❌ 未着手 / spec 未 land / **Wave 2 候補** | — |
| M3-05 | flow_sampler + ODE solver | ❌ 未着手 / spec 未 land / **Wave 2 候補** | — |
| M3-06 | mimi_rvq codec | ❌ 未着手 / spec 未 land | § 3（CosyVoice2 audit と一括、Mimi CC-BY 4.0 attribution NOTICE 記載）|
| M3-07 | hifigan_generator op | ✅ Draft spec land 済（`docs/tickets/m3/M3-07-hifigan-generator.md`、12 tickets / 6h、実装未着手）| — |
| M3-08 | length_conditioning op | 🟢 **Wave 1 実装 land 済（working tree、未 commit）** / `crates/vokra-ops/src/length_conditioning.rs` 326 行 + tests 2 本（IR 区別 + parity） / spec は retro 追加予定 | — |
| M3-09 | CosyVoice2 統合 | ❌ 未着手 / spec 未 land | § 3（audit）|
| M3-10 | Voxtral 統合 | ❌ 未着手 / spec 未 land | § 3（audit）|
| M3-11 | Godot GDExtension | ❌ 未着手 / spec 未 land | § 2（実機動作確認、M3-18 と連動）|
| M3-12 | piper-plus native の GPU 対応 | ❌ 未着手 / spec 未 land | — |
| M3-13 | RVV 1.0 基本対応 | ✅ Draft spec land 済（`docs/tickets/m3/M3-13-rvv-1.0-basic.md`、12 tickets / 6h、実装未着手）| — |
| M3-14 | barge-in（stream.interrupt()）| 🟢 **Wave 1 実装 land 済（working tree、未 commit）** / `Stream::interrupt()` + `InterruptHandle`（`Arc<AtomicBool>` + `Clone+Send+Sync`）+ `EventPoller::drain_all()` + `vokra_stream_interrupt` C ABI + 10 unit + 4 integration + 3 C-ABI tests / spec は retro 追加予定 | — |
| M3-15 | vokra-server multi-session + 75ms | ❌ 未着手 / spec 未 land | § 5（サーバ 75ms 実測）|
| M3-16 | v0.9 ABI 変更点の changelog 記録（凍結は M4-12 へ移動）| 🟢 **Wave 1 実装 land 済（working tree、未 commit）** / `docs/abi-changelog.md` schema + `docs/abi/vokra.h.v0.9-baseline.symbols` machine-anchor + `scripts/check-abi-changelog.sh`（verify/list/update-snapshot/self-test/help modes、zero-dep = bash+awk+grep+diff）+ cbindgen banner に M3-16/M4-12 参照 / spec は retro 追加予定 | — |
| M3-17 | prosody_control 統一 API | 🟢 **Wave 1 実装 land 済（working tree、未 commit）** / `crates/vokra-ops/src/prosody.rs` 440 行（`ApplyProsody` + `ProsodyControl`、attrs/dispatch/lib 配線） / spec は retro 追加予定 | — |
| M3-18 | 実機テスト: Android + Godot | 依頼者ボトルネック | § 1 + § 2 |
| M3-19 | Kill switch D + 四半期 review | 依頼者ボトルネック（暦月 2027-03〜05 頃）| § 4 |
| M2-14 carry-over | iOS 実機 RTF | 引き渡し済み | § 7 |

**Wave 1 (M3-16 / M3-14 / M3-08 / M3-17) 実装状況（正直な報告、2026-07-09 更新）**: **4 WP すべての実装 land 済（working tree、未 commit）** + **verify 5 面全 green**: cargo build 14.85s ok（全 12 crate）／cargo test 全体 = **1104 passed / 0 failed / 4 ignored**（4 ignored は scipy/librosa/torch fixture-gated parity で既存 internal-oracle 規律通り）／cargo fmt --check clean／cargo clippy `-D warnings` clean（13 crate）／`scripts/check-zero-deps.sh` OK（root Cargo.lock は `vokra-*` のみ、NFR-DS-02 保存）。**粒度差の正直な内訳**: orchestrator から詳細実装レポートが返ってきたのは M3-14（`Stream::interrupt()` / `InterruptHandle` の tests 一覧・生成ファイルパス・per-check 表付き）と M3-16（`docs/abi-changelog.md` schema + machine-anchor snapshot + `scripts/check-abi-changelog.sh` の 4 modes 検証表付き）のみ。**M3-08 と M3-17 は report text 未提供** — working tree の実ファイル（M3-08 = 326 行 + tests 2 本 344 行、M3-17 = 440 行）と ops crate 側の export（`pub mod length_conditioning; pub use prosody::{ApplyProsody, ProsodyControl};`）と 1104 tests 全 pass で完了状態を裏付ける（証跡は `git status` + `cargo test` 出力）。**残る CC 側タスク**: 4 WP を分割コミット + push（M2 パターンの feedback「適度にコミットする」に準拠 = 完成したまとまりごとに段階コミット）、Wave 1 4 WP の spec を `docs/tickets/m3/` に retro 追加（rolling wave 規律遵守）、Wave 2 着手。M2 の per-WP CI green（28/28）に相当する Wave 1 完了根拠は verify 出力で担保、CI 側の per-PR green は commit + push 後に確定する。

**チケット spec 化進捗**: 19 WP 中 **2 WP（M3-07 / M3-13）のみ Draft spec land 済**（M2 パターン、30 分単位、`docs/tickets/m3/`）。残 17 WP は rolling wave で着手時に spec 化する。

---

## Contact / Escalation

- CC 側で追加 workflow が必要になった場合（例: 新規モデル対応、GPU RTF 計測ハーネスの拡張）は本チェックリストに追記して依頼者から CC に振る。
- v0.9 Exit 判定は上記全項目の消化 + `docs/milestones.md` §7.3 Exit criteria を根拠に依頼者が最終判断。
- **参照 SoT**: `docs/milestones.md` §7（WP 一覧・Exit criteria・Kill switch）／`docs/tickets/m3/`（現時点 spec 済 = M3-07 と M3-13）／CLAUDE.md「M3（v0.9）🚧 進行中」節。
