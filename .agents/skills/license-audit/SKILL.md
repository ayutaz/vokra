---
name: license-audit
description: Vokra に新しいモデル weight・依存 crate・codec を追加する前後にライセンス/配布可否を監査するときに使う。deny.toml・docs/license-audit.md・NOTICE・model-zoo 除外・zero-dependency ルールを通す手順を示す。
---

# ライセンス audit を通す

Vokra は Unity / Godot / 商用組み込みを標的にするため、ライセンス違反は致命的。**GPL/LGPL 混入 PR はマージ不可**（NFR-LC-04）。単一事実源は `docs/license-audit.md` と `deny.toml`。

## 依存 crate（最優先ルール: zero-dependency）

- **外部 crate を足さない**（NFR-DS-02）。runtime / C ABI / models は **first-party `vokra-*` crate のみ**。まず std / 自前実装で解決できないか検討。
- どうしても必要なら設計レッドライン判断として escalate（`bash scripts/check-zero-deps.sh` が Cargo.lock を検査し、`vokra-*` 以外があれば fail）。
- **`cargo add` は Codex hook でブロック**される（`.codex/hooks.json`）。
- ライセンス許可域: **Apache-2.0 / MIT / BSD 系のみ**。GPL/LGPL は全面禁止。MPL-2.0（例: symphonia）は file-level copyleft を case-by-case 評価。
- **protobuf / prost / onnx / onnxruntime / ort / tract-onnx は deny.toml で ban**（FR-LD-05: runtime は ONNX を絶対にロードしない）。新種が現れたら deny.toml に追記。

## モデル weight

- **CC-BY-NC / CC-BY-NC-SA / 学習データ権利不明 → 公式 model zoo から除外**。engine 対応のみ・research flag で weight 非配布（例: F5-TTS = CC-BY-NC 4.0、Fish-Speech = CC-BY-NC-SA 4.0、EnCodec weight = CC-BY-NC）。
- 商用 OK 候補: DAC(MIT) / Mimi(CC-BY 4.0・**attribution 要**) / WavTokenizer(MIT) / X-Codec2(MIT) / Kokoro(Apache 2.0) / piper-plus(MIT・依頼者作)。
- **Piper（OHF-Voice/piper1-gpl）は非対応**（GPL-3.0 + eSpeak-NG 二重汚染）。**eSpeak-NG（GPL-3.0）も core 非対応**。
- **【2026-07-22 訂正】BigVGAN は MIT**（旧記述「NVIDIA Source Code License-NC → 論文からスクラッチ再実装」の**非商用前提は失効**、reference の直接移植が MIT 帰属表示で可能。AGENTS.md §Vocoder chain 参照）。旧「scratch reimpl」の `NOTICE` §1 記述は現在更新済。

## LicenseClass の SPDX 不整合ケース（converter 側で hard-map）

`vokra-core::compliance::LicenseClass` は SPDX 文字列から派生する（`from_license_str`）が、**SPDX 未登録のライセンス**は `Unknown` に落ち、upload gate 2/4（redistributable）で fail-closed refuse される。上流独自ライセンスの hard-map パターン:

- **CPML（Coqui Public Model License）** = SPDX 未登録。converter 側で SPDX resolver より前に `NonCommercial` に hard-map（XTTS-v2 系、`crates/vokra-convert/` 内 model 実装で明示）。加えて publish は `--allow-noncommercial` 明示 + `fetch_license.sh --url <上流 LICENSE.txt>` で実文書同梱。[[reference-cpml-spdx-nonregistration]]。
- **`from_license_str` の順序 = 特殊 → 一般で書く**（2026-07-23 latent bug 修正）: SA/AGPL/GPL の matcher を plain `cc-by` より先に置かないと `cc-by-sa-4.0` が `AttributionRequired` に誤分類される。新種を追加するときは pin test を追記（既存 pattern に倣う）。
- **HF vocabulary は SPDX と別空間**: `license: MIT`（upper-case）は HF API で 400 reject、`mit` に lower-case + dual 表現は `other`。publish 時は `hf_license_tag()` normalize を通す（skill `publish-model-to-hf` §tier）。

## compliance gate（runtime 強制、オフライン監査を補完）

`docs/license-audit.md` は**オフライン監査**だが、runtime も weight license を強制する。

- `vokra-core/src/compliance/` の `CompliancePolicy` + `LicenseClass` gate が **GGUF の `vokra.provenance.*` metadata**（`weight_license` / `license` / `model_id`）を読み、**CC-BY-NC 等の NC weight を research flag なしでロード拒否**する（`VokraError` を返す）。
- research weight（F5-TTS / Fish-Speech / EnCodec）は **research flag を明示的に立てたときのみ**解禁（`CompliancePolicy::with_research_license`、または config level が Research / Disabled）。既定（Standard）は拒否。
- 新 weight を公式 model-zoo に足すときは converter で `vokra.provenance.weight_license` に正準クラスを焼き込み、gate が読めるようにする（オフライン監査行と一致させる）。

## codec / DSP

- **soxr / rubberband（GPL）禁止** → speexdsp(BSD) / pocketfft(BSD-3) 設計ベースの自前実装。AEC は SpeexDSP(BSD) / WebRTC AEC3 port。

## §3.1 sign-off の primary-source rule（fail-closed default、agent は勝手に埋めない）

`docs/license-audit.md` §3.1 の sign-off 欄は **依頼者（owner）の判断印**。fail-closed default = 空欄のままロックされ続けることが正しい振る舞い。以下 2 条件が **両方揃った時のみ** agent 側で埋めてよい:

1. **依頼者が明示的に「自主判断で埋めてよい」と言った**（session 内の直接発話 or AGENTS.md 記載、暗黙推定は禁止）
2. **primary source で license class が clean と確認できた**:
   - upstream repo の LICENSE ファイル本文（GitHub raw、fork の README ではない）
   - authenticated HF API（`hf_hub_download` の meta 経由、README の「license:」だけを信じない）
   - upstream publish の DOI / 論文ライセンス声明

**署名は依頼者指示に従い `yousan`**（handle）+ 日付。判断根拠を row の右端に「(依頼者許可 = agent 判断)」で明記。片方でも欠けたら **空欄据置** — 許可済み row は monotonic に増えるだけで、fail-closed の default を破らない。

**precedent 例**:
- **CSM-1B ☑ Commercial** (2026-07-28): authenticated HF API で `license=apache-2.0` clean、README "Misuse and abuse" は非拘束 advisory precedent（Bark row 259 と同型）→ CC 判断で埋めた。
- **VibeVoice-Large ☑ Rejected** (2026-07-28): authenticated HF API で HTTP 404 = microsoft が withdraw、Community mirror は権限を継承しないので mirror が何個あっても Rejected → fail-closed 正常機能。
- **X-Codec-2 T4 Research-only** (2026-07-28): cc-by-nc-4.0 primary source、`--allow-noncommercial` 明示必須、初 precedent。
- **Bark row 259 = 非拘束 advisory 前例**: HF `suno/bark` の README「for research purposes」文言は Apache-2.0 の下では拘束力を持たない advisory と判定、以降の類似モデル判定で cite。

[[feedback-license-signoff-primary-source]]。sign-off の実務手順（publish gate との連携）は skill `publish-model-to-hf` §4 を参照。

## 手順（新規追加 PR、同一 PR 内で完結させる）

1. `docs/license-audit.md` に行追加（**code と weight 双方**のライセンス・商用可否・学習データ由来）。§3.1 sign-off 欄は **上記 primary-source rule** に従い、条件未達なら空欄据置。
2. attribution / 配布条件があれば `NOTICE` に追記（credit 要・NC・scratch-reimpl の別を明記）。
3. TTS/VC なら `docs/legal-compliance.md`（EU AI Act Art.50 / SB 942）も通す → skill `add-speech-model`。**watermark / C2PA 埋め込み（FR-CP-01/02）は 2026-07-04 依頼者ドロップで未実装**（`WatermarkConfig` は config 面のみ・`backend_status`=Deferred）。weight license は上記 compliance gate で強制。
4. HF 公開する場合 → skill `publish-model-to-hf`（5-tier gate）。
5. ゲートを走らせる。`cargo deny` / `cargo audit` は workspace を読むため VAST、shell / uv のゲートはローカルでよい:

```bash
# VAST
cargo deny check licenses advisories bans
cargo audit

# ローカル
bash scripts/check-forbidden-symbols.sh
bash scripts/check-zero-deps.sh
uv run --no-project --python 3.12 python scripts/publish/signoff_match.py --self-test
```

CONTRIBUTING.md §3（dependency license policy）/ §4（new model）と突き合わせる。
