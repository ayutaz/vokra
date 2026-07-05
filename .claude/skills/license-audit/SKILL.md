---
name: license-audit
description: Vokra に新しいモデル weight・依存 crate・codec を追加する前後にライセンス/配布可否を監査するときに使う。deny.toml・docs/license-audit.md・NOTICE・model-zoo 除外・zero-dependency ルールを通す手順を示す。
---

# ライセンス audit を通す

Vokra は Unity / Godot / 商用組み込みを標的にするため、ライセンス違反は致命的。**GPL/LGPL 混入 PR はマージ不可**（NFR-LC-04）。単一事実源は `docs/license-audit.md` と `deny.toml`。

## 依存 crate（最優先ルール: zero-dependency）

- **外部 crate を足さない**（NFR-DS-02）。runtime / C ABI / models は **first-party `vokra-*` crate のみ**。まず std / 自前実装で解決できないか検討。
- どうしても必要なら設計レッドライン判断として escalate（`bash scripts/check-zero-deps.sh` が Cargo.lock を検査し、`vokra-*` 以外があれば fail）。
- **`cargo add` は Claude hook でブロック**される（`.claude/settings.json`）。
- ライセンス許可域: **Apache-2.0 / MIT / BSD 系のみ**。GPL/LGPL は全面禁止。MPL-2.0（例: symphonia）は file-level copyleft を case-by-case 評価。
- **protobuf / prost / onnx / onnxruntime / ort / tract-onnx は deny.toml で ban**（FR-LD-05: runtime は ONNX を絶対にロードしない）。新種が現れたら deny.toml に追記。

## モデル weight

- **CC-BY-NC / CC-BY-NC-SA / 学習データ権利不明 → 公式 model zoo から除外**。engine 対応のみ・research flag で weight 非配布（例: F5-TTS = CC-BY-NC 4.0、Fish-Speech = CC-BY-NC-SA 4.0、EnCodec weight = CC-BY-NC）。
- 商用 OK 候補: DAC(MIT) / Mimi(CC-BY 4.0・**attribution 要**) / WavTokenizer(MIT) / X-Codec2(MIT) / Kokoro(Apache 2.0) / piper-plus(MIT・依頼者作)。
- **Piper（OHF-Voice/piper1-gpl）は非対応**（GPL-3.0 + eSpeak-NG 二重汚染）。**eSpeak-NG（GPL-3.0）も core 非対応**。
- **BigVGAN は NVIDIA Source Code License-NC → 論文からスクラッチ再実装**（`NOTICE` §1）。

## compliance gate（runtime 強制、オフライン監査を補完）

`docs/license-audit.md` は**オフライン監査**だが、runtime も weight license を強制する。

- `vokra-core/src/compliance/` の `CompliancePolicy` + `LicenseClass` gate が **GGUF の `vokra.provenance.*` metadata**（`weight_license` / `license` / `model_id`）を読み、**CC-BY-NC 等の NC weight を research flag なしでロード拒否**する（`VokraError` を返す）。
- research weight（F5-TTS / Fish-Speech / EnCodec）は **research flag を明示的に立てたときのみ**解禁（`CompliancePolicy::with_research_license`、または config level が Research / Disabled）。既定（Standard）は拒否。
- 新 weight を公式 model-zoo に足すときは converter で `vokra.provenance.weight_license` に正準クラスを焼き込み、gate が読めるようにする（オフライン監査行と一致させる）。

## codec / DSP

- **soxr / rubberband（GPL）禁止** → speexdsp(BSD) / pocketfft(BSD-3) 設計ベースの自前実装。AEC は SpeexDSP(BSD) / WebRTC AEC3 port。

## 手順（新規追加 PR、同一 PR 内で完結させる）

1. `docs/license-audit.md` に行追加（**code と weight 双方**のライセンス・商用可否・学習データ由来）。
2. attribution / 配布条件があれば `NOTICE` に追記（credit 要・NC・scratch-reimpl の別を明記）。
3. TTS/VC なら `docs/legal-compliance.md`（EU AI Act Art.50 / SB 942）も通す → skill `add-speech-model`。**watermark / C2PA 埋め込み（FR-CP-01/02）は 2026-07-04 依頼者ドロップで未実装**（`WatermarkConfig` は config 面のみ・`backend_status`=Deferred）。weight license は上記 compliance gate で強制。
4. ゲートを走らせる:

```
cargo deny check licenses advisories bans
cargo audit
bash scripts/check-forbidden-symbols.sh
bash scripts/check-zero-deps.sh
```

CONTRIBUTING.md §3（dependency license policy）/ §4（new model）と突き合わせる。
