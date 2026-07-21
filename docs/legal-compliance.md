# legal-compliance.md — Vokra 音声 AI 法務対応

**最終更新**: 2026-07-07（§1.4 に watermark embedding が Deferred の期間の deployment-side disclosure 必須運用を追記）
**目的**: EU AI Act、California SB 942、Tennessee ELVIS Act、連邦 NO FAKES Act、Apple App Store Guideline 5.5、Google Play Generative AI Content 等の音声 AI 特有の法的要件に対し、Vokra が実装すべき機能・運用・ドキュメントを列挙する。

**責任分界**:
- **Vokra provider (依頼者)**: SDK の基本機能提供、ドキュメント整備、default 設定の適切性
- **Vokra deployer (ゲーム開発者、SaaS 事業者等)**: 具体的なデプロイ環境での compliance 実装、user consent 取得、地域別対応

---

## 1. EU AI Act Article 50 (Transparency Obligations)

### 施行スケジュール
- **2026-08-02**: **主要 enforcement 日**
- **2026-12-02**: 2026-08-02 以前の既存製品の猶予期限

### 対象範囲
- **Article 50(2)**: AI 生成コンテンツ (音声、動画、テキスト、画像) に対する **machine-readable な marking** と **detectable な形での AI 生成表示** を義務化
- **Article 50(3)**: deepfake の場合、**明示的な disclosure** が deployer に義務
- **Article 50(4)**: emotional recognition / biometric categorization 使用時の transparency

### Vokra 実装要件

#### 1.1 Machine-readable marking (Vokra 必須実装)

- **AudioSeal (Meta, MIT)** を推奨デフォルトの watermark として組込
  - `audio_dialect.audioseal_embed(waveform, model_id)` op を第一級提供
  - 埋込 payload: `{version, model_id, timestamp, session_id, license_tier}`
  - detector: `audio_dialect.audioseal_detect(waveform) -> {is_ai: bool, model_id: str, confidence: float}`

- **C2PA 2.1 (ISO/IEC 22144)** manifest 埋込
  - Adobe `c2pa-rs` (Apache 2.0) 統合
  - `audio_dialect.c2pa_manifest(waveform, manifest_json) -> waveform_with_manifest`
  - PCM ファイル出力時に自動的に manifest 埋込 (opt-out 可)

- **SynthID audio (Google DeepMind)** 対応検討
  - Google DeepMind との個別ライセンス契約が必要 (エッジ推論のため API 経由不可)
  - **OSS 代替**: **SilentCipher** (arxiv 2404.03410、Apache 2.0)、**WaveGuard** を評価
  - v1.0 で SynthID 代替として SilentCipher 実装検討

#### 1.2 Detectable 表示 (Vokra API + deployer 責任)

- **Vokra API**:
  - `TTS.synthesize(text, options)` の `options.watermark_enabled: bool = true` (**デフォルト ON**)
  - `options.disclosure_text: Option<String>` — 発話冒頭/末尾に "This voice is AI-generated" 相当のアナウンス
  - `options.disclosure_audio_beacon: bool = false` — 特定周波数の音声 beacon 埋込

- **deployer 責任**:
  - UI 上での視覚的 AI 表示 (例: サーバサイド API サーバー / Unity SDK / Web アプリが "🤖 AI-generated voice" を発言前に表示)
  - EU 地域判定に基づく強制表示
  - user consent 取得ダイアログの提供

#### 1.3 ドキュメント責任

- Vokra README に **"Vokra implements EU AI Act Article 50 compliance mechanisms. Deployers are responsible for user-facing disclosure in their applications."** を明記
- `docs/legal-compliance.md` にこの文書を配置し、SDK ユーザーに周知

#### 1.4 Deployment-side disclosure requirement (watermark embedding が Deferred の期間、必須運用)

§1.1 の machine-readable marking（AudioSeal / C2PA / SynthID / SilentCipher）は §8 実装状況のとおり **2026-07-04 依頼者ドロップにより Deferred**（`crates/vokra-core/src/compliance/watermark.rs` の `WatermarkConfig::backend_status()` は常に `WatermarkBackendStatus::Deferred` を返し、runtime は生成音声に watermark を埋め込まない）。この deferred 期間、**Vokra 生成音声に対し Article 50(2)/(3) の "detectable" AI 生成 cue を提供する唯一の手段は、deployer 側の UI レベルでの visible AI-generated disclosure である**。本節は §1.2 の deployer 責任・§1.3 の README 記述・§11 の deployer checklist に散在する運用義務を、埋め込みが復活するまでの **MUST 運用要件**として集約する（法的分析の新規追加ではなく、既存要件の consolidation）。

- **a. Vokra TTS/VC 音声の再生 UI に、ユーザーが視認可能な "AI-generated voice" 表示を必ず表示する** — §1.2 の "UI 上での視覚的 AI 表示" と `crates/vokra-core/src/compliance/level.rs` の `DisclosureConfig::require_visible_ui = true`（default）に対応。表示形態（アイコン・テキスト・ラベル）は deployer 判断だが「表示すること自体」は EU AI Act Article 50(2)/(3) が deployer に義務付ける。
- **b. EU 地域配信では user 設定によらず常時表示** — §1.2 の "EU 地域判定に基づく強制表示" と [`docs/system-requirements.md`](system-requirements.md) NFR-LG-01（EU AI Act Article 50、2026-08-02 enforcement）に対応。deployer は EU 地域判定機構と強制表示 UI を実装する（Vokra core の自動地域判定は §8 実装状況のとおり locale ヒント最小版のみ、IP geolocation は据え置き）。
- **c. California SB 942 対象サービスは manifest disclosure workflow も併走** — §2 の "If your service exceeds 1M CA users/month, SB 942 applies." に対応し、月間 100 万 CA ユーザー超のサービスは SB 942（2026-01-01 施行）対象で、視覚的 disclosure に加え SB 942 の manifest-disclosure workflow を deployer 側で実装する。
- **d. Tennessee ELVIS Act（2024-07-01 施行）／ 連邦 NO FAKES Act（審議中）** — §3 / §4 のとおり、deployer は Vokra を「identifiable な個人音声の consent なき合成」に使用しない。これは disclosure 項目ではなく use-restriction だが、deployer の compliance stack は disclosure と use-restriction を組み合わせて成立するため §1.4 で cross-reference する。
- **e. 音声録音 / speaker embedding 抽出時の consent-taking dialog** — §1.2 の "user consent 取得ダイアログの提供" と §3.2 の signed consent manifest 要件に対応し、deployer が実装する。

**Owner 責任の再確認**: visible indicator の具体的文言・地域強制の実装方式・consent workflow の法的十分性は **deployer / 依頼者の判断**（FR-MD-13 / X-03、[`docs/m2-owner-verification-checklist.md`](m2-owner-verification-checklist.md) §4 の Article 50 checklist owner sign-off point）。Vokra は `ComplianceConfig::disclosure.require_visible_ui = true`（`crates/vokra-core/src/compliance/level.rs`）を設計意図としての default に据えるが、UI 実装・法的レビュー・地域判定の enforcement は Vokra は提供しない。§8 実装状況 の "法務的十分性の判断は FR-MD-13 / X-03（依頼者）に従属" 原則をここでも保持する（本節は既存の deployer 責務を集約するのみで、新規の法的義務を課すものではない）。

**関連参照**: §1.1（machine-readable marking、現状 Deferred）／ §1.2（deployer 責任 bullets、本節が MUST 化）／ §1.3（README 記述）／ §8 実装状況（Deferred 事実・`WatermarkConfig::backend_status()` の復帰接続点）／ §11 deployer checklist（項目別）／ §2（SB 942）／ §3（ELVIS Act）／ §4（NO FAKES Act）／ [`docs/license-audit.md`](license-audit.md) §Article 50 checklist（"運用側で TTS 出力に AI 生成表示（disclosure text）を必ず加える必要がある" 記述）／ `crates/vokra-core/src/compliance/level.rs`（`DisclosureConfig::require_visible_ui`）／ `crates/vokra-core/src/compliance/watermark.rs`（`WatermarkConfig::backend_status()` が `Deferred` を返す事実）。

### 罰則リスク
- Article 50 違反: **全世界売上高の 3% or €15M のいずれか高い方** の罰金
- Otonx (現 Vokra) を組み込んだアプリで disclosure 不備 → deployer が罰金対象、Vokra も contributory liability の可能性

---

## 2. California SB 942 (California AI Transparency Act)

### 施行
- **2026-01-01 施行済**

### 対象範囲
- 生成 AI ツール提供者に (a) latent disclosure (watermark 等) と (b) manifest disclosure の実装を義務化
- 対象: 月間 100 万ユーザー以上の generative AI システム

### Vokra 対応
- Vokra 自体は SDK / OSS でエンドユーザーサービスではないため直接的な該当は少ないが、Vokra を組み込む商用サービスは対象
- **1.1 の AudioSeal + C2PA + SB 942 の "latent disclosure" 要件を満たす**
- SDK ユーザー向け ドキュメントに **"If your service exceeds 1M CA users/month, SB 942 applies. Vokra's default watermark satisfies latent disclosure requirements."** を明記

---

## 3. Tennessee ELVIS Act (Ensuring Likeness Voice and Image Security Act)

### 施行
- **2024-07-01 施行**

### 対象範囲
- 個人の **voice / likeness の unauthorized commercial exploitation** を禁止
- **"a tool whose 'primary purpose or primary effect' is to produce such unauthorized replicas"** を **knowingly distribute** する者まで責任範囲を拡大 → **tool-distributor liability**

### Vokra 対策 (最重要)

#### 3.1 Voice cloning 機能を Vokra-core から完全分離
- **`vokra-voiceclone-experimental` 別リポジトリ** に分離
- Vokra-core は VAD / ASR / TTS (pre-trained voice のみ) をパッケージ
- README に **"Vokra is not primarily designed for voice cloning. Voice cloning experimental tools are provided under a separate repository with explicit consent requirements."** を明記

#### 3.2 Speaker embedding は core に残すが機能限定
- `speaker_encode` op は core に含める (現代 zero-shot TTS の必須入力)
- ただし **任意音声への転写 (voice cloning) 機能は含めない**
- `speaker_encode` の出力 embedding を TTS に渡す場合、**signed consent manifest** (WAV に埋込) を要求
- 未署名 embedding での TTS 生成は API level で reject

#### 3.3 Explicit consent workflow
- `vokra-voiceclone-experimental` は起動時に consent 確認 UI を表示
- consent manifest 例:
  ```json
  {
    "voice_owner_name": "Yamada Taro",
    "consent_scope": "commercial|personal|research",
    "grant_date": "2026-07-02",
    "signature": "PGP or similar cryptographic signature",
    "vokra_session_id": "uuid"
  }
  ```
- 生成した音声の全 payload に上記 manifest hash を C2PA + AudioSeal で埋込

### 罰則リスク
- ELVIS Act 違反: 民事責任 (right of publicity 訴訟) + 州司法長官による差止め
- distributor liability により Vokra 開発者本人が被告になる可能性

---

## 4. 連邦 NO FAKES Act (Nurture Originals, Foster Art, Keep Entertainment Safe Act)

### 進捗
- **2025-04 再導入 (119th Congress)**、2026-07 時点で 上下両院で審議中
- 施行時期は現時点で未確定、2027 施行予想

### 対象範囲
- ELVIS Act の連邦版
- **AI 生成 digital replica の unauthorized production / distribution** を全米で禁止
- tool-distributor liability を含む

### Vokra 対応
- ELVIS Act 対策 (3.1-3.3) を全米で有効化
- 連邦法施行後は米国内での voice cloning 機能配布を全面停止する可能性を考慮

---

## 5. Apple App Store Guideline 5.5 (2025-)

### 対象範囲
- Generative AI を含む iOS/macOS/tvOS/watchOS/visionOS アプリは App Review Metadata で "AI-generated content" declaration を必要

### Vokra 対応
- Vokra API に **`is_ai_generated: bool = true`** metadata の pass-through 経路
- Unity Package Manager 経由の Vokra iOS SDK に対応する Info.plist 拡張:
  ```xml
  <key>NSAIGeneratedContent</key>
  <true/>
  ```
- Vokra 提供 Swift Package の README に "Add 'AI voice included' to App Store Metadata" 記述

---

## 6. Google Play Generative AI Content ポリシー (2024-06 施行)

### 対象範囲
- Generative AI apps に **user report mechanism** と **guardrails** 義務化
- 対象: すべての Android generative AI apps

### Vokra 対応
- Vokra API に **`report_generated_content(content_hash, reason)`** 経路
- Unity SDK / サーバサイド API サーバー / モバイル SDK に **"Report AI Voice"** ボタンサンプルコード提供
- deployer ドキュメントで Google Play submission checklist を提供

---

## 7. 日本国内法対応

### 個人情報保護法 (2022 改正)
- Vokra が音声から抽出する speaker embedding は **個人識別符号** に該当する可能性
- SDK ユーザーは 匿名化・仮名化 / 目的明示 / 同意取得 の workflow を実装する必要
- Vokra API に `speaker_embedding_anonymize(embedding, k)` (k-anonymity 相当) op を検討

### 声優・タレント肖像権
- 日本には ELVIS Act 相当の音声保護法はないが、パブリシティ権侵害の判例 (ピンク・レディー事件、キング・クリムゾン事件) が適用可能
- VTuber キャラクターの声質模倣は事務所との個別契約が必須 (VOICEVOX/A.I.VOICE/CoeFont はキャラ別ライセンス)
- Vokra 公式ドキュメントに日本語音声モデル使用時の **キャラクター別ライセンス確認 workflow** を掲載
- 例: VOICEVOX ずんだもんの利用規約 (素材販売禁止、キャラ別)、A.I.VOICE、CoeFont、VOICEROID、UTAU

---

## 8. Vokra API での自動 compliance mode

### `Vokra::init(config)` の compliance 設定

```rust
Vokra::init(VokraConfig {
    compliance: ComplianceLevel::Strict, // EU AI Act + SB 942 + ELVIS + JP 個情法
    watermark: WatermarkConfig {
        audioseal: true,          // default ON
        c2pa: true,               // default ON
        synthid: false,           // Google 契約要
        silent_cipher: true,      // OSS 代替 (v1.0+)
    },
    voice_cloning: VoiceCloningPolicy::Disabled, // core では常に Disabled
    speaker_embedding: SpeakerEmbeddingPolicy::RequireConsent, // consent manifest 必須
    disclosure: DisclosureConfig {
        default_beacon_frequency_hz: 22050, // 人耳外の高周波 beacon
        require_visible_ui: true,
    },
})
```

### Compliance level

- **Strict** (default): 全 watermark ON、voice cloning disabled、speaker embedding は consent 必須、EU/CA/TN 地域自動判定
- **Standard**: watermark ON、voice cloning は vokra-voiceclone-experimental 経由のみ、speaker embedding OK
- **Research**: watermark opt-out 可、consent manifest 不要、research flag のモデル (F5-TTS/Fish-Speech CC-BY-NC) が使える
- **Disabled**: compliance 全無効 (self-responsibility、README に大警告)

### 自動地域判定

- Vokra runtime に system locale + IP geolocation (opt-in) 経由の地域判定
- EU 地域: Strict 強制 (deployer override 不可)
- CA (米カリフォルニア): Strict 推奨
- TN (米テネシー): voice cloning 全機能無効
- JP: 声優ライセンス警告表示

### 実装状況（M2-13、2026-07-04）

上記スケッチの **compliance 設定 API を `crates/vokra-core/src/compliance/` の型として実装**した（FR-CP-06）。実装と本スケッチの対応・乖離:

- **`ComplianceLevel`（Strict/Standard/Research/Disabled）・`WatermarkConfig`・`VoiceCloningPolicy`・`SpeakerEmbeddingPolicy`・`DisclosureConfig`（beacon 22050Hz）を型として提供**（default = Strict、voice_cloning は core で常時 `Disabled`＝単一 variant で表現不能化、speaker_embedding = `RequireConsent`）。init 統合点は `Vokra::init` グローバルではなく、当面 **model ローダーへ明示的な `CompliancePolicy` を渡す**形で配線（SRS の Session 中心 API と整合、グローバル init は据え置き）。
- **research flag の実挙動**: `ComplianceLevel::Research`（または `with_research_license(true)` / `VOKRA_ALLOW_RESEARCH_LICENSE=1`）が CC-BY-NC 系 weight（F5-TTS/Fish-Speech/EnCodec）を解錠する。Strict/Standard は解錠せず `VokraError::ResearchLicenseRequired` で拒否（fail-closed、`docs/license-audit.md` §3 参照）。
- **watermark は config 面のみ・埋め込みは据え置き（2026-07-04 依頼者ドロップ）**: `WatermarkConfig` は default ON の設計意図（audioseal/c2pa=true・synthid=false・silent_cipher=true）と opt-out 経路を保持するが、埋め込みバックエンド（AudioSeal/C2PA、旧 M1-07）は未実装。`WatermarkConfig::backend_status()` は `Deferred` を返し、**「埋め込み済み」と偽装しない**。したがって **EU AI Act Article 50（NFR-LG-01）/ SB 942（NFR-LG-02）の marking 義務は現時点で未充足**。復帰接続点は `backend_status()` を将来 `Active` に変える 1 箇所。法務的十分性の判断は FR-MD-13 / X-03（依頼者）に従属。
- **自動地域判定は locale ベース最小版のみ・IP geolocation は据え置き**: zero-dep 不変条件（NFR-DS-02）維持のため geoip 系 crate/DB を core に追加しない。locale ヒントによる Strict 強制/警告は後続の最小実装に委ね、実際の地域確定は deployer 責務（本節の EU 強制は deployer が最も安全側に倒す前提）。

---

## 9. Copyright / Training Data Provenance リスク (レビュアー D 指摘 R1)

### 現状の法的動向 (2026-07)
- **Kadrey v. Meta** (2024): AI 訓練データの著作権侵害訴訟、Meta 側は一部 fair use 主張が却下
- **Concord Music v. Anthropic** (2024): 楽曲歌詞の訓練データ使用に関する訴訟
- **RIAA vs Suno / Udio** (2024): 音楽生成 AI の訓練データが RIAA 加盟レーベル楽曲を含むと主張
- Section 230 は AI 生成コンテンツに適用されないという 2025 の判例動向

### Vokra 方針
- **公式 model zoo は "training data source が公開されている" model のみ**:
  - ✅ Kokoro (LibriTTS + custom)
  - ✅ CosyVoice (Alibaba 公開データセット)
  - ✅ Sesame CSM (英語主体音声 100万時間、公開明記)
  - ✅ piper-plus (依頼者作、training data 公開)
  - ✅ Whisper (OpenAI 公開 web-scraped、controversial だが Meta v Kadrey で fair use 傾向)
- **community models** は下流責任 disclaimer 付きで別途配布:
  - ⚠️ Bark (Suno、training data 非公開。code+weight とも現行 MIT = https://github.com/suno-ai/bark / https://huggingface.co/suno/bark、HF model card「research purposes only」付記、2026-07-21 CC fetch)
  - ⚠️ Fish-Speech (fishaudio、training data 一部非公開)
  - ⚠️ RVC 系派生 (learned from various)
  - ⚠️ StyleTTS 2 (yl4579、code MIT = https://github.com/yl4579/StyleTTS2。pretrained weight は voice-consent/disclosure 条件付き usage agreement、学習データは LJSpeech/VCTK/LibriTTS = README 記載、2026-07-21 CC fetch)
- **audit 材料整備済・tier 未確定（owner 判断待ち、M5-07）**:
  - Matcha-TTS: code MIT (https://github.com/shivammehta25/Matcha-TTS の LICENSE、2026-07-21 CC fetch)。配布 checkpoint は README に別段の license 記載なし（Google Drive 自動 DL）。学習データは README が LJ Speech を例示するのみで、released checkpoint の provenance を網羅記載していない。→ ✅（zoo 適格）/ ⚠️（community 別途配布）の最終 tier 確定と zoo 可否は owner（`docs/license-audit.md` §3.1 の Matcha sign-off + 本 §9 の tier 判断）。CC は事実記録のみ（tier を確定しない）

### Contributory infringement 対策
- Otonx が single 実装として "training data agnostic runtime" である旨を README/NOTICE で明記
- **"Vokra is a general-purpose inference runtime. Users are responsible for the licensing and legality of models they load."** 相当

---

## 10. サーバサイド SaaS デプロイ時の追加要件

### GDPR (EU)
- 音声データは **biometric data (Article 9)** に該当、明示的同意必須
- Data Processing Agreement (DPA) 必須
- Vokra サーバサイドデプロイガイドで DPA テンプレート提供

### HIPAA (米医療)
- 医療用途の Vokra サーバサイド deploy は NVIDIA CUDA EULA "not for medical" 制約に抵触
- **HIPAA 準拠は CPU/Vulkan-only ビルド ("Vokra-critical-safe" SKU)** で対応

### PCI DSS (決済)
- 通話録音を扱うサーバは PCI DSS 準拠が必要な場合あり
- Vokra provider の直接責任ではないが、deployer ドキュメントで警告

---

## 11. Vokra 提供の Compliance Checklist (deployer 向け)

新規 Vokra 統合プロジェクト開始時に確認すべき checklist:

```
□ AudioSeal watermark ON (Compliance Level: Strict または Standard)
□ C2PA manifest 埋込 有効
□ SB 942 (CA): 月間 100 万 CA user 超えるか確認 → watermark 必須
□ EU AI Act: 2026-08-02 以降 EU 地域配信するか → disclosure UI 実装
□ ELVIS Act (TN): TN 州向け voice cloning 機能を提供しない
□ NO FAKES Act: 施行後は米国全土で voice cloning 停止準備
□ Apple App Store 5.5: iOS/macOS アプリ Metadata で AI generated 表示
□ Google Play generative AI: user report mechanism 実装
□ 日本個情法: speaker embedding の匿名化/仮名化/同意取得
□ 日本声優: VOICEVOX/A.I.VOICE 等はキャラ別ライセンス確認
□ GDPR (EU): 音声 biometric data DPA 準備
□ HIPAA (医療): CUDA 経由でなく CPU/Vulkan-only SKU 使用
□ NVIDIA CUDA EULA: cudart bundle しない、system install 検出のみ
□ Training data provenance: community model は下流責任表示
□ Consent manifest: voice cloning experimental は署名付き manifest 必須
```

### 11.1 M5-05 通過記録（voice cloning experimental / FR-MD-13 / X-04(c)、2026-07-21）

M5-05（`vokra-voiceclone-experimental` 分離準備）は FR-MD-11（RVC v2 / GPT-SoVITS）が実際に配布される WP であり、FR-MD-13（新規モデル対応 PR ごとの license-audit 追記 + 本 §11 checklist 通過）の到来点である。CC 実装分（T05-T09, T13）が本 checklist の各項目をどこまで満たすかを**詐称せず**記録する（判定・sign-off は owner = T15）:

- **□ Consent manifest（`:311`「voice cloning experimental は署名付き manifest 必須」）— 部分充足（構造検証まで）**:
  - **満たす**: consent manifest の schema（`ConsentManifest` / `ConsentScope`、§3.3 の 5 field を転記）と **構造検証**（`ConsentManifest::parse` = field presence / scope enum 妥当性 / `vokra_session_id`・`grant_date` 非空 / `signature` field の存在、fail-closed reject）を `crates/vokra-core/src/compliance/consent.rs`（M5-05-T05/T06）に実装。`SpeakerEmbeddingPolicy::RequireConsent` を consent 型に接続し、未署名（signature 非存在＝空）manifest を API level で reject（§3.2、M5-05-T07）。別リポ scaffold binary が両 flag（`--i-understand-risks --research-only`）+ 署名付き consent を start-up gate で強制（M5-05-T08/T09）。
  - **満たさない（owner 待ち）**: **cryptographic signature 検証**は core 非対応（`SignatureStatus` は `Present`/`Absent` の構造判定のみ、`Verified` variant を作らない）。理由は zero-dep 制約ではなく (1) PGP/Ed25519 署名検証は security-critical で自前実装が不適切、(2) 信頼根（誰の鍵・配布・失効）が owner 決定であるため。署名検証方式の確定は M5-05-T04（owner）。
- **□ AudioSeal watermark ON / □ C2PA manifest 埋込（`:297`/`:298`）— 未充足（embedding Deferred）**: §1.4 / §8 のとおり watermark embedding は 2026-07-04 ドロップで Deferred（`WatermarkConfig::backend_status()==Deferred`）。voice cloning binary は watermark config を強制する意図だが、embedding backend が Deferred のため「強制埋込」leg は honest-UNMET（M5-05-T09 の leg 3、`docs/adr/M5-05-watermark-dependency.md` の owner resolution 待ち）。deferred 期間の Article 50 "detectable" cue は §1.4 の deployer 側 visible disclosure が唯一の手段（MUST）。
- **□ ELVIS Act / □ NO FAKES Act（`:301`/`:302`）— 機構は用意、法務判断は owner**: voice cloning を別リポ（`vokra-voiceclone-experimental`）に分離し core 配布経路から遮断（§3.1、FR-CP-04）。README に §3.1 warning + use-restriction を転記。tool-distributor liability の十分性判断は §1.4(d) のとおり owner/deployer（NFR-LG-02、M5-05-T04）。
- **□ Training data provenance（`:310`）— 下流責任表示を記載**: RVC v2 / GPT-SoVITS の「不明 / 学習権利疑い」を scaffold `NOTICE` と `docs/license-audit.md` §3.1（空欄 sign-off = fail-closed）に記録。

## 12. 罰則 / 損害額 まとめ

| 法令 | 主要罰則 |
|-----|-------|
| EU AI Act Article 50 | 全世界売上高の 3% or €15M いずれか高い方 |
| California SB 942 | 民事罰 + 差止め、規模に応じ |
| ELVIS Act (TN) | 民事責任、州司法長官差止め |
| NO FAKES Act (連邦、審議中) | 民事+刑事の可能性 |
| Apple App Store 5.5 違反 | App Rejection + Developer Program 停止リスク |
| Google Play 生成 AI ポリシー違反 | App 削除、Developer Account 停止 |
| GDPR | 全世界売上高の 4% or €20M |
| HIPAA | civil $50K/violation、criminal 10 年拘禁 |
| 日本個情法 | 1 億円以下罰金 (法人) |

---

## 13. 参考出典

- [EU AI Act Article 50 — Transparency Obligations](https://artificialintelligenceact.eu/article/50/)
- [What Actually Comes Due on August 2, 2026 — ComplianceHub](https://compliancehub.wiki/eu-ai-act-article-50-transparency-digital-omnibus-2026/)
- [ELVIS Act — Wikipedia](https://en.wikipedia.org/wiki/ELVIS_Act)
- [ELVIS Act Alston & Bird](https://www.alstonprivacy.com/tennessee-law-designed-to-combat-deepfakes-set-to-take-effect-in-july/)
- [NO FAKES Act — Temple 10-Q](https://law.temple.edu/10q/the-clone-wars-a-new-congress-reconsiders-the-no-fakes-act-to-combat-digital-deepfakes/)
- [SynthID / C2PA 2026 status — InfoQ](https://www.infoq.com/news/2026/05/google-synthid-content-detection/)
- [C2PA 2.1 ISO/IEC 22144](https://c2paviewer.com/articles/openai-google-c2pa-synthid-2026)
- [California SB 942 (2024)](https://leginfo.legislature.ca.gov/faces/billTextClient.xhtml?bill_id=202320240SB942)
- [Meta AudioSeal](https://github.com/facebookresearch/audioseal)
- [c2pa-rs (Adobe)](https://github.com/contentauth/c2pa-rs)
- [Google Play Generative AI Content Policy](https://support.google.com/googleplay/android-developer/answer/13985936)
- [Apple App Store Guideline 5.5](https://developer.apple.com/app-store/review/guidelines/#5.5)
- [NVIDIA CUDA EULA](https://docs.nvidia.com/cuda/eula/index.html)
- [Kadrey v. Meta case](https://en.wikipedia.org/wiki/Kadrey_v._Meta)
- [RIAA vs Suno/Udio suit](https://www.riaa.com/riaa-and-major-record-labels-sue-suno-and-udio/)
- [VOICEVOX 利用規約](https://voicevox.hiroshiba.jp/term/)
