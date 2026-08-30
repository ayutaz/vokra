# legal-compliance.md — Vokra 音声 AI 法務対応

**最終更新**: 2026-08-30（公式一次資料の確認日）
**目的**: EU AI Act Article 50、California SB 942、Tennessee ELVIS Act、連邦 NO FAKES Act、Apple App Review Guidelines、Google Play AI-Generated Content policy について、Vokra provider と deployer が法務確認すべき論点と現行実装の境界を記録する。

**重要な注意**: 本書は法的助言ではなく、Vokra が法令・ストア規約を満たすことを表明するものでもない。適用範囲、役割（provider/deployer）、地域、例外、契約、実際の音声出力を弁護士または各プラットフォームで確認すること。以下の「要法務確認」は未確定事項として扱う。

**責任分界**:
- **Vokra provider (依頼者)**: SDK の基本機能提供、ドキュメント整備、default 設定の適切性
- **Vokra deployer (ゲーム開発者、SaaS 事業者等)**: 具体的なデプロイ環境での compliance 実装、user consent 取得、地域別対応

---

## 1. EU AI Act Article 50 (Transparency Obligations)

### 施行スケジュール（2026-08-30確認）
- Regulation (EU) 2024/1689 は原則 **2026-08-02** から適用される（Article 113）。Article 50(1)–(4)の具体的な適用は、システムの役割・用途・出力に依存する。
- Regulation (EU) 2026/1744（Digital Omnibus on AI）は官報掲載後3日目の **2026-07-27** に発効（Article 4）。同規則が追加したArticle 111(4)は、2026-08-02より前に市場投入された合成音声等のシステムについて、Article 50(2)への必要な対応を **2026-12-02** までに行う旨を定める。これは新規システム全般の免除ではない。

### 対象範囲
- **Article 50(1)**: 人と直接対話するAIシステムのproviderは、AIとの対話であることを知らせる設計が必要（明白な場合等の例外あり）。
- **Article 50(2)**: 合成音声・画像・動画・テキストを生成するAIシステムのproviderは、技術的に可能な範囲で出力を機械可読にmarkし、人工生成・操作を検出可能にする必要がある。
- **Article 50(3)**: emotion recognition / biometric categorisationのdeployerに、対象者への通知と適用されるデータ保護法の遵守を求める。
- **Article 50(4)**: deepfakeの画像・音声・動画を生成・操作するAIシステムのdeployerに、人工生成・操作であることのdisclosureを求める（明らかに芸術的・創作的等の場合の限定的例外あり）。

### Vokra 実装要件

#### 1.1 Machine-readable marking（現行実装との境界）

**以下は法令適合済みという意味ではなく、候補技術と未実装事項の記録である。**

- **AudioSeal** の standalone 明示 embed/detect API は CPU/Metal で利用できるが、通常のTTS/VC経路への自動接続はない。

- **C2PA / SynthID 等**は現行Vokra runtimeの実装済み機能ではない。採用・ライセンス・検出性能は別途要法務/技術確認。

- SynthID、SilentCipher、WaveGuard等の名称は技術候補を示すものに過ぎず、現行Vokraが利用できること、ライセンスが許されること、採用予定があることを示さない。採用を検討する場合は、提供元の現行一次資料、利用条件、技術性能、配布条件を個別に確認する。

#### 1.2 Detectable 表示 (Vokra API + deployer 責任)

- `WatermarkConfig::backend_status()` は `Deferred` を返す。したがって、Vokraのdefault経路に常時提供されるmarking/disclosure cueや、法的十分性を保証する `watermark_enabled` の自動機構はない。

- **deployer 責任**: 該当する地域・出力・役割について、visible disclosure、必要なmarking、地域判定、user consent、適用範囲と検出可能性を実装・検証する。Vokraはこれを自動enforceしない。

#### 1.3 ドキュメント責任

- Vokra README / 利用者向け文書では、現行実装が法令適合を保証しないことと、deployerがuser-facing disclosureを実装する責任を明記する。
- `docs/legal-compliance.md` にこの文書を配置し、SDK ユーザーに周知

#### 1.4 Deployment-side disclosure（automatic watermark integration が Deferred の期間）

§1.1 のとおり `WatermarkConfig::backend_status()` は `WatermarkBackendStatus::Deferred` を返し、通常のTTS/VC runtimeは生成音声へ自動watermarkを埋め込まない。AudioSeal単体の明示APIも、それを呼ばない生成経路へ自動適用されず、法的十分性や変換後の検出可能性を保証しない。この期間、VokraがArticle 50やSB 942を満たすとは断定せず、該当するprovider/deployerが必要なvisible disclosure、marking、検出可能性を自分の経路で確認する。standalone AudioSealを使っても同意・権利・法的適合の代替にはならない。

- **a. EU対象のdeployer**: deepfake等に該当するかを確認し、該当時はclear and distinguishableなdisclosureを実装する。Vokra coreはUIや地域判定を提供しない。
- **b. EU対象のprovider**: Article 50(2)のmarkingが自分の役割に適用されるか、技術的手段と検出性能を検証する。Vokraのdefault経路に自動markingはない。
- **c. California SB 942対象のcovered provider**: §2のdetection tool、manifest/latent disclosure、保存・フィードバック要件を条文に照らして確認する。VokraのAPIを使えば自動的に適合するとはいえない。
- **d. ELVIS Act / NO FAKES Act**: disclosureだけではvoice rights、consent、配布責任を解消しない。§3/§4の要法務確認を行う。
- **e. 音声録音 / speaker embedding**: 同意、個人情報・生体情報、撤回、保存期間をdeployerが確認する。

**Owner 責任の再確認**: visible indicator の文言・地域強制・consent workflow・技術的markingの法的十分性は **deployer / 依頼者の判断**（FR-MD-13 / X-03）。`DisclosureConfig::require_visible_ui` は設計上の設定値であり、UI実装・法的レビュー・地域判定のenforcementや法令適合をVokraは提供・保証しない。

**関連参照**: §1.1（machine-readable marking、現状 Deferred）／ §1.2（deployer 責任）／ §1.3（README 記述）／ §8 実装状況（Deferred 事実・`WatermarkConfig::backend_status()` の復帰接続点）／ §11 deployer checklist（項目別）／ §2（SB 942）／ §3（ELVIS Act）／ §4（NO FAKES Act）／ [`docs/license-audit.md`](license-audit.md) §Article 50 checklist（運用側でdisclosure要否を確認する記述）／ `crates/vokra-core/src/compliance/level.rs`（`DisclosureConfig::require_visible_ui`）／ `crates/vokra-core/src/compliance/watermark.rs`（`WatermarkConfig::backend_status()` が `Deferred` を返す事実）。

### 罰則リスク
罰則の対象者・金額・各国実施法は違反類型と事実関係によるため、本書では断定しない。Article 50の適用とVokra provider/deployerの責任分界は要法務確認。

根拠: [EUR-Lex Regulation (EU) 2024/1689（CELEX: 32024R1689、Article 50 / 111 / 113）](https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX%3A32024R1689) および [Regulation (EU) 2026/1744（CELEX: 32026R1744、Article 4 / Article 111改正）](https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX%3A32026R1744)（いずれも公式官報、2026-08-30確認）。

---

## 2. California SB 942 (California AI Transparency Act)

### 施行・status（2026-08-30確認）
- **Chapter 291として2024-09-19に成立**し、§22757.6により **2026-01-01からoperative**。審議中法案ではない。

### 対象範囲
- California内で公衆利用可能で、covered providerが作成・コード化・生産したGenAI systemが **月間1,000,000人を超える visitors または users** を有する場合（§22757.1(b)）。「月間100万Californiaユーザー」と固定しない。専らnon-user-generatedのvideo game等には§22757.5の除外がある。
- covered providerには、無料のAI detection tool、ユーザーが選べるclear/conspicuousなmanifest disclosure、技術的に可能で合理的なlatent disclosure等の要件がある。法文上manifestは人が容易に知覚・理解できるものを指し、AudioSeal/C2PAを使えば自動適合するという意味ではない。

### Vokra 対応
- Vokra自体または組込みサービスがcovered providerに該当するかは事業形態・月間visitor/user数・公開範囲で判断する。現行Vokraのautomatic watermarkはDeferred、standalone AudioSealは明示APIのみであり、SB 942適合を断定しない。
- 対象provider/deployerは detection tool、manifest/latent disclosure、個人provenance dataの扱い、フィードバック、第三者ライセンス条件を条文に照らして実装・確認する。具体的適用は要法務確認。

根拠: [California SB 942 Chapter 291（章法・条文）](https://leginfo.legislature.ca.gov/faces/billVersionsCompareClient.xhtml?bill_id=202320240SB942)（2026-08-30確認）。

---

## 3. Tennessee ELVIS Act (Ensuring Likeness Voice and Image Security Act)

### 施行・status（2026-08-30確認）
- Tennessee **Public Chapter 588** として2024-03-26に成立し、公式Bill Informationの記載どおり **2024-07-01発効**。現行法であり審議中法案ではない。

### 対象範囲
- 個人（生存・死亡を含む）の識別可能なvoice等をproperty rightとして保護し、無権限利用・公衆への提供等について民事救済を定める。条文上の例外やFirst Amendmentとの関係を含む具体的適用は要法務確認。
- 無権限で特定個人のvoice/likeness等を生成することを主目的・主機能とするalgorithm/software/tool等を、無権限と知りながら提供する者にも規定が及び得る。Vokraが該当しないとは断定しない。

### Vokra 対策 (最重要)

#### 3.1 現行実装との境界
- Vokra coreの機能分離やモデルの意図だけで、ELVIS Act上の適用除外・safe harbor・法令適合が成立するとは断定しない。
- deployerは識別可能な個人のvoiceを扱う場合、権利、同意、契約、用途、配布先、例外を個別確認する。standalone AudioSealは同意や権利の代替ではない。

#### 3.2 Speaker embedding / consent
- `speaker_encode` op は core に含める (現代 zero-shot TTS の必須入力)
- speaker embeddingを扱う実際の経路、個人情報・生体情報該当性、同意取得・撤回・保管をdeployerが確認する。
- 現行runtimeの型や明示APIが署名済み同意の法的有効性を検証するものではない。manifest、署名、UI、記録保持を実装済みと断定しない。

### 罰則リスク
民事請求、差止めその他の救済があり得るが、請求主体、要件、救済範囲は条文と事実関係による。Vokra providerに責任がない、またはあると断定しない。

根拠: [Tennessee SB 2096 / Public Chapter 588 Bill Information](https://wapp.capitol.tn.gov/apps/BillInfo/Default?BillNumber=SB2096&GA=113)（2026-08-30確認）。

---

## 4. 連邦 NO FAKES Act (Nurture Originals, Foster Art, Keep Entertainment Safe Act)

### 進捗・status（2026-08-30確認）
- S.1367（119th Congress、2025-04-09提出）は Senate Judiciary Committee に付託された **introduced / referred の法案**であり、成立法ではない。
- 施行日、成立見込み、2027年施行という予測は確定していない。将来法案の内容を現行Vokraの義務として扱わない。

### 対象範囲
- 提出法案はvoice/visual likenessのdigital replica等を対象にする案だが、法案本文の定義・例外・救済を含め、成立前の提案内容である。

### Vokra 対応
- 現時点でNO FAKES Actに基づくVokraの停止義務や適合を断定しない。成立時は法文、施行日、既存州法との関係を再評価する。

根拠: [S.1367 introduced text（Congress.gov / GPO）](https://www.congress.gov/119/bills/s1367/BILLS-119s1367is.pdf)（2026-08-30確認）。

---

## 5. Apple App Review Guidelines（Guideline 5.5の訂正）

### 対象範囲・status（2026-08-30確認）
- 現行の [App Review Guidelines](https://developer.apple.com/app-store/review/guidelines/) の **5.5はMobile Device Management (MDM)** に関する規定であり、生成AIアプリの一律な「AI-generated content metadata declaration」規定ではない。
- 生成AI固有の一律施行日や、Vokra SDKに対する `NSAIGeneratedContent` Info.plistキーの要件は、確認したApple一次資料からは確認できない。旧記述は撤回する。
- UGCを扱う場合はGuideline 1.2（フィルタ、報告、ブロック、連絡先等）が関係し得るが、アプリの機能・配信形態ごとに要確認。

### Vokra 対応
- Apple提出物、アプリ内表示、UGC moderation、プライバシー開示等はdeployerが現行規約と個別審査で確認する。VokraはApple審査通過や規約適合を保証しない。

根拠: [Apple App Review Guidelines](https://developer.apple.com/app-store/review/guidelines/)（2026-08-30確認、ページ最終更新2026-06-08）。

## 6. Google Play AI-Generated Content policy

### 対象範囲・status（2026-08-30確認）
- Googleの公式ポリシーは、AI生成アプリについて禁止コンテンツ等を生成しないこと、既存のDeveloper Policiesを守ること、生成AIアプリにアプリ内のuser reporting/flagging機能を置くことを求める。
- 適用例には、AIで実在人物のvoice/video recordingを作成するアプリが含まれる。一方、単にAIコンテンツをホストするだけのアプリ等は当該ポリシーの対象外と説明されている場合がある（UGC policy等は別途適用され得る）。
- 公式ページで「2024-06施行」という独立した施行日を確認できないため、旧記述を撤回する。現行掲載ポリシーと審査時点のDeveloper Policiesを確認する。

### Vokra 対応
- Vokra APIにGoogle規約を満たす報告機構が実装済みとは断定しない。Androidアプリの報告導線、moderation、guardrails、Play Console申告はdeployerが実装・確認する。

根拠: [Google Play AI-generated content policy](https://support.google.com/googleplay/android-developer/answer/13985936) および [Understanding Google Play's AI-Generated Content policy](https://support.google.com/googleplay/android-developer/answer/14094294)（2026-08-30確認）。

---

## 7. 日本国内法対応

### 個人情報保護法 (2022 改正)
- 音声や speaker embedding が個人情報・個人識別符号・要配慮個人情報に該当するかは、データの内容、他情報との照合可能性、処理目的などに依存する。speaker embedding が常に個人識別符号になるとは扱わない。
- SDKユーザーは、適用される個人情報保護法、利用目的、取得・第三者提供・安全管理等を個別に確認する。匿名化・仮名化・同意は、法令上の適用される根拠や処理設計に応じて選択するもので、常に同意だけが唯一の根拠とは限らない。
- `speaker_embedding_anonymize(embedding, k)` は現行APIではなく、匿名化の法的効果も保証しない。必要性・方式は法務／プライバシー設計で判断する。

### 声優・タレント肖像権
- 音声、氏名、肖像、キャラクター、モデル重みの利用可否は、権利者、契約、利用目的、公開範囲、各サービス規約等に依存する。特定の声質利用に「契約必須」などの一律結論を本書で示さない。
- VOICEVOX、A.I.VOICE、CoeFont、VOICEROID、UTAU等を利用する場合は、対象モデルとキャラクターごとの現行利用規約を一次資料で確認し、必要な許諾・表示・配布条件を記録する。Vokraはこれらの許諾を提供・保証しない。
- 日本法の適用・判例の評価は、具体的な利用形態を踏まえて法務確認する。

根拠: [個人情報の保護に関する法律（e-Gov法令検索）](https://laws.e-gov.go.jp/law/415AC0000000057)（2026-08-30確認）。

---

## 8. Vokra API の compliance 設計スケッチ（現行適合の保証ではない）

この節の設定例は設計スケッチ・型の記録であり、Vokraが各法令を満たす自動
compliance modeを提供するという意味ではない。特に watermark、C2PA、地域判定、
UI disclosure、同意の法的十分性は未検証であり、deployerの責任で確認する。

### `Vokra::init(config)` の compliance 設定

```rust
Vokra::init(VokraConfig {
    compliance: ComplianceLevel::Strict, // 設定例。法令適合を意味しない
    watermark: WatermarkConfig {
        audioseal: true,          // default ON
        c2pa: true,               // default ON
        synthid: false,           // 採用・利用条件は別途確認
        silent_cipher: true,      // 設計スケッチ上の候補。採用・実装を表明しない
    },
    voice_cloning: VoiceCloningPolicy::Disabled, // core では常に Disabled
    speaker_embedding: SpeakerEmbeddingPolicy::RequireConsent, // policy例。法的要件は条件依存
    disclosure: DisclosureConfig {
        default_beacon_frequency_hz: 22050, // 人耳外の高周波 beacon
        require_visible_ui: true,
    },
})
```

### Compliance level

- **Strict / Standard / Research / Disabled** は設定上のレベルであり、選択しただけで法令適合・同意・marking・UI表示が成立するものではない。
- watermark、voice cloning、speaker embedding、地域対応の実際の挙動と適用範囲は、下記の実装状況とdeployerの検証結果を優先する。

### 自動地域判定

- 地域判定はVokraが法的に確定するものではない。system locale等のヒントを使う場合でも、EU/CA/TN/JPの適用判断とdeployer overrideの可否は要法務確認。

### 実装状況（2026-08-30確認）

上記スケッチの **compliance 設定 API を `crates/vokra-core/src/compliance/` の型として実装**した（FR-CP-06）。実装と本スケッチの対応・乖離:

- **`ComplianceLevel`（Strict/Standard/Research/Disabled）・`WatermarkConfig`・`VoiceCloningPolicy`・`SpeakerEmbeddingPolicy`・`DisclosureConfig`（beacon 22050Hz）を型として提供**（default = Strict、voice_cloning は core で常時 `Disabled`＝単一 variant で表現不能化、speaker_embedding = `RequireConsent`）。init 統合点は `Vokra::init` グローバルではなく、当面 **model ローダーへ明示的な `CompliancePolicy` を渡す**形で配線（SRS の Session 中心 API と整合、グローバル init は据え置き）。
- **research flag の実挙動**: `ComplianceLevel::Research`（または `with_research_license(true)` / `VOKRA_ALLOW_RESEARCH_LICENSE=1`）が CC-BY-NC 系 weight（F5-TTS/Fish-Speech/EnCodec）を解錠する。Strict/Standard は解錠せず `VokraError::ResearchLicenseRequired` で拒否（fail-closed、`docs/license-audit.md` §3 参照）。
- **watermark の自動 policy 接続はDeferred**: `WatermarkConfig` は default ONという設計意図（audioseal/c2pa=true・synthid=false・silent_cipher=true）とopt-out経路を保持するが、AudioSeal単体の明示embed/detect以外は実装済みと扱わない。`WatermarkConfig`から通常の生成経路へ自動適用する接続とC2PAは未実装で、`WatermarkConfig::backend_status()`は`Deferred`を返す。**通常の生成音声を「埋め込み済み」と表示しない**。したがってEU AI Act Article 50 / SB 942のmarking義務を自動充足するとはいえない。法務的十分性はFR-MD-13 / X-03（依頼者）に従属する。
- **自動地域判定は locale ベース最小版のみ・IP geolocation は据え置き**: zero-dep 不変条件（NFR-DS-02）維持のため geoip 系 crate/DB を core に追加しない。locale ヒントによる Strict 強制/警告は後続の最小実装に委ね、実際の地域確定は deployer 責務（本節の EU 強制は deployer が最も安全側に倒す前提）。
- **Qwen3-TTS-Tokenizer-12Hz decoder（2026-08-27）**: このcompanionのGGUF bindはwatermark埋込済み、同意取得済み、または法令適合済みを意味しない。speaker conditioningを使う場合の権利・同意・disclosureはdeployerが確認する。`WatermarkConfig::backend_status()==Deferred`の現状も変えない。

---

## 9. Copyright / Training Data Provenance リスク登録（2026-07時点の記録）

以下は2026-07に記録した継続確認事項であり、現行法の結論、裁判所の判断、または
Vokraのmodel-zoo適格性を示すものではない。法令・訴訟のstatusは変動するため、採用・
配布の都度、公式一次資料と個別の法務確認を行う。

- **米国著作権訴訟**: [Kadrey v. Meta（米国北部カリフォルニア地区裁判所の事件情報）](https://cand.uscourts.gov/cases-e-filing/cases/323-cv-03417-vc/kadrey-et-al-v-meta-platforms-inc) などの事件は、係属状況・争点・判断が更新され得る。本文書はfair use、責任、学習データの適法性について結論を示さない。
- **Section 230**: [47 U.S.C. §230（米国法典）](https://uscode.house.gov/view.xhtml?edition=prelim&num=0&req=granuleid%3AUSC-prelim-title47-section230) は条文上の制度であり、AI生成音声への適用可否や個別責任を本書だけで判断できない。2025年の判例動向を一般化しない。
- **Training data provenance**: 学習データ、重み、コード、モデルカード、利用規約、同意・撤回条件はモデルごとに異なる。公式model-zooへの収録可否・license tier・配布可否は、個別モデルの一次資料と [`docs/license-audit.md`](license-audit.md) のsign-offを根拠にownerが判断する。本節の一覧や「training data sourceが公開されている」ことだけで適格性を決めない。

Vokraは一般目的の推論ランタイムであり、利用者がロードするモデルのライセンス、学習データの権利、用途の適法性を自動判定・保証しない。これは法的免責やsafe harborの主張ではなく、責任分界の記録である。

---

## 10. サーバサイド SaaS デプロイ時の追加要件

### GDPR (EU)
- 音声・embeddingがpersonal dataに該当するか、またArticle 9のbiometric data（自然人を一意に識別する目的で処理するもの）に該当するかは、データと処理目的による。常にArticle 9該当、または常に明示的同意が唯一の根拠とは扱わない。
- controller / processorの関係、処理目的、データ移転、安全管理等に応じて、DPAその他の契約・措置の要否を個別確認する。VokraはDPAを提供・締結したことを意味しない。

### HIPAA (米医療)
- HIPAAはcovered entityやbusiness associate等、対象主体とPHIの取扱いに応じて適用される。CPU/Vulkanを選ぶだけでHIPAA適合になるとはいえず、契約、アクセス制御、監査、セキュリティ等を別途評価する。
- NVIDIA CUDA EULAに医療用途の一律禁止があるとは本書で断定しない。利用・再配布形態に照らし、現行EULAを確認する。
- `Vokra-critical-safe` は設計上の目標名であり、現行出荷物またはHIPAA認証済みSKUではない。

### PCI DSS (決済)
- 音声・通話録音を扱うことだけでPCI DSSの適用や準拠が決まるわけではない。カード会員データ環境との接続、保存・処理・伝送、事業者の役割と適用版に応じて、deployerが対象範囲と評価を確認する。
- Vokra providerの責任分界や必要な契約は、実際の構成と法務・セキュリティ評価に依存する。

根拠: [EUR-Lex GDPR（Regulation (EU) 2016/679、Article 9 / 83）](https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX%3A32016R0679)、[HHS: Covered Entities and Business Associates](https://www.hhs.gov/hipaa/for-professionals/covered-entities/index.html)、[HHS: Business Associates](https://www.hhs.gov/hipaa/for-professionals/privacy/guidance/business-associates/index.html)、[NVIDIA CUDA EULA](https://docs.nvidia.com/cuda/pdf/EULA.pdf)（2026-08-30確認）。

---

## 11. Vokra 提供の Compliance Checklist (deployer 向け)

新規 Vokra 統合プロジェクト開始時に確認すべき checklist:

```
□ AudioSealはstandalone明示APIとして使うか確認（自動watermark・法令適合を意味しない）
□ C2PA等のmarkingは別途実装・検出性能・ライセンスを確認
□ SB 942 (CA): covered provider、月間1,000,000超のvisitor/user、§22757.5除外、detection/manifest/latent要件を確認
□ EU AI Act: provider/deployerの役割、Article 50(1)-(4)、原則2 August 2026、既存対象へのArticle 111(4)の2 December 2026期限を公式条文で確認
□ ELVIS Act (TN): voice rights、同意、用途、配布、例外を要法務確認
□ NO FAKES Act: S.1367は未成立。成立・施行時に再評価
□ Apple App Review Guidelines 5.5はMDM。生成AI表示やUGC要件は該当ガイドラインを個別確認
□ Google Play AI-generated content: in-app reporting、guardrails、禁止コンテンツ、Play Console申告を確認
□ 日本個情法: 音声・embeddingの該当性、目的、取得・提供・安全管理の個別確認
□ 日本声優: 対象モデル／キャラクターごとの現行ライセンス・許諾条件確認
□ GDPR (EU): 音声・embeddingの個人データ／Article 9該当性、法的根拠、processor関係の有無を個別確認
□ HIPAA (医療): covered entity / business associate該当性、PHIの流れ、契約・安全管理を個別確認（CPU/Vulkanだけで適合とはしない）
□ NVIDIA CUDA EULA: 利用形態・配布形態に照らして現行EULAを確認（医療用途可否を本書で断定しない）
□ Training data provenance: 個別モデルの一次資料・license-auditのsign-off・配布条件を確認
□ Consent manifest: 適用法・契約・owner policyで必要な場合の構造／署名／記録要件を個別確認（実装だけで法的有効性を主張しない）
```

### 11.1 M5-05 通過記録（voice cloning experimental / FR-MD-13 / X-04(c)、2026-07-21）

M5-05（`vokra-voiceclone-experimental` 分離準備）は FR-MD-11（RVC v2 / GPT-SoVITS）が実際に配布される WP であり、FR-MD-13（新規モデル対応 PR ごとの license-audit 追記 + 本 §11 checklist 通過）の到来点である。CC 実装分（T05-T09, T13）が本 checklist の各項目をどこまで満たすかを**詐称せず**記録する（判定・sign-off は owner = T15）:

- **□ Consent manifest（`:311`、voice cloning experimental の owner policy / implementation check）— 部分充足（構造検証まで。法的要件の一律判断ではない）**:
  - **実装確認**: consent manifest の schema（`ConsentManifest` / `ConsentScope`、consent manifest の 5 field を転記）と **構造検証**（`ConsentManifest::parse` = field presence / scope enum 妥当性 / `vokra_session_id`・`grant_date` 非空 / `signature` field の存在、fail-closed reject）を `crates/vokra-core/src/compliance/consent.rs`（M5-05-T05/T06）に実装。`SpeakerEmbeddingPolicy::RequireConsent` を consent 型に接続し、未署名（signature 非存在＝空）manifest を API level で reject（§3.2、M5-05-T07）。別リポ scaffold binary が両 flag（`--i-understand-risks --research-only`）+ 署名付き consent を start-up gate で強制（M5-05-T08/T09）。これは構造・起動時チェックの事実であり、署名の法的有効性や適用法の充足を意味しない。
  - **満たさない（owner 待ち）**: **cryptographic signature 検証**は core 非対応（`SignatureStatus` は `Present`/`Absent` の構造判定のみ、`Verified` variant を作らない）。理由は zero-dep 制約ではなく (1) PGP/Ed25519 署名検証は security-critical で自前実装が不適切、(2) 信頼根（誰の鍵・配布・失効）が owner 決定であるため。署名検証方式の確定は M5-05-T04（owner）。
- **□ AudioSeal watermark ON / □ C2PA manifest 埋込（`:297`/`:298`）— 未充足（automatic integration Deferred）**: §1.4 / §8 のとおり AudioSeal 単体の明示 embed/detect は利用できるが、`WatermarkConfig::backend_status()==Deferred` であり voice cloning binary / 通常生成経路への強制接続はない。したがって「強制埋込」leg は honest-UNMET（M5-05-T09 の leg 3、`docs/adr/M5-05-watermark-dependency.md` の owner resolution 待ち）。該当する法令・役割・用途でvisible disclosureが必要か、その形式と実装責任をdeployer／ownerが確認する。standalone embedを使っても法的十分性は自動的に成立しない。
- **□ ELVIS Act / □ NO FAKES Act（`:301`/`:302`）— 実装の存在だけでは法務充足にならず、法務判断は owner**: voice cloningの構成、配布、同意、用途、tool-distributor liabilityの該当性をowner/deployerが確認する。§3.1の機能分離や警告があってもsafe harbor・適合を断定しない（NFR-LG-02、M5-05-T04）。
- **□ Training data provenance（`:310`）— 個別確認待ち**: RVC v2 / GPT-SoVITS等の学習データ・重み・配布条件は、各モデルの一次資料を確認して scaffold `NOTICE` と `docs/license-audit.md` §3.1（空欄 sign-off = fail-closed）に記録する。本項目は現時点の適法性やzoo適格性を断定しない。

## 12. 罰則 / 救済（要法務確認）

| 制度 | 現行文書で確認できる範囲 |
|-----|-------|
| EU AI Act Article 50 | 違反類型・事業者の役割・加盟国実施法に依存。金額を本書で断定しない。 |
| California SB 942 | Chapter 291 §22757.4にcivil penalty等の規定。covered provider該当性と救済は要確認。 |
| ELVIS Act (TN) | 民事請求・差止め等の条文上の救済。具体的要件は事実関係に依存。 |
| NO FAKES Act (連邦) | S.1367は未成立の法案であり、現行の罰則・施行日はない。 |
| Apple App Review Guidelines | Appleの審査・掲載判断はアプリと提出内容に依存。Guideline 5.5はMDM。 |
| Google Play AI policy | Playの掲載・アカウント措置は現行Developer Policiesと審査に依存。 |
| GDPR | Article 83等の上限・要件は違反類型、事業者、加盟国手続等に依存。固定額を本書で断定しない。 |
| HIPAA | 民事・行政・刑事上の措置、上限、適用主体は規則・違反類型・改正に依存。固定額を本書で断定しない。 |
| 日本個情法 | 違反類型、主体、改正時点等により制裁・救済が異なる。固定額を本書で断定しない。 |

---

## 13. 参考出典

- [EUR-Lex Regulation (EU) 2024/1689（CELEX: 32024R1689、Article 50 / 111 / 113）](https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX%3A32024R1689)
- [EUR-Lex Regulation (EU) 2026/1744（CELEX: 32026R1744、Article 4 / Article 111改正）](https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX%3A32026R1744)
- [California SB 942, Chapter 291 (official bill text)](https://leginfo.legislature.ca.gov/faces/billVersionsCompareClient.xhtml?bill_id=202320240SB942)
- [Tennessee SB 2096 / Public Chapter 588 (official Bill Information)](https://wapp.capitol.tn.gov/apps/BillInfo/Default?BillNumber=SB2096&GA=113)
- [S.1367 NO FAKES Act of 2025 (Congress.gov / GPO introduced text)](https://www.congress.gov/119/bills/s1367/BILLS-119s1367is.pdf)
- [Meta AudioSeal](https://github.com/facebookresearch/audioseal)
- [c2pa-rs (Adobe)](https://github.com/contentauth/c2pa-rs)
- [Google Play AI-generated content policy](https://support.google.com/googleplay/android-developer/answer/13985936)
- [Google Play AI-generated content policy overview](https://support.google.com/googleplay/android-developer/answer/14094294)
- [Apple App Review Guidelines](https://developer.apple.com/app-store/review/guidelines/)
- [NVIDIA CUDA EULA](https://docs.nvidia.com/cuda/eula/index.html)
- [個人情報の保護に関する法律（e-Gov法令検索）](https://laws.e-gov.go.jp/law/415AC0000000057)
- [HHS: Covered Entities and Business Associates](https://www.hhs.gov/hipaa/for-professionals/covered-entities/index.html)
- [HHS: Business Associates](https://www.hhs.gov/hipaa/for-professionals/privacy/guidance/business-associates/index.html)
- [Kadrey v. Meta case — U.S. District Court, Northern District of California](https://cand.uscourts.gov/cases-e-filing/cases/323-cv-03417-vc/kadrey-et-al-v-meta-platforms-inc)
- [47 U.S.C. §230（米国法典）](https://uscode.house.gov/view.xhtml?edition=prelim&num=0&req=granuleid%3AUSC-prelim-title47-section230)
- [VOICEVOX 利用規約](https://voicevox.hiroshiba.jp/term/)
