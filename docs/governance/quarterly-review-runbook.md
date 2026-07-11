# 四半期 Go/No-go review 運用手順 runbook

**文書 ID**: VOKRA-GOV-002
**最終更新**: 2026-07-11（初版、M2-15-T02）
**位置付け**: CLAUDE.md の Kill switch 表（撤退条件 A〜L、single source of truth）と
system-requirements.md **NFR-MT-05**（Kill switch 四半期 Go/No-go review をリリース
プロセスに組み込む）の**手動 review 運用手順**。判定そのものは依頼者（`ayutaz`）が
行う。本 runbook は「いつ・何を入力に・どこに出力し・どのように release process と
連動させるか」を固定する。

**兄弟 runbook**: [kill-switch-metrics-runbook.md](kill-switch-metrics-runbook.md)
（VOKRA-GOV-001）— review 実施時に依頼者が実行する **指標データ収集手順**
（GitHub star / contributor / Issues+Discussions active participants）を定義。
本 runbook はその出力を **入力の 1 つ** として消費する上位運用手順。

---

## 0. 前提と根拠

### 0.1 SSOT

- **Kill switch 表**: CLAUDE.md「Kill switch（撤退条件、四半期評価に格上げ）」節が
  single source of truth。撤退条件 A〜L の閾値・時期・監視頻度は本 runbook から
  独自に導出せず、CLAUDE.md の表を出典として転記する。
- **要件 ID**: NFR-MT-05（本 runbook の根拠）、NFR-MT-03（4 週間 release train と
  連動、`docs/system-requirements.md` §3.5）。
- **WP レイヤ**: M2-15（`docs/tickets/m2/M2-15-quarterly-review.md`）が v0.5 期の
  四半期 review の実体 WP。M3 以降も同型で M3-19 等が四半期 review イベントを
  担う（`docs/milestones.md` §7）。

### 0.2 廃止された自動監視（重要）

**FR-TL-05（`.github/workflows/kill-switch-check.yml`）は 2026-07-04 依頼者決定で
廃止済**（`docs/system-requirements.md` §2.11 の FR-TL-05 セル、`docs/requirements.md`
§監視、`docs/milestones.md` 2026-07-04 決定行を参照）。

- 旧 FR-TL-05 = 毎月競合 changelog（Unity IE / ORT / HF+ggml / Modular MAX /
  Candle / Kyutai）を自動 workflow でチェック → Discord 通知。
- 廃止理由: (1) community Discord 全体を非採用（2026-07-04）した結果、通知先が
  消失、(2) 判定は依頼者による手動 Go/No-go であり、changelog の機械的な
  「差分あり」検知は判定に直結しない、(3) NFR-MT-05 の四半期 review に統合すれば
  自動監視の冗長性が消える。
- **読み替え**: 「毎月自動チェック + Discord 通知」は **本 runbook の四半期手動
  review に全面読替**。競合 changelog 確認は §3.2（T04 手動収集）で行う。

### 0.3 自動監視 `.yml` を新設・再稼働しない（禁止事項）

本 runbook は **`.github/workflows/*.yml` として自動化しない**。以下を禁止する:

- 旧 `kill-switch-check.yml` を復活させる新規 PR。
- 競合 changelog 監視を任意名の別 workflow（例: `competitor-scan.yml`）で
  自動化する新規 PR。
- 本 runbook の §3 入力収集を GitHub Actions の `schedule:` トリガで自動実行
  させる変更（`schedule:` 自体は他の nightly parity / CD には使ってよいが、
  Kill switch 監視には使わない — 2026-07-04 決定の遵守）。

例外は無い。追加が必要になった場合は、まず本 runbook と `docs/tickets/m2/M2-15-quarterly-review.md`
「改訂記録 (b)」の廃止決定を再解釈できる依頼者判断を得てから行う。

---

## 1. 実施 cadence

**四半期ごとに 1 回**（3 月末 / 6 月末 / 9 月末 / 12 月末を目安）。

| trigger | 実施頻度 | 説明 |
|---------|---------|------|
| **定期** | 四半期 1 回 | 3/6/9/12 月末を目安。前四半期の release train 3 本
（NFR-MT-03、4 週 × 3 = 12 週）の直後を review 実施点とする |
| **緊急即時**（Kill switch E） | 検知次第 | HF + ggml が音声特化 op を llama.cpp に
追加した瞬間に緊急発動（CLAUDE.md「常時緊急監視」）。定期四半期を待たず、
検知後 1 週間以内に review を起動する |
| **マイルストーン連動** | Kill switch C/J/K/D | C = v0.1 MVP 公開後 3 ヶ月 =
5-6 ヶ月時点 / J・K = v0.5 時点 / D = v0.5 公開後 3 ヶ月 = 8-10 ヶ月時点
（CLAUDE.md、`docs/milestones.md` §7 の Kill switch 評価カレンダー）。
これらは判定日到来した四半期 review に合流させ、独立 review を起こさない |

**カレンダー登録は依頼者責任**（本 runbook は自動 CI に載せない = 2026-07-04
依頼者決定の遵守）。v0.1 MVP release tag `v0.1.0` を打った日、v0.5 release
tag `v0.5.0` を打った日を起点にカレンダー登録する運用を推奨（兄弟 runbook §5
と同じ扱い）。

---

## 2. v0.5 期の実施時点（Kill switch C / J / K / D）

`docs/milestones.md` §6「Kill switch 評価」表と `docs/tickets/m2/M2-15-quarterly-review.md`
「位置付け・スコープ境界」表からの転記。原文の閾値は CLAUDE.md（SSOT）に従う。

| Kill switch | 判定時期（暦月は目安） | 本 runbook での扱い |
|------|----------|------|
| **C** | v0.1 MVP 公開後 3 ヶ月 = 5-6 ヶ月時点 = 2026-12〜2027-01 頃（目安）。
v0.5 期間内に到来 | C 判定日 = 到来した四半期 review に合流。§3.1 で兄弟 runbook
の指標 JSON を消費し、star ≥ 500 かつ engagement proxy ≥ 20 を PASS 側とする |
| **J** | v0.5 時点 | M2-09（FR-SV-05 Wyoming Protocol サーバ実装）が判定の前提
（`docs/milestones.md` §6 M2-15 依存列）。§3.4 で HA Voice / Wyoming エコシステム
側の採用シグナルを M2-15-T08 材料として review へ投入 |
| **K** | v0.5 時点 | Exit criteria の補足ゲートと表裏
（`docs/milestones.md` §6 v0.5 補足ゲート「addressable market が競合の 10% 以上」）。
§3.1 で自社指標を、§3.5 で競合 baseline 候補を投入 |
| **D**（監視開始） | 判定は 8-10 ヶ月時点 = 2027-03〜2027-05 頃（目安）、
**M3 期に実施** | v0.5 期の review では **計測体制の確立と監視開始のみ**を行う
（`docs/tickets/m2/M2-15-quarterly-review.md` T12）。D の該当/非該当は v0.5 review
では結論を出さない — 起点日を M3 期 review へ引き継ぐ |

**Discord サブ閾値の読み替え**（2026-07-06 決定、`docs/tickets/m2/M2-15-quarterly-review.md`
「改訂記録 (a)」）: C の「Discord active user < 20」・K の「Discord DAU」は
Discord 全体非採用により実測不能。GitHub Issues + Discussions の直近 3 ヶ月
active participant 数を代替 proxy とする（兄弟 runbook §3 で計測手順を定義済）。
**proxy と閾値の対応確定は依頼者判断**であり、本 runbook では固定しない
（M2-15-T06/T07/T10/T11 の依頼者判定に委ねる）。

---

## 3. Inputs（review 実施時に用意する入力）

各 review 実施ごとに以下 4 種類の入力を揃える。CC は入力の**収集手順の実行**と
**事実の要約**を支援するが、**判断（撤退可否・proxy 対応・競合分母の確定）は
依頼者に帰属**する。

### 3.1 T03 指標 JSON（兄弟 runbook 出力）

- **収集ツール**: `scripts/kill-switch-metrics.sh`（推奨格納先、兄弟 runbook
  [kill-switch-metrics-runbook.md](kill-switch-metrics-runbook.md) §4 に定義。
  2026-07-11 時点では runbook 内スクリプトブロックとして提供、依頼者が手元で
  `chmod +x` して恒久配置する運用）。
- **入力先**: 上記スクリプトの stdout を四半期 review 記録ディレクトリに
  リダイレクト保存する:

  ```bash
  bash scripts/kill-switch-metrics.sh \
    > docs/governance/quarterly-reviews/YYYY-QN.metrics.json
  ```

  パス命名規約は兄弟 runbook §6「意思決定の記録先」に準拠。
- **含まれる指標**: (1) GitHub star 数、(2) 非 bot・非 CC contributor 数、
  (3) Issues + PR + Discussions 直近 3 ヶ月 active participants 数、
  (4) Kill switch C の暫定 verdict（PASS/FAIL）、(5) Kill switch K の
  verdict_input（自社値のみ、競合値は依頼者手動記入）。
- **使い所**: C 判定（§2）と K 自社側指標（§2）、D 監視開始（`docs/tickets/m2/M2-15-quarterly-review.md`
  T12）の共通入力。

### 3.2 T04 四半期競合スキャン（Kill switch A / B / H）

`docs/tickets/m2/M2-15-quarterly-review.md` T04 に定義された**手動収集**の要約。
CC は事実収集を支援し、該当有無の最終判断は依頼者に委ねる。

| Kill switch | 監視対象 | 情報源（手動確認） |
|------|--------|-------------|
| **A** | Unity Inference Engine（旧 Sentis）の VITS/Metal 修正 + 主要 TTS/ASR
（Whisper / piper-plus / Kokoro / CosyVoice）の net-native 対応 | Unity Inference
Engine の公式 changelog / release notes（バージョン・日付・対応 op を出典リンク付きで
記録） |
| **B** | sherpa-onnx が Unity UPM + Godot AssetLib の**公式一級バインディング**を
提供したか | sherpa-onnx リポジトリの Releases / README（"Unity" / "Godot" の
official 記述を確認） |
| **H** | Modular MAX Engine の音声特化 op 拡充状況（2024-08 Apache 2.0 OSS 化済） | Modular
公式 changelog / MAX blog（音声モデル対応のバージョン別追加を確認） |

`.yml` を使わず**手動確認**であること、記憶からの断定を避け出典リンクを付すこと
（CLAUDE.md ハルシネーション厳禁）を厳守する。

### 3.3 T05 常時・緊急監視（Kill switch E / F / L）

`docs/tickets/m2/M2-15-quarterly-review.md` T05 に定義。

| Kill switch | 監視対象 | 情報源（手動確認） | 特記 |
|------|--------|-------------|------|
| **E**（緊急） | HF + ggml が音声特化 op を **llama.cpp に追加した瞬間** | `ggml-org/llama.cpp`
の PR / releases（音声関連 op 追加の PR タイトル・merge 日を確認） | **緊急即時発動**。
検知した場合は §1 の「緊急即時」トリガで四半期を待たず review を起こす |
| **F** | HF Candle が Whisper / piper-plus / Kokoro / CosyVoice / Moshi / Mimi を
全面 native 対応（Whisper / Moshi / Mimi は既に対応済み） | `huggingface/candle` の
`candle-examples/` / `candle-transformers/` 追加状況を確認 | 常時監視、四半期 review
で状況要約 |
| **L** | 依頼者の burnout / 資金枯渇 | 依頼者の自己申告 | 本 runbook には記入欄
のみを提示、内容は依頼者が総合判定（T14）時に記入 |

### 3.4 T08 J 判定用材料（Wyoming / HA）

`docs/tickets/m2/M2-15-quarterly-review.md` T08 に定義。**J 判定時期のみ**（v0.5
四半期 review で実施）で用意する。

- M2-09 の FR-SV-05 Wyoming Protocol サーバ実装の readiness 現況（2026-07-07
  時点で **info reply の hard-assert が unit-level で完了、120+16 tests green**、
  `docs/tickets/m2/M2-15-quarterly-review.md` T08 Status セクション）。
- HA Voice / Wyoming エコシステム側の採用シグナル（Home Assistant / Wyoming の
  公式ドキュメント・issue・推奨サーバ一覧等の外形的状況を出典付きで記録）。
- 記憶から HA 採用可否を断定しない（CLAUDE.md ハルシネーション厳禁）。

### 3.5 M2 Exit criteria 実績

`docs/milestones.md` §6「Exit criteria」からの実績値転記。v0.5 四半期 review
（M2-15）では以下 4 項目 + 補足ゲートの実績を必ず review 記録に投入する:

1. **iOS で Whisper base RTF < 0.5**（NFR-PF-03、実機計測 = M2-14）— 実測値と
   計測日、達成/未達を記録。
2. **CUDA で Whisper large-v3 RTF < 0.1**（NFR-PF-04、実機計測 = M2-14）— 実測値、
   計測環境（vast.ai RTX 4090 の baseline 0.1133 は 2026-07-07 時点、formal
   always-on gate は M2-14 self-hosted runner + M3-01 5% regression gate に委譲済）。
3. **サーバ稼働の実証形態**（M2-10 descope 後の代替、`docs/milestones.md` §6
   Exit criteria 3）— コールセンター / 企業 API / HA Voice 等のいずれで
   実証しているかを記録（形態確定は M2-15 review の依頼者判断）。
4. **コミッター指標の計測体制確立**（`docs/milestones.md` §6 Exit criteria 4）—
   v0.5 では D の判定ではなく監視開始のみ（§2 参照）。

補足ゲート:
- サーバ TTS レイテンシ **90ms**（NFR-PF-05 v0.5 目標値）
- vokra-server 互換 API 4 種（FR-SV-02〜05）の CI 通過状況
- addressable market 指標（Unity Asset Store DL / GitHub star / GitHub Issues+
  Discussions active participants）が競合の 10% 以上か

**M3 以降の review** では対応マイルストーンの Exit criteria（§7.3 / §8.4 等）と
KPI 実績を同型で投入する。

### 3.6 前回 review の申し送り事項

前回の四半期 review 記録（`docs/governance/quarterly-reviews/YYYY-QN.md` の
「次回への引き継ぎ」欄）から:

- 前回「条件付き Go」で付された **是正条件**の進捗（M2-15-T14 の判定パターン参照）。
- D 監視の起点日と現況（v0.5 期以降、判定は M3 期）。
- 継続監視 A/B/H/E/F/L の前回時点との差分。
- 前回撤退分岐（§6）が起動されていた場合はその進捗と本 review での再評価点。

---

## 4. Outputs（review 実施後に発行する成果物）

### 4.1 記入済み T01 テンプレート

- **記入対象ファイル**: `docs/governance/vokra-go-nogo-<phase>.md`（v0.5 なら
  `vokra-go-nogo-v0.5.md`、M2-15-T01 の主成果物）。
- **記入内容**（M2-15-T01 の構成に従う）: 冒頭 review 目的 + 対象マイルストーン +
  出典（CLAUDE.md Kill switch 表 + `docs/milestones.md` §6 or §7 + NFR-MT-05）、
  Kill switch A〜L 状態表、C / J / K / D 判定欄、継続監視（A/B/H/E/F/L）現況欄、
  既知リスク欄（compliance/legal 含む）、末尾に総合 Go/No-go 判定 + exit path
  選択欄（撤退時のみ）と意思決定ログ（判定日・判定者）。
- **記入者と時期**: §3 の入力収集は CC が支援、判定欄の記入は依頼者（M2-15-T07 /
  T09 / T11 / T14）。
- **T01 テンプレート未整備時**: 初回 review 実施までに M2-15-T01 が land して
  いない場合は、本 runbook §5 の手順に沿った Markdown を review 実施者が
  ad-hoc に作成し、`docs/governance/vokra-go-nogo-<phase>.md` に配置する
  （空欄含む形で公開して次期に T01 テンプレを完成させる。fabricated pass 禁止）。

### 4.2 四半期 review 記録の公開

- **公開先**（M2-15-T15 = 依頼者選択）: (a) 公開リポジトリの governance 文書
  として `docs/governance/quarterly-reviews/YYYY-QN.md` を PR で反映、
  (b) GitHub Discussion にサマリ投稿、(c) post-mortem 相当 blog 公開。
- **repo 管理を選択する場合**: main への直接 push 不可（NFR-MT-06/07 = PR 必須 +
  CI ゲート）ゆえ、PR → required checks 通過 → マージの手順を踏む。docs のみの
  PR でも CI 品質ゲート（NFR-MT-07 の 7 項目のうち該当する doc / ライセンス
  チェック等）を通過させる。
- **内部計画文書の非露出**: 本 runbook / M2-15 チケット spec / `docs/milestones.md`
  等の**内部計画物**を公開記録に取り込まない。**review 結論・Kill switch 状態・
  指標・（撤退時）exit path のみ**を公開する切り分けを行う（M2-15-T15 = 依頼者判断）。
- **公開先 URL の記録**: WP 管理記録（M2-15 の WP Issue、`docs/milestones.md` §13.3）
  に公開 URL / マージ済み PR を残す。

### 4.3 T03 指標 JSON の格納

- 兄弟 runbook §6 に従い、`docs/governance/quarterly-reviews/YYYY-QN.metrics.json`
  として保存。判定 `.md` と `.metrics.json` の two-file 構成を推奨。
- 生 JSON にセンシティブ情報は無い（API token 等は含まれない）。公開 repo 反映
  可否は依頼者判断だが、CLAUDE.md 記載どおり本 runbook 自体は公開 governance
  ドキュメントとして扱って問題ない。

### 4.4 D 監視起点日の記録（v0.5 review 限定）

`docs/tickets/m2/M2-15-quarterly-review.md` T12 に従い、v0.5 公開日 = D 正式判定
（M3 期）の起点日を review 記録に残す。M3 期の review に引き継ぐための項目は:

- v0.5 release tag `v0.5.0` の日付。
- D 計測方法（bot / Claude Code を除外した外部コミッター数、兄弟 runbook §2
  参照）。
- D 起点からの経過月と正式判定予定日（起点 + 3 ヶ月）。

---

## 5. 意思決定者と分担

| 役割 | 担当 | 根拠 |
|------|------|------|
| **総合 Go/No-go 判定**（撤退可否 / 条件付き / 撤退時 exit path 選択） | **依頼者**
（`ayutaz`） | `docs/tickets/m2/M2-15-quarterly-review.md` T14。撤退可否・市場判断・
法務判断は Go/No-go の本質であり CC で代替不可（§11 分担原則） |
| Kill switch C / J / K 個別判定 | **依頼者** | M2-15-T07 / T09 / T11 |
| Kill switch D 監視開始（v0.5）+ 正式判定（M3 期） | **依頼者** | M2-15-T12 |
| L（burnout / 資金枯渇）自己評価 | **依頼者** | M2-15-T05 / T14 |
| A / B / H 該当有無の最終判断 | **依頼者** | M2-15-T04 は事実収集のみ、判断は T14 |
| E / F 発動判断 | **依頼者** | E は「HF+ggml の音声特化 op 追加を検知した瞬間」
自動発動を要するが、検知確認と発動宣言は依頼者。CC は事実として収集のみ |
| §3 指標収集 / 事実収集 / 材料編纂 | **CC**（判断を伴わない、M0-11 と同型分担） | M2-15-T02〜T06、T08、T10、T13 |
| §4 T01 テンプレートの整備 / exit playbook 整備 | **CC** | M2-15-T01 / T13 |
| §4.2 公開先選択と公開 PR 発行 | **依頼者** | M2-15-T15 |
| §4.2 WP 完了条件充足確認と Done 確定 | **依頼者** | M2-15-T16 |

---

## 6. Release process への組込（NFR-MT-03 4 週間 release train との連動）

### 6.1 チェックポイント設計

NFR-MT-03（`docs/system-requirements.md` §3.5）は 4 週間 release train を規定する。
四半期 = 12 週 = release train 3 本相当。

**チェックポイント配置**:

| 位置 | イベント | 本 runbook との関係 |
|-----|---------|---------------------|
| Release train N（週 0-4） | 通常リリース | 平常 CI（NFR-MT-07 required checks 7 項目）と NFR-QL-04（nightly 音声品質） |
| Release train N+1（週 4-8） | 通常リリース | 同上 |
| Release train N+2（週 8-12） | 通常リリース + **四半期 review 準備**（week 11
までに §3 入力を CC が準備完了させる） | Kill switch 状態表の暫定案・T03 指標 JSON
の暫定生成・T04/T05 の暫定要約を用意 |
| **四半期 review 実施**（週 12 前後） | 依頼者による総合判定 = §5 | §3 入力を
消費 → §4 出力を発行 |
| Release train N+3（次四半期の week 0-4） | **Go 判定確認後**に次フェーズ WP 着手 | 判定 = Go / 条件付き Go の場合のみ次フェーズ WP を通常運用で着手。No-go の場合は §6.2 分岐 |

### 6.2 撤退分岐（Kill switch 該当時の exit path 起動）

Kill switch A〜L のいずれかが該当（撤退側）と判定された場合、`docs/tickets/m2/M2-15-quarterly-review.md`
**T13 の exit path playbook**（提案先ファイル: `docs/governance/exit-path-playbook.md`）
から具体的経路を依頼者が選択する。

**Kill switch → exit path 対応の目安**（`docs/tickets/m2/M2-15-quarterly-review.md`
T13 内容、CLAUDE.md「Rhasspy 型『上位エコシステムに merge』撤退は幻想」注記の
継承）:

| 発動 Kill switch | 有力 exit path |
|------|------|
| **J**（HA Voice が Wyoming で Vokra を採用しない） | Wyoming Protocol 準拠実装
として HA に統合される（M2-09 の Wyoming 実装を受け皿化） |
| **F / E**（Candle / ggml エコシステム側で全面 native 化 or llama.cpp に音声特化
op 追加） | Candle audio extension として merge / HuggingFace / ggml-org へ acquire
される（Tracel AI Burn precedent） |
| **上記いずれも困難** | post-mortem blog 公開（Coqui 型の突然 shutdown を避ける） |

**release train 側の運用**:

- 撤退決定後は次 release train の通常リリースを停止し、選択した exit path の
  移行作業（Wyoming/HA 統合 PR、Candle への upstream PR、HF 側との acquire 交渉、
  post-mortem 執筆等）に切り替える。
- CI 品質ゲート（NFR-MT-07）は移行完了まで維持する（撤退時も semver 準拠・
  reproducible build・SBOM の運用は継続、NFR-MT-03）。
- **T13 exit playbook が未整備の段階**（本 runbook land 時点 = 2026-07-11 時点で
  `docs/governance/exit-path-playbook.md` は未 land）で撤退が発動した場合は、
  本 runbook §6.2 の対応表を暫定 playbook として使い、review 実施と並行して
  T13 の恒久 playbook を land する。

### 6.3 Go / 条件付き Go の release train 継続

- **Go**: 次四半期の release train を通常運用で継続。次フェーズ WP 着手を
  ブロックしない。
- **条件付き Go**: 是正条件を明記し、次四半期 review で再評価する（`docs/tickets/m2/M2-15-quarterly-review.md`
  T14 の判定パターン）。release train 自体は継続、当該領域の WP 着手は
  条件付き承認とする。

---

## 7. 実施手順（Step-by-step、四半期 1 回）

以下は 1 回の四半期 review 実施フロー。所要は準備 §3 で数時間、当日判定 §5 で
1〜2 時間を目安とする。

### 7.1 準備フェーズ（review 実施の 1〜2 週間前、CC 支援可）

1. **前回 review の申し送り確認**（§3.6）。
2. **T03 指標 JSON の生成**（§3.1）:
   ```bash
   bash scripts/kill-switch-metrics.sh \
     > docs/governance/quarterly-reviews/YYYY-QN.metrics.json
   ```
   `gh auth status` で `ayutaz` として認証済であることを事前確認する。
   兄弟 runbook §1〜§3 の各コマンドが個別に動作することも確認する。
3. **T04 競合スキャン**（§3.2）: A / B / H の changelog を出典リンク付きで
   要約収集。
4. **T05 常時・緊急監視**（§3.3）: E / F の現況を出典付きで要約収集。L の
   自己申告欄を用意（記入は §7.2）。
5. **T08 J 判定材料**（§3.4、v0.5 期のみ）: M2-09 の Wyoming 実装状況 + HA Voice
   採用シグナルを収集。
6. **M2 / M3 Exit criteria 実績転記**（§3.5）: 対応マイルストーンの実績値を
   §3.5 の項目に沿って転記。
7. **T01 テンプレートを YYYY-QN 用にコピー**（§4.1）: `docs/governance/vokra-go-nogo-<phase>.md`
   から `docs/governance/quarterly-reviews/YYYY-QN.md` を新規作成、Kill switch
   A〜L 状態表と判定欄を空欄で用意する。

### 7.2 判定フェーズ（review 当日、依頼者）

1. **Kill switch A / B / H**（§3.2 材料 → §5 依頼者判断）: 各スイッチの該当有無
   と根拠を状態表に記入。
2. **Kill switch E / F / L**（§3.3 材料 → §5 依頼者判断）: E は「検知されて
   いない」場合も明記。L は依頼者が現況を記入。
3. **Kill switch C**（v0.1 MVP 公開後 3 ヶ月経過している場合、§3.1 材料
   → M2-15-T07）: star ≥ 500 かつ engagement proxy ≥ 20 で PASS 判定。
   proxy 対応（どの GitHub 指標を Discord active user 20 に対応させるか）を
   依頼者が確定・記録する。
4. **Kill switch J**（v0.5 時点、§3.4 材料 → M2-15-T09）: HA Voice の Wyoming
   採用可否を評価。M2-09 が未達なら「判定不能（実装先行）」で次期持越し。
5. **Kill switch K**（v0.5 時点、§3.5 材料 → M2-15-T11）: 競合と分母を確定し、
   自社 addressable market が競合の 10% 以上か判定。
6. **Kill switch D**（v0.5 期は監視開始のみ、§3.1 の contributor 数 → M2-15-T12）:
   v0.5 では計測体制確立と起点日記録に留める。M3 期の review で正式判定。
7. **既知リスク欄**（M2-15-T14 の対象、`docs/tickets/m2/M2-15-quarterly-review.md`
   「改訂記録 (c)」）: watermark / C2PA（FR-CP-01/02）未実装の現況、EU AI Act
   Article 50（2026-08-02 enforcement）、California SB 942（施行済）の
   compliance / legal リスクを記録。
8. **総合 Go/No-go 判定**（§5 → M2-15-T14）: 上記 A〜L 統合 + Exit criteria 実績
   （§3.5）を統合し、Go / 条件付き Go / No-go（撤退）のいずれかを結論。
   - Go / 条件付き Go → §6.3、release train 継続。
   - No-go（撤退） → §6.2、T13 exit path から具体経路選択、release train 切替。
9. **判定日・判定者を記録**（§4.1）。E 緊急発動を検知していた場合は「四半期を
   待たず即時 review を起こしていた旨」も反映。

### 7.3 公開・引き継ぎフェーズ（依頼者、review 後 1 週間以内）

1. **公開先の選択と公開**（§4.2 → M2-15-T15）: repo PR / Discussion / blog の
   いずれか。内部計画文書非露出を厳守。
2. **公開 URL を WP 管理記録に残す**（M2-15 WP Issue、`docs/milestones.md` §13.3）。
3. **完了条件充足の逐条確認**（§5 → M2-15-T16）: C / J / K 判定実施 + D 監視
   開始 + 総合判定 + review 記録公開の 4 項目 + 撤退時 exit path 移行方針。
4. **次期への引き継ぎ**（§3.6 の逆方向）: D の M3 期正式判定 / 継続監視の次回
   確認 / 条件付き Go の是正条件を次 review の申し送り欄に登録。
5. **WP Done 確定**: `docs/milestones.md` §13.1 に基づき、依頼者確認をもって
   M2-15 を Done 確定（M2-15 の場合。M3-19 等も同型）。

---

## 8. Cross-references

- 兄弟 runbook: [kill-switch-metrics-runbook.md](kill-switch-metrics-runbook.md)（VOKRA-GOV-001）
- WP spec: `docs/tickets/m2/M2-15-quarterly-review.md`（M2-15 全 16 チケットの詳細）
- Kill switch SSOT: CLAUDE.md「Kill switch（撤退条件、四半期評価に格上げ）」節
- 要件: `docs/system-requirements.md` §3.5 NFR-MT-03（4 週間 release train）/
  NFR-MT-05（本 runbook の根拠）/ NFR-MT-06 / NFR-MT-07（CI 品質ゲート）/
  NFR-MT-08（CD 自動リリース）
- 廃止された FR-TL-05: `docs/system-requirements.md` §2.11 FR-TL-05 セル
  （廃止マーク済）
- マイルストーン計画: `docs/milestones.md` §6（M2 = v0.5、M2-15 の主成果物と
  Kill switch 評価カレンダー）/ §7（M3 = v0.9、M3-19 = 次四半期 review）/
  §12（Kill switch 評価カレンダー全体）
- exit path playbook（未 land、M2-15-T13 で整備予定）: `docs/governance/exit-path-playbook.md`
- T01 テンプレート（未 land、M2-15-T01 で整備予定）: `docs/governance/vokra-go-nogo-v0.5.md`

---

## 9. 変更履歴

| 日付 | 変更 | 根拠 / 出典 |
|---|---|---|
| 2026-07-11 | 初版（M2-15-T02）。FR-TL-05 廃止 + NFR-MT-05 手動 review への読替を
明示、自動 `.yml` 新設・再稼働禁止を §0.3 に明記、release train（NFR-MT-03、4 週）と
連動する四半期チェックポイント設計を §6 に定義、撤退分岐の T13 exit path 対応表を
§6.2 に整備、実施フローを §7 に step-by-step 化 | `docs/tickets/m2/M2-15-quarterly-review.md`
T02、2026-07-04 依頼者決定（FR-TL-05 廃止）、2026-07-06 依頼者決定（Discord 全体非採用） |
