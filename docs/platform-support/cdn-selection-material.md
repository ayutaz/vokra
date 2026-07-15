# CDN 選定材料 (M4-11-T09 — owner 記入式)

> **本ファイルは CC 作成の判断材料であって選定結果ではない。**
> **選定・契約・費用判断は一切 CC で行わない (owner 領分、T12)**。CC は要件 checklist と候補比較の枠のみ用意し、候補の性能・価格を出典なしに記入しない (ハルシネーション厳禁 = CLAUDE.md)。
> 選定結果は `docs/platform-support/v1.0-rc-support-matrix.md` Part 9 (CDN 節) に owner が記録する。

**対象**: Web 配布 (WASM バイナリ / npm 外の直配信 / GGUF モデルファイル) の CDN 選定 (milestones §8 M4-11 依頼者タスク「CDN 選定」、§11 M4/v1.0-rc close 行「実機テスト (Web ブラウザ実機) + CDN 選定」)。
**依存**: M4-01 (WASM/WebGPU/npm) の配布物確定後に実配信構成を詰める。要件 checklist の準備は M4-01 完了前に着手可 (先行タスク、spec §先行/Web 待ち分割)。

---

## 1. 要件 checklist (技術制約は既存文書から導出、発明しない)

| # | 要件 | なぜ必須か | 出典 | owner 確認 |
|---|------|-----------|------|-----------|
| R1 | **cross-origin isolation ヘッダ (COOP/COEP) の設定可否** | WASM Threads は `SharedArrayBuffer` + COOP (`Cross-Origin-Opener-Policy: same-origin`) + COEP (`Cross-Origin-Embedder-Policy: require-corp`) が前提。**配信側でレスポンスヘッダを制御できない CDN/ホスティングでは Threads leg が成立しない** (single-thread fallback に縮退)。 | CLAUDE.md ISA カバレッジ「WASM Threads (SharedArrayBuffer + COOP/COEP)」/ milestones §8 M4-01 対応 ISA 列 | ☐ 可 / ☐ 不可 |
| R2 | **WASM の MIME type (`application/wasm`) 配信 + streaming compilation 対応** | `WebAssembly.instantiateStreaming` は `application/wasm` MIME + 適切な `Content-Encoding` を要求。誤 MIME だと streaming compile が無効化され初期化が遅延。 | WebAssembly Web API 一般制約 (M4-01 の Web 実装が消費) | ☐ 可 / ☐ 不可 |
| R3 | **大容量 GGUF モデルファイルの Range request / キャッシュ / 転送コスト** | GGUF は mmap 前提の大容量 asset。ブラウザ配信では帯域コストが Web デモ運用費の支配項になる見込み。Range request 対応の可否・キャッシュ TTL・egress 課金体系が比較観点。**実測系の数値はここに書かない** (owner が候補ごとに確認)。 | CLAUDE.md 設計判断 3 (GGUF は mmap 可能・大容量) / deliverables §3.2 Web 行 | ☐ 確認 |
| R4 | **リージョン / 無料枠 / 帯域課金モデル** | OSS デモ運用費の見積り。無料枠の egress 上限、超過後の $/GB。 | 運用コスト観点 (X-07 packaging / release engineering、milestones §12 予算配分) | ☐ 確認 |
| R5 | **`vokra.dev` / `vokra.io` / `vokra.ai` ドメインとの接続性** | Web デモ・npm 配信の canonical ドメイン。未取得ならその旨を honest 記録し、取得は owner backlog。 | CLAUDE.md「ドメイン取得推奨」/ milestones §11 先行タスク「ドメイン取得」行 | ☐ 取得済 / ☐ 未取得 |
| R6 | **npm レジストリ配信との分離** | npm パッケージ自体は npm registry (CD 自動発行 = M4-01 完了条件) が担う。CDN が扱うのは (a) npm 外の直配信 (`<script>` 直読み / unpkg 相当) と (b) GGUF モデルファイル。二重配信の切り分けを owner が確定。 | milestones §8 M4-01 完了条件「npm パッケージが CD で自動発行される (NFR-MT-08)」 | ☐ 確認 |

---

## 2. 候補比較 scaffold (owner 記入式)

> **候補名の列挙は owner の選好・契約条件に依存するため、CC は枠のみ用意する。**
> 具体候補の性能・価格・無料枠の記入と検証は owner (T12)。CC が候補を出典なしに評価しない。

| 候補 CDN | R1 COOP/COEP 制御 | R2 wasm MIME | R3 Range/キャッシュ/egress | R4 無料枠 / 帯域課金 | R5 ドメイン接続 | 備考 |
|----------|-------------------|--------------|----------------------------|----------------------|-----------------|------|
| _(owner 記入)_ | ☐ 可 / ☐ 不可 | ☐ 可 / ☐ 不可 |  |  |  |  |
| _(owner 記入)_ | ☐ 可 / ☐ 不可 | ☐ 可 / ☐ 不可 |  |  |  |  |
| _(owner 記入)_ | ☐ 可 / ☐ 不可 | ☐ 可 / ☐ 不可 |  |  |  |  |

**必須 gate (R1)**: COOP/COEP レスポンスヘッダを制御できない候補は WASM Threads 不成立ゆえ選定不可 (single-thread のみで可とするかは owner 判断、その場合はデメリットを記録)。

---

## 3. 選定後の記録先

owner は T12 で選定を確定し、以下を `docs/platform-support/v1.0-rc-support-matrix.md` Part 9 に記録する:

- 採用 CDN + 選定根拠
- COOP/COEP 設定方法 (実配信での header 設定手順)
- 費用見込み
- ドメイン接続構成 (未取得ならその旨 + owner backlog 化)

---

## 改訂記録

- **2026-07-15 (初版・M4-11-T09)**: 要件 checklist (R1–R6、各項目に出典 ID) + 候補比較 scaffold (owner 記入式) を land。COOP/COEP を必須 gate (R1) として明記。候補の具体評価は owner (T12)、CC は発明しない。出典 = CLAUDE.md ISA カバレッジ / 設計判断 3 / milestones §8 M4-01 完了条件 / §11 M4 close 行 / deliverables §3.2 Web 行。
