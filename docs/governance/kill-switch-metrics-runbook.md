# Kill Switch C / K メトリクス計測 runbook

**文書 ID**: VOKRA-GOV-001
**最終更新**: 2026-10-08（exact GitHub `main`、tag/release APIとtracked review recordの現況を再照合）
**位置付け**: [`vokra-go-nogo-v0.5.md`](vokra-go-nogo-v0.5.md) の Kill switch 表
（NFR-MT-05、四半期手動 Go/No-go review）で
`C`（v0.1 MVP 公開後 3ヶ月で GitHub star < 500、active user < 20）と
`K`（v0.5 時点で addressable market が競合の 10% 未満）を判定するための
**再現可能で機械的なメトリクス収集手順**。判定そのものは依頼者（`ayutaz`）が行う。
本 runbook は「何を、いつ、どう数えるか」だけを固定する。

**現行状態（2026-10-08、exact `main` `d100d93778191ccab77bd1c37fe3552e3d889758`）**: `release-cadence.yml` と
`tools/release/test_cadence.py` は land 済み。all-pages APIは **git tag 1 / GitHub
release 1**（`v0.3.0`、2026-09-20）を返し、tracked quarterly review record は0。
単一releaseではrelease cadence は未確立。
この状態を Kill switch の判定結果とは扱わない。

**対象条件（現行判定表より抜粋）**:

| # | 条件 | チェック時期 |
|---|-----|-----|
| **C** | v0.1 MVP 公開後 3ヶ月で GitHub star < 500、Discord active user < 20 | 5–6 ヶ月時点 |
| **K** | v0.5 時点で addressable market（Unity Asset Store DL 数、GitHub star、Discord DAU）が競合の 10% 未満 | v0.5 時点 |

**Discord は非採用（2026-07-04 依頼者決定）** ゆえ、Kill switch C の "Discord active user < 20"
は GitHub Issues / Discussions の engagement **proxy** で代替判定する。proxy は
collector が返した author unique count であり、実際の直近 3 ヶ月の作成・コメント
イベントの完全な census ではない。本 runbook はこの proxy の手順と制限を定義する。

---

## 0. 前提

- リポジトリ: `ayutaz/vokra`（2026-07-04 public 化済）
- 認証: `gh auth status` で `ayutaz` として認証済（本 runbook は Bash + `gh` + `jq` のみを使う）
- 実行環境: 依頼者のローカルマシン（本 runbook は CI に載せない — 手動四半期 review 前提）
- 依存: `gh` CLI（GitHub 公式）、`jq`（JSON パーサ）。両方とも Vokra runtime の zero-dep 対象外
  なので brew / apt で導入して問題ない

---

## 1. GitHub Star 数

`stargazerCount` は `gh repo view --json` の 1 フィールドで取れる。

```bash
gh repo view ayutaz/vokra --json stargazerCount --jq .stargazerCount
```

**出力例**: `123`（整数のみ、改行付き）

**判定への使い方**:
- Kill switch C 閾値: `500`（v0.1 MVP 公開後 3ヶ月）
- Kill switch K: 競合（sherpa-onnx、whisper.cpp、Candle 等）の star 数と依頼者が手動比較。
  競合値は本 runbook では自動収集しない（`ayutaz/vokra` 以外のリポジトリの選定は
  依頼者判断のため）。

---

## 2. コントリビュータ数（bot と Claude Code を除外）

ここでの「Claude Codeを除外」は、Kill switch/DoDに定められた**規範上の集計閾値**
である。現行Codexのagent運用を理由に閾値や除外ルールを変更しない。実際のreview
では、ownerを除外する別集計も併記し、どちらを判定に使ったかをownerが記録する。

`GET /repos/{owner}/{repo}/contributors` は login と contributions を返す。
bot（`*[bot]` / `-bot` パターン）と `Claude*` を `jq` で除外する。

```bash
gh api "repos/ayutaz/vokra/contributors?per_page=100" --paginate \
  | jq '[.[] | select(.login | test("bot|Claude") | not)] | length'
```

**出力例**: `2`（整数のみ）

**注意事項**:
- `--paginate` を付けないと最大 100 人までしか集計されない（Link header page 2+ を追わない）
- `--paginate` の複数ページは JSON 配列ごとに `jq` へ渡るため、この形の filter は
  page-local な件数を複数行で返し、全ページを合算する保証がない。100人を超える
  contributor がいる場合、collector の出力だけでは全件数を証明せず、owner が別途
  確認して記録する。
- `jq` の `test("bot|Claude") | not` は case-sensitive 正規表現。実際の bot 命名慣行
  （`dependabot[bot]`、`github-actions[bot]`、`renovate[bot]`）は末尾 `[bot]` を含むが、
  部分マッチ `bot` で十分ヒットする。`Claude` は Claude Code のコミッター名
  （`Claude Code` / `Claude` を含むログイン）を想定
- Kill switch D（v0.5 公開後 3ヶ月で「Claude Code 以外のコミッター」3 名未満）にも
  同じコマンドが流用可能

---

## 3. Issues / Discussions engagement proxy（政策窓: 直近 3 ヶ月）

Discord 廃止に伴う代替として、GitHub Issues + PR + Discussions の参加状況を
集計する。**方針上の「active」**は、直近 3 ヶ月にコメントまたは issue / PR /
discussion の作成イベントを実際に記録した unique login とする。一方、現行 collector
の出力はこの方針を完全に測る census ではなく、API が返す author の unique count
（以下「engagement proxy」）である。REST の issue 一覧は更新日時で絞るため古い
issue / PR 作成者を含み得、Discussions も discussion の更新日時だけで絞るため古い
comment author を含み得る。metrics の値を厳密な「直近 3 ヶ月 active」と表記せず、
proxy と制限を記録する。reaction はこの収集経路では取得しないため、active participant
の根拠に含めない。

### 3a. Issues + PR の engagement proxy

Issues API と PR API は `/repos/{owner}/{repo}/issues/comments`
と `/repos/{owner}/{repo}/issues` を `since` 付きで叩いて `user.login` を uniq-count する。
この `since` は更新日時による絞り込みであり、issue / PR の作成日時による絞り込みではない。

```bash
# 直近 3 ヶ月の since (macOS BSD date)
SINCE=$(date -u -v-3m +%Y-%m-%dT%H:%M:%SZ)
# Linux GNU date の場合は: SINCE=$(date -u -d '3 months ago' +%Y-%m-%dT%H:%M:%SZ)

# 直近 3 ヶ月に投稿された全 issue / PR comment の投稿者
gh api "repos/ayutaz/vokra/issues/comments?since=$SINCE&per_page=100" --paginate \
  | jq -r '.[].user.login' \
  > /tmp/vokra-issue-comment-authors.txt

# 直近 3 ヶ月に更新された issue / PR の作成者（state=all で closed も含む）
gh api "repos/ayutaz/vokra/issues?since=$SINCE&state=all&per_page=100" --paginate \
  | jq -r '.[].user.login' \
  > /tmp/vokra-issue-authors.txt

# uniq 集計（bot と Claude を除外）
cat /tmp/vokra-issue-comment-authors.txt /tmp/vokra-issue-authors.txt \
  | grep -v -E 'bot|Claude' \
  | sort -u \
  | wc -l
```

**出力例**: `5`（整数）。これは上記の engagement proxy であり、作成イベントだけの
直近 3 ヶ月 census ではない。

### 3b. Discussions engagement proxy

Discussions は GraphQL API が公式経路。ただし Discussions 未有効なリポジトリでは
`hasDiscussionsEnabled: false` となる。有効時のみ集計する。現行の collector は
discussion を最大 100 件、各 discussion の comment も最大 100 件取得するが、GraphQL
ページネーションは行わない。discussion の `updatedAt` だけで窓を絞り、comment 自身の
`updatedAt` は絞り込まないため、最近更新された discussion に含まれる古い comment author
も数え得る。この制限を metrics 記録に明記する。

```bash
# Discussions 有効かチェック
DISC_ON=$(gh repo view ayutaz/vokra --json hasDiscussionsEnabled --jq .hasDiscussionsEnabled)

if [ "$DISC_ON" = "true" ]; then
  # discussion の updatedAt 窓に入った discussion と、返却された comment の author を集計
  # GraphQL: discussions(first: 100, orderBy: {field: UPDATED_AT, direction: DESC})
  gh api graphql -f query='
    query($owner:String!, $repo:String!) {
      repository(owner:$owner, name:$repo) {
        discussions(first: 100, orderBy: {field: UPDATED_AT, direction: DESC}) {
          nodes {
            author { login }
            updatedAt
            comments(first: 100) {
              nodes { author { login } updatedAt }
            }
          }
        }
      }
    }' -F owner=ayutaz -F repo=vokra \
    | jq -r --arg since "$SINCE" '
        .data.repository.discussions.nodes
        | map(select(.updatedAt >= $since))
        | (map(.author.login) + (map(.comments.nodes[]?.author.login))) []' \
    | grep -v -E 'bot|Claude' \
    | sort -u \
    | wc -l
else
  echo "0  # Discussions not enabled"
fi
```

**出力例**: `3`（整数、または `0  # Discussions not enabled`）。これも engagement
proxy であり、comment の実際の投稿日時だけを数えるものではない。

### 3c. Fallback: events API

Issues / Discussions API に到達できない or rate-limit 逼迫時は、Events API を fallback として
参照する（過去 90 日相当のイベントストリーム、bot 除外は同様）。

```bash
gh api "repos/ayutaz/vokra/events?per_page=100" --paginate \
  | jq -r --arg since "$SINCE" '
      [.[]
       | select(.created_at >= $since)
       | select(.type | IN("IssuesEvent","IssueCommentEvent","PullRequestEvent",
                            "PullRequestReviewCommentEvent","DiscussionEvent",
                            "DiscussionCommentEvent"))
       | .actor.login]
      | unique | length'
```

**注意**: Events API は過去 90 日 or 300 events のどちらか短い方までしか保持しない
（GitHub の仕様）。人気リポジトリでは 90 日未満で溢れる可能性があるため、Issues +
Discussions API での集計を primary、Events を fallback とする位置付けを堅持する。
ただし、tracked collector (`scripts/kill-switch-metrics.sh`) は Events API を自動で
呼ばない。primary API が利用できない場合の Events 集計は owner が手動で実施し、
fallback であることと API の保持限界を記録する。

---

## 4. 集約スクリプト（Kill switch C / K judgement input）

実行可能な canonical collector は [`scripts/kill-switch-metrics.sh`](../../scripts/kill-switch-metrics.sh)
だけである。ここにスクリプト本文を複製しない（複製は仕様 drift を生む）。tracked
file は既に executable なので、owner は次のように実行する。

```bash
bash scripts/kill-switch-metrics.sh --self-test
bash scripts/kill-switch-metrics.sh \
  > docs/governance/quarterly-reviews/2026-Q3.metrics.json
```

上記の `2026-Q3.metrics.json` は将来の実行で生成する出力例であり、現時点で
completed metrics snapshot や quarterly review record が存在することを意味しない。

collector が出力する JSON には stars、contributors（bot / Claude Code 除外、owner
除外版を含む）、Issues + PR + Discussions の unique participant **proxy** 数、C/K の input、
GA DoD item 4/5 の scaffold が含まれる。スクリプト自身の usage、依存、zero-activity
guard、JSON schema、現在の実装との差分は source file を正本とする。

**収集範囲の制限:** Issues の comments / issue 作成者は REST `since`（更新日時であり
作成日時ではない）と pagination を使う。Discussions は最大 100 discussion、各最大 100 comment で、GraphQL pagination は
なく、discussion の `updatedAt` のみを窓判定に使う。reaction は収集しない。Events API
fallback も collector には含まれない。したがって出力は API が返した範囲の snapshot で
あり、完全な participant census の証明ではない。review record にはこの制限、測定日時、
fallback の有無を転記する。競合比較と最終 Go/No-go は owner 判断であり、collector は
自動判定しない。

---

## 5. いつ走らせるか（cadence）

| Kill switch | 走らせる時期 | 契機イベント |
|-----|-----|-----|
| **C** | v0.1 MVP 公開後 3 ヶ月 経過時点（= 公開から 5–6 ヶ月目の四半期 review） | v0.1 MVP release tagを実際に打った日を起点にカレンダー登録（2026-09-09現在、tag/release 0件で未確定） |
| **D** | v0.5 公開後 3 ヶ月（コミッター 3 名未満判定） | v0.5 release tagを実際に打った日を起点にカレンダー登録。§2 のコマンドを流用（現時点は未確定） |
| **K** | v0.5 公開時点 | v0.5 release tag と同時に本 runbookを実行（現時点は未発行） |
| **その他四半期** | 四半期毎（3 月末 / 6 月末 / 9 月末 / 12 月末） | 手動 Go/No-go review（NFR-MT-05） |

**カレンダー登録は依頼者責任**（本 runbook は自動 CI に載せない = 2026-07-04 依頼者決定の
「kill-switch 自動監視 .yml は廃止 → 手動四半期 Go/No-go review」を尊重）。

**2026-10-08補足:** 表の2026-09-09時点の0件観測は履歴である。現在は
`v0.3.0`が1件公開されているが、予定上の`v0.1.0` / `v0.5.0`起点との対応は
owner未確定であり、既存release日を自動でカレンダーに代入しない。

---

## 6. 意思決定の記録先

判定の結果（Go / No-go、Kill switch 発動 / 継続、根拠、次アクション）は、
[`quarterly-reviews/README.md`](quarterly-reviews/README.md) が定める two-file 構成で
記録する。ここでは手順を複製せず、canonical な命名・テンプレート・公開方針を同 README
に集約する。

- Metrics snapshot: `docs/governance/quarterly-reviews/YYYY-QN.metrics.json`。
  `scripts/kill-switch-metrics.sh` の stdout を保存する。
- Review record: `docs/governance/quarterly-reviews/YYYY-QN.md`。blank template
  [`vokra-go-nogo-v0.5.md`](vokra-go-nogo-v0.5.md) をコピーして記入し、template 本体は
  編集しない。

`YYYY` は 4 桁西暦、`N` は 1〜4（Q1 = 1〜3 月、Q2 = 4〜6 月、Q3 = 7〜9 月、Q4 =
10〜12 月）とする。完了した review record は public `docs/governance/` に置くことを
既定とし、private planning material が必要な場合も結論だけを記録する。snapshot は
append-only、判断は `.md` に owner が追記する。

---

## 7. 変更履歴

| 日付 | 変更 |
|---|---|
| 2026-07-07 | 初版（Task #78）。Discord 廃止に伴い GitHub Issues + Discussions への代替判定を追加 |
