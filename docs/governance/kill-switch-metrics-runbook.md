# Kill Switch C / K メトリクス計測 runbook

**文書 ID**: VOKRA-GOV-001
**最終更新**: 2026-07-07（初版、Task #78）
**位置付け**: CLAUDE.md の Kill switch 表（NFR-MT-05、四半期手動 Go/No-go review）で
`C`（v0.1 MVP 公開後 3ヶ月で GitHub star < 500、active user < 20）と
`K`（v0.5 時点で addressable market が競合の 10% 未満）を判定するための
**再現可能で機械的なメトリクス収集手順**。判定そのものは依頼者（`ayutaz`）が行う。
本 runbook は「何を、いつ、どう数えるか」だけを固定する。

**対象条件（CLAUDE.md より抜粋）**:

| # | 条件 | チェック時期 |
|---|-----|-----|
| **C** | v0.1 MVP 公開後 3ヶ月で GitHub star < 500、Discord active user < 20 | 5–6 ヶ月時点 |
| **K** | v0.5 時点で addressable market（Unity Asset Store DL 数、GitHub star、Discord DAU）が競合の 10% 未満 | v0.5 時点 |

**Discord は非採用（2026-07-04 依頼者決定）** ゆえ、Kill switch C の "Discord active user < 20"
は **GitHub Issues / Discussions の直近 3 ヶ月の active participant 数** で代替判定する。
本 runbook はこの代替判定の具体手順を定義する。

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

`GET /repos/{owner}/{repo}/contributors` は login と contributions を返す。
bot（`*[bot]` / `-bot` パターン）と `Claude*` を `jq` で除外する。

```bash
gh api "repos/ayutaz/vokra/contributors?per_page=100" --paginate \
  | jq '[.[] | select(.login | test("bot|Claude") | not)] | length'
```

**出力例**: `2`（整数のみ）

**注意事項**:
- `--paginate` を付けないと最大 100 人までしか集計されない（Link header page 2+ を追わない）
- `jq` の `test("bot|Claude") | not` は case-sensitive 正規表現。実際の bot 命名慣行
  （`dependabot[bot]`、`github-actions[bot]`、`renovate[bot]`）は末尾 `[bot]` を含むが、
  部分マッチ `bot` で十分ヒットする。`Claude` は Claude Code のコミッター名
  （`Claude Code` / `Claude` を含むログイン）を想定
- Kill switch D（v0.5 公開後 3ヶ月で「Claude Code 以外のコミッター」3 名未満）にも
  同じコマンドが流用可能

---

## 3. Issues / Discussions active participants（直近 3 ヶ月）

Discord 廃止に伴う代替として、GitHub Issues + PR + Discussions への active participant
（コメント投稿者、issue/PR 作成者、reaction 発火者）を集計する。**「active」= 直近 3 ヶ月に
最低 1 回コメント or 作成を投稿した unique login**。

### 3a. Issues + PR の active participants

Issues API と PR API は `/repos/{owner}/{repo}/issues/comments`
と `/repos/{owner}/{repo}/issues` を `since` 付きで叩いて `user.login` を uniq-count する。

```bash
# 直近 3 ヶ月の since (macOS BSD date)
SINCE=$(date -u -v-3m +%Y-%m-%dT%H:%M:%SZ)
# Linux GNU date の場合は: SINCE=$(date -u -d '3 months ago' +%Y-%m-%dT%H:%M:%SZ)

# 直近 3 ヶ月に投稿された全 issue / PR comment の投稿者
gh api "repos/ayutaz/vokra/issues/comments?since=$SINCE&per_page=100" --paginate \
  | jq -r '.[].user.login' \
  > /tmp/vokra-issue-comment-authors.txt

# 直近 3 ヶ月に作成された全 issue / PR の作成者（state=all で closed も含む）
gh api "repos/ayutaz/vokra/issues?since=$SINCE&state=all&per_page=100" --paginate \
  | jq -r '.[].user.login' \
  > /tmp/vokra-issue-authors.txt

# uniq 集計（bot と Claude を除外）
cat /tmp/vokra-issue-comment-authors.txt /tmp/vokra-issue-authors.txt \
  | grep -v -E 'bot|Claude' \
  | sort -u \
  | wc -l
```

**出力例**: `5`（整数）

### 3b. Discussions active participants

Discussions は GraphQL API が公式経路。ただし Discussions 未有効なリポジトリでは
`hasDiscussionsEnabled: false` となる。有効時のみ集計する。

```bash
# Discussions 有効かチェック
DISC_ON=$(gh repo view ayutaz/vokra --json hasDiscussionsEnabled --jq .hasDiscussionsEnabled)

if [ "$DISC_ON" = "true" ]; then
  # 直近 3 ヶ月に投稿された discussion + comment の author を集計
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

**出力例**: `3`（整数、または `0  # Discussions not enabled`）

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

---

## 4. 集約スクリプト（Kill switch C / K judgement input）

上記 §1〜§3 を一括実行し、判定に必要な JSON を出力する bash スクリプト。

```bash
#!/usr/bin/env bash
# scripts/kill-switch-metrics.sh
# Usage: bash scripts/kill-switch-metrics.sh > docs/governance/quarterly-reviews/2026-Q3.metrics.json
set -euo pipefail

OWNER=ayutaz
REPO=vokra
TODAY=$(date -u +%Y-%m-%d)

# BSD date (macOS) と GNU date (Linux) 両対応
if date -u -v-3m +%Y-%m-%dT%H:%M:%SZ >/dev/null 2>&1; then
  SINCE=$(date -u -v-3m +%Y-%m-%dT%H:%M:%SZ)
else
  SINCE=$(date -u -d '3 months ago' +%Y-%m-%dT%H:%M:%SZ)
fi

# 1. Stars
STARS=$(gh repo view "$OWNER/$REPO" --json stargazerCount --jq .stargazerCount)

# 2. Contributors (excluding bots and Claude Code)
CONTRIB=$(gh api "repos/$OWNER/$REPO/contributors?per_page=100" --paginate \
  | jq '[.[] | select(.login | test("bot|Claude") | not)] | length')

# 3a. Issues + PR active participants (3 months)
ISSUE_AUTHORS=$(gh api "repos/$OWNER/$REPO/issues/comments?since=$SINCE&per_page=100" --paginate \
  | jq -r '.[].user.login')
ISSUE_CREATORS=$(gh api "repos/$OWNER/$REPO/issues?since=$SINCE&state=all&per_page=100" --paginate \
  | jq -r '.[].user.login')

# 3b. Discussions (if enabled)
DISC_ON=$(gh repo view "$OWNER/$REPO" --json hasDiscussionsEnabled --jq .hasDiscussionsEnabled)
DISC_AUTHORS=""
if [ "$DISC_ON" = "true" ]; then
  DISC_AUTHORS=$(gh api graphql -f query='
    query($owner:String!, $repo:String!) {
      repository(owner:$owner, name:$repo) {
        discussions(first: 100, orderBy: {field: UPDATED_AT, direction: DESC}) {
          nodes {
            author { login } updatedAt
            comments(first: 100) { nodes { author { login } updatedAt } }
          }
        }
      }
    }' -F owner="$OWNER" -F repo="$REPO" \
    | jq -r --arg since "$SINCE" '
        .data.repository.discussions.nodes
        | map(select(.updatedAt >= $since))
        | (map(.author.login) + (map(.comments.nodes[]?.author.login))) []')
fi

ACTIVE=$(printf '%s\n%s\n%s\n' "$ISSUE_AUTHORS" "$ISSUE_CREATORS" "$DISC_AUTHORS" \
  | grep -v -E 'bot|Claude' \
  | grep -v '^$' \
  | sort -u \
  | wc -l \
  | tr -d ' ')

# Kill switch C verdict
if [ "$STARS" -ge 500 ] && [ "$ACTIVE" -ge 20 ]; then
  KSC_VERDICT="PASS"
else
  KSC_VERDICT="FAIL"
fi

# JSON output
cat <<EOF
{
  "measurement_date": "$TODAY",
  "repo": "$OWNER/$REPO",
  "window_since": "$SINCE",
  "stars": $STARS,
  "contributors_non_bot_non_cc": $CONTRIB,
  "issues_discussions_active_3mo": $ACTIVE,
  "kill_switch_c": {
    "threshold": {"stars_min": 500, "active_min": 20},
    "verdict": "$KSC_VERDICT",
    "note": "Discord は非採用（2026-07-04）ゆえ 'active user' は GitHub Issues + Discussions の直近 3 ヶ月の unique participants で代替判定"
  },
  "kill_switch_k": {
    "note": "competitor comparison is owner judgement; addressable market 10% threshold. 競合値の選定と比較は依頼者判断（本 runbook では自動収集しない）。",
    "verdict_input": {
      "vokra_stars": $STARS,
      "vokra_active_3mo": $ACTIVE,
      "unity_asset_store_dl": null,
      "competitor_reference": "sherpa-onnx / whisper.cpp / Candle 等の star 数は依頼者が手動記入"
    }
  }
}
EOF
```

**格納先**（推奨）: `scripts/kill-switch-metrics.sh` に恒久配置。パーミッションは
`chmod +x`。実行時は `bash scripts/kill-switch-metrics.sh` で出力を JSON として得る。

---

## 5. いつ走らせるか（cadence）

| Kill switch | 走らせる時期 | 契機イベント |
|-----|-----|-----|
| **C** | v0.1 MVP 公開後 3 ヶ月 経過時点（= 公開から 5–6 ヶ月目の四半期 review） | v0.1 MVP release tag（`v0.1.0`）を打った日を起点にカレンダー登録 |
| **D** | v0.5 公開後 3 ヶ月（コミッター 3 名未満判定） | v0.5 release tag（`v0.5.0`）を打った日を起点にカレンダー登録。§2 のコマンドを流用 |
| **K** | v0.5 公開時点 | v0.5.0 release tag と同時に本 runbook を実行 |
| **その他四半期** | 四半期毎（3 月末 / 6 月末 / 9 月末 / 12 月末） | 手動 Go/No-go review（NFR-MT-05） |

**カレンダー登録は依頼者責任**（本 runbook は自動 CI に載せない = 2026-07-04 依頼者決定の
「kill-switch 自動監視 .yml は廃止 → 手動四半期 Go/No-go review」を尊重）。

---

## 6. 意思決定の記録先

判定の結果（Go / No-go、Kill switch 発動 / 継続、根拠、次アクション）は
以下の四半期 review 記録ファイルに追記する。

**パス命名規約**: `docs/governance/quarterly-reviews/YYYY-QN.md`

- `YYYY` = 4 桁西暦
- `N` = 1〜4（Q1 = 1〜3 月、Q2 = 4〜6 月、Q3 = 7〜9 月、Q4 = 10〜12 月）
- 例: `docs/governance/quarterly-reviews/2026-Q3.md`

**書式（推奨テンプレ）**:

```markdown
# 2026-Q3 Quarterly Go/No-go review

**開催日**: 2026-09-30
**参加者**: ayutaz（依頼者、意思決定者）
**参照メトリクス**: `docs/governance/quarterly-reviews/2026-Q3.metrics.json`
（`bash scripts/kill-switch-metrics.sh > docs/governance/quarterly-reviews/2026-Q3.metrics.json` で生成）

## 対象 Kill switch

- Kill switch C（v0.1 MVP 3 ヶ月経過時点）: <status>
- Kill switch K（v0.5 時点）: <status>
- その他監視項目（A/B/E/F/G/H）: <status>

## メトリクス（測定値）

| 項目 | 値 | 閾値 | 判定 |
|-----|---|---|-----|
| GitHub stars | N | 500 | PASS/FAIL |
| Issues/Discussions active (3mo) | N | 20 | PASS/FAIL |
| non-bot non-CC contributors | N | 3 (Kill switch D) | PASS/FAIL |

## 意思決定

- Go / No-go: **<GO | NO-GO | HOLD>**
- 発動する Kill switch: <該当なし | C | K | ...>
- 根拠（依頼者記入）:
- 次アクション: <続行 | 撤退 | 方針転換 | 次回 review 前倒し>
- 次回 review: YYYY-MM-DD
```

**記録は git 管理下**（`docs/governance/` は public repo に含めるかは依頼者判断。
CLAUDE.md 記載のとおり、本 runbook 自体は public 化した `ayutaz/vokra` の docs 配下に
配置しても問題ない — メトリクス収集手順にセンシティブ情報はない）。
`.metrics.json` は生 JSON、判断は `.md` に人手で追記する two-file 構成を推奨。

---

## 7. 変更履歴

| 日付 | 変更 |
|---|---|
| 2026-07-07 | 初版（Task #78）。Discord 廃止に伴い GitHub Issues + Discussions への代替判定を追加 |
