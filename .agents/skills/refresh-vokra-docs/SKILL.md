---
name: refresh-vokra-docs
description: Vokra の current/public documentation と dated engineering records を現状に合わせて更新するときに使う。履歴を壊さず、事実確認、静的 gate、保護対象、論理 commit の境界を守る。
---

# Vokra documentation refresh

目的は、読者が現行の使い方・status・制約を正しく判断できるようにすること。
対象と根拠を先に分け、履歴 evidence を現在値に書き換えない。Mac CPU/Metal
campaign を扱う場合は、次の canonical handoff を更新前後に読み直す。

- `docs/README.md` — current/public と dated records の境界、軽量 gate
- `docs/handoff/mac-pre-scaleway-remaining-tasks-2026-09-05.md` — live inventory と per-row facts
- `docs/handoff/mac-cpu-metal-execution-plan-2026-09-07.md` — execution order と exit gate
- `docs/handoff/mac-cpu-metal-owner-disposition-packet-2026-09-07.md` — owner/legal scope

## 対象を分類する

- **tracked current/public**: root README、`docs/README.md`、現行 guide/API/backend、
  platform tutorial、integrations、security/contributing/legal、generated-surface
  pointer。利用者向けの変更は English/Japanese twin と相対リンクを確認する。
- **tracked dated/historical**: `docs/handoff/`、`docs/bench-baselines/`、
  `docs/benchmarks/`、`docs/perf/`、tracked design records など。対象 commit、
  hardware、fixture、日付に紐づく evidence なので、古い
  数値・test count・HEAD を新しい値に置換しない。現状が変わったら冒頭または直後に
  audit-start/supersession note を追加する。
- **ignored local records**: `CLAUDE.md`、`docs/adr/`、`docs/tickets/`、
  `docs/_research/`、`docs/superpowers/`、ignored milestones/spec など。必要なら
  local planning 用に更新できるが、`git add -f` で公開物にしない。tracked
  current/public docs が canonical であることを明記する。

current snapshot には日付、baseline/HEAD、根拠、未完了範囲、非主張（例: Scaleway未実施、
UTMOS numeric parity未検証）を明記する。変動する件数や HEAD をこの skill に写さず、
ledger/API/script を都度再読して更新する。

## 事実確認と安全境界

1. `git status --short` を取り、既存の無関係な変更を保存する。
2. source、生成コード、manifest、checker、remote API など最も強い read-only source
   を照合し、推測・古いメモ・自作 mirror を事実として採用しない。資格情報、token、
   private endpoint を文書や command example に入れない。
3. モデル weight の download、load、forward、変換、実データ parity を maintainer Mac
   で行わない。2 GB 以上の artifact、workspace/`vokra-models` Cargo、重い audit は
   `vast-ai-workflow` に送り、VAST は rent/work/recover-small-evidence/destroy とする。
4. `tools/parity/cosyvoice2_llm_reference/license_gate_manifest.json` は owner の
   pre-existing dirty manifest で、閲覧・編集・stage・revert・cleanup の対象外。diff
   と commit の対象から必ず除外する。

## 更新と検証

同じ事実を複数の current docs に書くときは、canonical ledger へのリンクと日付付き
snapshot を添える。過去の記録を修正する代わりに supersession note を追加し、owner
sign-off、license class、provenance、parity 数値を発明しない。モデル status は
conversion、binding、native forward、independent parity、publication を分けて記す。

軽量な read-only/static gate を、環境の Python 3.12 は必ず `uv run` 経由で実行する。
必要な gate だけを選び、workspace/`vokra-models` Cargo はローカルで実行しない。

```sh
uv run --no-project --python 3.12 python tools/docs/check_doc_examples.py --self-test
uv run --no-project --python 3.12 python tools/docs/check_doc_examples.py
uv run --no-project --python 3.12 bash scripts/check-doc-references.sh --self-test
uv run --no-project --python 3.12 bash scripts/check-doc-references.sh
uv run --no-project --python 3.12 bash scripts/check-runbook-path-citations.sh
uv run --no-project --python 3.12 bash scripts/check-community-docs.sh
uv run --no-project --python 3.12 bash scripts/check-parity-sidecar-citations.sh
bash scripts/check-owner-checklist-drift.sh
bash scripts/check-platform-support.sh
bash scripts/check-abi-changelog.sh
bash scripts/check-workflow-hygiene.sh
if test -x scripts/check-codex-hooks.sh; then
  bash scripts/check-codex-hooks.sh
fi
git diff --check -- . ':(exclude)tools/parity/cosyvoice2_llm_reference/license_gate_manifest.json'
```

リンク、shell example、English/Japanese heading parity、runbook path、workflow hygiene
を修正後の全対象について確認する。`scripts/check-codex-hooks.sh` は hooks担当の
所有範囲で追加された場合だけ、実行前に `test -x` と担当からの完了を確認して実行する。
存在しないcheckoutではskipする。モデル実行や広域 Cargo を gate の一部にしない。

## 引き渡しと外部状態

変更は docs の論理単位（current/public、handoff/history、license/parity など）で
分け、各 commit の diff に protected manifest や unrelated change がないことを
確認する。commit は明示的に依頼された場合だけ行う。push、PR の作成・タイトル/本文
更新、CI の再実行・監視は別の外部状態変更であり、明示依頼がない限り行わない。
依頼された場合も、先に diff hygiene と relevant gates を通し、PR 本文の snapshot
hash/count と CI 状態を新しい HEAD に合わせる。モデル upload、repository withdrawal、
VAST/Scaleway allocation は docs refresh の許可に含まれない。
