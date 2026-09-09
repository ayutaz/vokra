---
name: complete-mac-cpu-metal
description: Vokra の公開モデルを Mac CPU と Apple Metal の完了判定まで調査・実行するときに使う。モデル一覧、状態遷移、VAST と Scaleway の境界、owner/legal の承認、保護対象 manifest、完了監査を扱う。
---

# Mac CPU / Apple Metal 完了ワークフロー

この skill は、公開モデルの現状を調査し、CPU parity と最後の Apple
CPU/Metal 実機判定までを fail-closed で進めるためのもの。件数、HEAD、
モデル名、既存 evidence をこの skill に複製しない。毎回、次の canonical
document を読み直し、そこに記録された日付・commit・scope を使う。

- `docs/handoff/mac-pre-scaleway-remaining-tasks-2026-09-05.md` — 一覧、行別事実、履歴 evidence
- `docs/handoff/mac-cpu-metal-execution-plan-2026-09-07.md` — 順序、状態遷移、Scaleway exit gate
- `docs/handoff/mac-cpu-metal-owner-disposition-packet-2026-09-07.md` — hash-bound owner/legal decisions
- `docs/README.md` — current/public docs と dated record の読み分け
- `AGENTS.md` — memory、依存、push、保護対象の共通制約

## 最初に行う調査

1. `git status --short` と現在の branch/HEAD を確認し、既存の変更を保存する。
2. canonical ledger の live inventory と per-row table を再読する。古い handoff の
   数値や HEAD は履歴として扱い、現在値と混ぜない。
3. 各行について source、artifact、license/dependency、converter/binder/native
   forward、独立 reference、CPU parity、Metal parity、publication のどこが
   未完かを分類する。build、manifest、synthetic fixture、zero-test、device-less
   Metal compileだけでは後段の完了に昇格させない。
4. `tools/parity/cosyvoice2_llm_reference/license_gate_manifest.json` は owner の
   dirty file であり、閲覧・編集・stage・revert・clean-up をしない。

## 行の状態遷移

各 unresolved row は後戻りせず、事実を満たした時だけ次へ進める。

1. `SOURCE_BLOCKED`: exact source/topology、依存、license、dataset、tokenizer、codec
   の事実が足りない。Scaleway で推測して埋めない。
2. `SOURCE_READY`: strict converter/binder/native/CLI と unsupported-path の明示的
   error、独立 upstream reference の道筋が存在する。
3. `APPROVAL_BLOCKED`: exact hash-bound scope は揃ったが owner/legal の署名がない。
   空欄や曖昧な承認は拒否として扱わず、この状態を維持する。
4. `VAST_READY`: 必要な承認、model-free API/dependency probe、license gate が green。
5. `CPU_PASS_METAL_NOT_RUN`: exact clean HEAD で no-upload conversion、独立 reference、
   実 weight CPU parity と直接転送 packet が VAST で pass。
6. `APPLE_PASS`: Scaleway の Apple CPU/reference、Metal/reference、Metal/CPU parity が
   explicit no-fallback で pass。hardware が無ければ pass ではない。
7. `PUBLICATION_BLOCKED` または `COMPLETE`: corrected artifact は別の upload/withdrawal
   authorization が済むまで publication blocked。live inventory、artifact、
   Apple evidence、決定記録が全て一致した時だけ complete。

skip、inspection-only、synthetic-only、未承認の bound、CPU fallback、単なる compile
では遷移させない。独立 reference は固定した upstream を実行し、自作 mirror を oracle
にしない。

## どこで何をするか

- Maintainer Mac: model download/execution を一切せず、軽量な read-only inspection、
  package-scoped check、format、docs/static gate のみ。workspace 全体または
  `vokra-models` の Cargo と 2 GB 以上（shard 合計）の artifact は実行しない。
- VAST: 2 GB 以上の conversion/reference/parity、workspace/`vokra-models` Cargo、
  大きい dependency audit を disposable worker で行う。まず model-free gate、
  owner-approved exact scope、次に no-upload real-weight CPU parity の順にする。
  `vast-ai-workflow` の `rent → provision → work → recover small evidence → destroy`
  を守り、`vastai-safe.sh` を経由し、無関係な instance を触らない。evidence 回収後
  は保存データを含めて destroy し、stop や storage を残さない（直近の Scaleway
  転送が明示され、backup 済みの短い handoff だけが例外）。
- Owner/legal: primary source の license、dataset、redistribution、operator の
  exact hash-bound decision を決める。agent は §3.1 sign-off、commercial/research
  判定、withdrawal を発明しない。未解決の source fact は compute で解消できない。
- Scaleway: 最後の compute/hardware service。全 row の non-Apple source/license/
  dependency/converter/binder/reference と、承認済み real-weight VAST CPU leg が
  完了または owner-approved withholding/withdrawal になり、final clean HEAD の
  static/remote gates と fresh transfer packet が揃ってから provision する。Apple
  worker は packet の hash、HEAD、model identity を検証し、CPU、Metal、no-fallback
  を個別に記録する。

Scaleway は不足した runtime、artifact、license、dependency、reference、owner decision
の代替ではない。Apple run が失敗したら implementation/VAST wave に戻し、完了とは記録
しない。HF upload や public repository withdrawal は、緑の VAST/Scaleway run から推定しない。

## 公開と完了の監査

upload は別許可である。明示的な repository-scoped authorization がある場合だけ、
既存の `publish-one.sh` gate と license/provenance checks を通して行う。dry-run、
packet、owner sign-off、Scaleway の green は upload 権限ではない。

完了を名乗る前に、canonical docs を再読し、次を exact HEAD について記録する。

- live model inventory の全 row が `COMPLETE`、または exact owner-approved disposition
  で accounted され、denominator から黙って消えていない。
- converter/binder/native/CLI、独立 reference、CPU parity、Apple CPU/Metal parity が
  必要な row ごとに evidence と hash で結ばれている。Metal は CPU fallback なし。
- protected manifest の dirty 差分を除き、レビュー済み logical commits、relevant
  static gates、VAST/Scaleway logs、PR/CI state を確認する。
- VAST と Scaleway の不要な instance/storage がなく、HF upload/withdrawal は別承認の
  有無と実施結果が明記されている。

数値や HEAD の更新は canonical ledger と同じ management change で行い、古い測定は
上書きせず supersession note を添える。完了監査に不足があれば、その不足状態を維持する。
