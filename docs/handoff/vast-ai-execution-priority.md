# vast.ai execution priority — large-model jobs (updated 2026-08-18)

**Execution update**: VoxCPM2-2B's VAST download, two-file preparation,
conversion, and structural real-weight verification are complete at `5bc62ae`.
The real Rust structural parity leg also passes after the BOOL metadata test fix
at `e8d016f`; numerical forward parity remains unavailable because the native
weight binder/forward has not landed.
Its publish dry-run is intentionally stopped at `UNKNOWN_REPO` pending the
voice-clone destination/legal decision. It is no longer a block-free publish
job. Instance `47955178` is stopped with both VoxCPM2 and corrected Voxtral
artifacts retained.

**Audience**: owner (yousan). **Purpose**: minimize vast.ai spend + owner
wall-clock, maximize learning curve efficiency, prioritize highest-value
publish across 3 CC-authored handoff docs from WF4 + previous waves.

**2026-08-17 operational override**: all model conversion, real-weight
verification, and upload work that could consume material memory is executed
on **vast.ai**, never on the M1 iMac. Local work is limited to
checkpoint-free preflight, source/doc review, and static tests. This
supersedes the older "local first" alternatives retained below as historical
cost comparisons.

**Related**:
- 総論: [`docs/handoff/vast-ai-large-model-publish.md`](vast-ai-large-model-publish.md) (recipe / provision.sh / lifecycle)
- Job A: [`docs/handoff/vast-ai-publish-voxcpm2-2b.md`](vast-ai-publish-voxcpm2-2b.md)
- Job B: [`docs/handoff/vast-ai-publish-higgs-audio-v3-tts-4b.md`](vast-ai-publish-higgs-audio-v3-tts-4b.md)
- Job C: [`docs/handoff/vast-ai-publish-firered-asr-llm-l.md`](vast-ai-publish-firered-asr-llm-l.md)
- Memory: `[[feedback-large-models-on-vast-ai]]` (vast.ai routing rule) /
  `[[project-huggingface-vokra-publication]]` (5-gate publish) /
  `[[project-restamp-provenance]]` (low-mem re-stamp) /
  `[[feedback-license-signoff-primary-source]]` (§3.1 sign-off rule)

## 0. TL;DR — 推奨実行順

| Order | Job | Rationale (1-sentence) |
|-------|-----|------------------------|
| **1 (conversion done)** | **VoxCPM2-2B** | 888-tensor GGUF conversion and structural verification are complete on VAST. Publish remains fail-closed until the voice-clone destination/legal decision and explicit upload authorization. |
| **2** | **FireRedASR-LLM-L** | Owner の bridge PR (`.pth.tar` → safetensors extraction) 完了後、combined 18-19 GB が M1 iMac 不可 = 定義的 vast.ai、Canary-Qwen precedent 継承の高価値 ASR (Encoder-Adapter-LLM mold の第 2 例、sibling firered_asr_aed_l の道も拓く) |
| **3** | **Higgs-Audio v3 TTS 4B** | Publish は **BosonAI custom R&NC license** で fail-closed default = gate 2 REFUSE、owner の Boson 契約締結 or ☑ Rejected sign-off 判断まで **vast.ai を借りない** ことを推奨 (借りても gate 2 refuse で無駄) |

**Cost total (current policy、Job B skip)**:
- **VoxCPM2-2B + FireRedASR-LLM-L on vast.ai**: ~$0.90-2.00 (normally two sessions; do not rely on a Mac-local conversion to reduce this estimate)
- **Job B pursued (Boson 契約後)**: 追加 ~$0.30-0.75 (別 session、~1-1.5h)

## 1. Per-job rationale — なぜこの順序か

### Priority 1: VoxCPM2-2B (~4.96 GB、apache-2.0)

**この位置に置く理由**:

1. **Conversion complete, publish policy-blocked** — Apache-2.0 redistribution
   permission is signed, but the upstream release explicitly advertises voice
   cloning and the main-repo destination is not yet approved.
2. **Tooling shakedown の最小コスト candidate** — 4.96 GB は最小、~1h + $0.30-0.50、
   provision.sh (Wave 12 idempotent) + run-one.sh (Phase B 自動化 chain) を最初に
   通す対象として最適
3. **M1 safety boundary** — historical streaming/restamp measurements do not
   authorize a new local checkpoint conversion. VoxCPM2-2B is smaller than
   Voxtral, but its conversion / verification / upload remains VAST-only
   under the current policy so the Mac never enters swap-pressure territory.
4. **Sibling precedent 継承** — 既 published `vokra/voxcpm-0.5b` 型を 4x scale-up
   で継承、converter code `crates/vokra-convert/src/models/voxcpm2.rs` の Wave 0
   ADR (Option C hybrid 推奨) の実 upstream 検証にも使える
5. **Publish 成功が §3.1 sign-off / CI flip-the-switch の unblock 材料** —
   `.github/workflows/parity-tts-continuous-vae-real.yml` が既に VoxCPM2-2B
   pinned SHA `bffb3df5a29440629464e5e839f4d214c8714c3d` で待機中、publish で
   `VOKRA_VOXCPM2_GGUF` fixture が有効化される

**Current execution state**:

- Fixed revision main + AudioVAE preparation and GGUF conversion were run on
  VAST only. The generic `run-one.sh` route is invalid for this two-file release;
  use the model-specific runbook. Resume the stopped instance only after a
  destination decision or for independent numerical parity work.

**Owner action 前提** (`docs/handoff/vast-ai-publish-voxcpm2-2b.md` §5, §8):

- Decide whether VoxCPM2 belongs in `vokra-voiceclone-experimental` or has a
  documented exception for the main namespace; ratify the M5-05 legal posture.
- After that, authorize the exact remote upload, add the destination slug's
  sign-off mapping, re-run dry-run, and only then use `--push`.

### Priority 2: FireRedASR-LLM-L (~18-19 GB combined、apache-2.0 + Qwen2 inheritance)

**この位置に置く理由**:

1. **定義的 vast.ai** — asr side 3.38 GiB + Qwen2-7B-Instruct ~15 GB combined =
   ~18-19 GB。**M1 iMac 16GB では絶対不可** (rule: >16 GiB → VAST_AI_REQUIRED)
2. **最高技術価値** — Encoder-Adapter-LLM mold の第 2 例 (sibling `canary_qwen`
   に続く precedent 継承)、以降の Qwen2-based ASR 系 (sibling `firered_asr_aed_l`
   AED variant を含む) 全部の道を拓く
3. **Owner 側 blocker が事前 dependency** — Priority 1 完了までの間、owner が
   bridge PR (`.pth.tar` → safetensors extraction、`nemo_pt_to_safetensors.py`
   precedent mirror) を並行して land できる。Priority 1 が vast.ai に飛ぶ間に
   owner Python 側で bridge PR 準備 = wall-clock 有効活用
4. **License clean** — apache-2.0 primary source 確認済 (§0.1)、Qwen2-7B-Instruct
   inheritance も Apache-2.0-analog (100M active user 未満で商用可) ゆえ
   `vokra.provenance.inherited_license` chunk を刻めば publishable
5. **Priority 1 で shakedown 完了後の応用** — provision.sh gotcha, run-one.sh
   chain, gate 7 auto-bypass の実地経験を持って安心して回せる

**Owner action 事前 dependency** (`docs/handoff/vast-ai-publish-firered-asr-llm-l.md`
§0.4, §4.0, §5):

1. **Bridge PR** — `tools/parity/firered_asr_llm_l/prepare_checkpoint.py` に
   `.pth.tar` extraction subcommand を追加 (owner Python side、precedent =
   `tools/parity/nemo_pt_to_safetensors.py`)
2. **Training data audit** — WenetSpeech (CC-BY 4.0) / AISHELL-1 (Apache 2.0) /
   AISHELL-2 (**非商用注意**) / KeSpeech (MIT) の混成疑義解消、AISHELL-2 が入って
   いれば restrictive 条件が全体伝播
3. **Config.yaml 対応** — HF 側 `config.yaml` が 0 bytes、実 config は GitHub
   `FireRedTeam/FireRedASR` clone から取得する必要 (§0.3)
4. **§3.1 sign-off** — 上記 training data audit がクリアな場合のみ ☑ Commercial、
   Qwen2 inheritance の記載も明示

**推奨 execution**:

- **Prerequisite (owner)**: bridge PR land + training data audit + §3.1 sign-off
- **Step 2a**: vast.ai instance rent (~$0.60-1.50、~2-3h)。RAM ≥ 64 GB、Disk ≥
  100 GB、GPU 最安 (converter は CPU only)
- **Step 2b**: provision.sh (Wave 12) → run-one.sh dry-run。全 gate green 後、
  exact artifact/repo の upload が明示承認された場合だけ `--push`
  (`VOKRA_PUBLISH_ON_VAST=1` は size backstop の bypass であり upload 権限ではない)

### Priority 3: Higgs-Audio v3 TTS 4B (~8.67 GB、BosonAI R&NC — publish blocked)

**この位置に置く理由**:

1. **Publish が upstream license で blocked** — 2026-08-13 の CC primary source
   照合で `LicenseRef-Boson-Higgs-TTS-3-Research-Non-Commercial` (SPDX 未登録
   custom) と判明、`LicenseClass::RedistributionForbidden` = `redistributable() =
   false` = **publish-one.sh gate 2 で REFUSE**、`--allow-noncommercial` (T4)
   でも bypass 不可
2. **借りても gate 2 refuse で無駄** — vast.ai 上で dry-run `run-one.sh` を叩いても
   gate 2 で refuse され、convert 成功しても publish 段で必ず落ちる。owner が
   Boson との commercial redistribution license を締結しない限り publish path
   は closed
3. **CC 側 converter code 既 land** — `crates/vokra-convert/src/models/higgs_audio_v3_tts_4b.rs`
   + `tools/parity/higgs_audio_v3_tts_4b/prepare_checkpoint.py` は既 land ゆえ
   owner が **local で個人利用 (Research/Non-Commercial 範囲、license §II)** の
   GGUF 生成は license 抵触しない ("他人に配る" = §II-A(c)(ii) 抵触 に該当しない)
4. **音色 conditioning capability** — reference audio → zero-shot voice cloning
   能力が upstream にある場合 ELVIS Act 適用の可能性 = `vokra-voiceclone-experimental`
   別リポへの追放判断が必要 (main repo `ayutaz/vokra` に絶対 land しない)

**推奨判断 (優先順、`docs/handoff/vast-ai-publish-higgs-audio-v3-tts-4b.md` §5, §8)**:

- **(推奨) (b) or (c) を選ぶ = vast.ai 借りない**:
  - **(b) Skip publish** — converter code 残置、owner の local convert のみ
    許可 (Research/Non-Commercial 範囲で個人利用)
  - **(c) Downstream guidance** — `docs/license-audit.md` §3.1 Notes 欄に
    「upstream から直接 DL し local で `vokra-cli convert` を叩くのは license
    §II 範囲、Vokra 側は artifact 配布はしない」を document
  - どちらでも **☑ Rejected 2026-XX-XX yousan (RedistributionForbidden per
    LICENSE §II-A(c)(ii))** で `docs/license-audit.md` §3.1 に row 追加、
    以降 publish path は closed
- **(条件付) (a) Boson と交渉** — contact@boson.ai / https://boson.ai で
  commercial redistribution license を締結後、`LicenseClass::from_license_str`
  に private license marker を case-by-case で追加 → publish 可。**この場合のみ**
  vast.ai を Priority 3 として使う (~$0.30-0.75、~1-1.5h)

**Alternative**: 音色 conditioning capability の ELVIS Act 判定を先に行い、
voice-cloning に該当する場合は Vokra scope 外 (`vokra-voiceclone-experimental`
別リポ) として本 handoff から永久除外

## 2. Session 見積 — vast.ai 何セッション必要か

| Path | Sessions | Wall-clock (owner side) |
|------|----------|-------------------------|
| **A: 全部 local (theoretical best)** | 0 | Job 1 local ~30 min、Job 2 不可 (18-19 GB > 16 GB)、Job 3 owner 判断 → not applicable |
| **B: 推奨 (Job 1 local + Job 2 vast.ai + Job 3 skip)** | 1 | Job 1 ~30 min local + Job 2 ~2-3h vast.ai = ~3-3.5h total |
| **C: Job 1 vast.ai + Job 2 vast.ai + Job 3 skip** | 2 (別 session) | Job 1 ~1h + Job 2 ~2-3h = ~3-4h total (spin-up 30 min × 2 追加) |
| **D: Job 1 + Job 2 combined session** | **不可** (Job 2 の bridge PR は事前 dependency、Job 1 と同 session で待てない) |
| **E: Job 3 も追加 (Boson 契約後)** | Path B/C に +1 session (~1-1.5h) | Path B なら total ~4-5h、Path C なら ~4-5.5h |

**Session combine 判定 (重要)**:

- **Job 1 + Job 2 は combine NG** — Job 2 は owner side bridge PR が事前 dependency、
  Job 1 が vast.ai で回っている間に bridge PR を並行 land する path が正解
- **Job 1 + Job 3 は combine 可能だが Job 3 skip 推奨ゆえ moot**
- **Job 2 + Job 3 は combine 可能 (Boson 契約後)** — 両方 vast.ai 定義的、~3-4h
  1 session で completed、~$0.90-2.25

## 3. Cost 見積 — vast.ai spot 4090 rate ~$0.30-0.50/hr baseline

**元データ**: handoff docs の "課金見込" section + user task spec の
`~$0.30/hr` baseline。実際の spot 相場は $0.30-0.50/hr range (2026-08 時点)。

| Job | Wall-clock | Cost range |
|-----|-----------|-----------|
| **Job 1 (VoxCPM2-2B) local** | ~30-60 min | **$0** |
| **Job 1 (VoxCPM2-2B) vast.ai fallback** | ~1h | ~$0.30-0.50 |
| **Job 2 (FireRedASR-LLM-L)** | ~2-3h | ~$0.60-1.50 |
| **Job 3 (Higgs-Audio, Boson 契約後のみ)** | ~1-1.5h | ~$0.30-0.75 |

**Total 見積 by path**:

| Path | Cost total | Value |
|------|-----------|-------|
| **Recommended B (local Job 1 + vast.ai Job 2 + skip Job 3)** | **$0.60-1.50** | 2 モデル publish、最安 |
| **C (vast.ai Job 1 + vast.ai Job 2 + skip Job 3)** | **$0.90-2.00** | 2 モデル publish、shakedown 実地 |
| **B + Job 3 (Boson 契約後)** | **$0.90-2.25** | 3 モデル publish |
| **All-in vast.ai (worst)** | **$1.20-2.75** | 3 モデル publish、Job 3 の要件充足前提 |

**Session spin-up overhead**: vast.ai 各 session の provision.sh (~5-10 min) +
build (`vokra-cli` cargo build --release ~10-15 min) が per-session fixed cost、
combine session の方が cost efficient。ただし Job 1 と Job 2 は combine 不可
(§2 参照)。

## 4. Pre-flight checklist — owner 実行前に確認

### 4.1 全 session 共通

- [ ] **HF token 準備** — 本機 `.env` の `HF=` 値を `HF_TOKEN` として export
      (instance destroy で消えるゆえ新 session ごと必須)
- [ ] **branch 確認** — `main` (or 現行 branch) を vast.ai に clone、scratch 系
      変更は事前 commit + push (`feat/post-audit-cc-gap-2026-08-13` は本 doc の
      branch、Job 実行は main 相当ゆえ merge 後 or main 直接 pull)
- [ ] **provision.sh gotcha 4 件確認** (`docs/handoff/vast-ai-large-model-publish.md`
      §3、memory `[[reference-vast-ai-hf-config-pth-shim]]` +
      `[[reference-huggingface-hub-lt-030-vast-ai]]`):
  - hf_config.pth shim 除去 (`nvidia/cuda:13.0.0` image の malicious mirror
    上書き対策)
  - huggingface_hub < 0.30 pin (xet-token routing 404 回避)
  - certifi CA bundle 再 install
  - stack tool (torch/numpy/safetensors) pre-install
  → provision.sh Wave 12 が全部 idempotent 対応済、1 コマンドで完了

### 4.2 Job 1 (VoxCPM2-2B) 事前 memory review

- `[[project-restamp-provenance]]` — Voxtral 8.7 GB 実績で M1 iMac 16 GB での
  peak footprint 6.4 MB を実測、4.96 GB の VoxCPM2-2B は local 試行の妥当性支持
- `[[project-vokra-cli-sharded-safetensors]]` — sharded safetensors 直渡し不可、
  `MappedSafetensors` 経路が streaming、tools/parity 側の事前 merge 不要
- `[[feedback-license-signoff-primary-source]]` — apache-2.0 primary source 確認
  済、owner 目視のみで ☑ Commercial 可

### 4.3 Job 2 (FireRedASR-LLM-L) 事前 memory review

- `[[feedback-large-models-on-vast-ai]]` — combined 18-19 GB は VAST_AI_REQUIRED、
  M1 不可
- `[[reference-safetensors-shared-tensor-dedup]]` — bridge 実装時の dedup logic
  流用
- **precedent bridge**: `tools/parity/nemo_pt_to_safetensors.py` の tarball
  extraction pattern (M4-20 T17 DFN3 Phase B)

### 4.4 Job 3 (Higgs-Audio) 事前 memory review

- `[[feedback-license-signoff-primary-source]]` — primary source が
  RedistributionForbidden で fail-closed default = ☑ Rejected sign-off が正解
- Voice-cloning capability の場合は `vokra-voiceclone-experimental` 別リポへの
  追放判断 (ELVIS Act、`docs/legal-compliance.md`)

### 4.5 vast.ai instance 推奨 spec

| Job | Image | RAM | Disk | GPU | 課金見込 |
|-----|-------|-----|------|-----|---------|
| Job 1 (VoxCPM2-2B) | `pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel` | ≥32 GB | ≥80 GB | 最安 | ~$0.30-0.50 (1h) |
| Job 2 (FireRedASR-LLM-L) | 同上 | ≥64 GB | ≥100 GB | 最安 | ~$0.60-1.50 (2-3h) |
| Job 3 (Higgs-Audio、契約後) | 同上 | ≥32 GB | ≥80 GB | 最安 | ~$0.30-0.75 (1-1.5h) |

**共通推奨**: ネットワーク非従量課金 or inclusive band (out-bound 数十 GB) /
`VOKRA_PUBLISH_ON_VAST=1` marker が provision.sh で自動 set → gate 7 auto-bypass

## 5. Decision tree — owner 判断のポイント

```
Job 1 (VoxCPM2-2B):
├── local 試行 ─┬── success ── publish 完了 (cost $0)
│               └── OOM fail ── vast.ai fallback (cost ~$0.50)
│
Job 2 (FireRedASR-LLM-L):
├── bridge PR ─┬── land 済 ── training data audit ─┬── AISHELL-2 なし ── ☑ Commercial → vast.ai (cost ~$1.50)
│               │                                    └── AISHELL-2 あり ── owner が restrictive 条件で全体判定
│               └── 未 land ── owner Python 側で PR land、Job 1 vast.ai と並行で wall-clock 有効活用
│
Job 3 (Higgs-Audio v3 TTS 4B):
├── (b) or (c) skip publish 判断 (推奨) ── ☑ Rejected sign-off、converter 残置
├── (a) Boson 契約 (低確率) ── vast.ai (cost ~$0.75) + LicenseClass override PR
└── 音色 conditioning 判定 ── voice-cloning 該当なら vokra-voiceclone-experimental 別リポへ
```

## 6. なぜこの順序か — 3 alternative 順序との比較

| 順序 | Pros | Cons | 判定 |
|------|------|------|------|
| **推奨 (voxcpm2 → firered → higgs)** | Ready-first で tooling shakedown 最小コスト、blocker 有無を尊重 (voxcpm2 = block-free、firered = owner action で unblock 可能、higgs = 恒久 blocked) | Job 1 が local success なら vast.ai shakedown 経験が Job 2 まで持ち越し (問題は Job 2 の complexity で shakedown 効果は限定的) | **✓ 選択** |
| **Cheapest first (voxcpm2 → higgs → firered)** | Higgs のコストが安い順 | Higgs は publish 不可 = 無駄セッション、firered を後回しにすると owner bridge PR 期間が短くなる | ✗ Higgs blocker 無視 |
| **Most-blocked first (firered → higgs → voxcpm2)** | Blocker 解消の owner 時間確保 | firered は bridge PR まで vast.ai 起動不可、firered → higgs は blocker 深度が違う | ✗ 逆順 |
| **Job 2+3 combined vast.ai session** | Session spin-up 節約 | Higgs が publish 不可のまま + firered の bridge PR 完了時期不明 = 待機コスト | ✗ blocker 待ち |

## 7. Non-goals — 本 doc の scope 外

- **Runtime forward 実装** (Job 1/2/3 いずれも `crates/vokra-models/src/<slug>/`
  native forward は future wave、publish は converter code + GGUF 変換のみが
  対象)
- **Parity CI flip-the-switch** (publish 完了後の `VOKRA_<MODEL>_ENABLE=1`
  variable set + fixture GGUF pointer 設定は owner separate action)
- **License 交渉** (Job 3 の Boson 契約は owner のみ、CC は情報提供のみ)
- **音色 cloning 判定** (Job 3 の ELVIS Act 適用性は owner 法務判断)
- **Number fabrication** — 本 doc の cost / size / wall-clock は全て handoff
  docs 記載値 + user task spec 指定 rate から derive、fabricate なし

## 8. Cross-reference — 3 handoff docs に本 doc への参照を追加

CC 側の commit で以下 3 doc の "関連" (or "See also") section に本 doc への
cross-reference を追加済:

- `docs/handoff/vast-ai-publish-voxcpm2-2b.md` "関連" section に
  `Priority ordering: docs/handoff/vast-ai-execution-priority.md`
- `docs/handoff/vast-ai-publish-higgs-audio-v3-tts-4b.md` "関連" section に同上
- `docs/handoff/vast-ai-publish-firered-asr-llm-l.md` "関連" section に同上

## 関連

- 総論: `docs/handoff/vast-ai-large-model-publish.md`
- Job A: `docs/handoff/vast-ai-publish-voxcpm2-2b.md`
- Job B: `docs/handoff/vast-ai-publish-higgs-audio-v3-tts-4b.md`
- Job C: `docs/handoff/vast-ai-publish-firered-asr-llm-l.md`
- Sibling published: `huggingface.co/vokra/voxcpm-0.5b`,
  `huggingface.co/vokra/nemotron-3.5-asr-streaming-0.6b`,
  `huggingface.co/vokra/xcodec2` (T4 precedent)
- Memory: `[[feedback-large-models-on-vast-ai]]` /
  `[[project-huggingface-vokra-publication]]` /
  `[[project-restamp-provenance]]` /
  `[[feedback-license-signoff-primary-source]]` /
  `[[reference-vast-ai-hf-config-pth-shim]]` /
  `[[reference-huggingface-hub-lt-030-vast-ai]]` /
  `[[project-vokra-cli-sharded-safetensors]]` /
  `[[reference-safetensors-shared-tensor-dedup]]`
