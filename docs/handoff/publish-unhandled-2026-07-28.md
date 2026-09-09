# HuggingFace `vokra` org — 未対応モデル一覧 (2026-07-28、residual wave 3 = 2026-07-30 refresh)

**Purpose**: 公開 (`huggingface.co/vokra`) に未対応のモデルを owner-visible な一覧にして
後続の対応判断を可能にする。依頼者指示 (2026-07-28)「未対応のものは未対応リストに入れて
後から対応するように」に基づく。

> **2026-08-30 supersession boundary:** This is a dated 2026-07-28/30
> backlog snapshot, not the current publication inventory. Reconcile live rows
> with `docs/license-audit.md` and `scripts/publish/check-catalog-reality.sh`.
> For resource handling, the current rule is VAST when the model-plus-shard
> total is >=2 GB; exact-size-unknown artefacts are not `local-safe`. The old
> “local M1” or “8 GB cutoff” wording below is historical and does not authorize
> local model processing or an upload.

**現在の公開状況**: **51 モデル live** (2026-07-28 時点)。詳細一覧は `docs/license-audit.md`
§3.1 sign-off 表を参照。

**Residual wave 3 (2026-07-30) 更新**: F0 pitch extractor 3 種 (RMVPE / FCPE / CREPE) +
FSMN-VAD + TitaNet-L + VoxCPM2-2B + StyleTTS 2 scaffold + Canary-Qwen-2.5B + omniASR variants +
Qwen3-TTS 1.7B + Charsiu 実 wav2vec2 CTC forward + JA-ASR エンコーダ / デコーダ op 3 種を
CC-side で land。converter は揃ったが publish 自体は owner の VAST 起動
(モデル本体と全 shard の合計が 2 GB 以上、またはサイズ未確認の artefact) 待ちである。
既知の 2 GB 未満はサイズ分類上 `size-safe` とできるが、現行 maintainer policy では
ローカル model processing を行わない。詳細な wave 3 の commit table は
`docs/handoff/residual-wave3-2026-07-30.md`。

**未対応 = 6 tier に分類** (blocker と復活条件を明記して future action を判断可能に):

---

## A. Converter 未実装 / 拡張が必要 (owner 判断 = 実装 GO/NO-GO)

CC で converter 側実装を追加すれば publish 可能な tier。owner の scope 判断待ち。

| Model | Upstream | §3.1 sign-off | Blocker | 復活条件 |
|-------|----------|---------------|---------|----------|
| **WavTokenizer** | (未確定) | 未取得 | converter 未実装 | M5-07 sign-off + converter 新規実装 (spec `docs/superpowers/specs/2026-07-28-wavtokenizer-design.md` 塩漬け) |
| **Matcha-TTS** | (未確定) | 未取得 | converter 未実装 | M5-07 sign-off + converter 新規実装 (spec `docs/superpowers/specs/2026-07-28-matcha-tts-design.md` は 見送り = 塩漬けのまま再開放しない) |
| **Bark** | `suno/bark` | 未取得 | converter 未実装、Suno の voice-cloning 再学習禁止方針 | M5-07 sign-off (v2.0+ 検討) + converter 新規実装 |
| **Voxtral-Small-24B** | `mistralai/Voxtral-Small-24B-2507` | ☑ Commercial (2026-07-23) | ~~converter 未実装~~ **✅ CC-side complete** (Wave 1 A-3, `12efb13` 2026-07-29 — `convert_voxtral_file_streaming` = header-only mmap + 1-tensor-at-a-time streaming implemented, M1 iMac 16GB RAM can convert 48GB BF16 with `max(shard_header) + max(tensor_payload)` footprint). Remaining = **owner publish action only** (K-quant path is widen-then-quantize in-memory ゆえ streaming 非対応 = pass-through BF16 のみ) | Owner: (a) local M1 で streaming path 実測 + `publish-one.sh --push`、または (b) vast.ai instance 経路 (`docs/handoff/vast-ai-large-model-publish.md` §2) の 2 択 |

---

## B. Upstream / license 未解決 = fail-closed で defer (owner action 待ち)

CC の primary-source rule に従い sign-off できない tier。owner slug 確認や distributor
権限判断が必要。

| Model | Upstream | Blocker | 復活条件 |
|-------|----------|---------|----------|
| **Speaker3d (ERes2Net)** | `iic/speech_eres2net_sv_zh-cn_16k-common` | 上流 HF slug 404、代替 `bandad/eres2netv2_pretrained` は license 未宣言 (`cardData.license = ?`, tags = `region:us` のみ) | owner が (a) 有効な HF slug を再確認 or (b) ModelScope 側にしか無いことを明示的に accept して distributor 権限確認 |

---

## C. Owner が明示的に Rejected 判定済 (方針変更前提)

`docs/license-audit.md` §3.1 で ☑ Rejected が sign 済み。方針変更しない限り復活しない。

| Model | Upstream | §3.1 Rejected 日 | Rejected 理由 | 復活条件 |
|-------|----------|------------------|---------------|----------|
| **DebertaV2** | `ku-nlp/deberta-v2-large-japanese-char-wwm` | 2026-07-27 | cc-by-sa-4.0 = T3 Copyleft ShareAlike の SA obligation を Vokra 配布に取り込む決定を保留 | owner が Commercial 変更 |
| **VibeVoice-Large-7B** | `microsoft/VibeVoice-Large` | 2026-07-28 | Microsoft 側で weight repo が 404 化 (withdrawn) = 上流に無いものを配布しない | Microsoft が re-publish (見込みなし) |

---

## D. T4 非商用 (依頼者判断待ち)

Workflow は X-Codec-2 (2026-07-28、初 T4 precedent、`--allow-noncommercial` flag 経路)
で確立済。owner が「T4 tier を追加公開する」判断次第で publish 可能。

| Model | Upstream | License | 復活条件 |
|-------|----------|---------|----------|
| **F5-TTS** | `SWivid/F5-TTS` | CC-BY-NC-4.0 | owner GO |
| **Fish-Speech v1.4-1.5** | `fishaudio/fish-speech-*` | CC-BY-NC-SA-4.0 | owner GO |
| **EnCodec** (Meta) | `facebook/encodec_*` | CC-BY-NC-4.0 (weight) | owner GO (現 `check-encodec-exclusion.sh` gate は Vokra converter tree の混入を防ぐが、`vokra/encodec` として T4 公開すること自体は独立判断) |
| **Style-Bert-VITS2 v2 のその他 SKU** (JP-Extra base 以外) | `litagin/*` family | 各 SKU 個別 | 現状 `sbv2-v2-jp-extra-base` (agpl-3.0) のみ公開。他 SKU は license 個別確認 |

---

## E. 契約禁止で配布不可 (`vokra-voiceclone-experimental` 分離扱い)

公式 `vokra` org には出さない。分離設計の現行境界は `docs/legal-compliance.md`
§§3–4 と `docs/system-requirements.md` FR-CP-04 に従う。

- **VOICEVOX 系** — 商用契約禁止
- **CSJ / JSUT・JVS 学習物** — corpus 規約が weight 再配布を明示禁止
- **VitsJa (plain VITS Japanese)** (`espnet/kan-bayashi_jsut_vits`) — Apache-2.0 code /
  **JSUT weight 再配布禁止** (`sites.google.com/site/shinnosuketakamichi/publication/jsut`
  "Re-distribution is not permitted")。converter は tool として実装済、weight は owner が
  自前調達して自環境で変換する運用のみ可。distributor に配布権利なし = **override 不可**、
  CC 判断でも復活せず。復活条件 = 別 corpus で学習した ESPnet VITS 派生 weight
  (permissive) の入手。
- **RVC v2 / GPT-SoVITS training data** — voice cloning tier、ELVIS Act 対応で別リポへ

これらは復活しない (方針)。

---

## F. Wave 3 (2026-07-30) land — converter ready、publish は owner 実行待ち

Residual wave 3 で CC-side 実装が揃った分。converter + native forward + `vokra-cli`
dispatch まで land 済ゆえ、owner が `publish-one.sh --push` を回すだけで `huggingface.co/vokra`
公開可能な tier。§3.1 sign-off 済 (yousan) or 一次資料 clean で fail-closed default 例外の
CC 判断で sign 済 (memory `feedback-license-signoff-primary-source`)。詳細な per-commit
context は `docs/handoff/residual-wave3-2026-07-30.md` per-wave table。

| Model | Upstream | License | §3.1 sign-off | Blocker | 復活条件 |
|-------|----------|---------|---------------|---------|----------|
| **VoxCPM2 2B multilingual** | `openbmb/VoxCPM2` | Apache-2.0 | ☑ Commercial (2026-07-28) | converter 拡張 (2B path) は wave 3 (`e369dde`) で完了。実 checkpoint fetch + 30-lang list 再照合が owner 側 | vast.ai or local M1 publish (~4 GB BF16 ボーダー) |
| **TitaNet-L** | NVIDIA NeMo (未確定 slug) | CC-BY-4.0 (NOTICE §7 attribution 済) | ☑ Commercial (2026-07-30 yousan、依頼者許可 = CC 判断) | 大サイズは vast.ai (`docs/handoff/vast-ai-large-model-publish.md` §1)、`titanet_speaker_encode` op landing は M5-ORPHAN-SCOPE T04 (別 wave) | owner が `publish-one.sh --push` |
| **RMVPE (F0)** | `yxlllc/RMVPE` architecture derived from `Dream-High/RMVPE` | code Apache-2.0 / weight Unknown | **2026-08-18監査訂正: sign-off撤回・空欄** | 旧「両upstream MIT」は誤り。`yxlllc` release checkpoint に明示的なgrantがないためconverterはfail-closed `unknown`。実forward自体は後続waveでland済み | checkpoint権利者の明示的な再配布grant取得 → §3.1 sign-off → vast.ai parity |
| **FCPE (F0)** | Conformer-based F0 (upstream MIT) | MIT | (§3 model zoo table に追加、§3.1 は owner 記入待ち) | `parity-fcpe-real.yml` 未 land (follow-up wave) | owner sign-off (fail-closed) + `publish-one.sh --push` |
| **CREPE (F0)** | `marl/crepe` (5 サイズ = tiny/small/medium/large/full 全て MIT) | MIT | ☑ Commercial (2026-07-30 yousan、依頼者許可 = CC 判断、`tools/parity/keras_h5_to_safetensors.py` = Keras .h5 export layer 追加) | サイズ選択 (5 サイズ全部? tiny + full のみ?) は owner 判断 | owner が サイズ選択 + `publish-one.sh --push` |
| **Silero v6.2.1** | `snakers4/silero-vad` | MIT | (既存 Silero 行 = signed。v6 upgrade は同 license row 継承) | `parity-silero-real.yml` matrix に v6 variant 追加 (follow-up)。v5 は既 published (`vokra/silero-vad-v5`) と共存 = 新 slug 候補 `vokra/silero-vad-v6-2-1` | owner が cron matrix 更新 + `publish-one.sh --push` |
| **FSMN-VAD** | `alibaba-damo-academy/FunASR` FSMN-VAD (MIT) | MIT | 未取得 (§3.1 追加待ち = fail-closed default) | 新 slug `vokra/fsmn-vad` = FR-OP-42 相当の VAD 代替候補、Silero と並列運用 | owner §3.1 追加 + `publish-one.sh --push` |
| **Canary-Qwen-2.5B** | `nvidia/canary-qwen-2.5b` (FastConformer + Voxtral-style Qwen decoder、`0a45ec3` cluster 内) | CC-BY-4.0 (Canary 系継承見込) | 未取得 (§3.1 追加待ち) | ~4.96 GB → local M1 borderline / vast.ai 推奨。**Note**: 別 Canary 1B v2 は wave 2 で publish 済 (`vokra/canary-1b-v2`) | owner §3.1 追加 + vast.ai or local BORDERLINE publish |
| **omniASR-CTC 300M** | `facebook/omniASR-CTC-300M` (`6b0effc` cluster) | Apache-2.0 (omniASR family 継承見込) | 未取得 (§3.1 追加待ち = family-signoff cascade 候補) | 1B は既 published (`vokra/omniasr-ctc-1b`)、300M は同一 family。local M1 で余裕 | owner §3.1 追加 + local publish |
| **omniASR-CTC 7B** | `facebook/omniASR-CTC-7B` (`6b0effc` cluster) | Apache-2.0 (同上) | 未取得 (§3.1 追加待ち) | ~14 GB BF16 → vast.ai 必須 (`docs/handoff/vast-ai-large-model-publish.md` §1) | owner §3.1 追加 + vast.ai publish |
| **Qwen3-TTS 1.7B** | Qwen3-TTS 1.7B fork of 0.6B (`6b0effc` first commit、hidden-size fork) | Apache-2.0 (Qwen family 継承見込) | 未取得 (§3.1 追加待ち = 0.6B row からの family cascade 候補) | 0.6B は既 published (`vokra/qwen3-tts-0.6b`)、1.7B は同一 hidden-size fork | owner §3.1 追加 + local publish |
| **StyleTTS 2** | `yl4579/StyleTTS2` (`735fe9d` = 上記 §3.1 row 260 の scaffold 部分) | code MIT / weight = voice-consent 条件付き usage agreement | ☑ Rejected (2026-07-23 yousan、既存判定) — weight にライセンス欄が存在せず条件不明 = 配布しない (fail-closed) | scaffold のみ (weight 消費なし) はコード配布 OK だが weight-consuming builds は `Unknown` fail-closed | owner が weight license 条件を accept する場合のみ (現状 Rejected 維持) |
| **Charsiu forced-alignment** | `charsiu/charsiu` (real wav2vec2 CTC forward = `3ef8f57`) | MIT (charsiu 系継承見込) | 未取得 (§3.1 追加待ち) | real-checkpoint alignment parity run 未実施 = spec 起票待ち | owner §3.1 追加 + `parity-charsiu-real.yml` 起票 (CC 別 wave) + `publish-one.sh --push` |

### Voice-clone 4 model (openvoice_v2 / knn_vc / freevc / meanvc)

上記 F tier に**入れない** — `docs/m5-owner-verification-checklist.md` §6.9、
`docs/legal-compliance.md` §§3–4、`docs/system-requirements.md` FR-CP-04 に従い
`vokra-voiceclone-experimental` 別リポ (未作成) 送り。ここに公式配布行は
永続的に載せない。converter wire は wave 3 (`5f7cb15`) で landed だが、これは main repo 上
での codegen 完全化のためで、`ayutaz/vokra` publish 対象ではない。

---

## 更新規律

- 新規 publish 完了 → 該当行を削除 + `docs/license-audit.md` §3.1 に signed row 追加
- 新規未対応の発覚 → 該当 tier に追加
- Owner 判断が変わった → tier 間移動 (例: C の Rejected → D の T4 GO / A の 実装 GO)
- 本ファイルは gitignore-free tracked = 公開時に owner から見える。関連 memory =
  `project-huggingface-vokra-publication` の「未対応リスト」節と対称に保つ (memory 側は
  gitignore-local ゆえ内部規律の SoT、tracked 側は公開時可視 handoff)。
