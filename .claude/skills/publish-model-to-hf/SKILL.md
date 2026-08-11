---
name: publish-model-to-hf
description: Vokra の GGUF を huggingface.co/vokra 配下に公式配布するときに使う。5-tier gate（catalog-reality / redistributable / provenance / §3.1 sign-off / allow-noncommercial）+ publish-one.sh / fetch_license.sh / upload.sh の chain + T4 (Research-only) precedent + restamp_provenance の低メモリ再刻印 + HF vocabulary normalize を示す。**手動 upload は禁止**、必ず publish-one.sh 経由。
---

# GGUF を huggingface.co/vokra に公開する

**単一事実源**: `scripts/publish/publish-one.sh` のスクリプト冒頭コメント。本 skill はそれを skill 表現に翻訳したもの。実装との drift は script が SoT で本 skill を追随する。

**大原則**: `hf` CLI や `huggingface_hub.upload_file` を CC が直接叩かない。**必ず `publish-one.sh` chain 経由**。理由 = 5 段 gate を bypass する経路が存在すると gate は fail-closed default を保てない。

## 0. 事前判断（着手前）

- **ライセンス audit を先に通す** → skill `license-audit`。§3.1 sign-off が空欄なら **publish しない**（gate 4 で fail-closed refuse される）。primary-source rule で埋められる条件が揃ってから戻る。
- **モデルサイズ確認**: safetensors 合計 >8 GB は M1 iMac で処理しない → skill `vast-ai-workflow`。`publish-one.sh` の gate 7 が `--allow-large` or `VOKRA_PUBLISH_ON_VAST=1` なしで refuse する。[[feedback-large-models-on-vast-ai]]。
- **既に GGUF がある場合**: 変換不要なら `restamp_provenance`（§7）で provenance だけ差替可能。

## 1. 5 段 gate の全体像

`publish-one.sh` が chain するのは以下 5 段。**全 gate を通ったモデルだけが `--push` で live 化**する。

| # | Gate | 実装 | fail-closed 条件 |
|---|------|------|----------------|
| 1 | **catalog-reality** | `check-catalog-reality.sh` | `docs/license-audit.md` §3 で `★ 公式 zoo` 宣言があるのに実装（runtime/op/converter のどれも）が無い |
| 2 | **redistributable** | `LicenseClass::redistributable()` + `make_model_card.py` | `RedistributionForbidden`（VOICEVOX/CSJ/JSUT・JVS）を publish しようとした |
| 3 | **provenance stamp** | `upload.sh` refuse | GGUF に `vokra.provenance.*` が刻まれていない（converter を通っていない、または schema stamp 前の旧成果物） |
| 4 | **§3.1 sign-off** | `signoff_match.py` | audit table の該当 row が blank（**空欄 = 「まだ誰も判断してない」= "no" ではない が publish は不可**） |
| 5 | **allow-noncommercial** | `publish-one.sh` refuse | T4（Research-only）weight を `--allow-noncommercial` 明示なしで publish しようとした |

加えて **Copyleft は gate 6a-6e**（AGPL / CC-BY-SA 等）で `--acknowledge-copyleft` + LICENSE + NOTICE + AGPL は SOURCE.md + card `license:` 一致を追加要求。

## 2. Tier の判定

| Tier | 例 | LicenseClass | 追加 flag | precedent |
|------|-----|-------------|-----------|-----------|
| T1 Commercial | whisper-large-v3 (MIT) / kokoro (Apache-2.0) | `Permissive` | なし | 大半 |
| T2 CC-BY attribution | mimi (CC-BY 4.0) | `AttributionRequired` | NOTICE 同梱必須 | mimi (kyutai) |
| T3 Copyleft | Style-Bert-VITS2 v2 (AGPL-3.0) / cc-by-sa-* | `Copyleft` | `--acknowledge-copyleft` + LICENSE + NOTICE + (AGPL は SOURCE.md) | SBV2 v2 |
| T4 Research-only | X-Codec-2 (cc-by-nc-4.0) / F5-TTS / Fish-Speech | `NonCommercial` | `--allow-noncommercial` 明示 | **X-Codec-2 が初 precedent (2026-07-28)** |
| T5 Rejected | VOICEVOX weight / VibeVoice-Large (404) | `RedistributionForbidden` | 公開不可 | VibeVoice fail-closed 実例 |

**T4 の workflow は X-Codec-2 precedent（[[project-x-codec2-t4-precedent]]）を踏襲**: `fetch_license.sh --spdx cc-by-nc-4.0` で canonical LICENSE 実文書を同梱、`publish-one.sh --allow-noncommercial` で明示、card は `hf_license_tag()` で `other` に normalize（HF は cc-by-nc-* を独立 tag として受けないケースあり）。

## 3. HF vocabulary の normalize（SPDX と別空間）

- HF `license:` タグは lower-case（`MIT` → 400 reject、`mit` OK）+ dual 表現は `other`（SPDX の `MIT OR Apache-2.0` は HF では `other`）。
- 正規化は `make_model_card.py` の `hf_license_tag()` に集約されている。CC が手で card front-matter を書かない — 常に generator 経由。
- CPML は SPDX 未登録 → converter で `NonCommercial` に hard-map（→ skill `license-audit`）+ publish は T4 workflow。[[reference-cpml-spdx-nonregistration]]。

## 4. §3.1 sign-off を埋める

skill `license-audit` の primary-source rule に従う。CC 側で埋めていい条件（**両方**）:

1. 依頼者が明示的に「自主判断で埋めてよい」と言った
2. primary source で clean 確認（authenticated HF API meta / upstream LICENSE raw / DOI）

**署名 = `yousan`** + 日付 + `(依頼者許可 = CC 判断)`。片方でも欠けたら空欄据置 = fail-closed 継続。埋めた row は `signoff_match.py --self-test` で機械検証（field 数 8 が正、pipe `|` が row 本文に入ると 9 になり parse 崩壊 → 過去 latent bug、row 318 で修正済）。

## 5. publish-one.sh の呼び方

```bash
# .env から HF_TOKEN を明示 source（デフォルトの環境継承では拾わない場合あり）
export HF=$(grep '^HF=' .env | cut -d= -f2-) && export HF_TOKEN="$HF"

# T1 (Permissive) 例: whisper-large-v3
scripts/publish/publish-one.sh \
  --gguf ~/scratch/whisper-large-v3.gguf \
  --repo vokra/whisper-large-v3 \
  --license-spdx mit \
  --push  # ← --push 無しは常時 dry-run。省略で staging のみ

# T4 (Research-only) 例: X-Codec-2
scripts/publish/publish-one.sh \
  --gguf ~/scratch/xcodec2.gguf \
  --repo vokra/xcodec2 \
  --license-url https://raw.githubusercontent.com/.../LICENSE \
  --allow-noncommercial \
  --push

# T3 (Copyleft AGPL) 例: SBV2 v2 base
scripts/publish/publish-one.sh \
  --gguf ~/scratch/sbv2-v2-base.gguf \
  --repo vokra/style-bert-vits2-v2-base \
  --license-spdx agpl-3.0 \
  --acknowledge-copyleft \
  --push
```

**dry-run default** = `--push` を明示しない限り stage のみ（gate 全通過を local で確認可能）。**publish は irreversible** = 一度 live 化した weight は minutes で mirror される、「あとで消せる」は復旧計画にならない。

## 6. `fetch_license.sh` の使い分け

upstream の LICENSE 実体をどこから取るかは、上流の配布形態で決まる:

- **上流が LICENSE ファイルを ship している**: `--url <raw-github-url>` で fetch（**その copyright 行が retention 要件を満たすので、canonical に置換してはいけない**）
- **上流が front-matter だけで LICENSE ファイルなし**: `--spdx <apache-2.0|mit|cc-by-4.0|cc-by-sa-4.0|cc-by-nc-4.0|agpl-3.0|...>` で canonical に取る
- **canonical URL がない** (openmdw-1.1 等): `inline_license_text()` で inline shipped（現状 openmdw-1.1 のみ、追加する時は明示 audit trail を付ける）

## 7. `restamp_provenance` — 低メモリ再刻印（tensor コピーなし）

**用途**: 既存 GGUF に provenance schema を後付け、または license 表記を差替。**tensor は mmap 読取して byte-copy せず、metadata だけ差替える**。

- 実測: **8.7 GB Voxtral を M1 16 GB で peak footprint 6.4 MB / RSS 5.4 GB (mmap pages) / swap 0** で再刻印可（[[project-restamp-provenance]]）。
- 使い所: (a) 旧成果物に schema stamp（`vokra.schema.version` / `vokra.schema.producer`）を追加、(b) 依頼者判断で license class 変更、(c) 上流 fork 変更に伴う `vokra.provenance.upstream_hf` 更新。
- **converter を再度回す必要はない** — provenance だけの差替なら restamp が正解、変換は tensor 全読みで RSS が跳ねる。
- 実装は `crates/vokra-convert/` の `restamp_provenance`（`GgufStreamWriter` 経由）。Python 単発 script でも書けるが `hf_license_tag()` normalize と ordering を統一するために crate 経由推奨。

## 8. HF_TOKEN の扱い

- **CLI 引数で渡さない**（shell history + `ps` output に残る）
- `.env` に `HF=hf_xxx` で保存（`.gitignore` 済 = 履歴に載らない、`git check-ignore .env` で確認可）
- Session で使う時は `export HF=$(grep '^HF=' .env | cut -d= -f2-) && export HF_TOKEN="$HF"` で明示 source（環境継承だけでは拾わないケースあり）
- 期限切れ or 403 なら **fresh token を新規発行**（既存を回転、CSM-1B publish で使った precedent）

## 9. Precedent 集（判断の記憶）

- **X-Codec-2 T4 初 precedent** (2026-07-28): cc-by-nc-4.0、`--allow-noncommercial`、`fetch_license --spdx cc-by-nc-4.0`。[[project-x-codec2-t4-precedent]]
- **CSM-1B ☑ Commercial** (2026-07-28): 依頼者許可 = CC 判断、authenticated HF API で `license=apache-2.0`
- **VibeVoice-Large Rejected** (2026-07-28): HF API 404 = microsoft withdraw = mirror が幾つあっても Rejected（fail-closed 正常）
- **Bark row 259**: README 「for research purposes」は Apache-2.0 下で非拘束 advisory 前例（以降類似モデルで cite）
- **§3.1 row 318 field-count fix**: 行本文の pipe `|` が `line.split("|")` を壊す latent bug（8→9 fields）。row を書く時は pipe を含まない言い回しに reword。
- **Voxtral 8.7 GB を M1 で publish 成功**: restamp_provenance 経路で peak 6.4 MB（vast.ai 不要になった実例）

## 10. Verify

publish 前後で以下を通す:

```bash
scripts/publish/check-catalog-reality.sh
python3 scripts/publish/signoff_match.py --self-test
scripts/publish/publishability-report.py  # 各モデルの 5-tier 現況
```

`upload.sh --self-test` は script 内 test（LICENSE 未同梱 / §3.1 blank / provenance 未刻印 の refuse path）を回す。

## 11. 出禁パターン（**やってはいけない**）

- **`hf` CLI や `huggingface_hub.upload_folder` を CC が直接叩く**: gate を全て bypass、依頼者にも見えない unaudited publish になる
- **§3.1 blank row を CC が勝手に埋める**: primary-source rule 違反、fail-closed default を破壊
- **T4 weight を `--allow-noncommercial` なしで publish**: gate 5 が refuse するが、gate を bypass すると license 違反配布
- **canonical LICENSE を canonical URL 側で置換**: 上流が独自 copyright を付けている場合、retention 要件違反
- **`--push` を最初から付ける**: dry-run で gate 全通過を確認しない = pipe 経由の undetected error を live 化する
