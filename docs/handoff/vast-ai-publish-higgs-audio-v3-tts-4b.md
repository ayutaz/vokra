# vast.ai publish runbook — Higgs-Audio v3 TTS 4B (bosonai/higgs-audio-v3-tts-4b)

**Owner-triggered.** CC は本 doc 作成のみ。実 vast.ai instance の起動・convert・
publish は owner が本 runbook を追いながら実行する。

**Related**:
- 本 runbook は `docs/handoff/vast-ai-large-model-publish.md`（総論 = §2 recipe /
  §3 provision.sh gotcha / §4 lifecycle）を **前提** とする。共通手順は総論を参照
  し、本 doc は Higgs-Audio v3 TTS 4B に固有の差分のみを記述する。
- Converter code: `crates/vokra-convert/src/models/higgs_audio_v3_tts_4b.rs`
  (Wave B fast-track land, coverage-audit-2026-08-03)。
- Sidecar: `tools/parity/higgs_audio_v3_tts_4b/prepare_checkpoint.py` (uv-managed
  Python 3.12)。
- Sibling published (別 tier): `huggingface.co/vokra/xcodec2` (T4 Research-only、
  cc-by-nc-4.0、2026-07-28 land) — 本モデルは T4 でも通らない可能性が高い、§0 参照。

## 0. Primary source correction — 本 doc land 時点 (2026-08-13) の重大な発見

**audit ticket / converter code / task 指示は apache-2.0 前提だったが、primary
source 照合の結果 upstream license は apache-2.0 ではなく BosonAI custom
Research-and-Non-Commercial license と判明**した。詳細:

### 0.1 HF cardData 照合結果 (2026-08-13、CC 直接 curl 確認)

```bash
curl -sL https://huggingface.co/api/models/bosonai/higgs-audio-v3-tts-4b | \
  python3 -c "import json,sys; d=json.load(sys.stdin); cd=d.get('cardData',{}); \
    print('license:', cd.get('license')); \
    print('license_name:', cd.get('license_name')); \
    print('license_link:', cd.get('license_link')); \
    print('gated:', d.get('gated'))"
```

出力:
```
license: other
license_name: boson-higgs-tts-3-research-and-non-commercial-license
license_link: LICENSE
gated: False
```

### 0.2 LICENSE 実文書 (2026-08-13、CC 直接 fetch)

```bash
curl -sL https://huggingface.co/bosonai/higgs-audio-v3-tts-4b/raw/main/LICENSE | head -60
```

タイトル: **"BOSON HIGGS TTS 3 RESEARCH AND NON-COMMERCIAL LICENSE AGREEMENT"** (Last
Updated: July 8, 2026)。

- §I INTRODUCTION 末尾: *"this Agreement is a research and non-commercial
  source-available model license and is **not an open source license**"*
- §II RESEARCH AND NON-COMMERCIAL USE LICENSE: 研究 / 非商用のみ
- §II-A CREATOR USE GRANT: Digital Creator (podcast / video / audiobook 等の作成者)
  が生成音声を含む creative content を publish + monetize することは attribution
  義務を satisfy すれば可
- §II-A(c) **What this grant does not cover** (Creator Use Grant の除外):
  - **(i) hosting, serving, or otherwise making the Higgs Materials available to
    others, whether via an API, SaaS offering, plug-in, hosted tool, or end-user
    application**
  - **(ii) redistributing, reselling, sublicensing, or otherwise distributing the
    Higgs Materials or any Derivative Work, or fine-tuning, distilling, or
    otherwise creating a Derivative Work for resale or distribution**
  - **(iii) embedding or integrating the Higgs Materials in a product,
    application, or service made available to third parties (including a text-
    to-speech, voice-generation, dubbing, or content-creation product or
    service)**

### 0.3 Vokra LicenseClass 判定

Vokra が GGUF を `huggingface.co/vokra/<name>` に上げる行為は §II-A(c)(ii) の
**"redistributing... the Higgs Materials or any Derivative Work"** に該当する。
したがって:

| 属性 | 値 |
|---|---|
| SPDX (Vokra 判定) | `LicenseRef-Boson-Higgs-TTS-3-Research-Non-Commercial` (custom、SPDX 未登録) |
| LicenseClass | **`RedistributionForbidden`** (`crates/vokra-core/src/compliance/license_class.rs`) |
| `redistributable()` | **`false`** |
| `publish-one.sh` gate 2 | **REFUSE** (fail-closed) |
| `--allow-noncommercial` bypass | **不可** (T4 flag は `NonCommercial` class 用、`RedistributionForbidden` は bypass する条件が code に無い) |

**結論**: このモデルは **現状のまま `vokra/higgs-audio-v3-tts-4b` として publish
できない**。converter code は `crates/vokra-convert/src/models/higgs_audio_v3_tts_4b.rs`
に既 land だが、weight redistribution は upstream license に阻まれる。

### 0.4 何が起きたか (audit → primary source drift の記録)

- coverage-audit-2026-08-03 の wave-b ticket は `docs/tickets/coverage-audit-2026-08-03/wave-b/higgs-audio-v3-tts-4b.md`
  で "Apache-2.0" を assume していた
- Converter code (Wave B fast-track land、commit `5c77597`) は audit ticket 記述の
  まま `DEFAULT_LICENSE = "apache-2.0"` を default に据えた (`crates/vokra-convert/src/models/higgs_audio_v3_tts_4b.rs`)
- しかし upstream `bosonai/higgs-audio-v3-tts-4b` の HF card は 2026-07-08 に
  BosonAI 独自 R&NC license に更新された (LICENSE 上部 "Last Updated: July 8,
  2026" ゆえ audit ticket 起票時期以降の更新の可能性、あるいは audit 側の primary
  source 未照合)
- Task 指示 (2026-08-13 依頼者経由) も "Apache-2.0" 記述だったが primary source
  で覆された

**教訓** (memory `[[feedback-license-signoff-primary-source]]` の実践): audit
ticket / task 指示 / converter DEFAULT_LICENSE は**参考にとどめ、publish 前に必ず
primary source (HF cardData API + LICENSE 実文書) を直接照合する**。今回は
CC-write-doc の段で catch した = §3.1 sign-off gate + primary-source rule が
正しく fail-closed で機能した典型例。

## 1. モデル情報 (primary source 照合後の実値)

| 項目 | 値 |
|---|---|
| Upstream HF repo | `bosonai/higgs-audio-v3-tts-4b` |
| Upstream HF URL | https://huggingface.co/bosonai/higgs-audio-v3-tts-4b |
| Upstream sha | `<未 pin — HF listing 時点で lastModified を owner 確認>` |
| License (upstream code) | Boson Higgs TTS 3 Research and Non-Commercial License |
| License (upstream weight) | 同上 (weight と code 一体) |
| SPDX (Vokra 判定) | `LicenseRef-Boson-Higgs-TTS-3-Research-Non-Commercial` (SPDX 未登録 custom) |
| Total safetensors size | **8.67 GiB (9,309,834,930 bytes) single-shard**（BF16、`model.safetensors` 単一ファイル + `model.safetensors.index.json` はメタデータ用 90 KB） |
| 判定 (`check-model-size.sh`) | `LOCAL_BORDERLINE` (8-16 GiB range) — 依頼者ルール #1 で ≥2GB は vast.ai 推奨 |
| Vokra ModelKind | `HiggsAudioV3Tts4b` (`--model higgs-audio-v3-tts-4b` / `higgs-audio-v3` / `higgs_audio_v3`) |
| Arch tag | `vokra.model.arch = "higgs_audio_v3_tts_4b"` (sibling TTS と区別) |
| Vokra HF slug | `vokra/higgs-audio-v3-tts-4b` (現時点は **publish 不可**、§0.3 参照) |
| Attribution 要求 | LICENSE §IV(a) attribution + §II-A(b) Creator Acknowledgment (Creator use のみ)  |
| Non-commercial 制限 | あり (Research + Non-Commercial のみ、Commercial は §III で別途書面契約要) |
| Redistribution 制限 | **あり (§II-A(c)(ii) で明示禁止、Commercial license 要)** — Vokra publish 不可 |
| Language coverage | 100+ 言語 (HF card tags = 97 languages 列挙、multilingual TTS) |
| Emotion inline tag | `[happy]` / `[sad]` 等が LM tokenizer に baked-in (audit ticket 記述) |

### Primary source verify command (本 doc land 時点で実行済)

```bash
# License verification (§0.1 参照)
curl -sL https://huggingface.co/api/models/bosonai/higgs-audio-v3-tts-4b | \
  python3 -c "import json,sys; d=json.load(sys.stdin); cd=d.get('cardData',{}); \
    print('license:', cd.get('license')); \
    print('license_name:', cd.get('license_name'))"
# → license: other / license_name: boson-higgs-tts-3-research-and-non-commercial-license

# LICENSE full text (§0.2 参照)
curl -sL https://huggingface.co/bosonai/higgs-audio-v3-tts-4b/raw/main/LICENSE > /tmp/higgs-license.txt
head -60 /tmp/higgs-license.txt
grep -A 3 "redistribut" /tmp/higgs-license.txt  # §II-A(c)(ii) の redistribution 禁止条項

# Size verification
./scripts/publish/check-model-size.sh bosonai/higgs-audio-v3-tts-4b
# expected verdict: LOCAL_BORDERLINE (8.67 GiB single-shard)
```

## 2. vast.ai instance recipe (**publish 不可のため参考のみ**)

**注意**: §0.3 の通り現状 publish 不可ゆえ、下記 vast.ai recipe は **owner が
Boson と commercial redistribution license を締結できた場合の参考手順** としてのみ
記載する。締結前は §4 の run-one.sh を実行しても `publish-one.sh` gate 2 で
refuse される。

**共通仕様**は `docs/handoff/vast-ai-large-model-publish.md` §2.2 を参照。
Higgs-Audio v3 TTS 4B 固有の値:

| 項目 | 推奨値 | 備考 |
|---|---|---|
| Image | `pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel` or `nvidia/cuda:13.0.0-*` | 総論 §2.2 と同じ。`nvidia/cuda:13.0.0` 系は provision.sh gotcha §3 参照 |
| RAM | **32 GB 以上**（総論の 64 GB より緩和可） | 8.67 GB single-shard + convert working set + upload buffer。11-shards Voxtral-Small-24B (48 GB) 級ではない |
| Disk | **80 GB 以上** | 上流 DL 8.67 GB + GGUF 8.67 GB + HF cache buffer + 余裕 |
| GPU | Convert には不要 (converter は CPU only) | vast.ai は GPU 前提販売ゆえ最安 GPU を選ぶ |
| Network | 非従量課金 or inclusive band | 上下 ~18 GB out-bound (DL 8.67 GB + upload 8.67 GB) |
| 課金見込 | ~1-1.5h × $0.3-0.5/hr = **$0.3-0.75** | Voxtral-Small-24B ($0.6-1.0) より安い |

## 3. provision.sh gotcha (総論 §3 を参照)

`scripts/publish/vast-ai/provision.sh` は下記 4 件を idempotent に修正済。
Higgs-Audio 4B の convert 前に一度だけ実行:

| Gotcha | 起因 | provision.sh の対応 |
|---|---|---|
| **hf_config.pth shim** | `nvidia/cuda:13.0.0` image が仕込む Python startup shim が `HF_ENDPOINT` を malicious mirror `117.175.104.83:8081` に上書き | shim 除去 + certifi CA 再植え付け (memory `[[reference-vast-ai-hf-config-pth-shim]]`) |
| **huggingface_hub < 0.30 pin** | 1.x xet-token routing が mirror 404 を投げ、`HF_HUB_DISABLE_XET` も一部 bypass、0.30+ `resume_download` deprecated で flaky egress を落とす | vast.ai 上のみ `huggingface_hub < 0.30` に pin (memory `[[reference-huggingface-hub-lt-030-vast-ai]]`) |
| **certifi CA bundle** | 空 or 古い CA bundle で HTTPS 検証失敗 | `certifi` 再 install + `SSL_CERT_FILE` export |
| **stack tool install (torch/numpy/safetensors)** | resilient_batch.sh の uv fallback / ad-hoc `python3 -c` が消費するが system 層に無い | provision.sh Wave 12 で pre-install |

```bash
# SSH 接続後、まず HF token を export (instance destroy で消える)
export HF_TOKEN='hf_xxxxxx'   # 本機 .env の HF= 値をここに貼る

# 1 コマンドで Rust + uv + Python 3.12 + hf-transfer + repo + vokra-cli build まで完了
curl -sSL https://raw.githubusercontent.com/ayutaz/vokra/main/scripts/publish/vast-ai/provision.sh | bash
source ~/.bashrc  # VOKRA_PUBLISH_ON_VAST=1 marker を pick up
```

## 4. Convert command (**publish は §0.3 で blocked**、convert は可能)

### 4.1 自動化 pipeline (推奨、Phase B) — publish 段で gate 2 refuse

```bash
# provision.sh 完了後、以下 1 コマンド (dry-run のみ、--push は fail-close する想定)
~/vokra/scripts/publish/vast-ai/run-one.sh \
  --hf-repo bosonai/higgs-audio-v3-tts-4b \
  --vokra-slug higgs-audio-v3-tts-4b \
  --model-kind higgs-audio-v3-tts-4b
  # --license-spdx は passthrough しない (SPDX 未登録の custom license)
  # --push を付けない: publish-one.sh gate 2 (redistributable()=false) で refuse
```

**期待挙動**: `run-one.sh` は下記 chain を実行:
1. HF snapshot_download で 8.67 GB weight + config + tokenizer + LICENSE を DL
2. `tools/parity/higgs_audio_v3_tts_4b/prepare_checkpoint.py` で `model.safetensors`
   を pass-through (single-shard ゆえ merge 不要、strip / dedup も 8.67 GB 単一ゆえ
   noop の可能性大 — 実行して確認)
3. `vokra-cli convert --model higgs-audio-v3-tts-4b` で BF16 pass-through GGUF 生成
4. `publish-one.sh` gate chain: **gate 2 で refuse** (`RedistributionForbidden` は
   `redistributable() = false`)

**convert 段は成功する** (BF16 pass-through は license に依存しない) — owner が
converter code の実挙動を確認する材料になる。

### 4.2 手動 fallback (総論 §2.5 に準拠)

```bash
mkdir -p ~/scratchpad/hf-cache ~/scratchpad/staging/higgs-audio-v3-tts-4b

cd ~/vokra/tools/parity
uv sync   # pyproject.toml + uv.lock から依存 install (higgs_audio_v3_tts_4b/ は sibling)

# HF から weight + config + LICENSE DL
uv run --with huggingface_hub python <<'PY'
import os
os.environ.setdefault('HF_HUB_CACHE', '/root/scratchpad/hf-cache')
from huggingface_hub import snapshot_download
path = snapshot_download(
    repo_id='bosonai/higgs-audio-v3-tts-4b',
    cache_dir='/root/scratchpad/hf-cache',
    allow_patterns=['*.safetensors', 'model.safetensors.index.json',
                    'config.json', 'generation_config.json',
                    'tokenizer*.json', 'chat_template.jinja',
                    '*.md', 'LICENSE'],
    token=os.environ['HF_TOKEN'],
)
print('DONE:', path)
PY

# Prepare checkpoint (single-shard ゆえ noop / verify のみ)
SNAP=$(ls -d /root/scratchpad/hf-cache/models--bosonai--higgs-audio-v3-tts-4b/snapshots/*/ | head -1)
cd ~/vokra
uv run --project tools/parity/higgs_audio_v3_tts_4b python \
  tools/parity/higgs_audio_v3_tts_4b/prepare_checkpoint.py \
  --input-dir "$SNAP" \
  --output   /root/scratchpad/staging/higgs-audio-v3-tts-4b/model.merged.safetensors

# Convert
./target/release/vokra-cli convert \
  --model higgs-audio-v3-tts-4b \
  --input /root/scratchpad/staging/higgs-audio-v3-tts-4b/model.merged.safetensors \
  --config "$SNAP/config.json" \
  --output /root/scratchpad/staging/higgs-audio-v3-tts-4b/model.gguf

# Publish dry-run (これで gate 2 refuse を確認)
./scripts/publish/publish-one.sh \
  --gguf /root/scratchpad/staging/higgs-audio-v3-tts-4b/model.gguf \
  --repo vokra/higgs-audio-v3-tts-4b
# expected: gate 2 REFUSE "LicenseClass::RedistributionForbidden — not publishable"

# --push は付けない (fail-closed)
```

### 4.3 SGLang sampler 前提

Higgs-Audio v3 は upstream で **SGLang** (structured generation runtime) sampler
経由の generation を前提としている (audit ticket 記述)。Vokra runtime は SGLang
に依存しない — 標準の Sampler primitive (temperature / top-p / top-k / repetition
penalty、`crates/vokra-core/src/decode/sampler.rs`) に置換する:

- **converter 側 impact**: なし (BF16 pass-through は sampler 独立)
- **runtime forward 側 impact**: `crates/vokra-models/src/higgs_audio_v3/` (未実装、
  future wave) で SGLang-specific inference-time behavior (structured generation /
  regex-guided decoding 等) を implement する時は Vokra Sampler primitive の
  extension point を使う (voxtral / kyutai_stt / canary_qwen precedent と同型)
- **emotion inline tag**: `[happy]` / `[sad]` 等は LM tokenizer に baked-in ゆえ
  Sampler 側で特殊処理は不要 (通常の token sampling で扱える)
- **音色 conditioning**: reference audio → speech token の zero-shot cloning
  能力 (音色 clone) は voice-cloning capability に該当する可能性 — 実装時に
  ELVIS Act (`docs/legal-compliance.md`) 適用性を再確認、該当する場合は
  `vokra-voiceclone-experimental` 別リポへ (memory `[[voice-clone-experimental]]`
  相当の判断)

## 5. §3.1 sign-off status

**現状: blank (fail-closed default) — かつ publish 不可**。

`docs/license-audit.md` §3.1 に `vokra/higgs-audio-v3-tts-4b` 行を **追加待ち**
(本 doc land 時点では未追加、owner の primary-source 判断を経てから追加)。追加後の
状態は以下のいずれか:

| 判断 | Vokra tier | Commercial sign-off | 前提 |
|---|---|---|---|
| **☑ Rejected (現状の default)** | — | ☑ Rejected 2026-XX-XX yousan | LicenseClass = RedistributionForbidden、publish blocked、owner が Boson と交渉しない前提 |
| **☑ Commercial** | T1 (Permissive) | ☑ Commercial 2026-XX-XX yousan | Boson との commercial redistribution license 締結後 — `LicenseClass::from_license_str` の side-car (or converter override) で `Permissive` にマップ可能な文書化された private license 取得後のみ |

**Owner action** (優先順):

1. **Primary source 直接照合** (本 doc §0.1〜0.3 の内容を owner 目視で再確認) —
   本 doc 記述内容が 2026-08-13 CC 実行時点の primary source と一致することを
   verify
2. **判断**:
   - (a) **Boson と commercial redistribution license 交渉** (contact@boson.ai / https://boson.ai) — 締結できれば ☑ Commercial + LicenseClass override で publish 可
   - (b) **Skip publish、converter code は残置** — converter code (`crates/vokra-convert/src/models/higgs_audio_v3_tts_4b.rs`) と sidecar (`tools/parity/higgs_audio_v3_tts_4b/prepare_checkpoint.py`) は既 land、実 weight を持つ owner が local で GGUF 生成することは license §II の Research/Non-Commercial 範囲で可能 (「他人に配る」わけではない = §II-A(c)(ii) 抵触しない)
   - (c) **Downstream user への case-by-case guidance** — 「upstream から直接
     download し、local で `vokra-cli convert` を叩くことは license §II 範囲」と
     document (`docs/license-audit.md` §3.1 の Notes 欄) — Vokra 側は artifact
     配布はしない、CC 判断だけを示す
3. **☑ Rejected sign-off** (現状の default) — `docs/license-audit.md` §3.1 row を
   `☑ Rejected 2026-XX-XX yousan (LicenseClass = RedistributionForbidden per
   LICENSE §II-A(c)(ii))` として land、以降 owner が (a) を締結するまで publish
   path は closed

**publish-one.sh gate map** (総論 §2.5 と同じ、5-gate + gate 7):

1. Catalog reality — 未実装の ★ 公式 zoo 宣言拒否 (Higgs は runtime forward 未
   実装ゆえ ★ 宣言しない前提、pass)
2. **Redistributable — `LicenseClass::redistributable() = false` で REFUSE (fail-
   closed default)** ← **本モデルはここで止まる**
3. Provenance chunk 刻印 — `vokra.provenance.*` chunk 群が missing なら拒否
   (converter が刻印済、pass 想定)
4. §3.1 sign-off 欄 blank 拒否 (fail-closed) — owner が ☑ を書くまで拒否
5. T4 (NonCommercial) は `--allow-noncommercial` 明示必須 — 本モデルは
   `RedistributionForbidden` ゆえ T4 flag では bypass 不可
6. (欠番)
7. >8 GiB fail-closed (8.67 GiB → **hit**) — `VOKRA_PUBLISH_ON_VAST=1` (provision.sh が set) or `--allow-large` で bypass。vast.ai 上では自動 bypass、local からの誤 publish 事故防止

## 6. 期待される artifacts (**owner が (a) を締結して publish に成功した場合**)

| ファイル | 内容 |
|---|---|
| `model.gguf` | ~8.67 GB (BF16 pass-through 変換、tensor 数は上流 safetensors index に依存) |
| `README.md` | `make_model_card.py` 自動生成、tier 表記 + license 表記 + upstream 情報 + Creator Acknowledgment (§II-A(b) 要求) を satisfy する文言 |
| `LICENSE` | Boson Higgs TTS 3 R&NC License 全文 (`fetch_license.sh --url https://huggingface.co/bosonai/higgs-audio-v3-tts-4b/raw/main/LICENSE` で取得、SPDX 未登録ゆえ inline text 経由も候補、`fetch_license.sh` の openmdw-1.1 pattern を mirror) |
| `NOTICE` | attribution required (LICENSE §IV(a) + §II-A(b) Creator Acknowledgment 文言 template) |
| `SOURCE.md` | 上流 URL + 再変換手順 + Vokra converter バージョン + commit SHA + LicenseClass override rationale |

### GGUF metadata (vokra.* chunk 群、converter が刻む)

| Key | 型 | 値 |
|---|---|---|
| `vokra.schema.version` | string | `"1"` (writer choke point で自動刻印) |
| `vokra.schema.producer` | string | `"vokra-cli-<version>"` |
| `vokra.model.arch` | string | `"higgs_audio_v3_tts_4b"` (sibling TTS と区別、converter header docstring §License 参照) |
| `vokra.model.category` | string | `"tts"` (sibling `magpietts_v2602` と同 posture) |
| `vokra.provenance.upstream_hf` | string | `"bosonai/higgs-audio-v3-tts-4b"` |
| `vokra.provenance.upstream_revision` | string | `"<owner が HF listing の lastModified から pin>"` |
| `vokra.provenance.upstream_license` | string | `"LicenseRef-Boson-Higgs-TTS-3-Research-Non-Commercial"` (SPDX 未登録の custom ゆえ LicenseRef prefix) |

## 7. Gate 発火状態 (parity CI)

**現状**: parity CI は未設定 (converter code は既 land、runtime forward は未
実装、`crates/vokra-models/src/higgs_audio_v3/` は future wave)。 publish が
blocked のため CI 発火は runtime forward 実装後 + license issue 解消後の順序。

Runtime forward 実装 (future wave) 後の flip-the-switch は:
1. `crates/vokra-models/src/higgs_audio_v3/` に native forward 実装
2. `.github/workflows/parity-tts-higgs-audio-v3-real.yml` scaffold 追加
3. Owner が §5 の (a) 経路で publish 済であれば fixture GGUF を CI が pull
4. PyTorch reference dump は SGLang 依存ゆえ dumper 側で `pip install sglang`
   前提 (Vokra runtime に SGLang は入れない、reference 独立性 rule)

## 8. Owner critical path (優先順)

**依頼者ルール #3 (publish は §3.1 sign-off 完了後 owner が判断) に従い、以下
順序で**:

1. **§0 primary source correction 内容の目視確認** — 2026-08-13 CC fetch と
   一致していること (LICENSE 上部 "Last Updated: July 8, 2026" 表記が変わっていな
   いこと、§II-A(c)(ii) 条項が消えていないこと)
2. **判断 (a) or (b) or (c)** (§5 参照):
   - (a) Boson と交渉 → 締結 → `LicenseClass::from_license_str` に private
     license marker を case-by-case で追加 → publish 可
   - (b) Skip publish、converter code は残置、owner の local convert のみ許可
   - (c) Downstream user への guidance を `docs/license-audit.md` §3.1 Notes 欄
     に document
3. **☑ Rejected sign-off (default)** — 上記 (a) を選ばない場合、`docs/license-audit.md`
   §3.1 に `vokra/higgs-audio-v3-tts-4b` 行を追加、`☑ Rejected 2026-XX-XX yousan
   (RedistributionForbidden per LICENSE §II-A(c)(ii), primary source verified
   YYYY-MM-DD)` を刻む
4. **(a) を選んだ場合のみ) vast.ai instance 起動** — §2 recipe、~$0.3-0.75、
   ~1-1.5 hour
5. **(a) を選んだ場合のみ) §3 provision.sh → §4.1 run-one.sh --push** — gate 2
   が pass する license mapping が code 側にあることを事前確認

## 9. Notes

- **converter code は既 land**: `crates/vokra-convert/src/models/higgs_audio_v3_tts_4b.rs`
  (Wave B fast-track、commit `5c77597`) と `tools/parity/higgs_audio_v3_tts_4b/`
  sidecar は既存。実 weight を local で持つ owner は `vokra-cli convert` で GGUF
  化できる (Research/Non-Commercial 範囲) — ただし **他人に配ることは license
  §II-A(c)(ii) で禁止**
- **音色 conditioning capability**: reference audio → zero-shot voice cloning 能力
  が upstream で確認された場合、ELVIS Act (`docs/legal-compliance.md`) 適用の
  可能性 — 該当時は `vokra-voiceclone-experimental` 別リポへ (main repo `ayutaz/vokra`
  に絶対に land しない、[[voice-clone-experimental]] 相当の分離ポリシー)
- **restamp_provenance で低メモリ再刻印可能**: publish が実現した後 LICENSE /
  NOTICE / SOURCE.md を差し替えたい場合、`restamp_provenance` で tensor コピー
  無しで刻印可能 (memory `[[project-restamp-provenance]]`、8.7 GB Voxtral を M1
  iMac 16 GB で peak footprint 6.4 MB 実測)。Higgs-Audio 4B の 8.67 GB は同スケール
  ゆえ同手法で余裕を持って再刻印可能 = vast.ai 再起動不要
- **BF16 pass-through**: converter は BF16 → BF16 の pass-through (K-quants
  スコープ外)。Runtime forward 側で BF16 → F32 は `crates/vokra-core/src/gguf/quant/mod.rs`
  `decode_bf16` が losslessly widen する
- **task 指示の "Apache-2.0" 記述との差分**: §0 参照。primary source 照合を優先、
  audit ticket / task 指示は参考、fail-closed default が正しく機能した

## 関連

- 総論: `docs/handoff/vast-ai-large-model-publish.md`
- Converter: `crates/vokra-convert/src/models/higgs_audio_v3_tts_4b.rs`
- Sidecar: `tools/parity/higgs_audio_v3_tts_4b/prepare_checkpoint.py`
- LicenseClass: `crates/vokra-core/src/compliance/license_class.rs` (`RedistributionForbidden` predicate)
- LicenseClass X-Codec-2 T4 precedent (Rejected 系ではなく NonCommercial 系): memory `[[project-x-codec2-t4-precedent.md]]`
- 5-gate publish: memory `[[project-huggingface-vokra-publication]]`
- Primary source rule: memory `[[feedback-license-signoff-primary-source]]`
- vast.ai routing: memory `[[feedback-large-models-on-vast-ai]]`
