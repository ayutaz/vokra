# HuggingFace `vokra` org — 未対応モデル一覧 (2026-07-28)

**Purpose**: 公開 (`huggingface.co/vokra`) に未対応のモデルを owner-visible な一覧にして
後続の対応判断を可能にする。依頼者指示 (2026-07-28)「未対応のものは未対応リストに入れて
後から対応するように」に基づく。

**現在の公開状況**: **51 モデル live** (2026-07-28 時点)。詳細一覧は `docs/license-audit.md`
§3.1 sign-off 表を参照。

**未対応 = 5 tier に分類** (blocker と復活条件を明記して future action を判断可能に):

---

## A. Converter 未実装 / 拡張が必要 (owner 判断 = 実装 GO/NO-GO)

CC で converter 側実装を追加すれば publish 可能な tier。owner の scope 判断待ち。

| Model | Upstream | §3.1 sign-off | Blocker | 復活条件 |
|-------|----------|---------------|---------|----------|
| **VoxCPM2 2B multilingual** | `openbmb/VoxCPM2` | ☑ Commercial (2026-07-28) | 現 `ModelKind::VoxCpm2` は 0.5B 用の tensor name mapping、2B は multilingual + 追加 hparam | converter に 2B path 追加 (arch 分岐 or 新 kind) |
| **WavTokenizer** | (未確定) | 未取得 | converter 未実装 | M5-07 sign-off + converter 新規実装 |
| **Matcha-TTS** | (未確定) | 未取得 | converter 未実装 | M5-07 sign-off + converter 新規実装 |
| **Bark** | `suno/bark` | 未取得 | converter 未実装、Suno の voice-cloning 再学習禁止方針 | M5-07 sign-off (v2.0+ 検討) + converter 新規実装 |
| **Voxtral-Small-24B** | `mistralai/Voxtral-Small-24B-2507` | ☑ Commercial (2026-07-23) | 48GB BF16 = local M1 (16GB RAM) では変換不可 | vast.ai instance + `scripts/publish/publish-one.sh --push` (converter 既 wire 済、直近 vast.ai 経路で publish 済のため同 workflow を再走) |

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

公式 `vokra` org には出さない。ELVIS Act 対応 (`CLAUDE.md` 設計判断 8) の分離設計。

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

## 更新規律

- 新規 publish 完了 → 該当行を削除 + `docs/license-audit.md` §3.1 に signed row 追加
- 新規未対応の発覚 → 該当 tier に追加
- Owner 判断が変わった → tier 間移動 (例: C の Rejected → D の T4 GO / A の 実装 GO)
- 本ファイルは gitignore-free tracked = 公開時に owner から見える。関連 memory =
  `project-huggingface-vokra-publication` の「未対応リスト」節と対称に保つ (memory 側は
  gitignore-local ゆえ内部規律の SoT、tracked 側は公開時可視 handoff)。
