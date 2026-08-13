# vast.ai publish runbook — RMVPE (Robust Model for Vocal Pitch Estimation)

**Owner-triggered.** CC は本 doc 作成のみ。実 vast.ai instance の起動・weight
provisioning・parity 検証は owner が本 runbook を追いながら実行する。

**✅ 2026-08-13 更新: loud-partial は resolved**（本 branch
`feat/post-audit-cc-gap-2026-08-13` の commit
[`e7b6810`](../../crates/vokra-models/src/f0/rmvpe.rs) で real U-Net +
BiGRU forward が land）。上流 `yxlllc/RMVPE` (MIT) の code inspect で
topology が **fully-specified** と判明したため、旧 "under-specified in
primary source" 判定は REVERSED。inline `pool2d` + `conv_transpose2d` +
`pytorch_gru` を実装（外部 op 依存なし、NFR-DS-02 保存）、`extract_real()`
は `VokraError::UnsupportedOp` を返さなくなった。

- **Path A** (`VOKRA_RMVPE_REAL_GGUF`): shape / finite / sigmoid-range
  contract を binding — owner が real GGUF を用意すれば即発火。
- **Path B** (`VOKRA_RMVPE_REAL_HIDDEN` + `_ARGMAX` +
  `_HIDDEN_FEATURE_DIM`): argmax-match rate ≥ 99 % gate を binding —
  owner が `tools/parity/rmvpe/dump_reference.py` を走らせて hidden.f32
  + argmax.u32 を用意すれば即発火。

**判断の evolution**:
- 2026-07-30 (CLAUDE.md wave 3): "topology は under-specified、owner が
  real .pt bundle を用意するまで silent-wrong risk を避けて defer" →
  `extract_real` は `VokraError::UnsupportedOp` を返す **loud-partial**
  で land、fake-complete を書かないという honesty 判断で正しかった。
- 2026-08-13 (本 branch, feasibility 調査 `wf_7062f2d5` の一次 code
  inspect): 上流 `yxlllc/RMVPE` の `src/model.py` を精査した結果、U-Net
  + BiGRU + head の shape / stride / padding / groups / bidirectional
  が primary-source-transcribable と判明 → real forward を land 可能な
  条件が揃った → commit e7b6810 で **REVERSED**。

これは "loud-partial は fake-complete より honest" 判断（memory
[[project-m4-implementation]] の verify-on-actual-HEAD 規律の horizontal
展開）の後続として、"primary source を再精査したら fully-specified
だった" 例が加わった形。以降 loud-partial の判定は上流を再精査したうえで
下すのが望ましい。

**Related**:
- 本 runbook は `docs/handoff/vast-ai-large-model-publish.md`（総論）を **前提** と
  する。共通手順は総論を参照し、本 doc は RMVPE に固有の差分のみを記述する。
- CI: `.github/workflows/parity-rmvpe-real.yml`（land 済、`VOKRA_RMVPE_ENABLE=1`
  で発火）
- Parity harness: `crates/vokra-models/tests/parity_rmvpe.rs`（Path A +
  Path B の fixture-gated leg 両方が land 済）
- Impl module: `crates/vokra-models/src/f0/rmvpe.rs`（e7b6810 で real
  forward land、`extract_real` は real U-Net + BiGRU を走らせる）
- Converter: `crates/vokra-convert/src/models/rmvpe.rs`（land 済）
- **Path B reference dumper**: `tools/parity/rmvpe/dump_reference.py`
  + `tools/parity/rmvpe/README.md`（本 branch でこの後 land）

## 1. モデル情報

| 項目 | 値 |
|---|---|
| Upstream primary | https://github.com/Dream-High/RMVPE |
| Upstream fork | https://github.com/yxlllc/RMVPE（同 architecture） |
| Paper | Wei et al. 2023 — "RMVPE: A Robust Model for Vocal Pitch Estimation in Polyphonic Music" (INTERSPEECH 2023) |
| Upstream release | GitHub Releases に `.pt` pickle（`model.pt` or `checkpoint_pretrain.pt`） |
| Upstream HF | **なし**（github release のみ、HF mirror 未存在） |
| License | **MIT**（両 upstream、Permissive、runtime-side attribution 不要） |
| SPDX (Vokra 判定) | `mit`（LicenseClass = Permissive） |
| Weight size | **~180 MB**（`.pt` pickle） |
| 判定 (`check-model-size.sh`) | 該当なし（HF mirror 不在）。実サイズは `LOCAL_SAFE`（<4 GiB） |
| Vokra ModelKind | `Rmvpe`（既存、CLI `--model rmvpe`） |
| Vokra HF slug | `vokra/rmvpe`（publish 予定、Wave 3 wave-B 決定） |
| Attribution 要求 | なし（MIT 標準 LICENSE 同梱のみ） |
| Non-commercial 制限 | なし |
| Share-alike | なし |

### なぜ vast.ai なのか（サイズ観点では不要）

**サイズだけを見れば local M1 iMac で処理可能**（180 MB × 2 shard 想定 = local
safe）。しかし本 runbook が vast.ai を指定する理由は 3 つ:

1. **依頼者ルール #3**: publish は §3.1 sign-off 完了後 owner が判断、CC は
   converter + test + docs までゆえ、実 publish action は owner-triggered
2. **Real checkpoint 未 owner-provisioned**: 現在 upstream GitHub Releases から
   `.pt` pickle を owner が **手動 fetch** する必要がある（HF mirror 不在、CI 側
   `snapshot_download` が使えない）
3. ~~**Kernel binding 待ち**~~ **✅ 解消済 (2026-08-13, commit e7b6810)**:
   real forward が動くための (a) real weight GGUF variant-emit → (b) U-Net
   + GRU kernel 実装 → (c) parity harness bit-exact verify、の 3 段階のうち
   **(b) は本 branch で land**（inline `pool2d` + `conv_transpose2d` +
   `pytorch_gru`、no external op deps）、**(c) の Path B reference dumper
   も本 branch で land**（`tools/parity/rmvpe/dump_reference.py`）。owner
   は (a) real `.pt` fetch + GGUF bridge + reference dump の 3 step で
   即 parity 発火可能（vast.ai 不要、local M1 iMac で完結、下記 §2.1）。

したがって本 runbook は「vast.ai 上での convert + publish」ではなく **「owner が
local M1 iMac 上で `.pt` → `.safetensors` → GGUF bridge + reference dump を実施
し、Path A + Path B 両方の fixture を CC の parity harness に渡す」** 手順を
記述する（vast.ai は optional、依頼者ルール #1 との整合上 tag はしているが実
運用上 local で足りる — 詳細は §2.1）。

### Primary source verify command

```bash
# License verification (github repository primary)
# GitHub raw LICENSE file
curl -sL https://raw.githubusercontent.com/Dream-High/RMVPE/main/LICENSE | head -5
# expected: MIT License Copyright (c) 2023 Haojie Wei

# Release verify - github release で pt file の存在を確認
curl -sL https://api.github.com/repos/Dream-High/RMVPE/releases/latest | \
  python3 -c "import json,sys; d=json.load(sys.stdin); \
    print('tag:', d.get('tag_name')); \
    [print(a['name'], a['size']) for a in d.get('assets',[])]"
```

## 2. Instance recipe（local M1 iMac 推奨、vast.ai は fallback）

### 2.1 Local M1 iMac 上での bridge（推奨、依頼者ルール #2 に該当）

**サイズ: 180 MB `.pt` pickle → ~180 MB safetensors → ~180 MB GGUF**。M1 iMac
16 GB RAM で余裕、依頼者ルール #2（<2GB は local OK）に該当。

```bash
cd ~/vokra/tools/parity
uv sync   # pyproject.toml + uv.lock から依存 install

# 1. Upstream .pt を fetch（github release から）
mkdir -p ~/rmvpe-fixtures
cd ~/rmvpe-fixtures
# 例: yxlllc/RMVPE release から model.pt
curl -L -o rmvpe.pt \
  "https://github.com/yxlllc/RMVPE/releases/download/230917/model.pt"
# 実 URL は release page で最新確認: https://github.com/yxlllc/RMVPE/releases

# 2. Flatten to safetensors（fair-use pickle → safetensors converter）
uv run python ~/vokra/tools/parity/nemo_pt_to_safetensors.py \
    --input  ~/rmvpe-fixtures/rmvpe.pt \
    --output ~/rmvpe-fixtures/rmvpe.safetensors

# 3. Convert to GGUF
cd ~/vokra
cargo build --release -p vokra-cli
./target/release/vokra-cli convert \
    --model rmvpe \
    --input  ~/rmvpe-fixtures/rmvpe.safetensors \
    --output ~/rmvpe-fixtures/rmvpe.gguf

# 4. Parity harness に real GGUF を feed（fixture verify）
export VOKRA_RMVPE_REAL_GGUF=~/rmvpe-fixtures/rmvpe.gguf
cargo test -p vokra-models --test parity_rmvpe -- --nocapture
```

**現状の parity harness expected behavior (2026-08-13 更新後)**:

- `from_gguf` は real weight を bind して pass
- Mel front-end は real STFT + mel filterbank を実行して bit-exact verify
- 360-class → Hz decoder は synthetic inputs で verify
- **Path A**: `extract_real` は real U-Net + BiGRU + head を走らせ、
  shape / finite / sigmoid-range contract を binding（`parity_rmvpe.rs:264-303`
  `parity_rmvpe_gguf_smoke`、e7b6810 で loud-partial 解消済）
- **Path B**: `forward_from_hidden` は上流 dumper の post-CNN hidden
  state を bit-exact 受け取り、argmax-match rate ≥ 99 % gate を binding
  （`parity_rmvpe.rs:320-438` `parity_rmvpe_from_hidden_argmax_match_rate`、
  fixture は `tools/parity/rmvpe/dump_reference.py` で生成）

Path B の追加 step は下記の通り:

```bash
# 5. Path B fixture 生成（上流の nn.Module を fair-use verbatim reference として使用）
cd ~/vokra/tools/parity/rmvpe
uv sync
git clone https://github.com/yxlllc/RMVPE.git ~/rmvpe-upstream    # 一度のみ
uv run python dump_reference.py \
    --pt-path      ~/rmvpe-fixtures/rmvpe.pt \
    --upstream-src ~/rmvpe-upstream \
    --canned \
    --out-dir      ~/rmvpe-fixtures/dump
# → hidden.f32 + argmax.u32 + meta.json が出力される

# 6. Path A + Path B 両方の env を export し parity harness を実行
export VOKRA_RMVPE_REAL_GGUF=~/rmvpe-fixtures/rmvpe.gguf
export VOKRA_RMVPE_REAL_HIDDEN=~/rmvpe-fixtures/dump/hidden.f32
export VOKRA_RMVPE_REAL_HIDDEN_FEATURE_DIM=$(python3 -c \
    'import json; print(json.load(open("'"$HOME"'/rmvpe-fixtures/dump/meta.json"))["feature_dim"])')
export VOKRA_RMVPE_REAL_ARGMAX=~/rmvpe-fixtures/dump/argmax.u32
cd ~/vokra
cargo test -p vokra-models --test parity_rmvpe -- --nocapture
```

### 2.2 vast.ai fallback（storage / dependency 問題があれば）

`.pt` unpickle が local M1 iMac で fail する場合（PyTorch version mismatch、
Python 3.12 環境で `.pt` pickle 互換性問題等）は vast.ai fallback:

| 項目 | 推奨値 |
|---|---|
| Image | `pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel` |
| RAM | 8 GB 以上（充分） |
| Disk | 20 GB 以上（充分） |
| GPU | 不要 |
| 課金見込 | ~30 min × $0.3/hr = **$0.15** |

## 3. provision.sh gotcha（vast.ai fallback 時のみ）

`docs/handoff/vast-ai-large-model-publish.md` §3 の 4 gotcha 全て該当:

1. **hf_config.pth shim** — 除去、`nvidia/cuda:13.0.0` image gotcha
2. **huggingface_hub < 0.30 pin** — RMVPE は HF mirror 未存在ゆえ hub 使用は
   本 runbook 内で **provision.sh の cache dir 準備のみ**（実 fetch は github
   release から curl）
3. **certifi CA bundle** — HTTPS 検証（github release download）
4. **stack tool install** — `torch` / `numpy` / `safetensors`（unpickle 用）

provision.sh 実行は 1 コマンドで対応（`docs/handoff/vast-ai-large-model-publish.md`
§0 TL;DR 参照）。

## 4. Convert + publish command

### 4.1 Local pattern（推奨、§2.1 の続き）

Fixture 生成後、publish は **owner § 3.1 sign-off 完了後** に:

```bash
# Publish 5-gate（dry-run → --push で本番）
./scripts/publish/publish-one.sh \
  --gguf ~/rmvpe-fixtures/rmvpe.gguf \
  --repo vokra/rmvpe \
  --license-spdx mit
# ↑ dry-run 全 gate 通過を確認してから ↓
./scripts/publish/publish-one.sh \
  --gguf ~/rmvpe-fixtures/rmvpe.gguf \
  --repo vokra/rmvpe \
  --license-spdx mit --push

# 検証
curl -sI https://huggingface.co/vokra/rmvpe | head -1
# HTTP/2 200 が返れば live
```

### 4.2 vast.ai automated pipeline（HF mirror 不在ゆえ非該当）

`scripts/publish/vast-ai/run-one.sh` は `--hf-repo` を必須引数として消費する
（HF snapshot_download 経路）。**RMVPE は upstream に HF mirror が無い**ため
`run-one.sh` は **使えない**。手動 fallback（§4.1）または vast.ai 上での手動
curl + convert に fall back する。

## 5. §3.1 sign-off status

**現状: blank（fail-closed default）**。

`docs/license-audit.md` §3.1 の RMVPE 行は既 land 済みだが Commercial sign-off は
**空欄**（CLAUDE.md wave 3 の "wave B commits 内で ☑ Commercial 2026-07-30 yousan
(依頼者許可 = CC 判断) 一次資料確認済み" 記述は **row 305（RMVPE 別レコード or
関連 row）** を指し、`vokra/rmvpe` の publish sign-off とは別）。

**Owner action**:

1. **Primary source を直接照合** — https://github.com/Dream-High/RMVPE/blob/main/LICENSE
   （MIT）+ https://github.com/yxlllc/RMVPE/blob/master/LICENSE（同 MIT、fork）
2. yousan として **☑ Commercial** sign-off（`docs/license-audit.md` §3.1 template、
   MIT は Permissive T1）
3. Sign-off 後に §4.1 の `publish-one.sh --push` が gate 通過

**publish-one.sh の 5 gate**（MIT の場合）:

1. Catalog reality — pass
2. Redistributable — pass（MIT は Permissive）
3. Provenance chunk 刻印 — pass（converter が刻む）
4. §3.1 sign-off blank 拒否 — **owner action 必須**
5. T4 非該当 — pass（`--allow-noncommercial` 不要）

## 6. 期待される artifacts

Publish 成功後の `huggingface.co/vokra/rmvpe` repo に含まれる:

| ファイル | 内容 |
|---|---|
| `model.gguf` | ~180 MB（F32 or BF16 pass-through、tensor 数は upstream U-Net + GRU + head に依存） |
| `README.md` | `make_model_card.py` 自動生成、tier T1 obligation + MIT + upstream 情報 |
| `LICENSE` | MIT canonical text（`fetch_license.sh --spdx mit` で取得） |
| `NOTICE` | 空 or attribution only（MIT は NOTICE 必須ではない） |
| `SOURCE.md` | 上流 URL（github release）+ `.pt` → safetensors bridge 手順 + `nemo_pt_to_safetensors.py` 手順 + Vokra converter バージョン + commit SHA |

### GGUF metadata（vokra.rmvpe.* chunk 群、既 converter 実装済）

| Key | 型 | 値 |
|---|---|---|
| `vokra.schema.version` | string | `"1"`（writer choke point で自動刻印） |
| `vokra.schema.producer` | string | `"vokra-cli-<version>"` |
| `vokra.model.arch` | string | `"rmvpe"` |
| `vokra.model.name` | string | `"rmvpe"` |
| `vokra.provenance.upstream_url` | string | `"https://github.com/Dream-High/RMVPE"` or fork URL |
| `vokra.provenance.upstream_license` | string | `"mit"` |
| `vokra.rmvpe.hop` | u32 | `160` |
| `vokra.rmvpe.n_fft` | u32 | `2048` |
| `vokra.rmvpe.win_length` | u32 | `1024` |
| `vokra.rmvpe.n_mels` | u32 | `128` |
| `vokra.rmvpe.sample_rate` | u32 | `16000` |
| `vokra.rmvpe.voiced_threshold` | f32 | `0.03` |
| `vokra.rmvpe.n_classes` | u32 | `360`（20 cents per class） |
| `vokra.rmvpe.base_hz` | f32 | `32.703`（C1） |

## 7. Gate 発火状態（parity CI）

`.github/workflows/parity-rmvpe-real.yml` は **既に owner-driven flip switch で
待機中**（audit 2026-08-10 Rank 8 で land 済）:

```yaml
env:
  VOKRA_RMVPE_REAL_GGUF: ${{ vars.VOKRA_RMVPE_REAL_GGUF_PATH || '' }}
```

Owner が下記を全て満たすと **flip the switch で発火**:

1. **CI variable**: `VOKRA_RMVPE_ENABLE=1` を GitHub repo settings で set
2. **Fixture path variable**: `VOKRA_RMVPE_REAL_GGUF_PATH` を set（CI runner 上
   で fixture を fetch する path、owner が upload 場所を決めて publish or artifact
   store 経由）
3. **`vokra/rmvpe` publish 完了**（§4.1）— CI runner が HF から fetch する場合

**現状の CI 動作 (2026-08-13 更新後)**:

- `VOKRA_RMVPE_ENABLE=1` 未設定 → cron / PR は `::notice::` で clean skip
  （fabricated pass 禁止、FR-EX-08）
- workflow_dispatch で owner が明示的に起動可能
- 起動時に `VOKRA_RMVPE_REAL_GGUF` が空 → harness 側 fixture gate
  で clean skip、Path B 側も 4 env のいずれか未設定なら clean skip
- **Path A**: `extract_real` は real U-Net + BiGRU + head を走らせ、
  shape / finite / sigmoid-range contract を bind — real GGUF を用意した
  瞬間に parity 発火
- **Path B**: `forward_from_hidden` は上流 dumper の hidden.f32 +
  argmax.u32 を受け取り、argmax-match rate ≥ 99 % gate を bind —
  reference dump を用意した瞬間に numerical parity 発火

CI runner に fixture を届ける 3 通り:

1. **GitHub Actions Artifact 経由**（owner 手動 upload、TTL 90 日）
2. **HF に private dataset repo として publish**（`snapshot_download`
   で fetch、CI variable に HF token 追加）
3. **owner 個人 S3 / R2 に置き pre-signed URL を CI variable に**（最短、
   provenance は owner control）

## 8. Owner critical path (2026-08-13 更新後)

**依頼者ルール #3** に従い、以下順序で。**kernel 実装が本 branch e7b6810
で land 済ゆえ CC wave B は不要になった** — owner は下記 6 step で完結:

1. **primary source 確認** — MIT license を GitHub 上で目視確認
   （`https://github.com/yxlllc/RMVPE/blob/master/LICENSE`）
2. **§3.1 sign-off** — `docs/license-audit.md` §3.1 に yousan として
   ☑ Commercial sign（MIT は Permissive T1、既 confirmed）
3. **Fixture 生成** — §2.1 の local pattern（180 MB ゆえ local M1 iMac
   で余裕、vast.ai 起動不要）:
   - `bash tools/parity/rmvpe/fetch_rmvpe_pt.sh --output ~/rmvpe-fixtures/rmvpe.pt`
   - `.pt` → safetensors → GGUF chain (`vokra-cli convert`)
   - `git clone https://github.com/yxlllc/RMVPE.git ~/rmvpe-upstream`
   - `uv run python tools/parity/rmvpe/dump_reference.py --canned ...`
     → `hidden.f32` + `argmax.u32` + `meta.json`
4. **Parity harness で verify** — Path A + Path B 両方の env を export し
   `cargo test -p vokra-models --test parity_rmvpe` が pass
   （Path A: shape / finite / sigmoid-range contract / Path B:
   argmax-match rate ≥ 99 %）
5. **Publish** — §4.1 の `publish-one.sh --push`（5-gate 全通過）
6. **CI flip the switch** — GitHub repo settings で `VOKRA_RMVPE_ENABLE=1`
   + `VOKRA_RMVPE_REAL_GGUF_PATH` + Path B env var 3 個 を set

**CC wave B は close** — 上流 topology が fully-specified と判明した
2026-08-13 feasibility 調査に基づき kernel 実装が本 branch で land 済。
以降 `extract_real` の topology 修正が必要な場合は Path B parity harness
の argmax-match rate が < 99 % に落ちる形で loud-fail する
（silent-wrong risk は Path B の argmax-match gate が catch）。

## 9. Notes

- **`.pt` pickle の security**: PyTorch pickle は arbitrary code execution 可能
  ゆえ、**信頼できる upstream source のみから fetch**（Dream-High / yxlllc の
  GitHub Releases）。`nemo_pt_to_safetensors.py` は fair-use pickle → safetensors
  offline converter で、実行時 sandbox は Python venv のみ = vast.ai instance
  上で実行が最も安全（M1 iMac local は依頼者判断）。
- **Silent-wrong risk の可視化 (2026-08-13 更新後)**: `extract_real` は
  e7b6810 で real U-Net + BiGRU + head を走らせるようになった。上流
  topology drift が起きた場合は Path B の argmax-match rate ≥ 99 % gate
  が catch（`parity_rmvpe.rs:104` `ARGMAX_MATCH_RATE_MIN`）。旧
  `hz=0.0` placeholder + `VokraError::UnsupportedOp` loud pending は
  **resolved** — 現在 `extract_real` は real forward の hz / voiced /
  confidence を返す。
- **RVC v2 との関係**: RMVPE は RVC v2 の必須 pitch 前段。**RVC v2 本体は
  `vokra-voiceclone-experimental` 別リポで扱う**（ELVIS Act 分離ポリシー、CLAUDE.md
  設計判断 8）。RMVPE 単独は `ayutaz/vokra` core に留まる（voice cloning trigger
  でない pitch estimator ゆえ非該当）。CLAUDE.md wave 3 教訓 (c) 参照。
- **FCPE / CREPE との共存**: 同 F0 tier の姉妹 model として FCPE（`vokra/fcpe`）
  と CREPE 5 サイズ（`vokra/crepe-*`）が既 published 済（wave 3 wave B、
  `docs/license-audit.md` row 305/306）。**RMVPE は 3 姉妹の trio の最後**、
  publish 順序は owner 判断（技術的 blocking なし）。

## 関連

- 総論: `docs/handoff/vast-ai-large-model-publish.md`
- CI workflow: `.github/workflows/parity-rmvpe-real.yml`（既 land、gate 待機中）
- Parity harness: `crates/vokra-models/tests/parity_rmvpe.rs`（Path A +
  Path B の 2 leg、fixture-gated skip、no fabricated pass）
- Impl: `crates/vokra-models/src/f0/rmvpe.rs`（**e7b6810 で real U-Net +
  BiGRU forward land、loud-partial resolved**）
- Converter: `crates/vokra-convert/src/models/rmvpe.rs`
- `.pt` → safetensors bridge: `tools/parity/nemo_pt_to_safetensors.py`（fair-use
  pickle converter、DFN3 / Kokoro / Kyutai-STT と共通）
- **Path B reference dumper**: `tools/parity/rmvpe/dump_reference.py` +
  `tools/parity/rmvpe/README.md`（本 branch cca69ba で land、Python
  3.12 uv-managed、上流 nn.Module を fair-use verbatim reference として
  import）
- Sibling F0: FCPE / CREPE（既 published、`docs/license-audit.md` row 305/306）
- Memory: [[feedback-large-models-on-vast-ai]] / [[feedback-license-signoff-primary-source]]
- CLAUDE.md 「現在のタスク状態」residual-wave3 §（RMVPE loud-partial
  判断の evolution: 2026-07-30 defer → 2026-08-13 REVERSED）
