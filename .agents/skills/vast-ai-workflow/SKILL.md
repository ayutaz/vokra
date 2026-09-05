---
name: vast-ai-workflow
description: メモリを食う作業を vast.ai へ逃がすときに使う。**≥2 GB のモデル artefact の変換 / 検証 / publish**、および **workspace 全体や `vokra-models` の cargo**（2026-08-16 に M1 iMac を OOM で再起動させた実績）が対象。`provision.sh` の 4 gotcha（hf_config.pth shim / huggingface_hub pre-0.30 pin / certifi / stack tool install）+ rent→provision→work→destroy lifecycle + 未 push コミットの git bundle 転送 + Voxtral streaming reader + FA v3 Hopper bakeoff の runbook を示す。
---

# vast.ai で大モデルを扱う

**単一事実源**: `scripts/publish/vast-ai/provision.sh` + `docs/handoff/vast-ai-large-model-publish.md`。本 skill はそれを skill 表現に翻訳したもの。

**大原則**: 依頼者機は **M1 iMac 16 GB**。ここでメモリを食う作業を走らせると swap が急伸し、**macOS が強制終了する**。実証は 2 系統:

- **モデル artefact**: Voxtral-Small-24B (48 GB) の mmap で swap 40 GB 到達
- **cargo**: 2026-08-16、`cargo test -p vokra-models --lib kyutai_stt` が exit 137 (OOM kill) → **macOS 再起動**

→ どちらも **vast.ai へ escalate**。

## 0. いつ vast.ai を使うか

**閾値は 2 GB**（依頼者指示 2026-08-16）。実測 safe な範囲（whisper-* 2.9 GB / csm-1b 6.21 GB は変換実績あり）より意図的に厳しい。**誤ってブロックした代償は環境変数 1 個、誤って通した代償は再起動**だから。

**要 vast.ai**:
- **≥2 GB のモデル artefact の convert**（sharded の場合は shard 単体でなく**合計**で判定）
- ≥2 GB GGUF の publish（30 Mbps outbound で数時間、vast.ai の 2.6 Gbps で数分）
- **workspace 全体の cargo**（`--workspace` / `--all` / `-p` 無しの `cargo test`）
- **`-p vokra-models` の cargo**（最大 crate、2026-08-16 の再起動の直接原因）
- H100 / A100 が必要な bakeoff（FA v3 Hopper、CoreML/ANE は別 = 実機 Mac / iPhone）

**M1 iMac で OK**:
- 軽い crate 単体: `-p vokra-convert` / `-cli` / `-eval` / `-core` / `-ops`（`CARGO_BUILD_JOBS=1` 併用）
- シェルゲート全般（`scripts/check-*.sh`）、`cargo fmt`、`cargo metadata`
- **restamp_provenance 経路**: **8.7 GB Voxtral を peak 6.4 MB で publish 実績あり**（mmap 読取のみ、tensor コピーなし。→ skill `publish-model-to-hf` §7）。**tensor を触らず provenance だけ差し替えるなら 2 GB 閾値の例外**

**強制されている**: `.codex/hooks/guard-local-memory.sh` が PreToolUse フックとして上記を**ブロック**する（`.codex/hooks.json` に登録済み、`--self-test` 43 ケース）。`.githooks/pre-push` も maintainer Mac の deep Cargo path を開始前に拒否する。意図的に通す場合は、その1回を依頼者が明示承認したときだけ `VOKRA_ALLOW_LOCAL_HEAVY=1` を前置する。

[[feedback-large-models-on-vast-ai]] / [[feedback-no-local-workspace-cargo]]。

## 1. Lifecycle（4 phase）

1. **rent**: vast.ai 上で GPU instance を借りる（cheapest でも RAM ≥64 GB / disk ≥200 GB は必須、convert 用途なら GPU は 4090 で十分、H100 は FA v3 bench 用）
2. **provision**: `scripts/publish/vast-ai/provision.sh` を SSH 上で実行（4 gotcha を pre-handle）
3. **work**: `run-one.sh` per model or 直接 cargo コマンド
4. **destroy**: **必ず `vastai-safe.sh destroy instance <instance-id>` で auto-destroy**（走らせっぱなしは $/h で課金継続、ADR §D6）。ただし、直近に再開することが明示された retained handoff（たとえば別環境への転送待ち）に限り、一時的な `stop` を許可できる。Stop は compute 課金を止めてデータを保持するが storage 課金は継続し、再開時の GPU 確保は保証されない。重要データは外部にも backup し、handoff 完了後は必ず destroy する。

### 一時停止の限定例外（retained handoff のみ）

通常原則は **work 完了後に destroy** であり、アイドル時間の節約を理由に
借りっぱなしの instance を一般的に stop してよい、という意味ではない。次の
条件をすべて満たす場合だけ、直近の再開予定が明示された retained handoff
として stop を選べる:

- 保持するデータと再開目的（例: Scaleway への転送待ち）が記録されている。
- 重要データを外部 backup 済みである（stop は backup の代替ではない）。
- stop 後も storage 課金が続くこと、再開時に GPU を確保できないリスクを了承する。
- 転送・再開・検証が完了したら、直ちに `destroy` する。

Vast の公式仕様では、Stop は compute 課金を止めてデータを保持しますが、
storage 課金は継続し、再開時の GPU 確保は保証されません。Destroy はデータを
削除して課金を停止します。詳細は [Manage instances](https://docs.vast.ai/guides/instances/manage-instances)
と [Storage types](https://docs.vast.ai/guides/instances/storage/types) を参照してください。

**retained handoff の現状 (2026-09-01)**: owner は Scaleway 実行までの待機が
長期化するため、instance `49168183` (`vokra-mac-coverage-771970dc`,
500 GB storage) と `49261078` (`vokra-htdemucs-inspection-20260830`,
200 GB storage) を保存データごと destroy する方針へ変更した。両方とも
`vastai-safe.sh destroy instance <id> -y` の後、個別 API が
`instances: null` を返した。`apple-transfer-bc9d1db2`、
`apple-transfer-reazon-a59c48c8`、`apple-transfer-bicodec-5cd97d12` は
VAST 上に存在しない。現在 Vokra 用の retained handoff はない。Apple 実機
検証を再開するときは、固定 revision/hash 契約から VAST で artefact と
reference packet を再生成し、新しい disposable instance から直接転送する。
旧2件を restart target として扱わない。

## 2. Rent phase（vast.ai 側）

```bash
# 手元 CLI（vastai）で GPU offer 検索
scripts/publish/vast-ai/vastai-safe.sh search offers 'gpu_name=RTX_4090 rentable=true' --order 'dph_total'
# ↑ 一番安いのを選ぶ

scripts/publish/vast-ai/vastai-safe.sh create instance <offer-id> \
  --image nvidia/cuda:12.4.1-devel-ubuntu22.04 \
  --disk 200 \
  --ssh
```

ローカルから Vast CLI を呼ぶときは、必ず
`scripts/publish/vast-ai/vastai-safe.sh` を経由する。ラッパーは stdout と
stderr の URL クエリに含まれる `api_key` 等の資格情報値を
`[REDACTED]` に置換し、CLI 本来の終了コードを返す。`VASTAI_BIN` を設定すれば
固定した CLI パスやオフラインのテストコマンドを指定できる。

**image 選定注意**: `nvidia/cuda:13.0.0` の stock image は 4 個の gotcha を含む（下記 §3）→ `12.4.1-devel-ubuntu22.04` が実績 stable。H100 bakeoff で 13.0.0 が必要な場合も `provision.sh` の `harden_vast_docker_image` が pre-handle する。

## 3. Provision phase — 4 gotcha を pre-handle

vast.ai の stock image が持つ 4 個の trap を `provision.sh` が **`install_uv` より前**に潰す。個別に対処するのは実績上 1 day 溶かす:

| # | Gotcha | 症状 | Fix |
|---|--------|------|-----|
| A | **hf_config.pth site-packages shim** | Python startup shim が `HF_ENDPOINT` を malicious mirror `117.175.104.83:8081` に書換、全 large DL が 404 | `/usr/local/lib/python3.10/dist-packages/hf_config.pth` + `/usr/lib/python3/dist-packages/pip/_vendor/hf_config.pth` を削除 |
| B | **huggingface_hub >= 0.30 non-xet regression** | xet-token routing が mirror 404 を投げる、`HF_HUB_DISABLE_XET` も一部 bypass | `provision.sh` の bootstrap-only system repair で pre-0.30 に pin（手動 package 操作は禁止、task Python は uv） |
| C | **certifi CA bundle 空/stale** | HTTPS 全般 fail | `certifi` を再植え付け |
| D | **torch/numpy/safetensors が system layer に無い** | stock image の bare-Python fallback や `uv_cmd` fallback が missing modules で死ぬ | pre-install で torch + numpy + safetensors を system-level に置く |

**呼び方**:
```bash
# vast.ai instance に SSH した後
export HF_TOKEN='hf_xxxxxxxx'
git clone https://github.com/ayutaz/vokra.git ~/vokra
cd ~/vokra
bash scripts/publish/vast-ai/provision.sh  # idempotent, rerun-safe
```

**idempotent**: rerun-safe。各 step が artifact を probe して skip する。`git pull` 後にも安全に再実行できる。

[[reference-vast-ai-hf-config-pth-shim]] / [[reference-huggingface-hub-lt-030-vast-ai]]。

## 4. Work phase

### 4.1 Convert + stage 1 モデル（publish は別承認）

```bash
# provision 済 instance 上
export VOKRA_PUBLISH_ON_VAST=1  # publish-one.sh gate 7 の implicit bypass
scripts/publish/vast-ai/run-one.sh \
  --hf-repo <upstream/repo> \
  --vokra-slug <slug> \
  --model-kind <kind> \
  --license-spdx <spdx>  # dry-run: convert → stage
```

`--push` は dry-run が green で、かつ依頼者がその repo / artifact の upload を明示承認した場合だけ追加する。VAST 作業の指示だけから HF upload 権限を推定しない。

### 4.2 Voxtral streaming reader パターン（sharded safetensors、mmap 節約）

- 通常は sharded safetensors を `tools/parity/<slug>_prepare_checkpoint.py` で事前 merge（→ skill `add-speech-model` §2.1）
- **Voxtral だけは例外**: TextDecoder の `Vec<f32>` eager binding が ~15 GB 要求で M1 を殺していた root cause → `MappedTextBlocks` / `MappedHeads`（mmap + tiled transpose、lm_head streaming）で **peak 15 GB → 3.55 GB**。同じ pattern を他 sharded モデルに横展開する場合は先例として参照
- 実装: `crates/vokra-models/src/voxtral/mapped_lazy.rs` 系。streaming 適用可能なモデルは事前 merge 不要（converter が sharded を直接読む）

### 4.3 provenance-only の低メモリ経路

```bash
# 大 GGUF を convert する必要がなく、既存 HF から DL → restamp → repush だけの場合
# ローカル (M1 iMac) の restamp_provenance で peak 6.4 MB で完結する
# → vast.ai を借りずに済む（Voxtral 8.7 GB で実証）
```

**vast.ai を借りる vs 借りない判断**: convert が要る = 借りる / provenance だけ = ローカル restamp。

### 4.4 workspace 検証を逃がす（GPU も provision.sh も不要）

`cargo test --workspace` をローカルで走らせないための実績レシピ（2026-08-16、**所要 ~25 分 / $0.03**）。**モデル変換ではないので GPU も HF token も `provision.sh` も要らない** — 必要なのは CPU と RAM だけ。

```bash
# 1. RAM で選ぶ（GPU 性能は無関係）。48 core / 125 GB が $0.082/hr だった
scripts/publish/vast-ai/vastai-safe.sh search offers 'rentable=true cpu_ram>=64 disk_space>=150 num_gpus=1' \
  --order 'dph_total' --raw | UV_CACHE_DIR="${TMPDIR:-/tmp}/vokra-uv-cache" \
  uv run --no-project --python 3.12 python -c "
import json,sys
for o in json.load(sys.stdin)[:10]:
    print(o['id'], o['dph_total'], o['cpu_cores_effective'], int(o['cpu_ram']/1024))"

scripts/publish/vast-ai/vastai-safe.sh create instance <offer-id> --image nvidia/cuda:12.4.1-devel-ubuntu22.04 \
  --disk 150 --ssh --direct --label vokra-verify

# 2. 接続は direct endpoint を使う（public_ipaddr + direct_port_start）。
#    ssh_host/ssh_port 側は Connection refused になることがある
scripts/publish/vast-ai/vastai-safe.sh show instance <id> --raw | UV_CACHE_DIR="${TMPDIR:-/tmp}/vokra-uv-cache" \
  uv run --no-project --python 3.12 python -c "
import json,sys; d=json.load(sys.stdin); print(d['public_ipaddr'], d['direct_port_start'])"

# 3. Rust。--profile minimal は rustfmt/clippy を入れないので明示追加が要る
curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
rustup component add rustfmt clippy
```

**未 push のコミットは push せずに bundle で渡す**（未検証コミットを remote branch に載せないため）:

```bash
# 手元
git bundle create /tmp/wave.bundle <remote-tip>..HEAD   # docs 数本なら 20 KB 程度
scp -P <port> /tmp/wave.bundle root@<ip>:/root/

# vast.ai 側
git clone -q --branch <branch> https://github.com/ayutaz/vokra.git vokra && cd vokra
git fetch -q /root/wave.bundle HEAD:work && git checkout -q work
git rev-parse --short HEAD    # 手元と一致することを必ず確認
```

**判定に使える実測値**: フル workspace = `6965 passed / 0 failed / 23 ignored / 234 suites`。ローカル並列実行で落ちる 2 件は **regression ではない** — `kyutai_stt` は正当に **155 秒**かかるため 180 秒タイムアウトに接触し、`csm_frame_loop_allocates_zero_after_open` は alloc カウンタが他スレッドに撹乱される。どちらも単独・大容量機では通る。

検証が green なら、手元からのコード push は pre-push の重い経路を踏まないよう `VOKRA_SKIP_HOOKS=1` を使う。**無検証のまま bypass しないこと** — 根拠はリモート検証結果。remote branch の削除-only push は compliance 回帰テストを実行した後、Cargo leg を自動 skip する。

**2026-08-18 incident**: 旧 remote branch の deletion-only push が空 diff と判定され、修正前の pre-push が workspace Cargo を起動した。process group を停止して残存 process がないことを確認し、`VOKRA_SKIP_HOOKS=1` で削除を完了した。現在は stdin の local SHA が全 zero の ref update だけを deletion-only と認識し、mixed/normal/malformed update は従来どおり検査へ送る。回帰テストは `scripts/test-pre-push-fastpath.sh`。

## 5. Destroy phase（**必ず**）

```bash
# 手元 CLI から
scripts/publish/vast-ai/vastai-safe.sh destroy instance <instance-id>
```

ラッパーの契約（資格情報クエリの redaction、通常出力、終了コード保持）は
ネットワークなしで確認できる:

```bash
scripts/publish/vast-ai/test-vastai-safe.sh
```

**auto-destroy を仕込む**: ローカルの lifecycle controller で、measure が終わったら trap で自動 destroy する pattern（ADR §D6）。Vast ホスト上ではローカル CLI/credential を前提にしない:
```bash
# 手元の lifecycle controller 内（VAST_CONTAINERLABEL は instance id）
trap 'scripts/publish/vast-ai/vastai-safe.sh destroy instance "$VAST_CONTAINERLABEL"' EXIT
# ↑ ローカル controller 終了時に自動 destroy
```

**destroy 忘れは $/h で継続課金**。H100 は $1.7-2.5/h、8h 忘れると $15+ 溶ける。

## 6. FA v3 Hopper bakeoff runbook（M4-07 実績）

- **hardware**: H100 PCIe (SM 9.0, 80 GB VRAM)
- **image**: nvidia/cuda:12.4.1-devel-ubuntu22.04
- **session cost 実績**: $1.73 / 60 min（2026-08-10）
- **手順**: `docs/m4-07-hopper-bench-handover.md` §3 / `provision.sh` + `cargo build --release -p vokra-cli -p vokra-backend-cuda` + `tools/parity/cuda_rtf_variance.sh --iters 10 --fa-mode {decomposed,v2,v3}`
- **結果**: FA v3 で **5.7% e2e speedup** vs FA v2、causal parity 1.206e-2 / non-causal 1.026e-2（atol 0.02 内 60% / 51%）
- **注意**: `provision.sh` の `crates/vokra-cli` sanity check が glob-based workspace で misfire するので、直接 `cargo build --release -p vokra-cli` を叩く

## 7. 費用 vs 手間の判断

| Task | vast.ai 費用目安 | 判断 |
|------|--------------|------|
| Voxtral-Small-24B convert + publish | RTX 4090 8h × $0.30/h = $2.4 | 妥当（M1 で試すと mac 強制終了 = 復旧に時間 loss） |
| H100 FA v3 bakeoff | H100 60 min × $1.73/h = $1.73 | 妥当（M4-07 実績） |
| **workspace 全体の cargo 検証** | **21 min × $0.082/h = $0.03**（実績） | **妥当**（ローカルは再起動、復旧コストが桁違い） |
| provenance だけの差替 | $0（ローカル restamp、tensor 不読み） | **借りない** |
| ≥2 GB checkpoint の convert | 借りる | 実測 safe でも**閾値どおり借りる**（2026-08-16 依頼者指示） |

## 8. 出禁パターン

- **「今回くらいはローカルで」**: これが 2026-08-16 に mac を再起動させた。現在の閾値は artefact 合計 2 GB、Cargo は workspace 全体または `-p vokra-models`。**判断で防げなかったので hook で強制した** = `guard-local-memory.sh`。`VOKRA_ALLOW_LOCAL_HEAVY=1` は依頼者がその1回を明示承認した場合だけ使う
- **未検証のまま `VOKRA_SKIP_HOOKS=1` で push**: bypass の根拠はリモート検証結果であって、急いでいることではない
- **vast.ai を借りっぱなしで放置**: $/h 課金継続、trap `vastai-safe.sh destroy instance <instance-id>` を必ず仕込む
- **provision.sh を skip して pip 手打ちで頑張る**: 4 gotcha に順番にハマる（実績 1 day 溶かす）
- **`huggingface_hub` を local と同じ最新版で使う**: vast.ai 上では <0.30 pin 必須（xet-token regression）、local との差分を明示的に持つ
- **`HF_TOKEN` を CLI 引数で渡す**: shell history + `ps` に残る → 環境変数経由で
- **borrowed instance で `.env` を書く**: destroy で消える + credential 漏洩リスク、環境変数を SSH セッション内で export

## 9. Cross-reference

- skill `publish-model-to-hf` — publish gate 5 段の詳細
- skill `add-speech-model` — sharded safetensors 事前 merge / GGUF 5D limit
- `scripts/publish/vast-ai/provision.sh` — 実装 (single source of truth)
- `docs/handoff/vast-ai-large-model-publish.md` — 実運用 runbook
- `docs/m4-07-hopper-bench-handover.md` — H100 bakeoff 実績
- `docs/bench-baselines/vast-2026-08-10-h100/` — H100 measurement data
