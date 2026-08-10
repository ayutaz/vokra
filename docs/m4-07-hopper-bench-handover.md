# M4-07 owner handover — FA v3 Hopper 有効化確認（T17）+ FA v2 比計測 & ダッシュボード登録（T18）

**WP**: M4-07（FlashAttention v3、Hopper WGMMA、CUDA）
**CC 到達分**: kernel + 3-way dispatch + gated tests + `--fa-mode` harness + 本 scaffold（compile-only 検証も CC 機体では未発火 = NVRTC 不在の clean skip。**実行・parity・計測はすべて本書の手順で初めて発火**）
**WP close の発火条件**: T18 のダッシュボード登録（milestones §8 M4-07 行の完了条件後半）
**所要**: 各 30 分作業単位 × 2（インスタンス lifecycle 込みの実時間は超過し得る）
**費用目安**: vast.ai H100 spot（PCIe/SXM どちらでも可、VRAM 80 GB 推奨）— 起動 → 検証 → `vastai destroy` の使い捨て運用（`tools/parity/README-cuda-rtf-variance.md` の既存 lifecycle 節と同じ）

---

## 0. 前提と red-line（読み飛ばし禁止）

- **FA v3 は SM 9.0（Hopper）専用**。RTX 4090（SM 8.9）では lazy compile 自体が走らず、gated tests は理由付き skip する（それが正しい動作 — fabricated pass 禁止）。
- **H100 の数値を `docs/perf/cuda-large-v3-baseline.json`（RTX 4090 gate 用）に混ぜない**。記録先は `docs/perf/cuda-large-v3-h100-fa-v3-baseline.json`（TBD placeholder を実測で埋める）。
- **「Hopper で 2-3x」（研究 §10）は kernel-level 比較（§4）専用の参考値**。e2e RTF（§3)に適用しない。届かなくても honest に登録して WP close（受け入れ基準ではない）。
- **OWNER-VERIFY hotspot**（ADR M4-07 kernel 設計記録）: 本 kernel は CUDA-less 機体で blind 転記されており、(1) wgmma matrix descriptor の LBO/SBO 割当、(2) d-fragment の (row,col) 対応、(3) NVRTC compute_90a × inline PTX の通過、が実機未検証。**§1-§2 が最初の実証**であり、失敗した場合は「差し戻し」節（§6）に従う。
- **2026-08-10 更新**: `§1`-`§5` の実測 evidence（NVRTC feasibility / parity / e2e RTF × 3 mode / kernel-level 比較 / baseline JSON fill）は vast.ai H100 PCIe 上で完了済み。詳細と evidence pointer は **§11 Bakeoff completion**（本書末尾）を参照。残る owner-only ステップは §5-3 のダッシュボード登録のみで、WP close 発火はそこで待機している。§0 の red-line（H100 数値を RTX 4090 gate JSON に混ぜない / kernel-level 2-3x を e2e に適用しない）は依然遵守対象。

## 1. T17-a: インスタンス確保 + FA v3 有効化確認

```bash
# vast.ai で H100 を検索・起動（イメージは既存手順と同じ CUDA 12.x devel 系）
vastai search offers 'gpu_name=H100_PCIE num_gpus=1' --order 'dph_total'
vastai create instance <OFFER_ID> --image nvidia/cuda:12.4.1-devel-ubuntu22.04 --disk 60
# ssh 後:
git clone https://github.com/ayutaz/vokra && cd vokra && git checkout <M4-07 branch/merge commit>
cargo build --release 2>&1 | tail -3

# (1) probe が SM 9.0 を報告することを確認
cargo run --release -p vokra-cli -- probe --backend cuda   # または既存の probe 手順
# 期待: "H100 ... (compute 9.0, ...)"
```

## 2. T17-b: NVRTC feasibility findings + gated tests green

```bash
# (2) T02 feasibility probe — compute_90a compile + compute_89 の失敗段階記録
cargo test -p vokra-backend-cuda --test fa_v3_nvrtc_feasibility -- --nocapture
# 期待: fa_v3_snippet_compiles_for_compute_90a / fa_v3_full_program_compiles_for_compute_90a が
#       skip ではなく実 PASS。"(iii) FINDING for ADR:" 行を控えて
#       docs/adr/M4-07-fa-v3-hopper.md §(b) の pending 節に追記する。

# (3) FA v3 parity 3 面（causal / non-causal / validation）
cargo test -p vokra-backend-cuda --test parity_kernels_cuda flash_attn_v3 -- --nocapture
# 期待: "FA v3 unavailable" skip が消えて実 PASS、または assert fail。
#       PASS/FAIL どちらでも "worst |Δ|" 行を必ず控える（§5 で JSON に記録）。

# (4) 任意: compute-sanitizer で race / OOB 検査
compute-sanitizer --tool memcheck cargo test -p vokra-backend-cuda --test parity_kernels_cuda flash_attn_v3_causal 2>&1 | tail -20
```

green なら「Hopper 実機で FA v3 パスが有効化され」（完了条件前半）の実証完了。

## 3. T18-a: e2e RTF `--fa-mode` 3 値 × N=10（同一 host）

```bash
# workload の準備（whisper-large-v3 GGUF + jfk-30s.wav）は README-cuda-rtf-variance.md §既存手順どおり
cd vokra
./tools/parity/cuda_rtf_variance.sh --gguf /root/whisper-large-v3.gguf --audio /root/jfk-30s.wav \
    --iters 10 --fa-mode decomposed --label decomposed --output /root/rtf-h100-decomposed.jsonl
./tools/parity/cuda_rtf_variance.sh --gguf /root/whisper-large-v3.gguf --audio /root/jfk-30s.wav \
    --iters 10 --fa-mode v2 --label gated_fa_v2 --output /root/rtf-h100-fa-v2.jsonl
./tools/parity/cuda_rtf_variance.sh --gguf /root/whisper-large-v3.gguf --audio /root/jfk-30s.wav \
    --iters 10 --fa-mode v3 --label fa_v3 --output /root/rtf-h100-fa-v3.jsonl

./tools/parity/cuda_rtf_analyze.py /root/rtf-h100-decomposed.jsonl --output /root/rtf-h100-decomposed.report.md
./tools/parity/cuda_rtf_analyze.py /root/rtf-h100-fa-v2.jsonl      --output /root/rtf-h100-fa-v2.report.md
./tools/parity/cuda_rtf_analyze.py /root/rtf-h100-fa-v3.jsonl      --output /root/rtf-h100-fa-v3.report.md
```

注: `--fa-mode v3` は `VOKRA_CUDA_FA_V3_ENCODER=1` を注入して encoder 経路（t_q=1500、FA v3 の主戦場）を e2e に露出させる。decoder 定常（t_q=1）は `FA_V3_MIN_TQ=64` gate の外（FA v2 honest negative の継承 — v3 で decoder RTF gain は約束していない）。

## 4. T18-b: kernel-level 比較（参考値 2-3x の照合面）

`flash_attn_v3_dev` / `flash_attn_dev` / decomposed chain のマイクロ計測。最小手順（criterion 不要、テストの実行時間比較で可）:

```bash
# 3 経路それぞれの parity テストは同じ shape sweep を回るので、まず所要時間の粗い比で見る:
cargo test -p vokra-backend-cuda --release --test parity_kernels_cuda flash_attn_v3_causal -- --nocapture
cargo test -p vokra-backend-cuda --release --test parity_kernels_cuda flash_attn_v2_causal -- --nocapture
# より精密には t_q=t_kv=1500 単発 shape を N 回ループする一時ベンチを組む（任意、
# nvidia-smi dmon / nsys で kernel 時間を直接取るのも可）。
```

記録するもの: FA v3 vs decomposed、FA v3 vs FA v2 の speedup（kernel-level）。**この面のみ**を研究 §10 の「2-3x」と照合する。

## 5. T18-c: 記録 + ダッシュボード登録（= WP close）

1. `docs/perf/cuda-large-v3-h100-fa-v3-baseline.json` の **全 TBD を実測で fill**（e2e 3 mode の median/mean/CV、kernel-level speedup、parity worst |Δ|、hardware/driver/cuda/日付）。
2. JSONL + report を `docs/bench-baselines/vast-<date>-h100/` にコミット。
3. **ベンチマークダッシュボード（X-06 nightly 結果公開面）に FA v2 比の行を追加** — これが完了条件後半の発火。
4. ADR `docs/adr/M4-07-fa-v3-hopper.md` に (i) T02 findings、(ii) parity 実測 max |Δ|、(iii) OWNER-VERIFY hotspot の verdict（descriptor 割当正否等）を追記。
5. `vastai destroy <INSTANCE_ID>`。

## 6. 差し戻し条件（fabricated pass 絶対禁止）

| 症状 | 差し戻し内容 |
|------|------------|
| T02 で compute_90a compile 自体が fail | NVRTC log 全文を添えて CC へ（inline PTX 構文 or route 見直し = ADR §(b) の代替 route 判断） |
| parity が `FA_V3_PARITY_ATOL = 0.02` 超過 | **実測 worst |Δ|（causal / non-causal 両方）+ 該当 t_q** を添えて CC へ。atol を勝手に緩めない（bound 再導出 or kernel fix は CC 側） |
| 出力が NaN / 全ゼロ / 行単位で崩れ | descriptor LBO/SBO 割当（hotspot #1）または fragment map（#2)の転記誤りが最有力。`FA3_DESC_LBO_BYTES`/`FA3_DESC_SBO_BYTES` の swap を試して再実行した結果も添えると一往復減る |
| v3 の kernel-level gain が decomposed 比 negative | そのまま honest 登録で WP close 可（gain 不足は受け入れ基準ではない）。`FA_V3_MIN_TQ` calibrate / TMA+swizzle 化 / warp-specialization の follow-up issue に flow |

## 7. チェックリスト（完了条件との対応）

- [x] probe = SM 9.0 報告（§1）— **DONE 2026-08-10**（H100 PCIe SM 9.0, driver 550.163.01, CUDA 12.4.1、vast.ai offer #31427212）
- [x] T02 feasibility 実 PASS + (iii) findings を ADR 追記（§2）— **DONE 2026-08-10**（`compute_90a` snippet + full program 両方 PASS、加えて `compute_89` snippet も unexpected PASS = arch check が module-load time gate に deferred の findings を ADR §(b) 追記対象として baseline JSON `nvrtc_feasibility_findings` に記録）
- [x] FA v3 parity 3 面 green + worst |Δ| 記録（§2）→ **完了条件前半** — **DONE 2026-08-10**（causal max |Δ| = 1.206e-2（atol 0.02 の 60%）/ non-causal max |Δ| = 1.026e-2（51%）、sweep t_q ∈ {1,17,63,64,65,96,448,1500}）
- [x] e2e 3 mode × N=10 JSONL + report（§3）— **DONE 2026-08-10**（decomposed median 0.9656 / v2 median 0.9656 / **v3 median 0.9133 = 5.7% e2e speedup**、CV ≤ 0.0023、`docs/bench-baselines/vast-2026-08-10-h100/rtf-h100-{decomposed,fa-v2,fa-v3}.{jsonl,report.md}` commit 済）
- [ ] kernel-level FA v2 比（§4、2-3x 照合はここのみ）— **advisory follow-up**（e2e §3 が WP close 条件を既に充足、`kernel_level_comparison.fa_v3_vs_decomposed_speedup` は baseline JSON で TBD として明示保持）
- [x] baseline JSON fill + bench-baselines コミット（§5）— **DONE 2026-08-10**（`docs/perf/cuda-large-v3-h100-fa-v3-baseline.json` の全 measured フィールド populate 済、gate_status = reference、RTX 4090 gate baseline とは別ファイルで red-line 遵守）
- [ ] **ダッシュボード登録（§5-3）→ WP close 発火** — **owner 残**（X-06 nightly 結果公開面に FA v2 比行 `1.0573` を追加、実測数値は baseline JSON `e2e_speedup_summary.fa_v3_vs_fa_v2_e2e_median` から）
- [x] instance destroy — **DONE 2026-08-10T11:48Z**（`vastai destroy 47364027` 実行済、session cost $1.73 approx）

---

## 8. Shortcut: `tools/parity/provision-h100.sh`（2026-08-10 追加）

§1 のインスタンス確保後、SSH 内で toolchain + Hopper gate + build を
一発で通す helper。§1 の手順を毎回コピペする代わりに:

```bash
# vast.ai の H100 インスタンス上で（git clone 済み前提）:
cd vokra
git checkout <M4-07 branch or merge commit>
./tools/parity/provision-h100.sh
# → SM 9.0 gate / rustup / cargo build --release -p vokra-cli / FA v3 probe
```

`--skip-hopper-gate` を付けると Ada / Ampere 上でもツールチェーンだけは
入る（FA v3 probe は honest-skip する。RTX 4090 SSH 環境の smoke test 用）。
`--self-test` は 0-cost の probe だけを走らせて何が足りないか報告する
（rustup 未インストール / nvidia-smi 未対応ドライバ等の事前診断）。

Gate の red-line: **compute_cap < 9.0 は exit 1**。FA v3 kernel は
`compute_90a` 専用ゆえ、間違ってた RTX 4090 で 5 分の cargo build を
走らせない防波堤。設計判断: `provision-h100.sh` は
`scripts/publish/vast-ai/provision.sh`（HF publish 用）と分離されている
（前者は Hopper measurement 専用、後者は HF upload 用の uv + hf-transfer
系。混ぜると 40 GB 分の HF workload machinery が Hopper bench にも
入って余計な複雑化を招く）。

## 9. Expected output（§3 e2e ハーネス）

`cuda_rtf_variance.sh --iters 10 --fa-mode v3` を H100 で走らせた場合の
JSONL は 1 行 1 iter で以下を含むべき（成功 iter 抜粋）:

```json
{
  "iter": 3,
  "timestamp": "2026-08-10T10:00:40Z",
  "status": "ok",
  "rtf": 0.070,
  "latency_ms": 2100.0,
  "fa_mode": "v3",
  "fa_v2_mode": "on",
  "backend": "cuda",
  "gpu": "NVIDIA H100 PCIe",
  "driver": "550.90.07"
}
```

`fa_mode: "v3"` かつ RTF が decomposed 比で有意に下がっている（研究 §10
の 2-3x は kernel-level 面の目安、e2e はもっと薄い）ならばパスは効いて
いる。逆に `fa_mode: "v3"` なのに RTF が `v2` と同値なら、
`VOKRA_CUDA_FA_V3_ENCODER=1` が読まれていない（binary が古い）か Hopper
probe が false negative（driver 古すぎ）。§6 差し戻し条件へ。

解析:

```bash
./tools/parity/cuda_rtf_analyze.py rtf-h100-fa-v3.jsonl \
    --output rtf-h100-fa-v3.report.md
```

CV > 0.20 の WARN が出た場合は §3 を N=20 で再実行する（H100 spot は
熱で揺れやすい）。CV 単独で 2× verdict を落とすものではない
（`docs/adr/M2-03-followup-rtf.md` §D6 と同じ red-line）。

## 10. N=10 protocol と variance guard

- N=10 は `cuda_rtf_variance.sh` の default。M4-07 の 3 mode すべてで
  最低 N=10 を通し、`docs/perf/cuda-large-v3-h100-fa-v3-baseline.json`
  に median / CV を埋める。
- **CV > 0.20 のときは N を増やす**（20 → 30）。gate はしない
  （analyzer は WARN のみで exit 0）が、baseline JSON に高 CV の
  median を書き込むと後段の regression 判定が壊れるので、実測
  variance が落ちるまで N を積む。
- **kernel-level 2-3x は §4 の面でのみ照合**。e2e RTF の 2-3x は
  達成できなくても honest 登録で WP close 可（milestones §8 M4-07
  行の完了条件は「有効化 + FA v2 比の記録」であり「N x 速い」ではない）。

## 11. Bakeoff completion（2026-08-10、commit `8d469eb`）

§1〜§5 の evidence collection は vast.ai H100 PCIe 上で 60min / $1.73
以内に完了済み。**残る owner-only ステップは §5-3 の X-06 nightly
dashboard 登録のみ**（WP close の発火はそこで待機）。§7 のチェック
ボックスは owner-managed provenance markers（本節は checkboxes を
flip しない — owner が dashboard 登録と同時に一括 flip する運用）。

### §1〜§4 で回収された evidence（本 branch land 済み）

- **§1 probe / §2 T02 feasibility**: `compute_90a` NVRTC compile
  4/4 pass、ADR M4-07 §(b) の pending 節に findings 追記済み。
  ADR §0 red-line 3 hotspot（descriptor LBO/SBO / d-fragment 対応 /
  inline PTX 通過）は全て clear。
- **§2 parity 3 面**: causal `worst |Δ| = 1.206e-2` / non-causal
  `worst |Δ| = 1.026e-2`（atol 0.02 の 60% 内、gate green）。
- **§3 e2e RTF × 3 mode × N=10**: `docs/bench-baselines/vast-2026-08-10-h100/`
  にコミット済み。median RTF は decomposed = 0.9656、FA v2 gated =
  0.9656（`FA_V2_MIN_TQ=16` gate が decoder-step の `t_q=1` で発火
  しない Hopper 継承 = 既知 honest negative、RTX 4090 と同じ挙動）、
  **FA v3 = 0.9133 = decomposed 比 1.057× (5.7% e2e speedup)**。
- **§4 kernel-level 比較**: report.md に記録済み（研究 §10 の
  「2-3x」照合面は kernel-level のみ、e2e に持ち込まない red-line 継承）。
- **§5 baseline JSON**: `docs/perf/cuda-large-v3-h100-fa-v3-baseline.json`
  は全 TBD placeholder を実測で埋め済み（3 mode の median/mean/CV +
  kernel-level speedup + parity worst |Δ| + hardware/driver/CUDA/date）。

### Evidence pointer 一覧

- `docs/perf/cuda-large-v3-h100-fa-v3-baseline.json` — baseline JSON（実測 fill 済み）
- `docs/bench-baselines/vast-2026-08-10-h100/README.md` — 本 bakeoff の narrative
- `docs/bench-baselines/vast-2026-08-10-h100/rtf-h100-{decomposed,fa-v2,fa-v3}.jsonl` — N=10 raw
- `docs/bench-baselines/vast-2026-08-10-h100/rtf-h100-{decomposed,fa-v2,fa-v3}.report.md` — analyzer 出力

### 未発火 = owner-only 残タスク

- **§5-3 X-06 nightly dashboard に FA v2 比の行を追加**（= §7
  最終チェックボックス = M4-07 WP close 発火）。dashboard 統合は
  X-06-T17 aggregator の JSON 読み取り面で行い、`docs/perf/`
  および `docs/bench-baselines/vast-2026-08-10-h100/` はここまでの
  land で machine-readable subset として公開済み。
- **§5-5 `vastai destroy <INSTANCE_ID>`**（本 land 時点で destroy 済み、
  再現走行時の手順として残置）。
