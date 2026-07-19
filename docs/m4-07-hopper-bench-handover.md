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

- [ ] probe = SM 9.0 報告（§1）
- [ ] T02 feasibility 実 PASS + (iii) findings を ADR 追記（§2）
- [ ] FA v3 parity 3 面 green + worst |Δ| 記録（§2）→ **完了条件前半**
- [ ] e2e 3 mode × N=10 JSONL + report（§3）
- [ ] kernel-level FA v2 比（§4、2-3x 照合はここのみ）
- [ ] baseline JSON fill + bench-baselines コミット（§5）
- [ ] **ダッシュボード登録（§5-3）→ WP close 発火**
- [ ] instance destroy
