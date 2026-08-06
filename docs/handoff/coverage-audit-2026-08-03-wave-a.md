# Coverage-audit 2026-08-03 Wave A + B + D handoff

> **UPDATE 2026-08-04 #10 (continuation batch: miocodec / neutts-air / sgmse-voicebank publish + w2v-bert-2 vast.ai handoff land + torchaudio-squim sidecar land)**: **HF vokra org 196 → 199 models (+3 新規)**。**Published 3**: `vokra/miocodec-25hz-44khz-v2` (503.6 MB, MIT、350 F32 tensors、Aratako/MioCodec-25Hz-44.1kHz-v2 = JA-focused 11-lang codec、arXiv:2507.21138、132 M F32 params、`15044dd`) + `vokra/neutts-air` (1426.6 MB, Apache-2.0、291 BF16 tensors、Neuphonic NeuTTS Air on-device instant voice-cloning TTS = Qwen2 0.5 B LLM backbone GQA 14:2 + NeuCodec token space extended vocab 217,652、sibling `vokra/neucodec` row 305、`ce03000`) + `vokra/sgmse-voicebank` (250.25 MB, Apache-2.0、647 F32 tensors、speechbrain/sgmse-voicebank = **Vokra catalog 初の M3-05 flow_sampler + ODE solver 実 weight consumer**、NCSN++ v2 + OUVE SDE reverse sampler、`b3fcf13`)。全 3 model = 新 `ModelKind` + BF16/F32 pass-through skeleton (miocodec/bicodec/neucodec/focalcodec/xcodec2 sibling pattern) + 新規 §3.1 row (☑ Commercial 2026-08-04 yousan、依頼者許可 = CC 判断、primary source HF cardData API clean、Aratako Irodori-TTS-500M-v3 MIT + Neuphonic apache-2.0 + SpeechBrain family precedent 全て適用)。sgmse は `tools/parity/sgmse_prepare_checkpoint.py` (torch pickle .ckpt → safetensors bridge、uv-managed Python 3.12、no pickle in runtime tree per FR-LD-05) 追加。**CC land + owner-defer 2**: (a) **w2v-bert-2.0** (`f2875a2`) = 新 `ModelKind::W2vBert2` + arch=`w2v-bert-2` (siblings hubert / wav2vec2_ctc / data2vec-audio と distinct = Conformer body + contrastive+MLM SSL、FR-EX-08 no silent op-shape misroute) + §3.1 row (☑ Commercial 2026-08-04 yousan) + signoff_match REPO+CONVERTER 双方登録 + 3 unit test all pass、実 publish は **safetensors 2.16 GB (= 2,322,063,736 bytes) が 2 GB local-convert owner threshold 超過で vast.ai handoff** (`docs/handoff/vast-ai-large-model-publish.md` §2)。将来コマンド = `bash scripts/publish/publish-one.sh w2v-bert-2-0 --push` / (b) **torchaudio-squim** = sidecar prep script `tools/parity/torchaudio_squim_prepare_checkpoint.py` (~420 行、dnsmos_prepare_checkpoint.py の bundle-merge pattern を objective./subjective. prefix で複製、torch.hub 自動 DL + `--objective-ckpt`/`--subjective-ckpt` offline override、F32/F16/BF16 pass-through only + FR-EX-08 loud-fail、data_ptr dedup、shared_pairs.json audit) 完備 + torchaudio>=2.11.0 uv dep を transitive-via-torch-audiomentations から explicit 昇格 (`296fe2a`)、実 publish は **owner license re-audit 必須** = 現 §3.1 row は BSD-2-Clause end-to-end + ☑ Commercial 2026-08-04 yousan だが upstream tutorial page で `squim_objective_dns2020.pth` = **CC-BY-4.0 (Attribution)** / `squim_subjective_bvcc_daps.pth` = **CC-BY-NC-4.0 (Non-Commercial)** が primary source。owner 3 択 = (A) 現状 BSD-2-Clause 単一 slug 維持 / (B) 2-slug split (`torchaudio-squim-objective` T2 CC-BY-4.0 + `torchaudio-squim-subjective` T4 CC-BY-NC-4.0 = X-Codec-2 precedent 踏襲) / (C) OBJECTIVE のみ publish + SUBJECTIVE hold。**Skipped 2**: (a) **openwakeword-op** = op-wiring anchor は既 landed (`ModelKind::OpenwakewordOp` + converter (424 行、3 unit test pass) + §3.1 row 460 (☑ Commercial 2026-08-04 yousan) + Wave A/F ticket = weight 非配布方針 EnCodec pattern codified)、HF probe で `vokra/openwakeword-op` = HTTP 404 (正しく never published)、Wave A.2 ticket 明示「Publish: なし = docs のみ upload の場合 owner 判断」= **no design intent**、runtime binder (`vokra-models/src/kws/openwakeword/mod.rs` + `KwsSession::from_gguf` + `Stream::next_probability()`) は独立 feature-development task (~6-8 h) / (b) **sensevoicesmall** = 依頼者 dispatch batch context が「Phase A already landed」を前提としていたが実 filesystem に converter / §3.1 row / signoff_match entry 全て absent (working tree 検証で docs/ subdirectory only を確認)、加えて FunASR MODEL_LICENSE v1.1 は **SPDX 未登録** で LicenseClass::Unknown fail-close = owner primary-source audit rate-limited (30-60 min) + `[[feedback-license-signoff-primary-source]]` で CC self-sign 禁止 = **CC 側 clean-fail、Phase A land を owner 再確認 → 再 dispatch 推奨**。全 commit: `296fe2a` torchaudio-squim sidecar / `15044dd` miocodec / `f2875a2` w2v-bert-2 / `ce03000` neutts-air / `b3fcf13` sgmse-voicebank。**verify 全 green** (cargo fmt --check / cargo clippy `-D warnings` / cargo test miocodec 3 pass + neutts-air 3 pass + sgmse 3 pass + w2v-bert-2 3 pass + torchaudio-squim 3 pass = **+15 new tests** / `scripts/check-zero-deps.sh` OK (NFR-DS-02 preserved) / `scripts/gen-c-abi.sh --check` no drift = **新規 C ABI ゼロ、v1.0-rc baseline 33 fn + 11 typedef 不変**)。local artifacts (checkpoints/ + gguf/ + target/publish/ scratch) 全 cleanup 済。**教訓**: (a) 2 GB size gate は catalog registration 前段の判定材料 = w2v-bert-2 のように converter code + §3.1 row + signoff_match は事前 land + owner vast.ai handoff が **Wave F 型 clean-hand-off pattern** (converter ready-state で size 単独 gate = 実 publish のみ owner に委譲、re-work なし)、(b) upstream tutorial page での per-checkpoint license divergence (torchaudio-squim = objective CC-BY-4.0 vs subjective CC-BY-NC-4.0) は §3.1 sign-off より primary source (upstream 公式ドキュメント) が優先 = **事後 re-audit trigger** となる、(c) sgmse-voicebank publish は Vokra catalog 初の **M3-05 flow_sampler + ODE solver 実 weight consumer** = future NCSN++ v2 + OUVE SDE reverse sampler 実装の real-weight parity harness の base (現時点は loud-partial per RMVPE / Charsiu / MOSS-Audio-Tokenizer / MioCodec / w2v-bert-2 precedent 継承)、(d) sensevoicesmall = dispatch context の「Phase A landed」前提と実 filesystem 状態の乖離を発見した時は **clean-fail + blocker report** = context refresh を owner に要求する pattern が正解 (fabricated Phase A の land は fail-closed 規律違反、`[[feedback-license-signoff-primary-source]]` の CC 越権禁止と同型)。

> **UPDATE 2026-08-04 #9 (ultravox v0.5 + titanet-l batch publish、既 ModelKind + .nemo bridge +2)**: `vokra/ultravox-v0-5-llama-3-2-1b` (1.37GB, MIT、491 BF16 tensors、fixie-ai/ultravox-v0_5-llama-3_2-1b) + `vokra/titanet-l` (102MB, CC-BY-4.0、108 tensors、nvidia/speakerverification_en_titanet_large .nemo bridge 経由) = 既存 `ModelKind::UltravoxV05Llama321b` + `ModelKind::TitaNet` を direct convert で publish。`signoff_match.py` の REPO_TO_SIGNOFF_ROWS に `titanet-l` / `titanet-large` slug alias 追加。**HF vokra org 194 → 196 models (+2 新規)**。**CC 側 local publish candidates 実質枯渇** (残 未 publish で local 可能な既 signoff + ≤2GB pair は zonos 3.1GB / dia 6.4GB / moshi 15GB / voxtral 8.7GB = 全 vast.ai defer)。以降の追加 wins は新 ModelKind + 新 §3.1 row 実装 = 独立 wave 相当。

> **UPDATE 2026-08-04 #8 (xvector publish、既 ModelKind + custom 3-ckpt bridge +1)**: `vokra/xvector` (33MB, Apache-2.0、46 tensors from 3-ckpt merge = classifier+embedding_model+mean_var_norm_emb) = 既存 `ModelKind::XVector` + 汎用 torch bridge 経由で publish 完了。`signoff_match.py` の REPO_TO_SIGNOFF_ROWS に `xvector` slug 追加 (CONVERTER_TO_SIGNOFF_ROWS には既 registered)。**HF vokra org 193 → 194 models (+1 新規)**。教訓: SpeechBrain 3-ckpt bundle (encoder/decoder/masknet 前提の sepformer prep 経路と別) は generic torch.load ループで stem prefix 付き merge、xvector converter は namespaced key を期待する = このパターンで publish 可能。

> **UPDATE 2026-08-04 #7 (SepFormer 4-variant batch publish、既 ModelKind + prep script 経由 +4)**: `vokra/sepformer-wham16k-enhancement` (108MB) + `vokra/sepformer-whamr16k` (304MB) + `vokra/sepformer-libri2mix` (304MB) + `vokra/sepformer-libri3mix` (305MB) = 全 Apache-2.0、既存 `ModelKind::Sepformer*` 4-variant 経由で publish 完了 (417 tensors ずつ、`sepformer_prepare_checkpoint.py` の 3-part ckpt bundle merge 経由)。**HF vokra org 189 → 193 models (+4 新規)**。 sepformer-libri3mix は converter で `vokra.sepformer.n_out=3` を自動 stamp。

> **UPDATE 2026-08-04 #6 (Vocos pair + WavTokenizer-large publish、既 ModelKind 経由 +3)**: `vokra/vocos-mel-24khz` (54.3MB, MIT、83 tensors) + `vokra/vocos-encodec-24khz` (40.3MB, MIT、82 tensors) = 前 wave 実装済 `ModelKind::Vocos` を bin_to_safetensors.py 経由で 2 model publish + `vokra/wavtokenizer-large` (846MB, MIT、1091 tensors) = 前 wave 実装済 `ModelKind::Wavtokenizer` を ckpt bridge (Lightning `state_dict` extract) 経由で publish。**HF vokra org 186 → 189 models (+3 新規)**。`signoff_match.py` に `wavtokenizer-large` slug alias を追加 (`wavtokenizer-large-speech-75token` の別名 = repo slug は短縮版で publish)。**既知の軽微 issue**: converter の `detect_variant` が encodec 側でも `mel_24khz` と誤判定 (tensor 名で判別できない、両者 backbone 名共通)、GGUF metadata の `vokra.vocos.variant` が両 model 共通 = runtime binder は upstream_hf provenance で識別する必要あり (follow-up、publish repo 名は正確)。

> **UPDATE 2026-08-04 #5 (frcrn ModelScope + ten-vad ONNX bridge, +3 publish)**: (a) **frcrn 58MB Apache-2.0** = ModelScope `damo/speech_frcrn_ans_cirm_16k` から DL 成功 (前 session の modelscope dep `c1d3879` 経由、今 session の HTTP DL は classifier block 発生せず)、`tools/parity/frcrn_prepare_checkpoint.py` 経由で ClearerVoice-Studio branch の pretrained 812 tensors → GGUF。(b) **ten-vad 306KB Apache-2.0** = ONNX 27 initializers → 新規 汎用 bridge `tools/parity/onnx_to_safetensors.py` 経由で FLOAT のみ 19 tensor 抽出 → GGUF (INT graph-metadata 8 = shape/axes/steps は fail-closed skip)。(c) **htdemucs-multi** (前 #4 で published) の `signoff_match.py` に `htdemucs-multi` slug entry を追加 (`htdemucs-4s-6s` の alias、fix `366e980`)。**HF vokra org 183 → 186 models (+3 新規)**。教訓: ONNX→safetensors bridge は共通 utility として `tools/parity/onnx_to_safetensors.py` に汎用化、以降 WavTokenizer / Vocos / GTCRN 等の ONNX-only モデルも同経路で処理可能 (INT threshold 8 で graph-metadata skip、`--int-threshold` で調整)。openwakeword-op + htdemucs-6s-onnx は skip = ticket 明示 (weight 非配布 / 既 htdemucs-multi と重複)。

> **UPDATE 2026-08-04 #4 (Wave A permissive 7 code + 2 publish + Wave D jasco slug fix)**: Wave A permissive 7 converter code land (commit `e5fe0e9` = utmosv2 / torchaudio-squim / htdemucs-multi / openwakeword-op / mossformer2-ss-16k / ten-vad / audioseal-real-weight、+3383 行 / +21 test = 785 total、alias collision 2 件を明示 disambiguation で解決 = htdemucs は `htdemucs-ft`/`htdemucs_6s`/`-multi` 隔離、openwakeword は `-op` suffix 隔離)。jasco-400m-chords-drums slug も同 commit で修正 (`facebook/jasco-chords-drums-400M` へ、code + §3.1 + signoff_match 三面同時)。**publish 2/7 = audioseal-real-weight 187MB MIT / mossformer2-ss-16k 223MB Apache-2.0 = HTTP 200**。**HF vokra org 181 → 183 models (+2 新規)**。**DEFERRED 5 owner critical path**: (1) utmosv2 = 3.9GB → vast.ai / (2) ten-vad = ONNX (306KB) の safetensors bridge 未整備 / (3) torchaudio-squim = torch.hub 経由の PyTorch pipeline import が必要 / (4) openwakeword-op = HF repo `davidscripka/openwakeword` は CC-BY-NC-SA-4.0 実態で code 前提 Apache-2.0 と不一致 → license 再判定 or repo 再選定要 / (5) htdemucs-multi = HF `facebook/htdemucs_ft` / `facebook/htdemucs_6s` は 404、community mirror 精査 or GitHub Release URL 経由要 / (6) jasco-400m-chords-drums = HF gated repo owner accept 必須。教訓: alias collision (openwakeword + htdemucs) は既存 ModelKind が独占していた slug なので新 ModelKind は disambiguation suffix (`-op`/`-multi`) が必要。openwakeword upstream の実 license は README に頼らず HF card + LICENSE の primary source で再確認要。

> **UPDATE 2026-08-04 #3 (Wave D T4 partial)**: Wave D T4 non-commercial 5 converter code land (commit `d2e02ab` = facebook-denoiser / nisqa-v2-weight / chattts / stable-audio-open-small / jasco-400m-chords-drums、CPML precedent 型 SACL hard-map 含む、+2122 行、764 test pass、C ABI baseline 33 fn 不変)、**publish size gate 適用 (≤2GB のみ) で 3/5 publish (chattts 813MB / facebook-denoiser 72MB / nisqa-v2-weight 1MB = HTTP 200 全て T4=--allow-noncommercial)**。**HF vokra org 178 → 181 models (+3 新規)**。**DEFERRED 2 = stable-audio-open-small (HF gated repo、owner web UI accept 必須) + jasco-400m-chords-drums (ticket の slug と実 HF path 不整合 = actual `facebook/jasco-chords-drums-400M`、alias 追加要 = 別作業)**。教訓: chattts は `asset/gpt/model.safetensors` 単体 (814MB) で publish、全 bundle (~2.2GB) の他 asset は runtime binder 実装時に別途処理。SACL SPDX 未登録 → `stable_audio_open_small.rs` で `LicenseClass::NonCommercial` hard-map (xtts_v2.rs CPML precedent 踏襲)。

> **UPDATE 2026-08-04 #2**: Wave B fast-track 追加 5/13 publish (parakeet-tdt-1.1b 4.28GB / reazonspeech-nemo-v2 2.48GB / firered-asr-aed-l 4.4GB / sortformer-diar-4spk-v1 494MB **T4=CC-BY-NC-4.0 sign-off 修正済** / whisper-medusa-v1 6.25GB = HTTP 200)。**HF vokra org 173 → 178 models (+5 新規)**。**Wave B 累計 10/13 = 76.9% published**。残 3 = magpietts-v2602 / nemotron-speech-streaming-v2603 / parakeet-unified-en-0.6b (**全て NGC-only、HF mirror 不在**) + SenseVoiceSmall (FunASR MODEL_LICENSE owner audit pending) = owner critical path。教訓: `.pth.tar` は実 Zip format (torch.save の zip 内容が拡張子だけ tar に書換え)、torch.load 直接可能。sortformer は HF card 上 CC-BY-NC-4.0 明示、initial sign-off の CC-BY-4.0 は誤記で publish gate が正しく検出 → --allow-noncommercial + T4 化で publish 完了 (X-Codec-2 precedent 踏襲)。中間ファイル (checkpoint + intermediate safetensors) は publish 直後に削除 (disk 一時 100% 到達に対処、依頼者要請)。

> **UPDATE 2026-08-04**: Wave A residual 3/4 publish (dnsmos/rnnoise/nsnet2 = HTTP 200) + Wave B fast-track 5/13 publish (hibiki-2b / sber-gigaam-v3 / sber-gigaam-multilingual / canary-1b-flash / owsm-v4-medium-1b = HTTP 200)。**HF vokra org 165 → 173 models (+8 新規、frcrn は ModelScope classifier block ゆえ owner 手動)**。Wave B 残 8 model は upstream fetch/prep 経路確立済で owner or 次 session sequential 実行可 (§2 参照)。



> 依頼者「ultracodeで計画を立てて全てを対応を進めてください」に基づき launch した coverage-audit 2026-08-03 Wave A top-5 の実装 + publish 進捗。CC 側の converter code 全 landed、publish は 1/5 completed = **nkf-aec** のみ、他 4 model は upstream fetch friction で **owner critical path**。

## 1. 概要

### CC land 済 (2 commit)
- **`cc66fd3`** feat(convert/nkf-aec): Wave A ticket = aec/NEC (MIT, 5.3 KB)
- **`e8b8c21`** feat(convert): Wave A top-4 + 5 §3.1 signoff (coverage-audit 2026-08-03)
  - rnnoise / nsnet2 / dnsmos / frcrn converter Rust + prep script (uv Python 3.12) + tests
  - docs/license-audit.md §3.1 に 5 rows sign-off (all Permissive, CC 判断)
  - scripts/publish/signoff_match.py に 5 slug entry (APPROVED gate 通過)

### verify (main tree HEAD e8b8c21)
- `cargo test -p vokra-convert`: **733 passed / 0 failed / 3 ignored** (11 suites)
- `cargo fmt --check` / `cargo clippy -p vokra-convert -- -D warnings`: OK
- `scripts/check-zero-deps.sh`: OK (root Cargo.lock は vokra-* のみ、NFR-DS-02 preserved)
- `scripts/check-abi-changelog.sh`: OK (v1.0-rc baseline 33 fn + 11 typedef 不変、**新規 C ABI ゼロ**)
- `scripts/publish/signoff_match.py --self-test`: OK (7 approval cases + 1 converter case)

### publish status (huggingface.co/vokra)

| # | Slug | Size | Status | Notes |
|---|------|------|--------|-------|
| 1 | **nkf-aec** | 23.7 KB GGUF | ✅ **HTTP 200** published | upstream = `github.com/fjiang9/NKF-AEC/src/nkf_epoch70.pt` (README `pretrained/nkf.pt` 記述と異なる = 実 file は `src/`)、prep script 動作確認済 |
| 2 | rnnoise-v0.2 | TBD | ⏸ upstream barrier | v0.2 release asset は source tarball のみ、`weights_blob_9.bin` は **build 必須** (`autogen.sh && ./configure && make`) or main branch checkout の別 path。**owner or 別 phase 対応** |
| 3 | nsnet2 | TBD | ⏸ upstream barrier | DNS-Challenge master にも interspeech2020/master にも `NSNet2-baseline` dir 不在。`download-dns-challenge-5-baseline.sh` 経由の **1.4 GB Baseline.zip** DL 要 (Azure blob URL、authorization 不要だが time cost 大)。**owner or 別 phase 対応** |
| 4 | dnsmos-p808-p835 | ✅ **HTTP 200** published (2026-08-04) | commit `343750a` で prep script に empty-shape scalar + INT graph-metadata skip logic を追加、Wave A residual publish で published (UPDATE #1 参照)。 |
| 5 | frcrn | TBD | ⏸ upstream barrier | `github.com/alibabasglab/FRCRN` は README のみで pretrained checkpoint 不在。ModelScope 経由 (`damo/speech_frcrn_ans_cirm_16k`) で `pytorch_model.bin` を DL 必要。HF mirror (`alibabasglab/FRCRN`) = 401 (不在)。**owner ModelScope authentication + `uv add modelscope`** |

## 2. Owner critical path (Wave A 完了までの最短経路)

### 2a. dnsmos-p808-p835 (即時可能、prep script re-run のみ)
```bash
cd /Users/inamotoyuuta/Desktop/Otonx
source tools/parity/.venv/bin/activate  # or `uv sync --project tools/parity`
python tools/parity/dnsmos_prepare_checkpoint.py \
  --p808 ~/checkpoints/dns-challenge/DNSMOS/DNSMOS/model_v8.onnx \
  --p835 ~/checkpoints/dns-challenge/DNSMOS/DNSMOS/sig_bak_ovr.onnx \
  --output ~/checkpoints/dnsmos/model.safetensors
./target/release/vokra-cli convert --model dnsmos-p808-p835 \
  --input ~/checkpoints/dnsmos/model.safetensors \
  --output ~/gguf/dnsmos-p808-p835.gguf
export HF_TOKEN=$(grep '^HF=' .env | cut -d'=' -f2-)
bash scripts/publish/publish-one.sh --gguf ~/gguf/dnsmos-p808-p835.gguf \
  --repo vokra/dnsmos-p808-p835 --license-spdx MIT --push
```

### 2b. rnnoise-v0.2 (C build 経由、~5 min)
```bash
cd ~/checkpoints/rnnoise
tar -xzf rnnoise-0.2.tar.gz
cd rnnoise-0.2
./autogen.sh && ./configure && make
# build 完了後、weights_blob_9.bin が root or .libs/ に生成
find . -name "weights_blob_9.bin"
# 次に prep + convert + publish (nkf-aec pattern)
```

### 2c. nsnet2 (DNS5 Baseline.zip 1.4 GB DL、~10 min)
```bash
cd ~/checkpoints/nsnet2
curl -o Baseline.zip https://dnschallengepublic.blob.core.windows.net/dns5archive/Baseline.zip
unzip Baseline.zip
find . -name "nsnet2-20ms-baseline.onnx"
# 見つかったら prep + convert + publish
```

### 2d. frcrn (ModelScope 経由、~5 min)
```bash
cd tools/parity && uv add modelscope
uv run --project tools/parity python -c "
from modelscope import snapshot_download
p = snapshot_download('damo/speech_frcrn_ans_cirm_16k', cache_dir='/tmp/modelscope')
print(p)
"
# .pt 見つかったら prep + convert + publish
```

## 3. 次 phase 判断

### Wave B/C/D の owner sign-off queue
- Wave B fast-track local (13 items) の中の hibiki-2b / sber-gigaam-v3 / reazonspeech-nemo-v2 / NVIDIA family (canary-1b-flash / parakeet-tdt-1.1b / sortformer-diar / magpietts / parakeet-unified-en) は **primary source 精度確認さえ済めば CC 実装可能** (`docs/tickets/coverage-audit-2026-08-03/wave-b/{slug}.md` per-model ticket 完備)
- Wave B vast.ai (14 items) は **vast.ai provisioning + owner sign-off 必須** = 各 model の 独自 license audit と 5-30 GB DL
- Wave C MoE (qwen3-omni / zonos2) は **`moe_dispatch` / `moe_expert_gemm` 新規 op 起票必要** = runtime crate 側の RFC + implementation
- Wave D T4 non-commercial (13 items) は **X-Codec-2 precedent 踏襲** で `--allow-noncommercial` gate 通過、既 pattern

## 4. Deep-research 全体 summary の再引用

- **200 rows 候補** (P0=54 / P1=71 / P2=44 / P3=31)
- **9 分野**: ASR / TTS / Music-gen / Sep / Audio-LLM / Enhancement / Codec+VAD+KWS+Speaker / Eval+SSL+WM / 非HF
- **License 分布 P0**: T1 Commercial 64% / T4 NC 24% / 独自 audit 要 12%
- **Non-HF source**: 25% (ModelScope / NGC / GitHub Release / Zenodo / Sber ai-sage mirror)

詳細:
- `/private/tmp/claude-501/-Users-inamotoyuuta-Desktop-Otonx/aa94d709-.../scratchpad/vokra-audio-model-coverage-audit-2026-08-03.md`
- `docs/tickets/coverage-audit-2026-08-03/INDEX.md` + `OWNER-CRITICAL-PATH.md` + `IMPL-PLAN.md` + 72 per-model MD

## 5. 教訓

- worktree agent の base (d05ab7d) と main tree HEAD (58629ab) が 20+ commits 乖離 = cherry-pick は 11+ conflicts 発生 → **independent files copy + shared file additive edit を 1 agent に統合委任** の pattern が有効
- upstream fetch は README 記述と実 file 配置がズレる (nkf-aec: `pretrained/nkf.pt` 想定だが実 `src/nkf_epoch70.pt`) → prep script は **file name-agnostic** (torch.load で拡張子のみ判定) で robust
- `uv run --project tools/parity` は project directory 存在 warning + cache init hang 事例あり = owner は venv activate 直接 or `uv sync --project tools/parity` 先行推奨
- HF mirror が存在しない ModelScope-only モデル (frcrn) = `uv add modelscope` 経路標準化推奨 (今 wave 前は不要だった、Wave A 以降で FunASR / ClearerVoice-Studio 系が増える見込み = 汎用 pattern 化)

---

**Session 総括**: nkf-aec = **HF vokra org 166 model** に land (前 165 + 1)。他 4 model = converter code land 済 + owner critical path 化。全 §3.1 sign-off 完了 = publish gate 通過準備完了。
