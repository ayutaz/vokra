# Vokra 実 weight 評価キャンペーン 統合レポート（vs onnxruntime、Apple M1 iMac）

**実施**: 2026-07-16 時点 / **評価 agent**: 11 本（ASR 5・TTS 3・codec 2・VAD+speaker 1）/ **対象 HEAD**: `0b7e7caa`（branch `feat/m4-plan-and-wave1`、rustc 1.95.0 release build）/ **repo**: 全 agent で未改変（git status clean。実測はすべて out-of-repo scratch harness・公開 API・vokra-cli 経由）

---

## 1. エグゼクティブサマリー

- **精度 parity は Whisper 全 4 サイズで完全達成**: base/small/medium/turbo とも実 HF checkpoint → GGUF 変換 → e2e 転写が成功し、**vokra と onnxruntime の転写が全ファイル一致**（正規化 word diff 0、WER/CER 両エンジン同値、JFK 4/4 一致）。piper-plus native も near-bit-exact（mel-L1 0.0033–0.0035、サンプル差 ≤3 int16 LSB、長さ完全一致）。
- **codec 系実 weight parity は全 PASS**: Mimi/DAC は 106/106 テンソル（bit-exact〜1.526e-5、atol 0.01）、WavTokenizer は実 codebook + jfk 由来実 codes 440 個で **max Δ = 0.0 の bit-identical**。CAM++ speaker は独立 frontend 同士で cosine ≈ 1.0・話者 ranking 完全一致。
- **CPU 速度は base を除き ORT 劣後**: base はほぼ互角（jfk で vokra +2.5% 速、corpus では ORT 1.44x 速）だが、small 2.90–3.11x / turbo 1.81x / medium 7.4–7.8x / Kokoro 4.27x / piper-plus 5.17–8.28x / Silero VAD ~5x で vokra が遅い（短い発話ほど gap 拡大、P2 perf issue）。
- **P1 級の honest negative を 4 系統検出**: (i) Kokoro は自前 fixture と 9/9 一致するも **upstream 実体と乖離**（bert Δ5.84、duration 902 vs 57）し実文が unintelligible（round-trip WER 1.0、root cause 5 点をソース行で特定）、(ii) **Silero VAD は公式 64-sample rolling context の欠落で実音声の speech 検出 0**（数値 parity は 1.13e-6 = semantics gap）、(iii) CosyVoice2 は実 checkpoint の q/k/v attention bias を表現できず true model と argmax 0/10 FAIL、(iv) Voxtral は TextDecoder の GQA shape 前提で実 weight ロード不可 + **bf16 入力で weightless GGUF を exit 0 で出す converter バグ**。
- **owner checklist 消化**: M4-14 Whisper family 実 weight 初回（turbo atol calibrate 不要の実測材料込み）、M4-04 Mimi/DAC real parity のローカル first-fire、M4-16 WavTokenizer 実 parity 前倒し、M3-10 Voxtral adapter.json 納品、M3-09 CosyVoice2 flip-the-switch（honest FAIL）を今回ローカルで討ち取り（詳細 §7）。

---

## 2. 共通環境・方法論（脚注 — 全表に適用）

| 項目 | 内容 |
|---|---|
| 機材 | Apple M1 iMac、8 CPU、16 GB RAM、macOS arm64。GPU leg は Metal bonus のみ（CUDA なし） |
| vokra | release build @ git `0b7e7caa`（metal feature 込みビルドあり）、rustc 1.95.0。スレッド = 8（`VOKRA_CPU_THREADS` 未設定 → `available_parallelism`） |
| ORT | onnxruntime **1.19.2** CPUExecutionProvider、optimum 2.1.0（+ optimum-onnx 0.1.0）、transformers 4.57.6、torch 2.8.0、numpy 2.0.2、python 3.9.6 venv。intra_op = 0（auto）が既定、明記箇所のみ intra_op=1 |
| 計測窓 | 両エンジンとも **inference-only（モデル preload・WAV 読込は窓外）、1 warmup + 3 計測の median（p50）**。RTF = wall 秒 / audio 秒。vokra bench の `rtf` フィールドは mean ベースのため median を再導出（HARD RULE 準拠、raw JSON 保存済） |
| ASR コーパス | jfk-30s.wav（11.00 s、canonical ref）+ LibriSpeech dev-clean 1272/128104。**base/small/turbo = 9 files / 97.995 s、medium のみ 5 files / 48.25 s（assignment 指定の truncation）** → medium の WER は他サイズと横比較不可 |
| スコアリング | jiwer 4.0.0 corpus-level + whisper `EnglishTextNormalizer`（openai-whisper 20250625、hyp/ref 両方に適用） |
| Whisper 特性 | 全 clip を 30 s window に pad するため短 clip の RTF は両エンジンとも悲観的。corpus RTF は長さ加重の honest 集計 |
| decode 条件 | greedy（num_beams=1）、prefix [50258,50259,50359,50363]（turbo は [50258,50259,50360,50364]）で両エンジン一致。**差異（文書化済・バグではない）**: vokra greedy は logit suppression なし、ORT は checkpoint の suppress lists 有効 — 本コーパスでは word-level 影響 0（small の 1 file で句読点のみ差） |
| 追加 venv | Kokoro oracle = py3.11（kokoro 0.9.4 / misaki 0.9.4 / torch 2.13.0）。codec RVQ = py3.11.15（torch 2.9.1 / moshi 0.2.13 / descript-audio-codec 1.0.0 / numpy 2.2.6、workflow pin 一致）。FSQ = vector-quantize-pytorch 1.17.8（fixtures 元 pin torch 2.5.0 → 2.8.0 で 8/8 sha256 一致 = immaterial） |
| ノイズ注意 | 複数 sibling agent が同一 M1 上で並走 → 一部 leg に timing noise（medium bench jitter 最大 11.6 s、Kokoro ORT RTF 0.51–0.88 等）。median 採用、raw 保存 |
| その他 | HF token は stale（public repo のみ使用のため影響なし）。vokra のモデル load 時間は CLI 非露出 → 「derived」（プロセス全体 wall − 推論 p50）として明記 |

**like-for-like でない比較（明示）**: turbo は vokra が fp16 貯蔵（fp32 計算）vs ORT fp32 weight（同一 checkpoint）。piper-plus の vokra 側 synthesis-only は減算推定（WAV write 含む）vs ORT は pure `session.run()`。Kokoro の RTF 分母は各エンジン自身の合成長（2.4875 s vs 3.25 s、duration が乖離するため）。CAM++ は thread model 不一致（vokra pool ≤8 vs ORT default/1-thread 両掲載）。codec の Vokra 数値は quantizer stage のみで full-codec RTF と非比較。Mimi/DAC/FSQ/CosyVoice2 LLM/Voxtral には公式 ONNX が存在せず ORT baseline なし → PyTorch reference parity が精度指標（repo CI 設計と同一）。

---

## 3. ASR 結果

| モデル | 変換 | vokra WER/CER | ORT WER/CER | vokra RTF (jfk / corpus) | ORT RTF (jfk / corpus) | 速度比 | JFK 転写一致 |
|---|---|---|---|---|---|---|---|
| whisper-base | ✅ 290,859,008 B・245 tensors・fp32 | 3.95% / 1.68% | 3.95% / 1.68% | 0.1766 / 0.2166 | 0.1809 / 0.1501 | corpus: ORT 1.44x 速（ratio 0.693）、jfk: vokra +2.5% 速（ratio 1.025） | ✅ raw byte 一致 9/9（JFK は canonical 文とも一致） |
| whisper-small | ✅ 967,441,632 B・479 tensors | 3.07% / 1.28% | 3.07% / 1.28% | 0.665 / 0.749 | 0.229 / 0.241 | vokra 2.90x（jfk）/ 3.11x（corpus）遅 | ✅ JFK 一致（正規化 9/9、raw 8/9 — 0011 のみ句読点差 = suppress-list 由来） |
| whisper-medium ※corpus 5 files/48.25 s | ✅ 3,055,971,200 B・947 tensors | 1.75% / 0.82% | 1.75% / 0.82% | 4.262 / 6.025 | 0.576 / 0.768 | vokra ~7.4x（jfk）/ ~7.8x（corpus）遅 | ✅ byte 一致（両エンジン正解） |
| whisper-large-v3-turbo | ✅ 1,618,266,208 B・587 tensors（fp16 貯蔵※） | 2.63% / 0.88% | 2.63% / 0.88% | 4.467 / 4.721 | 2.472 / 2.601 | vokra 1.81x 遅（ratio 0.551） | ✅ 9/9 character 一致（canonical とも一致） |
| Voxtral-Mini-3B-2507 | ✅※harness 経由 762/762 tensors・9.354 GB（stock CLI は sentinel hparams + tokenizer なし） | —（推論不可） | —（ORT baseline なし = assignment） | — | — | — | —（ran=false、捏造なし） |

※turbo: vokra は M2-06 F16 passthrough で fp16 貯蔵/fp32 計算、ORT export は fp32 weight — 同一 checkpoint・同一 decode 条件で weight 精度のみ非対称（明示）。
※Voxtral: `TextDecoder::load` が実 GQA 形状を拒否（q_proj [4096,3072] vs 期待 [3072,3072]、32 heads × head_dim 128 = 4096 ≠ hidden 3072）→ e2e 不成立。§6/§8 参照。

**tensor parity（committed synthetic-PCM fixture 対比、atol 0.01、8/8 PASS）**: base = log-mel 3.338e-6 / encoder 3.624e-5 / decoder logits 4.196e-5。small = 3.338e-6 / 8.392e-5 / 3.004e-5。turbo = 3.338e-6 / 4.357e-5 / 3.910e-5（**M4-14 の turbo 懸念は非発現、atol 余裕 ~3 桁 = per-size calibrate 不要の実測材料**）。medium は assignment 指定で parity leg skip。注: fixture は pre-4665d74 の合成 1 s PCM snapshot であり real-audio fixture ではない（regen/commit は owner T09）。

**Metal bonus（7b）**: base jfk RTF 0.2434（CPU 0.1766 より遅、memory-bound の既知傾向と整合）だが **転写は CPU と token 単位一致（26 tokens、scratch harness + repo metal e2e test green）**。small は Metal 0.727 vs CPU 0.665 でやはり遅く、転写一致は CLI で検証不能（`run` に `--backend` なし + metal e2e test が base-pinned）。

**load / one-time コスト**: ORT export（一回限り）= base 10.95 s / small 14.19 s / medium 50.26 s（+save 10.14 s）/ turbo 45.14 s。ORT session load = 0.48 / 1.31–2.42 / 6.33 / 6.81 s。vokra load は derived = base ~0.248 s / small ~0.678 s / medium 18.03 s（host 負荷で confounded、軽負荷時 ~4.5 s、両 raw 保存）/ turbo ~4.57 s。vokra convert: medium 7.94 s、turbo 3.35 s。

---

## 4. TTS 結果

| モデル | e2e レベル | parity 指標 | vokra RTF | ORT RTF | round-trip WER |
|---|---|---|---|---|---|
| piper-plus（css10-ja-6lang voice + 数学的中立 spk_proj） | text→PCM（実 G2P = vokra-piper-g2p 経由。CLI `--text` は placeholder tokenizer のため比較から除外） | mel-L1 EN 0.003455 / JA 0.003297、max sample Δ EN 6.104e-5（2 LSB）/ JA 9.155e-5（3 LSB）、サンプル数差 0 — near-bit-exact | EN 0.192 / JA 0.167（synthesis-only 推定。full-process 0.217 / 0.185） | EN 0.0231 / JA 0.0324（inference-only、default threads）→ vokra 8.28x（EN）/ 5.17x（JA）遅※窓非同一 | JA CER 0.278（**両エンジン転写が文字単位一致**）/ EN WER 1.0 **両エンジン**（deterministic scales でこの JA voice が EN を ~1.27 s/10 語で発話 = voice 特性、エンジン差ではない） |
| Kokoro-82M（af_heart） | phoneme-id→PCM（scratch harness で公開 `synthesize_phonemes` を駆動。`TtsEngine::synthesize` = NotImplemented、CLI 経路なし） | 自前 fixture parity **9/9 PASS**（f0 2.619e-2 ≤ 0.05 bound、他 ≤0.01、durations bit-exact）。**ただし upstream oracle 対比で bert Δ5.84 / text_encoder Δ1.62 / pred_dur 902 vs 57** = self-consistency のみ。mel-L1 vs ORT（同一入力）raw 0.530 / peak-norm 0.404（同 pipeline noise floor 0.021–0.107） | 3.368（sentence）/ 3.057（fixture） | 0.788（同一入力・default threads）/ 1.705（intra_op=1）/ 1.237（fixture）→ vokra 4.27x（RTF）/ 3.27x（同一 token wall）遅 | vokra **1.0（whisper-base 転写が空 = unintelligible）** / ORT 同一入力 dup-style 0.333 / ORT natural 256-dim style 0.0 / upstream oracle 0.0 |
| CosyVoice2-0.5B | **音声出力なし**（`synthesize()` = NotImplemented = T06/T07/T08/T10/T11/T13 残。LLM backbone forward のみ実 weight で out-of-tree 検証、GGUF 2.57 GB は verify-load 成功） | vokra vs bias-less PyTorch mirror（vokra がモデル化した演算そのもの）: L24 max Δ **1.45e-4・argmax 10/10 PASS**。vokra vs **true Qwen2（q/k/v bias あり）: max Δ 12.92・mean 0.856・argmax 0/10 FAIL**（構造的 — `LlmBlockWeights` に bias field なし。layer-0 max bias q=51.13 / k=62.49）。reference 妥当性は with-bias mirror vs transformers eager 2.098e-5 で確認 | —（TTS RTF なし。`bench --task cosyvoice2-synthetic` の RTF 0.000018 は scaffold overhead のみで実モデル値ではない。実 forward: 10 tokens/24 層 = median 1.615 s ≈ 0.16 s/token prefill、比較対象なし） | —（e2e 公式 ONNX 不在） | — |

---

## 5. Codec / VAD / Speaker 結果

### 5.1 Codec（ORT baseline なし = 公式 ONNX 不在。精度は pinned PyTorch reference parity — repo CI 設計と同一）

| モデル | 変換 | 実 weight parity（vs PyTorch reference） | Vokra stage RTF | PCM roundtrip |
|---|---|---|---|---|
| Mimi（kyutai tokenizer-e351c8d8、sha256 pin MATCH） | ✅ 518.9 MB・319 tensors（10.7 s） | codebook tables 最悪 5.722e-6（8 面 full 2048x512）/ decode 1.526e-5 — **106 行全 PASS**（atol 0.01、`real_codec_parity` 2/2） | RVQ decode 8-book 6.04e-6 / 32-book 6.44e-5（jfk 相当 11 s、67 µs / 711 µs） | ✗ 未配線: standalone GGUF に seanet config なし + runtime は構造名 bind vs converter は raw moshi 名 passthrough（**T29 owner adapter deferred**、verbatim error 保存） |
| DAC（descript 24 kHz 0.0.4、sha256 pin MATCH） | ✅ 301.1 MB・558 tensors（0.5 s、weight-norm 32 本 fold 済） | codebook / bias **bit-exact 0.0**、weight-norm fold 1.192e-7、decode 4.768e-6 — 全 PASS | RVQ decode 32-book 6.73e-3（825 frames、74.0 ms） | ✗ waveform SEANet が runtime 未実装（quantizer stage のみ） |
| WavTokenizer small-600-24k-4096（MIT、license live 確認） | ✗ converter 未配線（ADR M4-16 §D-d の設計的 defer = 失敗ではない） | **実 codebook [4096,512] + jfk 実 codes 440 個で max Δ = 0.0 bit-identical**（atol 1e-6）。repo 合成 parity 3/3 + fsq lib 18/18、fixtures 8/8 sha256 再現 | `wavtokenizer_vq` 9.5e-6（104.7 µs / 440 frames）— stage のみ | Vokra 側なし。upstream PyTorch roundtrip 参考値: mel-L1 0.3153（noise 基準 1.624 / silence 基準 4.488）、SNR −1.92 dB（0.48 kbps GAN codec の期待挙動）、encode/decode RTF 0.0498 / 0.0699 |
| X-Codec 2 | ✗ 同上 | **実 weight は honest skip（HF `HKUSTAudio/xcodec2` = cc-by-nc-4.0 を live 確認、fail-closed）**。released shape（levels [4;8] = vocab 65536、d_model 2048、550 frames）+ 合成 projection で max Δ = 0.0（atol 1e-5、vqp 1.17.8 pin 一致） | `xcodec2_fsq` 3.9e-4（4.332 ms / 550 frames）— stage のみ | — |

### 5.2 VAD / Speaker

| タスク | エンジン間一致（同一 interface） | 公式 usage / 実用性 | 速度 | 備考 |
|---|---|---|---|---|
| Silero VAD v5（master onnx sha 1a153a22、committed fixture を byte 再現） | raw-512 同一 interface で max Δ ≤ 1.13e-6・mean ~3e-8（3 clips）。harness は committed fixture を 7.893e-8 で再現（SPEC 記載 7.9e-8 と一致） | **P1**: vokra は全 3 clip で **speech 検出 0**（max prob 0.0037/0.0055/0.0363 < 閾値 0.5）。ORT 公式 ctx576（64-sample rolling context 付き [1,576]）は 4/1/1 segments、**IoU = 0.0 全 file**（mean Δ 0.69–0.86）。ORT も raw-512 では同様に崩壊 = **semantics gap であり数値バグではない** | vokra RTF 0.0187–0.0209（single-thread）vs ORT 0.0038–0.0040（intra_op=1、raw512）→ **ORT ~5x 速**。vokra load 0.0007–0.0017 s | 変換は repo 文書化済の both-rate 経路（committed fixture と byte 同一）。production converter の 8 kHz-only gap を実測確認（16 kHz 入力に明示エラー = FR-EX-08 posture 正、能力 gap は実在） |
| CAM++ speaker encoder（FunAudioLLM/CosyVoice2-0.5B 配布の campplus.onnx、Apache-2.0） | wav→192-d embedding の cosine **≥ 0.99999999998**（3 files）、成分 max Δ ≤ 1.87e-5 — **frontend 完全独立**（vokra native Kaldi fbank vs torchaudio）でこの一致 | 話者 ranking 両エンジン一致（小数 6 桁）: intra-speaker 0.872242 ≫ inter 0.28277 / 0.34324 | vokra 0.402 / 0.408 / 0.713 s per-embedding vs ORT 0.070 / 0.044 / 0.066 s（default threads）/ 0.178 / 0.096 / 0.161 s（intra_op=1）→ **~2–6x 遅（thread model 不一致、両設定併記）** | 文書記載の変換元 `ayousanz/campplus-onnx` が HF 404 → 同一 iic/speech_campplus モデルの公式 Alibaba 配布で代替（617 tensors + 2 synthesized、変換 clean） |

---

## 6. 正直な gap 一覧（skip / blocker とその理由）

1. **Voxtral: 推論不可（ran=false、RTF/WER 非算出 = 捏造なし）** — (a) `TextDecoder::load` が GQA 形状を拒否（verbatim: `q_proj.weight shape [4096, 3072] != expected [3072, 3072]`、loader が attn 幅 = hidden を仮定）。加えて (b) lm_head が embed_tokens と**非 tied**（byte 比較で不一致、vokra は tied 前提）、(c) audio encoder の transformer 32 層が T19+ stub（conv-stem のみ）、(d) projector の ×4 frame-stacking（[1500,1280]→[375,5120]）を表現できる AdapterKind が不在。**adapter.json 納品と実 tensor bind 検証は完了**（active=true、Mlp 5120→3072→3072 gelu）。
2. **CosyVoice2: e2e TTS 不可** — `synthesize()` 無条件 NotImplemented（T06 tokenizer / T07-T08 real binder / T10-T11 CFM estimator / T13 codebook / T14-T15 pipeline 残）。in-tree flip harness も stub（`assert_vs_hf_reference` / `LlmWeights::from_gguf` 共 NotImplemented、env-gated テストは load-smoke のみ）。実 weight parity は out-of-tree で実施し **qkv bias 由来の honest FAIL**（§4）。さらに upstream 配布は **Mimi codec を含まない**（FSQ speech_tokenizer_v2 + flow.pt + hift.pt）= vokra MimiBridge 設計と不一致。
3. **X-Codec 2 実 weight: license skip** — HF weight repo が cc-by-nc-4.0（live 確認、ADR M4-16 §D-e の 3-way 齟齬と整合）。fail-closed 維持、op parity は released shape + 合成 projection で代替。
4. **Mimi/DAC の PCM roundtrip 未配線** — Mimi neural chain の実名 adapter は T29 owner deferred、DAC waveform SEANet は runtime 未実装、vokra-cli `run` に codec arch なし。**stage RTF を codec PCM RTF と読み替え禁止**（codes も合成 seed — Vokra encoder が存在しないため）。
5. **FSQ family の GGUF 変換未配線** — converter kinds に wavtokenizer/xcodec2 なし（ADR §D-d で将来 WP へ defer）。converted=false は設計であり失敗ではない。
6. **Kokoro: text/CLI e2e 経路なし + upstream 忠実性 gap** — CLI は kokoro arch 非対応、`TtsEngine::synthesize` NotImplemented → phoneme-id harness で代替。9/9 parity は同一作者 re-implementation の self-consistency 検証に留まる（§8-2 root cause）。
7. **piper-plus の checkpoint 供給問題** — GitHub releases に voice asset なし、canonical v7 zero-shot repo（parity manifest 記載）は HF 401 = private。公開 checkpoint に v7 spk_proj MLP が無く、**数学的中立 spk_proj（ゼロ出力 MLP）を GGUF に付加**（ORT 側 zeros+mask=0 と同一数学、長さ差 0・Δ ≤3 LSB で実証）。唯一の Apache-2.0 voice（mera-multilingual）は dec.ups [64,32,8] vs 期待 [64,32,16] で native ロード不可。評価 voice の license は 'other'（css10-public-domain）= allowlist 外、owner 確認推奨。
8. **whisper small/medium の Metal 転写検証不可** — `run` に `--backend` なし + metal e2e parity test が base-pinned。medium は assignment により corpus 5 files truncation + parity/Metal leg skip（他サイズと WER 横比較不可）。
9. **whisper tensor parity は committed synthetic fixture 対比** — real-audio fixture の regen は repo 書込みを要するため read-only 規約下で見送り（owner T09 の regen→review→手動 commit が残）。
10. **ORT baseline 不在 leg** — Mimi/DAC/WavTokenizer/X-Codec 2/CosyVoice2 LLM は公式 ONNX 不在、Voxtral は 3B seq2seq の local export 非現実的（assignment 指定でも省略）→ PyTorch reference parity を精度指標とした。
11. **計測ノイズ / 導出値** — sibling agent 並走による jitter（medium 最大 11.6 s、Kokoro ORT RTF 0.51–0.88、CAM++ spread 等）。vokra load は CLI 非露出のため derived 値。EN piper round-trip WER 1.0 は両エンジン共通の voice 特性（波形ほぼ同一で立証）でエンジン評価に非算入。

---

## 7. Owner checklist への影響（docs/m4-owner-verification-checklist.md）

**今回ローカルで討ち取られたもの（substance ベース、fabricated pass なし）**:

- **§1.3 / M4-14（Whisper small/medium/turbo 実 weight 初回）**: 3 サイズとも実 checkpoint 変換（shape 自動検出が turbo の非対称形状 587 tensors を正しく処理）+ 実音声 e2e + ORT 転写一致を初めて実証。**T10 の turbo atol 判断材料が確定**: max Δ 4.357e-5（encoder）等で default atol 0.01 に ~3 桁の余裕 = **honest calibrate 不要見込み**。**残**: T09 real-audio fixture 再生成 + owner review commit、T10 `parity-whisper-real.yml` 3-size 初回 workflow_dispatch、T11 license sign-off（§3.3 — MIT の事実材料は揃った）。
- **§1.6 / M4-04-T21（Mimi/DAC real-checkpoint parity）**: `parity-rvq-real.yml` の recipe をローカルで**完全 first-fire**（checkpoint sha256 が workflow pin と MATCH、venv pin 一致、106/106 テンソル PASS、`real_codec_parity` 2/2）。**残**: GitHub Actions 上の正式初回 dispatch（+ encodec leg は今回未実行）。
- **§1.7 / M4-16（WavTokenizer 実 parity の前倒し）**: checklist 上「実 weight e2e は後続 WP」だった WavTokenizer を実 weight + 実 codes で bit-identical 0.0 まで討ち取り。**X-Codec 2 は §3.4 T14 の判断材料を更新**（HF tag cc-by-nc-4.0 の live 一次確認 raw JSON 保存）— NC 確定なら `license_class.rs` の Permissive 分類差し戻し（checklist §3.4(a) 記載どおり）が必要。
- **M3 carry-over: M3-10 Voxtral real safetensors inspection → adapter.json**: **納品 + 実 tensor bind 検証済 = 討ち取り**。ただし real ASR は新規 blocker 4 件（§6-1）で CC fixup 差し戻しへ（checklist Contact/Escalation 経路）。
- **M3 carry-over: M3-09 CosyVoice2 real HF checkpoint（flip-the-switch）**: 実 llm.pt 入手 + T02-style manifest（llm-pt-manifest.tsv）+ 実 weight forward を発火。結果は **honest FAIL（qkv bias）** — `LlmBlockWeights` bias 対応 + converter T04（hparams 0-placeholder）の CC 差し戻しが確定。upstream に Mimi が無い設計不一致（FSQ+flow+HiFT）も owner/CC 判断事項として新規記録。
- **M2 carry-over: Kokoro real-checkpoint parity**: CI recipe ローカル再現で 9/9 PASS（f0 2.619e-2、CI 履歴 3.268e-2 より良）。**ただし upstream fidelity gap の発見（§8-2）により、この 9/9 の意味が「upstream 一致」ではなく「re-implementation self-consistency」であることが判明** → fixture 再生成 + 実装修正の CC 差し戻し判断が owner に発生（既存の PROSODY_F0_ATOL follow-up より上位の課題）。
- **whisper-base（M2 CI は既に green）**: 本キャンペーンで初の**クロスエンジン WER/CER 比較**（byte-identical 9/9）と **Metal 転写一致**（bonus 7b）を追加。

**checklist 外だが owner triage が必要な新規事項**: Silero VAD ctx576 P1（M0 出荷済 VAD の製品可用性に直結、§8-1）／piper-plus v7 canonical checkpoint の private 状態（parity manifest の再現性に関わる）／css10 voice license 'other' の確認。

**本キャンペーンで未着手のまま残る checklist 項目**: §1.1 CSM・§1.2 Moshi・§1.4 DeepFilterNet・§1.5 UTMOS/DNSMOS、§2 実機群（Hopper/RISC-V/Android Vulkan/Web ブラウザ/CPU ISA/full-duplex）、§4 infra（npm/CDN/glslc/UNITY_LICENSE/CI dispatch 群）、§5 owner ADR。

---

## 8. 発見されたバグ / P1 issue 一覧

| # | 深刻度 | 対象 | 内容 |
|---|---|---|---|
| 1 | **P1（製品）** | Silero VAD | 公式 v5 wrapper の 64-sample rolling context（入力 [1,576]）を native subgraph が持たず [1,512] 直入力 → 実音声 3 clips で max prob 0.0037–0.0363、**検出 segment 0**（ORT 公式 mode は 4/1/1）。エンジン間 parity 1.13e-6 なので**数値でなく semantics の欠落**。committed 合成 fixture（ref max prob 0.0020）ではこの欠陥を検出できなかった（テスト設計の盲点） |
| 2 | **P1（忠実性）** | Kokoro-82M | vokra native + 自前 reference dumper が相互一致する一方、true upstream（kokoro 0.9.4 oracle + ONNX export、両者 ≤1.7e-5 で相互一致）と乖離。root cause 5 点をソースで特定: (a) LRELU_SLOPE 0.1 vs upstream LeakyReLU(0.2)、(b) ALBERT 共有層 4x 適用 vs config 12 層 + gelu vs gelu_new、(c) decoder 入力が length-regulated BERT 特徴 vs upstream の text_encoder t_en、(d) F0/N contour がゼロ vs F0Ntrain 出力、(e) duration ceil(exp) vs round(sigmoid-sum)。**実文出力は unintelligible（WER 1.0）** |
| 3 | **P1（blocker）** | Voxtral runtime | `TextDecoder::load`（text_decoder.rs:141,144）が q_proj/o_proj [hidden,hidden] を要求し head_dim = hidden/n_head を仮定 → 実 checkpoint（32 heads × 128 = 4096 ≠ 3072）をロード不能 |
| 4 | **P1（blocker）** | CosyVoice2 runtime | `LlmBlockWeights` に q/k/v bias field がなく、実 checkpoint（layer-0 max bias k=62.49）で logits max Δ 12.92・argmax 0/10。bias field 追加 or bias folding が必要（vokra の数学自体は bias-less mirror と 1.45e-4 一致で正しい） |
| 5 | **P1（converter）** | Voxtral 変換 | `is_float_dtype` が BF16 を欠く → 生 bf16 shard の変換が「0 float weights written, 152 non-float skipped」で **exit 0 のまま 1,696 B の weightless GGUF** を出力（success-shaped silent failure、FR-EX-08 精神に反する）。回避 = 事前 f16 precast（本 checkpoint では実測 lossless: inf 0 / flush-to-zero 9895/4.68e9） |
| 6 | P2（converter） | Voxtral 変換 | shape table が mini=28 層想定 vs 実 30 層、`derive_name` の失敗を `unwrap_or("voxtral-unknown")` で silent fallback（自身の FR-EX-08 docstring と矛盾）。CLI の `--config` 未使用（sentinel hparams + tokenizer なし GGUF）と `run` の voxtral 非 route も既知 T04 残 |
| 7 | P2（converter） | CosyVoice2 変換 | arch hparams を 0-placeholder で書く（全て tensor shape から導出可能なのに）→ 実 GGUF で `llm=None` bind（T04 残） |
| 8 | P2（model gap） | Voxtral | lm_head 非 tied（tied 前提と不一致）／audio encoder transformer 32 層 stub（T19+ 明記済）／×4 frame-stack adapter kind 不在 — GQA 修正後も real ASR には 3 点の追加対応が必要 |
| 9 | P2（性能） | CPU 全般 | ORT CPU 比: whisper-small 2.90–3.11x / medium 7.4–7.8x / turbo 1.81x / Kokoro 4.27x / piper-plus 5.17–8.28x / Silero VAD ~5x / CAM++ 2–6x 遅（base のみ jfk 互角・corpus 1.44x 劣）。短発話（= 30 s pad 比率大）で gap 集中 |
| 10 | P2（互換） | piper-plus loader | mera-multilingual（唯一の Apache-2.0 公開 voice）が dec.ups.2.weight [64,32,8] vs 期待 [64,32,16] でロード不可 — decoder upsample geometry の可変対応がない |
| 11 | P3 | vokra-cli | `convert --help` が mimi/dac/csm/moshi/cosyvoice2 を列挙しない（実装は受理 = stale help）／`run`/`bench` の `--text` route が placeholder char-tokenizer で JA をほぼ全 drop（文書化済）／bench JSON の `rtf` が mean ベース（median 要件と不整合、再導出を強制）／model load 時間が非露出／`run` に `--backend` なし（Metal 転写検証を阻害） |
| 12 | P3 | Silero converter | `--model silero-vad` が 8 kHz-only GGUF を出力（If 分岐 de-dup、既知文書化済。16 kHz 入力への明示エラー自体は FR-EX-08 準拠） |

**バグではない確認事項**: turbo の M4-14 P1 懸念は非発現（8/8、余裕 ~3 桁）。FSQ fixtures は torch 2.5.0→2.8.0 間で 8/8 sha256 再現。WavTokenizer roundtrip の負 SNR（−1.92 dB）は 0.48 kbps GAN codec の期待挙動（mel-L1 0.3153 が有意指標）。Kokoro ONNX の波形 max Δ 0.44 vs oracle は upstream decoder の確率的 SineGen 由来（内部テンソル・duration は一致）。

---

## 9. 成果物の所在

- 生ログ・per-file TSV・転写・bench JSON・スクリプト: `/Users/inamotoyuuta/.cache/vokra-eval/out/{asr-whisper-base,asr-whisper-small,asr-whisper-medium,asr-whisper-turbo,asr-voxtral,tts-piper,tts-kokoro,tts-cosyvoice2,codec-mimi-dac,codec-fsq,vad-spk}/`
- 変換済 GGUF: `/Users/inamotoyuuta/.cache/vokra-eval/gguf/`（whisper-base/-small/-medium/-turbo、piper-plus-css10-ja-6lang-neutralspk、kokoro-82m、cosyvoice2-0.5b-llm、mimi、dac、silero-vad 系、campplus、voxtral-mini-3b-f16-full）
- 全 agent とも repo は read-only 運用（HEAD `0b7e7caa`、git status clean を各 leg で確認済）