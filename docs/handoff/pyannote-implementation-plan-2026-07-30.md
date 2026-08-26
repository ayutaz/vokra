# pyannote 実装計画 handoff (2026-07-30)

> **2026-08-18 status review:** This is the original implementation plan, not
> a live list of missing files. The segmentation and weightless-pipeline
> converters, `vokra.pyannote{,_pipeline}.*` metadata, native PyanNet
> SincNet/BiLSTM/linear/classifier path, diarization scaffolding, CLI segment
> dispatch, and env-gated real-GGUF harness have landed. The real forward is
> deliberately not the default: it requires
> `VOKRA_PYANNET_ENABLE_FORWARD=1` until an upstream-independent probability
> dump pins numerical parity. Full default-on diarization and publication
> therefore remain pending. Treat the wave/file lists below as dated design
> provenance; use `crates/vokra-models/src/pyannote/` and
> `crates/vokra-models/tests/parity_pyannote_segmentation.rs` for current
> behavior.

> **2026-08-26 artifact correction:** The immutable public
> `vokra/pyannote-segmentation-3.0@50bf4e510e0c689668384aec0f866f02e0fcaea8`
> header contains 54 F32 tensors and recurrent tensors l0 through l3 in both
> directions. The preserved release config also sets `lstm.num_layers: 4`.
> Therefore the older two-layer/BF16-pass-through blueprint below was a
> pre-artifact class-default assumption and is superseded by the strict
> four-layer release contract.

## 背景 — なぜ本 handoff か

**License half unblock (2026-07-30)**: `docs/license-audit.md` §3.1 row 263 で pyannote weight license を primary source (authenticated HF API `pyannote/segmentation-3.0` + `pyannote/speaker-diarization-3.1` cardData tag `license: mit`、`gated: auto` は access control のみで追加条項なし) 確認済、CC 判断で ☑ Commercial sign-off (2026-07-30 yousan)。

**Trigger half は M5-residual**: `crates/vokra-core/src/m5_residual_ops.rs` の `DIARIZE_OP` (FR-OP-82) は op 未実装で residual anchor のまま。従前の "trigger + license (pyannote HF-gated) double blocker" は本 wave で "trigger only" に縮小 (blocker text 更新済)。

本文書は、残 trigger 側 (converter + runtime + parity) を実装する follow-up wave のための **完全な blueprint** を primary source ベースで scope out する。想像や推定を含まない (CLAUDE.md 「ハルシネーション厳禁」)。

## モデル 2 種の position

pyannote は **2 種の HF asset で 1 pipeline を構成する** 特殊な位置付け:

### pyannote/segmentation-3.0 (VAD backbone) — CC 実装対象

- **HF**: `pyannote/segmentation-3.0`
- **License**: MIT (primary source: HF cardData tag `license: mit`、2026-07-30 CC 直接照合)
- **Gated**: auto (HF UI accept で誰でも DL 可、追加条項なし)
- **Pipeline tag**: `voice-activity-detection`
- **Library**: `pyannote-audio`
- **Files** (HF API `siblings`):
  - `config.yaml` — architecture hparams
  - `pytorch_model.bin` — weight file (torch pickle, need `bin_to_safetensors.py` bridge)
  - `LICENSE` — canonical MIT
  - `README.md` + `example.png`
- **Model class**: PyanNet (下記 §Architecture)

### pyannote/speaker-diarization-3.1 (pipeline config) — Configuration only

- **HF**: `pyannote/speaker-diarization-3.1`
- **License**: MIT (primary source: HF cardData tag `license: mit`、2026-07-30 CC 直接照合)
- **Gated**: auto
- **Pipeline tag**: `automatic-speech-recognition` (**misleading — actual = speaker-diarization**)
- **Library**: `pyannote-audio`
- **Files** (HF API `siblings`):
  - `config.yaml` — **pipeline** config (references segmentation + embedding models)
  - `handler.py` — inference handler (Python, not portable)
  - **No `pytorch_model.bin`** — this is a *pipeline* not a model
- **Composition** (per pyannote-audio pipeline convention):
  - Segmentation backbone → `pyannote/segmentation-3.0` (above)
  - Speaker embedding → likely `pyannote/wespeaker-voxceleb-resnet34-LM` or similar
  - Clustering algorithm → runtime function (agglomerative hierarchical clustering)

**Implication**: 実装スコープの最小単位は `pyannote/segmentation-3.0` = PyanNet single model。diarization-3.1 は segmentation output → speaker embedding → clustering の runtime pipeline で、CAM++ speaker encoder が既に land 済 = Vokra は独自 pipeline を組める。**pyannote-audio 側の pipeline を wrap する必要はない** (whisper.cpp 型 clean-room re-imp の判断、CLAUDE.md 設計判断 4)。

## PyanNet architecture (primary source: MIT LICENSE)

**Source**: `github.com/pyannote/pyannote-audio/develop/src/pyannote/audio/models/segmentation/PyanNet.py` (CC 直接 fetch 2026-07-30、MIT LICENSE Copyright (c) 2020 CNRS)。**推定は含まない**。

### Forward path

```text
waveforms (batch, channel=1, samples)  # 16 kHz mono PCM
  -> SincNet frontend
     - stride=10 (default)
     - sample_rate=16000
     - output: (batch, 60, num_frames)  # 60 features per frame
  -> rearrange "batch feature frame -> batch frame feature"
  -> LSTM (monolithic=True default)
     - nn.LSTM(input_size=60, hidden_size=128, num_layers=2,
               bidirectional=True, batch_first=True)
     - output: (batch, num_frames, 256)  # 2 * 128 bidirectional
  -> Linear stack (num_layers=2, hidden_size=128)
     - Linear(256, 128) + leaky_relu
     - Linear(128, 128) + leaky_relu
  -> Classifier
     - Linear(128, dimension)
     - dimension = num_powerset_classes for segmentation-3.0
  -> Activation (default = self.default_activation(), typically Softmax
     for powerset multiclass)
```

**Output shape**: `(batch, num_frames, num_powerset_classes)` = per-frame class distribution.

### Powerset multiclass (segmentation-3.0)

pyannote 3.0 uses **powerset multiclass encoding** — segmentation is a joint classification of "which subset of speakers is active at this frame", not per-speaker binary. For segmentation-3.0 the config sets 3 speakers × 2 overlap = **7 classes**:
- Class 0: silence
- Class 1: speaker A only
- Class 2: speaker B only
- Class 3: speaker C only
- Class 4: A+B overlap
- Class 5: A+C overlap
- Class 6: B+C overlap

(Actual class count = `num_powerset_classes` in config.yaml, verify per-checkpoint.)

### SincNet frontend detail

**Source**: `github.com/pyannote/pyannote-audio/develop/src/pyannote/audio/models/blocks/sincnet.py` (要 fetch by implementer)。SincNet 一般的な構造:

- **Sinc conv1d** (learnable band-pass filter): first layer, output channels 80、kernel_size 251、stride 10 (default)
- **Conv1d + MaxPool1d + LayerNorm** stack: 2 blocks (output 60 features)
- **Total stride**: 10 (from `sincnet["stride"]` default)
- **Receptive field**: computed per `sincnet.receptive_field_size()` 

**Implementation notes**:
- Sinc filter = `2 * f2 * sinc(2π f2 t) - 2 * f1 * sinc(2π f1 t)`, where `f1, f2` are learnable per filter, `sinc(x) = sin(x)/x`
- Filter length must be odd; hamming window applied
- Filters initialized to mel-scale spaced

**Vokra 実装候補**: `crates/vokra-ops/src/sincnet.rs` (new op) or `crates/vokra-models/src/pyannote/sincnet.rs` (private module)。zero-dep 前提で自前 Rust 実装 (torch 依存排除)。

## Tensor manifest (pyannote/segmentation-3.0)

**Fetch method** (CC が実 checkpoint に触れずに tensor 名を確認する手段):

```bash
# 手順（HF gate accepted 後に VAST で実行）:
uv run --project tools/parity --frozen python - <<'PY'
import torch
w = torch.load('pytorch_model.bin', map_location='cpu', weights_only=True)
for k, v in w.items():
    print(f'{k}\\t{list(v.shape)}\\t{v.dtype}')
PY
```

**Expected pattern** (from PyanNet.py source):
- `sincnet.conv1d.0.weight` (sinc filter learnable params)
- `sincnet.conv1d.[1-2].weight`, `sincnet.conv1d.[1-2].bias`
- `sincnet.norm1d.[0-2].weight`, `sincnet.norm1d.[0-2].bias`
- `sincnet.pool1d.*` (MaxPool state, if stateful)
- `lstm.weight_ih_l0`, `lstm.weight_hh_l0`, `lstm.bias_ih_l0`, `lstm.bias_hh_l0` (bidirectional: also `_reverse` variants for both layers)
- `lstm.weight_ih_l1`, `lstm.weight_hh_l1`, ... (2nd layer)
- `linear.0.weight`, `linear.0.bias` (Linear(256, 128))
- `linear.1.weight`, `linear.1.bias` (Linear(128, 128))
- `classifier.weight`, `classifier.bias` (Linear(128, 7))

**Verify by owner or CC after HF gate accept**:
1. On VAST, HF login:
   `uv run --project tools/parity --frozen hf auth login`
2. Accept gate for `pyannote/segmentation-3.0`
3. Download:
   `uv run --project tools/parity --frozen hf download pyannote/segmentation-3.0 pytorch_model.bin`
4. Run tensor dump script above
5. Compare with expected manifest, update this handoff

## 実装 wave scope (follow-up)

### Wave 1 — Converter + config + license class (CC-actionable)

- `crates/vokra-convert/src/models/pyannote_segmentation.rs` (new file)
  - `bin_to_safetensors.py` bridge for `pytorch_model.bin` (owner-run, tools/parity)
  - Convert safetensors → GGUF with `vokra.pyannote.*` chunk group:
    - `vokra.pyannote.sample_rate` = 16000
    - `vokra.pyannote.sincnet.stride` = 10
    - `vokra.pyannote.lstm.hidden_size` = 128
    - `vokra.pyannote.lstm.num_layers` = 4 (2026-08-26 exact-release correction)
    - `vokra.pyannote.lstm.bidirectional` = true
    - `vokra.pyannote.linear.hidden_size` = 128
    - `vokra.pyannote.linear.num_layers` = 2
    - `vokra.pyannote.num_powerset_classes` = 7
    - `vokra.provenance.license` = "mit"
    - `vokra.provenance.upstream_hf` = "pyannote/segmentation-3.0"
    - `vokra.provenance.upstream_revision` = <HF revision SHA>
- Strict 54-tensor F32 manifest; foreign dtypes/topologies fail closed
- `ModelKind::PyannoteSegmentation` + CLI dispatch in `crates/vokra-convert/src/main.rs`
- `crates/vokra-cli/src/convert.rs` help text update

**Effort estimate**: 2-3 tickets (~1.5 h CC time)

### Wave 2 — Runtime module scaffold with loud-partial forward (CC-actionable、architecture-aware)

- `crates/vokra-models/src/pyannote/mod.rs` (new)
- `crates/vokra-models/src/pyannote/config.rs`:
  - `PyanNetConfig` struct with all hparams from `vokra.pyannote.*` chunk group
  - `PyanNetConfig::from_gguf(gguf)` = fail-closed hparam parse (FR-EX-08)
  - Primary-source constant fallback for a GGUF that never carried the chunk (mirror RMVPE pattern)
- `crates/vokra-models/src/pyannote/weights.rs` (new):
  - `PyanNetWeights` struct binding every tensor from the manifest
  - `PyanNetWeights::from_gguf(gguf)` = required-tensor load (missing/mis-shaped/wrong-dtype → `VokraError::ModelLoad`)
  - GGUF with no upstream PyanNet tensors → loud refuse (no all-zero silent forward)
- `crates/vokra-models/src/pyannote/mod.rs::PyanNet`:
  - `PyanNet::new_from_gguf(gguf, config)` real load
  - `PyanNet::segment(pcm: &[f32]) -> Result<Vec<[f32; N_CLASSES]>>` = loud-partial `VokraError::UnsupportedOp` until Wave 3 (real forward)
  - `PyanNet::num_frames(num_samples)` real receptive-field arithmetic (from PyanNet.py `sincnet.num_frames()`, CC が primary source から port)

**Effort estimate**: 5-8 tickets (~4 h CC time)

**Loud-partial rationale** (RMVPE `docs/handoff/residual-wave3-2026-07-30.md` pattern):
- `from_gguf` は real load (mis-shaped tensor → loud error)
- `num_frames` は real receptive-field arithmetic (config から algebraic)
- **`segment` の内部 forward だけ defer** → SincNet + LSTM + Linear + powerset decoder のうち LSTM + Linear は既存 primitive で実装可能だが、**SincNet は Vokra に不在の新規 op** = 実装コストが高い。Wave 3 で fresh scope として起票。
- Loud-partial は honest (fake-complete より良い、fail-closed with clear error message)。周辺 module は real で iterate 可能 = defer と best-guess の間の第三の道 (memory [[project-huggingface-vokra-publication]] pattern)。

### Wave 3 — SincNet primitive + real forward (architecture-heavy)

- `crates/vokra-ops/src/sincnet.rs` (new op) — learnable sinc conv1d
  - `sinc_conv1d_forward()` = 実 sinc kernel synthesis + conv1d
  - `sinc_conv1d_backward()` = **NOT NEEDED** (inference-only, per NFR-DS-02)
- `crates/vokra-models/src/pyannote/mod.rs::PyanNet::segment` real implementation:
  - Call `sincnet_forward()` (Vokra port of pyannote-audio SincNet)
  - Monolithic BiLSTM forward (existing `vokra_ops::lstm_bidirectional_forward()`)
  - Linear stack + leaky_relu (existing primitives)
  - Classifier + Softmax (existing primitives)
  - Powerset multiclass output
- Powerset decoder helper (`decode_powerset(class_probs, num_speakers) -> Vec<SpeakerActivity>`)

**Effort estimate**: 10-15 tickets (~8 h CC time)

**Blockers**:
- SincNet primary source (`sincnet.py` from pyannote-audio) 詳細 study が必要
- Real weight parity harness (新規 `crates/vokra-models/tests/parity_pyannote_segmentation.rs`、env-gated `PARITY_PYANNOTE_REAL_GGUF`) の owner-provisioned GGUF が必要 (owner の HF gate accept + DL 後)。**crate 名注意**: `crates/vokra-parity` は存在しない — root `Cargo.toml` の comment に名前が出るだけで、実 parity test 34 本はすべて `crates/vokra-models/tests/parity_*.rs` にある

### Wave 4 — Diarization pipeline (Vokra 独自 assembly、Vokra-native)

pyannote pipeline の Python 版に依存せず、Vokra native の diarization pipeline を組む:

- Segmentation (Wave 3 の PyanNet.segment output)
- Speaker embedding (既存 `speaker_encode` op = CAM++ 経路)
- Clustering (agglomerative hierarchical clustering、Vokra `crates/vokra-ops/src/clustering.rs` new)
- RTTM output writer (`.rttm` format, standard diarization file format)

**Effort estimate**: 6-10 tickets (~5 h CC time)

**Dependency**: Wave 3 完了後。CAM++ 既 land 済 (Apache-2.0、`speaker_encode` op)。

## Owner タスク (CC 越し不能)

1. **HF gate accept** for `pyannote/segmentation-3.0` + `pyannote/speaker-diarization-3.1` (HF UI で非拘束 advisory の accept ボタンをクリック、Vokra 配布側は Meta Llama tokenizer と同じ non-bundle 方式 = consumer 側 accept)
2. **pytorch_model.bin download** (~5.7 MB、gate accept 後は誰でも DL 可)
3. **Tensor manifest verify** (実 checkpoint で上記 Expected pattern を確認、drift があれば本 handoff 更新)
4. **Real weight parity harness owner run** — VAST で
   `PARITY_PYANNOTE_REAL_GGUF=<path> VOKRA_PYANNET_ENABLE_FORWARD=1 cargo test -p vokra-models --test parity_pyannote_segmentation -- --nocapture`
   を実行し、upstream-independent probability dump との parity を判定
5. **§3.1 publish sign-off** — pyannote weight を `huggingface.co/vokra/pyannote-segmentation-3.0` へ mirror publish するか (weight license MIT clean、Vokra converter output GGUF の再配布)、mirror でなく consumer-side download で済ませるか (Meta Llama tokenizer 前例と同じ non-bundle)

## verify (本 handoff の primary source claim)

- HF cardData license verified via authenticated API (`api/models/pyannote/segmentation-3.0` 2026-07-30 CC 直接 fetch = `license: mit, gated: auto`)
- PyanNet.py source verified via GitHub API + raw fetch (`github.com/pyannote/pyannote-audio/develop/src/pyannote/audio/models/segmentation/PyanNet.py` = MIT LICENSE header + full class definition CC 直接 fetch 2026-07-30)
- Powerset multiclass class count = docstring から algebraic (3 speakers × 2 overlap = 7)、実 config.yaml 確認は owner の HF gate accept 後 (推定を書かない = CLAUDE.md「ハルシネーション厳禁」)

## 参考

- ADR (要 新規起票): `docs/adr/M5-XX-pyannote-diarize.md` — Wave 3 開始時に scope + red-line を fix
- Registry pin: `crates/vokra-core/src/compliance/license_class.rs` (2026-07-30 land 済)
- DIARIZE_OP anchor: `crates/vokra-core/src/m5_residual_ops.rs::DIARIZE_OP` (2026-07-30 blocker text 更新済)
- §3.1 sign-off: `docs/license-audit.md` row 263 (2026-07-30 yousan ☑ Commercial CC 判断)
- CAM++ speaker encoder (Wave 4 diarization pipeline の依存): `crates/vokra-models/src/speaker/camplus.rs` (綴りは `camplus`、`p` 一つ — converter 側の `crates/vokra-convert/src/models/campplus.rs` は `p` 二つで非対称ゆえ注意)
- Loud-partial precedent: RMVPE (`crates/vokra-models/src/f0/rmvpe.rs`) + Charsiu (`crates/vokra-models/src/align/charsiu.rs`)
