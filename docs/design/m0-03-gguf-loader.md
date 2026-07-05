# M0-03 設計メモ: GGUF ローダー / 変換ツールの対応範囲と決定事項

- **チケット**: M0-03-T01（対応範囲の確定記録）。T12/T16 の tensor 命名契約、T05 の zero-copy 設計注記、T13/T16/T17 のローカル実行結果を含む
- **日付**: 2026-07-02
- **実装**: `crates/vokra-core/src/gguf/`（reader/writer/FrontendSpec/chunks）、`crates/vokra-convert/`（オフライン変換ツール、FR-TL-01）
- **関連文書**: `docs/design/vokra-gguf-chunks.md`（`vokra.*` chunk 命名仕様 = M0-03-T08）

## 1. GGUF 対応範囲（出典付き）

出典: ggml-org/ggml `docs/gguf.md` @ `eced84c86f8b`（2026-06-26 取得。以下「仕様書」）。

| 項目 | 対応 | 仕様書の根拠 |
|------|------|-------------|
| magic | `0x47475546`（"GGUF" をリトルエンディアンで読んだ u32） | gguf_header_t: 「Must be `GGUF` at the byte level: 0x47 0x47 0x55 0x46」 |
| version | **3 のみ受理**（それ以外は明示エラー） | 「Must be `3` for version described in this spec」 |
| エンディアン | リトルエンディアンのみ（BE 判別手段が仕様に無いため） | 「If no additional information is provided, assume the model is little-endian」 |
| metadata value type | **全 13 種**（UINT8=0〜FLOAT64=12、ARRAY=9 はネスト対応） | `gguf_metadata_value_type` enum |
| alignment | `general.alignment`（UINT32、8 の倍数）。**未指定時は 32** | 「If the alignment is not specified, assume it is `32`」 |
| tensor offset | tensor_data 起点、ALIGNMENT の倍数を検証 | gguf_tensor_info_t: 「Must be a multiple of `ALIGNMENT`」 |
| key 制約 | ASCII / lower_snake_case / `.` 区切り / ≤65535 bytes | gguf_metadata_kv_t の caveats |
| tensor name | ≤64 bytes（仕様値。**M0 writer は非強制** — Vokra 内 round-trip は可、厳格な GGUF 互換強制は将来検討。Whisper base の HF 名は全 245 本が上限内） | gguf_tensor_info_t の caveat |
| **tensor dtype（M0 範囲）** | **F32=0 / F16=1 のみ**。範囲外は明示エラー | `ggml_type` enum。K-quants（Q4_K=12/Q5_K=13/Q6_K=14）の**ロードは FR-LD-07 = v0.1 MVP（M1-02 管轄）でスコープ外** |

## 2. checkpoint 入手元の決定（出典付き）

| モデル | 形式 | 入手元 | 実測 |
|--------|------|--------|------|
| **Whisper base** | **safetensors**（FR-MD-02: 上流 checkpoint のみ、モデルコード非依存） | HuggingFace `openai/whisper-base` の `model.safetensors` | F32 のみ **245 tensors**、290,403,936 bytes（2026-07-02 取得・ヘッダ実検） |
| **Silero VAD v5** | **ONNX**（変換ツール内のみで扱う = FR-LD-05） | `snakers4/silero-vad` @ `dbacf536adad` の `src/silero_vad/data/silero_vad.onnx` | 2,327,524 bytes。**16k/8k 両対応の単一ファイル**（top-level `If` の then/else 分岐） |

- Whisper の frontend_spec 値の出典: `openai/whisper` @ `04f449b8a437` `whisper/audio.py` — SAMPLE_RATE=16000 / N_FFT=400 / HOP_LENGTH=160 / hann window / center=True（torch.stft default）/ n_mels=80（base）/ mel filter = `librosa.filters.mel(sr=16000, n_fft=400)`（librosa default = Slaney 正規化・非 HTK・fmin 0.0・fmax 8000.0）。転記実体は `crates/vokra-convert/src/models/whisper.rs` の rustdoc。
- ONNX デコードの方式（T14 決定）: **手書き最小 protobuf デコーダ**（varint + length-delimited、`crates/vokra-convert/src/onnx.rs`）。prost / protobuf crate は `vokra-convert` であっても追加しない（`deny.toml` が protobuf 系を ban 済み、依存隔離を依存グラフ上も自明にするため）。フィールド番号の出典: onnx/onnx @ `a5934724a474` `onnx/onnx.proto`（TensorProto: dims=1, data_type=2, float_data=4, name=8, raw_data=9 / GraphProto: initializer=5, node=1 / ModelProto: graph=7 / DataType: FLOAT=1, FLOAT16=10）。

## 3. llama.cpp 命名衝突回避の根拠

仕様書「Standardized key-value pairs」は `general.*` / `<arch>.*` / `tokenizer.*` を標準 namespace とし、「The community can develop their own key-value pairs … should be namespaced with the relevant community name to avoid collisions. For example, the `rustformers` community might use `rustformers.` as a prefix」と明記する。Vokra は **`vokra.` prefix** を採用し（FR-LD-02 / IF-07）、標準 namespace と構造的に衝突しない。詳細は `docs/design/vokra-gguf-chunks.md`。

## 4. tensor 命名契約（M0-06 / M0-05 との共有事項）

- **Whisper base（M0-06 が参照する契約）**: GGUF tensor 名 = **上流 HF safetensors 名の恒等写像**（例 `model.encoder.layers.0.self_attn.q_proj.weight`）。改名レイヤは設けない（stable な上流名をそのまま契約とする。Vokra 側命名の導入は将来の変更余地として `whisper.rs` rustdoc に記録）。次元順は **source order（PyTorch row-major、外側次元が先）**で格納し、ggml の `ne[]` 逆順には**しない** — 消費側（M0-06）も同順で読む。
- **Silero VAD v5（M0-05 が参照する契約）**: v5 ONNX は top-level `graph.initializer` を持たず、weight は `If` の then/else 分岐内 `Constant` ノードの `value` に格納される。GGUF tensor 名 = **各 Constant ノードの出力名**。両分岐が同名 weight を再計算するため**重複名は de-dup（初出のみ書込）**。実測: **float weight 19 本を書込 / 非 float 定数 307 本 skip（int64 shape/index 等、M0 dtype 範囲外）/ 重複 15 本 de-dup**、出力 944,160 bytes。
  - **M0-05 への検証事項**: 19 本が 16k 経路の全 weight を被覆するかは、M0-05 の 1:1 subgraph 結線時に確認する（8k/16k 分岐の weight 共有構造のため、tensor 完全性の最終判定は graph 再構築側でしか行えない）。出力サイズ 944KB は CLAUDE.md 記載「v5 約 2MB」より小さいが、これは非 float 定数 skip と de-dup による（重複 15 本 ≈ 全 float の 44%）。

## 5. zero-copy 戦略（T05 の設計注記と逸脱）

- チケット提案は `memmap2` だったが、M0 実装は **`std::fs::read` 全読み + `&[u8]` slice 貸出**とした。理由: (a) `vokra-core` は workspace lint で `unsafe_code = "deny"`（NFR-RL-07 の「コアは 100% safe Rust」）であり、mmap は unsafe API（`memmap2::Mmap::map` は unsafe fn）を要する、(b) M0 は runtime 外部依存ゼロを維持した（`cargo tree -p vokra-core` = 依存ゼロ、NFR-DS-02 の最強形）。
- ロードパスはヘッダ + metadata + tensor info のパース後、weight 本体はパース済みバッファからの **コピーなし slice 貸出**（offset は境界チェック済み）。
- **真の mmap（cold start ほぼゼロ = FR-LD-01 / NFR-PF-11）は当初 M1-02 の followup としたが、専用 micro-crate `vokra-mmap` として実装済み（2026-07-06 追記）**: unsafe 境界は「専用 micro-crate」案を採用（`gguf::reader` モジュール限定 allow は不採用）。seam は `vokra-core` の `pub trait AsBytes: Send + Sync`（`fn bytes(&self) -> &[u8]`）+ `GgufFile::from_external(Box<dyn AsBytes>)` で、`vokra-core` は unsafe-free のまま owned buffer と mmap を同一パーサ・同一 zero-copy accessor で処理する（内部 `GgufBytes::{Owned, External}`）。`vokra-mmap` は read-only mapping `Mmap`（`impl AsBytes`、`Send + Sync`、`Drop` で解放）と `open_gguf(path) -> Result<GgufFile, GgufError>` を提供。**`memmap2` は導入せず**、POSIX `mmap`/`munmap`（`PROT_READ | MAP_PRIVATE`）・Win32 `CreateFileMappingW`/`MapViewOfFile`/`UnmapViewOfFile`/`CloseHandle`（`PAGE_READONLY`/`FILE_MAP_READ`）を `unsafe extern` ブロックで自前宣言 — std が既に libc/kernel32 をリンクするため Cargo.lock に外部 crate を追加せず zero-dep（NFR-DS-02）を維持する（memmap2 採用案より強い保証）。`gguf/reader.rs` モジュール docs も実装済みの記述に更新済み。

## 6. ローカル実行検証の結果（T13/T16/T17、2026-07-02 実測）

```
$ vokra-convert --model whisper-base --input whisper-base.safetensors --output whisper-base.gguf
converted whisper-base: 245 tensors, 15 metadata keys, 290395648 bytes
verified load: version 3, alignment 32, 245 tensors, 15 metadata keys;
               frontend n_fft=400 hop=160 n_mels=80 sample_rate=16000

$ vokra-convert --model silero-vad --input silero_vad.onnx --output silero-vad-v5.gguf
converted silero-vad: 19 tensors, 2 metadata keys, 944160 bytes
  note: 19 float weights written, 307 non-float constants skipped, 15 duplicate names de-duped
verified load: version 3, alignment 32, 19 tensors, 2 metadata keys; arch=silero-vad
```

- 単体テスト: `vokra-core` 53 + gguf 統合 7、`vokra-convert` 19 + roundtrip 1（合成 fixture、CI 常設。実 checkpoint は repo 非コミット）。
- metadata keys 15 = `vokra.model.*` 2 + `vokra.frontend.*` 13。Silero は 2（frontend chunk なし、`vokra-gguf-chunks.md` §4 の決定）。

## 7. M1 側スコープ（本 WP で実装していないもの）

FR-LD-03（frontend_spec の bit-exact **検査** = M1-03）/ FR-LD-04（safetensors runtime 直接ロード）/ FR-LD-07（K-quants ロード = M1-02）/ FR-TL-02/03（vokra-cli / vokra-eval への統合）。`vokra-convert` は M0 の最小版独立バイナリである。
