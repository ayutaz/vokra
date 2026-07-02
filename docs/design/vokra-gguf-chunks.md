# `vokra.*` GGUF chunk 命名仕様書

- **チケット**: M0-03-T08（WP 主成果物「`vokra.*` chunk 仕様」）
- **日付**: 2026-07-02
- **正**: 本仕様のコード実体は `crates/vokra-core/src/gguf/chunks.rs`（キー定数）と `crates/vokra-core/src/gguf/frontend_spec.rs`（typed read/write）。本書とコードが乖離した場合はコードを修正して本書に合わせる（キー名は保存済みモデルとの互換契約のため、コード側の勝手な改名は破壊的変更）。
- **出典区分**: 「転記」= CLAUDE.md / SRS からの転記（変更不可）、「(提案)」= 本チケットで設計した提案値（M1 以降で変更余地あり）。

## 1. prefix 規約（転記 + 仕様書根拠）

音声固有 metadata は **`vokra.` prefix の独自 GGUF chunk** とし、llama.cpp 本体との命名衝突を回避する（FR-LD-02、IF-07、CLAUDE.md 設計判断 3「音声固有 metadata は `vokra.*` prefix の独自 chunk として追加」）。

GGUF 仕様書（ggml-org/ggml `docs/gguf.md` @ `eced84c86f8b`）はコミュニティ独自キーについて「should be namespaced with the relevant community name to avoid collisions」と定めており、`general.*` / `<arch>.*` / `tokenizer.*` の標準 namespace と `vokra.*` は構造的に排他。

M0 で定義するサブ namespace は 2 つ:

- `vokra.model.*` — モデル識別 (提案)
- `vokra.frontend.*` — frontend_spec（前処理パラメータ、FR-LD-03 の検査対象）

## 2. `vokra.frontend.*` — 13 キーと型対応

キー名の 13 フィールドは CLAUDE.md / FR-LD-03 からの**転記**: `{n_fft, hop, win_length, window_type, mel_norm, htk_mode, fmin, fmax, n_mels, pad_mode, dc_offset_removal, pre_emphasis, sample_rate}`。**GGUF value type の対応は本チケットの (提案)**:

| キー | GGUF value type | 意味 |
|------|-----------------|------|
| `vokra.frontend.n_fft` | UINT32 | FFT 窓サイズ |
| `vokra.frontend.hop` | UINT32 | フレーム間 hop 長 |
| `vokra.frontend.win_length` | UINT32 | 分析窓長（≤ n_fft、不足分は zero-pad） |
| `vokra.frontend.window_type` | STRING | 窓関数名（例 `"hann"`） |
| `vokra.frontend.mel_norm` | STRING | mel 正規化（例 `"slaney"`） |
| `vokra.frontend.htk_mode` | BOOL | HTK mel スケールか（false = Slaney） |
| `vokra.frontend.fmin` | FLOAT32 | mel 最低端周波数 [Hz] |
| `vokra.frontend.fmax` | FLOAT32 | mel 最高端周波数 [Hz] |
| `vokra.frontend.n_mels` | UINT32 | mel バンド数 |
| `vokra.frontend.pad_mode` | STRING | 信号 padding（例 `"reflect"`） |
| `vokra.frontend.dc_offset_removal` | BOOL | DC オフセット除去の有無 |
| `vokra.frontend.pre_emphasis` | FLOAT32 | pre-emphasis 係数（0.0 = off） |
| `vokra.frontend.sample_rate` | UINT32 | 入力サンプルレート [Hz] |

- 値は**上流実装からの転記のみ**を許す（値の発明・記憶からのハードコード禁止 — レビュアー C 指摘 #2 の bit-exact 保証。Whisper base の転記元は `docs/design/m0-03-gguf-loader.md` §2）。
- typed read/write は `FrontendSpec::write_into`（+ `to_gguf_kv`）/ `FrontendSpec::from_gguf`（round-trip 単体テストあり = T09）。

## 3. `vokra.model.*` — モデル識別キー (提案)

| キー | GGUF value type | 例 |
|------|-----------------|-----|
| `vokra.model.arch` | STRING | `"whisper"` / `"silero-vad"` |
| `vokra.model.name` | STRING | `"whisper-base"` / `"silero-vad-v5"` |

最小セットのみ定義。M1 の FR-CP-05（provenance / ライセンス metadata）は同 namespace の別サブキー（例 (提案): `vokra.provenance.*`）に載せる余地を確保する。

## 4. Silero VAD への適用方針（T08 決定）

**Silero VAD の GGUF には `vokra.frontend.*` chunk を書かない**（`vokra.model.*` のみ）。

根拠: Silero の内部疑似 STFT は 1:1 保存 subgraph（FR-LD-06、M0-05）に隠蔽される実装詳細であり、Vokra が制御する前処理（frontend）ではない。変換ツールが出典なく frontend 値を発明することは禁止事項に当たるため、書かないことが正しい（`crates/vokra-convert/src/models/silero.rs` rustdoc に同旨）。

## 5. M0 / M1 スコープ境界（転記）

- **M0（本 WP）**: chunk の**書き込み / 読み出し**まで（FR-LD-02）。
- **M1-03**: frontend_spec の**検査** — runtime 側が spec を検査し bit-exact match しなければ warn / fail（FR-LD-03、v0.1 MVP）。M0 実装には検査ロジックを意図的に含めていない（M1 スコープ混入防止、チケット T09 完了条件）。
- 標準キーのうち Vokra が書くのは `general.alignment`（UINT32、未指定時 32）のみ。`general.*` の他キーへの書込は行わない（llama.cpp 側 namespace の尊重）。
