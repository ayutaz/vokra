# `vokra.*` GGUF chunk 命名仕様書

- **チケット**: M0-03-T08（WP 主成果物「`vokra.*` chunk 仕様」）
- **日付**: 2026-07-02（初版、M0-03-T08）／ 2026-07-06 追記（M0-06 `vokra.whisper.*`・M0-08 `vokra.campplus.*`・M2-06 `vokra.tokenizer.model`・M2-13 `vokra.provenance.*` を §6〜§9 に追加）／ 2026-07-06 追記（§6 `vokra.whisper.decoder_start_ids` の anchoring 機構を現物コメントから明示化。Phase 5 follow-on と decoder-step Phase 1〜3b は GGUF chunk 変更なし＝カーネル / 型抽象のみの変更のため他 § は現状維持）
- **正**: `vokra.model.*` / `vokra.frontend.*` / `vokra.provenance.*` のキー定数は `crates/vokra-core/src/gguf/chunks.rs`、frontend の typed read/write は `crates/vokra-core/src/gguf/frontend_spec.rs`。モデル固有 chunk のキー定数は各コンバータ／モデル実装に定義（`vokra.whisper.*` = `crates/vokra-convert/src/models/whisper.rs`、`vokra.campplus.*` = `crates/vokra-convert/src/models/campplus.rs`、`vokra.tokenizer.model` = 同 whisper.rs ＋ `crates/vokra-models/src/whisper/tokenizer.rs`）。本書とコードが乖離した場合はコードを修正して本書に合わせる（キー名は保存済みモデルとの互換契約のため、コード側の勝手な改名は破壊的変更）。
- **出典区分**: 「転記」= CLAUDE.md / SRS からの転記（変更不可）、「(提案)」= 本チケットで設計した提案値（M1 以降で変更余地あり）。

## 1. prefix 規約（転記 + 仕様書根拠）

音声固有 metadata は **`vokra.` prefix の独自 GGUF chunk** とし、llama.cpp 本体との命名衝突を回避する（FR-LD-02、IF-07、CLAUDE.md 設計判断 3「音声固有 metadata は `vokra.*` prefix の独自 chunk として追加」）。

GGUF 仕様書（ggml-org/ggml `docs/gguf.md` @ `eced84c86f8b`）はコミュニティ独自キーについて「should be namespaced with the relevant community name to avoid collisions」と定めており、`general.*` / `<arch>.*` / `tokenizer.*` の標準 namespace と `vokra.*` は構造的に排他。

サブ namespace（初版 M0-03-T08 は上 2 つ「model / frontend」を定義。以降の WP で下 4 つを追加、§6〜§9 に詳細）:

- `vokra.model.*` — モデル識別 (提案、§3)
- `vokra.frontend.*` — frontend_spec（前処理パラメータ、FR-LD-03 の検査対象、§2）
- `vokra.whisper.*` — Whisper ハイパーパラメータ（M0-06、§6）
- `vokra.campplus.*` — CAM++ 話者エンコーダ構成（M0-08、§7）
- `vokra.tokenizer.model` — Whisper detokenizer blob（U8 array、M2-06、§8）
- `vokra.provenance.*` — weight ライセンス / provenance（M2-13、FR-CP-05/03、§9）

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

最小セットのみ定義。FR-CP-05（provenance / weight ライセンス metadata）は別 namespace `vokra.provenance.*` として **M2-13 で実装済み**（§9）。`vokra.provenance.*` を持たない GGUF について、runtime はこの `vokra.model.*` を built-in ライセンスレジストリの検索キーに使う（fail-closed、§9）。

## 4. Silero VAD への適用方針（T08 決定）

**Silero VAD の GGUF には `vokra.frontend.*` chunk を書かない**（`vokra.model.*` のみ）。

根拠: Silero の内部疑似 STFT は 1:1 保存 subgraph（FR-LD-06、M0-05）に隠蔽される実装詳細であり、Vokra が制御する前処理（frontend）ではない。変換ツールが出典なく frontend 値を発明することは禁止事項に当たるため、書かないことが正しい（`crates/vokra-convert/src/models/silero.rs` rustdoc に同旨）。

## 5. M0 / M1 スコープ境界（転記）

- **M0（本 WP）**: chunk の**書き込み / 読み出し**まで（FR-LD-02）。
- **M1-03**: frontend_spec の**検査** — runtime 側が spec を検査し bit-exact match しなければ warn / fail（FR-LD-03、v0.1 MVP）。M0 実装には検査ロジックを意図的に含めていない（M1 スコープ混入防止、チケット T09 完了条件）。
- 標準キーのうち Vokra が書くのは `general.alignment`（UINT32、未指定時 32）のみ。`general.*` の他キーへの書込は行わない（llama.cpp 側 namespace の尊重）。`vokra.tokenizer.model`（§8）は標準 `tokenizer.*` namespace ではなく `vokra.*` 側に置くため、この不変条件を破らない。

---

## 追記（2026-07-06）: 後続 WP で追加した chunk

以下は初版（M0-03-T08）以降の WP で実装した chunk。各 § はコード実体（現物）からの**転記**で、キー・型・値の出典を明記する（発明値なし）。

## 6. `vokra.whisper.*` — Whisper ハイパーパラメータ（M0-06、実装済み）

native Whisper 実装（`vokra-models`）が全ハイパーパラメータを**ハードコードせず GGUF metadata から読む**ための chunk（FR-LD-02 / FR-MD-02）。値はすべて **checkpoint のテンソル shape から導出**し発明しない（欠損テンソルは `0` を書き、runtime 側 `WhisperConfig` が load 時に reject）。コード実体は `crates/vokra-convert/src/models/whisper.rs`（`write_hparams`）。同一キーはモデル側 `crates/vokra-models/src/whisper/config.rs` に複製（両 crate は相互依存不可のため。`chunks.rs` への集約は follow-up）。

| キー | GGUF value type | 意味 / 出典 |
|------|-----------------|-------------|
| `vokra.whisper.n_mels` | UINT32 | **mel 入力チャンネル数。`model.encoder.conv1.weight` の shape[1] から config 駆動で導出（base/small/medium = 80、large-v3 = 128）。ハードコードしない**。`vokra.frontend.n_mels` と不一致の GGUF は runtime が reject |
| `vokra.whisper.n_audio_ctx` | UINT32 | encoder 位置長（1500）= `encoder.embed_positions.weight` shape[0] |
| `vokra.whisper.n_audio_state` | UINT32 | encoder/decoder hidden 幅 `d_model` = conv1 shape[0] |
| `vokra.whisper.n_audio_head` | UINT32 | encoder attention head 数（Whisper 不変則 head_dim=64 より `d_model / 64`） |
| `vokra.whisper.n_audio_layer` | UINT32 | encoder block 数（連番テンソルの計数） |
| `vokra.whisper.n_text_ctx` | UINT32 | decoder 位置長（448） |
| `vokra.whisper.n_text_state` | UINT32 | decoder hidden 幅 |
| `vokra.whisper.n_text_head` | UINT32 | decoder attention head 数 |
| `vokra.whisper.n_text_layer` | UINT32 | decoder block 数 |
| `vokra.whisper.n_vocab` | UINT32 | token vocab サイズ = `decoder.embed_tokens.weight` shape[0]（base 51865 / large-v3 51866） |
| `vokra.whisper.ffn_dim` | UINT32 | feed-forward inner 幅 |
| `vokra.whisper.eot` | UINT32 | end-of-transcript token id（多言語 tokenizer = 50257 固定、special token の floor） |
| `vokra.whisper.decoder_start_ids` | `ARRAY<U32>` | 既定英語転写 prefix（`startoftranscript` / `en` / `transcribe` / `notimestamps` の 4 special id）。ハードコード表を持たず **`eot` と `n_vocab` の双方に anchor** して large-v3 の +1 special-token shift を吸収する: `sot = eot + 1` と `<\|en\|> = eot + 2` は多言語 tokenizer で size 非依存（`eot = 50257` 固定）、`<\|transcribe\|>` と `<\|notimestamps\|>` は vocab 末尾から固定距離（`n_vocab − 1506` / `n_vocab − 1502`）で anchor される。base（n_vocab 51865）= `[50258, 50259, 50359, 50363]`、large-v3（n_vocab 51866）= `[50258, 50259, 50360, 50364]`（`transformers` `WhisperProcessor.get_decoder_prompt_ids` と一致検証済み） |

## 7. `vokra.campplus.*` — CAM++ 話者エンコーダ構成（M0-08、実装済み）

CAM++（3D-Speaker）話者エンコーダの native forward が構成を復元するための chunk。`vokra.model.arch` = `vokra.model.name` = `"campplus"`。コード実体は `crates/vokra-convert/src/models/campplus.rs`。導出値（block_config / growth / embed_dim）はグラフの initializer から、定数値（dilations / cam_seg_len / bn_eps / feat_dim）は reference `campplus.onnx` の全グラフ走査で検証済み。**CAM++ の GGUF は `vokra.frontend.*` を持たない**（入力は Kaldi fbank で、その次元は下記 `feat_dim` に集約。STFT frontend_spec は書かない）。

| キー | GGUF value type | 意味 / 出典 |
|------|-----------------|-------------|
| `vokra.campplus.block_config` | `ARRAY<U32>` | block ごとの D-TDNN dense 層数（initializer 名 `xvector.block<N>.tdnnd<M>` から導出、fallback `[12, 24, 16]`） |
| `vokra.campplus.growth` | UINT32 | dense 層ごとの channel growth（`cam_layer.linear_local` 出力ch、fallback 32） |
| `vokra.campplus.dilations` | `ARRAY<U32>` | block ごとの `cam_layer.linear_local` dilation（検証値 `[1, 2, 2]`） |
| `vokra.campplus.cam_seg_len` | UINT32 | CAM `seg_pool` `AvgPool1d` kernel = stride（検証値 100） |
| `vokra.campplus.bn_eps` | FLOAT32 | BatchNorm epsilon（load 時 fold 用、検証値 1e-5） |
| `vokra.campplus.feat_dim` | UINT32 | 入力 fbank 特徴次元（80） |
| `vokra.campplus.embed_dim` | UINT32 | 出力話者埋め込み次元（`xvector.dense.linear` 出力ch、fallback 192） |

## 8. `vokra.tokenizer.model` — Whisper detokenizer blob（M2-06、実装済み）

runtime が token id 列を自動でテキスト化する（bracketed id 列を出さない）ための、Whisper detokenizer 語彙を焼き込む単一キー。コード実体は書き込みが `crates/vokra-convert/src/models/whisper.rs`（`embed_tokenizer`）、読み出しが `crates/vokra-models/src/whisper/tokenizer.rs`（`WhisperTokenizer::from_gguf`）。

- **型は STRING ではなく `U8` array**: byte-level BPE のトークンは単独では valid UTF-8 でない（例: 単独の `0xC3`）ため GGUF `STRING` に格納できない。
- **バイナリ形式**: little-endian の `u32 count` ヘッダに続き、token id 順に `count` 個の `{ u8 special; u16 byte_len; [u8; byte_len] }` レコード。
- **書き込み条件**: 実在の多言語 Whisper 語彙（n_vocab ≥ 50257）のみ埋め込む。text レコード（id `0..50257`）は同梱リソース `whisper_multilingual_text_vocab.bin` を `include_bytes!` で読む（生データのため zero-dependency 不変条件 NFR-DS-02 は不変）。special tail（id ≥ 50257）は空 special レコード `{1, 0, 0}` を `n_vocab − 50257` 個（base 1608、large-v3 1609）。
- **復号**: special token を飛ばしバイト連結後に 1 回だけ lossy UTF-8 decode（不正列は U+FFFD、panic しない）。

## 9. `vokra.provenance.*` — weight ライセンス / provenance（M2-13、FR-CP-05/03、実装済み）

compliance gate（`crate::compliance`）が **weight ライセンスを分類し CC-BY-NC を research flag なしで拒否**するための chunk。記録するのは **weight のライセンス**で、crate/ソースコードのライセンスとは独立（例: F5-TTS / EnCodec は code=MIT だが weight=CC-BY-NC、`docs/license-audit.md` §3）。コード実体はキー定数が `crates/vokra-core/src/gguf/chunks.rs`、書き込みが `crate::compliance::stamp_provenance`（M2-13 では最小）。

| キー | GGUF value type | 意味 / 出典 |
|------|-----------------|-------------|
| `vokra.provenance.weight_license` | STRING | 解決済み `LicenseClass` 正準名（例 `"non-commercial"`）。明示的・**最優先**の override |
| `vokra.provenance.license` | STRING | 生の weight ライセンス文字列（例 `"CC-BY-NC-4.0"` / `"MIT"`） |
| `vokra.provenance.model_id` | STRING | built-in ライセンスレジストリ検索用のモデル識別子（例 `"f5-tts"` / `"encodec"`） |
| `vokra.provenance.source` | STRING | 上流ソース注記（URL / repo）。分類には未使用（advisory のみ） |

**解決優先順位**（FR-CP-03）: `weight_license`（パース可能なら最優先）→ `license`（生文字列）→ built-in registry（`vokra.model.*` をキーに検索）→ `LicenseClass::Unknown`。最後の Unknown は **fail-closed**（gate 必須、silent pass しない）。
