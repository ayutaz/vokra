# Web (WASM / WebGPU) チュートリアル

[English](web.md) | **日本語**

**Vokra Web ランタイム**（`web/pkg`、npm パッケージ `@vokra/web`）の使い方:
ブラウザ上の Whisper base ASR を

- **WASM CPU パス**（SIMD128 / scalar）と
- **WebGPU バックエンド**（`navigator.gpu` を生 wasm import shim + 手書き JS
  glue で駆動 — `wgpu` crate / `wasm-bindgen` 不使用、ランタイム npm 依存ゼロ）

の 2 経路で動かします。

## 1. インストール

```sh
npm install @vokra/web
```

> npm scope の登録はメンテナ作業（M4-01-T27）です。初回 registry publish
> までは repo からビルドしてください: `scripts/build-wasm.sh pkg` → `web/pkg/`。

## 2. モデルの用意

モデルは**同梱されません**。Whisper base checkpoint をオフラインで変換します:

```sh
cargo build --release -p vokra-cli
./target/release/vokra-cli convert \
  --model whisper \
  --input /path/to/whisper-base/model.safetensors \
  --output whisper-base.gguf
```

`whisper-base.gguf` を静的アセットとして配信してください。Web ランタイムは
これを**全量メモリロード**します — **WASM に mmap はありません**
（`vokra-mmap` は native 専用。ブラウザ経路は fetch → `ArrayBuffer` →
in-memory GGUF parser です）。

## 3. 文字起こし

```js
import { createSession } from "@vokra/web";

const model = await (await fetch("/models/whisper-base.gguf")).arrayBuffer();

// backend は明示選択 — Vokra は silent fallback しません:
const session = await createSession(model, { backend: "cpu" }); // または "webgpu"

const wav = await (await fetch("/audio/jfk-30s.wav")).arrayBuffer(); // 16 kHz mono PCM16
const { text, rtf } = await session.transcribe(wav);
console.log(text, `RTF ${rtf.toFixed(3)}`);

await session.close();
```

`transcribe` は 16 kHz mono PCM の `Float32Array` も受け付けます。

## 4. バックエンド選択は明示（FR-EX-08）

| 指定 | 挙動 |
|------|------|
| `{ backend: "cpu" }`（既定） | WASM CPU パス。特別なヘッダ不要でどこでも動作。 |
| `{ backend: "webgpu" }` | cross-origin isolation **と** WebGPU adapter の両方が必要。欠けている場合は**理由を明記したエラーで reject** — 裏で CPU に落ちることはありません。CPU で動かすのは呼び出し側の明示的な `"cpu"` 選択です。 |

エラーメッセージには対処法（下記 §5 の COOP/COEP 配備、または WebGPU 対応ブラウザ）が明記されます。

## 5. COOP/COEP 配備（`webgpu` に必須）

WebGPU の readback（`mapAsync`）は async のみ、Vokra の推論ループは同期です。
ランタイムは推論を専用 Web Worker で回し、`SharedArrayBuffer` 上の
`Atomics.wait` で main-thread GPU proxy と同期橋渡しします。ブラウザが
`SharedArrayBuffer` を有効にするのは **cross-origin isolated** なページのみ
なので、サーバは次の 2 ヘッダを返す必要があります:

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

ローカル確認用に依存ゼロのサーバを同梱しています:

```sh
scripts/build-wasm.sh pkg
node web/demo/serve.mjs          # http://localhost:8788/web/demo/
```

**CPU パスはどちらのヘッダも不要**です。

## 6. SIMD128: 2-artifact + 自動選択

WASM には **runtime CPU feature detection がありません** — SIMD の可否は
module validation 時に確定します。そのためパッケージは 2 つの artifact
（`vokra_wasm_simd128.wasm` / `vokra_wasm_base.wasm`）を同梱し、loader が
`WebAssembly.validate` probe で選択します（`session.meta.artifact` で確認
可能）。**Relaxed SIMD は不採用**（Safari 部分対応のみ = 四半期監視、
relaxed fma の非決定性は parity 規律と非整合）— kernel は決定的な
mul + add のみを使います。

## 7. Memory64 の現況（調査記録のみ）

Whisper base（~74M params）は wasm32 線形メモリに収まるため wasm32 を
対象とします。本 WP 調査時点: Rust `wasm64-unknown-unknown` は Tier 3、
ブラウザ Memory64 は Chromium/Firefox 系で shipped・Safari 未対応
（メンテナの実機 spot check で追認 — 対応可否は記録であり発明しません）。
大型モデルの Web 対応は follow-up です。

## 8. 性能ノート（honest な現状）

WebGPU バックエンドは現状 **per-op**（kernel ごとに upload → dispatch →
readback）で動きます。whisper-base 規模ではこれは WASM CPU パスより
*遅い*見込みです — Metal backend が device 常駐化前に通ったのと同じ段階
です。デモの RTF 表示 / `tools/wasm/parity.html` で実測し、記録は
`docs/bench-baselines/web-2026-07-15/README.md`（Kill switch G の対照
数値でもあります）へ。device 常駐化は follow-up です。

## 9. トラブルシューティング

| 症状 | 原因 / 対処 |
|------|------------|
| `backend webgpu needs SharedArrayBuffer…` | COOP/COEP ヘッダ未配備 — §5。 |
| `no WebGPU adapter…` | WebGPU の無いブラウザ/コンテキスト — 明示的に `"cpu"` を選ぶか WebGPU 対応ブラウザで。 |
| `WAV must be 16 kHz mono PCM16` | オフラインで変換（`ffmpeg -i in.wav -ar 16000 -ac 1 -c:a pcm_s16le out.wav`）するか `Float32Array` を渡す。 |
| Node で `backend: "webgpu"` が失敗 | 期待どおり — Node に `navigator.gpu` は無く、明示的な unavailability contract です。 |
