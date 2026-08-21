# M3-11 Godot Demo Runtime Verification — Handover

**Owner**: 依頼者 (Godot 4.3+ Editor での実 runtime dispatch verify は本質的に依頼者ボトルネック; CC cannot execute Godot Editor GUI operations).
**Predecessor**: M3-11 T01-T18 = 100% CC 完成 (Wave 3.5 + Wave 11 + Wave 13、`docs/tickets/m3/M3-11-godot-gdextension.md` §改訂記録)。
**Requirement under verification**: FR-API-05 (Godot GDExtension) + `docs/milestones.md` §7.3 Exit criteria 3 (Godot デモ動作)。

> **Current boundary (2026-08-22 correction)**: load/transcribe/TTS/stream に加え、`session_vad_open_stream` の実 `VokraStream::open`、Godot Object wrapping、ClassDB Object return が実装済み。公式 Godot 4.7.1 headless で実 Silero GGUF の push/poll/interrupt/reset も検証済みである。残る owner 項目は既存 ASR/TTS demo scene の対話的 editor/各 export target 確認であり、VAD runtime 実装穴ではない。

## 1. Prerequisites checklist

- [ ] **Godot 4.3-stable** (GDExtension は Godot 4 以降。Wave 11 で `GDExtensionClassCreationInfo3` (160 bytes) を Godot 4.3-stable header に対して compile-time layout assert 済み、`clang -m64` verify)。
- [ ] **`vokra_godot.dylib` / `libvokra_godot.so` / `vokra_godot.dll`** — 下記 §2 手順で生成。
- [ ] **`whisper-base.gguf`** — MIT weight、M2-06 検証済み。ASR demo で使用。
- [ ] **piper-plus voice GGUF** — 依頼者作 MIT、TTS demo で使用。M3-12-T14 の実 voice sanity 済 GGUF と共有可能。
- [ ] **`jfk-30s.wav`** (or 任意 16 kHz mono WAV) — ASR demo で使用。
- [ ] **各 platform の実機 or Editor環境** — macOS/Windows/Linux は Editor 上で直接検証可能、Android は Editor から export template + `adb install` (M3-18 と併走)。
- [ ] **禁則**: `godot-cpp` / `gdext-rs` / bindgen は使用禁止。生 FFI 実装 (`docs/adr/0011-godot-gdextension.md` §D1/D3)。

## 2. Build recipe (host-only + crossbuild matrix)

### 2.a. Host-only iteration (開発中)

```bash
cd integrations/vokra-godot
cargo build --release            # host cdylib
cargo test                       # 52 unit tests (Wave 13 baseline)
```

または FR-TL-04 helper:

```bash
bash scripts/build-godot-gdextension.sh              # host-only cdylib sync
bash scripts/build-godot-gdextension.sh --pack       # + assemble AssetLib zip
```

zip は `dist/godot/vokra-godot-<version>.zip` に生成。**dev iteration ONLY**、consumer 配布は §2.b の CD job から。

### 2.b. Crossbuild matrix (5 target、T12 Wave 13)

```bash
TARGET_TRIPLE=x86_64-apple-darwin       bash scripts/build-godot-gdextension.sh
TARGET_TRIPLE=aarch64-apple-darwin      bash scripts/build-godot-gdextension.sh
TARGET_TRIPLE=x86_64-unknown-linux-gnu  bash scripts/build-godot-gdextension.sh
TARGET_TRIPLE=x86_64-pc-windows-msvc    bash scripts/build-godot-gdextension.sh
TARGET_TRIPLE=aarch64-linux-android     bash scripts/build-godot-gdextension.sh
```

unknown triple は `exit 1` (FR-EX-08 no silent fallback)。

CI 経由 (`godot-crossbuild.yml`、workflow_dispatch + weekly cron):
```
gh workflow run godot-crossbuild.yml
```
初回 workflow_dispatch は owner (`docs/tickets/m3/M3-11-godot-gdextension.md` §改訂記録 Wave 13)。

### 2.c. Release zip (tagged、T17)

`release.yml` の `godot-package-release` job が tag SHA から artifact reassemble + deterministic zip pack + GitHub Release upload。**署名済 canonical zip は CD job 経由のみ** (NFR-MT-08、手動 build 配布禁止)。

## 3. Editor での動作確認 (T19 owner runbook)

### 3.a. Editor 起動 + Extension load

1. Godot 4.3-stable を起動、`Project → New Project` で新規 project 作成 (Compatibility renderer で可、Vulkan renderer でも可)。
2. `integrations/vokra-godot/demos/asr_demo/` を丸ごと新規 project にコピー、または `--pack` で生成された AssetLib zip を **`Project → Install Asset...`** から import。
3. `addons/vokra/vokra.gdextension` が Editor で load されることを確認。**Output ドックに `Vokra GDExtension loaded (version …)` 相当のログ**が出れば OK (registry.rs の init callback が実行されたことを示す)。
4. `ClassDB.class_exists("VokraSession")` を GDScript から呼んで `true` が返ることを Editor コンソール (`_@` prefix) で確認。
5. `ClassDB.class_exists("VokraStream")` も同様。

**FAIL 条件**:
- Extension load 失敗 (Output に error) → per-platform binary が demo project 内 `addons/vokra/bin/<platform>/<arch>/` に配置されていない可能性。§2.b の crossbuild で該当 target の artifact を配置。
- `ClassDB.class_exists("VokraSession")` == `false` → registry.rs の init 経路失敗。libvokra.so の symbol export を `nm libvokra_godot.so | grep vokra_gdextension_init` で確認。

### 3.b. Method dispatch 検証 (T19 honest scope note)

> **決着済み 2026-08-22 — 旧3 optionを選ぶ必要はありません**: Option C
> （CC follow-up 実装）が実施済みです。`ba33bd0` が 5 trampoline の real
> dispatch を land し、`71ea5ef` が `load` trampoline と inner-session
> binding（registry が常に None を返していた欠陥）を実装、あわせて headless
> 検証 leg も追加しました。旧`trampoline.rs` L236-408 の`TODO(M3-18)` markerは
> **残存0件**です。後から追加された`session_vad_open_stream`も Object return と
> 実 Silero headless smoke まで完了しました。旧Option A/B/Cは実装前の判断記録で
> あり、現在の作業選択肢ではありません。現在の owner 検証対象は実装穴ではなく、
> ASR/TTS demo の editor 操作と各 export target です。

### 3.c. Demo scene の smoke

`demos/asr_demo/main.gd` L28 以降の flow:
1. `res://models/whisper-base.gguf` を配置 (owner が手動、または `bash addons/vokra/fetch-demo-models.sh` を owner-side で作成)。
2. `res://audio/jfk.wav` を配置 (`tests/fixtures/audio/jfk-30s.wav` を copy)。
3. Editor で **Play (F5)** → `LoadButton` を押下 → `TranscriptLabel` に転写結果が表示されることを確認。
4. Backend selection: default = CPU。`--features metal` build であれば `session.load_model(MODEL_PATH, "metal")` で Metal 経由 (M1 iMac 実機で bit-identical vs CPU atol < 5e-4 が Wave 9 で verify 済)。

## 4. Export template (multi-platform、T19 拡張)

macOS/Linux/Windows は Editor 上で直接 Play 可能。Android は export template 経由:

1. `Project → Export...` → Android platform を追加。
2. Custom template = 使用しない (公式 template で可)。
3. Gradle build または pre-built APK 生成。
4. `adb install app-release.apk` → 実機で開くと Editor と同じ smoke が走る (M3-18 と併走)。

## 5. 結果報告テンプレート

```
### Godot GDExtension demo verification (M3-11-T19)

Godot version: 4.3-stable
Platforms tested:
- [ ] macOS Apple Silicon: ☐ Extension load / ☐ ClassDB / ☐ Method dispatch / ☐ Full demo
- [ ] macOS Intel:         ☐ Extension load / ☐ ClassDB / ☐ Method dispatch / ☐ Full demo
- [ ] Linux x86_64:        ☐ Extension load / ☐ ClassDB / ☐ Method dispatch / ☐ Full demo
- [ ] Windows x86_64:      ☐ Extension load / ☐ ClassDB / ☐ Method dispatch / ☐ Full demo
- [ ] Android arm64-v8a:   ☐ Extension load / ☐ ClassDB / ☐ Method dispatch / ☐ Full demo (M3-18 併走)

Implemented dispatch: ☐ transcribe / ☐ synthesize / ☐ stream_push_pcm / ☐ stream_poll
VAD stream object: ☐ implemented + verified / ☐ explicitly pending (must not count as full dispatch)

Verification date: YYYY-MM-DD
```

## 6. Escalation

- **Extension load 失敗が続く場合**: `docs/adr/0011-godot-gdextension.md` の resolve chain (dlopen → `vokra_gdextension_init` → `p_get_proc_address` の 8 API resolve) が正しく通っていない可能性。Wave 11 の compile-time layout assert は Godot 4.3-stable header 前提 → **Godot 4.4+ で ABI 変更があれば `GDExtensionClassCreationInfo3` layout mismatch** で init 失敗する。この場合は M4 (v1.0-rc、2026-07-14 再割当 #2) で `GDExtensionClassCreationInfo4` 対応が必要。
- **`session_vad_open_stream`**: 2026-08-22 に `VokraStream::open`、
  `StreamInstance` lifetime、Godot Object pack、公式 Godot 4.7.1 headless smoke
  まで完了。実 editor は上記 demo release check で確認する。
- **T20 (WP-close PR)**: 上記 §5 の verification report を PR description に貼付、`docs/milestones.md` §7.3 Exit criteria 3 の 判定材料として反映。

## 7. 参考

- `docs/adr/0011-godot-gdextension.md` — ADR (gitignore、Wave 3.5 + Wave 11 + Wave 13 反映済)
- `docs/tickets/m3/M3-11-godot-gdextension.md` — ticket spec (§改訂記録 Wave 3.5 / Wave 11 / Wave 13 参照)
- `integrations/vokra-godot/README.md` — crate-level doc (Wave 13 状態、T01-T18 = 100%)
- `integrations/vokra-godot/src/trampoline.rs` — 実装済み dispatch と
  `session_vad_open_stream` Object return
- `docs/adr/0007-unity-official-plugin.md` — sister binding (Unity UPM、Wave 11 の scanner を M3-11 で mirror)
- `.github/workflows/godot-crossbuild.yml` — CI (Wave 13、initial workflow_dispatch は owner)
