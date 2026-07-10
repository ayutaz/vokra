# M3-11 Godot Demo Runtime Verification — Handover

**Owner**: 依頼者 (Godot 4.3+ Editor での実 runtime dispatch verify は本質的に依頼者ボトルネック; CC cannot execute Godot Editor GUI operations).
**Predecessor**: M3-11 T01-T18 = 100% CC 完成 (Wave 3.5 + Wave 11 + Wave 13、`docs/tickets/m3/M3-11-godot-gdextension.md` §改訂記録)。
**Requirement under verification**: FR-API-05 (Godot GDExtension) + `docs/milestones.md` §7.3 Exit criteria 3 (Godot デモ動作)。

> **Explicit boundary**: 本 handover は M3-11-T19 (実 Godot 4.3+ editor での dispatch verify) + T20 (WP-close PR) を対象とする。CC 側で land 済のスコープは (a) ClassDB 登録 + method binding (T05-T13、`registry.rs` + `trampoline.rs`)、(b) crossbuild matrix (T12、5 target)、(c) demo scaffold (T14/T15、`asr_demo` + `tts_demo`)、(d) CI (T16/T17、`godot-crossbuild.yml` + release job)、(e) compliance scanner (T18、Unity mirror pattern + `libcudart*` glob gap 補修)。**runtime dispatch (Variant 実 unpack) は Wave 11 の trampoline layer で honest scope-out**: 各 trampoline は正しい signature + arity + panic firewall + `catch_unwind` を持つが、Variant unpack → real Rust dispatch (`crate::asr::transcribe` 等) は `TODO(M3-18)` として `InvalidMethod` を返す状態。**owner side で dispatch を実装するか、CC follow-up を明示的に依頼するか**が本 T19 の判断点。

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

Wave 11 の trampoline layer は **各 method の signature + arity + panic firewall + `catch_unwind` は正しく wire 済**だが、Variant unpack → real Rust dispatch は `TODO(M3-18)` として `InvalidMethod` を返す状態。以下の 3 option から本 T19 の scope を選択:

- **Option A: 現状で "CC 側完成" 判定を確定**する (最小 verify、`ClassDB.class_exists` + method 呼び出しが `InvalidMethod` で正しく返ることの確認のみ)。real dispatch は M3-11 の post-M3 follow-up として M4 に押し出し。
- **Option B: 依頼者側で Variant unpack を実装** (`trampoline.rs` L236 以降の `TODO(M3-18)` marker を実装)。C ABI 側の `include/vokra.h` は既に stable ゆえ、`p_args[0]` から `PackedFloat32Array` を取り出して `vokra_transcribe` に流すだけの thin wrapper。工数見積 = 2〜4 時間 (5 methods × 30〜60 min each)。
- **Option C: CC に follow-up 実装を明示的に依頼**する。本 handover の scope 外だが、Investigation を再実行して "CC-implementable" 判定に載せることは可能 (Variant layout は Godot 4.3-stable header + `docs/adr/0011-godot-gdextension.md` §D3 から reference 可能、ハルシネーション回避可)。

**推奨**: **Option A** で v0.9 Exit を通し、real dispatch を M4-XX の "M3-11 follow-up: full runtime dispatch" チケットに切る (M4 スコープ = 全 platform official support + C ABI 凍結、M3-11 の完全動作は M4/v1.0 GA の DoD)。

### 3.c. Demo scene の smoke (Option B/C を採用した場合のみ)

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

Dispatch mode: ☐ Option A (現状 InvalidMethod で pass、M4 follow-up) / ☐ Option B (owner 側で Variant unpack 実装) / ☐ Option C (CC follow-up 依頼)

Verification date: YYYY-MM-DD
```

## 6. Escalation

- **Extension load 失敗が続く場合**: `docs/adr/0011-godot-gdextension.md` の resolve chain (dlopen → `vokra_gdextension_init` → `p_get_proc_address` の 8 API resolve) が正しく通っていない可能性。Wave 11 の compile-time layout assert は Godot 4.3-stable header 前提 → **Godot 4.4+ で ABI 変更があれば `GDExtensionClassCreationInfo3` layout mismatch** で init 失敗する。この場合は M4 (v1.0 GA) で `GDExtensionClassCreationInfo4` 対応が必要。
- **Method dispatch を CC 側 follow-up として依頼**する場合は `docs/tickets/m3/M3-11-godot-gdextension.md` に T21 (or M4-XX) として新規 spec 起票を依頼。
- **T20 (WP-close PR)**: 上記 §5 の verification report を PR description に貼付、`docs/milestones.md` §7.3 Exit criteria 3 の 判定材料として反映。

## 7. 参考

- `docs/adr/0011-godot-gdextension.md` — ADR (gitignore、Wave 3.5 + Wave 11 + Wave 13 反映済)
- `docs/tickets/m3/M3-11-godot-gdextension.md` — ticket spec (§改訂記録 Wave 3.5 / Wave 11 / Wave 13 参照)
- `integrations/vokra-godot/README.md` — crate-level doc (Wave 13 状態、T01-T18 = 100%)
- `integrations/vokra-godot/src/trampoline.rs` L236-408 — `TODO(M3-18)` markers (Variant unpack scope-out points)
- `docs/adr/0007-unity-official-plugin.md` — sister binding (Unity UPM、Wave 11 の scanner を M3-11 で mirror)
- `.github/workflows/godot-crossbuild.yml` — CI (Wave 13、initial workflow_dispatch は owner)
