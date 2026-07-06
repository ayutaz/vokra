# M2 (v0.5) Owner Verification Checklist

**Owner**: 依頼者 (`ayutaz`) — 実機テスト・法務判断・鍵/秘密情報の provision を担当。
**CC-side status**: v0.5 15 WP のうち **11 完了 + 3 CC 部完了 + 1 継続監視**（M2-15）。以下のチェックポイントを依頼者が消化することで v0.5 milestone Exit 判定に進める。

各項目は「必要な準備 → 実行手順 → Exit 判定への寄与」の 3 段で記述。CC が既に整備した scaffold（scripts / CI / docs）へのポインタを併記する。

---

## 1. iOS 実機 RTF 計測（M2-14 / NFR-PF-03）

**依頼者タスク**: Xcode + 実機 iPhone/iPad で Whisper base RTF < 0.5 を計測する。

### 必要な準備

- [ ] Xcode 14+ が macOS 上で動作する状態。
- [ ] Apple Developer 署名プロファイル（開発用実機配布可）。
- [ ] iPhone / iPad（iOS 15+）。Simulator RTF は NFR-PF-03 の判定に使えない。
- [ ] `Vokra.xcframework` — tagged release（v* tag push で `release.yml.ios-xcframework` が生成、GH Release にアップロード）または `scripts/build-ios.sh` をローカル実行で生成。
- [ ] `whisper-base.gguf` — `vokra-convert convert --model whisper --input <safetensors> --output whisper-base.gguf` で作成。

### 実行手順

`docs/m2-14-ios-rtf-handover.md` の SwiftUI 最小計測アプリと計測手順に従う。

1. Xcode iOS App 新規プロジェクト作成 → Package Dependencies に `Package.swift` を追加。
2. `whisper-base.gguf` + `tests/fixtures/audio/jfk-30s.wav` を app target に bundle。
3. Signing & Capabilities → Team 設定 → 実機ビルド。
4. 手順書のコードで RTF を 3 回計測し median を記録。

### Exit 判定への寄与

- NFR-PF-03（実機 RTF < 0.5）— v0.5 Exit criteria 1。
- 未達の場合は「未達値をそのまま記録・公開」（新規閾値を発明しない）、M2-15 四半期 Go/No-go review へ入力。

---

## 2. CUDA large-v3 RTF 実測（M2-03 follow-up / NFR-PF-04）

**依頼者タスク**: vast.ai RTX 4090 を起動し `cargo test whisper_cuda_large_v3_rtf` を実行、mean RTF を記録・公開する。

### 必要な準備

- [ ] vast.ai account + API key（既存の運用手順で持っている想定）。
- [ ] RTX 4090（or Ampere/Ada with ≥16GB VRAM）が乗ったオファリング。
- [ ] `whisper-large-v3.safetensors`（Hugging Face `openai/whisper-large-v3` から DL）を `vokra-convert` で GGUF 化。

### 実行手順

```bash
# 1. vast.ai instance 起動（既存の運用手順）
vastai search offers 'gpu_name=RTX_4090 num_gpus=1'
vastai create instance <offer_id> --image nvidia/cuda:12.6.2-devel-ubuntu22.04/ssh ...

# 2. instance に ssh、リポジトリ clone、rustup + cuda toolkit 準備
ssh root@<vast_host>
git clone https://github.com/ayutaz/vokra.git && cd vokra
# rustup install ...

# 3. large-v3 GGUF 生成（別途 checkpoint DL）
cargo build --release -p vokra-convert
./target/release/vokra-convert convert --model whisper --input lv3.safetensors --output large-v3.gguf

# 4. RTF 測定（本 follow-up で追加した sanity test）
VOKRA_WHISPER_LARGE_V3_GGUF=$PWD/large-v3.gguf \
  cargo test --release -p vokra-backend-cuda --features cuda \
  --test whisper_cuda_large_v3_rtf -- --nocapture

# 5. 記録
# 出力例: rtf=0.087 mean_ms=2620.5 p50=2611.3 p95=2645.1 ...
# を docs/bench-baselines/ に json で保存

# 6. instance 即 destroy（コスト最小化、CLAUDE.md の運用パターン）
vastai destroy instance <instance_id>
```

### Exit 判定への寄与

- NFR-PF-04（CUDA large-v3 RTF < 0.1）— v0.5 Exit criteria 2。
- 実装は完了（FA v2 + inter-head + session pool、commits `88b17c8..1dede25`）。実測は運用側で確認。

---

## 3. Kokoro-82M / Whisper 全サイズの実 checkpoint parity（M2-06 T09-T11 / M2-07 T11-T21）

**依頼者タスク**: PyTorch + transformers env で reference dump を生成し、実 GGUF vs PyTorch reference の parity fixture を提供する。

### 必要な準備

- [ ] Python 3.10+ + PyTorch 2.0+ + transformers 4.30+ + numpy。
- [ ] Hugging Face access（`openai/whisper-{base,small,medium,large-v3,turbo}` + `hexgrad/Kokoro-82M`）。

### 実行手順

```bash
# Whisper 4 サイズ（M2-06 T09/T11）
for size in whisper-small whisper-medium whisper-large-v3 whisper-turbo; do
  python3 tools/parity/dump_whisper_reference.py --model $size
done
# → tests/parity/whisper_{size}/ に fixture が入る

# Kokoro-82M（M2-07 T11）— スクリプトは本 WP で提供済み（tools/parity/dump_kokoro_reference.py 相当）
# 実装は WP 完了時点で bindings/kokoro/ 相当に含まれるが、run は依頼者側
python3 tools/parity/dump_kokoro_reference.py --model hexgrad/Kokoro-82M

# fixture 揃った後
cargo test -p vokra-models --test parity_whisper -- --nocapture
cargo test -p vokra-models --test parity_kokoro -- --nocapture
```

### Exit 判定への寄与

- v0.5 の Exit criteria には直接含まれないが、model zoo publish（M2-06 T16、M2-07 T24）と法務 audit（T18/T20、T25/T26）の前提。

---

## 4. モデル license audit + legal-compliance checklist 承認（M2-06 T18/T20、M2-07 T25/T26）

**依頼者タスク**: MIT / Apache 2.0 weight の商用配布可否を最終判断し、`docs/license-audit.md` + `docs/legal-compliance.md` に sign-off を残す。

### 必要な準備

- [ ] Whisper 4 サイズ MIT weight（openai/whisper）: 商用 OK の確認。
- [ ] Kokoro-82M Apache 2.0（hexgrad/Kokoro-82M）: 商用 OK + training data 疑義なしの確認。
- [ ] research flag 対象（F5-TTS / Fish-Speech / EnCodec）が公式 zoo に混入していないことの目視確認（M2-13 compliance gate が自動拒否するが最終目視）。

### 実行手順

`docs/license-audit.md` の各行に `Owner sign-off: <date>` を追記、`docs/legal-compliance.md` の Article 50 checklist を通す。

### Exit 判定への寄与

- FR-MD-13 のプロセス承認。M2-06/M2-07 WP 完了確定。

---

## 5. Discord bot デモ稼働（M2-10 / BR-08）

**依頼者タスク**: Discord Application + bot token を発行し、`integrations/vokra-discord-bot`（提案）or 自前 client で `vokra-server` を経由した音声 bot が稼働することを実証する。

### 必要な準備

- [ ] Discord Developer Portal で application 作成 → bot token 発行。
- [ ] `vokra-server` を起動できる Linux/macOS 環境（Docker 不要、`integrations/vokra-server` の single-binary）。

### 実行手順

`vokra-server` を起動:

```bash
cd integrations/vokra-server
cargo run --release -- \
  --bind 127.0.0.1:8080 \
  --whisper-gguf /path/to/whisper-large-v3.gguf \
  --piper-gguf /path/to/en-medium.gguf
```

Discord bot client（Python discord.py など）から `/v1/audio/transcriptions` + `/api/tts` を叩く。詳細は `integrations/vokra-server/docs/security-ops.md` の deploy 事例を参照。

### Exit 判定への寄与

- v0.5 Exit criteria 3（Discord bot デモ稼働）。
- 未稼働の場合は原因を M2-15 review へ持ち込み。

---

## 6. 言語バインディング初回対象合意（M2-12 T03）

**依頼者タスク**: 初回言語 = **Python (PyPI wheel)** で確定するか判断し sign-off。他候補（Swift / Kotlin / JS/TS）は rolling wave 次段。

### 必要な準備

- 特になし。CC は plan.md D1 の rationale で Python を推奨済み。

### 実行手順

- YES: 本チェックリストに ✅ を書き、rolling wave の次段の言語決定へ進む。
- NO: 別言語を指定 → CC が該当 binding scaffold を新規に構築（rolling wave）。

### Exit 判定への寄与

- M2-12 T03 の依頼者 sign-off。M2-12 WP 完了に必要。

---

## 7. PyPI 予約 + PYPI_API_TOKEN provision（M2-12 T17）

**依頼者タスク**: PyPI に `vokra` パッケージ名を予約（trademark 保護）、`PYPI_API_TOKEN` を GH Actions secret に登録するか OIDC trusted publisher を設定する。

### 必要な準備

- [ ] PyPI アカウント（既存 or 新規）。
- [ ] 2FA 設定（PyPI ルール）。

### 実行手順

1. `pip install twine` → `twine upload --repository testpypi bindings/python/dist/*.whl`（TestPyPI で dry-run）。
2. PyPI project 作成 → `pyproject.toml` の name = `vokra` を予約。
3. GH Actions secret `PYPI_API_TOKEN` を登録、または trusted publisher を GH Actions workflow に紐付け。
4. `git tag v0.5.0-rc1 && git push --tags` → `release.yml.python-pypi-publish` が起動、dry-run mode を経由して実 upload。

### Exit 判定への寄与

- M2-12 T17 の CD 発行完了。NFR-DS-03（PyPI 配布）。

---

## 8. Unity Editor license provision（M2-11 T-nightly）

**依頼者タスク**: `secrets.UNITY_LICENSE` を GH Actions secret に登録すると `nightly-il2cpp.yml` が IL2CPP スモークテストを nightly で実行するようになる。

### 必要な準備

- [ ] Unity Personal または Pro license（Unity Hub の manual activation → `.ulf` を base64 encode）。
- [ ] Unity 2022.3 LTS が installed（CI が game-ci/unity-builder@v4 でハンドル）。

### 実行手順

1. Unity Personal license を activate、`.ulf` を base64 encode。
2. GH Actions secret `UNITY_LICENSE` に登録。
3. 次の nightly 実行で `nightly-il2cpp.yml` が回り、IL2CPP AOT + DllImport(__Internal) + VokraAndroidAssets passthrough が検証される。

### Exit 判定への寄与

- M2-11 「IL2CPP 対応デモ動作」の実運用検証。
- 未 provision の場合は TESTING.md の手動手順を依頼者がローカル実行して署名。

---

## 9. Wyoming / Home Assistant 統合検証（M2-15 Kill switch J）

**依頼者タスク**: HA Voice PE + Wyoming Protocol クライアントで `vokra-server` を「推奨 Wyoming Server」として認識・接続する試験。採用可否は依頼者判断。

### 必要な準備

- [ ] Home Assistant + HA Voice PE の実験環境（M5Stack 実機不要、docker 上でも可）。
- [ ] `vokra-server` を Wyoming モードで起動できる Linux/macOS 環境。

### 実行手順

`integrations/vokra-server/docs/wyoming-design.md` の HA 接続例を参照。

### Exit 判定への寄与

- Kill switch J 判定（HA 採用可否）。v0.5 時点で判定。
- 「未採用」= Kill switch 発動、「採用」= v1.0 の Wyoming 主要 endpoint 化。

---

## 10. Kill switch C/K 判定（M2-15 / 2026-12〜2027-01 目安）

**依頼者タスク**: v0.1 MVP 公開後 3 ヶ月時点（暦月目安 2026-12〜2027-01）に GitHub star 数と competitor community metric を再測し、以下を判定する。

### 判定閾値（milestones.md §6 転記）

- **C**: v0.1 MVP 公開後 3 ヶ月で GitHub star < 500 or 総合 engagement 過小 → 撤退検討。
- **K**: v0.5 時点で addressable market が競合の 10% 未満 → 撤退検討。

### 必要な準備

- 特になし。github.com/ayutaz/vokra の star 数 + Issues/Discussions active user proxy を集計。

### 実行手順

四半期 Go/No-go review record を独立公開ガバナンス記録として発行（governance docs / GitHub Discussion / post-mortem blog のいずれか）。

### Exit 判定への寄与

- Kill switch C/K の判定結果を M2-15 記録として公開。継続 or exit path 選択。

---

## Summary 進捗表（2026-07-06 時点）

| WP | 内容 | CC 進捗 | 依頼者残タスク |
|----|------|--------|--------------|
| M2-01 | Metal backend | ✅ 完了 | — |
| M2-02 | iOS build scaffold | ✅ 完了（scaffold） | § 1（iOS 実機 RTF） |
| M2-03 | CUDA backend + RTF<0.1 保証 | ✅ 完了（実装） | § 2（CUDA 実測） |
| M2-04 | graph fusion（log-mel 1 kernel） | ✅ 完了 | — |
| M2-05 | istft_streaming op | ✅ 完了 | — |
| M2-06 | Whisper large-v3/turbo | ✅ 部分完了 | § 3（parity fixture）+ § 4（audit） |
| M2-07 | Kokoro-82M | ✅ 骨格完了 | § 3（parity fixture）+ § 4（audit） |
| M2-08 | quantization policy | ✅ 完了 | — |
| M2-09 | vokra-server 4 互換 API | ✅ 完了 | § 5（deploy 事例）|
| M2-10 | Discord bot デモ | 🚧 部分（サーバ側） | § 5（bot token + 稼働）|
| M2-11 | Unity official plugin | ✅ 完了（UPM CD） | § 8（Unity license）|
| M2-12 | 言語バインディング（Python 初回） | ✅ 完了（wheel scaffold） | § 6（合意）+ § 7（PyPI token）|
| M2-13 | compliance 拡張 | ✅ 完了 | — |
| M2-14 | 実機ベンチ計測 | 引き渡し済み | § 1 + § 2 |
| M2-15 | 四半期 Go/No-go review | 継続監視 | § 9（Kill switch J）+ § 10（C/K）|

---

## Contact / Escalation

- CC 側で追加 workflow が必要になった場合（例: 新規言語バインディング着手、実測結果を受けた最適化 follow-up）は本チェックリストに追記して依頼者から CC に振る。
- v0.5 Exit 判定は上記全項目の消化 + milestones.md §6 Exit criteria を根拠に依頼者が最終判断。

