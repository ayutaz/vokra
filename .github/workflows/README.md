# Vokra CI/CD workflow 目次

本ドキュメントは Vokra の GitHub Actions workflow の **single source of truth (SoT)** です。
cron 時刻・required check name・trigger の記述に各 workflow file の comment との差異が
あった場合、**本 file が真** とみなし、各 workflow file 側を後追いで揃えます。

- 対象範囲: `.github/workflows/*.yml` 全件（**実数 45 file**。旧記載「2026-07-23 時点で 20 file」は
  その後の parity workflow 増設で陳腐化していたため実測値に更新）
- **2026-08-20 網羅確認**: 45 workflow file の全てを本 index に収録済み。
  required / advisory / weekly / nightly / release / manual のいずれかに各 `.yml` を
  明示し、cron 値は workflow の実 `schedule:` から転記した。
- required check name の実態: `gh api /repos/ayutaz/vokra/branches/main/protection/required_status_checks`
  を primary source として取得し、本 file の §1 に転記
- cron 時刻の実態: 各 workflow file 内の `schedule: - cron: '...'` 実定義から抽出。
  各 workflow の comment 側は本 file を参照する形に段階的に集約予定
- 変更禁止事項: 本 file の書式や見出し名の変更は required check job id の追跡性を
  壊すことがあるため、**§1 の table 構造は変更しない** こと（job id / check name / 定義 file
  の列は追加・削除禁止、値の差替のみ許可）

---

## 1. Required checks (main branch protection)

`main` への PR merge を gate する 14 checks。branch protection API 側の contexts と一対一に
対応しなければならない。`build (*)` と `test (*)` は `ci.yml` の matrix expansion 由来で、
`os: [ubuntu-latest, macos-latest, windows-latest]` の 3 展開を **必ずこの順** で確保する。
`parity` は matrix job (`parity-matrix`) の aggregator（Wave 10 で分離済、matrix 直下では
`parity (0)` `parity (1)` のような per-index context になり branch protection が pending で
停滞するため）。

| check name | job id | 定義 file | 目的 |
|---|---|---|---|
| build (ubuntu-latest) | build (matrix os=ubuntu-latest) | .github/workflows/ci.yml | Linux workspace release build (default features) |
| build (macos-latest) | build (matrix os=macos-latest) | .github/workflows/ci.yml | macOS workspace release build (default features) |
| build (windows-latest) | build (matrix os=windows-latest) | .github/workflows/ci.yml | Windows workspace release build (default features) |
| test (ubuntu-latest) | test (matrix os=ubuntu-latest) | .github/workflows/ci.yml | Linux `cargo test --workspace` (default features) |
| test (macos-latest) | test (matrix os=macos-latest) | .github/workflows/ci.yml | macOS `cargo test --workspace` (default features) |
| test (windows-latest) | test (matrix os=windows-latest) | .github/workflows/ci.yml | Windows `cargo test --workspace` (default features) |
| fmt | fmt | .github/workflows/ci.yml | `cargo fmt --all -- --check` |
| clippy | clippy | .github/workflows/ci.yml | `cargo clippy --workspace --all-targets -- -D warnings` |
| parity | parity | .github/workflows/ci.yml | fixture-only parity aggregator (needs: parity-matrix)、真の重量級 real-weight parity は §3 の weekly leg |
| license | license | .github/workflows/ci.yml | `cargo deny` + hard zoo-manifest completeness/no-NC + `scripts/check-*` compliance gate |
| workflow-security | workflow-security | .github/workflows/ci-security.yml | actionlint + ShellCheck + zizmor workflow security gate |
| dependency-review | dependency-review | .github/workflows/ci-security.yml | changed dependency の vulnerability / license / Scorecard gate |
| documentation-links | documentation-links | .github/workflows/ci-security.yml | public documentation surface の link validation |
| CodeQL | analyze (`codeql-rust`) | .github/workflows/codeql.yml | GitHub CodeQL Rust `security-extended` analysis |

---

## 2. Additional required and advisory checks

§1 の14 contexts以外はmergeをgateしないinformational check。定期監視で赤化を
検知し、依頼者判定でrequired昇格するかを判断する（初期 promotion 方針: 連続数週の
緑が確認できた時点でownerに昇格提案）。§2.2の先頭4項目は2026-08-20にrequiredへ
昇格済みで、詳細説明を重複掲載している。`pins.yaml ↔ workflow sync` は4週間の
advisory soakを完了し、2026-08-20にhard-fail + 全PR起動へ変更した。PRとmainのgreenを
確認後、§1とbranch protectionへ追加する（設定反映まではrequiredではない）。

### 2.1 ci*.yml 内 (PR/push で常時走行、advisory)

**2026-07-23 の CI 分割 (`chore/ci-cd-reorg-2026-07-23`) 反映済**: advisory job の大半は `ci.yml`
から `ci-quality.yml` (lint / audit / doc-drift / server-compat) と `ci-platform.yml`
(platform build target / GPU backend / RTF regression / server-deployment) へ移動した。
`ci.yml` に残る advisory は required 中核と artifact chain を共有する 4 件のみ。
**`定義 file` 列は分割後の実ファイルを指す** (下表は
`awk '/^jobs:/{i=1;next} i&&/^[^ ]/{i=0} i&&/^  [A-Za-z0-9_-]+:$/' .github/workflows/ci*.yml`
の実出力と一致することを確認済)。

| check name | 定義 file | 目的 |
|---|---|---|
| repo-hygiene | .github/workflows/ci-quality.yml | tracked file の scratch/gitignore drift 検査 |
| catalog-audit | .github/workflows/ci-quality.yml | model catalog の実在性 audit (catalog-reality gate) |
| unity-capi-lints | .github/workflows/ci-quality.yml | Unity C# 側 P/Invoke lint |
| capi-smoke | .github/workflows/ci-quality.yml | C ABI smoke test (bytes error-path / session / stream / aec / s2s) |
| msrv | .github/workflows/ci-quality.yml | Minimum Supported Rust Version 追随 |
| abi-surface | .github/workflows/ci-quality.yml | `include/vokra.h` drift + Rust public-api snapshot + m0 anchor 差分 |
| doc-examples | .github/workflows/ci-quality.yml | rustdoc code fence の compile & run |
| doctests | .github/workflows/ci-quality.yml | `cargo test --doc` |
| rustdoc | .github/workflows/ci-quality.yml | `cargo doc --workspace` warn-as-error |
| fa-v3-confinement | .github/workflows/ci-quality.yml | FlashAttention v3 が v1.5+ 前倒し禁止に閉じ込められていることの assert |
| python-license-audit | .github/workflows/ci-quality.yml | Python 補助 tool の pip-licenses audit |
| server-compat | .github/workflows/ci-quality.yml | OpenAI / vLLM / Wyoming プロトコル互換 leg (3-OS cross-build) |
| cargo-audit | .github/workflows/ci-quality.yml | RustSec advisory audit |
| typos | .github/workflows/ci-quality.yml | typo checker |
| excluded-workspace (5 matrix cells) | .github/workflows/ci-quality.yml | 専用CIを持つserver/Godot以外の除外workspace（Android / CLI server bench / Misaki G2P / Piper G2P / ureq server bench）を各lockfile付きでbuild/test |
| python-parity-oracles | .github/workflows/ci-quality.yml | Python 側 parity oracle script の self-test |
| shell-self-tests | .github/workflows/ci-quality.yml | `scripts/*.sh` の `--self-test` 集約 |
| coverage (advisory) | .github/workflows/coverage.yml | Rust差分を含むPR / main pushのcoverage計測（prose-only PRは非起動） |
| gitleaks (advisory) | .github/workflows/secret-scan.yml | PR / main push の secret scan |
| gpu-backends | .github/workflows/ci-platform.yml | macos=metal / ubuntu=cuda opt-in feature の build/clippy/test (+ coreml/qnn delegate scaffold arm) |
| build-target-vulkan-only | .github/workflows/ci-platform.yml | M4-15 CPU+Vulkan-only SKU build target 検証 |
| riscv-cross-build | .github/workflows/ci-platform.yml | RVV 1.0 (M3-13) cross build + ISA dispatch asm 検査 |
| build-windows-arm64 | .github/workflows/ci-platform.yml | `aarch64-pc-windows-msvc` cross build |
| cpu-isa-server-tier | .github/workflows/ci-platform.yml | M4-17 AVX-512/VNNI/BF16 + ARM64 dotprod/i8mm/bf16 dispatch build |
| bench-regression | .github/workflows/ci-platform.yml | 5% regression gate (M3-01 defer 分は M2-14 self-hosted runner まで aspirational) |
| server-deployment | .github/workflows/ci-platform.yml | vokra-server musl 静的リンク + runtime workspace 非汚染 (FR-SV-01 / NFR-DS-02) |
| ios-build | .github/workflows/ci.yml | iOS `libvokra.a` static build + `verify-xcframework.sh` |
| parity-matrix | .github/workflows/ci.yml | fixture parity matrix leg (aggregator `parity` の元) |
| unity-package | .github/workflows/ci.yml | Unity plugin package audit (M2-11、UNITY_LICENSE 未 provisioning ゆえ WARN skip) |
| python-wheel-build | .github/workflows/ci.yml + python-wheels.yml | 4 native wheels（manylinux x86_64 / macOS arm64+x86_64 / Windows x86_64）のrepair・archive/arch検証・Python 3.9/3.12 clean-installを集約 |

### 2.2 Security / supply chain / advanced Rust quality

`workflow-security` / `dependency-review` / `documentation-links` / `CodeQL`
は§1のrequired contexts。残りはadvisoryまたはschedule/manual専用である。

| check / workflow | trigger | 役割 |
|---|---|---|
| workflow-security (`.github/workflows/ci-security.yml`) | PR / main / weekly / manual | actionlint + ShellCheck と zizmor。workflow 構文、固定 SHA、最小権限、expression injection を gate |
| dependency-review (`.github/workflows/ci-security.yml`) | PR | moderate以上の脆弱性、GPL/LGPL/AGPL/SSPL系licenseを持つ新規runtime/development依存を拒否し、変更依存のOpenSSF Scorecardも表示 |
| documentation-links (`.github/workflows/ci-security.yml`) | PR / main / weekly / manual | README / CONTRIBUTING / `.github` / `docs` のリンク腐敗を lychee で検出 |
| CodeQL (`codeql-rust`, `.github/workflows/codeql.yml`) | PR / main / weekly / manual | Rust を buildless `security-extended` query suite で SAST |
| scorecard (`.github/workflows/scorecard.yml`) | main / branch-protection change / weekly / manual | OpenSSF Scorecard を SARIF と public results に送信 |
| cargo-hack-each-feature (`.github/workflows/rust-advanced.yml`) | Rust 関連 PR / main / weekly / manual | 全 feature の独立コンパイルと zero-dep lockfile 不変条件 |
| cargo-semver-checks (`.github/workflows/rust-advanced.yml`) | Rust 関連 PR | publishable crate の意図しない public API break を base SHA と比較 |
| miri / address-sanitizer (`.github/workflows/rust-advanced.yml`) | weekly / manual | unsafe boundary の UB、provenance、memory error を nightly toolchain で検出 |
| fuzz-* (`.github/workflows/rust-advanced.yml`) | Rust 関連 PR / main / weekly / manual | GGUF / safetensors / JSON の不正入力を libFuzzer で検査、reproducer を artifact 保存 |

weekly cron は CodeQL=`Thursday 02:41 UTC`、CI Security=`Wednesday 03:23 UTC`、
Scorecard=`Thursday 03:17 UTC`、Rust Advanced=`Sunday 01:37 UTC`。PR のセキュリティ
gate と、変動しやすい nightly toolchain / advisory DB の定期監視を分離する。

### 2.3 Repository-native security controls

2026-08-18 時点で GitHub 側の secret scanning、push protection、Dependabot alerts、
Dependabot security updates、private vulnerability reporting を有効化済み。
`SECURITY.md` の private advisory URL と repository setting は一致している。

**2026-08-20 post-merge audit**: full-SHA workflow移行と`workflow-security`での検証、
`workflow-security` / `dependency-review` / `documentation-links` / `codeql-rust`の
required追加は完了した。残るGitHub側設定は、repository Actions policyの
`sha_pinning_required`（現在false）と、branch protectionのstrict status checks /
conversation resolution / linear history / administrator enforcement（いずれも現在false）
である。CI hardening PRをPR上とmain上の両方でgreen確認してから、この順で有効化する。
`pins.yaml ↔ workflow sync`のrequired追加も同じmain-green確認後に行う。

### 2.4 release.yml 内 (tag v* 起動、advisory & release パイプ)

release パイプの詳細は §5 を参照。

---

## 3. Weekly parity (Monday/Tuesday stagger)

参照実装との重量級 parity や HF Hub 由来 checkpoint DL を伴うため、PR 単位では走らせず
週次 stagger で走らせる。**同一 self-hosted runner を共有する場合でも直列化しないよう
30 分刻みで slot 分割** している。cron の変更は本 table を必ず更新すること（各 workflow
file 側 comment に埋め込まれている「stagger 一覧」も本 table を参照して同期する）。

| 曜日 UTC | 時刻 UTC | workflow | 対象モデル / 用途 |
|---|---|---|---|
| Monday | 04:00 | .github/workflows/parity-kokoro-real.yml | Kokoro-82M pinned SHA `f3ff3571` 全 9 tensor per-tensor atol parity |
| Monday | 05:00 | .github/workflows/parity-whisper-real.yml | Whisper base (cron 既定) + workflow_dispatch で small/medium/turbo/large-v3 opt-in |
| Monday | 05:15 | .github/workflows/parity-voxtral-real.yml | Voxtral Mini real-checkpoint parity（opt-in gate） |
| Monday | 05:30 | .github/workflows/gpu-vulkan-parity.yml | Vulkan (mesa lavapipe) Whisper base parity |
| Monday | 06:00 | .github/workflows/gpu-cuda-rtf.yml | CUDA RTF measurement N=10 (self-hosted 必須、未登録なら clean skip) |
| Monday | 06:30 | .github/workflows/parity-csm-real.yml | CSM synthetic format pin + real T29 reference は opt-in |
| Monday | 07:00 | .github/workflows/web-wasm.yml | WASM SIMD128 + WebGPU + Whisper base WASM e2e (opt-in) |
| Monday | 07:15 | .github/workflows/parity-sbv2-real.yml | SBV2 JA/EN と opt-in ZH real parity + sidecar-hash gate |
| Monday | 07:30 | .github/workflows/parity-moshi-real.yml | Moshi 切詰め parity (full-7B は 16GB RAM で mmap converter blocked) |
| Monday | 08:00 | .github/workflows/parity-utmos.yml | UTMOS 22 STRONG parity (M4-18 un-defer 材料) |
| Monday | 08:30 | .github/workflows/godot-crossbuild.yml | Godot GDExtension 5-target crossbuild + AssetLib zip package |
| Monday | 08:45 | .github/workflows/parity-cosyvoice2-real.yml | CosyVoice2 real-checkpoint parity |
| Monday | 09:00 | .github/workflows/release-cadence.yml | リリース cadence レポート |
| Monday | 09:05 | .github/workflows/parity-nemo-asr-real.yml | NeMo ASR family real-checkpoint parity（dispatch matrix / opt-in） |
| Monday | 09:15 | .github/workflows/parity-rmvpe-real.yml | RMVPE real-checkpoint parity |
| Monday | 09:30 | .github/workflows/corpus-drift-detector.yml | `.github/pins.yaml` 全 entry と canonical license URL の drift 検査（upstream/license URL=advisory / mirror=hard_fail、informational） |
| Monday | 09:35 | .github/workflows/parity-whisper-extras-real.yml | Distil-Whisper / Kotoba-Whisper real parity（opt-in） |
| Monday | 10:00 | .github/workflows/silero-nostd-cross-build.yml | vokra-vad-micro (M5-03 no_std) thumbv8m cross build |
| Monday | 10:15 | .github/workflows/parity-tts-dac-real.yml | Dia / Zonos DAC-family TTS parity（opt-in） |
| Monday | 10:30 | .github/workflows/parity-tts-hiftnet-real.yml | HiFTNet-family TTS parity（CosyVoice variants、opt-in） |
| Monday | 11:00 | .github/workflows/parity-qwen3-tts-real.yml | Qwen3-TTS real-checkpoint parity（opt-in） |
| Monday | 11:30 | .github/workflows/parity-tts-continuous-vae-real.yml | VoxCPM2 / VibeVoice continuous-VAE TTS parity（opt-in） |
| Monday | 12:00 | .github/workflows/parity-tts-japanese-real.yml | Irodori / Japanese VITS parity（opt-in） |
| Monday | 12:30 | .github/workflows/parity-deepfilternet3-real.yml | DeepFilterNet3 real-checkpoint parity（opt-in） |
| Monday | 13:00 | .github/workflows/parity-deberta-v3-large-real.yml | DeBERTa-v3-large real-checkpoint parity（opt-in） |
| Tuesday | 06:00 | .github/workflows/parity-rvq-real.yml | Mimi/DAC RVQ codec 実 parity (Monday hub outage で全滅回避のため火曜) |

---

## 4. Nightly

日次で走らせる長時間 job。GH Actions の :00 cron 集中を避けるため
`XX:17` `XX:47` の 30 分オフセット刻みで stagger する。

| 時刻 UTC | workflow | 目的 |
|---|---|---|
| 04:17 | .github/workflows/nightly-il2cpp.yml | Unity IL2CPP Linux headless build (license check + native cdylib) |
| 04:47 | .github/workflows/nightly-webgl.yml | Unity WebGL preflight + wasm-harness + WebGL build |
| 05:17 | .github/workflows/nightly-asr-wer.yml | LibriSpeech `1272/128104` WER regression (`test_librispeech_wer.py`) |
| 05:47 | .github/workflows/nightly-tier2-device.yml | Raspberry Pi 3B/4B/5/Zero 2 W 実機 RTF (self-hosted 必須、未登録なら clean skip) |
| 06:17 | .github/workflows/nightly-full-parity.yml | staged GGUFを用いるdynamic full-parity matrix（全PRでSileroを実行、未設定legはplan summaryへ未測定記録） |

---

## 5. Release trigger (tag v*)

`v*` タグ push で起動する release パイプ。`workflow_dispatch` の `dry_run` 既定は `true`
（ローカル/CI 検証のみ、`gh release upload` と Package.swift patch は skip）。

| trigger | workflow | 主要 job (定義順) |
|---|---|---|
| push tag `v*` / workflow_dispatch | .github/workflows/release.yml | validate-tag → release-notes → ios / Unity / Python（同一runでpython-wheels.ymlをcall）/ Godot / npm / desktop / Android release assets。crates-io-dry-run は並列、crates-io-publish は release-notes + dry-run 後 |

release パイプ内の job は全て advisory (branch protection 対象外)。crate publish は
`crates-io-dry-run` の green を人手で確認したうえで `crates-io-publish` を走らせる 2 段。
Release 本体の作成と CHANGELOG 検証を成果物 upload より先に完了させ、
`gh release upload` の並列 race と、不正な release notes のまま registry を更新する経路を防ぐ。
`workflow_dispatch` の `dry_run=true` は artifact 確認のみで、Release / registry / branch を更新しない。

タグ release の XCFramework / Unity UPM / Python wheel / Godot AssetLib / npm / desktop /
Android AAR は SPDX SBOM 付き GitHub artifact attestation を発行する。ダウンロード後の
検証は次の形に統一する。

```bash
# Build provenance
gh attestation verify <artifact-path> --repo ayutaz/vokra

# SPDX 2.3 SBOM predicate
gh attestation verify <artifact-path> --repo ayutaz/vokra \
  --predicate-type https://spdx.dev/Document/v2.3
```

---

## 6. 手動 / docs 連動 (workflow_dispatch or docs path push)

PR/cron/tag のいずれでも起動せず、`workflow_dispatch` か特定 path への push だけを
trigger とする workflow。

| trigger | workflow | 目的 |
|---|---|---|
| workflow_dispatch only | .github/workflows/bench-baseline-capture.yml | `iters` iteration で bench baseline を再取得 |
| push main (`docs/bench-baselines/**`, `docs/perf/**`, `docs/benchmarks/**`, `tools/bench/build_dashboard.py`, `.github/workflows/dashboard.yml`) + workflow_dispatch | .github/workflows/dashboard.yml | perf dashboard 再生成 + GitHub Pages deploy |

path filter で PR trigger にも参加している workflow は §3 の weekly parity と
`nightly-full-parity.yml` に含めた。各 workflow の `pull_request.paths` に該当する
crate / dumper / fixture を触った場合のみ再実行され、いずれも branch-protection
context ではなく informational である。

---

## 7. 拡張プラットフォーム candidate と非採用理由 (2026-07-23、Phase 3b Task 4)

本 section は Phase 3b Task 4a / 4b / 4c の判定 primary source。判定の根拠は
[actions/runner-images README](https://github.com/actions/runner-images) (2026-07 snapshot)
および `docs.github.com/en/billing/reference/actions-runner-pricing`。判定を追加・変更する
場合は evidence source と併せてここに記録すること (docs 更新なしの workflow-only 変更を
禁止する目的で本 section を primary source と位置付ける)。

### 7.1 macOS Apple Silicon (Task 4a) — 追加不要（既に対応済）

- `macos-latest` は 2026-07 時点で **macOS 26 Arm64 (Apple Silicon)** にマッピング済
  (actions/runner-images README の「Available Images」table 参照)。
- 既存の required check `build (macos-latest)` / `test (macos-latest)` および
  advisory の `gpu-backends` metal / coreml arm はすべて Apple Silicon 上で実行される。
  `macos-14` (Apple Silicon 明示 alias) の matrix 追加は **不要** — かつ `macos-14` は
  runner-images README で `deprecated` badge 付き。
- Intel Mac coverage を明示するには `macos-latest-large` (macOS 26 Intel) を追加する
  余地があるが、これは larger runner tier で **public repo でも有料** (§7.3)。Vokra は
  Metal backend を Apple Silicon 上で検証済でありコード面の Intel Mac 依存は薄いため
  cost trade-off を正当化できない → **見送り**。

### 7.2 Windows ARM64 (Task 4b) — 採用、advisory job として追加

- `windows-11-arm` runner を **`build-windows-arm64` advisory job として
  `.github/workflows/ci-platform.yml` に追加** (2025-04 public preview 発表、2026-07 時点
  runner-images README では `preview` badge なし → GA 相当。public repo 無料)。
- ci.yml build/test matrix には **絶対に追加しない** — 既存の required check name
  `build (windows-latest)` / `test (windows-latest)` を不変に保つため (branch protection
  は name-pinned、§1 参照)。ci-platform.yml の advisory ゾーンに独立 job として追加
  すれば required check には触らずに ARM64 coverage を確保できる。
- Rust target = `aarch64-pc-windows-msvc` (tier-2)、`-p vokra-cli` + `-p vokra-capi` の
  release build + zero-dep `Cargo.lock` tripwire。`continue-on-error: true` で sibling
  job の cancellation を防ぐ。
- Promotion path (M4-15 build-target-vulkan-only と同型): 数週の連続緑で owner が
  required matrix 昇格を判定。

### 7.3 Larger runner (Task 4c) — 見送り

- GitHub docs (`docs.github.com/en/billing/reference/actions-runner-pricing`) が
  「**The larger runners are not free for public repositories.**」と明記。task 4c 説明
  にあった「public repo は larger runner でも無料枠 unlimited」の前提は **不成立**。
- Vokra は public repo 運用のため、`ubuntu-latest-4-cores` / `ubuntu-latest-8-cores` /
  `ubuntu-latest-16-cores` / `macos-latest-large` 等の適用は **有料化**。既存 workflow
  は standard runner (2 vCPU / 4 vCPU クラス) で許容 runtime 内 (parity-whisper-real
  4 サイズの実測でも 6h GitHub Actions limit の遥かに手前) のため cost trade-off が
  正当化されない → **適用しない**。
- 再評価条件: (a) 具体の workflow が 6h の GH Actions timeout に接触して fail 化、
  (b) GitHub が public repo 向け無料枠 larger runner を提供開始、いずれかで再判定。

---

## 8. Advisory step / job canonical form (SoT)

Advisory checks and steps record measurements without gating merges. §2 defines
WHICH checks are advisory; this section defines HOW to write one so the shape
is consistent across workflows. Adopting this idiom prevents the class of bugs
seen in `fix/nightly-asr-wer-2026-07-23` P1 — a step that DOCUMENTED itself as
advisory-downgraded but LEXICALLY was a bare `tar` under `set -euo pipefail`,
i.e. hard-failed on the very drift mode it claimed to tolerate.

### 8.1 Job level

- Do NOT add to branch-protection required list.
- If the job's failure signal itself should not fail the workflow, set
  `continue-on-error: true` on the job.
- Downstream aggregator jobs must use `if: always()` (or an explicit
  `needs.<job>.outputs.<gate>` guard); `success()` is wrong because
  `continue-on-error: true` still reports `success` on failure.

### 8.2 Step level — canonical bash block

```yaml
- name: <what it measures> (advisory, record-only)
  id: <slug>
  continue-on-error: true         # (a) defence in depth
  shell: bash
  run: |
    set -euo pipefail
    # ... measurement / probe ...
    if ! <probe>; then             # (b) catch expected failure mode
      echo "::warning::<what drifted> — advisory downgrade" \
           "(<upstream|environment>, not a Vokra regression). See <WP-ID>."
      echo "<gate>=false" >> "$GITHUB_OUTPUT"
      exit 0                       # step exits 0; downstream `if:` cascades skip
    fi
    echo "<gate>=true" >> "$GITHUB_OUTPUT"
```

Rules:

- **(a)** `continue-on-error: true` guards against an unhandled failure inside
  the block (nested `set -e`, `mktemp` `trap` fires, a helper script's exit
  leaking out). Keep it even when (b) is used — it is defence in depth.
- **(b)** For expected-and-benign failure modes (CDN drift, missing optional
  weights, upstream repackaging), catch them with `if ! <cmd>; then ... exit 0`
  so the step's exit code stays 0 AND the sticky `outputs.<gate>=false` signal
  drives downstream cascade skip. Do NOT rely on (a) alone — (a) makes the
  step exit 0 to the job, but `steps.<slug>.outputs.<gate>` will be UNSET,
  which is neither `'true'` nor `'false'` and can produce silent-skip bugs.
- **downstream `if:`** — every step downstream of an advisory probe MUST guard
  with `if: steps.<slug>.outputs.<gate> == 'true'`. Do not use `success()`:
  the advisory step succeeds on the failure path too (that is the whole point).
- **step name suffix** — `(advisory, record-only)` in the step name documents
  intent inline. Optional tripwire `scripts/check-workflow-advisory-suffix.sh`
  greps `continue-on-error: true` steps for the suffix (follow-up).

### 8.3 Placeholder / dev-branch guard

Advisory sections that branch on a placeholder (e.g. "when weights arrive")
must wrap the divergent block so distinct non-zero exits are preserved:

```bash
set +e
<placeholder command>
rc=$?
set -e
if [ "$rc" -ne 0 ]; then
  echo "::warning::placeholder branch inactive (rc=$rc) — advisory"
  echo "<gate>=false" >> "$GITHUB_OUTPUT"
  exit 0
fi
```

The naked `<cmd> || true` idiom is BANNED because it collapses all non-zero
exits into a single "OK" signal, making the drift detector (§8.5) unable to
distinguish "placeholder awaiting provisioning" from "advisory actively broken".

The distinction against FR-EX-08 (no silent CPU fallback): placeholder-inactive
is a KNOWN missing-input state (permitted; the gate cascade correctly skips
the downstream measurement), whereas silent CPU fallback is a WRONG-answer
state (banned; the caller believes it got GPU results but got CPU numbers).

### 8.4 CDN / corpus drift operational flow

1. `corpus-drift-detector.yml` (weekly Monday 09:30 UTC) emits `::warning::`
   for any pinned external URL whose current sha256 differs from
   `.github/pins.yaml`.
2. If the drift is persistent (2 consecutive weeks), owner opens or advances
   WP X-10-Txx (self-mirror) — see `docs/adr/X-10-corpus-self-mirror.md`.
3. Owner uploads a bit-identical mirror to
   `huggingface.co/datasets/vokra/<slug>` via the existing 5-gate
   `scripts/publish-one.sh` pipe. Mirror SHA lands in
   `.github/pins.yaml:entries[*].mirror.hf_revision` +
   `entries[*].mirror.sha256`.
4. Workflow env switches to the graceful-fallback seam:
   `${{ vars.VOKRA_CORPUS_<SLUG>_MIRROR_URL || '<upstream-url>' }}`.
   Existing behaviour is preserved when the org variable is unset — so the
   seam can land in a separate PR from the mirror upload (see
   `nightly-asr-wer.yml` commit B for the canonical implementation).
5. Once the mirror is live, drift detector continues to compare BOTH upstream
   (advisory) AND mirror (hard-fail: Vokra owns byte-identity).

### 8.5 Related tripwires

- `.github/workflows/pins-sync-check.yml` — every entry in
  `.github/pins.yaml` must appear verbatim in its `owning_workflow` and
  every typed SHA literal in a workflow env must be registered in the
  catalog (catches "workflow edited without updating catalog" and
  "unregistered pin added to a workflow"). The 2026-07-23–08-20 advisory
  soak completed without observed false positives; the job now hard-fails
  on every PR. Add its stable context to branch protection only after the
  promoted form is green on both its PR and main (matches §2 policy).
- `.github/workflows/corpus-drift-detector.yml` — weekly probe per §8.4.
- `docs/adr/X-10-corpus-self-mirror.md` — the mirror decision itself
  (gitignore-local internal ADR).

### 8.6 Python / uv policy

Workflow Python follows the same uv-only rule as local tooling. The 2026-08-18
migration started from 24 workflow files / 36 jobs / 152 bare invocations and
closed with zero exceptions:

- stdlib-only helpers use
  `uv run --no-project --python 3.12 python <script>`;
- dependency-bearing jobs create an environment with `uv venv`, install with
  `uv pip --python <venv-python>`, and run with an explicit interpreter;
- shell activation, direct venv executables, bare pip/pytest, and runner-user
  site mutation are prohibited;
- every Python-using job installs the pinned
  `astral-sh/setup-uv@ae62891fec2bb8e7d6c99fc78c9fec3a63790f8d` action (v10.0.0).

`scripts/check-workflow-hygiene.sh` enforces the command rule over block and
scalar `run:` entries. The migration inventory, cross-platform wheel recipe,
and completion evidence are recorded in
[`docs/handoff/workflow-python-uv-migration-2026-08-18.md`](../../docs/handoff/workflow-python-uv-migration-2026-08-18.md).

### 8.7 Immutable CI dependencies

Remote Actionsはすべてfull 40-hex commit SHA、Docker imageはversion tagに加えて
multi-architecture OCI `@sha256` digestを必須とする。`secret-scan.yml`のgitleaksは
`v8.30.1@sha256:c00b6bd0aeb3071cbcb79009cb16a60dd9e0a7c60e2be9ab65d25e6bc8abbb7f`
へ固定済み。tagはreview時の可読性、digestは実行identityとして両方を残す。

`scripts/check-workflow-hygiene.sh`はremote `uses:`、workflow container image、
`docker run` / `docker pull`を横断し、mutable refをhard-failする。self-testはmoving
Action tagとdigestなしDocker imageを実際に拒否し、tag+digestを受理できることも検証する。

---

## 参考

- branch protection の実態: `gh api /repos/ayutaz/vokra/branches/main/protection/required_status_checks`
- workflow 一覧: `ls .github/workflows/*.yml`
- 各 workflow の script side: `scripts/` 配下 (`check-*.sh` / `build-*.sh` / `parity/*.sh` など)
- Rust workspace の zero-dep 不変条件 (NFR-DS-02): `root Cargo.lock` は `vokra-*` のみ
  ゆえ、workflow から `cargo add`/`cargo update` を投入する変更は原則禁止 (uv-managed
  Python environment や wasm / iOS 側 build script 内での隔離依存導入のみ許可)
