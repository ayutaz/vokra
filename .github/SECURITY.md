# Security Policy

Vokra is a Rust inference runtime for speech AI models, developed and maintained
as an OSS project by a single maintainer with the assistance of Claude Code. This
document defines how to report security vulnerabilities and what to expect in
return.

## Supported versions

Vokra is pre-1.0 (release-candidate). Security fixes land on the current rc
line; older lines do not receive backports.

| Version                | Status              | Security fixes         |
| ---------------------- | ------------------- | ---------------------- |
| `v1.0-rc.*` (current)  | Active (M4 → M5)    | Best-effort, current line only |
| `v1.0` GA (future)     | Not yet released    | Will supersede rc line |
| `v0.9`                 | End-of-life         | None                   |
| `v0.5` and older       | End-of-life         | None                   |

"Best-effort" means a single maintainer with no SLA commitment. High-severity
issues (RCE, unrestricted heap write, cryptographic bypass in the compliance
gate) are prioritised over low-severity issues.

## Reporting a vulnerability

**Please do not open a public GitHub issue for security reports.** Public issues
are indexed by search engines within minutes and give attackers a head-start on
weaponisation.

Preferred: **GitHub private vulnerability reporting**

- <https://github.com/ayutaz/vokra/security/advisories/new>
- Uses GitHub's built-in coordinated-disclosure workflow. Details stay private
  between the reporter and the maintainer until a fix is ready. A CVE can be
  requested from within the advisory.

Fallback: **Email**

- <yousan@aihub.tokyo>
- Use this path if you cannot open a GitHub advisory (e.g. no GitHub account,
  or the advisory UI is not reachable). Do not include unencrypted secrets
  (real credentials, private keys). If you must share a proof-of-concept that
  contains sensitive material, ask for a PGP key first.

Please include, if possible:

- Vokra version and commit SHA (`vokra-cli --version` or the tag/hash you built)
- Platform (OS, architecture, backend: `cpu` / `metal` / `cuda` / `vulkan` /
  `webgpu`)
- Reproduction: minimal input, GGUF / safetensors file that triggers the issue
  (do not attach non-redistributable model weights — link the upstream source
  instead), and the command line
- Impact assessment (crash / memory corruption / information disclosure / etc.)

## Response expectations

Response is best-effort. There is no paid support tier and no SLA.

| Milestone                         | Target                          |
| --------------------------------- | ------------------------------- |
| Acknowledge receipt               | Within 7 days                   |
| Initial triage / severity call    | Within 14 days                  |
| Fix or disclosed workaround       | Depends on severity and scope   |
| Public advisory + release         | Coordinated with the reporter   |

If you have not heard back within 14 days, please ping the email address again;
a single-maintainer project can lose reports in spam filters.

## Scope

**In scope**

- The Vokra runtime crates in this repository (`vokra-core`, `vokra-ops`,
  `vokra-backend-cpu`, `vokra-backend-metal`, `vokra-backend-cuda`,
  `vokra-backend-vulkan`, `vokra-models`, `vokra-capi`, `vokra-cli`,
  `vokra-convert`, `vokra-eval`, `vokra-mmap`, `vokra-vad-micro`,
  `integrations/vokra-server`, `integrations/vokra-godot`, and the C ABI headers)
- Sample offline conversion tools under `tools/`
- GGUF loader, safetensors loader, and the frontend-spec validator
- The Wyoming / OpenAI-compatible / vLLM-compatible HTTP surfaces exposed by
  `vokra-server`

**Out of scope (please report upstream)**

- Upstream model weights (Whisper, Kokoro, piper-plus voice models,
  CosyVoice2, Voxtral, Silero VAD, CAM++, Mimi, DAC, WavTokenizer, etc.) —
  file with the model author or Hugging Face repository owner
- Upstream Rust crates in `Cargo.lock` (none — Vokra maintains a
  zero-dependency invariant, but any transient dev-dep issue should go to
  that crate's own repository)
- Upstream C libraries (CUDA driver, Metal, Vulkan loader) — these are
  loaded via `dlopen` and are the operator's responsibility to keep patched
- Third-party integrations that depend on Vokra but live in other repositories

**Not accepted as vulnerabilities**

- Denial-of-service via unbounded input (memory or CPU exhaustion) when the
  input is a model file supplied by the operator to their own process. Vokra
  loads models the operator asks it to load; the operator is trusted for the
  file's provenance. Untrusted-input scenarios (e.g. `vokra-server` accepting
  audio from arbitrary clients) are in scope; untrusted-model scenarios are
  not.
- Issues that require a physically or administratively privileged attacker
  on the host (root, ptrace, `/dev/mem`, disabled ASLR).
- Reports produced solely by static analysis without a demonstrated impact.

## Coordinated disclosure

We follow **coordinated disclosure**. If you request an embargo, we will hold
public disclosure until either (a) a fix is released, or (b) 90 days have
elapsed from the initial report, whichever comes first — extended by mutual
agreement if the fix is complex. Credit will be given in the advisory unless
you request anonymity.

---

# セキュリティポリシー

Vokra は音声 AI モデル向けの Rust 推論ランタイムで、単独メンテナが Claude
Code の支援を受けて開発・保守している OSS プロジェクトです。本ドキュメントは
脆弱性の報告方法と対応内容を定めます。

## サポート対象バージョン

Vokra は pre-1.0（release-candidate）です。セキュリティ修正は現行 rc ライン
にのみ適用され、旧ラインへの backport は行いません。

| バージョン              | ステータス                | セキュリティ修正           |
| ---------------------- | ------------------------ | -------------------------- |
| `v1.0-rc.*`（現行）     | Active（M4 → M5）        | Best-effort、現行ラインのみ |
| `v1.0` GA（将来）       | 未リリース                | rc ラインを supersede 予定 |
| `v0.9`                  | End-of-life              | 対応なし                    |
| `v0.5` 以前             | End-of-life              | 対応なし                    |

「Best-effort」とは、単独メンテナが SLA コミットなしで対応することを意味しま
す。高重大度（RCE、任意ヒープ書き込み、compliance gate の暗号バイパスなど）
を低重大度より優先します。

## 脆弱性の報告

**セキュリティ関連は GitHub の public issue に書かないでください。** public
issue は数分で検索エンジンにインデックスされ、攻撃者に先手を与えます。

推奨: **GitHub private vulnerability reporting**

- <https://github.com/ayutaz/vokra/security/advisories/new>
- GitHub 標準の coordinated disclosure ワークフローを使用します。詳細は修正
  完了までメンテナと報告者の間で非公開に保たれ、advisory 内から CVE を発行
  できます。

代替: **メール**

- <yousan@aihub.tokyo>
- GitHub advisory を開けない場合（アカウントがない、UI に到達できないなど）
  のみ利用してください。暗号化されていない秘密情報（実運用のクレデンシャル、
  秘密鍵）は含めないでください。センシティブな PoC を送る必要がある場合は先
  に PGP 鍵を要請してください。

可能な範囲で以下を含めてください:

- Vokra のバージョンとコミット SHA（`vokra-cli --version` またはビルド時の
  tag / hash）
- プラットフォーム（OS、アーキテクチャ、backend: `cpu` / `metal` / `cuda` /
  `vulkan` / `webgpu`）
- 再現手順: 最小入力、issue を trigger する GGUF / safetensors ファイル
  （再配布不可なモデル weight は添付せず、上流の URL を提示してください）、
  コマンドライン
- 影響評価（クラッシュ / メモリ破壊 / 情報漏洩 など）

## 対応目安

対応は best-effort です。有償サポート層および SLA はありません。

| マイルストーン                | 目標                                    |
| ---------------------------- | --------------------------------------- |
| 受領確認                     | 7 日以内                                |
| 初期トリアージ / 重大度判定   | 14 日以内                               |
| 修正または開示済みワークアラウンド | 重大度・範囲による                       |
| 公開 advisory + リリース      | 報告者と調整の上                        |

14 日以内に返信がない場合は再度メールしてください。単独メンテナの環境では
スパムフィルタで見落とすことがあります。

## スコープ

**対象**

- 本リポジトリの Vokra ランタイム crate 群（`vokra-core`、`vokra-ops`、
  `vokra-backend-cpu`、`vokra-backend-metal`、`vokra-backend-cuda`、
  `vokra-backend-vulkan`、`vokra-models`、`vokra-capi`、`vokra-cli`、
  `vokra-convert`、`vokra-eval`、`vokra-mmap`、`vokra-vad-micro`、
  `integrations/vokra-server`、`integrations/vokra-godot`、C ABI ヘッダ）
- `tools/` 配下のオフライン変換ツール
- GGUF ローダー、safetensors ローダー、frontend-spec バリデータ
- `vokra-server` が exposeする Wyoming / OpenAI 互換 / vLLM 互換 HTTP 面

**対象外（上流に報告してください）**

- 上流モデルの weight（Whisper、Kokoro、piper-plus voice model、
  CosyVoice2、Voxtral、Silero VAD、CAM++、Mimi、DAC、WavTokenizer 等）—
  モデル作者または Hugging Face リポジトリ管理者へ
- `Cargo.lock` 上の上流 Rust crate（Vokra は zero-dep 不変条件を維持している
  ため通常なし。dev-dep の一時的な問題は当該 crate 側へ）
- 上流の C ライブラリ（CUDA driver、Metal、Vulkan loader）— これらは
  `dlopen` で実行時ロードされる operator 責任範囲です
- Vokra に依存する第三者統合（別リポジトリで管理されるもの）

**脆弱性として受理しないケース**

- Operator 自身が自プロセスに供給するモデルファイルによる DoS（メモリ /
  CPU 枯渇）— Vokra は operator が要求したモデルをロードする設計で、ファイル
  の provenance は operator が信頼する前提です。信頼できない入力（例:
  `vokra-server` が任意クライアントから受け取る音声）は対象、信頼できない
  モデルは対象外。
- 物理的または管理者権限を持つ攻撃者を前提とする問題（root、ptrace、
  `/dev/mem`、ASLR 無効化など）
- 実際の影響を示さない静的解析のみによる指摘

## 協調的開示（coordinated disclosure）

**協調的開示**に従います。embargo が要請された場合、(a) 修正リリース、
または (b) 初回報告から 90 日、のいずれか早い時点まで公開を保留します
（複雑な修正では合意に基づき延長）。匿名を希望されない限り、advisory に
クレジットを記載します。
