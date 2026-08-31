# セキュリティポリシー

[English](SECURITY.md) | **日本語**

## 脆弱性の報告

脆弱性は
[GitHub Private Vulnerability Reporting](https://github.com/ayutaz/vokra/security/advisories/new)
からのみ報告してください。これが本プロジェクト唯一の非公開報告経路です。
Vokra はメールアドレスやプロジェクト固有の別の連絡先を公開しません。

疑わしい脆弱性を public issue に投稿しないでください。安全に共有できる範囲で、
対象 commit または version、platform と backend、影響、最小再現を private
advisory に含めてください。配布制限のあるモデル weight は添付・再配布せず、
入手元へのリンクを示してください。

## 対応と開示

Vokra は SLA なしの best-effort で保守されています。再現性と影響に基づいて
triage し、メモリ破壊、remote code execution、session 間の情報漏洩、release
gate の bypass を優先します。

修正または緩和策を準備する間は報告を非公開に保ってください。公開時期と報告者
credit は advisory 内で調整し、匿名の希望を尊重します。

## サポート対象

workspace は `0.3.0` development で、Git tag と公開済み release はまだ
ありません。セキュリティ修正は現行 development line に適用します。古い commit、
private build、未公開 snapshot への backport は行いません。

## 対象範囲

- 不正な GGUF、safetensors、音声、codec、request data による memory safety
  または validation failure。
- safe Rust API または C API から引き起こせる undefined behavior、不正な
  lifetime、session 間の情報漏洩。
- mmap loader、raw GPU FFI backend、offline converter、repository の release
  / provenance gate。
- `vokra-server` と repository 内 integration の request parsing、isolation、
  resource control に関する脆弱性。

## 対象外

- 上流モデル weight の品質、bias、license、意図された挙動。
- 第三者 driver、OS、hosted service、別 repository で保守される project の
  脆弱性。
- 未対応構成、fork に operator 自身が commit した secret、再現可能な影響を
  示さない scanner 出力。
- 信頼できない入力が強制済み制限を回避する場合や別 session に影響する場合を
  除く、モデル実行固有の性能限界・resource 消費。

## 依存関係について

root runtime に第三者 Cargo dependency はなく、first-party のみの
`Cargo.lock` を CI で強制しています。これは dependency chain の露出を減らし
ますが、Vokra 自身の parser、unsafe boundary、FFI、生成 artefact の review を
代替するものではありません。
