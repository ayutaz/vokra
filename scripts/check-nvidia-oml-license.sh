#!/usr/bin/env bash
# Deterministic compliance regression for NVIDIA OML (issue #53).

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
license_file="$repo_root/third_party/nvidia-open-model-license-agreement-june-2024.txt"
license_hash_file="$repo_root/third_party/nvidia-open-model-license-agreement-june-2024.sha256"
notice_file="$repo_root/NOTICE"
audit_file="$repo_root/docs/license-audit.md"
source_sha="396d7de220f6b0e6fcfe33836f1b99a9943769b97c56330444f8efe1b441f607"
required_notice="Licensed by NVIDIA Corporation under the NVIDIA Open Model License"

[[ -f "$license_file" ]] || {
  echo "NVIDIA OML check: missing vendored June-2024 agreement: $license_file" >&2
  exit 1
}
[[ -f "$license_hash_file" ]] || {
  echo "NVIDIA OML check: missing vendored transcription hash: $license_hash_file" >&2
  exit 1
}
(cd "$(dirname "$license_file")" && shasum -a 256 -c "$(basename "$license_hash_file")") >/dev/null || {
  echo "NVIDIA OML check: vendored agreement SHA-256 mismatch" >&2
  exit 1
}

grep -Fq "Version Release Date: June 14, 2024" "$license_file" || {
  echo "NVIDIA OML check: vendored agreement does not pin the June-2024 revision" >&2
  exit 1
}
grep -Fq "Source PDF SHA-256: $source_sha" "$license_file" || {
  echo "NVIDIA OML check: source PDF hash is missing or changed" >&2
  exit 1
}
grep -Fq "$required_notice" "$license_file" || {
  echo "NVIDIA OML check: vendored agreement lost the required NOTICE sentence" >&2
  exit 1
}
grep -Fq "$required_notice" "$notice_file" || {
  echo "NVIDIA OML check: project NOTICE does not emit NVIDIA's required sentence" >&2
  exit 1
}
grep -Fq "nvidia-open-model-license" "$audit_file" || {
  echo "NVIDIA OML check: docs/license-audit.md lacks the HF license_name" >&2
  exit 1
}
grep -Fq "October 24, 2025" "$audit_file" || {
  echo "NVIDIA OML check: audit does not distinguish the later 2025 revision" >&2
  exit 1
}

echo "NVIDIA OML check: PASS (June-2024 revision, source hash, NOTICE, audit)"
