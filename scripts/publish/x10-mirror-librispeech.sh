#!/usr/bin/env bash
# x10-mirror-librispeech.sh — mirror LibriSpeech dev-clean (OpenSLR resource
# 12) to `huggingface.co/datasets/vokra/librispeech-dev-clean`, bit-identical
# to the upstream tarball, with CC-BY-4.0 attribution enforced (WP X-10-T02
# per ADR X-10, `docs/adr/X-10-corpus-self-mirror.md`).
#
# DRY-RUN BY DEFAULT. Mirroring is a one-way, outward-facing action: once a
# corpus is public it can be mirrored within minutes, so "delete it later" is
# not a recovery plan. `--push` must be passed explicitly, every time.
#
# WHAT THIS ENFORCES, AND WHY
#
#   1. **Bit-identical to upstream**: the downloaded tarball must match the
#      pinned SHA256 exactly (default `76f87d09…8ab3`, from
#      `.github/workflows/nightly-asr-wer.yml` DEV_CLEAN_SHA256; override
#      with `--sha256`). This is the ADR X-10 Option A contract: the mirror
#      is a SHA-pinned revision of the upstream file, not a repackage.
#
#   2. **LICENSE co-located**: CC-BY-4.0 canonical text
#      (creativecommons.org/licenses/by/4.0/legalcode.txt) is fetched via
#      `fetch_license.sh` and bundled with the dataset. Attribution is
#      satisfied by mirror co-location, not by asking downstream consumers to
#      find it themselves.
#
#   3. **README attribution**: Panayotov, V., Chen, G., Povey, D., &
#      Khudanpur, S. (2015). "LibriSpeech: An ASR corpus based on public
#      domain audio books." ICASSP 2015 — the primary citation is written
#      into the README (bibliographic entry + BibTeX), so a consumer who
#      only reads the dataset card sees the attribution obligation.
#
#   4. **§3.1 sign-off check**: `docs/license-audit.md` §3.2 row for
#      LibriSpeech dev-clean must carry a non-blank Approval before push
#      (2026-07-30 yousan ☑ Commercial via CC judgment — memory
#      `feedback-license-signoff-primary-source` rule). A blank row means
#      "nobody has decided yet", which is not the same as "no".
#
# Usage:
#   scripts/publish/x10-mirror-librispeech.sh                       # dry-run
#   scripts/publish/x10-mirror-librispeech.sh --push                # publish
#   scripts/publish/x10-mirror-librispeech.sh --repo vokra/librispeech-dev-clean-alt
#   scripts/publish/x10-mirror-librispeech.sh --self-test           # defensive checks
#
# Credentials: HF_TOKEN in the environment. Never passed on the command line.
#
# Rollout: ADR X-10 §Rollout order — this script implements X-10-T02.
# After push, owner sets `vars.VOKRA_CORPUS_LIBRISPEECH_MIRROR_URL` +
# `vars.VOKRA_CORPUS_LIBRISPEECH_MIRROR_SHA256` (X-10-T03), populates
# `.github/pins.yaml` `mirror:` block, and verifies
# `corpus-drift-detector.yml` reports mirror match.

set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$here/../.." && pwd)"
fetch_license="$here/fetch_license.sh"
audit="$repo_root/docs/license-audit.md"

# --- defaults --------------------------------------------------------------
UPSTREAM_URL="https://www.openslr.org/resources/12/dev-clean.tar.gz"
DEFAULT_SHA256="76f87d090650617fca0cac8f88b9416e0ebf80350acb97b343a85fa903728ab3"
DEFAULT_REPO="vokra/librispeech-dev-clean"

url="$UPSTREAM_URL"
sha256="$DEFAULT_SHA256"
repo="$DEFAULT_REPO"
push=0
outdir=""
self_test=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --url) url="$2"; shift 2 ;;
    --sha256) sha256="$2"; shift 2 ;;
    --repo) repo="$2"; shift 2 ;;
    --out) outdir="$2"; shift 2 ;;
    --push) push=1; shift ;;
    --self-test) self_test=1; shift ;;
    -h|--help) sed -n '2,45p' "$0"; exit 0 ;;
    *) echo "x10-mirror-librispeech: unknown arg: $1" >&2; exit 2 ;;
  esac
done

# --- self-test -------------------------------------------------------------
if [[ "$self_test" == "1" ]]; then
  tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
  fail=0
  # (a) --push must never be the default.
  if grep -q '^push=1' <<<"$(sed -n 's/^push=\([0-9]\).*/push=\1/p' "$0" | head -1)"; then
    echo "self-test FAIL: push defaults to on" >&2; fail=1
  fi
  # (b) The pinned SHA256 must equal the nightly workflow's DEV_CLEAN_SHA256
  # default (they are the same upstream file — drift means one moved).
  workflow_sha="$(grep -m1 "DEV_CLEAN_SHA256" "$repo_root/.github/workflows/nightly-asr-wer.yml" | grep -oE '[a-f0-9]{64}' | head -1 || true)"
  if [[ -n "$workflow_sha" && "$workflow_sha" != "$DEFAULT_SHA256" ]]; then
    echo "self-test FAIL: DEFAULT_SHA256 ($DEFAULT_SHA256) drifted from" >&2
    echo "                nightly-asr-wer.yml ($workflow_sha)" >&2
    fail=1
  fi
  # (c) fetch_license.sh must know about cc-by-4.0.
  if ! grep -q 'cc-by-4.0' "$fetch_license"; then
    echo "self-test FAIL: fetch_license.sh does not know cc-by-4.0" >&2; fail=1
  fi
  # (d) The §3.2 sign-off row must exist in docs/license-audit.md.
  if ! grep -q 'LibriSpeech dev-clean' "$audit"; then
    echo "self-test FAIL: §3.2 row for LibriSpeech dev-clean is missing" >&2; fail=1
  fi
  # (e) The upstream URL must be OpenSLR resource 12 (bit-identical).
  if [[ "$UPSTREAM_URL" != *"openslr.org/resources/12/dev-clean.tar.gz" ]]; then
    echo "self-test FAIL: UPSTREAM_URL is not OpenSLR resource 12" >&2; fail=1
  fi
  [[ $fail -eq 0 ]] && echo "x10-mirror-librispeech self-test: OK (5 cases)" && exit 0
  exit 1
fi

# --- §3.1 sign-off gate ----------------------------------------------------
# LibriSpeech dev-clean is CC-BY-4.0 = redistributable + attribution
# required. The §3.2 row was signed 2026-07-30 yousan (☑ Commercial, CC
# judgment). Enforce by grepping the audit — if the row lost its ☑
# Commercial marker (e.g. reverted to blank), refuse.
if ! awk '/LibriSpeech dev-clean/,/^\|[^|]*\|/' "$audit" 2>/dev/null | \
  grep -q "☑ Commercial"; then
  echo "x10-mirror-librispeech: §3.2 sign-off gate failed — LibriSpeech" >&2
  echo "  dev-clean row in $audit does not carry ☑ Commercial approval." >&2
  echo "  Refusing to publish; a blank row is not the same as 'no'." >&2
  exit 3
fi

# --- HF token check --------------------------------------------------------
if [[ "$push" == "1" && -z "${HF_TOKEN:-${HF:-}}" ]]; then
  echo "x10-mirror-librispeech: HF_TOKEN (or HF) must be set for --push" >&2
  echo "  never pass tokens on the command line (they land in shell history" >&2
  echo "  and in ps output). Export the token before running." >&2
  exit 4
fi

# --- staging directory -----------------------------------------------------
: "${outdir:=$repo_root/dist/x10-librispeech-dev-clean}"
mkdir -p "$outdir"
staged="$outdir/staged"
mkdir -p "$staged"

echo "x10-mirror-librispeech: mode=$([[ $push == 1 ]] && echo push || echo dry-run)"
echo "  upstream:  $url"
echo "  sha256:    $sha256"
echo "  repo:      datasets/$repo"
echo "  staging:   $staged"

# --- download + verify -----------------------------------------------------
tarball="$staged/dev-clean.tar.gz"
if [[ -f "$tarball" ]] && (echo "$sha256  $tarball" | sha256sum -c - >/dev/null 2>&1); then
  echo "  download:  cached (bit-identical)"
else
  echo "  download:  fetching $url"
  # NOTE: curl fail-with-body so an HTML error page never masquerades as a
  # tarball; --location to follow OpenSLR mirror redirects.
  curl --fail --location --show-error --silent --output "$tarball" "$url"
  if ! echo "$sha256  $tarball" | sha256sum -c - >/dev/null 2>&1; then
    actual="$(sha256sum "$tarball" | awk '{print $1}')"
    echo "x10-mirror-librispeech: SHA256 mismatch (fail-closed)" >&2
    echo "  expected: $sha256" >&2
    echo "  actual:   $actual" >&2
    echo "  This means upstream drifted, our pin is wrong, or the fetch was" >&2
    echo "  corrupted. Do NOT publish a drifted tarball as 'bit-identical'." >&2
    exit 5
  fi
  echo "  download:  verified sha256=${sha256:0:12}…"
fi

# --- LICENSE canonical text ------------------------------------------------
"$fetch_license" --spdx cc-by-4.0 "$staged/LICENSE"
license_size="$(wc -c < "$staged/LICENSE")"
echo "  LICENSE:   CC-BY-4.0 canonical (${license_size} bytes)"

# --- README with attribution ----------------------------------------------
cat > "$staged/README.md" <<EOF
---
license: cc-by-4.0
task_categories:
- automatic-speech-recognition
language:
- en
tags:
- speech
- librispeech
- audiobook
- asr
- evaluation
size_categories:
- 1K<n<10K
pretty_name: LibriSpeech dev-clean (Vokra mirror)
---

# LibriSpeech dev-clean — Vokra self-mirror

A byte-identical mirror of \`dev-clean.tar.gz\` from OpenSLR resource 12,
maintained for the Vokra project's nightly ASR-WER regression harness
(\`.github/workflows/nightly-asr-wer.yml\`). Byte-for-byte identical to the
upstream tarball at
<https://www.openslr.org/resources/12/dev-clean.tar.gz>.

## Provenance

- **Upstream**: OpenSLR resource 12 (LibriSpeech ASR corpus), dev-clean split
- **URL**: <https://www.openslr.org/resources/12/dev-clean.tar.gz>
- **SHA256**: \`${sha256}\`
- **Bytes**: 337,926,286 (verified — do not repackage)
- **License**: CC-BY-4.0 (see \`LICENSE\` for canonical text)

## Attribution (CC-BY-4.0 requirement)

**Panayotov, V., Chen, G., Povey, D., & Khudanpur, S. (2015). "LibriSpeech:
An ASR corpus based on public domain audio books." In *IEEE International
Conference on Acoustics, Speech and Signal Processing (ICASSP)* (pp.
5206-5210). IEEE.**

BibTeX:

\`\`\`bibtex
@inproceedings{panayotov2015librispeech,
  title={{LibriSpeech}: An {ASR} corpus based on public domain audio books},
  author={Panayotov, Vassil and Chen, Guoguo and Povey, Daniel and Khudanpur, Sanjeev},
  booktitle={2015 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={5206--5210},
  year={2015},
  organization={IEEE}
}
\`\`\`

## Why a Vokra mirror?

The nightly ASR-WER leg previously fetched dev-clean directly from
\`openslr.org\` on every run. On CDN outage or upstream repackaging the leg
would silently drift, surfacing as "Vokra regression" in kill-switch review
noise despite being external. This mirror gives Vokra byte-identity control
while remaining transparently one-way: the upstream URL is authoritative,
this mirror is a pinned copy.

Per ADR X-10 (\`docs/adr/X-10-corpus-self-mirror.md\`) — Option A (HF Hub
dataset mirror) was chosen for provenance-stamping reuse and no-cost
attribution co-location.

## Usage

\`\`\`python
from huggingface_hub import hf_hub_download

tarball = hf_hub_download(
    repo_id="${repo}",
    filename="dev-clean.tar.gz",
    repo_type="dataset",
)
# tarball is bit-identical to https://www.openslr.org/resources/12/dev-clean.tar.gz
\`\`\`

To verify byte-identity against upstream:

\`\`\`bash
sha256sum dev-clean.tar.gz
# should print: ${sha256}
\`\`\`

## Vokra project context

- Consumer workflow: \`.github/workflows/nightly-asr-wer.yml\` (Whisper WER
  regression, 8-utterance slice: chapter 128104 speaker 1272)
- ADR: [\`docs/adr/X-10-corpus-self-mirror.md\`](https://github.com/ayutaz/vokra/blob/main/docs/adr/X-10-corpus-self-mirror.md)
- License audit: \`docs/license-audit.md\` §3.2 row (2026-07-30 yousan sign-off,
  ☑ Commercial CC judgment based on OpenSLR primary source)

## Non-goals

- **Not a repackage** — bit-identical to upstream, always. If you want a
  parsed / decoded / re-cut form, use a different dataset.
- **Not authoritative** — the authoritative source remains
  \`openslr.org/resources/12\`. This mirror exists for CI-time robustness.
- **Not extended** — dev-other / test-clean / test-other / train-* splits
  are NOT mirrored here. Each split is a separate ~300 MB - 30 GB tarball
  and this repo covers only what the Vokra nightly consumes.
EOF
echo "  README:    generated (attribution written)"

# --- self-test the staged bundle before push ------------------------------
# Even in dry-run mode we verify the assembled bundle would satisfy the
# gates. This is the last check before the network action.
if [[ ! -s "$staged/dev-clean.tar.gz" ]]; then
  echo "x10-mirror-librispeech: staged tarball is empty (fail-closed)" >&2
  exit 6
fi
if [[ ! -s "$staged/LICENSE" ]]; then
  echo "x10-mirror-librispeech: staged LICENSE is empty (fail-closed)" >&2
  exit 6
fi
if [[ ! -s "$staged/README.md" ]]; then
  echo "x10-mirror-librispeech: staged README is empty (fail-closed)" >&2
  exit 6
fi
if ! grep -q "Panayotov" "$staged/README.md"; then
  echo "x10-mirror-librispeech: staged README missing attribution (fail-closed)" >&2
  exit 6
fi

# --- dry-run summary ------------------------------------------------------
if [[ "$push" != "1" ]]; then
  echo ""
  echo "x10-mirror-librispeech: DRY-RUN complete. To publish, re-run with --push."
  echo "  staged bundle: $staged/"
  echo "    - dev-clean.tar.gz    ($(du -h "$staged/dev-clean.tar.gz" | awk '{print $1}'))"
  echo "    - LICENSE             ($(wc -c < "$staged/LICENSE") bytes)"
  echo "    - README.md           ($(wc -l < "$staged/README.md") lines, attribution present)"
  echo ""
  echo "  Next: ADR X-10 §Rollout order —"
  echo "    (X-10-T03) set vars.VOKRA_CORPUS_LIBRISPEECH_MIRROR_URL +"
  echo "               vars.VOKRA_CORPUS_LIBRISPEECH_MIRROR_SHA256"
  echo "    (X-10-T04/T05) landed already in commit B of"
  echo "                   fix/nightly-asr-wer-2026-07-23"
  exit 0
fi

# --- push via huggingface_hub ---------------------------------------------
# HfApi().create_repo(repo_type="dataset") + upload_folder — mirroring the
# upload.sh pattern (avoids the hf CLI binary dependency).
python3 - "$staged" "$repo" "${HF_TOKEN:-${HF:-}}" <<'PY'
import os, sys
staged, repo, token = sys.argv[1], sys.argv[2], sys.argv[3]

if not token:
    print("x10-mirror-librispeech: HF_TOKEN empty at push time", file=sys.stderr)
    sys.exit(4)

try:
    from huggingface_hub import HfApi
except ImportError:
    print("x10-mirror-librispeech: huggingface_hub not installed", file=sys.stderr)
    print("  run: pip install huggingface_hub", file=sys.stderr)
    sys.exit(7)

api = HfApi(token=token)

# Create dataset repo (idempotent).
api.create_repo(
    repo_id=repo,
    repo_type="dataset",
    exist_ok=True,
    private=False,
)
print(f"  create_repo: datasets/{repo} (idempotent)")

# Upload folder. huggingface_hub handles large-file LFS routing.
api.upload_folder(
    folder_path=staged,
    repo_id=repo,
    repo_type="dataset",
    commit_message="mirror LibriSpeech dev-clean (WP X-10-T02, ADR X-10)",
)
print(f"  upload_folder: datasets/{repo} = live")
print(f"  visit: https://huggingface.co/datasets/{repo}")
PY

echo ""
echo "x10-mirror-librispeech: PUSHED to https://huggingface.co/datasets/${repo}"
echo "  Next (X-10-T03): set repo variables + populate pins.yaml mirror row."
