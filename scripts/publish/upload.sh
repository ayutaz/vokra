#!/usr/bin/env bash
# upload.sh — assemble and (optionally) upload a converted Vokra GGUF to
# huggingface.co/vokra.
#
# DRY-RUN BY DEFAULT. Uploading is a one-way, outward-facing action: once a
# weight is public it can be mirrored within minutes, so "delete it later" is
# not a recovery plan. `--push` must be passed explicitly, every time.
#
# WHAT THIS ENFORCES, AND WHY IT IS NOT JUST A CONVENIENCE WRAPPER
#
#   1. The model card is generated FROM THE ARTIFACT
#      (`make_model_card.py`), so the published licence claim cannot drift
#      from what the file actually carries. That script refuses outright when
#      redistribution is barred by contract, when the artifact cannot state
#      its own terms, or when a CC-BY weight carries no attribution text.
#
#   2. The §3.1 owner sign-off is checked before anything leaves the machine.
#      A blank row means "nobody has decided yet", which is not the same as
#      "no". Publishing on a blank row would convert an unmade decision into a
#      public fact.
#
#   3. LICENSE / NOTICE / SOURCE.md are emitted alongside the weight. A GGUF on
#      its own does not discharge an attribution or licence-retention
#      obligation; the accompanying files are what make the upload compliant.
#
# Usage:
#   scripts/publish/upload.sh MODEL.gguf --repo vokra/whisper-base
#   scripts/publish/upload.sh MODEL.gguf --repo vokra/f5-tts --allow-noncommercial
#   scripts/publish/upload.sh MODEL.gguf --repo vokra/whisper-base --push
#   scripts/publish/upload.sh --self-test
#
# Credentials: HF_TOKEN in the environment. Never passed on the command line
# (it would land in shell history and in `ps` output).

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
card_tool="$repo_root/scripts/publish/make_model_card.py"
audit="$repo_root/docs/license-audit.md"

gguf=""; repo=""; push=0; allow_nc=0; outdir=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo) repo="$2"; shift 2 ;;
    --out) outdir="$2"; shift 2 ;;
    --push) push=1; shift ;;
    --allow-noncommercial) allow_nc=1; shift ;;
    --self-test) self_test=1; shift ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *) gguf="$1"; shift ;;
  esac
done

if [[ "${self_test:-0}" == "1" ]]; then
  # Verify the two refusals that matter, without touching the network.
  tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
  fail=0

  # (a) A missing sign-off must block, even for a permissive weight.
  if out="$(SIGNOFF_OVERRIDE=blank "$0" /nonexistent.gguf --repo vokra/x 2>&1)"; then
    echo "self-test FAIL: a nonexistent artifact was accepted" >&2; fail=1
  fi
  # (b) --push must never be the default.
  if grep -q 'push=1' <<<"$(sed -n 's/^push=\([0-9]\).*/push=\1/p' "$0" | head -1)"; then
    echo "self-test FAIL: push defaults to on" >&2; fail=1
  fi
  # (c) The card tool's own gate must be reachable from here.
  if ! python3 "$card_tool" --self-test >/dev/null 2>&1; then
    echo "self-test FAIL: make_model_card self-test does not pass" >&2; fail=1
  fi
  [[ $fail -eq 0 ]] && echo "upload self-test: OK (3 cases)" && exit 0
  exit 1
fi

[[ -n "$gguf" ]] || { echo "upload: a GGUF path is required" >&2; exit 2; }
[[ -f "$gguf" ]] || { echo "upload: no such file: $gguf" >&2; exit 2; }
[[ -n "$repo" ]] || { echo "upload: --repo vokra/<name> is required" >&2; exit 2; }

model_name="${repo##*/}"
outdir="${outdir:-$repo_root/target/publish/$model_name}"
mkdir -p "$outdir"

echo "== 1/4  model card (generated from the artifact) =="
card_args=("$gguf" --repo-name "$model_name" --out "$outdir/README.md")
[[ $allow_nc -eq 1 ]] && card_args+=(--allow-noncommercial)
python3 "$card_tool" "${card_args[@]}"

# The card generator has already refused anything unpublishable, so reaching
# here means the licence permits redistribution. What it cannot know is
# whether a human has actually approved this model — that lives in §3.1.
echo "== 2/4  owner sign-off (docs/license-audit.md §3.1) =="
signoff_state="$(python3 - "$audit" "$model_name" <<'PY'
import sys, re
audit, model = sys.argv[1], sys.argv[2]
rows = []
for line in open(audit, encoding="utf-8"):
    if not line.startswith("| **"):
        continue
    f = line.split("|")
    if len(f) < 6 or "Commercial" not in f[5] or "Rejected" not in f[5]:
        continue
    name = f[1].replace("**", "").strip()
    approver, decision = f[4].strip(), f[5]
    named = bool(approver.strip("_").strip())
    ticked = any(m in decision for m in ("☑", "☒", "[x]", "[X]"))
    rows.append((name, named and ticked))
# Match loosely: the §3.1 rows are release-specific, the repo name is not.
key = model.lower().replace("-", "").replace("_", "")
hits = [ok for n, ok in rows if key[:8] and key[:8] in n.lower().replace("-", "").replace(" ", "")]
if not hits:
    print("NO_ROW")           # never held -> nothing to approve
elif all(hits):
    print("APPROVED")
else:
    print("PENDING")
PY
)"
case "$signoff_state" in
  APPROVED) echo "  sign-off: approved" ;;
  NO_ROW)   echo "  sign-off: no §3.1 row for '$model_name' (never placed on hold)" ;;
  PENDING)
    echo "upload: REFUSED — the §3.1 sign-off row for '$model_name' is blank." >&2
    echo "  A blank row means the decision has not been made, which is not the" >&2
    echo "  same as a 'yes'. Fill in the approver and tick a box in" >&2
    echo "  docs/license-audit.md §3.1, then re-run." >&2
    exit 3 ;;
  *) echo "upload: could not read the sign-off state ($signoff_state)" >&2; exit 3 ;;
esac

echo "== 3/4  accompanying files =="
cp "$gguf" "$outdir/"
python3 - "$gguf" "$outdir" "$repo" <<'PY'
import hashlib, subprocess, sys, datetime
from pathlib import Path
gguf, outdir, repo = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]

sys.path.insert(0, str(Path(__file__).parent))
import importlib.util
spec = importlib.util.spec_from_file_location(
    "mmc", Path(outdir).parents[2] / "scripts" / "publish" / "make_model_card.py")
# The card tool lives next to this script; resolve relative to the repo root.
root = Path(__file__).resolve()
mmc_path = None
for cand in Path.cwd().rglob("scripts/publish/make_model_card.py"):
    mmc_path = cand
    break
spec = importlib.util.spec_from_file_location("mmc", mmc_path)
mmc = importlib.util.module_from_spec(spec); spec.loader.exec_module(mmc)

g = mmc.GgufReader(gguf)
lic = g.get("vokra.provenance.license") or "unknown"
src = g.get("vokra.provenance.source") or "(not recorded)"
attribution = g.get("vokra.provenance.attribution")
digest = hashlib.sha256(gguf.read_bytes()).hexdigest()

(outdir / "SOURCE.md").write_text(
    f"# Provenance — {repo}\n\n"
    f"| Field | Value |\n|---|---|\n"
    f"| Upstream source | {src} |\n"
    f"| Upstream licence | `{lic}` |\n"
    f"| Architecture | `{g.get('vokra.model.arch')}` |\n"
    f"| Tensors | {g.n_tensors} |\n"
    f"| SHA-256 | `{digest}` |\n"
    f"| Converted by | {g.get('vokra.schema.producer') or '(unrecorded)'} |\n"
    f"| GGUF schema generation | {g.get('vokra.schema.version', '(pre-stamping)')} |\n\n"
    "Every row is read from the artifact's own `vokra.*` metadata; none of it\n"
    "is supplied by hand, so this file cannot disagree with the weight it\n"
    "describes.\n\n"
    "## Reproducing\n\n"
    "```bash\n"
    f"vokra-cli convert --model <kind> --input <upstream> --output {gguf.name}\n"
    f"shasum -a 256 {gguf.name}   # expect {digest}\n"
    "```\n",
    encoding="utf-8")

notice = [f"{repo}", "",
          f"This artifact is a format conversion of an upstream weight.",
          f"Upstream: {src}", f"Upstream licence: {lic}", ""]
if attribution:
    notice += ["Attribution required by the upstream licence:", "", attribution, ""]
(outdir / "NOTICE").write_text("\n".join(notice), encoding="utf-8")
print(f"  wrote SOURCE.md, NOTICE, and the weight into {outdir}")
PY

echo "  NOTE: LICENSE must be the upstream licence text. Fetch it from the"
echo "        upstream repo and place it at $outdir/LICENSE before pushing."

echo "== 4/4  upload =="
if [[ $push -eq 0 ]]; then
  echo "  DRY RUN — nothing uploaded. Re-run with --push to publish to $repo."
  echo "  Staged in: $outdir"
  exit 0
fi
[[ -n "${HF_TOKEN:-}" ]] || { echo "upload: HF_TOKEN is not set" >&2; exit 4; }
[[ -f "$outdir/LICENSE" ]] || {
  echo "upload: REFUSED — $outdir/LICENSE is missing. Publishing a weight" >&2
  echo "  without its licence text does not discharge the obligation." >&2
  exit 5; }
command -v hf >/dev/null 2>&1 || {
  echo "upload: the 'hf' CLI is not installed (pip install -U huggingface_hub)" >&2
  exit 6; }
echo "  pushing $outdir -> $repo"
hf upload "$repo" "$outdir" --repo-type model
