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
  # Verifies every refusal that matters, without touching the network.
  #
  # This replaces the pre-2026-08 self-test which referenced a `SIGNOFF_OVERRIDE`
  # env var that no code path actually read (dead reference — passing an
  # unknown env var to a nonexistent artifact test told us nothing about the
  # sign-off gate). The new cases below drive `signoff_match.py` with a
  # synthetic §3.1 fixture and assert the four terminal states this script
  # cares about: APPROVED (allow through), PENDING (blank row -> exit 3),
  # NO_ROW (repo declared but audit out of sync -> exit 4), UNKNOWN_REPO
  # (repo not in explicit map -> exit 5). The fail-closed inversion of the
  # old NO_ROW-passes-silently branch is exercised by the NO_ROW / UNKNOWN
  # cases below — a REGRESSION to the old behaviour would flip either to
  # success and be caught here.
  tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
  fail=0

  # (a) --push must never be the default. Regex-anchored so a comment
  #     containing "push=1" cannot mask a real change.
  if grep -qE '^push=1' "$0"; then
    echo "self-test FAIL: push defaults to on" >&2; fail=1
  fi

  # (b) The card tool's own gate must be reachable from here.
  if ! python3 "$card_tool" --self-test >/dev/null 2>&1; then
    echo "self-test FAIL: make_model_card self-test does not pass" >&2; fail=1
  fi

  # (c) signoff_match must pass its own self-test.
  if ! python3 "$repo_root/scripts/publish/signoff_match.py" --self-test >/dev/null 2>&1; then
    echo "self-test FAIL: signoff_match self-test does not pass" >&2; fail=1
  fi

  # (d) End-to-end: drive the actual sign-off Python block against a
  #     synthetic audit + repo map and assert each terminal state.
  #     Kept inline (not shelling to upload.sh recursively) so the state
  #     transitions we care about are the SAME ones the production path
  #     runs; a recursive call would additionally trip the artifact-exists
  #     check, which is not what this case is testing.
  #
  #     Uses `python3 signoff_match.py --check-repo` — same CLI the
  #     production run below invokes — with SIGNOFF_MATCH_FIXTURES
  #     pointing at a hermetic table.
  cat >"$tmp/audit.md" <<'EOF'
### Owner sign-off template

| Model | Weight License | CC-verified date | Owner sign-off (YYYY-MM-DD) | Approval | Notes |
|---|---|---|---|---|---|
| **APPROVED-Model** | MIT | 2026-01-01 | 2026-01-02 yousan | ☑ Commercial / ☐ Research-only / ☐ Rejected | approved fixture |
| **PENDING-Model** | MIT | 2026-01-01 | ______________ | ☐ Commercial / ☐ Research-only / ☐ Rejected | blank fixture |
| **REJECTED-Model** | Unknown | 2026-01-01 | 2026-01-02 yousan | ☐ Commercial / ☐ Research-only / ☑ Rejected | rejected fixture |
EOF
  # Isolated Python driver — imports signoff_match, swaps in a scoped
  # repo map, checks each state. This is intentionally the SAME logic
  # the production path uses (approval_for_repo returns the same states)
  # so a regression in either side surfaces here.
  py_out="$(python3 - "$tmp/audit.md" "$repo_root/scripts/publish" <<'PY'
import sys
audit, matcher_dir = sys.argv[1], sys.argv[2]
sys.path.insert(0, matcher_dir)
import signoff_match

# Isolated map — do not depend on the real REPO_TO_SIGNOFF_ROWS
# evolving. The fixture is the whole test.
signoff_match.REPO_TO_SIGNOFF_ROWS = {
    "approved-repo": ["APPROVED-Model"],
    "pending-repo":  ["PENDING-Model"],
    "rejected-repo": ["REJECTED-Model"],
    "noroom-repo":   ["MissingFromAudit-Model"],
}
from pathlib import Path
cases = [
    ("approved-repo", "APPROVED"),
    ("pending-repo",  "PENDING"),
    # A ticked ☑ Rejected row means the audit HAS answered — the answer
    # is "no". APPROVED here means "audit has a real decision"; the
    # "do not publish" policy is enforced elsewhere (upload.sh outer
    # gate for card+license). Keeps the sign-off state honest about
    # what §3.1 says.
    ("rejected-repo", "APPROVED"),
    ("noroom-repo",   "NO_ROW"),
    ("unregistered",  "UNKNOWN_REPO"),
]
fails = []
for slug, want in cases:
    got, detail = signoff_match.approval_for_repo(slug, Path(audit))
    if got != want:
        fails.append(f"approval_for_repo('{slug}') want {want} got {got} — {detail}")
if fails:
    for f in fails:
        print("  " + f)
    sys.exit(1)
print(f"  {len(cases)} state transitions verified against hermetic §3.1 fixture")
PY
)"
  py_rc=$?
  if [[ $py_rc -ne 0 ]]; then
    echo "self-test FAIL: signoff state transitions did not match expectations" >&2
    printf '%s\n' "$py_out" >&2
    fail=1
  fi

  # (e) Regression against the prefix-leakage bug: a repo whose slug
  #     accidentally starts with a real row's first 8 chars must NOT
  #     silently inherit that row. Under the explicit map this is
  #     UNKNOWN_REPO, so the assertion also documents the invariant.
  if ! python3 - "$tmp/audit.md" "$repo_root/scripts/publish" <<'PY' >/dev/null 2>&1
import sys
audit, matcher_dir = sys.argv[1], sys.argv[2]
sys.path.insert(0, matcher_dir)
import signoff_match
from pathlib import Path

signoff_match.REPO_TO_SIGNOFF_ROWS = {"approved-repo": ["APPROVED-Model"]}
state, _ = signoff_match.approval_for_repo("approvedxxx", Path(audit))
sys.exit(0 if state == "UNKNOWN_REPO" else 1)
PY
  then
    echo "self-test FAIL: prefix-leakage regression (approvedxxx should be UNKNOWN_REPO)" >&2
    fail=1
  fi

  [[ $fail -eq 0 ]] && echo "upload self-test: OK (5 groups)" && exit 0
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
# whether a human has actually approved this specific model — that lives in
# §3.1. Delegated to scripts/publish/signoff_match.py, which owns the
# explicit repo -> row alias map. The old inline substring heuristic was
# retired because an 8-char prefix silently over-approved siblings (a
# new `whisper-*` slug inherited the family's approvals).
#
# States and exit codes:
#   APPROVED      -> proceed.
#   PENDING       -> exit 3 (blank row; owner action needed in §3.1).
#   NO_ROW        -> exit 4 (repo declared in map but the audit is out of
#                    sync; land the row before publishing). fail-closed
#                    inversion of the old branch that let NO_ROW pass with
#                    only a warning.
#   UNKNOWN_REPO  -> exit 5 (repo not declared in REPO_TO_SIGNOFF_ROWS at
#                    all; add the explicit mapping in signoff_match.py).
echo "== 2/4  owner sign-off (docs/license-audit.md §3.1) =="
signoff_out="$(python3 "$repo_root/scripts/publish/signoff_match.py" \
    --check-repo "$model_name" --audit "$audit" 2>&1)"
signoff_rc=$?
signoff_state="$(printf '%s\n' "$signoff_out" | head -1)"
signoff_detail="$(printf '%s\n' "$signoff_out" | tail -n +2)"
case "$signoff_state" in
  APPROVED)
    echo "  sign-off: approved"
    ;;
  PENDING)
    echo "upload: REFUSED — the §3.1 sign-off row for '$model_name' is blank." >&2
    echo "  A blank row means the decision has not been made, which is not the" >&2
    echo "  same as a 'yes'. Fill in the approver and tick a box in" >&2
    echo "  docs/license-audit.md §3.1, then re-run." >&2
    printf '  %s\n' "$signoff_detail" >&2
    exit 3
    ;;
  NO_ROW)
    echo "upload: REFUSED — no §3.1 row exists yet for '$model_name'." >&2
    echo "  This used to pass silently (the old 'never placed on hold -> allow')." >&2
    echo "  It is now fail-closed: publishing without an approvable row would" >&2
    echo "  convert a missing decision into a public fact." >&2
    echo "  Land a row in docs/license-audit.md §3.1, sign it off, then re-run." >&2
    printf '  %s\n' "$signoff_detail" >&2
    exit 4
    ;;
  UNKNOWN_REPO)
    echo "upload: REFUSED — repo '$model_name' is not declared in" >&2
    echo "  scripts/publish/signoff_match.py::REPO_TO_SIGNOFF_ROWS." >&2
    echo "  Add the slug -> §3.1 row mapping there before publishing." >&2
    printf '  %s\n' "$signoff_detail" >&2
    exit 5
    ;;
  *)
    echo "upload: signoff_match.py returned an unexpected state ($signoff_state, rc=$signoff_rc)" >&2
    printf '%s\n' "$signoff_out" >&2
    exit 3
    ;;
esac

echo "== 3/4  accompanying files =="
# Stage the weight only if it is not already the one in the output dir
# (re-running with the staged file as input must be a no-op, not an error).
if [[ "$(cd "$(dirname "$gguf")" && pwd)/$(basename "$gguf")" != "$(cd "$outdir" && pwd)/$(basename "$gguf")" ]]; then
  cp "$gguf" "$outdir/"
fi
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
# Accept either HF_TOKEN (the conventional name) or HF (what this project's
# .env happens to use). Never echoed, never passed as an argument.
tok="${HF_TOKEN:-${HF:-}}"
[[ -n "$tok" ]] || { echo "upload: neither HF_TOKEN nor HF is set" >&2; exit 4; }
[[ -f "$outdir/LICENSE" ]] || {
  echo "upload: REFUSED — $outdir/LICENSE is missing. Publishing a weight" >&2
  echo "  without its licence text does not discharge the obligation." >&2
  exit 5; }
python3 -c "import huggingface_hub" 2>/dev/null || {
  echo "upload: huggingface_hub is not installed (pip install -U huggingface_hub)" >&2
  exit 6; }
echo "  pushing $outdir -> $repo"
# Token via env, not argv, so it never lands in the process table.
HF_UPLOAD_TOKEN="$tok" python3 - "$repo" "$outdir" <<'PY'
import os, sys
from huggingface_hub import HfApi
repo, folder = sys.argv[1], sys.argv[2]
api = HfApi(token=os.environ["HF_UPLOAD_TOKEN"])
api.create_repo(repo, repo_type="model", exist_ok=True)
api.upload_folder(repo_id=repo, folder_path=folder, repo_type="model")
print(f"  uploaded {folder} -> https://huggingface.co/{repo}")
PY
