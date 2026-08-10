#!/usr/bin/env bash
# Claude Code PreToolUse hook (Bash): block Python package management commands
# that bypass uv, which the project requires. Owner directive 2026-07-27
# (memory feedback-python-uses-uv): Python は uv で管理する。pip / conda /
# requirements.txt / python -m pip は使わない。pyproject.toml + uv.lock を
# per-tree に置く。
#
# WHY THIS EXISTS
#
# `tools/parity/` and other per-tree Python sub-projects are uv-managed
# (pyproject.toml + uv.lock, Python 3.12 pin). Ad-hoc `pip install` bypasses
# the lockfile, breaks reproducibility, and diverges local vs vast.ai env.
# The pre-commit hook already routes internal python calls through
# `uv run --project tools/parity python ...`, but that only catches
# committed scripts — not one-off Bash calls Claude may run.
#
# Blocked forms:
#   * pip install
#   * pip freeze
#   * pip uninstall
#   * pip3 install / pip3 freeze / pip3 uninstall
#   * python -m pip <anything>
#   * python3 -m pip <anything>
#   * conda install / conda create / conda env create
#
# NOT blocked (legitimate uses):
#   * uv add / uv run / uv sync / uv pip (uv-managed pip pass-through)
#   * python3 <script>.py         (running an existing script)
#   * python3 -c '...'            (inline eval)
#
# Vast.ai instance provisioning uses `pip install 'huggingface_hub<0.30'` via
# scripts/publish/vast-ai/provision.sh (documented gotcha B). That script
# runs on remote instance, not through Claude Code — this hook is Claude-only.
#
# Exit 2 = block the tool call and return the message to Claude.

set -uo pipefail

payload="$(cat)"

cmd=""
if command -v jq >/dev/null 2>&1; then
    cmd="$(printf '%s' "$payload" | jq -r '.tool_input.command // empty' 2>/dev/null || true)"
elif command -v python3 >/dev/null 2>&1; then
    cmd="$(printf '%s' "$payload" \
        | python3 -c 'import sys,json; print(json.load(sys.stdin).get("tool_input",{}).get("command",""))' \
        2>/dev/null || true)"
fi

# Empty command? Let it through — no cargo-add-style spurious blocks.
[ -n "$cmd" ] || exit 0

# `uv pip` is legitimate (uv-managed pip surface). Peel it off before the
# bare `pip` match below so a legitimate `uv pip install X` is not blocked.
# Strip leading whitespace then check for uv pip prefix explicitly.
stripped_cmd="$(printf '%s' "$cmd" | sed -E 's/^[[:space:]]+//')"
case "$stripped_cmd" in
    "uv pip "*|"uv pip"*) exit 0 ;;
esac

# Match forms as command words (line start, or after ; & | or whitespace),
# so a path or a comment mention of the string does not trip it.
#
# Pattern A: bare pip / pip3 with mutation-verb (install/freeze/uninstall).
#   pip install X          → BLOCK
#   pip3 uninstall X       → BLOCK
#   pip list               → allowed (read-only introspection)
#   ./scripts/pip-helper.sh → NOT matched (not a command word)
#
# Pattern B: python -m pip / python3 -m pip (any sub-command).
#   python -m pip install X    → BLOCK
#   python3 -m pip freeze      → BLOCK
#
# Pattern C: conda (mutation-only).
#   conda install X        → BLOCK
#   conda create -n env    → BLOCK
#   conda env create -f X  → BLOCK
#   conda info             → allowed (read-only)
matched=""
if printf '%s' "$cmd" | grep -Eq '(^|[;&|[:space:]])pip3?[[:space:]]+(install|freeze|uninstall)([[:space:]]|$)'; then
    matched="pip install/freeze/uninstall"
elif printf '%s' "$cmd" | grep -Eq '(^|[;&|[:space:]])python3?[[:space:]]+-m[[:space:]]+pip([[:space:]]|$)'; then
    matched="python -m pip"
elif printf '%s' "$cmd" | grep -Eq '(^|[;&|[:space:]])conda[[:space:]]+(install|create|env[[:space:]]+create|env[[:space:]]+update|remove|uninstall)([[:space:]]|$)'; then
    matched="conda install/create/remove"
fi

if [ -n "$matched" ]; then
    {
        echo "Blocked: '$matched' bypasses uv. Vokra manages Python with uv"
        echo "(owner directive 2026-07-27, memory feedback-python-uses-uv):"
        echo "  * uv add <pkg>         instead of pip install"
        echo "  * uv sync              instead of pip install -r requirements.txt"
        echo "  * uv run <script>      instead of python script.py (uses .venv)"
        echo "  * uv pip <subcmd>      IS allowed (uv-managed pip surface)"
        echo ""
        echo "Per-tree layout: pyproject.toml + uv.lock, Python 3.12 pin"
        echo "(uv python pin 3.12 + requires-python \">=3.12\")."
        echo ""
        echo "Bypass this hook only when the user explicitly says so — e.g.,"
        echo "when reproducing a vast.ai provisioning step that must run pip"
        echo "on the remote instance (scripts/publish/vast-ai/provision.sh)."
    } >&2
    exit 2
fi
exit 0
