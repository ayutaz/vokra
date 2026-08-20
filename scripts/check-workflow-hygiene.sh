#!/usr/bin/env bash
# check-workflow-hygiene.sh — static health checks over .github/workflows/*.yml
# (X-08-T27; NFR-MT-07).
#
# WHY THIS EXISTS AS A SCRIPT
# ---------------------------
# The only cron-collision check in the tree today is a unittest method on ONE
# workflow's test class (`TestWorkflowSurface` in
# tools/eval/test_librispeech_wer.py), which has two structural defects:
#
#   1. Its regex is `-\s+cron:\s*"([^"]+)"` — DOUBLE quotes only. Five of the
#      thirteen crons in the tree are single-quoted (godot-crossbuild,
#      gpu-cuda-rtf, gpu-vulkan-parity, parity-kokoro-real, parity-rvq-real),
#      so it sees 8 of 13 and cannot see the other 5 at all.
#   2. It only compares *its own* workflow against the others, so a collision
#      between two workflows that are both "other" is structurally invisible.
#
# This script sees every cron in every workflow, regardless of quoting, and
# compares all pairs. It is a REGRESSION GUARD, not a fix: the current tree has
# no collisions (verified by running it).
#
# CHECKS
#   (a) cron collision — two workflows firing at the same (minute, hour) with
#       intersecting day-of-week sets. Staggering matters because concurrent
#       heavy workflows contend for the same runner pool.
#   (b) `needs:` integrity — every job id referenced by a `needs:` must be a
#       real job id in the SAME workflow file. A typo here makes the dependent
#       job silently never run (GitHub reports the workflow as invalid, but
#       only once someone triggers it — which for cron/dispatch-only workflows
#       can be weeks later).
#   (c) `run:` block shell syntax — `bash -n` over every run body whose shell
#       is bash/sh. Catches unbalanced quotes / `fi` / `done` in workflows that
#       nothing triggers on a PR.
#   (d) unquotable plain scalars — a single-line `key: value` whose UNQUOTED
#       value contains ": " is a YAML parse error, and GitHub rejects the whole
#       workflow file for it. This one is not hypothetical: it fired on
#       `- name: Workflow hygiene (cron collisions / needs: ids / ...)` while
#       this very script was being wired in. Empirically confirmed rule —
#       quoted values are fine, `colon:no-space` is fine, URLs are fine, and
#       block scalars (`|`) are fine because their body is not a key-line
#       scalar.
#   (e) uv-only Python — shell commands in every `run:` step must invoke
#       Python, pip, and pytest through uv. Manual `python -m venv`, venv
#       activation, direct `<venv>/bin/python`, and bare pip/pytest are hard
#       errors. Heredoc payloads are skipped after checking their launcher, so
#       Python source code is never mistaken for a shell command.
#   (f) setup-uv integrity — every job that executes uv must install the
#       repository-pinned setup-uv action in that same job. Unpinned actions,
#       duplicate setup steps, and setup steps in jobs that never invoke uv
#       are hard errors; this catches copy/paste placement mistakes.
#   (g) immutable CI dependencies — remote `uses:` entries require a full
#       40-hex commit SHA; `docker run` / `docker pull` and workflow container
#       images require an OCI `@sha256:<64-hex>` digest. Mutable tags such as
#       `@v5` and `:latest` are rejected.
#
# NON-GOALS
#   Not a YAML validator. Zero-dep (NFR-DS-02) forbids PyYAML, so this is a
#   line-oriented parser over the narrow subset of YAML shapes the tree
#   actually uses. Anything it cannot parse confidently is reported as a HARD
#   ERROR rather than skipped (FR-EX-08: never silently pass what you did not
#   check).
#
# Usage: bash scripts/check-workflow-hygiene.sh [--list | --self-test | --help]
#   --list       print the parsed cron / job / needs inventory and exit 0
#   --self-test  unit-test the parser against synthetic scratch trees
#
# Exit code: 0 = clean (or --list / --self-test / --help success), 1 = a
# hygiene violation, 2 = usage / parse error.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

MODE="${1:-check}"
case "$MODE" in
    check | --list | --self-test) ;;
    --help | -h)
        sed -n '2,50p' "$0"
        exit 0
        ;;
    *)
        echo "usage: $0 [--list | --self-test | --help]" >&2
        exit 2
        ;;
esac

# The checker body lives in Python (stdlib only — no PyYAML, NFR-DS-02) and
# is always launched through uv per the repository Python-tooling policy.
# `--self-test` re-invokes it against synthetic trees built in a mktemp dir.
run_checker() {
    # $1 = directory containing the workflow *.yml files
    # $2 = mode ("check" or "--list")
    uv run --no-project --python 3.12 python - "$1" "$2" <<'PY'
import pathlib
import re
import subprocess
import sys
import tempfile

wf_dir = pathlib.Path(sys.argv[1])
mode = sys.argv[2] if len(sys.argv) > 2 else "check"

problems = []


def expand_field(spec, lo, hi, where):
    """Expand one cron field into a concrete int set.

    Handles `*`, `a`, `a-b`, `a,b,c`, and `*/n` / `a-b/n`. Anything else is a
    hard error: a gate that silently ignores a cron form it cannot read is a
    gate that reports a fabricated pass.
    """
    out = set()
    for part in spec.split(","):
        step = 1
        if "/" in part:
            part, _, step_s = part.partition("/")
            if not step_s.isdigit() or int(step_s) == 0:
                problems.append(f"{where}: unparsable cron step '{step_s}'")
                return None
            step = int(step_s)
        if part == "*":
            lo_i, hi_i = lo, hi
        elif "-" in part.lstrip("-"):
            a, _, b = part.partition("-")
            if not (a.isdigit() and b.isdigit()):
                problems.append(f"{where}: unparsable cron range '{part}'")
                return None
            lo_i, hi_i = int(a), int(b)
        elif part.isdigit():
            lo_i = hi_i = int(part)
        else:
            problems.append(f"{where}: unparsable cron field '{part}'")
            return None
        if lo_i < lo or hi_i > hi or lo_i > hi_i:
            problems.append(f"{where}: cron field out of range '{part}'")
            return None
        out.update(range(lo_i, hi_i + 1, step))
    return out


def parse_cron(expr, where):
    """Return (minutes, hours, dows) as int sets, or None on a parse failure."""
    fields = expr.split()
    if len(fields) != 5:
        problems.append(f"{where}: cron '{expr}' does not have 5 fields")
        return None
    minute = expand_field(fields[0], 0, 59, where)
    hour = expand_field(fields[1], 0, 23, where)
    dow = expand_field(fields[4].replace("7", "0"), 0, 6, where)
    if minute is None or hour is None or dow is None:
        return None
    return (minute, hour, dow)


# `- cron: <expr>` in any of the three quoting styles the tree uses (double,
# single, bare). The old TestWorkflowSurface regex accepted only the first.
CRON_RE = re.compile(r"""^\s*-\s*cron:\s*(?:"([^"]+)"|'([^']+)'|([^\s#][^#]*?))\s*(?:#.*)?$""")


def parse_jobs(text):
    """Job ids = keys at exactly 2-space indent inside the top-level `jobs:`.

    Tracking the `jobs:` block matters: `on:` also has 2-space-indent keys
    (`  push:` / `  pull_request:`), so the naive `^  (\\w+):` grep used
    elsewhere would report them as jobs.
    """
    jobs = []
    in_jobs = False
    for line in text.splitlines():
        if re.match(r"^jobs:\s*(#.*)?$", line):
            in_jobs = True
            continue
        if in_jobs and line and not line[0].isspace() and not line.lstrip().startswith("#"):
            in_jobs = False  # a new top-level key ended the jobs block
        if in_jobs:
            m = re.match(r"^ {2}([A-Za-z_][A-Za-z0-9_-]*):\s*(#.*)?$", line)
            if m:
                jobs.append(m.group(1))
    return jobs


def parse_job_sections(text):
    """Return (job_id, first_line, last_line) for each top-level job."""
    lines = text.splitlines()
    starts = []
    in_jobs = False
    for lineno, line in enumerate(lines, 1):
        if re.match(r"^jobs:\s*(#.*)?$", line):
            in_jobs = True
            continue
        if in_jobs and line and not line[0].isspace() and not line.lstrip().startswith("#"):
            break
        if in_jobs:
            match = re.match(r"^ {2}([A-Za-z_][A-Za-z0-9_-]*):\s*(#.*)?$", line)
            if match:
                starts.append((match.group(1), lineno))
    out = []
    for index, (job_id, first) in enumerate(starts):
        last = starts[index + 1][1] - 1 if index + 1 < len(starts) else len(lines)
        out.append((job_id, first, last))
    return out


NEEDS_RE = re.compile(r"^\s*needs:\s*(.+?)\s*(?:#.*)?$")


def parse_needs(text):
    """Return [(lineno, [job-id, ...])] for scalar / inline-list / block-list."""
    out = []
    lines = text.splitlines()
    for i, line in enumerate(lines, 1):
        m = NEEDS_RE.match(line)
        if not m:
            # Block-list form:  needs:\n      - a\n      - b
            if re.match(r"^\s*needs:\s*(#.*)?$", line):
                ids, j = [], i  # i is 1-based; lines[i] is the NEXT line
                while j < len(lines):
                    mm = re.match(r"^\s*-\s*([A-Za-z_][A-Za-z0-9_-]*)\s*(?:#.*)?$", lines[j])
                    if not mm:
                        break
                    ids.append(mm.group(1))
                    j += 1
                if ids:
                    out.append((i, ids))
            continue
        raw = m.group(1)
        if raw.startswith("["):
            ids = [t.strip().strip("\"'") for t in raw.strip("[]").split(",") if t.strip()]
        else:
            ids = [raw.strip().strip("\"'")]
        out.append((i, ids))
    return out


# `run: |` / `run: >` blocks, plus the step's `shell:` if it declares one.
# The `(?:-\s+)?` prefix matters: `      - run: |` (run as the step's FIRST
# key) is as valid as `        run: |` under a preceding `- name:`. Without it
# the block is invisible and its shell never syntax-checked — a silent
# false-negative the self-test pins down.
RUN_RE = re.compile(r"^(\s*(?:-\s+)?)run:\s*([|>])[-+0-9]*\s*(?:#.*)?$")
RUN_SINGLE_RE = re.compile(r"^\s*(?:-\s+)?run:\s+(.+?)\s*$")


def parse_run_blocks(text):
    """Yield (lineno, shell_or_None, body) for every block-scalar `run:`."""
    lines = text.splitlines()
    out = []
    i = 0
    while i < len(lines):
        rm = RUN_RE.match(lines[i])
        if rm:
            # Column at which `run:` itself starts — body lines must be
            # indented strictly deeper than this, whether or not a `- ` sits
            # in front of the key.
            indent = len(rm.group(1))
            body, j = [], i + 1
            while j < len(lines):
                ln = lines[j]
                if ln.strip() == "":
                    body.append("")
                    j += 1
                    continue
                if len(ln) - len(ln.lstrip()) <= indent:
                    break
                body.append(ln)
                j += 1
            # The step's `shell:` may sit before or after the `run:` key; scan
            # the contiguous step block around it at the same indent.
            shell = None
            for k in range(max(0, i - 12), min(len(lines), j + 1)):
                sm = re.match(r"^\s*shell:\s*([A-Za-z0-9_-]+)", lines[k])
                if sm and abs((len(lines[k]) - len(lines[k].lstrip())) - indent) <= 2:
                    shell = sm.group(1)
            out.append((i + 1, shell, "\n".join(body)))
            i = j
            continue
        i += 1
    return out


# Python tooling policy checker. This deliberately recognises commands only
# at shell command positions (start of line, after a shell operator, or after
# a control keyword). Prose such as `echo "use python"` therefore remains
# legal, while command substitutions such as `VALUE="$(python3 -c ...)"` are
# still caught. `--python` is an uv option and cannot match because `-` is
# excluded by the command-token boundary.
PYTHON_COMMAND_RE = re.compile(
    r"""(?x)
    (?:
        ^
      | && | \|\| | [;|`(]
      | \b(?:if|elif|while|until|then|do|command|exec|time)\s+
      | \benv\s+(?:(?:-[^\s]+|[A-Za-z_][A-Za-z0-9_]*=(?:"[^"]*"|'[^']*'|[^\s]+))\s+)*
      | !
    )
    \s*
    (?:[A-Za-z_][A-Za-z0-9_]*=(?:"[^"]*"|'[^']*'|[^\s]+)\s+)*
    (?P<command>
        python3? | pip3? | pytest
      | (?:[^\s;&|()`"']+/)+(?:python(?:\.exe)?|pip(?:\.exe)?)
    )
    (?=\s|$)
    """
)
UV_COMMAND_RE = re.compile(
    r"""(?x)
    (?:
        ^
      | && | \|\| | [;|`(]
      | \b(?:if|elif|while|until|then|do|command|exec|time)\s+
      | \brun\s+(?:"[^"]*"|'[^']*'|[^\s]+)\s+
      | \benv\s+(?:(?:-[^\s]+|[A-Za-z_][A-Za-z0-9_]*=(?:"[^"]*"|'[^']*'|[^\s]+))\s+)*
      | !
    )
    \s*
    (?:[A-Za-z_][A-Za-z0-9_]*=(?:"[^"]*"|'[^']*'|[^\s]+)\s+)*
    uv\s+(?:run|sync|venv|pip|python|tool)\b
    """
)
VENV_ACTIVATE_RE = re.compile(
    r"(?:^|&&|\|\||[;|`(])\s*(?:source|\.)\s+"
    r"[^\s;&|]+/(?:bin|Scripts)/(?:activate|Activate\.ps1)(?=\s|$)"
)
HEREDOC_RE = re.compile(r"<<-?\s*['\"]?([A-Za-z_][A-Za-z0-9_]*)['\"]?")
SETUP_UV_PIN = "ae62891fec2bb8e7d6c99fc78c9fec3a63790f8d"
SETUP_UV_RE = re.compile(
    r"^\s*(?:-\s+)?uses:\s*astral-sh/setup-uv@([^\s#]+)\s*(?:#.*)?$"
)
USES_RE = re.compile(
    r'''^\s*(?:-\s+)?uses:\s*(?:"([^"]+)"|'([^']+)'|([^\s#]+))\s*(?:#.*)?$'''
)
CONTAINER_IMAGE_RE = re.compile(
    r'''^\s*image:\s*(?:"([^"]+)"|'([^']+)'|([^\s#]+))\s*(?:#.*)?$'''
)
FULL_ACTION_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
OCI_DIGEST_RE = re.compile(r"@sha256:[0-9a-f]{64}(?:\s|$)")
DOCKER_FETCH_RE = re.compile(r"\bdocker\s+(?:--[^\s]+\s+)*(?:run|pull)\b")


def scalar_value(match):
    """Return a YAML plain, single-quoted, or double-quoted scalar value."""
    return next(group for group in match.groups() if group is not None)


def iter_logical_shell_commands(run_lineno, body):
    """Yield joined shell commands while skipping heredoc payloads."""
    heredoc_end = None
    logical = ""
    logical_lineno = None
    for offset, line in enumerate(body.splitlines(), 1):
        stripped = line.strip()
        lineno = run_lineno + offset
        if heredoc_end is not None:
            if stripped == heredoc_end:
                heredoc_end = None
            continue
        if not logical and (not stripped or stripped.startswith("#")):
            continue
        if logical_lineno is None:
            logical_lineno = lineno
        if stripped.endswith("\\"):
            logical += stripped[:-1].rstrip() + " "
            continue
        logical += stripped
        yield logical_lineno, logical
        heredoc = HEREDOC_RE.search(logical)
        if heredoc:
            heredoc_end = heredoc.group(1)
        logical = ""
        logical_lineno = None

    # A dangling continuation is already a bash syntax error in check (c),
    # but still scan it so a bare command cannot disappear from this gate's
    # diagnostics merely because the block is malformed too.
    if logical:
        yield logical_lineno or run_lineno, logical


def find_bare_python_commands(run_lineno, body):
    """Yield (line, command) for non-uv Python tooling in one run block."""
    for lineno, logical in iter_logical_shell_commands(run_lineno, body):
        match = PYTHON_COMMAND_RE.search(logical)
        if match:
            yield lineno, match.group("command")
        if VENV_ACTIVATE_RE.search(logical):
            yield lineno, "venv activation"


def find_uv_commands(run_lineno, body):
    """Yield line numbers for uv commands in one run block."""
    for lineno, logical in iter_logical_shell_commands(run_lineno, body):
        if UV_COMMAND_RE.search(logical):
            yield lineno


def find_bare_python_in_single_run(lineno, line):
    """Return violations from a scalar `run: command` entry, if any."""
    match = RUN_SINGLE_RE.match(line)
    if not match or match.group(1).startswith(("|", ">")):
        return []
    command = match.group(1)
    out = []
    bare = PYTHON_COMMAND_RE.search(command)
    if bare:
        out.append((lineno, bare.group("command")))
    if VENV_ACTIVATE_RE.search(command):
        out.append((lineno, "venv activation"))
    return out


def single_run_uses_uv(line):
    match = RUN_SINGLE_RE.match(line)
    return bool(
        match
        and not match.group(1).startswith(("|", ">"))
        and UV_COMMAND_RE.search(match.group(1))
    )


# ------------------------------------------------------------------ collect --
inventory = []
files = sorted(wf_dir.glob("*.yml")) + sorted(wf_dir.glob("*.yaml"))
if not files:
    print(f"check-workflow-hygiene: no workflow files under {wf_dir}", file=sys.stderr)
    sys.exit(2)

all_crons = []  # (file, lineno, expr, parsed)
for f in files:
    text = f.read_text(encoding="utf-8")
    jobs = parse_jobs(text)

    # (g) Every remote action and container image must resolve immutably.
    # Local actions/reusable workflows (`./...`) are part of this checkout and
    # therefore inherit its commit identity.
    for lineno, line in enumerate(text.splitlines(), 1):
        uses_match = USES_RE.match(line)
        if uses_match:
            target = scalar_value(uses_match)
            if target.startswith("./"):
                continue
            if target.startswith("docker://"):
                if not OCI_DIGEST_RE.search(target):
                    problems.append(
                        f"{f.name}:{lineno}: docker action is not pinned to an OCI digest: {target}"
                    )
                continue
            action, separator, ref = target.rpartition("@")
            if not separator or not action or not FULL_ACTION_SHA_RE.fullmatch(ref):
                problems.append(
                    f"{f.name}:{lineno}: remote action must use a full 40-hex commit SHA: {target}"
                )

        image_match = CONTAINER_IMAGE_RE.match(line)
        if image_match and not OCI_DIGEST_RE.search(scalar_value(image_match)):
            problems.append(
                f"{f.name}:{lineno}: workflow container image must use @sha256 digest: "
                f"{scalar_value(image_match)}"
            )

    for lineno, line in enumerate(text.splitlines(), 1):
        cm = CRON_RE.match(line)
        if cm:
            expr = (cm.group(1) or cm.group(2) or cm.group(3) or "").strip()
            where = f"{f.name}:{lineno}"
            parsed = parse_cron(expr, where)
            if parsed:
                all_crons.append((f.name, lineno, expr, parsed))

    # (b) needs: integrity
    for lineno, ids in parse_needs(text):
        for jid in ids:
            if jid not in jobs:
                problems.append(
                    f"{f.name}:{lineno}: needs: '{jid}' is not a job id in this file "
                    f"(jobs: {', '.join(jobs) or '<none>'})"
                )

    # (d) plain scalars that YAML cannot parse. Only single-line `key: value`
    #     mappings are candidates; block scalars are handled by (c) and their
    #     bodies are exempt, so track and skip them.
    lines = text.splitlines()
    block_body_until = -1
    for lineno, line in enumerate(lines, 1):
        if lineno <= block_body_until:
            continue
        bm = re.match(r"^(\s*(?:-\s+)?)[A-Za-z_][\w-]*:\s*[|>][-+0-9]*\s*(?:#.*)?$", line)
        if bm:
            indent = len(bm.group(1))
            j = lineno  # lines[j] is the next line (0-based index == 1-based lineno)
            while j < len(lines) and (
                lines[j].strip() == "" or (len(lines[j]) - len(lines[j].lstrip())) > indent
            ):
                j += 1
            block_body_until = j
            continue
        km = re.match(r"^\s*(?:-\s+)?([A-Za-z_][\w-]*):[ \t]+(\S.*?)\s*$", line)
        if not km:
            continue
        value = km.group(2)
        if value[0] in "\"'" or value.startswith(("|", ">", "&", "*", "{", "[")):
            continue
        # Strip a trailing comment only when it is clearly one (space + #).
        value = re.sub(r"\s+#.*$", "", value)
        if ": " in value:
            problems.append(
                f"{f.name}:{lineno}: plain scalar contains ': ' and will fail YAML "
                f"parsing — quote it:  {km.group(1)}: \"{value}\""
            )

    # (c) run: block shell syntax
    uv_command_lines = []
    for lineno, shell, body in parse_run_blocks(text):
        if shell is not None and shell not in ("bash", "sh"):
            continue  # pwsh / python / cmd — not ours to syntax-check
        # `${{ ... }}` is GitHub expression syntax, not shell. Substitute a
        # inert token so `bash -n` sees a syntactically valid word.
        scrubbed = re.sub(r"\$\{\{[^}]*\}\}", "GHA_EXPR", body)
        with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as tmp:
            tmp.write(scrubbed + "\n")
            tmp_path = tmp.name
        res = subprocess.run(
            ["bash", "-n", tmp_path], capture_output=True, text=True
        )
        pathlib.Path(tmp_path).unlink()
        if res.returncode != 0:
            detail = res.stderr.strip().replace(tmp_path, f"{f.name}:run@{lineno}")
            problems.append(f"{f.name}:{lineno}: run: block is not valid bash — {detail}")

        # (e) All shell-launched Python tooling must go through uv. This is
        #     independent of shell syntax: a bare command is valid bash but
        #     bypasses the repository's lock/interpreter policy.
        for command_lineno, command in find_bare_python_commands(lineno, body):
            problems.append(
                f"{f.name}:{command_lineno}: bare Python tooling '{command}' in run block — "
                "use uv run / uv pip / uv venv without activation"
            )
        uv_command_lines.extend(find_uv_commands(lineno, body))
        for command_lineno, logical in iter_logical_shell_commands(lineno, body):
            if DOCKER_FETCH_RE.search(logical) and not OCI_DIGEST_RE.search(logical):
                problems.append(
                    f"{f.name}:{command_lineno}: docker run/pull must pin the image with "
                    "@sha256:<64-hex>"
                )

    # Single-line `run: python ...` entries are not returned by
    # parse_run_blocks(), so scan those separately.
    for lineno, line in enumerate(text.splitlines(), 1):
        for command_lineno, command in find_bare_python_in_single_run(lineno, line):
            problems.append(
                f"{f.name}:{command_lineno}: bare Python tooling '{command}' in scalar run — "
                "use uv run / uv pip / uv venv without activation"
            )
        if single_run_uses_uv(line):
            uv_command_lines.append(lineno)
        scalar_match = RUN_SINGLE_RE.match(line)
        if scalar_match and not scalar_match.group(1).startswith(("|", ">")):
            scalar_command = scalar_match.group(1)
            if DOCKER_FETCH_RE.search(scalar_command) and not OCI_DIGEST_RE.search(
                scalar_command
            ):
                problems.append(
                    f"{f.name}:{lineno}: docker run/pull must pin the image with "
                    "@sha256:<64-hex>"
                )

    # (f) uv execution and setup must live in the same job, with the action
    #     pinned to the repository-reviewed commit. A misplaced setup step is
    #     as dangerous as an omitted one: both make the intended job depend on
    #     ambient runner state.
    lines = text.splitlines()
    for job_id, first, last in parse_job_sections(text):
        uses_uv = any(first <= lineno <= last for lineno in uv_command_lines)
        setup_steps = []
        for lineno in range(first, last + 1):
            setup_match = SETUP_UV_RE.match(lines[lineno - 1])
            if setup_match:
                setup_steps.append((lineno, setup_match.group(1)))
        setup_lines = [lineno for lineno, _ in setup_steps]
        if uses_uv and not setup_lines:
            problems.append(
                f"{f.name}:{first}: job '{job_id}' executes uv without setup-uv"
            )
        if not uses_uv and setup_lines:
            problems.append(
                f"{f.name}:{setup_lines[0]}: job '{job_id}' installs setup-uv but never executes uv"
            )
        if uses_uv and setup_lines:
            first_uv = min(
                lineno for lineno in uv_command_lines if first <= lineno <= last
            )
            if first_uv < setup_lines[0]:
                problems.append(
                    f"{f.name}:{first_uv}: job '{job_id}' executes uv before setup-uv"
                )
        if len(setup_lines) > 1:
            problems.append(
                f"{f.name}:{setup_lines[1]}: job '{job_id}' has duplicate setup-uv steps"
            )
        for setup_lineno, setup_ref in setup_steps:
            if setup_ref != SETUP_UV_PIN:
                problems.append(
                    f"{f.name}:{setup_lineno}: setup-uv must be pinned to {SETUP_UV_PIN}"
                )

    inventory.append((f.name, jobs))

# (a) cron collisions — all pairs, both quote styles.
for i in range(len(all_crons)):
    fi, li, ei, (mi, hi, di) = all_crons[i]
    for j in range(i + 1, len(all_crons)):
        fj, lj, ej, (mj, hj, dj) = all_crons[j]
        if mi & mj and hi & hj and di & dj:
            problems.append(
                f"cron collision: {fi}:{li} '{ei}' and {fj}:{lj} '{ej}' "
                f"share minute/hour/day-of-week"
            )

if mode == "--list":
    print(f"workflows: {len(files)}")
    for name, jobs in inventory:
        print(f"  {name}: {len(jobs)} job(s)")
    print(f"crons: {len(all_crons)}")
    for name, lineno, expr, _ in all_crons:
        print(f"  {name}:{lineno}  {expr}")
    sys.exit(0)

if problems:
    print("check-workflow-hygiene: FAIL", file=sys.stderr)
    for p in problems:
        print(f"  {p}", file=sys.stderr)
    sys.exit(1)

print(
    f"check-workflow-hygiene: OK ({len(files)} workflow(s), {len(all_crons)} cron(s), "
    "no collisions / dangling needs / shell syntax errors / bare Python tooling / "
    "setup-uv drift / mutable CI dependencies)"
)
PY
}

# -------------------------------------------------------------- self-test ---
# Synthesize throwaway workflow trees and assert the checker's verdict on each.
# Mirrors scripts/check-platform-support.sh's self_test structure.
self_test() {
    local tmp rc=0
    tmp="$(mktemp -d)"
    # shellcheck disable=SC2064
    trap "rm -rf '$tmp'" RETURN

    # (1) A clean tree must pass, including a SINGLE-quoted cron (the exact
    #     form the old TestWorkflowSurface regex could not see).
    mkdir -p "$tmp/ok"
    cat >"$tmp/ok/a.yml" <<'YML'
name: A
on:
  schedule:
    - cron: '0 4 * * 1'
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - run: |
          echo hi
  after:
    needs: build
    runs-on: ubuntu-latest
    steps:
      - run: echo ok
YML
    cat >"$tmp/ok/b.yml" <<'YML'
name: B
on:
  schedule:
    - cron: "0 5 * * 1"
jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - run: |
          echo lint
YML
    if ! run_checker "$tmp/ok" check >/dev/null 2>&1; then
        echo "self-test FAILED: a clean tree should pass" >&2
        rc=1
    fi

    # (2) Cron collision across two SINGLE-quoted crons — the blind spot.
    mkdir -p "$tmp/collide"
    cp "$tmp/ok/a.yml" "$tmp/collide/a.yml"
    cat >"$tmp/collide/b.yml" <<'YML'
name: B
on:
  schedule:
    - cron: '0 4 * * 1'
jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - run: echo lint
YML
    if run_checker "$tmp/collide" check >/dev/null 2>&1; then
        echo "self-test FAILED: a single-quoted cron collision should fail" >&2
        rc=1
    fi

    # (2b) Collision between two workflows that are BOTH "other" relative to a
    #      third — the structural blind spot of the per-workflow unittest.
    mkdir -p "$tmp/collide2"
    cp "$tmp/ok/a.yml" "$tmp/collide2/a.yml"
    cat >"$tmp/collide2/c.yml" <<'YML'
name: C
on:
  schedule:
    - cron: "30 9 * * 3"
jobs:
  one:
    runs-on: ubuntu-latest
    steps:
      - run: echo one
YML
    cat >"$tmp/collide2/d.yml" <<'YML'
name: D
on:
  schedule:
    - cron: '30 9 * * 3'
jobs:
  two:
    runs-on: ubuntu-latest
    steps:
      - run: echo two
YML
    if run_checker "$tmp/collide2" check >/dev/null 2>&1; then
        echo "self-test FAILED: an other-vs-other cron collision should fail" >&2
        rc=1
    fi

    # (2c) A wildcard day-of-week must intersect a specific weekday.
    mkdir -p "$tmp/collide3"
    cat >"$tmp/collide3/a.yml" <<'YML'
name: A
on:
  schedule:
    - cron: "15 6 * * *"
jobs:
  one:
    runs-on: ubuntu-latest
    steps:
      - run: echo one
YML
    cat >"$tmp/collide3/b.yml" <<'YML'
name: B
on:
  schedule:
    - cron: "15 6 * * 4"
jobs:
  two:
    runs-on: ubuntu-latest
    steps:
      - run: echo two
YML
    if run_checker "$tmp/collide3" check >/dev/null 2>&1; then
        echo "self-test FAILED: '*' dow must intersect a specific weekday" >&2
        rc=1
    fi

    # (3) A dangling `needs:` job id must fail.
    mkdir -p "$tmp/needs"
    cat >"$tmp/needs/a.yml" <<'YML'
name: A
on: [push]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - run: echo build
  after:
    needs: buidl
    runs-on: ubuntu-latest
    steps:
      - run: echo after
YML
    if run_checker "$tmp/needs" check >/dev/null 2>&1; then
        echo "self-test FAILED: a dangling needs: id should fail" >&2
        rc=1
    fi

    # (3b) Inline-list needs: with one bad id must fail.
    mkdir -p "$tmp/needs2"
    cat >"$tmp/needs2/a.yml" <<'YML'
name: A
on: [push]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - run: echo build
  after:
    needs: [build, nope]
    runs-on: ubuntu-latest
    steps:
      - run: echo after
YML
    if run_checker "$tmp/needs2" check >/dev/null 2>&1; then
        echo "self-test FAILED: an inline-list dangling needs: id should fail" >&2
        rc=1
    fi

    # (3c) `on:`'s 2-space keys (push / pull_request) must NOT be read as job
    #      ids — otherwise a `needs: push` typo would be accepted.
    mkdir -p "$tmp/onkeys"
    cat >"$tmp/onkeys/a.yml" <<'YML'
name: A
on:
  push:
    branches: [main]
  pull_request:
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - run: echo build
  after:
    needs: push
    runs-on: ubuntu-latest
    steps:
      - run: echo after
YML
    if run_checker "$tmp/onkeys" check >/dev/null 2>&1; then
        echo "self-test FAILED: 'push' under on: must not count as a job id" >&2
        rc=1
    fi

    # (4) Broken shell in a `run:` block must fail.
    mkdir -p "$tmp/shell"
    cat >"$tmp/shell/a.yml" <<'YML'
name: A
on: [push]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - run: |
          if [ -f x ]; then
            echo yes
          # missing fi
YML
    if run_checker "$tmp/shell" check >/dev/null 2>&1; then
        echo "self-test FAILED: a broken run: block should fail" >&2
        rc=1
    fi

    # (4b) A `${{ }}` expression must NOT be mistaken for broken shell.
    mkdir -p "$tmp/expr"
    cat >"$tmp/expr/a.yml" <<'YML'
name: A
on: [push]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - run: |
          echo "${{ matrix.os }} ${{ github.sha }}"
          test -n "${{ inputs.thing }}"
YML
    if ! run_checker "$tmp/expr" check >/dev/null 2>&1; then
        echo "self-test FAILED: a \${{ }} expression must not read as broken shell" >&2
        rc=1
    fi

    # (4c) A pwsh step must be skipped, not syntax-checked as bash.
    mkdir -p "$tmp/pwsh"
    cat >"$tmp/pwsh/a.yml" <<'YML'
name: A
on: [push]
jobs:
  build:
    runs-on: windows-latest
    steps:
      - name: pwsh step
        shell: pwsh
        run: |
          if ($true) { Write-Host "ok" }
YML
    if ! run_checker "$tmp/pwsh" check >/dev/null 2>&1; then
        echo "self-test FAILED: a pwsh run: block must be skipped, not bash-linted" >&2
        rc=1
    fi

    # (4d) A plain scalar containing ": " must fail — this is the exact defect
    #      that slipped into ci.yml while X-08 was wiring this script in.
    mkdir -p "$tmp/scalar"
    cat >"$tmp/scalar/a.yml" <<'YML'
name: A
on: [push]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - name: Workflow hygiene (cron / needs: ids / shell)
        run: echo hi
YML
    if run_checker "$tmp/scalar" check >/dev/null 2>&1; then
        echo "self-test FAILED: a plain scalar containing ': ' should fail" >&2
        rc=1
    fi

    # (4e) The legitimate forms must NOT be flagged: a QUOTED value with ": ",
    #      a colon with no space, a URL, and ": " inside a block-scalar body.
    mkdir -p "$tmp/scalarok"
    cat >"$tmp/scalarok/a.yml" <<'YML'
name: A
on: [push]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - name: "Workflow hygiene (cron / needs: ids / shell)"
        run: echo hi
      - name: ratio 1:2 sweep
        run: curl -sS https://example.com/x
      - name: block body
        run: |
          echo "note: this colon-space is inside a block scalar"
          printf 'key: value\n'
YML
    if ! run_checker "$tmp/scalarok" check >/dev/null 2>&1; then
        echo "self-test FAILED: quoted / no-space / URL / block-scalar forms must pass" >&2
        rc=1
    fi

    # (5) An unparsable cron must be a hard error, never a silent pass.
    mkdir -p "$tmp/badcron"
    cat >"$tmp/badcron/a.yml" <<'YML'
name: A
on:
  schedule:
    - cron: "0 4 * *"
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - run: echo build
YML
    if run_checker "$tmp/badcron" check >/dev/null 2>&1; then
        echo "self-test FAILED: a 4-field cron should be a hard error" >&2
        rc=1
    fi

    # (6) Every prohibited Python family gets an independent red fixture. A
    #     single fixture containing all of them would still pass if all but one
    #     detector regressed, which is not a meaningful bidirectional test.
    assert_rejected_scalar() {
        local name="$1" command="$2" dir="$tmp/red-$1"
        mkdir -p "$dir"
        cat >"$dir/a.yml" <<YML
name: A
on: [push]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - run: $command
YML
        if run_checker "$dir" check >/dev/null 2>&1; then
            echo "self-test FAILED: '$name' should be rejected" >&2
            rc=1
        fi
    }

    assert_rejected_block() {
        local name="$1" command="$2" dir="$tmp/red-$1"
        mkdir -p "$dir"
        cat >"$dir/a.yml" <<YML
name: A
on: [push]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - run: |
          $command
YML
        if run_checker "$dir" check >/dev/null 2>&1; then
            echo "self-test FAILED: '$name' should be rejected" >&2
            rc=1
        fi
    }

    assert_rejected_scalar "scalar-python3" "python3 tools/check.py"
    assert_rejected_block  "block-python" "python -m venv /tmp/example"
    assert_rejected_block  "block-pip" "pip install pytest"
    assert_rejected_scalar "scalar-pip3" "pip3 freeze"
    assert_rejected_scalar "scalar-pytest" "pytest -q"
    assert_rejected_scalar "direct-venv-python" "/tmp/example/bin/python tools/check.py"
    assert_rejected_scalar "direct-windows-python" "C:/tmp/Scripts/python.exe tools/check.py"
    assert_rejected_scalar "direct-venv-pip" "/tmp/example/bin/pip install pytest"
    assert_rejected_block  "source-activation" "source /tmp/example/bin/activate"
    assert_rejected_block  "dot-activation" ". /tmp/example/bin/activate"
    # shellcheck disable=SC2016  # literal command substitution belongs in the fixture.
    assert_rejected_block  "command-substitution" 'VALUE="$(python3 -c pass)"'
    assert_rejected_block  "chained-command" "echo ready && python3 tools/check.py"
    assert_rejected_block  "env-wrapped-python" "env MODE=test python3 tools/check.py"

    # (6b) Every supported uv-managed form must pass, including an explicit
    #      venv interpreter and a heredoc whose Python body contains prose
    #      that resembles shell commands.
    mkdir -p "$tmp/uvpython"
    cat >"$tmp/uvpython/a.yml" <<'YML'
name: A
on: [push]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: astral-sh/setup-uv@ae62891fec2bb8e7d6c99fc78c9fec3a63790f8d # v10.0.0
      - run: uv run --no-project --python 3.12 python tools/check.py
      - run: env MODE=test uv run --no-project --python 3.12 python tools/check.py
      - run: |
          uv venv --python 3.12 /tmp/example
          uv pip install --python /tmp/example/bin/python pytest
          uv run --no-project --python /tmp/example/bin/python \
            python - <<'PY'
          note = "python -m pip and pytest are prose inside Python source"
          print(note)
          PY
YML
    if ! run_checker "$tmp/uvpython" check >/dev/null 2>&1; then
        echo "self-test FAILED: uv-managed Python forms should pass" >&2
        rc=1
    fi

    # (6c) A valid uv command without setup-uv in the SAME job must fail.
    mkdir -p "$tmp/missingsetupuv"
    cat >"$tmp/missingsetupuv/a.yml" <<'YML'
name: A
on: [push]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - run: uv run --no-project --python 3.12 python tools/check.py
YML
    if run_checker "$tmp/missingsetupuv" check >/dev/null 2>&1; then
        echo "self-test FAILED: uv execution without same-job setup-uv should fail" >&2
        rc=1
    fi

    # (6d) A floating setup-uv ref is not an acceptable substitute for the
    #      repository-reviewed SHA.
    mkdir -p "$tmp/unpinnedsetupuv"
    cat >"$tmp/unpinnedsetupuv/a.yml" <<'YML'
name: A
on: [push]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: astral-sh/setup-uv@v10
      - run: uv run --no-project --python 3.12 python tools/check.py
YML
    if run_checker "$tmp/unpinnedsetupuv" check >/dev/null 2>&1; then
        echo "self-test FAILED: unpinned setup-uv should fail" >&2
        rc=1
    fi

    # (6e) A comment mentioning the pinned action is not an installation step.
    mkdir -p "$tmp/commentedsetupuv"
    cat >"$tmp/commentedsetupuv/a.yml" <<'YML'
name: A
on: [push]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      # - uses: astral-sh/setup-uv@ae62891fec2bb8e7d6c99fc78c9fec3a63790f8d
      - run: uv run --no-project --python 3.12 python tools/check.py
YML
    if run_checker "$tmp/commentedsetupuv" check >/dev/null 2>&1; then
        echo "self-test FAILED: a commented setup-uv line must not satisfy the job" >&2
        rc=1
    fi

    # (6f) Duplicate installation is drift even when both refs are pinned.
    mkdir -p "$tmp/duplicatesetupuv"
    cat >"$tmp/duplicatesetupuv/a.yml" <<'YML'
name: A
on: [push]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: astral-sh/setup-uv@ae62891fec2bb8e7d6c99fc78c9fec3a63790f8d
      - uses: astral-sh/setup-uv@ae62891fec2bb8e7d6c99fc78c9fec3a63790f8d
      - run: uv run --no-project --python 3.12 python tools/check.py
YML
    if run_checker "$tmp/duplicatesetupuv" check >/dev/null 2>&1; then
        echo "self-test FAILED: duplicate setup-uv steps should fail" >&2
        rc=1
    fi

    # (6g) Installing uv in a job that never executes it is a copy/paste error.
    mkdir -p "$tmp/unusedsetupuv"
    cat >"$tmp/unusedsetupuv/a.yml" <<'YML'
name: A
on: [push]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: astral-sh/setup-uv@ae62891fec2bb8e7d6c99fc78c9fec3a63790f8d
      - run: echo no-python-here
YML
    if run_checker "$tmp/unusedsetupuv" check >/dev/null 2>&1; then
        echo "self-test FAILED: unused setup-uv should fail" >&2
        rc=1
    fi

    # (6h) Setup in a sibling job cannot satisfy the job that invokes uv.
    mkdir -p "$tmp/crossjobsetupuv"
    cat >"$tmp/crossjobsetupuv/a.yml" <<'YML'
name: A
on: [push]
jobs:
  setup_only:
    runs-on: ubuntu-latest
    steps:
      - uses: astral-sh/setup-uv@ae62891fec2bb8e7d6c99fc78c9fec3a63790f8d
      - run: echo setup-only
  execute:
    runs-on: ubuntu-latest
    steps:
      - run: uv run --no-project --python 3.12 python tools/check.py
YML
    if run_checker "$tmp/crossjobsetupuv" check >/dev/null 2>&1; then
        echo "self-test FAILED: setup-uv must be in the same job as uv execution" >&2
        rc=1
    fi

    # (6i) Same-job setup is still too late when uv was already executed.
    mkdir -p "$tmp/latesetupuv"
    cat >"$tmp/latesetupuv/a.yml" <<'YML'
name: A
on: [push]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - run: uv run --no-project --python 3.12 python tools/check.py
      - uses: astral-sh/setup-uv@ae62891fec2bb8e7d6c99fc78c9fec3a63790f8d
YML
    if run_checker "$tmp/latesetupuv" check >/dev/null 2>&1; then
        echo "self-test FAILED: setup-uv after its first use should fail" >&2
        rc=1
    fi

    # (7a) A moving major tag can silently replace reviewed action code.
    mkdir -p "$tmp/unpinnedaction"
    cat >"$tmp/unpinnedaction/a.yml" <<'YML'
name: A
on: [push]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v7
      - run: echo done
YML
    if run_checker "$tmp/unpinnedaction" check >/dev/null 2>&1; then
        echo "self-test FAILED: a remote action tag should fail immutable pinning" >&2
        rc=1
    fi

    # (7b) Quoting a YAML scalar must not bypass the moving-tag rejection.
    mkdir -p "$tmp/quotedunpinnedaction"
    cat >"$tmp/quotedunpinnedaction/a.yml" <<'YML'
name: A
on: [push]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: "actions/checkout@v7"
YML
    if run_checker "$tmp/quotedunpinnedaction" check >/dev/null 2>&1; then
        echo "self-test FAILED: a quoted remote action tag should fail immutable pinning" >&2
        rc=1
    fi

    # (7c) A quoted full commit SHA is still immutable and valid YAML.
    mkdir -p "$tmp/quotedpinnedaction"
    cat >"$tmp/quotedpinnedaction/a.yml" <<'YML'
name: A
on: [push]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: 'actions/checkout@aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'
YML
    if ! run_checker "$tmp/quotedpinnedaction" check >/dev/null 2>&1; then
        echo "self-test FAILED: a quoted full action SHA should pass immutable pinning" >&2
        rc=1
    fi

    # (7d) A version tag is still mutable in a registry; Docker execution
    #      requires the reviewed multi-arch digest as well.
    mkdir -p "$tmp/mutabledocker"
    cat >"$tmp/mutabledocker/a.yml" <<'YML'
name: A
on: [push]
jobs:
  scan:
    runs-on: ubuntu-latest
    steps:
      - run: docker run --rm example/scanner:v1 scan /repo
YML
    if run_checker "$tmp/mutabledocker" check >/dev/null 2>&1; then
        echo "self-test FAILED: a Docker version tag without digest should fail" >&2
        rc=1
    fi

    # (7e) A tag plus OCI digest is readable and immutable. The tag keeps the
    #      version visible to reviewers; the digest is the execution identity.
    mkdir -p "$tmp/pinneddocker"
    cat >"$tmp/pinneddocker/a.yml" <<'YML'
name: A
on: [push]
jobs:
  scan:
    runs-on: ubuntu-latest
    steps:
      - run: docker run --rm example/scanner:v1@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa scan /repo
YML
    if ! run_checker "$tmp/pinneddocker" check >/dev/null 2>&1; then
        echo "self-test FAILED: a Docker image pinned by OCI digest should pass" >&2
        rc=1
    fi

    if [ "$rc" -eq 0 ]; then
        echo "check-workflow-hygiene --self-test: OK"
    fi
    return "$rc"
}

case "$MODE" in
    --self-test) self_test ;;
    --list) run_checker "$ROOT/.github/workflows" --list ;;
    *) run_checker "$ROOT/.github/workflows" check ;;
esac
