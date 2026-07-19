#!/usr/bin/env python3
"""Flag `<unbounded producer> | grep -q` pipelines inside `set -o pipefail` shell.

WHY
---
Under `set -o pipefail`:

    if nm -u "$lib" 2>/dev/null | grep -qE '_(cudart|cudnn)'; then ... FAIL ...

fails OPEN. `grep -q` exits at its first match; the producer, still writing, is
killed by SIGPIPE (141); pipefail promotes 141 to the pipeline status; the `if`
takes the FALSE branch. The gate therefore reports "clean" for exactly the
artifact it was written to reject.

It only bites once the producer's output exceeds the ~64 KiB pipe buffer, so it
is invisible on small inputs and non-deterministic near the boundary — a gate
that passes precisely the artifacts big enough to matter.

FIX
---
Capture once, then filter without `-q` (grep without -q reads to EOF, so the
producer never sees SIGPIPE):

    dump="$(nm -u "$lib" 2>/dev/null || true)"
    hits="$(printf '%s\\n' "$dump" | grep -E '_(cudart|cudnn)' || true)"
    if [ -n "$hits" ]; then ... FAIL ...

SCOPE
-----
Checked: scripts/ and .githooks/ — the compliance-gate and git-hook surface.
tools/ is excluded (one-off measurement helpers, not gates).

Producers flagged are binary/tree inspectors and build tools whose output is
unbounded and artifact-controlled. Deliberately NOT flagged:

  * `printf` / `echo` — shell builtins emitting an already-in-memory variable.
    Bounded by what the script already read; the pipe-buffer race needs a
    producer that is still generating output. (A >64 KiB variable piped to
    `grep -q` would in principle be at risk, but no such site exists here.)
  * `file`, and any invocation carrying `--version` / `-V` — single-line output.
  * `grep` / `sed` / `awk` reading a repo file — bounded by repo content, and
    every current instance is `!`-guarded, i.e. an inversion fails CLOSED
    (spurious rejection) rather than open.

Suppress a reviewed instance with a trailing `# sigpipe-ok: <reason>` on the
same logical line.

Usage: python3 scripts/compliance/lint-pipefail-grep-q.py [paths...]
Exit 0 = clean, 1 = findings.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PRODUCERS = {
    "nm",
    "readelf",
    "objdump",
    "otool",
    "strings",
    "ldd",
    "dumpbin",
    "find",
    "tar",
    "cargo",
    "git",
    "cat",
    "lipo",
    "unzip",
}

# Words that may precede the actual command in a pipeline's first stage.
LEADERS = {"if", "!", "while", "until", "then", "elif", "do", "&&", "||", ";"}

GREP_Q = re.compile(r"\bgrep\b[^|;]*?\s-[A-Za-z]*q")
SUPPRESS = re.compile(r"#\s*sigpipe-ok:")


def logical_lines(text: str):
    """Yield (first_lineno, joined_text), joining backslash continuations."""
    raw = text.split("\n")
    i = 0
    while i < len(raw):
        start = i
        buf = raw[i]
        while buf.rstrip().endswith("\\") and i + 1 < len(raw):
            buf = buf.rstrip()[:-1] + " " + raw[i + 1]
            i += 1
        yield start + 1, buf
        i += 1


def split_stages(line: str):
    """Split on single `|`, leaving `||` intact."""
    stages, buf, k = [], "", 0
    while k < len(line):
        if line[k] == "|":
            if k + 1 < len(line) and line[k + 1] == "|":
                buf += "||"
                k += 2
                continue
            stages.append(buf)
            buf = ""
            k += 1
            continue
        buf += line[k]
        k += 1
    stages.append(buf)
    return stages


def first_command(stage: str) -> str | None:
    toks = stage.replace("(", " ").replace(")", " ").split()
    for tok in toks:
        if tok in LEADERS or tok.startswith("-"):
            continue
        if "=" in tok and not tok.startswith("$"):  # VAR=val prefix
            continue
        return tok.rsplit("/", 1)[-1]
    return None


def uses_pipefail(text: str) -> bool:
    return any(
        line.strip().startswith("set ") and "pipefail" in line
        for line in text.split("\n")
    )


def scan(path: Path) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    if not uses_pipefail(text):
        return []

    findings = []
    for lineno, line in logical_lines(text):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue  # whole-line comment (doc blocks quote the bad idiom)
        if SUPPRESS.search(line):
            continue
        stages = split_stages(line)
        if len(stages) < 2:
            continue
        if not GREP_Q.search(stages[-1]):
            continue
        producer = first_command(stages[0])
        if producer is None or producer not in PRODUCERS:
            continue
        if re.search(r"\s(--version|-V)\b", stages[0]):
            continue  # single-line, bounded
        findings.append(
            f"{path}:{lineno}: `{producer} ... | grep -q` under pipefail "
            f"fails open on >64 KiB output\n"
            f"    {stripped[:160]}"
        )
    return findings


def main(argv: list[str]) -> int:
    roots = [Path(a) for a in argv[1:]] or [Path("scripts"), Path(".githooks")]
    files: list[Path] = []
    for root in roots:
        if root.is_file():
            files.append(root)
        elif root.is_dir():
            files += [p for p in sorted(root.rglob("*")) if p.is_file()]

    findings = []
    for f in files:
        if f.suffix in {".sh", ".bash"} or f.parent.name == ".githooks":
            findings += scan(f)

    if findings:
        print("lint-pipefail-grep-q: FAIL — fail-open pipeline(s) found:\n")
        for item in findings:
            print(f"  {item}\n")
        print(
            "Capture the producer's output once, then filter it without `-q`:\n"
            '    out="$(producer ... || true)"\n'
            "    hits=\"$(printf '%s\\n' \"$out\" | grep -E 'pat' || true)\"\n"
            '    if [ -n "$hits" ]; then ... fi\n'
            "Reviewed exception: append `# sigpipe-ok: <reason>`."
        )
        return 1
    print(f"lint-pipefail-grep-q: OK ({len(files)} files scanned)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
