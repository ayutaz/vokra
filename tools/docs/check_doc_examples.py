#!/usr/bin/env python3
"""check_doc_examples.py — CI verification of the code examples in the docs
(X-08-T20..T25; NFR-MT-04 "documentation code examples are execution-verified
in CI, in both English and Japanese").

WHAT THIS IS
------------
A fenced-code-block extractor plus tier-A/B checkers, stdlib-only (NFR-DS-02
forbids adding dependencies, so there is no PyYAML / no markdown library —
the fence scanner is hand-rolled over the narrow subset the docs use).

THE THREE TIERS
---------------
  tier A  surface checks, no compiler and no network:
            * every `vokra-cli <sub> --flag` in a shell block names a real
              subcommand and a real flag, matched against the argument
              parsers in crates/vokra-cli/src/{run,convert,bench}.rs
            * every `--model <kind>` names a real ModelKind::from_arg value
            * every repo-relative path a block tells the reader to run
              (scripts/*.sh, web/demo/*.mjs, bindings/...) actually exists

  tier B  compile / API-existence checks:
            * `c` blocks compile against include/vokra.h
            * `python` blocks: names imported from `vokra` resolve in the
              binding, and methods called on a Session exist on Session
            * `js` blocks: names imported from "@vokra/web" are exported by
              web/pkg/index.js, and the package name matches package.json
            * `json` blocks parse, and repo paths they name exist

  tier C  DEFERRED, and announced as such on every run — never counted as a
          pass (fabricated-pass prohibition). These need a toolchain a PR
          runner does not have: `swift` blocks (Swift compiler), `csharp`
          blocks (Unity/C#), and the `sh` blocks that download real upstream
          checkpoints over the network and run real models. Those belong to
          the gated / nightly workflows (parity-*-real.yml, nightly-*.yml),
          not to a per-PR job.

WHY A "KNOWN GAP" LEDGER
------------------------
Two doc claims describe surface the implementation has NOT yet grown:
`bindings/python/src/vokra/__init__.py` is a self-declared "T06 skeleton"
that re-exports nothing, and `vokra.__abi_version__` is promised by
bindings/python/README.md but not implemented. Pretending those pass would
be a fabricated green; hard-failing them would make X-08 (a CI-wiring WP)
block on someone else's unfinished feature.

So they are PINNED in KNOWN_GAPS with a reason. A pinned gap is reported
loudly on every run but does not fail the build — AND the ledger is
self-cleaning: if a pinned gap becomes satisfiable (the implementation caught
up), the checker FAILS and tells you to delete the entry. That is what stops
the ledger from silently rotting into a permanent hole.

Usage:
    python3 tools/docs/check_doc_examples.py [--list | --self-test]
        --list       print the extracted block inventory and exit 0
        --self-test  run the bidirectional fixture tests (T24) and exit

Exit code: 0 = clean (announced tier-C / pinned gaps are not failures),
           1 = a drift was found, 2 = usage / setup error.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[2]

# The 12 documents NFR-MT-04 covers: 3 "core" systems + 3 platform tutorials,
# each in English and Japanese. Keeping en/ja adjacent is deliberate — the
# requirement is explicitly a two-language one, and a translated doc drifting
# away from its source is the exact failure this catches.
DOCS = [
    "docs/getting-started.md",
    "docs/getting-started.ja.md",
    "docs/tutorials/web.md",
    "docs/tutorials/web.ja.md",
    "docs/migration-guide.md",
    "docs/migration-guide.ja.md",
    "docs/tutorials/python.md",
    "docs/tutorials/python.ja.md",
    "docs/tutorials/ios.md",
    "docs/tutorials/ios.ja.md",
    "docs/tutorials/unity.md",
    "docs/tutorials/unity.ja.md",
]

# Languages we defer, with the reason announced on every run.
TIER_C_LANGS = {
    "swift": "needs a Swift compiler (owner / macOS toolchain)",
    "csharp": "needs the Unity C# toolchain (nightly-il2cpp.yml)",
    "(none)": "untagged prose block (UI steps / HTTP headers) — nothing to check",
}

# Pinned known gaps. Each entry: key -> (reason, closing-ticket pointer).
# See the module docstring for why these are pinned rather than hard-failed.
KNOWN_GAPS = {
    "python:import:vokra.__init__-reexport": (
        "bindings/python/src/vokra/__init__.py is a self-declared T06 skeleton "
        "(`__all__ = ['__version__']`); Session / errors live in submodules and "
        "are not re-exported yet. Names are therefore resolved package-wide.",
        "bindings/python/src/vokra/__init__.py docstring (T07-T11)",
    ),
    "python:attr:__abi_version__": (
        "vokra.__abi_version__ is promised by bindings/python/README.md:23 but "
        "not implemented in __init__.py (only __version__ exists).",
        "bindings/python/README.md:23",
    ),
}


# --------------------------------------------------------------- extractor --
class Block:
    __slots__ = ("doc", "lang", "start", "end", "body")

    def __init__(self, doc, lang, start, end, body):
        self.doc = doc
        self.lang = lang
        self.start = start
        self.end = end
        self.body = body

    def where(self):
        return f"{self.doc}:{self.start}"


def extract_blocks(path: pathlib.Path, rel: str):
    """Scan fenced blocks. Returns [Block]. Untagged fences get lang '(none)'."""
    blocks = []
    inblk = False
    lang = None
    start = 0
    buf: list[str] = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.startswith("```"):
            if not inblk:
                inblk, lang, start, buf = True, (line[3:].strip() or "(none)"), i, []
            else:
                inblk = False
                blocks.append(Block(rel, lang, start, i, "\n".join(buf)))
        elif inblk:
            buf.append(line)
    if inblk:
        raise SystemExit(f"{rel}: unterminated code fence opened at line {start}")
    return blocks


# ------------------------------------------------------- implementation surface --
FLAG_ARM = re.compile(r'"(--[a-z0-9-]+)"\s*=>')


def cli_surface(root: pathlib.Path):
    """Extract {subcommand: {flags}} + the --model kind set from the Rust source.

    Parsing the source (rather than shelling out to `vokra-cli --help`) keeps
    the tier-A leg cargo-free, which is what lets the doc-examples job stay a
    cheap checkout-only job.
    """
    subs = {}
    for sub, rel in (
        ("run", "crates/vokra-cli/src/run.rs"),
        ("convert", "crates/vokra-cli/src/convert.rs"),
        ("bench", "crates/vokra-cli/src/bench.rs"),
    ):
        src = root / rel
        if not src.is_file():
            raise SystemExit(f"setup error: expected CLI source {rel} not found")
        subs[sub] = set(FLAG_ARM.findall(src.read_text(encoding="utf-8")))
        if not subs[sub]:
            raise SystemExit(f"setup error: no flag arms parsed out of {rel}")

    conv = root / "crates/vokra-convert/src/lib.rs"
    if not conv.is_file():
        raise SystemExit("setup error: crates/vokra-convert/src/lib.rs not found")
    text = conv.read_text(encoding="utf-8")
    m = re.search(r"pub fn from_arg\(s: &str\) -> Option<Self> \{(.*?)\n    \}", text, re.S)
    if not m:
        raise SystemExit("setup error: could not parse ModelKind::from_arg")
    kinds = set(re.findall(r'"([a-z0-9_-]+)"\s*=>', m.group(1)))
    if not kinds:
        raise SystemExit("setup error: ModelKind::from_arg parsed to an empty set")
    return subs, kinds


def python_surface(root: pathlib.Path):
    """Names defined anywhere in the vokra Python package + Session methods."""
    pkg = root / "bindings/python/src/vokra"
    if not pkg.is_dir():
        raise SystemExit("setup error: bindings/python/src/vokra not found")
    names, session_methods = set(), set()
    for py in sorted(pkg.glob("*.py")):
        text = py.read_text(encoding="utf-8")
        names.update(re.findall(r"^class\s+([A-Za-z_][A-Za-z0-9_]*)", text, re.M))
        names.update(re.findall(r"^def\s+([A-Za-z_][A-Za-z0-9_]*)", text, re.M))
        names.update(re.findall(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=", text, re.M))
        if py.name in ("session.py", "_handles.py"):
            session_methods.update(re.findall(r"^    def\s+([A-Za-z_][A-Za-z0-9_]*)", text, re.M))
    return names, session_methods


def js_surface(root: pathlib.Path):
    idx = root / "web/pkg/index.js"
    pkg = root / "web/pkg/package.json"
    if not idx.is_file() or not pkg.is_file():
        raise SystemExit("setup error: web/pkg/{index.js,package.json} not found")
    text = idx.read_text(encoding="utf-8")
    exports = set(re.findall(r"^export\s+(?:async\s+)?(?:function|const|class)\s+(\w+)", text, re.M))
    exports.update(re.findall(r"^export\s*\{([^}]*)\}", text, re.M and re.S) and
                   [n.strip() for grp in re.findall(r"^export\s*\{([^}]*)\}", text, re.M | re.S)
                    for n in grp.split(",") if n.strip()])
    name = json.loads(pkg.read_text(encoding="utf-8")).get("name")
    return exports, name


# ------------------------------------------------------------------ tier A --
# A path token is "repo-relative" if it starts with one of these roots. Only
# these are existence-checked; bare filenames in examples (speech.wav,
# whisper-base.gguf) are user-supplied artifacts, not repo content.
REPO_PREFIXES = ("scripts/", "web/demo/", "web/pkg/", "bindings/", "tools/", "docs/", "include/",
                 "crates/", "tests/", "examples/")

CLI_TOKEN = re.compile(r"(?:^|[\s(])(?:\./)?(?:target/release/)?vokra-cli\b")


def _tokenize_invocation(lines, idx):
    """Join a shell invocation that uses trailing backslash continuations."""
    out = [lines[idx]]
    while out[-1].rstrip().endswith("\\") and idx + 1 < len(lines):
        idx += 1
        out.append(lines[idx])
    return " ".join(x.rstrip().rstrip("\\") for x in out), idx


def check_tier_a(block: Block, subs, kinds, root, problems):
    lines = block.body.splitlines()
    i = 0
    while i < len(lines):
        raw = lines[i]
        stripped = raw.strip()
        if stripped.startswith("#") or not stripped:
            i += 1
            continue

        if CLI_TOKEN.search(raw) or re.search(r"--\s+convert\b|--\s+run\b|--\s+bench\b", raw):
            joined, i = _tokenize_invocation(lines, i)
            toks = joined.split()
            # Find the subcommand: the token right after the vokra-cli word,
            # or after a bare `--` separator (`cargo run ... -- convert`).
            sub = None
            sub_idx = None
            for k, t in enumerate(toks):
                base = t.split("/")[-1]
                if base == "vokra-cli" or t == "--":
                    for off, cand in enumerate(toks[k + 1:], start=k + 1):
                        if cand.startswith("-"):
                            continue
                        sub, sub_idx = cand, off
                        break
                    if sub:
                        break
            if sub is not None:
                if sub not in subs:
                    problems.append(
                        f"{block.where()}: vokra-cli subcommand '{sub}' does not exist "
                        f"(have: {', '.join(sorted(subs))})"
                    )
                else:
                    # Only flags AFTER the subcommand belong to it. `cargo run
                    # --release --bin vokra-cli -- convert …` (docs/tutorials/
                    # python.md:37) puts cargo's own --release / --bin before
                    # the `--` separator; attributing those to `convert` was a
                    # false positive the self-test now pins.
                    for k, t in enumerate(toks):
                        if k < sub_idx:
                            continue
                        if t.startswith("--") and len(t) > 2:
                            flag = t.split("=")[0]
                            if flag not in subs[sub]:
                                problems.append(
                                    f"{block.where()}: `vokra-cli {sub}` has no flag {flag}"
                                )
                            elif flag == "--model" and sub == "convert" and k + 1 < len(toks):
                                kind = toks[k + 1]
                                if not kind.startswith("-") and kind not in kinds:
                                    problems.append(
                                        f"{block.where()}: `--model {kind}` is not a known "
                                        f"ModelKind (have: {', '.join(sorted(kinds))})"
                                    )
            i += 1
            continue

        # Repo-relative path existence (T19: ios.md's build-ios.sh /
        # verify-ios-xcframework.sh, web.md's build-wasm.sh + demo server).
        for tok in re.findall(r"[A-Za-z0-9_./-]+", raw):
            if tok.startswith(REPO_PREFIXES) and not tok.endswith("/"):
                if re.search(r"[*?]|\.\.\.|<|>", tok):
                    continue
                if not (root / tok).exists():
                    problems.append(f"{block.where()}: referenced repo path '{tok}' does not exist")
        i += 1


# ------------------------------------------------------------------ tier B --
def check_c_block(block: Block, root, problems, gaps):
    cc = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if cc is None:
        # FR-EX-08: announce, never silently pass.
        gaps.append(f"{block.where()}: no C compiler on this host — `c` block NOT compiled")
        return
    includes, rest = [], []
    for line in block.body.splitlines():
        (includes if line.strip().startswith("#include") else rest).append(line)
    # docs/getting-started.md's C block is a declaration FRAGMENT (statements at
    # file scope) that treats `pcm` / `num_samples` as caller-supplied inputs —
    # the surrounding prose says "link libvokra and call these". So the harness
    # wraps it in main() and declares exactly those two, with the types the
    # header itself specifies (const float* / size_t).
    #
    # The preamble is deliberately CLOSED: any other undeclared identifier a
    # future doc introduces still fails to compile, forcing a conscious
    # decision instead of silently widening the harness until it proves
    # nothing. Types come from vokra_asr_transcribe's real signature, so an
    # arity or type change in the C ABI still breaks this block.
    preamble = "  const float *pcm = NULL;\n  size_t num_samples = 0;\n"
    src = ("#include <stdio.h>\n#include <stddef.h>\n" + "\n".join(includes) +
           "\nint main(void) {\n" + preamble + "\n".join(rest) + "\n  return 0;\n}\n")
    with tempfile.TemporaryDirectory() as td:
        cfile = pathlib.Path(td) / "doc_example.c"
        cfile.write_text(src, encoding="utf-8")
        res = subprocess.run(
            [cc, "-fsyntax-only", "-I", str(root / "include"), str(cfile)],
            capture_output=True, text=True,
        )
        if res.returncode != 0:
            detail = res.stderr.strip().replace(str(cfile), block.where())
            problems.append(f"{block.where()}: `c` block does not compile against include/vokra.h:\n{detail}")


PY_IMPORT = re.compile(r"^from\s+vokra(?:\.\w+)?\s+import\s+(\(([^)]*)\)|(.+))$", re.M)


def check_python_block(block: Block, names, session_methods, problems, gaps):
    body = block.body
    # (1) Names imported from `vokra` must resolve somewhere in the package.
    for m in PY_IMPORT.finditer(body):
        raw = m.group(2) if m.group(2) is not None else (m.group(3) or "")
        raw = re.sub(r"#.*", "", raw)
        for nm in (x.strip() for x in raw.split(",")):
            if not nm or nm == "*":
                continue
            nm = nm.split(" as ")[0].strip()
            if nm not in names:
                problems.append(
                    f"{block.where()}: `from vokra import {nm}` — no such name in "
                    f"bindings/python/src/vokra/"
                )
    if PY_IMPORT.search(body):
        gaps.append(
            f"{block.where()}: names resolved package-wide "
            f"(KNOWN GAP python:import:vokra.__init__-reexport)"
        )

    # (2) Methods called on a Session-typed local must exist on Session.
    #     Free calls like `read_wav_mono_f32(f)` are NOT checked: a name that
    #     was never imported from vokra cannot be a vokra API, so it is a
    #     doc-local pseudo-helper by construction (T20's load-bearing rule).
    sess_vars = set(re.findall(r"(?:^|\s)(\w+)\s*=\s*Session\.open\(", body))
    sess_vars.update(re.findall(r"with\s+Session\.open\([^)]*\)\s+as\s+(\w+)", body))
    for var in sess_vars:
        for meth in re.findall(rf"\b{re.escape(var)}\.(\w+)\s*\(", body):
            if meth not in session_methods:
                problems.append(
                    f"{block.where()}: Session has no method '{meth}' "
                    f"(bindings/python/src/vokra/session.py)"
                )

    # (3) `vokra.<attr>` module attributes.
    for attr in re.findall(r"\bvokra\.(__\w+__)", body):
        if f"python:attr:{attr}" in KNOWN_GAPS:
            gaps.append(f"{block.where()}: vokra.{attr} (KNOWN GAP python:attr:{attr})")
        elif attr not in names:
            problems.append(f"{block.where()}: `vokra.{attr}` is not defined in the binding")


def check_js_block(block: Block, exports, pkg_name, problems):
    for m in re.finditer(r'import\s*\{([^}]*)\}\s*from\s*["\']([^"\']+)["\']', block.body):
        mod = m.group(2)
        if mod == pkg_name or mod == "@vokra/web":
            if mod != pkg_name:
                problems.append(
                    f"{block.where()}: imports from '{mod}' but web/pkg/package.json "
                    f"declares '{pkg_name}'"
                )
            for nm in (x.strip() for x in m.group(1).split(",")):
                if nm and nm not in exports:
                    problems.append(
                        f"{block.where()}: '{nm}' is not exported by web/pkg/index.js"
                    )


def check_json_block(block: Block, root, problems):
    try:
        data = json.loads(block.body)
    except json.JSONDecodeError as e:
        problems.append(f"{block.where()}: `json` block does not parse — {e}")
        return
    # unity.md's manifest snippet points at the UPM package by relative path;
    # verify the package it names is really in the tree.
    for val in _walk_strings(data):
        if val.startswith("file:"):
            tail = val.split("/")[-1]
            if tail and not any(p.name == tail for p in (root / "bindings/unity").glob("*")):
                problems.append(
                    f"{block.where()}: manifest references '{tail}' which is not under bindings/unity/"
                )


def _walk_strings(node):
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for v in node.values():
            yield from _walk_strings(v)
    elif isinstance(node, list):
        for v in node:
            yield from _walk_strings(v)


# ---------------------------------------------------------------- driver ----
def run_check(root: pathlib.Path, docs, listing=False):
    problems: list[str] = []
    gaps: list[str] = []
    deferred: list[str] = []
    blocks: list[Block] = []

    for rel in docs:
        p = root / rel
        if not p.is_file():
            problems.append(f"setup error: doc {rel} not found")
            continue
        blocks.extend(extract_blocks(p, rel))

    if listing:
        for b in blocks:
            print(f"{b.doc}:{b.start}-{b.end}  [{b.lang}]")
        print(f"total blocks: {len(blocks)}")
        return 0

    subs, kinds = cli_surface(root)
    py_names, sess_methods = python_surface(root)
    js_exports, js_name = js_surface(root)

    for b in blocks:
        if b.lang in TIER_C_LANGS:
            deferred.append(f"{b.where()} [{b.lang}] — {TIER_C_LANGS[b.lang]}")
            continue
        if b.lang == "sh":
            check_tier_a(b, subs, kinds, root, problems)
        elif b.lang == "c":
            check_c_block(b, root, problems, gaps)
        elif b.lang == "python":
            check_tier_a(b, subs, kinds, root, problems)  # paths inside comments too
            check_python_block(b, py_names, sess_methods, problems, gaps)
        elif b.lang == "js":
            check_js_block(b, js_exports, js_name, problems)
        elif b.lang == "json":
            check_json_block(b, root, problems)
        else:
            problems.append(
                f"{b.where()}: unhandled block language '{b.lang}' — add it to a tier "
                f"(silently ignoring it would be a fabricated pass)"
            )

    # Anti-rot: a pinned gap that no longer applies must be deleted.
    if "__abi_version__" in py_names:
        problems.append(
            "KNOWN_GAPS entry 'python:attr:__abi_version__' is stale — the binding now "
            "defines it. Delete the entry from tools/docs/check_doc_examples.py."
        )
    # Read the DECLARED export list, not the whole file: __init__.py's docstring
    # names Session/Stream/VokraError in prose ("populated by later tickets"),
    # so a substring test over the file would report the gap as closed while it
    # is still open.
    init_src = (root / "bindings/python/src/vokra/__init__.py").read_text(encoding="utf-8")
    m_all = re.search(r"^__all__\s*=\s*\[([^\]]*)\]", init_src, re.M)
    declared = {x.strip().strip("\"'") for x in (m_all.group(1).split(",") if m_all else [])}
    if "Session" in declared:
        problems.append(
            "KNOWN_GAPS entry 'python:import:vokra.__init__-reexport' is stale — "
            "__init__.py now exports Session. Delete the entry and switch the "
            "python import check to __init__-level resolution."
        )

    print(f"checked {len(blocks)} block(s) across {len(docs)} doc(s)")
    if deferred:
        print(f"\nTIER C — deferred, NOT verified ({len(deferred)}):")
        for d in deferred:
            print(f"  {d}")
    if gaps:
        print(f"\nANNOUNCED GAPS — checked partially ({len(gaps)}):")
        for g in gaps:
            print(f"  {g}")
        for key, (reason, ticket) in KNOWN_GAPS.items():
            print(f"  pinned: {key}\n      reason: {reason}\n      see: {ticket}")
    if problems:
        print(f"\ncheck-doc-examples: FAIL ({len(problems)})", file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        return 1
    print("\ncheck-doc-examples: OK")
    return 0


# -------------------------------------------------------------- self-test ---
GREEN_DOC = '''# Fixture

```sh
./target/release/vokra-cli convert \\
  --model whisper \\
  --input model.safetensors \\
  --output whisper.gguf
scripts/build-ios.sh
```

```python
from vokra import Session

with Session.open("m.gguf") as s:
    # `read_wav_mono_f32` is a doc-local pseudo-helper, NOT a vokra API.
    pcm, sr = read_wav_mono_f32(open("a.wav", "rb"))
    text = s.transcribe(pcm, sr)
```
'''

RED_DOCS = {
    "bad-flag": '```sh\nvokra-cli convert --model whisper --input a --outpt b.gguf\n```\n',
    "bad-sub": '```sh\nvokra-cli transcribe --model whisper\n```\n',
    "bad-kind": '```sh\nvokra-cli convert --model wisper --input a --output b.gguf\n```\n',
    "bad-path": '```sh\nscripts/build-nonexistent-thing.sh\n```\n',
    "bad-import": '```python\nfrom vokra import NoSuchSymbol\n```\n',
    "bad-method": (
        '```python\nfrom vokra import Session\n'
        's = Session.open("m.gguf")\ns.transcrybe(pcm, 16000)\n```\n'
    ),
    "bad-json": '```json\n{ "dependencies": { oops }\n```\n',
    "bad-lang": '```ruby\nputs "hi"\n```\n',
}


def self_test(root: pathlib.Path) -> int:
    rc = 0
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)
        # (a) GREEN side (T24): a legitimate doc-local pseudo-helper must NOT
        #     be mistaken for API drift, and a valid invocation must pass.
        gd = tmp / "green.md"
        gd.write_text(GREEN_DOC, encoding="utf-8")
        shutil.copytree(root / "crates", tmp / "crates", dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("target"))
        shutil.copytree(root / "bindings", tmp / "bindings", dirs_exist_ok=True)
        shutil.copytree(root / "web", tmp / "web", dirs_exist_ok=True)
        shutil.copytree(root / "include", tmp / "include", dirs_exist_ok=True)
        (tmp / "scripts").mkdir(exist_ok=True)
        shutil.copy(root / "scripts/build-ios.sh", tmp / "scripts/build-ios.sh")

        if run_check(tmp, ["green.md"]) != 0:
            print("self-test FAILED: the green fixture (doc-local helper) should pass", file=sys.stderr)
            rc = 1

        # (b) RED side: each fixture must be detected.
        for name, text in RED_DOCS.items():
            rd = tmp / f"red-{name}.md"
            rd.write_text(text, encoding="utf-8")
            if run_check(tmp, [rd.name]) == 0:
                print(f"self-test FAILED: red fixture '{name}' should have failed", file=sys.stderr)
                rc = 1
    if rc == 0:
        print("check-doc-examples --self-test: OK")
    return rc


def main() -> int:
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--root", default=str(ROOT))
    args = ap.parse_args()
    root = pathlib.Path(args.root).resolve()
    if args.self_test:
        return self_test(root)
    return run_check(root, DOCS, listing=args.list)


if __name__ == "__main__":
    sys.exit(main())
