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
          blocks (Unity/C#), `gdscript` blocks (Godot runtime), and the `sh`
          blocks that download real upstream checkpoints over the network and
          run real models. Those belong to the gated / nightly workflows
          (parity-*-real.yml, nightly-*.yml), not to a per-PR job.

PYTHON ROOT-SURFACE RULE
------------------------
`from vokra import Name` is checked against the names deliberately exported by
`vokra.__all__`, while imports from a concrete submodule are checked against
the package-wide source surface. There is no Python known-gap bypass: a root
re-export promised by a document must exist now or the document check fails.

Usage:
    uv run --no-project --python 3.12 python \
        tools/docs/check_doc_examples.py [--list | --self-test]
        --list       print the extracted block inventory and exit 0
        --self-test  run the bidirectional fixture tests (T24) and exit

Exit code: 0 = clean (announced tier-C / pinned gaps are not failures),
           1 = a drift was found, 2 = usage / setup error.
"""

from __future__ import annotations

import argparse
import ast
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
    # X-09b doc-hierarchy completion: backend-guide + the four platform
    # tutorials X-09a left out (cli / android / godot / server) + the
    # api-reference index, each en/ja. Registered here so their shell / C /
    # JSON examples are execution-verified too (NFR-MT-04, X-09b-T23). The
    # Rust snippets in these docs are intentionally untagged. GDScript is an
    # explicit tier-C language below: its examples are announced as deferred
    # here and exercised by the dedicated Godot headless workflow.
    "docs/backend-guide.md",
    "docs/backend-guide.ja.md",
    "docs/tutorials/cli.md",
    "docs/tutorials/cli.ja.md",
    "docs/tutorials/android.md",
    "docs/tutorials/android.ja.md",
    "docs/tutorials/godot.md",
    "docs/tutorials/godot.ja.md",
    "docs/tutorials/server.md",
    "docs/tutorials/server.ja.md",
    "docs/api-reference.md",
    "docs/api-reference.ja.md",
]

# Languages we defer, with the reason announced on every run.
TIER_C_LANGS = {
    "swift": "needs a Swift compiler (owner / macOS toolchain)",
    "csharp": "needs the Unity C# toolchain (nightly-il2cpp.yml)",
    "gdscript": "needs the Godot runtime (godot-headless.yml)",
    "(none)": "untagged prose block (UI steps / HTTP headers) — nothing to check",
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
    # The subcommand set is DERIVED from main.rs's dispatch, never hand-listed.
    # It used to be a literal tuple of (run, convert, bench), which went stale
    # the moment `f0` landed: the gate then rejected correct documentation of a
    # real subcommand, and — worse — could not have checked a single one of its
    # flags. A hand-maintained mirror of the code is exactly the shape this
    # repo keeps finding rotted, so the mirror is gone.
    main_rs = root / "crates/vokra-cli/src/main.rs"
    if not main_rs.is_file():
        raise SystemExit("setup error: crates/vokra-cli/src/main.rs not found")
    dispatch = re.findall(
        r'"([a-z0-9-]+)"\s*=>\s*([a-z0-9_]+)::main\(', main_rs.read_text(encoding="utf-8")
    )
    if not dispatch:
        raise SystemExit(
            "setup error: parsed no `\"<sub>\" => <mod>::main(` arms out of "
            "crates/vokra-cli/src/main.rs — the dispatch shape changed, and "
            "returning an empty subcommand set here would pass every doc"
        )

    subs = {}
    for sub, module in dispatch:
        rel = f"crates/vokra-cli/src/{module}.rs"
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
    """Return package-wide names, root exports, and Session methods."""
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

    init_path = pkg / "__init__.py"
    init_tree = ast.parse(init_path.read_text(encoding="utf-8"), filename=str(init_path))
    init_exports: set[str] = set()
    for node in init_tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets):
            continue
        if not isinstance(node.value, (ast.List, ast.Tuple)):
            raise SystemExit("setup error: vokra.__all__ must be a literal list or tuple")
        for item in node.value.elts:
            if not isinstance(item, ast.Constant) or not isinstance(item.value, str):
                raise SystemExit("setup error: every vokra.__all__ entry must be a string literal")
            init_exports.add(item.value)
    if not init_exports:
        raise SystemExit("setup error: parsed no names from vokra.__all__")
    return names, init_exports, session_methods


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


def _contains_cli_invocation(line: str) -> bool:
    """Return whether a shell line invokes the CLI rather than naming its crate.

    Cargo selectors such as ``-p vokra-cli``, ``--package vokra-cli``, and
    ``--bin vokra-cli`` name a build target.  Treating the next non-flag token
    as a Vokra subcommand turns ``--features metal`` into a bogus
    ``vokra-cli metal`` invocation.  A ``cargo run ... -- <subcommand>`` line
    remains an invocation through its argument separator.
    """
    if re.search(r"\bcargo\s+run\b.*(?:^|\s)--(?:\s|$)", line):
        return True
    for match in CLI_TOKEN.finditer(line):
        prefix = line[:match.start()].rstrip()
        if re.search(r"(?:^|\s)(?:-p|--package|--bin)$", prefix):
            continue
        return True
    return False


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

        joined, joined_end = _tokenize_invocation(lines, i)
        if _contains_cli_invocation(joined):
            i = joined_end
            toks = joined.split()
            # Find the subcommand: the token right after the vokra-cli word,
            # or after a bare `--` separator (`cargo run ... -- convert`).
            sub = None
            sub_idx = None
            for k, t in enumerate(toks):
                base = t.split("/")[-1]
                is_cargo_selector = (
                    base == "vokra-cli"
                    and k > 0
                    and toks[k - 1] in {"-p", "--package", "--bin"}
                )
                if is_cargo_selector:
                    continue
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
                    # --release -p vokra-cli --features metal -- convert …`
                    # puts cargo's own flags before the `--` separator;
                    # attributing those to `convert` would be a false positive.
                    for k, t in enumerate(toks):
                        if k < sub_idx:
                            continue
                        if t.startswith("--") and len(t) > 2:
                            flag = t.split("=")[0]
                            if flag not in subs[sub]:
                                problems.append(
                                    f"{block.where()}: `vokra-cli {sub}` has no flag {flag}"
                                )
                            elif flag == "--model" and sub == "convert":
                                kind = (
                                    t.split("=", 1)[1]
                                    if "=" in t
                                    else (toks[k + 1] if k + 1 < len(toks) else "")
                                )
                                if not kind.startswith("-") and kind not in kinds:
                                    problems.append(
                                        f"{block.where()}: `--model {kind}` is not a known "
                                        f"ModelKind (have: {', '.join(sorted(kinds))})"
                                    )
            i += 1
            continue

        # Repo-relative path existence (T19: ios.md's build-ios.sh /
        # verify-ios-xcframework.sh, web.md's build-wasm.sh + demo server).
        for tok in re.findall(r"[A-Za-z0-9_./-]+", joined):
            if tok.startswith(REPO_PREFIXES) and not tok.endswith("/"):
                if re.search(r"[*?]|\.\.\.|<|>", tok):
                    continue
                if not (root / tok).exists():
                    problems.append(f"{block.where()}: referenced repo path '{tok}' does not exist")
        i = joined_end + 1


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


PY_IMPORT = re.compile(
    r"^from\s+(vokra(?:\.\w+)?)\s+import\s+(\(([^)]*)\)|(.+))$", re.M
)


def check_python_block(
    block: Block,
    names,
    init_exports,
    session_methods,
    problems,
):
    body = block.body
    try:
        ast.parse(body, filename=block.where(), mode="exec")
    except SyntaxError as error:
        problems.append(
            f"{block.where()}: `python` block does not parse — "
            f"{error.msg} (line {error.lineno})"
        )
        return

    # (1) Root imports must be deliberate __all__ exports. Concrete-submodule
    # imports retain package-wide resolution until this lightweight checker
    # grows a per-module AST index.
    for m in PY_IMPORT.finditer(body):
        module = m.group(1)
        raw = m.group(3) if m.group(3) is not None else (m.group(4) or "")
        allowed = init_exports if module == "vokra" else names
        raw = re.sub(r"#.*", "", raw)
        for nm in (x.strip() for x in raw.split(",")):
            if not nm or nm == "*":
                continue
            nm = nm.split(" as ")[0].strip()
            if nm in allowed:
                continue
            if module == "vokra":
                problems.append(
                    f"{block.where()}: `from vokra import {nm}` — {nm!r} is not "
                    "exported by bindings/python/src/vokra/__init__.py::__all__"
                )
            else:
                problems.append(
                    f"{block.where()}: `from {module} import {nm}` — no such name "
                    "in bindings/python/src/vokra/"
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
        if attr not in init_exports:
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
    py_names, init_exports, sess_methods = python_surface(root)
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
            check_python_block(b, py_names, init_exports, sess_methods, problems)
        elif b.lang == "js":
            check_js_block(b, js_exports, js_name, problems)
        elif b.lang == "json":
            check_json_block(b, root, problems)
        else:
            problems.append(
                f"{b.where()}: unhandled block language '{b.lang}' — add it to a tier "
                f"(silently ignoring it would be a fabricated pass)"
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
cargo build --release -p vokra-cli --features metal
cargo build --release --package vokra-cli --features cuda
cargo build --release --package=vokra-cli --features metal
cargo build --release --bin vokra-cli
cargo build --release -p \\
  vokra-cli --features metal
./target/release/vokra-cli convert \\
  --model=whisper \\
  --input model.safetensors \\
  --output whisper.gguf
cargo run --release -p vokra-cli --features metal -- convert \\
  --model whisper \\
  --input model.safetensors \\
  --output whisper.gguf
scripts/build-ios.sh
```

```python
from vokra import Session
from vokra.errors import VokraError

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
    "bad-kind-equals": '```sh\nvokra-cli convert --model=wisper --input a --output b.gguf\n```\n',
    "bad-multiline-cargo-run-flag": (
        '```sh\ncargo run --release -p vokra-cli \\\n'
        '  --features metal \\\n'
        '  -- convert --model whisper --outpt b.gguf\n```\n'
    ),
    "bad-path": '```sh\nscripts/build-nonexistent-thing.sh\n```\n',
    "bad-import": '```python\nfrom vokra import NoSuchSymbol\n```\n',
    "bad-root-reexport": '```python\nfrom vokra import vokra_event_t\n```\n',
    "bad-python-syntax": '```python\ndef broken(:\n    pass\n```\n',
    "bad-method": (
        '```python\nfrom vokra import Session\n'
        's = Session.open("m.gguf")\ns.transcrybe(pcm, 16000)\n```\n'
    ),
    "bad-json": '```json\n{ "dependencies": { oops }\n```\n',
    "bad-lang": '```ruby\nputs "hi"\n```\n',
}


def check_tutorial_wav_helper(root: pathlib.Path) -> list[str]:
    """Execute the WAV helper directly from both tutorial code fences."""
    problems = []
    functions = []
    for rel in ("docs/tutorials/python.md", "docs/tutorials/python.ja.md"):
        blocks = extract_blocks(root / rel, rel)
        candidates = [
            block for block in blocks
            if block.lang == "python" and "def read_pcm16_wav_mono" in block.body
        ]
        if len(candidates) != 1:
            problems.append(f"{rel}: expected exactly one read_pcm16_wav_mono example")
            continue
        tree = ast.parse(candidates[0].body, filename=rel, mode="exec")
        function_nodes = [
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "read_pcm16_wav_mono"
        ]
        if len(function_nodes) != 1:
            problems.append(f"{rel}: WAV helper definition was not parseable as one function")
            continue
        functions.append((rel, function_nodes[0], tree))

    if len(functions) != 2:
        return problems
    if ast.dump(functions[0][1], include_attributes=False) != ast.dump(
        functions[1][1], include_attributes=False
    ):
        problems.append("Python tutorial WAV helpers differ between English and Japanese")

    namespace = {}
    executable = ast.Module(
        body=[
            node for node in functions[0][2].body
            if isinstance(node, (ast.Import, ast.FunctionDef))
        ],
        type_ignores=[],
    )
    exec(compile(executable, functions[0][0], "exec"), namespace)
    helper = namespace["read_pcm16_wav_mono"]
    struct_module = namespace["struct"]
    wave_module = namespace["wave"]

    with tempfile.TemporaryDirectory() as td:
        mono = pathlib.Path(td) / "mono.wav"
        with wave_module.open(str(mono), "wb") as sink:
            sink.setnchannels(1)
            sink.setsampwidth(2)
            sink.setframerate(16000)
            sink.writeframes(struct_module.pack("<hhh", -32768, 0, 32767))
        pcm, sample_rate = helper(str(mono))
        expected = [-1.0, 0.0, 32767 / 32768.0]
        if pcm != expected or sample_rate != 16000:
            problems.append(
                "Python tutorial WAV helper returned the wrong PCM normalization/sample rate"
            )

        stereo = pathlib.Path(td) / "stereo.wav"
        with wave_module.open(str(stereo), "wb") as sink:
            sink.setnchannels(2)
            sink.setsampwidth(2)
            sink.setframerate(8000)
            sink.writeframes(struct_module.pack("<hhhh", 0, 0, 1, -1))
        try:
            helper(str(stereo))
        except ValueError:
            pass
        else:
            problems.append("Python tutorial WAV helper accepted stereo input")

    return problems


def self_test(root: pathlib.Path) -> int:
    rc = 0
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)
        # (a) GREEN side (T24): a legitimate doc-local pseudo-helper must NOT
        #     be mistaken for API drift; cargo target selectors must NOT be
        #     mistaken for CLI invocations; valid direct and cargo-run
        #     invocations must pass.
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
        for problem in check_tutorial_wav_helper(root):
            print(f"self-test FAILED: {problem}", file=sys.stderr)
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
