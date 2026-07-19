#!/usr/bin/env bash
# check-fixture-eol-pins.sh
#
# Assert that every byte-hashed fixture has an EXPLICIT .gitattributes eol pin.
#
# WHY
# ---
# Fixtures whose SHA-256 is pinned in a manifest or a Rust `expected_sha256_hex`
# table must arrive on disk byte-identical on every platform. On the GitHub
# `windows-latest` runners `core.autocrlf=true` is the default, so any file git
# considers text is rewritten LF -> CRLF at checkout — changing its bytes and
# breaking a hash nobody touched.
#
# This has already fired twice on this branch: first on *.wgsl ("WGSL source
# drift for copy_f32"), then on csm/self-test/context.json. Each time the fix
# was to pin one more extension, which is whack-a-mole: the gate below finds
# the NEXT unpinned hashed fixture at commit time instead of on a Windows
# runner an hour later.
#
# Git's binary heuristic (NUL byte in the first 8000 bytes) is NOT a substitute.
# Hash sidecars and manifests are pure ASCII, so the heuristic never fires for
# them, and relying on it for binaries means the protection silently depends on
# whether a float fixture happens to contain a zero byte.
#
# WHAT COUNTS AS "HASHED" (all derived from the repo, not a hardcoded list)
#   (a) every *.sha256 sidecar, plus the file it names
#   (b) every manifest / SHA256SUMS / PROVENANCE ledger, plus every tracked file
#       its rows reference (token-scanned, so the row format is irrelevant)
#   (c) every file under the pinned kernel trees, whose blobs are hashed by
#       crates/vokra-backend-{webgpu,vulkan}/src/{wgsl,spirv}.rs at cargo-test time
#
# ACCEPTED VERDICTS
#   text: unset            -> OK   (`-text`, binary; never converted)
#   text: set + eol: lf    -> OK   (text, pinned to LF)
#   text: unspecified      -> FAIL (falls through to core.autocrlf)
#   text: set + eol: != lf -> FAIL (ambiguous; core.eol decides)
#   text: auto             -> FAIL (the NUL heuristic we are refusing to trust)
#
# The file set comes from `git ls-files`, so gitignored local-only trees
# (docs/tickets/, docs/adr/, models/, ...) are structurally invisible and no
# ignore list is needed.
#
# Usage: bash scripts/check-fixture-eol-pins.sh [--list]
#   --list  print the derived fixture set and exit 0 (no assertions)

set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

MODE="${1:-check}"

python3 - "$MODE" <<'PY'
import subprocess
import sys
from pathlib import PurePosixPath

mode = sys.argv[1] if len(sys.argv) > 1 else "check"


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, check=True
    ).stdout


tracked = set(git("ls-files", "-z").split("\0"))
tracked.discard("")

# Kernel trees whose blobs/sources are hash-pinned in Rust at cargo-test time.
KERNEL_TREES = (
    "crates/vokra-backend-webgpu/kernels/",
    "crates/vokra-backend-vulkan/kernels/",
)
# Only the shader blobs and their sources are hashed. Deliberately excluded:
#   *.md          — READMEs in those trees, not hashed
#   *.spv.rs      — kernels/handcrafted/*.spv.rs are Rust modules pulled in with
#                   #[path] (spirv.rs L75-77), i.e. compiled source, not a
#                   hash-pinned blob. CRLF would not break a digest.
KERNEL_EXTS = (".wgsl", ".comp", ".spv")

LEDGER_NAMES = ("SHA256SUMS", "PROVENANCE")
# "manifest" as a bare name prefix is far too common to key on: it would drag in
# crates/vokra-eval/src/manifest.rs (a manifest *reader*) and
# examples/unity-demo/Packages/manifest.json (Unity's UPM dependency list),
# neither of which records a digest. Require the fixture tree as well.
LEDGER_TREES = ("tests/",)
LEDGER_SUFFIXES = (".txt", ".json")

# path -> human-readable reason it must be pinned (for the failure message)
hashed: dict[str, str] = {}


def add(path: str, why: str) -> None:
    if path in tracked:
        hashed.setdefault(path, why)


for path in sorted(tracked):
    name = PurePosixPath(path).name

    # (a) hash sidecars: the sidecar is read byte-wise by `sha256sum -c`, and
    #     the file it names is the thing being hashed.
    if path.endswith(".sha256"):
        add(path, "hash sidecar (parsed by `sha256sum -c`)")
        add(path[: -len(".sha256")], f"digest pinned in {path}")
        continue

    # (c) pinned kernel trees.
    if path.startswith(KERNEL_TREES) and path.endswith(KERNEL_EXTS):
        add(path, "blob SHA-256 pinned in vokra-backend-*/src (cargo test)")
        continue

    # (b) manifests / ledgers.
    if name in LEDGER_NAMES or (
        name.startswith("manifest")
        and name.endswith(LEDGER_SUFFIXES)
        and path.startswith(LEDGER_TREES)
    ):
        add(path, "hash ledger (rows compared byte-wise)")

# Second pass: resolve the files each ledger references. Rows are token-scanned
# rather than parsed, so `sha256 <name> <hex>` (manifest.txt), `<hex>  <name>`
# (SHA256SUMS) and `source_sha256=<name>:<hex>` (PROVENANCE) all work, and a
# future row format needs no change here.
for ledger in [p for p, why in list(hashed.items()) if "ledger" in why]:
    base = PurePosixPath(ledger).parent
    try:
        text = open(ledger, encoding="utf-8", errors="replace").read()
    except OSError:
        continue
    for raw in text.split():
        for token in (raw, *raw.split("=")[-1:], *raw.split(":")[:1]):
            token = token.strip(",;'\"")
            if not token or "." not in token:
                continue
            add(str(base / token), f"digest pinned in {ledger}")

if not hashed:
    print("check-fixture-eol-pins: FAIL — derived an empty fixture set", file=sys.stderr)
    print("  (the derivation rules matched nothing; that is a bug in this gate,", file=sys.stderr)
    print("   not a clean repo — refusing to report success)", file=sys.stderr)
    sys.exit(1)

if mode == "--list":
    for path in sorted(hashed):
        print(f"{path}\n    {hashed[path]}")
    print(f"\n{len(hashed)} byte-hashed files derived")
    sys.exit(0)

# One batched check-attr for the whole set.
paths = sorted(hashed)
out = subprocess.run(
    ["git", "check-attr", "--stdin", "-z", "text", "eol"],
    input="\0".join(paths) + "\0",
    capture_output=True,
    text=True,
    check=True,
).stdout

fields = out.split("\0")
attrs: dict[str, dict[str, str]] = {}
for i in range(0, len(fields) - 2, 3):
    path, attr, value = fields[i], fields[i + 1], fields[i + 2]
    attrs.setdefault(path, {})[attr] = value

bad: list[tuple[str, str]] = []
for path in paths:
    a = attrs.get(path, {})
    text_v, eol_v = a.get("text", "unspecified"), a.get("eol", "unspecified")
    if text_v == "unset":
        continue  # `-text`: never converted
    if text_v == "set" and eol_v == "lf":
        continue  # explicit LF
    if text_v == "unspecified":
        bad.append((path, "no `text` attribute — falls through to core.autocrlf"))
    elif text_v == "auto":
        bad.append((path, "`text=auto` relies on the NUL-byte heuristic"))
    elif text_v == "set":
        bad.append((path, f"`text` set but eol={eol_v} — core.eol decides the bytes"))
    else:
        bad.append((path, f"unexpected text={text_v} eol={eol_v}"))

if bad:
    print("check-fixture-eol-pins: FAIL — byte-hashed file(s) without an explicit eol pin:\n", file=sys.stderr)
    for path, why in bad:
        print(f"  {path}", file=sys.stderr)
        print(f"      {why}", file=sys.stderr)
        print(f"      hashed because: {hashed[path]}", file=sys.stderr)
    print("\nAdd a rule to .gitattributes:", file=sys.stderr)
    print("    <glob> text eol=lf   # text fixtures, manifests, hash sidecars", file=sys.stderr)
    print("    <glob> -text         # binary fixtures (.f32/.u32/.spv/.wav/.gguf)", file=sys.stderr)
    print("\nA CRLF checkout on windows-latest changes these files' bytes and", file=sys.stderr)
    print("breaks the SHA-256 pin on a file nobody edited.", file=sys.stderr)
    sys.exit(1)

print(f"check-fixture-eol-pins: OK ({len(paths)} byte-hashed files, all explicitly pinned)")
PY
