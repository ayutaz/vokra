#!/usr/bin/env python3
"""generate_spdx.py — first-party SPDX 2.3 SBOM generator (M4-15-T06/T07).

Design (fixed by ADR M4-15 §(b)(c)):

* SPDX is the authoritative SBOM format (NFR-MT-03 / X-07). The
  scope-expansion doc's cargo-cyclonedx suggestion (CycloneDX output) is
  resolved by emitting SPDX 2.3 JSON natively — no conversion step.
* `scripts/` isolation, first-party implementation: python3 standard
  library only, driving `cargo tree` (a built-in command of the pinned
  stable toolchain). NO third-party SBOM crate is installed, so the root
  Cargo.lock cannot change (NFR-DS-02) and no unverifiable upstream CLI
  flag is baked in (CLAUDE.md hallucination red-line). The dependency
  graph this SBOM documents is vokra-*-only, which is exactly why a
  third-party tool is not worth its supply-chain surface here.
* Deterministic output (precondition for the owner's reproducible-build
  verify, M4-15-T10):
    - `created` comes from SOURCE_DATE_EPOCH, else the HEAD commit
      timestamp — never the wall clock;
    - `documentNamespace` is derived from the git SHA — no UUID;
    - JSON is emitted with sorted keys; packages / relationships are
      sorted by name; `cargo tree` absolute paths are NOT copied into
      the document (machine-independent output).
  Same commit + same toolchain + same feature selection => byte-identical
  document (asserted by scripts/sbom/test-generate-spdx.sh b4/b5).

The graph source is:

    cargo tree -p <package> [--no-default-features] [--features <list>]
               -e normal --prefix none

`-e normal` restricts the closure to runtime dependencies (no dev/build
deps) — the correct scope for an SBOM of the shipped artifact. The
`--prefix none` line format is `<name> v<version> (<path>)` with a
trailing ` (*)` on de-duplicated repeats; the parser strips the marker
and de-duplicates on (name, version).

Failure policy (FR-EX-08 spirit): any parse anomaly, empty graph, missing
git SHA, or cargo failure exits non-zero with the reason — the tool never
emits a plausible-but-wrong document.

Usage:
    python3 scripts/sbom/generate_spdx.py \
        --package vokra-capi --no-default-features --features vulkan \
        --doc-name vokra-capi-cpu-vulkan-only \
        --output dist/cpu-vulkan-only/vokra-capi.spdx.json
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import subprocess
import sys

REPO_URL = "https://github.com/ayutaz/vokra"

# `cargo tree --prefix none` line: "<name> v<version> (<abs path>)[ (*)]".
TREE_LINE = re.compile(r"^(?P<name>\S+) v(?P<version>\S+)(?: \([^)]*\))?(?: \(\*\))?$")


def fail(msg: str) -> "NoReturn":  # noqa: F821 - py3.9-compatible annotation
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(1)


def run(cmd: list[str], cwd: str) -> str:
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if proc.returncode != 0:
        fail(
            f"command failed ({' '.join(cmd)}):\n{proc.stderr.strip()}"
        )
    return proc.stdout


def repo_root() -> str:
    return os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
    )


def git_head_sha(root: str) -> str:
    return run(["git", "rev-parse", "HEAD"], cwd=root).strip()


def created_timestamp(root: str) -> str:
    """SOURCE_DATE_EPOCH, else HEAD commit time — never the wall clock."""
    epoch_env = os.environ.get("SOURCE_DATE_EPOCH")
    if epoch_env is not None:
        try:
            epoch = int(epoch_env)
        except ValueError:
            fail(f"SOURCE_DATE_EPOCH is not an integer: {epoch_env!r}")
    else:
        epoch = int(run(["git", "show", "-s", "--format=%ct", "HEAD"], cwd=root).strip())
    return (
        datetime.datetime.fromtimestamp(epoch, tz=datetime.timezone.utc)
        .strftime("%Y-%m-%dT%H:%M:%SZ")
    )


def resolve_graph(
    root: str,
    package: str,
    features: list[str],
    no_default_features: bool,
    tree_cwd: str | None = None,
) -> list[tuple[str, str]]:
    # `--color never`: CI exports CARGO_TERM_COLOR=always, which makes cargo
    # wrap the dedup marker in ANSI escapes ("... (/path) \x1b[33m\x1b[2m(*)")
    # and TREE_LINE then rejects the line as format drift. It only shows up
    # once a package appears twice in the graph, so it stayed latent until
    # vokra-mmap became a shared dependency.
    #
    # `tree_cwd` (X-07-T02): the excluded integration workspaces
    # (`integrations/vokra-godot` et al.) carry their OWN Cargo.toml / Cargo.lock
    # and are invisible to the root manifest, so `cargo tree` must run WITH THAT
    # DIRECTORY AS CWD to resolve their closure. Only the `cargo tree` invocation
    # moves; the git SHA + SOURCE_DATE_EPOCH determinism still derive from the
    # repository root (`root`), because the excluded workspace lives inside the
    # same git checkout. Defaults to `root` (the root workspace) when unset.
    cmd = [
        "cargo",
        "tree",
        "-p",
        package,
        "-e",
        "normal",
        "--prefix",
        "none",
        "--color",
        "never",
    ]
    if no_default_features:
        cmd.append("--no-default-features")
    if features:
        cmd += ["--features", ",".join(features)]
    out = run(cmd, cwd=tree_cwd or root)

    pkgs: dict[str, str] = {}
    for raw in out.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = TREE_LINE.match(line)
        if not m:
            fail(f"unparseable `cargo tree` line (format drift?): {line!r}")
        name, version = m.group("name"), m.group("version")
        prev = pkgs.get(name)
        if prev is not None and prev != version:
            fail(f"two versions of {name} in one graph: {prev} vs {version}")
        pkgs[name] = version

    if not pkgs:
        fail("`cargo tree` returned an empty graph")
    if package not in pkgs:
        fail(f"root package {package} missing from its own graph")
    return sorted(pkgs.items())


def spdxid_for(name: str) -> str:
    # SPDX idstring letters/digits/./- ; crate names are already conformant,
    # but sanitize defensively so a future crate name cannot corrupt the doc.
    return "SPDXRef-Package-" + re.sub(r"[^A-Za-z0-9.\-]", "-", name)


def build_document(
    doc_name: str,
    package: str,
    graph: list[tuple[str, str]],
    sha: str,
    created: str,
    feature_comment: str,
) -> dict:
    packages = []
    for name, version in graph:
        first_party = name.startswith("vokra")
        packages.append(
            {
                "SPDXID": spdxid_for(name),
                "name": name,
                "versionInfo": version,
                # First-party crates come from this repository at this
                # commit; anything else is recorded honestly as
                # NOASSERTION (no invented registry URLs).
                "downloadLocation": (
                    f"git+{REPO_URL}@{sha}" if first_party else "NOASSERTION"
                ),
                "filesAnalyzed": False,
                "licenseConcluded": "Apache-2.0" if first_party else "NOASSERTION",
                "licenseDeclared": "Apache-2.0" if first_party else "NOASSERTION",
                "copyrightText": (
                    "Copyright 2026 ayutaz" if first_party else "NOASSERTION"
                ),
                "supplier": "Person: ayutaz" if first_party else "NOASSERTION",
            }
        )

    root_id = spdxid_for(package)
    relationships = [
        {
            "spdxElementId": "SPDXRef-DOCUMENT",
            "relationshipType": "DESCRIBES",
            "relatedSpdxElement": root_id,
        }
    ]
    for name, _version in graph:
        if name == package:
            continue
        relationships.append(
            {
                "spdxElementId": root_id,
                "relationshipType": "DEPENDS_ON",
                "relatedSpdxElement": spdxid_for(name),
            }
        )

    return {
        "spdxVersion": "SPDX-2.3",
        "dataLicense": "CC0-1.0",
        "SPDXID": "SPDXRef-DOCUMENT",
        "name": doc_name,
        "documentNamespace": f"{REPO_URL}/spdx/{doc_name}/{sha}",
        "creationInfo": {
            "created": created,
            "creators": ["Tool: vokra-generate-spdx"],
            "comment": feature_comment,
        },
        "packages": packages,
        "relationships": relationships,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--package", required=True, help="root package (e.g. vokra-capi)")
    ap.add_argument(
        "--features",
        default="",
        help="comma-separated cargo features (e.g. vulkan)",
    )
    ap.add_argument(
        "--no-default-features",
        action="store_true",
        help="pass --no-default-features to cargo tree",
    )
    ap.add_argument(
        "--doc-name",
        default=None,
        help="SPDX document name (default: <package>-sbom)",
    )
    ap.add_argument("--output", required=True, help="output path for the SPDX JSON")
    ap.add_argument(
        "--manifest-dir",
        default=None,
        help=(
            "directory to run `cargo tree` in (default: repo root). Point this at "
            "an EXCLUDED integration workspace such as integrations/vokra-godot to "
            "document its closure; git SHA / SOURCE_DATE_EPOCH still derive from "
            "the repository root."
        ),
    )
    args = ap.parse_args()

    features = [f for f in args.features.split(",") if f]
    root = repo_root()
    sha = git_head_sha(root)
    created = created_timestamp(root)
    tree_cwd = None
    if args.manifest_dir:
        tree_cwd = args.manifest_dir
        if not os.path.isabs(tree_cwd):
            tree_cwd = os.path.join(root, tree_cwd)
        tree_cwd = os.path.abspath(tree_cwd)
        if not os.path.isdir(tree_cwd):
            fail(f"--manifest-dir is not a directory: {tree_cwd}")
        if not os.path.isfile(os.path.join(tree_cwd, "Cargo.toml")):
            fail(f"--manifest-dir has no Cargo.toml: {tree_cwd}")
    graph = resolve_graph(
        root, args.package, features, args.no_default_features, tree_cwd
    )

    feature_comment = (
        f"Dependency graph of {args.package} resolved by `cargo tree -e normal` "
        f"with feature selection: "
        f"{'--no-default-features ' if args.no_default_features else ''}"
        f"--features {','.join(features) if features else '(none)'}"
        f"{f' [manifest-dir {args.manifest_dir}]' if args.manifest_dir else ''}. "
        f"Runtime dependencies only (no dev/build deps). "
        f"Generated at commit {sha}."
    )

    doc = build_document(
        doc_name=args.doc_name or f"{args.package}-sbom",
        package=args.package,
        graph=graph,
        sha=sha,
        created=created,
        feature_comment=feature_comment,
    )

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(
        f"wrote {args.output}: {len(doc['packages'])} packages, "
        f"{len(doc['relationships'])} relationships"
    )


if __name__ == "__main__":
    main()
