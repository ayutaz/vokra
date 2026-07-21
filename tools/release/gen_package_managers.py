#!/usr/bin/env python3
"""gen_package_managers.py — render Homebrew / winget / vcpkg / Debian package
manifests from templates (X-07-T24..T28 common generator).

Vokra's distribution list (NFR-DS-03 / deliverables §3.5) includes Homebrew,
winget, vcpkg, and Debian-family packages. This generator renders the four
manifest sets from `packaging/*/**.tmpl`, filling one shared `@@VERSION@@`,
`@@BASE_URL@@` and `@@SBOM_URL@@`, plus per-asset checksums.

CHECKSUM MODEL (X-07-T28 §(a)/(b)):
  * A `@@SHA256:<asset>@@` / `@@SHA512:<asset>@@` token is filled with the REAL
    hash of `<assets-dir>/<release-filename>` when that file exists.
  * When the file is ABSENT (the T32-gated desktop/AAR binaries do not exist
    until owner enables T29/T30 + tags), the token is filled with an explicit
    PLACEHOLDER `SHA256-PENDING-RELEASE:<asset>` — NEVER a fabricated hash. The
    owner (or the release job) re-renders with the real assets present.
  Source-tarball-based managers (Homebrew, vcpkg, Debian in this scaffold)
  reference the tag-derived source tarball, which is not desktop-binary
  dependent; winget references the Windows CLI binary (T30).

CC does the generation + checksum emission; registry REGISTRATION is owner
(Homebrew tap / microsoft/winget-pkgs PR / microsoft/vcpkg PR / Debian sponsor
upload — X-07-T33).

Zero-dep (NFR-DS-02): python3 stdlib only.

Usage:
    python3 tools/release/gen_package_managers.py --version 1.0.0-rc.1 \
        --out-dir dist/packaging [--assets-dir dist/release-assets]
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
PACKAGING = os.path.join(ROOT, "packaging")
REPO = "https://github.com/ayutaz/vokra"

# Logical asset name -> release filename pattern (@@VERSION@@ substituted).
# Kept in ONE place so every manifest references identical URLs/filenames.
ASSETS = {
    "src": "vokra-@@VERSION@@-src.tar.gz",
    "cli-windows": "vokra-cli-@@VERSION@@-x86_64-windows.zip",
    "lib-linux": "libvokra-@@VERSION@@-x86_64-linux.so",
    "cli-linux": "vokra-cli-@@VERSION@@-x86_64-linux",
}

# Manifest template -> output path (relative to --out-dir). @@VERSION@@ in the
# output path is substituted too (Debian changelog / winget dirs).
TEMPLATES = {
    "homebrew/vokra.rb.tmpl": "homebrew/vokra.rb",
    "winget/Vokra.Vokra.yaml.tmpl": "winget/Vokra.Vokra.yaml",
    "winget/Vokra.Vokra.installer.yaml.tmpl": "winget/Vokra.Vokra.installer.yaml",
    "winget/Vokra.Vokra.locale.en-US.yaml.tmpl": "winget/Vokra.Vokra.locale.en-US.yaml",
    "vcpkg/ports/vokra/portfile.cmake.tmpl": "vcpkg/ports/vokra/portfile.cmake",
    "vcpkg/ports/vokra/vcpkg.json.tmpl": "vcpkg/ports/vokra/vcpkg.json",
    "debian/control.tmpl": "debian/control",
    "debian/rules.tmpl": "debian/rules",
    "debian/changelog.tmpl": "debian/changelog",
    "debian/copyright.tmpl": "debian/copyright",
}

SHA_TOKEN = re.compile(r"@@(SHA256|SHA512):([a-z0-9\-]+)@@")


def fail(msg: str) -> "None":
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(1)


def asset_filename(logical: str, version: str) -> str:
    if logical not in ASSETS:
        fail(f"unknown asset '{logical}' (templates reference an undefined asset)")
    return ASSETS[logical].replace("@@VERSION@@", version)


def hash_asset(logical: str, version: str, assets_dir: str | None, algo: str) -> str:
    """Real hash if the asset file exists, else an explicit PENDING placeholder."""
    fname = asset_filename(logical, version)
    if assets_dir:
        path = os.path.join(assets_dir, fname)
        if os.path.isfile(path):
            h = hashlib.new(algo.lower())
            with open(path, "rb") as fh:
                h.update(fh.read())
            return h.hexdigest()
    # T32-gated / not yet built: never fabricate a hash.
    return f"{algo}-PENDING-RELEASE:{fname}"


def render(text: str, version: str, base_url: str, sbom_url: str,
           assets_dir: str | None) -> str:
    text = text.replace("@@VERSION@@", version)
    text = text.replace("@@BASE_URL@@", base_url)
    text = text.replace("@@SBOM_URL@@", sbom_url)

    def repl(m: "re.Match") -> str:
        algo, logical = m.group(1), m.group(2)
        return hash_asset(logical, version, assets_dir, algo)

    return SHA_TOKEN.sub(repl, text)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--version", required=True, help="release version, e.g. 1.0.0-rc.1")
    ap.add_argument("--out-dir", required=True, help="output directory")
    ap.add_argument("--assets-dir", default=None,
                    help="directory of release asset files to checksum "
                         "(absent files -> explicit PENDING placeholder)")
    ap.add_argument("--base-url", default=None,
                    help="release asset download base URL "
                         "(default: <repo>/releases/download/v<version>)")
    args = ap.parse_args()

    version = args.version
    base_url = args.base_url or f"{REPO}/releases/download/v{version}"
    sbom_url = f"{base_url}/vokra-python-wheel.spdx.json"

    pending = 0
    for tmpl_rel, out_rel in TEMPLATES.items():
        tmpl_path = os.path.join(PACKAGING, tmpl_rel)
        if not os.path.isfile(tmpl_path):
            fail(f"template missing: {tmpl_path}")
        rendered = render(open(tmpl_path, encoding="utf-8").read(),
                          version, base_url, sbom_url, args.assets_dir)
        pending += rendered.count("-PENDING-RELEASE:")
        out_path = os.path.join(args.out_dir, out_rel.replace("@@VERSION@@", version))
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(rendered)
        print(f"rendered {out_rel}")

    if pending:
        print(f"note: {pending} checksum slot(s) are PENDING-RELEASE "
              f"(T32-gated binaries not present in --assets-dir) — the release "
              f"job re-renders with real assets.", file=sys.stderr)
    print(f"gen_package_managers: wrote {len(TEMPLATES)} manifest(s) to {args.out_dir}")


if __name__ == "__main__":
    main()
