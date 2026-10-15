from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import collect_metadata as leaf


TEMP_ROOT = Path(tempfile.gettempdir()).resolve()


def blob(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


WORKSPACE = """[workspace]
members = ["moshi-core", "other"]
resolver = "2"
[workspace.package]
license = "MIT/Apache-2.0"
"""
CORE = """[package]
name = "moshi"
version = "0.6.4"
[package.metadata]
license = "MIT/Apache-2.0"
[features]
default = []
cuda = []
metal = []
flash-attn = ["cuda"]
"""
LICENSE = b"MIT/Apache-2.0 declaration is separate from this receipt.\n"
LOCK = """version = 4
[[package]]
name = "moshi"
version = "0.6.4"
dependencies = ["memmap2"]

[[package]]
name = "memmap2"
version = "{memmap_version}"
source = "registry+https://github.com/rust-lang/crates.io-index"
checksum = "{memmap_checksum}"

[[package]]
name = "candle-core"
version = "0.9.1"
source = "registry+https://github.com/rust-lang/crates.io-index"
checksum = "{candle_checksum}"
"""


def metadata_json(*, gpu: bool = False, gpu_dependency: bool = False, unknown_ref: bool = False, missing_root: bool = False) -> str:
    moshi = "path+file:///tmp/projection/moshi-core#moshi@0.6.4"
    memmap = "registry+https://github.com/rust-lang/crates.io-index#memmap2@0.9.5"
    build = "registry+https://github.com/rust-lang/crates.io-index#build-helper@1.0.0"
    dev = "registry+https://github.com/rust-lang/crates.io-index#dev-helper@1.0.0"
    dep = "registry+https://example.invalid#missing@1.0.0" if unknown_ref else memmap
    features = ["cuda"] if gpu else []
    value = {
        "packages": [
            {"id": moshi, "name": "moshi", "version": "0.6.4", "source": None, "manifest_path": "/tmp/projection/moshi-core/Cargo.toml", "features": {"default": [], "cuda": [], "metal": [], "flash-attn": []}, "dependencies": [
                {"name": "memmap2", "source": "registry+https://github.com/rust-lang/crates.io-index", "req": "^0.9.5", "kind": None, "optional": False, "rename": None, "target": None, "features": [], "uses_default_features": True},
                {"name": "build-helper", "source": "registry+https://github.com/rust-lang/crates.io-index", "req": "^1.0.0", "kind": "build", "optional": False, "rename": None, "target": None, "features": [], "uses_default_features": True},
                {"name": "memmap2", "source": "registry+https://github.com/rust-lang/crates.io-index", "req": "^0.9.5", "kind": "build", "optional": False, "rename": None, "target": None, "features": [], "uses_default_features": True},
                {"name": "dev-helper", "source": "registry+https://github.com/rust-lang/crates.io-index", "req": "^1.0.0", "kind": "dev", "optional": False, "rename": None, "target": None, "features": [], "uses_default_features": True},
            ]},
            {"id": memmap, "name": "memmap2", "version": "0.9.5", "source": "registry+https://github.com/rust-lang/crates.io-index", "manifest_path": "/tmp/memmap/Cargo.toml", "features": {}, "dependencies": []},
            {"id": build, "name": "build-helper", "version": "1.0.0", "source": "registry+https://github.com/rust-lang/crates.io-index", "manifest_path": "/tmp/build/Cargo.toml", "features": {}, "dependencies": []},
            {"id": dev, "name": "dev-helper", "version": "1.0.0", "source": "registry+https://github.com/rust-lang/crates.io-index", "manifest_path": "/tmp/dev/Cargo.toml", "features": {}, "dependencies": []},
        ],
        "workspace_members": [moshi],
        "resolve": {"nodes": [
            {"id": moshi, "features": features, "deps": [
                {"name": "memmap2" if not unknown_ref else "missing", "pkg": dep, "dep_kinds": [{"kind": None, "target": None}]},
                {"name": "build-helper" if gpu_dependency else "memmap2", "pkg": build if gpu_dependency else memmap, "dep_kinds": [{"kind": "build", "target": None}]},
                {"name": "dev-helper", "pkg": dev, "dep_kinds": [{"kind": "dev", "target": None}]},
            ]},
            {"id": memmap, "features": [], "deps": []},
            {"id": build, "features": ["cuda"] if gpu_dependency else [], "deps": []},
            {"id": dev, "features": [], "deps": []},
        ]},
    }
    if not missing_root:
        value["resolve"]["root"] = None
    return json.dumps(value)


def whole_metadata_json(*, gpu: bool = False) -> str:
    value = json.loads(metadata_json(gpu=gpu))
    ids = {package["name"]: package["id"] for package in value["packages"]}
    keep = {ids["moshi"], ids["memmap2"]}
    candle = "registry+https://github.com/rust-lang/crates.io-index#candle-core@0.9.1"
    value["packages"] = [package for package in value["packages"] if package["id"] in keep]
    value["packages"].append({
        "id": candle,
        "name": "candle-core",
        "version": "0.9.1",
        "source": "registry+https://github.com/rust-lang/crates.io-index",
        "manifest_path": "/tmp/candle/Cargo.toml",
        "features": {},
        "dependencies": [],
    })
    value["resolve"]["nodes"] = [node for node in value["resolve"]["nodes"] if node["id"] in keep]
    for node in value["resolve"]["nodes"]:
        if node["id"] == ids["moshi"]:
            node["deps"] = [
                edge for edge in node["deps"]
                if edge["pkg"] == ids["memmap2"]
                and any(dep_kind.get("kind") in (None, "normal") for dep_kind in edge["dep_kinds"])
            ]
    value["resolve"]["nodes"].append({"id": candle, "features": [], "deps": []})
    return json.dumps(value)


def _small_removal_fixture(case: dict[str, object]) -> tuple[str, str, dict[str, object], dict[str, object]]:
    registry = "registry+https://github.com/rust-lang/crates.io-index"
    parent = str(case["parent"])
    parent_version = str(case["parent_version"])
    target = str(case["target"])
    target_version = str(case["target_version"])
    moshi_id = "path+file:///tmp/projection/moshi-core#moshi@0.6.4"
    parent_id = f"{registry}#{parent}@{parent_version}"
    target_id = f"{registry}#{target}@{target_version}"
    target_declaration = json.loads(json.dumps(SOURCE_DERIVED_DECLARATIONS[target]))
    declarations = [target_declaration]
    for extra in case.get("extra_declarations", []):
        if isinstance(extra, dict):
            declarations.append(json.loads(json.dumps(extra)))
        else:
            name, _req = extra
            declarations.append(json.loads(json.dumps(SOURCE_DERIVED_DECLARATIONS[name])))
    parent_package = {
        "id": parent_id, "name": parent, "version": parent_version, "source": registry,
        "manifest_path": f"/tmp/{parent}/Cargo.toml", "features": case["features"], "dependencies": declarations,
    }
    metadata_packages = [
            {"id": moshi_id, "name": "moshi", "version": "0.6.4", "source": None,
             "manifest_path": "/tmp/projection/moshi-core/Cargo.toml", "features": {}, "dependencies": []},
            parent_package,
        ]
    if bool(case["target_present"]):
        metadata_packages.append({"id": target_id, "name": target, "version": target_version, "source": registry,
                                  "manifest_path": f"/tmp/{target}/Cargo.toml", "features": {}, "dependencies": []})
    metadata = {
        "packages": metadata_packages,
        "workspace_members": [moshi_id],
        "resolve": {
            "root": None,
            "nodes": [
                {"id": moshi_id, "features": [], "deps": []},
                {"id": parent_id, "features": list(case["active_features"]), "deps": []},
            ],
        },
    }
    if bool(case["target_present"]):
        metadata["resolve"]["nodes"].append({"id": target_id, "features": [], "deps": []})
    cpu = {
        "packages": [metadata["packages"][0]],
        "workspace_members": [moshi_id],
        "resolve": {"root": None, "nodes": [{"id": moshi_id, "features": [], "deps": []}]},
    }
    lock = (
        "version = 4\n"
        "[[package]]\n"
        "name = \"moshi\"\nversion = \"0.6.4\"\n\n"
        f"[[package]]\nname = \"{parent}\"\nversion = \"{parent_version}\"\nsource = \"{registry}\"\nchecksum = \"{'b' * 64}\"\ndependencies = [\"{target} {target_version}\"]\n\n"
        f"[[package]]\nname = \"{target}\"\nversion = \"{target_version}\"\nsource = \"{registry}\"\nchecksum = \"{'a' * 64}\"\n"
    )
    projected = lock.replace(f'dependencies = ["{target} {target_version}"]\n', "dependencies = []\n")
    if not bool(case["target_present"]):
        projected = projected.replace(
            f'[[package]]\nname = "{target}"\nversion = "{target_version}"\nsource = "{registry}"\nchecksum = "{"a" * 64}"\n',
            "",
        )
    return lock, projected, cpu, metadata


REMOVAL_CASES = (
    {"parent": "cc", "parent_version": "1.2.19", "target": "jobserver", "target_version": "0.1.33", "req": "^0.1.30", "target_cfg": None, "features": {"jobserver": [], "parallel": ["dep:libc", "dep:jobserver"]}, "active_features": [], "extra_declarations": [("libc", "^0.2.62")]},
    {"parent": "cc", "parent_version": "1.2.19", "target": "libc", "target_version": "0.2.171", "req": "^0.2.62", "target_cfg": "cfg(unix)", "features": {"jobserver": [], "parallel": ["dep:libc", "dep:jobserver"]}, "active_features": [], "extra_declarations": [("jobserver", "^0.1.30")]},
    {"parent": "hashbrown", "parent_version": "0.15.2", "target": "allocator-api2", "target_version": "0.2.21", "req": "^0.2.9", "target_cfg": None, "features": {"alloc": ["dep:alloc"], "allocator-api2": ["dep:allocator-api2"], "compiler_builtins": ["dep:compiler_builtins"], "core": ["dep:core"], "default": ["default-hasher", "inline-more", "allocator-api2", "equivalent", "raw-entry"], "default-hasher": ["dep:foldhash"], "equivalent": ["dep:equivalent"], "inline-more": [], "nightly": ["allocator-api2?/nightly", "bumpalo/allocator_api"], "raw-entry": [], "rayon": ["dep:rayon"], "rustc-dep-of-std": ["nightly", "core", "compiler_builtins", "alloc", "rustc-internal-api", "raw-entry"], "rustc-internal-api": [], "serde": ["dep:serde"]}, "active_features": [], "extra_declarations": [("equivalent", "^1.0"), ("foldhash", "^0.1.2")]},
    {"parent": "hashbrown", "parent_version": "0.15.2", "target": "equivalent", "target_version": "1.0.2", "req": "^1.0", "target_cfg": None, "features": {"alloc": ["dep:alloc"], "allocator-api2": ["dep:allocator-api2"], "compiler_builtins": ["dep:compiler_builtins"], "core": ["dep:core"], "default": ["default-hasher", "inline-more", "allocator-api2", "equivalent", "raw-entry"], "default-hasher": ["dep:foldhash"], "equivalent": ["dep:equivalent"], "inline-more": [], "nightly": ["allocator-api2?/nightly", "bumpalo/allocator_api"], "raw-entry": [], "rayon": ["dep:rayon"], "rustc-dep-of-std": ["nightly", "core", "compiler_builtins", "alloc", "rustc-internal-api", "raw-entry"], "rustc-internal-api": [], "serde": ["dep:serde"]}, "active_features": [], "extra_declarations": [("allocator-api2", "^0.2.9"), ("foldhash", "^0.1.2")]},
    {"parent": "hashbrown", "parent_version": "0.15.2", "target": "foldhash", "target_version": "0.1.5", "req": "^0.1.2", "target_cfg": None, "features": {"alloc": ["dep:alloc"], "allocator-api2": ["dep:allocator-api2"], "compiler_builtins": ["dep:compiler_builtins"], "core": ["dep:core"], "default": ["default-hasher", "inline-more", "allocator-api2", "equivalent", "raw-entry"], "default-hasher": ["dep:foldhash"], "equivalent": ["dep:equivalent"], "inline-more": [], "nightly": ["allocator-api2?/nightly", "bumpalo/allocator_api"], "raw-entry": [], "rayon": ["dep:rayon"], "rustc-dep-of-std": ["nightly", "core", "compiler_builtins", "alloc", "rustc-internal-api", "raw-entry"], "rustc-internal-api": [], "serde": ["dep:serde"]}, "active_features": [], "extra_declarations": [("allocator-api2", "^0.2.9"), ("equivalent", "^1.0")]},
    {"parent": "toml_datetime", "parent_version": "0.6.8", "target": "serde", "target_version": "1.0.219", "req": "^1.0.145", "target_cfg": None, "features": {"serde": ["dep:serde"]}, "active_features": [], "extra_declarations": []},
    {"parent": "toml_edit", "parent_version": "0.22.24", "target": "serde", "target_version": "1.0.219", "req": "^1.0.145", "target_cfg": None, "features": {"default": ["parse", "display"], "display": [], "parse": ["dep:winnow"], "perf": ["dep:kstring"], "serde": ["dep:serde", "toml_datetime/serde", "dep:serde_spanned"], "unbounded": []}, "active_features": ["parse"], "extra_declarations": [("serde_spanned", "^0.6.8"), ("winnow", "^0.7.0")]},
    {"parent": "toml_edit", "parent_version": "0.22.24", "target": "serde_spanned", "target_version": "0.6.8", "req": "^0.6.8", "target_cfg": None, "features": {"default": ["parse", "display"], "display": [], "parse": ["dep:winnow"], "perf": ["dep:kstring"], "serde": ["dep:serde", "toml_datetime/serde", "dep:serde_spanned"], "unbounded": []}, "active_features": ["parse"], "extra_declarations": [("serde", "^1.0.145"), ("winnow", "^0.7.0")]},
    {"parent": "tracing", "parent_version": "0.1.41", "target": "log", "target_version": "0.4.27", "req": "^0.4.17", "target_cfg": None, "features": {"async-await": [], "attributes": ["tracing-attributes"], "default": ["std", "attributes"], "log": ["dep:log"], "log-always": ["log"], "max_level_debug": [], "max_level_error": [], "max_level_info": [], "max_level_off": [], "max_level_trace": [], "max_level_warn": [], "release_max_level_debug": [], "release_max_level_error": [], "release_max_level_info": [], "release_max_level_off": [], "release_max_level_trace": [], "release_max_level_warn": [], "std": ["tracing-core/std"], "tracing-attributes": ["dep:tracing-attributes"], "valuable": ["tracing-core/valuable"]}, "active_features": ["attributes", "default", "std", "tracing-attributes"], "extra_declarations": [("tracing-attributes", "^0.1.28"), ("tracing-core", "^0.1.33")]},
    {"parent": "tracing-core", "parent_version": "0.1.33", "target": "valuable", "target_version": "0.1.1", "req": "^0.1.0", "target_cfg": "cfg(tracing_unstable)", "features": {"default": ["std", "valuable?/std"], "once_cell": ["dep:once_cell"], "std": ["once_cell"], "valuable": ["dep:valuable"]}, "active_features": ["once_cell", "std"], "extra_declarations": [("once_cell", "^1.13.0")]},
)

SOURCE_DERIVED_DECLARATIONS = {
    "jobserver": {"features": [], "kind": None, "name": "jobserver", "optional": True, "registry": None, "rename": None, "req": "^0.1.30", "source": "registry+https://github.com/rust-lang/crates.io-index", "target": None, "uses_default_features": False},
    "libc": {"features": [], "kind": None, "name": "libc", "optional": True, "registry": None, "rename": None, "req": "^0.2.62", "source": "registry+https://github.com/rust-lang/crates.io-index", "target": "cfg(unix)", "uses_default_features": False},
    "allocator-api2": {"features": ["alloc"], "kind": None, "name": "allocator-api2", "optional": True, "registry": None, "rename": None, "req": "^0.2.9", "source": "registry+https://github.com/rust-lang/crates.io-index", "target": None, "uses_default_features": False},
    "equivalent": {"features": [], "kind": None, "name": "equivalent", "optional": True, "registry": None, "rename": None, "req": "^1.0", "source": "registry+https://github.com/rust-lang/crates.io-index", "target": None, "uses_default_features": False},
    "foldhash": {"features": [], "kind": None, "name": "foldhash", "optional": True, "registry": None, "rename": None, "req": "^0.1.2", "source": "registry+https://github.com/rust-lang/crates.io-index", "target": None, "uses_default_features": False},
    "serde": {"features": [], "kind": None, "name": "serde", "optional": True, "registry": None, "rename": None, "req": "^1.0.145", "source": "registry+https://github.com/rust-lang/crates.io-index", "target": None, "uses_default_features": True},
    "serde_spanned": {"features": ["serde"], "kind": None, "name": "serde_spanned", "optional": True, "registry": None, "rename": None, "req": "^0.6.8", "source": "registry+https://github.com/rust-lang/crates.io-index", "target": None, "uses_default_features": True},
    "winnow": {"features": [], "kind": None, "name": "winnow", "optional": True, "registry": None, "rename": None, "req": "^0.7.0", "source": "registry+https://github.com/rust-lang/crates.io-index", "target": None, "uses_default_features": True},
    "tracing-attributes": {"features": [], "kind": None, "name": "tracing-attributes", "optional": True, "registry": None, "rename": None, "req": "^0.1.28", "source": "registry+https://github.com/rust-lang/crates.io-index", "target": None, "uses_default_features": True},
    "tracing-core": {"features": [], "kind": None, "name": "tracing-core", "optional": False, "registry": None, "rename": None, "req": "^0.1.33", "source": "registry+https://github.com/rust-lang/crates.io-index", "target": None, "uses_default_features": False},
    "once_cell": {"features": [], "kind": None, "name": "once_cell", "optional": True, "registry": None, "rename": None, "req": "^1.13.0", "source": "registry+https://github.com/rust-lang/crates.io-index", "target": None, "uses_default_features": True},
    "log": {"features": [], "kind": None, "name": "log", "optional": True, "registry": None, "rename": None, "req": "^0.4.17", "source": "registry+https://github.com/rust-lang/crates.io-index", "target": None, "uses_default_features": True},
    "valuable": {"features": [], "kind": None, "name": "valuable", "optional": True, "registry": None, "rename": None, "req": "^0.1.0", "source": "registry+https://github.com/rust-lang/crates.io-index", "target": "cfg(tracing_unstable)", "uses_default_features": False},
}

RETAINED_REMOVAL_TARGETS = {("cc", "libc"), ("hashbrown", "equivalent"), ("toml_datetime", "serde"), ("toml_edit", "serde"), ("tracing", "log")}
for _case in REMOVAL_CASES:
    _case["target_present"] = (_case["parent"], _case["target"]) in RETAINED_REMOVAL_TARGETS


class CollectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="vokra-candle-test-", dir=TEMP_ROOT)
        self.root = Path(self.tmp.name)
        self.old_files = leaf.SOURCE_FILES
        self.old_urls = leaf.SOURCE_URLS
        leaf.SOURCE_FILES = {
            "Cargo.toml": (blob(WORKSPACE.encode()), len(WORKSPACE.encode())),
            "Cargo.lock": (blob(LOCK.format(memmap_version="0.9.5", memmap_checksum="a" * 64, candle_checksum="b" * 64).encode()), len(LOCK.format(memmap_version="0.9.5", memmap_checksum="a" * 64, candle_checksum="b" * 64).encode())),
            "LICENSE": (blob(LICENSE), len(LICENSE)),
            "README.md": (blob(b"readme\n"), 7),
            "moshi-core/Cargo.toml": (blob(CORE.encode()), len(CORE.encode())),
            "moshi-core/src/lib.rs": (blob(b"pub fn synthetic() {}\n"), len(b"pub fn synthetic() {}\n")),
        }
        leaf.SOURCE_URLS = {path: "https://raw.githubusercontent.com/kyutai-labs/moshi/test/" + path for path in leaf.SOURCE_FILES}

    def tearDown(self) -> None:
        leaf.SOURCE_FILES = self.old_files
        leaf.SOURCE_URLS = self.old_urls
        self.tmp.cleanup()

    def fixture_fetcher(self, bad: str | None = None, missing: str | None = None, oversized: bool = False):
        data = {
            "Cargo.toml": WORKSPACE.encode(),
            "Cargo.lock": LOCK.format(memmap_version="0.9.5", memmap_checksum="a" * 64, candle_checksum="b" * 64).encode(),
            "LICENSE": LICENSE,
            "README.md": b"readme\n",
            "moshi-core/Cargo.toml": CORE.encode(),
            "moshi-core/src/lib.rs": b"pub fn synthetic() {}\n",
        }
        def fetch(url: str, deadline: float) -> bytes:
            path = url.split("/test/", 1)[1]
            if path == missing:
                raise leaf.CaptureError("synthetic missing source")
            value = data[path]
            if path == bad:
                value = bytes([value[0] ^ 1]) + value[1:]
            if oversized:
                value = b"x" * (leaf.MAX_FILE_BYTES + 1)
            return value
        return fetch

    def test_authenticated_source_good_bad_missing_and_size(self) -> None:
        source = self.root / "source"
        receipt = leaf.fetch_and_verify_sources(source, deadline=time.monotonic() + 5, fetcher=self.fixture_fetcher())
        self.assertEqual(receipt["provenance"], "official-raw-fixed-git-blob-authenticated")
        self.assertEqual(receipt["files"]["LICENSE"]["bytes"], len(LICENSE))
        for bad, missing, oversized in (("README.md", None, False), (None, "README.md", False), (None, None, True)):
            with self.subTest(bad=bad, missing=missing, oversized=oversized):
                with self.assertRaises(leaf.CaptureError):
                    leaf.fetch_and_verify_sources(self.root / ("bad-" + str(bad or missing or oversized)), deadline=time.monotonic() + 5, fetcher=self.fixture_fetcher(bad, missing, oversized))

    def test_fully_usable_projection_and_lock_separation(self) -> None:
        source = self.root / "source"
        leaf.fetch_and_verify_sources(source, deadline=time.monotonic() + 5, fetcher=self.fixture_fetcher())
        projection = self.root / "projection"
        leaf._copy_source_to_projection(source, projection)
        self.assertEqual(leaf.read_toml(projection / "Cargo.toml")["workspace"]["members"], ["moshi-core"])
        self.assertTrue((projection / "moshi-core/src/lib.rs").is_file())
        base = self.root / "base.lock"
        base.write_bytes((projection / "Cargo.lock").read_bytes())
        self.assertEqual(leaf.validate_base_lock(source / "Cargo.lock", base)["status"], "UNCHANGED")
        patched = self.root / "patched.lock"
        patched.write_text(LOCK.format(memmap_version="0.9.11", memmap_checksum="c" * 64, candle_checksum="b" * 64))
        self.assertEqual(leaf.validate_patch_lock(base, patched)["status"], "PATCH_DELTA_ONLY")

    def test_base_pruning_and_unexpected_upgrade(self) -> None:
        original = self.root / "original.lock"
        original.write_text(LOCK.format(memmap_version="0.9.5", memmap_checksum="a" * 64, candle_checksum="b" * 64) + '\n[[package]]\nname = "unused"\nversion = "1"\n')
        projected = self.root / "projected.lock"
        projected.write_text(LOCK.format(memmap_version="0.9.5", memmap_checksum="a" * 64, candle_checksum="b" * 64))
        self.assertEqual(leaf.validate_base_lock(original, projected)["status"], "SUBSET_UNCHANGED")
        projected.write_text(LOCK.format(memmap_version="0.9.6", memmap_checksum="a" * 64, candle_checksum="b" * 64))
        with self.assertRaises(leaf.CaptureError):
            leaf.validate_base_lock(original, projected)

    def _semantic_validate(
        self,
        *,
        original_text: str | None = None,
        projected_text: str | None = None,
        cpu_value: dict[str, object] | None = None,
        whole_value: dict[str, object] | None = None,
    ) -> dict[str, object]:
        original = self.root / "semantic-original.lock"
        projected = self.root / "semantic-projected.lock"
        original.write_text(original_text or LOCK.format(memmap_version="0.9.5", memmap_checksum="a" * 64, candle_checksum="b" * 64))
        projected.write_text(projected_text or LOCK.format(memmap_version="0.9.5", memmap_checksum="a" * 64, candle_checksum="b" * 64))
        cpu = cpu_value or json.loads(metadata_json())
        whole = whole_value or json.loads(whole_metadata_json())
        return leaf.validate_base_lock(original, projected, cpu_metadata=cpu, whole_metadata=whole)

    def test_semantic_projection_accepts_canonical_alias_and_rejects_record_mutations(self) -> None:
        aliased = LOCK.format(memmap_version="0.9.5", memmap_checksum="a" * 64, candle_checksum="b" * 64).replace(
            'dependencies = ["memmap2"]', 'dependencies = ["memmap2 0.9.5"]'
        )
        result = self._semantic_validate(projected_text=aliased)
        self.assertEqual(result["status"], "SOURCE_ONLY_METADATA_REVIEW_PENDING")
        self.assertEqual(result["projection_status"], "SUBSET_UNCHANGED")
        for mutation in (
            ('version = "0.6.4"', 'version = "0.6.5"'),
            ('source = "registry+https://github.com/rust-lang/crates.io-index"', 'source = "git+https://example.invalid"'),
            ('checksum = "' + "b" * 64 + '"', 'checksum = "' + "c" * 64 + '"'),
        ):
            with self.subTest(mutation=mutation):
                mutated = LOCK.format(memmap_version="0.9.5", memmap_checksum="a" * 64, candle_checksum="b" * 64).replace(*mutation)
                with self.assertRaises(leaf.CaptureError):
                    self._semantic_validate(projected_text=mutated)
        unknown = aliased.replace('name = "moshi"', 'name = "moshi"\nunknown_scalar = true')
        with self.assertRaises(leaf.CaptureError):
            self._semantic_validate(projected_text=unknown)

    def test_semantic_projection_rejects_added_reordered_and_mandatory_edges(self) -> None:
        added = LOCK.format(memmap_version="0.9.5", memmap_checksum="a" * 64, candle_checksum="b" * 64).replace(
            'dependencies = ["memmap2"]', 'dependencies = ["memmap2", "candle-core"]'
        )
        with self.assertRaisesRegex(leaf.CaptureError, "canonical dependency"):
            self._semantic_validate(projected_text=added)
        ordered = LOCK.format(memmap_version="0.9.5", memmap_checksum="a" * 64, candle_checksum="b" * 64).replace(
            'dependencies = ["memmap2"]', 'dependencies = ["memmap2", "candle-core"]'
        )
        reversed_order = ordered.replace(
            'dependencies = ["memmap2", "candle-core"]', 'dependencies = ["candle-core", "memmap2"]'
        )
        whole = json.loads(whole_metadata_json())
        moshi = next(package for package in whole["packages"] if package["name"] == "moshi")
        candle = next(package for package in whole["packages"] if package["name"] == "candle-core")
        moshi["dependencies"].append({
            "name": "candle-core", "source": candle["source"], "req": "^0.9.1", "kind": None,
            "optional": False, "rename": None, "target": None, "features": [], "uses_default_features": True,
        })
        whole["resolve"]["nodes"][0]["deps"].append({
            "name": "candle-core", "pkg": candle["id"], "dep_kinds": [{"kind": None, "target": None}]
        })
        cpu = json.loads(metadata_json())
        cpu_moshi = next(package for package in cpu["packages"] if package["name"] == "moshi")
        cpu_moshi["dependencies"] = json.loads(json.dumps(moshi["dependencies"]))
        cpu["resolve"]["nodes"][0]["deps"] = []
        with self.assertRaisesRegex(leaf.CaptureError, "canonical dependency"):
            self._semantic_validate(original_text=ordered, projected_text=reversed_order, cpu_value=cpu, whole_value=whole)
        mandatory_removed = LOCK.format(memmap_version="0.9.5", memmap_checksum="a" * 64, candle_checksum="b" * 64).replace(
            'dependencies = ["memmap2"]', 'dependencies = []'
        )
        whole = json.loads(whole_metadata_json())
        whole["packages"][0]["dependencies"] = [whole["packages"][0]["dependencies"][0]]
        whole["resolve"]["nodes"][0]["deps"] = []
        cpu = json.loads(metadata_json())
        cpu["packages"][0]["dependencies"] = json.loads(json.dumps(whole["packages"][0]["dependencies"]))
        cpu["resolve"]["nodes"][0]["deps"] = []
        with self.assertRaisesRegex(leaf.CaptureError, "mandatory non-dev"):
            self._semantic_validate(projected_text=mandatory_removed, cpu_value=cpu, whole_value=whole)

    def test_semantic_projection_rejects_inconsistent_whole_metadata(self) -> None:
        whole = json.loads(whole_metadata_json())
        whole["packages"] = [package for package in whole["packages"] if package["name"] != "candle-core"]
        with self.assertRaisesRegex(leaf.CaptureError, "package identities|unknown"):
            self._semantic_validate(whole_value=whole)
        whole = json.loads(whole_metadata_json())
        whole["resolve"]["nodes"][0]["deps"][0]["pkg"] = "unknown-package-id"
        with self.assertRaisesRegex(leaf.CaptureError, "unknown package"):
            self._semantic_validate(whole_value=whole)
        whole = json.loads(whole_metadata_json())
        whole["resolve"]["nodes"][0]["deps"][0]["name"] = "wrong-alias"
        with self.assertRaisesRegex(leaf.CaptureError, "manifest declaration"):
            self._semantic_validate(whole_value=whole)
        whole = json.loads(whole_metadata_json())
        next(package for package in whole["packages"] if package["name"] == "memmap2")["name"] = "wrong-target"
        with self.assertRaisesRegex(leaf.CaptureError, "manifest declaration"):
            self._semantic_validate(whole_value=whole)
        whole = json.loads(whole_metadata_json())
        whole["resolve"]["nodes"][0]["deps"][0]["dep_kinds"][0]["target"] = "cfg(windows)"
        with self.assertRaisesRegex(leaf.CaptureError, "manifest declaration"):
            self._semantic_validate(whole_value=whole)
        whole = json.loads(whole_metadata_json())
        whole["resolve"]["nodes"].append(dict(whole["resolve"]["nodes"][1]))
        with self.assertRaisesRegex(leaf.CaptureError, "duplicated"):
            self._semantic_validate(whole_value=whole)
        whole = json.loads(whole_metadata_json())
        whole["resolve"]["nodes"] = [node for node in whole["resolve"]["nodes"] if node["id"] != "registry+https://github.com/rust-lang/crates.io-index#memmap2@0.9.5"]
        with self.assertRaisesRegex(leaf.CaptureError, "identities differ"):
            self._semantic_validate(whole_value=whole)
        whole = json.loads(whole_metadata_json())
        whole["resolve"]["nodes"][0]["dependencies"] = []
        with self.assertRaisesRegex(leaf.CaptureError, "disagree"):
            self._semantic_validate(whole_value=whole)
        whole = json.loads(whole_metadata_json())
        whole["resolve"]["nodes"][0]["dependencies"] = ["memmap-id"]
        with self.assertRaisesRegex(leaf.CaptureError, "disagree"):
            self._semantic_validate(whole_value=whole)

    def test_semantic_projection_rejects_added_projected_package(self) -> None:
        projected = LOCK.format(memmap_version="0.9.5", memmap_checksum="a" * 64, candle_checksum="b" * 64)
        projected += '\n[[package]]\nname = "extra"\nversion = "1.0.0"\n'
        with self.assertRaisesRegex(leaf.CaptureError, "unknown package"):
            self._semantic_validate(projected_text=projected)

    def test_semantic_projection_rejects_active_cpu_declaration_mutations(self) -> None:
        for field, value in (("req", "=0.1.0"), ("kind", "build"), ("target", "cfg(windows)")):
            with self.subTest(field=field):
                cpu = json.loads(metadata_json())
                cpu_package = next(package for package in cpu["packages"] if package["name"] == "moshi")
                cpu_package["dependencies"][0][field] = value
                with self.assertRaises(leaf.CaptureError):
                    self._semantic_validate(cpu_value=cpu)

    def test_semantic_projection_requires_whole_proof_for_retained_target_and_optional_activation(self) -> None:
        projected = LOCK.format(memmap_version="0.9.5", memmap_checksum="a" * 64, candle_checksum="b" * 64).replace(
            'dependencies = ["memmap2"]', 'dependencies = []'
        )
        whole = json.loads(whole_metadata_json())
        whole["packages"][0]["dependencies"] = [whole["packages"][0]["dependencies"][0]]
        whole["packages"][0]["dependencies"][0]["optional"] = True
        whole["resolve"]["nodes"][0]["deps"] = []
        cpu = json.loads(metadata_json())
        cpu["packages"][0]["dependencies"] = json.loads(json.dumps(whole["packages"][0]["dependencies"]))
        cpu["resolve"]["nodes"][0]["deps"] = []
        self.assertEqual(self._semantic_validate(projected_text=projected, cpu_value=cpu, whole_value=whole)["status"], "SOURCE_ONLY_METADATA_REVIEW_PENDING")
        whole["packages"][0]["dependencies"] = [{
            "name": "memmap2", "source": "registry+https://github.com/rust-lang/crates.io-index", "req": "^0.9.5",
            "kind": None, "optional": True, "rename": None, "target": None, "features": [], "uses_default_features": True,
        }]
        whole["packages"][0]["features"] = {"default": ["dep:memmap2"]}
        whole["resolve"]["nodes"][0]["features"] = ["default"]
        cpu["packages"][0]["dependencies"] = json.loads(json.dumps(whole["packages"][0]["dependencies"]))
        with self.assertRaisesRegex(leaf.CaptureError, "activated"):
            self._semantic_validate(projected_text=projected, cpu_value=cpu, whole_value=whole)
        whole["packages"][0]["features"] = {"default": ["memmap2?/serde"]}
        self.assertEqual(
            self._semantic_validate(projected_text=projected, cpu_value=cpu, whole_value=whole)["status"],
            "SOURCE_ONLY_METADATA_REVIEW_PENDING",
        )
        for expression in ("memmap2", "memmap2/serde"):
            whole["packages"][0]["features"] = {"default": [expression]}
            with self.subTest(expression=expression):
                with self.assertRaisesRegex(leaf.CaptureError, "activated"):
                    self._semantic_validate(projected_text=projected, cpu_value=cpu, whole_value=whole)
        mixed = json.loads(whole_metadata_json())
        mixed["packages"][0]["dependencies"] = [mixed["packages"][0]["dependencies"][0]]
        mixed["packages"][0]["dependencies"][0]["optional"] = True
        mixed["packages"][0]["dependencies"].append(dict(mixed["packages"][0]["dependencies"][0], optional=False))
        mixed["resolve"]["nodes"][0]["deps"] = []
        mixed_cpu = json.loads(metadata_json())
        mixed_cpu["packages"][0]["dependencies"] = json.loads(json.dumps(mixed["packages"][0]["dependencies"]))
        mixed_cpu["resolve"]["nodes"][0]["deps"] = []
        with self.assertRaisesRegex(leaf.CaptureError, "mandatory non-dev"):
            self._semantic_validate(projected_text=projected, cpu_value=mixed_cpu, whole_value=mixed)

    def test_source_derived_six_parent_removals_are_positive_with_small_shapes(self) -> None:
        self.assertEqual(len(REMOVAL_CASES), 10)
        self.assertEqual(len({case["parent"] for case in REMOVAL_CASES}), 6)
        self.assertEqual(sum(bool(case["target_present"]) for case in REMOVAL_CASES), 5)
        for case in REMOVAL_CASES:
            with self.subTest(parent=case["parent"], target=case["target"]):
                original, projected, cpu, whole = _small_removal_fixture(case)
                self.assertEqual(
                    self._semantic_validate(original_text=original, projected_text=projected, cpu_value=cpu, whole_value=whole)["status"],
                    "SOURCE_ONLY_METADATA_REVIEW_PENDING",
                )

    def test_strict_default_rejects_retained_target_edge_deletion_without_whole_proof(self) -> None:
        original = self.root / "strict-original.lock"
        projected = self.root / "strict-projected.lock"
        original.write_text(LOCK.format(memmap_version="0.9.5", memmap_checksum="a" * 64, candle_checksum="b" * 64))
        projected.write_text(LOCK.format(memmap_version="0.9.5", memmap_checksum="a" * 64, candle_checksum="b" * 64).replace(
            'dependencies = ["memmap2"]', 'dependencies = []'
        ))
        with self.assertRaises(leaf.CaptureError):
            leaf.validate_base_lock(original, projected)

    def test_semantic_projection_rejects_ambiguous_labels(self) -> None:
        complete = {
            ("dup", "1.0.0", None): {},
            ("dup", "1.1.0", None): {},
        }
        with self.assertRaisesRegex(leaf.CaptureError, "ambiguous"):
            leaf._lock_dependency_key("dup", complete)

    def test_patch_wrong_checksum_and_source_fail(self) -> None:
        base = self.root / "base.lock"
        base.write_text(LOCK.format(memmap_version="0.9.5", memmap_checksum="a" * 64, candle_checksum="b" * 64))
        patched = self.root / "patched.lock"
        patched.write_text(LOCK.format(memmap_version="0.9.11", memmap_checksum="a" * 64, candle_checksum="b" * 64))
        with self.assertRaises(leaf.CaptureError):
            leaf.validate_patch_lock(base, patched)

        base.write_text(LOCK.format(memmap_version="0.9.5", memmap_checksum="a" * 64, candle_checksum="b" * 64).replace('dependencies = ["memmap2"]', 'dependencies = ["memmap2 0.9.5"]'))
        patched.write_text(LOCK.format(memmap_version="0.9.11", memmap_checksum="c" * 64, candle_checksum="b" * 64).replace('dependencies = ["memmap2"]', 'dependencies = ["memmap2 9.9.9"]'))
        with self.assertRaises(leaf.CaptureError):
            leaf.validate_patch_lock(base, patched)
        patched.write_text(LOCK.format(memmap_version="0.9.11", memmap_checksum="c" * 64, candle_checksum="b" * 64).replace("registry+https://github.com/rust-lang/crates.io-index", "git+https://example.invalid"))
        with self.assertRaises(leaf.CaptureError):
            leaf.validate_patch_lock(base, patched)

    def test_normal_build_active_graph_and_gpu_injection(self) -> None:
        graph = leaf.inspect_metadata(metadata_json())
        self.assertEqual(graph["target"], leaf.TARGET)
        self.assertIn("normal_package_ids", graph)
        self.assertIn("build_package_ids", graph)
        self.assertIn("actual_activated_features", graph)
        self.assertNotIn("dev-helper", " ".join(graph["active_root_closure"]))
        with self.assertRaises(leaf.CaptureError):
            leaf.inspect_metadata(metadata_json(gpu=True))
        with self.assertRaises(leaf.CaptureError):
            leaf.inspect_metadata(metadata_json(gpu_dependency=True))
        with self.assertRaises(leaf.CaptureError):
            leaf.inspect_metadata(metadata_json(unknown_ref=True))
        with self.assertRaises(leaf.CaptureError):
            leaf.inspect_metadata(metadata_json(missing_root=True))
        missing_resolve = json.loads(metadata_json())
        missing_resolve.pop("resolve")
        with self.assertRaises(leaf.CaptureError):
            leaf.inspect_metadata(json.dumps(missing_resolve))
        empty_resolve = json.loads(metadata_json())
        empty_resolve["resolve"]["nodes"] = []
        with self.assertRaises(leaf.CaptureError):
            leaf.inspect_metadata(json.dumps(empty_resolve))
        untyped = json.loads(metadata_json())
        untyped["resolve"]["nodes"][0]["deps"][0].pop("dep_kinds")
        with self.assertRaises(leaf.CaptureError):
            leaf.inspect_metadata(json.dumps(untyped))
        absent_node = json.loads(metadata_json())
        orphan = "registry+https://github.com/rust-lang/crates.io-index#orphan@1.0.0"
        absent_node["packages"].append({"id": orphan, "name": "orphan", "manifest_path": "/tmp/orphan/Cargo.toml"})
        absent_node["resolve"]["nodes"][0]["deps"][0]["pkg"] = orphan
        with self.assertRaises(leaf.CaptureError):
            leaf.inspect_metadata(json.dumps(absent_node))
        bad_member = json.loads(metadata_json())
        bad_member["workspace_members"] = ["unknown-member"]
        with self.assertRaises(leaf.CaptureError):
            leaf.inspect_metadata(json.dumps(bad_member))

    def test_target_and_cargo_action_allowlist(self) -> None:
        cargo = ["cargo", "metadata", "--format-version", "1", "--filter-platform", leaf.TARGET, "--no-default-features"]
        leaf.validate_cargo_argv(cargo)
        whole = ["cargo", "metadata", "--format-version", "1", "--all-features"]
        leaf.validate_cargo_argv(whole)
        with self.assertRaises(leaf.CaptureError):
            leaf.validate_cargo_argv(["cargo", "metadata", "--filter-platform", "aarch64-unknown-linux-gnu"])
        for invalid in (
            whole + ["--filter-platform", leaf.TARGET],
            whole + ["--no-default-features"],
            ["cargo", "metadata", "--format-version", "1", "--all-features", "--locked"],
        ):
            with self.subTest(invalid=invalid):
                with self.assertRaises(leaf.CaptureError):
                    leaf.validate_cargo_argv(invalid)
        for action in ("build", "check", "test", "run"):
            with self.subTest(action=action):
                with self.assertRaises(leaf.CaptureError):
                    leaf.validate_cargo_argv(["cargo", action])

    def _bounded_output_cargo(self, size: int) -> Path:
        return self._bounded_output_executable("cargo", size)

    def _bounded_output_executable(self, name: str, size: int) -> Path:
        payload = self.root / "bounded-output.bin"
        payload.write_bytes(b"x" * size)
        executable = self.root / name
        executable.write_text(f"#!/bin/sh\nset -eu\ncat {payload}\n")
        executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
        return executable

    def test_cpu_metadata_keeps_512kib_capture_bound(self) -> None:
        cargo = self._bounded_output_cargo(leaf.MAX_CAPTURE_BYTES + 1)
        command = [str(cargo), "metadata", "--format-version", "1", "--filter-platform", leaf.TARGET, "--no-default-features"]
        with mock.patch.object(leaf, "_kill_process_group", side_effect=lambda process: process.terminate() or process.wait(timeout=1)):
            with self.assertRaisesRegex(leaf.CaptureError, "exceeded capture bound"):
                leaf.run_bounded_subprocess(command, cwd=self.root, env={"PATH": "/usr/bin:/bin"}, deadline=time.monotonic() + 5)

        rustc = self._bounded_output_executable("rustc", leaf.MAX_CAPTURE_BYTES + 1)
        command = [str(rustc), *leaf.WHOLE_LOCK_METADATA_ARGS]
        with mock.patch.object(leaf, "_kill_process_group", side_effect=lambda process: process.terminate() or process.wait(timeout=1)):
            with self.assertRaisesRegex(leaf.CaptureError, "exceeded capture bound"):
                leaf.run_bounded_subprocess(command, cwd=self.root, env={"PATH": "/usr/bin:/bin"}, deadline=time.monotonic() + 5, validate=False)

    def test_only_exact_whole_metadata_gets_2mib_bound(self) -> None:
        cargo = self._bounded_output_cargo(leaf.MAX_CAPTURE_BYTES + 1)
        command = [str(cargo), "metadata", "--format-version", "1", "--all-features"]
        result = leaf.run_bounded_subprocess(command, cwd=self.root, env={"PATH": "/usr/bin:/bin"}, deadline=time.monotonic() + 5)
        self.assertEqual(result["returncode"], 0)
        self.assertEqual(len(result["stdout"].encode()), leaf.MAX_CAPTURE_BYTES + 1)

        cargo = self._bounded_output_cargo(leaf.MAX_WHOLE_LOCK_CAPTURE_BYTES + 1)
        command[0] = str(cargo)
        with mock.patch.object(leaf, "_kill_process_group", side_effect=lambda process: process.terminate() or process.wait(timeout=1)):
            with self.assertRaisesRegex(leaf.CaptureError, "exceeded capture bound"):
                leaf.run_bounded_subprocess(command, cwd=self.root, env={"PATH": "/usr/bin:/bin"}, deadline=time.monotonic() + 5)

    def _fake_cargo(self, body: str) -> Path:
        path = self.root / "cargo"
        path.write_text("#!/bin/sh\nset -eu\n" + body + "\n")
        path.chmod(path.stat().st_mode | stat.S_IXUSR)
        return path

    def _full_capture_environment(self, *, rust_version: str = "1.99.0", rust_host: str = leaf.TARGET, metadata: str | None = None, whole_metadata: str | None = None, mutate_base: bool = False, mutate_final: bool = False, mutate_whole: bool = False) -> tuple[Path, Path]:
        fake_bin = self.root / "fake-bin"
        fake_bin.mkdir(exist_ok=True)
        metadata_path = self.root / "metadata-fixture.json"
        metadata_path.write_text(metadata if metadata is not None else metadata_json())
        whole_metadata_path = self.root / "whole-metadata-fixture.json"
        whole_metadata_path.write_text(whole_metadata if whole_metadata is not None else whole_metadata_json())
        patched_lock = self.root / "patched-fixture.lock"
        patched_lock.write_text(LOCK.format(memmap_version="0.9.11", memmap_checksum="c" * 64, candle_checksum="b" * 64))
        post_update_marker = self.root / "post-update.marker"
        mutated_base_lock = self.root / "mutated-base-fixture.lock"
        mutated_base_lock.write_text(
            LOCK.format(memmap_version="0.9.5", memmap_checksum="a" * 64, candle_checksum="b" * 64)
            .replace('dependencies = ["memmap2"]', 'dependencies = ["memmap2", "unexpected"]')
        )
        log_path = self.root / "cargo-argv.log"
        cargo = fake_bin / "cargo"
        cargo.write_text(
            "#!/bin/sh\n"
            "set -eu\n"
            "test \"${RUSTUP_TOOLCHAIN:-}\" = '1.99.0'\n"
            f"printf '%s\\n' \"$*\" >> {log_path}\n"
            "case \"$1\" in\n"
            f"metadata) case \"$*\" in *--all-features*) {'printf \'\\nwhole-mutation\\n\' >> \"$PWD/Cargo.lock\"; ' if mutate_whole else ''}cat {whole_metadata_path} ;; *) {'cp ' + str(mutated_base_lock) + ' \"$PWD/Cargo.lock\"; ' if mutate_base else ''}cat {metadata_path} ;; esac ;;\n"
            f"tree) case \"$8\" in normal) printf '%s\\n' 'moshi -> candle normal' ;; build) printf '%s\\n' 'moshi -> build-helper build' ;; features) printf '%s\\n' 'moshi features: default-disabled'; {'if test -f ' + str(post_update_marker) + '; then printf \'\\nmutated\\n\' >> \"$PWD/Cargo.lock\"; fi' if mutate_final else ':'} ;; esac ;;\n"
            f"update) cp {patched_lock} \"$PWD/Cargo.lock\"; touch {post_update_marker} ;;\n"
            "*) exit 90 ;;\n"
            "esac\n"
        )
        cargo.chmod(cargo.stat().st_mode | stat.S_IXUSR)
        rustc = fake_bin / "rustc"
        rustc.write_text(
            "#!/bin/sh\nset -eu\n"
            "test \"${RUSTUP_TOOLCHAIN:-}\" = '1.99.0'\n"
            f"printf '%s\\n' 'rustc {rust_version}' 'binary: rustc' 'commit-hash: synthetic' 'host: {rust_host}'\n"
        )
        rustc.chmod(rustc.stat().st_mode | stat.S_IXUSR)
        return fake_bin, log_path

    def _run_full_capture(self, *, via_main: bool = False, rust_version: str = "1.99.0", rust_host: str = leaf.TARGET, metadata: str | None = None, whole_metadata: str | None = None, output_name: str | None = None, mutate_base: bool = False, mutate_final: bool = False, mutate_whole: bool = False) -> tuple[Path, Path]:
        fake_bin, log_path = self._full_capture_environment(rust_version=rust_version, rust_host=rust_host, metadata=metadata, whole_metadata=whole_metadata, mutate_base=mutate_base, mutate_final=mutate_final, mutate_whole=mutate_whole)
        output = self.root / (output_name or ("main-output" if via_main else "capture-output"))
        fetcher = self.fixture_fetcher()
        environment = {"PATH": f"{fake_bin}:/usr/bin:/bin", "VOKRA_CANDLE_REMOTE_LINUX": "1", "VAST_API_KEY": "must-not-pass"}
        with mock.patch.dict(os.environ, environment, clear=False):
            if via_main:
                with mock.patch.object(leaf, "_bounded_url_fetch", side_effect=fetcher), mock.patch.object(leaf.platform, "system", return_value="Linux"):
                    self.assertEqual(leaf.main([str(output), "--deadline-seconds", "30"]), 0)
            else:
                leaf.capture(output, deadline_seconds=30, source_fetcher=fetcher, platform_name="Linux")
        return output, log_path

    def test_full_capture_pipeline_and_main_success(self) -> None:
        output, log_path = self._run_full_capture()
        packet = json.loads((output / "metadata.json").read_text())
        self.assertEqual(packet["status"], "METADATA_CAPTURED_REVIEW_PENDING")
        self.assertEqual(packet["workspace_member_packages"], ["moshi"])
        initial = json.loads((output / "initial_metadata.json").read_text())
        self.assertEqual(initial["status"], "INITIAL_METADATA_UNREVIEWED")
        self.assertEqual(initial["source_commit"], leaf.SOURCE_COMMIT)
        self.assertEqual(initial["target"], leaf.TARGET)
        self.assertTrue(initial["metadata"]["packages"])
        self.assertEqual(initial["approval"], None)
        self.assertEqual(initial["upload"], "NO_UPLOAD")
        self.assertEqual(initial["model_execution"], "NOT_AUTHORIZED")
        self.assertLessEqual((output / "initial_metadata.json").stat().st_size, leaf.MAX_JSON_BYTES)
        whole = json.loads((output / "whole_lock_metadata.json").read_text())
        self.assertEqual(whole["status"], "WHOLE_LOCK_METADATA_UNREVIEWED")
        self.assertIsNone(whole["target"])
        self.assertIsNone(whole["platform_filter"])
        self.assertEqual(whole["cpu_target"], initial["target"])
        self.assertEqual(whole["cpu_execution_profile"], "NOT_CPU_EXECUTION_PROFILE")
        self.assertEqual(whole["gpu_execution"], "NOT_AUTHORIZED")
        self.assertEqual(whole["model_execution"], "NOT_AUTHORIZED")
        self.assertEqual(whole["approval"], None)
        self.assertEqual(whole["upload"], "NO_UPLOAD")
        self.assertEqual(whole["projection_lock_before"], whole["projection_lock_after"])
        self.assertEqual(whole["projection_lock_after"]["sha256"], leaf.sha256_bytes((output / "base/Cargo.lock").read_bytes()))
        self.assertIn("version = 4", whole["postcommand_projection_lock"])
        self.assertLessEqual((output / "whole_lock_metadata.json").stat().st_size, leaf.MAX_JSON_BYTES)
        self.assertTrue(packet["initial"]["trees"]["normal"].strip())
        self.assertTrue(packet["initial"]["trees"]["build"].strip())
        self.assertEqual(set(packet["lock_receipts"]), {"original", "projected_base", "patched", "final_projection"})
        self.assertEqual(packet["lock_receipts"]["patched"]["sha256"], packet["lock_receipts"]["final_projection"]["sha256"])
        self.assertEqual(len(log_path.read_text().splitlines()), 10)
        self.assertTrue((output / "source/rust/Cargo.lock").is_file())
        self.assertTrue((output / "projection/Cargo.lock").is_file())
        self.assertTrue((output / "base/Cargo.lock").is_file())
        self.assertTrue((output / "patched/Cargo.lock").is_file())
        self.assertNotEqual((output / "source/rust/Cargo.lock").read_bytes(), (output / "patched/Cargo.lock").read_bytes())
        main_output, _ = self._run_full_capture(via_main=True)
        self.assertTrue((main_output / "metadata.json").is_file())

    def test_full_capture_intended_failures_for_gpu_and_toolchain(self) -> None:
        gpu_output = self.root / "gpu-output"
        fake_bin, _ = self._full_capture_environment()
        (self.root / "metadata-fixture.json").write_text(metadata_json(gpu_dependency=True))
        with mock.patch.dict(os.environ, {"PATH": f"{fake_bin}:/usr/bin:/bin", "VOKRA_CANDLE_REMOTE_LINUX": "1"}, clear=False):
            with self.assertRaises(leaf.CaptureError):
                leaf.capture(gpu_output, deadline_seconds=30, source_fetcher=self.fixture_fetcher(), platform_name="Linux")
        for version, host in (("1.98.0", leaf.TARGET), ("1.99.0", "aarch64-unknown-linux-gnu")):
            with self.subTest(version=version, host=host):
                fake_bin, _ = self._full_capture_environment(rust_version=version, rust_host=host)
                log_path = self.root / "cargo-argv.log"
                if log_path.exists():
                    log_path.unlink()
                with mock.patch.dict(os.environ, {"PATH": f"{fake_bin}:/usr/bin:/bin", "VOKRA_CANDLE_REMOTE_LINUX": "1"}, clear=False):
                    with self.assertRaises(leaf.CaptureError):
                        leaf.capture(self.root / f"toolchain-{version}-{host}", deadline_seconds=30, source_fetcher=self.fixture_fetcher(), platform_name="Linux")
                self.assertFalse(log_path.exists(), "Cargo must not run before toolchain rejection")

    def test_final_projection_lock_mutation_is_rejected(self) -> None:
        output = self.root / "final-mutation-output"
        with self.assertRaisesRegex(leaf.CaptureError, "projection Cargo.lock changed after patched lock receipt"):
            self._run_full_capture(output_name=output.name, mutate_final=True)
        self.assertFalse((output / "metadata.json").exists())

    def test_initial_metadata_survives_retained_base_lock_failure(self) -> None:
        output = self.root / "base-failure-output"
        with self.assertRaisesRegex(leaf.CaptureError, "dependency label is unresolved or ambiguous"):
            self._run_full_capture(output_name=output.name, mutate_base=True)
        initial = json.loads((output / "initial_metadata.json").read_text())
        self.assertEqual(initial["status"], "INITIAL_METADATA_UNREVIEWED")
        self.assertTrue(initial["metadata"]["packages"])
        self.assertEqual(initial["approval"], None)
        self.assertEqual(initial["upload"], "NO_UPLOAD")
        self.assertEqual(initial["model_execution"], "NOT_AUTHORIZED")
        self.assertTrue((output / "base/Cargo.lock").is_file())
        self.assertFalse((output / "patched/Cargo.lock").exists())
        self.assertFalse((output / "metadata.json").exists())

    def test_whole_lock_evidence_survives_strict_base_failure(self) -> None:
        output = self.root / "base-failure-whole-output"
        with self.assertRaisesRegex(leaf.CaptureError, "dependency label is unresolved or ambiguous"):
            self._run_full_capture(output_name=output.name, mutate_base=True)
        whole = json.loads((output / "whole_lock_metadata.json").read_text())
        self.assertEqual(whole["status"], "WHOLE_LOCK_METADATA_UNREVIEWED")
        self.assertEqual(whole["projection_lock_before"], whole["projection_lock_after"])
        self.assertFalse((output / "metadata.json").exists())

    def test_whole_lock_mutation_is_recovered_then_rejected(self) -> None:
        output = self.root / "whole-mutation-output"
        with self.assertRaisesRegex(leaf.CaptureError, "all-features metadata mutated projection Cargo.lock"):
            self._run_full_capture(output_name=output.name, mutate_whole=True)
        whole = json.loads((output / "whole_lock_metadata.json").read_text())
        self.assertNotEqual(whole["projection_lock_before"], whole["projection_lock_after"])
        self.assertIn("whole-mutation", whole["postcommand_projection_lock"])
        self.assertFalse((output / "metadata.json").exists())

    def test_whole_metadata_gpu_features_do_not_bypass_cpu_guard(self) -> None:
        output, _ = self._run_full_capture(whole_metadata=whole_metadata_json(gpu=True))
        initial = json.loads((output / "initial_metadata.json").read_text())
        whole = json.loads((output / "whole_lock_metadata.json").read_text())
        self.assertEqual(initial["graph"]["gpu_features_active"], [])
        self.assertEqual(whole["metadata"]["resolve"]["nodes"][0]["features"], ["cuda"])

    def test_cpu_gpu_activation_rejects_before_whole_lock_evidence(self) -> None:
        output = self.root / "cpu-gpu-output"
        with self.assertRaisesRegex(leaf.CaptureError, "GPU feature became active"):
            self._run_full_capture(output_name=output.name, metadata=metadata_json(gpu=True))
        self.assertFalse((output / "initial_metadata.json").exists())
        self.assertFalse((output / "whole_lock_metadata.json").exists())

    def test_symlink_ancestor_and_source_trickle_deadline(self) -> None:
        real_parent = self.root / "real-parent"
        real_parent.mkdir()
        alias = self.root / "alias-parent"
        alias.symlink_to(real_parent, target_is_directory=True)
        with mock.patch.dict(os.environ, {"VOKRA_CANDLE_REMOTE_LINUX": "1"}, clear=False):
            with self.assertRaises(leaf.CaptureError):
                leaf.capture(alias / "fresh-output", deadline_seconds=5, source_fetcher=self.fixture_fetcher(), platform_name="Linux")

        class TrickleResponse:
            headers: dict[str, str] = {}
            def __init__(self) -> None:
                self.offset = 0
            def __enter__(self) -> "TrickleResponse":
                return self
            def __exit__(self, *_args: object) -> None:
                return None
            def geturl(self) -> str:
                return leaf.SOURCE_URLS["README.md"]
            def read(self, size: int) -> bytes:
                time.sleep(0.02)
                if self.offset >= 7:
                    return b""
                chunk = b"readme\n"[self.offset:self.offset + min(size, 1)]
                self.offset += len(chunk)
                return chunk

        with mock.patch.object(leaf.urllib.request, "urlopen", return_value=TrickleResponse()):
            with self.assertRaises(leaf.CaptureError):
                leaf._bounded_url_fetch(leaf.SOURCE_URLS["README.md"], time.monotonic() + 0.01)

    def test_nonfinite_deadline_rejected_before_fetch_or_output(self) -> None:
        for value in ("nan", "inf", "-inf"):
            with self.subTest(value=value):
                captured = False
                def should_not_fetch(_url: str, _deadline: float) -> bytes:
                    nonlocal captured
                    captured = True
                    raise AssertionError("source fetch must not start")
                output = self.root / f"nonfinite-{value.replace('-', 'neg-')}"
                with mock.patch.dict(os.environ, {"VOKRA_CANDLE_REMOTE_LINUX": "1"}, clear=False):
                    with self.assertRaises(leaf.CaptureError):
                        leaf.capture(output, deadline_seconds=float(value), source_fetcher=should_not_fetch, platform_name="Linux")
                self.assertFalse(captured)
                self.assertFalse(output.exists())
                with mock.patch.dict(os.environ, {"VOKRA_CANDLE_REMOTE_LINUX": "1"}, clear=False), mock.patch.object(leaf.platform, "system", return_value="Linux"):
                    deadline_arg = f"--deadline-seconds={value}" if value.startswith("-") else value
                    main_args = [str(output), "--deadline-seconds", deadline_arg] if not value.startswith("-") else [str(output), deadline_arg]
                    self.assertEqual(leaf.main(main_args), 2)
                self.assertFalse(output.exists())

    def test_blocking_source_read_is_interrupted_and_timer_restored(self) -> None:
        class BlockingResponse:
            headers: dict[str, str] = {}
            def __enter__(self) -> "BlockingResponse":
                return self
            def __exit__(self, *_args: object) -> None:
                return None
            def geturl(self) -> str:
                return leaf.SOURCE_URLS["README.md"]
            def read(self, _size: int) -> bytes:
                time.sleep(5)
                return b""

        old_handler = leaf.signal.getsignal(leaf.signal.SIGALRM)
        old_timer = leaf.signal.setitimer(leaf.signal.ITIMER_REAL, 0.0)
        started = time.monotonic()
        try:
            with mock.patch.object(leaf.urllib.request, "urlopen", return_value=BlockingResponse()):
                with self.assertRaises(leaf.CaptureError):
                    leaf._bounded_url_fetch(leaf.SOURCE_URLS["README.md"], started + 0.05)
            self.assertLess(time.monotonic() - started, 1.0)
            self.assertIs(leaf.signal.getsignal(leaf.signal.SIGALRM), old_handler)
            self.assertEqual(leaf.signal.setitimer(leaf.signal.ITIMER_REAL, 0.0), (0.0, 0.0))
        finally:
            leaf.signal.signal(leaf.signal.SIGALRM, old_handler)
            if old_timer != (0.0, 0.0):
                leaf.signal.setitimer(leaf.signal.ITIMER_REAL, *old_timer)

    def test_hung_child_descendant_is_terminated_and_deadline_precheck(self) -> None:
        marker = self.root / "descendant.pid"
        leader = self.root / "leader.pid"
        cargo = self._fake_cargo(
            f"echo $$ > {leader}; sh -c 'trap \"\" INT TERM; echo $$ > {marker}; while :; do sleep 0.05; done' & wait"
        )
        argv = [str(cargo), "metadata", "--format-version", "1", "--filter-platform", leaf.TARGET, "--no-default-features"]
        with self.assertRaises(leaf.CaptureError):
            leaf.run_bounded_subprocess(argv, cwd=self.root, env={"PATH": "/bin:/usr/bin"}, deadline=time.monotonic() + 0.8)
        for _ in range(20):
            if marker.exists():
                break
            time.sleep(0.05)
        pid = int(marker.read_text())
        leader_pid = int(leader.read_text())
        self.assertNotEqual(pid, leader_pid)
        members = leaf._live_process_group_members(leader_pid)
        if members is not None:
            self.assertEqual(members, [])
        else:
            # The managed macOS sandbox denies ps(1); kill(pid, 0) is the
            # portable fallback proof that this recorded descendant is gone.
            with self.assertRaises(ProcessLookupError):
                os.kill(pid, 0)
        with self.assertRaises(leaf.CaptureError):
            leaf.run_bounded_subprocess(argv, cwd=self.root, env={"PATH": "/bin:/usr/bin"}, deadline=time.monotonic() - 1)

    def test_secret_environment_is_not_passed(self) -> None:
        marker = self.root / "env.txt"
        cargo = self._fake_cargo(f"env > {marker}; exit 0")
        argv = [str(cargo), "metadata", "--format-version", "1", "--filter-platform", leaf.TARGET, "--no-default-features"]
        with mock.patch.dict(os.environ, {"HF_TOKEN": "secret", "VASTAI_API_KEY": "secret", "VAST_API_KEY": "secret"}, clear=False):
            env = leaf.clean_cargo_environment(self.root / "cargo-home")
        leaf.run_bounded_subprocess(argv, cwd=self.root, env=env, deadline=time.monotonic() + 5)
        text = marker.read_text()
        self.assertNotIn("HF_TOKEN", text)
        self.assertNotIn("VASTAI_API_KEY", text)
        self.assertNotIn("VAST_API_KEY", text)


if __name__ == "__main__":
    unittest.main()
