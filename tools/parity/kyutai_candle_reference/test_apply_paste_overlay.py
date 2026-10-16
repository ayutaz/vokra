#!/usr/bin/env -S uv run --no-project --python 3.12
"""Model-free orchestration tests for the real paste overlay application path.

The test replaces only network/Cargo/source-packet boundaries.  The production
``apply_overlay`` entry point, archive extraction, manifest writes, tree proof,
lock delta, and graph validation remain real code under test.
"""

from __future__ import annotations

import copy
import gzip
import hashlib
import importlib.util
import io
import json
import os
import tarfile
import tempfile
import time
import tomllib
import unittest
from pathlib import Path
from unittest import mock


# Exact package identities from the accepted CPU metadata snapshot
# (candidate_cpu_metadata.json, SHA-256 9c351...); the bytes are not loaded by
# CI.  This is a source-only fixture, not a claim of local execution.
CPU_PACKAGE_KEYS = frozenset(tuple(item) for item in json.loads("""[[\"aho-corasick\",\"1.1.3\"],[\"autocfg\",\"1.4.0\"],[\"bit-set\",\"0.5.3\"],[\"bit-vec\",\"0.6.3\"],[\"bitflags\",\"1.3.2\"],[\"bitflags\",\"2.9.0\"],[\"bytemuck\",\"1.22.0\"],[\"bytemuck_derive\",\"1.9.3\"],[\"byteorder\",\"1.5.0\"],[\"candle-core\",\"0.9.1\"],[\"candle-nn\",\"0.9.1\"],[\"candle-transformers\",\"0.9.1\"],[\"cfg-if\",\"1.0.0\"],[\"crc32fast\",\"1.4.2\"],[\"crossbeam-deque\",\"0.8.6\"],[\"crossbeam-epoch\",\"0.9.20\"],[\"crossbeam-utils\",\"0.8.21\"],[\"displaydoc\",\"0.2.5\"],[\"dyn-stack\",\"0.10.0\"],[\"dyn-stack\",\"0.13.0\"],[\"either\",\"1.15.0\"],[\"equivalent\",\"1.0.2\"],[\"fancy-regex\",\"0.13.0\"],[\"gemm\",\"0.17.1\"],[\"gemm\",\"0.18.2\"],[\"gemm-c32\",\"0.17.1\"],[\"gemm-c32\",\"0.18.2\"],[\"gemm-c64\",\"0.17.1\"],[\"gemm-c64\",\"0.18.2\"],[\"gemm-common\",\"0.17.1\"],[\"gemm-common\",\"0.18.2\"],[\"gemm-f16\",\"0.17.1\"],[\"gemm-f16\",\"0.18.2\"],[\"gemm-f32\",\"0.17.1\"],[\"gemm-f32\",\"0.18.2\"],[\"gemm-f64\",\"0.17.1\"],[\"gemm-f64\",\"0.18.2\"],[\"getrandom\",\"0.3.2\"],[\"half\",\"2.6.0\"],[\"hashbrown\",\"0.15.2\"],[\"indexmap\",\"2.9.0\"],[\"itoa\",\"1.0.15\"],[\"libc\",\"0.2.171\"],[\"libloading\",\"0.8.6\"],[\"libm\",\"0.2.11\"],[\"memchr\",\"2.7.4\"],[\"memmap2\",\"0.9.11\"],[\"moshi\",\"0.6.4\"],[\"num\",\"0.4.3\"],[\"num-bigint\",\"0.4.6\"],[\"num-complex\",\"0.4.6\"],[\"num-integer\",\"0.1.46\"],[\"num-iter\",\"0.1.45\"],[\"num-rational\",\"0.4.2\"],[\"num-traits\",\"0.2.19\"],[\"num_cpus\",\"1.16.0\"],[\"num_enum\",\"0.7.3\"],[\"num_enum_derive\",\"0.7.3\"],[\"once_cell\",\"1.21.3\"],[\"paste\",\"1.0.15\"],[\"pin-project-lite\",\"0.2.16\"],[\"ppv-lite86\",\"0.2.21\"],[\"proc-macro-crate\",\"3.3.0\"],[\"proc-macro2\",\"1.0.94\"],[\"pulp\",\"0.18.22\"],[\"pulp\",\"0.21.4\"],[\"quote\",\"1.0.40\"],[\"rand\",\"0.9.3\"],[\"rand_chacha\",\"0.9.0\"],[\"rand_core\",\"0.9.3\"],[\"rand_distr\",\"0.5.1\"],[\"raw-cpuid\",\"10.7.0\"],[\"raw-cpuid\",\"11.5.0\"],[\"rayon\",\"1.10.0\"],[\"rayon-core\",\"1.12.1\"],[\"reborrow\",\"0.5.5\"],[\"regex-automata\",\"0.4.9\"],[\"regex-syntax\",\"0.8.5\"],[\"ryu\",\"1.0.20\"],[\"safetensors\",\"0.4.5\"],[\"seq-macro\",\"0.3.6\"],[\"serde\",\"1.0.219\"],[\"serde_derive\",\"1.0.219\"],[\"serde_json\",\"1.0.140\"],[\"serde_plain\",\"1.0.2\"],[\"stable_deref_trait\",\"1.2.0\"],[\"syn\",\"2.0.100\"],[\"synstructure\",\"0.13.1\"],[\"thiserror\",\"1.0.69\"],[\"thiserror-impl\",\"1.0.69\"],[\"toml_datetime\",\"0.6.8\"],[\"toml_edit\",\"0.22.24\"],[\"tracing\",\"0.22.24\"],[\"tracing\",\"0.1.41\"],[\"tracing-attributes\",\"0.1.28\"],[\"tracing-core\",\"0.1.33\"],[\"ug\",\"0.4.0\"],[\"unicode-ident\",\"1.0.18\"],[\"version_check\",\"0.9.5\"],[\"winnow\",\"0.7.6\"],[\"yoke\",\"0.7.5\"],[\"yoke-derive\",\"0.7.5\"],[\"zerocopy\",\"0.8.24\"],[\"zerofrom\",\"0.1.6\"],[\"zerofrom-derive\",\"0.1.6\"],[\"zip\",\"1.1.4\"]]"""))


# The accepted CPU snapshot has one tracing identity; keep the compact fixture
# exact even if a copied diagnostic line is edited by hand.
CPU_PACKAGE_KEYS = CPU_PACKAGE_KEYS - {("tracing", "0.22.24")}
MODULE_PATH = Path(__file__).with_name("apply_paste_overlay.py")
SPEC = importlib.util.spec_from_file_location("apply_paste_overlay", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
apply = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(apply)


def load_real(path: Path, name: str, digest: str):
    return apply._module(name, path, digest)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def tarball(files: dict[str, bytes], root: str) -> tuple[bytes, int]:
    expanded = io.BytesIO()
    with tarfile.open(fileobj=expanded, mode="w", format=tarfile.GNU_FORMAT) as archive:
        directory = tarfile.TarInfo(root)
        directory.type = tarfile.DIRTYPE
        directory.mode = 0o755
        archive.addfile(directory)
        if any(path.startswith("src/") for path in files):
            source_dir = tarfile.TarInfo(f"{root}/src")
            source_dir.type = tarfile.DIRTYPE
            source_dir.mode = 0o755
            archive.addfile(source_dir)
        for path, body in files.items():
            member = tarfile.TarInfo(f"{root}/{path}")
            member.size = len(body)
            member.mode = 0o644
            archive.addfile(member, io.BytesIO(body))
    raw = io.BytesIO()
    with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
        compressed.write(expanded.getvalue())
    return raw.getvalue(), len(expanded.getvalue())


def member(path: str, body: bytes, *, text_ref: bool = False) -> dict[str, object]:
    row = {"path": path, "bytes": len(body), "sha256": digest(body)}
    if text_ref:
        row["text_ref"] = row["sha256"]
    return row


class FakeCargoBoundary:
    """Synthetic Cargo/network boundary with real lock/graph inputs."""

    def __init__(self, module, apply_module, plan):
        self.module = module
        self.apply_module = apply_module
        self.plan = plan
        self.projection: Path | None = None
        self.snapshots: list[dict[str, object]] = []
        self.operation_deadlines: list[float] = []
        self.source_bodies: dict[str, bytes] = {}

    def _lock_doc(self) -> dict:
        assert self.projection is not None
        return tomllib.loads((self.projection / "Cargo.lock").read_text())

    def _metadata(self, *, cpu_only: bool) -> dict:
        packages = []
        ids = {}
        lock = self.module.lock_packages(self._lock_doc())
        def selected(key):
            return (not cpu_only) or (key[0], key[1]) in CPU_PACKAGE_KEYS or (key[0], key[1]) == ("pastey", "0.2.3") and ("paste", "1.0.15") in CPU_PACKAGE_KEYS
        selected_keys = {key for key in lock if selected(key)}
        parent_names = {proposal["identity"].split("@", 1)[0] for proposal in self.plan["proposals"]}
        for key in selected_keys:
            name, version, source = key
            ids[key] = f"synthetic:{name}:{version}:{source or 'local'}"
        for key in lock:
            if key not in selected_keys:
                continue
            name, version, source = key
            declarations = []
            for dependency in lock[key].get("dependencies", []):
                target_key = self.module._lock_dependency_key(dependency, lock)
                declaration_source = target_key[2] if target_key[2] is not None else apply.REGISTRY
                if target_key[0] == "paste":
                    declaration = {"name": "paste", "source": declaration_source, "req": "^1.0.15"}
                elif target_key[0] == "pastey":
                    declaration = {"name": "pastey", "rename": "paste", "source": declaration_source, "req": "=0.2.3"}
                else:
                    declaration = {"name": target_key[0], "source": declaration_source, "req": "*"}
                declaration.update({"kind": None, "target": None, "optional": False, "features": [], "uses_default_features": True})
                declarations.append(declaration)
            path_package = source is None and name in parent_names
            package_root = (self.projection.parent / "vendor" / f"{name}-{version}") if path_package and self.projection is not None else Path(f"/synthetic/{name}-{version}")
            manifest_path = str(package_root / "Cargo.toml")
            if name == "pastey" and version == "0.2.3":
                targets = [{"name": "pastey", "kind": ["proc-macro"], "crate_types": ["proc-macro"], "src_path": str(package_root / "src/lib.rs"), "edition": "2018"}]
            else:
                targets = [{"name": name, "kind": ["lib"], "crate_types": ["lib"], "src_path": str(package_root / "src/lib.rs"), "edition": "2021"}]
            package = {
                "id": ids[key],
                "name": name,
                "version": version,
                "source": source,
                "manifest_path": manifest_path,
                "dependencies": declarations,
                "features": {},
                "targets": targets,
            }
            if name == "pastey" and version == "0.2.3":
                package.update({"license": "MIT OR Apache-2.0", "edition": "2018", "rust_version": "1.54"})
            packages.append(package)
        nodes = []
        for key in selected_keys:
            edges = []
            for dependency in lock[key].get("dependencies", []):
                target_key = self.module._lock_dependency_key(dependency, lock)
                if target_key not in selected_keys:
                    continue
                edge_name = "paste" if target_key[0] == "pastey" else target_key[0]
                edges.append({"name": edge_name, "pkg": ids[target_key], "dep_kinds": [{"kind": None, "target": None}]})
            nodes.append({"id": ids[key], "deps": edges, "dependencies": [edge["pkg"] for edge in edges], "features": []})
        root = next(ids[key] for key in selected_keys if key[0] == "moshi")
        return {
            "packages": packages,
            "workspace_members": [root],
            "resolve": {"root": root, "nodes": nodes},
            "gpu_features_active": [],
        }

    def capture(self, packet_root: Path, **_kwargs) -> None:
        files = {}
        for relative, body in self.source_bodies.items():
            path = packet_root / "source" / "rust" / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(body)
            files[relative] = {
                "synthetic": True,
                "bytes": len(body),
                "sha256": digest(body),
                "git_blob_sha1": self.module.git_blob_sha1(body),
            }
        (packet_root / "source_receipt.json").write_text(json.dumps({"synthetic": True, "files": files}) + "\n")

    def clean_environment(self, _cargo_home: Path) -> dict[str, str]:
        return {"CARGO_HOME": str(_cargo_home)}

    def command(self, argv, cwd, env, deadline, **_kwargs):
        assert self.projection == cwd
        self.operation_deadlines.append(deadline)
        args = tuple(argv[1:])
        before = digest((cwd / "Cargo.lock").read_bytes())
        if args == apply.CPU_ARGS:
            self._write_overlay_lock()
        after = digest((cwd / "Cargo.lock").read_bytes())
        self.snapshots.append({"args": args, "before": before, "after": after})
        metadata = self._metadata(cpu_only=args in (apply.CPU_ARGS, apply.CPU_LOCKED_ARGS))
        if args in (apply.CPU_ARGS, apply.CPU_LOCKED_ARGS):
            assert len(metadata["packages"]) == 105
        elif args == apply.WHOLE_ARGS:
            assert len(metadata["packages"]) == 154
        return {"returncode": 0, "stdout": json.dumps(metadata), "stderr": ""}

    def _write_overlay_lock(self) -> None:
        assert self.projection is not None
        document = self._lock_doc()
        parent_keys = {(p["identity"].split("@", 1)[0], p["identity"].split("@", 1)[1], apply.REGISTRY) for p in self.plan["proposals"]}
        package_rows = []
        for record in document["package"]:
            key = (record["name"], record["version"], record.get("source"))
            replacement = dict(record)
            if key in parent_keys:
                replacement.pop("source", None)
                replacement.pop("checksum", None)
            elif key == ("paste", "1.0.15", apply.REGISTRY):
                replacement["name"] = "pastey"
                replacement["version"] = "0.2.3"
                replacement["checksum"] = "2ee67f1008b1ba2321834326597b8e186293b049a023cdef258527550b9935b4"
            if isinstance(replacement.get("dependencies"), list):
                replacement["dependencies"] = [
                    "pastey 0.2.3" if value in {"paste", "paste 1.0.15"} else value
                    for value in replacement["dependencies"]
                ]
            package_rows.append(replacement)
        self._write_lock({key: value for key, value in document.items() if key != "package"}, package_rows)

    def _write_lock(self, root: dict, packages: list[dict]) -> None:
        assert self.projection is not None
        lines: list[str] = []
        for key, value in root.items():
            lines.append(f"{key} = {json.dumps(value)}")
        lines.append("")
        for package in packages:
            lines.append("[[package]]")
            for key, value in package.items():
                if value is not None:
                    lines.append(f"{key} = {json.dumps(value)}")
            lines.append("")
        (self.projection / "Cargo.lock").write_text("\n".join(lines))


class ApplyPasteOverlayTests(unittest.TestCase):
    def _planner(self):
        return load_real(apply.ROOT / "tools/parity/kyutai_candle_reference/prepare_paste_overlay.py", "planner-test", apply.PLANNER_SHA256)

    def test_load_inputs_is_genuine_authenticated_fixture(self) -> None:
        parent, pastey, candidate, plan = apply._load_inputs(self._planner())
        self.assertEqual(len(plan["proposals"]), 16)
        self.assertEqual(plan["parent_counts"]["total"], 16)
        self.assertEqual(len(candidate), apply.CANDIDATE_BYTES)
        self.assertEqual(digest(candidate), apply.CANDIDATE_SHA256)
        self.assertEqual(parent["identity"], {"collected": 16, "expected": 16, "missing": []})
        self.assertEqual(pastey["manifest"]["package"]["name"], "pastey")
        self.assertEqual(sum(len(proposal["expected_unchanged_members"]) for proposal in plan["proposals"]), 292)
        for proposal in plan["proposals"]:
            diff_lines = proposal["unified_diff"].splitlines()
            self.assertEqual(len([line for line in diff_lines if line.startswith("-") and not line.startswith("---")]), 1)
            self.assertEqual(len([line for line in diff_lines if line.startswith("+") and not line.startswith("+++")]), 2)

    def _synthetic_inputs(self, planner):
        parent, pastey, candidate, plan = apply._load_inputs(planner)
        parent = copy.deepcopy(parent)
        pastey = copy.deepcopy(pastey)
        archives_by_key = {f'{p["identity"].split("@", 1)[0]}@{p["identity"].split("@", 1)[1]}': p for p in plan["proposals"]}
        fetches: dict[str, bytes] = {}
        for row in parent["archives"]:
            key = f'{row["name"]}@{row["version"]}'
            proposal = archives_by_key[key]
            manifest = parent["text_store"][row["manifest"]["text_ref"]].encode()
            original = b"[package]\nname = \"fixture-original\"\nversion = \"0.0.0\"\n"
            rust = b"pub const FIXTURE_SOURCE_UNCHANGED: &str = \"moshi-source\";\n"
            license_body = b"MIT\n"
            vcs = b"{\"git\":\"synthetic\"}\n"
            files = {"Cargo.toml": manifest, "Cargo.toml.orig": original, "src/lib.rs": rust, "LICENSE": license_body, ".cargo_vcs_info.json": vcs}
            target_count = len(proposal["expected_unchanged_members"]) + 1
            for filler_index in range(target_count - len(files)):
                files[f"synthetic/member-{filler_index:02d}.txt"] = f"fixture member {row['name']} {filler_index}\n".encode()
            raw, expanded = tarball(files, f'{row["name"]}-{row["version"]}')
            row["archive_bytes"] = len(raw)
            row["archive_sha256"] = digest(raw)
            row["expanded_bytes"] = expanded
            row["regular_members"] = [member(path, body) for path, body in files.items()]
            row["rust_sources"] = {"src/lib.rs": member("src/lib.rs", rust)}
            row["license_members"] = {"LICENSE": member("LICENSE", license_body, text_ref=True)}
            row["vcs_member"] = member(".cargo_vcs_info.json", vcs, text_ref=True)
            url = f"https://static.crates.io/crates/{row['name']}/{row['name']}-{row['version']}.crate"
            fetches[url] = raw

        paste_manifest = b"[package]\nname = \"pastey\"\nversion = \"0.2.3\"\n"
        paste_license = b"MIT OR Apache-2.0\n"
        paste_raw, paste_expanded = tarball({"Cargo.toml": paste_manifest, "LICENSE": paste_license}, "pastey-0.2.3")
        pastey["members"] = [
            {"path": "pastey-0.2.3/Cargo.toml", "bytes": len(paste_manifest), "sha256": digest(paste_manifest)},
            {"path": "pastey-0.2.3/LICENSE", "bytes": len(paste_license), "sha256": digest(paste_license)},
        ]
        pastey["expanded_bytes"] = paste_expanded
        pastey["archive"]["compressed_bytes"] = len(paste_raw)
        pastey["archive"]["compressed_sha256"] = digest(paste_raw)
        pastey["archive"]["expanded_bytes"] = paste_expanded
        fetches[pastey["archive"]["url"]] = paste_raw
        return parent, pastey, candidate, plan, fetches

    def _module_factory(self, cargo_boundary: FakeCargoBoundary):
        real_module = apply._module

        def load(name, path, digest_value):
            module = real_module(name, path, digest_value)
            if name == "candle_overlay_collector":
                cargo_boundary.module = module
                source_paths = (
                    "Cargo.toml", "Cargo.lock", "moshi-core/src/lib.rs",
                    *[f"moshi-core/src/synthetic_{index:02d}.rs" for index in range(20)],
                )
                source_bodies = {
                    path: (
                        '[workspace]\nmembers = ["root", "other"]\nresolver = "2"\n'
                        if path == "Cargo.toml" else
                        b"synthetic source lock boundary\n" if path == "Cargo.lock" else
                        b'pub const FIXTURE_SOURCE_UNCHANGED: &str = "moshi-source";\n' if path == "moshi-core/src/lib.rs" else
                        f"pub const SYNTHETIC_SOURCE_{index:02d}: &str = \"unchanged\";\n".encode()
                    )
                    for index, path in enumerate(source_paths)
                }
                source_bodies = {path: body.encode() if isinstance(body, str) else body for path, body in source_bodies.items()}
                module.SOURCE_FILES = {path: (module.git_blob_sha1(body), len(body)) for path, body in source_bodies.items()}
                module.SOURCE_URLS = {path: f"synthetic://{path}" for path in source_bodies}
                cargo_boundary.source_bodies = source_bodies
                module.capture = cargo_boundary.capture
                module.clean_cargo_environment = cargo_boundary.clean_environment
                module.run_bounded_subprocess = cargo_boundary.command
                module._toolchain_probe = lambda *_args, **_kwargs: None
            elif name == "candle_overlay_pin":
                def authenticate(_packet_root, collector):
                    return {"receipt": {"files": {
                        path: {"synthetic": True, "bytes": len(body), "sha256": digest(body), "git_blob_sha1": collector.git_blob_sha1(body)}
                        for path, body in cargo_boundary.source_bodies.items()
                    }}}

                module._authenticate_packet = authenticate

                def rebuild(_packet, workspace, _collector, candidate_bytes):
                    projection = workspace / "projection"
                    projection.mkdir(parents=True)
                    for relative, body in cargo_boundary.source_bodies.items():
                        destination = projection / relative
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        if relative == "Cargo.toml":
                            destination.write_bytes(_collector.replace_workspace_members(body.decode()).encode())
                        elif relative == "Cargo.lock":
                            destination.write_bytes(candidate_bytes)
                        else:
                            destination.write_bytes(body)
                    cargo_boundary.projection = projection
                    return projection

                module._rebuild_projection = rebuild
            return module

        return load

    def _run_apply(
        self,
        fetcher=None,
        *,
        deadline_seconds=30.0,
        fail_final_write=False,
        expire_after_final_write=False,
    ):
        planner = self._planner()
        parent, pastey, candidate, plan, synthetic_fetches = self._synthetic_inputs(planner)
        boundary = FakeCargoBoundary(None, apply, plan)
        module_factory = self._module_factory(boundary)
        def load_inputs(_planner):
            return parent, pastey, candidate, plan

        archive_fetcher = fetcher or (lambda url, _size, _deadline: synthetic_fetches[url])

        real_rewrite = apply._rewrite
        real_monotonic = time.monotonic
        clock_state: dict[str, float] = {}
        self._last_final_write_completed = False

        def rewrite(path, value):
            result = real_rewrite(path, value)
            if value.get("status") == "SOURCE_APPLICATION_REVIEW_PENDING":
                self._last_final_write_completed = True
                if expire_after_final_write:
                    self.assertTrue(boundary.operation_deadlines)
                    # Cargo's operation deadline leaves the documented outer
                    # deadline reserve for blocked-receipt recovery.  Move
                    # just past that inner cutoff, without sleeping.
                    clock_state["now"] = boundary.operation_deadlines[-1] + 0.01
                if fail_final_write:
                    raise OSError("synthetic final receipt write failure")
            return result

        def monotonic():
            return clock_state.get("now", real_monotonic())

        # The production loader is called once in _synthetic_inputs above; the
        # application path itself still uses its genuine entrypoint and all
        # source/lock/graph validators.  Only authenticated network bytes and
        # Cargo process output are replaced at their explicit boundaries.
        with mock.patch.object(apply, "_module", side_effect=module_factory), mock.patch.object(apply, "_load_inputs", side_effect=load_inputs), mock.patch.object(apply.shutil, "which", return_value="/usr/bin/cargo"), mock.patch.dict(os.environ, {"VOKRA_CANDLE_REMOTE_LINUX": "1"}, clear=False):
            with mock.patch.object(apply, "_rewrite", side_effect=rewrite), mock.patch.object(apply.time, "monotonic", side_effect=monotonic):
                with tempfile.TemporaryDirectory(prefix="paste-overlay-apply-") as directory:
                    root = Path(directory)
                    packet = root / "packet"
                    packet.mkdir()
                    output = root / "output"
                    workspace = root / "workspace"
                    result = None
                    error = None
                    try:
                        result = apply.apply_overlay(packet, output, workspace, platform_name="Linux", deadline_seconds=deadline_seconds, archive_fetcher=archive_fetcher)
                    except Exception as exc:  # the failure assertion inspects the receipt below
                        error = exc
                    receipt = json.loads((output / "application_receipt.json").read_text())
                    source_path = workspace / "projection/moshi-core/src/lib.rs"
                    source = source_path.read_bytes() if source_path.is_file() else b""
                    candidate_path = output / "candidate.lock"
                    candidate_has_source_none = candidate_path.is_file() and any(
                        row.get("source") is None for row in tomllib.loads(candidate_path.read_text()).get("package", [])
                    )
                    metadata_receipts = {
                        name: json.loads((output / name).read_text())
                        for name in ("baseline_cpu_metadata.json", "baseline_whole_metadata.json", "overlay_cpu_metadata.json", "overlay_whole_metadata.json")
                        if (output / name).is_file()
                    }
                    return result, error, receipt, boundary.snapshots, source, synthetic_fetches, candidate_has_source_none, metadata_receipts

    def test_apply_overlay_runs_real_sixteen_archive_pipeline(self) -> None:
        result, error, receipt, snapshots, source, _fetches, candidate_has_source_none, metadata_receipts = self._run_apply()
        self.assertIsNone(error, receipt.get("error"))
        self.assertEqual(result["status"], "SOURCE_APPLICATION_REVIEW_PENDING")
        self.assertEqual(len(receipt["archives"]), 17)
        self.assertEqual(len(receipt["member_postchecks"]), 16)
        self.assertEqual(receipt["lock_delta"]["status"], "COMPLETE_ORDERED_DIFF_REVIEW_PENDING")
        pastey_records = [row["record"] for row in receipt["lock_delta"]["added_records"] if row["package"].startswith("('pastey', '0.2.3'")]
        self.assertEqual(len(pastey_records), 1)
        self.assertEqual(pastey_records[0]["checksum"], "2ee67f1008b1ba2321834326597b8e186293b049a023cdef258527550b9935b4")
        self.assertEqual(len([s for s in snapshots if s["args"] == apply.CPU_ARGS]), 1)
        overlay_cpu = next(s for s in snapshots if s["args"] == apply.CPU_ARGS)
        self.assertNotEqual(overlay_cpu["before"], overlay_cpu["after"])
        for snapshot in snapshots:
            if snapshot["args"] != apply.CPU_ARGS:
                self.assertEqual(snapshot["before"], snapshot["after"])
        self.assertEqual(source, b'pub const FIXTURE_SOURCE_UNCHANGED: &str = "moshi-source";\n')
        self.assertEqual(sum(row["members_checked"] for row in receipt["member_postchecks"]), 308)
        self.assertTrue(all(row["members_checked"] >= 5 for row in receipt["member_postchecks"]))
        self.assertTrue(candidate_has_source_none)
        self.assertEqual(metadata_receipts["baseline_cpu_metadata.json"]["metadata"]["gpu_features_active"], [])
        self.assertEqual(metadata_receipts["overlay_cpu_metadata.json"]["metadata"]["gpu_features_active"], [])
        self.assertEqual(metadata_receipts["baseline_cpu_metadata.json"]["target"], apply.TARGET)
        self.assertEqual(metadata_receipts["baseline_whole_metadata.json"]["target"], None)
        self.assertEqual(receipt["source_proof_pre"]["files_checked"], 23)
        self.assertEqual(receipt["source_proof_pre"]["unchanged_files"], 21)
        self.assertEqual(receipt["source_proof_post"]["files_checked"], 23)
        self.assertEqual(receipt["source_proof_post"]["unchanged_files"], 21)
        graph_proof = receipt["graph_validation_overlay"]
        self.assertEqual(graph_proof["mode"], "EXACT_16_OWNED_PATH_CANONICAL_VIEW")
        self.assertTrue(graph_proof["raw_preserved"])
        self.assertTrue(graph_proof["cpu"]["raw_preserved"])
        self.assertTrue(graph_proof["whole"]["raw_preserved"])

    def test_archive_failure_retains_bounded_blocked_receipt(self) -> None:
        calls = 0

        def failing_fetcher(_url, _size, _deadline):
            nonlocal calls
            calls += 1
            raise apply.OverlayError("synthetic authenticated archive fetch failure")

        result, error, receipt, _snapshots, _source, _fetches, _source_none, _metadata = self._run_apply(failing_fetcher)
        self.assertIsNone(result)
        self.assertIsInstance(error, apply.OverlayError)
        self.assertEqual(calls, 1)
        self.assertEqual(receipt["status"], "SOURCE_APPLICATION_BLOCKED")
        self.assertIn("error", receipt)
        self.assertNotIn("REVIEW_PENDING", receipt["status"])
        self.assertEqual(receipt["approval"], None)
        self.assertEqual(receipt["upload"], "NO_UPLOAD")
        self.assertEqual(receipt["model_execution"], "NOT_AUTHORIZED")

    def test_final_success_write_failure_recovers_blocked_receipt(self) -> None:
        result, error, receipt, _snapshots, _source, _fetches, _source_none, _metadata = self._run_apply(fail_final_write=True)
        self.assertIsNone(result)
        self.assertIsInstance(error, OSError)
        self.assertIn("synthetic final receipt write failure", str(error))
        self.assertEqual(receipt["status"], "SOURCE_APPLICATION_BLOCKED")
        self.assertIn("synthetic final receipt write failure", receipt["error"])
        self.assertNotIn("REVIEW_PENDING", receipt["status"])

    def test_late_operation_cutoff_recovers_blocked_receipt_with_outer_reserve(self) -> None:
        result, error, receipt, _snapshots, _source, _fetches, _source_none, metadata_receipts = self._run_apply(expire_after_final_write=True)
        self.assertIsNone(result)
        self.assertIsInstance(error, apply.OverlayError)
        self.assertIn("deadline", str(error).lower())
        self.assertTrue(self._last_final_write_completed)
        self.assertEqual(len(receipt["archives"]), 17)
        self.assertIn("overlay_whole_metadata.json", metadata_receipts)
        self.assertEqual(receipt["status"], "SOURCE_APPLICATION_BLOCKED")
        self.assertNotIn("REVIEW_PENDING", receipt["status"])

    def test_graph_validator_rejects_unrelated_feature_edge_and_declaration_drift(self) -> None:
        package_id = "synthetic:moshi:0.6.4:local"
        baseline = {
            "gpu_features_active": [],
            "packages": [{"id": package_id, "name": "moshi", "version": "0.6.4", "source": None,
                           "manifest_path": "/synthetic/moshi-0.6.4/Cargo.toml", "dependencies": [], "features": {}}],
            "workspace_members": [package_id],
            "resolve": {"root": package_id, "nodes": [{"id": package_id, "deps": [], "dependencies": [], "features": []}]},
        }
        apply._assert_graph_delta(baseline, copy.deepcopy(baseline), [], "whole", Path("/private/tmp"))
        feature_drift = copy.deepcopy(baseline)
        feature_drift["resolve"]["nodes"][0]["features"].append("unrelated-feature")
        with self.assertRaisesRegex(apply.OverlayError, "activated features"):
            apply._assert_graph_delta(baseline, feature_drift, [], "whole", Path("/private/tmp"))

        edge_drift = copy.deepcopy(baseline)
        edge_drift["resolve"]["nodes"][0]["dependencies"].append("synthetic:unknown:0.0.0:registry")
        with self.assertRaisesRegex(apply.OverlayError, "plain dependency IDs"):
            apply._assert_graph_delta(baseline, edge_drift, [], "whole", Path("/private/tmp"))

        declaration_drift = copy.deepcopy(baseline)
        declaration_drift["packages"][0]["dependencies"].append({
            "name": "unrelated", "source": None, "req": "*", "kind": None,
            "target": None, "optional": False, "features": [], "uses_default_features": True,
        })
        with self.assertRaisesRegex(apply.OverlayError, "dependency declaration"):
            apply._assert_graph_delta(baseline, declaration_drift, [], "whole", Path("/private/tmp"))

    def test_lock_validator_rejects_root_order_and_retained_record_drift(self) -> None:
        planner = self._planner()
        _parent, _pastey, candidate, plan = apply._load_inputs(planner)
        collector = load_real(apply.ROOT / "tools/parity/kyutai_candle_reference/collect_metadata.py", "lock-drift-test", apply.COLLECTOR_SHA256)
        with tempfile.TemporaryDirectory(prefix="paste-overlay-lock-") as directory:
            projection = Path(directory)
            (projection / "Cargo.lock").write_bytes(candidate)
            boundary = FakeCargoBoundary(collector, apply, plan)
            boundary.projection = projection
            before_doc = tomllib.loads(candidate.decode())
            boundary._write_overlay_lock()
            after_doc = tomllib.loads((projection / "Cargo.lock").read_text())
            before = collector.lock_packages(before_doc)
            after = collector.lock_packages(after_doc)
            valid_delta = apply._lock_delta(collector, before, after, before_doc, after_doc, plan["proposals"])
            self.assertEqual(valid_delta["status"], "COMPLETE_ORDERED_DIFF_REVIEW_PENDING")
            with self.assertRaisesRegex(apply.OverlayError, "Cargo.lock root fields"):
                drift = copy.deepcopy(after_doc)
                drift["version"] = int(drift["version"]) + 1
                apply._lock_delta(collector, before, collector.lock_packages(drift), before_doc, drift, plan["proposals"])
            with self.assertRaisesRegex(apply.OverlayError, "retained lock record"):
                drift = copy.deepcopy(after_doc)
                retained = next(row for row in drift["package"] if row["name"] == "moshi")
                retained["dependencies"] = list(reversed(retained.get("dependencies", [])))
                apply._lock_delta(collector, before, collector.lock_packages(drift), before_doc, drift, plan["proposals"])
            with self.assertRaisesRegex(apply.OverlayError, "retained lock record"):
                drift = copy.deepcopy(after_doc)
                retained = next(row for row in drift["package"] if row["name"] == "moshi")
                retained["checksum"] = "0" * 64
                apply._lock_delta(collector, before, collector.lock_packages(drift), before_doc, drift, plan["proposals"])

    def test_apply_rejects_overlapping_roots_before_materialization(self) -> None:
        with tempfile.TemporaryDirectory(prefix="paste-overlay-overlap-") as directory:
            root = Path(directory)
            packet = root / "equal"
            packet.mkdir()
            packet.joinpath("sentinel.txt").write_text("equal")
            cases = [(packet, packet, root / "equal-workspace")]

            packet = root / "packet-parent-output-child"
            packet.mkdir()
            packet.joinpath("sentinel.txt").write_text("packet-parent")
            cases.append((packet, packet / "output-child", root / "separate-workspace"))

            output = root / "output-parent-packet-child"
            output.mkdir()
            output.joinpath("sentinel.txt").write_text("output-parent")
            packet = output / "packet-child"
            packet.mkdir()
            cases.append((packet, output, root / "separate-workspace-2"))

            packet = root / "packet-parent-workspace-child"
            packet.mkdir()
            packet.joinpath("sentinel.txt").write_text("packet-parent-workspace")
            cases.append((packet, root / "separate-output-3", packet / "workspace-child"))

            workspace = root / "workspace-parent-packet-child"
            workspace.mkdir()
            workspace.joinpath("sentinel.txt").write_text("workspace-parent")
            packet = workspace / "packet-child"
            packet.mkdir()
            cases.append((packet, root / "separate-output-4", workspace))

            output = root / "output-parent-workspace-child"
            output.mkdir()
            output.joinpath("sentinel.txt").write_text("output-parent-workspace")
            packet = root / "separate-packet-5"
            packet.mkdir()
            cases.append((packet, output, output / "workspace-child"))

            workspace = root / "workspace-parent-output-child"
            workspace.mkdir()
            workspace.joinpath("sentinel.txt").write_text("workspace-parent-output")
            packet = root / "separate-packet-6"
            packet.mkdir()
            cases.append((packet, workspace / "output-child", workspace))

            def snapshot():
                return {
                    str(path.relative_to(root)): (
                        "directory" if path.is_dir() else
                        "file" if path.is_file() else
                        "other",
                        path.read_bytes() if path.is_file() else None,
                    )
                    for path in root.rglob("*")
                }

            with mock.patch.dict(os.environ, {"VOKRA_CANDLE_REMOTE_LINUX": "1"}, clear=False):
                for packet, output, workspace in cases:
                    before = snapshot()
                    with mock.patch.object(apply, "_module", side_effect=AssertionError("overlap check reached external module boundary")) as module_boundary, mock.patch.object(apply, "_load_inputs", side_effect=AssertionError("overlap check reached input boundary")) as input_boundary:
                        with self.assertRaisesRegex(apply.OverlayError, r"roots must|overlap"):
                            apply.apply_overlay(packet, output, workspace, platform_name="Linux")
                    module_boundary.assert_not_called()
                    input_boundary.assert_not_called()
                    self.assertEqual(snapshot(), before)

    def test_apply_retains_blocked_receipt_when_deadline_expires_at_capture_boundary(self) -> None:
        def deadline_fetcher(_url, _size, _deadline):
            raise apply.OverlayError("synthetic deadline expired at archive capture boundary")

        result, error, receipt, _snapshots, _source, _fetches, _source_none, _metadata = self._run_apply(
            deadline_fetcher,
            deadline_seconds=30.0,
        )
        self.assertIsNone(result)
        self.assertIsInstance(error, apply.OverlayError)
        self.assertIn("deadline", receipt["error"])
        self.assertEqual(receipt["status"], "SOURCE_APPLICATION_BLOCKED")
        self.assertNotIn("REVIEW_PENDING", receipt["status"])

    def test_shared_deadline_interrupts_real_blocking_source_io(self) -> None:
        collector = load_real(apply.ROOT / "tools/parity/kyutai_candle_reference/collect_metadata.py", "deadline-test", apply.COLLECTOR_SHA256)
        with self.assertRaises(collector.CaptureError):
            with collector._SourceIODeadline(time.monotonic() + 0.02):
                time.sleep(0.2)
