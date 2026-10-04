from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import contract
from contract import (
    COMPOSITE_APPROVAL_SCHEMA,
    COMPOSITE_APPROVAL_SCOPE,
    CONTEXT,
    EXECUTION_APPROVAL_MAX_BYTES,
    ExecutionReadinessBlocked,
    HF_REVISION,
    MAX_PACKET_BYTES,
    MAX_CONTEXT,
    require_execution_readiness,
    require_pcm_source_packet,
    SOURCE_ROLE,
    require_source_packet,
    boundary_packet,
    git_blob_sha1,
    git_identity,
    sha256,
    validate_composite_approval,
    validate_input_packet,
)


def _git_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update({
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_SYSTEM": os.devnull,
    })
    return env


def _git(root: Path, *args: str, capture_output: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-c", "core.hooksPath=/dev/null", "-C", str(root), *args],
        check=True,
        capture_output=capture_output,
        text=True,
        env=_git_env(),
    )


@contextmanager
def _synthetic_source_contract():
    names = (
        "MOSHI_REVISION",
        "DSM_REVISION",
        "MOSHI_TREE_SHA",
        "DSM_TREE_SHA",
        "MOSHI_COMMIT_PACKET",
        "DSM_COMMIT_PACKET",
        "MOSHI_ROLE_PACKET_FILES",
        "MOSHI_LM_SHA256",
        "MOSHI_TRANSFORMER_SHA256",
    )
    saved = {name: getattr(contract, name) for name in names}
    moshi_revision = "1" * 40
    dsm_revision = "2" * 40
    moshi_tree = "3" * 40
    dsm_tree = "4" * 40
    lm_body = b"synthetic official lm\n"
    transformer_body = b"synthetic official transformer\n"
    contract.MOSHI_REVISION = moshi_revision
    contract.DSM_REVISION = dsm_revision
    contract.MOSHI_TREE_SHA = moshi_tree
    contract.DSM_TREE_SHA = dsm_tree
    contract.MOSHI_LM_SHA256 = hashlib.sha256(lm_body).hexdigest()
    contract.MOSHI_TRANSFORMER_SHA256 = hashlib.sha256(transformer_body).hexdigest()
    contract.MOSHI_ROLE_PACKET_FILES = {
        "synthetic/lm.py": (
            "synthetic-lm.py",
            len(lm_body),
            contract.MOSHI_LM_SHA256,
            git_blob_sha1(lm_body),
        ),
        "synthetic/transformer.py": (
            "synthetic-transformer.py",
            len(transformer_body),
            contract.MOSHI_TRANSFORMER_SHA256,
            git_blob_sha1(transformer_body),
        ),
    }
    contract.MOSHI_COMMIT_PACKET = ("moshi-commit.json", moshi_revision, moshi_tree)
    contract.DSM_COMMIT_PACKET = ("dsm-commit.json", dsm_revision, dsm_tree)
    try:
        yield {
            "lm": lm_body,
            "transformer": transformer_body,
            "moshi_tree": moshi_tree,
            "dsm_tree": dsm_tree,
        }
    finally:
        for name, value in saved.items():
            setattr(contract, name, value)


def _write_synthetic_source_packet(packet: Path) -> None:
    packet.mkdir(parents=True, exist_ok=True)
    role_files = {
        packet_name: body
        for _source_path, (packet_name, _size, _digest, _blob) in contract.MOSHI_ROLE_PACKET_FILES.items()
        for body in (
            b"synthetic official lm\n"
            if packet_name == "synthetic-lm.py"
            else b"synthetic official transformer\n",
        )
    }
    files = {
        "moshi-commit.json": json.dumps(
            {"sha": contract.MOSHI_REVISION, "commit": {"tree": {"sha": contract.MOSHI_TREE_SHA}}},
            sort_keys=True,
            separators=(",", ":"),
        ).encode(),
        "dsm-commit.json": json.dumps(
            {"sha": contract.DSM_REVISION, "commit": {"tree": {"sha": contract.DSM_TREE_SHA}}},
            sort_keys=True,
            separators=(",", ":"),
        ).encode(),
    }
    files.update(role_files)
    rows = [
        {"path": "synthetic", "mode": "040000", "type": "tree", "sha": "5" * 40, "url": "https://example.invalid/tree"},
    ]
    for source_path, (packet_name, size, _digest, blob) in contract.MOSHI_ROLE_PACKET_FILES.items():
        rows.append({
            "path": source_path,
            "mode": "100644",
            "type": "blob",
            "sha": blob,
            "size": size,
            "url": "https://example.invalid/blob",
        })
    files["moshi-tree.json"] = json.dumps(
        {"sha": contract.MOSHI_TREE_SHA, "truncated": False, "tree": rows},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    for name, body in files.items():
        (packet / name).write_bytes(body)
    manifest = {
        "schema": "vokra-flat-source-receipt-v1",
        "status": "SOURCE_ONLY_PREPARATION",
        "files": [
            {"name": name, "role": SOURCE_ROLE, "size": len(body), "sha256": hashlib.sha256(body).hexdigest()}
            for name, body in sorted(files.items())
        ],
    }
    (packet / "manifest.json").write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")


def _rewrite_manifest(packet: Path, mutate) -> None:
    path = packet / "manifest.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    mutate(document)
    path.write_text(json.dumps(document, sort_keys=True) + "\n", encoding="utf-8")


def _replace_packet_file(packet: Path, name: str, body: bytes) -> None:
    (packet / name).write_bytes(body)

    def update(document):
        for entry in document["files"]:
            if entry["name"] == name:
                entry["size"] = len(body)
                entry["sha256"] = hashlib.sha256(body).hexdigest()
                break
        else:
            raise AssertionError(f"missing synthetic manifest entry: {name}")

    _rewrite_manifest(packet, update)


@contextmanager
def _synthetic_pcm_source_contract():
    names = (
        "MOSHI_REVISION",
        "DSM_REVISION",
        "MOSHI_TREE_SHA",
        "DSM_TREE_SHA",
        "MOSHI_COMMIT_PACKET",
        "DSM_COMMIT_PACKET",
        "DSM_PCM_ROLE_PACKET_FILES",
        "MOSHI_PCM_ROLE_PACKET_FILES",
        "PCM_SOURCE_PACKET_MANIFEST_SHA256",
    )
    saved = {name: getattr(contract, name) for name in names}
    moshi_revision = "1" * 40
    dsm_revision = "2" * 40
    moshi_tree = "3" * 40
    dsm_tree = "4" * 40
    role_bodies = {
        "dsm-evaluate.py": b"synthetic dsm evaluate\n",
        "dsm-server.py": b"synthetic dsm server\n",
        "moshi-compression.py": b"synthetic moshi compression\n",
        "moshi-streaming.py": b"synthetic moshi streaming\n",
    }

    def role(path: str, packet_name: str):
        body = role_bodies[packet_name]
        return (packet_name, len(body), hashlib.sha256(body).hexdigest(), git_blob_sha1(body))

    contract.MOSHI_REVISION = moshi_revision
    contract.DSM_REVISION = dsm_revision
    contract.MOSHI_TREE_SHA = moshi_tree
    contract.DSM_TREE_SHA = dsm_tree
    contract.MOSHI_COMMIT_PACKET = ("moshi-commit.json", moshi_revision, moshi_tree)
    contract.DSM_COMMIT_PACKET = ("dsm-commit.json", dsm_revision, dsm_tree)
    contract.DSM_PCM_ROLE_PACKET_FILES = {
        "scripts/evaluate.py": role("scripts/evaluate.py", "dsm-evaluate.py"),
        "scripts/server.py": role("scripts/server.py", "dsm-server.py"),
    }
    contract.MOSHI_PCM_ROLE_PACKET_FILES = {
        "moshi/compression.py": role("moshi/compression.py", "moshi-compression.py"),
        "moshi/streaming.py": role("moshi/streaming.py", "moshi-streaming.py"),
    }
    try:
        yield role_bodies
    finally:
        for name, value in saved.items():
            setattr(contract, name, value)


def _write_synthetic_pcm_source_packet(packet: Path, role_bodies: dict[str, bytes]) -> None:
    packet.mkdir(parents=True, exist_ok=True)
    files: dict[str, bytes] = {
        "moshi-commit.json": json.dumps(
            {"sha": contract.MOSHI_REVISION, "commit": {"tree": {"sha": contract.MOSHI_TREE_SHA}},
             "source": "synthetic"},
            sort_keys=True,
            separators=(",", ":"),
        ).encode(),
        "dsm-commit.json": json.dumps(
            {"sha": contract.DSM_REVISION, "commit": {"tree": {"sha": contract.DSM_TREE_SHA}},
             "source": "synthetic"},
            sort_keys=True,
            separators=(",", ":"),
        ).encode(),
    }
    files.update(role_bodies)
    for tree_name, tree_digest, role_files in (
        ("dsm-tree.json", contract.DSM_TREE_SHA, contract.DSM_PCM_ROLE_PACKET_FILES),
        ("moshi-tree.json", contract.MOSHI_TREE_SHA, contract.MOSHI_PCM_ROLE_PACKET_FILES),
    ):
        rows = []
        for source_path, (packet_name, size, _digest, blob) in role_files.items():
            rows.append({
                "path": source_path,
                "mode": "100644",
                "type": "blob",
                "sha": blob,
                "size": size,
                "url": "https://example.invalid/blob",
            })
        files[tree_name] = json.dumps(
            {"sha": tree_digest, "truncated": False, "tree": rows},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    for name, body in files.items():
        (packet / name).write_bytes(body)
    manifest = {
        "schema": "vokra-flat-source-receipt-v1",
        "status": "SOURCE_ONLY_PREPARATION",
        "source_revision": "authenticated",
        "total_bytes": sum(len(body) for body in files.values()),
        "files": [
            {"name": name, "role": SOURCE_ROLE, "size": len(body), "sha256": hashlib.sha256(body).hexdigest()}
            for name, body in sorted(files.items())
        ],
    }
    manifest_body = (json.dumps(manifest, sort_keys=True) + "\n").encode()
    (packet / "manifest.json").write_bytes(manifest_body)
    contract.PCM_SOURCE_PACKET_MANIFEST_SHA256 = hashlib.sha256(manifest_body).hexdigest()


def _sync_pcm_manifest_digest(packet: Path) -> None:
    path = packet / "manifest.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document["total_bytes"] = sum(entry["size"] for entry in document["files"])
    path.write_text(json.dumps(document, sort_keys=True) + "\n", encoding="utf-8")
    contract.PCM_SOURCE_PACKET_MANIFEST_SHA256 = sha256(path)


class ContractTests(unittest.TestCase):
    def test_synthetic_authenticated_packet_uses_manifest_and_tree_bindings(self) -> None:
        with _synthetic_source_contract():
            with tempfile.TemporaryDirectory(prefix="kyutai-source-contract-") as raw:
                packet = Path(raw) / "packet"
                _write_synthetic_source_packet(packet)
                receipt = require_source_packet(packet)
                self.assertEqual(receipt["status"], "SOURCE_ONLY_PREPARATION")
                self.assertEqual(receipt["moshi_lm_sha256"], contract.MOSHI_LM_SHA256)

    def test_source_packet_rejects_missing_consumed_manifest_entries(self) -> None:
        with _synthetic_source_contract():
            for missing_name in ("moshi-commit.json", "dsm-commit.json", "moshi-tree.json", "synthetic-lm.py"):
                with self.subTest(missing_name=missing_name):
                    with tempfile.TemporaryDirectory(prefix="kyutai-source-missing-") as raw:
                        packet = Path(raw) / "packet"
                        _write_synthetic_source_packet(packet)

                        def remove_entry(document):
                            document["files"] = [entry for entry in document["files"] if entry["name"] != missing_name]

                        _rewrite_manifest(packet, remove_entry)
                        with self.assertRaises(ValueError):
                            require_source_packet(packet)

    def test_source_packet_rejects_duplicate_tree_paths(self) -> None:
        with _synthetic_source_contract():
            with tempfile.TemporaryDirectory(prefix="kyutai-source-duplicate-tree-") as raw:
                packet = Path(raw) / "packet"
                _write_synthetic_source_packet(packet)
                tree = json.loads((packet / "moshi-tree.json").read_text(encoding="utf-8"))
                tree["tree"].append(dict(tree["tree"][-1]))
                _replace_packet_file(
                    packet,
                    "moshi-tree.json",
                    json.dumps(tree, sort_keys=True, separators=(",", ":")).encode(),
                )
                with self.assertRaises(ValueError):
                    require_source_packet(packet)

    def test_source_packet_rejects_malformed_rows_and_missing_manifest_role(self) -> None:
        with _synthetic_source_contract():
            with tempfile.TemporaryDirectory(prefix="kyutai-source-malformed-") as raw:
                packet = Path(raw) / "packet"
                _write_synthetic_source_packet(packet)
                tree = json.loads((packet / "moshi-tree.json").read_text(encoding="utf-8"))
                del tree["tree"][1]["sha"]
                _replace_packet_file(
                    packet,
                    "moshi-tree.json",
                    json.dumps(tree, sort_keys=True, separators=(",", ":")).encode(),
                )
                with self.assertRaises(ValueError):
                    require_source_packet(packet)

                _write_synthetic_source_packet(packet)

                def add_missing_role(document):
                    document["files"].append({
                        "name": "unconsumed-extra.txt",
                        "size": 0,
                        "sha256": hashlib.sha256(b"").hexdigest(),
                    })

                _rewrite_manifest(packet, add_missing_role)
                with self.assertRaises(ValueError):
                    require_source_packet(packet)

    def test_source_packet_rejects_oversized_and_duplicate_json(self) -> None:
        with _synthetic_source_contract():
            with tempfile.TemporaryDirectory(prefix="kyutai-source-bounds-") as raw:
                packet = Path(raw) / "packet"
                _write_synthetic_source_packet(packet)

                def oversize_entry(document):
                    for entry in document["files"]:
                        if entry["name"] == "moshi-tree.json":
                            entry["size"] = MAX_PACKET_BYTES + 1
                            return
                    raise AssertionError("missing tree entry")

                _rewrite_manifest(packet, oversize_entry)
                with self.assertRaises(ValueError):
                    require_source_packet(packet)

                _write_synthetic_source_packet(packet)
                duplicate_json = (
                    b'{"sha":"' + contract.MOSHI_TREE_SHA.encode() +
                    b'","sha":"' + ("6" * 40).encode() +
                    b'","truncated":false,"tree":[]}'
                )
                _replace_packet_file(packet, "moshi-tree.json", duplicate_json)
                with self.assertRaises(ValueError):
                    require_source_packet(packet)

                _write_synthetic_source_packet(packet)

                def duplicate_manifest_entry(document):
                    document["files"].append(dict(document["files"][0]))

                _rewrite_manifest(packet, duplicate_manifest_entry)
                with self.assertRaises(ValueError):
                    require_source_packet(packet)

    def test_pcm_source_packet_authenticates_all_roles_and_trees(self) -> None:
        with _synthetic_pcm_source_contract() as role_bodies:
            with tempfile.TemporaryDirectory(prefix="kyutai-pcm-source-contract-") as raw:
                packet = Path(raw) / "packet"
                _write_synthetic_pcm_source_packet(packet, role_bodies)
                receipt = require_pcm_source_packet(packet)
                self.assertEqual(receipt["status"], "SOURCE_ONLY_PREPARATION")
                self.assertEqual(receipt["dsm_tree_sha"], contract.DSM_TREE_SHA)
                self.assertEqual(receipt["moshi_tree_sha"], contract.MOSHI_TREE_SHA)

    def test_pcm_source_packet_rejects_missing_consumed_manifest_entry(self) -> None:
        with _synthetic_pcm_source_contract() as role_bodies:
            with tempfile.TemporaryDirectory(prefix="kyutai-pcm-source-missing-") as raw:
                packet = Path(raw) / "packet"
                _write_synthetic_pcm_source_packet(packet, role_bodies)

                def remove_entry(document):
                    document["files"] = [
                        entry for entry in document["files"] if entry["name"] != "dsm-tree.json"
                    ]

                _rewrite_manifest(packet, remove_entry)
                _sync_pcm_manifest_digest(packet)
                with self.assertRaisesRegex(ValueError, "missing consumed files"):
                    require_pcm_source_packet(packet)

    def test_pcm_source_packet_rejects_duplicate_tree_paths_and_malformed_rows(self) -> None:
        with _synthetic_pcm_source_contract() as role_bodies:
            with tempfile.TemporaryDirectory(prefix="kyutai-pcm-source-tree-") as raw:
                packet = Path(raw) / "packet"
                _write_synthetic_pcm_source_packet(packet, role_bodies)
                tree = json.loads((packet / "dsm-tree.json").read_text(encoding="utf-8"))
                tree["tree"].append(dict(tree["tree"][-1]))
                _replace_packet_file(
                    packet,
                    "dsm-tree.json",
                    json.dumps(tree, sort_keys=True, separators=(",", ":")).encode(),
                )
                _sync_pcm_manifest_digest(packet)
                with self.assertRaisesRegex(ValueError, "not unique"):
                    require_pcm_source_packet(packet)

                _write_synthetic_pcm_source_packet(packet, role_bodies)
                tree = json.loads((packet / "moshi-tree.json").read_text(encoding="utf-8"))
                del tree["tree"][0]["sha"]
                _replace_packet_file(
                    packet,
                    "moshi-tree.json",
                    json.dumps(tree, sort_keys=True, separators=(",", ":")).encode(),
                )
                _sync_pcm_manifest_digest(packet)
                with self.assertRaisesRegex(ValueError, "row keys"):
                    require_pcm_source_packet(packet)

    def test_pcm_source_packet_rejects_oversized_and_wrong_digest_entries(self) -> None:
        with _synthetic_pcm_source_contract() as role_bodies:
            with tempfile.TemporaryDirectory(prefix="kyutai-pcm-source-bounds-") as raw:
                packet = Path(raw) / "packet"
                _write_synthetic_pcm_source_packet(packet, role_bodies)

                def oversize_entry(document):
                    for entry in document["files"]:
                        if entry["name"] == "moshi-tree.json":
                            entry["size"] = MAX_PACKET_BYTES + 1
                            return
                    raise AssertionError("missing tree entry")

                _rewrite_manifest(packet, oversize_entry)
                _sync_pcm_manifest_digest(packet)
                with self.assertRaisesRegex(ValueError, "size is invalid"):
                    require_pcm_source_packet(packet)

                _write_synthetic_pcm_source_packet(packet, role_bodies)

                def wrong_digest(document):
                    for entry in document["files"]:
                        if entry["name"] == "moshi-streaming.py":
                            entry["sha256"] = "0" * 64
                            return
                    raise AssertionError("missing streaming role")

                _rewrite_manifest(packet, wrong_digest)
                _sync_pcm_manifest_digest(packet)
                with self.assertRaisesRegex(ValueError, "manifest digest mismatch"):
                    require_pcm_source_packet(packet)

    def test_real_path_and_git_identity_reject_unsafe_or_dirty_inputs(self) -> None:
        with tempfile.TemporaryDirectory(prefix="kyutai-contract-") as raw:
            root = Path(raw) / "checkout"
            root.mkdir()
            _git(root, "init", "-q")
            _git(root, "config", "user.email", "test@example.invalid")
            _git(root, "config", "user.name", "test")
            (root / "README").write_text("clean\n", encoding="utf-8")
            _git(root, "add", "README")
            _git(root, "commit", "--no-verify", "-qm", "init")
            revision = _git(root, "rev-parse", "HEAD", capture_output=True).stdout.strip()
            _git(root, "remote", "add", "origin", "https://github.com/kyutai-labs/moshi.git")
            self.assertEqual(git_identity(root, "https://github.com/kyutai-labs/moshi.git", revision)["revision"], revision)
            (root / "dirty").write_text("x", encoding="utf-8")
            with self.assertRaises(ValueError):
                git_identity(root, "https://github.com/kyutai-labs/moshi.git", revision)
            outside = Path(raw) / "outside"
            outside.mkdir()
            link = Path(raw) / "link"
            link.symlink_to(outside, target_is_directory=True)
            with self.assertRaises(ValueError):
                # The explicit path contains a symlinked ancestor and is not a checkout.
                git_identity(link, "https://github.com/kyutai-labs/moshi.git", revision)

    def test_input_bounds_reject_bool_and_excess_context(self) -> None:
        packet = boundary_packet()
        validate_input_packet(packet)
        with self.assertRaises(ValueError):
            validate_input_packet(dict(packet, context=True))
        with self.assertRaises(ValueError):
            boundary_packet(MAX_CONTEXT + 1)
        with self.assertRaises(ValueError):
            validate_input_packet(dict(packet, frames=True))

    def test_composite_approval_is_distinct_and_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory(prefix="kyutai-approval-") as raw:
            root = Path(raw)
            checkout = root / "checkout"
            checkout.mkdir()
            approval = root / "approval.json"
            approval.write_text(json.dumps({
                "schema": COMPOSITE_APPROVAL_SCHEMA,
                "scope": COMPOSITE_APPROVAL_SCOPE,
                "decision": "APPROVED",
                "execution": "VAST_ONLY",
                "head": "a" * 40,
                "checkout": str(checkout),
            }) + "\n", encoding="utf-8")
            self.assertEqual(
                validate_composite_approval(approval, "a" * 40, sha256(approval), checkout)["scope"],
                COMPOSITE_APPROVAL_SCOPE,
            )
            approval.write_text(approval.read_text(encoding="utf-8").replace(COMPOSITE_APPROVAL_SCOPE, "KYUTAI_STT_DECODER_PARITY"), encoding="utf-8")
            with self.assertRaises(ValueError):
                validate_composite_approval(approval, "a" * 40, sha256(approval), checkout)

    def test_execution_readiness_blocks_unknown_closure_even_with_approved_receipt(self) -> None:
        with tempfile.TemporaryDirectory(prefix="kyutai-readiness-") as raw:
            root = Path(raw)
            checkout = root / "checkout"
            checkout.mkdir()
            approval = root / "approval.json"
            approval.write_text(json.dumps({
                "schema": COMPOSITE_APPROVAL_SCHEMA,
                "scope": COMPOSITE_APPROVAL_SCOPE,
                "decision": "APPROVED",
                "execution": "VAST_ONLY",
                "head": "a" * 40,
                "checkout": str(checkout),
            }) + "\n", encoding="utf-8")
            with self.assertRaises(ExecutionReadinessBlocked) as raised:
                require_execution_readiness(
                    approval,
                    expected_head="a" * 40,
                    expected_approval_sha256=sha256(approval),
                    checkout=checkout,
                    dependency_closure_sha256="b" * 64,
                    platform_system="Linux",
                    platform_machine="x86_64",
                )
            self.assertEqual(raised.exception.status, "BLOCKED_DEPENDENCY_CLOSURE")

    def test_execution_readiness_rejects_non_linux_before_path_access(self) -> None:
        # The maintainer-platform guard runs before checkout/approval access;
        # this deliberately supplies paths that do not exist.
        with self.assertRaises(ExecutionReadinessBlocked) as raised:
            require_execution_readiness(
                Path("/definitely-missing/approval.json"),
                expected_head="a" * 40,
                expected_approval_sha256="b" * 64,
                checkout=Path("/definitely-missing/checkout"),
                dependency_closure_sha256="c" * 64,
                platform_system="Darwin",
                platform_machine="arm64",
            )
        self.assertIn("Linux x86_64", str(raised.exception))

    def test_execution_readiness_rejects_inside_checkout_and_duplicate_json(self) -> None:
        with tempfile.TemporaryDirectory(prefix="kyutai-readiness-") as raw:
            root = Path(raw)
            checkout = root / "checkout"
            checkout.mkdir()
            approval = checkout / "approval.json"
            body = json.dumps({
                "schema": COMPOSITE_APPROVAL_SCHEMA,
                "scope": COMPOSITE_APPROVAL_SCOPE,
                "decision": "APPROVED",
                "execution": "VAST_ONLY",
                "head": "a" * 40,
                "checkout": str(checkout),
            }) + "\n"
            approval.write_text(body, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "outside checkout"):
                require_execution_readiness(
                    approval,
                    expected_head="a" * 40,
                    expected_approval_sha256=sha256(approval),
                    checkout=checkout,
                    dependency_closure_sha256="b" * 64,
                    platform_system="Linux",
                    platform_machine="x86_64",
                )

            approval = root / "duplicate.json"
            approval.write_bytes(
                (b'{"schema":"' + COMPOSITE_APPROVAL_SCHEMA.encode() +
                 b'","scope":"' + COMPOSITE_APPROVAL_SCOPE.encode() +
                 b'","scope":"duplicate","decision":"APPROVED",'
                 b'"execution":"VAST_ONLY","head":"' + b"a" * 40 +
                 b'","checkout":"' + str(checkout).encode() + b'"}'))
            with self.assertRaisesRegex(ValueError, "unique-key JSON"):
                require_execution_readiness(
                    approval,
                    expected_head="a" * 40,
                    expected_approval_sha256=sha256(approval),
                    checkout=checkout,
                    dependency_closure_sha256="b" * 64,
                    platform_system="Linux",
                    platform_machine="x86_64",
                )

    def test_execution_readiness_bounds_external_receipt_and_hex_fields(self) -> None:
        with tempfile.TemporaryDirectory(prefix="kyutai-readiness-") as raw:
            root = Path(raw)
            checkout = root / "checkout"
            checkout.mkdir()
            approval = root / "oversized.json"
            approval.write_bytes(b"{" + b"x" * EXECUTION_APPROVAL_MAX_BYTES + b"}")
            with self.assertRaises(ValueError):
                require_execution_readiness(
                    approval,
                    expected_head="a" * 40,
                    expected_approval_sha256=sha256(approval),
                    checkout=checkout,
                    dependency_closure_sha256="b" * 64,
                    platform_system="Linux",
                    platform_machine="x86_64",
                )
            valid = root / "valid.json"
            valid.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "lowercase hexadecimal"):
                require_execution_readiness(
                    valid,
                    expected_head="A" * 40,
                    expected_approval_sha256="b" * 64,
                    checkout=checkout,
                    dependency_closure_sha256="c" * 64,
                    platform_system="Linux",
                    platform_machine="x86_64",
                )


if __name__ == "__main__":
    unittest.main()
