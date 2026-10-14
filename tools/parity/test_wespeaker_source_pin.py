#!/usr/bin/env python3
"""Stdlib-only regression tests for the authenticated WeSpeaker source pin."""

from __future__ import annotations

import hashlib
import copy
import importlib.util
import json
import os
import re
import shutil
import sys
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock


HERE = Path(__file__).resolve().parent


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


DUMPER = load_module("wespeaker_dump_reference", HERE / "wespeaker_dump_reference.py")
GATE = load_module("wespeaker_preflight_gate", HERE / "wespeaker" / "preflight_gate.py")
FIXTURE = HERE / "wespeaker" / "source_fixture"
WORKER = HERE.parent.parent / "scripts" / "publish" / "vast-ai" / "run-wespeaker-validation.sh"


class WeSpeakerSourcePinTests(unittest.TestCase):
    def _git_env(self, source: Path) -> dict[str, str]:
        template = source.parent / "git-template"
        template.mkdir(exist_ok=True)
        env = os.environ.copy()
        env.update(
            GIT_CONFIG_NOSYSTEM="1",
            GIT_CONFIG_GLOBAL=os.devnull,
            GIT_CONFIG_SYSTEM=os.devnull,
            GIT_TEMPLATE_DIR=str(template),
        )
        for key in (
            "GIT_DIR",
            "GIT_WORK_TREE",
            "GIT_COMMON_DIR",
            "GIT_INDEX_FILE",
            "GIT_OBJECT_DIRECTORY",
            "GIT_ALTERNATE_OBJECT_DIRECTORIES",
            "GIT_CONFIG",
            "GIT_CONFIG_PARAMETERS",
        ):
            env.pop(key, None)
        for key in tuple(env):
            if key == "GIT_CONFIG_COUNT" or re.fullmatch(r"GIT_CONFIG_(?:KEY|VALUE)_\d+", key):
                env.pop(key, None)
        return env

    def _git(self, source: Path, *args: str) -> None:
        subprocess.run(["git", "-C", str(source), *args], check=True, env=self._git_env(source))

    def _fixture_checkout(self, root: Path) -> Path:
        source = root / "source"
        shutil.copytree(FIXTURE, source)
        env = self._git_env(source)
        subprocess.run(["git", "-C", str(source), "-c", f"init.templateDir={env['GIT_TEMPLATE_DIR']}", "init", "-q"], check=True, env=env)
        self._git(source, "config", "core.hooksPath", os.devnull)
        self._git(source, "config", "core.autocrlf", "false")
        self._git(source, "config", "user.email", "fixture@example.invalid")
        self._git(source, "config", "user.name", "fixture")
        self._git(source, "add", "LICENSE", "wespeaker/models/resnet.py", "wespeaker/models/pooling_layers.py")
        self._git(source, "-c", "commit.gpgsign=false", "commit", "-qm", "fixture")
        return source

    def _approved_bundle(self, root: Path, *, source_identity: dict | None = None) -> tuple[Path, Path, dict]:
        root.mkdir(parents=True, exist_ok=True)
        manifest = json.loads((HERE / "wespeaker" / "license_gate_manifest.json").read_text(encoding="utf-8"))
        fixed = {"source": copy.deepcopy(GATE.SOURCE_IDENTITY), **copy.deepcopy(GATE.FIXED_IDENTITIES)}
        if source_identity is not None:
            fixed["source"] = copy.deepcopy(source_identity)
        manifest["fixed_identities"] = fixed
        for row in manifest["dependency_reviews"]:
            row.update(status="REVIEWED", license="SELF_TEST", native_review="SELF_TEST", bundled_review="SELF_TEST", payload_sha256="a" * 64)
        for row, expected in zip(manifest["component_reviews"], GATE.expected_components(), strict=True):
            identity = copy.deepcopy(expected["identity"])
            if row["id"] == "source":
                identity = copy.deepcopy(fixed["source"])
            row.update(identity=identity, status="REVIEWED", license=expected["license"], payload_sha256="b" * 64, signer="self-test")
            row["approval_digest"] = GATE.canonical({key: row[key] for key in ("id", "identity", "license", "status", "payload_sha256")})
        manifest["dependency_reviews_sha256"] = GATE.canonical(manifest["dependency_reviews"])
        scope = {"lock_sha256": manifest["lock_sha256"], "pyproject_sha256": manifest["pyproject_sha256"], "package_rows_sha256": manifest["package_rows_sha256"], "dependency_reviews": manifest["dependency_reviews"], "component_reviews": manifest["component_reviews"], "fixed_identities": manifest["fixed_identities"], "no_upload": "NO_UPLOAD"}
        manifest["approval_scope_sha256"] = GATE.canonical(scope)
        manifest["operator_approval"] = {"decision": "APPROVED", "signer": "self-test", "digest": manifest["approval_scope_sha256"]}
        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        evidence = {"decision": "APPROVED", "digest": manifest["approval_scope_sha256"], "manifest_sha256": GATE.digest(manifest_path.read_bytes()), "scope_sha256": manifest["approval_scope_sha256"], "signer": "self-test"}
        evidence["evidence_sha256"] = GATE.canonical(evidence)
        evidence_path = root / "evidence.json"
        evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
        return manifest_path, evidence_path, manifest

    def _verify_fixture(self, source: Path) -> Path:
        real_run = DUMPER.subprocess.run
        env = self._git_env(source)

        def run(argv, *args, **kwargs):
            if "rev-parse" in argv:
                return mock.Mock(stdout=f"{DUMPER.SOURCE_REVISION}\n")
            kwargs["env"] = env
            return real_run(argv, *args, **kwargs)

        with mock.patch.object(DUMPER.subprocess, "run", side_effect=run):
            return DUMPER.verify_source_tree(source)

    def test_official_pooling_identity_is_exact(self) -> None:
        expected = (
            10_255,
            "768910c91e353f1d0c2f3edc8f9ae758d2ff7e477d37fb470ae946eba69adac8",
            "47120eb9029ba1b00d18fcd61dd80f8afa2037c6",
        )
        self.assertEqual(DUMPER.SOURCE_FILES["wespeaker/models/pooling_layers.py"] + (DUMPER.SOURCE_BLOBS["wespeaker/models/pooling_layers.py"],), expected)
        row = next(row for row in GATE.SOURCE_IDENTITY["files"] if row["path"] == "wespeaker/models/pooling_layers.py")
        self.assertEqual((row["bytes"], row["sha256"], row["git_blob_sha1"]), expected)
        self.assertEqual(GATE.SOURCE_IDENTITY["license"]["git_blob_sha1"], "261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64")

    def test_fixture_bytes_and_git_blobs_are_exact(self) -> None:
        expected = {
            "LICENSE": (11_357, "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4", "261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64"),
            "wespeaker/models/resnet.py": (9_564, "6f3c8219be2c9a8b9eabed8169c1abaec3e48670be7aaf1e792138b2b20e68c4", "17607e6d2c72627e15db4214cacfa9d7b89ca945"),
            "wespeaker/models/pooling_layers.py": (10_255, "768910c91e353f1d0c2f3edc8f9ae758d2ff7e477d37fb470ae946eba69adac8", "47120eb9029ba1b00d18fcd61dd80f8afa2037c6"),
        }
        for relative, (size, sha256, blob) in expected.items():
            with self.subTest(relative=relative):
                payload = (FIXTURE / relative).read_bytes()
                self.assertEqual(len(payload), size)
                self.assertEqual(hashlib.sha256(payload).hexdigest(), sha256)
                self.assertEqual(hashlib.sha1(b"blob " + str(len(payload)).encode() + b"\0" + payload).hexdigest(), blob)

    def test_manifest_binds_corrected_identity_and_scope(self) -> None:
        manifest_path = HERE / "wespeaker" / "license_gate_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        pooling = next(row for row in manifest["fixed_identities"]["source"]["files"] if row["path"].endswith("pooling_layers.py"))
        self.assertEqual(pooling["git_blob_sha1"], DUMPER.SOURCE_BLOBS["wespeaker/models/pooling_layers.py"])
        self.assertEqual(pooling["sha256"], DUMPER.SOURCE_FILES["wespeaker/models/pooling_layers.py"][1])
        scope = {
            "lock_sha256": manifest["lock_sha256"],
            "pyproject_sha256": manifest["pyproject_sha256"],
            "package_rows_sha256": manifest["package_rows_sha256"],
            "dependency_reviews": manifest["dependency_reviews"],
            "component_reviews": manifest["component_reviews"],
            "fixed_identities": manifest["fixed_identities"],
            "no_upload": "NO_UPLOAD",
        }
        actual = GATE.canonical(scope)
        self.assertEqual(manifest["approval_scope_sha256"], actual)
        self.assertEqual(manifest["operator_approval"], {"decision": "PENDING_REVIEW", "digest": None, "signer": None})

    def test_checkpoint_remote_and_staged_identity_are_distinct(self) -> None:
        manifest = json.loads((HERE / "wespeaker" / "license_gate_manifest.json").read_text(encoding="utf-8"))
        checkpoint = manifest["fixed_identities"]["checkpoint"]
        self.assertEqual(checkpoint["remote_filename"], "avg_model")
        self.assertEqual(checkpoint["staged_filename"], "avg_model.pt")
        self.assertEqual(checkpoint["path"], checkpoint["staged_filename"])
        self.assertIsNone(checkpoint["git_oid"])
        self.assertEqual(checkpoint["git_oid_status"], "NOT_AUTHENTICATED")

    def test_old_identity_cannot_reuse_new_scope(self) -> None:
        manifest_path = HERE / "wespeaker" / "license_gate_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory(prefix="wespeaker-source-pin-") as temp:
            root = Path(temp)
            project = root / "project"
            project.mkdir()
            shutil.copy2(HERE / "wespeaker" / "uv.lock", project / "uv.lock")
            shutil.copy2(HERE / "wespeaker" / "pyproject.toml", project / "pyproject.toml")
            current = root / "current.json"
            current.write_text(json.dumps(manifest), encoding="utf-8")
            current_ok, current_reason = GATE.validate(project, current, _self_test=True)
            self.assertFalse(current_ok)
            self.assertEqual(current_reason, "dependency review is unresolved: filelock@3.32.4")
        old = next(row for row in manifest["fixed_identities"]["source"]["files"] if row["path"].endswith("pooling_layers.py"))
        old.update(
            git_blob_sha1="47120eead47a511939267470496539804c17b7d3",
            sha256="768910f8e88cb47e742274563339d7e780cb9d56c629c4d4124605296686f0f9",
        )
        old_component = next(row for row in manifest["component_reviews"] if row["id"] == "source")
        old_component["identity"] = manifest["fixed_identities"]["source"]
        old_scope = {
            "lock_sha256": manifest["lock_sha256"],
            "pyproject_sha256": manifest["pyproject_sha256"],
            "package_rows_sha256": manifest["package_rows_sha256"],
            "dependency_reviews": manifest["dependency_reviews"],
            "component_reviews": manifest["component_reviews"],
            "fixed_identities": manifest["fixed_identities"],
            "no_upload": "NO_UPLOAD",
        }
        manifest["approval_scope_sha256"] = GATE.canonical(old_scope)
        with tempfile.TemporaryDirectory(prefix="wespeaker-source-pin-") as temp:
            root = Path(temp)
            project = root / "project"
            project.mkdir()
            shutil.copy2(HERE / "wespeaker" / "uv.lock", project / "uv.lock")
            shutil.copy2(HERE / "wespeaker" / "pyproject.toml", project / "pyproject.toml")
            candidate = root / "manifest.json"
            candidate.write_text(json.dumps(manifest), encoding="utf-8")
            ok, reason = GATE.validate(project, candidate, _self_test=True)
        self.assertFalse(ok)
        self.assertEqual(reason, "approval scope is not bound to exact closure")

    def test_source_mismatch_is_rejected_before_runtime_loader(self) -> None:
        with tempfile.TemporaryDirectory(prefix="wespeaker-source-pin-") as temp:
            temp_root = Path(temp).resolve()
            source = temp_root / "source"
            (source / "wespeaker" / "models").mkdir(parents=True)
            (source / "wespeaker" / "models" / "resnet.py").write_text("wrong", encoding="utf-8")
            def fake_git(argv, **_kwargs):
                if "status" in argv:
                    return mock.Mock(stdout="")
                if "rev-parse" in argv:
                    return mock.Mock(stdout=f"{DUMPER.SOURCE_REVISION}\n")
                return mock.Mock(stdout="")

            with mock.patch.object(DUMPER, "_load_runtime", side_effect=AssertionError("runtime loaded")), mock.patch.object(DUMPER.subprocess, "run", side_effect=fake_git):
                with mock.patch.object(sys, "argv", [
                    "wespeaker_dump_reference.py",
                    "--output-dir", str(temp_root / "output"),
                    "--checkpoint", str(temp_root / "avg_model.pt"),
                    "--wespeaker-source", str(source),
                ]):
                    with self.assertRaisesRegex(SystemExit, "not a WeSpeaker source tree|source identity mismatch"):
                        DUMPER.main()

    def test_verified_official_fixture_passes_without_model_import(self) -> None:
        with tempfile.TemporaryDirectory(prefix="wespeaker-source-fixture-") as temp:
            source = self._fixture_checkout(Path(temp).resolve())
            self.assertEqual(self._verify_fixture(source), source.resolve())

    def test_git_fixture_env_cannot_redirect_repository_or_config(self) -> None:
        with tempfile.TemporaryDirectory(prefix="wespeaker-source-fixture-") as temp:
            root = Path(temp).resolve()
            outside = root / "outside"
            outside.mkdir()
            injected = {
                "GIT_DIR": str(outside / "redirected.git"),
                "GIT_WORK_TREE": str(outside / "work-tree"),
                "GIT_COMMON_DIR": str(outside / "common"),
                "GIT_INDEX_FILE": str(outside / "index"),
                "GIT_OBJECT_DIRECTORY": str(outside / "objects"),
                "GIT_ALTERNATE_OBJECT_DIRECTORIES": str(outside / "alternates"),
                "GIT_CONFIG": str(outside / "config"),
                "GIT_CONFIG_PARAMETERS": "'core.hooksPath=/tmp/unsafe'",
                "GIT_CONFIG_COUNT": "1",
                "GIT_CONFIG_KEY_0": "core.hooksPath",
                "GIT_CONFIG_VALUE_0": "/tmp/unsafe",
            }
            with mock.patch.dict(os.environ, injected, clear=False):
                sanitized = self._git_env(root / "source")
                for key in injected:
                    self.assertNotIn(key, sanitized)
                source = self._fixture_checkout(root)
                self.assertEqual(self._verify_fixture(source), source.resolve())
            self.assertTrue((source / ".git").is_dir())
            self.assertFalse((outside / "redirected.git").exists())
            self.assertFalse((outside / "config").exists())
            self.assertFalse((outside / "index").exists())

    def test_one_byte_fixture_drift_fails_source_identity(self) -> None:
        with tempfile.TemporaryDirectory(prefix="wespeaker-source-fixture-") as temp:
            source = self._fixture_checkout(Path(temp).resolve())
            path = source / "wespeaker" / "models" / "pooling_layers.py"
            original = path.read_bytes()
            path.write_bytes(original[:-1] + bytes((original[-1] ^ 1,)))
            self._git(source, "add", "wespeaker/models/pooling_layers.py")
            self._git(source, "-c", "commit.gpgsign=false", "commit", "-qm", "drift")
            with self.assertRaisesRegex(SystemExit, "source identity mismatch|source Git blob mismatch"):
                self._verify_fixture(source)

    def test_synthetic_new_scope_and_evidence_pass_gate(self) -> None:
        with tempfile.TemporaryDirectory(prefix="wespeaker-source-approval-") as temp:
            root = Path(temp).resolve()
            project = root / "project"
            project.mkdir()
            shutil.copy2(HERE / "wespeaker" / "uv.lock", project / "uv.lock")
            shutil.copy2(HERE / "wespeaker" / "pyproject.toml", project / "pyproject.toml")
            manifest, evidence, _ = self._approved_bundle(root)
            self.assertEqual(GATE.validate(project, manifest, evidence, _self_test=True), (True, "PASS"))

    def test_old_scope_rejected_and_old_evidence_cannot_bind_new_manifest(self) -> None:
        old_source = copy.deepcopy(GATE.SOURCE_IDENTITY)
        pooling = next(row for row in old_source["files"] if row["path"].endswith("pooling_layers.py"))
        pooling.update(git_blob_sha1="47120eead47a511939267470496539804c17b7d3", sha256="768910f8e88cb47e742274563339d7e780cb9d56c629c4d4124605296686f0f9")
        with tempfile.TemporaryDirectory(prefix="wespeaker-source-approval-") as temp:
            root = Path(temp).resolve()
            project = root / "project"
            project.mkdir()
            shutil.copy2(HERE / "wespeaker" / "uv.lock", project / "uv.lock")
            shutil.copy2(HERE / "wespeaker" / "pyproject.toml", project / "pyproject.toml")
            old_manifest, old_evidence, _ = self._approved_bundle(root / "old", source_identity=old_source)
            new_manifest, new_evidence, new_data = self._approved_bundle(root / "new")
            self.assertEqual(GATE.validate(project, new_manifest, new_evidence, _self_test=True), (True, "PASS"))
            self.assertEqual(GATE.validate(project, new_manifest, old_evidence, _self_test=True), (False, "external approval evidence does not bind the manifest/scope"))
            self.assertFalse(GATE.validate(project, old_manifest, old_evidence, _self_test=True)[0])
            stale_data = json.loads(new_manifest.read_text(encoding="utf-8"))
            old_data = json.loads(old_manifest.read_text(encoding="utf-8"))
            stale_data["approval_scope_sha256"] = old_data["approval_scope_sha256"]
            stale_data["operator_approval"]["digest"] = old_data["approval_scope_sha256"]
            stale = root / "stale.json"
            stale.write_text(json.dumps(stale_data), encoding="utf-8")
            self.assertEqual(GATE.validate(project, stale, old_evidence, _self_test=True), (False, "approval scope is not bound to exact closure"))
            self.assertNotEqual(new_data["approval_scope_sha256"], old_data["approval_scope_sha256"])

    def test_worker_shell_binding_uses_corrected_pooling_identity(self) -> None:
        source = WORKER.read_text(encoding="utf-8")
        self.assertIn('UPSTREAM_CHECKPOINT_REMOTE="avg_model"', source)
        self.assertIn('UPSTREAM_CHECKPOINT="avg_model.pt"', source)
        self.assertIn('download_hf_file "$UPSTREAM_HF" "$UPSTREAM_REVISION" "$UPSTREAM_CHECKPOINT_REMOTE" "$(dirname "$checkpoint")" "$UPSTREAM_CHECKPOINT"', source)
        self.assertNotIn("7f92ddd059d244c7d2653650d3be85de9f136c41", source)
        self.assertIn('SOURCE_POOLING_SHA256="768910c91e353f1d0c2f3edc8f9ae758d2ff7e477d37fb470ae946eba69adac8"', source)
        self.assertIn('SOURCE_POOLING_GIT_BLOB="47120eb9029ba1b00d18fcd61dd80f8afa2037c6"', source)
        self.assertIn('hash-object --no-filters "$relative"', source)
        self.assertNotIn("768910f8e88cb47e742274563339d7e780cb9d56c629c4d4124605296686f0f9", source)
        self.assertNotIn("47120eead47a511939267470496539804c17b7d3", source)

    def test_worker_offline_self_test_exercises_staging_races_and_token_boundary(self) -> None:
        result = subprocess.run(
            ["bash", str(WORKER), "--self-test"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr or result.stdout)
        self.assertIn("run-wespeaker-validation.sh self-test: OK", result.stdout)

    def test_python_source_verifier_uses_raw_git_blobs(self) -> None:
        source = (HERE / "wespeaker_dump_reference.py").read_text(encoding="utf-8")
        self.assertIn('"hash-object", "--no-filters", relative', source)


if __name__ == "__main__":
    unittest.main()
