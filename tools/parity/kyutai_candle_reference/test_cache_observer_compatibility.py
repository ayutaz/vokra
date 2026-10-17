"""Offline source-only tests for the Moshi cache-observer compatibility runner."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from contextlib import contextmanager
from unittest import mock


LEAF = Path(__file__).resolve().parent
sys.path.insert(0, str(LEAF))
SPEC = importlib.util.spec_from_file_location(
    "cache_observer_compatibility", LEAF / "check_cache_observer_compatibility.py"
)
assert SPEC is not None and SPEC.loader is not None
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)

PINNED_ASR = LEAF / "pinned-asr.rs"
PINNED_TRANSFORMER = LEAF / "pinned-transformer.rs"
PINNED_LM = LEAF / "pinned-lm.rs"
PINNED_LICENSE = LEAF / "pinned-LICENSE"


@contextmanager
def temporary_directory(prefix: str):
    canonical_root = Path(tempfile.gettempdir()).resolve()
    with tempfile.TemporaryDirectory(prefix=prefix, dir=canonical_root) as directory:
        yield Path(directory).resolve()


class CacheObserverCompatibilityTests(unittest.TestCase):
    def test_legacy_application_contract_is_reused_exactly(self) -> None:
        self.assertIs(CHECKER.APPLICATION_FILES, CHECKER.legacy.APPLICATION_FILES)
        self.assertEqual(len(CHECKER.APPLICATION_FILES), 8)
        self.assertEqual(CHECKER.FINAL_LOCK_SHA256, CHECKER.legacy.FINAL_LOCK_SHA256)
        self.assertEqual(CHECKER.ROOT_MANIFEST_SHA256, CHECKER.legacy.ROOT_MANIFEST_SHA256)
        self.assertEqual(CHECKER.legacy.CPU_FINGERPRINT, CHECKER.CPU_FINGERPRINT)
        self.assertEqual(CHECKER.legacy.SOURCE_RECEIPT_SHA256, CHECKER.SOURCE_RECEIPT_SHA256)

    def test_distinct_remote_marker_is_required(self) -> None:
        self.assertNotEqual(CHECKER.REMOTE_OPT_IN, "VOKRA_CANDLE_REMOTE_LINUX")
        with temporary_directory("moshi-cache-compat-marker-") as temporary:
            output = temporary / "receipt.json"
            with mock.patch.object(CHECKER.sys, "platform", "linux"), mock.patch.dict(
                os.environ,
                {"VOKRA_CANDLE_REMOTE_LINUX": "1"},
                clear=False,
            ):
                self.assertEqual(CHECKER.check(temporary / "application", output, 1), 2)
            receipt = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(receipt["status"], CHECKER.STATUS_FAILED)
            self.assertIn(CHECKER.REMOTE_OPT_IN, receipt["error"]["message"])

    def test_success_receipt_is_source_only_and_fail_closed(self) -> None:
        receipt = CHECKER._initial_receipt()
        self.assertEqual(receipt["status"], CHECKER.STATUS_PENDING)
        self.assertIsNone(receipt["approval"])
        self.assertEqual(receipt["model_execution"], "NOT_AUTHORIZED")
        self.assertEqual(receipt["upload"], "NO_UPLOAD")
        for gate in ("runtime", "default_none", "batched", "pcm", "real_weight", "cpu", "metal", "raw_kv_capture", "independent_pcm_reference"):
            self.assertEqual(receipt["gates"][gate], "NOT_RUN")
        self.assertEqual(receipt["gates"]["dependency_license_owner_review"], "BLOCKED_PENDING_REVIEW")
        self.assertEqual(receipt["gates"]["model_weights"], "NOT_ACQUIRED")
        self.assertEqual(receipt["gates"]["publication"], "NO_UPLOAD")

    def test_snapshot_diff_allows_only_three_patched_sources(self) -> None:
        before = {
            "projection": {
                "moshi-core/src/asr.rs": {"bytes": 1, "sha256": "old-a"},
                "moshi-core/src/transformer.rs": {"bytes": 1, "sha256": "old-t"},
                "moshi-core/src/lm.rs": {"bytes": 1, "sha256": "old-l"},
                "Cargo.toml": {"bytes": 1, "sha256": "manifest"},
            },
            "vendor": {"vendor.rs": {"bytes": 1, "sha256": "vendor"}},
            "registry_sources": {"tiny@1.0.0": {"src.rs": {"bytes": 1, "sha256": "registry"}}},
        }
        after = json.loads(json.dumps(before))
        after["projection"]["moshi-core/src/asr.rs"]["sha256"] = "new-a"
        after["projection"]["moshi-core/src/transformer.rs"]["sha256"] = "new-t"
        after["projection"]["moshi-core/src/lm.rs"]["sha256"] = "new-l"
        CHECKER._assert_only_source_changes(
            before,
            after,
            patched_sha256={
                "moshi-core/src/asr.rs": "new-a",
                "moshi-core/src/transformer.rs": "new-t",
                "moshi-core/src/lm.rs": "new-l",
            },
        )
        after["vendor"]["unexpected.rs"] = {"bytes": 1, "sha256": "bad"}
        with self.assertRaises(CHECKER.CompatibilityError):
            CHECKER._assert_only_source_changes(
                before,
                after,
                patched_sha256={
                    "moshi-core/src/asr.rs": "new-a",
                    "moshi-core/src/transformer.rs": "new-t",
                    "moshi-core/src/lm.rs": "new-l",
                },
            )

    def test_snapshot_patch_map_rejects_missing_or_extra_byte_bindings(self) -> None:
        before = {"projection": {}, "vendor": {}, "registry_sources": {}}
        after = {"projection": {}, "vendor": {}, "registry_sources": {}}
        with self.assertRaises(CHECKER.CompatibilityError):
            CHECKER._assert_only_source_changes(
                before,
                after,
                patched_sha256={path: "hash" for path in CHECKER.SOURCE_FILES[:2]},
            )
        with self.assertRaises(CHECKER.CompatibilityError):
            CHECKER._assert_only_source_changes(
                before,
                after,
                patched_sha256={path: "hash" for path in CHECKER.SOURCE_FILES},
                patched_bytes={path: 1 for path in CHECKER.SOURCE_FILES[:2]},
            )

    def test_exact_source_replacement_preserves_other_projection_files(self) -> None:
        with temporary_directory("moshi-cache-compat-replace-") as workspace:
            projection = workspace / "projection"
            for relative in CHECKER.SOURCE_FILES:
                path = projection / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"original:" + relative.encode())
            manifest = projection / "Cargo.toml"
            manifest.write_bytes(b"manifest")
            patched = {relative: b"patched:" + relative.encode() for relative in CHECKER.SOURCE_FILES}
            CHECKER._replace_exact_sources(workspace, patched)
            for relative, expected in patched.items():
                self.assertEqual((projection / relative).read_bytes(), expected)
            self.assertEqual(manifest.read_bytes(), b"manifest")

    def test_receipt_binds_all_three_outputs_and_reversal(self) -> None:
        with temporary_directory("moshi-cache-compat-receipt-") as root:
            source_paths = {
                "asr": root / "asr.rs",
                "transformer": root / "transformer.rs",
                "lm": root / "lm.rs",
                "license": root / "LICENSE",
            }
            source_paths["asr"].write_bytes(PINNED_ASR.read_bytes())
            source_paths["transformer"].write_bytes(PINNED_TRANSFORMER.read_bytes())
            source_paths["lm"].write_bytes(PINNED_LM.read_bytes())
            source_paths["license"].write_bytes(PINNED_LICENSE.read_bytes())
            outputs = {key: root / f"patched-{key}.rs" for key in ("asr", "transformer", "lm")}
            receipt_path = root / "receipt.json"
            CHECKER.observer.apply(
                source_paths["asr"],
                source_paths["transformer"],
                source_paths["lm"],
                source_paths["license"],
                outputs["asr"],
                outputs["transformer"],
                outputs["lm"],
                receipt_path,
            )
            source_bytes = {key: path.read_bytes() for key, path in source_paths.items()}
            receipt = CHECKER._validate_observer_receipt(
                receipt_path,
                source_paths=source_paths,
                source_bytes=source_bytes,
                output_paths=outputs,
            )
            self.assertEqual(set(receipt["patched_outputs"]), {"asr", "transformer", "lm"})
            self.assertEqual(CHECKER.observer.revert_asr_bytes(outputs["asr"].read_bytes()), source_bytes["asr"])
            self.assertEqual(CHECKER.observer.revert_transformer_bytes(outputs["transformer"].read_bytes()), source_bytes["transformer"])
            self.assertEqual(CHECKER.observer.revert_lm_bytes(outputs["lm"].read_bytes()), source_bytes["lm"])

    def test_receipt_rejects_forged_suffix_paths_and_output_sets(self) -> None:
        with temporary_directory("moshi-cache-compat-receipt-negative-") as root:
            source_paths = {
                "asr": root / "asr.rs",
                "transformer": root / "transformer.rs",
                "lm": root / "lm.rs",
                "license": root / "LICENSE",
            }
            source_paths["asr"].write_bytes(PINNED_ASR.read_bytes())
            source_paths["transformer"].write_bytes(PINNED_TRANSFORMER.read_bytes())
            source_paths["lm"].write_bytes(PINNED_LM.read_bytes())
            source_paths["license"].write_bytes(PINNED_LICENSE.read_bytes())
            outputs = {key: root / f"patched-{key}.rs" for key in ("asr", "transformer", "lm")}
            receipt_path = root / "receipt.json"
            CHECKER.observer.apply(
                source_paths["asr"], source_paths["transformer"], source_paths["lm"], source_paths["license"],
                outputs["asr"], outputs["transformer"], outputs["lm"], receipt_path,
            )
            source_bytes = {key: path.read_bytes() for key, path in source_paths.items()}
            forged = json.loads(receipt_path.read_text(encoding="utf-8"))
            forged["source"]["path"] = "prefix/asr.rs"
            receipt_path.write_text(json.dumps(forged), encoding="utf-8")
            with self.assertRaises(CHECKER.CompatibilityError):
                CHECKER._validate_observer_receipt(receipt_path, source_paths=source_paths, source_bytes=source_bytes, output_paths=outputs)
            forged["source"]["path"] = "rust/moshi-core/src/asr.rs"
            forged["patched_outputs"].pop("lm")
            receipt_path.write_text(json.dumps(forged), encoding="utf-8")
            with self.assertRaises(CHECKER.CompatibilityError):
                CHECKER._validate_observer_receipt(receipt_path, source_paths=source_paths, source_bytes=source_bytes, output_paths=outputs)

    def test_receipt_rejects_each_forged_provenance_gate_and_output_binding(self) -> None:
        with temporary_directory("moshi-cache-compat-receipt-fields-") as root:
            source_paths, outputs, receipt_path, source_bytes = self._make_receipt_fixture(root)
            baseline = json.loads(receipt_path.read_text(encoding="utf-8"))

            def rejected(mutator) -> None:
                candidate = json.loads(json.dumps(baseline))
                mutator(candidate)
                receipt_path.write_text(json.dumps(candidate), encoding="utf-8")
                with self.assertRaises(CHECKER.CompatibilityError):
                    CHECKER._validate_observer_receipt(
                        receipt_path,
                        source_paths=source_paths,
                        source_bytes=source_bytes,
                        output_paths=outputs,
                    )

            rejected(lambda value: value["source"].__setitem__("commit", "wrong"))
            rejected(lambda value: value["source"].__setitem__("official_raw_url", "https://wrong.invalid/asr.rs"))
            rejected(lambda value: value.pop("license_approval"))
            rejected(lambda value: value.__setitem__("license_approval", "approved"))
            rejected(lambda value: value["candle_api_provenance"].__setitem__("commit", "wrong"))
            rejected(lambda value: value["candle_api_provenance"]["used_accessors"].__setitem__(0, "invented"))
            rejected(lambda value: value["candle_api_provenance"].pop("license_approval"))
            for gate in CHECKER.EXPECTED_OBSERVER_GATES:
                rejected(lambda value, gate=gate: value["gates"].__setitem__(gate, "PASS"))
            for gate in CHECKER.EXPECTED_OBSERVER_GATES:
                rejected(lambda value, gate=gate: value["gates"].pop(gate))
            rejected(lambda value: value["patched_outputs"]["asr"].__setitem__("sha256", "0" * 64))
            rejected(lambda value: value["patched_outputs"]["asr"].__setitem__("bytes", 1))
            rejected(lambda value: value["patched_outputs"]["asr"].pop("sha256"))
            rejected(lambda value: value["patched_outputs"].__setitem__("extra", {}))
            rejected(lambda value: value.__setitem__("source_sha256", "0" * 64))
            wrong_asr = bytearray(source_bytes["asr"])
            wrong_asr[0] ^= 1
            receipt_path.write_text(json.dumps(baseline), encoding="utf-8")
            with self.assertRaises(CHECKER.CompatibilityError):
                CHECKER._validate_observer_receipt(
                    receipt_path,
                    source_paths=source_paths,
                    source_bytes={**source_bytes, "asr": bytes(wrong_asr)},
                    output_paths=outputs,
                )

            symlink = root / "patched-asr-link.rs"
            symlink.symlink_to(outputs["asr"])
            with self.assertRaises(CHECKER.CompatibilityError):
                CHECKER._validate_observer_receipt(
                    receipt_path,
                    source_paths=source_paths,
                    source_bytes=source_bytes,
                    output_paths={**outputs, "asr": symlink},
                )
            receipt_path.write_bytes(b"{" + b" " * (CHECKER.MAX_RECEIPT_ARTIFACT_BYTES + 1) + b"}")
            with self.assertRaises(CHECKER.CompatibilityError):
                CHECKER._validate_observer_receipt(
                    receipt_path,
                    source_paths=source_paths,
                    source_bytes=source_bytes,
                    output_paths=outputs,
                )
    def test_snapshot_summary_binds_ninety_registry_sources_shape(self) -> None:
        registry = {
            f"package{i}@1.0.0": {"src.rs": {"bytes": i, "sha256": str(i)}}
            for i in range(90)
        }
        snapshot = {"projection": {}, "vendor": {}, "registry_sources": registry}
        summary = CHECKER._snapshot_summary(snapshot)
        self.assertEqual(summary["registry_sources"]["packages"], 90)
        self.assertEqual(len(summary["registry_sources"]["sha256"]), 64)

    def test_api_probe_contract_mentions_original_and_v2_types(self) -> None:
        source = (LEAF / "check_cache_observer_compatibility.py").read_text(encoding="utf-8")
        for token in (
            "Mimi::encode_step",
            "Mimi::decode_step",
            "State::model_step_idx",
            "State::set_observer",
            "State::set_observer_v2",
            "asr::AsrObserverV2",
            "transformer::StreamingTransformer::cache_observation",
            "lm::LmModel::cache_observation",
            "RawKvObservation",
            "k_offset",
            "v_offset",
            "k_dim",
            "v_dim",
            "k_max_seq_len",
            "v_max_seq_len",
            "k_current_seq_len",
            "v_current_seq_len",
            "AsrItemObservation",
            "global_model_step_idx",
        ):
            self.assertIn(token, source)
        self.assertNotIn("cargo run", source)
        self.assertNotIn("huggingface_hub", source.lower())

    def test_source_contract_has_no_activation_or_model_path(self) -> None:
        source = (LEAF / "check_cache_observer_compatibility.py").read_text(encoding="utf-8")
        self.assertIn("SOURCE_COMPATIBILITY_REVIEW_PENDING", source)
        self.assertIn('"model_execution": "NOT_AUTHORIZED"', source)
        self.assertIn('"upload": "NO_UPLOAD"', source)
        self.assertIn("observer.apply(", source)
        self.assertIn("_source_snapshot_detailed", source)
        self.assertIn("_assert_only_source_changes", source)
        self.assertNotIn("workflow_dispatch", source)

    def _proof(self) -> dict[str, object]:
        return {
            "source_proof_pre": {"files_checked": 23, "unchanged_files": 21},
            "source_proof_post": {"files_checked": 23, "unchanged_files": 21},
            "member_premetadata_checks": [{"members_checked": 308, "non_manifest_members": 292}],
            "member_postchecks": [{"members_checked": 308, "non_manifest_members": 292}],
        }

    def _make_receipt_fixture(self, root: Path):
        source_paths = {
            "asr": root / "asr.rs",
            "transformer": root / "transformer.rs",
            "lm": root / "lm.rs",
            "license": root / "LICENSE",
        }
        source_paths["asr"].write_bytes(PINNED_ASR.read_bytes())
        source_paths["transformer"].write_bytes(PINNED_TRANSFORMER.read_bytes())
        source_paths["lm"].write_bytes(PINNED_LM.read_bytes())
        source_paths["license"].write_bytes(PINNED_LICENSE.read_bytes())
        outputs = {key: root / f"patched-{key}.rs" for key in ("asr", "transformer", "lm")}
        receipt_path = root / "receipt.json"
        CHECKER.observer.apply(
            source_paths["asr"], source_paths["transformer"], source_paths["lm"], source_paths["license"],
            outputs["asr"], outputs["transformer"], outputs["lm"], receipt_path,
        )
        source_bytes = {key: path.read_bytes() for key, path in source_paths.items()}
        return source_paths, outputs, receipt_path, source_bytes

    def _mocked_check(
        self,
        *,
        preflight_stdout: str = "rustc 1.99.0\ncargo 1.99.0\n",
        drift: bool = False,
        post_build_drift: bool = False,
        final_drift: bool = False,
        receipt_error: Exception | None = None,
        compiler_error: Exception | None = None,
        partial_patch: bool = False,
    ):
        root = Path(tempfile.gettempdir()).resolve()
        temporary = tempfile.TemporaryDirectory(prefix="moshi-cache-compat-causal-", dir=root)
        directory = Path(temporary.name).resolve()
        output = directory / "receipt.json"
        proof = self._proof()
        stages: list[str] = []
        events: list[str] = []
        source_bytes = {
            "asr": PINNED_ASR.read_bytes(),
            "transformer": PINNED_TRANSFORMER.read_bytes(),
            "lm": PINNED_LM.read_bytes(),
            "license": PINNED_LICENSE.read_bytes(),
        }
        patched = CHECKER.observer.patch_bytes(source_bytes["asr"], source_bytes["transformer"], source_bytes["lm"])
        before = {
            "projection": {
                "moshi-core/src/asr.rs": {"bytes": len(source_bytes["asr"]), "sha256": "old-a"},
                "moshi-core/src/transformer.rs": {"bytes": len(source_bytes["transformer"]), "sha256": "old-t"},
                "moshi-core/src/lm.rs": {"bytes": len(source_bytes["lm"]), "sha256": "old-l"},
            },
            "vendor": {"vendor.rs": {"bytes": 1, "sha256": "vendor"}},
            "registry_sources": {"tiny@1.0.0": {"src.rs": {"bytes": 1, "sha256": "registry"}}},
        }
        after = json.loads(json.dumps(before))
        after["projection"]["moshi-core/src/asr.rs"] = {"bytes": len(patched[0]), "sha256": CHECKER.observer.sha256_bytes(patched[0])}
        after["projection"]["moshi-core/src/transformer.rs"] = {"bytes": len(patched[1]), "sha256": CHECKER.observer.sha256_bytes(patched[1])}
        after["projection"]["moshi-core/src/lm.rs"] = {"bytes": len(patched[2]), "sha256": CHECKER.observer.sha256_bytes(patched[2])}
        drift_after = json.loads(json.dumps(after))
        if drift:
            drift_after["vendor"]["vendor.rs"]["sha256"] = "drift"

        def fake_stage(stage, *, receipt, evidence_dir, argv, cwd, env, deadline, max_output=2 * 1024 * 1024):
            stages.append(stage)
            if stage == "checkout_head":
                return {"stdout": "a" * 40}
            if stage == "source_application":
                self.assertFalse(Path(argv[2]).exists())
                self.assertFalse(Path(argv[3]).exists())
                self.assertFalse(Path(argv[4]).exists())
                self.assertEqual(env.get("VOKRA_CANDLE_REMOTE_LINUX"), "1")
                return {"stdout": ""}
            if stage == "namespace_preflight":
                return {"stdout": preflight_stdout}
            if stage == "cargo_build":
                events.append("build")
                if compiler_error is not None:
                    raise compiler_error
                return {"stdout": ""}
            if stage == "post_build_cpu_metadata":
                events.append("postbuild_metadata")
                return {"stdout": "{}"}
            return {"stdout": ""}

        def fake_sha(path: Path) -> str:
            if path.name in {"final.lock", "Cargo.lock"}:
                return CHECKER.FINAL_LOCK_SHA256
            if path.name == "Cargo.toml":
                return CHECKER.ROOT_MANIFEST_SHA256
            if path.name == "source_receipt.json":
                return CHECKER.SOURCE_RECEIPT_SHA256
            return "unused"

        def fake_json(path: Path) -> dict[str, object]:
            if path.name == "source_receipt.json":
                return {"files": {str(index): {} for index in range(23)}}
            return {}

        def fake_apply(*args):
            events.append("patch_actual_3")
            output_paths = args[4:7]
            output_count = 1 if partial_patch else len(output_paths)
            for path, data in zip(output_paths[:output_count], patched[:output_count]):
                Path(path).write_bytes(data)
            Path(args[7]).write_text("{}", encoding="utf-8")

        snapshots_data = (before, after, after, after)
        snapshot_index = 0

        def fake_snapshot(*args, **kwargs):
            nonlocal snapshot_index
            events.append(("snapshot_prepatch", "snapshot_patched", "snapshot_postbuild", "snapshot_final")[snapshot_index])
            value = snapshots_data[snapshot_index]
            snapshot_index += 1
            return value

        if drift:
            snapshots_data = (before, drift_after, drift_after, drift_after)
        if post_build_drift:
            post_build = json.loads(json.dumps(after))
            post_build["vendor"]["vendor.rs"]["sha256"] = "post-build-drift"
            snapshots_data = (before, after, post_build, post_build)
        if final_drift:
            final = json.loads(json.dumps(after))
            final["vendor"]["vendor.rs"]["sha256"] = "final-drift"
            snapshots_data = (before, after, after, final)
        patchers = [
            mock.patch.object(CHECKER, "_verify_application", return_value=proof),
            mock.patch.object(CHECKER, "_verify_application_receipt", return_value=proof),
            mock.patch.object(CHECKER, "_verify_fixed_files"),
            mock.patch.object(CHECKER, "_run_stage", side_effect=fake_stage),
            mock.patch.object(CHECKER, "_sha256", side_effect=fake_sha),
            mock.patch.object(CHECKER, "_json", side_effect=fake_json),
            mock.patch.object(CHECKER, "_validate_cpu_metadata", return_value={"packages": 105, "nodes": 105, "fingerprint": CHECKER.CPU_FINGERPRINT}),
            mock.patch.object(CHECKER, "_authenticate_actual_inputs", return_value=source_bytes),
            mock.patch.object(CHECKER, "_source_snapshot_detailed", side_effect=fake_snapshot),
            mock.patch.object(CHECKER, "_validate_observer_receipt", side_effect=receipt_error or (lambda *args, **kwargs: {"schema": "mock"})),
            mock.patch.object(CHECKER.observer, "apply", side_effect=fake_apply),
            mock.patch.object(CHECKER, "_replace_exact_sources", side_effect=lambda *args, **kwargs: events.append("replace_actual_3")),
            mock.patch.object(CHECKER, "_artifact_ids", return_value=["moshi@0.0.0"]),
            mock.patch.object(CHECKER, "_metadata_package_identities", return_value={}),
            mock.patch.object(CHECKER, "_run_api_probe", side_effect=lambda **kwargs: (events.append("api_probe"), stages.append("api_probe"))),
            mock.patch.object(CHECKER, "_evidence_total", return_value=0),
            mock.patch.object(CHECKER, "_bind_trusted_toolchain"),
            mock.patch.object(CHECKER.os, "readlink", return_value="net:[host]"),
            mock.patch.object(CHECKER.sys, "platform", "linux"),
            mock.patch.dict(os.environ, {CHECKER.REMOTE_OPT_IN: "1", CHECKER.WORKFLOW_HEAD_ENV: "a" * 40}, clear=False),
        ]
        for patcher in patchers:
            patcher.start()
        try:
            result = CHECKER.check(directory / "application", output, 30)
            receipt = json.loads(output.read_text(encoding="utf-8")) if output.exists() else {}
            return temporary, result, receipt, stages, events
        finally:
            for patcher in reversed(patchers):
                patcher.stop()

    def test_mocked_full_check_order_and_success(self) -> None:
        temporary, result, receipt, stages, events = self._mocked_check()
        try:
            self.assertEqual(result, 0, receipt)
            self.assertEqual(stages, ["checkout_head", "source_application", "namespace_preflight", "cargo_build", "post_build_cpu_metadata", "api_probe"])
            self.assertEqual(events, ["snapshot_prepatch", "patch_actual_3", "replace_actual_3", "snapshot_patched", "build", "snapshot_postbuild", "postbuild_metadata", "api_probe", "snapshot_final"])
            self.assertEqual(receipt["status"], CHECKER.STATUS_PENDING)
            self.assertEqual(receipt["source_snapshot_before_build_state"], "PATCHED")
            self.assertIn("source_snapshot_post_build", receipt)
            self.assertEqual(receipt["replayed_application_proofs"]["source_pre"], 23)
            self.assertEqual(receipt["replayed_application_proofs"]["members_post"], 308)
        finally:
            temporary.cleanup()

    def test_frozen_hash_failure_prevents_any_stage(self) -> None:
        with temporary_directory("moshi-cache-compat-frozen-") as directory:
            output = directory / "receipt.json"
            stages: list[str] = []
            proof = self._proof()
            with mock.patch.object(CHECKER, "_verify_application", return_value=proof), mock.patch.object(CHECKER, "_verify_fixed_files", side_effect=CHECKER.CompatibilityError("frozen hash mismatch")), mock.patch.object(CHECKER, "_run_stage", side_effect=lambda *args, **kwargs: stages.append(args[0])) , mock.patch.object(CHECKER.sys, "platform", "linux"), mock.patch.dict(os.environ, {CHECKER.REMOTE_OPT_IN: "1", CHECKER.WORKFLOW_HEAD_ENV: "a" * 40}, clear=False):
                self.assertEqual(CHECKER.check(directory / "application", output, 30), 2)
            self.assertEqual(stages, [])
            self.assertIn("frozen hash mismatch", output.read_text(encoding="utf-8"))

    def test_preflight_wrong_version_prevents_build(self) -> None:
        temporary, result, receipt, stages, _events = self._mocked_check(preflight_stdout="rustc 1.98.0\ncargo 1.99.0\n")
        try:
            self.assertEqual(result, 2)
            self.assertIn("namespace preflight", receipt["error"]["message"])
            self.assertNotIn("cargo_build", stages)
        finally:
            temporary.cleanup()

    def test_snapshot_drift_prevents_build(self) -> None:
        temporary, result, receipt, stages, _events = self._mocked_check(drift=True)
        try:
            self.assertEqual(result, 2)
            self.assertIn("vendor source snapshot changed", receipt["error"]["message"])
            self.assertNotIn("namespace_preflight", stages)
        finally:
            temporary.cleanup()

    def test_receipt_validation_failure_prevents_build(self) -> None:
        temporary, result, receipt, stages, _events = self._mocked_check(receipt_error=CHECKER.CompatibilityError("output binding mismatch"))
        try:
            self.assertEqual(result, 2)
            self.assertIn("output binding mismatch", receipt["error"]["message"])
            self.assertNotIn("namespace_preflight", stages)
        finally:
            temporary.cleanup()

    def test_bad_reversal_prevents_build(self) -> None:
        with mock.patch.object(CHECKER.observer, "revert_asr_bytes", return_value=b"wrong"):
            temporary, result, receipt, stages, _events = self._mocked_check()
        try:
            self.assertEqual(result, 2)
            self.assertIn("ASR byte reversal failed", receipt["error"]["message"])
            self.assertNotIn("namespace_preflight", stages)
        finally:
            temporary.cleanup()

    def test_oracle_exception_prevents_build(self) -> None:
        temporary, result, receipt, stages, _events = self._mocked_check(
            receipt_error=ValueError("observer oracle failure")
        )
        try:
            self.assertEqual(result, 2)
            self.assertIn("observer oracle failure", receipt["error"]["message"])
            self.assertNotIn("namespace_preflight", stages)
        finally:
            temporary.cleanup()

    def test_partial_patch_prevents_build(self) -> None:
        temporary, result, receipt, stages, _events = self._mocked_check(partial_patch=True)
        try:
            self.assertEqual(result, 2)
            self.assertNotIn("namespace_preflight", stages)
        finally:
            temporary.cleanup()

    def test_post_build_snapshot_drift_prevents_metadata_and_probe(self) -> None:
        temporary, result, receipt, stages, _events = self._mocked_check(post_build_drift=True)
        try:
            self.assertEqual(result, 2)
            self.assertIn("cargo_build", stages)
            self.assertNotIn("post_build_cpu_metadata", stages)
            self.assertNotIn("api_probe", stages)
        finally:
            temporary.cleanup()

    def test_final_snapshot_drift_prevents_success(self) -> None:
        temporary, result, receipt, stages, _events = self._mocked_check(final_drift=True)
        try:
            self.assertEqual(result, 2)
            self.assertIn("api_probe", stages)
            self.assertEqual(receipt["status"], CHECKER.STATUS_FAILED)
        finally:
            temporary.cleanup()

    def test_compiler_failure_retains_bounded_failure_without_fake_pass(self) -> None:
        result = {"stdout": "o" * (2 * 1024 * 1024), "stderr": "e" * (2 * 1024 * 1024), "returncode": 1, "timed_out": False, "truncated": True}
        compiler_error = CHECKER.legacy.BoundedCommandError("compiler failed", result)
        temporary, status, receipt, stages, _events = self._mocked_check(compiler_error=compiler_error)
        try:
            self.assertEqual(status, 2)
            self.assertEqual(receipt["status"], CHECKER.STATUS_FAILED)
            self.assertLessEqual(len(receipt["failed_command"]["stdout"]), 32768)
            self.assertLessEqual(len(receipt["failed_command"]["stderr"]), 32768)
            self.assertNotIn("api_probe", stages)
        finally:
            temporary.cleanup()

    def test_timeout_flag_is_retained_in_bounded_failure(self) -> None:
        result = {"stdout": "o", "stderr": "e", "returncode": 124, "timed_out": True, "truncated": False}
        compiler_error = CHECKER.legacy.BoundedCommandError("compiler timed out", result)
        temporary, status, receipt, _stages, _events = self._mocked_check(compiler_error=compiler_error)
        try:
            self.assertEqual(status, 2)
            self.assertTrue(receipt["failed_command"]["timed_out"])
        finally:
            temporary.cleanup()

    def test_existing_output_fails_before_stage_launch(self) -> None:
        with temporary_directory("moshi-cache-compat-existing-") as directory:
            output = directory / "receipt.json"
            output.write_bytes(b"existing")
            with mock.patch.object(CHECKER, "_run_stage", side_effect=AssertionError("stage launched")):
                self.assertEqual(CHECKER.check(directory / "application", output, 30), 2)
            self.assertEqual(output.read_bytes(), b"existing")

    def test_output_overlapping_application_root_fails_before_stage_launch(self) -> None:
        with temporary_directory("moshi-cache-compat-overlap-") as directory:
            output = directory / "receipt.json"
            with mock.patch.object(CHECKER, "_run_stage", side_effect=AssertionError("stage launched")):
                self.assertEqual(CHECKER.check(directory, output, 30), 2)
            self.assertFalse(output.exists())

    def test_existing_evidence_root_and_packet_root_are_rejected_as_stale(self) -> None:
        with temporary_directory("moshi-cache-compat-stale-") as directory:
            output = directory / "receipt.json"
            (directory / "evidence").mkdir()
            with self.assertRaises(CHECKER.CompatibilityError):
                CHECKER._require_fresh_output_root(output, directory / "evidence")
            workspace_root = directory / "workspace"
            workspace_root.mkdir()
            (workspace_root / "packet").mkdir()
            with self.assertRaises(CHECKER.CompatibilityError):
                CHECKER._require_fresh_workspace_roots(workspace_root, ("packet", "application-output", "workspace"))


if __name__ == "__main__":
    unittest.main()
