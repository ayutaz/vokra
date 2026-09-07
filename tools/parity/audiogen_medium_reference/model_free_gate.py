#!/usr/bin/env python3
"""Dependency-free integrity gate for the reviewed AudioGen model-free evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any


EVIDENCE_SCHEMA = "vokra-audiogen-medium-model-free-evidence-v1"
EXPECTED_HEAD = "6371a40b9b6643c4754c2a0ba5a4f49a3195a4c7"
EXPECTED_HF_REVISION = "1277dd7dfd8fa57a205a70acc5de0ee90804502f"
EXPECTED_SOURCE_REVISION = "a2b96756956846e194c9255d0cdadc2b47c93f1b"
EXPECTED_ARTIFACTS = {
    "manifest.json": "6c2adb3e3948e138547aa79a2b36d8880db8bf39fc36af7b0d5b8eea9db29f4f",
    "tree.json": "c445caf1eb26c26ce414edb2fb9601744781a07240b0a2ff1333a1acf2f25a1d",
    "acquisition.log": "81d83040e72a2b2d4ad6eab59b823a6e1d2f2614171bdaa5fb3b6dc4c10d2960",
}
EXPECTED_PUBLIC_FILES = {
    ".gitattributes": {"bytes": 1519, "git_blob_sha1": "a6344aac8c09253b3b630fb776ae94478aa0275b"},
    "README.md": {"bytes": 2240, "git_blob_sha1": "31a77819df582937de900237706f104a325e223f"},
    "compression_state_dict.bin": {"bytes": 235740815, "lfs_pointer_git_blob_sha1": "0cc8de6c4cf0c16326ee3c693385370b98bbf0f2", "lfs_payload_sha256": "5a520e64ca99226a9956f83b06df0617b713183fcdc384779883a6bb46dc1095"},
    "state_dict.bin": {"bytes": 3678455287, "lfs_pointer_git_blob_sha1": "ae572ad32705a0a9ba679b0d2813cbae716d869e", "lfs_payload_sha256": "f3b20997834de1ca47d6a31d00a5dc37019b279c7c8f250fd482d56def04faaa"},
}
EXPECTED_SOURCE_ROLES = {
    "LICENSE": "b93be90515ccd0b9daedaa589e42bf5929693f1f",
    "LICENSE_weights": "108b5f002fc31efe11d881de2cd05329ebe8cc37",
    "audiocraft/models/audiogen.py": "5cb889982ddc027e2588b7cfb8ef428b313ce88a",
    "audiocraft/models/builders.py": "038bf99c3d0fbbb86005683d5a2a1b4edcac4298",
    "audiocraft/models/encodec.py": "40d133017c0a0eddaafb07d291b3845789775bc3",
    "audiocraft/models/lm.py": "8cefd2c58c3a337378579d6cd6469fd038cbb1ee",
    "audiocraft/models/loaders.py": "7fd49d84e21ed26c01919dcb8e05315fb3bdf398",
    "audiocraft/modules/codebooks_patterns.py": "3cf3bb41774700a679ffe4325236d0324a99c546",
    "audiocraft/modules/conditioners.py": "d10ac8dc96466375379c883cd62f7c04a1bb0a73",
    "audiocraft/modules/transformer.py": "048c06dfbb0ab4167afce95dffb73dcc343c2344",
    "audiocraft/modules/conv.py": "d115cbf8729b642ed78608bd00a4d0fd5afae6fd",
    "audiocraft/modules/lstm.py": "c0866175950c1ca4f6cca98649525e6481853bba",
    "audiocraft/modules/seanet.py": "3e5998e9153afb6e68ea410d565e00ea835db248",
    "audiocraft/quantization/core_vq.py": "da02a6ce3a7de15353f0fba9e826052beb67c436",
    "audiocraft/quantization/vq.py": "aa57bea59db95ddae35e0657f723ca3a29ee943b",
    "config/model/lm/audiogen_lm.yaml": "696f74620af193c12208ce66fdb93a37f8ea9d80",
    "config/solver/audiogen/audiogen_base_16khz.yaml": "dd6aee785c74db19ce9d6f488e68e6eeb471c026",
    "config/model/lm/model_scale/medium.yaml": "c825d1ff6c3b8cc9ae4959a898e14b40409d95e8",
    "audiocraft/grids/audiogen/audiogen_pretrained_16khz_eval.py": "12f6d402a3c4a113d4c37be062790fa435b72104",
    "config/conditioner/text2sound.yaml": "555d4b7c3cecf0ec06c8cb25440b2f426c098ad2",
    "config/solver/compression/encodec_audiogen_16khz.yaml": "654deaa01ba9cace3f7144cc91921791c081b32a",
    "config/model/encodec/encodec_large_nq4_s320.yaml": "5f2d77590afd8a81185358c705a6e42853e257c3",
    "config/model/encodec/default.yaml": "ec62c6c8ef9a686890bdca8b8f27a2f1c232205d",
}
EXPECTED_SOURCE_SEMANTICS = {
    "config/model/lm/audiogen_lm.yaml": {"conditioner": "text2sound", "n_q": 4, "card": 2048, "delays": [0, 1, 2, 3]},
    "config/solver/audiogen/audiogen_base_16khz.yaml": {"sample_rate_hz": 16000, "channels": 1, "compression_checkpoint": "internal"},
    "config/model/lm/model_scale/medium.yaml": {"dim": 1536, "heads": 24, "layers": 48},
    "audiocraft/grids/audiogen/audiogen_pretrained_16khz_eval.py": {"hf_repository": "facebook/audiogen-medium", "scale": "medium"},
    "config/conditioner/text2sound.yaml": {"model": "t5", "name": "t5-large", "finetune": False},
    "config/solver/compression/encodec_audiogen_16khz.yaml": {"sample_rate_hz": 16000, "channels": 1, "model": "encodec_large_nq4_s320"},
    "config/model/encodec/encodec_large_nq4_s320.yaml": {"n_filters": 64, "bins": 2048, "n_q": 4, "q_dropout": False},
    "config/model/encodec/default.yaml": {"dimension": 128, "ratios": [8, 5, 4, 2]},
}


def load_json(path: Path) -> Any:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_contract(evidence: dict[str, Any]) -> None:
    expected_keys = {"schema", "status", "signable", "vast_ready", "identity", "vast_artifacts", "public_files", "source_roles", "source_semantics", "license", "execution", "approval_scope", "blockers"}
    if set(evidence) != expected_keys or evidence["schema"] != EVIDENCE_SCHEMA:
        raise ValueError("evidence schema is incomplete")
    if evidence["status"] != "OWNER_INDEPENDENT_CLOSURE_INCOMPLETE" or evidence["signable"] is not False or evidence["vast_ready"] is not False:
        raise ValueError("incomplete evidence was marked signable/VAST_READY")
    identity = evidence["identity"]
    if identity != {"expected_head": EXPECTED_HEAD, "hf_repository": "facebook/audiogen-medium", "hf_revision": EXPECTED_HF_REVISION, "source_repository": "https://github.com/facebookresearch/audiocraft.git", "source_revision": EXPECTED_SOURCE_REVISION, "source_tag": "v1.0.0"}:
        raise ValueError("identity drift")
    if evidence["vast_artifacts"] != EXPECTED_ARTIFACTS:
        raise ValueError("VAST artifact hash drift")
    if evidence["public_files"] != EXPECTED_PUBLIC_FILES:
        raise ValueError("public file identity drift")
    roles = evidence["source_roles"]
    if roles != EXPECTED_SOURCE_ROLES or any("modules/quantization/" in path for path in roles):
        raise ValueError("source role/path/blob contract drift")
    if evidence["source_semantics"] != EXPECTED_SOURCE_SEMANTICS:
        raise ValueError("source semantic contract drift")
    if evidence["license"] != {"hf_model_card": "CC-BY-NC-4.0", "source_code": "MIT", "source_weights": "CC-BY-NC-4.0", "historical_v0_0_2_weights": "CC-BY-NC-ND-4.0", "status": "PROVENANCE_AMBIGUITY_BLOCKER"}:
        raise ValueError("license split/ambiguity drift")
    execution = evidence["execution"]
    if execution != {"payload_downloaded": False, "payload_bytes": 0, "model_loaded": False, "model_run": False, "upload": "NO_UPLOAD", "operator_or_real_execution": "PENDING_INCOMPLETE", "external_t5_repository_revision_weight": "PENDING_INCOMPLETE", "compression_weight_build_provenance": "PENDING_INCOMPLETE", "dependency_lock": "PENDING_INCOMPLETE"}:
        raise ValueError("execution gate is incomplete or unsafe")
    approval = evidence["approval_scope"]
    if approval != {"status": "PENDING_OWNER_APPROVAL", "record": None, "signer": None, "decision": None, "publication": "NO_UPLOAD", "signable": False}:
        raise ValueError("approval scope unexpectedly authorizes execution/publication")


def verify_artifact_bytes(artifact_dir: Path, expected_artifacts: dict[str, str]) -> None:
    for name, expected in expected_artifacts.items():
        path = artifact_dir / name
        if path.is_symlink() or not path.is_file() or sha256(path) != expected:
            raise ValueError(f"VAST artifact missing/tampered: {name}")


def verify_artifacts(evidence: dict[str, Any], artifact_dir: Path) -> None:
    if evidence.get("vast_artifacts") != EXPECTED_ARTIFACTS:
        raise ValueError("VAST artifact hash drift")
    verify_artifact_bytes(artifact_dir, EXPECTED_ARTIFACTS)
    manifest = load_json(artifact_dir / "manifest.json")
    if manifest.get("expected_head") != EXPECTED_HEAD or manifest.get("approval_evidence") != {"status": "PENDING_OWNER_APPROVAL"} or manifest.get("publication") != "NO_UPLOAD":
        raise ValueError("manifest approval/publication boundary drift")
    if any(row.get("payload_status") != "NOT_DOWNLOADED" or row.get("execution") != "NOT_PERFORMED" for row in manifest.get("archives", {}).values()):
        raise ValueError("manifest payload execution boundary drift")


def verify(evidence: dict[str, Any], artifact_dir: Path) -> None:
    verify_contract(evidence)
    verify_artifacts(evidence, artifact_dir)


def self_test() -> None:
    evidence_path = Path(__file__).with_name("model_free_evidence.json")
    evidence = load_json(evidence_path)
    with tempfile.TemporaryDirectory(prefix="audiogen-medium-gate-") as directory:
        root = Path(directory)
        for name in EXPECTED_ARTIFACTS:
            (root / name).write_bytes(b"fixture")
        synthetic_artifacts = {name: sha256(root / name) for name in EXPECTED_ARTIFACTS}
        verify_artifact_bytes(root, synthetic_artifacts)
        (root / "manifest.json").write_bytes(b"tampered")
        try:
            verify_artifact_bytes(root, synthetic_artifacts)
        except ValueError as error:
            assert str(error) == "VAST artifact missing/tampered: manifest.json"
        else:
            raise AssertionError("artifact bytes tamper was accepted")
        tampered = json.loads(json.dumps(evidence))
        tampered["signable"] = True
        try:
            verify_contract(tampered)
        except ValueError as error:
            assert str(error) == "incomplete evidence was marked signable/VAST_READY"
        else:
            raise AssertionError("unsafe signable evidence accepted")
        drifted = json.loads(json.dumps(evidence))
        drifted["identity"]["expected_head"] = "0" * 40
        try:
            verify_contract(drifted)
        except ValueError as error:
            assert str(error) == "identity drift"
        else:
            raise AssertionError("HEAD drift was accepted")
        drifted = json.loads(json.dumps(evidence))
        drifted["source_roles"] = {}
        try:
            verify_contract(drifted)
        except ValueError as error:
            assert str(error) == "source role/path/blob contract drift"
        else:
            raise AssertionError("source scope drift was accepted")
    print("audiogen_medium_model_free_gate --self-test: OK")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, default=Path(__file__).with_name("model_free_evidence.json"))
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.artifact_dir is None:
        parser.error("--artifact-dir is required")
    verify(load_json(args.evidence), args.artifact_dir)
    print(json.dumps({"status": "BLOCKED_OWNER_APPROVAL", "signable": False, "vast_ready": False, "publication": "NO_UPLOAD"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
