#!/usr/bin/env python3
"""Stdlib-only fail-closed gate for YuE xcodec-mini staging."""
from __future__ import annotations
import argparse, hashlib, json, re, sys, tempfile
from pathlib import Path
from typing import Any
import tomllib

LOCK_SHA256 = "5a05395c04e3c047714e4c3e6fa1f7849520c83e4343c3d07aaea23b3f1bf754"
PYPROJECT_SHA256 = "05ee8513b32d3bec6e9205c352363602177d7a52f3db525d3eb8bf1081181fb1"
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
PLACEHOLDERS = {"", "none", "null", "unresolved", "pending", "pending_review",
                "owner_review_required", "review_required", "todo"}
# The authenticated upstream import chain is not a lightweight RVQ-only
# module: its utility roles import these packages at module import time.  Keep
# the gate blocked until a genuine lock contains the whole closure; do not
# silently rely on a host/global installation.
IMPORT_CLOSURE_REQUIRED = {"matplotlib", "tensorboard", "omegaconf"}
VIRTUAL_PROJECT = {"virtual": "."}
PYPI_REGISTRY = "https://pypi.org/simple"
TORCH_CPU_REGISTRY = "https://download.pytorch.org/whl/cpu"
PROJECT_NAME = "vokra-yue-xcodec-mini-parity"
PROJECT_VERSION = "0.1.0"
MANIFEST_KEYS = {"approval_scope_sha256", "component_reviews", "dependency_reviews", "dependency_reviews_sha256", "fixed_identities", "gate_version", "lock_sha256", "no_upload", "operator_approval", "package_rows_sha256", "pyproject_sha256"}
UPSTREAM_REPO = "m-a-p/xcodec_mini_infer"
UPSTREAM_REVISION = "fe781a67815ab47b4a3a5fce1e8d0a692da7e4e5"
PUBLIC_IDENTITY = {"repo": "vokra/yue-xcodec-mini", "revision": "83c14a67ed792a0d5b3b61fff8ae35a04c6da8fa",
                   "path": "yue-xcodec-mini.gguf", "bytes": 1810001760,
                   "sha256": "60e21aa5335646080102196454d7ffad5e012467d6f5eb9b776bf07d666b02bc",
                   "manifest_sha256": "cc0a5e9a5a6f1cfbd93b1869bbcb70744814bd8c855d173949abbf6b6cc08f15", "tensor_count": 2145,
                   "license_spdx": "apache-2.0", "role": "historical_public_measurement_artifact"}
CHECKPOINTS = {
    "codec": {"repo": UPSTREAM_REPO, "revision": UPSTREAM_REVISION, "path": "final_ckpt/ckpt_00360000.pth", "bytes": 1360444883, "sha256": "c8c379ea2d3cbde1c8ba1b9717975220e79ba3f556bb161766fd5e4585dcd59c", "role": "codec_checkpoint"},
    "semantic": {"repo": UPSTREAM_REPO, "revision": UPSTREAM_REVISION, "path": "semantic_ckpts/hf_1_325000/pytorch_model.bin", "bytes": 377555286, "sha256": "c5ddbd7fa2468483cb9b2aa53117813471543dd278e65870333a56c54305f527", "role": "nonexecuted_semantic_checkpoint"},
    "decoder": {"repo": UPSTREAM_REPO, "revision": UPSTREAM_REVISION, "path": "decoders/decoder_151000.pth", "bytes": 72610550, "sha256": "8af97a29d3483f9d4a3755992837501bd7d6caa1a69382ed16e64039e0ea0998", "role": "vocos_decoder_checkpoint"},
}
SOURCE_IDENTITY = {
    "repo": "https://huggingface.co/m-a-p/xcodec_mini_infer", "revision": UPSTREAM_REVISION,
    "license_spdx": "apache-2.0", "role": "official_token_decoder_source",
    "card_readme": {"path": "README.md", "bytes": 31, "sha256": "4bcf87ecfbbb8e07a01b21415a970c8b53a5283bf6872b657040d3f45c9241f7", "git_blob_sha1": "7b95401dc46245ac339fc25059d4a56d90b4cde5", "license_spdx": "apache-2.0"},
    "license": {"path": None, "bytes": None, "sha256": None, "git_blob_sha1": None, "status": "UNRESOLVED", "reason": "HF tree has no LICENSE; code-source license object is not authenticated"},
    "files": [{"path": "quantization/__init__.py", "role": "rvq", "bytes": 271, "sha256": "34c806bc1cafc8b835926b6f6450bee769f95eb467cf1c19b4427e9dd7e55bbc", "git_blob_sha1": "bfabe52b8cb6f260cdda6137b34df2f4736bd02f"},
              {"path": "quantization/vq.py", "role": "rvq", "bytes": 4598, "sha256": "8f24a4a389bad6dec6d77a35526264a1acd07c29a69854274bc73ebda4c622f9", "git_blob_sha1": "aeca14f95177bcc8f5c3ab492845a38d01cff5f1"},
              {"path": "quantization/core_vq_lsx_version.py", "role": "rvq", "bytes": 16050, "sha256": "154e2c5ddbacd3b82c74bf18d7177ea4b011cbd71e6e5575c7265b70e58c2af0", "git_blob_sha1": "65f8d2405a8758087ca0590d2f8fe72053e7f65b"},
              {"path": "quantization/distrib.py", "role": "rvq", "bytes": 4109, "sha256": "79b8dbfe3dda4da10ea0d3e143b373d90dd920f40d4a7f6f7446412b3584f655", "git_blob_sha1": "e0985f5418ecad2c6fe5fe941c0e4dbafbc60d84"},
              {"path": "utils/utils.py", "role": "rvq_import", "bytes": 8484, "sha256": "8521062c4b1afae1366a100244449a7dcdcc79883bf1874e50f9954c66c2ccd2", "git_blob_sha1": "a3bf157c1ad64f6404078d2b8eaf8a864e451cd3"},
              {"path": "utils/ddp_utils.py", "role": "rvq_import", "bytes": 9108, "sha256": "a53a4efc83ab34c8655d61bbcae7e0965a573ecce3321f8c1cffc2ec6889644f", "git_blob_sha1": "2240124d8dcc3684a9831e1f40b55c2c916d463c"}],
    "repcodec_pcm_encode": {"license_spdx": None, "status": "UNRESOLVED", "reason": "mixed MIT and CC-BY-NC; PCM encode is not executed"},
}
FIXED = {"source": SOURCE_IDENTITY, "public": PUBLIC_IDENTITY, "checkpoints": CHECKPOINTS,
         "vocos": {"package": "vocos==0.1.0", "wheel_sha256": "0ac13eaef68596074301e912d781399b3defa4b4ca60b6bc52c8a4b9209ca235", "license_spdx": "apache-2.0", "role": "vocos_istft_decoder"}}

def sha256(data: bytes) -> str: return hashlib.sha256(data).hexdigest()
def canonical(value: Any) -> str: return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())
def load_json(path: Path | str) -> Any:
    text = path.read_text(encoding="utf-8") if isinstance(path, Path) else path
    duplicates: list[str] = []
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result: duplicates.append(key)
            result[key] = value
        return result
    value = json.loads(text, object_pairs_hook=pairs)
    if duplicates: raise ValueError("duplicate JSON keys: " + ", ".join(sorted(set(duplicates))))
    return value
def unresolved(value: Any) -> bool:
    return value is None or not isinstance(value, str) or re.sub(r"\s+", "_", value.strip().casefold()) in PLACEHOLDERS
def rows(lock: dict[str, Any]) -> list[dict[str, Any]]:
    packages = lock.get("package")
    if not isinstance(packages, list) or not packages: raise ValueError("lock package table missing")
    out = []; identities: set[tuple[str, str, str]] = set(); virtual_count = 0
    for item in packages:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str) or not isinstance(item.get("version"), str):
            raise ValueError("malformed package row")
        if not item["name"].strip() or not item["version"].strip(): raise ValueError("empty package identity")
        source = item.get("source")
        if not isinstance(source, dict) or (source != VIRTUAL_PROJECT and (set(source) != {"registry"} or source["registry"] not in {PYPI_REGISTRY, TORCH_CPU_REGISTRY})):
            raise ValueError("malformed package source")
        markers = item.get("resolution-markers", [])
        dependencies = item.get("dependencies", [])
        if not isinstance(markers, list) or not all(isinstance(v, str) for v in markers): raise ValueError("malformed resolution markers")
        if not isinstance(dependencies, list) or not all(isinstance(v, dict) and set(v) == {"name", "marker"} and isinstance(v["name"], str) and v["name"].strip() and isinstance(v["marker"], str) and v["marker"].strip() for v in dependencies): raise ValueError("malformed dependency list")
        sdist = item.get("sdist")
        wheels = item.get("wheels", [])
        if sdist is not None and not isinstance(sdist, dict): raise ValueError("malformed sdist row")
        if not isinstance(wheels, list) or not all(isinstance(v, dict) for v in wheels): raise ValueError("malformed wheel rows")
        identity = (item["name"], item["version"], json.dumps(source, sort_keys=True, separators=(",", ":")))
        if identity in identities: raise ValueError("duplicate package identity")
        identities.add(identity)
        if source == VIRTUAL_PROJECT: virtual_count += 1
        out.append({"name": item["name"], "version": item["version"], "source": item.get("source"),
                    "resolution-markers": markers, "dependencies": dependencies,
                    "sdist": sdist, "wheels": wheels})
    if virtual_count != 1: raise ValueError("lock must contain exactly one virtual project package")
    return sorted(out, key=lambda x: (x["name"], x["version"], json.dumps(x["source"], sort_keys=True)))
def artifact_blocker(package_rows: list[dict[str, Any]]) -> str | None:
    """Require every locked resolver artifact to be authenticated metadata."""
    for item in package_rows:
        source = item.get("source")
        if source == VIRTUAL_PROJECT:
            if item["name"] != PROJECT_NAME or item["version"] != PROJECT_VERSION:
                return f"unexpected virtual package: {item['name']}"
            continue
        if not isinstance(source, dict) or set(source) != {"registry"} or source["registry"] not in {PYPI_REGISTRY, TORCH_CPU_REGISTRY}:
            return f"resolver source is not an intended registry: {item['name']}"
        artifacts = [artifact for artifact in [item.get("sdist"), *item.get("wheels", [])] if artifact is not None]
        if not artifacts:
            return f"resolver artifact metadata missing: {item['name']}"
        for artifact in artifacts:
            if not isinstance(artifact, dict) or set(artifact) - {"url", "hash", "size", "upload-time"} or not isinstance(artifact.get("url"), str) or not re.match(r"^https://", artifact["url"]):
                return f"resolver artifact URL missing: {item['name']}"
            expected_host = "download-r2.pytorch.org" if source["registry"] == TORCH_CPU_REGISTRY else "files.pythonhosted.org"
            host_match = re.match(r"^https://([^/]+)/", artifact["url"])
            if host_match is None or host_match.group(1) != expected_host:
                return f"resolver artifact host is not bound to source: {item['name']}"
            digest = artifact.get("hash")
            if not isinstance(digest, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
                return f"resolver artifact hash missing: {item['name']}"
            if not isinstance(artifact.get("size"), int) or isinstance(artifact["size"], bool) or artifact["size"] <= 0:
                return f"resolver artifact size missing: {item['name']}"
    return None
def fixed_blocker() -> str | None:
    license_id = SOURCE_IDENTITY["license"]
    if license_id.get("path") is not None:
        if not isinstance(license_id.get("bytes"), int) or not HEX64.fullmatch(str(license_id.get("sha256"))) or not HEX40.fullmatch(str(license_id.get("git_blob_sha1"))):
            return "source LICENSE bytes/SHA/git-blob identity is unresolved"
    else:
        return "source code LICENSE object is absent from authenticated HF tree"
    for item in SOURCE_IDENTITY["files"]:
        if not isinstance(item.get("bytes"), int) or not HEX64.fullmatch(str(item.get("sha256"))) or not HEX40.fullmatch(str(item.get("git_blob_sha1"))):
            return f"source identity unresolved: {item['path']}"
    return None
def validate(project: Path, manifest_path: Path, evidence_path: Path | None = None, *, self_test=False) -> tuple[bool, str]:
    try:
        if manifest_path.is_symlink() or not manifest_path.is_file():
            return False, "manifest must be a regular file"
        if (project / "uv.lock").is_symlink() or (project / "pyproject.toml").is_symlink():
            return False, "lock/project must not be symlinks"
        lock_bytes = (project / "uv.lock").read_bytes(); project_bytes = (project / "pyproject.toml").read_bytes()
        manifest = load_json(manifest_path)
        project_doc = tomllib.loads(project_bytes.decode())
        package_rows = rows(tomllib.loads(lock_bytes.decode()))
    except (OSError, UnicodeError, json.JSONDecodeError, tomllib.TOMLDecodeError, ValueError) as exc:
        return False, f"gate input malformed: {exc}"
    if not isinstance(manifest, dict) or set(manifest) != MANIFEST_KEYS: return False, "manifest top-level schema drifted"
    project_meta = project_doc.get("project")
    if not isinstance(project_meta, dict) or project_meta.get("name") != PROJECT_NAME or project_meta.get("version") != PROJECT_VERSION: return False, "project identity drifted"
    if sha256(lock_bytes) != LOCK_SHA256 or manifest.get("lock_sha256") != LOCK_SHA256: return False, "lock bytes drifted"
    if sha256(project_bytes) != PYPROJECT_SHA256 or manifest.get("pyproject_sha256") != PYPROJECT_SHA256: return False, "project bytes drifted"
    if manifest.get("package_rows_sha256") != canonical(package_rows): return False, "package graph drifted"
    artifact_error = artifact_blocker(package_rows)
    if artifact_error and not self_test: return False, artifact_error
    missing_imports = sorted(IMPORT_CLOSURE_REQUIRED - {row["name"] for row in package_rows})
    if missing_imports and not self_test: return False, "official source import closure is not locked: " + ", ".join(missing_imports)
    if not self_test and fixed_blocker(): return False, fixed_blocker() or "fixed identity blocked"
    reviews = manifest.get("dependency_reviews")
    if not isinstance(reviews, list) or len(reviews) != len(package_rows) or manifest.get("dependency_reviews_sha256") != canonical(reviews): return False, "dependency review closure drifted"
    expected = {(r["name"], r["version"], json.dumps(r["source"], sort_keys=True)) for r in package_rows}
    seen = set()
    fields = {"id", "name", "version", "source", "status", "license", "native_review", "bundled_review", "payload_sha256"}
    for item in reviews:
        if not isinstance(item, dict) or set(item) != fields: return False, "dependency row schema drifted"
        key = (item.get("name"), item.get("version"), json.dumps(item.get("source"), sort_keys=True))
        if key in seen or key not in expected or item.get("id") != f"{item.get('name')}@{item.get('version')}": return False, "dependency identity drifted"
        seen.add(key)
        if item.get("status") != "REVIEWED" or unresolved(item.get("license")) or unresolved(item.get("native_review")) or unresolved(item.get("bundled_review")) or not HEX64.fullmatch(str(item.get("payload_sha256"))): return False, f"dependency review unresolved: {item.get('id')}"
    if seen != expected: return False, "dependency coverage drifted"
    if manifest.get("fixed_identities") != FIXED or manifest.get("no_upload") != "NO_UPLOAD": return False, "fixed identities or NO_UPLOAD drifted"
    components = manifest.get("component_reviews")
    component_expected = [{"id": key, "identity": value, "role": value.get("role", key)} for key, value in (("source", SOURCE_IDENTITY), ("public", PUBLIC_IDENTITY), *CHECKPOINTS.items(), ("vocos", FIXED["vocos"]))]
    if not isinstance(components, list) or len(components) != len(component_expected): return False, "component rows incomplete"
    for actual, fixed in zip(components, component_expected, strict=True):
        if not isinstance(actual, dict) or set(actual) != {"id", "identity", "role", "status", "license", "payload_sha256", "signer", "approval_digest"} or actual.get("id") != fixed["id"] or actual.get("identity") != fixed["identity"] or actual.get("role") != fixed["role"]: return False, "component identity drifted"
        if actual.get("status") != "REVIEWED" or unresolved(actual.get("license")) or not HEX64.fullmatch(str(actual.get("payload_sha256"))) or unresolved(actual.get("signer")): return False, f"component review unresolved: {actual.get('id')}"
        if actual.get("approval_digest") != canonical({k: actual[k] for k in ("id", "identity", "role", "status", "license", "payload_sha256")}): return False, "component approval is not bound"
    scope = {"lock_sha256": LOCK_SHA256, "pyproject_sha256": PYPROJECT_SHA256, "package_rows_sha256": manifest["package_rows_sha256"], "dependency_reviews": reviews, "component_reviews": components, "fixed_identities": FIXED, "no_upload": "NO_UPLOAD"}
    if manifest.get("approval_scope_sha256") != canonical(scope): return False, "approval scope drifted"
    approval = manifest.get("operator_approval")
    if not isinstance(approval, dict) or set(approval) != {"decision", "digest", "signer"} or approval.get("decision") != "APPROVED" or unresolved(approval.get("signer")) or approval.get("digest") != canonical(scope): return False, "operator approval pending"
    if any(row.get("signer") != approval.get("signer") for row in components): return False, "component signer is not operator-bound"
    if evidence_path is None: return False, "external approval evidence is required"
    try:
        if evidence_path.is_symlink() or not evidence_path.is_file():
            return False, "external approval evidence must be a regular file"
        evidence = load_json(evidence_path)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        return False, f"external approval evidence malformed: {exc}"
    fields = {"decision", "digest", "evidence_sha256", "manifest_sha256", "scope_sha256", "signer"}
    if not isinstance(evidence, dict) or set(evidence) != fields: return False, "external approval evidence schema drifted"
    unsigned = {key: evidence[key] for key in fields if key != "evidence_sha256"}
    if evidence.get("evidence_sha256") != canonical(unsigned): return False, "external evidence hash drifted"
    if evidence.get("decision") != "APPROVED" or evidence.get("digest") != canonical(scope) or evidence.get("scope_sha256") != canonical(scope) or evidence.get("manifest_sha256") != sha256(manifest_path.read_bytes()) or unresolved(evidence.get("signer")) or evidence.get("signer") != approval.get("signer"): return False, "external approval evidence does not bind closure"
    return True, "PASS"
def self_test() -> int:
    project = Path(__file__).resolve().parent; manifest = project / "license_gate_manifest.json"
    good, reason = validate(project, manifest)
    if good or not ("unresolved" in reason or "blocked" in reason or "pending" in reason or "resolver artifact" in reason or "LICENSE" in reason):
        print(f"unexpected production gate result: {reason}", file=sys.stderr); return 1
    # Exercise the byte/identity bindings with tampered evidence.  These must
    # fail before the unresolved production review rows are considered.
    original = load_json(manifest.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix="yue-xcodec-gate-") as directory:
        altered = Path(directory) / "manifest.json"
        for label, mutate in (
            ("lock", lambda m: m.__setitem__("lock_sha256", "0" * 64)),
            ("project", lambda m: m.__setitem__("pyproject_sha256", "0" * 64)),
            ("fixed source revision", lambda m: m["fixed_identities"]["source"].__setitem__("revision", "0" * 40)),
            ("public artifact", lambda m: m["fixed_identities"]["public"].__setitem__("sha256", "0" * 64)),
            ("manifest extra key", lambda m: m.__setitem__("extra", True)),
        ):
            candidate = load_json(json.dumps(original))
            mutate(candidate)
            altered.write_text(json.dumps(candidate), encoding="utf-8")
            accepted, actual = validate(project, altered)
            if accepted or not actual:
                print(f"tamper self-test failed ({label}): {actual}", file=sys.stderr)
                return 1
        package_rows = rows(tomllib.loads((project / "uv.lock").read_bytes().decode()))
        artifact_tamper = load_json(json.dumps(package_rows))
        first_artifact = artifact_tamper[0]["sdist"] or artifact_tamper[0]["wheels"][0]
        first_artifact["size"] = None
        if artifact_blocker(artifact_tamper) is None:
            print("artifact metadata tamper self-test failed", file=sys.stderr)
            return 1
        lock_doc = tomllib.loads((project / "uv.lock").read_bytes().decode())
        for label, mutate in (
            ("duplicate package", lambda p: p.extend([dict(p[0])])),
            ("malformed wheels", lambda p: p[0].__setitem__("wheels", {})),
            ("malformed sdist", lambda p: p[0].__setitem__("sdist", "bad")),
            ("malformed source", lambda p: p[0].__setitem__("source", "bad")),
            ("malformed markers", lambda p: p[0].__setitem__("resolution-markers", "bad")),
            ("malformed dependencies", lambda p: p[0].__setitem__("dependencies", "bad")),
        ):
            candidate = load_json(json.dumps(lock_doc)); mutate(candidate["package"])
            try: rows(candidate)
            except ValueError: pass
            else:
                print(f"lock row tamper self-test failed ({label})", file=sys.stderr)
                return 1
        # A fully synthetic, independently written approval is the positive
        # contract test.  It is never used by production: fixed source code
        # licensing, artifact metadata, and all real review rows remain
        # unresolved in the committed manifest.
        approved = load_json(json.dumps(original))
        approved["dependency_reviews"] = [
            {**row, "status": "REVIEWED", "license": "SELF_TEST_LICENSE",
             "native_review": "SELF_TEST", "bundled_review": "SELF_TEST",
             "payload_sha256": "a" * 64}
            for row in approved["dependency_reviews"]
        ]
        approved["dependency_reviews_sha256"] = canonical(approved["dependency_reviews"])
        approved["component_reviews"] = [
            {**row, "status": "REVIEWED", "license": "SELF_TEST_LICENSE",
             "payload_sha256": "b" * 64, "signer": "self-test-owner",
             "approval_digest": None}
            for row in approved["component_reviews"]
        ]
        for row in approved["component_reviews"]:
            row["approval_digest"] = canonical({k: row[k] for k in ("id", "identity", "role", "status", "license", "payload_sha256")})
        scope = {"lock_sha256": LOCK_SHA256, "pyproject_sha256": PYPROJECT_SHA256,
                 "package_rows_sha256": approved["package_rows_sha256"],
                 "dependency_reviews": approved["dependency_reviews"],
                 "component_reviews": approved["component_reviews"],
                 "fixed_identities": FIXED, "no_upload": "NO_UPLOAD"}
        approved["approval_scope_sha256"] = canonical(scope)
        approved["operator_approval"] = {"decision": "APPROVED", "signer": "self-test-owner", "digest": canonical(scope)}
        altered.write_text(json.dumps(approved, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        evidence = Path(directory) / "approval.json"
        unsigned = {"decision": "APPROVED", "digest": canonical(scope),
                    "manifest_sha256": sha256(altered.read_bytes()),
                    "scope_sha256": canonical(scope), "signer": "self-test-owner"}
        evidence.write_text(json.dumps({**unsigned, "evidence_sha256": canonical(unsigned)}, sort_keys=True) + "\n", encoding="utf-8")
        accepted, actual = validate(project, altered, evidence, self_test=True)
        if not accepted:
            print(f"approved baseline self-test failed: {actual}", file=sys.stderr)
            return 1
        approved_manifest_text = altered.read_text(encoding="utf-8")
        approved_evidence_text = evidence.read_text(encoding="utf-8")
        for label, target in (("manifest non-object", altered), ("evidence non-object", evidence)):
            target.write_text("[]\n", encoding="utf-8")
            accepted, reason = validate(project, altered, evidence, self_test=True)
            target.write_text(approved_manifest_text if target is altered else approved_evidence_text, encoding="utf-8")
            if accepted or "schema" not in reason:
                print(f"non-object JSON self-test failed ({label}): {reason}", file=sys.stderr)
                return 1
        duplicate_cases = (
            ("manifest duplicate", altered, approved_manifest_text.rstrip()[:-1] + ',\n  "gate_version": 1\n}'),
            ("manifest nested duplicate", altered, approved_manifest_text.replace('"fixed_identities": {', '"fixed_identities": {"source": {},')),
            ("evidence duplicate", evidence, approved_evidence_text.rstrip()[:-1] + ',\n  "decision": "APPROVED"\n}'),
            ("evidence nested duplicate", evidence, approved_evidence_text.rstrip()[:-1] + ',\n  "nested": {"scope": "ok", "scope": "tampered"}\n}'),
        )
        for label, target, text in duplicate_cases:
            target.write_text(text, encoding="utf-8")
            accepted, reason = validate(project, altered, evidence, self_test=True)
            target.write_text(approved_manifest_text if target is altered else approved_evidence_text, encoding="utf-8")
            if accepted or "duplicate JSON keys" not in reason:
                print(f"duplicate JSON self-test failed ({label}): {reason}", file=sys.stderr)
                return 1
        for label, mutate in (
            ("evidence manifest", lambda e: e.__setitem__("manifest_sha256", "0" * 64)),
            ("evidence scope", lambda e: e.__setitem__("scope_sha256", "0" * 64)),
            ("evidence signer", lambda e: e.__setitem__("signer", "tampered")),
            ("evidence decision", lambda e: e.__setitem__("decision", "REJECTED")),
            ("evidence hash", lambda e: e.__setitem__("evidence_sha256", "0" * 64)),
        ):
            candidate = load_json(evidence.read_text(encoding="utf-8"))
            mutate(candidate)
            evidence.write_text(json.dumps(candidate, sort_keys=True) + "\n", encoding="utf-8")
            accepted, _ = validate(project, altered, evidence, self_test=True)
            if accepted:
                print(f"evidence tamper self-test failed ({label})", file=sys.stderr)
                return 1
            evidence.write_text(json.dumps({**unsigned, "evidence_sha256": canonical(unsigned)}, sort_keys=True) + "\n", encoding="utf-8")
        recomputed = dict(unsigned)
        recomputed["signer"] = "different-owner"
        evidence.write_text(json.dumps({**recomputed, "evidence_sha256": canonical(recomputed)}, sort_keys=True) + "\n", encoding="utf-8")
        accepted, _ = validate(project, altered, evidence, self_test=True)
        if accepted:
            print("recomputed different-signer self-test failed", file=sys.stderr)
            return 1
    print("yue_xcodec_mini preflight gate: self-test PASS")
    return 0
if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--project", type=Path); parser.add_argument("--manifest", type=Path); parser.add_argument("--approval-evidence", type=Path); parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test: raise SystemExit(self_test())
    if args.project is None or args.manifest is None or args.approval_evidence is None: parser.error("--project, --manifest and --approval-evidence are required")
    ok, reason = validate(args.project, args.manifest, args.approval_evidence)
    if not ok: print(f"yue_xcodec_mini preflight gate: BLOCKED: {reason}", file=sys.stderr); raise SystemExit(2)
    print("yue_xcodec_mini preflight gate: PASS")
