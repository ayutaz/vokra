#!/usr/bin/env -S uv run --frozen --project tools/parity --python 3.12 python
"""Fail-closed Zonos evidence collector.

This tool deliberately does not download, execute, convert, or infer Zonos
weights. A VAST run may preserve server/local evidence and a supplied manifest
packet; the authenticated 246-tensor main model remains explicitly partial
until conditioning and the complete DAC path are bound.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
from pathlib import Path
from typing import Any

HF_REPOSITORY = "vokra/zonos-v0.1-transformer"
HF_REVISION = "b1bf5c56d470eb9097e9b04f9deca364576574ba"
UPSTREAM_HF_REPOSITORY = "Zyphra/Zonos-v0.1-transformer"
UPSTREAM_HF_REVISION = "9d8331fc49cb5ba8aad2bb56cafd809c66598f4e"
PUBLIC_GGUF = "zonos-v0.1-transformer.gguf"
PUBLIC_GGUF_BYTES = 3_248_843_808
PUBLIC_GGUF_SHA256 = "12d542bd219f7f31c91b893810d85b0d810285e603029c69fbd19fd3c7da2c5c"
EXPECTED_TENSOR_COUNT = 246
EXPECTED_MANIFEST_SHA256 = "6543af3747d3e85bde862c3337744eea31f0105f9df6d8617c1c9afdae805847"
FORMAT = "vokra-zonos-inspection-v1"
SOURCE_REPOSITORY = "https://github.com/Zyphra/Zonos.git"
SOURCE_REVISION = "bc40d98e1e1ab54fc65c483be127a90e3c7c0645"
SOURCE_ROLES = (
    "zonos/config.py",
    "zonos/model.py",
    "zonos/conditioning.py",
    "zonos/autoencoder.py",
    "zonos/codebook_pattern.py",
    "zonos/sampling.py",
    "zonos/backbone/_torch.py",
    "zonos/speaker_cloning.py",
)
STATUS_FIELDS = {
    "status": "BLOCKED",
    "evidence_stage": "AUTHENTICATED_ARTIFACT_SOURCE_EVIDENCE",
    "runtime_status": "SOURCE_STAGED_ONLY",
    "cpu_status": "NOT_RUN",
    "metal_status": "PENDING_REAL_APPLE_RUN",
    "parity_status": "NOT_RUN",
    "publication": "NO_UPLOAD",
}


def no_dupes(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def safe_path(value: Any) -> bool:
    if not isinstance(value, str) or not value or "\x00" in value or "\\" in value:
        return False
    path = Path(value)
    return not path.is_absolute() and all(part not in ("", ".", "..") for part in path.parts)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_sha256(rows: list[dict[str, Any]]) -> str:
    canonical = bytearray()
    for item in sorted(rows, key=lambda value: value["name"]):
        shape = item["shape"]
        canonical.extend(item["name"].encode("utf-8"))
        canonical.append(0)
        canonical.extend(struct.pack("<Q", len(shape)))
        for dimension in shape:
            canonical.extend(struct.pack("<Q", dimension))
    return hashlib.sha256(canonical).hexdigest()


def git_blob_sha1(path: Path) -> str:
    digest = hashlib.sha1()
    size = path.stat().st_size
    digest.update(f"blob {size}\0".encode())
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def lfs_pointer_sha1(oid: str, size: int) -> str:
    pointer = (
        "version https://git-lfs.github.com/spec/v1\n"
        f"oid sha256:{oid}\nsize {size}\n"
    ).encode()
    return hashlib.sha1(f"blob {len(pointer)}\0".encode() + pointer).hexdigest()


def local_files(root: Path) -> dict[str, Path]:
    if not root.is_dir():
        raise ValueError(f"missing snapshot: {root}")
    result: dict[str, Path] = {}
    base = root.resolve()
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(part in {".cache", ".git"} for part in relative.parts):
            continue
        if path.is_symlink():
            resolved = path.resolve()
            if not path.exists() or not path.is_file() or not resolved.is_relative_to(base):
                raise ValueError(f"unsafe snapshot symlink: {path}")
        elif path.is_dir():
            continue
        elif not path.is_file():
            raise ValueError(f"non-regular snapshot entry: {path}")
        name = relative.as_posix()
        if not safe_path(name) or name in result:
            raise ValueError(f"unsafe/duplicate snapshot path: {name}")
        result[name] = path
    return result


def file_packet(name: str, path: Path, lfs: str | None = None, lfs_size: int | None = None) -> dict[str, Any]:
    size = path.stat().st_size
    packet: dict[str, Any] = {
        "type": "file",
        "path": name,
        "size": size,
        "sha256": sha256(path),
        "git_blob_sha1": git_blob_sha1(path),
    }
    if lfs is not None:
        packet.update({"lfs_sha256": lfs, "lfs_size": lfs_size})
    return packet


def server_tree(snapshot: Path, packet_path: Path, blockers: list[str], repository: str = HF_REPOSITORY, revision: str = HF_REVISION, require_public: bool = False) -> dict[str, Any]:
    raw = json.loads(packet_path.read_text(encoding="utf-8"), object_pairs_hook=no_dupes)
    if (
        not isinstance(raw, dict)
        or raw.get("walk") != "recursive_file_only"
        or raw.get("complete_recursive") is not True
    ):
        raise ValueError("server tree must declare complete recursive_file-only evidence")
    rows = raw.get("files")
    if not isinstance(rows, list):
        raise ValueError("server tree files must be a list")
    remote: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or row.get("type") != "file":
            raise ValueError("server tree contains non-file entry")
        name, size, git, lfs = row.get("path"), row.get("size"), row.get("git_blob_sha1"), row.get("lfs_sha256")
        if not safe_path(name) or not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ValueError(f"invalid server row: {row!r}")
        if name in remote or not isinstance(git, str) or not re.fullmatch(r"[0-9a-f]{40}", git):
            raise ValueError(f"duplicate/invalid Git identity: {name}")
        if lfs is not None and (
            not isinstance(lfs, str)
            or not re.fullmatch(r"[0-9a-f]{64}", lfs)
            or row.get("lfs_size") != size
            or lfs_pointer_sha1(lfs, size) != git
        ):
            raise ValueError(f"invalid LFS identity: {name}")
        remote[name] = {"size": size, "git_blob_sha1": git, "lfs_sha256": lfs, "lfs_size": row.get("lfs_size")}

    local = local_files(snapshot)
    missing = sorted(set(remote) - set(local))
    extra = sorted(set(local) - set(remote))
    changed: list[str] = []
    for name in sorted(set(remote) & set(local)):
        expected, path = remote[name], local[name]
        actual_size = path.stat().st_size
        if expected["lfs_sha256"] is not None:
            good = actual_size == expected["lfs_size"] == expected["size"] and sha256(path) == expected["lfs_sha256"]
        else:
            good = actual_size == expected["size"] and git_blob_sha1(path) == expected["git_blob_sha1"]
        if not good:
            changed.append(name)
    if require_public and (PUBLIC_GGUF not in remote or PUBLIC_GGUF not in local):
        blockers.append(f"public Zonos artifact missing: {PUBLIC_GGUF}")
    elif require_public and (local[PUBLIC_GGUF].stat().st_size != PUBLIC_GGUF_BYTES or sha256(local[PUBLIC_GGUF]) != PUBLIC_GGUF_SHA256):
        blockers.append("public Zonos GGUF size/SHA256 mismatch")

    identity = (
        raw.get("repository") == repository
        and raw.get("revision") == revision
        and raw.get("resolved_revision") == revision
    )
    if not identity:
        blockers.append("Zonos HF server-tree identity mismatch")
    if missing or extra:
        blockers.append(f"Zonos server/local tree mismatch: missing={missing!r} extra={extra!r}")
    if changed:
        blockers.append(f"Zonos server/local content mismatch: {changed!r}")
    return {
        "repository": raw.get("repository"),
        "revision": raw.get("revision"),
        "resolved_revision": raw.get("resolved_revision"),
        "walk": raw.get("walk"),
        "files": remote,
        "missing": missing,
        "extra": extra,
        "content_mismatch": changed,
        "status": "MATCHED" if identity and not missing and not extra and not changed else "MISMATCH",
    }


def tensor_manifest(path: Path, blockers: list[str], expected_revision: str = HF_REVISION) -> dict[str, Any]:
    """Inspect a supplied JSON manifest, never infer tensor names/shapes."""
    try:
        packet = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=no_dupes)
        if not isinstance(packet, dict):
            raise ValueError("manifest must be an object")
        tensors = packet.get("tensors")
        digest = packet.get("manifest_sha256")
        if packet.get("revision") != expected_revision:
            raise ValueError(f"fixed revision mismatch: expected {expected_revision}")
        if not isinstance(tensors, list) or len(tensors) != EXPECTED_TENSOR_COUNT:
            raise ValueError("complete 246-entry tensor manifest required")
        normalized: list[dict[str, Any]] = []
        names: set[str] = set()
        for item in tensors:
            if not isinstance(item, dict) or set(item) != {"name", "shape", "dtype"}:
                raise ValueError("manifest tensor schema must be exactly name/shape/dtype")
            name, shape, dtype = item["name"], item["shape"], item["dtype"]
            if (not safe_path(name) or name in names or not isinstance(shape, list)
                    or not shape or any(isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0 for dim in shape)
                    or not isinstance(dtype, str) or not dtype):
                raise ValueError("manifest contains an unsafe or malformed tensor")
            names.add(name)
            normalized.append({"name": name, "shape": shape, "dtype": dtype})
        normalized.sort(key=lambda item: item["name"])
        derived = manifest_sha256(normalized)
        if digest != derived or digest != EXPECTED_MANIFEST_SHA256:
            raise ValueError("fixed 246 manifest hash mismatch")
        return {"status": "MANIFEST_PACKET", "tensor_count": len(normalized), "manifest_sha256": derived, "tensors": normalized}
    except Exception as error:
        blockers.append(f"Zonos tensor manifest blocked: {error}")
        return {"status": "BLOCKED_MANIFEST", "error": str(error)}


def source_inventory(source: Path, blockers: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {"repository": SOURCE_REPOSITORY, "pinned_revision": SOURCE_REVISION}
    try:
        import subprocess
        head = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
        origin = subprocess.run(["git", "-C", str(source), "remote", "get-url", "origin"], check=True, capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(source), "status", "--porcelain", "--untracked-files=all"], check=True, capture_output=True, text=True).stdout
        result.update({"resolved_revision": head, "origin": origin, "clean": not dirty, "roles": [], "tracked_files": []})
        if head != SOURCE_REVISION or origin != SOURCE_REPOSITORY or dirty:
            blockers.append("Zyphra/Zonos source identity/origin/clean mismatch")
        for role in SOURCE_ROLES:
            path = source / role
            if not path.is_file():
                blockers.append(f"Zyphra/Zonos source role missing: {role}")
                continue
            result["roles"].append({"path": role, "sha256": sha256(path), "git_blob_sha1": git_blob_sha1(path)})
        if len(result["roles"]) != len(SOURCE_ROLES):
            blockers.append("Zyphra/Zonos source role inventory is incomplete")
        tracked = subprocess.run(
            ["git", "-C", str(source), "ls-files", "-z"],
            check=True,
            capture_output=True,
        ).stdout.split(b"\0")
        for raw_path in tracked:
            if not raw_path:
                continue
            relative = raw_path.decode("utf-8")
            path = source / relative
            if not safe_path(relative) or not path.is_file() or path.is_symlink():
                blockers.append(f"Zyphra/Zonos tracked source entry is unsafe: {relative}")
                continue
            result["tracked_files"].append({
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "git_blob_sha1": git_blob_sha1(path),
            })
    except Exception as error:
        blockers.append(f"Zyphra/Zonos source inventory blocked: {error}")
        result["error"] = str(error)
    return result


def _evidence_file(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not safe_path(value):
        raise ValueError(f"{label} must be a relative evidence path")
    candidate = (root / value).resolve()
    if not candidate.is_relative_to(root.resolve()):
        raise ValueError(f"{label} escapes evidence root")
    return candidate


def reference_evidence(path: Path | None, blockers: list[str], evidence_root: Path) -> dict[str, Any]:
    """Authenticate the output envelope without treating it as a gate."""
    if path is None:
        blockers.append("official Zonos reference generation is required for validation")
        return {"status": "NOT_RUN"}
    try:
        record = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=no_dupes)
        required = {
            "format", "reference_status", "source_repository", "source_revision",
            "upstream_repository", "upstream_revision",
            "conditioning_packet_sha256", "conditioning_packet_content_digest",
            "codes_path", "codes_shape", "codes_sha256", "pcm_path", "pcm_sha256",
            "pcm_sample_rate", "runtime_status", "publication",
        }
        if not isinstance(record, dict) or not required.issubset(record):
            raise ValueError("reference record is missing required identity/status fields")
        if set(record) != required:
            raise ValueError("reference record has unexpected identity/status fields")
        if (
            record["format"] != "vokra-zonos-reference-v1"
            or record["reference_status"] != "MEASURED_NOT_GATED"
            or record["source_repository"] != SOURCE_REPOSITORY
            or record["source_revision"] != SOURCE_REVISION
            or record["upstream_repository"] != UPSTREAM_HF_REPOSITORY
            or record["upstream_revision"] != UPSTREAM_HF_REVISION
            or record["runtime_status"] != "REFERENCE_ONLY_NO_NATIVE_VERDICT"
            or record["publication"] != "NO_UPLOAD"
            or not re.fullmatch(r"[0-9a-f]{64}", record["conditioning_packet_sha256"])
            or not re.fullmatch(r"[0-9a-f]{64}", record["conditioning_packet_content_digest"])
            or not re.fullmatch(r"[0-9a-f]{64}", record["codes_sha256"])
            or not re.fullmatch(r"[0-9a-f]{64}", record["pcm_sha256"])
            or record["pcm_sample_rate"] != 44_100
            or not isinstance(record["codes_shape"], list)
            or not record["codes_shape"]
            or any(isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0 for dim in record["codes_shape"])
        ):
            raise ValueError("reference record identity/status is not fixed")
        codes = _evidence_file(evidence_root, record["codes_path"], "reference codes path")
        if not codes.is_file() or codes.stat().st_size == 0:
            raise ValueError("reference code output is missing or empty")
        actual = sha256(codes)
        if actual != record["codes_sha256"]:
            raise ValueError("reference code output digest mismatch")
        pcm = _evidence_file(evidence_root, record["pcm_path"], "reference PCM path")
        if not pcm.is_file() or pcm.stat().st_size == 0:
            raise ValueError("reference PCM output is missing or empty")
        if sha256(pcm) != record["pcm_sha256"]:
            raise ValueError("reference PCM output digest mismatch")
        return {"status": "MEASURED_NOT_GATED", **record}
    except Exception as error:
        blockers.append(f"Zonos reference evidence blocked: {error}")
        return {"status": "BLOCKED_REFERENCE", "error": str(error)}


def native_evidence(path: Path | None, blockers: list[str], evidence_root: Path) -> dict[str, Any]:
    if path is None:
        blockers.append("native CPU validation log is required")
        return {"status": "NOT_RUN"}
    try:
        resolved = path.resolve()
        if not resolved.is_relative_to(evidence_root.resolve()) or not resolved.is_file():
            raise ValueError("native CPU log must be an evidence-root file")
        content = resolved.read_text(encoding="utf-8")
        if "ZONOS_CPU_REFERENCE codes=EXACT" not in content:
            raise ValueError("native CPU exact-code marker is missing")
        if "verdict=MEASURED_NOT_GATED" not in content:
            raise ValueError("native CPU PCM marker is missing")
        return {
            "status": "MEASURED_NOT_GATED",
            "path": resolved.name,
            "sha256": sha256(resolved),
            "codes": "EXACT",
            "pcm": "MEASURED_NOT_GATED",
        }
    except Exception as error:
        blockers.append(f"Zonos native CPU evidence blocked: {error}")
        return {"status": "BLOCKED_NATIVE_CPU", "error": str(error)}


def inspect(snapshot: Path | None, packet: Path | None, manifest: Path | None, upstream_snapshot: Path | None, upstream_packet: Path | None, upstream_manifest: Path | None, source: Path | None, output: Path, reference_record: Path | None = None, native_log: Path | None = None) -> int:
    blockers = [
        "PCM numeric bound remains MEASURED_NOT_GATED",
        "Apple Metal exact-code/PCM validation is pending",
    ]
    evidence: dict[str, Any] = {
        **STATUS_FIELDS,
        "format": FORMAT,
        "model": HF_REPOSITORY,
        "fixed_revision": HF_REVISION,
        "upstream_model": {"repository": UPSTREAM_HF_REPOSITORY, "revision": UPSTREAM_HF_REVISION},
        "public_artifact": {
            "path": PUBLIC_GGUF,
            "bytes": PUBLIC_GGUF_BYTES,
            "sha256": PUBLIC_GGUF_SHA256,
            "tensor_count": EXPECTED_TENSOR_COUNT,
            "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        },
    }
    evidence_error = True
    try:
        if snapshot is not None and packet is not None:
            evidence["server_tree"] = server_tree(snapshot, packet, blockers, require_public=True)
            evidence_error = evidence["server_tree"]["status"] != "MATCHED"
        elif snapshot is not None or packet is not None:
            blockers.append("snapshot and server-tree packet must be supplied together")
            evidence_error = True
        else:
            blockers.append("public HF snapshot and server-tree packet are required")
            evidence_error = True
        if manifest is not None:
            evidence["tensor_manifest"] = tensor_manifest(manifest, blockers)
            evidence_error = evidence_error or evidence["tensor_manifest"]["status"] != "MANIFEST_PACKET"
        elif snapshot is not None:
            blockers.append("public 246 tensor manifest packet is required")
            evidence_error = True
        else:
            blockers.append("public 246 tensor manifest packet is required")
            evidence_error = True
        if upstream_manifest is not None:
            evidence["upstream_tensor_manifest"] = tensor_manifest(upstream_manifest, blockers, UPSTREAM_HF_REVISION)
            evidence_error = evidence_error or evidence["upstream_tensor_manifest"]["status"] != "MANIFEST_PACKET"
        else:
            blockers.append("upstream 246 tensor manifest packet is required")
            evidence_error = True
        if upstream_snapshot is not None and upstream_packet is not None:
            evidence["upstream_server_tree"] = server_tree(upstream_snapshot, upstream_packet, blockers, UPSTREAM_HF_REPOSITORY, UPSTREAM_HF_REVISION)
            evidence_error = evidence_error or evidence["upstream_server_tree"]["status"] != "MATCHED"
            for required in ("model.safetensors", "config.json"):
                if required not in evidence["upstream_server_tree"]["files"]:
                    blockers.append(f"upstream required file missing: {required}")
                    evidence_error = True
        else:
            blockers.append("upstream HF snapshot and server-tree packet are required")
            evidence_error = True
        if source is not None:
            evidence["source"] = source_inventory(source, blockers)
            evidence_error = evidence_error or evidence["source"].get("resolved_revision") != SOURCE_REVISION or evidence["source"].get("origin") != SOURCE_REPOSITORY or not evidence["source"].get("clean", False)
        else:
            blockers.append("fixed Zyphra/Zonos source checkout is required")
            evidence_error = True
        evidence["reference"] = reference_evidence(reference_record, blockers, output)
        evidence["native_cpu"] = native_evidence(native_log, blockers, output)
        evidence_error = evidence_error or evidence["reference"]["status"] != "MEASURED_NOT_GATED"
        evidence_error = evidence_error or evidence["native_cpu"]["status"] != "MEASURED_NOT_GATED"
        if evidence["native_cpu"]["status"] == "MEASURED_NOT_GATED":
            evidence.update({
                "runtime_status": "NATIVE_CPU_CODES_VALIDATED",
                "cpu_status": "MEASURED_NOT_GATED",
                "parity_status": "CPU_CODES_EXACT_PCM_MEASURED_NOT_GATED",
            })
        if manifest is not None and upstream_manifest is not None and not evidence_error:
            public = evidence["tensor_manifest"]["tensors"]
            upstream = evidence["upstream_tensor_manifest"]["tensors"]
            normalize = lambda items: sorted((item.get("name"), item.get("shape"), item.get("dtype")) for item in items if isinstance(item, dict))
            if normalize(public) != normalize(upstream):
                blockers.append("public/upstream 246 tensor manifest mismatch")
                evidence_error = True
    except Exception as error:
        blockers.append(f"Zonos evidence collection error: {error}")
        evidence_error = True
    evidence["inspection_status"] = "INSPECTION_ERROR" if evidence_error else "AUTHENTICATED_EVIDENCE_COMPLETE"
    if evidence_error:
        evidence["evidence_stage"] = "INCOMPLETE_REQUIRED_EVIDENCE"
        evidence["runtime_status"] = "SOURCE_STAGED_ONLY"
        evidence["cpu_status"] = "NOT_RUN"
        evidence["parity_status"] = "NOT_RUN"
    evidence["blockers"] = list(dict.fromkeys(blockers))
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if not evidence_error else 2


def self_test() -> None:
    assert not safe_path("../escape")
    assert not safe_path("\\escape")
    assert safe_path("config.json")
    try:
        json.loads('{"x":1,"x":2}', object_pairs_hook=no_dupes)
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate JSON keys must fail")
    with __import__("tempfile").TemporaryDirectory() as directory:
        root = Path(directory)
        empty_out = root / "empty-evidence"
        assert inspect(None, None, None, None, None, None, None, empty_out) == 2
        empty_record = json.loads((empty_out / "manifest.json").read_text(encoding="utf-8"))
        assert empty_record["inspection_status"] == "INSPECTION_ERROR"
        snapshot = root / "snapshot"
        snapshot.mkdir()
        artifact = snapshot / PUBLIC_GGUF
        artifact.write_bytes(b"x")
        packet = root / "tree.json"
        old_bytes, old_sha = PUBLIC_GGUF_BYTES, PUBLIC_GGUF_SHA256
        globals()["PUBLIC_GGUF_BYTES"] = 1
        globals()["PUBLIC_GGUF_SHA256"] = sha256(artifact)
        packet.write_text(json.dumps({"repository": HF_REPOSITORY, "revision": HF_REVISION, "resolved_revision": HF_REVISION, "walk": "recursive_file_only", "complete_recursive": True, "files": [file_packet(PUBLIC_GGUF, artifact)] }), encoding="utf-8")
        blockers: list[str] = []
        result = server_tree(snapshot, packet, blockers, require_public=True)
        assert result["status"] == "MATCHED" and not blockers
        globals()["PUBLIC_GGUF_BYTES"], globals()["PUBLIC_GGUF_SHA256"] = old_bytes, old_sha
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps({"revision": HF_REVISION, "tensors": [{}] * EXPECTED_TENSOR_COUNT, "manifest_sha256": EXPECTED_MANIFEST_SHA256}), encoding="utf-8")
        blockers = []
        assert tensor_manifest(manifest, blockers)["status"] == "BLOCKED_MANIFEST"
        rows = [{"name": f"tensor.{index}", "shape": [1], "dtype": "F32"} for index in range(EXPECTED_TENSOR_COUNT)]
        derived = manifest_sha256(rows)
        old_manifest = EXPECTED_MANIFEST_SHA256
        globals()["EXPECTED_MANIFEST_SHA256"] = derived
        manifest.write_text(json.dumps({"revision": HF_REVISION, "tensors": rows, "manifest_sha256": derived}), encoding="utf-8")
        blockers = []
        assert tensor_manifest(manifest, blockers)["status"] == "MANIFEST_PACKET" and not blockers
        rows[0]["shape"] = [2]
        manifest.write_text(json.dumps({"revision": HF_REVISION, "tensors": rows, "manifest_sha256": derived}), encoding="utf-8")
        blockers = []
        assert tensor_manifest(manifest, blockers)["status"] == "BLOCKED_MANIFEST"
        globals()["EXPECTED_MANIFEST_SHA256"] = old_manifest
        codes = root / "reference-codes.u32le"
        codes.write_bytes((1).to_bytes(4, "little"))
        reference = root / "reference-codes.json"
        reference.write_text(json.dumps({
            "format": "vokra-zonos-reference-v1",
            "reference_status": "MEASURED_NOT_GATED",
            "source_repository": SOURCE_REPOSITORY,
            "source_revision": SOURCE_REVISION,
            "upstream_repository": UPSTREAM_HF_REPOSITORY,
            "upstream_revision": UPSTREAM_HF_REVISION,
            "conditioning_packet_sha256": "0" * 64,
            "conditioning_packet_content_digest": "1" * 64,
            "codes_path": codes.name, "codes_shape": [1, 1, 1],
            "codes_sha256": sha256(codes),
            "pcm_path": "reference-pcm.f32le", "pcm_sha256": "",
            "pcm_sample_rate": 44_100,
            "runtime_status": "REFERENCE_ONLY_NO_NATIVE_VERDICT",
            "publication": "NO_UPLOAD",
        }), encoding="utf-8")
        pcm = root / "reference-pcm.f32le"
        pcm.write_bytes(struct.pack("<f", 0.0))
        reference.write_text(reference.read_text(encoding="utf-8").replace(
            '"pcm_sha256": "",', f'"pcm_sha256": "{sha256(pcm)}",',
        ), encoding="utf-8")
        blockers = []
        assert reference_evidence(reference, blockers, root)["status"] == "MEASURED_NOT_GATED"
        record = json.loads(reference.read_text(encoding="utf-8"))
        record["source_repository"] = "https://example.invalid/not-zonos"
        reference.write_text(json.dumps(record), encoding="utf-8")
        blockers = []
        assert reference_evidence(reference, blockers, root)["status"] == "BLOCKED_REFERENCE"
        record["source_repository"] = SOURCE_REPOSITORY
        record.pop("pcm_path")
        reference.write_text(json.dumps(record), encoding="utf-8")
        blockers = []
        assert reference_evidence(reference, blockers, root)["status"] == "BLOCKED_REFERENCE"
        record["pcm_path"] = pcm.name
        reference.write_text(json.dumps(record), encoding="utf-8")
        codes.write_bytes((2).to_bytes(4, "little"))
        blockers = []
        assert reference_evidence(reference, blockers, root)["status"] == "BLOCKED_REFERENCE"
        native = root / "native-cpu.log"
        native.write_text(
            "ZONOS_CPU_REFERENCE codes=EXACT pcm_max_abs=0 verdict=MEASURED_NOT_GATED\n",
            encoding="utf-8",
        )
        blockers = []
        assert native_evidence(native, blockers, root)["status"] == "MEASURED_NOT_GATED"
        reference.write_text(reference.read_text(encoding="utf-8").replace(
            f'"codes_path": "{codes.name}"', '"codes_path": "../reference-codes.u32le"',
        ), encoding="utf-8")
        blockers = []
        assert reference_evidence(reference, blockers, root)["status"] == "BLOCKED_REFERENCE"
        # A mismatch in otherwise well-formed public/upstream manifests is an
        # inspection error, not a runtime-only blocker.  Stub the evidence
        # readers so this status transition is tested without model files.
        originals = {name: globals()[name] for name in (
            "server_tree", "tensor_manifest", "source_inventory",
            "reference_evidence", "native_evidence",
        )}
        globals()["server_tree"] = lambda *args, **kwargs: {
            "status": "MATCHED", "files": {"model.safetensors": {}, "config.json": {}},
        }
        globals()["tensor_manifest"] = lambda path, blockers, *args: {
            "status": "MANIFEST_PACKET",
            "tensors": [{"name": "tensor", "shape": [2 if args else 1], "dtype": "F32"}],
        }
        globals()["source_inventory"] = lambda path, blockers: {
            "resolved_revision": SOURCE_REVISION, "origin": SOURCE_REPOSITORY, "clean": True,
        }
        globals()["reference_evidence"] = lambda path, blockers, evidence_root: {"status": "MEASURED_NOT_GATED"}
        globals()["native_evidence"] = lambda path, blockers, evidence_root: {"status": "MEASURED_NOT_GATED"}
        try:
            mismatch_output = root / "mismatch-output"
            assert inspect(
                root, root / "packet", root / "manifest", root, root / "upstream-packet",
                root / "upstream-manifest", root, mismatch_output, root / "reference", root / "native",
            ) == 2
            mismatch_manifest = json.loads((mismatch_output / "manifest.json").read_text(encoding="utf-8"))
            assert mismatch_manifest["inspection_status"] == "INSPECTION_ERROR"
            assert "public/upstream 246 tensor manifest mismatch" in mismatch_manifest["blockers"]
        finally:
            globals().update(originals)
    print("zonos_inspect.py self-test: OK")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--server-tree", type=Path)
    parser.add_argument("--tensor-manifest", type=Path)
    parser.add_argument("--upstream-tensor-manifest", type=Path)
    parser.add_argument("--upstream-snapshot", type=Path)
    parser.add_argument("--upstream-server-tree", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--reference-record", type=Path)
    parser.add_argument("--native-log", type=Path)
    parser.add_argument("--output", type=Path, default=Path("zonos-evidence"))
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    return inspect(args.snapshot, args.server_tree, args.tensor_manifest, args.upstream_snapshot, args.upstream_server_tree, args.upstream_tensor_manifest, args.source, args.output, args.reference_record, args.native_log)


if __name__ == "__main__":
    raise SystemExit(main())
