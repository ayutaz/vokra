#!/usr/bin/env -S uv run --frozen --project tools/parity --python 3.12 python
"""Fail-closed inspection of the complete VoxCPM-0.5B release.

This is an evidence collector, not a converter or a reference implementation.
Large files are intentionally handled only on VAST.  The historical Vokra
GGUF is checked as a legacy, main-only artifact; it is never treated as the
AudioVAE/tokenizer-complete replacement required by the official loader.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any

HF_REPOSITORY = "openbmb/VoxCPM-0.5B"
HF_REVISION = "e95e62437bb940c8aeb9f26dc3169d436d2bb455"
SOURCE_REPOSITORY = "https://github.com/OpenBMB/VoxCPM.git"
SOURCE_REVISION = "38a76704ee67935ccbafbe5b6725e83dbb1e9305"
PUBLIC_REPOSITORY = "vokra/voxcpm-0.5b"
PUBLIC_REVISION = "ee0ca6d5b9fab27bbb626b5cb3f01236e582d004"
PUBLIC_FILE = "model.gguf"
PUBLIC_BYTES = 1_304_607_744
PUBLIC_SHA256 = "2c5c3b2509368db3545ea44e66ddd3ef5050ceacd5b5a431d8d8acf1300c6cce"
PUBLIC_TENSOR_COUNT = 377
PUBLIC_MANIFEST_SHA256 = "d364689d5593ed8886029907a5d17e7659b94f7f310fe95b133c545b6901c509"
AUDIOVAE_BYTES = 301_494_192
MAX_HEADER_BYTES = 64 * 1024 * 1024
AUDIOVAE_SOURCE = "src/voxcpm/modules/audiovae/audio_vae.py"
AUDIOVAE_CONTRACT = {
    "sample_rate_hz": 16_000,
    "encoder_dim": 128,
    "encoder_rates": [2, 5, 8, 8],
    "latent_dim": 64,
    "decoder_dim": 1536,
    "decoder_rates": [8, 8, 5, 2],
    "depthwise": True,
    "noise_block": False,
}
REQUIRED = {
    ".gitattributes", "README.md", "config.json", "pytorch_model.bin",
    "audiovae.pth", "special_tokens_map.json", "tokenizer.json",
    "tokenizer_config.json",
}
SOURCE_ROLES = (
    "src/voxcpm/model/voxcpm.py",
    "src/voxcpm/modules/audiovae/audio_vae.py",
    "src/voxcpm/modules/layers/scalar_quantization_layer.py",
    "src/voxcpm/modules/locdit/local_dit.py",
    "src/voxcpm/modules/locdit/unified_cfm.py",
    "src/voxcpm/modules/locenc/local_encoder.py",
    "src/voxcpm/modules/minicpm4/cache.py",
    "src/voxcpm/modules/minicpm4/config.py",
    "src/voxcpm/modules/minicpm4/model.py",
)


def strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RuntimeError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=strict_pairs)


def safe_path(value: str) -> None:
    path = Path(value)
    if not value or "\0" in value or "\\" in value or path.is_absolute() or ".." in path.parts:
        raise RuntimeError(f"unsafe path: {value!r}")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git_blob(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def lfs_pointer_blob(size: int, sha: str) -> str:
    pointer = f"version https://git-lfs.github.com/spec/v1\noid sha256:{sha}\nsize {size}\n".encode()
    return hashlib.sha1(f"blob {len(pointer)}\0".encode() + pointer).hexdigest()


def check_config(path: Path) -> dict[str, Any]:
    config = load_json(path)
    if not isinstance(config, dict):
        raise RuntimeError("config.json must be an object")
    expected = {
        "architecture": "voxcpm", "dtype": "bfloat16", "patch_size": 2,
        "feat_dim": 64, "scalar_quantization_latent_dim": 256,
        "scalar_quantization_scale": 9, "residual_lm_num_layers": 6,
        "max_length": 4096,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise RuntimeError(f"config.{key} mismatch: {config.get(key)!r}")
    lm = config.get("lm_config")
    if not isinstance(lm, dict):
        raise RuntimeError("config.lm_config is missing")
    for key, value in {
        "hidden_size": 1024, "intermediate_size": 4096, "num_hidden_layers": 24,
        "num_attention_heads": 16, "num_key_value_heads": 2,
        "max_position_embeddings": 32768, "vocab_size": 73448,
        "rms_norm_eps": 1e-5, "rope_theta": 10000,
    }.items():
        if lm.get(key) != value:
            raise RuntimeError(f"config.lm_config.{key} mismatch")
    for block, values in {
        "encoder_config": {"hidden_size": 1024, "intermediate_size": 4096, "num_heads": 16, "num_layers": 4},
        "dit_config": {"hidden_size": 1024, "intermediate_size": 4096, "num_heads": 16, "num_layers": 4},
    }.items():
        obj = config.get(block)
        if not isinstance(obj, dict):
            raise RuntimeError(f"config.{block} is missing")
        for key, value in values.items():
            if obj.get(key) != value:
                raise RuntimeError(f"config.{block}.{key} mismatch")
    return config


def check_tokenizer(snapshot: Path) -> dict[str, Any]:
    config = load_json(snapshot / "tokenizer_config.json")
    tokens = load_json(snapshot / "special_tokens_map.json")
    if not isinstance(config, dict) or config.get("add_bos_token") is not True or config.get("add_eos_token") is not False:
        raise RuntimeError("tokenizer_config BOS/EOS contract mismatch")
    if not isinstance(tokens, dict) or not tokens.get("bos_token") or not tokens.get("eos_token"):
        raise RuntimeError("special token map is incomplete")
    tokenizer = load_json(snapshot / "tokenizer.json")
    if not isinstance(tokenizer, dict) or tokenizer.get("version") != "1.0":
        raise RuntimeError("tokenizer.json version mismatch")
    return {"config": config, "special_tokens": tokens, "json_sha256": digest(snapshot / "tokenizer.json")}


def check_model_card(snapshot: Path) -> dict[str, Any]:
    text = (snapshot / "README.md").read_text(encoding="utf-8")
    match = re.search(r"^license:\s*([^\s]+)\s*$", text, re.MULTILINE)
    if match is None or match.group(1).lower() != "apache-2.0":
        raise RuntimeError("HF model-card license is not the authenticated Apache-2.0 declaration")
    return {"license": match.group(1), "sha256": digest(snapshot / "README.md")}


def inspect_tree(snapshot: Path, packet: Path) -> dict[str, Any]:
    envelope = load_json(packet)
    if not isinstance(envelope, dict) or envelope.get("repository") != HF_REPOSITORY or envelope.get("revision") != HF_REVISION or envelope.get("resolved_revision") != HF_REVISION:
        raise RuntimeError("HF server identity mismatch")
    rows = envelope.get("files")
    if not isinstance(rows, list):
        raise RuntimeError("server tree files missing")
    by_path: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"path", "type", "size", "git_blob_sha1", "lfs_sha256", "lfs_size"}:
            raise RuntimeError("malformed server tree row")
        path = row["path"]
        safe_path(path)
        if row["type"] != "file" or path in by_path or not isinstance(row["size"], int) or isinstance(row["size"], bool) or row["size"] < 0:
            raise RuntimeError(f"invalid tree row {path!r}")
        if not re.fullmatch(r"[0-9a-f]{40}", str(row["git_blob_sha1"])):
            raise RuntimeError(f"invalid Git identity {path}")
        lfs = row["lfs_sha256"]
        if lfs is not None and not re.fullmatch(r"[0-9a-f]{64}", str(lfs)):
            raise RuntimeError(f"invalid LFS identity {path}")
        if lfs is not None and row["lfs_size"] != row["size"]:
            raise RuntimeError(f"LFS size mismatch {path}")
        by_path[path] = row
    if not REQUIRED.issubset(by_path):
        raise RuntimeError(f"required HF files missing: {sorted(REQUIRED - set(by_path))}")
    actual: set[str] = set()
    for path in snapshot.rglob("*"):
        relative = path.relative_to(snapshot)
        if ".cache" in relative.parts:
            continue
        if path.is_symlink() or (not path.is_file() and not path.is_dir()):
            raise RuntimeError(f"non-regular local member: {relative}")
        if path.is_file():
            actual.add(relative.as_posix())
    expected = set(by_path)
    if actual != expected:
        raise RuntimeError(f"local/server tree mismatch: missing={expected-actual}, extra={actual-expected}")
    records = []
    for name, row in sorted(by_path.items()):
        local = snapshot / name
        if local.stat().st_size != row["size"]:
            raise RuntimeError(f"size mismatch: {name}")
        actual_sha = digest(local)
        if row["lfs_sha256"] is not None:
            if actual_sha != row["lfs_sha256"]:
                raise RuntimeError(f"LFS payload mismatch: {name}")
            if lfs_pointer_blob(row["size"], actual_sha) != row["git_blob_sha1"]:
                raise RuntimeError(f"LFS pointer mismatch: {name}")
        elif git_blob(local) != row["git_blob_sha1"]:
            raise RuntimeError(f"Git blob mismatch: {name}")
        records.append({"path": name, "size": row["size"], "sha256": actual_sha, "lfs": row["lfs_sha256"] is not None})
    if (snapshot / "audiovae.pth").stat().st_size != AUDIOVAE_BYTES:
        raise RuntimeError("AudioVAE fixed payload size mismatch")
    return {"repository": HF_REPOSITORY, "revision": HF_REVISION, "files": records}


def tensor_manifest(path: Path, label: str) -> dict[str, Any]:
    try:
        import torch  # type: ignore
    except ImportError as error:
        raise RuntimeError("torch is required only on VAST for checkpoint inspection") from error
    unsafe = getattr(torch.serialization, "get_unsafe_globals_in_checkpoint", None)
    if unsafe is not None and unsafe(path):
        raise RuntimeError(f"{label} contains unsafe pickle globals")
    value = torch.load(path, map_location="cpu", weights_only=True)
    rows: list[dict[str, Any]] = []
    def walk(item: Any, name: str, depth: int = 0) -> None:
        if depth > 32 or len(rows) > 200_000:
            raise RuntimeError(f"{label} manifest bound exceeded")
        if isinstance(item, torch.Tensor):
            if not item.is_floating_point() and not item.is_complex() and item.dtype != torch.bool:
                raise RuntimeError(f"{label} non-floating tensor {name}")
            if item.is_floating_point() and not bool(torch.isfinite(item).all()):
                raise RuntimeError(f"{label} non-finite tensor {name}")
            rows.append({"name": name, "shape": [int(x) for x in item.shape], "dtype": str(item.dtype), "elements": int(item.numel())})
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                walk(item[key], f"{name}.{key}" if name else str(key), depth + 1)
        elif isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                walk(child, f"{name}[{index}]", depth + 1)
    walk(value, "")
    if not rows:
        raise RuntimeError(f"{label} has empty tensor manifest")
    rows.sort(key=lambda row: row["name"])
    canonical = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return {"label": label, "tensor_count": len(rows), "elements": sum(r["elements"] for r in rows), "manifest_sha256": hashlib.sha256(canonical).hexdigest(), "tensors": rows}


def inspect_public_gguf(path: Path) -> dict[str, Any]:
    if path.stat().st_size != PUBLIC_BYTES or digest(path) != PUBLIC_SHA256:
        raise RuntimeError("historical public GGUF identity mismatch")
    with path.open("rb") as stream:
        data = stream.read(MAX_HEADER_BYTES)
    cursor = 0
    def take(size: int) -> bytes:
        nonlocal cursor
        if size < 0 or cursor + size > len(data):
            raise RuntimeError("GGUF header truncated")
        result = data[cursor:cursor + size]; cursor += size; return result
    def u32() -> int: return struct.unpack("<I", take(4))[0]
    def u64() -> int: return struct.unpack("<Q", take(8))[0]
    def string() -> str:
        size = u64()
        if size > MAX_HEADER_BYTES: raise RuntimeError("GGUF string exceeds bound")
        return take(size).decode("utf-8")
    widths = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    def skip(kind: int) -> None:
        if kind in widths: take(widths[kind])
        elif kind == 8: string()
        elif kind == 9:
            element = u32(); count = u64()
            if count > 1_000_000: raise RuntimeError("GGUF array bound exceeded")
            for _ in range(count): skip(element)
        else: raise RuntimeError(f"unknown GGUF metadata type {kind}")
    if take(4) != b"GGUF" or u32() not in (2, 3): raise RuntimeError("invalid GGUF header")
    count, metadata = u64(), u64()
    if count != PUBLIC_TENSOR_COUNT or metadata > 1_000_000: raise RuntimeError("historical GGUF count mismatch")
    for _ in range(metadata): string(); skip(u32())
    descriptors = []
    for _ in range(count):
        name = string(); rank = u32()
        if rank > 8 or "\0" in name or "\\" in name or Path(name).is_absolute() or ".." in Path(name).parts: raise RuntimeError("unsafe GGUF tensor")
        shape = [u64() for _ in range(rank)]; dtype = u32(); offset = u64()
        descriptors.append({"name": name, "shape": shape, "dtype": dtype, "offset": offset})
    canonical = b"".join(d["name"].encode() + b"\0" + struct.pack("<Q", len(d["shape"])) + b"".join(struct.pack("<Q", x) for x in d["shape"]) for d in sorted(descriptors, key=lambda x: x["name"]))
    manifest = hashlib.sha256(canonical).hexdigest()
    if manifest != PUBLIC_MANIFEST_SHA256:
        raise RuntimeError("historical GGUF tensor manifest mismatch")
    return {"repository": PUBLIC_REPOSITORY, "revision": PUBLIC_REVISION, "bytes": path.stat().st_size, "tensor_count": count, "manifest_sha256": manifest, "status": "legacy_main_only"}


def inspect_source(source: Path) -> dict[str, Any]:
    head = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if head != SOURCE_REVISION:
        raise RuntimeError("official source HEAD mismatch")
    origin = subprocess.check_output(["git", "-C", str(source), "remote", "get-url", "origin"], text=True).strip().removesuffix(".git")
    if origin != SOURCE_REPOSITORY.removesuffix(".git"):
        raise RuntimeError("official source origin mismatch")
    if subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True):
        raise RuntimeError("official source checkout is dirty")
    missing = [role for role in SOURCE_ROLES if not (source / role).is_file()]
    if missing: raise RuntimeError(f"official source roles missing: {missing}")
    license_path = source / "LICENSE"
    if not license_path.is_file() or "apache" not in license_path.read_text(encoding="utf-8").lower():
        raise RuntimeError("official source LICENSE is missing or not Apache")
    return {"repository": SOURCE_REPOSITORY, "revision": SOURCE_REVISION, "license_sha256": digest(license_path), "roles": {role: digest(source / role) for role in SOURCE_ROLES}}


def inspect(snapshot: Path, packet: Path, evidence: Path, source: Path | None = None, public: Path | None = None) -> None:
    if source is None or public is None:
        raise RuntimeError("AUTHENTICATED_EVIDENCE_COMPLETE requires official source and historical public GGUF")
    tree = inspect_tree(snapshot, packet)
    config = check_config(snapshot / "config.json")
    tokenizer = check_tokenizer(snapshot)
    result: dict[str, Any] = {"status": "BLOCKED", "evidence_stage": "INSPECTION_ONLY", "inspection_status": "AUTHENTICATED_EVIDENCE_COMPLETE", "runtime_status": "NOT_IMPLEMENTED_FAIL_CLOSED", "cpu_status": "UNSUPPORTED", "metal_status": "BLOCKED_BY_CPU", "parity_status": "NOT_RUN", "publication": "NO_UPLOAD", "hf_tree": tree, "model_card": check_model_card(snapshot), "config": config, "tokenizer": tokenizer, "composite": "main+pytorch AudioVAE+tokenizer required; historical public GGUF is main-only"}
    result["main_checkpoint"] = tensor_manifest(snapshot / "pytorch_model.bin", "pytorch_model.bin")
    result["audiovae_checkpoint"] = tensor_manifest(snapshot / "audiovae.pth", "audiovae.pth")
    result["source"] = inspect_source(source)
    result["audio_vae_contract"] = {"source_role": AUDIOVAE_SOURCE, **AUDIOVAE_CONTRACT}
    result["historical_public_gguf"] = inspect_public_gguf(public)
    evidence.mkdir(parents=True, exist_ok=True)
    (evidence / "manifest.json").write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def self_test() -> None:
    assert strict_pairs([("x", 1)]) == {"x": 1}
    try: strict_pairs([("x", 1), ("x", 2)])
    except RuntimeError: pass
    else: raise AssertionError("duplicate JSON keys must fail")
    for bad in ("../x", "/x", "a\\b", ""):
        try: safe_path(bad)
        except RuntimeError: pass
        else: raise AssertionError("unsafe path accepted")
    assert lfs_pointer_blob(3, "0" * 64).isalnum()
    assert HF_REVISION == "e95e62437bb940c8aeb9f26dc3169d436d2bb455"
    assert SOURCE_REVISION == "38a76704ee67935ccbafbe5b6725e83dbb1e9305"
    assert AUDIOVAE_SOURCE in SOURCE_ROLES
    assert "src/voxcpm/modules/audiovae/audio_vae_v2.py" not in SOURCE_ROLES
    assert AUDIOVAE_CONTRACT["sample_rate_hz"] == 16_000
    assert AUDIOVAE_CONTRACT["encoder_rates"] == [2, 5, 8, 8]
    assert AUDIOVAE_CONTRACT["decoder_rates"] == [8, 8, 5, 2]
    assert PUBLIC_MANIFEST_SHA256 == "d364689d5593ed8886029907a5d17e7659b94f7f310fe95b133c545b6901c509"
    try:
        inspect(Path("/missing-snapshot"), Path("/missing-tree"), Path("/tmp/voxcpm-self-test"))
    except RuntimeError as error:
        assert "source and historical public GGUF" in str(error)
    else:
        raise AssertionError("inspection without source/public inputs must fail closed")
    print("voxcpm_0_5b_inspect --self-test: OK")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--server-tree", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--public-gguf", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.self_test:
        self_test(); return 0
    if not all((args.snapshot, args.server_tree, args.output)):
        parser.error("--snapshot, --server-tree and --output are required")
    try:
        inspect(args.snapshot, args.server_tree, args.output, args.source, args.public_gguf)
    except Exception as error:  # evidence failures are blockers, never PASS
        args.output.mkdir(parents=True, exist_ok=True)
        blocked = {"status": "BLOCKED", "evidence_stage": "INSPECTION_ONLY", "inspection_status": "INSPECTION_ERROR", "runtime_status": "NOT_IMPLEMENTED_FAIL_CLOSED", "cpu_status": "UNSUPPORTED", "metal_status": "BLOCKED_BY_CPU", "parity_status": "NOT_RUN", "publication": "NO_UPLOAD", "hf_repository": HF_REPOSITORY, "hf_revision": HF_REVISION, "source_repository": SOURCE_REPOSITORY, "source_revision": SOURCE_REVISION, "error": f"{type(error).__name__}: {error}"}
        (args.output / "manifest.json").write_text(json.dumps(blocked, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        print(f"voxcpm_0_5b_inspect: BLOCKED: {error}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
