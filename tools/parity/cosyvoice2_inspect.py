#!/usr/bin/env -S uv run --frozen --project tools/parity --python 3.12 python
"""Fail-closed, no-conversion evidence collector for CosyVoice2.

The upstream release is a composite TTS package.  This inspector records
identity and safe structural evidence only; it never emits a GGUF or executes
ONNX/custom model code.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

REPOSITORY = "FunAudioLLM/CosyVoice2-0.5B"
REVISION = "eec1ae6c79877dbd9379285cf8789c9e0879293d"
SOURCE_REPOSITORY = "https://github.com/FunAudioLLM/CosyVoice.git"
SOURCE_REVISION = "8555549e882236e6541748b1042d95693caa82ba"
MATCHA_REPOSITORY = "https://github.com/shivammehta25/Matcha-TTS.git"
MATCHA_REVISION = "dd9105b34bf2be2230f4aa1e4769fb586a3c824e"
TOTAL_BYTES = 4_856_505_002
EXPECTED = {
    ".gitattributes": (1914, "9f062ecd0b229ee53aac14bef087f154ef15caf1", None),
    "CosyVoice-BlankEN/config.json": (659, "463b055262b6c66c4629a74a4b300bfe2ed31d3c", None),
    "CosyVoice-BlankEN/generation_config.json": (242, "dfc11073787daf1b0f9c0f1499487ab5f4c93738", None),
    "CosyVoice-BlankEN/merges.txt": (1402109, "90d3d82d027eadcc6a5e77c38eb82d43fc51b53b", None),
    "CosyVoice-BlankEN/model.safetensors": (988097824, "3dff8ababe3dbf3bd7a556f5f143503ab2ef3c98", "130282af0dfa9fe5840737cc49a0d339d06075f83c5a315c3372c9a0740d0b96"),
    "CosyVoice-BlankEN/tokenizer_config.json": (1287, "ff55d7b9eb1384e5d4d7e75dc0f564c1a8833d6e", None),
    "CosyVoice-BlankEN/vocab.json": (2776833, "4783fe10ac3adce15ac8f358ef5462739852c569", None),
    "README.md": (12073, "797ae0f468d2e7b0546449982f4a446e7b236ced", None),
    "asset/dingding.png": (122824, "e407a9d3c0fc5a7fcac46aef09181a0bef330d37", "7f04815e2e676d31b089af6fa270135f3214f2193d5e0ad98b491d007d48f1c6"),
    "campplus.onnx": (28303423, "7b08523b2e28e437cfb1a0312723a5ab0bac287e", "a6ac6a63997761ae2997373e2ee1c47040854b4b759ea41ec48e4e42df0f4d73"),
    "config.json": (2, "9e26dfeeb6e641a33dae4961196235bdb965b21b", None),
    "configuration.json": (47, "5e812fae901c12933ac69ebf3eb79d0eb49bbab4", None),
    "cosyvoice2.yaml": (7330, "bc19267bbfd373c9a760b7667a74349ddd487db1", None),
    "flow.decoder.estimator.fp32.onnx": (286317026, "0bf91727ff5df059b971c025dc51b5cd1c3425c3", "cd54e4281701e6630730da64502d77b7e8b6e5c057cca65128bffb50f85cbf98"),
    "flow.pt": (450575567, "3d62976fa383bd42f02229c60069b6435e3552e7", "ff4c2f867674411e0a08cee702996df13fa67c1cd864c06108da88d16d088541"),
    "hift.pt": (83390254, "a2b2934ccff5c50637a026f4efdb29100810bb6f", "3386cc880324d4e98e05987b99107f49e40ed925b8ecc87c1f4939432d429879"),
    "llm.pt": (2023316821, "b8f93347f92a2ce505db9286dd8e72599847c2b1", "b144ef55b51ce8cfb79a73c90dbba0bdaba4e451c0ebcfab20f769264f84a608"),
    "speech_tokenizer_v2.batch.onnx": (496095794, "f300783904a266d10e36ba8be6d59310d43733da", "5b45a98572ed21e3a3ebf50201f3020567f7db40e9a57509b790b2982f5c07b7"),
    "speech_tokenizer_v2.onnx": (496082973, "8a28ef11e5ef40653382921d8a1406727aad3370", "d43342aa12163a80bf07bffb94c9de2e120a8df2f9917cd2f642e7f4219c6f71"),
}
EXPECTED_PATHS = set(EXPECTED)
HISTORICAL = {"repository": "vokra/cosyvoice2-0.5b", "revision": "d707e0277e2a29e8fbd3972aec1b6a1cba0192ea", "filename": "cosyvoice2-0.5b.gguf", "bytes": 2_567_779_936, "git_blob_sha1": "d8552b1b83e5e1578c2b01c8864103f8aa15d321", "lfs_sha256": "bf4d5eb7d4be00118be4fa3c2605957e3699185d9dd1159a8710e6e8dd07c4c4", "tensor_count": 295}
FORMAT = "vokra-cosyvoice2-inspection-v1"
IGNORE = {".cache", ".git"}
MAX_HEADER = 64 * 1024 * 1024
MAX_ARCHIVE_MEMBERS = 100_000
MAX_ARCHIVE_BYTES = 8_000_000_000
MAX_DEPTH = 64
MAX_ITEMS = 300_000


def digest(path: Path, algorithm: str = "sha256") -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_blob(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def lfs_pointer_blob(size: int, lfs_sha256: str) -> str:
    pointer = (
        "version https://git-lfs.github.com/spec/v1\n"
        f"oid sha256:{lfs_sha256}\n"
        f"size {size}\n"
    ).encode()
    return hashlib.sha1(f"blob {len(pointer)}\0".encode() + pointer).hexdigest()


def duplicate_pairs(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate JSON key: {key}")
        out[key] = value
    return out


def snapshot_files(root: Path) -> list[Path]:
    if not root.is_dir():
        raise RuntimeError(f"missing snapshot: {root}")
    base = root.resolve()
    out = []
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if any(part in IGNORE for part in rel.parts):
            continue
        if path.is_dir() and not path.is_symlink():
            continue
        if path.is_symlink() and (not path.exists() or not path.is_file()):
            raise RuntimeError(f"dangling/nonregular symlink: {rel}")
        if not path.is_file():
            raise RuntimeError(f"nonregular snapshot entry: {rel}")
        resolved = path.resolve()
        if resolved != base and base not in resolved.parents:
            raise RuntimeError(f"snapshot symlink escapes root: {rel}")
        out.append(path)
    if not out:
        raise RuntimeError("empty snapshot")
    return out


def json_file(path: Path, blockers: list[str]) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=duplicate_pairs)
    except Exception as exc:
        blockers.append(f"JSON blocked ({path.name}): {exc}")
        return None


def server_tree(snapshot: Path, packet: Path, blockers: list[str]) -> dict[str, Any]:
    remote = json.loads(packet.read_text(encoding="utf-8"), object_pairs_hook=duplicate_pairs)
    rows = remote.get("files")
    records: dict[str, dict[str, Any]] = {}
    invalid_rows = False
    if not isinstance(rows, list):
        blockers.append("server tree files must be an array")
        rows = []
    for row in rows:
        if not isinstance(row, dict) or row.get("type") != "file":
            blockers.append("server tree must contain file-only rows")
            invalid_rows = True
            continue
        name, size, blob, lfs, lfs_size = row.get("path"), row.get("size"), row.get("git_blob_sha1"), row.get("lfs_sha256"), row.get("lfs_size")
        if not isinstance(name, str) or not name or name.startswith("/") or "\\" in name or ".." in Path(name).parts:
            blockers.append(f"unsafe server path: {name!r}")
            invalid_rows = True
            continue
        if not isinstance(size, int) or isinstance(size, bool) or size < 0 or not re.fullmatch(r"[0-9a-f]{40}", str(blob)) or (lfs is not None and (not re.fullmatch(r"[0-9a-f]{64}", str(lfs)) or lfs_size != size)):
            blockers.append(f"invalid server identity: {name}")
            invalid_rows = True
            continue
        if lfs is not None and lfs_pointer_blob(size, lfs) != blob:
            blockers.append(f"invalid LFS pointer identity: {name}")
            invalid_rows = True
        if name in records:
            blockers.append(f"duplicate server path: {name}")
        records[name] = {"bytes": size, "git_blob_sha1": blob, "lfs_sha256": lfs, "lfs_size": lfs_size}
    local = {p.relative_to(snapshot).as_posix(): p for p in snapshot_files(snapshot)}
    missing, extra, changed = sorted(set(records) - set(local)), sorted(set(local) - set(records)), []
    for name in sorted(set(records) & set(local)):
        row, path = records[name], local[name]
        if path.stat().st_size != row["bytes"] or (row["lfs_sha256"] and (row["lfs_size"] != path.stat().st_size or digest(path) != row["lfs_sha256"])) or (not row["lfs_sha256"] and git_blob(path) != row["git_blob_sha1"]):
            changed.append(name)
    if missing or extra or changed:
        blockers.append(f"server/local tree mismatch: missing={missing!r} extra={extra!r} changed={changed!r}")
    identity_ok = remote.get("repository") == REPOSITORY and remote.get("revision") == REVISION and remote.get("resolved_revision") == REVISION and remote.get("walk") == "recursive_file_only"
    if not identity_ok:
        blockers.append("server tree identity/walk mismatch")
    return {"status": "MATCHED" if identity_ok and not invalid_rows and not missing and not extra and not changed else "MISMATCH", "repository": remote.get("repository"), "revision": remote.get("revision"), "resolved_revision": remote.get("resolved_revision"), "walk": remote.get("walk"), "files": records, "missing": missing, "extra": extra, "content_mismatch": changed}


def safe_safetensors(path: Path, blockers: list[str]) -> dict[str, Any]:
    result = {"path": path.name, "bytes": path.stat().st_size, "sha256": digest(path), "resident": "HEADER_ONLY"}
    try:
        size = path.stat().st_size
        with path.open("rb") as stream:
            raw = stream.read(8)
            if len(raw) != 8:
                raise ValueError("missing header length")
            header_len = int.from_bytes(raw, "little")
            if header_len <= 2 or header_len > MAX_HEADER or header_len > size - 8:
                raise ValueError(f"invalid header length {header_len}")
            header = json.loads(stream.read(header_len), object_pairs_hook=duplicate_pairs)
        if not isinstance(header, dict):
            raise ValueError("header is not an object")
        data_end = size - 8 - header_len
        widths = {
            "F64": 8, "F32": 4, "F16": 2, "BF16": 2,
            "I64": 8, "U64": 8, "I32": 4, "U32": 4,
            "I16": 2, "U16": 2, "I8": 1, "U8": 1,
        }
        descriptors = []
        for name, desc in header.items():
            if name == "__metadata__":
                if not isinstance(desc, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in desc.items()):
                    raise ValueError("metadata must be a string map")
                continue
            if not isinstance(name, str) or not name or "\0" in name or "\\" in name or name.startswith("/") or ".." in Path(name).parts or not isinstance(desc, dict) or set(desc) != {"dtype", "shape", "data_offsets"}:
                raise ValueError(f"unsafe or malformed tensor descriptor: {name!r}")
            shape, offsets, dtype = desc["shape"], desc["data_offsets"], desc["dtype"]
            if dtype not in widths or not isinstance(shape, list) or any(not isinstance(x, int) or isinstance(x, bool) or x < 0 for x in shape) or not isinstance(offsets, list) or len(offsets) != 2 or any(not isinstance(x, int) or isinstance(x, bool) or x < 0 for x in offsets) or offsets[0] > offsets[1] or offsets[1] > data_end:
                raise ValueError(f"malformed tensor range: {name}")
            numel = 1
            for dimension in shape:
                numel *= dimension
                if numel > data_end + 1:
                    raise ValueError(f"tensor shape product overflow: {name}")
            expected_bytes = numel * widths[dtype]
            if expected_bytes != offsets[1] - offsets[0]:
                raise ValueError(f"dtype/shape/data range mismatch: {name}")
            descriptors.append((offsets[0], offsets[1], name, shape, dtype))
        descriptors.sort()
        cursor = 0
        for start, end, *_ in descriptors:
            if start != cursor:
                raise ValueError("tensor data has a gap or overlap")
            cursor = end
        if cursor != data_end:
            raise ValueError("tensor data does not end at file boundary")
        result["tensor_count"] = len(descriptors)
        result["tensors"] = [{"name": n, "shape": s, "dtype": d, "offsets": [a, b]} for a, b, n, s, d in descriptors]
    except Exception as exc:
        blockers.append(f"safetensors header blocked: {exc}")
        result["status"] = "BLOCKED"
    return result


def safe_pt(path: Path, blockers: list[str]) -> dict[str, Any]:
    result = {"path": path.name, "bytes": path.stat().st_size, "sha256": digest(path), "resident": "BOUNDED"}
    try:
        with zipfile.ZipFile(path) as archive:
            infos, total, names = archive.infolist(), 0, set()
            if len(infos) > MAX_ARCHIVE_MEMBERS:
                raise ValueError("archive member bound exceeded")
            for info in infos:
                name, total = info.filename, total + info.file_size
                mode = info.external_attr >> 16
                if not name or name.startswith("/") or "\\" in name or ".." in Path(name).parts or name in names or info.is_dir() or info.flag_bits & 1 or mode not in (0, 0o100644, 0o100755) or len(name) > 4096 or total > MAX_ARCHIVE_BYTES:
                    raise ValueError(f"unsafe archive member: {name!r}")
                names.add(name)
            result["archive_members"] = [{"name": x.filename, "bytes": x.file_size} for x in infos]
    except zipfile.BadZipFile:
        result["archive_members"] = []
    except Exception as exc:
        blockers.append(f"checkpoint archive blocked: {exc}")
    try:
        import torch
        unsafe_fn = getattr(torch.serialization, "get_unsafe_globals_in_checkpoint", None)
        unsafe = unsafe_fn(str(path)) if unsafe_fn else ["unavailable"]
        safe_globals = {"torch.FloatStorage", "collections.OrderedDict", "torch._utils._rebuild_tensor_v2"}
        if unsafe == ["unavailable"]:
            blockers.append("checkpoint unsafe-global scanner unavailable")
        elif set(unsafe) - safe_globals:
            blockers.append(f"checkpoint unsafe globals: {sorted(set(unsafe) - safe_globals)!r}")
        result["unsafe_globals"] = unsafe
        value = torch.load(path, map_location="cpu", weights_only=True)
        seen, tensors, count = set(), [], 0
        def walk(item: Any, name: str, depth: int = 0) -> None:
            nonlocal count
            count += 1
            if count > MAX_ITEMS or depth > MAX_DEPTH:
                raise ValueError("checkpoint walk bound exceeded")
            if isinstance(item, torch.Tensor):
                finite = bool(torch.isfinite(item).all().item()) if item.is_floating_point() else True
                if not finite:
                    blockers.append(f"non-finite tensor: {name}")
                tensors.append({"name": name, "shape": list(item.shape), "dtype": str(item.dtype), "numel": item.numel(), "finite": finite})
                return
            if item is None or isinstance(item, (bool, int, float, str)):
                return
            ident = id(item)
            if ident in seen:
                raise ValueError(f"checkpoint cycle: {name}")
            seen.add(ident)
            if isinstance(item, dict):
                for key, child in item.items():
                    if not isinstance(key, str) or not key or "\0" in key or "\\" in key or "/" in key or ".." in Path(key).parts:
                        raise ValueError(f"unsafe checkpoint key: {key!r}")
                    walk(child, f"{name}.{key}" if name else key, depth + 1)
            elif isinstance(item, (list, tuple)):
                for index, child in enumerate(item):
                    walk(child, f"{name}[{index}]", depth + 1)
            else:
                raise ValueError(f"unsupported checkpoint object: {type(item).__name__}")
            seen.remove(ident)
        walk(value, "")
        canonical = json.dumps(
            [
                {"name": item["name"], "shape": item["shape"], "dtype": item["dtype"], "numel": item["numel"]}
                for item in sorted(tensors, key=lambda item: item["name"])
            ],
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        result.update(
            {
                "safe_load": "WEIGHTS_ONLY",
                "tensor_count": len(tensors),
                "tensors": tensors,
                "tensor_manifest_sha256": hashlib.sha256(canonical).hexdigest(),
                "tensor_manifest_definition": "sorted name/shape/dtype/numel JSON; values never read",
            }
        )
    except Exception as exc:
        blockers.append(f"weights_only load blocked: {exc}")
        result["safe_load"] = "BLOCKED"
    return result


def yaml_topology(text: str, blockers: list[str]) -> dict[str, Any]:
    """Check fixed scalar/list facts without constructing HyperPyYAML tags."""
    facts = {
        "sample_rate": r"(?m)^\s*sample_rate:\s*24000\s*(?:#.*)?$",
        "llm_input_size": r"(?m)^\s*llm_input_size:\s*896\s*(?:#.*)?$",
        "llm_output_size": r"(?m)^\s*llm_output_size:\s*896\s*(?:#.*)?$",
        "spk_embed_dim": r"(?m)^\s*spk_embed_dim:\s*192\s*(?:#.*)?$",
        "token_frame_rate": r"(?m)^\s*token_frame_rate:\s*25\s*(?:#.*)?$",
        "token_mel_ratio": r"(?m)^\s*token_mel_ratio:\s*2\s*(?:#.*)?$",
        "speech_token_size": r"(?m)^\s*speech_token_size:\s*6561\s*(?:#.*)?$",
        "flow_input_size": r"(?m)^\s*input_size:\s*512\s*(?:#.*)?$",
        "flow_output_size": r"(?m)^\s*output_size:\s*80\s*(?:#.*)?$",
        "pre_lookahead_len": r"(?m)^\s*pre_lookahead_len:\s*3\s*(?:#.*)?$",
        "flow_attention_heads": r"(?m)^\s*attention_heads:\s*8\s*(?:#.*)?$",
        "flow_num_blocks": r"(?m)^\s*num_blocks:\s*6\s*(?:#.*)?$",
        "cfm_channels": r"(?m)^\s*channels:\s*\[256\]\s*(?:#.*)?$",
        "cfm_in_channels": r"(?m)^\s*in_channels:\s*240\s*(?:#.*)?$",
        "cfm_estimator_in_channels": r"(?m)^\s*in_channels:\s*320\s*(?:#.*)?$",
        "cfm_num_blocks": r"(?m)^\s*n_blocks:\s*4\s*(?:#.*)?$",
        "cfm_mid_blocks": r"(?m)^\s*num_mid_blocks:\s*12\s*(?:#.*)?$",
        "cfm_heads": r"(?m)^\s*num_heads:\s*8\s*(?:#.*)?$",
        "cfm_sigma_min": r"(?m)^\s*sigma_min:\s*1e-06\s*(?:#.*)?$",
        "cfm_solver": r"(?m)^\s*solver:\s*['\"]?euler['\"]?\s*(?:#.*)?$",
        "cfm_t_scheduler": r"(?m)^\s*t_scheduler:\s*['\"]?cosine['\"]?\s*(?:#.*)?$",
        "cfm_inference_cfg_rate": r"(?m)^\s*inference_cfg_rate:\s*0\.7\s*(?:#.*)?$",
        "hift_in_channels": r"(?m)^\s*in_channels:\s*80\s*(?:#.*)?$",
        "hift_base_channels": r"(?m)^\s*base_channels:\s*512\s*(?:#.*)?$",
        "hift_harmonics": r"(?m)^\s*nb_harmonics:\s*8\s*(?:#.*)?$",
        "hift_rates": r"(?m)^\s*upsample_rates:\s*\[8,\s*5,\s*3\]\s*(?:#.*)?$",
        "hift_kernels": r"(?m)^\s*upsample_kernel_sizes:\s*\[16,\s*11,\s*7\]\s*(?:#.*)?$",
        "istft_n_fft": r"(?m)^\s*n_fft:\s*16\s*(?:#.*)?$",
        "istft_hop_len": r"(?m)^\s*hop_len:\s*4\s*(?:#.*)?$",
        "hift_nsf_alpha": r"(?m)^\s*nsf_alpha:\s*0\.1\s*(?:#.*)?$",
        "hift_nsf_sigma": r"(?m)^\s*nsf_sigma:\s*0\.003\s*(?:#.*)?$",
        "hift_voiced_threshold": r"(?m)^\s*nsf_voiced_threshold:\s*10\s*(?:#.*)?$",
        "hift_audio_limit": r"(?m)^\s*audio_limit:\s*0\.99\s*(?:#.*)?$",
    }
    missing = [name for name, pattern in facts.items() if not re.search(pattern, text)]
    if missing:
        blockers.append(f"cosyvoice2.yaml topology facts missing: {missing!r}")
    return {"status": "EXACT_TOPOLOGY" if not missing else "BLOCKED", "required_facts": sorted(facts), "missing": missing}


def source_inventory(root: Path, matcha: Path, blockers: list[str]) -> dict[str, Any]:
    roles = ("cosyvoice/cli/cosyvoice.py", "cosyvoice/llm/llm.py", "cosyvoice/flow/flow.py", "cosyvoice/hifigan/generator.py", "cosyvoice/tokenizer/tokenizer.py")
    matcha_roles = ("matcha/utils/audio.py", "matcha/hifigan/models.py")
    role_markers = {
        "cosyvoice/cli/cosyvoice.py": "CosyVoice2",
        "cosyvoice/llm/llm.py": "Qwen2LM",
        "cosyvoice/flow/flow.py": "CausalMaskedDiffWithXvec",
        "cosyvoice/hifigan/generator.py": "HiFTGenerator",
        "cosyvoice/tokenizer/tokenizer.py": "Qwen",
        "matcha/utils/audio.py": "mel_spectrogram",
        "matcha/hifigan/models.py": "MultiPeriodDiscriminator",
    }
    expected_role_hashes = {
        "cosyvoice/cli/cosyvoice.py": "8e44f0f0144378561a00ebc065fdb15a843bc4650e68683bebb6624827731859",
        "cosyvoice/llm/llm.py": "6439d57fcf78bcdcad6d31812f3f4b02bd34f513333711ee317d71d1fd14d2de",
        "cosyvoice/flow/flow.py": "a8497feb58336e7566b1f085d11acff9cb4f1a24949abd2c244fbf97c76f9b6d",
        "cosyvoice/hifigan/generator.py": "f74601e6febeb410a961e8ed8931b44074d385ded7f6f77ee918a029b3d42626",
        "cosyvoice/tokenizer/tokenizer.py": "94340fc7cdf270c69a3aeb63290c5241044e20714e01fea736f361f9e5a56df2",
        "matcha/utils/audio.py": "2f741064bfcc9485d19a42356496d8ae24dec80c3d6ef5ca7f4e8abfbf9c61ca",
        "matcha/hifigan/models.py": "2e7ec9ed8cde378877ded04a4078653dca0d7e65feffe9300644d9218258e6af",
    }
    expected_license_hashes = {
        SOURCE_REPOSITORY: "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4",
        MATCHA_REPOSITORY: "874d84104bdc7b301f369b2f2e66b31f07826a67495af909655f3699c857620d",
    }
    expected_role_git_blobs = {
        "cosyvoice/cli/cosyvoice.py": "cc443bed44c651a47492fc7e2142e3a88fb47627",
        "cosyvoice/llm/llm.py": "59ebd48fde1f1b69240391fdac6e2afc1035e123",
        "cosyvoice/flow/flow.py": "a068288f889aff4079b0c54c612897d31d08882a",
        "cosyvoice/hifigan/generator.py": "326a1a70ae7707662939c20493b3a8e4b0906216",
        "cosyvoice/tokenizer/tokenizer.py": "43fb39a2b543cc7ba4ec95fca9327596c34dcff0",
        "matcha/utils/audio.py": "0bcd74df47fb006f68deb5a5f4a4c2fb0aa84f57",
        "matcha/hifigan/models.py": "d209d9a4e99ec29e4167a5a2eaa62d72b3eff694",
    }
    expected_license_git_blobs = {
        SOURCE_REPOSITORY: "261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64",
        MATCHA_REPOSITORY: "858018e750da7be7b271bb7307e68d159ed67ef6",
    }
    result = {"repository": SOURCE_REPOSITORY, "pinned_revision": SOURCE_REVISION, "matcha_repository": MATCHA_REPOSITORY, "matcha_pinned_revision": MATCHA_REVISION, "license_records": []}
    for path, repo, rev, required in ((root, SOURCE_REPOSITORY, SOURCE_REVISION, roles), (matcha, MATCHA_REPOSITORY, MATCHA_REVISION, matcha_roles)):
        try:
            head = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
            origin = subprocess.run(["git", "-C", str(path), "remote", "get-url", "origin"], check=True, capture_output=True, text=True).stdout.strip()
            clean = subprocess.run(["git", "-C", str(path), "status", "--porcelain", "--untracked-files=all"], check=True, capture_output=True, text=True).stdout
            if head != rev or origin != repo or clean:
                blockers.append(f"source identity/clean mismatch: {repo}")
            role_hashes = {}
            for role in required:
                if not (path / role).is_file():
                    blockers.append(f"source role missing: {repo}:{role}")
                else:
                    role_sha = digest(path / role)
                    role_git_blob = git_blob(path / role)
                    role_hashes[role] = {"sha256": role_sha, "expected_sha256": expected_role_hashes[role], "git_blob_sha1": role_git_blob, "expected_git_blob_sha1": expected_role_git_blobs[role], "bytes": (path / role).stat().st_size}
                    if role_sha != expected_role_hashes[role] or role_git_blob != expected_role_git_blobs[role]:
                        blockers.append(f"source role hash mismatch: {repo}:{role}")
                    marker = role_markers.get(role)
                    if marker and marker not in (path / role).read_text(encoding="utf-8", errors="replace"):
                        blockers.append(f"source role marker missing: {repo}:{role}:{marker}")
            tracked = subprocess.run(["git", "-C", str(path), "ls-files", "-s"], check=True, capture_output=True, text=True).stdout.splitlines()
            special = []
            for entry in tracked:
                meta, relative = entry.split("\t", 1); mode = meta.split()[0]
                if mode in ("120000", "160000"):
                    target = subprocess.run(["git", "-C", str(path), "show", f"HEAD:{relative}"], check=True, capture_output=True, text=True).stdout.strip() if mode == "120000" else "gitlink"
                    special.append({"path": relative, "mode": mode, "target": target, "resolution": "NOT_FOLLOWED"})
            license_name = path / "LICENSE"
            license_text = license_name.read_text(encoding="utf-8", errors="replace") if license_name.is_file() else ""
            marker = "Apache License" if repo == SOURCE_REPOSITORY else "MIT License"
            if marker not in license_text:
                blockers.append(f"source license marker missing: {repo}")
            license_sha = digest(license_name) if license_name.is_file() else None
            license_git_blob = git_blob(license_name) if license_name.is_file() else None
            result["license_records"].append({"repository": repo, "path": "LICENSE", "bytes": license_name.stat().st_size if license_name.is_file() else None, "sha256": license_sha, "expected_sha256": expected_license_hashes[repo], "git_blob_sha1": license_git_blob, "expected_git_blob_sha1": expected_license_git_blobs[repo], "declared": marker if marker in license_text else "UNKNOWN"})
            if license_sha != expected_license_hashes[repo] or license_git_blob != expected_license_git_blobs[repo]:
                blockers.append(f"source license hash mismatch: {repo}")
            result.setdefault("checkouts", []).append({"repository": repo, "resolved_revision": head, "origin": origin, "clean": not bool(clean), "roles": list(required), "role_hashes": role_hashes, "tracked_special_entries": special})
            if repo == SOURCE_REPOSITORY:
                tree_line = subprocess.run(["git", "-C", str(path), "ls-tree", "HEAD", "--", "third_party/Matcha-TTS"], check=True, capture_output=True, text=True).stdout.strip()
                tree_fields = tree_line.split()
                oid = tree_fields[2] if len(tree_fields) >= 4 and tree_fields[1] == "commit" and tree_fields[3] == "third_party/Matcha-TTS" else None
                result["matcha_gitlink_oid"] = oid
                if oid != MATCHA_REVISION:
                    blockers.append(f"CosyVoice Matcha gitlink mismatch: {oid!r}")
        except Exception as exc:
            blockers.append(f"source inventory blocked: {repo}: {exc}")
    return result


def inspect(snapshot: Path, source: Path, matcha: Path, tree: Path, out: Path) -> int:
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise RuntimeError("inspection output must be absent or empty (stale evidence is rejected)")
    blockers: list[str] = []
    runtime_blockers = [
        "composite native TTS binder is not implemented",
        "CPU numerical parity is not run",
        "Metal parity is blocked by CPU",
        "historical public artifact is stale LLM-only",
        "dataset/dependency provenance requires separate audit",
    ]
    local = snapshot_files(snapshot)
    tree_packet = server_tree(snapshot, tree, blockers)
    if set(tree_packet["files"]) != EXPECTED_PATHS:
        blockers.append(
            "fixed CosyVoice2 tree path set mismatch: "
            f"missing={sorted(EXPECTED_PATHS - set(tree_packet['files']))!r} "
            f"extra={sorted(set(tree_packet['files']) - EXPECTED_PATHS)!r}"
        )
    if len(tree_packet["files"]) != 19:
        blockers.append(f"expected 19 model files, got {len(tree_packet['files'])}")
    if sum(row["bytes"] for row in tree_packet["files"].values()) != TOTAL_BYTES:
        blockers.append("model tree total bytes mismatch")
    required_names = {"config.json", "cosyvoice2.yaml", "llm.pt", "flow.pt", "hift.pt", "CosyVoice-BlankEN/model.safetensors"}
    names = {p.relative_to(snapshot).as_posix() for p in local}
    if not required_names <= names:
        blockers.append(f"required model files missing: {sorted(required_names - names)!r}")
    fixed = []
    for name, (size, pointer, lfs) in EXPECTED.items():
        path, row = snapshot / name, tree_packet["files"].get(name)
        local_sha256 = digest(path) if path.is_file() else None
        identity_bad = not path.is_file() or not row or path.stat().st_size != size or row.get("git_blob_sha1") != pointer or row.get("lfs_sha256") != lfs
        if path.is_file():
            if lfs is None:
                identity_bad = identity_bad or git_blob(path) != pointer
            else:
                identity_bad = identity_bad or local_sha256 != lfs or lfs_pointer_blob(size, lfs) != pointer
        if identity_bad:
            blockers.append(f"fixed identity mismatch: {name}")
        fixed.append({"path": name, "expected_bytes": size, "expected_git_pointer_sha1": pointer, "expected_lfs_sha256": lfs, "local_sha256": local_sha256})
    config = json_file(snapshot / "config.json", blockers) if (snapshot / "config.json").is_file() else None
    if config != {} or not (snapshot / "config.json").is_file() or (snapshot / "config.json").stat().st_size != 2:
        blockers.append("top-level config.json must be exactly {}")
    package = json_file(snapshot / "configuration.json", blockers) if (snapshot / "configuration.json").is_file() else None
    if package != {"framework": "Pytorch", "task": "text-to-speech"} or not (snapshot / "configuration.json").is_file() or (snapshot / "configuration.json").stat().st_size != 47:
        blockers.append("configuration.json must identify Pytorch text-to-speech")
    qwen = json_file(snapshot / "CosyVoice-BlankEN/config.json", blockers) if (snapshot / "CosyVoice-BlankEN/config.json").is_file() else None
    qwen_expected = {"hidden_size": 896, "intermediate_size": 4864, "num_hidden_layers": 24, "num_attention_heads": 14, "num_key_value_heads": 2, "vocab_size": 151936, "max_position_embeddings": 32768, "rms_norm_eps": 1e-6, "rope_theta": 1e6, "torch_dtype": "bfloat16", "tie_word_embeddings": True}
    if not isinstance(qwen, dict) or any(qwen.get(k) != v for k, v in qwen_expected.items()):
        blockers.append("Qwen configuration facts mismatch")
    vocab = json_file(snapshot / "CosyVoice-BlankEN/vocab.json", blockers) if (snapshot / "CosyVoice-BlankEN/vocab.json").is_file() else None
    if not isinstance(vocab, dict) or len(vocab) != 151643 or set(vocab.values()) != set(range(151643)):
        blockers.append("Qwen vocabulary must contain contiguous ids 0..151642")
    tokenizer_config_path = snapshot / "CosyVoice-BlankEN/tokenizer_config.json"
    tokenizer_config = json_file(tokenizer_config_path, blockers) if tokenizer_config_path.is_file() else None
    if not isinstance(tokenizer_config, dict) or tokenizer_config.get("tokenizer_class") != "Qwen2Tokenizer" or tokenizer_config.get("model_max_length") != 32768:
        blockers.append("Qwen tokenizer must be Qwen2Tokenizer with max length 32768")
    decoder = tokenizer_config.get("added_tokens_decoder", {}) if isinstance(tokenizer_config, dict) else {}
    special_ids = {int(key) for key in decoder if str(key).isdigit()} if isinstance(decoder, dict) else set()
    if not {151643, 151644, 151645} <= special_ids:
        blockers.append("Qwen tokenizer special ids 151643/151644/151645 are missing")
    merges_path = snapshot / "CosyVoice-BlankEN/merges.txt"; merges = merges_path.read_text(encoding="utf-8").splitlines() if merges_path.is_file() else []
    if len([line for line in merges if line and not line.startswith("#")]) != 134839:
        blockers.append("Qwen merges rule count mismatch")
    generation = json_file(snapshot / "CosyVoice-BlankEN/generation_config.json", blockers) if (snapshot / "CosyVoice-BlankEN/generation_config.json").is_file() else None
    generation_expected = {"bos_token_id": 151643, "pad_token_id": 151643, "do_sample": True, "eos_token_id": [151645, 151643], "repetition_penalty": 1.1, "temperature": 0.7, "top_p": 0.8, "top_k": 20}
    if not isinstance(generation, dict) or any(generation.get(k) != v for k, v in generation_expected.items()):
        blockers.append("generation configuration mismatch")
    yaml_path = snapshot / "cosyvoice2.yaml"; yaml_record = {"path": yaml_path.name, "sha256": digest(yaml_path) if yaml_path.is_file() else None, "git_blob_sha1": EXPECTED["cosyvoice2.yaml"][1], "expected_sha256": "0af2c0d010c477187c39f3e8fd5f1ae2e4e6f90ad03ba37c10ed6c6a87b05959"}
    if not yaml_path.is_file() or yaml_record["sha256"] != yaml_record["expected_sha256"] or git_blob(yaml_path) != yaml_record["git_blob_sha1"]:
        blockers.append("cosyvoice2.yaml fixed identity mismatch")
    if yaml_path.is_file():
        try:
            yaml_text = yaml_path.read_text(encoding="utf-8")
            yaml_record["topology"] = yaml_topology(yaml_text, blockers)
            import yaml
            class Loader(yaml.SafeLoader):
                pass
            def unknown(loader, tag, node):
                if isinstance(node, yaml.ScalarNode):
                    return loader.construct_scalar(node)
                if isinstance(node, yaml.SequenceNode):
                    return loader.construct_sequence(node)
                return loader.construct_mapping(node)
            Loader.add_multi_constructor("!", unknown)
            yaml_record["parsed"] = yaml.load(yaml_text, Loader=Loader)
        except Exception as exc:
            blockers.append(f"cosyvoice2.yaml safe parse blocked: {exc}")
    else: blockers.append("cosyvoice2.yaml missing")
    model = safe_safetensors(snapshot / "CosyVoice-BlankEN/model.safetensors", blockers) if (snapshot / "CosyVoice-BlankEN/model.safetensors").is_file() else None
    checkpoints = {name: safe_pt(snapshot / name, blockers) for name in ("llm.pt", "flow.pt", "hift.pt") if (snapshot / name).is_file()}
    onnx = [{"path": name, "bytes": row["bytes"], "git_blob_sha1": row["git_blob_sha1"], "lfs_sha256": row["lfs_sha256"], "execution": "NOT_RUN", "evidence": "IDENTITY_ONLY_NO_GRAPH_PARSE"} for name, row in tree_packet["files"].items() if name.endswith(".onnx")]
    if not onnx:
        blockers.append("ONNX companion inventory is empty")
    runtime_blockers.append("ONNX companions are identity-only; graph structure and native replacement are not inspected")
    source_record = source_inventory(source, matcha, blockers)
    evidence_blockers = list(blockers)
    inspection_status = "AUTHENTICATED_EVIDENCE_COMPLETE" if not evidence_blockers else "INSPECTION_ERROR"
    blockers += runtime_blockers
    payload = {"format": FORMAT, "status": "BLOCKED", "inspection_status": inspection_status, "evidence_stage": "INSPECTION_ONLY", "runtime_status": "NOT_IMPLEMENTED_FAIL_CLOSED", "cpu_status": "UNSUPPORTED", "metal_status": "BLOCKED_BY_CPU", "parity_status": "NOT_RUN", "publication": "NO_UPLOAD", "model": {"repository": REPOSITORY, "revision": REVISION, "total_file_bytes": TOTAL_BYTES, "files": fixed, "server_tree": tree_packet, "package_config": package, "qwen_config": qwen, "tokenizer_config": tokenizer_config, "generation_config": generation, "vocabulary_count": len(vocab) if isinstance(vocab, dict) else None, "merge_rule_count": len([line for line in merges if line and not line.startswith("#")]), "yaml": yaml_record, "safetensors": model, "checkpoints": checkpoints, "onnx_companions": onnx}, "official_source": source_record, "historical_public_artifact": HISTORICAL, "blockers": sorted(set(blockers))}
    out.mkdir(parents=True, exist_ok=True); (out / "manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    return 2


def self_test() -> None:
    assert len(EXPECTED_PATHS) == 19 and set(EXPECTED) == EXPECTED_PATHS
    assert sum(row[0] for row in EXPECTED.values()) == TOTAL_BYTES
    assert EXPECTED["cosyvoice2.yaml"] == (7330, "bc19267bbfd373c9a760b7667a74349ddd487db1", None)
    assert EXPECTED["llm.pt"] == (2023316821, "b8f93347f92a2ce505db9286dd8e72599847c2b1", "b144ef55b51ce8cfb79a73c90dbba0bdaba4e451c0ebcfab20f769264f84a608")
    assert HISTORICAL["tensor_count"] == 295
    with tempfile.TemporaryDirectory(prefix="cosyvoice2-inspect-") as tmp:
        root, snap = Path(tmp), Path(tmp) / "snapshot"; snap.mkdir()
        sample = snap / "x"; sample.write_bytes(b"payload"); lfs = digest(sample); pointer = lfs_pointer_blob(7, lfs)
        assert git_blob(sample) != pointer
        packet = root / "tree.json"; packet.write_text(json.dumps({"repository": REPOSITORY, "revision": REVISION, "resolved_revision": REVISION, "walk": "recursive_file_only", "files": [{"path": "x", "type": "file", "size": 7, "git_blob_sha1": pointer, "lfs_sha256": lfs, "lfs_size": 7}]}))
        bad = []; assert server_tree(snap, packet, bad)["status"] == "MATCHED" and not bad
        packet.write_text(json.dumps({"repository": REPOSITORY, "revision": REVISION, "resolved_revision": REVISION, "walk": "recursive_file_only", "files": [{"path": "x", "type": "file", "size": 7, "git_blob_sha1": "0" * 40, "lfs_sha256": lfs, "lfs_size": 7}]}))
        bad = []; assert server_tree(snap, packet, bad)["status"] == "MISMATCH" and any("LFS pointer" in x for x in bad)
        packet.write_text(json.dumps({"repository": REPOSITORY, "revision": REVISION, "resolved_revision": REVISION, "walk": "recursive_file_only", "files": [{"path": "x", "type": "file", "size": 7, "git_blob_sha1": pointer, "lfs_sha256": lfs, "lfs_size": 7}]}))
        sample.write_bytes(b"changed"); bad = []; assert server_tree(snap, packet, bad)["status"] == "MISMATCH" and bad
        snap_small = root / "snap-small"; snap_small.mkdir(); small = snap_small / "small"; small.write_bytes(b"abc")
        small_packet = root / "small-tree.json"; small_packet.write_text(json.dumps({"repository": REPOSITORY, "revision": REVISION, "resolved_revision": REVISION, "walk": "recursive_file_only", "files": [{"path": "small", "type": "file", "size": 3, "git_blob_sha1": git_blob(small), "lfs_sha256": None, "lfs_size": None}]}))
        bad = []; assert server_tree(snap_small, small_packet, bad)["status"] == "MATCHED" and not bad
        small.write_bytes(b"xyz"); bad = []; assert server_tree(snap_small, small_packet, bad)["status"] == "MISMATCH" and any("changed" in x for x in bad)
        archive = root / "bad.pt"
        with zipfile.ZipFile(archive, "w") as z: z.writestr("../escape", b"x")
        bad = []; safe_pt(archive, bad); assert any("checkpoint archive blocked" in x for x in bad)
        # Header-only safetensors checks must account for dtype width and
        # shape product, not merely trust declared offsets.
        header = json.dumps({"x": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}, separators=(",", ":")).encode()
        tensor = root / "tiny.safetensors"
        tensor.write_bytes(len(header).to_bytes(8, "little") + header + b"\0\0\0\0")
        bad = []; assert safe_safetensors(tensor, bad)["tensor_count"] == 1 and not bad
        broken = root / "broken.safetensors"
        broken_header = json.dumps({"x": {"dtype": "F32", "shape": [2], "data_offsets": [0, 4]}}, separators=(",", ":")).encode()
        broken.write_bytes(len(broken_header).to_bytes(8, "little") + broken_header + b"\0\0\0\0")
        bad = []; safe_safetensors(broken, bad); assert any("safetensors header blocked" in x for x in bad)
        bad = []; assert yaml_topology("sample_rate: 24000\nllm_input_size: 896\n", bad)["status"] == "BLOCKED" and bad
    print("cosyvoice2_inspect self-test: OK")


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--self-test", action="store_true"); parser.add_argument("--snapshot", type=Path); parser.add_argument("--source", type=Path); parser.add_argument("--matcha-source", type=Path); parser.add_argument("--server-tree", type=Path); parser.add_argument("--output", type=Path); args = parser.parse_args()
    if args.self_test:
        if any(x is not None for x in (args.snapshot, args.source, args.matcha_source, args.server_tree, args.output)): parser.error("--self-test accepts no paths")
        self_test(); return 0
    if any(x is None for x in (args.snapshot, args.source, args.matcha_source, args.server_tree, args.output)): parser.error("normal run requires snapshot/source/matcha-source/server-tree/output")
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        print("cosyvoice2 inspection BLOCKED: output must be absent or empty", file=sys.stderr)
        return 2
    try: return inspect(args.snapshot, args.source, args.matcha_source, args.server_tree, args.output)
    except Exception as exc:
        args.output.mkdir(parents=True, exist_ok=True); (args.output / "manifest.json").write_text(json.dumps({"format": FORMAT, "status": "BLOCKED", "inspection_status": "INSPECTION_ERROR", "evidence_stage": "INSPECTION_ONLY", "runtime_status": "NOT_IMPLEMENTED_FAIL_CLOSED", "cpu_status": "UNSUPPORTED", "metal_status": "BLOCKED_BY_CPU", "parity_status": "NOT_RUN", "publication": "NO_UPLOAD", "blockers": [str(exc)]}, indent=2) + "\n"); return 2


if __name__ == "__main__": raise SystemExit(main())
