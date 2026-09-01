#!/usr/bin/env python3
"""Dependency-free validator for the YuE xcodec-mini reference bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

FILES = ("manifest.json", "codes.u32le", "features.f32le", "backbone.f32le", "waveform.f32le")
KEYS = {
    "backbone_dtype", "bytes_backbone_f32le", "bytes_codes_u32le", "bytes_features_f32le", "bytes_waveform_f32le",
    "codes_dtype", "codebook_size", "codebooks", "codec_checkpoint_bytes", "codec_checkpoint_file", "codec_checkpoint_sha256",
    "contiguous", "decoder_checkpoint_bytes", "decoder_checkpoint_file", "decoder_checkpoint_sha256", "device", "feature_dim",
    "features_dtype", "format", "frames", "output_hop_length", "output_sample_rate", "pickle_load_policy", "runtime", "samples",
    "sha256_backbone_f32le", "sha256_codes_u32le", "sha256_features_f32le", "sha256_waveform_f32le", "source_package",
    "source_package_wheel_sha256", "source_files_sha256", "token_frame_rate", "token_hop_length", "token_sample_rate", "torch",
    "upstream_hf", "upstream_revision", "vocos_decoder_tensor_count", "waveform_dtype",
}
SOURCE_FILES = {
    "README.md": (31, "4bcf87ecfbbb8e07a01b21415a970c8b53a5283bf6872b657040d3f45c9241f7"),
    "quantization/__init__.py": (271, "34c806bc1cafc8b835926b6f6450bee769f95eb467cf1c19b4427e9dd7e55bbc"),
    "quantization/vq.py": (4598, "8f24a4a389bad6dec6d77a35526264a1acd07c29a69854274bc73ebda4c622f9"),
    "quantization/core_vq_lsx_version.py": (16050, "154e2c5ddbacd3b82c74bf18d7177ea4b011cbd71e6e5575c7265b70e58c2af0"),
    "quantization/distrib.py": (4109, "79b8dbfe3dda4da10ea0d3e143b373d90dd920f40d4a7f6f7446412b3584f655"),
    "utils/utils.py": (8484, "8521062c4b1afae1366a100244449a7dcdcc79883bf1874e50f9954c66c2ccd2"),
    "utils/ddp_utils.py": (9108, "a53a4efc83ab34c8655d61bbcae7e0965a573ecce3321f8c1cffc2ec6889644f"),
}
CODEC_CHECKPOINT = ("ckpt_00360000.pth", 1360444883, "c8c379ea2d3cbde1c8ba1b9717975220e79ba3f556bb161766fd5e4585dcd59c")
DECODER_CHECKPOINT = ("decoder_151000.pth", 72610550, "8af97a29d3483f9d4a3755992837501bd7d6caa1a69382ed16e64039e0ea0998")
VOCOS_WHEEL_SHA256 = "0ac13eaef68596074301e912d781399b3defa4b4ca60b6bc52c8a4b9209ca235"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_manifest(path: Path) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                # Reject during object construction so a duplicate nested
                # identity can never be observed as a last-wins value.
                raise ValueError(f"duplicate JSON keys: {key}")
            result[key] = value
        return result

    try:
        manifest = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"manifest is malformed: {exc}") from exc
    if not isinstance(manifest, dict) or set(manifest) != KEYS:
        raise ValueError("manifest key schema is not exact")
    return manifest


def validate(reference: Path) -> dict[str, Any]:
    if reference.is_symlink() or not reference.is_dir():
        raise ValueError("reference root must be a regular directory")
    entries = []
    for path in reference.rglob("*"):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"reference contains symlink/non-regular entry: {path.relative_to(reference)}")
        entries.append(path.relative_to(reference).as_posix())
    if set(entries) != set(FILES):
        raise ValueError("reference file set is not exact")
    manifest = parse_manifest(reference / "manifest.json")
    fixed = {
        "format": "vokra-yue-xcodec-mini-reference-v2", "pickle_load_policy": "weights_only=True_required",
        "upstream_hf": "m-a-p/xcodec_mini_infer", "upstream_revision": "fe781a67815ab47b4a3a5fce1e8d0a692da7e4e5",
        "frames": 5, "codebooks": 12, "codebook_size": 1024, "feature_dim": 1024,
        "token_sample_rate": 16000, "token_hop_length": 320, "token_frame_rate": 50,
        "output_sample_rate": 44100, "output_hop_length": 882, "runtime": "torch-cpu", "device": "cpu",
        "codes_dtype": "uint32-le", "features_dtype": "float32-le", "backbone_dtype": "float32-le", "waveform_dtype": "float32-le",
        "contiguous": True,
        "codec_checkpoint_file": CODEC_CHECKPOINT[0], "codec_checkpoint_bytes": CODEC_CHECKPOINT[1],
        "codec_checkpoint_sha256": CODEC_CHECKPOINT[2],
        "decoder_checkpoint_file": DECODER_CHECKPOINT[0], "decoder_checkpoint_bytes": DECODER_CHECKPOINT[1],
        "decoder_checkpoint_sha256": DECODER_CHECKPOINT[2],
        "source_package": "vocos==0.1.0", "source_package_wheel_sha256": VOCOS_WHEEL_SHA256,
        "vocos_decoder_tensor_count": 81, "torch": "2.7.1",
    }
    for key, expected in fixed.items():
        if manifest.get(key) != expected:
            raise ValueError(f"manifest field mismatch: {key}")
    source_map = manifest.get("source_files_sha256")
    if not isinstance(source_map, dict) or set(source_map) != set(SOURCE_FILES):
        raise ValueError("source_files_sha256 map is not exact")
    for name, (size, source_hash) in SOURCE_FILES.items():
        if source_map[name] != source_hash:
            raise ValueError(f"source hash mismatch: {name}")
    for name in FILES[1:]:
        path = reference / name
        size_key = f"bytes_{name.replace('.', '_')}"
        hash_key = f"sha256_{name.replace('.', '_')}"
        if manifest[size_key] != path.stat().st_size or manifest[hash_key] != sha256(path):
            raise ValueError(f"payload identity mismatch: {name}")
    return manifest


def self_test() -> int:
    import tempfile
    with tempfile.TemporaryDirectory(prefix="yue-reference-validator-") as directory:
        root = Path(directory)
        for name in FILES[1:]: (root / name).write_bytes(b"abc")
        payload_hash = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        manifest: dict[str, Any] = {key: "SELF_TEST" for key in KEYS}
        manifest.update({"format": "vokra-yue-xcodec-mini-reference-v2", "pickle_load_policy": "weights_only=True_required", "upstream_hf": "m-a-p/xcodec_mini_infer", "upstream_revision": "fe781a67815ab47b4a3a5fce1e8d0a692da7e4e5", "frames": 5, "codebooks": 12, "codebook_size": 1024, "feature_dim": 1024, "token_sample_rate": 16000, "token_hop_length": 320, "token_frame_rate": 50, "output_sample_rate": 44100, "output_hop_length": 882, "runtime": "torch-cpu", "device": "cpu", "codes_dtype": "uint32-le", "features_dtype": "float32-le", "backbone_dtype": "float32-le", "waveform_dtype": "float32-le", "contiguous": True, "source_files_sha256": {k: v[1] for k, v in SOURCE_FILES.items()}})
        for name in FILES[1:]: manifest[f"bytes_{name.replace('.', '_')}"] = 3; manifest[f"sha256_{name.replace('.', '_')}"] = payload_hash
        manifest.update({"codec_checkpoint_file": CODEC_CHECKPOINT[0], "codec_checkpoint_bytes": CODEC_CHECKPOINT[1], "codec_checkpoint_sha256": CODEC_CHECKPOINT[2], "decoder_checkpoint_file": DECODER_CHECKPOINT[0], "decoder_checkpoint_bytes": DECODER_CHECKPOINT[1], "decoder_checkpoint_sha256": DECODER_CHECKPOINT[2], "source_package": "vocos==0.1.0", "source_package_wheel_sha256": VOCOS_WHEEL_SHA256, "vocos_decoder_tensor_count": 81, "torch": "2.7.1"})
        (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        validate(root)
        (root / "extra").write_bytes(b"x")
        try: validate(root); return 1
        except ValueError: pass
        (root / "extra").unlink(missing_ok=True)
        manifest_text = (root / "manifest.json").read_text(encoding="utf-8")
        (root / "manifest.json").write_text(manifest_text.rstrip()[:-1] + ',\n  "format": "vokra-yue-xcodec-mini-reference-v2"\n}\n', encoding="utf-8")
        try: validate(root); return 1
        except ValueError: pass
        (root / "manifest.json").write_text(manifest_text, encoding="utf-8")
        (root / "manifest.json").write_text("[]\n", encoding="utf-8")
        try: validate(root); return 1
        except ValueError: pass
        (root / "manifest.json").write_text(manifest_text, encoding="utf-8")
        nested_duplicate = manifest_text.replace(
            '"source_files_sha256": {',
            '"source_files_sha256": {"README.md": "duplicate",',
        )
        (root / "manifest.json").write_text(nested_duplicate, encoding="utf-8")
        try: validate(root); return 1
        except ValueError: pass
        (root / "manifest.json").write_text(manifest_text, encoding="utf-8")
        (root / "codes.u32le").write_bytes(b"tampered")
        try: validate(root); return 1
        except ValueError: pass
        (root / "codes.u32le").write_bytes(b"abc")
        (root / "extra").unlink(missing_ok=True); (root / "codes.u32le").unlink(); (root / "codes.link").symlink_to(root / "features.f32le"); (root / "codes.u32le").symlink_to(root / "codes.link")
        try: validate(root); return 1
        except ValueError: pass
    print("yue reference validator self-test: PASS")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--reference", type=Path); parser.add_argument("--self-test", action="store_true"); args = parser.parse_args()
    if args.self_test: raise SystemExit(self_test())
    if args.reference is None: parser.error("--reference is required")
    try: validate(args.reference)
    except ValueError as exc: print(f"reference validator: BLOCKED: {exc}"); raise SystemExit(2)
    print("reference validator: PASS")
