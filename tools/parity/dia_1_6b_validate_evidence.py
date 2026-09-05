#!/usr/bin/env -S uv run --frozen --project tools/parity python
"""Independent validator for Dia official-reference evidence directories."""
from __future__ import annotations

import argparse
import hashlib
import json
import tomllib
from pathlib import Path


REQUIRED = {
    "text_ids", "text_padding_mask", "conditional_encoder",
    "unconditional_encoder", "decoder_logits", "decoder_sampling_probability",
    "selected_ids", "delayed_codes", "reverted_codes", "dac_latent", "pcm",
}
EXPECTED_HF = {
    ".gitattributes": (1519, "a6344aac8c09253b3b630fb776ae94478aa0275b", None),
    "README.md": (6551, "146916d420c1f14cf794a171811ac56e42d13dbd", None),
    "config.json": (941, "0a586180c3246fefa312c5e3977a6e419a7a113d", None),
    "preprocessor_config.json": (172, "a812a82c392c511dc04417b8f8bcde9411347af0", None),
    "dia-v0_1.pth": (6444788896, "8dc5b43681e210512ee8dbf8d028737c7f449180", "d12004b2f3121af763bdf2a3b575586b00c02bdd00a315a23c7b7bdb2a8f9475"),
    "model.safetensors": (6444682848, "d3cc9ed6f729aa7894307b7799aafe9330853c48", "caba289b60f6d7d1e58fc744f4dc25aae88995fcca46be3d05e220b971486a26"),
}
EXPECTED_PUBLIC = {"path": "dia-1.6b.gguf", "bytes": 6444673088, "git_blob_sha1": "e00731cd617132cf198f7bcaaee190de2df86c5f", "lfs_sha256": "a90733e9e6806cae66abf3eca1d575ecf6dab9298c07d39fc4217a509c952a6d"}
SOURCE_ROLE_BLOBS = {
    "LICENSE": "483d716cc886695f19971a99658c59851a8a2866",
    "dia/audio.py": "5c1947103bc0d95255d97618c699fa0a18993beb",
    "dia/config.py": "09c6d136a41e0296483d2617061d4261cbf4c42c",
    "dia/layers.py": "f9aed506b25e99d053dd71d6def7a0bd33075ace",
    "dia/model.py": "a3b0f9730a810fa170019511a2696e7f813090de",
    "dia/state.py": "172ec52c7c344781aad0552a6cddd6e5f1933894",
    "pyproject.toml": "dd844dd2fb0ab0c016520c4b070beaa7c159e3e1",
}
REFERENCE_PROJECT_LOCK_SHA256 = "ccdfaf4cfedd7780f8c1032a42341f28ac56bec7353f4563f9a1b44b764cf29c"
REFERENCE_PROJECT_PYPROJECT_SHA256 = "56430b6f50620df9ce3383f535dec1755843a4a9bab9758e34cf69e9913b6fc2"
DIRECT_DEPENDENCY_VERSIONS = {
    "einops": "0.8.2", "gguf": "0.19.0", "huggingface-hub": "0.30.2",
    "numpy": "2.2.5", "pydantic": "2.11.3", "soundfile": "0.13.1",
    "torch": "2.6.0+cpu", "torchaudio": "2.6.0+cpu",
}
DEPENDENCY_LICENSE_CONCLUSIONS = {
    "annotated-types": "MIT_REVIEWED", "certifi": "MPL-2.0_BLOCKED_BY_POLICY",
    "cffi": "MIT_NATIVE_LIBFFI_REVIEW_REQUIRED", "charset-normalizer": "MIT_REVIEWED",
    "colorama": "BSD-3-Clause_REVIEWED", "einops": "MIT_REVIEWED",
    "filelock": "UNLICENSE_POLICY_REVIEW_REQUIRED", "fsspec": "BSD-3-Clause_REVIEWED",
    "gguf": "MIT_REVIEWED", "huggingface-hub": "Apache-2.0_REVIEWED",
    "idna": "BSD-3-Clause_REVIEWED", "jinja2": "BSD-3-Clause_REVIEWED",
    "markupsafe": "BSD-3-Clause_REVIEWED", "mpmath": "BSD_STYLE_PRIMARY_REVIEW_REQUIRED",
    "networkx": "BSD-3-Clause_REVIEWED", "numpy": "BSD-3-Clause_NATIVE_BUNDLE_REVIEW_REQUIRED",
    "packaging": "Apache-2.0_REVIEWED", "pycparser": "BSD-3-Clause_REVIEWED",
    "pydantic": "MIT_REVIEWED", "pydantic-core": "MIT_NATIVE_EXTENSION_REVIEW_REQUIRED",
    "pyyaml": "MIT_NATIVE_EXTENSION_REVIEW_REQUIRED", "requests": "Apache-2.0_REVIEWED",
    "setuptools": "MIT_REVIEWED", "soundfile": "BSD-3-Clause_NATIVE_LIBSNDFILE_REVIEW_REQUIRED",
    "sympy": "BSD-3-Clause_REVIEWED", "torch": "BSD-3-Clause_BUNDLED_COMPONENT_REVIEW_REQUIRED",
    "torchaudio": "BSD-2-Clause_BUNDLED_COMPONENT_REVIEW_REQUIRED",
    "tqdm": "MPL-2.0_OR_MIT_POLICY_REVIEW_REQUIRED", "typing-extensions": "PSF-2.0_BLOCKED_BY_POLICY",
    "typing-inspection": "MIT_REVIEWED", "urllib3": "MIT_REVIEWED",
    "vokra-dia-1-6b-reference": "FIRST_PARTY_NOT_INDEPENDENT_DEPENDENCY_SCOPE",
}


def unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate manifest key: {key}")
        result[key] = value
    return result


def require_text_markers(values):
    values = list(values)
    if values.count(1) != 1 or values.count(2) != 1 or values.index(1) >= values.index(2):
        raise ValueError("complete official text_ids lacks ordered [S1]/[S2] markers")


def require_pcm_hop(pcm_samples: int, reverted_frames: int) -> None:
    if pcm_samples != reverted_frames * 512:
        raise ValueError("PCM sample count must equal reverted DAC frames * 512")


def require_dac_proof(mapping: dict) -> None:
    if mapping.get("status") != "PROVEN_EXACT" or mapping.get("sample_rate") != 44100 or mapping.get("n_codebooks") != 9 or mapping.get("hop_length") != 512 or not isinstance(mapping.get("vokra_dac_manifest_sha256"), str) or len(mapping["vokra_dac_manifest_sha256"]) != 64:
        raise ValueError("DAC exact checkpoint/Vokra manifest proof is unavailable")


def require_reference_project(identity: dict) -> None:
    if not isinstance(identity, dict) or identity.get("project") != "dia_1_6b_reference" or identity.get("python") != "3.12":
        raise ValueError("dedicated Dia reference project identity is missing")
    if identity.get("uv_lock_sha256") != REFERENCE_PROJECT_LOCK_SHA256 or identity.get("pyproject_sha256") != REFERENCE_PROJECT_PYPROJECT_SHA256 or identity.get("lock_schema") != "uv-lock-v1-python312":
        raise ValueError("dedicated Dia lock/schema identity mismatch")
    if identity.get("use_torch_compile") is not False:
        raise ValueError("torch.compile must remain disabled in the adapted reference closure")
    audit = identity.get("dependency_audit")
    if not isinstance(audit, dict) or audit.get("schema") != "vokra-dia-uv-lock-license-audit-v1" or audit.get("status") != "BLOCKED_UNREVIEWED_TRANSITIVE" or audit.get("package_count") != 34 or not isinstance(audit.get("rows"), list) or len(audit["rows"]) != 34 or len(audit.get("rows_sha256", "")) != 64:
        raise ValueError("complete CPU lock license-audit rows are missing")
    canonical_rows = []
    for row in audit["rows"]:
        if not isinstance(row, dict) or set(row) != {"name", "version", "source", "markers", "row_sha256", "license_conclusion"} or row["name"] not in DEPENDENCY_LICENSE_CONCLUSIONS or not isinstance(row["source"], dict) or not isinstance(row["markers"], list) or row["license_conclusion"] != DEPENDENCY_LICENSE_CONCLUSIONS[row["name"]]:
            raise ValueError("lock license-audit row is incomplete or has no primary conclusion")
        identity_row = {"name": row["name"], "version": row["version"], "source": row["source"], "markers": row["markers"]}
        if row["row_sha256"] != hashlib.sha256(json.dumps(identity_row, sort_keys=True, separators=(",", ":")).encode()).hexdigest():
            raise ValueError("lock license-audit row digest mismatch")
        canonical_rows.append(row)
    canonical_rows.sort(key=lambda row: (row["name"], row["version"], json.dumps(row["source"], sort_keys=True), row["markers"]))
    if hashlib.sha256(json.dumps(canonical_rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest() != audit["rows_sha256"]:
        raise ValueError("lock license-audit aggregate digest mismatch")
    lock_path = Path(__file__).parent / "dia_1_6b_reference" / "uv.lock"
    if not lock_path.is_file():
        raise ValueError("dedicated lock is missing for independent license-audit validation")
    lock_document = tomllib.loads(lock_path.read_text(encoding="utf-8"))
    expected_rows = []
    for package in lock_document.get("package", []):
        if not isinstance(package, dict) or not isinstance(package.get("name"), str) or not isinstance(package.get("version"), str):
            raise ValueError("uv.lock package row is malformed")
        name = package["name"]
        if name not in DEPENDENCY_LICENSE_CONCLUSIONS:
            raise ValueError(f"uv.lock package has no reviewed license conclusion: {name}")
        source = package.get("source", {})
        markers = sorted({dependency.get("marker") for dependency in package.get("dependencies", []) if isinstance(dependency, dict) and isinstance(dependency.get("marker"), str)})
        identity_row = {"name": name, "version": package["version"], "source": source, "markers": markers}
        expected_rows.append({**identity_row, "row_sha256": hashlib.sha256(json.dumps(identity_row, sort_keys=True, separators=(",", ":")).encode()).hexdigest(), "license_conclusion": DEPENDENCY_LICENSE_CONCLUSIONS[name]})
    expected_rows.sort(key=lambda row: (row["name"], row["version"], json.dumps(row["source"], sort_keys=True), row["markers"]))
    expected_digest = hashlib.sha256(json.dumps(expected_rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if audit["rows"] != expected_rows or audit["rows_sha256"] != expected_digest:
        raise ValueError("manifest lock license-audit rows do not match the dedicated lock")
    if identity.get("dependency_license_audit") != "AUDITED_ALLOW":
        raise ValueError("Dia dependency license/provenance audit is not affirmatively allowed")
    if identity.get("expected_versions") != DIRECT_DEPENDENCY_VERSIONS or identity.get("actual_versions") != DIRECT_DEPENDENCY_VERSIONS:
        raise ValueError("actual locked dependency versions are not bound to the manifest")


def require_sampling_cardinality(sampling: dict, logits: list, probability: list, selected: list) -> None:
    if sampling.get("global_torch_multinomial_scope") != "official_sampler_only" or sampling.get("selection_evidence") != "exact official selected IDs" or sampling.get("rng_equivalence") != "NOT_CLAIMED" or sampling.get("logits_calls") != len(logits) or sampling.get("probability_calls") != len(probability) or len(logits) != len(probability) or len(selected) != 1 or selected[0]["shape"] != [len(logits), 9] or any(entry["shape"] != [9, 1028] for entry in logits + probability):
        raise ValueError("decoder logits/probability/selected call alignment mismatch")


def validate(root: Path) -> None:
    import numpy as np
    if not root.is_dir() or not (root / "manifest.json").is_file():
        raise ValueError("evidence directory/manifest is missing")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"), object_pairs_hook=unique_pairs)
    if manifest.get("format") != "vokra-dia-1-6b-official-reference-v1" or manifest.get("status") != "REFERENCE_COMPLETE":
        raise ValueError("manifest is not a completed official-reference packet")
    if manifest.get("native_status") != "BLOCKED_UNTIL_VAST_AND_APPLE_EVIDENCE" or manifest.get("publication") != "NO_UPLOAD":
        raise ValueError("native/public status drift")
    if manifest.get("comparison_status") != "NOT_RUN_OFFICIAL_ONLY":
        raise ValueError("reference-only packet must say native comparison was not run")
    require_reference_project(manifest.get("reference_project"))
    source = manifest.get("source")
    hf = manifest.get("hf")
    public = manifest.get("public")
    dac = manifest.get("dac")
    for identity in (source, hf, public, dac):
        if not isinstance(identity, dict):
            raise ValueError("model/source/public/DAC identity is missing")
    if source.get("repository") != "https://github.com/nari-labs/dia.git" or source.get("resolved_revision") != source.get("revision") or len(source.get("revision", "")) != 40 or source.get("clean") is not True or not isinstance(source.get("files"), dict) or not source["files"]:
        raise ValueError("official source identity/role binding is incomplete")
    if set(source["files"]) != set(SOURCE_ROLE_BLOBS):
        raise ValueError("official source role set mismatch")
    for name, row in source["files"].items():
        if row.get("git_blob_sha1") != SOURCE_ROLE_BLOBS[name]:
            raise ValueError(f"official source role blob mismatch: {name}")
    if hf.get("repository") != "nari-labs/Dia-1.6B" or hf.get("revision") != "257bc72f9b78182ccc6fa07675a9ae4c1a44e2cd":
        raise ValueError("HF identity binding mismatch")
    expected_hf = {".gitattributes", "README.md", "config.json", "preprocessor_config.json", "dia-v0_1.pth", "model.safetensors"}
    if set(hf.get("files", {})) != expected_hf:
        raise ValueError("HF file tree binding mismatch")
    for name, record in hf["files"].items():
        if not isinstance(record, dict) or not isinstance(record.get("bytes"), int) or len(record.get("sha256", "")) != 64 or len(record.get("git_blob_sha1", "")) != 40:
            raise ValueError(f"HF file identity is incomplete: {name}")
        expected = EXPECTED_HF[name]
        if (record["bytes"], record["git_blob_sha1"], record.get("lfs_sha256")) != expected:
            raise ValueError(f"HF file identity mismatch: {name}")
    if public.get("repository") != "vokra/dia-1.6b" or public.get("revision") != "dd1df2a129fed7d15c365caeabaae227ccfe8537":
        raise ValueError("public identity binding mismatch")
    if public.get("file", {}).get("path") != EXPECTED_PUBLIC["path"] or public.get("file", {}).get("bytes") != EXPECTED_PUBLIC["bytes"] or public.get("file", {}).get("git_blob_sha1") != EXPECTED_PUBLIC["git_blob_sha1"] or public.get("file", {}).get("lfs_sha256") != EXPECTED_PUBLIC["lfs_sha256"] or len(public.get("file", {}).get("sha256", "")) != 64:
        raise ValueError("public artifact body identity is incomplete")
    mapping = dac.get("mapping")
    if dac.get("status") != "DAC_PROOF_REQUIRED" or not isinstance(mapping, dict):
        raise ValueError("DAC exact checkpoint/Vokra manifest proof is unavailable")
    require_dac_proof(mapping)
    if dac.get("source", {}).get("repository") != "https://github.com/descriptinc/descript-audio-codec" or not dac.get("source", {}).get("files") or dac.get("package", {}).get("name") != "descript-audio-codec-source-shell" or dac.get("package", {}).get("version") != "1.0.0" or dac.get("package", {}).get("source_root") != "dac" or not dac.get("package", {}).get("files"):
        raise ValueError("DAC source/package tree binding is incomplete")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != REQUIRED:
        raise ValueError("exact artifact role set mismatch")
    paths = set()
    for role, entries in artifacts.items():
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"empty artifact role: {role}")
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) != {"shape", "dtype", "finite", "path", "bytes", "sha256"}:
                raise ValueError(f"artifact schema mismatch: {role}")
            relative = entry["path"]
            if not isinstance(relative, str) or relative in paths or Path(relative).is_absolute() or ".." in Path(relative).parts:
                raise ValueError(f"duplicate/unsafe artifact path: {role}")
            paths.add(relative)
            file = root / relative
            if not file.is_file() or file.stat().st_size == 0 or file.stat().st_size != entry["bytes"]:
                raise ValueError(f"missing/empty artifact: {role}")
            digest = hashlib.sha256(file.read_bytes()).hexdigest()
            if digest != entry["sha256"] or len(digest) != 64:
                raise ValueError(f"artifact hash mismatch: {role}")
            array = np.load(file, allow_pickle=False)
            if list(array.shape) != entry["shape"] or array.dtype.name != entry["dtype"].removeprefix("torch.") or not np.isfinite(array).all() or entry["finite"] is not True:
                raise ValueError(f"artifact shape/dtype/finiteness mismatch: {role}")
    if {p.name for p in root.iterdir() if p.is_file()} != paths | {"manifest.json"}:
        raise ValueError("stale/orphan evidence file present")
    sampling = manifest.get("sampling")
    logits = artifacts["decoder_logits"]
    probability = artifacts["decoder_sampling_probability"]
    selected = artifacts["selected_ids"]
    if not isinstance(sampling, dict):
        raise ValueError("decoder sampling evidence is missing")
    require_sampling_cardinality(sampling, logits, probability, selected)
    if artifacts["conditional_encoder"][0]["shape"] != artifacts["unconditional_encoder"][0]["shape"] or artifacts["conditional_encoder"][0]["shape"][0] != 1:
        raise ValueError("CFG encoder row schema mismatch")
    if artifacts["delayed_codes"][0]["shape"][-1] != 9 or artifacts["reverted_codes"][0]["shape"][-1] != 9 or artifacts["dac_latent"][0]["shape"][1] != 1024 or artifacts["pcm"][0]["shape"][-1] <= 0:
        raise ValueError("audio code/DAC/PCM schema mismatch")
    text_array = np.load(root / artifacts["text_ids"][0]["path"], allow_pickle=False).reshape(-1).tolist()
    require_text_markers(text_array)
    schema = manifest.get("schema", {})
    if schema.get("decoder", {}).get("channels") != 9 or schema.get("decoder", {}).get("vocab") != 1028 or schema.get("decoder", {}).get("call_order") != "logits -> official multinomial probability -> selected IDs" or schema.get("audio_codes", {}).get("axis_order") != "[batch,frames,channels]":
        raise ValueError("decoder/code artifact schema is incomplete")
    delayed_frames = artifacts["delayed_codes"][0]["shape"][1]
    reverted_frames = artifacts["reverted_codes"][0]["shape"][1]
    latent_shape = artifacts["dac_latent"][0]["shape"]
    if delayed_frames != reverted_frames + 15 or len(latent_shape) != 3 or latent_shape[2] != reverted_frames:
        raise ValueError("DAC latent frame cardinality does not match delayed code frames")
    require_pcm_hop(artifacts["pcm"][0]["shape"][-1], reverted_frames)
    if schema.get("dac", {}).get("sample_rate") != 44100 or schema.get("dac", {}).get("hop_length") != 512:
        raise ValueError("DAC sample-rate/hop schema missing")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("evidence", nargs="?", type=Path)
    args = parser.parse_args()
    if args.self_test:
        assert unique_pairs([("x", 1)]) == {"x": 1}
        try:
            unique_pairs([("x", 1), ("x", 2)])
            raise AssertionError("duplicate manifest key accepted")
        except ValueError:
            pass
        try:
            require_text_markers([1, 2, 1])
            raise AssertionError("duplicate marker accepted")
        except ValueError:
            pass
        try:
            require_pcm_hop(4, 3)
        except ValueError:
            pass
        try:
            require_dac_proof({"status": "EXACT_TO_VOKRA_DAC_KHZ44", "sample_rate": 44100, "n_codebooks": 9, "hop_length": 512})
            raise AssertionError("self-asserted DAC mapping accepted")
        except ValueError:
            pass
        try:
            require_reference_project({"project": "dia_1_6b_reference", "python": "3.12", "uv_lock_sha256": REFERENCE_PROJECT_LOCK_SHA256, "pyproject_sha256": REFERENCE_PROJECT_PYPROJECT_SHA256, "lock_schema": "uv-lock-v1-python312", "use_torch_compile": False, "dependency_license_audit": "BLOCKED_UNREVIEWED_TRANSITIVE", "expected_versions": DIRECT_DEPENDENCY_VERSIONS, "actual_versions": DIRECT_DEPENDENCY_VERSIONS})
            raise AssertionError("blocked dependency audit accepted")
        except ValueError:
            pass
        assert set(DIRECT_DEPENDENCY_VERSIONS) == {
            "einops", "gguf", "huggingface-hub", "numpy", "pydantic",
            "soundfile", "torch", "torchaudio",
        }
        assert not set(DIRECT_DEPENDENCY_VERSIONS) & {
            "descript-audio-codec", "gradio", "librosa", "soxr", "triton",
        }
        try:
            require_sampling_cardinality({"global_torch_multinomial_scope": "official_sampler_only", "selection_evidence": "exact official selected IDs", "rng_equivalence": "NOT_CLAIMED", "logits_calls": 1, "probability_calls": 0}, [{"shape": [9, 1028]}], [], [{"shape": [1, 9]}])
            raise AssertionError("sampling cardinality mismatch accepted")
        except ValueError:
            pass
        try:
            expected = {"manifest.json", "text_ids-0000.npy"}
            actual = expected | {"orphan.npy"}
            if actual - expected:
                raise ValueError("orphan evidence file")
            raise AssertionError("orphan file accepted")
        except ValueError:
            pass
        print("dia evidence validator self-test: OK")
        return 0
    if args.evidence is None:
        parser.error("evidence directory is required")
    validate(args.evidence)
    print("Dia reference evidence validation: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
