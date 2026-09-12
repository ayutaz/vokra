#!/usr/bin/env -S uv run --frozen --project tools/parity python
"""Independent validator for Dia official-reference evidence directories."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
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
REFERENCE_PROJECT_LOCK_SHA256 = "58218102471c94979b1e9147759abf50fa3784793c193ff30cdde908400650dc"
REFERENCE_PROJECT_PYPROJECT_SHA256 = "fa675f2c7542bd9eebedcc6ba29963f49093305c7a518542d71fad424449e77b"
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
SOURCE_CONTRACT_FORMAT = "vokra-dia-1-6b-source-contract-v1"
SOURCE_CONTRACT_RUST_FILES = {
    "crates/vokra-models/src/dia/tokenizer.rs",
    "crates/vokra-models/src/dia/forward.rs",
    "crates/vokra-models/src/dia/mod.rs",
}


def unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate manifest key: {key}")
        result[key] = value
    return result


def require_canonical_existing_path(path: Path) -> None:
    if not path.is_absolute() or any(part in {".", ".."} for part in path.parts):
        raise ValueError("evidence path must be absolute and free of dot components")
    cursor = Path(path.anchor)
    for part in path.parts[1:]:
        cursor /= part
        if cursor.is_symlink():
            raise ValueError("evidence path has symlink ancestry")
    if not path.is_dir() or path.is_symlink():
        raise ValueError("evidence path must be a regular directory")


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


def require_source_contract(contract: dict) -> None:
    if not isinstance(contract, dict) or contract.get("format") != SOURCE_CONTRACT_FORMAT or contract.get("status") != "SOURCE_CONTRACT_COMPLETE_MODEL_FREE":
        raise ValueError("official Dia source-only contract is missing or incomplete")
    if contract.get("model_payload_access") != "NONE" or contract.get("checkpoint_loaded") is not False or contract.get("dac_loaded") is not False or contract.get("pcm_generated") is not False or contract.get("parity_status") != "NOT_RUN_SOURCE_CONTRACT_ONLY" or contract.get("publication") != "NO_UPLOAD":
        raise ValueError("source-only contract claims model execution or parity")
    source = contract.get("source")
    if not isinstance(source, dict) or source.get("repository") != "https://github.com/nari-labs/dia.git" or source.get("revision") != "2811af1c5f476b1f49f4744fabf56cf352be21e5" or source.get("resolved_revision") != source.get("revision") or source.get("clean") is not True or set(source.get("files", {})) != set(SOURCE_ROLE_BLOBS):
        raise ValueError("source-only contract source identity is incomplete")
    for name, blob in SOURCE_ROLE_BLOBS.items():
        row = source["files"].get(name)
        if not isinstance(row, dict) or row.get("git_blob_sha1") != blob or not isinstance(row.get("sha256"), str) or len(row["sha256"]) != 64:
            raise ValueError(f"source-only contract role identity mismatch: {name}")
    rust = contract.get("rust")
    if not isinstance(rust, dict) or rust.get("status") != "RUST_SOURCE_CONTRACT_MATCHED" or rust.get("delay_pattern") != [0, 8, 9, 10, 11, 12, 13, 14, 15] or rust.get("speaker_markers") != {"[S1]": 1, "[S2]": 2} or rust.get("staggered_eos_drain") != {"max_delay": 15, "apply_generation_drain": True, "generate_codes": True} or set(rust.get("files", {})) != SOURCE_CONTRACT_RUST_FILES:
        raise ValueError("Rust Dia source contract comparison is missing")
    tokenizer = contract.get("tokenizer")
    if not isinstance(tokenizer, dict) or tokenizer.get("implementation") not in {"official Dia._encode_text", "official Dia._prepare_text_input"} or tokenizer.get("markers") != {"[S1]": 1, "[S2]": 2} or tokenizer.get("ids") != [65, 1, 195, 169, 2] or tokenizer.get("truncation_probe") != [1, 120]:
        raise ValueError("official tokenizer source contract mismatch")
    audio = contract.get("audio_delay_revert")
    if not isinstance(audio, dict) or audio.get("delay_pattern") != [0, 8, 9, 10, 11, 12, 13, 14, 15] or audio.get("shape") != [1, 32, 9] or audio.get("bos_value") != 1026 or audio.get("pad_value") != 1025 or audio.get("valid_prefix_frames") != 17 or audio.get("bos_count") != 92 or audio.get("reverted_pad_count") != 0 or audio.get("revert_index_clamp") is not True or audio.get("revert_out_of_bounds_count") != 0:
        raise ValueError("official delay/revert source contract mismatch")
    sampler = contract.get("sampler")
    calls = sampler.get("multinomial_calls", []) if isinstance(sampler, dict) else []
    branches = sampler.get("branch_results", {}) if isinstance(sampler, dict) else {}
    eos_not_highest = branches.get("eos_not_highest") if isinstance(branches, dict) else None
    eos_highest = branches.get("eos_highest") if isinstance(branches, dict) else None
    calls_are_valid = isinstance(calls, list) and len(calls) == 2 and all(isinstance(call, dict) for call in calls)
    branches_are_valid = isinstance(eos_not_highest, dict) and isinstance(eos_highest, dict)
    if not isinstance(sampler, dict) or sampler.get("implementation") != "official _sample_next_token" or sampler.get("input_shape") != [9, 1025] or sampler.get("top_k") is not None or sampler.get("audio_eos_value") != 1024 or sampler.get("eos_not_highest_probe") is not True or sampler.get("eos_highest_probe") is not True or sampler.get("call_order") != ["temperature scaling", "EOS-not-highest mask", "top-k (disabled)", "top-p (disabled)", "softmax", "torch.multinomial", "selected channel ids"] or not calls_are_valid or not branches_are_valid or calls[0].get("branch") != "eos_not_highest" or calls[1].get("branch") != "eos_highest" or any(call.get("shape") != [9, 1025] or call.get("num_samples") != 1 for call in calls) or any(value != 0.0 for value in calls[0].get("eos_probability", [])) or not calls[1].get("eos_probability") or calls[1]["eos_probability"][0] <= 0.0 or sampler.get("selected") != [0] * 9 or eos_not_highest.get("selected") != [0] * 9 or not isinstance(eos_highest.get("selected"), list) or eos_highest["selected"][0:1] != [1024] or eos_highest["selected"][1:] != [0] * 8:
        raise ValueError("official sampler source contract mismatch")
    generation = contract.get("generation")
    stop = generation.get("stop", {}) if isinstance(generation, dict) else {}
    if not isinstance(generation, dict) or generation.get("delay_pattern") != [0, 8, 9, 10, 11, 12, 13, 14, 15] or generation.get("sampler_call_sites") != 0 or generation.get("sampler_call_sites_generate") != 0 or generation.get("sampler_call_sites_decoder_step") != 1 or generation.get("decoder_step_method") != "official Dia._decoder_step" or stop.get("eos_detected_variable") != "eos_detected_Bx" or stop.get("countdown_variable") != "eos_countdown_Bx" or stop.get("initial_countdown") != -1 or stop.get("countdown_start") != "eos_countdown_Bx[start_countdown_mask_Bx] = max_delay_pattern" or stop.get("max_delay_pattern") != 15 or stop.get("drain_steps") != 15 or stop.get("staggered_eos_pad") is not True or stop.get("source_checks") != {"eos_mask": "step_after_eos_Bx_ == delay_pattern_Cx_", "pad_mask": "step_after_eos_Bx_ > delay_pattern_Cx_", "countdown_decrement": "eos_countdown_Bx[padding_mask_Bx] -= 1"} or "extra_steps_after_eos" in stop:
        raise ValueError("official generation schedule/stop source contract mismatch")


def validate(root: Path, expected_head: str | None = None, approval_sha256: str | None = None) -> None:
    import numpy as np
    require_canonical_existing_path(root)
    if not (root / "manifest.json").is_file() or (root / "manifest.json").is_symlink():
        raise ValueError("evidence directory/manifest is missing or symlinked")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"), object_pairs_hook=unique_pairs)
    if expected_head is not None and manifest.get("expected_head") != expected_head:
        raise ValueError("evidence expected HEAD does not match caller")
    if approval_sha256 is not None and manifest.get("approval_sha256") != approval_sha256:
        raise ValueError("evidence approval SHA does not match caller")
    if manifest.get("format") != "vokra-dia-1-6b-official-reference-v1" or manifest.get("status") != "REFERENCE_COMPLETE":
        raise ValueError("manifest is not a completed official-reference packet")
    if manifest.get("native_status") != "BLOCKED_UNTIL_VAST_AND_APPLE_EVIDENCE" or manifest.get("publication") != "NO_UPLOAD":
        raise ValueError("native/public status drift")
    if manifest.get("comparison_status") != "NOT_RUN_OFFICIAL_ONLY":
        raise ValueError("reference-only packet must say native comparison was not run")
    require_reference_project(manifest.get("reference_project"))
    require_source_contract(manifest.get("source_contract"))
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
    parser.add_argument("--expected-head")
    parser.add_argument("--approval-sha256")
    args = parser.parse_args()
    if args.self_test:
        if args.evidence is not None or args.expected_head is not None or args.approval_sha256 is not None:
            parser.error("--self-test accepts no other arguments")
        assert re.fullmatch(r"[0-9a-f]{40}", "0" * 40)
        assert not re.fullmatch(r"[0-9a-f]{40}", "X" * 40)
        assert re.fullmatch(r"[0-9a-f]{64}", "0" * 64)
        assert not re.fullmatch(r"[0-9a-f]{64}", "X" * 64)
        try:
            require_canonical_existing_path(Path("."))
            raise AssertionError("relative evidence path accepted")
        except ValueError:
            pass
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
            require_source_contract({"format": SOURCE_CONTRACT_FORMAT, "status": "SOURCE_CONTRACT_COMPLETE_MODEL_FREE"})
            raise AssertionError("incomplete source-only contract accepted")
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
    if args.expected_head is None or args.approval_sha256 is None:
        parser.error("expected HEAD and approval SHA are required")
    if not re.fullmatch(r"[0-9a-f]{40}", args.expected_head):
        parser.error("expected_head must be lowercase 40-hex")
    if not re.fullmatch(r"[0-9a-f]{64}", args.approval_sha256):
        parser.error("approval_sha256 must be lowercase 64-hex")
    validate(args.evidence, args.expected_head, args.approval_sha256)
    print("Dia reference evidence validation: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
