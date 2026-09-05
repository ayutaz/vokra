#!/usr/bin/env bash
# VAST-only CSM-1B reference/binding gate.
#
# This worker intentionally stops before native execution until a converter
# produces a complete, authenticated CSM+Mimi GGUF and an accepted native CPU
# baseline.  It is therefore useful now as a staging contract and becomes a
# success-capable worker when those two artifacts exist; it never relabels the
# historical CSM-core-only GGUF as a composite.
set -euo pipefail

ROOT="${VOKRA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
PARITY="$ROOT/tools/parity"
REFERENCE="$PARITY/csm_1b_dump_reference.py"
REFERENCE_PROJECT="$PARITY/csm_1b_reference"
REFERENCE_LOCK_SHA256="62b70ae227b81a2eda59716c2a613f8322405abbf352dc74a5774ffa541a75bc"
UV=(uv run --no-sync --frozen --project "$REFERENCE_PROJECT" --python 3.12 python)
MIN_MEM_KIB=$((128 * 1024 * 1024))
MIN_SCRATCH_KIB=$((40 * 1024 * 1024))

die() { printf '[csm-1b-vast] ERROR: %s\n' "$*" >&2; exit 2; }

self_test() {
  local fail=0 token root py gate_line sync_line reference_line
  root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
  py=python
  for token in sesame/csm-1b c92a71e1c419772e25be7dc14d952c2521a740ab \
    945727948c1143a10ac6f7d811aa58bb0d126b5b csm_1b_inspect.py \
    csm_1b_dump_reference.py REFERENCE_EVIDENCE_COMPLETE \
    COMPLETE_COMPOSITE_GGUF complete-artifact-manifest.json ACCEPTED_VAST_CPU_BASELINE BLOCKED_NATIVE_BINDING \
    VOKRA_PUBLISH_ON_VAST findmnt CARGO_BUILD_JOBS=1 NO_UPLOAD NOT_RUN_OFFICIAL_ONLY \
    apply_chat_template audio_kwargs caller-owned NumPy librosa/soxr wav-pcm16-le pytorch-cpu uv.lock "$REFERENCE_LOCK_SHA256" BLOCKED_LICENSE_METADATA_REVIEW REVIEWED_LICENSE_AUDIT_COMPLETE source_transformers_requirement source_huggingface_hub_requirement isolated_transformers_pin isolated_huggingface_hub_pin GHSA-xrqw-3rrv-vx5w BLOCKED_UNVERIFIED_API_SMOKE --no-sync "uv sync --project" 2051 collection_status decoded_frame_codes.u32le; do
    grep -Fq -- "$token" "$0" || { echo "self-test missing $token" >&2; fail=1; }
  done
  if grep -En '(^|[;&|][[:space:]]*)git[[:space:]]+push|(^|[;&|][[:space:]]*)(curl|wget|huggingface-cli)[[:space:]]' "$0" >/dev/null; then
    echo 'self-test found publication/download command' >&2; fail=1
  fi
  if grep -En '(^|[;&|][[:space:]]*)(python|python3|pip)([[:space:]]|$)' "$0" >/dev/null; then
    echo 'self-test found bare Python command' >&2; fail=1
  fi
  gate_line="$(grep -n 'REFERENCE" --dependency-gate || die' "$0" | tail -1 | cut -d: -f1)"
  sync_line="$(grep -n '^uv sync --project' "$0" | tail -1 | cut -d: -f1)"
  reference_line="$(grep -n '^  --snapshot' "$0" | tail -1 | cut -d: -f1)"
  if [[ -z "$gate_line" || -z "$sync_line" || -z "$reference_line" || "$gate_line" -ge "$sync_line" || "$sync_line" -ge "$reference_line" ]]; then
    echo 'self-test sync must follow the affirmative gate and precede reference execution' >&2; fail=1
  fi
  UV_CACHE_DIR="${UV_CACHE_DIR:-/private/tmp/csm-uv-cache}" uv run --no-sync --frozen --project "$root/tools/parity" --python 3.12 "$py" "$root/tools/parity/csm_1b_dump_reference.py" --self-test || fail=1
  (( fail == 0 )) && echo 'run-csm-1b-validation.sh self-test: OK' || return 1
}

if [[ "${1:-}" == --self-test ]]; then
  [[ $# == 1 ]] || die '--self-test accepts no arguments'
  self_test
  exit 0
fi

[[ "${VOKRA_PUBLISH_ON_VAST:-0}" == 1 ]] || die 'VOKRA_PUBLISH_ON_VAST=1 is required'
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || die 'Linux x86_64 VAST is required'
memory="$(awk '$1 == "MemTotal:" {print $2; exit}' /proc/meminfo)"
[[ "$memory" =~ ^[0-9]+$ && "$memory" -ge "$MIN_MEM_KIB" ]] || die 'at least 128 GiB RAM is required'
work_dir="${VOKRA_CSM_WORK_DIR:-/dev/shm/vokra-csm-1b-validation}"
work_parent="$(dirname "$work_dir")"
[[ "$(findmnt -T "$work_parent" -no FSTYPE 2>/dev/null || true)" == tmpfs ]] || die 'work parent must be tmpfs'
scratch="$(df -Pk "$work_parent" | awk 'NR == 2 {print $4}')"
[[ "$scratch" =~ ^[0-9]+$ && "$scratch" -ge "$MIN_SCRATCH_KIB" ]] || die 'at least 40 GiB free scratch is required'
[[ ! -e "$work_dir" || -d "$work_dir" ]] || die 'work directory is not a directory'
[[ ! -e "$work_dir" || -z "$(find "$work_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]] || die 'work directory must be empty'
for tool in git uv sha256sum findmnt; do command -v "$tool" >/dev/null 2>&1 || die "missing tool: $tool"; done
[[ -f "$REFERENCE_PROJECT/pyproject.toml" && -f "$REFERENCE_PROJECT/uv.lock" && -f "$ROOT/scripts/publish/vast-ai/run-csm-1b-inspection.sh" && -f "$REFERENCE" ]] || die 'dedicated locked CSM reference project is missing'
[[ "$(sha256sum "$REFERENCE_PROJECT/uv.lock" | awk '{print $1}')" == "$REFERENCE_LOCK_SHA256" ]] || die 'dedicated CSM reference uv.lock identity mismatch'
"${UV[@]}" "$REFERENCE" --dependency-gate || die 'CSM dependency/license gate is not explicitly approved'
uv sync --project "$REFERENCE_PROJECT" --frozen --python 3.12
[[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] || die 'clean checkout required'
[[ -n "${CSM_INSPECTION_BUNDLE:-}" && -d "$CSM_INSPECTION_BUNDLE" ]] || die 'CSM_INSPECTION_BUNDLE is required'
[[ -f "$CSM_INSPECTION_BUNDLE/evidence/manifest.json" ]] || die 'authenticated inspection manifest is required in CSM_INSPECTION_BUNDLE'
[[ -n "${CSM_REFERENCE_PACKET:-}" && -f "$CSM_REFERENCE_PACKET" ]] || die 'CSM_REFERENCE_PACKET is required'
mkdir -p "$work_dir"; work_dir="$(cd "$work_dir" && pwd)"; evidence="$work_dir/evidence"; mkdir -p "$evidence"
export CARGO_BUILD_JOBS=1
cp -- "$CSM_REFERENCE_PACKET" "$evidence/packet.json"
cp -- "$CSM_INSPECTION_BUNDLE/evidence/manifest.json" "$evidence/inspection-manifest.json"
"${UV[@]}" - "$CSM_INSPECTION_BUNDLE/evidence/manifest.json" <<'PY'
import json, sys
def pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise SystemExit(f"duplicate inspection manifest key: {key}")
        result[key] = value
    return result
manifest = json.load(open(sys.argv[1], encoding="utf-8"), object_pairs_hook=pairs)
for key, expected in {
    "status": "BLOCKED",
    "evidence_stage": "INSPECTION_ONLY",
    "composite_status": "BLOCKED_ROLE_MAPPING_AND_PARITY",
    "comparison_status": "NOT_RUN_OFFICIAL_ONLY",
    "publication": "NO_UPLOAD",
    "inspection_status": "AUTHENTICATED_EVIDENCE_COMPLETE",
    "collection_status": "AUTHENTICATED",
}.items():
    if manifest.get(key) != expected:
        raise SystemExit(f"inspection manifest is not authenticated/fail-closed: {key}")
PY

"${UV[@]}" "$REFERENCE" \
  --snapshot "$CSM_INSPECTION_BUNDLE/model" \
  --transformers "$CSM_INSPECTION_BUNDLE/transformers" \
  --packet "$CSM_REFERENCE_PACKET" \
  --inspection-manifest "$CSM_INSPECTION_BUNDLE/evidence/manifest.json" \
  --output "$evidence/reference" >"$work_dir/reference.log" 2>&1 || die 'official reference failed'
"${UV[@]}" - "$evidence/reference/manifest.json" "$CSM_REFERENCE_PACKET" "$REFERENCE_PROJECT/uv.lock" <<'PY'
import hashlib, json, math, re, struct, sys, tomllib
from pathlib import Path
def pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise SystemExit(f"duplicate reference manifest key: {key}")
        result[key] = value
    return result
manifest = json.loads(open(sys.argv[1], encoding="utf-8").read(), object_pairs_hook=pairs)
if manifest.get("status") != "BLOCKED" or manifest.get("reference_status") != "REFERENCE_EVIDENCE_COMPLETE" or manifest.get("collection_status") != "AUTHENTICATED" or manifest.get("comparison_status") != "NOT_RUN_OFFICIAL_ONLY" or manifest.get("native_status") != "BLOCKED_NATIVE_BINDING" or manifest.get("cpu_status") != "UNSUPPORTED" or manifest.get("metal_status") != "BLOCKED_BY_CPU" or manifest.get("publication") != "NO_UPLOAD":
    raise SystemExit("official reference evidence is incomplete")
if manifest.get("generation", {}).get("do_sample") is not False or manifest.get("generation", {}).get("depth_decoder_do_sample") is not False:
    raise SystemExit("reference is not deterministic greedy")
generation = manifest.get("generation", {})
if generation.get("logit_selection_contract") != "backbone_processed_scores_argmax_selected_codebook0; backbone_raw_logits_recorded; depth_raw_hook_argmax_exact_trace":
    raise SystemExit("reference logit selection contract is not source-shaped")
if generation.get("depth_selection_contract") != "depth_raw_hook_logits_argmax_matches_codebooks1_to31; processed_depth_scores_unavailable":
    raise SystemExit("depth selection contract is not the exact raw-hook trace")
generation_config_contract = generation.get("generation_config_contract")
if not isinstance(generation_config_contract, dict) or generation_config_contract.get("reference_overrides") != {"do_sample": False, "depth_decoder_do_sample": False} or generation_config_contract.get("final_generation_semantics") != "reference_overrides_are_applied_at_model_generate_boundary":
    raise SystemExit("generation config source/override contract is missing")
if not re.fullmatch(r"[0-9a-f]{64}", generation.get("generation_config_sha256", "")):
    raise SystemExit("authenticated generation_config identity is missing")
if manifest.get("model", {}).get("repository") != "sesame/csm-1b" or manifest.get("model", {}).get("revision") != "c92a71e1c419772e25be7dc14d952c2521a740ab":
    raise SystemExit("reference model identity is not pinned")
if manifest.get("transformers", {}).get("commit") != "945727948c1143a10ac6f7d811aa58bb0d126b5b" or manifest.get("inspection_identity", {}).get("transformers_version") != "4.52.1":
    raise SystemExit("reference Transformers version/commit is not pinned")
environment = manifest.get("reference_environment", {})
if not isinstance(environment, dict) or not re.fullmatch(r"[0-9a-f]{64}", environment.get("lock_sha256", "")) or environment.get("python") != "3.12":
    raise SystemExit("dedicated reference lock identity is missing")
if environment.get("lock_sha256") != "62b70ae227b81a2eda59716c2a613f8322405abbf352dc74a5774ffa541a75bc":
    raise SystemExit("dedicated reference lock SHA does not match the reviewed lock")
if environment.get("selection_status") != "REVIEWED_ADAPTED_REFERENCE_ENVIRONMENT_NOT_UPSTREAM_REQUIREMENTS":
    raise SystemExit("reference package selection is missing its adapted-environment disclosure")
if environment.get("torch_index") != "https://download.pytorch.org/whl/cpu":
    raise SystemExit("reference is not bound to the official PyTorch CPU index")
if environment.get("source_transformers_requirement") != "transformers==4.52.1" or environment.get("source_huggingface_hub_requirement") != "huggingface-hub>=0.30,<1.0" or environment.get("transformers_security_advisory") != "GHSA-xrqw-3rrv-vx5w" or environment.get("transformers_security_patched_minimum") != "5.10.0" or environment.get("isolated_transformers_pin") != "5.10.4" or environment.get("isolated_huggingface_hub_pin") != "1.5.0" or environment.get("transformers_compatibility_status") != "BLOCKED_UNVERIFIED_API_SMOKE":
    raise SystemExit("reference Transformers security/provenance metadata is incomplete")
lock_document = tomllib.loads(Path(sys.argv[3]).read_text(encoding="utf-8"))
lock_rows = []
identities = set()
for package in lock_document.get("package", []):
    if not isinstance(package, dict) or set(package) - {"name", "version", "source", "resolution-markers", "dependencies", "sdist", "wheels", "metadata"}:
        raise SystemExit("uv.lock contains an unknown package-row field")
    markers = package.get("resolution-markers", [])
    if not isinstance(package.get("name"), str) or not isinstance(package.get("version"), str) or not isinstance(package.get("source"), dict) or not isinstance(markers, list) or not all(isinstance(marker, str) for marker in markers):
        raise SystemExit("uv.lock package identity is malformed")
    row = {"name": package["name"], "version": package["version"], "source": package["source"], "resolution_markers": markers}
    identity = json.dumps(row, sort_keys=True, separators=(",", ":"))
    if identity in identities:
        raise SystemExit("duplicate uv.lock package identity")
    identities.add(identity)
    lock_rows.append(row)
lock_rows.sort(key=lambda row: json.dumps(row, sort_keys=True, separators=(",", ":")))
lock_rows_hash = hashlib.sha256(json.dumps(lock_rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
if environment.get("locked_package_rows") != lock_rows or environment.get("locked_package_rows_sha256") != lock_rows_hash:
    raise SystemExit("reference package rows are not bound to the exact uv.lock")
if any(row["name"] in {"soundfile", "librosa", "soxr"} for row in lock_rows):
    raise SystemExit("forbidden audio dependency remains in the dedicated lock")
audit_expectations = {
    "annotated-doc": ("MIT", "https://pypi.org/pypi/annotated-doc/0.0.5/json"),
    "anyio": ("MIT", "https://pypi.org/pypi/anyio/4.14.2/json"),
    "certifi": ("MPL-2.0", "https://pypi.org/pypi/certifi/2026.7.22/json"),
    "colorama": ("BSD-3-Clause", "https://pypi.org/pypi/colorama/0.4.6/json"),
    "filelock": ("MIT", "https://pypi.org/pypi/filelock/3.32.4/json"),
    "fsspec": ("BSD-3-Clause", "https://pypi.org/pypi/fsspec/2026.7.0/json"),
    "hf-xet": ("Apache-2.0", "https://pypi.org/pypi/hf-xet/1.6.0/json"),
    "h11": ("MIT", "https://pypi.org/pypi/h11/0.16.0/json"),
    "httpcore": ("BSD-3-Clause", "https://pypi.org/pypi/httpcore/1.0.9/json"),
    "httpx": ("BSD-3-Clause", "https://pypi.org/pypi/httpx/0.28.1/json"),
    "huggingface-hub": ("Apache-2.0", "https://pypi.org/pypi/huggingface-hub/1.5.0/json"),
    "idna": ("BSD-3-Clause", "https://pypi.org/pypi/idna/3.19/json"),
    "jinja2": ("BSD-3-Clause", "https://pypi.org/pypi/jinja2/3.1.6/json"),
    "markupsafe": ("BSD-3-Clause", "https://pypi.org/pypi/markupsafe/3.0.3/json"),
    "markdown-it-py": ("MIT", "https://pypi.org/pypi/markdown-it-py/4.2.0/json"),
    "mdurl": ("MIT", "https://pypi.org/pypi/mdurl/0.1.2/json"),
    "mpmath": ("BSD", "https://pypi.org/pypi/mpmath/1.3.0/json"),
    "networkx": ("BSD-3-Clause", "https://pypi.org/pypi/networkx/3.6.1/json"),
    "numpy": ("BSD-3-Clause + bundled runtime notices (GPL/LGPL)", "https://pypi.org/pypi/numpy/2.2.6/json"),
    "packaging": ("Apache-2.0 AND BSD-2-Clause", "https://pypi.org/pypi/packaging/26.3/json"),
    "pyyaml": ("MIT", "https://pypi.org/pypi/pyyaml/6.0.3/json"),
    "pygments": ("BSD-2-Clause", "https://pypi.org/pypi/pygments/2.21.0/json"),
    "regex": ("Apache-2.0", "https://pypi.org/pypi/regex/2026.7.19/json"),
    "rich": ("MIT", "https://pypi.org/pypi/rich/15.0.0/json"),
    "safetensors": ("Apache-2.0", "https://pypi.org/pypi/safetensors/0.8.0/json"),
    "setuptools": ("MIT", "https://pypi.org/pypi/setuptools/84.0.0/json"),
    "shellingham": ("ISC", "https://pypi.org/pypi/shellingham/1.5.4/json"),
    "sympy": ("BSD-3-Clause", "https://pypi.org/pypi/sympy/1.14.0/json"),
    "tokenizers": ("Apache-2.0", "https://pypi.org/pypi/tokenizers/0.22.2/json"),
    "torch": ("BSD-3-Clause; official CPU index", "https://download.pytorch.org/whl/cpu"),
    "tqdm": ("MPL-2.0 AND MIT", "https://pypi.org/pypi/tqdm/4.70.0/json"),
    "transformers": ("Apache-2.0", "https://pypi.org/pypi/transformers/5.10.4/json"),
    "typer": ("MIT", "https://pypi.org/pypi/typer/0.27.2/json"),
    "typing-extensions": ("PSF-2.0; blocked by owner policy", "https://pypi.org/pypi/typing-extensions/4.16.0/json"),
    "vokra-csm-1b-reference": ("PROJECT_METADATA_ONLY", "tools/parity/csm_1b_reference/pyproject.toml"),
}
audit_rows = environment.get("license_audit_rows")
expected_audit = [{**row, "license": audit_expectations[row["name"]][0], "license_source": audit_expectations[row["name"]][1]} for row in lock_rows if row["name"] in audit_expectations]
if audit_rows != expected_audit or environment.get("license_audit_status") != "BLOCKED_OWNER_POLICY_AND_NATIVE_NOTICE_REVIEW" or environment.get("license_audit_rows_sha256") != hashlib.sha256(json.dumps(expected_audit, sort_keys=True, separators=(",", ":")).encode()).hexdigest():
    raise SystemExit("full locked dependency license audit is missing or tampered")
locked_third_party = {row["name"] for row in lock_rows if row["name"] != "vokra-csm-1b-reference"}
audited_third_party = set(audit_expectations) - {"vokra-csm-1b-reference"}
if audited_third_party != locked_third_party or "vokra-csm-1b-reference" not in audit_expectations:
    raise SystemExit("locked dependency license audit does not cover every package")
if environment.get("packages", {}).get("numpy") != "2.2.6" or environment.get("packages", {}).get("torch_distribution") not in {"2.7.1", "2.7.1+cpu"} or environment.get("packages", {}).get("transformers") != "5.10.4":
    raise SystemExit("dedicated reference package versions are not pinned")
torch_runtime = environment.get("packages", {}).get("torch_runtime")
if not isinstance(torch_runtime, str) or not re.fullmatch(r"2\.7\.1(?:\+[^+ ]+)?", torch_runtime):
    raise SystemExit("runtime torch version is missing or incompatible with the locked distribution")
identity = manifest.get("inspection_identity", {})
if identity.get("source_repository") != "https://github.com/SesameAILabs/csm.git" or identity.get("source_revision") != "8f6d947a26f6301deec9696f9bfb28e9e2e0d7d5" or identity.get("transformers_commit") != "945727948c1143a10ac6f7d811aa58bb0d126b5b":
    raise SystemExit("reference source/Transformers inspection identity is incomplete")
tokenizer = manifest.get("tokenizer", {})
if tokenizer.get("tokenizer_json_git_blob_sha1") != "8de5df033b78de76dbe15fdd8b934678b5017aaf" or not re.fullmatch(r"[0-9a-f]{64}", tokenizer.get("tokenizer_json_sha256", "")):
    raise SystemExit("reference tokenizer identity is incomplete")
if manifest.get("generation", {}).get("input_boundary") != "official processor.apply_chat_template(conversation, tokenize=False) then official CSM processor(text=rendered_prompt, audio=caller_numpy, sampling_rate=24000, return_tensors=pt)":
    raise SystemExit("reference did not use the official chat-template input boundary")
if manifest.get("generation", {}).get("processor_call") != {"method": "apply_chat_template_then_official_processor", "chat_template_kwargs": {"tokenize": False}, "processor_kwargs": {"sampling_rate": 24000, "return_tensors": "pt"}, "audio_input": "authenticated caller-owned NumPy arrays", "audio_argument": "None when audio_placeholder_count=0; otherwise the ordered non-empty authenticated array list", "adapter": "pinned_upstream_ProcessorMixin_boundary; no_reference_mirror"}:
    raise SystemExit("reference processor call arguments are not the official boundary")
if manifest.get("generation", {}).get("processor_source") != {"path": "src/transformers/processing_utils.py", "git_blob_sha1": "8dbc210fbcd0b4e9b741427f6f4d74d9ecbf7913", "markers": ["if not is_batched:\n            prompt = prompt[0]", "out = self(\n                text=prompt,", "audio=batch_audios if batch_audios else None"], "non_batched_text_argument": "text=prompt (a string after prompt=prompt[0])"}:
    raise SystemExit("reference does not bind the pinned non-batched ProcessorMixin boundary")
if manifest.get("pcm_semantics") != "Transformers codec-decoded PCM before the source CSM watermark/resample stage; not final watermarked PCM":
    raise SystemExit("PCM semantics are not honestly labelled")
ref = Path(sys.argv[1]).parent
packet_path = Path(sys.argv[2]).resolve()
packet = json.loads(packet_path.read_text(encoding="utf-8"), object_pairs_hook=pairs)
messages = packet.get("messages")
if not isinstance(messages, list) or not messages or not isinstance(messages[-1], dict) or messages[-1].get("role") != "0":
    raise SystemExit("reference conversation boundary is invalid")
packet_paths = []
for message in messages:
    if not isinstance(message, dict) or set(message) != {"role", "content"} or message.get("role") not in {"0", "1"} or not isinstance(message.get("content"), list):
        raise SystemExit("reference conversation message schema is invalid")
    for item in message["content"]:
        if not isinstance(item, dict) or item.get("type") not in {"text", "audio"}:
            raise SystemExit("reference conversation content item is invalid")
        if item["type"] == "audio":
            if set(item) != {"type", "path"} or not isinstance(item["path"], str) or Path(item["path"]).suffix.lower() != ".wav" or Path(item["path"]).is_absolute() or ".." in Path(item["path"]).parts:
                raise SystemExit("reference audio content path is unsafe")
            packet_paths.append(item["path"])
        elif set(item) != {"type", "text"} or not isinstance(item["text"], str) or not item["text"] or "\x00" in item["text"]:
            raise SystemExit("reference text content item is invalid")
    if message is messages[-1] and (any(item.get("type") == "audio" for item in message["content"]) or not any(item.get("type") == "text" for item in message["content"])):
        raise SystemExit("last conversation message must be target text without audio")
audio_rows = manifest.get("audio_inputs")
if not isinstance(audio_rows, list) or len(audio_rows) != len(packet_paths):
    raise SystemExit("reference audio identity evidence is incomplete")
if manifest.get("generation", {}).get("audio_placeholder_count") != len(packet_paths) or manifest.get("generation", {}).get("decoded_audio_count") != 1:
    raise SystemExit("reference audio placeholder/decoded output cardinality is invalid")
expected_conversation_paths = [str((packet_path.parent / raw_path).resolve()) for raw_path in packet_paths]
if manifest.get("generation", {}).get("conversation_audio_paths") != expected_conversation_paths:
    raise SystemExit("official conversation audio paths are not bound to packet paths")
for raw_path, row in zip(packet_paths, audio_rows):
    if not isinstance(raw_path, str) or Path(raw_path).is_absolute() or ".." in Path(raw_path).parts or not isinstance(row, dict) or row.get("path") != raw_path or row.get("format") != "wav-pcm16-le" or row.get("sample_rate_hz") != 24000:
        raise SystemExit("unsafe or mismatched caller audio path evidence")
    audio_path = (packet_path.parent / raw_path).resolve()
    try:
        audio_path.relative_to(packet_path.parent)
    except ValueError:
        raise SystemExit("caller audio path escapes packet directory")
    if not audio_path.is_file() or audio_path.is_symlink() or row.get("bytes") != audio_path.stat().st_size or hashlib.sha256(audio_path.read_bytes()).hexdigest() != row.get("sha256"):
        raise SystemExit("caller audio identity mismatch")
required = {
    "processor_input_ids.u32le", "processor_attention_mask.u32le", "generated_frame_codes.u32le", "decoded_frame_codes.u32le",
    "backbone_logits.f32le", "backbone_scores.f32le", "backbone_hidden_last.f32le",
    "depth_decoder_logits.f32le", "official_pcm_pre_watermark.f32le",
}
rows = manifest.get("artifacts")
if not isinstance(rows, list) or len(rows) != len(required) or {row.get("path") for row in rows} != required:
    raise SystemExit("reference artifact set is incomplete or duplicated")
for row in rows:
    path = row.get("path")
    if not isinstance(path, str) or Path(path).name != path or path.startswith("."):
        raise SystemExit(f"unsafe reference artifact path: {path!r}")
    artifact = ref / path
    if not artifact.is_file() or artifact.is_symlink() or artifact.stat().st_size <= 0:
        raise SystemExit(f"reference artifact missing/empty: {path}")
    if row.get("bytes") != artifact.stat().st_size or not re.fullmatch(r"[0-9a-f]{64}", row.get("sha256", "")) or row.get("dtype") not in {"u32", "float32"}:
        raise SystemExit(f"reference artifact metadata malformed: {path}")
    h = hashlib.sha256(artifact.read_bytes()).hexdigest()
    if h != row["sha256"]:
        raise SystemExit(f"reference artifact hash mismatch: {path}")
    shape = row.get("shape")
    if not isinstance(shape, list) or not shape or any(not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0 for dim in shape):
        raise SystemExit(f"reference artifact shape malformed: {path}")
    elements = 1
    for dim in shape:
        elements *= dim
    if row["bytes"] != elements * 4:
        raise SystemExit(f"reference artifact bytes do not match shape: {path}")
    if row["dtype"] == "float32":
        values = struct.unpack("<" + "f" * (row["bytes"] // 4), (ref / path).read_bytes())
        if not all(math.isfinite(value) for value in values):
            raise SystemExit(f"reference artifact contains non-finite values: {path}")
if {p.name for p in ref.iterdir()} != required | {"manifest.json"}:
    raise SystemExit("reference output contains stale or unlisted artifacts")
generation = manifest.get("generation", {})
shapes = {row["path"]: row["shape"] for row in rows}
if shapes["processor_input_ids.u32le"] != generation.get("processor_input_ids_shape") or shapes["processor_attention_mask.u32le"] != generation.get("processor_attention_mask_shape") or shapes["processor_input_ids.u32le"] != shapes["processor_attention_mask.u32le"]:
    raise SystemExit("processor IDs/mask shapes are not recorded and aligned")
mask_row = next(row for row in rows if row["path"] == "processor_attention_mask.u32le")
mask_values = struct.unpack("<" + "I" * (mask_row["bytes"] // 4), (ref / mask_row["path"]).read_bytes())
if any(value not in (0, 1) for value in mask_values):
    raise SystemExit("processor attention_mask is not binary")
codes_shape = shapes["generated_frame_codes.u32le"]
if len(codes_shape) != 3 or codes_shape[0] <= 0 or codes_shape[1] <= 0 or codes_shape[2] != 32:
    raise SystemExit("generated codes must have shape [batch,frames,32]")
batch, frames = codes_shape[:2]
if batch != 1:
    raise SystemExit("reference artifact format is mono and requires batch=1")
decoded_shape = shapes["decoded_frame_codes.u32le"]
if shapes["backbone_logits.f32le"] != [batch, frames, 2051] or shapes["backbone_scores.f32le"] != [batch, frames, 2051] or shapes["backbone_hidden_last.f32le"] != [batch, frames, 2048]:
    raise SystemExit("backbone evidence must have source-shaped raw/scores [batch,frames,2051] and [batch,frames,2048] hidden states")
if shapes["depth_decoder_logits.f32le"] != [frames * 31, batch, 1, 2051]:
    raise SystemExit("depth decoder evidence must have one [batch,1,2051] call per depth code")
code_artifact = next(row for row in rows if row["path"] == "generated_frame_codes.u32le")
codes = list(struct.unpack("<" + "I" * (code_artifact["bytes"] // 4), (ref / code_artifact["path"]).read_bytes()))
if any(code >= 2048 for code in codes):
    raise SystemExit("generated code exceeds the 2048-entry codebook")
decoded_artifact = next(row for row in rows if row["path"] == "decoded_frame_codes.u32le")
decoded_codes = list(struct.unpack("<" + "I" * (decoded_artifact["bytes"] // 4), (ref / decoded_artifact["path"]).read_bytes()))
for index in range(batch):
    frames_for_batch = [codes[(index * frames + frame) * 32:(index * frames + frame + 1) * 32] for frame in range(frames)]
    official_eos = [frame for frame, values in enumerate(frames_for_batch) if all(value == 0 for value in values[:-1])]
    codec_eos = [frame for frame, values in enumerate(frames_for_batch) if all(value == 0 for value in values)]
    if official_eos and official_eos != [frames - 1]:
        raise SystemExit("official 31-codebook EOS is not a final frame")
    if codec_eos and codec_eos != [frames - 1]:
        raise SystemExit("codec 32-codebook EOS is not a final frame")
if generation.get("official_eos_frame_indices") != official_eos or generation.get("codec_eos_frame_indices") != codec_eos:
    raise SystemExit("official/codec EOS evidence differs from generated codes")
if generation.get("official_eos_codebook_count") != 31 or generation.get("codec_eos_codebook_count") != 32:
    raise SystemExit("official/codec EOS codebook cardinalities are not authenticated")
decoded_frames = generation.get("codec_decoded_frame_count_by_batch", [None])[0]
if not isinstance(decoded_frames, int) or decoded_frames != (codec_eos[0] if codec_eos else frames):
    raise SystemExit("codec EOS decoded-frame cutoff is not source-shaped")
if decoded_frames <= 0 or decoded_shape != [batch, decoded_frames, 32] or shapes["official_pcm_pre_watermark.f32le"] != [decoded_frames * 1920]:
    raise SystemExit("PCM is not aligned to the pre-EOS decoded frame count")
if decoded_codes != codes[: batch * decoded_frames * 32]:
    raise SystemExit("decoded code artifact is not the EOS-excluded generated prefix")
depth_row = next(row for row in rows if row["path"] == "depth_decoder_logits.f32le")
depth_values = struct.unpack("<" + "f" * (depth_row["bytes"] // 4), (ref / depth_row["path"]).read_bytes())
for call in range(frames * 31):
    logits = depth_values[call * 2051:(call + 1) * 2051]
    if max(range(2051), key=lambda index: logits[index]) != codes[(call // 31) * 32 + 1 + call % 31]:
        raise SystemExit("depth decoder argmax does not match generated codebook 1..31")
backbone_row = next(row for row in rows if row["path"] == "backbone_scores.f32le")
backbone_values = struct.unpack("<" + "f" * (backbone_row["bytes"] // 4), (ref / backbone_row["path"]).read_bytes())
backbone_vocab = backbone_row["shape"][2]
for frame in range(frames):
    logits = backbone_values[frame * backbone_vocab:(frame + 1) * backbone_vocab]
    if max(range(backbone_vocab), key=lambda index: logits[index]) != codes[frame * 32]:
        raise SystemExit("backbone argmax does not match generated codebook 0")
if generation.get("depth_decoder_call_count") != frames * 31 or generation.get("backbone_hidden_generation_steps") != frames:
    raise SystemExit("reference did not capture exact frame/depth cardinalities")
input_shapes = generation.get("depth_decoder_input_shapes")
if not isinstance(input_shapes, list) or len(input_shapes) != frames * 31 or any(shape != [batch, 2 if index % 31 == 0 else 1] for index, shape in enumerate(input_shapes)):
    raise SystemExit("depth decoder prefill/step input cardinality is missing or incorrect")
input_ids = generation.get("depth_decoder_input_ids")
if not isinstance(input_ids, list) or len(input_ids) != frames * 31 or any(not isinstance(row, list) or len(row) != batch or any(not isinstance(tokens, list) or len(tokens) != input_shapes[index][1] or any(isinstance(value, bool) or not isinstance(value, int) for value in tokens) for tokens in row) for index, row in enumerate(input_ids)):
    raise SystemExit("depth decoder input IDs/call order evidence is missing or malformed")
expected_inputs = []
for frame in range(frames):
    expected_inputs.append([[codes[frame * 32]]])
    expected_inputs[-1][0].insert(0, 0)
    expected_inputs.extend([[[codes[frame * 32 + codebook]]] for codebook in range(1, 31)])
if input_ids != expected_inputs:
    raise SystemExit("depth decoder inputs do not follow placeholder/previous-codebook order")
if generation.get("generated_sequence_shape") != codes_shape or generation.get("decoded_frame_count_by_batch") != [decoded_frames] or generation.get("pcm_samples") != decoded_frames * 1920:
    raise SystemExit("generation relation metadata does not match artifacts")
mapping = generation.get("depth_decoder_frame_codebook_order")
if not isinstance(mapping, list) or len(mapping) != frames * 31 or any(item != {"call_index": index, "frame": index // 31, "codebook": 1 + index % 31} for index, item in enumerate(mapping)):
    raise SystemExit("depth decoder frame/codebook order is missing or reordered")
if generation.get("depth_decoder_call_order") != list(range(generation["depth_decoder_call_count"])):
    raise SystemExit("depth decoder call order evidence is missing or reordered")
PY
cp -- "$evidence/reference/manifest.json" "$evidence/reference-manifest.json"
native_status=BLOCKED_NATIVE_BINDING
native_reason='complete composite GGUF, accepted artifact manifest, and accepted VAST CPU baseline were not supplied; reference evidence is independent of the native gate'
if [[ -n "${CSM_COMPLETE_GGUF:-}" || -n "${CSM_COMPLETE_MANIFEST:-}" || -n "${CSM_ACCEPTED_VAST_CPU_BASELINE:-}" ]]; then
  [[ -n "${CSM_COMPLETE_GGUF:-}" && -f "$CSM_COMPLETE_GGUF" ]] || die 'partial native gate inputs: COMPLETE_COMPOSITE_GGUF is missing'
  [[ -n "${CSM_COMPLETE_MANIFEST:-}" && -f "$CSM_COMPLETE_MANIFEST" ]] || die 'partial native gate inputs: complete-artifact manifest is missing'
  [[ -n "${CSM_ACCEPTED_VAST_CPU_BASELINE:-}" && -f "$CSM_ACCEPTED_VAST_CPU_BASELINE" ]] || die 'partial native gate inputs: ACCEPTED_VAST_CPU_BASELINE is missing'
  cp -- "$CSM_COMPLETE_GGUF" "$evidence/csm-1b-complete.gguf"
  cp -- "$CSM_COMPLETE_MANIFEST" "$evidence/complete-artifact-manifest.json"
  native_reason='complete composite inputs were supplied and authenticated; native/public support remains blocked by the production binder gate'
  "${UV[@]}" - "$evidence/complete-artifact-manifest.json" "$evidence/csm-1b-complete.gguf" <<'PY'
import hashlib, json, re, sys
manifest = json.load(open(sys.argv[1], encoding="utf-8"))
if manifest.get("status") != "ACCEPTED" or manifest.get("artifact_role") != "COMPLETE_CSM_MIMI_COMPOSITE":
    raise SystemExit("complete artifact is not an accepted CSM+Mimi composite")
if manifest.get("model", {}).get("revision") != "c92a71e1c419772e25be7dc14d952c2521a740ab":
    raise SystemExit("complete artifact is not bound to the fixed HF snapshot")
artifact_hash = manifest.get("sha256") or manifest.get("artifact_sha256") or manifest.get("artifact", {}).get("sha256")
if not isinstance(artifact_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", artifact_hash):
    raise SystemExit("complete artifact manifest has no strong SHA-256 binding")
hasher = hashlib.sha256()
with open(sys.argv[2], "rb") as stream:
    for block in iter(lambda: stream.read(1 << 20), b""):
        hasher.update(block)
if hasher.hexdigest() != artifact_hash:
    raise SystemExit("complete artifact SHA-256 does not match its manifest")
PY
  sha256sum "$evidence/csm-1b-complete.gguf" "$evidence/packet.json" > "$evidence/input-sha256.txt"
fi
"${UV[@]}" - "$evidence/validation-manifest.json" "$native_status" "$native_reason" <<'PY'
import json, sys
json.dump({
    "status": "BLOCKED",
    "evidence_stage": "VAST_REFERENCE_AND_NATIVE_GATE",
    "reference_status": "REFERENCE_EVIDENCE_COMPLETE",
    "comparison_status": "NOT_RUN_OFFICIAL_ONLY",
    "native_status": sys.argv[2],
    "cpu_baseline": "ACCEPTED_VAST_CPU_BASELINE_REQUIRED_BY_CALLER",
    "metal_status": "BLOCKED_BY_CPU",
    "publication": "NO_UPLOAD",
    "reason": sys.argv[3],
}, open(sys.argv[1], "w", encoding="utf-8"), indent=2, sort_keys=True)
PY
echo '[csm-1b-vast] official greedy evidence staged; native binding remains BLOCKED_NATIVE_BINDING (NO_UPLOAD).' >&2
exit 2
