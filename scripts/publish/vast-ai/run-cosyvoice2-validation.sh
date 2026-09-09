#!/usr/bin/env bash
set -euo pipefail

# This worker is a VAST-only validation gate.  It deliberately stops at the
# official-reference/binder boundary; a green inspection cannot authorize a
# fake or partial CosyVoice2 GGUF.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
REFERENCE="$ROOT/tools/parity/cosyvoice2_dump_reference.py"
REFERENCE_PROJECT="$ROOT/tools/parity/cosyvoice2_reference"
SOURCE_REVISION="8555549e882236e6541748b1042d95693caa82ba"
MODEL_REVISION="eec1ae6c79877dbd9379285cf8789c9e0879293d"
MATCHA_REVISION="dd9105b34bf2be2230f4aa1e4769fb586a3c824e"
TRANSFORMERS_SOURCE_REQUIREMENT="transformers==4.40.1"
TRANSFORMERS_SECURITY_ADVISORY="GHSA-xrqw-3rrv-vx5w"
TRANSFORMERS_SECURITY_PATCHED_MINIMUM="5.10.0"
ISOLATED_TRANSFORMERS_PIN="5.10.4"
die() { echo "cosyvoice2-vast: ERROR: $*" >&2; exit 2; }

run_dependency_gate() {
  local gate_output gate_status
  set +e
  gate_output="$(UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python "$REFERENCE" --dependency-gate 2>&1)"
  gate_status=$?
  set -e
  printf '%s\n' "$gate_output" >&2
  [[ "$gate_status" == 2 ]] || die "dependency gate returned unexpected status: $gate_status"
  return 1
}

require_absent_path() {
  local target="$1" current
  [[ ! -e "$target" && ! -L "$target" ]] || die 'validation directory must be absent and not a symlink'
  current="$target"
  while [[ "$current" != / && "$current" != . && -n "$current" ]]; do
    [[ ! -L "$current" ]] || die "work path contains a symlink component: $current"
    current="$(dirname "$current")"
  done
}
canonical_uncreated() {
  local target="$1" current="$1" suffix="" parent
  while [[ ! -e "$current" && ! -L "$current" ]]; do
    suffix="/$(basename "$current")$suffix"
    parent="$(dirname "$current")"
    [[ "$parent" != "$current" ]] || break
    current="$parent"
  done
  [[ ! -L "$current" ]] || die "work path contains a symlink component: $current"
  printf '%s%s\n' "$(cd -P "$current" && pwd)" "$suffix"
}
require_disjoint_uncreated() {
  local candidate protected base
  candidate="$(canonical_uncreated "$1")"
  shift
  for protected in "$@"; do
    base="$(cd -P "$protected" && pwd)" || die "protected path is not accessible: $protected"
    [[ "$candidate" != "$base" && "$candidate" != "$base/"* && "$base" != "$candidate/"* ]] || die "work path overlaps protected path: $protected"
  done
}

self_test() {
  local fail=0 token
  for token in "$SOURCE_REVISION" "$MODEL_REVISION" "$MATCHA_REVISION" "$TRANSFORMERS_SOURCE_REQUIREMENT" "$TRANSFORMERS_SECURITY_ADVISORY" "$TRANSFORMERS_SECURITY_PATCHED_MINIMUM" "$ISOLATED_TRANSFORMERS_PIN" "BLOCKED_UNVERIFIED_API_SMOKE" "AUTHENTICATED_REFERENCE_EVIDENCE" "NOT_IMPLEMENTED_FAIL_CLOSED" "NO_UPLOAD" "qwen_prompt_embeddings" "flow_rand_noise_full" "flow_rand_noise_slice" "ras_calls" "ras_pre_ras_probability" "ras_nucleus_probability" "official_output_pcm" "execution_id" "llm_calls" "15000" "zero_shot" "cross_lingual" "prompt_wav_sha256" "cfm_time_grid" "ignored_eos"; do
    grep -Fq -- "$token" "$REFERENCE" "$REFERENCE_PROJECT/pyproject.toml" "$0" || { echo "missing contract: $token" >&2; fail=1; }
  done
  if grep -En 'git[[:space:]]+push|upload\.sh|publish-one\.sh|--push|--upload|OFFICIAL_ADAPTER_AUTHENTICATED_NOT_RUN' "$0" | grep -v 'grep -En' >/dev/null; then fail=1; fi
  # The dedicated lock is required for a real VAST run. Local self-test uses
  # the already-locked sidecar only for stdlib/schema checks because this
  # machine must not resolve the heavy reference environment.
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python "$REFERENCE" --self-test || fail=1
  (( fail == 0 )) || return 1
  echo 'run-cosyvoice2-validation.sh self-test: OK'
}

if [[ "${1:-}" == --self-test ]]; then
  [[ $# == 1 ]] || die '--self-test accepts no arguments'
  self_test
  exit 0
fi
[[ $# == 0 ]] || die 'usage: run-cosyvoice2-validation.sh [--self-test]'
run_dependency_gate || die 'CosyVoice2 dependency/license closure is unresolved; no source/model acquisition or reference execution is permitted'
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || die 'VAST requires Linux x86_64'
[[ "${VOKRA_PUBLISH_ON_VAST:-0}" == 1 ]] || die 'VOKRA_PUBLISH_ON_VAST=1 is absent'
[[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] || die 'checkout must be clean'
for command in git uv awk find df findmnt; do command -v "$command" >/dev/null || die "missing tool: $command"; done
[[ -f "$REFERENCE_PROJECT/uv.lock" ]] || die 'dedicated CosyVoice2 uv.lock is absent; generate and review it on VAST before execution'
mem_kib="$(awk '$1=="MemTotal:"{print $2;exit}' /proc/meminfo)"
[[ "$mem_kib" =~ ^[0-9]+$ && $mem_kib -ge $((128*1024*1024)) ]] || die '128 GiB memory guard failed'
[[ "$(findmnt -T /dev/shm -n -o FSTYPE 2>/dev/null)" == tmpfs ]] || die '/dev/shm must be tmpfs'
free_kib="$(df -Pk /dev/shm | awk 'NR==2{print $4}')"
[[ "$free_kib" =~ ^[0-9]+$ && $free_kib -ge $((32*1024*1024)) ]] || die 'tmpfs disk guard failed'

WORK="/dev/shm/vokra-cosyvoice2-validation"
require_absent_path "$WORK"
require_disjoint_uncreated "$WORK" "$ROOT" "$ROOT/tools/parity" "$REFERENCE_PROJECT"
mkdir -p "$WORK/source" "$WORK/evidence"
# The evidence log intentionally receives each lifecycle command separately.
# shellcheck disable=SC2129
git clone --filter=blob:none https://github.com/FunAudioLLM/CosyVoice.git "$WORK/source/repo" >>"$WORK/evidence/validation.log" 2>&1
git -C "$WORK/source/repo" checkout --detach "$SOURCE_REVISION" >>"$WORK/evidence/validation.log" 2>&1
git -C "$WORK/source/repo" submodule update --init --recursive >>"$WORK/evidence/validation.log" 2>&1
git clone --filter=blob:none https://github.com/shivammehta25/Matcha-TTS.git "$WORK/matcha" >>"$WORK/evidence/validation.log" 2>&1
git -C "$WORK/matcha" checkout --detach "$MATCHA_REVISION" >>"$WORK/evidence/validation.log" 2>&1
uv run --frozen --project "$REFERENCE_PROJECT" --python 3.12 python - "$WORK/model" <<'PY' >>"$WORK/evidence/validation.log" 2>&1
from huggingface_hub import snapshot_download
import sys
snapshot_download("FunAudioLLM/CosyVoice2-0.5B", revision="eec1ae6c79877dbd9379285cf8789c9e0879293d", local_dir=sys.argv[1])
PY
uv run --frozen --project "$REFERENCE_PROJECT" --python 3.12 python - "$WORK/source/repo/asset/zero_shot_prompt.wav" "$WORK/input.json" <<'PY' >>"$WORK/evidence/validation.log" 2>&1
import hashlib, json, sys, wave
wav = sys.argv[1]
with wave.open(wav, "rb") as stream:
    if (stream.getnchannels(), stream.getsampwidth(), stream.getframerate()) != (1, 2, 16000):
        raise RuntimeError("fixed source prompt must be mono 16 kHz PCM16")
    samples = stream.getnframes()
sha = hashlib.sha256(open(wav, "rb").read()).hexdigest()
json.dump({"mode":"zero_shot", "target_text":"收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。", "prompt_text":"希望你以后能够做的比我还好呦。", "prompt_wav":"asset/zero_shot_prompt.wav", "prompt_wav_sha256":sha, "prompt_wav_rate":16000, "prompt_wav_samples":samples, "seed":1986}, open(sys.argv[2], "w", encoding="utf-8"), ensure_ascii=False)
PY
mkdir -p "$WORK/evidence/reference"
set +e
UV_CACHE_DIR="${COSYVOICE2_UV_CACHE_DIR:-/tmp/vokra-cosyvoice2-uv-cache}" uv run --frozen --project "$REFERENCE_PROJECT" --python 3.12 python "$REFERENCE" --source "$WORK/source/repo" --matcha-source "$WORK/matcha" --model-dir "$WORK/model" --input "$WORK/input.json" --output "$WORK/evidence/reference" >>"$WORK/evidence/validation.log" 2>&1
rc=$?
set -e
[[ "$rc" == 0 ]] || die 'official reference execution did not produce authenticated evidence'
uv run --frozen --project "$REFERENCE_PROJECT" --python 3.12 python - "$WORK/evidence/reference/manifest.json" "$REFERENCE_PROJECT/uv.lock" <<'PY'
import hashlib, json, math, sys
import numpy as np
from pathlib import Path
def unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result: raise AssertionError(f"duplicate JSON key: {key}")
        result[key] = value
    return result
manifest_path = Path(sys.argv[1])
lock_path = Path(sys.argv[2])
assert lock_path.is_file() and not lock_path.is_symlink() and lock_path.stat().st_size > 0
manifest = json.loads(manifest_path.read_text(encoding="utf-8"), object_pairs_hook=unique)
assert manifest["status"] == "AUTHENTICATED_REFERENCE_EVIDENCE"
assert manifest["reference_status"] == "AUTHENTICATED_REFERENCE_EVIDENCE"
assert "REFERENCE_ERROR" not in json.dumps(manifest, sort_keys=True)
assert manifest["runtime_status"] == "NOT_IMPLEMENTED_FAIL_CLOSED"
assert manifest["route"]["native_status"] == "BLOCKED"
assert manifest["flow_noise"]["shape"] == [1, 80, 15000]
assert manifest["source"]["repository"] == "FunAudioLLM/CosyVoice" and manifest["source"]["revision"] == "8555549e882236e6541748b1042d95693caa82ba" and manifest["source"]["resolved_revision"] == manifest["source"]["revision"] and manifest["source"]["clean"] is True
assert manifest["matcha"]["repository"] == "shivammehta25/Matcha-TTS" and manifest["matcha"]["revision"] == "dd9105b34bf2be2230f4aa1e4769fb586a3c824e" and manifest["matcha"]["clean"] is True
assert manifest["model"]["repository"] == "FunAudioLLM/CosyVoice2-0.5B" and manifest["model"]["revision"] == "eec1ae6c79877dbd9379285cf8789c9e0879293d" and manifest["model"]["resolved_revision"] == manifest["model"]["revision"]
assert manifest["input"] == {"mode":"zero_shot", "target_text":"收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。", "prompt_text":"希望你以后能够做的比我还好呦。", "prompt_wav":"asset/zero_shot_prompt.wav", "prompt_wav_sha256":manifest["source"]["files"]["asset/zero_shot_prompt.wav"]["sha256"], "prompt_wav_rate":16000, "prompt_wav_samples":manifest["source"]["files"]["asset/zero_shot_prompt.wav"]["samples"], "seed":1986}
environment = manifest["reference_environment"]
assert environment["python"] == ">=3.12,<3.13"
assert isinstance(environment["python_version"], str) and environment["python_version"].startswith("3.12.")
assert environment["lock_sha256"] == hashlib.sha256(lock_path.read_bytes()).hexdigest()
assert environment["source_transformers_requirement"] == "transformers==4.40.1"
assert environment["transformers_security_advisory"] == "GHSA-xrqw-3rrv-vx5w"
assert environment["transformers_security_patched_minimum"] == "5.10.0"
assert environment["isolated_transformers_pin"] == "5.10.4"
assert environment["transformers_compatibility_status"] == "BLOCKED_UNVERIFIED_API_SMOKE"
assert environment["actual_versions"] == {"torch":"2.3.1", "torchaudio":"2.3.1", "transformers":"5.10.4", "HyperPyYAML":"1.2.2", "conformer":"0.3.2", "diffusers":"0.29.0", "onnxruntime":"1.18.0"}
from tools.parity.cosyvoice2_inspect import EXPECTED
assert set(manifest["model"]["files"]) == set(EXPECTED)
required = {"tokenizer_ids", "prompt_speech_tokens", "campplus_embedding", "qwen_prompt_embeddings", "qwen_prompt_speech_embeddings", "ras_logits", "ras_pre_ras_probability", "ras_nucleus_probability", "ras_multinomial_probability", "ras_calls", "generated_speech_tokens", "flow_rand_noise_full", "flow_rand_noise_slice", "flow_encoder_output", "cfm_terminal_mel", "prompt_mel", "generated_mel", "hift_input_mel", "hift_output_pcm", "official_output_pcm", "prompt_pcm16k"}
assert set(manifest["artifacts"]) <= required | {"ras_fallback_probability"} and required <= set(manifest["artifacts"])
for role in required:
    assert manifest["artifacts"][role], role
assert {"termination", "flow_return_contract", "token2wav_calls"} <= set(manifest["observations"])
execution_id = manifest["execution_id"]
assert isinstance(execution_id, str) and len(execution_id) == 64
referenced = set()
for role, rows in manifest["artifacts"].items():
    assert isinstance(rows, list) and rows, role
    for row in rows:
        assert row["execution_id"] == execution_id
        if "path" not in row: continue
        assert row["path"] not in referenced
        referenced.add(row["path"])
        path = manifest_path.parent / row["path"]
        raw = path.read_bytes()
        assert len(raw) == row["bytes"] > 0
        assert hashlib.sha256(raw).hexdigest() == row["sha256"]
        assert all(isinstance(d, int) and d >= 0 for d in row["shape"])
        storage_dtypes = {"torch.float64":np.float64, "torch.float32":np.float32, "torch.float16":np.float16, "torch.bfloat16":np.float32, "torch.int64":np.int64, "torch.int32":np.int32, "torch.int16":np.int16, "torch.int8":np.int8, "torch.uint8":np.uint8, "torch.bool":np.bool_}
        storage_dtype = row["storage_dtype"]
        assert storage_dtype in storage_dtypes
        values = np.frombuffer(raw, dtype=storage_dtypes[storage_dtype])
        assert values.size == math.prod(row["shape"]) and values.size > 0
        if np.issubdtype(values.dtype, np.floating):
            assert np.isfinite(values).all()
            if "pcm" in role:
                assert np.abs(values).max() <= 1.1
        if role == "flow_rand_noise_full": assert row["shape"] == [1,80,15000]
        if role == "flow_rand_noise_slice": assert len(row["shape"]) == 3 and row["shape"][:2] == [1,80] and 0 < row["shape"][2] <= 15000
        if role in {"generated_mel", "cfm_terminal_mel"}: assert len(row["shape"]) == 3 and row["shape"][:2] == [1,80] and row["shape"][2] > 0
        if role == "flow_encoder_output": assert len(row["shape"]) == 3 and row["shape"][0] == 1 and row["shape"][1] > 0 and row["shape"][2] > 0
        if role == "prompt_mel": assert len(row["shape"]) == 3 and row["shape"][0] == 1 and row["shape"][1] > 0 and row["shape"][2] == 80
        if role == "hift_input_mel": assert len(row["shape"]) == 3 and row["shape"][0] == 1 and row["shape"][1] == 80 and row["shape"][2] > 0
        if role == "generated_speech_tokens": assert len(row["shape"]) == 1 and row["shape"][0] > 0
        if role == "prompt_pcm16k":
            assert row["metadata"]["sample_rate"] == 16000 and row["metadata"]["channels"] == 1 and row["metadata"]["samples"] == row["shape"][-1]
        elif "pcm" in role:
            assert row["metadata"]["sample_rate"] == 24000 and row["metadata"]["channels"] == 1 and row["metadata"]["samples"] == row["shape"][-1]
        if role == "hift_output_pcm":
            assert row["metadata"]["samples"] == row["metadata"]["mel_frames"] * 480
actual = {p.name for p in manifest_path.parent.iterdir() if p.is_file() and p.name != "manifest.json"}
assert actual == referenced, (actual, referenced)
canonical = json.dumps(manifest["artifacts"], sort_keys=True, separators=(",", ":")).encode()
assert hashlib.sha256(canonical).hexdigest() == manifest["artifact_manifest_sha256"]
assert manifest["observations"]["sampling_call_count"] >= manifest["observations"]["llm_call_count"] >= 1
assert manifest["observations"]["ras_config"] == {"top_p":0.8, "top_k":25, "repetition_window":10, "repetition_tau":0.1, "domains":{"top_p":"(0,1]", "repetition_tau":"(0,inf)"}}
assert manifest["observations"]["cfm_steps"] == 10 and len(manifest["observations"]["cfm_time_grid"]) == 11 and len(manifest["observations"]["cfm_estimator_calls"]) == 10
assert manifest["observations"]["flow_return_contract"]["relation"] == "generated_frames = encoder_full_frames - prompt_frames"
assert manifest["artifacts"]["generated_mel"][0]["shape"][2] == manifest["observations"]["flow_return_contract"]["generated_frames"]
assert manifest["observations"]["termination"]["sampled_tokens"]
PY
echo 'run-cosyvoice2-validation.sh: authenticated official reference evidence; native route remains blocked' >&2
