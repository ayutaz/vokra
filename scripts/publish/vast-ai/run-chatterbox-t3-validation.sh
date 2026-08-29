#!/usr/bin/env bash
# VAST-only official T3 reference validation. No conversion or publication.
set -euo pipefail
ROOT="${VOKRA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
PARITY="$ROOT/tools/parity"
REFERENCE_PROJECT="$PARITY/chatterbox_t3"
REFERENCE_LOCK_SHA256="83879e5e0a3d16c550df9a13134c9f3cbe44e5869afe54674c28be72b5cdec37"
INSPECTOR="$PARITY/chatterbox_family_inspect.py"
REFERENCE="$PARITY/chatterbox_t3_reference.py"
VARIANT="${CHATTERBOX_VARIANT:-base}"
WORK="${CHATTERBOX_T3_WORK_DIR:-/dev/shm/vokra-chatterbox-t3-validation}"
UV_CACHE_DIR="${CHATTERBOX_T3_UV_CACHE_DIR:-/tmp/vokra-chatterbox-t3-uv-cache}"
die(){ echo "chatterbox-t3-vast: ERROR: $*" >&2; exit 2; }
license_audit_preflight(){
  set +e
  local audit_output audit_rc
  audit_output="$(UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python "$REFERENCE" --license-audit 2>&1)"
  audit_rc=$?
  set -e
  if [[ "$audit_rc" == 2 ]]; then
    [[ "$audit_output" == *"$REFERENCE_LOCK_SHA256"* ]] || die 'license audit did not report the reviewed lock identity'
    echo "$audit_output" >&2
    return 1
  fi
  [[ "$audit_rc" == 0 ]] || die 'dependency license audit command failed unexpectedly'
  return 0
}
require_absent_path(){
  local target="$1" current
  [[ ! -e "$target" && ! -L "$target" ]] || die 'work directory must be absent and not a symlink'
  current="$target"
  while [[ "$current" != / && "$current" != . && -n "$current" ]]; do
    [[ ! -L "$current" ]] || die "work path contains a symlink component: $current"
    current="$(dirname "$current")"
  done
}
canonical_uncreated(){
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
require_disjoint_uncreated(){
  local candidate protected base
  candidate="$(canonical_uncreated "$1")"
  shift
  for protected in "$@"; do
    base="$(cd -P "$protected" && pwd)" || die "protected path is not accessible: $protected"
    [[ "$candidate" != "$base" && "$candidate" != "$base/"* && "$base" != "$candidate/"* ]] || die "work path overlaps protected path: $protected"
  done
}
[[ "$VARIANT" == base || "$VARIANT" == nano || "$VARIANT" == turbo ]] || die 'CHATTERBOX_VARIANT must be base, nano, or turbo'
self_test(){
  local fail=0 token
for token in '5de7a54aa4e5e2baadb0182dde554908b48b85c2' 'SOURCE_ROLE_BLOBS' 't3_mtl23ls_v3.safetensors' 'Cangjie5_TC.json' 'mtl_tokenizer.json' 'REFERENCE_EVIDENCE_COMPLETE' 'torch.multinomial' 'NO_UPLOAD' 'CARGO_BUILD_JOBS=1' 'CHATTERBOX_T3_REFERENCE_PACKET' 'transformers==5.2.0' 'torch==2.6.0' 'mutable optional Perth' 'reference_environment' 'AUTHENTICATED_CPU_INDEX_METADATA_LOCKED' 'BLOCKED_UNRESOLVED' '2.6.0+cpu' 'https://download.pytorch.org/whl/cpu' 'nvidia-*' 'resemble-perth' 'from pathlib import Path' 'chatterbox_t3' 'uv.lock' '83879e5e0a3d16c550df9a13134c9f3cbe44e5869afe54674c28be72b5cdec37' 'f5cfab32caf3cc2340b434c1e9e0d3f8dbbab73a519925fbb6f08457c03e7e98' 'package_rows' 'license_conclusions' 'inference_turbo(max_gen_len=0)' 'tfmr.wte.' '--import-smoke' '--inspection' '--license-audit'; do
    grep -Fq -- "$token" "$REFERENCE" "$INSPECTOR" "$0" || { echo "missing contract: $token" >&2; fail=1; }
  done
  if grep -En '(^|[;&|][[:space:]]*)git[[:space:]]+push|hf_hub_upload|upload_file|--push' "$0" | grep -v 'grep -En' >/dev/null; then echo 'publication command found' >&2; fail=1; fi
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python "$REFERENCE" --self-test >/dev/null || fail=1
  (( fail == 0 )) || return 1
  echo 'run-chatterbox-t3-validation.sh self-test: OK'
}
[[ "${1:-}" != --self-test || $# == 1 ]] || die '--self-test accepts no arguments'
if [[ "${1:-}" == --self-test ]]; then self_test; exit 0; fi
if ! license_audit_preflight; then die 'dependency license audit is unresolved; no Chatterbox model acquisition or reference execution is permitted'; fi
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || die 'Linux x86_64 VAST required'
[[ "${VOKRA_PUBLISH_ON_VAST:-0}" == 1 ]] || die 'VOKRA_PUBLISH_ON_VAST=1 required'
[[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] || die 'clean checkout required'
[[ -f "$REFERENCE_PROJECT/pyproject.toml" && -f "$REFERENCE_PROJECT/uv.lock" ]] || die 'dedicated Chatterbox T3 pyproject.toml + uv.lock are required; generate them in a networked, reviewed environment before running VAST'
for command in awk cargo df find findmnt git uv; do command -v "$command" >/dev/null || die "missing tool: $command"; done
mem_kib="$(awk '$1=="MemTotal:"{print $2;exit}' /proc/meminfo)"; [[ "$mem_kib" =~ ^[0-9]+$ && "$mem_kib" -ge $((128*1024*1024)) ]] || die '128 GiB RAM required'
parent="$(dirname "$WORK")"; [[ "$(findmnt -T "$parent" -no FSTYPE 2>/dev/null || true)" == tmpfs ]] || die 'work parent must be tmpfs'
free_kib="$(df -Pk "$parent" | awk 'NR==2{print $4}')"; [[ "$free_kib" =~ ^[0-9]+$ && "$free_kib" -ge $((40*1024*1024)) ]] || die '40 GiB free tmpfs required'
require_absent_path "$WORK"
require_disjoint_uncreated "$WORK" "$ROOT" "$PARITY" "$REFERENCE_PROJECT"
mkdir -p "$WORK"/{model,source,evidence}; export CARGO_BUILD_JOBS=1
UV_CACHE_DIR="$UV_CACHE_DIR" uv run --frozen --project "$REFERENCE_PROJECT" --python 3.12 python - "$VARIANT" "$WORK/model" "$WORK/tree.json" <<'PY'
import json, sys
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download
variant, destination, tree_path = sys.argv[1:]
models = {
    "base": ("ResembleAI/chatterbox", "5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18", ["README.md", "Cangjie5_TC.json", "conds.pt", "grapheme_mtl_merged_expanded_v1.json", "mtl_tokenizer.json", "s3gen_v3.safetensors", "t3_mtl23ls_v3.safetensors", "tokenizer.json", "ve.safetensors"]),
    "nano": ("ResembleAI/chatterbox-nano", "71ccd1d0081b430592cea481f4307e764e07bc64", ["README.md", "added_tokens.json", "merges.txt", "s3gen.safetensors", "s3gen_meanflow.safetensors", "special_tokens_map.json", "t3_nano_v1.safetensors", "tokenizer_config.json", "ve.safetensors", "vocab.json"]),
    "turbo": ("ResembleAI/chatterbox-turbo", "749d1c1a46eb10492095d68fbcf55691ccf137cd", ["README.md", "added_tokens.json", "merges.txt", "s3gen.safetensors", "s3gen_meanflow.safetensors", "special_tokens_map.json", "t3_turbo_v1.safetensors", "tokenizer_config.json", "ve.safetensors", "vocab.json"]),
}
repo, revision, patterns = models[variant]
api = HfApi(); info = api.model_info(repo, revision=revision)
if info.sha != revision: raise RuntimeError(f"resolved revision drift: {info.sha}")
snapshot_download(repo_id=repo, revision=revision, local_dir=destination, allow_patterns=patterns)
rows=[]
for item in api.list_repo_tree(repo, revision=revision, recursive=True, expand=True):
    if getattr(item, "type", None) == "file":
        lfs = getattr(item, "lfs", None)
        rows.append({"path":item.path,"type":"file","size":item.size,
                     "git_blob_sha1":getattr(item, "oid", None) or getattr(item, "blob_id", None),
                     "lfs_sha256":(lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)) if lfs else None})
Path(tree_path).write_text(json.dumps({"repository":repo,"revision":revision,"resolved_revision":info.sha,"files":rows},sort_keys=True,indent=2)+"\n",encoding="utf-8")
PY
git clone --filter=blob:none --no-checkout https://github.com/resemble-ai/chatterbox.git "$WORK/source/chatterbox" >/dev/null 2>&1 || die 'source clone failed'
git -C "$WORK/source/chatterbox" checkout --detach 5de7a54aa4e5e2baadb0182dde554908b48b85c2 >/dev/null 2>&1 || die 'source checkout failed'
reference_run() {
  UV_CACHE_DIR="$UV_CACHE_DIR" uv run --frozen --project "$REFERENCE_PROJECT" --python 3.12 python "$@"
}
reference_run "$REFERENCE" --import-smoke --source "$WORK/source/chatterbox" >"$WORK/evidence/import-smoke.json"
set +e
UV_CACHE_DIR="$UV_CACHE_DIR" uv run --frozen --project "$REFERENCE_PROJECT" --python 3.12 python "$INSPECTOR" --variant "$VARIANT" --snapshot "$WORK/model" --server-tree "$WORK/tree.json" --source "$WORK/source/chatterbox" --evidence "$WORK/evidence/inspection" >"$WORK/evidence/inspection.log" 2>&1
inspect_rc=$?; set -e; [[ "$inspect_rc" == 2 ]] || die "inspector returned $inspect_rc (expected blocked status 2)"
[[ -n "${CHATTERBOX_T3_REFERENCE_PACKET:-}" && -f "$CHATTERBOX_T3_REFERENCE_PACKET" ]] || die 'CHATTERBOX_T3_REFERENCE_PACKET must name caller-owned packet'
UV_CACHE_DIR="$UV_CACHE_DIR" uv run --frozen --project "$REFERENCE_PROJECT" --python 3.12 python - "$WORK/evidence/inspection/manifest.json" <<'PY'
import hashlib,json,sys
from pathlib import Path
def pairs(items):
    out={}
    for key,value in items:
        if key in out: raise ValueError(f"duplicate inspection key: {key}")
        out[key]=value
    return out
m=json.load(open(sys.argv[1],encoding='utf-8'),object_pairs_hook=pairs)
if m.get("status")!="BLOCKED" or m.get("inspection_status")!="AUTHENTICATED_EVIDENCE_COMPLETE":
    raise SystemExit("authenticated Chatterbox inspection evidence is required")
PY
set +e
reference_run "$REFERENCE" --source "$WORK/source/chatterbox" --snapshot "$WORK/model" --inspection "$WORK/evidence/inspection/manifest.json" --packet "$CHATTERBOX_T3_REFERENCE_PACKET" --output "$WORK/evidence/reference" >"$WORK/evidence/reference.log" 2>&1
reference_rc=$?; set -e; [[ "$reference_rc" == 2 ]] || die "reference returned $reference_rc (expected blocked status 2)"
UV_CACHE_DIR="$UV_CACHE_DIR" uv run --frozen --project "$REFERENCE_PROJECT" --python 3.12 python - "$WORK/evidence/reference/manifest.json" "$WORK/evidence/reference" <<'PY'
import hashlib,json,re,sys,tomllib
from pathlib import Path
def pairs(items):
    out={}
    for key,value in items:
        if key in out: raise ValueError(f"duplicate manifest key: {key}")
        out[key]=value
    return out
m=json.load(open(sys.argv[1],encoding='utf-8'),object_pairs_hook=pairs)
expected_manifest={"format","status","evidence_stage","reference_status","runtime_status","publication","source","reference_environment","inspection","model_repository","model_revision","checkpoint","variant","tokenizer_calls","multinomial_calls","multinomial_probability_capture","caller_owned_draw","loaded_t3_manifest","generation_route","taps","pcm_status"}
if set(m) != expected_manifest: raise SystemExit(f"reference manifest schema mismatch: {sorted(m)}")
required={"status":"BLOCKED","evidence_stage":"INSPECTION_ONLY","reference_status":"REFERENCE_EVIDENCE_COMPLETE","runtime_status":"NOT_IMPLEMENTED_FAIL_CLOSED","publication":"NO_UPLOAD"}
if any(m.get(k)!=v for k,v in required.items()): raise SystemExit(f"reference manifest status mismatch: {m}")
if m.get("multinomial_calls") != 1 or m.get("tokenizer_calls") != 1: raise SystemExit("caller-owned T3 trace contract was not consumed exactly once")
environment=m.get("reference_environment")
if not isinstance(environment,dict) or set(environment) != {"path","sha256","python","core_versions","cpu_index","cpu_distribution_versions","package_rows_sha256","package_rows","excluded_packages","package_names","license_audit"}: raise SystemExit("reference environment identity missing")
expected_core={"numpy":"1.26.4","huggingface-hub":"1.27.0","einops":"0.8.2","safetensors":"0.5.3","torch":"2.6.0","torchaudio":"2.6.0","tqdm":"4.67.1","transformers":"5.2.0"}
if environment.get("python") != "==3.12.*" or not isinstance(environment.get("sha256"),str) or re.fullmatch(r"[0-9a-f]{64}",environment["sha256"]) is None or environment.get("core_versions") != expected_core: raise SystemExit("reference environment identity drifted")
if environment.get("cpu_index") != "https://download.pytorch.org/whl/cpu" or environment.get("cpu_distribution_versions") != {"torch":"2.6.0+cpu","torchaudio":"2.6.0+cpu"}: raise SystemExit("CPU PyTorch routing drifted")
lock_path=Path(environment["path"])
if not lock_path.is_file() or hashlib.sha256(lock_path.read_bytes()).hexdigest() != "83879e5e0a3d16c550df9a13134c9f3cbe44e5869afe54674c28be72b5cdec37": raise SystemExit("dedicated lock SHA drifted")
lock=tomllib.loads(lock_path.read_text(encoding="utf-8")); expected_rows=[]
for package in lock.get("package",[]):
    source=package.get("source",{})
    if set(source) not in ({"registry"},{"virtual"}) or not isinstance(package.get("name"),str) or not isinstance(package.get("version"),str): raise SystemExit("lock package row malformed")
    expected_rows.append({"name":package["name"],"version":package["version"],"source":{key:source[key] for key in sorted(source)},"markers":sorted(package.get("resolution-markers",[]))})
expected_rows.sort(key=lambda row:(row["name"],row["version"],json.dumps(row["source"],sort_keys=True),row["markers"]))
encoded=json.dumps(expected_rows,sort_keys=True,separators=(",",":")).encode("utf-8")
if environment.get("package_rows_sha256") != "f5cfab32caf3cc2340b434c1e9e0d3f8dbbab73a519925fbb6f08457c03e7e98" or hashlib.sha256(encoded).hexdigest() != environment["package_rows_sha256"] or environment.get("package_rows") != expected_rows: raise SystemExit("versioned lock package rows drifted")
if set(environment.get("excluded_packages",[])) != {"diffusers","resemble-perth","s3tokenizer","gradio"}: raise SystemExit("excluded reference packages drifted")
if not isinstance(environment.get("package_names"),list) or "vokra-chatterbox-t3-reference" not in environment["package_names"]: raise SystemExit("reference package inventory missing")
audit=environment.get("license_audit")
if not isinstance(audit,dict) or audit.get("status") != "AUTHENTICATED_CLEAR" or not audit.get("cuda","").startswith("NOT_IN_LOCK") or not audit.get("triton","").startswith("NOT_IN_LOCK"): raise SystemExit("dependency license audit is not cleared")
if not isinstance(audit.get("license_conclusions"),dict): raise SystemExit("versioned license conclusions missing")
expected_license_keys={row["name"]+"=="+row["version"] for row in expected_rows}
if set(audit["license_conclusions"]) != expected_license_keys: raise SystemExit("versioned license conclusion inventory drifted")
for key,value in audit["license_conclusions"].items():
    if not isinstance(value,dict) or set(value) != {"license","evidence","source"} or not all(isinstance(value[field],str) and value[field] for field in value): raise SystemExit("versioned license conclusion row malformed")
expected={"text_tokens","conditioning","multinomial_probs_0001","generated_tokens"}
taps=m.get("taps")
if not isinstance(taps,list) or len(taps)!=4 or {tap.get("name") for tap in taps}!=expected: raise SystemExit("T3 tap set/cardinality mismatch")
variant=m.get("variant")
if variant not in {"base","nano","turbo"}: raise SystemExit("unknown T3 variant")
loaded=m.get("loaded_t3_manifest")
if not isinstance(loaded,dict) or set(loaded)!={"parameter_count","parameters_sha256","removed_wrapper_parameter_prefix"}:
    raise SystemExit("loaded T3 manifest schema mismatch")
if not isinstance(loaded["parameter_count"],int) or loaded["parameter_count"]<=0 or not isinstance(loaded["parameters_sha256"],str) or len(loaded["parameters_sha256"])!=64 or any(c not in "0123456789abcdef" for c in loaded["parameters_sha256"]):
    raise SystemExit("loaded T3 manifest identity mismatch")
expected_removed=None if variant=="base" else "tfmr.wte."
if loaded["removed_wrapper_parameter_prefix"] != expected_removed: raise SystemExit("official wrapper deletion manifest mismatch")
expected_route="inference(max_new_tokens=1)" if variant=="base" else "inference_turbo(max_gen_len=0)"
if m.get("generation_route") != expected_route: raise SystemExit("variant generation route mismatch")
probability_width=8194 if variant=="base" else 6563
by_name={tap["name"]:tap for tap in taps}
if by_name["multinomial_probs_0001"]["shape"] != [1,probability_width]: raise SystemExit("variant probability shape mismatch")
if by_name["generated_tokens"]["shape"] != [1,1]: raise SystemExit("generated token shape mismatch")
draw=m.get("caller_owned_draw")
if not isinstance(draw,dict) or draw.get("native_torch_rng") is not False or not isinstance(draw.get("value"),(int,float)) or not 0 <= draw["value"] < 1: raise SystemExit("caller-owned draw contract missing")
root=Path(sys.argv[2])
allowed={"manifest.json"} | {f"{name}.bin" for name in expected}
if {path.name for path in root.iterdir()} != allowed: raise SystemExit("stale or extra T3 artifacts")
for tap in taps:
    if set(tap) != {"name","shape","dtype","bytes","sha256"}: raise SystemExit("T3 tap schema mismatch")
    if not isinstance(tap.get("shape"),list) or any(not isinstance(dim,int) or dim<=0 for dim in tap["shape"]): raise SystemExit("invalid T3 tap shape")
    if tap["name"] not in expected: raise SystemExit("unexpected T3 tap name")
    widths={"torch.float32":4,"torch.float16":2,"torch.bfloat16":2,"torch.int64":8,"torch.int32":4,"torch.int16":2,"torch.int8":1,"torch.uint8":1,"torch.bool":1}
    if tap.get("dtype") not in widths: raise SystemExit("unsupported T3 tap dtype")
    if not isinstance(tap.get("bytes"),int) or tap["bytes"]<=0 or not isinstance(tap.get("sha256"),str) or len(tap["sha256"])!=64 or any(c not in "0123456789abcdef" for c in tap["sha256"]): raise SystemExit("invalid T3 tap identity")
    numel=1
    for dim in tap["shape"]: numel*=dim
    if tap["bytes"] != numel*widths[tap["dtype"]]: raise SystemExit("T3 tap byte/shape mismatch")
    artifact=root / f"{tap['name']}.bin"
    raw=artifact.read_bytes()
    if len(raw) != tap["bytes"] or hashlib.sha256(raw).hexdigest() != tap["sha256"]: raise SystemExit("T3 artifact identity mismatch")
PY
echo 'official Chatterbox T3 reference evidence complete; composite PCM, conversion and upload remain blocked.' >&2
exit 2
