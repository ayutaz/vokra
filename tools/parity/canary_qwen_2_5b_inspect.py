#!/usr/bin/env -S uv run --frozen --project tools/parity --python 3.12 python
"""Fail-closed VAST evidence inspector for NVIDIA Canary-Qwen-2.5B."""
from __future__ import annotations
import argparse, datetime, hashlib, json, math, subprocess, sys, tempfile
from pathlib import Path
from typing import Any
import yaml

HF_REPOSITORY="nvidia/canary-qwen-2.5b"; HF_REVISION="b1469e1bba1cfe140205529c79c434ca47180960"
SOURCE_REPOSITORY="https://github.com/NVIDIA/NeMo.git"; SOURCE_TAG="v2.5.0"; SOURCE_REVISION="ddcb2d6935045a556329f1afa653b8d918c36479"
TOKENIZER_REPOSITORY="Qwen/Qwen3-1.7B"; TOKENIZER_REVISION="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
FORMAT="vokra-canary-qwen-2.5b-inspection-v1"; MAX_HEADER_BYTES=64*1024*1024
CANONICAL_INPUT_LENGTH_MARKER="**Input length.** The maximum audio duration in training was 40s, and the maximum token sequence length was 1024 tokens (including prompt, audio, and response)."
OPEN_ASR_LEADERBOARD_VALUES={"mean_wer":5.63,"rtfx":418.28,"ami_wer":10.19,"earnings22_wer":10.45,"gigaspeech_wer":9.43,"librispeech_clean_wer":1.61,"librispeech_other_wer":3.1,"spgispeech_wer":1.9,"tedlium_wer":2.71,"voxpopuli_wer":5.66}
MODEL_FILES={
    ".eval_results/open_asr_leaderboard.yaml":(2128,"9dc5e54acc69f650c5b9cb40492baadf6bb05430",None,"1b0dbb55d8d897f107baed2a8a57fa9b81e259141e45fd1d45ca8533079527bd"),
    ".gitattributes":(1627,"33f879d5e8f1c01e682db7c50613d2aae540b5d8",None,None),
    "LICENSES":(13649,"a171e1b0e4f785a6d206a814d1242464988de517",None,"64036e6306bdad699fda41984bfead9a648b0d2e7c0987cf29f301fb85a97ef4"),
    "README.md":(18967,"79d76b8880fda22f7b93b6432a41ba3c64e546b9",None,"0f5d9e5066ecc9a5a8b110f4c225b835f72d931630f2c38cac06d6837767ba8d"),
    "config.json":(2382,"af60039c3d20c4314762dc636bc7abfc61372edc",None,"37d15b0445fade873944c9b0cda7221953ec2207169cb6488a3a57d90e629d52"),
    "model.safetensors":(5119120624,"ece9b3ec258fb872cdf3a7e900329899000a02da","800cb0d099cf655a8887d8b741c3a4afa9891e2b2949870251c4d58c72b59175",None),
}
MODEL_REQUIRED_FILES=set(MODEL_FILES)
TOKENIZER_COMPLETE_FILES={".gitattributes","LICENSE","README.md","config.json","generation_config.json","merges.txt","model-00001-of-00002.safetensors","model-00002-of-00002.safetensors","model.safetensors.index.json","tokenizer.json","tokenizer_config.json","vocab.json"}
TOKENIZER_SELECTED_FILES={"LICENSE","README.md","config.json","generation_config.json","merges.txt","tokenizer.json","tokenizer_config.json","vocab.json"}
TOKENIZER_FILE_IDENTITIES={
    "LICENSE":(11343,"6634c8cc3133b3848ec74b9f275acaaa1ea618ab",None),"README.md":(13963,"88aaef81cfe31bcdfe47203649a69ff12042826a",None),"config.json":(726,"044a86ecf7cb32238f3fae4184e55d354787edec",None),"generation_config.json":(239,"20a8a9156fc8c3f25295ca067f61fdf120d517c5",None),"merges.txt":(1671853,"31349551d90c7606f325fe0f11bbb8bd5fa0d7c7",None),"tokenizer.json":(11422654,"cd71f61a15a522601badb3dc960d800d9cb3766c","aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"),"tokenizer_config.json":(9732,"417d038a63fa3de29cfde265caedae14d1a58d92",None),"vocab.json":(2776833,"4783fe10ac3adce15ac8f358ef5462739852c569",None),
}
SOURCE_ROLE_FILES=("nemo/collections/speechlm2/models/salm.py","nemo/collections/speechlm2/modules/perception.py","nemo/collections/speechlm2/parts/lora.py","nemo/collections/speechlm2/parts/pretrained.py","nemo/collections/speechlm2/parts/hf_hub.py","nemo/collections/asr/modules/audio_preprocessing.py","nemo/collections/asr/modules/conformer_encoder.py","nemo/collections/asr/parts/preprocessing/features.py","examples/speechlm2/salm_generate.py")
HISTORICAL_PUBLIC={"repository":"vokra/canary-qwen-2.5b","revision":"4a894c430ed793144acdaa1b07410b071501cb82","path":"canary-qwen.gguf","bytes":5119051360,"git_blob_sha1":"92be1b5e82dc093a6d791e9d3319953535e45296","lfs_sha256":"73b389d0c65c7e3ce714813c658ac8b421fbaff185ed20b164c589bf30a8d344","status":"STALE_PLACEHOLDER_NOT_ACCEPTED"}

def sha256(path:Path)->str:
    h=hashlib.sha256();
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()
def blob(path:Path)->str:
    h=hashlib.sha1(); h.update(f"blob {path.stat().st_size}\0".encode());
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()
def pairs(items:list[tuple[str,Any]])->dict[str,Any]:
    out={}
    for k,v in items:
        if k in out: raise RuntimeError(f"duplicate JSON key: {k}")
        out[k]=v
    return out
def load(path:Path)->Any: return json.loads(path.read_text(encoding="utf-8"),object_pairs_hook=pairs)
class StrictLoader(yaml.SafeLoader):
    pass
def yaml_pairs(loader:StrictLoader,node:yaml.nodes.MappingNode)->dict[str,Any]:
    out={}
    for key_node,value_node in node.value:
        key=loader.construct_object(key_node)
        if key in out: raise RuntimeError(f"duplicate YAML key: {key}")
        out[key]=loader.construct_object(value_node)
    return out
StrictLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,yaml_pairs)
def load_yaml(path:Path)->Any:
    try: return yaml.load(path.read_text(encoding="utf-8"),Loader=StrictLoader)
    except (OSError,UnicodeError,yaml.YAMLError,RuntimeError) as e: raise RuntimeError(f"strict YAML failure at {path}: {e}") from e
def validate_open_asr_leaderboard(value:Any)->list[dict[str,Any]]:
    if type(value) is not list or len(value)!=len(OPEN_ASR_LEADERBOARD_VALUES): raise RuntimeError("open ASR leaderboard must be the exact ten-element list")
    expected_keys={"dataset","value","date","source"}; dataset_keys={"id","task_id"}; source_keys={"url","name","user"}; seen=set(); validated=[]
    for entry in value:
        if type(entry) is not dict or set(entry)!=expected_keys: raise RuntimeError("open ASR leaderboard entry schema is not exact")
        dataset,source=entry["dataset"],entry["source"]
        if type(dataset) is not dict or set(dataset)!=dataset_keys or dataset["id"]!="hf-audio/open-asr-leaderboard" or type(dataset["task_id"]) is not str: raise RuntimeError("open ASR leaderboard dataset schema is invalid")
        task=dataset["task_id"]
        if task in seen or task not in OPEN_ASR_LEADERBOARD_VALUES: raise RuntimeError(f"open ASR leaderboard task is duplicate or unknown: {task!r}")
        expected_value=OPEN_ASR_LEADERBOARD_VALUES[task]
        if type(entry["value"]) is not type(expected_value) or entry["value"]!=expected_value: raise RuntimeError(f"open ASR leaderboard value mismatch: {task}")
        date=entry["date"]
        if not ((type(date) is str and date=="2025-06-26") or (type(date) is datetime.date and date.isoformat()=="2025-06-26")): raise RuntimeError(f"open ASR leaderboard date mismatch: {task}")
        if type(source) is not dict or set(source)!=source_keys or source!={"url":"https://huggingface.co/hf-audio","name":"open-asr-leaderboard","user":"hf-audio"}: raise RuntimeError(f"open ASR leaderboard source mismatch: {task}")
        seen.add(task); validated.append(entry)
    if seen!=set(OPEN_ASR_LEADERBOARD_VALUES): raise RuntimeError("open ASR leaderboard task set is incomplete")
    return validated
def read_front_matter(text:str)->dict[str,Any]:
    lines=text.splitlines()
    if not lines or lines[0].strip()!="---": raise RuntimeError("README YAML front matter missing")
    try: end=lines.index("---",1)
    except ValueError as e: raise RuntimeError("README YAML front matter is unterminated") from e
    parsed=yaml.load("\n".join(lines[1:end])+"\n",Loader=StrictLoader)
    if not isinstance(parsed,dict): raise RuntimeError("README YAML front matter is not a mapping")
    if parsed.get("license")!="cc-by-4.0" or parsed.get("language")!=["en"] or parsed.get("library_name")!="nemo" or not isinstance(parsed.get("datasets"),list) or not parsed["datasets"] or not all(isinstance(item,str) and item for item in parsed["datasets"]): raise RuntimeError("README license/language/library/dataset front matter contract failed")
    tags=parsed.get("tags")
    if not isinstance(tags,list) or "automatic-speech-recognition" not in tags: raise RuntimeError("README ASR tag contract failed")
    model_index=parsed.get("model-index")
    def has_asr_task(value:Any)->bool:
        if isinstance(value,dict): return (value.get("type")=="automatic-speech-recognition" or any(has_asr_task(v) for v in value.values()))
        if isinstance(value,list): return any(has_asr_task(v) for v in value)
        return False
    if not isinstance(model_index,list) or not has_asr_task(model_index): raise RuntimeError("README model-index ASR task contract failed")
    return parsed
def require_canonical_input_length(text:str)->None:
    if CANONICAL_INPUT_LENGTH_MARKER not in text: raise RuntimeError("Canary exact input-length contract missing")
def safe_path(value:str,label:str)->None:
    p=Path(value)
    if not value or "\0" in value or "\\" in value or p.is_absolute() or ".." in p.parts: raise RuntimeError(f"unsafe {label}: {value!r}")
def git(root:Path,*args:str)->str: return subprocess.check_output(["git","-C",str(root),*args],text=True,stderr=subprocess.STDOUT).strip()

def tree(root:Path,packet:Path,repo:str,rev:str,expected:set[str]|None=None)->tuple[dict[str,str],list[dict[str,Any]]]:
    env=load(packet)
    if not isinstance(env,dict) or env.get("repository")!=repo or env.get("revision")!=rev or env.get("resolved_revision")!=rev or not isinstance(env.get("files"),list): raise RuntimeError("server tree identity mismatch")
    actual=set()
    for p in root.rglob("*"):
        rel=p.relative_to(root)
        if ".cache" in rel.parts: continue
        if p.is_symlink():
            if not p.exists() or not p.is_file() or root not in p.resolve().parents: raise RuntimeError(f"invalid snapshot symlink: {p}")
            actual.add(rel.as_posix())
        elif p.is_file(): actual.add(rel.as_posix())
        elif not p.is_dir(): raise RuntimeError(f"non-regular snapshot member: {p}")
    rows=[]; names=set()
    for item in env["files"]:
        if not isinstance(item,dict) or set(item)!={"path","type","size","git_blob_sha1","lfs_sha256"}: raise RuntimeError("invalid server tree item")
        name,kind,size,gid,lfs=(item[k] for k in ("path","type","size","git_blob_sha1","lfs_sha256"))
        if kind!="file" or not isinstance(name,str) or not isinstance(size,int) or isinstance(size,bool) or size<0: raise RuntimeError("invalid server path/size")
        safe_path(name,"server path")
        if name in names or ".cache" in Path(name).parts or not isinstance(gid,str) or len(gid)!=40 or any(c not in "0123456789abcdefABCDEF" for c in gid): raise RuntimeError(f"invalid server identity: {name}")
        if lfs is not None and (not isinstance(lfs,str) or len(lfs)!=64 or any(c not in "0123456789abcdefABCDEF" for c in lfs)): raise RuntimeError(f"invalid LFS identity: {name}")
        p=root/name
        if not p.is_file() or p.stat().st_size!=size: raise RuntimeError(f"server/local size mismatch: {name}")
        local=sha256(p)
        if lfs is None:
            if blob(p).lower()!=gid.lower(): raise RuntimeError(f"Git blob mismatch: {name}")
        elif local.lower()!=lfs.lower(): raise RuntimeError(f"LFS SHA mismatch: {name}")
        names.add(name); rows.append({"path":name,"bytes":size,"sha256":local,"git_blob_sha1":gid,"lfs_sha256":lfs})
    if actual!=names or (expected is not None and names!=expected): raise RuntimeError(f"tree set mismatch: missing={sorted(names-actual)} extra={sorted(actual-names)}")
    return {"repository":repo,"revision":rev,"resolved_revision":rev},sorted(rows,key=lambda r:r["path"])

def packet(packet:Path,repo:str,rev:str,expected:set[str])->list[dict[str,Any]]:
    env=load(packet)
    if not isinstance(env,dict) or env.get("repository")!=repo or env.get("revision")!=rev or env.get("resolved_revision")!=rev or not isinstance(env.get("files"),list): raise RuntimeError("server packet identity mismatch")
    rows=[]; names=set()
    for item in env["files"]:
        if not isinstance(item,dict) or set(item)!={"path","type","size","git_blob_sha1","lfs_sha256"}: raise RuntimeError("invalid server packet item")
        name,kind,size,gid,lfs=(item[k] for k in ("path","type","size","git_blob_sha1","lfs_sha256"))
        if kind!="file" or not isinstance(name,str) or not isinstance(size,int) or isinstance(size,bool) or size<0 or not isinstance(gid,str) or len(gid)!=40 or any(c not in "0123456789abcdefABCDEF" for c in gid): raise RuntimeError("invalid server packet identity")
        safe_path(name,"server packet");
        if name in names or (lfs is not None and (not isinstance(lfs,str) or len(lfs)!=64 or any(c not in "0123456789abcdefABCDEF" for c in lfs))): raise RuntimeError("duplicate/invalid server packet path")
        names.add(name); rows.append({"path":name,"bytes":size,"git_blob_sha1":gid,"lfs_sha256":lfs})
    if names!=expected: raise RuntimeError(f"server packet set mismatch: {sorted(names)}")
    return sorted(rows,key=lambda r:r["path"])

def require(doc:Any,path:tuple[str,...],want:Any)->None:
    cur=doc
    for key in path:
        if not isinstance(cur,dict) or key not in cur: raise RuntimeError(f"missing config path: {'.'.join(path)}")
        cur=cur[key]
    if cur!=want: raise RuntimeError(f"config mismatch {'.'.join(path)}: {cur!r} != {want!r}")
def inspect_config(c:Any)->dict[str,Any]:
    if not isinstance(c,dict): raise RuntimeError("Canary-Qwen config is not a JSON object")
    # These are the literal paths in the pinned NeMo checkpoint config.  In
    # particular, do not accept the older Qwen2-shaped or flattened aliases.
    exact={("audio_locator_tag",):"<|audioplaceholder|>",("pretrained_asr",):"nvidia/canary-1b-flash",("pretrained_llm",):"Qwen/Qwen3-1.7B",("pretrained_weights",):False,("freeze_params",):[r"^llm\..+$",r"^embed_tokens\..+$"],("prevent_freeze_params",):[r"^.+\.lora_.+$"],("prompt_format",):"qwen",("torch_dtype",):"bfloat16",("perception","encoder","_target_"):"nemo.collections.asr.modules.ConformerEncoder",("perception","encoder","n_layers"):32,("perception","encoder","d_model"):1024,("perception","encoder","n_heads"):8,("perception","encoder","ff_expansion_factor"):4,("perception","encoder","conv_kernel_size"):9,("perception","encoder","feat_in"):128,("perception","encoder","subsampling"):"dw_striding",("perception","encoder","subsampling_factor"):8,("perception","encoder","subsampling_conv_channels"):256,("perception","encoder","self_attention_model"):"rel_pos",("perception","encoder","pos_emb_max_len"):5000,("perception","output_dim"):2048,("perception","modality_adapter","_target_"):"nemo.collections.speechlm2.modules.perception.IdentityConnector",("perception","modality_adapter","d_model"):1024,("perception","preprocessor","_target_"):"nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor",("perception","preprocessor","sample_rate"):16000,("perception","preprocessor","features"):128,("perception","preprocessor","n_fft"):512,("perception","preprocessor","normalize"):"per_feature",("perception","preprocessor","window"):"hann",("perception","preprocessor","frame_splicing"):1,("perception","preprocessor","pad_to"):0,("perception","preprocessor","dither"):1e-5,("perception","preprocessor","window_size"):0.025,("perception","preprocessor","window_stride"):0.01,("lora","r"):128,("lora","lora_alpha"):256,("lora","lora_dropout"):0.01,("lora","task_type"):"CAUSAL_LM",("lora","target_modules"): ["q_proj","v_proj"]}
    for p,v in exact.items(): require(c,p,v)
    return {"required_paths":[".".join(p) for p in exact],"values":{".".join(p):v for p,v in exact.items()}}
def inspect_tokenizer(root:Path,rows:list[dict[str,Any]])->dict[str,Any]:
    if {r["path"] for r in rows}!=TOKENIZER_SELECTED_FILES: raise RuntimeError("Qwen tokenizer materialization set mismatch")
    config=load(root/"config.json");
    exact={("model_type",):"qwen3",("hidden_size",):2048,("intermediate_size",):6144,("num_hidden_layers",):28,("num_attention_heads",):16,("num_key_value_heads",):8,("head_dim",):128,("max_position_embeddings",):40960,("vocab_size",):151936,("rope_theta",):1000000,("rms_norm_eps",):1e-6,("tie_word_embeddings",):True,("torch_dtype",):"bfloat16"}
    for p,v in exact.items(): require(config,p,v)
    tokenizer=load(root/"tokenizer_config.json")
    for p,v in {("tokenizer_class",):"Qwen2Tokenizer",("model_max_length",):131072,("eos_token",):"<|im_end|>",("pad_token",):"<|endoftext|>"}.items(): require(tokenizer,p,v)
    generation=load(root/"generation_config.json")
    for p,v in {("eos_token_id",):151645,("pad_token_id",):151643}.items(): require(generation,p,v)
    vocab=load(root/"vocab.json")
    if not isinstance(vocab,dict) or len(vocab)!=151643: raise RuntimeError("Qwen vocab length mismatch")
    merges=(root/"merges.txt").read_text(encoding="utf-8").splitlines()
    if not merges or not merges[0].startswith("#version:") or len(merges)-1!=151387: raise RuntimeError("Qwen merges semantic length mismatch")
    tokenizer_json=load(root/"tokenizer.json")
    for p,v in {("version",):"1.0",("model","type"):"BPE",("normalizer","type"):"NFC",("pre_tokenizer","type"):"Sequence",("decoder","type"):"ByteLevel",("post_processor","type"):"ByteLevel"}.items(): require(tokenizer_json,p,v)
    model_vocab=tokenizer_json.get("model",{}).get("vocab")
    if not isinstance(model_vocab,dict) or len(model_vocab)!=151643: raise RuntimeError("tokenizer.json BPE vocab length mismatch")
    model_merges=tokenizer_json.get("model",{}).get("merges")
    if not isinstance(model_merges,list) or len(model_merges)!=151387: raise RuntimeError("tokenizer.json BPE merges length mismatch")
    added=tokenizer_json.get("added_tokens")
    if not isinstance(added,list) or len(added)!=26: raise RuntimeError("tokenizer.json added_tokens length mismatch")
    return {"config":config,"tokenizer_config":{"tokenizer_class":tokenizer["tokenizer_class"],"model_max_length":tokenizer["model_max_length"],"eos_token":tokenizer["eos_token"],"pad_token":tokenizer["pad_token"],"chat_template_present":isinstance(tokenizer.get("chat_template"),str)},"generation_config":generation,"tokenizer_json":{"version":tokenizer_json["version"],"model_type":tokenizer_json["model"]["type"],"vocab_entries":len(model_vocab),"merges_entries":len(model_merges),"added_tokens":len(added),"normalizer":tokenizer_json["normalizer"]["type"],"pre_tokenizer":tokenizer_json["pre_tokenizer"]["type"],"decoder":tokenizer_json["decoder"]["type"],"post_processor":tokenizer_json["post_processor"]["type"]},"vocab_entries":len(vocab),"merges_lines":len(merges)-1}
def inspect_st(path:Path)->dict[str,Any]:
    size=path.stat().st_size
    with path.open("rb") as f:
        raw_len=f.read(8)
        if len(raw_len)!=8: raise RuntimeError("truncated safetensors")
        n=int.from_bytes(raw_len,"little")
        if n<=0 or n>MAX_HEADER_BYTES or n>size-8: raise RuntimeError("invalid/oversized safetensors header")
        raw=f.read(n)
    h=json.loads(raw.decode(),object_pairs_hook=pairs); meta=h.get("__metadata__")
    if meta is not None and (not isinstance(meta,dict) or not all(isinstance(k,str) and isinstance(v,str) for k,v in meta.items())): raise RuntimeError("invalid metadata")
    payload=size-8-n; intervals=[]; rows=[]
    for name,r in h.items():
        if name=="__metadata__": continue
        safe_path(name,"tensor name")
        if not isinstance(r,dict) or set(r)!={"dtype","shape","data_offsets"} or r["dtype"]!="BF16": raise RuntimeError(f"invalid non-BF16 tensor: {name}")
        shape,off=r["shape"],r["data_offsets"]
        if not isinstance(shape,list) or not isinstance(off,list) or len(off)!=2 or any(not isinstance(x,int) or isinstance(x,bool) or x<0 for x in shape+off): raise RuntimeError(f"invalid tensor descriptor: {name}")
        elems=math.prod(shape); start,end=off
        if end<start or end>payload or end-start!=2*elems: raise RuntimeError(f"invalid tensor range: {name}")
        intervals.append((start,end,name)); rows.append({"name":name,"dtype":"BF16","shape":shape,"elements":elems,"data_offsets":off})
    cur=0
    for start,end,name in sorted(intervals):
        if start!=cur: raise RuntimeError(f"range gap/overlap: {name}")
        cur=end
    if cur!=payload or not rows: raise RuntimeError("tensor body coverage/nonzero tensor contract failed")
    return {"path":path.name,"bytes":size,"sha256":sha256(path),"header_bytes":n,"tensor_count":len(rows),"parameter_count":sum(r["elements"] for r in rows),"all_dtype":"BF16","resident_scope":"header-only; tensor body never read","tensors":rows}

def source_inventory(source:Path)->dict[str,Any]:
    if git(source,"status","--porcelain","--untracked-files=all") or git(source,"rev-parse","HEAD")!=SOURCE_REVISION or git(source,"describe","--exact-match","--tags","HEAD")!=SOURCE_TAG: raise RuntimeError("NeMo source identity/clean check failed")
    origin=git(source,"remote","get-url","origin").rstrip("/")
    if origin.endswith(".git"): origin=origin[:-4]
    if origin!=SOURCE_REPOSITORY.removesuffix(".git"): raise RuntimeError("NeMo origin mismatch")
    tracked=set(git(source,"ls-files").splitlines()); missing=[p for p in SOURCE_ROLE_FILES if p not in tracked or not (source/p).is_file()]
    if missing: raise RuntimeError(f"NeMo role missing: {missing}")
    license_file=source/"LICENSE"
    if not license_file.is_file() or "LICENSE" not in tracked: raise RuntimeError("NeMo LICENSE is missing")
    license_text=license_file.read_text(encoding="utf-8")
    declared="Apache-2.0" if "Apache License" in license_text and "Version 2.0" in license_text else "UNKNOWN"
    if declared=="UNKNOWN": raise RuntimeError("NeMo LICENSE declaration is not authenticated")
    return {"repository":SOURCE_REPOSITORY,"tag":SOURCE_TAG,"revision":SOURCE_REVISION,"origin":origin,"license":{"path":"LICENSE","sha256":sha256(license_file),"declared":declared},"role_files":[{"path":p,"sha256":sha256(source/p)} for p in SOURCE_ROLE_FILES],"tracked_files":len(tracked)}
def blocked(out:Path,error:Exception,status="INSPECTION_ERROR",**extra:Any)->None:
    out.mkdir(parents=True,exist_ok=True); payload={"format":FORMAT,"status":"BLOCKED","inspection_status":status,"evidence_stage":"INSPECTION_ONLY","runtime_status":"NOT_IMPLEMENTED_FAIL_CLOSED","cpu_status":"UNSUPPORTED","metal_status":"BLOCKED_BY_CPU","parity_status":"NOT_RUN","publication":"NO_UPLOAD","upstream":{"repository":HF_REPOSITORY,"revision":HF_REVISION,"license":"CC-BY-4.0"},"official_source":{"repository":SOURCE_REPOSITORY,"revision":SOURCE_REVISION},"error_type":type(error).__name__,"reason":str(error),"blockers":[str(error)],**extra}; (out/"manifest.json").write_text(json.dumps(payload,sort_keys=True,indent=2)+"\n",encoding="utf-8")
def inspect(snapshot:Path,tokenizer:Path,source:Path,model_tree:Path,tok_tree:Path,tok_selected_tree:Path,out:Path)->int:
    identity,files=tree(snapshot,model_tree,HF_REPOSITORY,HF_REVISION);
    if {r["path"] for r in files}!=MODEL_REQUIRED_FILES: raise RuntimeError(f"Canary-Qwen server snapshot must contain the authenticated six-file release set: {sorted(MODEL_REQUIRED_FILES)}")
    for name,(size,gid,lfs,expected_sha) in MODEL_FILES.items():
        row=next(r for r in files if r["path"]==name)
        if row["bytes"]!=size or row["git_blob_sha1"]!=gid or row["lfs_sha256"]!=lfs or (expected_sha is not None and row["sha256"]!=expected_sha): raise RuntimeError(f"fixed model identity mismatch: {name}")
    readme=(snapshot/"README.md").read_text(encoding="utf-8");
    front_matter=read_front_matter(readme)
    if "cc-by-4.0" not in readme.lower(): raise RuntimeError("HF CC-BY-4.0 card declaration missing")
    if "Transcribe the following: <|audioplaceholder|>" not in readme: raise RuntimeError("Canary prompt marker missing")
    require_canonical_input_length(readme)
    parsed={p.relative_to(snapshot).as_posix():{"sha256":sha256(p),"json":load(p)} for p in snapshot.rglob("*.json") if p.is_file() and ".cache" not in p.relative_to(snapshot).parts}; config=inspect_config(parsed.get("config.json",{}).get("json"))
    eval_evidence=validate_open_asr_leaderboard(load_yaml(snapshot/".eval_results/open_asr_leaderboard.yaml"))
    licenses=(snapshot/"LICENSES").read_text(encoding="utf-8")
    if "Canary" not in licenses or "CC-BY-4.0" not in licenses or "Qwen3-1.7B" not in licenses or "Apache-2.0" not in licenses: raise RuntimeError("component license declarations are incomplete")
    tokcomplete=packet(tok_tree,TOKENIZER_REPOSITORY,TOKENIZER_REVISION,TOKENIZER_COMPLETE_FILES)
    tokid,tokfiles=tree(tokenizer,tok_selected_tree,TOKENIZER_REPOSITORY,TOKENIZER_REVISION,TOKENIZER_SELECTED_FILES);
    for row in tokcomplete:
        expected=TOKENIZER_FILE_IDENTITIES.get(row["path"])
        if expected is not None and (row["bytes"],row["git_blob_sha1"],row["lfs_sha256"])!=expected: raise RuntimeError(f"fixed tokenizer identity mismatch: {row['path']}")
    tensors=inspect_st(snapshot/"model.safetensors");
    if any(Path(r["path"]).suffix in (".safetensors",".bin",".pt") for r in tokfiles): raise RuntimeError("tokenizer companion contains forbidden model weights")
    tokconfig=inspect_tokenizer(tokenizer,tokfiles)
    sources=source_inventory(source); out.mkdir(parents=True,exist_ok=True)
    evidence={"snapshot-inventory.json":{"server_tree":identity,"files":files},"tensor-inventory.json":tensors,"parsed-json.json":parsed,"readme-front-matter.json":front_matter,"leaderboard-evidence.json":eval_evidence,"license-evidence.txt":{"sha256":sha256(snapshot/"LICENSES"),"declarations":"Canary CC-BY-4.0; Qwen3-1.7B Apache-2.0"},"tokenizer-inventory.json":{"complete_server_tree":{"repository":TOKENIZER_REPOSITORY,"revision":TOKENIZER_REVISION,"files":tokcomplete},"selected_server_tree":tokid,"files":tokfiles,"model_weights":"NOT_DOWNLOADED","semantic_validation":tokconfig},"source-inventory.json":sources}
    for n,v in evidence.items(): (out/n).write_text(json.dumps(v,sort_keys=True,indent=2)+"\n",encoding="utf-8")
    packets={p.name:{"bytes":p.stat().st_size,"sha256":sha256(p)} for p in out.glob("*-inventory.json")}; blocked(out,RuntimeError("native runtime, tokenizer, dependency, and dataset provenance remain unauthenticated"),"AUTHENTICATED_EVIDENCE_COMPLETE",config=config,tensors={k:v for k,v in tensors.items() if k!="tensors"},tokenizer={"repository":TOKENIZER_REPOSITORY,"revision":TOKENIZER_REVISION,"files":tokfiles,"model_weights":"NOT_DOWNLOADED","semantic_validation":tokconfig},source=sources,historical_public_artifact=HISTORICAL_PUBLIC,policy={"status":"BLOCKED_REVIEW_REQUIRED","research_or_out_of_scope":"UNRESOLVED"},packets=packets); return 2
def self_test()->None:
    src=Path(__file__).read_text(encoding="utf-8"); assert "inspect_st" in src and "safe_"+"open" not in src and "1_"+"017_626_722" not in src
    assert len(HF_REVISION)==len(SOURCE_REVISION)==len(TOKENIZER_REVISION)==40
    for name, identity in MODEL_FILES.items():
        assert len(identity)==4 and type(identity[0]) is int and identity[0]>=0
        assert isinstance(identity[1], str) and len(identity[1])==40 and all(c in "0123456789abcdef" for c in identity[1])
        assert identity[2] is None or (isinstance(identity[2], str) and len(identity[2])==64 and all(c in "0123456789abcdef" for c in identity[2]))
        assert identity[3] is None or (isinstance(identity[3], str) and len(identity[3])==64 and all(c in "0123456789abcdef" for c in identity[3]))
    try: json.loads('{"x":1,"x":2}',object_pairs_hook=pairs)
    except RuntimeError: pass
    else: raise AssertionError("duplicate JSON accepted")
    assert read_front_matter("---\nlicense: cc-by-4.0\nlanguage: [en]\nlibrary_name: nemo\ndatasets:\n  - librispeech\ntags:\n  - automatic-speech-recognition\nmodel-index:\n  - results:\n      - task:\n          type: automatic-speech-recognition\n---\nbody")["license"]=="cc-by-4.0"
    require_canonical_input_length(CANONICAL_INPUT_LENGTH_MARKER)
    for legacy_readme in (
        "1024",
        "The maximum audio duration was 40 seconds and max_tokens was 1024.",
        "The maximum audio duration was 40s and the max sequence was 1024.",
    ):
        try: require_canonical_input_length(legacy_readme)
        except RuntimeError: pass
        else: raise AssertionError("ambiguous input-length README marker was accepted")
    leaderboard_source={"url":"https://huggingface.co/hf-audio","name":"open-asr-leaderboard","user":"hf-audio"}
    valid_leaderboard=[{"dataset":{"id":"hf-audio/open-asr-leaderboard","task_id":task},"value":value,"date":"2025-06-26","source":dict(leaderboard_source)} for task,value in OPEN_ASR_LEADERBOARD_VALUES.items()]
    assert len(validate_open_asr_leaderboard(valid_leaderboard))==10
    date_object_leaderboard=[dict(entry) for entry in valid_leaderboard]
    date_object_leaderboard[0]={**date_object_leaderboard[0],"date":datetime.date(2025,6,26)}
    assert len(validate_open_asr_leaderboard(date_object_leaderboard))==10
    duplicate_task=[dict(entry) for entry in valid_leaderboard]
    duplicate_task[1]={**duplicate_task[1],"dataset":{**duplicate_task[1]["dataset"],"task_id":duplicate_task[0]["dataset"]["task_id"]}}
    bool_value=[dict(entry) for entry in valid_leaderboard]
    bool_value[0]={**bool_value[0],"value":True}
    extra_field=[dict(entry,unexpected=True) for entry in valid_leaderboard]
    for invalid in (duplicate_task,bool_value,extra_field):
        try: validate_open_asr_leaderboard(invalid)
        except RuntimeError: pass
        else: raise AssertionError("invalid leaderboard evidence was accepted")
    try: yaml.load("x: 1\nx: 2\n",Loader=StrictLoader)
    except RuntimeError: pass
    else: raise AssertionError("duplicate YAML accepted")
    try: safe_path("../bad","fixture")
    except RuntimeError: pass
    else: raise AssertionError("traversal accepted")
    c=inspect_config({"audio_locator_tag":"<|audioplaceholder|>","pretrained_asr":"nvidia/canary-1b-flash","pretrained_llm":"Qwen/Qwen3-1.7B","pretrained_weights":False,"freeze_params":[r"^llm\..+$",r"^embed_tokens\..+$"],"prevent_freeze_params":[r"^.+\.lora_.+$"],"prompt_format":"qwen","torch_dtype":"bfloat16","perception":{"encoder":{"_target_":"nemo.collections.asr.modules.ConformerEncoder","n_layers":32,"d_model":1024,"n_heads":8,"ff_expansion_factor":4,"conv_kernel_size":9,"feat_in":128,"subsampling":"dw_striding","subsampling_factor":8,"subsampling_conv_channels":256,"self_attention_model":"rel_pos","pos_emb_max_len":5000},"output_dim":2048,"modality_adapter":{"_target_":"nemo.collections.speechlm2.modules.perception.IdentityConnector","d_model":1024},"preprocessor":{"_target_":"nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor","sample_rate":16000,"features":128,"n_fft":512,"normalize":"per_feature","window":"hann","frame_splicing":1,"pad_to":0,"dither":1e-5,"window_size":0.025,"window_stride":0.01}},"lora":{"r":128,"lora_alpha":256,"lora_dropout":0.01,"task_type":"CAUSAL_LM","target_modules":["q_proj","v_proj"]}}); assert c
    try: inspect_config({"audio_locator_tag":"<|audioplaceholder|>","pretrained_llm":"Qwen/Qwen2-1.5B","torch_dtype":"bfloat16"})
    except RuntimeError: pass
    else: raise AssertionError("old Qwen2 config accepted")
    try: safe_path("tensor\\name", "tensor")
    except RuntimeError: pass
    else: raise AssertionError("unsafe tensor path accepted")
    with tempfile.TemporaryDirectory() as d:
        path=Path(d)/"bad.safetensors"; path.write_bytes((MAX_HEADER_BYTES+1).to_bytes(8,"little"))
        try: inspect_st(path)
        except RuntimeError: pass
        else: raise AssertionError("oversized header accepted")
        path.write_bytes((76).to_bytes(8,"little")+b'{"../bad":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]}}'+b'\0\0')
        try: inspect_st(path)
        except RuntimeError: pass
        else: raise AssertionError("unsafe tensor name accepted")
        def st_fixture(header:dict[str,Any],body:int)->None:
            raw=json.dumps(header,separators=(",",":")).encode(); path.write_bytes(len(raw).to_bytes(8,"little")+raw+bytes(body))
        for header,body in (
            ({"a":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]},"b":{"dtype":"BF16","shape":[1],"data_offsets":[1,3]}},3),
            ({"a":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]},"b":{"dtype":"BF16","shape":[1],"data_offsets":[3,5]}},5),
            ({"a":{"dtype":"BF16","shape":[1],"data_offsets":[0,4]}},2),
            ({"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}},4),
        ):
            st_fixture(header,body)
            try: inspect_st(path)
            except RuntimeError: pass
            else: raise AssertionError("malformed safetensors accepted")
        raw=b'{"a":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]},"a":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]}}'; path.write_bytes(len(raw).to_bytes(8,"little")+raw+b"\0\0")
        try: inspect_st(path)
        except RuntimeError: pass
        else: raise AssertionError("duplicate tensor header key accepted")
    with tempfile.TemporaryDirectory() as d:
        out=Path(d)/"error"; blocked(out,RuntimeError("fixture")); m=load(out/"manifest.json"); assert m["inspection_status"]=="INSPECTION_ERROR" and "AUTHENTICATED_EVIDENCE_COMPLETE" not in m
        complete=Path(d)/"complete"; blocked(complete,RuntimeError("remaining review blocker"),"AUTHENTICATED_EVIDENCE_COMPLETE",config=c); m=load(complete/"manifest.json"); assert m["inspection_status"]=="AUTHENTICATED_EVIDENCE_COMPLETE" and m["config"]["values"]["perception.encoder.n_layers"]==32
    print("canary_qwen_2_5b_inspect.py self-test: OK")
def main()->int:
    p=argparse.ArgumentParser(); p.add_argument("--snapshot",type=Path); p.add_argument("--tokenizer",type=Path); p.add_argument("--source",type=Path); p.add_argument("--server-tree",type=Path); p.add_argument("--tokenizer-complete-tree",type=Path); p.add_argument("--tokenizer-server-tree",type=Path); p.add_argument("--output",type=Path); p.add_argument("--self-test",action="store_true"); a=p.parse_args()
    if a.self_test: self_test(); return 0
    if any(v is None for v in (a.snapshot,a.tokenizer,a.source,a.server_tree,a.tokenizer_complete_tree,a.tokenizer_server_tree,a.output)): p.error("all inspection paths required")
    try: return inspect(a.snapshot,a.tokenizer,a.source,a.server_tree,a.tokenizer_complete_tree,a.tokenizer_server_tree,a.output)
    except Exception as e: blocked(a.output,e); print(f"Canary-Qwen inspection BLOCKED: {e}",file=sys.stderr); return 2
if __name__=="__main__": raise SystemExit(main())
