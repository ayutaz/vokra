#!/usr/bin/env -S uv run --frozen --project tools/parity --python 3.12 python
"""Inspect fixed Kyutai STT-2.6B-EN evidence without conversion or runtime."""
from __future__ import annotations
import argparse, hashlib, io, json, os, re, subprocess, tarfile, tempfile, zipfile
from pathlib import Path, PurePosixPath
from typing import Any

REPO="kyutai/stt-2.6b-en"; REV="a07aec56d22be5589cd0bc8709c75b6cf3e3039d"
SOURCE_URL="https://github.com/kyutai-labs/delayed-streams-modeling.git"; SOURCE_REV="4c4f65e147df056adf3346290d64c7b9649b18c9"
MOSHI_URL="https://github.com/kyutai-labs/moshi.git"; MOSHI_REV="e6a55d2722a65870ef52a6c9f6ecfc0e90f38362"
TOKENIZER_NAME="tokenizer_en_audio_4000.model"
LEGACY_TOKENIZER_NAME="tokenizer_spm_4k_en.model"
MODEL_FILES={".gitattributes","README.md","config.json","mimi-pytorch-e351c8d8@125.safetensors","model.safetensors",TOKENIZER_NAME}
ARTIFACTS={
".gitattributes":(1519,"a6344aac8c09253b3b630fb776ae94478aa0275b",None),
"README.md":(3782,"e201a9359812bf30ba1279688afc59d39a2f9164",None),
"config.json":(1257,"cb670551e4f81233fa9f60e025f9101e14b7bc88",None),
"mimi-pytorch-e351c8d8@125.safetensors":(384644900,"c8d5e4cd18a5c1ce05bb89d81144a46cf1b9076c","09b782f0629851a271227fb9d36db65c041790365f11bbe5d3d59369cf863f50"),
"model.safetensors":(5234275128,"9b68cd59de6c20acf683281b53999639d442beab","2471add7da1fdb2d5dc4561e88a9069376333d992760d55d29d1db46c52849b2"),
TOKENIZER_NAME:(59339,"1820a7cbb15efc6a33dd365113c07e3df9d28d80","d461765ae179566678c93091c5fa6f2984c31bbe990bf1aa62d92c64d91bc3f6"),
}
FIXED_SHA256={"README.md":"6a93b7d998b32cb65f07e8948508004421042f100130c3572de13af5cab9e4f9","config.json":"b79ea52a30329887a2d0ce2dd5473a63fc5083e441e7986f64f01050c06239c9"}
HISTORICAL_PUBLIC_ARTIFACT={"repository":"vokra/kyutai-stt-2.6b-en","revision":"c8f5779f1471f34734aafe1999082ca33862bc5e","path":"model.gguf","bytes":5234266976,"git_blob_sha1":"f3735afd029fa1f168af6ba9bed7e8f83045344b","lfs_sha256":"d25302da6650309c094d0cbf10cfecfb507c31408b820304bda0c3195482f990","status":"STALE_PRECONTRACT_BLOCKER"}
TOTAL=sum(v[0] for v in ARTIFACTS.values())
MAX_HEADER=64*1024*1024
ROLES=("configs/config-stt-en-hf.toml","scripts/stt_evaluate_on_dataset.py","scripts/stt_from_file_mlx.py","scripts/stt_from_file_pytorch.py","scripts/stt_from_file_rust_server.py","scripts/stt_from_file_with_prompt_pytorch.py","scripts/stt_from_mic_mlx.py","scripts/stt_from_mic_rust_server.py","stt-rs/src/main.rs","README.md","LICENSE-APACHE","LICENSE-MIT")
MOSHI_ROLES=("moshi/moshi/models/lm.py","moshi/moshi/models/lm_utils.py","moshi/moshi/models/loaders.py","moshi/moshi/conditioners/__init__.py","moshi/moshi/conditioners/base.py","moshi/moshi/conditioners/tensors.py","moshi/moshi/conditioners/text.py","rust/moshi-core/src/lm.rs","rust/moshi-core/src/lm_generate.rs","rust/moshi-core/src/lm_generate_multistream.rs","rust/moshi-core/src/mimi.rs","rust/moshi-core/src/conditioner.rs")
HEX40=re.compile(r"^[0-9a-f]{40}$"); HEX64=re.compile(r"^[0-9a-f]{64}$")

def require_fixed_model_files(names:set[str])->None:
 if LEGACY_TOKENIZER_NAME in names: raise ValueError("legacy tokenizer filename is not accepted")
 if TOKENIZER_NAME not in names: raise ValueError("authenticated tokenizer filename is missing")
 if names!=MODEL_FILES: raise ValueError("fixed six-file tree mismatch")

def sha(path:Path)->str:
 h=hashlib.sha256()
 with path.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""): h.update(b)
 return h.hexdigest()
def blob(path:Path)->str:
 h=hashlib.sha1(); h.update(f"blob {path.stat().st_size}\0".encode())
 with path.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""): h.update(b)
 return h.hexdigest()
def unique(pairs:list[tuple[str,Any]])->dict[str,Any]:
 out={}
 for k,v in pairs:
  if k in out: raise ValueError(f"duplicate JSON key {k}")
  out[k]=v
 return out
def safe(name:str)->None:
 p=PurePosixPath(name)
 if not name or "\x00" in name or "\\" in name or p.is_absolute() or ".." in p.parts: raise ValueError(f"unsafe path {name!r}")
def files(root:Path)->dict[str,Path]:
 root=root.resolve(); out={}
 for p in root.rglob("*"):
  rel=p.relative_to(root).as_posix()
  if rel==".cache" or rel.startswith(".cache/"): continue
  safe(rel); resolved=p.resolve(strict=False)
  if not str(resolved).startswith(str(root)+os.sep): raise ValueError("snapshot escape")
  if p.is_symlink() and not resolved.is_file(): raise ValueError("unsafe symlink")
  if p.is_file(): out[rel]=p
  elif not p.is_dir(): raise ValueError("nonregular snapshot entry")
 return out
def server_tree(packet:Any,root:Path, fixed:bool=True)->dict[str,Any]:
 if not isinstance(packet,dict) or set(packet)!={"repository","revision","resolved_revision","files"} or packet["repository"]!=REPO or packet["revision"]!=REV or packet["resolved_revision"]!=REV: raise ValueError("server identity mismatch")
 remote={}
 for e in packet["files"]:
  if not isinstance(e,dict) or set(e)!={"path","type","size","git_blob_sha1","lfs_sha256"}: raise ValueError("server entry schema")
  p,k,size,g,l=(e[x] for x in ("path","type","size","git_blob_sha1","lfs_sha256")); safe(p)
  if k!="file" or not isinstance(size,int) or isinstance(size,bool) or size<0 or p in remote or not isinstance(g,str) or not HEX40.fullmatch(g) or (l is not None and (not isinstance(l,str) or not HEX64.fullmatch(l))): raise ValueError("server entry identity")
  remote[p]=e
 local=files(root)
 if set(remote)!=set(local): raise ValueError("server/local tree mismatch")
 for p,e in remote.items():
  if fixed and (p not in ARTIFACTS or e["size"] != ARTIFACTS[p][0] or e["git_blob_sha1"] != ARTIFACTS[p][1] or e["lfs_sha256"] != ARTIFACTS[p][2]):
   raise ValueError(f"fixed artifact identity mismatch {p}")
  actual=sha(local[p]) if e["lfs_sha256"] else blob(local[p])
  if local[p].stat().st_size!=e["size"] or actual!=(e["lfs_sha256"] or e["git_blob_sha1"]): raise ValueError(f"content identity mismatch {p}")
  if fixed and p in FIXED_SHA256 and sha(local[p]) != FIXED_SHA256[p]: raise ValueError(f"fixed SHA-256 mismatch {p}")
 return {"repository":REPO,"revision":REV,"resolved_revision":REV,"files":sorted(remote.values(),key=lambda e:e["path"])}
def st_header(path:Path)->dict[str,Any]:
 size=path.stat().st_size
 with path.open("rb") as f:
  raw=f.read(8)
  if len(raw)!=8: raise ValueError("short safetensors")
  n=int.from_bytes(raw,"little")
  if n<=0 or n>MAX_HEADER or n+8>size: raise ValueError("header bound")
  head=json.loads(f.read(n).decode(),object_pairs_hook=unique)
 if not isinstance(head,dict): raise ValueError("header root")
 meta=head.pop("__metadata__",{})
 if not isinstance(meta,dict) or any(not isinstance(k,str) or not isinstance(v,str) for k,v in meta.items()): raise ValueError("metadata must string map")
 data=size-8-n; ranges=[]; tensors={}
 for name,d in head.items():
  safe(name)
  if not isinstance(d,dict) or set(d)!={"dtype","shape","data_offsets"}: raise ValueError("descriptor schema")
  dtype,shape,off=d["dtype"],d["shape"],d["data_offsets"]
  if dtype not in {"F32","BF16","F16"} or not isinstance(shape,list) or not isinstance(off,list) or len(off)!=2 or any(isinstance(x,bool) or not isinstance(x,int) or x<0 for x in shape+off): raise ValueError("descriptor value")
  elems=1
  for dim in shape: elems*=dim
  if off[0]>off[1] or off[1]>data or off[1]-off[0]!=elems*{"F32":4,"BF16":2,"F16":2}[dtype]: raise ValueError("tensor bounds")
  ranges.append(tuple(off)); tensors[name]={"dtype":dtype,"shape":shape,"data_offsets":off,"bytes":off[1]-off[0]}
 cur=0
 for a,b in sorted(ranges):
  if a!=cur: raise ValueError("gap/overlap")
  cur=b
 if cur!=data: raise ValueError("trailing data gap")
 return {"header_bytes":n,"data_bytes":data,"metadata":meta,"tensors":tensors}

def archive_inventory(path:Path)->dict[str,Any]:
 """Inventory a checkpoint container without deserializing its payload."""
 if not path.is_file(): raise ValueError("checkpoint container missing")
 rows=[]; seen=set(); limit=4096; max_member=2*1024*1024*1024
 try:
  if zipfile.is_zipfile(path):
   with zipfile.ZipFile(path) as archive:
    for info in archive.infolist():
     name=info.filename; safe(name)
     if name in seen: raise ValueError("duplicate archive member")
     seen.add(name)
     if info.flag_bits & 0x1: raise ValueError("encrypted archive member")
     mode=(info.external_attr >> 16) & 0o170000
     is_dir=info.is_dir() or name.endswith("/")
     if mode not in (0,0o100000,0o040000) or (not is_dir and mode==0o040000): raise ValueError("unsafe ZIP member type")
     if info.file_size>max_member: raise ValueError("archive member too large")
     rows.append({"name":name,"directory":is_dir,"size":info.file_size})
  else:
   with tarfile.open(path, mode="r:*") as archive:
    for info in archive.getmembers():
     name=info.name; safe(name)
     if name in seen: raise ValueError("duplicate archive member")
     seen.add(name)
     if not (info.isdir() or info.isreg()): raise ValueError("unsafe TAR member type")
     if info.size>max_member: raise ValueError("archive member too large")
     rows.append({"name":name,"directory":info.isdir(),"size":info.size})
 except (OSError,ValueError,tarfile.TarError,zipfile.BadZipFile) as exc:
  raise ValueError(f"archive inventory failed: {exc}") from exc
 if len(rows)>limit: raise ValueError("too many archive members")
 return {"format":"zip" if zipfile.is_zipfile(path) else "tar","members":rows}
def source(root:Path,url:str,rev:str,roles:tuple[str,...])->dict[str,Any]:
 origin=subprocess.check_output(["git","-C",str(root),"remote","get-url","origin"],text=True).strip(); head=subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],text=True).strip(); status=subprocess.check_output(["git","-C",str(root),"status","--porcelain","--untracked-files=all"],text=True)
 if origin!=url or head!=rev or status: raise ValueError("source identity/clean mismatch")
 tracked=set(filter(None,subprocess.check_output(["git","-C",str(root),"ls-files","-z"]).decode().split("\0")))
 role_rows=[]
 for role in roles:
  if role not in tracked or not (root/role).is_file(): raise ValueError(f"source role missing {role}")
  role_rows.append({"path":role,"bytes":(root/role).stat().st_size,"sha256":sha(root/role)})
 license_rows=[]
 for name in ("LICENSE-APACHE","LICENSE-MIT","README.md"):
  path=root/name
  if name not in tracked or not path.is_file(): raise ValueError(f"required license record missing {name}")
  text=path.read_text(encoding="utf-8")
  if name=="LICENSE-APACHE" and "Apache License" not in text: raise ValueError("Apache license marker missing")
  if name=="LICENSE-MIT" and "Permission is hereby granted" not in text: raise ValueError("MIT license marker missing")
  license_rows.append({"path":name,"bytes":path.stat().st_size,"sha256":sha(path),"encoding":"utf-8"})
 return {"origin":origin,"revision":head,"roles":role_rows,"license_records":license_rows}
def base()->dict[str,Any]:
 return {"format":"vokra-kyutai-stt-2.6b-en-inspection-v1","status":"BLOCKED","inspection_status":"PENDING","evidence_stage":"INSPECTION_ONLY","runtime_status":"NOT_IMPLEMENTED_FAIL_CLOSED","cpu_status":"UNSUPPORTED","metal_status":"BLOCKED_BY_CPU","parity_status":"NOT_RUN","publication":"NO_UPLOAD","model":{"repository":REPO,"revision":REV,"license":"CC-BY-4.0","files":ARTIFACTS,"total_bytes":TOTAL},"historical_public_artifact":HISTORICAL_PUBLIC_ARTIFACT,"blockers":["real model/Mimi/tokenizer tensor binder is not implemented","native STT forward and independent CPU parity are not complete","dataset provenance and dependencies require separate review","Metal backend is blocked by CPU completion"]}
def inspect(a:argparse.Namespace)->int:
 m=base()
 try:
  snap=Path(a.snapshot); m["server_tree"]=server_tree(json.loads(Path(a.server_tree).read_text(encoding="utf-8"),object_pairs_hook=unique),snap); fs=files(snap)
  require_fixed_model_files(set(fs))
  if sum(p.stat().st_size for p in fs.values())!=TOTAL: raise ValueError("fixed six-file tree/total mismatch")
  readme=fs["README.md"].read_text(encoding="utf-8").lower()
  markers=["license: cc-by-4.0","streaming","mimi","12.5","32","english-only","2.5","dataset"]
  if any(x not in readme for x in markers): raise ValueError("README evidence marker missing")
  m["model_card"]={"bytes":fs["README.md"].stat().st_size,"sha256":sha(fs["README.md"]),"markers":markers}
  m["config"]={"bytes":fs["config.json"].stat().st_size,"sha256":sha(fs["config.json"]),"json":json.loads(fs["config.json"].read_text(encoding="utf-8"),object_pairs_hook=unique)}
  cfg=m["config"]["json"]; expected={"card":2048,"n_q":32,"dep_q":0,"delays":[0]*33,"dim":2048,"text_card":4000,"existing_text_padding_id":3,"num_heads":32,"num_layers":48,"hidden_scale":4.125,"causal":True,"layer_scale":None,"context":375,"max_period":100000.0,"gating":"silu","norm":"rms_norm_f32","positional_embedding":"rope","depformer_dim":1024,"depformer_num_heads":16,"depformer_num_layers":6,"depformer_dim_feedforward":None,"depformer_multi_linear":True,"depformer_pos_emb":"none","depformer_weights_per_step":True,"conditioners":{},"cross_attention":False,"model_id":{"sig":"dabcc802","epoch":50},"lm_gen_config":{"temp":0.0,"temp_text":0.0,"top_k":250,"top_k_text":50},"stt_config":{"audio_delay_seconds":2.5,"audio_silence_prefix_seconds":1.0},"model_type":"stt","mimi_name":"mimi-pytorch-e351c8d8@125.safetensors","tokenizer_name":"tokenizer_en_audio_4000.model"}
  if set(cfg)!=set(expected) or any(cfg.get(k)!=v for k,v in expected.items()): raise ValueError("config exact axes mismatch")
  m["weights"]={}
  for name in ("model.safetensors","mimi-pytorch-e351c8d8@125.safetensors"):
   h=st_header(fs[name]); m["weights"][name]={"header":h,"bytes":fs[name].stat().st_size,"sha256":sha(fs[name]),"tensor_count":len(h["tensors"])}
  try:
   import sentencepiece as spm
   sp=spm.SentencePieceProcessor(model_file=str(fs[TOKENIZER_NAME])); m["tokenizer"]={"name":TOKENIZER_NAME,"piece_count":sp.get_piece_size(),"first": [sp.id_to_piece(i) for i in range(min(20,sp.get_piece_size()))],"last":[sp.id_to_piece(i) for i in range(max(0,sp.get_piece_size()-10),sp.get_piece_size())],"unk_id":sp.unk_id(),"bos_id":sp.bos_id(),"eos_id":sp.eos_id(),"pad_id":sp.pad_id()}
   if sp.get_piece_size()!=4000: raise ValueError("tokenizer text_card mismatch")
  except Exception as e: raise ValueError(f"SentencePiece inspection failed: {e}") from e
  m["official_source"]=source(Path(a.source),SOURCE_URL,SOURCE_REV,ROLES) if a.source else (_ for _ in ()).throw(ValueError("source required"))
  m["moshi_source"]=source(Path(a.moshi_source),MOSHI_URL,MOSHI_REV,MOSHI_ROLES) if a.moshi_source else (_ for _ in ()).throw(ValueError("Moshi source required"))
  m["inspection_status"]="AUTHENTICATED_EVIDENCE_COMPLETE"
 except Exception as e:
  m["inspection_status"]="INSPECTION_ERROR"; m["blockers"].append(f"inspection error: {type(e).__name__}: {e}")
 Path(a.evidence).mkdir(parents=True,exist_ok=True); (Path(a.evidence)/"manifest.json").write_text(json.dumps(m,indent=2,sort_keys=True)+"\n")
 return 2
def self_test()->None:
 assert TOKENIZER_NAME=="tokenizer_en_audio_4000.model" and ARTIFACTS[TOKENIZER_NAME]==(59339,"1820a7cbb15efc6a33dd365113c07e3df9d28d80","d461765ae179566678c93091c5fa6f2984c31bbe990bf1aa62d92c64d91bc3f6")
 assert LEGACY_TOKENIZER_NAME not in MODEL_FILES and LEGACY_TOKENIZER_NAME not in ARTIFACTS
 assert sum(v[0] for v in ARTIFACTS.values())==5618985925
 for bad in (MODEL_FILES-{TOKENIZER_NAME},(MODEL_FILES-{TOKENIZER_NAME})|{LEGACY_TOKENIZER_NAME},MODEL_FILES|{LEGACY_TOKENIZER_NAME}):
  try:require_fixed_model_files(bad)
  except ValueError:pass
  else:raise AssertionError("legacy/missing/duplicate tokenizer tree accepted")
 try:json.loads('{"x":1,"x":2}',object_pairs_hook=unique)
 except ValueError:pass
 else:raise AssertionError("duplicate JSON key accepted")
 try:json.loads('{"x":{"dtype":"F32","dtype":"BF16"}}',object_pairs_hook=unique)
 except ValueError:pass
 else:raise AssertionError("duplicate descriptor key accepted")
 for bad in ("../x","/x","a\\b","a\x00b",""):
  try:safe(bad)
  except ValueError:pass
  else:raise AssertionError("unsafe path accepted")
 with tempfile.TemporaryDirectory(prefix="kyutai-stt-") as d:
  root=Path(d)/"s";root.mkdir(); p=root/"x";p.write_text("x")
  (root/".cache").mkdir();(root/".cache"/"bad.json").write_text("{}")
  assert set(files(root))=={"x"}
  packet={"repository":REPO,"revision":REV,"resolved_revision":REV,"files":[{"path":"x","type":"file","size":1,"git_blob_sha1":"1"*40,"lfs_sha256":sha(p)}]}
  server_tree(packet,root,False)
  try:server_tree(packet,root)
  except ValueError:pass
  else:raise AssertionError("unfixed artifact identity accepted")
  p.write_text("y")
  try:server_tree(packet,root,False)
  except ValueError:pass
  else:raise AssertionError("same-size content mutation accepted")
  p.write_text("x")
  for bad in (dict(packet,repository="bad"),dict(packet,files=[]),dict(packet,files=[dict(packet["files"][0],lfs_sha256="bad")])):
   try:server_tree(bad,root)
   except ValueError:pass
   else:raise AssertionError("bad server tree accepted")
  good_archive=Path(d)/"good.tar"
  with tarfile.open(good_archive,"w") as tar:
   info=tarfile.TarInfo("folder/");info.type=tarfile.DIRTYPE;tar.addfile(info)
   info=tarfile.TarInfo("folder/payload");info.size=1;tar.addfile(info,io.BytesIO(b"x"))
  assert archive_inventory(good_archive)["format"]=="tar"
  safe_tensor=Path(d)/"tiny.safetensors"; header=json.dumps({"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}).encode()
  safe_tensor.write_bytes(len(header).to_bytes(8,"little")+header+b"\0"*4)
  assert st_header(safe_tensor)["tensors"]["x"]["bytes"]==4
  def reject_header(descriptors, payload):
   raw=json.dumps(descriptors).encode(); candidate=Path(d)/f"bad-{len(list(Path(d).glob('bad-*')))}.safetensors"
   candidate.write_bytes(len(raw).to_bytes(8,"little")+raw+payload)
   try:st_header(candidate)
   except ValueError:pass
   else:raise AssertionError("malformed safetensors accepted")
  reject_header({"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"y":{"dtype":"F32","shape":[1],"data_offsets":[3,7]}},b"\0"*7)
  reject_header({"x":{"dtype":"F32","shape":[1],"data_offsets":[0,5]}},b"\0"*5)
  reject_header({"../bad":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}},b"\0"*4)
  reject_header({"x":{"dtype":"I64","shape":[1],"data_offsets":[0,8]}},b"\0"*8)
  malformed=Path(d)/"malformed.safetensors"; malformed.write_bytes((len(header)).to_bytes(8,"little")+header+b"\0"*3)
  try:st_header(malformed)
  except ValueError:pass
  else:raise AssertionError("malformed safetensors accepted")
  bad_archive=Path(d)/"bad.tar"
  with tarfile.open(bad_archive,"w") as tar:
   info=tarfile.TarInfo("../escape");info.size=1;tar.addfile(info,io.BytesIO(b"x"))
  try:archive_inventory(bad_archive)
  except ValueError:pass
  else:raise AssertionError("bad archive accepted")
  duplicate_archive=Path(d)/"duplicate.tar"
  with tarfile.open(duplicate_archive,"w") as tar:
   for _ in range(2):
    info=tarfile.TarInfo("same");info.size=1;tar.addfile(info,io.BytesIO(b"x"))
  try:archive_inventory(duplicate_archive)
  except ValueError:pass
  else:raise AssertionError("duplicate archive member accepted")
  out=Path(d)/"e"; assert inspect(argparse.Namespace(snapshot=str(root/"missing"),server_tree=str(root/"missing.json"),source=None,moshi_source=None,evidence=str(out)))==2
  mm=json.loads((out/"manifest.json").read_text()); assert mm["inspection_status"]=="INSPECTION_ERROR"
 print("kyutai STT inspector self-test PASS")
def main()->int:
 p=argparse.ArgumentParser();p.add_argument("--self-test",action="store_true");p.add_argument("--snapshot");p.add_argument("--server-tree");p.add_argument("--source");p.add_argument("--moshi-source");p.add_argument("--evidence",default="evidence");a=p.parse_args()
 if a.self_test:self_test();return 0
 if not a.snapshot or not a.server_tree:p.error("snapshot and server-tree required")
 return inspect(a)
if __name__=="__main__":raise SystemExit(main())
