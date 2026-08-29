#!/usr/bin/env -S uv run --frozen --project tools/parity --python 3.12 python
"""VAST-only Dia-1.6B identity and safe-structure inspector; never converts or executes.

The companion worker obtains the packet with ``list_repo_tree`` and
``snapshot_download`` after ``cargo metadata --locked --no-deps --format-version 1``;
its expected result is an evidence manifest plus exit 2.  These are workflow
markers, not a permission to download or convert locally.
"""
from __future__ import annotations
import argparse,base64,hashlib,json,re,struct,subprocess,tempfile,zipfile,zlib
from pathlib import Path
from typing import Any
HF_REPOSITORY="nari-labs/Dia-1.6B"
HF_REVISION="257bc72f9b78182ccc6fa07675a9ae4c1a44e2cd"
SOURCE_REPOSITORY="https://github.com/nari-labs/dia.git"
SOURCE_REVISION="2811af1c5f476b1f49f4744fabf56cf352be21e5"
PUBLIC_REPOSITORY="vokra/dia-1.6b"
PUBLIC_REVISION="dd1df2a129fed7d15c365caeabaae227ccfe8537"
FORMAT="vokra-dia-1-6b-inspection-v1"
TOTAL_BYTES=12900029096
MANIFEST_SHA256="55fce2a39cafba838bd800f6a6aefe63a8e3b1dd86f2727f9a20d87fe6d252f7"
EXPECTED_FILES={
 ".gitattributes":(1519,"a6344aac8c09253b3b630fb776ae94478aa0275b",None),
 "README.md":(6551,"146916d420c1f14cf794a171811ac56e42d13dbd",None),
 "config.json":(941,"0a586180c3246fefa312c5e3977a6e419a7a113d",None),
 "preprocessor_config.json":(172,"a812a82c392c511dc04417b8f8bcde9411347af0",None),
 "dia-v0_1.pth":(6444788896,"8dc5b43681e210512ee8dbf8d028737c7f449180","d12004b2f3121af763bdf2a3b575586b00c02bdd00a315a23c7b7bdb2a8f9475"),
 "model.safetensors":(6444682848,"d3cc9ed6f729aa7894307b7799aafe9330853c48","caba289b60f6d7d1e58fc744f4dc25aae88995fcca46be3d05e220b971486a26"),
}
EXPECTED_GGUF={"path":"dia-1.6b.gguf","bytes":6444673088,"git_blob_sha1":"e00731cd617132cf198f7bcaaee190de2df86c5f","lfs_sha256":"a90733e9e6806cae66abf3eca1d575ecf6dab9298c07d39fc4217a509c952a6d"}
SOURCE_ROLES=("LICENSE","dia/audio.py","dia/config.py","dia/layers.py","dia/model.py","dia/state.py","pyproject.toml")
SOURCE_ROLE_BLOBS={"LICENSE":"483d716cc886695f19971a99658c59851a8a2866","dia/audio.py":"5c1947103bc0d95255d97618c699fa0a18993beb","dia/config.py":"09c6d136a41e0296483d2617061d4261cbf4c42c","dia/layers.py":"f9aed506b25e99d053dd71d6def7a0bd33075ace","dia/model.py":"a3b0f9730a810fa170019511a2696e7f813090de","dia/state.py":"172ec52c7c344781aad0552a6cddd6e5f1933894","pyproject.toml":"dd844dd2fb0ab0c016520c4b070beaa7c159e3e1"}
# Canonical (name, shape, dtype) list from the fixed 343-tensor public main-model contract.
_MANIFEST_B64="eNq1nN1qGlEURt/F60Hc5//0AfoSIYiNk8Q2OsYxCaH03ZvJpdXo1G9dFtKPQdcsNizw5vdkuX/ftpNvk+/eTZrJZrEe/rFs77plu5u26x/tcrnaPPTT2fStXT087j/+qH9cDP/lxmauNG4Wyu2f5vIhUw051ZBXDQXVUFQNJdVQVg2Vq4aeFu/tboDxbtf1/Xyx37eb/arbTH/Nt7vu59Ht0FhqzF2z3p1a/xy+9tmfT6wPs9c/+6v0k1k/badvq/n9S98uTzyxa4pVN3az+3dtmBn72W537fxuMd90u/Xxxxu59fFsurFe9GB9+3R/Afuf30YY+QUfbGvJPxjXgn8w/qr7VAzVjaG6MVQ3hurGAN2YUDcm1I0pdWNC3RioGyN1Y6RujNQNe94Ye98Ye+AYe+EYceKY8sYx5ZFj0ivHlGeOkXeOoYeOoZeOoacOfOvAxw587cDnDnLvSA8e6cWjPXmkNw969LBXD3v2oHePY93jWPc41j2OdY8j3OOU7nFK9zipe5zSPY50j0Pd41D3ONI9nnWPZ93jWfd41j2ecI9Xuscr3eOl7vFK93jSPR51j0fd40n3BNY9gXVPYN0TWPcEwj1B6Z6gdE+Quico3RNI9wTUPQF1TyDdE1n3RNY9kXVPZN0TCfdEpXui0j1R6p6odE8k3RNR90TUPZF0T2Ldk1j3JNY9iXVPItyTlO5JSvckqXuS0j2JdE9C3ZNQ9yTSPZl1T2bdk1n3ZNY9mXBPVronK92Tpe7JSvdk0j0ZdU9G3ZNB97CZi61cbORiGxeRuJSFSxm4pH1LmbfIuoXGLbRtkWmLLVts2GK7Fpu1iKqljFrKpiVNWsqiRQYttGehOYusWWzMYlsWm7LYkkWELGXHUmYsacVSRiyyYaEJCy1YZMBi+xWbr9h6xcYrol0p05WyXEnDlbJbkdkKrVZotCKbFZus2GLFBiu2VxG5SlmrlLFK2qqUqYosVWioQjsVmanYSsVGKrZRsYmKKFTKQKXsU9I8paxTZJxC2xSapsgyVVDdFFQ3BdVNQXVTAN0UoW6KUDdFqZsi1E0BdVNI3RRSNwXUTUV1U1HdVFQ3FdVNBXRThbqpQt1UpW6qUDcV1E0ldVNJ3VREN93Dat/Pl+2mb08s1Wb4FcLzU//zxbebg19BPDIQ0/AA4fzIhb/E96kF14RZTWM3j7zCw8zY59t2/f70qzJ27KvXbuTWha/dRWI9N37mvbvuyZ/JJ38VjhsAqwlhNSWsJoTVSFiNhNVIWA2FlVCrKd1qUrma0q6G6tVQvxoqWGMNiyhW6litZKWWZTXLepYVLWpaBzDrhMg6JbFOCKwjeXUkro6k1ZGwegBWL4TVK2H1Qlg9CasnYfUkrJ6ENQCwBiGsQQlrEMIaSFgDCWsgYQ0krBGANQphjUpYoxDWSMIaSVgjCWskYU0ArEkIa1LCmoSwJhLWRMKaSFgTCWsGYM1CWLMS1iyENZOwZhLWTMKaSVgLAGsRwlqUsBYhrIWEtZCwFhLWQsJaAVirENaqhLUKYa0krJWEtZKwVgTWr76v27+auqA9"
EXPECTED_TENSORS=json.loads(zlib.decompress(base64.b64decode(_MANIFEST_B64)))
DTYPE_BYTES={"F32":4,"F16":2,"BF16":2,"F64":8,"I8":1,"U8":1,"I16":2,"U16":2,"I32":4,"U32":4,"I64":8,"U64":8}

def no_dupes(pairs):
 out={}
 for k,v in pairs:
  if k in out: raise ValueError(f"duplicate JSON key: {k}")
  out[k]=v
 return out
def sha256(path):
 h=hashlib.sha256()
 with path.open("rb") as f:
  for block in iter(lambda:f.read(1<<20),b""): h.update(block)
 return h.hexdigest()
def git_blob(path):
 h=hashlib.sha1(); h.update(f"blob {path.stat().st_size}\0".encode())
 with path.open("rb") as f:
  for block in iter(lambda:f.read(1<<20),b""): h.update(block)
 return h.hexdigest()
def safe_path(name):
 return isinstance(name,str) and bool(name) and "\0" not in name and "\\" not in name and not name.startswith("/") and all(p not in ("",".","..") for p in Path(name).parts)
def local_files(root):
 if not root.is_dir(): raise RuntimeError(f"missing snapshot: {root}")
 out=[]
 for p in sorted(root.rglob("*")):
  rel=p.relative_to(root)
  if any(x in {".cache",".git"} for x in rel.parts): continue
  if p.is_symlink():
   if not p.exists() or not p.is_file() or not p.resolve().is_relative_to(root.resolve()): raise RuntimeError(f"unsafe symlink: {p}")
  elif p.is_dir(): continue
  elif not p.is_file(): raise RuntimeError(f"non-regular file: {p}")
  out.append(p)
 return out
def identity(path,root):
 return {"path":path.relative_to(root).as_posix(),"bytes":path.stat().st_size,"sha256":sha256(path),"git_blob_sha1":git_blob(path)}
def lfs_pointer_sha1(oid,size):
 raw=f"version https://git-lfs.github.com/spec/v1\noid sha256:{oid}\nsize {size}\n".encode()
 h=hashlib.sha1(); h.update(f"blob {len(raw)}\0".encode()); h.update(raw); return h.hexdigest()
def canonical_manifest_hash(items):
 buf=bytearray()
 for x in sorted(items,key=lambda x:x["name"]):
  buf.extend(x["name"].encode()); buf.append(0); buf.extend(struct.pack("<Q",len(x["shape"])))
  for dim in x["shape"]: buf.extend(struct.pack("<Q",dim))
 return hashlib.sha256(buf).hexdigest()

def server_tree(snapshot,packet,blockers):
 remote=json.loads(packet.read_text(encoding="utf-8"),object_pairs_hook=no_dupes)
 rows=remote.get("files") if isinstance(remote,dict) else None
 if not isinstance(rows,list): raise ValueError("server tree files is not an array")
 records={}
 for row in rows:
  if not isinstance(row,dict) or row.get("type")!="file" or not safe_path(row.get("path")) or not isinstance(row.get("size"),int) or isinstance(row.get("size"),bool) or row["size"]<0: raise ValueError(f"invalid server row: {row!r}")
  name=row["path"]
  if name in records: raise ValueError(f"duplicate server path: {name}")
  git=row.get("git_blob_sha1"); lfs=row.get("lfs_sha256")
  if not isinstance(git,str) or not re.fullmatch(r"[0-9a-f]{40}",git): raise ValueError(f"invalid Git blob: {name}")
  lfs_size=row.get("lfs_size")
  if lfs is not None and (not isinstance(lfs,str) or not re.fullmatch(r"[0-9a-f]{64}",lfs) or lfs_size != row["size"]): raise ValueError(f"invalid LFS SHA/size: {name}")
  if lfs is not None and lfs_pointer_sha1(lfs,lfs_size) != git: raise ValueError(f"invalid LFS pointer Git blob: {name}")
  records[name]={"bytes":row["size"],"git_blob_sha1":git,"lfs_sha256":lfs,"lfs_size":lfs_size}
 local={p.relative_to(snapshot).as_posix():identity(p,snapshot) for p in local_files(snapshot)}
 missing=sorted(set(records)-set(local)); extra=sorted(set(local)-set(records)); mismatch=[]
 for name in sorted(set(records)&set(local)):
  e,a=records[name],local[name]
  good=a["bytes"]==e["bytes"] and (a["sha256"]==e["lfs_sha256"] and e["lfs_size"]==a["bytes"] if e["lfs_sha256"] else a["git_blob_sha1"]==e["git_blob_sha1"])
  if not good: mismatch.append(name)
 ident=remote.get("repository")==HF_REPOSITORY and remote.get("revision")==HF_REVISION and remote.get("resolved_revision")==HF_REVISION and remote.get("walk")=="recursive_file_only"
 if not ident: blockers.append("HF server tree identity/walk mismatch")
 if missing or extra: blockers.append(f"server/local tree mismatch: missing={missing!r} extra={extra!r}")
 if mismatch: blockers.append(f"server/local content mismatch: {mismatch!r}")
 return {"status":"MATCHED" if ident and not missing and not extra and not mismatch else "MISMATCH","repository":remote.get("repository"),"revision":remote.get("revision"),"resolved_revision":remote.get("resolved_revision"),"walk":remote.get("walk"),"files":records,"missing":missing,"extra":extra,"content_mismatch":mismatch,"packet_sha256":sha256(packet)}

def safe_header(path,root,blockers,expected=True):
 item=identity(path,root); size=path.stat().st_size
 try:
  with path.open("rb") as f:
   raw=f.read(8)
   if len(raw)!=8: raise ValueError("short header length")
   length=struct.unpack("<Q",raw)[0]
   if length==0 or length>64*1024*1024 or length>size-8: raise ValueError("unsafe header length")
   header=json.loads(f.read(length),object_pairs_hook=no_dupes)
  if not isinstance(header,dict): raise ValueError("header is not object")
  metadata=header.get("__metadata__",{})
  if not isinstance(metadata,dict) or any(not isinstance(k,str) or not isinstance(v,str) for k,v in metadata.items()): raise ValueError("metadata must be string map")
  start=8+length; ranges=[]; tensors=[]
  for name,spec in sorted(header.items()):
   if name=="__metadata__": continue
   if not safe_path(name) or not isinstance(spec,dict) or set(spec)!={"dtype","shape","data_offsets"}: raise ValueError(f"invalid descriptor {name}")
   dtype,shape,offsets=spec["dtype"],spec["shape"],spec["data_offsets"]
   if dtype not in DTYPE_BYTES or not isinstance(shape,list) or not isinstance(offsets,list) or len(offsets)!=2 or any(isinstance(x,bool) or not isinstance(x,int) or x<0 for x in shape+offsets): raise ValueError(f"invalid descriptor types {name}")
   n=1
   for d in shape: n*=d
   begin,end=offsets
   if end<begin or end-begin != n*DTYPE_BYTES[dtype] or start+end>size: raise ValueError(f"invalid range {name}")
   ranges.append((begin,end)); tensors.append({"name":name,"shape":shape,"dtype":dtype,"numel":n})
  cursor=0
  for begin,end in sorted(ranges):
   if begin<cursor: raise ValueError("tensor overlap")
   if begin>cursor: raise ValueError("tensor gap")
   cursor=end
  if cursor != size-start: raise ValueError("tensor data does not end at boundary")
  if expected and (len(tensors)!=343 or canonical_manifest_hash(tensors)!=MANIFEST_SHA256): raise ValueError("official 343 tensor manifest mismatch")
  if expected and sorted((x["name"],x["shape"],x["dtype"]) for x in tensors) != sorted((x["name"],x["shape"],x["dtype"]) for x in EXPECTED_TENSORS): raise ValueError("tensor names/shapes/dtypes differ from fixed 343-tensor main-model contract")
  return {**item,"status":"HEADER_ONLY","header_bytes":length,"metadata":metadata,"tensor_count":len(tensors),"tensors":tensors,"manifest_sha256":canonical_manifest_hash(tensors),"resident_scope":"header only; body never read"}
 except Exception as error:
  blockers.append(f"safe header blocked {path}: {error}"); return {**item,"status":"BLOCKED_HEADER","error":str(error)}

def pth_inventory(path,root,blockers):
 item=identity(path,root)
 try:
  with zipfile.ZipFile(path) as archive:
   if len(archive.infolist())>100000: raise ValueError("archive member bound exceeded")
   names=set()
   for info in archive.infolist():
    if info.filename in names or not safe_path(info.filename) or info.is_dir() or ((info.external_attr>>16)&0o170000 not in (0,0o100000)): raise ValueError(f"unsafe archive member: {info.filename}")
    names.add(info.filename)
  import torch
  scanner=getattr(torch.serialization,"get_unsafe_globals_in_checkpoint",None)
  if scanner is None: raise RuntimeError("unsafe-global scanner unavailable")
  unsafe=scanner(str(path))
  if unsafe: raise RuntimeError(f"unsafe globals present: {unsafe[:8]}")
  obj=torch.load(path,map_location="cpu",weights_only=True)
  seen=set(); records=[]
  def walk(value,name,depth=0):
   if depth>32: raise ValueError("checkpoint depth exceeded")
   if isinstance(value,torch.Tensor):
    if name in seen: raise ValueError(f"duplicate tensor path: {name}")
    seen.add(name); records.append({"path":name,"shape":list(value.shape),"dtype":str(value.dtype),"numel":value.numel(),"finite":bool(torch.isfinite(value).all().item()) if value.is_floating_point() else True}); return
   if isinstance(value,dict):
    if len(value)>100000: raise ValueError("mapping bound exceeded")
    for key,child in value.items():
     if not isinstance(key,str) or not safe_path(key): raise ValueError(f"unsafe checkpoint key: {key!r}")
     walk(child,f"{name}.{key}" if name else key,depth+1)
   elif isinstance(value,(list,tuple)):
    if len(value)>100000: raise ValueError("sequence bound exceeded")
    for i,child in enumerate(value): walk(child,f"{name}[{i}]",depth+1)
   elif value is None or isinstance(value,(str,int,float,bool)): return
   else: raise ValueError(f"unsupported checkpoint object: {type(value).__name__}")
  walk(obj,"")
  return {**item,"inventory":{"status":"SAFE_WEIGHTS_ONLY_INVENTORY","tensor_count":len(records),"tensors":records,"manifest_sha256":hashlib.sha256(json.dumps(records,sort_keys=True,separators=(",",":")).encode()).hexdigest(),"resident_scope":"CPU tensor inventory only; no forward"}}
 except Exception as error:
  blockers.append(f"PTH safe inventory blocked: {error}"); return {**item,"inventory":{"status":"BLOCKED_PTH","error":str(error)}}

def public_gguf(path,root,blockers):
 """Read GGUF metadata/tensor descriptors through the pinned reader only."""
 item=identity(path,root)
 item["body_git_blob_sha1"]=item.pop("git_blob_sha1")
 item["git_blob_sha1"]=lfs_pointer_sha1(EXPECTED_GGUF["lfs_sha256"],EXPECTED_GGUF["bytes"])
 try:
  import gguf
  reader=gguf.GGUFReader(str(path),mode="r")
  if int(reader.data_offset)!=30762: raise ValueError(f"header bytes {reader.data_offset} != 30762")
  if len(reader.tensors)!=343: raise ValueError(f"tensor count {len(reader.tensors)} != 343")
  tensors=[]
  for tensor in reader.tensors:
   name=tensor.name; shape=[int(x) for x in tensor.shape]
   dtype=str(tensor.tensor_type).split(".")[-1]
   tensors.append({"name":name,"shape":shape,"dtype":dtype,"numel":int(tensor.n_elements),"offset":int(tensor.data_offset)})
  normalized=sorted((x["name"],x["shape"],x["dtype"]) for x in tensors)
  expected=sorted((x["name"],x["shape"],x["dtype"]) for x in EXPECTED_TENSORS)
  if normalized!=expected: raise ValueError("historical GGUF names/shapes/dtypes differ from fixed 343-tensor main-model contract")
  metadata={}
  for key in ("general.architecture","general.name","vokra.model.arch","vokra.model.name","vokra.provenance.license","vokra.provenance.weight_license"):
   field=reader.get_field(key)
   if field is not None:
    value=field.parts[-1][0]
    if hasattr(value,"item"): value=value.item()
    if isinstance(value,bytes): value=value.decode("utf-8")
    metadata[key]=value
  if metadata.get("vokra.model.arch")!="dia" or metadata.get("vokra.model.name")!="dia-1.6b": raise ValueError("historical GGUF Dia identity metadata mismatch")
  return {**item,"status":"AUTHENTICATED_PARTIAL_GGUF","header_bytes":int(reader.data_offset),"tensor_count":len(tensors),"manifest_sha256":MANIFEST_SHA256,"metadata":metadata,"tensors":tensors,"resident_scope":"GGUF descriptors via read-only mmap; tensor values not loaded"}
 except Exception as error:
  blockers.append(f"historical GGUF contract blocked: {error}"); return {**item,"status":"BLOCKED_PUBLIC_GGUF","error":str(error)}

def source_inventory(source,blockers):
 out={"repository":SOURCE_REPOSITORY,"pinned_revision":SOURCE_REVISION}
 try:
  head=subprocess.run(["git","-C",str(source),"rev-parse","HEAD"],check=True,capture_output=True,text=True).stdout.strip()
  origin=subprocess.run(["git","-C",str(source),"remote","get-url","origin"],check=True,capture_output=True,text=True).stdout.strip()
  status=subprocess.run(["git","-C",str(source),"status","--porcelain","--untracked-files=all"],check=True,capture_output=True,text=True).stdout
  out.update({"resolved_revision":head,"origin":origin,"clean":not status,"roles":[]})
  if head!=SOURCE_REVISION or origin!=SOURCE_REPOSITORY or status: blockers.append("Dia source identity/origin/clean mismatch")
  for role in SOURCE_ROLES:
   path=source/role
   if not path.is_file(): blockers.append(f"source role missing: {role}"); continue
   record=identity(path,source); record["expected_git_blob_sha1"]=SOURCE_ROLE_BLOBS[role]; out["roles"].append(record)
   if record["git_blob_sha1"]!=SOURCE_ROLE_BLOBS[role]: blockers.append(f"source role identity mismatch: {role}")
  lic=(source/"LICENSE").read_text(encoding="utf-8")
  if "Apache License" not in lic or "Version 2.0" not in lic: blockers.append("Dia source LICENSE is not Apache-2.0")
  out["license"]={"path":"LICENSE","declaration":"Apache-2.0","sha256":sha256(source/"LICENSE")}
 except Exception as error: blockers.append(f"source inventory failed: {error}"); out["error"]=str(error)
 return out

def inspect(snapshot,source,tree,output,public=None):
 blockers=[]; packet=server_tree(snapshot,tree,blockers); local=local_files(snapshot)
 for name,(size,git,lfs) in EXPECTED_FILES.items():
  path=snapshot/name
  if not path.is_file(): blockers.append(f"required HF file missing: {name}"); continue
  row=identity(path,snapshot)
  server_row=packet["files"].get(name)
  pointer_ok=lfs is None or lfs_pointer_sha1(lfs,size)==git
  if row["bytes"]!=size or row["git_blob_sha1"]!=git or (lfs and row["sha256"]!=lfs) or not server_row or server_row.get("bytes")!=size or server_row.get("git_blob_sha1")!=git or server_row.get("lfs_sha256")!=lfs or (lfs and server_row.get("lfs_size")!=size) or not pointer_ok: blockers.append(f"fixed HF identity mismatch: {name}")
 total=sum(p.stat().st_size for p in local)
 if total!=TOTAL_BYTES: blockers.append(f"HF total bytes mismatch: {total}")
 config={}; config_packet=None
 try:
  config=json.loads((snapshot/"config.json").read_text(encoding="utf-8"),object_pairs_hook=no_dupes)
  facts={("model","src_vocab_size"):256,("model","tgt_vocab_size"):1028,("model","encoder","n_layer"):12,("model","encoder","n_embd"):1024,("model","encoder","n_head"):16,("model","encoder","head_dim"):128,("model","encoder","n_hidden"):4096,("model","decoder","n_layer"):18,("model","decoder","n_embd"):2048,("model","decoder","gqa_query_heads"):16,("model","decoder","kv_heads"):4,("model","decoder","gqa_head_dim"):128,("model","decoder","cross_query_heads"):16,("model","decoder","cross_head_dim"):128,("model","decoder","n_hidden"):8192,("model","normalization_layer_epsilon"):1e-5,("model","rope_max_timescale"):10000,("model","rope_min_timescale"):1,("data","channels"):9,("data","text_length"):1024,("data","audio_length"):3072,("data","text_pad_value"):0,("data","audio_bos_value"):1026,("data","audio_eos_value"):1024,("data","audio_pad_value"):1025,("data","delay_pattern"): [0,8,9,10,11,12,13,14,15]}
  observed={}
  for parts,want in facts.items():
   value=config
   try:
    for part in parts: value=value[part]
    observed[".".join(parts)]=value
    if value!=want: blockers.append(f"config fact mismatch: {'.'.join(parts)}")
   except (KeyError,TypeError): blockers.append(f"config fact missing: {'.'.join(parts)}")
  if not isinstance(config,dict) or not isinstance(config.get("model"),dict) or not isinstance(config.get("data"),dict): blockers.append("config model/data sections are malformed")
  config_packet={"status":"EXACT_PRIMARY_JSON" if not any(x.startswith("config fact") or x.startswith("config model") for x in blockers) else "BLOCKED_CONFIG_FACTS","identity":identity(snapshot/"config.json",snapshot),"facts":observed,"raw":config}
 except Exception as error: blockers.append(f"config parse failed: {error}")
 readme_evidence={"status":"NOT_CHECKED"}
 try:
  readme=(snapshot/"README.md").read_text(encoding="utf-8")
  if not re.search(r"(?im)^license\s*:\s*apache-2\.0\s*$",readme) and "Apache License, Version 2.0" not in readme: raise ValueError("README does not declare Apache-2.0")
  readme_evidence={"status":"APACHE_2_0_DECLARATION","identity":identity(snapshot/"README.md",snapshot)}
 except Exception as error: blockers.append(f"README license evidence blocked: {error}")
 preprocessor_evidence={"status":"NOT_CHECKED"}
 try:
  pre=json.loads((snapshot/"preprocessor_config.json").read_text(encoding="utf-8"),object_pairs_hook=no_dupes)
  if not isinstance(pre,dict): raise ValueError("preprocessor config is not object")
  preprocessor_evidence={"status":"PARSED_PRIMARY_JSON","identity":identity(snapshot/"preprocessor_config.json",snapshot),"raw":pre}
 except Exception as error: blockers.append(f"preprocessor config parse failed: {error}")
 st=safe_header(snapshot/"model.safetensors",snapshot,blockers) if (snapshot/"model.safetensors").is_file() else None
 pth=pth_inventory(snapshot/"dia-v0_1.pth",snapshot,blockers) if (snapshot/"dia-v0_1.pth").is_file() else None
 mapping={"status":"NOT_AVAILABLE"}
 if st and st.get("status")=="HEADER_ONLY" and pth and pth.get("inventory",{}).get("status")=="SAFE_WEIGHTS_ONLY_INVENTORY":
  safe_shapes={x["name"]:(x["shape"],x["dtype"]) for x in st["tensors"]}
  pth_shapes={x["path"]:(x["shape"],x["dtype"].replace("torch.","").upper()) for x in pth["inventory"]["tensors"]}
  mapping={"status":"EXACT_NAME_SHAPE_DTYPE" if safe_shapes==pth_shapes else "AUTHENTICATED_MAPPING_DIFFERENCE","safetensors_count":len(safe_shapes),"pth_count":len(pth_shapes),"only_safetensors":sorted(set(safe_shapes)-set(pth_shapes)),"only_pth":sorted(set(pth_shapes)-set(safe_shapes)),"shape_or_dtype_difference":sorted(k for k in set(safe_shapes)&set(pth_shapes) if safe_shapes[k]!=pth_shapes[k])}
 else:
  mapping={"status":"UNAVAILABLE","reason":"both safe tensor-header and weights-only PTH inventories are required"}
  blockers.append("PTH↔safetensors mapping evidence unavailable")
 src=source_inventory(source,blockers)
 public_evidence={"status":"NOT_SUPPLIED","contract":{"repository":PUBLIC_REPOSITORY,"revision":PUBLIC_REVISION,"artifact":EXPECTED_GGUF,"tensor_count":343,"header_bytes":30762,"manifest_sha256":MANIFEST_SHA256,"contract_source":"fixed public GGUF contract; body comparison requires VAST materialization"}}
 if public:
  if public.is_file():
   row=identity(public,public.parent); row["body_git_blob_sha1"]=row.pop("git_blob_sha1"); row["git_blob_sha1"]=lfs_pointer_sha1(EXPECTED_GGUF["lfs_sha256"],EXPECTED_GGUF["bytes"]); public_evidence["local_identity"]=row
   if row["bytes"]!=EXPECTED_GGUF["bytes"] or row["sha256"]!=EXPECTED_GGUF["lfs_sha256"]: blockers.append("historical GGUF identity mismatch")
   public_evidence["header"]=public_gguf(public,public.parent,blockers)
  else: blockers.append("historical GGUF path is not a regular file")
 else: blockers.append("historical GGUF not supplied; VAST must inspect public composite-partial artifact")
 blockers.extend(["native Dia encoder/decoder delayed-AR math is staged but unauthenticated and uncompared","full PCM requires crate::dac::Dac plus accepted same-execution Dia AR evidence","DAC/tokenizer/generation parity is not run","CPU_UNSUPPORTED_FULL_TTS","Metal_BLOCKED_BY_CPU"])
 payload={"format":FORMAT,"status":"BLOCKED","evidence_stage":"INSPECTION_ONLY","inspection_status":"AUTHENTICATED_EVIDENCE_COMPLETE" if not any(x.startswith(("HF server","server/local","fixed HF","HF total","safe header","PTH safe","PTH↔safetensors","Dia source","source role","config","README license","preprocessor config","historical GGUF")) for x in blockers) else "INSPECTION_ERROR","runtime_status":"PARTIAL_RUNTIME_FAIL_CLOSED","cpu_status":"UNSUPPORTED_FULL_TTS","metal_status":"BLOCKED_BY_CPU","parity_status":"NOT_RUN","publication":"NO_UPLOAD","model":{"repository":HF_REPOSITORY,"revision":HF_REVISION,"expected_files":EXPECTED_FILES,"server_tree":packet,"files":[identity(p,snapshot) for p in local],"config":config_packet,"preprocessor_config":preprocessor_evidence,"readme_license":readme_evidence,"safetensors":st,"pth":pth,"checkpoint_mapping":mapping},"public_partial_artifact":public_evidence,"official_source":src,"blockers":sorted(set(blockers))}
 output.mkdir(parents=True,exist_ok=True); (output/"manifest.json").write_text(json.dumps(payload,sort_keys=True,indent=2)+"\n",encoding="utf-8"); return 2

def self_test():
 assert len(HF_REVISION)==len(SOURCE_REVISION)==len(PUBLIC_REVISION)==40
 assert len(EXPECTED_TENSORS)==343 and MANIFEST_SHA256==canonical_manifest_hash(EXPECTED_TENSORS)
 with tempfile.TemporaryDirectory(prefix="dia-inspect-") as d:
  root=Path(d); h=json.dumps({"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}).encode(); good=root/"x.safetensors"; good.write_bytes(struct.pack("<Q",len(h))+h+b"\0"*4); b=[]; assert safe_header(good,root,b,expected=False)["status"]=="HEADER_ONLY" and not b
  dup=root/"dup.safetensors"; raw=b'{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}'; dup.write_bytes(struct.pack("<Q",len(raw))+raw+b"\0"*4); b=[]; safe_header(dup,root,b,expected=False); assert any("blocked" in x for x in b)
  huge=root/"huge"; huge.write_bytes(struct.pack("<Q",65*1024*1024)+b"{}"); b=[]; safe_header(huge,root,b,expected=False); assert b
  snap=root/"snap"; snap.mkdir(); small=snap/"x"; small.write_bytes(b"abcd"); tree=root/"tree.json"; tree.write_text(json.dumps({"repository":HF_REPOSITORY,"revision":HF_REVISION,"resolved_revision":HF_REVISION,"walk":"recursive_file_only","files":[{"type":"file","path":"x","size":4,"git_blob_sha1":git_blob(small),"lfs_sha256":None}]})); b=[]; assert server_tree(snap,tree,b)["status"]=="MATCHED" and not b
  small.write_bytes(b"abce"); b=[]; assert server_tree(snap,tree,b)["status"]=="MISMATCH" and b
  lfs_file=snap/"lfs"; lfs_file.write_bytes(b"payload"); lfs_sha=sha256(lfs_file); lfs_pointer=lfs_pointer_sha1(lfs_sha,lfs_file.stat().st_size)
  lfs_tree=root/"lfs-tree.json"; lfs_tree.write_text(json.dumps({"repository":HF_REPOSITORY,"revision":HF_REVISION,"resolved_revision":HF_REVISION,"walk":"recursive_file_only","files":[{"type":"file","path":"x","size":4,"git_blob_sha1":git_blob(small),"lfs_sha256":None},{"type":"file","path":"lfs","size":7,"git_blob_sha1":lfs_pointer,"lfs_sha256":lfs_sha,"lfs_size":7}]})); b=[]; assert server_tree(snap,lfs_tree,b)["status"]=="MATCHED" and not b
  lfs_tree.write_text(json.dumps({"repository":HF_REPOSITORY,"revision":HF_REVISION,"resolved_revision":HF_REVISION,"walk":"recursive_file_only","files":[{"type":"file","path":"x","size":4,"git_blob_sha1":git_blob(small),"lfs_sha256":None},{"type":"file","path":"lfs","size":7,"git_blob_sha1":"0"*40,"lfs_sha256":lfs_sha,"lfs_size":7}]})); b=[]
  try: server_tree(snap,lfs_tree,b)
  except ValueError: pass
  else: raise AssertionError("invalid LFS pointer accepted")
  lfs_tree.write_text(json.dumps({"repository":HF_REPOSITORY,"revision":HF_REVISION,"resolved_revision":HF_REVISION,"walk":"recursive_file_only","files":[{"type":"file","path":"x","size":4,"git_blob_sha1":git_blob(small),"lfs_sha256":None},{"type":"file","path":"lfs","size":7,"git_blob_sha1":lfs_pointer,"lfs_sha256":lfs_sha,"lfs_size":6}]})); b=[]
  try: server_tree(snap,lfs_tree,b)
  except ValueError: pass
  else: raise AssertionError("invalid LFS size accepted")
  lfs_tree.write_text(json.dumps({"repository":HF_REPOSITORY,"revision":HF_REVISION,"resolved_revision":HF_REVISION,"walk":"recursive_file_only","files":[{"type":"file","path":"x","size":4,"git_blob_sha1":git_blob(small),"lfs_sha256":None},{"type":"file","path":"lfs","size":7,"git_blob_sha1":lfs_pointer,"lfs_sha256":lfs_sha,"lfs_size":7}]})); lfs_file.write_bytes(b"changed"); b=[]; assert server_tree(snap,lfs_tree,b)["status"]=="MISMATCH" and b
  assert lfs_pointer_sha1("0"*64,1) != git_blob(small)
 print("dia_1_6b_inspect self-test: OK")

def main():
 ap=argparse.ArgumentParser(); ap.add_argument("--self-test",action="store_true"); ap.add_argument("--snapshot",type=Path); ap.add_argument("--source",type=Path); ap.add_argument("--server-tree",type=Path); ap.add_argument("--public-gguf",type=Path); ap.add_argument("--output",type=Path); a=ap.parse_args()
 if a.self_test:
  if any(x is not None for x in (a.snapshot,a.source,a.server_tree,a.public_gguf,a.output)): ap.error("--self-test accepts no other arguments")
  self_test(); return 0
 if any(x is None for x in (a.snapshot,a.source,a.server_tree,a.output)): ap.error("normal run requires snapshot, source, server-tree, output")
 if a.output.exists() and any(a.output.iterdir()): ap.error("output directory must be absent or empty; stale evidence is rejected")
 try: return inspect(a.snapshot,a.source,a.server_tree,a.output,a.public_gguf)
 except Exception as error:
  a.output.mkdir(parents=True,exist_ok=True); (a.output/"manifest.json").write_text(json.dumps({"format":FORMAT,"status":"BLOCKED","evidence_stage":"INSPECTION_ONLY","inspection_status":"INSPECTION_ERROR","runtime_status":"PARTIAL_RUNTIME_FAIL_CLOSED","cpu_status":"UNSUPPORTED_FULL_TTS","metal_status":"BLOCKED_BY_CPU","parity_status":"NOT_RUN","publication":"NO_UPLOAD","upstream":{"repository":HF_REPOSITORY,"revision":HF_REVISION},"error":str(error),"blockers":[str(error)]},indent=2)+"\n"); return 2
if __name__=="__main__": raise SystemExit(main())
