#!/usr/bin/env -S uv run --frozen --project tools/parity --python 3.12 python
"""VAST-only authenticated inventory for Chatterbox base/Nano/Turbo.

This inspector intentionally does not convert, deserialize, or claim a native
runtime.  The three releases are composite pipelines; a single T3 file is not
a valid runtime artifact.
"""
from __future__ import annotations
import argparse, hashlib, io, json, os, re, subprocess, tarfile, tempfile, zipfile
from pathlib import Path, PurePosixPath
from typing import Any

HF={
 "base":("ResembleAI/chatterbox","5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18",13866212931),
 "nano":("ResembleAI/chatterbox-nano","71ccd1d0081b430592cea481f4307e764e07bc64",2998605148),
 "turbo":("ResembleAI/chatterbox-turbo","749d1c1a46eb10492095d68fbcf55691ccf137cd",4044184736),
}
SOURCE_URL="https://github.com/resemble-ai/chatterbox.git"; SOURCE_REV="5de7a54aa4e5e2baadb0182dde554908b48b85c2"
HEX40=re.compile(r"^[0-9a-f]{40}$"); HEX64=re.compile(r"^[0-9a-f]{64}$")
def rows(*items): return {p:(n,g,l) for p,n,g,l in items}
TREES={
"base":rows((".gitattributes",1519,"a6344aac8c09253b3b630fb776ae94478aa0275b",None),("Cangjie5_TC.json",1920163,"d77891f84ca1db0d6f7058a4ee081d4bb0bfe88e",None),("README.md",10081,"8e0091740326a09c226c43cc9294993c68cbd9d5",None),("conds.pt",107374,"e13b43d1ce809473454627428ff413ebfc7e8660","6552d70568833628ba019c6b03459e77fe71ca197d5c560cef9411bee9d87f4e"),("grapheme_mtl_merged_expanded_v1.json",69989,"d27fb3f2fd38ca39b7bbfbf83a13b3c617e551df",None),("mtl_tokenizer.json",68125,"54c8910a05341d2131a30256705f4fdb703f8ffa",None),("s3gen.pt",1057165844,"d4ef76740efef1aeb5eb1415214083288cf2cfde","9b9ff07e60b20c136e2b1b3d7563a24604e8d2c4c267888d1ee929dd0151d2a3"),("s3gen.safetensors",1056484620,"b752a028b2a1c2843b76e0df9582d8d81d10669d", "2b78103c654207393955e4900aac14a12de8ef25f4b09424f1ef91941f161d4e"),("s3gen_v3.pt",1056903694,"c622050c65325d203922c7aff5aee362f26141c5","f7abce4b196dae2d08d9296cbebc6521b046079577643b42a19a03499d08721e"),("s3gen_v3.safetensors",1056381804,"da4e499e980f32d2cf87615a8b71711dcca1cc68","4a46190f3dccc2230fbb3488a930bccc925862ee68f2662433dfcfe93ce6c2cb"),("t3_23lang.safetensors",2143154168,"b4c4f161254c45b959e6eb154b8352f3810dde5f","5b2194350053f9fc01544e614d923f1e5c8cd785ba596acc142d3d1d8614908c"),("t3_cfg.pt",1064892246,"bc5e765e1f29c03db68cf31eb09bad8f00c7d0bf","0b2dd5439fe7e94f379561419847a45bf2c79d0e8ea751c6bbe947ce337789cc"),("t3_cfg.safetensors",2129653744,"2dd9884f4acb611912740cf3d9c8b33711a694ce","914cb1696f47527fe8852ca8f1fe1fa63cb34f76f9c715e84e067b744dd0da81"),("t3_mtl23ls_v2.safetensors",2143989752,"482ccdb7195ef1a44cafee78c72020bdb89c7e86","b1237586127ce98e7800a68e49938eb5092846862aabcb6e17b2fda7889a6c75"),("t3_mtl23ls_v3.safetensors",2143989928,"1c209b87ad60fc1d4924b0ae2aee5c830c2a4a91","5abca8321ede76f8e61f1cc0d19aea6c946b28871017ce8726f8a69203f05953"),("tokenizer.json",25470,"abd07c710243ba89bf1b21780e7c37ddde92334e",None),("ve.pt",5698626,"adae22451d455ceb0592efc42464cadb21978b2a","4b16d836bc598509860f6fa068165a8bb5e9ac84f05582dfcf278a5a372879f1"),("ve.safetensors",5695784,"0713f1587e627f23d93121e154a7de490d549dfb","f0921cab452fa278bc25cd23ffd59d36f816d7dc5181dd1bef9751a7fb61f63c")),
"nano":rows((".gitattributes",1519,"a6344aac8c09253b3b630fb776ae94478aa0275b",None),("README.md",10773,"0445115f28b92ed511397bbccdcc9f3910139086",None),("added_tokens.json",418,"fd85325cdfafc690469b6f0e8aeb5cf4649c1450",None),("conds.pt",169454,"a7f5dfc71c3a2cb450ef4fbda5a0a52c8d6102fe","b1852099306fd6a7814eb9d0bd10186caba7249596cc23868f78a0eefbfa5033"),("merges.txt",456318,"226b0752cac7789c48f0cb3ec53eda48b7be36cc",None),("s3gen.safetensors",1056484620,"b752a028b2a1c2843b76e0df9582d8d81d10669d","2b78103c654207393955e4900aac14a12de8ef25f4b09424f1ef91941f161d4e"),("s3gen_meanflow.safetensors",1064875036,"f6cfa2a96f3ac1f6c65f371c42b856c16409fe12","d65cb687a2ed581ee6cc297e919ffefa63386944f42364ae13b78a594945514f"),("special_tokens_map.json",470,"b2d2dbc845800c78fe98656af927d831ab4ff7b7",None),("t3_nano_v1.safetensors",869899204,"0aa6bda24f6d331c100686e506d6a640fb5d5fdc","72b110185087d945dbdf54dee4e333848e1811bdd5fd6cb16ceb8da50006f0c9"),("t3_nano_v1.yaml",8488,"1a1c3c937c231ee4090fe8624d5e8c09a9cc14c1",None),("tokenizer_config.json",3878,"58bc759ea7dab3dd5442f0500e493f170eeba67f",None),("ve.safetensors",5695784,"0713f1587e627f23d93121e154a7de490d549dfb","f0921cab452fa278bc25cd23ffd59d36f816d7dc5181dd1bef9751a7fb61f63c"),("vocab.json",999186,"a15dd0028acd1dd2c1c2394c80e1de8f3f12a0e4",None)),
"turbo":rows((".gitattributes",1519,"a6344aac8c09253b3b630fb776ae94478aa0275b",None),("README.md",9544,"7d61412fa834e0e8d9217e86a8ad0bf44de04b6d",None),("added_tokens.json",418,"fd85325cdfafc690469b6f0e8aeb5cf4649c1450",None),("conds.pt",169454,"a7f5dfc71c3a2cb450ef4fbda5a0a52c8d6102fe","b1852099306fd6a7814eb9d0bd10186caba7249596cc23868f78a0eefbfa5033"),("merges.txt",456318,"226b0752cac7789c48f0cb3ec53eda48b7be36cc",None),("s3gen.safetensors",1056484620,"b752a028b2a1c2843b76e0df9582d8d81d10669d","2b78103c654207393955e4900aac14a12de8ef25f4b09424f1ef91941f161d4e"),("s3gen_meanflow.safetensors",1064875036,"f6cfa2a96f3ac1f6c65f371c42b856c16409fe12","d65cb687a2ed581ee6cc297e919ffefa63386944f42364ae13b78a594945514f"),("special_tokens_map.json",470,"b2d2dbc845800c78fe98656af927d831ab4ff7b7",None),("t3_turbo_v1.safetensors",1915480052,"d25802c72273ca6e1f776a85b7cd7b042e1c3247","fcf1f8c1d651bb7e3acd69ee5be269b4ac10c02980b7708213d598bc9f7cdf87"),("t3_turbo_v1.yaml",8457,"ff62b05c534a082aa5c5399c98986b8450205cc8",None),("tokenizer_config.json",3878,"58bc759ea7dab3dd5442f0500e493f170eeba67f",None),("ve.safetensors",5695784,"0713f1587e627f23d93121e154a7de490d549dfb","f0921cab452fa278bc25cd23ffd59d36f816d7dc5181dd1bef9751a7fb61f63c"),("vocab.json",999186,"a15dd0028acd1dd2c1c2394c80e1de8f3f12a0e4",None)),
}
def sha(p):
 h=hashlib.sha256();
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""): h.update(b)
 return h.hexdigest()
def blob(p):
 h=hashlib.sha1(); h.update(f"blob {p.stat().st_size}\0".encode())
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""): h.update(b)
 return h.hexdigest()
def lfs_pointer(size,digest):
 return f"version https://git-lfs.github.com/spec/v1\noid sha256:{digest}\nsize {size}\n".encode()
def safe(n):
 q=PurePosixPath(n)
 if not n or "\x00" in n or "\\" in n or q.is_absolute() or ".." in q.parts: raise ValueError(f"unsafe path {n!r}")
def local(root):
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
def validate_packet(packet,variant):
 repo,rev,total=HF[variant]
 if not isinstance(packet,dict) or set(packet)!={"repository","revision","resolved_revision","files"} or packet["repository"]!=repo or packet["revision"]!=rev or packet["resolved_revision"]!=rev: raise ValueError("tree identity mismatch")
 expected=TREES[variant]; remote={}
 for e in packet["files"]:
  if not isinstance(e,dict) or set(e)!={"path","type","size","git_blob_sha1","lfs_sha256"}: raise ValueError("tree schema")
  p,k,n,g,l=(e[x] for x in ("path","type","size","git_blob_sha1","lfs_sha256")); safe(p)
  if k!="file" or not isinstance(n,int) or isinstance(n,bool) or n<0 or not isinstance(g,str) or not HEX40.fullmatch(g) or (l is not None and (not isinstance(l,str) or not HEX64.fullmatch(l))) or p in remote: raise ValueError("tree entry identity")
  remote[p]=e
 if set(remote)!=set(expected): raise ValueError("complete server tree mismatch")
 if sum(e["size"] for e in remote.values())!=total: raise ValueError("fixed server total mismatch")
 for p,e in remote.items():
  n,g,l=expected[p]
  if (e["size"],e["git_blob_sha1"],e["lfs_sha256"])!=(n,g,l): raise ValueError(f"fixed identity mismatch {p}")
 return {"repository":repo,"revision":rev,"resolved_revision":rev,"files":sorted(remote.values(),key=lambda x:x["path"]),"total_bytes":sum(x["size"] for x in remote.values()),"identity":"Git blob SHA-1 plus LFS SHA-256"}
def tree(packet,root,variant,selected):
 remote=validate_packet(packet,variant); entries={e["path"]:e for e in remote["files"]}; fs=local(root)
 if set(fs)!=set(selected): raise ValueError("selected materialization mismatch")
 for p in selected:
  e=entries[p]; check_content_identity(e,fs[p])
 return {"server":remote,"selected":sorted((entries[p] for p in selected),key=lambda x:x["path"])}

def check_content_identity(entry,path):
 if path.stat().st_size!=entry["size"]: raise ValueError(f"content size mismatch {entry['path']}")
 digest=sha(path) if entry["lfs_sha256"] else blob(path)
 expected=entry["lfs_sha256"] or entry["git_blob_sha1"]
 if digest!=expected: raise ValueError(f"content identity mismatch {entry['path']}")
 if entry["lfs_sha256"]:
  pointer=lfs_pointer(entry["size"],entry["lfs_sha256"])
  pointer_blob=hashlib.sha1(f"blob {len(pointer)}\0".encode()+pointer).hexdigest()
  if pointer_blob!=entry["git_blob_sha1"]: raise ValueError(f"LFS pointer identity mismatch {entry['path']}")

def archive_inventory(path):
 if not path.is_file(): raise ValueError("checkpoint component missing")
 seen=set(); rows=[]
 try:
  if zipfile.is_zipfile(path):
   with zipfile.ZipFile(path) as z:
    for info in z.infolist():
     safe(info.filename)
     if info.filename in seen: raise ValueError("duplicate archive member")
     seen.add(info.filename)
     if info.flag_bits & 1: raise ValueError("encrypted archive member")
     mode=(info.external_attr>>16)&0o170000; directory=info.is_dir() or info.filename.endswith("/")
     if mode not in (0,0o100000,0o040000) or (directory and mode==0o100000): raise ValueError("unsafe archive member type")
     if info.file_size>2*1024*1024*1024: raise ValueError("archive member too large")
     rows.append({"name":info.filename,"directory":directory,"size":info.file_size})
  else:
   with tarfile.open(path,mode="r:*") as t:
    for info in t.getmembers():
     safe(info.name)
     if info.name in seen: raise ValueError("duplicate archive member")
     seen.add(info.name)
     if not (info.isdir() or info.isreg()): raise ValueError("unsafe archive member type")
     if info.size>2*1024*1024*1024: raise ValueError("archive member too large")
     rows.append({"name":info.name,"directory":info.isdir(),"size":info.size})
 except (OSError,ValueError,tarfile.TarError,zipfile.BadZipFile) as exc:
  raise ValueError(f"archive inventory failed: {exc}") from exc
 if len(rows)>4096: raise ValueError("too many archive members")
 return {"format":"zip" if zipfile.is_zipfile(path) else "tar","members":rows}

def safe_checkpoint_probe(path):
 """Probe a small conditioning checkpoint using weights-only PyTorch APIs."""
 try:
  import torch
 except ImportError as exc:
  raise ValueError("PyTorch is required for the weights-only checkpoint probe") from exc
 scanner=getattr(getattr(torch,"serialization",None),"get_unsafe_globals_in_checkpoint",None)
 if scanner is None: raise ValueError("PyTorch lacks get_unsafe_globals_in_checkpoint")
 unsafe=scanner(str(path))
 if unsafe: raise ValueError(f"unsafe checkpoint globals: {unsafe!r}")
 obj=torch.load(str(path),map_location="cpu",weights_only=True)
 if obj is None: raise ValueError("empty weights-only checkpoint")
 rows=[]; seen=set(); count=0
 def walk(value,name="",depth=0):
  nonlocal count
  count+=1
  if count>200000 or depth>64: raise ValueError("checkpoint walk bound exceeded")
  if isinstance(value,torch.Tensor):
   finite=bool(torch.isfinite(value).all().item()) if value.is_floating_point() else True
   if not finite: raise ValueError(f"non-finite tensor: {name}")
   rows.append({"name":name,"shape":[int(x) for x in value.shape],"dtype":str(value.dtype),"numel":int(value.numel()),"finite":finite})
   return
  if value is None or isinstance(value,(bool,int,float,str)): return
  identity=id(value)
  if identity in seen: raise ValueError(f"checkpoint cycle: {name}")
  seen.add(identity)
  if isinstance(value,dict):
   for key,child in value.items():
    if not isinstance(key,str) or not key or "\0" in key or "\\" in key or key.startswith("/") or ".." in PurePosixPath(key).parts: raise ValueError(f"unsafe checkpoint key: {key!r}")
    walk(child,f"{name}.{key}" if name else key,depth+1)
  elif isinstance(value,(list,tuple)):
   for index,child in enumerate(value): walk(child,f"{name}[{index}]",depth+1)
  else: raise ValueError(f"unsupported checkpoint object: {type(value).__name__}")
  seen.remove(identity)
 walk(obj)
 canonical=json.dumps(rows,separators=(",",":"),sort_keys=True).encode()
 return {"loader":"torch.load(weights_only=True,map_location=cpu)","unsafe_globals":[],"top_level_type":type(obj).__name__,"tensor_count":len(rows),"tensor_manifest":rows,"tensor_manifest_sha256":hashlib.sha256(canonical).hexdigest()}

MAX_HEADER=64*1024*1024
def st_header(path):
 size=path.stat().st_size
 with path.open("rb") as f:
  raw=f.read(8)
  if len(raw)!=8: raise ValueError("short safetensors")
  n=int.from_bytes(raw,"little")
  if n<=0 or n>MAX_HEADER or n+8>size: raise ValueError("header bound")
  head=json.loads(f.read(n).decode("utf-8"),object_pairs_hook=lambda pairs: _unique(pairs))
 if not isinstance(head,dict): raise ValueError("header root")
 meta=head.pop("__metadata__",{})
 if not isinstance(meta,dict) or any(not isinstance(k,str) or not isinstance(v,str) for k,v in meta.items()): raise ValueError("metadata map")
 data=size-8-n; ranges=[]; count=0; tensors={}
 for name,d in head.items():
  safe(name)
  if not isinstance(d,dict) or set(d)!={"dtype","shape","data_offsets"}: raise ValueError("descriptor schema")
  dtype,shape,off=d["dtype"],d["shape"],d["data_offsets"]
  if dtype not in {"F32","BF16","F16"} or not isinstance(shape,list) or not isinstance(off,list) or len(off)!=2 or any(isinstance(x,bool) or not isinstance(x,int) or x<0 for x in shape+off): raise ValueError("descriptor values")
  elems=1
  for dim in shape: elems*=dim
  width={"F32":4,"BF16":2,"F16":2}[dtype]
  if off[0]>off[1] or off[1]>data or off[1]-off[0]!=elems*width: raise ValueError("tensor bounds")
  ranges.append((off[0],off[1])); count+=1
  tensors[name]={"dtype":dtype,"shape":shape,"data_offsets":off}
  if count>200000: raise ValueError("too many tensors")
 cur=0
 for a,b in sorted(ranges):
  if a!=cur: raise ValueError("gap/overlap")
  cur=b
 if cur!=data: raise ValueError("trailing data")
 return {"header_bytes":n,"data_bytes":data,"tensor_count":count,"metadata":meta,"tensors":tensors,"tensor_names":sorted(tensors)}

def _unique(pairs):
 out={}
 for k,v in pairs:
  if k in out: raise ValueError(f"duplicate JSON key {k}")
  out[k]=v
 return out

def require_markers(text,markers,context):
 missing=[marker for marker in markers if marker not in text]
 if missing: raise ValueError(f"{context} source markers missing: {missing}")

def executable_source(root,variant):
 root=Path(root)
 files={p: (root/p).read_text(encoding="utf-8") for p in SOURCE_ROLES if (root/p).is_file()}
 if variant=="base":
  checks={"src/chatterbox/tts.py":("T3Config","Llama_520M"),"src/chatterbox/mtl_tts.py":("T3Config","multilingual"),"src/chatterbox/models/t3/llama_configs.py":("Llama_520M","hidden_size","intermediate_size","num_hidden_layers","1024","4096","30","16"),"src/chatterbox/models/t3/modules/t3_config.py":("speech_tokens_dict_size","max_text_tokens","max_speech_tokens","speech_cond_prompt_len","speaker_embed_size","use_perceiver_resampler","emotion_adv")}
 else:
  backbone="GPT2_small" if variant=="nano" else "GPT2_medium"
  checks={"src/chatterbox/tts_turbo.py":("T3Config","50276","6563","2048","4096","375","6561","6562",backbone),"src/chatterbox/models/t3/modules/t3_config.py":("speech_tokens_dict_size","max_text_tokens","max_speech_tokens","speech_cond_prompt_len","speaker_embed_size")}
 for name,markers in checks.items():
  if name not in files: raise ValueError(f"executable source role missing: {name}")
  require_markers(files[name],markers,name)
 return {name:{"bytes":len(text.encode()),"sha256":sha(root/name),"markers":list(markers)} for name,markers in checks.items()}

def checkpoint_topology(header,variant):
 names=header["tensor_names"]; low=[name.lower() for name in names]
 def find(*parts):
  hits=[(i,n) for i,n in enumerate(low) if all(part in n for part in parts)]
  if not hits: raise ValueError(f"checkpoint topology marker missing: {'+'.join(parts)}")
  return header["tensors"][names[hits[0][0]]],names[hits[0][0]]
 text,text_name=find("text","emb"); speech,speech_name=find("speech","emb")
 if variant=="base":
  if text["shape"][0]!=2454 or speech["shape"][0]!=8194: raise ValueError("base vocabulary topology mismatch")
 else:
  if text["shape"][0]!=50276 or speech["shape"][0]!=6563: raise ValueError("GPT-2 vocabulary topology mismatch")
 expected=EFFECTIVE[variant]
 layer_re=re.compile(r"(?:^|\.)(?:h|layers)\.(\d+)(?:\.|$)")
 layer_ids=sorted({int(match.group(1)) for name in names if (match:=layer_re.search(name))})
 if layer_ids != list(range(expected["layers"])):
  raise ValueError(f"{variant} transformer layer indices mismatch: {layer_ids[:4]}..{layer_ids[-4:] if layer_ids else []}")
 hidden=expected["hidden"]
 attention=[header["tensors"][name] for name in names if any(token in name.lower() for token in ("q_proj.weight", "c_attn.weight", "qkv.weight"))]
 if not attention: raise ValueError("checkpoint attention projection topology marker missing")
 if not any((row["shape"][-1] == hidden and row["shape"][0] in (hidden, 3*hidden)) for row in attention):
  raise ValueError(f"{variant} hidden attention shape mismatch")
 ffns=[header["tensors"][name] for name in names if any(token in name.lower() for token in ("gate_proj.weight", "c_fc.weight", "mlp.fc_in.weight"))]
 if not ffns or not any(expected.get("ffn", 4*hidden) in row["shape"] for row in ffns):
  raise ValueError(f"{variant} FFN shape mismatch")
 if variant=="base":
  qkv=[header["tensors"][name] for name in names if "q_proj.weight" in name.lower() or "k_proj.weight" in name.lower()]
  if not qkv or not all(row["shape"][-1] == hidden for row in qkv): raise ValueError("base Llama Q/K projection shape mismatch")
 else:
  positions=[header["tensors"][name] for name in names if any(token in name.lower() for token in ("wpe", "position_embedding", "pos_emb"))]
  if not positions or not any(row["shape"][0] == expected["positions"] and row["shape"][-1] == hidden for row in positions): raise ValueError(f"{variant} GPT-2 position embedding shape mismatch")
 if not any(("head" in n and ("text" in n or "speech" in n)) or "logits" in n for n in low):
  raise ValueError("checkpoint output-head topology marker missing")
 return {"family":expected["backbone"],"layers":expected["layers"],"hidden":hidden,"heads":expected["heads"],"kv_heads":expected.get("kv_heads"),"positions":expected.get("positions"),"text_embedding":{"name":text_name,"shape":text["shape"]},"speech_embedding":{"name":speech_name,"shape":speech["shape"]},"output_head_markers":[n for n in names if "head" in n.lower() or "logits" in n.lower()][:32]}

def tokenizer_evidence(fs,variant):
 out={}
 if variant=="base":
  for name in ("tokenizer.json","mtl_tokenizer.json","grapheme_mtl_merged_expanded_v1.json","Cangjie5_TC.json"):
   value=json.loads(fs[name].read_text(encoding="utf-8"),object_pairs_hook=_unique)
   if not isinstance(value,(dict,list)): raise ValueError(f"tokenizer JSON root invalid: {name}")
   if name=="tokenizer.json":
    model=value.get("model") if isinstance(value,dict) else None
    if not isinstance(model,dict) or model.get("type")!="BPE" or not isinstance(model.get("vocab"),dict) or not isinstance(model.get("merges"),list) or not model["merges"]: raise ValueError("base tokenizer BPE structure mismatch")
   if name=="mtl_tokenizer.json":
    model=value.get("model") if isinstance(value,dict) else None
    vocab=model.get("vocab") if isinstance(model,dict) else None
    if not isinstance(vocab,dict) or len(vocab)!=2454: raise ValueError("multilingual tokenizer vocabulary mismatch")
   out[name]={"sha256":sha(fs[name]),"root_type":type(value).__name__}
 else:
  vocab=json.loads(fs["vocab.json"].read_text(encoding="utf-8"),object_pairs_hook=_unique)
  added=json.loads(fs["added_tokens.json"].read_text(encoding="utf-8"),object_pairs_hook=_unique)
  special=json.loads(fs["special_tokens_map.json"].read_text(encoding="utf-8"),object_pairs_hook=_unique)
  tags=("[angry]","[fear]","[surprised]","[whispering]","[advertisement]","[dramatic]","[narration]","[crying]","[happy]","[sarcastic]","[clear throat]","[sigh]","[shush]","[cough]","[groan]","[sniff]","[gasp]","[chuckle]","[laugh]")
  if not isinstance(vocab,dict) or len(vocab)!=50257: raise ValueError("GPT-2 base vocabulary mismatch")
  if not isinstance(added,dict) or set(added)!=set(tags) or set(added.values())!=set(range(50257,50276)): raise ValueError("paralinguistic tag ids/content mismatch")
  if not isinstance(special,dict): raise ValueError("special-token map invalid")
  tokenizer=json.loads(fs["tokenizer_config.json"].read_text(encoding="utf-8"),object_pairs_hook=_unique)
  if not isinstance(tokenizer,dict) or any(tokenizer.get(key)!="<|endoftext|>" for key in ("bos_token","eos_token","pad_token","unk_token")): raise ValueError("GPT-2 tokenizer sentinel mismatch")
  merges=fs["merges.txt"].read_text(encoding="utf-8").splitlines()
  if not merges or merges[0]!="#version: 0.2" or len(merges[1:])!=50000 or len(set(merges[1:]))!=50000 or any(len(line.split())!=2 for line in merges[1:]): raise ValueError("GPT-2 merges structure mismatch")
  for name in ("vocab.json","merges.txt","added_tokens.json","special_tokens_map.json","tokenizer_config.json"):
   out[name]={"sha256":sha(fs[name]),"bytes":fs[name].stat().st_size}
  out["base_vocab_count"]=len(vocab); out["paralinguistic_tag_count"]=len(added); out["special_tokens"]=special
 return out

SELECTED={
 "base":("README.md","Cangjie5_TC.json","conds.pt","grapheme_mtl_merged_expanded_v1.json","mtl_tokenizer.json","s3gen_v3.safetensors","t3_mtl23ls_v3.safetensors","tokenizer.json","ve.safetensors"),
 "nano":("README.md","added_tokens.json","conds.pt","merges.txt","s3gen.safetensors","s3gen_meanflow.safetensors","special_tokens_map.json","t3_nano_v1.safetensors","t3_nano_v1.yaml","tokenizer_config.json","ve.safetensors","vocab.json"),
 "turbo":("README.md","added_tokens.json","conds.pt","merges.txt","s3gen.safetensors","s3gen_meanflow.safetensors","special_tokens_map.json","t3_turbo_v1.safetensors","t3_turbo_v1.yaml","tokenizer_config.json","ve.safetensors","vocab.json"),
}
SOURCE_ROLES=("LICENSE","README.md","pyproject.toml","src/chatterbox/tts.py","src/chatterbox/mtl_tts.py","src/chatterbox/tts_turbo.py","src/chatterbox/models/t3/t3.py","src/chatterbox/models/t3/llama_configs.py","src/chatterbox/models/t3/modules/t3_config.py","src/chatterbox/models/t3/modules/cond_enc.py","src/chatterbox/models/t3/inference/t3_hf_backend.py","src/chatterbox/models/s3tokenizer/s3tokenizer.py","src/chatterbox/models/tokenizers/__init__.py","src/chatterbox/models/tokenizers/tokenizer.py","src/chatterbox/models/s3gen/s3gen.py","src/chatterbox/models/s3gen/const.py","src/chatterbox/models/s3gen/configs.py","src/chatterbox/models/s3gen/decoder.py","src/chatterbox/models/s3gen/flow.py","src/chatterbox/models/s3gen/flow_matching.py","src/chatterbox/models/s3gen/f0_predictor.py","src/chatterbox/models/s3gen/hifigan.py","src/chatterbox/models/s3gen/xvector.py","src/chatterbox/models/s3gen/utils/mel.py","src/chatterbox/models/s3gen/transformer/upsample_encoder.py","src/chatterbox/models/voice_encoder/config.py","src/chatterbox/models/voice_encoder/melspec.py","src/chatterbox/models/voice_encoder/voice_encoder.py")
# These are Git blob identities from the fixed clean source checkout.  Marker
# checks alone are not provenance: the reference and inspector must reject a
# source file that merely happens to contain the same symbols.
SOURCE_ROLE_BLOBS={
 "LICENSE":"c1e82fe06523f528c975719352c223a82c8f28ce",
 "README.md":"b0086550bfae600e6e0c1e65b7387935c4a5bb5d",
 "src/chatterbox/tts.py":"4737f1823cf97b7ea98b1b06c1d5d8e3be62cacd",
 "src/chatterbox/models/t3/t3.py":"d83de261e249648f6654e2bac7cb10390af983c9",
 "src/chatterbox/models/t3/llama_configs.py":"eb38a7eb867bf6c0c01129722d22c62906607ca3",
 "src/chatterbox/models/t3/modules/t3_config.py":"000129ac389d0c4417060ce790e20417e7b3ac6a",
 "src/chatterbox/models/t3/modules/cond_enc.py":"b5f15c685783fbb048f6c0e86fc2ea8fbf1ec3de",
 "src/chatterbox/models/t3/inference/t3_hf_backend.py":"239374b21c41700ef7340ca8ca80353af09e4b2a",
 "src/chatterbox/models/s3tokenizer/s3tokenizer.py":"8648608ae4d8f28bfeec090b5fdb426b6b0ad336",
 "src/chatterbox/models/tokenizers/__init__.py":"fdf6d727a14bc20a0ce3a5dd41cf1ce44b6b330a",
 "src/chatterbox/models/tokenizers/tokenizer.py":"84d45d35d2db9c6c576a4af98a7ab91a704af9f2",
 "src/chatterbox/mtl_tts.py":"ec5ebff418f6abf283127c4de6bbe99580f29e69",
 "pyproject.toml":"381ed774eae577cb244d699f47b64953980ce72f",
 "src/chatterbox/tts_turbo.py":"e708f0c88abd6c615b64da725f344a1312098433",
 "src/chatterbox/models/s3gen/s3gen.py":"1207616c3fd4761d5f7bb71bc4bd7e24664e4366",
 "src/chatterbox/models/s3gen/const.py":"ef86d13e752a9571646c2a78f8fffa106ac1ce58",
 "src/chatterbox/models/s3gen/configs.py":"b09b2e52c2873095c81a0d1d7cb97130cefdc7f5",
 "src/chatterbox/models/s3gen/decoder.py":"ccd913a1488cde2a37343e36c72b2473f3d1a9c6",
 "src/chatterbox/models/s3gen/flow.py":"12f6715ecfc27f2114cdb66894ec78d448b97c75",
 "src/chatterbox/models/s3gen/flow_matching.py":"6a9635bb0c12ad19f4511632333a4b4856bff031",
 "src/chatterbox/models/s3gen/f0_predictor.py":"172c5f50bdece3d4ac2b3874b0a32deb9f957b93",
 "src/chatterbox/models/s3gen/hifigan.py":"33f9387e8018169d175fba777a9d70d89035348a",
 "src/chatterbox/models/s3gen/xvector.py":"6eb99af4aad25b33698211aa033d182d2f753379",
 "src/chatterbox/models/s3gen/utils/mel.py":"907d2b5770d2690b3e53a05e1952e3848b96ee41",
 "src/chatterbox/models/s3gen/transformer/upsample_encoder.py":"766a5e4e77070ff5579b1a567607c2879391bf8a",
 "src/chatterbox/models/voice_encoder/config.py":"8e9782a20eac8bc41afaf38d80a8af862adac232",
 "src/chatterbox/models/voice_encoder/melspec.py":"69147fc8c591c9364ff829a157af0ea3fcbd5770",
 "src/chatterbox/models/voice_encoder/voice_encoder.py":"d986f17fd6afab59364863b5e92fd56eec21236b",
}
EFFECTIVE={
 "base":{"kind":"multilingual-v3","text_vocab":2454,"speech_vocab":8194,"max_text":2048,"max_speech":4096,"speaker_dim":256,"backbone":"Llama_520M","layers":30,"hidden":1024,"heads":16,"kv_heads":16,"cond_prompt":150},
 "nano":{"kind":"nano","text_vocab":50276,"speech_vocab":6563,"max_text":2048,"max_speech":4096,"backbone":"GPT2_small","layers":12,"hidden":768,"heads":12,"positions":8196,"cond_prompt":375},
 "turbo":{"kind":"turbo","text_vocab":50276,"speech_vocab":6563,"max_text":2048,"max_speech":4096,"backbone":"GPT2_medium","layers":24,"hidden":1024,"heads":16,"positions":8196,"cond_prompt":375},
}
COMPONENTS=("feature_extractor","language_model","projection_model","scheduler","text_encoder","text_encoder_2","tokenizer","tokenizer_2","unet","vae","vocoder")
HISTORICAL={
 "base":{"repository":"vokra/chatterbox-multilingual-v3","revision":"95c8bf4409c237de930c2eec0274fb2b99a21a09","path":"chatterbox-multilingual-v3.gguf","bytes":2143980064,"git_blob_sha1":"4032155c333a43f47ea6efa78dcf46ede49860b1","lfs_sha256":"32733495d1379fc495e091f527139d2b0b5a0fbaf7ec8a53c03f0cebbf939d32"},
 "nano":{"repository":"vokra/chatterbox-nano-v1","revision":"49b2f3612ec3e479eb64ce49ab27ae82cbf0b206","path":"model.gguf","bytes":869895424,"git_blob_sha1":"45e1d339d7bf867dca4fb565900e735a5d27a159","lfs_sha256":"624bec40b1f590ecf3e336f1ffe0deb42b49089b24abdfc3e1944ff5154cc39d"},
 "turbo":{"repository":"vokra/chatterbox-turbo-v1","revision":"10fee774c6c5ed890e39cea76d0ae1a320f7a4eb","path":"model.gguf","bytes":1915470144,"git_blob_sha1":"19942b6472706335fce7198b3541320c0a7f9923","lfs_sha256":"ab1a266a42e41a9b4c2ab48fc60040abd9f1c320f807df154c08da986cd601b5"},
}

def source(root):
 root=Path(root); origin=subprocess.check_output(["git","-C",str(root),"remote","get-url","origin"],text=True).strip(); head=subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],text=True).strip(); dirty=subprocess.check_output(["git","-C",str(root),"status","--porcelain","--untracked-files=all"],text=True)
 if origin!=SOURCE_URL or head!=SOURCE_REV or dirty: raise ValueError("source identity/clean checkout mismatch")
 tracked=set(subprocess.check_output(["git","-C",str(root),"ls-files"],text=True).splitlines()); roles=[]
 for role in SOURCE_ROLES:
  p=root/role
  if role not in tracked or not p.is_file(): raise ValueError(f"missing source role {role}")
  roles.append({"path":role,"bytes":p.stat().st_size,"sha256":sha(p),"git_blob_sha1":blob(p),"expected_git_blob_sha1":SOURCE_ROLE_BLOBS.get(role)})
  if role in SOURCE_ROLE_BLOBS and blob(p)!=SOURCE_ROLE_BLOBS[role]:
   raise ValueError(f"fixed source role identity mismatch: {role}")
 license_rows=[]
 for role in ("LICENSE","README.md","pyproject.toml"):
  p=root/role
  if not p.is_file(): raise ValueError(f"missing license record {role}")
  text=p.read_text(encoding="utf-8")
  if role=="LICENSE" and "MIT License" not in text: raise ValueError("MIT license marker missing")
  license_rows.append({"path":role,"bytes":p.stat().st_size,"sha256":sha(p),"encoding":"utf-8"})
 return {"origin":origin,"revision":head,"roles":roles,"license_records":license_rows}

def manifest(variant):
 return {"format":"vokra-chatterbox-family-inspection-v1","variant":variant,"status":"BLOCKED","evidence_stage":"INSPECTION_ONLY","inspection_status":"PENDING","runtime_status":"NOT_IMPLEMENTED_FAIL_CLOSED","cpu_status":"UNSUPPORTED","metal_status":"BLOCKED_BY_CPU","parity_status":"NOT_RUN","publication":"NO_UPLOAD","components":list(COMPONENTS),"model":{"repository":HF[variant][0],"revision":HF[variant][1],"total_bytes":HF[variant][2],"files":TREES[variant]},"effective_inference_config":EFFECTIVE[variant],"historical_public_artifact":HISTORICAL[variant],"blockers":["full composite native binder is not implemented","tokenizer/T3 generation, VE, S3Gen/meanflow and conditioning binding require independent review","CPU parity and Metal complete graph are not run"]}

def inspect(a):
 variant=a.variant; m=manifest(variant)
 try:
  packet=json.loads(Path(a.server_tree).read_text(encoding="utf-8"),object_pairs_hook=_unique)
  m["server_tree"]=validate_packet(packet,variant)
  m["selected_materialization"]=tree(packet,Path(a.snapshot),variant,SELECTED[variant])
  fs=local(Path(a.snapshot))
  readme=fs["README.md"].read_text(encoding="utf-8").lower()
  if variant=="base":
   required=("license: mit", "multilingual", "t3", "s3gen", "voice")
  else:
   required=("license: mit", "nano" if variant=="nano" else "turbo", "s3gen", "voice")
  if any(x not in readme for x in required): raise ValueError("model card evidence marker missing")
  m["model_card"]={"path":"README.md","bytes":fs["README.md"].stat().st_size,"sha256":sha(fs["README.md"]),"required_markers":required,"weight_license":"MIT","license_status":"PRIMARY_CARD_DECLARATION_AUTHENTICATED"}
  m["license_evidence"]={"weights":"MIT","official_source":"MIT","dependencies":"SEPARATE_REVIEW_REQUIRED"}
  m["executable_source_evidence"]=executable_source(a.source,variant)
  m["tokenizer_evidence"]=tokenizer_evidence(fs,variant)
  m["semantic_components"]=[]
  for name in SELECTED[variant]:
   p=fs[name]
   row={"path":name,"bytes":p.stat().st_size,"sha256":sha(p)}
   if p.suffix==".safetensors": row["safetensors_header"]=st_header(p)
   if p.name.startswith("t3_") and p.suffix==".safetensors": row["checkpoint_topology"]=checkpoint_topology(row["safetensors_header"],variant)
   if p.suffix in {".pt",".pth"}:
    row["archive_inventory"]=archive_inventory(p)
    row["deserialization"]=safe_checkpoint_probe(p)
   m["semantic_components"].append(row)
  yaml=[n for n in SELECTED[variant] if n.endswith(".yaml")]
  if yaml:
   y=fs[yaml[0]].read_text(encoding="utf-8")
   m["training_yaml"]={"path":yaml[0],"bytes":len(y.encode()),"sha256":sha(fs[yaml[0]]),"status":"STALE_TRAINING_RECORD_NOT_RUNTIME_AUTHORITY","effective_override":EFFECTIVE[variant],"contains_legacy_markers":{x:(x in y) for x in ("402","604","250","30","Llama_520M")}}
  m["source"]=source(a.source) if a.source else (_ for _ in ()).throw(ValueError("source required"))
  m["inspection_status"]="AUTHENTICATED_EVIDENCE_COMPLETE"
 except Exception as e:
  m["inspection_status"]="INSPECTION_ERROR"; m["blockers"].append(f"inspection error: {type(e).__name__}: {e}")
 out=Path(a.evidence); out.mkdir(parents=True,exist_ok=True); (out/"manifest.json").write_text(json.dumps(m,indent=2,sort_keys=True)+"\n",encoding="utf-8")
 return 2

def self_test():
 base_manifest=manifest("base")
 parsed_manifest=json.loads(json.dumps(base_manifest),object_pairs_hook=_unique)
 assert parsed_manifest["runtime_status"]=="NOT_IMPLEMENTED_FAIL_CLOSED" and parsed_manifest["publication"]=="NO_UPLOAD"
 assert "scheduler" in parsed_manifest["components"] and len(parsed_manifest["components"])==11
 try: require_markers("T3Config GPT2 2048", ("50276",), "synthetic source")
 except ValueError: pass
 else: raise AssertionError("source drift accepted")
 for bad in ("../x","/x","a\\b","a\x00b",""):
  try:safe(bad)
  except ValueError:pass
  else: raise AssertionError("unsafe path accepted")
 with tempfile.TemporaryDirectory(prefix="chatterbox-inspect-") as d:
  root=Path(d)/"s"; root.mkdir(); (root/"x").write_text("x"); (root/".cache").mkdir(); (root/".cache"/"ignored").write_text("x")
  assert set(local(root))=={"x"}
  packet={"repository":HF["base"][0],"revision":HF["base"][1],"resolved_revision":HF["base"][1],"files":[{"path":"x","type":"file","size":1,"git_blob_sha1":"1"*40,"lfs_sha256":None}]}
  local_entry={"path":"x","size":1,"git_blob_sha1":blob(root/"x"),"lfs_sha256":None}
  check_content_identity(local_entry,root/"x")
  (root/"x").write_text("y")
  try:check_content_identity(local_entry,root/"x")
  except ValueError:pass
  else:raise AssertionError("same-size content mutation accepted")
  (root/"x").write_text("x")
  # A positive authenticated path uses a packet outside the materialized
  # snapshot and proves the Git blob is the canonical LFS pointer, while the
  # local payload is checked against the LFS SHA-256.
  positive_root=root/"positive"; positive_root.mkdir(); payload=positive_root/"payload"; payload.write_bytes(b"payload")
  payload_sha=sha(payload); pointer=lfs_pointer(payload.stat().st_size,payload_sha)
  pointer_path=root/"pointer"; pointer_path.write_bytes(pointer)
  original_tree,original_hf=TREES["base"],HF["base"]
  TREES["base"]={"payload":(payload.stat().st_size,blob(pointer_path),payload_sha)}; HF["base"]=(HF["base"][0],HF["base"][1],payload.stat().st_size)
  positive={"repository":HF["base"][0],"revision":HF["base"][1],"resolved_revision":HF["base"][1],"files":[{"path":"payload","type":"file","size":payload.stat().st_size,"git_blob_sha1":blob(pointer_path),"lfs_sha256":payload_sha}]}
  assert tree(positive,positive_root,"base",("payload",))["server"]["files"][0]["lfs_sha256"]==payload_sha
  TREES["base"],HF["base"]=original_tree,original_hf
  try:validate_packet(packet,"base")
  except ValueError:pass
  else:raise AssertionError("incomplete server tree accepted")
  for bad in (dict(packet,repository="attacker/model"),dict(packet,revision="0"*40),dict(packet,files=packet["files"]+[dict(packet["files"][0],path="extra")] )):
   try:validate_packet(bad,"base")
   except ValueError:pass
   else:raise AssertionError("invalid server identity accepted")
  good_archive=Path(d)/"good.pt"
  with tarfile.open(good_archive,"w") as t:
   directory=tarfile.TarInfo("folder/"); directory.type=tarfile.DIRTYPE; t.addfile(directory)
   member=tarfile.TarInfo("folder/state"); member.size=1; t.addfile(member,io.BytesIO(b"x"))
  assert archive_inventory(good_archive)["members"][0]["directory"]
  import torch
  safe_pt=Path(d)/"safe.pt"; torch.save({"x":torch.ones(1)},safe_pt)
  assert safe_checkpoint_probe(safe_pt)["unsafe_globals"]==[]
  bad_archive=Path(d)/"bad.pt"
  with tarfile.open(bad_archive,"w") as t:
   member=tarfile.TarInfo("../escape"); member.size=1; t.addfile(member,io.BytesIO(b"x"))
  try:archive_inventory(bad_archive)
  except ValueError:pass
  else:raise AssertionError("archive traversal accepted")
  h=json.dumps({"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}).encode(); p=root/"x.safetensors"; p.write_bytes(len(h).to_bytes(8,"little")+h+b"\0"*4); assert st_header(p)["tensor_count"]==1
  descriptors={}; offset=0
  for index in range(30):
   for name,shape,width in ((f"transformer.h.{index}.attn.q_proj.weight",[1024,1024],4),(f"transformer.h.{index}.mlp.gate_proj.weight",[4096,1],4)):
    size=shape[0]*shape[1]*width; descriptors[name]={"dtype":"F32","shape":shape,"data_offsets":[offset,offset+size]}; offset+=size
  for name,shape in (("text_emb.weight",[2454,1]),("speech_emb.weight",[8194,1]),("text_head.weight",[1,1])):
   size=shape[0]*shape[1]*4; descriptors[name]={"dtype":"F32","shape":shape,"data_offsets":[offset,offset+size]}; offset+=size
  raw=json.dumps(descriptors).encode(); topo_path=root/"topology.safetensors"; topo_path.write_bytes(len(raw).to_bytes(8,"little")+raw+b"\0"*offset); topo_header=st_header(topo_path); assert checkpoint_topology(topo_header,"base")["text_embedding"]["shape"]==[2454,1]
  try:checkpoint_topology(topo_header,"nano")
  except ValueError:pass
  else:raise AssertionError("checkpoint vocabulary drift accepted")
  token_root=root/"tokens"; token_root.mkdir()
  token_vocab={str(i):i for i in range(50257)}
  token_root.joinpath("vocab.json").write_text(json.dumps(token_vocab))
  tags=("[angry]","[fear]","[surprised]","[whispering]","[advertisement]","[dramatic]","[narration]","[crying]","[happy]","[sarcastic]","[clear throat]","[sigh]","[shush]","[cough]","[groan]","[sniff]","[gasp]","[chuckle]","[laugh]")
  token_root.joinpath("added_tokens.json").write_text(json.dumps({tag:50257+i for i,tag in enumerate(tags)}))
  token_root.joinpath("special_tokens_map.json").write_text(json.dumps({"eos_token":"<|endoftext|>"}))
  token_root.joinpath("tokenizer_config.json").write_text(json.dumps({"bos_token":"<|endoftext|>","eos_token":"<|endoftext|>","pad_token":"<|endoftext|>","unk_token":"<|endoftext|>"}))
  token_root.joinpath("merges.txt").write_text("#version: 0.2\n"+"\n".join(f"a{i} b{i}" for i in range(50000))+"\n")
  token_fs={name:token_root/name for name in ("vocab.json","added_tokens.json","special_tokens_map.json","tokenizer_config.json","merges.txt")}
  assert tokenizer_evidence(token_fs,"nano")["paralinguistic_tag_count"]==19
  token_root.joinpath("added_tokens.json").write_text(json.dumps({**{tag:50257+i for i,tag in enumerate(tags[:-1])},"[drift]":50275}))
  try:tokenizer_evidence(token_fs,"nano")
  except ValueError:pass
  else:raise AssertionError("tokenizer tag drift accepted")
  bad=root/"bad.safetensors"; bad.write_bytes(len(h).to_bytes(8,"little")+h+b"\0"*3)
  try:st_header(bad)
  except ValueError:pass
  else:raise AssertionError("trailing tensor data accepted")
  evidence=Path(d)/"e"; assert inspect(argparse.Namespace(variant="nano",snapshot=str(root/"missing"),server_tree=str(root/"missing.json"),source=None,evidence=str(evidence)))==2
  mm=json.loads((evidence/"manifest.json").read_text()); assert mm["status"]=="BLOCKED" and mm["inspection_status"]=="INSPECTION_ERROR" and mm["publication"]=="NO_UPLOAD"
 print("chatterbox inspector self-test PASS")

def main():
 p=argparse.ArgumentParser(); p.add_argument("--self-test",action="store_true"); p.add_argument("--variant",choices=tuple(HF)); p.add_argument("--snapshot"); p.add_argument("--server-tree"); p.add_argument("--source"); p.add_argument("--evidence",default="evidence"); a=p.parse_args()
 if a.self_test: self_test(); return 0
 if not all((a.variant,a.snapshot,a.server_tree,a.source)): p.error("variant, snapshot, server-tree, and source are required")
 return inspect(a)
if __name__=="__main__": raise SystemExit(main())
