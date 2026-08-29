#!/usr/bin/env -S uv run --frozen --project tools/parity --python 3.12 python
"""Fail-closed evidence collector for kyutai/hibiki-2b-pytorch-bf16."""
from __future__ import annotations
import argparse, hashlib, json, re, struct, subprocess, tempfile
from pathlib import Path
from typing import Any

HF_REPOSITORY="kyutai/hibiki-2b-pytorch-bf16"; HF_REVISION="bd71144c96f26040612f6414716f5f48ee4fce69"
HIBIKI_REPOSITORY="https://github.com/kyutai-labs/hibiki.git"; HIBIKI_REVISION="f1cf9293e35c1dceffbe60dd325bdd702bc8305e"
MOSHI_REPOSITORY="https://github.com/kyutai-labs/moshi.git"; MOSHI_REVISION="e6a55d2722a65870ef52a6c9f6ecfc0e90f38362"
FORMAT="vokra-hibiki-2b-inspection-v1"
MAIN="hibiki-pytorch-ccef4858@200.safetensors"; MIMI="mimi-pytorch-e351c8d8@125.safetensors"; SPM="tokenizer_spm_48k_multi6_2.model"
ARTIFACTS={MAIN:(5_574_762_720,"0847a768f01f3e78c42ddb779e7aa9c610b7bab71306a71d62d98a7d9cff3bdb",327,"BF16"),MIMI:(384_644_900,"09b782f0629851a271227fb9d36db65c041790365f11bbe5d3d59369cf863f50",318,"F32"),SPM:(857_314,"c22110fb855aa049e17346ea2e88355bdd664f06cbfd09948380ab5e85b39697",48_000,"SPM")}
EXPECTED_GIT_BLOBS={MAIN:"a1f6cf83e90f4cfa83a294d468e5820c2a12ebc6",MIMI:"c8d5e4cd18a5c1ce05bb89d81144a46cf1b9076c",SPM:"e3d3dac8d55cf70915d8a4b1915becbdb89b828a"}
TREE_FILES={".gitattributes","README.md","config.json",MAIN,MIMI,SPM}
IGNORE={".cache",".git"}
MOSHI_ROLES=("moshi/moshi/models/lm.py","moshi/moshi/models/lm_utils.py","moshi/moshi/models/loaders.py","moshi/moshi/models/tts.py","moshi/moshi/conditioners/__init__.py","moshi/moshi/conditioners/base.py","moshi/moshi/conditioners/tensors.py","moshi/moshi/conditioners/text.py","moshi/moshi/run_tts.py","rust/moshi-core/src/lm.rs","rust/moshi-core/src/lm_generate.rs","rust/moshi-core/src/lm_generate_multistream.rs","rust/moshi-core/src/mimi.rs","rust/moshi-core/src/conditioner.rs","rust/moshi-core/src/tts.rs","rust/moshi-core/src/tts_streaming.rs")
HIBIKI_ROLES=("hibiki-rs/src/audio_io.rs","hibiki-rs/src/gen.rs","hibiki-rs/src/main.rs")
HIBIKI_ROLE_BLOBS={"hibiki-rs/src/audio_io.rs":"5625eafbb7b68e4c99f693ae812bac8f7212f070","hibiki-rs/src/gen.rs":"42df14d865f8de183a99d592bf97c6b31f6c13de","hibiki-rs/src/main.rs":"c34f6716ffaaa34590cc97825b716780832b48bb"}
MOSHI_ROLE_BLOBS={"moshi/moshi/conditioners/__init__.py":"8319f3d9c9d0f105a9c9350180870cf45dd192bb","moshi/moshi/conditioners/base.py":"75678b74e80a68e297d50ed93ffa93b2fe2822f5","moshi/moshi/conditioners/tensors.py":"dcc10edc83c6f853c7b6e24917edd322a94e9590","moshi/moshi/conditioners/text.py":"a1535b67bbb8ebbe5c28891829509ad5f57f5390","moshi/moshi/models/lm.py":"209b7a59c9c086810a81a7d8b99c5233a0ad87ff","moshi/moshi/models/lm_utils.py":"7397067c4d5eb6cb6eb06bdf0f94bdf5a6985444","moshi/moshi/models/loaders.py":"fd0e56a571e1a7f0de9d53827503a0669c1787f2","moshi/moshi/models/tts.py":"62a129dd4a1fbea4c17405c4f7c770d9a18e345d","moshi/moshi/run_tts.py":"3a210598bf30e197f2c0356bad802d50e2b7fcf3","rust/moshi-core/src/conditioner.rs":"42e1a382ee424df9f3978f834f87c89e6e74e8a7","rust/moshi-core/src/lm.rs":"6775ec8c5453c40a4f777303bd583db76e089dd8","rust/moshi-core/src/lm_generate.rs":"7fdcfd173a6bb8fe75f442a8969cc677ad78abba","rust/moshi-core/src/lm_generate_multistream.rs":"5e395c22d73bc3463cb6bddd957576a02cb6f48c","rust/moshi-core/src/mimi.rs":"603ecae8b89f9a718bdace7375b84984c2636847","rust/moshi-core/src/tts.rs":"9022ccc89a11b757400a20e0c5d0102c219464cf","rust/moshi-core/src/tts_streaming.rs":"66157784bbda4c3d500331e7d672f66783a77b54"}
HIBIKI_LICENSE_BLOBS={"LICENSE-APACHE":"261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64","LICENSE-MIT":"31aa79387f27e730e33d871925e152e35e428031","hibiki-rs/LICENSE":"261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64"}
MOSHI_LICENSE_BLOBS={"LICENSE-APACHE":"261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64","LICENSE-MIT":"31aa79387f27e730e33d871925e152e35e428031","client/LICENSE":"31aa79387f27e730e33d871925e152e35e428031","moshi/LICENSE":"31aa79387f27e730e33d871925e152e35e428031","moshi/LICENSE.audiocraft":"b93be90515ccd0b9daedaa589e42bf5929693f1f","moshi_mlx/LICENSE":"31aa79387f27e730e33d871925e152e35e428031","rust/LICENSE":"261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64"}
CONFIG_FACTS={("model_type",):"hibiki",("card",):2048,("n_q",):32,("dep_q",):16,("delays",):[0,0]+[2]*15+[0]+[2]*15,("dim",):2560,("text_card",):48000,("existing_text_padding_id",):3,("num_heads",):20,("num_layers",):24,("hidden_scale",):4.125,("causal",):True,("layer_scale",):None,("context",):1500,("max_period",):100000.0,("gating",):"silu",("norm",):"rms_norm_f32",("positional_embedding",):"rope",("depformer_dim",):1024,("depformer_num_heads",):16,("depformer_num_layers",):4,("depformer_dim_feedforward",):3072,("depformer_multi_linear",):True,("depformer_pos_emb",):"none",("depformer_weights_per_step",):True,("depformer_low_rank_embeddings",):128,("conditioners","description","type"):"lut",("conditioners","description","lut","n_bins"):31,("conditioners","description","lut","dim"):16,("conditioners","description","lut","tokenizer"):"noop",("conditioners","description","lut","possible_values"): ["very_bad","bad","neutral","good","very_good"],("fuser","cross_attention_pos_emb"):False,("fuser","cross_attention_pos_emb_scale"):1,("fuser","sum"): ["description"],("fuser","prepend"):[],("fuser","cross"):[],("cross_attention",):[],("model_id",):"ccef4858",("epoch",):200,("schedule",):[0,1,2,3,4,5,6,7,8,8,8,8,8,8,8,8],("lm_gen_config","temp"):0.8,("lm_gen_config","temp_text"):0.8,("lm_gen_config","top_k"):250,("lm_gen_config","top_k_text"):50}

def sha256(path:Path)->str:
 d=hashlib.sha256();
 with path.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""): d.update(b)
 return d.hexdigest()
def blob_sha1(path:Path)->str:
 h=hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
 with path.open("rb") as stream:
  for block in iter(lambda:stream.read(1<<20),b""): h.update(block)
 return h.hexdigest()
def files(root:Path)->list[Path]:
 if not root.is_dir(): raise RuntimeError(f"missing snapshot: {root}")
 out=[]
 for p in sorted(root.rglob("*")):
  relative=p.relative_to(root); parts=relative.parts
  if p.is_symlink(): raise RuntimeError(f"payload symlink is forbidden: {p}")
  if parts == (".cache",):
   if not p.is_dir(): raise RuntimeError(".cache transport parent must be a directory")
   continue
  if len(parts)>=2 and parts[:2] == (".cache","huggingface"):
   if parts == (".cache","huggingface") and not p.is_dir(): raise RuntimeError("HF transport cache must be a directory")
   continue
  if ".cache" in parts or ".git" in parts: raise RuntimeError(f"unauthenticated cache/metadata path: {p}")
  if p.is_dir(): continue
  if not p.is_file(): raise RuntimeError(f"nonregular snapshot member: {p}")
  out.append(p)
 if not out: raise RuntimeError("empty snapshot")
 return out
def identity(path:Path,root:Path)->dict[str,Any]: return {"path":path.relative_to(root).as_posix(),"bytes":path.stat().st_size,"sha256":sha256(path),"git_blob_sha1":blob_sha1(path)}
def no_dupes(pairs):
 d={}
 for k,v in pairs:
  if k in d: raise ValueError(f"duplicate JSON key: {k}")
  d[k]=v
 return d
def server_tree(snapshot:Path,packet:Path,blockers:list[str])->dict[str,Any]:
 remote=json.loads(packet.read_text(),object_pairs_hook=no_dupes)
 if not isinstance(remote,dict) or set(remote) != {"repository","requested_revision","resolved_revision","walk","files"}:
  blockers.append("server packet schema mismatch"); remote={}
 rows=remote.get("files",[]); records={}
 if not isinstance(rows,list): blockers.append("server tree files is not a list"); rows=[]
 for row in rows:
  if not isinstance(row,dict) or set(row) != {"path","type","size","git_blob_sha1","lfs_pointer_git_blob_sha1","lfs_payload_sha256","lfs_payload_size"} or row.get("type")!="file": blockers.append("server tree row schema/type mismatch"); continue
  name,size=row.get("path"),row.get("size"); git_id=row.get("git_blob_sha1"); pointer_id=row.get("lfs_pointer_git_blob_sha1"); lfs=row.get("lfs_payload_sha256"); payload_size=row.get("lfs_payload_size")
  if not isinstance(name,str) or not isinstance(size,int) or isinstance(size,bool) or not name or ".." in Path(name).parts or "\\" in name or name.startswith("/"): blockers.append(f"unsafe server member: {name!r}"); continue
  if name in records: blockers.append(f"duplicate server member: {name}"); continue
  if lfs is None:
   if not re.fullmatch(r"[0-9a-f]{40}",str(git_id)) or pointer_id is not None or payload_size is not None: blockers.append(f"invalid regular Git identity: {name}"); continue
  elif not re.fullmatch(r"[0-9a-f]{40}",str(pointer_id)) or git_id is not None or not re.fullmatch(r"[0-9a-f]{64}",str(lfs)) or not isinstance(payload_size,int) or isinstance(payload_size,bool) or payload_size != size: blockers.append(f"invalid LFS identity: {name}"); continue
  records[name]={"bytes":size,"git_blob_sha1":git_id,"lfs_pointer_git_blob_sha1":pointer_id,"lfs_payload_sha256":lfs,"lfs_payload_size":payload_size}
 local={p.relative_to(snapshot).as_posix():identity(p,snapshot) for p in files(snapshot)}; missing=sorted(set(records)-set(local)); extra=sorted(set(local)-set(records)); changed=[]
 for name in sorted(set(records)&set(local)):
  e,a=records[name],local[name]
  if e["lfs_payload_sha256"] is not None:
   pointer=f"version https://git-lfs.github.com/spec/v1\noid sha256:{e['lfs_payload_sha256']}\nsize {e['bytes']}\n".encode()
   pointer_git=hashlib.sha1(f"blob {len(pointer)}\0".encode()+pointer).hexdigest()
   if e["bytes"]!=a["bytes"] or e["lfs_payload_sha256"]!=a["sha256"] or pointer_git != e["lfs_pointer_git_blob_sha1"]: changed.append(name)
  elif e["bytes"]!=a["bytes"] or e["git_blob_sha1"]!=a["git_blob_sha1"]: changed.append(name)
 if remote.get("repository")!=HF_REPOSITORY or remote.get("requested_revision")!=HF_REVISION or remote.get("resolved_revision")!=HF_REVISION or remote.get("walk")!="recursive_file_only": blockers.append("server tree identity/walk mismatch")
 if missing or extra: blockers.append(f"server/local tree mismatch: {missing!r} {extra!r}")
 if changed: blockers.append(f"server/local content mismatch: {changed!r}")
 return {"status":"MATCHED" if not (missing or extra or changed or blockers) else "MISMATCH","repository":remote.get("repository"),"requested_revision":remote.get("requested_revision"),"resolved_revision":remote.get("resolved_revision"),"walk":remote.get("walk"),"files":records,"missing":missing,"extra":extra,"content_mismatch":changed}
def safe_header(path:Path,root:Path,expected_count:int,expected_dtype:str,blockers:list[str])->dict[str,Any]:
 item=identity(path,root); bad=False
 try:
  size=path.stat().st_size
  with path.open("rb") as f:
   raw=f.read(8)
   if len(raw)!=8: raise ValueError("short header")
   n=struct.unpack("<Q",raw)[0]
   if n==0 or n>64*1024*1024 or n>max(0,size-8): raise ValueError("unsafe header length")
   header=json.loads(f.read(n),object_pairs_hook=no_dupes)
 except Exception as e: blockers.append(f"header blocked {path}: {e}"); return {**item,"status":"BLOCKED_HEADER","error":str(e)}
 if not isinstance(header,dict): blockers.append(f"header is not object: {path}"); return {**item,"status":"BLOCKED_HEADER"}
 meta=header.pop("__metadata__",{})
 if not isinstance(meta,dict) or any(not isinstance(k,str) or not isinstance(v,str) for k,v in meta.items()): blockers.append(f"metadata is not string map: {path}"); bad=True
 start=8+n; ranges=[]; tensors=[]
 for name,spec in sorted(header.items()):
  try:
   if not isinstance(name,str) or not name or "\x00" in name or "\\" in name or name.startswith("/") or any(x in ("",".","..") for x in Path(name).parts): raise ValueError("unsafe tensor name")
   if not isinstance(spec,dict) or set(spec)!={"dtype","shape","data_offsets"}: raise ValueError("descriptor keys")
   shape,off,dtype=spec["shape"],spec["data_offsets"],spec["dtype"]
   if dtype!=expected_dtype or not isinstance(shape,list) or not isinstance(off,list) or len(off)!=2 or any(isinstance(x,bool) or not isinstance(x,int) or x<0 for x in shape+off): raise ValueError("dtype/shape/offset type")
   elems=1
   for x in shape: elems*=x
   if off[1]<off[0] or off[1]-off[0]!=elems*({"BF16":2,"F32":4}[dtype]) or start+off[1]>size: raise ValueError("byte range")
   ranges.append((off[0],off[1],name)); tensors.append({"name":name,"shape":shape,"dtype":dtype,"data_offsets":off,"finite":"NOT_CHECKED_HEADER_ONLY"})
  except Exception as e: blockers.append(f"invalid tensor {path}:{name}: {e}"); bad=True
 cur=0
 for a,b,name in sorted(ranges):
  if a<cur: blockers.append(f"overlap: {path}:{name}"); bad=True
  if a>cur: blockers.append(f"gap: {path}:{name}"); bad=True
  cur=max(cur,b)
 if cur!=size-start: blockers.append(f"data does not end at boundary: {path}"); bad=True
 if len(tensors)!=expected_count: blockers.append(f"tensor count mismatch {path}: {len(tensors)} != {expected_count}"); bad=True
 item.update({"status":"BLOCKED_HEADER" if bad else "HEADER_ONLY","header_bytes":n,"metadata":meta,"tensor_count":len(tensors),"tensors":tensors,"resident_scope":"header-only; tensor bodies never read"}); return item
def json_packet(path:Path,root:Path,blockers:list[str])->dict[str,Any]:
 try:
  value=json.loads(path.read_text(),object_pairs_hook=no_dupes)
  return {**identity(path,root),"status":"PARSED_CANONICAL_JSON","top_level_keys":sorted(value) if isinstance(value,dict) else None,"raw":value}
 except Exception as e: blockers.append(f"malformed/duplicate JSON {path}: {e}"); return {"path":path.relative_to(root).as_posix(),"status":"BLOCKED_JSON","error":str(e)}
def config_packet(path:Path,root:Path,blockers:list[str])->dict[str,Any]:
 packet=json_packet(path,root,blockers)
 try: value=json.loads(path.read_text(),object_pairs_hook=no_dupes)
 except Exception: return packet
 observed={}; missing=[]
 mismatched=[]
 for path,expected in CONFIG_FACTS.items():
  current=value
  try:
   for part in path: current=current[part]
  except (KeyError,TypeError): missing.append(".".join(path)); continue
  observed[".".join(path)]=current
  if current != expected: mismatched.append(".".join(path)); blockers.append(f"config fact mismatch: {'.'.join(path)}")
 if missing: blockers.append(f"config facts missing: {missing}")
 packet.update({"contract_status":"EXACT_FACTS_MATCHED" if not missing and not mismatched else "BLOCKED_FACTS","expected_facts":{".".join(k):v for k,v in CONFIG_FACTS.items()},"observed_facts":observed}); return packet
def spm_evidence(path:Path,root:Path,blockers:list[str])->dict[str,Any]:
 packet=identity(path,root)
 try:
  import sentencepiece as sp
  model=sp.SentencePieceProcessor(model_file=str(path)); count=model.GetPieceSize(); first=[model.IdToPiece(i) for i in range(5)]
  if count!=48_000: blockers.append(f"SentencePiece count mismatch: {count}")
  if first != ["<unk>","<s>","</s>","<pad>","<|im_start|>"]: blockers.append(f"SentencePiece first pieces mismatch: {first!r}")
  for i,want in enumerate(["<unk>","<s>","</s>","<pad>"]):
   if model.PieceToId(want)!=i: blockers.append(f"SentencePiece id mismatch: {want}")
  packet.update({"status":"STRUCTURE_PARSED","piece_count":count,"first_pieces":first,"special_ids":{w:model.PieceToId(w) for w in ["<unk>","<s>","</s>","<pad>"]}})
 except Exception as e: blockers.append(f"SentencePiece parse blocked: {e}"); packet.update({"status":"BLOCKED_STRUCTURE","error":str(e)})
 return packet
def source_inventory(root:Path,repo:str,revision:str,roles:tuple[str,...],blockers:list[str])->dict[str,Any]:
 result={"repository":repo,"pinned_revision":revision,"tracked_files":[]}
 try:
  head=subprocess.run(["git","-C",str(root),"rev-parse","HEAD"],check=True,capture_output=True,text=True).stdout.strip(); origin=subprocess.run(["git","-C",str(root),"remote","get-url","origin"],check=True,capture_output=True,text=True).stdout.strip().rstrip("/")
  if head!=revision or origin.removesuffix(".git")!=repo.removesuffix(".git"): blockers.append(f"source identity mismatch: {repo}")
  entries=subprocess.run(["git","-C",str(root),"ls-files","-s","-z"],check=True,capture_output=True).stdout.split(b"\0")
  paths=[]; tracked={}; tracked_rows=[]
  for raw in entries:
   if not raw: continue
   meta,rel=raw.split(b"\t",1); fields=meta.split(); mode=fields[0].decode(); index_object=fields[1].decode(); stage=fields[2].decode(); relative=rel.decode(); path=root/relative
   if relative in tracked: blockers.append(f"duplicate tracked path: {relative}")
   tracked[relative]=(mode,index_object,stage); paths.append(path)
   expected_mode={"100644":0o644,"100755":0o755}.get(mode)
   if expected_mode is None or stage!="0" or path.is_symlink() or not path.is_file() or (path.stat().st_mode&0o7777)!=expected_mode: blockers.append(f"source non-regular/stage/mode drift: {path}"); continue
   head_object=subprocess.run(["git","-C",str(root),"rev-parse",f"HEAD:{relative}"],check=True,capture_output=True,text=True).stdout.strip(); working=blob_sha1(path)
   if index_object!=head_object or index_object!=working: blockers.append(f"source object drift: {relative}")
   tracked_row=identity(path,root); tracked_row.update({"mode":mode,"stage":int(stage),"index_object":index_object,"head_object":head_object,"working_git_blob_sha1":working}); tracked_rows.append(tracked_row)
  licenses=[p for p in sorted(root.rglob("LICENSE*")) if p.is_file() and not p.is_symlink()]
  if not licenses: blockers.append(f"source license missing: {repo}")
  license_roles=(
   {"LICENSE-APACHE":("apache license, version 2.0","without warranties or conditions"),"LICENSE-MIT":("permission is hereby granted, free of charge","the software is provided \"as is\""),"hibiki-rs/LICENSE":("apache license, version 2.0","without warranties or conditions")}
   if repo==HIBIKI_REPOSITORY else
   {"LICENSE-APACHE":("apache license, version 2.0","without warranties or conditions"),"LICENSE-MIT":("permission is hereby granted, free of charge","the software is provided \"as is\""),"client/LICENSE":("permission is hereby granted, free of charge","the software is provided \"as is\""),"moshi/LICENSE":("permission is hereby granted, free of charge","the software is provided \"as is\""),"moshi/LICENSE.audiocraft":("creative commons","noncommercial"),"moshi_mlx/LICENSE":("permission is hereby granted, free of charge","the software is provided \"as is\""),"rust/LICENSE":("apache license, version 2.0","without warranties or conditions")})
  expected_license_blobs=HIBIKI_LICENSE_BLOBS if repo==HIBIKI_REPOSITORY else MOSHI_LICENSE_BLOBS
  if set(expected_license_blobs)!=set(license_roles): blockers.append(f"complete license Git object table unavailable: {repo}")
  for license_name, markers in license_roles.items():
   license_path=root/license_name
   if license_name not in tracked or not license_path.is_file() or license_path.is_symlink() or tracked[license_name][0]!="100644" or tracked[license_name][2]!="0": blockers.append(f"required license role missing/nonregular: {license_name}"); continue
   license_text=license_path.read_text(encoding="utf-8",errors="strict").lower()
   if any(marker not in license_text for marker in markers): blockers.append(f"license grant/warranty clauses missing: {license_name}")
   if license_name in expected_license_blobs and blob_sha1(license_path)!=expected_license_blobs[license_name]: blockers.append(f"license Git object drift: {license_name}")
  clean=subprocess.run(["git","-C",str(root),"status","--porcelain","--untracked-files=all"],check=True,capture_output=True,text=True).stdout
  if clean: blockers.append(f"source checkout is dirty: {repo}")
  expected_blobs=HIBIKI_ROLE_BLOBS if repo==HIBIKI_REPOSITORY else MOSHI_ROLE_BLOBS
  if set(expected_blobs)!=set(roles): blockers.append(f"complete fixed role Git table unavailable: {repo}")
  role_files=[]
  for role in roles:
   if role not in tracked or not (root/role).is_file() or (root/role).is_symlink(): blockers.append(f"source role is not tracked/present: {role}")
   else:
    if tracked[role][0]!="100644" or tracked[role][2]!="0" or (root/role).stat().st_mode&0o7777!=0o644: blockers.append(f"fixed source role mode/stage mismatch: {role}")
    head_object=subprocess.run(["git","-C",str(root),"rev-parse",f"HEAD:{role}"],check=True,capture_output=True,text=True).stdout.strip(); working=blob_sha1(root/role)
    row=identity(root/role,root); row.update({"mode":"0644","stage":0,"index_object":tracked[role][1],"head_object":head_object,"working_git_blob_sha1":working}); role_files.append(row)
    if role in expected_blobs and row["git_blob_sha1"]!=expected_blobs[role]: blockers.append(f"source role Git object drift: {role}")
  result.update({"resolved_revision":head,"origin":origin,"clean_status":"CLEAN" if not clean else "DIRTY","tracked_files":sorted(tracked_rows,key=lambda row:row["path"]),"role_files":role_files,"license_files":[identity(p,root) for p in licenses]})
  for r in roles:
   if not (root/r).is_file(): blockers.append(f"source role missing: {r}")
 except Exception as e: blockers.append(f"source inventory blocked: {e}")
 return result
def inspect(snapshot:Path,hibiki:Path,moshi:Path,tree:Path,out:Path)->int:
 blockers=[]; local=files(snapshot); tree_packet=server_tree(snapshot,tree,blockers)
 if set(p.relative_to(snapshot).as_posix() for p in local) != TREE_FILES: blockers.append("HF tree is not the exact six-file contract")
 for name,(size,digest,*_) in ARTIFACTS.items():
  path=snapshot/name
  if not path.is_file(): blockers.append(f"missing artifact: {name}"); continue
  got=identity(path,snapshot)
  server_record=tree_packet.get("files",{}).get(name,{})
  if got["bytes"]!=size or got["sha256"]!=digest or server_record.get("lfs_payload_sha256")!=digest or server_record.get("lfs_payload_size")!=size or server_record.get("lfs_pointer_git_blob_sha1")!=EXPECTED_GIT_BLOBS.get(name): blockers.append(f"artifact identity mismatch: {name}")
 main_packet=safe_header(snapshot/MAIN,snapshot,327,"BF16",blockers) if (snapshot/MAIN).is_file() else None
 mimi_packet=safe_header(snapshot/MIMI,snapshot,318,"F32",blockers) if (snapshot/MIMI).is_file() else None
 config=config_packet(snapshot/"config.json",snapshot,blockers) if (snapshot/"config.json").is_file() else None
 if config is None: blockers.append("config.json missing")
 spm=spm_evidence(snapshot/SPM,snapshot,blockers) if (snapshot/SPM).is_file() else None
 if spm is None: blockers.append("SentencePiece model missing")
 source=source_inventory(hibiki,HIBIKI_REPOSITORY,HIBIKI_REVISION,HIBIKI_ROLES,blockers)
 source["moshi"]=source_inventory(moshi,MOSHI_REPOSITORY,MOSHI_REVISION,MOSHI_ROLES,blockers)
 collection_complete=not blockers
 blockers += ["native FR↔EN streaming translation runtime is not implemented","upstream numerical parity is not run","dependency provenance is unreviewed","dataset provenance is unauthenticated"]
 payload={"format":FORMAT,"status":"BLOCKED","evidence_stage":"INSPECTION_ONLY","inspection_status":"AUTHENTICATED_EVIDENCE_COMPLETE" if collection_complete else "INSPECTION_ERROR","collection_status":"AUTHENTICATED" if collection_complete else "UNVERIFIED","runtime_status":"NOT_IMPLEMENTED_FAIL_CLOSED","cpu_status":"UNSUPPORTED","metal_status":"BLOCKED_BY_CPU","parity_status":"NOT_RUN","publication":"NO_UPLOAD","model":{"repository":HF_REPOSITORY,"requested_revision":HF_REVISION,"resolved_revision":HF_REVISION if collection_complete else None,"server_tree":tree_packet,"files":[identity(p,snapshot) for p in local],"main":main_packet,"mimi":mimi_packet,"sentencepiece":spm,"config":config,"expected_artifacts":ARTIFACTS},"official_source":source,"license_evidence":{"weights":"cc-by-4.0 requires review","hibiki_source":"MIT/Apache-2.0 requires review","moshi_source":"MIT/Apache-2.0 requires review","dependencies":"UNREVIEWED_BLOCKER","datasets":"UNAUTHENTICATED_BLOCKER"},"blockers":sorted(set(blockers))}
 out.mkdir(parents=True,exist_ok=True); (out/"manifest.json").write_text(json.dumps(payload,sort_keys=True,indent=2,default=list)+"\n"); return 2
def self_test()->None:
 assert MAIN=="hibiki-pytorch-ccef4858@200.safetensors" and MIMI=="mimi-pytorch-e351c8d8@125.safetensors" and SPM=="tokenizer_spm_48k_multi6_2.model"
 assert ARTIFACTS[SPM][1]=="c22110fb855aa049e17346ea2e88355bdd664f06cbfd09948380ab5e85b39697"
 assert not any(old in TREE_FILES for old in {"hibiki-pytorch-bf16.safetensors","tokenizer-e351c8d8-checkpoint125.safetensors","tokenizer_spm_32k_3.model"})
 assert "moshi/moshi/conditioners/base.py" in MOSHI_ROLES and "moshi/moshi/run_tts.py" in MOSHI_ROLES
 assert "moshi/conditioners/base.py" not in MOSHI_ROLES and "moshi/run_tts.py" not in MOSHI_ROLES
 assert HIBIKI_ROLE_BLOBS["hibiki-rs/src/audio_io.rs"]=="5625eafbb7b68e4c99f693ae812bac8f7212f070"
 assert HIBIKI_LICENSE_BLOBS["LICENSE-MIT"]=="31aa79387f27e730e33d871925e152e35e428031"
 assert MOSHI_ROLE_BLOBS["moshi/moshi/models/lm.py"]=="209b7a59c9c086810a81a7d8b99c5233a0ad87ff"
 assert MOSHI_LICENSE_BLOBS["moshi/LICENSE.audiocraft"]=="b93be90515ccd0b9daedaa589e42bf5929693f1f"
 with tempfile.TemporaryDirectory(prefix="hibiki-inspect-") as d:
  root=Path(d); p=root/"x.safetensors"; h=json.dumps({"x":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]}}).encode(); p.write_bytes(struct.pack("<Q",len(h))+h+b"\0\0"); bad=[]; assert safe_header(p,root,1,"BF16",bad)["status"]=="HEADER_ONLY" and not bad
  huge=root/"huge"; huge.write_bytes(struct.pack("<Q",65*1024*1024)+b"{}"); bad=[]; safe_header(huge,root,1,"BF16",bad); assert bad
  unsafe=root/"unsafe"; h=json.dumps({"../x":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]}}).encode(); unsafe.write_bytes(struct.pack("<Q",len(h))+h+b"\0\0"); bad=[]; safe_header(unsafe,root,1,"BF16",bad); assert any("unsafe tensor name" in x for x in bad)
  snapshot=root/"tree-snapshot"; snapshot.mkdir(); material=snapshot/"x"; material.write_bytes(b"abcd"); payload_sha=sha256(material); pointer=f"version https://git-lfs.github.com/spec/v1\noid sha256:{payload_sha}\nsize 4\n".encode(); pointer_git=hashlib.sha1(f"blob {len(pointer)}\0".encode()+pointer).hexdigest(); packet=root/"tree.json"; packet.write_text(json.dumps({"repository":HF_REPOSITORY,"requested_revision":HF_REVISION,"resolved_revision":HF_REVISION,"walk":"recursive_file_only","files":[{"path":"x","type":"file","size":4,"git_blob_sha1":None,"lfs_pointer_git_blob_sha1":pointer_git,"lfs_payload_sha256":payload_sha,"lfs_payload_size":4}]})); bad=[]; assert server_tree(snapshot,packet,bad)["status"]=="MATCHED" and not bad
  (snapshot/".cache"/"huggingface").mkdir(parents=True); (snapshot/".cache"/"huggingface"/"transport").write_bytes(b"cache"); assert len(files(snapshot))==1
  (snapshot/".cache"/"other").mkdir(parents=True); bad=[]
  try: files(snapshot)
  except RuntimeError: pass
  else: raise AssertionError("non-transport cache accepted")
  (snapshot/".cache"/"other").rmdir(); (snapshot/"link").symlink_to(material); bad=[]
  try: files(snapshot)
  except RuntimeError: pass
  else: raise AssertionError("payload symlink accepted")
  (snapshot/"link").unlink()
  material.write_bytes(b"abce"); bad=[]; assert server_tree(snapshot,packet,bad)["status"]=="MISMATCH" and any("content mismatch" in x for x in bad)
  non_lfs=root/"non-lfs"; non_lfs.mkdir(); plain=non_lfs/"x"; plain.write_bytes(b"abcd"); plain_packet=root/"non-lfs.json"; plain_packet.write_text(json.dumps({"repository":HF_REPOSITORY,"requested_revision":HF_REVISION,"resolved_revision":HF_REVISION,"walk":"recursive_file_only","files":[{"path":"x","type":"file","size":4,"git_blob_sha1":blob_sha1(plain),"lfs_pointer_git_blob_sha1":None,"lfs_payload_sha256":None,"lfs_payload_size":None}]})); bad=[]; assert server_tree(non_lfs,plain_packet,bad)["status"]=="MATCHED" and not bad
  spoof=json.loads(packet.read_text()); spoof["files"][0]["lfs_pointer_git_blob_sha1"]="0"*40; packet.write_text(json.dumps(spoof)); bad=[]; assert server_tree(snapshot,packet,bad)["status"]=="MISMATCH" and bad
  config_obj={}
  for path_parts,expected in CONFIG_FACTS.items():
   cursor=config_obj
   for part in path_parts[:-1]: cursor=cursor.setdefault(part,{})
   cursor[path_parts[-1]]=expected
  config_file=root/"config.json"; config_file.write_text(json.dumps(config_obj)); bad=[]; assert config_packet(config_file,root,bad)["contract_status"]=="EXACT_FACTS_MATCHED" and not bad
  config_file.write_text(json.dumps({"misnested":config_obj})); bad=[]; assert config_packet(config_file,root,bad)["contract_status"]=="BLOCKED_FACTS" and bad
 print("hibiki_2b_inspect self-test: OK")
def main()->int:
 parser=argparse.ArgumentParser(); parser.add_argument("--self-test",action="store_true"); parser.add_argument("--snapshot",type=Path); parser.add_argument("--hibiki-source",type=Path); parser.add_argument("--moshi-source",type=Path); parser.add_argument("--server-tree",type=Path); parser.add_argument("--output",type=Path); args=parser.parse_args()
 if args.self_test:
  if any(x is not None for x in (args.snapshot,args.hibiki_source,args.moshi_source,args.server_tree,args.output)): parser.error("--self-test accepts no other arguments")
  self_test(); return 0
 if any(x is None for x in (args.snapshot,args.hibiki_source,args.moshi_source,args.server_tree,args.output)): parser.error("normal run requires snapshot, both sources, server-tree, and output")
 try: return inspect(args.snapshot,args.hibiki_source,args.moshi_source,args.server_tree,args.output)
 except Exception as error:
  args.output.mkdir(parents=True,exist_ok=True); (args.output/"manifest.json").write_text(json.dumps({"format":FORMAT,"status":"BLOCKED","evidence_stage":"INSPECTION_ONLY","inspection_status":"INSPECTION_ERROR","collection_status":"UNVERIFIED","runtime_status":"NOT_IMPLEMENTED_FAIL_CLOSED","cpu_status":"UNSUPPORTED","metal_status":"BLOCKED_BY_CPU","parity_status":"NOT_RUN","publication":"NO_UPLOAD","model":{"repository":HF_REPOSITORY,"requested_revision":HF_REVISION,"resolved_revision":None},"error":str(error),"blockers":[str(error)]},indent=2)+"\n"); return 2
if __name__=="__main__": raise SystemExit(main())
