#!/usr/bin/env -S uv run --frozen --project tools/parity --python 3.12 python
"""Inspection-only evidence collector for ESPnet OWSM v4 medium 1B."""
from __future__ import annotations
import argparse, hashlib, json, re, struct, subprocess, tempfile, zipfile
from pathlib import Path
from typing import Any

HF_REPOSITORY="espnet/owsm_v4_medium_1B"; HF_REVISION="e10985c8f1d592e905c24d2ac2b2c53e3feb24dc"
SOURCE_REPOSITORY="https://github.com/espnet/espnet.git"; SOURCE_REVISION="cccc29023d43a3f504e28df7d1324bb4eb6daedd"; SOURCE_TAG="v.202412"
FORMAT="vokra-owsm-v4-medium-1b-inspection-v1"
MAIN="exp/s2t_train_conv2d8_size1024_e18_d18_mel128_raw_bpe50000/valid.total_count.ave_5best.pth"; BPE="data/token_list/bpe_unigram50000/bpe.model"; STATS="exp/s2t_stats_raw_bpe50000/train/feats_stats.npz"; CONFIG="exp/s2t_train_conv2d8_size1024_e18_d18_mel128_raw_bpe50000/config.yaml"; README="README.md"
KNOWN_LFS={MAIN:(4_089_134_806,"b02d79f29a4daa31dd49ce145d9bb4cda0a1b68cdad91ae0af170ec3a4e92e09"),BPE:(1_044_580,"7ddb01f03dab493c18ab69391e98744c090f897890d8b529b30cae52a8d9eef4"),STATS:(1_786,"00c22dba27594df8f1d8f74a491b20c6e6e8c17e92159f81dfd634f98c098654"),CONFIG:(494_398,None),README:(5_458,None)}
EXPECTED_BLOBS={MAIN:"4b69fd6e811f46357c6e930868ae27412bed8d87",BPE:"e28ada30c05cbd3a8fc727fe9aefb03dcf2e29ed",STATS:"81dc0b816d8ccfe65c4606442e80552f2bec95ed",CONFIG:"fbf425c85d183f9103cb5e2c84ebffb0f425a930",README:"b44deaf39631cd6b0a55faeaa0c4bbe6e1a23f63"}
EXPECTED_README_SHA="0a6706b003418c3d64aabb153afdb08627c52be7add1cca0944b63b9e9849055"; EXPECTED_README_SIZE=5_458
EXPECTED_STATS_KEYS={"count","sum","sum_square"}; TOKEN_SHA="e19396ec012b0294a11fe85c35e36a1d903bc83e60ea602ddf6cc59b7c0e92f9"
IGNORE={".cache",".git"}

def sha256(path:Path)->str:
 d=hashlib.sha256()
 with path.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""): d.update(b)
 return d.hexdigest()
def git_blob_sha1(path:Path)->str:
 data=path.read_bytes(); return hashlib.sha1(f"blob {len(data)}\0".encode()+data).hexdigest()
def files(root:Path)->list[Path]:
 if not root.is_dir(): raise RuntimeError(f"missing root: {root}")
 out=[]; base=root.resolve()
 for path in sorted(root.rglob("*")):
  if any(x in IGNORE for x in path.relative_to(root).parts): continue
  if path.is_dir() and not path.is_symlink(): continue
  if path.is_symlink() and (not path.exists() or not path.is_file()): raise RuntimeError(f"dangling/nonregular symlink: {path}")
  if not path.is_file(): raise RuntimeError(f"nonregular member: {path}")
  resolved=path.resolve()
  if resolved!=base and base not in resolved.parents: raise RuntimeError(f"symlink escapes root: {path}")
  out.append(path)
 if not out: raise RuntimeError(f"empty root: {root}")
 return out
def identity(path:Path,root:Path)->dict[str,Any]: return {"path":path.relative_to(root).as_posix(),"bytes":path.stat().st_size,"sha256":sha256(path),"git_blob_sha1":git_blob_sha1(path)}
def no_dupes(pairs):
 out={}
 for key,value in pairs:
  if key in out: raise ValueError(f"duplicate key: {key}")
  out[key]=value
 return out
def server_tree(snapshot:Path,packet:Path,blockers:list[str])->dict[str,Any]:
 remote=json.loads(packet.read_text(),object_pairs_hook=no_dupes)
 all_rows=remote.get("files",[])
 # The worker preserves the complete remote tree in `files`, while only the
 # five authenticated inputs are materialised locally.  Match the selected
 # subset without throwing away the complete-tree evidence.
 rows=remote.get("materialized_files",all_rows)
 if not isinstance(all_rows,list) or not isinstance(rows,list):
  blockers.append("server tree files/materialized_files must be arrays")
  all_rows=[]; rows=[]
 records={}; complete_records={}
 def parse_rows(source:Any,target:dict[str,Any],label:str)->None:
  for row in source:
   if not isinstance(row,dict) or row.get("type")!="file": blockers.append(f"{label} is not recursive file-only"); continue
   name,size=row.get("path"),row.get("size"); lfs=row.get("lfs_sha256"); blob=row.get("git_blob_sha1")
   if not isinstance(name,str) or not isinstance(size,int) or isinstance(size,bool) or not name or ".." in Path(name).parts or "\\" in name or name.startswith("/"): blockers.append(f"unsafe server path: {name!r}"); continue
   if not re.fullmatch(r"[0-9a-f]{40}",str(blob)): blockers.append(f"invalid Git blob SHA1: {name}")
   if lfs is not None and not re.fullmatch(r"[0-9a-f]{64}",str(lfs)): blockers.append(f"invalid LFS SHA256: {name}")
   if name in target: blockers.append(f"duplicate {label} path: {name}")
   target[name]={"bytes":size,"git_blob_sha1":blob,"lfs_sha256":lfs}
 parse_rows(all_rows,complete_records,"complete server tree")
 parse_rows(rows,records,"materialized server tree")
 local={p.relative_to(snapshot).as_posix():identity(p,snapshot) for p in files(snapshot)}; missing=sorted(set(records)-set(local)); extra=sorted(set(local)-set(records)); changed=[]
 for name in sorted(set(records)&set(local)):
  expected,actual=records[name],local[name]
  if expected["lfs_sha256"] is not None:
   if expected["bytes"]!=actual["bytes"] or expected["lfs_sha256"]!=actual["sha256"]: changed.append(name)
  elif expected["bytes"]!=actual["bytes"] or expected["git_blob_sha1"]!=actual["git_blob_sha1"]: changed.append(name)
 identity_ok=remote.get("repository")==HF_REPOSITORY and remote.get("revision")==HF_REVISION and remote.get("resolved_revision")==HF_REVISION and remote.get("walk")=="recursive_file_only"
 if not identity_ok: blockers.append("server tree identity/walk mismatch")
 if missing or extra: blockers.append(f"server/local tree mismatch: {missing!r} {extra!r}")
 if changed: blockers.append(f"server/local content mismatch: {changed!r}")
 return {"status":"MATCHED" if identity_ok and not missing and not extra and not changed else "MISMATCH","repository":remote.get("repository"),"revision":remote.get("revision"),"resolved_revision":remote.get("resolved_revision"),"walk":remote.get("walk"),"files":records,"complete_files":complete_records,"materialized_scope":"selected_runtime_inputs" if "materialized_files" in remote else "complete_tree","missing":missing,"extra":extra,"content_mismatch":changed}
def yaml_value(path:Path,blockers:list[str])->Any:
 try:
  return yaml_value_text(path.read_text(),blockers)
 except Exception as e: blockers.append(f"config YAML blocked: {e}"); return None
def yaml_value_text(text:str,blockers:list[str])->Any:
 try:
  import yaml
  class Loader(yaml.SafeLoader): pass
  def mapping(loader,node,deep=False):
   pairs=loader.construct_pairs(node,deep=deep); out={}
   for key,value in pairs:
    if key in out: raise ValueError(f"duplicate YAML key: {key}")
    out[key]=value
   return out
  Loader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,mapping)
  return yaml.load(text,Loader=Loader)
 except Exception as e: blockers.append(f"YAML parse blocked: {e}"); return None
def model_card_frontmatter(text:str,blockers:list[str])->Any:
 if not text.startswith("---\n"): blockers.append("README model-card front matter missing"); return None
 end=text.find("\n---\n",4)
 if end<0: blockers.append("README model-card front matter unterminated"); return None
 return yaml_value_text(text[4:end]+"\n",blockers)
def validate_model_card(raw:Any,text:str,blockers:list[str])->dict[str,Any]:
 if not isinstance(raw,dict): return {"status":"BLOCKED_MODEL_CARD"}
 license_value=raw.get("license"); dataset_value=raw.get("datasets")
 if license_value!="cc-by-4.0": blockers.append("README license declaration mismatch")
 if dataset_value!=["espnet/yodas_owsmv4"]: blockers.append("README dataset declaration mismatch")
 required_markers=["language identification","recognition","translation","timestamp","long-form"]; lowered=text.lower(); missing=[marker for marker in required_markers if marker not in lowered]
 if missing: blockers.append(f"README task markers missing: {missing}")
 return {"status":"AUTHENTICATED_MODEL_CARD" if not missing and license_value=="cc-by-4.0" and dataset_value==["espnet/yodas_owsmv4"] else "BLOCKED_MODEL_CARD","license":license_value,"datasets":dataset_value,"task_markers":required_markers,"task_markers_missing":missing}
def readme_evidence(path:Path,root:Path,blockers:list[str])->dict[str,Any]:
 try:
  text=path.read_text(encoding="utf-8")
 except Exception as e:
  blockers.append(f"README UTF-8 blocked: {e}"); return {"status":"BLOCKED_README"}
 packet={**identity(path,root),"status":"BLOCKED_README"}
 identity_ok=packet["bytes"]==EXPECTED_README_SIZE and packet["sha256"]==EXPECTED_README_SHA and git_blob_sha1(path)==EXPECTED_BLOBS[README]
 if not identity_ok: blockers.append("README fixed identity mismatch")
 raw=model_card_frontmatter(text,blockers)
 card=validate_model_card(raw,text,blockers)
 packet.update({**card,"status":"AUTHENTICATED_MODEL_CARD" if identity_ok and card["status"]=="AUTHENTICATED_MODEL_CARD" else "BLOCKED_MODEL_CARD"})
 return packet
def json_packet(path:Path,root:Path,blockers:list[str])->dict[str,Any]:
 try:
  value=json.loads(path.read_text(),object_pairs_hook=no_dupes); return {**identity(path,root),"status":"PARSED_CANONICAL_JSON","top_level_keys":sorted(value) if isinstance(value,dict) else None}
 except Exception as e: blockers.append(f"duplicate/malformed JSON: {e}"); return {"path":path.relative_to(root).as_posix(),"status":"BLOCKED_JSON"}
def config_evidence(path:Path,root:Path,blockers:list[str])->dict[str,Any]:
 raw=yaml_value(path,blockers); packet={**identity(path,root),"status":"PARSED_STRICT_YAML" if raw is not None else "BLOCKED_YAML"}
 facts={
  "model":"espnet","model_conf.ctc_weight":0.3,"model_conf.lsm_weight":0.1,"model_conf.length_normalized_loss":False,"model_conf.sym_na":"<na>",
  "frontend":"default","frontend_conf.n_fft":512,"frontend_conf.win_length":400,"frontend_conf.hop_length":160,"frontend_conf.n_mels":128,"frontend_conf.fs":"16k",
  "specaug":"specaug","specaug_conf.apply_time_warp":False,"specaug_conf.time_warp_window":5,"specaug_conf.time_warp_mode":"bicubic","specaug_conf.apply_freq_mask":True,"specaug_conf.freq_mask_width_range":[0,27],"specaug_conf.num_freq_mask":2,"specaug_conf.apply_time_mask":True,"specaug_conf.time_mask_width_ratio_range":[0.0,0.05],"specaug_conf.num_time_mask":10,
  "normalize":"global_mvn","normalize_conf.stats_file":STATS,
  "encoder":"e_branchformer","encoder_conf.output_size":1024,"encoder_conf.attention_heads":16,"encoder_conf.attention_layer_type":"selfattn","encoder_conf.pos_enc_layer_type":"abs_pos","encoder_conf.rel_pos_type":"latest","encoder_conf.cgmlp_linear_units":4096,"encoder_conf.cgmlp_conv_kernel":31,"encoder_conf.use_linear_after_conv":False,"encoder_conf.gate_activation":"identity","encoder_conf.num_blocks":18,"encoder_conf.dropout_rate":0.1,"encoder_conf.positional_dropout_rate":0.1,"encoder_conf.attention_dropout_rate":0.0,"encoder_conf.input_layer":"conv2d8","encoder_conf.layer_drop_rate":0.0,"encoder_conf.linear_units":4096,"encoder_conf.positionwise_layer_type":"linear","encoder_conf.use_ffn":True,"encoder_conf.macaron_ffn":True,"encoder_conf.merge_conv_kernel":31,"encoder_conf.use_flash_attn":True,
  "decoder":"transformer","decoder_conf.attention_heads":16,"decoder_conf.linear_units":4096,"decoder_conf.num_blocks":18,"decoder_conf.dropout_rate":0.1,"decoder_conf.positional_dropout_rate":0.1,"decoder_conf.self_attention_dropout_rate":0.0,"decoder_conf.src_attention_dropout_rate":0.0,"decoder_conf.use_flash_attn":True,
  "preprocessor":"s2t","preprocessor_conf.text_prev_name":"text_prev","preprocessor_conf.text_ctc_name":"text_ctc","preprocessor_conf.fs":16000,"preprocessor_conf.na_symbol":"<na>","preprocessor_conf.speech_length":30,"preprocessor_conf.speech_resolution":0.02,"preprocessor_conf.speech_init_silence":30,"preprocessor_conf.text_prev_apply_prob":0.5,"preprocessor_conf.time_apply_prob":0.5,"preprocessor_conf.notime_symbol":"<notimestamps>","preprocessor_conf.first_time_symbol":"<0.00>","preprocessor_conf.last_time_symbol":"<30.00>",
  "token_type":"bpe","bpemodel":BPE,"version":"202412"
 }
 observed={}; missing=[]; mismatched=[]
 for dotted,expected in facts.items():
  cur=raw
  try:
   for part in dotted.split("."): cur=cur[part]
  except (KeyError,TypeError): missing.append(dotted); continue
  observed[dotted]=cur
  if cur!=expected: mismatched.append(dotted); blockers.append(f"config fact mismatch: {dotted}")
 token_info={"status":"BLOCKED_TOKEN_LIST"}
 tokens=raw.get("token_list") if isinstance(raw,dict) else None
 if not isinstance(tokens,list) or not all(isinstance(token,str) for token in tokens):
  blockers.append("config token_list must be a string array")
 else:
  canonical=json.dumps(tokens,ensure_ascii=False,separators=(",",":")).encode("utf-8"); digest=hashlib.sha256(canonical).hexdigest()
  token_mismatch=len(tokens)!=50_002 or digest!=TOKEN_SHA or tokens[:5]!=["<blank>","<unk>","<na>","<nospeech>","<abk>"] or tokens[-4:]!= ["巓","<sos>","<eos>","<sop>"]
  if token_mismatch: blockers.append("config token_list count/canonical SHA/boundary mismatch")
  token_info={"status":"BLOCKED_TOKEN_LIST" if token_mismatch else "PARSED_CANONICAL_TOKEN_LIST","count":len(tokens),"canonical_sha256":digest,"first":tokens[:8],"last":tokens[-8:]}
 if missing: blockers.append(f"config fact missing: {missing}")
 packet.update({"contract_status":"EXACT_FACTS_MATCHED" if raw is not None and not missing and not mismatched and token_info["status"]=="PARSED_CANONICAL_TOKEN_LIST" else "BLOCKED_FACTS","expected_facts":facts,"observed_facts":observed,"token_list":token_info}); return packet
def checkpoint_evidence(path:Path,root:Path,blockers:list[str])->dict[str,Any]:
 packet=identity(path,root); members=[]
 try:
  with zipfile.ZipFile(path) as archive:
   seen=set()
   if len(archive.infolist())>MAX_ARCHIVE_MEMBERS: blockers.append(f"checkpoint archive member bound exceeded: {MAX_ARCHIVE_MEMBERS}")
   total_uncompressed=0
   for info in archive.infolist():
    name=info.filename
    mode=info.external_attr>>16
    total_uncompressed+=info.file_size
    if len(name)>MAX_ARCHIVE_MEMBER_NAME or total_uncompressed>MAX_ARCHIVE_BYTES: blockers.append("checkpoint archive size/name bound exceeded")
    if name in seen or not name or name.startswith("/") or ".." in Path(name).parts or "\\" in name or info.is_dir() or mode not in (0,0o100644,0o100755) or info.flag_bits & 1: blockers.append(f"unsafe/duplicate/encrypted checkpoint member: {name!r}")
    seen.add(name); members.append({"name":name,"bytes":info.file_size,"mode":mode,"encrypted":bool(info.flag_bits & 1)})
 except Exception as e: blockers.append(f"checkpoint archive inventory blocked: {e}")
 try:
  import torch
  unsafe=getattr(torch.serialization,"get_unsafe_globals_in_checkpoint",lambda _: ["inventory unavailable"])(str(path))
  if unsafe: blockers.append(f"checkpoint unsafe globals: {unsafe}")
  state=torch.load(path,map_location="cpu",weights_only=True)
  if not isinstance(state,dict): blockers.append("safe checkpoint is not a state dict")
  tensors,metadata=inventory_loaded_checkpoint(state,torch,blockers)
  packet.update({"safe_load":"WEIGHTS_ONLY","tensor_count":len(tensors),"tensors":tensors,"metadata_count":len(metadata),"metadata":metadata,"resident_scope":"safe recursive state inventory only"})
 except Exception as e: blockers.append(f"weights_only checkpoint load blocked: {e}"); packet["safe_load"]="BLOCKED"
 packet["archive_members"]=members; return packet

MAX_CHECKPOINT_TENSORS=200_000; MAX_CHECKPOINT_METADATA=50_000; MAX_CHECKPOINT_ITEMS=300_000; MAX_CHECKPOINT_DEPTH=64
MAX_ARCHIVE_MEMBERS=100_000; MAX_ARCHIVE_MEMBER_NAME=4096; MAX_ARCHIVE_BYTES=16_000_000_000
def inventory_loaded_checkpoint(state:Any,torch:Any,blockers:list[str])->tuple[list[dict[str,Any]],list[dict[str,Any]]]:
 """Inventory a weights-only object without flattening or executing it."""
 tensors=[]; metadata=[]; active=set(); item_count=0; bound_hit=False; metadata_bound_hit=False
 def walk(value,path,depth=0):
  nonlocal item_count,bound_hit,metadata_bound_hit
  if bound_hit: return
  item_count+=1
  if item_count>MAX_CHECKPOINT_ITEMS:
   blockers.append(f"checkpoint item bound exceeded: {MAX_CHECKPOINT_ITEMS}"); bound_hit=True; return
  if depth>MAX_CHECKPOINT_DEPTH: blockers.append(f"checkpoint nesting bound exceeded: {path}"); return
  if isinstance(value,torch.Tensor):
   if len(tensors)>=MAX_CHECKPOINT_TENSORS: blockers.append(f"checkpoint tensor bound exceeded: {MAX_CHECKPOINT_TENSORS}"); bound_hit=True; return
   finite=bool(torch.isfinite(value).all().item()) if value.is_floating_point() else "NOT_APPLICABLE"
   if finite is False: blockers.append(f"non-finite checkpoint tensor: {path}")
   tensors.append({"name":path,"shape":list(value.shape),"dtype":str(value.dtype),"numel":value.numel(),"finite":finite}); return
  if value is None or isinstance(value,(bool,int,float,str)):
   if len(metadata)<MAX_CHECKPOINT_METADATA: metadata.append({"path":path,"type":type(value).__name__})
   elif not metadata_bound_hit: blockers.append(f"checkpoint metadata bound exceeded: {MAX_CHECKPOINT_METADATA}"); metadata_bound_hit=True
   return
  ident=id(value)
  if ident in active: blockers.append(f"checkpoint cycle: {path}"); return
  active.add(ident)
  if isinstance(value,dict):
   for key,item in value.items():
    if not isinstance(key,str) or not key or "\0" in key or "\\" in key or "/" in key or key.startswith("/") or ".." in Path(key).parts:
     blockers.append(f"unsafe checkpoint key: {path}"); continue
    walk(item,f"{path}.{key}" if path else key,depth+1)
  elif isinstance(value,(list,tuple)):
   for index,item in enumerate(value): walk(item,f"{path}[{index}]",depth+1)
  else: blockers.append(f"unsupported checkpoint object at {path}: {type(value).__name__}")
  active.remove(ident)
 walk(state,"")
 return tensors,metadata
def stats_evidence(path:Path,root:Path,blockers:list[str])->dict[str,Any]:
 packet=identity(path,root)
 try:
  import numpy as np
  with np.load(path,allow_pickle=False) as stats:
   if set(stats.files)!=EXPECTED_STATS_KEYS: blockers.append(f"stats keys mismatch: {stats.files}")
   if stats["count"].shape!=() or stats["count"].dtype!=np.int64 or int(stats["count"])!=224596934698: blockers.append("stats count mismatch")
   for key in ("sum","sum_square"):
    if stats[key].shape!=(128,) or stats[key].dtype!=np.float32: blockers.append(f"stats {key} shape/dtype mismatch")
   packet.update({"status":"NPZ_SAFE_PARSED","keys":stats.files,"count":int(stats["count"]),"sum_shape":list(stats["sum"].shape),"sum_square_shape":list(stats["sum_square"].shape)})
 except Exception as e: blockers.append(f"stats blocked: {e}"); packet["status"]="BLOCKED_NPZ"
 return packet
def bpe_evidence(path:Path,root:Path,blockers:list[str])->dict[str,Any]:
 packet=identity(path,root)
 try:
  import sentencepiece as sp
  model=sp.SentencePieceProcessor(model_file=str(path)); pieces=model.GetPieceSize(); first=[model.IdToPiece(i) for i in range(min(12,pieces))]
  if pieces!=50_000: blockers.append(f"BPE piece count mismatch: {pieces}")
  for i,want in enumerate(["<unk>","<s>","</s>"]):
   if model.PieceToId(want)!=i: blockers.append(f"BPE special ID mismatch: {want}")
  pad=model.pad_id()
  if pad!=-1: blockers.append(f"BPE pad ID mismatch: {pad}")
  expected_first=["<unk>","<s>","</s>","<na>","<nospeech>","<abk>","<afr>","<amh>","<ara>","<asm>","<ast>","<aze>"]
  if first[:12] != expected_first: blockers.append(f"BPE first pieces mismatch: {first!r}")
  packet.update({"status":"STRUCTURE_PARSED","piece_count":pieces,"first_pieces":first,"pad_id":pad})
 except Exception as e: blockers.append(f"BPE parse blocked: {e}"); packet["status"]="BLOCKED_STRUCTURE"
 return packet
def source_inventory(root:Path,repo:str,revision:str,roles:tuple[str,...],blockers:list[str])->dict[str,Any]:
 result={"repository":repo,"pinned_revision":revision}
 try:
  head=subprocess.run(["git","-C",str(root),"rev-parse","HEAD"],check=True,capture_output=True,text=True).stdout.strip(); origin=subprocess.run(["git","-C",str(root),"remote","get-url","origin"],check=True,capture_output=True,text=True).stdout.strip(); entries=subprocess.run(["git","-C",str(root),"ls-files","-s","-z"],check=True,capture_output=True).stdout.split(b"\0")
  tags=subprocess.run(["git","-C",str(root),"tag","--points-at",head],check=True,capture_output=True,text=True).stdout.splitlines()
  tracked=set(); paths=[]
  for raw in entries:
   if not raw: continue
   meta,rel=raw.split(b"\t",1); mode=meta.split()[0].decode(); relative=rel.decode(); tracked.add(relative); path=root/relative
   if mode not in ("100644","100755"): blockers.append(f"source nonregular/gitlink: {path}")
   elif path.is_file(): paths.append(path)
  clean=subprocess.run(["git","-C",str(root),"status","--porcelain","--untracked-files=all"],check=True,capture_output=True,text=True).stdout
  if head!=revision or origin!=repo: blockers.append(f"source identity mismatch: {repo}")
  if SOURCE_TAG not in tags: blockers.append(f"source tag missing at fixed revision: {SOURCE_TAG}")
  if clean: blockers.append(f"source checkout is dirty: {repo}")
  role_files=[]
  for role in roles:
   if role not in tracked or not (root/role).is_file(): blockers.append(f"source role is not tracked/present: {role}")
   else: role_files.append(identity(root/role,root))
  licenses=[p for p in sorted(root.glob("LICENSE*")) if p.is_file()]
  if not licenses: blockers.append(f"source license missing: {repo}")
  result.update({"resolved_revision":head,"origin":origin,"tags_at_revision":tags,"clean_status":"CLEAN" if not clean else "DIRTY","tracked_files":[identity(p,root) for p in sorted(paths)],"role_files":role_files,"license_files":[identity(p,root) for p in licenses]})
 except Exception as e: blockers.append(f"source inventory blocked: {e}")
 return result
def inspect(snapshot:Path,source:Path,tree:Path,out:Path)->int:
 blockers=[]; local=files(snapshot); tree_packet=server_tree(snapshot,tree,blockers)
 if not {CONFIG,MAIN,BPE,STATS,README}.issubset({p.relative_to(snapshot).as_posix() for p in local}): blockers.append("required OWSM files missing")
 for name,(size,digest) in KNOWN_LFS.items():
  path=snapshot/name
  if not path.is_file(): continue
  record=tree_packet.get("files",{}).get(name,{})
  got=identity(path,snapshot)
  if got["bytes"]!=size or record.get("bytes")!=size or record.get("git_blob_sha1")!=EXPECTED_BLOBS[name]: blockers.append(f"fixed Git identity mismatch: {name}")
  if name==README and got["sha256"]!=EXPECTED_README_SHA: blockers.append(f"fixed README SHA256 mismatch: {name}")
  if digest is not None and (got["sha256"]!=digest or record.get("lfs_sha256")!=digest): blockers.append(f"fixed LFS identity mismatch: {name}")
 config=config_evidence(snapshot/CONFIG,snapshot,blockers) if (snapshot/CONFIG).is_file() else None; stats=stats_evidence(snapshot/STATS,snapshot,blockers) if (snapshot/STATS).is_file() else None; bpe=bpe_evidence(snapshot/BPE,snapshot,blockers) if (snapshot/BPE).is_file() else None; checkpoint=checkpoint_evidence(snapshot/MAIN,snapshot,blockers) if (snapshot/MAIN).is_file() else None; readme=readme_evidence(snapshot/README,snapshot,blockers) if (snapshot/README).is_file() else None
 source=source_inventory(source,SOURCE_REPOSITORY,SOURCE_REVISION,("espnet2/s2t/espnet_model.py","espnet2/tasks/s2t.py","espnet2/asr/encoder/e_branchformer_encoder.py","espnet2/asr/decoder/transformer_decoder.py","espnet2/asr/frontend/default.py","espnet2/asr/specaug/specaug.py","espnet2/layers/global_mvn.py","espnet2/asr/ctc.py","espnet2/train/preprocessor.py"),blockers)
 inspection_status="AUTHENTICATED_EVIDENCE_COMPLETE" if not blockers else "INSPECTION_ERROR"
 blockers += ["native ESPnet S2T frontend/subsampling/encoder/decoder is not implemented","joint CTC/attention beam search and special-token semantics are not implemented","independent CPU numerical parity is not run","Metal is blocked by CPU runtime","dependency provenance is unreviewed","dataset provenance is unauthenticated"]
 payload={"format":FORMAT,"status":"BLOCKED","inspection_status":inspection_status,"evidence_stage":"INSPECTION_ONLY","runtime_status":"NOT_IMPLEMENTED_FAIL_CLOSED","cpu_status":"UNSUPPORTED","metal_status":"BLOCKED_BY_CPU","parity_status":"NOT_RUN","publication":"NO_UPLOAD","model":{"repository":HF_REPOSITORY,"revision":HF_REVISION,"server_tree":tree_packet,"files":[identity(p,snapshot) for p in local],"config":config,"checkpoint":checkpoint,"bpe":bpe,"readme":readme,"stats":stats},"official_source":source,"license_evidence":{"weights":{"status":"AUTHENTICATED_FROM_MODEL_CARD" if readme and readme.get("status")=="AUTHENTICATED_MODEL_CARD" else "BLOCKED_MODEL_CARD","spdx":"cc-by-4.0","card":"README.md"},"espnet_source":"Apache/MIT source declaration requires review","dependencies":"UNREVIEWED_BLOCKER","datasets":"UNAUTHENTICATED_BLOCKER"},"blockers":sorted(set(blockers))}
 out.mkdir(parents=True,exist_ok=True); (out/"manifest.json").write_text(json.dumps(payload,sort_keys=True,indent=2,default=list)+"\n"); return 2
def write_error_manifest(out:Path,error:Exception)->None:
 payload={"format":FORMAT,"status":"BLOCKED","inspection_status":"INSPECTION_ERROR","evidence_stage":"INSPECTION_ONLY","runtime_status":"NOT_IMPLEMENTED_FAIL_CLOSED","cpu_status":"UNSUPPORTED","metal_status":"BLOCKED_BY_CPU","parity_status":"NOT_RUN","publication":"NO_UPLOAD","error":str(error),"blockers":[str(error)]}
 out.mkdir(parents=True,exist_ok=True); (out/"manifest.json").write_text(json.dumps(payload,indent=2)+"\n")
def self_test()->None:
 assert len(HF_REVISION)==len(SOURCE_REVISION)==40 and MAIN.endswith(".pth") and BPE.endswith("bpe.model")
 assert CONFIG.endswith("config.yaml") and STATS.endswith("feats_stats.npz") and BPE.endswith("bpe.model") and README=="README.md"
 assert KNOWN_LFS[MAIN]==(4_089_134_806,"b02d79f29a4daa31dd49ce145d9bb4cda0a1b68cdad91ae0af170ec3a4e92e09")
 assert EXPECTED_BLOBS[CONFIG]=="fbf425c85d183f9103cb5e2c84ebffb0f425a930"
 with tempfile.TemporaryDirectory(prefix="owsm-inspect-") as d:
  root=Path(d); p=root/"x.safetensors"; h=json.dumps({"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}).encode(); p.write_bytes(struct.pack("<Q",len(h))+h+b"\0"*4)
  assert identity(p,root)["bytes"]==len(p.read_bytes())
  snapshot=root/"snapshot"; snapshot.mkdir(); isolated=snapshot/"x.safetensors"; isolated.write_bytes(p.read_bytes()); tree=root/"tree.json"; tree.write_text(json.dumps({"repository":HF_REPOSITORY,"revision":HF_REVISION,"resolved_revision":HF_REVISION,"walk":"recursive_file_only","files":[{"path":"x.safetensors","type":"file","size":isolated.stat().st_size,"git_blob_sha1":git_blob_sha1(isolated),"lfs_sha256":None}]})); bad=[]; assert server_tree(snapshot,tree,bad)["status"]=="MATCHED" and not bad
  isolated.write_bytes(isolated.read_bytes()[:-1]+b"x"); bad=[]; assert server_tree(snapshot,tree,bad)["status"]=="MISMATCH" and bad
  tree.write_text(json.dumps({"repository":"wrong/repository","revision":HF_REVISION,"resolved_revision":HF_REVISION,"walk":"recursive_file_only","files":[{"path":"x.safetensors","type":"file","size":isolated.stat().st_size,"git_blob_sha1":git_blob_sha1(isolated),"lfs_sha256":None}]})); bad=[]; assert server_tree(snapshot,tree,bad)["status"]=="MISMATCH" and any("identity" in x for x in bad)
  try:
   import torch
   nested={"encoder":[torch.tensor([1.0,2.0]),{"step":3}],"meta":"ok"}; bad=[]; tensors,metadata=inventory_loaded_checkpoint(nested,torch,bad); assert tensors[0]["name"]=="encoder[0]" and tensors[0]["finite"] and metadata
   bad=[]; tensors,_=inventory_loaded_checkpoint({"encoder":torch.tensor([float("nan")])},torch,bad); assert tensors[0]["finite"] is False and any("non-finite" in x for x in bad)
   cycle={}; cycle["self"]=cycle; bad=[]; inventory_loaded_checkpoint(cycle,torch,bad); assert any("cycle" in x for x in bad)
   bad=[]; inventory_loaded_checkpoint({"bad/key":object()},torch,bad); assert any("unsafe checkpoint key" in x for x in bad)
   bad=[]; inventory_loaded_checkpoint({"bad":object()},torch,bad); assert any("unsupported" in x for x in bad)
   deep={}; cursor=deep
   for index in range(MAX_CHECKPOINT_DEPTH+2): cursor["nested"]={}; cursor=cursor["nested"]
   bad=[]; inventory_loaded_checkpoint(deep,torch,bad); assert any("nesting bound" in x for x in bad)
   bounded={"x":list(range(MAX_CHECKPOINT_METADATA+1))}; bad=[]; inventory_loaded_checkpoint(bounded,torch,bad); assert any("metadata bound" in x for x in bad)
  except ImportError: pass
  unsafe_archive=root/"unsafe.pth"
  with zipfile.ZipFile(unsafe_archive,"w") as archive:
   member=zipfile.ZipInfo("../escape"); member.external_attr=(0o100644<<16); archive.writestr(member,b"x")
  bad=[]; evidence=checkpoint_evidence(unsafe_archive,root,bad); assert evidence["archive_members"] and any("unsafe" in x for x in bad)
  bounded_archive=root/"bounded.pth"
  with zipfile.ZipFile(bounded_archive,"w") as archive: archive.writestr(zipfile.ZipInfo("x"*(MAX_ARCHIVE_MEMBER_NAME+1)),b"x")
  bad=[]; checkpoint_evidence(bounded_archive,root,bad); assert any("size/name bound" in x for x in bad)
  duplicate=root/"duplicate.json"; duplicate.write_text('{"x":1,"x":2}'); bad=[]; json_packet(duplicate,root,bad); assert bad
  config=root/"config.yaml"; config.write_text("encoder_conf:\n  output_size: 1024\n"); bad=[]; packet=config_evidence(config,root,bad); assert packet["contract_status"]=="BLOCKED_FACTS" and bad
  config.write_text("model: wrong\n"); bad=[]; packet=config_evidence(config,root,bad); assert packet["contract_status"]=="BLOCKED_FACTS" and any("mismatch" in x for x in bad)
  config.write_text("model: espnet\ntoken_list: [1]\n"); bad=[]; packet=config_evidence(config,root,bad); assert packet["token_list"]["status"]=="BLOCKED_TOKEN_LIST" and any("string array" in x for x in bad)
  readme=root/README; readme.write_bytes(b"not a model card\xff"); bad=[]; packet=readme_evidence(readme,root,bad); assert packet["status"]=="BLOCKED_README" and bad
  card="---\nlicense: cc-by-4.0\ndatasets:\n- espnet/yodas_owsmv4\n---\nLanguage identification, recognition, translation, timestamp and long-form.\n"; bad=[]; parsed=model_card_frontmatter(card,bad); contract=validate_model_card(parsed,card,bad); assert parsed["datasets"]==["espnet/yodas_owsmv4"] and contract["status"]=="AUTHENTICATED_MODEL_CARD" and not bad
  bad=[]; string_card="---\nlicense: cc-by-4.0\ndatasets: espnet/yodas_owsmv4\n---\nLanguage identification, recognition, translation, timestamp and long-form.\n"; parsed=model_card_frontmatter(string_card,bad); contract=validate_model_card(parsed,string_card,bad); assert parsed["datasets"]=="espnet/yodas_owsmv4" and contract["status"]=="BLOCKED_MODEL_CARD" and any("README dataset declaration mismatch" in item for item in bad)
  bad_yaml=root/"bad.yaml"; bad_yaml.write_text("x: 1\nx: 2\n"); bad=[]; yaml_value(bad_yaml,bad); assert bad
  error_out=root/"error-evidence"; write_error_manifest(error_out,RuntimeError("self-test failure")); error=json.loads((error_out/"manifest.json").read_text()); assert error["inspection_status"]=="INSPECTION_ERROR" and error["publication"]=="NO_UPLOAD" and error["runtime_status"]=="NOT_IMPLEMENTED_FAIL_CLOSED"
 print("owsm_v4_medium_1b_inspect self-test: OK")
def main()->int:
 parser=argparse.ArgumentParser(); parser.add_argument("--self-test",action="store_true"); parser.add_argument("--snapshot",type=Path); parser.add_argument("--source",type=Path); parser.add_argument("--server-tree",type=Path); parser.add_argument("--output",type=Path); args=parser.parse_args()
 if args.self_test:
  if any(x is not None for x in (args.snapshot,args.source,args.server_tree,args.output)): parser.error("--self-test accepts no other arguments")
  self_test(); return 0
 if any(x is None for x in (args.snapshot,args.source,args.server_tree,args.output)): parser.error("normal run requires snapshot/source/server-tree/output")
 try: return inspect(args.snapshot,args.source,args.server_tree,args.output)
 except Exception as error:
  write_error_manifest(args.output,error); return 2
if __name__=="__main__": raise SystemExit(main())
