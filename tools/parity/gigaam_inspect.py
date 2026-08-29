#!/usr/bin/env -S uv run --frozen --project tools/parity --python 3.12 python
"""Fixed-revision, inspection-only evidence collector for GigaAM variants.

The v3 release is RNNT; the multilingual release is CTC.  This tool records
the primary artifacts and safe structural evidence but deliberately produces
no converted checkpoint and no runtime/parity result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Any

SOURCE_REPOSITORY = "https://github.com/salute-developers/GigaAM.git"
SOURCE_REVISION = "7447938d791c4f3e643386ee22c33777004293a5"
SOURCE_ROLES = (
    "gigaam/preprocess.py", "gigaam/encoder.py", "gigaam/decoder.py",
    "gigaam/decoding.py", "gigaam/model.py", "gigaam/timestamps_utils.py",
    "gigaam/types.py", "gigaam/utils.py",
)
MULTILINGUAL_VOCABULARY = [
    " ", "'", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k",
    "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x",
    "y", "z", "а", "б", "в", "г", "д", "е", "ж", "з", "и", "й", "к",
    "л", "м", "н", "о", "п", "р", "с", "т", "у", "ф", "х", "ц", "ч", "ш",
    "щ", "ъ", "ы", "ь", "э", "ю", "я", "ё", "і", "ғ", "қ", "ң", "ү", "ұ",
    "һ", "ә", "ө",
]
HISTORICAL_ARTIFACTS = {
    "v3": {
        "repository": "vokra/sber-gigaam-v3",
        "revision": "61f78f803eb436b4b86cfa71724eec7d57e6f9c2",
        "filename": "sber-gigaam-v3.gguf",
        "bytes": 448762880,
        "git_blob_sha1": "9a211147928ca7729cb2b4737000e18b702986b5",
        "lfs_sha256": "fa3d5ac4bb885292a8b8046cef6ea559564a3f1b1aa891316242b435b87590dd",
    },
    "multilingual": {
        "repository": "vokra/sber-gigaam-multilingual",
        "revision": "dc0a9f4e1150adc05418b2bb5eabbdf19099a5b4",
        "filename": "sber-gigaam-multilingual.gguf",
        "bytes": 883033824,
        "git_blob_sha1": "3d8c79798f3d6a48d11aaecfb01237e56b4b4dfb",
        "lfs_sha256": "74fd0d6dd84de2ce2ecaf0f8b8f13572bb3a81c44241f59814beb85389186b5a",
    },
}
VARIANTS: dict[str, dict[str, Any]] = {
    "v3": {"repository": "ai-sage/GigaAM-v3", "revision": "ec1dc1f01d0d627ab2c0d3acc1e235702300d95e", "model_class": "rnnt", "model_name": "v3_e2e_rnnt", "topology": "RNNT", "files": {
        ".gitattributes": (1519, "a6344aac8c09253b3b630fb776ae94478aa0275b"), "README.md": (2928, "a2a122b942e7707c3af2c0f5726705cc0f81fb77"), "config.json": (1867, "1a435114bb773fddfe73877cbc6da836ef2d6510"), "modeling_gigaam.py": (49135, "36221634c2c0dd0043eb26454b79e0e7aa061b25"), "pytorch_model.bin": (448928167, "ddb55c91b623df941d22d43b5369675921921f57", "afc6dcbae8320ea56f2cddebc0f13fbf62c9d59b6ddcad899782623c8610826a"), "tokenizer.model": (255336, "49739e92d267c28f71ac343cb5affe2f9cedf162", "828c12c991019eef952a960661f25a92d6ad279591e2ea466b4aeddf1d20a18a")}},
    "multilingual": {"repository": "ai-sage/GigaAM-Multilingual", "revision": "2f8a57144e6ec3adfd32fe0484d9ea9913305bc8", "model_class": "ctc", "model_name": "multilingual_ctc", "topology": "CTC", "files": {
        ".gitattributes": (1519, "a6344aac8c09253b3b630fb776ae94478aa0275b"), "README.md": (4454, "8b03cc3f9b67ff7ee3ebd5c07e5a0480793590f2"), "config.json": (2623, "056ee1175f04f4a202750b4d7bee431c6401dd4f"), "modeling_gigaam.py": (72778, "c50962bdb5c66d12b780b59719fc3c752a42e74f"), "pytorch_model.bin": (883170115, "2b0f1a4a05b27622fe5eb2732742d4f10bcf068b", "e1db43873ec5e296f229572e06e2470fc157ac9f8d4aacabda295630b9b91728")}},
}
FORMAT="vokra-gigaam-inspection-v1"; IGNORE={".cache",".git"}; MAX_ARCHIVE_MEMBERS=100_000; MAX_ARCHIVE_NAME=4096; MAX_ARCHIVE_BYTES=4_000_000_000_000; MAX_ITEMS=300_000; MAX_METADATA=50_000; MAX_DEPTH=64
V3_TOKENIZER_PIECE_COUNT=1024; V3_RNNT_NUM_CLASSES=1025
V3_TOKENIZER_FIRST_PIECES=("<unk>", ".", ",", "▁")
V3_TOKENIZER_LAST_PIECES=((1020, "₽"), (1021, "€"), (1022, "$"), (1023, "«"))

def sha256(path:Path)->str:
 d=hashlib.sha256()
 with path.open("rb") as stream:
  for chunk in iter(lambda:stream.read(1<<20),b""): d.update(chunk)
 return d.hexdigest()
def git_blob_sha1(path:Path)->str:
 data=path.read_bytes(); return hashlib.sha1(f"blob {len(data)}\0".encode()+data).hexdigest()
def files(root:Path)->list[Path]:
 if not root.is_dir(): raise RuntimeError(f"missing snapshot: {root}")
 base=root.resolve(); out=[]
 for path in sorted(root.rglob("*")):
  if any(part in IGNORE for part in path.relative_to(root).parts): continue
  if path.is_dir() and not path.is_symlink(): continue
  if path.is_symlink() and (not path.exists() or not path.is_file()): raise RuntimeError(f"dangling/nonregular symlink: {path}")
  if not path.is_file(): raise RuntimeError(f"nonregular snapshot entry: {path}")
  resolved=path.resolve()
  if resolved!=base and base not in resolved.parents: raise RuntimeError(f"symlink escapes snapshot: {path}")
  out.append(path)
 if not out: raise RuntimeError("empty snapshot")
 return out
def identity(path:Path,root:Path)->dict[str,Any]: return {"path":path.relative_to(root).as_posix(),"bytes":path.stat().st_size,"sha256":sha256(path),"git_blob_sha1":git_blob_sha1(path)}
def no_dupes(pairs):
 out={}
 for key,value in pairs:
  if key in out: raise ValueError(f"duplicate JSON key: {key}")
  out[key]=value
 return out
def server_tree(snapshot:Path,packet:Path,variant:dict[str,Any],blockers:list[str])->dict[str,Any]:
 remote=json.loads(packet.read_text(),object_pairs_hook=no_dupes); rows=remote.get("files")
 if not isinstance(rows,list): blockers.append("server tree files must be an array"); rows=[]
 records={}
 for row in rows:
  if not isinstance(row,dict) or row.get("type")!="file": blockers.append("server tree is not file-only"); continue
  name,size,blob,lfs=row.get("path"),row.get("size"),row.get("git_blob_sha1"),row.get("lfs_sha256")
  if not isinstance(name,str) or not name or name.startswith("/") or "\\" in name or ".." in Path(name).parts: blockers.append(f"unsafe server path: {name!r}"); continue
  if not isinstance(size,int) or isinstance(size,bool) or size<0: blockers.append(f"invalid server size: {name}"); continue
  if not re.fullmatch(r"[0-9a-f]{40}",str(blob)): blockers.append(f"invalid server Git SHA1: {name}")
  if lfs is not None and not re.fullmatch(r"[0-9a-f]{64}",str(lfs)): blockers.append(f"invalid server LFS SHA256: {name}")
  if name in records: blockers.append(f"duplicate server path: {name}")
  records[name]={"bytes":size,"git_blob_sha1":blob,"lfs_sha256":lfs}
 local={p.relative_to(snapshot).as_posix():identity(p,snapshot) for p in files(snapshot)}; missing=sorted(set(records)-set(local)); extra=sorted(set(local)-set(records)); changed=[]
 for name in sorted(set(records)&set(local)):
  expected,actual=records[name],local[name]
  if expected["bytes"]!=actual["bytes"] or (expected["lfs_sha256"] is not None and expected["lfs_sha256"]!=actual["sha256"]) or (expected["lfs_sha256"] is None and expected["git_blob_sha1"]!=actual["git_blob_sha1"]): changed.append(name)
 if missing or extra: blockers.append(f"server/local tree mismatch: {missing!r} {extra!r}")
 if changed: blockers.append(f"server/local content mismatch: {changed!r}")
 identity_ok=remote.get("repository")==variant["repository"] and remote.get("revision")==variant["revision"] and remote.get("resolved_revision")==variant["revision"] and remote.get("walk")=="recursive_file_only"
 if not identity_ok: blockers.append("server tree identity/walk mismatch")
 return {"status":"MATCHED" if identity_ok and not missing and not extra and not changed else "MISMATCH","repository":remote.get("repository"),"revision":remote.get("revision"),"resolved_revision":remote.get("resolved_revision"),"walk":remote.get("walk"),"files":records,"missing":missing,"extra":extra,"content_mismatch":changed}
def fixed_identity(path:Path,row:dict[str,Any]|None,expected:tuple[Any,...],root:Path)->bool:
 got=identity(path,root); size,blob,*lfs=expected
 if got["bytes"]!=size: return False
 if lfs:
  return bool(row) and row.get("git_blob_sha1")==blob and row.get("lfs_sha256")==lfs[0] and got["sha256"]==lfs[0]
 return got["git_blob_sha1"]==blob
def json_value(path:Path,blockers:list[str])->Any:
 try: return json.loads(path.read_text(encoding="utf-8"),object_pairs_hook=no_dupes)
 except Exception as e: blockers.append(f"JSON blocked: {e}"); return None
def at(raw:Any,path:str,expected:Any,observed:dict[str,Any],blockers:list[str])->None:
 cur=raw
 try:
  for part in path.split("."): cur=cur[part]
 except (KeyError,TypeError): blockers.append(f"config missing: {path}"); return
 observed[path]=cur
 if cur!=expected: blockers.append(f"config mismatch: {path}")
def config_expectations(variant:dict[str,Any])->dict[str,Any]:
 prefix="cfg.model.cfg"
 expected={
  "cfg.model._target_":"modeling_gigaam.GigaAMASR",
  f"{prefix}.model_class":variant["model_class"],f"{prefix}.sample_rate":16000,f"{prefix}.model_name":variant["model_name"],
  f"{prefix}.preprocessor._target_":"modeling_gigaam.FeatureExtractor",f"{prefix}.preprocessor.sample_rate":16000,
  f"{prefix}.preprocessor.features":64,f"{prefix}.preprocessor.win_length":320,f"{prefix}.preprocessor.hop_length":160,
  f"{prefix}.preprocessor.n_fft":320,f"{prefix}.preprocessor.center":False,
  f"{prefix}.encoder._target_":"modeling_gigaam.ConformerEncoder",f"{prefix}.encoder.feat_in":64,
  f"{prefix}.encoder.n_layers":16,f"{prefix}.encoder.d_model":768,f"{prefix}.encoder.subsampling":"conv1d",
  f"{prefix}.encoder.subs_kernel_size":5,f"{prefix}.encoder.subsampling_factor":4,f"{prefix}.encoder.ff_expansion_factor":4,
  f"{prefix}.encoder.self_attention_model":"rotary",f"{prefix}.encoder.pos_emb_max_len":5000,f"{prefix}.encoder.n_heads":16,
  f"{prefix}.encoder.conv_norm_type":"layer_norm",f"{prefix}.encoder.conv_kernel_size":5,f"{prefix}.encoder.flash_attn":False,
 }
 if variant["model_class"]=="rnnt": expected.update({
  f"{prefix}.preprocessor.mel_scale":"htk",f"{prefix}.preprocessor.mel_norm":None,
  f"{prefix}.head._target_":"modeling_gigaam.RNNTHead",f"{prefix}.head.decoder.pred_hidden":320,
  f"{prefix}.head.decoder.pred_rnn_layers":1,f"{prefix}.head.decoder.num_classes":V3_RNNT_NUM_CLASSES,
  f"{prefix}.head.joint.enc_hidden":768,f"{prefix}.head.joint.pred_hidden":320,
  f"{prefix}.head.joint.joint_hidden":320,f"{prefix}.head.joint.num_classes":V3_RNNT_NUM_CLASSES,
  f"{prefix}.decoding._target_":"modeling_gigaam.RNNTGreedyDecoding",f"{prefix}.decoding.vocabulary":None,
  f"{prefix}.decoding.model_path":"tokenizer.model",
 })
 else: expected.update({f"{prefix}.head._target_":"modeling_gigaam.CTCHead",f"{prefix}.head.feat_in":768,f"{prefix}.head.num_classes":71,f"{prefix}.decoding._target_":"modeling_gigaam.CTCGreedyDecoding",f"{prefix}.decoding.vocabulary":MULTILINGUAL_VOCABULARY})
 return expected
def config_evidence(path:Path,variant:dict[str,Any],blockers:list[str])->dict[str,Any]:
 raw=json_value(path,blockers); packet={"status":"BLOCKED_CONFIG","path":path.name,"sha256":sha256(path)}
 if not isinstance(raw,dict): return packet
 observed={}; expected=config_expectations(variant)
 for key,value in expected.items(): at(raw,key,value,observed,blockers)
 packet.update({"status":"EXACT_CONFIG" if not blockers else "BLOCKED_CONFIG","model_class":variant["model_class"],"model_name":variant["model_name"],"expected":expected,"observed":observed}); return packet

def validate_v3_tokenizer_structure(piece_count:int, first:list[str], last:list[tuple[int,str]], rnnt_num_classes:int)->dict[str,Any]:
 if type(piece_count) is not int or piece_count!=V3_TOKENIZER_PIECE_COUNT: raise ValueError("v3 tokenizer piece count mismatch")
 if tuple(first)!=V3_TOKENIZER_FIRST_PIECES: raise ValueError("v3 tokenizer leading pieces mismatch")
 if tuple(last)!=V3_TOKENIZER_LAST_PIECES: raise ValueError("v3 tokenizer trailing pieces mismatch")
 if type(rnnt_num_classes) is not int or rnnt_num_classes!=piece_count+1: raise ValueError("RNNT num_classes must equal tokenizer piece count + 1")
 return {"piece_count":piece_count,"first":list(first),"last":[{"id":index,"piece":piece} for index,piece in last],"rnnt_num_classes":rnnt_num_classes,"class_count_contract":"num_classes == piece_count + 1"}

def card_evidence(path:Path,blockers:list[str])->dict[str,Any]:
 try: text=path.read_text(encoding="utf-8")
 except Exception as e: blockers.append(f"README UTF-8 blocked: {e}"); return {"status":"BLOCKED_CARD"}
 if not text.startswith("---\n"): blockers.append("README front matter missing"); return {"status":"BLOCKED_CARD"}
 end=text.find("\n---\n",4)
 if end<0: blockers.append("README front matter unterminated"); return {"status":"BLOCKED_CARD"}
 raw=yaml_frontmatter(text[4:end]+"\n",blockers)
 license_value=raw.get("license") if isinstance(raw,dict) else None
 if license_value!="mit": blockers.append("README license is not MIT")
 return {"status":"AUTHENTICATED_CARD" if license_value=="mit" else "BLOCKED_CARD","license":license_value,"sha256":sha256(path)}
def json_value_text(text:str,blockers:list[str])->Any:
 try: return json.loads(text,object_pairs_hook=no_dupes)
 except Exception as e: blockers.append(f"front matter JSON/YAML parse blocked: {e}"); return None
def checkpoint_evidence(path:Path,blockers:list[str])->dict[str,Any]:
 packet={"path":path.name,"bytes":path.stat().st_size,"sha256":sha256(path),"archive_members":[]}
 try:
  with zipfile.ZipFile(path) as archive:
   infos=archive.infolist(); total=0; seen=set()
   if len(infos)>MAX_ARCHIVE_MEMBERS: blockers.append("checkpoint archive member bound exceeded")
   for info in infos:
    name=info.filename; total+=info.file_size; mode=info.external_attr>>16
    unsafe=not name or name.startswith("/") or "\\" in name or ".." in Path(name).parts or info.is_dir() or mode not in (0,0o100644,0o100755) or info.flag_bits&1
    if len(name)>MAX_ARCHIVE_NAME or total>MAX_ARCHIVE_BYTES or name in seen or unsafe: blockers.append(f"unsafe checkpoint archive member: {name!r}")
    seen.add(name); packet["archive_members"].append({"name":name,"bytes":info.file_size,"mode":mode,"encrypted":bool(info.flag_bits&1)})
 except Exception as e: blockers.append(f"checkpoint archive inventory blocked: {e}")
 try:
  import torch
  unsafe=getattr(torch.serialization,"get_unsafe_globals_in_checkpoint",lambda _: ["unavailable"])(str(path))
  if unsafe: blockers.append(f"checkpoint unsafe globals: {unsafe}")
  value=torch.load(path,map_location="cpu",weights_only=True); tensors=[]; metadata=[]; active=set(); count=0; metadata_bound=False
  def walk(item,name,depth=0):
   nonlocal count,metadata_bound
   count+=1
   if count>MAX_ITEMS: blockers.append("checkpoint item bound exceeded"); return
   if depth>MAX_DEPTH: blockers.append(f"checkpoint depth bound exceeded: {name}"); return
   if isinstance(item,torch.Tensor):
    finite=bool(torch.isfinite(item).all().item()) if item.is_floating_point() else "NOT_APPLICABLE"
    if finite is False: blockers.append(f"non-finite checkpoint tensor: {name}")
    tensors.append({"name":name,"shape":list(item.shape),"dtype":str(item.dtype),"numel":item.numel(),"finite":finite}); return
   if item is None or isinstance(item,(bool,int,float,str)):
    if len(metadata)<MAX_METADATA: metadata.append({"path":name,"type":type(item).__name__})
    elif not metadata_bound: blockers.append("checkpoint metadata bound exceeded"); metadata_bound=True
    return
   identity=id(item)
   if identity in active: blockers.append(f"checkpoint cycle: {name}"); return
   active.add(identity)
   if isinstance(item,dict):
    for key,child in item.items():
     if not isinstance(key,str) or not key or "\0" in key or "\\" in key or "/" in key or ".." in Path(key).parts: blockers.append(f"unsafe checkpoint key: {name}"); continue
     walk(child,f"{name}.{key}" if name else key,depth+1)
   elif isinstance(item,(list,tuple)):
    for index,child in enumerate(item): walk(child,f"{name}[{index}]",depth+1)
   else: blockers.append(f"unsupported checkpoint object: {type(item).__name__}")
   active.remove(identity)
  walk(value,""); packet.update({"safe_load":"WEIGHTS_ONLY","tensor_count":len(tensors),"tensors":tensors,"metadata_count":len(metadata)})
 except Exception as e: blockers.append(f"weights_only load blocked: {e}"); packet["safe_load"]="BLOCKED"
 return packet
def source_inventory(root:Path,blockers:list[str])->dict[str,Any]:
 result={"repository":SOURCE_REPOSITORY,"pinned_revision":SOURCE_REVISION}
 try:
  head=subprocess.run(["git","-C",str(root),"rev-parse","HEAD"],check=True,capture_output=True,text=True).stdout.strip(); origin=subprocess.run(["git","-C",str(root),"remote","get-url","origin"],check=True,capture_output=True,text=True).stdout.strip(); tracked=subprocess.run(["git","-C",str(root),"ls-files","-s"],check=True,capture_output=True,text=True).stdout.splitlines(); clean=subprocess.run(["git","-C",str(root),"status","--porcelain","--untracked-files=all"],check=True,capture_output=True,text=True).stdout
  if head!=SOURCE_REVISION or origin!=SOURCE_REPOSITORY: blockers.append("source identity mismatch")
  if clean: blockers.append("source checkout is dirty")
  tracked_records=[]
  for entry in tracked:
   meta,relative=entry.split("\t",1); mode=meta.split()[0]; tracked_path=root/relative
   if mode not in ("100644","100755"): blockers.append(f"source nonregular/gitlink: {relative}")
   elif tracked_path.is_file(): tracked_records.append(identity(tracked_path,root))
   else: blockers.append(f"source tracked file missing: {relative}")
  records=[]
  for role in SOURCE_ROLES:
   if not (root/role).is_file(): blockers.append(f"source role missing: {role}")
   else: records.append(identity(root/role,root))
  licenses=[identity(path,root) for path in sorted(root.glob("LICENSE*")) if path.is_file()]
  if not licenses: blockers.append("source LICENSE missing")
  license_texts=[]
  for path in sorted(root.glob("LICENSE*")):
   if path.is_file():
    try: license_texts.append(path.read_text(encoding="utf-8"))
    except Exception: blockers.append(f"source license is not UTF-8: {path.name}")
  source_license="MIT_DECLARATION_FOUND" if any("MIT License" in text for text in license_texts) else "UNKNOWN_BLOCKER"
  if source_license!="MIT_DECLARATION_FOUND": blockers.append("source license MIT declaration missing")
  result.update({"resolved_revision":head,"origin":origin,"tracked_count":len(tracked),"clean_status":"CLEAN" if not clean else "DIRTY","tracked_files":tracked_records,"role_files":records,"license_files":licenses,"license_status":source_license})
 except Exception as e: blockers.append(f"source inventory blocked: {e}")
 return result
def inspect(snapshot:Path,source:Path,tree:Path,out:Path,variant_name:str)->int:
 variant=VARIANTS[variant_name]; blockers=[]; local=files(snapshot); tree_packet=server_tree(snapshot,tree,variant,blockers); names={p.relative_to(snapshot).as_posix() for p in local}; required=set(variant["files"])
 if names!=required: blockers.append(f"snapshot file set mismatch: missing={sorted(required-names)!r} extra={sorted(names-required)!r}")
 fixed=[]
 for name,identity_expected in variant["files"].items():
  path=snapshot/name; row=tree_packet.get("files",{}).get(name)
  if not path.is_file(): continue
  got=identity(path,snapshot); size,blob,*lfs=identity_expected; lfs_value=lfs[0] if lfs else None
  if not fixed_identity(path,row,identity_expected,snapshot): blockers.append(f"fixed artifact identity mismatch: {name}")
  fixed.append({"expected_bytes":size,"expected_git_blob_sha1":blob,"expected_lfs_sha256":lfs_value,**got})
 config=config_evidence(snapshot/"config.json",variant,blockers) if (snapshot/"config.json").is_file() else None; card=card_evidence(snapshot/"README.md",blockers) if (snapshot/"README.md").is_file() else None; checkpoint=checkpoint_evidence(snapshot/"pytorch_model.bin",blockers) if (snapshot/"pytorch_model.bin").is_file() else None; tokenizer=None
 if variant_name=="v3" and (snapshot/"tokenizer.model").is_file():
  try:
   import sentencepiece as sp
   model=sp.SentencePieceProcessor(model_file=str(snapshot/"tokenizer.model"))
   structure=validate_v3_tokenizer_structure(model.GetPieceSize(),[model.IdToPiece(index) for index in range(4)],[(index,model.IdToPiece(index)) for index,_ in V3_TOKENIZER_LAST_PIECES],V3_RNNT_NUM_CLASSES)
   tokenizer={"status":"STRUCTURE_PARSED",**structure}
  except Exception as e: blockers.append(f"v3 tokenizer blocked: {e}")
 elif variant_name=="v3": blockers.append("v3 tokenizer missing")
 source_evidence=source_inventory(source,blockers)
 inspection_status="AUTHENTICATED_EVIDENCE_COMPLETE" if not blockers else "INSPECTION_ERROR"
 historical=HISTORICAL_ARTIFACTS[variant_name]
 blockers += ["native GigaAM forward is not implemented","CPU numerical parity is not run","Metal parity is not run",f"historical public GGUF is stale/incompatible: {historical['repository']}@{historical['revision']}","dataset provenance is not authenticated"]
 payload={"format":FORMAT,"status":"BLOCKED","inspection_status":inspection_status,"evidence_stage":"INSPECTION_ONLY","runtime_status":"NOT_IMPLEMENTED_FAIL_CLOSED","cpu_status":"UNSUPPORTED","metal_status":"BLOCKED_BY_CPU","parity_status":"NOT_RUN","publication":"NO_UPLOAD","variant":variant_name,"model":{"repository":variant["repository"],"revision":variant["revision"],"topology":variant["topology"],"files":fixed,"server_tree":tree_packet,"config":config,"card":card,"checkpoint":checkpoint,"tokenizer":tokenizer},"historical_public_artifact":historical,"official_source":source_evidence,"license_evidence":{"weights": "MIT from authenticated model card" if card and card.get("status")=="AUTHENTICATED_CARD" else "BLOCKED_MODEL_CARD","source":"MIT requires separate source license audit","datasets":"UNAUTHENTICATED_BLOCKER"},"blockers":sorted(set(blockers))}
 out.mkdir(parents=True,exist_ok=True); (out/"manifest.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n"); return 2
def self_test()->None:
 assert set(VARIANTS)=={"v3","multilingual"} and VARIANTS["v3"]["model_class"]!="ctc" and VARIANTS["multilingual"]["model_class"]!="rnnt"
 assert V3_TOKENIZER_PIECE_COUNT==1024 and V3_RNNT_NUM_CLASSES==V3_TOKENIZER_PIECE_COUNT+1
 assert V3_TOKENIZER_FIRST_PIECES==("<unk>", ".", ",", "▁") and V3_TOKENIZER_LAST_PIECES==((1020, "₽"), (1021, "€"), (1022, "$"), (1023, "«"))
 assert validate_v3_tokenizer_structure(1024,list(V3_TOKENIZER_FIRST_PIECES),list(V3_TOKENIZER_LAST_PIECES),1025)["class_count_contract"]=="num_classes == piece_count + 1"
 for piece_count,first,last,num_classes in ((1025,list(V3_TOKENIZER_FIRST_PIECES),list(V3_TOKENIZER_LAST_PIECES),1025),(1024,["<unk>", ".", ",", "<space>"],list(V3_TOKENIZER_LAST_PIECES),1025),(1024,list(V3_TOKENIZER_FIRST_PIECES),[(1020, "₽"), (1021, "€"), (1022, "$"), (1023, "!")],1025),(1024,list(V3_TOKENIZER_FIRST_PIECES),list(V3_TOKENIZER_LAST_PIECES),1024)):
  try:validate_v3_tokenizer_structure(piece_count,first,last,num_classes)
  except ValueError:pass
  else:raise AssertionError("invalid v3 tokenizer/RNNT class contract accepted")
 assert all(spec["files"][".gitattributes"][1]=="a6344aac8c09253b3b630fb776ae94478aa0275b" for spec in VARIANTS.values())
 rnnt=config_expectations(VARIANTS["v3"]); ctc=config_expectations(VARIANTS["multilingual"])
 assert rnnt["cfg.model.cfg.head.decoder.pred_hidden"]==320 and "cfg.model.cfg.head.encoder_dim" not in rnnt
 assert rnnt["cfg.model.cfg.head._target_"]=="modeling_gigaam.RNNTHead" and ctc["cfg.model.cfg.head._target_"]=="modeling_gigaam.CTCHead"
 assert rnnt["cfg.model.cfg.decoding._target_"]!=ctc["cfg.model.cfg.decoding._target_"]
 # The official config embeds 70 symbols; CTC's 71st class is its implicit
 # blank (CTCGreedyDecoding.blank_id == len(vocabulary)), not a fake symbol.
 assert len(MULTILINGUAL_VOCABULARY)==70 and len(set(MULTILINGUAL_VOCABULARY))==70
 assert all(key in HISTORICAL_ARTIFACTS[name] for name in ("v3","multilingual") for key in ("repository","revision","filename","bytes","git_blob_sha1","lfs_sha256"))
 with tempfile.TemporaryDirectory(prefix="gigaam-inspect-") as temp:
  root=Path(temp); snapshot=root/"snapshot"; snapshot.mkdir(); sample=snapshot/"x"; sample.write_bytes(b"x"); packet=root/"tree.json"; packet.write_text(json.dumps({"repository":VARIANTS["v3"]["repository"],"revision":VARIANTS["v3"]["revision"],"resolved_revision":VARIANTS["v3"]["revision"],"walk":"recursive_file_only","files":[{"path":"x","type":"file","size":1,"git_blob_sha1":git_blob_sha1(sample),"lfs_sha256":None}]})); bad=[]; assert server_tree(snapshot,packet,VARIANTS["v3"],bad)["status"]=="MATCHED" and not bad
  sample.write_bytes(b"y"); bad=[]; assert server_tree(snapshot,packet,VARIANTS["v3"],bad)["status"]=="MISMATCH" and bad
  lfs_sample=snapshot/"lfs"; lfs_sample.write_bytes(b"payload"); payload_hash=sha256(lfs_sample); pointer="0"*40; assert fixed_identity(lfs_sample,{"git_blob_sha1":pointer,"lfs_sha256":payload_hash},(7,pointer,payload_hash),snapshot); assert not fixed_identity(lfs_sample,{"git_blob_sha1":git_blob_sha1(lfs_sample),"lfs_sha256":payload_hash},(7,pointer,payload_hash),snapshot)
  def nested(expected):
   value={}
   for dotted,item in expected.items():
    cursor=value
    parts=dotted.split(".")
    for part in parts[:-1]: cursor=cursor.setdefault(part,{})
    cursor[parts[-1]]=item
   return value
  config_path=root/"config.json"; config_path.write_text(json.dumps(nested(rnnt)))
  bad=[]; assert config_evidence(config_path,VARIANTS["v3"],bad)["status"]=="EXACT_CONFIG" and not bad
  wrong=nested(rnnt); del wrong["cfg"]["model"]["cfg"]["head"]["decoder"]["pred_hidden"]; wrong["cfg"]["model"]["cfg"]["head"]["encoder_dim"]=768; config_path.write_text(json.dumps(wrong)); bad=[]; assert config_evidence(config_path,VARIANTS["v3"],bad)["status"]=="BLOCKED_CONFIG" and any("pred_hidden" in item for item in bad)
  bad=[]; card=model_card_frontmatter("---\nlicense: mit\n---\n",bad); assert card["license"]=="mit" and not bad
  bad=[]; card=model_card_frontmatter("---\nlicense: mit\ndatasets: x\n---\n",bad); assert card["datasets"]=="x" and not bad
  archive=root/"unsafe.bin";
  with zipfile.ZipFile(archive,"w") as handle: handle.writestr(zipfile.ZipInfo("../escape"),b"x")
  bad=[]; checkpoint_evidence(archive,bad); assert any("unsafe checkpoint archive member" in item for item in bad)
 print("gigaam_inspect self-test: OK")
def model_card_frontmatter(text:str,blockers:list[str])->Any:
 if not text.startswith("---\n"): blockers.append("README front matter missing"); return None
 end=text.find("\n---\n",4)
 if end<0: blockers.append("README front matter unterminated"); return None
 return yaml_frontmatter(text[4:end],blockers)
def yaml_frontmatter(text:str,blockers:list[str])->Any:
 try:
  import yaml
  class Loader(yaml.SafeLoader): pass
  def mapping(loader,node,deep=False):
   pairs=loader.construct_pairs(node,deep=deep); out={}
   for key,value in pairs:
    if key in out: raise ValueError(f"duplicate YAML key: {key}")
    out[key]=value
   return out
  Loader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,mapping); return yaml.load(text,Loader=Loader)
 except Exception as e: blockers.append(f"model card YAML blocked: {e}"); return None
def main()->int:
 parser=argparse.ArgumentParser(); parser.add_argument("variant",choices=sorted(VARIANTS)); parser.add_argument("--self-test",action="store_true"); parser.add_argument("--snapshot",type=Path); parser.add_argument("--source",type=Path); parser.add_argument("--server-tree",type=Path); parser.add_argument("--output",type=Path); args=parser.parse_args()
 if args.self_test:
  if any(value is not None for value in (args.snapshot,args.source,args.server_tree,args.output)): parser.error("--self-test accepts no paths")
  self_test(); return 0
 if any(value is None for value in (args.snapshot,args.source,args.server_tree,args.output)): parser.error("normal run requires snapshot/source/server-tree/output")
 try: return inspect(args.snapshot,args.source,args.server_tree,args.output,args.variant)
 except Exception as error:
  args.output.mkdir(parents=True,exist_ok=True); (args.output/"manifest.json").write_text(json.dumps({"format":FORMAT,"status":"BLOCKED","inspection_status":"INSPECTION_ERROR","evidence_stage":"INSPECTION_ONLY","runtime_status":"NOT_IMPLEMENTED_FAIL_CLOSED","cpu_status":"UNSUPPORTED","metal_status":"BLOCKED_BY_CPU","parity_status":"NOT_RUN","publication":"NO_UPLOAD","error":str(error),"blockers":[str(error)]},indent=2)+"\n"); return 2
if __name__=="__main__": raise SystemExit(main())
