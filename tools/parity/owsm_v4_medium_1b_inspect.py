#!/usr/bin/env -S uv run --frozen --project tools/parity --python 3.12 python
"""Inspection-only evidence collector for ESPnet OWSM v4 medium 1B."""
from __future__ import annotations
import argparse, hashlib, itertools, json, os, posixpath, re, stat, struct, subprocess, sys, tempfile, zipfile
from pathlib import Path, PurePosixPath
from typing import Any

HF_REPOSITORY="espnet/owsm_v4_medium_1B"; HF_REVISION="e10985c8f1d592e905c24d2ac2b2c53e3feb24dc"
SOURCE_REPOSITORY="https://github.com/espnet/espnet.git"; SOURCE_REVISION="cccc29023d43a3f504e28df7d1324bb4eb6daedd"; SOURCE_TAG="v.202412"
FORMAT="vokra-owsm-v4-medium-1b-inspection-v1"
MAIN="exp/s2t_train_conv2d8_size1024_e18_d18_mel128_raw_bpe50000/valid.total_count.ave_5best.pth"; BPE="data/token_list/bpe_unigram50000/bpe.model"; STATS="exp/s2t_stats_raw_bpe50000/train/feats_stats.npz"; CONFIG="exp/s2t_train_conv2d8_size1024_e18_d18_mel128_raw_bpe50000/config.yaml"; README="README.md"
KNOWN_LFS={MAIN:(4_089_134_806,"b02d79f29a4daa31dd49ce145d9bb4cda0a1b68cdad91ae0af170ec3a4e92e09"),BPE:(1_044_580,"7ddb01f03dab493c18ab69391e98744c090f897890d8b529b30cae52a8d9eef4"),STATS:(1_786,"00c22dba27594df8f1d8f74a491b20c6e6e8c17e92159f81dfd634f98c098654"),CONFIG:(494_398,None),README:(5_458,None)}
EXPECTED_BLOBS={MAIN:"4b69fd6e811f46357c6e930868ae27412bed8d87",BPE:"e28ada30c05cbd3a8fc727fe9aefb03dcf2e29ed",STATS:"81dc0b816d8ccfe65c4606442e80552f2bec95ed",CONFIG:"fbf425c85d183f9103cb5e2c84ebffb0f425a930",README:"b44deaf39631cd6b0a55faeaa0c4bbe6e1a23f63"}
EXPECTED_README_SHA="0a6706b003418c3d64aabb153afdb08627c52be7add1cca0944b63b9e9849055"; EXPECTED_README_SIZE=5_458
EXPECTED_STATS_KEYS={"count","sum","sum_square"}; TOKEN_SHA="e19396ec012b0294a11fe85c35e36a1d903bc83e60ea602ddf6cc59b7c0e92f9"
RUNTIME_STATUS="LOUD_PARTIAL_FAIL_CLOSED"; CPU_STATUS="NOT_RUN"
IGNORE={".cache",".git"}
SOURCE_SEMANTIC_FILES={
 "frontend.default":"espnet2/asr/frontend/default.py",
 "frontend.stft":"espnet2/layers/stft.py",
 "frontend.log_mel":"espnet2/layers/log_mel.py",
 "global_mvn":"espnet2/layers/global_mvn.py",
}
SOURCE_SEMANTIC_MARKERS={
 "frontend.default":(
  r"class\s+DefaultFrontend\b", r"Stft\s*\(", r"LogMel\s*\(",
  r"_compute_stft", r"self\.stft", r"self\.logmel",
 ),
 "frontend.stft":(r"class\s+Stft\b", r"center", r"torch\.stft"),
 "frontend.log_mel":(r"class\s+LogMel\b", r"n_mels", r"mel"),
 "global_mvn":(
  r"class\s+GlobalMVN\b", r"[\"']count[\"']", r"[\"']sum[\"']",
  r"[\"']sum_square[\"']", r"mean", r"std",
 ),
}
SOURCE_SEMANTIC_CONTRACT={
 "frontend.default":(
  ("stft_domain_conversion", r"self\._compute_stft\(input\s*,\s*input_lengths\)"),
  ("complex_to_power", r"input_stft\.real\s*\*\*\s*2\s*\+\s*input_stft\.imag\s*\*\*\s*2"),
  ("log_mel_application", r"self\.logmel\s*\(input_power\s*,\s*feats_lens\)"),
  ("return_feature_lengths", r"return\s+input_feats\s*,\s*feats_lens"),
 ),
 "frontend.stft":(
  ("torch_stft", r"output\s*=\s*torch\.stft\(input\.float\(\),\s*\*\*stft_kwargs\)"),
  ("torch_stft_return_complex", r"stft_kwargs\[\s*[\"']return_complex[\"']\s*\]\s*=\s*True"),
  ("center_binding", r"self\.center\s*=\s*center"),
  ("torch_stft_kwargs_without_pad_mode", r"stft_kwargs\s*=\s*dict\(\s*n_fft=self\.n_fft,\s*win_length=self\.win_length,\s*hop_length=self\.hop_length,\s*center=self\.center,\s*window=window,\s*normalized=self\.normalized,\s*onesided=self\.onesided,\s*\)"),
  ("center_length_pad", r"if\s+self\.center\s*:\s*pad\s*=\s*self\.n_fft\s*//\s*2"),
  ("frame_length_formula", r"torch\.div\(\s*ilens\s*-\s*self\.n_fft\s*,\s*self\.hop_length,\s*rounding_mode\s*=\s*[\"']trunc[\"']\s*\)"),
 ),
 "frontend.log_mel":(
  ("mel_filterbank", r"MelScale|melmat|mel_filter"),
  ("matrix_application", r"torch\.matmul\s*\(\s*feat\s*,\s*self\.melmat\s*\)"),
  ("log_output", r"log\s*\(|torch\.log|np\.log"),
 ),
 "global_mvn":(
  ("npz_count", r"stats\s*\[\s*[\"']count[\"']\s*\]"),
  ("npz_sum", r"stats\s*\[\s*[\"']sum[\"']\s*\]"),
  ("npz_sum_square", r"stats\s*\[\s*[\"']sum_square[\"']\s*\]"),
  ("mean_from_sum", r"mean\s*=\s*sum_v\s*/\s*count"),
  ("variance_from_second_moment", r"var\s*=\s*sum_square_v\s*/\s*count\s*-\s*mean\s*\*\s*mean"),
  ("epsilon_clamped_std", r"np\.sqrt\s*\(\s*np\.maximum\s*\(\s*var\s*,\s*eps\s*\)\s*\)"),
  ("padding_mask", r"make_pad_mask\s*\(\s*ilens\s*,\s*x\s*,\s*1\s*\)"),
  ("mean_normalization", r"x\s*-\=\s*self\.mean"),
  ("variance_normalization", r"x\s*/=\s*self\.std"),
 ),
}

def sha256(path:Path)->str:
 d=hashlib.sha256()
 with path.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""): d.update(b)
 return d.hexdigest()
_TEMP_COUNTER=itertools.count()
def reject_raw_path(path:Path,label:str)->None:
 raw=os.fspath(path)
 if not raw or "\x00" in raw or any(part in (".","..") for part in raw.split("/")): raise RuntimeError(f"unsafe {label} path")
def validate_raw_cli_paths(argv:list[str])->None:
 options={"--snapshot":"snapshot","--source":"source","--server-tree":"server-tree","--output":"output"}
 index=0
 while index<len(argv):
  argument=argv[index]
  matched=None; raw=None
  for option,label in options.items():
   if argument==option:
    matched=label
    if index+1<len(argv): raw=argv[index+1]
    index+=1
    break
   if argument.startswith(option+"="):
    matched=label; raw=argument[len(option)+1:]; break
  if matched is not None:
   if raw is None: index+=1; continue
   if not raw.startswith("/") or raw=="/" or "\x00" in raw or any(part in (".","..") for part in raw.split("/")): raise RuntimeError(f"unsafe {matched} CLI path")
  index+=1
def validate_existing_dir(path:Path,label:str)->None:
 reject_raw_path(path,label)
 absolute=Path(os.path.abspath(os.fspath(path))); current=Path(absolute.anchor)
 for part in absolute.parts[1:]:
  current/=part; info=os.lstat(current)
  if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode): raise RuntimeError(f"unsafe {label} directory ancestry: {current}")
def validate_existing_file(path:Path,label:str)->None:
 reject_raw_path(path,label)
 absolute=Path(os.path.abspath(os.fspath(path))); current=Path(absolute.anchor)
 for part in absolute.parts[1:-1]:
  current/=part; info=os.lstat(current)
  if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode): raise RuntimeError(f"unsafe {label} directory ancestry: {current}")
 info=os.lstat(absolute)
 if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode): raise RuntimeError(f"unsafe {label} input type")
def validate_dir(path:Path,label:str)->None:
 absolute=Path(os.path.abspath(os.fspath(path))); current=Path(absolute.anchor)
 for part in absolute.parts[1:]:
  current /= part; info=os.lstat(current)
  if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode): raise RuntimeError(f"unsafe {label} directory ancestry: {current}")
def ensure_dir(path:Path,label:str)->None:
 reject_raw_path(path,label)
 if os.path.lexists(path) and (path.is_symlink() or not path.is_dir()): raise RuntimeError(f"{label} must be a regular directory")
 path.mkdir(parents=True,exist_ok=True); validate_dir(path,label)
def write_atomic_no_replace(path:Path,payload:bytes)->None:
 """Atomically publish one evidence file without clobbering prior output."""
 reject_raw_path(path,"evidence"); ensure_dir(path.parent,"evidence parent")
 if os.path.lexists(path): raise FileExistsError(path)
 flags=os.O_CREAT|os.O_EXCL|getattr(os,"O_NOFOLLOW",0)|os.O_WRONLY
 temporary=None; fd=-1
 for _ in range(128):
  candidate=path.parent/f".{path.name}.owsm-tmp-{os.getpid()}-{next(_TEMP_COUNTER)}"
  try:
   fd=os.open(candidate,flags,0o600); temporary=candidate; break
  except FileExistsError: continue
 if temporary is None: raise RuntimeError("unable to allocate unique evidence temporary")
 info=os.fstat(fd); identity=(info.st_dev,info.st_ino); linked=False
 try:
  offset=0
  while offset<len(payload):
   written=os.write(fd,payload[offset:])
   if written<=0: raise OSError("zero-byte evidence write")
   offset += written
  os.fsync(fd); os.close(fd); fd=-1
  validate_dir(path.parent,"evidence parent")
  verify_fd=os.open(temporary,os.O_RDONLY|getattr(os,"O_NOFOLLOW",0))
  try:
   verified=os.fstat(verify_fd)
   if not stat.S_ISREG(verified.st_mode) or (verified.st_dev,verified.st_ino)!=identity: raise RuntimeError("evidence temporary identity changed")
   os.fsync(verify_fd)
  finally: os.close(verify_fd)
  os.link(temporary,path,follow_symlinks=False); linked=True
  try:
   directory_fd=os.open(path.parent,os.O_RDONLY|getattr(os,"O_DIRECTORY",0)|getattr(os,"O_NOFOLLOW",0))
   try: os.fsync(directory_fd)
   finally: os.close(directory_fd)
  except OSError: pass
 finally:
  if fd>=0: os.close(fd)
  try:
   if linked: temporary.unlink()
   else:
    current=os.lstat(temporary)
    if (current.st_dev,current.st_ino)==identity and stat.S_ISREG(current.st_mode): temporary.unlink()
  except FileNotFoundError: pass
  except OSError: pass
def git_blob_sha1(path:Path)->str:
 data=path.read_bytes(); return hashlib.sha1(f"blob {len(data)}\0".encode()+data).hexdigest()
def git_blob_sha1_bytes(data:bytes)->str:
 return hashlib.sha1(f"blob {len(data)}\0".encode()+data).hexdigest()
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
def normalized_symlink_target(relative:str,target_bytes:bytes)->tuple[str,str]:
 if not target_bytes or b"\x00" in target_bytes: raise ValueError("NUL in symlink target")
 if target_bytes.startswith(b"/"): raise ValueError("absolute symlink target")
 target=os.fsdecode(target_bytes)
 normalized=posixpath.normpath(posixpath.join(posixpath.dirname(relative),target))
 if normalized==".." or normalized.startswith("../"): raise ValueError("symlink target escapes checkout")
 return target,normalized
def tracked_symlink(root:Path,relative:str,index_object_id:str,blockers:list[str])->dict[str,Any]|None:
 """Authenticate a tracked symlink from Git blobs without dereferencing it."""
 try:
  relative_path=PurePosixPath(relative)
  if (not relative or "\x00" in relative or "\\" in relative or relative_path.is_absolute()
      or ".." in relative_path.parts): raise ValueError("unsafe tracked symlink path")
  if not re.fullmatch(r"[0-9a-f]{40}",index_object_id): raise ValueError("invalid index blob object ID")
  path=root/relative
  if not path.is_symlink(): raise ValueError("working tree entry is not a symlink")
  index_bytes=subprocess.run(["git","-C",str(root),"cat-file","blob",index_object_id],check=True,capture_output=True).stdout
  if git_blob_sha1_bytes(index_bytes)!=index_object_id: raise ValueError("index blob object does not match target bytes")
  target,normalized=normalized_symlink_target(relative,index_bytes)
  head_spec=f"HEAD:{relative}"
  head_object_id=subprocess.run(["git","-C",str(root),"rev-parse","--verify",head_spec],check=True,capture_output=True,text=True).stdout.strip()
  if not re.fullmatch(r"[0-9a-f]{40}",head_object_id): raise ValueError("invalid HEAD blob object ID")
  head_bytes=subprocess.run(["git","-C",str(root),"cat-file","blob",head_spec],check=True,capture_output=True).stdout
  if head_bytes!=index_bytes: raise ValueError("HEAD blob differs from index blob")
  working_target=os.readlink(path); working_bytes=os.fsencode(working_target)
  _,working_normalized=normalized_symlink_target(relative,working_bytes)
  if working_bytes!=index_bytes: raise ValueError("working-tree symlink target differs from Git blobs")
  return {"path":relative,"index_object_id":index_object_id,"head_object_id":head_object_id,
          "index_target":target,"head_target":target,"working_target":working_target,
          "normalized_target":normalized,"working_normalized_target":working_normalized,"target_scope":"CHECKOUT_RELATIVE_NO_DEREFERENCE",
          "target_git_blob_sha1":git_blob_sha1_bytes(index_bytes)}
 except (OSError,UnicodeError,ValueError,subprocess.CalledProcessError) as error:
  blockers.append(f"unsafe tracked symlink {relative}: {error}")
  return None
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
  tracked=set(); paths=[]; symlinks=[]; gitlinks=[]
  for raw in entries:
   if not raw: continue
   meta,rel=raw.split(b"\t",1); fields=meta.split(); mode=fields[0].decode(); object_id=fields[1].decode() if len(fields)>1 else ""; relative=rel.decode(); tracked.add(relative); path=root/relative
   if mode=="160000":
    gitlinks.append({"path":relative,"object_id":object_id,"status":"GITLINK_NOT_CHECKED_OUT"}); blockers.append(f"source nonregular/gitlink: {path}")
   elif mode=="120000":
    record=tracked_symlink(root,relative,object_id,blockers)
    if record is not None: symlinks.append(record)
   elif mode not in ("100644","100755"): blockers.append(f"source nonregular/gitlink: {path}")
   elif path.is_file() and not path.is_symlink(): paths.append(path)
   else: blockers.append(f"source tracked file missing/nonregular: {path}")
  clean=subprocess.run(["git","-C",str(root),"status","--porcelain","--untracked-files=all"],check=True,capture_output=True,text=True).stdout
  if head!=revision or origin!=repo: blockers.append(f"source identity mismatch: {repo}")
  if SOURCE_TAG not in tags: blockers.append(f"source tag missing at fixed revision: {SOURCE_TAG}")
  if clean: blockers.append(f"source checkout is dirty: {repo}")
  role_files=[]
  for role in roles:
   path=root/role
   if role not in tracked or not path.is_file() or path.is_symlink(): blockers.append(f"source role is not tracked/present/nonregular: {role}")
   else:
    actual_blob=git_blob_sha1(path)
    index_blob=subprocess.run(["git","-C",str(root),"rev-parse","--verify",f":{role}"],check=True,capture_output=True,text=True).stdout.strip()
    head_blob=subprocess.run(["git","-C",str(root),"rev-parse","--verify",f"HEAD:{role}"],check=True,capture_output=True,text=True).stdout.strip()
    if actual_blob!=index_blob or actual_blob!=head_blob: blockers.append(f"source role Git blob mismatch: {role}")
    role_files.append({**identity(path,root),"index_git_blob_sha1":index_blob,"head_git_blob_sha1":head_blob})
  licenses=[p for p in sorted(root.glob("LICENSE*")) if p.is_file() and not p.is_symlink()]
  if not licenses: blockers.append(f"source license missing: {repo}")
  result.update({"resolved_revision":head,"origin":origin,"tags_at_revision":tags,"clean_status":"CLEAN" if not clean else "DIRTY","tracked_files":[identity(p,root) for p in sorted(paths)],"symlinks":symlinks,"gitlinks":gitlinks,"role_files":role_files,"license_files":[identity(p,root) for p in licenses]})
 except Exception as e: blockers.append(f"source inventory blocked: {e}")
 return result


def source_semantic_evidence(root:Path, blockers:list[str])->dict[str,Any]:
 """Authenticate source-level frontend/MVN seams without executing ESPnet."""
 evidence={}
 for name,relative in SOURCE_SEMANTIC_FILES.items():
  path=root/relative
  packet={"path":relative,"status":"BLOCKED_SOURCE_SEMANTICS"}
  try:
   if not path.is_file() or path.is_symlink():
    raise RuntimeError("source role is missing or non-regular")
   text=path.read_text(encoding="utf-8")
   markers=SOURCE_SEMANTIC_MARKERS[name]
   matched=[marker for marker in markers if re.search(marker,text,re.MULTILINE)]
   missing=[marker for marker in markers if marker not in matched]
   expressions=SOURCE_SEMANTIC_CONTRACT[name]
   expression_matches=[label for label,expression in expressions if re.search(expression,text,re.MULTILINE)]
   expression_missing=[label for label,_ in expressions if label not in expression_matches]
   packet.update({"sha256":sha256(path),"git_blob_sha1":git_blob_sha1(path),"markers":markers,"matched":matched,"missing":missing,"semantic_expressions":[{"label":label,"pattern":expression} for label,expression in expressions],"semantic_expression_matches":expression_matches,"semantic_expression_missing":expression_missing})
   if missing:
    blockers.append(f"source semantic markers missing: {name}: {missing}")
   if expression_missing:
    blockers.append(f"source semantic expressions missing: {name}: {expression_missing}")
   if not missing and not expression_missing:
    packet["status"]="SOURCE_MARKERS_MATCHED"
  except (OSError,UnicodeError,RuntimeError) as error:
   blockers.append(f"source semantic evidence blocked: {name}: {error}")
   packet["error"]=str(error)
  evidence[name]=packet
 frontend=evidence.get("frontend.default",{})
 stft=evidence.get("frontend.stft",{})
 log_mel=evidence.get("frontend.log_mel",{})
 mvn=evidence.get("global_mvn",{})
 all_matched=all(packet.get("status")=="SOURCE_MARKERS_MATCHED" for packet in evidence.values())
 return {
  "status":"SOURCE_SEMANTICS_AUTHENTICATED" if all_matched else "BLOCKED_SOURCE_SEMANTICS",
  "frontend":{
   "pipeline":["waveform", "DefaultFrontend._compute_stft", "Stft", "LogMel"],
   "config_binding":{"fs":"16k","n_fft":512,"win_length":400,"hop_length":160,"n_mels":128},
   "source_files":{"default":frontend,"stft":stft,"log_mel":log_mel},
  },
  "global_mvn":{
   "stats_file":STATS,
   "stats_keys":sorted(EXPECTED_STATS_KEYS),
   "source_file":mvn,
   "normalization_formula_status":"SOURCE_EXPRESSIONS_AUTHENTICATED_NO_RUNTIME_FORMULA_EXECUTED" if all_matched else "BLOCKED_SOURCE_EXPRESSIONS",
  },
 }
def inspect(snapshot:Path,source:Path,tree:Path,out:Path)->int:
 validate_existing_dir(snapshot,"snapshot")
 validate_existing_dir(source,"source")
 validate_existing_file(tree,"server-tree")
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
 source_root=source
 source_inventory_packet=source_inventory(source_root,SOURCE_REPOSITORY,SOURCE_REVISION,("espnet2/s2t/espnet_model.py","espnet2/tasks/s2t.py","espnet2/asr/encoder/e_branchformer_encoder.py","espnet2/asr/decoder/transformer_decoder.py","espnet2/asr/frontend/default.py","espnet2/layers/stft.py","espnet2/layers/log_mel.py","espnet2/asr/specaug/specaug.py","espnet2/layers/global_mvn.py","espnet2/asr/ctc.py","espnet2/train/preprocessor.py"),blockers)
 source_semantics=source_semantic_evidence(source_root,blockers)
 inspection_status="AUTHENTICATED_EVIDENCE_COMPLETE" if not blockers else "INSPECTION_ERROR"
 blockers += ["native ESPnet S2T frontend/subsampling/encoder/decoder is not implemented","joint CTC/attention beam search and special-token semantics are not implemented","independent CPU numerical parity is not run","Metal is blocked by CPU runtime","dependency provenance is unreviewed","dataset provenance is unauthenticated"]
 payload={"format":FORMAT,"status":"BLOCKED","inspection_status":inspection_status,"evidence_stage":"INSPECTION_ONLY","runtime_status":RUNTIME_STATUS,"cpu_status":CPU_STATUS,"metal_status":"BLOCKED_BY_CPU","parity_status":"NOT_RUN","publication":"NO_UPLOAD","model":{"repository":HF_REPOSITORY,"revision":HF_REVISION,"server_tree":tree_packet,"files":[identity(p,snapshot) for p in local],"config":config,"checkpoint":checkpoint,"bpe":bpe,"readme":readme,"stats":stats},"official_source":source_inventory_packet,"source_semantics":source_semantics,"license_evidence":{"weights":{"status":"AUTHENTICATED_FROM_MODEL_CARD" if readme and readme.get("status")=="AUTHENTICATED_MODEL_CARD" else "BLOCKED_MODEL_CARD","spdx":"cc-by-4.0","card":"README.md"},"espnet_source":"Apache/MIT source declaration requires review","dependencies":"UNREVIEWED_BLOCKER","datasets":"UNAUTHENTICATED_BLOCKER"},"blockers":sorted(set(blockers))}
 ensure_dir(out,"evidence output")
 write_atomic_no_replace(out/"manifest.json",(json.dumps(payload,sort_keys=True,indent=2,default=list)+"\n").encode("utf-8")); return 2
def write_error_manifest(out:Path,error:Exception)->None:
 payload={"format":FORMAT,"status":"BLOCKED","inspection_status":"INSPECTION_ERROR","evidence_stage":"INSPECTION_ONLY","runtime_status":RUNTIME_STATUS,"cpu_status":CPU_STATUS,"metal_status":"BLOCKED_BY_CPU","parity_status":"NOT_RUN","publication":"NO_UPLOAD","error":str(error),"blockers":[str(error)]}
 ensure_dir(out,"error evidence output")
 try: write_atomic_no_replace(out/"manifest.json",(json.dumps(payload,indent=2)+"\n").encode("utf-8"))
 except FileExistsError: pass
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
  config.write_text("model: espnet\nfrontend: wrong\n"); bad=[]; packet=config_evidence(config,root,bad); assert packet["contract_status"]=="BLOCKED_FACTS" and bad
  config.write_text("model: espnet\ntoken_list: [1]\n"); bad=[]; packet=config_evidence(config,root,bad); assert packet["token_list"]["status"]=="BLOCKED_TOKEN_LIST" and any("string array" in x for x in bad)
  symlink_tmp=tempfile.TemporaryDirectory(prefix="owsm-symlink-"); symlink_repo=Path(symlink_tmp.name)
  subprocess.run(["git","init","-q",str(symlink_repo)],check=True,capture_output=True)
  subprocess.run(["git","-C",str(symlink_repo),"config","user.email","test@example.invalid"],check=True)
  subprocess.run(["git","-C",str(symlink_repo),"config","user.name","OWSM self-test"],check=True)
  (symlink_repo/"README.md").write_text("ok\n"); (symlink_repo/"docs").mkdir(); link=symlink_repo/"docs"/"README.md"; link.symlink_to("../README.md")
  subprocess.run(["git","-C",str(symlink_repo),"add","README.md","docs/README.md"],check=True,capture_output=True); subprocess.run(["git","-C",str(symlink_repo),"commit","-qm","initial"],check=True,capture_output=True)
  index_record=subprocess.run(["git","-C",str(symlink_repo),"ls-files","-s","--","docs/README.md"],check=True,capture_output=True,text=True).stdout.strip(); index_fields,index_path=index_record.split("\t",1); index_object=index_fields.split()[1]
  symlink_bad=[]; symlink_packet=tracked_symlink(symlink_repo,index_path,index_object,symlink_bad)
  assert symlink_packet is not None and symlink_packet["index_target"]=="../README.md" and symlink_packet["head_object_id"]==index_object and not symlink_bad
  link.unlink(); link.symlink_to("../../outside"); symlink_bad=[]; assert tracked_symlink(symlink_repo,index_path,index_object,symlink_bad) is None and any("escapes checkout" in item for item in symlink_bad)
  link.unlink(); link.symlink_to("/outside"); symlink_bad=[]; assert tracked_symlink(symlink_repo,index_path,index_object,symlink_bad) is None and any("absolute" in item for item in symlink_bad)
  link.unlink(); link.symlink_to("../README.md"); nul_object=subprocess.run(["git","-C",str(symlink_repo),"hash-object","-w","--stdin"],input=b"../README\x00",check=True,capture_output=True).stdout.decode().strip(); symlink_bad=[]; assert tracked_symlink(symlink_repo,index_path,nul_object,symlink_bad) is None and any("NUL" in item for item in symlink_bad)
  link.unlink(); link.symlink_to("README.md"); symlink_bad=[]; assert tracked_symlink(symlink_repo,index_path,index_object,symlink_bad) is None and any("differs" in item for item in symlink_bad)
  gitlink_tmp=tempfile.TemporaryDirectory(prefix="owsm-gitlink-"); gitlink_repo=Path(gitlink_tmp.name); subprocess.run(["git","init","-q",str(gitlink_repo)],check=True,capture_output=True); subprocess.run(["git","-C",str(gitlink_repo),"remote","add","origin","wrong/repository"],check=True,capture_output=True); subprocess.run(["git","-C",str(gitlink_repo),"config","user.email","test@example.invalid"],check=True); subprocess.run(["git","-C",str(gitlink_repo),"config","user.name","OWSM self-test"],check=True); (gitlink_repo/"README").write_text("ok\n"); subprocess.run(["git","-C",str(gitlink_repo),"add","README"],check=True,capture_output=True); subprocess.run(["git","-C",str(gitlink_repo),"commit","-qm","initial"],check=True,capture_output=True); subprocess.run(["git","-C",str(gitlink_repo),"update-index","--add","--cacheinfo","160000,"+"1"*40+",submodule"],check=True,capture_output=True); gitlink_bad=[]; gitlink_packet=source_inventory(gitlink_repo,"wrong/repository","0"*40,tuple(),gitlink_bad); assert gitlink_packet["gitlinks"] and any("source nonregular/gitlink" in item for item in gitlink_bad)
  readme=root/README; readme.write_bytes(b"not a model card\xff"); bad=[]; packet=readme_evidence(readme,root,bad); assert packet["status"]=="BLOCKED_README" and bad
  card="---\nlicense: cc-by-4.0\ndatasets:\n- espnet/yodas_owsmv4\n---\nLanguage identification, recognition, translation, timestamp and long-form.\n"; bad=[]; parsed=model_card_frontmatter(card,bad)
  try:
   import yaml  # type: ignore[import-not-found]
  except ImportError:
   assert parsed is None and bad
  else:
   contract=validate_model_card(parsed,card,bad); assert parsed["datasets"]==["espnet/yodas_owsmv4"] and contract["status"]=="AUTHENTICATED_MODEL_CARD" and not bad
   bad=[]; string_card="---\nlicense: cc-by-4.0\ndatasets: espnet/yodas_owsmv4\n---\nLanguage identification, recognition, translation, timestamp and long-form.\n"; parsed=model_card_frontmatter(string_card,bad); contract=validate_model_card(parsed,string_card,bad); assert parsed["datasets"]=="espnet/yodas_owsmv4" and contract["status"]=="BLOCKED_MODEL_CARD" and any("README dataset declaration mismatch" in item for item in bad)
  bad_yaml=root/"bad.yaml"; bad_yaml.write_text("x: 1\nx: 2\n"); bad=[]; yaml_value(bad_yaml,bad); assert bad
  semantic_root=root/"semantic-source"; (semantic_root/"espnet2/asr/frontend").mkdir(parents=True); (semantic_root/"espnet2/layers").mkdir(parents=True)
  (semantic_root/"espnet2/asr/frontend/default.py").write_text("""class DefaultFrontend:
 def __init__(self):
  self.stft = Stft()
  self.logmel = LogMel()
 def forward(self, input, input_lengths):
  input_stft, feats_lens = self._compute_stft(input, input_lengths)
  input_power = input_stft.real ** 2 + input_stft.imag ** 2
  input_feats, _ = self.logmel(input_power, feats_lens)
  return input_feats, feats_lens
""")
  (semantic_root/"espnet2/layers/stft.py").write_text("""import torch
class Stft:
 def __init__(self, center=True):
  self.center = center
  pad_mode = 'reflect'
 def forward(self, input):
  stft_kwargs = dict(n_fft=self.n_fft, win_length=self.win_length, hop_length=self.hop_length, center=self.center, window=window, normalized=self.normalized, onesided=self.onesided,)
  stft_kwargs['return_complex'] = True
  output = torch.stft(input.float(), **stft_kwargs)
 if self.center:
  pad = self.n_fft // 2
 olens = torch.div(ilens - self.n_fft, self.hop_length, rounding_mode='trunc') + 1
""")
  (semantic_root/"espnet2/layers/log_mel.py").write_text("""class LogMel:
 def __init__(self):
  self.n_mels = 128
  self.melmat = True
 def forward(self, feat):
  mel_feat = torch.matmul(feat, self.melmat)
  return mel_feat.log()
""")
  (semantic_root/"espnet2/layers/global_mvn.py").write_text("""import numpy as np
class GlobalMVN:
 def __init__(self, stats):
  count = stats['count']
  sum_v = stats['sum']
  sum_square_v = stats['sum_square']
  mean = sum_v / count
  var = sum_square_v / count - mean * mean
  std = np.sqrt(np.maximum(var, eps))
 def forward(self, x, ilens):
  mask = make_pad_mask(ilens, x, 1)
 x -= self.mean
  x /= self.std
  return x, ilens
""")
  semantic_bad=[]; semantic=source_semantic_evidence(semantic_root,semantic_bad); assert semantic["status"]=="SOURCE_SEMANTICS_AUTHENTICATED" and not semantic_bad
  (semantic_root/"espnet2/layers/log_mel.py").write_text("class LogMel:\n")
  semantic_bad=[]; semantic=source_semantic_evidence(semantic_root,semantic_bad); assert semantic["status"]=="BLOCKED_SOURCE_SEMANTICS" and semantic_bad
 safe_parent=Path("/private/tmp") if Path("/private/tmp").is_dir() else Path(tempfile.gettempdir())
 with tempfile.TemporaryDirectory(prefix="owsm-error-",dir=safe_parent) as error_tmp:
  error_out=Path(error_tmp); write_error_manifest(error_out,RuntimeError("self-test failure")); error=json.loads((error_out/"manifest.json").read_text()); assert error["inspection_status"]=="INSPECTION_ERROR" and error["publication"]=="NO_UPLOAD" and error["runtime_status"]==RUNTIME_STATUS and error["cpu_status"]==CPU_STATUS
 with tempfile.TemporaryDirectory(prefix="owsm-output-",dir=safe_parent) as output_tmp:
  output_root=Path(output_tmp); evidence_dir=output_root/"evidence"; manifest=evidence_dir/"manifest.json"
  validate_raw_cli_paths(["--output="+str(manifest)])
  for unsafe in (("--output="+str(output_root)+"/../unsafe.json",), ("--snapshot","/"), ("--source=relative/input",)):
   try: validate_raw_cli_paths(list(unsafe))
   except RuntimeError: pass
   else: raise AssertionError("unsafe raw CLI path was accepted")
  write_atomic_no_replace(manifest,b"first\n"); assert manifest.read_bytes()==b"first\n" and not list(evidence_dir.glob(".manifest.json.owsm-tmp-*"))
  try: write_atomic_no_replace(manifest,b"replacement\n")
  except FileExistsError: pass
  else: raise AssertionError("existing evidence was clobbered")
  assert manifest.read_bytes()==b"first\n"
  error_manifest=output_root/"error"/"manifest.json"; write_error_manifest(error_manifest.parent,RuntimeError("first error")); before=error_manifest.read_bytes(); write_error_manifest(error_manifest.parent,RuntimeError("second error")); assert error_manifest.read_bytes()==before
  real=output_root/"real"; real.mkdir(); link=output_root/"link"; link.symlink_to(real,target_is_directory=True)
  try: write_atomic_no_replace(link/"blocked.json",b"blocked")
  except RuntimeError: pass
  else: raise AssertionError("symlink ancestry was accepted")
  input_root=output_root/"inputs"; input_root.mkdir(); (input_root/"snapshot").mkdir(); (input_root/"source").mkdir(); (input_root/"tree.json").write_text("{}")
  input_link=output_root/"input-link"; input_link.symlink_to(input_root,target_is_directory=True)
  for unsafe_input,validator in ((input_link/"snapshot",validate_existing_dir),(input_link/"source",validate_existing_dir),(input_link/"tree.json",validate_existing_file)):
   try: validator(unsafe_input,"self-test input")
   except RuntimeError: pass
   else: raise AssertionError("input symlink ancestry was accepted")
  try: reject_raw_path(Path(str(output_root)+"/../dot.json"),"self-test")
  except RuntimeError: pass
  else: raise AssertionError("lexical dot path was accepted")
  original_link=os.link
  def fail_link(*args:Any,**kwargs:Any)->None: raise OSError("injected link failure")
  os.link=fail_link  # type: ignore[assignment]
  failed=output_root/"failed.json"
  try:
   try: write_atomic_no_replace(failed,b"failed")
   except OSError: pass
   else: raise AssertionError("injected publish failure was ignored")
  finally: os.link=original_link  # type: ignore[assignment]
  assert not failed.exists() and not list(output_root.glob(".failed.json.owsm-tmp-*"))
  original_open=os.open; replacement_temp=None
  def replace_temp(path:Any,flags:int,mode:int=0o777,*,dir_fd:Any=None)->int:
   nonlocal replacement_temp
   candidate=Path(path)
   if replacement_temp is None and candidate.name.startswith(".replacement.json.owsm-tmp-") and not (flags&os.O_CREAT):
    candidate.unlink(); replacement_temp=candidate; attacker_fd=original_open(candidate,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600); original_write=0
    while original_write<8:
     written=os.write(attacker_fd,b"attacker\n"[original_write:])
     if written<=0: raise OSError("self-test attacker write stalled")
     original_write+=written
    os.close(attacker_fd)
   if dir_fd is None: return original_open(path,flags,mode)
   return original_open(path,flags,mode,dir_fd=dir_fd)
  os.open=replace_temp  # type: ignore[assignment]
  replacement=output_root/"replacement.json"
  try:
   try: write_atomic_no_replace(replacement,b"owner\n")
   except RuntimeError as error: assert "identity changed" in str(error)
   else: raise AssertionError("replaced evidence temporary was published")
  finally: os.open=original_open  # type: ignore[assignment]
  assert not replacement.exists() and replacement_temp is not None and replacement_temp.read_bytes()==b"attacker\n"
  replacement_temp.unlink()
 print("owsm_v4_medium_1b_inspect self-test: OK")
def main()->int:
 parser=argparse.ArgumentParser(); parser.add_argument("--self-test",action="store_true"); parser.add_argument("--snapshot",type=Path); parser.add_argument("--source",type=Path); parser.add_argument("--server-tree",type=Path); parser.add_argument("--output",type=Path); args=parser.parse_args()
 try: validate_raw_cli_paths(sys.argv[1:])
 except RuntimeError as error: parser.error(str(error))
 if args.self_test:
  if any(x is not None for x in (args.snapshot,args.source,args.server_tree,args.output)): parser.error("--self-test accepts no other arguments")
  self_test(); return 0
 if any(x is None for x in (args.snapshot,args.source,args.server_tree,args.output)): parser.error("normal run requires snapshot/source/server-tree/output")
 try: return inspect(args.snapshot,args.source,args.server_tree,args.output)
 except Exception as error:
  write_error_manifest(args.output,error); return 2
if __name__=="__main__": raise SystemExit(main())
