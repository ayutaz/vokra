#!/usr/bin/env bash
# VAST/Linux-x86_64 model-free/source-only audit wave before Scaleway.
set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd -P)"
SELF="${BASH_SOURCE[0]}"
WORK_PARENT=/dev/shm
FORMAT=vokra-mac-pre-scaleway-model-free-batch-v1
MAX_FILE=1048576
MAX_TOTAL=8388608
ACTIVE_HEAD=''
ACTIVE_LANGUAGE=''

die() { printf 'mac-pre-scaleway-batch: BLOCKED: %s\n' "$*" >&2; exit 2; }
usage() {
  cat >&2 <<'EOF'
usage: run-mac-pre-scaleway-model-free-batch.sh \
  --expected-head HEX40 --mms-language CODE --output-dir ABSOLUTE_ABSENT_DIR
       run-mac-pre-scaleway-model-free-batch.sh --self-test
EOF
}
hex40() { [[ "${1-}" =~ ^[0-9a-f]{40}$ ]]; }
language_code() { [[ "${1-}" =~ ^[a-z][a-z0-9_-]{1,31}$ ]]; }
overlap() { [[ "$1" == "$2" || "$1" == "$2"/* || "$2" == "$1"/* ]]; }

canonical_absent() {
  local path="$1" suffix='' name parent
  [[ "$path" == /* && "$path" != *'//'* && "$path" != */./* && "$path" != */../* && "$path" != */. && "$path" != */.. ]] || return 1
  [[ ! -e "$path" && ! -L "$path" ]] || return 1
  while [[ ! -d "$path" || -L "$path" ]]; do
    [[ ! -L "$path" ]] || return 1
    name="${path##*/}"; [[ -n "$name" ]] || return 1
    suffix="/$name$suffix"; parent="${path%/*}"
    [[ "$parent" == "$path" ]] && parent=/
    path="$parent"
  done
  (cd -P "$path" && printf '%s%s\n' "$PWD" "$suffix")
}
no_symlink_ancestors() {
  local path="$1"
  while [[ "$path" != / ]]; do
    [[ ! -L "$path" ]] || return 1
    path="$(dirname "$path")"
  done
}
require_clean_head() {
  local expected="$1" actual
  hex40 "$expected" || die 'expected HEAD must be lowercase HEX40'
  [[ -d "$ROOT/.git" && -f "$ROOT/Cargo.toml" ]] || die 'Vokra checkout missing'
  [[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] || die 'checkout must be clean before host probes, uv, or network'
  actual="$(git -C "$ROOT" rev-parse --verify HEAD)"
  [[ "$actual" == "$expected" ]] || die "HEAD mismatch: expected $expected, observed $actual"
}
require_tools() {
  local tool
  for tool in bash git uv awk sha256sum wc tr findmnt; do
    command -v "$tool" >/dev/null 2>&1 || die "missing required command: $tool"
  done
}
require_output_dir() {
  local requested="$1" canonical root_real
  canonical="$(canonical_absent "$requested")" || die 'output-dir must be absolute, dot-free, absent, and have real ancestors'
  no_symlink_ancestors "$requested" || die 'output-dir has symlinked ancestry'
  root_real="$(cd -P "$ROOT" && pwd)"
  overlap "$canonical" "$root_real" && die 'output-dir overlaps checkout'
  overlap "$canonical" "$WORK_PARENT" && die 'output-dir overlaps tmpfs work root'
  [[ "$canonical" != /tmp/* && "$canonical" != /private/tmp/* && "$canonical" != /var/tmp/* ]] || die 'output-dir must not be a temporary path'
}
require_work() {
  local work="$1" canonical root_real
  [[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || die 'normal mode requires Linux x86_64'
  [[ -d "$WORK_PARENT" && ! -L "$WORK_PARENT" ]] || die '/dev/shm is missing or symlinked'
  [[ "$(findmnt -T "$WORK_PARENT" -n -o FSTYPE 2>/dev/null || true)" == tmpfs ]] || die '/dev/shm must be tmpfs'
  no_symlink_ancestors "$work" || die 'work root has symlinked ancestry'
  [[ ! -e "$work" && ! -L "$work" ]] || die 'work root must be absent (no-clobber)'
  canonical="$(canonical_absent "$work")" || die 'work root is not canonical'
  root_real="$(cd -P "$ROOT" && pwd)"
  if overlap "$canonical" "$root_real"; then
    die 'work root overlaps checkout'
  fi
  return 0
}
safe_evidence_path() {
  local path="$1" size="${2:-0}"
  [[ "$path" == /* && "$path" != *'//'* && "$path" != */./* && "$path" != */../* ]] || return 1
  [[ "$path" != *.safetensors && "$path" != *.pt && "$path" != *.pth && "$path" != *.ckpt && "$path" != *.bin && "$path" != *.gguf && "$path" != *.onnx ]] || return 1
  [[ "$size" =~ ^[0-9]+$ && "$size" -le "$MAX_FILE" ]] || return 1
}
json_status() {
  local path="$1" schema="$2" status="$3" publication="$4"
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python - "$path" "$schema" "$status" "$publication" <<'PY'
import hashlib, json, pathlib, sys
path, schema, status, publication = sys.argv[1:]
def unique(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise SystemExit("duplicate JSON key: " + key)
        out[key] = value
    return out
value = json.loads(pathlib.Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique)
if not isinstance(value, dict) or value.get("schema", value.get("format")) != schema:
    raise SystemExit("evidence schema mismatch")
observed_status = value.get("status")
if status == "AUTHENTICATED":
    activity = value.get("activity")
    if not isinstance(activity, dict) or activity.get("model_weight_acquisition") is not False or activity.get("model_weight_execution") is not False:
        raise SystemExit("source contract model boundary mismatch")
elif status == "SBV2_CONTRACT":
    activity = value.get("activity")
    if not isinstance(activity, dict) or activity.get("model_weight_acquisition") is not False or activity.get("model_weight_execution") is not False:
        raise SystemExit("SBV2 model boundary mismatch")
elif observed_status != status:
    raise SystemExit("evidence factual status mismatch")
observed_publication = value.get("publication")
if observed_publication is None and isinstance(value.get("activity"), dict):
    observed_publication = value["activity"].get("publication")
if observed_publication != publication:
    raise SystemExit("evidence publication mismatch")
PY
}
validate_evidence() {
  local label="$1" path="$2" schema="$3" status="$4" publication="$5" size
  [[ -f "$path" && ! -L "$path" ]] || die "$label did not produce regular evidence"
  size="$(wc -c < "$path" | tr -d ' ')"
  safe_evidence_path "$path" "$size" || die "$label evidence is oversized, unsafe, or payload-like"
  json_status "$path" "$schema" "$status" "$publication" || die "$label evidence disposition is not the documented factual result"
  case "$label" in
    qwen2-audio)
      UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python "$ROOT/tools/parity/qwen2_audio_source_license_audit.py" --validate-evidence "$path" --expected-head "$ACTIVE_HEAD" >/dev/null || die 'Qwen2-Audio deep evidence validator rejected the report' ;;
    mms)
      UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python - "$path" "$ACTIVE_HEAD" "$ACTIVE_LANGUAGE" "$ROOT/tools/parity/mms_1b_all/hf_metadata_audit.py" <<'PY'
import importlib.util, json, pathlib, sys
evidence, expected_head, language, source = sys.argv[1:]
spec = importlib.util.spec_from_file_location("mms_metadata_audit", pathlib.Path(source))
if spec is None or spec.loader is None: raise SystemExit("MMS validator import failed")
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
value = module.strict_json_load(pathlib.Path(evidence).read_bytes())
module.validate_report(value, language, expected_head)
PY
      ;;
    vibevoice-asr)
      UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python "$ROOT/tools/parity/vibevoice_asr_reference/qwen_metadata_audit.py" --validate-evidence --expected-head "$ACTIVE_HEAD" --repo-root "$ROOT" --evidence "$path" >/dev/null || die 'VibeVoice-Qwen deep evidence validator rejected the report' ;;
  esac
}
child_self_test() {
  local child="$1" path="$ROOT/scripts/publish/vast-ai/$child"
  [[ -f "$path" && ! -L "$path" ]] || die "child runner missing: $child"
  bash "$path" --self-test >/dev/null 2>&1 || die "child self-test failed: $child"
}
run_child() {
  local label="$1" log="$2"; shift 2
  [[ ! -e "$log" && ! -L "$log" ]] || die "$label log destination already exists"
  set +e
  "$@" 2>&1 | LC_ALL=C awk -v cap="$MAX_FILE" 'BEGIN { n=0; clipped=0 } { s=$0 "\n"; r=cap-n; if (r>0) { if (length(s)<=r) { printf "%s",s; n+=length(s) } else { printf "%s",substr(s,1,r); n=cap; clipped=1 } } } END { if (clipped) print "[bounded log truncated]" > "/dev/stderr" }' > "$log"
  local -a ps=("${PIPESTATUS[@]}")
  set -e
  [[ "${ps[1]:-1}" == 0 ]] || die "$label log capture failed"
  printf '%s\n' "${ps[0]:-1}"
}
exit_is_expected() { [[ "$1" == "$2" ]]; }
run_audit() {
  local name="$1" expected="$2" evidence="$3" schema="$4" status="$5" publication="$6" log="$7"; shift 7
  local actual
  actual="$(run_child "$name" "$log" "$@")"
  exit_is_expected "$actual" "$expected" || die "$name returned $actual; expected $expected"
  validate_evidence "$name" "$evidence" "$schema" "$status" "$publication"
  printf '%s\t%s\t%s\t%s\t%s\n' "$name" "$expected" "$actual" "$evidence" "$log"
}
build_manifest() {
  local out="$1" expected="$2" lang="$3" actual="$4" rows="$5" work="$6" manifest
  manifest="$out/batch-manifest.json"
  [[ ! -e "$out" && ! -L "$out" ]] || die 'output-dir appeared before final manifest write'
  local total=0 row name expected_rc actual_rc evidence log size
  while IFS=$'\t' read -r name expected_rc actual_rc evidence log; do
    [[ -n "$name" ]] || continue
    for file in "$evidence" "$log"; do
      [[ -f "$file" && ! -L "$file" ]] || die "evidence disappeared: $file"
      size="$(wc -c < "$file" | tr -d ' ')"
      [[ "$size" =~ ^[0-9]+$ && "$size" -le "$MAX_FILE" ]] || die 'evidence exceeds per-file cap'
      total=$((total + size))
      [[ "$total" -le "$MAX_TOTAL" ]] || die 'evidence exceeds total cap'
    done
  done < "$rows"
  mkdir "$out"
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python - "$manifest" "$FORMAT" "$expected" "$actual" "$lang" "$work" "$rows" "$MAX_FILE" "$MAX_TOTAL" <<'PY'
import hashlib, json, pathlib, sys
manifest, fmt, expected, actual, language, work, rows_path, max_file, max_total = sys.argv[1:]
max_file, max_total = int(max_file), int(max_total)
output_root = pathlib.Path(manifest).parent
rows = []
total = 0
for raw in pathlib.Path(rows_path).read_text(encoding="utf-8").splitlines():
    if not raw: continue
    name, expected_rc, actual_rc, evidence, log = raw.split("\t")
    files = []
    for role, raw_path, suffix in (("evidence", evidence, ".evidence.json"), ("bounded_log", log, ".log")):
        source = pathlib.Path(raw_path)
        if source.is_symlink() or not source.is_file(): raise SystemExit("unsafe evidence path")
        size = source.stat().st_size
        if size > max_file: raise SystemExit("evidence file exceeds cap")
        total += size
        if total > max_total: raise SystemExit("evidence total exceeds cap")
        destination = output_root / (name + suffix)
        if destination.exists() or destination.is_symlink(): raise SystemExit("recovered evidence destination is not absent")
        digest = hashlib.sha256()
        with source.open("rb") as reader, destination.open("xb") as writer:
            while True:
                chunk = reader.read(1 << 16)
                if not chunk: break
                digest.update(chunk)
                writer.write(chunk)
        files.append({"role": role, "path": str(destination), "bytes": size, "sha256": digest.hexdigest()})
    rows.append({"name": name, "expected_exit": int(expected_rc), "actual_exit": int(actual_rc), "status": "ACCEPTED_FACTUAL_DISPOSITION", "evidence": files})
if len(rows) != 9 or len({r["name"] for r in rows}) != 9:
    raise SystemExit("exactly nine unique audit rows required")
value = {
    "schema": fmt, "status": "PASS_MODEL_FREE", "expected_head": expected, "actual_head": actual,
    "head_match": expected == actual, "command_identity": "run-mac-pre-scaleway-model-free-batch.sh@" + actual,
    "mms_language": language, "audits": rows,
    "model_payloads": "NOT_ACQUIRED", "publication": "NO_UPLOAD",
    "evidence_limits": {"per_file_bytes": max_file, "total_bytes": max_total, "observed_total_bytes": total},
    "work_root": work,
}
if not value["head_match"] or language == "" or value["model_payloads"] != "NOT_ACQUIRED" or value["publication"] != "NO_UPLOAD":
    raise SystemExit("unsafe batch disposition")
path = pathlib.Path(manifest)
if path.exists() or path.is_symlink(): raise SystemExit("manifest destination is not absent")
path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
PY
  [[ -f "$manifest" && ! -L "$manifest" ]] || die 'batch manifest write failed'
  validate_batch_manifest "$manifest" "$expected" "$lang"
}
validate_batch_manifest() {
  local path="$1" expected="$2" language="$3"
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python - "$path" "$FORMAT" "$expected" "$language" "$MAX_FILE" "$MAX_TOTAL" <<'PY'
import json, pathlib, sys
import hashlib
path, schema, expected, language, max_file, max_total = sys.argv[1:]
max_file, max_total = int(max_file), int(max_total)
def unique(pairs):
    out = {}
    for key, value in pairs:
        if key in out: raise SystemExit("duplicate manifest key: " + key)
        out[key] = value
    return out
value = json.loads(pathlib.Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique)
required = {"schema", "status", "expected_head", "actual_head", "head_match", "command_identity", "mms_language", "audits", "model_payloads", "publication", "evidence_limits", "work_root"}
if set(value) != required: raise SystemExit("batch manifest root schema drift")
if value["schema"] != schema or value["status"] != "PASS_MODEL_FREE" or value["expected_head"] != expected or value["actual_head"] != expected or value["head_match"] is not True or value["mms_language"] != language: raise SystemExit("batch manifest identity/status drift")
if value["model_payloads"] != "NOT_ACQUIRED" or value["publication"] != "NO_UPLOAD": raise SystemExit("unsafe batch manifest disposition")
if value["command_identity"] != "run-mac-pre-scaleway-model-free-batch.sh@" + expected: raise SystemExit("command identity drift")
limits = value["evidence_limits"]
if set(limits) != {"per_file_bytes", "total_bytes", "observed_total_bytes"} or limits["per_file_bytes"] != max_file or limits["total_bytes"] != max_total: raise SystemExit("evidence limits schema drift")
if not isinstance(limits["observed_total_bytes"], int) or limits["observed_total_bytes"] < 0 or limits["observed_total_bytes"] > limits["total_bytes"]: raise SystemExit("evidence total is unsafe")
audits = value["audits"]
expected_exits = {"audiogen": 2, "cosyvoice3": 2, "irodori": 2, "moss-audio": 0, "owsm": 2, "sbv2": 0, "mms": 2, "qwen2-audio": 2, "vibevoice-asr": 2}
expected_order = list(expected_exits)
if not isinstance(audits, list) or len(audits) != 9 or [row.get("name") for row in audits if isinstance(row, dict)] != expected_order: raise SystemExit("batch audit rows are not exact or sequential")
if value["work_root"] != "/dev/shm/vokra-mac-pre-scaleway-model-free-batch-" + expected: raise SystemExit("batch work root identity drift")
observed_total = 0
for row in audits:
    if set(row) != {"name", "expected_exit", "actual_exit", "status", "evidence"} or row["status"] != "ACCEPTED_FACTUAL_DISPOSITION" or row["expected_exit"] != row["actual_exit"]: raise SystemExit("batch audit row schema/status drift")
    if row["expected_exit"] != expected_exits[row["name"]]: raise SystemExit("unexpected child exit contract")
    if not isinstance(row["evidence"], list) or len(row["evidence"]) != 2: raise SystemExit("batch evidence row is not exact")
    expected_files = {"evidence": row["name"] + ".evidence.json", "bounded_log": row["name"] + ".log"}
    if {item.get("role") for item in row["evidence"] if isinstance(item, dict)} != set(expected_files): raise SystemExit("evidence roles are not exact")
    for item in row["evidence"]:
        if set(item) != {"role", "path", "bytes", "sha256"} or item["role"] not in expected_files: raise SystemExit("evidence object schema drift")
        evidence_path = pathlib.Path(item["path"])
        if evidence_path.parent != pathlib.Path(path).parent or evidence_path.name != expected_files[item["role"]] or evidence_path.is_symlink() or not evidence_path.is_file(): raise SystemExit("evidence path escaped output-dir")
        if evidence_path.suffix not in {".json", ".log"} or isinstance(item["bytes"], bool) or not isinstance(item["bytes"], int) or item["bytes"] <= 0 or item["bytes"] != evidence_path.stat().st_size or item["bytes"] > max_file: raise SystemExit("evidence bytes/extension drift")
        digest = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
        if digest != item["sha256"]: raise SystemExit("evidence digest drift")
        observed_total += item["bytes"]
if observed_total != limits["observed_total_bytes"]: raise SystemExit("evidence total drift")
PY
}
self_test() {
  [[ "$#" == 1 ]] || die '--self-test accepts no arguments'
  local fail=0 token
  for token in '--expected-head' '--mms-language' '--output-dir' 'NOT_ACQUIRED' 'NO_UPLOAD' 'PIPESTATUS' 'object_pairs_hook' 'MAX_FILE' 'MAX_TOTAL' 'batch-manifest.json' 'run-audiogen-medium-inspection.sh' 'audit-cosyvoice3-source.sh' 'audit-irodori-text-block-dependencies.sh' 'run-moss-audio-api-smoke.sh' 'run-owsm-v4-medium-1b-inspection.sh' 'run-sbv2-jp-extra-g2p-contract.sh' 'run-mms-1b-all-validation.sh' 'run-qwen2-audio-7b-instruct-inspection.sh' 'run-vibevoice-asr-inspection.sh'; do
    grep -Fq -- "$token" "$SELF" || { echo "self-test missing token: $token" >&2; fail=1; }
  done
  if safe_evidence_path /private/tmp/illegal.safetensors 1 || safe_evidence_path /private/tmp/illegal.json "$((MAX_FILE + 1))"; then
    echo 'self-test extension/size rejection failed' >&2; fail=1
  fi
  safe_evidence_path /private/tmp/ok.json 1 || { echo 'self-test valid evidence rejected' >&2; fail=1; }
  canonical_absent /dev/stdin/child >/dev/null 2>&1 && { echo 'self-test symlink ancestry rejection failed' >&2; fail=1; }
  (require_output_dir /dev/shm/child) >/dev/null 2>&1 && { echo 'self-test tmpfs output rejection failed' >&2; fail=1; }
  if (require_output_dir /tmp/child) >/dev/null 2>&1 || (require_output_dir /var/tmp/child) >/dev/null 2>&1; then echo 'self-test temporary output rejection failed' >&2; fail=1; fi
  if grep -En '(^|[[:space:]])git[[:space:]]+push|(^|[[:space:]])(publish-one|upload)\.sh' "$SELF" | grep -v 'grep -En' >/dev/null; then echo 'self-test publication command found' >&2; fail=1; fi
  if exit_is_expected 7 0; then echo 'self-test accepted an unexpected child exit status' >&2; fail=1; fi
  if printf '%s\n' '{"schema":"wrong"}' | validate_batch_manifest /dev/stdin aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa jpn >/dev/null 2>&1; then echo 'self-test accepted malformed batch manifest' >&2; fail=1; fi
  local fixture head
  fixture="$(mktemp -d /private/tmp/vokra-mac-batch-self-test.XXXXXX)"
  head="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
  local require_work_body
  require_work_body="$(awk '/^require_work\(\) \{/{capture=1} capture {print} capture && /^\}/{exit}' "$SELF")"
  if ! bash -c '
    set -euo pipefail
    WORK_PARENT="$1"; ROOT="$2"
    uname() { [[ "${1-}" == -s ]] && printf "%s\\n" Linux || printf "%s\\n" x86_64; }
    findmnt() { printf "%s\\n" tmpfs; }
    no_symlink_ancestors() { return 0; }
    canonical_absent() { printf "%s\\n" "$1"; }
    overlap() { return 1; }
    '"$require_work_body"'
    require_work "$WORK_PARENT/child"
  ' bash "$fixture" "$fixture"; then
    echo 'self-test non-overlapping work root returned failure' >&2
    fail=1
  fi
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python - "$fixture" "$FORMAT" "$head" <<'PY'
import hashlib, json, pathlib, sys
root, schema, head = sys.argv[1:]
root = pathlib.Path(root)
names = ["audiogen", "cosyvoice3", "irodori", "moss-audio", "owsm", "sbv2", "mms", "qwen2-audio", "vibevoice-asr"]
exits = {"audiogen": 2, "cosyvoice3": 2, "irodori": 2, "moss-audio": 0, "owsm": 2, "sbv2": 0, "mms": 2, "qwen2-audio": 2, "vibevoice-asr": 2}
rows = []
total = 0
for name in names:
    files = []
    for role, suffix, content in (("evidence", ".evidence.json", b"{}\n"), ("bounded_log", ".log", b"ok\n")):
        path = root / (name + suffix)
        path.write_bytes(content)
        files.append({"role": role, "path": str(path), "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()})
        total += len(content)
    rows.append({"name": name, "expected_exit": exits[name], "actual_exit": exits[name], "status": "ACCEPTED_FACTUAL_DISPOSITION", "evidence": files})
value = {"schema": schema, "status": "PASS_MODEL_FREE", "expected_head": head, "actual_head": head, "head_match": True, "command_identity": "run-mac-pre-scaleway-model-free-batch.sh@" + head, "mms_language": "jpn", "audits": rows, "model_payloads": "NOT_ACQUIRED", "publication": "NO_UPLOAD", "evidence_limits": {"per_file_bytes": 1048576, "total_bytes": 8388608, "observed_total_bytes": total}, "work_root": "/dev/shm/vokra-mac-pre-scaleway-model-free-batch-" + head}
(root / "valid.json").write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
mutations = {}
mutations["path-escape.json"] = json.loads(json.dumps(value)); mutations["path-escape.json"]["audits"][0]["evidence"][0]["path"] = "/private/tmp/escape.json"
mutations["hash-tamper.json"] = json.loads(json.dumps(value)); mutations["hash-tamper.json"]["audits"][0]["evidence"][0]["sha256"] = "0" * 64
mutations["size-tamper.json"] = json.loads(json.dumps(value)); mutations["size-tamper.json"]["audits"][0]["evidence"][0]["bytes"] = 1048577
mutations["total-tamper.json"] = json.loads(json.dumps(value)); mutations["total-tamper.json"]["evidence_limits"]["observed_total_bytes"] += 1
mutations["exit-tamper.json"] = json.loads(json.dumps(value)); mutations["exit-tamper.json"]["audits"][0]["actual_exit"] = 0
mutations["name-tamper.json"] = json.loads(json.dumps(value)); mutations["name-tamper.json"]["audits"][0]["name"] = "unknown"
mutations["status-tamper.json"] = json.loads(json.dumps(value)); mutations["status-tamper.json"]["status"] = "BLOCKED"
mutations["role-tamper.json"] = json.loads(json.dumps(value)); mutations["role-tamper.json"]["audits"][0]["evidence"][1]["role"] = "evidence"
mutations["work-root-tamper.json"] = json.loads(json.dumps(value)); mutations["work-root-tamper.json"]["work_root"] = "/dev/shm/other"
for name, payload in mutations.items():
    (root / name).write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
PY
  if ! validate_batch_manifest "$fixture/valid.json" "$head" jpn >/dev/null 2>&1; then echo 'self-test rejected valid synthetic batch manifest' >&2; fail=1; fi
  for mutation in path-escape hash-tamper size-tamper total-tamper exit-tamper name-tamper status-tamper role-tamper work-root-tamper; do
    if validate_batch_manifest "$fixture/$mutation.json" "$head" jpn >/dev/null 2>&1; then echo "self-test accepted $mutation manifest tamper" >&2; fail=1; fi
  done
  if [[ "$fixture" == /private/tmp/vokra-mac-batch-self-test.* && "$fixture" != /private/tmp/vokra-mac-batch-self-test. && -d "$fixture" ]]; then
    rm -rf -- "$fixture"
  else
    echo 'self-test fixture cleanup guard failed; preserving fixture' >&2
    fail=1
  fi
  local head_line host_line work_line
  head_line="$(grep -n '^  require_clean_head "\$expected_head"$' "$SELF" | head -n 1 | cut -d: -f1)"
  host_line="$(grep -n '^  require_tools$' "$SELF" | tail -n 1 | cut -d: -f1)"
  work_line="$(grep -n '^  mkdir "\$work" ||' "$SELF" | tail -n 1 | cut -d: -f1)"
  if [[ ! "$head_line" =~ ^[0-9]+$ || ! "$host_line" =~ ^[0-9]+$ || ! "$work_line" =~ ^[0-9]+$ || "$head_line" -ge "$host_line" || "$head_line" -ge "$work_line" ]]; then echo 'self-test HEAD gate ordering drifted' >&2; fail=1; fi
  if bash "$SELF" --expected-head "$(printf '0%.0s' {1..40})" --expected-head "$(printf '1%.0s' {1..40})" --mms-language jpn --output-dir /var/tmp/nope >/dev/null 2>&1; then echo 'self-test duplicate head accepted' >&2; fail=1; fi
  if bash "$SELF" --expected-head "$(printf '0%.0s' {1..40})" --mms-language jpn --mms-language eng --output-dir /var/tmp/nope >/dev/null 2>&1; then echo 'self-test duplicate language accepted' >&2; fail=1; fi
  if bash "$SELF" --expected-head "$(printf '0%.0s' {1..40})" --output-dir /var/tmp/nope >/dev/null 2>&1; then echo 'self-test missing language accepted' >&2; fail=1; fi
  if bash "$SELF" --expected-head "$(printf '0%.0s' {1..40})" --mms-language jpn >/dev/null 2>&1; then echo 'self-test missing output accepted' >&2; fail=1; fi
  for child in run-audiogen-medium-inspection.sh audit-cosyvoice3-source.sh audit-irodori-text-block-dependencies.sh run-moss-audio-api-smoke.sh run-owsm-v4-medium-1b-inspection.sh run-sbv2-jp-extra-g2p-contract.sh run-mms-1b-all-validation.sh run-qwen2-audio-7b-instruct-inspection.sh run-vibevoice-asr-inspection.sh; do
    child_self_test "$child" || fail=1
  done
  ((fail == 0)) || return 1
  echo 'run-mac-pre-scaleway-model-free-batch.sh self-test: OK'
}
main() {
  if [[ "${1-}" == --self-test ]]; then self_test "$@"; return; fi
  local expected_head='' mms_language='' output_dir='' seen_head=0 seen_language=0 seen_output=0
  while (($#)); do
    case "$1" in
      --expected-head) ((seen_head == 0)) || die 'duplicate --expected-head'; [[ "$#" -ge 2 ]] || die '--expected-head requires HEX40'; expected_head="$2"; seen_head=1; shift 2 ;;
      --mms-language) ((seen_language == 0)) || die 'duplicate --mms-language'; [[ "$#" -ge 2 ]] || die '--mms-language requires code'; mms_language="$2"; seen_language=1; shift 2 ;;
      --output-dir) ((seen_output == 0)) || die 'duplicate --output-dir'; [[ "$#" -ge 2 ]] || die '--output-dir requires absolute absent dir'; output_dir="$2"; seen_output=1; shift 2 ;;
      -h|--help) usage; return ;;
      *) usage; die "unknown argument: $1" ;;
    esac
  done
  ((seen_head == 1 && seen_language == 1 && seen_output == 1)) || { usage; die 'all three options are required'; }
  hex40 "$expected_head" || die '--expected-head must be lowercase HEX40'
  language_code "$mms_language" || die '--mms-language requires explicit lowercase code; no default is permitted'
  require_clean_head "$expected_head"
  ACTIVE_HEAD="$expected_head"
  ACTIVE_LANGUAGE="$mms_language"
  [[ "${VOKRA_PUBLISH_ON_VAST:-0}" == 1 ]] || die 'VOKRA_PUBLISH_ON_VAST=1 is required'
  require_tools
  require_output_dir "$output_dir"
  local work="$WORK_PARENT/vokra-mac-pre-scaleway-model-free-batch-$expected_head"
  require_work "$work"
  mkdir "$work" || die 'work root claim failed'
  mkdir "$work/logs" "$work/evidence" "$work/irodori" "$work/mms" "$work/vibevoice" "$work/sbv2" || die 'work subdirectory creation failed'
  local rows="$work/rows.tsv" actual_head
  : > "$rows"
  actual_head="$(git -C "$ROOT" rev-parse --verify HEAD)"
  run_audit audiogen 2 "/dev/shm/vokra-audiogen-medium-model-free-$expected_head/evidence/manifest.json" vokra-audiogen-medium-inspection-v2 BLOCKED NO_UPLOAD "$work/logs/audiogen.log" env VOKRA_PUBLISH_ON_VAST=1 bash "$ROOT/scripts/publish/vast-ai/run-audiogen-medium-inspection.sh" --model-free --expected-head "$expected_head" >> "$rows"
  run_audit cosyvoice3 2 "$work/cosyvoice3/evidence/source-dependency-audit.json" vokra-cosyvoice3-source-dependency-audit-v1 BLOCKED_UNRESOLVED_COSYVOICE3_COMPOSITE NO_UPLOAD "$work/logs/cosyvoice3.log" env VOKRA_PUBLISH_ON_VAST=1 COSYVOICE3_SOURCE_AUDIT_WORK_DIR="$work/cosyvoice3" bash "$ROOT/scripts/publish/vast-ai/audit-cosyvoice3-source.sh" >> "$rows"
  run_audit irodori 2 "$work/irodori/dependency-audit.json" vokra-irodori-text-block-dependency-audit-v1 BLOCKED_OWNER_REVIEW NO_UPLOAD "$work/logs/irodori.log" env VOKRA_PUBLISH_ON_VAST=1 VOKRA_VAST_AUDIT=1 bash "$ROOT/scripts/publish/vast-ai/audit-irodori-text-block-dependencies.sh" --expected-head "$expected_head" --output "$work/irodori/dependency-audit.json" >> "$rows"
  run_audit moss-audio 0 "$work/moss/model-free-api-smoke-evidence.json" vokra-moss-audio-model-free-api-smoke-v1 PASS_MODEL_FREE NO_UPLOAD "$work/logs/moss-audio.log" env VOKRA_PUBLISH_ON_VAST=1 bash "$ROOT/scripts/publish/vast-ai/run-moss-audio-api-smoke.sh" --model-free --variant all --expected-head "$expected_head" --work-dir "$work/moss" >> "$rows"
  run_audit owsm 2 "/dev/shm/vokra-owsm-v4-medium-1b-source-only/evidence/dependency-audit.json" vokra-owsm-v4-medium-1b-dependency-audit-v1 BLOCKED_OWNER_REVIEW NO_UPLOAD "$work/logs/owsm.log" env VOKRA_PUBLISH_ON_VAST=1 bash "$ROOT/scripts/publish/vast-ai/run-owsm-v4-medium-1b-inspection.sh" --source-only --expected-head "$expected_head" >> "$rows"
  run_audit sbv2 0 "$work/sbv2/contract.json" vokra-sbv2-jp-extra-g2p-v1 SBV2_CONTRACT NO_UPLOAD "$work/logs/sbv2.log" env VOKRA_PUBLISH_ON_VAST=1 bash "$ROOT/scripts/publish/vast-ai/run-sbv2-jp-extra-g2p-contract.sh" --expected-head "$expected_head" --work-dir "$work/sbv2-work" --output "$work/sbv2/contract.json" >> "$rows"
  run_audit mms 2 "$work/mms/metadata.json" vokra-mms-1b-all-hf-metadata-evidence-v2 BLOCKED_PENDING_OWNER_REVIEW NO_UPLOAD "$work/logs/mms.log" env VOKRA_PUBLISH_ON_VAST=1 bash "$ROOT/scripts/publish/vast-ai/run-mms-1b-all-validation.sh" --metadata-only --language "$mms_language" --expected-head "$expected_head" --output "$work/mms/metadata.json" >> "$rows"
  run_audit qwen2-audio 2 "$work/qwen2-audio/source-license-history.json" vokra-qwen2-audio-source-license-history-v1 SOURCE_LICENSE_UNKNOWN_BLOCKER NO_UPLOAD "$work/logs/qwen2-audio.log" env VOKRA_PUBLISH_ON_VAST=1 bash "$ROOT/scripts/publish/vast-ai/run-qwen2-audio-7b-instruct-inspection.sh" --source-license-audit --expected-head "$expected_head" --work-dir "$work/qwen2-audio" >> "$rows"
  run_audit vibevoice-asr 2 "$work/vibevoice/metadata.json" vokra-vibevoice-asr-qwen2-5-7b-metadata-v1 BLOCKED NO_UPLOAD "$work/logs/vibevoice-asr.log" env VOKRA_PUBLISH_ON_VAST=1 bash "$ROOT/scripts/publish/vast-ai/run-vibevoice-asr-inspection.sh" --metadata-only --expected-head "$expected_head" --output "$work/vibevoice/metadata.json" >> "$rows"
  require_clean_head "$expected_head"
  actual_head="$(git -C "$ROOT" rev-parse --verify HEAD)"
  build_manifest "$output_dir" "$expected_head" "$mms_language" "$actual_head" "$rows" "$work"
  printf 'PASS_MODEL_FREE_BATCH manifest=%s model_payloads=NOT_ACQUIRED publication=NO_UPLOAD\n' "$output_dir/batch-manifest.json"
}
main "$@"
