#!/usr/bin/env bash
# Source-only Apple handoff gate for VoxCPM-0.5B.
# The 0.5B native composite (LM + AudioVAE + tokenizer) is not implemented;
# this worker therefore stops before reading model/reference payloads.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL_REPOSITORY="openbmb/VoxCPM-0.5B"
MODEL_REVISION="e95e62437bb940c8aeb9f26dc3169d436d2bb455"
SOURCE_REPOSITORY="https://github.com/OpenBMB/VoxCPM.git"
SOURCE_REVISION="38a76704ee67935ccbafbe5b6725e83dbb1e9305"
PUBLIC_REPOSITORY="vokra/voxcpm-0.5b"
PUBLIC_REVISION="ee0ca6d5b9fab27bbb626b5cb3f01236e582d004"
AUDIOVAE_SOURCE="src/voxcpm/modules/audiovae/audio_vae.py"
TOKENIZER_FILES='["config.json","special_tokens_map.json","tokenizer.json","tokenizer_config.json"]'

die() { printf 'voxcpm-apple: ERROR: %s\n' "$*" >&2; exit 2; }
sha256_file() { shasum -a 256 "$1" | awk '{print $1}'; }

canonical_existing_path() {
  local path="$1" rest component current=/ parent base
  [[ "$path" == /* && -e "$path" && ! -L "$path" ]] || return 1
  rest="${path#/}"
  while [[ -n "$rest" ]]; do
    component="${rest%%/*}"; [[ "$rest" == "$component" ]] && rest="" || rest="${rest#*/}"
    [[ -n "$component" && "$component" != . && "$component" != .. ]] || return 1
    current="${current%/}/$component"; [[ ! -L "$current" ]] || return 1
  done
  if [[ -d "$path" ]]; then (cd -P "$path" && pwd); else
    parent="$(dirname "$path")"; base="$(basename "$path")"
    parent="$(cd -P "$parent" && pwd)" || return 1; printf '%s/%s\n' "$parent" "$base"
  fi
}

canonical_absent_path() {
  local path="$1" target="$1" rest component current=/ suffix='' real
  [[ "$path" == /* && ! -e "$path" && ! -L "$path" ]] || return 1
  rest="${path#/}"
  while [[ -n "$rest" ]]; do
    component="${rest%%/*}"; [[ "$rest" == "$component" ]] && rest="" || rest="${rest#*/}"
    [[ -n "$component" && "$component" != . && "$component" != .. ]] || return 1
    current="${current%/}/$component"; [[ ! -L "$current" ]] || return 1
  done
  while [[ ! -e "$target" && ! -L "$target" ]]; do
    component="$(basename "$target")"; suffix="/$component$suffix"; target="$(dirname "$target")"
  done
  [[ -d "$target" && ! -L "$target" ]] || return 1
  real="$(cd -P "$target" && pwd)" || return 1; printf '%s%s\n' "$real" "$suffix"
}

require_clean_expected_head() {
  local expected="$1" actual
  [[ "$expected" =~ ^[0-9a-f]{40}$ ]] || die '--expected-head must be lowercase 40-hex'
  [[ -d "$ROOT/.git" && -f "$ROOT/Cargo.toml" ]] || die 'Vokra checkout is missing'
  [[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] || die 'checkout must be clean'
  actual="$(git -C "$ROOT" rev-parse HEAD)" || die 'cannot read checkout HEAD'
  [[ "$actual" == "$expected" ]] || die "checkout HEAD $actual differs from --expected-head $expected"
}

validate_approval() {
  local path="$1" expected_head="$2" supplied_sha="$3"
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python - "$path" "$expected_head" "$supplied_sha" "$MODEL_REPOSITORY" "$MODEL_REVISION" "$SOURCE_REPOSITORY" "$SOURCE_REVISION" "$PUBLIC_REPOSITORY" "$PUBLIC_REVISION" "$AUDIOVAE_SOURCE" "$TOKENIZER_FILES" <<'PY'
import hashlib, json, pathlib, sys
path, expected_head, supplied_sha, model_repo, model_rev, source_repo, source_rev, public_repo, public_rev, audio_vae_source, tokenizer_files = sys.argv[1:]
raw = pathlib.Path(path).read_bytes()
if hashlib.sha256(raw).hexdigest() != supplied_sha:
    raise SystemExit("approval bytes changed")
def pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("duplicate approval key: " + key)
        result[key] = value
    return result
data = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs)
keys = {"schema", "status", "disposition", "expected_head", "model_repository", "model_revision", "source_repository", "source_revision", "public_repository", "public_revision", "audio_vae_source", "tokenizer_files", "license_status", "dependency_status", "audio_vae_status", "tokenizer_status", "native_status", "no_upload", "scope_sha256"}
if set(data) != keys:
    raise ValueError("approval schema is not exact")
expected = {
    "schema": "vokra-voxcpm-0.5b-approval-v1", "status": "BLOCKED", "disposition": "INSPECTION_ONLY",
    "expected_head": expected_head, "model_repository": model_repo, "model_revision": model_rev,
    "source_repository": source_repo, "source_revision": source_rev, "public_repository": public_repo,
    "public_revision": public_rev, "audio_vae_source": audio_vae_source, "tokenizer_files": json.loads(tokenizer_files),
    "license_status": "DOCS_SIGNED_APACHE_2_0", "dependency_status": "BLOCKED_UNREVIEWED",
    "audio_vae_status": "UNRESOLVED", "tokenizer_status": "UNRESOLVED",
    "native_status": "NOT_IMPLEMENTED_FAIL_CLOSED", "no_upload": True,
}
if any(data[key] != value for key, value in expected.items()):
    raise ValueError("approval identity or unresolved gate mismatch")
scope = {key: data[key] for key in keys if key != "scope_sha256"}
if data["scope_sha256"] != hashlib.sha256(json.dumps(scope, sort_keys=True, separators=(",", ":")).encode()).hexdigest():
    raise ValueError("approval scope mismatch")
PY
}

self_test() {
  local failed=0 token gate
  for token in VOKRA_REMOTE_APPLE_SILICON=1 Darwin arm64 Metal INSPECTION_ONLY BLOCKED \
    NO_UPLOAD UNRESOLVED NOT_IMPLEMENTED_FAIL_CLOSED BLOCKED_UNRESOLVED_AUDIOVAE_TOKENIZER_NATIVE \
    --expected-head --approval-evidence --approval-sha256 --offline --test-threads=1 "$MODEL_REPOSITORY" \
    "$MODEL_REVISION" "$SOURCE_REVISION" "$PUBLIC_REVISION" "$AUDIOVAE_SOURCE" \
    'duplicate approval key' 'approval schema is not exact' 'DOCS_SIGNED_APACHE_2_0' \
    'canonical_absent_path' 'unsafe ancestry'; do
    grep -Fq -- "$token" "$0" || { printf 'self-test missing %s\n' "$token" >&2; failed=1; }
  done
  if grep -En 'git[[:space:]]+push|upload\.sh|publish-one\.sh|--push|--upload|status=PASS|PASS' "$0" | grep -v 'grep -En' >/dev/null; then
    printf 'self-test found success/upload marker\n' >&2; failed=1
  fi
  if grep -En '(^|[[:space:]])(curl|wget|snapshot_download|hf_hub_download)([[:space:]]|$)' "$0" >/dev/null; then
    printf 'self-test found acquisition command\n' >&2; failed=1
  fi
  gate="$(awk '/^  die '\''BLOCKED_UNRESOLVED_AUDIOVAE_TOKENIZER_NATIVE/{print NR; exit}' "$0")"
  [[ "$gate" =~ ^[0-9]+$ ]] || { printf 'self-test cannot locate blocker\n' >&2; failed=1; }
  main_line="$(awk '/^main\(\)/{print NR; exit}' "$0")"
  [[ "$main_line" =~ ^[0-9]+$ ]] || { printf 'self-test cannot locate main\n' >&2; failed=1; }
  if ! awk -v start="$main_line" -v gate="$gate" 'NR > start && NR < gate && /(xcrun|cargo|gguf|reference|transfer)/ {bad=1} END {exit bad}' "$0"; then
    printf 'hardware/payload operation precedes blocker\n' >&2; failed=1
  fi
  if "$0" --self-test --self-test >/dev/null 2>&1 || \
    "$0" --expected-head bad >/dev/null 2>&1 || \
    "$0" --expected-head "$(printf '0%.0s' {1..40})" --expected-head "$(printf '1%.0s' {1..40})" >/dev/null 2>&1 || \
    "$0" --approval-sha256 bad >/dev/null 2>&1 || \
    "$0" --transfer-manifest /tmp/a --transfer-manifest /tmp/b >/dev/null 2>&1; then
    printf 'self-test accepted malformed or duplicate options\n' >&2; failed=1
  fi
  (( failed == 0 )) && printf 'apple-silicon-voxcpm-0-5b.sh self-test: OK\n' || return 1
}

usage() {
  cat <<'EOF'
usage: apple-silicon-voxcpm-0-5b.sh --expected-head HEX40 \
  --approval-evidence FILE --approval-sha256 HEX64
       apple-silicon-voxcpm-0-5b.sh --self-test
EOF
}

main() {
  local expected_head='' approval='' approval_sha='' seen=''
  while (( $# )); do
    case "$1" in
      --self-test) [[ $# == 1 ]] || die '--self-test accepts no arguments'; self_test; return 0;;
      --expected-head) [[ "$seen" != *' expected_head '* ]] || die 'duplicate --expected-head'; [[ $# -ge 2 && "$2" =~ ^[0-9a-f]{40}$ ]] || die '--expected-head requires lowercase 40-hex'; expected_head="$2"; seen="$seen expected_head"; shift 2;;
      --approval-evidence) [[ "$seen" != *' approval '* ]] || die 'duplicate --approval-evidence'; [[ $# -ge 2 && "$2" == /* ]] || die '--approval-evidence requires absolute path'; approval="$2"; seen="$seen approval"; shift 2;;
      --approval-sha256) [[ "$seen" != *' approval_sha '* ]] || die 'duplicate --approval-sha256'; [[ $# -ge 2 && "$2" =~ ^[0-9a-f]{64}$ ]] || die '--approval-sha256 requires lowercase 64-hex'; approval_sha="$2"; seen="$seen approval_sha"; shift 2;;
      -h|--help) usage; return 0;; *) die "unknown argument: $1";;
    esac
  done
  [[ "$seen" == *' expected_head '* && "$seen" == *' approval '* && "$seen" == *' approval_sha '* ]] || { usage; die 'expected-head and approval arguments are required'; }
  [[ -f "$approval" && ! -L "$approval" && -s "$approval" ]] || die 'approval evidence is missing, empty, or symlinked'
  canonical_existing_path "$approval" >/dev/null || die 'approval path has unsafe ancestry'
  [[ "$(sha256_file "$approval")" == "$approval_sha" ]] || die 'approval evidence SHA-256 mismatch'
  require_clean_expected_head "$expected_head"
  validate_approval "$approval" "$expected_head" "$approval_sha"
  # Do not hash/read the large bundle or probe hardware before this factual
  # companion/native gate. INSPECTION_ONLY cannot authorize execution.
  die 'BLOCKED_UNRESOLVED_AUDIOVAE_TOKENIZER_NATIVE: no model, reference, hardware, Cargo, or evidence work is permitted'
}

main "$@"
