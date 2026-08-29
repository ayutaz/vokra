#!/usr/bin/env bash
# VAST-only official parity and full Rust validation for SpeechT5 TTS.
# This worker never uploads, publishes, pushes, stops, or destroys instances.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VOKRA_ROOT="${VOKRA_ROOT:-$DEFAULT_ROOT}"
VOKRA_SCRATCH="${VOKRA_SCRATCH:-$HOME/scratchpad}"
PARITY_PROJECT="$VOKRA_ROOT/tools/parity/speecht5_tts"
PREFLIGHT_GATE="$PARITY_PROJECT/preflight_gate.py"
PREFLIGHT_MANIFEST="$PARITY_PROJECT/license_gate_manifest.json"
PARITY_DUMPER="$VOKRA_ROOT/tools/parity/speecht5_tts_dump_reference.py"
TTS_PREP="$VOKRA_ROOT/tools/parity/speecht5_tts_prepare_checkpoint.py"
VOCODER_PREP="$VOKRA_ROOT/tools/parity/speecht5_hifigan_prepare_checkpoint.py"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

TTS_REVISION="30fcde30f19b87502b8435427b5f5068e401d5f6"
TTS_SOURCE_SHA256="d60d28067349ef66b50d8cd643ae56b6d6b8f27def929bc4ef6fcad907954190"
PUBLIC_TTS_REVISION="43cf6592038616d116a98fde4764d827ece59033"
PUBLIC_TTS_BYTES="585382432"
PUBLIC_TTS_SHA256="f26019f5e2f7106d834b0b1fd4f66286839e000350caad169388467452c8dde0"
VOCODER_REVISION="bb6f429406e86a9992357a972c0698b22043307d"
VOCODER_SOURCE_SHA256="b171e9bcd8a2b50dc9780040478dfa26783a9ee4be012cf5776914f091d6887b"
MIN_VAST_MEM_KIB=60000000
MIN_FREE_DISK_KIB=30000000

log() { printf '[speecht5-vast] %s\n' "$*" >&2; }
step() { printf '\n[speecht5-vast] ==== %s ====\n' "$*" >&2; }
die() { log "ERROR: $*"; return 2; }

canonicalize_uncreated() {
  local path="$1" suffix='' name parent
  local scan rest component
  [[ "$path" == /* ]] || path="$PWD/$path"
  rest="${path#/}"; scan=''
  while [[ -n "$rest" ]]; do
    component="${rest%%/*}"; rest="${rest#*/}"
    [[ "$component" == "$rest" ]] && rest=''
    [[ -n "$component" && "$component" != . && "$component" != .. ]] || continue
    scan="$scan/$component"; [[ ! -L "$scan" || "$scan" == "/var" ]] || return 1
  done
  while [[ ! -d "$path" || -L "$path" ]]; do
    name="${path##*/}"; [[ -n "$name" ]] && suffix="/$name$suffix"
    parent="${path%/*}"; [[ "$parent" == "$path" ]] && parent='.'
    [[ -n "$parent" ]] || parent='/'; path="$parent"
    [[ ! -L "$path" ]] || return 1
  done
  (cd -P "$path" && printf '%s%s\n' "$PWD" "$suffix")
}
paths_overlap() { [[ "$1" == "$2" || "$1" == "$2"/* || "$2" == "$1"/* ]]; }
require_absent_work_dir() {
  local target="$1" approval="$2" canonical protected other
  [[ ! -e "$target" && ! -L "$target" ]] || { die "--work-dir must be absent and non-symlink: $target"; return 2; }
  canonical="$(canonicalize_uncreated "$target")" || { die "cannot canonicalize --work-dir: $target"; return 2; }
  for protected in "$VOKRA_ROOT" "$PARITY_PROJECT" "$PREFLIGHT_GATE" "$PREFLIGHT_MANIFEST" \
    "$PARITY_PROJECT/uv.lock" "$PARITY_PROJECT/pyproject.toml" "$approval"; do
    [[ -e "$protected" || -L "$protected" ]] || continue
    [[ ! -L "$protected" ]] || { die "protected path is symlinked: $protected"; return 2; }
    other="$(canonicalize_uncreated "$protected")" || { die "cannot canonicalize protected path: $protected"; return 2; }
    paths_overlap "$canonical" "$other" && { die "--work-dir overlaps protected path: $protected"; return 2; }
  done
  return 0
}

usage() {
  cat <<'EOF' >&2
usage: run-speecht5-tts-validation.sh --approval-evidence <json> [--work-dir <absent-dir>]
       run-speecht5-tts-validation.sh --self-test

VAST-only SpeechT5 TTS validation worker. It downloads immutable Microsoft
SpeechT5 TTS + HiFi-GAN sources, verifies their identities, converts both,
authenticates the fixed public tokenizer-less GGUF, runs the independent
Transformers 5.5.0 oracle against both canonical and public text models,
exercises both complete CLI waveform routes, then runs workspace Rust gates.

Actual runs require Linux, VOKRA_PUBLISH_ON_VAST=1, at least 60,000,000 KiB
RAM and 30,000,000 KiB free disk. There is no upload/publish operation. Pull
the small evidence/reference directory and logs, then destroy the instance.
EOF
}

sha256_file() {
  local path="$1"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$path" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$path" | awk '{print $1}'
  else
    die "neither sha256sum nor shasum is available"
  fi
}

require_vast_host() {
  local mem_kib free_kib
  [[ "${VOKRA_PUBLISH_ON_VAST:-0}" == "1" ]] \
    || die "VOKRA_PUBLISH_ON_VAST=1 is absent; run provision.sh first"
  [[ "$(uname -s)" == "Linux" ]] \
    || die "model work is Linux/VAST-only; refusing host $(uname -s)"
  [[ "$(uname -m)" == "x86_64" ]] \
    || die "the locked parity environment targets Linux x86_64, got $(uname -m)"
  mem_kib="$(awk '$1 == "MemTotal:" {print $2; exit}' /proc/meminfo)"
  [[ -n "$mem_kib" ]] || die "could not read MemTotal"
  if (( mem_kib < MIN_VAST_MEM_KIB )); then
    die "MemTotal=${mem_kib} KiB is below the VAST 64-GB class guard"
  fi
  local disk_path="$VOKRA_SCRATCH"
  [[ -e "$disk_path" ]] || disk_path="$(dirname "$disk_path")"
  free_kib="$(df -Pk "$disk_path" | awk 'NR == 2 {print $4}')"
  [[ -n "$free_kib" ]] || die "could not read free disk"
  if (( free_kib < MIN_FREE_DISK_KIB )); then
    die "free disk=${free_kib} KiB is below the 30-GB run guard"
  fi
}

require_tooling() {
  local tool
  for tool in uv cargo rustc git awk curl grep find tee wc tr rustfmt cargo-deny cargo-audit; do
    command -v "$tool" >/dev/null 2>&1 || die "required VAST tool missing: $tool"
  done
  cargo clippy --version >/dev/null 2>&1 \
    || die "the clippy component is missing on the VAST host"
  [[ -d "$VOKRA_ROOT/.git" && -f "$VOKRA_ROOT/Cargo.toml" ]] \
    || die "VOKRA_ROOT is not the repository checkout: $VOKRA_ROOT"
  [[ -f "$PARITY_PROJECT/uv.lock" ]] || die "dedicated parity uv.lock is missing"
  [[ -f "$PARITY_PROJECT/pyproject.toml" && -f "$PREFLIGHT_GATE" && \
    -f "$PREFLIGHT_MANIFEST" ]] || die "SpeechT5 preflight gate inputs are missing"
  for path in "$PARITY_DUMPER" "$TTS_PREP" "$VOCODER_PREP"; do
    [[ -f "$path" ]] || die "required SpeechT5 tool is missing: $path"
  done
  [[ -z "$(git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all)" ]] \
    || die "VAST checkout must be clean so evidence names an exact commit"
}

pre_sync_gate() {
  local approval="$1"
  [[ -f "$PARITY_PROJECT/uv.lock" && ! -L "$PARITY_PROJECT/uv.lock" && \
    -f "$PARITY_PROJECT/pyproject.toml" && ! -L "$PARITY_PROJECT/pyproject.toml" && \
    -f "$PREFLIGHT_GATE" && ! -L "$PREFLIGHT_GATE" && \
    -f "$PREFLIGHT_MANIFEST" && ! -L "$PREFLIGHT_MANIFEST" ]] \
    || die "SpeechT5 pre-sync gate inputs are missing"
  [[ -s "$approval" && ! -L "$approval" ]] || die "approval evidence must be a non-empty regular non-symlink file"
  step "Validate locked dependencies and model/license identities before any sync"
  UV_NO_CACHE=1 UV_CACHE_DIR="${SPEECHT5_UV_CACHE_DIR:-/private/tmp/vokra-speecht5-uv-cache}" \
    uv run --no-cache --no-project --offline --python 3.12 python "$PREFLIGHT_GATE" \
      --project "$PARITY_PROJECT" --manifest "$PREFLIGHT_MANIFEST" \
      --evidence "$approval"
}

write_apple_invocation() {
  local output="$1" public_gguf="$2" reference_dir="$3" reference_digest="$4"
  : "$public_gguf" "$reference_dir"
  {
    printf '# Generated for the separate no-upload Apple validation step.\n'
    printf 'scripts/verify/apple-silicon-speecht5-tts.sh \\\n'
    printf '  --gguf '\''<APPLE_SPEECHT5_GGUF>'\'' \\\n'
    printf '  --reference '\''<APPLE_SPEECHT5_REFERENCE>'\'' \\\n'
    printf '  --reference-sha256 %q \\\n' "$reference_digest"
    printf '  --approval-evidence '\''<APPLE_SPEECHT5_APPROVAL_EVIDENCE>'\'' \\\n'
    printf '  --evidence-dir '\''<APPLE_SPEECHT5_EVIDENCE_DIR>'\''\n'
  } > "$output"
}

require_one_named_test_passed() {
  local log_path="$1" test_name="$2" test_count named_line_count result_count total_result_count
  test_count="$(grep -Ec "^test ${test_name} \.\.\. ok$" "$log_path" || true)"
  named_line_count="$(grep -Ec "^test ${test_name} \.\.\." "$log_path" || true)"
  result_count="$(grep -Ec '^test result: ok\. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out(; finished in [0-9]+\.[0-9]+s)?$' "$log_path" || true)"
  total_result_count="$(grep -Ec '^test result:' "$log_path" || true)"
  [[ "$test_count" == 1 ]] || { die "expected exactly one passing $test_name, got $test_count"; return 2; }
  [[ "$named_line_count" == 1 ]] || { die "expected exactly one total named $test_name line, got $named_line_count"; return 2; }
  [[ "$result_count" == 1 ]] || { die "expected exactly one Cargo result with 1 passed/0 failed/0 ignored"; return 2; }
  [[ "$total_result_count" == 1 ]] || { die "expected exactly one total Cargo result line, got $total_result_count"; return 2; }
}

require_exact_cpu_sentinel() {
  local log_path="$1" count
  local pattern='^SPEECHT5_TTS_OFFICIAL_PARITY backend=cpu frames=[0-9]+ decoder_steps=[0-9]+ before_max_abs=[-+0-9.eE]+ before_index=[0-9]+ after_max_abs=[-+0-9.eE]+ after_index=[0-9]+ bound=[-+0-9.eE]+ verdict=PASS$'
  count="$(grep -Ec "$pattern" "$log_path" || true)"
  [[ "$count" == 1 ]] || die "expected exactly one complete CPU parity sentinel, got $count"
}

run_self_test() {
  local tmp script_path fail=0 cases=0
  tmp="$(mktemp -d)"
  # shellcheck disable=SC2064
  trap "rm -rf '$tmp'" EXIT
  script_path="${BASH_SOURCE[0]}"
  printf '{}\n' > "$tmp/approval.json"
  require_absent_work_dir "$tmp/new-work" "$tmp/approval.json" || fail=1
  mkdir "$tmp/empty-work"
  if require_absent_work_dir "$tmp/empty-work" "$tmp/approval.json" >/dev/null 2>&1; then fail=1; fi
  rmdir "$tmp/empty-work"
  ln -s "$tmp/missing-work" "$tmp/link-work"
  if require_absent_work_dir "$tmp/link-work" "$tmp/approval.json" >/dev/null 2>&1; then fail=1; fi
  rm "$tmp/link-work"
  mkdir -p "$tmp/real-parent/child"
  ln -s "$tmp/real-parent" "$tmp/link-parent"
  if require_absent_work_dir "$tmp/link-parent/child/new-work" "$tmp/approval.json" >/dev/null 2>&1; then fail=1; fi
  rm -rf "$tmp/real-parent" "$tmp/link-parent"
  if require_absent_work_dir "$VOKRA_ROOT/speecht5-self-test-work" "$tmp/approval.json" >/dev/null 2>&1; then fail=1; fi
  if require_absent_work_dir "$tmp/approval.json/child" "$tmp/approval.json" >/dev/null 2>&1; then fail=1; fi

  cases=$((cases + 1))
  UV_CACHE_DIR="${SPEECHT5_UV_CACHE_DIR:-/private/tmp/vokra-speecht5-uv-cache}" \
    uv run --no-cache --no-project --offline --python 3.12 python "$PREFLIGHT_GATE" --self-test \
      >/dev/null 2>&1 || { log "self-test FAIL: preflight gate self-test"; fail=1; }

  cases=$((cases + 1))
  UV_CACHE_DIR="${UV_CACHE_DIR:-$tmp/uv-cache}" \
    uv run --no-cache --no-project --offline --python 3.12 python "$PARITY_DUMPER" --self-test \
      >/dev/null 2>&1 || { log "self-test FAIL: dumper self-test"; fail=1; }

  cases=$((cases + 1))
  for required in "$TTS_REVISION" "$TTS_SOURCE_SHA256" "$VOCODER_REVISION" \
    "$VOCODER_SOURCE_SHA256" "$PUBLIC_TTS_REVISION" "$PUBLIC_TTS_SHA256" \
    "SPEECHT5_TTS_OFFICIAL_PARITY backend=cpu" \
    "--vocoder" "--speaker-embedding" "--frozen --python 3.12" \
    "write_apple_invocation" "--reference-sha256" "<APPLE_SPEECHT5_REFERENCE>" "--approval-evidence"; do
    if ! grep -Fq -- "$required" "$script_path"; then
      log "self-test FAIL: worker contract lost token: $required"
      fail=1
    fi
  done
  local sentinel sentinel_log
  sentinel='SPEECHT5_TTS_OFFICIAL_PARITY backend=cpu frames=2 decoder_steps=1 before_max_abs=1.000000000e-3 before_index=0 after_max_abs=2.000000000e-3 after_index=1 bound=1.000000000e-2 verdict=PASS'
  sentinel_log="$tmp/sentinel.log"
  printf '%s\n' "$sentinel" > "$sentinel_log"
  require_exact_cpu_sentinel "$sentinel_log"
  printf '%s\n%s\n' "$sentinel" "$sentinel" > "$sentinel_log"
  if require_exact_cpu_sentinel "$sentinel_log" >/dev/null 2>&1; then fail=1; fi
  printf 'prefix%s\n' "$sentinel" > "$sentinel_log"
  if require_exact_cpu_sentinel "$sentinel_log" >/dev/null 2>&1; then fail=1; fi
  printf '%s suffix\n' "$sentinel" > "$sentinel_log"
  if require_exact_cpu_sentinel "$sentinel_log" >/dev/null 2>&1; then fail=1; fi
  printf '%s\n' "${sentinel/verdict=PASS/verdict=FAIL}" > "$sentinel_log"
  if require_exact_cpu_sentinel "$sentinel_log" >/dev/null 2>&1; then fail=1; fi
  local cargo_log
  cargo_log="$tmp/cargo.log"
  printf '%s\n%s\n' \
    'test released_cpu_mel_matches_official_transformers ... ok' \
    'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out' \
    > "$cargo_log"
  require_one_named_test_passed "$cargo_log" released_cpu_mel_matches_official_transformers
  printf '%s\n%s\n' \
    'test released_cpu_mel_matches_official_transformers ... ok' \
    'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s' \
    > "$cargo_log"
  require_one_named_test_passed "$cargo_log" released_cpu_mel_matches_official_transformers
  printf '%s\n%s\n' \
    'test released_cpu_mel_matches_official_transformers ... ok' \
    'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; unexpected' \
    > "$cargo_log"
  if require_one_named_test_passed "$cargo_log" released_cpu_mel_matches_official_transformers >/dev/null 2>&1; then fail=1; fi
  printf '%s\n%s\n' \
    'test released_cpu_mel_matches_official_transformers ... FAILED' \
    'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out' \
    > "$cargo_log"
  if require_one_named_test_passed "$cargo_log" released_cpu_mel_matches_official_transformers >/dev/null 2>&1; then fail=1; fi
  printf '%s\n%s\n%s\n' \
    'test released_cpu_mel_matches_official_transformers ... ok' \
    'test released_cpu_mel_matches_official_transformers ... ok' \
    'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out' \
    > "$cargo_log"
  if require_one_named_test_passed "$cargo_log" released_cpu_mel_matches_official_transformers >/dev/null 2>&1; then fail=1; fi
  printf '%s\n%s\n%s\n' \
    'test released_cpu_mel_matches_official_transformers ... ok' \
    'test result: ok. 1 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out' \
    'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out' \
    > "$cargo_log"
  if require_one_named_test_passed "$cargo_log" released_cpu_mel_matches_official_transformers >/dev/null 2>&1; then fail=1; fi
  local apple_args
  apple_args="$tmp/apple.args.sh"
  write_apple_invocation "$apple_args" "$tmp/public.gguf" "$tmp/reference" \
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  grep -F -- "--gguf '<APPLE_SPEECHT5_GGUF>'" "$apple_args" >/dev/null || fail=1
  grep -F -- "--reference '<APPLE_SPEECHT5_REFERENCE>'" "$apple_args" >/dev/null || fail=1
  grep -F -- "--reference-sha256 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef" "$apple_args" >/dev/null || fail=1
  grep -F -- "--approval-evidence '<APPLE_SPEECHT5_APPROVAL_EVIDENCE>'" "$apple_args" >/dev/null || fail=1
  grep -F -- "--evidence-dir '<APPLE_SPEECHT5_EVIDENCE_DIR>'" "$apple_args" >/dev/null || fail=1
  local gate_line sync_line build_line pre_gate_block
  gate_line="$(grep -n '^  pre_sync_gate ' "$script_path" | head -1 | cut -d: -f1)"
  sync_line="$(grep -n '^  uv sync --project' "$script_path" | tail -1 | cut -d: -f1)"
  build_line="$(grep -n '^  cargo build --manifest-path' "$script_path" | tail -1 | cut -d: -f1)"
  [[ "$gate_line" =~ ^[0-9]+$ && "$sync_line" =~ ^[0-9]+$ && "$build_line" =~ ^[0-9]+$ ]] || fail=1
  (( gate_line < sync_line && gate_line < build_line )) || fail=1
  pre_gate_block="$(awk '/^main\(\)/,/^  pre_sync_gate / {print}' "$script_path")"
  [[ "$pre_gate_block" != *"uv sync"* && "$pre_gate_block" != *"cargo build"* && "$pre_gate_block" != *"download_checkpoint"* ]] || fail=1

  cases=$((cases + 1))
  local fake_root fake_home fake_bin trace fake_scratch fake_work rc
  fake_root="$tmp/fake-checkout"
  fake_home="$tmp/fake-home"
  fake_bin="$fake_home/.local/bin"
  trace="$tmp/trace.log"
  fake_scratch="$tmp/fake-scratch"
  fake_work="$fake_root/work"
  mkdir -p "$fake_root/tools/parity/speecht5_tts" "$fake_bin"
  cp "$PARITY_PROJECT/uv.lock" "$fake_root/tools/parity/speecht5_tts/uv.lock"
  cp "$PARITY_PROJECT/pyproject.toml" "$fake_root/tools/parity/speecht5_tts/pyproject.toml"
  printf '{}\n' > "$fake_root/approval.json"
  cp "$PREFLIGHT_GATE" "$fake_root/tools/parity/speecht5_tts/preflight_gate.py"
  cp "$PREFLIGHT_MANIFEST" "$fake_root/tools/parity/speecht5_tts/license_gate_manifest.json"
  cp "$script_path" "$fake_root/run-worker.sh"
  cat > "$fake_bin/uv" <<'EOF'
#!/usr/bin/env bash
printf 'uv %s\n' "$*" >> "${SPEECHT5_TRACE:?}"
exec "${SPEECHT5_REAL_UV:?}" "$@"
EOF
  cat > "$fake_bin/curl" <<'EOF'
#!/usr/bin/env bash
printf 'curl %s\n' "$*" >> "${SPEECHT5_TRACE:?}"
exit 99
EOF
  cat > "$fake_bin/cargo" <<'EOF'
#!/usr/bin/env bash
printf 'cargo %s\n' "$*" >> "${SPEECHT5_TRACE:?}"
exit 99
EOF
  chmod +x "$fake_bin/uv" "$fake_bin/curl" "$fake_bin/cargo"
  git -C "$fake_root" init -q
  git -C "$fake_root" config user.email self-test@example.invalid
  git -C "$fake_root" config user.name self-test
  git -C "$fake_root" add .
  git -C "$fake_root" commit -qm baseline
  printf 'dirty checkout must not outrank the gate\n' > "$fake_root/dirty.txt"
  set +e
  HOME="$fake_home" PATH="$fake_bin:$PATH" SPEECHT5_TRACE="$trace" SPEECHT5_REAL_UV="$(command -v uv)" \
    VOKRA_ROOT="$fake_root" VOKRA_SCRATCH="$fake_scratch" \
    VOKRA_PUBLISH_ON_VAST=1 bash "$fake_root/run-worker.sh" --approval-evidence "$fake_root/approval.json" --work-dir "$fake_work" \
    >/dev/null 2>&1
  rc=$?
  set -e
  [[ "$rc" == 2 ]] || fail=1
  [[ ! -e "$fake_work" && ! -e "$fake_scratch" ]] || fail=1
  if ! grep -Fq 'uv run --no-cache --no-project --offline --python 3.12 python' "$trace"; then
    log "self-test FAIL: pre-sync gate was not traced"
    fail=1
  fi
  if grep -Eq 'uv sync|curl |cargo ' "$trace"; then
    log "self-test FAIL: sync/download/Cargo reached before blocked gate"
    fail=1
  fi

  cases=$((cases + 1))
  if grep -En '^[[:space:]]*(python3|python|pip)([[:space:]]|$)' "$script_path" >/dev/null; then
    log "self-test FAIL: direct Python/pip command found"
    fail=1
  fi

  rm -rf "$tmp"
  trap - EXIT
  if [[ $fail -eq 0 ]]; then
    echo "run-speecht5-tts-validation.sh self-test: OK ($cases cases)"
    return 0
  fi
  return 1
}

record_environment() {
  local output="$1" cpu_model cpu_flags
  cpu_model="$(awk -F ':' '$1 ~ /model name/ {sub(/^[[:space:]]+/, "", $2); print $2; exit}' /proc/cpuinfo)"
  cpu_flags="$(awk -F ':' '$1 ~ /flags/ {sub(/^[[:space:]]+/, "", $2); print $2; exit}' /proc/cpuinfo)"
  [[ -n "$cpu_model" && -n "$cpu_flags" ]] || die "CPU provenance unavailable"
  {
    echo "utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "git_commit=$(git -C "$VOKRA_ROOT" rev-parse HEAD)"
    echo "uname=$(uname -a)"
    echo "cpu_model=$cpu_model"
    echo "cpu_flags=$cpu_flags"
    echo "nproc=$(nproc)"
    awk '$1 == "MemTotal:" {print "mem_total_kib=" $2; exit}' /proc/meminfo
    rustc --version --verbose
    cargo --version
    uv --version
    uv run --project "$PARITY_PROJECT" --frozen --python 3.12 python -c \
      'import platform, torch, transformers; print(f"python={platform.python_version()}"); print(f"torch={torch.__version__}"); print(f"transformers={transformers.__version__}"); print(f"torch_cpu_capability={torch.backends.cpu.get_cpu_capability()}")'
  } | tee "$output"
}

main() {
  local self_test=0 requested_work_dir="" approval_evidence="" run_stamp work_dir source_dir logs_dir reference_dir
  local tts_source vocoder_source tts_gguf public_tts_gguf vocoder_gguf output_wav
  local public_output_wav parity_text public_url public_bytes
  local run_log env_log compile_log parity_log public_parity_log cli_log public_cli_log
  local workspace_log clippy_log summary_file input_hashes_file reference_manifest_sha256 apple_args_file
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --work-dir)
        [[ $# -ge 2 && -n "$2" && "$2" != -* && -z "$requested_work_dir" ]] || { die "--work-dir requires one non-option value"; return 2; }
        requested_work_dir="$2"
        shift 2
        ;;
      --approval-evidence)
        [[ $# -ge 2 && -n "$2" && "$2" != -* && -z "$approval_evidence" ]] || { die "--approval-evidence requires one non-option value"; return 2; }
        approval_evidence="$2"
        shift 2
        ;;
      --self-test) self_test=1; shift ;;
      -h|--help) usage; return 0 ;;
      *) die "unknown argument: $1"; usage; return 2 ;;
    esac
  done
  if [[ $self_test -eq 1 ]]; then
    [[ -z "$requested_work_dir$approval_evidence" ]] || { die "--self-test accepts no other arguments"; return 2; }
    run_self_test
    return $?
  fi

  [[ -n "$approval_evidence" ]] || { usage; die "--approval-evidence is required"; return 2; }
  pre_sync_gate "$approval_evidence"
  require_vast_host
  require_tooling
  run_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  work_dir="${requested_work_dir:-$VOKRA_SCRATCH/speecht5-tts-validation/$run_stamp}"
  require_absent_work_dir "$work_dir" "$approval_evidence"
  mkdir -p "$work_dir"
  source_dir="$work_dir/source"
  logs_dir="$work_dir/evidence/logs"
  reference_dir="$work_dir/evidence/reference"
  tts_source="$source_dir/tts"
  vocoder_source="$source_dir/vocoder"
  tts_gguf="$work_dir/speecht5-tts.gguf"
  public_tts_gguf="$work_dir/speecht5-public.gguf"
  vocoder_gguf="$work_dir/speecht5-hifigan.gguf"
  output_wav="$work_dir/speecht5-cli.wav"
  public_output_wav="$work_dir/speecht5-public-cli.wav"
  mkdir -p "$logs_dir" "$reference_dir" "$tts_source" "$vocoder_source"
  export UV_CACHE_DIR="$VOKRA_SCRATCH/uv-cache-speecht5-tts"
  export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-4}"
  export RUST_BACKTRACE=1
  run_log="$logs_dir/run.log"
  env_log="$logs_dir/environment.txt"
  compile_log="$logs_dir/models-compile.log"
  parity_log="$logs_dir/official-cpu.log"
  public_parity_log="$logs_dir/public-official-cpu.log"
  cli_log="$logs_dir/cli-waveform.log"
  public_cli_log="$logs_dir/public-cli-waveform.log"
  workspace_log="$logs_dir/workspace-test.log"
  clippy_log="$logs_dir/workspace-clippy.log"
  summary_file="$logs_dir/summary.txt"
  input_hashes_file="$logs_dir/input-hashes.txt"
  exec > >(tee -a "$run_log") 2>&1
  # shellcheck disable=SC2154
  trap 'rc=$?; if [[ -n "${summary_file:-}" && ! -f "$summary_file" ]]; then printf "execution_status=FAIL\nexit_code=%s\n" "$rc" > "$summary_file"; fi; exit "$rc"' EXIT

  step "Sync pinned Python 3.12 parity environment"
  uv sync --project "$PARITY_PROJECT" --frozen --python 3.12

  step "Prepare immutable SpeechT5 text and HiFi-GAN sources"
  uv run --project "$PARITY_PROJECT" --frozen --python 3.12 python \
    "$TTS_PREP" --output-dir "$tts_source"
  uv run --project "$PARITY_PROJECT" --frozen --python 3.12 python \
    "$VOCODER_PREP" --output-dir "$vocoder_source" --revision "$VOCODER_REVISION"
  [[ "$(sha256_file "$tts_source/pytorch_model.bin")" == "$TTS_SOURCE_SHA256" ]] \
    || die "SpeechT5 source SHA-256 mismatch after preparation"
  [[ "$(sha256_file "$vocoder_source/pytorch_model.bin")" == "$VOCODER_SOURCE_SHA256" ]] \
    || die "SpeechT5 HiFi-GAN source SHA-256 mismatch after preparation"

  step "Record environment before numerical output"
  record_environment "$env_log"

  step "Compile converter, CLI and focused parity target on VAST"
  cargo build --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release \
    -p vokra-convert -p vokra-cli 2>&1 | tee "$compile_log"
  cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release \
    -p vokra-models --test parity_speecht5_tts_real --no-run \
    2>&1 | tee -a "$compile_log"

  step "Convert both strict GGUF artifacts"
  "$VOKRA_ROOT/target/release/vokra-cli" convert \
    --model speecht5-tts \
    --input "$tts_source/model.safetensors" \
    --tokenizer "$tts_source/spm_char.model" \
    --output "$tts_gguf"
  "$VOKRA_ROOT/target/release/vokra-cli" convert \
    --model speecht5-hifigan \
    --input "$vocoder_source/model.safetensors" \
    --output "$vocoder_gguf"

  step "Download and authenticate exact historical public SpeechT5 GGUF"
  public_url="https://huggingface.co/vokra/speecht5-tts/resolve/$PUBLIC_TTS_REVISION/speecht5.gguf"
  curl --fail --location --retry 5 --retry-all-errors \
    --output "$public_tts_gguf" "$public_url"
  public_bytes="$(wc -c < "$public_tts_gguf" | tr -d '[:space:]')"
  [[ "$public_bytes" == "$PUBLIC_TTS_BYTES" ]] \
    || die "public SpeechT5 GGUF has $public_bytes bytes, expected $PUBLIC_TTS_BYTES"
  [[ "$(sha256_file "$public_tts_gguf")" == "$PUBLIC_TTS_SHA256" ]] \
    || die "public SpeechT5 GGUF SHA-256 mismatch"

  step "Generate independent official Transformers reference"
  uv run --project "$PARITY_PROJECT" --frozen --python 3.12 python \
    "$PARITY_DUMPER" --checkpoint "$tts_source" --output-dir "$reference_dir"
  cp "$reference_dir/reference.json" "$logs_dir/reference-manifest.json"
  reference_manifest_sha256="$(sha256_file "$reference_dir/reference.json")"
  {
    echo "reference_json_sha256=$reference_manifest_sha256"
    echo "reference_json_path=$reference_dir/reference.json"
  } | tee "$input_hashes_file"
  apple_args_file="$logs_dir/apple-silicon-speecht5-tts.args.sh"
  write_apple_invocation "$apple_args_file" "$public_tts_gguf" "$reference_dir" \
    "$reference_manifest_sha256"

  step "Compare native CPU mel with official encoder/decoder/postnet"
  VOKRA_SPEECHT5_TTS_GGUF="$tts_gguf" \
  VOKRA_SPEECHT5_TTS_REFERENCE_DIR="$reference_dir" \
    cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release \
      -p vokra-models --test parity_speecht5_tts_real \
      released_cpu_mel_matches_official_transformers -- --exact --nocapture \
      2>&1 | tee "$parity_log"
  require_one_named_test_passed "$parity_log" \
    released_cpu_mel_matches_official_transformers
  require_exact_cpu_sentinel "$parity_log"

  step "Compare exact public legacy GGUF CPU mel with official reference"
  VOKRA_SPEECHT5_TTS_GGUF="$public_tts_gguf" \
  VOKRA_SPEECHT5_TTS_REFERENCE_DIR="$reference_dir" \
    cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release \
      -p vokra-models --test parity_speecht5_tts_real \
      released_cpu_mel_matches_official_transformers -- --exact --nocapture \
      2>&1 | tee "$public_parity_log"
  require_one_named_test_passed "$public_parity_log" \
    released_cpu_mel_matches_official_transformers
  require_exact_cpu_sentinel "$public_parity_log"

  step "Exercise complete CLI text-to-waveform route"
  IFS= read -r parity_text < "$reference_dir/text.txt"
  "$VOKRA_ROOT/target/release/vokra-cli" run \
    --model "$tts_gguf" \
    --vocoder "$vocoder_gguf" \
    --speaker-embedding "$reference_dir/speaker.f32" \
    --text "$parity_text" --backend cpu --deterministic --output "$output_wav" \
    2>&1 | tee "$cli_log"
  [[ -s "$output_wav" ]] || die "complete SpeechT5 CLI route emitted no WAV"

  step "Exercise exact public legacy GGUF CLI text-to-waveform route"
  "$VOKRA_ROOT/target/release/vokra-cli" run \
    --model "$public_tts_gguf" \
    --vocoder "$vocoder_gguf" \
    --speaker-embedding "$reference_dir/speaker.f32" \
    --text "$parity_text" --backend cpu --deterministic --output "$public_output_wav" \
    2>&1 | tee "$public_cli_log"
  [[ -s "$public_output_wav" ]] \
    || die "public legacy SpeechT5 CLI route emitted no WAV"

  step "Run full workspace verification on VAST"
  cargo fmt --manifest-path "$VOKRA_ROOT/Cargo.toml" --all -- --check
  bash "$VOKRA_ROOT/scripts/check-forbidden-symbols.sh"
  bash "$VOKRA_ROOT/scripts/check-zero-deps.sh"
  bash "$VOKRA_ROOT/scripts/check-bound-arch-coverage.sh"
  bash "$VOKRA_ROOT/scripts/check-arch-handshake.sh"
  cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --workspace \
    2>&1 | tee "$workspace_log"
  cargo clippy --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --workspace \
    --all-targets -- -D warnings 2>&1 | tee "$clippy_log"
  cargo deny --manifest-path "$VOKRA_ROOT/Cargo.toml" check licenses advisories bans
  cargo audit --file "$VOKRA_ROOT/Cargo.lock"

  {
    echo "execution_status=PASS"
    echo "numeric_verdict=PASS"
    echo "numeric_bound=0.01"
    echo "git_commit=$(git -C "$VOKRA_ROOT" rev-parse HEAD)"
    echo "tts_revision=$TTS_REVISION"
    echo "tts_source_sha256=$TTS_SOURCE_SHA256"
    echo "public_tts_revision=$PUBLIC_TTS_REVISION"
    echo "public_tts_bytes=$PUBLIC_TTS_BYTES"
    echo "public_tts_sha256=$(sha256_file "$public_tts_gguf")"
    echo "vocoder_revision=$VOCODER_REVISION"
    echo "vocoder_source_sha256=$VOCODER_SOURCE_SHA256"
    echo "tts_gguf_sha256=$(sha256_file "$tts_gguf")"
    echo "vocoder_gguf_sha256=$(sha256_file "$vocoder_gguf")"
    echo "reference_manifest_sha256=$reference_manifest_sha256"
    echo "cli_wav_sha256=$(sha256_file "$output_wav")"
    echo "public_cli_wav_sha256=$(sha256_file "$public_output_wav")"
    grep -F "SPEECHT5_TTS_OFFICIAL_PARITY" "$parity_log"
    grep -F "SPEECHT5_TTS_OFFICIAL_PARITY" "$public_parity_log"
  } | tee "$summary_file"
  trap - EXIT
  log "PASS: pull $work_dir/evidence, then destroy the VAST instance"
  log "Do not pull the source/safetensors/GGUF model artifacts to the maintainer Mac"
}

main "$@"
