#!/usr/bin/env bash
# Reproduce Bark Small/Full CPU parity on VAST. No upload path.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VOKRA_ROOT="${VOKRA_ROOT:-$DEFAULT_ROOT}"
VOKRA_SCRATCH="${VOKRA_SCRATCH:-$HOME/scratchpad}"
PARITY_PROJECT="$VOKRA_ROOT/tools/parity/bark"
LICENSE_GATE="$PARITY_PROJECT/license_gate.py"
LICENSE_MANIFEST="$PARITY_PROJECT/license_gate_manifest.json"
DEPENDENCY_AUDIT="$PARITY_PROJECT/dependency_audit.py"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

TRANSFORMERS_REVISION="c1c34249fa27deefbd4a377dfbf883a39baf5c6d"
GENERATION_CONFIG_BYTES=4908
GENERATION_CONFIG_SHA256="ab2969fcd40e085bc924ad99ad419c27f62f5acb61afac5de7490ab0c796b5b9"

SMALL_PUBLIC_REPO="vokra/bark-small"
SMALL_PUBLIC_REVISION="09802c56a2b2e8ad87835115b94b38031fde29b6"
SMALL_PUBLIC_BYTES=1674074848
SMALL_PUBLIC_SHA256="43b781a0dcd66f1e7451005e461ec20e2141bc9c4f529feb4a9a8c0e352ea137"
SMALL_UPSTREAM_REPO="suno/bark-small"
SMALL_UPSTREAM_REVISION="1dbd7a128513b8ae4a4e2130fed57b7ac9da5bcd"
SMALL_CHECKPOINT_BYTES=1676663913
SMALL_CHECKPOINT_SHA256="f0f7f16b24f65789ce42b3c491aa6a1cdf219f7ef425066fcd194485245e65d9"
SMALL_CONFIG_BYTES=8803
SMALL_CONFIG_SHA256="9d95e9c3027cd79cf5f762cc03a69b6393cea87c51e9dd6b998fde3a7f01510e"

FULL_PUBLIC_REPO="vokra/bark"
FULL_PUBLIC_REVISION="f304ddcdfd9218994731ec3b09e89b9961b8b751"
FULL_PUBLIC_BYTES=4466390272
FULL_PUBLIC_SHA256="fd628312ce7d8e1cbc41718741614116d5c7f08d0763f81622edbac320b208ec"
FULL_UPSTREAM_REPO="suno/bark"
FULL_UPSTREAM_REVISION="70a8a7d34168586dc5d028fa9666aceade177992"
FULL_CHECKPOINT_BYTES=4486643861
FULL_CHECKPOINT_SHA256="4e3d407b9b3b619da184c85786c88e5e35f90f9089303e16db696ed0be477989"
FULL_CONFIG_BYTES=8806
FULL_CONFIG_SHA256="48be144c0232acd8c55786d1eea9161ae6c973f21ec4a2f02627c844065ea695"
TRANSFORMERS_SDIST_SHA256="c8db656cf51c600cd8c75f06b20ef85c72e8b8ff9abc880c5d3e8bc70e0ddcbd"
TRANSFORMERS_WHEEL_SHA256="821a9ff0961abbb29eb1eb686d78df1c85929fdf213a3fe49dc6bd94f9efa944"
SMALL_TEST="real_bark_small_matches_official_transformers"
FULL_TEST="real_bark_full_matches_official_transformers"

MIN_VAST_MEM_KIB=23000000
MIN_FREE_DISK_KIB=40000000

log() { printf '[bark-vast] %s\n' "$*" >&2; }
step() { printf '\n[bark-vast] ==== %s ====\n' "$*" >&2; }
die() { log "ERROR: $*"; return 2; }

usage() {
  cat <<'EOF' >&2
usage: run-bark-validation.sh --approval-evidence <file> [--work-dir <absent-dir>]
       run-bark-validation.sh --self-test

VAST-only, non-publishing Bark Small/Full validation. The worker downloads and
verifies both exact public Vokra GGUFs and exact immutable Suno checkpoints,
uses locked official Transformers 5.5.0 for independent greedy references,
audits the already synchronized Python closure without importing model code,
compiles the workspace plus Apple target, verifies CLI routing, and compares
native CPU generated codes plus embedded-codec PCM.

There is no --push flag and no upload command. Pull the small logs/reference
directory and destroy the VAST instance rather than stopping it. Real Metal
execution is a separate remote Apple Silicon gate; never run it on the
maintainer Mac.
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

verify_file() {
  local path="$1" expected_bytes="$2" expected_hash="$3" actual_bytes actual_hash
  [[ -f "$path" && ! -L "$path" ]] || { die "missing, non-regular, or symlinked pinned input: $path"; return 2; }
  actual_bytes="$(wc -c < "$path" | tr -d '[:space:]')"
  [[ "$actual_bytes" == "$expected_bytes" ]] \
    || { die "byte-size mismatch for $path: got $actual_bytes, expected $expected_bytes"; return 2; }
  actual_hash="$(sha256_file "$path")"
  [[ "$actual_hash" == "$expected_hash" ]] \
    || { die "SHA-256 mismatch for $path: got $actual_hash, expected $expected_hash"; return 2; }
  log "identity OK: $path bytes=$actual_bytes sha256=$actual_hash"
}

license_preflight() {
  local approval="$1"
  command -v uv >/dev/null 2>&1 || { die "required tool missing: uv"; return 2; }
  [[ -f "$LICENSE_GATE" && ! -L "$LICENSE_GATE" && -f "$LICENSE_MANIFEST" && ! -L "$LICENSE_MANIFEST" ]] \
    || { die "Bark license gate or tracked manifest is missing"; return 2; }
  [[ -f "$approval" && ! -L "$approval" && -s "$approval" ]] \
    || { die "--approval-evidence must be a nonempty regular approval file"; return 2; }
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python "$LICENSE_GATE" \
    --lock "$PARITY_PROJECT/uv.lock" --project "$PARITY_PROJECT/pyproject.toml" \
    --manifest "$LICENSE_MANIFEST" \
    --small-public-repo "$SMALL_PUBLIC_REPO" --small-upstream-repo "$SMALL_UPSTREAM_REPO" \
    --full-public-repo "$FULL_PUBLIC_REPO" --full-upstream-repo "$FULL_UPSTREAM_REPO" \
    --transformers-version 5.5.0 \
    --small-public-bytes "$SMALL_PUBLIC_BYTES" --small-checkpoint-bytes "$SMALL_CHECKPOINT_BYTES" \
    --small-config-bytes "$SMALL_CONFIG_BYTES" --full-public-bytes "$FULL_PUBLIC_BYTES" \
    --full-checkpoint-bytes "$FULL_CHECKPOINT_BYTES" --full-config-bytes "$FULL_CONFIG_BYTES" \
    --generation-config-bytes "$GENERATION_CONFIG_BYTES" \
    --small-public-revision "$SMALL_PUBLIC_REVISION" \
    --small-upstream-revision "$SMALL_UPSTREAM_REVISION" \
    --full-public-revision "$FULL_PUBLIC_REVISION" \
    --full-upstream-revision "$FULL_UPSTREAM_REVISION" \
    --transformers-source-revision "$TRANSFORMERS_REVISION" \
    --small-public-sha256 "$SMALL_PUBLIC_SHA256" \
    --full-public-sha256 "$FULL_PUBLIC_SHA256" \
    --small-checkpoint-sha256 "$SMALL_CHECKPOINT_SHA256" \
    --full-checkpoint-sha256 "$FULL_CHECKPOINT_SHA256" \
    --small-config-sha256 "$SMALL_CONFIG_SHA256" \
    --full-config-sha256 "$FULL_CONFIG_SHA256" \
    --generation-config-sha256 "$GENERATION_CONFIG_SHA256" \
    --transformers-sdist-sha256 "$TRANSFORMERS_SDIST_SHA256" \
    --transformers-wheel-sha256 "$TRANSFORMERS_WHEEL_SHA256" \
    --approval "$approval"
}

require_disjoint_work_dir() {
  local work="$1" approval="$2" candidate root_real approval_parent approval_real
  candidate="$(canonical_absent_path "$work")" || return 2
  root_real="$(cd -P "$VOKRA_ROOT" 2>/dev/null && pwd)" \
    || { die "Vokra checkout is inaccessible"; return 2; }
  approval_parent="$(cd -P "$(dirname "$approval")" 2>/dev/null && pwd)" \
    || { die "approval parent is inaccessible"; return 2; }
  approval_real="$approval_parent/$(basename "$approval")"
  [[ "$candidate" != "$root_real" && "$candidate/" != "$root_real/"* && "$root_real/" != "$candidate/"* ]] \
    || { die "work-dir overlaps the checkout"; return 2; }
  [[ "$candidate" != "$approval_real" && "$candidate/" != "$approval_real/"* && "$approval_real/" != "$candidate/"* ]] \
    || { die "work-dir overlaps approval evidence"; return 2; }
}

canonical_absent_path() {
  local target="$1" current suffix component real lexical
  [[ "$target" = /* ]] || target="$PWD/$target"
  lexical="${target#/}"; current="/"
  while [[ -n "$lexical" ]]; do
    component="${lexical%%/*}"
    if [[ "$lexical" == "$component" ]]; then lexical=""; else lexical="${lexical#*/}"; fi
    [[ "$component" == "." || -z "$component" ]] && continue
    [[ "$component" != ".." ]] || { die "work-dir path contains '..'"; return 2; }
    current="${current%/}/$component"
    if [[ -L "$current" ]]; then
      real="$(cd -P "$current" 2>/dev/null && pwd)" || { die "work-dir path contains an inaccessible component"; return 2; }
      case "$current:$real" in
        /var:/private/var|/tmp:/private/tmp) current="$real" ;;
        *) die "work-dir path contains a symlinked component"; return 2 ;;
      esac
    fi
  done
  current="$target"
  suffix=""
  while [[ ! -e "$current" && ! -L "$current" ]]; do
    component="$(basename "$current")"
    suffix="/$component$suffix"
    current="$(dirname "$current")"
  done
  [[ -d "$current" && ! -L "$current" ]] || { die "work-dir has an inaccessible or symlinked existing parent"; return 2; }
  real="$(cd -P "$current" 2>/dev/null && pwd)" || { die "work-dir parent is inaccessible"; return 2; }
  printf '%s%s\n' "$real" "$suffix"
}

require_absent_work_dir() {
  local work="$1" approval="$2"
  require_disjoint_work_dir "$work" "$approval" || return 2
  [[ ! -e "$work" && ! -L "$work" ]] \
    || { die "--work-dir must be absent before validation: $work"; return 2; }
}

require_one_test_pass() {
  local log_path="$1" test_name="$2" marker="$3" test_count named_count result_count result_lines marker_count
  test_count="$(grep -Ev '^test result:' "$log_path" | grep -Ec '^test ' || true)"
  named_count="$(grep -Ec "^test ${test_name} \.\.\. ok$" "$log_path" || true)"
  result_count="$(grep -Ec '^test result: ok\. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out(; finished in [0-9]+\.[0-9]+s)?$' "$log_path" || true)"
  result_lines="$(grep -Ec '^test result:' "$log_path" || true)"
  marker_count="$(grep -Ec "^${marker} frames=[0-9]+, codes=exact, decode_max_abs=[0-9.eE+-]+, decode_rmse=[0-9.eE+-]+, end_to_end_max_abs=[0-9.eE+-]+, end_to_end_rmse=[0-9.eE+-]+$" "$log_path" || true)"
  if [[ "$test_count" != 1 || "$named_count" != 1 || "$result_count" != 1 || "$result_lines" != 1 || "$marker_count" != 1 ]]; then
    die "${test_name} evidence must contain exactly one named pass, full result, and metric sentinel"; return 2
  fi
}

download_hf_file() {
  local repository="$1" revision="$2" filename="$3" output="$4" url
  mkdir -p "$(dirname "$output")"
  url="https://huggingface.co/$repository/resolve/$revision/$filename?download=true"
  curl --fail --location --retry 5 --retry-delay 2 --output "$output" "$url"
}

require_vast_host() {
  local mem_kib free_kib
  [[ "${VOKRA_PUBLISH_ON_VAST:-0}" == "1" ]] \
    || die "VOKRA_PUBLISH_ON_VAST=1 is absent; run provision.sh first"
  [[ "$(uname -s)" == "Linux" ]] \
    || die "real Bark model work is VAST/Linux-only; refusing $(uname -s)"
  mem_kib="$(awk '$1 == "MemTotal:" {print $2; exit}' /proc/meminfo)"
  [[ "$mem_kib" =~ ^[0-9]+$ ]] || die "could not read MemTotal"
  if (( mem_kib < MIN_VAST_MEM_KIB )); then
    die "MemTotal=${mem_kib} KiB is below the VAST 24-GB guard"
  fi
  mkdir -p "$VOKRA_SCRATCH"
  free_kib="$(df -Pk "$VOKRA_SCRATCH" | awk 'NR == 2 {print $4}')"
  [[ "$free_kib" =~ ^[0-9]+$ ]] || die "could not read free disk"
  if (( free_kib < MIN_FREE_DISK_KIB )); then
    die "free disk=${free_kib} KiB is below the 40-GB guard"
  fi
}

require_tooling() {
  local tool
  for tool in uv cargo rustc rustup git curl awk find tee wc tr readelf; do
    command -v "$tool" >/dev/null 2>&1 || die "required tool missing: $tool"
  done
  [[ -d "$VOKRA_ROOT/.git" ]] || die "$VOKRA_ROOT is not a git checkout"
  [[ -f "$PARITY_PROJECT/uv.lock" ]] || die "Bark parity uv.lock is missing"
  [[ -f "$PARITY_PROJECT/dump_reference.py" ]] || die "Bark reference dumper is missing"
  if [[ -n "$(git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all)" ]]; then
    die "VAST checkout must be clean so evidence names one exact commit"
  fi
}

record_environment() {
  local output="$1" cpu_model cpu_flags
  cpu_model="$(awk -F ':' '$1 ~ /model name/ {sub(/^[[:space:]]+/, "", $2); print $2; exit}' /proc/cpuinfo)"
  cpu_flags="$(awk -F ':' '$1 ~ /flags/ {sub(/^[[:space:]]+/, "", $2); print $2; exit}' /proc/cpuinfo)"
  {
    echo "utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "git_commit=$(git -C "$VOKRA_ROOT" rev-parse HEAD)"
    echo "git_branch=$(git -C "$VOKRA_ROOT" branch --show-current)"
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

run_self_test() {
  local tmp payload actual script_path cases=0 fail=0 fake_root fake_home fake_log rc test_log audit_removed
  tmp="$(mktemp -d)"
  # shellcheck disable=SC2064
  trap "rm -rf '$tmp'" EXIT
  payload="$tmp/payload"
  printf 'vokra-bark-self-test\n' > "$payload"
  actual="$(sha256_file "$payload")"
  printf '{}\n' > "$tmp/path-approval.json"
  mkdir -p "$tmp/nested-parent"
  require_absent_work_dir "$tmp/nested-parent/model/work" "$tmp/path-approval.json" || { log "self-test FAIL: nested absent work path rejected"; fail=1; }
  mkdir -p "$tmp/intermediate"
  ln -s "$VOKRA_ROOT" "$tmp/intermediate/checkout-link"
  if require_absent_work_dir "$tmp/intermediate/checkout-link/work" "$tmp/path-approval.json" >/dev/null 2>&1; then log "self-test FAIL: intermediate checkout symlink accepted"; fail=1; fi
  mkdir -p "$tmp/real/existing"
  ln -s "$tmp/real" "$tmp/ancestor-link"
  if require_absent_work_dir "$tmp/ancestor-link/existing/nested/new" "$tmp/path-approval.json" >/dev/null 2>&1; then log "self-test FAIL: symlinked ancestor bypass accepted"; fail=1; fi
  ln -s "$tmp/missing-target" "$tmp/dangling-work"
  if require_absent_work_dir "$tmp/dangling-work" "$tmp/path-approval.json" >/dev/null 2>&1; then log "self-test FAIL: dangling work symlink accepted"; fail=1; fi
  if require_absent_work_dir "$VOKRA_ROOT/tools" "$tmp/path-approval.json" >/dev/null 2>&1; then log "self-test FAIL: checkout overlap accepted"; fail=1; fi
  if require_absent_work_dir "$tmp/path-approval.json/child" "$tmp/path-approval.json" >/dev/null 2>&1; then log "self-test FAIL: approval overlap accepted"; fail=1; fi
  mkdir "$tmp/existing-empty"
  if require_absent_work_dir "$tmp/existing-empty" "$tmp/path-approval.json" >/dev/null 2>&1; then log "self-test FAIL: existing empty work directory accepted"; fail=1; fi

  cases=$((cases + 1))
  verify_file "$payload" "$(wc -c < "$payload" | tr -d '[:space:]')" "$actual" \
    >/dev/null 2>&1 || { log "self-test FAIL: valid identity rejected"; fail=1; }
  cases=$((cases + 1))
  if verify_file "$payload" 1 "$actual" >/dev/null 2>&1; then
    log "self-test FAIL: invalid byte size accepted"; fail=1
  fi
  cases=$((cases + 1))
  if verify_file "$payload" "$(wc -c < "$payload" | tr -d '[:space:]')" \
    "$(printf '%064d' 0)" >/dev/null 2>&1; then
    log "self-test FAIL: invalid SHA-256 accepted"; fail=1
  fi
  ln -s "$payload" "$tmp/payload-link"
  cases=$((cases + 1))
  if verify_file "$tmp/payload-link" "$(wc -c < "$payload" | tr -d '[:space:]')" "$actual" >/dev/null 2>&1; then
    log "self-test FAIL: symlinked identity accepted"; fail=1
  fi
  cases=$((cases + 1))
  script_path="${BASH_SOURCE[0]}"
  for required in "$SMALL_PUBLIC_REVISION" "$FULL_PUBLIC_REVISION" \
    "$SMALL_UPSTREAM_REVISION" "$FULL_UPSTREAM_REVISION" \
    "$TRANSFORMERS_REVISION" "$SMALL_PUBLIC_SHA256" "$FULL_PUBLIC_SHA256" \
    "$SMALL_CHECKPOINT_SHA256" "$FULL_CHECKPOINT_SHA256" \
    "bark/dump_reference.py" "license_preflight" "dependency_audit.py" "--no-sync" "--offline" "scripts/verify/apple-silicon-bark.sh" \
    "--approval-evidence" "<APPLE_APPROVAL_EVIDENCE>" "<APPLE_EVIDENCE_DIR>" \
    "real_bark_small_matches_official_transformers" \
    "real_bark_full_matches_official_transformers" \
    "test result: ok. 1 passed; 0 failed; 0 ignored" "parity_bark_real" \
    "load_session_routes_only_named_bark_releases_to_tts" \
    "aarch64-apple-darwin" "--test-threads=1" "--frozen --python 3.12"; do
    if ! grep -Fq -- "$required" "$script_path"; then
      log "self-test FAIL: worker contract lost token: $required"; fail=1
    fi
  done
  for bad_args in \
    "--self-test --approval-evidence x" \
    "--approval-evidence" \
    "--approval-evidence --work-dir x" \
    "--approval-evidence x --approval-evidence y" \
    "--work-dir x --work-dir y" \
    "--unknown x"; do
    # shellcheck disable=SC2086
    if "$script_path" $bad_args >/dev/null 2>&1; then
      log "self-test FAIL: invalid argument accepted: $bad_args"; fail=1
    fi
  done
  # Keep these checks tied to the actual command sites.  A function-definition
  # mention is insufficient: removing or moving the production audit must
  # fail this regression before a real VAST run can acquire model files.
  local gate_call_line sync_call_line audit_call_line download_call_line cargo_call_line
  gate_call_line="$(grep -nF "python \"\$LICENSE_GATE\"" "$script_path" | tail -n 1 | cut -d: -f1)"
  sync_call_line="$(grep -nF 'uv sync --project' "$script_path" | tail -n 1 | cut -d: -f1)"
  audit_call_line="$(grep -nF "python \"\$DEPENDENCY_AUDIT\"" "$script_path" | tail -n 1 | cut -d: -f1)"
  download_call_line="$(grep -nF "download_hf_file \"\$SMALL_PUBLIC_REPO\"" "$script_path" | tail -n 1 | cut -d: -f1)"
  cargo_call_line="$(grep -nF 'cargo test --manifest-path' "$script_path" | tail -n 1 | cut -d: -f1)"
  if [[ -z "$gate_call_line" || -z "$sync_call_line" || -z "$audit_call_line" || -z "$download_call_line" || -z "$cargo_call_line" ]] \
    || ! (( gate_call_line < sync_call_line && sync_call_line < audit_call_line \
      && audit_call_line < download_call_line && download_call_line < cargo_call_line )); then
    log "self-test FAIL: gate/sync/audit/download/Cargo actual-call order drifted"
    fail=1
  fi
  # Delete the exact production audit invocation from a temporary worker and
  # prove the self-test rejects that worker, catching deletion as well as
  # simple command reordering.
  audit_removed="$tmp/run-bark-without-audit.sh"
  sed '/step "Audit the synchronized Python closure without model acquisition"/,+4d' "$script_path" > "$audit_removed"
  chmod +x "$audit_removed"
  if VOKRA_ROOT="$VOKRA_ROOT" "$audit_removed" --self-test >/dev/null 2>&1; then
    log "self-test FAIL: deleting the production audit invocation was accepted"
    fail=1
  fi
  cases=$((cases + 1))
  if grep -En '^[[:space:]]*(python3|python|pip)([[:space:]]|$)' "$script_path" >/dev/null; then
    log "self-test FAIL: direct Python/pip command found"; fail=1
  fi
  cases=$((cases + 1))
  if grep -En -- '^[[:space:]]*(git[[:space:]]+push|.*upload\.sh|.*publish-one\.sh)([[:space:]]|$)' "$script_path" >/dev/null; then
    log "self-test FAIL: publishing command found"; fail=1
  fi
  test_log="$tmp/test.log"
  printf 'test %s ... ok\ntest result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out\nBark SMALL Cpu: frames=1, codes=exact, decode_max_abs=1.0e-9, decode_rmse=1.0e-9, end_to_end_max_abs=1.0e-9, end_to_end_rmse=1.0e-9\n' \
    "$SMALL_TEST" > "$test_log"
  if ! require_one_test_pass "$test_log" "$SMALL_TEST" 'Bark SMALL Cpu:'; then
    log "self-test FAIL: valid singleton evidence rejected"; fail=1
  fi
  printf 'test %s ... ok\ntest %s ... ok\ntest result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out\nBark SMALL Cpu: frames=1, codes=exact, decode_max_abs=1.0e-9, decode_rmse=1.0e-9, end_to_end_max_abs=1.0e-9, end_to_end_rmse=1.0e-9\n' \
    "$SMALL_TEST" "$SMALL_TEST" > "$test_log"
  if require_one_test_pass "$test_log" "$SMALL_TEST" 'Bark SMALL Cpu:'; then
    log "self-test FAIL: duplicate named test accepted"; fail=1
  fi
  printf 'test %s ... ok\ntest result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out\ntest result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out\nBark SMALL Cpu: frames=1, codes=exact, decode_max_abs=1.0e-9, decode_rmse=1.0e-9, end_to_end_max_abs=1.0e-9, end_to_end_rmse=1.0e-9\n' \
    "$SMALL_TEST" > "$test_log"
  if require_one_test_pass "$test_log" "$SMALL_TEST" 'Bark SMALL Cpu:'; then
    log "self-test FAIL: duplicate result accepted"; fail=1
  fi

  # A production-shaped invocation must stop in the dependency gate. The fake
  # uv records calls and exits 2; reaching VAST probing/scratch would fail this
  # assertion, proving ordering without network, sync, or Cargo.
  fake_root="$tmp/root"
  fake_home="$tmp/home"
  fake_log="$tmp/fake-uv.log"
  mkdir -p "$fake_root/tools/parity/bark" "$fake_home/.local/bin"
  cp "$PARITY_PROJECT/license_gate.py" "$PARITY_PROJECT/license_gate_manifest.json" \
    "$PARITY_PROJECT/uv.lock" "$PARITY_PROJECT/pyproject.toml" "$fake_root/tools/parity/bark/"
  cat > "$fake_home/.local/bin/uv" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' "$*" >> "${BARK_SELF_TEST_UV_LOG:?}"
exit 2
EOF
  chmod +x "$fake_home/.local/bin/uv"
  printf '{"invalid":true}\n' > "$tmp/approval.json"
  set +e
  HOME="$fake_home" PATH="$fake_home/.local/bin:$PATH" \
    BARK_SELF_TEST_UV_LOG="$fake_log" VOKRA_ROOT="$fake_root" \
    VOKRA_SCRATCH="$tmp/scratch" "$script_path" \
      --approval-evidence "$tmp/approval.json" --work-dir "$tmp/work" >"$tmp/worker.log" 2>&1
  rc=$?
  set -e
  if [[ $rc -ne 2 || ! -s "$fake_log" || -e "$tmp/scratch" ]]; then
    log "self-test FAIL: production gate did not block before host/scratch"; fail=1
  fi

  rm -rf "$tmp"
  trap - EXIT
  if [[ $fail -eq 0 ]]; then
    echo "run-bark-validation.sh self-test: OK ($cases cases)"
    return 0
  fi
  return 1
}

write_failure_summary_on_exit() {
  local rc=$?
  if [[ -n "${summary_file:-}" && ! -f "$summary_file" ]]; then
    printf "execution_status=FAIL\nexit_code=%s\n" "$rc" > "$summary_file"
  fi
  exit "$rc"
}

main() {
  local self_test=0 requested_work_dir="" approval_evidence="" run_stamp work_dir inputs_dir logs_dir reference_dir
  local small_public small_upstream full_public full_upstream
  local run_log env_log compile_log apple_log cli_log dependency_audit_log small_cpu_log full_cpu_log summary_file
  local seen_work_dir=0 seen_approval=0 seen_self_test=0
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --work-dir)
        (( seen_work_dir == 0 )) || die "duplicate --work-dir"
        [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || { die "--work-dir requires a nonempty value"; return 2; }
        seen_work_dir=1
        requested_work_dir="$2"; shift 2 ;;
      --approval-evidence)
        (( seen_approval == 0 )) || die "duplicate --approval-evidence"
        [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || { die "--approval-evidence requires a nonempty value"; return 2; }
        seen_approval=1
        approval_evidence="$2"; shift 2 ;;
      --self-test) (( seen_self_test == 0 )) || die "duplicate --self-test"; seen_self_test=1; self_test=1; shift ;;
      -h|--help) usage; return 0 ;;
      *) die "unknown argument: $1"; usage; return 2 ;;
    esac
  done
  if [[ $self_test -eq 1 ]]; then
    [[ $seen_approval -eq 0 && $seen_work_dir -eq 0 ]] || { die "--self-test accepts no other arguments"; return 2; }
    run_self_test
    return $?
  fi
  [[ $seen_approval -eq 1 ]] || { usage; die "--approval-evidence is required"; return 2; }

  # This is deliberately the first substantive operation: unresolved package
  # or model approval must stop before host probing, scratch creation, sync,
  # downloads, compilation, or Cargo.
  license_preflight "$approval_evidence"
  run_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  work_dir="${requested_work_dir:-$VOKRA_SCRATCH/bark-validation/$run_stamp}"
  require_absent_work_dir "$work_dir" "$approval_evidence"
  require_vast_host
  require_tooling
  inputs_dir="$work_dir/inputs"
  logs_dir="$work_dir/logs"
  reference_dir="$work_dir/reference"
  small_public="$inputs_dir/public-small/model.gguf"
  small_upstream="$inputs_dir/upstream-small"
  full_public="$inputs_dir/public-full/model.gguf"
  full_upstream="$inputs_dir/upstream-full"
  mkdir -p "$logs_dir" "$small_upstream" "$full_upstream" \
    "$(dirname "$small_public")" "$(dirname "$full_public")" "$reference_dir"
  export UV_CACHE_DIR="$VOKRA_SCRATCH/uv-cache"
  run_log="$logs_dir/run.log"
  env_log="$logs_dir/environment.txt"
  compile_log="$logs_dir/compile.log"
  apple_log="$logs_dir/apple-cross-check.log"
  cli_log="$logs_dir/cli-route.log"
  dependency_audit_log="$logs_dir/dependency-audit.log"
  small_cpu_log="$logs_dir/small-cpu.log"
  full_cpu_log="$logs_dir/full-cpu.log"
  summary_file="$logs_dir/summary.txt"
  exec > >(tee -a "$run_log") 2>&1
  trap write_failure_summary_on_exit EXIT

  step "Sync locked Python 3.12 official-reference environment"
  uv sync --project "$PARITY_PROJECT" --frozen --python 3.12

  step "Audit the synchronized Python closure without model acquisition"
  UV_NO_CACHE=1 uv run --no-cache --project "$PARITY_PROJECT" --frozen --no-sync --python 3.12 python "$DEPENDENCY_AUDIT" \
    --project "$PARITY_PROJECT" \
    --output "$logs_dir/dependency-audit.json" --fetch-model-licenses 2>&1 | tee "$dependency_audit_log"

  step "Download exact public and upstream Bark Small inputs"
  download_hf_file "$SMALL_PUBLIC_REPO" "$SMALL_PUBLIC_REVISION" model.gguf "$small_public"
  download_hf_file "$SMALL_UPSTREAM_REPO" "$SMALL_UPSTREAM_REVISION" pytorch_model.bin "$small_upstream/pytorch_model.bin"
  download_hf_file "$SMALL_UPSTREAM_REPO" "$SMALL_UPSTREAM_REVISION" config.json "$small_upstream/config.json"
  download_hf_file "$SMALL_UPSTREAM_REPO" "$SMALL_UPSTREAM_REVISION" generation_config.json "$small_upstream/generation_config.json"
  verify_file "$small_public" "$SMALL_PUBLIC_BYTES" "$SMALL_PUBLIC_SHA256"
  verify_file "$small_upstream/pytorch_model.bin" "$SMALL_CHECKPOINT_BYTES" "$SMALL_CHECKPOINT_SHA256"
  verify_file "$small_upstream/config.json" "$SMALL_CONFIG_BYTES" "$SMALL_CONFIG_SHA256"
  verify_file "$small_upstream/generation_config.json" "$GENERATION_CONFIG_BYTES" "$GENERATION_CONFIG_SHA256"

  step "Download exact public and upstream Bark Full inputs"
  download_hf_file "$FULL_PUBLIC_REPO" "$FULL_PUBLIC_REVISION" model.gguf "$full_public"
  download_hf_file "$FULL_UPSTREAM_REPO" "$FULL_UPSTREAM_REVISION" pytorch_model.bin "$full_upstream/pytorch_model.bin"
  download_hf_file "$FULL_UPSTREAM_REPO" "$FULL_UPSTREAM_REVISION" config.json "$full_upstream/config.json"
  download_hf_file "$FULL_UPSTREAM_REPO" "$FULL_UPSTREAM_REVISION" generation_config.json "$full_upstream/generation_config.json"
  verify_file "$full_public" "$FULL_PUBLIC_BYTES" "$FULL_PUBLIC_SHA256"
  verify_file "$full_upstream/pytorch_model.bin" "$FULL_CHECKPOINT_BYTES" "$FULL_CHECKPOINT_SHA256"
  verify_file "$full_upstream/config.json" "$FULL_CONFIG_BYTES" "$FULL_CONFIG_SHA256"
  verify_file "$full_upstream/generation_config.json" "$GENERATION_CONFIG_BYTES" "$GENERATION_CONFIG_SHA256"

  step "Record environment before numerical output"
  record_environment "$env_log"

  step "Generate independent official Bark Small reference"
  uv run --project "$PARITY_PROJECT" --frozen --python 3.12 python \
    "$PARITY_PROJECT/dump_reference.py" --variant small \
    --model-dir "$small_upstream" --output "$reference_dir/small"
  cp "$reference_dir/small/manifest.json" "$logs_dir/reference-small-manifest.json"

  step "Generate independent official Bark Full reference"
  uv run --project "$PARITY_PROJECT" --frozen --python 3.12 python \
    "$PARITY_PROJECT/dump_reference.py" --variant full \
    --model-dir "$full_upstream" --output "$reference_dir/full"
  cp "$reference_dir/full/manifest.json" "$logs_dir/reference-full-manifest.json"
  {
    printf '#!/usr/bin/env bash\nset -eu\n'
    printf '%q ' scripts/verify/apple-silicon-bark.sh
    printf '%q ' --small-gguf '<APPLE_BARK_SMALL_GGUF>' --small-reference '<APPLE_BARK_SMALL_REFERENCE>'
    printf '%q ' --small-reference-manifest-sha256 "$(sha256_file "$reference_dir/small/manifest.json")"
    printf '%q ' --full-gguf '<APPLE_BARK_FULL_GGUF>' --full-reference '<APPLE_BARK_FULL_REFERENCE>'
    printf '%q ' --full-reference-manifest-sha256 "$(sha256_file "$reference_dir/full/manifest.json")"
    printf '%q ' --approval-evidence '<APPLE_APPROVAL_EVIDENCE>'
    printf '%q\n' --evidence-dir '<APPLE_EVIDENCE_DIR>'
  } > "$logs_dir/apple-silicon-bark-args.sh"

  step "Compile all workspace targets on VAST"
  cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release \
    --workspace --no-run 2>&1 | tee "$compile_log"

  step "Cross-check the Apple Metal target compiles"
  rustup target add aarch64-apple-darwin
  cargo check --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release \
    -p vokra-models --features metal --target aarch64-apple-darwin \
    2>&1 | tee "$apple_log"

  step "Verify the Bark CLI dispatch route"
  cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release \
    -p vokra-cli engine::tests::load_session_routes_only_named_bark_releases_to_tts \
    -- --exact --nocapture 2>&1 | tee "$cli_log"

  step "Compare native CPU generation/codec with both official references"
  VOKRA_BARK_SMALL_GGUF="$small_public" \
  VOKRA_BARK_SMALL_PARITY_DIR="$reference_dir/small" \
  VOKRA_BARK_BACKEND=cpu \
    cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release \
      -p vokra-models --test parity_bark_real "$SMALL_TEST" \
      -- --exact --nocapture --test-threads=1 2>&1 | tee "$small_cpu_log"
  require_one_test_pass "$small_cpu_log" "$SMALL_TEST" 'Bark SMALL Cpu:'
  VOKRA_BARK_FULL_GGUF="$full_public" \
  VOKRA_BARK_FULL_PARITY_DIR="$reference_dir/full" \
  VOKRA_BARK_BACKEND=cpu \
    cargo test --manifest-path "$VOKRA_ROOT/Cargo.toml" --locked --release \
      -p vokra-models --test parity_bark_real "$FULL_TEST" \
      -- --exact --nocapture --test-threads=1 2>&1 | tee "$full_cpu_log"
  require_one_test_pass "$full_cpu_log" "$FULL_TEST" 'Bark FULL Cpu:'

  {
    echo "execution_status=PASS"
    echo "numeric_verdict=FP32_ATOL_0.01_PASS"
    echo "generated_codes=exact"
    echo "git_commit=$(git -C "$VOKRA_ROOT" rev-parse HEAD)"
    echo "small_public_revision=$SMALL_PUBLIC_REVISION"
    echo "small_public_sha256=$SMALL_PUBLIC_SHA256"
    echo "small_upstream_revision=$SMALL_UPSTREAM_REVISION"
    echo "small_checkpoint_sha256=$SMALL_CHECKPOINT_SHA256"
    echo "full_public_revision=$FULL_PUBLIC_REVISION"
    echo "full_public_sha256=$FULL_PUBLIC_SHA256"
    echo "full_upstream_revision=$FULL_UPSTREAM_REVISION"
    echo "full_checkpoint_sha256=$FULL_CHECKPOINT_SHA256"
    echo "transformers_source_revision=$TRANSFORMERS_REVISION"
    echo "small_reference_manifest_sha256=$(sha256_file "$reference_dir/small/manifest.json")"
    echo "full_reference_manifest_sha256=$(sha256_file "$reference_dir/full/manifest.json")"
    echo "small_cpu_test=$SMALL_TEST"
    echo "full_cpu_test=$FULL_TEST"
    echo "metal_runtime=REQUIRES_REMOTE_APPLE_SILICON"
  } | tee "$summary_file"
  trap - EXIT
  log "PASS: pull $logs_dir and $reference_dir, then destroy the VAST instance"
}

main "$@"
