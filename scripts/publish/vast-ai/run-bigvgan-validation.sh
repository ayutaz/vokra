#!/usr/bin/env bash
# shellcheck disable=SC2016 # literal source tokens are intentional self-test contracts
# VAST/Linux-only BigVGAN conversion and CPU parity worker.
# All checkpoint/source identities are operator-supplied and mandatory because
# this repository does not contain an authenticated release digest. There is
# deliberately no upload, publish, or Git-push path.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VOKRA_ROOT="${VOKRA_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
VOKRA_SCRATCH="${VOKRA_SCRATCH:-$HOME/scratchpad}"
PARITY_PROJECT="$VOKRA_ROOT/tools/parity/bigvgan"
PREPARER="$VOKRA_ROOT/tools/parity/bigvgan_prepare_checkpoint.py"
DUMPER="$VOKRA_ROOT/tools/parity/bigvgan_dump_reference.py"
LICENSE_GATE="$PARITY_PROJECT/license_gate.py"
LINUX_CLOSURE_AUDITOR="$PARITY_PROJECT/audit_linux_closure.py"
LICENSE_MANIFEST="$PARITY_PROJECT/license_gate_manifest.json"
SOURCE_REPOSITORY="https://github.com/NVIDIA/BigVGAN"
MODEL_REPOSITORY="nvidia/bigvgan_base_24khz_100band"
MODEL_VARIANT="base_v1_24khz_100band"
MODEL_KIND="bigvgan-base-24khz-100band"
TEST_NAME="parity_bigvgan_base_real_weight_mel_to_waveform"
EXPECTED_MODEL_REVISION="0f6305d0e010eaafdbf649978f46c3b5af099343"
EXPECTED_CHECKPOINT_SHA256="ca8bced4d3ef588e654742f732455c16abb004e49d7d3bf03edade84d3e982f2"
EXPECTED_CONFIG_SHA256="885553969751bfd87f1980017364e968917cd34347376ed08238db673ea5b46b"
EXPECTED_SOURCE_REVISION="7d2b454564a6c7d014227f635b7423881f14bdac"
MIN_VAST_MEM_KIB=16000000
MIN_FREE_DISK_KIB=20000000
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

log() { printf '[bigvgan-vast] %s\n' "$*" >&2; }
die() { log "ERROR: $*"; return 2; }

usage() {
  cat <<'EOF' >&2
usage: run-bigvgan-validation.sh --approval-evidence <file> [--work-dir <absent-dir>]
       run-bigvgan-validation.sh --self-test

VAST/Linux-only BigVGAN base-v1 conversion and independent CPU parity worker.
The reviewed model/source identities are committed in the parity license
manifest; the matching values remain mandatory environment inputs:
  BIGVGAN_MODEL_REVISION=<40 lowercase hex>
  BIGVGAN_CHECKPOINT_SHA256=<64 lowercase hex>
  BIGVGAN_CONFIG_SHA256=<64 lowercase hex>
  BIGVGAN_SOURCE_REVISION=<40 lowercase hex>
The worker creates a GGUF and reference only on VAST, then runs the focused
vokra-models parity test. Before owner approval, stage exact lock artifacts and
run audit_linux_closure.py to produce a closure candidate; it never creates a
signature. The worker never uploads, publishes, or pushes anything.
EOF
}

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}';
  elif command -v shasum >/dev/null 2>&1; then shasum -a 256 "$1" | awk '{print $1}';
  else die 'neither sha256sum nor shasum is available'; fi
}

require_vast_host() {
  local mem_kib free_kib
  [[ "${VOKRA_PUBLISH_ON_VAST:-0}" == 1 ]] || die 'VOKRA_PUBLISH_ON_VAST=1 is absent'
  [[ "$(uname -s)" == Linux ]] || die 'BigVGAN conversion is VAST/Linux-only'
  [[ "$(uname -m)" == x86_64 ]] || die 'VAST host must be x86_64'
  mem_kib="$(awk '$1 == "MemTotal:" {print $2; exit}' /proc/meminfo)"
  [[ "$mem_kib" =~ ^[0-9]+$ ]] || die 'could not read MemTotal'
  (( mem_kib >= MIN_VAST_MEM_KIB )) || die 'VAST memory guard failed'
  mkdir -p "$VOKRA_SCRATCH"
  free_kib="$(df -Pk "$VOKRA_SCRATCH" | awk 'NR == 2 {print $4}')"
  [[ "$free_kib" =~ ^[0-9]+$ ]] || die 'could not read free disk'
  (( free_kib >= MIN_FREE_DISK_KIB )) || die 'VAST disk guard failed'
}

require_identity_inputs() {
  [[ "${BIGVGAN_MODEL_REVISION:-}" =~ ^[0-9a-f]{40}$ ]] \
    || die 'BIGVGAN_MODEL_REVISION must be a 40-character lowercase revision'
  [[ "${BIGVGAN_CHECKPOINT_SHA256:-}" =~ ^[0-9a-f]{64}$ ]] \
    || die 'BIGVGAN_CHECKPOINT_SHA256 must be a 64-character lowercase digest'
  [[ "${BIGVGAN_CONFIG_SHA256:-}" =~ ^[0-9a-f]{64}$ ]] \
    || die 'BIGVGAN_CONFIG_SHA256 must be a 64-character lowercase digest'
  [[ "${BIGVGAN_SOURCE_REVISION:-}" =~ ^[0-9a-f]{40}$ ]] \
    || die 'BIGVGAN_SOURCE_REVISION must be a 40-character lowercase revision'
  [[ "$BIGVGAN_MODEL_REVISION" == "$EXPECTED_MODEL_REVISION" ]] || die 'BIGVGAN_MODEL_REVISION does not match the reviewed HF revision'
  [[ "$BIGVGAN_CHECKPOINT_SHA256" == "$EXPECTED_CHECKPOINT_SHA256" ]] || die 'BIGVGAN_CHECKPOINT_SHA256 does not match the reviewed LFS payload'
  [[ "$BIGVGAN_CONFIG_SHA256" == "$EXPECTED_CONFIG_SHA256" ]] || die 'BIGVGAN_CONFIG_SHA256 does not match the reviewed config payload'
  [[ "$BIGVGAN_SOURCE_REVISION" == "$EXPECTED_SOURCE_REVISION" ]] || die 'BIGVGAN_SOURCE_REVISION does not match the reviewed source revision'
}

require_tooling() {
  local tool
  for tool in uv cargo rustc git curl awk find tee wc tr grep tar; do
    command -v "$tool" >/dev/null 2>&1 || die "required tool missing: $tool"
  done
  [[ -d "$VOKRA_ROOT/.git" ]] || die 'not a Vokra checkout'
  [[ -f "$PARITY_PROJECT/pyproject.toml" && -f "$PARITY_PROJECT/uv.lock" ]] || die 'BigVGAN lock is missing'
  [[ -f "$PREPARER" && -f "$DUMPER" && -f "$LICENSE_GATE" && -f "$LICENSE_MANIFEST" && -f "$LINUX_CLOSURE_AUDITOR" ]] || die 'BigVGAN parity tools or license gate are missing'
  [[ -z "$(git -C "$VOKRA_ROOT" status --porcelain --untracked-files=all)" ]] || die 'VAST checkout must be clean'
}

license_gate() {
  local evidence="$1"
  [[ -f "$LICENSE_GATE" && ! -L "$LICENSE_GATE" && -f "$LICENSE_MANIFEST" && ! -L "$LICENSE_MANIFEST" ]] \
    || { die 'BigVGAN license gate or manifest is missing or symlinked'; return 2; }
  [[ -f "$evidence" && ! -L "$evidence" && -s "$evidence" ]] \
    || { die '--approval-evidence must be a nonempty regular file'; return 2; }
  UV_NO_CACHE=1 uv run --no-cache --no-project --offline --python 3.12 python "$LICENSE_GATE" \
    --lock "$PARITY_PROJECT/uv.lock" --project "$PARITY_PROJECT/pyproject.toml" --manifest "$LICENSE_MANIFEST" \
    --source-revision "${BIGVGAN_SOURCE_REVISION:-}" --model-revision "${BIGVGAN_MODEL_REVISION:-}" \
    --checkpoint-sha256 "${BIGVGAN_CHECKPOINT_SHA256:-}" --config-sha256 "${BIGVGAN_CONFIG_SHA256:-}" \
    --license-evidence "$evidence"
}

require_disjoint_work_dir() {
  local work="$1" approval="$2" candidate root_real approval_parent approval_real
  candidate="$(canonical_absent_path "$work")" || return 2
  root_real="$(cd -P "$VOKRA_ROOT" 2>/dev/null && pwd)" \
    || { die 'Vokra checkout is inaccessible'; return 2; }
  approval_parent="$(cd -P "$(dirname "$approval")" 2>/dev/null && pwd)" \
    || { die 'approval parent is inaccessible'; return 2; }
  approval_real="$approval_parent/$(basename "$approval")"
  [[ "$candidate" != "$root_real" && "$candidate/" != "$root_real/"* && "$root_real/" != "$candidate/"* ]] \
    || { die 'work-dir overlaps the checkout'; return 2; }
  [[ "$candidate" != "$approval_real" && "$candidate/" != "$approval_real/"* && "$approval_real/" != "$candidate/"* ]] \
    || { die 'work-dir overlaps approval evidence'; return 2; }
}

canonical_absent_path() {
  local target="$1" current suffix component real lexical
  [[ "$target" = /* ]] || target="$PWD/$target"
  lexical="${target#/}"; current="/"
  while [[ -n "$lexical" ]]; do
    component="${lexical%%/*}"
    if [[ "$lexical" == "$component" ]]; then lexical=""; else lexical="${lexical#*/}"; fi
    [[ "$component" == "." || -z "$component" ]] && continue
    [[ "$component" != ".." ]] || { die 'work-dir path contains ..'; return 2; }
    current="${current%/}/$component"
    if [[ -L "$current" ]]; then
      real="$(cd -P "$current" 2>/dev/null && pwd)" || { die 'work-dir path contains an inaccessible component'; return 2; }
      case "$current:$real" in
        /var:/private/var|/tmp:/private/tmp) current="$real" ;;
        *) die 'work-dir path contains a symlinked component'; return 2 ;;
      esac
    fi
  done
  current="$target"; suffix=""
  while [[ ! -e "$current" && ! -L "$current" ]]; do
    component="$(basename "$current")"; suffix="/$component$suffix"; current="$(dirname "$current")"
  done
  [[ -d "$current" && ! -L "$current" ]] || { die 'work-dir has an inaccessible or symlinked existing parent'; return 2; }
  real="$(cd -P "$current" 2>/dev/null && pwd)" || { die 'work-dir parent is inaccessible'; return 2; }
  printf '%s%s\n' "$real" "$suffix"
}

require_absent_work_dir() {
  local work="$1" approval="$2"
  require_disjoint_work_dir "$work" "$approval" || return 2
  [[ ! -e "$work" && ! -L "$work" ]] || { die '--work-dir must be absent before validation'; return 2; }
}

verify_downloaded_file() {
  local path="$1" expected_hash="$2" actual_hash
  [[ -f "$path" && ! -L "$path" && -s "$path" ]] \
    || { die "downloaded input is missing, empty, non-regular, or symlinked: $path"; return 2; }
  actual_hash="$(sha256_file "$path")"
  [[ "$actual_hash" == "$expected_hash" ]] \
    || { die "downloaded input SHA-256 mismatch for $path: got $actual_hash, expected $expected_hash"; return 2; }
}

download_file() {
  local revision="$1" filename="$2" output="$3"
  mkdir -p "$(dirname "$output")"
  curl --fail --location --retry 5 --retry-delay 2 --silent --show-error \
    "https://huggingface.co/$MODEL_REPOSITORY/resolve/$revision/$filename?download=true" \
    --output "$output"
}

checkout_source() {
  local output="$1"
  git clone --filter=blob:none --no-checkout "$SOURCE_REPOSITORY" "$output"
  git -C "$output" checkout --detach "$BIGVGAN_SOURCE_REVISION"
  [[ "$(git -C "$output" rev-parse HEAD)" == "$BIGVGAN_SOURCE_REVISION" ]] || die 'source revision mismatch'
  [[ "$(git -C "$output" remote get-url origin | sed 's/\.git$//')" == "$SOURCE_REPOSITORY" ]] || die 'source origin mismatch'
  [[ -z "$(git -C "$output" status --porcelain --untracked-files=all)" ]] || die 'source checkout is dirty'
}

run_self_test() {
  # Self-test tokens intentionally remain literal source contracts.
  # shellcheck disable=SC2016
  local path="${BASH_SOURCE[0]}" fail=0 token self_project="$PARITY_PROJECT" worker_block
  # A maintainer Mac may not have all Linux lock wheels cached. The isolated
  # safe-load self-test can use the repository parity environment there; the
  # real worker always syncs the dedicated VAST lock below.
  [[ "$(uname -s)" == Linux ]] || self_project="$VOKRA_ROOT/tools/parity"
  for token in 'VOKRA_PUBLISH_ON_VAST=1' 'Linux' 'x86_64' 'BIGVGAN_MODEL_REVISION' 'BIGVGAN_CHECKPOINT_SHA256' 'BIGVGAN_CONFIG_SHA256' \
    'BIGVGAN_SOURCE_REVISION' 'nvidia/bigvgan_base_24khz_100band' \
    'https://github.com/NVIDIA/BigVGAN' 'bigvgan_prepare_checkpoint.py' \
    'bigvgan_dump_reference.py' 'weights_only=True' 'cargo test --locked --release' \
    'parity_bigvgan_real' 'NO_UPLOAD' 'uv sync --project' '--frozen --python 3.12' 'license_gate.py' \
    'audit_linux_closure.py' 'OWNER_REVIEW_REQUIRED' 'bigvgan-evidence.tar.gz' 'archive_sha256=' \
    'license_gate_manifest.json' '--no-project --offline --python 3.12' '--approval-evidence' \
    '<APPLE_APPROVAL_EVIDENCE>' '<APPLE_EVIDENCE_DIR>'; do
    grep -Fq -- "$token" "$path" || { log "self-test FAIL: missing token: $token"; fail=1; }
  done
  worker_block="$(awk '/^  VOKRA_BIGVGAN_BASE_GGUF=.*VOKRA_BIGVGAN_REFERENCE=/{seen=1} seen {print} seen && /grep -Fq .BIGVGAN_CPU_PARITY_SENTINEL/ {exit}' "$path")"
  for token in 'VOKRA_BIGVGAN_REFERENCE="$reference"' 'cargo test --locked --release' '"$TEST_NAME" --exact --nocapture' \
    'tee -a "$log_file"' 'require_vast_test_pass "$log_file"'; do
    grep -Fq -- "$token" <<<"$worker_block" || { log "self-test FAIL: real test command contract: $token"; fail=1; }
  done
  grep -Fq 'BIGVGAN_CPU_PARITY_METRICS' "$path" || { log 'self-test FAIL: CPU metrics contract missing'; fail=1; }
  if grep -En '(^|[[:space:]])(git[[:space:]]+push|.*upload\.sh|.*publish-one\.sh|--push|--upload)([[:space:]]|$)' "$path" >/dev/null; then
    log 'self-test FAIL: publication command found'; fail=1
  fi
  if "$path" --self-test --work-dir /tmp/not-accepted >/dev/null 2>&1; then
    log 'self-test FAIL: extra self-test argument accepted'; fail=1
  fi
  for bad_args in \
    '--self-test --approval-evidence x' \
    '--approval-evidence' \
    '--approval-evidence --work-dir x' \
    '--approval-evidence x --approval-evidence y' \
    '--work-dir x --work-dir y' \
    '--unknown x'; do
    # shellcheck disable=SC2086
    if "$path" $bad_args >/dev/null 2>&1; then
      log "self-test FAIL: invalid argument accepted: $bad_args"; fail=1
    fi
  done
  if "$path" --unknown-flag >/dev/null 2>&1; then log 'self-test FAIL: unknown flag accepted'; fail=1; fi
  UV_CACHE_DIR="${BIGVGAN_UV_CACHE_DIR:-/tmp/vokra-bigvgan-uv-cache}" \
    uv run --no-project --offline --python 3.12 python "$LICENSE_GATE" --self-test \
    >/dev/null || { log 'self-test FAIL: license gate self-test'; fail=1; }
  local gate_line sync_line
  gate_line="$(grep -n 'python \"\$LICENSE_GATE\"' "$path" | head -n 1 | cut -d: -f1)"
  sync_line="$(grep -n 'uv sync --project' "$path" | tail -n 1 | cut -d: -f1)"
  [[ "$gate_line" =~ ^[0-9]+$ && "$sync_line" =~ ^[0-9]+$ && "$gate_line" -lt "$sync_line" ]] \
    || { log 'self-test FAIL: license gate is not before sync'; fail=1; }
  local sandbox trace worker_log real_uv fake_bin fake_uv fake_curl fake_cargo rc
  sandbox="$(mktemp -d "${TMPDIR:-/tmp}/bigvgan-worker-selftest.XXXXXX")"
  trace="$(mktemp "${TMPDIR:-/tmp}/bigvgan-worker-trace.XXXXXX")"
  worker_log="$(mktemp "${TMPDIR:-/tmp}/bigvgan-worker-log.XXXXXX")"
  real_uv="$(command -v uv)"; fake_bin="$sandbox/bin"
  mkdir -p "$fake_bin" "$sandbox/tools/parity/bigvgan" "$sandbox/scripts/publish/vast-ai"
  printf '{}\n' > "$sandbox/path-approval.json"
  require_absent_work_dir "$sandbox/nested-parent/model/work" "$sandbox/path-approval.json" || { log 'self-test FAIL: nested absent work path rejected'; fail=1; }
  mkdir -p "$sandbox/intermediate"
  ln -s "$VOKRA_ROOT" "$sandbox/intermediate/checkout-link"
  if require_absent_work_dir "$sandbox/intermediate/checkout-link/work" "$sandbox/path-approval.json" >/dev/null 2>&1; then log 'self-test FAIL: intermediate checkout symlink accepted'; fail=1; fi
  mkdir -p "$sandbox/real/existing"
  ln -s "$sandbox/real" "$sandbox/ancestor-link"
  if require_absent_work_dir "$sandbox/ancestor-link/existing/nested/new" "$sandbox/path-approval.json" >/dev/null 2>&1; then log 'self-test FAIL: symlinked ancestor bypass accepted'; fail=1; fi
  ln -s "$sandbox/missing-target" "$sandbox/dangling-work"
  if require_absent_work_dir "$sandbox/dangling-work" "$sandbox/path-approval.json" >/dev/null 2>&1; then log 'self-test FAIL: dangling work symlink accepted'; fail=1; fi
  if require_absent_work_dir "$VOKRA_ROOT/tools" "$sandbox/path-approval.json" >/dev/null 2>&1; then log 'self-test FAIL: checkout overlap accepted'; fail=1; fi
  if require_absent_work_dir "$sandbox/path-approval.json/child" "$sandbox/path-approval.json" >/dev/null 2>&1; then log 'self-test FAIL: approval overlap accepted'; fail=1; fi
  mkdir "$sandbox/existing-empty"
  if require_absent_work_dir "$sandbox/existing-empty" "$sandbox/path-approval.json" >/dev/null 2>&1; then log 'self-test FAIL: existing empty work directory accepted'; fail=1; fi
  cp "$path" "$sandbox/scripts/publish/vast-ai/run-bigvgan-validation.sh"
  cp "$LICENSE_GATE" "$sandbox/tools/parity/bigvgan/license_gate.py"
  cp "$LICENSE_MANIFEST" "$sandbox/tools/parity/bigvgan/license_gate_manifest.json"
  cp "$PARITY_PROJECT/pyproject.toml" "$sandbox/tools/parity/bigvgan/pyproject.toml"
  cp "$PARITY_PROJECT/uv.lock" "$sandbox/tools/parity/bigvgan/uv.lock"
  : > "$sandbox/Cargo.toml"
  : > "$sandbox/tools/parity/bigvgan_prepare_checkpoint.py"
  : > "$sandbox/tools/parity/bigvgan_dump_reference.py"
  git -C "$sandbox" init -q
  git -C "$sandbox" config user.email bigvgan-selftest@example.invalid
  git -C "$sandbox" config user.name bigvgan-selftest
  fake_uv="$fake_bin/uv"; fake_curl="$fake_bin/curl"; fake_cargo="$fake_bin/cargo"
  printf '%s\n' '#!/usr/bin/env bash' 'printf "uv %s\n" "$*" >> "$BIGVGAN_SELFTEST_TRACE"' 'if [[ " $* " == *" --no-project "* ]]; then exec "$BIGVGAN_SELFTEST_REAL_UV" "$@"; fi' 'exit 97' > "$fake_uv"
  printf '%s\n' '#!/usr/bin/env bash' 'printf "curl %s\n" "$*" >> "$BIGVGAN_SELFTEST_TRACE"' 'exit 97' > "$fake_curl"
  printf '%s\n' '#!/usr/bin/env bash' 'printf "cargo %s\n" "$*" >> "$BIGVGAN_SELFTEST_TRACE"' 'exit 97' > "$fake_cargo"
  chmod +x "$fake_uv" "$fake_curl" "$fake_cargo"
  git -C "$sandbox" add .
  git -C "$sandbox" -c user.email=bigvgan-selftest@example.invalid -c user.name=bigvgan-selftest commit -qm self-test
  printf '{"invalid":true}\n' > "$sandbox/approval.json"
  set +e
  PATH="$fake_bin:$PATH" HOME="$sandbox/home" VOKRA_ROOT="$sandbox" VOKRA_SCRATCH="$sandbox/scratch" \
    BIGVGAN_SELFTEST_TRACE="$trace" BIGVGAN_SELFTEST_REAL_UV="$real_uv" \
    bash "$sandbox/scripts/publish/vast-ai/run-bigvgan-validation.sh" \
      --approval-evidence "$sandbox/approval.json" --work-dir "$sandbox/work" >"$worker_log" 2>&1
  rc=$?
  set -e
  [[ "$rc" -eq 2 ]] || { log "self-test FAIL: real worker gate returned $rc, expected 2"; fail=1; }
  if [[ ! -f "$trace" ]]; then
    log "self-test FAIL: no-project gate was not invoked: $(sed -n l "$worker_log")"
    fail=1
  else
    grep -Fq 'uv run --no-cache --no-project --offline' "$trace" || { log 'self-test FAIL: no-cache no-project gate was not invoked'; fail=1; }
  fi
  if grep -Eq 'uv sync|curl |cargo ' "$trace"; then
    log 'self-test FAIL: sync/download/build was reached after the blocking gate'; fail=1
  fi
  [[ ! -d "$sandbox/scratch" ]] || { log 'self-test FAIL: worker created scratch output before the gate'; fail=1; }
  printf 'bigvgan-self-test\n' > "$sandbox/downloaded"
  local downloaded_sha downloaded_link
  downloaded_sha="$(sha256_file "$sandbox/downloaded")"; downloaded_link="$sandbox/downloaded-link"
  ln -s downloaded "$downloaded_link"
  if verify_downloaded_file "$downloaded_link" "$downloaded_sha" >/dev/null 2>&1; then
    log 'self-test FAIL: symlinked downloaded identity accepted'; fail=1
  fi
  rm -rf -- "$sandbox" "$trace" "$worker_log"
  UV_CACHE_DIR="${BIGVGAN_UV_CACHE_DIR:-/tmp/vokra-bigvgan-uv-cache}" \
    uv run --project "$self_project" --frozen --python 3.12 python "$PREPARER" --self-test \
    >/dev/null || { log 'self-test FAIL: safe preparer self-test'; fail=1; }
  UV_CACHE_DIR="${BIGVGAN_UV_CACHE_DIR:-/tmp/vokra-bigvgan-uv-cache}" \
    uv run --project "$self_project" --frozen --python 3.12 python "$DUMPER" --self-test \
    >/dev/null || { log 'self-test FAIL: safe dumper self-test'; fail=1; }
  UV_CACHE_DIR="${BIGVGAN_UV_CACHE_DIR:-/tmp/vokra-bigvgan-uv-cache}" \
    uv run --project "$self_project" --frozen --python 3.12 python "$LINUX_CLOSURE_AUDITOR" --self-test \
    >/dev/null || { log 'self-test FAIL: Linux closure auditor self-test'; fail=1; }
  local parity_log
  parity_log="$(mktemp "${TMPDIR:-/tmp}/bigvgan-parity-selftest.XXXXXX")"
  printf '%s\n' \
    "test $TEST_NAME ... ok" \
    'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out' \
    'BIGVGAN_CPU_PARITY_METRICS samples=256 max_abs=0.000010000 atol=0.000020000 reference=NVIDIA.BigVGAN fixture=vast_generated_official' \
    'BIGVGAN_CPU_PARITY_SENTINEL samples=256 max_abs=0.000010000 atol=0.000020000 reference=NVIDIA.BigVGAN fixture=vast_generated_official' \
    > "$parity_log"
  require_vast_test_pass "$parity_log" || { log 'self-test FAIL: valid CPU parity evidence rejected'; fail=1; }
  printf '%s\n' \
    "test $TEST_NAME ... ok" \
    'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out' \
    'BIGVGAN_CPU_PARITY_METRICS samples=256 max_abs=0.000020001 atol=0.000020000 reference=NVIDIA.BigVGAN fixture=vast_generated_official' \
    'BIGVGAN_CPU_PARITY_SENTINEL samples=256 max_abs=0.000020001 atol=0.000020000 reference=NVIDIA.BigVGAN fixture=vast_generated_official' \
    > "$parity_log"
  if require_vast_test_pass "$parity_log" >/dev/null 2>&1; then
    log 'self-test FAIL: over-bound CPU parity evidence accepted'; fail=1
  fi
  rm -f -- "$parity_log"
  (( fail == 0 )) || return 1
  echo 'run-bigvgan-validation.sh self-test: OK'
}

require_vast_test_pass() {
  local output="$1" test_count named_count result_count result_lines metric_count sentinel_count
  test_count="$(grep -Ev '^test result:' "$output" | grep -Ec '^test ' || true)"
  named_count="$(grep -Ec "^test ${TEST_NAME//./\\.} \.\.\. ok$" "$output" || true)"
  result_count="$(grep -Ec '^test result: ok\. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out(; finished in [0-9]+\.[0-9]+s)?$' "$output" || true)"
  result_lines="$(grep -Ec '^test result:' "$output" || true)"
  metric_count="$(grep -Ec '^BIGVGAN_CPU_PARITY_METRICS samples=256 max_abs=[0-9]+\.[0-9]{9} atol=0\.000020000 reference=NVIDIA\.BigVGAN fixture=vast_generated_official$' "$output" || true)"
  sentinel_count="$(grep -Ec '^BIGVGAN_CPU_PARITY_SENTINEL samples=256 max_abs=[0-9]+\.[0-9]{9} atol=0\.000020000 reference=NVIDIA\.BigVGAN fixture=vast_generated_official$' "$output" || true)"
  [[ "$test_count" == 1 && "$named_count" == 1 && "$result_count" == 1 && "$result_lines" == 1 && "$metric_count" == 1 && "$sentinel_count" == 1 ]] \
    || die 'focused CPU evidence must contain one exact test/result/sentinel'
  awk '/^BIGVGAN_CPU_PARITY_METRICS / { for (i = 1; i <= NF; i++) { split($i, pair, "="); if (pair[1] == "max_abs" && (pair[2] + 0) > 0.00002) exit 1 } }' "$output" \
    || die 'focused CPU metric exceeds registered 0.000020000 bound'
}

main() {
  local work_dir='' approval_evidence='' self_test=0 stamp inputs source checkpoint config prepared reference gguf log_file archive archive_sha
  local seen_work_dir=0 seen_approval=0 seen_self_test=0
  while (( $# > 0 )); do
    case "$1" in
      --work-dir) (( seen_work_dir == 0 )) || die 'duplicate --work-dir'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die '--work-dir requires a nonempty value'; seen_work_dir=1; work_dir="$2"; shift 2 ;;
      --approval-evidence) (( seen_approval == 0 )) || die 'duplicate --approval-evidence'; [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die '--approval-evidence requires a nonempty value'; seen_approval=1; approval_evidence="$2"; shift 2 ;;
      --self-test) (( seen_self_test == 0 )) || die 'duplicate --self-test'; seen_self_test=1; self_test=1; shift ;;
      -h|--help) usage; return 0 ;;
      *) usage; die "unknown argument: $1" ;;
    esac
  done
  if (( self_test )); then [[ -z "$work_dir" && -z "$approval_evidence" ]] || die '--self-test accepts no other arguments'; run_self_test; return $?; fi
  (( seen_approval == 1 )) || { usage; die '--approval-evidence is required'; return 2; }
  license_gate "$approval_evidence"
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  work_dir="${work_dir:-$VOKRA_SCRATCH/bigvgan-validation/$stamp}"
  require_absent_work_dir "$work_dir" "$approval_evidence"
  require_tooling; require_vast_host; require_identity_inputs
  mkdir -p "$work_dir/inputs" "$work_dir/source" "$work_dir/prepared" "$work_dir/reference" "$work_dir/logs"
  work_dir="$(cd "$work_dir" && pwd)"
  inputs="$work_dir/inputs"; source="$work_dir/source/BigVGAN"; checkpoint="$inputs/bigvgan_generator.pt"; config="$inputs/config.json"
  prepared="$work_dir/prepared/bigvgan.safetensors"; reference="$work_dir/reference/base_reference.csv"; gguf="$work_dir/bigvgan-base-24khz-100band.gguf"; log_file="$work_dir/logs/validation.log"
  exec > >(tee -a "$log_file") 2>&1
  # shellcheck disable=SC2154
  # rc is assigned by the EXIT trap itself.
  trap 'rc=$?; echo "execution_status=${rc}" > "$work_dir/logs/summary.txt"; exit "$rc"' EXIT
  printf '%s\n' \
    "uv run --no-project --offline --python 3.12 python $LINUX_CLOSURE_AUDITOR --lock $PARITY_PROJECT/uv.lock --artifacts-dir <STAGED_LOCK_ARTIFACTS> --output $work_dir/logs/linux-closure-candidate.json" \
    > "$work_dir/logs/linux-closure-candidate-command.txt"
  uv sync --project "$PARITY_PROJECT" --frozen --python 3.12
  download_file "$BIGVGAN_MODEL_REVISION" bigvgan_generator.pt "$checkpoint"
  download_file "$BIGVGAN_MODEL_REVISION" config.json "$config"
  verify_downloaded_file "$checkpoint" "$BIGVGAN_CHECKPOINT_SHA256"
  verify_downloaded_file "$config" "$BIGVGAN_CONFIG_SHA256"
  checkout_source "$source"
  uv run --project "$PARITY_PROJECT" --frozen --python 3.12 python "$PREPARER" \
    --pt "$checkpoint" --config "$config" --variant "$MODEL_VARIANT" \
    --source-dir "$source" --source-revision "$BIGVGAN_SOURCE_REVISION" \
    --checkpoint-sha256 "$BIGVGAN_CHECKPOINT_SHA256" --config-sha256 "$BIGVGAN_CONFIG_SHA256" \
    --output "$prepared" --config-out "$work_dir/prepared/config.json"
  uv run --project "$PARITY_PROJECT" --frozen --python 3.12 python "$DUMPER" \
    --upstream-dir "$source" --checkpoint "$checkpoint" --checkpoint-sha256 "$BIGVGAN_CHECKPOINT_SHA256" \
    --config-sha256 "$BIGVGAN_CONFIG_SHA256" --source-revision "$BIGVGAN_SOURCE_REVISION" \
    --config "$config" --output "$reference"
  {
    printf '#!/usr/bin/env bash\nset -eu\n'
    printf '%q ' scripts/verify/apple-silicon-bigvgan.sh
    printf '%q ' --gguf '<APPLE_BIGVGAN_GGUF>' --gguf-sha256 "$(sha256_file "$gguf")"
    printf '%q ' --reference '<APPLE_BIGVGAN_REFERENCE>' --reference-sha256 "$(sha256_file "$reference")"
    printf '%q ' --model-revision "$BIGVGAN_MODEL_REVISION" --checkpoint-sha256 "$BIGVGAN_CHECKPOINT_SHA256"
    printf '%q ' --config-sha256 "$BIGVGAN_CONFIG_SHA256" --source-revision "$BIGVGAN_SOURCE_REVISION"
    printf '%q ' --approval-evidence '<APPLE_APPROVAL_EVIDENCE>'
    printf '%q\n' --evidence-dir '<APPLE_EVIDENCE_DIR>'
  } > "$work_dir/logs/apple-silicon-bigvgan-args.sh"
  CARGO_BUILD_JOBS=1 cargo run --locked --release -p vokra-cli -- convert \
    --model "$MODEL_KIND" --input "$prepared" --output "$gguf"
  VOKRA_BIGVGAN_BASE_GGUF="$gguf" VOKRA_BIGVGAN_REFERENCE="$reference" CARGO_BUILD_JOBS=1 \
    cargo test --locked --release -p vokra-models --test parity_bigvgan_real -- \
    "$TEST_NAME" --exact --nocapture 2>&1 | tee -a "$log_file"
  require_vast_test_pass "$log_file"
  printf 'execution_status=PASS\nmodel_repository=%s\nmodel_revision=%s\ncheckpoint_sha256=%s\nconfig_sha256=%s\nsource_repository=%s\nsource_revision=%s\ngguf_sha256=%s\nreference_sha256=%s\nregistered_cpu_atol=0.000020000\npublication=NO_UPLOAD\n' \
    "$MODEL_REPOSITORY" "$BIGVGAN_MODEL_REVISION" "$BIGVGAN_CHECKPOINT_SHA256" "$BIGVGAN_CONFIG_SHA256" \
    "$SOURCE_REPOSITORY" "$BIGVGAN_SOURCE_REVISION" "$(sha256_file "$gguf")" "$(sha256_file "$reference")" > "$work_dir/logs/summary.txt"
  archive="$work_dir/bigvgan-evidence.tar.gz"
  tar -czf "$archive" -C "$work_dir" logs reference
  archive_sha="$(sha256_file "$archive")"
  printf 'archive=%s\narchive_sha256=%s\npublication=NO_UPLOAD\n' "$archive" "$archive_sha" >> "$work_dir/logs/summary.txt"
  trap - EXIT
  log "PASS: pull $work_dir/logs and $reference, then destroy the VAST instance"
}

main "$@"
