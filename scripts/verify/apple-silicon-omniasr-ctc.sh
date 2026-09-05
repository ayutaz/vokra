#!/usr/bin/env bash
# Apple Silicon consumer of an authenticated VAST OmniASR-CTC packet.
# It never downloads, converts, or substitutes a reference.
set -euo pipefail

ROOT="${VOKRA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PARITY_TEST="real_omniasr_ctc_encoder_logits_and_tokens_match_official"
die() { echo "[omniasr-ctc-apple] ERROR: $*" >&2; exit 2; }

validate_expected_commit() {
  local expected="$1" actual="$2"
  [[ "$expected" =~ ^[0-9a-f]{40}$ && "$actual" == "$expected" ]]
}

validate_clean_checkout() {
  [[ -z "$1" ]]
}

structured_unsupported_log() {
  local log="$1"
  [[ "$(printf '%s\n' "$log" | grep -Fxc 'OMNIASR_UNSUPPORTED_OP')" == 1 ]] || return 1
  [[ "$(printf '%s\n' "$log" | grep -Ec "^test ${PARITY_TEST} \.\.\. FAILED$")" == 1 ]] || return 1
  [[ "$(printf '%s\n' "$log" | grep -Fxc 'OMNIASR_REAL_PARITY_PASS')" == 0 ]] || return 1
}

backend_sentinel_log() {
  local log="$1" expected="$2" opposite="Cpu"
  [[ "$expected" == Cpu ]] && opposite=Metal
  [[ "$(printf '%s\n' "$log" | grep -Ec "^OmniASR ${expected}: frames=[0-9]+, frontend_max_abs=[^,]+, encoder_max_abs=[^,]+, logits_max_abs=[^,]+, tokens=exact$")" == 1 ]] || return 1
  [[ "$(printf '%s\n' "$log" | grep -Ec "^OmniASR ${opposite}:")" == 0 ]]
}

self_test() {
  local fail=0 required unsupported_log contaminated_log cpu_log metal_log valid_commit
  for required in Darwin arm64 VOKRA_REMOTE_APPLE_SILICON \
    VOKRA_EXPECTED_COMMIT environment.txt input-sha256-before.txt \
    head_commit sysctl_hw.model sysctl_hw.machine sysctl_hw.memsize \
    rustc cargo xcrun_metal_path xcrun_metal_version \
    assert_bundle_unchanged bundle_symlink 'type l' BLOCKED_UNSUPPORTED 'exit 3' \
    validation-summary.txt CPU_PARITY_COMPLETE NO_UPLOAD REFERENCE_COMPLETE \
    ctc_logits.f32le encoder.f32le tokens.u32le \
    VOKRA_OMNIASR_BACKEND metal OMNIASR_UNSUPPORTED_OP "$PARITY_TEST" \
    reference_manifest_sha256 frontend_atol OMNIASR_REAL_PARITY_PASS; do
    grep -Fq -- "$required" "$0" || { echo "self-test missing $required" >&2; fail=1; }
  done
  if grep -En '(^|[;&|][[:space:]]*)(curl|wget|snapshot_download|git[[:space:]]+push|publish-one\.sh|upload\.sh)([[:space:]]|$)' "$0" >/dev/null; then
    echo "self-test found download/publication command" >&2
    fail=1
  fi
  [[ "$(grep -Fc 'assert_bundle_unchanged' "$0")" -ge 3 ]] || {
    echo "self-test does not check bundle immutability after both cargo invocations" >&2
    fail=1
  }
  valid_commit=0123456789abcdef0123456789abcdef01234567
  validate_expected_commit "$valid_commit" "$valid_commit" || {
    echo "self-test rejected valid expected commit" >&2
    fail=1
  }
  validate_expected_commit "0123456789ABCDEF0123456789ABCDEF01234567" "$valid_commit" && {
    echo "self-test accepted uppercase expected commit" >&2
    fail=1
  } || true
  validate_expected_commit "$valid_commit" "0123456789abcdef0123456789abcdef01234568" && {
    echo "self-test accepted mismatched HEAD" >&2
    fail=1
  } || true
  validate_clean_checkout "" || {
    echo "self-test rejected clean checkout" >&2
    fail=1
  }
  validate_clean_checkout " M unrelated-file" && {
    echo "self-test accepted dirty checkout" >&2
    fail=1
  } || true
  cpu_log='OmniASR Cpu: frames=123, frontend_max_abs=1.0e-3, encoder_max_abs=2.0e-3, logits_max_abs=3.0e-3, tokens=exact'
  metal_log='OmniASR Metal: frames=123, frontend_max_abs=1.0e-3, encoder_max_abs=2.0e-3, logits_max_abs=3.0e-3, tokens=exact'
  backend_sentinel_log "$cpu_log" Cpu || {
    echo "self-test rejected exact CPU backend sentinel" >&2
    fail=1
  }
  backend_sentinel_log "$metal_log" Metal || {
    echo "self-test rejected exact Metal backend sentinel" >&2
    fail=1
  }
  backend_sentinel_log "$cpu_log" Metal && {
    echo "self-test accepted CPU log for Metal request" >&2
    fail=1
  } || true
  unsupported_log=$'OMNIASR_UNSUPPORTED_OP\ntest real_omniasr_ctc_encoder_logits_and_tokens_match_official ... FAILED'
  contaminated_log=$'OMNIASR_UNSUPPORTED_OP\ntest real_omniasr_ctc_encoder_logits_and_tokens_match_official ... FAILED\nOMNIASR_REAL_PARITY_PASS'
  structured_unsupported_log "$unsupported_log" || { echo "self-test rejected valid structured UnsupportedOp" >&2; fail=1; }
  structured_unsupported_log "$contaminated_log" && { echo "self-test accepted contaminated UnsupportedOp log" >&2; fail=1; } || true
  (( fail == 0 )) && echo "apple-silicon-omniasr-ctc.sh self-test: OK" || return 1
}

if [[ "${1:-}" == --self-test ]]; then
  [[ $# == 1 ]] || die "--self-test accepts no arguments"
  self_test
  exit 0
fi
[[ $# == 3 && "$2" == --evidence-dir ]] || die "usage: apple-silicon-omniasr-ctc.sh VAST_BUNDLE --evidence-dir /absolute/absent/path | --self-test"
[[ "${VOKRA_REMOTE_APPLE_SILICON:-0}" == 1 ]] || die "VOKRA_REMOTE_APPLE_SILICON=1 is required"
[[ "$(uname -s)" == Darwin && "$(uname -m)" == arm64 ]] || die "Darwin arm64 is required"
memory="$(sysctl -n hw.memsize 2>/dev/null || true)"
[[ "$memory" =~ ^[0-9]+$ && "$memory" -ge 34359738368 ]] || die "at least 32 GiB RAM is required"
command -v cargo >/dev/null 2>&1 || die "cargo is required"
command -v xcrun >/dev/null 2>&1 || die "xcrun is required"
xcrun -sdk macosx metal -v >/dev/null 2>&1 || die "Metal compiler unavailable"
checkout_status="$(git -C "$ROOT" status --porcelain --untracked-files=all)"
validate_clean_checkout "$checkout_status" || die "clean committed checkout required"
expected_commit="${VOKRA_EXPECTED_COMMIT:-}"
actual_commit="$(git -C "$ROOT" rev-parse --verify HEAD 2>/dev/null)" || die "unable to resolve checkout HEAD"
validate_expected_commit "$expected_commit" "$actual_commit" || \
  die "VOKRA_EXPECTED_COMMIT must be lowercase 40-hex and match HEAD"

bundle="$1"
evidence_dir="$3"
[[ "$bundle" = /* && -d "$bundle" && ! -L "$bundle" ]] || die "bundle must be an absolute non-symlink directory"
cursor="$bundle"
while [[ "$cursor" != "/" ]]; do
  [[ ! -L "$cursor" ]] || die "bundle has a symlink ancestor: $cursor"
  cursor="$(dirname "$cursor")"
done
bundle="$(cd "$bundle" && pwd -P)"
canonical_absent_path() {
  local path="$1" label="$2" parent candidate cursor
  [[ "$path" = /* ]] || die "$label must be absolute: $path"
  cursor="$path"
  while [[ "$cursor" != "/" ]]; do
    [[ ! -L "$cursor" ]] || die "$label has a symlink ancestor: $cursor"
    cursor="$(dirname "$cursor")"
  done
  parent="$(dirname "$path")"
  [[ -d "$parent" ]] || die "$label parent is missing: $parent"
  candidate="$(cd "$parent" && pwd -P)/$(basename "$path")"
  [[ "$candidate" == "$path" ]] || die "$label is a lexical/canonical alias: $path -> $candidate"
  [[ ! -e "$path" ]] || die "$label must be absent before gate: $path"
}
canonical_absent_path "$evidence_dir" evidence-dir
evidence_parent="$(dirname "$evidence_dir")"
evidence_canonical="$(cd "$evidence_parent" && pwd -P)/$(basename "$evidence_dir")"
case "$evidence_canonical/" in
  "$bundle/"* ) die "evidence-dir overlaps input bundle" ;;
esac
case "$bundle/" in
  "$evidence_canonical/"* ) die "input bundle overlaps evidence-dir" ;;
esac
for file in validation-summary.txt omniasr-ctc-1b.gguf reference/manifest.json \
  reference/pcm.f32le reference/frontend.f32le reference/encoder.f32le \
  reference/ctc_logits.f32le reference/tokens.u32le; do
  [[ -f "$bundle/$file" && ! -L "$bundle/$file" ]] || die "bundle file missing/symlinked: $file"
done
[[ -d "$bundle/reference" && ! -L "$bundle/reference" ]] || die "reference directory missing/symlinked"
bundle_symlink="$(find "$bundle" -type l -print -quit)" || die "unable to inspect bundle symlinks"
[[ -z "$bundle_symlink" ]] || die "input bundle contains a symlink: $bundle_symlink"

summary="$bundle/validation-summary.txt"
summary_keys=(
  schema status publication model_id hf_revision checkpoint_sha256 tokenizer_sha256
  prepared_sha256 gguf_sha256 omnilingual_repository omnilingual_revision
  fairseq2_repository fairseq2_revision tensor_count reference_manifest
  reference_manifest_sha256 parity_test parity_status token_exact max_abs metal_status
)
[[ "$(wc -l < "$summary" | tr -d ' ')" == "21" ]] || die "summary line count is not exact"
for key in "${summary_keys[@]}"; do
  [[ "$(grep -Ec "^${key}=[^=]+$" "$summary")" == 1 ]] || die "summary key is missing/duplicated/ill-typed: $key"
done
grep -Fxq 'schema=omniasr-ctc-validation-v1' "$summary" || die "summary schema mismatch"
grep -Fxq 'status=CPU_PARITY_COMPLETE' "$summary" || die "VAST CPU parity was not complete"
grep -Fxq 'publication=NO_UPLOAD' "$summary" || die "bundle is not NO_UPLOAD"
grep -Fxq "model_id=facebook/omniASR-CTC-1B" "$summary" || die "summary model identity mismatch"
grep -Fxq 'hf_revision=8c22e3ffdaa4aab6431b128b84b991a7d9c2515c' "$summary" || die "summary HF revision mismatch"
grep -Fxq 'checkpoint_sha256=e8564fa59dab7caedbcdb54ab7fb9bd6c96989f4d19add2ad81ddd969716952c' "$summary" || die "summary checkpoint hash mismatch"
grep -Fxq 'prepared_sha256=cda8d7dd7cad2a0361b6946c42342b85ef7b0a8d672b99631dc75b4c3123dbc5' "$summary" || die "summary prepared hash mismatch"
grep -Fxq 'omnilingual_repository=https://github.com/facebookresearch/omnilingual-asr' "$summary" || die "summary Omnilingual repository mismatch"
grep -Fxq 'omnilingual_revision=a7fb36017a46eee8953f76bd628c174d51aefeef' "$summary" || die "summary Omnilingual revision mismatch"
grep -Fxq 'fairseq2_repository=https://github.com/facebookresearch/fairseq2' "$summary" || die "summary fairseq2 repository mismatch"
grep -Fxq 'fairseq2_revision=8ae890e1b4d3e36307d0ba5fb695f0fc4815ecca' "$summary" || die "summary fairseq2 revision mismatch"
grep -Fxq 'tensor_count=807' "$summary" || die "summary tensor count mismatch"
grep -Fxq 'reference_manifest=reference/manifest.json' "$summary" || die "summary reference path must be bundle-relative"
grep -Fxq "parity_test=$PARITY_TEST" "$summary" || die "summary parity test mismatch"
grep -Fxq 'parity_status=CPU_PASS' "$summary" || die "summary parity status mismatch"
grep -Fxq 'token_exact=true' "$summary" || die "summary token status mismatch"
grep -Fxq 'max_abs=recorded-by-parity-test' "$summary" || die "summary max-abs field mismatch"
grep -Fxq 'metal_status=NOT_RUN_APPLE_WORKER' "$summary" || die "summary Metal status mismatch"
gguf_sha="$(grep -E '^gguf_sha256=[0-9a-f]{64}$' "$summary" | cut -d= -f2)"
[[ -n "$gguf_sha" ]] || die "GGUF hash is missing"
actual_gguf_sha="$(shasum -a 256 "$bundle/omniasr-ctc-1b.gguf" | awk '{print $1}')"
[[ "$actual_gguf_sha" == "$gguf_sha" ]] || die "GGUF hash mismatch"
manifest_sha="$(grep -E '^reference_manifest_sha256=[0-9a-f]{64}$' "$summary" | cut -d= -f2)"
[[ -n "$manifest_sha" ]] || die "reference manifest hash is missing"
tokenizer_sha="$(grep -E '^tokenizer_sha256=[0-9a-f]{64}$' "$summary" | cut -d= -f2)"
[[ -n "$tokenizer_sha" ]] || die "tokenizer hash is missing or ill-typed"
actual_manifest_sha="$(shasum -a 256 "$bundle/reference/manifest.json" | awk '{print $1}')"
[[ "$actual_manifest_sha" == "$manifest_sha" ]] || die "reference manifest hash mismatch"

# Only the external evidence directory is created; the authenticated input
# bundle remains immutable.  The exact named Rust parity test validates the
# manifest schema, source pins, artifacts, and numerical outputs, so Apple
# needs no uv environment or upstream source checkout.
mkdir "$evidence_dir"
run_dir="$evidence_dir"
capture_one_line() {
  local output
  if output="$("$@" 2>&1)"; then
    printf '%s' "$output" | tr '\n' ';'
  else
    printf 'unavailable'
  fi
}
sysctl_value() {
  local value
  if value="$(sysctl -n "$1" 2>/dev/null)"; then
    printf '%s' "$value"
  else
    printf 'unavailable'
  fi
}
{
  echo 'schema=omniasr-ctc-apple-environment-v1'
  echo "expected_commit=$expected_commit"
  echo "head_commit=$actual_commit"
  echo "uname=$(capture_one_line uname -a)"
  echo "sw_vers=$(capture_one_line sw_vers)"
  echo "sysctl_hw.model=$(sysctl_value hw.model)"
  echo "sysctl_hw.machine=$(sysctl_value hw.machine)"
  echo "sysctl_hw.memsize=$(sysctl_value hw.memsize)"
  echo "sysctl_hw.ncpu=$(sysctl_value hw.ncpu)"
  echo "rustc=$(capture_one_line rustc --version --verbose)"
  echo "cargo=$(capture_one_line cargo --version --verbose)"
  echo "xcrun_metal_path=$(capture_one_line xcrun --find metal)"
  echo "xcrun_metal_version=$(capture_one_line xcrun -sdk macosx metal -v)"
} > "$run_dir/environment.txt"
bundle_digest() {
  local symlink
  symlink="$(find "$bundle" -type l -print -quit)" || return 1
  [[ -z "$symlink" ]] || return 1
  find "$bundle" -type f -exec shasum -a 256 {} + | LC_ALL=C sort
}
input_digest_before="$(bundle_digest)" || die "unable to fingerprint immutable input bundle"
printf '%s\n' "$input_digest_before" > "$run_dir/input-sha256-before.txt"
assert_bundle_unchanged() {
  local input_digest_after
  input_digest_after="$(bundle_digest)" || die "unable to re-fingerprint input bundle"
  [[ "$input_digest_after" == "$input_digest_before" ]] || \
    die "authenticated input bundle changed during Apple validation"
}
set +e
VOKRA_OMNIASR_GGUF="$bundle/omniasr-ctc-1b.gguf" \
VOKRA_OMNIASR_REFERENCE_DIR="$bundle/reference" \
VOKRA_OMNIASR_BACKEND=cpu cargo test --locked -p vokra-models \
  --test parity_omniasr_ctc_real -- --ignored --exact "$PARITY_TEST" --nocapture >"$run_dir/cpu.log" 2>&1
cpu_status=$?
set -e
assert_bundle_unchanged
[[ "$cpu_status" == 0 ]] || die "Apple CPU parity failed; see $run_dir/cpu.log"
grep -Ec "^test ${PARITY_TEST} \.\.\. ok$" "$run_dir/cpu.log" | grep -Fxq 1 || die "CPU named test result is not exactly one PASS"
grep -Fxc 'OMNIASR_REAL_PARITY_PASS' "$run_dir/cpu.log" | grep -Fxq 1 || die "CPU parity sentinel is not exactly one"
cpu_log="$(<"$run_dir/cpu.log")"
backend_sentinel_log "$cpu_log" Cpu || die "CPU log lacks the exact CPU backend sentinel"
echo 'cpu_status=PASS' > "$run_dir/result.txt"

set +e
VOKRA_OMNIASR_GGUF="$bundle/omniasr-ctc-1b.gguf" \
VOKRA_OMNIASR_REFERENCE_DIR="$bundle/reference" \
VOKRA_OMNIASR_BACKEND=metal cargo test --locked -p vokra-models --features metal \
  --test parity_omniasr_ctc_real -- --ignored --exact "$PARITY_TEST" --nocapture >"$run_dir/metal.log" 2>&1
metal_status=$?
set -e
assert_bundle_unchanged
if [[ "$metal_status" == 0 ]]; then
  grep -Ec "^test ${PARITY_TEST} \.\.\. ok$" "$run_dir/metal.log" | grep -Fxq 1 || die "Metal named test result is not exactly one PASS"
  grep -Fxc 'OMNIASR_REAL_PARITY_PASS' "$run_dir/metal.log" | grep -Fxq 1 || die "Metal parity sentinel is not exactly one"
  metal_log="$(<"$run_dir/metal.log")"
  backend_sentinel_log "$metal_log" Metal || die "Metal log lacks the exact Metal backend sentinel"
  echo 'metal_status=PASS' >> "$run_dir/result.txt"
  echo '[omniasr-ctc-apple] CPU_PASS; METAL_PASS; NO_UPLOAD'
  exit 0
fi
metal_log="$(<"$run_dir/metal.log")"
structured_unsupported_log "$metal_log" || \
  die "Metal parity failed without exactly one structured UnsupportedOp marker"
echo 'metal_status=BLOCKED_UNSUPPORTED' >> "$run_dir/result.txt"
echo '[omniasr-ctc-apple] CPU_PASS; METAL_BLOCKED_UNSUPPORTED (structured UnsupportedOp); no CPU fallback; NO_UPLOAD' >&2
exit 3
