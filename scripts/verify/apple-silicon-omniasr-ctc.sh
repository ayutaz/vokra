#!/usr/bin/env bash
# Apple Silicon consumer of an authenticated VAST OmniASR-CTC packet.
# It never downloads, converts, or substitutes a reference.
set -euo pipefail

ROOT="${VOKRA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PARITY_TEST="real_omniasr_ctc_encoder_logits_and_tokens_match_official"
die() { echo "[omniasr-ctc-apple] ERROR: $*" >&2; exit 2; }

structured_unsupported_log() {
  local log="$1"
  [[ "$(printf '%s\n' "$log" | grep -Fxc 'OMNIASR_UNSUPPORTED_OP')" == 1 ]] || return 1
  [[ "$(printf '%s\n' "$log" | grep -Ec "^test ${PARITY_TEST} \.\.\. FAILED$")" == 1 ]] || return 1
  [[ "$(printf '%s\n' "$log" | grep -Fxc 'OMNIASR_REAL_PARITY_PASS')" == 0 ]] || return 1
}

self_test() {
  local fail=0 required unsupported_log contaminated_log
  for required in Darwin arm64 VOKRA_REMOTE_APPLE_SILICON \
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
[[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]] || die "clean committed checkout required"

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
set +e
VOKRA_OMNIASR_GGUF="$bundle/omniasr-ctc-1b.gguf" \
VOKRA_OMNIASR_REFERENCE_DIR="$bundle/reference" \
VOKRA_OMNIASR_BACKEND=cpu cargo test --locked -p vokra-models \
  --test parity_omniasr_ctc_real -- --ignored --exact "$PARITY_TEST" --nocapture >"$run_dir/cpu.log" 2>&1
cpu_status=$?
set -e
[[ "$cpu_status" == 0 ]] || die "Apple CPU parity failed; see $run_dir/cpu.log"
grep -Ec "^test ${PARITY_TEST} \.\.\. ok$" "$run_dir/cpu.log" | grep -Fxq 1 || die "CPU named test result is not exactly one PASS"
grep -Fxc 'OMNIASR_REAL_PARITY_PASS' "$run_dir/cpu.log" | grep -Fxq 1 || die "CPU parity sentinel is not exactly one"
echo 'cpu_status=PASS' > "$run_dir/result.txt"

set +e
VOKRA_OMNIASR_GGUF="$bundle/omniasr-ctc-1b.gguf" \
VOKRA_OMNIASR_REFERENCE_DIR="$bundle/reference" \
VOKRA_OMNIASR_BACKEND=metal cargo test --locked -p vokra-models --features metal \
  --test parity_omniasr_ctc_real -- --ignored --exact "$PARITY_TEST" --nocapture >"$run_dir/metal.log" 2>&1
metal_status=$?
set -e
if [[ "$metal_status" == 0 ]]; then
  grep -Ec "^test ${PARITY_TEST} \.\.\. ok$" "$run_dir/metal.log" | grep -Fxq 1 || die "Metal named test result is not exactly one PASS"
  grep -Fxc 'OMNIASR_REAL_PARITY_PASS' "$run_dir/metal.log" | grep -Fxq 1 || die "Metal parity sentinel is not exactly one"
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
