#!/usr/bin/env bash
set -euo pipefail

root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
validator="$root/scripts/release/validate_published_cli_pairing.py"
pins="$root/tools/parity/release_candidate_published_cli_pairing_pins.json"
tag=v0.4.0-rc.1
cli_name=vokra-cli-0.4.0-rc.1-x86_64-linux
driver_blob_oid=4107f83b64fe59d6a121d051271c9b41e763649b

if [[ "${1:-}" == "--metadata-only" ]]; then
  exec uv run --no-project --python 3.12 python "$validator" --metadata-only --pins "$pins" --head metadata
fi

evidence="${RUNNER_TEMP:-/tmp}/vokra-rc1-pairing-evidence"
mkdir -p "$evidence"
output="$evidence/pairing-receipt.json"
source_head="${GITHUB_SHA:-}"
workflow_sha="${GITHUB_WORKFLOW_SHA:-}"
run_id="${GITHUB_RUN_ID:-}"
attempt="${GITHUB_RUN_ATTEMPT:-}"
blocked() {
  uv run --no-project --python 3.12 python "$validator" --pins "$pins" --head "$source_head" --output "$output" --driver-blob-oid "$driver_blob_oid" --workflow-sha "$workflow_sha" --run-id "$run_id" --attempt "$attempt" --blocked-reason "$1" || true
  exit 2
}
if [[ "${GITHUB_ACTIONS:-}" != "true" ]]; then
  blocked "requires genuine GitHub-hosted Actions runtime"
fi
if [[ "${1:-}" != "--allow-models" || "${2:-}" != "true" ]]; then
  blocked "explicit --allow-models true is required"
fi
if [[ "$source_head" != "$(git -C "$root" rev-parse HEAD)" ]]; then
  blocked "GITHUB_SHA does not match the checked-out source"
fi
if [[ "$(git -C "$root" hash-object "$root/tools/parity/release_candidate_cpu_smoke.py")" != "$driver_blob_oid" ]]; then
  blocked "accepted driver blob identity mismatch"
fi
export VOKRA_RELEASE_CPU_SMOKE=1

work=$(mktemp -d "${RUNNER_TEMP:-/tmp}/vokra-rc1-pairing.XXXXXX")
cleanup() {
  rc=$?
  if [[ ! -s "$output" ]]; then
    uv run --no-project --python 3.12 python "$validator" --pins "$pins" --head "$source_head" --output "$output" --driver-blob-oid "$driver_blob_oid" --workflow-sha "$workflow_sha" --run-id "$run_id" --attempt "$attempt" --blocked-reason "workflow terminated before evidence: exit $rc" || true
  fi
  rm -rf "$work"
  exit "$rc"
}
trap cleanup EXIT
release="$work/release.json"
tag_json="$work/tag.json"
cli="$work/$cli_name"
sidecar="$work/$cli_name.sha256"
manifest="$work/vokra-desktop-manifest.json"
audio="$work/jfk-30s.wav"
receipt="$work/driver-receipt.json"

curl --fail --location --proto '=https' --tlsv1.2 --connect-timeout 30 --max-time 120 --silent --show-error "https://api.github.com/repos/ayutaz/vokra/releases/tags/$tag" -o "$release"
curl --fail --location --proto '=https' --tlsv1.2 --connect-timeout 30 --max-time 120 --silent --show-error "https://api.github.com/repos/ayutaz/vokra/git/ref/tags/$tag" -o "$tag_json"
curl --fail --location --proto '=https' --tlsv1.2 --connect-timeout 30 --max-time 120 --silent --show-error "https://github.com/ayutaz/vokra/releases/download/$tag/$cli_name" -o "$cli"
curl --fail --location --proto '=https' --tlsv1.2 --connect-timeout 30 --max-time 120 --silent --show-error "https://github.com/ayutaz/vokra/releases/download/$tag/$cli_name.sha256" -o "$sidecar"
curl --fail --location --proto '=https' --tlsv1.2 --connect-timeout 30 --max-time 120 --silent --show-error "https://github.com/ayutaz/vokra/releases/download/$tag/vokra-desktop-manifest.json" -o "$manifest"
curl --fail --location --proto '=https' --tlsv1.2 --connect-timeout 30 --max-time 120 --silent --show-error "https://raw.githubusercontent.com/ayutaz/vokra/$tag/tests/fixtures/audio/jfk-30s.wav" -o "$audio"

uv run --no-project --python 3.12 python "$validator" --pins "$pins" --release "$release" --tag "$tag_json" --cli "$cli" --cli-name "$cli_name" --sidecar "$sidecar" --manifest "$manifest" --audio "$audio" --head "$source_head"
chmod 700 "$cli"
set +e
uv run --no-project --python 3.12 python "$root/tools/parity/release_candidate_cpu_smoke.py" --cli "$cli" --audio "$audio" --receipt "$receipt" --expected-head "$source_head" --pins-json "$pins" >"$work/driver.log" 2>&1
driver_rc=$?
set -e
if (( driver_rc != 0 )); then
  set +e
  uv run --no-project --python 3.12 python "$validator" --pins "$pins" --head "$source_head" --receipt "$receipt" --output "$output" --driver-log "$work/driver.log" --driver-exit-code "$driver_rc" --driver-blob-oid "$driver_blob_oid" --workflow-sha "$workflow_sha" --run-id "$run_id" --attempt "$attempt"
  set -e
  exit "$driver_rc"
fi
uv run --no-project --python 3.12 python "$validator" --pins "$pins" --head "$source_head" --receipt "$receipt" --output "$output" --driver-blob-oid "$driver_blob_oid" --workflow-sha "$workflow_sha" --run-id "$run_id" --attempt "$attempt"
