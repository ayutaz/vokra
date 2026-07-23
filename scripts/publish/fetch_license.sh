#!/usr/bin/env bash
# fetch_license.sh — obtain the correct LICENSE text for a model being published.
#
# The right text depends on where the licence actually lives:
#   * some upstreams ship a LICENSE file (fetch it — it carries the specific
#     copyright line the licence requires be retained);
#   * some declare the licence only in HF model-card front-matter and ship no
#     LICENSE file (use the canonical SPDX text);
#   * CC-BY / CC-BY-SA point at the canonical Creative Commons legalcode.
#
# Redistribution obligation: MIT and BSD require the copyright notice travel
# with the work, Apache-2.0 requires the licence (and NOTICE if present),
# CC-BY requires attribution + the licence. Shipping the wrong text, or none,
# fails that — so this refuses rather than emit a placeholder.
#
# Usage:
#   fetch_license.sh --url  <raw-license-url>  <out-file>   # upstream ships one
#   fetch_license.sh --spdx <apache-2.0|mit|cc-by-4.0|...> <out-file>  # canonical
#   fetch_license.sh --self-test

set -euo pipefail
CURL="/usr/bin/curl"

canonical_url() {
  case "$1" in
    apache-2.0)  echo "https://www.apache.org/licenses/LICENSE-2.0.txt" ;;
    cc-by-4.0)   echo "https://creativecommons.org/licenses/by/4.0/legalcode.txt" ;;
    cc-by-sa-4.0) echo "https://creativecommons.org/licenses/by-sa/4.0/legalcode.txt" ;;
    *) return 1 ;;
  esac
}

if [[ "${1:-}" == "--self-test" ]]; then
  # Canonical Apache-2.0 must fetch and look like Apache-2.0.
  tmp="$(mktemp)"; trap 'rm -f "$tmp"' EXIT
  if "$0" --spdx apache-2.0 "$tmp" >/dev/null 2>&1 && grep -qi "Apache License" "$tmp"; then
    echo "fetch_license self-test: OK"; exit 0
  fi
  echo "fetch_license self-test: FAIL (could not fetch canonical Apache-2.0)" >&2
  exit 1
fi

mode="$1"; val="$2"; out="$3"

case "$mode" in
  --url)
    code="$("$CURL" -sL -o "$out" -w '%{http_code}' "$val")"
    [[ "$code" == "200" ]] || { echo "fetch_license: $val returned HTTP $code" >&2; exit 2; }
    ;;
  --spdx)
    spdx="$(printf '%s' "$val" | tr '[:upper:]' '[:lower:]')"
    if url="$(canonical_url "$spdx")"; then
      code="$("$CURL" -sL -o "$out" -w '%{http_code}' "$url")"
      [[ "$code" == "200" ]] || { echo "fetch_license: canonical $spdx ($url) HTTP $code" >&2; exit 2; }
    else
      echo "fetch_license: no canonical URL known for SPDX '$spdx'. Pass --url with the upstream LICENSE instead." >&2
      exit 3
    fi
    ;;
  *) echo "fetch_license: mode must be --url or --spdx" >&2; exit 2 ;;
esac

[[ -s "$out" ]] || { echo "fetch_license: wrote an empty file" >&2; exit 4; }
echo "fetch_license: wrote $(wc -l < "$out" | tr -d ' ') lines to $out"
