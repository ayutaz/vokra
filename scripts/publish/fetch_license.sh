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
#   fetch_license.sh --spdx <spdx-id> <out-file>            # canonical
#     supported spdx-id:
#       apache-2.0 | mit
#       bsd-2-clause | bsd-3-clause | isc | unlicense | cc0-1.0
#       gpl-3.0 | lgpl-3.0 | agpl-3.0 | mpl-2.0 | epl-2.0
#       cc-by-4.0 | cc-by-sa-3.0 | cc-by-sa-4.0
#       cc-by-nc-4.0 | cc-by-nc-sa-4.0
#       openmdw-1.1  (inline, no canonical URL)
#   fetch_license.sh --self-test          # deterministic offline contract tests
#   fetch_license.sh --network-self-test  # optional live canonical-URL probe

set -euo pipefail
CURL="${CURL:-/usr/bin/curl}"

canonical_url() {
  # Source selection policy:
  #   * apache.org for Apache-2.0 (canonical from ASF)
  #   * gnu.org for the GNU family: GPL-3.0 / LGPL-3.0 / AGPL-3.0 (canonical from FSF)
  #   * creativecommons.org for the CC family (canonical from Creative Commons)
  #   * unlicense.org for The Unlicense (canonical from the licence authors)
  #   * SPDX license-list-data raw for permissive / weak-copyleft SPDX texts
  #     that lack a stable primary-source plain-text URL:
  #     MIT / BSD-2-Clause / BSD-3-Clause / ISC / MPL-2.0 / EPL-2.0
  #     — the SPDX repo is the reference plain-text mirror the SPDX standard
  #     itself points at, and every entry here is the byte-exact licence text
  #     the SPDX id resolves to.
  case "$1" in
    apache-2.0)  echo "https://www.apache.org/licenses/LICENSE-2.0.txt" ;;
    mit)         echo "https://raw.githubusercontent.com/spdx/license-list-data/main/text/MIT.txt" ;;
    cc-by-4.0)   echo "https://creativecommons.org/licenses/by/4.0/legalcode.txt" ;;
    cc-by-sa-3.0) echo "https://creativecommons.org/licenses/by-sa/3.0/legalcode.txt" ;;
    cc-by-sa-4.0) echo "https://creativecommons.org/licenses/by-sa/4.0/legalcode.txt" ;;
    cc-by-nc-4.0) echo "https://creativecommons.org/licenses/by-nc/4.0/legalcode.txt" ;;
    cc-by-nc-sa-4.0) echo "https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.txt" ;;
    gpl-3.0)     echo "https://www.gnu.org/licenses/gpl-3.0.txt" ;;
    lgpl-3.0)    echo "https://www.gnu.org/licenses/lgpl-3.0.txt" ;;
    agpl-3.0)    echo "https://www.gnu.org/licenses/agpl-3.0.txt" ;;
    mpl-2.0)     echo "https://raw.githubusercontent.com/spdx/license-list-data/main/text/MPL-2.0.txt" ;;
    epl-2.0)     echo "https://raw.githubusercontent.com/spdx/license-list-data/main/text/EPL-2.0.txt" ;;
    isc)         echo "https://raw.githubusercontent.com/spdx/license-list-data/main/text/ISC.txt" ;;
    unlicense)   echo "https://unlicense.org/UNLICENSE" ;;
    bsd-2-clause) echo "https://raw.githubusercontent.com/spdx/license-list-data/main/text/BSD-2-Clause.txt" ;;
    bsd-3-clause) echo "https://raw.githubusercontent.com/spdx/license-list-data/main/text/BSD-3-Clause.txt" ;;
    cc0-1.0)      echo "https://creativecommons.org/publicdomain/zero/1.0/legalcode.txt" ;;
    *) return 1 ;;
  esac
}

# Inline license text for SPDX ids without a stable plain-text canonical URL.
# openmdw-1.1 falls in this category: openmdw.ai/license/1-1/ serves HTML only,
# and the Linux Foundation has not published a canonical plain-text mirror.
# The verbatim text below is transcribed from openmdw.ai/license/1-1/
# (CC-directly fetched 2026-07-30). Redistribution obligation §D requires
# a copy of the agreement to travel with the Model Materials — inlining the
# text here is the simplest way to satisfy that discipline without a
# canonical URL.
inline_license_text() {
  case "$1" in
    openmdw-1.1)
      cat <<'OPENMDW_EOF'
OpenMDW License Agreement, version 1.1 (OpenMDW-1.1)

By exercising rights granted to you under this agreement, you accept and agree to its terms.

As used in this agreement, "Model Materials" means the materials provided to you under this agreement, consisting of: (1) one or more machine learning models (including architecture and parameters); and (2) all related artifacts (including associated data, documentation and software) that are provided to you hereunder.

Subject to your compliance with this agreement, permission is hereby granted, free of charge, to deal in the Model Materials without restriction, including under all copyright, patent, database, and trade secret rights included or embodied therein.

If you distribute any portion of the Model Materials, you shall retain in your distribution (1) a copy of this agreement, and (2) all copyright notices and other notices of origin included in the Model Materials that are applicable to your distribution.

If you file, maintain, or voluntarily participate in a lawsuit against any person or entity asserting that the Model Materials directly or indirectly infringe any patent or copyright, then all rights and grants made to you hereunder are terminated, unless that lawsuit was in response to a corresponding lawsuit first brought against you.

This agreement does not impose any restrictions or obligations with respect to any use, modification, or sharing of any outputs generated by using the Model Materials.

THE MODEL MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, TITLE, NONINFRINGEMENT, ACCURACY, OR THE ABSENCE OF LATENT OR OTHER DEFECTS OR ERRORS, WHETHER OR NOT DISCOVERABLE, ALL TO THE GREATEST EXTENT PERMISSIBLE UNDER APPLICABLE LAW.

YOU ARE SOLELY RESPONSIBLE FOR (1) CLEARING RIGHTS OF OTHER PERSONS THAT MAY APPLY TO THE MODEL MATERIALS OR ANY USE THEREOF, INCLUDING WITHOUT LIMITATION ANY PERSON'S COPYRIGHTS OR OTHER RIGHTS INCLUDED OR EMBODIED IN THE MODEL MATERIALS; (2) OBTAINING ANY NECESSARY CONSENTS, PERMISSIONS OR OTHER RIGHTS REQUIRED FOR ANY USE OF THE MODEL MATERIALS; OR (3) PERFORMING ANY DUE DILIGENCE OR UNDERTAKING ANY OTHER INVESTIGATIONS INTO THE MODEL MATERIALS OR ANYTHING INCORPORATED OR EMBODIED THEREIN.

IN NO EVENT SHALL THE PROVIDERS OF THE MODEL MATERIALS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE MODEL MATERIALS, THE USE THEREOF OR OTHER DEALINGS THEREIN.

Copyright The Linux Foundation and its contributors.
OPENMDW_EOF
      ;;
    *) return 1 ;;
  esac
}

if [[ "${1:-}" == "--self-test" ]]; then
  # Required PR CI must be deterministic: a public licence host timing out is
  # not evidence that this repository's resolver regressed. Exercise the real
  # --spdx path with a curl fixture, including content needles and fail-closed
  # HTTP handling. The separate --network-self-test mode retains the live URL
  # drift probe for scheduled/manual use.
  tmp_dir="$(mktemp -d)"
  trap 'rm -f "$tmp_dir"/*; rmdir "$tmp_dir"' EXIT
  mock_curl="$tmp_dir/curl"
  cat > "$mock_curl" <<'MOCK_CURL'
#!/usr/bin/env bash
set -euo pipefail
out=""
url=""
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    -o|-w|--connect-timeout|--max-time) [[ "$1" == "-o" ]] && out="$2"; shift 2 ;;
    -*) shift ;;
    *) url="$1"; shift ;;
  esac
done
code="${MOCK_LICENSE_HTTP_CODE:-200}"
if [[ "$code" == "200" ]]; then
  case "$url" in
    *LICENSE-2.0.txt) body="Apache License" ;;
    *MIT.txt) body="MIT License" ;;
    *BSD-2-Clause.txt) body="Redistribution and use in source and binary forms" ;;
    *BSD-3-Clause.txt) body="Neither the name of" ;;
    *ISC.txt) body="Permission to use, copy, modify" ;;
    *UNLICENSE) body="This is free and unencumbered software" ;;
    *publicdomain/zero/1.0/*) body="CC0 1.0 Universal" ;;
    *lgpl-3.0.txt) body="GNU LESSER GENERAL PUBLIC LICENSE" ;;
    *agpl-3.0.txt) body="GNU AFFERO GENERAL PUBLIC LICENSE" ;;
    *gpl-3.0.txt) body="GNU GENERAL PUBLIC LICENSE" ;;
    *MPL-2.0.txt) body="Mozilla Public License" ;;
    *EPL-2.0.txt) body="Eclipse Public License" ;;
    *licenses/by/4.0/*) body="Attribution 4.0 International" ;;
    *licenses/by-sa/3.0/*) body="Attribution-ShareAlike 3.0" ;;
    *licenses/by-sa/4.0/*) body="Attribution-ShareAlike 4.0 International" ;;
    *licenses/by-nc/4.0/*) body="Attribution-NonCommercial 4.0 International" ;;
    *licenses/by-nc-sa/4.0/*) body="Attribution-NonCommercial-ShareAlike 4.0 International" ;;
    *) body="unexpected fixture URL: $url" ;;
  esac
  printf '%s\n' "$body" > "$out"
fi
printf '%s' "$code"
MOCK_CURL
  chmod +x "$mock_curl"

  suites=(
    "apache-2.0|Apache License"
    "mit|MIT License"
    "bsd-2-clause|Redistribution and use in source and binary forms"
    "bsd-3-clause|Neither the name of"
    "isc|Permission to use, copy, modify"
    "unlicense|This is free and unencumbered software"
    "cc0-1.0|CC0 1.0 Universal"
    "gpl-3.0|GNU GENERAL PUBLIC LICENSE"
    "lgpl-3.0|GNU LESSER GENERAL PUBLIC LICENSE"
    "agpl-3.0|GNU AFFERO GENERAL PUBLIC LICENSE"
    "mpl-2.0|Mozilla Public License"
    "epl-2.0|Eclipse Public License"
    "cc-by-4.0|Attribution 4.0 International"
    "cc-by-sa-3.0|Attribution-ShareAlike 3.0"
    "cc-by-sa-4.0|Attribution-ShareAlike 4.0 International"
    "cc-by-nc-4.0|Attribution-NonCommercial 4.0 International"
    "cc-by-nc-sa-4.0|Attribution-NonCommercial-ShareAlike 4.0 International"
    "openmdw-1.1|OpenMDW License Agreement"
  )

  pass=0; fail=0
  for suite in "${suites[@]}"; do
    spdx="${suite%%|*}"; needle="${suite#*|}"
    out="$tmp_dir/$spdx.txt"
    if ! CURL="$mock_curl" "$0" --spdx "$spdx" "$out" >/dev/null 2>&1; then
      echo "fetch_license self-test: FAIL — --spdx $spdx rejected fixture HTTP 200" >&2
      fail=$((fail+1)); continue
    fi
    if ! grep -qi -- "$needle" "$out"; then
      echo "fetch_license self-test: FAIL — --spdx $spdx body missing needle '$needle'" >&2
      fail=$((fail+1)); continue
    fi
    pass=$((pass+1))
  done

  if MOCK_LICENSE_HTTP_CODE=503 CURL="$mock_curl" \
      "$0" --spdx apache-2.0 "$tmp_dir/http-503.txt" >/dev/null 2>&1; then
    echo "fetch_license self-test: FAIL — HTTP 503 must fail closed" >&2
    fail=$((fail+1))
  else
    pass=$((pass+1))
  fi
  if CURL="$mock_curl" "$0" --spdx unknown-license "$tmp_dir/unknown.txt" >/dev/null 2>&1; then
    echo "fetch_license self-test: FAIL — unknown SPDX id must be rejected" >&2
    fail=$((fail+1))
  else
    pass=$((pass+1))
  fi

  if [[ "$fail" -gt 0 ]]; then
    echo "fetch_license self-test: FAIL ($pass passed, $fail failed)" >&2
    exit 1
  fi
  echo "fetch_license self-test: OK ($pass deterministic cases passed)"
  exit 0
fi

if [[ "${1:-}" == "--network-self-test" ]]; then
  # Coverage discipline: every SPDX id resolved by canonical_url() must be
  # reachable AND its fetched body must contain a needle unique to that
  # licence. If we accept HTTP 200 alone we would accept a captive-portal
  # HTML page as "GPL" — so grep the fetched text for a phrase that only
  # appears in the real licence body.
  #
  # Offline / firewalled runs must SKIP, not FAIL: probe network reachability
  # against apache.org first (any SPDX-mirror host outage would be caught in
  # the per-suite loop below).
  tmp="$(mktemp)"; trap 'rm -f "$tmp"' EXIT

  probe_url="https://www.apache.org/licenses/LICENSE-2.0.txt"
  if ! "$CURL" -sSfI --max-time 10 "$probe_url" >/dev/null 2>&1; then
    echo "fetch_license network-self-test: SKIP (network unreachable — offline / firewalled CI)"
    exit 0
  fi

  # spdx-id | grep-needle. Needle must appear byte-for-byte in the licence
  # body (case-insensitive, so header-vs-title differences don't matter).
  # Every canonical_url() branch and inline_license_text() branch is covered.
  suites=(
    "apache-2.0|Apache License"
    "mit|MIT License"
    "bsd-2-clause|Redistribution and use in source and binary forms"
    "bsd-3-clause|Neither the name of"
    "isc|Permission to use, copy, modify"
    "unlicense|This is free and unencumbered software"
    "cc0-1.0|CC0 1.0 Universal"
    "gpl-3.0|GNU GENERAL PUBLIC LICENSE"
    "lgpl-3.0|GNU LESSER GENERAL PUBLIC LICENSE"
    "agpl-3.0|GNU AFFERO GENERAL PUBLIC LICENSE"
    "mpl-2.0|Mozilla Public License"
    "epl-2.0|Eclipse Public License"
    "cc-by-4.0|Attribution 4.0 International"
    "cc-by-sa-3.0|Attribution-ShareAlike 3.0"
    "cc-by-sa-4.0|Attribution-ShareAlike 4.0 International"
    "cc-by-nc-4.0|Attribution-NonCommercial 4.0 International"
    "cc-by-nc-sa-4.0|Attribution-NonCommercial-ShareAlike 4.0 International"
    "openmdw-1.1|OpenMDW License Agreement"
  )

  # Per-host reachability cache (bash-3.2-compatible, no associative arrays).
  seen_reach=""      # space-separated list of hosts confirmed reachable
  seen_unreach=""    # space-separated list of hosts confirmed unreachable
  probe_host() {
    # 0 = reachable, 1 = unreachable. Result cached.
    local host="$1"
    case " $seen_reach " in *" $host "*) return 0;; esac
    case " $seen_unreach " in *" $host "*) return 1;; esac
    if "$CURL" -sSfI --max-time 10 "https://$host/" >/dev/null 2>&1; then
      seen_reach="$seen_reach $host"; return 0
    fi
    seen_unreach="$seen_unreach $host"; return 1
  }

  pass=0; fail=0; skip=0
  for suite in "${suites[@]}"; do
    spdx="${suite%%|*}"; needle="${suite#*|}"
    : > "$tmp"
    if "$0" --spdx "$spdx" "$tmp" >/dev/null 2>&1; then
      # Fetch (or inline) succeeded — content must contain the needle.
      if ! [[ -s "$tmp" ]]; then
        echo "fetch_license network-self-test: FAIL — --spdx $spdx wrote an empty file" >&2
        fail=$((fail+1)); continue
      fi
      if ! grep -qi -- "$needle" "$tmp"; then
        echo "fetch_license network-self-test: FAIL — --spdx $spdx body missing needle '$needle'" >&2
        fail=$((fail+1)); continue
      fi
      pass=$((pass+1))
      continue
    fi
    # Fetch failed. Distinguish "host unreachable" (SKIP) from "URL/content
    # broken" (FAIL). Inline-only SPDX ids (canonical_url returns non-zero)
    # never depend on the network, so failure there is a real bug → FAIL.
    url="$(canonical_url "$spdx" 2>/dev/null || true)"
    if [[ -z "$url" ]]; then
      echo "fetch_license network-self-test: FAIL — --spdx $spdx has no canonical URL and inline_license_text failed" >&2
      fail=$((fail+1)); continue
    fi
    host="$(printf '%s' "$url" | awk -F/ '{print $3}')"
    if probe_host "$host"; then
      echo "fetch_license network-self-test: FAIL — --spdx $spdx (host $host reachable but $url did not return HTTP 200)" >&2
      fail=$((fail+1))
    else
      echo "fetch_license network-self-test: SKIP — --spdx $spdx (host $host unreachable — transient network / firewall)"
      skip=$((skip+1))
    fi
  done

  if [[ "$fail" -gt 0 ]]; then
    echo "fetch_license network-self-test: FAIL ($pass passed, $skip skipped, $fail failed)" >&2
    exit 1
  fi
  echo "fetch_license network-self-test: OK ($pass passed, $skip skipped)"
  exit 0
fi

mode="$1"; val="$2"; out="$3"

case "$mode" in
  --url)
    # --connect-timeout bounds TCP handshake only (not download body time), so
    # a slow upstream still completes; but a dead host (DNS OK, TCP times out)
    # fails in seconds instead of hanging the whole publish pipeline for
    # curl's default ~5 min. --max-time is a hard ceiling for the full call.
    code="$("$CURL" -sL --connect-timeout 15 --max-time 120 -o "$out" -w '%{http_code}' "$val")"
    [[ "$code" == "200" ]] || { echo "fetch_license: $val returned HTTP $code" >&2; exit 2; }
    ;;
  --spdx)
    spdx="$(printf '%s' "$val" | tr '[:upper:]' '[:lower:]')"
    if url="$(canonical_url "$spdx")"; then
      code="$("$CURL" -sL --connect-timeout 15 --max-time 120 -o "$out" -w '%{http_code}' "$url")"
      [[ "$code" == "200" ]] || { echo "fetch_license: canonical $spdx ($url) HTTP $code" >&2; exit 2; }
    elif inline_license_text "$spdx" > "$out" 2>/dev/null && [[ -s "$out" ]]; then
      : # inline text written — HTTP-fetch not required.
    else
      echo "fetch_license: no canonical URL or inline text known for SPDX '$spdx'. Pass --url with the upstream LICENSE instead." >&2
      exit 3
    fi
    ;;
  *) echo "fetch_license: mode must be --url or --spdx" >&2; exit 2 ;;
esac

[[ -s "$out" ]] || { echo "fetch_license: wrote an empty file" >&2; exit 4; }
echo "fetch_license: wrote $(wc -l < "$out" | tr -d ' ') lines to $out"
