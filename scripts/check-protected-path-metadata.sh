#!/usr/bin/env bash
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel)"
GUARD="$ROOT/.codex/hooks/guard-protected-path.sh"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/vokra-protected-path-regression.XXXXXX")"
trap 'rm -rf "$TEST_ROOT"' EXIT
git -C "$TEST_ROOT" init -q
ln -s "$ROOT" "$TEST_ROOT/repo-link"
run_case() {
    local name="$1" expected="$2" payload="$3" cwd="${4:-$ROOT}" rc
    set +e
    printf '%s' "$payload" | (cd "$cwd" && bash "$GUARD") >/dev/null 2>&1
    rc=$?
    set -e
    [ "$rc" -eq "$expected" ] || { printf 'FAIL %s expected=%s actual=%s\n' "$name" "$expected" "$rc"; return 1; }
    printf 'PASS %s rc=%s\n' "$name" "$rc"
}
patch_payload() {
    local header="$1" target="$2" cwd="${3-__MISSING__}" body="${4-+x}" command
    command="*** Begin Patch
$header$target
$body
*** End Patch"
    if [ "$cwd" = __MISSING__ ]; then
        jq -cn --arg command "$command" '{tool_name:"apply_patch",tool_input:{command:$command}}'
    else
        jq -cn --arg command "$command" --arg cwd "$cwd" '{cwd:$cwd,tool_name:"apply_patch",tool_input:{command:$command}}'
    fi
}
eof_payload() {
    local command
    command="*** Begin Patch
*** Add File: /private/tmp/vokra-eof
+x
*** End of File
*** End Patch"
    jq -cn --arg command "$command" '{tool_name:"apply_patch",tool_input:{command:$command}}'
}
target_file_payload() {
    local tool="$1" target="$2" cwd="${3-__MISSING__}"
    if [ "$cwd" = __MISSING__ ]; then
        jq -cn --arg tool "$tool" --arg target "$target" '{tool_name:$tool,tool_input:{file_path:$target}}'
    else
        jq -cn --arg tool "$tool" --arg target "$target" --arg cwd "$cwd" '{cwd:$cwd,tool_name:$tool,tool_input:{file_path:$target}}'
    fi
}
target_path_payload() {
    local tool="$1" target="$2" cwd="${3-__MISSING__}"
    if [ "$cwd" = __MISSING__ ]; then
        jq -cn --arg tool "$tool" --arg target "$target" '{tool_name:$tool,tool_input:{path:$target}}'
    else
        jq -cn --arg tool "$tool" --arg target "$target" --arg cwd "$cwd" '{cwd:$cwd,tool_name:$tool,tool_input:{path:$target}}'
    fi
}
target_pair_payload() {
    local tool="$1" file_path="$2" path="$3"
    jq -cn --arg tool "$tool" --arg file_path "$file_path" --arg path "$path" \
        '{tool_name:$tool,tool_input:{file_path:$file_path,path:$path}}'
}
failures=0
run_case nested-protected 2 "$(patch_payload '*** Add File: ' cosyvoice2_llm_reference/license_gate_manifest.json "$ROOT/tools/parity")" || failures=$((failures+1))
run_case nested-exec-protected 2 "$(patch_payload '*** Add File: ' cosyvoice2_llm_reference/license_gate_manifest.json)" "$ROOT/tools/parity" || failures=$((failures+1))
run_case external-owner-absolute 2 "$(patch_payload '*** Add File: ' "$ROOT/tools/parity/cosyvoice2_llm_reference/license_gate_manifest.json" "$TEST_ROOT")" || failures=$((failures+1))
run_case actual-external-cwd-absolute 2 "$(patch_payload '*** Add File: ' "$ROOT/tools/parity/cosyvoice2_llm_reference/license_gate_manifest.json")" "$TEST_ROOT" || failures=$((failures+1))
run_case safe-absolute 0 "$(patch_payload '*** Add File: ' "$ROOT/.codex/hooks/guard-protected-path.sh")" || failures=$((failures+1))
run_case eof-marker 0 "$(eof_payload)" || failures=$((failures+1))
run_case body-not-shell 0 "$(patch_payload '*** Add File: ' /private/tmp/vokra-body __MISSING__ '+git add tools/parity/cosyvoice2_llm_reference/license_gate_manifest.json; rm -rf /')" || failures=$((failures+1))
run_case add-protected 2 "$(patch_payload '*** Add File: ' tools/parity/cosyvoice2_llm_reference/new.txt)" || failures=$((failures+1))
run_case update-protected 2 "$(patch_payload '*** Update File: ' tools/parity/cosyvoice2_llm_reference/license_gate_manifest.json)" || failures=$((failures+1))
run_case delete-protected 2 "$(patch_payload '*** Delete File: ' tools/parity/cosyvoice2_llm_reference/license_gate_manifest.json)" || failures=$((failures+1))
run_case move-to-protected 2 "$(patch_payload '*** Move to: ' tools/parity/cosyvoice2_llm_reference/new.txt)" || failures=$((failures+1))
run_case leading-space-absolute 2 "$(patch_payload '*** Update File:   ' "$ROOT/tools/parity/cosyvoice2_llm_reference/license_gate_manifest.json")" || failures=$((failures+1))
run_case leading-space-relative 2 "$(patch_payload '*** Add File:  ' tools/parity/cosyvoice2_llm_reference/new.txt)" || failures=$((failures+1))
run_case tab-relative 2 "$(patch_payload $'*** Add File:\t' tools/parity/cosyvoice2_llm_reference/new.txt)" || failures=$((failures+1))
run_case move-source-safe-destination-protected 2 "$(jq -cn --arg command $'*** Begin Patch\n*** Move from: /private/tmp/vokra-source\n*** Move to: tools/parity/cosyvoice2_llm_reference/new.txt\n*** End Patch' '{tool_name:"apply_patch",tool_input:{command:$command}}')" || failures=$((failures+1))
run_case tool-trailing-lf 2 "$(jq -cn --arg command 'echo ok' '{tool_name:"Bash\n",tool_input:{command:$command}}')" || failures=$((failures+1))
run_case tool-nul 2 "$(jq -cn --arg command 'echo ok' '{tool_name:("Bash"+"\u0000"),tool_input:{command:$command}}')" || failures=$((failures+1))
run_case unknown-tool 2 "$(jq -cn --arg command 'echo ok' '{tool_name:"NoSuchTool",tool_input:{command:$command}}')" || failures=$((failures+1))
run_case mixed-case-relative-protected 2 "$(patch_payload '*** Add File: ' ToOlS/PaRiTy/CoSyVoIcE2_LlM_ReFeReNcE/new.txt)" || failures=$((failures+1))
upper_owner="$(printf '%s' "$ROOT/tools/parity/cosyvoice2_llm_reference/license_gate_manifest.json" | tr '[:lower:]' '[:upper:]')"
run_case uppercase-absolute-protected 2 "$(patch_payload '*** Add File: ' "$upper_owner")" || failures=$((failures+1))
run_case mixed-case-safe 0 "$(patch_payload '*** Add File: ' /private/tmp/Vokra-Mixed-Safe)" || failures=$((failures+1))
for tool in Edit Write; do
    run_case "${tool}-symlink-component" 2 "$(target_file_payload "$tool" "$TEST_ROOT/repo-link/tools/parity/cosyvoice2_llm_reference/new.txt")" || failures=$((failures+1))
    run_case "${tool}-traversal-component" 2 "$(target_file_payload "$tool" "$TEST_ROOT/../vokra-extra-safe")" || failures=$((failures+1))
    run_case "${tool}-relative-cwd" 2 "$(target_file_payload "$tool" /private/tmp/vokra-safe .)" || failures=$((failures+1))
    run_case "${tool}-control-character-file-path" 2 "$(target_file_payload "$tool" $'/private/tmp/vokra\nunsafe')" || failures=$((failures+1))
    run_case "${tool}-control-character-path" 2 "$(target_path_payload "$tool" $'/private/tmp/vokra\nunsafe')" || failures=$((failures+1))
    run_case "${tool}-file-path-precedence-protected" 2 "$(target_pair_payload "$tool" "$ROOT/tools/parity/cosyvoice2_llm_reference/new.txt" /private/tmp/vokra-safe)" || failures=$((failures+1))
    run_case "${tool}-file-path-precedence-safe" 0 "$(target_pair_payload "$tool" /private/tmp/vokra-safe "$ROOT/tools/parity/cosyvoice2_llm_reference/new.txt")" || failures=$((failures+1))
done
if [ "$failures" -eq 0 ]; then echo focused-regression:PASS; else exit 1; fi
