#!/usr/bin/env bash
# Build the smallest honest nightly-full-parity matrix for the current event.
#
# PR/schedule runs always execute the committed Silero fixture and add only
# legs whose repository staging variable is configured. Manual all-leg runs
# retain the full diagnostic matrix; a manual selector emits exactly one leg.
# This keeps skip provenance in the plan summary without allocating six
# redundant runners merely to discover that an owner variable is unset.

set -euo pipefail

event=${GITHUB_EVENT_NAME:-pull_request}
requested=${INPUT_LEG:-}
matrix='[]'
selected=''
omitted=''

append_leg() {
  local leg=$1 model_name gguf_var fixture_dir test_crate test_flag test_filter needs_gguf
  case "$leg" in
    kokoro)
      model_name='Kokoro-82M'
      gguf_var='VOKRA_NIGHTLY_KOKORO_GGUF'
      fixture_dir='tests/parity/kokoro'
      test_crate='vokra-models'
      test_flag='--test parity_kokoro'
      test_filter=''
      needs_gguf='true'
      ;;
    whisper-base)
      model_name='Whisper base (log-mel + encoder/decoder/greedy fixtures)'
      gguf_var='VOKRA_NIGHTLY_WHISPER_BASE_GGUF'
      fixture_dir='tests/parity/whisper_base'
      test_crate='vokra-models'
      test_flag='--test parity_whisper'
      test_filter=''
      needs_gguf='true'
      ;;
    silero-vad-v5)
      model_name='Silero VAD v5 (fixture-only always-on)'
      gguf_var=''
      fixture_dir='tests/parity/silero_vad'
      test_crate='vokra-models'
      test_flag='--lib'
      test_filter='silero_vad::parity'
      needs_gguf='false'
      ;;
    piper-plus)
      model_name='piper-plus v7 (JA/multilingual zero-shot voice fixture)'
      gguf_var='VOKRA_NIGHTLY_PIPER_V7_GGUF'
      fixture_dir='tests/parity/piper_plus_v7'
      test_crate='vokra-models'
      test_flag='--lib'
      test_filter='piper_plus::parity_v7'
      needs_gguf='true'
      ;;
    campplus)
      model_name='CAM++ speaker encoder (fbank80 -> 192-d embedding)'
      gguf_var='VOKRA_NIGHTLY_CAMPLUS_GGUF'
      fixture_dir='tests/parity/camplus'
      test_crate='vokra-models'
      test_flag='--lib'
      test_filter='speaker::parity'
      needs_gguf='true'
      ;;
    campplus-capi)
      model_name='CAM++ through the C ABI (vokra_speaker_embed + backend wiring)'
      gguf_var='VOKRA_NIGHTLY_CAMPLUS_GGUF'
      fixture_dir='tests/parity/camplus'
      test_crate='vokra-capi'
      test_flag=''
      test_filter=''
      needs_gguf='true'
      ;;
    dac)
      model_name='DAC 24kHz codec (RVQ decode against sliced fixtures)'
      gguf_var='VOKRA_NIGHTLY_DAC_GGUF'
      fixture_dir='tests/parity/dac'
      test_crate='vokra-models'
      test_flag='--test real_codec_parity'
      test_filter='real_dac'
      needs_gguf='true'
      ;;
    *)
      echo "ERROR: unknown parity leg: $leg" >&2
      return 2
      ;;
  esac

  local item
  item=$(jq -cn \
    --arg leg "$leg" \
    --arg model_name "$model_name" \
    --arg gguf_var "$gguf_var" \
    --arg fixture_dir "$fixture_dir" \
    --arg test_crate "$test_crate" \
    --arg test_flag "$test_flag" \
    --arg test_filter "$test_filter" \
    --arg needs_gguf "$needs_gguf" \
    '{leg:$leg,model_name:$model_name,gguf_var:$gguf_var,fixture_dir:$fixture_dir,test_crate:$test_crate,test_flag:$test_flag,test_filter:$test_filter,needs_gguf:$needs_gguf}')
  matrix=$(jq -cn --argjson matrix "$matrix" --argjson item "$item" '$matrix + [$item]')
  selected="${selected}${selected:+, }${leg}"
}

omit_leg() {
  omitted="${omitted}${omitted:+, }$1"
}

if [[ -n "$requested" ]]; then
  append_leg "$requested"
elif [[ "$event" == 'workflow_dispatch' ]]; then
  for leg in kokoro whisper-base silero-vad-v5 piper-plus campplus campplus-capi dac; do
    append_leg "$leg"
  done
else
  append_leg silero-vad-v5
  if [[ -n "${VOKRA_NIGHTLY_KOKORO_GGUF:-}" ]]; then append_leg kokoro; else omit_leg kokoro; fi
  if [[ -n "${VOKRA_NIGHTLY_WHISPER_BASE_GGUF:-}" ]]; then append_leg whisper-base; else omit_leg whisper-base; fi
  if [[ -n "${VOKRA_NIGHTLY_PIPER_V7_GGUF:-}" ]]; then append_leg piper-plus; else omit_leg piper-plus; fi
  if [[ -n "${VOKRA_NIGHTLY_CAMPLUS_GGUF:-}" ]]; then
    append_leg campplus
    append_leg campplus-capi
  else
    omit_leg campplus
    omit_leg campplus-capi
  fi
  if [[ -n "${VOKRA_NIGHTLY_DAC_GGUF:-}" ]]; then append_leg dac; else omit_leg dac; fi
fi

output=$(jq -cn --argjson include "$matrix" '{include:$include}')
if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
  printf 'matrix=%s\n' "$output" >> "$GITHUB_OUTPUT"
fi
if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
  {
    echo '### nightly-full-parity execution plan'
    echo
    echo "- event: \`$event\`"
    echo "- selected: \`$selected\`"
    if [[ -n "$omitted" ]]; then
      echo "- omitted before runner allocation (staging variable unset): \`$omitted\`"
      echo
      echo 'Omitted legs did not run parity and are not reported as passing.'
    fi
  } >> "$GITHUB_STEP_SUMMARY"
fi
printf '%s\n' "$output"
