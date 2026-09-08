# Fun-CosyVoice3 dependency-route decision (2026-09-08)

This note records the model-free decision for
`FunAudioLLM/Fun-CosyVoice3-0.5B-2512`. It does not authorize checkpoint
acquisition, model execution, conversion, publication, or Apple verification.

## Primary-source facts

The source checkout is pinned to CosyVoice revision
`0d990d60740bf174904a5185cce910b847bd3684`; the Matcha-TTS submodule is pinned
to `dd9105b34bf2be2230f4aa1e4769fb586a3c824e`. The official model card and
source repository are:

- <https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512>
- <https://github.com/FunAudioLLM/CosyVoice/tree/0d990d60740bf174904a5185cce910b847bd3684>
- <https://raw.githubusercontent.com/FunAudioLLM/CosyVoice/0d990d60740bf174904a5185cce910b847bd3684/requirements.txt>
- <https://github.com/shivammehta25/Matcha-TTS/tree/dd9105b34bf2be2230f4aa1e4769fb586a3c824e>
- <https://pypi.org/pypi/librosa/0.10.2/json>

The official requirements declare `librosa==0.10.2`. The pinned Matcha audio
implementation imports `librosa.filters.mel`; the CosyVoice3 flow/HiFT path
uses that mel implementation. PyPI's authenticated `librosa==0.10.2` metadata
declares `soxr>=0.3.2`. `soxr` is forbidden by Vokra's license policy.

Therefore an authenticated Python 3.12 lock for the official complete
composite cannot be accepted while this dependency declaration remains. A
larger VAST machine or Scaleway Apple host cannot resolve a license-closure
failure.

## Boundary

The LLM source roles may be inspected as a component contract, but that route
does not produce PCM and is not a CosyVoice3 TTS implementation. The flow/HiFT
component remains blocked by the official mel/frontend dependency closure. No
narrower implementation may be labelled as the complete official composite.

`tools/parity/cosyvoice3_source_audit.py` records these facts and verifies a
clean pinned source checkout, exact dependency inventory, absence of a
forbidden `uv.lock`, and the separated component statuses. Its self-test is
stdlib-only and model-free. The VAST and Apple workers call the self-test but
remain fail-closed before source/model acquisition until an owner-approved,
dependency-clean route exists.

