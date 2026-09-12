# Zonos-v0.1 reference environment

This is a dedicated Python 3.12 project for the exact upstream Zonos commit
`bc40d98e1e1ab54fc65c483be127a90e3c7c0645`. It is separate from
`tools/parity/pyproject.toml` so unrelated parity tools cannot silently widen
the reference closure.

The project intentionally contains only the pinned import/runtime closure for
`Zonos.from_local`, the official conditioning modules, generation, and the
official DAC decoder. It does not include `phonemizer`, eSpeak, `librosa`,
`soundfile`, `soxr`, `cffi`, `libsndfile`, ONNX, or any optional hybrid/UI
packages. Zonos conditioning packets carry offline phoneme IDs; if the pinned
source attempts text phonemization, the reference import policy raises an
explicit error.

`uv.lock` is generated with `uv lock` and is bound by SHA-256 in every Zonos
consumer. The dependency audit remains `BLOCKED_UNREVIEWED_TRANSITIVE` until
an owner/legal decision reviews the Linux x86_64 closure, native ELF files,
and publisher LICENSE/NOTICE bytes. No model/source/checkpoint acquisition,
execution, or publication is authorized by this project alone; all workers
remain `NO_UPLOAD`.
