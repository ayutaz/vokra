# Workflow Python/uv migration — 2026-08-18

## Outcome

The legacy GitHub Actions Python surface has been migrated to uv on
`agent/reconcile-docs-and-operations-2026-08-18`.

The opening inventory found **24 workflow files**, **36 jobs**, and **152 shell
invocations** that bypassed uv through bare `python`, `python3`, `pip`,
`pytest`, direct venv executables, or activation scripts. The closing static
inventory reports zero such invocations. There are no compatibility
exceptions.

This task changes workflow provisioning and command launch only. It does not
change model weights, numerical tolerances, release permissions, trigger
schedules, or branch-protection contexts.

## Migration plan and disposition

| Workstream | Baseline | Disposition |
|---|---:|---|
| Stdlib-only helpers and small server commands | 7 workflows / 8 jobs | Complete: pinned setup-uv plus `uv run --no-project --python 3.12 python ...` |
| Isolated dependency environments and parity helpers | 16 workflows / 18 jobs | Complete: `uv venv`, `uv pip --python`, no activation, explicit interpreter on `uv run` |
| Release pipeline helpers | 1 workflow / 10 jobs | Complete: every Python-using release job installs pinned uv; Twine uses an ephemeral `--with twine` environment |
| Regression prevention | repository-wide | Complete: workflow hygiene rejects bare Python/pip/pytest, direct venv executables, and venv activation |

The 24 migrated files are:

- lightweight or stdlib use: `bench-baseline-capture.yml`, `ci-platform.yml`,
  `gpu-cuda-rtf.yml`, `nightly-tier2-device.yml`, `nightly-webgl.yml`,
  `release-cadence.yml`, and `secret-scan.yml`;
- dependency-bearing CI/nightly use: `ci.yml`, `ci-quality.yml`,
  `corpus-drift-detector.yml`, `nightly-asr-wer.yml`,
  `pins-sync-check.yml`, and `web-wasm.yml`;
- real-parity use: `parity-moshi-real.yml`, `parity-qwen3-tts-real.yml`,
  `parity-rvq-real.yml`, `parity-tts-dac-real.yml`,
  `parity-tts-hiftnet-real.yml`, `parity-tts-japanese-real.yml`,
  `parity-utmos.yml`, `parity-voxtral-real.yml`,
  `parity-whisper-extras-real.yml`, and `parity-whisper-real.yml`;
- release use: `release.yml`.

## Canonical recipe

Each Python-using job installs the repository-pinned action:

```yaml
- uses: astral-sh/setup-uv@c771a70e6277c0a99b617c7a806ffedaca235ff9 # v9.0.0
```

Stdlib-only commands run without a project environment:

```sh
uv run --no-project --python 3.12 python path/to/helper.py
```

Jobs with third-party dependencies use a job-local interpreter without shell
activation:

```sh
uv venv --python 3.12 /tmp/vokra-example
uv pip install --python /tmp/vokra-example/bin/python package-name
uv run --no-project --python /tmp/vokra-example/bin/python python path/to/helper.py
```

Cross-platform jobs resolve the venv interpreter once (`bin/python` on Unix,
`Scripts/python.exe` on Windows), export that path through `GITHUB_ENV`, and
pass it explicitly to every `uv pip` and `uv run` call. One-shot release tools
use `uv run --with <tool>` instead of mutating the runner's user site.

`actions/setup-python` remains in jobs where another action or matrix contract
expects the selected CPython installation. It is no longer used as permission
to call pip or Python directly.

## Regression gate

`scripts/check-workflow-hygiene.sh` now checks every block and scalar `run:`
entry. It fails on:

- bare `python`, `python3`, `pip`, `pip3`, or `pytest` commands;
- direct `<venv>/bin/python`, `<venv>/bin/pip`, or Windows equivalents;
- `source <venv>/bin/activate` and equivalent activation forms.

The scanner joins shell continuation lines, recognises command substitutions,
and skips heredoc payloads after validating their uv launcher. Its self-test
contains both red fixtures and the accepted uv-managed forms.

## Completion checks

```sh
UV_CACHE_DIR=/private/tmp/vokra-workflow-audit-uv-cache \
  bash scripts/check-workflow-hygiene.sh --self-test
UV_CACHE_DIR=/private/tmp/vokra-workflow-audit-uv-cache \
  bash scripts/check-workflow-hygiene.sh
git diff --check
```

The full workflow check covers all 40 workflow files and all 32 cron entries;
success means no cron collisions, dangling `needs`, shell syntax errors,
unquotable YAML scalars, or bare Python tooling remain.
