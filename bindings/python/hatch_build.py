"""Hatchling custom build hook — force a platform-tagged wheel.

The Vokra Python binding is a hybrid distribution: pure-Python at the Python
layer, plus a pre-built native shared library (`libvokra.so` /
`libvokra.dylib` / `vokra.dll`) that CI injects under
`src/vokra/_lib/` via `CIBW_BEFORE_BUILD_*` (see
`.github/workflows/ci.yml`'s `python-wheel-build` job). Without this hook
hatchling emits a `py3-none-any` wheel — a pure-Python tag — which
cibuildwheel v2.20+ hard-rejects with

    Build failed because a pure Python wheel was generated.

The check exists to keep cibuildwheel-users from accidentally producing
tag-mismatched wheels for a project that intended to compile C. Our
intent IS platform-specific (one wheel per OS × arch), so this hook
tells hatchling to infer the correct tag from the current interpreter
and mark the wheel as non-pure.

Registered under `[tool.hatch.build.targets.wheel.hooks.custom]` in
`pyproject.toml`.
"""

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    """Force `pure_python = False` and `infer_tag = True` on every wheel."""

    PLUGIN_NAME = "custom"

    def initialize(self, version: str, build_data: dict) -> None:  # noqa: D401 - hatchling contract
        # `pure_python = False` sets the wheel's Root-Is-Purelib metadata to
        # False (WHEEL file), signalling that native code is bundled. The tag
        # infers from the running Python (`cp39-cp39-linux_x86_64` etc.),
        # which is exactly what cibuildwheel expects for a platform wheel.
        # We do NOT hard-code a tag here: cibuildwheel launches the build
        # inside the target Python's environment, so `infer_tag` picks up
        # the right (Python, ABI, platform) triple for every matrix cell.
        build_data["pure_python"] = False
        build_data["infer_tag"] = True
