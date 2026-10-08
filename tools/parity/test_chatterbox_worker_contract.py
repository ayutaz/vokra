#!/usr/bin/env -S uv run --no-project --offline --python 3.12 python
"""Focused, dependency-free contract tests for blocked Chatterbox workers.

Run with ``--self-test`` to execute the offline fake-tool contract suite.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
import tomllib


ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "tools/parity/chatterbox_t3_reference.py"
REFERENCE_PROJECT = ROOT / "tools/parity/chatterbox_t3"
LOCK = ROOT / "tools/parity/chatterbox_t3/uv.lock"
WORKERS = (
    ROOT / "scripts/publish/vast-ai/run-chatterbox-t3-validation.sh",
    ROOT / "scripts/publish/vast-ai/run-chatterbox-family-inspection.sh",
)
TEMP_ROOT = Path(tempfile.gettempdir()).resolve()
LOCK_SHA256 = hashlib.sha256(LOCK.read_bytes()).hexdigest()
EXPECTED_CORE = {
    "numpy": "1.26.4",
    "huggingface-hub": "1.27.0",
    "einops": "0.8.2",
    "safetensors": "0.5.3",
    "torch": "2.13.0",
    "torchaudio": "2.11.0",
    "tqdm": "4.67.1",
    "transformers": "5.10.4",
}
EXPECTED_CPU = {"torch": "2.13.0+cpu", "torchaudio": "2.11.0+cpu"}
EXPECTED_CPU_INDEX = "https://download.pytorch.org/whl/cpu"
EXPECTED_HEAD = "0" * 40
SCOPE = (
    '{"source_url":"https://github.com/resemble-ai/chatterbox.git",'
    '"source_revision":"5de7a54aa4e5e2baadb0182dde554908b48b85c2",'
    '"variants":["base","nano","turbo"]}'
)


def package_rows_sha256(lock: dict[str, object]) -> str:
    rows = []
    for package in lock["package"]:  # type: ignore[index]
        source = package["source"]  # type: ignore[index]
        rows.append(
            {
                "name": package["name"],  # type: ignore[index]
                "version": package["version"],  # type: ignore[index]
                "source": {key: source[key] for key in sorted(source)},  # type: ignore[index]
                "markers": sorted(package.get("resolution-markers", [])),  # type: ignore[union-attr]
            }
        )
    rows.sort(
        key=lambda row: (
            row["name"],
            row["version"],
            json.dumps(row["source"], sort_keys=True),
            row["markers"],
        )
    )
    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def expected_audit(
    lock_sha: str = LOCK_SHA256,
    rows_sha: str | None = None,
    *,
    core: dict[str, str] | None = None,
    cpu: dict[str, str] | None = None,
    cpu_index: str = EXPECTED_CPU_INDEX,
    audit_lock: str | None = None,
) -> str:
    if rows_sha is None:
        rows_sha = package_rows_sha256(tomllib.loads(LOCK.read_text(encoding="utf-8")))
    if core is None:
        core = EXPECTED_CORE
    if cpu is None:
        cpu = EXPECTED_CPU
    if audit_lock is None:
        audit_lock = lock_sha
    value = {
        "reference_environment": {
            "sha256": lock_sha,
            "package_rows_sha256": rows_sha,
            "core_versions": core,
            "cpu_distribution_versions": cpu,
            "cpu_index": cpu_index,
        },
        "license_audit": {
            "lock_sha256": audit_lock,
            "status": "BLOCKED_UNRESOLVED",
        },
    }
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def duplicate_audit(nested: bool) -> str:
    valid = json.loads(expected_audit())
    environment = valid["reference_environment"]
    audit = valid["license_audit"]
    audit_text = json.dumps(audit, sort_keys=True, separators=(",", ":"))
    if nested:
        environment_text = (
            '{"sha256":"'
            + LOCK_SHA256
            + '","sha256":"'
            + LOCK_SHA256
            + '","package_rows_sha256":"'
            + environment["package_rows_sha256"]
            + '","core_versions":'
            + json.dumps(environment["core_versions"], sort_keys=True, separators=(",", ":"))
            + ',"cpu_distribution_versions":'
            + json.dumps(environment["cpu_distribution_versions"], sort_keys=True, separators=(",", ":"))
            + ',"cpu_index":"'
            + EXPECTED_CPU_INDEX
            + '"}'
        )
    else:
        environment_text = json.dumps(environment, sort_keys=True, separators=(",", ":"))
    return (
        '{"reference_environment":'
        + environment_text
        + (',"reference_environment":' + environment_text if not nested else "")
        + ',"license_audit":'
        + audit_text
        + "}"
    )


def write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def inline_validator_sha(worker: Path, marker: str) -> str:
    blocks = re.findall(r"<<'PY'[^\n]*\n(.*?)\nPY", worker.read_text(encoding="utf-8"), re.DOTALL)
    source = next((block for block in blocks if marker in block), None)
    if source is None:
        raise AssertionError(f"{worker.name} is missing inline validator {marker!r}")
    return hashlib.sha256((source + "\n").encode()).hexdigest()


def approval_fixture(path: Path) -> str:
    scope_sha = hashlib.sha256(SCOPE.encode()).hexdigest()
    value = {
        "schema": "chatterbox-vast-approval-v1",
        "decision": "BLOCKED",
        "status": "BLOCKED",
        "evidence_stage": "INSPECTION_ONLY",
        "no_upload": True,
        "expected_head": EXPECTED_HEAD,
        "source_url": "https://github.com/resemble-ai/chatterbox.git",
        "source_revision": "5de7a54aa4e5e2baadb0182dde554908b48b85c2",
        "variants": ["base", "nano", "turbo"],
        "scope_sha256": scope_sha,
    }
    path.write_text(json.dumps(value, separators=(",", ":")) + "\n", encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fake_uv_content(real_uv: str) -> str:
    return (
        "#!/bin/sh\n"
        "printf 'uv %s\\n' \"$*\" >> \"$FAKE_EVENT_LOG\"\n"
        "if [ \"$*\" = \"$FAKE_AUDIT_ARGS\" ]; then\n"
        "  printf '%s' \"$FAKE_AUDIT_OUTPUT\"; exit \"${FAKE_AUDIT_RC:-2}\"\n"
        "fi\n"
        "expected=\"\"\n"
        "if [ \"$*\" = \"$FAKE_APPROVAL_ARGS\" ]; then expected=\"$FAKE_APPROVAL_SHA\"; fi\n"
        "if [ \"$*\" = \"$FAKE_VALIDATOR_ARGS\" ]; then expected=\"$FAKE_VALIDATOR_SHA\"; fi\n"
        "if [ -n \"$expected\" ]; then\n"
        "  source=\"$FAKE_EVENT_LOG.stdin.$$\"\n"
        "  cat >\"$source\"\n"
        "  actual=$(shasum -a 256 \"$source\" | awk '{print $1}')\n"
        "  if [ \"$actual\" != \"$expected\" ]; then\n"
        "    printf 'UNEXPECTED_UV_STDIN %s\\n' \"$*\" >> \"$FAKE_EVENT_LOG\"\n"
        "    /bin/rm -f \"$source\"\n"
        "    exit 97\n"
        "  fi\n"
        "  printf 'REAL_UV_DISPATCH %s\\n' \"$*\" >> \"$FAKE_EVENT_LOG\"\n"
        "  "
        + shlex.quote(real_uv)
        + " \"$@\" <\"$source\"\n"
        "  status=$?; /bin/rm -f \"$source\"; exit $status\n"
        "fi\n"
        "printf 'UNEXPECTED_UV %s\\n' \"$*\" >> \"$FAKE_EVENT_LOG\"\n"
        "exit 97\n"
    )


def probe_fake_uv_deny() -> None:
    with tempfile.TemporaryDirectory(prefix="vokra-chatterbox-uv-deny-", dir=str(TEMP_ROOT)) as directory:
        root = Path(directory)
        event_log = root / "events.log"
        fake_uv = root / "uv"
        real_uv = shutil.which("uv")
        if not real_uv:
            raise AssertionError("uv is required for the focused contract test")
        write_executable(fake_uv, fake_uv_content(real_uv))
        env = os.environ.copy()
        env.update({"PATH": f"{root}{os.pathsep}{env['PATH']}", "FAKE_EVENT_LOG": str(event_log)})
        try:
            result = subprocess.run(
                [str(fake_uv), "run", "--unexpected-contract-probe"],
                env=env,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except subprocess.TimeoutExpired as error:
            raise AssertionError("fake UV deny probe exceeded focused-test timeout") from error
        events = event_log.read_text(encoding="utf-8") if event_log.exists() else ""
        if result.returncode != 97 or "UNEXPECTED_UV" not in events or "REAL_UV_DISPATCH" in events:
            raise AssertionError(f"fake UV deny probe was not causal: rc={result.returncode}, events={events}")


def run_blocked_worker(
    worker: Path,
    audit_output: str,
    expected_message: str,
    *,
    audit_rc: int = 2,
) -> None:
    with tempfile.TemporaryDirectory(
        prefix="vokra-chatterbox-worker-contract-", dir=str(TEMP_ROOT)
    ) as directory:
        root = Path(directory) / "root"
        root.mkdir()
        (root / ".git").mkdir()
        reference_project = root / "tools/parity/chatterbox_t3"
        reference_project.mkdir(parents=True)
        (reference_project / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
        (reference_project / "uv.lock").write_text("version = 1\n", encoding="utf-8")
        approval = root / "approval.json"
        approval_digest = approval_fixture(approval)
        event_log = Path(directory) / "events.log"
        fake_bin = Path(directory) / "bin"
        fake_bin.mkdir()
        real_uv = shutil.which("uv")
        if not real_uv:
            raise AssertionError("uv is required for the focused contract test")
        reference = root / "tools/parity/chatterbox_t3_reference.py"
        audit_args = " ".join(
            [
                "run",
                "--no-cache",
                "--no-project",
                "--offline",
                "--python",
                "3.12",
                "python",
                str(reference),
                "--license-audit",
            ]
        )
        scope_sha = hashlib.sha256(SCOPE.encode()).hexdigest()
        approval_args = " ".join(
            [
                "run",
                "--no-cache",
                "--no-project",
                "--offline",
                "--python",
                "3.12",
                "python",
                "-",
                str(approval),
                EXPECTED_HEAD,
                scope_sha,
            ]
        )
        validator_args = " ".join(
            [
                "run",
                "--no-cache",
                "--no-project",
                "--offline",
                "--python",
                "3.12",
                "python",
                "-",
                LOCK_SHA256,
                package_rows_sha256(tomllib.loads(LOCK.read_text(encoding="utf-8"))),
                "2.13.0",
                "2.11.0",
                "2.13.0+cpu",
                "2.11.0+cpu",
            ]
        )
        approval_validator_sha = inline_validator_sha(worker, "approval schema is not exact")
        validator_sha = inline_validator_sha(worker, "audit envelope schema drifted")
        write_executable(fake_bin / "uv", fake_uv_content(real_uv))
        write_executable(
            fake_bin / "git",
            "#!/bin/sh\n"
            "printf 'git %s\\n' \"$*\" >> \"$FAKE_EVENT_LOG\"\n"
            "case \" $* \" in\n"
            "  *' status --porcelain --untracked-files=all '*) exit 0 ;;\n"
            "  *' rev-parse HEAD '*) printf '%s\\n' \"$EXPECTED_HEAD\"; exit 0 ;;\n"
            "  *) printf 'unexpected git operation: %s\\n' \"$*\" >&2; exit 99 ;;\n"
            "esac\n",
        )
        for command in ("cargo", "curl", "wget", "ssh", "scp", "hf", "huggingface-cli", "vastai", "aws"):
            write_executable(
                fake_bin / command,
                "#!/bin/sh\n"
                f"printf 'FORBIDDEN {command} %s\\n' \"$*\" >> \"$FAKE_EVENT_LOG\"\n"
                "exit 98\n",
            )
        work = Path(directory) / "work"
        env = os.environ.copy()
        env.update(
            {
                "PATH": f"{fake_bin}{os.pathsep}{env['PATH']}",
                "FAKE_AUDIT_OUTPUT": audit_output,
                "FAKE_AUDIT_RC": str(audit_rc),
                "FAKE_AUDIT_ARGS": audit_args,
                "FAKE_APPROVAL_ARGS": approval_args,
                "FAKE_VALIDATOR_ARGS": validator_args,
                "FAKE_APPROVAL_SHA": approval_validator_sha,
                "FAKE_VALIDATOR_SHA": validator_sha,
                "FAKE_EVENT_LOG": str(event_log),
                "EXPECTED_HEAD": EXPECTED_HEAD,
                "VOKRA_ROOT": str(root),
                "UV_NO_CACHE": "1",
                "UV_OFFLINE": "1",
                "CHATTERBOX_T3_UV_CACHE_DIR": str(Path(directory) / "cache"),
                "CHATTERBOX_UV_CACHE_DIR": str(Path(directory) / "cache"),
            }
        )
        args = ["bash", str(worker)]
        if worker.name == "run-chatterbox-t3-validation.sh":
            env["CHATTERBOX_T3_WORK_DIR"] = str(work)
        else:
            args.extend(["--work-dir", str(work)])
        args.extend(
            [
                "--approval-evidence",
                str(approval),
                "--approval-sha256",
                approval_digest,
                "--expected-head",
                EXPECTED_HEAD,
            ]
        )
        try:
            result = subprocess.run(args, env=env, capture_output=True, text=True, timeout=15)
        except subprocess.TimeoutExpired as error:
            raise AssertionError(f"{worker.name} exceeded focused-test timeout") from error
        combined = result.stdout + result.stderr
        if result.returncode != 2:
            raise AssertionError(f"{worker.name} returned {result.returncode}: {combined}")
        if expected_message not in combined:
            raise AssertionError(f"{worker.name} omitted {expected_message!r}: {combined}")
        if work.exists():
            raise AssertionError(f"{worker.name} created work before blocked preflight")
        events = event_log.read_text(encoding="utf-8") if event_log.exists() else ""
        forbidden = ("FORBIDDEN", "UNEXPECTED_UV", "UNEXPECTED_UV_STDIN", "unexpected git operation")
        if any(token in events for token in forbidden):
            raise AssertionError(f"{worker.name} reached forbidden work/provider operation: {events}")


def main(argv: list[str]) -> None:
    if argv not in ([], ["--self-test"]):
        raise SystemExit("usage: test_chatterbox_worker_contract.py [--self-test]")
    probe_fake_uv_deny()
    lock = tomllib.loads(LOCK.read_text(encoding="utf-8"))
    rows_sha = package_rows_sha256(lock)
    project_text = (REFERENCE_PROJECT / "pyproject.toml").read_text(encoding="utf-8")
    if LOCK_SHA256 != "3c1a295bd6d45e6b83f7a182a4421bbb7cc5904a554f4305b9f7d08d1e92029d":
        raise AssertionError("committed Chatterbox lock identity changed")
    if rows_sha != "a47f8a74ef9d990289002eaccd346d7d5cbbc9d1480d1213596bc01b0ed36c24":
        raise AssertionError("committed Chatterbox package rows changed")
    if 'source_exact_dependencies = [' not in project_text or '"torch==2.6.0"' not in project_text:
        raise AssertionError("official source Torch 2.6.0 fact was not retained")
    old_tokens = ("2fa167c5d2587d7fef6ac2c589a193f9cbd9a8d4495e22487a53a7ba5da6798f", "1feb25cd45b465dc7fb37dce07599c16218584211640357d541ba969917342d8")
    for worker in WORKERS:
        text = worker.read_text(encoding="utf-8")
        if any(token in text for token in old_tokens):
            raise AssertionError(f"stale identity retained in {worker}")
        for token in (
            LOCK_SHA256,
            rows_sha,
            "2.13.0+cpu",
            "2.11.0+cpu",
            "BLOCKED_UNRESOLVED",
            "license audit identity is stale or malformed",
            "BLOCKED_APPROVAL/INSPECTION_ONLY",
        ):
            if token not in text:
                raise AssertionError(f"{worker} is missing {token}")
    t3_text = WORKERS[0].read_text(encoding="utf-8")
    if "AUTHENTICATED_CLEAR" not in t3_text:
        raise AssertionError("dormant post-acquisition clearance proof was weakened")

    stale_core = dict(EXPECTED_CORE)
    stale_core["torch"] = "2.6.0"
    stale_cpu = dict(EXPECTED_CPU)
    stale_cpu["torch"] = "2.6.0+cpu"
    current = expected_audit(lock_sha=LOCK_SHA256, rows_sha=rows_sha)
    cases = (
        (current, "dependency license audit is unresolved", 2),
        (expected_audit(lock_sha="0" * 64, rows_sha=rows_sha), "license audit identity is stale or malformed", 2),
        (expected_audit(rows_sha="0" * 64), "license audit identity is stale or malformed", 2),
        (expected_audit(core=stale_core), "license audit identity is stale or malformed", 2),
        (expected_audit(cpu=stale_cpu), "license audit identity is stale or malformed", 2),
        (expected_audit(cpu_index="https://example.invalid/cpu"), "license audit identity is stale or malformed", 2),
        (expected_audit(audit_lock="0" * 64), "license audit identity is stale or malformed", 2),
        (duplicate_audit(nested=False), "license audit identity is stale or malformed", 2),
        (duplicate_audit(nested=True), "license audit identity is stale or malformed", 2),
        ("not-json", "license audit identity is stale or malformed", 2),
        (current, "license audit unexpectedly cleared before owner approval", 0),
        (current, "dependency license audit command failed unexpectedly (exit 3)", 3),
    )
    for worker in WORKERS:
        for audit_output, expected_message, audit_rc in cases:
            run_blocked_worker(worker, audit_output, expected_message, audit_rc=audit_rc)
    print("test_chatterbox_worker_contract.py: PASS (UV deny probe + 24 causal blocked-preflight cases; no work/provider reached)")


if __name__ == "__main__":
    main(sys.argv[1:])
