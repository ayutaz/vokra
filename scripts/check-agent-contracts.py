#!/usr/bin/env -S uv run --script
"""Validate the repository contract used by Codex agents.

This is intentionally a small, dependency-free checker.  It checks the
machine-readable parts of the agent configuration and the discoverability
contract for repository skills without importing or executing any skill.
"""

from __future__ import annotations

import argparse
import re
import tempfile
import tomllib
from pathlib import Path


MAX_AGENTS_BYTES = 32 * 1024
MAX_SKILL_NAME_LENGTH = 64
MAX_DESCRIPTION_LENGTH = 1024
SKILL_NAME = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
SKILL_TOKEN = re.compile(r"\$([a-z0-9][a-z0-9-]*)")
MARKDOWN_LINK = re.compile(r"!??\[[^\]]*\]\(([^)]+)\)")


def fail(message: str, errors: list[str]) -> None:
    errors.append(message)


def read_frontmatter(path: Path, errors: list[str]) -> dict[str, str]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        fail(f"{path}: cannot read: {exc}", errors)
        return {}
    if not text.startswith("---\n"):
        fail(f"{path}: missing YAML frontmatter", errors)
        return {}
    end = text.find("\n---", 4)
    if end < 0:
        fail(f"{path}: unterminated YAML frontmatter", errors)
        return {}
    values: dict[str, str] = {}
    for line in text[4:end].splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        key, separator, value = line.partition(":")
        if not separator or not key.strip() or not value.strip():
            fail(f"{path}: malformed frontmatter line", errors)
            continue
        values[key.strip()] = value.strip().strip("'\"")
    return values


def check_skill_links(path: Path, errors: list[str]) -> None:
    text = path.read_text(encoding="utf-8")
    for target in MARKDOWN_LINK.findall(text):
        target = target.strip().split("#", 1)[0].split("?", 1)[0]
        if not target or target.startswith(("http://", "https://", "mailto:", "<")):
            continue
        if target.startswith("/"):
            fail(f"{path}: absolute Markdown link is not local: {target}", errors)
            continue
        if not (path.parent / target).exists():
            fail(f"{path}: missing local Markdown link: {target}", errors)


def validate_tree(root: Path) -> list[str]:
    errors: list[str] = []
    agents = root / "AGENTS.md"
    if not agents.is_file():
        fail("AGENTS.md is missing", errors)
    elif agents.stat().st_size > MAX_AGENTS_BYTES:
        fail(f"AGENTS.md exceeds {MAX_AGENTS_BYTES} bytes", errors)

    config_path = root / ".codex/config.toml"
    agent_path = root / ".codex/agents/luna-implementer.toml"
    try:
        config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        fail(f"{config_path}: invalid TOML: {exc}", errors)
        config = {}
    if config.get("model") != "gpt-5.6-sol":
        fail(".codex/config.toml must keep gpt-5.6-sol as the manager model", errors)
    delegation = config.get("agents")
    if not isinstance(delegation, dict) or delegation.get("enabled") is not True:
        fail(".codex/config.toml must keep agents.enabled = true", errors)
    if not isinstance(delegation, dict) or delegation.get("default_subagent_model") != "gpt-5.6-luna":
        fail(".codex/config.toml must keep gpt-5.6-luna as the default implementer", errors)
    try:
        luna = tomllib.loads(agent_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        fail(f"{agent_path}: invalid TOML: {exc}", errors)
        luna = {}
    if luna.get("model") != "gpt-5.6-luna":
        fail("luna-implementer.toml must use gpt-5.6-luna", errors)
    if luna.get("sandbox_mode") != "workspace-write":
        fail("luna-implementer.toml must use workspace-write", errors)

    try:
        agents_text = agents.read_text(encoding="utf-8")
    except OSError:
        agents_text = ""
    skill_root = root / ".agents/skills"
    skill_names: list[str] = []
    if skill_root.is_dir():
        for directory in sorted(p for p in skill_root.iterdir() if p.is_dir()):
            skill = directory / "SKILL.md"
            if not skill.is_file():
                fail(f"{directory}: SKILL.md is missing", errors)
                continue
            metadata = read_frontmatter(skill, errors)
            name = metadata.get("name", "")
            description = metadata.get("description", "")
            if not SKILL_NAME.fullmatch(name):
                fail(f"{skill}: name is not a bounded kebab-case name", errors)
            if len(name) > MAX_SKILL_NAME_LENGTH:
                fail(f"{skill}: name exceeds {MAX_SKILL_NAME_LENGTH} characters", errors)
            if name != directory.name:
                fail(f"{skill}: frontmatter name does not match directory", errors)
            if len(description.strip()) < 20:
                fail(f"{skill}: description is empty or not discriminating", errors)
            if len(description) > MAX_DESCRIPTION_LENGTH:
                fail(f"{skill}: description exceeds {MAX_DESCRIPTION_LENGTH} characters", errors)
            skill_names.append(name)
            check_skill_links(skill, errors)
    if len(skill_names) != len(set(skill_names)):
        fail("skill names must be unique", errors)
    routed = set(SKILL_TOKEN.findall(agents_text))
    expected = set(skill_names)
    missing = sorted(expected - routed)
    unknown = sorted(routed - expected)
    if missing:
        fail("AGENTS.md does not route every repository skill: " + ", ".join(missing), errors)
    if unknown:
        fail("AGENTS.md routes unknown skill(s): " + ", ".join(unknown), errors)
    return errors


def self_test() -> int:
    with tempfile.TemporaryDirectory(prefix="vokra-agent-contract-") as directory:
        root = Path(directory)
        (root / ".codex/agents").mkdir(parents=True)
        (root / ".agents/skills/demo").mkdir(parents=True)
        (root / "AGENTS.md").write_text("$demo\n", encoding="utf-8")
        (root / ".codex/config.toml").write_text(
            'model = "gpt-5.6-sol"\n[agents]\nenabled = true\ndefault_subagent_model = "gpt-5.6-luna"\n', encoding="utf-8"
        )
        (root / ".codex/agents/luna-implementer.toml").write_text(
            'model = "gpt-5.6-luna"\nsandbox_mode = "workspace-write"\n', encoding="utf-8"
        )
        (root / ".agents/skills/demo/SKILL.md").write_text(
            "---\nname: demo\ndescription: Demonstrate a bounded repository skill.\n---\n",
            encoding="utf-8",
        )
        assert not validate_tree(root), "valid fixture must pass"
        (root / "AGENTS.md").write_text("$missing\n", encoding="utf-8")
        assert validate_tree(root), "unknown/missing routing must fail"
        (root / "AGENTS.md").write_text("$demo\n" + "x" * MAX_AGENTS_BYTES, encoding="utf-8")
        assert validate_tree(root), "oversize AGENTS.md must fail"
        (root / "AGENTS.md").write_text("$demo\n", encoding="utf-8")
        (root / ".codex/config.toml").write_text(
            'model = "gpt-5.5"\n[agents]\nenabled = true\ndefault_subagent_model = "gpt-5.6-luna"\n', encoding="utf-8"
        )
        assert validate_tree(root), "wrong manager model must fail"
        (root / ".codex/config.toml").write_text(
            'model = "gpt-5.6-sol"\n[agents]\nenabled = true\ndefault_subagent_model = "gpt-5.6-luna"\n', encoding="utf-8"
        )
        (root / ".codex/agents/luna-implementer.toml").write_text(
            'model = "gpt-5.6-luna"\nsandbox_mode = "read-only"\n', encoding="utf-8"
        )
        assert validate_tree(root), "wrong Luna sandbox must fail"
        (root / ".codex/agents/luna-implementer.toml").write_text(
            'model = "gpt-5.6-luna"\nsandbox_mode = "workspace-write"\n', encoding="utf-8"
        )
        (root / ".agents/skills/demo/SKILL.md").write_text(
            "---\nname: demo\ndescription: Demonstrate a bounded repository skill.\n---\n[missing](missing.md)\n",
            encoding="utf-8",
        )
        assert validate_tree(root), "missing local link must fail"
        (root / ".agents/skills/demo/SKILL.md").write_text(
            "---\nname: " + "a" * 65 + "\ndescription: " + "x" * 1025 + "\n---\n",
            encoding="utf-8",
        )
        assert validate_tree(root), "oversize/bad skill metadata must fail"
    print("check-agent-contracts --self-test: OK")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    errors = validate_tree(Path(__file__).resolve().parents[1])
    if errors:
        for error in errors:
            print(f"check-agent-contracts: {error}")
        return 1
    print("Agent contract: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
