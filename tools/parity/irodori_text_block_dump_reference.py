#!/usr/bin/env python3
"""Dump an independent Irodori text-block reference fixture.

The oracle is ``irodori_tts.model.TextBlock`` imported from the official
Aratako/Irodori-TTS repository pinned to ``UPSTREAM_REVISION``. No Vokra
formula is imported or mirrored. The fixture covers non-causal masked MHA,
head-specific Q/K RMSNorm, adjacent-pair RoPE, the sigmoid output gate,
SwiGLU, and both residual additions.

Run with the repository's uv-managed Python 3.12 environment:

    uv run --project tools/parity \
      tools/parity/irodori_text_block_dump_reference.py \
      --upstream /work/Irodori-TTS \
      --output crates/vokra-models/tests/fixtures/irodori/text_block.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

from irodori_text_block_source_audit import audit_source, validate_reference_project

UPSTREAM_REVISION = "8224dafb46d0aba89209a8f905f1cb7e3299d9c1"
REFERENCE_BLOCKED_PENDING_DEPENDENCY_LICENSE_AUDIT = (
    "REFERENCE_BLOCKED_PENDING_DEPENDENCY_LICENSE_AUDIT"
)


def self_test() -> None:
    """Exercise only the source/lock contract; never import torch or a model."""

    assert validate_reference_project()["status"] == "PASS_ISOLATED_LOCK_STRUCTURAL_ONLY"
    allowed, reason = execution_preflight(
        {"dependency_audit_status": "PENDING_DEPENDENCY_LICENSE_AUDIT"}
    )
    assert not allowed and reason == REFERENCE_BLOCKED_PENDING_DEPENDENCY_LICENSE_AUDIT
    allowed, reason = execution_preflight({"dependency_audit_status": "APPROVED"})
    assert not allowed and reason == "REFERENCE_BLOCKED_NO_APPROVAL_SCHEMA"
    assert UPSTREAM_REVISION == "8224dafb46d0aba89209a8f905f1cb7e3299d9c1"
    source = Path(__file__).read_text(encoding="utf-8")
    assert "audit_source(args.upstream)" in source
    assert "irodori_tts.model.TextBlock" in source
    audit_at = source.index("\n    source_evidence = audit_source(args.upstream)\n")
    torch_at = source.index("\n    import torch\n")
    path_at = source.index("\n    sys.path.insert(0, str(args.upstream))\n")
    assert audit_at < torch_at and audit_at < path_at
    assert source.index("\n    if not allowed:\n") < torch_at
    print("irodori_text_block_dump_reference.py self-test: OK")


def flat(tensor):
    return tensor.detach().cpu().float().contiguous().view(-1).tolist()


def execution_preflight(source_evidence: dict[str, Any]) -> tuple[bool, str]:
    """Block before Torch import until an owner approval schema exists.

    The structural source/lock audit is useful evidence, but it is not a
    dependency-license approval.  There is currently no authenticated owner
    approval schema for this route, so no input—including an invented
    ``APPROVED`` marker—can authorize execution.
    """

    if source_evidence.get("dependency_audit_status") == "PENDING_DEPENDENCY_LICENSE_AUDIT":
        return False, REFERENCE_BLOCKED_PENDING_DEPENDENCY_LICENSE_AUDIT
    return False, "REFERENCE_BLOCKED_NO_APPROVAL_SCHEMA"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--upstream", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.self_test:
        if args.upstream is not None or args.output is not None:
            parser.error("--self-test is not compatible with --upstream/--output")
        self_test()
        return 0
    if args.upstream is None or args.output is None:
        parser.error("--upstream and --output are required unless --self-test is used")

    revision = subprocess.check_output(
        ["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != UPSTREAM_REVISION:
        raise SystemExit(
            f"Irodori-TTS checkout revision {revision} != pinned {UPSTREAM_REVISION}"
        )
    source_evidence = audit_source(args.upstream)
    allowed, reason = execution_preflight(source_evidence)
    if not allowed:
        print(f"Irodori TextBlock reference blocked: {reason}", file=sys.stderr)
        return 2

    sys.path.insert(0, str(args.upstream))
    # Import the official module without executing its package __init__, which
    # eagerly imports tokenizer/training-only dependencies irrelevant to this
    # numerical oracle.
    package = types.ModuleType("irodori_tts")
    package.__path__ = [str(args.upstream / "irodori_tts")]
    sys.modules["irodori_tts"] = package
    import torch
    from irodori_tts.model import TextBlock, precompute_freqs_cis

    torch.manual_seed(0x49524F44)
    dim = 16
    heads = 4
    mlp_ratio = 2.0
    norm_eps = 1.0e-5
    layer = TextBlock(
        dim=dim,
        heads=heads,
        mlp_ratio=mlp_ratio,
        norm_eps=norm_eps,
        dropout=0.0,
    ).eval().float()
    hidden = torch.linspace(-0.7, 0.9, steps=3 * dim, dtype=torch.float32)
    hidden = hidden.reshape(1, 3, dim)
    key_mask = torch.tensor([[True, True, False]], dtype=torch.bool)
    freqs = precompute_freqs_cis(dim // heads, hidden.shape[1])

    with torch.no_grad():
        output = layer(hidden, mask=key_mask, freqs_cis=freqs)

    state = layer.state_dict()
    names = {
        "attention_norm": "attention_norm.weight",
        "wq": "attention.wq.weight",
        "wk": "attention.wk.weight",
        "wv": "attention.wv.weight",
        "q_norm": "attention.q_norm.weight",
        "k_norm": "attention.k_norm.weight",
        "gate": "attention.gate.weight",
        "wo": "attention.wo.weight",
        "mlp_norm": "mlp_norm.weight",
        "w1": "mlp.w1.weight",
        "w2": "mlp.w2.weight",
        "w3": "mlp.w3.weight",
    }
    payload = {
        "upstream_repo": "https://github.com/Aratako/Irodori-TTS",
        "upstream_revision": revision,
        "oracle": "irodori_tts.model.TextBlock.forward",
        "torch_version": torch.__version__,
        "config": {
            "dim": dim,
            "n_head": heads,
            "mlp_ratio": mlp_ratio,
        },
        "key_mask": key_mask[0].tolist(),
        "hidden": flat(hidden),
        "weights": {key: flat(state[name]) for key, name in names.items()},
        "output": flat(output),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output} ({args.output.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
