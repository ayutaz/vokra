#!/usr/bin/env python3
"""Download + prepare SBV2 v2 base + DeBERTa v2 (JA) + DeBERTa v3 (EN)
checkpoints for Vokra parity fixture provisioning (SBV2 v2 Blocker 2b/2c/3/5
remediation plan, Task 4:
``docs/superpowers/plans/2026-08-11-sbv2-v2-blockers-2b-2c-3.md``).

This is an **offline** sidecar tool (FR-LD-05: no Python / PyTorch is ever
pulled into the runtime).

# Why this is a SEPARATE file from ``sbv2_prepare_checkpoint.py``

``tools/parity/sbv2_prepare_checkpoint.py`` already exists (landed in the
earlier SBV2 v2 Phase 1 plan, Task 29, PR #22/#27) and does a different,
narrower job: download **one** HF repo (default ``--hf-repo`` overridable)
and best-effort map its ``config.json`` (or shape-derive from the
safetensors header, or fall back to cited clean-room MIT defaults) onto the
``vokra.sbv2.*`` side-car schema. It is unit-tested directly
(``tools/parity/test_sbv2_prepare_shape_derivation.py`` does
``import sbv2_prepare_checkpoint as prep`` and calls
``prep._derive_shape_fields`` / ``prep.build_config_side_car`` /
``prep.ALL_TARGET_KEYS``) and referenced from
``tools/parity/sbv2_dump_reference.py``, ``tests/fixtures/sbv2/README.md``,
and rustdoc comments in ``crates/vokra-cli/src/convert.rs`` /
``crates/vokra-convert/src/models/sbv2.rs`` /
``crates/vokra-convert/tests/sbv2_convert.rs``. Overwriting it with this
plan's Task 4 brief (a *bulk, 3-repo* downloader with a fixed
``sbv2-v2-base``/``deberta-v2-ja``/``deberta-v3-en`` directory layout and no
config-mapping logic at all) would silently break that existing, tested,
cross-referenced tool. This file exists alongside it instead — same
directory, distinct name, distinct job (fetch the 3 raw checkpoint bundles
Task 5's ``vokra-cli convert`` calls need; it does **not** build a
``vokra.sbv2.*`` config side-car — that remains ``sbv2_prepare_checkpoint
.py``'s job for the SBV2 leg, run separately if/when its config-mapping is
wanted).

# NOT REFERENCED (clean-room)

- github.com/litagin02/Style-Bert-VITS2 (AGPL-3.0)
- github.com/fishaudio/Bert-VITS2 (AGPL-3.0)
- Any AGPL derivative of the above.

This script only calls the public ``huggingface_hub`` API (Apache-2.0) to
fetch upstream-*published* files (safetensors/pickle weights + tokenizer
JSON/vocab/spm files) via ``snapshot_download``. No AGPL source code is
read, copied, or referenced to write this tool.

# Usage

::

    uv run python sbv2_v2_bundle_prepare_checkpoint.py --out ~/vokra-checkpoints/sbv2-v2

Re-running is cheap, not because this script tracks its own "already
populated" state, but because ``huggingface_hub.snapshot_download`` itself
only re-fetches blobs that are missing or changed (ETag-checked against its
local cache) — a rerun with everything already cached does no network
transfer beyond a handful of HEAD requests.

# Known layout quirks (recorded here so Task 5 does not have to
# rediscover them by trial and error)

- ``litagin/Style-Bert-VITS2-2.0-base-JP-Extra`` ships **no** ``config.json``
  at all — only ``G_0.safetensors`` (generator / inference weights),
  ``D_0.safetensors`` and ``WD_0.safetensors`` (discriminators, training-only,
  not needed for inference). There is no single file named
  ``model.safetensors``; the inference checkpoint is ``G_0.safetensors``.
- ``microsoft/deberta-v3-large`` ships **no** ``model.safetensors`` — only a
  torch-pickle ``pytorch_model.bin`` (plus a training-only
  ``pytorch_model.generator.bin`` / ``generator_config.json`` pair for the
  ELECTRA-style generator head, irrelevant to inference). Converting
  ``pytorch_model.bin`` -> ``.safetensors`` is a separate step (see the
  sibling ``tools/parity/bin_to_safetensors.py``, already in this
  directory and explicitly designed to mirror
  ``sbv2_prepare_checkpoint.download_checkpoint``'s conventions) — this
  script deliberately does not invoke it, to keep "download exactly what
  upstream publishes" and "convert format" as separate, individually
  inspectable steps.
- ``ku-nlp/deberta-v2-large-japanese-char-wwm`` ships a native
  ``model.safetensors`` — no conversion needed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LOG_PREFIX = "[sbv2-v2-bundle-prep]"

REPOS: list[tuple[str, str]] = [
    ("litagin/Style-Bert-VITS2-2.0-base-JP-Extra", "sbv2-v2-base"),
    ("ku-nlp/deberta-v2-large-japanese-char-wwm", "deberta-v2-ja"),
    ("microsoft/deberta-v3-large", "deberta-v3-en"),
]


def download_repo(repo_id: str, out_dir: Path) -> Path:
    """Download ``repo_id`` (full snapshot: weights + every JSON/vocab/spm
    tokenizer sidecar file it publishes) directly into ``out_dir``. Returns
    the local path (== ``out_dir``, per ``snapshot_download``'s own
    contract when ``local_dir`` is given).

    ``huggingface_hub`` is imported lazily (not at module level) so
    ``--help`` works even in an interpreter without it installed — same
    convention ``sbv2_prepare_checkpoint.py``'s ``download_checkpoint``
    uses.

    No explicit ``token=`` — ``huggingface_hub`` resolves ``HF_TOKEN`` /
    ``HUGGING_FACE_HUB_TOKEN`` from the environment or a cached login on
    its own (same env-only-token convention as
    ``scripts/publish/upload.sh`` / ``sbv2_prepare_checkpoint.py``: argv
    tokens can leak via ``ps``/shell history).
    """
    from huggingface_hub import snapshot_download

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{LOG_PREFIX} [DL] {repo_id} -> {out_dir}")
    local = snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        local_dir=str(out_dir),
        # Excludes formats none of the 3 REPOS entries' consumers (Rust
        # `vokra-cli convert`, this repo's torch/safetensors-based
        # dump_reference.py scripts) can read at all, or that are
        # training-only and never loaded for inference:
        #   *.h5        - TensorFlow SavedModel weights (deberta-v3-large
        #                 ships one; nothing here uses TF).
        #   *.msgpack   - Flax/JAX weights (none of these repos currently
        #                 ship one, excluded defensively).
        #   *generator* - deberta-v3-large's ELECTRA-style generator head
        #                 (`pytorch_model.generator.bin` +
        #                 `generator_config.json`) — training-only, per
        #                 `deberta_v3_dump_reference.py`'s own module doc
        #                 ("the one inference-relevant delta lives
        #                 entirely inside transformers.DebertaV2Model
        #                 already"; the generator is never loaded by
        #                 `AutoModel.from_pretrained` for inference).
        # Deliberately NOT excluding `pytorch_model.bin` globally even
        # though it duplicates `model.safetensors` on the deberta-v2-ja
        # repo (safe to ignore downstream, ~1.2 GB) — deberta-v3-large
        # ships NO safetensors at all, so the same filename is the only
        # inference weight file that repo has.
        ignore_patterns=["*.h5", "*.msgpack", "*generator*"],
    )
    return Path(local)


def flatten_pth_if_needed(local_dir: Path) -> None:
    """If ``local_dir`` contains a torch-pickle ``*.pth`` (full
    ``torch.save`` state_dict — distinct from a HF-style
    ``pytorch_model.bin``, which none of the 3 REPOS entries currently
    require flattening either, see module docstring "Known layout
    quirks"), flatten it to a ``.safetensors`` file so ``vokra-cli
    convert`` can read it uniformly. Deduplicates shared tensors (by
    ``data_ptr``, since ``safetensors`` refuses pointer collisions) — same
    pattern as ``tools/parity/nemo_pt_to_safetensors.py``.

    No-op (and does not import ``torch``) when no ``*.pth`` is present —
    as of 2026-08-11 none of the 3 REPOS ship one, so this is defensive
    future-proofing, not exercised code.
    """
    pth_paths = sorted(local_dir.glob("*.pth"))
    if not pth_paths:
        return
    import torch
    from safetensors.torch import save_file

    pth = pth_paths[0]
    st_out = pth.with_suffix(".safetensors")
    if st_out.exists():
        print(f"{LOG_PREFIX} [SKIP] {st_out.name} already exists")
        return
    print(f"{LOG_PREFIX} [FLATTEN] {pth.name} -> {st_out.name}")
    state = torch.load(pth, map_location="cpu")
    # SBV2-family checkpoints sometimes wrap under {"model": {...}} — unwrap.
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    seen: dict[int, str] = {}
    flat: dict[str, "torch.Tensor"] = {}
    shared_pairs: list[tuple[str, str]] = []
    for name, tensor in state.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        ptr = tensor.data_ptr()
        if ptr in seen:
            shared_pairs.append((name, seen[ptr]))
            continue
        seen[ptr] = name
        flat[name] = tensor.contiguous().clone()
    save_file(flat, str(st_out))
    if shared_pairs:
        audit = local_dir / "shared_pairs.json"
        audit.write_text(json.dumps(shared_pairs, indent=2))
        print(f"{LOG_PREFIX} [AUDIT] {len(shared_pairs)} shared tensors -> {audit.name}")


def dedupe_redundant_pytorch_bin(local_dir: Path) -> None:
    """If ``local_dir`` has BOTH ``model.safetensors`` and
    ``pytorch_model.bin`` (``ku-nlp/deberta-v2-large-japanese-char-wwm``
    ships both — same weights, two serializations), removes the ``.bin``.
    ``vokra-cli convert`` and every dumper in this tree read
    ``.safetensors``; keeping the pickle duplicate around only costs ~1.2
    GB of disk for zero benefit. Trivially re-fetchable (from
    ``huggingface_hub``'s own local blob cache, no network) if ever
    needed — this is not the tensor source of truth, ``model.safetensors``
    is. No-op when only one of the two is present (e.g.
    ``deberta-v3-en``, which ships no ``.safetensors`` at all and must
    keep its ``.bin``).
    """
    st = local_dir / "model.safetensors"
    binf = local_dir / "pytorch_model.bin"
    if st.exists() and binf.exists():
        size = binf.stat().st_size
        binf.unlink()
        print(
            f"{LOG_PREFIX} [DEDUP] removed {binf.relative_to(local_dir.parent)} "
            f"({size:,} bytes) — model.safetensors already covers these weights"
        )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Download the 3 raw upstream checkpoints (SBV2 v2 base + "
            "DeBERTa v2 JA + DeBERTa v3 EN) the SBV2 v2 parity fixture "
            "pipeline needs, into a fixed sbv2-v2-base/deberta-v2-ja/"
            "deberta-v3-en directory layout ready for "
            "`vokra-cli convert` (Task 5)."
        )
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path.home() / "vokra-checkpoints" / "sbv2-v2",
        help="Bundle root directory (default: ~/vokra-checkpoints/sbv2-v2).",
    )
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    for repo_id, slug in REPOS:
        local_dir = args.out / slug
        try:
            download_repo(repo_id, local_dir)
            flatten_pth_if_needed(local_dir)
            dedupe_redundant_pytorch_bin(local_dir)
        except Exception as e:  # noqa: BLE001 - FR-EX-08: surface, don't swallow
            print(f"{LOG_PREFIX} [FAIL] {repo_id}: {e}", file=sys.stderr)
            return 1

    print(f"{LOG_PREFIX} [DONE] All 3 checkpoints ready:")
    for _, slug in REPOS:
        for f in sorted((args.out / slug).iterdir()):
            if f.is_file():
                print(f"{LOG_PREFIX}   {f.relative_to(args.out)}  ({f.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
