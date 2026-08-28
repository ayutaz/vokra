#!/usr/bin/env python3
"""Download an HF checkpoint that ships pytorch_model.bin only (no
model.safetensors), convert the .bin to .safetensors, and preserve the
tokenizer / config side-cars unchanged.

Owner Task 3 Wave 2b (2026-07-27, docs/tickets/sbv2/task-3-decisions.md
Decision ③): SBV2 v2's EN BERT branch requires ``microsoft/deberta-v3-large``,
which ships only ``pytorch_model.bin`` — there is no ``model.safetensors``
mirror at the source. Every other DeBERTa v3 variant we surveyed
(``deberta-v3-base`` / ``mdeberta-v3-base``) has the same distribution
shape (see the same docs/tickets file). Vokra's Rust converter
(``vokra-convert``) is safetensors-only by design (keeps zero-dep
NFR-DS-02: no pickle parser in the runtime, and no arbitrary-code-exec
risk on load); so we do the bin→safetensors conversion in Python — where
torch already knows how to safely deserialize its own pickle — up-front,
as an offline sidecar tool.

# NOT REFERENCED (clean-room, matches sbv2_prepare_checkpoint.py)

- github.com/litagin02/Style-Bert-VITS2 (AGPL-3.0)
- github.com/fishaudio/Bert-VITS2 (AGPL-3.0)

Uses only ``torch.load`` (BSD-3, weights_only=True — the safe path that
disallows arbitrary object construction) plus ``safetensors.torch.save_file``
(Apache-2.0) plus ``huggingface_hub.snapshot_download`` (Apache-2.0). No
AGPL source is read or referenced.

# FR-EX-08 loud-error posture

- Missing / malformed pickle → propagates torch.load's own exception with
  full traceback (never a silent partial write).
- Detected non-tensor object in the state_dict → fail-loud with the offending
  key name, refuse to write partial output.
- Empty state_dict after load → fail-loud (a real DeBERTa v3 large has
  ~400 tensors, an empty one is a corrupt download).
- torch.load with weights_only=False is NEVER called (silently promoting
  weights_only=True's failure to False would allow arbitrary pickle
  reduce/opcode execution — a documented CVE surface).

# Usage

    uv run python bin_to_safetensors.py \\
        --hf-repo microsoft/deberta-v3-large \\
        --output-dir /tmp/deberta-v3-large-safetensors

Downloads .bin + config.json + tokenizer files into ``--output-dir``,
writes ``model.safetensors`` alongside them, and prints sha256 of the
output.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

LOG_PREFIX = "bin_to_safetensors:"

# Files the downstream user (Vokra's DeBERTa converter, plus dumpers under
# ``tools/parity/``) actually needs. Deliberately excludes ``tf_model.h5``
# (TensorFlow weights, unused) and ``pytorch_model.generator.bin`` (the
# separate MLM generator, unused by DeBERTa v3 discriminator inference).
HF_ALLOW_PATTERNS = [
    "*.bin",
    "*.safetensors",  # if upstream *does* ship one, skip conversion (see main)
    "config.json",
    "generator_config.json",
    "added_tokens.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.txt",
    "spm.model",
    "spm_char.model",
]

# Basenames that DO NOT need conversion (auxiliary or non-primary weights).
CONVERT_SKIP_BASENAMES = {
    "pytorch_model.generator.bin",
}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_checkpoint(hf_repo: str, output_dir: Path, revision: "str | None") -> Path:
    """Mirrors ``sbv2_prepare_checkpoint.download_checkpoint`` — token is
    resolved from the environment by ``huggingface_hub``, never accepted as
    a CLI flag (argv leaks via ``ps``)."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        sys.exit(
            f"missing Python dep ({exc}); install with "
            "`uv add huggingface_hub` in tools/parity/"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    local_dir = snapshot_download(
        repo_id=hf_repo,
        repo_type="model",
        revision=revision,
        local_dir=str(output_dir),
        allow_patterns=HF_ALLOW_PATTERNS,
    )
    return Path(local_dir)


def convert_bin_to_safetensors(
    bin_path: Path,
    out_path: Path,
    skip_tensor_names: "frozenset[str] | None" = None,
) -> tuple[int, int]:
    """Loads ``bin_path`` with ``torch.load(weights_only=True)``, verifies
    every value is a real tensor, then writes ``out_path`` via
    ``safetensors.torch.save_file``.

    Returns ``(tensor_count, total_param_count)``. Fails loud (raises) on
    the corruption modes documented in the module docstring — never writes
    a partial file. ``skip_tensor_names`` is deliberately narrow: every name
    must exist and identify a scalar int32/int64 training counter. It cannot
    be used as a broad dtype or pattern-based filter.
    """
    try:
        import torch
    except ImportError as exc:
        sys.exit(f"missing Python dep torch ({exc}); install with `uv add torch`")
    try:
        from safetensors.torch import save_file
    except ImportError as exc:
        sys.exit(
            f"missing Python dep safetensors ({exc}); install with "
            "`uv add safetensors`"
        )

    # weights_only=True is the safe path: it disallows arbitrary object
    # construction (torch.load's default until torch 2.6 was the UNSAFE
    # path with full pickle reduce/opcode execution). Never fall back to
    # False silently — see docstring "FR-EX-08" section.
    state_dict = torch.load(str(bin_path), map_location="cpu", weights_only=True)

    if not isinstance(state_dict, dict):
        sys.exit(
            f"{bin_path} did not deserialize to a dict "
            f"(got {type(state_dict).__name__}); refusing to convert."
        )

    if len(state_dict) == 0:
        sys.exit(
            f"{bin_path} deserialized to an empty state_dict — treated as a "
            "corrupt download rather than a valid empty model. Re-download "
            "and retry."
        )

    # Classic wrapped checkpoint: `torch.save({'state_dict': model.state_dict()}, ...)`
    # or {'model': model.state_dict()} — a single well-known wrapper key whose
    # value is itself a dict of tensors. Unwrap explicitly (log to stderr so
    # it is never silent) but ONLY for the known-safe patterns; any other shape
    # falls through to the existing non-tensor refusal. Encountered on
    # openbmb/VoxCPM-0.5B pytorch_model.bin (state_dict wrapper).
    if len(state_dict) == 1:
        (only_key,) = state_dict.keys()
        (only_val,) = state_dict.values()
        if only_key in ("state_dict", "model", "weights", "module") and isinstance(
            only_val, dict
        ):
            print(
                f"{LOG_PREFIX}   detected wrapped checkpoint — unwrapping "
                f"top-level key {only_key!r} ({len(only_val)} tensors inside)",
                file=sys.stderr,
            )
            state_dict = only_val

    # Validate every value is a tensor and every key is a str.
    non_tensor_offenders: list[str] = []
    non_str_keys: list[object] = []
    for k, v in state_dict.items():
        if not isinstance(k, str):
            non_str_keys.append(k)
        if not isinstance(v, torch.Tensor):
            non_tensor_offenders.append(f"{k!r} -> {type(v).__name__}")
    if non_str_keys:
        sys.exit(
            f"{bin_path} state_dict has {len(non_str_keys)} non-str key(s) "
            f"(first: {non_str_keys[0]!r}); refusing to convert — safetensors "
            "requires str keys."
        )
    if non_tensor_offenders:
        sys.exit(
            f"{bin_path} state_dict has {len(non_tensor_offenders)} non-tensor "
            f"value(s): {non_tensor_offenders[:5]!r} (showing first 5). "
            "Refusing to convert — safetensors is a tensor-only format."
        )

    if skip_tensor_names:
        missing_skips = skip_tensor_names.difference(state_dict)
        if missing_skips:
            sys.exit(
                f"{bin_path} is missing {len(missing_skips)} explicitly requested "
                f"skip tensor(s): {sorted(missing_skips)!r}. Refusing topology drift."
            )
        for name in sorted(skip_tensor_names):
            value = state_dict[name]
            if value.ndim != 0 or value.dtype not in (torch.int32, torch.int64):
                sys.exit(
                    f"{bin_path} requested skip tensor {name!r} is "
                    f"shape={tuple(value.shape)!r} dtype={value.dtype}, expected a "
                    "scalar integer training counter. Refusing a broad skip."
                )
            del state_dict[name]
            print(
                f"{LOG_PREFIX}   excluding training-only scalar {name!r}",
                file=sys.stderr,
            )

    # Force contiguous layout — safetensors requires it, and .bin sometimes
    # ships views onto shared underlying storage.
    state_dict = {k: v.contiguous() for k, v in state_dict.items()}

    # Tied-weight shared storage: models with `tie_word_embeddings=true`
    # (Qwen2, Llama, ...) ship `lm_head.weight` and `embed_tokens.weight`
    # as views onto the same tensor storage. safetensors REFUSES to save
    # aliased storage (would double-load at unpack). Clone the alias so
    # the on-disk copy is independent — the runtime is free to re-tie on
    # load. Fail-loud if a shared-storage cluster is detected AND
    # cloning would produce a non-tensor (should never happen).
    # Encountered on FunAudioLLM/CosyVoice2-0.5B llm.pt (Qwen2 backbone).
    storage_ids: dict[int, str] = {}
    for k in list(state_dict.keys()):
        v = state_dict[k]
        sid = v.untyped_storage().data_ptr()
        if sid in storage_ids:
            print(
                f"{LOG_PREFIX}   detected tied-weight shared storage: "
                f"{k!r} aliases {storage_ids[sid]!r} — cloning independent copy",
                file=sys.stderr,
            )
            state_dict[k] = v.clone()
        else:
            storage_ids[sid] = k

    total_params = sum(v.numel() for v in state_dict.values())
    save_file(state_dict, str(out_path))
    return len(state_dict), total_params


def convert_local(
    input_path: Path,
    output_path: Path,
    skip_tensor_names: "frozenset[str] | None" = None,
) -> int:
    """Mode b: convert a single local torch pickle (.bin / .pt) to a specific
    .safetensors path. Same fail-loud posture as Mode a — see the module
    docstring. Refuses to overwrite an existing output.

    ``skip_tensor_names`` is intended only for a model-specific wrapper that
    pins exact training-only scalar counter names.

    Encountered on `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` (llm.pt / flow.pt /
    hift.pt shipped side-by-side; the whole-repo Mode a walk would fail its
    single-file assumption).
    """
    if not input_path.exists():
        sys.exit(f"{LOG_PREFIX} --input {input_path} does not exist.")
    if not input_path.is_file():
        sys.exit(f"{LOG_PREFIX} --input {input_path} is not a regular file.")
    if output_path.exists():
        sys.exit(
            f"{LOG_PREFIX} refusing to overwrite existing --output {output_path}. "
            "Remove it first or pick a different --output path."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"{LOG_PREFIX} converting {input_path} -> {output_path}")
    tensor_count, total_params = convert_bin_to_safetensors(
        input_path, output_path, skip_tensor_names
    )
    print(
        f"{LOG_PREFIX} wrote {output_path} "
        f"({tensor_count} tensors, {total_params:,} total params)"
    )
    print(f"{LOG_PREFIX} sha256 {sha256_of(output_path)}  {output_path.name}")
    print(f"{LOG_PREFIX} done.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a torch pickle checkpoint (.bin / .pt) to safetensors. "
            "Two mutually-exclusive modes: (a) --hf-repo + --output-dir "
            "downloads a whole HF snapshot then converts the single .bin file "
            "in it (the original DeBERTa-v3-large use case); (b) --input + "
            "--output takes a local .bin / .pt file already on disk and "
            "writes a specific .safetensors path (the Fun-CosyVoice3 use case "
            "— the release ships llm.pt / flow.pt / hift.pt so the caller picks "
            "one component at a time). See the module docstring for the "
            "FR-EX-08 fail-loud posture."
        )
    )
    parser.add_argument(
        "--hf-repo",
        default=None,
        help=(
            "(Mode a) HuggingFace repo id (e.g. microsoft/deberta-v3-large). "
            "Requires --output-dir; incompatible with --input/--output."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        type=Path,
        help=(
            "(Mode a) Directory to download the checkpoint into (created if "
            "absent)."
        ),
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="(Mode a) Optional HF revision (branch / tag / commit sha) to pin.",
    )
    parser.add_argument(
        "--out-basename",
        default="model.safetensors",
        help=(
            "(Mode a) Output filename for the converted safetensors "
            "(default: model.safetensors, mirrors HF convention)."
        ),
    )
    parser.add_argument(
        "--input",
        default=None,
        type=Path,
        help=(
            "(Mode b) Local torch pickle file (.bin / .pt) already on disk to "
            "convert. Requires --output; incompatible with --hf-repo/--output-dir. "
            "The file is loaded with torch.load(weights_only=True) — see the "
            "module docstring for the fail-loud posture."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        type=Path,
        help=(
            "(Mode b) Local .safetensors file to write. The parent directory "
            "is created if absent. Refuses to overwrite an existing file."
        ),
    )
    args = parser.parse_args()

    # Mode selection — exactly one of the two triples must be complete.
    mode_a = args.hf_repo is not None or args.output_dir is not None
    mode_b = args.input is not None or args.output is not None
    if mode_a and mode_b:
        sys.exit(
            f"{LOG_PREFIX} --hf-repo/--output-dir (download mode) and --input/--output "
            "(local mode) are mutually exclusive. Pass one triple, not both."
        )
    if not mode_a and not mode_b:
        sys.exit(
            f"{LOG_PREFIX} no mode selected — pass either --hf-repo + --output-dir "
            "(download mode) or --input + --output (local mode). See --help."
        )
    if mode_b:
        if args.input is None or args.output is None:
            sys.exit(
                f"{LOG_PREFIX} local mode requires BOTH --input and --output."
            )
        return convert_local(args.input, args.output)
    # Mode a fall-through — the download path.
    if args.hf_repo is None or args.output_dir is None:
        sys.exit(
            f"{LOG_PREFIX} download mode requires BOTH --hf-repo and --output-dir."
        )

    print(f"{LOG_PREFIX} downloading {args.hf_repo!r} -> {args.output_dir}")
    local_dir = download_checkpoint(args.hf_repo, args.output_dir, args.revision)
    print(f"{LOG_PREFIX}   -> snapshot at {local_dir}")

    # If upstream ALREADY ships model.safetensors, skip conversion. This
    # keeps the tool idempotent-safe: running it against a repo that later
    # gets a safetensors mirror will just report "already safetensors" and
    # exit 0, not shadow the upstream mirror with a re-serialized copy.
    already_safetensors = sorted(local_dir.rglob("*.safetensors"))
    if already_safetensors:
        print(
            f"{LOG_PREFIX} upstream already ships safetensors "
            f"({len(already_safetensors)} file(s) found) — no conversion needed."
        )
        for st in already_safetensors:
            print(f"{LOG_PREFIX}   {st.relative_to(local_dir)}")
            print(f"{LOG_PREFIX}   sha256 {sha256_of(st)}")
        return 0

    bins = sorted(
        p
        for p in local_dir.rglob("*.bin")
        if p.name not in CONVERT_SKIP_BASENAMES
    )
    if not bins:
        sys.exit(
            f"{LOG_PREFIX} no *.bin files found under {local_dir} "
            f"(and no model.safetensors either) — allow_patterns "
            f"{HF_ALLOW_PATTERNS!r} may need widening for this repo."
        )
    if len(bins) > 1:
        # Multi-shard .bin (pytorch_model-00001-of-000NN.bin) is a distinct
        # merge problem; refuse to guess.
        sys.exit(
            f"{LOG_PREFIX} multiple .bin files found under {local_dir} — this "
            f"tool converts a single-shard .bin only. Files: "
            f"{[str(b.relative_to(local_dir)) for b in bins]!r}. Merge shards "
            "externally (transformers save_pretrained / accelerate load_state_dict) "
            "before running this converter."
        )

    bin_path = bins[0]
    out_path = local_dir / args.out_basename
    print(f"{LOG_PREFIX} converting {bin_path.relative_to(local_dir)} -> "
          f"{out_path.relative_to(local_dir)}")

    if out_path.exists():
        sys.exit(
            f"{LOG_PREFIX} refusing to overwrite existing {out_path}. "
            "Remove it first or pass --out-basename with a different name."
        )

    tensor_count, total_params = convert_bin_to_safetensors(bin_path, out_path)

    print(
        f"{LOG_PREFIX} wrote {out_path} "
        f"({tensor_count} tensors, {total_params:,} total params)"
    )
    print(f"{LOG_PREFIX} sha256 {sha256_of(out_path)}  {out_path.name}")
    print(f"{LOG_PREFIX} done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
