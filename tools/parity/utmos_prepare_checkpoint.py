#!/usr/bin/env python3
"""Prepare an authenticated UTMOS state-dict → config side-car (M5-15 T14).

An **offline** sidecar tool (FR-LD-05: no Python / PyTorch ever enters the
runtime). Upstream ships a PyTorch-Lightning checkpoint
(``epoch=3-step=7459.ckpt`` on the ``sarulab-speech/UTMOS-demo`` HF space);
the Rust converter (``crates/vokra-convert/src/models/utmos.rs``) reads
safetensors + JSON only, so this script bridges the two:

* accepts only a tensor-only ``.safetensors`` state-dict. This is the
  canonical safe path: it contains no pickle program or Python objects. The
  historical Lightning ``.ckpt`` path is permanently refused;
* takes the state-dict verbatim — the dotted upstream keys are preserved,
  nothing is renamed here (the Rust converter owns the name mapping so the
  mapping is covered by its unit tests, exactly as
  ``dac_prepare_checkpoint.py`` leaves the weight-norm fold to Rust);
* drops **only** ``…ssl_model.mask_emb`` (present but unused at inference:
  upstream calls the SSL model with ``mask=False``), and says so on stdout;
* derives the config side-car from the *tensor shapes themselves* plus the
  two inference constants upstream's ``score.py`` pins (``domains = 0``,
  ``judge_id = 288``) — nothing about the architecture is invented here;
* prints a sha256 manifest line per output.

Anything unexpected — a missing key, a shape that disagrees with the rest of
the checkpoint, an unknown top-level layout — is a **loud failure**, never a
silently patched default (FR-EX-08).

# Usage

::

    uv run --project tools/parity/utmos --frozen python \\
        tools/parity/utmos_prepare_checkpoint.py \\
        --state-dict /vast/utmos22-strong.state_dict.safetensors \\
        --output /tmp/utmos22-strong.safetensors \\
        --config-out /tmp/utmos22-strong-config.json

Then::

    vokra-cli convert --model utmos --input /tmp/utmos22-strong.safetensors \\
        --config /tmp/utmos22-strong-config.json --output /tmp/utmos.gguf
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

# Upstream inference constants, quoted from the HF space's score.py:
#     'domains':  torch.zeros(bs, dtype=torch.int)
#     'judge_id': torch.ones(bs, dtype=torch.int) * 288
# and the final affine `output.mean(dim=1).squeeze(1) * 2 + 3`.
DOMAIN_ID = 0
JUDGE_ID = 288
HEAD_SCALE = 2.0
HEAD_OFFSET = 3.0
SAMPLE_RATE = 16000

SSL = "feature_extractors.0.ssl_model"
DOMAIN_EMB = "feature_extractors.1.embedding.weight"
LD = "output_layers.0"
PROJ = "output_layers.1.net"
# Present in the checkpoint but unused at inference (mask=False).
DROP = {f"{SSL}.mask_emb"}


def die(msg: str) -> "None":
    print(f"utmos_prepare_checkpoint: {msg}", file=sys.stderr)
    raise SystemExit(2)


def load_safetensors_state_dict(path: str) -> "dict[str, torch.Tensor]":
    """Load a tensor-only state-dict without invoking pickle at all.

    ``safetensors.torch.load_file`` parses the bounded safetensors header and
    maps tensor storage; it has no object deserialization or global allowlist.
    The extension check is deliberate: accepting an arbitrary file here would
    make the safety claim depend on caller intent rather than the format.
    """
    source = Path(path)
    if source.suffix != ".safetensors":
        die(f"safe state-dict must use the `.safetensors` extension: {source.name}")
    try:
        from safetensors.torch import load_file
    except ImportError:
        die("`safetensors` is not installed in this interpreter")
    try:
        state = load_file(str(source), device="cpu")
    except Exception as error:  # noqa: BLE001 — malformed input is terminal
        die(f"safetensors state-dict could not be loaded safely ({type(error).__name__}: {error})")
    if not isinstance(state, dict) or not state:
        die("safetensors state-dict is empty or has an unexpected root type")
    if any(not isinstance(key, str) or not key for key in state):
        die("safetensors state-dict contains a non-string or empty tensor key")
    return state


def need(sd, key):
    if key not in sd:
        die(f"checkpoint is missing the required tensor `{key}`")
    return sd[key]


def shape(sd, key):
    return list(need(sd, key).shape)


def derive_config(sd: "dict[str, torch.Tensor]") -> dict:
    """Reads the architecture off the tensor shapes (nothing invented)."""
    # --- conv feature encoder -------------------------------------------
    conv_channels, conv_kernels, gn_layers, gn_groups = [], [], [], []
    i = 0
    while f"{SSL}.feature_extractor.conv_layers.{i}.0.weight" in sd:
        c_out, _c_in, k = shape(sd, f"{SSL}.feature_extractor.conv_layers.{i}.0.weight")
        conv_channels.append(c_out)
        conv_kernels.append(k)
        # Sequential index 2 is the norm slot (0=conv, 1=Dropout, 2=norm,
        # 3=GELU); its presence is what marks a group-normed layer.
        if f"{SSL}.feature_extractor.conv_layers.{i}.2.weight" in sd:
            gn = shape(sd, f"{SSL}.feature_extractor.conv_layers.{i}.2.weight")
            if gn != [c_out]:
                die(f"conv layer {i} norm affine {gn} != [{c_out}]")
            gn_layers.append(i)
            # fairseq's Fp32GroupNorm(dim, dim) = one group per channel.
            gn_groups.append(c_out)
        i += 1
    if not conv_channels:
        die("no `feature_extractor.conv_layers.*` found — not a wav2vec2 checkpoint?")

    # Strides are NOT recoverable from shapes; they come from the checkpoint's
    # own conv_feature_layers arg, which the probe recorded verbatim. Refuse
    # to guess (a wrong stride silently changes the frame rate).
    strides_by_kernel = {10: 5, 3: 2, 2: 2}
    conv_strides = []
    for k in conv_kernels:
        if k not in strides_by_kernel:
            die(
                f"conv kernel {k} has no pinned stride — upstream's "
                f"conv_feature_layers is '[(512,10,5)] + [(512,3,2)]*4 + "
                f"[(512,2,2)]*2'; extend the table deliberately rather than "
                f"guessing"
            )
        conv_strides.append(strides_by_kernel[k])

    # --- projection / encoder --------------------------------------------
    d, c_last = shape(sd, f"{SSL}.post_extract_proj.weight")
    if c_last != conv_channels[-1]:
        die(f"post_extract_proj in-dim {c_last} != last conv channel {conv_channels[-1]}")
    pos_v = shape(sd, f"{SSL}.encoder.pos_conv.0.weight_v")  # [d, d/groups, k]
    if pos_v[0] != d:
        die(f"pos_conv weight_v out-channels {pos_v[0]} != hidden dim {d}")
    pos_kernel = pos_v[2]
    if d % pos_v[1] != 0:
        die(f"pos_conv in-channels-per-group {pos_v[1]} does not divide {d}")
    pos_groups = d // pos_v[1]

    n_layer = 0
    while f"{SSL}.encoder.layers.{n_layer}.fc1.weight" in sd:
        n_layer += 1
    if n_layer == 0:
        die("no encoder layers found")
    ffn_dim, _ = shape(sd, f"{SSL}.encoder.layers.0.fc1.weight")
    # Head count is not in the state dict; wav2vec2-base pins 12 and the
    # checkpoint's own args (wav2vec-small-args.txt: encoder_attention_heads
    # = 12, encoder_embed_dim = 768) agree. Derive as d / 64 (the fairseq
    # base head width) and cross-check divisibility rather than hard-coding.
    if d % 64 != 0:
        die(f"hidden dim {d} is not a multiple of the 64-wide fairseq head")
    n_head = d // 64

    # --- conditioning + BLSTM + head --------------------------------------
    n_domains, domain_dim = shape(sd, DOMAIN_EMB)
    n_judges, judge_dim = shape(sd, f"{LD}.judge_embedding.weight")
    g, blstm_in = shape(sd, f"{LD}.decoder_rnn.weight_ih_l0")
    if g % 4 != 0:
        die(f"BLSTM gate block {g} is not a multiple of 4")
    blstm_hidden = g // 4
    if blstm_in != d + domain_dim + judge_dim:
        die(
            f"BLSTM input {blstm_in} != hidden {d} + domain {domain_dim} + "
            f"judge {judge_dim}"
        )
    h0_out, h0_in = shape(sd, f"{PROJ}.0.weight")
    if h0_in != 2 * blstm_hidden:
        die(f"head linear 0 in-dim {h0_in} != 2 × BLSTM hidden {blstm_hidden}")
    h1_out, h1_in = shape(sd, f"{PROJ}.3.weight")
    if h1_in != h0_out or h1_out != 1:
        die(f"head linear 1 is [{h1_out}, {h1_in}], expected [1, {h0_out}]")
    if DOMAIN_ID >= n_domains or JUDGE_ID >= n_judges:
        die(
            f"pinned conditioning ids out of range: domain {DOMAIN_ID}/{n_domains}, "
            f"judge {JUDGE_ID}/{n_judges}"
        )

    return {
        "arch_variant": "wav2vec2_regression.v1",
        "sample_rate": SAMPLE_RATE,
        "conv_channels": conv_channels,
        "conv_kernels": conv_kernels,
        "conv_strides": conv_strides,
        "conv_activation": "gelu",
        "conv_group_norm_layers": gn_layers,
        "conv_group_norm_groups": gn_groups,
        # torch.nn.GroupNorm / LayerNorm defaults; fairseq overrides neither.
        "group_norm_eps": 1e-5,
        "ln_eps": 1e-5,
        "n_layer": n_layer,
        "n_head": n_head,
        "hidden_dim": d,
        "ffn_dim": ffn_dim,
        "norm": "post",
        "pos_conv_kernel": pos_kernel,
        "pos_conv_groups": pos_groups,
        "domain_dim": domain_dim,
        "domain_id": DOMAIN_ID,
        "judge_dim": judge_dim,
        "judge_id": JUDGE_ID,
        "blstm_hidden": blstm_hidden,
        "head_dims": [h0_out, 1],
        "head_pool": "mean_after",
        "head_activation": "relu",
        "head_scale": HEAD_SCALE,
        "head_offset": HEAD_OFFSET,
    }


def _output_path(raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = Path.cwd() / path
    if ".." in path.parts:
        die(f"output path must not contain '..': {raw}")
    return path


def _validate_output_parent(path: Path) -> None:
    parent = path.parent
    if not parent.exists() or parent.is_symlink() or not parent.is_dir():
        die(f"output parent must be an existing non-symlink directory: {parent}")
    current = parent
    while True:
        if current.is_symlink():
            resolved = current.resolve(strict=True)
            if (current, resolved) in ((Path("/var"), Path("/private/var")), (Path("/tmp"), Path("/private/tmp"))):
                current = resolved
                continue
            die(f"output parent contains an unexpected symlink component: {current}")
        if not current.is_dir():
            die(f"output parent contains a symlink or non-directory component: {current}")
        if current.parent == current:
            break
        current = current.parent


def validate_output_boundary(output: str, config_out: str) -> tuple[Path, Path]:
    """Validate create-new, disjoint output targets before loading tensors."""
    targets = (_output_path(output), _output_path(config_out))
    for target in targets:
        _validate_output_parent(target)
        if target.exists() or target.is_symlink():
            die(f"output target must be absent and non-symlinked: {target}")
    output_path, config_path = targets
    if output_path == config_path:
        die("--output and --config-out must be distinct paths")
    if output_path in config_path.parents or config_path in output_path.parents:
        die("--output and --config-out must be disjoint file paths")
    return targets


def _link_create_new(temp_path: Path, target: Path) -> None:
    """Publish one staged file without replacing a caller path."""
    try:
        os.link(temp_path, target)
    except FileExistsError:
        die(f"output target appeared during preparation; refusing to clobber: {target}")
    except OSError as error:
        die(f"could not atomically publish {target} without replacement ({type(error).__name__}: {error})")


def publish_outputs(tensors, config: dict, output: Path, config_out: Path) -> None:
    """Stage both outputs, then create-new link them into the caller paths."""
    staged: list[Path] = []
    published: list[tuple[Path, Path]] = []
    try:
        for target in (output, config_out):
            fd, raw = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
            os.close(fd)
            staged.append(Path(raw))
        try:
            from safetensors.torch import save_file
        except ImportError:
            die("`safetensors` is not installed in this interpreter")
        save_file(tensors, str(staged[0]))
        with staged[1].open("w", encoding="utf-8") as stream:
            json.dump(config, stream, indent=2, sort_keys=True)
            stream.write("\n")
        for temp_path, target in zip(staged, (output, config_out)):
            _link_create_new(temp_path, target)
            published.append((temp_path, target))
    except BaseException:
        for temp_path, target in reversed(published):
            try:
                if target.is_file() and os.path.samestat(os.stat(temp_path), os.stat(target, follow_symlinks=False)):
                    target.unlink()
            except (FileNotFoundError, OSError):
                pass
        raise
    finally:
        for temp_path in staged:
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass


def self_test() -> None:
    source = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"load", "load_state_dict"}
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "torch"
        for node in ast.walk(tree)
    ), "UTMOS preparation must not call torch pickle loaders"
    assert not any(isinstance(node, ast.ClassDef) and node.name.endswith("Unpickler") for node in ast.walk(tree))
    assert any(
        isinstance(node, ast.FunctionDef) and node.name == "load_safetensors_state_dict"
        for node in ast.walk(tree)
    ), "canonical tensor-only state-dict path is missing"
    source_text = source
    assert "load_file(str(source), device=\"cpu\")" in source_text
    assert "source.suffix != \".safetensors\"" in source_text
    assert ("safe_" + "globals") not in source_text
    assert "validate_output_boundary(args.output, args.config_out)" in source_text
    assert "os.link(temp_path, target)" in source_text
    with tempfile.TemporaryDirectory(prefix="utmos-prepare-self-test-") as raw_root:
        root = Path(raw_root)
        output = root / "out.safetensors"
        config = root / "config.json"
        validate_output_boundary(str(output), str(config))

        output.write_text("caller-owned", encoding="utf-8")
        try:
            validate_output_boundary(str(output), str(config))
        except SystemExit as error:
            assert error.code == 2
        else:
            raise AssertionError("existing output target was accepted")
        output.unlink()

        link_parent = root / "link-parent"
        link_parent.symlink_to(root, target_is_directory=True)
        try:
            validate_output_boundary(str(link_parent / "out"), str(config))
        except SystemExit as error:
            assert error.code == 2
        else:
            raise AssertionError("symlinked output parent was accepted")

        config.write_text("caller-owned", encoding="utf-8")
        try:
            validate_output_boundary(str(output), str(config))
        except SystemExit as error:
            assert error.code == 2
        else:
            raise AssertionError("existing config target was accepted")
        config.unlink()

        staged = root / "staged"
        staged.write_text("new", encoding="utf-8")
        output.write_text("caller-owned", encoding="utf-8")
        try:
            _link_create_new(staged, output)
        except SystemExit as error:
            assert error.code == 2
        else:
            raise AssertionError("create-new publish accepted a caller target")
        assert output.read_text(encoding="utf-8") == "caller-owned"
    print("utmos_prepare_checkpoint: safe state-dict self-test PASS")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--ckpt", help="legacy UTMOS22-strong .ckpt (always refused)")
    group.add_argument(
        "--state-dict",
        help="canonical tensor-only UTMOS state-dict (.safetensors; no pickle)",
    )
    ap.add_argument("--output", help="flat safetensors out")
    ap.add_argument("--config-out", help="config JSON side-car out")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        if any(value is not None for value in (args.ckpt, args.state_dict, args.output, args.config_out)):
            ap.error("--self-test accepts no checkpoint or output arguments")
        self_test()
        return 0
    if args.ckpt is not None:
        die(
            "BLOCKED_UNSAFE_PICKLE: legacy Lightning .ckpt inputs are permanently "
            "refused; provide an authenticated tensor-only .safetensors state-dict"
        )
    if args.state_dict is None:
        ap.error("--state-dict is required unless --self-test is used")
    if any(value is None for value in (args.output, args.config_out)):
        ap.error("--output and --config-out are required unless --self-test is used")

    output_path, config_path = validate_output_boundary(args.output, args.config_out)

    import torch

    sd = load_safetensors_state_dict(args.state_dict)
    tensors, dropped, non_tensor = {}, [], []
    for k, v in sd.items():
        if k in DROP:
            dropped.append(k)
            continue
        if not isinstance(v, torch.Tensor):
            non_tensor.append(k)
            continue
        tensors[k] = v.detach().to(torch.float32).contiguous()
    if non_tensor:
        die(f"state_dict has non-tensor entries: {non_tensor[:5]}")
    if not tensors:
        die("no tensors survived filtering")

    config = derive_config(tensors)

    publish_outputs(tensors, config, output_path, config_path)

    total = sum(t.numel() for t in tensors.values())
    print(f"tensors written : {len(tensors)} ({total:,} params)")
    print(f"dropped         : {dropped if dropped else '(none)'}")
    for path in (output_path, config_path):
        h = hashlib.sha256(open(path, "rb").read()).hexdigest()
        print(f"sha256 {h}  {path}")
    print(json.dumps(config, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
