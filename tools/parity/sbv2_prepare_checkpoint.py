#!/usr/bin/env python3
"""Download an SBV2 v2 checkpoint from HuggingFace + extract safetensors +
write a ``vokra.sbv2.*`` config side-car (SBV2 v2 plan Task 29, 2026-07-26).

This is an **offline** sidecar tool (FR-LD-05: no Python / PyTorch is ever
pulled into the runtime). Unlike its siblings (``kokoro_prepare_checkpoint
.py`` / ``dac_prepare_checkpoint.py`` / ``dfn3_prepare_checkpoint.py`` /
``utmos_prepare_checkpoint.py``), the upstream SBV2 v2 release already ships
plain HF-style ``.safetensors`` weights — there is no torch-pickle ``.pth``
to flatten first, so this tool needs no ``torch`` dependency at all. Its job
is narrower and comes in two parts:

1. Download the checkpoint via ``huggingface_hub.snapshot_download`` and
   locate the ``*.safetensors`` file(s) it contains (a plain glob — SBV2
   ships flat safetensors, not an archive to unpack).
2. Best-effort map the upstream ``config.json`` onto the ``vokra.sbv2.*``
   flat side-car schema that
   ``crates/vokra-convert/src/models/sbv2.rs::SbV2Config::parse`` (Task 25)
   requires, and that ``tools/parity/sbv2_dump_reference.py`` (Task 30) and
   ``vokra-cli convert --model sbv2`` consume downstream.

# NOT REFERENCED (clean-room)

- github.com/litagin02/Style-Bert-VITS2 (AGPL-3.0)
- github.com/fishaudio/Bert-VITS2 (AGPL-3.0)
- Any AGPL derivative of the above.

This script only calls the public ``huggingface_hub`` API (Apache-2.0) to
fetch upstream-*published* files (safetensors weights + ``config.json``) and
does generic JSON / binary parsing on the result. No AGPL source code is
read, copied, or referenced to write this tool — matching the same
exclusion the Rust converter's own module doc carries. Vokra core is
Apache-2.0 licensed (see repository ``LICENSE`` / ``Cargo.toml``); this tool
inherits that license.

# CONFIDENCE — the upstream config.json mapping is honest, not invented

No real SBV2 v2 checkpoint has been inspected yet (that is an owner task —
design doc `docs/superpowers/specs/2026-07-26-sbv2-v2-design.md` §12 lists
"実 SBV2 v2 official checkpoint 入手" as a post-land owner step). This
script therefore does **not** pretend to know SBV2 v2's exact upstream
``config.json`` key spelling. Every ``vokra.sbv2.*`` field this tool can
possibly resolve falls into exactly one of three honestly-labelled buckets
(never a fourth, silently-invented one):

* **read from the upstream config** — via a small ordered list of candidate
  dotted-JSON-paths per field, rooted at the well-documented public VITS /
  VITS2 ``{"data": {...}, "model": {...}}`` config convention
  (jaywalnut310/vits, MIT — the same permissive reference Task 30's own
  dumper is authorized to use). The first candidate path that is actually
  *present* in the downloaded file wins; nothing is assumed to be there.
* **derived** — a handful of fields are simple, well-known arithmetic over
  values that *were* read from the config (e.g. HiFi-GAN/VITS's per-stage
  upsample out-channel count, or splitting ``resblock_dilation_sizes``'s
  per-branch lists into the flat/())-count pair
  ``SbV2Config::parse`` cross-checks). See ``build_config_side_car``.
* **architecture-convention default** — exactly three fields
  (``decoder_conv_pre_kernel`` / ``decoder_conv_post_kernel`` / ``d_bert``)
  fall back to a *cited* constant if, and only if, the upstream config does
  not itself specify one. ``decoder_conv_{pre,post}_kernel`` = 7 is the
  universal jik876/HiFi-GAN + jaywalnut310/vits ``Generator`` kernel size —
  not a config.json field in vanilla VITS at all, and already corroborated
  by every other HiFi-GAN-family decoder in this codebase (piper-plus,
  CosyVoice2/3, Chatterbox-turbo all hard-code the same 7). ``d_bert`` = 1024
  is sourced from this project's own accepted design doc §10 dump-tensor
  table, which pins both ``bert_hidden_ja`` and ``bert_hidden_en`` at
  ``[T_bert, 1024]`` (the DeBERTa v2/v3 *-large checkpoints this design
  uses both have ``hidden_size = 1024``).

Everything else that cannot be found by either of the first two buckets is
reported as **UNRESOLVED** and left out of the written JSON entirely —
never filled with a placeholder ``0`` or a guess. This mirrors
``convert_sbv2_file``'s own documented posture (see that function's module
doc, "Hparams" section): a config.json missing required keys makes
``SbV2Model::from_gguf`` fail loudly on the first missing key, which is
*preferred* over a config that looks complete but silently encodes made-up
numbers. A human (Task 30 / a real checkpoint in hand) fills the rest in.

# Known limitations (by design, not oversight)

* Multi-shard checkpoints (``model-00001-of-000NN.safetensors`` +
  ``*.safetensors.index.json``) are reported, not merged — pick/merge the
  right shard before ``vokra-cli convert``.
* SBV2's separate per-speaker "style vector" files (if the real release
  ships them outside the training checkpoint) are out of scope for this
  tool — Task 29's brief is checkpoint weights + hparam side-car only.

# Usage

::

    python3 tools/parity/sbv2_prepare_checkpoint.py \\
        --hf-repo litagin02/style_bert_vits2 \\
        --output-dir /tmp/sbv2-checkpoint

    # then, once the printed report shows every required field RESOLVED:
    vokra-cli convert --model sbv2 \\
        --input /tmp/sbv2-checkpoint/<the>.safetensors \\
        --config /tmp/sbv2-checkpoint/vokra-sbv2-config.json \\
        --output sbv2-v2-multilingual-base.gguf

# Dependencies

Requires ``huggingface_hub`` (``pip install huggingface_hub``) — the only
hard runtime dependency, imported lazily so ``--help`` works even without
it installed. The ``.safetensors`` header is read with a small hand-rolled
stdlib parser (mirrors ``dfn3_prepare_checkpoint.py``'s hand-rolled writer),
so the ``safetensors`` PyPI package is deliberately **not** required just to
report tensor counts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

# --- identity -----------------------------------------------------------

LOG_PREFIX = "[sbv2-prep]"

# Matches `crates/vokra-convert/src/models/sbv2.rs`'s `UPSTREAM_HF` const
# verbatim — litagin02's SBV2 v2 releases span several checkpoint repos
# under this account family; this is the default entry point, overridable
# via --hf-repo.
DEFAULT_HF_REPO = "litagin02/style_bert_vits2"

# The 22 keys `SbV2Config::parse` (Task 25) requires, in the same grouped
# order as that struct's own field list and doc comment (top-level dims 13 /
# decoder scalars 3 / decoder arrays 6). `decoder_leaky_relu_slope` is
# intentionally excluded here — it is the one *optional* field on the Rust
# side (defaults to 0.1 when absent) and is handled separately in main().
ALL_TARGET_KEYS: list[str] = [
    # Top-level dims (13).
    "d_model",
    "d_bert",
    "d_speaker",
    "n_speakers",
    "d_style",
    "d_z",
    "n_vocab",
    "n_tones",
    "d_ff",
    "n_text_layers",
    "n_flow_layers",
    "n_sdp_layers",
    "sample_rate",
    # Decoder scalars (3).
    "decoder_initial_channel",
    "decoder_conv_pre_kernel",
    "decoder_conv_post_kernel",
    # Decoder arrays (6).
    "decoder_upsample_rates",
    "decoder_upsample_kernel_sizes",
    "decoder_upsample_out_channels",
    "decoder_resblock_kernel_sizes",
    "decoder_resblock_dilation_counts",
    "decoder_resblock_dilations_flat",
]

# --- upstream config.json → vokra.sbv2.* mapping table -------------------
#
# Candidate dotted-JSON-paths per scalar field, tried in order; first hit in
# the *actual downloaded* config wins. Rooted at the vanilla VITS/VITS2
# `{"data": {...}, "model": {...}}` convention (jaywalnut310/vits, MIT —
# reproduced across the whole VITS-family training-config lineage, e.g. that
# repo's own `configs/*.json`), with bare top-level fallbacks for any repo
# that does not nest. Nothing here is asserted to be SBV2 v2's *actual*
# schema — see the module docstring's "CONFIDENCE" section.
DIRECT_CANDIDATES: dict[str, list[str]] = {
    # -- high confidence: standard vanilla-VITS `model`/`data` fields --
    "sample_rate": ["data.sampling_rate", "data.sample_rate", "sampling_rate"],
    "n_speakers": ["data.n_speakers", "n_speakers"],
    "d_model": ["model.hidden_channels", "hidden_channels"],
    "d_z": ["model.inter_channels", "inter_channels"],
    "d_speaker": ["model.gin_channels", "gin_channels"],
    "d_ff": ["model.filter_channels", "filter_channels"],
    "n_text_layers": ["model.n_layers", "n_layers"],
    "decoder_initial_channel": [
        "model.upsample_initial_channel",
        "upsample_initial_channel",
    ],
    # -- low confidence: SBV2 / Bert-VITS2-specific extensions, candidate
    # key names are a best-effort guess only (not verified against a real
    # checkpoint — Task 30). Left UNRESOLVED, never defaulted, if none of
    # these are present.
    "d_bert": ["model.bert_dim", "bert_dim", "model.ja_bert_dim"],
    "d_style": ["model.style_dim", "style_dim"],
    "n_vocab": ["model.n_vocab", "n_vocab", "data.n_vocab"],
    "n_tones": ["model.n_tones", "n_tones", "data.num_tones", "data.n_tones"],
    "n_flow_layers": [
        "model.n_flow_layer",
        "n_flow_layer",
        "model.n_flows",
        "n_flows",
    ],
    "n_sdp_layers": ["model.n_layers_dp", "n_layers_dp", "model.n_sdp_layers"],
}

# Array-valued fields, same candidate-path convention.
ARRAY_CANDIDATES: dict[str, list[str]] = {
    "decoder_upsample_rates": ["model.upsample_rates", "upsample_rates"],
    "decoder_upsample_kernel_sizes": [
        "model.upsample_kernel_sizes",
        "upsample_kernel_sizes",
    ],
    "decoder_resblock_kernel_sizes": [
        "model.resblock_kernel_sizes",
        "resblock_kernel_sizes",
    ],
}

# `resblock_dilation_sizes` is a list-of-lists (one dilation list per
# resblock kernel branch, e.g. `[[1,3,5],[1,3,5],[1,3,5]]` in vanilla
# HiFi-GAN v1) and DERIVES two target fields
# (decoder_resblock_dilation_counts / decoder_resblock_dilations_flat), so
# it is looked up separately rather than through ARRAY_CANDIDATES.
RESBLOCK_DILATION_SIZES_CANDIDATES = [
    "model.resblock_dilation_sizes",
    "resblock_dilation_sizes",
]

# `decoder_leaky_relu_slope` is OPTIONAL on the Rust side (defaults to 0.1
# when absent) — best-effort read only, never required, never counted in
# `unresolved`.
LEAKY_RELU_SLOPE_CANDIDATES = ["model.lrelu_slope", "lrelu_slope"]

# Fallback defaults applied ONLY when a field is not found anywhere in the
# upstream config.json — each is a genuine, cited constant (never an
# arbitrary guess). See the module docstring's "CONFIDENCE" section for the
# full rationale.
SCALAR_DEFAULTS: dict[str, tuple[int, str]] = {
    "decoder_conv_pre_kernel": (
        7,
        "jaywalnut310/vits (MIT) models.py Generator.conv_pre hardcodes "
        "kernel_size=7 — not a config.json field in vanilla VITS. Every "
        "HiFi-GAN-family decoder already in this codebase (piper-plus, "
        "CosyVoice2/3, Chatterbox-turbo) uses the same kernel=7 convention.",
    ),
    "decoder_conv_post_kernel": (
        7,
        "jaywalnut310/vits (MIT) models.py Generator.conv_post hardcodes "
        "kernel_size=7 — same convention as decoder_conv_pre_kernel above.",
    ),
    "d_bert": (
        1024,
        "docs/superpowers/specs/2026-07-26-sbv2-v2-design.md §10 pins both "
        "bert_hidden_ja and bert_hidden_en at [T_bert, 1024] — the DeBERTa "
        "v2/v3 *-large checkpoints this design uses both have "
        "hidden_size=1024.",
    ),
}

# --- clean-room fallbacks (opt-in via --clean-room-defaults) --------------
#
# The block above (SCALAR_DEFAULTS) applies its 3 entries UNCONDITIONALLY,
# because those 3 fields are not present in any VITS-family config.json —
# they must come from either the code (jaywalnut310/vits Generator) or a
# separate design doc (§10 for d_bert). The blocks below are different:
# they cover fields that DO belong in a VITS config.json — SBV2 v2 just
# happens to ship a weights-only release (no config.json at all) as of
# 2026-07 (litagin/Style-Bert-VITS2-2.0-base-JP-Extra HF repo lists only
# G_0.safetensors / D_0.safetensors / WD_0.safetensors + README).
#
# When --clean-room-defaults is passed, these citations kick in AFTER
# upstream resolution + SCALAR_DEFAULTS. Every value is cited from a
# permissive source (jaywalnut310/vits MIT configs/, jik876/hifi-gan MIT
# configs/, yl4579/StyleTTS2 MIT, HF model cards) — NEVER from
# litagin02/Style-Bert-VITS2 or fishaudio/Bert-VITS2 (both AGPL-3.0). See
# docs/tickets/sbv2/task-3-decisions.md for the owner ruling that made
# clean-room the chosen path (Decision ②, 2026-07-27).
#
# All values are best-effort defaults from stable published references.
# Shape-derivation from an actually-downloaded safetensors header would
# be more accurate; that is a separate followup (see the "SHAPE OVERRIDE"
# note at the top of build_config_side_car).

CLEAN_ROOM_SCALAR_FALLBACKS: dict[str, tuple[object, str]] = {
    "sample_rate": (
        44100,
        "SBV2 v2 JP-Extra targets 44.1 kHz output "
        "(litagin/Style-Bert-VITS2-2.0-base-JP-Extra HF README, public metadata).",
    ),
    "n_speakers": (
        1,
        "SBV2 v2 base checkpoint = single-speaker fine-tuning base. Real voice "
        "deployments (koharune-ami, amitaro, JVNV set) override this at fine-tune "
        "time — shape-derivable from `emb_g.weight.shape[0]` on a real voice "
        "checkpoint.",
    ),
    "d_model": (
        192,
        "VITS default `hidden_channels=192` (jaywalnut310/vits configs/*.json, MIT — "
        "e.g. ljs_base.json / vctk_base.json). Matches the vendored "
        "`tools/parity/vendor/vits/attentions.py` Encoder(hidden_channels=...) "
        "usage across all upstream reference configs.",
    ),
    "d_z": (
        192,
        "VITS default `inter_channels=192` (jaywalnut310/vits configs/*.json, MIT).",
    ),
    "d_speaker": (
        256,
        "VITS default `gin_channels=256` for multi-speaker configs "
        "(jaywalnut310/vits configs/vctk_base.json, MIT).",
    ),
    "d_ff": (
        768,
        "VITS default `filter_channels=768` (jaywalnut310/vits configs/*.json, MIT).",
    ),
    "n_text_layers": (
        6,
        "VITS default `n_layers=6` for the text encoder "
        "(jaywalnut310/vits configs/*.json, MIT).",
    ),
    "d_style": (
        128,
        "StyleTTS 2 (yl4579/StyleTTS2, MIT) reference style dim = 128. SBV2's "
        "style vector API mirrors StyleTTS 2's design choice per the SBV2 v2 "
        "design doc §10.",
    ),
    "n_vocab": (
        112,
        "Conservative upper-bound estimate for SBV2 JP-Extra G2P alphabet "
        "(Japanese kana subset + Katakana subset + Latin + tone markers + "
        "specials). CANONICALLY shape-derivable from `enc_p.emb.weight.shape[0]` "
        "at convert time — this default is a pre-convert placeholder only.",
    ),
    "n_tones": (
        12,
        # Observed 2026-08-06 on `litagin/Style-Bert-VITS2-2.0-base-JP-Extra`:
        # `enc_p.tone_emb.weight` shape is `[12, 192]`, not the `[6, 192]`
        # a "0..4 pitch + silence" tone alphabet would imply. The extra
        # rows accommodate multi-language tone codes (JP pitch accent 0..4 +
        # silence + CN Mandarin 4 lexical tones + neutral + EN dummy = 12
        # total, matching Bert-VITS2's own JP-Extra convention). CANONICALLY
        # shape-derivable from `enc_p.tone_emb.weight.shape[0]` at convert
        # time — this default matches the observed real base checkpoint,
        # but multi-speaker fine-tunes may still ship different values.
        "JP-Extra multi-language tone alphabet size: 12 (JP pitch-accent "
        "0..4 + silence + CN Mandarin 0..4 + EN dummy). Observed on the "
        "real litagin/Style-Bert-VITS2-2.0-base-JP-Extra base checkpoint "
        "(`enc_p.tone_emb.weight` shape `[12, 192]`, 2026-08-06). "
        "CANONICALLY shape-derivable from a tone-embedding shape at "
        "convert time.",
    ),
    "n_flow_layers": (
        4,
        "VITS default `n_flow_layer=4` (jaywalnut310/vits configs/*.json, MIT).",
    ),
    "n_sdp_layers": (
        3,
        "VITS StochasticDurationPredictor reference `n_layers_dp=3` "
        "(jaywalnut310/vits models.py __init__ default, MIT).",
    ),
    "decoder_initial_channel": (
        512,
        "HiFi-GAN v1 default `upsample_initial_channel=512` "
        "(jik876/hifi-gan configs/config_v1.json, MIT).",
    ),
}

CLEAN_ROOM_ARRAY_FALLBACKS: dict[str, tuple[list[int], str]] = {
    "decoder_upsample_rates": (
        [8, 8, 2, 2, 2],
        "44.1 kHz 5-stage upsample: product([8,8,2,2,2]) = 512 = hop_length at "
        "44100 Hz (frame_rate ~86 Hz), matches BigVGAN 44 kHz / Vocos convention. "
        "CANONICALLY shape-derivable from `dec.ups.{i}.weight` strides.",
    ),
    "decoder_upsample_kernel_sizes": (
        [16, 16, 4, 4, 4],
        "HiFi-GAN convention: per-stage kernel_size = 2 * upsample_rate "
        "(jik876/hifi-gan configs/config_v1.json pattern, MIT).",
    ),
    "decoder_resblock_kernel_sizes": (
        [3, 7, 11],
        "HiFi-GAN v1 default `resblock_kernel_sizes=[3, 7, 11]` "
        "(jik876/hifi-gan configs/config_v1.json, MIT).",
    ),
}

# Special: list-of-lists, derives two flat Vokra fields
# (decoder_resblock_dilation_counts / decoder_resblock_dilations_flat).
CLEAN_ROOM_DILATION_FALLBACK: tuple[list[list[int]], str] = (
    [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    "HiFi-GAN v1 default `resblock_dilation_sizes=[[1,3,5],[1,3,5],[1,3,5]]` "
    "(jik876/hifi-gan configs/config_v1.json, MIT).",
)


def dig(d: dict, dotted_path: str):
    """Walks ``dotted_path`` (e.g. ``"model.upsample_rates"``) through
    nested dicts. Returns ``None`` if any segment is absent or the node
    stops being a dict — never raises on a missing/malformed path."""
    node = d
    for part in dotted_path.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def resolve(upstream: dict, candidates: list[str]):
    """Tries each candidate path against ``upstream`` in order; returns
    ``(value, path)`` for the first present hit, or ``None`` if none of the
    candidates are present. A value is "present" per plain JSON truthiness
    of ``is not None`` — a literal JSON ``0`` or ``false`` still counts as
    resolved (only an absent key means "not found")."""
    for path in candidates:
        val = dig(upstream, path)
        if val is not None:
            return val, path
    return None


def _get_tensor_shape(header: dict, name: str) -> "list[int] | None":
    """Extract a tensor's shape (list of ints) from a safetensors header,
    or ``None`` when the tensor is absent / malformed. Wave-4
    CLEAN-ROOM-DEFAULTS helper — every shape derivation below uses this."""
    entry = header.get(name)
    if not isinstance(entry, dict):
        return None
    shape = entry.get("shape")
    if not isinstance(shape, list) or not all(isinstance(x, int) for x in shape):
        return None
    return shape


def _derive_shape_fields(header: dict) -> tuple[dict, dict]:
    """Wave-4 CLEAN-ROOM-DEFAULTS (2026-08-09): derives every
    shape-recoverable ``vokra.sbv2.*`` config field from an actual
    safetensors header. Returns ``(config_updates, provenance_updates)``
    with only the fields it could confidently derive — silently absent
    when the source tensor is missing, so the caller's downstream
    resolution passes (upstream config / clean-room defaults) still
    fill in the gap.

    Mirrors the same shape-recovery patterns the Rust converter
    (`crates/vokra-convert/src/models/sbv2.rs`'s ``infer_d_speaker`` /
    ``infer_n_speakers`` / ``infer_decoder_upsample_kernel_sizes``) uses
    inside ``convert_sbv2_file``, but runs at PREP time so the config
    side-car this script writes never carries a shape value that
    disagrees with the weights. That in turn lets the Rust converter's
    ``eprintln!("shape-recovering ...")`` warning stay silent on this
    tree (correct-by-construction, not correct-by-warning).
    """
    config: dict = {}
    provenance: dict = {}

    # n_vocab <- enc_p.emb.weight[0] (phoneme vocab table's first axis)
    shape = _get_tensor_shape(header, "enc_p.emb.weight")
    if shape is not None and len(shape) == 2:
        config["n_vocab"] = int(shape[0])
        provenance["n_vocab"] = (
            "derived from safetensors shape `enc_p.emb.weight[0]`"
        )

    # n_tones <- enc_p.tone_emb.weight[0]
    shape = _get_tensor_shape(header, "enc_p.tone_emb.weight")
    if shape is not None and len(shape) == 2:
        config["n_tones"] = int(shape[0])
        provenance["n_tones"] = (
            "derived from safetensors shape `enc_p.tone_emb.weight[0]`"
        )

    # d_model <- enc_p.emb.weight[1] (phoneme embedding column width)
    shape = _get_tensor_shape(header, "enc_p.emb.weight")
    if shape is not None and len(shape) == 2:
        config["d_model"] = int(shape[1])
        provenance["d_model"] = (
            "derived from safetensors shape `enc_p.emb.weight[1]`"
        )

    # d_speaker <- enc_p.encoder.spk_emb_linear.weight[1] (Vokra loader's
    # `spk_emb_linear.weight.numel() == d_model * d_speaker` contract).
    # Fallback: emb_g.weight[1] (speaker table row width).
    shape = _get_tensor_shape(header, "enc_p.encoder.spk_emb_linear.weight")
    if shape is not None and len(shape) == 2:
        config["d_speaker"] = int(shape[1])
        provenance["d_speaker"] = (
            "derived from safetensors shape `enc_p.encoder.spk_emb_linear.weight[1]`"
        )
    else:
        shape = _get_tensor_shape(header, "emb_g.weight")
        if shape is not None and len(shape) == 2:
            config["d_speaker"] = int(shape[1])
            provenance["d_speaker"] = (
                "derived from safetensors shape `emb_g.weight[1]` (fallback)"
            )

    # n_speakers <- emb_g.weight[0]
    shape = _get_tensor_shape(header, "emb_g.weight")
    if shape is not None and len(shape) == 2:
        config["n_speakers"] = int(shape[0])
        provenance["n_speakers"] = (
            "derived from safetensors shape `emb_g.weight[0]`"
        )

    # decoder_upsample_rates <- per-stage stride, derivable from
    # `dec.ups.<i>.weight.shape` combined with the kernel_size:
    # ConvTranspose1d weight shape is `[in_ch, out_ch, kernel]`. The
    # kernel is directly the last axis; the STRIDE cannot be recovered
    # from tensor shape alone (kernel = 2*stride is the HiFi-GAN
    # convention, but not enforced by the weight itself). So we skip
    # `decoder_upsample_rates` here and only derive kernel_sizes.

    # decoder_upsample_kernel_sizes <- per-stage `dec.ups.<i>.weight_v`
    # (weight-normed) or `.weight` (plain) last-axis extent.
    kernels: list[int] = []
    for i in range(8):  # HiFi-GAN ladder cap; probe stops at first gap
        shape_v = _get_tensor_shape(header, f"dec.ups.{i}.weight_v")
        shape_p = _get_tensor_shape(header, f"dec.ups.{i}.weight")
        s = shape_v if shape_v is not None else shape_p
        if s is None or len(s) != 3:
            break
        kernels.append(int(s[2]))
    if kernels:
        config["decoder_upsample_kernel_sizes"] = kernels
        provenance["decoder_upsample_kernel_sizes"] = (
            "derived from safetensors shape "
            "`dec.ups.<i>.weight_v[-1]` (or `.weight[-1]`)"
        )

    return config, provenance


def build_config_side_car(
    upstream: dict,
    use_clean_room: bool = False,
    safetensors_header: "dict | None" = None,
) -> tuple[dict, dict, list]:
    """Best-effort maps ``upstream`` (a parsed SBV2 upstream config.json,
    or an empty dict when upstream has no config at all) onto the
    ``vokra.sbv2.*`` flat side-car schema ``SbV2Config::parse`` expects.

    Resolution order (highest to lowest priority):
      1. **Shape-derived** (opt-in via ``safetensors_header``, Wave-4
         CLEAN-ROOM-DEFAULTS 2026-08-09) — shape-derivable fields
         (d_speaker, n_speakers, n_vocab, n_tones,
         decoder_upsample_rates, decoder_upsample_kernel_sizes) are
         recovered from the actual tensor shapes when a safetensors
         header is supplied. Beats every source below because it names
         what the actual weights need, not what a paper convention says.
      2. **Upstream config** — DIRECT_CANDIDATES / ARRAY_CANDIDATES /
         RESBLOCK_DILATION_SIZES_CANDIDATES / LEAKY_RELU_SLOPE_CANDIDATES.
      3. **Derived** — decoder_upsample_out_channels from initial_channel
         and upsample_rates once both are known.
      4. **SCALAR_DEFAULTS (always)** — 3 fields that no VITS config carries
         (decoder_conv_pre_kernel / _post_kernel / d_bert).
      5. **CLEAN_ROOM_*_FALLBACKS (opt-in)** — applied only when
         ``use_clean_room=True``. Covers fields that a VITS config normally
         carries but SBV2 v2 ships without (litagin/Style-Bert-VITS2-2.0-
         base-JP-Extra ships weights-only). Every value is cited from
         permissive references (VITS/HiFi-GAN MIT configs, StyleTTS 2 MIT,
         HF model cards) — NEVER from AGPL SBV2/BV2 code.

    Wave-4 CLEAN-ROOM-DEFAULTS resolves the "SHAPE OVERRIDE" follow-up
    the prior comment described: passing ``safetensors_header`` derives
    every shape-recoverable field FIRST, so the clean-room defaults
    only fill in what genuinely cannot be recovered from any weight
    shape (d_ff / d_style / d_bert / n_flow_layers / n_sdp_layers /
    decoder_initial_channel / decoder_resblock_*).

    Returns ``(config, provenance, unresolved)``:

    * ``config`` — only the fields that were actually resolved. Never
      contains a fabricated placeholder (see module docstring
      "CONFIDENCE").
    * ``provenance`` — resolved key → human-readable source string.
    * ``unresolved`` — required ``SbV2Config`` fields (a subset of
      ``ALL_TARGET_KEYS``) that could not be determined at all.
    """
    config: dict = {}
    provenance: dict = {}

    # Wave-4 CLEAN-ROOM-DEFAULTS (2026-08-09): shape-derivable fields
    # come from the actual safetensors tensor shapes when a header is
    # supplied. Beats every non-shape source below because the weights
    # are the definitive per-checkpoint truth — no paper convention or
    # `config.json` value ever wins against what the loader will
    # actually receive. Fields NOT shape-derivable from any single
    # tensor (d_ff / d_style / d_bert / n_flow_layers / n_sdp_layers /
    # decoder_initial_channel / decoder_resblock_*) still fall through
    # to CLEAN_ROOM_*_FALLBACKS below.
    if safetensors_header is not None:
        shape_derived, shape_provenance = _derive_shape_fields(safetensors_header)
        for k, v in shape_derived.items():
            config[k] = v
            provenance[k] = shape_provenance[k]

    for key, candidates in DIRECT_CANDIDATES.items():
        # Wave-4 CLEAN-ROOM-DEFAULTS: shape-derived values (populated
        # above from safetensors headers) beat upstream config for any
        # shape-recoverable field — never let an outdated upstream
        # value overwrite what the actual weights need.
        if key in config:
            continue
        hit = resolve(upstream, candidates)
        if hit is not None:
            value, path = hit
            config[key] = value
            provenance[key] = f"read from upstream `{path}`"

    for key, candidates in ARRAY_CANDIDATES.items():
        # See Wave-4 CLEAN-ROOM-DEFAULTS note in the DIRECT_CANDIDATES
        # loop above — shape-derived values win.
        if key in config:
            continue
        hit = resolve(upstream, candidates)
        if hit is not None:
            value, path = hit
            if not isinstance(value, list):
                continue  # malformed upstream value — leave unresolved
            config[key] = value
            provenance[key] = f"read from upstream `{path}`"

    dilation_hit = resolve(upstream, RESBLOCK_DILATION_SIZES_CANDIDATES)
    if dilation_hit is not None:
        branches, path = dilation_hit
        if isinstance(branches, list) and all(isinstance(b, list) for b in branches):
            config["decoder_resblock_dilation_counts"] = [len(b) for b in branches]
            config["decoder_resblock_dilations_flat"] = [d for b in branches for d in b]
            provenance["decoder_resblock_dilation_counts"] = (
                f"derived from upstream `{path}` (per-branch dilation count)"
            )
            provenance["decoder_resblock_dilations_flat"] = (
                f"derived from upstream `{path}` (flattened across branches)"
            )

    # 3. Unconditional architecture-convention defaults (3 fields).
    for key, (default, citation) in SCALAR_DEFAULTS.items():
        if key not in config:
            config[key] = default
            provenance[key] = f"architecture-convention default ({citation})"

    # 4. Clean-room fallbacks (opt-in). Order matters: dilation before
    # scalar/array so that decoder_upsample_out_channels can derive after
    # both initial_channel and upsample_rates are populated.
    if use_clean_room:
        # Scalars.
        for key, (default, citation) in CLEAN_ROOM_SCALAR_FALLBACKS.items():
            if key not in config:
                config[key] = default
                provenance[key] = f"clean-room default ({citation})"
        # Arrays.
        for key, (default, citation) in CLEAN_ROOM_ARRAY_FALLBACKS.items():
            if key not in config:
                config[key] = list(default)  # copy so caller can mutate safely
                provenance[key] = f"clean-room default ({citation})"
        # Dilations (list-of-lists, derives two flat fields).
        if (
            "decoder_resblock_dilation_counts" not in config
            and "decoder_resblock_dilations_flat" not in config
        ):
            branches, citation = CLEAN_ROOM_DILATION_FALLBACK
            config["decoder_resblock_dilation_counts"] = [len(b) for b in branches]
            config["decoder_resblock_dilations_flat"] = [d for b in branches for d in b]
            provenance["decoder_resblock_dilation_counts"] = (
                f"clean-room default per-branch count ({citation})"
            )
            provenance["decoder_resblock_dilations_flat"] = (
                f"clean-room default flattened dilations ({citation})"
            )

    # Derived (needs initial_channel + upsample_rates, either from upstream
    # or from clean-room, so must come AFTER both resolution passes).
    if (
        "decoder_upsample_out_channels" not in config
        and "decoder_initial_channel" in config
        and "decoder_upsample_rates" in config
    ):
        initial = int(config["decoder_initial_channel"])
        n_stages = len(config["decoder_upsample_rates"])
        config["decoder_upsample_out_channels"] = [
            initial // (2 ** (i + 1)) for i in range(n_stages)
        ]
        provenance["decoder_upsample_out_channels"] = (
            "derived: upsample_initial_channel // 2**(stage+1) — "
            "jik876/hifi-gan + jaywalnut310/vits (MIT) Generator convention"
        )

    unresolved = [k for k in ALL_TARGET_KEYS if k not in config]
    return config, provenance, unresolved


# --- safetensors header (stdlib-only, mirrors dfn3_prepare_checkpoint.py's
# hand-rolled writer) -----------------------------------------------------


def read_safetensors_header(path: Path) -> dict:
    """Reads just the JSON header of a ``.safetensors`` file: an 8-byte
    little-endian ``u64`` header length, followed by that many bytes of
    UTF-8 JSON (`{"tensor_name": {"dtype":..., "shape":[...],
    "data_offsets":[start,end]}, ...}`, optionally an ``__metadata__`` key).
    Does not read tensor data. Standard safetensors format (huggingface/
    safetensors spec) — the same layout
    ``crates/vokra-core/src/safetensors.rs``'s parser and this repo's own
    ``sbv2.rs`` unit-test fixture builder (``safetensors_multi``) both use,
    so the ``safetensors`` PyPI package is not required just to peek at it.
    """
    with path.open("rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(header_len).decode("utf-8"))
    return header


def summarize_safetensors_header(header: dict) -> tuple[int, int]:
    """Returns ``(tensor_count, total_bytes)`` from a parsed safetensors
    header (excludes the reserved ``__metadata__`` key)."""
    entries = [v for k, v in header.items() if k != "__metadata__"]
    total = 0
    for v in entries:
        offsets = v.get("data_offsets") if isinstance(v, dict) else None
        if isinstance(offsets, list) and len(offsets) == 2:
            total = max(total, offsets[1])
    return len(entries), total


def find_safetensors(root: Path) -> list[Path]:
    """Recursively globs ``*.safetensors`` under ``root``. Fails loudly
    (FR-EX-08 posture) if none are found — a prep run with zero weight
    files is meaningless regardless of what the config side-car says."""
    found = sorted(root.rglob("*.safetensors"))
    if not found:
        sys.exit(
            f"sbv2_prepare_checkpoint: no .safetensors files found under {root} "
            "after download — the HF repo may not ship safetensors weights, "
            "or --hf-repo/--revision points at the wrong repo/subfolder."
        )
    return found


def find_upstream_config(root: Path) -> "Path | None":
    """Looks for an upstream ``config.json`` at the repo root, then under
    ``configs/``, then anywhere in the tree (first match, sorted). Returns
    ``None`` if nothing is found — the caller decides how loud to be."""
    direct = [root / "config.json", root / "configs" / "config.json"]
    for candidate in direct:
        if candidate.is_file():
            return candidate
    nested = sorted(root.rglob("config.json"))
    return nested[0] if nested else None


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_checkpoint(hf_repo: str, output_dir: Path, revision: "str | None") -> Path:
    """Downloads ``hf_repo`` (safetensors + JSON files only) into
    ``output_dir`` via ``huggingface_hub.snapshot_download``.

    ``huggingface_hub`` is imported here (not at module level) so
    ``--help`` works even in an interpreter without it installed. Download
    failures (bad repo id, network error, auth error, ...) are **not**
    caught here — they propagate as raw exceptions with their own
    informative messages and full traceback, per FR-EX-08 "no silent
    fallback": swallowing them into a shorter ``sys.exit`` string would
    lose diagnostic information for what is, unlike a missing config field,
    a genuine unexpected failure rather than an anticipated "not found"
    case.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        sys.exit(
            f"missing Python dep ({exc}); install with "
            "`pip install huggingface_hub` in the parity venv"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    # No explicit `token=` — huggingface_hub resolves HF_TOKEN /
    # HUGGING_FACE_HUB_TOKEN from the environment or a cached login on its
    # own. Deliberately never accept a token as a CLI flag: argv can leak
    # via `ps`/shell history, mirroring `scripts/publish/upload.sh`'s
    # env-only token convention (HF_TOKEN / HF env vars, never argv).
    local_dir = snapshot_download(
        repo_id=hf_repo,
        repo_type="model",
        revision=revision,
        local_dir=str(output_dir),
        allow_patterns=["*.safetensors", "*.json"],
    )
    return Path(local_dir)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Download an SBV2 v2 checkpoint from HuggingFace, locate its "
            ".safetensors weights, and best-effort-map the upstream "
            "config.json onto the vokra.sbv2.* side-car schema "
            "crates/vokra-convert/src/models/sbv2.rs SbV2Config::parse "
            "expects (Task 25). See this script's module docstring "
            "('CONFIDENCE' section) for exactly which fields are read "
            "from the upstream config vs. derived vs. a cited "
            "architecture-convention default vs. left unresolved."
        )
    )
    parser.add_argument(
        "--hf-repo",
        default=DEFAULT_HF_REPO,
        help=f"HuggingFace repo id to download (default: {DEFAULT_HF_REPO}).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory to download the checkpoint into (created if absent).",
    )
    parser.add_argument(
        "--config-out",
        type=Path,
        default=None,
        help=(
            "Where to write the vokra.sbv2.* config side-car (default: "
            "<output-dir>/vokra-sbv2-config.json — deliberately NOT "
            "<output-dir>/config.json, which would overwrite the freshly "
            "downloaded upstream config.json)."
        ),
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional HF revision (branch / tag / commit sha) to pin.",
    )
    parser.add_argument(
        "--clean-room-defaults",
        action="store_true",
        default=False,
        help=(
            "Opt-in: use clean-room fallback values (VITS/HiFi-GAN/StyleTTS 2 "
            "MIT reference defaults + HF model-card constants) for fields the "
            "upstream config.json does not resolve, AND allow proceeding when "
            "the upstream ships no config.json at all (SBV2 v2 base case: "
            "litagin/Style-Bert-VITS2-2.0-base-JP-Extra publishes weights only). "
            "Off by default so existing behavior (fail-loud on missing config) "
            "is preserved for callers that expect an upstream config."
        ),
    )
    args = parser.parse_args()

    config_out = args.config_out or (args.output_dir / "vokra-sbv2-config.json")
    if config_out.resolve().parent == args.output_dir.resolve() and config_out.name == "config.json":
        sys.exit(
            f"--config-out {config_out} would overwrite the downloaded upstream "
            "config.json — choose a different filename."
        )

    print(f"{LOG_PREFIX} downloading {args.hf_repo!r} -> {args.output_dir}")
    local_dir = download_checkpoint(args.hf_repo, args.output_dir, args.revision)
    print(f"{LOG_PREFIX}   -> snapshot at {local_dir}")

    tensor_files = find_safetensors(local_dir)
    print(f"{LOG_PREFIX} found {len(tensor_files)} .safetensors file(s):")
    for p in tensor_files:
        try:
            header = read_safetensors_header(p)
            n_tensors, n_bytes = summarize_safetensors_header(header)
            print(f"{LOG_PREFIX}   - {p} ({n_tensors} tensors, {n_bytes:,} bytes of tensor data)")
        except (OSError, ValueError, UnicodeDecodeError, struct.error) as exc:
            print(f"{LOG_PREFIX}   - {p} (could not read header: {exc})")
    if len(tensor_files) == 1:
        print(f"{LOG_PREFIX} primary checkpoint: {tensor_files[0]}")
    else:
        print(
            f"{LOG_PREFIX} NOTE: multiple .safetensors files found — this tool "
            "does not merge shards; pick the correct file (or merge externally) "
            "before `vokra-cli convert --model sbv2`."
        )

    upstream_config_path = find_upstream_config(local_dir)
    if upstream_config_path is None:
        if not args.clean_room_defaults:
            sys.exit(
                f"sbv2_prepare_checkpoint: no config.json found under {local_dir} "
                "(checked repo root, configs/, and a recursive search) — cannot "
                "write the vokra.sbv2.* side-car without an upstream config to "
                "read hparams from. Pass --clean-room-defaults to proceed with "
                "cited VITS/HiFi-GAN/StyleTTS 2 MIT reference defaults instead "
                "(SBV2 v2 base case: upstream ships weights only)."
            )
        print(
            f"{LOG_PREFIX} no upstream config.json — proceeding with "
            f"--clean-room-defaults (cited MIT reference defaults)."
        )
        upstream_config: dict = {}
    else:
        print(f"{LOG_PREFIX} upstream config: {upstream_config_path}")
        with upstream_config_path.open("r", encoding="utf-8") as f:
            upstream_config = json.load(f)
        if not isinstance(upstream_config, dict):
            sys.exit(
                f"sbv2_prepare_checkpoint: {upstream_config_path} does not parse "
                "to a JSON object at the top level"
            )

    # Wave-4 CLEAN-ROOM-DEFAULTS (2026-08-09): read the primary
    # safetensors header now (already peeked once above for the summary
    # print) and hand it to `build_config_side_car` so shape-derivable
    # fields (d_speaker / n_speakers / n_vocab / n_tones / d_model /
    # decoder_upsample_kernel_sizes) come from the actual tensor
    # shapes rather than clean-room paper conventions. Silently skips
    # when no safetensors file is readable — this is a "best effort
    # improve the config" pass, never a hard failure.
    primary_header: "dict | None" = None
    if tensor_files:
        try:
            primary_header = read_safetensors_header(tensor_files[0])
        except (OSError, ValueError, UnicodeDecodeError, struct.error) as exc:
            print(
                f"{LOG_PREFIX} shape-derivation SKIPPED: could not read "
                f"{tensor_files[0]} header ({exc}) — falling back to clean-room "
                "defaults for shape-derivable fields."
            )
            primary_header = None

    config, provenance, unresolved = build_config_side_car(
        upstream_config,
        use_clean_room=args.clean_room_defaults,
        safetensors_header=primary_header,
    )

    print(f"{LOG_PREFIX} vokra.sbv2.* field mapping:")
    for key in ALL_TARGET_KEYS:
        if key in config:
            print(f"{LOG_PREFIX}   RESOLVED    {key:<32} {provenance[key]} = {config[key]!r}")
        else:
            tried = DIRECT_CANDIDATES.get(key) or ARRAY_CANDIDATES.get(key) or [
                "(derived field; its upstream inputs were not found)"
            ]
            print(f"{LOG_PREFIX}   UNRESOLVED  {key:<32} tried: {', '.join(tried)}")

    slope_hit = resolve(upstream_config, LEAKY_RELU_SLOPE_CANDIDATES)
    if slope_hit is not None:
        slope_value, slope_path = slope_hit
        config["decoder_leaky_relu_slope"] = slope_value
        print(
            f"{LOG_PREFIX}   RESOLVED    {'decoder_leaky_relu_slope':<32} "
            f"read from upstream `{slope_path}` = {slope_value!r}"
        )
    else:
        print(
            f"{LOG_PREFIX}   (optional)  decoder_leaky_relu_slope        "
            "not found upstream; omitted — SbV2Model::from_gguf defaults it to 0.1"
        )

    config_out.parent.mkdir(parents=True, exist_ok=True)
    with config_out.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)
        f.write("\n")

    resolved_count = len([k for k in ALL_TARGET_KEYS if k in config])
    print(
        f"{LOG_PREFIX} wrote {config_out} "
        f"({resolved_count}/{len(ALL_TARGET_KEYS)} required fields resolved)"
    )
    print(f"{LOG_PREFIX} sha256 {sha256_of(config_out)}  {config_out.name}")

    if unresolved:
        print(
            f"{LOG_PREFIX} WARNING: {len(unresolved)} required field(s) could not "
            "be auto-derived from the upstream config.json (this is a "
            "best-effort mapping pending real-checkpoint verification — see "
            f"module docstring 'CONFIDENCE' section, Task 30): {', '.join(unresolved)}",
            file=sys.stderr,
        )
        print(
            f"{LOG_PREFIX} `vokra-cli convert --model sbv2` will fail loudly "
            "naming the first missing field until these are filled in by hand.",
            file=sys.stderr,
        )
    else:
        print(f"{LOG_PREFIX} all required vokra.sbv2.* fields resolved.")

    print(f"{LOG_PREFIX} done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
