#!/usr/bin/env python3
"""Dump upstream Voxtral reference tensors for M3-10-T19/T20 parity.

This is an **offline** tool (FR-LD-05: no Python/PyTorch ever enters the
runtime). It generates the committed fixture set under ``tests/parity/voxtral/``
that ``crates/vokra-models/tests/parity_voxtral.rs`` compares against, and can
also regenerate the full-size tower taps that
``crates/vokra-models/tests/voxtral_tower_parity.rs`` consumes (``--full-taps``).

The reference is the upstream ``transformers.models.voxtral`` implementation
(``VoxtralEncoder`` + ``VoxtralMultiModalProjector``) plus the Llama text
decoder the checkpoint carries (``text_config.model_type = "llama"``), run in
**fp32 with eager attention** on the real ``mistralai/Voxtral-Mini-3B-2507``
BF16 shards (BF16 → fp32 widening is exact, so weight representation is
bit-identical to what the Vokra GGUF loader sees).

RSS discipline (16 GB machines): the full model in fp32 is ~18.7 GB and does
NOT fit, so the dump runs in two stages —

1. **tower stage**: audio tower + projector only, strict-loaded fp32
   (measured 6.84 GiB peak on M1); freed before stage 2;
2. **text stage**: the 30 decoder layers are held as **bf16** state dicts
   (~6.3 GiB) and widened to fp32 **one layer at a time** into meta-built
   ``LlamaDecoderLayer`` modules (``load_state_dict(assign=True)``), with the
   lm_head streamed in row chunks. Numerics are identical to a monolithic
   fp32 forward — same modules, same order, only residency differs — and a
   mandatory **self-check** proves it: the first 2 real layers are run both
   through this streaming orchestration and through an in-RAM upstream
   ``LlamaModel``, and the outputs must match **bitwise** (prompt pass and
   incremental cached step) or the dump aborts.

The prompt is the REAL transcription request layout (mistral_common tekken
tokenizer, ``TranscriptionRequest`` with ``language="en"`` — the same path
``VoxtralProcessor.apply_transcription_request`` drives): ``pre_audio`` ids +
one contiguous ``[AUDIO]`` (id 24) run replaced by the projector's soft-prefix
rows + ``post_audio`` ids, exactly the ``masked_scatter`` semantics of
``VoxtralForConditionalGeneration.forward``.

What is dumped to ``tests/parity/voxtral/`` (bounded — whisper fixture-size
precedent; full-size tensors would be ~7.7 MB each):

* ``input_pcm.f32``               – the raw 16 kHz mono f32 samples (NOT padded;
                                    both front-ends pad to 30 s internally);
* ``log_mel.f32.bin``             – upstream ``WhisperFeatureExtractor`` mel,
                                    first ``MEL_FRAMES`` frames ``[n_mels, F]``;
* ``audio_encoder_out.f32.bin``   – encoder final hidden, first ``ENC_POS``
                                    rows ``[ENC_POS, d_audio]``;
* ``soft_prefix.f32.bin``         – post-projector soft prefix, first
                                    ``PREFIX_ROWS`` rows (= the frame-stack of
                                    the first ``ENC_POS`` encoder positions);
* ``text_decoder_step_out.f32.bin`` – decoder logits at the last prompt
                                    position ``[vocab_size]`` (the first
                                    sampling step);
* ``asr_tokens.i64.bin``          – greedy transcription ids (EOS included);
* ``manifest.txt``                – shapes / prompt ids / greedy ids / text /
                                    provenance, ``key = value``.

Run (from the repo root; the venv needs torch / transformers / numpy /
safetensors / mistral_common — verified with torch 2.8.0, transformers 4.57.6,
mistral_common 1.8.5, numpy 2.0.2)::

    python tools/parity/dump_voxtral_reference.py \
        --checkpoint-dir ~/.cache/vokra-eval/weights/voxtral \
        --audio tests/fixtures/audio/jfk-30s.wav \
        [--full-taps ~/.cache/vokra-eval/out/p1-voxtral-asr/ref]

``--checkpoint-dir`` is required (the 9.4 GB checkpoint is never auto
-downloaded): an HF snapshot of ``mistralai/Voxtral-Mini-3B-2507`` with
``config.json`` + ``model-*.safetensors`` + ``model.safetensors.index.json`` +
``tekken.json`` + ``preprocessor_config.json``. FR-EX-08 throughout: wrong
audio format, missing files, non-contiguous audio-token runs and self-check
mismatches are hard errors, never silent substitutions.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import resource
import struct
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

MODEL_ID = "mistralai/Voxtral-Mini-3B-2507"
SIZE = "voxtral-mini-3b"
TORCH_SEED = 1234  # tie-break determinism only; inputs are real audio

# Bounded fixture slices (whisper precedent: MEL_FRAMES = 100, ENC_POS = 32).
MEL_FRAMES = 100
ENC_POS = 32
MAX_NEW = 64  # greedy cap — mirrors the Rust harness cap

DEFAULT_AUDIO = (
    Path(__file__).resolve().parents[2]
    / "tests"
    / "fixtures"
    / "audio"
    / "jfk-30s.wav"
)
DEFAULT_OUT = Path(__file__).resolve().parents[2] / "tests" / "parity" / "voxtral"

AUDIO_TOKEN_ID = 24  # VoxtralProcessor.__init__: self.audio_token_id = 24


def load_pcm(path: Path) -> np.ndarray:
    """Load mono 16 kHz PCM from a real WAV (PCM16 or IEEE_FLOAT32), raw length.

    Same strict RIFF walk as ``dump_whisper_reference.load_pcm`` (FR-EX-08: no
    silent resample / mixdown / format guess), except the clip is NOT padded —
    both front-ends (upstream ``WhisperFeatureExtractor`` and Vokra
    ``whisper::mel::log_mel``) pad to the 30 s window internally, and the
    committed fixture stays the honest raw sample count.
    """
    if not path.is_file():
        sys.exit(
            f"audio file not found: {path}\n"
            "  --audio must point at a real 16 kHz mono WAV (PCM16 or "
            "IEEE_FLOAT32)."
        )
    data = path.read_bytes()
    if len(data) < 12 or data[0:4] != b"RIFF" or data[8:12] != b"WAVE":
        sys.exit(f"{path}: not a RIFF/WAVE file")

    fmt = None
    payload = None
    pos = 12
    while pos + 8 <= len(data):
        cid = data[pos : pos + 4]
        (size,) = struct.unpack_from("<I", data, pos + 4)
        body_start = pos + 8
        body_end = body_start + size
        if body_end > len(data):
            sys.exit(f"{path}: chunk {cid!r} size {size} exceeds file length")
        body = data[body_start:body_end]
        if cid == b"fmt ":
            if size < 16:
                sys.exit(f"{path}: fmt chunk too small ({size} bytes)")
            audio_format, channels, sample_rate = struct.unpack_from("<HHI", body, 0)
            (bits,) = struct.unpack_from("<H", body, 14)
            fmt = (audio_format, channels, sample_rate, bits)
        elif cid == b"data":
            payload = body
        pos = body_end + (size & 1)

    if fmt is None:
        sys.exit(f"{path}: no fmt chunk")
    if payload is None:
        sys.exit(f"{path}: no data chunk")

    audio_format, channels, sample_rate, bits = fmt
    if channels != 1:
        sys.exit(f"{path}: expected mono, got {channels} channels (FR-EX-08)")
    if sample_rate != 16000:
        sys.exit(f"{path}: expected 16 kHz, got {sample_rate} Hz (FR-EX-08)")
    if audio_format == 3 and bits == 32:
        return np.frombuffer(payload, dtype="<f4").astype(np.float32)
    if audio_format == 1 and bits == 16:
        ints = np.frombuffer(payload, dtype="<i2").astype(np.float32)
        return (ints / 32768.0).astype(np.float32)
    sys.exit(
        f"{path}: unsupported PCM format (audio_format={audio_format}, "
        f"bits={bits}); use mono float32 or int16"
    )


def write_f32(path: Path, arr) -> None:
    a = np.asarray(arr, dtype="<f4").reshape(-1)
    path.write_bytes(a.tobytes())


def rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30  # macOS: bytes


def build_prompt(ckpt: Path, audio_path: Path):
    """Real transcription-request prompt ids via mistral_common (tekken).

    Returns ``(pre_audio, n_audio_tokens, post_audio, tokenizer)``. The
    ``[AUDIO]`` placeholder run must be contiguous (the segment replay both
    here and in the Rust harness depends on it) — a non-contiguous run is a
    hard error.
    """
    from mistral_common.audio import Audio
    from mistral_common.protocol.instruct.messages import RawAudio
    from mistral_common.protocol.transcription.request import TranscriptionRequest
    from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

    tok = MistralTokenizer.from_file(str(ckpt / "tekken.json"))
    audio = Audio.from_file(str(audio_path), strict=False)
    req = TranscriptionRequest(
        model="voxtral-mini-2507", audio=RawAudio.from_audio(audio), language="en"
    )
    ids = list(tok.instruct_tokenizer.encode_transcription(req).tokens)
    n_audio = ids.count(AUDIO_TOKEN_ID)
    if n_audio == 0:
        sys.exit("transcription prompt contains no [AUDIO] placeholder tokens")
    first = ids.index(AUDIO_TOKEN_ID)
    last = len(ids) - 1 - ids[::-1].index(AUDIO_TOKEN_ID)
    if any(t != AUDIO_TOKEN_ID for t in ids[first : last + 1]):
        sys.exit("non-contiguous [AUDIO] run in the transcription prompt (FR-EX-08)")
    return ids[:first], n_audio, ids[last + 1 :], tok


def run_tower(ckpt: Path, pcm: np.ndarray, full_taps: Path | None):
    """Stage 1: upstream mel + fp32 tower + projector (strict, eager).

    Returns ``(mel_2d, soft_prefix, enc_final, n_ctx, d_audio, frame_stack,
    d_text)`` as numpy fp32 arrays / ints. Optionally re-dumps the full-size
    taps for ``voxtral_tower_parity.rs`` into ``full_taps``.
    """
    from transformers import WhisperFeatureExtractor
    from transformers.models.voxtral.configuration_voxtral import VoxtralConfig
    from transformers.models.voxtral.modeling_voxtral import (
        VoxtralEncoder,
        VoxtralMultiModalProjector,
    )

    fe = WhisperFeatureExtractor.from_pretrained(ckpt)
    if not (fe.feature_size == 128 and fe.hop_length == 160 and fe.n_fft == 400):
        sys.exit(
            f"unexpected feature extractor (feature_size={fe.feature_size}, "
            f"hop={fe.hop_length}, n_fft={fe.n_fft}) — not the shipping mini"
        )
    feats = fe(pcm, sampling_rate=16000, return_tensors="pt")["input_features"]
    if tuple(feats.shape) != (1, 128, 3000):
        sys.exit(f"mel shape {tuple(feats.shape)} != (1, 128, 3000)")
    mel = feats.to(torch.float32)

    cfg = VoxtralConfig.from_pretrained(ckpt)
    enc_cfg = cfg.audio_config
    # Canonical math reference: eager attention.
    enc_cfg._attn_implementation = "eager"
    cfg._attn_implementation = "eager"

    tower_sd: dict[str, torch.Tensor] = {}
    proj_sd: dict[str, torch.Tensor] = {}
    index = json.loads((ckpt / "model.safetensors.index.json").read_text())
    for shard in sorted(set(index["weight_map"].values())):
        with safe_open(ckpt / shard, framework="pt") as f:
            for key in f.keys():
                if key.startswith("audio_tower."):
                    tower_sd[key[len("audio_tower.") :]] = f.get_tensor(key).float()
                elif key.startswith("multi_modal_projector."):
                    proj_sd[key[len("multi_modal_projector.") :]] = (
                        f.get_tensor(key).float()
                    )

    encoder = VoxtralEncoder(enc_cfg)
    encoder.load_state_dict(tower_sd, strict=True)
    encoder = encoder.float().eval()
    projector = VoxtralMultiModalProjector(cfg)
    projector.load_state_dict(proj_sd, strict=True)
    projector = projector.float().eval()
    del tower_sd, proj_sd
    gc.collect()

    captured: dict[str, torch.Tensor] = {}
    if full_taps is not None:
        # Full-size taps for voxtral_tower_parity.rs (same filenames the
        # cc-05/cc-07 harness established).
        def pre_hook(_mod, args, kwargs):
            captured["conv_stem_pos"] = args[0] if args else kwargs["hidden_states"]

        def out_hook(name):
            def h(_mod, _inp, out):
                captured[name] = out[0] if isinstance(out, tuple) else out

            return h

        encoder.layers[0].register_forward_pre_hook(pre_hook, with_kwargs=True)
        encoder.layers[0].register_forward_hook(out_hook("after_layer_0"))
        encoder.layers[15].register_forward_hook(out_hook("after_layer_15"))

    out = encoder(mel)
    final = out.last_hidden_state
    n_ctx, d_audio = final.shape[1], final.shape[2]
    if (n_ctx, d_audio) != (1500, 1280):
        sys.exit(f"encoder output {final.shape} != [1, 1500, 1280]")

    inter = cfg.audio_config.intermediate_size
    frame_stack = inter // d_audio  # 5120 / 1280 = 4
    stacked = final.reshape(-1, inter)  # get_audio_features semantics
    soft_prefix = projector(stacked)
    d_text = cfg.text_config.hidden_size
    if soft_prefix.shape != (n_ctx // frame_stack, d_text):
        sys.exit(f"soft prefix shape {tuple(soft_prefix.shape)} unexpected")

    if full_taps is not None:
        full_taps.mkdir(parents=True, exist_ok=True)
        write_f32(full_taps / "jfk_pcm.f32.bin", pcm)
        write_f32(full_taps / "input_mel.f32.bin", mel[0].numpy())
        write_f32(full_taps / "conv_stem_pos.f32.bin", captured["conv_stem_pos"][0].numpy())
        write_f32(full_taps / "after_layer_0.f32.bin", captured["after_layer_0"][0].numpy())
        write_f32(full_taps / "after_layer_15.f32.bin", captured["after_layer_15"][0].numpy())
        write_f32(full_taps / "encoder_final.f32.bin", final[0].numpy())
        write_f32(full_taps / "stacked.f32.bin", stacked.numpy())
        write_f32(full_taps / "soft_prefix.f32.bin", soft_prefix.numpy())
        print(f"full-size tower taps re-dumped to {full_taps}")

    mel_np = mel[0].numpy().copy()
    final_np = final[0].numpy().copy()
    prefix_np = soft_prefix.numpy().copy()
    del encoder, projector, out, final, soft_prefix, stacked, captured, mel, feats
    gc.collect()
    print(f"tower stage done (RSS peak so far {rss_gib():.2f} GiB)")
    return mel_np, prefix_np, final_np, n_ctx, d_audio, frame_stack, d_text


class StreamedTextDecoder:
    """fp32 Llama forward with layer-at-a-time weight residency.

    Weights are held as **bf16** state dicts (exact upstream bits) and widened
    to fp32 per layer per pass into meta-built ``LlamaDecoderLayer`` modules
    (``load_state_dict(assign=True)``). The KV cache is a transformers
    ``DynamicCache``, the causal mask comes from transformers'
    ``create_causal_mask`` and rotary embeddings from ``LlamaRotaryEmbedding``
    — i.e. every numerical component is the upstream one; only weight
    residency is orchestrated here, and ``self_check()`` proves the
    orchestration bitwise against an in-RAM upstream ``LlamaModel``.
    """

    P = "language_model."  # checkpoint key prefix

    def __init__(self, ckpt: Path, text_cfg) -> None:
        from transformers.models.llama.modeling_llama import (
            LlamaDecoderLayer,
            LlamaRMSNorm,
            LlamaRotaryEmbedding,
        )

        self.cfg = text_cfg
        self.cfg._attn_implementation = "eager"
        self.n_layer = text_cfg.num_hidden_layers
        self.d = text_cfg.hidden_size
        self.vocab = text_cfg.vocab_size

        index = json.loads((ckpt / "model.safetensors.index.json").read_text())
        self.weight_map = index["weight_map"]
        self.ckpt = ckpt
        self._shards: dict[str, object] = {}

        # bf16 state dicts per layer (exact checkpoint bits, ~6.3 GiB total).
        self.layer_sd: list[dict[str, torch.Tensor]] = []
        for i in range(self.n_layer):
            prefix = f"{self.P}model.layers.{i}."
            sd = {
                k[len(prefix) :]: self._raw(k)
                for k in self.weight_map
                if k.startswith(prefix)
            }
            if not sd:
                sys.exit(f"no tensors found for text layer {i} ({prefix}*)")
            self.layer_sd.append(sd)

        norm_w = self._raw(f"{self.P}model.norm.weight").float()
        self.norm = LlamaRMSNorm(self.d, eps=text_cfg.rms_norm_eps)
        self.norm.load_state_dict({"weight": norm_w})
        self.norm.eval()

        # lm_head held bf16 ([vocab, d], untied on the mini), widened in chunks.
        self.lm_head_bf16 = self._raw(f"{self.P}lm_head.weight")
        if tuple(self.lm_head_bf16.shape) != (self.vocab, self.d):
            sys.exit(f"lm_head shape {tuple(self.lm_head_bf16.shape)} unexpected")

        self.rotary = LlamaRotaryEmbedding(config=text_cfg)

        # Meta-built layer modules: zero weight residency until a pass assigns
        # the widened fp32 tensors (replaced again on the next pass).
        self.layers = []
        for i in range(self.n_layer):
            with torch.device("meta"):
                layer = LlamaDecoderLayer(text_cfg, i)
            layer.eval()
            self.layers.append(layer)

    def _raw(self, key: str) -> torch.Tensor:
        shard = self.weight_map.get(key)
        if shard is None:
            sys.exit(f"checkpoint index has no tensor {key!r}")
        if shard not in self._shards:
            self._shards[shard] = safe_open(self.ckpt / shard, framework="pt")
        return self._shards[shard].get_tensor(key)

    def embed_rows(self, ids: list[int]) -> torch.Tensor:
        """Gather embed_tokens rows (bf16 → fp32 exact) without materialising
        the full [131072, 3072] matrix."""
        key = f"{self.P}model.embed_tokens.weight"
        shard = self.weight_map.get(key)
        if shard is None:
            sys.exit(f"checkpoint index has no tensor {key!r}")
        if shard not in self._shards:
            self._shards[shard] = safe_open(self.ckpt / shard, framework="pt")
        sl = self._shards[shard].get_slice(key)
        return torch.stack([sl[i] for i in ids]).float()

    def _mask_and_rope(self, embeds: torch.Tensor, cache, pos0: int):
        from transformers.masking_utils import create_causal_mask

        t = embeds.shape[1]
        cache_position = torch.arange(pos0, pos0 + t)
        position_ids = cache_position.unsqueeze(0)
        mask = create_causal_mask(
            config=self.cfg,
            input_embeds=embeds,
            attention_mask=None,
            cache_position=cache_position,
            past_key_values=cache,
            position_ids=position_ids,
        )
        cos_sin = self.rotary(embeds, position_ids)
        return mask, position_ids, cache_position, cos_sin

    def forward_layers(
        self, embeds: torch.Tensor, cache, pos0: int, n_layer: int | None = None
    ) -> torch.Tensor:
        """Streams ``embeds`` [1, t, d] through the (first ``n_layer``) decoder
        layers with the cache appending at ``pos0``; returns POST-NORM hidden."""
        mask, position_ids, cache_position, cos_sin = self._mask_and_rope(
            embeds, cache, pos0
        )
        hidden = embeds
        for i in range(n_layer if n_layer is not None else self.n_layer):
            layer = self.layers[i]
            fp32_sd = {k: v.float() for k, v in self.layer_sd[i].items()}
            layer.load_state_dict(fp32_sd, strict=True, assign=True)
            hidden = layer(
                hidden,
                attention_mask=mask,
                position_ids=position_ids,
                past_key_values=cache,
                use_cache=True,
                cache_position=cache_position,
                position_embeddings=cos_sin,
            )
            del fp32_sd  # fp32 stays referenced by the module until next assign
        return self.norm(hidden)

    def logits_last(self, post_norm: torch.Tensor, chunk: int = 16384) -> np.ndarray:
        """lm_head on the last position, streamed in row chunks (bf16 → fp32)."""
        h = post_norm[0, -1, :]
        out = np.empty(self.vocab, dtype=np.float32)
        for c0 in range(0, self.vocab, chunk):
            c1 = min(c0 + chunk, self.vocab)
            w = self.lm_head_bf16[c0:c1].float()
            out[c0:c1] = (w @ h).numpy()
        return out

    def self_check(self, embeds: torch.Tensor) -> None:
        """Prove the streaming orchestration bitwise against upstream
        ``LlamaModel`` on the REAL first two layers (prompt pass + one cached
        incremental step). Any non-zero delta aborts the dump (FR-EX-08)."""
        import copy

        from transformers.cache_utils import DynamicCache
        from transformers.models.llama.modeling_llama import LlamaModel

        cfg2 = copy.deepcopy(self.cfg)
        cfg2.num_hidden_layers = 2
        cfg2.vocab_size = 8  # embed_tokens never indexed (inputs_embeds path)
        cfg2._attn_implementation = "eager"
        ref = LlamaModel(cfg2)
        ref_sd = {}
        for i in range(2):
            for k, v in self.layer_sd[i].items():
                ref_sd[f"layers.{i}.{k}"] = v.float()
        ref_sd["norm.weight"] = self.norm.weight.data
        load = ref.load_state_dict(ref_sd, strict=False)
        if load.unexpected_keys or set(load.missing_keys) != {"embed_tokens.weight"}:
            sys.exit(
                f"self-check LlamaModel load mismatch: missing={load.missing_keys} "
                f"unexpected={load.unexpected_keys}"
            )
        ref.eval()

        t = embeds.shape[1]
        with torch.no_grad():
            ref_cache = DynamicCache()
            h1 = ref(
                inputs_embeds=embeds[:, : t - 1],
                past_key_values=ref_cache,
                use_cache=True,
            ).last_hidden_state
            h2 = ref(
                inputs_embeds=embeds[:, t - 1 :],
                past_key_values=ref_cache,
                use_cache=True,
            ).last_hidden_state

            my_cache = DynamicCache()
            m1 = self.forward_layers(embeds[:, : t - 1], my_cache, 0, n_layer=2)
            m2 = self.forward_layers(embeds[:, t - 1 :], my_cache, t - 1, n_layer=2)

        d1 = (m1 - h1).abs().max().item()
        d2 = (m2 - h2).abs().max().item()
        if d1 != 0.0 or d2 != 0.0:
            sys.exit(
                f"self-check FAILED: streaming orchestration is not bitwise vs "
                f"upstream LlamaModel (prompt max |Δ| = {d1:.3e}, step max |Δ| = "
                f"{d2:.3e}) — refusing to dump a reference from a divergent "
                "orchestration"
            )
        print("self-check OK: streaming == upstream LlamaModel bitwise (2 layers)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Dump upstream Voxtral reference tensors for parity. Regenerates "
            "the committed fixtures under tests/parity/voxtral/."
        )
    )
    parser.add_argument(
        "--checkpoint-dir",
        required=True,
        help=(
            "LOCAL HF snapshot of mistralai/Voxtral-Mini-3B-2507 (config.json "
            "+ model-*.safetensors + index + tekken.json + "
            "preprocessor_config.json). Required — the 9.4 GB checkpoint is "
            "never auto-downloaded."
        ),
    )
    parser.add_argument(
        "--audio",
        default=str(DEFAULT_AUDIO),
        help=(
            "Real 16 kHz mono WAV (PCM16 or IEEE_FLOAT32). Default: "
            "%(default)s. FR-EX-08: unsupported format is a hard error."
        ),
    )
    parser.add_argument(
        "--full-taps",
        default=None,
        help=(
            "Optional directory to ALSO re-dump the full-size tower taps for "
            "crates/vokra-models/tests/voxtral_tower_parity.rs "
            "(input_mel/conv_stem_pos/after_layer_{0,15}/encoder_final/"
            "stacked/soft_prefix + jfk_pcm; ~52 MB — never committed)."
        ),
    )
    parser.add_argument(
        "out_dir",
        nargs="?",
        default=None,
        help="Output directory. Defaults to tests/parity/voxtral/.",
    )
    args = parser.parse_args()

    ckpt = Path(args.checkpoint_dir).expanduser()
    for required in (
        "config.json",
        "model.safetensors.index.json",
        "tekken.json",
        "preprocessor_config.json",
        "generation_config.json",
    ):
        if not (ckpt / required).is_file():
            sys.exit(f"--checkpoint-dir {ckpt} is missing {required} (FR-EX-08)")
    out_dir = Path(args.out_dir) if args.out_dir is not None else DEFAULT_OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    full_taps = Path(args.full_taps).expanduser() if args.full_taps else None

    audio_path = Path(args.audio).resolve()
    pcm = load_pcm(audio_path)
    pcm_sha256 = hashlib.sha256(audio_path.read_bytes()).hexdigest()
    repo_root = Path(__file__).resolve().parents[2]
    try:
        pcm_source = audio_path.relative_to(repo_root).as_posix()
    except ValueError:
        pcm_source = str(audio_path)

    gen_cfg = json.loads((ckpt / "generation_config.json").read_text())
    bos = int(gen_cfg["bos_token_id"])
    eos = int(gen_cfg["eos_token_id"])

    torch.manual_seed(TORCH_SEED)
    torch.set_grad_enabled(False)

    # ---- prompt (real transcription-request layout) ------------------------
    pre_audio, n_audio_tokens, post_audio, tok = build_prompt(ckpt, audio_path)
    print(
        f"prompt: pre={pre_audio} audio×{n_audio_tokens} post={post_audio} "
        f"(bos={bos} eos={eos})"
    )

    # ---- stage 1: mel + tower + projector (fp32, strict, eager) ------------
    mel, soft_prefix, enc_final, n_ctx, d_audio, frame_stack, d_text = run_tower(
        ckpt, pcm, full_taps
    )
    n_mels, n_frames = mel.shape
    if soft_prefix.shape[0] != n_audio_tokens:
        sys.exit(
            f"soft prefix rows {soft_prefix.shape[0]} != prompt audio tokens "
            f"{n_audio_tokens} — prompt and tower disagree (FR-EX-08)"
        )
    if ENC_POS % frame_stack != 0:
        sys.exit(f"ENC_POS {ENC_POS} not a multiple of frame_stack {frame_stack}")
    prefix_rows = ENC_POS // frame_stack

    write_f32(out_dir / "input_pcm.f32", pcm)
    write_f32(out_dir / "log_mel.f32.bin", mel[:, :MEL_FRAMES])
    write_f32(out_dir / "audio_encoder_out.f32.bin", enc_final[:ENC_POS, :])
    write_f32(out_dir / "soft_prefix.f32.bin", soft_prefix[:prefix_rows, :])

    # ---- stage 2: streamed fp32 text decoder -------------------------------
    from transformers.cache_utils import DynamicCache
    from transformers.models.voxtral.configuration_voxtral import VoxtralConfig

    text_cfg = VoxtralConfig.from_pretrained(ckpt).text_config
    dec = StreamedTextDecoder(ckpt, text_cfg)
    vocab = dec.vocab

    pre_embed = dec.embed_rows(pre_audio)
    post_embed = dec.embed_rows(post_audio)
    prompt_embeds = torch.cat(
        [pre_embed, torch.from_numpy(soft_prefix), post_embed], dim=0
    ).unsqueeze(0)
    t_prompt = prompt_embeds.shape[1]
    print(f"prompt embeds: [{t_prompt}, {d_text}] (RSS {rss_gib():.2f} GiB)")

    # Mandatory orchestration proof before any reference is dumped.
    dec.self_check(prompt_embeds)

    cache = DynamicCache()
    post_norm = dec.forward_layers(prompt_embeds, cache, 0)
    logits = dec.logits_last(post_norm)
    write_f32(out_dir / "text_decoder_step_out.f32.bin", logits)
    print(
        f"prompt pass done: logits[{vocab}], argmax={int(logits.argmax())} "
        f"(RSS {rss_gib():.2f} GiB)"
    )

    greedy: list[int] = []
    next_id = int(logits.argmax())
    greedy.append(next_id)
    pos = t_prompt
    while next_id != eos and len(greedy) < MAX_NEW:
        step_embed = dec.embed_rows([next_id]).unsqueeze(0)
        post_norm = dec.forward_layers(step_embed, cache, pos)
        step_logits = dec.logits_last(post_norm)
        next_id = int(step_logits.argmax())
        greedy.append(next_id)
        pos += 1
        if len(greedy) % 8 == 0:
            print(f"  greedy {len(greedy)} tokens… (RSS {rss_gib():.2f} GiB)")

    greedy_text = tok.decode([t for t in greedy if t != eos])
    (out_dir / "asr_tokens.i64.bin").write_bytes(
        np.asarray(greedy, dtype="<i8").tobytes()
    )
    print(f"greedy ({len(greedy)} ids, eos={'yes' if greedy[-1] == eos else 'NO'}):")
    print(f"  {greedy}")
    print(f"  text={greedy_text!r}")

    manifest = {
        "model": MODEL_ID,
        "size": SIZE,
        "torch_seed": TORCH_SEED,
        "pcm_source": pcm_source,
        "pcm_sha256": pcm_sha256,
        "n_samples": int(pcm.size),
        "n_mels": int(n_mels),
        "mel_frames": MEL_FRAMES,
        "n_frames": int(n_frames),
        "enc_pos": ENC_POS,
        "n_ctx": int(n_ctx),
        "d_audio": int(d_audio),
        "frame_stack": int(frame_stack),
        "prefix_rows": prefix_rows,
        "n_audio_tokens": int(n_audio_tokens),
        "d_text": int(d_text),
        "vocab_size": int(vocab),
        "audio_token_id": AUDIO_TOKEN_ID,
        "pre_audio": pre_audio,
        "post_audio": post_audio,
        "bos": bos,
        "eos": eos,
        "max_new": MAX_NEW,
        "greedy_tokens": greedy,
        "greedy_text": greedy_text,
    }
    with (out_dir / "manifest.txt").open("w", encoding="utf-8") as f:
        f.write("# Voxtral parity manifest (M3-10-T19/T20). Generated by\n")
        f.write("# tools/parity/dump_voxtral_reference.py. `key = value`;\n")
        f.write("# list values are space-separated. `pcm_source` /\n")
        f.write("# `pcm_sha256` pin the exact WAV that produced this fixture.\n")
        for k, v in manifest.items():
            if isinstance(v, list):
                f.write(f"{k} = {' '.join(str(x) for x in v)}\n")
            else:
                f.write(f"{k} = {v}\n")

    print(f"wrote fixtures to {out_dir} (RSS peak {rss_gib():.2f} GiB)")


if __name__ == "__main__":
    main()
