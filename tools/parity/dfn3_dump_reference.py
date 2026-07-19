#!/usr/bin/env python3
"""Dump per-stage DeepFilterNet3 reference taps for the vokra `denoise` parity
run (M4-20 T17).

Offline reference generator (numerical-parity discipline: reference =
the REAL upstream ``deepfilternet`` package running the REAL released
checkpoint — never a re-implementation of ourselves). For a given noisy wav
it replicates ``df.enhance.enhance(pad=True)`` stage by stage, dumping every
intermediate the vokra Rust port must reproduce:

* ``spec.f32``       [T, 481, 2]  — raw libDF analysis spectrum (stage 0)
* ``feat_erb.f32``   [T, 32]      — erb_norm'd dB ERB features (model input)
* ``feat_spec.f32``  [T, 96, 2]   — unit_norm'd complex spec features
* ``e0.f32``         [64, T, 32]  — enc.erb_conv0 output      (post-BN+ReLU)
* ``e1.f32``         [64, T, 16]  — enc.erb_conv1 output
* ``e2.f32``         [64, T, 8]   — enc.erb_conv2 output
* ``e3.f32``         [64, T, 8]   — enc.erb_conv3 output
* ``c0.f32``         [64, T, 96]  — enc.df_conv0 output
* ``cemb.f32``       [T, 512]     — enc.df_fc_emb(flatten(df_conv1)) output
* ``emb_in.f32``     [T, 512]     — e3-flatten + cemb (emb_gru input)
* ``emb.f32``        [T, 512]     — enc.emb_gru output
* ``lsnr.f32``       [T]          — local SNR estimate (dB)
* ``m.f32``          [T, 32]      — erb_dec mask output (sigmoid)
* ``df_gru_out.f32`` [T, 256]     — df_dec GRU output + df_skip (pre df_out)
* ``coefs.f32``      [T, 96, 10]  — df_dec coefficient output (order-major
                                    (re, im) pairs, pre DfOutputReshapeMF)
* ``spec_e.f32``     [T, 481, 2]  — final enhanced spectrum fed to synthesis
* ``enhanced.f32``   [n]          — final enhanced samples (delay-trimmed)

T = (len(noisy) + n_fft) / hop frames (enhance() right-pads with n_fft
zeros before analysis; the final wav trim removes the n_fft − hop = 480
sample STFT/ISTFT delay, df/enhance.py L229-248).

Self-check: the stage-by-stage replication is asserted equal to a plain
``model(spec, feat_erb, feat_spec)`` call (max |Δ| must be 0.0) so the taps
provably describe the real forward, and the synthesized wav is compared to
``df.enhance.enhance`` output.

# Usage

::

    ~/.cache/vokra-eval/venv-dfn3/bin/python tools/parity/dfn3_dump_reference.py \\
        --model-dir ~/.cache/vokra-eval/weights/dfn3/DeepFilterNet3 \\
        --noisy ~/.cache/vokra-eval/out/dfn3-real/noisy_48k.wav \\
        --out ~/.cache/vokra-eval/out/dfn3-real/taps
"""

import argparse
import json
import os
import sys

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

from df.enhance import df_features, enhance, init_df


def dump(path: str, t: torch.Tensor) -> None:
    a = t.detach().cpu().numpy().astype("<f4")
    np.ascontiguousarray(a).tofile(path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--noisy", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    model, df_state, _ = init_df(args.model_dir, post_filter=False, log_level="WARNING", log_file=None)
    model = model.to("cpu").eval()

    noisy, sr = sf.read(args.noisy, dtype="float32", always_2d=False)
    assert sr == 48000, sr
    audio = torch.from_numpy(noisy)[None, :]

    n_fft, hop = df_state.fft_size(), df_state.hop_size()
    padded = F.pad(audio, (0, n_fft))
    nb_df = model.nb_df

    with torch.no_grad():
        spec, feat_erb, feat_spec = df_features(padded, df_state, nb_df, device="cpu")
        # Reference forward (whole model in one call).
        spec_e_ref, m_ref, lsnr_ref, _ = model(spec.clone(), feat_erb, feat_spec)

        # Stage-by-stage replication of DfNet.forward (deepfilternet3.py
        # L389-456) with taps.
        fs = feat_spec.squeeze(1).permute(0, 3, 1, 2)  # [B, 2, T, F']
        fe_p = model.pad_feat(feat_erb)
        fs_p = model.pad_feat(fs)
        enc = model.enc
        e0 = enc.erb_conv0(fe_p)
        e1 = enc.erb_conv1(e0)
        e2 = enc.erb_conv2(e1)
        e3 = enc.erb_conv3(e2)
        c0 = enc.df_conv0(fs_p)
        c1 = enc.df_conv1(c0)
        cemb = enc.df_fc_emb(c1.permute(0, 2, 3, 1).flatten(2))
        emb_in = enc.combine(e3.permute(0, 2, 3, 1).flatten(2), cemb)
        emb, _ = enc.emb_gru(emb_in)
        lsnr = enc.lsnr_fc(emb) * enc.lsnr_scale + enc.lsnr_offset

        m = model.erb_dec(emb, e3, e2, e1, e0)
        spec_m = model.mask(spec, m)

        dd = model.df_dec
        c, _ = dd.df_gru(emb)
        if dd.df_skip is not None:
            c = c + dd.df_skip(emb)
        c0p = dd.df_convp(c0).permute(0, 2, 3, 1)
        coefs = dd.df_out(c).view(c.shape[0], c.shape[1], dd.df_bins, dd.df_out_ch) + c0p
        df_coefs = model.df_out_transform(coefs)
        spec_e = model.df_op(spec.clone(), df_coefs)
        spec_e[..., model.nb_df:, :] = spec_m[..., model.nb_df:, :]

        # Self-check: taps describe the real forward exactly.
        d_spec = (spec_e - spec_e_ref).abs().max().item()
        d_m = (m - m_ref).abs().max().item()
        d_lsnr = (lsnr - lsnr_ref).abs().max().item()
        assert d_spec == 0.0 and d_m == 0.0 and d_lsnr == 0.0, (d_spec, d_m, d_lsnr)

        # Synthesis + delay trim (enhance() pad=True path). NB: libDF's
        # synthesis DESTROYS its input buffer (realfft c2r uses the input as
        # scratch) and `.numpy()` on a contiguous view ALIASES the tensor
        # storage — pass a copy or the spec_e dump below would silently be
        # post-synthesis scratch garbage (found the hard way: a 0.64 max-Δ
        # "mismatch" that the final wav contradicted).
        enh_c = torch.view_as_complex(spec_e.squeeze(1).contiguous())
        wav = torch.as_tensor(df_state.synthesis(enh_c.numpy().copy()))
        d = n_fft - hop
        wav = wav[:, d: audio.shape[-1] + d]
        wav_ref = enhance(model, df_state, audio.clone(), pad=True)
        d_wav = (wav - wav_ref).abs().max().item()

    dump(f"{args.out}/spec.f32", spec.squeeze(0).squeeze(0))
    dump(f"{args.out}/feat_erb.f32", feat_erb.squeeze(0).squeeze(0))
    dump(f"{args.out}/feat_spec.f32", feat_spec.squeeze(0).squeeze(0))
    dump(f"{args.out}/e0.f32", e0.squeeze(0))
    dump(f"{args.out}/e1.f32", e1.squeeze(0))
    dump(f"{args.out}/e2.f32", e2.squeeze(0))
    dump(f"{args.out}/e3.f32", e3.squeeze(0))
    dump(f"{args.out}/c0.f32", c0.squeeze(0))
    dump(f"{args.out}/cemb.f32", cemb.squeeze(0))
    dump(f"{args.out}/emb_in.f32", emb_in.squeeze(0))
    dump(f"{args.out}/emb.f32", emb.squeeze(0))
    dump(f"{args.out}/lsnr.f32", lsnr.squeeze(0).squeeze(-1))
    dump(f"{args.out}/m.f32", m.squeeze(0).squeeze(0))
    dump(f"{args.out}/df_gru_out.f32", c.squeeze(0))
    dump(f"{args.out}/coefs.f32", coefs.squeeze(0))
    dump(f"{args.out}/spec_e.f32", spec_e.squeeze(0).squeeze(0))
    dump(f"{args.out}/enhanced.f32", wav.squeeze(0))

    meta = {
        "frames": int(feat_erb.shape[2]),
        "n_samples": int(audio.shape[-1]),
        "sample_rate": sr,
        "self_check_max_delta": {"spec_e": d_spec, "m": d_m, "lsnr": d_lsnr, "wav_vs_enhance": d_wav},
    }
    with open(f"{args.out}/meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(meta))
    return 0


if __name__ == "__main__":
    sys.exit(main())
