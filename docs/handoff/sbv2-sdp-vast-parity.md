# SBV2 SDP body — VAST real-parity gate

This runbook turns the 2026-08-18 SBV2 SDP body measurement into a
repeatable, fail-closed VAST job. The canonical worker is
`scripts/publish/vast-ai/run-sbv2-sdp-parity.sh`.

The worker is intentionally not a local or GitHub-hosted CI job. It downloads
three real checkpoints, converts about 2.7 GB of GGUFs, and builds/tests
`vokra-models`; current policy routes all of that work to VAST so the M1 Mac
never loads model weights or runs workspace/model Cargo.

## Contract

An actual run refuses to start unless all of these are true:

- the host is Linux and `scripts/publish/vast-ai/provision.sh` has exported
  `VOKRA_PUBLISH_ON_VAST=1`;
- `/proc/meminfo` reports the VAST 64 GB memory class and at least 50 GB of
  disk is free;
- Python is available through the committed `tools/parity/uv.lock` and every
  Python command is launched by `uv`;
- none of the seven generated GGUF/reference targets already exists in the
  checkout. Use a fresh disposable instance instead of silently reusing or
  overwriting an old measurement.

At runtime the worker resolves the current immutable Hugging Face commit for
each fixed public repo, pins all downloads to those three revisions, converts
the three GGUFs, and compares each output with its committed SHA-256 sidecar.
Only after those checks pass does it generate the independent MIT VITS
reference and run:

```bash
cargo test -p vokra-models --test sbv2_sdp_torch_parity -- --ignored --nocapture
```

The environment record is emitted before the numerical result and includes
the repo commit, branch, CPU model and ISA flags, thread count, memory, GPU,
Python, torch CPU capability, uv, rustc, and Cargo versions. The final
`logs/summary.txt` also records all three upstream revisions and hashes of all
generated artifacts.

## Lifecycle

1. Commit and push the branch that will be measured.
2. Rent a fresh VAST instance using the constraints in
   `docs/handoff/vast-ai-large-model-publish.md` §2.2: RAM at least 64 GB,
   disk at least 200 GB, and the cheapest suitable GPU on the CUDA 12.4 /
   Ubuntu 22.04 image.
3. Clone the exact branch, run the tracked provisioner, and load its marker:

```bash
git clone --branch <branch> https://github.com/ayutaz/vokra.git ~/vokra
cd ~/vokra
bash scripts/publish/vast-ai/provision.sh --branch <branch>
source ~/.bashrc
```

These checkpoints are public, so this parity-only job needs no HF token. Do
not put a token in argv or a tracked file.

4. Run the worker:

```bash
cd ~/vokra
bash scripts/publish/vast-ai/run-sbv2-sdp-parity.sh
```

For a checkpoint-free local contract check, and only for that check:

```bash
bash scripts/publish/vast-ai/run-sbv2-sdp-parity.sh --self-test
```

5. Copy `logs/summary.txt`, `logs/environment.txt`, `logs/parity.log`, and
   `logs/source-revisions.txt` out of the instance. Do not copy the GGUFs,
   safetensors, or `sdp_body_*.f32.bin` files into the source tree.
6. Destroy the instance even after failure, then verify that the account has
   no running instance left.

## Numerical decision rule

The first manual VAST run on 2026-08-18 measured
`max |Δ| = 9.536743164e-6` at channel 96 / time 31 against the candidate
`1e-5` bound. One host and one fixture are not enough evidence to loosen or
tighten the tolerance. Preserve the explicit `#[ignore]` posture while the
real artifacts remain uncommitted, collect repeated environment-qualified
measurements with this worker, and investigate any miss before changing the
bound.

## Recorded automated run — 2026-08-18

The fail-closed worker was executed end to end at commit
`cdfb3e21328c2c2813dc2c26c5060ca992b0259d` on a VAST Xeon E5-2699 v4
host (88 visible threads, AVX2, 1,056,682,100 KiB RAM), with Rust 1.97.1,
Python 3.12.14, and torch 2.13.0+cu130. It pinned:

- SBV2 JP-Extra: `a731761009f3c96d104487be6ad332bf1bb5a3a5`;
- DeBERTa v2 JA: `547b0e8b044fba3f9b84d0ab9f990440bd130c8b`;
- DeBERTa v3 EN: `64a8c8eab3e352a784c658aef62be1662607476f`.

All three converted GGUFs matched their committed sidecars. The independent
input fixture hashes were `034b3fd65a2757ee5f834ac25c88b71bfefc6f5a0f56a7eb7b3709f713ccf5a1`
(hidden) and `9cba13f42455d3fab3e6fe1b9c548273879de6808f3d89f8587dfe66f359fcc8`
(speaker); the captured body hash was
`3a05ab141972f202b9e8f5c7fd1807c0c913cf45c8f2e95f2a886dd6c8080094`.
The Rust gate passed in 79.90 seconds with
`max |Δ| = 8.583068848e-6` at channel 118 / time 48. This differs from the
first manual environment while remaining below `1e-5`, so no tolerance or
`#[ignore]` posture was changed. The text logs were collected, the disposable
instance was destroyed, and zero running VAST instances were verified.
