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
