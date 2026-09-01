# Qwen3-ASR independent real-checkpoint parity

This directory stages the VAST-only oracle for the exact released
`Qwen/Qwen3-ASR-0.6B` and `Qwen/Qwen3-ASR-1.7B` revisions accepted by Vokra.
It imports the official Apache-2.0 `qwen-asr==0.0.6` package; there is no
second, locally reimplemented Qwen model in the dumper.

The official calls used as reference are:

- `Qwen3ASRModel._build_text_prompt` and its official processor for prompt and
  16 kHz log-mel preparation;
- `model.thinker.get_audio_features` for the final projected audio rows;
- `model.generate` for the greedy token sequence;
- the official tokenizer and `parse_asr_output` for raw/final text.

The environment is locked by this directory's `uv.lock`. The model snapshot is
downloaded by exact 40-hex revision, then all source files and generated
artifacts are hashed into the output. Missing imports, revision/shape drift,
sidecar drift, a non-local snapshot, non-FP32 execution, or a non-CPU official
reference aborts loudly.

Before synchronization or any snapshot/download/build, the VAST worker runs
the dependency-free `preflight_gate.py` against the exact `uv.lock` and
`pyproject.toml` bytes. Its tracked manifest binds canonical version/source/
marker/dependency rows, both fixed model revisions, the fixed reference-audio
hash, and every version-keyed license/native/bundled-code review row. The
production manifest intentionally remains `PENDING_REVIEW` with null signer
and digest; the worker therefore exits 2 before `uv sync` until an authorized
human review records an approval digest equal to the complete scope digest.
The gate also rejects `UNRESOLVED` rows even if an approval-shaped value is
later supplied.

The factual-audit route is a separately authorized, named VAST/Linux setup
step. It performs only the exact frozen environment sync (no model download,
reference execution, or Cargo):

```sh
VOKRA_PUBLISH_ON_VAST=1 \
  uv sync --project tools/parity/qwen3_asr --frozen --python 3.12
```

After that setup step, run the model-free closure audit before any checkpoint
acquisition:

```sh
scripts/publish/vast-ai/audit-qwen3-asr-dependencies.sh \
  --output /root/scratchpad/qwen3-asr-dependency-audit.json
```

The audit consumes only the committed project/lock and the synchronized
Python environment. It records all exact distribution metadata, publisher
license/notice file hashes, native artifact hashes and ELF `NEEDED` entries.
For pinned distributions whose wheels omit publisher files, it may fetch only
an exact locked `files.pythonhosted.org` release sdist when that lock row
provides one (the lock artifact must contain exactly `url`, `hash`, `size`, and
non-empty `upload-time`), then inspect bounded LICENSE/COPYING/NOTICE/COPYRIGHT
members in memory; archive bytes are never extracted or executed. A missing
locked sdist (currently the `dynet38` row) remains a structured factual blocker; there is
no README, alternate release, or wheel fallback. The two fixed model
revisions permit only their exact HF `LICENSE` paths. No weights, model
imports, or Cargo are part of this audit, and `uv sync` is performed only by
the separately authorized setup step above.
The VAST host must provide `readelf`; the wrapper refuses to run without it.
Any sdist or LICENSE redirect to a non-allowlisted path, archive traversal or
link, size/hash mismatch, or non-license model response is rejected before
unbounded bytes are accepted.
The expected successful output is:
`qwen3-asr dependency audit: PASS (...)`. A missing license, native inspection
failure, closure drift, or any non-license model response exits 2.

The same audit is invoked by `run-qwen3-asr-validation.sh` immediately after
its frozen sync and before the Vokra build, snapshot download, reference
dumper, or Cargo parity test. That production path becomes reachable only
after the audit facts have populated the review rows and an owner has approved
the resulting scope. The production manifest remains
`PENDING_REVIEW`; no audit result is an operator approval.

No numerical fixture is committed before an actual run. The Rust consumer in
`crates/vokra-models/tests/qwen3_asr_real.rs` is environment-gated and uses the
repository FP32 bound `atol=0.01` for projected audio. Greedy token ids,
language, and text must match exactly. Metal uses the same CPU model run as its
oracle and also requires exact greedy ids; it never falls back to CPU.

Run only through the VAST worker after provisioning:

```sh
scripts/publish/vast-ai/run-qwen3-asr-validation.sh \
  --variant all --approval-evidence /root/scratchpad/qwen3-asr-owner-approval.json
```

The worker uses the committed two-second mono 16 kHz JFK-derived clip at
`tests/parity/utmos/ref-clip.wav`, performs no upload, and leaves only the small
reference/evidence directory to pull before the VAST instance is destroyed.
Do not pull the source snapshot or GGUF back to the maintainer Mac.

After both CPU runs pass, transfer the two GGUFs and their reference
directories directly from VAST to a disposable Apple Silicon host with at
least 32 GB RAM. The Apple worker refuses the maintainer machine class unless
all remote-host gates are satisfied:

```sh
VOKRA_REMOTE_APPLE_SILICON=1 \
scripts/verify/apple-silicon-qwen3-asr.sh \
  --gguf-0.6b /remote/stage/qwen3-asr-0.6b.gguf \
  --gguf-0.6b-sha256 <sha256-from-vast-evidence> \
  --reference-0.6b /remote/stage/reference-0.6b \
  --reference-0.6b-sha256 <manifest-sha256-from-vast-evidence> \
  --gguf-1.7b /remote/stage/qwen3-asr-1.7b.gguf \
  --gguf-1.7b-sha256 <sha256-from-vast-evidence> \
  --reference-1.7b /remote/stage/reference-1.7b \
  --reference-1.7b-sha256 <manifest-sha256-from-vast-evidence> \
  --approval-evidence /remote/stage/qwen3-asr-owner-approval.json \
  --evidence-dir /remote/evidence/qwen3-asr-metal
```

The VAST worker emits the same complete command in
`evidence/apple-verifier-command.txt` when run with `--variant all`, including
both GGUF paths/hashes, both reference paths/manifest hashes, and an evidence
directory placeholder. Apple requires all four caller-supplied hashes, records
both expected and actual reference manifest hashes, and verifies each reference
manifest hash before inner payload hashes or Cargo. It requires both
per-variant PASS markers and performs no network or publication action. Pull
only the evidence, then remove the staged model data or destroy the remote
host.
