#!/usr/bin/env bash
# M5-15-T38 — UTMOS reference-dumper execution-environment probe.
#
# Measures (never assumes) whether the upstream SaruLab UTMOS22 inference
# stack can be imported on this machine, so T18/T19 know which branch they
# are on (CC-completable vs owner-delegated Linux/x86 run). Records the
# outcome verbatim; a failure is a recorded result, not a reason to
# substitute a self-written mirror (Kokoro lesson, NFR-QL-04).
#
# Everything installs into a throw-away venv under ~/.cache/vokra-eval — the
# root Cargo.lock is never touched (NFR-DS-02: python deps stay in a venv,
# the same discipline as the other parity dumpers).
#
# Usage: tools/parity/utmos_env_probe.sh [outdir]
set -uo pipefail

OUT="${1:-$HOME/.cache/vokra-eval/out/utmos-flip}"
VENV="$HOME/.cache/vokra-eval/venv-utmos"
LOG="$OUT/env-probe.log"
mkdir -p "$OUT"

# Upstream pins (HF space sarulab-speech/UTMOS-demo requirements.txt @ rev
# 47212055c2ec, recorded in ~/.cache/vokra-eval/out/utmos-probe/decision-memo.md §(iii)).
FAIRSEQ_SHA=d03f4e771484a433f025f47744017c2eb6e9c6bc
PINNED_TORCH=1.11.0

say() { echo "== $*" | tee -a "$LOG"; }
run() { echo "\$ $*" >>"$LOG"; "$@" >>"$LOG" 2>&1; local rc=$?; echo "-> exit $rc" >>"$LOG"; return $rc; }

: >"$LOG"
say "probe started $(date -u +%Y-%m-%dT%H:%M:%SZ)"
say "host: $(uname -sm)  python3: $(python3 --version 2>&1)"

# --- Stage A: is the pinned env reproducible here at all? -------------------
say "stage A: pinned torch==$PINNED_TORCH availability on $(uname -sm)"
PY311="${PY311:-/opt/homebrew/bin/python3.11}"
if [ ! -x "$PY311" ]; then PY311="$(command -v python3)"; fi
say "stage A interpreter: $PY311 ($($PY311 --version 2>&1))"
if run "$PY311" -m pip download --no-deps --dest "$OUT/_pinprobe" "torch==$PINNED_TORCH"; then
  say "stage A RESULT: pinned torch $PINNED_TORCH IS installable here (unexpected — re-read decision-memo)"
else
  say "stage A RESULT: pinned torch $PINNED_TORCH is NOT installable here (no matching wheel) — env delta is unavoidable"
fi

# --- Stage B: modern torch + fairseq @ upstream pin -------------------------
say "stage B: build venv $VENV and install a modern torch + fairseq@$FAIRSEQ_SHA"
rm -rf "$VENV"
run "$PY311" -m venv "$VENV" || { say "stage B RESULT: venv creation FAILED"; exit 0; }
PIP="$VENV/bin/pip"
run "$PIP" install --upgrade "pip<25" "setuptools<70" wheel || say "WARN: bootstrap tooling install returned non-zero"
# numpy<2 and cython<3: fairseq@2022 builds .pyx against the numpy 1.x C API.
run "$PIP" install "numpy<2" "cython<3" || say "WARN: numpy/cython install returned non-zero"
run "$PIP" install "torch==2.2.2" "torchaudio==2.2.2" || say "WARN: torch install returned non-zero"
run "$PIP" install "omegaconf==2.0.6" "hydra-core==1.0.7" || say "WARN: hydra/omegaconf install returned non-zero"

say "stage B.1: fairseq @ $FAIRSEQ_SHA (upstream pin, --no-build-isolation so it sees numpy<2)"
if run "$PIP" install --no-build-isolation --no-deps "git+https://github.com/facebookresearch/fairseq.git@$FAIRSEQ_SHA"; then
  say "stage B.1 RESULT: fairseq pin INSTALLED"
else
  say "stage B.1 RESULT: fairseq pin build FAILED"
fi

# --- Stage C: does the upstream stack import + run? -------------------------
say "stage C: import probe"
run "$VENV/bin/python" -c '
import sys, traceback
def probe(name, fn):
    try:
        v = fn()
        print(f"IMPORT-OK   {name}: {v}")
    except Exception as e:
        print(f"IMPORT-FAIL {name}: {type(e).__name__}: {e}")
        traceback.print_exc(limit=3)
probe("torch",   lambda: __import__("torch").__version__)
probe("numpy",   lambda: __import__("numpy").__version__)
probe("omegaconf", lambda: __import__("omegaconf").__version__)
probe("fairseq", lambda: __import__("fairseq").__version__)
probe("fairseq.checkpoint_utils", lambda: __import__("fairseq.checkpoint_utils", fromlist=["x"]).__name__)
probe("fairseq.models.wav2vec.Wav2Vec2Model", lambda: __import__("fairseq.models.wav2vec", fromlist=["Wav2Vec2Model"]).Wav2Vec2Model.__name__)
'
say "probe finished $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "log: $LOG"
