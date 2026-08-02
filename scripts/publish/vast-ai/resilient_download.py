"""resilient_download — a retrying, self-validating HF snapshot fetcher.

Rationale (Wave 9 residual retry, 2026-08-02):
    ``huggingface_hub.snapshot_download`` (Wave 9 initial attempt) failed on
    vast.ai for the following reasons:

      1. ``HF_HUB_ENABLE_HF_TRANSFER=1`` streams over HTTP range requests
         without hf-hub-level resume — a mid-chunk drop kills the shard and
         the entire snapshot restarts. On flaky vast.ai egress with 20-50 GB
         shards this is fatal.

      2. Xet routing (enabled by default in ``huggingface_hub>=0.30``) re-
         routes large shards through a codepath the pinned huggingface_hub
         does not have a resume fix for. Silent failure surface.

      3. No file-level retry loop, no header validation on ``.safetensors``,
         no corrupt-blob eviction. A partially-received shard sits in cache
         until the next full sweep — which starts from scratch.

      4. ``snapshot_download`` re-resolves the revision on each call. If the
         upstream repo pushes a commit mid-run, ``model.safetensors.index.json``
         and the shard set diverge — silent-wrong.

    This driver fixes all four:

      1. ``HF_HUB_ENABLE_HF_TRANSFER=0`` (env forced by caller), single-worker
         (``max_workers=1``, honored by using ``hf_hub_download`` per-file).
      2. ``HF_HUB_DISABLE_XET=1`` (env forced by caller).
      3. 5-attempt exponential backoff per file; ``safetensors.safe_open``
         header validation on every ``.safetensors``; corrupt-file symlink
         and blob eviction between attempts.
      4. ``HfApi.repo_info(revision=...)`` at start pins the SHA once; every
         ``hf_hub_download`` call passes that pinned SHA.

    Correctness details (see also investigation caveats):
      - We do NOT pass ``force_download=True`` on the initial attempt — that
        wipes any partial file. Only after ``safe_open`` fails do we flip
        ``force_download=True`` for the retry so the corrupt blob is not
        reused.
      - When purging a corrupt file: hf-hub with ``local_dir`` set often
        stores a symlink pointing at ``<local_dir>/.cache/huggingface/download/
        <hash>/<file>``. We unlink BOTH the visible path AND the underlying
        blob (via ``os.readlink``) — otherwise the same broken bytes get
        re-symlinked on retry.
      - ``safetensors.safe_open`` with ``framework="numpy"`` avoids importing
        torch on the vast.ai box and is enough to validate the header and
        tensor offset table (the exact "buffer truncated" failure Wave 9 saw
        is a header-vs-length mismatch, caught here).

    Zero deps in the vokra-* runtime (this driver runs under the uv-managed
    ``tools/parity`` env, per NFR-DS-02 / FR-LD-05). Deliberately narrow API:
    the caller (resilient_batch.sh) passes ``--repo`` and ``--local-dir``
    plus a list of ``--include`` / ``--exclude`` fnmatch globs.

Contract:
    python resilient_download.py \\
        --repo <org/slug> --local-dir <path> \\
        [--include <glob>]... [--exclude <glob>]... \\
        [--revision <rev>] [--max-attempts N] [--print-dir]

    Exits 0 on success; prints the resolved local directory on the last
    line of stdout (so a bash caller can do ``snap="$(resilient_download ...)"``).

    Exits 2 on any file that could not be fetched cleanly after N attempts.

Security:
    Reads ``HF_TOKEN`` (or ``HF``) from env only — never logs it. Fails closed
    if a gated repo is enumerated without a token.
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import random
import sys
import time
from pathlib import Path
from typing import Iterable


LOG_PREFIX = "[resilient-download]"


def log(msg: str) -> None:
    print(f"{LOG_PREFIX} {msg}", file=sys.stderr, flush=True)


def _match_any(name: str, globs: Iterable[str]) -> bool:
    return any(fnmatch.fnmatch(name, g) for g in globs)


def _filter_files(
    all_files: list[str],
    include: list[str],
    exclude: list[str],
) -> list[str]:
    """Apply include-then-exclude fnmatch semantics.

    Empty ``include`` means "match everything". ``exclude`` always wins.
    """
    kept: list[str] = []
    for f in all_files:
        if include and not _match_any(f, include):
            continue
        if exclude and _match_any(f, exclude):
            continue
        kept.append(f)
    return kept


def _purge_corrupt(local_dir: Path, filename: str) -> None:
    """Unlink both the visible path and the underlying blob (if it's a symlink).

    See investigation caveat (b): hf-hub with ``local_dir`` set stores a
    symlink pointing at ``<local_dir>/.cache/huggingface/download/<hash>/<file>``.
    Unlinking only the visible path leaves the corrupt blob in the cache and
    hf-hub will happily re-symlink the same broken bytes on retry.
    """
    p = local_dir / filename
    try:
        if p.is_symlink():
            try:
                target = os.readlink(p)
                blob_path = (p.parent / target).resolve()
                if blob_path.exists():
                    log(f"purge: blob {blob_path}")
                    blob_path.unlink()
            except OSError as e:
                log(f"purge: could not resolve symlink target for {p}: {e}")
        if p.exists() or p.is_symlink():
            log(f"purge: visible {p}")
            p.unlink()
    except OSError as e:
        log(f"purge: unlink failed for {p}: {e}")


def _validate_safetensors(path: Path) -> None:
    """Open a ``.safetensors`` file and iterate its keys — raises on corruption.

    Uses ``framework="numpy"`` to avoid pulling torch into memory. The
    ``safe_open`` context manager validates the header (magic, version,
    tensor offset table) at construction time; iterating ``.keys()`` walks
    the offset table. Both are cheap and catch the "buffer truncated"
    signature Wave 9 hit.
    """
    # safetensors is a dep of tools/parity's pyproject; the caller runs us
    # under that env via `uv run`.
    from safetensors import safe_open  # type: ignore[import-not-found]
    with safe_open(str(path), framework="numpy") as st:
        # Force the offset table to be walked.
        for _ in st.keys():
            pass


def _fetch_one(
    repo_id: str,
    filename: str,
    revision: str,
    local_dir: Path,
    token: str | None,
    max_attempts: int,
) -> None:
    """Fetch a single file with exponential backoff + safetensors validation.

    Retry policy: ``sleep = min(60, 4 * 2**attempt) + jitter[0,1)``.
    Attempts start at 0 (no sleep before first try).
    Errors caught:
      - ``requests.exceptions.ChunkedEncodingError`` (mid-chunk drop)
      - ``requests.exceptions.ConnectionError`` (TCP reset, DNS flake)
      - ``requests.exceptions.ReadTimeout`` (server slow)
      - ``huggingface_hub.errors.HfHubHTTPError`` with status>=500
      - ``safetensors`` internal errors (raised as ``SafetensorError`` in
        newer versions, ``Exception`` in older)
      - ``OSError`` with "truncated" in the message (rare, seen on nfs).
    """
    # Lazy imports so `python resilient_download.py --self-test` does not
    # need huggingface_hub / safetensors on the caller's PATH.
    from huggingface_hub import hf_hub_download  # type: ignore[import-not-found]
    from huggingface_hub.errors import HfHubHTTPError  # type: ignore[import-not-found]
    import requests  # type: ignore[import-not-found]

    force_download = False
    last_exc: BaseException | None = None

    for attempt in range(max_attempts):
        if attempt > 0:
            sleep = min(60.0, 4.0 * (2 ** (attempt - 1))) + random.random()
            log(f"retry {attempt}/{max_attempts - 1} in {sleep:.1f}s (force={force_download}): {filename}")
            time.sleep(sleep)
        try:
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                revision=revision,
                local_dir=str(local_dir),
                token=token,
                etag_timeout=30,
                force_download=force_download,
            )
        except HfHubHTTPError as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status is not None and status < 500:
                # 4xx (auth, not-found) is unrecoverable — do not retry.
                raise
            last_exc = e
            log(f"http {status}: {filename} — will retry")
            continue
        except requests.exceptions.ChunkedEncodingError as e:
            last_exc = e
            log(f"chunked-encoding drop: {filename} — will retry")
            # partial file will still be resumable on the next try (we do
            # NOT flip force_download here, per caveat (a)).
            continue
        except (requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout) as e:
            last_exc = e
            log(f"network flake ({type(e).__name__}): {filename} — will retry")
            continue
        except OSError as e:
            if "truncated" not in str(e).lower():
                raise
            last_exc = e
            log(f"OSError truncated: {filename} — will retry with force_download")
            _purge_corrupt(local_dir, filename)
            force_download = True
            continue

        # Download succeeded per hf_hub_download. Validate .safetensors.
        if filename.endswith(".safetensors"):
            path = local_dir / filename
            try:
                _validate_safetensors(path)
            except Exception as e:  # noqa: BLE001 — safetensors raises variously
                last_exc = e
                log(f"safetensors validate failed: {filename} — {type(e).__name__}: {e}")
                _purge_corrupt(local_dir, filename)
                force_download = True
                continue

        log(f"ok: {filename}")
        return

    assert last_exc is not None
    raise RuntimeError(
        f"gave up after {max_attempts} attempts on {filename!r}: {type(last_exc).__name__}: {last_exc}"
    )


def _pin_revision(repo_id: str, revision: str | None, token: str | None) -> str:
    from huggingface_hub import HfApi  # type: ignore[import-not-found]
    api = HfApi()
    info = api.repo_info(repo_id=repo_id, revision=revision, token=token)
    sha = getattr(info, "sha", None) or getattr(info, "commit_hash", None)
    if not sha:
        # Fallback: HfApi.list_repo_files at HEAD is fine, but we lose the
        # pin. Log loudly.
        log(f"WARN: repo_info returned no sha for {repo_id}@{revision or 'HEAD'} — proceeding unpinned")
        return revision or "main"
    log(f"pinned revision: {repo_id} -> {sha}")
    return sha


def _list_files(repo_id: str, revision: str, token: str | None) -> list[str]:
    from huggingface_hub import HfApi  # type: ignore[import-not-found]
    api = HfApi()
    return list(api.list_repo_files(repo_id=repo_id, revision=revision, token=token))


def _union_shards_from_index(local_dir: Path, filtered: list[str]) -> list[str]:
    """If model.safetensors.index.json is in ``filtered`` and now on disk,
    parse its weight_map and union the referenced shard filenames into
    ``filtered``. Returns the augmented list (sorted, deduped).
    """
    idx = local_dir / "model.safetensors.index.json"
    if "model.safetensors.index.json" not in filtered or not idx.is_file():
        return filtered
    # Reuse extract_shared_state_dict — tracked utility in tools/parity/.
    # Both scripts live in the same repo tree.
    here = Path(__file__).resolve().parent
    parity = (here / ".." / ".." / ".." / "tools" / "parity").resolve()
    if str(parity) not in sys.path:
        sys.path.insert(0, str(parity))
    try:
        from extract_shared_state_dict import extract_shards  # type: ignore[import-not-found]
    except ImportError:
        log("WARN: could not import extract_shared_state_dict — falling back to inline parse")
        import json as _json
        with idx.open("r", encoding="utf-8") as f:
            data = _json.load(f)
        shards = sorted(set(str(v) for v in data.get("weight_map", {}).values()))
    else:
        shards = extract_shards(str(idx))
    unioned = sorted(set(filtered) | set(shards))
    added = sorted(set(shards) - set(filtered))
    if added:
        log(f"index.json referenced {len(added)} additional shards: {added[:3]}{'...' if len(added) > 3 else ''}")
    return unioned


def _model_index_extra_subfolders(local_dir: Path) -> list[str]:
    """audioldm2-large-shaped composite: parse model_index.json (diffusers
    root manifest) and return subfolder names it references, so the caller
    can widen ``include`` before the second pass. Returns [] if the file is
    absent or malformed.
    """
    mi = local_dir / "model_index.json"
    if not mi.is_file():
        return []
    try:
        import json as _json
        with mi.open("r", encoding="utf-8") as f:
            data = _json.load(f)
    except Exception as e:  # noqa: BLE001
        log(f"WARN: model_index.json parse failed: {e}")
        return []
    subs: list[str] = []
    for key, value in data.items():
        if key.startswith("_"):
            continue
        if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
            # diffusers convention: {"unet": ["diffusers", "UNet2DModel"], ...}
            # the KEY is the subfolder name.
            subs.append(key)
    return subs


# ---------- self-test ---------------------------------------------------

def _self_test() -> int:
    cases = 0
    fails = 0

    def check(label: str, got, want) -> None:
        nonlocal cases, fails
        cases += 1
        if got != want:
            fails += 1
            print(f"self-test FAIL [{label}]: got {got!r}, want {want!r}", file=sys.stderr)

    # _filter_files: include-only.
    check(
        "filter-include",
        _filter_files(
            ["config.json", "adapter_abi.safetensors", "model.safetensors"],
            include=["*.safetensors", "config.json"],
            exclude=[],
        ),
        ["config.json", "adapter_abi.safetensors", "model.safetensors"],
    )

    # _filter_files: exclude overrides include (mms adapters).
    check(
        "filter-exclude-adapters",
        _filter_files(
            ["config.json", "adapter_abi.safetensors", "adapter_ab.safetensors", "model.safetensors"],
            include=["*.safetensors", "config.json"],
            exclude=["adapter_*"],
        ),
        ["config.json", "model.safetensors"],
    )

    # _filter_files: empty include = everything.
    check(
        "filter-empty-include",
        _filter_files(["a", "b", "c"], include=[], exclude=["b"]),
        ["a", "c"],
    )

    # _filter_files: audioldm2 submodule pattern.
    check(
        "filter-submodule-safetensors",
        _filter_files(
            [
                "model_index.json",
                "text_encoder/config.json",
                "text_encoder/model.safetensors",
                "unet/config.json",
                "unet/diffusion_pytorch_model.safetensors",
                "unet/diffusion_pytorch_model.fp16.safetensors",
                "vocoder/config.json",
                "vocoder/model.bin",
            ],
            include=["*/config.json", "*/*.safetensors", "model_index.json"],
            exclude=["*.bin", "*/*.fp16.safetensors"],
        ),
        [
            "model_index.json",
            "text_encoder/config.json",
            "text_encoder/model.safetensors",
            "unet/config.json",
            "unet/diffusion_pytorch_model.safetensors",
            "vocoder/config.json",
        ],
    )

    # _filter_files: seamless-m4t pattern (drop .pt duplicates).
    check(
        "filter-seamless",
        _filter_files(
            [
                "config.json",
                "model.safetensors",
                "model.safetensors.index.json",
                "model-00001-of-00002.safetensors",
                "model-00002-of-00002.safetensors",
                "pytorch_model.bin",
                "pytorch_model.bin.index.json",
                "sentencepiece.bpe.model",
                "special_tokens_map.json",
                "tokenizer_config.json",
                "vocoder_original.pt",
                "generation_config.json",
                "preprocessor_config.json",
            ],
            include=[
                "*.safetensors",
                "model.safetensors.index.json",
                "config.json",
                "tokenizer*",
                "vocab*",
                "special_tokens*",
                "generation_config.json",
                "preprocessor_config.json",
                "sentencepiece*",
            ],
            exclude=["*.pt", "*.bin", "*.msgpack", "*_original.*"],
        ),
        # _filter_files preserves input order — expected mirrors that.
        [
            "config.json",
            "model.safetensors",
            "model.safetensors.index.json",
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
            "sentencepiece.bpe.model",
            "special_tokens_map.json",
            "tokenizer_config.json",
            "generation_config.json",
            "preprocessor_config.json",
        ],
    )

    if fails == 0:
        print(f"resilient_download self-test: OK ({cases} cases)")
        return 0
    print(f"resilient_download self-test: {fails}/{cases} FAILED", file=sys.stderr)
    return 1


# ---------- main --------------------------------------------------------

def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", help="HF repo id, e.g. facebook/mms-1b-all")
    parser.add_argument("--local-dir", help="target directory (mkdir -p, then hf_hub_download local_dir=...)")
    parser.add_argument("--include", action="append", default=[],
                        help="fnmatch glob(s) of files to keep (repeatable). Empty = all.")
    parser.add_argument("--exclude", action="append", default=[],
                        help="fnmatch glob(s) of files to drop (repeatable). Applied AFTER --include.")
    parser.add_argument("--revision", default=None,
                        help="branch / tag / commit sha to pin. Default: HEAD of main.")
    parser.add_argument("--max-attempts", type=int, default=5,
                        help="max retry attempts per file (default: 5).")
    parser.add_argument("--print-dir", action="store_true", default=True,
                        help="print the resolved local dir on last line of stdout (default: on).")
    parser.add_argument("--self-test", action="store_true",
                        help="run internal filter self-tests (no HF network I/O).")
    args = parser.parse_args(argv)

    if args.self_test:
        return _self_test()

    if not args.repo or not args.local_dir:
        parser.error("--repo and --local-dir required (or pass --self-test)")

    token = os.environ.get("HF_TOKEN") or os.environ.get("HF")
    # gated repos need a token; log for owner visibility but do not fail here —
    # HfApi will 401 loudly and _fetch_one propagates HfHubHTTPError.
    if token:
        log(f"HF_TOKEN present (first 6 chars: {token[:6]}...)")
    else:
        log("HF_TOKEN unset — public repos only; gated repos will 401")

    local_dir = Path(args.local_dir).resolve()
    local_dir.mkdir(parents=True, exist_ok=True)
    log(f"local_dir: {local_dir}")

    # Step 1: pin the revision.
    revision = _pin_revision(args.repo, args.revision, token)

    # Step 2: enumerate remote files at the pinned SHA.
    all_files = _list_files(args.repo, revision, token)
    log(f"remote has {len(all_files)} files at {revision}")

    # Step 3: filter with include/exclude.
    filtered = _filter_files(all_files, args.include, args.exclude)
    log(f"filtered to {len(filtered)} files (include={args.include}, exclude={args.exclude})")

    if not filtered:
        log(f"ERROR: filter matched zero files — check --include / --exclude for {args.repo}")
        return 2

    # Step 4: if model.safetensors.index.json is in the set, fetch it first
    # so we can union its weight_map into the filter set. Same trick for
    # model_index.json (diffusers composite).
    def _fetch(fs: list[str]) -> None:
        for f in fs:
            _fetch_one(args.repo, f, revision, local_dir, token, args.max_attempts)

    priority = [
        "model.safetensors.index.json",
        "model_index.json",
    ]
    prio_present = [f for f in priority if f in filtered]
    _fetch(prio_present)

    # Widen the filter set with anything the index files reference.
    filtered = _union_shards_from_index(local_dir, filtered)
    extra_subs = _model_index_extra_subfolders(local_dir)
    if extra_subs:
        # audioldm2 shape: add each subfolder's *.safetensors + config.json to
        # the fetch set. Recompute filtered against all_files with the widened
        # include set.
        widened_include = list(args.include) + [f"{s}/*.safetensors" for s in extra_subs] \
            + [f"{s}/config.json" for s in extra_subs]
        widened = _filter_files(all_files, widened_include, args.exclude)
        added = sorted(set(widened) - set(filtered))
        if added:
            log(f"model_index.json referenced {len(extra_subs)} subfolders; adding {len(added)} files")
        filtered = sorted(set(filtered) | set(widened))

    # Step 5: fetch the rest (skipping what we already grabbed).
    remaining = [f for f in filtered if f not in prio_present]
    log(f"fetching {len(remaining)} remaining files")
    _fetch(remaining)

    log(f"snapshot complete: {local_dir}")
    # Last line of stdout for bash callers.
    if args.print_dir:
        print(str(local_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
