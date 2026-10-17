#!/usr/bin/env -S uv run --no-project --offline --python 3.12 python
"""Fail-closed source-only Moshi cache/state observation seam candidate.

The patcher authenticates the exact upstream ASR, transformer, LM, and root
Rust license fixtures before applying additive source edits.  It never imports
or executes Moshi/Candle code and never acquires models.  Outputs and the
receipt are exclusive-create artifacts; the authenticated inputs are never
modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from pathlib import Path

import asr_observer_patch as asr


UPSTREAM_COMMIT = "e6a55d2722a65870ef52a6c9f6ecfc0e90f38362"
SOURCE_URL = (
    "https://raw.githubusercontent.com/kyutai-labs/moshi/"
    f"{UPSTREAM_COMMIT}/rust/moshi-core/src/asr.rs"
)
MOSHI_KV_CACHE_URL = (
    "https://raw.githubusercontent.com/kyutai-labs/moshi/"
    f"{UPSTREAM_COMMIT}/rust/moshi-core/src/kv_cache.rs"
)
TRANSFORMER_URL = (
    "https://raw.githubusercontent.com/kyutai-labs/moshi/"
    f"{UPSTREAM_COMMIT}/rust/moshi-core/src/transformer.rs"
)
LM_URL = (
    "https://raw.githubusercontent.com/kyutai-labs/moshi/"
    f"{UPSTREAM_COMMIT}/rust/moshi-core/src/lm.rs"
)
LICENSE_URL = (
    "https://raw.githubusercontent.com/kyutai-labs/moshi/"
    f"{UPSTREAM_COMMIT}/rust/LICENSE"
)
SOURCE_BLOB_SHA1 = asr.SOURCE_BLOB_SHA1
SOURCE_BYTES = asr.SOURCE_BYTES
TRANSFORMER_BLOB_SHA1 = "4382d71044bd723ebbedade5043ff883683a70b0"
TRANSFORMER_BYTES = 40480
LM_BLOB_SHA1 = "6775ec8c5453c40a4f777303bd583db76e089dd8"
LM_BYTES = 37671
LICENSE_BLOB_SHA1 = asr.LICENSE_BLOB_SHA1
LICENSE_BYTES = asr.LICENSE_BYTES
PATCH_NAME = "moshi-cache-observer-seam-v2"
PATCH_MARKER = b"// Vokra candidate: cache/state observation seam v2."


class PatchError(ValueError):
    """Raised for source drift, unsafe paths, duplicate patches, or writes."""


def raw_sha1(value: bytes) -> str:
    return hashlib.sha1(value).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def git_blob_sha1(value: bytes) -> str:
    header = f"blob {len(value)}\0".encode("ascii")
    return hashlib.sha1(header + value).hexdigest()


def _authenticated(path: Path, expected_sha1: str, expected_bytes: int, label: str) -> bytes:
    """Authenticate complete bytes, including descriptor identity stability."""
    try:
        return asr._authenticated(path, expected_sha1, expected_bytes, label)
    except (OSError, asr.PatchError) as exc:
        raise PatchError(str(exc)) from exc


def _new_destination(path: Path, label: str) -> None:
    try:
        asr._new_destination(path, label)
    except (OSError, asr.PatchError) as exc:
        raise PatchError(str(exc)) from exc


def _write_exclusive(path: Path, data: bytes, label: str) -> tuple[int, int]:
    try:
        return asr._write_exclusive(path, data, label)
    except (OSError, asr.PatchError) as exc:
        raise PatchError(str(exc)) from exc


TRANSFORMER_TYPE = """#[derive(Debug, Clone)]
pub struct RawKvObservation {
    pub k: Option<Tensor>,
    pub v: Option<Tensor>,
    pub k_offset: usize,
    pub v_offset: usize,
    pub k_dim: usize,
    pub v_dim: usize,
    pub k_max_seq_len: usize,
    pub v_max_seq_len: usize,
    pub k_current_seq_len: usize,
    pub v_current_seq_len: usize,
}

#[derive(Debug, Clone)]
pub struct CacheObservation {
    pub upstream_current_seq_len: Vec<usize>,
    pub upstream_positions_for_zero_query: Vec<Vec<usize>>,
    pub last_reset_pos: Vec<usize>,
    pub raw_kv: Vec<RawKvObservation>,
}

"""
TRANSFORMER_METHOD = """    pub fn cache_observation(&self) -> Result<CacheObservation> {
        let mut upstream_current_seq_len = Vec::with_capacity(self.layers.len());
        let mut upstream_positions_for_zero_query = Vec::with_capacity(self.layers.len());
        let mut raw_kv = Vec::with_capacity(self.layers.len());
        for layer in self.layers.iter() {
            let cache = &layer.self_attn.kv_cache;
            upstream_current_seq_len.push(cache.current_seq_len());
            upstream_positions_for_zero_query.push(cache.positions(0));
            let raw = match cache {
                crate::kv_cache::KvCache::Rotating(cache) => RawKvObservation {
                    k: cache.k()?,
                    v: cache.v()?,
                    k_offset: cache.k_cache().offset(),
                    v_offset: cache.v_cache().offset(),
                    k_dim: cache.k_cache().dim(),
                    v_dim: cache.v_cache().dim(),
                    k_max_seq_len: cache.k_cache().max_seq_len(),
                    v_max_seq_len: cache.v_cache().max_seq_len(),
                    k_current_seq_len: cache.k_cache().current_seq_len(),
                    v_current_seq_len: cache.v_cache().current_seq_len(),
                },
            };
            raw_kv.push(raw);
        }
        Ok(CacheObservation {
            upstream_current_seq_len,
            upstream_positions_for_zero_query,
            last_reset_pos: self.last_reset_pos.clone(),
            raw_kv,
        })
    }

"""

LM_TYPE = """#[derive(Debug, Clone)]
pub struct LmObservation {
    pub transformer: transformer::CacheObservation,
}

"""
LM_METHOD = """    fn cache_observation(&self) -> Result<LmObservation> {
        match self {
            StreamingTransformer::Normal(transformer) => Ok(LmObservation {
                transformer: transformer.cache_observation()?,
            }),
            StreamingTransformer::Batched(_) => {
                candle::bail!("cache observation is unsupported for batched transformer")
            }
        }
    }

"""
LM_PUBLIC_METHOD = """    pub fn cache_observation(&self) -> Result<LmObservation> {
        self.transformer.cache_observation()
    }

"""

ASR_TYPES = """#[derive(Debug, Clone)]
pub struct AsrItemObservation {
    pub step_idx: usize,
    pub text_token: u32,
}

#[derive(Debug, Clone)]
pub struct AsrObservation {
    pub global_model_step_idx: usize,
    pub items: Vec<AsrItemObservation>,
    pub lm: crate::lm::LmObservation,
}

pub type AsrObserverV2 = Box<
    dyn FnMut(
        &AsrObservation,
        &crate::StreamMask,
        &Tensor,
        &Tensor,
        &[u32],
    ) -> Result<()> + Send,
>;

"""
ASR_FIELD = "    observer_v2: Option<AsrObserverV2>,\n"
ASR_INIT = "            observer_v2: None,\n"
ASR_SETTER = """    pub fn set_observer_v2(&mut self, observer: Option<AsrObserverV2>) {
        self.observer_v2 = observer;
    }

"""
ASR_CALL = """            // Vokra candidate: cache/state observation seam v2.
            if self.observer_v2.is_some() {
                let observation = AsrObservation {
                    global_model_step_idx: self.model_step_idx,
                    items: self
                        .batch
                        .iter()
                        .map(|item| AsrItemObservation {
                            step_idx: item.step_idx,
                            text_token: item.text_token,
                        })
                        .collect(),
                    lm: self.lm.cache_observation()?,
                };
                if let Some(observer) = self.observer_v2.as_mut() {
                    observer(
                        &observation,
                        mask,
                        &text_logits,
                        &transformer_out,
                        &text_tokens,
                    )?;
                }
            }
"""


def _replace_once(text: str, old: str, new: str, description: str) -> str:
    count = text.count(old)
    if count != 1:
        raise PatchError(f"{description} anchor count is {count}, expected 1")
    return text.replace(old, new, 1)


def _decode(source: bytes, label: str) -> str:
    try:
        return source.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PatchError(f"{label} is not UTF-8 Rust text") from exc


def _encode(text: str, label: str) -> bytes:
    value = text.encode("utf-8")
    if PATCH_MARKER not in value:
        raise PatchError(f"{label} does not contain the v2 patch marker")
    return value


def patch_transformer_bytes(source: bytes) -> bytes:
    text = _decode(source, "transformer source")
    if PATCH_MARKER in source:
        raise PatchError("cache observer marker already present in transformer")
    text = _replace_once(
        text,
        "impl StreamingTransformer {\n",
        TRANSFORMER_TYPE + "impl StreamingTransformer {\n",
        "transformer observation type",
    )
    text = _replace_once(
        text,
        "    pub fn forward(&mut self, xs: &Tensor) -> Result<Tensor> {\n",
        TRANSFORMER_METHOD
        + "    // Vokra candidate: cache/state observation seam v2.\n"
        + "    pub fn forward(&mut self, xs: &Tensor) -> Result<Tensor> {\n",
        "transformer observation method",
    )
    return _encode(text, "patched transformer")


def patch_lm_bytes(source: bytes) -> bytes:
    text = _decode(source, "lm source")
    if PATCH_MARKER in source:
        raise PatchError("cache observer marker already present in lm")
    text = _replace_once(
        text,
        "#[derive(Debug, Clone)]\nenum StreamingTransformer {\n",
        LM_TYPE + "#[derive(Debug, Clone)]\nenum StreamingTransformer {\n",
        "lm observation type",
    )
    text = _replace_once(
        text,
        "impl StreamingTransformer {\n    fn reset_batch_idx",
        "impl StreamingTransformer {\n"
        + "    // Vokra candidate: cache/state observation seam v2.\n"
        + LM_METHOD
        + "    fn reset_batch_idx",
        "lm observation method",
    )
    text = _replace_once(
        text,
        "    pub fn in_audio_codebooks(&self) -> usize {\n",
        LM_PUBLIC_METHOD + "    pub fn in_audio_codebooks(&self) -> usize {\n",
        "lm public observation method",
    )
    return _encode(text, "patched lm")


def patch_asr_bytes(source: bytes) -> bytes:
    try:
        patched = asr.patch_bytes(source)
    except asr.PatchError as exc:
        raise PatchError(str(exc)) from exc
    text = _decode(patched, "ASR source")
    if PATCH_MARKER in patched:
        raise PatchError("cache observer marker already present in asr")
    text = _replace_once(text, "pub struct State {\n", ASR_TYPES + "pub struct State {\n", "ASR observation types")
    text = _replace_once(text, "    observer: Option<AsrObserver>,\n", "    observer: Option<AsrObserver>,\n" + ASR_FIELD, "ASR v2 field")
    text = _replace_once(text, "            observer: None,\n", "            observer: None,\n" + ASR_INIT, "ASR v2 initialization")
    text = _replace_once(text, "    pub fn device(&self) -> &candle::Device {\n", ASR_SETTER + "    pub fn device(&self) -> &candle::Device {\n", "ASR v2 setter")
    text = _replace_once(text, asr.OBSERVER_CALL, asr.OBSERVER_CALL + ASR_CALL, "ASR v2 callback")
    return _encode(text, "patched asr")


def patch_bytes(source: bytes, transformer: bytes, lm: bytes) -> tuple[bytes, bytes, bytes]:
    return patch_asr_bytes(source), patch_transformer_bytes(transformer), patch_lm_bytes(lm)


def revert_asr_bytes(patched: bytes) -> bytes:
    text = _decode(patched, "patched ASR")
    reversals = (
        (ASR_CALL, ""),
        (ASR_SETTER, ""),
        (ASR_INIT, ""),
        (ASR_FIELD, ""),
        (ASR_TYPES, ""),
    )
    for old, new in reversals:
        text = _replace_once(text, old, new, "ASR v2 reversal")
    return asr.revert_bytes(text.encode("utf-8"))


def revert_transformer_bytes(patched: bytes) -> bytes:
    text = _decode(patched, "patched transformer")
    text = _replace_once(text, TRANSFORMER_METHOD, "", "transformer method reversal")
    text = _replace_once(text, "    // Vokra candidate: cache/state observation seam v2.\n", "", "transformer marker reversal")
    text = _replace_once(text, TRANSFORMER_TYPE, "", "transformer type reversal")
    return text.encode("utf-8")


def revert_lm_bytes(patched: bytes) -> bytes:
    text = _decode(patched, "patched lm")
    text = _replace_once(text, LM_PUBLIC_METHOD, "", "lm public method reversal")
    text = _replace_once(text, "    // Vokra candidate: cache/state observation seam v2.\n" + LM_METHOD, "", "lm method reversal")
    text = _replace_once(text, LM_TYPE, "", "lm type reversal")
    return text.encode("utf-8")


def _artifact(path: Path, data: bytes) -> dict[str, object]:
    return {
        "path": str(path),
        "git_blob_sha1": git_blob_sha1(data),
        "raw_sha1": raw_sha1(data),
        "bytes": len(data),
        "sha256": sha256_bytes(data),
    }


def apply(
    source_path: Path,
    transformer_path: Path,
    lm_path: Path,
    license_path: Path,
    output_path: Path,
    transformer_output_path: Path,
    lm_output_path: Path,
    receipt_path: Path,
) -> dict[str, object]:
    source = _authenticated(source_path, SOURCE_BLOB_SHA1, SOURCE_BYTES, "ASR source")
    transformer = _authenticated(transformer_path, TRANSFORMER_BLOB_SHA1, TRANSFORMER_BYTES, "transformer source")
    lm = _authenticated(lm_path, LM_BLOB_SHA1, LM_BYTES, "lm source")
    license_bytes = _authenticated(license_path, LICENSE_BLOB_SHA1, LICENSE_BYTES, "root Rust LICENSE")
    destinations = (output_path, transformer_output_path, lm_output_path, receipt_path)
    if len({os.path.abspath(path) for path in destinations}) != len(destinations):
        raise PatchError("all patched outputs and receipt must be distinct paths")
    for path, label in zip(
        destinations,
        ("patched ASR output", "patched transformer output", "patched lm output", "receipt"),
    ):
        _new_destination(path, label)
    patched_asr, patched_transformer, patched_lm = patch_bytes(source, transformer, lm)
    receipt: dict[str, object] = {
        "schema": "vokra-moshi-cache-observer-source-only-v2",
        "patch": PATCH_NAME,
        "source": {
            "repository": "kyutai-labs/moshi",
            "commit": UPSTREAM_COMMIT,
            "path": "rust/moshi-core/src/asr.rs",
            "official_raw_url": SOURCE_URL,
            "git_blob_sha1": SOURCE_BLOB_SHA1,
            "bytes": SOURCE_BYTES,
            "authenticated_input": str(source_path),
        },
        "transformer_source": {
            "path": "rust/moshi-core/src/transformer.rs",
            "official_raw_url": TRANSFORMER_URL,
            "git_blob_sha1": TRANSFORMER_BLOB_SHA1,
            "bytes": TRANSFORMER_BYTES,
            "authenticated_input": str(transformer_path),
        },
        "lm_source": {
            "path": "rust/moshi-core/src/lm.rs",
            "official_raw_url": LM_URL,
            "git_blob_sha1": LM_BLOB_SHA1,
            "bytes": LM_BYTES,
            "authenticated_input": str(lm_path),
        },
        "root_rust_license": {
            "path": "rust/LICENSE",
            "official_raw_url": LICENSE_URL,
            "git_blob_sha1": LICENSE_BLOB_SHA1,
            "bytes": LICENSE_BYTES,
            "authenticated_input": str(license_path),
        },
        "candle_api_provenance": {
            "repository": "huggingface/candle",
            "tag": "0.9.1",
            "commit": "cd96fa80da255e34f7b16b4ff98b6a31d557201b",
            "path_in_vcs": "candle-nn",
            "official_url": "https://github.com/huggingface/candle/blob/cd96fa80da255e34f7b16b4ff98b6a31d557201b/candle-nn/src/kv_cache.rs",
            "archive_sha256": "c1980d53280c8f9e2c6cbe1785855d7ff8010208b46e21252b978badf13ad69d",
            "used_accessors": [
                "RotatingKvCache::k",
                "RotatingKvCache::v",
                "RotatingKvCache::k_cache",
                "RotatingKvCache::v_cache",
                "RotatingCache::offset",
                "RotatingCache::dim",
                "RotatingCache::max_seq_len",
                "RotatingCache::current_seq_len",
            ],
            "license_approval": None,
        },
        "moshi_api_provenance": {
            "repository": "kyutai-labs/moshi",
            "commit": UPSTREAM_COMMIT,
            "path": "rust/moshi-core/src/kv_cache.rs",
            "official_raw_url": MOSHI_KV_CACHE_URL,
            "used_accessors": ["KvCache::current_seq_len", "KvCache::positions"],
        },
        "patched_outputs": {
            "asr": _artifact(output_path, patched_asr),
            "transformer": _artifact(transformer_output_path, patched_transformer),
            "lm": _artifact(lm_output_path, patched_lm),
        },
        "reversible": {
            "asr": revert_asr_bytes(patched_asr) == source,
            "transformer": revert_transformer_bytes(patched_transformer) == transformer,
            "lm": revert_lm_bytes(patched_lm) == lm,
        },
        "observer_contract": {
            "callback_default": "None",
            "callback_error": "propagates as candle::Result error and aborts the step",
            "observed": [
                "global_model_step_idx",
                "owned per-item step_idx and text_token",
                "Normal LM cache observation",
                "StreamMask",
                "text_logits",
                "transformer_out",
                "selected tokens",
            ],
            "call_site": "immediately after selected text_tokens.to_vec1 conversion and before state updates",
            "cache_fields": [
                "actual per-layer upstream_current_seq_len",
                "actual per-layer upstream_positions_for_zero_query from positions(0)",
                "actual last_reset_pos.clone()",
                "raw_kv per-layer k/v handles, None when upstream cache is empty",
                "raw_kv k/v offset, dim, max_seq_len capacity, and current_seq_len counters",
                "k()/v() handles alias current_data, are not deep immutable snapshots, and require synchronous serialization",
                "current_data is None when empty, narrows along cache.dim() from offset 0 to current_seq_len before capacity, then clones full physical rotating-buffer order at capacity",
                "all_data is the full allocated physical buffer and may include unused capacity; neither view is chronologically reordered here",
                "positions(0) is the upstream mapping result; it implies no reordering or tensor copy",
            ],
            "batched": "explicit unsupported error; no fallback",
            "transformations": "none",
        },
        "license_approval": None,
        "raw_kv_semantics": {
            "handles": "RotatingKvCache::k()/v() aliases; None when upstream cache is empty",
            "current_data": "Candle API: None when empty; narrow(cache.dim(), 0, current_seq_len) before capacity; clone() of full physical rotating-buffer order at/after capacity",
            "all_data": "Candle API: full allocated physical buffer, potentially including unused capacity",
            "k_v_delegate": "RotatingKvCache::k()/v() call current_data",
            "moshi_wrapper": "KvCache::current_seq_len/positions are distinct Moshi wrapper accessors",
            "positions_zero_query": "upstream mapping result with no implied reordering or copy",
            "collector_rule": "serialize synchronously and do not mutate tensor storage",
            "later_forwards": "may change aliased storage",
        },
        "gates": {
            "remote_build": "NOT_RUN",
            "runtime": "NOT_RUN",
            "cpu": "NOT_RUN",
            "metal": "NOT_RUN",
            "raw_kv_capture": "NOT_RUN",
            "independent_pcm_reference": "NOT_RUN",
            "dependency_license_owner_review": "BLOCKED_PENDING_REVIEW",
            "model_weights": "NOT_ACQUIRED",
            "publication": "NO_UPLOAD",
        },
        "source_sha256": sha256_bytes(source),
        "transformer_source_sha256": sha256_bytes(transformer),
        "lm_source_sha256": sha256_bytes(lm),
        "license_sha256": sha256_bytes(license_bytes),
    }
    if not all(receipt["reversible"].values()):
        raise PatchError("byte-exact reversal proof failed")
    _write_exclusive(output_path, patched_asr, "patched ASR output")
    _write_exclusive(transformer_output_path, patched_transformer, "patched transformer output")
    _write_exclusive(lm_output_path, patched_lm, "patched lm output")
    _write_exclusive(receipt_path, json.dumps(receipt, indent=2, sort_keys=True).encode() + b"\n", "receipt")
    return receipt


def _self_test() -> int:
    leaf = Path(__file__).resolve().parent
    source = _authenticated(leaf / "pinned-asr.rs", SOURCE_BLOB_SHA1, SOURCE_BYTES, "ASR source")
    transformer = _authenticated(leaf / "pinned-transformer.rs", TRANSFORMER_BLOB_SHA1, TRANSFORMER_BYTES, "transformer source")
    lm = _authenticated(leaf / "pinned-lm.rs", LM_BLOB_SHA1, LM_BYTES, "lm source")
    license_bytes = _authenticated(leaf / "pinned-LICENSE", LICENSE_BLOB_SHA1, LICENSE_BYTES, "root Rust LICENSE")
    if not license_bytes:
        raise PatchError("authenticated license fixture is empty")
    patched = patch_bytes(source, transformer, lm)
    if revert_asr_bytes(patched[0]) != source or revert_transformer_bytes(patched[1]) != transformer or revert_lm_bytes(patched[2]) != lm:
        raise PatchError("self-test reversal failed")
    print("cache observer self-test: PASS")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--transformer", type=Path)
    parser.add_argument("--lm", type=Path)
    parser.add_argument("--license", dest="license_path", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--transformer-output", type=Path)
    parser.add_argument("--lm-output", type=Path)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.self_test:
            return _self_test()
        required = (args.source, args.transformer, args.lm, args.license_path, args.output, args.transformer_output, args.lm_output, args.receipt)
        if any(value is None for value in required):
            parser.error("all source, output, and receipt paths are required unless --self-test is used")
        print(json.dumps(apply(*required), sort_keys=True))
    except (OSError, PatchError) as exc:
        print(f"cache observer patch refused: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
