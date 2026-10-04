from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pcm_dump
from contract import COMPOSITE_APPROVAL_SCHEMA, COMPOSITE_APPROVAL_SCOPE, ExecutionReadinessBlocked


class _Finite:
    def __init__(self, value: bool):
        self.value = value

    def all(self):
        return self

    def item(self):
        return self.value


class _Tensor:
    def __init__(self, raw: bytes, shape: tuple[int, ...], dtype: str, finite: bool = True):
        self.raw = raw
        self.shape = shape
        self.dtype = dtype
        self.finite = finite

    def numel(self):
        result = 1
        for dimension in self.shape:
            result *= dimension
        return result

    def element_size(self):
        return len(self.raw) // self.numel()

    def storage_offset(self):
        return 0

    def untyped_storage(self):
        return self.raw

    def isfinite(self):
        return _Finite(self.finite)

    def detach(self):
        return self

    def cpu(self):
        return self

    def float(self):
        raw = self.raw
        if self.dtype == "torch.bfloat16":
            raw = b"\x00" * (self.numel() * 4)
        return _Tensor(raw, self.shape, "torch.float32", self.finite)

    def contiguous(self):
        return self

    def reshape(self, *_shape):
        shape = tuple(int(dimension) for dimension in _shape)
        if len(shape) == 1 and shape[0] == -1:
            shape = (self.numel(),)
        return _Tensor(self.raw, shape, self.dtype, self.finite)

    def tolist(self):
        return [0] * self.numel()


class _Frame:
    def __init__(self, start: int):
        self.start = start
        self.shape = (1, 1, 1_920)


class _Padded:
    def clone(self):
        return self

    def view(self, *_shape):
        return self

    def __getitem__(self, key):
        start = key[2].start or 0
        return _Frame(start)


class _FakeNoGrad:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


class _FakeTorch:
    bfloat16 = "torch.bfloat16"
    float32 = "torch.float32"
    int64 = "torch.int64"
    bool = "torch.bool"

    class backends:
        class cpu:
            @staticmethod
            def get_cpu_capability():
                return "synthetic"

    class nn:
        class functional:
            @staticmethod
            def pad(value, _padding):
                return value if isinstance(value, _Padded) else _Padded()

    @staticmethod
    def no_grad():
        return _FakeNoGrad()

    @staticmethod
    def frombuffer(_body, dtype):
        assert dtype == _FakeTorch.float32
        return _Padded()

    @staticmethod
    def ones(*_args, **_kwargs):
        return object()

    _threads = 1
    _interop_threads = 1

    @classmethod
    def set_num_threads(cls, value):
        cls._threads = value

    @classmethod
    def get_num_threads(cls):
        return cls._threads

    @classmethod
    def set_num_interop_threads(cls, value):
        cls._interop_threads = value

    @classmethod
    def get_num_interop_threads(cls):
        return cls._interop_threads

    @staticmethod
    def use_deterministic_algorithms(_value):
        return None

    @staticmethod
    def are_deterministic_algorithms_enabled():
        return True


class _FakeMimi:
    bad_codes = False

    def __init__(self):
        self.sample_rate = 24_000
        self.frame_rate = 12.5
        self.frame_size = 1_920
        self.channels = 1
        self.num_codebooks = 32
        self.reset_masks = []
        self.frames = []
        self.codes = []

    def parameters(self):
        return iter([_Tensor(b"\x00\x00\x00\x00", (1,), "torch.float32")])

    def streaming(self, _batch):
        return _FakeNoGrad()

    def reset_streaming(self, mask):
        self.reset_masks.append(mask)

    def encode(self, frame):
        self.frames.append(frame)
        shape = (1, 31, 1) if self.bad_codes else (1, 32, 1)
        codes = _Tensor(b"\x00" * (32 * 8), shape, "torch.int64")
        self.codes.append(codes)
        return codes


class _FakeLM:
    dep_q = 0
    n_q = 32
    text_card = 4_000
    dim = 2_048
    context = 375
    num_codebooks = 33
    device = "cpu"
    dtype = "torch.bfloat16"


class _FakeLMGen:
    emit_logits = True
    emit_tokens = True
    last = None

    def __init__(self, lm, **kwargs):
        self.lm = lm
        self.kwargs = kwargs
        self._streaming_state = object()
        self.calls = []
        self.reset_masks = []
        type(self).last = self

    def streaming(self, _batch):
        return _FakeNoGrad()

    def reset_streaming(self, mask):
        self.reset_masks.append(mask)

    def step(self, codes):
        self.calls.append(codes)
        if self.emit_logits:
            self.kwargs["on_text_logits_hook"](
                _Tensor(b"\x00" * (4_000 * 2), (1, 1, 1, 4_000), "torch.bfloat16")
            )
        if self.emit_tokens:
            self.kwargs["on_text_hook"](_Tensor(b"\x00" * 8, (1,), "torch.int64"))
        return _Tensor(b"\x00" * 8, (1, 1, 1), "torch.int64")


class _FakeCapture:
    last = None

    def __init__(self, checkpoints):
        self.checkpoints = checkpoints
        self.files = {}
        self.events = []
        self.state_snapshots = []
        self.phase = "warmup"
        self.step = 0
        self.installed = False
        self.uninstalled = False
        type(self).last = self

    def install(self, _transformer, _lm):
        self.installed = True

    def record_state_snapshot(self, phase, step):
        self.state_snapshots.append((phase, step))

    def require_complete_coverage(self):
        assert self.installed
        assert self.state_snapshots == [("initial", 0), ("after_reset", 0)]

    def uninstall(self, _transformer):
        self.uninstalled = True


class _FakeDecoder:
    MODEL_NAME = "model.safetensors"
    MIMI_NAME = "mimi.safetensors"
    TOKENIZER_NAME = "tokenizer.model"
    DSM_REPOSITORY = "https://example.invalid/dsm.git"
    DSM_REVISION = "d" * 40

    def require_clean_head(self, _head):
        return None

    def authenticate_streaming_source_contract(self, _dsm, _moshi):
        return {"status": "SOURCE_ONLY_PREPARATION"}

    def authenticate_model(self, _model, _config):
        return {"tensor_manifest_sha256": pcm_dump.MODEL_TENSOR_MANIFEST_SHA256}


class _FakeInfo:
    raw_config = {"model_type": "stt", "mimi_config_name": None, "lora_name": None}
    mimi_config = None
    lora_weights = None
    model_type = "stt"
    stt_config = {"audio_silence_prefix_seconds": 1.0, "audio_delay_seconds": 2.5}

    def __init__(self):
        self.mimi = _FakeMimi()
        self.lm = _FakeLM()

    def get_mimi(self, device):
        assert device == "cpu"
        return self.mimi

    def get_moshi(self, device, dtype):
        assert device == "cpu" and dtype == _FakeTorch.bfloat16
        return self.lm


class PcmDumpTests(unittest.TestCase):
    def _args(self, root: Path, *, source_packet: Path | None = None) -> argparse.Namespace:
        return argparse.Namespace(
            out=root / "not-created",
            expected_head="0" * 40,
            approval_evidence=root / "approval.json",
            approval_sha256="0" * 64,
            dependency_closure_sha256="1" * 64,
            model=root / "model",
            dsm_source=root / "dsm",
            moshi_source=root / "moshi",
            source_packet=source_packet,
            pcm_input=root / "audio.f32le",
            pcm_input_sha256="0" * 64,
        )

    def test_padding_reproduces_official_ceil_without_native_shift(self) -> None:
        plan = pcm_dump._padding_plan(
            1,
            sample_rate=24_000,
            frame_size=1_920,
            prefix_seconds=1.0,
            suffix_seconds=3.0,
        )
        self.assertEqual(plan["prefix_samples"], 24_000)
        self.assertEqual(plan["suffix_samples"], 72_000)
        self.assertEqual(plan["unrounded_samples"], 96_001)
        self.assertEqual(plan["padded_samples"], 97_920)
        self.assertEqual(plan["frames"], 51)

    def test_pcm_input_is_bounded_and_authenticated(self) -> None:
        with tempfile.TemporaryDirectory(prefix="kyutai-pcm-input-") as raw:
            path = Path(raw) / "audio.f32le"
            body = b"\x00\x00\x00\x00" * 3
            path.write_bytes(body)
            expected = hashlib.sha256(body).hexdigest()
            observed, metadata = pcm_dump._read_pcm_input(path, expected)
            self.assertEqual(observed, body)
            self.assertEqual(metadata["sha256"], expected)
            self.assertTrue(metadata["sha256_verified"])
            self.assertEqual(metadata["sample_rate"], 24_000)
            with self.assertRaises(ValueError):
                pcm_dump._read_pcm_input(path, "0" * 64)
            path.write_bytes(b"\x00")
            with self.assertRaises(ValueError):
                pcm_dump._read_pcm_input(path, hashlib.sha256(b"\x00").hexdigest())

    def test_pcm_input_rejects_growth_and_oversize_before_unbounded_read(self) -> None:
        with tempfile.TemporaryDirectory(prefix="kyutai-pcm-bounds-") as raw:
            path = Path(raw) / "audio.f32le"
            body = b"\x00\x00\x00\x00"
            path.write_bytes(body)
            expected = hashlib.sha256(body).hexdigest()
            with patch.object(pcm_dump, "_read_bounded", create=True, return_value=body + b"grow"):
                with self.assertRaises(ValueError):
                    pcm_dump._read_pcm_input(path, expected)
            with path.open("wb") as stream:
                stream.truncate(pcm_dump.PCM_MAX_INPUT_BYTES + 1)
            with self.assertRaises(ValueError):
                pcm_dump._read_pcm_input(path, expected)

    def test_logits_require_official_four_dimensional_shape_and_finite_values(self) -> None:
        body = b"\x00" * (pcm_dump.TEXT_CARD * 4)
        logits = _Tensor(body, (1, 1, 1, pcm_dump.TEXT_CARD), "torch.bfloat16")
        source_shape, comparison_shape, source_dtype, exported = pcm_dump._validate_pcm_logits(logits, reserved_bytes=0)
        self.assertEqual(source_shape, [1, 1, 1, pcm_dump.TEXT_CARD])
        self.assertEqual(comparison_shape, [1, pcm_dump.TEXT_CARD])
        self.assertEqual(source_dtype, "torch.bfloat16")
        self.assertEqual(len(exported), pcm_dump.TEXT_CARD * 4)
        with self.assertRaises(ValueError):
            pcm_dump._validate_pcm_logits(
                _Tensor(body, (1, pcm_dump.TEXT_CARD), "torch.bfloat16"), reserved_bytes=0
            )
        with self.assertRaises(ValueError):
            pcm_dump._validate_pcm_logits(
                _Tensor(body, (1, 1, 1, pcm_dump.TEXT_CARD), "torch.bfloat16", finite=False),
                reserved_bytes=0,
            )

    def test_codes_require_exact_official_integer_boundary(self) -> None:
        body = b"\x00" * (32 * 8)
        shape, dtype, observed = pcm_dump._validate_codes(
            _Tensor(body, (1, 32, 1), "torch.int64"), expected_codebooks=32
        )
        self.assertEqual(shape, (1, 32, 1))
        self.assertEqual(dtype, "torch.int64")
        self.assertEqual(observed, body)
        with self.assertRaises(ValueError):
            pcm_dump._validate_codes(
                _Tensor(body, (1, 31, 1), "torch.int64"), expected_codebooks=32
            )
        with self.assertRaises(ValueError):
            pcm_dump._validate_codes(
                _Tensor(body, (1, 32, 1), "torch.int32"), expected_codebooks=32
            )

    def test_checkpoint_factory_has_all_local_overrides_and_rejects_optional_branches(self) -> None:
        calls = {}

        class CheckpointInfo:
            @staticmethod
            def from_hf_repo(repository, **kwargs):
                calls["repository"] = repository
                calls["kwargs"] = kwargs
                return _FakeInfo()

        loaders = type("Loaders", (), {"CheckpointInfo": CheckpointInfo})
        with tempfile.TemporaryDirectory(prefix="kyutai-pcm-factory-") as raw:
            info = pcm_dump._build_checkpoint_info(loaders, Path(raw), _FakeDecoder)
        self.assertIsInstance(info, _FakeInfo)
        self.assertEqual(calls["repository"], pcm_dump.HF_REPOSITORY)
        self.assertEqual(calls["kwargs"]["revision"], pcm_dump.HF_REVISION)
        self.assertTrue(all(isinstance(calls["kwargs"][key], Path) for key in ("moshi_weights", "mimi_weights", "tokenizer", "config_path")))
        self.assertNotIn("mimi_config_path", calls["kwargs"])
        self.assertNotIn("lora_weights", calls["kwargs"])

        class BadInfo(_FakeInfo):
            raw_config = {"model_type": "stt", "mimi_config_name": "remote.json", "lora_name": None}

        class BadCheckpointInfo:
            @staticmethod
            def from_hf_repo(_repository, **_kwargs):
                return BadInfo()

        bad_loaders = type("Loaders", (), {"CheckpointInfo": BadCheckpointInfo})
        with tempfile.TemporaryDirectory(prefix="kyutai-pcm-factory-bad-") as raw:
            with self.assertRaises(ValueError):
                pcm_dump._build_checkpoint_info(bad_loaders, Path(raw), _FakeDecoder)

    def _run_fake_real(self, *, emit_logits: bool = True, emit_tokens: bool = True, bad_codes: bool = False):
        with tempfile.TemporaryDirectory(prefix="kyutai-pcm-orchestration-") as raw:
            root = Path(raw)
            args = self._args(root, source_packet=root / "source")
            raw_metadata = {
                "path": str(root / "audio.f32le"),
                "bytes": 4,
                "sha256": "0" * 64,
                "sha256_expected": "0" * 64,
                "sha256_verified": True,
                "sample_rate": 24_000,
                "channels": 1,
                "dtype": "float32-le",
                "endianness": "little",
                "layout": "interleaved-mono",
                "samples": 627_840,
            }
            class RuntimeLM(_FakeLMGen):
                pass

            RuntimeLM.emit_logits = emit_logits
            RuntimeLM.emit_tokens = emit_tokens
            fake_info = _FakeInfo()
            fake_info.mimi.bad_codes = bad_codes
            written = {}
            readiness = {"approval": {"decision": "APPROVED", "scope": "synthetic"}, "dependency_closure_sha256": "1" * 64}
            with patch.object(pcm_dump, "require_execution_readiness", return_value=readiness), \
                    patch.object(pcm_dump, "require_pcm_source_packet", return_value={"status": "SOURCE_ONLY_PREPARATION"}), \
                    patch.object(pcm_dump, "require_source_packet", return_value={"status": "SOURCE_ONLY_PREPARATION"}), \
                    patch.object(pcm_dump, "_decoder_helpers", return_value=_FakeDecoder()), \
                    patch.object(pcm_dump, "git_identity", return_value={"commit": "a" * 40}), \
                    patch.object(pcm_dump, "authenticate_composite_model", return_value={}), \
                    patch.object(pcm_dump, "_read_pcm_input", return_value=(b"\x00\x00\x00\x00", raw_metadata)), \
                    patch.object(pcm_dump, "_import_official_runtime", return_value=(_FakeTorch, object(), RuntimeLM, object())), \
                    patch.object(pcm_dump, "_build_checkpoint_info", return_value=fake_info), \
                    patch.object(pcm_dump, "RingCapture", _FakeCapture), \
                    patch.object(pcm_dump, "_write_output", side_effect=lambda out, files, manifest: written.update({"out": out, "files": files, "manifest": manifest})):
                pcm_dump.real(args)
            return RuntimeLM.last, fake_info.mimi, _FakeCapture.last, written

    def test_real_mock_orchestration_uses_same_codes_and_reset_mask(self) -> None:
        lm_gen, mimi, capture, written = self._run_fake_real()
        self.assertEqual(len(mimi.frames), pcm_dump.PCM_MAX_FRAMES + 1)
        self.assertEqual(len(lm_gen.calls), pcm_dump.PCM_MAX_FRAMES + 1)
        self.assertTrue(all(codes is mimi.codes[index] for index, codes in enumerate(lm_gen.calls)))
        self.assertEqual(len(mimi.reset_masks), 1)
        self.assertEqual(len(lm_gen.reset_masks), 1)
        self.assertIs(mimi.reset_masks[0], lm_gen.reset_masks[0])
        self.assertEqual(capture.state_snapshots, [("initial", 0), ("after_reset", 0)])
        self.assertTrue(capture.uninstalled)
        manifest = written["manifest"]
        self.assertEqual(manifest["coverage"]["mimi_encode_calls"], pcm_dump.PCM_MAX_FRAMES + 1)
        self.assertEqual(manifest["ring_cache"]["calls_per_pcm_frame"], 1)
        self.assertEqual(manifest["ring_cache"]["checkpoints"], [0, 1, 374, 375, 376])
        self.assertEqual(
            manifest["ring_cache"]["checkpoint_file_pattern"],
            "kv/{phase}-step-{step:04d}-layer-{layer:02d}-{keys|values}.bin",
        )
        self.assertEqual(manifest["lm"]["generation"]["use_sampling"], False)
        self.assertEqual(manifest["input"]["padding"]["suffix_seconds"], 3.0)
        self.assertEqual(manifest["joint_steps"][0]["phase"], "warmup")
        self.assertEqual(manifest["joint_steps"][0]["step"], 0)
        self.assertEqual(manifest["joint_steps"][-1]["phase"], "after_reset")
        self.assertEqual(manifest["joint_steps"][-1]["step"], 0)
        self.assertEqual(manifest["text"]["logits"][0]["source_shape"], [1, 1, 1, 4_000])
        self.assertEqual(manifest["text"]["logits"][0]["comparison_shape"], [1, 4_000])
        self.assertTrue(all(row["lm_call_ordinal"] == 0 for row in manifest["joint_steps"]))
        self.assertTrue(all(row["lm_call_ordinal"] == 0 for row in manifest["text"]["logits"]))
        self.assertTrue(all(row["lm_call_ordinal"] == 0 for row in manifest["text"]["tokens"]))

    def test_real_mock_orchestration_rejects_missing_logit_or_token_events(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "missing a joint step, logits"):
            self._run_fake_real(emit_logits=False)
        with self.assertRaisesRegex(RuntimeError, "missing a joint step, logits"):
            self._run_fake_real(emit_tokens=False)

    def test_real_mock_orchestration_rejects_bad_mimi_codes_before_lm_step(self) -> None:
        with self.assertRaisesRegex(ValueError, "Mimi code shape"):
            self._run_fake_real(bad_codes=True)

    def test_text_tokens_require_exact_shape_dtype_and_range(self) -> None:
        with self.assertRaises(ValueError):
            pcm_dump._tensor_values(
                _Tensor(b"\x00" * 8, (1, 1), "torch.int64"),
                label="token",
                expected_shape=(1,),
            )

        class OutOfRange(_Tensor):
            def reshape(self, *_shape):
                return self

            def tolist(self):
                return [pcm_dump.TEXT_CARD]

        with self.assertRaises(ValueError):
            pcm_dump._tensor_values(
                OutOfRange(b"\x00" * 8, (1,), "torch.int64"),
                label="token",
                expected_shape=(1,),
            )

    def test_artifact_budget_rejects_before_writer(self) -> None:
        class SizedOnly:
            def __len__(self):
                return pcm_dump._MAX_ARTIFACT_BYTES

        with self.assertRaises(ValueError):
            pcm_dump._check_artifact_budget({"huge.bin": SizedOnly()}, {"format": pcm_dump.PCM_SCHEMA})

    def test_real_blocks_on_non_linux_before_model_or_source_access(self) -> None:
        with tempfile.TemporaryDirectory(prefix="kyutai-pcm-readiness-") as raw:
            root = Path(raw)
            args = self._args(root)
            with patch.object(pcm_dump.platform, "system", return_value="Darwin"), \
                    patch.object(pcm_dump.platform, "machine", return_value="arm64"), \
                    patch.object(pcm_dump, "authenticate_composite_model", side_effect=AssertionError("model reached")), \
                    patch.object(pcm_dump, "require_pcm_source_packet", side_effect=AssertionError("source reached")):
                with self.assertRaises(ExecutionReadinessBlocked):
                    pcm_dump.real(args)

    def test_real_blocks_unknown_closure_before_model_or_source_access(self) -> None:
        with tempfile.TemporaryDirectory(prefix="kyutai-pcm-closure-") as raw:
            root = Path(raw)
            args = self._args(root)
            approval = {
                "schema": COMPOSITE_APPROVAL_SCHEMA,
                "scope": COMPOSITE_APPROVAL_SCOPE,
                "decision": "APPROVED",
                "execution": "VAST_ONLY",
                "head": args.expected_head,
                "checkout": str(Path.cwd()),
            }
            body = (json.dumps(approval, sort_keys=True) + "\n").encode()
            args.approval_evidence.write_bytes(body)
            args.approval_sha256 = hashlib.sha256(body).hexdigest()
            with patch.object(pcm_dump.platform, "system", return_value="Linux"), \
                    patch.object(pcm_dump.platform, "machine", return_value="x86_64"), \
                    patch.object(pcm_dump, "authenticate_composite_model", side_effect=AssertionError("model reached")), \
                    patch.object(pcm_dump, "require_pcm_source_packet", side_effect=AssertionError("source reached")):
                with self.assertRaises(ExecutionReadinessBlocked):
                    pcm_dump.real(args)

    def test_real_requires_both_source_contracts_after_readiness(self) -> None:
        with tempfile.TemporaryDirectory(prefix="kyutai-pcm-source-gates-") as raw:
            root = Path(raw)
            args = self._args(root, source_packet=root / "source")
            readiness = {"approval": {"decision": "APPROVED", "scope": "synthetic"}, "dependency_closure_sha256": "1" * 64}
            with patch.object(pcm_dump, "require_execution_readiness", return_value=readiness), \
                    patch.object(pcm_dump, "require_pcm_source_packet", return_value={"status": "SOURCE_ONLY_PREPARATION"}) as pcm_gate, \
                    patch.object(pcm_dump, "require_source_packet", return_value={"status": "SOURCE_ONLY_PREPARATION"}) as lm_gate, \
                    patch.object(pcm_dump, "_decoder_helpers", side_effect=AssertionError("source helpers reached")):
                with self.assertRaises(AssertionError):
                    pcm_dump.real(args)
                pcm_gate.assert_called_once_with(args.source_packet)
                lm_gate.assert_called_once_with(args.source_packet)


if __name__ == "__main__":
    unittest.main()
