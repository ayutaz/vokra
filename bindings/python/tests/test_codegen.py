# SPDX-License-Identifier: Apache-2.0
"""C-header to ctypes generator coverage and fail-closed tests."""

from __future__ import annotations

import ctypes
import importlib.util
import sys
from pathlib import Path

import pytest

_PYTHON_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = _PYTHON_ROOT.parents[1]
_SRC = _PYTHON_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from vokra import _bindings as bindings  # noqa: E402

_GENERATOR_PATH = _PYTHON_ROOT / "scripts" / "gen-py-bindings.py"
_HEADER_PATH = _REPO_ROOT / "include" / "vokra.h"
_SPEC = importlib.util.spec_from_file_location("vokra_gen_py_bindings", _GENERATOR_PATH)
assert _SPEC is not None and _SPEC.loader is not None
generator = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(generator)


def _header_source() -> str:
    return generator.strip_comments(_HEADER_PATH.read_text(encoding="utf-8"))


def test_every_current_header_function_has_one_prototype() -> None:
    source = _header_source()
    parsed = generator.parse_functions(source)
    parsed_names = [name for name, _, _ in parsed]
    discovered_names = generator.discover_function_names(source)

    assert len(parsed_names) == 41
    assert len(parsed_names) == len(set(parsed_names))
    assert parsed_names == discovered_names
    assert set(bindings.PROTOTYPES) == set(parsed_names)


def test_parser_includes_previously_missed_return_shapes() -> None:
    parsed = {name: (ret, args) for name, ret, args in generator.parse_functions(_header_source())}

    assert parsed["vokra_backend_available"] == ("bool", ["int32_t"])
    assert parsed["vokra_session_options_create"] == (
        "struct vokra_session_options_t *",
        [],
    )


def test_current_wide_and_pointer_prototypes_are_exact() -> None:
    assert bindings.PROTOTYPES["vokra_backend_available"] == (
        ctypes.c_bool,
        (ctypes.c_int32,),
    )
    assert bindings.PROTOTYPES["vokra_session_options_create"] == (
        ctypes.c_void_p,
        (),
    )
    assert bindings.PROTOTYPES["vokra_aec_ref_push"] == (
        ctypes.c_int,
        (
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_size_t),
        ),
    )
    assert bindings.PROTOTYPES["vokra_session_create_from_bytes"][1][0] == ctypes.POINTER(
        ctypes.c_uint8
    )
    assert bindings.PROTOTYPES["vokra_asr_transcribe"][1][-1] == ctypes.POINTER(
        ctypes.c_char_p
    )
    assert bindings.PROTOTYPES["vokra_tts_synthesize"][1][1] is ctypes.c_char_p
    assert bindings.PROTOTYPES["vokra_string_free"][1] == (
        ctypes.POINTER(ctypes.c_char),
    )
    assert bindings.PROTOTYPES["vokra_s2s_text"][1][1] == ctypes.POINTER(
        ctypes.c_char
    )
    assert bindings.PROTOTYPES["vokra_model_attribution"][1][1] == ctypes.POINTER(
        ctypes.c_char
    )


def test_generated_struct_layout_matches_c_abi_contract() -> None:
    assert bindings.vokra_event_t._fields_ == [
        ("kind", ctypes.c_int),
        ("a", ctypes.c_uint32),
        ("b", ctypes.c_float),
    ]
    assert ctypes.sizeof(bindings.vokra_event_t) == 12
    assert [getattr(bindings.vokra_event_t, name).offset for name in ("kind", "a", "b")] == [
        0,
        4,
        8,
    ]

    assert [name for name, _ in bindings.vokra_aec_config_t._fields_] == [
        "sample_rate",
        "frame_size",
        "filter_length",
        "ref_queue_capacity_samples",
    ]
    if ctypes.sizeof(ctypes.c_size_t) == 8:
        assert ctypes.sizeof(bindings.vokra_aec_config_t) == 32
        expected_offsets = [0, 8, 16, 24]
    else:
        assert ctypes.sizeof(bindings.vokra_aec_config_t) == 16
        expected_offsets = [0, 4, 8, 12]
    assert [
        getattr(bindings.vokra_aec_config_t, name).offset
        for name, _ in bindings.vokra_aec_config_t._fields_
    ] == expected_offsets


def test_all_seven_opaque_handles_are_void_p_aliases() -> None:
    names = generator.parse_opaque_structs(_header_source())
    assert names == [
        "vokra_aec_ref_writer_t",
        "vokra_aec_t",
        "vokra_s2s_duplex_t",
        "vokra_s2s_interrupt_t",
        "vokra_session_options_t",
        "vokra_session_t",
        "vokra_stream_t",
    ]
    for name in names:
        assert getattr(bindings, name) is ctypes.c_void_p


def test_unknown_c_type_fails_closed() -> None:
    with pytest.raises(SystemExit, match="unmapped C type 'future_scalar_t'"):
        generator.map_c_type("future_scalar_t")


def test_unrecognised_future_declaration_cannot_silently_shrink_output() -> None:
    source = """
    typedef enum vokra_status_t { VOKRA_OK = 0 } vokra_status_t;
    unsigned long vokra_future_counter(void);
    """
    assert generator.discover_function_names(source) == ["vokra_future_counter"]
    with pytest.raises(SystemExit, match="function parser coverage mismatch"):
        generator.emit(source)


class _FakeFunction:
    restype = object()
    argtypes = object()


class _CompleteFakeLibrary:
    def __init__(self) -> None:
        for name in bindings.PROTOTYPES:
            setattr(self, name, _FakeFunction())


def test_bind_attaches_every_prototype_and_rejects_version_skew() -> None:
    lib = _CompleteFakeLibrary()
    assert bindings.bind(lib) is lib
    for name, (restype, argtypes) in bindings.PROTOTYPES.items():
        symbol = getattr(lib, name)
        assert symbol.restype is restype
        assert symbol.argtypes == list(argtypes)

    delattr(lib, "vokra_backend_available")
    with pytest.raises(AttributeError, match="vokra_backend_available"):
        bindings.bind(lib)
