# WeSpeaker TorchAudio archive facts — 2026-10-14

This is Mac-side static inspection of one small software distribution from
the current WeSpeaker lock. No installation, target import, native execution,
model download, Cargo, provider allocation or owner approval occurred.
The current 14-row dependency and five-component review remain pending.
These facts do not resolve the [setuptools finding](wespeaker-setuptools84-static-finding-2026-10-14.md),
the legacy public artifact, CPU parity or Apple gates.

## Exact primary archive and retained bytes

The locked registry is `download.pytorch.org/whl/cpu`, but the exact locked
[wheel URL](https://download-r2.pytorch.org/whl/cpu/torchaudio-2.11.0%2Bcpu-cp312-cp312-manylinux_2_28_x86_64.whl)
uses PyTorch's `download-r2.pytorch.org`. The manager's initial host constraint
named only the registry host; the worker used the exact locked distribution
host. The manager reconciled that distinction against the lock and independently
verified the received archive. No credential, query, proxy or redirect was used;
the correction made no additional network request or host substitution.

| Retained input | Bytes | SHA-256 |
|---|---:|---|
| Linux x86_64 CPython 3.12 TorchAudio 2.11.0+cpu wheel | 341,338 | `2354248848d06a9ae1e7a12165f800f0dda7df60ecac9fca892322b722b922c0` |
| Distribution-owned LICENSE | 1,338 | `93a58861a858cc108e6b6b833e08e76e8b2a66339e4a8007c8a5a8c1ff9c40d6` |
| Distribution-owned METADATA | 6,865 | `e77a84f87ce319a673f35dccb5658112ed7e5c1465e8a10ddc947f0136ed2d7c` |

The actual LICENSE is BSD 2-Clause; metadata is only corroboration.
The archive has 104 ZIP entries, including 85 files and 19 directories,
and 1,293,136 uncompressed bytes. Its 85-row RECORD covers every file;
the manager independently reproduced all non-RECORD hashes/sizes and the
blank self-row. Raw LICENSE bytes equal the retained copy. No NOTICE member
is present. No model weight was acquired or present in this inventory.

## Rejected parser result and corrected native facts

The initial temporary parser omitted `e_entry` from its ELF header unpack.
It consequently reported 56 program headers and no dynamic linkage. The
manager rejected that result by reading the original header bytes at their
specified offsets. The original report/inventory are retained, explicitly
superseded; they are not rewritten into a clean native verdict.

Both native members instead have `e_phoff=64`, `e_phentsize=56`, `e_phnum=9`
and PT_DYNAMIC. Independent byte-level decoding finds the same eight
DT_NEEDED names in both: `libc10.so`, `libtorch.so`, `libtorch_cpu.so`,
`libstdc++.so.6`, `libm.so.6`, `libgcc_s.so.1`, `libpthread.so.0`, `libc.so.6`.
This does not authenticate those dependencies' resolved runtime payloads or
prove Linux ABI/API compatibility from a Mac.

The corrected collector includes the full ELF32/64 header, rejects invalid
program/segment/dynamic/string-table bounds, rejects unsafe/duplicate/
encrypted/symlink/oversized ZIP inputs before payload reads, and has a shared
90-second POSIX alarm. Known ELF fixtures and malformed-input tests are
separate from the actual-wheel evidence. Neither kind supplies a model,
whole-closure, native-runtime or owner decision.

Manager independently passed the frozen ELF/ZIP fixtures through offline
UV/Python 3.12 and reran the collector into a separate output directory.
Archive, member, RECORD, payload and native facts matched the corrected
receipt; elapsed-time fields are not treated as deterministic identity.
Collector/test SHA-256 values are
`d22b0dfaa13aa94284704427b3c5e9f568f27e7c6736609d0e794ebb6b947aed` /
`a518ca300326a101593e06ceb7caaa5e52308945c4b713b10693d72255f1657a`.

Small archive, raw payloads, old rejected results and corrected results remain
under `/private/tmp/vokra-torchaudio211-software-audit-utv7H4`.
No WeSpeaker manifest, approval field or protected CosyVoice2 input was changed.
