#!/usr/bin/env -S uv run --frozen --project tools/parity --python 3.12 python
"""VAST-only FireRedASR-AED-L inspection with no checkpoint conversion."""
from __future__ import annotations
import argparse, hashlib, io, json, os, platform, re, subprocess, sys, tarfile, tempfile, zipfile
from pathlib import Path, PurePosixPath
from typing import Any

REPOSITORY = "FireRedTeam/FireRedASR-AED-L"
REVISION = "e57f5960d03cff1071ff7acbb409314d1e70ed3d"
SOURCE_URL = "https://github.com/FireRedTeam/FireRedASR.git"
SOURCE_REVISION = "834635e4cf277ed8ca92049fc375b17c3dc20748"
MODEL_LICENSE = "Apache-2.0"
MODEL_FILES = {".gitattributes", "README.md", "cmvn.ark", "cmvn.txt", "config.yaml", "dict.txt", "model.pth.tar", "train_bpe1000.model"}
DICT_SHA256 = "6907215aeb034f6926b26bf8abfd650f756781622480a2342ec1f29b2072cafe"
CMVN_SHA256 = "11816db612b43318ab01f9cfd05ee121dd3900b7a39d893f59d0104a06c199d2"
ARTIFACTS = {
    ".gitattributes": (1519, "a6344aac8c09253b3b630fb776ae94478aa0275b", None),
    "README.md": (6458, "5baa221616743b808a12ba7bfb25e8ba28e1689f", None),
    "cmvn.ark": (1311, "e26b4a310132492d05bfdd506ab3f6bbf89d9059", None),
    "cmvn.txt": (2985, "f425c7dec4fcb1a62ba57bb7c2de173fb4e47dce", None),
    "config.yaml": (0, "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391", None),
    "dict.txt": (71448, "afd1d79290c76a05e9eb653d76984434c18bb371", None),
    "model.pth.tar": (4678597714, "57ca39f0218ea04e2718c04e4496a5e6e6cca7b5", "12380d0b4b6b83b09306292f3ab7e276bc84e2feeec33ce956b1a488cd4867e3"),
    "train_bpe1000.model": (251707, "ab52c664b93b31764820ef193037a209a2089087", "473bbc157cb4eade2059b30a3c877a1c29bd50cadbfbed869ae36eeade7fee07"),
}
TOTAL_BYTES = sum(v[0] for v in ARTIFACTS.values())
MAX_HEADER_BYTES = 64 * 1024 * 1024
SOURCE_ROLES = (
    "fireredasr/data/asr_feat.py", "fireredasr/data/token_dict.py",
    "fireredasr/models/fireredasr.py", "fireredasr/models/fireredasr_aed.py",
    "fireredasr/models/module/conformer_encoder.py",
    "fireredasr/models/module/transformer_decoder.py", "fireredasr/speech2text.py",
    "fireredasr/tokenizer/aed_tokenizer.py", "fireredasr/utils/param.py",
    "examples/inference_fireredasr_aed.sh", "LICENSE",
)
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
README_MARKERS = (
    "license: apache-2.0",
    "it utilizes an attention-based encoder-decoder (aed) architecture.",
    "beam_size", "nbest", "decode_max_len", "smoothing", "aed_length_penalty", "eos_penalty",
)
EXPECTED_UNSAFE_GLOBALS = ["argparse.Namespace"]


class UnsafeFixture:
    pass


def digest(path: Path, algorithm: str = "sha256") -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git_blob_sha1(path: Path) -> str:
    h = hashlib.sha1()
    h.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def digest_bytes(data: bytes, algorithm: str = "sha256") -> str:
    return hashlib.new(algorithm, data).hexdigest()


def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def safe_path(name: str) -> None:
    path = PurePosixPath(name)
    if not name or "\x00" in name or "\\" in name or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe path: {name!r}")


def local_files(root: Path) -> dict[str, Path]:
    root = root.resolve()
    result: dict[str, Path] = {}
    for item in root.rglob("*"):
        relative = item.relative_to(root).as_posix()
        if relative == ".cache" or relative.startswith(".cache/"):
            continue
        safe_path(relative)
        resolved = item.resolve(strict=False)
        if not str(resolved).startswith(str(root) + os.sep):
            raise ValueError(f"path escapes snapshot: {relative}")
        if item.is_symlink() and not resolved.is_file():
            raise ValueError(f"dangling/nonregular symlink: {relative}")
        if item.is_file():
            result[relative] = item
        elif not item.is_dir():
            raise ValueError(f"nonregular snapshot entry: {relative}")
    return result


def validate_server_tree(packet: Any, root: Path) -> dict[str, Any]:
    if not isinstance(packet, dict) or set(packet) != {"repository", "revision", "resolved_revision", "files"}:
        raise ValueError("server-tree envelope is not exact")
    if packet["repository"] != REPOSITORY or packet["revision"] != REVISION or packet["resolved_revision"] != REVISION:
        raise ValueError("server identity mismatch")
    remote: dict[str, dict[str, Any]] = {}
    if not isinstance(packet["files"], list):
        raise ValueError("server files is not a list")
    for entry in packet["files"]:
        if not isinstance(entry, dict) or set(entry) != {"path", "type", "size", "git_blob_sha1", "lfs_sha256"}:
            raise ValueError("server entry fields are not exact")
        path, kind, size, blob, lfs = (entry[key] for key in ("path", "type", "size", "git_blob_sha1", "lfs_sha256"))
        safe_path(path)
        if kind != "file" or not isinstance(size, int) or isinstance(size, bool) or size < 0 or not isinstance(blob, str) or not HEX40.fullmatch(blob) or path in remote:
            raise ValueError(f"invalid server identity: {path!r}")
        if lfs is not None and (not isinstance(lfs, str) or not HEX64.fullmatch(lfs)):
            raise ValueError(f"invalid LFS identity: {path!r}")
        remote[path] = entry
    local = local_files(root)
    if set(remote) != set(local):
        raise ValueError("server/local tree mismatch")
    for path, entry in remote.items():
        actual = digest(local[path]) if entry["lfs_sha256"] else git_blob_sha1(local[path])
        if local[path].stat().st_size != entry["size"] or actual != (entry["lfs_sha256"] or entry["git_blob_sha1"]):
            raise ValueError(f"content identity mismatch: {path}")
    return {"repository": REPOSITORY, "revision": REVISION, "files": sorted(remote.values(), key=lambda x: x["path"]), "identity": "LFS SHA256 plus Git blob SHA1"}


def validate_artifact_identity(name: str, path: Path, expected: tuple[int, str, str | None], server_entry: dict[str, Any]) -> dict[str, Any]:
    """Validate one artifact without treating an LFS payload as its Git blob."""
    size, expected_blob, expected_lfs = expected
    if server_entry.get("path") != name or server_entry.get("size") != size or server_entry.get("git_blob_sha1") != expected_blob or server_entry.get("lfs_sha256") != expected_lfs:
        raise ValueError(f"server artifact identity mismatch: {name}")
    if path.stat().st_size != size:
        raise ValueError(f"fixed artifact size mismatch: {name}")
    payload_sha = digest(path)
    if expected_lfs is None:
        if git_blob_sha1(path) != expected_blob:
            raise ValueError(f"fixed non-LFS Git blob mismatch: {name}")
    elif payload_sha != expected_lfs:
        raise ValueError(f"fixed LFS payload SHA-256 mismatch: {name}")
    return {"bytes": size, "git_blob_sha1": expected_blob, "lfs_sha256": expected_lfs, "sha256": payload_sha}

def archive_inventory(path: Path) -> dict[str, Any]:
    members: list[dict[str, Any]] = []
    seen: set[str] = set()
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            if len(archive.infolist()) > 100_000:
                raise ValueError("zip member count exceeds bound")
            total = 0
            for item in archive.infolist():
                safe_path(item.filename)
                if len(item.filename) > 4096 or item.flag_bits & 1:
                    raise ValueError("unsafe/encrypted zip member")
                if item.filename in seen or (item.is_dir() and not item.filename.endswith("/")):
                    raise ValueError("duplicate/invalid zip member")
                mode = (item.external_attr >> 16) & 0o170000
                if mode not in {0, 0o040000, 0o100000}:
                    raise ValueError("unsafe zip member type")
                total += item.file_size
                if total > 16 * 1024 * 1024 * 1024:
                    raise ValueError("zip uncompressed size exceeds bound")
                seen.add(item.filename)
                members.append({"name": item.filename, "type": "directory" if item.is_dir() else "file", "bytes": item.file_size})
    elif tarfile.is_tarfile(path):
        with tarfile.open(path, "r:*") as archive:
            total = 0
            for item in archive:
                safe_path(item.name)
                if len(seen) >= 100_000 or len(item.name) > 4096 or item.name in seen or not (item.isdir() or item.isfile()):
                    raise ValueError("duplicate/unsafe tar member")
                total += item.size
                if total > 16 * 1024 * 1024 * 1024:
                    raise ValueError("tar uncompressed size exceeds bound")
                seen.add(item.name)
                members.append({"name": item.name, "type": "directory" if item.isdir() else "file", "bytes": item.size})
    else:
        raise ValueError("checkpoint is not a recognized safe archive")
    return {"path": path.name, "members": members}

def summarize(value: Any, path: str = "$", state: dict[str, Any] | None = None, depth: int = 0) -> list[dict[str, Any]]:
    """Recursively inventory tensors and bounded JSON-safe checkpoint metadata."""
    if state is None:
        state = {"active": set(), "count": 0, "nonfinite": False}
    if depth > 32 or len(path) > 4096 or state["count"] >= 100_000:
        raise ValueError("checkpoint inventory depth/item bound exceeded")
    if isinstance(value, argparse.Namespace):
        identity = id(value)
        if identity in state["active"]:
            raise ValueError(f"checkpoint metadata cycle at {path}")
        state["active"].add(identity)
        rows = summarize(vars(value), path, state, depth + 1)
        state["active"].remove(identity)
        return rows
    if isinstance(value, (dict, list, tuple)):
        identity = id(value)
        if identity in state["active"]:
            raise ValueError(f"checkpoint metadata cycle at {path}")
        state["active"].add(identity)
        rows: list[dict[str, Any]] = []
        if isinstance(value, dict):
            for key, child in value.items():
                if not isinstance(key, str) or not key or len(key) > 4096:
                    raise ValueError(f"unsafe checkpoint metadata key at {path}")
                safe_path(key)
                rows.extend(summarize(child, f"{path}.{key}", state, depth + 1))
        else:
            for index, child in enumerate(value):
                rows.extend(summarize(child, f"{path}[{index}]", state, depth + 1))
        state["active"].remove(identity)
        return rows
    try:
        import torch
    except ImportError as error:
        raise ValueError("torch is required for strict checkpoint tensor inventory") from error
    if isinstance(value, torch.Tensor):
        state["count"] += 1
        row = {"path": path, "type": "tensor", "shape": list(value.shape), "dtype": str(value.dtype), "numel": int(value.numel())}
        if hasattr(value, "is_floating_point") and value.is_floating_point():
            row["finite"] = bool(torch.isfinite(value).all().item())
            state["nonfinite"] = state["nonfinite"] or (not row["finite"])
        return [row]
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, str) and len(value) > 4096:
            raise ValueError(f"checkpoint metadata string exceeds bound at {path}")
        state["count"] += 1
        return [{"path": path, "type": type(value).__name__, "value": value}]
    raise ValueError(f"unsupported checkpoint object at {path}: {type(value).__name__}")

def inspect_checkpoint(path: Path) -> dict[str, Any]:
    inventory = archive_inventory(path)
    try:
        import torch
        unsafe = list(torch.serialization.get_unsafe_globals_in_checkpoint(str(path)))
        if unsafe != EXPECTED_UNSAFE_GLOBALS:
            raise ValueError(f"unsafe globals are not the exact approved set: {unsafe!r}")
        with torch.serialization.safe_globals([argparse.Namespace]):
            object_loaded = torch.load(path, map_location="cpu", weights_only=True)
        state = {"active": set(), "count": 0, "nonfinite": False}
        inventory["object_inventory"] = summarize(object_loaded, state=state)
        inventory["nonfinite_tensor"] = state["nonfinite"]
    except Exception as error:
        raise ValueError(f"weights_only checkpoint inspection failed: {error}") from error
    return inventory

def source_identity(root: Path) -> dict[str, Any]:
    origin = subprocess.check_output(["git", "-C", str(root), "remote", "get-url", "origin"], text=True).strip()
    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"], text=True)
    if origin != SOURCE_URL or head != SOURCE_REVISION or status:
        raise ValueError("source origin/revision/clean identity mismatch")
    tracked = set(filter(None, subprocess.check_output(["git", "-C", str(root), "ls-files", "-z"], text=False).decode().split("\0")))
    roles = []
    for role in SOURCE_ROLES:
        if role not in tracked or not (root / role).is_file():
            raise ValueError(f"source role missing or untracked: {role}")
        roles.append({"path": role, "bytes": (root / role).stat().st_size, "sha256": digest(root / role)})
    licenses = []
    for relative in sorted(tracked):
        path = root / relative
        if path.is_file() and (path.name.upper().startswith("LICENSE") or path.name.upper() in {"NOTICE", "README", "README.MD", "PYPROJECT.TOML"}):
            licenses.append({"path": relative, "bytes": path.stat().st_size, "sha256": digest(path)})
    if not licenses:
        raise ValueError("source license evidence missing")
    return {"origin": origin, "revision": head, "roles": roles, "license_records": licenses, "code_license": "Apache-2.0 requires review", "dependencies": "separate license review required"}


def inspect_sentencepiece(path: Path) -> dict[str, Any]:
    import sentencepiece as spm

    processor = spm.SentencePieceProcessor(model_file=str(path))
    first = [processor.id_to_piece(index) for index in range(20)]
    last = [processor.id_to_piece(index) for index in range(990, 1000)]
    expected_first = ["<unk>", "<s>", "</s>", "▁T", "HE", "▁A", "▁THE", "IN", "▁S", "▁W", "▁O", "RE", "ND", "▁B", "▁H", "ER", "▁M", "▁I", "OU", "▁C"]
    if processor.get_piece_size() != 1000 or processor.unk_id() != 0 or processor.bos_id() != 1 or processor.eos_id() != 2 or processor.pad_id() != -1 or first != expected_first or last != ["G", "Y", "P", "B", "V", "K", "'", "X", "J", "Q"]:
        raise ValueError("train_bpe1000 SentencePiece structure mismatch")
    return {"bytes": path.stat().st_size, "sha256": digest(path), "piece_count": 1000, "unk_id": processor.unk_id(), "bos_id": processor.bos_id(), "eos_id": processor.eos_id(), "pad_id": processor.pad_id(), "first_20": first, "last_10": last}


def inspect_dict(path: Path) -> dict[str, Any]:
    if digest(path) != DICT_SHA256:
        raise ValueError("dict.txt SHA256 mismatch")
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) != 7832 or any(not line for line in lines):
        raise ValueError("dict.txt must contain exactly 7832 non-empty lines")
    tokens: list[str] = []
    ids: list[int] = []
    for line in lines:
        fields = line.split()
        if len(fields) != 2 or not fields[1].isdigit():
            raise ValueError("dict.txt line is not token/id")
        tokens.append(fields[0]); ids.append(int(fields[1]))
    if len(set(tokens)) != len(tokens) or ids != list(range(7832)) or list(zip(tokens[:5], ids[:5])) != [("<blank>", 0), ("<unk>", 1), ("<pad>", 2), ("<sos>", 3), ("<eos>", 4)] or list(zip(tokens[-3:], ids[-3:])) != [("龟", 7829), ("龠", 7830), ("龢", 7831)]:
        raise ValueError("dict.txt token/id structure mismatch")
    return {"bytes": path.stat().st_size, "sha256": digest(path), "lines": len(lines), "first": list(zip(tokens[:5], ids[:5])), "last": list(zip(tokens[-3:], ids[-3:]))}


def inspect_cmvn(path: Path) -> dict[str, Any]:
    if digest(path) != CMVN_SHA256:
        raise ValueError("cmvn.txt SHA256 mismatch")
    return parse_cmvn(path.read_text(encoding="ascii"), path.stat().st_size, digest(path))


def parse_cmvn(raw: str, size: int, sha: str) -> dict[str, Any]:
    lines = raw.splitlines()
    if len(lines) != 3 or lines[0].strip() != "[" or lines[2].strip()[-1:] != "]":
        raise ValueError("cmvn.txt requires bracketed three-line matrix")
    rows = []
    for index, line in enumerate(lines[1:]):
        fields = (line[:-1] if index == 1 else line).split()
        if len(fields) != 81:
            raise ValueError("cmvn.txt requires two rows of 81 values")
        rows.append([float(item) for item in fields])
    if len(rows) != 2 or not all(value == value and abs(value) != float("inf") for row in rows for value in row) or rows[0][-1] != 1183022220.0 or rows[1][-1] != 0.0:
        raise ValueError("cmvn.txt numeric structure mismatch")
    return {"bytes": size, "sha256": sha, "rows": 2, "columns": 81, "count": rows[0][-1], "terminal_second": rows[1][-1]}


def require_readme_markers(card: str) -> tuple[str, ...]:
    missing = [marker for marker in README_MARKERS if marker not in card.lower()]
    if missing:
        raise ValueError(f"model card markers missing: {missing}")
    return README_MARKERS

def base_manifest() -> dict[str, Any]:
    return {"format": "vokra-firered-asr-aed-l-inspection-v1", "status": "BLOCKED", "inspection_status": "PENDING", "evidence_stage": "INSPECTION_ONLY", "runtime_status": "NOT_IMPLEMENTED_FAIL_CLOSED", "cpu_status": "UNSUPPORTED", "metal_status": "BLOCKED_BY_CPU", "parity_status": "NOT_RUN", "publication": "NO_UPLOAD", "environment": {"python": sys.version, "platform": platform.platform()}, "model": {"repository": REPOSITORY, "revision": REVISION, "license": MODEL_LICENSE, "files": ARTIFACTS, "total_bytes": TOTAL_BYTES}, "source": {"origin": SOURCE_URL, "revision": SOURCE_REVISION}, "blockers": ["checkpoint container/config/tensor manifest requires authenticated review", "frontend/CMVN and Conformer/AED decoder contract requires review", "official beam search/tokenizer rendering requires review", "independent CPU numerical parity is not run", "complete Metal graph is not implemented", "training data and dependency provenance require review"]}

def inspect(args: argparse.Namespace) -> int:
    manifest = base_manifest()
    try:
        snapshot = Path(args.snapshot)
        packet = json.loads(Path(args.server_tree).read_text(encoding="utf-8"), object_pairs_hook=unique_pairs)
        manifest["server_tree"] = validate_server_tree(packet, snapshot)
        files = local_files(snapshot)
        if set(files) != MODEL_FILES or sum(path.stat().st_size for path in files.values()) != TOTAL_BYTES:
            raise ValueError("HF exact file set/total mismatch")
        readme = files["README.md"]
        card = readme.read_text(encoding="utf-8").lower()
        markers = require_readme_markers(card)
        manifest["model_card"] = {"path": "README.md", "bytes": readme.stat().st_size, "sha256": digest(readme), "markers": markers}
        manifest["artifacts"] = {}
        server_artifacts = {entry["path"]: entry for entry in manifest["server_tree"]["files"]}
        for name, (size, blob, lfs) in ARTIFACTS.items():
            path = files[name]
            manifest["artifacts"][name] = validate_artifact_identity(name, path, (size, blob, lfs), server_artifacts.get(name, {}))
        manifest["artifacts"]["train_bpe1000.model"].update({"structure": inspect_sentencepiece(files["train_bpe1000.model"]), "status": "STRUCTURE_AUTHENTICATED"})
        manifest["artifacts"]["dict.txt"].update({"structure": inspect_dict(files["dict.txt"])})
        manifest["artifacts"]["cmvn.txt"].update({"structure": inspect_cmvn(files["cmvn.txt"]), "status": "STRUCTURE_AUTHENTICATED"})
        if files["config.yaml"].stat().st_size != 0:
            raise ValueError("config.yaml must be the authenticated empty file")
        manifest["structures"] = {"dict.txt": {"bytes": files["dict.txt"].stat().st_size, "sha256": digest(files["dict.txt"]), "status": "STRUCTURE_AUTHENTICATED"}, "cmvn.ark": {"bytes": files["cmvn.ark"].stat().st_size, "sha256": digest(files["cmvn.ark"]), "status": "STRUCTURAL_REVIEW_REQUIRED"}, "cmvn.txt": {"bytes": files["cmvn.txt"].stat().st_size, "sha256": digest(files["cmvn.txt"]), "status": "STRUCTURE_AUTHENTICATED"}, "config_yaml": {"bytes": 0, "sha256": digest(files["config.yaml"]), "status": "BLOCKER_EMPTY_CONFIG"}, "tokenizer": {"bytes": files["train_bpe1000.model"].stat().st_size, "sha256": digest(files["train_bpe1000.model"]), "status": "STRUCTURE_AUTHENTICATED"}}
        manifest["checkpoint"] = inspect_checkpoint(files["model.pth.tar"])
        if manifest["checkpoint"].get("nonfinite_tensor"):
            raise ValueError("checkpoint contains non-finite floating tensor")
        if not args.source:
            raise ValueError("fixed source checkout is required")
        manifest["official_source"] = source_identity(Path(args.source))
        manifest["inspection_status"] = "AUTHENTICATED_EVIDENCE_COMPLETE"
    except Exception as error:
        manifest["inspection_status"] = "INSPECTION_ERROR"
        manifest.setdefault("blockers", []).append(f"inspection error: {type(error).__name__}: {error}")
    Path(args.evidence).mkdir(parents=True, exist_ok=True)
    (Path(args.evidence) / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return 2

def self_test() -> None:
    positive_card = "\n".join(README_MARKERS)
    for bad in ("../x", "/x", "a\\b", "a\x00b", ""):
        try: safe_path(bad)
        except ValueError: pass
        else: raise AssertionError(f"unsafe path accepted: {bad!r}")
    with tempfile.TemporaryDirectory(prefix="firered-inspect-") as directory:
        root = Path(directory) / "snapshot"; root.mkdir()
        readme_fixture = Path(directory) / "README.md"
        readme_fixture.write_text(positive_card, encoding="utf-8")
        assert require_readme_markers(readme_fixture.read_text(encoding="utf-8")) == README_MARKERS
        for invalid_phrase in (
            "attention encoder decoder",
            "it utilizes an attention-based encoder decoder (aed) architecture.",
            "it utilizes an attention-based encoder-decoder (llm) architecture.",
        ):
            readme_fixture.write_text(positive_card.replace(README_MARKERS[1], invalid_phrase), encoding="utf-8")
            try:
                require_readme_markers(readme_fixture.read_text(encoding="utf-8"))
            except ValueError:
                pass
            else:
                raise AssertionError("README fixture accepted an invalid AED marker")
        payload = root / "x"; payload.write_text("x", encoding="utf-8")
        (root / ".cache").mkdir(); (root / ".cache" / "ignored.json").write_text("{}", encoding="utf-8")
        assert set(local_files(root)) == {"x"}
        packet = {"repository": REPOSITORY, "revision": REVISION, "resolved_revision": REVISION, "files": [{"path": "x", "type": "file", "size": 1, "git_blob_sha1": "1" * 40, "lfs_sha256": digest(payload)}]}
        validate_server_tree(packet, root)
        lfs_identity = validate_artifact_identity("x", payload, (1, "1" * 40, digest(payload)), packet["files"][0])
        assert lfs_identity["git_blob_sha1"] == "1" * 40 and lfs_identity["lfs_sha256"] == digest(payload)
        try:
            validate_artifact_identity("x", payload, (1, git_blob_sha1(payload), digest(payload)), packet["files"][0])
        except ValueError:
            pass
        else:
            raise AssertionError("LFS payload Git blob was accepted as the authenticated pointer blob")
        for malformed in (dict(packet, files=[dict(packet["files"][0], lfs_sha256="bad")]), dict(packet, files=[]), dict(packet, repository="wrong/repo")):
            try: validate_server_tree(malformed, root)
            except ValueError: pass
            else: raise AssertionError("malformed server packet accepted")
        archive = Path(directory) / "unsafe.tar"
        with tarfile.open(archive, "w") as tar:
            info = tarfile.TarInfo("../escape"); info.size = 1; tar.addfile(info, io.BytesIO(b"x"))
        try: archive_inventory(archive)
        except ValueError: pass
        else: raise AssertionError("traversal archive accepted")
        import torch
        nested = {"outer": [{"weights": torch.tensor([1.0, 2.0]), "meta": ("ok", 3)}]}
        rows = summarize(nested)
        assert {row["path"] for row in rows} >= {"$.outer[0].weights", "$.outer[0].meta[0]", "$.outer[0].meta[1]"}
        namespace = argparse.Namespace(meta="ok")
        namespace_rows = summarize(argparse.Namespace(payload=namespace))
        assert any(row["path"] == "$.payload.meta" and row["value"] == "ok" for row in namespace_rows)
        unsafe_namespace = argparse.Namespace(**{"../bad": 1})
        try: summarize(unsafe_namespace)
        except ValueError: pass
        else: raise AssertionError("unsafe Namespace attribute accepted")
        cyclic_namespace = argparse.Namespace(); cyclic_namespace.self = cyclic_namespace
        try: summarize(cyclic_namespace)
        except ValueError: pass
        else: raise AssertionError("cyclic Namespace accepted")
        finite_state = {"active": set(), "count": 0, "nonfinite": False}
        nonfinite = summarize({"bad": torch.tensor([float("nan")]), "good": torch.tensor([1.0])}, state=finite_state)
        assert nonfinite[0]["finite"] is False and finite_state["nonfinite"] is True
        cycle: list[Any] = []; cycle.append(cycle)
        for bad in (cycle, {"../bad": 1}, object(), "x" * 4097, list(range(100001))):
            try: summarize(bad)
            except ValueError: pass
            else: raise AssertionError("unsafe/bounded metadata accepted")
        deep: Any = 0
        for _ in range(34): deep = [deep]
        try: summarize(deep)
        except ValueError: pass
        else: raise AssertionError("deep metadata accepted")
        cmvn_positive = "[\n" + " ".join(["0"] * 80 + ["1183022220"]) + "\n" + " ".join(["0"] * 81) + "]\n"
        assert parse_cmvn(cmvn_positive, len(cmvn_positive), "synthetic") ["rows"] == 2
        try: parse_cmvn("[\n0\n0\n]\n", 9, "synthetic")
        except ValueError: pass
        else: raise AssertionError("malformed CMVN accepted")
        namespace_checkpoint = Path(directory) / "namespace.pth.tar"
        torch.save({"namespace": argparse.Namespace(label="fixture"), "tensor": torch.tensor([1.0])}, namespace_checkpoint)
        checkpoint_evidence = inspect_checkpoint(namespace_checkpoint)
        assert any(row["path"] == "$.namespace.label" and row["value"] == "fixture" for row in checkpoint_evidence["object_inventory"])
        unsafe_checkpoint = Path(directory) / "unsafe-global.pth.tar"
        torch.save(UnsafeFixture(), unsafe_checkpoint)
        try: inspect_checkpoint(unsafe_checkpoint)
        except ValueError as error: assert "exact approved set" in str(error)
        else: raise AssertionError("unknown checkpoint global accepted")
        manifest_dir = Path(directory) / "evidence"
        rc = inspect(argparse.Namespace(snapshot=str(root / "missing"), server_tree=str(root / "missing.json"), source=None, evidence=str(manifest_dir)))
        assert rc == 2
        manifest = json.loads((manifest_dir / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["status"] == "BLOCKED" and manifest["evidence_stage"] == "INSPECTION_ONLY" and manifest["inspection_status"] == "INSPECTION_ERROR"
        assert "AUTHENTICATED_EVIDENCE_COMPLETE" not in json.dumps(manifest)
    print("firered inspector self-test PASS")

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--snapshot")
    parser.add_argument("--server-tree")
    parser.add_argument("--source")
    parser.add_argument("--evidence", default="evidence")
    args = parser.parse_args()
    if args.self_test:
        self_test(); return 0
    if not args.snapshot or not args.server_tree:
        parser.error("--snapshot and --server-tree required")
    return inspect(args)

if __name__ == "__main__":
    raise SystemExit(main())
