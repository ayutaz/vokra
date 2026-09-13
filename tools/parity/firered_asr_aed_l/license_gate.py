#!/usr/bin/env python3
"""Offline, fail-closed FireRedASR dependency license evidence gate.

The VAST packet is an evidence record, not an owner approval.  This gate
binds the checked-in Linux closure to the packet's exact row and payload
digests, makes the native/bundled review disposition explicit, and keeps
publication disabled until a separately supplied owner/legal decision exists.
It never imports the reference environment or accesses a checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tempfile
from pathlib import Path
from typing import Any
import tomllib

GATE_VERSION = 1
HEX64 = re.compile(r"^[0-9a-f]{64}$")
HEX40 = re.compile(r"^[0-9a-f]{40}$")
AUDIT_JSON_SHA256 = "430acd2ec0ccd4c4122fc9638ce3251017b4c57c99a9f3586668207a22a87d8b"
AUDIT_ARCHIVE_SHA256 = "580ee2b1750a7839bb2e9b57925a7e34ad33ecce4113d931891678bf09af7ad2"
AUDIT_EXPECTED_HEAD = "77984f048069bd9130f8f414a4dd915595a3aee8"
AUDIT_FORMAT = "vokra-firered-asr-aed-l-dependency-audit-only-v1"
AUDIT_SCOPE = {
    "active_closure_sha256": "b79e93fabc422b5b9a1c4347402829ad46d2e82afc342e7cf5125f50286e768c",
    "distribution_evidence_sha256": "b6c341bb6e9244936ee6339bd5101ad3441bd38d8a71972685546026371a57ae",
    "expected_head": AUDIT_EXPECTED_HEAD,
    "fire_red_source_file_aggregate_sha256": "6024e41faa03f4981215024add740b65b03bb6ee4f336b373a062fa4d8a88a01",
    "fire_red_source_identity_sha256": "a93d202f09efe7a59686850413186eb26546e80d5fddcc29b8fedd958eab720c",
    "license_candidate_aggregate_sha256": "fb0ed78afaba8614a04b3b2893915e71cef07ae3c23b8591244ff90052ec659c",
    "lock_sha256": "d129bd9f12fcee2083243ce2312eff1a6c4a74ab2845b2ef069f60454709c1eb",
    "native_payload_aggregate_sha256": "d74c080e24b13f1b0483e7c35ee3dfbe18be972e4d0df008f5146c6a8bdb9931",
    "publisher_url_aggregate_sha256": "b8fca4c78d182abe827b3bd006b38e801cc31df3d02cf7793a4dea9809c1ee1f",
    "review_ledger_sha256": "61ba58e8da8a8d2279b36ea560ffd442bae0ed79ec6dae99eb0cdc1c1e4c9763",
}
SOURCE_FILES = {
    "LICENSE": (11357, "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4", "261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64"),
    "README.md": (7183, "4d9f96f6485690ad598b63cb10aa9b2ff44aec7975036016c18f932772014b08", "50102dc79e5c43dd46c2c4503ae273714e2dc6a0"),
    "pretrained_models/README.md": (28, "b6f4b724759dd430194d7e53e95034a9774439b89518462d1331e02a18be0c60", "7d8271f29b961be90788f7fe9e0364f76f74e8f1"),
    "runtime/triton_tensorrt/README.md": (2192, "0ccee4a04fa95031bf94fb28ef41d4db3d82551e7ecdb82cc4c92ac1fa595e70", "c8246cb38f9928068580470e89fb6f4e987ff97c"),
}
EXPECTED_ROWS = {
    "anyio@4.14.2": ("c5e6fc4dfd833319913b287be7a30d9935e9208c7ef917630621855aaac4d419", "07481299ccb68fc1b3b187a4feb88b4ad5274a8bd750fa86cdfb1b74c08f423e", "cabd19d24c9b440ecea88f641363bc9e84a3018e73c4de9b4ccf2f81b22b4141", "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945", 1, 0),
    "certifi@2026.7.22": ("ef5af1638fbb23676ac3c5777dfcfc2cd9c348fe4172ed5ba3d277655b248090", "b2bbf2f4cdfa818d8f7390bc3b8731be5a69f4ff005389a007ae4971bc6bc8f4", "c215f1219047abb12f714504c9893a95da2b47ce548f3d120f567d94d79de36f", "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945", 1, 0),
    "click@8.5.0": ("e87bce0bd194de70dfb8e708e0a6b0483009f25611804eaf87bfeb63e6c20501", "4b88c6d7d899c489183e2907b9631bf25abb91609eeeb29d32d48777d0809c6d", "fbfeb7055cdb6afca2c34fe89346f34342629cd9a92637dcf4e3e15445cd1772", "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945", 1, 0),
    "filelock@3.32.5": ("7886b747148c0aa7889c4ec33f3529b8c21f15142063b5e1ddbb96a8a5442724", "dbc2e6ca1ba0032757482f682420350e1045a7ea1ff404eebaa9a7d64f39dbc9", "c31636861fb7620e54ed75c2d9c16a50c3c5dfaae2247cd8e63c43d29c2312ae", "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945", 1, 0),
    "fsspec@2026.7.0": ("88b1657585de14e3bb82aa5278aa9595b30aa4ca74ead073d8950615c808099d", "effb442941dc49840c73cf8fbc39242da182eebc84d137fea71ba2752aed7d8b", "a5ee9839a485fbff00b31fcca372cd3949dec9e8742a5e294b9eb140ec97019c", "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945", 1, 0),
    "h11@0.16.0": ("28f326098ac09fcba79b8f180f96087c841fe2456215cb7b872a9c7d5cd19d64", "563d7a22dca381d7d09455b88a2775313747e5126ad350459d08d795a86cb548", "541879227b5c677f5bb765a9f4dd46ed72a223513f25d684ba7b1d43f0734b3a", "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945", 1, 0),
    "hf-xet@1.6.0": ("becdd6ec73b515dc38c59501956134eb0b903bfed33ffcd99b252361acd926e9", "c823b489ae28bbcf3934d8ffa0f2c1fc8d0632722c87281664a3ff09626d5e09", "c1a797527646a5dcee18317a9ce2054094a0b90948b104834a40ca9d189d2a26", "5dada5221e396b92528fc725d83f7a697ae287ae4cec0bee5d121e7461419343", 1, 1),
    "httpcore@1.0.9": ("fe2d4fda6199128978779e0cf27f3c045c531863fcdd987eacc74f3a18d41c21", "53fc85b65ea904a873c7e9bfa00e354161a1e3423ade987c5fc61e4d98c045dd", "2d95736e4a15999af9a65d179913b30a8280c0ccbc18a3a0a9b469c1d97147be", "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945", 1, 0),
    "httpx@0.28.1": ("febb9b0f8f3e80d57c8199c304f35c4336e8581d1d18d7983c92766b82793b25", "0d7591290b39cd7e83becc6f873c86a9d50a58cb05cbac867ed3883c8cdb4c12", "bd1655c5a3880ee41bf2556bccc94916a2709cdab8618b0c9176742cce3f5c98", "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945", 1, 0),
    "huggingface-hub@1.29.0": ("60b2bcb58f6d852eaa21eeae6f0030354278ecab9620dfe83e350a93a268e647", "532e97f4d9f0a5efba7c44f6f903518de7f189d43f7a9b78f744da6a2e2d6707", "a3058943e193c0ff5d4718039771a68a791a3eeadaca6c1f7476a1ae37d8946e", "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945", 1, 0),
    "idna@3.19": ("4d113161aca8582e8d28fddf3ea50f19b607209c2b3e4364379a163454680084", "c43b115e0ccb58422192084665b9b1ca7fc7d05e0a490d4200a31e3f09a5cffa", "2424cf63ecee2cfad4c8b502cd82fc4cf98c83f795abe1f3bc425a7bef3359cd", "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945", 1, 0),
    "jinja2@3.1.6": ("68c5548fb67c4132a13898d9b31ec50c6bea2abdd915d921f214355c3a6499c8", "50ed16eba7c82c455d346154f7151f0ab80c2ab2d875b356c094de8d76ff633a", "7797c297e57460b4d9e0b3f300f0fb8568fc1a4813a237ef973f724560934005", "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945", 1, 0),
    "kaldi-native-fbank@1.15": ("6c43dacd8008fb7613b54a215d9d419e479c3af16512f10de4bd9399cd9b0cd5", "d85b541ba46ab96cfc99b315ef2c16845c2a4c043b366a6b9cf773f8790aba4e", "8ba1524ddb92d8f79c5fe12cffe4de154593db5dce107a2004e866f60ed2bb99", "2f711484d299a7426dc0c2f1d9dc81e713a7f30d0d77759d7d1a42740a78afae", 1, 2),
    "kaldiio@2.18.1": ("654f114c1b49054c265e4e2e894f4c4489ee3b25430fa5e549d155ef3b7d24e9", "d23365b84af119c090d763c4c127063bf870786864f2bcc31c554460fb333ff0", "359f2884ee5c5d9c9a226155a808c7e14688c4fcb8d20b173d455a77a77a09be", "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945", 1, 0),
    "markupsafe@3.0.3": ("12b4cc61a7fa288cf7667ee3f213786d9619db57fb33ff6f934afbcb5c12ec81", "58ba72ec0be90a65fe40fed7cc7bc3aa97e4a7652f4afe6bd1865f678a3259bb", "ed8165a9a55f926b37bd170dd2c335b4f43f09eccd2e57a347c9adca9648eaa2", "7be2537dd359d998ae1416ab59c1a42e11b5aca5644f8a73038dd7ccf94fa25b", 1, 1),
    "mpmath@1.3.0": ("44b66ea444b9c0d19ae94815d356bf047ae6b680c19268b5c265687cd6a81406", "3f4cf7d80b532e8df98ce08d418d259e0c2a12358344c8b6efdbb1ad72f8eda4", "c671c0bb6ffc4784dfe3d5bac6ec9172e85d436d39b7fd2322ce83bc092cc277", "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945", 1, 0),
    "networkx@3.6.1": ("aca5d94a97d1f70f301d033addb635f6e66be8973a00aba4127637eed2ef316a", "44f9410ccae91642b32ab65b7e76d4ba6c409d1c9792d51ea6d520207df19150", "9cd3e804f3195a590587e138de33c2b6856d7428cfd49f2c9f045ce2852f211b", "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945", 1, 0),
    "numpy@2.5.2": ("9a895d801184d2176f926af84eb47f2db1af2756afac99d64a2bf482d5d5e024", "b49a7d03d1fecdd4a60c8d1c0f688a0a397669435cf88588254617ed4255d4a8", "206a52e979df3f56250d2ebe3db84a2887392411953b41fed852eec9dc258941", "394de51c30518b9138df9c6f0e1ebb060746e9cd9d30c4872c63568d3538bc14", 20, 22),
    "packaging@26.3": ("70fdb89fc4d4a9a043bf7372b8972bcc883fddff34ab55e9cf80d73875384763", "248a43c2166bcda0ea35c28da078bdcad8d6408959adfecbc96b646adbfd42de", "73b85014087e44fdf55adf7cb0dacaa2f21dd05a861d9634d5f6b2f67adc2772", "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945", 3, 0),
    "pyyaml@6.0.3": ("03c3b415ed38d09faedc49360e930769b40c58f68585e6de00e2e4916a858d34", "895cdb36053935e1554285a05f1c00aa64fcfa11ccf655b22b85547236011a04", "d1ed007cf002c38c444532deab3307370818bd9ba5c8b7e6ea0f058d5539af84", "316aecde752257912d3d2e526fc9607d4181093f28503848f6a5ed6590947926", 1, 1),
    "sentencepiece@0.2.2": ("bb49c79797fff1b8f3048efd4849d86dce4adbe26f7f3f0a5b37239a59d88ee8", "fe5198ade54c50ccb562a898e349eeb17226dacbc15907644da4c983fa88c8c1", "80a2d2c2389f0b1126fe1496f5548521cf454b6c44d10e97341845ecb6c33ecc", "35517b1f80d8b18a139a63b635fd28b901477689062025178983220e5f58a7e4", 0, 1),
    "setuptools@83.0.0": ("26e45e90de763c6938d93d1df2ecd6782a0f1514d41cbd22ebef4e0ab6475771", "89cc0825a456acba9a13b7d462283c47fb3e80bce28201fbbf12846270d91181", "fe531270fdd6dfd7bbe53f295e79ce7a71883f922b3d7ba92202809bff60cd38", "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945", 17, 0),
    "sympy@1.14.0": ("b756c2fbfd5be05ac5bdb0ebca61f55618f30f633ed92d88a5687429313a7595", "1d8ab556984b60f3fc63de3c438b13757cd26c2c9c2cf5ebf2380ad49f673c23", "f4115bd154e9b52e3b180182538ce6d47090638c8756e0401e7b9dd6e761f6ab", "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945", 2, 0),
    "torch@2.13.0+cpu": ("b3c7f4b3f77d06f86a42aab7d0facef7b661ca3c22967bfcc396917d935dd335", "023431aa72a9f2ae9e34adb9622c34c825e87e7989070458c79b0ebb7efcf67c", "96828e0eb20116123d5c4907c46dedac5e9aa0210d290efd37010b253a9ae008", "607e09390e289f6db7751a77e1fb4180864d580e4a49d06870ac5a2b6490db3c", 98, 136),
    "tqdm@4.70.0": ("0d95b85b90428f8776afc4a92b17c5094f78ee98d150b3255acbcdf4e9d57941", "24b263a42a27d98d03f2b9836fa287ed262ea6ef800136184770a89bc9df10eb", "bc9fd2f28012018cf50e8bf364acdd3e8bcce9d2dfe3438b87135c4586ae51f8", "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945", 0, 0),
    "typing-extensions@4.16.0": ("b05084ca1d50879865178d9fff9fabeab61bdfb1f361bfbde95421ffc8f9be46", "89c532549428fd4458cf984a459d8109710305118dd90ab8ce33c6c686751d4d", "59d7fa7deaa79203083dae2bbe0ff67e7f983bfea94420e8913edb2c9a63c976", "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945", 1, 0),
    "vokra-firered-asr-aed-l-reference@0.1.0": ("e39cdebb978dff399c76edad97f32f000e6f950c1cd27a4a35ecefd61579afc2", "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945", "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945", "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945", 0, 0),
}

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def canonical(value: Any) -> str:
    return sha256_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode())

def load_json(path: Path) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs)

def block(message: str) -> None:
    print(f"firered license gate: BLOCKED: {message}", file=sys.stderr)
    raise SystemExit(2)

def package_rows(lock: dict[str, Any]) -> list[dict[str, Any]]:
    packages = lock.get("package")
    if not isinstance(packages, list) or len(packages) != len(EXPECTED_ROWS):
        block("uv.lock active package count is not the reviewed 27-row closure")
    rows = []
    seen: set[tuple[str, str]] = set()
    for package in packages:
        if not isinstance(package, dict) or not isinstance(package.get("name"), str) or not isinstance(package.get("version"), str):
            block("uv.lock package identity is malformed")
        key = (package["name"], package["version"])
        if key in seen:
            block(f"uv.lock duplicate package identity: {key!r}")
        seen.add(key)
        source = package.get("source")
        if not isinstance(source, dict) or len(source) != 1 or set(source) not in ({"registry"}, {"git"}, {"virtual"}):
            block(f"uv.lock source identity is malformed: {key!r}")
        rows.append({"name": package["name"], "version": package["version"], "source": source, "dependencies": package.get("dependencies", []), "resolution_markers": package.get("resolution-markers")})
    rows.sort(key=lambda row: (row["name"], row["version"]))
    result = []
    for row in rows:
        result.append({**row, "row_sha256": canonical(row)})
    return result

def validate_manifest(lock_path: Path, project_path: Path, manifest_path: Path) -> dict[str, Any]:
    if any(path.is_symlink() or not path.is_file() for path in (lock_path, project_path, manifest_path)):
        block("uv.lock, pyproject.toml, or tracked license manifest is missing/non-regular")
    try:
        manifest = load_json(manifest_path)
        lock_bytes = lock_path.read_bytes()
        project_bytes = project_path.read_bytes()
        lock = tomllib.loads(lock_bytes.decode())
        project = tomllib.loads(project_bytes.decode())
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError, ValueError) as error:
        block(f"closure input is unreadable: {error}")
    expected_keys = {"gate_version", "lock_sha256", "project_sha256", "package_rows_sha256", "package_review_rows", "package_review_rows_sha256", "forbidden_dependencies", "identities", "license_rows", "license_rows_sha256", "audit_evidence", "approval_scope_sha256", "approval", "publication"}
    if not isinstance(manifest, dict) or set(manifest) != expected_keys or manifest.get("gate_version") != GATE_VERSION:
        block("gate manifest schema/version drifted")
    if sha256_bytes(lock_bytes) != manifest["lock_sha256"] or manifest["lock_sha256"] != AUDIT_SCOPE["lock_sha256"]:
        block("uv.lock digest is not the VAST-reviewed lock")
    if sha256_bytes(project_bytes) != manifest["project_sha256"]:
        block("pyproject.toml digest differs from the reviewed project")
    if project.get("project", {}).get("name") != "vokra-firered-asr-aed-l-reference" or project.get("project", {}).get("version") != "0.1.0":
        block("project identity drifted")
    rows = package_rows(lock)
    if canonical(rows) != manifest["package_rows_sha256"] or canonical(rows) != AUDIT_SCOPE["active_closure_sha256"]:
        block("locked package/source/dependency evidence digest drifted")
    review_rows = manifest["package_review_rows"]
    if not isinstance(review_rows, list) or len(review_rows) != len(rows) or canonical(review_rows) != manifest["package_review_rows_sha256"]:
        block("package review row set/hash is malformed")
    by_id = {f"{row['name']}@{row['version']}": row for row in rows}
    expected_review_ids = set(by_id)
    seen: set[str] = set()
    for review in review_rows:
        if not isinstance(review, dict) or set(review) != {"id", "status", "license", "native_bundled_review", "evidence"}:
            block("package review schema is malformed")
        ident = review["id"]
        if ident in seen or ident not in expected_review_ids:
            block(f"package review identity is not the exact lock row: {ident!r}")
        seen.add(ident)
        if review["status"] != "REVIEWED_FACTS_OWNER_APPROVAL_REQUIRED" or not isinstance(review["license"], str) or not review["license"].strip() or not isinstance(review["native_bundled_review"], str) or not review["native_bundled_review"].strip():
            block(f"package review is unresolved or falsely approved: {ident!r}")
        evidence = review["evidence"]
        if not isinstance(evidence, dict) or set(evidence) != {"row_sha256", "metadata_sha256", "publisher_urls_sha256", "license_candidates_sha256", "native_payloads_sha256", "publisher_file_count", "native_payload_count"}:
            block(f"package evidence schema is malformed: {ident!r}")
        expected = EXPECTED_ROWS.get(ident)
        if expected is None or tuple(evidence[field] for field in ("metadata_sha256", "publisher_urls_sha256", "license_candidates_sha256", "native_payloads_sha256", "publisher_file_count", "native_payload_count")) != expected[0:1] + expected[1:]:
            block(f"package evidence identity/hash drifted: {ident!r}")
        if evidence["row_sha256"] != by_id[ident]["row_sha256"]:
            block(f"package row SHA drifted: {ident!r}")
        if any(not isinstance(evidence[field], str) or not HEX64.fullmatch(evidence[field]) for field in ("row_sha256", "metadata_sha256", "publisher_urls_sha256", "license_candidates_sha256", "native_payloads_sha256")):
            block(f"package evidence digest is malformed: {ident!r}")
        if any(not isinstance(evidence[field], int) or evidence[field] < 0 for field in ("publisher_file_count", "native_payload_count")):
            block(f"package evidence count is malformed: {ident!r}")
    if seen != expected_review_ids:
        block("package review rows do not cover the exact closure")
    forbidden = manifest["forbidden_dependencies"]
    if not isinstance(forbidden, list) or set(forbidden) != {"librosa", "soxr", "soundfile", "triton", "nvidia-cuda"}:
        block("forbidden dependency policy drifted")
    if expected_review_ids & set(forbidden):
        block("forbidden dependency entered the reviewed closure")
    identities = manifest["identities"]
    if not isinstance(identities, dict) or identities.get("source_repository") != "https://github.com/FireRedTeam/FireRedASR.git" or identities.get("source_revision") != "834635e4cf277ed8ca92049fc375b17c3dc20748" or identities.get("model_repository") != "FireRedTeam/FireRedASR-AED-L" or identities.get("model_revision") != "e57f5960d03cff1071ff7acbb409314d1e70ed3d":
        block("FireRed source/model identity drifted")
    source_files = identities.get("source_files")
    if not isinstance(source_files, dict) or set(source_files) != set(SOURCE_FILES):
        block("authenticated source file set is incomplete")
    for name, expected in SOURCE_FILES.items():
        record = source_files[name]
        if not isinstance(record, dict) or tuple(record.get(key) for key in ("bytes", "sha256", "git_blob_oid")) != expected:
            block(f"source evidence drifted: {name}")
    if identities.get("model_acquisition") != "NOT_ACQUIRED" or identities.get("model_import") != "NOT_PERFORMED" or identities.get("execution") != "NOT_PERFORMED" or identities.get("training_provenance") != "OWNER_REVIEW_REQUIRED":
        block("model/training boundary is not fail-closed")
    evidence = manifest["audit_evidence"]
    expected_evidence = {"format": AUDIT_FORMAT, "expected_head": AUDIT_EXPECTED_HEAD, "archive_sha256": AUDIT_ARCHIVE_SHA256, "dependency_audit_json_sha256": AUDIT_JSON_SHA256, "distribution_rows": 27, "required_file_paths": 186, "unique_file_payloads": 135, "status": "BLOCKED_UNREVIEWED_TRANSITIVE", "publication": "NO_UPLOAD", "scope": AUDIT_SCOPE}
    if evidence != expected_evidence:
        block("VAST evidence identity/scope/counts drifted")
    license_rows = manifest["license_rows"]
    if not isinstance(license_rows, list) or not license_rows or canonical(license_rows) != manifest["license_rows_sha256"]:
        block("license disposition rows are missing or tampered")
    required_license_row_ids = {"firered-source-license", "firered-model-weight", "firered-training-provenance", "python-dependency-closure", "kaldi-native-fbank-source"}
    if {row.get("id") for row in license_rows if isinstance(row, dict)} != required_license_row_ids:
        block("license disposition row set is incomplete")
    if any(not isinstance(row, dict) or not isinstance(row.get("status"), str) or not isinstance(row.get("license"), str) or not row["status"].strip() or not row["license"].strip() for row in license_rows):
        block("license disposition contains an empty conclusion")
    if manifest["publication"] != "NO_UPLOAD" or manifest["approval"] != {"status": "OWNER_SIGNOFF_REQUIRED", "signer": None, "digest": None}:
        block("publication/owner gate is not fail-closed")
    scope = {"lock_sha256": manifest["lock_sha256"], "project_sha256": manifest["project_sha256"], "package_rows_sha256": manifest["package_rows_sha256"], "package_review_rows_sha256": manifest["package_review_rows_sha256"], "license_rows_sha256": manifest["license_rows_sha256"], "audit_evidence": evidence, "publication": manifest["publication"]}
    if manifest["approval_scope_sha256"] != canonical(scope):
        block("approval scope digest is not bound to all reviewed evidence")
    return manifest

def run(lock_path: Path, project_path: Path, manifest_path: Path) -> None:
    validate_manifest(lock_path, project_path, manifest_path)
    block("owner/legal approval is intentionally absent; dependency facts are reviewed but publication remains NO_UPLOAD")

def self_test() -> None:
    root = Path(__file__).resolve().parent
    production = load_json(root / "license_gate_manifest.json")
    validate_manifest(root / "uv.lock", root / "pyproject.toml", root / "license_gate_manifest.json")
    with tempfile.TemporaryDirectory(prefix="firered-license-gate-") as directory:
        path = Path(directory) / "manifest.json"
        for label, mutate in (("license", lambda value: value["license_rows"].__getitem__(0).update(license="GPL-3.0-only")), ("evidence", lambda value: value["audit_evidence"].update(archive_sha256="0" * 64)), ("approval", lambda value: value["approval"].update(status="OWNER_SIGNOFF_APPROVED")), ("duplicate", None)):
            if label == "duplicate":
                path.write_text('{"gate_version":1,"gate_version":1}', encoding="utf-8")
            else:
                value = json.loads(json.dumps(production))
                mutate(value)
                path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
            try:
                validate_manifest(root / "uv.lock", root / "pyproject.toml", path)
            except SystemExit as error:
                if error.code != 2:
                    raise
            else:
                raise AssertionError(f"tampered {label} manifest was accepted")
    print("firered_asr_aed_l license_gate self-test PASS")

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--project", type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    if args.self_test:
        if any(value is not None for value in (args.lock, args.project, args.manifest)):
            parser.error("--self-test accepts no path options")
        self_test()
        return 0
    if any(value is None for value in (args.lock, args.project, args.manifest)):
        parser.error("--lock, --project, and --manifest are required")
    run(args.lock, args.project, args.manifest)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
