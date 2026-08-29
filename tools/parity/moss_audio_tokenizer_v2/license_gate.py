#!/usr/bin/env python3
"""Dependency-free, offline MOSS Audio Tokenizer v2 approval gate."""
from __future__ import annotations
import argparse, hashlib, json, re, sys, tempfile
from pathlib import Path
import tomllib
from urllib.parse import urlparse

REVISION = "f6e20e543b33d2c252a7ef71bdf8aa71e5ff9169"
REPO = "OpenMOSS-Team/MOSS-Audio-Tokenizer-v2"
LOCK_SHA256 = "22df7f7823d148eb11644b7c2bb40f9d457a72968eb9487cd8f28a95838cc364"
PROJECT_SHA256 = "ca7997270f39408084330da250c2e55d60fd8404db296ed1997302c6329290fa"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
UNRESOLVED = ("UNRESOLVED", "OWNER_REVIEW_REQUIRED", "PENDING_REVIEW", "REVIEW_REQUIRED")
PACKAGE_REVIEW_SCHEMA = {"name", "version", "source", "license", "status", "native_bundled_review"}
MANIFEST_SCHEMA = {"gate_version", "lock_sha256", "project_sha256", "package_rows", "package_rows_sha256", "package_review_rows", "package_review_rows_sha256", "identities", "tensor_manifest", "license_rows", "license_rows_sha256", "publication_decision", "approval"}
APPROVAL_SCHEMA = {"status", "signer", "digest"}
LOCK_KEYS = {"version", "revision", "requires-python", "resolution-markers", "supported-markers", "package"}
PACKAGE_KEYS = {
    frozenset({"name", "version", "source", "sdist", "wheels"}),
    frozenset({"dependencies", "name", "sdist", "source", "version", "wheels"}),
    frozenset({"name", "source", "version", "wheels"}),
    frozenset({"dependencies", "name", "source", "version", "wheels"}),
    frozenset({"dependencies", "metadata", "name", "source", "version"}),
}
ARTIFACT_KEYS = {"url", "hash", "size", "upload-time"}
REGISTRY_HOSTS = {
    "https://pypi.org/simple": "files.pythonhosted.org",
    "https://download.pytorch.org/whl/cu126": "download-r2.pytorch.org",
}
FILES = {
    "LICENSE": (11324, "50e6751797c50dedd75ef1b8a0d9e42f5f8472e9fbce91f34718e9f97b0c780a"),
    "config.json": (10166, "aeb9a0e9d88c74bf9fbaa81ee54443d463e09b5f335b3306bb798e282a10e564"),
    "configuration_moss_audio_tokenizer.py": (19772, "f87a7a975868ce3f0077f374f46ebd2aab610fd7a26cd7569d16827a14e29529"),
    "model.safetensors.index.json": (191718, "912f52f053e04ff7e9abc8f05aa75dfbb40b31c86a0f4ad5c5a36e4aa28a624f"),
    "modeling_moss_audio_tokenizer.py": (105970, "7f807e6ee77a60d512e5aa4a8f58a1d5af4e3722f4ab350d70dd538429391cb9"),
    "model-00001-of-00003.safetensors": (3978639168, "2d9f9182f17b143a23937feb87c63c08221bd28e685e4bc2fa55dcdce17fcde7"),
    "model-00002-of-00003.safetensors": (3992738352, "d4e48106d0254fe3b00ea0707e88fc6aee076993825e108dd9cef847f9db236e"),
    "model-00003-of-00003.safetensors": (523681336, "d0449fe1b0ef1f6045946867148d8166b9a91a58d0feca4a18b641494d0b22da"),
}
TENSOR = {"tensor_count": 2094, "parameter_count": 2123701248, "tensor_bytes_f32": 8494804992, "sample_rate": 48000, "channels": 2, "quantizers": 12, "codebook_size": 1024, "samples_per_frame": 3840}

def sha(data: bytes) -> str: return hashlib.sha256(data).hexdigest()
def canon(value: object) -> str: return sha(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())
def load_json(text: str) -> object:
    def reject(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result
    return json.loads(text, object_pairs_hook=reject)
def lock_rows(lock: dict) -> list[dict]:
    if set(lock) != LOCK_KEYS or lock.get("version") != 1 or type(lock.get("version")) is not int or lock.get("revision") != 3 or type(lock.get("revision")) is not int:
        raise ValueError("lock top-level schema drifted")
    if not isinstance(lock.get("requires-python"), str) or not isinstance(lock.get("resolution-markers"), list) or any(not isinstance(item, str) for item in lock["resolution-markers"]) or not isinstance(lock.get("supported-markers"), list) or any(not isinstance(item, str) for item in lock["supported-markers"]):
        raise ValueError("lock marker schema malformed")
    packages = lock.get("package")
    if not isinstance(packages, list) or not packages: raise ValueError("lock package table missing/empty")
    result = []
    seen = set()
    for p in packages:
        if not isinstance(p, dict) or not isinstance(p.get("name"), str) or not p["name"] or not isinstance(p.get("version"), str) or not p["version"] or frozenset(p) not in PACKAGE_KEYS:
            raise ValueError("malformed lock package row")
        key = (p["name"], p["version"])
        if key in seen: raise ValueError("duplicate lock package identity")
        seen.add(key)
        markers = p.get("resolution-markers", [])
        dependencies = p.get("dependencies", [])
        if not isinstance(markers, list) or any(not isinstance(marker, str) for marker in markers): raise ValueError("malformed lock resolution markers")
        if not isinstance(dependencies, list) or any(not isinstance(dep, dict) or set(dep) != {"name", "marker"} or not isinstance(dep.get("name"), str) or not dep["name"] or not isinstance(dep["marker"], str) for dep in dependencies): raise ValueError("malformed lock dependency row")
        source = p.get("source")
        if not isinstance(source, dict) or len(source) != 1 or set(source) not in ({"registry"}, {"virtual"}): raise ValueError("malformed lock source")
        if "registry" in source and source["registry"] not in REGISTRY_HOSTS: raise ValueError("unsupported lock registry")
        if "virtual" in source and source["virtual"] != ".": raise ValueError("malformed virtual source")
        if "virtual" in source:
            metadata = p.get("metadata")
            if not isinstance(metadata, dict) or set(metadata) != {"requires-dist"} or not isinstance(metadata["requires-dist"], list) or any(not isinstance(req, dict) or set(req) not in ({"name", "specifier"}, {"index", "name", "specifier"}) or not isinstance(req.get("name"), str) or not req["name"] or not isinstance(req.get("specifier"), str) or ("index" in req and req["index"] != "https://download.pytorch.org/whl/cu126") for req in metadata["requires-dist"]): raise ValueError("malformed virtual metadata")
        result.append({"name":p["name"],"version":p["version"],"source":p.get("source"),"resolution-markers":p.get("resolution-markers",[]),"dependencies":p.get("dependencies",[])})
    return sorted(result, key=lambda x:(x["name"],x["version"]))

def artifact_error(lock: dict) -> str | None:
    """Require resolver-pinned distribution metadata for every real package."""
    packages = lock.get("package")
    if not isinstance(packages, list):
        return "package table is not a list"
    virtual_count = 0
    for package in packages:
        if not isinstance(package, dict):
            return "package row is not a table"
        source = package.get("source")
        if not isinstance(source, dict):
            return f"package {package.get('name')!r} has malformed source"
        if frozenset(package) not in PACKAGE_KEYS:
            return f"package {package.get('name')!r} has an inexact row schema"
        if "sdist" in package and not isinstance(package["sdist"], dict):
            return f"package {package.get('name')!r} has malformed sdist"
        if "wheels" in package and not isinstance(package["wheels"], list):
            return f"package {package.get('name')!r} has malformed wheels"
        if source == {"virtual": "."}:
            if "sdist" in package or "wheels" in package:
                return "virtual project source cannot carry resolver artifacts"
            if set(package) != frozenset({"dependencies", "metadata", "name", "source", "version"}) or not isinstance(package.get("metadata"), dict) or set(package["metadata"]) != {"requires-dist"} or not isinstance(package["metadata"]["requires-dist"], list):
                return "virtual project metadata schema is not exact"
            virtual_count += 1
            continue
        if set(source) != {"registry"} or source.get("registry") not in REGISTRY_HOSTS:
            return f"package {package.get('name')!r} has malformed registry source"
        artifacts = []
        sdist = package.get("sdist")
        if isinstance(sdist, dict):
            artifacts.append(sdist)
        wheels = package.get("wheels")
        if isinstance(wheels, list):
            artifacts.extend(wheels)
        if not artifacts:
            return f"package {package.get('name')!r} has no resolver artifacts"
        for artifact in artifacts:
            if not isinstance(artifact, dict) or set(artifact) != ARTIFACT_KEYS:
                return f"package {package.get('name')!r} has malformed artifact"
            parsed = None
            try:
                parsed = urlparse(artifact["url"])
                hostname = parsed.hostname
            except (TypeError, ValueError):
                hostname = None
            expected_host = REGISTRY_HOSTS[source["registry"]]
            if (parsed is None or not isinstance(artifact["url"], str) or parsed.scheme != "https" or hostname != expected_host or parsed.netloc != hostname or not parsed.path.startswith("/") or parsed.query or parsed.fragment):
                return f"package {package.get('name')!r} has invalid artifact URL"
            if not isinstance(artifact["hash"], str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", artifact["hash"]):
                return f"package {package.get('name')!r} has invalid artifact hash"
            if not isinstance(artifact["size"], int) or isinstance(artifact["size"], bool) or artifact["size"] <= 0:
                return f"package {package.get('name')!r} has invalid artifact size"
            if not isinstance(artifact["upload-time"], str) or not artifact["upload-time"].strip():
                return f"package {package.get('name')!r} has invalid artifact upload-time"
    if virtual_count != 1:
        return "lock must contain exactly one virtual project source"
    return None

def project_identity(project: bytes) -> tuple[str, str]:
    data = tomllib.loads(project.decode())
    metadata = data.get("project")
    if not isinstance(metadata, dict) or not isinstance(metadata.get("name"), str) or not metadata["name"] or not isinstance(metadata.get("version"), str) or not metadata["version"]:
        raise ValueError("project must declare nonempty name and version")
    return metadata["name"], metadata["version"]
def resolved(value: object) -> bool:
    if not isinstance(value, str): return False
    normalized = "_".join(value.strip().casefold().split())
    return bool(normalized) and normalized not in {"", "null", "none", "unresolved", "pending", "pending_review", "owner_review_required", "review_required", "todo"}
def scope(m: dict) -> str:
    return canon({"schema":"moss-audio-tokenizer-v2-approval-v1","lock_sha256":m.get("lock_sha256"),"project_sha256":m.get("project_sha256"),"package_rows":m.get("package_rows"),"package_rows_sha256":m.get("package_rows_sha256"),"package_review_rows":m.get("package_review_rows"),"package_review_rows_sha256":m.get("package_review_rows_sha256"),"license_rows":m.get("license_rows"),"license_rows_sha256":m.get("license_rows_sha256"),"identities":m.get("identities"),"tensor_manifest":m.get("tensor_manifest"),"publication_decision":m.get("publication_decision"),"expected_decision":"APPROVED"})
def blocked(message: str) -> None:
    print(f"moss v2 license gate: BLOCKED: {message}", file=sys.stderr); raise SystemExit(2)
def run(lock_path: Path, project_path: Path, manifest_path: Path, approval: Path|None) -> None:
    for path, label in ((lock_path, "lock"), (project_path, "project"), (manifest_path, "manifest")):
        if path.is_symlink() or not path.is_file(): blocked(f"{label} input is missing or not a regular file")
    try:
        lock_bytes=lock_path.read_bytes(); project_bytes=project_path.read_bytes(); m=load_json(manifest_path.read_text()); lock=tomllib.loads(lock_bytes.decode()); project_name, project_version = project_identity(project_bytes)
        rows=lock_rows(lock)
    except (OSError,UnicodeDecodeError,tomllib.TOMLDecodeError,json.JSONDecodeError,ValueError) as e: blocked(f"invalid closure: {e}")
    if (error := artifact_error(lock)) is not None: blocked(f"resolver artifact metadata: {error}")
    virtuals = [p for p in lock["package"] if isinstance(p, dict) and p.get("source") == {"virtual": "."}]
    if len(virtuals) != 1 or virtuals[0].get("name") != project_name or virtuals[0].get("version") != project_version:
        blocked("virtual project row does not bind to pyproject identity")
    if not isinstance(m, dict) or set(m) != MANIFEST_SCHEMA: blocked("manifest schema drifted")
    if m.get("gate_version") != 1 or type(m.get("gate_version")) is not int: blocked("unsupported gate_version")
    if sha(lock_bytes) != LOCK_SHA256 or sha(project_bytes) != PROJECT_SHA256: blocked("lock/project bytes differ from code-bound closure")
    if m.get("lock_sha256") != LOCK_SHA256 or m.get("project_sha256") != PROJECT_SHA256: blocked("manifest lock/project hashes differ from code-bound closure")
    if m.get("package_rows") != rows or m.get("package_rows_sha256") != canon(rows): blocked("canonical lock rows drifted")
    reviews=m.get("package_review_rows")
    if not isinstance(reviews,list) or len(reviews)!=len(rows): blocked("every locked package needs a review row")
    actual={(x["name"],x["version"]):x for x in rows}; seen=set()
    for r in reviews:
        if not isinstance(r,dict) or set(r) != PACKAGE_REVIEW_SCHEMA: blocked("package review row schema drifted")
        key=(r.get("name"),r.get("version")) if isinstance(r,dict) else None
        if key in seen or key not in actual or r.get("source") != actual[key].get("source"): blocked("package review identity/source drifted")
        seen.add(key)
        if r.get("status") != "REVIEWED" or not resolved(r.get("license")) or not resolved(r.get("native_bundled_review")): blocked(f"package review unresolved: {key}")
    if seen != set(actual) or m.get("package_review_rows_sha256") != canon(reviews): blocked("package review rows drifted")
    identities=m.get("identities")
    if not isinstance(identities,dict) or identities.get("repo") != REPO or identities.get("revision") != REVISION: blocked("fixed upstream identity drifted")
    if set(identities) != {"repo", "revision"} | {f"{n}_{suffix}" for n in FILES for suffix in ("bytes", "sha256")}: blocked("snapshot identity entries are missing or extra")
    for name,(expected_bytes, expected_hash) in FILES.items():
        if identities.get(f"{name}_bytes") != expected_bytes or identities.get(f"{name}_sha256") != expected_hash: blocked(f"fixed file identity drifted: {name}")
    license_rows=m.get("license_rows")
    required_license_ids=["source-apache", "weights-apache", "python-closure"]
    if not isinstance(license_rows,list) or len(license_rows)!=3 or any(not isinstance(r,dict) for r in license_rows) or [r.get("id") for r in license_rows] != required_license_ids: blocked("license rows are missing, duplicated, reordered, or extra")
    if any(set(r) != {"id","license","status","conclusion","native_bundled_review"} for r in license_rows): blocked("license row schema is not canonical")
    if m.get("license_rows_sha256") != canon(license_rows): blocked("license rows digest drifted")
    if any(r.get("status") != "REVIEWED" or not resolved(r.get("license")) or not resolved(r.get("conclusion")) or not resolved(r.get("native_bundled_review")) for r in license_rows): blocked("license conclusion/native disposition unresolved")
    if m.get("publication_decision") != "NO_UPLOAD": blocked("publication decision is not NO_UPLOAD")
    tensor_manifest=m.get("tensor_manifest")
    if not isinstance(tensor_manifest,dict) or set(tensor_manifest) != set(TENSOR) or any(type(tensor_manifest[key]) is not int for key in TENSOR) or tensor_manifest != TENSOR: blocked("tensor manifest/topology is not fixed")
    a=m.get("approval")
    if not isinstance(a,dict) or set(a) != APPROVAL_SCHEMA or a.get("status") != "OWNER_SIGNOFF_APPROVED": blocked("owner signoff remains required")
    expected_scope=scope(m)
    if a.get("digest") != expected_scope or not isinstance(a.get("signer"),str) or not a["signer"] or not HEX64.fullmatch(str(a.get("digest"))): blocked("approval digest is not canonical")
    if approval is None or approval.is_symlink() or not approval.is_file(): blocked("approval evidence missing or is not a regular file")
    try: e=load_json(approval.read_text())
    except (OSError,json.JSONDecodeError,ValueError) as x: blocked(f"approval evidence unreadable: {x}")
    if not isinstance(e, dict) or set(e) != {"scope_schema", "scope_sha256", "approval_digest", "decision", "signer", "manifest_sha256"} or e.get("scope_schema") != "moss-audio-tokenizer-v2-approval-v1" or e.get("scope_sha256") != expected_scope or e.get("approval_digest") != expected_scope or e.get("decision") != "APPROVED" or e.get("signer") != a["signer"] or e.get("manifest_sha256") != sha(manifest_path.read_bytes()): blocked("approval evidence does not bind canonical scope")
    print("moss v2 license gate: PASS")
def self_test() -> None:
    for value in (None, "", " null ", "OWNER_REVIEW_REQUIRED", " pending_review ", "TODO"):
        if resolved(value): raise SystemExit(f"self-test resolved placeholder: {value!r}")
    if not resolved("owner_review_required is documented as the historical status"):
        raise SystemExit("self-test rejected a legitimate longer review citation")
    with tempfile.TemporaryDirectory() as d:
        p=Path(d); lock=p/"uv.lock"; project=p/"pyproject.toml"; manifest=p/"manifest.json"; approval=p/"approval.json"
        virtual='\n[[package]]\nname="demo"\nversion="0.1.0"\nsource={virtual="."}\ndependencies=[]\n[package.metadata]\nrequires-dist=[]\n'
        lock.write_text('version=1\nrevision=3\nrequires-python="==3.12.*"\nresolution-markers=[]\nsupported-markers=[]\n[[package]]\nname="demo"\nversion="1"\nsource={registry="https://pypi.org/simple"}\nsdist={url="https://files.pythonhosted.org/packages/demo.tar.gz",hash="sha256:' + 'a'*64 + '",size=1,upload-time="2026-01-01T00:00:00Z"}\nwheels=[{url="https://files.pythonhosted.org/packages/demo.whl",hash="sha256:' + 'b'*64 + '",size=2,upload-time="2026-01-01T00:00:00Z"}]\n' + virtual,encoding="utf-8"); project.write_text('[project]\nname="demo"\nversion="0.1.0"\n',encoding="utf-8")
        valid_lock=tomllib.loads(lock.read_text())
        for label, mutate in (("top-extra", lambda value: value.update(unexpected=True)), ("top-missing", lambda value: value.pop("revision")), ("package-extra", lambda value: value["package"][0].update(unexpected=True))):
            candidate = load_json(json.dumps(valid_lock)); mutate(candidate)
            try:
                lock_rows(candidate)
            except ValueError:
                continue
            raise SystemExit(f"self-test accepted malformed lock schema: {label}")
        for label, mutate in (("sdist", lambda x: x["package"][0].update(sdist="bad")), ("wheels", lambda x: x["package"][0].update(wheels={})), ("source", lambda x: x["package"][0].update(source="bad")), ("virtual-source", lambda x: x["package"][0].update(source={"virtual":"other"})), ("missing-virtual", lambda x: x["package"].pop()), ("duplicate-virtual", lambda x: x["package"].append(dict(x["package"][-1]))), ("duplicate-package", lambda x: x["package"].append(dict(x["package"][0]))), ("bool-size", lambda x: x["package"][0]["sdist"].update(size=True))):
            candidate = load_json(json.dumps(valid_lock)); mutate(candidate)
            rejected = artifact_error(candidate) is not None
            if label == "duplicate-package":
                try: lock_rows(candidate)
                except ValueError: rejected = True
            if not rejected: raise SystemExit(f"self-test accepted malformed lock: {label}")
        for field in ("url", "hash", "size", "upload-time"):
            candidate = load_json(json.dumps(valid_lock)); candidate["package"][0]["sdist"].pop(field)
            if artifact_error(candidate) is None: raise SystemExit(f"self-test accepted missing artifact field: {field}")
        for label, mutate in (("extra-artifact-field", lambda value: value["package"][0]["sdist"].update(extra=True)), ("empty-upload-time", lambda value: value["package"][0]["sdist"].update(**{"upload-time": " "})), ("evil-host", lambda value: value["package"][0]["sdist"].update(url="https://evil.example/packages/demo.tar.gz"))):
            candidate = load_json(json.dumps(valid_lock)); mutate(candidate)
            if artifact_error(candidate) is None: raise SystemExit(f"self-test accepted malformed artifact: {label}")
        rows=lock_rows(valid_lock); reviews=[{"name":"demo","version":"1","source":{"registry":"https://pypi.org/simple"},"license":"MIT","status":"REVIEWED","native_bundled_review":"reviewed"},{"name":"demo","version":"0.1.0","source":{"virtual":"."},"license":"project","status":"REVIEWED","native_bundled_review":"reviewed"}]
        identities={"repo":REPO,"revision":REVISION,**{f"{n}_{suffix}":value for n,(size,h) in FILES.items() for suffix,value in (("bytes",size),("sha256",h))}}
        licenses=[{"id":"source-apache","license":"Apache-2.0","status":"REVIEWED","conclusion":"reviewed","native_bundled_review":"reviewed"},{"id":"weights-apache","license":"Apache-2.0","status":"REVIEWED","conclusion":"reviewed","native_bundled_review":"reviewed"},{"id":"python-closure","license":"MIT","status":"REVIEWED","conclusion":"reviewed","native_bundled_review":"reviewed"}]
        global LOCK_SHA256, PROJECT_SHA256
        LOCK_SHA256, PROJECT_SHA256 = sha(lock.read_bytes()), sha(project.read_bytes())
        m={"gate_version":1,"lock_sha256":LOCK_SHA256,"project_sha256":PROJECT_SHA256,"package_rows":rows,"package_rows_sha256":canon(rows),"package_review_rows":reviews,"package_review_rows_sha256":canon(reviews),"license_rows":licenses,"license_rows_sha256":canon(licenses),"identities":identities,"tensor_manifest":TENSOR,"publication_decision":"NO_UPLOAD","approval":{"status":"OWNER_SIGNOFF_APPROVED","signer":"test","digest":None}}
        m["approval"]["digest"]=scope(m); manifest.write_text(json.dumps(m,sort_keys=True)); e={"scope_schema":"moss-audio-tokenizer-v2-approval-v1","scope_sha256":m["approval"]["digest"],"approval_digest":m["approval"]["digest"],"decision":"APPROVED","signer":"test","manifest_sha256":sha(manifest.read_bytes())}; approval.write_text(json.dumps(e)); run(lock,project,manifest,approval)
        for input_path, label in ((lock, "lock-input"), (project, "project-input"), (manifest, "manifest-input")):
            target = p / (label + "-target"); target.write_bytes(b"input-target")
            original = input_path.read_bytes(); input_path.unlink(); input_path.symlink_to(target)
            try: run(lock,project,manifest,approval)
            except SystemExit as x:
                if x.code!=2: raise
            else: raise SystemExit(f"self-test accepted symlink {label}")
            input_path.unlink(); input_path.write_bytes(original)
        for label,mutate in (("artifact",lambda x: x["lock"].get("package", [])[0].update(sdist={"url":"https://files.pythonhosted.org/demo.tar.gz","hash":"sha256:" + "a"*64})),("scope",lambda x:x["manifest"]["tensor_manifest"].update(tensor_count=1)),("arbitrary",lambda x:x["manifest"]["approval"].update(digest="a"*64)),("publication",lambda x:x["manifest"].update(publication_decision="UPLOAD")),("conclusion",lambda x:x["manifest"]["license_rows"][0].update(conclusion="OWNER_REVIEW_REQUIRED")),("manifest-schema",lambda x:x["manifest"].update(extra=True)),("approval-schema",lambda x:x["manifest"]["approval"].update(extra=True))):
            if label == "artifact":
                candidate_lock=tomllib.loads(lock.read_text()); mutate({"lock":candidate_lock,"manifest":load_json(manifest.read_text())})
                lock.write_text('version=1\n[[package]]\nname="demo"\nversion="1"\nsource={registry="https://pypi.org/simple"}\nsdist={url="https://files.pythonhosted.org/demo.tar.gz",hash="sha256:' + 'a'*64 + '"}\n' + virtual,encoding="utf-8")
                try: run(lock,project,manifest,approval)
                except SystemExit as x:
                    if x.code!=2: raise
                else: raise SystemExit("self-test accepted artifact tamper")
                lock.write_text('version=1\n[[package]]\nname="demo"\nversion="1"\nsource={registry="https://pypi.org/simple"}\nsdist={url="https://files.pythonhosted.org/demo.tar.gz",hash="sha256:' + 'a'*64 + '",size=1}\n' + virtual,encoding="utf-8")
                continue
            c=load_json(manifest.read_text()); mutate({"manifest":c}); manifest.write_text(json.dumps(c,sort_keys=True))
            try: run(lock,project,manifest,approval)
            except SystemExit as x:
                if x.code!=2: raise
            else: raise SystemExit(f"self-test accepted {label} tamper")
            manifest.write_text(json.dumps(m,sort_keys=True)); approval.write_text(json.dumps(e))
        for label, mutate in (("evidence-scope", lambda x: x.update(scope_sha256="a" * 64)), ("evidence-signer", lambda x: x.update(signer="other")), ("evidence-decision", lambda x: x.update(decision="PENDING")), ("evidence-extra", lambda x: x.update(extra=True))):
            candidate = load_json(approval.read_text()); mutate(candidate); approval.write_text(json.dumps(candidate))
            try: run(lock,project,manifest,approval)
            except SystemExit as x:
                if x.code!=2: raise
            else: raise SystemExit(f"self-test accepted {label} tamper")
            approval.write_text(json.dumps(e))
        approval.unlink(); approval.symlink_to(manifest)
        try: run(lock,project,manifest,approval)
        except SystemExit as x:
            if x.code!=2: raise
        else: raise SystemExit("self-test accepted symlink evidence")
        approval.unlink(); approval.write_text(json.dumps(e))
    print("license_gate.py self-test: PASS")
if __name__ == "__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--self-test",action="store_true"); ap.add_argument("--lock",type=Path); ap.add_argument("--project",type=Path); ap.add_argument("--manifest",type=Path); ap.add_argument("--approval-evidence", dest="approval", type=Path); a=ap.parse_args()
    if a.self_test: self_test()
    elif not all((a.lock,a.project,a.manifest)): ap.error("--lock, --project, and --manifest are required")
    else: run(a.lock,a.project,a.manifest,a.approval)
