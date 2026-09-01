#!/usr/bin/env bash
# Hermetic checks for provision.sh's non-mutating dependency contract.
# No provisioning, package resolution, network, model, or Cargo work occurs.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
provision="$script_dir/provision.sh"
tmp_dir="$(mktemp -d "${TMPDIR:-/tmp}/vokra-provision-test.XXXXXX")"
trap 'rm -rf "$tmp_dir"' EXIT HUP INT TERM

fail() {
  echo "test-provision.sh: $*" >&2
  exit 1
}

# A source-level guard makes the repository-mutation contract explicit. The
# real self-test below additionally executes the probe path with a fake uv.
if grep -Eq '(^|[[:space:]])uv[[:space:]]+add([[:space:]]|$)|uv[[:space:]]+run[[:space:]]+--with' "$provision"; then
  fail "provision.sh still contains a mutating/dynamic uv dependency command"
fi
grep -Fq 'command -v ninja' "$provision" || fail "native build probe omits ninja"
grep -Fq 'ninja-build' "$provision" || fail "Debian native build install omits ninja-build"

fixture="$tmp_dir/checkout"
fake_bin="$tmp_dir/bin"
fake_home="$tmp_dir/home"
uv_log="$tmp_dir/uv-args.log"
mkdir -p "$fixture/.git" "$fixture/tools/parity" "$fake_bin" "$fake_home"
printf '%s\n' '[project]' 'name = "fixture-parity"' >"$fixture/tools/parity/pyproject.toml"
printf '%s\n' 'lock-fixture-v1' >"$fixture/tools/parity/uv.lock"

# Fake commands are intentionally incapable of installing or resolving. Any
# unexpected invocation (notably `uv add`/`uv run --with`) fails the test.
cat >"$fake_bin/uv" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$*" >>"${PROVISION_FAKE_UV_LOG:?}"
case " $* " in
  *' add '*|*' --with '*)
    echo 'unexpected dependency mutation/resolution' >&2
    exit 91
    ;;
esac
if [[ "${1:-}" == "--version" ]]; then
  printf '%s\n' 'uv 0.0.0-provision-self-test'
  exit 0
fi
exit 92
EOF
cat >"$fake_bin/cargo" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' 'cargo 0.0.0-provision-self-test'
EOF
chmod 700 "$fake_bin/uv" "$fake_bin/cargo"

if command -v sha256sum >/dev/null 2>&1; then
  digest_cmd=(sha256sum)
elif command -v shasum >/dev/null 2>&1; then
  digest_cmd=(shasum -a 256)
else
  fail "neither sha256sum nor shasum is available"
fi
before_files="$tmp_dir/files.before"
after_files="$tmp_dir/files.after"
before_hashes="$tmp_dir/hashes.before"
after_hashes="$tmp_dir/hashes.after"
( cd "$fixture" && find . -print | LC_ALL=C sort ) >"$before_files"
"${digest_cmd[@]}" "$fixture/tools/parity/pyproject.toml" "$fixture/tools/parity/uv.lock" >"$before_hashes"

HOME="$fake_home" \
  PATH="$fake_bin:/usr/bin:/bin" \
  VOKRA_ROOT="$fixture" \
  PROVISION_FAKE_UV_LOG="$uv_log" \
  bash "$provision" --self-test >"$tmp_dir/self-test.out"

( cd "$fixture" && find . -print | LC_ALL=C sort ) >"$after_files"
"${digest_cmd[@]}" "$fixture/tools/parity/pyproject.toml" "$fixture/tools/parity/uv.lock" >"$after_hashes"
cmp -s "$before_files" "$after_files" || fail "self-test changed fixture checkout file set"
cmp -s "$before_hashes" "$after_hashes" || fail "self-test changed fixture dependency files"
if grep -Eq '(^|[[:space:]])add([[:space:]]|$)|--with' "$uv_log"; then
  fail "self-test attempted dependency mutation/resolution"
fi
grep -Fq 'no side effects' "$tmp_dir/self-test.out" || fail "self-test did not report its no-side-effects contract"

echo 'test-provision.sh: OK (self-test is offline and repository-preserving)'
