# FACodec urllib3 minimal reconciliation — 2026-10-15 JST

This source-only correction reviews Dependabot [PR #252](https://github.com/ayutaz/vokra/pull/252)
at `c36498e4fdfd1a03c986bdfc1e778b69c9fc33d7`, against accepted main
`154919820b71166a375d0e1307e25aef5aa06e7e`. Checks on the bot's original
HEAD do not certify the corrected candidate.

## Exact correction and independent review

The bot updates urllib3 2.7.0 to 2.8.0, but also rewrites 17 optional
CUDA-toolkit dependency markers, removing Linux/Windows conditions and Windows
`AMD64` alternatives. These changes are outside the urllib3 update and are
not accepted as harmless formatting.

Luna restores all 17 marker entries byte-identically to accepted main.
Manager reviews the complete diff and independently parses base, bot and
corrected locks with offline Python 3.12 stdlib TOML. All 61 non-urllib3
package records and root fields match base. The urllib3 record matches the
bot exactly. The final lock delta is three insertions and three deletions;
no environment is resolved, installed or imported.

| Input | SHA-256 |
|---|---|
| Base dedicated lock | `3e5d8e8ecc022064e402b9da76356bbd1d6425495c7f1542df909f98d614c4b1` |
| Bot dedicated lock | `084884bc78f94bf5b0293b7f4130b25c1ca6b18b9f642e7fd636545fa8aaf635` |
| Corrected dedicated lock | `4d60e2685d97eb18283d575c742bdb28fb36e28360fb665527145e21303b0f0b` |
| Unchanged dedicated project | `23fae61f87ed373070a74305d72f086212f120755abb395a2c3fb48179e60bfa` |

The [official PyPI 2.8.0 metadata](https://pypi.org/pypi/urllib3/2.8.0/json)
matches the updated sdist/wheel URLs, SHA-256 values and byte sizes. Neither
artifact is yanked; Python `>=3.10` admits this project's Python 3.12. The
fixed official release commit
`b1d30ab61fe0db8f11092805e8c5ac43e091064a` contains the
[MIT primary license](https://github.com/urllib3/urllib3/blob/b1d30ab61fe0db8f11092805e8c5ac43e091064a/LICENSE.txt).
This narrowly reviewed dependency update does not approve FACodec's complete
Python/native closure or any model execution.

## Verification and acceptance boundary

Manager confirms that accepted main and recorded green VAST source head
`4263f41631bd15288f06d3cd5c6b83ea4a8fdf60` have no differences in `crates`,
root Cargo manifests/lock, `tests`, `.cargo`, `.config` or root Rust/build
configuration. This is source-identity evidence, not a new workspace replay.
Fresh corrected-head CI, all required context/application pairs and normal
merge protection remain necessary.

The ordinary pre-commit passes all five checks. Ordinary pre-push passes
all eight compliance regressions, then refuses its deep path because the
nested dedicated `uv.lock` is outside the existing Rust-neutral allowlist.
No local Cargo compile/test starts and no hook is modified. For this
corresponding Rust-identical source push only, the documented
`VOKRA_SKIP_HOOKS=1` path uses the recorded green VAST source identity above;
it is not a new full-workspace or Python-environment validation.

No numerical bound, fixture, source/model identity, owner/legal decision or
publication gate changes. No package/model execution, local heavy Cargo,
new paid worker or publication occurs. Protected CosyVoice2 owner material
and the Silero model fixture are not materialized or inspected in the private
checkout. The recorded 58 unresolved public rows and frozen 63-row owner
scope remain separate and unchanged; real-weight CPU and final Apple
transfer evidence remain unfinished.
