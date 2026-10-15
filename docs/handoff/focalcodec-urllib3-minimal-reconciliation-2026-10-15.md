# FocalCodec urllib3 minimal reconciliation — 2026-10-15 JST

This source-only correction reviews Dependabot PR #250 at
`9b5e916c041ab635d8024f753df12e0cbc5cc78c`, against accepted main
`8bd30d9ab3857f77f3d0fa1edce34c450ea8c102`. The old candidate's 69
successful checks and one skipped check do not certify the corrected HEAD.

## Exact correction and independent review

The bot's lock update changes urllib3 2.7.0 to 2.8.0, but also rewrites 17
optional CUDA-toolkit dependency markers. Those unrelated rewrites remove
the original Linux/Windows conditions and Windows `AMD64` alternatives.
They are not treated as harmless formatting or part of the urllib3 update.

Luna restored all 17 marker entries byte-for-byte to accepted main. Manager
reviewed the complete diff and independently parsed both locks with Python
3.12 stdlib TOML: all 42 non-urllib3 package records and all root/manifest
fields are unchanged. The urllib3 record exactly matches the bot's update.
The final lock delta against main is three insertions and three deletions;
the correction does not regenerate or install the environment.

Corrected lock SHA-256:
`9c53af3f16dcd5d2e796acb7ee2ffbe33894ea31eafb59448b4e1db8e34701a5`.

Manager's bounded read of the [official PyPI release metadata](https://pypi.org/pypi/urllib3/2.8.0/json)
confirms both artifact URLs, SHA-256 values and byte sizes. The release and
artifacts are not yanked, and Python `>=3.10` admits this project's 3.12.
The official 2.8.0 tag resolves to
`b1d30ab61fe0db8f11092805e8c5ac43e091064a`; its
[primary LICENSE.txt](https://github.com/urllib3/urllib3/blob/b1d30ab61fe0db8f11092805e8c5ac43e091064a/LICENSE.txt)
is MIT. This is a narrowly reviewed dependency update, not a new audit or
execution approval for FocalCodec's whole Python/native closure.

## Acceptance boundary

Manager compared accepted main with the recorded green VAST source head
`4263f41631bd15288f06d3cd5c6b83ea4a8fdf60`: `crates`, root Cargo manifests
and lock, `tests`, `.cargo`, `.config` and root Rust/build configuration
have no differences. The correction adds no Rust or compiled-fixture delta.
This is source-identity evidence for the existing remote verification, not
a new full-workspace replay at the corrected lock HEAD.

Fresh corrected-head CI, required checks and normal review remain necessary.
No fixture, numerical bound, model/source revision, owner decision, legal
classification or publication gate changes. No model download/execution,
package installation/import, local heavy Cargo or new paid worker occurs.
The protected CosyVoice2 owner manifest is not inspected or changed. The
58 unresolved public rows and frozen 63-row owner scope remain distinct and
unchanged; final real-weight CPU evidence and Apple packets remain pending.
