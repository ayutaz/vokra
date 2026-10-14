# Kyutai package metadata receipt

`package_metadata_audit.py` authenticates a narrowly scoped supplemental
receipt from explicitly supplied GitHub API JSON captures. It is
`SOURCE_METADATA_ONLY`: it does not import or execute captured Python, resolve
or install a dependency, inspect a wheel, inspect native payloads, load audio
or weights, or change the authenticated Kyutai source packet.

The fixed identities are:

| repository | commit | complete recursive tree |
| --- | --- | --- |
| `kyutai-labs/moshi` | `e6a55d2722a65870ef52a6c9f6ecfc0e90f38362` | `2a6d6afe53d70bac490117651dfc478cf87a940e` |
| `kyutai-labs/delayed-streams-modeling` | `4c4f65e147df056adf3346290d64c7b9649b18c9` | `1ab73718d99c5bb6ff94c1bc84a783a4d2e3a7e0` |

The CPU `moshi/` subtree must contain exactly these three *packaging metadata*
files (the subtree also contains ordinary source files), with their fixed Git
blob IDs and byte sizes: `moshi/pyproject.toml`
(`0a99f52ea834cdcfe1b07ec3cfcbb7e96083fb61`, 1305),
`moshi/requirements.txt`
(`89cee794c64d70932c56be7f964f23aa921dd0fa`, 195), and `moshi/setup.cfg`
(`4c7f6cd5b66032ce7b1d45a779b7692ad9c1d443`, 125). The auditor also reconstructs every Git tree object
bottom-up from the recursive capture; directory entries are converted from
GitHub's API mode `040000` to Git's canonical tree-object mode `40000`, so it
does not trust a copied root hash or `truncated: false`. Other Moshi packaging
paths (for example `moshi_mlx/` and Rust-side metadata) are explicitly
reported as `unassessed_packaging_paths`; they are outside this CPU `moshi/`
scope and are not silently treated as resolved. DSM is checked for complete-tree absence of
`pyproject.toml`, `setup.py`, `setup.cfg`, `requirements*`, and `uv.lock`.

## Offline use

Each argument is a small, regular JSON file containing the corresponding
official commit, recursive-tree, or blob response. Blob responses use the
GitHub `base64` encoding. A capture must be acquired and reviewed separately;
the auditor does not fetch URLs or perform checkouts.

```sh
UV_CACHE_DIR=/private/tmp/vokra-kyutai-uv-cache \
  uv run --offline --no-project --no-sync --python 3.12 python -B -S \
  tools/parity/kyutai_stt_streaming_reference/package_metadata_audit.py \
  --moshi-commit /private/tmp/vokra-kyutai-package-metadata-20261007/moshi-commit.json \
  --moshi-tree /private/tmp/vokra-kyutai-package-metadata-20261007/moshi-tree.json \
  --moshi-pyproject /private/tmp/vokra-kyutai-package-metadata-20261007/moshi-pyproject.json \
  --moshi-requirements /private/tmp/vokra-kyutai-package-metadata-20261007/moshi-requirements.json \
  --moshi-setup-cfg /private/tmp/vokra-kyutai-package-metadata-20261007/moshi-setup.cfg.json \
  --dsm-commit /private/tmp/vokra-kyutai-package-metadata-20261007/dsm-commit.json \
  --dsm-tree /private/tmp/vokra-kyutai-package-metadata-20261007/dsm-tree.json \
  --output /private/tmp/vokra-kyutai-package-metadata-20261007/package-metadata-receipt.json
```

The output schema is `vokra-kyutai-package-metadata-receipt-v1` and its status
is `SOURCE_METADATA_ONLY`. It records the raw constraint lines and preserves
the `sphn` conflict (`>=0.2.0,<0.3.0` in pyproject versus `==0.1.4` in
requirements) without selecting precedence. The Linux-required
`bitsandbytes>=0.45,<0.50.0` constraint is mandatory evidence. The existing
seven third-party graph candidates remain `CANDIDATE_UNKNOWN`; the dynamic
helper row remains `DYNAMIC_IMPORT_LITERAL_AUTHENTICATED_SOURCE` with unknown
runtime origin. Reviewed dependency closure remains empty, owner/legal review
remains false, license/native facts remain unresolved, and `no_upload` remains
true.

Input captures are bounded regular files and output creation is exclusive:
existing files and symlink targets are refused. The fixed-spec public audit
path is used by the CLI; synthetic alternate specs exist only in private test
helpers and cannot be selected through the CLI.

This receipt cannot promote source graph, execution, parity, license,
publication, or model readiness. It must not be inserted into the existing
`vokra-flat-source-receipt-v1` packet or used as a lock file. A future factual
dependency collector requires separately authenticated package artifacts and
owner/legal review; no versions are guessed here.
