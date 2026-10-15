# RMVPE primary license readback — 2026-10-14

> **2026-10-15 transport-only correction:** hosted documentation checking
> returned HTTP 403 for the recursive GitHub API citation. An authenticated
> read still returns the same fixed identity, all 16 entries and
> `truncated=false`; the browsable fixed-revision source tree is accessible.
> The primary-source link below now uses that tree page, without changing
> the inspected revision, historical evidence or unresolved license verdict.

Read-only official-source investigation. No archive/model payload was
downloaded, no model/package was imported, and no cloud worker or owner
decision was created. The historical public `vokra/rmvpe` metadata and its
live license stamp are not proof of upstream rights.

## Exact evidence

The fixed official source is `yxlllc/RMVPE` at
`0aabafba18289ca938a73af0b0297686abf4922d`. The complete recursive tree has
no `LICENSE`, `COPYING` or equivalent license path. The inspected README and
headers do not state an explicit code/weight usage or redistribution grant.
README and the five source-role blob identities agree with the current
inspector; matching source identity does not establish a license.

The official author release `230917` provides `rmvpe.zip` (340,638,958 bytes),
described as containing `model.pt`; the release metadata supplies no digest
for that ZIP. The separately digested ONNX ZIP is a different asset, not a
checkpoint-byte proof or license grant. The tag tree also has no license file.
The README/release describes MIR-1K and PTDB processing plus M4Singer
vocoder-synthetic data, without an explicit checkpoint/data redistribution
grant. Current GitHub repository metadata reports `license: null`; no later
official statement was located that explicitly covers the fixed source and
checkpoint. These findings mean no grant was established by this audit, not
a legal assertion that no permission could exist elsewhere.

Primary sources:

- [Fixed official source tree](https://github.com/yxlllc/RMVPE/tree/0aabafba18289ca938a73af0b0297686abf4922d)
- [Fixed official README](https://raw.githubusercontent.com/yxlllc/RMVPE/0aabafba18289ca938a73af0b0297686abf4922d/README.md)
- [Official release 230917](https://github.com/yxlllc/RMVPE/releases/tag/230917)

## Gate and next action

The existing model-free inspector self-test passes. Its dependency gate exits
2 with `BLOCKED_UNREVIEWED_NATIVE_AND_MODEL_LICENSE`. Missing exact-source
grant, checkpoint terms/digest and native/bundled dependency review remain
open, including soxr/libsndfile LGPL and wheel notice boundaries.

First obtain an explicit author/rightsholder grant covering the exact code
and checkpoint, together with the applicable training-data disposition.
Then establish checkpoint bytes on an authorized disposable VAST worker and
bind the exact dependency/native/owner scope. A fork, paper license, model-card
classifier or current public MIT stamp must not substitute for that evidence.
No withdrawal, replacement, real-weight run or publication is authorized by
this document. Retain RMVPE in the unresolved inventory and the frozen owner
packet; do not auto-promote or silently drop its row.

## Fork and license history — 2026-10-15

A separate bounded primary-source review corrects the current audit table's
old statement that the repositories have no GitHub fork relationship. GitHub
identifies `yxlllc/RMVPE` as a fork whose parent/source is
`Dream-High/RMVPE`. The exact fixed commit exists in both repositories, and
the Dream-High tree at that commit also lacks a license path. Fork metadata
does not itself supply a grant for the fixed modifications or checkpoint.

Dream-High added its [Apache-2.0 license text](https://raw.githubusercontent.com/Dream-High/RMVPE/a6db1cd7d26014aa739383367afd9bab57fc624c/LICENSE)
in [commit a6db1cd7](https://github.com/Dream-High/RMVPE/commit/a6db1cd7d26014aa739383367afd9bab57fc624c)
on 2024-01-25. Its license blob is
`261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64`. Manager independently reads the
primary text and checks the commit and comparison metadata. Comparing the
fixed source to this license-addition commit reports diverged branches,
one commit ahead and 15 behind, with merge base
`905c189aa8955ce1861500d5370ab83a33d4fcf6`. The license-addition commit is
not a descendant of the exact fixed source commit. This bounded history
review does not establish that the later grant covers the divergent fixed
code, and does not grant checkpoint/data rights. Source and weight remain
separately unresolved, all sign-off fields stay unchanged, and no package,
checkpoint or model was acquired or executed for this review.
