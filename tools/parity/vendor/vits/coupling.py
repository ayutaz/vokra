"""VITS ``ResidualCouplingLayer`` (upstream ``modules.py`` lines 298-343).

This target module is a thin re-export from the sibling ``modules.py``
(which was vendored verbatim from ``jaywalnut310/vits/modules.py``).
``ResidualCouplingLayer`` and the ``WN`` helper it wraps both live in
that vendored ``modules.py`` — re-exporting from there (rather than
extracting a second byte-identical copy into this file) keeps the
vendored source de-duplicated: a future re-audit that diffs
``modules.py`` against upstream automatically covers ``coupling.py``'s
implementation too, and there is no chance of the two copies drifting
apart.

See the sibling ``README.md`` mapping table for the class-to-target-
file contract this satisfies (design doc §7 / ``crates/vokra-models/src/
sbv2/flow.rs``: this is the affine-coupling step inside the sibling
``flow.py``'s ``ResidualCouplingBlock``).

# NOT REFERENCED: github.com/litagin02/Style-Bert-VITS2 (AGPL-3.0),
# github.com/fishaudio/Bert-VITS2 (AGPL-3.0). Only ``jaywalnut310/vits``
# MIT source has been read or copied. See
# ``tools/parity/vendor/vits/README.md`` for the full attribution +
# clean-room rationale.
"""

from .modules import ResidualCouplingLayer, WN

__all__ = ["ResidualCouplingLayer", "WN"]
