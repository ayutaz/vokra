"""Independent Kyutai STT streaming reference helpers.

The package contains only the orchestration and packet contract.  The real
path imports the pinned upstream Moshi implementation at run time on VAST;
the repository never carries model weights or generated fixtures.

The only executable real path is the separately approved
``KYUTAI_STT_PYTORCH_PCM_ORACLE_CAPTURE`` VAST contract.  Source-only
preparation, synthetic capture tests, or the older decoder approval do not
authorize execution or establish numerical parity.
"""
