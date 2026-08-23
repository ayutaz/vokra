# M5-02 QNN/Hexagon prerequisite evidence — 2026-08-24 JST

This is a prerequisite failure record, not a QNN performance result. The
authoring host has no Qualcomm AI Runtime SDK, QNN headers/tools/runtime, or
Snapdragon/Hexagon target. Consequently no CPU or QNN latency number and no HTP
placement fraction is recorded; substituting the Apple CPU baseline would
violate the same-host/same-session protocol.

The repository's existing QNN loader remains SDK-free and only checks dynamic
library/symbol presence on supported targets. It does not call an entry point or
declare guessed QNN ABI layouts. Functional graph construction must start from
the exact headers shipped by the owner-approved SDK version.

Unblock contract:

1. Owner obtains the Qualcomm AI Runtime SDK and records EULA acceptance and
   the exact version.
2. `QNN_SDK_ROOT` points at that installation and `QnnInterface.h` plus the HTP
   headers are available.
3. A Snapdragon/Hexagon device reachable from the build host supplies matching
   QNN runtime libraries and profiling tools.
4. CC transcribes only the used ABI from those headers, implements the complete
   Whisper encoder graph, and verifies tensor-level parity.
5. The release harness alternates first-party CPU and QNN on that same device,
   while the QNN profile proves at least 90% HTP placement.
