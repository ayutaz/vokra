# ProjectSettings

This project is committed **without** the generated `ProjectSettings/*.asset`
files (including `ProjectVersion.txt`). Per the M0-10-T01 spec, the Unity
version is **not invented here** — it is fixed to your environment:

1. Open `examples/unity-demo/` in your **Unity LTS** editor. Unity generates
   `ProjectSettings/` (and `ProjectVersion.txt` recording your exact version) on
   first open.
2. Commit the generated `ProjectSettings/` afterwards so CI/other machines pin
   the same version (recommended: a current LTS, e.g. 2021 LTS or 2022 LTS).

The demo's design does not depend on a specific Unity version: it uses only
standard modules (see `Packages/manifest.json`), an IMGUI (`OnGUI`) UI, and the
Vokra C ABI. Mono is the default scripting backend; the callback pattern
(`Assets/Scripts/Vokra/VokraCallbacks.cs`) is already IL2CPP-safe (NFR-RL-02).
