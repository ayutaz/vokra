# Assets/Plugins — native Vokra library

The Vokra runtime is the `vokra-capi` cdylib (`libvokra.dylib` / `libvokra.so` /
`vokra.dll`). It is **not committed** (it is a build artifact); place it with:

```sh
# run on each OS (no cross-compile) — see scripts/build-unity-plugin.sh
bash ../../scripts/build-unity-plugin.sh
```

That copies the library into the per-platform folder Unity expects:

| OS | folder | file |
|----|--------|------|
| macOS | `Assets/Plugins/macOS/` | `libvokra.dylib` |
| Windows (x64) | `Assets/Plugins/Windows/x86_64/` | `vokra.dll` |
| Linux (x64) | `Assets/Plugins/Linux/x86_64/` | `libvokra.so` |

`DllImport("vokra")` (see `Assets/Scripts/Vokra/NativeMethods.cs`) resolves to
these names on the respective OS.

**Windows without a Rust toolchain**: fetch the `vokra.dll` built by the M0-01 CI
Windows runner (`cargo build --release -p vokra-capi`) and drop it into
`Assets/Plugins/Windows/x86_64/`.

After placing a library, open the project in Unity once and set the plugin's
platform import settings (target OS/CPU) in the Inspector, then commit the
generated `.meta` if you want the settings tracked. iOS static linking
(`DllImport("__Internal")`, NFR-RL-03) is v0.5 official-plugin scope (FR-API-04),
not part of this demo.
