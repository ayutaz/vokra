# Installing the Vulkan shader-compile toolchain (developer-side, M3-02-T13)

`scripts/compile-vulkan-shaders.sh` needs **`glslc`** (from the Vulkan SDK or a
distro `glslang` package) plus a `sha256` tool (`sha256sum` on Linux, `shasum`
on macOS — both preinstalled). This is a developer-side tool only:
`cargo build` never invokes it (NFR-DS-02: `Cargo.lock` stays `vokra-*`-only;
NFR-RL-05: no CPU-side JIT — SPIR-V is produced ahead of time by the developer
and committed to `crates/vokra-backend-vulkan/kernels/precompiled/*.spv`).

CI is out of scope for this ADR: T36 follow-up will add a `glslc install` +
`recompile → diff` gate.

---

## macOS (Apple Silicon or Intel)

Two supported paths — pick one.

### Option 1: Homebrew (recommended for M1 iMac authoring host)

```bash
brew install glslang
```

Verify:

```bash
glslc --version
# Expected output starts with:
#   shaderc v2024.x
#   spirv-tools v...
#   glslang ...
#   Target: SPIR-V 1.x
```

Homebrew installs `glslc` on `PATH` as a peer of `glslangValidator`. No
environment variables needed. This does NOT install a Vulkan **loader** or
ICD — Metal is the macOS GPU path in Vokra; `glslc` is used only to produce
`.spv` bytecode for the Vulkan **backend** which runs on Linux / Android /
Windows CI runners and real devices.

### Option 2: LunarG Vulkan SDK (matches Windows / Linux setup)

Download the macOS installer from <https://vulkan.lunarg.com/sdk/home#mac>,
run it, then source the SDK's environment script (path varies by version):

```bash
source ~/VulkanSDK/1.4.xxx.x/setup-env.sh
glslc --version
```

This installs both `glslc` and the MoltenVK ICD (a Vulkan-on-Metal
translator). Vokra does NOT use MoltenVK for the Metal backend, but MoltenVK
lets you dry-run the smoke dispatch test on macOS if desired (see
`crates/vokra-backend-vulkan/tests/smoke_dispatch.rs`).

---

## Ubuntu / Debian

`glslang-tools` ships `glslc`:

```bash
sudo apt update
sudo apt install glslang-tools
glslc --version
```

For a Vulkan loader + lavapipe (the CPU-only ICD used in CI), also install:

```bash
sudo apt install libvulkan1 mesa-vulkan-drivers
```

CI-runner reproducibility: pin the Ubuntu release (`ubuntu-22.04` or later —
`glslang-tools` on 22.04 provides SPIR-V 1.6 compatible with our `vulkan1.1`
target). Older versions may lack `--target-env=vulkan1.1` support; if
`compile-vulkan-shaders.sh` errors on that flag, upgrade.

---

## Windows

Download the LunarG Vulkan SDK installer from
<https://vulkan.lunarg.com/sdk/home#windows>. Install with defaults; the
installer sets `%VULKAN_SDK%` and adds `%VULKAN_SDK%\Bin` to `PATH`.

Verify from PowerShell:

```powershell
glslc --version
```

If `glslc` is not found after install, restart the terminal (PATH refresh) or
manually add `%VULKAN_SDK%\Bin` to your user PATH. On WSL2, use the Ubuntu
instructions above — do NOT try to run the Windows `glslc.exe` from inside
WSL.

---

## Verifying end-to-end

From the repo root:

```bash
scripts/compile-vulkan-shaders.sh
```

The script:

1. Fails fast if `glslc` is missing (printing an OS-specific install hint).
2. Recompiles every `crates/vokra-backend-vulkan/kernels/glsl/*.comp`.
3. Writes `crates/vokra-backend-vulkan/kernels/precompiled/SHA256SUMS`.
4. Prints one line per compiled shader.

After a successful run, paste each shader's SHA-256 into the corresponding
`SpirvShader::expected_sha256_hex` entry in
`crates/vokra-backend-vulkan/src/spirv.rs` and run:

```bash
cargo test -p vokra-backend-vulkan verify_pinned_hashes_is_ok
```

to confirm the manifest matches the byte content of the committed `.spv`.

---

## Why not just `cargo install glslc` or add `shaderc-rs`?

Both would break the zero-dependency invariant (NFR-DS-02) — the workspace's
`Cargo.lock` MUST stay `vokra-*`-only. `deny.toml` explicitly bans
`shaderc-rs` / `shaderc` / `spirv-tools` / `glslang-sys` / `naga` / `ash` /
`vulkano` / `erupt`. `glslc` is a developer-side one-shot toolchain,
committed `.spv` blobs are the artifact `cargo build` consumes — no compile-
time toolchain dependency, no runtime JIT (NFR-RL-05, Android SELinux W^X
constraint).
