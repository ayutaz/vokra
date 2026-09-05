# Conv2d / ConvTranspose2d reference fixtures

This directory is reserved for committed, offline PyTorch reference fixtures
for the CPU Conv2d seams. The generator calls the official
`torch.nn.functional.conv2d` and `torch.nn.functional.conv_transpose2d`
implementations directly; it does not import or mirror Vokra code.

The fixture set is not generated in this checkout yet. Do not add fabricated
outputs, zero digests, or digest placeholders. Generate it on VAST/Scaleway
with the pinned `tools/parity` environment:

```text
uv run --frozen --project tools/parity --python 3.12 python \
  tools/parity/conv2d_dump_reference.py --output tests/parity/conv2d
```

The generator writes three deterministic cases:

- grouped/dilated Conv2d with asymmetric stride and padding;
- grouped/dilated ConvTranspose2d with asymmetric stride, padding, and
  output-padding;
- the ATen edge case `stride=1, dilation=2, output_padding=1`, which is valid
  because output-padding is smaller than dilation.

Each tensor is a raw little-endian IEEE-754 binary32 file named
`<case>_<role>.f32`. `manifest.json` records exact shapes, byte lengths, and
SHA-256 digests for `input`, `weight`, `bias`, and `output`; `manifest.sha256`
pins the manifest itself. The Rust integration test rejects missing files,
symlinks, schema drift, digest mismatch, and uncommitted extra files. It
requires bit-for-bit f32 equality for CPU-vs-PyTorch comparison: the
deterministic signed-power-of-two products and sums stay below the exact
binary32 integer range, so no tolerance is selected before observing a
reference result.

After remote generation, commit the complete directory (including the two
manifest files) and run the ignored integration test explicitly:

```text
CARGO_BUILD_JOBS=1 cargo test -p vokra-backend-cpu --test conv2d_torch_parity -- --ignored
```

The command above is a post-generation verification command; it must not be
run on the maintainer machine under the local large-model/Cargo policy.
