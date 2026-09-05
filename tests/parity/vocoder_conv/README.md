# Vocoder convolution parity fixtures

These fixtures are the independent PyTorch reference for the bounded vocoder
convolution seam. They were generated on VAST with the exact command below,
using `tools/parity/vocoder_conv_dump_reference.py` at commit
`7377bf3aea622318e0d91972e457c13067e16f7b`:

```sh
UV_CACHE_DIR=/tmp/vokra-htdemucs-uv-cache uv run --frozen --project tools/parity --python 3.12 python tools/parity/vocoder_conv_dump_reference.py --output /workspace/vokra-vocoder-conv-fixture-7377bf3a
```

The reference runtime was PyTorch `2.13.0+cu130`. The dumper calls
`torch.nn.functional.conv1d` and
`torch.nn.functional.conv_transpose1d` directly; it does not import Vokra,
load a checkpoint, or use a model. Inputs, weights, and biases are deterministic
signed powers of two, with no RNG. Their products and accumulations remain in
the exact integer range of binary32, so the Rust parity tests require exact
(`atol = 0.0`) output equality.

The outer `manifest.json` SHA-256 is:

```text
b438a28b6bc64754dc119d186080749775a78ad2b9345a5f20f480cdbcaa0c07
```

| File | SHA-256 |
| --- | --- |
| `conv1d_d2_s2_p2_input.f32` | `a4f56cd83fd32de6408ef57ae00485a5011d5b116a02fba42a9484007e15c023` |
| `conv1d_d2_s2_p2_weight.f32` | `90c78d26b89e56e561d5d0fa6992c0d002908ca0136dde9a2165b994f9864ee7` |
| `conv1d_d2_s2_p2_bias.f32` | `f4505f8d66bf6d6b09e944e5f7127218c22ee22f42bdd01d14d69405bd1d4fe8` |
| `conv1d_d2_s2_p2_output.f32` | `5f14f1b6062da2d3a33f00d3cb965ec2591fb97476e56ce69c3414df589e9c81` |
| `conv_transpose1d_s3_p1_op2_input.f32` | `49f0296cc1b3ff9563e72140673b29d59552002c473f8a09bb1fdb5f5ad23b5e` |
| `conv_transpose1d_s3_p1_op2_weight.f32` | `7ef81d6eb98996dbe5ee03afc30a6ba691fa17cf21013162638a593bd54ec9e6` |
| `conv_transpose1d_s3_p1_op2_bias.f32` | `dc7d7b60851f628068c49c6fda93fb304ae80f8a9745bc1f5c62a63edc660e3b` |
| `conv_transpose1d_s3_p1_op2_output.f32` | `dd091a038d1cb21a2398c95eae1f53c5a1cf85ce2e880e3659e9c15ec192a443` |

The manifest records all tensor shapes, byte counts, dtypes, and hashes. The
CPU and Metal tests independently verify those fields before comparing their
new convolution seams against the recorded oracle output. No model weights
are included in this fixture set.
