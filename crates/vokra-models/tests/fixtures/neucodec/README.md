# NeuCodec official decode fixtures

The `base/` and `distill/` fixtures are generated independently by
`tools/parity/neucodec/dump_reference.py`. The dumper verifies the official
`neuphonic/neucodec` source at commit
`ed3e6cd1bdc374ce14a21355e5eee66a777149ce`, each public GGUF SHA-256,
`torchtune==0.3.1` plus its official RoPE source hash, and
`vector-quantize-pytorch==1.17.8`. It restores the GGUF weights into the
official `CodecDecoderVocos` module and calls the upstream FSQ and decoder
forward methods.

The four-code fixtures are deliberately small enough to commit while still
executing FSQ projection, all four ResNet blocks, all twelve Transformer
blocks, the magnitude/phase head, and same-padded iSTFT. The base GGUF is
larger than 2 GB, so regeneration and its Rust real-weight test belong on
VAST. Distill Metal parity may run on the maintainer Mac because that artifact
is below the repository's 2 GB threshold.
