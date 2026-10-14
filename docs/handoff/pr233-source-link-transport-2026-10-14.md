# PR #233 source-link transport reconciliation — 2026-10-14

## Actual failure and bounded correction

At head `71cc20853d4bbf81d8355e20d580a9e921c8b192`, documentation-links job
`115727679705` in run `38526404211` reports 1,104 URLs and 13 error entries,
all existing GitHub blob pages returning HTTP 503. The unchanged checker has
`MAX_FALLBACK_URLS=10` and rejects before any API recovery. Luna's offline
reproduction identifies `fallback URL limit exceeded`, not an API failure.
The original failed result remains failed; there is no blind rerun.

Eleven affected file citations now use the same official raw host, repository,
ref and path. The full commit for historical GGUF prefix `eced84c86f8b` is
`eced84c86f8b012c752c016f7fe789adea168e1e`, verified through the official
commit endpoint. This expands an existing source identity, not a new revision.
The two VibeVoice line-anchor citations stay unchanged: their exact file/line
contract is supported by the existing authenticated recovery gate.

M2D's historical API response quotation is preserved byte-for-byte as literal
inline code, with a separate live raw PDF citation. Do not rewrite a measured
API quotation to pretend it originally returned the raw URL. The PDF is not
passed to a UTF-8 source validator, and its license is not newly classified.

No URL exclusion, accepted error status, fallback-count/deadline increase,
workflow change, checker change or protection override is introduced.
The license rows, owner decisions, dates, scope hashes, reference numerics
and source contracts remain unchanged. The original `master`, `dev`, `main`
and `4.9.3` aliases retain their historical mutability; this does not silently
freeze them or restamp an old license review.

## Primary transport identity readback

A credential-free three-thread read-only probe used a shared 60-second deadline,
eight-second requests and one-MiB body caps. For each citation, the official
raw bytes returned HTTP 200 and matched the official Contents API's size and
Git blob SHA-1. Bodies stayed in memory; no model/tokenizer/weight or local
model execution was involved. These are transport receipts, not license,
execution, model or numerical approvals.

| Same source path/ref | Bytes | Git blob SHA-1 |
|---|---:|---|
| M2D `master/LICENSE.pdf` | 93,913 | `69a5ef7929a35d26b766375e93c61842b863f77c` |
| piper-plus `dev/README_EN.md` | 21,503 | `2c7f4b09f76e8d95e58d00a0234c6d1858bf6e38` |
| BigVGAN `main/LICENSE` | 1,076 | `4c78361c86d4f685117d60d6623e2197fcfed706` |
| Vokra `fe1f5fe2` XCodec2 disposition | 33,887 | `42cfce1dbb3f26db4f98751575fc0aa6d6859749` |
| Spark-TTS `2f1ea908/LICENSE` | 11,357 | `261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64` |
| FireRed `834635e4` conformer encoder | 12,651 | `b41e68e6eb2bed90611e65c6bc5e2025dc6753df` |
| ANTLR `4.9.3/LICENSE.txt` | 2,699 | `2042d1bda6c933e504d9dc2fe3197a6e42a71fe2` |
| Candle `cd96fa80/LICENSE-MIT` | 1,023 | `31aa79387f27e730e33d871925e152e35e428031` |
| DSM `4c4f65e1` PyTorch caller | 8,452 | `cf3fb05b0e0c1f265a667276d2886ce2664d79ff` |
| Moshi `e6a55d27/moshi/LICENSE` | 1,023 | `31aa79387f27e730e33d871925e152e35e428031` |
| GGML `eced84c86f8b012c752c016f7fe789adea168e1e/docs/gguf.md` | 43,169 | `13e5334b899ee646288bb3a1540546e8410b88c8` |

The PDF begins with `%PDF-`; transport identity does not establish permission
to use or distribute M2D weights. Its fail-closed unknown-license row remains.
Existing primary citation locations are retained in their original documents.

Fresh exact-head CI must validate the corrected documents before normal merge.
PR #232's separately corrected dependency gate also awaits fresh CI and the
accepted documentation baseline. No cloud lease or model row is promoted.

## Subsequent exact-head failure and readback

At `53a9e68efb3c06ca12926f05e76918b02171e4c3`, job `115738841213`
(run `38529849478`) reports four entries: the HF organization page twice with
HTTP 500, and two other existing GitHub file citations with HTTP 503. The
advisory coverage job `115738981688` separately fails on HTTP 500. These
remain failed results, not green checks or a completed live inventory.

The two additional citations now use the same official raw source/ref/path.
Credential-free readback authenticated both against official Contents API
size and Git blob identity, without storing source bodies or acquiring models:

| Same source path/ref | Bytes | Git blob SHA-1 |
|---|---:|---|
| Candle `cd96fa80/LICENSE-APACHE` | 11,357 | `261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64` |
| Moshi `e6a55d27/moshi/moshi/models/lm.py` | 37,312 | `209b7a59c9c086810a81a7d8b99c5233a0ad87ff` |

Both raw requests returned 200. The unchanged HF organization page and the
bounded model-list API subsequently returned 200 as well; this verifies
service recovery only, not all catalog rows. No HF URL is excluded and no
HTTP error is accepted. Fresh CI still controls ordinary acceptance.
