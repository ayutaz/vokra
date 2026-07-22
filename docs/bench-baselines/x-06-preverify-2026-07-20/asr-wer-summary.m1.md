### X-06 nightly ASR-WER (LibriSpeech dev-clean)

**PASS: WER 4.3689% <= threshold 6.0000%**

model: `whisper-base`  
utterances: 8 (campaign-8)  
reference words: 206

| metric | value |
| --- | --- |
| WER | 4.3689% |
| CER | 1.8325% |
| word edits | 9 (sub 7 / del 1 / ins 1) |
| threshold | 6.0000% |
| baseline (`campaign-8`) | 4.3689% |
| delta vs baseline | +0.0000% |

<details><summary>per-utterance</summary>

| utterance | WER | CER | ref words |
| --- | --- | --- | --- |
| `1272-128104-0000` | 0.0000% | 0.0000% | 17 |
| `1272-128104-0002` | 6.2500% | 3.4884% | 32 |
| `1272-128104-0003` | 8.0000% | 3.7313% | 25 |
| `1272-128104-0005` | 0.0000% | 0.0000% | 18 |
| `1272-128104-0007` | 5.2632% | 2.0202% | 19 |
| `1272-128104-0009` | 9.3023% | 3.1873% | 43 |
| `1272-128104-0011` | 0.0000% | 0.0000% | 34 |
| `1272-128104-0013` | 0.0000% | 0.0000% | 18 |

</details>

_Advisory leg (X-06). Not a required check; a breach is an investigate/revert signal per NFR-MT-07, not a merge blocker._
