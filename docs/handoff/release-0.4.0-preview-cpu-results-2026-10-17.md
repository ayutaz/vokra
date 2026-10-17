# v0.4 preview代表CPU検証の結果 — 2026-10-17

## 結論と範囲

元のrelease準備工程1〜3（対象固定、版・変更履歴整理、代表機能の再検証）は、
以下の限定したsource／model scopeで完了した。公開版の作成ではない。
**Linux CPUのWhisper base ASRとSilero VAD v5の機能smokeは実モデルで成功。**
独立数値parity、WER／CER、性能比較、Mac CPU／Metal、全catalogの完成を意味しない。

## SourceとPR受理

- 準備[PR #261](https://github.com/ayutaz/vokra/pull/261)はhead
  `09c610b7186ca4965eb337541353347b62ced3b9`、base `d15ce4cb`。
- exact-head check runsは70 SUCCESS、意図的な`gitleaks full history`一件だけSKIPPED。
  strict必須16 context/application pairは全て成功（Actions 15368／外部CodeQL 57789）。
  十workflow全てattempt 1で成功。管理者bypassなしで通常squash mergeした。
- mergeは2026-10-17 11:05:47 UTC、main
  `9ee5834d56249fd7d7cc243c44cf01acb4951997`。ローカルもclean mainへ同期し、
  レビュー済みPR headとのtree差分がないことを確認した。
- 通常pre-commitの五gate成功。Silero fixture ledger読み取りは別途明示許可済み。
  通常pre-pushのcompliance回帰八件成功、既存のRust-build-neutral軽量経路を使用。
  hook・安全設定・owner manifest・fixtureは変更していない。

## 固定モデル情報とsource回帰

[model-free source run 39025886152](https://github.com/ayutaz/vokra/actions/runs/39025886152)
は17 smoke-driver＋7 metadata testsをskipなしで成功。固定revisionのAPI JSONのみを
取得し、weight／config／tokenizer／Cargo実行は含めていない。
PR test-merge `ca5f87a0c47ec851860f8924cb638e442aedcbe0`の結果であり、
実モデルrunのmain HEADと混同しない。

回収物は`pins.json`と`metadata-receipt.json`だけ（archive 1099 bytes）。
rootがrun/source identity、exact二repo／revision、filename、digest、sizeを独立照合した。

- pins SHA-256: `e5398feba215f7f5b026b2e2572d36489eec58684375799f3417ec4bb9b2e6e4`
- metadata receipt SHA-256: `ed1d6ca92db94377b2e23924ed9667454b1783132dc8c8d3b13972fb872ff15b`
- GGUF合計293,043,104 bytes。実weightのSHA／sizeも後段runで一致した。
  個別repo／revision／digestは下の回収receiptに保持する。

ローカルの新規Python回帰と公開JSON参照は、明示許可後も安全hookに拒否された。
迂回や例外追加をせず、許可済みhosted CIへ移管した。ローカルでのモデル実行はない。

## 実CPUの観測結果

唯一の手動[run 39027424216](https://github.com/ayutaz/vokra/actions/runs/39027424216)
はattempt 1で成功。jobは11:07:23〜11:09:49 UTC、2分26秒で、許可上限30分以内。
これはcheckout／build／download等を含むjob時間であり、推論速度やRTFのbenchmarkではない。
CPU-only CLIを`--locked --release -p vokra-cli --no-default-features`でbuildし、
各コマンドで`--backend cpu`を指定した。protected二treeはcheckoutから除外した。

| native case | 観測したstdout／結果 |
| --- | --- |
| Whisper JFK | `And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country.` |
| Silero JFK | 343 frames、speech 214、mean probability 0.5889 |
| Silero同duration無音 | 343 frames、speech 0、mean probability 0.0025 |

三コマンドともexit 0、stderr空。JFKは16kHz mono PCM16／176,000 samples＝11秒。
ファイル名の30sは実clip長ではない。入力hash、生成無音のhash／geometry、実モデルの
hash／size、HEAD、argv、stdout、判定をrootが回収receiptから独立再照合した。
無音比較はlabel付き精度評価でも独立upstream照合でもない。

## 証跡と後処理

回収artifactはJSON receipt一件のみ（archive 1359 bytes）。モデル／音声は回収・公開していない。
元receipt SHA-256は
`92a43ccc788e2c576884cedd0b5388cb7ddaee9579408d82b18c7d17ec4ff3c5`。
以下に原本JSONを保存し、短期artifactの期限だけに依存しないようにする。

モデルはworkerのtemporary directoryで処理し、正常終了時に削除する実装。
全job/post-stepの終了を確認した。VAST／Scalewayの有料instanceやstorageを今回は
作成していない。既存provider全accountのinventoryを監査したという意味ではない。
HF/provider key、`.env`は参照せず、モデル公開・Release・tag・X投稿も行っていない。

## 残るrelease／告知工程

次版候補は`v0.4.0-rc.1`、workspaceはまだ0.3.0。次にversion/path/binding同期、
changelog roll、配布物の実buildと導入確認、pre-release／Latest設定の確認が必要。
この二モデルの成功から他モデル・TTS・Apple・全catalogへ拡張しない。
この結果記録は管理文書だけの変更で、実行対象は上記9ee5834dのままである。

[speech.cpp等との比較と告知方針](competitive-positioning-2026-10-17.md)に従い、
Rust-native、デフォルトruntimeの外部Cargo依存ゼロ、音声処理・stateの共通境界を
説明する。速度優位、全136モデル動作、Metal完全対応、production-readyは主張しない。
公開版と実配布物が未作成のため、公開済みpreviewとは告知しない。

## 回収原本JSON

```json
{
  "audio": {
    "bytes": 352078,
    "geometry": {
      "channels": 1,
      "frames": 176000,
      "sample_rate": 16000,
      "sample_width_bytes": 2
    },
    "path": "/home/runner/work/vokra/vokra/tests/fixtures/audio/jfk-30s.wav",
    "sha256": "58adb4ea501d955fcd40bfbb69128f8f40428b81d8716b9ed337949773be253f"
  },
  "commands": [
    {
      "argv": [
        "/home/runner/work/vokra/vokra/target/release/vokra-cli",
        "run",
        "--model",
        "/tmp/vokra-release-cpu-smoke-r6vcn4zl/whisper-base.gguf",
        "--input",
        "/home/runner/work/vokra/vokra/tests/fixtures/audio/jfk-30s.wav",
        "--backend",
        "cpu"
      ],
      "label": "asr-jfk",
      "returncode": 0,
      "stderr": "",
      "stdout": "asr:  And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country.\n"
    },
    {
      "argv": [
        "/home/runner/work/vokra/vokra/target/release/vokra-cli",
        "run",
        "--model",
        "/tmp/vokra-release-cpu-smoke-r6vcn4zl/silero-vad-v5.gguf",
        "--input",
        "/home/runner/work/vokra/vokra/tests/fixtures/audio/jfk-30s.wav",
        "--backend",
        "cpu"
      ],
      "label": "vad-jfk",
      "returncode": 0,
      "stderr": "",
      "stdout": "vad: 343 frames, speech_frames=214, mean_prob=0.5889\n"
    },
    {
      "argv": [
        "/home/runner/work/vokra/vokra/target/release/vokra-cli",
        "run",
        "--model",
        "/tmp/vokra-release-cpu-smoke-r6vcn4zl/silero-vad-v5.gguf",
        "--input",
        "/tmp/vokra-release-cpu-smoke-r6vcn4zl/silence.wav",
        "--backend",
        "cpu"
      ],
      "label": "vad-silence",
      "returncode": 0,
      "stderr": "",
      "stdout": "vad: 343 frames, speech_frames=0, mean_prob=0.0025\n"
    }
  ],
  "functional_smoke_only": true,
  "generated_silence": {
    "bytes": 352044,
    "channels": 1,
    "frames": 176000,
    "sample_rate": 16000,
    "sample_width_bytes": 2,
    "sha256": "1a092b23d9dfe64d53943d18262c0a49eaf32c5ab5f2b016cd8edd16ec9440dd"
  },
  "models": [
    {
      "actual_bytes": 290859360,
      "actual_sha256": "7e774425585d6e9ba58ac5337b406522b1408ab6ff6765d2938b092aef4c8e27",
      "expected_bytes": 290859360,
      "expected_sha256": "7e774425585d6e9ba58ac5337b406522b1408ab6ff6765d2938b092aef4c8e27",
      "filename": "whisper-base.gguf",
      "name": "whisper-base",
      "repo": "vokra/whisper-base",
      "revision": "dddd93ff6301a82aa3750d398cb2e0a599cd9e83",
      "url": "https://huggingface.co/vokra/whisper-base/resolve/dddd93ff6301a82aa3750d398cb2e0a599cd9e83/whisper-base.gguf"
    },
    {
      "actual_bytes": 2183744,
      "actual_sha256": "23b19eadba72fb9726fa73a0bfd095afcba2f8c5bcb4f03a1b5e57a67eb13df6",
      "expected_bytes": 2183744,
      "expected_sha256": "23b19eadba72fb9726fa73a0bfd095afcba2f8c5bcb4f03a1b5e57a67eb13df6",
      "filename": "silero-vad-v5.gguf",
      "name": "silero-vad-v5",
      "repo": "vokra/silero-vad-v5",
      "revision": "3670dfed054980307eeb1711c6329c29b80bdf14",
      "url": "https://huggingface.co/vokra/silero-vad-v5/resolve/3670dfed054980307eeb1711c6329c29b80bdf14/silero-vad-v5.gguf"
    }
  ],
  "not_metal_or_performance": true,
  "not_numerical_parity": true,
  "reviewed_pins": {
    "silero-vad-v5": {
      "filename": "silero-vad-v5.gguf",
      "repo": "vokra/silero-vad-v5",
      "revision": "3670dfed054980307eeb1711c6329c29b80bdf14",
      "sha256": "23b19eadba72fb9726fa73a0bfd095afcba2f8c5bcb4f03a1b5e57a67eb13df6",
      "size_bytes": 2183744,
      "url": "https://huggingface.co/vokra/silero-vad-v5/resolve/3670dfed054980307eeb1711c6329c29b80bdf14/silero-vad-v5.gguf"
    },
    "whisper-base": {
      "filename": "whisper-base.gguf",
      "repo": "vokra/whisper-base",
      "revision": "dddd93ff6301a82aa3750d398cb2e0a599cd9e83",
      "sha256": "7e774425585d6e9ba58ac5337b406522b1408ab6ff6765d2938b092aef4c8e27",
      "size_bytes": 290859360,
      "url": "https://huggingface.co/vokra/whisper-base/resolve/dddd93ff6301a82aa3750d398cb2e0a599cd9e83/whisper-base.gguf"
    }
  },
  "runtime": {
    "expected_head": "9ee5834d56249fd7d7cc243c44cf01acb4951997",
    "github_actions": "true",
    "head": "9ee5834d56249fd7d7cc243c44cf01acb4951997",
    "marker": "1",
    "platform": "linux"
  },
  "schema": 1,
  "semantic_checks": {
    "asr_jfk_keyphrase": {
      "normalized_text": "and so my fellow americans ask not what your country can do for you ask what you can do for your country",
      "passed": true
    },
    "vad": {
      "jfk": {
        "frames": 343,
        "mean_prob": 0.5889,
        "speech_frames": 214
      },
      "passed": true,
      "silence": {
        "frames": 343,
        "mean_prob": 0.0025,
        "speech_frames": 0
      }
    }
  },
  "status": "PASS"
}
```
