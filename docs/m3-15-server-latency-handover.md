# M3-15 vokra-server 75 ms TTS Latency Measurement — Handover

**Owner**: 依頼者 (実機 GPU 環境での reference 計測は本質的に依頼者側; CC が採取した in-process bench 参考値は既に land 済 = `integrations/vokra-server/benches/tts_latency.rs`).
**Predecessor**: M3-15 = 100% CC 完成 (Wave 3.5 T01-T14 scaffold + Wave 11 Whisper beam server surface + Wave 12 core-side `no_repeat_ngram_size` plumbing、`docs/tickets/m3/M3-15-vokra-server-multi-session.md`)。
**Requirement under measurement**: NFR-PF-05 (サーバ TTS レイテンシ 75 ms、v1.0 値)。

> **Explicit boundary**: 本 handover は **reference 環境での参考値採取** に限定する。**実機 GPU での 75 ms 実測 always-on gate は X-06 nightly self-hosted matrix で継続監視** (`docs/tickets/m3/M3-15-vokra-server-multi-session.md` L61 = "実機 GPU での 75ms 実測ゲートは X-06 nightly self-hosted matrix で継続監視、本 WP は CC 側の実装計測 + 参考記録に留め、未達時は四半期 review (NFR-MT-05) へ申し送り")。M2-14 と同型の分担: 一回性判定は依頼者 lab、continuous 監視は X-06、本 WP は CC 側実装 + 参考値。

## 1. Prerequisites checklist

- [ ] **Client machine + server machine が同一 LAN** (loopback で計測する場合は 1 台で完結、network noise 除去)。
- [ ] **`vokra-server` binary** — `cargo build --release -p vokra-server` (excluded workspace、独自 Cargo.lock)。
- [ ] **`whisper-base.gguf`** (ASR side) — MIT weight。
- [ ] **piper-plus voice GGUF** (TTS side) — 依頼者作 MIT。M3-12-T14 sanity 済 voice を使用可能。
- [ ] **`vokra-cli`** — `cargo build --release -p vokra-cli` (root workspace)。
- [ ] **reference audio fixture** — `tests/fixtures/audio/jfk-30s.wav` (ASR)、TTS の reference utterance は `benches/tts_latency.rs` の `SHORT_UTTERANCE` = "Hello world" 相当 (spec 側で utterance を fix しない場合は本 handover の §4 で明示)。
- [ ] **reference 環境**: M2-09-T18 (M2 で確定した 90 ms 参考測定) と **同一 reference 環境**を使用する。**具体的なハードウェア・GPU 型番は本 handover でも断定しない** (ハルシネーション禁止 = CLAUDE.md)。M2-09-T18 の記録を参照する。

## 2. In-process reference bench (CC-side、既 land)

CC 側で採取済の参考値: `integrations/vokra-server/benches/tts_latency.rs` = harness = false + non-Criterion + zero external deps。

```bash
cd integrations/vokra-server
cargo bench --bench tts_latency 2>/dev/null
```

出力 (stdout に JSON blob emit、CI が build artifact として capture):
```
{"boundary":"http_end_to_end","short_utterance":"Hello world",
 "iterations":100,"warmup":10,
 "median_us":..,"p50_us":..,"p95_us":..,"max_us":..,"budget_ms":90}
```

**注意**:
- `budget_ms=90` は NFR-PF-05 の **v0.5 値** (M2-09-T18 で確立)。v0.9 では `budget_ms=75` を目標とする (下記 §4)。
- bench は `FakeSynth` (deterministic in-memory PCM generator) を使用 → engine backend 独立の schema decode + dispatch + WAV encode オーバーヘッドの **floor** を測る (real engine はこれより遅くなる方向)。
- 実 Metal/CUDA 計測は本 bench の scope 外 (`benches/tts_latency.rs` L58-72 の "What this harness does NOT do")。

## 3. Server bring-up (実機計測用)

Server:
```bash
./target/release/vokra-server \
    --http-bind 127.0.0.1:8080 \
    --model whisper=/path/to/whisper-base.gguf \
    --model piper=/path/to/piper-voice.gguf \
    --backend metal    # or cuda / cpu — FR-EX-08: unavailable backend は起動時 explicit error
```

**Backend 選択の意味論** (FR-EX-08):
- `--backend metal` は M1 iMac (M3-12-T14 sanity 済み)、`--backend cuda` は vast.ai RTX 4090 or self-hosted。
- 非対応 backend を指定した場合は起動時に `BackendUnavailable` explicit error で終了、silent CPU fallback しない。
- **CPU baseline は reference として計測しておく**と regression 判定が容易。

## 4. TTS レイテンシ計測 (実 HTTP client、client-side は現時点 owner-side)

**Honest state**: `vokra-cli bench` は現時点 in-process の GGUF ベンチ (`--model` / `--input` / `--text` / `--iters` / `--warmup` / `--format` / `--baseline` / `--backend` / `--task` フラグ) のみサポートし、**`--server` / `--endpoint` / `--concurrent` フラグは未実装** (`crates/vokra-cli/src/bench.rs::parse_args`)。M3-15-T11 の ticket spec 上は "vokra-cli bench を multi-session server mode に拡張" と書かれているが、実 land は `integrations/vokra-server/benches/tts_latency.rs` に in-process bench として着地しており、実 HTTP round-trip 計測は分離されている。

したがって v0.9 参考採取は以下の 3 経路のいずれか (推奨は A):

### A. In-process reference bench を primary artifact として採用 (推奨、CC land 済)

`benches/tts_latency.rs` (§2 参照) の JSON blob を v0.9 reference として記録。**FakeSynth 経由ゆえ engine backend 独立の floor 値**であり、実 engine (Metal / CUDA) の値ではない旨を明記する。

### B. `curl` + timing loop で HTTP レイテンシ実測 (owner-side、最小工数)

```bash
# server 側で /api/tts を LISTEN (§3 参照)
# client 側で timing loop
for i in $(seq 1 100); do
    /usr/bin/time -f '%e' curl -s -o /dev/null -w '%{time_total}\n' \
        -X POST http://127.0.0.1:8080/api/tts \
        -H 'Content-Type: application/json' \
        -d '{"text":"Hello world","voice":"en_US-libritts-high"}' \
    2>/dev/null
done | tee tts_curl_100.tsv

# quantiles (awk で p50/p95/p99)
sort -n tts_curl_100.tsv | awk '{
    a[NR]=$1
} END {
    n=NR; asort(a);
    printf "p50=%.4fs p95=%.4fs p99=%.4fs max=%.4fs\n",
        a[int(n*0.5)], a[int(n*0.95)], a[int(n*0.99)], a[n];
}'
```

`curl` は真の TTFA (Time To First Audio byte) を測るには `-w '%{time_starttransfer}'` を使う (final byte を測るなら `%{time_total}`)。**両方採取して報告するのが正確**。

### C. `vokra-cli bench` に server mode を CC follow-up で追加 (M4 スコープ、未着手)

M3-15-T11 の "server mode 拡張" は実装未完 = 現時点は in-process bench に着地。**follow-up として `--server` / `--endpoint` / `--concurrent` を追加する CC 実装は Investigation を再実行して "CC-implementable" 判定に載せることは可能** (工数見積 = 4〜8 時間、reqwest 等の HTTP client crate を root workspace 非干渉 = excluded workspace の追加で対応、zero-dep NFR-DS-02 を破らない設計が必要)。

**判定基準 (NFR-PF-05 v0.9)**:
- **PASS reference (経路 A)**: `benches/tts_latency.rs` の `http_end_to_end.median_us` が `75_000` (= 75 ms) 未満。**FakeSynth 経路ゆえ floor 値であり実 engine 値ではない**旨を報告に明記。
- **PASS 実測 (経路 B)**: `curl -w '%{time_starttransfer}'` (TTFA) の p50 が 75 ms 未満。
- **未達**: 未達値をそのまま記録・公開 = 新規閾値を発明しない (M2-14 と同運用)。四半期 review (NFR-MT-05) に申し送り。

## 5. Multi-session smoke (FR-SV-06)

**FR-SV-06 の 5 点意味論** (`docs/tickets/m3/M3-15-vokra-server-multi-session.md` L24):
- (a) 同時複数リクエストを paged KV cache = M3-03 の 3D 論理アドレス [time, stream, codebook] で分離
- (b) 単一 GPU/CPU リソースを time-slice / batching 相当で share
- (c) HTTP endpoint 側で connection pooling / request queue を提供
- (d) 過負荷時の graceful degradation (明示エラー、silent fallback 禁止 = FR-EX-08)
- (e) TTS TTFA 75ms を P50/P95/P99 で採測

**Server 側の CC land 済 実装** (`integrations/vokra-server/src/scheduler.rs`):
- `SchedulerConfig::max_concurrent_sessions` = tokio `Semaphore` permit で soft bound、`SessionRegistryConfig::n_stream` = paged KV cache の hard bound と同値。
- `Scheduler::acquire_or_503` → 過負荷時は `ServerError::ServiceUnavailable` → HTTP 503 (T12 land、FR-EX-08 で silent fallback 禁止)。
- `Scheduler::try_acquire_now` → fast-fail 版 (queue しない)。

**Smoke シナリオ** (経路 B = `curl` + xargs `-P` で並列度制御):

```bash
# baseline (1 concurrent)
for i in $(seq 1 100); do echo Hello; done | xargs -P 1 -I{} curl -s -o /dev/null \
    -w '%{time_starttransfer}\n' \
    -X POST http://127.0.0.1:8080/api/tts \
    -H 'Content-Type: application/json' \
    -d '{"text":"{}","voice":"en_US-libritts-high"}' \
    > tts_ttfa_p1.tsv

# concurrent 8
for i in $(seq 1 100); do echo Hello; done | xargs -P 8 -I{} curl -s -o /dev/null \
    -w '%{time_starttransfer}\n' \
    -X POST http://127.0.0.1:8080/api/tts \
    -H 'Content-Type: application/json' \
    -d '{"text":"{}","voice":"en_US-libritts-high"}' \
    > tts_ttfa_p8.tsv

# 過負荷 (default max_concurrent_sessions を超える設定)
for i in $(seq 1 500); do echo Hello; done | xargs -P 64 -I{} curl -s -o /dev/null \
    -w '%{http_code} %{time_starttransfer}\n' \
    -X POST http://127.0.0.1:8080/api/tts \
    -H 'Content-Type: application/json' \
    -d '{"text":"{}","voice":"en_US-libritts-high"}' \
    > tts_overload.tsv

# 503 の割合を確認 (graceful degradation)
awk '{ codes[$1]++ } END { for (c in codes) print c, codes[c] }' tts_overload.tsv
# 期待: "200 ..." と "503 ..." のみ (silent hang / 502 が出たら FAIL)
```

**判定**:
1. baseline p50 (経路 B) と 8-concurrent p50 の差分が < 10 ms なら scheduler が正しく paged KV cache と share していると判断。
2. 過負荷時は 503 が返り、200 の TTFA は 75 ms 前後を維持 (刺し込み許容)。silent hang / 502 は FAIL。

## 6. 結果報告テンプレート

```
### vokra-server v0.9 TTS latency reference (M3-15)

Server binary: vokra-server (SHA-256 ______________)
Backend: ☐ metal (M1 iMac) / ☐ cuda (vast.ai RTX 4090) / ☐ cpu (baseline)
Reference environment: (M2-09-T18 と同一 = 記述内容を M2-09-T18 の記録から引用)

| Concurrent | TTFA p50 (ms) | TTFA p95 (ms) | TTFA p99 (ms) | Verdict (vs 75 ms) |
|-----------:|--------------:|--------------:|--------------:|:------------------:|
| 1          |          __.__ |          __.__ |          __.__ | PASS / FAIL |
| 8          |          __.__ |          __.__ |          __.__ | PASS / FAIL |
| 32         |          __.__ |          __.__ |          __.__ | PASS / FAIL (over-capacity) |

Measurement date: YYYY-MM-DD
Fixture utterance: "Hello world" (7 tokens)
Piper voice: (SHA-256 or upstream model_id) ______________
whisper-base checkpoint SHA-256: ______________ (ASR path 併存の場合)

Multi-session graceful degradation (over-capacity):
- `--concurrent 32` returns explicit ☐ 503 / ☐ 429 (no silent hang, FR-EX-08) — Verdict: PASS / FAIL

四半期 review 申し送り (未達時のみ): __________________
```

## 7. Escalation

- **TTFA が 75 ms を超えた場合**: 未達値を上記テンプレートにそのまま記録し、四半期 review (NFR-MT-05) に申し送る。**新規緩和目標を発明しない** (`docs/tickets/m3/M3-15-vokra-server-multi-session.md` L81)。
- **実機 GPU での always-on gate は X-06 nightly self-hosted matrix** で継続監視 (本 handover の scope 外、`docs/tickets/m3/M3-15-vokra-server-multi-session.md` §M3-15-T01)。
- **Multi-session で silent fallback / silent 502 が出た場合**: FR-EX-08 違反として即修正対象 (server 側 bug)、issue 起票。

## 8. 参考

- `integrations/vokra-server/benches/tts_latency.rs` — in-process reference bench (Wave 3.5 land、CC 側参考値)
- `integrations/vokra-server/src/latency.rs` — latency recorder (`docs/tickets/m3/M3-15-vokra-server-multi-session.md` M3-15-T11)
- `integrations/vokra-server/src/service.rs` — dispatch layer (Wave 11: Voxtral + Whisper beam 対称配線 / Wave 12: `no_repeat_ngram_size` core-side plumbing)
- `docs/tickets/m3/M3-15-vokra-server-multi-session.md` — ticket spec (T01-T14 全 CC、依頼者チケット無し。実測は本 handover が引き取る)
- `docs/tickets/m2/M2-09-vokra-server.md` — M2-09 の spec、T18 = TTS レイテンシ 90 ms 計測ハーネス確立 (v0.5 値、`benches/tts_latency.rs` の起源)
- `docs/m3-owner-verification-checklist.md` §5 — 本 handover が展開する owner runbook
