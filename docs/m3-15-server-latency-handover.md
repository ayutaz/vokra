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

## 4. TTS レイテンシ計測 (実 HTTP client)

**Honest state (2026-07-12 更新)**: `vokra-cli bench` は 2026-07-12 land で **`--server` / `--endpoint` / `--concurrent` / `--voice` / `--budget-ms` / `--timeout-secs` フラグを recognise する** (`crates/vokra-cli/src/bench.rs::parse_args`) ようになった。ただし vokra-cli 自身は TCP ソケットを開かず、`--server` が渡されると FR-EX-08 の明示的な redirect (exit code = 4) を出して `integrations/vokra-cli-bench-server/` (pure-`std::net::TcpStream`、zero third-party deps) を指す。M3-15-T11 の ticket spec 上は "vokra-cli bench を multi-session server mode に拡張" と書かれていたが、実 land は 3 か所に分離した:
1. **in-process (schema-layer floor)**: `integrations/vokra-server/benches/tts_latency.rs` (Wave 3.5 land)
2. **HTTP-boundary (wire) — ureq/serde_json 版**: `integrations/vokra-server-bench/` (2026-07-11 land、excluded workspace、独自 Cargo.lock、ureq + serde_json ベース、TLS 対応)
3. **HTTP-boundary (wire) — zero-dep pure-std 版**: `integrations/vokra-cli-bench-server/` (2026-07-12 land、excluded workspace、独自 Cargo.lock は自身のみ、pure-`std::net::TcpStream` + hand-written HTTP/1.1 + hand-crafted JSON、TLS 非対応)

以下 4 経路のいずれかで v0.9 参考採取する (推奨は **C = ureq 版 or D = pure-std 版**、次点で A = floor 値):

### A. In-process reference bench を primary artifact として採用 (floor 値、CC land 済)

`benches/tts_latency.rs` (§2 参照) の JSON blob を v0.9 reference として記録。**FakeSynth 経由ゆえ engine backend 独立の floor 値**であり、実 engine (Metal / CUDA) の値ではない旨を明記する。**wire (network + tokio scheduler) は含まない**。

### B. `curl` + timing loop で HTTP レイテンシ実測 (owner-side、最小工数、fallback)

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

`curl` は真の TTFA (Time To First Audio byte) を測るには `-w '%{time_starttransfer}'` を使う (final byte を測るなら `%{time_total}`)。**両方採取して報告するのが正確**。sustained concurrency は `xargs -P` で回すが sample の正確な quantile は awk で自前計算する必要があり、fragile な shell math ゆえ経路 C を推奨。

### C. `vokra-server-bench` binary で HTTP レイテンシ実測 (推奨、CC land 済 2026-07-11)

`integrations/vokra-server-bench/` = excluded workspace binary。ureq (blocking HTTP) + serde_json を含み **独自 Cargo.lock で isolate** (root workspace 非干渉、NFR-DS-02 保存)。**sustained concurrency + nearest-rank percentile + JSON schema 準拠** で経路 A/B の弱点を両方埋める。

```bash
# 一回きり (root workspace 非干渉なのでこの ディレクトリで build する)
cd integrations/vokra-server-bench
cargo build --release
./target/release/vokra-server-bench --help  # 全フラグ + exit code contract

# server 側で /api/tts を LISTEN (§3 参照)
# client 側 (single-session baseline)
./target/release/vokra-server-bench \
    --server http://127.0.0.1:8080 \
    --endpoint /api/tts \
    --text "Hello world" \
    --voice en_US-libritts-high \
    --iters 100 --warmup 10 --concurrent 1 \
    --format json --budget-ms 75 > tts_p1.json

# 8-concurrent (multi-session dispatch)
./target/release/vokra-server-bench \
    --server http://127.0.0.1:8080 \
    --iters 100 --warmup 10 --concurrent 8 \
    --format json --budget-ms 75 > tts_p8.json

# 過負荷 (default max_concurrent_sessions を超える設定、FR-SV-06 graceful degradation)
./target/release/vokra-server-bench \
    --server http://127.0.0.1:8080 \
    --iters 500 --warmup 0 --concurrent 64 \
    --format json --budget-ms 75 > tts_p64.json
# → tts_p64.json の counters.over_capacity_503 と counters.ok_2xx で graceful degradation を判定
```

**出力 JSON schema** (M2-14 の schema と互換、`--format kv` は同キーの `key=value` 版):
```json
{"endpoint":"http://127.0.0.1:8080/api/tts","utterance":"Hello world","voice":"en_US-libritts-high",
 "iterations":100,"warmup":10,"concurrent":8,
 "ttfa_ms":{"p50":48.1,"p95":72.3,"p99":91.0,"median":48.1,"max":125.4},
 "total_ms":{"p50":48.5,"p95":72.7,"p99":91.4,"median":48.5,"max":125.8},
 "counters":{"ok_2xx":100,"over_capacity_503":0,"rate_limited_429":0,"client_error_4xx":0,"server_error_5xx":0,"transport_errors":0},
 "budget_ms":75,"verdict":"PASS"}
```

**Exit code contract** (FR-EX-08、silent fallback 禁止):
- `0` — 測定完了 (verdict は PASS / FAIL のいずれもあり得る、bench 側は判定を強制しない)
- `2` — CLI 引数エラー
- `3` — 全リクエストが transport 失敗 (server 不到達 = 実測不能)

**Boundary** (`integrations/vokra-server-bench/src/lib.rs` module doc に詳細):
- `ttfa_ms` = start → ureq の `send_bytes(...)` return (status + headers 受信、TTFA を近似; 現行 non-streaming vocoder では body drain とほぼ等価)
- `total_ms` = start → response body 完全 drain

### D. `vokra-cli-bench-server` binary で HTTP レイテンシ実測 (推奨 zero-dep、CC land 済 2026-07-12)

`integrations/vokra-cli-bench-server/` = excluded workspace binary。**pure-`std::net::TcpStream` + hand-written HTTP/1.1 + hand-crafted JSON**、`Cargo.lock` は自身のみ (第三者クレート ゼロ)、TLS 非対応 (loopback 参照環境専用)。経路 C (ureq + serde_json) と **byte-for-byte 同一の JSON 出力スキーマ + KV スキーマ + exit code contract** を維持する。`vokra-cli bench --server URL` を打つと exit 4 で本 binary への redirect 案内が stderr に出る (silent fallback 禁止 = FR-EX-08)。

```bash
# 一回きり build (ureq 版と同じディレクトリ流儀)
cd integrations/vokra-cli-bench-server
cargo build --release
./target/release/vokra-cli-bench-server --help  # 全フラグ + exit code contract

# server 側で /api/tts を LISTEN (§3 参照)
# client 側 (single-session baseline)
./target/release/vokra-cli-bench-server \
    --server http://127.0.0.1:8080 \
    --endpoint /api/tts \
    --text "Hello world" \
    --voice en_US-libritts-high \
    --iters 100 --warmup 10 --concurrent 1 \
    --format json --budget-ms 75 > tts_p1.json

# 8-concurrent (経路 C と同じフラグ)
./target/release/vokra-cli-bench-server \
    --server http://127.0.0.1:8080 \
    --iters 100 --warmup 10 --concurrent 8 \
    --format json --budget-ms 75 > tts_p8.json
```

**Boundary** (`integrations/vokra-cli-bench-server/src/lib.rs` module doc に詳細):
- `ttfa_ms` = start → `\r\n\r\n` header terminator 受信 (経路 C の ureq `send_bytes(...)` return とほぼ等価、Content-Length'd body なら差はゼロ)
- `total_ms` = start → response body 完全 drain (Content-Length OR chunked OR read-to-EOF)

**経路 D と C の使い分け**:
- **TLS が要る (LAN 越し `https://`、reverse proxy 経由)** → 経路 C (ureq + rustls)。経路 D は `https://` を URL parse 時 explicit error で拒否 (FR-EX-08)。
- **`Cargo.lock` の第三者クレート ゼロを厳格に要求** (NFR-DS-02 belt-and-suspenders) → 経路 D。経路 C は excluded workspace ゆえ root `Cargo.lock` は汚さないが、excluded workspace の `Cargo.lock` は ureq + rustls + webpki-roots 系の transitive を含む。
- **どちらも loopback で通したい** → 経路 C を primary 参考値、経路 D を cross-check とする ({ureq → std}, {rustls → 生 TCP} の 2 系統 で JSON が byte-一致することを確認)。

**Exit code contract** (FR-EX-08 = 経路 C と同一):
- `0` — 測定完了 (verdict は PASS / FAIL のいずれもあり得る)
- `2` — CLI 引数エラー
- `3` — 全リクエストが transport 失敗 (server 不到達 = 実測不能)

**判定基準 (NFR-PF-05 v0.9)**:
- **PASS reference (経路 A)**: `benches/tts_latency.rs` の `http_end_to_end.median_us` が `75_000` (= 75 ms) 未満。**FakeSynth 経路ゆえ floor 値であり実 engine 値ではない**旨を報告に明記。
- **PASS 実測 (経路 B)**: `curl -w '%{time_starttransfer}'` (TTFA) の p50 が 75 ms 未満。
- **PASS 実測 (経路 C、推奨・TLS 対応)**: `vokra-server-bench` の `ttfa_ms.p50` (JSON 出力) が 75 ms 未満。
- **PASS 実測 (経路 D、推奨 zero-dep)**: `vokra-cli-bench-server` の `ttfa_ms.p50` (JSON 出力) が 75 ms 未満。
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

- `integrations/vokra-server/benches/tts_latency.rs` — in-process reference bench (Wave 3.5 land、CC 側参考値 = 経路 A、schema-layer floor)
- `integrations/vokra-server-bench/` — HTTP-boundary bench (2026-07-11 land、excluded workspace = 経路 C、推奨・TLS 対応。ureq + serde_json、独自 Cargo.lock、`--server` / `--endpoint` / `--concurrent` / `--budget-ms` / `--format kv|json` サポート、38 tests = 30 unit + 8 e2e mock)
- `integrations/vokra-cli-bench-server/` — HTTP-boundary bench (2026-07-12 land、excluded workspace = 経路 D、推奨 zero-dep。pure-`std::net::TcpStream` + hand-written HTTP/1.1 + hand-crafted JSON、独自 Cargo.lock は自身のみ、`Cargo.lock` に第三者クレート ゼロ、`--server` / `--endpoint` / `--concurrent` / `--voice` / `--budget-ms` / `--timeout-secs` / `--format kv|json` サポート、66 tests = 53 unit + 13 e2e mock。`vokra-cli bench --server URL` を打つと exit 4 で本 binary への redirect 案内を stderr に出す)
- `crates/vokra-cli/src/bench.rs` — `--server` / `--endpoint` / `--concurrent` / `--voice` / `--budget-ms` / `--timeout-secs` フラグの parse + FR-EX-08 redirect (2026-07-12 land、M3-15-T11 の gap-fill = handover doc §4 が指摘していた "vokra-cli には無い" 状態を解消。vokra-cli 自身は TCP を開かず、`--server` を渡すと exit 4 で `integrations/vokra-cli-bench-server` へ redirect。10 new tests + 6 test literal extensions)
- `integrations/vokra-server/src/latency.rs` — latency recorder (`docs/tickets/m3/M3-15-vokra-server-multi-session.md` M3-15-T11)
- `integrations/vokra-server/src/service.rs` — dispatch layer (Wave 11: Voxtral + Whisper beam 対称配線 / Wave 12: `no_repeat_ngram_size` core-side plumbing)
- `docs/tickets/m3/M3-15-vokra-server-multi-session.md` — ticket spec (T01-T14 全 CC、依頼者チケット無し。実測は本 handover が引き取る)
- `docs/tickets/m2/M2-09-vokra-server.md` — M2-09 の spec、T18 = TTS レイテンシ 90 ms 計測ハーネス確立 (v0.5 値、`benches/tts_latency.rs` の起源)
- `docs/m3-owner-verification-checklist.md` §5 — 本 handover が展開する owner runbook
