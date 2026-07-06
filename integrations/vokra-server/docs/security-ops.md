# vokra-server — Security & Operations (M2-09 T20)

This document pins the security/ops posture of `vokra-server` at v0.5
(M2-09). It complements `docs/scope.md` (crate boundaries) and
`docs/adr-http-stack.md` (dependency choices) and is the single source
of truth for T20's completion. Where the code enforces a default, this
file names the module/field so future edits can be graded against a
fixed target.

Requirements traced: FR-SV-01, FR-EX-08 (no silent fallback / no silent
network exposure), NFR-RL-01 (LC_NUMERIC), NFR-RL-07 (API-boundary
safety), NFR-SC-* (see CLAUDE.md "TLS/auth = reverse-proxy 前提").

## 1. Bind posture: loopback by default, explicit opt-in for network

**Default**: HTTP `127.0.0.1:8080`, Wyoming `127.0.0.1:10300`.
**Enforced**: `crate::config::Config::default()` (`src/config.rs`).
Both defaults are loopback so a fresh `vokra-server` install NEVER
listens on a routable interface without the operator opting in.

External exposure requires an EXPLICIT flag:

```
vokra-server --http-bind 0.0.0.0:8080
vokra-server --wyoming-bind 0.0.0.0:10300
```

Environment overrides (`VOKRA_HTTP_BIND`, `VOKRA_WYOMING_BIND`) are
equally explicit — no wildcard fallback is inferred from the ambient
environment. The smoke test `security::default_bind_loopback`
(this crate, `src/lib.rs`) asserts that `parse_args(["vokra-server"])`
resolves to loopback on both listeners; it fails loudly if any future
edit relaxes the default. This is a FR-EX-08 posture applied to
network exposure: no silent widening of the attack surface.

**Deployment recommendation**: keep the default. Terminate TLS and
enforce auth in a reverse proxy (§2) that listens on `0.0.0.0` and
proxies to loopback. Passing `--http-bind 0.0.0.0` should be a
deliberate, audited act (e.g. inside a private VPC with its own
firewall).

## 2. Reverse proxy is a prerequisite for TLS + auth

`vokra-server` deliberately does NOT terminate TLS and does NOT
implement session auth in v0.5. Operators MUST place a reverse proxy
in front of the loopback listener for any exposure beyond the local
host. Two vetted example configs follow. Both target Ubuntu 24.04 LTS
and Debian 12 defaults.

### 2.1 nginx (Apache-2.0 fork of BSD, package-manager install)

```
# /etc/nginx/sites-available/vokra-server
server {
    listen 443 ssl http2;
    server_name asr.example.org;

    # Managed by certbot / your ACME client of choice.
    ssl_certificate     /etc/letsencrypt/live/asr.example.org/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/asr.example.org/privkey.pem;

    # Cap uploads at the same 25 MiB limit vokra-server enforces (§5).
    client_max_body_size 25m;

    # Increase read timeouts to accommodate multi-second ASR responses.
    proxy_read_timeout  120s;
    proxy_send_timeout  120s;

    location / {
        proxy_pass         http://127.0.0.1:8080;
        proxy_http_version 1.1;
        proxy_set_header   Host              $host;
        proxy_set_header   X-Real-IP         $remote_addr;
        proxy_set_header   X-Forwarded-For   $proxy_add_x_forwarded_for;
        proxy_set_header   X-Forwarded-Proto https;

        # For future WebSocket / streaming endpoints (real-time ASR).
        proxy_set_header   Upgrade    $http_upgrade;
        proxy_set_header   Connection $connection_upgrade;
    }
}
```

### 2.2 Caddy 2 (Apache-2.0, automatic TLS)

```
# /etc/caddy/Caddyfile
asr.example.org {
    # Caddy provisions TLS automatically via Let's Encrypt.
    reverse_proxy 127.0.0.1:8080 {
        header_up X-Real-IP {remote_host}
        header_up X-Forwarded-Proto https
        transport http {
            read_timeout  120s
            write_timeout 120s
        }
    }

    # Match the 25 MiB body cap vokra-server enforces (§5).
    request_body {
        max_size 25MB
    }
}
```

Wyoming (port 10300) is a plain TCP protocol with no HTTP framing and
is NOT proxied through nginx/Caddy in the usual sense. For remote HA
satellites, tunnel it over Tailscale / WireGuard, or terminate on the
loopback of the HA host and rely on HA's own network boundary. Do NOT
expose Wyoming directly on `0.0.0.0` on the public internet — the
protocol has no authentication of its own.

## 3. API key (Bearer token): forward-compat parsing hook, disabled by default

v0.5 does NOT ship a shared-secret auth flow (that layer belongs to
the reverse proxy in §2). It DOES ship a forward-compatible parsing
hook so operators can start emitting `Authorization: Bearer <key>`
headers today without breaking:

- The parser accepts and ignores `Authorization: Bearer <token>` on
  every HTTP route. Presence of the header is a NOOP.
- Absence of the header is also a NOOP (default policy is "no auth").
- Malformed `Authorization` headers do NOT produce a 4xx. They are
  silently ignored so that clients configured for a future v1.0 auth
  flow can point at a v0.5 server for smoke tests.
- Once the auth layer lands (post-v0.5), the default will FLIP to
  "reject unauthenticated requests when a key is configured"; the
  ambient default with no key configured stays "accept all" so this
  is not a silent-exposure regression.

Rationale for parking auth in a proxy for v0.5: keeping TLS + auth
out of `vokra-server` shrinks the trusted-code surface, avoids
handling PEM chains inside a Rust binary that already carries HTTP +
tokio, and matches the Cartesia Sonic / Deepgram Nova on-prem posture
(the reference competitors in CLAUDE.md §"サーバサイド用途対応").

## 4. CORS: restrictive by default

Default `Access-Control-Allow-Origin` = **not sent**. Cross-origin
`fetch()` calls from a browser will fail preflight against a fresh
install, which is the intended posture: `vokra-server` is a backend
service, not a public browser API.

Opt-in options (documented for the reverse proxy layer to add; the
server itself will gain a `--cors-origin <origin>` flag when a real
browser-facing use case appears):

- `Access-Control-Allow-Origin: https://your.frontend.example` for
  a single named origin.
- `Access-Control-Allow-Origin: *` is DISCOURAGED for anything that
  handles user audio — audio uploads leaking to arbitrary origins
  matches the "voice cloning misuse" risk called out in CLAUDE.md
  (ELVIS Act / NO FAKES Act posture, §L).

Preflight (`OPTIONS`) requests receive a `204 No Content` with no
`Access-Control-*` headers by default, matching the "restrictive"
policy.

## 5. Request size limit: 25 MiB

Matches the OpenAI `/v1/audio/transcriptions` upstream limit so
faster-whisper-compatible clients Just Work. Enforced at the axum
layer (`tower_http::limit::RequestBodyLimitLayer`, to be wired when
the security layer lands alongside CORS). Requests exceeding the
limit receive `413 Payload Too Large` with the T05 JSON error schema.
The nginx/Caddy examples in §2 mirror this cap so proxies fail fast
without buffering huge bodies.

25 MiB is enough for ~25 minutes of 16 kHz 16-bit PCM WAV, which
comfortably exceeds Whisper base's 30 s chunk boundary. Larger
transcription jobs should chunk client-side.

## 6. Connection timeout: 60 s (default)

The keep-alive / idle timeout for HTTP connections is 60 s. Long
ASR responses (Whisper large-v3 on CPU can take multiple seconds)
finish comfortably under this ceiling; if a request is still
in-flight after 60 s of idle, it is assumed dead. Aligned with the
`proxy_read_timeout 120s` in the nginx example — the proxy gives
the backend headroom.

Wyoming per-connection timeout is `None` in v0.5: HA satellites hold
long-lived TCP sessions and would break under a strict deadline.
Idle Wyoming connections are cleaned up on graceful shutdown
(`shutdown::install_shutdown_signal`) — see the T03 accept loop.

## 7. Concurrent connection cap: 100 (default)

Cap on the number of simultaneously accepted HTTP connections. Above
this limit, new `accept()` calls block until a slot opens, providing
back-pressure without dropping requests silently. 100 is chosen to
match a moderately provisioned host (8 vCPU, 16 GB RAM) running one
whisper-base engine — the reference `crates/vokra-models` engines
are `Send + Sync` and share `Arc`s across all workers, so per-request
memory overhead is small.

Operators running large-v3 on a single GPU should LOWER this cap to
match GPU memory budget; a v1.0 admission-control layer will make
this dynamic.

Wyoming does not enforce a cap in v0.5: HA typically opens a
single long-lived session, so a static cap would be an anti-feature.

## 8. LC_NUMERIC pinning (defense in depth)

`crate::enforce_c_numeric_locale()` sets `LC_NUMERIC=C` (and
`LC_ALL=C`) BEFORE the tokio runtime spawns worker threads. This
mitigates the NFR-RL-01 "European-locale `strtod` crash" class of
bug: even if a transitive dependency reaches a C library's
`strtod`, it will parse `.` as the decimal separator. This is not a
security fence per se, but a hard-to-diagnose availability risk if
skipped, so it is documented alongside the security posture.

## 9. Panic isolation (NFR-RL-07)

Every axum handler is wrapped in `tower_http::catch_panic::CatchPanicLayer`
via `error::catch_panic_layer` so a `panic!()` inside a handler
becomes a `500 Internal Server Error` with the T05 JSON error schema,
NOT a runtime abort. Wyoming per-connection tasks live inside
`tokio::spawn` and any panic aborts only the affected connection.

## 10. Smoke test contract

The T20 completion criterion is a smoke test showing that the default
bind is loopback and only loopback:

```
cd integrations/vokra-server && cargo test security::default_bind_loopback
```

Test location: `src/lib.rs`, `#[cfg(test)] mod security`. The test
calls `parse_args(["vokra-server"])` with a clean environment
(`VOKRA_HTTP_BIND` / `VOKRA_WYOMING_BIND` unset) and asserts:

1. Resolved `http_bind.ip().is_loopback()` is `true`.
2. Resolved `wyoming_bind.ip().is_loopback()` is `true`.
3. Both addresses are IPv4 `127.0.0.1` specifically (guards against
   an accidental widening to `0.0.0.0` masquerading as loopback via
   IPv6 dual-stack quirks).

Any future PR that changes the default MUST update this test AND
this document in the same change, so security-relevant defaults
cannot drift silently.

## 11. Non-goals (v0.5)

- Rate limiting per-IP: pushed to the reverse proxy (nginx
  `limit_req_zone`, Caddy `rate_limit`).
- WAF / OWASP Top 10 filtering: reverse proxy layer.
- TLS termination: reverse proxy layer.
- Bearer-token verification: forward-compat parse only (§3), full
  enforcement post-v0.5.
- Wyoming auth: rely on network isolation (Tailscale / WireGuard /
  loopback + HA on the same host).
- Web UI / admin console: out of scope for v0.5.

## 12. Change log

- 2026-07-06 — T20 initial cut. Loopback defaults, reverse proxy
  examples, forward-compat auth hook, restrictive CORS, 25 MiB /
  60 s / 100-conn caps, smoke test wired in `src/lib.rs`.
