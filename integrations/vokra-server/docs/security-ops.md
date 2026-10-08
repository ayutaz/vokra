# vokra-server — Security & Operations (M2-09 T20)

**Current-status note (2026-10-08):** This document retains the M2-09 security
and operations decision record. Current implementation status is the source
and [`../README.md`](../README.md): Wyoming full ASR/TTS handling and its
connection-scoped barge-in path are wired; HTTP `stream=true` and word-level
timestamps remain unexposed; authentication is still a reverse-proxy concern,
not a built-in server policy. Repository and legal boundaries are maintained
in [`AGENTS.md`](../../../AGENTS.md), [`CONTRIBUTING.md`](../../../CONTRIBUTING.md),
and [`docs/legal-compliance.md`](../../../docs/legal-compliance.md).

This document pins the security/ops posture of `vokra-server` at v0.5
(M2-09). It complements `docs/scope.md` (crate boundaries) and
`docs/adr-http-stack.md` (dependency choices) and is the single source
of truth for T20's completion. Where the code enforces a default, this
file names the module/field so future edits can be graded against a
fixed target.

Requirements traced: FR-SV-01, FR-EX-08 (no silent fallback / no silent
network exposure), NFR-RL-01 (LC_NUMERIC), NFR-RL-07 (API-boundary
safety), NFR-SC-* (see [`CONTRIBUTING.md`](../../../CONTRIBUTING.md) and
[`docs/legal-compliance.md`](../../../docs/legal-compliance.md) for the
reverse-proxy/auth and legal-review boundary).

The Wyoming connection task is panic-isolated; the production HTTP router
does not currently attach the reusable `CatchPanicLayer`, so HTTP panic
isolation remains an open implementation follow-up (§9).

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
host. The examples below are illustrative, not deployment-tested, and do
not claim Ubuntu/Debian package or filesystem compatibility. Adapt the
certificate paths, package layout, firewall, and authentication policy to
the target host before deployment. Both examples include an explicit HTTP
Basic-auth gate; this is not the OpenAI Bearer scheme that `vokra-server`
ignores, so clients may need an authentication adapter.

### 2.1 nginx (official two-clause BSD-style license)

See the official [`ngx_http_auth_basic_module` documentation](https://nginx.org/en/docs/http/ngx_http_auth_basic_module.html)
and [NGINX license](https://nginx.org/LICENSE). Generate the htpasswd file
with `htpasswd` or an equivalent approved tool; never put a real password in
this document.

```
# /etc/nginx/sites-available/vokra-server
server {
    listen 443 ssl http2;
    server_name asr.example.org;

    # Managed by certbot / your ACME client of choice.
    ssl_certificate     /etc/letsencrypt/live/asr.example.org/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/asr.example.org/privkey.pem;

    # Cap uploads at the same 25 MiB limit vokra-server enforces (§5).
    client_max_body_size 26214400;

    # Increase read timeouts to accommodate multi-second ASR responses.
    proxy_read_timeout  120s;
    proxy_send_timeout  120s;

    location / {
        auth_basic           "vokra-server";
        auth_basic_user_file /etc/nginx/.htpasswd;
        proxy_pass         http://127.0.0.1:8080;
        proxy_http_version 1.1;
        proxy_set_header   Host              $host;
        proxy_set_header   X-Real-IP         $remote_addr;
        proxy_set_header   X-Forwarded-For   $proxy_add_x_forwarded_for;
        proxy_set_header   X-Forwarded-Proto https;
    }
}
```

### 2.2 Caddy 2 (Apache-2.0, automatic TLS)

This example targets Caddy **v2.8.0 or later**, where the directive is named
`basic_auth` (older versions called it `basicauth`). See the official
[`basic_auth` documentation](https://caddyserver.com/docs/caddyfile/directives/basic_auth),
[`request_body` documentation](https://caddyserver.com/docs/caddyfile/directives/request_body),
and [Caddy source license](https://github.com/caddyserver/caddy/blob/master/LICENSE).
The hash must be generated out of band with `caddy hash-password`; the
placeholder below is not a credential.

```
# /etc/caddy/Caddyfile
asr.example.org {
    basic_auth {
        operator {env.VOKRA_BASIC_AUTH_HASH}
    }

    # Match the 25 MiB body cap vokra-server enforces (§5).
    request_body {
        max_size 26214400
    }

    # Caddy provisions TLS automatically via Let's Encrypt.
    reverse_proxy 127.0.0.1:8080 {
        header_up X-Real-IP {remote_host}
        header_up X-Forwarded-Proto https
        transport http {
            read_timeout  120s
            write_timeout 120s
        }
    }
}
```

Wyoming (port 10300) is a plain TCP protocol with no HTTP framing and
is NOT proxied through nginx/Caddy in the usual sense. For remote HA
satellites, tunnel it over Tailscale / WireGuard, or terminate on the
loopback of the HA host and rely on HA's own network boundary. Do NOT
expose Wyoming directly on `0.0.0.0` on the public internet — the
protocol has no authentication of its own.

## 3. API key (Bearer token): reverse-proxy boundary only

v0.5 does NOT ship a shared-secret auth flow or an `Authorization` parser.
Operators MUST place a reverse proxy in front of the loopback listener for
any exposure beyond the local host. The proxy terminates TLS and authenticates
requests before forwarding them. A future built-in auth layer is a separate
design decision; this server must not be described as accepting or validating
Bearer credentials today.

Rationale for parking auth in a proxy for v0.5: keeping TLS + auth
out of `vokra-server` shrinks the trusted-code surface, avoids
handling PEM chains inside a Rust binary that already carries HTTP +
tokio, and matches the reference on-prem posture recorded in the
historical project planning material.

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
  matches the voice-cloning misuse risk and deployment/legal-review boundary
  documented in [`docs/legal-compliance.md`](../../../docs/legal-compliance.md)
  §§3–4.

Preflight (`OPTIONS`) requests receive a `204 No Content` with no
`Access-Control-*` headers by default, matching the "restrictive"
policy.

## 5. Request size limit: 25 MiB

Matches the OpenAI `/v1/audio/transcriptions` upstream limit so
faster-whisper-compatible clients Just Work. Enforced at the axum
layer (`axum::extract::DefaultBodyLimit::max` on the OpenAI
transcription route). The route is fail-closed at this limit before
inference. Handler-side multipart parsing maps `MultipartError` and other
request-shape failures to `400 invalid_multipart`; an extractor-level
body-limit rejection is outside that mapper. Therefore the current contract
does not promise one uniform `413 Payload Too Large` status or T05 JSON
envelope for every over-limit path. This limit is route-local; it is not a
claim that every future endpoint accepts the same body size.
The nginx/Caddy examples in §2 mirror this cap so proxies fail fast
without buffering huge bodies.

25 MiB is enough for ~25 minutes of 16 kHz 16-bit PCM WAV, which
comfortably exceeds Whisper base's 30 s chunk boundary. Larger
transcription jobs should chunk client-side.

## 6. Connection timeout: not enforced by the server

The current `vokra-server` HTTP listener does not install a 60-second
keep-alive or request timeout layer. Operators MUST set an appropriate
timeout at the reverse proxy (the nginx example uses `proxy_read_timeout
120s`) and should account for long CPU ASR requests. Adding a server-side
timeout is a future implementation change, not current v0.5 behavior.

Wyoming per-connection timeout is `None` in v0.5: HA satellites hold
long-lived TCP sessions and would break under a strict deadline.
Idle Wyoming connections are cleaned up on graceful shutdown
(`shutdown::install_shutdown_signal`) — see the T03 accept loop.

## 7. Wyoming session concurrency: no HTTP or raw-TCP accept cap

The service-aware startup path passes the shared
`--max-concurrent-sessions` / `VOKRA_MAX_CONCURRENT_SESSIONS` value to the
Wyoming connection loop. It defaults to `4` and bounds concurrent configured
Wyoming model sessions. The scheduler is not attached to the OpenAI, vLLM, or
piper HTTP routers, so this value is not an HTTP-request cap. It also does not
limit raw TCP accepts; there is no independent Wyoming accept semaphore.

Use the reverse proxy for HTTP connection/rate limiting and a firewall or
authenticated tunnel for Wyoming network exposure.

Operators running large-v3 on a single GPU should LOWER this cap to
match GPU memory budget.

## 8. LC_NUMERIC pinning (defense in depth)

`crate::enforce_c_numeric_locale()` sets `LC_NUMERIC=C` (and
`LC_ALL=C`) BEFORE the tokio runtime spawns worker threads. This
mitigates the NFR-RL-01 "European-locale `strtod` crash" class of
bug: even if a transitive dependency reaches a C library's
`strtod`, it will parse `.` as the decimal separator. This is not a
security fence per se, but a hard-to-diagnose availability risk if
skipped, so it is documented alongside the security posture.

## 9. Panic isolation (NFR-RL-07) — HTTP attachment pending

The reusable `error::catch_panic_layer` helper and its unit test exist, but
`server::build_http_app` does not currently attach that layer to the
production router. Therefore HTTP panic-to-500 behavior is **not claimed**
for v0.5 and must remain a follow-up before this section can be marked
complete. Wyoming per-connection tasks do run inside the `catch_unwind`
guard and a panic closes only the affected connection.

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
- Bearer-token verification: reverse-proxy boundary only (§3); no server-side
  parser or enforcement is present in v0.5.
- Wyoming auth: rely on network isolation (Tailscale / WireGuard /
  loopback + HA on the same host).
- Web UI / admin console: out of scope for v0.5.

## 12. Change log

- 2026-07-06 — T20 initial cut. Loopback defaults, reverse-proxy
  examples, restrictive CORS, and the initial 25 MiB / timeout /
  connection-cap design record were drafted; current enforcement is
  summarized in the status note and §§3/5–9 above.
