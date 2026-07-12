# Remote capture — reaching your daemon from anywhere

Cloud memory is convenient because it's reachable from everything you use. Revien can be too — from your phone, another machine, or a hosted agent platform — without your memory ever passing through someone else's server. The daemon stays on your box, the SQLite file stays on your disk, and the network path between you and it is one you own. Same convenience, zero egress.

## The trust model, stated plainly

Two facts about the daemon shape everything below:

- **Capture is token-gated for remote callers; loopback never is.** `POST /v1/ingest` from `127.0.0.1` works unchanged whether or not a token is configured. A remote caller is refused outright (403) unless `REVIEN_CAPTURE_TOKEN` is set on the daemon — remote capture is opt-in — and with it set must send `Authorization: Bearer <token>` (401 otherwise).
- **Recall is not token-gated.** Anyone who can reach the port can query your memory via `/v1/recall`, list nodes, and export the graph.

So the token protects writes, not reads. The real boundary is the network: put the daemon on a private network you control (Tailscale, below), not the open internet. If you must expose it publicly, front it with a proxy that authenticates everything (Recipe B).

The daemon binds `127.0.0.1:7437` by default — unreachable from anywhere else, on purpose. Remote access means either binding a reachable interface (`revien start --host <addr>`) or leaving the bind on loopback and fronting it with a proxy on the same box.

## Recipe A (recommended): Tailscale

Tailscale builds a private WireGuard network between your devices. The daemon becomes reachable from your phone or laptop at a stable hostname, encrypted end to end, with nothing exposed to the internet and no third party in the data path.

1. Install Tailscale on the daemon box and on each device that needs access (phone, laptop). Sign both into the same tailnet.
2. Set a capture token and bind the daemon to the Tailscale interface (or `0.0.0.0` if you're comfortable with the LAN also reaching it):

   ```bash
   export REVIEN_CAPTURE_TOKEN="$(openssl rand -hex 32)"
   revien start --host 0.0.0.0
   ```

   On Windows, set the variable in the environment the daemon starts from.
3. Find the daemon box's tailnet hostname (`tailscale status`). Every device in the tailnet can now reach it at `http://<hostname>:7437`.
4. Send the Bearer token on every remote capture call:

   ```bash
   curl -X POST http://my-desktop:7437/v1/ingest \
     -H "Authorization: Bearer $REVIEN_CAPTURE_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"source_id": "phone", "content": "Decided to ship P3.5 this week.", "defer_embed": true}'
   ```

`defer_embed: true` makes capture return before any embedding-model load. The capture is persisted at once and recallable at the next recall (the search embeds queued captures first) or within ~30 seconds via the idle sweep.

Remember: inside the tailnet, recall is open to any tailnet device. If your tailnet is just your own devices, that's the correct trust boundary. If you share it, use Tailscale ACLs to restrict which devices can reach port 7437.

## Recipe B: public HTTPS via your own reverse proxy

Some platforms will only talk to a public HTTPS endpoint — ChatGPT custom connectors, for example, refuse plain HTTP and private addresses. If you need that, run your own TLS proxy in front of the daemon. Caddy does it in a few lines:

```
revien.example.com {
    # Proxy-level auth on EVERY route — recall is not token-gated in the daemon.
    @unauthorized not header Authorization "Bearer {$REVIEN_CAPTURE_TOKEN}"
    respond @unauthorized 401

    reverse_proxy 127.0.0.1:7437
}
```

Keep the daemon bound to `127.0.0.1` and let Caddy be the only thing listening publicly. The proxy check is load-bearing: without it, the whole API — recall included — is readable by anyone who finds the hostname. Set `REVIEN_CAPTURE_TOKEN` on the daemon too, so capture stays gated even if the proxy is misconfigured.

Be honest with yourself about the trade: a public endpoint is a bigger surface than a tailnet, full stop. TLS certificates advertise your hostname in certificate-transparency logs; scanners will find it. If you can meet the requirement with Tailscale Funnel (public HTTPS terminated by your own node) or an IP allowlist on the proxy, prefer that.

## What this is not: tunnels through someone else's edge

OpenAI's Secure MCP Tunnel, ngrok, and Cloudflare Tunnel all solve the same reachability problem — by routing your traffic through the vendor's infrastructure. Your ingest and recall payloads transit and are terminated at a third party's edge before reaching your daemon. That may be a fine trade for a quick test, and these tools are well-built. But it is not zero-egress: your memory traffic leaves the set of machines you control, and the sovereignty claim this project makes stops being true for that path.

The recipes above keep the edge owned. Tailscale coordinates connections through its control plane but the data path is peer-to-peer WireGuard between your devices; a reverse proxy on your own box is yours entirely. That's the line: who terminates the connection that carries your memory.

## What this unlocks

This recipe is the reachability layer for everything that isn't running on the daemon box: claude.ai custom connectors pointed at your endpoint, iOS Shortcuts capturing thoughts from your phone (see [ios.md](./ios.md)), or any hosted agent platform that needs to read and write your memory. They all reduce to the same two things — a route to port 7437 that you own, and a Bearer token on every capture call.
