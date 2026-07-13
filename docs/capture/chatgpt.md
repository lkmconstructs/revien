# ChatGPT — a custom MCP connector against your own daemon

Connect the ChatGPT app to your Revien daemon and ChatGPT recalls and stores against *your* memory file — the same SQLite file on your disk that Claude Code and every other adapter uses, via the `revien_recall` and `revien_store` tools. Codex inside the unified app has its own path (`revien connect codex`, which adds the MCP entry to an existing `~/.codex/config.toml` or prints the block for you to paste); this doc is the Chat side.

## Daemon side

Three things, two of which you already have if you followed [remote.md](./remote.md):

1. **Enable the MCP mount.** It is off by default; set `REVIEN_MCP_HTTP=1` in the daemon's environment (and install the extra: `pip install revien[mcp]`). The daemon then serves MCP at `/mcp`, gated by the same rule as capture: loopback callers pass, remote callers need `Authorization: Bearer <REVIEN_CAPTURE_TOKEN>` — and unlike the REST surface, the gate covers recall too.
2. **Set `REVIEN_CAPTURE_TOKEN`.** Without it, any remote call to `/mcp` is refused outright.
3. **A public HTTPS endpoint.** ChatGPT only talks to remote HTTPS MCP servers — there is no local stdio option, and a tailnet-only address won't be reachable from OpenAI's side. That means Recipe B from [remote.md](./remote.md) (your own reverse proxy) or Tailscale Funnel. OpenAI's Secure MCP Tunnel solves the same problem by routing your memory traffic through OpenAI's infrastructure; remote.md has the full contrast.

One non-step: leave `REVIEN_MCP_ALLOWED_ORIGINS` alone. ChatGPT's connector backend is a server-side MCP client, not a browser — it should send no Origin header, so the daemon's browser-origin guard never fires on it. If every connector call comes back 403 with "Browser-origin requests … refused," that assumption broke: the 403 detail names the origin that was sent — add exactly that value to `REVIEN_MCP_ALLOWED_ORIGINS`.

## The auth seam: the edge carries the secret

ChatGPT's connector form offers OAuth or no authentication. There is no field for a static bearer token (verified against OpenAI's docs, July 2026), so the connector itself cannot send the `Authorization` header the daemon wants from remote callers.

Carry the secret in the URL path at your edge instead. Your proxy talks to the daemon over loopback, which is exempt from the token gate — so the path segment is the credential. Treat it like one: generate it with `openssl rand -hex 24`, and know that anyone holding the full URL can read your memory.

Caddy:

```
revien.example.com {
    handle_path /<long-random-segment>/* {
        reverse_proxy 127.0.0.1:7437
    }
    handle {
        respond 404
    }
}
```

`handle_path` strips the prefix, so `https://revien.example.com/<long-random-segment>/mcp` reaches the daemon as `/mcp`. Everything without the segment gets a 404, REST surface included.

Tailscale Funnel works the same way — TLS terminates on your own node, still zero third parties in the data path — and the same rule applies: never funnel the bare daemon port. Funnel forwards to the daemon as a loopback caller, so a bare funnel is your whole memory, unauthenticated, on a public hostname. Serve `/mcp` behind a secret path there too.

Keep `REVIEN_CAPTURE_TOKEN` set on the daemon regardless. It costs nothing and keeps direct remote calls gated if the daemon ever binds a reachable interface.

## ChatGPT side

Verified against OpenAI's help center, July 2026. The feature now lives under **Apps** (the old "Connectors" naming is gone) and the exact path depends on plan:

1. **Enable developer mode.**
   - Business: admins/owners only — *User settings → Apps → Advanced settings → Developer mode*. Each admin toggles it for themselves.
   - Enterprise/Edu: an admin grants access under *Workspace Settings → Permissions & Roles → Connected Data*; granted members then toggle *Settings → Apps → Advanced Settings*.
   - Pro: enable developer mode in settings; connectors are read/fetch only (see limits below).
2. **Create the app.** *Settings → Apps → Create* (admins can also use *Workspace settings → Apps → Create*). Endpoint: your edge URL ending in `/mcp`, e.g. `https://revien.example.com/<long-random-segment>/mcp`. Authentication: **No authentication** — the secret is already in the URL.
3. **Scan Tools.** ChatGPT connects to the server and should list `revien_recall` and `revien_store`. If the scan fails, check that `REVIEN_MCP_HTTP=1` actually reached the daemon's environment — when it's unset the mount simply doesn't exist — and that your edge answers over HTTPS from outside your network.
4. **Create.** The app lands as a draft with a *Dev* label. In a chat, pick it from the tools menu or name it in the prompt ("use the revien app to recall what we decided about the database"). On Business/Enterprise, an admin can publish the draft to the workspace after testing.

## Honest limits

- **Availability shifts by plan and it's still beta.** Full MCP including write actions is rolling out to Business, Enterprise, and Edu; Pro can connect read/fetch tools in developer mode — recall works, `revien_store` may not be callable. The help center commits to ChatGPT web; treat the unified desktop app's Chat surface as unverified until you've seen it work in yours. Check the article below for where your plan sits this month.
- **This is tool-calling, not ambient memory.** ChatGPT calls `revien_recall` when the model decides to (or when you name the app in the prompt). A Claude Code session with the shipped CLAUDE.md snippet recalls at session start and stores decisions as standing instruction; ChatGPT has no equivalent hook, so expect to prompt for it.
- **Write confirmations.** ChatGPT may ask you to confirm before `revien_store` runs. That friction is OpenAI's, not the daemon's.
- **Published apps freeze their tool list.** On workspace plans, ChatGPT snapshots the tools at publish time; if a Revien update changes tool definitions, an admin has to refresh the app before the new shape works.

## Sources

- [Developer mode and MCP apps in ChatGPT](https://help.openai.com/en/articles/12584461-developer-mode-and-mcp-apps-in-chatgpt) — OpenAI Help Center, fetched 2026-07-12 (plan gating, settings paths, Scan Tools flow, write-action confirmations, web-only note).
- [ChatGPT Developer mode](https://developers.openai.com/api/docs/guides/developer-mode) — OpenAI developer docs, fetched 2026-07-12 (authentication options: OAuth or no auth; remote HTTPS requirement; SSE/streamable HTTP).
- [remote.md](./remote.md) — the reachability recipes this doc rides on.
