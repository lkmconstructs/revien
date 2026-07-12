# iOS Capture — Apple Shortcuts

Capture a thought from your phone into your own Revien daemon. No app to install,
no cloud service in the path — the phone talks directly to the daemon you run.

## Before you start

The daemon binds `127.0.0.1:7437` by default, which your phone cannot reach.
You need a route from the phone to the daemon — Tailscale is the recommended
path. See [remote.md](./remote.md) for the reachability recipe (Tailscale or
LAN binding, plus the `REVIEN_CAPTURE_TOKEN` setup).

What you need in hand:

- The daemon's Tailscale hostname (or LAN address). The URL form is
  `http://<tailscale-hostname>:7437/v1/ingest`.
- The capture token you set as `REVIEN_CAPTURE_TOKEN` on the daemon. Remote
  callers must send `Authorization: Bearer <token>` — without a token
  configured the daemon refuses remote capture outright (403), and a wrong
  token gets a 401. Loopback callers are exempt; your phone is not loopback.

Both shortcuts POST this JSON body:

```json
{
  "source_id": "ios",
  "content": "<your captured text>",
  "content_type": "note",
  "defer_embed": true
}
```

`defer_embed: true` means the daemon persists immediately and embeds later, so
capture never waits on a model load. The note becomes recallable at your next
recall (the search embeds queued captures first) or within about 30 seconds
via the daemon's idle sweep, whichever comes first.

## Shortcut 1: Typed capture

Manual or share-sheet capture. Build it in the Shortcuts app:

1. New shortcut. Name it **Capture to Revien**.
2. Add **Text**. Paste your capture token into it. Rename the variable to
   `Token` (tap the action's arrow > Rename).
3. Add **Ask for Input**. Input type: Text. Prompt: "What's on your mind?"
4. Add **Get Contents of URL**.
   - URL: `http://<tailscale-hostname>:7437/v1/ingest`
   - Tap the arrow to expand. Method: **POST**.
   - Headers: add `Authorization` with value `Bearer ` followed by the
     `Token` variable (type "Bearer ", space included, then insert the
     variable from the variable picker).
   - Request Body: **JSON**. Add fields:
     - `source_id` (Text): `ios`
     - `content` (Text): the **Provided Input** variable
     - `content_type` (Text): `note`
     - `defer_embed` (Boolean): true
5. Add **Show Notification**. Body: "Captured." Optionally insert the
   **Contents of URL** variable to see the daemon's response.

To capture from the share sheet as well: open the shortcut's settings (info
icon), enable **Show in Share Sheet**, accept **Text**. Then replace the
**Ask for Input** step with a **Receive Text input from Share Sheet** header
and point `content` at the **Shortcut Input** variable. Now any selected text
or shared page excerpt can be sent straight in.

## Shortcut 2: Brain dump (Siri voice capture)

Same POST, voice-driven. The shortcut's name becomes the Siri phrase — name it
**Brain dump** and "Hey Siri, brain dump" invokes it hands-free.

1. New shortcut. Name it **Brain dump**.
2. Add **Text** with your token, renamed to `Token` (same as above).
3. Add **Dictate Text**. Language: your default. Stop Listening: **After
   Pause** works well for a single thought; **On Tap** for longer rambles.
4. Add **Get Contents of URL** — identical to the typed shortcut, except
   `content` points at the **Dictated Text** variable.
5. Add a confirmation. **Show Notification** ("Dumped.") is quiet; if you
   want Siri to speak it back, use **Speak Text** with a short phrase like
   "Got it" instead.

Invoked by voice, the whole exchange runs eyes-free: speak, pause, Siri
confirms, the thought is in your graph.

## What the daemon returns

A successful capture returns JSON with `context_node_id`, node/edge counts,
and graph totals. You don't need any of it for capture to work — but wiring
the response into the notification is a cheap way to confirm the daemon
actually took the write, not just that the request left the phone.

## Failure modes

- **403 "Remote capture is disabled"** — the daemon has no
  `REVIEN_CAPTURE_TOKEN` set. Set it and restart. See
  [remote.md](./remote.md).
- **401 "Invalid or missing capture token"** — the header is wrong. Check for
  a missing `Bearer ` prefix or a stray space in the Token text action.
- **Could not connect** — the phone can't reach the daemon. Confirm Tailscale
  is up on both ends and the daemon is bound where the phone can see it
  ([remote.md](./remote.md)).

One more time, because it's the point: the token lives in your shortcut, the
daemon lives on your hardware, and nothing between your voice and your memory
graph belongs to anyone else.
