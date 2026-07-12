// Revien capture bookmarklet (P3.3, capture leg).
//
// Captures the current selection (or the page title if nothing is selected)
// and POSTs it to a running Revien daemon at /v1/ingest with defer_embed=true,
// so capture returns immediately and never blocks on a cold embedding model.
//
// Install: copy the one-liner at the bottom of this file into the URL field
// of a new bookmark. Click the bookmark on any page to capture.
//
// Edit two placeholders before installing:
//   REVIEN_URL   — daemon base URL. Default "http://127.0.0.1:7437" (loopback).
//   REVIEN_TOKEN — leave "" for loopback. For a remote daemon, set it to the
//                  daemon's REVIEN_CAPTURE_TOKEN; it is sent as
//                  "Authorization: Bearer <token>". The header is omitted
//                  entirely when the token is empty.
//
// No dependencies, no eval — plain fetch and DOM. A page whose CSP restricts
// connect-src can still block the request; the failure toast will show it.

/* ── Readable source ─────────────────────────────────────────────────── */

javascript:(function () {
  var REVIEN_URL = "http://127.0.0.1:7437"; // edit for remote daemon
  var REVIEN_TOKEN = "";                    // edit for remote daemon

  var selection = String(window.getSelection()).trim();
  var content = selection || document.title;

  var headers = { "Content-Type": "application/json" };
  if (REVIEN_TOKEN) {
    headers["Authorization"] = "Bearer " + REVIEN_TOKEN;
  }

  function toast(msg) {
    var el = document.createElement("div");
    el.textContent = msg;
    el.style.cssText =
      "position:fixed;bottom:20px;right:20px;z-index:2147483647;" +
      "background:#1a1a1a;color:#eee;padding:8px 14px;border-radius:6px;" +
      "font:13px/1.4 system-ui,sans-serif;box-shadow:0 2px 8px rgba(0,0,0,.4)";
    (document.body || document.documentElement).appendChild(el);
    setTimeout(function () { el.remove(); }, 2200);
  }

  fetch(REVIEN_URL + "/v1/ingest", {
    method: "POST",
    headers: headers,
    body: JSON.stringify({
      source_id: "browser",
      content: content,
      content_type: "note",
      metadata: { title: document.title, url: location.href },
      defer_embed: true
    })
  }).then(function (r) {
    toast(r.ok ? "Saved to Revien" : "Revien: HTTP " + r.status);
  }).catch(function () {
    toast("Revien: no response");
  });
})();

/* ── One-liner (paste this into the bookmark URL) ────────────────────── */

// javascript:(function(){var U="http://127.0.0.1:7437",T="",s=String(window.getSelection()).trim(),h={"Content-Type":"application/json"};if(T)h["Authorization"]="Bearer "+T;function t(m){var e=document.createElement("div");e.textContent=m;e.style.cssText="position:fixed;bottom:20px;right:20px;z-index:2147483647;background:#1a1a1a;color:#eee;padding:8px 14px;border-radius:6px;font:13px/1.4 system-ui,sans-serif;box-shadow:0 2px 8px rgba(0,0,0,.4)";(document.body||document.documentElement).appendChild(e);setTimeout(function(){e.remove()},2200)}fetch(U+"/v1/ingest",{method:"POST",headers:h,body:JSON.stringify({source_id:"browser",content:s||document.title,content_type:"note",metadata:{title:document.title,url:location.href},defer_embed:true})}).then(function(r){t(r.ok?"Saved to Revien":"Revien: HTTP "+r.status)}).catch(function(){t("Revien: no response")})})();
