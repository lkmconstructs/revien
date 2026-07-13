"""
MCP client-config installers — `revien connect cursor|windsurf|cline|continue|vscode`.

Writes the Revien MCP server entry ({"command": "revien", "args": ["mcp"]})
into each tool's own MCP config, at its documented location. Discipline
inherited from the codex installer (cli.py):

- Never create a foreign config file unless that tool's own docs sanction
  creating it. Where the file only exists after the tool has run, print the
  paste-block instead.
- Never destroy user content: parse, merge exactly one entry, write back
  pretty-printed. Anything unparseable gets the paste-block and is left
  untouched.
- Presence means the ACTIVE entry (the key in the parsed config), not
  substring containment.
- Encoding tolerance for user-owned files: UTF-8 (with or without BOM) is
  read and written back preserving the BOM; UTF-16 (PowerShell 5.1's
  default) is detected and refused rather than corrupted or crashed on.

Per-tool conventions, verified against current docs (2026-07):

- cursor    ~/.cursor/mcp.json, {"mcpServers": ...}. Docs: "Create
            ~/.cursor/mcp.json in your home directory for tools available
            everywhere" — creation is the documented flow, so a missing file
            is created. (cursor.com/docs/context/mcp)
- windsurf  ~/.codeium/windsurf/mcp_config.json, {"mcpServers": ...}. Docs
            say to EDIT the raw file ("you can add it manually by editing
            the raw mcp_config.json file") — no creation language, so a
            missing file gets the paste-block. (docs.windsurf.com/windsurf/
            cascade/mcp)
- cline     <VS Code User>/globalStorage/saoudrizwan.claude-dev/settings/
            cline_mcp_settings.json, {"mcpServers": ...}. The file is
            created and managed by the Cline extension ("Configure MCP
            Servers" opens it) — never created here. (docs.cline.bot/mcp/
            configuring-mcp-servers)
- continue  ~/.continue/config.yaml, mcpServers: [- name/command/args].
            "A config file is automatically created the first time you use
            Continue" — never created here. Comment-safe: if the file has no
            mcpServers block one is APPENDED (codex-style); if a block
            already exists the entry is printed for manual add rather than
            rewriting the user's YAML. (docs.continue.dev)
- vscode    <VS Code User>/mcp.json, {"servers": ...} — note the top-level
            key is "servers", not "mcpServers". Docs: "You can manually
            configure MCP servers by editing the mcp.json file." Created
            only when the VS Code User directory already exists (proof VS
            Code is installed). (code.visualstudio.com/docs/copilot/chat/
            mcp-servers)
"""

import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

MCP_ENTRY = {"command": "revien", "args": ["mcp"]}

SUPPORTED_TOOLS = ("cursor", "windsurf", "cline", "continue", "vscode")

DISPLAY_NAMES = {
    "cursor": "Cursor",
    "windsurf": "Windsurf",
    "cline": "Cline",
    "continue": "Continue",
    "vscode": "VS Code",
}

_CONTINUE_BLOCK = (
    "mcpServers:\n"
    "  - name: revien\n"
    "    command: revien\n"
    '    args: ["mcp"]\n'
)
_CONTINUE_ENTRY = (
    "  - name: revien\n"
    "    command: revien\n"
    '    args: ["mcp"]\n'
)


@dataclass
class InstallOutcome:
    tool: str
    path: Path
    status: str  # "created" | "merged" | "already" | "skipped"
    detail: str = ""
    snippet: str = ""


# ── Path resolution ───────────────────────────────────────

def vscode_user_dir() -> Path:
    """VS Code's per-user config directory.

    %APPDATA%\\Code\\User on Windows, ~/Library/Application Support/Code/User
    on macOS, ~/.config/Code/User elsewhere (per VS Code's settings.json
    docs). APPDATA wins whenever set so tests and unusual setups can
    redirect it.
    """
    appdata = os.environ.get("APPDATA")
    if appdata:
        return Path(appdata) / "Code" / "User"
    if sys.platform == "win32":  # pragma: no cover - APPDATA is always set
        return Path.home() / "AppData" / "Roaming" / "Code" / "User"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "Code" / "User"
    return Path.home() / ".config" / "Code" / "User"


def config_path_for(tool: str) -> Path:
    if tool == "cursor":
        return Path.home() / ".cursor" / "mcp.json"
    if tool == "windsurf":
        return Path.home() / ".codeium" / "windsurf" / "mcp_config.json"
    if tool == "cline":
        return (
            vscode_user_dir()
            / "globalStorage"
            / "saoudrizwan.claude-dev"
            / "settings"
            / "cline_mcp_settings.json"
        )
    if tool == "continue":
        return Path.home() / ".continue" / "config.yaml"
    if tool == "vscode":
        return vscode_user_dir() / "mcp.json"
    raise ValueError(f"Unknown MCP client tool: {tool}")


# ── Foreign-file reading (codex discipline) ───────────────

def _read_foreign(path: Path):
    """Read a user-owned config file, tolerating BOMs and UTF-16.

    Returns (text, kind) where kind is "utf-8", "utf-8-bom", "utf-16", or
    None when the bytes decode as neither.
    """
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
        kind = "utf-8-bom" if raw.startswith(b"\xef\xbb\xbf") else "utf-8"
        return text, kind
    except (UnicodeDecodeError, UnicodeError):
        pass
    try:
        return raw.decode("utf-16"), "utf-16"
    except (UnicodeDecodeError, UnicodeError):
        return None, None


def _json_snippet(top_key: str) -> str:
    return json.dumps({top_key: {"revien": dict(MCP_ENTRY)}}, indent=2)


def _atomic_write_text(path: Path, text: str, encoding: str) -> None:
    """Write via tmp-in-same-dir + os.replace so a crash mid-write can never
    leave a truncated foreign config — the exact failure class this module
    exists to prevent."""
    tmp = path.with_name(path.name + ".revien-tmp")
    tmp.write_text(text, encoding=encoding)
    os.replace(tmp, path)


# ── JSON installers (cursor, windsurf, cline, vscode) ─────

def _install_json(
    tool: str,
    path: Path,
    top_key: str,
    create_missing: bool,
    missing_detail: str,
    require_parent: Optional[Path] = None,
) -> InstallOutcome:
    snippet = _json_snippet(top_key)

    if not path.exists():
        if create_missing and require_parent is not None and not require_parent.exists():
            return InstallOutcome(
                tool, path, "skipped",
                detail=f"{require_parent} does not exist — is "
                       f"{DISPLAY_NAMES[tool]} installed?",
                snippet=snippet,
            )
        if create_missing:
            path.parent.mkdir(parents=True, exist_ok=True)
            _atomic_write_text(path, snippet + "\n", "utf-8")
            return InstallOutcome(tool, path, "created")
        return InstallOutcome(tool, path, "skipped", detail=missing_detail,
                              snippet=snippet)

    text, kind = _read_foreign(path)
    if text is None or kind == "utf-16":
        # Rewriting a UTF-16 file as UTF-8 (or guessing at unknown bytes)
        # risks corrupting a user-owned file — hand over the block instead.
        reason = ("unrecognized encoding" if text is None
                  else "file is not UTF-8")
        return InstallOutcome(tool, path, "skipped",
                              detail=f"{reason} — left untouched",
                              snippet=snippet)

    if text.strip() == "":
        data = {}
    else:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            return InstallOutcome(
                tool, path, "skipped",
                detail=f"could not parse as JSON ({e.msg}, line {e.lineno}) "
                       f"— left untouched",
                snippet=snippet,
            )
    if not isinstance(data, dict):
        return InstallOutcome(tool, path, "skipped",
                              detail="top level is not a JSON object — left untouched",
                              snippet=snippet)

    servers = data.get(top_key)
    if servers is None:
        servers = {}
        data[top_key] = servers
    if not isinstance(servers, dict):
        return InstallOutcome(
            tool, path, "skipped",
            detail=f'"{top_key}" is not a JSON object — left untouched',
            snippet=snippet,
        )
    if "revien" in servers:
        return InstallOutcome(tool, path, "already")

    servers["revien"] = dict(MCP_ENTRY)
    encoding = "utf-8-sig" if kind == "utf-8-bom" else "utf-8"
    # ensure_ascii=False: a user-owned, human-readable config must not come
    # back with its unicode server names escaped to \uXXXX on every merge.
    _atomic_write_text(
        path, json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding
    )
    return InstallOutcome(tool, path, "merged")


# ── Continue (YAML, append-only) ──────────────────────────

_CONTINUE_NAME_RE = re.compile(r"^\s*-\s+name:\s*['\"]?revien['\"]?\s*(#.*)?$")


def _continue_scan(text: str):
    """Fallback presence check without a YAML parser: an ACTIVE top-level
    mcpServers key, and an active `- name: revien` item INSIDE that block.

    key_exists uses startswith, not equality — a flow-style
    ``mcpServers: [...]`` line must count, or the append below would write a
    duplicate top-level key that YAML parsers resolve last-wins, silently
    clobbering the user's servers. present is scoped to lines between the
    mcpServers key and the next column-0 key — a model or prompt elsewhere
    in the file named "revien" must not read as already-installed."""
    key_exists = False
    present = False
    in_block = False
    for line in text.splitlines():
        stripped = line.split("#", 1)[0].rstrip()
        if stripped and not line[:1].isspace():
            in_block = stripped.startswith("mcpServers:")
            if in_block:
                key_exists = True
            continue
        if in_block and _CONTINUE_NAME_RE.match(line):
            present = True
    return key_exists, present


def _install_continue(path: Path) -> InstallOutcome:
    if not path.exists():
        return InstallOutcome(
            "continue", path, "skipped",
            detail="Continue generates config.yaml on first run — run "
                   "Continue once, then re-run this, or add the block yourself",
            snippet=_CONTINUE_BLOCK,
        )

    text, kind = _read_foreign(path)
    if text is None or kind == "utf-16":
        reason = "unrecognized encoding" if text is None else "file is not UTF-8"
        return InstallOutcome("continue", path, "skipped",
                              detail=f"{reason} — left untouched",
                              snippet=_CONTINUE_BLOCK)

    try:
        import yaml  # transitive dep in most installs; optional here
    except ImportError:
        yaml = None

    if yaml is not None:
        try:
            parsed = yaml.safe_load(text)
        except yaml.YAMLError as e:
            return InstallOutcome(
                "continue", path, "skipped",
                detail=f"could not parse as YAML ({e.__class__.__name__}) "
                       f"— left untouched",
                snippet=_CONTINUE_BLOCK,
            )
        if parsed is None:
            parsed = {}
        if not isinstance(parsed, dict):
            return InstallOutcome("continue", path, "skipped",
                                  detail="top level is not a YAML mapping — left untouched",
                                  snippet=_CONTINUE_BLOCK)
        key_exists = "mcpServers" in parsed
        servers = parsed.get("mcpServers")
        present = isinstance(servers, list) and any(
            isinstance(s, dict) and s.get("name") == "revien" for s in servers
        )
    else:
        key_exists, present = _continue_scan(text)

    if present:
        return InstallOutcome("continue", path, "already")
    if key_exists:
        # An existing mcpServers block would have to be rewritten to gain an
        # entry, and a YAML round-trip drops the user's comments and
        # formatting — hand over the entry instead.
        return InstallOutcome(
            "continue", path, "skipped",
            detail="config.yaml already has an mcpServers block — add this "
                   "entry under it yourself",
            snippet=_CONTINUE_ENTRY,
        )

    # No mcpServers block: append one, comment-safe (codex-style append).
    encoding = "utf-8-sig" if kind == "utf-8-bom" else "utf-8"
    with open(path, "a", encoding=encoding) as f:
        if text and not text.endswith("\n"):
            f.write("\n")
        f.write("\n" + _CONTINUE_BLOCK)
    return InstallOutcome("continue", path, "merged")


# ── Entry point ───────────────────────────────────────────

def install_mcp_client(tool: str, override_path: Optional[Path] = None) -> InstallOutcome:
    """Install the Revien MCP entry into `tool`'s config. `override_path`
    replaces the documented config location (power users, tests)."""
    if tool not in SUPPORTED_TOOLS:
        raise ValueError(f"Unknown MCP client tool: {tool}")

    path = Path(override_path) if override_path else config_path_for(tool)

    if tool == "continue":
        return _install_continue(path)
    if tool == "cursor":
        return _install_json(
            "cursor", path, "mcpServers", create_missing=True,
            missing_detail="",
        )
    if tool == "windsurf":
        return _install_json(
            "windsurf", path, "mcpServers", create_missing=False,
            missing_detail="Windsurf's docs cover editing this file, not "
                           "creating it — open Windsurf once, then re-run "
                           "this, or add the block yourself",
        )
    if tool == "cline":
        return _install_json(
            "cline", path, "mcpServers", create_missing=False,
            missing_detail="this file is created by the Cline extension — "
                           "open Cline once, then re-run this, or add the "
                           "block via Cline's 'Configure MCP Servers'",
        )
    # vscode
    user_dir = path.parent if override_path else vscode_user_dir()
    return _install_json(
        "vscode", path, "servers", create_missing=True,
        missing_detail="",
        require_parent=user_dir,
    )
