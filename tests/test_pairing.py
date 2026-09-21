"""
Pairing token (Leg C): the file-backed credential that unlocks remote
/v1/ingest and remote mutation endpoints when REVIEN_CAPTURE_TOKEN isn't
set explicitly.

Every test isolates REVIEN_HOME to a tmp_path so nothing here ever touches
the real ~/.revien on the machine running the suite.
"""

import os

import pytest
from fastapi import HTTPException

from revien import pairing
from revien.daemon.server import check_capture_auth, require_mutation_auth


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    """Point pairing.token_path() at a scratch dir and make sure no stray
    REVIEN_CAPTURE_TOKEN from the environment leaks into a test."""
    monkeypatch.setenv("REVIEN_HOME", str(tmp_path))
    monkeypatch.delenv("REVIEN_CAPTURE_TOKEN", raising=False)
    return tmp_path


class TestTokenPath:
    def test_token_path_honors_revien_home(self, tmp_path):
        assert pairing.token_path() == tmp_path / "pairing.token"

    def test_token_path_falls_back_to_dot_revien(self, monkeypatch, tmp_path):
        monkeypatch.delenv("REVIEN_HOME", raising=False)
        monkeypatch.setattr(pairing.Path, "home", lambda: tmp_path)
        assert pairing.token_path() == tmp_path / ".revien" / "pairing.token"


class TestMintAndLoad:
    def test_load_token_none_when_absent(self):
        assert pairing.load_token() is None

    def test_mint_creates_and_persists(self, tmp_path):
        token = pairing.mint_token()
        assert token
        assert pairing.token_path().exists()
        assert pairing.load_token() == token

    def test_mint_without_rotate_returns_existing(self):
        first = pairing.mint_token()
        second = pairing.mint_token(rotate=False)
        assert first == second

    def test_mint_with_rotate_replaces(self):
        first = pairing.mint_token()
        second = pairing.mint_token(rotate=True)
        assert first != second
        assert pairing.load_token() == second

    def test_minted_token_is_url_safe_and_long(self):
        token = pairing.mint_token()
        # secrets.token_urlsafe(32) -> 43 chars, alnum plus -_
        assert len(token) >= 40

    def test_file_permissions_owner_only_best_effort(self, tmp_path):
        pairing.mint_token()
        path = pairing.token_path()
        if os.name != "nt":
            mode = path.stat().st_mode & 0o777
            assert mode == 0o600


class TestConfiguredToken:
    def test_none_when_nothing_configured(self):
        assert pairing.configured_token() is None

    def test_file_token_used_when_present(self):
        token = pairing.mint_token()
        assert pairing.configured_token() == token

    def test_env_wins_over_file(self, monkeypatch):
        pairing.mint_token()
        monkeypatch.setenv("REVIEN_CAPTURE_TOKEN", "env-token-wins")
        assert pairing.configured_token() == "env-token-wins"

    def test_env_token_is_stripped(self, monkeypatch):
        monkeypatch.setenv("REVIEN_CAPTURE_TOKEN", "  padded-token  ")
        assert pairing.configured_token() == "padded-token"


class TestCaptureAuthWithPairingFile:
    """check_capture_auth must read pairing.configured_token(), not just the
    env var — a minted `revien token` file alone should unlock remote."""

    def test_loopback_unaffected_by_file_token(self):
        pairing.mint_token()
        check_capture_auth("127.0.0.1", "")  # must not raise

    def test_remote_refused_when_nothing_configured(self):
        with pytest.raises(HTTPException) as exc:
            check_capture_auth("100.64.0.7", "")
        assert exc.value.status_code == 403
        assert "revien token" in exc.value.detail

    def test_remote_with_file_token_passes(self):
        token = pairing.mint_token()
        check_capture_auth("100.64.0.7", f"Bearer {token}")  # must not raise

    def test_remote_with_wrong_token_401(self):
        pairing.mint_token()
        with pytest.raises(HTTPException) as exc:
            check_capture_auth("100.64.0.7", "Bearer wrong")
        assert exc.value.status_code == 401

    def test_env_token_still_required_over_stale_file(self, monkeypatch):
        pairing.mint_token()  # file token, e.g. "file-token"
        monkeypatch.setenv("REVIEN_CAPTURE_TOKEN", "env-token")
        with pytest.raises(HTTPException) as exc:
            check_capture_auth("100.64.0.7", "Bearer wrong-guess-of-file-token")
        assert exc.value.status_code == 401
        check_capture_auth("100.64.0.7", "Bearer env-token")  # must not raise


class TestRequireMutationAuth:
    """Same rule, separate name, for Leg D's mutation endpoints."""

    def test_loopback_unaffected(self):
        require_mutation_auth("127.0.0.1", "")  # must not raise

    def test_remote_without_token_403(self):
        with pytest.raises(HTTPException) as exc:
            require_mutation_auth("100.64.0.7", "")
        assert exc.value.status_code == 403

    def test_remote_with_correct_token_passes(self):
        token = pairing.mint_token()
        require_mutation_auth("100.64.0.7", f"Bearer {token}")  # must not raise
