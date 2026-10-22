"""A failing forget test must not print a fake key (conftest redacts it)."""
import subprocess
import sys
import textwrap


def test_a_failing_assert_on_a_key_prints_no_key(tmp_path):
    t = tmp_path / "test_x.py"
    t.write_text(textwrap.dedent('''
        def test_fails():
            key = "ghp_" + "".join(chr(97 + i % 6) for i in range(32))
            assert key not in "see " + key
    '''))
    conftest = (__import__("pathlib").Path(__file__).parent / "conftest.py").read_text()
    (tmp_path / "conftest.py").write_text(conftest)
    import os
    repo = str(__import__("pathlib").Path(__file__).resolve().parents[1])
    env = dict(os.environ, PYTHONPATH=repo + os.pathsep + os.environ.get("PYTHONPATH", ""))
    out = subprocess.run([sys.executable, "-m", "pytest", str(t), "-q", "-p", "no:cacheprovider"],
                         capture_output=True, text=True, cwd=str(tmp_path), env=env)
    text = out.stdout + out.stderr
    assert "1 failed" in text
    assert "abcdefabcdefabcdef" not in text and "<fake-key>" in text
