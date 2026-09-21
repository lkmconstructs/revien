"""G11: version string is untouched at 0.3.0 and CHANGELOG Unreleased names
every shipped feature.

CHECK: python scripts/gates/check_docs.py
EXPECT: docs verification passed
"""
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def fail(msg):
    print(f"check_docs: {msg}", file=sys.stderr)
    sys.exit(1)


def main():
    # ── revien/__init__.py __version__ == "0.3.0" exactly ──
    import revien
    if revien.__version__ != "0.3.0":
        fail(f"revien.__version__ == {revien.__version__!r}, expected '0.3.0'")

    # ── CHANGELOG.md Unreleased section mentions every required term ──
    changelog = (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    m = re.search(r"^## \[Unreleased\]\s*$", changelog, re.MULTILINE)
    if not m:
        fail("CHANGELOG.md has no '## [Unreleased]' heading")
    start = m.end()
    next_heading = re.search(r"^## \[", changelog[start:], re.MULTILINE)
    end = start + next_heading.start() if next_heading else len(changelog)
    unreleased_section = changelog[start:end]

    required_changelog_terms = [
        "origin_runtime", "--source", "revien token", "skills ingest", "propose",
    ]
    missing_changelog = [
        term for term in required_changelog_terms
        if term.lower() not in unreleased_section.lower()
    ]
    if missing_changelog:
        fail(f"CHANGELOG.md Unreleased section is missing: {missing_changelog}")

    # ── README.md "Getting started" (or similarly named) section mentions
    # --source, skills ingest, revien token ──
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    headings = list(re.finditer(r"^## (.+)$", readme, re.MULTILINE))
    getting_started_idx = None
    for i, h in enumerate(headings):
        if "getting started" in h.group(1).strip().lower():
            getting_started_idx = i
            break
    if getting_started_idx is None:
        fail("README.md has no 'Getting started' (or similarly named) section")
    gs_heading = headings[getting_started_idx]
    gs_start = gs_heading.end()
    gs_end = (
        headings[getting_started_idx + 1].start()
        if getting_started_idx + 1 < len(headings)
        else len(readme)
    )
    getting_started_section = readme[gs_start:gs_end]

    required_readme_terms = ["--source", "skills ingest", "revien token"]
    missing_readme = [
        term for term in required_readme_terms
        if term.lower() not in getting_started_section.lower()
    ]
    if missing_readme:
        fail(f"README.md 'Getting started' section is missing: {missing_readme}")

    print("docs verification passed")


if __name__ == "__main__":
    main()
