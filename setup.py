from pathlib import Path

from setuptools import setup, find_packages

# Rendered on the PyPI project page.
long_description = Path(__file__).parent.joinpath("README.md").read_text(encoding="utf-8")

setup(
    name="revien",
    version="0.3.0",
    description="Local-first, graph-based memory engine for AI systems. Memory that returns.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="LKM Constructs LLC",
    author_email="melissa@lkmconstructs.com",
    license="Apache-2.0",
    url="https://github.com/lkmconstructs/revien",
    project_urls={
        "Homepage": "https://lkmconstructs.com",
        "Source": "https://github.com/lkmconstructs/revien",
        "Changelog": "https://github.com/lkmconstructs/revien/blob/main/CHANGELOG.md",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="memory, graph, retrieval, ai, agents, local-first, obsidian, rag",
    # revien_bench is a DEV-ONLY benchmark harness (LoCoMo). It is intentionally
    # excluded from the wheel — never shipped to end users.
    packages=find_packages(exclude=["revien_bench", "revien_bench.*", "tests", "tests.*"]),
    python_requires=">=3.10",
    install_requires=[
        "pydantic>=2.0",
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "networkx>=3.0",
        "apscheduler>=3.10.0",
        "watchdog>=3.0.0",
        "httpx>=0.25.0",
        "click>=8.1.0",
        # Semantic/vector retrieval is SPINE, not an extra. Graph-only recall
        # has no query-relevance signal beyond keyword overlap (LoCoMo
        # recall@10: 0.05 graph-only vs 0.47 hybrid), so shipping without the
        # semantic layer ships degraded recall. sqlite-vec stores embeddings in
        # the SAME SQLite db; fastembed (bge-small, 384-dim) embeds locally —
        # still zero-network, zero-cloud on the default path.
        "sqlite-vec>=0.1.0",
        "fastembed>=0.3.0",
    ],
    extras_require={
        "dev": ["pytest>=7.4.0", "pytest-asyncio>=0.21.0"],
        "langchain": ["langchain-core>=0.1.0"],
        # MCP surface (LEG P5): `revien mcp` (stdio) and the daemon's /mcp
        # mount (REVIEN_MCP_HTTP=1). Peer dependency — revien imports of the
        # SDK are guarded, so the core install stays lean without it.
        "mcp": ["mcp>=1.28.1"],
        # Hermes Agent memory provider (LEG P6). Peer dependency — revien
        # imports of the Hermes SDK are guarded (HERMES_AVAILABLE), so the core
        # install stays lean without it. `hermes-agent` ships the MemoryProvider
        # ABC this integrates against. Pinned to the verified line (>=0.18.2,
        # 2026.7.7.2). Pre-1.0: the adapter is thin so an ABC bump is a small
        # edit, but pin conservatively.
        "hermes": ["hermes-agent>=0.18.2"],
        # Opt-in Leiden community-detection backend. Compiled deps — not
        # installed by default. Enable with: pip install revien[leiden]
        "leiden": ["leidenalg>=0.10.0", "python-igraph>=0.11.0"],
        # Opt-in neural reranker (TF-IDF + LogisticRegression). Heavy but
        # local — NOT installed by default. Enable with:
        #   pip install revien[neural]
        # When absent, recall() runs base scoring (three-factor + community +
        # confidence) unchanged and the neural adjustment is silently skipped.
        "neural": ["scikit-learn>=1.3.0", "numpy>=1.24.0"],
        # Semantic deps are now CORE (see install_requires). This extra is kept
        # as a backward-compatible alias so `pip install revien[semantic]` and
        # existing docs/scripts keep working. REVIEN_SEMANTIC=0 force-disables;
        # REVIEN_SEMANTIC=require makes a broken/missing layer a hard error.
        "semantic": ["sqlite-vec>=0.1.0", "fastembed>=0.3.0"],
        "all": [
            "langchain-core>=0.1.0",
            "mcp>=1.28.1",
            "leidenalg>=0.10.0",
            "python-igraph>=0.11.0",
            "scikit-learn>=1.3.0",
            "numpy>=1.24.0",
            "sqlite-vec>=0.1.0",
            "fastembed>=0.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "revien=revien.cli:main",
        ],
        # NOTE: a `hermes_agent.plugins` pip entry-point group was considered for
        # Hermes provider discovery, but Hermes' developer guide documents ONLY
        # filesystem discovery (~/.hermes/plugins/memory/<name>/) and the loader
        # source could not be reached to confirm an entry-point group exists.
        # `revien connect hermes` installs the verified filesystem plugin dir;
        # the entry-point path stays out until it's confirmed against their loader.
    },
)
