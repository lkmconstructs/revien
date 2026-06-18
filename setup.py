from setuptools import setup, find_packages

setup(
    name="revien",
    version="0.1.0",
    description="Graph-based memory engine for AI systems. Memory that returns.",
    author="LKM Constructs LLC",
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
    ],
    extras_require={
        "dev": ["pytest>=7.4.0", "pytest-asyncio>=0.21.0"],
        "langchain": ["langchain-core>=0.1.0"],
        # Opt-in Leiden community-detection backend. Compiled deps — not
        # installed by default. Enable with: pip install revien[leiden]
        "leiden": ["leidenalg>=0.10.0", "python-igraph>=0.11.0"],
        # Opt-in neural reranker (TF-IDF + LogisticRegression). Heavy but
        # local — NOT installed by default. Enable with:
        #   pip install revien[neural]
        # When absent, recall() runs base scoring (three-factor + community +
        # confidence) unchanged and the neural adjustment is silently skipped.
        "neural": ["scikit-learn>=1.3.0", "numpy>=1.24.0"],
        # Opt-in LOCAL-FIRST semantic/vector search. sqlite-vec stores node
        # embeddings in a vec0 virtual table inside the SAME SQLite db (no
        # separate service); fastembed (BAAI/bge-small-en-v1.5, 384-dim) is the
        # LOCAL default embedder — no network on the default path. NOT installed
        # by default. Enable with:
        #   pip install revien[semantic]
        # When absent, recall()/ingest() run the unchanged graph-only path and
        # the semantic layer is silently disabled (REVIEN_SEMANTIC gates it; it
        # defaults on iff sqlite-vec is importable). A cloud embedder (OpenAI)
        # is opt-in via REVIEN_EMBEDDER=openai and is disclosed once on use.
        "semantic": ["sqlite-vec>=0.1.0", "fastembed>=0.3.0"],
        "all": [
            "langchain-core>=0.1.0",
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
    },
)
