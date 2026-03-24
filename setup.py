from setuptools import setup, find_packages

setup(
    name="revien",
    version="0.1.0",
    description="Graph-based memory engine for AI systems. Memory that returns.",
    author="LKM Constructs LLC",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "pydantic>=2.0",
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "apscheduler>=3.10.0",
        "watchdog>=3.0.0",
        "httpx>=0.25.0",
        "click>=8.1.0",
    ],
    extras_require={
        "dev": ["pytest>=7.4.0", "pytest-asyncio>=0.21.0"],
    },
    entry_points={
        "console_scripts": [
            "revien=revien.cli:main",
        ],
    },
)
