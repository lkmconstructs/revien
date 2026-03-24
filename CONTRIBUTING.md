# Contributing to Revien

Thank you for your interest in contributing to Revien. This document covers the process for contributing to the project.

## How to Contribute

### Reporting Issues

Open an issue on GitHub with a clear description of the problem. Include:

- What you expected to happen
- What actually happened
- Steps to reproduce
- Your Python version and OS

### Submitting Changes

1. Fork the repository
2. Create a feature branch from `main`
3. Make your changes
4. Add or update tests for your changes
5. Run the full test suite: `pytest tests/ -v`
6. Submit a pull request against `main`

### Code Style

- Python 3.10+
- Type hints on all public functions
- Pydantic v2 for all data models
- Docstrings on all public classes and functions

### Test Requirements

All pull requests must maintain 100% test pass rate. The current suite covers:

- Graph CRUD operations
- Ingestion pipeline (entity, topic, decision, fact, preference extraction)
- Retrieval engine (three-factor scoring, graph traversal)
- Daemon API endpoints
- Adapter functionality
- End-to-end integration

Run tests with:

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

### Adapters

New adapters are welcome. To add support for a new AI system:

1. Create `revien/adapters/your_adapter.py`
2. Implement the `RevienAdapter` base class from `revien/adapters/base.py`
3. Add a CLI connect command in `revien/cli.py`
4. Add tests in `tests/test_adapters.py`
5. Document the adapter in the README

### Neural Classifiers

The `revien/neural/` module contains scaffolded interfaces for ML-enhanced extraction and scoring. If you want to contribute a trained classifier:

1. It must implement the interface in `revien/neural/classifier.py`
2. It must produce output in the same format as the rule-based extractor
3. It must run on CPU via ONNX runtime (no GPU requirement)
4. Model files must be under 50MB
5. Include training scripts and evaluation metrics

## What We're Looking For

- New adapters for popular AI tools
- Improved extraction patterns for the rule-based engine
- Performance optimizations in the graph walker
- Documentation improvements
- Bug fixes with test coverage

## What We're Not Looking For

- Changes that add GPU requirements to core functionality
- Embedding-based retrieval (this is a graph engine, not a vector store)
- Features that require cloud connectivity for local mode
- Breaking changes to the REST API without discussion

## License

By contributing to Revien, you agree that your contributions will be licensed under the Apache License 2.0.
