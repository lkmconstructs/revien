"""
Revien Semantic Layer — OPT-IN, LOCAL-FIRST hybrid vector retrieval.

This package adds an optional semantic/vector search layer that sits OVER the
existing graph retrieval. Its entire surface is guarded: when the `semantic`
extra (sqlite-vec + fastembed) is NOT installed, ``SemanticIndex`` self-disables
and every existing graph-retrieval path runs byte-for-byte unchanged.

Enable with::

    pip install revien[semantic]

See ``revien/semantic/index.py`` for the storage/embedding/search design and
``setup.py`` extras_require["semantic"].
"""

from .index import (
    SemanticIndex,
    EmbeddingProvider,
    FastEmbedProvider,
    OpenAIEmbeddingProvider,
    build_embedder,
    SEMANTIC_AVAILABLE,
)

__all__ = [
    "SemanticIndex",
    "EmbeddingProvider",
    "FastEmbedProvider",
    "OpenAIEmbeddingProvider",
    "build_embedder",
    "SEMANTIC_AVAILABLE",
]
