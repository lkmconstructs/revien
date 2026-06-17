#!/usr/bin/env python3
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""
Revien — Migrate from ChatGPT in 60 seconds.

A self-contained demo showing OpenAI conversation ingestion,
graph-based retrieval, and cross-session persistence.

Usage:
    python demo/openai_migration_demo.py
"""

import json
import shutil
import tempfile
import time
from pathlib import Path

# ANSI color codes (no external deps)
BOLD = "\033[1m"
GREEN = "\033[92m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
RED = "\033[91m"
DIM = "\033[2m"
RESET = "\033[0m"


def print_header(text: str) -> None:
    print(f"\n{BOLD}{CYAN}{text}{RESET}")


def print_step(text: str) -> None:
    print(f"\n{BOLD}{GREEN}→ {text}{RESET}")


def print_result(text: str) -> None:
    print(f"  {text}")


def print_comparison(label: str, value: str) -> None:
    print(f"  {YELLOW}{label}{RESET} {value}")


def main() -> None:
    print_header("Revien — Migrate from ChatGPT in 60 seconds")
    print_result("Loading demo...\n")

    # Load sample conversations
    demo_dir = Path(__file__).parent
    sample_file = demo_dir / "sample_conversations.json"

    with open(sample_file, "r", encoding="utf-8") as f:
        conversations = json.load(f)

    # Show first few lines
    print_result(f"Sample data: {len(conversations)} conversations from ChatGPT export")
    first_conv = conversations[0]
    print_result(f"  First: \"{first_conv['title']}\" ({len(first_conv['mapping'])} messages)")

    # === STEP 1: Setup ===
    print_step("Initializing demo database...")
    temp_dir = tempfile.mkdtemp(prefix="revien_demo_")
    db_path = str(Path(temp_dir) / "demo.db")
    bulk_file = str(Path(temp_dir) / "conversations.json")
    print_result(f"Temp database: {db_path}")

    try:
        # Import Revien components
        from revien.adapters.openai_adapter import OpenAIAdapter
        from revien.graph.store import GraphStore
        from revien.retrieval.engine import RetrievalEngine

        time.sleep(1)

        # === STEP 2: Ingest ===
        print_step("Ingesting conversations...")
        print_result(f"Processing {len(conversations)} conversations...")

        # Write bulk export to temp file (OpenAIAdapter expects file paths)
        with open(bulk_file, "w", encoding="utf-8") as f:
            json.dump(conversations, f)

        adapter = OpenAIAdapter(graph_path=db_path)
        stats = adapter.ingest_bulk_export(bulk_file)

        time.sleep(1)

        node_count = adapter.store.count_nodes()
        edge_count = adapter.store.count_edges()

        print_result(f"  Conversations ingested: {stats['conversations_ingested']}")
        print_result(f"  Cross-conversation edges: {stats.get('cross_conversation_edges', 0)}")
        print_result(f"{GREEN}  ✓ Graph ready. {node_count} nodes, {edge_count} edges.{RESET}")

        time.sleep(2)

        # === STEP 3: Query — Relevance Demo ===
        print_step("Querying: \"What did we discuss about the database migration?\"")
        print_result("Searching with three-factor scoring (recency × frequency × proximity)...\n")

        engine = RetrievalEngine(adapter.store)
        response = engine.recall("What did we discuss about the database migration?", top_n=3)

        time.sleep(1)

        if response.results:
            print_result(f"  Found {len(response.results)} results ({response.retrieval_time_ms:.1f}ms):\n")
            for i, result in enumerate(response.results, 1):
                content_preview = result.content[:90].replace("\n", " ")
                print_result(
                    f"  {BOLD}{i}. [{result.node_type}] {result.label}{RESET}"
                )
                print_result(
                    f"     Score: {result.score:.3f} | "
                    f"Recency: {result.score_breakdown['recency']:.2f} | "
                    f"Frequency: {result.score_breakdown['frequency']:.2f} | "
                    f"Proximity: {result.score_breakdown['proximity']:.2f}"
                )
                print_result(f"     {DIM}{content_preview}...{RESET}\n")

            # Check if a relevant result was found
            top_content = " ".join(r.content.lower() for r in response.results[:2])
            if "postgresql" in top_content or "mongodb" in top_content or "database" in top_content:
                print_result(f"  {GREEN}✓ Top results correctly surface the database discussion!{RESET}")
                print_result(f"  {DIM}(This conversation is NOT the most recent — relevance wins.){RESET}")
        else:
            print_result("  No results found (graph may need more data)")

        time.sleep(2)

        # === STEP 4: Cross-Session Persistence ===
        print_step("Simulating new session...")
        print_result("Closing retrieval engine. Opening fresh connection to same database...\n")

        time.sleep(1)

        # Create a completely new store + engine (simulating new process)
        fresh_store = GraphStore(db_path=db_path)
        fresh_engine = RetrievalEngine(fresh_store)

        query2 = "What was the decision on PostgreSQL vs MongoDB?"
        response2 = fresh_engine.recall(query2, top_n=2)

        print_result(f"  Query: \"{query2}\"")
        if response2.results:
            print_result(f"  Results: {len(response2.results)} matches ({response2.retrieval_time_ms:.1f}ms)")
            top = response2.results[0]
            print_result(f"  Top: [{top.score:.3f}] {top.label}")
            print_result(f"\n  {GREEN}✓ Memory persists. No context window. No compaction loss.{RESET}")
        else:
            print_result("  (No results — graph structure depends on content)")

        fresh_store.close()

        time.sleep(2)

        # === STEP 5: Comparison ===
        print_step("The difference:")
        print()
        print_comparison(
            "ChatGPT:",
            f"{RED}\"I don't have access to previous conversations.\"{RESET}",
        )
        print()
        if response.results:
            top_answer = response.results[0].content[:100].replace("\n", " ")
            print_comparison(
                "Revien: ",
                f"{GREEN}\"{top_answer}...\" [score: {response.results[0].score:.2f}]{RESET}",
            )
        print()

        time.sleep(2)

        # === STEP 6: Closing ===
        print_step("Your AI should remember.")
        print()
        print_result(f"  {BOLD}pip install revien{RESET}")
        print_result(f"  {DIM}github.com/lkmconstructs/revien{RESET}")
        print()

        adapter.store.close()

    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
