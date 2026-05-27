#!/usr/bin/env python3
"""
Independent HotpotQA Benchmark for PGDL db_agent_single('reasoning').

Tests the COST of running n INDEPENDENT reasoning queries, each processing
1 candidate through the full pipeline (cross_encoder → top-1 → deberta_reader → decode).

Each independent query:
    SELECT unnest(db_agent_single('reasoning', sub_table.reasoning_vec)) AS score
    FROM (SELECT * FROM reasoning_vector_test LIMIT 1 OFFSET {i}) AS sub_table;

Usage:
    cd /home/why/dbagent/pgdl/test
    uv run python test_reasoning_independent.py [--queries 10]
"""

import json
import csv
import os
import sys
import time
import argparse
from datetime import datetime
import psycopg2

from morphingdb_test.config import db_config

# Number of independent queries to run sequentially
DEFAULT_N_QUERIES = 10

# Each query processes 1 row (the minimum batch for reasoning pipeline)
QUERY_TEMPLATE = (
    "SELECT unnest(db_agent_single('reasoning', sub_table.reasoning_vec)) AS score "
    "FROM (SELECT * FROM reasoning_vector_test LIMIT 1 OFFSET {offset}) AS sub_table;"
)


def run_single_query(offset):
    """Run one independent reasoning query."""
    sql = QUERY_TEMPLATE.format(offset=offset)
    try:
        conn = psycopg2.connect(**db_config)
        cur = conn.cursor()
        cur.execute("SELECT register_process();")

        start = time.time()
        cur.execute(sql)
        rows = cur.fetchall()
        elapsed = time.time() - start

        conn.close()
        return {
            "offset": offset,
            "status": "success",
            "elapsed": round(elapsed, 4),
            "rows_returned": len(rows),
        }
    except psycopg2.OperationalError as e:
        return {
            "offset": offset,
            "status": "server_crash",
            "error": str(e),
        }
    except Exception as e:
        return {
            "offset": offset,
            "status": "failed",
            "error": str(e),
        }


def run_independent_benchmark(n_queries=DEFAULT_N_QUERIES):
    """Run n independent reasoning queries sequentially."""
    print("=" * 80)
    print("Independent HotpotQA Benchmark")
    print("=" * 80)
    print(f"Running {n_queries} INDEPENDENT reasoning queries sequentially.")
    print("Each query: cross_encoder(top-1) → deberta_reader(2 rows) → decode → LLM")
    print()

    all_results = {
        "timestamp": datetime.now().isoformat(),
        "n_independent_queries": n_queries,
        "results": [],
    }

    total_time = 0
    for i in range(n_queries):
        print(f"  Query {i+1}/{n_queries} (offset={i})...", end=" ", flush=True)
        result = run_single_query(i)
        all_results["results"].append(result)

        if result["status"] == "success":
            total_time += result["elapsed"]
            print(f"OK  {result['elapsed']:.2f}s  ({result['rows_returned']} rows)")
        elif result["status"] == "server_crash":
            print(f"SERVER CRASH -- stopping")
            break
        else:
            print(f"FAILED  {result.get('error', 'unknown')}")
            break

    successful = [r for r in all_results["results"] if r["status"] == "success"]
    if successful:
        avg_time = total_time / len(successful)
        print()
        print("-" * 60)
        print(f"Completed: {len(successful)}/{n_queries} queries")
        print(f"Total time: {total_time:.2f}s")
        print(f"Average per independent query: {avg_time:.2f}s")
        print(f"Estimated time for {n_queries} queries if parallelized: {avg_time:.2f}s (1 pipeline)")
        print("-" * 60)

    # Save results
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(results_dir, f"reasoning_independent_benchmark_{ts}.json")

    # Clean error messages for JSON
    clean_results = []
    for r in all_results["results"]:
        cr = {k: v for k, v in r.items() if k != "error"}
        clean_results.append(cr)
    all_results["results"] = clean_results

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    csv_path = json_path.replace(".json", ".csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["offset", "elapsed_s", "rows_returned", "status"])
        for r in all_results["results"]:
            writer.writerow([
                r.get("offset", ""),
                r.get("elapsed", "N/A"),
                r.get("rows_returned", ""),
                r.get("status", ""),
            ])

    print(f"\nResults saved to:")
    print(f"  JSON: {json_path}")
    print(f"  CSV:  {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Independent HotpotQA benchmark")
    parser.add_argument("--queries", type=int, default=DEFAULT_N_QUERIES,
                        help=f"Number of independent queries to run (default: {DEFAULT_N_QUERIES})")
    args = parser.parse_args()
    run_independent_benchmark(n_queries=args.queries)


if __name__ == "__main__":
    main()
