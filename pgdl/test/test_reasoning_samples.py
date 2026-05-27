#!/usr/bin/env python3
"""
Multi-Sample Benchmark for PGDL db_agent_single('reasoning').

Tests the reasoning_vector_test table with sample sizes: 1, 5, 10, 20.
Each run executes:
    SELECT unnest(db_agent_single('reasoning', sub_table.reasoning_vec)) AS score
    FROM (SELECT * FROM reasoning_vector_test LIMIT {n}) AS sub_table;

Reports per-sample-size latency. Does NOT modify or affect any other dataset logic.

Usage:
    cd /home/why/dbagent/pgdl/test
    uv run python test_reasoning_samples.py
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

SAMPLE_SIZES = [1, 5, 10, 20]
DEFAULT_REPETITIONS = 3

QUERY_TEMPLATE = (
    "SELECT unnest(db_agent_single('reasoning', sub_table.reasoning_vec)) AS score "
    "FROM (SELECT * FROM reasoning_vector_test LIMIT {n_samples}) AS sub_table;"
)


def run_sample(n_samples, repetitions=DEFAULT_REPETITIONS):
    """Run the reasoning query with a given sample size, repeated N times.

    Returns dict with status, total_latency, avg_latency, and row_count.
    """
    sql = QUERY_TEMPLATE.format(n_samples=n_samples)
    try:
        conn = psycopg2.connect(**db_config)
        cur = conn.cursor()
        cur.execute("SELECT register_process();")

        start = time.time()
        total_rows = 0
        for _ in range(repetitions):
            cur.execute(sql)
            total_rows += len(cur.fetchall())
        elapsed = time.time() - start

        conn.close()
        return {
            "n_samples": n_samples,
            "status": "success",
            "total_latency": round(elapsed, 4),
            "avg_latency": round(elapsed / repetitions, 4),
            "repetitions": repetitions,
            "total_rows": total_rows,
        }
    except psycopg2.OperationalError as e:
        return {
            "n_samples": n_samples,
            "status": "server_crash",
            "error": str(e),
            "repetitions": repetitions,
        }
    except Exception as e:
        return {
            "n_samples": n_samples,
            "status": "failed",
            "error": str(e),
            "repetitions": repetitions,
        }


def run_benchmark(repetitions=DEFAULT_REPETITIONS):
    """Run benchmarks for all sample sizes and save results.

    Handles server crashes gracefully: if a sample size crashes the server,
    the script restarts PostgreSQL and continues with the next sample size.
    """
    pg_ctl = "/home/why/dbagent/pg_base/bin/pg_ctl"
    pg_data = "/home/why/dbagent/pg_base/data"
    pg_log = "/home/why/dbagent/pg_base/data/logfile"

    print("=" * 80)
    print("Reasoning Multi-Sample Benchmark")
    print("=" * 80)
    print("Table:       reasoning_vector_test")
    print("Query:       db_agent_single('reasoning', sub_table.reasoning_vec)")
    print("Sample sizes:", SAMPLE_SIZES)
    print("Repetitions:", repetitions)
    print()

    all_results = {
        "timestamp": datetime.now().isoformat(),
        "repetitions": repetitions,
        "results": [],
    }

    for n in SAMPLE_SIZES:
        print(f"  Testing n_samples={n} ...", end=" ", flush=True)
        result = run_sample(n, repetitions)
        all_results["results"].append(result)

        if result["status"] == "success":
            print(f"OK  total={result['total_latency']:.4f}s  "
                  f"avg={result['avg_latency']:.4f}s  rows={result['total_rows']}")
        elif result["status"] == "server_crash":
            print(f"SERVER CRASH -- restarting PostgreSQL...")
            # Restart server to recover from crash
            import subprocess
            subprocess.run([pg_ctl, "-D", pg_data, "stop", "-m", "immediate"],
                          capture_output=True, timeout=30)
            time.sleep(1)
            subprocess.run([pg_ctl, "-D", pg_data, "start", "-l", pg_log],
                          capture_output=True, timeout=30)
            time.sleep(3)
            print(f"  Server restarted. Continuing with next sample size.")
        else:
            print(f"FAILED  {result.get('error', 'unknown error')}")

    # Print summary table
    print()
    print("-" * 70)
    print(f"{'Samples':>10} {'Total (s)':>12} {'Avg (s)':>12} {'Rows':>8} {'Status':>12}")
    print("-" * 70)
    for r in all_results["results"]:
        if r["status"] == "success":
            print(f"{r['n_samples']:>10} {r['total_latency']:>12.4f} "
                  f"{r['avg_latency']:>12.4f} {r['total_rows']:>8} {r['status']:>12}")
        else:
            print(f"{r['n_samples']:>10} {'N/A':>12} {'N/A':>12} {'N/A':>8} {r['status']:>12}")
    print("-" * 70)

    # Save results
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(results_dir, f"reasoning_sample_benchmark_{ts}.json")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # Also write a simple CSV
    csv_path = json_path.replace(".json", ".csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["n_samples", "total_latency_s", "avg_latency_s", "total_rows", "status"])
        for r in all_results["results"]:
            if r["status"] == "success":
                writer.writerow([
                    r["n_samples"], r["total_latency"], r["avg_latency"],
                    r["total_rows"], r["status"],
                ])
            else:
                writer.writerow([
                    r["n_samples"], "N/A", "N/A", "N/A", r["status"],
                ])

    print(f"\nResults saved to:")
    print(f"  JSON: {json_path}")
    print(f"  CSV:  {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Reasoning multi-sample benchmark")
    parser.add_argument("--reps", type=int, default=DEFAULT_REPETITIONS,
                        help=f"Number of repetitions per sample size (default: {DEFAULT_REPETITIONS})")
    args = parser.parse_args()
    run_benchmark(repetitions=args.reps)


if __name__ == "__main__":
    main()
