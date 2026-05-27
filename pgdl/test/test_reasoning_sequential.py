#!/usr/bin/env python3
"""
Simple reasoning benchmark: run n INDEPENDENT queries sequentially, measure total time.

Each query processes 1 HotpotQA sample through the full pipeline:
  cross_encoder (GPU) -> deberta_reader (CPU) -> decode -> LLM

Usage:
    cd /home/why/dbagent/pgdl/test
    uv run python test_reasoning_sequential.py
"""

import time
import psycopg2
from morphingdb_test.config import db_config

# Test: run 1, 5, 10, 20 independent queries
N_QUERIES_LIST = [1, 5, 10, 20]

QUERY_TEMPLATE = """
SELECT unnest(db_agent_single('reasoning', sub_table.reasoning_vec)) AS score
FROM (SELECT * FROM reasoning_vector_test LIMIT 1 OFFSET {offset}) AS sub_table;
"""

def run_sequential(n_queries):
    """Run n independent reasoning queries sequentially."""
    conn = psycopg2.connect(**db_config)
    cur = conn.cursor()
    
    times = []
    for i in range(n_queries):
        sql = QUERY_TEMPLATE.format(offset=i)
        cur.execute("SELECT register_process();")
        
        start = time.time()
        cur.execute(sql)
        rows = cur.fetchall()
        elapsed = time.time() - start
        
        times.append(elapsed)
        print(f"  Query {i+1:2d}: {len(rows):3d} rows in {elapsed:7.2f}s")
    
    conn.close()
    return sum(times), times

def main():
    print("=" * 60)
    print("Reasoning Sequential Benchmark")
    print("=" * 60)
    print()
    
    results = []
    for n in N_QUERIES_LIST:
        print(f"Running {n} independent queries...")
        total, times = run_sequential(n)
        avg = total / n
        results.append((n, total, avg))
        print(f"  Total: {total:.2f}s, Average: {avg:.2f}s")
        print()
    
    print("-" * 60)
    print(f"{'Queries':>10} {'Total (s)':>12} {'Avg (s)':>12}")
    print("-" * 60)
    for n, total, avg in results:
        print(f"{n:>10} {total:>12.2f} {avg:>12.2f}")
    print("-" * 60)

if __name__ == "__main__":
    main()
