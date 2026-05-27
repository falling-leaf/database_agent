#!/usr/bin/env python3
"""
NeurDB Baseline Concurrency Test.

Finds the maximum safe concurrency level for predict_batch_float8 queries
using binary search with an upper limit of 400 threads.
Tests all 9 datasets (3 series, 3 text, 3 image).

Key features:
- Binary search for max concurrency (upper limit: 400)
- Orphan process cleanup via process groups
- Any single thread failure = test fails for that concurrency level

Usage:
    python test_neurdb_concurrency.py
"""

import json
import os
import signal
import sys
import time
import multiprocessing
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

import psycopg2
from morphingdb_test.config import db_config

# =============================================================================
# Configuration
# =============================================================================
MAX_CONCURRENCY_LIMIT = 400
COARSE_STEP = 50  # Coarse search step size
QUERY_TIMEOUT = 600  # seconds per query
BINARY_SLEEP = 2  # seconds between binary search rounds

# =============================================================================
# 9 datasets: 3 series (CPU), 3 text (GPU), 3 image (GPU)
# =============================================================================
DATASETS = [
    # Series tests (CPU)
    {"name": "slice_predict", "type": "series", "symbol": "cpu",
     "table": "slice_test", "model": "slice", "column": "data"},
    {"name": "swarm_predict", "type": "series", "symbol": "cpu",
     "table": "swarm_test", "model": "swarm", "column": "data"},
    {"name": "year_predict_test", "type": "series", "symbol": "cpu",
     "table": "year_predict_test", "model": "year_predict", "column": "data"},
    # Text/NLP tests (GPU)
    {"name": "imdb_vector_predict", "type": "text", "symbol": "gpu",
     "table": "imdb_vector_test", "model": "sst2_vec", "column": "comment_vec"},
    {"name": "financial_phrasebank_predict", "type": "text", "symbol": "gpu",
     "table": "financial_phrasebank_vector_test", "model": "finance", "column": "comment_vec"},
    {"name": "nlp_vector_predict", "type": "text", "symbol": "gpu",
     "table": "nlp_vector_test", "model": "sst2_vec", "column": "comment_vec"},
    # Image tests (GPU)
    {"name": "cifar_image_predict", "type": "image", "symbol": "gpu",
     "table": "cifar_image_vector_table", "model": "googlenet_cifar10", "column": "image_vector"},
    {"name": "stanford_dogs_image_predict", "type": "image", "symbol": "gpu",
     "table": "stanford_dogs_image_vector_table", "model": "alexnet_stanford_dogs", "column": "image_vector"},
    {"name": "imagenet_image_predict", "type": "image", "symbol": "gpu",
     "table": "imagenet_image_vector_table", "model": "defect_vec", "column": "image_vector"},
]


def build_sql_query(ds_info, row_count):
    """Build the predict_batch_float8 SQL query."""
    return (
        f"select predict_batch_float8('{ds_info['model']}', '{ds_info['symbol']}', "
        f"{ds_info['column']}) over (rows between current row and 31 following) "
        f"from {ds_info['table']} limit {row_count};"
    )


# =============================================================================
# Worker: runs a single query task in a subprocess with process group isolation
# =============================================================================
def run_single_task_worker(task_id, row_count, query_times, symbol, sql_query):
    """Execute a SQL query repeatedly in an isolated process with its own process group."""
    # Create new process group for this worker so we can kill it and all children
    try:
        os.setpgid(0, 0)
    except OSError:
        pass  # Already leader or not supported

    conn = None
    try:
        conn = psycopg2.connect(**db_config)
        cur = conn.cursor()
        cur.execute("select register_process();")

        start_time = time.time()
        return_rows = 0
        for _ in range(query_times):
            cur.execute(sql_query)
            return_rows += len(cur.fetchall())

        end_time = time.time()
        conn.close()
        conn = None
        print(f"  [OK] Task {task_id}: rows={return_rows}, expected={row_count * query_times}, "
              f"latency={end_time - start_time:.2f}s")
        return {"task_id": task_id, "status": "success", "latency": end_time - start_time}
    except Exception as e:
        print(f"  [FAIL] Task {task_id}: {e}")
        return {"task_id": task_id, "status": "failed", "error": str(e)}
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def kill_process_group(pid):
    """Kill an entire process group to prevent orphans."""
    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        pass


# =============================================================================
# Concurrency test with orphan cleanup
# =============================================================================
def test_concurrency_level(concurrency_level, row_count, query_times, symbol, sql_query):
    """
    Test a specific concurrency level. Returns True if ALL threads pass.
    Uses ProcessPoolExecutor with orphan cleanup.
    """
    print(f"  Testing concurrency level: {concurrency_level}")

    # Use multiprocessing context with spawn to ensure clean process groups
    ctx = multiprocessing.get_context('spawn')
    pool = ctx.Pool(
        processes=concurrency_level,
        initializer=os.setpgid,
        initargs=(0, 0)
    )

    results = []
    try:
        async_results = []
        for i in range(concurrency_level):
            ar = pool.apply_async(
                run_single_task_worker,
                args=(i, row_count, query_times, symbol, sql_query)
            )
            async_results.append(ar)

        # Collect results with timeout
        for i, ar in enumerate(async_results):
            try:
                result = ar.get(timeout=QUERY_TIMEOUT)
                results.append(result)
            except Exception as e:
                results.append({"task_id": i, "status": "failed", "error": str(e)})
    finally:
        # Ensure all workers are terminated - no orphans
        pool.terminate()
        pool.join()

    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "failed"]
    all_passed = len(failed) == 0

    print(f"  Level {concurrency_level}: {len(successful)}/{len(results)} passed "
          f"{'[PASS]' if all_passed else '[FAIL]'}")
    if failed:
        for f in failed:
            print(f"    Failed task {f['task_id']}: {f.get('error', 'unknown')}")

    return all_passed, results


# =============================================================================
# Binary search for max concurrency
# =============================================================================
def binary_search_max_concurrency(low, high, row_count, query_times, symbol, sql_query):
    """
    Binary search for the exact maximum concurrency level where ALL threads pass.
    Returns the max concurrency level.
    """
    max_pass = low - 1  # Highest level that passed

    while low <= high:
        mid = (low + high) // 2
        if mid == 0:
            low = 1
            continue

        all_passed, _ = test_concurrency_level(mid, row_count, query_times, symbol, sql_query)

        if all_passed:
            max_pass = mid
            print(f"  -> {mid} passed, searching [{mid + 1}, {high}]")
            low = mid + 1
        else:
            print(f"  -> {mid} failed, searching [{low}, {mid - 1}]")
            high = mid - 1

        time.sleep(BINARY_SLEEP)  # Cool down between rounds

    return max_pass


def find_max_concurrency_with_coarse(row_count=1000, query_times=1, symbol='cpu', sql_query=None):
    """
    Two-phase search:
    Phase 1: Coarse stepping to find approximate boundary quickly
    Phase 2: Binary search for exact maximum
    """
    # Phase 1: Coarse search
    print(f"  Phase 1: Coarse search (step={COARSE_STEP})")
    last_pass = 0
    current = COARSE_STEP

    while current <= MAX_CONCURRENCY_LIMIT:
        all_passed, _ = test_concurrency_level(current, row_count, query_times, symbol, sql_query)
        if all_passed:
            last_pass = current
            if current >= MAX_CONCURRENCY_LIMIT:
                print(f"  Reached MAX limit {MAX_CONCURRENCY_LIMIT} - all passed!")
                return MAX_CONCURRENCY_LIMIT
            current += COARSE_STEP
        else:
            print(f"  Failed at {current}, last pass was {last_pass}")
            break

    # Phase 2: Binary search
    if last_pass == 0:
        # Even COARSE_STEP failed, search [1, COARSE_STEP-1]
        print(f"  Phase 2: Binary search in [1, {COARSE_STEP - 1}]")
        return binary_search_max_concurrency(1, COARSE_STEP - 1, row_count, query_times, symbol, sql_query)
    elif current > MAX_CONCURRENCY_LIMIT or current - last_pass <= 1:
        return last_pass
    else:
        print(f"  Phase 2: Binary search in [{last_pass + 1}, {min(current - 1, MAX_CONCURRENCY_LIMIT)}]")
        return binary_search_max_concurrency(
            last_pass + 1, min(current - 1, MAX_CONCURRENCY_LIMIT),
            row_count, query_times, symbol, sql_query
        )


# =============================================================================
# Main: run tests on all 9 datasets
# =============================================================================
def run_all_tests():
    """Run concurrency tests on all 9 datasets and report results."""
    print("=" * 80)
    print("NeurDB Baseline Concurrency Test")
    print(f"Upper limit: {MAX_CONCURRENCY_LIMIT} threads")
    print(f"Binary search + coarse stepping")
    print("=" * 80)

    all_results = {}
    ROW_COUNT = 1000

    for ds in DATASETS:
        ds_name = ds["name"]
        ds_type = ds["type"]
        symbol = ds["symbol"]
        sql_query = build_sql_query(ds, ROW_COUNT)

        print(f"\n{'=' * 60}")
        print(f"Dataset: {ds_name} [{ds_type.upper()}]")
        print(f"Table: {ds['table']}, Model: {ds['model']}, Symbol: {symbol}")
        print(f"Row count per thread: {ROW_COUNT}")
        print(f"{'=' * 60}")

        max_conc = find_max_concurrency_with_coarse(
            row_count=ROW_COUNT, query_times=1, symbol=symbol, sql_query=sql_query
        )

        all_results[ds_name] = {
            "type": ds_type,
            "table": ds["table"],
            "model": ds["model"],
            "symbol": symbol,
            "row_count": ROW_COUNT,
            "max_concurrency": max_conc,
        }

        print(f"\n>>> {ds_name}: MAX_CONCURRENCY = {max_conc}")

    # Print summary table
    print(f"\n{'=' * 80}")
    print("SUMMARY: NeurDB Baseline Max Concurrency (all 9 datasets)")
    print(f"{'=' * 80}")
    print(f"{'Dataset':<40} {'Type':<10} {'Max Concurrency':>15}")
    print(f"{'-' * 40} {'-' * 10} {'-' * 15}")

    for ds_name, info in all_results.items():
        print(f"{ds_name:<40} {info['type']:<10} {info['max_concurrency']:>15}")

    # Overall minimum (bottleneck)
    min_conc = min(info["max_concurrency"] for info in all_results.values())
    print(f"\nOverall minimum (guaranteed for all datasets): {min_conc}")

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_data = {
        "timestamp": timestamp,
        "test": "neurdb_baseline_concurrency",
        "max_limit": MAX_CONCURRENCY_LIMIT,
        "row_count": ROW_COUNT,
        "results": all_results,
        "overall_min_concurrency": min_conc,
    }

    filename = f"neurdb_concurrency_results_{timestamp}.json"
    filepath = os.path.join(os.path.dirname(__file__), filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {filepath}")

    return all_results


if __name__ == "__main__":
    run_all_tests()
