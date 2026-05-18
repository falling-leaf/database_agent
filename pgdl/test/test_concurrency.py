#!/usr/bin/env python3
"""
Concurrency Stress Test for PGDL/MorphingDB.

Finds the maximum safe concurrency level for predict_batch_float8 queries
using a two-phase search: coarse-grained stepping + binary search.
Tests all 9 datasets (3 series, 3 NLP, 3 image).

Usage:
    python concurrency_test.py
"""

import json
import csv
import time
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict

import psycopg2
from morphingdb_test.config import db_config
from json_to_csv import json_to_csv


# =============================================================================
# Worker: runs a single query task in a subprocess
# =============================================================================
def run_single_task_worker(task_id, row_count, query_times, symbol, sql_query):
    """Execute a SQL query repeatedly in an isolated process."""
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
        print(f"  Task {task_id}: rows={return_rows}, expected={row_count * query_times}")
        return {"task_id": task_id, "status": "success", "latency": end_time - start_time}
    except Exception as e:
        print(f"  Error in task {task_id}: {e}")
        return {"task_id": task_id, "status": "failed", "error": str(e)}


# =============================================================================
# Concurrency test helpers
# =============================================================================
def test_concurrency(concurrency_level, row_count, query_times, symbol, sql_query):
    """Test a specific concurrency level and return success/failure stats."""
    print(f"  Testing concurrency level: {concurrency_level}")

    with ProcessPoolExecutor(max_workers=concurrency_level) as executor:
        futures = [executor.submit(run_single_task_worker, i, row_count, query_times, symbol, sql_query)
                   for i in range(concurrency_level)]
        results = []
        for i, future in enumerate(futures):
            try:
                results.append(future.result(timeout=600))
            except Exception as e:
                results.append({"task_id": i, "status": "failed", "error": str(e)})

    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "failed"]
    success_rate = len(successful) / len(results) if results else 0

    print(f"  Level {concurrency_level}: {len(successful)}/{len(results)} passed ({success_rate:.0%})")
    return {
        "concurrency_level": concurrency_level,
        "total_tasks": len(results),
        "successful_tasks": len(successful),
        "failed_tasks": len(failed),
        "success_rate": success_rate,
        "results": results,
        "row_count_per_task": row_count,
        "query_times_per_task": query_times,
        "sql_query": sql_query,
    }


def find_exact_max_concurrency(start, end, row_count, query_times, symbol, sql_query):
    """Binary search for the exact maximum safe concurrency level."""
    print(f"  Binary search in [{start}, {end}]")
    all_results = []

    while start <= end:
        if start == end:
            result = test_concurrency(start, row_count, query_times, symbol, sql_query)
            all_results.append(result)
            return all_results, start if result["failed_tasks"] == 0 else start - 1

        mid = (start + end) // 2
        result = test_concurrency(mid, row_count, query_times, symbol, sql_query)
        all_results.append(result)

        if result["failed_tasks"] == 0:
            start = mid + 1
        else:
            end = mid - 1
        time.sleep(1)

    return all_results, end


def find_max_concurrency_optimized(start_level=1, max_level=1000, coarse_step=32,
                                    row_count=1000, query_times=1, symbol='cpu', sql_query=None):
    """
    Two-phase concurrency search:
    Phase 1: coarse stepping to find approximate boundary
    Phase 2: binary search for exact maximum
    """
    print(f"Concurrency search: start={start_level}, max={max_level}, step={coarse_step}")
    all_results = []

    # Phase 1: coarse search
    current = start_level
    last_success = start_level - 1
    while current <= max_level:
        result = test_concurrency(current, row_count, query_times, symbol, sql_query)
        all_results.append(result)
        if result["failed_tasks"] == 0:
            last_success = current
            current += coarse_step
        else:
            print(f"  Failure at level {current}, last success was {last_success}")
            break

    # Phase 2: binary search if needed
    if last_success >= max_level:
        return all_results, max_level
    elif last_success < start_level:
        return all_results, 0 if start_level == 1 else -1
    elif current - last_success == 1:
        return all_results, last_success
    else:
        binary_results, precise = find_exact_max_concurrency(
            last_success + 1, current - 1, row_count, query_times, symbol, sql_query)
        all_results.extend(binary_results)
        return all_results, precise


# =============================================================================
# Query definitions: 9 test cases across 3 data types
# =============================================================================
ORIGINAL_SQL_QUERIES = [
    # Series tests (CPU)
    {"name": "slice_predict", "table": "slice_test", "model": "slice", "column": "data",
     "query": "select predict_batch_float8('{model}', '{symbol}', {column}) over (rows between current row and 31 following) from {table} limit {row_count};"},
    {"name": "swarm_predict", "table": "swarm_test", "model": "swarm", "column": "data",
     "query": "select predict_batch_float8('{model}', '{symbol}', {column}) over (rows between current row and 31 following) from {table} limit {row_count};"},
    {"name": "year_predict_test", "table": "year_predict_test", "model": "year_predict", "column": "data",
     "query": "select predict_batch_float8('{model}', '{symbol}', {column}) over (rows between current row and 31 following) from {table} limit {row_count};"},
    # NLP tests (GPU)
    {"name": "imdb_vector_predict", "table": "imdb_vector_test", "model": "sst2_vec", "column": "comment_vec",
     "query": "select predict_batch_float8('{model}', '{symbol}', {column}) over (rows between current row and 31 following) from {table} limit {row_count};"},
    {"name": "financial_phrasebank_predict", "table": "financial_phrasebank_vector_test", "model": "finance", "column": "comment_vec",
     "query": "select predict_batch_float8('{model}', '{symbol}', {column}) over (rows between current row and 31 following) from {table} limit {row_count};"},
    {"name": "nlp_vector_predict", "table": "nlp_vector_test", "model": "sst2_vec", "column": "comment_vec",
     "query": "select predict_batch_float8('{model}', '{symbol}', {column}) over (rows between current row and 31 following) from {table} limit {row_count};"},
    # Image tests (GPU)
    {"name": "cifar_image_predict", "table": "cifar_image_vector_table", "model": "googlenet_cifar10", "column": "image_vector",
     "query": "select predict_batch_float8('{model}', '{symbol}', {column}) over (rows between current row and 31 following) from {table} limit {row_count};"},
    {"name": "stanford_dogs_image_predict", "table": "stanford_dogs_image_vector_table", "model": "alexnet_stanford_dogs", "column": "image_vector",
     "query": "select predict_batch_float8('{model}', '{symbol}', {column}) over (rows between current row and 31 following) from {table} limit {row_count};"},
    {"name": "imagenet_image_predict", "table": "imagenet_image_vector_table", "model": "defect_vec", "column": "image_vector",
     "query": "select predict_batch_float8('{model}', '{symbol}', {column}) over (rows between current row and 31 following) from {table} limit {row_count};"},
]

# db_agent_single queries (all commented out - enable as needed)
NEW_SQL_QUERIES = []


def format_query(sql_info, row_count, symbol):
    """Format a query template with actual values."""
    if 'model' in sql_info:
        return sql_info['query'].format(
            model=sql_info['model'], symbol=symbol,
            column=sql_info['column'], table=sql_info['table'],
            row_count=row_count)
    else:
        template = sql_info['query'].format(
            func_type=sql_info['func_type'],
            column=sql_info['column'], table=sql_info['table'])
        return template.format(row_count=row_count)


def get_symbol(sql_info):
    """Determine CPU or GPU based on test type."""
    if 'model' in sql_info:
        return 'cpu' if sql_info['name'] in ['slice_predict', 'swarm_predict', 'year_predict_test'] else 'gpu'
    return 'cpu' if sql_info.get('func_type') == 'series' else 'gpu'


def run_concurrency_tests_for_queries(queries, query_type):
    """Run concurrency tests for a set of queries across different row counts."""
    results_by_query = {}
    ROW_COUNTS = [1000]

    for sql_info in queries:
        symbol = get_symbol(sql_info)
        print(f"\n{'='*60}")
        print(f"Testing {query_type}: {sql_info['name']}")
        print(f"Table: {sql_info['table']}, {'Model: ' + sql_info.get('model', 'N/A')}, Column: {sql_info['column']}")
        print(f"{'='*60}")

        query_results = {}
        for row_count in ROW_COUNTS:
            print(f"\n  Row count: {row_count}")
            formatted_query = format_query(sql_info, row_count, symbol)
            all_results, max_concurrency = find_max_concurrency_optimized(
                row_count=row_count, query_times=1, symbol=symbol, sql_query=formatted_query)
            query_results[row_count] = {"results": all_results, "max_concurrency": max_concurrency}

        results_by_query[sql_info['name']] = {
            "table": sql_info['table'],
            "model": sql_info.get('model', sql_info.get('func_type', 'N/A')),
            "column": sql_info['column'],
            "row_counts": query_results,
        }

    return results_by_query


def run_concurrency_test():
    """Main entry: run concurrency tests for all query sets."""
    print("Optimized Concurrency Stress Test")
    print("="*80)

    print("\n--- ORIGINAL queries (predict_batch_float8) ---")
    original_results = run_concurrency_tests_for_queries(ORIGINAL_SQL_QUERIES, "ORIGINAL")

    all_results_dict = {
        "timestamp": datetime.now().isoformat(),
        "test_type": "optimized_concurrency_stress_test",
        "original_queries_results": original_results,
    }

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"optimized_concurrency_test_results_{timestamp}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(all_results_dict, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {filename}")
    return original_results


if __name__ == "__main__":
    run_concurrency_test()
