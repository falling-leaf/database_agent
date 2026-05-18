#!/usr/bin/env python3
"""
Single-Thread Latency Benchmark for PGDL/MorphingDB.

Runs predict_batch_float8 queries sequentially across different row counts
and query repetitions. Tests both original (predict_batch_float8) and
db_agent_single query styles.

Usage:
    python single_test.py              # runs both query sets
    python -c "from single_test import run_single_slice_test; run_single_slice_test(False)"  # original only
"""

import json
import csv
import time
import sys
from datetime import datetime
from collections import defaultdict
import psycopg2
from concurrent.futures import ProcessPoolExecutor

from morphingdb_test.config import db_config
from json_to_csv import json_to_csv


# =============================================================================
# Worker: runs a single query task in a subprocess
# =============================================================================
def run_single_task_worker(task_id, row_count, query_times=10, symbol='cpu', sql_query=None):
    """Execute a SQL query repeatedly in an isolated process."""
    try:
        conn = psycopg2.connect(**db_config)
        cur = conn.cursor()
        cur.execute("select register_process();")

        start_time = time.time()
        sql = sql_query.format(row_count=row_count, symbol=symbol)
        return_rows = 0
        for _ in range(query_times):
            cur.execute(sql)
            return_rows += len(cur.fetchall())

        end_time = time.time()
        conn.close()

        print(f"  task {task_id}: rows={return_rows}, expected={row_count * query_times}")
        return {"task_id": task_id, "status": "success", "latency": end_time - start_time}
    except Exception as e:
        print(f"  Error in task {task_id}: {e}")
        return {"task_id": task_id, "status": "failed", "error": str(e)}


# =============================================================================
# Query definitions: 9 test cases across 3 data types
# =============================================================================
# Original: predict_batch_float8 window function queries
SQL_QUERIES = [
    # --- Series tests (CPU) ---
    {
        "name": "slice_predict", "table": "slice_test",
        "model": "slice", "column": "data",
        "query": "select predict_batch_float8('{model}', '{symbol}', {column}) over (rows between current row and 31 following) from {table} limit {row_count};"
    },
    {
        "name": "swarm_predict", "table": "swarm_test",
        "model": "swarm", "column": "data",
        "query": "select predict_batch_float8('{model}', '{symbol}', {column}) over (rows between current row and 31 following) from {table} limit {row_count};"
    },
    {
        "name": "year_predict_test", "table": "year_predict_test",
        "model": "year_predict", "column": "data",
        "query": "select predict_batch_float8('{model}', '{symbol}', {column}) over (rows between current row and 31 following) from {table} limit {row_count};"
    },
    # --- NLP / text tests (GPU) ---
    {
        "name": "imdb_vector_predict", "table": "imdb_vector_test",
        "model": "sst2_vec", "column": "comment_vec",
        "query": "select predict_batch_float8('{model}', '{symbol}', {column}) over (rows between current row and 31 following) from {table} limit {row_count};"
    },
    {
        "name": "financial_phrasebank_predict", "table": "financial_phrasebank_vector_test",
        "model": "finance", "column": "comment_vec",
        "query": "select predict_batch_float8('{model}', '{symbol}', {column}) over (rows between current row and 31 following) from {table} limit {row_count};"
    },
    {
        "name": "nlp_vector_predict", "table": "nlp_vector_test",
        "model": "sst2_vec", "column": "comment_vec",
        "query": "select predict_batch_float8('{model}', '{symbol}', {column}) over (rows between current row and 31 following) from {table} limit {row_count};"
    },
    # --- Image tests (GPU) ---
    {
        "name": "cifar_image_predict", "table": "cifar_image_vector_table",
        "model": "googlenet_cifar10", "column": "image_vector",
        "query": "select predict_batch_float8('{model}', '{symbol}', {column}) over (rows between current row and 31 following) from {table} limit {row_count};"
    },
    {
        "name": "stanford_dogs_image_predict", "table": "stanford_dogs_image_vector_table",
        "model": "alexnet_stanford_dogs", "column": "image_vector",
        "query": "select predict_batch_float8('{model}', '{symbol}', {column}) over (rows between current row and 31 following) from {table} limit {row_count};"
    },
    {
        "name": "imagenet_image_predict", "table": "imagenet_image_vector_table",
        "model": "defect_vec", "column": "image_vector",
        "query": "select predict_batch_float8('{model}', '{symbol}', {column}) over (rows between current row and 31 following) from {table} limit {row_count};"
    },
]

# Alternative: db_agent_single function queries (9 test cases)
NEW_SQL_QUERIES = [
    # --- Series tests using db_agent_single (CPU) ---
    {
        "name": "slice_db_agent", "table": "slice_test",
        "func_type": "series", "column": "data",
        "query": "select unnest(db_agent_single('{func_type}', sub_table.{column})) AS score FROM (SELECT * FROM {table} limit {{row_count}}) AS sub_table;"
    },
    {
        "name": "swarm_db_agent", "table": "swarm_test",
        "func_type": "series", "column": "data",
        "query": "select unnest(db_agent_single('{func_type}', sub_table.{column})) AS score FROM (SELECT * FROM {table} limit {{row_count}}) AS sub_table;"
    },
    {
        "name": "year_predict_db_agent", "table": "year_predict_test",
        "func_type": "series", "column": "data",
        "query": "select unnest(db_agent_single('{func_type}', sub_table.{column})) AS score FROM (SELECT * FROM {table} limit {{row_count}}) AS sub_table;"
    },
    # --- NLP tests using db_agent_single (GPU) ---
    {
        "name": "imdb_db_agent", "table": "imdb_vector_test",
        "func_type": "nlp", "column": "comment_vec",
        "query": "select unnest(db_agent_single('{func_type}', sub_table.{column})) AS score FROM (SELECT * FROM {table} limit {{row_count}}) AS sub_table;"
    },
    {
        "name": "financial_phrasebank_db_agent", "table": "financial_phrasebank_vector_test",
        "func_type": "nlp", "column": "comment_vec",
        "query": "select unnest(db_agent_single('{func_type}', sub_table.{column})) AS score FROM (SELECT * FROM {table} limit {{row_count}}) AS sub_table;"
    },
    {
        "name": "nlp_db_agent", "table": "nlp_vector_test",
        "func_type": "nlp", "column": "comment_vec",
        "query": "select unnest(db_agent_single('{func_type}', sub_table.{column})) AS score FROM (SELECT * FROM {table} limit {{row_count}}) AS sub_table;"
    },
    # --- Image tests using db_agent_single (GPU) ---
    {
        "name": "cifar_db_agent", "table": "cifar_image_vector_table",
        "func_type": "image_classification", "column": "image_vector",
        "query": "select unnest(db_agent_single('{func_type}', sub_table.{column})) AS score FROM (SELECT * FROM {table} limit {{row_count}}) AS sub_table;"
    },
    {
        "name": "stanford_dogs_db_agent", "table": "stanford_dogs_image_vector_table",
        "func_type": "image_classification", "column": "image_vector",
        "query": "select unnest(db_agent_single('{func_type}', sub_table.{column})) AS score FROM (SELECT * FROM {table} limit {{row_count}}) AS sub_table;"
    },
    {
        "name": "imagenet_db_agent", "table": "imagenet_image_vector_table",
        "func_type": "image_classification", "column": "image_vector",
        "query": "select unnest(db_agent_single('{func_type}', sub_table.{column})) AS score FROM (SELECT * FROM {table} limit {{row_count}}) AS sub_table;"
    },
]


# =============================================================================
# Main test runner
# =============================================================================
def run_single_slice_test(use_new_queries=False):
    """
    Run single-thread latency tests.

    Args:
        use_new_queries: False = predict_batch_float8, True = db_agent_single
    """
    ROW_COUNTS = [1000, 2000, 5000, 10000]
    QUERY_TIMES_LIST = [1]
    # QUERY_TIMES_LIST = [1, 5, 10, 20, 50, 100]

    queries = NEW_SQL_QUERIES if use_new_queries else SQL_QUERIES
    queries_name = "NEW_SQL_QUERIES (db_agent_single)" if use_new_queries else "SQL_QUERIES (predict_batch_float8)"
    print(f"\nUsing {queries_name}")

    all_results = {"timestamp": datetime.now().isoformat(), "queries_used": queries_name, "tests": []}

    for sql_info in queries:
        print(f"\n{'='*80}")
        print(f"Testing: {sql_info['name']}  |  Table: {sql_info['table']}  |  Column: {sql_info['column']}")
        if 'model' in sql_info:
            print(f"Model: {sql_info['model']}")
        if 'func_type' in sql_info:
            print(f"FuncType: {sql_info['func_type']}")
        print(f"{'='*80}")

        test_results = {
            "name": sql_info["name"], "table": sql_info["table"],
            "results": []
        }
        if 'model' in sql_info:
            test_results["model"] = sql_info["model"]
        if 'func_type' in sql_info:
            test_results["func_type"] = sql_info["func_type"]
        test_results["column"] = sql_info["column"]

        # Determine symbol: CPU for series, GPU for nlp/image
        if use_new_queries:
            symbol = 'cpu' if sql_info['func_type'] == 'series' else 'gpu'
        else:
            cpu_tests = ['slice_predict', 'swarm_predict', 'year_predict_test']
            symbol = 'cpu' if sql_info['name'] in cpu_tests else 'gpu'

        for row_count in ROW_COUNTS:
            for query_times in QUERY_TIMES_LIST:
                print(f"\n  rows={row_count}, query_times={query_times}, symbol={symbol}")

                # Format query
                if use_new_queries:
                    formatted_query = sql_info['query'].format(
                        func_type=sql_info['func_type'],
                        column=sql_info['column'],
                        table=sql_info['table']
                    ).format(row_count=row_count)
                else:
                    formatted_query = sql_info['query'].format(
                        model=sql_info['model'], symbol=symbol,
                        column=sql_info['column'], table=sql_info['table'],
                        row_count=row_count
                    )

                result = run_single_task_worker(0, row_count, query_times, symbol, formatted_query)
                result['row_count'] = row_count
                result['query_times'] = query_times
                result['formatted_query'] = formatted_query
                test_results["results"].append(result)
                print(f"  Result: {result['status']} latency={result.get('latency', 'N/A')}s")

        all_results["tests"].append(test_results)

    # Print summary
    print(f"\n{'='*80}")
    print(f"SUMMARY - {queries_name}")
    print(f"{'='*80}")
    for test in all_results["tests"]:
        print(f"\n  {test['name']} ({test['table']}):")
        for r in test["results"]:
            if r["status"] == "success":
                print(f"    rows={r['row_count']}, qt={r['query_times']}, latency={r['latency']:.4f}s")
            else:
                print(f"    rows={r['row_count']}, qt={r['query_times']}, FAILED: {r.get('error', '')}")

    # Save results
    suffix = "new" if use_new_queries else "original"
    filename = f"single_test_results_{suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    json_to_csv(filename, filename.replace('.json', '.csv'))
    print(f"\nResults saved to: {filename}")


def run_both_tests():
    """Run both query sets sequentially."""
    run_single_slice_test(use_new_queries=False)
    print("\n" + "="*80)
    run_single_slice_test(use_new_queries=True)


if __name__ == "__main__":
    run_both_tests()
