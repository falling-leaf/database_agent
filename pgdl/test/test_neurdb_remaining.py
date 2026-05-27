#!/usr/bin/env python3
"""
Targeted concurrency test for remaining 2 image datasets.
Uses smaller steps and lower upper limit to avoid DB crashes.
"""

import json
import os
import signal
import time
import multiprocessing
from datetime import datetime

import psycopg2
from morphingdb_test.config import db_config

MAX_CONCURRENCY_LIMIT = 50
COARSE_STEP = 10
QUERY_TIMEOUT = 600
BINARY_SLEEP = 2

DATASETS = [
    {"name": "stanford_dogs_image_predict", "type": "image", "symbol": "gpu",
     "table": "stanford_dogs_image_vector_table", "model": "alexnet_stanford_dogs", "column": "image_vector"},
    {"name": "imagenet_image_predict", "type": "image", "symbol": "gpu",
     "table": "imagenet_image_vector_table", "model": "defect_vec", "column": "image_vector"},
]


def build_sql_query(ds_info, row_count):
    return (
        f"select predict_batch_float8('{ds_info['model']}', '{ds_info['symbol']}', "
        f"{ds_info['column']}) over (rows between current row and 31 following) "
        f"from {ds_info['table']} limit {row_count};"
    )


def run_single_task_worker(task_id, row_count, query_times, symbol, sql_query):
    try:
        os.setpgid(0, 0)
    except OSError:
        pass

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
        print(f"  [OK] Task {task_id}: rows={return_rows}, latency={end_time - start_time:.2f}s")
        return {"task_id": task_id, "status": "success", "latency": end_time - start_time}
    except Exception as e:
        err_msg = str(e)
        print(f"  [FAIL] Task {task_id}: {err_msg}")
        return {"task_id": task_id, "status": "failed", "error": err_msg}
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def test_concurrency_level(concurrency_level, row_count, query_times, symbol, sql_query):
    print(f"  Testing concurrency level: {concurrency_level}")

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

        for i, ar in enumerate(async_results):
            try:
                result = ar.get(timeout=QUERY_TIMEOUT)
                results.append(result)
            except Exception as e:
                results.append({"task_id": i, "status": "failed", "error": str(e)})
    finally:
        pool.terminate()
        pool.join()

    failed = [r for r in results if r["status"] == "failed"]
    all_passed = len(failed) == 0
    successful = len(results) - len(failed)

    # Check for DB crash errors
    db_crash = any("recovery mode" in r.get("error", "") for r in failed if r["status"] == "failed")

    print(f"  Level {concurrency_level}: {successful}/{len(results)} passed {'[PASS]' if all_passed else '[FAIL]'}")
    if db_crash:
        print("  *** DATABASE CRASH DETECTED ***")

    return all_passed, db_crash, results


def binary_search_max_concurrency(low, high, row_count, query_times, symbol, sql_query):
    max_pass = low - 1
    while low <= high:
        mid = (low + high) // 2
        if mid == 0:
            low = 1
            continue

        all_passed, db_crash, _ = test_concurrency_level(mid, row_count, query_times, symbol, sql_query)

        if db_crash:
            print(f"  -> DB crashed at {mid}, searching [{low}, {mid - 1}]")
            high = mid - 1
            continue

        if all_passed:
            max_pass = mid
            print(f"  -> {mid} passed, searching [{mid + 1}, {high}]")
            low = mid + 1
        else:
            print(f"  -> {mid} failed, searching [{low}, {mid - 1}]")
            high = mid - 1

        time.sleep(BINARY_SLEEP)
    return max_pass


def find_max_concurrency(row_count=1000, query_times=1, symbol='gpu', sql_query=None):
    # Phase 1: coarse search with smaller steps
    print(f"  Phase 1: Coarse search (step={COARSE_STEP})")
    last_pass = 0
    current = COARSE_STEP
    db_crashed = False

    while current <= MAX_CONCURRENCY_LIMIT:
        all_passed, db_crash, _ = test_concurrency_level(current, row_count, query_times, symbol, sql_query)
        if db_crash:
            print(f"  *** DB crashed at {current}! Stopping coarse search. ***")
            db_crashed = True
            break
        if all_passed:
            last_pass = current
            if current >= MAX_CONCURRENCY_LIMIT:
                return MAX_CONCURRENCY_LIMIT
            current += COARSE_STEP
        else:
            print(f"  Failed at {current}, last pass was {last_pass}")
            break

    time.sleep(BINARY_SLEEP)

    # Phase 2: binary search
    if db_crashed:
        search_high = current - 1
    elif last_pass == 0:
        search_high = COARSE_STEP - 1
    elif current > MAX_CONCURRENCY_LIMIT or current - last_pass <= 1:
        return last_pass
    else:
        search_high = min(current - 1, MAX_CONCURRENCY_LIMIT)

    if last_pass == 0:
        print(f"  Phase 2: Binary search in [1, {search_high}]")
        return binary_search_max_concurrency(1, search_high, row_count, query_times, symbol, sql_query)
    else:
        print(f"  Phase 2: Binary search in [{last_pass + 1}, {search_high}]")
        result = binary_search_max_concurrency(
            last_pass + 1, search_high, row_count, query_times, symbol, sql_query)
        return result if result > last_pass else last_pass


def run_all_tests():
    print("=" * 80)
    print("NeurDB Baseline Concurrency Test - Remaining Image Datasets")
    print(f"Upper limit: {MAX_CONCURRENCY_LIMIT} threads (smaller steps to avoid DB crash)")
    print("=" * 80)

    all_results = {}
    ROW_COUNT = 1000

    for ds in DATASETS:
        ds_name = ds["name"]
        sql_query = build_sql_query(ds, ROW_COUNT)

        print(f"\n{'=' * 60}")
        print(f"Dataset: {ds_name} [{ds['type'].upper()}]")
        print(f"Table: {ds['table']}, Model: {ds['model']}, Symbol: {ds['symbol']}")
        print(f"{'=' * 60}")

        max_conc = find_max_concurrency(
            row_count=ROW_COUNT, query_times=1, symbol=ds['symbol'], sql_query=sql_query
        )

        all_results[ds_name] = {
            "type": ds["type"],
            "table": ds["table"],
            "model": ds["model"],
            "symbol": ds["symbol"],
            "row_count": ROW_COUNT,
            "max_concurrency": max_conc,
        }

        print(f"\n>>> {ds_name}: MAX_CONCURRENCY = {max_conc}")
        # Cool down between datasets
        print(f"\n  Cooling down 5 seconds before next dataset...")
        time.sleep(5)

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filepath = os.path.join(os.path.dirname(__file__), f"neurdb_concurrency_remaining_{timestamp}.json")
    with open(filepath, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {filepath}")
    return all_results


if __name__ == "__main__":
    run_all_tests()
