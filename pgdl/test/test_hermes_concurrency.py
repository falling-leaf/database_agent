#!/usr/bin/env python3
"""
Hermes Baseline Concurrency Test.

Finds the maximum safe concurrency level for the Hermes baseline pipeline
using binary search with upper limit 400.

Key features:
- Pre-loads models in parent, workers inherit via fork CoW (memory efficient)
- Each worker runs a complete Hermes baseline pipeline
- Binary search + coarse stepping
- Any single thread failure = test fails
- Process groups for orphan cleanup

Usage:
    python test_hermes_concurrency.py
"""

import os
import signal
import sys
import time
import json
import subprocess
import multiprocessing
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch
import psycopg2

# =============================================================================
# Configuration
# =============================================================================
MAX_CONCURRENCY_LIMIT = 400
COARSE_STEP = 50
QUERY_TIMEOUT = 600
BINARY_SLEEP = 1

MODEL_DIR = "/home/why/dbagent/pgdl/test/morphingdb_test/models"
TOOLS_DIR = "/home/why/dbagent/pgdl/test/tools"
DB_CONFIG = {"dbname": "postgres", "host": "localhost", "port": "5432", "user": "why", "password": "123456"}

# =============================================================================
# Global models (pre-loaded in parent, shared via fork CoW)
# =============================================================================
_model_cross = None
_model_reader = None

def load_models():
    """Load models once in parent process."""
    global _model_cross, _model_reader
    if _model_cross is None:
        print("  Loading cross_encoder.pt...")
        _model_cross = torch.jit.load(f"{MODEL_DIR}/cross_encoder.pt", map_location="cpu")
        _model_cross.eval()
        print("  Loading deberta_reader.pt...")
        _model_reader = torch.jit.load(f"{MODEL_DIR}/deberta_reader.pt", map_location="cpu")
        _model_reader.eval()
        print("  Models loaded.")
    return _model_cross, _model_reader


def call_reader_decoder(res_str):
    """Call reader_decoder.py subprocess."""
    result = subprocess.run(
        ["uv", "run", f"{TOOLS_DIR}/reader_decoder.py", "--ids", res_str],
        capture_output=True, text=True, cwd=TOOLS_DIR, timeout=60
    )
    return result.stdout.strip()


# =============================================================================
# Worker: runs one Hermes baseline pipeline
# =============================================================================
def run_hermes_worker(task_id):
    """Execute one complete Hermes baseline run in a forked process."""
    try:
        os.setpgid(0, 0)
    except OSError:
        pass

    global _model_cross, _model_reader
    conn = None
    try:
        # Models inherited from parent via fork CoW
        model_cross = _model_cross
        model_reader = _model_reader

        # Load data from DB
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute("SELECT get_mvec_data(reasoning_vec) FROM musique_vector_test ORDER BY id LIMIT 1")
        input_vec = np.array(cur.fetchone()[0], dtype=np.float32).reshape(4, 128)
        cur.execute("SELECT get_mvec_data(text_vec) FROM musique_vector_step2 ORDER BY id LIMIT 2")
        step2_vecs = [np.array(r[0], dtype=np.float32).reshape(4, 128) for r in cur.fetchall()]
        conn.close()
        conn = None

        # Hermes pipeline: cross_encoder -> batched reader -> decode each
        t0 = time.perf_counter()

        all_inputs = np.stack([input_vec[:2, :]] * 1, axis=0).astype(np.float32)
        batch = torch.from_numpy(all_inputs)
        with torch.no_grad():
            scores = model_cross(batch).squeeze(-1).tolist()

        reader_batch = torch.from_numpy(np.stack([step2_vecs[0][:2, :], step2_vecs[1][:2, :]], axis=0).astype(np.float32))
        with torch.no_grad():
            reader_output = model_reader(reader_batch)

        for i in range(1):
            start_logits = reader_output[i*2, 0, :]
            end_logits = reader_output[i*2, 1, :]
            res_str_parts = []
            for j in [0, 1]:
                input_ids = step2_vecs[j][0, :].astype(int).tolist()
                s = int(start_logits.argmax().item())
                e = int(end_logits.argmax().item())
                res_str_parts.append(",".join(str(t) for t in input_ids[s:e+1]))
            call_reader_decoder(";".join(res_str_parts))

        elapsed = time.perf_counter() - t0
        print(f"  [OK] Task {task_id}: {elapsed:.2f}s")
        return {"task_id": task_id, "status": "success", "latency": elapsed}

    except Exception as e:
        err_msg = str(e)
        if len(err_msg) > 200:
            err_msg = err_msg[:200] + "..."
        print(f"  [FAIL] Task {task_id}: {err_msg}")
        return {"task_id": task_id, "status": "failed", "error": err_msg}
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


# =============================================================================
# Concurrency test
# =============================================================================
def test_concurrency_level(concurrency_level):
    """Test a specific concurrency level."""
    print(f"  Testing concurrency level: {concurrency_level}")

    # Fork-based pool (models shared via CoW)
    ctx = multiprocessing.get_context('fork')
    pool = ctx.Pool(
        processes=concurrency_level,
        initializer=os.setpgid,
        initargs=(0, 0)
    )

    results = []
    try:
        async_results = []
        for i in range(concurrency_level):
            ar = pool.apply_async(run_hermes_worker, args=(i,))
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
    successful = [r for r in results if r["status"] == "success"]
    all_passed = len(failed) == 0

    print(f"  Level {concurrency_level}: {len(successful)}/{len(results)} passed {'[PASS]' if all_passed else '[FAIL]'}")
    if failed:
        for f in failed[:3]:
            print(f"    Failed task {f['task_id']}: {f.get('error', 'unknown')[:100]}")
        if len(failed) > 3:
            print(f"    ... and {len(failed)-3} more")

    return all_passed, results


# =============================================================================
# Binary search
# =============================================================================
def binary_search_max_concurrency(low, high):
    max_pass = low - 1
    while low <= high:
        mid = (low + high) // 2
        if mid == 0:
            low = 1
            continue

        all_passed, _ = test_concurrency_level(mid)
        if all_passed:
            max_pass = mid
            print(f"  -> {mid} passed, searching [{mid + 1}, {high}]")
            low = mid + 1
        else:
            print(f"  -> {mid} failed, searching [{low}, {mid - 1}]")
            high = mid - 1

        time.sleep(BINARY_SLEEP)
    return max_pass


def find_max_concurrency():
    # Phase 1: coarse
    print(f"Phase 1: Coarse search (step={COARSE_STEP})")
    last_pass = 0
    current = COARSE_STEP

    while current <= MAX_CONCURRENCY_LIMIT:
        all_passed, _ = test_concurrency_level(current)
        if all_passed:
            last_pass = current
            if current >= MAX_CONCURRENCY_LIMIT:
                print(f"Reached MAX limit {MAX_CONCURRENCY_LIMIT}!")
                return MAX_CONCURRENCY_LIMIT
            current += COARSE_STEP
        else:
            print(f"Failed at {current}, last pass was {last_pass}")
            break

    time.sleep(BINARY_SLEEP)

    # Phase 2: binary
    if last_pass == 0:
        return binary_search_max_concurrency(1, COARSE_STEP - 1)
    elif current - last_pass <= 1:
        return last_pass
    else:
        search_high = min(current - 1, MAX_CONCURRENCY_LIMIT)
        print(f"Phase 2: Binary search in [{last_pass + 1}, {search_high}]")
        result = binary_search_max_concurrency(last_pass + 1, search_high)
        return result if result > last_pass else last_pass


# =============================================================================
# Main
# =============================================================================
def main():
    print("=" * 80)
    print("Hermes Baseline Concurrency Test")
    print(f"Upper limit: {MAX_CONCURRENCY_LIMIT} threads")
    print(f"Pipeline: cross_encoder -> batched reader -> reader_decoder")
    print(f"Models: pre-loaded in parent, shared via fork CoW")
    print("=" * 80)

    # Pre-load models
    load_models()
    print()

    max_conc = find_max_concurrency()

    print(f"\n{'=' * 80}")
    print(f"RESULT: Hermes Baseline MAX_CONCURRENCY = {max_conc}")
    print(f"{'=' * 80}")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_data = {
        "timestamp": timestamp,
        "test": "hermes_baseline_concurrency",
        "max_limit": MAX_CONCURRENCY_LIMIT,
        "max_concurrency": max_conc,
    }

    filepath = f"/home/why/dbagent/pgdl/test/hermes_concurrency_results_{timestamp}.json"
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)
    print(f"Results saved to: {filepath}")


if __name__ == "__main__":
    main()
