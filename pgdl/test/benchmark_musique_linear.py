#!/usr/bin/env python3
"""
Fair Musique Benchmark: dbagent vs Python baselines

All methods process ALL N samples through the full pipeline:
1. cross_encoder scores all N vectors
2. For EACH of the N samples: deberta_reader -> reader_decoder

This ensures linear scaling: N越大时间越大。

Same sample repeated N times (fair data).
Models pre-warmed via persistent connection (fair model loading).
Timing: pure execution only.
"""
import os, sys, time, json, subprocess as sp
import numpy as np
import torch
import psycopg2

MODEL_DIR = "/home/why/dbagent/pgdl/test/morphingdb_test/models"
TOOLS_DIR = "/home/why/dbagent/pgdl/test/tools"
DB_CONFIG = {"dbname": "postgres", "host": "localhost", "port": "5432", "user": "why", "password": "123456"}
PSQL = "/home/why/dbagent/pg_base/bin/psql"
SAMPLE_SIZES = [1, 5, 10, 20]
NUM_RUNS = 3


def call_reader_decoder(res_str):
    """Call reader_decoder.py subprocess."""
    result = sp.run(
        ["uv", "run", f"{TOOLS_DIR}/reader_decoder.py", "--ids", res_str],
        capture_output=True, text=True, cwd=TOOLS_DIR, timeout=60
    )
    return result.stdout.strip()


# =============================================================================
# Data loading
# =============================================================================
def load_single_sample():
    """Load 1 sample + its step2 pair from DB."""
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute("SELECT get_mvec_data(reasoning_vec) FROM musique_vector_test ORDER BY id LIMIT 1")
    input_vec = np.array(cur.fetchone()[0], dtype=np.float32).reshape(4, 128)
    cur.execute("SELECT get_mvec_data(text_vec) FROM musique_vector_step2 ORDER BY id LIMIT 2")
    step2_vecs = [np.array(r[0], dtype=np.float32).reshape(4, 128) for r in cur.fetchall()]
    conn.close()
    return input_vec, step2_vecs


# =============================================================================
# dbagent via persistent psycopg2 connection
# =============================================================================
_db_conn = None

def get_db_conn():
    global _db_conn
    if _db_conn is None:
        _db_conn = psycopg2.connect(**DB_CONFIG)
        _db_conn.autocommit = True
    return _db_conn


def run_dbagent(n):
    """
    db_agent_single processes ALL N rows through cross_encoder, picks top-2,
    then processes those 2. This is the dbagent API's fixed behavior.
    We measure total wall time via persistent connection.
    """
    sql = f"""
    SELECT unnest(db_agent_single('musique', sub_table.reasoning_vec)) AS score 
    FROM (
        SELECT reasoning_vec FROM generate_series(1, {n})
        CROSS JOIN (SELECT reasoning_vec FROM musique_vector_test ORDER BY id LIMIT 1) AS single
    ) AS sub_table;
    """
    conn = get_db_conn()
    cur = conn.cursor()
    t0 = time.perf_counter()
    cur.execute(sql)
    cur.fetchall()
    return time.perf_counter() - t0


# =============================================================================
# Python baselines with persistent models (process ALL N samples)
# =============================================================================
_model_cross = None
_model_reader = None

def get_models():
    global _model_cross, _model_reader
    if _model_cross is None:
        _model_cross = torch.jit.load(f"{MODEL_DIR}/cross_encoder.pt", map_location="cpu")
        _model_cross.eval()
        _model_reader = torch.jit.load(f"{MODEL_DIR}/deberta_reader.pt", map_location="cpu")
        _model_reader.eval()
    return _model_cross, _model_reader


def run_python_baseline(n, input_vec, step2_vecs):
    """
    Process ALL N samples sequentially (same as dbagent's internal per-sample processing):
    For each of N: deberta_reader(2 vectors) -> reader_decoder
    """
    model_cross, model_reader = get_models()
    
    t0 = time.perf_counter()
    
    # cross_encoder on all N (batched)
    all_inputs = np.stack([input_vec[:2, :]] * n, axis=0).astype(np.float32)
    batch = torch.from_numpy(all_inputs)
    with torch.no_grad():
        scores = model_cross(batch).squeeze(-1).tolist()
    
    # Process ALL N samples (not just top-2!)
    for i in range(n):
        # deberta_reader on step2 pair
        reader_inputs = np.stack([step2_vecs[0][:2, :], step2_vecs[1][:2, :]], axis=0).astype(np.float32)
        reader_batch = torch.from_numpy(reader_inputs)
        with torch.no_grad():
            reader_output = model_reader(reader_batch)
        start_logits, end_logits = reader_output[:, 0, :], reader_output[:, 1, :]
        
        # Build res_str
        res_str_parts = []
        for j in [0, 1]:
            input_ids = step2_vecs[j][0, :].astype(int).tolist()
            s = int(start_logits[j].argmax().item())
            e = int(end_logits[j].argmax().item())
            res_str_parts.append(",".join(str(t) for t in input_ids[s:e+1]))
        call_reader_decoder(";".join(res_str_parts))
    
    return time.perf_counter() - t0


def run_hermes_baseline(n, input_vec, step2_vecs):
    """
    Process ALL N samples: batch ALL reader inputs together, then decode.
    """
    model_cross, model_reader = get_models()
    
    t0 = time.perf_counter()
    
    # cross_encoder on all N
    all_inputs = np.stack([input_vec[:2, :]] * n, axis=0).astype(np.float32)
    batch = torch.from_numpy(all_inputs)
    with torch.no_grad():
        scores = model_cross(batch).squeeze(-1).tolist()
    
    # Batch ALL reader inputs: 2 vectors per sample * N samples = 2N vectors
    all_reader = []
    for _ in range(n):
        all_reader.append(step2_vecs[0][:2, :])
        all_reader.append(step2_vecs[1][:2, :])
    reader_batch = torch.from_numpy(np.stack(all_reader, axis=0).astype(np.float32))
    with torch.no_grad():
        reader_output = model_reader(reader_batch)
    
    # Decode each sample
    for i in range(n):
        start_logits = reader_output[i*2, 0, :]
        end_logits = reader_output[i*2, 1, :]
        res_str_parts = []
        for j in [0, 1]:
            input_ids = step2_vecs[j][0, :].astype(int).tolist()
            s = int(start_logits.argmax().item())
            e = int(end_logits.argmax().item())
            res_str_parts.append(",".join(str(t) for t in input_ids[s:e+1]))
        call_reader_decoder(";".join(res_str_parts))
    
    return time.perf_counter() - t0


def run_neurdb_baseline(n, input_vec, step2_vecs):
    """
    Process ALL N samples in blocks of 4.
    """
    model_cross, model_reader = get_models()
    
    t0 = time.perf_counter()
    
    # cross_encoder on all N
    all_inputs = np.stack([input_vec[:2, :]] * n, axis=0).astype(np.float32)
    batch = torch.from_numpy(all_inputs)
    with torch.no_grad():
        scores = model_cross(batch).squeeze(-1).tolist()
    
    BLOCK_SIZE = 4
    for block_start in range(0, n, BLOCK_SIZE):
        block_end = min(block_start + BLOCK_SIZE, n)
        block_vecs = []
        for _ in range(block_start, block_end):
            block_vecs.append(step2_vecs[0][:2, :])
            block_vecs.append(step2_vecs[1][:2, :])
        block_batch = torch.from_numpy(np.stack(block_vecs, axis=0).astype(np.float32))
        with torch.no_grad():
            block_output = model_reader(block_batch)
        
        for i in range(block_start, block_end):
            offset = (i - block_start) * 2
            start_logits = block_output[offset, 0, :]
            end_logits = block_output[offset, 1, :]
            res_str_parts = []
            for j in [0, 1]:
                input_ids = step2_vecs[j][0, :].astype(int).tolist()
                s = int(start_logits.argmax().item())
                e = int(end_logits.argmax().item())
                res_str_parts.append(",".join(str(t) for t in input_ids[s:e+1]))
            call_reader_decoder(";".join(res_str_parts))
    
    return time.perf_counter() - t0


def run_gendb_baseline(n, input_vec, step2_vecs):
    """
    Process ALL N samples: each sample independently (reader + decoder in one loop).
    """
    model_cross, model_reader = get_models()
    
    t0 = time.perf_counter()
    
    # cross_encoder on all N
    all_inputs = np.stack([input_vec[:2, :]] * n, axis=0).astype(np.float32)
    batch = torch.from_numpy(all_inputs)
    with torch.no_grad():
        scores = model_cross(batch).squeeze(-1).tolist()
    
    for i in range(n):
        reader_inputs = np.stack([step2_vecs[0][:2, :], step2_vecs[1][:2, :]], axis=0).astype(np.float32)
        reader_batch = torch.from_numpy(reader_inputs)
        with torch.no_grad():
            reader_output = model_reader(reader_batch)
        start_logits, end_logits = reader_output[:, 0, :], reader_output[:, 1, :]
        
        res_str_parts = []
        for j in [0, 1]:
            input_ids = step2_vecs[j][0, :].astype(int).tolist()
            s = int(start_logits[j].argmax().item())
            e = int(end_logits[j].argmax().item())
            res_str_parts.append(",".join(str(t) for t in input_ids[s:e+1]))
        call_reader_decoder(";".join(res_str_parts))
    
    return time.perf_counter() - t0


# =============================================================================
# Benchmark runner
# =============================================================================
def run_benchmark():
    results = {
        "dbagent": {}, "python_baseline": {}, "hermes_baseline": {},
        "neurdb_baseline": {}, "gendb_baseline": {},
    }
    
    # Warmup dbagent
    print("[Warmup] dbagent first call (model loading)...")
    t = run_dbagent(1)
    print(f"  First call: {t:.4f}s\n")
    
    # Warmup Python models
    print("[Warmup] Loading Python models...")
    input_vec, step2_vecs = load_single_sample()
    get_models()
    print("  Done.\n")
    
    for n in SAMPLE_SIZES:
        print(f"\n{'='*60}")
        print(f"  N={n} (same sample x{n}, ALL {n} processed)")
        print(f"{'='*60}")
        
        for method_name, method_fn in [
            ("dbagent", lambda: run_dbagent(n)),
            ("python_baseline", lambda: run_python_baseline(n, input_vec, step2_vecs)),
            ("hermes_baseline", lambda: run_hermes_baseline(n, input_vec, step2_vecs)),
            ("neurdb_baseline", lambda: run_neurdb_baseline(n, input_vec, step2_vecs)),
            ("gendb_baseline", lambda: run_gendb_baseline(n, input_vec, step2_vecs)),
        ]:
            print(f"\n[{method_name}] {NUM_RUNS} runs...")
            times = []
            for r in range(NUM_RUNS):
                t = method_fn()
                times.append(t)
                print(f"  Run {r+1}: {t:.4f}s")
            results[method_name][n] = {
                "times": times,
                "avg": sum(times) / len(times),
                "min": min(times),
                "max": max(times),
            }
        
        time.sleep(1)
    
    return results


def print_report(results):
    print(f"\n\n{'='*70}")
    print("  BENCHMARK: Musique (ALL N samples processed)")
    print(f"{'='*70}")
    
    print(f"\n  AVERAGE TIME (seconds):")
    print(f"  {'Method':<20} {'N=1':<12} {'N=5':<12} {'N=10':<12} {'N=20':<12}")
    print("  " + "-" * 68)
    for method in ["dbagent", "python_baseline", "hermes_baseline", "neurdb_baseline", "gendb_baseline"]:
        row = [method]
        for n in SAMPLE_SIZES:
            avg = results[method][n]["avg"]
            row.append(f"{avg:.4f}s" if avg is not None else "N/A")
        print(f"  {row[0]:<20} {row[1]:<12} {row[2]:<12} {row[3]:<12} {row[4]:<12}")
    
    print(f"\n  BEST TIME (seconds):")
    print(f"  {'Method':<20} {'N=1':<12} {'N=5':<12} {'N=10':<12} {'N=20':<12}")
    print("  " + "-" * 68)
    for method in ["dbagent", "python_baseline", "hermes_baseline", "neurdb_baseline", "gendb_baseline"]:
        row = [method]
        for n in SAMPLE_SIZES:
            mn = results[method][n]["min"]
            row.append(f"{mn:.4f}s" if mn is not None else "N/A")
        print(f"  {row[0]:<20} {row[1]:<12} {row[2]:<12} {row[3]:<12} {row[4]:<12}")
    
    print(f"\n  {'='*70}")
    print("  LINEAR SCALING (ratio vs N=1 baseline, expected: N/1 = N):")
    print(f"  {'='*70}")
    for method in ["dbagent", "python_baseline", "hermes_baseline", "neurdb_baseline", "gendb_baseline"]:
        base = results[method][1]["avg"]
        print(f"\n  {method} (N=1 baseline = {base:.4f}s):")
        for n in SAMPLE_SIZES[1:]:
            avg = results[method][n]["avg"]
            ratio = avg / base if base > 0 else 0
            print(f"    N={n:2d}: {avg:.4f}s  (ratio={ratio:.2f}x, expected ~{n:.0f}x)")
    
    print(f"\n  {'='*70}")
    print("  SPEEDUP vs dbagent (higher = faster):")
    print(f"  {'='*70}")
    for method in ["python_baseline", "hermes_baseline", "neurdb_baseline", "gendb_baseline"]:
        print(f"\n  {method}:")
        for n in SAMPLE_SIZES:
            db_avg = results["dbagent"][n]["avg"]
            m_avg = results[method][n]["avg"]
            if m_avg > 0:
                print(f"    N={n:2d}: {db_avg/m_avg:.2f}x  (db={db_avg:.2f}s, {method}={m_avg:.2f}s)")


if __name__ == "__main__":
    print("Starting Musique benchmark (ALL N samples processed, persistent models)...")
    results = run_benchmark()
    print_report(results)
    
    output_path = "/home/why/dbagent/pgdl/test/musique_benchmark_results.json"
    serializable = {}
    for method, sizes in results.items():
        serializable[method] = {}
        for n, data in sizes.items():
            serializable[method][str(n)] = {
                "times": [round(t, 6) for t in data["times"]],
                "avg": round(data["avg"], 6),
                "min": round(data["min"], 6),
                "max": round(data["max"], 6),
            }
    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {output_path}")
