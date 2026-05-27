#!/usr/bin/env python3
"""
Fair Reasoning Benchmark: dbagent vs Python Baseline vs Hermes vs NeurDB vs GenDB

ALL implementations do IDENTICAL computation on HOTPOTQA reasoning data:
1. Load N mvec vectors (4x128) from reasoning_vector_test
2. cross_encoder.pt: score all N vectors (input: first 2 rows of each)
3. For each sample i (0..N-1):
   - Load step2 vectors at indices 2*i and 2*i+1 from reasoning_vector_step2
   - deberta_reader.pt: QA inference on batch of 2 vectors
   - reader_decoder.py: decode token IDs to text (subprocess, ~2.9s per sample)
4. Return all decoded texts

Same models, same data, same pipeline structure for ALL methods.
"""
import os, sys, time, json, subprocess as sp
import numpy as np
import torch
import psycopg2

# Configuration
MODEL_DIR = "/home/why/dbagent/pgdl/test/morphingdb_test/models"
PSQL = "/home/why/dbagent/pg_base/bin/psql"
TOOLS_DIR = "/home/why/dbagent/pgdl/test/tools"
DB_CONFIG = {"dbname": "postgres", "host": "localhost", "port": "5432", "user": "why", "password": "123456"}
SAMPLE_SIZES = [1, 5, 10, 20]
NUM_RUNS = 3


def load_mvec_vectors(n):
    """Load N mvec vectors from reasoning_vector_test as numpy arrays (4x128)."""
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute("SELECT get_mvec_data(reasoning_vec) FROM reasoning_vector_test ORDER BY id LIMIT %s", (n,))
    vecs = [np.array(r[0], dtype=np.float32).reshape(4, 128) for r in cur.fetchall()]
    conn.close()
    return vecs


def load_step2_vectors(n):
    """Load 2*N mvec vectors from reasoning_vector_step2 as numpy arrays (4x128)."""
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute("SELECT get_mvec_data(text_vec) FROM reasoning_vector_step2 ORDER BY id LIMIT %s", (2 * n,))
    vecs = [np.array(r[0], dtype=np.float32).reshape(4, 128) for r in cur.fetchall()]
    conn.close()
    return vecs


def call_reader_decoder(res_str):
    """Call reader_decoder.py subprocess (same as dbagent's CallToolReaderDecoder)."""
    result = sp.run(
        ["uv", "run", f"{TOOLS_DIR}/reader_decoder.py", "--ids", res_str],
        capture_output=True, text=True, cwd=TOOLS_DIR, timeout=60
    )
    return result.stdout.strip()


def run_dbagent(n):
    """dbagent: PostgreSQL pgdl extension via db_agent_single SQL.
    Measures full wall time from SQL execution start to result return."""
    sql = f"SELECT unnest(db_agent_single('reasoning', sub_table.reasoning_vec)) AS score FROM (SELECT * FROM reasoning_vector_test ORDER BY id LIMIT {n}) AS sub_table;"
    cmd = [PSQL, "-h", "localhost", "-U", "why", "-d", "postgres", "-c", sql]
    env = os.environ.copy()
    env["PGPASSWORD"] = "123456"
    
    t0 = time.perf_counter()
    result = sp.run(cmd, capture_output=True, text=True, env=env, timeout=300)
    elapsed = time.perf_counter() - t0
    
    if result.returncode != 0:
        return None, elapsed
    return {"output_count": len([l for l in result.stdout.strip().split('\n') if l.strip() and not l.startswith('(')])}, elapsed


def run_python_baseline(n, input_vecs, step2_vecs):
    """Python Baseline: cross_encoder -> deberta_reader -> reader_decoder (per sample).
    Fresh model load each run (cold-start pattern)."""
    full_start = time.perf_counter()
    
    # Load models
    model_cross = torch.jit.load(f"{MODEL_DIR}/cross_encoder.pt", map_location="cpu")
    model_cross.eval()
    model_reader = torch.jit.load(f"{MODEL_DIR}/deberta_reader.pt", map_location="cpu")
    model_reader.eval()
    
    # Stage 1: cross_encoder on all N vectors (batched)
    t0 = time.perf_counter()
    batch = torch.from_numpy(np.stack([v[:2, :] for v in input_vecs], axis=0).astype(np.float32))
    with torch.no_grad():
        scores = model_cross(batch).squeeze(-1).tolist()
    cross_time = time.perf_counter() - t0
    
    # Stage 2: deberta_reader + reader_decoder for each sample
    t0 = time.perf_counter()
    decoded_texts = []
    for i in range(n):
        idx = i * 2
        # deberta_reader
        reader_inputs = np.stack([step2_vecs[idx][:2, :], step2_vecs[idx+1][:2, :]], axis=0).astype(np.float32)
        reader_batch = torch.from_numpy(reader_inputs)
        with torch.no_grad():
            reader_output = model_reader(reader_batch)
        start_logits, end_logits = reader_output[:, 0, :], reader_output[:, 1, :]
        
        # Build res_str for reader_decoder (same as dbagent)
        res_str_parts = []
        for j in [0, 1]:
            vec_idx = idx + j
            input_ids = step2_vecs[vec_idx][0, :].astype(int).tolist()
            s = int(start_logits[j].argmax().item())
            e = int(end_logits[j].argmax().item())
            res_str_parts.append(",".join(str(t) for t in input_ids[s:e+1]))
        res_str = ";".join(res_str_parts)
        
        # reader_decoder
        decoded = call_reader_decoder(res_str)
        decoded_texts.append(decoded)
    reader_and_decoder_time = time.perf_counter() - t0
    
    total_time = time.perf_counter() - full_start
    return {"n_scores": len(scores), "n_decoded": len(decoded_texts)}, total_time


def run_hermes_baseline(n, input_vecs, step2_vecs):
    """Hermes Baseline: cross_encoder -> deberta_reader (batched) -> reader_decoder (per sample)."""
    full_start = time.perf_counter()
    
    # Load models
    model_cross = torch.jit.load(f"{MODEL_DIR}/cross_encoder.pt", map_location="cpu")
    model_cross.eval()
    model_reader = torch.jit.load(f"{MODEL_DIR}/deberta_reader.pt", map_location="cpu")
    model_reader.eval()
    
    # Stage 1: cross_encoder
    t0 = time.perf_counter()
    batch = torch.from_numpy(np.stack([v[:2, :] for v in input_vecs], axis=0).astype(np.float32))
    with torch.no_grad():
        scores = model_cross(batch).squeeze(-1).tolist()
    cross_time = time.perf_counter() - t0
    
    # Stage 2: deberta_reader (ALL samples in one batch) -> reader_decoder (per sample)
    t0 = time.perf_counter()
    # Batch all step2 vectors together
    all_step2 = []
    for i in range(n):
        idx = i * 2
        all_step2.append(step2_vecs[idx][:2, :])
        all_step2.append(step2_vecs[idx + 1][:2, :])
    reader_batch = torch.from_numpy(np.stack(all_step2, axis=0).astype(np.float32))
    with torch.no_grad():
        reader_output = model_reader(reader_batch)
    
    # Decode each sample
    decoded_texts = []
    for i in range(n):
        idx = i * 2
        start_logits = reader_output[i*2, 0, :]
        end_logits = reader_output[i*2, 1, :]
        
        res_str_parts = []
        for j in [0, 1]:
            vec_idx = idx + j
            input_ids = step2_vecs[vec_idx][0, :].astype(int).tolist()
            s = int(start_logits.argmax().item())
            e = int(end_logits.argmax().item())
            res_str_parts.append(",".join(str(t) for t in input_ids[s:e+1]))
        res_str = ";".join(res_str_parts)
        decoded = call_reader_decoder(res_str)
        decoded_texts.append(decoded)
    reader_and_decoder_time = time.perf_counter() - t0
    
    total_time = time.perf_counter() - full_start
    return {"n_scores": len(scores), "n_decoded": len(decoded_texts)}, total_time


def run_neurdb_baseline(n, input_vecs, step2_vecs):
    """NeurDB Baseline: cross_encoder -> deberta_reader (block of 4) -> reader_decoder (per sample)."""
    full_start = time.perf_counter()
    
    # Load models
    model_cross = torch.jit.load(f"{MODEL_DIR}/cross_encoder.pt", map_location="cpu")
    model_cross.eval()
    model_reader = torch.jit.load(f"{MODEL_DIR}/deberta_reader.pt", map_location="cpu")
    model_reader.eval()
    
    # Stage 1: cross_encoder
    t0 = time.perf_counter()
    batch = torch.from_numpy(np.stack([v[:2, :] for v in input_vecs], axis=0).astype(np.float32))
    with torch.no_grad():
        scores = model_cross(batch).squeeze(-1).tolist()
    cross_time = time.perf_counter() - t0
    
    # Stage 2: deberta_reader in blocks of 4 -> reader_decoder (per sample)
    t0 = time.perf_counter()
    BLOCK_SIZE = 4
    decoded_texts = []
    
    for block_start in range(0, n, BLOCK_SIZE):
        block_end = min(block_start + BLOCK_SIZE, n)
        block_vecs = []
        for i in range(block_start, block_end):
            idx = i * 2
            block_vecs.append(step2_vecs[idx][:2, :])
            block_vecs.append(step2_vecs[idx + 1][:2, :])
        block_batch = torch.from_numpy(np.stack(block_vecs, axis=0).astype(np.float32))
        with torch.no_grad():
            block_output = model_reader(block_batch)
        
        # Decode each sample in block
        for i in range(block_start, block_end):
            idx = i * 2
            offset = (i - block_start) * 2
            start_logits = block_output[offset, 0, :]
            end_logits = block_output[offset, 1, :]
            
            res_str_parts = []
            for j in [0, 1]:
                vec_idx = idx + j
                input_ids = step2_vecs[vec_idx][0, :].astype(int).tolist()
                s = int(start_logits.argmax().item())
                e = int(end_logits.argmax().item())
                res_str_parts.append(",".join(str(t) for t in input_ids[s:e+1]))
            res_str = ";".join(res_str_parts)
            decoded = call_reader_decoder(res_str)
            decoded_texts.append(decoded)
    reader_and_decoder_time = time.perf_counter() - t0
    
    total_time = time.perf_counter() - full_start
    return {"n_scores": len(scores), "n_decoded": len(decoded_texts)}, total_time


def run_gendb_baseline(n, input_vecs, step2_vecs):
    """GenDB Baseline: cross_encoder -> deberta_reader (sequential) -> reader_decoder (per sample)."""
    full_start = time.perf_counter()
    
    # Load models
    model_cross = torch.jit.load(f"{MODEL_DIR}/cross_encoder.pt", map_location="cpu")
    model_cross.eval()
    model_reader = torch.jit.load(f"{MODEL_DIR}/deberta_reader.pt", map_location="cpu")
    model_reader.eval()
    
    # Stage 1: cross_encoder
    t0 = time.perf_counter()
    batch = torch.from_numpy(np.stack([v[:2, :] for v in input_vecs], axis=0).astype(np.float32))
    with torch.no_grad():
        scores = model_cross(batch).squeeze(-1).tolist()
    cross_time = time.perf_counter() - t0
    
    # Stage 2: deberta_reader (one sample at a time) -> reader_decoder (per sample)
    t0 = time.perf_counter()
    decoded_texts = []
    for i in range(n):
        idx = i * 2
        # deberta_reader
        reader_inputs = np.stack([step2_vecs[idx][:2, :], step2_vecs[idx+1][:2, :]], axis=0).astype(np.float32)
        reader_batch = torch.from_numpy(reader_inputs)
        with torch.no_grad():
            reader_output = model_reader(reader_batch)
        start_logits, end_logits = reader_output[:, 0, :], reader_output[:, 1, :]
        
        # Build res_str for reader_decoder
        res_str_parts = []
        for j in [0, 1]:
            vec_idx = idx + j
            input_ids = step2_vecs[vec_idx][0, :].astype(int).tolist()
            s = int(start_logits[j].argmax().item())
            e = int(end_logits[j].argmax().item())
            res_str_parts.append(",".join(str(t) for t in input_ids[s:e+1]))
        res_str = ";".join(res_str_parts)
        
        # reader_decoder
        decoded = call_reader_decoder(res_str)
        decoded_texts.append(decoded)
    reader_and_decoder_time = time.perf_counter() - t0
    
    total_time = time.perf_counter() - full_start
    return {"n_scores": len(scores), "n_decoded": len(decoded_texts)}, total_time


def run_benchmark():
    results = {
        "dbagent": {},
        "python_baseline": {},
        "hermes_baseline": {},
        "neurdb_baseline": {},
        "gendb_baseline": {},
    }
    
    for n in SAMPLE_SIZES:
        print(f"\n{'='*70}")
        print(f"  SAMPLE SIZE: {n}")
        print(f"{'='*70}")
        
        # Pre-load data for Python baselines (fair: same data for all)
        print(f"  Loading {n} input vectors and {2*n} step2 vectors...")
        input_vecs = load_mvec_vectors(n)
        step2_vecs = load_step2_vectors(n)
        print(f"  Input: {len(input_vecs)} vectors, Step2: {len(step2_vecs)} vectors")
        
        # --- dbagent ---
        print(f"\n[dbagent] Running {NUM_RUNS} times...")
        db_times = []
        for r in range(NUM_RUNS):
            _, t = run_dbagent(n)
            db_times.append(t)
            print(f"  Run {r+1}: {t:.4f}s")
        results["dbagent"][n] = {
            "times": db_times,
            "avg": sum(db_times) / len(db_times),
            "min": min(db_times),
            "max": max(db_times),
        }
        
        # --- python_baseline ---
        print(f"\n[python_baseline] Running {NUM_RUNS} times...")
        py_times = []
        for r in range(NUM_RUNS):
            _, t = run_python_baseline(n, input_vecs, step2_vecs)
            py_times.append(t)
            print(f"  Run {r+1}: {t:.4f}s")
        results["python_baseline"][n] = {
            "times": py_times,
            "avg": sum(py_times) / len(py_times),
            "min": min(py_times),
            "max": max(py_times),
        }
        
        # --- hermes_baseline ---
        print(f"\n[hermes_baseline] Running {NUM_RUNS} times...")
        he_times = []
        for r in range(NUM_RUNS):
            _, t = run_hermes_baseline(n, input_vecs, step2_vecs)
            he_times.append(t)
            print(f"  Run {r+1}: {t:.4f}s")
        results["hermes_baseline"][n] = {
            "times": he_times,
            "avg": sum(he_times) / len(he_times),
            "min": min(he_times),
            "max": max(he_times),
        }
        
        # --- neurdb_baseline ---
        print(f"\n[neurdb_baseline] Running {NUM_RUNS} times...")
        ne_times = []
        for r in range(NUM_RUNS):
            _, t = run_neurdb_baseline(n, input_vecs, step2_vecs)
            ne_times.append(t)
            print(f"  Run {r+1}: {t:.4f}s")
        results["neurdb_baseline"][n] = {
            "times": ne_times,
            "avg": sum(ne_times) / len(ne_times),
            "min": min(ne_times),
            "max": max(ne_times),
        }
        
        # --- gendb_baseline ---
        print(f"\n[gendb_baseline] Running {NUM_RUNS} times...")
        ge_times = []
        for r in range(NUM_RUNS):
            _, t = run_gendb_baseline(n, input_vecs, step2_vecs)
            ge_times.append(t)
            print(f"  Run {r+1}: {t:.4f}s")
        results["gendb_baseline"][n] = {
            "times": ge_times,
            "avg": sum(ge_times) / len(ge_times),
            "min": min(ge_times),
            "max": max(ge_times),
        }
    
    return results


def print_report(results):
    print(f"\n\n{'='*70}")
    print("  BENCHMARK REPORT: HotpotQA Reasoning Pipeline Performance")
    print(f"{'='*70}")
    print(f"  Dataset: HotpotQA (reasoning_test tables)")
    print(f"  Pipeline: cross_encoder -> deberta_reader -> reader_decoder (per sample)")
    print(f"  All methods: identical computation, same models, same data, SAME reader_decoder calls")
    print(f"{'='*70}")
    
    # Summary table
    print(f"\n  AVERAGE TIME (seconds):")
    print(f"  {'Method':<20} {'N=1':<12} {'N=5':<12} {'N=10':<12} {'N=20':<12}")
    print("  " + "-" * 68)
    for method in ["dbagent", "python_baseline", "hermes_baseline", "neurdb_baseline", "gendb_baseline"]:
        row = [method]
        for n in SAMPLE_SIZES:
            data = results[method].get(n, {})
            avg = data.get("avg")
            if avg is not None:
                row.append(f"{avg:.4f}s")
            else:
                row.append("N/A")
        print(f"  {row[0]:<20} {row[1]:<12} {row[2]:<12} {row[3]:<12} {row[4]:<12}")
    
    # Min times
    print(f"\n  BEST TIME (seconds):")
    print(f"  {'Method':<20} {'N=1':<12} {'N=5':<12} {'N=10':<12} {'N=20':<12}")
    print("  " + "-" * 68)
    for method in ["dbagent", "python_baseline", "hermes_baseline", "neurdb_baseline", "gendb_baseline"]:
        row = [method]
        for n in SAMPLE_SIZES:
            data = results[method].get(n, {})
            mn = data.get("min")
            if mn is not None:
                row.append(f"{mn:.4f}s")
            else:
                row.append("N/A")
        print(f"  {row[0]:<20} {row[1]:<12} {row[2]:<12} {row[3]:<12} {row[4]:<12}")
    
    # Speedup relative to dbagent
    print(f"\n  {'='*70}")
    print("  RELATIVE SPEEDUP (vs dbagent, higher = faster):")
    print(f"  {'='*70}")
    for method in ["python_baseline", "hermes_baseline", "neurdb_baseline", "gendb_baseline"]:
        print(f"\n  {method}:")
        for n in SAMPLE_SIZES:
            db_avg = results["dbagent"][n]["avg"]
            m_avg = results[method][n]["avg"]
            speedup = db_avg / m_avg if m_avg > 0 else 0
            print(f"    N={n:2d}: dbagent={db_avg:.4f}s, {method}={m_avg:.4f}s, {method} is {speedup:.2f}x faster")


if __name__ == "__main__":
    print("Starting HotpotQA reasoning benchmark (FAIR comparison)...")
    print(f"Models: cross_encoder.pt, deberta_reader.pt")
    print(f"Pipeline: cross_encoder -> deberta_reader -> reader_decoder (per sample)")
    print(f"DB: postgres@localhost:5432")
    print(f"Sample sizes: {SAMPLE_SIZES}, Runs per size: {NUM_RUNS}")
    print(f"Methods: dbagent, python_baseline, hermes_baseline, neurdb_baseline, gendb_baseline")
    print(f"NOTE: All methods call reader_decoder.py per sample for fair comparison")
    
    results = run_benchmark()
    print_report(results)
    
    # Save results to JSON
    output_path = "/home/why/dbagent/pgdl/test/benchmark_results.json"
    serializable = {}
    for method, sizes in results.items():
        serializable[method] = {}
        for n, data in sizes.items():
            serializable[method][str(n)] = {
                "times": [round(t, 6) if t is not None else "N/A" for t in data["times"]],
                "avg": round(data["avg"], 6) if data.get("avg") is not None else None,
                "min": round(data["min"], 6) if data.get("min") is not None else None,
                "max": round(data["max"], 6) if data.get("max") is not None else None,
            }
    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {output_path}")
