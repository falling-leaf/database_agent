#!/usr/bin/env python3
"""
Benchmark: db_agent_single('musique') vs pure Python pipeline.

Both run the exact same reasoning pipeline on the same musique data:
1. Load 10 mvec vectors from musique_vector_test
2. cross_encoder ranks all 10, picks top-2
3. For each top-2: load idx*2 and idx*2+1 from musique_vector_step2 / musique_step2
4. deberta_reader performs QA on the 4 loaded vectors
5. reader_decoder.py decodes token IDs to text
6. llm_generate.py generates final answer

Measures wall-clock time for each implementation.
"""

import time
import json
import subprocess
import numpy as np
import torch
import psycopg2

# =============================================================================
# Configuration
# =============================================================================
DB_CONFIG = {
    "dbname": "postgres",
    "host": "localhost",
    "port": "5432",
    "user": "why",
    "password": "123456"
}

MODEL_DIR = "/home/why/dbagent/pgdl/test/morphingdb_test/models"
TOOLS_DIR = "/home/why/dbagent/pgdl/test/tools"
ROW_COUNT = 10  # Same as "LIMIT 10" in the SQL query
TOP_K = 2


def get_conn():
    return psycopg2.connect(**DB_CONFIG)


# =============================================================================
# Step 0: Load all required data from DB (shared by both implementations)
# =============================================================================
def load_data_from_db():
    """Load all vectors and text data from the database."""
    conn = get_conn()
    cur = conn.cursor()

    # Load input vectors: musique_vector_test
    cur.execute("SELECT get_mvec_data(reasoning_vec) FROM musique_vector_test LIMIT %s", (ROW_COUNT,))
    input_vecs = [np.array(row[0], dtype=np.float32).reshape(4, 128) for row in cur.fetchall()]

    # Load step2 vectors and text: musique_vector_step2 / musique_step2
    # We load all 20 rows (10 samples * 2 duplicates each)
    cur.execute("SELECT get_mvec_data(text_vec) FROM musique_vector_step2")
    step2_vecs = [np.array(row[0], dtype=np.float32).reshape(4, 128) for row in cur.fetchall()]

    cur.execute("SELECT query, context FROM musique_step2")
    step2_texts = [(row[0], row[1]) for row in cur.fetchall()]

    conn.close()
    return input_vecs, step2_vecs, step2_texts


# =============================================================================
# Implementation 1: Pure Python pipeline
# =============================================================================
def run_python_pipeline(input_vecs, step2_vecs, step2_texts):
    """Run the full reasoning pipeline in pure Python."""
    model_cross = torch.jit.load(f"{MODEL_DIR}/cross_encoder.pt")
    model_reader = torch.jit.load(f"{MODEL_DIR}/deberta_reader.pt")

    timings = {}

    # --- Stage 1: cross_encoder ranking ---
    t0 = time.perf_counter()
    # Prepare batch: [N, 2, 128] - take input_ids (idx 0) and mask (idx 1)
    batch = np.stack([v[:2, :] for v in input_vecs], axis=0).astype(np.float32)
    batch_tensor = torch.from_numpy(batch)
    with torch.no_grad():
        scores = model_cross(batch_tensor)
    scores = scores.squeeze(-1).tolist()
    # Sort descending, pick top-K
    indexed_scores = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    top_k = indexed_scores[:TOP_K]
    timings["cross_encoder"] = time.perf_counter() - t0
    print(f"  [Python] cross_encoder: {timings['cross_encoder']:.4f}s")
    print(f"  [Python] Top-{TOP_K}: {[(idx, f'{s:.4f}') for idx, s in top_k]}")

    # --- Stage 2: Load step2 data for top-K ---
    t0 = time.perf_counter()
    reader_inputs = []
    text_data_list = []
    for orig_idx, _ in top_k:
        idx = orig_idx * 2
        for j in [idx, idx + 1]:
            vec = step2_vecs[j]
            reader_inputs.append(vec[:2, :])  # input_ids + mask
            text_data_list.append(step2_texts[j])
    timings["load_step2"] = time.perf_counter() - t0

    # --- Stage 3: deberta_reader inference ---
    t0 = time.perf_counter()
    reader_batch = np.stack(reader_inputs, axis=0).astype(np.float32)
    reader_tensor = torch.from_numpy(reader_batch)
    with torch.no_grad():
        reader_output = model_reader(reader_tensor)
    # Output shape: [batch, 2, 128] where dim 1 = [start_logits, end_logits]
    start_logits = reader_output[:, 0, :]  # [batch, 128]
    end_logits = reader_output[:, 1, :]    # [batch, 128]
    timings["deberta_reader"] = time.perf_counter() - t0
    print(f"  [Python] deberta_reader: {timings['deberta_reader']:.4f}s")

    # --- Stage 4: reader_decoder ---
    t0 = time.perf_counter()
    # Extract token IDs from input vectors at [start, end] positions
    res_str_parts = []
    for i, (orig_idx, _) in enumerate(top_k):
        idx = orig_idx * 2
        for j in [0, 1]:
            vec_idx = idx + j
            input_vec = step2_vecs[vec_idx]
            input_ids = input_vec[0, :].astype(int).tolist()
            s = int(start_logits[i * 2 + j].argmax().item())
            e = int(end_logits[i * 2 + j].argmax().item())
            tokens = input_ids[s:e + 1]
            res_str_parts.append(",".join(str(t) for t in tokens))
    res_str = ";".join(res_str_parts)

    # Call reader_decoder.py
    cmd = [
        "uv", "run", f"{TOOLS_DIR}/reader_decoder.py", "--ids", res_str
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=TOOLS_DIR)
    decoder_result = result.stdout
    timings["reader_decoder"] = time.perf_counter() - t0
    print(f"  [Python] reader_decoder: {timings['reader_decoder']:.4f}s")
    print(f"  [Python] decoder output: {decoder_result.strip()[:120]}")

    # --- Stage 5: LLM generation ---
    t0 = time.perf_counter()
    try:
        decoder_json = json.loads(decoder_result)
    except json.JSONDecodeError:
        decoder_json = {}

    valid_results = {}
    for i in range(1, 5):
        key = f"paragraph_{i}"
        val = decoder_json.get(key, "")
        if val:
            valid_results[i] = val

    # Build context
    question = text_data_list[0][0] if text_data_list else "Question"
    context_parts = []
    for idx, val in valid_results.items():
        text_idx = idx - 1
        if 0 <= text_idx < len(text_data_list):
            context_parts.append(f" <{text_data_list[text_idx][0]}: {val}> ")
    context = f"Question: {question}\nContext: \n" + "\n".join(context_parts)

    # Call llm_generate.py
    cmd = [
        "uv", "run", f"{TOOLS_DIR}/llm_generate.py", "--input", context
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=TOOLS_DIR)
    llm_result = result.stdout.strip()
    timings["llm_generate"] = time.perf_counter() - t0
    print(f"  [Python] llm_generate: {timings['llm_generate']:.4f}s")
    print(f"  [Python] LLM answer: {llm_result}")

    total = sum(timings.values())
    timings["total"] = total
    return timings, llm_result


# =============================================================================
# Implementation 2: db_agent_single via PostgreSQL
# =============================================================================
def run_dbagent_pipeline():
    """Run the full reasoning pipeline via db_agent_single SQL function."""
    conn = get_conn()
    cur = conn.cursor()

    sql = f"""
    SELECT unnest(db_agent_single('musique', sub_table.reasoning_vec)) AS score
    FROM (SELECT * FROM musique_vector_test LIMIT {ROW_COUNT}) AS sub_table;
    """

    t0 = time.perf_counter()
    cur.execute(sql)
    results = cur.fetchall()
    elapsed = time.perf_counter() - t0

    conn.close()
    print(f"  [DB] db_agent_single total: {elapsed:.4f}s")
    print(f"  [DB] Results: {[r[0] for r in results]}")
    return elapsed, [r[0] for r in results]


# =============================================================================
# Main
# =============================================================================
def main():
    print(f"=== Benchmark: Musique Reasoning Pipeline ({ROW_COUNT} rows, top-{TOP_K}) ===\n")

    # Load data once
    print("[0] Loading data from database...")
    input_vecs, step2_vecs, step2_texts = load_data_from_db()
    print(f"  Loaded {len(input_vecs)} input vectors, {len(step2_vecs)} step2 vectors, {len(step2_texts)} text rows\n")

    # Run Python pipeline
    print("[1] Running pure Python pipeline...")
    py_timings, py_answer = run_python_pipeline(input_vecs, step2_vecs, step2_texts)
    print(f"  [Python] Final answer: {py_answer}\n")

    # Run db_agent_single pipeline
    print("[2] Running db_agent_single('musique') pipeline...")
    db_elapsed, db_results = run_dbagent_pipeline()
    print()

    # Summary
    print("=" * 60)
    print("TIME COMPARISON")
    print("=" * 60)
    print(f"  Pure Python total:  {py_timings['total']:.4f}s")
    print(f"  db_agent_single:    {db_elapsed:.4f}s")
    print(f"  Speedup:            {py_timings['total'] / db_elapsed:.2f}x")
    print()
    print("Python breakdown:")
    for stage, t in py_timings.items():
        if stage != "total":
            print(f"    {stage:20s}: {t:.4f}s ({t / py_timings['total'] * 100:5.1f}%)")


if __name__ == "__main__":
    main()
