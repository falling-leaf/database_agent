#!/usr/bin/env python3
"""
Setup musique dataset with 20 samples for benchmark testing.
Extends the existing setup_musique_data.py to support 20 samples.
"""

import json
import psycopg2

DB_CONFIG = {
    "dbname": "postgres",
    "host": "localhost",
    "port": "5432",
    "user": "why",
    "password": "123456"
}

MUSIQUE_DATA_PATH = "/home/why/dbagent/pgdl/test/morphingdb_test/data/reasoning/musique/musique_full_v1.0_test.jsonl"
SPLICE_MODEL_PATH = "/home/why/dbagent/pgdl/test/morphingdb_test/models/spiece.model.old"
NUM_SAMPLES = 20


def load_samples(count):
    samples = []
    with open(MUSIQUE_DATA_PATH) as f:
        for i, line in enumerate(f):
            if i >= count:
                break
            data = json.loads(line)
            question = data["question"]
            supporting = [p for p in data["paragraphs"] if p.get("is_supporting", False)]
            if not supporting:
                supporting = data["paragraphs"][:2]
            context_parts = [p["paragraph_text"] for p in supporting]
            context = " || ".join(context_parts)
            answer = ", ".join([p["title"] for p in supporting])
            if len(context) > 200:
                context = context[:200]
            samples.append({
                "context": context,
                "question": question,
                "answer": answer,
            })
    return samples


def main():
    print(f"Setting up musique datasets ({NUM_SAMPLES} samples)...")
    samples = load_samples(NUM_SAMPLES)
    print(f"  Loaded {len(samples)} samples")

    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    # musique_test
    cur.execute("DROP TABLE IF EXISTS musique_test CASCADE;")
    cur.execute("""
        CREATE TABLE musique_test (
            id serial primary key,
            context text,
            question text,
            answer text
        );
    """)
    for s in samples:
        cur.execute(
            "INSERT INTO musique_test (context, question, answer) VALUES (%s, %s, %s)",
            (s["context"], s["question"], s["answer"])
        )
    print(f"  musique_test: {len(samples)} rows")

    # musique_vector_test
    cur.execute("DROP TABLE IF EXISTS musique_vector_test CASCADE;")
    cur.execute("""
        CREATE TABLE musique_vector_test (
            id serial primary key,
            reasoning_vec mvec
        );
    """)
    for s in samples:
        combined = f"{s['context']} || {s['question']}"
        combined_escaped = combined.replace("'", "''")
        cur.execute(f"INSERT INTO musique_vector_test (reasoning_vec) VALUES (text_to_vector('{SPLICE_MODEL_PATH}', '{combined_escaped}'))")
    print(f"  musique_vector_test: {len(samples)} rows")

    # musique_step2
    cur.execute("DROP TABLE IF EXISTS musique_step2 CASCADE;")
    cur.execute("""
        CREATE TABLE musique_step2 (
            id serial primary key,
            query text,
            context text
        );
    """)
    for s in samples:
        q = s["question"][:100]
        c = s["context"][:100]
        cur.execute("INSERT INTO musique_step2 (query, context) VALUES (%s, %s)", (q, c))
        cur.execute("INSERT INTO musique_step2 (query, context) VALUES (%s, %s)", (q, c))
    print(f"  musique_step2: {len(samples)*2} rows")

    # musique_vector_step2
    cur.execute("DROP TABLE IF EXISTS musique_vector_step2 CASCADE;")
    cur.execute("""
        CREATE TABLE musique_vector_step2 (
            id serial primary key,
            text_vec mvec
        );
    """)
    for s in samples:
        combined = f"{s['context']} || {s['question']}"
        combined_escaped = combined.replace("'", "''")
        cur.execute(f"INSERT INTO musique_vector_step2 (text_vec) VALUES (text_to_vector('{SPLICE_MODEL_PATH}', '{combined_escaped}'))")
        cur.execute(f"INSERT INTO musique_vector_step2 (text_vec) VALUES (text_to_vector('{SPLICE_MODEL_PATH}', '{combined_escaped}'))")
    print(f"  musique_vector_step2: {len(samples)*2} rows")

    conn.commit()
    conn.close()
    print("Done!")


if __name__ == "__main__":
    main()
