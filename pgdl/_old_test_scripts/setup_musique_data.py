#!/usr/bin/env python3
"""
Setup musique dataset for db_agent_single('reasoning', ...) pipeline.

Uses 1 sample from musique_full_v1.0_test.jsonl to create:
- musique_test: 1 context/question/answer row
- musique_vector_test: 1 mvec vector (context || question)
- musique_step2: 2 query/context rows (for pair loading)
- musique_vector_step2: 2 mvec vectors (matching step2 text)

Tables are separate from the existing reasoning_* tables to avoid conflicts.
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

# Use the first sample from the test set
MUSIQUE_DATA_PATH = "/home/why/dbagent/pgdl/test/morphingdb_test/data/reasoning/musique/musique_full_v1.0_test.jsonl"
SPLICE_MODEL_PATH = "/home/why/dbagent/pgdl/test/morphingdb_test/models/spiece.model.old"


def load_samples(count=10):
    """Load samples from musique dataset."""
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
            # Truncate context to 200 chars to fit model max length
            if len(context) > 200:
                context = context[:200]
            samples.append({
                "context": context,
                "question": question,
                "answer": answer,
            })
    return samples


def setup_musique_test(cur, samples):
    print("[1/4] Setting up musique_test...")
    cur.execute("DROP TABLE IF EXISTS musique_test CASCADE;")
    cur.execute("""
        CREATE TABLE musique_test (
            id serial primary key,
            context text,
            question text,
            answer text
        );
    """)
    for sample in samples:
        cur.execute(
            "INSERT INTO musique_test (context, question, answer) VALUES (%s, %s, %s)",
            (sample["context"], sample["question"], sample["answer"])
        )
    print(f"  Inserted {len(samples)} rows")


def setup_musique_vector_test(cur, samples):
    print("[2/4] Setting up musique_vector_test...")
    cur.execute("DROP TABLE IF EXISTS musique_vector_test CASCADE;")
    cur.execute("""
        CREATE TABLE musique_vector_test (
            id serial primary key,
            reasoning_vec mvec
        );
    """)
    for sample in samples:
        combined = f"{sample['context']} || {sample['question']}"
        combined_escaped = combined.replace("'", "''")
        sql = f"INSERT INTO musique_vector_test (reasoning_vec) VALUES (text_to_vector('{SPLICE_MODEL_PATH}', '{combined_escaped}'))"
        cur.execute(sql)
    print(f"  Inserted {len(samples)} rows")


def setup_musique_step2(cur, samples):
    print("[3/4] Setting up musique_step2...")
    cur.execute("DROP TABLE IF EXISTS musique_step2 CASCADE;")
    cur.execute("""
        CREATE TABLE musique_step2 (
            id serial primary key,
            query text,
            context text
        );
    """)
    # For each sample, create 2 rows (idx and idx+1 for pair loading)
    # Use truncated question and context to avoid model length issues
    for sample in samples:
        q = sample["question"][:100]
        c = sample["context"][:100]
        cur.execute(
            "INSERT INTO musique_step2 (query, context) VALUES (%s, %s)",
            (q, c)
        )
        cur.execute(
            "INSERT INTO musique_step2 (query, context) VALUES (%s, %s)",
            (q, c)
        )
    print(f"  Inserted {len(samples) * 2} rows")


def setup_musique_vector_step2(cur, samples):
    print("[4/4] Setting up musique_vector_step2...")
    cur.execute("DROP TABLE IF EXISTS musique_vector_step2 CASCADE;")
    cur.execute("""
        CREATE TABLE musique_vector_step2 (
            id serial primary key,
            text_vec mvec
        );
    """)
    for sample in samples:
        combined = f"{sample['context']} || {sample['question']}"
        combined_escaped = combined.replace("'", "''")
        sql = f"INSERT INTO musique_vector_step2 (text_vec) VALUES (text_to_vector('{SPLICE_MODEL_PATH}', '{combined_escaped}'))"
        cur.execute(sql)
        # Insert duplicate for pair loading
        sql = f"INSERT INTO musique_vector_step2 (text_vec) VALUES (text_to_vector('{SPLICE_MODEL_PATH}', '{combined_escaped}'))"
        cur.execute(sql)
    print(f"  Inserted {len(samples) * 2} rows")


def main():
    print("Setting up musique datasets (10 samples)...")
    samples = load_samples(10)
    print(f"  First question: {samples[0]['question']}")
    print(f"  First context (first 100): {samples[0]['context'][:100]}...")
    print(f"  First answer: {samples[0]['answer']}")
    print()

    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    setup_musique_test(cur, samples)
    conn.commit()

    setup_musique_vector_test(cur, samples)
    conn.commit()

    setup_musique_step2(cur, samples)
    conn.commit()

    setup_musique_vector_step2(cur, samples)
    conn.commit()

    # Register models
    print("\nRegistering models...")
    models = [
        ("cross_encoder", "/home/why/dbagent/pgdl/test/morphingdb_test/models/cross_encoder.pt"),
        ("deberta_reader", "/home/why/dbagent/pgdl/test/morphingdb_test/models/deberta_reader.pt"),
        ("flan_t5_reader", "/home/why/dbagent/pgdl/test/morphingdb_test/models/flan_t5_reader.pt"),
    ]
    for name, path in models:
        cur.execute("SELECT COUNT(*) FROM model_info WHERE model_name = %s;", (name,))
        if cur.fetchone()[0] == 0:
            cur.execute("SELECT create_model(%s, %s, '', '');", (name, path))
            print(f"  Registered {name}")
        else:
            print(f"  {name} already registered")
    conn.commit()

    # Verify
    print("\nVerification:")
    for table in ["musique_test", "musique_vector_test", "musique_step2", "musique_vector_step2"]:
        cur.execute(f"SELECT COUNT(*) FROM {table};")
        print(f"  {table}: {cur.fetchone()[0]} rows")

    conn.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
