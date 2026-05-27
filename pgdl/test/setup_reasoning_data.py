#!/usr/bin/env python3
"""
Setup reasoning datasets for db_agent_single('reasoning', ...) pipeline.

The reasoning pipeline:
1. Input: mvec vectors from reasoning_vector_test (N rows)
2. cross_encoder ranks all N vectors and picks top-2
3. For each top-2, loads pairs from reasoning_step2 and reasoning_vector_step2
4. deberta_reader performs QA on the loaded data

This script creates:
- reasoning_test: 10 context/question/answer rows
- reasoning_vector_test: 10 mvec vectors (context || question)
- reasoning_step2: 20 query/context rows (2 per sample, for pair loading)
- reasoning_vector_step2: 20 mvec vectors (matching step2 text)

The pipeline uses idx = original_index * 2 and idx+1 to load from step2.
For 10 input rows ranked 0-9, top-2 produces indices like 0-9 * 2 = 0-18.
"""

import psycopg2
import subprocess

DB_CONFIG = {
    "dbname": "postgres",
    "host": "localhost",
    "port": "5432",
    "user": "why",
    "password": "123456"
}

PSQL = "/home/why/dbagent/pg_base/bin/psql"
SPLICE_MODEL_PATH = "/home/why/dbagent/pgdl/test/morphingdb_test/models/spiece.model.old"

# 10 sample Q&A pairs
SAMPLE_DATA = [
    {
        "context": "The quick brown fox jumps over the lazy dog.",
        "question": "What does the fox jump over?",
        "answer": "the lazy dog"
    },
    {
        "context": "Python is a high-level programming language known for its readability.",
        "question": "What is Python known for?",
        "answer": "readability"
    },
    {
        "context": "The Earth revolves around the Sun in approximately 365 days.",
        "question": "How many days does it take Earth to revolve around the Sun?",
        "answer": "365"
    },
    {
        "context": "Water boils at 100 degrees Celsius at sea level.",
        "question": "At what temperature does water boil?",
        "answer": "100 degrees Celsius"
    },
    {
        "context": "The Great Wall of China is over 13,000 miles long.",
        "question": "How long is the Great Wall of China?",
        "answer": "13,000 miles"
    },
    {
        "context": "Photosynthesis converts sunlight into chemical energy in plants.",
        "question": "What does photosynthesis convert sunlight into?",
        "answer": "chemical energy"
    },
    {
        "context": "The human body has 206 bones in adulthood.",
        "question": "How many bones does an adult human have?",
        "answer": "206"
    },
    {
        "context": "DNA stands for deoxyribonucleic acid.",
        "question": "What does DNA stand for?",
        "answer": "deoxyribonucleic acid"
    },
    {
        "context": "The speed of light is approximately 299,792,458 meters per second.",
        "question": "What is the speed of light?",
        "answer": "299,792,458 meters per second"
    },
    {
        "context": "Mount Everest is the highest mountain above sea level at 8,849 meters.",
        "question": "How high is Mount Everest?",
        "answer": "8,849 meters"
    },
    {
        "context": "Jupiter is the largest planet in our solar system with a diameter of about 143,000 kilometers.",
        "question": "Which planet is the largest in our solar system?",
        "answer": "Jupiter"
    },
    {
        "context": "The periodic table was created by Dmitri Mendeleev in 1869.",
        "question": "Who created the periodic table?",
        "answer": "Dmitri Mendeleev"
    },
    {
        "context": "The Amazon River is the second longest river in the world after the Nile.",
        "question": "What is the second longest river in the world?",
        "answer": "Amazon River"
    },
    {
        "context": "Albert Einstein developed the theory of relativity.",
        "question": "Who developed the theory of relativity?",
        "answer": "Albert Einstein"
    },
    {
        "context": "The Pacific Ocean is the largest ocean on Earth covering about 165 million square kilometers.",
        "question": "Which ocean is the largest on Earth?",
        "answer": "Pacific Ocean"
    },
    {
        "context": "The human brain weighs approximately 1.4 kilograms.",
        "question": "How much does the human brain weigh?",
        "answer": "1.4 kilograms"
    },
    {
        "context": "The atomic number of carbon is 6.",
        "question": "What is the atomic number of carbon?",
        "answer": "6"
    },
    {
        "context": "The Moon orbits Earth at an average distance of 384,400 kilometers.",
        "question": "How far is the Moon from Earth?",
        "answer": "384,400 kilometers"
    },
    {
        "context": "Leonardo da Vinci painted the Mona Lisa in the early 16th century.",
        "question": "Who painted the Mona Lisa?",
        "answer": "Leonardo da Vinci"
    },
    {
        "context": "Sound travels at approximately 343 meters per second in air at room temperature.",
        "question": "How fast does sound travel in air?",
        "answer": "343 meters per second"
    },
]


def setup_reasoning_test(cur):
    print("[1/4] Setting up reasoning_test...")
    cur.execute("DROP TABLE IF EXISTS reasoning_test CASCADE;")
    cur.execute("""
        CREATE TABLE reasoning_test (
            id serial primary key,
            context text,
            question text,
            answer text
        );
    """)
    for item in SAMPLE_DATA:
        cur.execute(
            "INSERT INTO reasoning_test (context, question, answer) VALUES (%s, %s, %s)",
            (item["context"], item["question"], item["answer"])
        )
    print(f"  Inserted {len(SAMPLE_DATA)} rows")


def setup_reasoning_vector_test(cur):
    print("[2/4] Setting up reasoning_vector_test...")
    cur.execute("DROP TABLE IF EXISTS reasoning_vector_test CASCADE;")
    cur.execute("""
        CREATE TABLE reasoning_vector_test (
            id serial primary key,
            reasoning_vec mvec
        );
    """)
    for item in SAMPLE_DATA:
        combined = f"{item['context']} || {item['question']}"
        combined_escaped = combined.replace("'", "''")
        sql = f"INSERT INTO reasoning_vector_test (reasoning_vec) VALUES (text_to_vector('{SPLICE_MODEL_PATH}', '{combined_escaped}'))"
        cur.execute(sql)
    print(f"  Inserted {len(SAMPLE_DATA)} rows")


def setup_reasoning_step2(cur):
    print("[3/4] Setting up reasoning_step2...")
    cur.execute("DROP TABLE IF EXISTS reasoning_step2 CASCADE;")
    cur.execute("""
        CREATE TABLE reasoning_step2 (
            id serial primary key,
            query text,
            context text
        );
    """)
    # For each sample, create 2 rows (idx and idx+1 for pair loading)
    for item in SAMPLE_DATA:
        cur.execute(
            "INSERT INTO reasoning_step2 (query, context) VALUES (%s, %s)",
            (item["question"], item["context"])
        )
        cur.execute(
            "INSERT INTO reasoning_step2 (query, context) VALUES (%s, %s)",
            (item["question"], item["context"])
        )
    print(f"  Inserted {len(SAMPLE_DATA) * 2} rows")


def setup_reasoning_vector_step2(cur):
    print("[4/4] Setting up reasoning_vector_step2...")
    cur.execute("DROP TABLE IF EXISTS reasoning_vector_step2 CASCADE;")
    cur.execute("""
        CREATE TABLE reasoning_vector_step2 (
            id serial primary key,
            text_vec mvec
        );
    """)
    for item in SAMPLE_DATA:
        combined = f"{item['context']} || {item['question']}"
        combined_escaped = combined.replace("'", "''")
        sql = f"INSERT INTO reasoning_vector_step2 (text_vec) VALUES (text_to_vector('{SPLICE_MODEL_PATH}', '{combined_escaped}'))"
        cur.execute(sql)
        # Insert duplicate for pair loading
        sql = f"INSERT INTO reasoning_vector_step2 (text_vec) VALUES (text_to_vector('{SPLICE_MODEL_PATH}', '{combined_escaped}'))"
        cur.execute(sql)
    print(f"  Inserted {len(SAMPLE_DATA) * 2} rows")


def main():
    print("Setting up reasoning datasets...")
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    setup_reasoning_test(cur)
    conn.commit()

    setup_reasoning_vector_test(cur)
    conn.commit()

    setup_reasoning_step2(cur)
    conn.commit()

    setup_reasoning_vector_step2(cur)
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
    for table in ["reasoning_test", "reasoning_vector_test", "reasoning_step2", "reasoning_vector_step2"]:
        cur.execute(f"SELECT COUNT(*) FROM {table};")
        print(f"  {table}: {cur.fetchone()[0]} rows")

    cur.execute("SELECT model_name FROM model_info WHERE model_name IN ('cross_encoder', 'deberta_reader', 'flan_t5_reader') ORDER BY model_name;")
    print(f"  Models: {[r[0] for r in cur.fetchall()]}")

    conn.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
