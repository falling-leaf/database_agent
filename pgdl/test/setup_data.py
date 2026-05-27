#!/usr/bin/env python3
"""
Data Initialization Script for PGDL/MorphingDB Tests.

Consolidates all data setup scripts into a single entry point.
Sets up reasoning, musique, and SST2 datasets for testing.

Usage:
    uv run python setup_data.py [--reasoning] [--musique] [--musique-20] [--sst2] [--all]

Flags:
    --reasoning    Setup reasoning dataset (10 samples)
    --musique      Setup musique dataset (10 samples)
    --musique-20   Setup musique dataset (20 samples)
    --sst2         Setup SST2 text dataset
    --all          Setup all datasets (default if no flags given)
"""

import os
import sys
import json
import time
import psycopg2
import subprocess

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

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PSQL = "/home/why/dbagent/pg_base/bin/psql"
MODEL_DIR = os.path.join(TEST_DIR, "morphingdb_test/models")
SPLICE_MODEL_PATH = os.path.join(MODEL_DIR, "spiece.model.old")
MUSIQUE_DATA_PATH = os.path.join(
    TEST_DIR, "morphingdb_test/data/reasoning/musique/musique_full_v1.0_test.jsonl"
)


def get_conn():
    return psycopg2.connect(**DB_CONFIG)


# =============================================================================
# Reasoning Data Setup (from setup_reasoning_data.py)
# =============================================================================
REASONING_SAMPLES = [
    {"context": "The quick brown fox jumps over the lazy dog.", "question": "What does the fox jump over?", "answer": "the lazy dog"},
    {"context": "Python is a high-level programming language known for its readability.", "question": "What is Python known for?", "answer": "readability"},
    {"context": "The Earth revolves around the Sun in approximately 365 days.", "question": "How many days does it take Earth to revolve around the Sun?", "answer": "365"},
    {"context": "Water boils at 100 degrees Celsius at sea level.", "question": "At what temperature does water boil?", "answer": "100 degrees Celsius"},
    {"context": "The Great Wall of China is over 13,000 miles long.", "question": "How long is the Great Wall of China?", "answer": "over 13,000 miles"},
    {"context": "Photosynthesis converts sunlight into chemical energy in plants.", "question": "What does photosynthesis convert sunlight into?", "answer": "chemical energy"},
    {"context": "The human body has 206 bones in adulthood.", "question": "How many bones does an adult human have?", "answer": "206"},
    {"context": "DNA stands for deoxyribonucleic acid.", "question": "What does DNA stand for?", "answer": "deoxyribonucleic acid"},
    {"context": "The speed of light is approximately 299,792,458 meters per second.", "question": "What is the speed of light?", "answer": "299,792,458 meters per second"},
    {"context": "Mount Everest is the highest mountain above sea level at 8,849 meters.", "question": "How high is Mount Everest?", "answer": "8,849 meters"},
]


def setup_reasoning_data():
    """Create reasoning_test, reasoning_vector_test, reasoning_step2, reasoning_vector_step2."""
    print("\n" + "=" * 60)
    print("  Setting up REASONING dataset (10 samples)")
    print("=" * 60)

    conn = get_conn()
    cur = conn.cursor()

    # reasoning_test
    cur.execute("DROP TABLE IF EXISTS reasoning_test CASCADE;")
    cur.execute("""
        CREATE TABLE reasoning_test (
            id serial primary key,
            context text,
            question text,
            answer text
        );
    """)
    for s in REASONING_SAMPLES:
        cur.execute(
            "INSERT INTO reasoning_test (context, question, answer) VALUES (%s, %s, %s)",
            (s["context"], s["question"], s["answer"])
        )
    print(f"  reasoning_test: {len(REASONING_SAMPLES)} rows")

    # reasoning_vector_test
    cur.execute("DROP TABLE IF EXISTS reasoning_vector_test CASCADE;")
    cur.execute("""
        CREATE TABLE reasoning_vector_test (
            id serial primary key,
            reasoning_vec mvec
        );
    """)
    for s in REASONING_SAMPLES:
        combined = f"{s['context']} || {s['question']}"
        combined_escaped = combined.replace("'", "''")
        cur.execute(
            f"INSERT INTO reasoning_vector_test (reasoning_vec) "
            f"VALUES (text_to_vector('{SPLICE_MODEL_PATH}', '{combined_escaped}'))"
        )
    print(f"  reasoning_vector_test: {len(REASONING_SAMPLES)} rows")

    # reasoning_step2 (2 rows per sample)
    cur.execute("DROP TABLE IF EXISTS reasoning_step2 CASCADE;")
    cur.execute("""
        CREATE TABLE reasoning_step2 (
            id serial primary key,
            query text,
            context text
        );
    """)
    for s in REASONING_SAMPLES:
        q = s["question"][:100]
        c = s["context"][:100]
        cur.execute("INSERT INTO reasoning_step2 (query, context) VALUES (%s, %s)", (q, c))
        cur.execute("INSERT INTO reasoning_step2 (query, context) VALUES (%s, %s)", (q, c))
    print(f"  reasoning_step2: {len(REASONING_SAMPLES) * 2} rows")

    # reasoning_vector_step2
    cur.execute("DROP TABLE IF EXISTS reasoning_vector_step2 CASCADE;")
    cur.execute("""
        CREATE TABLE reasoning_vector_step2 (
            id serial primary key,
            text_vec mvec
        );
    """)
    for s in REASONING_SAMPLES:
        combined = f"{s['context']} || {s['question']}"
        combined_escaped = combined.replace("'", "''")
        cur.execute(
            f"INSERT INTO reasoning_vector_step2 (text_vec) "
            f"VALUES (text_to_vector('{SPLICE_MODEL_PATH}', '{combined_escaped}'))"
        )
        cur.execute(
            f"INSERT INTO reasoning_vector_step2 (text_vec) "
            f"VALUES (text_to_vector('{SPLICE_MODEL_PATH}', '{combined_escaped}'))"
        )
    print(f"  reasoning_vector_step2: {len(REASONING_SAMPLES) * 2} rows")

    conn.commit()
    conn.close()
    print("  Reasoning dataset setup complete!")


# =============================================================================
# Musique Data Setup (from setup_musique_data.py / setup_musique_20.py)
# =============================================================================
def load_musique_samples(count=10):
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
            if len(context) > 200:
                context = context[:200]
            samples.append({
                "context": context,
                "question": question,
                "answer": answer,
            })
    return samples


def setup_musique_data(num_samples=10):
    """Create musique_test, musique_vector_test, musique_step2, musique_vector_step2."""
    print(f"\n{'=' * 60}")
    print(f"  Setting up MUSIQUE dataset ({num_samples} samples)")
    print("=" * 60)

    if not os.path.exists(MUSIQUE_DATA_PATH):
        print(f"  WARNING: Musique data file not found at {MUSIQUE_DATA_PATH}")
        print(f"  Skipping musique setup.")
        return False

    samples = load_musique_samples(num_samples)
    print(f"  Loaded {len(samples)} samples from musique dataset")

    conn = get_conn()
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
        cur.execute(
            f"INSERT INTO musique_vector_test (reasoning_vec) "
            f"VALUES (text_to_vector('{SPLICE_MODEL_PATH}', '{combined_escaped}'))"
        )
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
    print(f"  musique_step2: {len(samples) * 2} rows")

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
        cur.execute(
            f"INSERT INTO musique_vector_step2 (text_vec) "
            f"VALUES (text_to_vector('{SPLICE_MODEL_PATH}', '{combined_escaped}'))"
        )
        cur.execute(
            f"INSERT INTO musique_vector_step2 (text_vec) "
            f"VALUES (text_to_vector('{SPLICE_MODEL_PATH}', '{combined_escaped}'))"
        )
    print(f"  musique_vector_step2: {len(samples) * 2} rows")

    conn.commit()
    conn.close()
    print(f"  Musique dataset setup complete ({num_samples} samples)!")
    return True


# =============================================================================
# SST2 Data Setup (from run_tests.py)
# =============================================================================
def setup_sst2_data():
    """Setup SST2 text dataset for testing."""
    print("\n" + "=" * 60)
    print("  Setting up SST2 text dataset")
    print("=" * 60)

    morphingdb_test_dir = os.path.join(TEST_DIR, "morphingdb_test")
    sst2_dir = os.path.join(morphingdb_test_dir, "text_test/sst2")
    morphingdb_test_file = os.path.join(sst2_dir, "morphingdb_test.py")

    if os.path.exists(morphingdb_test_file):
        print(f"  Running SST2 setup via: {morphingdb_test_file}")
        try:
            result = subprocess.run(
                ["uv", "run", "python", "-c",
                 f"import sys; sys.path.insert(0, '{TEST_DIR}'); "
                 f"from morphingdb_test.text_test.sst2.morphingdb_test import sst2_all_test; "
                 f"sst2_all_test()"],
                cwd=TEST_DIR,
                capture_output=True,
                text=True,
                timeout=300
            )
            if result.returncode == 0:
                print("  SST2 dataset setup complete!")
            else:
                print(f"  SST2 setup had issues (return code {result.returncode})")
                if result.stderr:
                    print(f"  stderr: {result.stderr[-500:]}")
        except subprocess.TimeoutExpired:
            print("  SST2 setup timed out (300s)")
        except Exception as e:
            print(f"  SST2 setup error: {e}")
    else:
        print(f"  WARNING: SST2 test file not found at {morphingdb_test_file}")
        print(f"  Skipping SST2 setup.")


# =============================================================================
# Main
# =============================================================================
def main():
    parser = __import__('argparse').ArgumentParser(
        description="Setup all test datasets for PGDL/MorphingDB"
    )
    parser.add_argument("--reasoning", action="store_true", help="Setup reasoning dataset")
    parser.add_argument("--musique", action="store_true", help="Setup musique dataset (10 samples)")
    parser.add_argument("--musique-20", action="store_true", help="Setup musique dataset (20 samples)")
    parser.add_argument("--sst2", action="store_true", help="Setup SST2 text dataset")
    parser.add_argument("--all", action="store_true", help="Setup all datasets (default)")
    args = parser.parse_args()

    # If no specific flags, default to --all
    run_all = not any([args.reasoning, args.musique, args.musique_20, args.sst2, args.all])
    if args.all:
        run_all = True

    start = time.time()

    if run_all or args.reasoning:
        try:
            setup_reasoning_data()
        except Exception as e:
            print(f"  ERROR in reasoning setup: {e}")

    if run_all or args.musique:
        try:
            setup_musique_data(num_samples=10)
        except Exception as e:
            print(f"  ERROR in musique setup: {e}")

    if run_all or args.musique_20:
        try:
            setup_musique_data(num_samples=20)
        except Exception as e:
            print(f"  ERROR in musique-20 setup: {e}")

    if run_all or args.sst2:
        try:
            setup_sst2_data()
        except Exception as e:
            print(f"  ERROR in SST2 setup: {e}")

    elapsed = time.time() - start
    print(f"\n{'=' * 60}")
    print(f"  All data initialization complete in {elapsed:.1f}s")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
