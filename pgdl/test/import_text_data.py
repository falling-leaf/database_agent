#!/usr/bin/env python3
"""
Import raw text rows for all text datasets (IMDB, SST2, Financial Phrasebank).
Creates tables if they don't exist. Standalone -- no package imports needed.
"""
import os
import pandas as pd
import psycopg2

DB_CONFIG = {
    "dbname": "postgres",
    "host": "localhost",
    "port": "5432",
    "user": "why",
    "password": "123456"
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "morphingdb_test", "data")
MAX_ROWS = 10000

conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

# --- IMDB ---
print("  [4a] Importing IMDB text...")
imdb_parquet = os.path.join(DATA_DIR, "text", "imdb", "data", "test-00000-of-00001.parquet")
if os.path.exists(imdb_parquet):
    cur.execute("CREATE TABLE IF NOT EXISTS imdb_test (comment text);")
    cur.execute("DELETE FROM imdb_test;")
    conn.commit()
    df = pd.read_parquet(imdb_parquet)
    count = 0
    for idx, row in df.iterrows():
        if count >= MAX_ROWS:
            break
        text = str(row['text']).replace("'", "''")
        cur.execute("INSERT INTO imdb_test (comment) VALUES ('{}');".format(text))
        count += 1
        if count % 2000 == 0:
            conn.commit()
    conn.commit()
    print("    IMDB text: {} rows".format(count))
else:
    print("    SKIPPED")

# --- SST2 ---
print("  [4b] Importing SST2 text...")
sst2_tsv = os.path.join(DATA_DIR, "text", "sst2", "data", "train.tsv")
if os.path.exists(sst2_tsv):
    cur.execute("CREATE TABLE IF NOT EXISTS nlp_test (comment text);")
    cur.execute("DELETE FROM nlp_test;")
    conn.commit()
    df = pd.read_csv(sst2_tsv, sep='\t')
    count = 0
    for idx, row in df.iterrows():
        if count >= MAX_ROWS:
            break
        sentence = str(row['sentence']).replace("'", "''")
        cur.execute("INSERT INTO nlp_test (comment) VALUES ('{}');".format(sentence))
        count += 1
        if count % 2000 == 0:
            conn.commit()
    conn.commit()
    print("    SST2 text: {} rows".format(count))
else:
    print("    SKIPPED")

# --- Financial Phrasebank ---
print("  [4c] Importing financial_phrasebank text...")
fp_dir = os.path.join(DATA_DIR, "text", "financial_phrasebank")
fp_files = [
    os.path.join(fp_dir, 'data', 'Sentences_50Agree.txt'),
    os.path.join(fp_dir, 'data', 'Sentences_66Agree.txt'),
    os.path.join(fp_dir, 'data', 'Sentences_75Agree.txt'),
    os.path.join(fp_dir, 'data', 'Sentences_AllAgree.txt'),
]
if any(os.path.exists(f) for f in fp_files):
    cur.execute("CREATE TABLE IF NOT EXISTS financial_phrasebank_test (comment text);")
    cur.execute("DELETE FROM financial_phrasebank_test;")
    conn.commit()
    sentences = []
    for txt in fp_files:
        if os.path.exists(txt):
            with open(txt, 'r', encoding='ISO-8859-1') as f:
                for line in f:
                    part = line.split('@')[0].strip()
                    if part:
                        sentences.append(part)
    count = 0
    for sent in sentences:
        if count >= MAX_ROWS:
            break
        safe = sent.replace("'", "''")
        cur.execute("INSERT INTO financial_phrasebank_test (comment) VALUES ('{}');".format(safe))
        count += 1
        if count % 2000 == 0:
            conn.commit()
    conn.commit()
    print("    Financial Phrasebank text: {} rows".format(count))
else:
    print("    SKIPPED")

conn.close()
print("  All text imports done.")
