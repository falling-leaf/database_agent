import psycopg2
import json
import torch
from transformers import AutoTokenizer
from config import db_config

VECTOR_TABLE = "reasoning_vector_test"

DATASET_PATH = "/home/why/reasoning_model/HotpotQA/raw/distractor_sample.json"
MODEL_PATH = "/home/why/reasoning_model/ms-marco-MiniLM-L6-v2"

MAX_INSERT = 10000


def load_dataset():
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    question = data["question"]
    contexts = data["context"]

    processed = []

    for context in contexts:
        title = context[0]
        sentences = context[1]

        for sentence in sentences:
            processed.append((question, f"{title}: {sentence}"))

    return processed[:MAX_INSERT]


def tensor_to_mvec_str(tensor):

    data = tensor.flatten().tolist()
    shape = tensor.shape

    data_str = ",".join(str(float(x)) for x in data)
    shape_str = ",".join(str(x) for x in shape)

    return f"[{data_str}]{{{shape_str}}}"


def main():

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    data = load_dataset()

    conn = psycopg2.connect(**db_config)
    cur = conn.cursor()

    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {VECTOR_TABLE}(
            id serial,
            text_vec mvec
        );
    """)

    conn.commit()

    cur.execute(f"DELETE FROM {VECTOR_TABLE};")
    conn.commit()

    inserted = 0

    for q, c in data:

        inputs = tokenizer(
            q,
            c,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=128
        )

        input_ids = inputs["input_ids"][0]
        attention_mask = inputs["attention_mask"][0]

        stacked = torch.stack([
            input_ids,
            attention_mask
        ]).float().unsqueeze(0)

        mvec_str = tensor_to_mvec_str(stacked)

        cur.execute(
            f"INSERT INTO {VECTOR_TABLE}(text_vec) VALUES (%s::mvec)",
            (mvec_str,)
        )

        inserted += 1

        if inserted % 100 == 0:
            conn.commit()
            print("Inserted", inserted)

    conn.commit()
    conn.close()

    print("Finished.")


if __name__ == "__main__":
    main()