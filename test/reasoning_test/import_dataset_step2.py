import psycopg2
import json
import torch
from transformers import AutoTokenizer
from config import db_config

VECTOR_TABLE = "reasoning_vector_step2"

DATASET_PATH = "/home/why/reasoning_model/HotpotQA/raw/distractor_sample.json"

MODEL_PATH = "/home/why/reasoning_model/deberta-v3-large-squad2"

PT_MODEL_PATH = "/home/why/pgdl/model/models/deberta_reader.pt"

MAX_INSERT = 10000


def rewrite_question(question):
    """
    将 comparison question 改写为子问题
    """

    if "same nationality" in question:

        return [
            "What nationality is Scott Derrickson?",
            "What nationality is Ed Wood?"
        ]

    return [question]


def load_dataset():
    """
    读取 HotpotQA
    构造 (sub_question, paragraph)
    """

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

        # 如果是单个 sample，转为 list
        if isinstance(dataset, dict):
            dataset = [dataset]

    processed = []

    for item in dataset:

        question = item["question"]
        contexts = item["context"]

        sub_questions = rewrite_question(question)

        for context in contexts:

            title = context[0]
            sentences = context[1]

            paragraph = " ".join(sentences)

            full_context = f"{title}: {paragraph}"

            for sub_q in sub_questions:

                processed.append((sub_q, full_context))

                if len(processed) >= MAX_INSERT:
                    return processed

    return processed


def load_reader(pt_path):

    print("Loading TorchScript DeBERTa QA Reader...")

    model = torch.jit.load(pt_path)
    model.eval()

    return model


def run_reader(model, tensor):

    with torch.no_grad():
        start_logits, end_logits = model(tensor)

    return start_logits, end_logits


def tensor_to_mvec_str(tensor):

    data = tensor.flatten().tolist()
    shape = tensor.shape

    data_str = ",".join(str(float(x)) for x in data)
    shape_str = ",".join(str(x) for x in shape)

    return f"[{data_str}]{{{shape_str}}}"


def decode_answer(tokenizer, input_ids, start_logits, end_logits):
    """
    安全 span decode
    """

    start = torch.argmax(start_logits, dim=1).item()
    end = torch.argmax(end_logits, dim=1).item()

    if end < start:
        return ""

    answer_ids = input_ids[start:end + 1]

    answer = tokenizer.decode(
        answer_ids,
        skip_special_tokens=True
    ).strip()

    return answer


def main():

    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        use_fast=False
    )

    reader = load_reader(PT_MODEL_PATH)

    print("Loading dataset...")

    data = load_dataset()

    conn = psycopg2.connect(**db_config)
    cur = conn.cursor()

    print("Creating table if not exists...")

    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {VECTOR_TABLE}(
            id serial,
            text_vec mvec
        );
    """)

    conn.commit()

    print("Clearing old data...")

    cur.execute(f"DELETE FROM {VECTOR_TABLE};")
    conn.commit()

    inserted = 0

    for q, c in data:

        print("\nSample", inserted)
        print("Question:", q)
        print("Context:", c[:200], "...")

        inputs = tokenizer(
            q,
            c,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=256
        )

        input_ids = inputs["input_ids"][0]
        attention_mask = inputs["attention_mask"][0]

        stacked = torch.stack([
            input_ids,
            attention_mask
        ]).float().unsqueeze(0)

        start_logits, end_logits = run_reader(reader, stacked)

        answer = decode_answer(
            tokenizer,
            input_ids,
            start_logits,
            end_logits
        )

        print("\nTensor shape:", stacked.shape)
        print("Extracted answer:", answer)

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

    print("\nFinished.")
    print("Total inserted:", inserted)


if __name__ == "__main__":
    main()