import psycopg2
import json
import torch
from transformers import AutoTokenizer
from config import db_config

VECTOR_TABLE = "reasoning_vector_step2"

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
    从数据库表 reasoning_step1 读取数据
    构造 (id, sub_question, paragraph)
    """
    processed = []
    
    # 连接数据库
    conn = psycopg2.connect(**db_config)
    cur = conn.cursor()
    
    # 从 reasoning_step1 表读取数据
    cur.execute("SELECT id, query, context FROM reasoning_step1 ORDER BY id")
    rows = cur.fetchall()
    
    for row in rows:
        id, query, context = row
        
        sub_questions = rewrite_question(query)
        
        for sub_q in sub_questions:
            processed.append((id, sub_q, context))
            
            if len(processed) >= MAX_INSERT:
                conn.close()
                return processed
    
    conn.close()
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

    # 创建 reasoning_step2 表
    cur.execute("""
        CREATE TABLE IF NOT EXISTS reasoning_step2(
            id integer PRIMARY KEY,
            query text,
            context text
        );
    """)

    # 创建向量表
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {VECTOR_TABLE}(
            id integer PRIMARY KEY,
            text_vec mvec
        );
    """)

    conn.commit()

    print("Clearing old data...")

    cur.execute("DELETE FROM reasoning_step2;")
    cur.execute(f"DELETE FROM {VECTOR_TABLE};")
    conn.commit()

    inserted = 0

    for id, q, c in data:

        print("\nSample", inserted)
        print("ID:", id)
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

        # 插入到 reasoning_step2 表（tokenize 前的文本）
        cur.execute(
            "INSERT INTO reasoning_step2(id, query, context) VALUES (%s, %s, %s)",
            (id, q, c)
        )

        # 插入到向量表
        cur.execute(
            f"INSERT INTO {VECTOR_TABLE}(id, text_vec) VALUES (%s, %s::mvec)",
            (id, mvec_str)
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