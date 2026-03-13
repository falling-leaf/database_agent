import psycopg2
import json
import torch
from transformers import AutoTokenizer
from config import db_config

VECTOR_TABLE = "reasoning_vector_test"

DATASET_PATH = "/home/why/reasoning_model/HotpotQA/raw/distractor_sample.json"
MODEL_PATH = "/home/why/reasoning_model/ms-marco-MiniLM-L6-v2"

# CrossEncoder TorchScript
PT_MODEL_PATH = "/home/why/pgdl/model/models/cross_encoder.pt"

MAX_INSERT = 10000


def load_dataset():
    """
    读取 HotpotQA 数据
    生成 (question, context_sentence)
    """
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


def load_cross_encoder(pt_path):
    """
    加载 TorchScript CrossEncoder
    """
    model = torch.jit.load(pt_path)
    model.eval()
    return model


def run_cross_encoder(model, tensor):
    """
    调用 TorchScript 模型进行推理
    """
    with torch.no_grad():
        logits = model(tensor)

    return logits


def tensor_to_mvec_str(tensor):
    """
    将 tensor 转为数据库 mvec 字符串
    格式:
    [data]{shape}
    """

    data = tensor.flatten().tolist()
    shape = tensor.shape

    data_str = ",".join(str(float(x)) for x in data)
    shape_str = ",".join(str(x) for x in shape)

    return f"[{data_str}]{{{shape_str}}}"


def main():

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    print("Loading TorchScript CrossEncoder...")
    cross_encoder = load_cross_encoder(PT_MODEL_PATH)

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
        print("Context:", c)

        # Tokenize
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

        # CrossEncoder 输入 shape = [1,2,128]
        stacked = torch.stack([
            input_ids,
            attention_mask
        ]).float().unsqueeze(0)

        # ---------- 推理验证 ----------
        logits = run_cross_encoder(cross_encoder, stacked)

        print("\nSample", inserted)
        print("Tensor shape:", stacked.shape)
        print("CrossEncoder score:", logits)

        # ---------- 插入数据库 ----------
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