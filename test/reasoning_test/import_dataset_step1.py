import json
import torch
from transformers import AutoTokenizer

VECTOR_TABLE = "reasoning_vector_step1"

DATASET_PATH = "/home/why/reasoning_model/HotpotQA/raw/distractor_sample.json"
MODEL_PATH = "/home/why/reasoning_model/ms-marco-MiniLM-L6-v2"

PT_MODEL_PATH = "/home/why/pgdl/model/models/cross_encoder.pt"

MAX_INSERT = 10000


def split_question(question):
    """
    简单规则拆分（针对当前任务）
    """
    # 示例专用（HotpotQA）
    return [
        "Scott Derrickson nationality",
        "Ed Wood nationality"
    ]


def load_dataset():
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    question = data["question"]
    contexts = data["context"]

    sub_questions = split_question(question)

    processed = []

    for context in contexts:
        title = context[0]
        sentences = context[1]

        paragraph = " ".join(sentences)
        full_context = f"{title}: {paragraph}"

        for sub_q in sub_questions:
            processed.append((sub_q, full_context))

    return processed[:MAX_INSERT], len(contexts)


def load_cross_encoder(pt_path):
    model = torch.jit.load(pt_path)
    model.eval()
    return model


def run_cross_encoder(model, tensor):
    with torch.no_grad():
        logits = model(tensor)
    return logits


def tensor_to_mvec_str(tensor):
    data = tensor.flatten().tolist()
    shape = tensor.shape

    data_str = ",".join(str(float(x)) for x in data)
    shape_str = ",".join(str(x) for x in shape)

    return f"[{data_str}]{{{shape_str}}}"


def main():

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    cross_encoder = load_cross_encoder(PT_MODEL_PATH)

    data, context_count = load_dataset()

    results = []

    # ---------- 推理 ----------
    for i, (q, c) in enumerate(data):

        inputs = tokenizer(
            q, c,
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

        logits = run_cross_encoder(cross_encoder, stacked)
        score = logits.item()

        results.append({
            "question": q,
            "context": c,
            "score": score,
            "tensor": stacked
        })

    # ---------- 聚合（关键）----------
    # 每两个为一组（对应同一个 context）
    aggregated = []

    for i in range(0, len(results), 2):
        score_sum = results[i]["score"] + results[i+1]["score"]

        aggregated.append({
            "context": results[i]["context"],
            "score": score_sum
        })

    # 排序（仅展示）
    aggregated_sorted = sorted(aggregated, key=lambda x: x["score"], reverse=True)

    print("\n=== 聚合排序结果 ===")
    for i, r in enumerate(aggregated_sorted[:10]):
        print(f"\nRank {i+1}")
        print(f"Score: {r['score']:.4f}")
        print(f"Context: {r['context'][:200]}...")

    # ---------- 写数据库 ----------
    try:
        import psycopg2
        from config import db_config
        
        conn = psycopg2.connect(**db_config)
        cur = conn.cursor()

        # 表结构不变（你要求不改约束）
        cur.execute("""
            CREATE TABLE IF NOT EXISTS reasoning_step1(
                id serial PRIMARY KEY,
                query text,
                context text
            );
        """)

        cur.execute("DELETE FROM reasoning_step1;")

        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {VECTOR_TABLE}(
                id serial,
                text_vec mvec
            );
        """)

        conn.commit()

        cur.execute(f"DELETE FROM {VECTOR_TABLE};")
        conn.commit()

        # 插入（注意：现在是2N条）
        for result in results:
            cur.execute(
                "INSERT INTO reasoning_step1(query, context) VALUES (%s, %s)",
                (result["question"], result["context"])
            )

        conn.commit()

        for result in results:
            mvec_str = tensor_to_mvec_str(result["tensor"])

            cur.execute(
                f"INSERT INTO {VECTOR_TABLE}(text_vec) VALUES (%s::mvec)",
                (mvec_str,)
            )

        conn.commit()
        conn.close()

        print("\nFinished (2x data inserted).")

    except ImportError:
        print("\npsycopg2 not available.")


if __name__ == "__main__":
    main()