import psycopg2
import torch
from config import db_config

VECTOR_TABLE = "reasoning_vector_test"

PT_MODEL_PATH = "/home/why/reasoning_model/cross_encoder.pt"


def load_cross_encoder(pt_path):
    """
    加载 TorchScript CrossEncoder 模型
    """
    model = torch.jit.load(pt_path)
    model.eval()
    return model


def mvec_str_to_tensor(mvec_str):
    """
    将数据库中的 mvec 字符串转换为 torch tensor

    输入示例:
    [1.0,2.0,...]{1,2,128}

    输出:
    tensor shape = [1,2,128]
    """

    data_part, shape_part = mvec_str.split("{")

    data_part = data_part.strip("[]")
    shape_part = shape_part.strip("}")

    data = [float(x) for x in data_part.split(",")]
    shape = [int(x) for x in shape_part.split(",")]

    tensor = torch.tensor(data, dtype=torch.float32).view(*shape)

    return tensor


def run_cross_encoder(model, tensor):
    """
    调用 TorchScript 模型进行推理
    """
    with torch.no_grad():
        logits = model(tensor)

    return logits


def infer_from_db(pt_path, limit=5):
    """
    从数据库读取 embedding 并进行推理
    """

    model = load_cross_encoder(pt_path)

    conn = psycopg2.connect(**db_config)
    cur = conn.cursor()

    cur.execute(f"""
        SELECT text_vec::text
        FROM {VECTOR_TABLE}
        LIMIT %s
    """, (limit,))

    rows = cur.fetchall()

    print("Loaded rows:", len(rows))

    for i, row in enumerate(rows):

        mvec_str = row[0]

        tensor = mvec_str_to_tensor(mvec_str)

        print("\nSample", i)
        print("Tensor shape:", tensor.shape)

        logits = run_cross_encoder(model, tensor)

        print("CrossEncoder score:", logits)

    conn.close()


def infer_from_pt_file(pt_path, tensor_path):
    """
    直接从 .pt 文件加载 embedding 并推理
    """
    model = load_cross_encoder(pt_path)

    tensor = torch.load(tensor_path)

    print("Loaded tensor shape:", tensor.shape)

    with torch.no_grad():
        logits = model(tensor)

    print("CrossEncoder score:", logits)

    return logits


if __name__ == "__main__":

    # 方式1：从数据库读取
    infer_from_db(PT_MODEL_PATH, limit=5)

    # 方式2：直接从 embedding pt 文件读取
    # infer_from_pt_file(
    #     PT_MODEL_PATH,
    #     "/home/why/reasoning_model/test_embedding.pt"
    # )