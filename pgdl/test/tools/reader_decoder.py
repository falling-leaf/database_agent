import torch
import argparse
import json
import re
from transformers import AutoTokenizer
import time

MODEL_PATH = "/home/why/reasoning_model/deberta-v3-large-squad2"

def clean_control_chars(text):
    """
    使用正则表达式移除所有不可见/控制字符 (Unicode 类别 C)
    这将清除像 \u001a, \u0013, \x00 等干扰项
    """
    # 匹配所有 Unicode 控制字符
    return re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)

def main():
    parser = argparse.ArgumentParser(description="Batch decode token IDs to clean JSON.")
    parser.add_argument("--ids", type=str, required=True)
    args = parser.parse_args()

    # 1. 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. 解析输入
    try:
        segments = [s.strip() for s in args.ids.split(";") if s.strip()]
        token_id_lists = [[int(i.strip()) for i in seg.split(",") if i.strip()] for seg in segments]
        if not token_id_lists: return
    except ValueError:
        print(json.dumps({"error": "Invalid input format"}, indent=4))
        return

    # 3. 批量推理 (Padding)
    encoded_inputs = tokenizer.pad(
        {"input_ids": token_id_lists},
        padding=True,
        return_tensors="pt"
    )

    # 4. 批量解码
    # skip_special_tokens 处理 [PAD] 等
    raw_decoded_texts = tokenizer.batch_decode(
        encoded_inputs["input_ids"], 
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )

    # 5. 二次清洗：去除不可见控制字符
    output_data = {}
    for i, text in enumerate(raw_decoded_texts):
        # 核心修复步骤：正则清理 + 去首尾空格
        clean_text = clean_control_chars(text).strip()
        output_data[f"paragraph_{i+1}"] = clean_text

    # 6. 输出 JSON
    print(json.dumps(output_data, ensure_ascii=False, indent=4))

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.4f} seconds")