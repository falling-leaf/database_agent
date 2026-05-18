import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import argparse

# =========================
# 模型路径
# =========================
MODEL_PATH = "/home/why/reasoning_model/flan-t5-large"

# =========================
# 命令行参数解析
# =========================
parser = argparse.ArgumentParser(description='Generate text using FLAN-T5 model')
parser.add_argument('--input', type=str, required=True, help='Input text for the model')
args = parser.parse_args()

# =========================
# 加载 tokenizer & model
# =========================
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)

# 强制使用 CPU
device = torch.device("cpu")
model = model.to(device)
model.eval()

# =========================
# 构造输入
# =========================
input_text = args.input

# tokenize
inputs = tokenizer(
    input_text,
    return_tensors="pt",
    truncation=True,
    max_length=512
)

input_ids = inputs["input_ids"].to(device)
attention_mask = inputs["attention_mask"].to(device)

# =========================
# 推理（生成答案）
# =========================
with torch.no_grad():
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=32,
        num_beams=4,          # beam search 提高质量
        early_stopping=True
    )

# decode
answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

# =========================
# 仅输出大模型的输出
# =========================
print(answer)