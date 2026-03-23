import torch
from sentence_transformers import CrossEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_path = "/home/why/reasoning_model/ms-marco-MiniLM-L6-v2"

query = "How many people live in Berlin?"
passage = "There are 100 people living in Berlin."

# ==========================
# tokenizer
# ==========================

tokenizer = AutoTokenizer.from_pretrained(model_path)

inputs = tokenizer(
    query,
    passage,
    padding="max_length",
    truncation=True,
    max_length=128,
    return_tensors="pt"
)

input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

print("input_ids:", input_ids.shape)
print("attention_mask:", attention_mask.shape)

# ==========================
# 1 CrossEncoder
# ==========================

cross = CrossEncoder(model_path)

score = cross.predict([(query, passage)])

print("\nCrossEncoder score:", score)

# ==========================
# 2 Huggingface 原始模型
# ==========================

hf_model = AutoModelForSequenceClassification.from_pretrained(model_path)
hf_model.eval()

with torch.no_grad():
    out = hf_model(
        input_ids=input_ids,
        attention_mask=attention_mask
    )

hf_logits = out.logits

print("\nHF logits:", hf_logits)

# ==========================
# 3 TorchScript
# ==========================

ids = input_ids.squeeze(0)
mask = attention_mask.squeeze(0)

mvec = torch.cat([ids, mask], dim=0).float()
mvec = mvec.unsqueeze(0)

print("\nmvec shape:", mvec.shape)

ts_model = torch.jit.load("/home/why/pgdl/model/models/cross_encoder.pt")
ts_model.eval()

with torch.no_grad():
    ts_logits = ts_model(mvec)

print("\nTorchScript logits:", ts_logits)