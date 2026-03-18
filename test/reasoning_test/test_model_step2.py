import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

MODEL_PATH = "/home/why/reasoning_model/deberta-v3-large-squad2"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForQuestionAnswering.from_pretrained(MODEL_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

question = "What nationality is Ed Wood?"

context = """Ed Wood: Edward Davis Wood Jr. (October 10, 1924 – December 10, 1978)
was an American filmmaker, actor, writer, producer, and director."""

inputs = tokenizer(
    question,
    context,
    max_length=256,
    truncation=True,
    padding="max_length",
    return_tensors="pt"
)

inputs = {k: v.to(device) for k, v in inputs.items()}

print("input_ids shape:", inputs["input_ids"].shape)

with torch.no_grad():

    outputs = model(**inputs)

    start_logits = outputs.start_logits
    end_logits = outputs.end_logits

    start = torch.argmax(start_logits)
    end = torch.argmax(end_logits) + 1

answer = tokenizer.decode(
    inputs["input_ids"][0][start:end],
    skip_special_tokens=True
)

print("\nQuestion:", question)
print("Answer:", answer)