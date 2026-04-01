import json
import time
import subprocess
import torch
from transformers import AutoTokenizer

DATASET_PATH = "/home/why/reasoning_model/HotpotQA/raw/distractor_sample.json"

CROSS_ENCODER_MODEL_PATH = "/home/why/reasoning_model/ms-marco-MiniLM-L6-v2"
CROSS_ENCODER_PT_PATH = "/home/why/pgdl/model/models/cross_encoder.pt"

READER_MODEL_PATH = "/home/why/reasoning_model/deberta-v3-large-squad2"
READER_PT_PATH = "/home/why/pgdl/model/models/deberta_reader.pt"

LLM_SCRIPT_PATH = "/home/why/pgdl/test/tools/llm_generate.py"


def split_question(question):
    if "same nationality" in question:
        return [
            "Scott Derrickson nationality",
            "Ed Wood nationality"
        ]
    return [question]


def rewrite_question(question):
    if "same nationality" in question:
        return [
            "What nationality is Scott Derrickson?",
            "What nationality is Ed Wood?"
        ]
    return [question]


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

    return question, processed, len(contexts)


def load_cross_encoder(pt_path):
    print(f"[Step1] Loading cross-encoder model from {pt_path}")
    model = torch.jit.load(pt_path)
    model.eval()
    return model


def load_reader(pt_path):
    print(f"[Step2] Loading reader model from {pt_path}")
    model = torch.jit.load(pt_path)
    model.eval()
    return model


def run_cross_encoder(model, tensor):
    with torch.no_grad():
        logits = model(tensor)
    return logits


def run_reader(model, tensor):
    with torch.no_grad():
        start_logits, end_logits = model(tensor)
    return start_logits, end_logits


def decode_answer(tokenizer, input_ids, start_logits, end_logits):
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


def call_llm_generate(context):
    cmd = [
        "uv", "run", LLM_SCRIPT_PATH,
        "--input", context
    ]
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd="/home/why/pgdl/test/tools"
    )
    
    if result.returncode != 0:
        print(f"LLM generation error: {result.stderr}")
        return ""
    
    return result.stdout.strip()


def main():
    total_start = time.time()
    
    print("=" * 60)
    print("Reasoning Pipeline - Step1 + Step2 + LLM Generation")
    print("=" * 60)
    
    print("\n[Init] Loading dataset...")
    question, data, context_count = load_dataset()
    print(f"Original question: {question}")
    print(f"Total samples: {len(data)} (contexts: {context_count})")
    
    print("\n" + "=" * 60)
    print("STEP 1: Cross-Encoder Scoring")
    print("=" * 60)
    
    step1_start = time.time()
    
    cross_encoder_tokenizer = AutoTokenizer.from_pretrained(CROSS_ENCODER_MODEL_PATH)
    cross_encoder = load_cross_encoder(CROSS_ENCODER_PT_PATH)
    
    results = []
    cross_encoder_tokenize_time = 0
    cross_encoder_forward_time = 0
    
    for i, (q, c) in enumerate(data):
        # 计时 tokenize
        tokenize_start = time.time()
        inputs = cross_encoder_tokenizer(
            q, c,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=128
        )
        tokenize_end = time.time()
        cross_encoder_tokenize_time += (tokenize_end - tokenize_start)

        input_ids = inputs["input_ids"][0]
        attention_mask = inputs["attention_mask"][0]

        stacked = torch.stack([
            input_ids,
            attention_mask
        ]).float().unsqueeze(0)

        # 计时 forward
        forward_start = time.time()
        logits = run_cross_encoder(cross_encoder, stacked)
        forward_end = time.time()
        cross_encoder_forward_time += (forward_end - forward_start)
        
        score = logits.item()

        results.append({
            "question": q,
            "context": c,
            "score": score,
            "tensor": stacked
        })
    
    print(f"\n[Step1] Tokenize time: {cross_encoder_tokenize_time:.4f}s")
    print(f"[Step1] Forward time:  {cross_encoder_forward_time:.4f}s")
    
    aggregated = []
    for i in range(0, len(results), 2):
        score_sum = results[i]["score"] + results[i+1]["score"]
        aggregated.append({
            "context": results[i]["context"],
            "score": score_sum
        })
    
    aggregated_sorted = sorted(aggregated, key=lambda x: x["score"], reverse=True)
    
    step1_end = time.time()
    print(f"\n[Step1] Completed in {step1_end - step1_start:.2f}s")
    
    print("\n[Step1] Top 2 contexts by aggregated score:")
    for i, r in enumerate(aggregated_sorted[:2]):
        print(f"\n  Rank {i+1}: Score = {r['score']:.4f}")
        print(f"  Context: {r['context'][:150]}...")
    
    print("\n" + "=" * 60)
    print("STEP 2: Reader (DeBERTa QA)")
    print("=" * 60)
    
    step2_start = time.time()
    
    reader_tokenizer = AutoTokenizer.from_pretrained(
        READER_MODEL_PATH,
        use_fast=False
    )
    reader = load_reader(READER_PT_PATH)
    
    sub_questions = rewrite_question(question)
    
    top_n = 2
    text_data_list = []
    reader_results = []
    
    reader_tokenize_time = 0
    reader_forward_time = 0
    
    for i in range(top_n):
        context_text = aggregated_sorted[i]["context"]
        
        for sub_q in sub_questions:
            # 计时 tokenize
            tokenize_start = time.time()
            inputs = reader_tokenizer(
                sub_q,
                context_text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=256
            )
            tokenize_end = time.time()
            reader_tokenize_time += (tokenize_end - tokenize_start)

            input_ids = inputs["input_ids"][0]
            attention_mask = inputs["attention_mask"][0]

            stacked = torch.stack([
                input_ids,
                attention_mask
            ]).float().unsqueeze(0)

            # 计时 forward
            forward_start = time.time()
            with torch.no_grad():
                start_logits, end_logits = reader(stacked)
            forward_end = time.time()
            reader_forward_time += (forward_end - forward_start)

            answer = decode_answer(
                reader_tokenizer,
                input_ids,
                start_logits,
                end_logits
            )
            
            text_data_list.append({
                "query": sub_q,
                "context": context_text
            })
            
            reader_results.append({
                "sub_question": sub_q,
                "context": context_text,
                "answer": answer
            })
            
            print(f"\n  Sub-question: {sub_q}")
            print(f"  Answer: {answer if answer else '(empty)'}")
    
    step2_end = time.time()
    print(f"\n[Step2] Completed in {step2_end - step2_start:.2f}s")
    print(f"[Step2] Tokenize time: {reader_tokenize_time:.4f}s")
    print(f"[Step2] Forward time:  {reader_forward_time:.4f}s")
    
    print("\n" + "=" * 60)
    print("STEP 3: Build Prompt & LLM Generation")
    print("=" * 60)
    
    step3_start = time.time()
    
    valid_results = {}
    for i, r in enumerate(reader_results):
        if r["answer"]:
            valid_results[i + 1] = r["answer"]
    
    print(f"\n[Step3] Valid answers extracted: {len(valid_results)}")
    for idx, ans in valid_results.items():
        print(f"  paragraph_{idx}: {ans}")
    
    context_parts = []
    for idx, answer in valid_results.items():
        index = idx - 1
        if index >= 0 and index < len(text_data_list):
            text_data = text_data_list[index]
            part = text_data["query"] + ": " + answer
            context_parts.append(part)
    
    context = "Question: Were Scott Derrickson and Ed Wood of the same nationality? \n Context: \n"
    for part in context_parts:
        context += " <" + part + "> \n"
    
    print(f"\n[Step3] Generated context for LLM:")
    print("-" * 40)
    print(context)
    print("-" * 40)
    
    print("\n[Step3] Calling LLM...")
    llm_result = call_llm_generate(context)
    
    step3_end = time.time()
    print(f"\n[Step3] Completed in {step3_end - step3_start:.2f}s")
    
    total_end = time.time()
    
    print("\n" + "=" * 60)
    print("FINAL RESULT")
    print("=" * 60)
    print(f"\nOriginal Question: {question}")
    print(f"\nLLM Answer: {llm_result}")
    print("\n" + "=" * 60)
    print("TIMING SUMMARY")
    print("=" * 60)
    print(f"Step1 (Cross-Encoder): {step1_end - step1_start:.2f}s")
    print(f"Step2 (Reader):        {step2_end - step2_start:.2f}s")
    print(f"Step3 (LLM):           {step3_end - step3_start:.2f}s")
    print(f"Total Time:            {total_end - total_start:.2f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
