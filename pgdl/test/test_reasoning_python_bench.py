#!/usr/bin/env python3
"""
Python benchmark matching dbagent batch sizes exactly.

dbagent batch sizes:
- cross_encoder: batch=n (all n samples in one GPU forward pass)
- deberta_reader: batch=2 (one query-context pair per sample, CPU)

Pipeline:
1. Load models (one-time cost)
2. For n samples: cross_encoder forward on batch of n → get n scores
3. For each sample: deberta_reader forward on batch of 2 (query+context pair)
4. Return 2n results

Usage:
    cd /home/why/dbagent/pgdl/test
    uv run python test_reasoning_python_bench.py
"""

import time
import os
import torch
import sentencepiece as spm
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

# Paths
MODEL_DIR = "/home/why/dbagent/pgdl/test/morphingdb_test/models/"
CROSS_ENCODER_PATH = os.path.join(MODEL_DIR, "cross_encoder.pt")
DEBERTA_HF_PATH = os.path.join(MODEL_DIR, "deberta-v3-large-squad2")
SPIECE_PATH = os.path.join(MODEL_DIR, "spiece.model.old")

N_SAMPLES_LIST = [1, 5, 10, 20]

SAMPLES = [
    "The quick brown fox jumps over the lazy dog. || What does the fox jump over?",
    "Python is a high-level programming language known for its readability. || What is Python known for?",
    "The Earth revolves around the Sun in approximately 365 days. || How many days does it take Earth to revolve around the Sun?",
    "Water boils at 100 degrees Celsius at sea level. || At what temperature does water boil?",
    "The Great Wall of China is over 13,000 miles long. || How long is the Great Wall of China?",
    "Photosynthesis converts sunlight into chemical energy in plants. || What does photosynthesis convert sunlight into?",
    "The human body has 206 bones in adulthood. || How many bones does an adult human have?",
    "DNA stands for deoxyribonucleic acid. || What does DNA stand for?",
    "The speed of light is approximately 299,792,458 meters per second. || What is the speed of light?",
    "Mount Everest is the highest mountain above sea level at 8,849 meters. || How high is Mount Everest?",
    "Jupiter is the largest planet in our solar system with a diameter of about 143,000 kilometers. || Which planet is the largest in our solar system?",
    "The periodic table was created by Dmitri Mendeleev in 1869. || Who created the periodic table?",
    "The Amazon River is the second longest river in the world after the Nile. || What is the second longest river in the world?",
    "Albert Einstein developed the theory of relativity. || Who developed the theory of relativity?",
    "The Pacific Ocean is the largest ocean on Earth covering about 165 million square kilometers. || Which ocean is the largest on Earth?",
    "The human brain weighs approximately 1.4 kilograms. || How much does the human brain weigh?",
    "The atomic number of carbon is 6. || What is the atomic number of carbon?",
    "The Moon orbits Earth at an average distance of 384,400 kilometers. || How far is the Moon from Earth?",
    "Leonardo da Vinci painted the Mona Lisa in the early 16th century. || Who painted the Mona Lisa?",
    "Sound travels at approximately 343 meters per second in air at room temperature. || How fast does sound travel in air?",
]


def load_models():
    """Load models (matches dbagent's lazy loading)."""
    print("Loading models...")
    start = time.time()
    
    cross_encoder = torch.jit.load(CROSS_ENCODER_PATH)
    cross_encoder.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(DEBERTA_HF_PATH)
    deberta = AutoModelForQuestionAnswering.from_pretrained(DEBERTA_HF_PATH)
    deberta.eval()
    
    sp = spm.SentencePieceProcessor()
    sp.load(SPIECE_PATH)
    
    elapsed = time.time() - start
    print(f"  Models loaded in {elapsed:.2f}s")
    return cross_encoder, tokenizer, deberta, sp


def encode_for_cross_encoder(sp, text):
    """Encode text for cross_encoder. Returns tensor [1, 4, 128] as float32.
    
    The model was traced to accept a single tensor [batch, 4, 128] where:
    - Feature 0 (dim=1, idx=0): token_ids (stored as float, cast internally)
    - Feature 1 (dim=1, idx=1): attention_mask (stored as float, cast internally)
    - Features 2,3: token_type_ids, position_ids (may or may not be used)
    """
    tokens = sp.encode(text)
    tokens = [sp.piece_to_id("[CLS]")] + tokens + [sp.piece_to_id("[SEP]")]
    
    max_len = 128
    token_ids = tokens[:max_len] + [sp.piece_to_id("<pad>")] * max(0, max_len - len(tokens))
    
    # All features stored as float (matches mvec format)
    # The model casts to long internally for the embedding layer
    feature_0 = [[float(t) for t in token_ids]]  # token_ids
    feature_1 = [[float(1 if t != sp.piece_to_id("<pad>") else 0) for t in token_ids]]  # mask
    feature_2 = [[0.0] * max_len]  # token_type_ids (all zeros)
    feature_3 = [[float(i) for i in range(max_len)]]  # position_ids
    
    tensor = torch.tensor([feature_0, feature_1, feature_2, feature_3], dtype=torch.float32)  # [4, 1, 128]
    tensor = tensor.permute(1, 0, 2)  # [1, 4, 128] - matches dbagent's batch layout
    
    return tensor


def cross_encoder_forward(cross_encoder, batch_tensor):
    """Run cross_encoder on batch of n inputs. Input shape: [n, 4, 128]."""
    batch_tensor = batch_tensor.cpu()
    cross_encoder = cross_encoder.cpu()
    
    with torch.no_grad():
        output = cross_encoder(batch_tensor)
    
    if isinstance(output, tuple):
        scores = output[0].cpu()
    else:
        scores = output.cpu()
    
    return scores


def deberta_reader_forward(tokenizer, deberta, query, context):
    """Run deberta_reader on one query-context pair (matches dbagent's STEP2 with batch=2)."""
    inputs = tokenizer(
        query,
        context,
        return_tensors="pt",
        max_length=512,
        truncation=True
    )
    
    with torch.no_grad():
        outputs = deberta(**inputs)
    
    start_logits = outputs.start_logits
    end_logits = outputs.end_logits
    
    start_idx = torch.argmax(start_logits, dim=-1).item()
    end_idx = torch.argmax(end_logits, dim=-1).item()
    
    return start_idx, end_idx


def run_benchmark(n_samples, cross_encoder, tokenizer, deberta, sp):
    """Run the full pipeline matching dbagent's batch sizes."""
    test_data = SAMPLES[:n_samples]
    
    # Step 1: Encode all n samples → list of [1, 4, 128] tensors
    encoded_list = [encode_for_cross_encoder(sp, text) for text in test_data]
    # Concatenate along batch dim → [n, 4, 128]
    batch_tensor = torch.cat(encoded_list, dim=0)
    
    # Step 2: Cross-encoder forward on batch of n
    start = time.time()
    scores = cross_encoder_forward(cross_encoder, batch_tensor)
    cross_encoder_time = time.time() - start
    
    print(f"  Cross-encoder (batch={n_samples}): {cross_encoder_time:.2f}s")
    print(f"  Scores: {scores.flatten()[:min(n_samples, 5)].tolist()}{'...' if n_samples > 5 else ''}")
    
    # Step 3: For each sample, run deberta_reader (batch=2: query+context)
    total_deberta_time = 0
    for i, text in enumerate(test_data):
        parts = text.split(" || ")
        query = parts[1] if len(parts) > 1 else parts[0]
        context = parts[0]
        
        start = time.time()
        start_idx, end_idx = deberta_reader_forward(tokenizer, deberta, query, context)
        elapsed = time.time() - start
        total_deberta_time += elapsed
        
        if i == 0:
            print(f"  DeBERTa sample 1: {elapsed:.2f}s (start={start_idx}, end={end_idx})")
        elif i == 1:
            print(f"  DeBERTa sample 2: {elapsed:.2f}s")
    
    avg_deberta = total_deberta_time / n_samples
    print(f"  DeBERTa total: {total_deberta_time:.2f}s (avg {avg_deberta:.2f}s/sample)")
    
    total_time = cross_encoder_time + total_deberta_time
    return total_time, cross_encoder_time, total_deberta_time


def main():
    print("=" * 60)
    print("Python Benchmark (matching dbagent batch sizes)")
    print("=" * 60)
    print()
    
    cross_encoder, tokenizer, deberta, sp = load_models()
    print()
    
    for n in N_SAMPLES_LIST:
        print(f"Running n={n}...")
        total, ce_time, deb_time = run_benchmark(n, cross_encoder, tokenizer, deberta, sp)
        print(f"  Total: {total:.2f}s, Cross-encoder: {ce_time:.2f}s, DeBERTa: {deb_time:.2f}s")
        print()
    
    print("-" * 60)
    print(f"{'Samples':>10} {'Total (s)':>12} {'Avg (s)':>12}")
    print("-" * 60)
    
    # Reload models for clean runs
    cross_encoder, tokenizer, deberta, sp = load_models()
    print()
    
    for n in N_SAMPLES_LIST:
        print(f"Running n={n} (fresh)...")
        total, ce_time, deb_time = run_benchmark(n, cross_encoder, tokenizer, deberta, sp)
        avg = total / n
        print(f"{'':>10} {total:>12.2f} {avg:>12.2f}")
    
    print()
    print("-" * 60)


if __name__ == "__main__":
    main()
