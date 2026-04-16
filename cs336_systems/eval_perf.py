import argparse
import timeit
from contextlib import nullcontext

import torch
from einops import rearrange

from cs336_basics.cs336_basics import optimizer
from cs336_basics.cs336_basics.model import BasicsTransformerLM
from cs336_basics.cs336_basics.nn_utils import cross_entropy

parser = argparse.ArgumentParser(description="CS336 Systems - Performance Benchmark")
parser.add_argument("-b", "--backward", action="store_true", help="Include backward pass in timing")
parser.add_argument("-w", "--warmup_steps", type=int, default=5, help="Number of warmup steps")
parser.add_argument("-n", "--num_steps", type=int, default=20, help="Number of measured steps")
parser.add_argument("--context_length", type=int, default=256, help="Sequence length")
parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
parser.add_argument("--bf16", action="store_true", help="Use BF16 mixed precision")
parser.add_argument("--profile_memory", action="store_true", help="Record and dump memory history snapshot")
args = parser.parse_args()

ifBackward = args.backward
warmup_steps = args.warmup_steps
num_steps = args.num_steps

# hyperparameters
vocab_size = 10000
batch_size = args.batch_size
context_length = args.context_length  # varying
d_model = 768
d_ff = 3072
num_layers = 12
num_heads = 12
rope_theta = 10000

device = "cuda" if torch.cuda.is_available() else "cpu"

# init model and optimizer
model = BasicsTransformerLM(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta)
model.to(device)
optimizer = optimizer.AdamW(params=model.parameters())

# set context manager
if args.bf16 and device == "cuda":
    ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    print("Mode: BF16 Mixed Precision")
else:
    ctx = nullcontext()
    print("Mode: Full Precision")

# generate batched data
input_data = torch.randint(low=0, high=vocab_size, size=(batch_size, context_length), device=device)
target = torch.randint(low=0, high=vocab_size, size=(batch_size, context_length), device=device)
flatten_targets = rearrange(target, "b c -> (b c)")

# ==========================================
# Warm up
# ==========================================
for _ in range(warmup_steps):
    with ctx:
        logits = model(input_data)
        if ifBackward:
            flatten_logits = rearrange(logits, "b c v -> (b c) v")
            loss = cross_entropy(inputs=flatten_logits.float(), targets=flatten_targets)

    if ifBackward:
        loss.backward()
        optimizer.step()      
        optimizer.zero_grad()
        
    if device == "cuda":
        torch.cuda.synchronize()

# ==========================================
# Benchmarking
# ==========================================
print(f"Benchmarking for {num_steps} steps...")

fwd_time_total = 0.0
bwd_time_total = 0.0
opt_time_total = 0.0  # optimizer step time

#  tuen on Memory Profile
if args.profile_memory and device == "cuda":
    print("📸 Started recording PyTorch memory history...")
    torch.cuda.memory._record_memory_history(max_entries=1000000)

start_time = timeit.default_timer()

for _ in range(num_steps):
    # 1. Forward 
    fwd_start = timeit.default_timer()
    with ctx:
        logits = model(input_data)
        if ifBackward:
            flatten_logits = rearrange(logits, "b c v -> (b c) v")
            loss = cross_entropy(inputs=flatten_logits.float(), targets=flatten_targets)
            
    if device == "cuda":
        torch.cuda.synchronize() 
    fwd_time_total += (timeit.default_timer() - fwd_start)

    if ifBackward:
        # 2. Backward
        bwd_start = timeit.default_timer()
        loss.backward()
        if device == "cuda":
            torch.cuda.synchronize() 
        bwd_time_total += (timeit.default_timer() - bwd_start)

        # 3. Optimizer Step
        opt_start = timeit.default_timer()
        optimizer.step()
        optimizer.zero_grad()
        if device == "cuda":
            torch.cuda.synchronize()
        opt_time_total += (timeit.default_timer() - opt_start)

end_time = timeit.default_timer()

# save and close Memory Profile
if args.profile_memory and device == "cuda":
    torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
    torch.cuda.memory._record_memory_history(enabled=None)
    print("💾 Memory snapshot saved to 'memory_snapshot.pickle'")

# ==========================================
# Calculate Metrics
# ==========================================
total_time = end_time - start_time
avg_time_per_iter = total_time / num_steps
tokens_per_iter = batch_size * context_length
throughput = tokens_per_iter / avg_time_per_iter

avg_fwd_time = fwd_time_total / num_steps
avg_bwd_time = (bwd_time_total / num_steps) if ifBackward else 0.0
avg_opt_time = (opt_time_total / num_steps) if ifBackward else 0.0

print("\n" + "=" * 45)
print("📊 Benchmark Results")
print("=" * 45)
print(f"Mode         : {'Forward + Backward' if args.backward else 'Forward Only'}")
print(f"Precision    : {'BF16 (Mixed)' if args.bf16 else 'FP32 (Full)'}")
print(f"Total Steps  : {num_steps}")
print("-" * 45)
print(f"Avg Fwd Time : {avg_fwd_time:.5f} seconds/step")
if ifBackward:
    print(f"Avg Bwd Time : {avg_bwd_time:.5f} seconds/step")
    print(f"Avg Opt Time : {avg_opt_time:.5f} seconds/step") 
print(f"Total Time/St: {avg_time_per_iter:.5f} seconds/step")
print("-" * 45)
print(f"Throughput   : {throughput:.2f} tokens/second")
print("=" * 45)