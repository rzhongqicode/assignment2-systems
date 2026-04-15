import argparse
import timeit

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

# generate batched data
input_data = torch.randint(low=0, high=vocab_size, size=(batch_size, context_length), device=device)
target = torch.randint(low=0, high=vocab_size, size=(batch_size, context_length), device=device)
flatten_targets = rearrange(target, "b c -> (b c)")
# warm up
for _ in range(warmup_steps):
    logits = model(input_data)
    if ifBackward:
        flatten_logits = rearrange(logits, "b c v -> (b c) v")
        loss = cross_entropy(inputs=flatten_logits, targets=flatten_targets)
        loss.backward()
        optimizer.zero_grad()
    if device == "cuda":
        torch.cuda.synchronize()

# calculate forward pass time
print(f"Benchmarking for {num_steps} steps...")

start_time = timeit.default_timer()
for _ in range(num_steps):
    logits = model(input_data)
    if ifBackward:
        flatten_logits = rearrange(logits, "b c v -> (b c) v")
        loss = cross_entropy(inputs=flatten_logits, targets=flatten_targets)
        loss.backward()
        optimizer.zero_grad()
    if device == "cuda":
        torch.cuda.synchronize()
end_time = timeit.default_timer()

total_time = end_time - start_time
avg_time_per_iter = total_time / num_steps
tokens_per_iter = batch_size * context_length
throughput = tokens_per_iter / avg_time_per_iter

print("\n" + "=" * 40)
print("📊 Benchmark Results")
print("=" * 40)
print(f"Mode         : {'Forward + Backward' if args.backward else 'Forward Only'}")
print(f"Total Steps  : {num_steps}")
print(f"Time/Step    : {avg_time_per_iter:.5f} seconds")
print(f"Throughput   : {throughput:.2f} tokens/second")
print("=" * 40)
