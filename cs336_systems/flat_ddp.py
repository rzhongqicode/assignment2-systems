import os
import time  # Imported for benchmarking

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from einops import rearrange

from cs336_basics.cs336_basics import optimizer

# from cs336_basics.cs336_basics.data import get_batch # Commented out for benchmarking
from cs336_basics.cs336_basics.model import BasicsTransformerLM
from cs336_basics.cs336_basics.nn_utils import cross_entropy


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def benchmark_batched_ddp(rank, world_size, batch_size):
    setup(rank, world_size)
    device = f"cuda:{rank}"

    # ---------------------------------------------------------
    # 1. Model Parameters (XL Model Size)
    # ---------------------------------------------------------
    vocab_size = 10000
    context_length = 256
    d_model = 1600
    num_layers = 28
    num_heads = 25
    d_ff = 6400
    rope_theta = 10000
    weight_decay = 0.05
    max_learning_rate = 1e-5
    min_learning_rate = 1e-6

    # Init model and move to assigned GPU
    model = BasicsTransformerLM(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta)
    model = model.to(device)

    # Broadcast initial parameters from rank 0 to ensure all start identically
    for param in model.parameters():
        dist.broadcast(tensor=param.data, src=0, async_op=False)

    # Init optimizer
    opt = optimizer.AdamW(params=model.parameters(), lr=0.0, weight_decay=weight_decay)

    # ---------------------------------------------------------
    # 2. Benchmarking Configuration
    # ---------------------------------------------------------
    warmup_steps = 5
    benchmark_steps = 20  # 20 steps are sufficient for stable measurements
    total_steps = warmup_steps + benchmark_steps

    total_step_time = 0.0
    total_comm_time = 0.0

    if rank == 0:
        print(f"Starting Batched DDP (Flat All-Reduce) Benchmark | Model: XL | Batch Size: {batch_size} per GPU")
        print(f"Warmup steps: {warmup_steps}, Benchmark steps: {benchmark_steps}...")

    # Start training loop
    for step in range(1, total_steps + 1):
        # NOTE: Using synthetic data to prevent disk I/O from bottlenecking the benchmark
        inputs = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
        targets = torch.randint(0, vocab_size, (batch_size, context_length), device=device)

        # ==========================================================
        # [TIMING START]: Total Step Time
        # ==========================================================
        if step > warmup_steps:
            torch.cuda.synchronize()  # Ensure previous steps are completely finished
            step_start_time = time.time()

        # --- Forward Pass & Loss ---
        opt.zero_grad()
        logits = model(inputs)
        flatten_logits = rearrange(logits, "b c v -> (b c) v")
        flatten_targets = rearrange(targets, "b c -> (b c)")
        loss = cross_entropy(inputs=flatten_logits, targets=flatten_targets)

        # --- Backward Pass ---
        loss.backward()

        # ==========================================================
        # [TIMING START]: Communication Time
        # ==========================================================
        if step > warmup_steps:
            torch.cuda.synchronize()  # Wait for backward computation to finish
            comm_start_time = time.time()

        # --- Sync Gradients (Batched DDP with Flatten/Unflatten) ---

        # Step A: Collect all valid gradients
        grad_list = []
        for param in model.parameters():
            if param.grad is not None:
                grad_list.append(param.grad.data)

        # Step B: Flatten into a single massive 1D tensor
        flattened_tensor = torch._utils._flatten_dense_tensors(grad_list)

        # Step C: Single All-Reduce call (Maximum bandwidth utilization)
        dist.all_reduce(tensor=flattened_tensor, async_op=False)
        flattened_tensor /= world_size

        # Step D: Unflatten back to original shapes (requires grad_list as a shape template)
        unflattened_grad_list = torch._utils._unflatten_dense_tensors(flattened_tensor, grad_list)

        # Step E: Write the averaged gradients back to the parameters
        i = 0
        for param in model.parameters():
            if param.grad is not None:
                param.grad.data = unflattened_grad_list[i]
                i += 1

        # ==========================================================
        # [TIMING END]: Communication Time
        # ==========================================================
        if step > warmup_steps:
            torch.cuda.synchronize()  # Wait for the batched all_reduce to finish
            comm_end_time = time.time()
            total_comm_time += comm_end_time - comm_start_time

        # --- Optimizer Step ---
        lr = optimizer.get_cosine_lr(
            step, max_learning_rate, min_learning_rate, warmup_iters=total_steps // 10, cosine_cycle_iters=total_steps
        )
        for group in opt.param_groups:
            group["lr"] = lr

        opt.step()

        # ==========================================================
        # [TIMING END]: Total Step Time
        # ==========================================================
        if step > warmup_steps:
            torch.cuda.synchronize()  # Wait for optimizer update to finish
            step_end_time = time.time()
            total_step_time += step_end_time - step_start_time

    # ---------------------------------------------------------
    # 3. Print Results (Only on Rank 0)
    # ---------------------------------------------------------
    if rank == 0:
        avg_step_time = total_step_time / benchmark_steps
        avg_comm_time = total_comm_time / benchmark_steps
        comm_proportion = (avg_comm_time / avg_step_time) * 100

        print("\n" + "=" * 50)
        print("📊 BATCHED DDP BENCHMARK RESULTS (1 Node x 2 GPUs)")
        print("=" * 50)
        print(f"Average Time per Step:        {avg_step_time:.4f} seconds")
        print(f"Average Communication Time:   {avg_comm_time:.4f} seconds")
        print(f"Communication Overhead:       {comm_proportion:.2f}%")
        print("=" * 50)

    # Clean up process group
    dist.destroy_process_group()


def main():
    if not torch.cuda.is_available():
        print("No CUDA GPUs available. Please check your environment.")
        return

    world_size = 2
    batch_size = 32

    mp.spawn(
        fn=benchmark_batched_ddp,
        args=(world_size, batch_size),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
