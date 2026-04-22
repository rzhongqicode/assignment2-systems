import os
import timeit

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def setup(rank, world_size, backend):
    """
    Initialize the distributed environment.
    rank: Unique identifier for each process (0 to world_size - 1).
    world_size: Total number of processes participating in the job.
    backend: Communication backend ('gloo' for CPU/Generic, 'nccl' for NVIDIA GPUs).
    """
    # Set the master node address and port for process coordination
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    # Initialize the process group
    dist.init_process_group(backend, rank=rank, world_size=world_size)

    # If using NCCL, bind this process to a specific GPU device
    if backend == "nccl":
        torch.cuda.set_device(rank)


def distributed_comm(rank, world_size, backend, size_mb):
    """
    Core logic for benchmarking distributed communication performance.
    """
    # 1. Setup environment and prepare data
    setup(rank, world_size, backend)
    device = f"cuda:{rank}" if backend == "nccl" else "cpu"

    # Calculate number of elements for the given MB size (1 float32 = 4 bytes)
    num_elements = (size_mb * 1024 * 1024) // 4
    data = torch.randn(num_elements, dtype=torch.float32, device=device)

    # 2. Warm-up phase
    # Ensures kernels are loaded and memory is allocated before actual measurement
    warmup_steps = 5
    for _ in range(warmup_steps):
        dist.all_reduce(data, async_op=False)
        if backend == "nccl":
            torch.cuda.synchronize()  # Wait for GPU kernel to finish

    # 3. Benchmark phase
    steps = 20
    total_time = 0.0
    for _ in range(steps):
        # Ensure GPU is idle before starting the timer
        if backend == "nccl":
            torch.cuda.synchronize()

        begin_time = timeit.default_timer()

        # Execute All-Reduce (Sum data from all ranks and redistribute to all)
        dist.all_reduce(data, async_op=False)

        # Synchronize again because CUDA calls are asynchronous
        if backend == "nccl":
            torch.cuda.synchronize()

        end_time = timeit.default_timer()
        total_time += end_time - begin_time

    # 4. Local calculation
    avarage_time = total_time / steps
    print(f"rank:{rank}, avarage time: {avarage_time}")

    # 5. Global synchronization of results
    # Use all_gather_object to collect the timing (a Python float) from all ranks
    object_list = [None for _ in range(world_size)]
    dist.all_gather_object(object_list, avarage_time)

    # 6. Reporting (Rank 0 acts as the coordinator/logger)
    if rank == 0:
        # The bottleneck of distributed systems is usually the slowest member (Max time)
        max_time = max(object_list)
        print(f"Backend: {backend:^4} | Size: {size_mb:^4}MB | Processes: {world_size}")
        print(f"Ranks time respectively: {[f'{t:.5f}s' for t in object_list]}")
        print(f"Max communication time: {max_time:.5f}s\n" + "-" * 50)

    # Clean up and release the process group
    dist.destroy_process_group()


if __name__ == "__main__":
    # Configuration parameters
    world_size = 4  # Number of parallel processes to spawn
    backend = "gloo"  # 'gloo' for CPU testing; use 'nccl' for NVIDIA GPUs
    size_mb = 10  # Payload size per communication in Megabytes

    # Spawn the processes
    # mp.spawn handles the creation of 'world_size' processes,
    # passing the 'rank' automatically as the first argument to 'distributed_comm'.
    mp.spawn(fn=distributed_comm, args=(world_size, backend, size_mb), nprocs=world_size, join=True)
