import itertools
import os

import submitit

# 1. 配置 Slurm 的要求 (等同于上面的 #SBATCH)
executor = submitit.AutoExecutor(folder="slurm_logs")
executor.update_parameters(timeout_min=60, slurm_partition="gpu", slurm_gres="gpu:1")

# 2. 准备你要扫描的参数网格
batch_sizes = [8, 16, 32]
precisions = ["fp32", "bf16"]

# 获取所有参数组合 (产生 6 种组合)
combinations = list(itertools.product(batch_sizes, precisions))

# 3. 提交任务
jobs = []
for bs, prec in combinations:
    # 这里的 command 就相当于你在终端里敲的命令
    command = ["uv", "run", "python", "benchmark.py", "--batch_size", str(bs), "--precision", prec]

    # 将命令提交给集群
    job = executor.submit(submitit.helpers.CommandFunction(command))
    jobs.append(job)
    print(f"Submitted Job {job.job_id} with BS={bs}, Precision={prec}")

print(f"总共提交了 {len(jobs)} 个任务！")
