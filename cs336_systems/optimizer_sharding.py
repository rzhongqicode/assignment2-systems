from typing import Any, Type

import torch
import torch.distributed as dist
from torch.optim import Optimizer


class OptimizerSharding(torch.optim.Optimizer):
    """
    A wrapper class that shards optimizer states across multiple distributed processes.
    This implements the core idea of ZeRO Stage 1: instead of every GPU holding a full
    copy of the optimizer states (like Adam's momentum and variance), each GPU only
    maintains and updates a fraction of the states, drastically reducing memory usage.
    """

    def __init__(self, params, optimizer_cls: Type[Optimizer], **kwargs: Any):
        # 1. Retrieve distributed environment information
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1

        # 2. Store the optimizer class and its hyperparameters (e.g., lr, weight_decay)
        # We don't instantiate the underlying optimizer yet, because the parameters
        # haven't been sharded.
        self.optimizer_cls = optimizer_cls
        self.kwargs = kwargs
        self.optimizer = None

        # 3. Call the superclass constructor.
        # MAGIC ALERT: This automatically calls `self.add_param_group` for each group in `params`.
        # Note: torch.optim.Optimizer expects `params` and `defaults` as arguments.
        super().__init__(params, defaults=kwargs)

    def add_param_group(self, param_group: dict[str, Any]):
        """
        Adds a parameter group to the optimizer and handles the sharding logic.
        This is called automatically during __init__ and can be called manually later.
        """
        # 1. Let the base class record the FULL parameter group.
        # The wrapper needs to know all parameters for the broadcast step later.
        super().add_param_group(param_group)

        # 2. Extract the full list of parameters for this group
        params = param_group["params"]

        # 3. CORE SHARDING LOGIC: Assign a subset of parameters to this specific rank.
        # Using step slicing [start :: step] ensures that regardless of the number
        # of parameters, they are distributed as evenly as possible across all ranks.
        sharded_params = params[self.rank :: self.world_size]

        # 4. Create a new dictionary for the underlying optimizer,
        # containing the exact same hyperparameters but ONLY the sharded parameters.
        sharded_group = {k: v for k, v in param_group.items() if k != "params"}
        sharded_group["params"] = sharded_params

        # 5. Initialize the underlying optimizer if it doesn't exist yet,
        # or add the new sharded group to it if it already exists.
        if self.optimizer is None:
            self.optimizer = self.optimizer_cls([sharded_group], **self.kwargs)
        else:
            self.optimizer.add_param_group(sharded_group)

    def step(self, closure=None, **kwargs):
        """
        Performs a single optimization step and synchronizes updated parameters.
        """
        # 1. The underlying optimizer updates ONLY the parameters assigned to this rank.
        # This saves massive amounts of memory because the optimizer states
        # are only allocated for these specific parameters.
        loss = self.optimizer.step(closure, **kwargs)

        # 2. Synchronization: Broadcast the updated parameters to all other ranks.
        # We iterate over the FULL parameter groups stored in the wrapper (super class).
        for group in self.param_groups:
            for i, param in enumerate(group["params"]):
                # Determine which rank "owns" and just updated this parameter.
                # This math must perfectly match the slicing logic used in `add_param_group`.
                owner_rank = i % self.world_size

                # The owner broadcasts the updated parameter data to all other ranks.
                # All other ranks receive the new data and overwrite their stale copies.
                dist.broadcast(param.data, src=owner_rank, async_op=False)

        return loss
