import torch
import torch.distributed as dist
from torch import nn


class Overlap_Wrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        # Broadcast initial parameters from rank 0 to ensure all start identically
        for param in model.parameters():
            dist.broadcast(tensor=param.data, src=0, async_op=False)
        # get world size
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1

        self.handles = []
        # set hook on model parameters
        for param in self.model.parameters():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(hook=self.hook)

    def forward(self, *inputs, **kwargs):
        return self.model(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        for handle in self.handles:
            handle.wait()
        self.handles.clear()

        # avarage grad
        for param in self.model.parameters():
            if param.requires_grad and param.grad is not None:
                param.grad.data /= self.world_size

    def hook(self, tensor):
        handle = dist.all_reduce(tensor.grad.data, async_op=True)
        self.handles.append(handle)


import torch
import torch.distributed as dist
from torch import nn


class Overlap_Wrapper(nn.Module):
    """
    Naive DDP implementation: Overlaps computation and communication
    by asynchronously all-reducing each parameter's gradient individually.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

        # Broadcast initial parameters from rank 0 to ensure identical initialization across all ranks
        for param in model.parameters():
            dist.broadcast(tensor=param.data, src=0, async_op=False)

        # Get the total number of processes (world size) for gradient averaging
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1

        self.handles = []

        # Register a post-accumulate-grad hook for every parameter that requires a gradient
        for param in self.model.parameters():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(hook=self.hook)

    def forward(self, *inputs, **kwargs):
        # Pass-through all arguments to the underlying model
        return self.model(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        """
        Wait for all asynchronous communication to finish, then average the gradients.
        Must be called before optimizer.step().
        """
        # Block until all async all-reduce operations are completed
        for handle in self.handles:
            handle.wait()
        self.handles.clear()

        # Average the gradients across all processes
        for param in self.model.parameters():
            if param.requires_grad and param.grad is not None:
                param.grad.data /= self.world_size

    def hook(self, tensor):
        """
        Triggered immediately when a parameter's gradient finishes computation.
        Dispatches an async all-reduce operation.
        """
        handle = dist.all_reduce(tensor.grad.data, async_op=True)
        self.handles.append(handle)


class Overlap_Wrapper_bucketed(nn.Module):
    """
    Advanced DDP implementation: Overlaps computation and communication
    while minimizing communication overhead using Gradient Bucketing.
    """

    def __init__(self, model: nn.Module, bucket_size_mb: float):
        super().__init__()
        self.model = model
        self.bucket_size_mb = bucket_size_mb

        # Broadcast initial parameters from rank 0
        for param in model.parameters():
            dist.broadcast(tensor=param.data, src=0, async_op=False)

        self.world_size = dist.get_world_size() if dist.is_initialized() else 1

        # ---------------------------------------------------------
        # Bucketing Configuration & Pre-allocation
        # ---------------------------------------------------------
        self.buckets = []  # Stores lists of parameters for each bucket
        self.bucket_ready_counts = []  # Tracks how many gradients are ready in each bucket
        self.bucket_total_counts = []  # Tracks the total number of gradients expected in each bucket
        self.param_to_bucket_id = {}  # Maps a parameter object to its corresponding bucket ID

        # Convert max bucket size from Megabytes to number of float32 elements (4 bytes per float)
        max_numel = (self.bucket_size_mb * 1024 * 1024) // 4

        cur_bucket_numel = 0
        cur_bucket = []
        cur_bucket_id = 0

        # Iterate in reverse order: later layers compute backward first,
        # so they should be bucketed and communicated first to maximize overlap.
        for param in reversed(list(self.model.parameters())):
            if param.requires_grad:
                # Add condition `not cur_bucket` to handle the edge case where a single
                # parameter is larger than the bucket size limit.
                if cur_bucket_numel + param.numel() <= max_numel or not cur_bucket:
                    cur_bucket_numel += param.numel()
                    cur_bucket.append(param)
                    self.param_to_bucket_id[param] = cur_bucket_id
                else:
                    # Current bucket is full, save it and start a new one
                    self.buckets.append(cur_bucket)
                    self.bucket_ready_counts.append(0)
                    self.bucket_total_counts.append(len(cur_bucket))

                    cur_bucket = [param]
                    cur_bucket_numel = param.numel()
                    cur_bucket_id += 1
                    self.param_to_bucket_id[param] = cur_bucket_id

        # Append the final partially-filled bucket (if any)
        if len(cur_bucket) > 0:
            self.buckets.append(cur_bucket)
            self.bucket_ready_counts.append(0)
            self.bucket_total_counts.append(len(cur_bucket))

        # infobox stores tuples of: (bucket_id, handle, flattened_tensor, grad_list_template)
        self.infobox = []

        # ---------------------------------------------------------
        # Hook Registration
        # ---------------------------------------------------------
        for param in self.model.parameters():
            if param.requires_grad:
                # Pass the parameter to the closure factory to retain its identity
                param.register_post_accumulate_grad_hook(hook=self._my_hook(param))

    def forward(self, *inputs, **kwargs):
        return self.model(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        """
        Wait for all bucketed async communications to finish, unflatten the tensors,
        average the gradients, and assign them back to the model parameters.
        """
        # Process each bucket that was dispatched for communication
        for bucket_id, handle, cur_tensor, cur_grad_list in self.infobox:
            handle.wait()

            # Divide the entire flattened tensor by world_size (highly optimized GPU operation)
            cur_tensor.div_(self.world_size)

            # Unflatten the continuous memory block back into individual gradient shapes
            unflattened_grad_list = torch._utils._unflatten_dense_tensors(cur_tensor, cur_grad_list)

            # Distribute the averaged gradients back to their respective parameters
            i = 0
            for p in self.buckets[bucket_id]:
                p.grad.data = unflattened_grad_list[i]
                i += 1

            # Reset the ready count specifically for this completed bucket
            self.bucket_ready_counts[bucket_id] = 0

        # Clear the pending communication queue for the next training step
        self.infobox.clear()

    def _my_hook(self, param):
        """
        Closure factory: Generates a specific hook for a parameter,
        allowing it to know which bucket it belongs to.
        """

        def hook(tensor):
            bucket_id = self.param_to_bucket_id[param]
            self.bucket_ready_counts[bucket_id] += 1

            # Check if this parameter is the last piece of the puzzle for its bucket
            if self.bucket_ready_counts[bucket_id] == self.bucket_total_counts[bucket_id]:
                # Gather all gradients in this bucket
                grad_list = [p.grad.data for p in self.buckets[bucket_id]]

                # Flatten them into a contiguous 1D tensor to maximize network bandwidth
                flattened_tensor = torch._utils._flatten_dense_tensors(grad_list)

                # Dispatch the async all-reduce operation
                handle = dist.all_reduce(tensor=flattened_tensor, async_op=True)

                # Store the handle and necessary unpacking metadata
                self.infobox.append((bucket_id, handle, flattened_tensor, grad_list))

        return hook
