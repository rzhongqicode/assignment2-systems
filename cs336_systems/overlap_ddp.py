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
