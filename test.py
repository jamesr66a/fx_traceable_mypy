import torch

def traceable(fn):
    return fn

@traceable
class MyModule:
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x.foo()
        if x.sum() > 0:
            return x + 1
        else:
            return x - 1
