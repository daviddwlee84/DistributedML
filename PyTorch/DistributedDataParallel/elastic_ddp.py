from typing import Literal
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim

from torch.nn.parallel import DistributedDataParallel as DDP


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net2(self.relu(self.net1(x)))


def demo_basic(backend: Literal["nccl", "gloo", "mpi", "ucc", None] = "nccl") -> None:
    """
    Recommend nccl

    If use None it will try to use gloo and cause error
    [E815 16:11:29.420629547 ProcessGroupGloo.cpp:143] Gloo connectFullMesh failed with [../third_party/gloo/gloo/transport/tcp/pair.cc:144] no error

    Use `ip a` to find the interface corresponding to your IP address
    Then setting GLOO_SOCKET_IFNAME might help
    e.g. export GLOO_SOCKET_IFNAME='eno1'
    """
    dist.init_process_group(backend)
    rank = dist.get_rank()
    print(f"Start running basic DDP example on rank {rank}.")

    # create model and move it to GPU with id rank
    device_id = rank % torch.cuda.device_count()
    model = ToyModel().to(device_id)
    ddp_model = DDP(model, device_ids=[device_id])

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(device_id)
    loss_fn(outputs, labels).backward()
    optimizer.step()
    dist.destroy_process_group()


if __name__ == "__main__":
    demo_basic()
