# https://pytorch.org/docs/stable/distributed.html#initialization
import torch
import torch.distributed

print(torch.cuda.is_available())
print(torch.distributed.is_available())
print(torch.distributed.is_nccl_available())
print(torch.distributed.is_gloo_available())
print(torch.distributed.is_mpi_available())
