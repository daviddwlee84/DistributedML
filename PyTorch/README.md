
## Concept

[PyTorch Distributed Overview — PyTorch Tutorials 2.4.0+cu121 documentation](https://pytorch.org/tutorials/beginner/dist_overview.html)

1. Use [DistributedDataParallel (DDP)](https://pytorch.org/docs/stable/notes/ddp.html), if your model fits in a single GPU but you want to easily scale up training using multiple GPUs.
    - Use [**torchrun**](https://pytorch.org/docs/stable/elastic/run.html), to launch multiple pytorch processes if you are you using more than one node.
      - [Quickstart — PyTorch 2.4 documentation](https://pytorch.org/docs/stable/elastic/quickstart.html)
    - See also: [**Getting Started with Distributed Data Parallel**](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
2. Use [FullyShardedDataParallel (FSDP)](https://pytorch.org/docs/stable/fsdp.html) when your model cannot fit on one GPU.
    - See also: [Getting Started with FSDP](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
3. Use [Tensor Parallel (TP)](https://pytorch.org/docs/stable/distributed.tensor.parallel.html) and/or [Pipeline Parallel (PP)](https://pytorch.org/docs/main/distributed.pipelining.html) if you reach scaling limitations with FSDP.
    - Try our [Tensor Parallelism Tutorial](https://pytorch.org/tutorials/intermediate/TP_tutorial.html)
    - See also: [TorchTitan end to end example of 3D parallelism](https://github.com/pytorch/torchtitan)

[Distributed communication package - torch.distributed — PyTorch 2.4 documentation](https://pytorch.org/docs/stable/distributed.html)

Which backend to use?

- Rule of thumb
    - Use the NCCL backend for distributed **GPU** training
    - Use the Gloo backend for distributed **CPU** training.
- GPU hosts with InfiniBand interconnect
    - Use NCCL, since it’s the only backend that currently supports InfiniBand and GPUDirect.
- GPU hosts with Ethernet interconnect
    - Use NCCL, since it currently provides the best distributed GPU training performance, especially for multiprocess single-node or multi-node distributed training. If you encounter any problem with NCCL, use Gloo as the fallback option. (Note that Gloo currently runs slower than NCCL for GPUs.)
- CPU hosts with InfiniBand interconnect
    - If your InfiniBand has enabled IP over IB, use Gloo, otherwise, use MPI instead. We are planning on adding InfiniBand support for Gloo in the upcoming releases.
- CPU hosts with Ethernet interconnect
    - Use Gloo, unless you have specific reasons to use MPI.

## Resources

Tutorial

- [Basics of Distributed Training — PyTorch edition | by Rohit Kewalramani | Medium](https://medium.com/@rohit.k/basics-of-distributed-training-pytorch-edition-5cbd8fb06bf8)
- [Efficient Training on Multiple GPUs](https://huggingface.co/docs/transformers/perf_train_gpu_many)
- [What is Distributed Data Parallel (DDP) — PyTorch Tutorials 2.4.0+cu121 documentation](https://pytorch.org/tutorials/beginner/ddp_series_theory.html)
  - [Multi GPU training with DDP — PyTorch Tutorials 2.4.0+cu121 documentation](https://pytorch.org/tutorials/beginner/ddp_series_multigpu.html)
- [Multinode Training — PyTorch Tutorials 2.4.0+cu121 documentation](https://pytorch.org/tutorials/intermediate/ddp_series_multinode.html)

Example

- [examples/imagenet/main.py at main · pytorch/examples](https://github.com/pytorch/examples/blob/26de41904319c7094afc53a3ee809de47112d387/imagenet/main.py#L136C17-L141)

---

- [pytorch/elastic: PyTorch elastic training](https://github.com/pytorch/elastic)
- [Torch Distributed Elastic — PyTorch 2.4 documentation](https://pytorch.org/docs/stable/distributed.elastic.html)
