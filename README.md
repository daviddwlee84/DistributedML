# DistributedML

Distributed Machine Learning (multiple machine, multiple GPUs)

## Getting Started

```bash
python -m venv .venv
source .venv/bin/activate
```

> Assume you already have CUDA setup on the machine

Check CUDA version

```bash
# https://stackoverflow.com/questions/9727688/how-to-get-the-cuda-version
$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2022 NVIDIA Corporation
Built on Wed_Jun__8_16:49:14_PDT_2022
Cuda compilation tools, release 11.7, V11.7.99
Build cuda_11.7.r11.7/compiler.31442593_0

$ nvidia-smi
...
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.54.14              Driver Version: 550.54.14      CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
...

$ ls -l /usr/local | grep cuda
```

Install PyTorch

```bash
# https://pytorch.org/get-started/locally/
$ pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 
```
