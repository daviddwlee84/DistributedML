# Distributed Machine Learning (multiple machine (nodes), multiple GPUs (workers))

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

Install Other Dependencies

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
```

---

## Trouble Shooting

Cuda 12.0 x PyTorch

- [ImportError: /usr/local/lib/python3.10/dist-packages/torch/lib/../../nvidia/cusparse/lib/libcusparse.so.12: undefined symbol: __nvJitLinkAddData_12_1, version libnvJitLink.so.12 · Issue #111469 · pytorch/pytorch](https://github.com/pytorch/pytorch/issues/111469)
- [Is CUDA 12.0 supported with any pytorch version? - PyTorch Forums](https://discuss.pytorch.org/t/is-cuda-12-0-supported-with-any-pytorch-version/197636/5)
- [Pytorch for CUDA12.0 - PyTorch Live - PyTorch Forums](https://discuss.pytorch.org/t/pytorch-for-cuda12-0/192737)

```bash
export LD_LIBRARY_PATH="`pwd`/.venv/lib64/python3.8/site-packages/nvidia/nvjitlink/lib":$LD_LIBRARY_PATH
```
