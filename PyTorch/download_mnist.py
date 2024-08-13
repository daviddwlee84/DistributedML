from torchvision import datasets
import os

curr_dir = os.path.dirname(os.path.abspath(__file__))
train_dataset = datasets.MNIST(
    root=os.path.join(curr_dir, "DistributedDataParallel", "data"),
    train=True,
    download=True,
)
