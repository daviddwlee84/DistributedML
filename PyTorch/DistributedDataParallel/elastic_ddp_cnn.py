import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms
import os

curr_dir = os.path.dirname(os.path.abspath(__file__))


def setup_and_get_rank() -> int:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    print(f"Initialized process group with rank {rank}.")
    return rank


def cleanup():
    dist.destroy_process_group()


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.layer1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.layer2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.flatten(x, 1)
        # RuntimeError: GET was unable to find an engine to execute this computation
        # https://stackoverflow.com/questions/75776895/runtimeerror-get-was-unable-to-find-an-engine-to-execute-this-computation-when
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train(epochs=5):
    rank = setup_and_get_rank()
    # NOTE: device_id should be unique, so make sure you have proper worker number
    device_id = rank % torch.cuda.device_count()
    print(f"Running on rank {rank}. (local device: {device_id})")

    # Create model and move it to GPU with id rank
    model = SimpleCNN().to(device_id)
    ddp_model = DDP(model, device_ids=[device_id])

    # Define loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(ddp_model.parameters(), lr=0.001)

    # Data loading code
    transform = transforms.Compose([transforms.ToTensor()])
    # NOTE: we should download it first
    train_dataset = datasets.MNIST(
        root=os.path.join(curr_dir, "data"),
        train=True,
        download=False,
        transform=transform,
    )

    train_sampler = DistributedSampler(train_dataset, rank=rank)
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=64, sampler=train_sampler
    )

    # Training loop
    for epoch in range(1, epochs + 1):
        ddp_model.train()
        train_sampler.set_epoch(epoch)  # works when use shuffle
        epoch_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device_id), target.to(device_id)

            optimizer.zero_grad()
            output = ddp_model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if batch_idx % 100 == 0 and rank == 0:
                print(
                    f"Epoch [{epoch}/{epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}"
                )

        if rank == 0:
            print(
                f"Epoch [{epoch}/{epochs}] completed. Loss: {epoch_loss / len(train_loader):.4f}"
            )
            # Save the model checkpoint
            torch.save(ddp_model.state_dict(), f"./model_epoch_{epoch}.pth")

    # Final model saving
    if rank == 0:
        torch.save(ddp_model.state_dict(), "./final_model.pth")

    cleanup()


if __name__ == "__main__":
    train(epochs=10)
