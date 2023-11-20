import os
from glob import glob
from dataset import KittiGraphDataset, KittiSequenceDataset
from torch_geometric.loader import DataLoader
from scipy.spatial.transform import Rotation as R
import torch
from tqdm import tqdm
from model import PoseGNN, PoseLoss
from utils import normalize_position_and_rotation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

basedir = "/home/pcktm/inzynierka/kitti/dataset"
train_sequences = ["02"]

os.makedirs("models", exist_ok=True)

dataset = KittiGraphDataset(basedir, train_sequences[0])
train_dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

example_graph = next(iter(train_dataloader))
print(example_graph)

model = PoseGNN().to(device)
optimizer = torch.optim.NAdam(model.parameters(), lr=0.001)
criterion = PoseLoss()

for epoch in range(100):
    model.train()
    loss_history = []
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch}", total=len(train_dataloader))
    for batch in progress_bar:
        batch = batch.to(device)
        optimizer.zero_grad()
        output = model(batch)
        loss = criterion(output, batch.y)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
        progress_bar.set_postfix({"loss": loss.item()})

    print(f"Epoch {epoch} loss: {sum(loss_history) / len(loss_history)}")

    torch.save(model.state_dict(), f"models/model_{epoch}.pt")
