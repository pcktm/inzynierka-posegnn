import os
import pykitti
from glob import glob
from torch_geometric.data import Data
from dataset import KittiSequenceDataset
import numpy as np

basedir = "/home/pcktm/inzynierka/kitti/dataset"
train_sequences = ["05"]

# create directory for features
os.makedirs("models", exist_ok=True)

dataset = KittiSequenceDataset(basedir, train_sequences[0])

for i, data in enumerate(dataset):
    print(i)
    print(data)
    if i == 10:
        break
