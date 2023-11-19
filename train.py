import os
import pykitti
from glob import glob
from torch_geometric.data import Data
from dataset import KittiSequenceDataset
from torch.utils.data import DataLoader
from scipy.spatial.transform import Rotation as R

basedir = "/home/pcktm/inzynierka/kitti/dataset"
train_sequences = ["01"]

# create directory for features
os.makedirs("models", exist_ok=True)

dataset = KittiSequenceDataset(basedir, train_sequences[0], load_images=True)
train_dataloader = DataLoader(dataset, batch_size=5, shuffle=True)

train_features, train_labels = next(iter(train_dataloader))
train_features = train_features.squeeze()

print(train_features)
print()                                                                                            
