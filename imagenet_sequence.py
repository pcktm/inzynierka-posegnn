import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import pykitti
from tqdm import tqdm
import os

model = torch.hub.load("pytorch/vision:v0.10.0", "resnet50", pretrained=True)

# Replace the last fully-connected layer with Identity to remove the classification head
model.fc = nn.Identity()

if torch.cuda.is_available():
    model.to("cuda")

basedir = "/home/pcktm/inzynierka/kitti/dataset"
sequences = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]

transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

for sequence in tqdm(sequences, total=len(sequences)):

    os.makedirs(f"features/{sequence}", exist_ok=True)
    kitti_dataset = pykitti.odometry(basedir, sequence)

    for index, img in tqdm(
        enumerate(kitti_dataset.cam2), total=kitti_dataset.__len__(), desc=f"Sequence {sequence}"
    ):
        input_tensor = transform(img)
        input_batch = input_tensor.unsqueeze(0)

        if torch.cuda.is_available():
            input_batch = input_batch.to("cuda")

        with torch.no_grad():
            output = model(input_batch)

        output = output.cpu().numpy()
        output = output.squeeze()

        np.save(f"features/{sequence}/{index}.npy", output)
