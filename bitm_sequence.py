import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pykitti
from tqdm import tqdm
import os
import tensorflow_hub as hub

basedir = "/home/pcktm/inzynierka/kitti/dataset"
sequences = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]

module = hub.KerasLayer(
    "https://www.kaggle.com/models/google/bit/frameworks/TensorFlow2/variations/m-r50x1/versions/1"
)

for sequence in tqdm(sequences, total=len(sequences)):
    os.makedirs(f"features_bitm/{sequence}", exist_ok=True)
    kitti_dataset = pykitti.odometry(basedir, sequence)

    for index, img in tqdm(
        enumerate(kitti_dataset.cam2),
        total=kitti_dataset.__len__(),
        desc=f"Sequence {sequence}",
    ):
      input = np.array(img)

      output = module([input])
      output = output.numpy()

      np.save(f"features_bitm/{sequence}/{index}.npy", output[0])