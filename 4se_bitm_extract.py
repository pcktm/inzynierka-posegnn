import numpy as np
import matplotlib.pyplot as plt
from dataset import FourSeasonsDataset
from tqdm import tqdm
import os
import tensorflow_hub as hub
import cv2

basedir = "/home/pcktm/inzynierka/4seasons"
sequences = ["office_loop_3", "neighboorhood_6"]

module = hub.KerasLayer(
    "https://www.kaggle.com/models/google/bit/frameworks/TensorFlow2/variations/m-r50x1/versions/1"
)

for sequence in tqdm(sequences, total=len(sequences)):
    os.makedirs(f"features_4se_bitm/{sequence}", exist_ok=True)
    dataset = FourSeasonsDataset(basedir, sequence, load_features=False, return_rich_sample=True)

    for index, sample in tqdm(
        enumerate(dataset),
        total=dataset.__len__(),
        desc=f"Sequence {sequence}",
    ):  
        input = sample["frame_id"]
        img = cv2.imread(f"{basedir}/{sequence}/undistorted_images/cam0/{input}.png", cv2.IMREAD_GRAYSCALE)

        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        cv2.imshow("img", img)
        cv2.waitKey(1)

        output = module([img])
        output = output.numpy()

        np.save(f"features_4se_bitm/{sequence}/{input}.npy", output[0])

cv2.destroyAllWindows()