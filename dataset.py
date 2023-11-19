import torch
import torch.utils.data.dataset as dataset
import pykitti
import numpy as np
from utils import extract_position_rotation


class KittiSequenceDataset(dataset.Dataset):
    def __init__(
        self,
        basedir,
        sequence,
        transform=None,
        load_images=True,
        return_rich_sample=False,
    ):
        self.basedir = basedir
        self.sequence = sequence
        self.transform = transform
        self.load_images = load_images
        self.dataset = pykitti.odometry(basedir, sequence)
        self.features = self.load_features()
        self.return_rich_sample = return_rich_sample

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        if self.load_images:
            image_rgb = self.dataset.get_cam2(
                index
            )  # rgb left since this is what the pose is referring to
        else:
            image_rgb = None
        try:
            pose = self.dataset.poses[index]
        except IndexError as e:
            print(f"Sequence {self.sequence} has no pose at index {index}")
            raise e

        sample = {
            "image": image_rgb,
            "pose": extract_position_rotation(pose),
            "features": self.features[index],
            "timestamp": self.dataset.timestamps[index],
        }

        if self.transform:
            sample = self.transform(sample)

        if self.return_rich_sample:
            return sample
        
        # for label concat position and rotation as quaternions
        label = np.concatenate(
            (sample["pose"]["position"], sample["pose"]["rotation"].as_quat())
        )

        return sample["features"], label

    def load_features(self):
        features = []
        for index in range(self.__len__()):
            try:
                f = np.load(f"features/{self.sequence}/{index}.npy")
                assert f.shape == (
                    2048,
                ), f"Features at index {index} have shape {f.shape}"
                features.append(f)
            except FileNotFoundError as e:
                print(f"Sequence {self.sequence} has no features at index {index}")
                raise e

        return features
