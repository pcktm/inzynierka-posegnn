import torch
import torch.utils.data.dataset as dataset
from torch_geometric.data import Dataset as GraphDataset
from torch_geometric.data import Data
import pykitti
import numpy as np
from utils import extract_position_rotation
import os
from typing import Union, List

class KittiSequenceDataset(dataset.Dataset):
    def __init__(
        self,
        basedir,
        sequence,
        feature_dir="features",
        transform=None,
        load_images=False,
        return_rich_sample=False,
        use_position_only=True,
    ):
        self.basedir = basedir
        self.sequence = sequence
        self.transform = transform
        self.load_images = load_images
        self.feature_dir = feature_dir
        self.return_rich_sample = return_rich_sample
        self.use_position_only = use_position_only

        self.dataset = pykitti.odometry(basedir, sequence)
        self.features = self.load_features()

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
        if self.use_position_only:
            label = sample["pose"]["position"]
        else:
            label = np.concatenate(
                (sample["pose"]["position"], sample["pose"]["rotation"].as_quat())
            )

        return torch.tensor(sample["features"], dtype=torch.float32), torch.tensor(
            label, dtype=torch.float32
        )

    def load_features(self):
        features = []
        for index in range(self.__len__()):
            try:
                f = np.load(f"{self.feature_dir}/{self.sequence}/{index}.npy")
                assert f.shape == (
                    2048,
                ), f"Features at index {index} have shape {f.shape}"
                features.append(f)
            except FileNotFoundError as e:
                print(f"Sequence {self.sequence} has no features at index {index}")
                raise e

        return features

    def find_closest_to(self, index, N):
        """
        Find N closest frames to the frame at index.
        """

        def distance(a, b):
            return np.linalg.norm(a - b)

        features = self.features
        distances = [
            distance(features[index], features[i]) for i in range(len(features))
        ]
        sorted_indices = np.argsort(distances)
        return sorted_indices[1 : N + 1]


class FourSeasonsSequenceDataset(dataset.Dataset):
    def __init__(
        self,
        basedir,
        sequence,
        feature_dir="features",
        transform=None,
        load_features=True,
        return_rich_sample=False,
    ):
        super().__init__()
        self.basedir = basedir
        self.sequence = sequence
        self.transform = transform
        self.feature_dir = feature_dir
        self.poses = self.load_poses()
        self.features = self.load_features() if load_features else None
        self.return_rich_sample = return_rich_sample

    def __len__(self):
        return len(self.poses)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
            
        sample = {
            "frame_id": self.poses[index]["frame_id"],
            "translation": [self.poses[index]["translation"][0], self.poses[index]["translation"][2], self.poses[index]["translation"][1]],
            "rotation": self.poses[index]["rotation"],
            "features": self.features[index] if self.features else None,
        }

        if self.transform:
            sample = self.transform(sample)

        if self.return_rich_sample:
            return sample

        return torch.tensor(sample["features"], dtype=torch.float32), torch.tensor(
            sample["translation"], dtype=torch.float32
        )

    def load_poses(self):
        poses = []
        with open(f"{self.basedir}/{self.sequence}/GNSSPoses.txt") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                line = line.split(",")
                frame_id = line[0]
                pose = {
                    "frame_id": frame_id,
                    "translation": np.array(line[1:4], dtype=np.float32),
                    "rotation": np.array(line[4:8], dtype=np.float32),
                    "scale": line[8],
                }
                poses.append(pose)
        return poses

    def load_features(self):
        features = []
        for pose in self.poses:
            try:
                f = np.load(
                    f"{self.feature_dir}/{self.sequence}/{pose['frame_id']}.npy"
                )
                assert f.shape == (
                    2048,
                ), f"Features at id {pose['frame_id']} have shape {f.shape}"
                features.append(f)
            except FileNotFoundError as e:
                print(
                    f"Sequence {self.sequence} has no features at id {pose['frame_id']}"
                )
                raise e
        return features


class SequenceGraphDataset(dataset.Dataset):
    def __init__(
        self,
        base_dataset: Union[KittiSequenceDataset, FourSeasonsSequenceDataset],
        graph_length=5,
        transform=None,
    ) -> None:
        super().__init__()
        self.dataset = base_dataset
        self.graph_length = graph_length
        self.transform = transform

    def __getitem__(self, index):
        """
        Returns a graph of length self.graph_length, constructed by taking the frame at index as the last node
        and the previous self.graph_length frames as leading nodes.
        """
        if torch.is_tensor(index):
            index = index.tolist()

        nodes = []
        y = []

        for i in range(self.graph_length - 1):
            node, label = self.dataset[index + i]
            nodes.append(node)
            y.append(label)

        # add the last node
        node, label = self.dataset[index + self.graph_length]
        nodes.append(node)
        y.append(label)

        # add edges, unidirected, all nodes are connected to each other
        edge_index = []
        for i in range(len(nodes)):
            for j in range(len(nodes)):
                if i != j:
                    edge_index.append([i, j])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        nodes = torch.stack(nodes)
        y = torch.stack(y)

        if self.transform:
            nodes, edge_index, y = self.transform(nodes, edge_index, y)

        return Data(x=nodes, edge_index=edge_index, y=y)

    def __len__(self):
        return self.dataset.__len__() - self.graph_length

class MultipleSequenceGraphDataset(dataset.Dataset):
    def __init__(
        self,
        sequences: List[Union[KittiSequenceDataset, FourSeasonsSequenceDataset]],
        transform=None,
        graph_length=5,
    ) -> None:
        super().__init__()
        self.graph_length = graph_length
        self.datasets = [
            SequenceGraphDataset(sequence, graph_length=graph_length, transform=transform)
            for sequence in sequences
        ]

    def __getitem__(self, index):
        # find the dataset that contains the index and remember that index in that dataset should be local
        dataset_index = 0
        while index >= len(self.datasets[dataset_index]):
            index -= len(self.datasets[dataset_index])
            dataset_index += 1
        return self.datasets[dataset_index][index]

    def __len__(self):
        return sum([len(dataset) for dataset in self.datasets])
