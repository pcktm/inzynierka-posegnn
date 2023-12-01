import torch
import torch.utils.data.dataset as dataset
from torch_geometric.data import Dataset as GraphDataset
from torch_geometric.data import Data
import pykitti
import numpy as np
from utils import extract_position_rotation


class KittiSequenceDataset(dataset.Dataset):
    def __init__(
        self,
        basedir,
        sequence,
        feature_dir="features",
        transform=None,
        load_images=True,
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


class KittiGraphDataset(dataset.Dataset):
    def __init__(
        self, basedir, sequence, graph_length=5, transform=None, feature_dir="features"
    ) -> None:
        super().__init__()
        self.dataset = KittiSequenceDataset(
            basedir,
            sequence,
            load_images=False,
            return_rich_sample=False,
            feature_dir=feature_dir,
        )
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


class KittiGraphDatasetWithGraphBasedOnVectorDistance(dataset.Dataset):
    def __init__(self, basedir, sequence, graph_length=5, transform=None) -> None:
        super().__init__()
        self.dataset = KittiSequenceDataset(
            basedir, sequence, load_images=False, return_rich_sample=False
        )
        self.graph_length = graph_length
        self.transform = transform

    def __getitem__(self, index):
        """
        Returns a graph of length self.graph_length, constructed by taking the frame at index and
        the closest self.graph_length frames with distance of the feature vector.
        """
        if torch.is_tensor(index):
            index = index.tolist()

        nodes = []
        y = []

        first_node, first_label = self.dataset[index]
        nodes.append(first_node)
        y.append(first_label)

        closest_indices = self.dataset.find_closest_to(index, self.graph_length - 1)
        for i in closest_indices:
            node, label = self.dataset[i]
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
        basedir,
        sequences,
        dataset=KittiGraphDataset,
        transform=None,
        graph_length=5,
        feature_dir="features",
    ) -> None:
        super().__init__()
        self.sequences = sequences
        self.graph_length = graph_length
        self.datasets = [
            dataset(basedir, seq, graph_length, transform, feature_dir=feature_dir)
            for seq in sequences
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
