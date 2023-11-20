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

        return torch.tensor(sample["features"], dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

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

class KittiGraphDataset(dataset.Dataset):
    def __init__(self, basedir, sequence, graph_length = 5) -> None:
        super().__init__()
        self.dataset = KittiSequenceDataset(basedir, sequence, load_images=False, return_rich_sample=False)
        self.graph_length = graph_length

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

        # add edges
        edge_index = []
        for i in range(self.graph_length - 1):
            edge_index.append([i, i + 1])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        return Data(x=torch.stack(nodes), edge_index=edge_index, y=torch.stack(y))
    
    def __len__(self):
        return self.dataset.__len__() - self.graph_length