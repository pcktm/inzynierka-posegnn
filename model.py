import torch
import torch.nn.functional as F
from torch_geometric.nn import GraphConv


class PoseLoss(torch.nn.Module):
    def __init__(self, alpha=200):
        super(PoseLoss, self).__init__()
        self.alpha = alpha

    def forward(self, pred, target):
        # loss(X) = ||pˆ− p||2 + α||qˆ− q||2
        # pˆ, qˆ - predicted position and orientation
        # p, q - ground truth position and orientation
        # alpha - hyperparameter to weight the orientation loss

        # pred is a tuple of (position, orientation)
        pred_position, pred_orientation = pred
        target_position, target_orientation = target

        position_loss = F.mse_loss(pred_position, target_position)
        orientation_loss = F.mse_loss(pred_orientation, target_orientation)

        return position_loss + orientation_loss * self.alpha


class PoseGNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GraphConv(
            2048, 256
        )  # GraphConv means Weisfeiler and Leman graph convolution, which paper suggests as superior
        self.conv2 = GraphConv(256, 128)
        self.conv3 = GraphConv(128, 64)

        # 3 for x, y, z
        self.position = torch.nn.Linear(64, 3)

        # 4 for quaternion
        self.orientation = torch.nn.Linear(64, 4)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))

        position = self.position(x)
        orientation = self.orientation(x)

        # should quaternions be normalized?
        orientation = F.normalize(orientation, p=2, dim=-1)

        return position, orientation
