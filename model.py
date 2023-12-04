import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, GATv2Conv, JumpingKnowledge


class PoseLoss(torch.nn.Module):
    def __init__(self, batch_size, graph_length, alpha=200, use_position_only=False):
        super(PoseLoss, self).__init__()
        self.alpha = alpha
        self.batch_size = batch_size
        self.graph_length = graph_length
        self.use_position_only = use_position_only

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        # loss(X) = ||pˆ− p||2 + α||qˆ− q||2
        # pˆ, qˆ - predicted position and orientation
        # p, q - ground truth position and orientation
        # alpha - hyperparameter to weight the orientation loss

        # take the last and second to last elements of the graph, remember that
        # input pred and target are in the form [x, y, z, w, x, y, z] tensor x graph_length x batch_size
        # torch.Size([512, 7]) for graph_length=8 and batch_size=64

        pred = pred.view(-1, self.graph_length, 3 if self.use_position_only else 7)
        target = target.view(-1, self.graph_length, 3 if self.use_position_only else 7)

        # Get the last and second-to-last nodes for each element in the batch
        last_pred = pred[:, -1, :]
        second_last_pred = pred[:, -2, :]
        last_target = target[:, -1, :]
        second_last_target = target[:, -2, :]

        # Calculate shifts in position and orientation
        if not self.use_position_only:
            shift_pred_orientation = last_pred[:, 3:] - second_last_pred[:, 3:]
            shift_target_orientation = last_target[:, 3:] - second_last_target[:, 3:]

        shift_pred_position = last_pred[:, :3] - second_last_pred[:, :3]
        shift_target_position = last_target[:, :3] - second_last_target[:, :3]

        # Calculate loss
        position_loss = F.mse_loss(shift_pred_position, shift_target_position)
        if not self.use_position_only:
            orientation_loss = F.mse_loss(
                shift_pred_orientation, shift_target_orientation
            )

        if self.use_position_only:
            return position_loss

        # Return weighted sum of position and orientation loss
        total_loss = position_loss + self.alpha * orientation_loss
        return total_loss


class JustLastNodePositionLoss(torch.nn.Module):
    def __init__(self, batch_size, graph_length):
        super(JustLastNodePositionLoss, self).__init__()
        self.batch_size = batch_size
        self.graph_length = graph_length

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        pred = pred.view(-1, self.graph_length, 3)
        target = target.view(-1, self.graph_length, 3)

        # Get the last nodes for each element in the batch
        last_pred = pred[:, -1, :]
        last_target = target[:, -1, :]

        # Return MSE loss
        return F.mse_loss(last_pred, last_target)


class PoseGNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout1 = nn.Dropout(0.5)
        self.lin1 = torch.nn.Linear(2048, 128)
        self.conv1 = GraphConv(
            128, 64
        )  # GraphConv means Weisfeiler and Leman graph convolution, which paper suggests as superior
        # self.conv5 = GraphConv(1024, 512)
        self.dropout2 = nn.Dropout(0.5)
        # self.conv3 = GraphConv(512, 128)
        self.conv4 = GraphConv(64, 32)

        # 3 for x, y, z
        self.position = torch.nn.Linear(32, 3)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.dropout1(x)
        x = F.leaky_relu(self.lin1(x))
        x = F.leaky_relu(self.conv1(x, edge_index))
        # x = F.leaky_relu(self.conv5(x, edge_index))
        x = self.dropout2(x)
        # x = F.leaky_relu(self.conv3(x, edge_index))
        x = F.leaky_relu(self.conv4(x, edge_index))

        position = self.position(x)

        return position

        # should quaternions be normalized?
        orientation = self.orientation(x)
        orientation = F.normalize(orientation, p=2, dim=-1)

        # return position and orientation as one tensor
        return torch.cat((position, orientation), dim=1)
