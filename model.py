import torch
import torch.nn.functional as F
from torch_geometric.nn import GraphConv


class PoseLoss(torch.nn.Module):
    def __init__(self, batch_size, graph_length, alpha=200):
        super(PoseLoss, self).__init__()
        self.alpha = alpha
        self.batch_size = batch_size
        self.graph_length = graph_length

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        # loss(X) = ||pˆ− p||2 + α||qˆ− q||2
        # pˆ, qˆ - predicted position and orientation
        # p, q - ground truth position and orientation
        # alpha - hyperparameter to weight the orientation loss

        # take the last and second to last elements of the graph, remember that
        # input pred and target are in the form [x, y, z, w, x, y, z] tensor x graph_length x batch_size
        # torch.Size([512, 7]) for graph_length=8 and batch_size=64

        pred = pred.view(-1, self.graph_length, 7)
        target = target.view(-1, self.graph_length, 7)

        # Get the last and second-to-last nodes for each element in the batch
        last_pred = pred[:, -1, :]
        second_last_pred = pred[:, -2, :]
        last_target = target[:, -1, :]
        second_last_target = target[:, -2, :]

        # Calculate shifts in position and orientation
        shift_pred_position = last_pred[:, :3] - second_last_pred[:, :3]
        shift_target_position = last_target[:, :3] - second_last_target[:, :3]
        shift_pred_orientation = last_pred[:, 3:] - second_last_pred[:, 3:]
        shift_target_orientation = last_target[:, 3:] - second_last_target[:, 3:]

        # Calculate loss
        position_loss = F.mse_loss(shift_pred_position, shift_target_position)
        orientation_loss = F.mse_loss(shift_pred_orientation, shift_target_orientation)

        # Return weighted sum of position and orientation loss
        total_loss = position_loss + self.alpha * orientation_loss
        return total_loss


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

        x = F.leaky_relu(self.conv1(x, edge_index))
        x = F.leaky_relu(self.conv2(x, edge_index))
        x = F.leaky_relu(self.conv3(x, edge_index))

        position = self.position(x)
        orientation = self.orientation(x)

        # should quaternions be normalized?
        orientation = F.normalize(orientation, p=2, dim=-1)

        # return position and orientation as one tensor
        return torch.cat((position, orientation), dim=1)
