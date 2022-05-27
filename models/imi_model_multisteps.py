"""
This file will include the networks for imitation learning
"""
from torchvision.models import resnet18
from torchvision.models.feature_extraction import create_feature_extractor,get_graph_node_names
import torch
from torch import nn

class Imi_network_multisteps(nn.Module):
    def __init__(self):
        super().__init__()
        # self.K = K

        ##input dim 3: xyz
        self.eef = nn.Sequential(
            nn.Linear(3, 100),
            nn.ReLU(),
            nn.Linear(100, 60),
            nn.ReLU(),
            nn.Linear(60, 9),
        )

    def forward(self, x, pos):
        xy_space = {0: -.003, 1: 0, 2: .003}
        z_space = {0:-.0015, 1: 0, 2: .0015}
        actions = []        
        action = self.image_model(x)
        actions.append(action)
        for i in range(self.K):
            action = action.view(-1, 3, 3)
            action_pred = torch.argmax(action, dim=-1)
            x_action = xy_space[action_pred[:,0].data]
            y_action = xy_space[action_pred[:,1].data]
            z_action = z_space[action_pred[:,2].data]
            pred_pos = torch.as_tensor([pos[:,0]+x_action, pos[:,1]+y_action, pos[:,2]+z_action])
            action = self.eef(pred_pos)
            actions.append(action)
        return torch.as_tensor(actions)


