import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel


class DNNModel(BaseModel):
    def __init__(self, num_targets=3):
        super(DNNModel, self).__init__()
        # self.bn1 = nn.BatchNorm1d(800)
        # self.bn2 = nn.BatchNorm1d(50)
        self.fc1 = nn.Linear(10000, 800)
        self.fc2 = nn.Linear(800, 50)
        self.fc3 = nn.Linear(50, num_targets)

    def forward(self, x):
        # TODO: dropout more
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
