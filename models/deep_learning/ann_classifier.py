import torch
import torch.nn as nn
import torch.nn.functional as F

class ANNClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_classes=4):
        super(ANNClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)
