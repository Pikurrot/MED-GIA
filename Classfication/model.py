import torch
import torch.nn as nn
import torch.nn.functional as F

class HelicobacterClassifier(nn.Module):
    def __init__(self):
        super(HelicobacterClassifier, self).__init__()

        # (B, C, H, W) -> (B, 32, H/2, W/2)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        # (B, 32, H/2, W/2) -> (B, 64, H/4, W/4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # (B, 64, H/4, W/4) -> (B, 128, H/8, W/8)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        # (B, 128, H/8, W/8) -> (B, 128, H/16, W/16)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(128 * 32 * 32, 512)
        self.fc2 = nn.Linear(512, 2)  

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 32 * 32)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
