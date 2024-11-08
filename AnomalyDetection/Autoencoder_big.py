import torch 
from torch.functional import F
from torch import nn

class Autoencoder_Modified(nn.Module):
    def __init__(self):
        super(Autoencoder_Modified, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),  # (Batch_size, 3, H, W) -> (Batch_size, 16, H, W)
            nn.ReLU(),
            nn.MaxPool2d(2),  # (Batch_size, 16, H, W) -> (Batch_size, 16, H/2, W/2)
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # (Batch_size, 16, H/2, W/2) -> (Batch_size, 32, H/2, W/2)
            nn.ReLU(),
            nn.MaxPool2d(2),  # (Batch_size, 32, H/2, W/2) -> (Batch_size, 32, H/4, W/4)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # (Batch_size, 32, H/4, W/4) -> (Batch_size, 64, H/4, W/4)
            nn.ReLU(),
            nn.MaxPool2d(2),  # (Batch_size, 64, H/4, W/4) -> (Batch_size, 64, H/8, W/8)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # (Batch_size, 64, H/8, W/8) -> (Batch_size, 128, H/8, W/8)
            nn.ReLU(),
            nn.MaxPool2d(2),  # (Batch_size, 128, H/8, W/8) -> (Batch_size, 128, H/16, W/16)
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # (Batch_size, 128, H/16, W/16) -> (Batch_size, 128, H/16, W/16)
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # (Batch_size, 128, H/16, W/16) -> (Batch_size, 128, H/16, W/16)
            nn.ReLU(),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # (Batch_size, 128, H/16, W/16) -> (Batch_size, 128, H/8, W/8)
            nn.Conv2d(128, 64, kernel_size=3, padding=1),  # (Batch_size, 128, H/8, W/8) -> (Batch_size, 64, H/8, W/8)
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # (Batch_size, 64, H/8, W/8) -> (Batch_size, 64, H/4, W/4)
            nn.Conv2d(64, 32, kernel_size=3, padding=1),  # (Batch_size, 64, H/4, W/4) -> (Batch_size, 32, H/4, W/4)
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # (Batch_size, 32, H/4, W/4) -> (Batch_size, 32, H/2, W/2)
            nn.Conv2d(32, 16, kernel_size=3, padding=1),  # (Batch_size, 32, H/2, W/2) -> (Batch_size, 16, H/2, W/2)
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # (Batch_size, 16, H/2, W/2) -> (Batch_size, 16, H, W)
            nn.Conv2d(16, 3, kernel_size=3, padding=1),  # (Batch_size, 16, H, W) -> (Batch_size, 3, H, W)
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        bottleneck_output = self.bottleneck(encoded)
        decoded = self.decoder(bottleneck_output)
        return decoded
