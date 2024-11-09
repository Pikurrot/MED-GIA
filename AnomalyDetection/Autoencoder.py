import torch 
from torch.functional import F 
from torch import nn

# Autoencoder class input three channels
class Autoencoder(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            # Input = (Batch_size, C, W, H) 
            # (Batch_size, C, W, H) -> (Batch_size, 16, W/2, H/2) 
            # TODO: remove alpha channel
            nn.Conv2d(3, 16, kernel_size = 3, stride=2, padding=1),
            nn.ReLU(),
            
            # (Batch_size, 16, W/2, H/2) -> (Batch_size, 32, W/4, H/4)
            nn.Conv2d(16, 32, kernel_size = 3, stride=2, padding=1),
            nn.ReLU(),

            # (Batch_size, 32, W/4, H/4) -> (Batch_size, 64, W/8, H/8)
            nn.Conv2d(32, 64, kernel_size = 3, stride=2, padding=1),
            nn.ReLU(),

            # (Batch_size, 64, W/8, H/8) -> (Batch_size, 128, W/16, H/16)
            nn.Conv2d(64, 128, kernel_size = 3, stride=2, padding=1),
            nn.ReLU(),

        )
        self.bottleneck = torch.nn.Sequential(
            # (Batch_size, 128, W/16, H/16) -> (Batch_size, 128, W/16, H/16)
            nn.Conv2d(128, 128, kernel_size = 3, stride=1, padding=1),
            nn.ReLU(),

            # (Batch_size, 128, W/16, H/16) -> (Batch_size, 128, W/16, H/16)
            nn.Conv2d(128, 128, kernel_size = 3, stride=1, padding=1),
            nn.ReLU(),
            
        ) 

        self.decoder = torch.nn.Sequential(
            # (Batch_size, 128, W/16, H/16) -> (Batch_size, 64, W/8, H/18)
            nn.ConvTranspose2d(128, 64, kernel_size = 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),

            # (Batch_size, 64, W/8, H/8) -> (Batch_size, 32, W/4, H/4)
            nn.ConvTranspose2d(64, 32, kernel_size = 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),

            # (Batch_size, 32, W/4, H/4) -> (Batch_size, 16, W/2, H/2)
            nn.ConvTranspose2d(32, 16, kernel_size = 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),

             # (Batch_size, 16, W/2, H/2) -> (Batch_size, 3, W, H)
            nn.ConvTranspose2d(16, 3, kernel_size = 3, stride=2, padding=1, output_padding=1),
			nn.Tanh()
        )
    
    
    def forward(self, x) -> torch.Tensor:

        self.encoded  = self.encoder(x)
        self.intermediate_step = self.bottleneck(self.encoded)
        self.decoded = self.decoder(self.intermediate_step)
        return self.decoded
