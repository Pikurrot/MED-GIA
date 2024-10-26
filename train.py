import torch
import torch.nn as nn
import numpy as np
from Autoencoder import Autoencoder 


model = Autoencoder()
loss = nn.MSELoss() 
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

