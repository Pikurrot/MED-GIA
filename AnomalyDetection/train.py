import torch
import torch.nn as nn
import numpy as np
from Autoencoder import Autoencoder 
from utils import HelicoDatasetAnomalyDetection
from torch.utils.data import DataLoader



def train(model, loss_function, optimizer, dataset, device, num_epochs=10):
	"""
	Train the model on the given dataset for the specified number of epochs.

	:param model: The model to train
	:param loss_function: The loss function to use
	:param optimizer: The optimizer to use
	:param dataset: The dataset to train on
	:param num_epochs: The number of epochs to train for
	"""
	model = model.to(device) 
	dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
	for epoch in range(num_epochs):
		model.train()
		total_loss = 0
		print(len(dataloader))
		for i, data in enumerate(dataloader): # Data is a tensor (B, C, H, W)
			optimizer.zero_grad()
			data = data.to(device)
			output = model(data)
			print(output.shape, data.shape)
			loss = loss_function(output, data)
			loss.backward()
			optimizer.step()
			total_loss += loss.item()
		print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataset)}")

if __name__ == "__main__":
	# Load the dataset
	dataset = HelicoDatasetAnomalyDetection()
	# Initialize the model
	model = Autoencoder()
	loss_function = nn.MSELoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Using device {device}")
	# Train the model
	train(model, loss_function, optimizer, dataset, device)
	# Save the model
	torch.save(model.state_dict(), "model.pth")
