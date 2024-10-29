import torch
import torch.nn as nn
import numpy as np
from Autoencoder import Autoencoder 
from utils import HelicoDatasetAnomalyDetection



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
	for epoch in range(num_epochs):
		model.train()
		total_loss = 0
		for i, data in enumerate(dataset): # Data is a tensor (B, C, H, W)
			optimizer.zero_grad()
			data = data.to(device)
			output = model(data)
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
	# Train the model
	train(model, loss_function, optimizer, dataset, device)
	# Save the model
	torch.save(model.state_dict(), "model.pth")
