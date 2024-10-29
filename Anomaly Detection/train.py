import torch
import torch.nn as nn
import numpy as np
from Autoencoder import Autoencoder 
from utils import HelicoDatasetAnomalyDetection


def train(model, loss_function, optimizer, dataset, num_epochs=10):
	"""
	Train the model on the given dataset for the specified number of epochs.

	:param model: The model to train
	:param loss_function: The loss function to use
	:param optimizer: The optimizer to use
	:param dataset: The dataset to train on
	:param num_epochs: The number of epochs to train for
	"""
	for epoch in range(num_epochs):
		model.train()
		total_loss = 0
		for i, data in enumerate(dataset):
			optimizer.zero_grad()
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
	# Train the model
	train(model, loss_function, optimizer, dataset)
	# Save the model
	torch.save(model.state_dict(), "model.pth")
