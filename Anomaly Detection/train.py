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
		# Set the model to training mode
		model.train()
		# Initialize the total loss for this epoch
		total_loss = 0
		# Iterate over the dataset
		for i, (data, _) in enumerate(dataset):
			# Zero the gradients
			optimizer.zero_grad()
			# Forward pass
			output = model(data)
			# Calculate the loss
			loss = loss_function(output, data)
			# Backward pass
			loss.backward()
			# Optimize
			optimizer.step()
			# Add the loss to the total loss
			total_loss += loss.item()
		# Print the average loss for this epoch
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
