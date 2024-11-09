import torch
import torch.nn as nn
import numpy as np
import wandb
from Autoencoder import Autoencoder 
from utils import HelicoDatasetAnomalyDetection
from torch.utils.data import DataLoader
from Autoencoder_big import ImprovedAutoencoder



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
	dataloader = DataLoader(dataset, batch_size=wandb.config["batch_size"], shuffle=True)
	print("Starting training")
	for epoch in range(num_epochs):
		model.train()
		total_loss = 0
		for i, data in enumerate(dataloader): # Data is a tensor (B, C, H, W)
			optimizer.zero_grad()
			data = data.to(device)
			output = model(data)
			loss = loss_function(output, data)
			loss.backward()
			optimizer.step()
			total_loss += loss.item()
			wandb.log({"batch_loss": loss.item()})
		avg_loss = total_loss / len(dataset)
		print(f"Epoch {epoch + 1}, Loss: {avg_loss}")
		wandb.log({"epoch": epoch + 1, "loss": avg_loss})

if __name__ == "__main__":
	# Initialize wandb
	wandb.login(key="07313fef21f9b32f2fb1fb00a2672258c8b5c3d4")
	wandb.init(project="MED-GIA")
	
	# Set hyperparameters
	wandb.config = {
		"learning_rate": 0.001,
		"epochs": 1,
		"batch_size": 256,
		"optimizer" : "adam"
	}

	print("num_epochs: ", wandb.config["epochs"])
	print("batch_size: ", wandb.config["batch_size"])
	print("learning_rate: ", wandb.config["learning_rate"])
	# Load the dataset
	dataset = HelicoDatasetAnomalyDetection()
	# Initialize the model
	model = Autoencoder()
	loss_function = nn.MSELoss()
	# No se si tiene mucho sentido probar con distintos optimizadores que no sean Adam
	# optimizer = torch.optim.SGD(model.parameters(), lr=wandb.config.learning_rate)
	# optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config.learning_rate)
	# optimizer = torch.optim.RMSprop(model.parameters(), lr=wandb.config.learning_rate)
	optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config["learning_rate"])
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Using device {device}")
	# Train the model
	train(model, loss_function, optimizer, dataset, device, num_epochs=wandb.config["epochs"])
	# Save the model
	model_name = "Autoencoder.pth"
	torch.save(model.state_dict(), model_name)
	wandb.save(model_name)
