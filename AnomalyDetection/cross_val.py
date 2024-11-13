import torch
import torch.nn as nn
import wandb
import yaml
import os
from sklearn.model_selection import KFold
from train import train
from Autoencoder import Autoencoder
from Autoencoder_big import ImprovedAutoencoder
from utils import HelicoDatasetAnomalyDetection, get_negative_patient_ids
from torch.utils.data import DataLoader

def evaluate(model, loss_function, dataloader, device):
	model.eval()
	total_loss = 0
	with torch.no_grad():
		for data in dataloader:
			data = data.to(device)
			output = model(data)
			loss = loss_function(output, data)
			total_loss += loss.item()
	avg_loss = total_loss / len(dataloader)
	return avg_loss

def k_fold_cross_validation(k=5, num_epochs=10):
	# Set device
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Using device {device}")

	# Path to your CSV file
	dataset_path = yaml.safe_load(open("config.yml", "r"))["dataset_path"]
	csv_file_path = os.path.join(dataset_path, "PatientDiagnosis.csv")

	# Get all negative patient IDs
	patient_ids = get_negative_patient_ids(csv_file_path)

	# Set up KFold cross-validator
	kf = KFold(n_splits=k, shuffle=True, random_state=42)

	# Models to train
	models = {
		'Autoencoder': Autoencoder,
		'ImprovedAutoencoder': ImprovedAutoencoder
	}

	# Loop over each fold
	for fold, (train_index, test_index) in enumerate(kf.split(patient_ids)):
		print(f"\nStarting fold {fold + 1}/{k}")

		# Get train and test patient IDs
		train_patient_ids = [patient_ids[i] for i in train_index]
		test_patient_ids = [patient_ids[i] for i in test_index]

		# Create datasets for this fold
		train_dataset = HelicoDatasetAnomalyDetection(
			patient_ids_to_include=train_patient_ids,
			train_ratio=1.0
		)
		test_dataset = HelicoDatasetAnomalyDetection(
			patient_ids_to_include=test_patient_ids,
			train_ratio=1.0
		)

		# Create DataLoaders
		train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
		test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

		# Loop over each model
		for model_name, ModelClass in models.items():
			print(f"\nTraining model: {model_name} on fold {fold + 1}")

			# Initialize WandB for this model and fold
			wandb.init(project="MED-GIA", name=f"{model_name}_fold{fold + 1}", reinit=True)
			wandb.config = {
				"learning_rate": 0.001,
				"epochs": num_epochs,
				"batch_size": 256,
				"optimizer": "adam",
				"model": model_name,
				"fold": fold + 1
			}

			# Initialize the model, loss function, optimizer, scheduler
			model = ModelClass()
			loss_function = nn.MSELoss()
			optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config["learning_rate"])
			scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

			# Train the model
			train(model, loss_function, optimizer, scheduler, train_loader, device, num_epochs=num_epochs)

			# Evaluate the model
			val_loss = evaluate(model, loss_function, test_loader, device)
			print(f"Validation Loss for model {model_name} on fold {fold + 1}: {val_loss}")
			wandb.log({"validation_loss": val_loss})

			# Save the model
			model_filename = f"{model_name}_fold{fold + 1}.pth"
			torch.save(model.state_dict(), model_filename)
			wandb.save(model_filename)

			# Finish WandB run
			wandb.finish()


if __name__ == "__main__":
	k = 5
	num_epochs = 2

	k_fold_cross_validation(k=k, num_epochs=num_epochs)