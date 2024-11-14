import torch
import torch.nn as nn
import wandb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from train import train_ae
from Autoencoder import Autoencoder
from Autoencoder_big import ImprovedAutoencoder
from utils import HelicoDatasetAnomalyDetection, HelicoDatasetClassification, check_red_fraction, postprocess
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix


def evaluate_ae(model, loss_function, dataloader, device):
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

def inference_ae(model, dataloader, device):
	red_fracs = []
	labels = []
	patient_ids = []
	for batch in tqdm(dataloader):
		b_orig_images = batch[0].to(device)
		b_labels = batch[1].to(device)
		patient_ids.extend(batch[2])
		with torch.no_grad():
			reco_images = model(b_orig_images)
		
		for i in range(len(b_orig_images)):
			red_fracs.append(check_red_fraction(
				postprocess(b_orig_images[i]),
				postprocess(reco_images[i])
			))
		labels.extend(b_labels.cpu().numpy())
	return np.array(red_fracs), np.array(labels), patient_ids

def find_optimal_threshold(preds, labels):
	fpr, tpr, thresholds = roc_curve(labels, preds)
	roc_auc = roc_auc_score(labels, preds)
	optimal_idx = np.argmax(tpr - fpr)
	optimal_threshold = thresholds[optimal_idx]
	return optimal_threshold, roc_auc

def classify_patches(model, dataloader, device, threshold):
	red_fracs, labels, patient_ids = inference_ae(model, dataloader, device)
	predictions = (red_fracs > threshold).astype(int)
	return predictions, labels, patient_ids

def evaluate_classification(predictions, labels):
	tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
	accuracy = (tp + tn) / (tp + tn + fp + fn)
	precision = tp / (tp + fp)
	recall = tp / (tp + fn)
	f1_score = 2 * (precision * recall) / (precision + recall)
	conf_matrix = np.array([[tn, fp], [fn, tp]])
	return accuracy, precision, recall, f1_score, conf_matrix

def aggregate_results(lst_conf_matrix):
	agg_conf_matrix = np.sum(lst_conf_matrix, axis=0)
	tn, fp, fn, tp = agg_conf_matrix.ravel()
	accuracy = (tp + tn) / (tp + tn + fp + fn)
	precision = tp / (tp + fp)
	recall = tp / (tp + fn)
	f1_score = 2 * (precision * recall) / (precision + recall)
	return accuracy, precision, recall, f1_score, agg_conf_matrix

def get_percentage_positive_patches(patient_ids, patch_preds, gt_patient_diagnosis):
	results = pd.DataFrame({
		"Patient ID": patient_ids,
		"Pred Patch Class": patch_preds,
		"GT Patient diagnosis": gt_patient_diagnosis
	})

	# percentage of positive patches per patient
	pos_patches_per_patient = results.groupby("Patient ID")["Pred Patch Class"].mean().reset_index()
	pos_patches_per_patient["Pred Patch Class"] = pos_patches_per_patient["Pred Patch Class"] * 100
	pos_patches_per_patient.rename(columns={"Pred Patch Class": "Percentage of Positive Patches"}, inplace=True)

	# add the patient diagnosis column
	patient_diagnosis = results.groupby("Patient ID")["GT patient diagnosis"].first().reset_index()
	pos_patches_per_patient = pos_patches_per_patient.merge(patient_diagnosis, on="Patient ID")

	gt = pos_patches_per_patient["GT patient diagnosis"]
	pred = pos_patches_per_patient["Percentage of Positive Patches"]
	return gt, pred


def k_fold_cross_validation(k=5, num_epochs=1):
	# Set device
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Using device {device}")

	# Get all negative patient IDs from Cropped
	patient_ids = HelicoDatasetAnomalyDetection(train_ratio=1.0).patient_ids

	# Set up KFold cross-validator
	kf = KFold(n_splits=k, shuffle=True, random_state=42)

	# Models to train
	models = {
		'Autoencoder': Autoencoder,
		'ImprovedAutoencoder': ImprovedAutoencoder
	}

	lst_conf_matrix = []
	lst_conf_matrix_improved = []

	# Loop over each fold
	for fold, (train_index, test_index) in enumerate(kf.split(patient_ids)):
		print(f"\nStarting fold {fold + 1}/{k}")

		# Get train and test patient IDs
		train_patient_ids = [patient_ids[i] for i in train_index]
		test_patient_ids = [patient_ids[i] for i in test_index]

		# Create datasets for this fold
		train_dataset_ae = HelicoDatasetAnomalyDetection(
			patient_ids_to_include=train_patient_ids,
			train_ratio=1.0
		)
		val_dataset_ae = HelicoDatasetAnomalyDetection(
			patient_ids_to_include=test_patient_ids,
			train_ratio=1.0
		)
		print(f"AE Train set size: {len(train_dataset_ae)}")
		print(f"AE Validation set size: {len(val_dataset_ae)}")
		train_dataset_clas = HelicoDatasetClassification(
			patient_id=True,
			patient_ids_to_include=train_patient_ids,
			train_ratio=1.0
		)
		val_dataset_clas = HelicoDatasetClassification(
			patient_id=True,
			patient_ids_to_include=test_patient_ids,
			train_ratio=1.0
		)
		print(f"Classification Train set size: {len(train_dataset_clas)}")
		print(f"Classification Validation set size: {len(val_dataset_clas)}")

		# Create DataLoaders
		train_loader_ae = DataLoader(train_dataset_ae, batch_size=256, shuffle=True)
		val_loader_ae = DataLoader(val_dataset_ae, batch_size=256, shuffle=False)
		train_loader_clas = DataLoader(train_dataset_clas, batch_size=256, shuffle=True)
		val_loader_clas = DataLoader(val_dataset_clas, batch_size=256, shuffle=False)

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

			# Train the AE model
			train_ae(model, loss_function, optimizer, scheduler, train_loader_ae, device, num_epochs=num_epochs)

			# Evaluate the AE model
			val_loss = evaluate_ae(model, loss_function, val_loader_ae, device)
			print(f"Validation Loss for model {model_name} on fold {fold + 1}: {val_loss}")
			wandb.log({"validation_loss": val_loss})

			# Save the model
			model_filename = f"{model_name}_fold{fold + 1}.pth"
			torch.save(model.state_dict(), model_filename)
			wandb.save(model_filename)

			# Inference on the train patch classification set
			train_red_fracs, train_labels, train_patient_ids = inference_ae(model, train_loader_clas, device)

			# Find optimal threshold on red fraction
			optimal_threshold, roc_auc = find_optimal_threshold(train_red_fracs, train_labels)
			print(f"Optimal threshold on red fraction: {optimal_threshold}")
			print(f"ROC AUC: {roc_auc}")

			# Classify patches using the optimal threshold
			predictions, labels, patient_ids = classify_patches(model, val_loader_clas, device, optimal_threshold)

			# Evaluate patch classification
			_, _, _, _, conf_matrix = evaluate_classification(predictions, labels)
			if model_name == "Autoencoder":
				lst_conf_matrix.append(conf_matrix)
			else:
				lst_conf_matrix_improved.append(conf_matrix)

			# Aggregate results and log to WandB if last fold
			if fold == k - 1:
				if model_name == "Autoencoder":
					accuracy, precision, recall, f1_score, conf_matrix = aggregate_results(lst_conf_matrix)
					print("\nAggregated results for Autoencoder:")
					print(f"- Accuracy: {accuracy}")
					print(f"- Precision: {precision}")
					print(f"- Recall: {recall}")
					print(f"- F1 Score: {f1_score}")
					print(f"- Confusion Matrix:\n{conf_matrix}")
				else:
					accuracy_improved, precision_improved, recall_improved, f1_score_improved, conf_matrix_improved = aggregate_results(lst_conf_matrix_improved)
					print("\nAggregated results for Improved Autoencoder:")
					print(f"- Accuracy: {accuracy_improved}")
					print(f"- Precision: {precision_improved}")
					print(f"- Recall: {recall_improved}")
					print(f"- F1 Score: {f1_score_improved}")
					print(f"- Confusion Matrix:\n{conf_matrix_improved}")

			# Finish WandB run
			wandb.finish()


if __name__ == "__main__":
	k = 5
	num_epochs = 1

	k_fold_cross_validation(k=k, num_epochs=num_epochs)
