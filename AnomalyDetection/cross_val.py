import torch
import torch.nn as nn
import wandb
import numpy as np
import pandas as pd
import yaml
import os
from sklearn.model_selection import KFold
from train import train_ae
from Autoencoder import Autoencoder
from Autoencoder_big import ImprovedAutoencoder
from utils import HelicoDatasetAnomalyDetection, HelicoDatasetClassification, HelicoDatasetPatientDiagnosis,\
	check_red_fraction,	get_cropped_patient_ids, postprocess, get_negative_patient_ids, get_diagnosis_patient_ids
from torch.utils.data import DataLoader
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
	for batch in dataloader:
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

def classify_patches_diag(model, dataloader, device, threshold):
	red_fracs = []
	patient_ids = []
	gt_patient_diagnosis = []
	for batch in dataloader:
		b_orig_images = batch[0].to(device)
		patient_ids.extend(batch[1])
		gt_patient_diagnosis.extend(batch[2])
		with torch.no_grad():
			reco_images = model(b_orig_images)
		for i in range(len(b_orig_images)):
			red_fracs.append(check_red_fraction(
				postprocess(b_orig_images[i]),
				postprocess(reco_images[i])
			))
	red_fracs = np.array(red_fracs)
	patient_ids = np.array(patient_ids)
	gt_patient_diagnosis = np.array(gt_patient_diagnosis)
	predictions = (red_fracs > threshold).astype(int)
	return predictions, patient_ids, gt_patient_diagnosis

def evaluate_classification(predictions, labels):
	tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
	epsilon = 1e-10
	accuracy = (tp + tn) / (tp + tn + fp + fn + epsilon)
	precision = tp / (tp + fp + epsilon)
	recall = tp / (tp + fn + epsilon)
	f1_score = 2 * (precision * recall) / (precision + recall + epsilon)
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
		"GT patient diagnosis": gt_patient_diagnosis
	})

	# percentage of positive patches per patient
	pos_patches_per_patient = results.groupby("Patient ID")["Pred Patch Class"].mean().reset_index()
	pos_patches_per_patient["Pred Patch Class"] = pos_patches_per_patient["Pred Patch Class"] * 100
	pos_patches_per_patient.rename(columns={"Pred Patch Class": "Percentage of Positive Patches"}, inplace=True)

	# add the patient diagnosis column
	patient_diagnosis = results.groupby("Patient ID")["GT patient diagnosis"].first().reset_index()
	pos_patches_per_patient = pos_patches_per_patient.merge(patient_diagnosis, on="Patient ID")

	gt = pos_patches_per_patient["GT patient diagnosis"].to_numpy(dtype=np.int32)
	pred = pos_patches_per_patient["Percentage of Positive Patches"].to_numpy(dtype=np.float32)
	return gt, pred


def k_fold_cross_validation(k=5, num_epochs=1):
	# Set device
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Using device {device}")

	# Get all patient IDs from Cropped
	dataset_path = yaml.safe_load(open("config.yml", "r"))["dataset_path"]
	cropped_path = os.path.join(dataset_path, "CrossValidation", "Cropped")
	crop_patient_ids = get_cropped_patient_ids(cropped_path)

	# Get all negative patient IDs
	csv_file_path = os.path.join(dataset_path, "PatientDiagnosis.csv")
	neg_patient_ids = get_negative_patient_ids(csv_file_path)

	# Get all diagnosis patient IDs
	diag_patient_ids = get_diagnosis_patient_ids(csv_file_path)

	# Set up KFold cross-validator
	kf = KFold(n_splits=k, shuffle=True, random_state=42)

	# Models to train
	models = {
		'Autoencoder': Autoencoder,
		'ImprovedAutoencoder': ImprovedAutoencoder
	}

	lst_conf_matrix_class = []
	lst_conf_matrix_improved_class = []
	lst_conf_matrix_diag = []
	lst_conf_matrix_improved_diag = []

	# Loop over each fold
	for fold, (train_index, test_index) in enumerate(kf.split(crop_patient_ids)):
		print(f"\nStarting fold {fold + 1}/{k}")

		# Get train and test patient IDs
		train_patient_ids = [crop_patient_ids[i] for i in train_index]
		val_patient_ids = [crop_patient_ids[i] for i in test_index]

		# Create datasets for this fold
		train_patient_ids_neg = list(set(train_patient_ids) & set(neg_patient_ids))
		val_patient_ids_neg = list(set(val_patient_ids) & set(neg_patient_ids))
		train_dataset_ae = HelicoDatasetAnomalyDetection(
			patient_ids_to_include=train_patient_ids_neg,
			train_ratio=1.0
		)
		val_dataset_ae = HelicoDatasetAnomalyDetection(
			patient_ids_to_include=val_patient_ids_neg,
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
			patient_ids_to_include=val_patient_ids,
			train_ratio=1.0
		)
		print(f"Classification Train set size: {len(train_dataset_clas)}")
		print(f"Classification Validation set size: {len(val_dataset_clas)}")
		train_patient_ids_diag = list(set(crop_patient_ids) & set(diag_patient_ids))
		val_patient_ids_diag = list(set(val_patient_ids) & set(diag_patient_ids))
		train_dataset_diag = HelicoDatasetPatientDiagnosis(
			patient_ids_to_include=train_patient_ids_diag,
			train_ratio=1.0
		)
		val_dataset_diag = HelicoDatasetPatientDiagnosis(
			patient_ids_to_include=val_patient_ids_diag,
			train_ratio=1.0
		)
		print(f"Patient Diagnosis Train set size: {len(train_dataset_diag)}")
		print(f"Patient Diagnosis Validation set size: {len(val_dataset_diag)}")

		# Create DataLoaders
		train_loader_ae = DataLoader(train_dataset_ae, batch_size=256, shuffle=True)
		val_loader_ae = DataLoader(val_dataset_ae, batch_size=256, shuffle=False)
		train_loader_clas = DataLoader(train_dataset_clas, batch_size=256, shuffle=True)
		val_loader_clas = DataLoader(val_dataset_clas, batch_size=256, shuffle=False)
		train_loader_diag = DataLoader(train_dataset_diag, batch_size=256, shuffle=True)
		val_loader_diag = DataLoader(val_dataset_diag, batch_size=256, shuffle=False)

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
			train_red_fracs, train_labels, _ = inference_ae(model, train_loader_clas, device)

			# Find optimal threshold on red fraction
			optimal_threshold_rf, roc_auc_rf = find_optimal_threshold(train_red_fracs, train_labels)
			print(f"Optimal threshold on red fraction: {optimal_threshold_rf}")
			print(f"ROC AUC: {roc_auc_rf}")

			# Classify patches using the optimal threshold
			print("Classifying patches using the optimal threshold")
			pred_patch_class, gt_patch_class, _ = classify_patches(model, val_loader_clas, device, optimal_threshold_rf)

			# Evaluate patch classification
			print("Evaluating patch classification")
			_, _, _, _, conf_matrix_class = evaluate_classification(pred_patch_class, gt_patch_class)

			# Classify patches again for diagnosis
			print("Classifying patches for patient diagnosis")
			pred_diag_train, patient_ids_diag_train, gt_patient_diagnosis_train = classify_patches_diag(model, train_loader_diag, device, optimal_threshold_rf)

			# Get percentage of positive patches per patient
			print("Getting percentage of positive patches per patient")
			gt_pp_train, pred_pp_train = get_percentage_positive_patches(patient_ids_diag_train, pred_diag_train, gt_patient_diagnosis_train)

			# Find optimal threshold on percentage of positive patches
			print("Finding optimal threshold on percentage of positive patches")
			optimal_threshold_pp, roc_auc_pp = find_optimal_threshold(pred_pp_train, gt_pp_train)
			print(f"Optimal threshold on percentage of positive patches: {optimal_threshold_pp}")
			print(f"ROC AUC: {roc_auc_pp}")

			# Predict patient diagnosis
			pred_pp_val, patient_ids_diag_val, gt_patient_diagnosis_val = classify_patches_diag(model, val_loader_diag, device, optimal_threshold_rf)
			gt_pp_val, pred_pp_val = get_percentage_positive_patches(patient_ids_diag_val, pred_pp_val, gt_patient_diagnosis_val)
			pred_diagnosis = (pred_pp_val > optimal_threshold_pp).astype(int)

			# Evaluate patient diagnosis
			_, _, _, _, conf_matrix_diag = evaluate_classification(pred_diagnosis, gt_pp_val)

			# Aggregate results and log to WandB if last fold
			if model_name == "Autoencoder":
				lst_conf_matrix_class.append(conf_matrix_class)
				lst_conf_matrix_diag.append(conf_matrix_diag)
			else:
				lst_conf_matrix_improved_class.append(conf_matrix_class)
				lst_conf_matrix_improved_diag.append(conf_matrix_diag)

			if fold == k - 1:
				if model_name == "Autoencoder":
					accuracy, precision, recall, f1_score, conf_matrix_class = aggregate_results(lst_conf_matrix_class)
					print("\nAggregated results for Autoencoder:")
					print("- Patch classification:")
					print(f"\t- Accuracy: {accuracy}")
					print(f"\t- Precision: {precision}")
					print(f"\t- Recall: {recall}")
					print(f"\t- F1 Score: {f1_score}")
					print(f"\t- Confusion Matrix:\n{conf_matrix_class}")
					print("- Patient diagnosis:")
					accuracy, precision, recall, f1_score, conf_matrix_diag = aggregate_results(lst_conf_matrix_diag)
					print(f"\t- Accuracy: {accuracy}")
					print(f"\t- Precision: {precision}")
					print(f"\t- Recall: {recall}")
					print(f"\t- F1 Score: {f1_score}")
					print(f"\t- Confusion Matrix:\n{conf_matrix_diag}")
				else:
					accuracy_improved, precision_improved, recall_improved, f1_score_improved, conf_matrix_improved = aggregate_results(lst_conf_matrix_improved_class)
					print("\nAggregated results for Improved Autoencoder:")
					print("- Patch classification:")
					print(f"\t- Accuracy: {accuracy_improved}")
					print(f"\t- Precision: {precision_improved}")
					print(f"\t- Recall: {recall_improved}")
					print(f"\t- F1 Score: {f1_score_improved}")
					print(f"\t- Confusion Matrix:\n{conf_matrix_improved}")
					print("- Patient diagnosis:")
					accuracy_improved, precision_improved, recall_improved, f1_score_improved, conf_matrix_improved = aggregate_results(lst_conf_matrix_improved_diag)
					print(f"\t- Accuracy: {accuracy_improved}")
					print(f"\t- Precision: {precision_improved}")
					print(f"\t- Recall: {recall_improved}")
					print(f"\t- F1 Score: {f1_score_improved}")
					print(f"\t- Confusion Matrix:\n{conf_matrix_improved}")

			# Finish WandB run
			wandb.finish()


if __name__ == "__main__":
	k = 5
	num_epochs = 1

	k_fold_cross_validation(k=k, num_epochs=num_epochs)
