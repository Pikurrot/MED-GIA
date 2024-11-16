import torch
import torch.nn as nn
import wandb
import yaml
import os
from train import train_ae
from Autoencoder import Autoencoder
from Autoencoder_big import ImprovedAutoencoder
from utils import HelicoDatasetAnomalyDetection, HelicoDatasetClassification, HelicoDatasetPatientDiagnosis, \
	get_cropped_patient_ids, get_negative_patient_ids, get_diagnosis_patient_ids
from cross_val import inference_ae, find_optimal_threshold, evaluate_classification, classify_patches_diag, \
	get_percentage_positive_patches
from torch.utils.data import DataLoader


def main():
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

	# Models to train
	models = {
		'Autoencoder': Autoencoder,
		'ImprovedAutoencoder': ImprovedAutoencoder
	}

	# Create datasets
	crop_neg_patient_ids = list(set(crop_patient_ids) & set(neg_patient_ids))
	neg_cropped_dataset = HelicoDatasetAnomalyDetection(
		patient_ids_to_include=crop_neg_patient_ids,
		train_ratio=1.0
	)
	print(f"AE Train set size: {len(neg_cropped_dataset)}")
	clas_annot_dataset = HelicoDatasetClassification(
		patient_id=True,
		patient_ids_to_include=crop_patient_ids,
		train_ratio=1.0,
	)
	print(f"Classification Train set size: {len(clas_annot_dataset)}")
	patient_ids_diag = list(set(crop_patient_ids) & set(diag_patient_ids))
	train_diag_cropped_dataset = HelicoDatasetPatientDiagnosis(
		patient_ids_to_include=patient_ids_diag,
		split="train",
		train_ratio=1.0
	)
	test_diag_holdout_dataset = HelicoDatasetPatientDiagnosis(
		split="test",
	)
	print(f"Patient Diagnosis Train set size: {len(train_diag_cropped_dataset)}")
	print(f"Patient Diagnosis Validation set size: {len(test_diag_holdout_dataset)}")

	# Create DataLoaders
	neg_cropped_dataloader = DataLoader(neg_cropped_dataset, batch_size=256, shuffle=True)
	clas_annot_dataloader = DataLoader(clas_annot_dataset, batch_size=256, shuffle=True)
	train_diag_cropped_dataloader = DataLoader(train_diag_cropped_dataset, batch_size=256, shuffle=True)
	test_diag_holdout_dataloader = DataLoader(test_diag_holdout_dataset, batch_size=256, shuffle=True)

	# Loop over each model
	for model_name, ModelClass in models.items():
		# Initialize wandb for this model
		wandb.init(project="MED-GIA", name=model_name, reinit=True)
		wandb.config = {
			"learning_rate": 0.001,
			"epochs": 1,
			"batch_size": 256,
			"optimizer": "adam",
			"model": model_name,
		}

		# ===================== TRAINING =====================
		
		print(f"\n\nTraining model: {model_name}")

		# Initialize the model, loss function, optimizer, scheduler
		model = ModelClass()
		loss_function = nn.MSELoss()
		optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config["learning_rate"])
		scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

		# Train the model (cropped neg)
		train_ae(model, loss_function, optimizer, scheduler, neg_cropped_dataloader, device, num_epochs=1)

		# Inferece on the patch classification set (annotated)
		red_fracs, patch_labels, _ = inference_ae(model, clas_annot_dataloader, device)

		# Find the optimal threshold on red fraction (annotated)
		optimal_threshold_pc, roc_auc_pc = find_optimal_threshold(red_fracs, patch_labels)
		print(f"Optimal threshold on red fraction: {optimal_threshold_pc}")
		print(f"ROC AUC: {roc_auc_pc}")

		# Classify patches using the optimal threshold (cropped)
		pred_diag_train, patient_ids_diag_train, gt_patient_diagnosis_train = classify_patches_diag(
			model, train_diag_cropped_dataloader, device, optimal_threshold_pc
		)

		# Get percentage of positive patches for each patient (cropped)
		gt_pp_train, pred_pp_train = get_percentage_positive_patches(
			patient_ids_diag_train, pred_diag_train, gt_patient_diagnosis_train
		)

		# Find optimal threshold on percentage of positive patches (cropped)
		optimal_threshold_pp, roc_auc_pp = find_optimal_threshold(pred_pp_train, gt_pp_train)
		print(f"Optimal threshold on percentage of positive patches: {optimal_threshold_pp}")
		print(f"ROC AUC: {roc_auc_pp}")

		
		# ===================== TESTING =====================

		print(f"\n\nTesting model: {model_name}")

		# Classify patches using the optimal threshold (holdout)
		pred_diag_holdout, patient_ids_diag_holdout, gt_patient_diagnosis_holdout = classify_patches_diag(
			model, test_diag_holdout_dataloader, device, optimal_threshold_pc
		)

		# Get percentage of positive patches for each patient (holdout)
		gt_patient_diagnosis_holdout, pred_pp_holdout = get_percentage_positive_patches(
			patient_ids_diag_holdout, pred_diag_holdout, gt_patient_diagnosis_holdout
		)

		# Predict patient diagnosis (holdout)
		pred_diag_holdout = (pred_pp_holdout > optimal_threshold_pp).astype(int)

		# Evaluate patient diagnosis (holdout)
		accuracy, precision, recall, f1_score, conf_matrix_diag = evaluate_classification(
			pred_diag_holdout, gt_patient_diagnosis_holdout
		)

		# Log metrics to wandb
		print(f"\nResults for model: {model_name}")
		print(f"- Accuracy: {accuracy}")
		print(f"- Precision: {precision}")
		print(f"- Recall: {recall}")
		print(f"- F1 Score: {f1_score}")
		print(f"- Confusion Matrix:\n{conf_matrix_diag}")

		# Finish WandB run
		wandb.finish()


if __name__ == "__main__":
	main()
