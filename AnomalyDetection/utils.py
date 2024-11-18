from typing import Any
from torch.utils.data import Dataset
from PIL import Image
import os
import pandas as pd
import yaml
import numpy as np
import torch
import cv2
from torchvision import transforms
from typing import List, Optional, Tuple, Literal

default_path = "/fhome/vlia/HelicoDataSet"
config_path = "config.yml"

def ensure_dataset_path_yaml() -> int:
	"""
	Ensures that config.yml contains the dataset path.
	:return: 0 if path is valid, 1 if path is not present, 2 if path is invalid
	"""
	# Check config.yml exists
	config = {}

	# Check if config.yml exists
	if not os.path.exists(config_path):
		config["dataset_path"] = default_path
		with open(config_path, "w") as file:
			yaml.safe_dump(config, file)
		return 1

	# Load existing config
	try:
		with open(config_path, "r") as file:
			config = yaml.safe_load(file) or {}
	except yaml.YAMLError as e:
		raise ValueError(f"Error parsing {config_path}: {e}")

	# Check if 'dataset_path' exists in config
	if "dataset_path" not in config:
		config["dataset_path"] = default_path
		with open(config_path, "w") as file:
			yaml.safe_dump(config, file)
		return 1

	# Check if the dataset path exists on the filesystem
	if not os.path.exists(config["dataset_path"]):
		return 2

	return 0

def listdir(path: str, filter: str = None, extension: str = None) -> list:
	"""
	Returns a list of directories in the given path.
	If a filter is provided, only directories that contain the filter in their name will be returned.

	:param path: Path to the directory
	:param filter: Filter to apply to the directories
	:return: List of directory names as strings
	"""
	directories = os.listdir(path)
	if filter is not None:
		directories = [directory for directory in directories if filter in directory]
	if extension is not None:
		directories = [directory for directory in directories if directory.endswith(extension)]
	return directories

def transform_image(image: Image, size: tuple) -> Image:
	transformations = transforms.Compose([
		transforms.Resize(size),
		transforms.ToTensor(),
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])
	return transformations(image)

def get_cropped_patient_ids(cropped_path: str) -> List[str]:
	# Get all the patient IDs from the cropped path
	patient_ids = listdir(cropped_path)
	patient_ids = [patient_id[:-2] for patient_id in patient_ids]
	return patient_ids

def get_negative_patient_ids(csv_file_path: str) -> List[str]:
	# Gets all the patient IDs with a negative diagnosis
	data = pd.read_csv(csv_file_path)
	negative_diagnosis = data[data["DENSITAT"] == "NEGATIVA"]
	patient_ids = negative_diagnosis["CODI"].astype(str).tolist()
	return patient_ids

def get_classification_patient_ids(excel_file_path: str) -> List[str]:
	# Gets all the patient IDs from the excel file
	data = pd.read_excel(excel_file_path)
	patient_ids = data["Pat_ID"].astype(str).tolist()
	unique_patient_ids = sorted(list(set(patient_ids)))
	return unique_patient_ids

def get_diagnosis_patient_ids(csv_file_path: str) -> List[str]:
	# Gets all the patient IDs except with diagnosis BAIXA
	data = pd.read_csv(csv_file_path)
	negative_diagnosis = data[data["DENSITAT"] != "BAIXA"]
	patient_ids = negative_diagnosis["CODI"].astype(str).tolist()
	return patient_ids

def postprocess(tensor, p=4):
	if tensor.dim() == 4:
		tensor = tensor.squeeze(0)
	tensor = tensor.cpu()
	tensor = tensor * 0.5 + 0.5
	tensor = tensor * 255
	tensor = tensor.clamp(0, 255)
	tensor = tensor.type(torch.uint8)
	tensor = tensor.permute(1, 2, 0).numpy()
	if p > 0:
		# center crop
		tensor = tensor[p:256-p, p:256-p, :]
		# padd again with blue color
		tensor = cv2.copyMakeBorder(tensor, p, p, p, p, cv2.BORDER_CONSTANT, value=[228, 232, 248])
	return tensor

def check_red_fraction(orig, reco):
	# Convert the images from RGB to HSV
	input_image_hsv = cv2.cvtColor(orig, cv2.COLOR_RGB2HSV)
	reconstructed_image_hsv = cv2.cvtColor(reco, cv2.COLOR_RGB2HSV)

	# Define the hue range for red pixels as suggested
	# For -20 to 0 in HSV, which corresponds to hues 160-180 in OpenCV's scale
	input_lower_hsv1 = np.array([340, 0, 0])
	input_upper_hsv1 = np.array([360, 255, 255])

	# For 0 to 20 in HSV
	input_lower_hsv2 = np.array([0, 0, 0])
	input_upper_hsv2 = np.array([20, 255, 255])

	# Apply masks to the original image
	mask_ori1 = cv2.inRange(input_image_hsv, input_lower_hsv1, input_upper_hsv1)
	mask_ori2 = cv2.inRange(input_image_hsv, input_lower_hsv2, input_upper_hsv2)
	mask_ori = cv2.bitwise_or(mask_ori1, mask_ori2)
	count_red_ori = np.count_nonzero(mask_ori)

	# Apply masks to the reconstructed image
	mask_rec1 = cv2.inRange(reconstructed_image_hsv, input_lower_hsv1, input_upper_hsv1)
	mask_rec2 = cv2.inRange(reconstructed_image_hsv, input_lower_hsv2, input_upper_hsv2)
	mask_rec = cv2.bitwise_or(mask_rec1, mask_rec2)
	count_red_rec = np.count_nonzero(mask_rec)

	# Calculate the fraction of red pixels
	F_red = count_red_ori / (count_red_rec+1)

	return F_red


class HelicoDatasetAnomalyDetection(Dataset):
	def __init__(
			self,
			patient_id: bool=False,
			split: Literal["train", "val"]="train",
			train_ratio: float=0.8,
			random_seed: int=42,
			patient_ids_to_include: Optional[List[str]]=None
	):
		super().__init__()
		# Initialize paths
		path_error = ensure_dataset_path_yaml()
		if path_error == 1:
			print(f"Dataset path not found in config.yml. Defaulting to {default_path}")
			if not os.path.exists(default_path):
				raise FileNotFoundError(f"Default path {default_path} does not exist. Specify a valid path in config.yml.")
		elif path_error == 2:
			current_path = yaml.safe_load(open("config.yml", "r"))["dataset_path"]
			raise FileNotFoundError(f"Dataset path {current_path} does not exist. Specify a valid path in config.yml.")
		
		self.dataset_path = yaml.safe_load(open("config.yml", "r"))["dataset_path"]
		self.csv_file_path = os.path.join(self.dataset_path, "PatientDiagnosis.csv")
		self.cropped_path = os.path.join(self.dataset_path, "CrossValidation", "Cropped")
		self.patient_id = patient_id

		# Find all the negative diagnosis directories
		paths_negatives, patients_ids = self.get_negative_diagnosis_directories(self.csv_file_path)

		# Split the data into train and test sets
		if patient_ids_to_include is None:
			patient_ids_to_include = get_negative_patient_ids(self.csv_file_path)
		train_size = int(len(patient_ids_to_include) * train_ratio)
		train_indices = np.random.RandomState(random_seed).choice(len(patient_ids_to_include), train_size, replace=False)
		test_indices = np.array([i for i in range(len(patient_ids_to_include)) if i not in train_indices])
		if split == "train":
			patient_ids_to_include = [patient_ids_to_include[i] for i in train_indices]
		else:
			patient_ids_to_include = [patient_ids_to_include[i] for i in test_indices]

		# Filter by patient_ids_to_include if provided
		paths = [os.path.join(self.cropped_path, filename) for filename in listdir(self.cropped_path)]
		filtered_paths_negatives = []
		filtered_patients_ids = []
		for path_negative, pid in zip(paths_negatives, patients_ids):
			if pid in patient_ids_to_include:
				for path in paths:
					if path_negative == path[:-2]:
						filtered_paths_negatives.append(path)
						filtered_patients_ids.append(os.path.basename(path)[:-2])
						break
		paths_negatives = filtered_paths_negatives
		self.patients_ids = filtered_patients_ids

		# Retrieve all the patches from the directories
		self.paths_patches = []
		self.patient_ids_patches = []
		for i, directory in enumerate(paths_negatives):
			patches_names = listdir(directory, extension=".png")
			patches_paths = [os.path.join(directory, patches_name) for patches_name in patches_names]
			self.paths_patches.extend(patches_paths)
			self.patient_ids_patches.extend([self.patients_ids[i]] * len(patches_paths))

	def get_negative_diagnosis_directories(self, csv_path: str) -> Tuple[List[str], List[str]]:
		data = pd.read_csv(csv_path)

		# Filter rows where the DENSITAT is "NEGATIVA"
		negative_diagnosis = data[data["DENSITAT"] == "NEGATIVA"]

		# Create directory paths and patient IDs based on the CODI values
		directories = [
			os.path.join(self.cropped_path, str(codi))
			for codi in negative_diagnosis["CODI"]
		]
		patient_ids = negative_diagnosis["CODI"].astype(str).tolist()

		return directories, patient_ids

	def __getitem__(self, index) -> Any:
		image = transform_image(Image.open(self.paths_patches[index]).convert("RGB"), (256, 256))
		if self.patient_id:
			return image, self.patient_ids_patches[index]  # (image, patient_id)
		else:
			return image  # (image,)

	def __len__(self) -> int:
		return len(self.paths_patches)



class HelicoDatasetClassification(Dataset):
	def __init__(
			self,
			patient_id: bool=False,
			split: Literal["train", "val"]="train",
			train_ratio: float=0.8,
			random_seed: int=42,
			patient_ids_to_include: Optional[List[str]]=None
	):
		super().__init__()
		# Initialize paths
		path_error = ensure_dataset_path_yaml()
		if path_error == 1:
			print(f"Dataset path not found in config.yml. Defaulting to {default_path}")
			if not os.path.exists(default_path):
				raise FileNotFoundError(f"Default path {default_path} does not exist. Specify a valid path in config.yml.")
		elif path_error == 2:
			current_path = yaml.safe_load(open("config.yml", "r"))["dataset_path"]
			raise FileNotFoundError(f"Dataset path {current_path} does not exist. Specify a valid path in config.yml.")
		
		self.dataset_path = yaml.safe_load(open("config.yml", "r"))["dataset_path"]
		self.annotated_path = os.path.join(self.dataset_path, "CrossValidation", "Annotated")
		self.excel_file_path = os.path.join(self.dataset_path, "HP_WSI-CoordAnnotatedPatches.xlsx")
		self.patient_id = patient_id

		if patient_ids_to_include is None:
			patient_ids_to_include = get_classification_patient_ids(self.excel_file_path)
		self.paths_labels = self.get_paths_and_labels(
			self.annotated_path, self.excel_file_path,
			split, train_ratio, random_seed, patient_ids_to_include
		)
	
	def get_paths_and_labels(self, annotated_path: str, excel_path: str, *args) -> tuple:
		"""
		Given the annotated path and an Excel file path with columns "Path_ID", "Window_ID", and "Presence" (which ranges -1, 0, 1),
		returns a list of tuples (path, label). The label is 0 if the presence is -1, and 1 if the presence is 1.
		Samples with "Presence" 0 are ignored.
		"""
		split, train_ratio, random_seed, patient_ids_to_include = args

		# Load the Excel file
		data = pd.read_excel(excel_path)

		# Filter the rows with "Presence" -1 or 1
		data = data[(data["Presence"] == -1) | (data["Presence"] == 1)]

		# Create a list of tuples (path, label)
		paths_labels = [
			(os.path.join(annotated_path, f"{row['Pat_ID']}",f"{row['Window_ID']}.png"), 0 if row["Presence"] == -1 else 1)
			for _, row in data.iterrows()
		]
		paths = [os.path.join(annotated_path, filename) for filename in listdir(annotated_path)]
		actual_paths = []
		patient_ids = []
		for path_label_tup in paths_labels:
			path_label = os.path.dirname(path_label_tup[0])
			for path in paths:
				patient_id = os.path.basename(path)[:-2]
				if path_label == path[:-2] and patient_id in patient_ids_to_include:
					basename = os.path.splitext(os.path.basename(path_label_tup[0]))[0].zfill(5)
					listed_basenames = listdir(path)
					actual_basenames = [listed_basename for listed_basename in listed_basenames if listed_basename.startswith(basename)]
					label_append = path_label_tup[1]
					for actual_basename in actual_basenames:
						path_append = os.path.join(path, actual_basename)
						actual_paths.append((path_append, label_append, patient_id))
						patient_ids.append(patient_id)
					break

		# Split the data into train and test sets
		unique_patient_ids = sorted(list(set(patient_ids)))
		train_size = int(len(unique_patient_ids) * train_ratio)
		train_indices = np.random.RandomState(random_seed).choice(len(unique_patient_ids), train_size, replace=False)
		test_indices = np.array([i for i in range(len(unique_patient_ids)) if i not in train_indices])
		if split == "train":
			unique_patient_ids = [unique_patient_ids[i] for i in train_indices]
		else:
			unique_patient_ids = [unique_patient_ids[i] for i in test_indices]
		
		# Filter by unique_patient_ids
		filtered_paths_labels = []
		for path_label in actual_paths:
			if path_label[2] in unique_patient_ids:
				if self.patient_id:
					filtered_paths_labels.append(path_label)
				else:
					filtered_paths_labels.append(path_label[:2])

		return filtered_paths_labels
		
	def __getitem__(self, index) -> Any:
		if self.patient_id:
			path, label, patient_id = self.paths_labels[index]
			return transform_image(Image.open(path).convert("RGB"), (256, 256)), label, patient_id
		else:
			path, label = self.paths_labels[index]
			return transform_image(Image.open(path).convert("RGB"), (256, 256)), label
	
	def __len__(self) -> int:
		return len(self.paths_labels)


class HelicoDatasetPatientDiagnosis(Dataset):
	def __init__(
			self,
			split: Literal["train", "val", "test"]="train",
			train_ratio: float=0.8, # only for train-val, test is always the whole Holdout
			random_seed: int=42,
			patient_ids_to_include: Optional[List[str]]=None
	):
		# Initialize paths
		path_error = ensure_dataset_path_yaml()
		if path_error == 1:
			print(f"Dataset path not found in config.yml. Defaulting to {default_path}")
			if not os.path.exists(default_path):
				raise FileNotFoundError(f"Default path {default_path} does not exist. Specify a valid path in config.yml.")
		elif path_error == 2:
			current_path = yaml.safe_load(open("config.yml", "r"))["dataset_path"]
			raise FileNotFoundError(f"Dataset path {current_path} does not exist. Specify a valid path in config.yml.")
		
		self.dataset_path = yaml.safe_load(open("config.yml", "r"))["dataset_path"]
		self.csv_file_path = os.path.join(self.dataset_path, "PatientDiagnosis.csv")
		self.cropped_path = os.path.join(self.dataset_path, "CrossValidation", "Cropped")
		self.holdout_path = os.path.join(self.dataset_path, "HoldOut")
		if split == "test":
			data_path = self.holdout_path
			train_ratio = 0
		else:
			data_path = self.cropped_path

		# Load the CSV file
		csv = pd.read_csv(self.csv_file_path)
		patients_ids = csv["CODI"].to_numpy()
		patients_diagnosis = csv["DENSITAT"].to_numpy()
		# remove BAIXA
		patients_ids = patients_ids[patients_diagnosis != "BAIXA"]
		patients_diagnosis = patients_diagnosis[patients_diagnosis != "BAIXA"]
		# patient_diagnosis to 0 or 1
		patients_diagnosis = [0 if diagnosis == "NEGATIVA" else 1 for diagnosis in patients_diagnosis]
		# Build a mapping from patient_id to diagnosis
		patient_id_to_diagnosis = dict(zip(patients_ids, patients_diagnosis))

		if patient_ids_to_include is None:
			patient_ids_to_include = patients_ids.tolist()

		# Split into train and test sets
		train_size = int(len(patient_ids_to_include) * train_ratio)
		train_indices = np.random.RandomState(random_seed).choice(len(patient_ids_to_include), train_size, replace=False)
		test_indices = np.array([i for i in range(len(patient_ids_to_include)) if i not in train_indices])
		if split == "train":
			patient_ids_to_include = [patient_ids_to_include[i] for i in train_indices]
		else:
			patient_ids_to_include = [patient_ids_to_include[i] for i in test_indices]

		# Initialize empty lists to store valid patient data
		valid_patient_ids = []
		valid_patient_diagnosis = []
		patient_paths = []

		# Fetch all the patient directories and ensure alignment
		for patient_directory in listdir(data_path):
			for patient_id in patient_ids_to_include:
				if patient_directory.startswith(patient_id):
					patient_paths.append(os.path.join(data_path, patient_directory))
					valid_patient_ids.append(patient_id)
					valid_patient_diagnosis.append(patient_id_to_diagnosis[patient_id])
					break

		# Now, use valid_patient_ids and valid_patient_diagnosis for indexing
		self.triplets = []  # (patch_path, patient_id, patient_diagnosis)
		for i, patient_path in enumerate(patient_paths):
			patches = listdir(patient_path, extension=".png")
			for patch in patches:
				self.triplets.append((os.path.join(patient_path, patch), valid_patient_ids[i], valid_patient_diagnosis[i]))

	def __getitem__(self, index) -> Any:
		path, patient_id, patient_diagnosis = self.triplets[index]
		# returns (image, patient_id, patient_diagnosis)
		return transform_image(Image.open(path).convert("RGB"), (256, 256)), patient_id, patient_diagnosis
	
	def __len__(self) -> int:
		return len(self.triplets)


if __name__ == "__main__":
	# dataset = HelicoDatasetAnomalyDetection()
	dataset = HelicoDatasetPatientDiagnosis(split="test", train_ratio=0.8, random_seed=42)
	print(len(dataset))
	print(dataset[0])
	# print(dataset[0][0].shape, dataset[0][1], dataset[0][2])
	# ensure all iterations work
	# for i in range(len(dataset)):
	# 	dataset[i]
