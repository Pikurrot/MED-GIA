from typing import Any
from torch.utils.data import Dataset
from PIL import Image
import os
import pandas as pd
import yaml
from torchvision import transforms

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

class HelicoDatasetAnomalyDetection(Dataset):
	def __init__(self, patient_id: bool=False) -> None:
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
		paths_negatives = self.get_negative_diagnosis_directories(self.csv_file_path)
		paths = [os.path.join(self.cropped_path, filename) for filename in listdir(self.cropped_path)]
		actual_paths = []
		patients_ids = []
		for path_negative in paths_negatives:
			for path in paths:
				if path_negative == path[:-2]:
					actual_paths.append(path)
					if patient_id:
						patients_ids.append(os.path.basename(path)[:-2])
					break

		# Retrieve all the patches from the directories
		self.paths_patches = []
		self.patient_ids_patches = []
		for i, directory in enumerate(actual_paths):
			patches_names = listdir(directory, extension=".png")
			patches_paths = [os.path.join(directory, patches_name) for patches_name in patches_names]
			self.paths_patches.extend(patches_paths)
			self.patient_ids_patches.extend([patients_ids[i]] * len(patches_paths))

	def get_negative_diagnosis_directories(self, csv_path: str) -> list:
		"""
		Given a CSV file path, returns a list of directories for the NEGATIVE Diagnosis.
		Each directory follows the format "/fhome/vlia/helicoDataSet/CrossValidation/Cropped/patientCODI",
		where "patientCODI" is based on the CODI column in the CSV.

		:param csv_path: Path to the CSV file
		:return: List of directory paths as strings
		"""
		data = pd.read_csv(csv_path)

		# Filter rows where the DENSITAT is "NEGATIVA"
		negative_diagnosis = data[data["DENSITAT"] == "NEGATIVA"]

		# Create directory paths based on the CODI values
		directories = [
			os.path.join(self.cropped_path, codi)
			for codi in negative_diagnosis["CODI"]
		]

		return directories

	def __getitem__(self, index) -> Any:
		image = transform_image(Image.open(self.paths_patches[index]).convert("RGB"), (256, 256))
		if self.patient_id:
			return image, self.patient_ids_patches[index] # (image, patient_id)
		else:
			return image # (image,)

	def __len__(self) -> int:
		return len(self.paths_patches)


class HelicoDatasetClassification(Dataset):
	def __init__(self, patient_id: bool=False) -> None:
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

		self.paths_labels = self.get_paths_and_labels(self.annotated_path, self.excel_file_path)
	
	def get_paths_and_labels(self, annotated_path: str, excel_path: str) -> tuple:
		"""
		Given the annotated path and an Excel file path with columns "Path_ID", "Window_ID", and "Presence" (which ranges -1, 0, 1),
		returns a list of tuples (path, label). The label is 0 if the presence is -1, and 1 if the presence is 1.
		Samples with "Presence" 0 are ignored.
		"""
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
		for path_label_tup in paths_labels:
			path_label = os.path.dirname(path_label_tup[0])
			for path in paths:
				if path_label == path[:-2]:
					basename = os.path.splitext(os.path.basename(path_label_tup[0]))[0].zfill(5)
					listed_basenames = listdir(path)
					actual_basenames = [listed_basename for listed_basename in listed_basenames if listed_basename.startswith(basename)]
					label_append = path_label_tup[1]
					for actual_basename in actual_basenames:
						path_append = os.path.join(path, actual_basename)
						if self.patient_id:
							actual_paths.append((path_append, label_append, os.path.basename(path)[:-2]))
						else:
							actual_paths.append((path_append, label_append))
					break

		return actual_paths
		
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
	def __init__(self) -> None:
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
		self.holdout_path = os.path.join(self.dataset_path, "HoldOut")

		# Load the CSV file
		csv = pd.read_csv(self.csv_file_path)
		patients_ids = csv["CODI"].to_numpy()
		patients_diagnosis = csv["DENSITAT"].to_numpy()
		# remove BAIXA
		patients_ids = patients_ids[patients_diagnosis != "BAIXA"]
		patients_diagnosis = patients_diagnosis[patients_diagnosis != "BAIXA"]
		# patient_diagnosis to 0 or 1
		patients_diagnosis = [0 if diagnosis == "NEGATIVA" else 1 for diagnosis in patients_diagnosis]

		# fetch all the patient directories
		patient_directories = []
		for patient_id in patients_ids:
			for patient_directory in listdir(self.holdout_path):
				if patient_directory.startswith(patient_id):
					patient_directories.append(patient_directory)
					break
		patient_paths = [os.path.join(self.holdout_path, patient_directory) for patient_directory in patient_directories]
		
		# fetch all the patches from the patient directories
		self.triplets = [] # (patch_path, patient_id, patient_diagnosis)
		for i, patient_path in enumerate(patient_paths):
			patches = listdir(patient_path, extension=".png")
			for patch in patches:
				self.triplets.append((os.path.join(patient_path, patch), patients_ids[i], patients_diagnosis[i]))

	def __getitem__(self, index) -> Any:
		path, patient_id, patient_diagnosis = self.triplets[index]
		# returns (image, patient_id, patient_diagnosis)
		return transform_image(Image.open(path).convert("RGB"), (256, 256)), patient_id, patient_diagnosis
	
	def __len__(self) -> int:
		return len(self.triplets)


if __name__ == "__main__":
	# dataset = HelicoDatasetAnomalyDetection()
	dataset = HelicoDatasetPatientDiagnosis()
	print(len(dataset))
	print(dataset[0][0].shape, dataset[0][1], dataset[0][2])
	# ensure all iterations work
	for i in range(len(dataset)):
		dataset[i]
