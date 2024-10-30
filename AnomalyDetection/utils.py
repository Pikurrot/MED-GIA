from typing import Any
from torch.utils.data import Dataset
from PIL import Image
import os
import pandas as pd
import yaml
from torchvision import transforms

default_path = "/fhome/vlia/HelicoDataSet"
confg_path = "../config.yml"

def ensure_dataset_path_yaml() -> int:
	"""
	Ensures that config.yml contains the dataset path.
	:return: 0 if path is valid, 1 if path is not present, 2 if path is invalid
	"""
	# Check config.yml exists
	if not os.path.exists(confg_path):
		with open(confg_path, "w") as file:
			file.write(f"dataset_path: {default_path}")
		return 1
	# Check dataset_path exists
	with open(confg_path, "r") as file:
		config = yaml.safe_load(file)
	if "dataset_path" not in config:
		with open(confg_path, "a") as file:
			file.write(f"dataset_path: {default_path}")
		return 1
	# Check if its valid
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

class HelicoDatasetAnomalyDetection(Dataset):
	def __init__(self) -> None:
		super().__init__()
		# Initialize paths
		"""
		path_error = ensure_dataset_path_yaml()
		if path_error == 1:
			print(f"Dataset path not found in config.yml. Defaulting to {default_path}")
			if not os.path.exists(default_path):
				raise FileNotFoundError(f"Default path {default_path} does not exist. Specify a valid path in config.yml.")
		elif path_error == 2:
			raise FileNotFoundError("Dataset path in config.yml is invalid.")"""
		
		#self.dataset_path = yaml.safe_load(open("config.yml", "r"))["dataset_path"]
		self.dataset_path = default_path
		self.csv_file_path = os.path.join(self.dataset_path, "PatientDiagnosis.csv")
		self.cropped_path = os.path.join(self.dataset_path, "CrossValidation", "Cropped")
		self.excel_file_path = os.path.join(self.dataset_path, "HP_WSI-CoordCroppedPatches.xlsx")

		# Find all the negative diagnosis directories
		paths_negatives = self.get_negative_diagnosis_directories(self.csv_file_path)
		paths = [os.path.join(self.cropped_path, filename) for filename in listdir(self.cropped_path)]
		actual_paths = []
		for path_negative in paths_negatives:
			for path in paths:
				if path_negative == path[:-2]:
					actual_paths.append(path)
					break

		# Retrieve all the patches from the directories
		self.paths_patches = []
		for directory in actual_paths:
			patches_names = listdir(directory, extension=".png")
			patches_paths = [os.path.join(directory, patches_name) for patches_name in patches_names]
			self.paths_patches.extend(patches_paths)

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
		return transforms.ToTensor()(Image.open(self.paths_patches[index]))

	def __len__(self) -> int:
		return len(self.paths_patches)
