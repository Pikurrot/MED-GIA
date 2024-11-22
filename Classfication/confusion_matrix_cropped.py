from PIL import Image
import pathlib
import pandas as pd
import torchvision
import numpy as np
from model import HelicobacterClassifier   
import tqdm
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path_to_models = r"/export/fhome/vlia01/MED-GIA/Classfication/best_model_fold"
for n in range(5):
    model = HelicobacterClassifier()
    model.load_state_dict(torch.load(path_to_models + str(n) + ".pth"))
    globals()[f'model_fold_{n}'] = model

# Ensemble voting to predict new patches
def ensemble_predict(models, img):
    outputs = [torch.softmax(model(img), dim=1) for model in models]  # Get probabilities [0.1,0.9]
    max_confidences = [torch.max(output).item() for output in outputs]  # Get max confidence for each model
    best_model_index = max_confidences.index(max(max_confidences))  # Find the index of the model with the highest confidence
    _, predicted = torch.max(outputs[best_model_index].data, 1)  # Get the prediction of the best model
    return predicted

# Load models into a list
models = [globals()[f'model_fold_{n}'] for n in range(5)]

def transform_image(image, size):
    return image.resize(size)

class Cropped_Dataset():
    def __init__(self):
        path_to_cropped = r"/export/fhome/vlia/HelicoDataSet/CrossValidation/Cropped"
        patient_directories = [patient for patient in pathlib.Path(path_to_cropped).iterdir()]
        path_to_Patient_Diagnois = r'/export/fhome/vlia/HelicoDataSet/PatientDiagnosis.csv'
        patient_diagnosisDF = pd.read_csv(path_to_Patient_Diagnois)
        patient_diagnosisDF = patient_diagnosisDF[(patient_diagnosisDF['DENSITAT'] == 'ALTA') | (patient_diagnosisDF['DENSITAT'] == 'NEGATIVA')]
        patient_diagnosisDF['DENSITAT'] = [1 if x == 'ALTA' else -1 for x in patient_diagnosisDF['DENSITAT']]
        patient_diagnosisDF = patient_diagnosisDF.rename(columns={'DENSITAT': 'DiagnosisGT'})
        
        self.dictionary = {}
        
        for patient in patient_directories:
            if patient.name[:-2] in patient_diagnosisDF['CODI'].values:
                images = [x for x in patient.iterdir() if x.is_file() and x.name.endswith('.png')]
                self.dictionary[patient.name[:-2]] = (images, patient_diagnosisDF[patient_diagnosisDF['CODI'] == patient.name[:-2]]['DiagnosisGT'].values)
        
    def __len__(self):
        return len(self.dictionary)
    
    def __getitem__(self, idx):
        patient_id = list(self.dictionary.keys())[idx]
        images, patient_diagnosis = self.dictionary[patient_id]
        transformed_images = [transform_image(Image.open(image_path).convert("RGB"), (256, 256)) for image_path in images]
        return transformed_images, patient_id, patient_diagnosis

cropped_dataset = Cropped_Dataset() 

# Create a DataLoader for the holdout dataset
cropped_loader = DataLoader(cropped_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x)

# Predict diagnosis for each patient using ensemble
predictions = []
ground_truth = []

for data in tqdm.tqdm(cropped_loader, desc="Processing patients"):
    data = data[0]  # Remove batch dimension
    images, patient_id, patient_diagnosis = data
    images = torch.stack([torchvision.transforms.ToTensor()(image) for image in images])
    images = images.to(device)
    
    # Predict for each image
    positive_patch_found = False
    for img in images:
        img = img.unsqueeze(0)  # Add batch dimension
        predicted = ensemble_predict(models, img)
        if predicted.item() == 1:  # If any patch is positive
            positive_patch_found = True
            break
    
    # If a positive patch is found, the diagnosis is positive (1), otherwise negative (-1)
    final_prediction = 1 if positive_patch_found else -1
    predictions.append(final_prediction)
    ground_truth.append(patient_diagnosis[0].item())

# Compute confusion matrix
conf_matrix = confusion_matrix(ground_truth, predictions)
print("Confusion Matrix:")
print(conf_matrix)

# Plot confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.savefig('confusion_matrix.png')  # Save the plot as an image file
